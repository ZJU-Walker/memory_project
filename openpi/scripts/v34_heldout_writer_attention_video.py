"""Render every-frame run5 writer-to-top-camera attention for the four heldouts.

The diagnostic reconstructs the model and preprocessing from the immutable run5 launch source
through :class:`v34_fixed_writer_probe_eval._Run5Runtime`.  Each episode starts from one fresh
M0.  Every raw frame runs the real ``v32_query_attention_step`` against the currently threaded
memory state, while only frames ``0, 15, 30, ...`` commit the returned candidate write tokens.

The main H.264 MP4 for each episode runs at the dataset's native 30 fps and shows the exact
224x224 top-camera model input beside the writer attention averaged over all eight heads and
16 write slots.  One robust scale is fitted once across every selected episode/frame and then
held fixed for every video.  The full head-mean per-slot maps are saved as FP16
``[frames, 16, 256]`` arrays.  An optional 4x4 per-slot video contains scheduled write frames
only and uses the established readable v3.2 relative rendering at 4 fps.

The script must run with the extracted run5 source snapshot first on ``PYTHONPATH``::

    V34_RUN5_SOURCE_ROOT=/tmp/v34-run5-source \
    PYTHONPATH=/tmp/v34-run5-source/src \
    .venv/bin/python -u scripts/v34_heldout_writer_attention_video.py \
      --checkpoint diagnostic_checkpoints/v34_run5_eta0_resume_copies/11000 \
      --dataset-root /iris/u/kewalk/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask \
      --output-dir diagnostic_outputs/v34_heldout_writer_attention/11000 \
      --config pi05_yam_mem_v34_run5_eta0 --parameter-source raw

``--smoke-only`` processes exactly frames 0..16 of heldout episode 15, exercising both commit
frames 0/15 and candidate-only frames.  Full mode processes all frames of episodes
15/29/44/59.  Artifacts are built in a private sibling directory, verified, marked COMPLETE,
and only then atomically published under the ``raw`` or ``ema`` child of ``--output-dir``.
"""

# ruff: noqa: SLF001, I001 - audited script-private helpers and pyarrow-before-JAX order are intentional.
from __future__ import annotations

import pyarrow.parquet as pq

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
from typing import Any, Literal
import uuid

# Import this helper before OpenPI/JAX. It enforces the immutable run5 source and provides the
# generic numeric-checkpoint/raw-vs-EMA loader used by the fixed fresh-probe evaluation.
import v34_fixed_writer_probe_eval as fixed

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from openpi.diagnostics import token_heatmap
from openpi.diagnostics import v33_writer_attention as writer_attention
from openpi.shared import nnx_utils


SCHEMA_VERSION = "openpi.v34.heldout_writer_attention_video.v1"
RUN5_CONFIG = fixed.RUN5_CONFIG
HELDOUT_EPISODES = fixed.HELDOUT_EPISODES
EXPECTED_CELLS = fixed.EXPECTED_HELDOUT_CELLS
EXPECTED_FRAME_COUNTS = {15: 876, 29: 876, 44: 863, 59: 740}
EXPECTED_COMMIT_COUNTS = {15: 59, 29: 59, 44: 58, 59: 50}
FULL_TOTAL_FRAME_COUNT = sum(EXPECTED_FRAME_COUNTS.values())

WRITE_STRIDE = 15
SOURCE_FPS = 30.0
SMOKE_EPISODE = 15
SMOKE_FRAME_COUNT = WRITE_STRIDE + 2
NUM_HEADS = 8
NUM_SLOTS = 16
NUM_PATCHES = token_heatmap.TOKEN_COUNT
GRID_SIZE = token_heatmap.TOKEN_GRID_SIZE
VALID_PATCH_COUNT = 12 * GRID_SIZE
GLOBAL_SCALE_PERCENTILE = 99.0
RETRIEVAL_ZERO_TOL = 1e-7
ATTENTION_SUM_ATOL = 2e-5
SAVED_ATTENTION_SUM_ATOL = 3e-3
INVALID_ATTENTION_ATOL = 1e-8
SMOKE_INVARIANCE_ATOL = 1e-6
SMOKE_MAX_EXTRAPOLATED_QSTEP_SECONDS = 2.5 * 60.0 * 60.0
MAIN_DISPLAY_SCALE = 2
MAIN_PANEL_SIZE = token_heatmap.MODEL_IMAGE_SIZE * MAIN_DISPLAY_SCALE
MAIN_WIDTH = 2 * MAIN_PANEL_SIZE
MAIN_HEADER_HEIGHT = 96
MAIN_HEIGHT = MAIN_HEADER_HEIGHT + MAIN_PANEL_SIZE
SLOT_GRID_FPS = 4.0
QSTEP_EXECUTABLE_DESCRIPTION = "unmodified nnx_utils.module_jit(model.v32_query_attention_step)"


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint: Path
    dataset_root: Path
    output_dir: Path
    config: str
    parameter_source: Literal["raw", "ema"]
    smoke_only: bool = False
    write_frame_slot_grid: bool = True

    def __post_init__(self) -> None:
        for name in ("checkpoint", "dataset_root", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        if not self.checkpoint.name.isdigit() or int(self.checkpoint.name) < 0:
            raise ValueError(f"checkpoint must end in a nonnegative numeric step, got {self.checkpoint.name!r}")
        if self.config != RUN5_CONFIG:
            raise ValueError(f"writer-attention video is pinned to --config {RUN5_CONFIG!r}")
        if self.parameter_source not in ("raw", "ema"):
            raise ValueError("--parameter-source must be raw or ema")
        if fixed.causal._is_relative_to(self.output_dir, self.checkpoint):
            raise ValueError("diagnostic output must be outside the checkpoint directory")

    @property
    def checkpoint_step(self) -> int:
        return int(self.checkpoint.name)

    @property
    def episodes(self) -> tuple[int, ...]:
        return (SMOKE_EPISODE,) if self.smoke_only else tuple(HELDOUT_EPISODES)

    @property
    def artifact_dir(self) -> Path:
        return self.output_dir / self.parameter_source


@dataclasses.dataclass(frozen=True)
class _RuntimeArgs:
    checkpoint: Path
    dataset_root: Path
    config: str
    parameter_source: Literal["raw", "ema"]
    artifact_dir: Path


@dataclasses.dataclass
class _EpisodeResult:
    episode: int
    prompt: str
    side: str
    fps: float
    frames: np.ndarray
    task_indices: np.ndarray
    scheduled_commits: np.ndarray
    slot_maps: np.ndarray
    mean_maps: np.ndarray
    top_images: list[np.ndarray]
    retrieval_rms: np.ndarray
    write_token_rms: np.ndarray
    qstep_seconds: np.ndarray
    commit_state_deltas: np.ndarray
    commit_grad_norms: np.ndarray
    commit_clip_factors: np.ndarray
    final_state_max_abs: float
    source_identity: dict[str, Any]
    task_names: tuple[str, ...]
    label_runs: list[dict[str, Any]]
    smoke_fresh_m0_gate: dict[str, Any] | None
    smoke_timing_gate: dict[str, Any] | None
    inference_elapsed_seconds: float


def _parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parameter-source", choices=("raw", "ema"), required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--write-frame-slot-grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also render the 4x4 per-slot video for scheduled write frames only",
    )
    return Args(**vars(parser.parse_args(argv)))


def _is_scheduled_commit(frame: int) -> bool:
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError(f"frame must be a nonnegative integer, got {frame!r}")
    return frame % WRITE_STRIDE == 0


def _expected_commit_count(frame_count: int) -> int:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    return (frame_count - 1) // WRITE_STRIDE + 1


def _qstep_call_accounting(frame_count: int, *, smoke_only: bool) -> dict[str, int]:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError("frame_count must be a positive integer")
    control_calls = 1 if smoke_only else 0
    return {
        "recurrence_read_attention_calls": frame_count,
        "recurrence_candidate_writer_calls": frame_count,
        "smoke_fresh_M0_control_qstep_calls": control_calls,
        "total_qstep_calls_including_smoke_controls": frame_count + control_calls,
    }


def _smoke_timing_gate(qstep_seconds: Sequence[float]) -> dict[str, Any]:
    timings = np.asarray(qstep_seconds, dtype=np.float64)
    if (
        timings.shape != (SMOKE_FRAME_COUNT,)
        or not np.all(np.isfinite(timings))
        or np.any(timings <= 0)
    ):
        raise RuntimeError(f"invalid smoke qstep timings: {timings}")
    steady = timings[1:]
    steady_mean = float(np.mean(steady))
    steady_p95 = float(np.percentile(steady, 95.0))
    extrapolated_mean = steady_mean * FULL_TOTAL_FRAME_COUNT
    extrapolated_p95 = steady_p95 * FULL_TOTAL_FRAME_COUNT
    return {
        "status": "pass" if extrapolated_mean <= SMOKE_MAX_EXTRAPOLATED_QSTEP_SECONDS else "fail",
        "science_interpretation": "none; operational runtime gate only",
        "coverage": "qstep-only; excludes transforms, commits, rendering, encoding, and validation",
        "compile_excluded_frame": 0,
        "steady_timed_frames": list(range(1, SMOKE_FRAME_COUNT)),
        "steady_unmodified_qstep_seconds_mean": steady_mean,
        "steady_unmodified_qstep_seconds_p95": steady_p95,
        "full_frame_count": FULL_TOTAL_FRAME_COUNT,
        "extrapolated_full_qstep_seconds_from_mean": extrapolated_mean,
        "extrapolated_full_qstep_seconds_from_p95": extrapolated_p95,
        "operational_max_extrapolated_qstep_seconds": SMOKE_MAX_EXTRAPOLATED_QSTEP_SECONDS,
        "launcher_slurm_step_time_limit_seconds": 3.0 * 60.0 * 60.0,
    }


def _expected_source_valid() -> np.ndarray:
    expected = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    expected[2:14, :] = True
    return expected.reshape(NUM_PATCHES)


def _validate_source_valid(value: Any) -> np.ndarray:
    valid = np.asarray(value, dtype=bool)
    if valid.shape != (NUM_PATCHES,):
        raise ValueError(f"top-camera source-valid mask must have shape ({NUM_PATCHES},), got {valid.shape}")
    expected = _expected_source_valid()
    if not np.array_equal(valid, expected) or int(valid.sum()) != VALID_PATCH_COUNT:
        raise ValueError("run5 top-camera letterbox-valid mask changed from rows 2..13 of the 16x16 grid")
    return np.array(valid, copy=True)


def _head_mean_per_slot(attention: Any, source_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate exact model attention and return head-mean slots plus the slot/head mean map."""
    values = np.asarray(attention, dtype=np.float32)
    expected_shape = (1, NUM_HEADS, NUM_SLOTS, NUM_PATCHES)
    if values.shape != expected_shape:
        raise ValueError(f"write attention must have shape {expected_shape}, got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise FloatingPointError("write attention contains NaN/Inf or negative probability")
    row_sums = values.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=ATTENTION_SUM_ATOL):
        raise RuntimeError(
            f"write attention rows no longer sum to one: min={row_sums.min():.9g}, max={row_sums.max():.9g}"
        )
    invalid_max = float(np.max(values[..., ~source_valid]))
    if invalid_max > INVALID_ATTENTION_ATOL:
        raise RuntimeError(f"letterbox-invalid patches received attention mass {invalid_max:.9g}")
    slots = values[0].mean(axis=0, dtype=np.float32)
    slot_sums = slots.sum(axis=-1)
    if not np.allclose(slot_sums, 1.0, rtol=0.0, atol=ATTENTION_SUM_ATOL):
        raise RuntimeError("head-mean per-slot attention rows no longer sum to one")
    mean_map = slots.mean(axis=0, dtype=np.float32)
    if not math.isclose(float(mean_map.sum()), 1.0, rel_tol=0.0, abs_tol=ATTENTION_SUM_ATOL):
        raise RuntimeError("head/slot-mean attention map no longer sums to one")
    return slots, mean_map


def _fit_global_scale(results: Sequence[_EpisodeResult], source_valid: np.ndarray) -> token_heatmap.ColorScale:
    if not results:
        raise ValueError("cannot fit a global attention scale without episode results")
    values = []
    for result in results:
        maps = np.asarray(result.mean_maps, dtype=np.float64)
        if maps.ndim != 2 or maps.shape[1] != NUM_PATCHES or not np.all(np.isfinite(maps)):
            raise ValueError(f"episode {result.episode} has invalid mean maps {maps.shape}")
        values.append(maps[:, source_valid].reshape(-1))
    joined = np.concatenate(values)
    vmax = float(np.percentile(joined, GLOBAL_SCALE_PERCENTILE))
    if not math.isfinite(vmax) or vmax <= 0.0:
        raise ValueError(f"global attention scale has invalid vmax {vmax!r}")
    return token_heatmap.ColorScale(
        vmin=0.0,
        vmax=vmax,
        lower_percentile=0.0,
        upper_percentile=GLOBAL_SCALE_PERCENTILE,
        anchor_zero=True,
        excluded_letterbox_padding=True,
    )


def _state_max_abs(state: Any) -> float:
    leaves = jax.tree.leaves(state)
    if not leaves:
        raise ValueError("memory state has no leaves")
    finite = jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves])
    if not bool(np.asarray(jnp.all(finite))):
        raise FloatingPointError("memory state contains NaN or infinity")
    maxima = jnp.stack([jnp.max(jnp.abs(leaf.astype(jnp.float32))) for leaf in leaves])
    result = float(np.asarray(jnp.max(maxima)))
    if not math.isfinite(result):
        raise FloatingPointError("memory state maximum is not finite")
    return result


def _safe_text(value: str) -> str:
    return value.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def _render_main_frame(
    top_image: np.ndarray,
    mean_map: np.ndarray,
    scale: token_heatmap.ColorScale,
    *,
    episode: int,
    frame: int,
    frame_count: int,
    fps: float,
    prompt: str,
    side: str,
    gt: str,
    checkpoint_step: int,
    parameter_source: str,
    scheduled_commit: bool,
) -> np.ndarray:
    image = np.asarray(top_image)
    if image.shape != (token_heatmap.MODEL_IMAGE_SIZE, token_heatmap.MODEL_IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError(f"top image must be exact uint8 224x224 RGB, got {image.shape}/{image.dtype}")
    overlay, _normalized = token_heatmap.heatmap_overlay(
        image,
        mean_map,
        scale,
        alpha=0.62,
        colormap="inferno",
        draw_grid=True,
        scale_mode="video",
    )
    left = cv2.resize(image, (MAIN_PANEL_SIZE, MAIN_PANEL_SIZE), interpolation=cv2.INTER_NEAREST)
    right = cv2.resize(overlay, (MAIN_PANEL_SIZE, MAIN_PANEL_SIZE), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((MAIN_HEIGHT, MAIN_WIDTH, 3), dtype=np.uint8)
    canvas[MAIN_HEADER_HEIGHT:, :MAIN_PANEL_SIZE] = left
    canvas[MAIN_HEADER_HEIGHT:, MAIN_PANEL_SIZE:] = right

    status = "COMMIT WRITE" if scheduled_commit else "CANDIDATE ONLY - NOT COMMITTED"
    status_color = (80, 245, 120) if scheduled_commit else (255, 205, 80)
    lines = (
        (
            f"ep{episode:03d}  frame {frame:04d}/{frame_count - 1:04d}  t={frame / fps:6.2f}s  "
            f"checkpoint={checkpoint_step}  params={parameter_source.upper()}  {status}",
            status_color,
        ),
        (f"prompt={_safe_text(prompt)}  |  heldout side={side}  |  GT(t)={_safe_text(gt)}", (245, 245, 245)),
        (
            "LEFT: exact model top input (2x nearest)  |  RIGHT: mean writer attention "
            f"(8 heads x 16 slots), per-patch global scale 0.00%..{100.0 * scale.vmax:.2f}% (higher clipped)",
            (190, 220, 255),
        ),
        ("Every frame: writer candidate + read.  Memory write only when frame % 15 == 0.", (185, 185, 185)),
    )
    for index, (text, color) in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (8, 19 + 23 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    bar_left, bar_top, bar_width, bar_height = 646, 77, 170, 12
    bar_gray = np.broadcast_to(np.linspace(0, 255, bar_width, dtype=np.uint8), (bar_height, bar_width))
    bar_rgb = cv2.cvtColor(cv2.applyColorMap(bar_gray, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    canvas[bar_top : bar_top + bar_height, bar_left : bar_left + bar_width] = bar_rgb
    cv2.rectangle(
        canvas,
        (bar_left, bar_top),
        (bar_left + bar_width - 1, bar_top + bar_height - 1),
        (235, 235, 235),
        1,
    )
    cv2.putText(canvas, "0%", (620, 87), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"{100.0 * scale.vmax:.2f}%",
        (820, 87),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return canvas


class _AtomicMp4Writer:
    def __init__(self, path: Path, *, fps: float, width: int, height: int):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite video: {path}")
        if not math.isfinite(fps) or fps <= 0.0 or width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("H.264 fps and even frame dimensions must be positive")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError("ffmpeg is required for H.264 output")
        self.path = path
        self.temporary = path.with_name(f".{path.name}.tmp")
        if self.temporary.exists():
            raise FileExistsError(f"stale temporary video exists: {self.temporary}")
        self.fps = float(fps)
        self.width = int(width)
        self.height = int(height)
        self.frames = 0
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps:g}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                str(self.temporary),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if self.process.stdin is None:
            raise RuntimeError("failed to open ffmpeg stdin")

    def write(self, frame: np.ndarray) -> None:
        image = np.asarray(frame)
        if image.shape != (self.height, self.width, 3) or image.dtype != np.uint8:
            raise ValueError(f"invalid MP4 frame {image.shape}/{image.dtype}")
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(np.ascontiguousarray(image).tobytes())
        except BrokenPipeError as exc:
            stderr = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
            raise RuntimeError(f"ffmpeg pipe failed: {stderr}") from exc
        self.frames += 1

    def close(self) -> dict[str, Any]:
        if self.frames == 0:
            self.abort()
            raise ValueError("refusing to finalize an empty MP4")
        assert self.process.stdin is not None
        self.process.stdin.close()
        self.process.stdin = None
        _stdout, stderr = self.process.communicate()
        if self.process.returncode != 0:
            raise RuntimeError(f"ffmpeg failed ({self.process.returncode}): {stderr.decode(errors='replace')}")
        if not self.temporary.is_file() or self.temporary.stat().st_size <= 0:
            raise RuntimeError("ffmpeg produced no MP4")
        with self.temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(self.temporary, self.path)
        return fixed.causal._file_identity(self.path)

    def abort(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin is not None:
                self.process.stdin.close()
                self.process.stdin = None
            self.process.terminate()
            self.process.wait(timeout=10)
        self.temporary.unlink(missing_ok=True)


def _rate_to_float(value: str) -> float:
    numerator, separator, denominator = value.partition("/")
    if not separator:
        return float(value)
    return float(numerator) / float(denominator)


def _probe_mp4(path: Path, *, fps: float, width: int, height: int, frames: int) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise FileNotFoundError("ffprobe is required for MP4 verification")
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise RuntimeError(f"ffprobe found an invalid video stream set for {path}")
    stream = streams[0]
    actual = {
        "codec_name": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "width": int(stream.get("width", -1)),
        "height": int(stream.get("height", -1)),
        "fps": _rate_to_float(str(stream.get("r_frame_rate"))),
        "frames": int(stream.get("nb_read_frames", -1)),
    }
    expected = {"codec_name": "h264", "pixel_format": "yuv420p", "width": width, "height": height, "frames": frames}
    for key, value in expected.items():
        if actual[key] != value:
            raise RuntimeError(f"MP4 {path.name} {key} mismatch: expected {value!r}, got {actual[key]!r}")
    if not math.isclose(actual["fps"], fps, rel_tol=0.0, abs_tol=1e-6):
        raise RuntimeError(f"MP4 {path.name} fps mismatch: expected {fps}, got {actual['fps']}")
    format_payload = payload.get("format")
    if not isinstance(format_payload, dict):
        raise RuntimeError(f"ffprobe returned no format duration for {path}")
    actual_duration = float(format_payload.get("duration", "nan"))
    expected_duration = frames / fps
    duration_tolerance = max(1e-6, 0.5 / fps)
    if not math.isfinite(actual_duration) or not math.isclose(
        actual_duration, expected_duration, rel_tol=0.0, abs_tol=duration_tolerance
    ):
        raise RuntimeError(
            f"MP4 {path.name} duration mismatch: expected {expected_duration}, got {actual_duration}"
        )
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required for full MP4 decode verification")
    decode = subprocess.run(
        [ffmpeg, "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    if decode.returncode != 0:
        raise RuntimeError(f"full ffmpeg decode failed for {path}: {decode.stderr.strip()}")
    actual["duration_seconds"] = actual_duration
    actual["duration_tolerance_seconds"] = duration_tolerance
    actual["full_ffmpeg_decode"] = "pass"
    return actual


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite NPZ: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"stale NPZ temporary exists: {temporary}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return fixed.causal._file_identity(path)


def _validate_saved_npz(path: Path, *, frame_count: int, source_valid: np.ndarray) -> dict[str, Any]:
    expected_keys = {
        "attention_head_mean_slots",
        "attention_mean_heads_slots",
        "frame_index",
        "retrieval_rms",
        "scheduled_commit",
        "source_valid",
        "task_index",
        "write_token_rms",
    }
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != expected_keys:
            raise RuntimeError(f"NPZ schema mismatch: expected {sorted(expected_keys)}, got {sorted(data.files)}")
        slots = np.asarray(data["attention_head_mean_slots"])
        means = np.asarray(data["attention_mean_heads_slots"])
        frames = np.asarray(data["frame_index"])
        tasks = np.asarray(data["task_index"])
        commits = np.asarray(data["scheduled_commit"])
        valid = np.asarray(data["source_valid"])
        if slots.shape != (frame_count, NUM_SLOTS, NUM_PATCHES) or slots.dtype != np.float16:
            raise RuntimeError(f"saved slot-map contract failed: {slots.shape}/{slots.dtype}")
        if means.shape != (frame_count, NUM_PATCHES) or means.dtype != np.float16:
            raise RuntimeError(f"saved mean-map contract failed: {means.shape}/{means.dtype}")
        if frames.shape != (frame_count,) or not np.array_equal(frames, np.arange(frame_count, dtype=np.int32)):
            raise RuntimeError("saved frame indices are not contiguous from zero")
        if tasks.shape != (frame_count,) or tasks.dtype != np.int16:
            raise RuntimeError("saved task indices have the wrong shape or dtype")
        expected_commits = np.asarray([_is_scheduled_commit(int(frame)) for frame in frames], dtype=bool)
        if commits.shape != (frame_count,) or commits.dtype != np.bool_ or not np.array_equal(commits, expected_commits):
            raise RuntimeError("saved write schedule differs from frame_index % 15 == 0")
        if valid.dtype != np.bool_ or not np.array_equal(valid, source_valid):
            raise RuntimeError("saved source-valid mask differs from runtime mask")
        if not np.all(np.isfinite(slots)) or np.any(slots < 0):
            raise RuntimeError("saved slot maps contain invalid values")
        if not np.allclose(slots.astype(np.float32).sum(axis=-1), 1.0, rtol=0.0, atol=SAVED_ATTENTION_SUM_ATOL):
            raise RuntimeError("saved FP16 slot maps no longer sum approximately to one")
        if float(np.max(slots[..., ~source_valid])) > INVALID_ATTENTION_ATOL:
            raise RuntimeError("saved slot maps assign mass to letterbox padding")
        if not np.allclose(means.astype(np.float32), slots.astype(np.float32).mean(axis=1), rtol=0.0, atol=8e-4):
            raise RuntimeError("saved aggregate maps differ from the mean of saved slots")
        for key in ("retrieval_rms", "write_token_rms"):
            values = np.asarray(data[key])
            if values.shape != (frame_count,) or values.dtype != np.float32 or not np.all(np.isfinite(values)):
                raise RuntimeError(f"saved {key} contract failed")
    return {
        "schema_exact": True,
        "frame_count": frame_count,
        "slot_shape": [frame_count, NUM_SLOTS, NUM_PATCHES],
        "slot_dtype": "float16",
        "all_finite_nonnegative": True,
        "rows_sum_to_one_with_fp16_tolerance": SAVED_ATTENTION_SUM_ATOL,
        "letterbox_invalid_mass_max": 0.0,
    }


class _Stage:
    """Private whole-directory staging followed by one atomic final rename."""

    def __init__(self, final_dir: Path):
        self.final_dir = final_dir
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.final_dir.exists():
            raise FileExistsError(f"refusing to overwrite artifact directory: {self.final_dir}")
        self.stage_dir = self.final_dir.parent / f".{self.final_dir.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
        self.published = False

    def __enter__(self) -> _Stage:
        self.stage_dir.mkdir(parents=False, exist_ok=False)
        incomplete = {
            "schema": SCHEMA_VERSION,
            "status": "private staging; not complete or published",
            "final_destination": str(self.final_dir),
        }
        (self.stage_dir / "INCOMPLETE.json").write_text(json.dumps(incomplete, indent=2) + "\n", encoding="utf-8")
        return self

    def publish(self, identities: Mapping[str, Mapping[str, Any]]) -> None:
        fixed.decode_video._write_complete_atomic(self.stage_dir, dict(identities))
        (self.stage_dir / "INCOMPLETE.json").unlink()
        directory_fd = os.open(self.stage_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if self.final_dir.exists():
            raise FileExistsError(f"artifact destination appeared during run: {self.final_dir}")
        os.rename(self.stage_dir, self.final_dir)
        parent_fd = os.open(self.final_dir.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        self.published = True

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if not self.published and self.stage_dir.exists():
            shutil.rmtree(self.stage_dir)


class HeldoutWriterAttentionVideo:
    def __init__(self, args: Args):
        self.args = args
        if args.artifact_dir.exists():
            raise FileExistsError(f"refusing to overwrite artifact directory: {args.artifact_dir}")
        stage_probe = args.output_dir / f".{args.parameter_source}.runtime-preflight.{os.getpid()}.{uuid.uuid4().hex}"
        runtime_args = _RuntimeArgs(
            checkpoint=args.checkpoint,
            dataset_root=args.dataset_root,
            config=args.config,
            parameter_source=args.parameter_source,
            artifact_dir=stage_probe,
        )
        self.runtime = fixed._Run5Runtime(runtime_args)
        self._qstep = nnx_utils.module_jit(self.runtime.model.v32_query_attention_step)
        self._write = self.runtime._write
        if int(self.runtime.data_config.memory_stride_frames) != WRITE_STRIDE:
            raise ValueError("run5 memory_stride_frames is no longer exactly 15")
        step_identity = self.runtime.parameter_provenance.get("train_state_step_identity")
        expected_step_identity = {
            "checkpoint_manager_step_label": args.checkpoint_step,
            "internal_train_state_step": args.checkpoint_step + 1,
        }
        if step_identity != expected_step_identity:
            raise ValueError(f"checkpoint label/internal-step gate failed: {step_identity} != {expected_step_identity}")
        if float(self.runtime.model.memory.config.eta_scale) != 0.0:
            raise ValueError("run5 loaded memory eta_scale is not exactly zero")
        self.source_valid = _validate_source_valid(self.runtime.model.top_patch_valid)

    def _load_payloads(self) -> dict[int, dict[str, Any]]:
        all_plans = fixed.causal._probe._plan_all_episodes(
            SimpleNamespace(dataset_root=self.args.dataset_root), self.runtime.data_config
        )
        plans = {int(plan.episode): plan for plan in all_plans}
        if len(plans) != len(all_plans):
            raise ValueError("episode plan list contains duplicate ids")
        metadata = {}
        for row in fixed.causal._wc._read_jsonl(self.args.dataset_root / "meta" / "episodes.jsonl"):
            episode = row.get("episode_index")
            length = row.get("length")
            if (
                isinstance(episode, bool)
                or not isinstance(episode, int)
                or isinstance(length, bool)
                or not isinstance(length, int)
                or length <= 0
                or episode in metadata
            ):
                raise ValueError(f"invalid or duplicate episode metadata record: {row}")
            metadata[episode] = dict(row)

        sources = fixed.causal._wc._load_lerobot_sources(self.args.dataset_root, list(self.args.episodes))
        columns = [
            "image",
            "left_wrist_image",
            "right_wrist_image",
            "state",
            "frame_index",
            "episode_index",
            "task_index",
        ]
        payloads = {}
        for episode, source in zip(self.args.episodes, sources, strict=True):
            if episode not in plans or episode not in metadata:
                raise ValueError(f"heldout episode {episode} is missing from plan or metadata")
            plan = plans[episode]
            expected_cell = tuple(EXPECTED_CELLS[episode])
            if (plan.prompt, plan.side) != expected_cell:
                raise ValueError(
                    f"heldout ep{episode} cell changed: expected {expected_cell}, got {(plan.prompt, plan.side)}"
                )
            expected_length = EXPECTED_FRAME_COUNTS[episode]
            if int(plan.length) != expected_length or int(metadata[episode]["length"]) != expected_length:
                raise ValueError(f"heldout ep{episode} length changed from {expected_length}")
            task_names = tuple(source.task_names)
            if task_names != tuple(self.runtime.data_config.memory_subtask_vocab):
                raise ValueError("dataset task vocabulary/order differs from run5 config")
            fps = float(source.control_hz)
            if not math.isclose(fps, SOURCE_FPS, rel_tol=0.0, abs_tol=0.0):
                raise ValueError(f"episode {episode} source fps changed: expected {SOURCE_FPS}, got {fps}")
            parquet = pq.ParquetFile(source.path)
            if set(columns) - set(parquet.schema_arrow.names):
                raise ValueError(f"episode {episode} parquet is missing required columns")
            if parquet.metadata.num_rows != expected_length:
                raise ValueError(f"episode {episode} parquet row count changed")
            rows = parquet.read(columns=columns).to_pylist()
            label_runs = fixed.decode_video._validate_rows(
                rows,
                episode=episode,
                expected_length=expected_length,
                task_names=task_names,
            )
            if self.args.smoke_only:
                rows = rows[:SMOKE_FRAME_COUNT]
                label_runs = fixed.decode_video._validate_rows(
                    rows,
                    episode=episode,
                    expected_length=SMOKE_FRAME_COUNT,
                    task_names=task_names,
                )
            payloads[episode] = {
                "plan": plan,
                "source": source,
                "source_identity": fixed.causal._file_identity(source.path),
                "rows": rows,
                "task_names": task_names,
                "label_runs": label_runs,
                "full_frame_count": expected_length,
            }
        return payloads

    def _infer_episode(self, payload: dict[str, Any]) -> _EpisodeResult:
        plan = payload["plan"]
        episode = int(plan.episode)
        rows = payload["rows"]
        task_names = payload["task_names"]
        memory_state = self.runtime.model.memory.init_state(1)
        _state_max_abs(memory_state)

        frames = []
        task_indices = []
        commits = []
        slot_maps = []
        mean_maps = []
        top_images = []
        retrievals = []
        write_token_rms = []
        qstep_seconds = []
        commit_state_deltas = []
        commit_grad_norms = []
        commit_clip_factors = []
        smoke_fresh_m0_gate = None
        started = time.monotonic()

        for expected_frame, row in enumerate(rows):
            frame = int(row["frame_index"])
            if frame != expected_frame:
                raise RuntimeError(f"episode {episode} recurrence expected frame {expected_frame}, got {frame}")
            observation, _normalized_state = self.runtime.observation(row, frame, plan.prompt)
            top = fixed.decode_video._processed_images(observation, 0)["base_0_rgb"]
            qstep_started = time.monotonic()
            output = self._qstep(observation, memory_state)
            jax.block_until_ready(output)
            qstep_seconds.append(time.monotonic() - qstep_started)

            write_attention = np.asarray(output["write_attention"], dtype=np.float32)
            slots, mean_map = _head_mean_per_slot(write_attention, self.source_valid)
            _head_mean_per_slot(output["read_attention"], self.source_valid)
            tokens = np.asarray(output["write_tokens"], dtype=np.float32)
            retrieved = np.asarray(output["retrieved"], dtype=np.float32)
            if tokens.ndim != 3 or tokens.shape[:2] != (1, NUM_SLOTS) or not np.all(np.isfinite(tokens)):
                raise RuntimeError(f"episode {episode} frame {frame}: invalid candidate write tokens {tokens.shape}")
            if retrieved.ndim != 3 or retrieved.shape[:2] != (1, NUM_SLOTS) or not np.all(np.isfinite(retrieved)):
                raise RuntimeError(f"episode {episode} frame {frame}: invalid retrieval {retrieved.shape}")
            retrieval_rms = float(np.sqrt(np.mean(np.square(retrieved, dtype=np.float64))))
            candidate_rms = float(np.sqrt(np.mean(np.square(tokens, dtype=np.float64))))
            if frame == 0 and retrieval_rms > RETRIEVAL_ZERO_TOL:
                raise RuntimeError(f"episode {episode}: fresh M0 retrieval is not zero: {retrieval_rms}")

            if self.args.smoke_only and frame == SMOKE_FRAME_COUNT - 1:
                fresh_state = self.runtime.model.memory.init_state(1)
                fresh_output = self._qstep(observation, fresh_state)
                jax.block_until_ready(fresh_output)
                fresh_attention = np.asarray(fresh_output["write_attention"], dtype=np.float32)
                _head_mean_per_slot(fresh_attention, self.source_valid)
                fresh_tokens = np.asarray(fresh_output["write_tokens"], dtype=np.float32)
                fresh_retrieved = np.asarray(fresh_output["retrieved"], dtype=np.float32)
                attention_max_abs_difference = float(np.max(np.abs(write_attention - fresh_attention)))
                write_token_max_abs_difference = float(np.max(np.abs(tokens - fresh_tokens)))
                fresh_retrieval_rms = float(np.sqrt(np.mean(np.square(fresh_retrieved, dtype=np.float64))))
                retrieval_difference_rms = float(
                    np.sqrt(np.mean(np.square(retrieved - fresh_retrieved, dtype=np.float64)))
                )
                if attention_max_abs_difference > SMOKE_INVARIANCE_ATOL:
                    raise RuntimeError(
                        "smoke carried-state writer attention changed from fresh-M0 attention: "
                        f"max_abs={attention_max_abs_difference}"
                    )
                if write_token_max_abs_difference > SMOKE_INVARIANCE_ATOL:
                    raise RuntimeError(
                        "smoke carried-state candidate write tokens changed from fresh-M0 tokens: "
                        f"max_abs={write_token_max_abs_difference}"
                    )
                if fresh_retrieval_rms > RETRIEVAL_ZERO_TOL:
                    raise RuntimeError(f"smoke fresh-M0 retrieval is not zero: {fresh_retrieval_rms}")
                if retrieval_difference_rms <= RETRIEVAL_ZERO_TOL:
                    raise RuntimeError(
                        "smoke carried memory did not change retrieval relative to fresh M0: "
                        f"difference_rms={retrieval_difference_rms}"
                    )
                smoke_fresh_m0_gate = {
                    "status": "pass",
                    "episode": episode,
                    "frame": frame,
                    "comparison": "identical transformed observation; carried state after f0/f15 vs fresh M0",
                    "attention_max_abs_difference": attention_max_abs_difference,
                    "write_token_max_abs_difference": write_token_max_abs_difference,
                    "invariance_atol": SMOKE_INVARIANCE_ATOL,
                    "carried_retrieval_rms": retrieval_rms,
                    "fresh_m0_retrieval_rms": fresh_retrieval_rms,
                    "retrieval_difference_rms": retrieval_difference_rms,
                    "retrieval_difference_minimum": RETRIEVAL_ZERO_TOL,
                    "qstep_executable": QSTEP_EXECUTABLE_DESCRIPTION,
                    "same_executable_used_for_both_states": True,
                }

            scheduled = _is_scheduled_commit(frame)
            if scheduled:
                next_state, write_aux = self._write(memory_state, output["write_tokens"])
                jax.block_until_ready((next_state, write_aux))
                delta = self.runtime.state_max_abs_difference(next_state, memory_state)
                if not math.isfinite(delta) or delta <= 0.0:
                    raise RuntimeError(f"episode {episode} frame {frame}: committed write did not change state")
                eta = np.asarray(write_aux.get("eta"), dtype=np.float64)
                grad = np.asarray(write_aux.get("grad_norm"), dtype=np.float64)
                clip = np.asarray(write_aux.get("clip_factor"), dtype=np.float64)
                if eta.shape != (1,) or not np.all(np.isfinite(eta)) or float(eta[0]) != 0.0:
                    raise RuntimeError(f"episode {episode} frame {frame}: write eta is not exact zero: {eta}")
                if grad.shape != (1,) or not np.all(np.isfinite(grad)) or float(grad[0]) < 0.0:
                    raise RuntimeError(f"episode {episode} frame {frame}: invalid write grad norm: {grad}")
                if clip.shape != (1,) or not np.all(np.isfinite(clip)) or not 0.0 <= float(clip[0]) <= 1.0:
                    raise RuntimeError(f"episode {episode} frame {frame}: invalid write clip factor: {clip}")
                for name in ("surprise", "theta", "alpha"):
                    value = np.asarray(write_aux.get(name), dtype=np.float64)
                    if value.shape != (1,) or not np.all(np.isfinite(value)) or float(value[0]) < 0.0:
                        raise RuntimeError(f"episode {episode} frame {frame}: invalid write auxiliary {name}")
                    if name in ("theta", "alpha") and float(value[0]) > 1.0:
                        raise RuntimeError(f"episode {episode} frame {frame}: {name} is outside [0,1]")
                grad_value = float(grad[0])
                expected_clip = min(
                    1.0, float(self.runtime.model.memory.config.max_grad_norm) / (grad_value + 1e-12)
                )
                clip_factor = float(clip[0])
                if not math.isclose(clip_factor, expected_clip, rel_tol=0.0, abs_tol=1e-6):
                    raise RuntimeError(
                        f"episode {episode} frame {frame}: clip factor {clip_factor} != expected {expected_clip}"
                    )
                commit_state_deltas.append(delta)
                commit_grad_norms.append(grad_value)
                commit_clip_factors.append(clip_factor)
                memory_state = next_state

            frames.append(frame)
            task_indices.append(int(row["task_index"]))
            commits.append(scheduled)
            slot_maps.append(slots)
            mean_maps.append(mean_map)
            top_images.append(top)
            retrievals.append(retrieval_rms)
            write_token_rms.append(candidate_rms)
            if frame % 50 == 0 or frame + 1 == len(rows):
                print(
                    f"[ep{episode}] attention frame {frame}/{len(rows) - 1} "
                    f"({'COMMIT' if scheduled else 'candidate-only'}) elapsed={time.monotonic() - started:.1f}s",
                    flush=True,
                )

        frame_count = len(rows)
        expected_commits = _expected_commit_count(frame_count)
        observed_commits = int(np.count_nonzero(commits))
        if observed_commits != expected_commits:
            raise RuntimeError(f"episode {episode}: expected {expected_commits} commits, got {observed_commits}")
        if not self.args.smoke_only and EXPECTED_COMMIT_COUNTS[episode] != expected_commits:
            raise RuntimeError(f"episode {episode}: registered full commit count changed")
        if frame_count > 1 and not any(value > RETRIEVAL_ZERO_TOL for value in retrievals[1:]):
            raise RuntimeError(f"episode {episode}: retrieval never became nonzero after the first commit")
        if self.args.smoke_only and smoke_fresh_m0_gate is None:
            raise RuntimeError("smoke run did not execute the carried-state versus fresh-M0 gate")
        smoke_timing_gate = None
        if self.args.smoke_only:
            smoke_timing_gate = _smoke_timing_gate(qstep_seconds)
            print(f"[smoke timing gate] {json.dumps(smoke_timing_gate, sort_keys=True)}", flush=True)
            extrapolated_mean = smoke_timing_gate["extrapolated_full_qstep_seconds_from_mean"]
            if smoke_timing_gate["status"] != "pass":
                raise RuntimeError(
                    "smoke extrapolates stride-1 unmodified qstep inference beyond the 2.5h operational gate: "
                    f"{extrapolated_mean / 3600.0:.3f}h"
                )
        inference_elapsed_seconds = time.monotonic() - started
        return _EpisodeResult(
            episode=episode,
            prompt=str(plan.prompt),
            side=str(plan.side),
            fps=float(payload["source"].control_hz),
            frames=np.asarray(frames, dtype=np.int32),
            task_indices=np.asarray(task_indices, dtype=np.int16),
            scheduled_commits=np.asarray(commits, dtype=bool),
            slot_maps=np.stack(slot_maps).astype(np.float32),
            mean_maps=np.stack(mean_maps).astype(np.float32),
            top_images=top_images,
            retrieval_rms=np.asarray(retrievals, dtype=np.float32),
            write_token_rms=np.asarray(write_token_rms, dtype=np.float32),
            qstep_seconds=np.asarray(qstep_seconds, dtype=np.float32),
            commit_state_deltas=np.asarray(commit_state_deltas, dtype=np.float32),
            commit_grad_norms=np.asarray(commit_grad_norms, dtype=np.float32),
            commit_clip_factors=np.asarray(commit_clip_factors, dtype=np.float32),
            final_state_max_abs=_state_max_abs(memory_state),
            source_identity=payload["source_identity"],
            task_names=task_names,
            label_runs=payload["label_runs"],
            smoke_fresh_m0_gate=smoke_fresh_m0_gate,
            smoke_timing_gate=smoke_timing_gate,
            inference_elapsed_seconds=inference_elapsed_seconds,
        )

    def _write_episode_artifacts(
        self,
        stage_dir: Path,
        result: _EpisodeResult,
        scale: token_heatmap.ColorScale,
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        episode = result.episode
        stem = f"episode_{episode:06d}"
        frame_count = len(result.frames)
        identities: dict[str, dict[str, Any]] = {}

        npz_name = f"{stem}_writer_attention_maps.npz"
        npz_path = stage_dir / npz_name
        identities[npz_name] = _write_npz_atomic(
            npz_path,
            {
                "frame_index": result.frames,
                "task_index": result.task_indices,
                "scheduled_commit": result.scheduled_commits,
                "source_valid": self.source_valid,
                "attention_head_mean_slots": result.slot_maps.astype(np.float16),
                "attention_mean_heads_slots": result.mean_maps.astype(np.float16),
                "retrieval_rms": result.retrieval_rms,
                "write_token_rms": result.write_token_rms,
            },
        )
        npz_checks = _validate_saved_npz(npz_path, frame_count=frame_count, source_valid=self.source_valid)

        video_name = f"{stem}_writer_attention.mp4"
        video_path = stage_dir / video_name
        video = _AtomicMp4Writer(video_path, fps=result.fps, width=MAIN_WIDTH, height=MAIN_HEIGHT)
        try:
            for index, frame in enumerate(result.frames):
                gt = result.task_names[int(result.task_indices[index])]
                rendered = _render_main_frame(
                    result.top_images[index],
                    result.mean_maps[index],
                    scale,
                    episode=episode,
                    frame=int(frame),
                    frame_count=frame_count,
                    fps=result.fps,
                    prompt=result.prompt,
                    side=result.side,
                    gt=gt,
                    checkpoint_step=self.args.checkpoint_step,
                    parameter_source=self.args.parameter_source,
                    scheduled_commit=bool(result.scheduled_commits[index]),
                )
                video.write(rendered)
            identities[video_name] = video.close()
        except BaseException:
            video.abort()
            raise
        video_probe = _probe_mp4(
            video_path,
            fps=result.fps,
            width=MAIN_WIDTH,
            height=MAIN_HEIGHT,
            frames=frame_count,
        )

        slot_name = None
        slot_probe = None
        if self.args.write_frame_slot_grid:
            indices = np.flatnonzero(result.scheduled_commits)
            slot_name = f"{stem}_scheduled_write_slots_v32.mp4"
            slot_path = stage_dir / slot_name
            labels = [
                f"ep{episode} f{int(result.frames[index])} COMMIT | {result.prompt} | "
                f"GT={result.task_names[int(result.task_indices[index])]}"
                for index in indices
            ]
            writer_attention.render_slot_grid(
                [result.top_images[index] for index in indices],
                result.slot_maps[indices].astype(np.float64),
                labels,
                slot_path,
                fps=SLOT_GRID_FPS,
                style="v32",
            )
            identities[slot_name] = fixed.causal._file_identity(slot_path)
            slot_probe = _probe_mp4(
                slot_path,
                fps=SLOT_GRID_FPS,
                width=4 * token_heatmap.MODEL_IMAGE_SIZE,
                height=writer_attention._SLOT_GRID_HEADER + 4 * token_heatmap.MODEL_IMAGE_SIZE,
                frames=len(indices),
            )

        image_digest = hashlib.sha256()
        for frame, image in zip(result.frames, result.top_images, strict=True):
            image_digest.update(np.asarray(frame, dtype=np.int32).tobytes())
            image_digest.update(np.ascontiguousarray(image).tobytes())
        episode_summary = {
            "episode": episode,
            "prompt": result.prompt,
            "heldout_side": result.side,
            "fps": result.fps,
            "frame_count": frame_count,
            "frame_range": [int(result.frames[0]), int(result.frames[-1])],
            "inference_elapsed_seconds": result.inference_elapsed_seconds,
            "label_runs": result.label_runs,
            "source_parquet_identity": result.source_identity,
            "top_model_input_sequence_sha256": image_digest.hexdigest(),
            "attention": {
                "raw_model_shape": [1, NUM_HEADS, NUM_SLOTS, NUM_PATCHES],
                "saved_head_mean_per_slot_shape": [frame_count, NUM_SLOTS, NUM_PATCHES],
                "token_layout": token_heatmap.TOKEN_LAYOUT,
                "main_overlay_reduction": "arithmetic mean over 8 heads, then arithmetic mean over 16 slots",
                "global_scale": scale.to_dict(),
                "npz_validation": npz_checks,
            },
            "online_memory": {
                "recurrence_M0_initializations": 1,
                "smoke_control_M0_initializations": 1 if self.args.smoke_only else 0,
                **_qstep_call_accounting(frame_count, smoke_only=self.args.smoke_only),
                "commit_write_calls": int(np.count_nonzero(result.scheduled_commits)),
                "write_schedule": "frame_index % 15 == 0",
                "state_threading": "post-write state on commit frames; identical input state on candidate-only frames",
                "retrieval_rms_min": float(np.min(result.retrieval_rms)),
                "retrieval_rms_max": float(np.max(result.retrieval_rms)),
                "candidate_write_token_rms_min": float(np.min(result.write_token_rms)),
                "candidate_write_token_rms_max": float(np.max(result.write_token_rms)),
                "commit_state_delta_min": float(np.min(result.commit_state_deltas)),
                "commit_state_delta_max": float(np.max(result.commit_state_deltas)),
                "commit_grad_norm_min": float(np.min(result.commit_grad_norms)),
                "commit_grad_norm_max": float(np.max(result.commit_grad_norms)),
                "commit_clip_factor_min": float(np.min(result.commit_clip_factors)),
                "commit_clip_factor_max": float(np.max(result.commit_clip_factors)),
                "eta_exact_zero_on_every_commit": True,
                "final_state_max_abs": result.final_state_max_abs,
            },
            "smoke_fresh_m0_gate": result.smoke_fresh_m0_gate,
            "smoke_timing_gate": result.smoke_timing_gate,
            "artifacts": {
                "maps_npz": npz_name,
                "main_video": video_name,
                "scheduled_write_slot_video": slot_name,
            },
            "video_validation": {"main": video_probe, "scheduled_write_slots": slot_probe},
        }
        return episode_summary, identities

    def _provenance(
        self, payloads: Mapping[int, dict[str, Any]], results: Sequence[_EpisodeResult]
    ) -> dict[str, Any]:
        step_identity = self.runtime.parameter_provenance["train_state_step_identity"]
        return {
            "schema": SCHEMA_VERSION,
            "checkpoint": {
                "resolved_path": str(self.args.checkpoint),
                "checkpoint_manager_step_label": self.args.checkpoint_step,
                "internal_train_state_step": self.args.checkpoint_step + 1,
                "validated_identity": step_identity,
            },
            "checkpoint_origin": self.runtime.checkpoint_origin,
            "parameter_source": self.args.parameter_source,
            "parameter_provenance": self.runtime.parameter_provenance,
            "normalization_asset_provenance": self.runtime.norm_provenance,
            "tokenizer_asset_provenance": self.runtime.tokenizer_provenance,
            "source_provenance": {
                "exact_run5_launch_source": self.runtime.source_provenance,
                "current_diagnostic_script": fixed.causal._file_identity(Path(__file__).resolve()),
                "fixed_runtime_script": fixed.causal._file_identity(Path(fixed.__file__).resolve()),
                "writer_attention_renderer": fixed.causal._file_identity(Path(writer_attention.__file__).resolve()),
                "token_heatmap_renderer": fixed.causal._file_identity(Path(token_heatmap.__file__).resolve()),
            },
            "config": self.args.config,
            "run5_static_gates": {
                "eta_scale": float(self.runtime.model.memory.config.eta_scale),
                "memory_write_stride_frames": int(self.runtime.data_config.memory_stride_frames),
                "heldout_episodes": list(self.runtime.data_config.heldout_episodes),
                "top_camera_valid_mask_sha256": fixed.causal._array_sha256(self.source_valid),
                "top_camera_valid_patch_count": int(self.source_valid.sum()),
            },
            "diagnostic_runtime_contract": {
                "qstep_implementation": QSTEP_EXECUTABLE_DESCRIPTION,
                "full_model_diagnostic_output_pytree_materialized_every_call": True,
                "frozen_nnx_state_semantics": True,
                "token_layout": token_heatmap.TOKEN_LAYOUT,
                "smoke_fresh_m0_invariance_atol": SMOKE_INVARIANCE_ATOL,
                "smoke_carried_and_fresh_use_same_qstep_executable": True,
                "smoke_measured_gates": {
                    str(result.episode): {
                        "fresh_m0": result.smoke_fresh_m0_gate,
                        "timing": result.smoke_timing_gate,
                    }
                    for result in results
                },
            },
            "dataset_root": str(self.args.dataset_root),
            "dataset_metadata_identity": {
                name: fixed.causal._file_identity(self.args.dataset_root / "meta" / name)
                for name in ("info.json", "tasks.jsonl", "episodes.jsonl", "episode_prompts.json")
            },
            "source_parquet_identity": {
                str(episode): payload["source_identity"] for episode, payload in payloads.items()
            },
        }

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        payloads = self._load_payloads()
        results = [self._infer_episode(payloads[episode]) for episode in self.args.episodes]
        scale = _fit_global_scale(results, self.source_valid)
        provenance = self._provenance(payloads, results)

        with _Stage(self.args.artifact_dir) as stage:
            identities: dict[str, dict[str, Any]] = {}
            episode_summaries = []
            for result in results:
                summary, episode_identities = self._write_episode_artifacts(stage.stage_dir, result, scale)
                episode_summaries.append(summary)
                overlap = set(identities) & set(episode_identities)
                if overlap:
                    raise RuntimeError(f"duplicate artifact names: {sorted(overlap)}")
                identities.update(episode_identities)

            provenance_name = "provenance.json"
            identities[provenance_name] = fixed.decode_video._write_json_atomic(
                stage.stage_dir / provenance_name, provenance
            )
            report = {
                "schema": SCHEMA_VERSION,
                "mode": "smoke-only" if self.args.smoke_only else "full",
                "status": "pass",
                "checkpoint_step": self.args.checkpoint_step,
                "internal_train_state_step": self.args.checkpoint_step + 1,
                "parameter_source": self.args.parameter_source,
                "episodes": episode_summaries,
                "global_attention_scale": {
                    **scale.to_dict(),
                    "scope": "all selected heldout episodes and all processed raw frames",
                    "fit_values": "mean over 8 heads and 16 slots; letterbox-invalid patches excluded",
                },
                "inference_contract": {
                    "raw_frame_stride": 1,
                    "recurrence_qstep_and_read_called_every_frame": True,
                    "recurrence_candidate_write_tokens_computed_every_frame": True,
                    "commit_write_schedule": "frame_index % 15 == 0",
                    "recurrence_memory_state_initialized_once_per_episode": True,
                    "smoke_fresh_M0_control_initializations": 1 if self.args.smoke_only else 0,
                    "memory_state_threaded_strictly_in_frame_order": True,
                    "off_schedule_candidate_writes_not_committed": True,
                    "eta_exact_zero_on_all_commits": True,
                    "checkpoint_label_internal_step_gate": True,
                    "qstep_execution": {
                        "callable": QSTEP_EXECUTABLE_DESCRIPTION,
                        "full_model_diagnostic_output_pytree_materialized_every_call": True,
                        "same_executable_carried_vs_fresh_M0_checked_in_smoke": self.args.smoke_only,
                    },
                },
                "render_contract": {
                    "main_video_fps": SOURCE_FPS,
                    "main_video_size": [MAIN_WIDTH, MAIN_HEIGHT],
                    "main_video_scale_fixed_globally": True,
                    "token_layout": token_heatmap.TOKEN_LAYOUT,
                    "main_legend": "per-patch attention percent with inferno color bar; values above p99 clipped",
                    "top_image": "exact 224x224 model input converted to uint8 and displayed by 2x nearest-neighbour",
                    "slot_grid": (
                        "scheduled write frames only; 4 fps; v3.2 per-frame per-slot relative scale"
                        if self.args.write_frame_slot_grid
                        else "disabled by explicit CLI flag"
                    ),
                    "map_caveat": (
                        "these are writer-query pooling weights over spatially aligned layer-8 top-camera token "
                        "positions; layer-8 token content is contextualized, so this is not raw-pixel causality"
                    ),
                },
                "provenance_file": provenance_name,
                "elapsed_seconds": time.monotonic() - started,
            }
            summary_name = "summary.json"
            identities[summary_name] = fixed.decode_video._write_json_atomic(stage.stage_dir / summary_name, report)
            stage.publish(identities)

        print(
            json.dumps(
                {
                    "status": "pass",
                    "artifact_dir": str(self.args.artifact_dir),
                    "episodes": list(self.args.episodes),
                    "global_scale_vmax": scale.vmax,
                    "elapsed_seconds": report["elapsed_seconds"],
                },
                indent=2,
            ),
            flush=True,
        )
        return report


def main(argv: list[str] | None = None) -> int:
    HeldoutWriterAttentionVideo(_parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
