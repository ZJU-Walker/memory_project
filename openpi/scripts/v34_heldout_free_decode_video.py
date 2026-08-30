"""Render stride-1, reset-memory free decodes for the four run5 held-out episodes.

This is a deliberately simple behavioral diagnostic.  At every parquet frame it gives the
run5 checkpoint the native three-camera observation, robot state, and episode instruction, but
always starts from a fresh M0.  No history is replayed, writes are disabled, and the returned
state is never threaded.  The model greedily decodes its subtask; both the observation-time
label at ``t`` and the deliberately look-ahead-shifted training label at ``min(t+15, T-1)`` are
shown beside the exact 224x224 images used by inference.

Two processes can split the work deterministically.  For ``--num-shards 2``, shard 0 receives
episodes 15+44 and shard 1 receives 29+59.  Each shard writes to its own child directory under
``--output-dir``, so both may safely run concurrently.

Example (run from the OpenPI repository root):

    # First extract cluster_v34/provenance/v34_run5_eta0_main_launch/source_snapshot.tar.gz
    # into /tmp/v34_run5_source (once on the allocated node), then:
    V34_RUN5_SOURCE_ROOT=/tmp/v34_run5_source \
    PYTHONPATH=/tmp/v34_run5_source/src \
    .venv/bin/python scripts/v34_heldout_free_decode_video.py \
      --checkpoint diagnostic_checkpoints/v34_run5_eta0_pilot_copies/2500 \
      --dataset-root /iris/u/kewalk/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask \
      --output-dir diagnostic_outputs/v34_heldout_free_decode/2500/raw \
      --config pi05_yam_mem_v34_run5_eta0 --parameter-source raw \
      --num-shards 2 --shard-id 0

Run once with ``--smoke-only`` and a separate output directory before the full render.  The
smoke checks batch-vs-single decode equivalence, M0 read-vs-zero-read equivalence, and subtask
token invariance between one and ten action denoising steps on phase-boundary frames.
"""

# ruff: noqa: SLF001, I001 - pyarrow must precede the audited OpenPI/JAX import stack.
from __future__ import annotations

import pyarrow.parquet as pq

import argparse
from collections.abc import Sequence
import dataclasses
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
from types import SimpleNamespace
from typing import Any, Literal

# Evidence must never fetch a different tokenizer/config revision.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import cv2
import jax
import jax.numpy as jnp
import numpy as np

import v34_causal_memory_eval as causal


SCHEMA_VERSION = "openpi.v34.heldout_free_decode_video.v1"
CHECKPOINT_STEP = 2500
INTERNAL_TRAIN_STEP = 2501
HELDOUT_EPISODES = causal.EXPECTED_HELDOUT
EXPECTED_CELLS = causal.EXPECTED_CELLS
TRAIN_TARGET_LOOKAHEAD = 15
PRODUCTION_NUM_STEPS = 1
PRODUCTION_MAX_DECODE_STEPS = 10
PALIGEMMA_EOS_TOKEN = 1
CAMERA_KEYS = (
    ("TOP", "base_0_rgb"),
    ("LEFT WRIST", "left_wrist_0_rgb"),
    ("RIGHT WRIST", "right_wrist_0_rgb"),
)
IMAGE_SIZE = 224
DISPLAY_SCALE = 2
DISPLAY_IMAGE_SIZE = IMAGE_SIZE * DISPLAY_SCALE
CANVAS_WIDTH = 3 * DISPLAY_IMAGE_SIZE
HEADER_HEIGHT = 32
TEXT_HEIGHT = 224
CANVAS_HEIGHT = HEADER_HEIGHT + DISPLAY_IMAGE_SIZE + TEXT_HEIGHT
RETRIEVAL_ZERO_TOL = 1e-7
RUN5_SOURCE_ENV = "V34_RUN5_SOURCE_ROOT"
RUN5_SOURCE_FILES = (
    "src/openpi/training/config.py",
    "src/openpi/transforms.py",
    "src/openpi/training/data_loader.py",
    "src/openpi/models/pi0.py",
    "src/openpi/models/memory.py",
    "src/openpi/models/model.py",
    "src/openpi/models/tokenizer.py",
    "src/openpi/policies/yam_policy.py",
)


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint: Path
    dataset_root: Path
    output_dir: Path
    config: str
    parameter_source: Literal["raw", "ema"]
    batch_size: int = 8
    num_shards: int = 1
    shard_id: int = 0
    smoke_only: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", Path(self.checkpoint).expanduser().resolve())
        object.__setattr__(self, "dataset_root", Path(self.dataset_root).expanduser().resolve())
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        if self.checkpoint.name != str(CHECKPOINT_STEP):
            raise ValueError(
                f"free-decode video is pinned to checkpoint {CHECKPOINT_STEP}; got {self.checkpoint.name!r}"
            )
        if self.config != causal.RUN5_CONFIG:
            raise ValueError(f"free-decode video is pinned to --config {causal.RUN5_CONFIG!r}; got {self.config!r}")
        if self.parameter_source not in ("raw", "ema"):
            raise ValueError("--parameter-source must be raw or ema")
        if isinstance(self.batch_size, bool) or not 1 <= self.batch_size <= 64:
            raise ValueError("--batch-size must lie in [1, 64]")
        if isinstance(self.num_shards, bool) or not 1 <= self.num_shards <= len(HELDOUT_EPISODES):
            raise ValueError(f"--num-shards must lie in [1, {len(HELDOUT_EPISODES)}]")
        if isinstance(self.shard_id, bool) or not 0 <= self.shard_id < self.num_shards:
            raise ValueError("--shard-id must lie in [0, num-shards)")
        if self.seed != 0:
            raise ValueError("the diagnostic requires --seed 0")

    @property
    def episodes(self) -> tuple[int, ...]:
        return _episodes_for_shard(self.num_shards, self.shard_id)

    @property
    def artifact_dir(self) -> Path:
        if self.num_shards == 1:
            return self.output_dir
        return self.output_dir / f"shard_{self.shard_id:02d}_of_{self.num_shards:02d}"


def _parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parameter-source", choices=("raw", "ema"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return Args(**vars(parser.parse_args(argv)))


def _episodes_for_shard(num_shards: int, shard_id: int) -> tuple[int, ...]:
    if isinstance(num_shards, bool) or not 1 <= num_shards <= len(HELDOUT_EPISODES):
        raise ValueError(f"num_shards must lie in [1, {len(HELDOUT_EPISODES)}]")
    if isinstance(shard_id, bool) or not 0 <= shard_id < num_shards:
        raise ValueError("shard_id must lie in [0, num_shards)")
    episodes = tuple(episode for position, episode in enumerate(HELDOUT_EPISODES) if position % num_shards == shard_id)
    if not episodes:
        raise ValueError("shard mapping unexpectedly produced no episodes")
    if num_shards == 2:
        expected = ((15, 44), (29, 59))[shard_id]
        if episodes != expected:
            raise AssertionError(f"two-shard mapping changed: expected {expected}, got {episodes}")
    return episodes


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    episode: int,
    expected_length: int,
    task_names: Sequence[str],
) -> list[dict[str, Any]]:
    """Fail closed on frame identity, task identity, and discontinuous label blocks."""
    if not rows:
        raise ValueError(f"episode {episode} has no parquet rows")
    if len(rows) != expected_length:
        raise ValueError(
            f"episode {episode} frame count mismatch: plan/metadata={expected_length}, parquet={len(rows)}"
        )
    if not task_names:
        raise ValueError("global task vocabulary is empty")

    run_task_indices: list[int] = []
    previous_task: int | None = None
    for expected_frame, row in enumerate(rows):
        frame = row.get("frame_index")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame != expected_frame:
            raise ValueError(
                f"episode {episode} frame_index must be contiguous from zero: expected {expected_frame}, got {frame!r}"
            )
        row_episode = row.get("episode_index")
        if row_episode is not None and (
            isinstance(row_episode, bool) or not isinstance(row_episode, int) or row_episode != episode
        ):
            raise ValueError(f"episode {episode} row {frame} has mismatched episode_index {row_episode!r}")
        task = row.get("task_index")
        if isinstance(task, bool) or not isinstance(task, int) or not 0 <= task < len(task_names):
            raise ValueError(f"episode {episode} frame {frame} has invalid task_index {task!r}")
        state = np.asarray(row.get("state"), dtype=np.float32)
        if state.shape != (14,) or not np.all(np.isfinite(state)):
            raise ValueError(f"episode {episode} frame {frame} has invalid state shape/content")
        if task != previous_task:
            run_task_indices.append(task)
            previous_task = task

    if len(run_task_indices) != len(set(run_task_indices)):
        raise ValueError(f"episode {episode} task labels are not contiguous blocks: {run_task_indices}")
    return [
        {
            "start": start,
            "end": end,
            "task_index": int(rows[start]["task_index"]),
            "label": task_names[int(rows[start]["task_index"])],
        }
        for start, end in _run_bounds([int(row["task_index"]) for row in rows])
    ]


def _run_bounds(values: Sequence[int]) -> list[tuple[int, int]]:
    if not values:
        return []
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        if index == len(values) or values[index] != values[start]:
            bounds.append((start, index - 1))
            start = index
    return bounds


def _boundary_frames(rows: Sequence[dict[str, Any]]) -> tuple[int, ...]:
    if not rows:
        raise ValueError("cannot select boundaries from empty rows")
    tasks = [int(row["task_index"]) for row in rows]
    frames = {0, len(rows) - 1}
    for start, end in _run_bounds(tasks):
        frames.add(start)
        frames.add(end)
    return tuple(sorted(frames))


def _training_target(rows: Sequence[dict[str, Any]], frame: int, lookahead: int) -> tuple[int, int]:
    if not 0 <= frame < len(rows):
        raise ValueError(f"frame {frame} is outside [0, {len(rows)})")
    if lookahead < 0:
        raise ValueError("lookahead must be nonnegative")
    target = min(frame + lookahead, len(rows) - 1)
    return target, int(rows[target]["task_index"])


def _pad_batch(entries: Sequence[Any], batch_size: int) -> tuple[list[Any], int]:
    if not entries or len(entries) > batch_size:
        raise ValueError("batch entries must be nonempty and no larger than batch_size")
    live = len(entries)
    return [*entries, *([entries[-1]] * (batch_size - live))], live


def _normalized_image_to_rgb(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError(f"processed model image must be 224x224x3, got {image.shape}")
    if image.dtype == np.uint8:
        return np.array(image, copy=True)
    if image.dtype.kind != "f" or not np.all(np.isfinite(image)):
        raise TypeError("processed model image must be uint8 or contain finite floats")
    if float(np.min(image)) < -1.00001 or float(np.max(image)) > 1.00001:
        raise ValueError("processed model image is outside [-1, 1]")
    return np.rint((np.clip(image, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)


def _processed_images(observation: Any, index: int) -> dict[str, np.ndarray]:
    if tuple(observation.images) != tuple(key for _label, key in CAMERA_KEYS):
        raise ValueError(
            f"model image keys/order changed: expected {[key for _label, key in CAMERA_KEYS]}, "
            f"got {list(observation.images)}"
        )
    if set(observation.images) != set(observation.image_masks):
        raise ValueError("model image and mask keys differ")
    result = {}
    for _label, key in CAMERA_KEYS:
        if not bool(np.asarray(observation.image_masks[key])[index]):
            raise ValueError(f"camera {key} is unexpectedly masked at batch row {index}")
        result[key] = _normalized_image_to_rgb(np.asarray(observation.images[key])[index])
    return result


def _safe_overlay_text(value: str) -> str:
    return value.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def _put_wrapped(
    canvas: np.ndarray,
    text: str,
    *,
    y: int,
    color: tuple[int, int, int],
    prefix: str = "",
    width: int = 180,
) -> int:
    rendered = prefix + _safe_overlay_text(text)
    lines = textwrap.wrap(
        rendered,
        width=width,
        replace_whitespace=False,
        drop_whitespace=False,
        break_long_words=True,
        break_on_hyphens=False,
    ) or [""]
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 20
    return y


def _render_frame(
    images: dict[str, np.ndarray],
    *,
    episode: int,
    frame: int,
    fps: float,
    prompt: str,
    decoded: str,
    decoded_token_count: int,
    decoded_terminated: bool,
    decoded_truncated: bool,
    gt_now: str,
    target_frame: int,
    gt_train_target: str,
    parameter_source: str,
    config: str,
    memory_caption: str | None = None,
) -> np.ndarray:
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    source_tiles = [images[key] for _label, key in CAMERA_KEYS]
    if any(tile.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or tile.dtype != np.uint8 for tile in source_tiles):
        raise ValueError("render inputs must be exact uint8 224x224 processed camera images")
    # Nearest-neighbour 2x display preserves every model-input pixel as an exact 2x2 block.
    tiles = [
        cv2.resize(tile, (DISPLAY_IMAGE_SIZE, DISPLAY_IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        for tile in source_tiles
    ]

    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    for index, ((label, _key), tile) in enumerate(zip(CAMERA_KEYS, tiles, strict=True)):
        x0 = index * DISPLAY_IMAGE_SIZE
        canvas[
            HEADER_HEIGHT : HEADER_HEIGHT + DISPLAY_IMAGE_SIZE,
            x0 : x0 + DISPLAY_IMAGE_SIZE,
        ] = tile
        size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        cv2.putText(
            canvas,
            label,
            (x0 + (DISPLAY_IMAGE_SIZE - size[0]) // 2, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )

    if decoded_terminated == decoded_truncated:
        raise ValueError("a decode must be exactly one of terminated or truncated")
    decode_status = "TERMINATED" if decoded_terminated else "TRUNCATED AT LIMIT"
    y = HEADER_HEIGHT + DISPLAY_IMAGE_SIZE + 22
    y = _put_wrapped(
        canvas,
        decoded,
        y=y,
        color=(80, 255, 120),
        prefix=(f"MODEL FREE DECODE [{decode_status}; tokens={decoded_token_count}/{PRODUCTION_MAX_DECODE_STEPS}]: "),
    )
    y = _put_wrapped(canvas, gt_now, y=y, color=(255, 225, 60), prefix="GT NOW (t): ")
    y = _put_wrapped(
        canvas,
        gt_train_target,
        y=y,
        color=(255, 165, 70),
        prefix=f"GT TRAIN TARGET (min(t+15,T-1)=f{target_frame}): ",
    )
    y = _put_wrapped(
        canvas,
        f"episode={episode} frame={frame} time={frame / fps:.3f}s/{fps:g}fps | prompt={prompt}",
        y=y,
        color=(225, 225, 225),
    )
    y = _put_wrapped(
        canvas,
        f"checkpoint=2500 params={parameter_source} | config={config}",
        y=y,
        color=(185, 210, 255),
    )
    _put_wrapped(
        canvas,
        memory_caption
        or (
            "Fresh independent M0 for this frame (fixed padded batch is throughput only) | "
            "NO history | writes OFF/frozen | read enabled | returned state discarded"
        ),
        y=y,
        color=(185, 210, 255),
    )
    return canvas


class _AtomicMp4Writer:
    def __init__(self, path: Path, fps: float):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite video: {path}")
        if CANVAS_WIDTH % 2 or CANVAS_HEIGHT % 2:
            raise ValueError("H.264 yuv420p dimensions must be even")
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise FileNotFoundError("ffmpeg is required for H.264 MP4 output")
        self.path = path
        self.temporary = path.with_name(f".{path.name}.tmp")
        if self.temporary.exists():
            raise FileExistsError(f"stale temporary video exists: {self.temporary}")
        self.process = subprocess.Popen(
            [
                executable,
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{CANVAS_WIDTH}x{CANVAS_HEIGHT}",
                "-r",
                f"{fps:g}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
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
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != (CANVAS_HEIGHT, CANVAS_WIDTH, 3) or frame.dtype != np.uint8:
            raise ValueError(f"invalid rendered video frame: {frame.shape}/{frame.dtype}")
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
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
        if not self.temporary.is_file() or self.temporary.stat().st_size == 0:
            raise RuntimeError("ffmpeg succeeded without producing a nonempty MP4")
        with self.temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(self.temporary, self.path)
        return causal._file_identity(self.path)

    def abort(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin is not None:
                self.process.stdin.close()
                self.process.stdin = None
            self.process.terminate()
            self.process.wait(timeout=10)


def _write_json_atomic(path: Path, value: Any) -> dict[str, Any]:
    encoded = json.dumps(causal._strict_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite JSON artifact: {path}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return causal._file_identity(path)


def _write_complete_atomic(output_dir: Path, artifacts: dict[str, dict[str, Any]]) -> None:
    path = output_dir / "COMPLETE"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite completion marker: {path}")
    lines = []
    for name in sorted(artifacts):
        identity = artifacts[name]
        if not isinstance(identity.get("sha256"), str) or len(identity["sha256"]) != 64:
            raise ValueError(f"artifact {name} lacks a SHA-256 identity")
        lines.append(f"{identity['sha256']}  {name}")
    temporary = output_dir / ".COMPLETE.tmp"
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_run5_source_root(repo: Path) -> dict[str, Any]:
    raw_root = os.environ.get(RUN5_SOURCE_ENV)
    if not raw_root:
        raise RuntimeError(
            f"{RUN5_SOURCE_ENV} is required and must point to an extracted run5 "
            "source_snapshot.tar.gz; launch with that root's src directory first on PYTHONPATH"
        )
    source_root = Path(raw_root).expanduser().resolve()
    source_src = source_root / "src"
    if not source_src.is_dir():
        raise FileNotFoundError(f"{RUN5_SOURCE_ENV} lacks src/: {source_root}")
    snapshot = repo / causal.LAUNCH_PROVENANCE / "source_snapshot.tar.gz"
    if not snapshot.is_file():
        raise FileNotFoundError(f"run5 launch source snapshot is missing: {snapshot}")

    expected: dict[str, str] = {}
    with tarfile.open(snapshot, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}
        for relative in RUN5_SOURCE_FILES:
            member = members.get(relative)
            if member is None:
                raise ValueError(f"run5 source snapshot lacks {relative}")
            handle = archive.extractfile(member)
            if handle is None:
                raise ValueError(f"cannot read run5 source snapshot member {relative}")
            expected[relative] = hashlib.sha256(handle.read()).hexdigest()

    actual: dict[str, dict[str, Any]] = {}
    for relative in RUN5_SOURCE_FILES:
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"extracted run5 source lacks {relative}: {path}")
        identity = causal._file_identity(path)
        if identity["sha256"] != expected[relative]:
            raise ValueError(
                f"extracted run5 source hash mismatch for {relative}: "
                f"expected {expected[relative]}, got {identity['sha256']}"
            )
        actual[relative] = identity

    modules = {
        "src/openpi/training/config.py": importlib.import_module("openpi.training.config"),
        "src/openpi/transforms.py": importlib.import_module("openpi.transforms"),
        "src/openpi/training/data_loader.py": importlib.import_module("openpi.training.data_loader"),
        "src/openpi/models/pi0.py": importlib.import_module("openpi.models.pi0"),
        "src/openpi/models/memory.py": importlib.import_module("openpi.models.memory"),
        "src/openpi/models/model.py": importlib.import_module("openpi.models.model"),
        "src/openpi/models/tokenizer.py": importlib.import_module("openpi.models.tokenizer"),
        "src/openpi/policies/yam_policy.py": importlib.import_module("openpi.policies.yam_policy"),
    }
    resolved_modules = {}
    for relative, module in modules.items():
        module_path = Path(module.__file__).resolve()
        expected_path = (source_root / relative).resolve()
        if module_path != expected_path or not causal._is_relative_to(module_path, source_src):
            raise RuntimeError(
                f"{module.__name__} resolved outside the extracted run5 source: "
                f"expected {expected_path}, got {module_path}; put {source_src} first on PYTHONPATH"
            )
        resolved_modules[module.__name__] = str(module_path)

    return {
        "required_environment_variable": RUN5_SOURCE_ENV,
        "resolved_root": str(source_root),
        "resolved_src": str(source_src),
        "launch_snapshot_identity": causal._file_identity(snapshot),
        "validated_file_identities": actual,
        "resolved_module_paths": resolved_modules,
        "python_sys_path": list(sys.path),
        "all_required_hashes_and_module_origins_match": True,
    }


class _Run5Runtime:
    """Audited runtime loaded from the exact extracted run5 launch-source snapshot.

    The current diagnostic script stays outside that snapshot, while every OpenPI module that
    defines config, transforms, data loading, model, memory, tokenizer, or YAM policy is required
    to resolve inside ``V34_RUN5_SOURCE_ROOT`` and match the archived launch snapshot byte for
    byte.  Checkpoint origin/internal step, parameter source, tokenizer files, normalization
    assets, and static run5 semantics are independently validated as well.
    """

    def __init__(self, args: Args):
        self.repo = Path(__file__).resolve().parents[1]
        self.run5_source_provenance = _validate_run5_source_root(self.repo)
        self.checkpoint_info = causal._checkpoint_metadata(args.checkpoint)
        self.checkpoint_origin = causal._validate_checkpoint_origin(args.checkpoint, self.repo, self.checkpoint_info)
        causal._preflight_output(args.artifact_dir, args.checkpoint)
        if not args.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")

        self.train_config = causal._config.get_config(args.config)
        self.tokenizer_asset_provenance = causal._resolve_tokenizer_assets()
        self.data_config = self.train_config.data.create(self.train_config.assets_dirs, self.train_config.model)
        causal._validate_run5_semantics(
            args.config,
            self.train_config.model,
            self.data_config,
            self.train_config.ema_decay,
        )

        asset_id = self.data_config.asset_id
        if asset_id is None:
            raise ValueError("run5 data config has no normalization asset id")
        norm_path = args.checkpoint / "assets" / asset_id
        if not norm_path.is_dir():
            raise FileNotFoundError(f"checkpoint-local normalization stats missing: {norm_path}")
        self.norm_asset_identity = causal._directory_identity(norm_path)
        norm_stats = causal._normalize.load(norm_path)

        self.model, self.parameter_provenance = causal._load_model(
            args.checkpoint, self.train_config, args.parameter_source
        )
        if float(self.model.memory.config.eta_scale) != 0.0:
            raise ValueError(f"loaded GraphDef has eta_scale={self.model.memory.config.eta_scale}, expected 0")
        if not bool(self.model.memory.config.blank_initial_output):
            raise ValueError("loaded GraphDef does not enforce blank_initial_output")

        input_transforms = [
            transform
            for transform in self.data_config.data_transforms.inputs
            if not isinstance(transform, causal._transforms.BuildMemorySequence)
        ]
        self.input_transform = causal._transforms.compose(
            [
                *input_transforms,
                causal._transforms.Normalize(norm_stats, use_quantiles=self.data_config.use_quantile_norm),
                *self.data_config.model_transforms.inputs,
            ]
        )
        fast_snapshot_path = self.tokenizer_asset_provenance["fast_snapshot"]["resolved_path"]
        self.tokenizer = causal._tokenizer.FASTSubtaskTokenizer(
            self.train_config.model.max_token_len,
            fast_tokenizer_path=fast_snapshot_path,
        )._paligemma_tokenizer
        self.stop_token = int(self.tokenizer.encode("placeholder subtask\n")[-1])
        self._sample = causal.nnx_utils.module_jit(
            self.model.sample_with_memory,
            static_argnames=(
                "stop_token",
                "max_decode_steps",
                "num_steps",
                "zero_read",
                "allow_write",
                "write_mode",
            ),
        )

    def _observation(self, row: dict[str, Any], frame: int, prompt: str):
        decode = causal._wc.WriterContributionRunner._decode_inline_image
        transformed = self.input_transform(
            {
                "observation/image": decode(row["image"], field="image", raw_frame=frame),
                "observation/left_wrist_image": decode(
                    row["left_wrist_image"], field="left_wrist_image", raw_frame=frame
                ),
                "observation/right_wrist_image": decode(
                    row["right_wrist_image"], field="right_wrist_image", raw_frame=frame
                ),
                "observation/state": np.asarray(row["state"], dtype=np.float32),
                "prompt": prompt,
            }
        )
        transformed_state = np.asarray(transformed["state"], dtype=np.float32)
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
        return causal._model.Observation.from_dict(batched), transformed_state

    @staticmethod
    def _state_max_abs_diff(left: Any, right: Any) -> float:
        leaves = [
            jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))
            for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
        ]
        if not leaves:
            raise ValueError("memory state has no leaves")
        return float(np.asarray(jnp.max(jnp.stack(leaves))))


class HeldoutFreeDecodeVideo:
    def __init__(self, args: Args):
        self.args = args
        self.base = _Run5Runtime(args)
        if int(self.base.data_config.subtask_lookahead) != TRAIN_TARGET_LOOKAHEAD:
            raise ValueError(
                f"run5 subtask lookahead changed: expected {TRAIN_TARGET_LOOKAHEAD}, "
                f"got {self.base.data_config.subtask_lookahead}"
            )
        if not bool(self.base.data_config.subtask_from_task):
            raise ValueError("run5 data config no longer derives subtasks from task_index")
        if not bool(self.base.data_config.prompt_from_episode_meta):
            raise ValueError("run5 data config no longer derives prompts from episode metadata")
        if not bool(self.base.train_config.model.predict_with_memory):
            raise ValueError("run5 model config no longer enables sample_with_memory")
        if int(self.base.train_config.model.causal_token_len) < PRODUCTION_MAX_DECODE_STEPS:
            raise ValueError("run5 causal token buffer is shorter than the pinned free decode")
        step = self.base.parameter_provenance["train_state_step_identity"]
        if step != {
            "checkpoint_manager_step_label": CHECKPOINT_STEP,
            "internal_train_state_step": INTERNAL_TRAIN_STEP,
        }:
            raise ValueError(f"checkpoint label/internal step identity changed: {step}")
        self._first_batch_contract: dict[str, Any] | None = None

    def _load_payloads(self) -> dict[int, dict[str, Any]]:
        all_plans = causal._probe._plan_all_episodes(
            SimpleNamespace(dataset_root=self.args.dataset_root), self.base.data_config
        )
        by_episode = {int(plan.episode): plan for plan in all_plans}
        if len(by_episode) != len(all_plans):
            raise ValueError("episode plans contain duplicate episode ids")
        if not set(HELDOUT_EPISODES).issubset(by_episode):
            raise ValueError(f"failed to resolve exact heldouts {HELDOUT_EPISODES}")

        episode_metadata: dict[int, dict[str, Any]] = {}
        for record in causal._wc._read_jsonl(self.args.dataset_root / "meta" / "episodes.jsonl"):
            index = record.get("episode_index")
            length = record.get("length")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or isinstance(length, bool)
                or not isinstance(length, int)
                or length <= 0
            ):
                raise ValueError(f"invalid episode metadata record: {record}")
            if index in episode_metadata:
                raise ValueError(f"duplicate episode metadata index {index}")
            episode_metadata[index] = dict(record)

        sources = causal._wc._load_lerobot_sources(self.args.dataset_root, list(self.args.episodes))
        payloads: dict[int, dict[str, Any]] = {}
        columns = [
            "image",
            "left_wrist_image",
            "right_wrist_image",
            "state",
            "frame_index",
            "episode_index",
            "task_index",
        ]
        for episode, source in zip(self.args.episodes, sources, strict=True):
            plan = by_episode[episode]
            expected_prompt, expected_side = EXPECTED_CELLS[episode]
            if (plan.prompt, plan.side) != (expected_prompt, expected_side):
                raise ValueError(
                    f"heldout ep{episode} cell changed: expected {(expected_prompt, expected_side)}, "
                    f"got {(plan.prompt, plan.side)}"
                )
            task_names = tuple(source.task_names)
            expected_vocab = tuple(self.base.data_config.memory_subtask_vocab)
            if task_names != expected_vocab:
                raise ValueError(f"global task indices/vocabulary changed: expected {expected_vocab}, got {task_names}")
            if episode not in episode_metadata:
                raise ValueError(f"heldout episode {episode} is absent from episodes.jsonl")
            metadata_length = int(episode_metadata[episode]["length"])
            if int(plan.length) != metadata_length:
                raise ValueError(
                    f"episode {episode} plan/episodes.jsonl frame count mismatch: {plan.length} != {metadata_length}"
                )
            parquet = pq.ParquetFile(source.path)
            missing = set(columns) - set(parquet.schema_arrow.names)
            if missing:
                raise ValueError(f"episode {episode} parquet lacks columns {sorted(missing)}")
            if parquet.metadata.num_rows != metadata_length:
                raise ValueError(
                    f"episode {episode} parquet/episodes.jsonl frame count mismatch: "
                    f"{parquet.metadata.num_rows} != {metadata_length}"
                )
            rows = parquet.read(columns=columns).to_pylist()
            label_runs = _validate_rows(
                rows,
                episode=episode,
                expected_length=metadata_length,
                task_names=task_names,
            )
            payloads[episode] = {
                "plan": plan,
                "source": source,
                "source_identity": causal._file_identity(source.path),
                "rows": rows,
                "task_names": task_names,
                "label_runs": label_runs,
            }
        return payloads

    def _observations(self, entries: Sequence[tuple[dict[str, Any], int, str]]) -> Any:
        observations = [self.base._observation(row, frame, prompt)[0] for row, frame, prompt in entries]
        return jax.tree.map(lambda *values: jnp.concatenate(values, axis=0), *observations)

    def _state_diff(self, left: Any, right: Any) -> float:
        return self.base._state_max_abs_diff(left, right)

    def _infer(self, observation: Any, *, num_steps: int, zero_read: bool) -> tuple[Any, Any, dict[str, Any]]:
        batch = int(observation.state.shape[0])
        state = self.base.model.memory.init_state(batch)
        noise = jnp.zeros(
            (batch, self.base.model.action_horizon, self.base.model.action_dim),
            dtype=jnp.float32,
        )
        output = self.base._sample(
            jax.random.key(np.uint32(self.args.seed)),
            observation,
            state,
            stop_token=self.base.stop_token,
            max_decode_steps=PRODUCTION_MAX_DECODE_STEPS,
            num_steps=num_steps,
            noise=noise,
            action_prefix=None,
            forced_subtask_tokens=None,
            forced_subtask_mask=None,
            zero_read=zero_read,
            allow_write=False,
            write_mode="frozen",
        )
        jax.block_until_ready(output)
        actions, returned_state, aux = output
        contract = self._assert_inference_contract(state, returned_state, aux, batch=batch)
        if self._first_batch_contract is None:
            self._first_batch_contract = contract
        return actions, returned_state, aux

    def _assert_inference_contract(
        self, state: Any, returned_state: Any, aux: dict[str, Any], *, batch: int
    ) -> dict[str, Any]:
        writes = np.asarray(aux.get("write_occurred"))
        retrieval = np.asarray(aux.get("retrieval_norm"), dtype=np.float64)
        if writes.shape != (batch,) or np.any(writes):
            raise RuntimeError(f"write_occurred must be false for every row, got {writes}")
        if retrieval.shape != (batch,) or not np.all(np.isfinite(retrieval)):
            raise RuntimeError(f"invalid retrieval_norm: shape={retrieval.shape}")
        max_retrieval = float(np.max(np.abs(retrieval)))
        if max_retrieval > RETRIEVAL_ZERO_TOL:
            raise RuntimeError(f"fresh M0 retrieval is not zero: max norm {max_retrieval} > {RETRIEVAL_ZERO_TOL}")
        state_diff = self._state_diff(returned_state, state)
        if state_diff != 0.0:
            raise RuntimeError(f"frozen inference returned a changed memory state: max abs {state_diff}")
        return {
            "batch_size": batch,
            "write_occurred_all_false": True,
            "retrieval_norm_max_abs": max_retrieval,
            "returned_state_max_abs_difference": state_diff,
        }

    def _decode_batch(self, aux: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        tokens = np.asarray(aux["tokens"])
        masks = np.asarray(aux["token_mask"], dtype=bool)
        if tokens.shape != masks.shape or tokens.ndim != 2:
            raise ValueError(f"invalid decoded token buffers: {tokens.shape}/{masks.shape}")
        decoded = []
        statuses = []
        for index in range(tokens.shape[0]):
            live_tokens = tokens[index][masks[index]].tolist()
            if not live_tokens or len(live_tokens) > PRODUCTION_MAX_DECODE_STEPS:
                raise ValueError(f"invalid live decoded token count at batch row {index}: {len(live_tokens)}")
            last = int(live_tokens[-1])
            terminated = last in (self.base.stop_token, PALIGEMMA_EOS_TOKEN)
            truncated = len(live_tokens) == PRODUCTION_MAX_DECODE_STEPS and not terminated
            if not terminated and not truncated:
                raise RuntimeError(f"decode stopped before its limit without stop/EOS at batch row {index}")
            decoded.append(self.base.tokenizer.decode(live_tokens).strip())
            statuses.append(
                {
                    "token_count": len(live_tokens),
                    "terminated": terminated,
                    "truncated": truncated,
                    "termination_token": (
                        "stop/newline"
                        if last == self.base.stop_token
                        else "paligemma_eos"
                        if last == PALIGEMMA_EOS_TOKEN
                        else None
                    ),
                }
            )
        return decoded, tokens, masks, statuses

    @staticmethod
    def _same_tokens(
        left_tokens: np.ndarray,
        left_masks: np.ndarray,
        right_tokens: np.ndarray,
        right_masks: np.ndarray,
    ) -> bool:
        return bool(
            np.array_equal(left_masks, right_masks)
            and np.array_equal(left_tokens[left_masks], right_tokens[right_masks])
        )

    def _reserve_output(self, payloads: dict[int, dict[str, Any]]) -> Path:
        output = self.args.artifact_dir
        output.mkdir(parents=False, exist_ok=False)
        incomplete = output / "INCOMPLETE.json"
        value = {
            "schema": SCHEMA_VERSION,
            "status": "reserved; inference/rendering not complete",
            "checkpoint": str(self.args.checkpoint),
            "checkpoint_step": CHECKPOINT_STEP,
            "internal_train_step": INTERNAL_TRAIN_STEP,
            "parameter_source": self.args.parameter_source,
            "episodes": list(payloads),
            "shard": {"num_shards": self.args.num_shards, "shard_id": self.args.shard_id},
        }
        incomplete.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return incomplete

    def _common_summary(self, payloads: dict[int, dict[str, Any]]) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "checkpoint": {
                "resolved_path": str(self.args.checkpoint),
                "checkpoint_manager_step_label": CHECKPOINT_STEP,
                "internal_train_state_step": INTERNAL_TRAIN_STEP,
            },
            "config": self.args.config,
            "parameter_source": self.args.parameter_source,
            "parameter_provenance": self.base.parameter_provenance,
            "checkpoint_origin": self.base.checkpoint_origin,
            "normalization_asset_identity": self.base.norm_asset_identity,
            "tokenizer_asset_provenance": self.base.tokenizer_asset_provenance,
            "source_provenance": {
                "exact_run5_openpi_source": self.base.run5_source_provenance,
                "current_diagnostic_script": causal._file_identity(Path(__file__).resolve()),
            },
            "dataset_root": str(self.args.dataset_root),
            "dataset_metadata_identity": {
                name: causal._file_identity(self.args.dataset_root / "meta" / name)
                for name in ("info.json", "tasks.jsonl", "episodes.jsonl", "episode_prompts.json")
            },
            "source_parquet_identity": {episode: payload["source_identity"] for episode, payload in payloads.items()},
            "episodes": list(payloads),
            "heldout_cells": {
                episode: {"prompt": payload["plan"].prompt, "side": payload["plan"].side}
                for episode, payload in payloads.items()
            },
            "shard": {"num_shards": self.args.num_shards, "shard_id": self.args.shard_id},
            "inference_contract": {
                "frame_stride": 1,
                "batch_size_fixed_and_padded": self.args.batch_size,
                "memory_input": (
                    "fresh model.memory.init_state(batch_size) for every padded batch; its batch "
                    "axis gives each frame/row an independent M0 and padded outputs are discarded"
                ),
                "history_replayed": False,
                "returned_state_threaded": False,
                "allow_write": False,
                "write_mode": "frozen",
                "zero_read": False,
                "num_steps": PRODUCTION_NUM_STEPS,
                "max_decode_steps": PRODUCTION_MAX_DECODE_STEPS,
                "decoder": "free greedy sample_with_memory",
                "decode_completeness": (
                    "each row records whether a stop/EOS terminated decoding or the 10-token "
                    "production limit truncated it"
                ),
                "action_noise": "all-zero float32 tensor",
                "rng_seed": self.args.seed,
                "first_batch_assertions": self._first_batch_contract,
            },
            "ground_truth": {
                "observation_label": "task_index at frame t",
                "training_target_label": "task_index at min(t+15, T-1)",
                "lookahead_frames": TRAIN_TARGET_LOOKAHEAD,
            },
            "render": {
                "camera_source": (
                    "the three uint8 or normalized 224x224 tensors passed to the model, converted "
                    "to RGB and displayed as exact nearest-neighbour 2x pixel blocks"
                ),
                "camera_order": [label for label, _key in CAMERA_KEYS],
                "source_camera_width": IMAGE_SIZE,
                "source_camera_height": IMAGE_SIZE,
                "display_scale": DISPLAY_SCALE,
                "canvas_width": CANVAS_WIDTH,
                "canvas_height": CANVAS_HEIGHT,
                "codec": "H.264/libx264",
                "pixel_format": "yuv420p",
            },
        }

    def run_smoke(self) -> dict[str, Any]:
        started = time.monotonic()
        payloads = self._load_payloads()
        incomplete = self._reserve_output(payloads)

        candidates: list[tuple[int, int]] = []
        boundaries = {episode: _boundary_frames(payload["rows"]) for episode, payload in payloads.items()}
        max_boundaries = max(len(frames) for frames in boundaries.values())
        for index in range(max_boundaries):
            for episode in payloads:
                frames = boundaries[episode]
                if index < len(frames):
                    candidates.append((episode, frames[index]))
        selected = candidates[: self.args.batch_size]
        if not selected:
            raise RuntimeError("smoke boundary selection is empty")
        entries = [
            (payloads[episode]["rows"][frame], frame, payloads[episode]["plan"].prompt) for episode, frame in selected
        ]
        padded, live = _pad_batch(entries, self.args.batch_size)
        observation = self._observations(padded)

        _actions, _state, batch_aux = self._infer(observation, num_steps=PRODUCTION_NUM_STEPS, zero_read=False)
        _batch_decoded, batch_tokens, batch_masks, batch_statuses = self._decode_batch(batch_aux)

        singles = []
        for index in range(live):
            single_observation = jax.tree.map(lambda value, row=index: value[row : row + 1], observation)
            _a, _s, single_aux = self._infer(single_observation, num_steps=PRODUCTION_NUM_STEPS, zero_read=False)
            decoded, tokens, masks, statuses = self._decode_batch(single_aux)
            if not self._same_tokens(batch_tokens[index], batch_masks[index], tokens[0], masks[0]):
                raise RuntimeError(f"batch-vs-single decode mismatch at {selected[index]}")
            if statuses[0] != batch_statuses[index]:
                raise RuntimeError(f"batch-vs-single decode status mismatch at {selected[index]}")
            singles.append(
                {
                    "episode": selected[index][0],
                    "frame": selected[index][1],
                    "decoded": decoded[0],
                    **statuses[0],
                }
            )

        _a, _s, zero_aux = self._infer(observation, num_steps=PRODUCTION_NUM_STEPS, zero_read=True)
        _zero_decoded, zero_tokens, zero_masks, zero_statuses = self._decode_batch(zero_aux)
        if not all(
            self._same_tokens(batch_tokens[i], batch_masks[i], zero_tokens[i], zero_masks[i]) for i in range(live)
        ):
            raise RuntimeError("M0 read-vs-zero-read subtask tokens differ")
        if zero_statuses[:live] != batch_statuses[:live]:
            raise RuntimeError("M0 read-vs-zero-read decode statuses differ")

        _a, _s, ten_aux = self._infer(observation, num_steps=10, zero_read=False)
        _ten_decoded, ten_tokens, ten_masks, ten_statuses = self._decode_batch(ten_aux)
        if not all(
            self._same_tokens(batch_tokens[i], batch_masks[i], ten_tokens[i], ten_masks[i]) for i in range(live)
        ):
            raise RuntimeError("num_steps=1 vs num_steps=10 subtask tokens differ")
        if ten_statuses[:live] != batch_statuses[:live]:
            raise RuntimeError("num_steps=1 vs num_steps=10 decode statuses differ")

        report = {
            **self._common_summary(payloads),
            "mode": "smoke-only",
            "status": "pass",
            "selected_boundary_frames": singles,
            "checks": {
                "batch_vs_single_tokens_equal": True,
                "fresh_M0_read_vs_zero_read_tokens_equal": True,
                "num_steps_1_vs_10_tokens_equal": True,
                "padded_batch_live_rows": live,
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        identity = _write_json_atomic(self.args.artifact_dir / "smoke.json", report)
        _write_complete_atomic(self.args.artifact_dir, {"smoke.json": identity})
        incomplete.unlink()
        print(json.dumps(report["checks"], indent=2), flush=True)
        return report

    def _episode_video(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        plan = payload["plan"]
        episode = int(plan.episode)
        source = payload["source"]
        fps = float(source.control_hz)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"episode {episode} has invalid source fps {fps!r}")
        rows = payload["rows"]
        task_names = payload["task_names"]
        video_name = f"episode_{episode:06d}_free_decode.mp4"
        jsonl_name = f"episode_{episode:06d}_frames.jsonl"
        video = _AtomicMp4Writer(self.args.artifact_dir / video_name, fps)
        jsonl_path = self.args.artifact_dir / jsonl_name
        jsonl_temporary = jsonl_path.with_name(f".{jsonl_path.name}.tmp")
        if jsonl_path.exists() or jsonl_temporary.exists():
            video.abort()
            raise FileExistsError(f"refusing to overwrite frame records: {jsonl_path}")

        counts: dict[str, int] = {}
        decode_status_counts = {"terminated": 0, "truncated": 0}
        decoded_runs: list[dict[str, Any]] = []
        previous_decoded: str | None = None
        run_start = 0
        started = time.monotonic()
        try:
            with jsonl_temporary.open("w", encoding="utf-8") as jsonl:
                for batch_start in range(0, len(rows), self.args.batch_size):
                    live_rows = rows[batch_start : batch_start + self.args.batch_size]
                    entries = [(row, int(row["frame_index"]), plan.prompt) for row in live_rows]
                    padded, live = _pad_batch(entries, self.args.batch_size)
                    observation = self._observations(padded)
                    _actions, _returned, aux = self._infer(observation, num_steps=PRODUCTION_NUM_STEPS, zero_read=False)
                    decoded, tokens, masks, statuses = self._decode_batch(aux)
                    for batch_index in range(live):
                        row = live_rows[batch_index]
                        frame = int(row["frame_index"])
                        prediction = decoded[batch_index]
                        decode_status = statuses[batch_index]
                        target_frame, target_task = _training_target(rows, frame, TRAIN_TARGET_LOOKAHEAD)
                        gt_now = task_names[int(row["task_index"])]
                        gt_target = task_names[target_task]
                        images = _processed_images(observation, batch_index)
                        rendered = _render_frame(
                            images,
                            episode=episode,
                            frame=frame,
                            fps=fps,
                            prompt=plan.prompt,
                            decoded=prediction,
                            decoded_token_count=int(decode_status["token_count"]),
                            decoded_terminated=bool(decode_status["terminated"]),
                            decoded_truncated=bool(decode_status["truncated"]),
                            gt_now=gt_now,
                            target_frame=target_frame,
                            gt_train_target=gt_target,
                            parameter_source=self.args.parameter_source,
                            config=self.args.config,
                        )
                        video.write(rendered)
                        record = {
                            "episode": episode,
                            "frame": frame,
                            "time_seconds": frame / fps,
                            "fps": fps,
                            "prompt": plan.prompt,
                            "heldout_side": plan.side,
                            "decoded_subtask": prediction,
                            "decoded_side": causal._phase_side(prediction),
                            "decoded_token_ids": tokens[batch_index][masks[batch_index]].tolist(),
                            "decoded_token_count": decode_status["token_count"],
                            "decoded_terminated": decode_status["terminated"],
                            "decoded_truncated": decode_status["truncated"],
                            "decode_termination_token": decode_status["termination_token"],
                            "gt_now": gt_now,
                            "gt_now_task_index": int(row["task_index"]),
                            "gt_training_target_frame": target_frame,
                            "gt_training_target": gt_target,
                            "gt_training_target_task_index": target_task,
                            "processed_camera_sha256": {
                                key: causal._array_sha256(image) for key, image in images.items()
                            },
                            "memory": {
                                "fresh_independent_M0_for_this_frame": True,
                                "fixed_padded_batch_is_throughput_only": True,
                                "history_replayed": False,
                                "writes_enabled": False,
                                "write_mode": "frozen",
                                "zero_read": False,
                                "returned_state_threaded": False,
                                "retrieval_norm": float(np.asarray(aux["retrieval_norm"])[batch_index]),
                            },
                            "inference": {
                                "parameter_source": self.args.parameter_source,
                                "config": self.args.config,
                                "checkpoint_step": CHECKPOINT_STEP,
                                "num_steps": PRODUCTION_NUM_STEPS,
                                "max_decode_steps": PRODUCTION_MAX_DECODE_STEPS,
                                "action_noise": "zeros",
                            },
                        }
                        jsonl.write(json.dumps(causal._strict_json(record), sort_keys=True, allow_nan=False) + "\n")
                        counts[prediction] = counts.get(prediction, 0) + 1
                        decode_status_counts["terminated" if decode_status["terminated"] else "truncated"] += 1
                        if previous_decoded is None:
                            previous_decoded = prediction
                            run_start = frame
                        elif prediction != previous_decoded:
                            decoded_runs.append(
                                {"start": run_start, "end": frame - 1, "decoded_subtask": previous_decoded}
                            )
                            run_start = frame
                            previous_decoded = prediction
                    print(
                        f"[ep{episode}] frames {batch_start}-{batch_start + live - 1}/{len(rows) - 1}",
                        flush=True,
                    )
                if previous_decoded is not None:
                    decoded_runs.append({"start": run_start, "end": len(rows) - 1, "decoded_subtask": previous_decoded})
                jsonl.flush()
                os.fsync(jsonl.fileno())
            os.replace(jsonl_temporary, jsonl_path)
            video_identity = video.close()
        except BaseException:
            video.abort()
            raise

        jsonl_identity = causal._file_identity(jsonl_path)
        if video.frames != len(rows):
            raise RuntimeError(f"episode {episode} rendered frame count mismatch: {video.frames} != {len(rows)}")
        episode_summary = {
            "episode": episode,
            "prompt": plan.prompt,
            "heldout_side": plan.side,
            "source_parquet": str(source.path),
            "fps": fps,
            "frame_count": len(rows),
            "expected_frames": [0, len(rows) - 1],
            "label_runs": payload["label_runs"],
            "decoded_counts": counts,
            "decode_status_counts": decode_status_counts,
            "decoded_runs": decoded_runs,
            "video": video_name,
            "frame_records": jsonl_name,
            "elapsed_seconds": time.monotonic() - started,
        }
        return episode_summary, {video_name: video_identity, jsonl_name: jsonl_identity}

    def run(self) -> dict[str, Any]:
        if self.args.smoke_only:
            return self.run_smoke()
        started = time.monotonic()
        payloads = self._load_payloads()
        incomplete = self._reserve_output(payloads)
        episodes = []
        artifacts: dict[str, dict[str, Any]] = {}
        for episode in self.args.episodes:
            episode_summary, episode_artifacts = self._episode_video(payloads[episode])
            episodes.append(episode_summary)
            artifacts.update(episode_artifacts)

        summary = {
            **self._common_summary(payloads),
            "mode": "production",
            "status": "complete",
            "episode_results": episodes,
            "total_frames": sum(item["frame_count"] for item in episodes),
            "decode_status_counts": {
                status: sum(item["decode_status_counts"][status] for item in episodes)
                for status in ("terminated", "truncated")
            },
            "elapsed_seconds": time.monotonic() - started,
            "artifacts": artifacts,
        }
        summary_identity = _write_json_atomic(self.args.artifact_dir / "summary.json", summary)
        artifacts["summary.json"] = summary_identity
        _write_complete_atomic(self.args.artifact_dir, artifacts)
        incomplete.unlink()
        print(
            json.dumps(
                {
                    "status": "complete",
                    "episodes": list(self.args.episodes),
                    "total_frames": summary["total_frames"],
                    "output": str(self.args.artifact_dir),
                },
                indent=2,
            ),
            flush=True,
        )
        return summary


def main(argv: list[str] | None = None) -> None:
    HeldoutFreeDecodeVideo(_parse_args(argv)).run()


if __name__ == "__main__":
    main()
