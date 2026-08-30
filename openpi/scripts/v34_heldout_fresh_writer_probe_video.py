"""Render every-frame fresh-writer-probe predictions for the four run5 heldouts.

The classifier is not the checkpoint's accumulated online head.  It is reconstructed exactly
from a completed :mod:`v34_fixed_writer_probe_eval` artifact for the same checkpoint: one
deterministic zero-initialized scaler/logistic head is fitted on the 56 nonheldout, episode-mean
evidence writer features.  The four heldout episodes are excluded from the scaler, fit, and all
model-selection decisions.

Every raw heldout frame is then run through the exact run5 preprocessing and writer interface
with a fresh M0.  No write is performed and no returned state is threaded.  Each 30 fps video
shows two deliberately different readings of that fixed head:

* ``CURRENT W_t`` scores the current per-frame normalized writer feature.  This changes the
  probe's episode-mean aggregation unit; frames outside evidence are additionally phase-OOD.
* ``EVIDENCE PREFIX`` causally averages only evidence features observed up through the current
  frame.  It is unavailable before evidence, updates only while ``task_index == 4``, and freezes
  afterward.  Its final score must reproduce the fixed-probe artifact's full-evidence heldout
  score.

Run with the immutable run5 launch source first on ``PYTHONPATH``.  The completed fixed-probe
artifact is a required, hash-validated prerequisite::

    V34_RUN5_SOURCE_ROOT=/tmp/v34-run5-source \
    PYTHONPATH=/tmp/v34-run5-source/src \
    .venv/bin/python -u scripts/v34_heldout_fresh_writer_probe_video.py \
      --checkpoint diagnostic_checkpoints/v34_run5_eta0_resume_copies/11000 \
      --dataset-root /iris/u/kewalk/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask \
      --probe-artifact-dir diagnostic_outputs/v34_fixed_writer_probe/full_17099350/11000/raw \
      --output-dir diagnostic_outputs/v34_heldout_fresh_writer_probe/full_17099350/11000 \
      --config pi05_yam_mem_v34_run5_eta0 --parameter-source raw

``--smoke-only`` renders five ep15 frames straddling evidence onset.  Full mode renders every
frame of episodes 15/29/44/59.  A private sibling directory is published only after the NPZ
schemas, final probe-score reproduction, H.264 metadata, exact frame counts, durations, and a
full ffmpeg decode all pass.
"""

# ruff: noqa: SLF001, I001 - pyarrow must precede the audited OpenPI/JAX import stack.
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
from typing import Any, Literal

# Import the fixed evaluator before OpenPI/JAX.  It enforces the immutable run5 launch source.
import v34_fixed_writer_probe_eval as fixed
import v34_heldout_writer_attention_video as attention_video

import cv2
import jax
import numpy as np


SCHEMA_VERSION = "openpi.v34.heldout_fresh_writer_probe_video.v1"
RUN5_CONFIG = fixed.RUN5_CONFIG
HELDOUT_EPISODES = fixed.HELDOUT_EPISODES
EXPECTED_CELLS = fixed.EXPECTED_HELDOUT_CELLS
EXPECTED_FRAME_COUNTS = attention_video.EXPECTED_FRAME_COUNTS
SOURCE_FPS = attention_video.SOURCE_FPS
EVIDENCE_TASK_INDEX = fixed.EXPECTED_EVIDENCE_TASK_INDEX
EVIDENCE_LABEL = fixed.EXPECTED_EVIDENCE_LABEL
EXPECTED_TRAIN_EPISODES = fixed.EXPECTED_TRAIN_EPISODES
SMOKE_EPISODE = 15
SMOKE_CONTEXT_FRAMES = 2
SMOKE_FRAME_COUNT = 2 * SMOKE_CONTEXT_FRAMES + 1
MODEL_IMAGE_SIZE = 224
DISPLAY_SCALE = 3
DISPLAY_SIZE = MODEL_IMAGE_SIZE * DISPLAY_SCALE
HEADER_HEIGHT = 150
CANVAS_WIDTH = DISPLAY_SIZE
CANVAS_HEIGHT = HEADER_HEIGHT + DISPLAY_SIZE
FINAL_FEATURE_ATOL = 2e-6
FINAL_LOGIT_ATOL = 2e-5
INSTANT_SCORE_ATOL = 1e-9

IMAGE_COLUMNS = ("image", "left_wrist_image", "right_wrist_image")
ROW_COLUMNS = (*IMAGE_COLUMNS, "state", "frame_index", "episode_index", "task_index")


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint: Path
    dataset_root: Path
    probe_artifact_dir: Path
    output_dir: Path
    config: str
    parameter_source: Literal["raw", "ema"]
    batch_size: int = 8
    smoke_only: bool = False

    def __post_init__(self) -> None:
        for name in ("checkpoint", "dataset_root", "probe_artifact_dir", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        if not self.checkpoint.name.isdigit() or int(self.checkpoint.name) < 0:
            raise ValueError(f"checkpoint must end in a nonnegative numeric step, got {self.checkpoint.name!r}")
        if self.config != RUN5_CONFIG:
            raise ValueError(f"fresh writer-probe video is pinned to --config {RUN5_CONFIG!r}")
        if self.parameter_source not in ("raw", "ema"):
            raise ValueError("--parameter-source must be raw or ema")
        if isinstance(self.batch_size, bool) or not 1 <= self.batch_size <= 64:
            raise ValueError("--batch-size must lie in [1, 64]")
        if fixed.causal._is_relative_to(self.output_dir, self.checkpoint):
            raise ValueError("diagnostic output must be outside the checkpoint directory")
        if fixed.causal._is_relative_to(self.output_dir, self.probe_artifact_dir):
            raise ValueError("video output must be outside the fixed-probe prerequisite artifact")

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


@dataclasses.dataclass(frozen=True)
class ProbeBundle:
    model: fixed.ProbeModel
    episode_writer_evidence: np.ndarray
    episode_labels: np.ndarray
    evidence_counts: np.ndarray
    train_indices: tuple[int, ...]
    heldout_report_logits: dict[int, float]
    report: dict[str, Any]
    artifact_identities: dict[str, dict[str, Any]]


def _parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--probe-artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parameter-source", choices=("raw", "ema"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--smoke-only", action="store_true")
    return Args(**vars(parser.parse_args(argv)))


def _manifest_identities(
    artifact_dir: Path, *, expected_names: set[str]
) -> dict[str, dict[str, Any]]:
    """Hash-check an exact, path-safe COMPLETE manifest."""
    artifact_dir = Path(artifact_dir).resolve()
    if not artifact_dir.is_dir():
        raise FileNotFoundError(f"fixed-probe artifact directory not found: {artifact_dir}")
    if (artifact_dir / "INCOMPLETE.json").exists():
        raise RuntimeError(f"fixed-probe artifact still has INCOMPLETE marker: {artifact_dir}")
    complete = artifact_dir / "COMPLETE"
    if not complete.is_file():
        raise FileNotFoundError(f"fixed-probe artifact lacks COMPLETE: {complete}")
    entries: dict[str, str] = {}
    for line in complete.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not name
            or Path(name).name != name
            or name in entries
        ):
            raise ValueError(f"unsafe or malformed COMPLETE line: {line!r}")
        entries[name] = digest
    if set(entries) != expected_names:
        raise ValueError(
            f"fixed-probe COMPLETE schema changed: expected {sorted(expected_names)}, got {sorted(entries)}"
        )
    identities = {}
    for name, expected in entries.items():
        path = artifact_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"fixed-probe COMPLETE references missing artifact: {path}")
        identity = fixed.causal._file_identity(path)
        if identity["sha256"] != expected:
            raise RuntimeError(
                f"fixed-probe artifact hash mismatch for {name}: expected {expected}, got {identity['sha256']}"
            )
        identities[name] = identity
    identities["COMPLETE"] = fixed.causal._file_identity(complete)
    return identities


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"fixed-probe report field {name} must be an object")
    return value


def _load_probe_bundle(
    args: Args,
    runtime: fixed._Run5Runtime,
    specs: Sequence[fixed.EpisodeSpec],
    current_dataset_provenance: Mapping[str, Any],
) -> ProbeBundle:
    identities = _manifest_identities(
        args.probe_artifact_dir, expected_names={"features.npz", "report.json"}
    )
    report = json.loads((args.probe_artifact_dir / "report.json").read_text(encoding="utf-8"))
    expected_scalars = {
        "schema_version": fixed.SCHEMA_VERSION,
        "status": "complete",
        "checkpoint_step_label": args.checkpoint_step,
        "config": args.config,
        "parameter_source": args.parameter_source,
    }
    for name, expected in expected_scalars.items():
        if report.get(name) != expected:
            raise ValueError(
                f"fixed-probe report {name} mismatch: expected {expected!r}, got {report.get(name)!r}"
            )
    design = _require_mapping(report.get("design"), "design")
    if design.get("episodes") != 60 or tuple(design.get("heldout_episodes", ())) != tuple(HELDOUT_EPISODES):
        raise ValueError("fixed-probe report is not the exact 60-episode/four-heldout design")
    extraction = _require_mapping(report.get("extraction_contract"), "extraction_contract")
    extraction_required = {
        "mode": "full",
        "fresh_independent_m0_per_frame_row": True,
        "writes_during_production_extraction": False,
        "returned_state_threaded": False,
        "feature_formula": "float32 L2(mean_slots(tokens)), epsilon=1e-12",
        "episode_formula": "mean(per-frame feature), no second normalization",
        "evidence_selector": "unshifted parquet task_index == 4",
    }
    for name, expected in extraction_required.items():
        if extraction.get(name) != expected:
            raise ValueError(f"fixed-probe extraction contract changed at {name!r}")
    protocol = _require_mapping(report.get("probe_protocol"), "probe_protocol")
    protocol_required = {
        "fit_unit": "one equal-weight episode feature",
        "feature_aggregation": "mean of per-frame features; no second normalization",
        "classifier": "binary logistic regression, right=1, left=0",
        "fresh_initialization": "all weights and bias exactly zero for every fold/source/checkpoint",
        "standardization": "evidence-fit episodes only; reused unchanged for matched approach",
        "l2": fixed.PROBE_L2,
        "steps": fixed.PROBE_STEPS,
        "learning_rate": fixed.PROBE_LR,
        "heldout_excluded_from_every_fit": True,
        "train_episode_count": EXPECTED_TRAIN_EPISODES,
    }
    for name, expected in protocol_required.items():
        if protocol.get(name) != expected:
            raise ValueError(f"fixed-probe fitting contract changed at {name!r}")

    artifact_parameter = _require_mapping(report.get("parameter_provenance"), "parameter_provenance")
    runtime_parameter = runtime.parameter_provenance
    for name in ("parameter_source", "train_state_step_identity", "restored_parameter_tree_identity"):
        if artifact_parameter.get(name) != runtime_parameter.get(name):
            raise ValueError(f"fixed-probe artifact and video runtime parameter identity differ at {name}")
    artifact_dataset = _require_mapping(report.get("dataset_provenance"), "dataset_provenance")
    if Path(str(artifact_dataset.get("dataset_root", ""))).resolve() != args.dataset_root:
        raise ValueError("fixed-probe artifact dataset root differs from video dataset root")
    metadata = _require_mapping(
        artifact_dataset.get("metadata_file_identities"), "dataset_provenance.metadata_file_identities"
    )
    for name in ("info.json", "tasks.jsonl", "episodes.jsonl", "episode_prompts.json"):
        current = fixed.causal._file_identity(args.dataset_root / "meta" / name)
        if metadata.get(name) != current:
            raise ValueError(f"fixed-probe artifact dataset metadata changed for {name}")
    for name in (
        "all_frame_episode_task_protocol_sha256",
        "parquet_storage_metadata",
    ):
        if artifact_dataset.get(name) != current_dataset_provenance.get(name):
            raise ValueError(
                f"fixed-probe artifact and current dataset provenance differ at {name}"
            )

    artifact_source = _require_mapping(report.get("source_provenance"), "source_provenance")
    runtime_source = runtime.source_provenance
    artifact_snapshot = _require_mapping(
        artifact_source.get("launch_snapshot_identity"),
        "source_provenance.launch_snapshot_identity",
    )
    runtime_snapshot = _require_mapping(
        runtime_source.get("launch_snapshot_identity"),
        "runtime.source_provenance.launch_snapshot_identity",
    )
    if artifact_snapshot.get("sha256") != runtime_snapshot.get("sha256"):
        raise ValueError("fixed-probe artifact and video runtime use different run5 source snapshots")

    artifact_norm = _require_mapping(
        report.get("normalization_asset_provenance"), "normalization_asset_provenance"
    )
    runtime_norm = runtime.norm_provenance
    if artifact_norm.get("files") != runtime_norm.get("files"):
        raise ValueError("fixed-probe artifact and video runtime normalization asset identities differ")

    artifact_tokenizer = _require_mapping(
        report.get("tokenizer_asset_provenance"), "tokenizer_asset_provenance"
    )
    runtime_tokenizer = runtime.tokenizer_provenance
    tokenizer_bindings = {
        "all_expected_hashes_match": (
            artifact_tokenizer.get("all_expected_hashes_match"),
            runtime_tokenizer.get("all_expected_hashes_match"),
        ),
        "fast_expected_commit": (
            artifact_tokenizer.get("fast_expected_commit"),
            runtime_tokenizer.get("fast_expected_commit"),
        ),
        "fast_snapshot.files": (
            _require_mapping(
                artifact_tokenizer.get("fast_snapshot"),
                "tokenizer_asset_provenance.fast_snapshot",
            ).get("files"),
            _require_mapping(
                runtime_tokenizer.get("fast_snapshot"),
                "runtime.tokenizer_asset_provenance.fast_snapshot",
            ).get("files"),
        ),
        "paligemma_model identity": (
            {
                name: _require_mapping(
                    artifact_tokenizer.get("paligemma_model"),
                    "tokenizer_asset_provenance.paligemma_model",
                ).get(name)
                for name in ("bytes", "sha256")
            },
            {
                name: _require_mapping(
                    runtime_tokenizer.get("paligemma_model"),
                    "runtime.tokenizer_asset_provenance.paligemma_model",
                ).get(name)
                for name in ("bytes", "sha256")
            },
        ),
    }
    for name, (artifact_value, runtime_value) in tokenizer_bindings.items():
        if artifact_value != runtime_value:
            raise ValueError(
                f"fixed-probe artifact and video runtime tokenizer identities differ at {name}"
            )

    with np.load(args.probe_artifact_dir / "features.npz", allow_pickle=False) as arrays:
        required = {
            "episode_ids",
            "episode_labels",
            "episode_heldout",
            "episode_evidence_counts",
            "episode_writer_evidence",
        }
        missing = required - set(arrays.files)
        if missing:
            raise ValueError(f"fixed-probe features.npz lacks {sorted(missing)}")
        episode_ids = np.asarray(arrays["episode_ids"], dtype=np.int64)
        labels = np.asarray(arrays["episode_labels"], dtype=np.int64)
        heldout = np.asarray(arrays["episode_heldout"], dtype=bool)
        evidence_counts = np.asarray(arrays["episode_evidence_counts"], dtype=np.int64)
        writer = np.asarray(arrays["episode_writer_evidence"], dtype=np.float32)
    expected_ids = np.arange(60, dtype=np.int64)
    expected_labels = np.asarray([spec.label for spec in specs], dtype=np.int64)
    expected_heldout = np.asarray([spec.heldout for spec in specs], dtype=bool)
    expected_counts = np.asarray([len(spec.evidence_frames) for spec in specs], dtype=np.int64)
    if not np.array_equal(episode_ids, expected_ids):
        raise ValueError("fixed-probe episode ids are not exactly 0..59")
    if not np.array_equal(labels, expected_labels) or not np.array_equal(heldout, expected_heldout):
        raise ValueError("fixed-probe labels/heldout mask differ from current exact design")
    if not np.array_equal(evidence_counts, expected_counts):
        raise ValueError("fixed-probe evidence counts differ from current corrected task labels")
    if writer.ndim != 2 or writer.shape[0] != 60 or writer.shape[1] < 1 or not np.all(np.isfinite(writer)):
        raise ValueError(f"fixed-probe episode writer features are invalid: {writer.shape}")

    train_indices, _folds = fixed._validate_design(specs)
    if len(train_indices) != EXPECTED_TRAIN_EPISODES or set(train_indices) & set(HELDOUT_EPISODES):
        raise AssertionError("heldout episodes entered fresh probe fit")
    model = fixed._fit_probe(writer[np.asarray(train_indices)], labels[np.asarray(train_indices)])

    stream = _require_mapping(report.get("fresh_probe_streams"), "fresh_probe_streams")
    writer_report = _require_mapping(stream.get("writer"), "fresh_probe_streams.writer")
    heldout_report = _require_mapping(
        writer_report.get("fit_all_56_test_exact_heldout_4"),
        "fresh_probe_streams.writer.fit_all_56_test_exact_heldout_4",
    )
    rows = heldout_report.get("episodes")
    if not isinstance(rows, list) or len(rows) != len(HELDOUT_EPISODES):
        raise ValueError("fixed-probe report lacks the four heldout writer score rows")
    report_logits: dict[int, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("fixed-probe heldout score row is not an object")
        episode = row.get("episode")
        score = row.get("evidence_logit_right_minus_left")
        if (
            isinstance(episode, bool)
            or not isinstance(episode, int)
            or episode not in HELDOUT_EPISODES
            or episode in report_logits
            or isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(float(score))
        ):
            raise ValueError(f"invalid fixed-probe heldout score row: {row}")
        report_logits[episode] = float(score)
    if set(report_logits) != set(HELDOUT_EPISODES):
        raise ValueError("fixed-probe heldout score coverage changed")
    recomputed = model.scores(writer[np.asarray(HELDOUT_EPISODES)])
    for episode, score in zip(HELDOUT_EPISODES, recomputed, strict=True):
        if not math.isclose(float(score), report_logits[episode], rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError(f"fresh-probe refit failed to reproduce report score for ep{episode}")

    return ProbeBundle(
        model=model,
        episode_writer_evidence=writer,
        episode_labels=labels,
        evidence_counts=evidence_counts,
        train_indices=train_indices,
        heldout_report_logits=report_logits,
        report=report,
        artifact_identities=identities,
    )


def _evidence_prefix_scores(
    features: np.ndarray,
    is_evidence: np.ndarray,
    model: fixed.ProbeModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Causal prefix scores: N/A before evidence, update on evidence, freeze afterward."""
    values = np.asarray(features, dtype=np.float32)
    mask = np.asarray(is_evidence, dtype=bool)
    if values.ndim != 2 or mask.shape != (len(values),) or not np.all(np.isfinite(values)):
        raise ValueError("invalid frame features/evidence mask")
    counts = np.zeros(len(values), dtype=np.int32)
    scores = np.full(len(values), np.nan, dtype=np.float64)
    running = np.zeros(values.shape[1], dtype=np.float64)
    count = 0
    for index, feature in enumerate(values):
        if mask[index]:
            running += feature.astype(np.float64)
            count += 1
        counts[index] = count
        if count:
            scores[index] = float(model.scores((running / count)[None, :])[0])
    return scores, counts


def _prediction(score: float) -> str:
    if not math.isfinite(score):
        raise ValueError("prediction score must be finite")
    return fixed.LABEL_TO_SIDE[int(score > 0.0)]


def _phase_category(spec: fixed.EpisodeSpec, frame: int, task_index: int) -> str:
    """Return a short, direct phase label for manual video inspection."""
    if spec.approach_frames[0] <= frame <= spec.approach_frames[-1]:
        return "MATCHED PRE-EVIDENCE"
    if task_index == EVIDENCE_TASK_INDEX:
        return "EVIDENCE"
    if task_index == 3:
        return "RESET"
    if task_index in (1, 6):
        return "WAIT"
    if task_index in (2, 5):
        return "EXECUTE"
    if task_index == 0:
        return "OPEN / PRE-EVIDENCE"
    raise ValueError(f"unknown run5 task index {task_index}")


def _safe_text(value: str) -> str:
    return value.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def _put_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int] = (235, 235, 235),
    scale: float = 0.43,
    thickness: int = 1,
) -> None:
    cv2.putText(
        canvas,
        _safe_text(text),
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _score_text(prefix: str, score: float | None, *, suffix: str = "") -> str:
    if score is None or not math.isfinite(score):
        return f"{prefix}: N/A{suffix}"
    side = _prediction(score).upper()
    return f"{prefix}: {side}   signed R-L score={score:+.3f}{suffix}"


def _render_frame(
    top_image: np.ndarray,
    *,
    episode: int,
    frame: int,
    full_frame_count: int,
    fps: float,
    prompt: str,
    truth_side: str,
    gt_subtask: str,
    phase_category: str,
    instantaneous_score: float,
    cumulative_score: float | None,
    evidence_seen: int,
    evidence_total: int,
    is_evidence: bool,
    checkpoint_step: int,
    parameter_source: str,
) -> np.ndarray:
    image = np.asarray(top_image)
    if image.shape != (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError(f"top image must be exact uint8 224x224 RGB, got {image.shape}/{image.dtype}")
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    enlarged = cv2.resize(image, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
    canvas[HEADER_HEIGHT:] = enlarged

    phase_note = "evidence" if is_evidence else "outside evidence"
    if evidence_seen == 0:
        prefix_note = "no evidence observed"
    elif evidence_seen < evidence_total:
        prefix_note = f"causal prefix {evidence_seen}/{evidence_total}"
    else:
        prefix_note = f"final/frozen {evidence_seen}/{evidence_total}"
    _put_text(
        canvas,
        f"ep{episode:03d}  frame {frame:04d}/{full_frame_count - 1:04d}  t={frame / fps:6.2f}s  "
        f"ckpt={checkpoint_step} {parameter_source.upper()}",
        (8, 19),
    )
    _put_text(
        canvas,
        f"PHASE={phase_category}  |  GT side={truth_side.upper()}",
        (8, 42),
        color=(170, 225, 255),
    )
    _put_text(
        canvas,
        f"GT subtask={gt_subtask}  |  prompt={prompt}",
        (8, 65),
        color=(245, 245, 245),
        scale=0.39,
    )
    instant_color = (255, 213, 85) if not is_evidence else (110, 245, 150)
    _put_text(
        canvas,
        _score_text("PROBE THIS FRAME", instantaneous_score, suffix=f"  [{phase_note}]"),
        (8, 91),
        color=instant_color,
        scale=0.40,
    )
    _put_text(
        canvas,
        _score_text("PROBE FROM EVIDENCE SO FAR", cumulative_score, suffix=f"  [{prefix_note}]"),
        (8, 117),
        color=(110, 215, 255),
        scale=0.40,
    )
    _put_text(
        canvas,
        "Fit: 56 non-heldout demos, one full-evidence average/demo. Top shown; probe also uses wrists/state/instruction.",
        (8, 140),
        color=(175, 175, 175),
        scale=0.30,
    )

    # A compact badge is deliberately placed on the image itself so the prediction remains
    # readable when a video player crops the diagnostic header.
    badge = canvas[HEADER_HEIGHT : HEADER_HEIGHT + 82].copy()
    dark = np.zeros_like(badge)
    canvas[HEADER_HEIGHT : HEADER_HEIGHT + 82] = cv2.addWeighted(badge, 0.28, dark, 0.72, 0.0)
    _put_text(
        canvas,
        _score_text("THIS FRAME", instantaneous_score),
        (12, HEADER_HEIGHT + 29),
        color=instant_color,
        scale=0.55,
        thickness=2,
    )
    _put_text(
        canvas,
        _score_text("EVIDENCE SO FAR", cumulative_score),
        (12, HEADER_HEIGHT + 63),
        color=(110, 215, 255),
        scale=0.55,
        thickness=2,
    )
    border_color = (105, 230, 140) if is_evidence else (255, 200, 70)
    cv2.rectangle(
        canvas,
        (1, HEADER_HEIGHT + 1),
        (CANVAS_WIDTH - 2, CANVAS_HEIGHT - 2),
        border_color,
        3,
    )
    return canvas


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


class HeldoutFreshWriterProbeVideo:
    def __init__(self, args: Args):
        self.args = args
        if args.artifact_dir.exists():
            raise FileExistsError(f"refusing to overwrite artifact directory: {args.artifact_dir}")
        runtime_preflight = (
            args.output_dir
            / f".{args.parameter_source}.runtime-preflight.{os.getpid()}"
        )
        runtime_args = _RuntimeArgs(
            checkpoint=args.checkpoint,
            dataset_root=args.dataset_root,
            config=args.config,
            parameter_source=args.parameter_source,
            artifact_dir=runtime_preflight,
        )
        self.runtime = fixed._Run5Runtime(runtime_args)
        self.specs, self.dataset_provenance = fixed._load_specs(self.runtime, runtime_args)
        self.probe = _load_probe_bundle(
            args,
            self.runtime,
            self.specs,
            self.dataset_provenance,
        )
        step_identity = self.runtime.parameter_provenance.get("train_state_step_identity")
        expected_step = {
            "checkpoint_manager_step_label": args.checkpoint_step,
            "internal_train_state_step": args.checkpoint_step + 1,
        }
        if step_identity != expected_step:
            raise ValueError(f"checkpoint label/internal-step gate failed: {step_identity} != {expected_step}")

    def _load_episode(self, episode: int) -> dict[str, Any]:
        spec = self.specs[episode]
        if not spec.heldout or spec.cell != tuple(EXPECTED_CELLS[episode]):
            raise ValueError(f"heldout episode {episode} cell/design changed")
        if spec.length != EXPECTED_FRAME_COUNTS[episode]:
            raise ValueError(f"heldout episode {episode} length changed from {EXPECTED_FRAME_COUNTS[episode]}")
        sources = fixed.causal._wc._load_lerobot_sources(self.args.dataset_root, [episode])
        if len(sources) != 1:
            raise ValueError(f"expected one source for heldout episode {episode}, got {len(sources)}")
        source = sources[0]
        if Path(source.path).resolve() != spec.parquet:
            raise ValueError(f"episode {episode} source path differs from fixed-probe design")
        task_names = tuple(source.task_names)
        if task_names != tuple(self.runtime.data_config.memory_subtask_vocab):
            raise ValueError("dataset task vocabulary/order differs from exact run5 config")
        fps = float(source.control_hz)
        if not math.isclose(fps, SOURCE_FPS, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"episode {episode} fps changed: expected {SOURCE_FPS}, got {fps}")
        parquet = pq.ParquetFile(source.path)
        missing = set(ROW_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"episode {episode} parquet lacks {sorted(missing)}")
        if parquet.metadata.num_rows != spec.length:
            raise ValueError(f"episode {episode} parquet length differs from fixed-probe design")
        rows = parquet.read(columns=list(ROW_COLUMNS)).to_pylist()
        label_runs = fixed.decode_video._validate_rows(
            rows,
            episode=episode,
            expected_length=spec.length,
            task_names=task_names,
        )
        selected_rows = rows
        if self.args.smoke_only:
            onset = spec.evidence_frames[0]
            start = onset - SMOKE_CONTEXT_FRAMES
            end = onset + SMOKE_CONTEXT_FRAMES + 1
            selected_rows = rows[start:end]
            if len(selected_rows) != SMOKE_FRAME_COUNT:
                raise ValueError("smoke evidence-onset window is incomplete")
        stat = Path(source.path).stat()
        return {
            "spec": spec,
            "source": source,
            "rows": selected_rows,
            "task_names": task_names,
            "label_runs": label_runs,
            "storage": {
                "resolved_path": str(Path(source.path).resolve()),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "parquet_rows": parquet.metadata.num_rows,
            },
        }

    def _infer_episode(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        spec: fixed.EpisodeSpec = payload["spec"]
        rows = payload["rows"]
        task_names = payload["task_names"]
        frame_features = []
        top_images = []
        task_indices = []
        frame_indices = []
        input_digest = hashlib.sha256()
        model_batches = 0
        for start in range(0, len(rows), self.args.batch_size):
            live_rows = rows[start : start + self.args.batch_size]
            padded_rows, live = fixed._pad_batch(live_rows, self.args.batch_size)
            observations = []
            for row in padded_rows:
                frame = int(row["frame_index"])
                observation, _state = self.runtime.observation(row, frame, spec.prompt)
                observations.append(observation)
            batched = self.runtime.batch_observations(observations)
            fresh_m0 = self.runtime.model.memory.init_state(self.args.batch_size)
            output = self.runtime._interface(batched, fresh_m0)
            jax.block_until_ready(output)
            pooled = fixed._online_pool(output["write_tokens"])
            for index, row in enumerate(padded_rows[:live]):
                frame = int(row["frame_index"])
                task = int(row["task_index"])
                fixed._update_selected_input_hash(
                    input_digest,
                    spec=spec,
                    phase="all-frame-probe-video",
                    row=row,
                )
                frame_indices.append(frame)
                task_indices.append(task)
                frame_features.append(pooled[index])
                top_images.append(fixed.decode_video._processed_images(batched, index)["base_0_rgb"])
            model_batches += 1
            processed = min(start + live, len(rows))
            if processed == len(rows) or processed % 100 < self.args.batch_size:
                print(
                    f"ep{spec.episode}: writer features {processed}/{len(rows)} rendered-source frames",
                    flush=True,
                )

        features = np.stack(frame_features).astype(np.float32)
        frames = np.asarray(frame_indices, dtype=np.int32)
        tasks = np.asarray(task_indices, dtype=np.int16)
        is_evidence = tasks == EVIDENCE_TASK_INDEX
        instantaneous = self.probe.model.scores(features).astype(np.float64)
        cumulative, evidence_seen = _evidence_prefix_scores(features, is_evidence, self.probe.model)
        if not np.all(np.isfinite(instantaneous)):
            raise FloatingPointError(f"episode {spec.episode} instantaneous probe score is NaN/Inf")
        if not self.args.smoke_only:
            if not np.array_equal(frames, np.arange(spec.length, dtype=np.int32)):
                raise RuntimeError(f"episode {spec.episode} full frame coverage is not exact stride one")
            expected_mask = np.zeros(spec.length, dtype=bool)
            expected_mask[np.asarray(spec.evidence_frames)] = True
            if not np.array_equal(is_evidence, expected_mask):
                raise RuntimeError(f"episode {spec.episode} evidence mask differs from corrected labels")
            evidence_features = features[is_evidence]
            final_feature = np.mean(evidence_features, axis=0, dtype=np.float32)
            artifact_feature = self.probe.episode_writer_evidence[spec.episode]
            max_difference = float(np.max(np.abs(final_feature - artifact_feature)))
            if max_difference > FINAL_FEATURE_ATOL:
                raise RuntimeError(
                    f"episode {spec.episode} full evidence feature differs from prerequisite artifact: "
                    f"max_abs={max_difference}"
                )
            final_score = float(self.probe.model.scores(final_feature[None])[0])
            artifact_score = self.probe.heldout_report_logits[spec.episode]
            prefix_final = float(cumulative[-1])
            if not math.isclose(final_score, artifact_score, rel_tol=0.0, abs_tol=FINAL_LOGIT_ATOL):
                raise RuntimeError(
                    f"episode {spec.episode} fresh head failed final artifact-score gate: "
                    f"{final_score} != {artifact_score}"
                )
            if not math.isclose(prefix_final, artifact_score, rel_tol=0.0, abs_tol=FINAL_LOGIT_ATOL):
                raise RuntimeError(
                    f"episode {spec.episode} causal prefix failed frozen final-score gate: "
                    f"{prefix_final} != {artifact_score}"
                )
        else:
            max_difference = None
            final_score = None
            artifact_score = self.probe.heldout_report_logits[spec.episode]

        return {
            "spec": spec,
            "source": payload["source"],
            "task_names": task_names,
            "label_runs": payload["label_runs"],
            "storage": payload["storage"],
            "frames": frames,
            "tasks": tasks,
            "is_evidence": is_evidence,
            "features": features,
            "instantaneous": instantaneous,
            "cumulative": cumulative,
            "evidence_seen": evidence_seen,
            "top_images": top_images,
            "selected_model_inputs_sha256": input_digest.hexdigest(),
            "model_batches": model_batches,
            "full_feature_max_abs_difference_from_probe_artifact": max_difference,
            "full_evidence_logit": final_score,
            "artifact_heldout_logit": artifact_score,
        }

    def _validate_saved_scores(self, path: Path, result: Mapping[str, Any]) -> dict[str, Any]:
        expected_keys = {
            "cumulative_evidence_logit_right_minus_left",
            "evidence_seen_count",
            "frame_index",
            "frame_writer_feature",
            "instantaneous_logit_right_minus_left",
            "is_evidence",
            "task_index",
        }
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != expected_keys:
                raise RuntimeError(f"saved score NPZ schema changed: {sorted(arrays.files)}")
            features = np.asarray(arrays["frame_writer_feature"])
            frames = np.asarray(arrays["frame_index"])
            tasks = np.asarray(arrays["task_index"])
            mask = np.asarray(arrays["is_evidence"])
            instant = np.asarray(arrays["instantaneous_logit_right_minus_left"])
            cumulative = np.asarray(arrays["cumulative_evidence_logit_right_minus_left"])
            counts = np.asarray(arrays["evidence_seen_count"])
        frame_count = len(result["frames"])
        feature_width = self.probe.model.mean.size
        if features.shape != (frame_count, feature_width) or features.dtype != np.float32:
            raise RuntimeError(f"saved frame writer feature contract failed: {features.shape}/{features.dtype}")
        if frames.dtype != np.int32 or not np.array_equal(frames, result["frames"]):
            raise RuntimeError("saved frame indices differ from inference")
        if tasks.dtype != np.int16 or not np.array_equal(tasks, result["tasks"]):
            raise RuntimeError("saved task indices differ from inference")
        if mask.dtype != np.bool_ or not np.array_equal(mask, result["is_evidence"]):
            raise RuntimeError("saved evidence mask differs from corrected task labels")
        recomputed_instant = self.probe.model.scores(features)
        if instant.dtype != np.float64 or not np.allclose(
            instant, recomputed_instant, rtol=0.0, atol=INSTANT_SCORE_ATOL
        ):
            raise RuntimeError("saved instantaneous scores do not reproduce from saved features/head")
        recomputed_cumulative, recomputed_counts = _evidence_prefix_scores(features, mask, self.probe.model)
        if cumulative.dtype != np.float64 or not np.allclose(
            cumulative, recomputed_cumulative, rtol=0.0, atol=INSTANT_SCORE_ATOL, equal_nan=True
        ):
            raise RuntimeError("saved evidence-prefix scores do not reproduce causally")
        if counts.dtype != np.int32 or not np.array_equal(counts, recomputed_counts):
            raise RuntimeError("saved evidence-prefix counts do not reproduce causally")
        return {
            "schema_exact": True,
            "frame_count": frame_count,
            "feature_width": feature_width,
            "instantaneous_scores_reproduced_from_saved_features_and_head": True,
            "evidence_prefix_recomputed_in_order": True,
        }

    def _write_episode(
        self, stage_dir: Path, result: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        spec: fixed.EpisodeSpec = result["spec"]
        episode = spec.episode
        stem = f"episode_{episode:06d}"
        identities: dict[str, dict[str, Any]] = {}
        score_name = f"{stem}_fresh_writer_probe_scores.npz"
        score_path = stage_dir / score_name
        identities[score_name] = _write_npz_atomic(
            score_path,
            {
                "frame_index": result["frames"],
                "task_index": result["tasks"],
                "is_evidence": result["is_evidence"],
                "frame_writer_feature": result["features"],
                "instantaneous_logit_right_minus_left": result["instantaneous"],
                "cumulative_evidence_logit_right_minus_left": result["cumulative"],
                "evidence_seen_count": result["evidence_seen"],
            },
        )
        score_checks = self._validate_saved_scores(score_path, result)

        video_name = f"{stem}_fresh_writer_probe.mp4"
        video_path = stage_dir / video_name
        video = attention_video._AtomicMp4Writer(
            video_path,
            fps=SOURCE_FPS,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
        )
        try:
            for index, frame in enumerate(result["frames"]):
                cumulative = float(result["cumulative"][index])
                rendered = _render_frame(
                    result["top_images"][index],
                    episode=episode,
                    frame=int(frame),
                    full_frame_count=spec.length,
                    fps=SOURCE_FPS,
                    prompt=spec.prompt,
                    truth_side=spec.side,
                    gt_subtask=result["task_names"][int(result["tasks"][index])],
                    phase_category=_phase_category(
                        spec, int(frame), int(result["tasks"][index])
                    ),
                    instantaneous_score=float(result["instantaneous"][index]),
                    cumulative_score=cumulative if math.isfinite(cumulative) else None,
                    evidence_seen=int(result["evidence_seen"][index]),
                    evidence_total=len(spec.evidence_frames),
                    is_evidence=bool(result["is_evidence"][index]),
                    checkpoint_step=self.args.checkpoint_step,
                    parameter_source=self.args.parameter_source,
                )
                video.write(rendered)
            identities[video_name] = video.close()
        except BaseException:
            video.abort()
            raise
        video_checks = attention_video._probe_mp4(
            video_path,
            fps=SOURCE_FPS,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            frames=len(result["frames"]),
        )
        evidence_indices = np.flatnonzero(result["is_evidence"])
        episode_summary = {
            "episode": episode,
            "prompt": spec.prompt,
            "truth_side": spec.side,
            "full_source_frame_count": spec.length,
            "rendered_frame_count": len(result["frames"]),
            "rendered_frame_range": [int(result["frames"][0]), int(result["frames"][-1])],
            "fps": SOURCE_FPS,
            "label_runs": result["label_runs"],
            "evidence": {
                "task_index": EVIDENCE_TASK_INDEX,
                "label": EVIDENCE_LABEL,
                "full_frame_range": [spec.evidence_frames[0], spec.evidence_frames[-1]],
                "full_frame_count": len(spec.evidence_frames),
                "rendered_evidence_frame_count": len(evidence_indices),
            },
            "predictions": {
                "instantaneous_definition": "fresh fixed head applied to current normalized mean-slot W_t",
                "instantaneous_phase_ood_marked_outside_evidence": True,
                "instantaneous_aggregation_shift_even_inside_evidence": (
                    "head was fit on episode-mean evidence features, not individual frames"
                ),
                "evidence_prefix_definition": (
                    "fresh fixed head applied to causal mean of only task_index==4 W features observed so far; "
                    "N/A before evidence and frozen after evidence"
                ),
                "score": "raw signed logistic logit, right-minus-left; threshold strictly >0 predicts right",
                "sigmoid_probability_displayed": False,
                "final_full_evidence_logit": result["full_evidence_logit"],
                "fixed_probe_artifact_heldout_logit": result["artifact_heldout_logit"],
                "final_prediction": (
                    _prediction(float(result["artifact_heldout_logit"]))
                    if not self.args.smoke_only
                    else None
                ),
                "full_feature_max_abs_difference_from_probe_artifact": result[
                    "full_feature_max_abs_difference_from_probe_artifact"
                ],
            },
            "inference": {
                "raw_frame_stride": 1,
                "fixed_batch_size": self.args.batch_size,
                "padded_rows_discarded": True,
                "model_batches": result["model_batches"],
                "fresh_independent_M0_per_batch": True,
                "writes": 0,
                "returned_state_threaded": False,
                "selected_model_inputs_sha256": result["selected_model_inputs_sha256"],
                "writer_input_scope": "all three cameras + robot state + instruction after exact run5 transforms",
                "displayed_camera": "exact transformed 224x224 top camera only",
            },
            "source_parquet_storage": result["storage"],
            "artifacts": {"video": video_name, "scores": score_name},
            "validation": {"scores_npz": score_checks, "video": video_checks},
        }
        return episode_summary, identities

    def _provenance(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "checkpoint": {
                "resolved_path": str(self.args.checkpoint),
                "checkpoint_manager_step_label": self.args.checkpoint_step,
                "internal_train_state_step": self.args.checkpoint_step + 1,
            },
            "checkpoint_origin": self.runtime.checkpoint_origin,
            "parameter_source": self.args.parameter_source,
            "parameter_provenance": self.runtime.parameter_provenance,
            "fixed_probe_prerequisite": {
                "resolved_directory": str(self.args.probe_artifact_dir),
                "artifact_identities": self.probe.artifact_identities,
                "heldout_excluded_from_fit": True,
                "train_episode_ids": list(self.probe.train_indices),
                "train_episode_count": len(self.probe.train_indices),
                "fit_feature": "episode_writer_evidence",
                "fit_head": {
                    "classifier": "binary logistic regression, right=1, left=0",
                    "standardizer_fit": "56 nonheldout evidence episode means only",
                    "zero_initialized": True,
                    "l2": fixed.PROBE_L2,
                    "steps": fixed.PROBE_STEPS,
                    "learning_rate": fixed.PROBE_LR,
                },
            },
            "source_provenance": {
                "exact_run5_launch_source": self.runtime.source_provenance,
                "current_video_script": fixed.causal._file_identity(Path(__file__).resolve()),
                "fixed_probe_script": fixed.causal._file_identity(Path(fixed.__file__).resolve()),
                "media_helper_script": fixed.causal._file_identity(Path(attention_video.__file__).resolve()),
            },
            "normalization_asset_provenance": self.runtime.norm_provenance,
            "tokenizer_asset_provenance": self.runtime.tokenizer_provenance,
            "dataset_root": str(self.args.dataset_root),
            "dataset_metadata_identity": {
                name: fixed.causal._file_identity(self.args.dataset_root / "meta" / name)
                for name in ("info.json", "tasks.jsonl", "episodes.jsonl", "episode_prompts.json")
            },
            "interpretation_limits": [
                "CURRENT W_t changes the probe's fit aggregation unit from episode mean to one frame.",
                "CURRENT W_t is additionally phase-OOD outside the evidence task block.",
                "Partial evidence prefixes are shorter than the complete episode means used for fitting.",
                "The probe establishes side decodability, not that the side signal came from the visible bin.",
                "Only top camera is displayed; writer W is conditioned on all cameras, robot state, and instruction.",
                "Raw signed logits are not calibrated probabilities.",
            ],
        }

    def run(self) -> dict[str, Any]:
        with attention_video._Stage(self.args.artifact_dir) as stage:
            identities: dict[str, dict[str, Any]] = {}
            head_name = "fresh_writer_probe_head.npz"
            identities[head_name] = _write_npz_atomic(
                stage.stage_dir / head_name,
                {
                    "mean": np.asarray(self.probe.model.mean, dtype=np.float64),
                    "scale": np.asarray(self.probe.model.scale, dtype=np.float64),
                    "weights": np.asarray(self.probe.model.weights, dtype=np.float64),
                    "train_episode_ids": np.asarray(self.probe.train_indices, dtype=np.int16),
                    "heldout_episode_ids": np.asarray(HELDOUT_EPISODES, dtype=np.int16),
                    "heldout_full_evidence_logits_right_minus_left": np.asarray(
                        [self.probe.heldout_report_logits[episode] for episode in HELDOUT_EPISODES],
                        dtype=np.float64,
                    ),
                },
            )
            episode_summaries = []
            for episode in self.args.episodes:
                payload = self._load_episode(episode)
                result = self._infer_episode(payload)
                summary, episode_identities = self._write_episode(stage.stage_dir, result)
                overlap = set(identities) & set(episode_identities)
                if overlap:
                    raise RuntimeError(f"duplicate artifact names: {sorted(overlap)}")
                identities.update(episode_identities)
                episode_summaries.append(summary)
                print(
                    f"ep{episode}: {len(result['frames'])} frames, "
                    f"artifact evidence logit={result['artifact_heldout_logit']:+.3f}",
                    flush=True,
                )

            provenance_name = "provenance.json"
            identities[provenance_name] = fixed.decode_video._write_json_atomic(
                stage.stage_dir / provenance_name, self._provenance()
            )
            report = {
                "schema": SCHEMA_VERSION,
                "status": "pass",
                "mode": "smoke-only" if self.args.smoke_only else "full",
                "checkpoint_step": self.args.checkpoint_step,
                "internal_train_state_step": self.args.checkpoint_step + 1,
                "parameter_source": self.args.parameter_source,
                "episodes": episode_summaries,
                "probe_contract": {
                    "fit_episode_count": EXPECTED_TRAIN_EPISODES,
                    "fit_episode_ids": list(self.probe.train_indices),
                    "heldout_episode_ids": list(HELDOUT_EPISODES),
                    "heldout_never_fit_scaled_or_tuned": True,
                    "head_artifact": head_name,
                    "score_sign": "right-minus-left",
                    "decision_rule": "score > 0 predicts right; score <= 0 predicts left",
                    "sigmoid_not_displayed_because_logits_are_not_calibrated": True,
                },
                "inference_contract": {
                    "fresh_M0": True,
                    "memory_writes": 0,
                    "memory_state_threaded": False,
                    "raw_frame_stride": 1,
                    "displayed_camera": "exact transformed top camera",
                    "writer_conditioning": "all three images + state + instruction",
                },
                "render_contract": {
                    "fps": SOURCE_FPS,
                    "size": [CANVAS_WIDTH, CANVAS_HEIGHT],
                    "full_exact_frame_counts": None if self.args.smoke_only else EXPECTED_FRAME_COUNTS,
                    "main_scores": ["PROBE THIS FRAME", "PROBE FROM EVIDENCE SO FAR"],
                    "current_score_phase_ood_visibly_marked_outside_evidence": True,
                    "evidence_prefix_NA_before_updates_only_evidence_freezes_after": True,
                },
                "provenance_file": provenance_name,
            }
            summary_name = "summary.json"
            identities[summary_name] = fixed.decode_video._write_json_atomic(
                stage.stage_dir / summary_name, report
            )
            stage.publish(identities)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "artifact_dir": str(self.args.artifact_dir),
                    "episodes": list(self.args.episodes),
                },
                indent=2,
            ),
            flush=True,
        )
        return report


def main(argv: list[str] | None = None) -> int:
    HeldoutFreshWriterProbeVideo(_parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
