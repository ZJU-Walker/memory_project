"""Render evidence-only dual-writer-head videos under native and switched instructions.

For each run5 heldout episode, the evaluator visits only the unshifted
``inspect both bins`` evidence block (``task_index == 4``), where both objects
are visible.  It evaluates two otherwise identical counterfactual conditions:

* ``native`` keeps the episode's original instruction.
* ``switched`` replaces only the instruction with the other task instruction.

Every frame is independently evaluated from a fresh M0; no fast-memory write is
committed and no returned state is threaded.  The same writer feature is scored
simultaneously by two deliberately different heads:

* ``FRESH FIXED`` is reconstructed from a hash-validated
  ``v34_fixed_writer_probe_eval`` artifact for the same checkpoint.  Its scaler
  and logistic head are fit on one complete evidence average from each of the 56
  nonheldout demos.  The four heldouts never enter fitting, standardization, or
  threshold selection.
* ``CHECKPOINT ONLINE`` is the accumulated two-class ``ladder_writer_head``
  stored inside that exact raw/EMA checkpoint.  It is not refit here.  "Online"
  describes the head's training history, not this evaluator's memory mode.

Both heads show the current-frame score and the score of the causal evidence
mean seen so far.  In native full mode, both final scores must reproduce the
hash-validated prerequisite report for the same checkpoint.

The switched target side is the opposite physical bin because this task places
exactly one object in each bin.  This counterfactual tests whether task-conditioned
writer features change when the prompt changes while pixels and robot state stay
fixed.  It does not test fast-memory retention or downstream action use.
"""

# ruff: noqa: SLF001, I001 - pyarrow must precede the audited OpenPI/JAX import stack.
from __future__ import annotations

import pyarrow.parquet as pq

import argparse
from collections.abc import Mapping
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Literal

import v34_heldout_fresh_writer_probe_video as base

import cv2
import jax
import jax.numpy as jnp
import numpy as np


fixed = base.fixed
attention_video = base.attention_video

SCHEMA_VERSION = "openpi.v34.heldout_evidence_instruction_probe_video.v2"
RUN5_CONFIG = base.RUN5_CONFIG
HELDOUT_EPISODES = tuple(base.HELDOUT_EPISODES)
EXPECTED_CELLS = base.EXPECTED_CELLS
SOURCE_FPS = base.SOURCE_FPS
EVIDENCE_TASK_INDEX = base.EVIDENCE_TASK_INDEX
EVIDENCE_LABEL = base.EVIDENCE_LABEL
MODEL_IMAGE_SIZE = base.MODEL_IMAGE_SIZE
DISPLAY_SCALE = base.DISPLAY_SCALE
DISPLAY_SIZE = base.DISPLAY_SIZE
HEADER_HEIGHT = 150
CANVAS_WIDTH = DISPLAY_SIZE
CANVAS_HEIGHT = HEADER_HEIGHT + DISPLAY_SIZE
FINAL_FEATURE_ATOL = base.FINAL_FEATURE_ATOL
FINAL_LOGIT_ATOL = base.FINAL_LOGIT_ATOL
INSTANT_SCORE_ATOL = base.INSTANT_SCORE_ATOL
ONLINE_CANONICAL_REPRO_ATOL = 0.0
# XLA's linear dot can change its reduction tree with call-batch shape.  The independent
# checkpoint14750 audit measured at most ~9.6e-4 drift from the same extracted FP32 affine.
ONLINE_RUNTIME_PARITY_ATOL = 2e-3
ONLINE_FINAL_LOGIT_ATOL = 2e-3
CONDITIONS = ("native", "switched")
HEAD_NAMES = ("fresh_fixed_56_demo", "checkpoint_online_stored")
SMOKE_FRAMES_PER_CONDITION = 5
ROW_COLUMNS = base.ROW_COLUMNS


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
            raise ValueError("checkpoint must end in a nonnegative numeric step")
        if self.config != RUN5_CONFIG:
            raise ValueError(f"instruction-probe video is pinned to --config {RUN5_CONFIG!r}")
        if self.parameter_source not in ("raw", "ema"):
            raise ValueError("--parameter-source must be raw or ema")
        if isinstance(self.batch_size, bool) or not 1 <= self.batch_size <= 64:
            raise ValueError("--batch-size must lie in [1, 64]")
        if fixed.causal._is_relative_to(self.output_dir, self.checkpoint):
            raise ValueError("diagnostic output must be outside the checkpoint directory")
        if fixed.causal._is_relative_to(self.output_dir, self.probe_artifact_dir):
            raise ValueError("video output must be outside the fixed-probe artifact")

    @property
    def checkpoint_step(self) -> int:
        return int(self.checkpoint.name)

    @property
    def artifact_dir(self) -> Path:
        return self.output_dir / self.parameter_source

    @property
    def episodes(self) -> tuple[int, ...]:
        return (HELDOUT_EPISODES[0],) if self.smoke_only else HELDOUT_EPISODES


@dataclasses.dataclass(frozen=True)
class Condition:
    name: Literal["native", "switched"]
    prompt: str
    target_side: str


@dataclasses.dataclass(frozen=True)
class OnlineWriterHead:
    """Exact two-class linear parameters extracted from the loaded checkpoint."""

    kernel: np.ndarray
    bias: np.ndarray

    def __post_init__(self) -> None:
        kernel = np.asarray(self.kernel)
        bias = np.asarray(self.bias)
        if kernel.ndim != 2 or kernel.shape[1] != 2 or kernel.shape[0] < 1:
            raise ValueError(f"online writer kernel must have shape [width, 2], got {kernel.shape}")
        if bias.shape != (2,):
            raise ValueError(f"online writer bias must have shape [2], got {bias.shape}")
        if kernel.dtype != np.float32 or bias.dtype != np.float32:
            raise ValueError(f"online writer parameters must be float32, got {kernel.dtype}/{bias.dtype}")
        if not np.all(np.isfinite(kernel)) or not np.all(np.isfinite(bias)):
            raise ValueError("online writer parameters contain NaN/Inf")
        object.__setattr__(self, "kernel", kernel.copy())
        object.__setattr__(self, "bias", bias.copy())

    @property
    def feature_width(self) -> int:
        return int(self.kernel.shape[0])

    def logits(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.feature_width:
            raise ValueError(f"online writer features must have shape [N, {self.feature_width}], got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("online writer features contain NaN/Inf")
        return np.asarray(values @ self.kernel + self.bias, dtype=np.float32)

    def scores(self, features: np.ndarray) -> np.ndarray:
        logits = self.logits(features)
        return np.asarray(logits[:, 1] - logits[:, 0], dtype=np.float64)


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


def _other_prompt(prompt: str) -> str:
    if prompt not in fixed.PROMPTS or len(fixed.PROMPTS) != 2:
        raise ValueError(f"run5 prompt vocabulary changed: {fixed.PROMPTS!r}")
    return fixed.PROMPTS[1] if prompt == fixed.PROMPTS[0] else fixed.PROMPTS[0]


def _opposite_side(side: str) -> str:
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    raise ValueError(f"unknown side {side!r}")


def _conditions(spec: fixed.EpisodeSpec) -> tuple[Condition, Condition]:
    return (
        Condition("native", spec.prompt, spec.side),
        Condition("switched", _other_prompt(spec.prompt), _opposite_side(spec.side)),
    )


def _parameter_value(parameter: Any, *, name: str) -> np.ndarray:
    if parameter is None:
        raise ValueError(f"checkpoint online writer head lacks {name}")
    value = getattr(parameter, "value", parameter)
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"checkpoint online writer head {name} is not array-like") from error


def _extract_online_writer_head(runtime: Any) -> OnlineWriterHead:
    module = getattr(runtime, "_online_writer_head", None)
    if module is None:
        raise ValueError("runtime lacks the checkpoint online writer head")
    return OnlineWriterHead(
        kernel=_parameter_value(getattr(module, "kernel", None), name="kernel"),
        bias=_parameter_value(getattr(module, "bias", None), name="bias"),
    )


def _online_report_logits(report: Mapping[str, Any], *, parameter_source: str) -> dict[int, float]:
    """Read the independently generated native heldout logits from the prerequisite."""
    online = base._require_mapping(report.get("stored_online_writer_head"), "stored_online_writer_head")
    if online.get("parameter_source") != parameter_source:
        raise ValueError("stored online-head report parameter source differs from this evaluator")
    heldout = base._require_mapping(online.get("heldout_4"), "stored_online_writer_head.heldout_4")
    rows = heldout.get("episodes")
    if not isinstance(rows, list) or len(rows) != len(HELDOUT_EPISODES):
        raise ValueError("stored online-head report lacks the exact four heldout rows")
    result: dict[int, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("stored online-head heldout row is not an object")
        episode = row.get("episode")
        score = row.get("evidence_logit_right_minus_left")
        if (
            isinstance(episode, bool)
            or not isinstance(episode, int)
            or episode not in HELDOUT_EPISODES
            or episode in result
            or isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(float(score))
        ):
            raise ValueError(f"invalid stored online-head heldout row: {row}")
        prediction = row.get("evidence_prediction")
        truth_side = row.get("truth_side")
        if prediction != base._prediction(float(score)):
            raise ValueError(f"stored online-head prediction disagrees with its score for ep{episode}")
        if truth_side != EXPECTED_CELLS[episode][1]:
            raise ValueError(f"stored online-head truth side changed for ep{episode}")
        result[episode] = float(score)
    if set(result) != set(HELDOUT_EPISODES):
        raise ValueError("stored online-head heldout coverage changed")
    return result


def _evidence_prefix_features(features: np.ndarray) -> np.ndarray:
    """Causal float32 evidence means for an evidence-only frame sequence."""
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or len(values) < 1 or values.shape[1] < 1:
        raise ValueError(f"evidence features must have nonempty shape [N, width], got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("evidence features contain NaN/Inf")
    result = np.empty_like(values)
    for index in range(len(values)):
        # Match the fixed-probe artifact's per-episode float32 mean at the final prefix.
        result[index] = np.mean(values[: index + 1], axis=0, dtype=np.float32)
    return result


def _checkpoint_head_scores(head: Any, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or len(values) < 1 or values.shape[1] < 1:
        raise ValueError(f"checkpoint-head features must have nonempty shape [N, width], got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("checkpoint-head features contain NaN/Inf")
    logits = head(jnp.asarray(values))
    jax.block_until_ready(logits)
    logits = np.asarray(logits)
    if logits.shape != (len(values), 2) or not np.all(np.isfinite(logits)):
        raise ValueError(f"checkpoint online writer head returned invalid logits {logits.shape}")
    return np.asarray(logits[:, 1] - logits[:, 0], dtype=np.float64)


def _compact_score(score: float) -> str:
    if not math.isfinite(score):
        return "N/A"
    return f"{base._prediction(score).upper()} {score:+.3f}"


def _paired_effect_aggregates(pair_summaries: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not pair_summaries:
        raise ValueError("paired instruction summaries cannot be empty")
    fields = (
        "native_final_correct",
        "switched_final_correct",
        "shift_toward_switched_target",
        "instruction_switch_changed_prediction",
    )
    aggregates: dict[str, Any] = {}
    for head_name in HEAD_NAMES:
        effects = []
        for pair in pair_summaries:
            heads = pair.get("heads")
            if not isinstance(heads, Mapping) or not isinstance(heads.get(head_name), Mapping):
                raise ValueError(f"paired summary lacks head {head_name!r}")
            effect = heads[head_name]
            for field in fields:
                if not isinstance(effect.get(field), bool):
                    raise ValueError(f"paired summary {head_name}.{field} must be bool")
            effects.append(effect)
        denominator = len(effects)
        counts = {field: sum(bool(effect[field]) for effect in effects) for field in fields}
        aggregates[head_name] = {
            "episode_count": denominator,
            "native_correct_count": counts["native_final_correct"],
            "switched_correct_count": counts["switched_final_correct"],
            "shift_toward_switched_target_count": counts["shift_toward_switched_target"],
            "prediction_flip_count": counts["instruction_switch_changed_prediction"],
            "native_correct_fraction": counts["native_final_correct"] / denominator,
            "switched_correct_fraction": counts["switched_final_correct"] / denominator,
            "shift_toward_switched_target_fraction": (counts["shift_toward_switched_target"] / denominator),
            "prediction_flip_fraction": (counts["instruction_switch_changed_prediction"] / denominator),
        }
    return aggregates


def _render_frame(
    top_image: np.ndarray,
    *,
    episode: int,
    frame: int,
    evidence_index: int,
    evidence_total: int,
    condition: Condition,
    current_score: float,
    prefix_score: float,
    online_current_score: float,
    online_prefix_score: float,
    checkpoint_step: int,
    parameter_source: str,
) -> np.ndarray:
    image = np.asarray(top_image)
    if image.shape != (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 3) or image.dtype != np.uint8:
        raise ValueError(f"top image must be exact uint8 224x224 RGB, got {image.shape}/{image.dtype}")
    if not 1 <= evidence_index <= evidence_total:
        raise ValueError("evidence index is outside the evidence block")
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    enlarged = cv2.resize(image, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
    canvas[HEADER_HEIGHT:] = enlarged
    mode = "ORIGINAL" if condition.name == "native" else "SWITCHED"
    color = (110, 245, 150) if condition.name == "native" else (255, 190, 110)
    base._put_text(
        canvas,
        f"ep{episode:03d}  raw frame={frame:04d}  evidence={evidence_index:02d}/{evidence_total:02d}  "
        f"ckpt={checkpoint_step} {parameter_source.upper()}",
        (8, 19),
        scale=0.41,
    )
    base._put_text(
        canvas,
        f"INSTRUCTION={mode}: {condition.prompt}",
        (8, 43),
        color=color,
        scale=0.44,
    )
    base._put_text(
        canvas,
        f"TARGET OBJECT SIDE={condition.target_side.upper()}  |  both objects visible  |  {EVIDENCE_LABEL}",
        (8, 66),
        color=(180, 225, 255),
        scale=0.38,
    )
    base._put_text(
        canvas,
        f"FRESH FIXED (56-demo refit): frame {_compact_score(current_score)}  |  "
        f"evidence-prefix {_compact_score(prefix_score)} [{evidence_index}/{evidence_total}]",
        (8, 92),
        color=(110, 215, 255),
        scale=0.36,
    )
    base._put_text(
        canvas,
        f"CHECKPOINT ONLINE (stored): frame {_compact_score(online_current_score)}  |  "
        f"evidence-prefix {_compact_score(online_prefix_score)} [{evidence_index}/{evidence_total}]",
        (8, 118),
        color=(255, 155, 235),
        scale=0.36,
    )
    base._put_text(
        canvas,
        "Same W feature; fresh M0/row; 0 writes. Online=head training history, not online memory. Top shown; W uses all inputs.",
        (8, 141),
        color=(175, 175, 175),
        scale=0.28,
    )

    badge = canvas[HEADER_HEIGHT : HEADER_HEIGHT + 82].copy()
    canvas[HEADER_HEIGHT : HEADER_HEIGHT + 82] = cv2.addWeighted(badge, 0.28, np.zeros_like(badge), 0.72, 0.0)
    base._put_text(
        canvas,
        f"FRESH   frame {_compact_score(current_score)}  |  prefix {_compact_score(prefix_score)}",
        (12, HEADER_HEIGHT + 29),
        color=(110, 215, 255),
        scale=0.46,
        thickness=2,
    )
    base._put_text(
        canvas,
        f"ONLINE  frame {_compact_score(online_current_score)}  |  prefix {_compact_score(online_prefix_score)}",
        (12, HEADER_HEIGHT + 63),
        color=(255, 155, 235),
        scale=0.46,
        thickness=2,
    )
    cv2.rectangle(
        canvas,
        (1, HEADER_HEIGHT + 1),
        (CANVAS_WIDTH - 2, CANVAS_HEIGHT - 2),
        color,
        3,
    )
    return canvas


class EvidenceInstructionProbeVideo:
    def __init__(self, args: Args):
        self.args = args
        if args.artifact_dir.exists():
            raise FileExistsError(f"refusing to overwrite artifact directory: {args.artifact_dir}")
        runtime_args = base._RuntimeArgs(
            checkpoint=args.checkpoint,
            dataset_root=args.dataset_root,
            config=args.config,
            parameter_source=args.parameter_source,
            artifact_dir=args.output_dir / f".{args.parameter_source}.runtime-preflight.{os.getpid()}",
        )
        self.runtime = fixed._Run5Runtime(runtime_args)
        self.specs, self.dataset_provenance = fixed._load_specs(self.runtime, runtime_args)
        self.probe = base._load_probe_bundle(
            args,
            self.runtime,
            self.specs,
            self.dataset_provenance,
        )
        self.online_head = _extract_online_writer_head(self.runtime)
        if self.online_head.feature_width != self.probe.model.mean.size:
            raise ValueError("fresh fixed and checkpoint online heads expect different writer feature widths")
        self.online_artifact_logits = _online_report_logits(
            self.probe.report,
            parameter_source=args.parameter_source,
        )
        expected_step = {
            "checkpoint_manager_step_label": args.checkpoint_step,
            "internal_train_state_step": args.checkpoint_step + 1,
        }
        if self.runtime.parameter_provenance.get("train_state_step_identity") != expected_step:
            raise ValueError("checkpoint label/internal-step gate failed")

    def _load_episode(self, episode: int) -> dict[str, Any]:
        spec = self.specs[episode]
        if not spec.heldout or spec.cell != tuple(EXPECTED_CELLS[episode]):
            raise ValueError(f"heldout episode {episode} design changed")
        sources = fixed.causal._wc._load_lerobot_sources(self.args.dataset_root, [episode])
        if len(sources) != 1:
            raise ValueError(f"expected one source for heldout episode {episode}")
        source = sources[0]
        if Path(source.path).resolve() != spec.parquet:
            raise ValueError("heldout parquet path differs from fixed-probe design")
        task_names = tuple(source.task_names)
        if task_names != tuple(self.runtime.data_config.memory_subtask_vocab):
            raise ValueError("task vocabulary/order differs from exact run5 config")
        if float(source.control_hz) != SOURCE_FPS:
            raise ValueError(f"episode {episode} fps changed")
        parquet = pq.ParquetFile(source.path)
        missing = set(ROW_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"episode {episode} parquet lacks {sorted(missing)}")
        rows = parquet.read(columns=list(ROW_COLUMNS)).to_pylist()
        label_runs = fixed.decode_video._validate_rows(
            rows,
            episode=episode,
            expected_length=spec.length,
            task_names=task_names,
        )
        selected = [rows[frame] for frame in spec.evidence_frames]
        for expected_frame, row in zip(spec.evidence_frames, selected, strict=True):
            if int(row["frame_index"]) != expected_frame or int(row["task_index"]) != EVIDENCE_TASK_INDEX:
                raise ValueError("selected evidence frames differ from exact unshifted task labels")
        if self.args.smoke_only:
            selected = selected[:SMOKE_FRAMES_PER_CONDITION]
        stat = Path(source.path).stat()
        return {
            "spec": spec,
            "rows": selected,
            "task_names": task_names,
            "label_runs": label_runs,
            "storage": {
                "resolved_path": str(Path(source.path).resolve()),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "parquet_rows": parquet.metadata.num_rows,
            },
        }

    def _infer(self, payload: Mapping[str, Any], condition: Condition) -> dict[str, Any]:
        spec: fixed.EpisodeSpec = payload["spec"]
        rows = payload["rows"]
        frame_features: list[np.ndarray] = []
        top_images: list[np.ndarray] = []
        frame_indices: list[int] = []
        input_digest = hashlib.sha256()
        nontext_digest = hashlib.sha256()
        batches = 0
        for start in range(0, len(rows), self.args.batch_size):
            live_rows = rows[start : start + self.args.batch_size]
            padded_rows, live = fixed._pad_batch(live_rows, self.args.batch_size)
            observations = []
            model_states = []
            for row in padded_rows:
                frame = int(row["frame_index"])
                observation, model_state = self.runtime.observation(row, frame, condition.prompt)
                observations.append(observation)
                model_states.append(model_state)
            batched = self.runtime.batch_observations(observations)
            output = self.runtime._interface(
                batched,
                self.runtime.model.memory.init_state(self.args.batch_size),
            )
            jax.block_until_ready(output)
            pooled = fixed._online_pool(output["write_tokens"])
            for index, row in enumerate(padded_rows[:live]):
                frame = int(row["frame_index"])
                fixed._update_selected_input_hash(
                    input_digest,
                    spec=spec,
                    phase=f"evidence-instruction-{condition.name}",
                    row=row,
                )
                input_digest.update(f"inference_prompt={condition.prompt}\n".encode())
                frame_indices.append(frame)
                frame_features.append(pooled[index])
                processed_images = fixed.decode_video._processed_images(batched, index)
                top_images.append(processed_images["base_0_rgb"])
                nontext_digest.update(f"frame={frame}\n".encode())
                state = np.asarray(model_states[index], dtype=np.float32)
                nontext_digest.update(b"state<float32>\0" + state.tobytes(order="C"))
                for image_name in sorted(processed_images):
                    image = np.asarray(processed_images[image_name], dtype=np.uint8)
                    nontext_digest.update(image_name.encode() + b"<uint8>\0")
                    nontext_digest.update(image.tobytes(order="C"))
            batches += 1

        features = np.stack(frame_features).astype(np.float32)
        frames = np.asarray(frame_indices, dtype=np.int32)
        current = self.probe.model.scores(features).astype(np.float64)
        prefix, counts = base._evidence_prefix_scores(
            features,
            np.ones(len(features), dtype=bool),
            self.probe.model,
        )
        prefix_features = _evidence_prefix_features(features)
        # Canonical checkpoint-head scores use the extracted FP32 affine.  This is the exact
        # semantic linear head and is independently reproducible from the saved kernel/bias.
        # The module call is retained only as a bounded parity control because XLA dot
        # reductions exhibit small batch-shape-dependent numerical drift on GPU.
        online_current = self.online_head.scores(features)
        online_prefix = self.online_head.scores(prefix_features)
        runtime_online_current = _checkpoint_head_scores(
            self.runtime._online_writer_head,
            features,
        )
        runtime_online_prefix = _checkpoint_head_scores(
            self.runtime._online_writer_head,
            prefix_features,
        )
        runtime_current_difference = float(np.max(np.abs(runtime_online_current - online_current)))
        runtime_prefix_difference = float(np.max(np.abs(runtime_online_prefix - online_prefix)))
        if max(runtime_current_difference, runtime_prefix_difference) > ONLINE_RUNTIME_PARITY_ATOL:
            raise RuntimeError(
                "checkpoint module call differs from the extracted FP32 online-head affine "
                f"by {max(runtime_current_difference, runtime_prefix_difference):.6g}"
            )
        if not all(np.all(np.isfinite(scores)) for scores in (current, prefix, online_current, online_prefix)):
            raise FloatingPointError("one of the writer heads produced NaN/Inf")
        if not np.array_equal(counts, np.arange(1, len(features) + 1, dtype=np.int32)):
            raise RuntimeError("evidence-prefix count is not causal 1..N")

        feature_difference = None
        artifact_logit = None
        online_artifact_logit = None
        if condition.name == "native" and not self.args.smoke_only:
            final_feature = np.mean(features, axis=0, dtype=np.float32)
            artifact_feature = self.probe.episode_writer_evidence[spec.episode]
            feature_difference = float(np.max(np.abs(final_feature - artifact_feature)))
            if feature_difference > FINAL_FEATURE_ATOL:
                raise RuntimeError("native evidence feature differs from prerequisite artifact")
            artifact_logit = self.probe.heldout_report_logits[spec.episode]
            if not math.isclose(float(prefix[-1]), artifact_logit, rel_tol=0.0, abs_tol=FINAL_LOGIT_ATOL):
                raise RuntimeError("native final prefix does not reproduce prerequisite heldout logit")
            online_artifact_logit = self.online_artifact_logits[spec.episode]
            if not math.isclose(
                float(online_prefix[-1]),
                online_artifact_logit,
                rel_tol=0.0,
                abs_tol=ONLINE_FINAL_LOGIT_ATOL,
            ):
                raise RuntimeError("native checkpoint online-head prefix does not reproduce prerequisite heldout logit")
            if base._prediction(float(online_prefix[-1])) != base._prediction(online_artifact_logit):
                raise RuntimeError("native checkpoint online-head prediction sign differs from prerequisite report")

        return {
            "spec": spec,
            "condition": condition,
            "frames": frames,
            "features": features,
            "current": current,
            "prefix": prefix,
            "online_current": online_current,
            "online_prefix": online_prefix,
            "runtime_online_current_max_abs_difference_from_canonical": (runtime_current_difference),
            "runtime_online_prefix_max_abs_difference_from_canonical": (runtime_prefix_difference),
            "counts": counts,
            "top_images": top_images,
            "model_batches": batches,
            "selected_model_inputs_sha256": input_digest.hexdigest(),
            "model_visible_nontext_inputs_sha256": nontext_digest.hexdigest(),
            "native_feature_max_abs_difference": feature_difference,
            "native_artifact_logit": artifact_logit,
            "native_online_artifact_logit": online_artifact_logit,
            "label_runs": payload["label_runs"],
            "storage": payload["storage"],
        }

    def _write_result(
        self,
        stage_dir: Path,
        result: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        spec: fixed.EpisodeSpec = result["spec"]
        condition: Condition = result["condition"]
        stem = f"episode_{spec.episode:06d}_{condition.name}_instruction_evidence_dual_writer_heads"
        identities: dict[str, dict[str, Any]] = {}
        score_name = f"{stem}_scores.npz"
        score_path = stage_dir / score_name
        identities[score_name] = base._write_npz_atomic(
            score_path,
            {
                "frame_index": result["frames"],
                "frame_writer_feature": result["features"],
                "instantaneous_logit_right_minus_left": result["current"],
                "cumulative_evidence_logit_right_minus_left": result["prefix"],
                "checkpoint_online_instantaneous_logit_right_minus_left": result["online_current"],
                "checkpoint_online_cumulative_evidence_logit_right_minus_left": result["online_prefix"],
                "evidence_seen_count": result["counts"],
            },
        )
        with np.load(score_path, allow_pickle=False) as arrays:
            if set(arrays.files) != {
                "frame_index",
                "frame_writer_feature",
                "instantaneous_logit_right_minus_left",
                "cumulative_evidence_logit_right_minus_left",
                "checkpoint_online_instantaneous_logit_right_minus_left",
                "checkpoint_online_cumulative_evidence_logit_right_minus_left",
                "evidence_seen_count",
            }:
                raise RuntimeError("saved condition-score NPZ schema changed")
            saved_features = np.asarray(arrays["frame_writer_feature"])
            saved_frames = np.asarray(arrays["frame_index"])
            saved_current = np.asarray(arrays["instantaneous_logit_right_minus_left"])
            saved_prefix = np.asarray(arrays["cumulative_evidence_logit_right_minus_left"])
            saved_online_current = np.asarray(arrays["checkpoint_online_instantaneous_logit_right_minus_left"])
            saved_online_prefix = np.asarray(arrays["checkpoint_online_cumulative_evidence_logit_right_minus_left"])
            saved_counts = np.asarray(arrays["evidence_seen_count"])
        expected_frames = np.asarray(result["frames"], dtype=np.int32)
        if saved_frames.dtype != np.int32 or not np.array_equal(saved_frames, expected_frames):
            raise RuntimeError("saved evidence frame indices differ from inference")
        if saved_features.dtype != np.float32 or saved_features.shape != result["features"].shape:
            raise RuntimeError("saved frame writer feature shape/dtype changed")
        if any(
            scores.dtype != np.float64
            for scores in (
                saved_current,
                saved_prefix,
                saved_online_current,
                saved_online_prefix,
            )
        ):
            raise RuntimeError("saved dual-head scores must be float64")
        if saved_counts.dtype != np.int32:
            raise RuntimeError("saved evidence-prefix counts must be int32")
        if not np.allclose(
            self.probe.model.scores(saved_features),
            saved_current,
            rtol=0.0,
            atol=INSTANT_SCORE_ATOL,
        ):
            raise RuntimeError("saved instantaneous scores do not reproduce from saved features")
        recomputed_prefix, recomputed_counts = base._evidence_prefix_scores(
            saved_features,
            np.ones(len(saved_features), dtype=bool),
            self.probe.model,
        )
        if not np.allclose(saved_prefix, recomputed_prefix, rtol=0.0, atol=INSTANT_SCORE_ATOL):
            raise RuntimeError("saved evidence-prefix scores do not reproduce")
        if not np.array_equal(saved_counts, recomputed_counts):
            raise RuntimeError("saved evidence-prefix counts do not reproduce")
        reconstructed_online_current = self.online_head.scores(saved_features)
        reconstructed_online_prefix = self.online_head.scores(_evidence_prefix_features(saved_features))
        if not np.allclose(
            saved_online_current,
            reconstructed_online_current,
            rtol=0.0,
            atol=ONLINE_CANONICAL_REPRO_ATOL,
        ):
            raise RuntimeError("saved checkpoint online current scores do not reproduce")
        if not np.allclose(
            saved_online_prefix,
            reconstructed_online_prefix,
            rtol=0.0,
            atol=ONLINE_CANONICAL_REPRO_ATOL,
        ):
            raise RuntimeError("saved checkpoint online prefix scores do not reproduce")

        video_name = f"{stem}.mp4"
        video_path = stage_dir / video_name
        writer = attention_video._AtomicMp4Writer(
            video_path,
            fps=SOURCE_FPS,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
        )
        try:
            total = len(result["frames"])
            for index, frame in enumerate(result["frames"]):
                writer.write(
                    _render_frame(
                        result["top_images"][index],
                        episode=spec.episode,
                        frame=int(frame),
                        evidence_index=index + 1,
                        evidence_total=total,
                        condition=condition,
                        current_score=float(result["current"][index]),
                        prefix_score=float(result["prefix"][index]),
                        online_current_score=float(result["online_current"][index]),
                        online_prefix_score=float(result["online_prefix"][index]),
                        checkpoint_step=self.args.checkpoint_step,
                        parameter_source=self.args.parameter_source,
                    )
                )
            identities[video_name] = writer.close()
        except BaseException:
            writer.abort()
            raise
        video_checks = attention_video._probe_mp4(
            video_path,
            fps=SOURCE_FPS,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            frames=len(result["frames"]),
        )
        final_score = float(result["prefix"][-1])
        online_final_score = float(result["online_prefix"][-1])
        fresh_prediction = base._prediction(final_score)
        online_prediction = base._prediction(online_final_score)
        return (
            {
                "episode": spec.episode,
                "condition": condition.name,
                "native_episode_prompt": spec.prompt,
                "inference_prompt": condition.prompt,
                "target_side": condition.target_side,
                "counterfactual_target_rule": (
                    None
                    if condition.name == "native"
                    else "opposite physical bin: exactly one task object occupies each bin"
                ),
                "evidence_frame_range": [int(result["frames"][0]), int(result["frames"][-1])],
                "evidence_frame_count": len(result["frames"]),
                # Backward-compatible unqualified fields refer to the fresh fixed head.
                "final_evidence_logit_right_minus_left": final_score,
                "final_prediction": fresh_prediction,
                "final_correct_for_target_side": fresh_prediction == condition.target_side,
                "checkpoint_online_final_evidence_logit_right_minus_left": online_final_score,
                "checkpoint_online_final_prediction": online_prediction,
                "checkpoint_online_final_correct_for_target_side": (online_prediction == condition.target_side),
                "heads": {
                    "fresh_fixed_56_demo": {
                        "final_evidence_logit_right_minus_left": final_score,
                        "final_prediction": fresh_prediction,
                        "final_correct_for_target_side": fresh_prediction == condition.target_side,
                        "native_prerequisite_report_logit": result["native_artifact_logit"],
                    },
                    "checkpoint_online_stored": {
                        "final_evidence_logit_right_minus_left": online_final_score,
                        "final_prediction": online_prediction,
                        "final_correct_for_target_side": online_prediction == condition.target_side,
                        "native_prerequisite_report_logit": result["native_online_artifact_logit"],
                    },
                },
                "native_artifact_logit": result["native_artifact_logit"],
                "native_online_artifact_logit": result["native_online_artifact_logit"],
                "native_feature_max_abs_difference": result["native_feature_max_abs_difference"],
                "selected_model_inputs_sha256": result["selected_model_inputs_sha256"],
                "model_visible_nontext_inputs_sha256": result["model_visible_nontext_inputs_sha256"],
                "model_batches": result["model_batches"],
                "checkpoint_online_runtime_module_parity": {
                    "current_max_abs_difference_from_canonical": result[
                        "runtime_online_current_max_abs_difference_from_canonical"
                    ],
                    "prefix_max_abs_difference_from_canonical": result[
                        "runtime_online_prefix_max_abs_difference_from_canonical"
                    ],
                    "absolute_tolerance": ONLINE_RUNTIME_PARITY_ATOL,
                    "passes": True,
                },
                "source_parquet_storage": result["storage"],
                "artifacts": {"video": video_name, "scores": score_name},
                "validation": {
                    "score_npz_recomputed": True,
                    "causal_prefix_recomputed": True,
                    "checkpoint_online_scores_recomputed_from_extracted_head": True,
                    "checkpoint_online_native_report_prediction_sign_gate": (
                        condition.name != "native"
                        or self.args.smoke_only
                        or base._prediction(online_final_score)
                        == base._prediction(float(result["native_online_artifact_logit"]))
                    ),
                    "video": video_checks,
                },
            },
            identities,
        )

    def _provenance(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "checkpoint": {
                "resolved_path": str(self.args.checkpoint),
                "checkpoint_manager_step_label": self.args.checkpoint_step,
                "internal_train_state_step": self.args.checkpoint_step + 1,
            },
            "checkpoint_origin": self.runtime.checkpoint_origin,
            "checkpoint_metadata": self.runtime.checkpoint_info,
            "parameter_source": self.args.parameter_source,
            "parameter_provenance": self.runtime.parameter_provenance,
            "source_provenance": self.runtime.source_provenance,
            "normalization_asset_provenance": self.runtime.norm_provenance,
            "tokenizer_asset_provenance": self.runtime.tokenizer_provenance,
            "fixed_probe_prerequisite": {
                "resolved_directory": str(self.args.probe_artifact_dir),
                "artifact_identities": self.probe.artifact_identities,
                "train_episode_ids": list(self.probe.train_indices),
                "heldouts_excluded_from_fit_standardization_and_thresholding": True,
                "fresh_head_artifact": "fresh_writer_probe_head.npz",
            },
            "checkpoint_online_writer_head": {
                "semantics": (
                    "checkpoint's accumulated two-class ladder_writer_head; class-1 minus "
                    "class-0 logit; never refit by this evaluator"
                ),
                "canonical_scoring": (
                    "NumPy FP32 affine using the exact extracted checkpoint kernel+bias; "
                    "this avoids XLA dot reduction drift across call-batch shapes"
                ),
                "runtime_module_parity_control_absolute_tolerance": (ONLINE_RUNTIME_PARITY_ATOL),
                "native_prerequisite_report_absolute_tolerance": ONLINE_FINAL_LOGIT_ATOL,
                "native_prerequisite_prediction_sign_must_match": True,
                "parameter_source": self.args.parameter_source,
                "restored_parameter_tree_identity": self.runtime.parameter_provenance.get(
                    "restored_parameter_tree_identity"
                ),
                "extracted_head_artifact": "checkpoint_online_writer_head.npz",
                "kernel_shape": list(self.online_head.kernel.shape),
                "kernel_dtype": str(self.online_head.kernel.dtype),
                "bias_shape": list(self.online_head.bias.shape),
                "bias_dtype": str(self.online_head.bias.dtype),
                "native_heldout_reference_logits_right_minus_left": {
                    str(episode): score for episode, score in sorted(self.online_artifact_logits.items())
                },
            },
            "dataset_provenance": self.dataset_provenance,
            "current_script": fixed.causal._file_identity(Path(__file__).resolve()),
            "base_video_script": fixed.causal._file_identity(Path(base.__file__).resolve()),
            "fixed_probe_script": fixed.causal._file_identity(Path(fixed.__file__).resolve()),
            "media_helper_script": fixed.causal._file_identity(Path(attention_video.__file__).resolve()),
            "interpretation_limits": [
                "Both displays are writer-representation head scores, not fast-memory reads.",
                "Online names the checkpoint head's accumulated training history; evaluation still uses fresh M0 and zero writes.",
                "The switched condition changes only instruction tokens; pixels and state remain paired by frame.",
                "The fixed probe was trained on complete evidence averages, so individual-frame scores change its fit unit.",
                "The checkpoint online head was trained on individual evidence frames; its prefix display applies that linear head to a causal evidence mean.",
                "Training writer accuracy pools sampled evidence frames over a moving log window under train-time augmentation and 50% state masking; these videos use unaugmented evaluation transforms, full state, and every evidence frame.",
                "Top camera is displayed, but writer W also uses wrist images, robot state, and instruction.",
                "Signed logits are not calibrated probabilities.",
            ],
        }

    def run(self) -> dict[str, Any]:
        with attention_video._Stage(self.args.artifact_dir) as stage:
            identities: dict[str, dict[str, Any]] = {}
            fresh_head_name = "fresh_writer_probe_head.npz"
            identities[fresh_head_name] = base._write_npz_atomic(
                stage.stage_dir / fresh_head_name,
                {
                    "mean": np.asarray(self.probe.model.mean, dtype=np.float64),
                    "scale": np.asarray(self.probe.model.scale, dtype=np.float64),
                    "weights": np.asarray(self.probe.model.weights, dtype=np.float64),
                    "class_labels_left_right": np.asarray([0, 1], dtype=np.int8),
                    "train_episode_ids": np.asarray(self.probe.train_indices, dtype=np.int16),
                    "heldout_episode_ids": np.asarray(HELDOUT_EPISODES, dtype=np.int16),
                    "native_heldout_full_evidence_logits_right_minus_left": np.asarray(
                        [self.probe.heldout_report_logits[episode] for episode in HELDOUT_EPISODES],
                        dtype=np.float64,
                    ),
                },
            )
            online_head_name = "checkpoint_online_writer_head.npz"
            identities[online_head_name] = base._write_npz_atomic(
                stage.stage_dir / online_head_name,
                {
                    "kernel": self.online_head.kernel,
                    "bias": self.online_head.bias,
                    "class_labels_left_right": np.asarray([0, 1], dtype=np.int8),
                    "heldout_episode_ids": np.asarray(HELDOUT_EPISODES, dtype=np.int16),
                    "native_heldout_full_evidence_logits_right_minus_left": np.asarray(
                        [self.online_artifact_logits[episode] for episode in HELDOUT_EPISODES],
                        dtype=np.float64,
                    ),
                },
            )
            rows = []
            pair_summaries = []
            for episode in self.args.episodes:
                payload = self._load_episode(episode)
                episode_results = []
                episode_rows = []
                for condition in _conditions(payload["spec"]):
                    result = self._infer(payload, condition)
                    episode_results.append(result)
                    row, condition_identities = self._write_result(stage.stage_dir, result)
                    if set(identities) & set(condition_identities):
                        raise RuntimeError("duplicate condition artifact name")
                    identities.update(condition_identities)
                    rows.append(row)
                    episode_rows.append(row)
                    print(
                        f"ep{episode} {condition.name}: {len(result['frames'])} evidence frames, "
                        f"fresh={result['prefix'][-1]:+.3f}, "
                        f"online={result['online_prefix'][-1]:+.3f}",
                        flush=True,
                    )
                native, switched = episode_results
                if native["condition"].name != "native" or switched["condition"].name != "switched":
                    raise AssertionError("instruction-condition order changed")
                if not np.array_equal(native["frames"], switched["frames"]):
                    raise RuntimeError("native/switched conditions use different evidence frames")
                if native["model_visible_nontext_inputs_sha256"] != switched["model_visible_nontext_inputs_sha256"]:
                    raise RuntimeError("native/switched conditions changed pixels or robot state")
                if native["condition"].prompt == switched["condition"].prompt:
                    raise RuntimeError("switched condition did not change the instruction")
                native_row, switched_row = episode_rows
                target_sign = 1.0 if switched_row["target_side"] == "right" else -1.0
                prompt_shift = (
                    switched_row["final_evidence_logit_right_minus_left"]
                    - native_row["final_evidence_logit_right_minus_left"]
                )
                online_prompt_shift = (
                    switched_row["checkpoint_online_final_evidence_logit_right_minus_left"]
                    - native_row["checkpoint_online_final_evidence_logit_right_minus_left"]
                )
                fresh_effect = {
                    "native_final_logit_right_minus_left": native_row["final_evidence_logit_right_minus_left"],
                    "switched_final_logit_right_minus_left": switched_row["final_evidence_logit_right_minus_left"],
                    "native_final_prediction": native_row["final_prediction"],
                    "switched_final_prediction": switched_row["final_prediction"],
                    "native_final_correct": native_row["final_correct_for_target_side"],
                    "switched_final_correct": switched_row["final_correct_for_target_side"],
                    "instruction_switch_changed_prediction": (
                        native_row["final_prediction"] != switched_row["final_prediction"]
                    ),
                    "right_minus_left_logit_shift_switched_minus_native": prompt_shift,
                    "shift_toward_switched_target": target_sign * prompt_shift > 0.0,
                }
                online_effect = {
                    "native_final_logit_right_minus_left": native_row[
                        "checkpoint_online_final_evidence_logit_right_minus_left"
                    ],
                    "switched_final_logit_right_minus_left": switched_row[
                        "checkpoint_online_final_evidence_logit_right_minus_left"
                    ],
                    "native_final_prediction": native_row["checkpoint_online_final_prediction"],
                    "switched_final_prediction": switched_row["checkpoint_online_final_prediction"],
                    "native_final_correct": native_row["checkpoint_online_final_correct_for_target_side"],
                    "switched_final_correct": switched_row["checkpoint_online_final_correct_for_target_side"],
                    "instruction_switch_changed_prediction": (
                        native_row["checkpoint_online_final_prediction"]
                        != switched_row["checkpoint_online_final_prediction"]
                    ),
                    "right_minus_left_logit_shift_switched_minus_native": online_prompt_shift,
                    "shift_toward_switched_target": target_sign * online_prompt_shift > 0.0,
                }
                pair_summaries.append(
                    {
                        "episode": episode,
                        "native_prompt": native_row["inference_prompt"],
                        "switched_prompt": switched_row["inference_prompt"],
                        "native_final_prediction": native_row["final_prediction"],
                        "switched_final_prediction": switched_row["final_prediction"],
                        "switched_target_side": switched_row["target_side"],
                        "switched_final_correct": switched_row["final_correct_for_target_side"],
                        "instruction_switch_changed_prediction": (
                            native_row["final_prediction"] != switched_row["final_prediction"]
                        ),
                        "right_minus_left_logit_shift_switched_minus_native": prompt_shift,
                        "shift_toward_switched_target": target_sign * prompt_shift > 0.0,
                        "checkpoint_online_native_final_prediction": online_effect["native_final_prediction"],
                        "checkpoint_online_switched_final_prediction": online_effect["switched_final_prediction"],
                        "checkpoint_online_switched_final_correct": online_effect["switched_final_correct"],
                        "checkpoint_online_instruction_switch_changed_prediction": online_effect[
                            "instruction_switch_changed_prediction"
                        ],
                        "checkpoint_online_right_minus_left_logit_shift_switched_minus_native": (online_prompt_shift),
                        "checkpoint_online_shift_toward_switched_target": online_effect["shift_toward_switched_target"],
                        "heads": {
                            "fresh_fixed_56_demo": fresh_effect,
                            "checkpoint_online_stored": online_effect,
                        },
                        "exact_nontext_input_hash": native["model_visible_nontext_inputs_sha256"],
                    }
                )

            paired_aggregate_counts = _paired_effect_aggregates(pair_summaries)
            provenance_name = "provenance.json"
            identities[provenance_name] = fixed.decode_video._write_json_atomic(
                stage.stage_dir / provenance_name,
                self._provenance(),
            )
            report = {
                "schema": SCHEMA_VERSION,
                "status": "pass",
                "mode": "smoke-only" if self.args.smoke_only else "full",
                "checkpoint_step": self.args.checkpoint_step,
                "internal_train_state_step": self.args.checkpoint_step + 1,
                "parameter_source": self.args.parameter_source,
                "episodes": rows,
                "paired_instruction_effects": pair_summaries,
                "paired_instruction_effect_aggregates": paired_aggregate_counts,
                "contract": {
                    "heldouts": list(HELDOUT_EPISODES),
                    "conditions_per_episode": list(CONDITIONS),
                    "only_unshifted_task_index_4_evidence_frames": True,
                    "both_objects_visible": True,
                    "native_and_switched_conditions_share_exact_frame_pixels_and_state": True,
                    "native_and_switched_nontext_input_hash_equality_enforced": True,
                    "fresh_M0_per_batch_row": True,
                    "memory_writes": 0,
                    "memory_state_threaded": False,
                    "returned_memory_state_used": False,
                    "checkpoint_online_names_head_training_history_not_memory_mode": True,
                    "training_log_comparison": {
                        "training_writer_accuracy": ("sampled evidence frames pooled over the moving logging window"),
                        "training_inputs": "train-time augmentation with 50% state masking",
                        "video_inputs": ("unaugmented evaluation transforms, full state, every evidence frame"),
                        "direct_numeric_equivalence_claimed": False,
                    },
                    "heads": {
                        "fresh_fixed_56_demo": {
                            "artifact": fresh_head_name,
                            "fit_episodes": list(self.probe.train_indices),
                            "heldouts_excluded_from_fit_standardization_and_thresholding": True,
                        },
                        "checkpoint_online_stored": {
                            "artifact": online_head_name,
                            "refit_by_this_evaluator": False,
                            "source": "exact restored checkpoint parameter tree",
                            "canonical_score_formula": (
                                "float32 features @ extracted float32 kernel + bias; right-minus-left"
                            ),
                            "runtime_module_parity_absolute_tolerance": (ONLINE_RUNTIME_PARITY_ATOL),
                            "native_report_absolute_tolerance": ONLINE_FINAL_LOGIT_ATOL,
                            "native_report_prediction_sign_gate": True,
                        },
                    },
                    "score_sign": "right-minus-left",
                    "decision_rule": "score > 0 predicts right; otherwise left",
                    "score_npz_unqualified_logit_fields": "fresh_fixed_56_demo",
                    "score_npz_checkpoint_online_fields_have_checkpoint_online_prefix": True,
                },
                "render": {
                    "fps": SOURCE_FPS,
                    "size": [CANVAS_WIDTH, CANVAS_HEIGHT],
                    "expected_full_video_count": 8,
                    "heads_overlaid_together": [
                        *HEAD_NAMES,
                    ],
                    "scores_per_head": ["current_frame", "causal_evidence_prefix"],
                    "expected_full_frame_counts_per_condition": {
                        str(episode): len(self.specs[episode].evidence_frames) for episode in HELDOUT_EPISODES
                    },
                },
                "provenance_file": provenance_name,
            }
            summary_name = "summary.json"
            identities[summary_name] = fixed.decode_video._write_json_atomic(
                stage.stage_dir / summary_name,
                report,
            )
            stage.publish(identities)
        print(
            json.dumps(
                {
                    "status": "pass",
                    "artifact_dir": str(self.args.artifact_dir),
                    "video_count": len(rows),
                },
                indent=2,
            ),
            flush=True,
        )
        return report


def main(argv: list[str] | None = None) -> int:
    EvidenceInstructionProbeVideo(_parse_args(argv)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
