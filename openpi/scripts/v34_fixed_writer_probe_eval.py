"""Fixed, fresh per-checkpoint writer probe for the exact v3.4 run5 protocol.

This diagnostic asks whether side information is linearly decodable from the writer pathway at
each checkpoint without inheriting the online probe head's training history.  It evaluates every
unshifted evidence frame (``task_index == 4``) from all 60 episodes, plus an equal-length window
immediately before evidence.  Every model call receives an independent fresh M0; no returned
memory state is ever threaded and no production write is performed.

The primary feature is exactly the feature seen by the online writer head::

    normalize(mean(write_tokens, slot_axis))

Projected value and key features use the same per-frame pooling.  Frame features are averaged
within episode with *no second normalization*.  A deterministic L2=1 logistic probe is initialized
at zero and fitted afresh in each of 14 fixed, four-cell-balanced OOF folds.  The evidence-trained
head is also applied to the matched pre-evidence window, using the same train-only standardizer.

The evaluator must be launched with the extracted run5 source snapshot ahead of the current tree::

    V34_RUN5_SOURCE_ROOT=/tmp/v34_run5_source \
    PYTHONPATH=/tmp/v34_run5_source/src \
    .venv/bin/python scripts/v34_fixed_writer_probe_eval.py \
      --checkpoint checkpoints/pi05_yam_mem_v34_run5_eta0/v34_run5_eta0/2500 \
      --dataset-root /iris/u/kewalk/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask \
      --output-dir diagnostic_outputs/v34_fixed_writer_probe/2500 \
      --parameter-source raw

``--output-dir`` is a checkpoint-level parent.  Results are always written below its ``raw`` or
``ema`` child, preventing the two parameter sources from being mixed.  The child must not exist.
"""

# ruff: noqa: SLF001, I001 - pyarrow must be imported before the OpenPI/JAX stack on Iris.
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
from typing import Any, Literal

# Evaluation must never resolve a different remote tokenizer/config revision.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import jax
import jax.numpy as jnp
import numpy as np

import v34_causal_memory_eval as causal
import v34_heldout_free_decode_video as decode_video


SCHEMA_VERSION = "openpi.v34.fixed_writer_probe.v1"
RUN5_CONFIG = causal.RUN5_CONFIG
HELDOUT_EPISODES = causal.EXPECTED_HELDOUT
EXPECTED_HELDOUT_CELLS = causal.EXPECTED_CELLS
EXPECTED_EPISODES = tuple(range(60))
EXPECTED_EVIDENCE_TASK_INDEX = 4
EXPECTED_EVIDENCE_LABEL = "inspect both bins"
EXPECTED_EVIDENCE_FRAMES = 3112
EXPECTED_TRAIN_EPISODES = 56
EXPECTED_CELL_TRAIN_EPISODES = 14
EXPECTED_FOLDS = 14
SIDE_TO_LABEL = {"left": 0, "right": 1}
LABEL_TO_SIDE = {value: key for key, value in SIDE_TO_LABEL.items()}
PROMPTS = ("find the banana", "find the grey pepper box")
STREAMS = ("writer", "value", "key", "state", "prompt")
PRIMARY_STREAMS = ("writer", "value", "key")
PROBE_L2 = 1.0
PROBE_STEPS = 400
PROBE_LR = 0.5
DEFAULT_NULL_REPEATS = 8
DEFAULT_SEED = 3401
MEMORY_INDEPENDENCE_ATOL = 1e-6


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint: Path
    dataset_root: Path
    output_dir: Path
    config: str = RUN5_CONFIG
    parameter_source: Literal["raw", "ema"] = "raw"
    batch_size: int = 8
    seed: int = DEFAULT_SEED
    null_repeats: int = DEFAULT_NULL_REPEATS
    smoke_only: bool = False

    def __post_init__(self) -> None:
        for name in ("checkpoint", "dataset_root", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        if not self.checkpoint.name.isdigit() or int(self.checkpoint.name) < 0:
            raise ValueError(f"checkpoint must end in a nonnegative numeric step, got {self.checkpoint.name!r}")
        if self.config != RUN5_CONFIG:
            raise ValueError(f"fixed writer probe is pinned to --config {RUN5_CONFIG!r}")
        if self.parameter_source not in ("raw", "ema"):
            raise ValueError("--parameter-source must be raw or ema")
        if isinstance(self.batch_size, bool) or not 1 <= self.batch_size <= 64:
            raise ValueError("--batch-size must lie in [1, 64]")
        if self.seed != DEFAULT_SEED:
            raise ValueError(f"fixed protocol requires --seed {DEFAULT_SEED}")
        if isinstance(self.null_repeats, bool) or self.null_repeats != DEFAULT_NULL_REPEATS:
            raise ValueError(f"fixed protocol requires --null-repeats {DEFAULT_NULL_REPEATS}")

    @property
    def artifact_dir(self) -> Path:
        return self.output_dir / self.parameter_source


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    episode: int
    prompt: str
    side: str
    label: int
    length: int
    parquet: Path
    evidence_frames: tuple[int, ...]
    approach_frames: tuple[int, ...]
    heldout: bool

    @property
    def cell(self) -> tuple[str, str]:
        return self.prompt, self.side


@dataclasses.dataclass(frozen=True)
class ProbeModel:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray

    def scores(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        standardized = (x - self.mean) / self.scale
        return standardized @ self.weights[:-1] + self.weights[-1]


@dataclasses.dataclass(frozen=True)
class Fold:
    fold: int
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def _parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", default=RUN5_CONFIG)
    parser.add_argument("--parameter-source", choices=("raw", "ema"), default="raw")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--null-repeats", type=int, default=DEFAULT_NULL_REPEATS)
    parser.add_argument("--smoke-only", action="store_true")
    return Args(**vars(parser.parse_args(argv)))


def _online_pool(tokens: Any) -> np.ndarray:
    """Exact float32 online-ladder feature: L2(mean slots), epsilon included."""
    value = jnp.asarray(tokens).astype(jnp.float32)
    if value.ndim != 3 or value.shape[1] < 1 or value.shape[2] < 1:
        raise ValueError(f"tokens must have shape [batch, slots, width], got {value.shape}")
    pooled = jnp.mean(value, axis=1)
    result = pooled * jax.lax.rsqrt(
        jnp.sum(jnp.square(pooled), axis=-1, keepdims=True) + 1e-12
    )
    result = np.asarray(result, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("online-pooled feature contains NaN/Inf")
    return result.astype(np.float32, copy=False)


def _l2_rows(features: Any) -> np.ndarray:
    value = jnp.asarray(features).astype(jnp.float32)
    if value.ndim != 2 or value.shape[1] < 1:
        raise ValueError(f"features must have shape [batch, width], got {value.shape}")
    result = value * jax.lax.rsqrt(
        jnp.sum(jnp.square(value), axis=-1, keepdims=True) + 1e-12
    )
    result = np.asarray(result, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("row-normalized feature contains NaN/Inf")
    return result


def _pad_batch(entries: Sequence[Any], batch_size: int) -> tuple[list[Any], int]:
    if not entries:
        raise ValueError("cannot pad an empty batch")
    if len(entries) > batch_size:
        raise ValueError("live batch cannot be larger than fixed batch size")
    return [*entries, *([entries[-1]] * (batch_size - len(entries)))], len(entries)


def _fit_probe(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    l2: float = PROBE_L2,
    steps: int = PROBE_STEPS,
    lr: float = PROBE_LR,
) -> ProbeModel:
    """Fit one deterministic logistic head from an explicit all-zero initialization."""
    x = np.asarray(train_features, dtype=np.float64)
    y = np.asarray(train_labels, dtype=np.float64)
    if x.ndim != 2 or y.shape != (x.shape[0],) or x.shape[0] < 4:
        raise ValueError(f"invalid probe shapes: features={x.shape}, labels={y.shape}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("probe inputs must be finite")
    if set(np.unique(y).tolist()) != {0.0, 1.0}:
        raise ValueError("probe training labels must contain exactly classes 0 and 1")
    if not math.isfinite(l2) or l2 != PROBE_L2:
        raise ValueError(f"fixed protocol requires l2={PROBE_L2}")
    if steps != PROBE_STEPS or not math.isfinite(lr) or lr != PROBE_LR:
        raise ValueError(f"fixed protocol requires steps={PROBE_STEPS}, lr={PROBE_LR}")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    standardized = (x - mean) / scale
    design = np.concatenate([standardized, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    weights = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(steps):
        logits = design @ weights
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        gradient = design.T @ (probabilities - y) / len(y)
        gradient[:-1] += l2 * weights[:-1] / len(y)
        weights -= lr * gradient
    if not np.all(np.isfinite(weights)):
        raise FloatingPointError("probe fit produced NaN/Inf")
    return ProbeModel(mean=mean, scale=scale, weights=weights)


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    for end in range(1, len(scores) + 1):
        if end == len(scores) or sorted_scores[end] != sorted_scores[start]:
            ranks[order[start:end]] = np.mean(ranks[order[start:end]])
            start = end
    return float((np.sum(ranks[labels == 1]) - positives * (positives + 1) / 2) / (positives * negatives))


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or labels.size == 0:
        raise ValueError(f"invalid metric shapes: labels={labels.shape}, scores={scores.shape}")
    if set(np.unique(labels).tolist()) - {0, 1} or not np.all(np.isfinite(scores)):
        raise ValueError("metrics require finite scores and binary labels")
    predictions = (scores > 0.0).astype(np.int64)
    recalls = [float(np.mean(predictions[labels == side] == side)) for side in (0, 1) if np.any(labels == side)]
    truth_sign = 2.0 * labels.astype(np.float64) - 1.0
    return {
        "n": int(labels.size),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(np.mean(recalls)),
        "auc": _roc_auc(labels, scores),
        "log_loss": float(np.mean(np.logaddexp(0.0, scores) - labels * scores)),
        "truth_margin": float(np.mean(truth_sign * scores)),
    }


def _paired_evidence_minus_approach(
    labels: np.ndarray, evidence_scores: np.ndarray, approach_scores: np.ndarray
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    evidence = np.asarray(evidence_scores, dtype=np.float64)
    approach = np.asarray(approach_scores, dtype=np.float64)
    if evidence.shape != labels.shape or approach.shape != labels.shape:
        raise ValueError("paired scores and labels must have identical vector shapes")
    aligned_delta = (2.0 * labels - 1.0) * (evidence - approach)
    return {
        "definition": "(2*y-1) * (evidence_logit - matched_approach_logit), paired by episode",
        "mean": float(np.mean(aligned_delta)),
        "median": float(np.median(aligned_delta)),
        "positive_fraction": float(np.mean(aligned_delta > 0.0)),
        "values": aligned_delta.tolist(),
    }


def _validate_design(specs: Sequence[EpisodeSpec]) -> tuple[tuple[int, ...], tuple[Fold, ...]]:
    """Validate the exact 60-episode design and build 14 deterministic balanced OOF folds."""
    if tuple(spec.episode for spec in specs) != EXPECTED_EPISODES:
        raise ValueError("episodes must be unique, sorted, and exactly 0..59")
    heldout = tuple(spec.episode for spec in specs if spec.heldout)
    if heldout != HELDOUT_EPISODES:
        raise ValueError(f"heldout set changed: expected {HELDOUT_EPISODES}, got {heldout}")
    for episode in HELDOUT_EPISODES:
        spec = specs[episode]
        expected = EXPECTED_HELDOUT_CELLS[episode]
        if spec.cell != expected:
            raise ValueError(f"heldout ep{episode} cell changed: expected {expected}, got {spec.cell}")

    train_indices = tuple(index for index, spec in enumerate(specs) if not spec.heldout)
    if len(train_indices) != EXPECTED_TRAIN_EPISODES:
        raise ValueError(f"expected {EXPECTED_TRAIN_EPISODES} fit episodes, got {len(train_indices)}")
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index in train_indices:
        spec = specs[index]
        if spec.prompt not in PROMPTS or spec.side not in SIDE_TO_LABEL or spec.label != SIDE_TO_LABEL[spec.side]:
            raise ValueError(f"episode {spec.episode} has invalid prompt/side/label cell")
        cells[spec.cell].append(index)
    expected_cells = {(prompt, side) for prompt in PROMPTS for side in SIDE_TO_LABEL}
    if set(cells) != expected_cells:
        raise ValueError(f"training cells changed: expected {sorted(expected_cells)}, got {sorted(cells)}")
    for cell, indices in cells.items():
        if len(indices) != EXPECTED_CELL_TRAIN_EPISODES:
            raise ValueError(f"training cell {cell} must contain 14 episodes, got {len(indices)}")
        indices.sort(key=lambda index: specs[index].episode)

    ordered_cells = tuple(sorted(expected_cells))
    folds = []
    train_set = set(train_indices)
    for fold_index in range(EXPECTED_FOLDS):
        test = tuple(cells[cell][fold_index] for cell in ordered_cells)
        fit = tuple(sorted(train_set - set(test)))
        if len(test) != 4 or len(fit) != 52 or set(fit) & set(test):
            raise AssertionError("balanced OOF fold construction failed")
        folds.append(Fold(fold=fold_index, train_indices=fit, test_indices=test))
    tested = [index for fold in folds for index in fold.test_indices]
    if sorted(tested) != sorted(train_indices) or len(tested) != len(set(tested)):
        raise AssertionError("each nonheldout episode must be tested exactly once")
    if any(set(fold.train_indices) & set(HELDOUT_EPISODES) for fold in folds):
        raise AssertionError("heldout episode entered an OOF fit")
    return train_indices, tuple(folds)


def _oof_probe(
    evidence_features: np.ndarray,
    approach_features: np.ndarray,
    labels: np.ndarray,
    folds: Sequence[Fold],
) -> dict[str, Any]:
    evidence_features = np.asarray(evidence_features, dtype=np.float64)
    approach_features = np.asarray(approach_features, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if evidence_features.shape != approach_features.shape or evidence_features.shape[0] != len(labels):
        raise ValueError("evidence/approach features and labels are not aligned")
    evidence_scores = np.full(len(labels), np.nan, dtype=np.float64)
    approach_scores = np.full(len(labels), np.nan, dtype=np.float64)
    tested: list[int] = []
    fold_records = []
    for fold in folds:
        fit = np.asarray(fold.train_indices, dtype=np.int64)
        test = np.asarray(fold.test_indices, dtype=np.int64)
        model = _fit_probe(evidence_features[fit], labels[fit])
        evidence_scores[test] = model.scores(evidence_features[test])
        # Critically: no refit and no approach-derived standardization.
        approach_scores[test] = model.scores(approach_features[test])
        tested.extend(test.tolist())
        fold_records.append(
            {
                "fold": fold.fold,
                "train_indices": fit.tolist(),
                "test_indices": test.tolist(),
                "zero_initialized": True,
            }
        )
    test_indices = np.asarray(sorted(tested), dtype=np.int64)
    if len(test_indices) != len(set(test_indices.tolist())) or np.any(~np.isfinite(evidence_scores[test_indices])):
        raise AssertionError("OOF prediction coverage is not exact")
    evidence = evidence_scores[test_indices]
    approach = approach_scores[test_indices]
    truth = labels[test_indices]
    return {
        "test_indices": test_indices,
        "evidence_scores": evidence,
        "approach_scores": approach,
        "evidence_metrics": _binary_metrics(truth, evidence),
        "approach_same_head_metrics": _binary_metrics(truth, approach),
        "paired_evidence_minus_approach": _paired_evidence_minus_approach(truth, evidence, approach),
        "folds": fold_records,
    }


def _fit_all_then_heldout(
    evidence_features: np.ndarray,
    approach_features: np.ndarray,
    labels: np.ndarray,
    train_indices: Sequence[int],
    heldout_indices: Sequence[int],
) -> dict[str, Any]:
    train = np.asarray(train_indices, dtype=np.int64)
    heldout = np.asarray(heldout_indices, dtype=np.int64)
    if set(train.tolist()) & set(heldout.tolist()):
        raise ValueError("heldout episodes entered final probe fit")
    model = _fit_probe(evidence_features[train], labels[train])
    evidence_scores = model.scores(evidence_features[heldout])
    approach_scores = model.scores(approach_features[heldout])
    truth = labels[heldout]
    return {
        "indices": heldout,
        "evidence_scores": evidence_scores,
        "approach_scores": approach_scores,
        "evidence_metrics": _binary_metrics(truth, evidence_scores),
        "approach_same_head_metrics": _binary_metrics(truth, approach_scores),
        "paired_evidence_minus_approach": _paired_evidence_minus_approach(
            truth, evidence_scores, approach_scores
        ),
    }


def _within_prompt_shuffles(
    labels: np.ndarray,
    prompts: Sequence[str],
    train_indices: Sequence[int],
    *,
    repeats: int = DEFAULT_NULL_REPEATS,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, ...]:
    labels = np.asarray(labels, dtype=np.int64)
    if repeats != DEFAULT_NULL_REPEATS or seed != DEFAULT_SEED:
        raise ValueError("within-prompt null parameters are pinned by the protocol")
    train = np.asarray(train_indices, dtype=np.int64)
    rng = np.random.default_rng(seed)
    outputs = []
    for _ in range(repeats):
        shuffled = labels.copy()
        for prompt in PROMPTS:
            group = train[np.asarray([prompts[index] == prompt for index in train], dtype=bool)]
            if len(group) != 28:
                raise ValueError(f"prompt {prompt!r} must have 28 nonheldout episodes, got {len(group)}")
            shuffled[group] = rng.permutation(labels[group])
            if sorted(shuffled[group].tolist()) != sorted(labels[group].tolist()):
                raise AssertionError("within-prompt shuffle changed class counts")
        outputs.append(shuffled)
    return tuple(outputs)


def _null_summary(
    evidence_features: np.ndarray,
    approach_features: np.ndarray,
    shuffled_labels: Sequence[np.ndarray],
    folds: Sequence[Fold],
) -> dict[str, Any]:
    results = [
        _oof_probe(evidence_features, approach_features, labels, folds)
        for labels in shuffled_labels
    ]
    metric_names = ("accuracy", "balanced_accuracy", "auc", "log_loss", "truth_margin")
    summary: dict[str, Any] = {"repeats": len(results), "per_repeat": []}
    for result in results:
        summary["per_repeat"].append(
            {
                "evidence": result["evidence_metrics"],
                "approach_same_head": result["approach_same_head_metrics"],
                "paired_mean": result["paired_evidence_minus_approach"]["mean"],
            }
        )
    for view in ("evidence_metrics", "approach_same_head_metrics"):
        name = "evidence" if view == "evidence_metrics" else "approach_same_head"
        summary[name] = {}
        for metric in metric_names:
            values = np.asarray([result[view][metric] for result in results], dtype=np.float64)
            summary[name][metric] = {
                "mean": float(np.nanmean(values)),
                "p05": float(np.nanpercentile(values, 5)),
                "p95": float(np.nanpercentile(values, 95)),
            }
    return summary


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class _AtomicOutput:
    """Own an exclusive output directory and publish COMPLETE only after atomic artifacts."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir).resolve()
        self.stage_dir: Path | None = None
        self._published = False

    def __enter__(self) -> _AtomicOutput:
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        # mkdir is the no-replace publication lock: exactly one process can own this namespace.
        try:
            self.output_dir.mkdir()
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to overwrite output directory: {self.output_dir}") from exc
        self.stage_dir = self.output_dir
        return self

    def _path(self, name: str) -> Path:
        if self.stage_dir is None or Path(name).name != name:
            raise ValueError("atomic output is not active or artifact name is unsafe")
        return self.stage_dir / name

    @staticmethod
    def _identity(path: Path) -> dict[str, Any]:
        return causal._file_identity(path)

    def _fsync_directory(self) -> None:
        if self.stage_dir is None:
            raise RuntimeError("atomic output is not active")
        descriptor = os.open(self.stage_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def write_npz(self, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
        path = self._path("features.npz")
        temporary = self._path(".features.npz.tmp")
        with temporary.open("wb") as handle:
            # Neural features are effectively incompressible; uncompressed NPZ avoids a long
            # single-core compression tail after GPU extraction.
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        self._fsync_directory()
        return self._identity(path)

    def write_report(self, report: Mapping[str, Any]) -> dict[str, Any]:
        path = self._path("report.json")
        temporary = self._path(".report.json.tmp")
        payload = json.dumps(_jsonable(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        self._fsync_directory()
        return self._identity(path)

    def complete(self, identities: Mapping[str, Mapping[str, Any]]) -> None:
        path = self._path("COMPLETE")
        temporary = self._path(".COMPLETE.tmp")
        lines = []
        for name in ("features.npz", "report.json"):
            digest = identities.get(name, {}).get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"artifact {name} lacks a SHA-256 identity")
            lines.append(f"{digest}  {name}")
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        self._fsync_directory()
        self._published = True
        self.stage_dir = None

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if self.stage_dir is not None and not self._published:
            shutil.rmtree(self.stage_dir)
            self.stage_dir = None


def _validate_checkpoint_origin(
    checkpoint: Path, repo: Path, checkpoint_info: Mapping[str, Any]
) -> dict[str, Any]:
    """Accept any finalized numeric step under the exact run5 live/archive roots.

    Unlike the older causal audit helper, this does not demand that an ongoing run's pilot log
    already contain a save line for every later checkpoint.  Finalization metadata, the numeric
    label/internal TrainState step, the exact run5 path, and launch-source identities are checked
    independently.  Archived copies additionally require their immutable copy manifest.
    """
    checkpoint = checkpoint.resolve()
    repo = repo.resolve()
    live_root = (repo / "checkpoints/pi05_yam_mem_v34_run5_eta0/v34_run5_eta0").resolve()
    archive_root = (repo / "diagnostic_checkpoints").resolve()
    if checkpoint.parent == live_root:
        origin_kind = "exact live run5 checkpoint root"
        archive_manifest = None
        accepted_root = live_root
    elif causal._is_relative_to(checkpoint, archive_root) and any(
        "v34_run5_eta0" in part for part in checkpoint.relative_to(archive_root).parts[:-1]
    ):
        origin_kind = "run5-labelled diagnostic archive root"
        accepted_root = archive_root
        archive_manifest = causal._validate_archive_snapshot_manifest(
            checkpoint, live_root / checkpoint.name, dict(checkpoint_info)
        )
    else:
        raise ValueError(
            "checkpoint must be a direct numeric child of the exact run5 live root or a "
            f"manifest-bound run5 archive under {archive_root}; got {checkpoint}"
        )
    if checkpoint_info.get("step_label") != int(checkpoint.name):
        raise ValueError("checkpoint metadata step label differs from its numeric directory")
    pilot_log = repo / causal.PILOT_LOG
    return {
        "origin_kind": origin_kind,
        "accepted_root": str(accepted_root),
        "resolved_step_path": str(checkpoint),
        "archive_snapshot_manifest": archive_manifest,
        "pilot_log_identity_if_present": causal._file_identity(pilot_log) if pilot_log.is_file() else None,
        "binding_scope": (
            "exact run5 path + finalized checkpoint metadata + numeric/internal step identity + "
            "exact launch-source snapshot; archive copies additionally require their copy manifest"
        ),
    }


def _preflight_target(args: Args) -> None:
    output = args.artifact_dir
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output}")
    if causal._is_relative_to(output, args.checkpoint):
        raise ValueError("diagnostic output must be outside the checkpoint directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.parent.is_dir():
        raise NotADirectoryError(output.parent)


class _Run5Runtime:
    """Model and exact inference transforms reconstructed from the run5 launch snapshot."""

    def __init__(self, args: Args):
        self.repo = Path(__file__).resolve().parents[1]
        _preflight_target(args)
        if not args.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")

        self.source_provenance = decode_video._validate_run5_source_root(self.repo)
        checkpoint_info = causal._checkpoint_metadata(args.checkpoint)
        self.checkpoint_origin = _validate_checkpoint_origin(args.checkpoint, self.repo, checkpoint_info)
        self.checkpoint_info = checkpoint_info

        self.train_config = causal._config.get_config(args.config)
        self.tokenizer_provenance = causal._resolve_tokenizer_assets()
        self.data_config = self.train_config.data.create(
            self.train_config.assets_dirs, self.train_config.model
        )
        causal._validate_run5_semantics(
            args.config,
            self.train_config.model,
            self.data_config,
            self.train_config.ema_decay,
        )
        self._validate_writer_semantics()

        asset_id = self.data_config.asset_id
        if not isinstance(asset_id, str) or not asset_id:
            raise ValueError("run5 data config lacks a normalization asset id")
        norm_path = args.checkpoint / "assets" / asset_id
        if not norm_path.is_dir():
            raise FileNotFoundError(f"checkpoint-local normalization stats missing: {norm_path}")
        self.norm_provenance = causal._directory_identity(norm_path)
        norm_stats = causal._normalize.load(norm_path)

        self.model, self.parameter_provenance = causal._load_model(
            args.checkpoint, self.train_config, args.parameter_source
        )
        if float(self.model.memory.config.eta_scale) != 0.0:
            raise ValueError("loaded checkpoint GraphDef does not preserve run5 eta_scale=0")
        if not hasattr(self.model, "ladder_writer_head"):
            raise ValueError("loaded run5 model has no stored online writer head")

        input_transforms = [
            transform
            for transform in self.data_config.data_transforms.inputs
            if not isinstance(transform, causal._transforms.BuildMemorySequence)
        ]
        self.input_transform = causal._transforms.compose(
            [
                *input_transforms,
                causal._transforms.Normalize(
                    norm_stats, use_quantiles=self.data_config.use_quantile_norm
                ),
                *self.data_config.model_transforms.inputs,
            ]
        )
        self._interface, self._write, self._online_writer_head = _bind_runtime_callables(
            self.model
        )

    def _validate_writer_semantics(self) -> None:
        failures = []
        if tuple(self.data_config.heldout_episodes) != HELDOUT_EPISODES:
            failures.append(f"heldout_episodes={tuple(self.data_config.heldout_episodes)!r}")
        if tuple(self.data_config.evidence_subtasks) != (EXPECTED_EVIDENCE_LABEL,):
            failures.append(f"evidence_subtasks={tuple(self.data_config.evidence_subtasks)!r}")
        required = tuple(self.data_config.memory_required_subtasks)
        if required != ("wait; target bin is left", "wait; target bin is right"):
            failures.append(f"memory_required_subtasks={required!r}")
        vocab = tuple(self.data_config.memory_subtask_vocab)
        if len(vocab) <= EXPECTED_EVIDENCE_TASK_INDEX or vocab[EXPECTED_EVIDENCE_TASK_INDEX] != EXPECTED_EVIDENCE_LABEL:
            failures.append("task_index 4 is no longer the exact evidence label")
        if not bool(getattr(self.train_config.model, "memory_ladder_probes", False)):
            failures.append("memory_ladder_probes=False")
        if getattr(self.train_config.model, "memory_architecture", None) != "v32_layer8_dual_query":
            failures.append(
                f"memory_architecture={getattr(self.train_config.model, 'memory_architecture', None)!r}"
            )
        if failures:
            raise ValueError("run5 writer-probe semantics mismatch: " + ", ".join(failures))

    def observation(self, row: Mapping[str, Any], frame: int, prompt: str) -> tuple[Any, np.ndarray]:
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
        state = np.asarray(transformed["state"], dtype=np.float32)
        if state.ndim != 1 or not np.all(np.isfinite(state)):
            raise ValueError(f"frame {frame} transformed state is invalid: {state.shape}")
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
        return causal._model.Observation.from_dict(batched), state

    @staticmethod
    def batch_observations(observations: Sequence[Any]) -> Any:
        return jax.tree.map(lambda *values: jnp.concatenate(values, axis=0), *observations)

    @staticmethod
    def state_max_abs_difference(left: Any, right: Any) -> float:
        leaves = [
            jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))
            for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
        ]
        if not leaves:
            raise ValueError("memory state has no leaves")
        return float(np.asarray(jnp.max(jnp.stack(leaves))))


def _bind_runtime_callables(model: Any) -> tuple[Any, Any, Any]:
    """JIT bound model methods while leaving the callable NNX head module unwrapped."""
    interface = causal.nnx_utils.module_jit(model.v32_memory_interface_step)
    write = causal.nnx_utils.module_jit(model.memory.write)
    # ``ladder_writer_head`` is an NNX module object, not a bound method.  Calling the tiny
    # two-class linear head directly is correct; ``module_jit`` intentionally rejects modules.
    return interface, write, model.ladder_writer_head


def _read_episode_metadata(dataset_root: Path) -> dict[int, int]:
    records = causal._wc._read_jsonl(dataset_root / "meta" / "episodes.jsonl")
    lengths: dict[int, int] = {}
    for record in records:
        episode = record.get("episode_index")
        length = record.get("length")
        if (
            isinstance(episode, bool)
            or not isinstance(episode, int)
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
            or episode in lengths
        ):
            raise ValueError(f"invalid or duplicate episodes.jsonl record: {record}")
        lengths[episode] = length
    if tuple(sorted(lengths)) != EXPECTED_EPISODES:
        raise ValueError("episodes.jsonl must contain exactly episode ids 0..59")
    return lengths


def _validate_task_rows(
    table: pa.Table, *, episode: int, expected_length: int, task_names: Sequence[str]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if table.num_rows != expected_length:
        raise ValueError(
            f"episode {episode} parquet/metadata frame count mismatch: {table.num_rows} != {expected_length}"
        )
    rows = table.to_pydict()
    frame_indices = rows["frame_index"]
    episode_indices = rows["episode_index"]
    task_indices = rows["task_index"]
    if frame_indices != list(range(expected_length)):
        raise ValueError(f"episode {episode} frame_index is not contiguous from zero")
    if any(value != episode for value in episode_indices):
        raise ValueError(f"episode {episode} parquet has a mismatched episode_index")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(task_names)
        for value in task_indices
    ):
        raise ValueError(f"episode {episode} contains an invalid task_index")
    runs = []
    start = 0
    for end in range(1, len(task_indices) + 1):
        if end == len(task_indices) or task_indices[end] != task_indices[start]:
            runs.append(int(task_indices[start]))
            start = end
    if len(runs) != len(set(runs)):
        raise ValueError(f"episode {episode} task labels are not contiguous blocks: {runs}")

    evidence = tuple(index for index, task in enumerate(task_indices) if task == EXPECTED_EVIDENCE_TASK_INDEX)
    if not evidence or evidence != tuple(range(evidence[0], evidence[-1] + 1)):
        raise ValueError(f"episode {episode} must have one nonempty contiguous task_index==4 block")
    count = len(evidence)
    if evidence[0] < count:
        raise ValueError(f"episode {episode} lacks {count} immediately-pre-evidence frames")
    approach = tuple(range(evidence[0] - count, evidence[0]))
    if any(task_indices[index] == EXPECTED_EVIDENCE_TASK_INDEX for index in approach):
        raise ValueError(f"episode {episode} matched approach window overlaps evidence")
    return evidence, approach


def _load_specs(runtime: _Run5Runtime, args: Args) -> tuple[list[EpisodeSpec], dict[str, Any]]:
    plans = causal._probe._plan_all_episodes(
        SimpleNamespace(dataset_root=args.dataset_root), runtime.data_config
    )
    by_episode = {int(plan.episode): plan for plan in plans}
    if tuple(sorted(by_episode)) != EXPECTED_EPISODES or len(by_episode) != len(plans):
        raise ValueError("run5 planner must resolve exactly one plan for every episode 0..59")
    lengths = _read_episode_metadata(args.dataset_root)
    sources = causal._wc._load_lerobot_sources(args.dataset_root, EXPECTED_EPISODES)
    expected_vocab = tuple(runtime.data_config.memory_subtask_vocab)
    specs = []
    storage = []
    task_protocol_digest = hashlib.sha256()
    for episode, source in zip(EXPECTED_EPISODES, sources, strict=True):
        plan = by_episode[episode]
        if tuple(source.task_names) != expected_vocab:
            raise ValueError(f"episode {episode} task vocabulary differs from run5 config")
        if int(plan.length) != lengths[episode]:
            raise ValueError(
                f"episode {episode} planner/metadata length mismatch: {plan.length} != {lengths[episode]}"
            )
        parquet = pq.ParquetFile(source.path)
        columns = {"frame_index", "episode_index", "task_index"}
        missing = columns - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"episode {episode} parquet lacks {sorted(missing)}")
        table = parquet.read(columns=sorted(columns))
        task_rows = table.to_pydict()
        task_protocol_digest.update(f"episode={episode}\n".encode())
        for name in ("frame_index", "episode_index", "task_index"):
            task_protocol_digest.update(name.encode() + b"\0")
            task_protocol_digest.update(
                np.asarray(task_rows[name], dtype=np.int64).tobytes(order="C")
            )
        evidence, approach = _validate_task_rows(
            table,
            episode=episode,
            expected_length=lengths[episode],
            task_names=source.task_names,
        )
        side = str(plan.side)
        prompt = str(plan.prompt)
        if side not in SIDE_TO_LABEL or prompt not in PROMPTS:
            raise ValueError(f"episode {episode} has invalid cell {(prompt, side)}")
        specs.append(
            EpisodeSpec(
                episode=episode,
                prompt=prompt,
                side=side,
                label=SIDE_TO_LABEL[side],
                length=lengths[episode],
                parquet=Path(source.path).resolve(),
                evidence_frames=evidence,
                approach_frames=approach,
                heldout=episode in set(HELDOUT_EPISODES),
            )
        )
        stat = Path(source.path).stat()
        storage.append(
            {
                "episode": episode,
                "resolved_path": str(Path(source.path).resolve()),
                "bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "parquet_rows": parquet.metadata.num_rows,
            }
        )
    total = sum(len(spec.evidence_frames) for spec in specs)
    if total != EXPECTED_EVIDENCE_FRAMES:
        raise ValueError(f"evidence frame count changed: expected {EXPECTED_EVIDENCE_FRAMES}, got {total}")
    if sum(len(spec.approach_frames) for spec in specs) != total:
        raise AssertionError("matched approach coverage differs from evidence coverage")
    _validate_design(specs)
    metadata_files = {}
    for name in ("info.json", "tasks.jsonl", "episodes.jsonl", "episode_prompts.json"):
        path = args.dataset_root / "meta" / name
        if not path.is_file():
            raise FileNotFoundError(f"dataset metadata file missing: {path}")
        metadata_files[name] = causal._file_identity(path)
    provenance = {
        "dataset_root": str(args.dataset_root),
        "metadata_file_identities": metadata_files,
        "parquet_storage_metadata": storage,
        "all_frame_episode_task_protocol_sha256": task_protocol_digest.hexdigest(),
        "selected_input_sha256": "filled after exact selected rows are streamed",
        "full_38GB_parquet_content_hash_scope": False,
    }
    return specs, provenance


SMOKE_EPISODES = (0, 7, 30, 45)
IMAGE_COLUMNS = ("image", "left_wrist_image", "right_wrist_image")
EXTRACTION_COLUMNS = (*IMAGE_COLUMNS, "state", "frame_index", "episode_index", "task_index")


def _selected_frames(spec: EpisodeSpec, *, smoke_only: bool) -> tuple[tuple[str, int], ...]:
    if not smoke_only:
        return tuple(("approach", frame) for frame in spec.approach_frames) + tuple(
            ("evidence", frame) for frame in spec.evidence_frames
        )
    if spec.episode not in SMOKE_EPISODES:
        return ()
    midpoint = len(spec.evidence_frames) // 2
    return (
        ("approach", spec.approach_frames[midpoint]),
        ("evidence", spec.evidence_frames[midpoint]),
    )


def _read_selected_rows(path: Path, frames: Sequence[int]) -> list[dict[str, Any]]:
    """Read only parquet row groups intersecting the exact requested frame indices."""
    wanted = tuple(int(frame) for frame in frames)
    if not wanted or len(wanted) != len(set(wanted)) or tuple(sorted(wanted)) != wanted:
        raise ValueError("selected frame indices must be nonempty, unique, and sorted")
    parquet = pq.ParquetFile(path)
    missing = set(EXTRACTION_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"parquet {path} lacks extraction columns {sorted(missing)}")
    selected: dict[int, dict[str, Any]] = {}
    offset = 0
    wanted_set = set(wanted)
    for row_group in range(parquet.num_row_groups):
        count = parquet.metadata.row_group(row_group).num_rows
        group_frames = [frame for frame in wanted if offset <= frame < offset + count]
        if group_frames:
            table = parquet.read_row_group(row_group, columns=list(EXTRACTION_COLUMNS))
            local = pa.array([frame - offset for frame in group_frames], type=pa.int64())
            rows = table.take(local).to_pylist()
            for frame, row in zip(group_frames, rows, strict=True):
                if int(row["frame_index"]) != frame:
                    raise ValueError(f"parquet row position/frame_index mismatch at {path}: {frame}")
                selected[frame] = row
        offset += count
    if set(selected) != wanted_set:
        raise ValueError(f"failed to read selected frames from {path}: missing {sorted(wanted_set - set(selected))}")
    return [selected[frame] for frame in wanted]


def _update_selected_input_hash(
    digest: Any, *, spec: EpisodeSpec, phase: str, row: Mapping[str, Any]
) -> None:
    """Bind every exact raw model input selected by the protocol in streaming order."""
    digest.update(f"ep={spec.episode};phase={phase};prompt={spec.prompt}\n".encode())
    for name in ("frame_index", "episode_index", "task_index"):
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"episode {spec.episode} has invalid {name}: {value!r}")
        digest.update(name.encode() + b"=" + str(value).encode() + b"\n")
    state = np.asarray(row.get("state"), dtype=np.float32)
    if state.shape != (14,) or not np.all(np.isfinite(state)):
        raise ValueError(f"episode {spec.episode} frame {row.get('frame_index')} has invalid raw state")
    digest.update(b"state<float32>\0" + state.tobytes(order="C"))
    for name in IMAGE_COLUMNS:
        payload = row.get(name)
        if not isinstance(payload, dict) or not isinstance(payload.get("bytes"), bytes):
            raise ValueError(f"episode {spec.episode} frame {row.get('frame_index')} has invalid {name}")
        path_value = payload.get("path")
        if path_value is not None and not isinstance(path_value, str):
            raise ValueError(f"episode {spec.episode} frame {row.get('frame_index')} has invalid {name}.path")
        digest.update(name.encode() + b"\0")
        digest.update((path_value or "").encode() + b"\0")
        digest.update(payload["bytes"])


def _max_array_difference(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if left_array.shape != right_array.shape:
        raise ValueError(f"array shape mismatch: {left_array.shape} != {right_array.shape}")
    return float(np.max(np.abs(left_array - right_array)))


def _memory_independence_smoke(
    runtime: _Run5Runtime, observation: Any, fresh_output: Mapping[str, Any], fresh_state: Any
) -> dict[str, Any]:
    """Verify W/K/V do not change when only the incoming memory state changes."""
    written_state, _ = runtime._write(fresh_state, fresh_output["write_tokens"])
    jax.block_until_ready(written_state)
    state_change = runtime.state_max_abs_difference(written_state, fresh_state)
    if not math.isfinite(state_change) or state_change <= 0.0:
        raise RuntimeError(f"scratch write failed to create a nonempty memory state: diff={state_change}")
    changed_output = runtime._interface(observation, written_state)
    jax.block_until_ready(changed_output)
    differences = {
        name: _max_array_difference(fresh_output[key], changed_output[key])
        for name, key in (
            ("writer", "write_tokens"),
            ("key", "write_keys"),
            ("value", "write_values"),
        )
    }
    failures = {name: value for name, value in differences.items() if value > MEMORY_INDEPENDENCE_ATOL}
    if failures:
        raise RuntimeError(
            f"writer features depend on incoming memory state above atol={MEMORY_INDEPENDENCE_ATOL}: {failures}"
        )
    return {
        "checked_on_first_fixed_padded_batch": True,
        "scratch_written_state_max_abs_change_from_m0": state_change,
        "fresh_vs_nonempty_max_abs_difference": differences,
        "absolute_tolerance": MEMORY_INDEPENDENCE_ATOL,
        "scratch_written_state_discarded": True,
        "production_returned_state_threaded": False,
    }


def _episode_feature_arrays(
    specs: Sequence[EpisodeSpec], frame_arrays: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    episode_ids = np.asarray(frame_arrays["frame_episode"], dtype=np.int64)
    phases = np.asarray(frame_arrays["frame_phase"], dtype=np.int8)
    output: dict[str, np.ndarray] = {
        "episode_ids": np.asarray([spec.episode for spec in specs], dtype=np.int64),
        "episode_labels": np.asarray([spec.label for spec in specs], dtype=np.int8),
        "episode_prompt_ids": np.asarray([PROMPTS.index(spec.prompt) for spec in specs], dtype=np.int8),
        "episode_heldout": np.asarray([spec.heldout for spec in specs], dtype=bool),
        "episode_evidence_counts": np.asarray([len(spec.evidence_frames) for spec in specs], dtype=np.int64),
        "episode_approach_counts": np.asarray([len(spec.approach_frames) for spec in specs], dtype=np.int64),
    }
    prompt = np.eye(len(PROMPTS), dtype=np.float32)[output["episode_prompt_ids"]]
    output["episode_prompt_evidence"] = prompt
    output["episode_prompt_approach"] = prompt.copy()
    for stream in ("writer", "value", "key", "state"):
        frames = np.asarray(frame_arrays[f"frame_{stream}"], dtype=np.float32)
        for phase_name, phase_id in (("approach", 0), ("evidence", 1)):
            means = []
            for spec in specs:
                mask = (episode_ids == spec.episode) & (phases == phase_id)
                expected = len(spec.approach_frames if phase_id == 0 else spec.evidence_frames)
                if int(np.sum(mask)) != expected:
                    raise ValueError(
                        f"episode {spec.episode} {phase_name} extraction count changed: "
                        f"expected {expected}, got {int(np.sum(mask))}"
                    )
                means.append(np.mean(frames[mask], axis=0, dtype=np.float32))
            # Deliberately no second normalization after the per-frame online-exact normalization.
            output[f"episode_{stream}_{phase_name}"] = np.stack(means).astype(np.float32)
    return output


def _extract_features(
    runtime: _Run5Runtime, args: Args, specs: Sequence[EpisodeSpec]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected_specs = [spec for spec in specs if not args.smoke_only or spec.episode in SMOKE_EPISODES]
    if args.smoke_only:
        cells = {spec.cell for spec in selected_specs}
        if tuple(spec.episode for spec in selected_specs) != SMOKE_EPISODES or len(cells) != 4:
            raise ValueError("smoke subset must be the fixed four-cell-balanced episodes 0,7,30,45")

    collected: dict[str, list[Any]] = defaultdict(list)
    selected_digest = hashlib.sha256()
    independence: dict[str, Any] | None = None
    started = time.monotonic()
    batches = 0
    for spec_number, spec in enumerate(selected_specs, start=1):
        phase_frames = _selected_frames(spec, smoke_only=args.smoke_only)
        ordered_frames = tuple(frame for _phase, frame in phase_frames)
        rows = _read_selected_rows(spec.parquet, ordered_frames)
        entries = []
        for (phase, frame), row in zip(phase_frames, rows, strict=True):
            observed_evidence = int(row["task_index"]) == EXPECTED_EVIDENCE_TASK_INDEX
            if int(row["episode_index"]) != spec.episode or observed_evidence != (phase == "evidence"):
                raise ValueError(f"episode {spec.episode} selected row phase identity changed at frame {frame}")
            _update_selected_input_hash(selected_digest, spec=spec, phase=phase, row=row)
            entries.append((phase, frame, row))

        for start in range(0, len(entries), args.batch_size):
            padded, live = _pad_batch(entries[start : start + args.batch_size], args.batch_size)
            observations = []
            model_states = []
            for _phase, frame, row in padded:
                observation, model_state = runtime.observation(row, frame, spec.prompt)
                observations.append(observation)
                model_states.append(model_state)
            batched_observation = runtime.batch_observations(observations)
            fresh_state = runtime.model.memory.init_state(args.batch_size)
            output = runtime._interface(batched_observation, fresh_state)
            jax.block_until_ready(output)
            if independence is None:
                independence = _memory_independence_smoke(
                    runtime, batched_observation, output, fresh_state
                )

            pooled = {
                "writer": _online_pool(output["write_tokens"]),
                "value": _online_pool(output["write_values"]),
                "key": _online_pool(output["write_keys"]),
            }
            normalized_states = _l2_rows(np.stack(model_states))
            for index, (phase, frame, _row) in enumerate(padded[:live]):
                collected["frame_episode"].append(spec.episode)
                collected["frame_index"].append(frame)
                collected["frame_phase"].append(1 if phase == "evidence" else 0)
                for stream in PRIMARY_STREAMS:
                    collected[f"frame_{stream}"].append(pooled[stream][index])
                collected["frame_state"].append(normalized_states[index])
            batches += 1
        print(
            f"episode {spec.episode:02d} {spec.prompt} / {spec.side}: "
            f"{len(phase_frames)} frames ({spec_number}/{len(selected_specs)})",
            flush=True,
        )

    if independence is None:
        raise AssertionError("no extraction batch was evaluated")
    arrays = {
        "frame_episode": np.asarray(collected["frame_episode"], dtype=np.int64),
        "frame_index": np.asarray(collected["frame_index"], dtype=np.int64),
        "frame_phase": np.asarray(collected["frame_phase"], dtype=np.int8),
        "frame_writer": np.stack(collected["frame_writer"]).astype(np.float32),
        "frame_value": np.stack(collected["frame_value"]).astype(np.float32),
        "frame_key": np.stack(collected["frame_key"]).astype(np.float32),
        "frame_state": np.stack(collected["frame_state"]).astype(np.float32),
    }
    expected_rows = 2 * (len(SMOKE_EPISODES) if args.smoke_only else EXPECTED_EVIDENCE_FRAMES)
    if len(arrays["frame_episode"]) != expected_rows:
        raise ValueError(f"extracted frame count changed: expected {expected_rows}, got {len(arrays['frame_episode'])}")
    for name, value in arrays.items():
        if value.shape[0] != expected_rows or not np.all(np.isfinite(value)):
            raise ValueError(f"invalid extracted array {name}: {value.shape}")

    if not args.smoke_only:
        arrays.update(_episode_feature_arrays(specs, arrays))
    contract = {
        "mode": "smoke" if args.smoke_only else "full",
        "fixed_padded_batch_size": args.batch_size,
        "padded_outputs_discarded": True,
        "model_calls": batches,
        "fresh_independent_m0_per_frame_row": True,
        "writes_during_production_extraction": False,
        "returned_state_threaded": False,
        "feature_formula": "float32 L2(mean_slots(tokens)), epsilon=1e-12",
        "state_feature_formula": "float32 L2(exact transformed model-visible state), epsilon=1e-12",
        "episode_formula": "mean(per-frame feature), no second normalization",
        "evidence_selector": "unshifted parquet task_index == 4",
        "approach_selector": "equal-size immediately preceding contiguous window",
        "selected_frame_rows": expected_rows,
        "selected_input_sha256": selected_digest.hexdigest(),
        "memory_state_independence_smoke": independence,
        "elapsed_seconds": time.monotonic() - started,
    }
    return arrays, contract


def _score_rows(
    indices: Sequence[int], labels: np.ndarray, evidence: np.ndarray, approach: np.ndarray
) -> list[dict[str, Any]]:
    return [
        {
            "episode": int(index),
            "truth_side": LABEL_TO_SIDE[int(labels[index])],
            "truth_label": int(labels[index]),
            "evidence_logit_right_minus_left": float(evidence[position]),
            "approach_logit_right_minus_left": float(approach[position]),
            "evidence_prediction": LABEL_TO_SIDE[int(evidence[position] > 0.0)],
            "approach_prediction": LABEL_TO_SIDE[int(approach[position] > 0.0)],
            "truth_aligned_evidence_minus_approach": float(
                (2 * labels[index] - 1) * (evidence[position] - approach[position])
            ),
        }
        for position, index in enumerate(indices)
    ]


def _online_head_report(
    runtime: _Run5Runtime,
    arrays: Mapping[str, np.ndarray],
    labels: np.ndarray,
    train_indices: Sequence[int],
) -> dict[str, Any]:
    evidence_logits = runtime._online_writer_head(jnp.asarray(arrays["episode_writer_evidence"]))
    approach_logits = runtime._online_writer_head(jnp.asarray(arrays["episode_writer_approach"]))
    jax.block_until_ready((evidence_logits, approach_logits))
    evidence_logits = np.asarray(evidence_logits, dtype=np.float64)
    approach_logits = np.asarray(approach_logits, dtype=np.float64)
    if evidence_logits.shape != (60, 2) or approach_logits.shape != (60, 2):
        raise ValueError(
            f"stored online writer head returned invalid logits: {evidence_logits.shape}/{approach_logits.shape}"
        )
    evidence = evidence_logits[:, 1] - evidence_logits[:, 0]
    approach = approach_logits[:, 1] - approach_logits[:, 0]
    train = np.asarray(train_indices, dtype=np.int64)
    heldout = np.asarray(HELDOUT_EPISODES, dtype=np.int64)
    return {
        "semantics": (
            "checkpoint's accumulated two-class ladder_writer_head; right-minus-left logit; "
            "not refitted by this evaluator"
        ),
        "parameter_source": runtime.parameter_provenance["parameter_source"],
        "training_56": {
            "evidence_metrics": _binary_metrics(labels[train], evidence[train]),
            "approach_same_head_metrics": _binary_metrics(labels[train], approach[train]),
            "paired_evidence_minus_approach": _paired_evidence_minus_approach(
                labels[train], evidence[train], approach[train]
            ),
        },
        "heldout_4": {
            "evidence_metrics": _binary_metrics(labels[heldout], evidence[heldout]),
            "approach_same_head_metrics": _binary_metrics(labels[heldout], approach[heldout]),
            "paired_evidence_minus_approach": _paired_evidence_minus_approach(
                labels[heldout], evidence[heldout], approach[heldout]
            ),
            "episodes": _score_rows(
                heldout.tolist(), labels, evidence[heldout], approach[heldout]
            ),
        },
        "all_episode_evidence_logits": evidence.tolist(),
        "all_episode_approach_logits": approach.tolist(),
    }


def _analyze_full(
    runtime: _Run5Runtime,
    args: Args,
    specs: Sequence[EpisodeSpec],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    train_indices, folds = _validate_design(specs)
    labels = np.asarray(arrays["episode_labels"], dtype=np.int64)
    prompts = [spec.prompt for spec in specs]
    shuffles = _within_prompt_shuffles(
        labels,
        prompts,
        train_indices,
        repeats=args.null_repeats,
        seed=args.seed,
    )
    stream_reports = {}
    heldout = tuple(HELDOUT_EPISODES)
    for stream in STREAMS:
        evidence_features = np.asarray(arrays[f"episode_{stream}_evidence"], dtype=np.float64)
        approach_features = np.asarray(arrays[f"episode_{stream}_approach"], dtype=np.float64)
        oof = _oof_probe(evidence_features, approach_features, labels, folds)
        final = _fit_all_then_heldout(
            evidence_features,
            approach_features,
            labels,
            train_indices,
            heldout,
        )
        oof_indices = oof.pop("test_indices").tolist()
        oof_evidence = oof.pop("evidence_scores")
        oof_approach = oof.pop("approach_scores")
        # Fold membership is identical for every stream and is recorded once at report root.
        oof.pop("folds")
        heldout_indices = final.pop("indices").tolist()
        heldout_evidence = final.pop("evidence_scores")
        heldout_approach = final.pop("approach_scores")
        stream_reports[stream] = {
            "role": "primary writer pathway" if stream in PRIMARY_STREAMS else "baseline",
            "feature_width": int(evidence_features.shape[1]),
            "fresh_balanced_oof": {
                **oof,
                "episodes": _score_rows(oof_indices, labels, oof_evidence, oof_approach),
            },
            "fit_all_56_test_exact_heldout_4": {
                **final,
                "episodes": _score_rows(
                    heldout_indices, labels, heldout_evidence, heldout_approach
                ),
            },
            "within_prompt_shuffled_label_null": _null_summary(
                evidence_features, approach_features, shuffles, folds
            ),
        }
        print(
            f"probe {stream:>6}: OOF acc={oof['evidence_metrics']['accuracy']:.3f} "
            f"AUC={oof['evidence_metrics']['auc']:.3f}; "
            f"heldout acc={final['evidence_metrics']['accuracy']:.3f}",
            flush=True,
        )

    return {
        "probe_protocol": {
            "fit_unit": "one equal-weight episode feature",
            "feature_aggregation": "mean of per-frame features; no second normalization",
            "classifier": "binary logistic regression, right=1, left=0",
            "fresh_initialization": "all weights and bias exactly zero for every fold/source/checkpoint",
            "standardization": "evidence-fit episodes only; reused unchanged for matched approach",
            "l2": PROBE_L2,
            "steps": PROBE_STEPS,
            "learning_rate": PROBE_LR,
            "folds": [dataclasses.asdict(fold) for fold in folds],
            "heldout_excluded_from_every_fit": True,
            "train_episode_count": len(train_indices),
            "heldout_episodes": list(heldout),
            "null": {
                "kind": "labels permuted only within each instruction among the 56 fit episodes",
                "seed": args.seed,
                "repeats": args.null_repeats,
                "identical_permutations_reused_for_every_stream": True,
            },
        },
        "fresh_probe_streams": stream_reports,
        "stored_online_writer_head": _online_head_report(
            runtime, arrays, labels, train_indices
        ),
    }


def _feature_manifest(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(arrays.items())
    }


def _base_report(
    runtime: _Run5Runtime,
    args: Args,
    specs: Sequence[EpisodeSpec],
    dataset_provenance: Mapping[str, Any],
    extraction: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    dataset = dict(dataset_provenance)
    dataset["selected_input_sha256"] = extraction["selected_input_sha256"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "smoke_complete" if args.smoke_only else "complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_step_label": int(args.checkpoint.name),
        "config": args.config,
        "parameter_source": args.parameter_source,
        "output_namespace": str(args.artifact_dir),
        "source_provenance": runtime.source_provenance,
        "checkpoint_metadata": runtime.checkpoint_info,
        "checkpoint_origin": runtime.checkpoint_origin,
        "parameter_provenance": runtime.parameter_provenance,
        "normalization_asset_provenance": runtime.norm_provenance,
        "tokenizer_asset_provenance": runtime.tokenizer_provenance,
        "dataset_provenance": dataset,
        "design": {
            "episodes": len(specs),
            "episode_ids": [spec.episode for spec in specs],
            "heldout_episodes": list(HELDOUT_EPISODES),
            "heldout_cells": {str(key): list(value) for key, value in EXPECTED_HELDOUT_CELLS.items()},
            "evidence_task_index": EXPECTED_EVIDENCE_TASK_INDEX,
            "evidence_label": EXPECTED_EVIDENCE_LABEL,
            "all_evidence_frames": sum(len(spec.evidence_frames) for spec in specs),
            "all_matched_approach_frames": sum(len(spec.approach_frames) for spec in specs),
            "smoke_subset_episodes": list(SMOKE_EPISODES) if args.smoke_only else None,
        },
        "extraction_contract": extraction,
        "feature_manifest": _feature_manifest(arrays),
        "counterfactual_prompt": {
            "status": "deferred",
            "reason": (
                "not part of the pre-registered primary fixed writer/approach protocol; adding it "
                "would double extraction and delay checkpoint trend measurement"
            ),
        },
        "interpretation_limits": [
            (
                "the eight within-prompt shuffles describe this fixed estimator's null; they are "
                "not a calibrated hypothesis test"
            ),
            (
                "the state baseline uses the exact checkpoint-transformed model-visible state, "
                "then per-frame L2 normalization"
            ),
            (
                "the immediately-pre-evidence window is temporally matched but is not assumed to "
                "be visually side-neutral; cameras may already contain anticipatory cues"
            ),
        ],
    }


def run(args: Args) -> dict[str, Any]:
    runtime = _Run5Runtime(args)
    specs, dataset_provenance = _load_specs(runtime, args)
    arrays, extraction = _extract_features(runtime, args, specs)
    report = _base_report(
        runtime, args, specs, dataset_provenance, extraction, arrays
    )
    if args.smoke_only:
        report["smoke"] = {
            "balanced_fixed_subset": list(SMOKE_EPISODES),
            "probe_fit_performed": False,
            "full_mode_still_requires_60_episodes_and_3112_evidence_frames": True,
            "memory_state_independence_passed": True,
        }
    else:
        report.update(_analyze_full(runtime, args, specs, arrays))

    with _AtomicOutput(args.artifact_dir) as output:
        feature_identity = output.write_npz(arrays)
        report["features_npz_identity"] = feature_identity
        report_identity = output.write_report(report)
        output.complete({"features.npz": feature_identity, "report.json": report_identity})
    print(f"COMPLETE: {args.artifact_dir}", flush=True)
    return report


def main(argv: list[str] | None = None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
