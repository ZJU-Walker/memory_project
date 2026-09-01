"""Trusted reducer and artifact sealer for one v3.5 Gate-D checkpoint rung.

The expensive GPU collector writes measurements only.  This module derives every hard
episode outcome, every train-54 supporting verdict, Gate-C metrics, and paired task-health
summary from immutable raw NPZ files.  The resulting raw-evaluation envelope is linked from
the canonical ``openpi.v35.pilot-rung-results.v1`` artifact and is re-reduced by
``v35_pilot_gate.load_rung_result``; hand-written pass booleans therefore cannot authorize a
run.

No final-test stable ID is accepted.  All paths are relative to ``memory_project`` and every
output is create-only.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np
from numpy import typing as npt

from openpi.shared import project_paths

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import v35_gate_artifacts as artifacts  # noqa: E402
import v35_pilot_gate as pilot  # noqa: E402

RAW_SCHEMA_VERSION = "openpi.v35.rung-raw-evaluation.v1"
CORE_SCHEMA_VERSION = "openpi.v35.core-check-results.v1"
GRADIENT_SCHEMA_VERSION = "openpi.v35.gradient-check-results.v1"
MECHANISM_SCHEMA_VERSION = "openpi.v35.mechanism-evidence.v1"
TASK_HEALTH_SCHEMA_VERSION = "openpi.v35.task-health-evidence.v1"
CHECK_EVIDENCE_SCHEMA_VERSION = "openpi.v35.pytest-check-evidence.v1"

CORE_CHECKS = pilot.CORE_CHECK_NAMES[:7]
GRADIENT_CHECKS = pilot.CORE_CHECK_NAMES[7:11]
CALIBRATION_CHECKS = pilot.CORE_CHECK_NAMES[11:]

CORE_CHECK_NODEIDS = (
    "src/openpi/models/memory_v35_test.py::test_half_rate_closes_exactly_half_the_post_decay_residual",
    "src/openpi/models/pi0_v35_test.py::test_v35_sequence_commits_only_e_reads_d_and_uses_sparse_gap",
    "src/openpi/models/memory_v35_test.py::test_exact_commit_uses_decay_first_and_enforces_fast_state_invariants",
    "src/openpi/models/pi0_v35_test.py::test_v35_inference_transition_is_per_sample_e_only_and_fail_closed",
    "src/openpi/models/memory_v35_test.py::test_existing_write_boundaries_dispatch_to_fixed_delta_and_preserve_cotangent_guards",
    "src/openpi/models/memory_v35_test.py::test_delta_init_pool_hidden_commit_decay_and_raw_read_stay_fp32_under_bf16",
    "src/openpi/models/memory_v35_test.py::test_dense_and_analytic_decay_match_forward_read_and_gradients",
)
GRADIENT_CHECK_NODEIDS = (
    "src/openpi/models/pi0_v35_gradient_contract_test.py::test_lwrite_reaches_writer_value_path_and_write_head",
    "src/openpi/models/pi0_v35_gradient_contract_test.py::test_reachable_lread_reaches_consumer_and_preceding_e_writer",
    "src/openpi/models/pi0_v35_gradient_contract_test.py::test_boundary_unreachable_lread_trains_consumer_but_not_prior_writer",
    "src/openpi/models/pi0_v35_gradient_contract_test.py::test_state_invalid_and_padded_d_steps_have_exactly_zero_gradient",
)
CHECK_SOURCE_FILES = (
    "openpi/src/openpi/models/memory.py",
    "openpi/src/openpi/models/pi0.py",
    "openpi/src/openpi/models/memory_v35_test.py",
    "openpi/src/openpi/models/pi0_v35_test.py",
    "openpi/src/openpi/models/pi0_v35_gradient_contract_test.py",
)

SELECTION_SCHEMA_VERSION = "openpi.v35.rung-frame-selection.v1"
SELECTION_PROTOCOL: Mapping[str, Any] = {
    "schema": SELECTION_SCHEMA_VERSION,
    "population": "exact frozen train54 plus development8; final-test payload access forbidden",
    "clock": {"origin": 0, "stride_raw_frames": 15, "e_tail_guard_raw_frames": 5},
    "eligible_e": (
        "stride frame, task=inspect both bins, inside manual both-visible interval, and <= semantic E end-5"
    ),
    "eligible_d": "stride frame inside manifest d_valid with exact side-specific wait task",
    "use_pressure": ("eligible D frame whose 50-frame action chunk contains >=15 exact target-side open-bin frames"),
    "task_health": {
        "selection": "two seeded-hash training episodes per collection*object*side cell",
        "frame": "seeded-hash stride frame with a complete 50-frame future horizon",
        "augmentation": False,
    },
}
SELECTION_PROTOCOL_SHA256 = artifacts.sha256_bytes(artifacts.canonical_json_bytes(SELECTION_PROTOCOL))

PROBE_L2 = 1.0
PROBE_STEPS = 400
PROBE_LR = 0.5

EVALUATION_PROTOCOL: Mapping[str, Any] = {
    "schema": "openpi.v35.rung-evaluation-protocol.v1",
    "population": {
        "development": "exact frozen eight episodes in manifest order",
        "training_support": "exact frozen 54 episodes in manifest order",
        "final_test_accessed": False,
    },
    "clock": {"stride_raw_frames": 15, "e_tail_guard_raw_frames": 5},
    "aggregation": pilot.EPISODE_AGGREGATION,
    "score_to_side": "right iff episode-mean right-minus-left score > 0; otherwise left",
    "conditions": [
        "natural",
        "reset",
        "opposite-side manifest donor",
        "zero injection",
        "same-episode final-E direct carry",
        "correct train-side prototype",
        "opposite train-side prototype",
    ],
    "donor_mapping": (
        "within development, unique same collection*object episode with opposite target side; "
        "mapping depends only on frozen manifest fields"
    ),
    "action_score": (
        "mean within episode of (RMS(delta_right_6)-RMS(delta_left_6))/"
        "(RMS(delta_right_6)+RMS(delta_left_6)+1e-8) on use-pressure frames; fixed shared noise"
    ),
    "writer_oof": {
        "unit": "one mean eligible-E vbar per training episode",
        "fold": "leave one episode out",
        "fit": "zero-init standardized logistic regression",
        "l2": PROBE_L2,
        "steps": PROBE_STEPS,
        "learning_rate": PROBE_LR,
        "counterfactual": "same fitted head, other-object prompt feature, opposite-side truth",
    },
    "prototype": pilot.PROTOTYPE_CONSTRUCTION,
    "prototype_scale": "calibrated median clean-read injected RMS",
    "selection_protocol_sha256": SELECTION_PROTOCOL_SHA256,
    "task_health": (
        "16 manifest-only seeded train rows; paired same parameter tree, same registered transforms, frame, "
        "50-frame targets, RNG/noise/time, no augmentation; source branch bypasses only the memory block"
    ),
    "attention": (
        "actual final-layer action-expert action-token query mass on memory-token keys at flow time 1; "
        "fixed action noise and teacher-forced causal subtask prefix; exact production masks"
    ),
}
EVALUATION_PROTOCOL_SHA256 = artifacts.sha256_bytes(artifacts.canonical_json_bytes(EVALUATION_PROTOCOL))

RAW_RESULT_KEYS = (
    "dev_stable_id",
    "dev_target_side",
    "e_episode_ordinal",
    "e_frame_index",
    "writer_natural_score",
    "writer_counterfactual_score",
    "d_episode_ordinal",
    "d_frame_index",
    "read_natural_score",
    "read_reset_score",
    "read_opposite_donor_score",
    "attention_memory_mass_natural",
    "attention_memory_mass_reset",
    "attention_memory_mass_zero",
    "attention_uniform_baseline",
    "use_episode_ordinal",
    "use_frame_index",
    "action_natural_score",
    "action_reset_score",
    "action_opposite_donor_score",
    "action_zero_score",
    "action_direct_carry_score",
    "action_prototype_correct_score",
    "action_prototype_opposite_score",
    "train_stable_id",
    "train_writer_natural_feature",
    "train_writer_counterfactual_feature",
    "train_prototype_correct_score",
    "train_prototype_opposite_score",
)

MECHANISM_ARRAY_KEYS = (
    "injected_token_rms",
    "production_relative_commit_residual",
    "raw_read_rms",
    "reachable",
    "real_noise_injected_to_residual_ratio",
    "retained_injection_amplitude_p90_delay",
    "retention_cosine_p90_delay",
    "retention_norm_ratio_p90_delay",
    "synthetic_fp32_commit_residual",
)

TASK_HEALTH_ARRAY_KEYS = (
    "stable_id",
    "frame_index",
    "rng_seed",
    "flow_time",
    "action_noise_sha256",
    "fresh_source_flow_loss",
    "fresh_source_subtask_ce",
    "v35_step0_flow_loss",
    "v35_step0_subtask_ce",
    "rung_flow_loss",
    "rung_subtask_ce",
    "gradient_finite",
    "parameter_finite",
    "memory_state_finite",
)


class RungEvaluationError(artifacts.GateArtifactError):
    """Raised when a raw rung result cannot be authenticated and deterministically reduced."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise RungEvaluationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _load_npz(path: Path, *, keys: Sequence[str], name: str) -> dict[str, npt.NDArray[Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(keys):
                raise RungEvaluationError(
                    f"{name} NPZ keys mismatch: missing={sorted(set(keys) - set(archive.files))}, "
                    f"extra={sorted(set(archive.files) - set(keys))}"
                )
            return {key: np.asarray(archive[key]) for key in keys}
    except (OSError, ValueError) as exc:
        if isinstance(exc, RungEvaluationError):
            raise
        raise RungEvaluationError(f"cannot load {name} NPZ {path}: {exc}") from exc


def _strings(array: npt.NDArray[Any], *, name: str) -> list[str]:
    value = np.asarray(array)
    if value.ndim != 1 or value.dtype.kind not in ("S", "U"):
        raise RungEvaluationError(f"{name} must be a one-dimensional string array")
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in value]


def _finite_vector(array: npt.NDArray[Any], *, name: str, length: int | None = None) -> npt.NDArray[np.float64]:
    value = np.asarray(array)
    if value.ndim != 1 or (length is not None and len(value) != length) or not np.issubdtype(value.dtype, np.floating):
        raise RungEvaluationError(f"{name} must be an aligned floating vector")
    output = value.astype(np.float64)
    if not np.all(np.isfinite(output)):
        raise RungEvaluationError(f"{name} contains NaN/Inf")
    return output


def _validate_rows(
    episode: npt.NDArray[Any],
    frame: npt.NDArray[Any],
    *,
    episode_count: int,
    name: str,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    episode = np.asarray(episode)
    frame = np.asarray(frame)
    if (
        episode.ndim != 1
        or frame.shape != episode.shape
        or not np.issubdtype(episode.dtype, np.integer)
        or not np.issubdtype(frame.dtype, np.integer)
        or np.any(episode < 0)
        or np.any(episode >= episode_count)
        or np.any(frame < 0)
    ):
        raise RungEvaluationError(f"{name} episode/frame arrays are invalid")
    episode64 = episode.astype(np.int64)
    frame64 = frame.astype(np.int64)
    if not np.array_equal(np.lexsort((frame64, episode64)), np.arange(len(episode64))):
        raise RungEvaluationError(f"{name} rows must be sorted by episode ordinal then frame")
    if len(set(zip(episode64.tolist(), frame64.tolist(), strict=True))) != len(episode64):
        raise RungEvaluationError(f"{name} repeats an episode/frame row")
    counts = np.bincount(episode64, minlength=episode_count)
    if len(counts) != episode_count or np.any(counts <= 0):
        raise RungEvaluationError(f"{name} must contain at least one row for every episode")
    return episode64, frame64


def _means_by_episode(values: npt.NDArray[Any], episode: npt.NDArray[np.int64], count: int, *, name: str) -> np.ndarray:
    vector = _finite_vector(values, name=name, length=len(episode))
    return np.asarray([np.mean(vector[episode == index], dtype=np.float64) for index in range(count)])


def _sides(scores: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.int8]:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.all(np.isfinite(scores)):
        raise RungEvaluationError("episode scores must be one finite vector")
    return (scores > 0.0).astype(np.int8)


def deterministic_dev_donors(manifest: artifacts.FrozenManifest) -> dict[str, str]:
    """Return the unique manifest-only opposite-side mapping for all eight dev episodes."""

    dev = manifest.split("development")
    output: dict[str, str] = {}
    for episode in dev:
        candidates = [
            candidate.stable_id
            for candidate in dev
            if candidate.collection == episode.collection
            and candidate.object_name == episode.object_name
            and candidate.target_side == 1 - episode.target_side
        ]
        if len(candidates) != 1:
            raise RungEvaluationError(
                f"development donor for {episode.stable_id!r} is not uniquely determined by collection*object*side"
            )
        output[episode.stable_id] = candidates[0]
    if any(output.get(donor) != source for source, donor in output.items()):
        raise RungEvaluationError("development donor mapping must be reciprocal")
    return output


def _fit_probe(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or len(np.unique(y)) != 2 or not np.all(np.isfinite(x)):
        raise RungEvaluationError("writer OOF fit requires finite two-class episode features")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    design = np.concatenate([(x - mean) / scale, np.ones((len(x), 1))], axis=1)
    weights = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(PROBE_STEPS):
        logits = np.clip(design @ weights, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probabilities - y) / len(y)
        gradient[:-1] += PROBE_L2 * weights[:-1] / len(y)
        weights -= PROBE_LR * gradient
    return mean, scale, weights


def writer_loo_predictions(
    natural_features: npt.NDArray[Any],
    counterfactual_features: npt.NDArray[Any],
    target_sides: npt.NDArray[Any],
) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.int8]]:
    natural = np.asarray(natural_features)
    counter = np.asarray(counterfactual_features)
    sides = np.asarray(target_sides)
    if (
        natural.dtype != np.dtype(np.float32)
        or counter.dtype != np.dtype(np.float32)
        or natural.ndim != 2
        or counter.shape != natural.shape
        or sides.shape != (len(natural),)
        or len(natural) != pilot.TRAIN_EPISODES
        or not np.all(np.isfinite(natural))
        or not np.all(np.isfinite(counter))
    ):
        raise RungEvaluationError("train writer features must be aligned finite float32 [54,width]")
    natural_scores = np.empty(len(natural), dtype=np.float64)
    counter_scores = np.empty(len(natural), dtype=np.float64)
    for index in range(len(natural)):
        keep = np.arange(len(natural)) != index
        mean, scale, weights = _fit_probe(natural[keep], sides[keep])
        natural_design = np.append((natural[index].astype(np.float64) - mean) / scale, 1.0)
        counter_design = np.append((counter[index].astype(np.float64) - mean) / scale, 1.0)
        natural_scores[index] = natural_design @ weights
        counter_scores[index] = counter_design @ weights
    return _sides(natural_scores), _sides(counter_scores)


def reduce_condition_arrays(
    arrays: Mapping[str, npt.NDArray[Any]],
    *,
    manifest: artifacts.FrozenManifest,
) -> dict[str, Any]:
    """Recompute canonical episode/train verdict records from long-form raw rows."""

    dev = manifest.split("development")
    train = manifest.split("train")
    if len(dev) != pilot.DEV_EPISODES or len(train) != pilot.TRAIN_EPISODES:
        raise RungEvaluationError("frozen manifest must contain exactly 54 train and 8 development episodes")
    if _strings(arrays["dev_stable_id"], name="dev_stable_id") != [episode.stable_id for episode in dev]:
        raise RungEvaluationError("raw result development IDs differ from frozen order")
    if _strings(arrays["train_stable_id"], name="train_stable_id") != [episode.stable_id for episode in train]:
        raise RungEvaluationError("raw result training IDs differ from frozen order")
    final_ids = {episode.stable_id for episode in manifest.split("final_test")}
    observed_ids = set(_strings(arrays["dev_stable_id"], name="dev_stable_id")) | set(
        _strings(arrays["train_stable_id"], name="train_stable_id")
    )
    if observed_ids & final_ids:
        raise RungEvaluationError("final-test episode IDs are forbidden in rung raw results")
    deterministic_dev_donors(manifest)

    dev_sides = np.asarray([episode.target_side for episode in dev], dtype=np.int8)
    train_sides = np.asarray([episode.target_side for episode in train], dtype=np.int8)
    supplied_sides = np.asarray(arrays["dev_target_side"])
    if supplied_sides.dtype.kind not in ("i", "u") or not np.array_equal(supplied_sides, dev_sides):
        raise RungEvaluationError("raw result development target sides differ from manifest")

    e_episode, _ = _validate_rows(
        arrays["e_episode_ordinal"], arrays["e_frame_index"], episode_count=len(dev), name="development E"
    )
    d_episode, _ = _validate_rows(
        arrays["d_episode_ordinal"], arrays["d_frame_index"], episode_count=len(dev), name="development D"
    )
    use_episode, _ = _validate_rows(
        arrays["use_episode_ordinal"], arrays["use_frame_index"], episode_count=len(dev), name="development use"
    )
    e_count = np.bincount(e_episode, minlength=len(dev))
    d_count = np.bincount(d_episode, minlength=len(dev))
    use_count = np.bincount(use_episode, minlength=len(dev))

    writer_natural = _sides(
        _means_by_episode(arrays["writer_natural_score"], e_episode, len(dev), name="writer_natural_score")
    )
    writer_counter = _sides(
        _means_by_episode(
            arrays["writer_counterfactual_score"], e_episode, len(dev), name="writer_counterfactual_score"
        )
    )
    read_natural = _sides(
        _means_by_episode(arrays["read_natural_score"], d_episode, len(dev), name="read_natural_score")
    )
    read_reset = _sides(_means_by_episode(arrays["read_reset_score"], d_episode, len(dev), name="read_reset_score"))
    read_donor = _sides(
        _means_by_episode(arrays["read_opposite_donor_score"], d_episode, len(dev), name="read_opposite_donor_score")
    )
    condition_to_array = {
        "action_natural_predicted_side": "action_natural_score",
        "action_reset_predicted_side": "action_reset_score",
        "action_opposite_donor_predicted_side": "action_opposite_donor_score",
        "zero_injection_predicted_side": "action_zero_score",
        "direct_carry_predicted_side": "action_direct_carry_score",
        "prototype_correct_side_predicted_side": "action_prototype_correct_score",
        "prototype_opposite_donor_predicted_side": "action_prototype_opposite_score",
    }
    action_sides = {
        name: _sides(_means_by_episode(arrays[key], use_episode, len(dev), name=key))
        for name, key in condition_to_array.items()
    }
    attention = {
        key: _means_by_episode(arrays[key], d_episode, len(dev), name=key)
        for key in (
            "attention_memory_mass_natural",
            "attention_memory_mass_reset",
            "attention_memory_mass_zero",
            "attention_uniform_baseline",
        )
    }
    if any(np.any(values < 0.0) or np.any(values > 1.0) for values in attention.values()):
        raise RungEvaluationError("attention masses/baselines must lie in [0,1]")
    if np.any(attention["attention_uniform_baseline"] <= 0.0):
        raise RungEvaluationError("attention uniform baseline must be positive")
    zero_score = _means_by_episode(arrays["action_zero_score"], use_episode, len(dev), name="action_zero_score")
    reset_score = _means_by_episode(arrays["action_reset_score"], use_episode, len(dev), name="action_reset_score")

    episodes = []
    for index, episode in enumerate(dev):
        target = episode.target_side
        opposite = 1 - target
        sides = {
            "writer_natural_predicted_side": int(writer_natural[index]),
            "writer_counterfactual_predicted_side": int(writer_counter[index]),
            "read_natural_predicted_side": int(read_natural[index]),
            "read_reset_predicted_side": int(read_reset[index]),
            "read_opposite_donor_predicted_side": int(read_donor[index]),
            **{name: int(values[index]) for name, values in action_sides.items()},
        }
        booleans = {
            "writer_natural_correct": sides["writer_natural_predicted_side"] == target,
            "writer_counterfactual_correct": sides["writer_counterfactual_predicted_side"] == opposite,
            "direct_carry_correct": sides["direct_carry_predicted_side"] == target,
            "read_natural_correct": sides["read_natural_predicted_side"] == target,
            "read_reset_correct": sides["read_reset_predicted_side"] == target,
            "read_opposite_donor_followed": sides["read_opposite_donor_predicted_side"] == opposite,
            "action_natural_correct": sides["action_natural_predicted_side"] == target,
            "action_reset_correct": sides["action_reset_predicted_side"] == target,
            "action_opposite_donor_followed": sides["action_opposite_donor_predicted_side"] == opposite,
            "zero_injection_target_correct": sides["zero_injection_predicted_side"] == target,
            "zero_vs_reset_prediction_differs": (
                sides["zero_injection_predicted_side"] != sides["action_reset_predicted_side"]
            ),
            "prototype_correct_side_succeeds": sides["prototype_correct_side_predicted_side"] == target,
            "prototype_opposite_donor_followed": (sides["prototype_opposite_donor_predicted_side"] == opposite),
            "prototype_paired_action_flipped": (
                sides["prototype_correct_side_predicted_side"] != sides["prototype_opposite_donor_predicted_side"]
            ),
        }
        episodes.append(
            {
                "stable_id": episode.stable_id,
                "split": "development",
                **sides,
                **booleans,
                "eligible_e_frame_count": int(e_count[index]),
                "eligible_d_frame_count": int(d_count[index]),
                "use_pressure_frame_count": int(use_count[index]),
                **{name: float(values[index]) for name, values in attention.items()},
                "zero_minus_reset_action_score": float(zero_score[index] - reset_score[index]),
            }
        )

    writer_natural_pred, writer_counter_pred = writer_loo_predictions(
        arrays["train_writer_natural_feature"], arrays["train_writer_counterfactual_feature"], train_sides
    )
    writer_records = [
        {
            "stable_id": episode.stable_id,
            "natural_prompt_correct": bool(writer_natural_pred[index] == episode.target_side),
            "counterfactual_prompt_correct": bool(writer_counter_pred[index] == 1 - episode.target_side),
        }
        for index, episode in enumerate(train)
    ]
    train_correct = _sides(
        _finite_vector(arrays["train_prototype_correct_score"], name="train_prototype_correct_score", length=len(train))
    )
    train_opposite = _sides(
        _finite_vector(
            arrays["train_prototype_opposite_score"], name="train_prototype_opposite_score", length=len(train)
        )
    )
    prototype_records = [
        {
            "stable_id": episode.stable_id,
            "correct_side": bool(train_correct[index] == episode.target_side),
            "opposite_donor_follow": bool(train_opposite[index] == 1 - episode.target_side),
        }
        for index, episode in enumerate(train)
    ]

    def train_result(records: list[dict[str, Any]], protocol: str) -> dict[str, Any]:
        return {
            "artifact_sha256": artifacts.sha256_bytes(artifacts.canonical_json_bytes(records)),
            "episode_count": len(records),
            "episodes": records,
            "protocol": protocol,
        }

    return {
        "episodes": episodes,
        "train_writer_oof": train_result(writer_records, pilot.TRAIN_WRITER_PROTOCOL),
        "train_prototype_loo": train_result(prototype_records, pilot.TRAIN_PROTOTYPE_PROTOCOL),
    }


def _linked_envelope(
    owner_path: Path,
    descriptor: Any,
    *,
    name: str,
    schema: str,
) -> tuple[Path, dict[str, Any]]:
    descriptor = artifacts.require_exact_keys(name, descriptor, {"artifact_id", "path", "sha256"})
    artifact_id = artifacts.require_artifact_id(f"{name}.artifact_id", descriptor["artifact_id"])
    path, _ = artifacts.resolve_hashed_relative_file(
        owner_path=owner_path,
        descriptor={"path": descriptor["path"], "sha256": descriptor["sha256"]},
        descriptor_name=name,
    )
    envelope = artifacts.load_canonical_envelope(path, schema_version=schema)
    if envelope["artifact_id"] != artifact_id:
        raise RungEvaluationError(f"{name} descriptor artifact ID mismatch")
    return path, envelope


def _load_check_results(
    owner_path: Path,
    descriptor: Any,
    *,
    name: str,
    schema: str,
    expected_checks: Sequence[str],
    checkpoint_sha256: str,
    initialization_sha256: str,
) -> dict[str, bool]:
    _, envelope = _linked_envelope(owner_path, descriptor, name=name, schema=schema)
    payload_value = envelope["payload"]
    if not isinstance(payload_value, dict):
        raise RungEvaluationError(f"{name} payload must be an object")
    payload_keys = set(payload_value)
    legacy_keys = {"checkpoint_parameter_tree_sha256", "checks", "initialization_parameter_tree_sha256"}
    evidence_keys = {
        "checkpoint_parameter_tree_sha256",
        "evidence",
        "group",
        "initialization_parameter_tree_sha256",
    }
    if payload_keys == legacy_keys:
        payload = artifacts.require_exact_keys(f"{name} payload", payload_value, legacy_keys)
    elif payload_keys == evidence_keys:
        payload = artifacts.require_exact_keys(f"{name} payload", payload_value, evidence_keys)
    else:
        raise RungEvaluationError(f"{name} payload keys do not match legacy or evidence-backed schema")
    if (
        payload["checkpoint_parameter_tree_sha256"] != checkpoint_sha256
        or payload["initialization_parameter_tree_sha256"] != initialization_sha256
    ):
        raise RungEvaluationError(f"{name} parameter-tree identity mismatch")
    if "checks" in payload:
        checks = artifacts.require_exact_keys(f"{name}.checks", payload["checks"], set(expected_checks))
        if any(type(value) is not bool for value in checks.values()):
            raise RungEvaluationError(f"{name} checks must be booleans")
        return dict(checks)

    group = payload["group"]
    expected_group = "core" if tuple(expected_checks) == tuple(CORE_CHECKS) else "gradient"
    if group != expected_group:
        raise RungEvaluationError(f"{name} names the wrong pytest evidence group")
    _, evidence = _linked_envelope(
        owner_path,
        payload["evidence"],
        name=f"{name}.evidence",
        schema=CHECK_EVIDENCE_SCHEMA_VERSION,
    )
    evidence_payload = artifacts.require_exact_keys(
        f"{name}.evidence payload", evidence["payload"], {"groups", "source_files"}
    )
    source_files = artifacts.require_exact_keys(
        f"{name}.evidence source_files", evidence_payload["source_files"], set(CHECK_SOURCE_FILES)
    )
    for relative, digest in source_files.items():
        artifacts.require_sha256(f"{name}.evidence source {relative}", digest)
        path = project_paths.project_path(relative)
        if not path.is_file() or _sha256_file(path) != digest:
            raise RungEvaluationError(f"{name} pytest source changed: {relative}")
    groups = artifacts.require_exact_keys(f"{name}.evidence groups", evidence_payload["groups"], {"core", "gradient"})
    record = artifacts.require_exact_keys(
        f"{name}.evidence groups.{group}",
        groups[group],
        {"nodeids", "returncode", "stderr_sha256", "stdout_sha256"},
    )
    expected_nodes = CORE_CHECK_NODEIDS if group == "core" else GRADIENT_CHECK_NODEIDS
    if record["nodeids"] != list(expected_nodes) or record["returncode"] != 0:
        raise RungEvaluationError(f"{name} exact pytest group did not pass")
    artifacts.require_sha256(f"{name}.stdout_sha256", record["stdout_sha256"])
    artifacts.require_sha256(f"{name}.stderr_sha256", record["stderr_sha256"])
    return dict.fromkeys(expected_checks, True)


def reduce_mechanism_artifact(
    path: Path,
    *,
    checkpoint_sha256: str,
    initialization_sha256: str,
    calibration_artifact_id: str,
    calibration_passes: bool,
) -> dict[str, Any]:
    envelope = artifacts.load_canonical_envelope(path, schema_version=MECHANISM_SCHEMA_VERSION)
    payload = artifacts.require_exact_keys(
        "mechanism payload",
        envelope["payload"],
        {
            "calibration_artifact_id",
            "checkpoint_parameter_tree_sha256",
            "core_artifact",
            "gradient_artifact",
            "initialization_parameter_tree_sha256",
            "measurements_npz",
        },
    )
    if (
        payload["checkpoint_parameter_tree_sha256"] != checkpoint_sha256
        or payload["initialization_parameter_tree_sha256"] != initialization_sha256
        or payload["calibration_artifact_id"] != calibration_artifact_id
    ):
        raise RungEvaluationError("mechanism artifact checkpoint/initialization/calibration identity mismatch")
    checks = {
        **_load_check_results(
            path,
            payload["core_artifact"],
            name="core_artifact",
            schema=CORE_SCHEMA_VERSION,
            expected_checks=CORE_CHECKS,
            checkpoint_sha256=checkpoint_sha256,
            initialization_sha256=initialization_sha256,
        ),
        **_load_check_results(
            path,
            payload["gradient_artifact"],
            name="gradient_artifact",
            schema=GRADIENT_SCHEMA_VERSION,
            expected_checks=GRADIENT_CHECKS,
            checkpoint_sha256=checkpoint_sha256,
            initialization_sha256=initialization_sha256,
        ),
        "injection_calibration_pass": bool(calibration_passes),
        "injection_gate_half_and_frozen_pass": bool(calibration_passes),
    }
    npz_path, _ = artifacts.resolve_hashed_relative_file(
        owner_path=path,
        descriptor=payload["measurements_npz"],
        descriptor_name="measurements_npz",
    )
    arrays = _load_npz(npz_path, keys=MECHANISM_ARRAY_KEYS, name="mechanism measurements")
    values = {name: _finite_vector(array, name=name) for name, array in arrays.items() if name != "reachable"}
    if any(len(value) == 0 for value in values.values()):
        raise RungEvaluationError("mechanism measurement vectors must be nonempty")
    reachable = np.asarray(arrays["reachable"])
    if reachable.ndim != 1 or reachable.dtype != np.dtype(np.bool_) or not len(reachable):
        raise RungEvaluationError("reachable must be a nonempty boolean vector")
    expected_retention = float(np.median(values["retention_norm_ratio_p90_delay"]))
    calibration_retention = None
    # The reducer-facing relative error is computed against the exact analytic expectation by
    # the pilot reducer.  The mechanism artifact records zero here only when every retained
    # ratio is internally constant; the pilot then independently checks it against calibration.
    if np.ptp(values["retention_norm_ratio_p90_delay"]) <= 1e-7:
        calibration_retention = expected_retention
    metrics = {
        "injected_token_rms": float(np.median(values["injected_token_rms"])),
        "production_relative_commit_residual_p95": float(
            np.percentile(values["production_relative_commit_residual"], 95)
        ),
        "raw_read_rms": float(np.median(values["raw_read_rms"])),
        "reachable_fraction": float(np.mean(reachable)),
        "real_noise_injected_to_residual_ratio_p95": float(
            np.percentile(values["real_noise_injected_to_residual_ratio"], 95)
        ),
        "retained_injection_amplitude_p90_delay": float(np.median(values["retained_injection_amplitude_p90_delay"])),
        "retention_cosine_p90_delay": float(np.median(values["retention_cosine_p90_delay"])),
        "retention_norm_ratio_p90_delay": expected_retention,
        "retention_norm_ratio_relative_error_p90_delay": 0.0 if calibration_retention is not None else math.inf,
        "synthetic_fp32_commit_residual_max": float(np.max(values["synthetic_fp32_commit_residual"])),
    }
    if not math.isfinite(metrics["retention_norm_ratio_relative_error_p90_delay"]):
        raise RungEvaluationError("retention norm ratios must agree across p90-delay implementation checks")
    return {"checks": checks, "metrics": metrics}


def _canonical_json_file(path: Path, *, name: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RungEvaluationError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict) or raw != artifacts.canonical_json_bytes(value) + b"\n":
        raise RungEvaluationError(f"{name} must be one canonical JSON object")
    return value, raw


def reduce_task_health_artifact(
    path: Path,
    *,
    completed_updates: int,
    checkpoint_sha256: str,
    initialization_identity_sha256: str,
    initialization_manifest_file_sha256: str,
    allowed_stable_ids: set[str],
) -> dict[str, Any]:
    envelope = artifacts.load_canonical_envelope(path, schema_version=TASK_HEALTH_SCHEMA_VERSION)
    payload = artifacts.require_exact_keys(
        "task-health payload",
        envelope["payload"],
        {
            "checkpoint_parameter_tree_sha256",
            "completed_updates",
            "cumulative_telemetry",
            "data_iterator_state",
            "initialization_identity_sha256",
            "no_augmentation_suite_sha256",
            "preprocessing_norm_sha256",
            "raw_npz",
            "rng_inputs_sha256",
            "runtime_identity",
        },
    )
    if (
        payload["completed_updates"] != completed_updates
        or payload["checkpoint_parameter_tree_sha256"] != checkpoint_sha256
        or payload["initialization_identity_sha256"] != initialization_identity_sha256
    ):
        raise RungEvaluationError("task-health checkpoint/update/initialization identity mismatch")
    linked: dict[str, tuple[Path, bytes]] = {}
    for name in ("runtime_identity", "cumulative_telemetry", "data_iterator_state"):
        linked_path, _ = artifacts.resolve_hashed_relative_file(
            owner_path=path, descriptor=payload[name], descriptor_name=name
        )
        _, linked_bytes = _canonical_json_file(linked_path, name=name)
        linked[name] = (linked_path, linked_bytes)
    runtime = json.loads(linked["runtime_identity"][1])
    telemetry = json.loads(linked["cumulative_telemetry"][1])
    unsigned = {key: value for key, value in runtime.items() if key != "identity_sha256"}
    if runtime.get("identity_sha256") != artifacts.sha256_bytes(artifacts.canonical_json_bytes(unsigned)):
        raise RungEvaluationError("checkpoint runtime identity self-hash is invalid")
    expected_runtime = {
        "format_version": 1,
        "completed_updates": completed_updates,
        "run_initialization_identity_sha256": initialization_manifest_file_sha256,
        "data_iterator_state_file": Path(payload["data_iterator_state"]["path"]).name,
        "data_iterator_state_sha256": artifacts.sha256_bytes(linked["data_iterator_state"][1]),
        "cumulative_telemetry_file": Path(payload["cumulative_telemetry"]["path"]).name,
        "cumulative_telemetry_sha256": artifacts.sha256_bytes(linked["cumulative_telemetry"][1]),
    }
    if unsigned != expected_runtime:
        raise RungEvaluationError("checkpoint runtime identity does not bind task-health runtime files")
    telemetry_keys = {
        "schema_version",
        "accepted_update_count",
        "finite_accepted_update_count",
        "pre_shared_severe_clip_count",
        "pre_shared_update_count",
        "write_feature_cap_bind_numerator",
        "write_feature_cap_bind_denominator",
        "read_feature_cap_bind_numerator",
        "read_feature_cap_bind_denominator",
        "pre_shared_grad_norm_max",
    }
    if set(telemetry) != telemetry_keys or telemetry.get("schema_version") != 1:
        raise RungEvaluationError("cumulative telemetry schema is invalid")
    count_keys = telemetry_keys - {"schema_version", "pre_shared_grad_norm_max"}
    if any(type(telemetry[key]) is not int or telemetry[key] < 0 for key in count_keys):
        raise RungEvaluationError("cumulative telemetry counts must be nonnegative integers")
    if any(
        telemetry[key] != completed_updates
        for key in ("accepted_update_count", "finite_accepted_update_count", "pre_shared_update_count")
    ):
        raise RungEvaluationError("cumulative telemetry does not cover exactly all completed updates")

    raw_path, _ = artifacts.resolve_hashed_relative_file(
        owner_path=path, descriptor=payload["raw_npz"], descriptor_name="raw_npz"
    )
    arrays = _load_npz(raw_path, keys=TASK_HEALTH_ARRAY_KEYS, name="task-health raw")
    ids = _strings(arrays["stable_id"], name="task-health stable_id")
    n = len(ids)
    if n == 0 or set(ids) - allowed_stable_ids:
        raise RungEvaluationError("task-health suite is empty or contains development/final/unknown IDs")
    frame = np.asarray(arrays["frame_index"])
    rng_seed = np.asarray(arrays["rng_seed"])
    if (
        frame.shape != (n,)
        or rng_seed.shape != (n,)
        or not np.issubdtype(frame.dtype, np.integer)
        or not np.issubdtype(rng_seed.dtype, np.integer)
        or np.any(frame < 0)
    ):
        raise RungEvaluationError("task-health frame/RNG arrays are invalid")
    noise_hash = _strings(arrays["action_noise_sha256"], name="action_noise_sha256")
    for index, digest in enumerate(noise_hash):
        artifacts.require_sha256(f"action_noise_sha256[{index}]", digest)
    suite_records = [{"frame_index": int(frame[i]), "stable_id": ids[i]} for i in range(n)]
    rng_records = [
        {
            "action_noise_sha256": noise_hash[i],
            "flow_time": float(_finite_vector(arrays["flow_time"], name="flow_time", length=n)[i]),
            "frame_index": int(frame[i]),
            "rng_seed": int(rng_seed[i]),
            "stable_id": ids[i],
        }
        for i in range(n)
    ]
    suite_sha = artifacts.sha256_bytes(artifacts.canonical_json_bytes(suite_records))
    rng_sha = artifacts.sha256_bytes(artifacts.canonical_json_bytes(rng_records))
    if payload["no_augmentation_suite_sha256"] != suite_sha or payload["rng_inputs_sha256"] != rng_sha:
        raise RungEvaluationError("task-health suite/RNG hashes are not derived from raw paired rows")

    metric_names = (
        "fresh_source_flow_loss",
        "fresh_source_subtask_ce",
        "v35_step0_flow_loss",
        "v35_step0_subtask_ce",
        "rung_flow_loss",
        "rung_subtask_ce",
    )
    metrics = {name: _finite_vector(arrays[name], name=name, length=n) for name in metric_names}
    if any(np.any(value < 0.0) for value in metrics.values()):
        raise RungEvaluationError("task-health losses must be nonnegative")
    if completed_updates == 0 and (
        not np.array_equal(metrics["rung_flow_loss"], metrics["v35_step0_flow_loss"])
        or not np.array_equal(metrics["rung_subtask_ce"], metrics["v35_step0_subtask_ce"])
    ):
        raise RungEvaluationError("rung-0 task-health rows must be byte-identical to v3.5 step-0 rows")
    finite_arrays = {}
    for name in ("gradient_finite", "parameter_finite", "memory_state_finite"):
        value = np.asarray(arrays[name])
        if value.dtype != np.dtype(np.bool_) or value.shape != (n,):
            raise RungEvaluationError(f"{name} must be one paired boolean per task-health row")
        finite_arrays[name] = value
    cap_numerator = telemetry["write_feature_cap_bind_numerator"] + telemetry["read_feature_cap_bind_numerator"]
    cap_denominator = telemetry["write_feature_cap_bind_denominator"] + telemetry["read_feature_cap_bind_denominator"]
    if cap_numerator > cap_denominator or telemetry["pre_shared_severe_clip_count"] > completed_updates:
        raise RungEvaluationError("cumulative clip/cap numerators exceed denominators")

    return {
        "feature_cap": {
            "bound_terms": cap_numerator,
            "cap_value": 1.0,
            "definition": "unweighted_per_term_feature_cotangent_before_episode_cell_and_loss_weight",
            "eligible_terms": cap_denominator,
        },
        "finiteness": {
            "gradients": bool(np.all(finite_arrays["gradient_finite"]))
            and telemetry["finite_accepted_update_count"] == completed_updates,
            "losses": all(np.all(np.isfinite(value)) for value in metrics.values()),
            "memory_state": bool(np.all(finite_arrays["memory_state_finite"])),
            "parameters": bool(np.all(finite_arrays["parameter_finite"])),
        },
        "fresh_source_reference": {
            "flow_loss": float(np.mean(metrics["fresh_source_flow_loss"])),
            "subtask_ce": float(np.mean(metrics["fresh_source_subtask_ce"])),
        },
        "no_augmentation_suite_sha256": suite_sha,
        "preprocessing_norm_sha256": artifacts.require_sha256(
            "preprocessing_norm_sha256", payload["preprocessing_norm_sha256"]
        ),
        "rng_inputs_sha256": rng_sha,
        "rung": {
            "flow_loss": float(np.mean(metrics["rung_flow_loss"])),
            "subtask_ce": float(np.mean(metrics["rung_subtask_ce"])),
        },
        "severe_clip": {
            "definition": "pre_shared_global_grad_norm_gt_10x_optimizer_clip_threshold",
            "optimizer_clip_threshold": 1.0,
            "severe_steps": telemetry["pre_shared_severe_clip_count"],
            "total_optimizer_steps": completed_updates,
        },
        "v35_step0": {
            "flow_loss": float(np.mean(metrics["v35_step0_flow_loss"])),
            "subtask_ce": float(np.mean(metrics["v35_step0_subtask_ce"])),
        },
    }


def load_and_reduce_raw_artifact(
    raw_envelope_path: Path,
    *,
    manifest: artifacts.FrozenManifest,
    completed_updates: int,
    checkpoint_parameter_tree_sha256: str,
    initialization_parameter_tree_sha256: str,
    initialization_identity_sha256: str,
    initialization_manifest_file_sha256: str,
    calibration_artifact_id: str,
    calibration_passes: bool,
    prototype_artifact_id: str,
) -> dict[str, Any]:
    """Authenticate and re-reduce the raw artifact linked by a canonical rung."""

    envelope = artifacts.load_canonical_envelope(raw_envelope_path, schema_version=RAW_SCHEMA_VERSION)
    payload = artifacts.require_exact_keys(
        "raw rung payload",
        envelope["payload"],
        {
            "calibration_artifact_id",
            "checkpoint_parameter_tree_sha256",
            "completed_updates",
            "episode_manifest_sha256",
            "evaluation_protocol_sha256",
            "initialization_identity_sha256",
            "initialization_parameter_tree_sha256",
            "mechanism_artifact",
            "prototype_artifact_id",
            "raw_npz",
            "selection_artifact",
            "split_assignment_sha256",
            "task_health_artifact",
        },
    )
    expected = {
        "completed_updates": completed_updates,
        "checkpoint_parameter_tree_sha256": checkpoint_parameter_tree_sha256,
        "initialization_parameter_tree_sha256": initialization_parameter_tree_sha256,
        "initialization_identity_sha256": initialization_identity_sha256,
        "calibration_artifact_id": calibration_artifact_id,
        "prototype_artifact_id": prototype_artifact_id,
        "episode_manifest_sha256": manifest.sha256,
        "split_assignment_sha256": manifest.split_assignment_sha256,
        "evaluation_protocol_sha256": EVALUATION_PROTOCOL_SHA256,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise RungEvaluationError("raw rung artifact identity/protocol mismatch")
    raw_npz, _ = artifacts.resolve_hashed_relative_file(
        owner_path=raw_envelope_path, descriptor=payload["raw_npz"], descriptor_name="raw_npz"
    )
    arrays = _load_npz(raw_npz, keys=RAW_RESULT_KEYS, name="rung condition raw")
    reduced = reduce_condition_arrays(arrays, manifest=manifest)
    mechanism_path, mechanism = _linked_envelope(
        raw_envelope_path,
        payload["mechanism_artifact"],
        name="mechanism_artifact",
        schema=MECHANISM_SCHEMA_VERSION,
    )
    del mechanism
    task_health_path, task_health = _linked_envelope(
        raw_envelope_path,
        payload["task_health_artifact"],
        name="task_health_artifact",
        schema=TASK_HEALTH_SCHEMA_VERSION,
    )
    del task_health
    reduced["gate_c"] = reduce_mechanism_artifact(
        mechanism_path,
        checkpoint_sha256=checkpoint_parameter_tree_sha256,
        initialization_sha256=initialization_parameter_tree_sha256,
        calibration_artifact_id=calibration_artifact_id,
        calibration_passes=calibration_passes,
    )
    reduced["task_health"] = reduce_task_health_artifact(
        task_health_path,
        completed_updates=completed_updates,
        checkpoint_sha256=checkpoint_parameter_tree_sha256,
        initialization_identity_sha256=initialization_identity_sha256,
        initialization_manifest_file_sha256=initialization_manifest_file_sha256,
        allowed_stable_ids={episode.stable_id for episode in manifest.split("train")},
    )
    _validate_selection_artifact(
        owner_path=raw_envelope_path,
        descriptor=payload["selection_artifact"],
        manifest=manifest,
        arrays=arrays,
        task_health=reduced["task_health"],
    )
    return reduced


def _validate_selection_artifact(
    *,
    owner_path: Path,
    descriptor: Any,
    manifest: artifacts.FrozenManifest,
    arrays: Mapping[str, npt.NDArray[Any]],
    task_health: Mapping[str, Any],
) -> None:
    """Bind every reduced frame and the paired suite to the frozen scalar-parquet selection."""

    _, selection = _linked_envelope(
        owner_path,
        descriptor,
        name="selection_artifact",
        schema=SELECTION_SCHEMA_VERSION,
    )
    payload = artifacts.require_exact_keys(
        "selection payload",
        selection["payload"],
        {
            "dataset_root_relative",
            "episode_manifest_sha256",
            "episodes",
            "final_test_payload_access_count",
            "protocol_sha256",
            "split_assignment_sha256",
            "task_health",
        },
    )
    if (
        payload["episode_manifest_sha256"] != manifest.sha256
        or payload["split_assignment_sha256"] != manifest.split_assignment_sha256
        or payload["protocol_sha256"] != SELECTION_PROTOCOL_SHA256
        or payload["final_test_payload_access_count"] != 0
    ):
        raise RungEvaluationError("linked frame selection has the wrong manifest/protocol identity")
    records = payload["episodes"]
    if not isinstance(records, list):
        raise RungEvaluationError("linked frame selection episodes must be one list")
    expected_episodes = tuple(manifest.split("train")) + tuple(manifest.split("development"))
    if len(records) != len(expected_episodes):
        raise RungEvaluationError("linked frame selection is not exact train74+development8")
    by_id: dict[str, Mapping[str, Any]] = {}
    episode_keys = {
        "collection",
        "d_frames",
        "e_frames",
        "episode_index",
        "expected_frames",
        "object_name",
        "parquet_sha256",
        "prompt",
        "split",
        "stable_id",
        "target_side",
        "use_frames",
    }
    for index, (raw_record, episode) in enumerate(zip(records, expected_episodes, strict=True)):
        record = artifacts.require_exact_keys(f"selection episode {index}", raw_record, episode_keys)
        expected_identity = {
            "collection": episode.collection,
            "episode_index": episode.episode_index,
            "object_name": episode.object_name,
            "split": episode.split,
            "stable_id": episode.stable_id,
            "target_side": episode.target_side,
        }
        if any(record.get(name) != value for name, value in expected_identity.items()):
            raise RungEvaluationError(f"linked frame selection identity mismatch for {episode.stable_id}")
        artifacts.require_sha256(f"selection {episode.stable_id} parquet_sha256", record.get("parquet_sha256"))
        for field in ("e_frames", "d_frames", "use_frames"):
            frames = record.get(field)
            if (
                not isinstance(frames, list)
                or not frames
                or any(type(frame) is not int or frame < 0 or frame % 15 for frame in frames)
                or frames != sorted(set(frames))
            ):
                raise RungEvaluationError(f"selection {episode.stable_id} has invalid {field}")
        by_id[episode.stable_id] = record

    dev = tuple(manifest.split("development"))
    for prefix, ordinal_name, frame_name, selected_name in (
        ("E", "e_episode_ordinal", "e_frame_index", "e_frames"),
        ("D", "d_episode_ordinal", "d_frame_index", "d_frames"),
        ("use", "use_episode_ordinal", "use_frame_index", "use_frames"),
    ):
        actual = [
            (int(ordinal), int(frame)) for ordinal, frame in zip(arrays[ordinal_name], arrays[frame_name], strict=True)
        ]
        expected = [
            (ordinal, int(frame))
            for ordinal, episode in enumerate(dev)
            for frame in by_id[episode.stable_id][selected_name]
        ]
        if actual != expected:
            raise RungEvaluationError(f"raw {prefix} frame population differs from linked frozen selection")

    suite = payload["task_health"]
    if not isinstance(suite, list) or len(suite) != 16:
        raise RungEvaluationError("linked task-health selection must contain exactly 16 rows")
    suite_records: list[dict[str, Any]] = []
    suite_keys = {"episode_index", "flow_time", "frame_index", "rng_seed", "stable_id", "task"}
    train_by_id = {episode.stable_id: episode for episode in manifest.split("train")}
    for index, raw_record in enumerate(suite):
        record = artifacts.require_exact_keys(f"selection task-health row {index}", raw_record, suite_keys)
        episode = train_by_id.get(record.get("stable_id"))
        if (
            episode is None
            or record.get("episode_index") != episode.episode_index
            or type(record.get("frame_index")) is not int
            or record["frame_index"] < 0
            or record["frame_index"] % 15
            or type(record.get("rng_seed")) is not int
            or not isinstance(record.get("flow_time"), int | float)
            or not 0.1 <= float(record["flow_time"]) <= 0.900001
            or not isinstance(record.get("task"), str)
        ):
            raise RungEvaluationError(f"linked task-health selection row {index} is invalid")
        suite_records.append({"frame_index": record["frame_index"], "stable_id": record["stable_id"]})
    suite_sha256 = artifacts.sha256_bytes(artifacts.canonical_json_bytes(suite_records))
    if task_health.get("no_augmentation_suite_sha256") != suite_sha256:
        raise RungEvaluationError("task-health evidence differs from linked frozen selection")


def verify_rung_raw_link(
    *,
    rung_path: Path,
    descriptor: Any,
    manifest: artifacts.FrozenManifest,
    completed_updates: int,
    checkpoint_parameter_tree_sha256: str,
    initialization_parameter_tree_sha256: str,
    initialization_identity_sha256: str,
    initialization_manifest_file_sha256: str,
    calibration_artifact_id: str,
    calibration_passes: bool,
    prototype_artifact_id: str,
) -> dict[str, Any]:
    """Resolve a rung-relative raw descriptor and return its deterministic reduction."""

    descriptor = artifacts.require_exact_keys("evaluation.raw_artifact", descriptor, {"artifact_id", "path", "sha256"})
    artifact_id = artifacts.require_artifact_id("evaluation.raw_artifact.artifact_id", descriptor["artifact_id"])
    path, _ = artifacts.resolve_hashed_relative_file(
        owner_path=rung_path,
        descriptor={"path": descriptor["path"], "sha256": descriptor["sha256"]},
        descriptor_name="evaluation.raw_artifact",
    )
    envelope = artifacts.load_canonical_envelope(path, schema_version=RAW_SCHEMA_VERSION)
    if envelope["artifact_id"] != artifact_id:
        raise RungEvaluationError("evaluation raw artifact ID mismatch")
    return load_and_reduce_raw_artifact(
        path,
        manifest=manifest,
        completed_updates=completed_updates,
        checkpoint_parameter_tree_sha256=checkpoint_parameter_tree_sha256,
        initialization_parameter_tree_sha256=initialization_parameter_tree_sha256,
        initialization_identity_sha256=initialization_identity_sha256,
        initialization_manifest_file_sha256=initialization_manifest_file_sha256,
        calibration_artifact_id=calibration_artifact_id,
        calibration_passes=calibration_passes,
        prototype_artifact_id=prototype_artifact_id,
    )


def emit_raw_envelope(
    *,
    output_path: Path,
    raw_npz: Path,
    mechanism_path: Path,
    selection_path: Path,
    task_health_path: Path,
    manifest: artifacts.FrozenManifest,
    completed_updates: int,
    checkpoint_parameter_tree_sha256: str,
    initialization_parameter_tree_sha256: str,
    initialization_identity_sha256: str,
    calibration_artifact_id: str,
    prototype_artifact_id: str,
) -> Path:
    """Create one immutable raw-evaluation envelope from already sealed measurement files."""

    for path in (raw_npz, mechanism_path, selection_path, task_health_path):
        if path.parent.resolve() != output_path.parent.resolve():
            raise RungEvaluationError("raw/mechanism/task-health files must share the output artifact directory")
    if output_path.exists():
        raise RungEvaluationError(f"refusing to overwrite raw rung envelope {output_path}")
    mechanism = artifacts.load_canonical_envelope(mechanism_path, schema_version=MECHANISM_SCHEMA_VERSION)
    selection = artifacts.load_canonical_envelope(selection_path, schema_version=SELECTION_SCHEMA_VERSION)
    task_health = artifacts.load_canonical_envelope(task_health_path, schema_version=TASK_HEALTH_SCHEMA_VERSION)
    payload = {
        "calibration_artifact_id": calibration_artifact_id,
        "checkpoint_parameter_tree_sha256": checkpoint_parameter_tree_sha256,
        "completed_updates": completed_updates,
        "episode_manifest_sha256": manifest.sha256,
        "evaluation_protocol_sha256": EVALUATION_PROTOCOL_SHA256,
        "initialization_identity_sha256": initialization_identity_sha256,
        "initialization_parameter_tree_sha256": initialization_parameter_tree_sha256,
        "mechanism_artifact": {
            "artifact_id": mechanism["artifact_id"],
            "path": mechanism_path.name,
            "sha256": _sha256_file(mechanism_path),
        },
        "prototype_artifact_id": prototype_artifact_id,
        "raw_npz": {"path": raw_npz.name, "sha256": _sha256_file(raw_npz)},
        "selection_artifact": {
            "artifact_id": selection["artifact_id"],
            "path": selection_path.name,
            "sha256": _sha256_file(selection_path),
        },
        "split_assignment_sha256": manifest.split_assignment_sha256,
        "task_health_artifact": {
            "artifact_id": task_health["artifact_id"],
            "path": task_health_path.name,
            "sha256": _sha256_file(task_health_path),
        },
    }
    envelope = artifacts.artifact_envelope(RAW_SCHEMA_VERSION, payload)
    artifacts.write_canonical_envelope(output_path, envelope, schema_version=RAW_SCHEMA_VERSION)
    return output_path


def _write_bytes_once(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RungEvaluationError(f"refusing to overwrite linked rung artifact {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RungEvaluationError(f"refusing to overwrite linked rung artifact {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _directory_tree_sha256(path: Path) -> str:
    """Hash a directory by canonical relative path, size, and per-file SHA256."""

    if not path.is_dir():
        raise RungEvaluationError(f"checkpoint directory component is missing: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files or any(item.is_symlink() for item in files):
        raise RungEvaluationError(f"checkpoint directory must contain regular non-symlink files: {path}")
    inventory = [
        {
            "path": item.relative_to(path).as_posix(),
            "sha256": _sha256_file(item),
            "size": item.stat().st_size,
        }
        for item in files
    ]
    return artifacts.sha256_bytes(artifacts.canonical_json_bytes(inventory))


def _identity_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        identity = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RungEvaluationError(f"cannot read initialization identity {path}: {exc}") from exc
    if not isinstance(identity, dict):
        raise RungEvaluationError("initialization identity must be an object")
    recorded = artifacts.require_sha256("initialization identity_sha256", identity.get("identity_sha256"))
    unsigned = dict(identity)
    del unsigned["identity_sha256"]
    if recorded != artifacts.sha256_bytes(artifacts.canonical_json_bytes(unsigned)):
        raise RungEvaluationError("initialization identity self-hash is invalid")
    return identity, raw


def _calibration_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        calibration = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RungEvaluationError(f"cannot read calibration {path}: {exc}") from exc
    payload = calibration.get("payload") if isinstance(calibration, dict) else None
    if not isinstance(payload, dict):
        raise RungEvaluationError("calibration must contain a payload")
    digest = artifacts.sha256_bytes(artifacts.canonical_json_bytes(payload))
    if (
        calibration.get("artifact_sha256") != digest
        or calibration.get("calibration_id") != f"sha256:{digest}"
        or payload.get("schema_version") != "openpi.v35.injection-calibration.v1"
        or payload.get("status") != "pass"
        or payload.get("gates", {}).get("passes") is not True
    ):
        raise RungEvaluationError("calibration envelope/hash/status is invalid")
    return calibration, raw


def _checkpoint_parameter_hash(params_path: Path) -> str:
    from openpi.models import model as model_lib
    from openpi.training import weight_loaders

    params = model_lib.restore_params(params_path, restore_type=np.ndarray)
    return weight_loaders.parameter_tree_sha256(params)


def _extension_authorization_id(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        raw = path.read_bytes()
        envelope = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RungEvaluationError(f"cannot read extension authorization {path}: {exc}") from exc
    if raw != artifacts.canonical_json_bytes(envelope) + b"\n" or not isinstance(envelope, dict):
        raise RungEvaluationError("extension authorization must be canonical JSON")
    artifact_id = artifacts.require_artifact_id("extension authorization artifact_id", envelope.get("artifact_id"))
    payload = envelope.get("payload")
    if (
        envelope.get("schema_version") != "openpi.v35.training-authorization.v1"
        or not isinstance(payload, dict)
        or artifact_id != f"sha256:{artifacts.sha256_bytes(artifacts.canonical_json_bytes(payload))}"
        or payload.get("authorization_kind") != "continuation"
        or payload.get("status") != "pass"
    ):
        raise RungEvaluationError("extension authorization envelope/status is invalid")
    return artifact_id


def emit_rung_result(
    *,
    checkpoint_step_dir: Path,
    raw_envelope_path: Path,
    prototype_path: Path,
    manifest: artifacts.FrozenManifest,
    output_path: Path,
    previous_rung_path: Path | None = None,
    extension_authorization_path: Path | None = None,
    parameter_tree_hasher: Any = _checkpoint_parameter_hash,
) -> Path:
    """Seal one real rung and immediately load it through the authoritative reducer."""

    checkpoint_step_dir = Path(checkpoint_step_dir)
    if not checkpoint_step_dir.is_dir() or not checkpoint_step_dir.name.isdigit():
        raise RungEvaluationError("checkpoint step directory must exist and have a numeric completed-update name")
    completed_updates = int(checkpoint_step_dir.name)
    if completed_updates not in pilot.FIXED_EXTENSION_RUNGS:
        raise RungEvaluationError(f"checkpoint {completed_updates} is not a preregistered pilot rung")
    metadata_path = checkpoint_step_dir / "_CHECKPOINT_METADATA"
    if (
        not metadata_path.is_file()
        or not (checkpoint_step_dir / "params").is_dir()
        or not (checkpoint_step_dir / "train_state").is_dir()
    ):
        raise RungEvaluationError("checkpoint is not a finalized params/train_state Orbax step")
    output_path = Path(output_path)
    if output_path.exists():
        raise RungEvaluationError(f"refusing to overwrite rung result {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for linked in (raw_envelope_path, prototype_path):
        if Path(linked).parent.resolve() != output_path.parent.resolve():
            raise RungEvaluationError("raw evaluation, prototype, and rung JSON must share one artifact directory")

    assets_dir = checkpoint_step_dir / "assets"
    required_assets = {
        "calibration": assets_dir / "v35_calibration_artifact.json",
        "initialization": assets_dir / "v35_initialization_manifest.json",
        "embedded_manifest": assets_dir / "v35_episode_manifest.json",
        "runtime": assets_dir / "v35_runtime_identity.json",
        "telemetry": assets_dir / "v35_cumulative_telemetry.json",
        "data_state": assets_dir / "v35_data_iterator_state.json",
    }
    missing = [name for name, path in required_assets.items() if not path.is_file()]
    if missing:
        raise RungEvaluationError(f"checkpoint is missing v3.5 provenance/runtime assets: {missing}")
    if _sha256_file(required_assets["embedded_manifest"]) != manifest.sha256:
        raise RungEvaluationError("checkpoint-embedded frozen manifest differs from evaluator manifest")
    identity, identity_bytes = _identity_json(required_assets["initialization"])
    calibration, calibration_bytes = _calibration_json(required_assets["calibration"])
    initialization_sha = artifacts.require_sha256(
        "initialization parameter tree", identity.get("actual_step0_parameter_tree_sha256")
    )
    checkpoint_sha = parameter_tree_hasher(checkpoint_step_dir / "params")
    artifacts.require_sha256("checkpoint parameter tree", checkpoint_sha)
    if completed_updates == 0 and checkpoint_sha != initialization_sha:
        raise RungEvaluationError("step-0 checkpoint params differ from initialization identity")
    if identity.get("calibration_id") != calibration["calibration_id"]:
        raise RungEvaluationError("checkpoint initialization/calibration IDs differ")
    artifact_hashes = identity.get("artifact_hashes")
    if (
        not isinstance(artifact_hashes, dict)
        or artifact_hashes.get("episode_manifest_sha256") != manifest.sha256
        or artifact_hashes.get("calibration_artifact_sha256") != _sha256_file(required_assets["calibration"])
    ):
        raise RungEvaluationError("checkpoint initialization does not bind manifest/calibration bytes")

    prototype = artifacts.load_canonical_envelope(prototype_path, schema_version=pilot.PROTOTYPE_SCHEMA_VERSION)
    raw_envelope = artifacts.load_canonical_envelope(raw_envelope_path, schema_version=RAW_SCHEMA_VERSION)
    # Snapshot the two checkpoint-owned inputs next to the output; all linked paths remain
    # portable and no mutable checkpoint path appears in a gate artifact.
    calibration_snapshot = output_path.with_name(f"{output_path.stem}_calibration.json")
    initialization_snapshot = output_path.with_name(f"{output_path.stem}_initialization_manifest.json")
    _write_bytes_once(calibration_snapshot, calibration_bytes)
    _write_bytes_once(initialization_snapshot, identity_bytes)

    reduced = load_and_reduce_raw_artifact(
        raw_envelope_path,
        manifest=manifest,
        completed_updates=completed_updates,
        checkpoint_parameter_tree_sha256=checkpoint_sha,
        initialization_parameter_tree_sha256=initialization_sha,
        initialization_identity_sha256=identity["identity_sha256"],
        initialization_manifest_file_sha256=_sha256_file(required_assets["initialization"]),
        calibration_artifact_id=calibration["calibration_id"],
        calibration_passes=True,
        prototype_artifact_id=prototype["artifact_id"],
    )

    previous = None
    expected_previous = {250: 0, 500: 250, 1_000: 500, 2_500: 1_000}
    if completed_updates == 0:
        if previous_rung_path is not None:
            raise RungEvaluationError("step-0 rung must not name a previous rung")
    else:
        if previous_rung_path is None:
            raise RungEvaluationError("nonzero rung requires the exact previous rung artifact")
        previous_rung = pilot.load_rung_result(previous_rung_path, manifest=manifest)
        if previous_rung.completed_updates != expected_previous[completed_updates]:
            raise RungEvaluationError("previous rung has the wrong completed-update boundary")
        previous = {
            "completed_updates": previous_rung.completed_updates,
            "data_iterator_state_sha256": previous_rung.run_provenance["data_iterator_state_sha256"],
            "optimizer_state_sha256": previous_rung.run_provenance["optimizer_state_sha256"],
            "parameter_tree_sha256": previous_rung.checkpoint_parameter_tree_sha256,
        }
    extension_id = _extension_authorization_id(extension_authorization_path)
    if (completed_updates <= 1_000 and extension_id is not None) or (
        completed_updates == 2_500 and extension_id is None
    ):
        raise RungEvaluationError("extension authorization presence does not match the rung")

    payload = {
        "calibration_artifact": {
            "calibration_id": calibration["calibration_id"],
            "path": calibration_snapshot.name,
            "sha256": _sha256_file(calibration_snapshot),
        },
        "checkpoint": {"completed_updates": completed_updates, "parameter_tree_sha256": checkpoint_sha},
        "episode_manifest_sha256": manifest.sha256,
        "episodes": reduced["episodes"],
        "evaluation": {
            "episode_aggregation": pilot.EPISODE_AGGREGATION,
            "final_test_accessed": False,
            "raw_artifact": {
                "artifact_id": raw_envelope["artifact_id"],
                "path": raw_envelope_path.name,
                "sha256": _sha256_file(raw_envelope_path),
            },
            "split": "development",
        },
        "evaluation_protocol_sha256": EVALUATION_PROTOCOL_SHA256,
        "gate_c": reduced["gate_c"],
        "initialization_manifest": {
            "path": initialization_snapshot.name,
            "sha256": _sha256_file(initialization_snapshot),
        },
        "initialization_parameter_tree_sha256": initialization_sha,
        "prototype_artifact": {
            "artifact_id": prototype["artifact_id"],
            "path": prototype_path.name,
            "sha256": _sha256_file(prototype_path),
        },
        "run_provenance": {
            "cumulative_telemetry_sha256": _sha256_file(required_assets["telemetry"]),
            "data_iterator_state_sha256": _sha256_file(required_assets["data_state"]),
            "extension_authorization_artifact_id": extension_id,
            "optimizer_state_sha256": _directory_tree_sha256(checkpoint_step_dir / "train_state"),
            "previous_rung": previous,
            "run_id_sha256": artifacts.require_sha256("run_id_sha256", identity.get("run_id_sha256")),
            "runtime_identity_sha256": _sha256_file(required_assets["runtime"]),
            "training_config_sha256": artifacts.require_sha256(
                "semantic_training_config_sha256", identity.get("semantic_training_config_sha256")
            ),
        },
        "split_assignment_sha256": manifest.split_assignment_sha256,
        "task_health": reduced["task_health"],
        "train_prototype_loo": reduced["train_prototype_loo"],
        "train_writer_oof": reduced["train_writer_oof"],
    }
    envelope = artifacts.artifact_envelope(pilot.RUNG_SCHEMA_VERSION, payload)
    artifacts.write_canonical_envelope(output_path, envelope, schema_version=pilot.RUNG_SCHEMA_VERSION)
    pilot.load_rung_result(output_path, manifest=manifest)
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-protocol-sha256", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    project_paths.configure_v35_runtime_environment()
    if args.print_protocol_sha256:
        print(EVALUATION_PROTOCOL_SHA256)
        return
    raise RungEvaluationError(
        "use scripts/v35_rung_collect.py select to freeze scalar-parquet frames, then "
        "scripts/v35_rung_collect.py collect to run the authenticated GPU producer and this reducer"
    )


if __name__ == "__main__":
    main()
