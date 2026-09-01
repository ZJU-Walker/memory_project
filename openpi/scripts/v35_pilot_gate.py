"""Reduce sealed v3.5 rung results into the preregistered Gate-D pilot decision.

Inputs are one canonical result artifact for every fixed rung (0/250/500/1000, plus 2500
only after an authenticated inconclusive 1k decision).  Each rung contains exactly the eight
development episodes and links a checkpoint-specific, train-only prototype artifact.  The
reducer rejects unknown/non-development episode IDs, so final-test observations cannot enter
pilot or extension decisions.

Writer-claim gates are reported separately: failure of counterfactual prompt binding removes
the prompt-bound writer claim but does not override an otherwise valid natural-memory chain.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

from openpi.shared import project_paths

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import v35_gate_artifacts as artifacts  # noqa: E402

PROTOTYPE_SCHEMA_VERSION = "openpi.v35.side-prototypes.v1"
RUNG_SCHEMA_VERSION = "openpi.v35.pilot-rung-results.v2"
DECISION_SCHEMA_VERSION = "openpi.v35.pilot-gate-decision.v1"
CRITERIA_VERSION = "openpi.v35.gate-d.rev5.v1"
PROTOTYPE_CONSTRUCTION = "episode_mean_then_side_episode_macro_l2_normalized.v1"
EPISODE_AGGREGATION = "aggregate_eligible_frames_within_episode_then_one_hard_outcome.v1"
TRAIN_WRITER_PROTOCOL = "fig1_episode_grouped_oof_train54.v1"
TRAIN_PROTOTYPE_PROTOCOL = "leave_one_episode_out_train54_episode_macro.v1"
FIXED_1K_RUNGS = (0, 250, 500, 1_000)
FIXED_EXTENSION_RUNGS = (0, 250, 500, 1_000, 2_500)

DEV_EPISODES = 8
TRAIN_EPISODES = 54
PASS_COUNT = 7
RESET_MAX = 4
ZERO_RESET_DIFF_MAX = 1
WRITER_TRAIN_ACCURACY_MIN = 0.90
PRODUCTION_RESIDUAL_P95_MAX = 1e-2
SYNTHETIC_RESIDUAL_MAX = 1e-5
RETENTION_COSINE_MIN = 0.9999
RETENTION_RATIO_ERROR_MAX = 1e-3
RETENTION_NORM_RATIO_MIN = 0.55
RETAINED_AMPLITUDE_MIN = 0.40
REAL_NOISE_RATIO_P95_MAX = 0.10
FLOW_RATIO_MAX = 1.10
SUBTASK_CE_DELTA_MAX = 0.05
SEVERE_CLIP_RATE_MAX = 0.01
FEATURE_CAP_BIND_RATE_MAX = 0.05

EPISODE_BOOLEAN_FIELDS = (
    "writer_natural_correct",
    "writer_counterfactual_correct",
    "direct_carry_correct",
    "read_natural_correct",
    "read_reset_correct",
    "read_opposite_donor_followed",
    "action_natural_correct",
    "action_reset_correct",
    "action_opposite_donor_followed",
    "zero_injection_target_correct",
    "zero_vs_reset_prediction_differs",
    "prototype_correct_side_succeeds",
    "prototype_opposite_donor_followed",
    "prototype_paired_action_flipped",
)
EPISODE_FLOAT_FIELDS = (
    "attention_memory_mass_natural",
    "attention_memory_mass_reset",
    "attention_memory_mass_zero",
    "attention_uniform_baseline",
    "zero_minus_reset_action_score",
)
EPISODE_SIDE_FIELDS = (
    "action_natural_predicted_side",
    "action_opposite_donor_predicted_side",
    "action_reset_predicted_side",
    "direct_carry_predicted_side",
    "prototype_correct_side_predicted_side",
    "prototype_opposite_donor_predicted_side",
    "read_natural_predicted_side",
    "read_opposite_donor_predicted_side",
    "read_reset_predicted_side",
    "writer_counterfactual_predicted_side",
    "writer_natural_predicted_side",
    "zero_injection_predicted_side",
)
EPISODE_COUNT_FIELDS = ("eligible_d_frame_count", "eligible_e_frame_count", "use_pressure_frame_count")
EPISODE_KEYS = {
    "stable_id",
    "split",
    *EPISODE_BOOLEAN_FIELDS,
    *EPISODE_FLOAT_FIELDS,
    *EPISODE_SIDE_FIELDS,
    *EPISODE_COUNT_FIELDS,
}
CORE_CHECK_NAMES = (
    "half_delta_residual_pass",
    "clock_and_read_before_transition_pass",
    "fast_hidden_bias_momentum_invariants_pass",
    "padding_strict_noop_pass",
    "legacy_mode_bit_exact_pass",
    "fp32_memory_path_pass",
    "dense_skip_forward_and_gradient_match_pass",
    "write_gradient_reaches_value_and_backbone_pass",
    "reachable_read_gradient_pass",
    "unreachable_read_query_gradient_pass",
    "invalid_and_padding_gradient_mask_pass",
    "injection_calibration_pass",
    "injection_gate_half_and_frozen_pass",
)
GATE_C_METRIC_KEYS = {
    "injected_token_rms",
    "production_relative_commit_residual_p95",
    "raw_read_rms",
    "reachable_fraction",
    "real_noise_injected_to_residual_ratio_p95",
    "retained_injection_amplitude_p90_delay",
    "retention_cosine_p90_delay",
    "retention_norm_ratio_p90_delay",
    "retention_norm_ratio_relative_error_p90_delay",
    "synthetic_fp32_commit_residual_max",
}


class PilotGateError(artifacts.GateArtifactError):
    """Raised when Gate D cannot make an authenticated decision."""


def _resolve_project_cli_path(path: Path, *, name: str) -> Path:
    """Resolve one production CLI path inside the portable project root."""
    raw = Path(path)
    if raw.is_absolute():
        raise PilotGateError(f"{name} must be relative to memory_project, got {str(raw)!r}")
    if ".." in raw.parts:
        raise PilotGateError(f"{name} must not escape memory_project, got {str(raw)!r}")
    try:
        return project_paths.project_path(raw)
    except project_paths.ProjectRootError as exc:
        raise PilotGateError(f"invalid {name}: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class PrototypeIdentity:
    artifact_id: str
    file_sha256: str
    checkpoint_parameter_tree_sha256: str


@dataclasses.dataclass(frozen=True)
class CalibrationIdentity:
    artifact_id: str
    file_sha256: str
    prototype_injected_rms_target: float
    official_base_source_sha256: str
    dataset_protocol_sha256: str
    raw_gate_sha256: str
    parameters: Mapping[str, float]
    p90_n_delay: int
    p90_retention_factor: float


@dataclasses.dataclass(frozen=True)
class RungResult:
    path: Path
    artifact_id: str
    file_sha256: str
    completed_updates: int
    checkpoint_parameter_tree_sha256: str
    initialization_parameter_tree_sha256: str
    initialization_identity: artifacts.InitializationIdentity
    calibration: CalibrationIdentity
    evaluation_protocol_sha256: str
    prototype: PrototypeIdentity
    run_provenance: Mapping[str, Any]
    gate_c: Mapping[str, Any]
    task_health: Mapping[str, Any]
    train_writer_oof: Mapping[str, Any]
    train_prototype_loo: Mapping[str, Any]
    episodes: tuple[Mapping[str, Any], ...]

    @property
    def calibration_artifact_id(self) -> str:
        return self.calibration.artifact_id


def _finite_float(name: str, value: Any, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PilotGateError(f"{name} must be a finite number")
    output = float(value)
    if not math.isfinite(output):
        raise PilotGateError(f"{name} must be finite")
    if minimum is not None and output < minimum:
        raise PilotGateError(f"{name} must be >= {minimum}")
    return output


def _integer_count(name: str, value: Any, *, maximum: int) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise PilotGateError(f"{name} must be an integer in [0,{maximum}]")
    return value


def _load_calibration(
    descriptor: Any,
    *,
    rung_path: Path,
    manifest: artifacts.FrozenManifest,
    expected_parameter_tree_sha256: str,
) -> CalibrationIdentity:
    descriptor = artifacts.require_exact_keys(
        "calibration_artifact",
        descriptor,
        {"calibration_id", "path", "sha256"},
    )
    descriptor_id = artifacts.require_artifact_id("calibration_artifact.calibration_id", descriptor["calibration_id"])
    calibration_path, calibration_sha256 = artifacts.resolve_hashed_relative_file(
        owner_path=rung_path,
        descriptor={"path": descriptor["path"], "sha256": descriptor["sha256"]},
        descriptor_name="calibration_artifact",
    )
    try:
        raw_bytes = calibration_path.read_bytes()
        calibration = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotGateError(f"cannot read calibration artifact {calibration_path}: {exc}") from exc
    calibration = artifacts.require_exact_keys(
        "calibration artifact",
        calibration,
        {"artifact_sha256", "calibration_id", "hash_scope", "payload"},
    )
    payload = calibration["payload"]
    if not isinstance(payload, dict):
        raise PilotGateError("calibration artifact payload must be an object")
    payload_sha256 = artifacts.sha256_bytes(artifacts.canonical_json_bytes(payload))
    if (
        calibration["artifact_sha256"] != payload_sha256
        or calibration["calibration_id"] != f"sha256:{payload_sha256}"
        or calibration["calibration_id"] != descriptor_id
        or calibration["hash_scope"] != "SHA256 of canonical_json($.payload)"
    ):
        raise PilotGateError("calibration artifact payload hash/ID is invalid")
    if raw_bytes != artifacts.canonical_json_bytes(calibration) + b"\n":
        raise PilotGateError("calibration artifact is not in the canonical JSON byte representation")
    if (
        payload.get("schema_version") != "openpi.v35.injection-calibration.v1"
        or payload.get("status") != "pass"
        or not isinstance(payload.get("gates"), dict)
        or payload["gates"].get("passes") is not True
    ):
        raise PilotGateError("calibration artifact has the wrong schema or is not passing")
    expected_train_ids = [episode.stable_id for episode in manifest.split("train")]
    population = payload.get("population")
    if (
        not isinstance(population, dict)
        or population.get("split") != "train"
        or population.get("episode_count") != TRAIN_EPISODES
        or population.get("stable_ids") != expected_train_ids
    ):
        raise PilotGateError("calibration membership must exactly match the frozen 54 training episodes")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("split_sha256") != manifest.sha256:
        raise PilotGateError("calibration split hash does not match the frozen episode manifest")
    if provenance.get("source_sha256") != expected_parameter_tree_sha256:
        raise PilotGateError("calibration was not collected from this exact fresh step-0 parameter tree")
    provenance_hashes = (
        "collector_source_sha256",
        "dataset_sha256",
        "official_base_source_sha256",
        "preflight_sha256",
        "replay_protocol_sha256",
        "source_sha256",
    )
    for name in provenance_hashes:
        artifacts.require_sha256(f"calibration.provenance.{name}", provenance.get(name))
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise PilotGateError("calibration artifact is missing parameters")
    calibrated_parameters = {
        "alpha_step": _finite_float("calibration.parameters.alpha_step", parameters.get("alpha_step"), minimum=0.0),
        "memory_injection_c": _finite_float(
            "calibration.parameters.memory_injection_c",
            parameters.get("memory_injection_c"),
            minimum=np.finfo(np.float32).tiny,
        ),
        "memory_injection_tau": _finite_float(
            "calibration.parameters.memory_injection_tau",
            parameters.get("memory_injection_tau"),
            minimum=np.finfo(np.float32).tiny,
        ),
    }
    if not math.isclose(calibrated_parameters["alpha_step"], 0.01, rel_tol=0.0, abs_tol=2 * np.finfo(np.float32).eps):
        raise PilotGateError("Gate-D calibration must use frozen alpha_step=0.01")
    target = _finite_float(
        "calibration.parameters.prototype_injected_rms_target",
        parameters.get("prototype_injected_rms_target"),
        minimum=np.finfo(np.float32).tiny,
    )
    gate = payload.get("gate")
    channel_count = population.get("channel_count")
    if not isinstance(gate, dict) or type(channel_count) is not int or channel_count <= 0:
        raise PilotGateError("calibration gate/channel metadata is missing")
    raw_gate_sha256 = artifacts.require_sha256("calibration.gate.raw_w_sha256", gate.get("raw_w_sha256"))
    if (
        not math.isclose(float(gate.get("target_effective_tanh_gate", float("nan"))), 0.5, abs_tol=1e-6)
        or gate.get("open_channel_count") != channel_count
        or not math.isclose(float(gate.get("effective_tanh_gate_min", float("nan"))), 0.5, abs_tol=1e-6)
        or not math.isclose(float(gate.get("effective_tanh_gate_max", float("nan"))), 0.5, abs_tol=1e-6)
    ):
        raise PilotGateError("calibration does not preserve the all-open fixed tanh(w)=0.5 gate")
    statistics = payload.get("statistics")
    p90_delay = statistics.get("p90_delay") if isinstance(statistics, dict) else None
    if not isinstance(p90_delay, dict) or type(p90_delay.get("n_delay")) is not int or p90_delay["n_delay"] < 0:
        raise PilotGateError("calibration artifact is missing the frozen p90 delay")
    p90_retention_factor = _finite_float(
        "calibration.statistics.p90_delay.retention_factor",
        p90_delay.get("retention_factor"),
        minimum=0.0,
    )
    expected_retention = (1.0 - calibrated_parameters["alpha_step"]) ** p90_delay["n_delay"]
    if not math.isclose(p90_retention_factor, expected_retention, rel_tol=1e-6, abs_tol=1e-7):
        raise PilotGateError("calibration p90 retention factor is inconsistent with alpha and n_delay")
    return CalibrationIdentity(
        artifact_id=descriptor_id,
        file_sha256=calibration_sha256,
        prototype_injected_rms_target=target,
        official_base_source_sha256=provenance["official_base_source_sha256"],
        dataset_protocol_sha256=provenance["dataset_sha256"],
        raw_gate_sha256=raw_gate_sha256,
        parameters=calibrated_parameters,
        p90_n_delay=p90_delay["n_delay"],
        p90_retention_factor=p90_retention_factor,
    )


def _load_prototype(
    descriptor: Any,
    *,
    rung_path: Path,
    manifest: artifacts.FrozenManifest,
    calibration_artifact_id: str,
    prototype_injected_rms_target: float,
    checkpoint_parameter_tree_sha256: str,
) -> PrototypeIdentity:
    descriptor = artifacts.require_exact_keys(
        "prototype_artifact",
        descriptor,
        {"artifact_id", "path", "sha256"},
    )
    descriptor_artifact_id = artifacts.require_artifact_id("prototype_artifact.artifact_id", descriptor["artifact_id"])
    prototype_path, prototype_sha256 = artifacts.resolve_hashed_relative_file(
        owner_path=rung_path,
        descriptor={"path": descriptor["path"], "sha256": descriptor["sha256"]},
        descriptor_name="prototype_artifact",
    )
    envelope = artifacts.load_canonical_envelope(
        prototype_path,
        schema_version=PROTOTYPE_SCHEMA_VERSION,
    )
    if envelope["artifact_id"] != descriptor_artifact_id:
        raise PilotGateError("prototype descriptor artifact ID does not match the linked prototype envelope")
    payload = artifacts.require_exact_keys(
        "prototype payload",
        envelope["payload"],
        {
            "calibration_artifact_id",
            "checkpoint_parameter_tree_sha256",
            "construction_protocol",
            "directions_npz",
            "episode_manifest_sha256",
            "prototype_injected_rms_target",
            "source_evidence_npz",
            "split_assignment_sha256",
            "training_stable_ids",
        },
    )
    if payload["episode_manifest_sha256"] != manifest.sha256:
        raise PilotGateError("prototype artifact manifest hash mismatch")
    if payload["split_assignment_sha256"] != manifest.split_assignment_sha256:
        raise PilotGateError("prototype artifact split-assignment hash mismatch")
    if payload["calibration_artifact_id"] != calibration_artifact_id:
        raise PilotGateError("prototype artifact calibration ID mismatch")
    if payload["checkpoint_parameter_tree_sha256"] != checkpoint_parameter_tree_sha256:
        raise PilotGateError("prototype artifact checkpoint parameter-tree hash mismatch")
    if payload["construction_protocol"] != PROTOTYPE_CONSTRUCTION:
        raise PilotGateError(f"prototype construction must be {PROTOTYPE_CONSTRUCTION!r}")
    observed_target = _finite_float(
        "prototype_injected_rms_target",
        payload["prototype_injected_rms_target"],
        minimum=np.finfo(np.float32).tiny,
    )
    if observed_target != prototype_injected_rms_target:
        raise PilotGateError(
            "prototype injected-RMS target does not exactly match the calibrated median clean-read injected RMS"
        )
    expected_train_ids = [episode.stable_id for episode in manifest.split("train")]
    if payload["training_stable_ids"] != expected_train_ids:
        raise PilotGateError("prototype artifact training IDs must exactly match manifest order")

    artifacts.resolve_hashed_relative_file(
        owner_path=prototype_path,
        descriptor=payload["source_evidence_npz"],
        descriptor_name="prototype source_evidence_npz",
    )

    npz_path, _ = artifacts.resolve_hashed_relative_file(
        owner_path=prototype_path,
        descriptor=payload["directions_npz"],
        descriptor_name="prototype directions_npz",
    )
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            required = {"episode_stable_id", "episode_mean_vbar", "left_direction", "right_direction"}
            if set(archive.files) != required:
                raise PilotGateError(
                    f"prototype NPZ keys mismatch: missing={sorted(required - set(archive.files))}, "
                    f"extra={sorted(set(archive.files) - required)}"
                )
            ids_array = archive["episode_stable_id"]
            if ids_array.ndim != 1 or ids_array.dtype.kind not in ("S", "U"):
                raise PilotGateError("prototype episode_stable_id must be a string vector")
            ids = [item.decode() if isinstance(item, bytes) else str(item) for item in ids_array]
            if ids != expected_train_ids:
                raise PilotGateError("prototype NPZ episode IDs do not match the frozen train order")
            means = archive["episode_mean_vbar"]
            left = archive["left_direction"]
            right = archive["right_direction"]
            if (
                means.dtype != np.dtype(np.float32)
                or left.dtype != np.dtype(np.float32)
                or right.dtype != np.dtype(np.float32)
                or means.ndim != 2
                or means.shape[0] != TRAIN_EPISODES
                or left.ndim != 1
                or right.ndim != 1
                or means.shape[1:] != left.shape
                or right.shape != left.shape
                or not np.all(np.isfinite(means))
                or not np.all(np.isfinite(left))
                or not np.all(np.isfinite(right))
            ):
                raise PilotGateError("prototype directions must be finite aligned float32 arrays")
            train_by_id = {episode.stable_id: episode for episode in manifest.split("train")}
            sides = np.asarray([train_by_id[stable_id].target_side for stable_id in ids], dtype=np.int64)
            for side, stored in ((0, left), (1, right)):
                expected = np.mean(means[sides == side], axis=0, dtype=np.float32)
                norm = np.linalg.norm(expected.astype(np.float64))
                if not math.isfinite(norm) or norm <= np.finfo(np.float32).tiny:
                    raise PilotGateError(f"side {side} prototype is degenerate")
                expected = np.asarray(expected / np.float32(norm), dtype=np.float32)
                if not np.allclose(stored, expected, rtol=1e-5, atol=1e-6):
                    raise PilotGateError(f"stored side {side} prototype does not match episode-first construction")
    except (OSError, ValueError) as exc:
        if isinstance(exc, PilotGateError):
            raise
        raise PilotGateError(f"cannot load prototype directions {npz_path}: {exc}") from exc

    return PrototypeIdentity(
        artifact_id=envelope["artifact_id"],
        file_sha256=prototype_sha256,
        checkpoint_parameter_tree_sha256=checkpoint_parameter_tree_sha256,
    )


def _validate_gate_c(value: Any) -> dict[str, Any]:
    value = artifacts.require_exact_keys("gate_c", value, {"checks", "metrics"})
    checks = artifacts.require_exact_keys("gate_c.checks", value["checks"], set(CORE_CHECK_NAMES))
    for name, passed in checks.items():
        if type(passed) is not bool:
            raise PilotGateError(f"gate_c.checks.{name} must be boolean")
    metrics = artifacts.require_exact_keys("gate_c.metrics", value["metrics"], GATE_C_METRIC_KEYS)
    validated_metrics = {
        name: _finite_float(f"gate_c.metrics.{name}", raw, minimum=0.0) for name, raw in metrics.items()
    }
    if validated_metrics["reachable_fraction"] > 1.0:
        raise PilotGateError("gate_c.metrics.reachable_fraction must be <= 1")
    return {"checks": dict(checks), "metrics": validated_metrics}


def _validate_task_health(value: Any, *, completed_updates: int) -> dict[str, Any]:
    value = artifacts.require_exact_keys(
        "task_health",
        value,
        {
            "feature_cap",
            "finiteness",
            "fresh_source_reference",
            "no_augmentation_suite_sha256",
            "preprocessing_norm_sha256",
            "rng_inputs_sha256",
            "rung",
            "severe_clip",
            "v35_step0",
        },
    )
    hashes = {
        name: artifacts.require_sha256(name, value[name])
        for name in ("no_augmentation_suite_sha256", "preprocessing_norm_sha256", "rng_inputs_sha256")
    }
    scalar_pairs: dict[str, dict[str, float]] = {}
    for name in ("fresh_source_reference", "v35_step0", "rung"):
        pair = artifacts.require_exact_keys(f"task_health.{name}", value[name], {"flow_loss", "subtask_ce"})
        scalar_pairs[name] = {
            metric: _finite_float(f"task_health.{name}.{metric}", raw, minimum=0.0) for metric, raw in pair.items()
        }
    finiteness = artifacts.require_exact_keys(
        "task_health.finiteness",
        value["finiteness"],
        {"gradients", "losses", "memory_state", "parameters"},
    )
    if any(type(item) is not bool for item in finiteness.values()):
        raise PilotGateError("all task-health finiteness fields must be boolean")
    severe = artifacts.require_exact_keys(
        "task_health.severe_clip",
        value["severe_clip"],
        {"definition", "optimizer_clip_threshold", "severe_steps", "total_optimizer_steps"},
    )
    if severe["definition"] != "pre_shared_global_grad_norm_gt_10x_optimizer_clip_threshold":
        raise PilotGateError("severe-clip telemetry has the wrong definition")
    optimizer_clip_threshold = _finite_float(
        "task_health.severe_clip.optimizer_clip_threshold",
        severe["optimizer_clip_threshold"],
        minimum=np.finfo(np.float32).tiny,
    )
    if optimizer_clip_threshold != 1.0:
        raise PilotGateError("v3.5 optimizer global clip threshold is frozen at 1.0")
    severe_steps = _integer_count(
        "task_health.severe_clip.severe_steps", severe["severe_steps"], maximum=completed_updates
    )
    total_steps = _integer_count(
        "task_health.severe_clip.total_optimizer_steps",
        severe["total_optimizer_steps"],
        maximum=completed_updates,
    )
    if total_steps != completed_updates or severe_steps > total_steps:
        raise PilotGateError("severe-clip telemetry must cover exactly all completed optimizer updates")
    cap = artifacts.require_exact_keys(
        "task_health.feature_cap",
        value["feature_cap"],
        {"bound_terms", "cap_value", "definition", "eligible_terms"},
    )
    if cap["definition"] != "unweighted_per_term_feature_cotangent_before_episode_cell_and_loss_weight":
        raise PilotGateError("feature-cotangent-cap telemetry has the wrong definition")
    cap_value = _finite_float(
        "task_health.feature_cap.cap_value",
        cap["cap_value"],
        minimum=np.finfo(np.float32).tiny,
    )
    if cap_value != 1.0:
        raise PilotGateError("v3.5 branch-local feature-cotangent cap is frozen at 1.0")
    eligible_maximum = 2**63 - 1
    bound_terms = _integer_count("task_health.feature_cap.bound_terms", cap["bound_terms"], maximum=eligible_maximum)
    eligible_terms = _integer_count(
        "task_health.feature_cap.eligible_terms", cap["eligible_terms"], maximum=eligible_maximum
    )
    if bound_terms > eligible_terms or (completed_updates > 0 and eligible_terms == 0):
        raise PilotGateError("feature-cap telemetry must contain eligible terms and bound_terms <= eligible_terms")
    if completed_updates == 0 and scalar_pairs["rung"] != scalar_pairs["v35_step0"]:
        raise PilotGateError("rung-0 task-health values must equal the frozen v3.5 step-0 reference")
    return {
        **hashes,
        **scalar_pairs,
        "finiteness": dict(finiteness),
        "severe_clip": {
            "definition": severe["definition"],
            "optimizer_clip_threshold": optimizer_clip_threshold,
            "severe_steps": severe_steps,
            "total_optimizer_steps": total_steps,
        },
        "feature_cap": {
            "bound_terms": bound_terms,
            "cap_value": cap_value,
            "definition": cap["definition"],
            "eligible_terms": eligible_terms,
        },
    }


def _validate_train_counts(
    value: Any,
    *,
    name: str,
    protocol: str,
    boolean_fields: tuple[str, ...],
    manifest: artifacts.FrozenManifest,
) -> dict[str, Any]:
    required = {"artifact_sha256", "episode_count", "episodes", "protocol"}
    value = artifacts.require_exact_keys(name, value, required)
    if value["episode_count"] != TRAIN_EPISODES or value["protocol"] != protocol:
        raise PilotGateError(f"{name} must use {protocol!r} on exactly 54 training episodes")
    episode_results = value["episodes"]
    if not isinstance(episode_results, list) or len(episode_results) != TRAIN_EPISODES:
        raise PilotGateError(f"{name}.episodes must contain exactly 54 episode-first results")
    expected_ids = [episode.stable_id for episode in manifest.split("train")]
    required_episode_keys = {"stable_id", *boolean_fields}
    for index, raw_record in enumerate(episode_results):
        record = artifacts.require_exact_keys(f"{name}.episodes[{index}]", raw_record, required_episode_keys)
        if record["stable_id"] != expected_ids[index]:
            raise PilotGateError(f"{name} episode IDs must match the frozen training order")
        for field in boolean_fields:
            if type(record[field]) is not bool:
                raise PilotGateError(f"{name}.episodes[{index}].{field} must be boolean")
    actual_artifact_sha256 = artifacts.sha256_bytes(artifacts.canonical_json_bytes(episode_results))
    if value["artifact_sha256"] != actual_artifact_sha256:
        raise PilotGateError(f"{name}.artifact_sha256 must seal its canonical per-episode results")
    output: dict[str, Any] = {
        "artifact_sha256": actual_artifact_sha256,
        "episode_count": TRAIN_EPISODES,
        "protocol": protocol,
    }
    for field in boolean_fields:
        output[f"{field}_count"] = sum(bool(record[field]) for record in episode_results)
    return output


def _validate_run_provenance(value: Any, *, completed_updates: int) -> dict[str, Any]:
    value = artifacts.require_exact_keys(
        "run_provenance",
        value,
        {
            "cumulative_telemetry_sha256",
            "data_iterator_state_sha256",
            "extension_authorization_artifact_id",
            "optimizer_state_sha256",
            "previous_rung",
            "run_id_sha256",
            "runtime_identity_sha256",
            "training_config_sha256",
        },
    )
    output = {
        name: artifacts.require_sha256(f"run_provenance.{name}", value[name])
        for name in (
            "cumulative_telemetry_sha256",
            "data_iterator_state_sha256",
            "optimizer_state_sha256",
            "run_id_sha256",
            "runtime_identity_sha256",
            "training_config_sha256",
        )
    }
    previous = value["previous_rung"]
    if completed_updates == 0:
        if previous is not None:
            raise PilotGateError("rung-0 run provenance must not have a previous rung")
        output["previous_rung"] = None
    else:
        previous = artifacts.require_exact_keys(
            "run_provenance.previous_rung",
            previous,
            {
                "completed_updates",
                "data_iterator_state_sha256",
                "optimizer_state_sha256",
                "parameter_tree_sha256",
            },
        )
        previous_step = previous["completed_updates"]
        if type(previous_step) is not int or previous_step < 0:
            raise PilotGateError("run_provenance.previous_rung.completed_updates must be a nonnegative integer")
        output["previous_rung"] = {
            "completed_updates": previous_step,
            **{
                name: artifacts.require_sha256(f"run_provenance.previous_rung.{name}", previous[name])
                for name in (
                    "data_iterator_state_sha256",
                    "optimizer_state_sha256",
                    "parameter_tree_sha256",
                )
            },
        }
    authorization = value["extension_authorization_artifact_id"]
    if authorization is not None:
        authorization = artifacts.require_artifact_id(
            "run_provenance.extension_authorization_artifact_id", authorization
        )
    output["extension_authorization_artifact_id"] = authorization
    return output


def _validate_episode_results(value: Any, *, manifest: artifacts.FrozenManifest) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or len(value) != DEV_EPISODES:
        raise PilotGateError("each rung must contain exactly eight development episode results")
    expected_ids = [episode.stable_id for episode in manifest.split("development")]
    output: list[Mapping[str, Any]] = []
    for index, raw in enumerate(value):
        record = artifacts.require_exact_keys(f"episodes[{index}]", raw, EPISODE_KEYS)
        if record["stable_id"] != expected_ids[index] or record["split"] != "development":
            raise PilotGateError(
                "episode results must match the frozen development IDs in manifest order; "
                "train/final-test/unknown IDs are forbidden"
            )
        for field in EPISODE_BOOLEAN_FIELDS:
            if type(record[field]) is not bool:
                raise PilotGateError(f"episodes[{index}].{field} must be boolean")
        for field in EPISODE_SIDE_FIELDS:
            if type(record[field]) is not int or record[field] not in (0, 1):
                raise PilotGateError(f"episodes[{index}].{field} must be integer side 0/1")
        for field in EPISODE_COUNT_FIELDS:
            if type(record[field]) is not int or record[field] <= 0:
                raise PilotGateError(f"episodes[{index}].{field} must be a positive integer")
        target = manifest.split("development")[index].target_side
        opposite = 1 - target
        derived = {
            "writer_natural_correct": record["writer_natural_predicted_side"] == target,
            "writer_counterfactual_correct": record["writer_counterfactual_predicted_side"] == opposite,
            "direct_carry_correct": record["direct_carry_predicted_side"] == target,
            "read_natural_correct": record["read_natural_predicted_side"] == target,
            "read_reset_correct": record["read_reset_predicted_side"] == target,
            "read_opposite_donor_followed": record["read_opposite_donor_predicted_side"] == opposite,
            "action_natural_correct": record["action_natural_predicted_side"] == target,
            "action_reset_correct": record["action_reset_predicted_side"] == target,
            "action_opposite_donor_followed": record["action_opposite_donor_predicted_side"] == opposite,
            "zero_injection_target_correct": record["zero_injection_predicted_side"] == target,
            "zero_vs_reset_prediction_differs": (
                record["zero_injection_predicted_side"] != record["action_reset_predicted_side"]
            ),
            "prototype_correct_side_succeeds": record["prototype_correct_side_predicted_side"] == target,
            "prototype_opposite_donor_followed": (record["prototype_opposite_donor_predicted_side"] == opposite),
            "prototype_paired_action_flipped": (
                record["prototype_correct_side_predicted_side"] != record["prototype_opposite_donor_predicted_side"]
            ),
        }
        inconsistent = [field for field, expected in derived.items() if record[field] is not expected]
        if inconsistent:
            raise PilotGateError(f"episodes[{index}] has side-verdict/boolean inconsistencies: {inconsistent}")
        float_fields = {
            field: _finite_float(
                f"episodes[{index}].{field}",
                record[field],
                minimum=None if field == "zero_minus_reset_action_score" else 0.0,
            )
            for field in EPISODE_FLOAT_FIELDS
        }
        for field in EPISODE_FLOAT_FIELDS:
            if field != "zero_minus_reset_action_score" and float_fields[field] > 1.0:
                raise PilotGateError(f"episodes[{index}].{field} must be <= 1")
        if float_fields["attention_uniform_baseline"] <= 0.0:
            raise PilotGateError(f"episodes[{index}].attention_uniform_baseline must be positive")
        output.append({**record, **float_fields})
    return tuple(output)


def load_rung_result(path: Path, *, manifest: artifacts.FrozenManifest) -> RungResult:
    path = Path(path)
    envelope = artifacts.load_canonical_envelope(path, schema_version=RUNG_SCHEMA_VERSION)
    payload = artifacts.require_exact_keys(
        "rung payload",
        envelope["payload"],
        {
            "calibration_artifact",
            "checkpoint",
            "episode_manifest_sha256",
            "episodes",
            "evaluation",
            "evaluation_protocol_sha256",
            "gate_c",
            "initialization_manifest",
            "initialization_parameter_tree_sha256",
            "prototype_artifact",
            "run_provenance",
            "split_assignment_sha256",
            "task_health",
            "train_prototype_loo",
            "train_writer_oof",
        },
    )
    if payload["episode_manifest_sha256"] != manifest.sha256:
        raise PilotGateError(f"rung {path} manifest hash mismatch")
    if payload["split_assignment_sha256"] != manifest.split_assignment_sha256:
        raise PilotGateError(f"rung {path} split-assignment hash mismatch")
    initialization_sha256 = artifacts.require_sha256(
        "initialization_parameter_tree_sha256", payload["initialization_parameter_tree_sha256"]
    )
    calibration = _load_calibration(
        payload["calibration_artifact"],
        rung_path=path,
        manifest=manifest,
        expected_parameter_tree_sha256=initialization_sha256,
    )
    initialization_identity = artifacts.load_initialization_identity(
        owner_path=path,
        descriptor=payload["initialization_manifest"],
        manifest=manifest,
        expected_parameter_tree_sha256=initialization_sha256,
        calibration_id=calibration.artifact_id,
        calibration_file_sha256=calibration.file_sha256,
        calibration_raw_gate_sha256=calibration.raw_gate_sha256,
        calibration_parameters=calibration.parameters,
    )
    if initialization_identity.official_source_tree_sha256 != calibration.official_base_source_sha256:
        raise PilotGateError("calibration official-base source hash does not match initialization identity")
    evaluation_protocol_sha256 = artifacts.require_sha256(
        "evaluation_protocol_sha256", payload["evaluation_protocol_sha256"]
    )
    checkpoint = artifacts.require_exact_keys(
        "checkpoint", payload["checkpoint"], {"completed_updates", "parameter_tree_sha256"}
    )
    completed_updates = checkpoint["completed_updates"]
    if type(completed_updates) is not int or completed_updates not in FIXED_EXTENSION_RUNGS:
        raise PilotGateError(f"completed updates {completed_updates!r} is not a preregistered Gate-D rung")
    checkpoint_sha256 = artifacts.require_sha256(
        "checkpoint.parameter_tree_sha256", checkpoint["parameter_tree_sha256"]
    )
    if completed_updates == 0 and checkpoint_sha256 != initialization_sha256:
        raise PilotGateError("rung-0 checkpoint parameter tree must equal the authenticated initialization tree")
    raw_evaluation = payload["evaluation"]
    if not isinstance(raw_evaluation, dict):
        raise PilotGateError("evaluation must be an object")
    allowed_evaluation_keys = {"episode_aggregation", "final_test_accessed", "split", "raw_artifact"}
    if not {"episode_aggregation", "final_test_accessed", "split"} <= set(raw_evaluation) or (
        set(raw_evaluation) - allowed_evaluation_keys
    ):
        raise PilotGateError("evaluation has missing or unknown keys")
    raw_artifact_descriptor = raw_evaluation.get("raw_artifact")
    evaluation = {key: raw_evaluation[key] for key in ("episode_aggregation", "final_test_accessed", "split")}
    if evaluation != {
        "episode_aggregation": EPISODE_AGGREGATION,
        "final_test_accessed": False,
        "split": "development",
    }:
        raise PilotGateError("Gate-D rung evaluation must be development-only, episode-first, and final-test sealed")
    prototype = _load_prototype(
        payload["prototype_artifact"],
        rung_path=path,
        manifest=manifest,
        calibration_artifact_id=calibration.artifact_id,
        prototype_injected_rms_target=calibration.prototype_injected_rms_target,
        checkpoint_parameter_tree_sha256=checkpoint_sha256,
    )
    train_writer = _validate_train_counts(
        payload["train_writer_oof"],
        name="train_writer_oof",
        protocol=TRAIN_WRITER_PROTOCOL,
        boolean_fields=("counterfactual_prompt_correct", "natural_prompt_correct"),
        manifest=manifest,
    )
    train_prototype = _validate_train_counts(
        payload["train_prototype_loo"],
        name="train_prototype_loo",
        protocol=TRAIN_PROTOTYPE_PROTOCOL,
        boolean_fields=("correct_side", "opposite_donor_follow"),
        manifest=manifest,
    )
    gate_c = _validate_gate_c(payload["gate_c"])
    task_health = _validate_task_health(payload["task_health"], completed_updates=completed_updates)
    episodes = _validate_episode_results(payload["episodes"], manifest=manifest)

    # Production rung artifacts link their raw measurement envelope.  Re-run its reducer here
    # and compare every reducer-facing value, so a producer cannot authorize itself by writing
    # convenient episode booleans or Gate-C/task-health summaries.  The legacy no-link form is
    # retained only for old unit fixtures and historical artifacts; the v3.5 rung producer
    # always emits the authenticated link and the training authorization consumes its result.
    if raw_artifact_descriptor is not None:
        import v35_rung_eval as rung_eval

        if evaluation_protocol_sha256 != rung_eval.EVALUATION_PROTOCOL_SHA256:
            raise PilotGateError("linked raw rung uses a non-frozen evaluation protocol SHA256")
        reduced = rung_eval.verify_rung_raw_link(
            rung_path=path,
            descriptor=raw_artifact_descriptor,
            manifest=manifest,
            completed_updates=completed_updates,
            checkpoint_parameter_tree_sha256=checkpoint_sha256,
            initialization_parameter_tree_sha256=initialization_sha256,
            initialization_identity_sha256=initialization_identity.identity_sha256,
            initialization_manifest_file_sha256=initialization_identity.file_sha256,
            calibration_artifact_id=calibration.artifact_id,
            calibration_passes=True,
            prototype_artifact_id=prototype.artifact_id,
        )
        expected_raw_reduction = {
            "episodes": list(episodes),
            "gate_c": gate_c,
            "task_health": task_health,
            "train_prototype_loo": payload["train_prototype_loo"],
            "train_writer_oof": payload["train_writer_oof"],
        }
        if reduced != expected_raw_reduction:
            raise PilotGateError("rung summaries do not equal deterministic reduction of linked raw measurements")

    return RungResult(
        path=path,
        artifact_id=envelope["artifact_id"],
        file_sha256=artifacts.sha256_bytes(path.read_bytes()),
        completed_updates=completed_updates,
        checkpoint_parameter_tree_sha256=checkpoint_sha256,
        initialization_parameter_tree_sha256=initialization_sha256,
        initialization_identity=initialization_identity,
        calibration=calibration,
        evaluation_protocol_sha256=evaluation_protocol_sha256,
        prototype=prototype,
        run_provenance=_validate_run_provenance(payload["run_provenance"], completed_updates=completed_updates),
        gate_c=gate_c,
        task_health=task_health,
        train_writer_oof=train_writer,
        train_prototype_loo=train_prototype,
        episodes=episodes,
    )


def _gate_c_summary(rung: RungResult) -> dict[str, Any]:
    metrics = rung.gate_c["metrics"]
    checks_pass = all(rung.gate_c["checks"].values())
    expected_ratio = rung.calibration.p90_retention_factor
    computed_ratio_error = abs(metrics["retention_norm_ratio_p90_delay"] - expected_ratio) / max(expected_ratio, 1e-12)
    reported_ratio_error_matches = math.isclose(
        metrics["retention_norm_ratio_relative_error_p90_delay"],
        computed_ratio_error,
        rel_tol=1e-5,
        abs_tol=1e-7,
    )
    metric_gates = {
        "synthetic_commit_residual": metrics["synthetic_fp32_commit_residual_max"] <= SYNTHETIC_RESIDUAL_MAX,
        "production_commit_residual": (
            metrics["production_relative_commit_residual_p95"] <= PRODUCTION_RESIDUAL_P95_MAX
        ),
        "retention_cosine": metrics["retention_cosine_p90_delay"] >= RETENTION_COSINE_MIN,
        "retention_norm_ratio_error": (
            reported_ratio_error_matches and computed_ratio_error <= RETENTION_RATIO_ERROR_MAX
        ),
        "retention_norm_ratio": metrics["retention_norm_ratio_p90_delay"] >= RETENTION_NORM_RATIO_MIN,
        "retained_injection_amplitude": (metrics["retained_injection_amplitude_p90_delay"] >= RETAINED_AMPLITUDE_MIN),
        "real_noise": metrics["real_noise_injected_to_residual_ratio_p95"] <= REAL_NOISE_RATIO_P95_MAX,
    }
    return {
        "checks_pass": checks_pass,
        "metric_gates": metric_gates,
        "passes": checks_pass and all(metric_gates.values()),
        "metrics": dict(metrics),
        "frozen_p90_n_delay": rung.calibration.p90_n_delay,
        "expected_retention_norm_ratio": expected_ratio,
        "computed_retention_norm_ratio_relative_error": computed_ratio_error,
    }


def _task_health_summary(rung: RungResult) -> dict[str, Any]:
    health = rung.task_health
    source = health["fresh_source_reference"]
    step0 = health["v35_step0"]
    current = health["rung"]
    severe = health["severe_clip"]
    cap = health["feature_cap"]
    severe_rate = severe["severe_steps"] / severe["total_optimizer_steps"] if severe["total_optimizer_steps"] else 0.0
    cap_rate = cap["bound_terms"] / cap["eligible_terms"] if cap["eligible_terms"] else 0.0
    gates = {
        "step0_flow_vs_source": step0["flow_loss"] <= FLOW_RATIO_MAX * source["flow_loss"],
        "step0_ce_vs_source": step0["subtask_ce"] <= source["subtask_ce"] + SUBTASK_CE_DELTA_MAX,
        "rung_flow_vs_source": current["flow_loss"] <= FLOW_RATIO_MAX * source["flow_loss"],
        "rung_flow_vs_step0": current["flow_loss"] <= FLOW_RATIO_MAX * step0["flow_loss"],
        "rung_ce_vs_source": current["subtask_ce"] <= source["subtask_ce"] + SUBTASK_CE_DELTA_MAX,
        "rung_ce_vs_step0": current["subtask_ce"] <= step0["subtask_ce"] + SUBTASK_CE_DELTA_MAX,
        "all_finite": all(health["finiteness"].values()),
        "severe_clip_rate": severe_rate <= SEVERE_CLIP_RATE_MAX,
        "feature_cap_bind_rate": cap_rate <= FEATURE_CAP_BIND_RATE_MAX,
    }
    return {
        "passes": all(gates.values()),
        "gates": gates,
        "severe_clip_rate": severe_rate,
        "feature_cap_bind_rate": cap_rate,
        "fresh_source_reference": dict(source),
        "v35_step0": dict(step0),
        "rung": dict(current),
    }


def summarize_gate_c(rung: RungResult) -> dict[str, Any]:
    """Public deterministic Gate-C summary used by the launch authorization reducer."""

    return _gate_c_summary(rung)


def summarize_task_health(rung: RungResult) -> dict[str, Any]:
    """Public deterministic task-health summary used by the launch authorization reducer."""

    return _task_health_summary(rung)


def _episode_counts(rung: RungResult) -> dict[str, int]:
    return {field: sum(bool(record[field]) for record in rung.episodes) for field in EPISODE_BOOLEAN_FIELDS}


def _writer_claim_summary(rung: RungResult, counts: Mapping[str, int]) -> dict[str, Any]:
    train_natural = rung.train_writer_oof["natural_prompt_correct_count"] / TRAIN_EPISODES
    train_counterfactual = rung.train_writer_oof["counterfactual_prompt_correct_count"] / TRAIN_EPISODES
    preliminary = (
        train_natural >= WRITER_TRAIN_ACCURACY_MIN
        and train_counterfactual >= WRITER_TRAIN_ACCURACY_MIN
        and counts["writer_natural_correct"] >= PASS_COUNT
        and counts["writer_counterfactual_correct"] >= PASS_COUNT
    )
    direct_carry_required = preliminary
    direct_carry_pass = not direct_carry_required or counts["direct_carry_correct"] >= PASS_COUNT
    supported = preliminary and direct_carry_pass
    return {
        "status": "supported" if supported else "not_supported",
        "gates_only_writer_claim_not_natural_chain": True,
        "train_natural_accuracy": train_natural,
        "train_counterfactual_accuracy": train_counterfactual,
        "train_minimum": WRITER_TRAIN_ACCURACY_MIN,
        "development_natural_count": counts["writer_natural_correct"],
        "development_counterfactual_count": counts["writer_counterfactual_correct"],
        "development_minimum": PASS_COUNT,
        "direct_carry_count": counts["direct_carry_correct"],
        "direct_carry_required": direct_carry_required,
        "direct_carry_pass": direct_carry_pass,
    }


def _attention_summary(rung: RungResult) -> dict[str, Any]:
    natural = np.asarray([record["attention_memory_mass_natural"] for record in rung.episodes], dtype=np.float64)
    reset = np.asarray([record["attention_memory_mass_reset"] for record in rung.episodes], dtype=np.float64)
    zero = np.asarray([record["attention_memory_mass_zero"] for record in rung.episodes], dtype=np.float64)
    uniform = np.asarray([record["attention_uniform_baseline"] for record in rung.episodes], dtype=np.float64)
    return {
        "natural_episode_macro_mean": float(np.mean(natural)),
        "reset_episode_macro_mean": float(np.mean(reset)),
        "zero_episode_macro_mean": float(np.mean(zero)),
        "uniform_episode_macro_mean": float(np.mean(uniform)),
        "natural_enrichment_over_uniform_episode_macro_mean": float(np.mean(natural / uniform)),
        "paired_natural_minus_reset_episode_macro_mean": float(np.mean(natural - reset)),
        "paired_natural_minus_zero_episode_macro_mean": float(np.mean(natural - zero)),
        "is_hard_gate": False,
    }


def _validate_rung_series(rungs: Sequence[RungResult], *, endpoint: int) -> tuple[RungResult, ...]:
    expected_steps = FIXED_1K_RUNGS if endpoint == 1_000 else FIXED_EXTENSION_RUNGS if endpoint == 2_500 else None
    if expected_steps is None:
        raise PilotGateError("Gate D can decide only the fixed 1,000 or one-time 2,500 endpoint")
    by_step = {rung.completed_updates: rung for rung in rungs}
    if len(by_step) != len(rungs) or tuple(sorted(by_step)) != expected_steps:
        raise PilotGateError(f"Gate-D endpoint {endpoint} requires exact rung set {expected_steps}")
    ordered = tuple(by_step[step] for step in expected_steps)
    shared_fields = (
        "initialization_parameter_tree_sha256",
        "calibration_artifact_id",
        "evaluation_protocol_sha256",
    )
    for field in shared_fields:
        if len({getattr(rung, field) for rung in ordered}) != 1:
            raise PilotGateError(f"all rung artifacts must share one {field}")
    if len({rung.initialization_identity.identity_sha256 for rung in ordered}) != 1:
        raise PilotGateError("all rung artifacts must share one authenticated initialization identity")
    calibration_bindings = {
        (
            rung.calibration.file_sha256,
            rung.calibration.official_base_source_sha256,
            rung.calibration.dataset_protocol_sha256,
            tuple(sorted(rung.calibration.parameters.items())),
        )
        for rung in ordered
    }
    if len(calibration_bindings) != 1:
        raise PilotGateError("calibration provenance or c/tau/alpha changed between rungs")
    if len({rung.run_provenance["run_id_sha256"] for rung in ordered}) != 1:
        raise PilotGateError("fixed rungs do not belong to one run identity")
    if len({rung.run_provenance["training_config_sha256"] for rung in ordered}) != 1:
        raise PilotGateError("training configuration changed between fixed rungs")
    for index, rung in enumerate(ordered):
        authorization = rung.run_provenance["extension_authorization_artifact_id"]
        if rung.completed_updates <= 1_000 and authorization is not None:
            raise PilotGateError("pre-extension rungs must not carry an extension authorization")
        if index == 0:
            continue
        previous = ordered[index - 1]
        expected_previous = {
            "completed_updates": previous.completed_updates,
            "data_iterator_state_sha256": previous.run_provenance["data_iterator_state_sha256"],
            "optimizer_state_sha256": previous.run_provenance["optimizer_state_sha256"],
            "parameter_tree_sha256": previous.checkpoint_parameter_tree_sha256,
        }
        if rung.run_provenance["previous_rung"] != expected_previous:
            raise PilotGateError("fixed rung run/optimizer/data lineage is broken")
    health_identity_fields = (
        "no_augmentation_suite_sha256",
        "preprocessing_norm_sha256",
        "rng_inputs_sha256",
        "fresh_source_reference",
        "v35_step0",
    )
    for field in health_identity_fields:
        values = [rung.task_health[field] for rung in ordered]
        if any(value != values[0] for value in values[1:]):
            raise PilotGateError(f"task-health identity/reference {field} changed between rungs")
    for rung in ordered:
        if rung.task_health["preprocessing_norm_sha256"] != rung.initialization_identity.artifact_hashes.get(
            "norm_stats_sha256"
        ):
            raise PilotGateError("task-health suite normalization does not match initialization identity")
    if len({rung.task_health["severe_clip"]["optimizer_clip_threshold"] for rung in ordered}) != 1:
        raise PilotGateError("optimizer clip threshold changed between rungs")
    if len({rung.task_health["feature_cap"]["cap_value"] for rung in ordered}) != 1:
        raise PilotGateError("feature-cotangent cap value changed between rungs")
    return ordered


def _load_prior_decision(
    path: Path,
    *,
    manifest: artifacts.FrozenManifest,
    initialization_sha256: str,
    calibration_artifact_id: str,
    rungs_1k: Sequence[RungResult],
) -> dict[str, Any]:
    decision = artifacts.load_canonical_envelope(path, schema_version=DECISION_SCHEMA_VERSION)
    payload = decision["payload"]
    if (
        payload.get("criteria_version") != CRITERIA_VERSION
        or payload.get("endpoint_completed_updates") != 1_000
        or payload.get("outcome") != "inconclusive"
        or payload.get("action") != "extend_same_run_once_to_2500"
    ):
        raise PilotGateError("2,500 evaluation requires the authenticated inconclusive 1k extension decision")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise PilotGateError("prior 1k decision is missing provenance")
    if (
        provenance.get("episode_manifest_sha256") != manifest.sha256
        or provenance.get("split_assignment_sha256") != manifest.split_assignment_sha256
        or provenance.get("initialization_parameter_tree_sha256") != initialization_sha256
        or provenance.get("calibration_artifact_id") != calibration_artifact_id
    ):
        raise PilotGateError("prior 1k decision provenance does not match the extension run")
    expected_rungs = [
        {
            "artifact_id": rung.artifact_id,
            "completed_updates": rung.completed_updates,
            "file_sha256": rung.file_sha256,
            "checkpoint_parameter_tree_sha256": rung.checkpoint_parameter_tree_sha256,
            "prototype_artifact_id": rung.prototype.artifact_id,
            "prototype_file_sha256": rung.prototype.file_sha256,
        }
        for rung in rungs_1k
    ]
    if provenance.get("rung_artifacts") != expected_rungs:
        raise PilotGateError("2,500 extension does not continue the exact rung artifacts authorized at 1k")
    return decision


def evaluate_gate_d(
    rungs: Sequence[RungResult],
    *,
    manifest: artifacts.FrozenManifest,
    endpoint: int,
    prior_decision_path: Path | None = None,
) -> dict[str, Any]:
    """Apply the fixed count, core, task-health, and one-extension exit rules."""
    ordered = _validate_rung_series(rungs, endpoint=endpoint)
    endpoint_rung = ordered[-1]
    if endpoint == 1_000 and prior_decision_path is not None:
        raise PilotGateError("the initial 1k decision must not consume a prior branch decision")
    prior_decision: dict[str, Any] | None = None
    if endpoint == 2_500:
        if prior_decision_path is None:
            raise PilotGateError("the 2,500 endpoint requires the prior authenticated 1k decision")
        expected_prior_decision = evaluate_gate_d(
            ordered[:4],
            manifest=manifest,
            endpoint=1_000,
        )
        prior_decision = _load_prior_decision(
            prior_decision_path,
            manifest=manifest,
            initialization_sha256=endpoint_rung.initialization_parameter_tree_sha256,
            calibration_artifact_id=endpoint_rung.calibration_artifact_id,
            rungs_1k=ordered[:4],
        )
        if prior_decision != expected_prior_decision:
            raise PilotGateError("prior 1k decision does not exactly equal the deterministic Gate-D reduction")
        if endpoint_rung.run_provenance["extension_authorization_artifact_id"] != prior_decision["artifact_id"]:
            raise PilotGateError("2,500 rung is not bound to the exact authenticated 1k extension authorization")

    rung_gates: dict[str, Any] = {}
    all_core_pass = True
    all_health_pass = True
    for rung in ordered:
        core = _gate_c_summary(rung)
        health = _task_health_summary(rung)
        all_core_pass &= core["passes"]
        all_health_pass &= health["passes"]
        rung_gates[str(rung.completed_updates)] = {
            "gate_c": core,
            "task_health": health,
            "raw_read_rms": rung.gate_c["metrics"]["raw_read_rms"],
            "injected_token_rms": rung.gate_c["metrics"]["injected_token_rms"],
            "reachable_fraction": rung.gate_c["metrics"]["reachable_fraction"],
        }

    counts = _episode_counts(endpoint_rung)
    step0_counts = _episode_counts(ordered[0])
    reset_gates = {
        "read_reset_at_most_4": counts["read_reset_correct"] <= RESET_MAX,
        "action_reset_at_most_4": counts["action_reset_correct"] <= RESET_MAX,
        "zero_injection_at_most_4": counts["zero_injection_target_correct"] <= RESET_MAX,
        "zero_vs_reset_differs_at_most_1": (counts["zero_vs_reset_prediction_differs"] <= ZERO_RESET_DIFF_MAX),
    }
    learned_fields = (
        "read_natural_correct",
        "read_opposite_donor_followed",
        "action_natural_correct",
        "action_opposite_donor_followed",
    )
    prototype_fields = (
        "prototype_correct_side_succeeds",
        "prototype_opposite_donor_followed",
        "prototype_paired_action_flipped",
    )
    learned_counts = {field: counts[field] for field in learned_fields}
    prototype_counts = {field: counts[field] for field in prototype_fields}
    prototype_improvements = {field: counts[field] - step0_counts[field] for field in prototype_fields}

    hard_failure_reasons: list[str] = []
    if not all_core_pass:
        hard_failure_reasons.append("gate_c_failed_at_one_or_more_rungs")
    if not all_health_pass:
        hard_failure_reasons.append("task_health_failed_at_one_or_more_rungs")
    if not all(reset_gates.values()):
        hard_failure_reasons.append("reset_or_zero_control_failed")
    if any(value <= 4 for value in learned_counts.values()):
        hard_failure_reasons.append("learned_natural_or_donor_at_most_4_of_8")
    if any(value <= 5 for value in prototype_counts.values()):
        hard_failure_reasons.append("prototype_oracle_at_most_5_of_8")

    learned_pass = all(value >= PASS_COUNT for value in learned_counts.values())
    learned_inconclusive = not learned_pass and all(value >= 5 for value in learned_counts.values())
    prototype_pass = all(value >= PASS_COUNT for value in prototype_counts.values())
    prototype_inconclusive = (
        not prototype_pass
        and all(value >= 6 for value in prototype_counts.values())
        and all(value >= 1 for value in prototype_improvements.values())
    )
    if not prototype_pass and not prototype_inconclusive and not any(value <= 5 for value in prototype_counts.values()):
        hard_failure_reasons.append("prototype_6_of_8_without_each_count_improving_from_step0")

    if hard_failure_reasons:
        outcome = "fail"
    elif learned_pass and prototype_pass:
        outcome = "pass"
    elif learned_inconclusive or prototype_inconclusive:
        outcome = "inconclusive" if endpoint == 1_000 else "fail"
        if endpoint == 2_500:
            hard_failure_reasons.append("one_time_2500_extension_did_not_pass")
    else:
        outcome = "fail"
        hard_failure_reasons.append("endpoint_does_not_match_a_preregistered_exit_case")

    action = {
        (1_000, "pass"): "continue_to_fixed_10000_budget",
        (1_000, "inconclusive"): "extend_same_run_once_to_2500",
        (1_000, "fail"): "stop_branch",
        (2_500, "pass"): "continue_to_fixed_10000_budget",
        (2_500, "fail"): "stop_branch_no_second_extension",
    }[(endpoint, outcome)]
    writer_claim = _writer_claim_summary(endpoint_rung, counts)
    zero_score_mean = float(np.mean([record["zero_minus_reset_action_score"] for record in endpoint_rung.episodes]))
    payload = {
        "status": "complete",
        "criteria_version": CRITERIA_VERSION,
        "endpoint_completed_updates": endpoint,
        "outcome": outcome,
        "action": action,
        "final_test_remains_sealed": True,
        "counts_out_of_8": counts,
        "step0_counts_out_of_8": step0_counts,
        "control_gates": reset_gates,
        "learned_chain": {
            "counts": learned_counts,
            "pass": learned_pass,
            "inconclusive_band_5_or_6": learned_inconclusive,
        },
        "prototype_oracles": {
            "counts": prototype_counts,
            "improvement_from_step0": prototype_improvements,
            "pass": prototype_pass,
            "inconclusive_only_if_every_count_at_least_6_and_improves": prototype_inconclusive,
            "training_leave_one_out_supporting_only": dict(endpoint_rung.train_prototype_loo),
        },
        "writer_claim": writer_claim,
        "attention_diagnostic": _attention_summary(endpoint_rung),
        "zero_minus_reset_action_score_episode_macro_mean": zero_score_mean,
        "rung_gates": rung_gates,
        "failure_reasons": hard_failure_reasons,
        "provenance": {
            "episode_manifest_sha256": manifest.sha256,
            "split_assignment_sha256": manifest.split_assignment_sha256,
            "initialization_parameter_tree_sha256": endpoint_rung.initialization_parameter_tree_sha256,
            "initialization_identity_sha256": endpoint_rung.initialization_identity.identity_sha256,
            "initialization_manifest_file_sha256": endpoint_rung.initialization_identity.file_sha256,
            "official_base_source_tree_sha256": endpoint_rung.calibration.official_base_source_sha256,
            "calibration_dataset_protocol_sha256": endpoint_rung.calibration.dataset_protocol_sha256,
            "calibration_artifact_id": endpoint_rung.calibration_artifact_id,
            "calibration_file_sha256": endpoint_rung.calibration.file_sha256,
            "prototype_injected_rms_target": endpoint_rung.calibration.prototype_injected_rms_target,
            "evaluation_protocol_sha256": endpoint_rung.evaluation_protocol_sha256,
            "rung_artifacts": [
                {
                    "artifact_id": rung.artifact_id,
                    "completed_updates": rung.completed_updates,
                    "file_sha256": rung.file_sha256,
                    "checkpoint_parameter_tree_sha256": rung.checkpoint_parameter_tree_sha256,
                    "prototype_artifact_id": rung.prototype.artifact_id,
                    "prototype_file_sha256": rung.prototype.file_sha256,
                }
                for rung in ordered
            ],
            "prior_1k_decision_artifact_id": None if prior_decision is None else prior_decision["artifact_id"],
        },
    }
    return artifacts.artifact_envelope(DECISION_SCHEMA_VERSION, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Every path argument must be relative to memory_project; absolute paths and '..' are rejected.",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen schema-v2 70-episode manifest.")
    parser.add_argument("--manifest-sha256", required=True, help="Expected exact manifest byte hash.")
    parser.add_argument(
        "--rung-result",
        type=Path,
        action="append",
        required=True,
        help="Canonical rung result; repeat for every fixed rung.",
    )
    parser.add_argument("--endpoint", type=int, choices=(1_000, 2_500), required=True)
    parser.add_argument(
        "--prior-1k-decision",
        type=Path,
        help="Required only for the single authorized 2,500-update extension.",
    )
    parser.add_argument("--output", type=Path, required=True, help="New canonical Gate-D decision JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        project_paths.configure_v35_runtime_environment()
        manifest_path = _resolve_project_cli_path(args.manifest, name="manifest path")
        rung_paths = [_resolve_project_cli_path(path, name="rung-result path") for path in args.rung_result]
        prior_decision_path = (
            None
            if args.prior_1k_decision is None
            else _resolve_project_cli_path(args.prior_1k_decision, name="prior-1k-decision path")
        )
        output_path = _resolve_project_cli_path(args.output, name="output path")
        manifest = artifacts.load_frozen_manifest(manifest_path, expected_sha256=args.manifest_sha256)
        rungs = [load_rung_result(path, manifest=manifest) for path in rung_paths]
        decision = evaluate_gate_d(
            rungs,
            manifest=manifest,
            endpoint=args.endpoint,
            prior_decision_path=prior_decision_path,
        )
        artifacts.write_canonical_envelope(output_path, decision, schema_version=DECISION_SCHEMA_VERSION)
    except (
        PilotGateError,
        artifacts.GateArtifactError,
        project_paths.ProjectRootError,
        FileExistsError,
        OSError,
    ) as exc:
        parser.error(str(exc))
    payload = decision["payload"]
    print(f"Gate D: {payload['outcome']} -> {payload['action']} ({decision['artifact_id']}); wrote {args.output}")


if __name__ == "__main__":
    main()
