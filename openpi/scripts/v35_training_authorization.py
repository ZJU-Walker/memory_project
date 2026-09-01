"""Aggregate sealed v3.5 gates into pilot and continuation authorizations.

This reducer is deliberately the only place where independent release evidence becomes
permission to run the optimizer.  It never edits an input artifact, every linked path is
``memory_project``-relative, and the output is canonical self-hashed JSON.  Training
revalidates the output and copies its exact bytes into every checkpoint.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import openpi.shared.project_paths as project_paths

# v35_authorization imports the JAX-backed NNX helpers.  Install the portable cache
# contract before that import can initialize any cache-sensitive runtime.
project_paths.configure_v35_runtime_environment()

import openpi.training.v35_authorization as authorization  # noqa: E402

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import v35_gate_artifacts as artifacts  # noqa: E402
import v35_leakage_gate as leakage  # noqa: E402
import v35_pilot_gate as pilot  # noqa: E402

DATA_GATE_SCHEMA_VERSION = "openpi.v35.data-gate-decision.v1"
DATA_GATE_REPORT_SCHEMA_VERSION = "openpi.v35.data-gate-report.v1"
DATA_GATE_CRITERIA_VERSION = "openpi.v35.gate-a.rev5.v1"
STEP0_CRITERIA_VERSION = "openpi.v35.gate-c-step0-task-health.rev5.v1"

DATA_GATE_REQUIRED_CHECKS = frozenset(
    {
        "block_confound_audit_pass",
        "every_train_d_window_has_final_eligible_e_anchor",
        "every_train_episode_has_eligible_sampled_e_step",
        "every_train_episode_has_skip_o_d_candidate",
        "manual_review_complete",
        "stable_id_mapping_exact",
        "stable_split_frozen",
        "successful_commit_accounting_complete",
        "task_vocabulary_exact",
        "train_only_normalization",
        "zero_state_invalid_d_loss_steps",
    }
)


class TrainingAuthorizationError(artifacts.GateArtifactError):
    """Raised when independent gate evidence cannot authorize training."""


def _resolve_project_cli_path(path: Path, *, name: str) -> Path:
    """Resolve one production CLI path inside the portable project root."""
    raw = Path(path)
    if raw.is_absolute():
        raise TrainingAuthorizationError(f"{name} must be relative to memory_project, got {str(raw)!r}")
    if ".." in raw.parts:
        raise TrainingAuthorizationError(f"{name} must not escape memory_project, got {str(raw)!r}")
    try:
        return project_paths.project_path(raw)
    except project_paths.ProjectRootError as exc:
        raise TrainingAuthorizationError(f"invalid {name}: {exc}") from exc


def _project_relative(path: Path, *, name: str) -> str:
    try:
        return project_paths.project_relative_path(path).as_posix()
    except project_paths.ProjectRootError as exc:
        raise TrainingAuthorizationError(f"{name} must be inside memory_project: {path}") from exc


def _descriptor(
    path: Path,
    *,
    artifact_id: str,
    criteria_version: str | None = None,
) -> dict[str, str]:
    artifact_id = artifacts.require_artifact_id("descriptor artifact_id", artifact_id)
    descriptor = {
        "artifact_id": artifact_id,
        "path_relative": _project_relative(path, name="authorization evidence"),
        "sha256": artifacts.sha256_bytes(path.read_bytes()),
    }
    if criteria_version is not None:
        descriptor["criteria_version"] = criteria_version
    return descriptor


def _source_checkpoint_descriptor(rung: pilot.RungResult) -> dict[str, Any]:
    """Freeze every live checkpoint input needed to authorize an optimizer resume."""

    return {
        "completed_updates": rung.completed_updates,
        "cumulative_telemetry_sha256": rung.run_provenance["cumulative_telemetry_sha256"],
        "data_iterator_state_sha256": rung.run_provenance["data_iterator_state_sha256"],
        "optimizer_state_sha256": rung.run_provenance["optimizer_state_sha256"],
        "parameter_tree_sha256": rung.checkpoint_parameter_tree_sha256,
        "rung_artifact_id": rung.artifact_id,
        "rung_file_sha256": rung.file_sha256,
        "rung_path_relative": _project_relative(rung.path, name="source checkpoint rung"),
        "runtime_identity_sha256": rung.run_provenance["runtime_identity_sha256"],
    }


def _load_json_object(path: Path, *, name: str) -> tuple[bytes, Mapping[str, Any]]:
    try:
        raw_bytes = Path(path).read_bytes()
        value = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingAuthorizationError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TrainingAuthorizationError(f"{name} must be a JSON object")
    return raw_bytes, value


def _validate_data_gate_detail_report(
    descriptor: Any,
    *,
    dataset_identity: Any,
    manifest: artifacts.FrozenManifest,
) -> Mapping[str, Any]:
    """Authenticate the project-local per-episode report behind Gate A."""

    descriptor = artifacts.require_exact_keys(
        "Gate-A detail_report descriptor",
        descriptor,
        {"artifact_id", "path_relative", "sha256"},
    )
    artifact_id = artifacts.require_artifact_id("Gate-A detail_report artifact_id", descriptor["artifact_id"])
    file_sha256 = artifacts.require_sha256("Gate-A detail_report sha256", descriptor["sha256"])
    relative = descriptor["path_relative"]
    if not isinstance(relative, str) or not relative.strip():
        raise TrainingAuthorizationError("Gate-A detail_report path_relative must be a non-empty string")
    try:
        report_path = project_paths.project_path(relative)
    except project_paths.ProjectRootError as exc:
        raise TrainingAuthorizationError(
            f"Gate-A detail_report must be a confined memory_project-relative path: {relative!r}"
        ) from exc
    try:
        actual_file_sha256 = artifacts.sha256_bytes(report_path.read_bytes())
    except OSError as exc:
        raise TrainingAuthorizationError(f"cannot read Gate-A detail_report {report_path}: {exc}") from exc
    if actual_file_sha256 != file_sha256:
        raise TrainingAuthorizationError("Gate-A detail_report file SHA256 does not match its descriptor")
    report = artifacts.load_canonical_envelope(report_path, schema_version=DATA_GATE_REPORT_SCHEMA_VERSION)
    if report["artifact_id"] != artifact_id:
        raise TrainingAuthorizationError("Gate-A detail_report artifact ID does not match its descriptor")
    report_payload = report["payload"]
    if (
        not isinstance(report_payload, Mapping)
        or report_payload.get("dataset_identity") != dataset_identity
        or report_payload.get("episode_manifest_id") != f"sha256:{manifest.sha256}"
    ):
        raise TrainingAuthorizationError(
            "Gate-A detail_report does not bind the decision dataset identity and frozen episode manifest"
        )
    return report


def load_data_gate_decision(path: Path, *, manifest: artifacts.FrozenManifest) -> Mapping[str, Any]:
    envelope = artifacts.load_canonical_envelope(Path(path), schema_version=DATA_GATE_SCHEMA_VERSION)
    payload = artifacts.require_exact_keys(
        "Gate-A payload",
        envelope["payload"],
        {
            "checks",
            "criteria_version",
            "dataset_identity",
            "detail_report",
            "episode_manifest_sha256",
            "final_test_accessed",
            "population",
            "split_assignment_sha256",
            "status",
            "training_counts",
        },
    )
    if (
        payload["criteria_version"] != DATA_GATE_CRITERIA_VERSION
        or payload["status"] != "pass"
        or payload["episode_manifest_sha256"] != manifest.sha256
        or payload["split_assignment_sha256"] != manifest.split_assignment_sha256
        or payload["final_test_accessed"] is not False
    ):
        raise TrainingAuthorizationError("Gate A is not a passing decision on the frozen manifest")
    population = artifacts.require_exact_keys(
        "Gate-A population", payload["population"], {"development", "final_test", "total", "train"}
    )
    if population != {"development": 8, "final_test": 8, "total": 70, "train": 54}:
        raise TrainingAuthorizationError("Gate A must cover the exact frozen 70/54/8/8 population")
    checks = artifacts.require_exact_keys("Gate-A checks", payload["checks"], set(DATA_GATE_REQUIRED_CHECKS))
    if any(value is not True for value in checks.values()):
        failed = sorted(name for name, value in checks.items() if value is not True)
        raise TrainingAuthorizationError(f"Gate A has failed or non-boolean checks: {failed}")
    counts = artifacts.require_exact_keys(
        "Gate-A training counts",
        payload["training_counts"],
        {
            "episodes_with_final_eligible_e_anchor",
            "episodes_with_skip_o_d_candidate",
            "episodes_with_successful_commit_accounting",
            "minimum_eligible_sampled_e_steps",
            "state_invalid_d_loss_steps",
            "training_episodes",
        },
    )
    if counts != {
        "episodes_with_final_eligible_e_anchor": 54,
        "episodes_with_skip_o_d_candidate": 54,
        "episodes_with_successful_commit_accounting": 54,
        "minimum_eligible_sampled_e_steps": counts.get("minimum_eligible_sampled_e_steps"),
        "state_invalid_d_loss_steps": 0,
        "training_episodes": 54,
    }:
        raise TrainingAuthorizationError("Gate-A episode/state accounting is incomplete")
    minimum = counts["minimum_eligible_sampled_e_steps"]
    if type(minimum) is not int or minimum < 1:
        raise TrainingAuthorizationError("Gate A requires at least one eligible sampled E step in every train episode")
    detail_report = _validate_data_gate_detail_report(
        payload["detail_report"],
        dataset_identity=payload["dataset_identity"],
        manifest=manifest,
    )
    return {"detail_report": detail_report, "envelope": envelope, "payload": payload}


def _load_gate_b(path: Path, *, manifest: artifacts.FrozenManifest) -> Mapping[str, Any]:
    envelope = artifacts.load_canonical_envelope(Path(path), schema_version=leakage.DECISION_SCHEMA_VERSION)
    payload = envelope["payload"]
    provenance = payload.get("provenance") if isinstance(payload, Mapping) else None
    gates = payload.get("gates") if isinstance(payload, Mapping) else None
    population = payload.get("population") if isinstance(payload, Mapping) else None
    if (
        payload.get("status") != "pass"
        or payload.get("criteria_version") != leakage.CRITERIA_VERSION
        or not isinstance(gates, Mapping)
        or gates.get("passes") is not True
        or not isinstance(population, Mapping)
        or population.get("split") != "train"
        or population.get("episode_count") != 54
        or not isinstance(provenance, Mapping)
        or provenance.get("episode_manifest_sha256") != manifest.sha256
        or provenance.get("split_assignment_sha256") != manifest.split_assignment_sha256
    ):
        raise TrainingAuthorizationError("Gate B is not a passing train-54 decision on the frozen manifest")
    return {"envelope": envelope, "payload": payload}


def _norm_identity(
    *,
    norm_stats_path: Path,
    norm_provenance_path: Path,
    manifest: artifacts.FrozenManifest,
) -> dict[str, str]:
    norm_bytes, _ = _load_json_object(norm_stats_path, name="norm stats")
    provenance_bytes, provenance = _load_json_object(norm_provenance_path, name="norm provenance")
    manifest_info = provenance.get("manifest")
    selection = provenance.get("selection")
    storage = provenance.get("train_storage")
    computation = provenance.get("computation")
    norm_info = provenance.get("norm_stats")
    if not all(isinstance(item, Mapping) for item in (manifest_info, selection, storage, computation, norm_info)):
        raise TrainingAuthorizationError("norm provenance is missing a required object")
    if (
        provenance.get("schema_version") != 2
        or provenance.get("status") != "complete"
        or manifest_info.get("sha256") != manifest.sha256
        or manifest_info.get("active_split") != "train"
        or "path" in manifest_info
        or selection.get("selected_num_episodes") != 54
        or storage.get("root_contract") != "memory_project-relative-v1"
        or "root" in storage
        or storage.get("scope") != "selected train episode parquet, optional videos, plus structural meta files"
        or computation.get("protocol") != "raw-train-rows-delta-action-horizon-v1"
        or norm_info.get("sha256") != artifacts.sha256_bytes(norm_bytes)
    ):
        raise TrainingAuthorizationError("norm provenance is not the portable frozen train-54 raw-row protocol")
    storage_sha256 = artifacts.require_sha256("train_storage.sha256", storage.get("sha256"))
    return {
        "episode_manifest_sha256": manifest.sha256,
        "norm_computation_protocol": computation["protocol"],
        "norm_stats_provenance_sha256": artifacts.sha256_bytes(provenance_bytes),
        "norm_stats_sha256": artifacts.sha256_bytes(norm_bytes),
        "split_assignment_sha256": manifest.split_assignment_sha256,
        "train_storage_sha256": storage_sha256,
    }


def build_pilot_authorization(
    *,
    gate_a_path: Path,
    gate_b_path: Path,
    step0_rung_path: Path,
    manifest: artifacts.FrozenManifest,
    norm_stats_path: Path,
    norm_provenance_path: Path,
    config_name: str,
    experiment_name: str,
    initialization_seed: int,
    semantic_training_config_sha256: str,
    prepilot_source_identity_path: Path,
) -> dict[str, Any]:
    """Reduce Gate A/B and the Gate-C/task-health step-0 rung into pilot permission."""

    semantic_training_config_sha256 = artifacts.require_sha256(
        "semantic_training_config_sha256", semantic_training_config_sha256
    )
    prepilot_source = authorization.load_and_validate_frozen_source_identity(prepilot_source_identity_path)
    gate_a = load_data_gate_decision(gate_a_path, manifest=manifest)
    gate_b = _load_gate_b(gate_b_path, manifest=manifest)
    step0 = pilot.load_rung_result(step0_rung_path, manifest=manifest)
    if step0.completed_updates != 0:
        raise TrainingAuthorizationError("prelaunch Gate C/task health must be the completed-update-0 rung")
    gate_c_summary = pilot.summarize_gate_c(step0)
    task_health_summary = pilot.summarize_task_health(step0)
    if not gate_c_summary["passes"] or not task_health_summary["passes"]:
        raise TrainingAuthorizationError("Gate C and step-0 task health must both pass before pilot launch")
    if step0.run_provenance["training_config_sha256"] != semantic_training_config_sha256:
        raise TrainingAuthorizationError("step-0 rung was evaluated with another semantic training config")
    if step0.initialization_identity.config_name != config_name:
        raise TrainingAuthorizationError("step-0 initialization config name differs from the requested launch")
    if step0.initialization_identity.initialization_seed != initialization_seed:
        raise TrainingAuthorizationError("step-0 initialization seed differs from the requested launch")

    run_id = authorization.run_id_sha256(
        config_name=config_name,
        experiment_name=experiment_name,
        initialization_seed=initialization_seed,
        initialization_parameter_tree_sha256=step0.initialization_parameter_tree_sha256,
        calibration_artifact_id=step0.calibration_artifact_id,
        semantic_config_sha256=semantic_training_config_sha256,
    )
    if step0.run_provenance["run_id_sha256"] != run_id:
        raise TrainingAuthorizationError("step-0 rung run ID is not the deterministic requested-run identity")

    gate_b_provenance = gate_b["payload"]["provenance"]
    expected_gate_b = {
        "initialization_parameter_tree_sha256": step0.initialization_parameter_tree_sha256,
        "initialization_manifest_sha256": step0.initialization_identity.file_sha256,
        "official_base_source_tree_sha256": step0.initialization_identity.official_source_tree_sha256,
    }
    if any(gate_b_provenance.get(name) != value for name, value in expected_gate_b.items()):
        raise TrainingAuthorizationError("Gate B and step 0 do not share one official fresh initialization")

    dataset_identity = _norm_identity(
        norm_stats_path=norm_stats_path,
        norm_provenance_path=norm_provenance_path,
        manifest=manifest,
    )
    expected_gate_a_dataset = {
        **dataset_identity,
        "calibration_dataset_protocol_sha256": step0.calibration.dataset_protocol_sha256,
    }
    gate_a_dataset = artifacts.require_exact_keys(
        "Gate-A dataset_identity",
        gate_a["payload"]["dataset_identity"],
        set(expected_gate_a_dataset),
    )
    if dict(gate_a_dataset) != expected_gate_a_dataset:
        raise TrainingAuthorizationError("Gate A does not bind the current frozen manifest/storage/norm identity")
    identity_hashes = step0.initialization_identity.artifact_hashes
    if (
        identity_hashes.get("episode_manifest_sha256") != dataset_identity["episode_manifest_sha256"]
        or identity_hashes.get("norm_stats_sha256") != dataset_identity["norm_stats_sha256"]
        or identity_hashes.get("norm_stats_provenance_sha256") != dataset_identity["norm_stats_provenance_sha256"]
        or identity_hashes.get("train_storage_sha256") != dataset_identity["train_storage_sha256"]
    ):
        raise TrainingAuthorizationError("step0/calibration identity is not bound to the Gate-A dataset")

    payload = {
        "authorization_kind": authorization.PILOT_AUTHORIZATION_KIND,
        "authorized_target_completed_updates": 1_000,
        "calibration_identity": {
            "artifact_id": step0.calibration.artifact_id,
            "collector_source_sha256": _calibration_provenance(step0.path)["collector_source_sha256"],
            "dataset_protocol_sha256": step0.calibration.dataset_protocol_sha256,
            "file_sha256": step0.calibration.file_sha256,
            "preflight_sha256": _calibration_provenance(step0.path)["preflight_sha256"],
            "replay_protocol_sha256": _calibration_provenance(step0.path)["replay_protocol_sha256"],
        },
        "config_identity": {
            "config_name": config_name,
            "experiment_name": experiment_name,
            "initialization_seed": initialization_seed,
        },
        "dataset_identity": dataset_identity,
        "final_test_remains_sealed": True,
        "gate_evidence": {
            "gate_a": _descriptor(
                gate_a_path,
                artifact_id=gate_a["envelope"]["artifact_id"],
                criteria_version=DATA_GATE_CRITERIA_VERSION,
            ),
            "gate_b": _descriptor(
                gate_b_path,
                artifact_id=gate_b["envelope"]["artifact_id"],
                criteria_version=leakage.CRITERIA_VERSION,
            ),
            "step0": _descriptor(
                step0_rung_path,
                artifact_id=step0.artifact_id,
                criteria_version=STEP0_CRITERIA_VERSION,
            ),
        },
        "initialization_identity": {
            "initialization_identity_sha256": step0.initialization_identity.identity_sha256,
            "initialization_manifest_file_sha256": step0.initialization_identity.file_sha256,
            "official_source_tree_sha256": step0.initialization_identity.official_source_tree_sha256,
            "official_source_uri": authorization.OFFICIAL_BASE_URI,
            "parameter_tree_sha256": step0.initialization_parameter_tree_sha256,
        },
        "prepilot_source_identity": {
            "artifact_id": prepilot_source.artifact_id,
            "path_relative": _project_relative(
                prepilot_source.path,
                name="prepilot source identity",
            ),
            "sha256": prepilot_source.file_sha256,
            "source_aggregate_sha256": prepilot_source.source_identity["aggregate_sha256"],
        },
        "protocols": {
            "calibration_schema_version": "openpi.v35.injection-calibration.v1",
            "checkpoint_labels": "completed_optimizer_updates",
            "gate_a_criteria_version": DATA_GATE_CRITERIA_VERSION,
            "gate_b_criteria_version": leakage.CRITERIA_VERSION,
            "step0_criteria_version": STEP0_CRITERIA_VERSION,
        },
        "run_identity": {"run_id_sha256": run_id},
        "semantic_config_schema_version": authorization.SEMANTIC_CONFIG_SCHEMA_VERSION,
        "semantic_training_config_sha256": semantic_training_config_sha256,
        "source_checkpoint": _source_checkpoint_descriptor(step0),
        "status": "pass",
    }
    return authorization.authorization_envelope(payload)


def _calibration_provenance(rung_path: Path) -> Mapping[str, Any]:
    rung = artifacts.load_canonical_envelope(rung_path, schema_version=pilot.RUNG_SCHEMA_VERSION)
    descriptor = rung["payload"]["calibration_artifact"]
    calibration_path, _ = artifacts.resolve_hashed_relative_file(
        owner_path=rung_path,
        descriptor={"path": descriptor["path"], "sha256": descriptor["sha256"]},
        descriptor_name="calibration_artifact",
    )
    _, calibration = _load_json_object(calibration_path, name="calibration artifact")
    provenance = calibration.get("payload", {}).get("provenance")
    if not isinstance(provenance, Mapping):
        raise TrainingAuthorizationError("calibration artifact is missing provenance")
    return provenance


def _load_pilot_for_reducer(path: Path) -> authorization.AuthorizationRecord:
    record = authorization.load_authorization(path, expected_kind=authorization.PILOT_AUTHORIZATION_KIND)
    # The runtime validator performs config/asset comparisons.  The reducer still requires the
    # complete strict pilot payload before it may inherit any identity fields.
    required = {
        "authorization_kind",
        "authorized_target_completed_updates",
        "calibration_identity",
        "config_identity",
        "dataset_identity",
        "final_test_remains_sealed",
        "gate_evidence",
        "initialization_identity",
        "prepilot_source_identity",
        "protocols",
        "run_identity",
        "semantic_config_schema_version",
        "semantic_training_config_sha256",
        "source_checkpoint",
        "status",
    }
    artifacts.require_exact_keys("pilot authorization payload", record.payload, required)
    return record


def build_continuation_authorization(
    *,
    pilot_authorization_path: Path,
    gate_d_decision_path: Path,
    rung_paths: Sequence[Path],
    manifest: artifacts.FrozenManifest,
    endpoint: int,
    target: int,
    prior_1k_decision_path: Path | None = None,
    prior_continuation_authorization_path: Path | None = None,
) -> dict[str, Any]:
    """Reduce an authenticated Gate-D outcome into one exact continuation target."""

    if target not in (2_500, 10_000):
        raise TrainingAuthorizationError("continuation target must be exactly 2,500 or 10,000 updates")
    pilot_authorization = _load_pilot_for_reducer(pilot_authorization_path)
    rungs = [pilot.load_rung_result(path, manifest=manifest) for path in rung_paths]
    expected_decision = pilot.evaluate_gate_d(
        rungs,
        manifest=manifest,
        endpoint=endpoint,
        prior_decision_path=prior_1k_decision_path,
    )
    decision = artifacts.load_canonical_envelope(
        gate_d_decision_path,
        schema_version=pilot.DECISION_SCHEMA_VERSION,
    )
    if decision != expected_decision:
        raise TrainingAuthorizationError("Gate-D decision does not equal deterministic reduction of supplied rungs")
    decision_payload = decision["payload"]
    expected_exit = (
        (1_000, "inconclusive", "extend_same_run_once_to_2500")
        if target == 2_500
        else (endpoint, "pass", "continue_to_fixed_10000_budget")
    )
    if (
        endpoint not in (1_000, 2_500)
        or (
            decision_payload["endpoint_completed_updates"],
            decision_payload["outcome"],
            decision_payload["action"],
        )
        != expected_exit
    ):
        raise TrainingAuthorizationError("Gate-D exit does not authorize the requested continuation")
    if target == 2_500 and endpoint != 1_000:
        raise TrainingAuthorizationError("the one-time 2.5k extension can be authorized only at 1k")

    endpoint_rung = next((rung for rung in rungs if rung.completed_updates == endpoint), None)
    if endpoint_rung is None:
        raise TrainingAuthorizationError("supplied rungs do not contain the Gate-D endpoint")
    pilot_payload = pilot_authorization.payload
    if (
        endpoint_rung.run_provenance["run_id_sha256"] != pilot_payload["run_identity"]["run_id_sha256"]
        or endpoint_rung.run_provenance["training_config_sha256"] != pilot_payload["semantic_training_config_sha256"]
        or endpoint_rung.initialization_parameter_tree_sha256
        != pilot_payload["initialization_identity"]["parameter_tree_sha256"]
        or endpoint_rung.initialization_identity.identity_sha256
        != pilot_payload["initialization_identity"]["initialization_identity_sha256"]
        or endpoint_rung.calibration_artifact_id != pilot_payload["calibration_identity"]["artifact_id"]
    ):
        raise TrainingAuthorizationError("Gate-D endpoint does not belong to the pilot-authorized run")

    prior_descriptor = None
    if target == 10_000 and endpoint == 2_500:
        if prior_continuation_authorization_path is None:
            raise TrainingAuthorizationError("a 2.5k pass requires the prior one-time continuation authorization")
        prior = authorization.load_authorization(
            prior_continuation_authorization_path,
            expected_kind=authorization.CONTINUATION_AUTHORIZATION_KIND,
        )
        if (
            prior.payload.get("authorized_target_completed_updates") != 2_500
            or prior.payload.get("run_identity") != pilot_payload["run_identity"]
            or prior.payload.get("gate_d", {}).get("artifact_id")
            != decision_payload["provenance"]["prior_1k_decision_artifact_id"]
        ):
            raise TrainingAuthorizationError("prior 2.5k authorization is not the one used by this Gate-D series")
        prior_descriptor = _descriptor(
            prior_continuation_authorization_path,
            artifact_id=prior.artifact_id,
        )
    elif prior_continuation_authorization_path is not None:
        raise TrainingAuthorizationError("prior continuation authorization is valid only for a 2.5k->10k pass")

    payload = {
        "authorization_kind": authorization.CONTINUATION_AUTHORIZATION_KIND,
        "authorized_target_completed_updates": target,
        "calibration_identity": pilot_payload["calibration_identity"],
        "config_identity": pilot_payload["config_identity"],
        "final_test_remains_sealed": True,
        "gate_d": {
            "action": decision_payload["action"],
            "artifact_id": decision["artifact_id"],
            "criteria_version": pilot.CRITERIA_VERSION,
            "endpoint_completed_updates": endpoint,
            "file_sha256": artifacts.sha256_bytes(gate_d_decision_path.read_bytes()),
            "outcome": decision_payload["outcome"],
            "path_relative": _project_relative(gate_d_decision_path, name="Gate-D decision"),
        },
        "initialization_identity": pilot_payload["initialization_identity"],
        "pilot_authorization": _descriptor(
            pilot_authorization_path,
            artifact_id=pilot_authorization.artifact_id,
        ),
        "prior_continuation_authorization": prior_descriptor,
        "run_identity": pilot_payload["run_identity"],
        "semantic_config_schema_version": authorization.SEMANTIC_CONFIG_SCHEMA_VERSION,
        "semantic_training_config_sha256": pilot_payload["semantic_training_config_sha256"],
        "source_checkpoint": {
            **_source_checkpoint_descriptor(endpoint_rung),
        },
        "status": "pass",
    }
    return authorization.authorization_envelope(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Every path argument must be relative to memory_project; absolute paths and '..' are rejected.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    pilot_parser = subparsers.add_parser("pilot", help="Authorize only the frozen 1,000-update pilot.")
    pilot_parser.add_argument("--manifest", type=Path, required=True)
    pilot_parser.add_argument("--manifest-sha256", required=True)
    pilot_parser.add_argument("--gate-a", type=Path, required=True)
    pilot_parser.add_argument("--gate-b", type=Path, required=True)
    pilot_parser.add_argument("--step0-rung", type=Path, required=True)
    pilot_parser.add_argument("--norm-stats", type=Path, required=True)
    pilot_parser.add_argument("--norm-provenance", type=Path, required=True)
    pilot_parser.add_argument("--config-name", required=True)
    pilot_parser.add_argument("--experiment-name", required=True)
    pilot_parser.add_argument("--initialization-seed", type=int, required=True)
    pilot_parser.add_argument("--semantic-training-config-sha256", required=True)
    pilot_parser.add_argument("--prepilot-source-identity", type=Path, required=True)
    pilot_parser.add_argument("--output", type=Path, required=True)

    continuation = subparsers.add_parser("continuation", help="Authorize one Gate-D continuation target.")
    continuation.add_argument("--manifest", type=Path, required=True)
    continuation.add_argument("--manifest-sha256", required=True)
    continuation.add_argument("--pilot-authorization", type=Path, required=True)
    continuation.add_argument("--gate-d-decision", type=Path, required=True)
    continuation.add_argument("--rung-result", type=Path, action="append", required=True)
    continuation.add_argument("--endpoint", type=int, choices=(1_000, 2_500), required=True)
    continuation.add_argument("--target", type=int, choices=(2_500, 10_000), required=True)
    continuation.add_argument("--prior-1k-decision", type=Path)
    continuation.add_argument("--prior-continuation-authorization", type=Path)
    continuation.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        project_paths.configure_v35_runtime_environment()
        manifest_path = _resolve_project_cli_path(args.manifest, name="manifest path")
        output_path = _resolve_project_cli_path(args.output, name="output path")
        manifest = artifacts.load_frozen_manifest(manifest_path, expected_sha256=args.manifest_sha256)
        if args.command == "pilot":
            result = build_pilot_authorization(
                gate_a_path=_resolve_project_cli_path(args.gate_a, name="Gate-A path"),
                gate_b_path=_resolve_project_cli_path(args.gate_b, name="Gate-B path"),
                step0_rung_path=_resolve_project_cli_path(args.step0_rung, name="step-0 rung path"),
                manifest=manifest,
                norm_stats_path=_resolve_project_cli_path(args.norm_stats, name="norm-stats path"),
                norm_provenance_path=_resolve_project_cli_path(
                    args.norm_provenance,
                    name="norm-provenance path",
                ),
                config_name=args.config_name,
                experiment_name=args.experiment_name,
                initialization_seed=args.initialization_seed,
                semantic_training_config_sha256=args.semantic_training_config_sha256,
                prepilot_source_identity_path=_resolve_project_cli_path(
                    args.prepilot_source_identity,
                    name="prepilot-source-identity path",
                ),
            )
        else:
            result = build_continuation_authorization(
                pilot_authorization_path=_resolve_project_cli_path(
                    args.pilot_authorization,
                    name="pilot-authorization path",
                ),
                gate_d_decision_path=_resolve_project_cli_path(
                    args.gate_d_decision,
                    name="Gate-D decision path",
                ),
                rung_paths=[_resolve_project_cli_path(path, name="rung-result path") for path in args.rung_result],
                manifest=manifest,
                endpoint=args.endpoint,
                target=args.target,
                prior_1k_decision_path=(
                    None
                    if args.prior_1k_decision is None
                    else _resolve_project_cli_path(args.prior_1k_decision, name="prior-1k-decision path")
                ),
                prior_continuation_authorization_path=(
                    None
                    if args.prior_continuation_authorization is None
                    else _resolve_project_cli_path(
                        args.prior_continuation_authorization,
                        name="prior-continuation-authorization path",
                    )
                ),
            )
        authorization.write_authorization_once(output_path, result)
    except (
        TrainingAuthorizationError,
        artifacts.GateArtifactError,
        authorization.V35AuthorizationError,
        project_paths.ProjectRootError,
        FileExistsError,
        OSError,
    ) as exc:
        parser.error(str(exc))
    print(f"v3.5 {args.command} authorization: {result['artifact_id']} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
