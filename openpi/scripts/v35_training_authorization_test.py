from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest
import v35_gate_artifacts as artifacts
import v35_training_authorization as reducer

import openpi.shared.project_paths as project_paths
import openpi.training.v35_authorization as runtime


def _project(root: pathlib.Path) -> pathlib.Path:
    (root / "openpi/src/openpi").mkdir(parents=True)
    (root / "openpi/pyproject.toml").touch()
    return root


def _manifest() -> artifacts.FrozenManifest:
    return artifacts.FrozenManifest(
        path=pathlib.Path("manifest.json"),
        sha256="a" * 64,
        split_assignment_sha256="b" * 64,
        episodes=(),
    )


def _write_data_gate(
    path: pathlib.Path,
    *,
    report_path: pathlib.Path,
    minimum: int = 1,
    failed: str | None = None,
    report_dataset_identity: dict | None = None,
) -> dict:
    checks = dict.fromkeys(reducer.DATA_GATE_REQUIRED_CHECKS, True)
    if failed is not None:
        checks[failed] = False
    dataset_identity = {}
    report_payload = {
        "dataset_identity": dataset_identity if report_dataset_identity is None else report_dataset_identity,
        "episode_manifest_id": f"sha256:{'a' * 64}",
        "eligible_sampled_e_steps_distribution": {"minimum": minimum},
    }
    report = artifacts.artifact_envelope(reducer.DATA_GATE_REPORT_SCHEMA_VERSION, report_payload)
    artifacts.write_canonical_envelope(
        report_path,
        report,
        schema_version=reducer.DATA_GATE_REPORT_SCHEMA_VERSION,
    )
    payload = {
        "checks": checks,
        "criteria_version": reducer.DATA_GATE_CRITERIA_VERSION,
        "dataset_identity": dataset_identity,
        "detail_report": {
            "artifact_id": report["artifact_id"],
            "path_relative": project_paths.project_relative_path(report_path).as_posix(),
            "sha256": artifacts.sha256_bytes(report_path.read_bytes()),
        },
        "episode_manifest_sha256": "a" * 64,
        "final_test_accessed": False,
        "population": {"development": 8, "final_test": 8, "total": 70, "train": 54},
        "split_assignment_sha256": "b" * 64,
        "status": "pass",
        "training_counts": {
            "episodes_with_final_eligible_e_anchor": 54,
            "episodes_with_skip_o_d_candidate": 54,
            "episodes_with_successful_commit_accounting": 54,
            "minimum_eligible_sampled_e_steps": minimum,
            "state_invalid_d_loss_steps": 0,
            "training_episodes": 54,
        },
    }
    envelope = artifacts.artifact_envelope(reducer.DATA_GATE_SCHEMA_VERSION, payload)
    artifacts.write_canonical_envelope(path, envelope, schema_version=reducer.DATA_GATE_SCHEMA_VERSION)
    return envelope


def test_gate_a_requires_all_episode_eligibility_and_every_named_check(monkeypatch, tmp_path):
    root = _project(tmp_path / "memory_project")
    monkeypatch.setenv(project_paths.MEMORY_PROJECT_ROOT_ENV, str(root))
    diagnostics = root / "v35/diagnostics"
    diagnostics.mkdir(parents=True)
    good = diagnostics / "good.json"
    _write_data_gate(good, report_path=diagnostics / "good-report.json")
    reducer.load_data_gate_decision(good, manifest=_manifest())

    zero = diagnostics / "zero.json"
    _write_data_gate(zero, report_path=diagnostics / "zero-report.json", minimum=0)
    with pytest.raises(reducer.TrainingAuthorizationError, match="at least one eligible"):
        reducer.load_data_gate_decision(zero, manifest=_manifest())

    failed = diagnostics / "failed.json"
    _write_data_gate(
        failed,
        report_path=diagnostics / "failed-report.json",
        failed="zero_state_invalid_d_loss_steps",
    )
    with pytest.raises(reducer.TrainingAuthorizationError, match="failed or non-boolean"):
        reducer.load_data_gate_decision(failed, manifest=_manifest())

    mismatched = diagnostics / "mismatched.json"
    _write_data_gate(
        mismatched,
        report_path=diagnostics / "mismatched-report.json",
        report_dataset_identity={"wrong": "dataset"},
    )
    with pytest.raises(reducer.TrainingAuthorizationError, match="does not bind"):
        reducer.load_data_gate_decision(mismatched, manifest=_manifest())


def test_continuation_reducer_binds_deterministic_gate_d_run_and_checkpoint(monkeypatch, tmp_path):
    root = _project(tmp_path / "memory_project")
    monkeypatch.setenv(project_paths.MEMORY_PROJECT_ROOT_ENV, str(root))
    pilot_payload = {
        "authorization_kind": runtime.PILOT_AUTHORIZATION_KIND,
        "authorized_target_completed_updates": 1_000,
        "calibration_identity": {"artifact_id": f"sha256:{'1' * 64}"},
        "config_identity": {"config_name": "v35", "experiment_name": "run", "initialization_seed": 42},
        "dataset_identity": {},
        "final_test_remains_sealed": True,
        "gate_evidence": {},
        "initialization_identity": {
            "initialization_identity_sha256": "2" * 64,
            "parameter_tree_sha256": "3" * 64,
        },
        "prepilot_source_identity": {
            "artifact_id": f"sha256:{'8' * 64}",
            "path_relative": "v35/diagnostics/prepilot_source_identity.json",
            "sha256": "9" * 64,
            "source_aggregate_sha256": "a" * 64,
        },
        "protocols": {},
        "run_identity": {"run_id_sha256": "4" * 64},
        "semantic_config_schema_version": runtime.SEMANTIC_CONFIG_SCHEMA_VERSION,
        "semantic_training_config_sha256": "5" * 64,
        "source_checkpoint": {
            "completed_updates": 0,
            "cumulative_telemetry_sha256": "b" * 64,
            "data_iterator_state_sha256": "c" * 64,
            "optimizer_state_sha256": "d" * 64,
            "parameter_tree_sha256": "3" * 64,
            "rung_artifact_id": f"sha256:{'e' * 64}",
            "rung_file_sha256": "f" * 64,
            "rung_path_relative": "v35/diagnostics/rung_0.json",
            "runtime_identity_sha256": "0" * 64,
        },
        "status": "pass",
    }
    pilot_path = root / "v35/diagnostics/pilot.json"
    runtime.write_authorization_once(pilot_path, runtime.authorization_envelope(pilot_payload))

    rung_path = root / "v35/diagnostics/rung_1000.json"
    rung_path.parent.mkdir(parents=True, exist_ok=True)
    rung_path.write_bytes(b"rung")
    rung = SimpleNamespace(
        path=rung_path,
        completed_updates=1_000,
        checkpoint_parameter_tree_sha256="6" * 64,
        artifact_id=f"sha256:{'7' * 64}",
        file_sha256=artifacts.sha256_bytes(rung_path.read_bytes()),
        run_provenance={
            "cumulative_telemetry_sha256": "8" * 64,
            "data_iterator_state_sha256": "9" * 64,
            "optimizer_state_sha256": "a" * 64,
            "run_id_sha256": "4" * 64,
            "runtime_identity_sha256": "b" * 64,
            "training_config_sha256": "5" * 64,
        },
        initialization_parameter_tree_sha256="3" * 64,
        initialization_identity=SimpleNamespace(identity_sha256="2" * 64),
        calibration_artifact_id=f"sha256:{'1' * 64}",
    )
    decision_payload = {
        "action": "extend_same_run_once_to_2500",
        "endpoint_completed_updates": 1_000,
        "outcome": "inconclusive",
        "provenance": {"prior_1k_decision_artifact_id": None},
    }
    decision = artifacts.artifact_envelope(reducer.pilot.DECISION_SCHEMA_VERSION, decision_payload)
    decision_path = root / "v35/diagnostics/gate_d_1000.json"
    artifacts.write_canonical_envelope(
        decision_path,
        decision,
        schema_version=reducer.pilot.DECISION_SCHEMA_VERSION,
    )
    monkeypatch.setattr(reducer.pilot, "load_rung_result", lambda path, manifest: rung)
    monkeypatch.setattr(reducer.pilot, "evaluate_gate_d", lambda *args, **kwargs: decision)

    result = reducer.build_continuation_authorization(
        pilot_authorization_path=pilot_path,
        gate_d_decision_path=decision_path,
        rung_paths=[rung_path],
        manifest=_manifest(),
        endpoint=1_000,
        target=2_500,
    )

    assert result["payload"]["source_checkpoint"] == {
        "completed_updates": 1_000,
        "cumulative_telemetry_sha256": "8" * 64,
        "data_iterator_state_sha256": "9" * 64,
        "optimizer_state_sha256": "a" * 64,
        "parameter_tree_sha256": "6" * 64,
        "rung_artifact_id": f"sha256:{'7' * 64}",
        "rung_file_sha256": artifacts.sha256_bytes(rung_path.read_bytes()),
        "rung_path_relative": rung_path.relative_to(root).as_posix(),
        "runtime_identity_sha256": "b" * 64,
    }
    assert result["payload"]["gate_d"]["artifact_id"] == decision["artifact_id"]
    assert result["payload"]["authorized_target_completed_updates"] == 2_500

    bad_decision = artifacts.artifact_envelope(
        reducer.pilot.DECISION_SCHEMA_VERSION,
        {**decision_payload, "outcome": "fail", "action": "stop_branch"},
    )
    monkeypatch.setattr(reducer.pilot, "evaluate_gate_d", lambda *args, **kwargs: bad_decision)
    with pytest.raises(reducer.TrainingAuthorizationError, match="does not equal deterministic reduction"):
        reducer.build_continuation_authorization(
            pilot_authorization_path=pilot_path,
            gate_d_decision_path=decision_path,
            rung_paths=[rung_path],
            manifest=_manifest(),
            endpoint=1_000,
            target=2_500,
        )


@pytest.mark.parametrize("bad_path", [pathlib.Path("../foreign.json"), pathlib.Path("/iris/u/user/foreign.json")])
def test_production_cli_paths_reject_unconfined_paths(bad_path: pathlib.Path) -> None:
    with pytest.raises(reducer.TrainingAuthorizationError, match="memory_project"):
        reducer._resolve_project_cli_path(bad_path, name="authorization path")  # noqa: SLF001
