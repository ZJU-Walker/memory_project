from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib

import pytest

import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.project_paths as project_paths
import openpi.training.config as training_config
import openpi.training.v35_authorization as authorization


@dataclasses.dataclass(frozen=True)
class _Loader:
    params_path: str = authorization.OFFICIAL_BASE_URI


@dataclasses.dataclass(frozen=True)
class _BaseData:
    memory_episode_manifest_path: str
    memory_episode_manifest_sha256: str


@dataclasses.dataclass(frozen=True)
class _Assets:
    assets_dir: str
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class _Data:
    repo_id: str
    assets: _Assets
    base_config: _BaseData


@dataclasses.dataclass(frozen=True)
class _Memory:
    alpha_step: float = 0.01


@dataclasses.dataclass(frozen=True)
class _Model:
    memory_v35_calibration_id: str
    memory_v35_calibration_path: str
    memory: _Memory = dataclasses.field(default_factory=_Memory)
    feature_weight: float = 0.3


@dataclasses.dataclass(frozen=True)
class _Config:
    name: str
    exp_name: str
    seed: int
    model: _Model
    data: _Data
    weight_loader: _Loader = dataclasses.field(default_factory=_Loader)
    freeze_filter: nnx_utils.PathRegex = dataclasses.field(
        default_factory=lambda: nnx_utils.PathRegex(r".*/memory.*", sep="/")
    )
    num_train_steps: int = 1_000
    checkpoint_steps: tuple[int, ...] = (250, 500, 1_000)
    overwrite: bool = False
    resume: bool = False
    v35_pilot_authorization_path: str = "v35/diagnostics/authorization/pilot.json"
    v35_continuation_authorization_path: str | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        raise AssertionError("explicit test assets_dir must be used")


def _project(root: pathlib.Path) -> pathlib.Path:
    (root / "openpi/src/openpi").mkdir(parents=True)
    (root / "openpi/pyproject.toml").touch()
    return root


def _json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _artifact_id(payload: object) -> str:
    return f"sha256:{hashlib.sha256(authorization.canonical_json_bytes(payload)).hexdigest()}"


def _fixture(monkeypatch, tmp_path: pathlib.Path) -> tuple[_Config, dict, pathlib.Path]:
    root = _project(tmp_path / "memory_project")
    monkeypatch.setenv(project_paths.MEMORY_PROJECT_ROOT_ENV, str(root))
    manifest = root / "data/manifest.json"
    _json(manifest, {"schema_version": 2, "episodes": []})
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    norm_dir = root / "v35/assets/test/yam/repo"
    norm_stats = norm_dir / "norm_stats.json"
    _json(norm_stats, {"norm": "train-only"})
    storage_sha = "6" * 64
    norm_provenance = norm_dir / "norm_stats_provenance.json"
    _json(
        norm_provenance,
        {
            "train_storage": {"sha256": storage_sha},
            "computation": {"protocol": "raw-train-rows-delta-action-horizon-v1"},
        },
    )
    calibration_payload = {
        "provenance": {
            "collector_source_sha256": "1" * 64,
            "dataset_sha256": "2" * 64,
            "preflight_sha256": "3" * 64,
            "replay_protocol_sha256": "4" * 64,
        }
    }
    calibration_id = _artifact_id(calibration_payload)
    calibration = root / "v35/diagnostics/calibration.json"
    _json(calibration, {"payload": calibration_payload})
    config = _Config(
        name="pi05_yam_mem_v35",
        exp_name="portable_run",
        seed=42,
        model=_Model(
            memory_v35_calibration_id=calibration_id,
            memory_v35_calibration_path=str(calibration),
        ),
        data=_Data(
            repo_id="yam/repo",
            assets=_Assets(assets_dir=str(root / "v35/assets/test")),
            base_config=_BaseData(
                memory_episode_manifest_path=str(manifest),
                memory_episode_manifest_sha256=manifest_sha,
            ),
        ),
    )
    semantic_sha = authorization.semantic_training_config_sha256(config)
    initialization = {
        "initialization_identity_sha256": "7" * 64,
        "initialization_manifest_file_sha256": "8" * 64,
        "official_source_tree_sha256": "9" * 64,
        "official_source_uri": authorization.OFFICIAL_BASE_URI,
        "parameter_tree_sha256": "a" * 64,
    }
    run_id = authorization.run_id_sha256(
        config_name=config.name,
        experiment_name=config.exp_name,
        initialization_seed=config.seed,
        initialization_parameter_tree_sha256=initialization["parameter_tree_sha256"],
        calibration_artifact_id=calibration_id,
        semantic_config_sha256=semantic_sha,
    )
    evidence = {}
    criteria = {"gate_a": "gate-a-v1", "gate_b": "gate-b-v1", "step0": "step0-v1"}
    for name, criteria_version in criteria.items():
        artifact_payload = (
            {
                "checkpoint": {
                    "completed_updates": 0,
                    "parameter_tree_sha256": initialization["parameter_tree_sha256"],
                },
                "run_provenance": {
                    "cumulative_telemetry_sha256": "b" * 64,
                    "data_iterator_state_sha256": "c" * 64,
                    "optimizer_state_sha256": "d" * 64,
                    "runtime_identity_sha256": "e" * 64,
                },
            }
            if name == "step0"
            else {"name": name}
        )
        artifact_path = root / f"v35/diagnostics/{name}.json"
        artifact = {"artifact_id": _artifact_id(artifact_payload), "payload": artifact_payload}
        _json(artifact_path, artifact)
        evidence[name] = {
            "artifact_id": artifact["artifact_id"],
            "criteria_version": criteria_version,
            "path_relative": artifact_path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        }
    prepilot_path = root / "v35/diagnostics/prepilot_source_identity.json"
    _json(prepilot_path, authorization.frozen_source_identity_envelope())
    prepilot = authorization.load_and_validate_frozen_source_identity(prepilot_path)
    source_checkpoint = {
        "completed_updates": 0,
        "cumulative_telemetry_sha256": "b" * 64,
        "data_iterator_state_sha256": "c" * 64,
        "optimizer_state_sha256": "d" * 64,
        "parameter_tree_sha256": initialization["parameter_tree_sha256"],
        "rung_artifact_id": evidence["step0"]["artifact_id"],
        "rung_file_sha256": evidence["step0"]["sha256"],
        "rung_path_relative": evidence["step0"]["path_relative"],
        "runtime_identity_sha256": "e" * 64,
    }
    payload = {
        "authorization_kind": authorization.PILOT_AUTHORIZATION_KIND,
        "authorized_target_completed_updates": 1_000,
        "calibration_identity": {
            "artifact_id": calibration_id,
            "collector_source_sha256": "1" * 64,
            "dataset_protocol_sha256": "2" * 64,
            "file_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
            "preflight_sha256": "3" * 64,
            "replay_protocol_sha256": "4" * 64,
        },
        "config_identity": {
            "config_name": config.name,
            "experiment_name": config.exp_name,
            "initialization_seed": config.seed,
        },
        "dataset_identity": {
            "episode_manifest_sha256": manifest_sha,
            "norm_computation_protocol": "raw-train-rows-delta-action-horizon-v1",
            "norm_stats_provenance_sha256": hashlib.sha256(norm_provenance.read_bytes()).hexdigest(),
            "norm_stats_sha256": hashlib.sha256(norm_stats.read_bytes()).hexdigest(),
            "split_assignment_sha256": "5" * 64,
            "train_storage_sha256": storage_sha,
        },
        "final_test_remains_sealed": True,
        "gate_evidence": evidence,
        "initialization_identity": initialization,
        "prepilot_source_identity": {
            "artifact_id": prepilot.artifact_id,
            "path_relative": prepilot_path.relative_to(root).as_posix(),
            "sha256": prepilot.file_sha256,
            "source_aggregate_sha256": prepilot.source_identity["aggregate_sha256"],
        },
        "protocols": {
            "calibration_schema_version": "openpi.v35.injection-calibration.v1",
            "checkpoint_labels": "completed_optimizer_updates",
            "gate_a_criteria_version": criteria["gate_a"],
            "gate_b_criteria_version": criteria["gate_b"],
            "step0_criteria_version": criteria["step0"],
        },
        "run_identity": {"run_id_sha256": run_id},
        "semantic_config_schema_version": authorization.SEMANTIC_CONFIG_SCHEMA_VERSION,
        "semantic_training_config_sha256": semantic_sha,
        "source_checkpoint": source_checkpoint,
        "status": "pass",
    }
    pilot_path = project_paths.project_path(config.v35_pilot_authorization_path)
    authorization.write_authorization_once(pilot_path, authorization.authorization_envelope(payload))
    return config, payload, pilot_path


def test_semantic_hash_excludes_only_target_auth_controls_and_is_relocation_stable(monkeypatch, tmp_path):
    config, _, _ = _fixture(monkeypatch, tmp_path / "first")
    first = authorization.semantic_training_config_sha256(config)
    changed_target = dataclasses.replace(
        config,
        num_train_steps=10_000,
        checkpoint_steps=(250, 500, 1_000, 2_500, 5_000, 10_000),
        resume=True,
        v35_continuation_authorization_path="v35/diagnostics/authorization/full.json",
    )
    assert authorization.semantic_training_config_sha256(changed_target) == first
    assert (
        authorization.semantic_training_config_sha256(
            dataclasses.replace(config, model=dataclasses.replace(config.model, feature_weight=0.31))
        )
        != first
    )

    second_config, _, _ = _fixture(monkeypatch, tmp_path / "second")
    assert authorization.semantic_training_config_sha256(second_config) == first


def test_registered_v35_semantic_hash_handles_exact_tyro_missing_and_path_regex():
    config = training_config.get_config("pi05_yam_mem_v35")
    payload = authorization.semantic_training_config_payload(config)

    assert payload["fields"]["exp_name"] == {"__tyro_missing__": "propagating"}
    assert payload["fields"]["freeze_filter"]["__path_regex__"] == config.freeze_filter.pattern.pattern
    assert payload["source_identity"]["protocol"] == "openpi.v35.training-and-gate-source-tree.v2"
    assert "openpi/src/openpi/training/optimizer.py" in payload["source_identity"]["files"]
    assert "openpi/src/openpi/models/gemma.py" in payload["source_identity"]["files"]
    assert "openpi/uv.lock" in payload["source_identity"]["files"]
    assert payload["source_identity"]["runtime"]["distributions"]["jax"]
    assert len(authorization.semantic_training_config_sha256(config)) == 64


def test_semantic_hash_changes_when_current_source_identity_changes(monkeypatch):
    config = training_config.get_config("pi05_yam_mem_v35")
    before = authorization.semantic_training_config_sha256(config)
    source = authorization.semantic_source_identity()
    monkeypatch.setattr(
        authorization,
        "semantic_source_identity",
        lambda: {**source, "aggregate_sha256": "f" * 64},
    )

    assert authorization.semantic_training_config_sha256(config) != before


def test_pilot_authorization_is_fail_closed_and_binds_initialized_tree(monkeypatch, tmp_path):
    config, payload, _ = _fixture(monkeypatch, tmp_path)
    record = authorization.load_and_validate_pilot_authorization(config)
    identity = {
        "actual_step0_parameter_tree_sha256": "a" * 64,
        "calibration_id": config.model.memory_v35_calibration_id,
        "config_name": config.name,
        "experiment_name": config.exp_name,
        "identity_sha256": "7" * 64,
        "official_source_uri": authorization.OFFICIAL_BASE_URI,
        "run_id_sha256": payload["run_identity"]["run_id_sha256"],
        "semantic_training_config_sha256": payload["semantic_training_config_sha256"],
        "source_tree_sha256": "9" * 64,
        "step0_checkpoint": 0,
    }
    authorization.validate_pilot_run_binding(
        config,
        record,
        initialization_identity=identity,
        actual_parameter_tree_sha256="a" * 64,
    )
    with pytest.raises(authorization.V35AuthorizationError, match="step-0 parameter tree"):
        authorization.validate_pilot_run_binding(
            config,
            record,
            initialization_identity=identity,
            actual_parameter_tree_sha256="b" * 64,
        )

    gate_b = project_paths.project_path(payload["gate_evidence"]["gate_b"]["path_relative"])
    gate_b.write_bytes(gate_b.read_bytes() + b"\n")
    with pytest.raises(authorization.V35AuthorizationError, match="file SHA256 changed"):
        authorization.load_and_validate_pilot_authorization(config)


@pytest.mark.parametrize(
    ("tampered_field", "expected"),
    [
        ("parameter_tree_sha256", "sealed rung"),
        ("optimizer_state_sha256", "sealed rung"),
        ("runtime_identity_sha256", "sealed rung"),
        ("cumulative_telemetry_sha256", "sealed rung"),
        ("data_iterator_state_sha256", "sealed rung"),
    ],
)
def test_live_source_checkpoint_rejects_params_train_state_and_runtime_tamper(
    monkeypatch,
    tmp_path,
    tampered_field: str,
    expected: str,
):
    config, payload, _ = _fixture(monkeypatch, tmp_path)
    record = authorization.load_and_validate_pilot_authorization(config)
    source = payload["source_checkpoint"]
    actual = {
        "completed_updates": 0,
        "parameter_tree_sha256": source["parameter_tree_sha256"],
        "optimizer_state_sha256": source["optimizer_state_sha256"],
        "runtime_identity_sha256": source["runtime_identity_sha256"],
        "cumulative_telemetry_sha256": source["cumulative_telemetry_sha256"],
        "data_iterator_state_sha256": source["data_iterator_state_sha256"],
    }
    authorization.validate_live_source_checkpoint_binding(record, **actual)

    actual[tampered_field] = "f" * 64
    with pytest.raises(authorization.V35AuthorizationError, match=expected):
        authorization.validate_live_source_checkpoint_binding(record, **actual)


def test_live_source_checkpoint_rejects_unsealed_intermediate_resume(monkeypatch, tmp_path):
    config, payload, _ = _fixture(monkeypatch, tmp_path)
    record = authorization.load_and_validate_pilot_authorization(config)
    source = payload["source_checkpoint"]

    with pytest.raises(authorization.V35AuthorizationError, match="authorized source"):
        authorization.validate_live_source_checkpoint_binding(
            record,
            completed_updates=250,
            parameter_tree_sha256=source["parameter_tree_sha256"],
            optimizer_state_sha256=source["optimizer_state_sha256"],
            runtime_identity_sha256=source["runtime_identity_sha256"],
            cumulative_telemetry_sha256=source["cumulative_telemetry_sha256"],
            data_iterator_state_sha256=source["data_iterator_state_sha256"],
        )


def test_pilot_launch_rejects_source_runtime_drift_after_prepilot_freeze(monkeypatch, tmp_path):
    config, _, _ = _fixture(monkeypatch, tmp_path)
    frozen = authorization.semantic_source_identity()
    monkeypatch.setattr(
        authorization,
        "semantic_source_identity",
        lambda: {**frozen, "aggregate_sha256": "f" * 64},
    )

    with pytest.raises(authorization.V35AuthorizationError, match="source/runtime identity changed"):
        authorization.load_and_validate_pilot_authorization(config)


def test_continuation_requires_exact_gate_d_exit_and_checkpoint_binding(monkeypatch, tmp_path):
    config, pilot_payload, pilot_path = _fixture(monkeypatch, tmp_path)
    pilot_record = authorization.load_and_validate_pilot_authorization(config)
    root = project_paths.memory_project_root()
    rung_payload = {
        "checkpoint": {"completed_updates": 1_000, "parameter_tree_sha256": "c" * 64},
        "run_provenance": {
            "cumulative_telemetry_sha256": "1" * 64,
            "data_iterator_state_sha256": "2" * 64,
            "optimizer_state_sha256": "3" * 64,
            "runtime_identity_sha256": "4" * 64,
        },
    }
    rung = {"artifact_id": _artifact_id(rung_payload), "payload": rung_payload}
    rung_path = root / "v35/diagnostics/rung_1000.json"
    _json(rung_path, rung)
    rung_file_sha256 = hashlib.sha256(rung_path.read_bytes()).hexdigest()
    decision_payload = {
        "outcome": "inconclusive",
        "provenance": {
            "rung_artifacts": [
                {
                    "artifact_id": rung["artifact_id"],
                    "checkpoint_parameter_tree_sha256": "c" * 64,
                    "completed_updates": 1_000,
                    "file_sha256": rung_file_sha256,
                }
            ]
        },
    }
    decision = {"artifact_id": _artifact_id(decision_payload), "payload": decision_payload}
    decision_path = root / "v35/diagnostics/gate_d_1000.json"
    _json(decision_path, decision)
    continuation_payload = {
        "authorization_kind": authorization.CONTINUATION_AUTHORIZATION_KIND,
        "authorized_target_completed_updates": 2_500,
        "calibration_identity": pilot_payload["calibration_identity"],
        "config_identity": pilot_payload["config_identity"],
        "final_test_remains_sealed": True,
        "gate_d": {
            "action": "extend_same_run_once_to_2500",
            "artifact_id": decision["artifact_id"],
            "criteria_version": "openpi.v35.gate-d.rev5.v1",
            "endpoint_completed_updates": 1_000,
            "file_sha256": hashlib.sha256(decision_path.read_bytes()).hexdigest(),
            "outcome": "inconclusive",
            "path_relative": decision_path.relative_to(root).as_posix(),
        },
        "initialization_identity": pilot_payload["initialization_identity"],
        "pilot_authorization": {
            "artifact_id": pilot_record.artifact_id,
            "path_relative": pilot_path.relative_to(root).as_posix(),
            "sha256": pilot_record.file_sha256,
        },
        "prior_continuation_authorization": None,
        "run_identity": pilot_payload["run_identity"],
        "semantic_config_schema_version": authorization.SEMANTIC_CONFIG_SCHEMA_VERSION,
        "semantic_training_config_sha256": pilot_payload["semantic_training_config_sha256"],
        "source_checkpoint": {
            "completed_updates": 1_000,
            "cumulative_telemetry_sha256": "1" * 64,
            "data_iterator_state_sha256": "2" * 64,
            "optimizer_state_sha256": "3" * 64,
            "parameter_tree_sha256": "c" * 64,
            "rung_artifact_id": rung["artifact_id"],
            "rung_file_sha256": rung_file_sha256,
            "rung_path_relative": rung_path.relative_to(root).as_posix(),
            "runtime_identity_sha256": "4" * 64,
        },
        "status": "pass",
    }
    continuation_path = root / "v35/diagnostics/authorization/extend_2500.json"
    authorization.write_authorization_once(
        continuation_path,
        authorization.authorization_envelope(continuation_payload),
    )
    extension = dataclasses.replace(
        config,
        resume=True,
        num_train_steps=2_500,
        checkpoint_steps=(250, 500, 1_000, 2_500),
        v35_continuation_authorization_path=continuation_path.relative_to(root).as_posix(),
    )
    record = authorization.load_and_validate_continuation_authorization(
        extension,
        pilot_authorization=pilot_record,
        latest_checkpoint_step=1_000,
    )
    authorization.validate_continuation_checkpoint_binding(
        record,
        latest_checkpoint_step=1_000,
        actual_parameter_tree_sha256="c" * 64,
        embedded_authorization_bytes=None,
    )
    with pytest.raises(authorization.V35AuthorizationError, match="parameter hash"):
        authorization.validate_continuation_checkpoint_binding(
            record,
            latest_checkpoint_step=1_000,
            actual_parameter_tree_sha256="f" * 64,
            embedded_authorization_bytes=None,
        )
    with pytest.raises(authorization.V35AuthorizationError, match="source rung"):
        authorization.validate_continuation_checkpoint_binding(
            record,
            latest_checkpoint_step=5_000,
            actual_parameter_tree_sha256="f" * 64,
            embedded_authorization_bytes=None,
        )
    with pytest.raises(authorization.V35AuthorizationError, match="source rung"):
        authorization.validate_continuation_checkpoint_binding(
            record,
            latest_checkpoint_step=5_000,
            actual_parameter_tree_sha256="f" * 64,
            embedded_authorization_bytes=continuation_path.read_bytes(),
        )
