# ruff: noqa: SLF001
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import v35_train


@dataclasses.dataclass(frozen=True)
class _Config:
    resume: bool = False
    overwrite: bool = False
    num_train_steps: int = 1_000
    checkpoint_steps: tuple[int, ...] = (250, 500, 1_000)
    v35_pilot_authorization_path: str | None = None
    v35_continuation_authorization_path: str | None = None


def _patch_project(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(v35_train.project_paths, "project_path", lambda value: root / Path(value))
    monkeypatch.setattr(
        v35_train.bootstrap,
        "_calibrated_config",
        lambda **_: _Config(),
    )


def _artifacts(root: Path) -> None:
    for relative in (
        "v35/diagnostics/calibration.json",
        "v35/diagnostics/authorization/pilot.json",
        "v35/diagnostics/authorization/continuation.json",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")


def test_pilot_builds_only_the_frozen_authorized_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_project(monkeypatch, tmp_path)
    _artifacts(tmp_path)

    config = v35_train.build_config(
        experiment_name="pilot",
        calibration="v35/diagnostics/calibration.json",
        pilot_authorization="v35/diagnostics/authorization/pilot.json",
        target=1_000,
        fsdp_devices=4,
    )

    assert config.resume is True
    assert config.overwrite is False
    assert config.num_train_steps == 1_000
    assert config.checkpoint_steps == (250, 500, 1_000)
    assert config.v35_pilot_authorization_path == "v35/diagnostics/authorization/pilot.json"
    assert config.v35_continuation_authorization_path is None


@pytest.mark.parametrize(
    ("target", "steps"),
    [
        (2_500, (250, 500, 1_000, 2_500)),
        (10_000, (250, 500, 1_000, 2_500, 5_000, 10_000)),
    ],
)
def test_continuation_requires_and_installs_its_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: int,
    steps: tuple[int, ...],
) -> None:
    _patch_project(monkeypatch, tmp_path)
    _artifacts(tmp_path)

    config = v35_train.build_config(
        experiment_name="pilot",
        calibration="v35/diagnostics/calibration.json",
        pilot_authorization="v35/diagnostics/authorization/pilot.json",
        continuation_authorization="v35/diagnostics/authorization/continuation.json",
        target=target,
    )

    assert config.resume is True
    assert config.num_train_steps == target
    assert config.checkpoint_steps == steps
    assert config.v35_continuation_authorization_path == "v35/diagnostics/authorization/continuation.json"


def test_launch_rejects_missing_or_out_of_project_authorizations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_project(monkeypatch, tmp_path)
    _artifacts(tmp_path)

    with pytest.raises(v35_train.V35TrainLaunchError, match="must not use"):
        v35_train.build_config(
            experiment_name="pilot",
            calibration="v35/diagnostics/calibration.json",
            pilot_authorization="v35/diagnostics/authorization/pilot.json",
            continuation_authorization="v35/diagnostics/authorization/continuation.json",
            target=1_000,
        )
    with pytest.raises(v35_train.V35TrainLaunchError, match="requires"):
        v35_train.build_config(
            experiment_name="pilot",
            calibration="v35/diagnostics/calibration.json",
            pilot_authorization="v35/diagnostics/authorization/pilot.json",
            target=2_500,
        )
    with pytest.raises(v35_train.V35TrainLaunchError, match="relative"):
        v35_train.build_config(
            experiment_name="pilot",
            calibration="/outside/calibration.json",
            pilot_authorization="v35/diagnostics/authorization/pilot.json",
        )


def test_main_delegates_only_the_built_resume_config(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = _Config(resume=True)
    monkeypatch.setattr(v35_train, "build_config", lambda **_: expected)
    received = SimpleNamespace(config=None)
    authenticated = SimpleNamespace(config=None)
    monkeypatch.setattr(
        v35_train,
        "reauthenticate_pilot_evidence",
        lambda config: setattr(authenticated, "config", config),
    )
    monkeypatch.setattr(v35_train.train_script, "main", lambda config: setattr(received, "config", config))

    assert (
        v35_train.main(
            [
                "--experiment-name",
                "pilot",
                "--calibration",
                "v35/diagnostics/calibration.json",
            ]
        )
        == 0
    )
    assert authenticated.config is expected
    assert received.config is expected


def test_print_semantic_hash_does_not_require_pilot_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _patch_project(monkeypatch, tmp_path)
    _artifacts(tmp_path)
    expected = "a" * 64
    monkeypatch.setattr(v35_train.authorization, "semantic_training_config_sha256", lambda _: expected)
    monkeypatch.setattr(
        v35_train,
        "build_config",
        lambda **_: pytest.fail("printing the preauthorization identity must not build an authorized resume"),
    )

    assert (
        v35_train.main(
            [
                "--experiment-name",
                "pilot",
                "--calibration",
                "v35/diagnostics/calibration.json",
                "--print-semantic-config-sha256",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == expected + "\n"


def test_verify_only_authenticates_without_calling_training(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected = _Config(resume=True)
    monkeypatch.setattr(v35_train, "build_config", lambda **_: expected)
    received = SimpleNamespace(config=None)
    monkeypatch.setattr(
        v35_train,
        "verify_authorized_checkpoint0",
        lambda config: setattr(received, "config", config),
    )
    monkeypatch.setattr(v35_train.train_script, "main", lambda _: pytest.fail("verify-only must not train"))

    assert (
        v35_train.main(
            [
                "--experiment-name",
                "pilot",
                "--calibration",
                "v35/diagnostics/calibration.json",
                "--verify-only",
            ]
        )
        == 0
    )
    assert received.config is expected
    assert "verified" in capsys.readouterr().out


def _live_checkpoint_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train = v35_train.train_script
    step_dir = tmp_path / "0"
    assets = step_dir / "assets"
    state_dir = step_dir / "train_state"
    assets.mkdir(parents=True)
    state_dir.mkdir()
    runtime_files = {
        train._checkpoints.V35_RUNTIME_IDENTITY_FILENAME: b"runtime\n",
        train._checkpoints.V35_CUMULATIVE_TELEMETRY_FILENAME: b"telemetry\n",
        train._checkpoints.V35_DATA_ITERATOR_STATE_FILENAME: b"iterator\n",
    }
    for name, contents in runtime_files.items():
        (assets / name).write_bytes(contents)
    (state_dir / "state").write_bytes(b"optimizer")
    parameter_sha = "a" * 64
    rung_payload = {
        "checkpoint": {"completed_updates": 0, "parameter_tree_sha256": parameter_sha},
        "run_provenance": {
            "cumulative_telemetry_sha256": hashlib.sha256(
                runtime_files[train._checkpoints.V35_CUMULATIVE_TELEMETRY_FILENAME]
            ).hexdigest(),
            "data_iterator_state_sha256": hashlib.sha256(
                runtime_files[train._checkpoints.V35_DATA_ITERATOR_STATE_FILENAME]
            ).hexdigest(),
            "optimizer_state_sha256": train._checkpoints.v35_checkpoint_component_tree_sha256(state_dir),
            "runtime_identity_sha256": hashlib.sha256(
                runtime_files[train._checkpoints.V35_RUNTIME_IDENTITY_FILENAME]
            ).hexdigest(),
        },
    }
    rung_id = f"sha256:{hashlib.sha256(train._v35_authorization.canonical_json_bytes(rung_payload)).hexdigest()}"
    rung = {"artifact_id": rung_id, "payload": rung_payload}
    rung_path = tmp_path / "rung0.json"
    rung_path.write_bytes(train._v35_authorization.canonical_json_bytes(rung) + b"\n")
    rung_file_sha = hashlib.sha256(rung_path.read_bytes()).hexdigest()
    source = {
        "completed_updates": 0,
        **rung_payload["run_provenance"],
        "parameter_tree_sha256": parameter_sha,
        "rung_artifact_id": rung_id,
        "rung_file_sha256": rung_file_sha,
        "rung_path_relative": rung_path.name,
    }
    authorization = train._v35_authorization.AuthorizationRecord(
        path=tmp_path / "pilot.json",
        file_sha256="b" * 64,
        artifact_id=f"sha256:{'c' * 64}",
        payload={
            "authorization_kind": train._v35_authorization.PILOT_AUTHORIZATION_KIND,
            "gate_evidence": {
                "step0": {
                    "artifact_id": rung_id,
                    "criteria_version": "fixture",
                    "path_relative": rung_path.name,
                    "sha256": rung_file_sha,
                }
            },
            "source_checkpoint": source,
        },
    )
    telemetry = {"fixture": 0}
    params = SimpleNamespace(to_pure_dict=lambda: {"parameter": 1})
    restored_state = SimpleNamespace(step=0, params=params)
    monkeypatch.setattr(train.project_paths, "project_path", lambda value: tmp_path / Path(value))
    monkeypatch.setattr(train._checkpoints, "restore_state", lambda *args, **kwargs: restored_state)
    monkeypatch.setattr(train._checkpoints, "restore_v35_runtime_state", lambda *args, **kwargs: telemetry)
    monkeypatch.setattr(train, "_validate_v35_cumulative_telemetry", lambda *args, **kwargs: None)
    monkeypatch.setattr(train, "_validate_v35_initialized_gate", lambda *args, **kwargs: None)
    monkeypatch.setattr(train.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(train._weight_loaders, "parameter_tree_sha256", lambda _: parameter_sha)
    config = SimpleNamespace(checkpoint_dir=tmp_path)
    return train, config, authorization, restored_state, telemetry, state_dir, assets


def test_shared_live_checkpoint_validator_restores_and_accepts_exact_checkpoint0(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, config, authorization, restored_state, telemetry, _, _ = _live_checkpoint_fixture(tmp_path, monkeypatch)
    state, restored_telemetry, parameter_sha = train._restore_and_validate_v35_authorized_source_checkpoint(
        config,
        checkpoint_manager=object(),
        checkpoint_step=0,
        state_shape=object(),
        data_loader=object(),
        source_authorization=authorization,
    )
    assert state is restored_state
    assert restored_telemetry is telemetry
    assert parameter_sha == "a" * 64


@pytest.mark.parametrize("tamper", ["params", "train_state", "runtime"])
def test_shared_live_checkpoint_validator_rejects_live_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    train, config, authorization, _, _, state_dir, assets = _live_checkpoint_fixture(tmp_path, monkeypatch)
    if tamper == "params":
        monkeypatch.setattr(train._weight_loaders, "parameter_tree_sha256", lambda _: "f" * 64)
    elif tamper == "train_state":
        (state_dir / "state").write_bytes(b"tampered optimizer")
    else:
        (assets / train._checkpoints.V35_RUNTIME_IDENTITY_FILENAME).write_bytes(b"tampered runtime\n")

    with pytest.raises(train._v35_authorization.V35AuthorizationError, match="sealed rung"):
        train._restore_and_validate_v35_authorized_source_checkpoint(
            config,
            checkpoint_manager=object(),
            checkpoint_step=0,
            state_shape=object(),
            data_loader=object(),
            source_authorization=authorization,
        )
