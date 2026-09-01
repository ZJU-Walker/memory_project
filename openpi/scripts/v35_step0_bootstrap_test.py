# Production helpers are intentionally private; these tests exercise their fail-closed seams.
# ruff: noqa: SLF001

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys
import types

import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v35_calibration_replay as replay
    import v35_step0_bootstrap as bootstrap
finally:
    sys.path.remove(str(_SCRIPTS_DIR))

train = bootstrap.train_script


@dataclasses.dataclass(frozen=True)
class _FakeConfig:
    checkpoint_dir: Path
    name: str = bootstrap.CONFIG_NAME
    exp_name: str = "unit"
    seed: int = 42
    fsdp_devices: int = 1
    keep_period: int = 250
    model: object = dataclasses.field(default_factory=lambda: types.SimpleNamespace(memory_v35_enabled=True))
    v35_pilot_authorization_path: str | None = "v35/diagnostics/authorization/pilot.json"


class _FakeManager:
    def __init__(self, steps=()):
        self.steps = tuple(steps)
        self.waited = False

    def all_steps(self):
        return self.steps

    def latest_step(self):
        return self.steps[-1] if self.steps else None

    def wait_until_finished(self):
        self.waited = True


class _FakeParams:
    @staticmethod
    def to_pure_dict():
        return {"leaf": 1}


@dataclasses.dataclass(frozen=True)
class _FakeState:
    step: int = 0
    params: object = dataclasses.field(default_factory=_FakeParams)
    model_def: object = "calibrated-graphdef"
    tx: object = "optimizer"
    ema_decay: float | None = None


class _NeverIteratedLoader:
    def __init__(self, state):
        self._state = state

    def state_dict(self):
        return self._state

    def __iter__(self):
        raise AssertionError("step-0 finalization must not construct a data iterator")


def _patch_project(monkeypatch, root: Path) -> None:
    monkeypatch.setattr(bootstrap.project_paths, "configure_v35_runtime_environment", dict)
    monkeypatch.setattr(bootstrap.project_paths, "memory_project_root", lambda: root)
    monkeypatch.setattr(bootstrap.project_paths, "project_path", lambda value: (root / Path(value)).resolve())
    monkeypatch.setattr(
        bootstrap.project_paths,
        "project_relative_path",
        lambda value: Path(value).resolve().relative_to(root.resolve()),
    )


def test_initialize_saves_zero_update_without_constructing_loader(monkeypatch, tmp_path: Path):
    _patch_project(monkeypatch, tmp_path)
    root = tmp_path / "v35/checkpoints/pi05_yam_mem_v35/unit"
    config = _FakeConfig(checkpoint_dir=root)
    config.model.memory_injection_c = 0.8
    config.model.memory_injection_tau = 0.03
    monkeypatch.setattr(bootstrap, "_base_config", lambda **_: config)
    monkeypatch.setattr(bootstrap, "_dataset_identity", lambda _: {"manifest": "bound"})
    monkeypatch.setattr(bootstrap.authorization, "semantic_training_config_sha256", lambda _: "a" * 64)

    graft = root / "initialization_graft_manifest.json"

    def initialize_state(_config, *, graft_manifest_path):
        assert graft_manifest_path == graft
        graft_manifest_path.write_bytes(b"graft")
        return _FakeState(), replay.Step0Identity(
            actual_parameter_tree_sha256="1" * 64,
            official_base_source_sha256="2" * 64,
            target_schema_sha256="3" * 64,
            graft_manifest_sha256="4" * 64,
            raw_gate=None,
        )

    monkeypatch.setattr(bootstrap.calibration_replay, "_initialize_actual_step0_state", initialize_state)
    manager = _FakeManager()
    monkeypatch.setattr(bootstrap.checkpoints, "initialize_checkpoint_dir", lambda *args, **kwargs: (manager, False))
    saved = {}

    def save_state(_manager, state, loader, step, **kwargs):
        assert state.step == step == 0
        assert isinstance(loader, bootstrap._NoDatasetBootstrapLoader)
        assert loader.data_config().norm_stats is None
        saved.update(kwargs)
        manager.steps = (0,)

    monkeypatch.setattr(bootstrap.checkpoints, "save_state", save_state)
    output = bootstrap.initialize(experiment_name="unit", fsdp_devices=1)

    assert output == root / bootstrap.PROVISIONAL_FILENAME
    value = json.loads(output.read_bytes())
    assert value["completed_optimizer_updates"] == 0
    assert value["data_loader_constructed"] is False
    assert value["data_batches_consumed"] == 0
    assert not any(str(tmp_path) in str(item) for item in value.values())
    assert saved["provenance_assets"][bootstrap.PROVISIONAL_FILENAME] == output.read_bytes()
    assert manager.waited


def test_finalize_consumes_no_batch_and_saves_exact_initial_state(monkeypatch, tmp_path: Path):
    _patch_project(monkeypatch, tmp_path)
    root = tmp_path / "v35/checkpoints/pi05_yam_mem_v35/unit"
    root.mkdir(parents=True)
    config = _FakeConfig(checkpoint_dir=root)
    config.model.memory_injection_c = 0.8
    config.model.memory_injection_tau = 0.03
    provisional = {"actual_step0_parameter_tree_sha256": "7" * 64}
    monkeypatch.setattr(bootstrap, "_calibrated_config", lambda **_: config)
    monkeypatch.setattr(bootstrap, "_authenticate_provisional", lambda _: (provisional, b"provisional\n"))
    monkeypatch.setattr(bootstrap.train_script, "_validate_v35_training_ready", lambda _: None)
    monkeypatch.setattr(bootstrap.sharding, "make_mesh", lambda _: object())
    monkeypatch.setattr(bootstrap.jax.random, "key", lambda _: "key")
    monkeypatch.setattr(bootstrap.jax.random, "split", lambda _: ("train", "init"))
    monkeypatch.setattr(bootstrap.train_script, "init_train_state", lambda *args, **kwargs: (_FakeState(), None))
    monkeypatch.setattr(bootstrap.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(bootstrap.weight_loaders, "parameter_tree_sha256", lambda _: "7" * 64)
    monkeypatch.setattr(
        bootstrap.nnx,
        "merge",
        lambda *_: types.SimpleNamespace(memory_injection_c=0.8, memory_injection_tau=0.03),
    )
    monkeypatch.setattr(bootstrap.train_script, "_validate_v35_initialized_gate", lambda *args: None)
    identity = root / "initialization_manifest.json"

    def write_identity(*args):
        identity.write_bytes(b"identity\n")
        return identity

    monkeypatch.setattr(bootstrap.train_script, "_write_v35_initialization_identity", write_identity)
    monkeypatch.setattr(bootstrap.train_script, "_validate_v35_root_identity", lambda *args: {})
    monkeypatch.setattr(bootstrap.jax.sharding, "NamedSharding", lambda *args: object())
    monkeypatch.setattr(bootstrap.jax.sharding, "PartitionSpec", lambda *args: object())

    initial_state = {"batches_yielded": 0, "sampler": "exact-initial"}
    loaders = [_NeverIteratedLoader(initial_state), _NeverIteratedLoader(initial_state)]
    monkeypatch.setattr(bootstrap.data_loader, "create_data_loader", lambda *args, **kwargs: loaders.pop(0))
    monkeypatch.setattr(
        bootstrap.train_script,
        "_snapshot_v35_checkpoint_provenance",
        lambda cfg, path: {"v35_initialization_manifest.json": path.read_bytes()},
    )
    telemetry = {"accepted_update_count": 0}
    monkeypatch.setattr(bootstrap.train_script, "_new_v35_cumulative_telemetry", lambda: telemetry)
    monkeypatch.setattr(bootstrap.train_script, "_validate_v35_cumulative_telemetry", lambda *args, **kwargs: None)

    raw_manager = _FakeManager((0,))
    final_manager = _FakeManager()

    def manager_factory(path, **kwargs):
        return (raw_manager, True) if Path(path).name == bootstrap.RAW_CHECKPOINT_DIRNAME else (final_manager, False)

    monkeypatch.setattr(bootstrap.checkpoints, "initialize_checkpoint_dir", manager_factory)
    monkeypatch.setattr(bootstrap.checkpoints, "restore_state", lambda *args, **kwargs: _FakeState())
    saved = {}

    def save_state(_manager, state, loader, step, **kwargs):
        assert state.step == step == 0
        assert loader.state_dict() == initial_state
        saved.update(kwargs)
        final_manager.steps = (0,)

    monkeypatch.setattr(bootstrap.checkpoints, "save_state", save_state)
    monkeypatch.setattr(bootstrap.checkpoints, "restore_v35_runtime_state", lambda *args, **kwargs: telemetry)

    output = bootstrap.finalize(
        experiment_name="unit",
        calibration_relative="v35/diagnostics/calibration.json",
        fsdp_devices=1,
    )
    assert output == identity
    assert saved["v35_cumulative_telemetry"] == telemetry
    assert saved["provenance_assets"] == {"v35_initialization_manifest.json": b"identity\n"}
    assert final_manager.waited
    assert not loaders


def test_provisional_identity_tamper_and_write_once_fail_closed(tmp_path: Path):
    path = tmp_path / "identity.json"
    unsigned = {"schema": "fixture", "relative_path": "v35/checkpoints/run/0"}
    value = {**unsigned, "identity_sha256": bootstrap._canonical_payload_sha256(unsigned)}
    bootstrap._write_once(path, bootstrap._canonical_json_bytes(value))
    loaded, _ = bootstrap._load_canonical_self_hashed(path, hash_key="identity_sha256", name="fixture")
    assert loaded == value
    with pytest.raises(bootstrap.Step0BootstrapError, match="overwrite"):
        bootstrap._write_once(path, bootstrap._canonical_json_bytes(value))

    value["relative_path"] = "/machine/local/leak"
    path.write_bytes(bootstrap._canonical_json_bytes(value))
    with pytest.raises(bootstrap.Step0BootstrapError, match="self-hash"):
        bootstrap._load_canonical_self_hashed(path, hash_key="identity_sha256", name="fixture")


def test_step0_resume_omits_only_prelaunch_pilot_provenance(monkeypatch, tmp_path: Path):
    config = types.SimpleNamespace(checkpoint_dir=tmp_path)
    identity = tmp_path / "initialization_manifest.json"
    identity.write_bytes(b"identity")
    expected = {
        "v35_calibration_artifact.json": b"calibration",
        "v35_initialization_manifest.json": b"identity",
        train._V35_PILOT_AUTHORIZATION_FILENAME: b"external-pilot",
    }
    monkeypatch.setattr(train, "_snapshot_v35_checkpoint_provenance", lambda *args: dict(expected))

    step0_assets = tmp_path / "0/assets"
    step0_assets.mkdir(parents=True)
    for name, contents in expected.items():
        if name != train._V35_PILOT_AUTHORIZATION_FILENAME:
            (step0_assets / name).write_bytes(contents)
    # The validated external authorization is returned for embedding at rung 250+.
    assert train._validate_v35_resume_checkpoint_assets(
        config,
        checkpoint_step=0,
        identity_path=identity,
    ) == expected

    (step0_assets / train._V35_PILOT_AUTHORIZATION_FILENAME).write_bytes(b"another-pilot")
    with pytest.raises(ValueError, match="validated external"):
        train._validate_v35_resume_checkpoint_assets(config, checkpoint_step=0, identity_path=identity)
    (step0_assets / train._V35_PILOT_AUTHORIZATION_FILENAME).unlink()

    (step0_assets / "v35_calibration_artifact.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="not byte-identical"):
        train._validate_v35_resume_checkpoint_assets(config, checkpoint_step=0, identity_path=identity)

    step250_assets = tmp_path / "250/assets"
    step250_assets.mkdir(parents=True)
    for name, contents in expected.items():
        if name != train._V35_PILOT_AUTHORIZATION_FILENAME:
            (step250_assets / name).write_bytes(contents)
    with pytest.raises(FileNotFoundError, match="pilot"):
        train._validate_v35_resume_checkpoint_assets(config, checkpoint_step=250, identity_path=identity)


def test_final_state_requires_calibrated_static_c_tau(monkeypatch):
    config = types.SimpleNamespace(model=types.SimpleNamespace(memory_injection_c=0.8, memory_injection_tau=0.03))
    state = _FakeState()
    monkeypatch.setattr(
        bootstrap.nnx,
        "merge",
        lambda *_: types.SimpleNamespace(memory_injection_c=0.8, memory_injection_tau=0.03),
    )
    bootstrap._validate_calibrated_static_model(config, state)
    monkeypatch.setattr(
        bootstrap.nnx,
        "merge",
        lambda *_: types.SimpleNamespace(memory_injection_c=1.0, memory_injection_tau=0.02),
    )
    with pytest.raises(bootstrap.Step0BootstrapError, match="sealed calibration c/tau"):
        bootstrap._validate_calibrated_static_model(config, state)


def test_normal_train_requires_authorization_before_checkpoint_access(monkeypatch):
    config = dataclasses.replace(
        train._config.get_config(bootstrap.CONFIG_NAME),
        exp_name="authorization-order",
        resume=True,
    )
    monkeypatch.setattr(train, "init_logging", lambda: None)
    monkeypatch.setattr(train, "_configure_v35_runtime_environment", lambda _: None)
    monkeypatch.setattr(train, "_log_training_identity", lambda _: None)
    monkeypatch.setattr(train, "_validate_v35_training_ready", lambda _: None)
    monkeypatch.setattr(
        train._v35_authorization,
        "load_and_validate_pilot_authorization",
        lambda _: (_ for _ in ()).throw(RuntimeError("authorization required")),
    )
    monkeypatch.setattr(
        train._checkpoints,
        "initialize_checkpoint_dir",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("checkpoint accessed before authorization")),
    )
    with pytest.raises(RuntimeError, match="authorization required"):
        train.main(config)


def test_normal_train_rejects_fresh_v35_even_with_authorization(monkeypatch):
    config = dataclasses.replace(
        train._config.get_config(bootstrap.CONFIG_NAME),
        exp_name="fresh-is-forbidden",
        resume=False,
    )
    monkeypatch.setattr(train, "init_logging", lambda: None)
    monkeypatch.setattr(train, "_configure_v35_runtime_environment", lambda _: None)
    monkeypatch.setattr(train, "_log_training_identity", lambda _: None)
    monkeypatch.setattr(train, "_validate_v35_training_ready", lambda _: None)
    monkeypatch.setattr(train._v35_authorization, "load_and_validate_pilot_authorization", lambda _: object())
    monkeypatch.setattr(
        train._checkpoints,
        "initialize_checkpoint_dir",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fresh run touched checkpoint directory")),
    )
    with pytest.raises(ValueError, match="must --resume"):
        train.main(config)
