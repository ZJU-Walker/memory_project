# ruff: noqa: SLF001

import json
import pathlib

from etils import epath
import pytest

from openpi.training import checkpoints


class _StatefulLoader:
    def __init__(self, state=None):
        self.state = state or {"schema_version": 1, "cursor": 17}
        self.restored = None

    def state_dict(self):
        return self.state

    def load_state_dict(self, state):
        self.restored = state


def test_write_provenance_assets_is_exact_byte_copy(tmp_path: pathlib.Path):
    assets_dir = epath.Path(tmp_path / "assets")
    assets_dir.mkdir()
    expected = {
        "one.json": b'{"spacing":  true}\n',
        "binary.dat": b"\x00\xff\x80\n",
    }

    checkpoints._write_provenance_assets(assets_dir, expected)

    assert {name: (tmp_path / "assets" / name).read_bytes() for name in expected} == expected


def test_step_zero_resume_is_opt_in_and_legacy_default_is_unchanged(tmp_path: pathlib.Path, monkeypatch):
    class FakeManager:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def all_steps(self):
            return (0,)

    monkeypatch.setattr(checkpoints.ocp, "CheckpointManager", FakeManager)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()

    _, legacy_resuming = checkpoints.initialize_checkpoint_dir(
        checkpoint_dir,
        keep_period=250,
        overwrite=False,
        resume=True,
    )
    _, v35_resuming = checkpoints.initialize_checkpoint_dir(
        checkpoint_dir,
        keep_period=250,
        overwrite=False,
        resume=True,
        allow_step_zero_resume=True,
    )

    assert legacy_resuming is False
    assert v35_resuming is True


def test_v35_runtime_identity_binds_iterator_hash_and_cumulative_counters(tmp_path: pathlib.Path):
    loader = _StatefulLoader()
    telemetry = {
        "schema_version": 1,
        "accepted_update_count": 250,
        "finite_accepted_update_count": 250,
        "pre_shared_severe_clip_count": 2,
        "pre_shared_update_count": 250,
        "write_feature_cap_bind_numerator": 7,
        "write_feature_cap_bind_denominator": 4_000,
        "read_feature_cap_bind_numerator": 5,
        "read_feature_cap_bind_denominator": 3_700,
        "pre_shared_grad_norm_max": 8.25,
    }
    initialization_identity = b'{"identity_sha256":"root"}\n'
    runtime_assets = checkpoints._snapshot_v35_runtime_assets(
        loader,
        telemetry,
        completed_updates=250,
        run_initialization_identity=initialization_identity,
    )
    assets_dir = tmp_path / "250" / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "v35_initialization_manifest.json").write_bytes(initialization_identity)
    for name, contents in runtime_assets.items():
        (assets_dir / name).write_bytes(contents)

    identity = json.loads(runtime_assets[checkpoints.V35_RUNTIME_IDENTITY_FILENAME])
    assert identity["completed_updates"] == 250
    assert len(identity["data_iterator_state_sha256"]) == 64
    restored_loader = _StatefulLoader(state={"fresh": True})
    restored_telemetry = checkpoints.restore_v35_runtime_state(tmp_path, 250, restored_loader)

    assert restored_loader.restored == loader.state
    assert restored_telemetry == telemetry


def test_v35_runtime_state_rejects_tampered_iterator_bytes(tmp_path: pathlib.Path):
    loader = _StatefulLoader()
    runtime_assets = checkpoints._snapshot_v35_runtime_assets(
        loader,
        {"counter": 1},
        completed_updates=1,
        run_initialization_identity=b"root identity",
    )
    assets_dir = tmp_path / "1" / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "v35_initialization_manifest.json").write_bytes(b"root identity")
    for name, contents in runtime_assets.items():
        (assets_dir / name).write_bytes(contents)
    iterator_path = assets_dir / checkpoints.V35_DATA_ITERATOR_STATE_FILENAME
    iterator = json.loads(iterator_path.read_bytes())
    iterator["cursor"] = 18
    iterator_path.write_bytes(checkpoints._canonical_json_bytes(iterator))

    with pytest.raises(ValueError, match="identity/hash binding"):
        checkpoints.restore_v35_runtime_state(tmp_path, 1, _StatefulLoader())
