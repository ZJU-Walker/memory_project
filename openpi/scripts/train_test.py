# ruff: noqa: I001, SLF001

import dataclasses
import hashlib
import json
import os
import pathlib
import types

# The backend must be selected before importing Flax/JAX; setting it afterwards is not an
# effective CPU interlock and can make this test module unexpectedly claim a training GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from . import train
from openpi.models import model as _model
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import utils as training_utils
from openpi.training import weight_loaders as _weight_loaders


def test_v35_bootstrap_zero_creates_wandb_identity_then_resumes_it(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoints" / "run"
    checkpoint_dir.mkdir(parents=True)
    config = types.SimpleNamespace(
        checkpoint_dir=checkpoint_dir,
        exp_name="run",
        project_name="openpi",
    )
    calls = []
    fake_wandb = types.SimpleNamespace(
        run=types.SimpleNamespace(id="fresh-run-id", log_code=lambda *_: None),
        init=lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(train, "wandb", fake_wandb)
    monkeypatch.setattr(train.dataclasses, "asdict", lambda _: {})

    train.init_wandb(config, resuming=True, allow_new_run_from_bootstrap_zero=True)
    assert (checkpoint_dir / "wandb_id.txt").read_text() == "fresh-run-id"
    assert calls[0]["name"] == "run"

    train.init_wandb(config, resuming=True)
    assert calls[1] == {"id": "fresh-run-id", "resume": "must", "project": "openpi"}


def _calibrated_v35_config(
    monkeypatch,
    tmp_path: pathlib.Path,
    *,
    source_sha256: str = "0" * 64,
    include_videos: bool = True,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    project_root = tmp_path / "memory_project"
    (project_root / "openpi/src/openpi").mkdir(parents=True)
    (project_root / "openpi/pyproject.toml").touch()
    monkeypatch.setenv("MEMORY_PROJECT_ROOT", str(project_root))
    config = _config.get_config("pi05_yam_mem_v35")
    stable_ids = [f"train-{index:03d}" for index in range(54)]
    manifest = {
        "schema_version": 1,
        "split_seed": 36,
        "episodes": [
            {"stable_id": stable_id, "split": "train", "include": True, "episode_index": index}
            for index, stable_id in enumerate(stable_ids)
        ],
    }
    manifest_path = project_root / "data/episode_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    membership = [{"split": "train", "stable_id": stable_id} for stable_id in stable_ids]
    membership_sha256 = hashlib.sha256(train._calibration_canonical_json(membership).encode()).hexdigest()
    dataset_protocol_sha256 = "3" * 64
    raw_gate = np.full((4,), np.arctanh(np.float32(0.5)), dtype=np.float32)
    payload = {
        "schema_version": "openpi.v35.injection-calibration.v1",
        "status": "pass",
        "provenance": {
            "source_sha256": source_sha256,
            "dataset_sha256": dataset_protocol_sha256,
            "split_sha256": manifest_sha256,
            "observed_membership_sha256": membership_sha256,
        },
        "population": {"split": "train", "episode_count": 54, "stable_ids": stable_ids},
        "parameters": {"alpha_step": 0.01, "memory_injection_c": 3.0, "memory_injection_tau": 0.04},
        "gate": {
            "target_effective_tanh_gate": 0.5,
            "open_channel_count": config.model.memory.d_value,
            "raw_w_sha256": hashlib.sha256(raw_gate.tobytes(order="C")).hexdigest(),
        },
        "gates": {"passes": True, "all_episodes_train": True, "fixed_effective_gate_is_0_5": True},
    }
    digest = hashlib.sha256(train._calibration_canonical_json(payload).encode()).hexdigest()
    artifact = {"artifact_sha256": digest, "calibration_id": f"sha256:{digest}", "payload": payload}
    calibration_path = project_root / "v35/diagnostics/calibration.json"
    calibration_path.parent.mkdir(parents=True)
    calibration_path.write_text(json.dumps(artifact), encoding="utf-8")
    model = dataclasses.replace(
        config.model,
        memory_v35_calibrated=True,
        memory_v35_calibration_id=f"sha256:{digest}",
        memory_v35_calibration_path=str(calibration_path),
        memory_injection_c=3.0,
        memory_injection_tau=0.04,
    )
    base_data = dataclasses.replace(
        config.data.base_config,
        lerobot_dataset_root=str(project_root / "data/lerobot" / config.data.repo_id),
        memory_episode_manifest_path=str(manifest_path),
        memory_episode_manifest_sha256=manifest_sha256,
    )
    assets_dir = project_root / "v35/assets/pi05_yam_0830_0831_v36"
    norm_dir = assets_dir / config.data.repo_id
    norm_dir.mkdir(parents=True)
    norm_payload = "{}\n"
    (norm_dir / "norm_stats.json").write_text(norm_payload, encoding="utf-8")
    frame_counts = [2] * 54
    selected_frames = sum(frame_counts)
    dataset_root = (project_root / "data/lerobot" / config.data.repo_id).resolve()
    data_dir = dataset_root / "data" / "chunk-000"
    video_dir = dataset_root / "videos" / "chunk-000" / "observation.images.top"
    meta_dir = dataset_root / "meta"
    data_dir.mkdir(parents=True)
    if include_videos:
        video_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    (meta_dir / "info.json").write_bytes(b'{"fixture":true}\n')
    for index in range(54):
        stem = f"episode_{index:06d}"
        (data_dir / f"{stem}.parquet").write_bytes(f"data-{index}".encode())
        if include_videos:
            (video_dir / f"{stem}.mp4").write_bytes(f"video-{index}".encode())
    storage_paths = sorted(
        [*data_dir.rglob("*"), *(video_dir.rglob("*") if include_videos else ()), *meta_dir.rglob("*")],
        key=lambda path: path.relative_to(dataset_root).as_posix(),
    )
    storage_records = [
        {
            "path": path.relative_to(dataset_root).as_posix(),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in storage_paths
        if path.is_file()
    ]
    storage_sha256 = hashlib.sha256(train._canonical_json(storage_records).encode()).hexdigest()
    provenance = {
        "schema_version": 2,
        "status": "complete",
        "repo_id": config.data.repo_id,
        "manifest": {
            "path_relative": manifest_path.relative_to(project_root).as_posix(),
            "sha256": manifest_sha256,
            "active_split": "train",
            "split_seed": 36,
        },
        "selection": {
            "dataset_num_episodes": 54,
            "selected_num_episodes": 54,
            "selected_episode_indices": list(range(54)),
            "selected_stable_ids": stable_ids,
            "selected_episode_frame_counts": frame_counts,
            "selected_num_frames": selected_frames,
            "dataset_episode_frame_protocol_sha256": dataset_protocol_sha256,
        },
        "train_storage": {
            "root_contract": "memory_project-relative-v1",
            "root_relative": dataset_root.relative_to(project_root).as_posix(),
            "sha256": storage_sha256,
            "files": storage_records,
            "selected_episode_indices": list(range(54)),
            "scope": "selected train episode parquet, optional videos, plus structural meta files",
        },
        "computation": {
            "protocol": "raw-train-rows-delta-action-horizon-v1",
            "requested_batch_size": config.batch_size,
            "num_batches_including_partial_final_batch": (selected_frames + config.batch_size - 1) // config.batch_size,
            "processed_base_rows": selected_frames,
            "drop_last_rows": 0,
        },
        "norm_stats": {"file": "norm_stats.json", "sha256": hashlib.sha256(norm_payload.encode()).hexdigest()},
    }
    (norm_dir / "norm_stats_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    data = dataclasses.replace(
        config.data,
        base_config=base_data,
        assets=dataclasses.replace(config.data.assets, assets_dir=str(assets_dir)),
    )
    # These fixture-level unit tests exercise calibration/storage/identity helpers directly;
    # main() separately requires the production pilot authorization path.
    return dataclasses.replace(
        config,
        model=model,
        data=data,
        assets_base_dir=str(project_root / "v35/assets"),
        checkpoint_base_dir=str(project_root / "v35/checkpoints"),
        v35_pilot_authorization_path=None,
    )


def test_training_identity_log_includes_config_and_effective_eta_scale(caplog):
    config = dataclasses.replace(
        _config.get_config("pi05_yam_mem_v34_run5_eta0"),
        exp_name="v34_run5_eta0_smoke",
    )
    with caplog.at_level("INFO"):
        train._log_training_identity(config)
    assert "name=pi05_yam_mem_v34_run5_eta0" in caplog.text
    assert "exp_name=v34_run5_eta0_smoke" in caplog.text
    assert "eta_scale=0.0" in caplog.text


def test_console_metric_filter_keeps_aux_phase_but_hides_only_position_suffixes():
    assert not train._is_per_position_metric("diagnostic/aux_phase_accuracy")
    assert not train._is_per_position_metric("diagnostic/some_prefix_metric")
    assert train._is_per_position_metric("diagnostic/probe_accuracy_by_step_p0")
    assert train._is_per_position_metric("diagnostic/probe_accuracy_by_step_p39")


def test_v35_cell_macro_is_episode_first_and_equal_over_present_cells():
    # Cell means are 1 and 3; the absent third cell is excluded rather than counted as zero.
    value = train._v35_cell_macro_ce(
        jnp.asarray([2.0, 9.0, 0.0]),
        jnp.asarray([2.0, 3.0, 0.0]),
    )
    assert float(value) == pytest.approx(2.0)


def _valid_v35_runtime_guard_fixture():
    config = _config.get_config("pi05_yam_mem_v35")
    observation = types.SimpleNamespace(
        seq_step_mask=jnp.asarray([[True, True, True]]),
        seq_write_mask=jnp.asarray([[True, False, False]]),
        seq_decision_mask=jnp.asarray([[False, False, True]]),
        seq_read_state_valid=jnp.asarray([[False, True, True]]),
        seq_read_credit_reachable=jnp.asarray([[False, True, True]]),
        seq_decay_gap_before=jnp.asarray([[0, 0, 2]], dtype=jnp.int32),
        seq_use_pressure_mask=jnp.asarray([[False, False, True]]),
        seq_memory_cell=jnp.asarray([3], dtype=jnp.int32),
        seq_side_label=jnp.asarray([1], dtype=jnp.int32),
    )
    one_episode = jnp.zeros((config.model.memory_num_side_cells,), dtype=jnp.float32).at[3].set(1.0)
    info = {
        "diagnostic/v35_write_eligible_count": jnp.asarray(1.0),
        "diagnostic/v35_commit_success_count": jnp.asarray(1.0),
        "diagnostic/v35_write_feature_term_count": jnp.asarray(1.0),
        "diagnostic/v35_read_state_valid_count": jnp.asarray(1.0),
        "diagnostic/v35_read_feature_term_count": jnp.asarray(1.0),
        "diagnostic/v35_transition_count": jnp.asarray(3.0),
        "diagnostic/v35_write_episode_count": one_episode,
        "diagnostic/v35_read_episode_count": one_episode,
        "diagnostic/v35_degenerate_write_count": jnp.asarray(0.0),
        "diagnostic/v35_state_invalid_d_count": jnp.asarray(0.0),
        "diagnostic/v35_state_valid_mismatch_count": jnp.asarray(0.0),
        "diagnostic/v35_reachable_mismatch_count": jnp.asarray(0.0),
        "diagnostic/v35_invalid_gap_count": jnp.asarray(0.0),
        "diagnostic/v35_padding_gap_count": jnp.asarray(0.0),
        "diagnostic/v35_illegal_write_decision_overlap_count": jnp.asarray(0.0),
        "diagnostic/v35_invalid_cell_count": jnp.asarray(0.0),
    }
    return config, observation, info


def test_v35_runtime_guard_reconciles_valid_effective_batch_exactly():
    config, observation, info = _valid_v35_runtime_guard_fixture()
    violations = train._v35_runtime_guard_vector(config, observation, info)
    np.testing.assert_array_equal(violations, np.zeros(len(train._V35_RUNTIME_GUARD_NAMES), dtype=bool))


def test_v35_runtime_guard_names_commit_and_runtime_failures():
    config, observation, info = _valid_v35_runtime_guard_fixture()
    info["diagnostic/v35_commit_success_count"] = jnp.asarray(0.0)
    info["diagnostic/v35_degenerate_write_count"] = jnp.asarray(1.0)
    violations = np.asarray(train._v35_runtime_guard_vector(config, observation, info))
    failed = {name for name, value in zip(train._V35_RUNTIME_GUARD_NAMES, violations, strict=True) if value}
    assert failed == {"commit_count_mismatch", "degenerate_write"}


def test_v35_runtime_guard_checkify_throws_named_error_before_accepting_update():
    violations = jnp.zeros((len(train._V35_RUNTIME_GUARD_NAMES),), dtype=bool)
    bad_index = train._V35_RUNTIME_GUARD_NAMES.index("credit_reachable_mismatch")
    violations = violations.at[bad_index].set(True)
    checked = train.checkify.checkify(train._check_v35_runtime_guard)
    error, _ = checked(violations)
    with pytest.raises(Exception, match="credit_reachable_mismatch"):
        error.throw()


def test_v35_registered_config_is_fresh_base_and_calibration_locked(monkeypatch, tmp_path: pathlib.Path):
    config = _config.get_config("pi05_yam_mem_v35")
    assert isinstance(config.weight_loader, _config.weight_loaders.AuditedPartialCheckpointWeightLoader)
    assert config.weight_loader.params_path == "gs://openpi-assets/checkpoints/pi05_base/params"
    assert config.model.memory_injection_gate_init == 0.5
    assert config.ema_decay is None
    with pytest.raises(ValueError, match="train-only injection calibration"):
        train._validate_v35_training_ready(config)
    calibrated = _calibrated_v35_config(monkeypatch, tmp_path)
    train._validate_v35_training_ready(calibrated)
    norm_path = pathlib.Path(calibrated.data.assets.assets_dir) / calibrated.data.repo_id / "norm_stats.json"
    norm_path.write_text('{"heldout_leak": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="norm_stats.json bytes"):
        train._validate_v35_training_ready(calibrated)


def test_v35_norm_storage_seal_rechecks_train_files_but_not_heldout_media(monkeypatch, tmp_path: pathlib.Path):
    config = _calibrated_v35_config(monkeypatch, tmp_path)
    dataset_root = pathlib.Path(config.data.base_config.lerobot_dataset_root)
    heldout_data = dataset_root / "data" / "chunk-000" / "episode_000074.parquet"
    heldout_video = dataset_root / "videos" / "chunk-000" / "observation.images.top" / "episode_000074.mp4"
    heldout_data.write_bytes(b"heldout-data")
    heldout_video.write_bytes(b"heldout-video")

    train._validate_v35_training_ready(config)
    heldout_video.write_bytes(b"changed-heldout-video")
    train._validate_v35_training_ready(config)

    train_data = dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    train_data.write_bytes(b"changed-train-data")
    with pytest.raises(ValueError, match="file list/size/SHA256"):
        train._validate_v35_training_ready(config)


def test_v35_norm_storage_seal_allows_embedded_images_and_ignores_transfer_bundle(monkeypatch, tmp_path: pathlib.Path):
    config = _calibrated_v35_config(monkeypatch, tmp_path, include_videos=False)
    transfer_bundle = pathlib.Path(config.data.base_config.lerobot_dataset_root) / "meta/v35_training_bundle"
    transfer_bundle.mkdir()
    (transfer_bundle / "TRANSFER_MANIFEST.json").write_text('{"portable": true}')

    train._validate_v35_training_ready(config)


def test_v35_norm_provenance_requires_raw_train_row_protocol(monkeypatch, tmp_path: pathlib.Path):
    config = _calibrated_v35_config(monkeypatch, tmp_path)
    provenance_path = pathlib.Path(config.data.assets.assets_dir) / config.data.repo_id / "norm_stats_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["computation"]["protocol"] = "transformed-dataset-v1"
    provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="frozen train-only split"):
        train._validate_v35_training_ready(config)


def test_v35_checkpoint_rungs_are_completed_update_counts_and_legacy_is_unchanged():
    v35 = _config.get_config("pi05_yam_mem_v35")
    assert v35.num_workers == 0
    assert train._step_labels_and_save_decision(v35, loop_step=0, start_step=0) == (1, False, 1)
    assert train._step_labels_and_save_decision(v35, loop_step=249, start_step=0) == (250, True, 250)
    assert train._step_labels_and_save_decision(v35, loop_step=749, start_step=0) == (750, False, 750)
    assert train._step_labels_and_save_decision(v35, loop_step=999, start_step=0) == (1_000, True, 1_000)

    legacy = _config.get_config("pi05_yam_mem_v34")
    assert train._step_labels_and_save_decision(legacy, loop_step=250, start_step=0) == (250, True, 250)


def test_v35_cumulative_gate_d_counters_survive_multiple_accepted_updates():
    telemetry = train._new_v35_cumulative_telemetry()

    def info(*, severe, write_bind, write_terms, read_bind, read_terms, grad_max):
        return {
            "diagnostic/v35_pre_shared_clip_update_count": jnp.asarray(1.0),
            "diagnostic/v35_pre_shared_clip_severe_count": jnp.asarray(severe, dtype=jnp.float32),
            "diagnostic/v35_write_feature_clip_bind_sum": jnp.asarray(write_bind, dtype=jnp.float32),
            "diagnostic/v35_write_feature_term_count": jnp.asarray(write_terms, dtype=jnp.float32),
            "diagnostic/v35_read_feature_clip_bind_sum": jnp.asarray(read_bind, dtype=jnp.float32),
            "diagnostic/v35_read_feature_term_count": jnp.asarray(read_terms, dtype=jnp.float32),
            "diagnostic/v35_pre_shared_clip_grad_norm_max": jnp.asarray(grad_max, dtype=jnp.float32),
        }

    train._accumulate_v35_cumulative_telemetry(
        telemetry,
        info(severe=0, write_bind=2, write_terms=10, read_bind=1, read_terms=8, grad_max=3.0),
        completed_updates=1,
    )
    train._accumulate_v35_cumulative_telemetry(
        telemetry,
        info(severe=1, write_bind=3, write_terms=12, read_bind=2, read_terms=9, grad_max=12.0),
        completed_updates=2,
    )

    assert telemetry == {
        "schema_version": 1,
        "accepted_update_count": 2,
        "finite_accepted_update_count": 2,
        "pre_shared_severe_clip_count": 1,
        "pre_shared_update_count": 2,
        "write_feature_cap_bind_numerator": 5,
        "write_feature_cap_bind_denominator": 22,
        "read_feature_cap_bind_numerator": 3,
        "read_feature_cap_bind_denominator": 17,
        "pre_shared_grad_norm_max": 12.0,
    }
    train._validate_v35_cumulative_telemetry(telemetry, completed_updates=2)


def test_v35_checkpoint_protocol_requires_exact_branch_targets_and_rungs():
    pilot = _config.get_config("pi05_yam_mem_v35")
    train._validate_v35_checkpoint_protocol(pilot, resuming=False)

    with pytest.raises(ValueError, match="exact completed-update"):
        train._validate_v35_checkpoint_protocol(
            dataclasses.replace(pilot, checkpoint_steps=(250, 500)),
            resuming=False,
        )
    with pytest.raises(ValueError, match="frozen 1,000-update pilot"):
        train._validate_v35_checkpoint_protocol(
            dataclasses.replace(
                pilot,
                num_train_steps=2_500,
                checkpoint_steps=(250, 500, 1_000, 2_500),
            ),
            resuming=False,
        )

    extension = dataclasses.replace(
        pilot,
        resume=True,
        num_train_steps=2_500,
        checkpoint_steps=(250, 500, 1_000, 2_500),
    )
    train._validate_v35_checkpoint_protocol(extension, resuming=True, latest_step=1_000)
    full = dataclasses.replace(
        pilot,
        resume=True,
        num_train_steps=10_000,
        checkpoint_steps=(250, 500, 1_000, 2_500, 5_000, 10_000),
    )
    train._validate_v35_checkpoint_protocol(full, resuming=True, latest_step=1_000)
    train._validate_v35_checkpoint_protocol(full, resuming=True, latest_step=2_500)
    with pytest.raises(ValueError, match="separately sealed external rung"):
        train._validate_v35_checkpoint_protocol(full, resuming=True, latest_step=5_000)

    with pytest.raises(ValueError, match="exact completed-update"):
        train._validate_v35_checkpoint_protocol(
            dataclasses.replace(full, checkpoint_steps=(250, 500, 1_000, 2_500, 10_000)),
            resuming=True,
            latest_step=2_500,
        )
    with pytest.raises(ValueError, match="may resume only"):
        train._validate_v35_checkpoint_protocol(full, resuming=True, latest_step=500)


def test_v35_frozen_cast_keeps_injection_gate_fp32_and_validates_effective_value():
    class TinyGate(nnx.Module):
        def __init__(self):
            self.memory_inject_w = nnx.Param(jnp.full((4,), jnp.arctanh(0.5), dtype=jnp.float32))
            self.memory_gate = nnx.Param(jnp.ones((4,), dtype=jnp.float32))

    config = _config.get_config("pi05_yam_mem_v35")
    params = train._cast_frozen_params(config, nnx.state(TinyGate()))
    assert params["memory_inject_w"].value.dtype == jnp.float32
    assert params["memory_gate"].value.dtype == jnp.bfloat16
    train._validate_v35_initialized_gate(config, params)

    bad_model = TinyGate()
    bad_model.memory_inject_w.value = jnp.zeros((4,), dtype=jnp.float32)
    with pytest.raises(ValueError, match="does not match"):
        train._validate_v35_initialized_gate(config, nnx.state(bad_model))


def test_v35_initialization_identity_hashes_actual_step0_tree_and_binds_calibration(
    monkeypatch,
    tmp_path: pathlib.Path,
):
    class TinyGate(nnx.Module):
        def __init__(self):
            self.memory_inject_w = nnx.Param(jnp.full((4,), jnp.arctanh(0.5), dtype=jnp.float32))
            self.memory_gate = nnx.Param(jnp.ones((4,), dtype=jnp.float32))

    template = _config.get_config("pi05_yam_mem_v35")
    params = train._cast_frozen_params(template, nnx.state(TinyGate()))
    step0_sha256 = _weight_loaders.parameter_tree_sha256(params.to_pure_dict())
    config = dataclasses.replace(
        _calibrated_v35_config(monkeypatch, tmp_path / "inputs", source_sha256=step0_sha256),
        exp_name="identity-test",
        checkpoint_base_dir=str(pathlib.Path(os.environ["MEMORY_PROJECT_ROOT"]) / "v35/checkpoints"),
    )
    checkpoint_dir = pathlib.Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True)
    graft = {"tree_hashes": {"source_sha256": "1" * 64, "target_schema_sha256": "2" * 64}}
    graft["manifest_sha256"] = hashlib.sha256(train._canonical_json(graft).encode()).hexdigest()
    (checkpoint_dir / "initialization_graft_manifest.json").write_text(json.dumps(graft), encoding="utf-8")

    identity_path = train._write_v35_initialization_identity(config, params)

    assert identity_path is not None
    identity = json.loads(identity_path.read_text())
    assert identity["initialization_seed"] == config.seed
    assert identity["actual_step0_parameter_tree_sha256"] == step0_sha256
    assert identity["calibration_id"] == config.model.memory_v35_calibration_id
    assert identity["effective_gate_min"] == pytest.approx(0.5)
    assert identity["memory_calibration"] == {
        # Artifact-boundary alpha is the FP32 runtime value, not the float64 config literal.
        "alpha_step": float(np.float32(0.01)),
        "memory_injection_c": 3.0,
        "memory_injection_tau": 0.04,
    }
    assert set(identity["artifact_hashes"]) == {
        "calibration_artifact_sha256",
        "episode_manifest_sha256",
        "norm_stats_provenance_sha256",
        "norm_stats_sha256",
        "train_storage_sha256",
        "initialization_graft_manifest_file_sha256",
        "initialization_graft_manifest_self_sha256",
    }
    unsigned_identity = {key: value for key, value in identity.items() if key != "identity_sha256"}
    assert identity["identity_sha256"] == hashlib.sha256(train._canonical_json(unsigned_identity).encode()).hexdigest()

    provenance = train._snapshot_v35_checkpoint_provenance(config, identity_path)
    assert set(provenance) == set(train._V35_CHECKPOINT_PROVENANCE_FILENAMES.values())
    assert provenance["v35_initialization_manifest.json"] == identity_path.read_bytes()

    checkpoint_assets = checkpoint_dir / "1000" / "assets"
    checkpoint_assets.mkdir(parents=True)
    for name, contents in provenance.items():
        (checkpoint_assets / name).write_bytes(contents)
    resumed = dataclasses.replace(
        config,
        resume=True,
        num_train_steps=2_500,
        checkpoint_steps=(250, 500, 1_000, 2_500),
    )
    train._validate_v35_resume_checkpoint_assets(
        resumed,
        checkpoint_step=1_000,
        identity_path=identity_path,
    )

    embedded_calibration = checkpoint_assets / "v35_calibration_artifact.json"
    embedded_calibration.write_bytes(embedded_calibration.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="not byte-identical"):
        train._validate_v35_resume_checkpoint_assets(
            resumed,
            checkpoint_step=1_000,
            identity_path=identity_path,
        )

    identity["config_name"] = "tampered"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash is invalid"):
        train._validate_v35_root_identity(resumed, identity_path)


class _ToyModel(_model.BaseModel):
    """Deterministic model for exact full-batch/accumulated-update comparisons."""

    def __init__(self, *, memory_probe_weight: float = 0.5, memory_probe_diagnostic: bool = False):
        super().__init__(action_dim=1, action_horizon=1, max_token_len=1)
        self.kernel = nnx.Param(jnp.asarray([[0.7]], dtype=jnp.float32))
        self.probe_head = nnx.Param(jnp.asarray([[0.9]], dtype=jnp.float32))
        self.ce_loss_weight = 0.3
        self.memory_probe_weight = memory_probe_weight
        self.memory_probe_diagnostic = memory_probe_diagnostic

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, train
        prediction = observation.state[..., 0] * self.kernel.value[0, 0]
        target = actions[..., 0, 0]
        flow = jnp.mean(jnp.square(prediction - target), axis=1)
        ce = jnp.mean(jnp.square(prediction + 0.25 - target), axis=1)

        # Deterministic synthetic write telemetry exercises the exact same sum/count/max
        # reducers as the production sequence model, including gradient accumulation.
        valid = observation.seq_step_mask
        synthetic_write_norm = jnp.abs(observation.state[..., 0])
        write_metrics = {
            "write_grad_norm_sum": jnp.sum(jnp.where(valid, synthetic_write_norm, 0.0), axis=1),
            "write_valid_count": jnp.sum(valid.astype(jnp.float32), axis=1),
            "write_clip_count": jnp.sum((valid & (synthetic_write_norm > 0.5)).astype(jnp.float32), axis=1),
            "write_severe_clip_count": jnp.sum((valid & (synthetic_write_norm > 0.9)).astype(jnp.float32), axis=1),
            "write_grad_norm_max": jnp.max(jnp.where(valid, synthetic_write_norm, 0.0), axis=1),
        }

        if self.memory_probe_weight == 0 and not self.memory_probe_diagnostic:
            return {"flow": flow, "ce": ce, **write_metrics}

        probe_prediction = prediction * self.probe_head.value[0, 0]
        if self.memory_probe_diagnostic:
            probe_prediction = jax.lax.stop_gradient(probe_prediction)
        probe_mask = observation.seq_probe_mask.astype(jnp.float32)
        probe_target = observation.seq_probe_labels.astype(jnp.float32)
        probe_ce = jnp.square(probe_prediction - probe_target) * probe_mask
        probe_correct = (jnp.abs(probe_prediction - probe_target) < 0.5).astype(jnp.float32) * probe_mask
        visible = observation.seq_probe_visible.astype(jnp.float32) * probe_mask
        return {
            "flow": flow,
            "ce": ce,
            **write_metrics,
            "probe_ce_sum": jnp.sum(probe_ce, axis=1),
            "probe_count": jnp.sum(probe_mask, axis=1),
            "probe_correct": jnp.sum(probe_correct, axis=1),
            "probe_count_visible": jnp.sum(visible, axis=1),
            "probe_correct_visible": jnp.sum(probe_correct * visible, axis=1),
            "probe_correct_grid": probe_correct,
            "probe_active_grid": probe_mask,
        }

    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        return observation.state[..., :1, None]


def _toy_train_state(config: _config.TrainConfig) -> training_utils.TrainState:
    model = _ToyModel(
        memory_probe_weight=config.model.memory_probe_weight,
        memory_probe_diagnostic=config.model.memory_probe_diagnostic,
    )
    params = nnx.state(model)
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule)
    trainable_params = params.filter(config.trainable_filter)
    return training_utils.TrainState(
        step=jnp.asarray(7),
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(trainable_params),
        ema_decay=config.ema_decay,
        ema_params=params,
    )


def _toy_batch(batch_size: int = 12, sequence_steps: int = 4):
    state = jnp.linspace(-1.0, 1.0, batch_size * sequence_steps).reshape(batch_size, sequence_steps, 1)
    actions = (0.4 * state + 0.1)[..., None]
    step = jnp.arange(sequence_steps)[None]
    sample = jnp.arange(batch_size)[:, None]
    probe_mask = (step + sample) % 3 != 0
    observation = _model.Observation(
        images={},
        image_masks={},
        state=state,
        seq_step_mask=jnp.ones((batch_size, sequence_steps), dtype=bool),
        seq_block_boundary=jnp.zeros((batch_size, sequence_steps), dtype=bool),
        seq_probe_labels=((step + sample) % 2).astype(jnp.int32),
        seq_probe_mask=probe_mask,
        seq_probe_visible=probe_mask & (step < 2),
    )
    return observation, actions


def test_probe_grid_metrics_stack_across_sequence_buckets():
    correct20, active20 = train._pad_probe_grids(jnp.asarray([1.0] * 20), jnp.asarray([2.0] * 20), 60)
    correct40, active40 = train._pad_probe_grids(jnp.asarray([3.0] * 40), jnp.asarray([4.0] * 40), 60)
    correct60, active60 = train._pad_probe_grids(jnp.asarray([5.0] * 60), jnp.asarray([10.0] * 60), 60)

    reduced = train._reduce_infos(
        [
            {
                "diagnostic/probe_correct_grid": correct20,
                "diagnostic/probe_active_grid": active20,
            },
            {
                "diagnostic/probe_correct_grid": correct40,
                "diagnostic/probe_active_grid": active40,
            },
            {
                "diagnostic/probe_correct_grid": correct60,
                "diagnostic/probe_active_grid": active60,
            },
        ]
    )

    assert set(reduced) == {"diagnostic/probe_accuracy_by_step"}
    np.testing.assert_allclose(reduced["diagnostic/probe_accuracy_by_step"][:20], 9 / 16, rtol=1e-6)
    np.testing.assert_allclose(reduced["diagnostic/probe_accuracy_by_step"][20:40], 8 / 14, rtol=1e-6)
    np.testing.assert_allclose(reduced["diagnostic/probe_accuracy_by_step"][40:], 0.5, rtol=1e-6)


def test_expensive_norm_metrics_are_count_weighted_across_log_window():
    reduced = train._reduce_infos(
        [
            {
                "loss": jnp.asarray(1.0),
                "grad_norm": jnp.asarray(9.0),
                "param_norm": jnp.asarray(12.0),
                "_expensive_norm_count": jnp.asarray(1.0),
            },
            {
                "loss": jnp.asarray(3.0),
                "grad_norm": jnp.asarray(0.0),
                "param_norm": jnp.asarray(0.0),
                "_expensive_norm_count": jnp.asarray(0.0),
            },
            {
                "loss": jnp.asarray(5.0),
                "grad_norm": jnp.asarray(0.0),
                "param_norm": jnp.asarray(0.0),
                "_expensive_norm_count": jnp.asarray(0.0),
            },
        ]
    )

    assert reduced["loss"] == pytest.approx(3.0)
    assert reduced["grad_norm"] == pytest.approx(9.0)
    assert reduced["param_norm"] == pytest.approx(12.0)
    assert "_expensive_norm_count" not in reduced


def test_write_inner_metrics_pool_valid_writes_and_keep_true_window_max():
    reduced = train._reduce_infos(
        [
            {
                "diagnostic/write_inner_grad_sum": jnp.asarray(4.0),
                "diagnostic/write_inner_valid_count": jnp.asarray(2.0),
                "diagnostic/write_inner_clip_count": jnp.asarray(1.0),
                "diagnostic/write_inner_severe_clip_count": jnp.asarray(0.0),
                "diagnostic/write_inner_grad_max": jnp.asarray(3.0),
            },
            {
                "diagnostic/write_inner_grad_sum": jnp.asarray(36.0),
                "diagnostic/write_inner_valid_count": jnp.asarray(9.0),
                "diagnostic/write_inner_clip_count": jnp.asarray(6.0),
                "diagnostic/write_inner_severe_clip_count": jnp.asarray(3.0),
                "diagnostic/write_inner_grad_max": jnp.asarray(11.0),
            },
        ]
    )

    assert reduced["diagnostic/write_inner_grad_norm"] == pytest.approx(40 / 11)
    assert reduced["diagnostic/write_inner_clip_fraction"] == pytest.approx(7 / 11)
    assert reduced["diagnostic/write_inner_severe_clip_fraction"] == pytest.approx(3 / 11)
    assert reduced["diagnostic/write_inner_grad_max"] == pytest.approx(11.0)


def test_diagnostic_probe_metrics_are_count_weighted_and_namespaced():
    reduced = train._reduce_infos(
        [
            {
                "diagnostic/probe_loss_numerator": jnp.asarray(0.0),
                "diagnostic/probe_count": jnp.asarray(0.0),
                "diagnostic/probe_correct": jnp.asarray(0.0),
                "diagnostic/probe_visible_count": jnp.asarray(0.0),
                "diagnostic/probe_visible_correct": jnp.asarray(0.0),
            },
            {
                "diagnostic/probe_loss_numerator": jnp.asarray(2.0),
                "diagnostic/probe_count": jnp.asarray(4.0),
                "diagnostic/probe_correct": jnp.asarray(3.0),
                "diagnostic/probe_visible_count": jnp.asarray(1.0),
                "diagnostic/probe_visible_correct": jnp.asarray(1.0),
            },
        ]
    )

    assert not any(key.startswith("probe_") for key in reduced)
    assert reduced["diagnostic/probe_count"] == 2.0  # mean live probes per optimizer step
    assert reduced["diagnostic/probe_loss"] == 0.5
    assert reduced["diagnostic/probe_accuracy"] == 0.75
    assert reduced["diagnostic/probe_accuracy_visible"] == 1.0
    assert reduced["diagnostic/probe_accuracy_hidden"] == 2 / 3


def test_pad_probe_grids_rejects_overlong_metric():
    with pytest.raises(ValueError, match="exceeds configured maximum"):
        train._pad_probe_grids(jnp.ones(61), jnp.ones(61), 60)


def test_detached_diagnostic_probe_does_not_change_total_loss_or_main_update():
    debug = _config.get_config("debug_mem")
    common_model = dataclasses.replace(debug.model, memory_probe_weight=0.0)
    common = dataclasses.replace(
        debug,
        model=dataclasses.replace(common_model, memory_probe_diagnostic=False),
        batch_size=12,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0e6),
        ema_decay=0.9,
    )
    diagnostic = dataclasses.replace(
        common,
        model=dataclasses.replace(common.model, memory_probe_diagnostic=True),
    )
    batch = _toy_batch()

    plain_state = _toy_train_state(common)
    diagnostic_state = _toy_train_state(diagnostic)
    legacy_probe_state = _toy_train_state(dataclasses.replace(common, model=debug.model))
    assert jax.tree.structure(plain_state.opt_state) == jax.tree.structure(legacy_probe_state.opt_state)

    # Simulate a resumed probe-trained checkpoint, whose saved EMA probe can legitimately differ
    # from its raw train parameter. No-probe continuation must preserve both representations.
    probe_filter = nnx_utils.PathRegex(r".*probe_head.*")

    def offset_probe_ema(state):
        return dataclasses.replace(
            state,
            ema_params=nnx_utils.state_map(
                state.ema_params,
                probe_filter,
                lambda variable: variable.replace(variable.value + 0.25),
            ),
        )

    plain_state = offset_probe_ema(plain_state)
    diagnostic_state = offset_probe_ema(diagnostic_state)
    plain_next, plain_info = train.train_step(common, jax.random.key(123), plain_state, batch)
    diagnostic_next, diagnostic_info = train.train_step(diagnostic, jax.random.key(123), diagnostic_state, batch)

    np.testing.assert_array_equal(plain_info["loss"], diagnostic_info["loss"])
    np.testing.assert_array_equal(plain_info["flow_loss"], diagnostic_info["flow_loss"])
    np.testing.assert_array_equal(plain_info["ce_loss"], diagnostic_info["ce_loss"])
    assert not any(key.startswith("diagnostic/probe") for key in plain_info)
    assert "diagnostic/probe_loss_numerator" in diagnostic_info
    assert "probe_loss" not in diagnostic_info

    # Identical post-update parameters/optimizer state prove that the diagnostic auxiliary
    # outputs did not alter any gradient consumed by the main optimizer.
    for plain, with_diagnostic in zip(
        jax.tree.leaves(plain_next.params), jax.tree.leaves(diagnostic_next.params), strict=True
    ):
        np.testing.assert_array_equal(plain, with_diagnostic)
    for plain, with_diagnostic in zip(
        jax.tree.leaves(plain_next.opt_state), jax.tree.leaves(diagnostic_next.opt_state), strict=True
    ):
        np.testing.assert_array_equal(plain, with_diagnostic)
    for plain, with_diagnostic in zip(
        jax.tree.leaves(plain_next.ema_params), jax.tree.leaves(diagnostic_next.ema_params), strict=True
    ):
        np.testing.assert_array_equal(plain, with_diagnostic)
    np.testing.assert_array_equal(plain_state.params["probe_head"].value, plain_next.params["probe_head"].value)
    np.testing.assert_array_equal(
        diagnostic_state.params["probe_head"].value, diagnostic_next.params["probe_head"].value
    )
    np.testing.assert_array_equal(plain_state.ema_params["probe_head"].value, plain_next.ema_params["probe_head"].value)
    np.testing.assert_array_equal(
        diagnostic_state.ema_params["probe_head"].value,
        diagnostic_next.ema_params["probe_head"].value,
    )


@pytest.mark.parametrize("accumulation_steps", [3, 6])
def test_gradient_accumulation_matches_one_effective_full_batch_update(accumulation_steps: int):
    base_config = dataclasses.replace(
        _config.get_config("debug"),
        batch_size=12,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0e6),
        ema_decay=0.9,
    )
    accumulated_config = dataclasses.replace(base_config, gradient_accumulation_steps=accumulation_steps)
    observation, actions = _toy_batch()
    # Heterogeneous valid lengths catch the old bug where per-sequence fractions were
    # averaged and short sequences received the same weight as long ones.
    lengths = (jnp.arange(base_config.batch_size) % 4) + 1
    heterogeneous_valid = jnp.arange(observation.seq_step_mask.shape[1])[None, :] < lengths[:, None]
    observation = dataclasses.replace(observation, seq_step_mask=heterogeneous_valid)
    full_batch = (observation, actions)
    microbatch_size = base_config.batch_size // accumulation_steps
    accumulated_batch = jax.tree.map(lambda x: x.reshape(accumulation_steps, microbatch_size, *x.shape[1:]), full_batch)

    full_state, full_info = train.train_step(
        base_config, jax.random.key(123), _toy_train_state(base_config), full_batch
    )
    accumulated_state, accumulated_info = train.train_step(
        accumulated_config,
        jax.random.key(123),
        _toy_train_state(accumulated_config),
        accumulated_batch,
    )

    assert int(full_state.step) == int(accumulated_state.step) == 8
    for full, accumulated in zip(
        jax.tree.leaves(full_state.params), jax.tree.leaves(accumulated_state.params), strict=True
    ):
        np.testing.assert_allclose(full, accumulated, rtol=2e-6, atol=2e-6)
    for full, accumulated in zip(
        jax.tree.leaves(full_state.opt_state), jax.tree.leaves(accumulated_state.opt_state), strict=True
    ):
        np.testing.assert_allclose(full, accumulated, rtol=2e-6, atol=2e-6)
    for full, accumulated in zip(
        jax.tree.leaves(full_state.ema_params), jax.tree.leaves(accumulated_state.ema_params), strict=True
    ):
        np.testing.assert_allclose(full, accumulated, rtol=2e-6, atol=2e-6)
    assert set(full_info) == set(accumulated_info)
    for key in full_info:
        np.testing.assert_allclose(full_info[key], accumulated_info[key], rtol=2e-6, atol=2e-6, err_msg=key)
    assert float(full_info["diagnostic/write_inner_grad_max"]) == pytest.approx(1.0)
    assert float(accumulated_info["diagnostic/write_inner_valid_count"]) == pytest.approx(30.0)


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)


class _ToyV34Model(_model.BaseModel):
    """Toy model emitting the v3.4 aux/ladder loss keys so train_step's plumbing -- macro CE,
    isolated ladder SGD, metric namespacing -- is exercised exactly as the real model would."""

    def __init__(self, *, ladder_scale: float, aux_weight: float):
        super().__init__(action_dim=1, action_horizon=1, max_token_len=1)
        self.kernel = nnx.Param(jnp.asarray([[0.7]], dtype=jnp.float32))
        # Name matches train.MEMORY_PATH_FILTER; zero-init so every pre-existing expected value
        # (pooled etc.) is unchanged while the param still receives a nonzero gradient.
        self.memory_gain = nnx.Param(jnp.asarray([[0.0]], dtype=jnp.float32))
        self.ladder_writer_head = nnx.Param(jnp.asarray([[0.5]], dtype=jnp.float32))
        self.ladder_read_head = nnx.Param(jnp.asarray([[0.25]], dtype=jnp.float32))
        self.ce_loss_weight = 0.3
        self.memory_probe_weight = 0.0
        self.memory_probe_diagnostic = False
        self.memory_aux_loss_weight = aux_weight
        self.memory_aux_margin_weight = 0.0
        self.memory_aux_side_class_ids = (1,)
        self.ladder_scale = ladder_scale

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, train
        prediction = observation.state[..., 0] * (self.kernel.value[0, 0] + self.memory_gain.value[0, 0])
        target = actions[..., 0, 0]
        flow = jnp.mean(jnp.square(prediction - target), axis=1)
        ce = jnp.mean(jnp.square(prediction + 0.25 - target), axis=1)
        pooled = jnp.mean(prediction)
        losses = {
            "flow": flow,
            "ce": ce,
            # three aux classes with counts (2, 1, 0): macro CE = (sum0/2 + sum1/1) / 2
            "aux_ce_class_sum": jnp.stack([2.0 * jnp.square(pooled), jnp.square(pooled - 1.0), 0.0 * pooled]),
            "aux_count_class": jnp.asarray([2.0, 1.0, 0.0]),
            "aux_correct_class": jnp.asarray([1.0, 0.0, 0.0]),
        }
        blocked = jax.lax.stop_gradient(pooled)
        for name, head in (("ladder_writer", self.ladder_writer_head), ("ladder_read", self.ladder_read_head)):
            head_out = head.value[0, 0] * blocked
            losses[f"{name}_ce_sum"] = self.ladder_scale * jnp.square(head_out - 1.0)
            losses[f"{name}_count"] = jnp.asarray(1.0)
            losses[f"{name}_correct"] = jnp.asarray(0.0)
        return losses

    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        return observation.state[..., :1, None]


def _v34_toy_setup(*, ladder_scale: float, aux_weight: float = 0.5):
    v34 = _config.get_config("pi05_yam_mem_v34")
    config = dataclasses.replace(
        v34,
        exp_name="toy",
        batch_size=4,
        gradient_accumulation_steps=1,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        probe_lr=0.05,
        ema_decay=None,
        freeze_filter=nnx.Nothing,
    )
    model = _ToyV34Model(ladder_scale=ladder_scale, aux_weight=aux_weight)
    params = nnx.state(model)
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule)
    state = training_utils.TrainState(
        step=jnp.asarray(3),
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(config.trainable_filter)),
        ema_decay=None,
        ema_params=None,
    )
    batch_state = jnp.linspace(-0.4, 1.0, 4 * 2).reshape(4, 2, 1)  # nonzero mean: aux grads live
    observation = _model.Observation(
        images={},
        image_masks={},
        state=batch_state,
        seq_step_mask=jnp.ones((4, 2), dtype=bool),
        seq_block_boundary=jnp.zeros((4, 2), dtype=bool),
    )
    actions = (0.4 * batch_state + 0.1)[..., None]
    return config, state, (observation, actions)


def test_v34_ladder_isolation_and_probe_sgd_inside_train_step():
    """Huge ladder-head gradients must not perturb the main update (they are removed before
    the global clip), the ladder heads must move by exactly -probe_lr * grad, and their Adam
    moments must stay zero."""
    rng = jax.random.key(0)
    config_huge, state_huge, batch = _v34_toy_setup(ladder_scale=1e6)
    config_zero, state_zero, _ = _v34_toy_setup(ladder_scale=0.0)

    new_huge, info_huge = train.train_step(config_huge, rng, state_huge, batch)
    new_zero, info_zero = train.train_step(config_zero, rng, state_zero, batch)

    np.testing.assert_array_equal(
        np.asarray(new_huge.params["kernel"].value), np.asarray(new_zero.params["kernel"].value)
    )
    np.testing.assert_array_equal(np.asarray(info_huge["grad_norm"]), np.asarray(info_zero["grad_norm"]))

    # the ladder heads take exactly one isolated SGD step: d/dW [s*(W*p - 1)^2] = 2s(W*p-1)p
    observation, _ = batch
    model = nnx.merge(state_huge.model_def, state_huge.params)
    pooled = float(jnp.mean(observation.state[..., 0] * 0.7))
    for name, w0 in (("ladder_writer_head", 0.5), ("ladder_read_head", 0.25)):
        grad = 2.0 * 1e6 * (w0 * pooled - 1.0) * pooled
        expected = w0 - config_huge.probe_lr * grad
        np.testing.assert_allclose(float(new_huge.params[name].value[0, 0]), expected, rtol=1e-5)
    del model

    # ladder Adam moments untouched (grads were zeroed before tx.update)
    for path, leaf in jax.tree_util.tree_leaves_with_path(new_huge.opt_state):
        if "ladder_" in jax.tree_util.keystr(path):
            np.testing.assert_array_equal(np.asarray(leaf), 0.0)

    # metric plumbing: rung losses and correct/count reach the info dict
    for rung in ("ladder_writer", "ladder_read"):
        assert f"diagnostic/{rung}_loss" in info_huge
        assert f"diagnostic/{rung}_count" in info_huge


def test_v34_aux_macro_ce_reaches_loss_and_kernel():
    rng = jax.random.key(1)
    config_on, state_on, batch = _v34_toy_setup(ladder_scale=0.0, aux_weight=0.5)
    config_off, state_off, _ = _v34_toy_setup(ladder_scale=0.0, aux_weight=0.0)

    new_on, info_on = train.train_step(config_on, rng, state_on, batch)
    new_off, info_off = train.train_step(config_off, rng, state_off, batch)

    observation, _ = batch
    pooled = float(jnp.mean(observation.state[..., 0] * 0.7))
    expected_macro = ((2.0 * pooled**2) / 2.0 + (pooled - 1.0) ** 2) / 2.0
    np.testing.assert_allclose(float(info_on["aux_loss"]), expected_macro, rtol=1e-5)
    # the aux CE trains the main model: the kernel's Adam first moment (= the clipped gradient
    # on a fresh optimizer) differs with the aux weight on vs off. (The parameter itself moves
    # identically on the very first step -- fresh-moment Adam is sign-normalized -- so the
    # moment, not the update, is the right witness.)
    assert float(info_on["loss"]) != float(info_off["loss"])
    moment_on = moment_off = None
    for path, leaf in jax.tree_util.tree_leaves_with_path(new_on.opt_state):
        if "kernel" in jax.tree_util.keystr(path):
            moment_on = np.asarray(leaf)
            break
    for path, leaf in jax.tree_util.tree_leaves_with_path(new_off.opt_state):
        if "kernel" in jax.tree_util.keystr(path):
            moment_off = np.asarray(leaf)
            break
    assert moment_on is not None
    assert not np.array_equal(moment_on, moment_off)
    # class-group accuracy metrics flow to reduction
    assert float(info_on["diagnostic/aux_side_count"]) == 1.0
    assert float(info_on["diagnostic/aux_phase_count"]) == 2.0


def test_v34_memory_grad_group_clip():
    """The memory-path group pre-clip (v34_run1 postmortem) zeroes/rescales ONLY the
    memory-path gradients before the shared global clip, logs their norm, and is a bit-exact
    no-op when it does not bind."""
    rng = jax.random.key(2)

    # clip = 0: memory-path grads are erased before tx.update -> their Adam moments stay
    # exactly zero, while the non-memory kernel still trains.
    config, state, batch = _v34_toy_setup(ladder_scale=0.0)
    config_zero = dataclasses.replace(config, memory_grad_clip=0.0)
    new_zero, info_zero = train.train_step(config_zero, rng, state, batch)
    assert float(info_zero["memory_grad_norm"]) > 0.0
    kernel_moments = memory_moments = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(new_zero.opt_state):
        key = jax.tree_util.keystr(path)
        if "memory_gain" in key:
            np.testing.assert_array_equal(np.asarray(leaf), 0.0)
            memory_moments += 1
        elif "'kernel'" in key:
            kernel_moments += 1
            assert np.any(np.asarray(leaf) != 0.0)
    assert memory_moments > 0
    assert kernel_moments > 0

    # a clip far above the actual norm is bit-exact with the clip disabled
    _, state_loose, _ = _v34_toy_setup(ladder_scale=0.0)
    _, state_off, _ = _v34_toy_setup(ladder_scale=0.0)
    new_loose, info_loose = train.train_step(dataclasses.replace(config, memory_grad_clip=1e9), rng, state_loose, batch)
    new_off, info_off = train.train_step(dataclasses.replace(config, memory_grad_clip=None), rng, state_off, batch)
    assert "memory_grad_norm" in info_loose
    assert "memory_grad_norm" not in info_off
    for name in ("kernel", "memory_gain"):
        np.testing.assert_array_equal(np.asarray(new_loose.params[name].value), np.asarray(new_off.params[name].value))
