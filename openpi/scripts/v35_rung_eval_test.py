from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v35_gate_artifacts as artifacts
    import v35_pilot_gate as pilot
    import v35_rung_eval as rung
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


def _manifest(tmp_path: Path) -> artifacts.FrozenManifest:
    episodes = [
        artifacts.ManifestEpisode(
            stable_id=f"train/{index:02d}",
            episode_index=index,
            collection="0830" if index < 23 else "0831",
            object_name="banana" if (index // 2) % 2 == 0 else "grey_pepper_box",
            part="" if index < 56 else "part1",
            target_side=index % 2,
            split="train",
        )
        for index in range(54)
    ]
    episodes.extend(
        artifacts.ManifestEpisode(
            stable_id=f"dev/{index}",
            episode_index=74 + index,
            collection="0816" if index < 4 else "0830",
            object_name="banana" if index % 4 < 2 else "grey_pepper_box",
            part="" if index < 4 else "part1",
            target_side=index % 2,
            split="development",
        )
        for index in range(8)
    )
    episodes.extend(
        artifacts.ManifestEpisode(
            stable_id=f"final/{index}",
            episode_index=82 + index,
            collection="0830",
            object_name="banana" if index < 4 else "grey_pepper_box",
            part="part2",
            target_side=index % 2,
            split="final_test",
        )
        for index in range(8)
    )
    return artifacts.FrozenManifest(
        path=tmp_path / "manifest.json",
        sha256="1" * 64,
        split_assignment_sha256="2" * 64,
        episodes=tuple(episodes),
    )


def _condition_arrays(manifest: artifacts.FrozenManifest) -> dict[str, np.ndarray]:
    dev = manifest.split("development")
    train = manifest.split("train")
    e_episode = np.repeat(np.arange(8, dtype=np.int16), 2)
    d_episode = np.repeat(np.arange(8, dtype=np.int16), 3)
    use_episode = np.repeat(np.arange(8, dtype=np.int16), 2)
    e_frame = np.concatenate([np.asarray([15, 30]) + 60 * i for i in range(8)]).astype(np.int32)
    d_frame = np.concatenate([np.asarray([45, 60, 75]) + 90 * i for i in range(8)]).astype(np.int32)
    use_frame = np.concatenate([np.asarray([60, 75]) + 90 * i for i in range(8)]).astype(np.int32)
    e_sign = np.asarray([1.0 if dev[i].target_side else -1.0 for i in e_episode], dtype=np.float32)
    d_sign = np.asarray([1.0 if dev[i].target_side else -1.0 for i in d_episode], dtype=np.float32)
    use_sign = np.asarray([1.0 if dev[i].target_side else -1.0 for i in use_episode], dtype=np.float32)
    train_sign = np.asarray([1.0 if episode.target_side else -1.0 for episode in train], dtype=np.float32)
    train_feature = np.stack([train_sign, np.linspace(-0.1, 0.1, 54, dtype=np.float32)], axis=1)
    return {
        "dev_stable_id": np.asarray([episode.stable_id for episode in dev]),
        "dev_target_side": np.asarray([episode.target_side for episode in dev], dtype=np.int8),
        "e_episode_ordinal": e_episode,
        "e_frame_index": e_frame,
        "writer_natural_score": e_sign,
        "writer_counterfactual_score": -e_sign,
        "d_episode_ordinal": d_episode,
        "d_frame_index": d_frame,
        "read_natural_score": d_sign,
        "read_reset_score": -np.ones_like(d_sign),
        "read_opposite_donor_score": -d_sign,
        "attention_memory_mass_natural": np.full(len(d_episode), 0.2, dtype=np.float32),
        "attention_memory_mass_reset": np.full(len(d_episode), 0.1, dtype=np.float32),
        "attention_memory_mass_zero": np.full(len(d_episode), 0.1, dtype=np.float32),
        "attention_uniform_baseline": np.full(len(d_episode), 0.05, dtype=np.float32),
        "use_episode_ordinal": use_episode,
        "use_frame_index": use_frame,
        "action_natural_score": use_sign,
        "action_reset_score": -np.ones_like(use_sign),
        "action_opposite_donor_score": -use_sign,
        "action_zero_score": -np.ones_like(use_sign),
        "action_direct_carry_score": use_sign,
        "action_prototype_correct_score": use_sign,
        "action_prototype_opposite_score": -use_sign,
        "train_stable_id": np.asarray([episode.stable_id for episode in train]),
        "train_writer_natural_feature": train_feature.astype(np.float32),
        "train_writer_counterfactual_feature": (-train_feature).astype(np.float32),
        "train_prototype_correct_score": train_sign,
        "train_prototype_opposite_score": -train_sign,
    }


def _write_envelope(path: Path, schema: str, payload: dict) -> dict:
    envelope = artifacts.artifact_envelope(schema, payload)
    artifacts.write_canonical_envelope(path, envelope, schema_version=schema)
    return envelope


def test_condition_reducer_derives_counts_donors_and_never_accepts_final_test(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    arrays = _condition_arrays(manifest)
    reduced = rung.reduce_condition_arrays(arrays, manifest=manifest)

    assert len(reduced["episodes"]) == 8
    assert all(record["eligible_e_frame_count"] == 2 for record in reduced["episodes"])
    assert all(record["eligible_d_frame_count"] == 3 for record in reduced["episodes"])
    assert all(record["use_pressure_frame_count"] == 2 for record in reduced["episodes"])
    assert all(record["read_opposite_donor_followed"] for record in reduced["episodes"])
    assert all(record["prototype_paired_action_flipped"] for record in reduced["episodes"])
    assert all(record["natural_prompt_correct"] for record in reduced["train_writer_oof"]["episodes"])
    assert all(record["counterfactual_prompt_correct"] for record in reduced["train_writer_oof"]["episodes"])
    mapping = rung.deterministic_dev_donors(manifest)
    assert mapping["dev/0"] == "dev/1"
    assert mapping["dev/1"] == "dev/0"

    contaminated = dict(arrays)
    ids = arrays["dev_stable_id"].copy()
    ids[-1] = "final/0"
    contaminated["dev_stable_id"] = ids
    with pytest.raises(rung.RungEvaluationError, match="development IDs"):
        rung.reduce_condition_arrays(contaminated, manifest=manifest)


def _mechanism(tmp_path: Path) -> tuple[Path, str, str, str]:
    checkpoint = "3" * 64
    initialization = "4" * 64
    calibration_id = f"sha256:{'5' * 64}"
    core_path = tmp_path / "core.json"
    core = _write_envelope(
        core_path,
        rung.CORE_SCHEMA_VERSION,
        {
            "checkpoint_parameter_tree_sha256": checkpoint,
            "checks": dict.fromkeys(rung.CORE_CHECKS, True),
            "initialization_parameter_tree_sha256": initialization,
        },
    )
    gradient_path = tmp_path / "gradient.json"
    gradient = _write_envelope(
        gradient_path,
        rung.GRADIENT_SCHEMA_VERSION,
        {
            "checkpoint_parameter_tree_sha256": checkpoint,
            "checks": dict.fromkeys(rung.GRADIENT_CHECKS, True),
            "initialization_parameter_tree_sha256": initialization,
        },
    )
    measurements = tmp_path / "mechanism.npz"
    retention = np.full(4, 0.99**16, dtype=np.float32)
    np.savez(
        measurements,
        injected_token_rms=np.full(4, 0.3, dtype=np.float32),
        production_relative_commit_residual=np.full(4, 0.005, dtype=np.float32),
        raw_read_rms=np.full(4, 0.2, dtype=np.float32),
        reachable=np.ones(4, dtype=np.bool_),
        real_noise_injected_to_residual_ratio=np.full(4, 0.05, dtype=np.float32),
        retained_injection_amplitude_p90_delay=np.full(4, 0.7, dtype=np.float32),
        retention_cosine_p90_delay=np.ones(4, dtype=np.float32),
        retention_norm_ratio_p90_delay=retention,
        synthetic_fp32_commit_residual=np.full(4, 1e-6, dtype=np.float32),
    )
    mechanism_path = tmp_path / "mechanism.json"
    _write_envelope(
        mechanism_path,
        rung.MECHANISM_SCHEMA_VERSION,
        {
            "calibration_artifact_id": calibration_id,
            "checkpoint_parameter_tree_sha256": checkpoint,
            "core_artifact": {
                "artifact_id": core["artifact_id"],
                "path": core_path.name,
                "sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
            },
            "gradient_artifact": {
                "artifact_id": gradient["artifact_id"],
                "path": gradient_path.name,
                "sha256": hashlib.sha256(gradient_path.read_bytes()).hexdigest(),
            },
            "initialization_parameter_tree_sha256": initialization,
            "measurements_npz": {
                "path": measurements.name,
                "sha256": hashlib.sha256(measurements.read_bytes()).hexdigest(),
            },
        },
    )
    return mechanism_path, checkpoint, initialization, calibration_id


def test_mechanism_links_core_gradient_and_detects_hash_tamper(tmp_path: Path) -> None:
    path, checkpoint, initialization, calibration_id = _mechanism(tmp_path)
    reduced = rung.reduce_mechanism_artifact(
        path,
        checkpoint_sha256=checkpoint,
        initialization_sha256=initialization,
        calibration_artifact_id=calibration_id,
        calibration_passes=True,
    )
    assert set(reduced["checks"]) == set(pilot.CORE_CHECK_NAMES)
    assert all(reduced["checks"].values())
    assert reduced["metrics"]["production_relative_commit_residual_p95"] == pytest.approx(0.005)

    (tmp_path / "core.json").write_bytes((tmp_path / "core.json").read_bytes() + b" ")
    with pytest.raises(artifacts.GateArtifactError, match="SHA256 mismatch"):
        rung.reduce_mechanism_artifact(
            path,
            checkpoint_sha256=checkpoint,
            initialization_sha256=initialization,
            calibration_artifact_id=calibration_id,
            calibration_passes=True,
        )


def _task_health(tmp_path: Path, *, paired: bool = True) -> tuple[Path, str, str, str]:
    checkpoint = "6" * 64
    identity = "7" * 64
    identity_file_sha = "8" * 64
    data_state_path = tmp_path / "v35_data_iterator_state.json"
    data_state_path.write_bytes(artifacts.canonical_json_bytes({"sampler": 0}) + b"\n")
    telemetry = {
        "schema_version": 1,
        "accepted_update_count": 0,
        "finite_accepted_update_count": 0,
        "pre_shared_severe_clip_count": 0,
        "pre_shared_update_count": 0,
        "write_feature_cap_bind_numerator": 0,
        "write_feature_cap_bind_denominator": 0,
        "read_feature_cap_bind_numerator": 0,
        "read_feature_cap_bind_denominator": 0,
        "pre_shared_grad_norm_max": 0.0,
    }
    telemetry_path = tmp_path / "v35_cumulative_telemetry.json"
    telemetry_path.write_bytes(artifacts.canonical_json_bytes(telemetry) + b"\n")
    unsigned_runtime = {
        "format_version": 1,
        "completed_updates": 0,
        "run_initialization_identity_sha256": identity_file_sha,
        "data_iterator_state_file": data_state_path.name,
        "data_iterator_state_sha256": hashlib.sha256(data_state_path.read_bytes()).hexdigest(),
        "cumulative_telemetry_file": telemetry_path.name,
        "cumulative_telemetry_sha256": hashlib.sha256(telemetry_path.read_bytes()).hexdigest(),
    }
    runtime = dict(unsigned_runtime)
    runtime["identity_sha256"] = artifacts.sha256_bytes(artifacts.canonical_json_bytes(unsigned_runtime))
    runtime_path = tmp_path / "v35_runtime_identity.json"
    runtime_path.write_bytes(artifacts.canonical_json_bytes(runtime) + b"\n")

    ids = np.asarray(["train/00", "train/01"])
    frames = np.asarray([15, 30], dtype=np.int32)
    seeds = np.asarray([10, 11], dtype=np.uint32)
    times = np.asarray([0.25, 0.75], dtype=np.float32)
    noise = np.asarray(["9" * 64, "a" * 64])
    suite_records = [{"frame_index": int(frames[i]), "stable_id": str(ids[i])} for i in range(len(ids))]
    rng_records = [
        {
            "action_noise_sha256": str(noise[i]),
            "flow_time": float(times[i]),
            "frame_index": int(frames[i]),
            "rng_seed": int(seeds[i]),
            "stable_id": str(ids[i]),
        }
        for i in range(len(ids))
    ]
    raw_path = tmp_path / "task_health.npz"
    step0_flow = np.asarray([1.01, 1.03], dtype=np.float32)
    np.savez(
        raw_path,
        stable_id=ids,
        frame_index=frames,
        rng_seed=seeds,
        flow_time=times,
        action_noise_sha256=noise,
        fresh_source_flow_loss=np.asarray([1.0, 1.0], dtype=np.float32),
        fresh_source_subtask_ce=np.asarray([1.0, 1.0], dtype=np.float32),
        v35_step0_flow_loss=step0_flow,
        v35_step0_subtask_ce=np.asarray([1.01, 1.01], dtype=np.float32),
        rung_flow_loss=step0_flow if paired else step0_flow + np.float32(0.1),
        rung_subtask_ce=np.asarray([1.01, 1.01], dtype=np.float32),
        gradient_finite=np.ones(2, dtype=np.bool_),
        parameter_finite=np.ones(2, dtype=np.bool_),
        memory_state_finite=np.ones(2, dtype=np.bool_),
    )

    def descriptor(path: Path) -> dict[str, str]:
        return {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    path = tmp_path / "task_health.json"
    _write_envelope(
        path,
        rung.TASK_HEALTH_SCHEMA_VERSION,
        {
            "checkpoint_parameter_tree_sha256": checkpoint,
            "completed_updates": 0,
            "cumulative_telemetry": descriptor(telemetry_path),
            "data_iterator_state": descriptor(data_state_path),
            "initialization_identity_sha256": identity,
            "no_augmentation_suite_sha256": artifacts.sha256_bytes(artifacts.canonical_json_bytes(suite_records)),
            "preprocessing_norm_sha256": "b" * 64,
            "raw_npz": descriptor(raw_path),
            "rng_inputs_sha256": artifacts.sha256_bytes(artifacts.canonical_json_bytes(rng_records)),
            "runtime_identity": descriptor(runtime_path),
        },
    )
    return path, checkpoint, identity, identity_file_sha


def test_task_health_is_paired_and_runtime_authenticated(tmp_path: Path) -> None:
    path, checkpoint, identity, identity_file_sha = _task_health(tmp_path)
    reduced = rung.reduce_task_health_artifact(
        path,
        completed_updates=0,
        checkpoint_sha256=checkpoint,
        initialization_identity_sha256=identity,
        initialization_manifest_file_sha256=identity_file_sha,
        allowed_stable_ids={"train/00", "train/01"},
    )
    assert reduced["rung"] == reduced["v35_step0"]
    assert reduced["severe_clip"]["total_optimizer_steps"] == 0
    assert all(reduced["finiteness"].values())

    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    bad_path, checkpoint, identity, identity_file_sha = _task_health(bad_dir, paired=False)
    with pytest.raises(rung.RungEvaluationError, match="byte-identical"):
        rung.reduce_task_health_artifact(
            bad_path,
            completed_updates=0,
            checkpoint_sha256=checkpoint,
            initialization_identity_sha256=identity,
            initialization_manifest_file_sha256=identity_file_sha,
            allowed_stable_ids={"train/00", "train/01"},
        )


def test_protocol_sha_is_fixed_and_raw_envelope_refuses_overwrite(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    raw = tmp_path / "raw.npz"
    np.savez(raw, **_condition_arrays(manifest))
    mechanism, checkpoint, initialization, calibration_id = _mechanism(tmp_path)
    health_dir = tmp_path / "health"
    health_dir.mkdir()
    health, _, identity, _ = _task_health(health_dir)
    # emit_raw_envelope deliberately requires a single relative artifact directory, preventing
    # path topology from becoming an unbound input.
    with pytest.raises(rung.RungEvaluationError, match="share the output"):
        rung.emit_raw_envelope(
            output_path=tmp_path / "raw.json",
            raw_npz=raw,
            mechanism_path=mechanism,
            selection_path=tmp_path / "selection.json",
            task_health_path=health,
            manifest=manifest,
            completed_updates=0,
            checkpoint_parameter_tree_sha256=checkpoint,
            initialization_parameter_tree_sha256=initialization,
            initialization_identity_sha256=identity,
            calibration_artifact_id=calibration_id,
            prototype_artifact_id=f"sha256:{'c' * 64}",
        )
    assert artifacts.sha256_bytes(artifacts.canonical_json_bytes(rung.EVALUATION_PROTOCOL)) == (
        rung.EVALUATION_PROTOCOL_SHA256
    )
