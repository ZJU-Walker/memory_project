import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v35_gate_artifacts as artifacts
    import v35_pilot_gate as pilot
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


def _manifest(tmp_path: Path) -> tuple[artifacts.FrozenManifest, Path, str]:
    records = []
    for part, repeats in (("part1", (4, 4, 4, 4)), ("part2", (4, 3, 4, 3))):
        demo = 1
        for (object_name, side), count in zip(
            (("banana", "left"), ("banana", "right"), ("grey_pepper_box", "left"), ("grey_pepper_box", "right")),
            repeats,
            strict=True,
        ):
            for _ in range(count):
                records.append(
                    {
                        "stable_id": f"0830_bin_{part}/demo{demo}",
                        "episode_index": len(records),
                        "collection": "0830",
                        "object": object_name,
                        "part": part,
                        "target_side": side,
                        "split": "train",
                        "include": True,
                    }
                )
                demo += 1
    demo = 1
    for (object_name, side), count in zip(
        (("banana", "left"), ("banana", "right"), ("grey_pepper_box", "left"), ("grey_pepper_box", "right")),
        (10, 10, 10, 10),
        strict=True,
    ):
        for _ in range(count):
            records.append(
                {
                    "stable_id": f"0831_bin/demo{demo}",
                    "episode_index": len(records),
                    "collection": "0831",
                    "object": object_name,
                    "part": "",
                    "target_side": side,
                    "split": "train",
                    "include": True,
                }
            )
            demo += 1
    expected = artifacts._expected_frozen_splits(records)  # noqa: SLF001
    for record in records:
        record["split"] = expected[record["stable_id"]]
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "dataset_version": "v36",
                "episodes": records,
                "review_status": "frozen",
                "schema_version": 2,
                "split_algorithm": artifacts.V35_SPLIT_ALGORITHM,
                "split_algorithm_sha256": artifacts.V35_SPLIT_ALGORITHM_SHA256,
                "split_seed": 36,
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return artifacts.load_frozen_manifest(path, expected_sha256=digest), path, digest


def _checkpoint_sha(step: int) -> str:
    return "a" * 64 if step == 0 else f"{step + 1:064x}"


def _optimizer_sha(step: int) -> str:
    return hashlib.sha256(f"optimizer:{step}".encode()).hexdigest()


def _data_state_sha(step: int) -> str:
    return hashlib.sha256(f"data:{step}".encode()).hexdigest()


def _write_calibration(tmp_path: Path, manifest: artifacts.FrozenManifest) -> tuple[Path, dict]:
    path = tmp_path / "calibration.json"
    if path.exists():
        return path, json.loads(path.read_text())
    payload = {
        "schema_version": "openpi.v35.injection-calibration.v1",
        "status": "pass",
        "gates": {"passes": True},
        "gate": {
            "effective_tanh_gate_max": 0.5,
            "effective_tanh_gate_min": 0.5,
            "open_channel_count": 2,
            "raw_w_sha256": "6" * 64,
            "target_effective_tanh_gate": 0.5,
        },
        "parameters": {
            "alpha_step": 0.01,
            "memory_injection_c": 0.8,
            "memory_injection_tau": 1.2,
            "prototype_injected_rms_target": 0.3,
        },
        "population": {
            "channel_count": 2,
            "episode_count": 54,
            "split": "train",
            "stable_ids": [episode.stable_id for episode in manifest.split("train")],
        },
        "provenance": {
            "collector_source_sha256": "1" * 64,
            "dataset_sha256": "2" * 64,
            "official_base_source_sha256": "b" * 64,
            "preflight_sha256": "3" * 64,
            "replay_protocol_sha256": "4" * 64,
            "source_sha256": "a" * 64,
            "split_sha256": manifest.sha256,
        },
        "statistics": {"p90_delay": {"n_delay": 16, "retention_factor": 0.99**16}},
    }
    digest = hashlib.sha256(artifacts.canonical_json_bytes(payload)).hexdigest()
    calibration = {
        "artifact_sha256": digest,
        "calibration_id": f"sha256:{digest}",
        "hash_scope": "SHA256 of canonical_json($.payload)",
        "payload": payload,
    }
    path.write_bytes(artifacts.canonical_json_bytes(calibration) + b"\n")
    return path, calibration


def _write_initialization(
    tmp_path: Path,
    manifest: artifacts.FrozenManifest,
    *,
    calibration_path: Path,
    calibration: dict,
) -> Path:
    path = tmp_path / "initialization_manifest.json"
    if path.exists():
        return path
    payload = {
        "actual_step0_parameter_tree_sha256": "a" * 64,
        "artifact_hashes": {
            "calibration_artifact_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
            "episode_manifest_sha256": manifest.sha256,
            "norm_stats_sha256": "7" * 64,
        },
        "calibration_id": calibration["calibration_id"],
        "config_name": "pi05_yam_mem_v35",
        "format_version": 2,
        "initialization_seed": 35,
        "memory_calibration": {
            "alpha_step": 0.01,
            "memory_injection_c": 0.8,
            "memory_injection_tau": 1.2,
        },
        "memory_inject_w_sha256": "6" * 64,
        "official_source_uri": "gs://openpi-assets/checkpoints/pi05_base/params",
        "source_tree_sha256": "b" * 64,
        "step0_checkpoint": 0,
    }
    payload["identity_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_prototype(
    tmp_path: Path,
    manifest: artifacts.FrozenManifest,
    *,
    step: int,
    calibration_id: str,
) -> tuple[Path, dict]:
    train = manifest.split("train")
    ids = [episode.stable_id for episode in train]
    sides = np.asarray([episode.target_side for episode in train], dtype=np.int64)
    means = np.stack(
        [
            np.asarray((1.0 + index * 0.001, -0.3 if side == 0 else 0.4), dtype=np.float32)
            for index, side in enumerate(sides)
        ]
    )
    directions = []
    for side in (0, 1):
        direction = np.mean(means[sides == side], axis=0, dtype=np.float32)
        direction = np.asarray(direction / np.float32(np.linalg.norm(direction.astype(np.float64))), dtype=np.float32)
        directions.append(direction)
    npz_path = tmp_path / f"prototype_{step}.npz"
    np.savez(
        npz_path,
        episode_stable_id=np.asarray(ids),
        episode_mean_vbar=means,
        left_direction=directions[0],
        right_direction=directions[1],
    )
    source_path = tmp_path / f"prototype_source_{step}.npz"
    source_path.write_bytes(f"raw-source-{step}".encode())
    payload = {
        "calibration_artifact_id": calibration_id,
        "checkpoint_parameter_tree_sha256": _checkpoint_sha(step),
        "construction_protocol": pilot.PROTOTYPE_CONSTRUCTION,
        "directions_npz": {
            "path": npz_path.name,
            "sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        },
        "episode_manifest_sha256": manifest.sha256,
        "prototype_injected_rms_target": 0.3,
        "source_evidence_npz": {
            "path": source_path.name,
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "split_assignment_sha256": manifest.split_assignment_sha256,
        "training_stable_ids": ids,
    }
    envelope = artifacts.artifact_envelope(pilot.PROTOTYPE_SCHEMA_VERSION, payload)
    path = tmp_path / f"prototype_{step}.json"
    artifacts.write_canonical_envelope(path, envelope, schema_version=pilot.PROTOTYPE_SCHEMA_VERSION)
    return path, envelope


def _base_counts(value: int = 7) -> dict[str, int]:
    prototype_counts = {
        5: (5, 5, 6),
        6: (6, 6, 6),
        7: (7, 8, 7),
    }[value]
    return {
        "writer_natural_correct": value,
        "writer_counterfactual_correct": value,
        "direct_carry_correct": value,
        "read_natural_correct": value,
        "read_reset_correct": 4,
        "read_opposite_donor_followed": value,
        "action_natural_correct": value,
        "action_reset_correct": 4,
        "action_opposite_donor_followed": value,
        "zero_injection_target_correct": 3,
        "zero_vs_reset_prediction_differs": 1,
        "prototype_correct_side_succeeds": prototype_counts[0],
        "prototype_opposite_donor_followed": prototype_counts[1],
        "prototype_paired_action_flipped": prototype_counts[2],
    }


def _episodes(
    manifest: artifacts.FrozenManifest,
    counts: dict[str, int],
    *,
    foreign_final_id: bool = False,
) -> list[dict]:
    prototype_vectors = None
    for correct in itertools.product((False, True), repeat=8):
        if sum(correct) != counts["prototype_correct_side_succeeds"]:
            continue
        for donor in itertools.product((False, True), repeat=8):
            if (
                sum(donor) == counts["prototype_opposite_donor_followed"]
                and sum(left == right for left, right in zip(correct, donor, strict=True))
                == counts["prototype_paired_action_flipped"]
            ):
                prototype_vectors = (correct, donor)
                break
        if prototype_vectors is not None:
            break
    if prototype_vectors is None:
        raise AssertionError("requested prototype counts are not binary-verdict feasible")
    records = []
    dev = manifest.split("development")
    for index, episode in enumerate(dev):
        record = {
            "attention_memory_mass_natural": 0.2,
            "attention_memory_mass_reset": 0.1,
            "attention_memory_mass_zero": 0.1,
            "attention_uniform_baseline": 0.05,
            "stable_id": episode.stable_id,
            "split": "development",
            "zero_minus_reset_action_score": 0.01 * index,
        }
        record.update({field: index < counts[field] for field in pilot.EPISODE_BOOLEAN_FIELDS})
        record["prototype_correct_side_succeeds"] = prototype_vectors[0][index]
        record["prototype_opposite_donor_followed"] = prototype_vectors[1][index]
        record["prototype_paired_action_flipped"] = prototype_vectors[0][index] == prototype_vectors[1][index]
        record["zero_injection_target_correct"] = 0 < index <= counts["zero_injection_target_correct"]
        record["zero_vs_reset_prediction_differs"] = index < counts["zero_vs_reset_prediction_differs"]
        target = episode.target_side
        opposite = 1 - target
        record.update(
            {
                "action_natural_predicted_side": target if record["action_natural_correct"] else opposite,
                "action_opposite_donor_predicted_side": (
                    opposite if record["action_opposite_donor_followed"] else target
                ),
                "action_reset_predicted_side": target if record["action_reset_correct"] else opposite,
                "direct_carry_predicted_side": target if record["direct_carry_correct"] else opposite,
                "eligible_d_frame_count": 2,
                "eligible_e_frame_count": 2,
                "prototype_correct_side_predicted_side": (
                    target if record["prototype_correct_side_succeeds"] else opposite
                ),
                "prototype_opposite_donor_predicted_side": (
                    opposite if record["prototype_opposite_donor_followed"] else target
                ),
                "read_natural_predicted_side": target if record["read_natural_correct"] else opposite,
                "read_opposite_donor_predicted_side": (opposite if record["read_opposite_donor_followed"] else target),
                "read_reset_predicted_side": target if record["read_reset_correct"] else opposite,
                "use_pressure_frame_count": 1,
                "writer_counterfactual_predicted_side": (
                    opposite if record["writer_counterfactual_correct"] else target
                ),
                "writer_natural_predicted_side": target if record["writer_natural_correct"] else opposite,
                "zero_injection_predicted_side": (target if record["zero_injection_target_correct"] else opposite),
            }
        )
        records.append(record)
    if foreign_final_id:
        records[-1]["stable_id"] = manifest.split("final_test")[0].stable_id
    return records


def _train_results(
    manifest: artifacts.FrozenManifest,
    *,
    protocol: str,
    counts: dict[str, int],
) -> dict:
    episodes = [
        {
            "stable_id": episode.stable_id,
            **{field: index < count for field, count in counts.items()},
        }
        for index, episode in enumerate(manifest.split("train"))
    ]
    return {
        "artifact_sha256": hashlib.sha256(artifacts.canonical_json_bytes(episodes)).hexdigest(),
        "episode_count": 54,
        "episodes": episodes,
        "protocol": protocol,
    }


def _write_rung(
    tmp_path: Path,
    manifest: artifacts.FrozenManifest,
    *,
    step: int,
    counts: dict[str, int] | None = None,
    writer_train_count: int = 67,
    foreign_final_id: bool = False,
    bad_core: bool = False,
    severe_steps: int = 0,
    extension_authorization: str | None = None,
) -> Path:
    calibration_path, calibration = _write_calibration(tmp_path, manifest)
    calibration_id = calibration["calibration_id"]
    initialization_path = _write_initialization(
        tmp_path,
        manifest,
        calibration_path=calibration_path,
        calibration=calibration,
    )
    prototype_path, prototype = _write_prototype(
        tmp_path,
        manifest,
        step=step,
        calibration_id=calibration_id,
    )
    gate_checks = dict.fromkeys(pilot.CORE_CHECK_NAMES, True)
    gate_metrics = {
        "injected_token_rms": 0.3,
        "production_relative_commit_residual_p95": 0.005,
        "raw_read_rms": 0.2,
        "reachable_fraction": 0.8,
        "real_noise_injected_to_residual_ratio_p95": 0.05,
        "retained_injection_amplitude_p90_delay": 0.7,
        "retention_cosine_p90_delay": 1.0,
        "retention_norm_ratio_p90_delay": 0.99**16,
        "retention_norm_ratio_relative_error_p90_delay": 0.0,
        "synthetic_fp32_commit_residual_max": 1e-6,
    }
    if bad_core:
        gate_metrics["production_relative_commit_residual_p95"] = 0.02
    counts = _base_counts() if counts is None else counts
    payload = {
        "calibration_artifact": {
            "calibration_id": calibration_id,
            "path": calibration_path.name,
            "sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        },
        "checkpoint": {
            "completed_updates": step,
            "parameter_tree_sha256": _checkpoint_sha(step),
        },
        "episode_manifest_sha256": manifest.sha256,
        "episodes": _episodes(manifest, counts, foreign_final_id=foreign_final_id),
        "evaluation": {
            "episode_aggregation": pilot.EPISODE_AGGREGATION,
            "final_test_accessed": False,
            "split": "development",
        },
        "evaluation_protocol_sha256": "e" * 64,
        "gate_c": {"checks": gate_checks, "metrics": gate_metrics},
        "initialization_manifest": {
            "path": initialization_path.name,
            "sha256": hashlib.sha256(initialization_path.read_bytes()).hexdigest(),
        },
        "initialization_parameter_tree_sha256": "a" * 64,
        "prototype_artifact": {
            "artifact_id": prototype["artifact_id"],
            "path": prototype_path.name,
            "sha256": hashlib.sha256(prototype_path.read_bytes()).hexdigest(),
        },
        "run_provenance": {
            "cumulative_telemetry_sha256": f"{step + 11:064x}",
            "data_iterator_state_sha256": _data_state_sha(step),
            "extension_authorization_artifact_id": extension_authorization,
            "optimizer_state_sha256": _optimizer_sha(step),
            "previous_rung": (
                None
                if step == 0
                else {
                    "completed_updates": {250: 0, 500: 250, 1_000: 500, 2_500: 1_000}[step],
                    "data_iterator_state_sha256": _data_state_sha({250: 0, 500: 250, 1_000: 500, 2_500: 1_000}[step]),
                    "optimizer_state_sha256": _optimizer_sha({250: 0, 500: 250, 1_000: 500, 2_500: 1_000}[step]),
                    "parameter_tree_sha256": _checkpoint_sha({250: 0, 500: 250, 1_000: 500, 2_500: 1_000}[step]),
                }
            ),
            "run_id_sha256": "d" * 64,
            "runtime_identity_sha256": f"{step + 12:064x}",
            "training_config_sha256": "e" * 64,
        },
        "split_assignment_sha256": manifest.split_assignment_sha256,
        "task_health": {
            "feature_cap": {
                "bound_terms": 0,
                "cap_value": 1.0,
                "definition": "unweighted_per_term_feature_cotangent_before_episode_cell_and_loss_weight",
                "eligible_terms": 0 if step == 0 else 100,
            },
            "finiteness": {
                "gradients": True,
                "losses": True,
                "memory_state": True,
                "parameters": True,
            },
            "fresh_source_reference": {"flow_loss": 1.0, "subtask_ce": 1.0},
            "no_augmentation_suite_sha256": "1" * 64,
            "preprocessing_norm_sha256": "7" * 64,
            "rng_inputs_sha256": "3" * 64,
            "rung": {"flow_loss": 1.0, "subtask_ce": 1.0},
            "severe_clip": {
                "definition": "pre_shared_global_grad_norm_gt_10x_optimizer_clip_threshold",
                "optimizer_clip_threshold": 1.0,
                "severe_steps": severe_steps,
                "total_optimizer_steps": step,
            },
            "v35_step0": {"flow_loss": 1.0, "subtask_ce": 1.0},
        },
        "train_prototype_loo": _train_results(
            manifest,
            protocol=pilot.TRAIN_PROTOTYPE_PROTOCOL,
            counts={"correct_side": 70, "opposite_donor_follow": 70},
        ),
        "train_writer_oof": _train_results(
            manifest,
            protocol=pilot.TRAIN_WRITER_PROTOCOL,
            counts={
                "counterfactual_prompt_correct": writer_train_count,
                "natural_prompt_correct": writer_train_count,
            },
        ),
    }
    envelope = artifacts.artifact_envelope(pilot.RUNG_SCHEMA_VERSION, payload)
    path = tmp_path / f"rung_{step}.json"
    artifacts.write_canonical_envelope(path, envelope, schema_version=pilot.RUNG_SCHEMA_VERSION)
    return path


def _load_rungs(paths: list[Path], manifest: artifacts.FrozenManifest) -> list[pilot.RungResult]:
    return [pilot.load_rung_result(path, manifest=manifest) for path in paths]


def test_pass_outcome_is_independent_of_prompt_bound_writer_claim(tmp_path: Path) -> None:
    manifest, _, _ = _manifest(tmp_path)
    paths = []
    for step in pilot.FIXED_1K_RUNGS:
        counts = _base_counts(5 if step == 0 else 7)
        paths.append(
            _write_rung(
                tmp_path,
                manifest,
                step=step,
                counts=counts,
                writer_train_count=44 if step == 1_000 else 49,
            )
        )

    decision = pilot.evaluate_gate_d(_load_rungs(paths, manifest), manifest=manifest, endpoint=1_000)
    payload = decision["payload"]

    assert artifacts.verify_envelope(decision, schema_version=pilot.DECISION_SCHEMA_VERSION)
    assert payload["outcome"] == "pass"
    assert payload["action"] == "continue_to_fixed_10000_budget"
    assert payload["writer_claim"]["status"] == "not_supported"
    assert payload["writer_claim"]["gates_only_writer_claim_not_natural_chain"]
    assert payload["final_test_remains_sealed"]
    assert [item["completed_updates"] for item in payload["provenance"]["rung_artifacts"]] == [0, 250, 500, 1_000]


def test_1k_inconclusive_requires_counts_and_prototype_improvement(tmp_path: Path) -> None:
    manifest, _, _ = _manifest(tmp_path)
    paths = []
    for step in pilot.FIXED_1K_RUNGS:
        counts = _base_counts(5 if step == 0 else 7)
        if step == 1_000:
            counts["read_natural_correct"] = 6
            counts["prototype_correct_side_succeeds"] = 6
            counts["prototype_opposite_donor_followed"] = 7
            counts["prototype_paired_action_flipped"] = 7
        paths.append(_write_rung(tmp_path, manifest, step=step, counts=counts))

    decision = pilot.evaluate_gate_d(_load_rungs(paths, manifest), manifest=manifest, endpoint=1_000)

    assert decision["payload"]["outcome"] == "inconclusive"
    assert decision["payload"]["action"] == "extend_same_run_once_to_2500"
    assert decision["payload"]["prototype_oracles"]["improvement_from_step0"] == {
        "prototype_correct_side_succeeds": 1,
        "prototype_opposite_donor_followed": 2,
        "prototype_paired_action_flipped": 1,
    }


def test_2500_is_one_extension_only_and_must_pass_or_stop(tmp_path: Path) -> None:
    manifest, _, _ = _manifest(tmp_path)
    paths = []
    for step in pilot.FIXED_1K_RUNGS:
        counts = _base_counts(5 if step == 0 else 7)
        if step == 1_000:
            counts["read_natural_correct"] = 6
            counts["prototype_correct_side_succeeds"] = 6
            counts["prototype_opposite_donor_followed"] = 7
            counts["prototype_paired_action_flipped"] = 7
        paths.append(_write_rung(tmp_path, manifest, step=step, counts=counts))
    loaded = _load_rungs(paths, manifest)
    prior = pilot.evaluate_gate_d(loaded, manifest=manifest, endpoint=1_000)
    prior_path = tmp_path / "prior.json"
    artifacts.write_canonical_envelope(prior_path, prior, schema_version=pilot.DECISION_SCHEMA_VERSION)
    extension_counts = _base_counts(7)
    extension_counts["read_natural_correct"] = 6
    extension_counts["prototype_correct_side_succeeds"] = 6
    extension_counts["prototype_opposite_donor_followed"] = 7
    extension_counts["prototype_paired_action_flipped"] = 7
    extension_path = _write_rung(
        tmp_path,
        manifest,
        step=2_500,
        counts=extension_counts,
        extension_authorization=prior["artifact_id"],
    )
    loaded.append(pilot.load_rung_result(extension_path, manifest=manifest))

    decision = pilot.evaluate_gate_d(
        loaded,
        manifest=manifest,
        endpoint=2_500,
        prior_decision_path=prior_path,
    )

    assert decision["payload"]["outcome"] == "fail"
    assert decision["payload"]["action"] == "stop_branch_no_second_extension"
    assert "one_time_2500_extension_did_not_pass" in decision["payload"]["failure_reasons"]
    with pytest.raises(pilot.PilotGateError, match="requires the prior"):
        pilot.evaluate_gate_d(loaded, manifest=manifest, endpoint=2_500)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("reset", "reset_or_zero_control_failed"),
        ("prototype_no_improvement", "prototype_6_of_8_without_each_count_improving"),
        ("core", "gate_c_failed"),
        ("clip", "task_health_failed"),
    ],
)
def test_fail_conditions_are_not_reclassified_as_inconclusive(tmp_path: Path, mutation: str, expected: str) -> None:
    manifest, _, _ = _manifest(tmp_path)
    paths = []
    for step in pilot.FIXED_1K_RUNGS:
        step0_value = 6 if mutation == "prototype_no_improvement" else 5
        counts = _base_counts(step0_value if step == 0 else 7)
        if step == 1_000:
            if mutation == "reset":
                counts["read_reset_correct"] = 5
            elif mutation == "prototype_no_improvement":
                for field in (
                    "prototype_correct_side_succeeds",
                    "prototype_opposite_donor_followed",
                    "prototype_paired_action_flipped",
                ):
                    counts[field] = 6
        paths.append(
            _write_rung(
                tmp_path,
                manifest,
                step=step,
                counts=counts,
                bad_core=mutation == "core" and step == 500,
                severe_steps=20 if mutation == "clip" and step == 1_000 else 0,
            )
        )

    decision = pilot.evaluate_gate_d(_load_rungs(paths, manifest), manifest=manifest, endpoint=1_000)

    assert decision["payload"]["outcome"] == "fail"
    assert any(expected in reason for reason in decision["payload"]["failure_reasons"])


def test_rung_loader_rejects_final_test_id_and_tampered_prototype(tmp_path: Path) -> None:
    manifest, _, _ = _manifest(tmp_path)
    foreign = _write_rung(tmp_path, manifest, step=0, foreign_final_id=True)
    with pytest.raises(pilot.PilotGateError, match="final-test"):
        pilot.load_rung_result(foreign, manifest=manifest)

    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    clean_manifest, _, _ = _manifest(clean_dir)
    rung_path = _write_rung(clean_dir, clean_manifest, step=0)
    prototype_path = clean_dir / "prototype_0.json"
    with prototype_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(artifacts.GateArtifactError, match="SHA256 mismatch"):
        pilot.load_rung_result(rung_path, manifest=clean_manifest)


def test_missing_fixed_rung_and_noncanonical_result_fail_closed(tmp_path: Path) -> None:
    manifest, _, _ = _manifest(tmp_path)
    paths = [_write_rung(tmp_path, manifest, step=step) for step in (0, 250, 1_000)]
    with pytest.raises(pilot.PilotGateError, match="exact rung set"):
        pilot.evaluate_gate_d(_load_rungs(paths, manifest), manifest=manifest, endpoint=1_000)

    path = paths[0]
    value = json.loads(path.read_text())
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    with pytest.raises(artifacts.GateArtifactError, match="canonical JSON"):
        pilot.load_rung_result(path, manifest=manifest)


@pytest.mark.parametrize("bad_path", [Path("../foreign.json"), Path("/iris/u/user/foreign.json")])
def test_production_cli_paths_reject_unconfined_paths(bad_path: Path) -> None:
    with pytest.raises(pilot.PilotGateError, match="memory_project"):
        pilot._resolve_project_cli_path(bad_path, name="rung-result path")  # noqa: SLF001
