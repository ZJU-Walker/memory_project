"""Fail-closed contracts for the manifest-driven v3.5 raw-episode evaluator."""

# ruff: noqa: SLF001

import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import eval_yam_mem_subtask_raw as evaluator
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


APPROACH = "open both lids"
EVIDENCE = "inspect both bins"
OCCLUSION = "close both lids and reset arms"
DECISION = "wait; target bin is left"
EXECUTE = "open left bin"
DECISION_RIGHT = "wait; target bin is right"
EXECUTE_RIGHT = "open right bin"
TASK_VOCABULARY = (
    APPROACH,
    DECISION,
    EXECUTE,
    OCCLUSION,
    EVIDENCE,
    EXECUTE_RIGHT,
    DECISION_RIGHT,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_protocol(tmp_path: Path, **record_overrides):
    stable_id = "0831_bin/demo7"
    demo = tmp_path / stable_id
    demo.mkdir(parents=True)
    labels = [
        {"task": APPROACH, "start": 0, "end": 14},
        {"task": EVIDENCE, "start": 15, "end": 49},
        {"task": OCCLUSION, "start": 50, "end": 64},
        {"task": DECISION, "start": 65, "end": 79},
        {"task": EXECUTE, "start": 80, "end": 94},
    ]
    label_path = demo / "labels.json"
    label_path.write_text(json.dumps(labels))
    records = []
    cells = [(object_name, side) for object_name in evaluator.V35_OBJECT_PROMPTS for side in ("left", "right")]
    for part, demo_indices in (("part1", range(1, 17)), ("part2", (*range(1, 14), 15))):
        for offset, demo_index in enumerate(demo_indices):
            object_name, side = cells[offset % len(cells)]
            records.append(
                {
                    "stable_id": f"0830_bin_{part}/demo{demo_index}",
                    "raw_dir": f"0830_bin_{part}/demo{demo_index}",
                    "collection": "0830",
                    "part": part,
                    "object": object_name,
                    "prompt": evaluator.V35_OBJECT_PROMPTS[object_name],
                    "target_side": side,
                    "include": True,
                }
            )
    demo_index = 1
    for object_name in evaluator.V35_OBJECT_PROMPTS:
        for side in ("left", "right"):
            for _ in range(10):
                records.append(
                    {
                        "stable_id": f"0831_bin/demo{demo_index}",
                        "raw_dir": f"0831_bin/demo{demo_index}",
                        "collection": "0831",
                        "part": "",
                        "object": object_name,
                        "prompt": evaluator.V35_OBJECT_PROMPTS[object_name],
                        "target_side": side,
                        "include": True,
                    }
                )
                demo_index += 1
    splits = evaluator._v35_expected_frozen_splits(records, seed=36)
    for candidate in records:
        candidate["split"] = splits[candidate["stable_id"]]

    record = next(candidate for candidate in records if candidate["stable_id"] == stable_id)
    record.update(
        {
            "target_side": "left",
            "label_file": "labels.json",
            "label_sha256": _digest(label_path),
            "expected_num_frames": 95,
            "e_visibility": {
                "manual_reviewed": True,
                "both_objects_visible": True,
                "first_valid_visible_frame": 15,
                "last_clean_visible_frame": 44,
                "contact_sheet_sha256": "a" * 64,
            },
            "d_valid": {
                "start": 65,
                "end": 79,
                "state_dim": 14,
                "detector": evaluator.V35_D_VALID_DETECTOR,
            },
        }
    )
    record.update(record_overrides)
    records.append(
        {
            "stable_id": "0830_bin_part2/demo14",
            "raw_dir": "0830_bin_part2/demo14",
            "include": False,
            "exclude_reason": "no terminal execute phase",
        }
    )
    audit_path = tmp_path / "block_confound_audit.json"
    audit_path.write_text(json.dumps({"status": "pass", "manifest_fields_only": True}))
    manifest = {
        "schema_version": evaluator.V35_MANIFEST_SCHEMA_VERSION,
        "dataset_version": "v36",
        "review_status": "frozen",
        "split_seed": 36,
        "split_algorithm": evaluator.V35_SPLIT_ALGORITHM,
        "split_algorithm_sha256": evaluator.V35_SPLIT_ALGORITHM_SHA256,
        "raw_root": ".",
        "task_vocabulary": list(TASK_VOCABULARY),
        "block_confound_audit": {
            "status": "pass",
            "manifest_fields_only": True,
            "report_file": audit_path.name,
            "report_sha256": _digest(audit_path),
        },
        "episodes": records,
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return demo, label_path, manifest_path


def _load(demo: Path, manifest: Path, **overrides):
    values = {
        "manifest_path": manifest,
        "manifest_sha256": _digest(manifest),
        "raw_demo": demo,
        "expected_split_seed": 36,
        "subtask_vocabulary": TASK_VOCABULARY,
        "evidence_subtasks": (EVIDENCE,),
        "occlusion_subtasks": (OCCLUSION,),
        "decision_subtasks": (DECISION,),
        "execute_subtasks": (EXECUTE,),
        "stride": 15,
        "tail_guard": 5,
    }
    values.update(overrides)
    return evaluator._load_v35_episode_protocol(**values)


def _selected_record(manifest: dict) -> dict:
    return next(record for record in manifest["episodes"] if record.get("stable_id") == "0831_bin/demo7")


def _canonical(value, *, ensure_ascii=True) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=ensure_ascii, separators=(",", ":"), sort_keys=True).encode()


def _self_hashed(value: dict, key: str) -> dict:
    result = dict(value)
    result[key] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def _write_checkpoint_protocol(tmp_path: Path, manifest_path: Path):
    cfg = evaluator._config.get_config("pi05_yam_mem_v35")
    ckpt_dir = tmp_path / "checkpoint" / "1000"
    assets = ckpt_dir / "assets"
    assets.mkdir(parents=True)
    embedded_manifest = assets / "v35_episode_manifest.json"
    embedded_manifest.write_bytes(manifest_path.read_bytes())
    manifest_sha = _digest(embedded_manifest)

    raw_gate = np.full((cfg.model.memory.d_value,), np.arctanh(0.5), dtype=np.float32)
    raw_gate_sha = hashlib.sha256(raw_gate.tobytes()).hexdigest()
    graft = _self_hashed(
        {"tree_hashes": {"source_sha256": "2" * 64, "target_schema_sha256": "3" * 64}},
        "manifest_sha256",
    )
    graft_path = assets / "v35_initialization_graft_manifest.json"
    graft_path.write_text(json.dumps(graft))

    manifest = json.loads(manifest_path.read_text())
    train_ids = [
        record["stable_id"]
        for record in manifest["episodes"]
        if record.get("include") is True and record.get("split") == "train"
    ]
    membership = [{"split": "train", "stable_id": stable_id} for stable_id in train_ids]
    dataset_protocol_sha = "4" * 64
    calibration_payload = {
        "schema_version": "openpi.v35.injection-calibration.v1",
        "status": "pass",
        "parameters": {
            "alpha_step": 0.01,
            "memory_injection_c": 3.0,
            "memory_injection_tau": 0.04,
            "prototype_injected_rms_target": 0.3,
        },
        "provenance": {
            "source_sha256": "1" * 64,
            "official_base_source_sha256": "2" * 64,
            "split_sha256": manifest_sha,
            "dataset_sha256": dataset_protocol_sha,
            "observed_membership_sha256": hashlib.sha256(_canonical(membership, ensure_ascii=False)).hexdigest(),
        },
        "population": {"split": "train", "episode_count": 54, "stable_ids": train_ids},
        "gate": {
            "target_effective_tanh_gate": 0.5,
            "open_channel_count": cfg.model.memory.d_value,
            "required_atol": float(2 * np.finfo(np.float32).eps),
            "raw_w_sha256": raw_gate_sha,
        },
        "gates": {
            "passes": True,
            "all_episodes_train": True,
            "fixed_effective_gate_is_0_5": True,
        },
    }
    calibration_digest = hashlib.sha256(_canonical(calibration_payload, ensure_ascii=False)).hexdigest()
    calibration = {
        "artifact_sha256": calibration_digest,
        "calibration_id": f"sha256:{calibration_digest}",
        "payload": calibration_payload,
    }
    calibration_path = assets / "v35_calibration_artifact.json"
    calibration_path.write_text(json.dumps(calibration))

    norm_bytes = b'{"actions":{"mean":[0.0],"std":[1.0]}}\n'
    norm_path = assets / "test_asset" / "norm_stats.json"
    norm_path.parent.mkdir()
    norm_path.write_bytes(norm_bytes)
    train_storage_sha = "5" * 64
    norm_provenance = {
        "status": "complete",
        "manifest": {"sha256": manifest_sha},
        "selection": {"dataset_episode_frame_protocol_sha256": dataset_protocol_sha},
        "norm_stats": {"sha256": hashlib.sha256(norm_bytes).hexdigest()},
        "train_storage": {"sha256": train_storage_sha},
    }
    norm_provenance_path = assets / "v35_norm_stats_provenance.json"
    norm_provenance_path.write_text(json.dumps(norm_provenance))

    identity = _self_hashed(
        {
            "format_version": 2,
            "config_name": cfg.name,
            "official_source_uri": "gs://openpi-assets/checkpoints/pi05_base/params",
            "initialization_seed": cfg.seed,
            "graft_manifest_sha256": graft["manifest_sha256"],
            "actual_step0_parameter_tree_sha256": "1" * 64,
            "calibration_id": calibration["calibration_id"],
            "memory_inject_w_sha256": raw_gate_sha,
            "memory_calibration": {
                "alpha_step": 0.01,
                "memory_injection_c": 3.0,
                "memory_injection_tau": 0.04,
            },
            "artifact_hashes": {
                "calibration_artifact_sha256": _digest(calibration_path),
                "episode_manifest_sha256": manifest_sha,
                "norm_stats_provenance_sha256": _digest(norm_provenance_path),
                "norm_stats_sha256": hashlib.sha256(norm_bytes).hexdigest(),
                "train_storage_sha256": train_storage_sha,
                "initialization_graft_manifest_file_sha256": _digest(graft_path),
                "initialization_graft_manifest_self_sha256": graft["manifest_sha256"],
            },
        },
        "identity_sha256",
    )
    (assets / "v35_initialization_manifest.json").write_text(json.dumps(identity))
    return cfg, ckpt_dir, raw_gate


def test_protocol_is_hash_pinned_split_owned_and_tail_guarded(tmp_path):
    demo, label_path, manifest = _write_protocol(tmp_path)
    protocol = _load(demo, manifest)

    assert protocol.stable_id == "0831_bin/demo7"
    assert protocol.split == "development"
    assert protocol.collection == "0831"
    assert protocol.prompt == "find the banana"
    assert protocol.object_name == "banana"
    assert protocol.manifest_sha256 == _digest(manifest)
    assert protocol.label_sha256 == _digest(label_path)
    assert protocol.expected_num_frames == 95
    # E is [15, 49], tail guard makes the last eligible raw frame 44, clock origin is zero.
    assert protocol.write_frames == (15, 30)
    assert (protocol.d_valid_start, protocol.d_valid_end) == (65, 79)
    assert protocol.phase_by_frame[15] == EVIDENCE
    assert protocol.phase_by_frame[45] == EVIDENCE  # semantic E, correctly ineligible by guard


@pytest.mark.parametrize("stride", [1, 14, 30])
def test_v35_cadence_other_than_exactly_15_fails_closed(tmp_path, stride):
    demo, _, manifest = _write_protocol(tmp_path)
    with pytest.raises(ValueError, match="exactly 15 raw frames"):
        _load(demo, manifest, stride=stride)


def test_mutated_or_unpinned_sidecar_cannot_drive_writes(tmp_path):
    demo, label_path, manifest = _write_protocol(tmp_path)
    label_path.write_text(label_path.read_text() + "\n")
    with pytest.raises(ValueError, match="label SHA-256 mismatch"):
        _load(demo, manifest)

    demo2, _, manifest2 = _write_protocol(tmp_path / "other", label_sha256=None)
    with pytest.raises(ValueError, match="label_sha256 is required"):
        _load(demo2, manifest2)


@pytest.mark.parametrize("split", [None, "dev", "unsealed"])
def test_episode_requires_a_frozen_split(tmp_path, split):
    demo, _, manifest = _write_protocol(tmp_path, split=split)
    with pytest.raises(ValueError, match="do not reproduce the frozen algorithm"):
        _load(demo, manifest)


def test_missing_manifest_hash_or_wrong_demo_fails_closed(tmp_path):
    demo, _, manifest = _write_protocol(tmp_path)
    with pytest.raises(ValueError, match="manifest_sha256 is required"):
        _load(demo, manifest, manifest_sha256=None)
    other_demo = tmp_path / "other_demo"
    other_demo.mkdir()
    with pytest.raises(ValueError, match="match exactly one"):
        _load(other_demo, manifest)


def test_schema2_vocabulary_and_block_confound_audit_are_required_and_hash_pinned(tmp_path):
    demo, _, manifest = _write_protocol(tmp_path)
    raw = json.loads(manifest.read_text())
    raw["schema_version"] = 1
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="schema_version=2"):
        _load(demo, manifest)

    demo2, _, manifest2 = _write_protocol(tmp_path / "vocab")
    raw = json.loads(manifest2.read_text())
    raw["task_vocabulary"] = list(reversed(raw["task_vocabulary"]))
    manifest2.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="task_vocabulary/order"):
        _load(demo2, manifest2)

    demo3, _, manifest3 = _write_protocol(tmp_path / "audit")
    (manifest3.parent / "block_confound_audit.json").write_text("mutated after manifest freeze")
    with pytest.raises(ValueError, match="report bytes"):
        _load(demo3, manifest3)


def test_checkpoint_calibration_is_authenticated_applied_and_gate_verified(tmp_path):
    _, _, manifest = _write_protocol(tmp_path / "protocol")
    cfg, ckpt_dir, raw_gate = _write_checkpoint_protocol(tmp_path, manifest)
    protocol = evaluator._load_v35_checkpoint_protocol(
        ckpt_dir,
        expected_config_name=cfg.name,
        expected_seed=cfg.seed,
        expected_value_width=cfg.model.memory.d_value,
    )
    calibrated = evaluator._apply_v35_checkpoint_calibration(cfg, protocol)
    assert calibrated.model.memory_v35_calibrated
    assert calibrated.model.memory_v35_calibration_id == protocol.calibration_id
    assert calibrated.model.memory_injection_c == 3.0
    assert calibrated.model.memory_injection_tau == 0.04
    assert calibrated.model.memory.alpha_step == 0.01
    evaluator._validate_v35_norm_asset(protocol, asset_id="test_asset")
    evaluator._validate_v35_loaded_gate(
        types.SimpleNamespace(memory_inject_w=types.SimpleNamespace(value=raw_gate)), protocol
    )

    calibration_path = ckpt_dir / "assets" / "v35_calibration_artifact.json"
    calibration_path.write_bytes(calibration_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="identity hashes"):
        evaluator._load_v35_checkpoint_protocol(
            ckpt_dir,
            expected_config_name=cfg.name,
            expected_seed=cfg.seed,
            expected_value_width=cfg.model.memory.d_value,
        )


def test_manual_visibility_and_independent_d_detector_are_mandatory(tmp_path):
    demo, _, manifest = _write_protocol(tmp_path, e_visibility=None)
    with pytest.raises(ValueError, match="manual E-visibility"):
        _load(demo, manifest)

    demo2, _, manifest2 = _write_protocol(
        tmp_path / "dvalid",
        d_valid={"start": 65, "end": 79, "state_dim": 14, "detector": "untrusted"},
    )
    with pytest.raises(ValueError, match="wrong d_valid detector"):
        _load(demo2, manifest2)


def test_label_protocol_rejects_gaps_and_strict_d_outside_semantic_d(tmp_path):
    demo, label_path, manifest = _write_protocol(tmp_path)
    labels = json.loads(label_path.read_text())
    labels[1]["start"] = 16
    label_path.write_text(json.dumps(labels))
    raw = json.loads(manifest.read_text())
    _selected_record(raw)["label_sha256"] = _digest(label_path)
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="gap-free"):
        _load(demo, manifest)

    demo2, _, manifest2 = _write_protocol(
        tmp_path / "other",
        d_valid={
            "start": 64,
            "end": 79,
            "state_dim": 14,
            "detector": evaluator.V35_D_VALID_DETECTOR,
        },
    )
    with pytest.raises(ValueError, match="lies outside semantic D"):
        _load(demo2, manifest2)


def test_bool_and_shape_contract_remains_explicit_in_caller_source():
    source = Path(evaluator.__file__).read_text()
    assert "v35_transition_valid=transition_valid" in source
    assert "v35_write_mask=write_mask" in source
    assert "mem_state = model.memory.init_state(1)" in source
    assert "transition_valid = jnp.ones((), dtype=bool)" in source
    assert "jnp.asarray(expected_write, dtype=bool)" in source
    assert "zero_read=args.zero_gate" in source
    assert "actual_write != expected_write" in source
    # The legacy branch must retain the old call without v3.5 masks.
    assert "Preserve the v3-v3.4 call signature" in source


def test_manifest_and_label_digests_are_lowercase_hex_only(tmp_path):
    demo, _, manifest = _write_protocol(tmp_path)
    with pytest.raises(ValueError, match="lower-case hexadecimal"):
        _load(demo, manifest, manifest_sha256=_digest(manifest).upper())
    with pytest.raises(ValueError, match="required for trusted"):
        _load(demo, manifest, manifest_sha256=35)

    raw = json.loads(manifest.read_text())
    _selected_record(raw)["label_sha256"] = "0" * 63
    manifest.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="lower-case hexadecimal"):
        _load(demo, manifest)


def test_protocol_arrays_are_not_model_or_prediction_derived(tmp_path):
    demo, _, manifest = _write_protocol(tmp_path)
    protocol = _load(demo, manifest)
    eval_frames = np.arange(0, protocol.expected_num_frames, 15)
    write_mask = np.isin(eval_frames, protocol.write_frames)
    np.testing.assert_array_equal(eval_frames, [0, 15, 30, 45, 60, 75, 90])
    np.testing.assert_array_equal(write_mask, [False, True, True, False, False, False, False])
