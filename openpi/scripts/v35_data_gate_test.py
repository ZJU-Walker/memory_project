from __future__ import annotations

# ruff: noqa: SLF001
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import v35_data_gate as gate
import v35_gate_artifacts as artifacts

import openpi.shared.project_paths as project_paths
from openpi.training import config as train_config


def test_cold_help_import_obeys_torch_before_config_contract() -> None:
    scripts = Path(__file__).parent.resolve()
    source = scripts.parent / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(source), str(scripts)))
    result = subprocess.run(
        [sys.executable, str(scripts / "v35_data_gate.py"), "--help"],
        cwd=scripts.parent,
        env=environment,
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout + result.stderr).decode(errors="replace")


def _data_config() -> train_config.DataConfig:
    return train_config.DataConfig(
        subtask_from_task=True,
        subtask_lookahead=15,
        memory_stride_frames=15,
        memory_min_slice_steps=14,
        evidence_subtasks=("inspect both bins",),
        memory_required_subtasks=("wait; target bin is left", "wait; target bin is right"),
        memory_critical_start_pad=75,
        memory_subtask_vocab=(
            "open both lids",
            "wait; target bin is left",
            "open left bin",
            "close both lids and reset arms",
            "inspect both bins",
            "open right bin",
            "wait; target bin is right",
        ),
        memory_waiting_max_speed=4e-3,
        memory_waiting_max_excursion=0.02,
        memory_v35_enabled=True,
        memory_e_tail_guard_frames=5,
        memory_occlusion_subtasks=("close both lids and reset arms",),
        memory_execute_subtasks=("open left bin", "open right bin"),
        memory_sparse_skip_o_prob=0.5,
        memory_waiting_state_dim=14,
        memory_episode_manifest_path="manifest.json",
        memory_manifest_split="train",
        memory_manifest_split_seed=35,
    )


def _phase_fixture(side: str = "left") -> tuple[np.ndarray, list[dict[str, object]]]:
    segments: list[dict[str, object]] = [
        {"start": 0, "end": 29, "task": "open both lids"},
        {"start": 30, "end": 69, "task": "inspect both bins"},
        {"start": 70, "end": 99, "task": "close both lids and reset arms"},
        {"start": 100, "end": 119, "task": f"wait; target bin is {side}"},
        {"start": 120, "end": 149, "task": f"open {side} bin"},
    ]
    names = np.asarray(
        [str(segment["task"]) for segment in segments for _ in range(int(segment["end"]) - int(segment["start"]) + 1)],
        dtype=object,
    )
    return names, segments


def _candidate(family: str, *, sampled_e_count: int = 1) -> gate.Candidate:
    return gate.Candidate(
        family=family,
        start_frame=1,
        sampled_e_count=sampled_e_count,
        n_delay=2,
        commit_frames=(31,),
        d_frames=(106,),
        use_pressure_frames=(),
        state_invalid_d_steps=0,
        credit_reachable_d_steps=1,
    )


def test_zero_e_and_zero_skip_o_fail_closed() -> None:
    with pytest.raises(gate.DataGateError, match="zero final-E-to-D"):
        gate._require_candidate_coverage([], stable_id="episode")
    with pytest.raises(gate.DataGateError, match="zero skip-O"):
        gate._require_candidate_coverage([_candidate("natural")], stable_id="episode")
    with pytest.raises(gate.DataGateError, match="zero-E"):
        gate._require_candidate_coverage([_candidate("skip_o", sampled_e_count=0)], stable_id="episode")


def test_authoritative_layout_has_both_families_and_tail_guard() -> None:
    names, _ = _phase_fixture()
    record = {
        "stable_id": "train/demo1",
        "episode_index": 0,
        "target_side": "left",
        "e_visibility": {
            "first_valid_visible_frame": 30,
            "last_clean_visible_frame": 64,
        },
        "d_valid": {"start": 100, "end": 119},
    }
    candidates, phase = gate._critical_candidates(
        record=record,
        task_names=names,
        data_config=_data_config(),
        max_steps=40,
        action_horizon=15,
        collection_id=0,
        object_id=0,
        cell_id=0,
    )
    assert {candidate.family for candidate in candidates} == {"natural", "skip_o"}
    assert all(candidate.sampled_e_count >= 1 for candidate in candidates)
    assert all(candidate.state_invalid_d_steps == 0 for candidate in candidates)
    assert phase["final_eligible_e_limit"] == 64


def test_short_real_d_interval_with_false_origin0_hint_is_residue_trainable() -> None:
    segments = [
        {"start": 0, "end": 221, "task": "open both lids"},
        {"start": 222, "end": 266, "task": "inspect both bins"},
        {"start": 267, "end": 427, "task": "close both lids and reset arms"},
        {"start": 428, "end": 439, "task": "wait; target bin is right"},
        {"start": 440, "end": 763, "task": "open right bin"},
    ]
    names = np.asarray(
        [str(segment["task"]) for segment in segments for _ in range(int(segment["end"]) - int(segment["start"]) + 1)],
        dtype=object,
    )
    record = {
        "stable_id": "0816_banana/demo23",
        "episode_index": 22,
        "target_side": "right",
        "e_visibility": {
            "first_valid_visible_frame": 222,
            "last_clean_visible_frame": 266,
        },
        # This frozen legacy hint is false. Gate A must enumerate sampler start residues
        # instead of treating grid-origin 0 eligibility as a hard condition.
        "d_valid": {"start": 428, "end": 439, "eligible_at_stride_15": False},
    }
    candidates, _ = gate._critical_candidates(
        record=record,
        task_names=names,
        data_config=_data_config(),
        max_steps=40,
        action_horizon=15,
        collection_id=0,
        object_id=0,
        cell_id=0,
    )
    gate._require_candidate_coverage(candidates, stable_id=str(record["stable_id"]))
    assert {candidate.family for candidate in candidates} == {"natural", "skip_o"}
    assert {frame for candidate in candidates for frame in candidate.d_frames} <= set(range(428, 440))


def test_train_audit_never_requests_final_test_parquet(monkeypatch, tmp_path: Path) -> None:
    names, segments = _phase_fixture()
    label_dir = tmp_path / "train/demo1"
    label_dir.mkdir(parents=True)
    label_path = label_dir / "labels.json"
    label_bytes = (json.dumps(segments) + "\n").encode()
    label_path.write_bytes(label_bytes)
    raw_manifest = {"raw_root": "."}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(raw_manifest), encoding="utf-8")
    record = {
        "stable_id": "train/demo1",
        "episode_index": 0,
        "collection": "0816",
        "object": "banana",
        "part": "",
        "target_side": "left",
        "split": "train",
        "raw_dir": "train/demo1",
        "label_file": "labels.json",
        "label_sha256": hashlib.sha256(label_bytes).hexdigest(),
        "expected_num_frames": len(names),
        "e_visibility": {
            "first_valid_visible_frame": 30,
            "last_clean_visible_frame": 64,
        },
        "d_valid": {"start": 100, "end": 119},
    }
    frozen = artifacts.FrozenManifest(
        path=manifest_path,
        sha256="a" * 64,
        split_assignment_sha256="b" * 64,
        episodes=(
            artifacts.ManifestEpisode("train/demo1", 0, "0816", "banana", "", 0, "train"),
            artifacts.ManifestEpisode("final/demo1", 1, "0830", "banana", "part1", 0, "final_test"),
        ),
    )
    storage = {
        "data/chunk-000/episode_000000.parquet": {
            "path": "data/chunk-000/episode_000000.parquet",
            "size": 0,
            "sha256": "c" * 64,
        },
        "data/chunk-000/episode_000001.parquet": {
            "path": "data/chunk-000/episode_000001.parquet",
            "size": 0,
            "sha256": "d" * 64,
        },
    }
    task_vocab = _data_config().memory_subtask_vocab
    tasks = dict(enumerate(task_vocab))
    task_to_index = {task: index for index, task in tasks.items()}
    requested: list[int] = []

    def reader(path: Path, episode_index: int):
        requested.append(episode_index)
        if episode_index != 0:
            raise AssertionError(f"held-out payload was accessed: {path}")
        return gate.collector.EpisodeColumns(
            episode_index=np.zeros(len(names), dtype=np.int64),
            frame_index=np.arange(len(names), dtype=np.int64),
            dataset_index=np.arange(len(names), dtype=np.int64),
            task_index=np.asarray([task_to_index[str(name)] for name in names], dtype=np.int64),
        )

    audited = gate._audit_train_episodes(
        records={0: record},
        manifest=frozen,
        protocol=SimpleNamespace(train_episode_indices=(0,)),
        storage=storage,
        tasks=tasks,
        data_config=_data_config(),
        max_steps=40,
        action_horizon=15,
        block_steps=25,
        read_episode_columns=reader,
        dataset_root=tmp_path,
    )
    assert requested == [0]
    assert [item["stable_id"] for item in audited] == ["train/demo1"]


def _project(root: Path) -> Path:
    (root / "openpi/src/openpi").mkdir(parents=True)
    (root / "openpi/pyproject.toml").touch()
    return root


def test_stale_train_storage_hash_is_rejected(monkeypatch, tmp_path: Path) -> None:
    root = _project(tmp_path / "memory_project")
    monkeypatch.setenv(project_paths.MEMORY_PROJECT_ROOT_ENV, str(root))
    dataset = root / "data/lerobot/yam/bin_memory_0830_0831_v36_subtask"
    path = dataset / "data/chunk-000/episode_000000.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"current")
    provenance = {
        "train_storage": {
            "root_relative": "data/lerobot/yam/bin_memory_0830_0831_v36_subtask",
            "files": [
                {
                    "path": "data/chunk-000/episode_000000.parquet",
                    "size": len(b"current"),
                    "sha256": hashlib.sha256(b"stale!!").hexdigest(),
                }
            ],
        }
    }
    with pytest.raises(gate.DataGateError, match="bytes changed"):
        gate._storage_record_map(provenance, dataset_root=dataset, train_indices={0})


def test_stale_calibration_dataset_provenance_is_rejected(tmp_path: Path) -> None:
    protocol = SimpleNamespace(
        dataset_protocol_sha256="1" * 64,
        train_stable_ids=("train/demo1",),
    )
    manifest = artifacts.FrozenManifest(
        path=tmp_path / "manifest.json",
        sha256="2" * 64,
        split_assignment_sha256="3" * 64,
        episodes=(),
    )
    payload = {
        "schema_version": "openpi.v35.injection-calibration.v1",
        "status": "pass",
        "gates": {"passes": True},
        "population": {"split": "train", "episode_count": 54, "stable_ids": ["train/demo1"]},
        "provenance": {
            "split_sha256": manifest.sha256,
            "dataset_sha256": "4" * 64,
        },
    }
    digest = artifacts.sha256_bytes(artifacts.canonical_json_bytes(payload))
    descriptor = {
        "artifact_sha256": digest,
        "calibration_id": f"sha256:{digest}",
        "hash_scope": "SHA256 of canonical_json($.payload)",
        "payload": payload,
    }
    path = tmp_path / "calibration.json"
    path.write_bytes(artifacts.canonical_json_bytes(descriptor) + b"\n")
    with pytest.raises(gate.DataGateError, match="stale"):
        gate._load_calibration_dataset_identity(path, protocol=protocol, manifest=manifest)


def test_sealed_calibration_supplies_all_74_successful_commit_records(tmp_path: Path) -> None:
    stable_ids = tuple(f"train/demo{index}" for index in range(54))
    protocol = SimpleNamespace(dataset_protocol_sha256="1" * 64, train_stable_ids=stable_ids)
    manifest = artifacts.FrozenManifest(
        path=tmp_path / "manifest.json",
        sha256="2" * 64,
        split_assignment_sha256="3" * 64,
        episodes=(),
    )
    metrics = [
        {
            "clean_injected_to_residual_rms": 0.2,
            "clean_episode_raw_rms": 0.3,
            "clean_slot_raw_rms_p50": 0.3,
            "decayed_injected_to_residual_rms": 0.1,
            "decayed_retained_amplitude": 0.5,
            "n_delay": 4,
            "stable_id": stable_id,
        }
        for stable_id in stable_ids
    ]
    provenance = {
        name: (
            protocol.dataset_protocol_sha256
            if name == "dataset_sha256"
            else manifest.sha256
            if name == "split_sha256"
            else "4" * 64
        )
        for name in (
            "collector_source_sha256",
            "dataset_sha256",
            "input_npz_sha256",
            "observed_membership_sha256",
            "official_base_source_sha256",
            "preflight_sha256",
            "replay_protocol_sha256",
            "source_sha256",
            "split_sha256",
        )
    }
    provenance["npz_keys"] = [
        "clean_raw_retrieved",
        "episode_split",
        "episode_stable_id",
        "layer8_residual",
        "n_delay",
    ]
    payload = {
        "schema_version": "openpi.v35.injection-calibration.v1",
        "status": "pass",
        "gates": {"passes": True},
        "population": {
            "split": "train",
            "episode_count": 54,
            "stable_ids": list(stable_ids),
            "clean_slots_per_episode": 16,
        },
        "provenance": provenance,
        "statistics": {"episode_metrics": metrics},
    }
    digest = artifacts.sha256_bytes(artifacts.canonical_json_bytes(payload))
    descriptor = {
        "artifact_sha256": digest,
        "calibration_id": f"sha256:{digest}",
        "hash_scope": "SHA256 of canonical_json($.payload)",
        "payload": payload,
    }
    path = tmp_path / "calibration.json"
    path.write_bytes(artifacts.canonical_json_bytes(descriptor) + b"\n")
    identity = gate._load_calibration_dataset_identity(path, protocol=protocol, manifest=manifest)
    assert identity.kind == "injection_calibration"
    assert len(identity.successful_commit_metrics) == 54


def test_preflight_alone_cannot_claim_successful_commits(tmp_path: Path) -> None:
    payload = {"schema_version": gate.replay.PREFLIGHT_SCHEMA}
    digest = artifacts.sha256_bytes(artifacts.canonical_json_bytes(payload))
    preflight = {
        "artifact_sha256": digest,
        "preflight_id": f"sha256:{digest}",
        "hash_scope": "SHA256 of canonical_json($.payload)",
        "payload": payload,
    }
    path = tmp_path / "preflight.json"
    path.write_bytes(artifacts.canonical_json_bytes(preflight) + b"\n")
    with pytest.raises(gate.DataGateError, match="no successful commits"):
        gate._load_calibration_dataset_identity(
            path,
            protocol=SimpleNamespace(),
            manifest=artifacts.FrozenManifest(path, "1" * 64, "2" * 64, ()),
        )


def test_writer_refuses_overwrite(tmp_path: Path) -> None:
    report = artifacts.artifact_envelope(gate.REPORT_SCHEMA_VERSION, {"x": 1})
    decision = artifacts.artifact_envelope(gate.DECISION_SCHEMA_VERSION, {"x": 2})
    report_path = tmp_path / "report.json"
    decision_path = tmp_path / "decision.json"
    gate.write_gate_artifacts(
        decision_path=decision_path,
        detail_report_path=report_path,
        decision=decision,
        report=report,
    )
    with pytest.raises(gate.DataGateError, match="overwrite"):
        gate.write_gate_artifacts(
            decision_path=decision_path,
            detail_report_path=report_path,
            decision=decision,
            report=report,
        )
