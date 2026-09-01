import dataclasses
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v35_calibration_replay as replay
    import v35_injection_calibration as calibration
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _frozen_protocol_files(tmp_path: Path) -> tuple[Path, str, Path, Path]:
    episodes = [
        {
            "stable_id": f"train-{index:03d}",
            "episode_index": index,
            "include": True,
            "split": "train",
        }
        for index in range(54)
    ] + [
        {
            "stable_id": f"dev-{index:03d}",
            "episode_index": 54 + index,
            "include": True,
            "split": "dev",
        }
        for index in range(8)
    ]
    manifest = tmp_path / "manifest.json"
    _json(manifest, {"schema_version": 1, "split_seed": 36, "episodes": episodes})
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    norm_stats = tmp_path / "norm_stats.json"
    norm_stats.write_text('{"state":{"mean":[0.0],"std":[1.0]}}', encoding="utf-8")
    frame_counts = [2] * 54
    episode_protocol = [
        {
            "episode_index": index,
            "stable_id": record["stable_id"],
            "split": record["split"],
            "include": record["include"],
            "frame_count": 2,
        }
        for index, record in enumerate(episodes)
    ]
    # The production dataset embeds images in parquet and intentionally has no videos/ tree.
    storage_records = sorted(
        [
            {
                "path": f"data/chunk-000/episode_{index:06d}.parquet",
                "size": 1,
                "sha256": "a" * 64,
            }
            for index in range(54)
        ]
        + [{"path": "meta/info.json", "size": 1, "sha256": "c" * 64}],
        key=lambda record: record["path"],
    )
    storage_sha = hashlib.sha256(
        json.dumps(storage_records, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    dataset_protocol = {
        "manifest_sha256": manifest_sha,
        "episodes": episode_protocol,
        "train_storage_sha256": storage_sha,
    }
    dataset_protocol_sha = hashlib.sha256(
        json.dumps(dataset_protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    provenance = {
        "schema_version": 2,
        "status": "complete",
        "repo_id": "yam/test-v35",
        "manifest": {"sha256": manifest_sha, "active_split": "train", "split_seed": 36},
        "train_storage": {
            "root": str(tmp_path.resolve()),
            "sha256": storage_sha,
            "files": storage_records,
            "selected_episode_indices": list(range(54)),
            "scope": replay.TRAIN_STORAGE_SCOPE,
        },
        "selection": {
            "dataset_num_frames": 124,
            "dataset_num_episodes": 62,
            "selected_num_episodes": 54,
            "selected_episode_indices": list(range(54)),
            "selected_stable_ids": [f"train-{index:03d}" for index in range(54)],
            "selected_episode_frame_counts": frame_counts,
            "selected_num_frames": sum(frame_counts),
            "dataset_episode_frame_protocol_sha256": dataset_protocol_sha,
        },
        "computation": {
            "protocol": replay.NORMALIZATION_PROTOCOL,
            "requested_batch_size": 16,
            "processed_base_rows": sum(frame_counts),
            "drop_last_rows": 0,
            "num_batches_including_partial_final_batch": 7,
        },
        "norm_stats": {
            "file": "norm_stats.json",
            "sha256": hashlib.sha256(norm_stats.read_bytes()).hexdigest(),
        },
    }
    norm_provenance = tmp_path / "norm_stats_provenance.json"
    _json(norm_provenance, provenance)
    return manifest, manifest_sha, norm_provenance, norm_stats


def _data_protocol(tmp_path: Path) -> replay.FrozenDataProtocol:
    manifest, manifest_sha, provenance, stats = _frozen_protocol_files(tmp_path)
    return replay.validate_frozen_data_protocol(
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        norm_provenance_path=provenance,
        norm_stats_path=stats,
        expected_repo_id="yam/test-v35",
        expected_split_seed=36,
        production=False,
    )


def _bindings() -> replay.ReplayBindings:
    return replay.ReplayBindings(
        step0_parameter_tree_sha256="1" * 64,
        official_base_source_sha256="2" * 64,
        replay_protocol_sha256=replay.replay_protocol_sha256(),
        collector_source_sha256=replay.collector_source_sha256(),
        manifest_sha256="3" * 64,
        dataset_protocol_sha256="4" * 64,
    )


def _record(index: int, *, bindings: replay.ReplayBindings | None = None) -> replay.EpisodeReplayRecord:
    channels = 4
    return replay.EpisodeReplayRecord(
        stable_id=f"train-{index:03d}",
        split="train",
        evidence_frame_index=100,
        decision_frame_index=200,
        n_delay=12 + index % 5,
        evidence_observation_sha256="5" * 64,
        decision_observation_sha256="6" * 64,
        clean_raw_retrieved=np.ones((16, channels), dtype=np.float32),
        layer8_residual=np.full((channels,), 0.3, dtype=np.float32),
        mixed_precision_noise_raw_retrieved=np.full((channels,), 0.005, dtype=np.float32),
        query_noise_raw_retrieved=np.full((1, channels), 0.01, dtype=np.float32),
        query_noise_cosine=np.asarray([0.05], dtype=np.float32),
        query_noise_kind=("low_cos_query",),
        query_noise_frame_index=np.asarray([190], dtype=np.int64),
        query_noise_observation_sha256=("7" * 64,),
        commit_relative_residual=1e-7,
        commit_applied=True,
        bindings=_bindings() if bindings is None else bindings,
    )


def _preflight(tmp_path: Path) -> tuple[Path, replay.ReplayBindings]:
    data = _data_protocol(tmp_path)
    gate = np.full((4,), np.arctanh(np.float32(0.5)), dtype=np.float32)
    step0 = replay.Step0Identity(
        actual_parameter_tree_sha256="1" * 64,
        official_base_source_sha256="2" * 64,
        target_schema_sha256="8" * 64,
        graft_manifest_sha256="9" * 64,
        raw_gate=gate,
    )
    controls_name = "step0_controls.npz"
    controls_sha = replay._write_npz_once(  # noqa: SLF001
        tmp_path / controls_name,
        {
            "memory_inject_w": gate,
            "step0_parameter_tree_sha256": np.asarray(step0.actual_parameter_tree_sha256),
            "official_base_source_sha256": np.asarray(step0.official_base_source_sha256),
        },
    )
    artifact = replay.make_preflight_artifact(
        config_name="pi05_yam_mem_v35",
        config_seed=42,
        alpha_step=0.01,
        data=data,
        step0=step0,
        controls_file=controls_name,
        controls_sha256=controls_sha,
    )
    path = tmp_path / "preflight.json"
    replay._write_json_once(path, artifact)  # noqa: SLF001
    return path, replay.bindings_from_preflight(artifact)


def test_frozen_data_protocol_is_exactly_train74_and_norm_bound(tmp_path: Path) -> None:
    protocol = _data_protocol(tmp_path)

    assert len(protocol.train_stable_ids) == 54
    assert protocol.train_stable_ids[0] == "train-000"
    assert protocol.train_stable_ids[-1] == "train-053"
    assert protocol.dataset_num_episodes == 62
    assert protocol.dataset_num_frames == 124
    assert len(protocol.train_episode_frame_counts) == 54
    assert protocol.train_storage_file_count == 55
    assert len(protocol.train_storage_sha256) == 64

    manifest = json.loads(protocol.manifest_path.read_text(encoding="utf-8"))
    manifest["episodes"][0]["split"] = "dev"
    _json(protocol.manifest_path, manifest)
    changed_sha = hashlib.sha256(protocol.manifest_path.read_bytes()).hexdigest()
    with pytest.raises(replay.ReplayPreflightError, match="exactly 54"):
        replay.validate_frozen_data_protocol(
            manifest_path=protocol.manifest_path,
            manifest_sha256=changed_sha,
            norm_provenance_path=protocol.norm_provenance_path,
            norm_stats_path=protocol.norm_stats_path,
            expected_repo_id="yam/test-v35",
            expected_split_seed=36,
            production=False,
        )


def test_frozen_protocol_requires_self_hashed_train_storage(tmp_path: Path) -> None:
    manifest, manifest_sha, provenance_path, stats = _frozen_protocol_files(tmp_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["train_storage"]["sha256"] = "0" * 64
    _json(provenance_path, provenance)

    with pytest.raises(replay.ReplayPreflightError, match="aggregate SHA256"):
        replay.validate_frozen_data_protocol(
            manifest_path=manifest,
            manifest_sha256=manifest_sha,
            norm_provenance_path=provenance_path,
            norm_stats_path=stats,
            expected_repo_id="yam/test-v35",
            expected_split_seed=36,
            production=False,
        )

    provenance.pop("train_storage")
    _json(provenance_path, provenance)
    with pytest.raises(replay.ReplayPreflightError, match="requires a train_storage seal"):
        replay.validate_frozen_data_protocol(
            manifest_path=manifest,
            manifest_sha256=manifest_sha,
            norm_provenance_path=provenance_path,
            norm_stats_path=stats,
            expected_repo_id="yam/test-v35",
            expected_split_seed=36,
            production=False,
        )


def test_production_protocol_resolves_only_project_relative_paths(monkeypatch, tmp_path: Path) -> None:
    manifest, manifest_sha, provenance_path, stats = _frozen_protocol_files(tmp_path)
    dataset_root = tmp_path / "data" / "lerobot" / "yam" / "test-v35"
    dataset_root.mkdir(parents=True)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["manifest"]["path_relative"] = "manifest.json"
    storage = provenance["train_storage"]
    storage.pop("root")
    storage["root_contract"] = replay.TRAIN_STORAGE_ROOT_CONTRACT
    storage["root_relative"] = "data/lerobot/yam/test-v35"
    _json(provenance_path, provenance)
    monkeypatch.setattr(replay.project_paths, "memory_project_root", lambda: tmp_path.resolve())

    protocol = replay.validate_frozen_data_protocol(
        manifest_path=manifest,
        manifest_sha256=manifest_sha,
        norm_provenance_path=provenance_path,
        norm_stats_path=stats,
        expected_repo_id="yam/test-v35",
        expected_split_seed=36,
    )

    assert protocol.production is True
    assert protocol.train_storage_root == dataset_root.resolve()
    gate = np.full((4,), np.arctanh(np.float32(0.5)), dtype=np.float32)
    artifact = replay.make_preflight_artifact(
        config_name="pi05_yam_mem_v35",
        config_seed=42,
        alpha_step=0.01,
        data=protocol,
        step0=replay.Step0Identity(
            actual_parameter_tree_sha256="1" * 64,
            official_base_source_sha256="2" * 64,
            target_schema_sha256="8" * 64,
            graft_manifest_sha256="9" * 64,
            raw_gate=gate,
        ),
        controls_file="step0_controls.npz",
        controls_sha256="a" * 64,
    )
    recorded = artifact["payload"]["data"]
    assert recorded["production"] is True
    assert recorded["manifest_path_relative"] == "manifest.json"
    assert recorded["norm_provenance_path_relative"] == "norm_stats_provenance.json"
    assert recorded["norm_stats_path_relative"] == "norm_stats.json"
    assert not ({"manifest_path", "norm_provenance_path", "norm_stats_path"} & recorded.keys())
    preflight_path = tmp_path / "production_preflight.json"
    _json(preflight_path, artifact)
    assert replay.read_preflight(preflight_path) == artifact

    artifact["payload"]["data"]["manifest_path"] = "/iris/u/kewalk/memory_project/data/manifest.json"
    forged = replay._envelope(artifact["payload"], id_prefix="preflight")  # noqa: SLF001
    _json(preflight_path, forged)
    with pytest.raises(replay.ReplayPreflightError, match="forbidden machine-local paths"):
        replay.read_preflight(preflight_path)


def test_production_protocol_rejects_machine_local_storage_root(monkeypatch, tmp_path: Path) -> None:
    manifest, manifest_sha, provenance_path, stats = _frozen_protocol_files(tmp_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["manifest"]["path_relative"] = "manifest.json"
    provenance["train_storage"]["root_relative"] = "data/lerobot/yam/test-v35"
    provenance["train_storage"]["root_contract"] = replay.TRAIN_STORAGE_ROOT_CONTRACT
    _json(provenance_path, provenance)
    monkeypatch.setattr(replay.project_paths, "memory_project_root", lambda: tmp_path.resolve())

    with pytest.raises(replay.ReplayPreflightError, match="legacy machine-local root"):
        replay.validate_frozen_data_protocol(
            manifest_path=manifest,
            manifest_sha256=manifest_sha,
            norm_provenance_path=provenance_path,
            norm_stats_path=stats,
            expected_repo_id="yam/test-v35",
            expected_split_seed=36,
        )


def test_norm_protocol_and_optional_video_contract_fail_closed(tmp_path: Path) -> None:
    manifest, manifest_sha, provenance_path, stats = _frozen_protocol_files(tmp_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["computation"]["protocol"] = "transformed-dataset-v1"
    _json(provenance_path, provenance)
    with pytest.raises(replay.ReplayPreflightError, match="protocol must be exactly"):
        replay.validate_frozen_data_protocol(
            manifest_path=manifest,
            manifest_sha256=manifest_sha,
            norm_provenance_path=provenance_path,
            norm_stats_path=stats,
            expected_repo_id="yam/test-v35",
            expected_split_seed=36,
            production=False,
        )

    provenance["computation"]["protocol"] = replay.NORMALIZATION_PROTOCOL
    provenance["train_storage"]["files"].append(
        {"path": "videos/chunk-000/camera/episode_000000.mp4", "size": 1, "sha256": "b" * 64}
    )
    provenance["train_storage"]["files"].sort(key=lambda record: record["path"])
    _json(provenance_path, provenance)
    with pytest.raises(replay.ReplayPreflightError, match="either zero or every train video"):
        replay.validate_frozen_data_protocol(
            manifest_path=manifest,
            manifest_sha256=manifest_sha,
            norm_provenance_path=provenance_path,
            norm_stats_path=stats,
            expected_repo_id="yam/test-v35",
            expected_split_seed=36,
            production=False,
        )


@pytest.mark.parametrize("bad_path", [Path("../foreign"), Path("/iris/u/user/data")])
def test_production_cli_paths_reject_unconfined_paths(bad_path: Path) -> None:
    with pytest.raises(replay.ReplayPreflightError, match="memory_project"):
        replay._resolve_project_cli_path(bad_path, name="manifest path")  # noqa: SLF001


def test_live_dataset_contract_recomputes_norm_protocol(tmp_path: Path) -> None:
    protocol = _data_protocol(tmp_path)
    episode_index = np.repeat(np.arange(62, dtype=np.int64), 2)

    class _Columns:
        column_names = ("episode_index",)

        def __getitem__(self, key: str) -> np.ndarray:
            assert key == "episode_index"
            return episode_index

    class _HF:
        def with_format(self, _format: None) -> _Columns:
            return _Columns()

    class _Inner:
        hf_dataset = _HF()

    class _Dataset:
        _dataset = _Inner()

        def __len__(self) -> int:
            return len(episode_index)

    dataset = _Dataset()
    digest = replay._validate_actual_dataset_rows(dataset, expected=protocol)  # noqa: SLF001

    assert digest == protocol.dataset_protocol_sha256


def test_episode_shards_are_byte_reproducible_and_pickle_free(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    record = _record(0)

    first_sha = replay.write_episode_replay_shard(first, record)
    second_sha = replay.write_episode_replay_shard(second, record)
    restored, restored_sha = replay.load_episode_replay_shard(first)

    assert first.read_bytes() == second.read_bytes()
    assert first_sha == second_sha == restored_sha
    assert restored.stable_id == record.stable_id
    assert restored.bindings == record.bindings


def test_episode_shard_rejects_nontrain_and_placeholder_clean_read(tmp_path: Path) -> None:
    with pytest.raises(replay.ReplayPreflightError, match="split='train'"):
        replay.write_episode_replay_shard(tmp_path / "dev.npz", dataclasses.replace(_record(0), split="dev"))
    with pytest.raises(replay.ReplayPreflightError, match="exact zero"):
        replay.write_episode_replay_shard(
            tmp_path / "zero.npz",
            dataclasses.replace(_record(0), clean_raw_retrieved=np.zeros((16, 4), dtype=np.float32)),
        )


def test_sealer_requires_and_emits_exact_authenticated_train74(tmp_path: Path) -> None:
    preflight, bindings = _preflight(tmp_path)
    shards = tmp_path / "shards"
    shards.mkdir()
    for index in range(54):
        replay.write_episode_replay_shard(shards / f"{index:03d}.npz", _record(index, bindings=bindings))
    output = tmp_path / "replay.npz"

    output_sha = replay.seal_replay_npz(
        preflight_path=preflight,
        shards_dir=shards,
        output_path=output,
    )
    second_output = tmp_path / "replay_second.npz"
    second_sha = replay.seal_replay_npz(
        preflight_path=preflight,
        shards_dir=shards,
        output_path=second_output,
    )
    loaded = calibration.load_replay_stats(output)
    artifact = calibration.calibrate_injection(
        loaded.stats,
        input_sha256=loaded.input_sha256,
        npz_keys=loaded.npz_keys,
    )

    assert output_sha == second_sha == hashlib.sha256(output.read_bytes()).hexdigest()
    assert output.read_bytes() == second_output.read_bytes()
    assert loaded.stats.episode_stable_id == tuple(f"train-{index:03d}" for index in range(54))
    assert loaded.stats.clean_raw_retrieved.shape == (54, 16, 4)
    assert loaded.stats.noise_raw_retrieved.shape == (108, 4)
    assert loaded.stats.source_sha256 == bindings.step0_parameter_tree_sha256
    assert artifact["payload"]["parameters"]["prototype_injected_rms_target"] == pytest.approx(0.3)
    assert artifact["payload"]["provenance"]["official_base_source_sha256"] == (bindings.official_base_source_sha256)
    assert artifact["payload"]["provenance"]["replay_protocol_sha256"] == bindings.replay_protocol_sha256
    with np.load(output, allow_pickle=False) as archive:
        assert str(archive["official_base_source_sha256"]) == bindings.official_base_source_sha256
        assert str(archive["replay_protocol_sha256"]) == bindings.replay_protocol_sha256
        assert archive["commit_applied"].all()


def test_sealer_rejects_missing_or_wrong_step0_shard(tmp_path: Path) -> None:
    preflight, bindings = _preflight(tmp_path)
    shards = tmp_path / "shards"
    shards.mkdir()
    for index in range(53):
        replay.write_episode_replay_shard(shards / f"{index:03d}.npz", _record(index, bindings=bindings))
    with pytest.raises(replay.ReplayPreflightError, match="exactly 54 NPZ shards"):
        replay.seal_replay_npz(
            preflight_path=preflight,
            shards_dir=shards,
            output_path=tmp_path / "missing.npz",
        )

    wrong = dataclasses.replace(bindings, step0_parameter_tree_sha256="a" * 64)
    replay.write_episode_replay_shard(shards / "053.npz", _record(53, bindings=wrong))
    with pytest.raises(replay.ReplayPreflightError, match="mismatched provenance"):
        replay.seal_replay_npz(
            preflight_path=preflight,
            shards_dir=shards,
            output_path=tmp_path / "wrong.npz",
        )


def test_preflight_hash_fails_after_tampering(tmp_path: Path) -> None:
    preflight, _ = _preflight(tmp_path)
    artifact = json.loads(preflight.read_text(encoding="utf-8"))
    artifact["payload"]["config"]["seed"] = 999
    preflight.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(replay.ReplayPreflightError, match="payload hash"):
        replay.read_preflight(preflight)
