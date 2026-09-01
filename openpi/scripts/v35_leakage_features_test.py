import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v35_gate_artifacts as artifacts
    import v35_leakage_features as producer
    import v35_leakage_gate as gate
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


def _manifest(tmp_path: Path) -> tuple[artifacts.FrozenManifest, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    episodes = []
    for part, repeats in (("part1", (4, 4, 4, 4)), ("part2", (4, 3, 4, 3))):
        demo = 1
        for (object_name, side), count in zip(
            (("banana", "left"), ("banana", "right"), ("grey_pepper_box", "left"), ("grey_pepper_box", "right")),
            repeats,
            strict=True,
        ):
            for _ in range(count):
                episodes.append(
                    {
                        "stable_id": f"0830_bin_{part}/demo{demo}",
                        "episode_index": len(episodes),
                        "collection": "0830",
                        "object": object_name,
                        "part": part,
                        "target_side": side,
                        "split": "train",
                        "include": True,
                        "d_valid": {
                            "detector": "14d-max-step-lt-0.004-max-excursion-lte-0.02-v1",
                            "state_dim": 14,
                            "start": 1,
                            "end": 3,
                        },
                        "expected_num_frames": 7,
                        "prompt": producer.EXPECTED_PROMPTS[object_name],
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
            episodes.append(
                {
                    "stable_id": f"0831_bin/demo{demo}",
                    "episode_index": len(episodes),
                    "collection": "0831",
                    "object": object_name,
                    "part": "",
                    "target_side": side,
                    "split": "train",
                    "include": True,
                    "d_valid": {
                        "detector": "14d-max-step-lt-0.004-max-excursion-lte-0.02-v1",
                        "state_dim": 14,
                        "start": 2,
                        "end": 4,
                    },
                    "expected_num_frames": 8,
                    "prompt": producer.EXPECTED_PROMPTS[object_name],
                }
            )
            demo += 1
    expected = artifacts._expected_frozen_splits(episodes)  # noqa: SLF001
    for record in episodes:
        record["split"] = expected[record["stable_id"]]
    raw = {
        "dataset_version": "v36",
        "episodes": episodes,
        "review_status": "frozen",
        "schema_version": 2,
        "split_algorithm": artifacts.V35_SPLIT_ALGORITHM,
        "split_algorithm_sha256": artifacts.V35_SPLIT_ALGORITHM_SHA256,
        "split_seed": 36,
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return artifacts.load_frozen_manifest(path, expected_sha256=digest), raw


def _fake_inputs(tmp_path: Path) -> producer.AuthenticatedInputs:
    manifest, raw = _manifest(tmp_path)
    episodes = producer._manifest_episode_sources(manifest, raw)  # noqa: SLF001
    init = tmp_path / "initialization_manifest.json"
    graft = tmp_path / "initialization_graft_manifest.json"
    init.write_text("{}", encoding="utf-8")
    graft.write_text("{}", encoding="utf-8")
    return producer.AuthenticatedInputs(
        manifest=manifest,
        manifest_raw=raw,
        episodes=episodes,
        dataset_root=tmp_path / "dataset",
        norm_dir=tmp_path / "norm",
        norm_stats_sha256="9" * 64,
        norm_provenance_sha256="8" * 64,
        train_storage_sha256="7" * 64,
        dataset_protocol_sha256="6" * 64,
        initialization_path=init,
        initialization_raw={},
        initialization_sha256=hashlib.sha256(init.read_bytes()).hexdigest(),
        initialization_parameter_tree_sha256="4" * 64,
        official_base_source_tree_sha256="5" * 64,
        graft_path=graft,
        graft_sha256=hashlib.sha256(graft.read_bytes()).hexdigest(),
        step0_params_path=tmp_path / "params",
    )


def _arrays(stable_ids: list[str] | None = None) -> dict[str, np.ndarray]:
    stable_ids = stable_ids or [f"episode-{index:02d}" for index in range(54)]
    arrays: dict[str, np.ndarray] = {
        "episode_stable_id": np.asarray(stable_ids),
    }
    arrays.update(
        {
            name: np.arange(54 * (offset + 1), dtype=np.float32).reshape(54, offset + 1)
            for offset, name in enumerate(gate.ALL_FEATURES, start=1)
        }
    )
    return arrays


def _project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "memory_project"
    (root / "openpi" / "src" / "openpi").mkdir(parents=True)
    (root / "openpi" / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    monkeypatch.setenv("MEMORY_PROJECT_ROOT", str(root))
    return root


def test_protocol_hashes_are_frozen_literals() -> None:
    producer._assert_frozen_protocol_hashes()  # noqa: SLF001
    assert (
        producer._canonical_sha256(  # noqa: SLF001
            producer.IMAGE_PREPROCESSING_SPEC
        )
        == producer.FROZEN_IMAGE_PREPROCESSING_SHA256
    )
    assert (
        producer._canonical_sha256(  # noqa: SLF001
            producer.PREPROCESSING_PROTOCOL_SPEC
        )
        == producer.FROZEN_PREPROCESSING_PROTOCOL_SHA256
    )
    assert (
        producer._canonical_sha256(  # noqa: SLF001
            producer.FEATURE_PROTOCOL_SPEC
        )
        == producer.FROZEN_FEATURE_PROTOCOL_SHA256
    )
    option_names = {option for action in producer._parser()._actions for option in action.option_strings}  # noqa: SLF001
    assert "--feature-protocol-sha256" not in option_names
    assert "--preprocessing-protocol-sha256" not in option_names
    assert "--manifest" not in option_names


def test_collection_calls_only_train74_in_frozen_order(tmp_path: Path) -> None:
    inputs = _fake_inputs(tmp_path)

    class SpyExtractor:
        def __init__(self):
            self.calls: list[int] = []

        def extract_episode(self, episode: producer.EpisodeSource) -> producer.EpisodeFeatureRow:
            self.calls.append(episode.episode_index)
            features = {name: np.full(3, episode.episode_index, dtype=np.float32) for name in gate.ALL_FEATURES}
            return producer.EpisodeFeatureRow(episode.stable_id, episode.d_count, features)

    extractor = SpyExtractor()
    arrays = producer.collect_feature_arrays(inputs, extractor)

    expected = [episode.episode_index for episode in inputs.manifest.split("train")]
    forbidden = {
        episode.episode_index for split in ("development", "final_test") for episode in inputs.manifest.split(split)
    }
    assert extractor.calls == expected
    assert not forbidden.intersection(extractor.calls)
    assert arrays["episode_stable_id"].tolist() == [episode.stable_id for episode in inputs.episodes]
    assert all(
        array.dtype == np.float32 and array.shape == (54, 3)
        for name, array in arrays.items()
        if name != "episode_stable_id"
    )


def test_collection_rejects_wrong_identity_count_dtype_and_nonfinite(tmp_path: Path) -> None:
    inputs = _fake_inputs(tmp_path)

    class BadExtractor:
        def extract_episode(self, episode: producer.EpisodeSource) -> producer.EpisodeFeatureRow:
            features = {name: np.ones(2, dtype=np.float32) for name in gate.ALL_FEATURES}
            if episode == inputs.episodes[0]:
                features[gate.ALL_FEATURES[0]][0] = np.nan
            return producer.EpisodeFeatureRow(episode.stable_id, episode.d_count, features)

    with pytest.raises(producer.LeakageFeatureError, match="non-finite"):
        producer.collect_feature_arrays(inputs, BadExtractor())


def test_canonical_npz_is_byte_deterministic_and_strict(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    arrays = _arrays()

    first_sha = producer.write_canonical_npz(first, arrays)
    second_sha = producer.write_canonical_npz(second, arrays)

    assert first.read_bytes() == second.read_bytes()
    assert first_sha == second_sha == hashlib.sha256(first.read_bytes()).hexdigest()
    producer.validate_canonical_npz(first)
    with pytest.raises(producer.LeakageFeatureError, match="overwrite"):
        producer.write_canonical_npz(first, arrays)

    noncanonical = tmp_path / "noncanonical.npz"
    np.savez(noncanonical, **arrays)
    with pytest.raises(producer.LeakageFeatureError, match="frozen NPY v2.0"):
        producer.validate_canonical_npz(noncanonical)


def test_storage_seal_is_exactly_train74_and_rejects_final_parquet(tmp_path: Path) -> None:
    inputs = _fake_inputs(tmp_path)
    files = [
        {
            "path": f"data/chunk-000/episode_{episode.episode_index:06d}.parquet",
            "size": 1,
            "sha256": f"{episode.episode_index:064x}",
        }
        for episode in inputs.episodes
    ]
    files.extend(
        {"path": path, "size": 1, "sha256": f"{offset + 100:064x}"}
        for offset, path in enumerate(
            (
                "meta/episode_prompts.json",
                "meta/episodes.jsonl",
                "meta/info.json",
                "meta/tasks.jsonl",
            )
        )
    )
    protocol_sha = "a" * 64
    provenance = {
        "selection": {"dataset_episode_frame_protocol_sha256": protocol_sha},
        "train_storage": {
            "root_contract": producer.ROOT_CONTRACT,
            "root_relative": producer.DATASET_ROOT.as_posix(),
            "scope": producer.STORAGE_SCOPE,
            "selected_episode_indices": [episode.episode_index for episode in inputs.episodes],
            "files": files,
            "sha256": producer._canonical_sha256(files),  # noqa: SLF001
        },
    }

    assert producer._validate_storage_seal(  # noqa: SLF001
        provenance,
        dataset_root=producer.project_paths.project_path(producer.DATASET_ROOT),
        train_episodes=inputs.episodes,
    ) == (provenance["train_storage"]["sha256"], protocol_sha)

    final = inputs.manifest.split("final_test")[0]
    tampered_files = list(files)
    tampered_files[len(inputs.episodes) - 1] = {
        "path": f"data/chunk-000/episode_{final.episode_index:06d}.parquet",
        "size": 1,
        "sha256": "f" * 64,
    }
    tampered = json.loads(json.dumps(provenance))
    tampered["train_storage"]["files"] = tampered_files
    tampered["train_storage"]["sha256"] = producer._canonical_sha256(tampered_files)  # noqa: SLF001
    with pytest.raises(producer.LeakageFeatureError, match="development/final"):
        producer._validate_storage_seal(  # noqa: SLF001
            tampered,
            dataset_root=producer.project_paths.project_path(producer.DATASET_ROOT),
            train_episodes=inputs.episodes,
        )


def test_norm_schema2_dataset_protocol_is_recomputed_not_trusted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _project_root(tmp_path, monkeypatch)
    manifest, raw = _manifest(tmp_path / "inputs")
    episodes = producer._manifest_episode_sources(manifest, raw)  # noqa: SLF001
    dataset_root = root.joinpath(*producer.DATASET_ROOT.parts)
    dataset_root.mkdir(parents=True)
    norm_dir = root / "norm"
    norm_dir.mkdir()
    norm_path = norm_dir / producer.NORM_STATS_FILE
    norm_path.write_text("{}", encoding="utf-8")
    train_frames = sum(episode.expected_frames for episode in episodes)
    dataset_frames = sum(record["expected_num_frames"] for record in raw["episodes"] if record.get("include", True))
    monkeypatch.setattr(producer, "EXPECTED_TRAIN_FRAMES", train_frames)
    monkeypatch.setattr(producer, "EXPECTED_DATASET_FRAMES", dataset_frames)

    files = [
        {
            "path": f"data/chunk-000/episode_{episode.episode_index:06d}.parquet",
            "size": 1,
            "sha256": f"{episode.episode_index:064x}",
        }
        for episode in episodes
    ]
    files.extend(
        {"path": path, "size": 1, "sha256": f"{offset + 100:064x}"}
        for offset, path in enumerate(
            (
                "meta/episode_prompts.json",
                "meta/episodes.jsonl",
                "meta/info.json",
                "meta/tasks.jsonl",
            )
        )
    )
    storage_sha = producer._canonical_sha256(files)  # noqa: SLF001
    by_index = {record["episode_index"]: record for record in raw["episodes"] if record.get("include", True)}
    episode_protocol = [
        {
            "episode_index": episode.episode_index,
            "stable_id": episode.stable_id,
            "split": episode.split,
            "include": True,
            "frame_count": by_index[episode.episode_index]["expected_num_frames"],
        }
        for episode in manifest.episodes
    ]
    dataset_protocol_sha = producer._canonical_sha256(  # noqa: SLF001
        {
            "manifest_sha256": manifest.sha256,
            "episodes": episode_protocol,
            "train_storage_sha256": storage_sha,
        }
    )
    frame_counts = [episode.expected_frames for episode in episodes]
    excluded = sorted(
        episode.episode_index for split in ("development", "final_test") for episode in manifest.split(split)
    )
    provenance = {
        "schema_version": 2,
        "status": "complete",
        "scope": producer.NORM_SCOPE,
        "repo_id": producer.project_paths.V35_REPO_ID,
        "manifest": {
            "path_relative": producer.FROZEN_MANIFEST.as_posix(),
            "sha256": manifest.sha256,
            "schema_version": 2,
            "split_seed": 36,
            "active_split": "train",
        },
        "selection": {
            "dataset_num_frames": dataset_frames,
            "dataset_num_episodes": 70,
            "selected_num_frames": train_frames,
            "selected_num_episodes": 54,
            "selected_episode_indices": [episode.episode_index for episode in episodes],
            "selected_stable_ids": [episode.stable_id for episode in episodes],
            "selected_episode_frame_counts": frame_counts,
            "excluded_episode_indices": excluded,
            "dataset_episode_frame_protocol_sha256": dataset_protocol_sha,
        },
        "train_storage": {
            "root_contract": producer.ROOT_CONTRACT,
            "root_relative": producer.DATASET_ROOT.as_posix(),
            "sha256": storage_sha,
            "files": files,
            "selected_episode_indices": [episode.episode_index for episode in episodes],
            "scope": producer.STORAGE_SCOPE,
        },
        "computation": {
            "protocol": producer.NORM_COMPUTATION_PROTOCOL,
            "requested_batch_size": 12,
            "num_batches_including_partial_final_batch": (train_frames + 11) // 12,
            "processed_base_rows": train_frames,
            "drop_last_rows": 0,
        },
        "norm_stats": {
            "file": producer.NORM_STATS_FILE,
            "sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
        },
    }
    provenance_path = norm_dir / producer.NORM_PROVENANCE_FILE
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    result = producer._validate_norm_provenance(  # noqa: SLF001
        norm_dir=norm_dir,
        manifest=manifest,
        manifest_raw=raw,
        episodes=episodes,
        dataset_root=dataset_root,
    )
    assert result[2:] == (storage_sha, dataset_protocol_sha)

    provenance["selection"]["dataset_episode_frame_protocol_sha256"] = "f" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(producer.LeakageFeatureError, match="not reproducible"):
        producer._validate_norm_provenance(  # noqa: SLF001
            norm_dir=norm_dir,
            manifest=manifest,
            manifest_raw=raw,
            episodes=episodes,
            dataset_root=dataset_root,
        )


def test_parquet_reader_requests_only_final_d_columns_and_train_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _fake_inputs(tmp_path)
    episode = inputs.episodes[0]
    wait_task = producer.EXPECTED_WAIT_TASKS[episode.target_side]
    extractor = object.__new__(producer.ProductionEpisodeExtractor)
    extractor.inputs = inputs
    extractor._train_indices = {item.episode_index for item in inputs.episodes}  # noqa: SLF001
    extractor._task_names = {11: wait_task}  # noqa: SLF001
    calls = []
    rows = [
        {
            "image": b"image",
            "left_wrist_image": b"left",
            "right_wrist_image": b"right",
            "state": np.zeros(14, dtype=np.float32),
            "frame_index": frame,
            "episode_index": episode.episode_index,
            "task_index": 11,
        }
        for frame in range(episode.d_start, episode.d_end + 1)
    ]

    def fake_read_table(path, *, columns, filters):
        calls.append((Path(path), tuple(columns), filters))
        return SimpleNamespace(to_pylist=lambda: rows)

    import pyarrow.parquet as pq

    monkeypatch.setattr(pq, "read_table", fake_read_table)
    assert extractor._read_final_d_rows(episode) == rows  # noqa: SLF001
    assert calls == [
        (
            inputs.dataset_root / "data" / "chunk-000" / f"episode_{episode.episode_index:06d}.parquet",
            producer.PARQUET_COLUMNS,
            [("frame_index", ">=", episode.d_start), ("frame_index", "<=", episode.d_end)],
        )
    ]

    forbidden = inputs.manifest.split("final_test")[0]
    foreign = dataclasses.replace(episode, stable_id=forbidden.stable_id, episode_index=forbidden.episode_index)
    with pytest.raises(producer.LeakageFeatureError, match="forbidden"):
        extractor._parquet_path(foreign)  # noqa: SLF001
    assert len(calls) == 1


def test_initialization_authenticates_actual_base_step0_and_all_data_hashes(tmp_path: Path) -> None:
    inputs = _fake_inputs(tmp_path)
    base_sha = "1" * 64
    step0_sha = "2" * 64
    graft = {
        "checkpoint_root": producer.OFFICIAL_BASE_URI,
        "format_version": 1,
        "loader": "AuditedPartialCheckpointWeightLoader",
        "tree_hashes": {"source_sha256": base_sha},
    }
    graft["manifest_sha256"] = producer._canonical_sha256(graft)  # noqa: SLF001
    inputs.graft_path.write_text(json.dumps(graft), encoding="utf-8")
    graft_file_sha = hashlib.sha256(inputs.graft_path.read_bytes()).hexdigest()
    identity = {
        "actual_step0_parameter_tree_sha256": step0_sha,
        "artifact_hashes": {
            "episode_manifest_sha256": inputs.manifest.sha256,
            "initialization_graft_manifest_file_sha256": graft_file_sha,
            "initialization_graft_manifest_self_sha256": graft["manifest_sha256"],
            "norm_stats_provenance_sha256": inputs.norm_provenance_sha256,
            "norm_stats_sha256": inputs.norm_stats_sha256,
            "train_storage_sha256": inputs.train_storage_sha256,
        },
        "config_name": producer.CONFIG_NAME,
        "format_version": 2,
        "graft_manifest_file": inputs.graft_path.name,
        "graft_manifest_sha256": graft["manifest_sha256"],
        "initialization_seed": producer.EXPECTED_INITIALIZATION_SEED,
        "official_source_uri": producer.OFFICIAL_BASE_URI,
        "source_tree_sha256": base_sha,
        "step0_checkpoint": 0,
    }
    identity["identity_sha256"] = producer._canonical_sha256(identity)  # noqa: SLF001
    inputs.initialization_path.write_text(json.dumps(identity), encoding="utf-8")

    validated, _, graft_path, _ = producer._validate_initialization(  # noqa: SLF001
        initialization_path=inputs.initialization_path,
        manifest=inputs.manifest,
        norm_stats_sha256=inputs.norm_stats_sha256,
        norm_provenance_sha256=inputs.norm_provenance_sha256,
        train_storage_sha256=inputs.train_storage_sha256,
        actual_step0_sha256=step0_sha,
        actual_official_base_sha256=base_sha,
    )
    assert validated["actual_step0_parameter_tree_sha256"] == step0_sha
    assert graft_path == inputs.graft_path

    with pytest.raises(producer.LeakageFeatureError, match="fresh v3.5 inputs"):
        producer._validate_initialization(  # noqa: SLF001
            initialization_path=inputs.initialization_path,
            manifest=inputs.manifest,
            norm_stats_sha256=inputs.norm_stats_sha256,
            norm_provenance_sha256=inputs.norm_provenance_sha256,
            train_storage_sha256=inputs.train_storage_sha256,
            actual_step0_sha256=step0_sha,
            actual_official_base_sha256="3" * 64,
        )


def test_emitted_artifact_loads_through_gate_reducer_without_absolute_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = _project_root(tmp_path, monkeypatch)
    inputs = _fake_inputs(tmp_path / "inputs")
    inputs.initialization_path.parent.mkdir(parents=True, exist_ok=True)
    identity = {
        "actual_step0_parameter_tree_sha256": inputs.initialization_parameter_tree_sha256,
        "artifact_hashes": {
            "episode_manifest_sha256": inputs.manifest.sha256,
            "norm_stats_sha256": inputs.norm_stats_sha256,
        },
        "config_name": producer.CONFIG_NAME,
        "format_version": 2,
        "initialization_seed": producer.EXPECTED_INITIALIZATION_SEED,
        "memory_inject_w_sha256": "6" * 64,
        "official_source_uri": producer.OFFICIAL_BASE_URI,
        "source_tree_sha256": inputs.official_base_source_tree_sha256,
        "step0_checkpoint": 0,
    }
    identity["identity_sha256"] = producer._canonical_sha256(identity)  # noqa: SLF001
    inputs.initialization_path.write_text(json.dumps(identity), encoding="utf-8")
    inputs.graft_path.write_text("{}", encoding="utf-8")
    inputs = dataclasses.replace(
        inputs,
        initialization_sha256=hashlib.sha256(inputs.initialization_path.read_bytes()).hexdigest(),
        graft_sha256=hashlib.sha256(inputs.graft_path.read_bytes()).hexdigest(),
    )
    output = root / "v35" / "diagnostics" / "gate_b" / "features.json"

    result = producer.emit_feature_artifacts(
        inputs=inputs,
        arrays=_arrays([episode.stable_id for episode in inputs.episodes]),
        output_envelope=output,
    )
    loaded = gate.load_feature_dataset(result, manifest=inputs.manifest)

    assert loaded.stable_ids == tuple(episode.stable_id for episode in inputs.manifest.split("train"))
    envelope = artifacts.load_canonical_envelope(result, schema_version=gate.FEATURE_SCHEMA_VERSION)
    serialized = result.read_text(encoding="utf-8")
    assert str(root) not in serialized
    assert envelope["payload"]["feature_protocol_sha256"] == producer.FROZEN_FEATURE_PROTOCOL_SHA256
    assert envelope["payload"]["development_or_final_test_accessed"] is False


def test_project_paths_reject_absolute_cli_inputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _project_root(tmp_path, monkeypatch)
    with pytest.raises(producer.LeakageFeatureError, match="project-relative"):
        producer._relative_project_path("/tmp/escape", name="test")  # noqa: SLF001
    with pytest.raises(producer.LeakageFeatureError, match="project-relative"):
        producer._relative_project_path("../escape", name="test")  # noqa: SLF001
