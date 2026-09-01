"""Focused v3.5 train-only normalization leakage tests."""

from __future__ import annotations

# This test intentionally exercises script-private fail-closed helpers.
# ruff: noqa: SLF001
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pytest

_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS))
try:
    import compute_norm_stats as norm_script
finally:
    sys.path.remove(str(_SCRIPTS))


class _Columns:
    column_names = ("episode_index",)

    def __init__(self, episodes: np.ndarray):
        self._episodes = episodes

    def with_format(self, _format):
        return self

    def __getitem__(self, key: str):
        assert key == "episode_index"
        return self._episodes


class _RawDataset:
    def __init__(self, episodes: list[int], values: list[float], root: Path | None = None):
        self._episodes = np.asarray(episodes, dtype=np.int64)
        self._values = np.asarray(values, dtype=np.float32)
        self.hf_dataset = _Columns(self._episodes)
        self.root = root
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return len(self._episodes)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        self.accessed.append(index)
        value = self._values[index]
        return {
            "state": np.asarray([value], dtype=np.float32),
            "actions": np.asarray([[value]], dtype=np.float32),
        }


def _data_config(manifest: Path, *, split: str = "train", enabled: bool = True):
    empty_group = SimpleNamespace(inputs=())
    return SimpleNamespace(
        repo_id="local/v35_fixture",
        memory_v35_enabled=enabled,
        memory_episode_manifest_path=str(manifest),
        memory_manifest_split=split,
        memory_manifest_split_seed=35,
        repack_transforms=empty_group,
        data_transforms=empty_group,
    )


def _write_manifest(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "split_seed": 36,
        "episodes": [
            {"episode_index": 0, "stable_id": "train/a", "include": True, "split": "train"},
            {"episode_index": 1, "stable_id": "sealed/b", "include": True, "split": "final_test"},
            {"episode_index": 2, "stable_id": "train/c", "include": True, "split": "train"},
        ],
    }
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode()).hexdigest()


def _project_root(monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / "memory_project"
    (root / "openpi/src/openpi").mkdir(parents=True)
    (root / "openpi/pyproject.toml").touch()
    monkeypatch.setenv("MEMORY_PROJECT_ROOT", str(root))
    return root


def _compute_fixture(monkeypatch, tmp_path: Path, heldout_value: float):
    manifest = _project_root(monkeypatch, tmp_path) / "data/episodes.json"
    digest = _write_manifest(manifest)
    dataset = _RawDataset(
        episodes=[0, 0, 1, 1, 2, 2],
        values=[0.0, 2.0, heldout_value, heldout_value, 4.0, 6.0],
    )
    monkeypatch.setattr(norm_script._data_loader, "create_torch_dataset", lambda *_args: dataset)
    loader, num_batches, selection = norm_script.create_torch_dataloader(
        _data_config(manifest),
        action_horizon=1,
        batch_size=3,
        model_config=SimpleNamespace(),
        num_workers=0,
        manifest_sha256=digest,
    )
    stats = norm_script._compute_stats(loader, num_batches)
    return dataset, selection, stats


def test_v35_stats_subset_excludes_every_heldout_row(monkeypatch, tmp_path: Path) -> None:
    dataset, selection, stats = _compute_fixture(monkeypatch, tmp_path, heldout_value=1_000_000.0)

    assert selection is not None
    assert selection.selected_episode_indices == (0, 2)
    assert selection.selected_stable_ids == ("train/a", "train/c")
    assert selection.selected_indices.tolist() == [0, 1, 4, 5]
    assert set(dataset.accessed) == {0, 1, 4, 5}
    assert not ({2, 3} & set(dataset.accessed))
    np.testing.assert_allclose(stats["state"].mean, [3.0])
    np.testing.assert_allclose(stats["actions"].mean, [3.0])


def test_heldout_values_have_exactly_zero_effect_on_v35_stats(monkeypatch, tmp_path: Path) -> None:
    _dataset, _selection, high = _compute_fixture(monkeypatch, tmp_path / "high", heldout_value=1_000_000.0)
    _dataset, _selection, low = _compute_fixture(monkeypatch, tmp_path / "low", heldout_value=-1_000_000.0)

    for key in ("state", "actions"):
        np.testing.assert_array_equal(high[key].mean, low[key].mean)
        np.testing.assert_array_equal(high[key].std, low[key].std)
        np.testing.assert_array_equal(high[key].q01, low[key].q01)
        np.testing.assert_array_equal(high[key].q99, low[key].q99)


def test_v35_manifest_hash_and_train_split_fail_closed(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "episodes.json"
    digest = _write_manifest(manifest)
    dataset = _RawDataset([0, 1, 2], [0.0, 1.0, 2.0])
    monkeypatch.setattr(norm_script._data_loader, "create_torch_dataset", lambda *_args: dataset)

    with pytest.raises(ValueError, match="manifest_sha256 is required"):
        norm_script.create_torch_dataloader(_data_config(manifest), 1, 1, SimpleNamespace(), 0, manifest_sha256=None)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        norm_script.create_torch_dataloader(
            _data_config(manifest), 1, 1, SimpleNamespace(), 0, manifest_sha256="0" * 64
        )
    with pytest.raises(ValueError, match="must be exactly 'train'"):
        norm_script.create_torch_dataloader(
            _data_config(manifest, split="final_test"), 1, 1, SimpleNamespace(), 0, manifest_sha256=digest
        )


def test_manifest_mutation_after_selection_fails_before_commit(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "episodes.json"
    digest = _write_manifest(manifest)
    dataset = _RawDataset([0, 1, 2], [0.0, 1.0, 2.0])
    monkeypatch.setattr(norm_script._data_loader, "create_torch_dataset", lambda *_args: dataset)
    _loader, _batches, selection = norm_script.create_torch_dataloader(
        _data_config(manifest), 1, 1, SimpleNamespace(), 0, manifest_sha256=digest
    )
    assert selection is not None

    manifest.write_text(manifest.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="changed during computation"):
        norm_script._manifest_is_unchanged(selection)


def test_v35_stats_commit_binds_manifest_selection_and_stats_hash(monkeypatch, tmp_path: Path) -> None:
    _dataset, selection, stats = _compute_fixture(monkeypatch, tmp_path / "fixture", heldout_value=999.0)
    assert selection is not None
    output = tmp_path / "assets"

    norm_script._save_v35_stats_with_provenance(
        output,
        stats,
        selection,
        repo_id="local/v35_fixture",
        batch_size=3,
        num_batches=2,
    )

    provenance = json.loads((output / "norm_stats_provenance.json").read_text())
    stats_payload = (output / "norm_stats.json").read_bytes()
    assert provenance["manifest"]["sha256"] == selection.manifest_sha256
    assert provenance["manifest"]["active_split"] == "train"
    assert provenance["selection"]["selected_num_episodes"] == 2
    assert provenance["selection"]["selected_num_frames"] == 4
    assert provenance["computation"]["processed_base_rows"] == 4
    assert provenance["computation"]["drop_last_rows"] == 0
    assert provenance["norm_stats"]["sha256"] == hashlib.sha256(stats_payload).hexdigest()


def test_legacy_path_does_not_require_manifest_hash(monkeypatch, tmp_path: Path) -> None:
    manifest = tmp_path / "unused.json"
    dataset = _RawDataset([0, 1], [1.0, 3.0])
    monkeypatch.setattr(norm_script._data_loader, "create_torch_dataset", lambda *_args: dataset)

    loader, num_batches, selection = norm_script.create_torch_dataloader(
        _data_config(manifest, enabled=False), 1, 2, SimpleNamespace(), 0
    )
    stats = norm_script._compute_stats(loader, num_batches)

    assert selection is None
    np.testing.assert_allclose(stats["state"].mean, [2.0])


def test_v35_output_path_matches_data_factory_assets_location(tmp_path: Path) -> None:
    config = SimpleNamespace(
        assets_dirs=tmp_path / "legacy",
        data=SimpleNamespace(assets=SimpleNamespace(assets_dir=str(tmp_path / "v35"), asset_id=None)),
    )
    data_config = SimpleNamespace(repo_id="yam/v35")

    assert norm_script._norm_output_path(config, data_config, v35=True) == tmp_path / "v35" / "yam/v35"
    assert norm_script._norm_output_path(config, data_config, v35=False) == tmp_path / "legacy" / "yam/v35"


def test_train_storage_seal_hashes_train_files_without_reading_final_video(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "chunk-000" / "observation.images.top").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "meta" / "info.json").write_text('{"version": 1}')
    for episode_index, value in enumerate((b"train-a", b"sealed", b"train-c")):
        name = f"episode_{episode_index:06d}"
        (root / "data" / "chunk-000" / f"{name}.parquet").write_bytes(value + b"-data")
        (root / "videos" / "chunk-000" / "observation.images.top" / f"{name}.mp4").write_bytes(value + b"-video")
    dataset = _RawDataset([0, 1, 2], [0.0, 1.0, 2.0], root=root)

    _root, records, first = norm_script._v35_training_storage_seal(dataset, (0, 2))
    assert all("episode_000001" not in record["path"] for record in records)

    transfer_bundle = root / "meta" / "v35_training_bundle"
    transfer_bundle.mkdir()
    (transfer_bundle / "TRANSFER_MANIFEST.json").write_text('{"portable": true}')
    assert norm_script._v35_training_storage_seal(dataset, (0, 2))[2] == first

    final_video = root / "videos" / "chunk-000" / "observation.images.top" / "episode_000001.mp4"
    final_video.write_bytes(b"changed-final-observation")
    assert norm_script._v35_training_storage_seal(dataset, (0, 2))[2] == first

    train_video = root / "videos" / "chunk-000" / "observation.images.top" / "episode_000002.mp4"
    train_video.write_bytes(b"changed-training-observation")
    assert norm_script._v35_training_storage_seal(dataset, (0, 2))[2] != first


def test_train_storage_seal_accepts_images_embedded_in_parquet_without_videos(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "meta" / "info.json").write_text('{"embedded_images": true}')
    (data / "episode_000000.parquet").write_bytes(b"parquet-with-embedded-images")

    sealed_root, records, digest = norm_script._v35_training_storage_seal(root, (0,))

    assert sealed_root == root.resolve()
    assert [record["path"] for record in records] == [
        "data/chunk-000/episode_000000.parquet",
        "meta/info.json",
    ]
    assert len(digest) == 64


def test_raw_v35_norm_reads_numeric_columns_only_for_exact_train74(monkeypatch, tmp_path: Path) -> None:
    project_root = _project_root(monkeypatch, tmp_path)
    repo_id = norm_script.project_paths.V35_REPO_ID
    hf_root = project_root / "data/lerobot"
    root = hf_root / repo_id
    data = root / "data" / "chunk-000"
    meta = root / "meta"
    data.mkdir(parents=True)
    meta.mkdir()
    tasks = ("open left bin", "open right bin")
    (meta / "tasks.jsonl").write_text(
        "".join(json.dumps({"task_index": index, "task": task}) + "\n" for index, task in enumerate(tasks))
    )
    (meta / "episode_prompts.json").write_text(json.dumps({str(index): "find the object" for index in range(90)}))
    manifest = project_root / "data/manifest.json"
    manifest_payload = json.dumps({"schema_version": 2}) + "\n"
    manifest.write_text(manifest_payload)
    manifest_sha256 = hashlib.sha256(manifest_payload.encode()).hexdigest()

    vector_type = pa.list_(pa.float32(), 14)
    for episode_index in range(90):
        rows = 2 if episode_index == 0 else 1
        columns: dict[str, pa.Array] = {
            "episode_index": pa.array([episode_index] * rows, type=pa.int64()),
            "task_index": pa.array([episode_index % 2] * rows, type=pa.int64()),
            "image": pa.array([b"must-not-be-read"] * rows, type=pa.binary()),
        }
        if episode_index < 74:
            if episode_index == 0:
                states = [[10.0] * 14, [20.0] * 14]
                actions = [[11.0] * 14, [22.0] * 14]
            else:
                states = [[float(episode_index)] * 14]
                actions = [[float(episode_index + 1)] * 14]
            columns["state"] = pa.array(states, type=vector_type)
            columns["actions"] = pa.array(actions, type=vector_type)
        norm_script.pq.write_table(pa.table(columns), data / f"episode_{episode_index:06d}.parquet")

    allowed = np.zeros(90, dtype=bool)
    allowed[:74] = True
    manifest_info = {
        "sampling_allowed": allowed,
        "stable_id": tuple(f"episode/{index:03d}" for index in range(90)),
        "manifest_split": tuple("train" if index < 74 else "final_test" for index in range(90)),
    }
    monkeypatch.setattr(norm_script._data_loader, "_load_v35_episode_manifest", lambda *_args, **_kwargs: manifest_info)
    real_read_table = norm_script.pq.read_table
    reads: list[tuple[int, tuple[str, ...]]] = []

    def audited_read_table(path, *, columns):
        match = norm_script.re.search(r"episode_(\d{6})\.parquet$", str(path))
        assert match is not None
        reads.append((int(match.group(1)), tuple(columns)))
        return real_read_table(path, columns=columns)

    monkeypatch.setattr(norm_script.pq, "read_table", audited_read_table)
    config = SimpleNamespace(
        repo_id=repo_id,
        lerobot_dataset_root=str(root),
        memory_episode_manifest_path=str(manifest),
        memory_manifest_split="train",
        memory_manifest_split_seed=35,
    )

    batches, num_batches, selection = norm_script.create_v35_raw_norm_dataloader(
        config,
        action_horizon=3,
        batch_size=11,
        manifest_sha256=manifest_sha256,
    )

    assert selection.selected_episode_indices == tuple(range(74))
    assert selection.excluded_episode_indices == tuple(range(74, 90))
    assert selection.selected_episode_frame_counts == (2, *([1] * 73))
    assert selection.normalization_protocol == "raw-train-rows-delta-action-horizon-v1"
    assert len(batches) == num_batches == 7
    assert all("image" not in columns for _, columns in reads)
    numeric_reads = {episode for episode, columns in reads if columns == ("state", "actions")}
    assert numeric_reads == set(range(74))
    np.testing.assert_array_equal(batches._actions[0, :, 0], np.asarray([1.0, 12.0, 12.0]))
    np.testing.assert_array_equal(batches._actions[0, :, 6], np.asarray([11.0, 22.0, 22.0]))
