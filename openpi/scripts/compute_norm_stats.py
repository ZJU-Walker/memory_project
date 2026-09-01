"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

# ruff: noqa: I001 - data_loader (torch) must import before config (tensorflow), or the
# interpreter segfaults; same pinned order as data_loader_test.py.
from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import multiprocessing
import os
import pathlib
import re
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.shared.project_paths as project_paths
import openpi.training.data_loader as _data_loader
import openpi.training.config as _config
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


@dataclasses.dataclass(frozen=True)
class V35NormSelection:
    """Immutable provenance for the manifest-filtered v3.5 normalization rows."""

    manifest_path: pathlib.Path
    manifest_sha256: str
    schema_version: int
    split_seed: int
    active_split: str
    selected_indices: np.ndarray
    selected_episode_indices: tuple[int, ...]
    selected_stable_ids: tuple[str, ...]
    selected_episode_frame_counts: tuple[int, ...]
    excluded_episode_indices: tuple[int, ...]
    dataset_num_frames: int
    dataset_num_episodes: int
    dataset_episode_frame_protocol_sha256: str
    dataset_storage_root: pathlib.Path | None
    dataset_storage_files: tuple[dict[str, Any], ...]
    dataset_storage_sha256: str | None
    normalization_protocol: str = "transformed-dataset-v1"


class _V35RawNormBatches:
    """Finite batches from train-only scalar rows; never decodes an image column."""

    def __init__(
        self,
        parquet_paths: tuple[pathlib.Path, ...],
        episode_indices: tuple[int, ...],
        *,
        action_horizon: int,
        batch_size: int,
    ):
        if action_horizon <= 0:
            raise ValueError("v3.5 raw normalization requires a positive action horizon.")
        if batch_size <= 0:
            raise ValueError("v3.5 raw normalization requires a positive batch size.")
        states: list[np.ndarray] = []
        action_chunks: list[np.ndarray] = []
        delta_mask = np.asarray([True] * 6 + [False] + [True] * 6 + [False], dtype=bool)
        for episode_index in episode_indices:
            table = pq.read_table(parquet_paths[episode_index], columns=["state", "actions"])
            state = _fixed_size_list_numpy(table["state"], width=14)
            action = _fixed_size_list_numpy(table["actions"], width=14)
            if state.shape != action.shape or state.shape[0] == 0:
                raise ValueError(f"v3.5 norm episode {episode_index} has invalid state/action row shapes.")
            if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
                raise ValueError(f"v3.5 norm episode {episode_index} has non-finite state/action values.")
            offsets = np.minimum(
                np.arange(len(action), dtype=np.int64)[:, None] + np.arange(action_horizon, dtype=np.int64)[None, :],
                len(action) - 1,
            )
            chunk = action[offsets].copy()
            chunk[..., delta_mask] -= state[:, None, delta_mask]
            states.append(state)
            action_chunks.append(chunk)
        self._state = np.concatenate(states, axis=0)
        self._actions = np.concatenate(action_chunks, axis=0)
        self._batch_size = int(batch_size)
        if len(self._state) != len(self._actions):
            raise ValueError("v3.5 raw norm batches require aligned state/action rows.")

    def __iter__(self):
        for start in range(0, len(self._state), self._batch_size):
            stop = min(start + self._batch_size, len(self._state))
            yield {"state": self._state[start:stop], "actions": self._actions[start:stop]}

    def __len__(self) -> int:
        return (len(self._state) + self._batch_size - 1) // self._batch_size


class _IndexedDataset:
    """Read-only index view that prevents held-out base rows from reaching transforms."""

    def __init__(self, dataset: _data_loader.Dataset, indices: np.ndarray):
        self._dataset = dataset
        self._indices = np.asarray(indices, dtype=np.int64)

    def __getitem__(self, index):
        return self._dataset[int(self._indices[index.__index__()])]

    def __len__(self) -> int:
        return int(self._indices.size)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return _sha256_bytes(encoded)


def _fixed_size_list_numpy(column: Any, *, width: int) -> np.ndarray:
    array = column.combine_chunks()
    if getattr(array.type, "list_size", None) != width:
        raise ValueError(f"expected a fixed-size-list column of width {width}, found {array.type}.")
    values = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    return values.reshape(len(array), width)


def _normalize_sha256(value: str | None, *, name: str) -> str:
    if value is None:
        raise ValueError(f"{name} is required for v3.5 normalization (64 hexadecimal characters).")
    digest = value.removeprefix("sha256:").lower()
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{name} must be a SHA-256 digest (64 hexadecimal characters).")
    return digest


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"v3.5 manifest contains duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _read_hashed_manifest(
    path_value: str | None, expected_sha256: str | None
) -> tuple[dict[str, Any], pathlib.Path, str]:
    if path_value is None:
        raise ValueError("v3.5 normalization requires memory_episode_manifest_path.")
    path = pathlib.Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"v3.5 normalization manifest does not exist: {path}")
    expected = _normalize_sha256(expected_sha256, name="manifest_sha256")
    payload = path.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != expected:
        raise ValueError(f"v3.5 normalization manifest SHA-256 mismatch: expected {expected}, found {actual}.")
    try:
        raw = json.loads(payload, object_pairs_hook=_strict_json_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"v3.5 normalization manifest is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("v3.5 normalization manifest must be a JSON object.")
    return raw, path, actual


def _unwrap_episode_column(dataset: _data_loader.Dataset) -> np.ndarray:
    """Return the raw per-row episode index without relying on transformed samples."""

    inner: Any = dataset
    visited: set[int] = set()
    while not hasattr(inner, "hf_dataset"):
        identity = id(inner)
        if identity in visited or not hasattr(inner, "_dataset"):
            raise ValueError("v3.5 normalization could not locate the LeRobot episode_index column.")
        visited.add(identity)
        inner = inner._dataset  # noqa: SLF001
    columns = inner.hf_dataset.with_format(None)
    if "episode_index" not in columns.column_names:
        raise ValueError("v3.5 normalization dataset is missing the episode_index column.")
    episode = np.asarray(columns["episode_index"])
    if episode.ndim != 1 or len(episode) != len(dataset):
        raise ValueError(
            "v3.5 normalization episode_index must be one scalar per dataset row; "
            f"got shape={episode.shape}, rows={len(dataset)}."
        )
    if not np.issubdtype(episode.dtype, np.integer):
        raise ValueError(f"v3.5 normalization episode_index must be integer, got {episode.dtype}.")
    episode = episode.astype(np.int64, copy=False)
    if np.any(episode < 0):
        raise ValueError("v3.5 normalization episode_index contains a negative value.")
    return episode


def _unwrap_lerobot_root(dataset: _data_loader.Dataset) -> pathlib.Path:
    inner: Any = dataset
    visited: set[int] = set()
    while not hasattr(inner, "root"):
        identity = id(inner)
        if identity in visited or not hasattr(inner, "_dataset"):
            raise ValueError("v3.5 normalization could not locate the LeRobot dataset root.")
        visited.add(identity)
        inner = inner._dataset  # noqa: SLF001
    root = pathlib.Path(inner.root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"v3.5 LeRobot dataset root does not exist: {root}.")
    return root


def _stream_file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _v35_training_storage_seal(
    dataset: _data_loader.Dataset | pathlib.Path, selected_episode_indices: tuple[int, ...]
) -> tuple[pathlib.Path, tuple[dict[str, Any], ...], str]:
    """Hash train-only parquet/optional videos and structural metadata.

    Held-out episode storage is enumerated for membership only and is never opened.  The
    portable transfer bundle is deliberately outside this seal, so adding it after norm-stat
    computation cannot invalidate the authenticated training dataset.
    """
    root = dataset.expanduser().resolve() if isinstance(dataset, pathlib.Path) else _unwrap_lerobot_root(dataset)
    if not root.is_dir():
        raise ValueError(f"v3.5 LeRobot dataset root does not exist: {root}.")
    selected = set(selected_episode_indices)
    episode_pattern = re.compile(r"episode_(\d{6})\.(?:parquet|mp4)$")
    files: list[pathlib.Path] = []
    data_seen: set[int] = set()
    video_seen: set[int] = set()
    storage_directories = ["data"]
    if (root / "videos").is_dir():
        storage_directories.append("videos")
    for directory_name in storage_directories:
        directory = root / directory_name
        if not directory.is_dir():
            raise ValueError(f"v3.5 dataset is missing required {directory_name}/ storage under {root}.")
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            match = episode_pattern.search(path.name)
            if match is None:
                raise ValueError(f"v3.5 dataset has an unrecognized episode storage filename: {path}.")
            episode_index = int(match.group(1))
            if episode_index not in selected:
                continue
            files.append(path)
            (data_seen if directory_name == "data" else video_seen).add(episode_index)
    missing_data = sorted(selected - data_seen)
    missing_video = sorted(selected - video_seen) if "videos" in storage_directories else []
    if missing_data or missing_video:
        raise ValueError(
            "v3.5 train-only storage seal cannot find every selected episode: "
            f"missing_data={missing_data}, missing_video={missing_video}."
        )
    meta = root / "meta"
    if not meta.is_dir():
        raise ValueError(f"v3.5 dataset is missing structural meta/ storage under {root}.")
    files.extend(
        path for path in meta.rglob("*") if path.is_file() and "v35_training_bundle" not in path.relative_to(meta).parts
    )

    records: list[dict[str, Any]] = []
    for path in sorted(set(files), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        records.append({"path": relative, "size": path.stat().st_size, "sha256": _stream_file_sha256(path)})
    aggregate = _canonical_sha256(records)
    return root, tuple(records), aggregate


def _manifest_split_seed(raw: Mapping[str, Any]) -> Any:
    seed = raw.get("split_seed")
    if seed is None and isinstance(raw.get("split"), Mapping):
        seed = raw["split"].get("seed")
    return seed


def _v35_norm_selection(
    dataset: _data_loader.Dataset,
    data_config: _config.DataConfig,
    *,
    manifest_sha256: str | None,
) -> V35NormSelection:
    """Validate the frozen manifest and select only included training episode rows."""

    if data_config.memory_manifest_split != "train":
        raise ValueError(
            "v3.5 normalization is train-only: memory_manifest_split must be exactly 'train', "
            f"got {data_config.memory_manifest_split!r}."
        )
    raw, manifest_path, actual_sha256 = _read_hashed_manifest(data_config.memory_episode_manifest_path, manifest_sha256)
    schema_version = raw.get("schema_version")
    if type(schema_version) is not int or schema_version <= 0:
        raise ValueError("v3.5 normalization manifest requires a positive integer schema_version.")
    expected_seed = data_config.memory_manifest_split_seed
    split_seed = _manifest_split_seed(raw)
    if type(expected_seed) is not int or split_seed != expected_seed:
        raise ValueError(
            f"v3.5 normalization manifest split_seed mismatch: expected {expected_seed!r}, found {split_seed!r}."
        )
    entries = raw.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ValueError("v3.5 normalization manifest requires a non-empty episodes list.")

    records: dict[int, tuple[str, str, bool]] = {}
    stable_ids: set[str] = set()
    for offset, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"v3.5 normalization manifest episode {offset} must be an object.")
        stable_id = entry.get("stable_id")
        if not isinstance(stable_id, str) or not stable_id.strip():
            raise ValueError(f"v3.5 normalization manifest episode {offset} has invalid stable_id.")
        stable_id = stable_id.strip()
        if stable_id in stable_ids:
            raise ValueError(f"v3.5 normalization manifest has duplicate stable_id {stable_id!r}.")
        stable_ids.add(stable_id)
        include = entry.get("include", True)
        if type(include) is not bool:
            raise ValueError(f"v3.5 normalization manifest episode {stable_id!r} has non-boolean include.")
        episode_index = entry.get("episode_index", entry.get("lerobot_episode_index"))
        if episode_index is None:
            if include:
                raise ValueError(
                    f"included v3.5 normalization manifest episode {stable_id!r} is missing episode_index."
                )
            continue
        if type(episode_index) is not int or episode_index < 0:
            raise ValueError(f"v3.5 normalization manifest episode {stable_id!r} has invalid episode_index.")
        if episode_index in records:
            raise ValueError(f"v3.5 normalization manifest has duplicate episode_index {episode_index}.")
        split = entry.get("split")
        if not isinstance(split, str) or not split.strip():
            raise ValueError(f"v3.5 normalization manifest episode {stable_id!r} has invalid split.")
        records[episode_index] = (stable_id, split.strip(), include)

    episode = _unwrap_episode_column(dataset)
    dataset_episode_indices = np.unique(episode)
    expected_episode_indices = np.arange(len(dataset_episode_indices), dtype=np.int64)
    if not np.array_equal(dataset_episode_indices, expected_episode_indices):
        raise ValueError(
            "v3.5 normalization dataset episode indices must be contiguous from zero; "
            f"found {dataset_episode_indices.tolist()}."
        )
    if set(records) != set(dataset_episode_indices.tolist()):
        missing = sorted(set(dataset_episode_indices.tolist()) - set(records))
        extra = sorted(set(records) - set(dataset_episode_indices.tolist()))
        raise ValueError(f"v3.5 normalization manifest/dataset episode mismatch: missing={missing}, extra={extra}.")

    active_episodes = tuple(index for index in sorted(records) if records[index][2] and records[index][1] == "train")
    if not active_episodes:
        raise ValueError("v3.5 normalization manifest has no included training episodes.")
    active = np.asarray(active_episodes, dtype=np.int64)
    selected_mask = np.isin(episode, active)
    selected_indices = np.flatnonzero(selected_mask).astype(np.int64)
    if selected_indices.size == 0:
        raise ValueError("v3.5 normalization selected zero training frames.")
    selected_counts = tuple(int(np.count_nonzero(episode == index)) for index in active_episodes)
    if any(count <= 0 for count in selected_counts):
        raise ValueError("v3.5 normalization found a selected training episode with zero dataset frames.")

    episode_protocol = [
        {
            "episode_index": index,
            "stable_id": records[index][0],
            "split": records[index][1],
            "include": records[index][2],
            "frame_count": int(np.count_nonzero(episode == index)),
        }
        for index in sorted(records)
    ]
    storage_root = None
    storage_files: tuple[dict[str, Any], ...] = ()
    storage_sha256 = None
    if bool(getattr(data_config, "memory_v35_frozen_population", False)):
        storage_root, storage_files, storage_sha256 = _v35_training_storage_seal(dataset, active_episodes)
    dataset_protocol_sha256 = _canonical_sha256(
        {
            "manifest_sha256": actual_sha256,
            "episodes": episode_protocol,
            "train_storage_sha256": storage_sha256,
        }
    )
    return V35NormSelection(
        manifest_path=manifest_path,
        manifest_sha256=actual_sha256,
        schema_version=schema_version,
        split_seed=expected_seed,
        active_split="train",
        selected_indices=selected_indices,
        selected_episode_indices=active_episodes,
        selected_stable_ids=tuple(records[index][0] for index in active_episodes),
        selected_episode_frame_counts=selected_counts,
        excluded_episode_indices=tuple(index for index in sorted(records) if index not in active_episodes),
        dataset_num_frames=len(dataset),
        dataset_num_episodes=len(records),
        dataset_episode_frame_protocol_sha256=dataset_protocol_sha256,
        dataset_storage_root=storage_root,
        dataset_storage_files=storage_files,
        dataset_storage_sha256=storage_sha256,
    )


def create_v35_raw_norm_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    manifest_sha256: str | None,
    max_frames: int | None = None,
) -> tuple[_V35RawNormBatches, int, V35NormSelection]:
    """Build train-only norm batches from numeric parquet columns, never observations.

    The normal memory dataset expands every base row into up to 40 image-bearing sequence
    steps.  That is the correct training sample, but it is the wrong and prohibitively costly
    way to define raw-row normalization.  This path authenticates the same frozen manifest and
    labels, reads structural task/episode columns for all 70 episodes, then reads state/actions
    only for the 54 training episodes.
    """
    if data_config.repo_id is None:
        raise ValueError("v3.5 raw normalization requires a LeRobot repo_id.")
    if data_config.memory_manifest_split != "train":
        raise ValueError("v3.5 raw normalization requires memory_manifest_split='train'.")
    raw_manifest, manifest_path, actual_manifest_sha = _read_hashed_manifest(
        data_config.memory_episode_manifest_path,
        manifest_sha256,
    )
    root_value = data_config.lerobot_dataset_root
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("v3.5 raw normalization requires an explicit project-local lerobot_dataset_root.")
    root = pathlib.Path(root_value).expanduser().resolve()
    expected_root = project_paths.project_path(project_paths.V35_DATASET_DIR)
    if root != expected_root:
        raise ValueError(
            "v3.5 raw normalization dataset root is outside the portable project contract: "
            f"expected {expected_root}, found {root}."
        )
    if not root.is_dir():
        raise ValueError(f"v3.5 LeRobot dataset root does not exist: {root}.")
    parquet_by_episode: dict[int, pathlib.Path] = {}
    pattern = re.compile(r"episode_(\d{6})\.parquet$")
    for path in (root / "data").rglob("*.parquet"):
        match = pattern.fullmatch(path.name)
        if match is None:
            raise ValueError(f"v3.5 dataset has an unrecognized parquet filename: {path}.")
        episode_index = int(match.group(1))
        if episode_index in parquet_by_episode:
            raise ValueError(f"v3.5 dataset has duplicate parquet episode {episode_index}.")
        parquet_by_episode[episode_index] = path
    num_episodes = len(parquet_by_episode)
    if set(parquet_by_episode) != set(range(num_episodes)):
        raise ValueError("v3.5 parquet episode indices must be contiguous from zero.")

    tasks: dict[int, str] = {}
    for line in (root / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            entry = json.loads(line)
            tasks[int(entry["task_index"])] = str(entry["task"])
    if sorted(tasks) != list(range(len(tasks))):
        raise ValueError("v3.5 tasks.jsonl indices must be contiguous from zero.")
    prompt_entries = json.loads((root / "meta" / "episode_prompts.json").read_text(encoding="utf-8"))
    prompts = tuple(str(prompt_entries.get(str(index), "")) for index in range(num_episodes))

    lengths = np.zeros(num_episodes, dtype=np.int32)
    sides = np.full(num_episodes, -1, dtype=np.int32)
    episode_tasks: list[np.ndarray] = []
    for episode_index in range(num_episodes):
        table = pq.read_table(parquet_by_episode[episode_index], columns=["episode_index", "task_index"])
        episode_column = np.asarray(table["episode_index"].combine_chunks().to_numpy(zero_copy_only=False))
        task_column = np.asarray(
            table["task_index"].combine_chunks().to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        if len(task_column) == 0 or not np.all(episode_column == episode_index):
            raise ValueError(f"v3.5 parquet episode column mismatch for episode {episode_index}.")
        if np.any(task_column < 0) or np.any(task_column >= len(tasks)):
            raise ValueError(f"v3.5 parquet task index is out of range in episode {episode_index}.")
        lengths[episode_index] = len(task_column)
        episode_tasks.append(task_column)
        final_task = tasks[int(task_column[-1])]
        sides[episode_index] = 0 if "left" in final_task else 1 if "right" in final_task else -1

    manifest_info = _data_loader._load_v35_episode_manifest(  # noqa: SLF001
        data_config,
        num_episodes=num_episodes,
        side=sides,
        episode_length=lengths,
        episode_tasks=tuple(episode_tasks),
        tasks=tasks,
        prompts=prompts,
    )
    sampling_allowed = np.asarray(manifest_info["sampling_allowed"], dtype=bool)
    selected_episode_indices = tuple(int(index) for index in np.flatnonzero(sampling_allowed))
    if len(selected_episode_indices) != 54:
        raise ValueError(
            f"v3.5 raw normalization requires exactly 54 train episodes, found {len(selected_episode_indices)}."
        )
    selected_counts = tuple(int(lengths[index]) for index in selected_episode_indices)
    selected_num_frames = sum(selected_counts)
    if max_frames is not None and max_frames < selected_num_frames:
        raise ValueError(
            "v3.5 normalization must cover every train row; "
            f"max_frames={max_frames} is less than {selected_num_frames}."
        )

    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)))
    selected_indices = np.concatenate(
        [np.arange(offsets[index], offsets[index + 1], dtype=np.int64) for index in selected_episode_indices]
    )
    storage_root, storage_files, storage_sha256 = _v35_training_storage_seal(root, selected_episode_indices)
    episode_protocol = [
        {
            "episode_index": index,
            "stable_id": manifest_info["stable_id"][index],
            "split": manifest_info["manifest_split"][index],
            "include": True,
            "frame_count": int(lengths[index]),
        }
        for index in range(num_episodes)
    ]
    dataset_protocol_sha256 = _canonical_sha256(
        {
            "manifest_sha256": actual_manifest_sha,
            "episodes": episode_protocol,
            "train_storage_sha256": storage_sha256,
        }
    )
    selection = V35NormSelection(
        manifest_path=manifest_path,
        manifest_sha256=actual_manifest_sha,
        schema_version=int(raw_manifest["schema_version"]),
        split_seed=int(data_config.memory_manifest_split_seed),
        active_split="train",
        selected_indices=selected_indices,
        selected_episode_indices=selected_episode_indices,
        selected_stable_ids=tuple(manifest_info["stable_id"][index] for index in selected_episode_indices),
        selected_episode_frame_counts=selected_counts,
        excluded_episode_indices=tuple(int(index) for index in np.flatnonzero(~sampling_allowed)),
        dataset_num_frames=int(lengths.sum()),
        dataset_num_episodes=num_episodes,
        dataset_episode_frame_protocol_sha256=dataset_protocol_sha256,
        dataset_storage_root=storage_root,
        dataset_storage_files=storage_files,
        dataset_storage_sha256=storage_sha256,
        normalization_protocol="raw-train-rows-delta-action-horizon-v1",
    )
    paths = tuple(parquet_by_episode[index] for index in range(num_episodes))
    batches = _V35RawNormBatches(
        paths,
        selected_episode_indices,
        action_horizon=action_horizon,
        batch_size=batch_size,
    )
    if len(batches) != (selected_num_frames + batch_size - 1) // batch_size:
        raise AssertionError("v3.5 raw norm batch count is inconsistent with selected rows.")
    return batches, len(batches), selection


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    manifest_sha256: str | None = None,
) -> tuple[_data_loader.Dataset, int, V35NormSelection | None]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    if data_config.memory_v35_enabled:
        if data_config.memory_manifest_split != "train":
            raise ValueError(
                "v3.5 normalization is train-only: memory_manifest_split must be exactly 'train', "
                f"got {data_config.memory_manifest_split!r}."
            )
        # Authenticate the exact manifest before create_torch_dataset is allowed to consume it
        # for sequence metadata. _v35_norm_selection reads and checks it again afterwards.
        _read_hashed_manifest(data_config.memory_episode_manifest_path, manifest_sha256)
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    selection = None
    if data_config.memory_v35_enabled:
        selection = _v35_norm_selection(dataset, data_config, manifest_sha256=manifest_sha256)
        dataset = _IndexedDataset(dataset, selection.selected_indices)
        if max_frames is not None and max_frames < len(dataset):
            raise ValueError(
                "v3.5 normalization must cover the complete manifest-selected training frame set; "
                f"max_frames={max_frames} is less than {len(dataset)}."
            )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if selection is not None:
        # Norm stats are a finite reduction, so keep the final partial batch instead of
        # silently dropping its rows (the training loader intentionally drops it).
        multiprocessing_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
        generator = torch.Generator()
        generator.manual_seed(0)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing_context,
            persistent_workers=num_workers > 0,
            collate_fn=_data_loader._collate_fn,  # noqa: SLF001
            worker_init_fn=_data_loader._worker_init_fn,  # noqa: SLF001
            generator=generator,
        )
        return data_loader, (len(dataset) + batch_size - 1) // batch_size, selection
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches, selection


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _compute_stats(data_loader, num_batches: int) -> dict[str, normalize.NormStats]:
    keys = ("state", "actions")
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))
    return {key: stats.get_statistics() for key, stats in stats.items()}


def _manifest_is_unchanged(selection: V35NormSelection) -> None:
    if not selection.manifest_path.is_file():
        raise ValueError(f"v3.5 normalization manifest disappeared during computation: {selection.manifest_path}")
    final_sha256 = _sha256_bytes(selection.manifest_path.read_bytes())
    if final_sha256 != selection.manifest_sha256:
        raise ValueError(
            "v3.5 normalization manifest changed during computation: "
            f"started at {selection.manifest_sha256}, ended at {final_sha256}."
        )


def _save_v35_stats_with_provenance(
    output_path: pathlib.Path,
    norm_stats: dict[str, normalize.NormStats],
    selection: V35NormSelection,
    *,
    repo_id: str,
    batch_size: int,
    num_batches: int,
) -> None:
    """Commit stats plus their train-only provenance, with provenance written last."""

    _manifest_is_unchanged(selection)
    norm_payload = normalize.serialize_json(norm_stats) + "\n"
    norm_sha256 = _sha256_bytes(norm_payload.encode())
    provenance = {
        "schema_version": 2,
        "status": "complete",
        "scope": "v3.5 manifest-selected included training episodes only",
        "repo_id": repo_id,
        "manifest": {
            "path_relative": project_paths.project_relative_path(selection.manifest_path).as_posix(),
            "sha256": selection.manifest_sha256,
            "schema_version": selection.schema_version,
            "split_seed": selection.split_seed,
            "active_split": selection.active_split,
        },
        "selection": {
            "dataset_num_frames": selection.dataset_num_frames,
            "dataset_num_episodes": selection.dataset_num_episodes,
            "selected_num_frames": int(selection.selected_indices.size),
            "selected_num_episodes": len(selection.selected_episode_indices),
            "selected_episode_indices": list(selection.selected_episode_indices),
            "selected_stable_ids": list(selection.selected_stable_ids),
            "selected_episode_frame_counts": list(selection.selected_episode_frame_counts),
            "excluded_episode_indices": list(selection.excluded_episode_indices),
            "dataset_episode_frame_protocol_sha256": selection.dataset_episode_frame_protocol_sha256,
        },
        "train_storage": (
            None
            if selection.dataset_storage_sha256 is None
            else {
                "root_contract": "memory_project-relative-v1",
                "root_relative": project_paths.project_relative_path(selection.dataset_storage_root).as_posix(),
                "sha256": selection.dataset_storage_sha256,
                "files": list(selection.dataset_storage_files),
                "selected_episode_indices": list(selection.selected_episode_indices),
                "scope": "selected train episode parquet, optional videos, plus structural meta files",
            }
        ),
        "computation": {
            "protocol": selection.normalization_protocol,
            "requested_batch_size": batch_size,
            "num_batches_including_partial_final_batch": num_batches,
            "processed_base_rows": int(selection.selected_indices.size),
            "drop_last_rows": 0,
        },
        "norm_stats": {"file": "norm_stats.json", "sha256": norm_sha256},
    }
    provenance_payload = json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    output_path.mkdir(parents=True, exist_ok=True)
    norm_tmp = output_path / f".norm_stats.json.tmp.{os.getpid()}"
    provenance_tmp = output_path / f".norm_stats_provenance.json.tmp.{os.getpid()}"
    norm_tmp.write_text(norm_payload, encoding="utf-8")
    provenance_tmp.write_text(provenance_payload, encoding="utf-8")
    os.replace(norm_tmp, output_path / "norm_stats.json")
    # This file is the commit marker: consumers can hash-check the already-installed stats.
    os.replace(provenance_tmp, output_path / "norm_stats_provenance.json")


def _norm_output_path(config, data_config, *, v35: bool) -> pathlib.Path:
    if not v35:
        return pathlib.Path(config.assets_dirs) / data_config.repo_id
    asset_id = config.data.assets.asset_id or data_config.repo_id
    return pathlib.Path(config.data.assets.assets_dir or config.assets_dirs) / asset_id


def main(config_name: str, max_frames: int | None = None, manifest_sha256: str | None = None):
    config = _config.get_config(config_name)
    if getattr(config.model, "memory_v35_enabled", False):
        project_paths.configure_v35_runtime_environment()
    registered_base = config.data.base_config
    raw_v35 = bool(
        registered_base is not None
        and registered_base.memory_v35_enabled
        and registered_base.memory_v35_frozen_population
    )
    # The frozen v3.5 protocol reads only raw numeric parquet columns. Building the full model
    # transform stack here would initialize remote-code tokenizers that are neither used nor
    # part of the norm definition, creating an unnecessary external cache dependency.
    data_config = (
        config.data.create_base_config(config.assets_dirs, config.model)
        if raw_v35
        else config.data.create(config.assets_dirs, config.model)
    )

    if raw_v35:
        data_loader, num_batches, selection = create_v35_raw_norm_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            manifest_sha256=manifest_sha256,
            max_frames=max_frames,
        )
    elif data_config.rlds_data_dir is not None:
        if data_config.memory_v35_enabled:
            raise ValueError("v3.5 manifest-filtered normalization currently requires a LeRobot dataset.")
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
        selection = None
    else:
        data_loader, num_batches, selection = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
            manifest_sha256,
        )

    norm_stats = _compute_stats(data_loader, num_batches)

    # v3.5 writes exactly where DataConfigFactory.create_base_config will read; legacy keeps
    # the historical per-config destination.
    output_path = _norm_output_path(config, data_config, v35=selection is not None)
    print(f"Writing stats to: {output_path}")
    if selection is None:
        normalize.save(output_path, norm_stats)
    else:
        _save_v35_stats_with_provenance(
            pathlib.Path(output_path),
            norm_stats,
            selection,
            repo_id=data_config.repo_id,
            batch_size=config.batch_size,
            num_batches=num_batches,
        )


if __name__ == "__main__":
    tyro.cli(main)
