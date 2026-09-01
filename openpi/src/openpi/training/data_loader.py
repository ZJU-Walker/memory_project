import base64
from collections.abc import Iterator, Sequence
import dataclasses
import functools
import hashlib
import itertools
import json
import logging
import multiprocessing
import os
import pathlib
import random
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")

    def state_dict(self) -> dict[str, typing.Any]:
        """Return exact-resume state when the loader was created for that contract."""
        raise NotImplementedError("This data loader does not support exact resume.")

    def load_state_dict(self, state: dict[str, typing.Any]) -> None:
        """Restore exact-resume state before constructing the resumed iterator."""
        raise NotImplementedError("This data loader does not support exact resume.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        # Bool specs default to all-False and int specs to large randoms, which would leave the
        # memory plumbing untested on fake data: make all but the last step real, put one
        # gradient-block fence mid-sequence, and (when the quiz is enabled) quiz the newer half
        # with valid class labels.
        if observation.seq_step_mask is not None:
            t = observation.seq_step_mask.shape[0]
            step_mask = jnp.arange(t) < max(t - 1, 1)
            observation = dataclasses.replace(
                observation,
                seq_step_mask=step_mask,
                seq_block_boundary=(jnp.arange(t) == t // 2) & (t > 2),
            )
        if observation.seq_probe_labels is not None:
            t = observation.seq_probe_labels.shape[0]
            probe_mask = observation.seq_step_mask & (jnp.arange(t) >= t // 2)
            observation = dataclasses.replace(
                observation,
                seq_probe_labels=observation.seq_probe_labels % 2,
                seq_probe_mask=probe_mask,
                seq_probe_visible=probe_mask & (jnp.arange(t) < 3 * t // 4),
            )
        # v3.4 fields: keep fake labels inside their valid ranges and give the phase masks a
        # deterministic evidence-then-waiting shape so the aux/ladder paths execute.
        replacements = {}
        if observation.seq_subtask_class is not None:
            replacements["seq_subtask_class"] = observation.seq_subtask_class % 2
        if observation.seq_side_label is not None:
            t = observation.seq_step_mask.shape[0]
            replacements["seq_side_label"] = observation.seq_side_label % 2
            replacements["seq_evidence_mask"] = observation.seq_step_mask & (jnp.arange(t) < t // 2)
            replacements["seq_waiting_mask"] = observation.seq_step_mask & (jnp.arange(t) >= t // 2)
        if observation.seq_state_masked is not None:
            replacements["seq_state_masked"] = (jnp.asarray(index.__index__()) % 2).astype(bool)
        if observation.token_state_mask is not None:
            token_len = observation.token_state_mask.shape[-1]
            span = (jnp.arange(token_len) >= token_len // 2) & (jnp.arange(token_len) < token_len // 2 + 3)
            replacements["token_state_mask"] = jnp.broadcast_to(span, observation.token_state_mask.shape)
        if replacements:
            observation = dataclasses.replace(observation, **replacements)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    # v3.5 pins a project-local dataset root so copying ``memory_project`` to a
    # different machine cannot silently fall back to that machine's global HF
    # cache.  Legacy configs leave this unset and retain LeRobot's default.
    dataset_root = data_config.lerobot_dataset_root
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=dataset_root)
    use_memory = getattr(model_config, "predict_with_memory", False) and data_config.memory_stride_frames > 0
    if use_memory:
        # Memory sequence training: each sample is T consecutive prediction steps anchored at
        # the sampled base frame -- per-step images/state at (base + k*stride), the flat action
        # stream for all T chunks, and the per-step (lookahead-shifted) task_index for the
        # subtask labels. lerobot clamps offsets past the episode end by repeating the last
        # frame; BuildMemorySequence masks those steps out.
        steps = model_config.memory_seq_steps
        stride = data_config.memory_stride_frames
        step_offsets = [k * stride / dataset_meta.fps for k in range(steps)]
        delta_timestamps = {
            key: [(k * stride + j) / dataset_meta.fps for k in range(steps) for j in range(action_horizon)]
            for key in data_config.action_sequence_keys
        }
        for key in ("image", "left_wrist_image", "right_wrist_image", "state"):
            delta_timestamps[key] = step_offsets
        # NOTE: no task_index delta_timestamps -- lerobot requires a scalar task_index per item
        # (it .item()s it); the per-step subtask labels come from MemorySequenceSubtasks below.
    else:
        delta_timestamps = {
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        }
        if data_config.subtask_from_task and data_config.subtask_lookahead > 0:
            # Deliver the *future* task_index: the subtask label that conditions this frame's chunk.
            delta_timestamps["task_index"] = [data_config.subtask_lookahead / dataset_meta.fps]
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        root=dataset_root,
        delta_timestamps=delta_timestamps,
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    if data_config.prompt_from_episode_meta:
        prompts = _load_episode_prompts(dataset)
        if prompts is None:
            raise ValueError(
                f"prompt_from_episode_meta is set but {data_config.repo_id} has no meta/episode_prompts.json."
            )
        dataset = TransformedDataset(dataset, [_transforms.InjectPromptFromEpisode(prompts)])

    if data_config.subtask_from_task and not use_memory:
        dataset = TransformedDataset(dataset, [_transforms.SubtaskFromLeRobotTask(dataset_meta.tasks)])

    if use_memory:
        info = _episode_info_table(dataset, dataset_meta, data_config)
        quiz = getattr(model_config, "memory_probe_weight", 0) > 0 or getattr(
            model_config, "memory_probe_diagnostic", False
        )
        seq_transforms: list[_transforms.DataTransformFn] = []
        if data_config.subtask_from_task:
            seq_transforms.append(
                _transforms.MemorySequenceSubtasks(
                    stride=data_config.memory_stride_frames,
                    steps=model_config.memory_seq_steps,
                    lookahead=data_config.subtask_lookahead,
                    episode_tasks=info["episode_tasks"],
                    tasks=dataset_meta.tasks,
                    episode_waiting_valid=info.get("episode_waiting_valid", ()),
                )
            )
        seq_transforms.append(
            _transforms.MemoryEpisodeInfo(
                episode_length=info["length"],
                episode_side=info["side"] if quiz else None,
                episode_reveal=info["reveal"] if quiz else None,
                episode_close=info["close"] if quiz else None,
                episode_memory_window=_memory_critical_windows(info, data_config),
                # v3.4 ladder-probe side label; dropped at repack unless the config carries it.
                episode_side_label=info["side"],
            )
        )
        dataset = TransformedDataset(dataset, seq_transforms)

    return dataset


# Quiz defaults when no reveal-frames json is provided: the bin contents become visible between
# frames ~200 and ~300 in every bin_memory_banana episode, so 300 is the conservative
# "definitely revealed by now" bound; the bins are closed again by ~450.
_DEFAULT_REVEAL_FRAME = 300
_DEFAULT_VISIBLE_SPAN = 150


_V35_MANIFEST_SCHEMA_VERSION = 2
# v36 population (0830+0831): the leaking 0816 collection is excluded per the sealed Gate-B
# stop in v35/diagnostics/runs/v35_fresh_pilot_20260831_r7/gates/gate_b.json.
_V36_DATASET_VERSION = "v36"
_V36_COLLECTION_COUNTS = {"0830": 30, "0831": 40}
_V36_INCLUDED_EPISODES = 70
_V36_RAW_RECORDS = 71  # 70 converted episodes plus the excluded 0830_bin_part2/demo14 provenance record.
_V36_TRAIN_EPISODES = 54
_V36_BLOCK_EPISODE_COUNTS = {"0830_part1": 16, "0830_part2": 14, "0831": 40}
_V35_SPLIT_ALGORITHM = "openpi.v36.sha256-ranked-manifest-fields.v1"
_V35_SPLIT_ALGORITHM_SPEC = (
    "seed=36; sort stable_id; rank by sha256(algorithm,stage,seed,stable_id,part,object,target_side); "
    "select one final-test episode per collection*object*side cell; then one development episode per "
    "collection*object*side cell while preserving one train episode in every 0830 part*object*side "
    "and 0831 object*side cell; assign every other included episode to train"
)
_V35_SPLIT_ALGORITHM_SHA256 = hashlib.sha256(_V35_SPLIT_ALGORITHM_SPEC.encode("utf-8")).hexdigest()
_V35_OBJECT_PROMPTS = {
    "banana": "find the banana",
    "grey_pepper_box": "find the grey pepper box",
}
_V35_D_VALID_DETECTOR = "14d-max-step-lt-0.004-max-excursion-lte-0.02-v1"


def _v35_canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _v35_block_confound_summary(records: list[dict]) -> dict[str, object]:
    """Manifest-only temporal-block audit for the three v36 recording blocks.

    This intentionally uses no pixels, robot state, labels, or model outputs.  A part passes
    only when object and target side both occur in each temporal half, change at least three
    times, and cannot be perfectly separated by one collection-time threshold.
    """

    def feature_summary(values: list[str]) -> dict[str, object]:
        classes = sorted(set(values))
        if len(classes) != 2:
            return {
                "classes": classes,
                "transitions": 0,
                "max_run": len(values),
                "first_half_counts": {},
                "second_half_counts": {},
                "perfect_single_threshold": True,
                "pass": False,
            }
        split = len(values) // 2
        halves = (values[:split], values[split:])
        transitions = sum(left != right for left, right in itertools.pairwise(values))
        max_run = 0
        run = 0
        previous = None
        for value in values:
            run = run + 1 if value == previous else 1
            max_run = max(max_run, run)
            previous = value
        perfect_threshold = any(
            set(values[:cut]) == {left} and set(values[cut:]) == {right}
            for cut in range(1, len(values))
            for left, right in ((classes[0], classes[1]), (classes[1], classes[0]))
        )
        half_counts = [{name: half.count(name) for name in classes} for half in halves]
        passed = (
            transitions >= 3
            and not perfect_threshold
            and all(all(counts[name] > 0 for name in classes) for counts in half_counts)
        )
        return {
            "classes": classes,
            "transitions": transitions,
            "max_run": max_run,
            "first_half_counts": half_counts[0],
            "second_half_counts": half_counts[1],
            "perfect_single_threshold": perfect_threshold,
            "pass": passed,
        }

    included = [record for record in records if bool(record.get("include", True))]
    parts: dict[str, object] = {}
    for block, expected_count in sorted(_V36_BLOCK_EPISODE_COUNTS.items()):
        collection, _, part = block.partition("_")
        ordered = sorted(
            (
                record
                for record in included
                if str(record.get("collection", "")) == collection and str(record.get("part", "")) == part
            ),
            key=lambda record: (str(record.get("timestamp", "")), str(record.get("stable_id", ""))),
        )
        features = {
            "object": feature_summary([str(record.get("object", "")) for record in ordered]),
            "target_side": feature_summary([str(record.get("target_side", "")) for record in ordered]),
        }
        parts[block] = {
            "episode_count": len(ordered),
            "ordered_stable_ids": [str(record.get("stable_id", "")) for record in ordered],
            "features": features,
            "pass": len(ordered) == expected_count and all(feature["pass"] for feature in features.values()),
        }
    return {
        "algorithm": "openpi.v36.manifest-only-temporal-block-audit.v1",
        "parts": parts,
        "pass": all(bool(summary["pass"]) for summary in parts.values()),
    }


def _v35_load_hashed_sidecar(
    manifest_path: pathlib.Path,
    descriptor: object,
    *,
    name: str,
) -> dict:
    if not isinstance(descriptor, dict):
        raise ValueError(f"v3.5 frozen manifest is missing its {name} descriptor.")
    report_file = descriptor.get("report_file")
    report_sha256 = descriptor.get("report_sha256")
    if (
        not isinstance(report_file, str)
        or not report_file.strip()
        or pathlib.Path(report_file).is_absolute()
        or ".." in pathlib.Path(report_file).parts
        or not isinstance(report_sha256, str)
        or len(report_sha256) != 64
        or any(char not in "0123456789abcdef" for char in report_sha256)
    ):
        raise ValueError(f"v3.5 {name} descriptor must contain a relative file and canonical SHA256.")
    report_path = manifest_path.parent / report_file
    if not report_path.is_file() or hashlib.sha256(report_path.read_bytes()).hexdigest() != report_sha256:
        raise ValueError(f"v3.5 {name} report bytes do not match their manifest hash.")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot parse v3.5 {name} report: {exc}") from exc
    if not isinstance(report, dict):
        raise ValueError(f"v3.5 {name} report must be a JSON object.")
    return report


def _unwrap_lerobot(dataset: Dataset) -> lerobot_dataset.LeRobotDataset | None:
    inner = dataset
    while isinstance(inner, TransformedDataset):
        inner = inner._dataset  # noqa: SLF001
    return inner if isinstance(inner, lerobot_dataset.LeRobotDataset) else None


def _load_episode_prompts(dataset: Dataset) -> tuple[str, ...] | None:
    """Per-episode instructions from the dataset's meta/episode_prompts.json, or None.

    The sidecar ({"<episode_index>": "<instruction>"}) is written by the converter for
    multi-task datasets. Episodes without an entry get "" (InjectPromptFromEpisode then lets
    `default_prompt` fill them).
    """
    inner = _unwrap_lerobot(dataset)
    if inner is None:
        return None
    path = pathlib.Path(inner.root) / "meta" / "episode_prompts.json"
    if not path.exists():
        return None
    entries = {int(k): str(v) for k, v in json.loads(path.read_text()).items()}
    prompts = tuple(entries.get(e, "") for e in range(max(entries) + 1))
    logging.info(f"episode prompts: {len(entries)} entries, {len(set(entries.values()))} distinct instructions")
    return prompts


def _v35_split_rank(record: dict, *, stage: str, seed: int) -> str:
    """Version-stable pseudo-random rank using only preregistered manifest fields."""
    fields = (
        _V35_SPLIT_ALGORITHM,
        stage,
        str(seed),
        str(record["stable_id"]),
        str(record.get("part", "")),
        str(record["object"]),
        str(record["target_side"]),
    )
    return hashlib.sha256("\0".join(fields).encode("utf-8")).hexdigest()


def _v36_split_guard_cell(record: dict) -> tuple[str, ...]:
    """The finer stratification whose train coverage must never empty out."""
    collection = str(record.get("collection", "")).strip()
    if collection == "0830":
        return (
            "0830",
            str(record.get("part", "")).strip(),
            str(record.get("object", "")).strip(),
            str(record.get("target_side", "")).strip(),
        )
    return (collection, str(record.get("object", "")).strip(), str(record.get("target_side", "")).strip())


def _v35_expected_frozen_splits(records: list[dict], *, seed: int) -> dict[str, str]:
    """Reproduce the v36 54/8/8 split without reading images or model outputs."""
    included = [record for record in records if bool(record.get("include", True))]
    by_collection: dict[str, list[dict]] = {name: [] for name in _V36_COLLECTION_COUNTS}
    for record in included:
        collection = str(record.get("collection", "")).strip()
        if collection not in by_collection:
            raise ValueError(f"v3.5 frozen manifest has unknown collection {collection!r}.")
        by_collection[collection].append(record)
    if {key: len(value) for key, value in by_collection.items()} != _V36_COLLECTION_COUNTS:
        raise ValueError(f"v36 frozen manifest requires exactly {_V36_COLLECTION_COUNTS} included episodes.")

    cell_names = {
        (collection, object_name, side)
        for collection in _V36_COLLECTION_COUNTS
        for object_name in _V35_OBJECT_PROMPTS
        for side in ("left", "right")
    }
    cell_records: dict[tuple[str, str, str], list[dict]] = {cell: [] for cell in cell_names}
    for record in included:
        cell = (
            str(record.get("collection", "")).strip(),
            str(record.get("object", "")).strip(),
            str(record.get("target_side", "")).strip(),
        )
        if cell not in cell_records:
            raise ValueError(f"v36 frozen episode {record['stable_id']!r} has invalid collection/object/side {cell}.")
        cell_records[cell].append(record)
    if any(not values for values in cell_records.values()):
        missing = sorted(cell for cell, values in cell_records.items() if not values)
        raise ValueError(f"v36 frozen manifest is missing collection*object*side cells: {missing}.")

    final_ids: set[str] = set()
    remaining_by_cell: dict[tuple[str, str, str], list[dict]] = {}
    for cell, values in sorted(cell_records.items()):
        selected = min(values, key=lambda record: _v35_split_rank(record, stage="final_test", seed=seed))
        final_ids.add(str(selected["stable_id"]))
        remaining_by_cell[cell] = [record for record in values if record is not selected]

    development_ids: set[str] = set()
    for cell in sorted(cell_names):
        pool = remaining_by_cell[cell]
        guard_counts: dict[tuple[str, ...], int] = {}
        for record in pool:
            guard = _v36_split_guard_cell(record)
            guard_counts[guard] = guard_counts.get(guard, 0) + 1
        candidates = [record for record in pool if guard_counts[_v36_split_guard_cell(record)] >= 2]
        if not candidates:
            raise ValueError(
                f"v36 cannot select a seeded development episode for cell {cell} while preserving train coverage."
            )
        selected = min(candidates, key=lambda record: _v35_split_rank(record, stage="development", seed=seed))
        development_ids.add(str(selected["stable_id"]))
        remaining_by_cell[cell] = [record for record in pool if record is not selected]

    expected: dict[str, str] = {}
    for record in included:
        stable_id = str(record["stable_id"])
        expected[stable_id] = (
            "final_test" if stable_id in final_ids else "development" if stable_id in development_ids else "train"
        )
    if list(expected.values()).count("train") != _V36_TRAIN_EPISODES:
        raise ValueError(
            f"internal v36 split construction did not produce exactly {_V36_TRAIN_EPISODES} training episodes."
        )
    train_guard_counts: dict[tuple[str, ...], int] = {}
    for record in included:
        if expected[str(record["stable_id"])] == "train":
            guard = _v36_split_guard_cell(record)
            train_guard_counts[guard] = train_guard_counts.get(guard, 0) + 1
    all_guards = {_v36_split_guard_cell(record) for record in included}
    if any(train_guard_counts.get(guard, 0) < 1 for guard in all_guards):
        raise ValueError("v36 split leaves an empty guarded train cell.")
    return expected


def _v35_resolve_label_path(manifest_path: pathlib.Path, raw: dict, record: dict) -> pathlib.Path:
    raw_root_value = raw.get("raw_root")
    raw_dir_value = record.get("raw_dir", record.get("source_path"))
    label_file_value = record.get("label_file")
    if not all(isinstance(value, str) and value.strip() for value in (raw_root_value, raw_dir_value, label_file_value)):
        raise ValueError(f"v3.5 included episode {record['stable_id']!r} is missing raw_root/raw_dir/label_file.")
    raw_root = pathlib.Path(raw_root_value)
    if not raw_root.is_absolute():
        raw_root = manifest_path.parent / raw_root
    raw_root = raw_root.resolve()
    raw_dir = pathlib.Path(raw_dir_value)
    if raw_dir.is_absolute():
        raise ValueError(f"v3.5 raw_dir must be relative for {record['stable_id']!r}.")
    if not raw_dir.as_posix().rstrip("/").endswith(str(record["stable_id"])):
        raise ValueError(f"v3.5 raw_dir does not identify stable_id {record['stable_id']!r}.")
    source_dir = (raw_root / raw_dir).resolve()
    try:
        source_dir.relative_to(raw_root)
    except ValueError as exc:
        raise ValueError(f"v3.5 raw_dir escapes raw_root for {record['stable_id']!r}.") from exc
    label_file = pathlib.Path(label_file_value)
    if label_file.is_absolute() or len(label_file.parts) != 1:
        raise ValueError(f"v3.5 label_file must be a basename for {record['stable_id']!r}.")
    return source_dir / label_file


def _validate_v35_frozen_record_provenance(
    *,
    manifest_path: pathlib.Path,
    raw: dict,
    records: dict[int, dict],
    num_episodes: int,
    episode_length: np.ndarray,
    episode_tasks: tuple[np.ndarray, ...],
    tasks: dict[int, str],
    prompts: tuple[str, ...] | None,
    visibility_records: dict[str, dict],
    d_valid_records: dict[str, dict],
) -> tuple[np.ndarray, np.ndarray]:
    """Authenticate labels/prompts/D sidecars without touching held-out observations."""
    if raw.get("dataset_version") != _V36_DATASET_VERSION or raw.get("review_status") != "frozen":
        raise ValueError("v36 production manifest must have dataset_version='v36' and review_status='frozen'.")
    if raw.get("split_algorithm") != _V35_SPLIT_ALGORITHM:
        raise ValueError(f"v3.5 production manifest must use split_algorithm={_V35_SPLIT_ALGORITHM!r}.")
    if raw.get("split_algorithm_sha256") != _V35_SPLIT_ALGORITHM_SHA256:
        raise ValueError("v3.5 manifest split algorithm specification hash is invalid.")
    if prompts is None or len(prompts) != num_episodes or any(not prompt.strip() for prompt in prompts):
        raise ValueError("v3.5 frozen dataset requires one nonempty episode_prompts.json entry per episode.")
    if episode_length.shape != (num_episodes,) or len(episode_tasks) != num_episodes:
        raise ValueError("v3.5 frozen provenance validation received inconsistent episode tables.")

    d_lo = np.full(num_episodes, -1, dtype=np.int32)
    d_hi = np.full(num_episodes, -1, dtype=np.int32)
    for episode_index in range(num_episodes):
        record = records[episode_index]
        stable_id = str(record["stable_id"])
        if not bool(record.get("include", True)) or str(record.get("exclude_reason", "")).strip():
            raise ValueError(f"converted v3.5 episode {stable_id!r} must be included with no exclusion reason.")
        object_name = str(record.get("object", "")).strip()
        prompt = str(record.get("prompt", "")).strip()
        if object_name not in _V35_OBJECT_PROMPTS or prompt != _V35_OBJECT_PROMPTS[object_name]:
            raise ValueError(f"v3.5 episode {stable_id!r} has a noncanonical object/prompt pair.")
        if prompts[episode_index] != prompt:
            raise ValueError(
                f"v3.5 prompt mismatch for episode {episode_index} ({stable_id}): "
                f"manifest={prompt!r}, dataset={prompts[episode_index]!r}."
            )
        expected_frames = record.get("expected_num_frames")
        if not isinstance(expected_frames, int) or expected_frames != int(episode_length[episode_index]):
            raise ValueError(f"v3.5 episode {stable_id!r} frame count does not match the frozen manifest.")

        label_sha256 = record.get("label_sha256")
        if (
            not isinstance(label_sha256, str)
            or len(label_sha256) != 64
            or any(char not in "0123456789abcdef" for char in label_sha256)
        ):
            raise ValueError(f"v3.5 episode {stable_id!r} is missing a canonical label_sha256.")
        label_path = _v35_resolve_label_path(manifest_path, raw, record)
        if not label_path.is_file() or hashlib.sha256(label_path.read_bytes()).hexdigest() != label_sha256:
            raise ValueError(f"v3.5 label bytes/hash mismatch for {stable_id!r}: {label_path}.")
        try:
            segments = json.loads(label_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot parse frozen v3.5 labels for {stable_id!r}: {exc}") from exc
        if not isinstance(segments, list) or len(segments) != 5:
            raise ValueError(f"v3.5 episode {stable_id!r} must have exactly five semantic label segments.")
        side_name = str(record["target_side"])
        expected_tasks = [
            "open both lids",
            "inspect both bins",
            "close both lids and reset arms",
            f"wait; target bin is {side_name}",
            f"open {side_name} bin",
        ]
        if [segment.get("task") if isinstance(segment, dict) else None for segment in segments] != expected_tasks:
            raise ValueError(f"v3.5 episode {stable_id!r} does not have the exact side-consistent five-phase schema.")
        visibility = record.get("e_visibility")
        if not isinstance(visibility, dict):
            raise ValueError(f"v3.5 episode {stable_id!r} is missing manual E-visibility provenance.")
        sealed_visibility = visibility_records.get(stable_id)
        if not isinstance(sealed_visibility, dict) or visibility != sealed_visibility:
            raise ValueError(f"v3.5 episode {stable_id!r} E-visibility fields do not match the hashed sidecar.")
        contact_sha256 = visibility.get("contact_sheet_sha256")
        semantic_e_start = segments[1]["start"]
        semantic_e_end = segments[1]["end"]
        if (
            visibility.get("manual_reviewed") is not True
            or visibility.get("both_objects_visible") is not True
            or visibility.get("first_valid_visible_frame") != semantic_e_start
            or not isinstance(visibility.get("last_clean_visible_frame"), int)
            or visibility["last_clean_visible_frame"] < semantic_e_end - 5
            or not isinstance(contact_sha256, str)
            or len(contact_sha256) != 64
            or any(char not in "0123456789abcdef" for char in contact_sha256)
        ):
            raise ValueError(f"v3.5 episode {stable_id!r} has invalid manual E-visibility/anchor provenance.")
        rebuilt: list[str] = []
        next_start = 0
        for segment in segments:
            if not isinstance(segment, dict):
                raise ValueError(f"v3.5 episode {stable_id!r} has a non-object label segment.")
            start, end, task = segment.get("start"), segment.get("end"), segment.get("task")
            if not isinstance(start, int) or not isinstance(end, int) or start != next_start or end < start:
                raise ValueError(f"v3.5 episode {stable_id!r} has non-contiguous label coverage.")
            rebuilt.extend([str(task)] * (end - start + 1))
            next_start = end + 1
        if next_start != int(episode_length[episode_index]):
            raise ValueError(f"v3.5 episode {stable_id!r} labels do not end at the final converted frame.")
        converted = [tasks[int(task_id)] for task_id in episode_tasks[episode_index]]
        if rebuilt != converted:
            raise ValueError(f"v3.5 converted task labels do not match frozen label bytes for {stable_id!r}.")

        d_valid = record.get("d_valid")
        if not isinstance(d_valid, dict):
            raise ValueError(f"v3.5 episode {stable_id!r} is missing its independent D_valid sidecar.")
        sealed_d_valid = d_valid_records.get(stable_id)
        if not isinstance(sealed_d_valid, dict) or d_valid != sealed_d_valid:
            raise ValueError(f"v3.5 episode {stable_id!r} D_valid fields do not match the hashed sidecar.")
        if d_valid.get("detector") != _V35_D_VALID_DETECTOR or d_valid.get("state_dim") != 14:
            raise ValueError(f"v3.5 episode {stable_id!r} has the wrong D_valid detector provenance.")
        start, end = d_valid.get("start"), d_valid.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end < expected_frames:
            raise ValueError(f"v3.5 episode {stable_id!r} has invalid D_valid bounds.")
        d_lo[episode_index], d_hi[episode_index] = start, end
    return d_lo, d_hi


def _load_v35_episode_manifest(
    data_config: "_config.DataConfig",
    *,
    num_episodes: int,
    side: np.ndarray,
    episode_length: np.ndarray,
    episode_tasks: tuple[np.ndarray, ...],
    tasks: dict[int, str],
    prompts: tuple[str, ...] | None,
) -> dict[str, np.ndarray | tuple[str, ...]]:
    """Load and fail-closed validate the stable v3.5 episode/split manifest.

    The manifest owns stable identity and split assignment; the loader never embeds final-test
    episode numbers.  Accepted top-level form is ``{"schema_version": ..., "split_seed": ...,
    "episodes": [...]}``.  Every converted episode must appear exactly once with
    ``episode_index``, ``stable_id``, ``collection``, ``object``, ``target_side``, ``split``,
    and optional ``include`` (default true).
    """
    path_value = data_config.memory_episode_manifest_path
    if path_value is None:
        raise ValueError("v3.5 episode manifest path is not configured.")
    path = pathlib.Path(path_value)
    if not path.is_file():
        raise ValueError(f"v3.5 episode manifest does not exist: {path}")
    manifest_bytes = path.read_bytes()
    expected_sha256 = data_config.memory_episode_manifest_sha256
    if data_config.memory_v35_frozen_population and expected_sha256 is None:
        raise ValueError("v3.5 frozen population requires an exact manifest SHA256.")
    if expected_sha256 is not None:
        if len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256):
            raise ValueError("v3.5 episode manifest SHA256 must be 64 lower-case hex characters.")
        actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"v3.5 episode manifest SHA256 mismatch: expected {expected_sha256}, found {actual_sha256}."
            )
    raw = json.loads(manifest_bytes)
    if not isinstance(raw, dict) or not isinstance(raw.get("episodes"), list):
        raise ValueError("v3.5 episode manifest must be an object containing an episodes list.")
    if int(raw.get("schema_version", 0)) <= 0:
        raise ValueError("v3.5 episode manifest requires a positive schema_version.")
    visibility_records: dict[str, dict] = {}
    d_valid_records: dict[str, dict] = {}
    if data_config.memory_v35_frozen_population:
        if raw.get("schema_version") != _V35_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"v3.5 frozen manifest requires schema_version={_V35_MANIFEST_SCHEMA_VERSION}.")
        expected_vocabulary = list(data_config.memory_subtask_vocab)
        if raw.get("task_vocabulary") != expected_vocabulary:
            raise ValueError("v3.5 frozen manifest task_vocabulary/order does not match the configured seven tasks.")
        dataset_vocabulary = [tasks[index] for index in sorted(tasks)]
        if sorted(tasks) != list(range(len(tasks))) or dataset_vocabulary != expected_vocabulary:
            raise ValueError("v3.5 converted dataset task IDs are not the exact frozen seven-task vocabulary/order.")
        block_audit = raw.get("block_confound_audit")
        if (
            not isinstance(block_audit, dict)
            or block_audit.get("status") != "pass"
            or block_audit.get("manifest_fields_only") is not True
        ):
            raise ValueError("v36 frozen manifest is missing a passing object/side block-confound audit.")
        block_report = _v35_load_hashed_sidecar(path, block_audit, name="block-confound audit")
        audit_fields = [
            {
                key: record.get(key)
                for key in ("stable_id", "include", "collection", "part", "object", "target_side", "timestamp")
            }
            for record in raw["episodes"]
            if bool(record.get("include", True))
        ]
        audit_fields_sha256 = hashlib.sha256(_v35_canonical_json(audit_fields).encode("utf-8")).hexdigest()
        expected_block_summary = _v35_block_confound_summary(raw["episodes"])
        if (
            block_report.get("schema_version") != "openpi.v36.block-confound-audit.v1"
            or block_report.get("status") != "pass"
            or block_report.get("manifest_fields_only") is not True
            or block_report.get("manifest_fields_sha256") != audit_fields_sha256
            or block_audit.get("manifest_fields_sha256") != audit_fields_sha256
            or block_report.get("summary") != expected_block_summary
            or expected_block_summary.get("pass") is not True
        ):
            raise ValueError("v3.5 block-confound report does not reproduce the frozen manifest-only audit.")

        visibility_descriptor = raw.get("e_visibility_review")
        visibility_report = _v35_load_hashed_sidecar(
            path,
            visibility_descriptor,
            name="E-visibility review",
        )
        if (
            visibility_report.get("schema_version") != "openpi.v36.e-visibility-review.v1"
            or visibility_report.get("status") != "user_approved"
            or not isinstance(visibility_descriptor, dict)
            or visibility_descriptor.get("status") != "user_approved"
            or not isinstance(visibility_report.get("episodes"), list)
        ):
            raise ValueError("v3.5 frozen E-visibility review is incomplete or not user-approved.")
        visibility_records = {
            str(entry.get("stable_id", "")): entry.get("e_visibility")
            for entry in visibility_report["episodes"]
            if isinstance(entry, dict)
        }
        if (
            len(visibility_report["episodes"]) != _V36_INCLUDED_EPISODES
            or len(visibility_records) != _V36_INCLUDED_EPISODES
            or "" in visibility_records
        ):
            raise ValueError("v36 frozen E-visibility sidecar must cover 70 unique converted stable IDs.")

        d_valid_descriptor = raw.get("d_valid_sidecar")
        d_valid_report = _v35_load_hashed_sidecar(path, d_valid_descriptor, name="D_valid sidecar")
        if (
            d_valid_report.get("schema_version") != "openpi.v36.d-valid.v1"
            or d_valid_report.get("status") != "complete"
            or d_valid_report.get("detector") != _V35_D_VALID_DETECTOR
            or d_valid_report.get("state_dim") != 14
            or not isinstance(d_valid_descriptor, dict)
            or d_valid_descriptor.get("detector") != _V35_D_VALID_DETECTOR
            or d_valid_descriptor.get("state_dim") != 14
            or not isinstance(d_valid_report.get("episodes"), list)
        ):
            raise ValueError("v3.5 frozen D_valid sidecar has the wrong detector or schema.")
        d_valid_records = {
            str(entry.get("stable_id", "")): entry.get("d_valid")
            for entry in d_valid_report["episodes"]
            if isinstance(entry, dict)
        }
        if (
            len(d_valid_report["episodes"]) != _V36_INCLUDED_EPISODES
            or len(d_valid_records) != _V36_INCLUDED_EPISODES
            or "" in d_valid_records
        ):
            raise ValueError("v36 frozen D_valid sidecar must cover 70 unique converted stable IDs.")

    expected_seed = data_config.memory_manifest_split_seed
    manifest_seed = raw.get("split_seed")
    if manifest_seed is None and isinstance(raw.get("split"), dict):
        manifest_seed = raw["split"].get("seed")
    if expected_seed is not None and manifest_seed != expected_seed:
        raise ValueError(f"v3.5 manifest split_seed mismatch: expected {expected_seed}, found {manifest_seed!r}.")

    records: dict[int, dict] = {}
    stable_ids: set[str] = set()
    excluded_records: list[dict] = []
    for record in raw["episodes"]:
        if not isinstance(record, dict):
            raise ValueError("v3.5 manifest episode entries must be objects.")
        stable_id = str(record.get("stable_id", "")).strip()
        if not stable_id:
            raise ValueError("v3.5 manifest episode is missing stable_id.")
        if stable_id in stable_ids:
            raise ValueError(f"duplicate v3.5 manifest stable_id {stable_id!r}.")
        stable_ids.add(stable_id)
        index_value = record.get("episode_index", record.get("lerobot_episode_index"))
        if index_value is None:
            if not bool(record.get("include", True)):
                # Raw excluded episodes (for example incomplete 0830 part2/demo14) remain in
                # the provenance manifest but correctly have no converted LeRobot index.
                excluded_records.append(record)
                continue
            raise ValueError(f"included v3.5 manifest episode {stable_id!r} is missing episode_index.")
        episode_index = int(index_value)
        if episode_index in records:
            raise ValueError(f"duplicate v3.5 manifest episode_index {episode_index}.")
        records[episode_index] = record

    expected_indices = set(range(num_episodes))
    if set(records) != expected_indices:
        missing = sorted(expected_indices - set(records))
        extra = sorted(set(records) - expected_indices)
        raise ValueError(f"v3.5 manifest/dataset episode mismatch: missing={missing}, extra={extra}.")

    manifest_d_lo = np.full(num_episodes, -1, dtype=np.int32)
    manifest_d_hi = np.full(num_episodes, -1, dtype=np.int32)
    if data_config.memory_v35_frozen_population:
        if len(raw["episodes"]) != _V36_RAW_RECORDS or len(excluded_records) != 1:
            raise ValueError("v36 frozen provenance must contain 70 converted episodes plus one excluded raw episode.")
        excluded = excluded_records[0]
        if (
            excluded.get("stable_id") != "0830_bin_part2/demo14"
            or not str(excluded.get("exclude_reason", "")).strip()
            or not str(excluded.get("raw_dir", excluded.get("source_path", ""))).strip()
        ):
            raise ValueError("v3.5 frozen provenance must retain excluded 0830_bin_part2/demo14 and its reason/path.")
        expected_splits = _v35_expected_frozen_splits(raw["episodes"], seed=int(expected_seed))
        wrong_splits = {
            stable_id: (expected_splits[stable_id], str(record.get("split", "")))
            for record in records.values()
            if (stable_id := str(record["stable_id"])) in expected_splits
            and str(record.get("split", "")) != expected_splits[stable_id]
        }
        if wrong_splits:
            raise ValueError(f"v3.5 manifest split assignments do not reproduce the frozen algorithm: {wrong_splits}.")
        manifest_d_lo, manifest_d_hi = _validate_v35_frozen_record_provenance(
            manifest_path=path,
            raw=raw,
            records=records,
            num_episodes=num_episodes,
            episode_length=episode_length,
            episode_tasks=episode_tasks,
            tasks=tasks,
            prompts=prompts,
            visibility_records=visibility_records,
            d_valid_records=d_valid_records,
        )

    stable_id_values: list[str] = []
    collections: list[str] = []
    objects: list[str] = []
    parts: list[str] = []
    splits: list[str] = []
    included = np.zeros(num_episodes, dtype=bool)
    manifest_side = np.full(num_episodes, -1, dtype=np.int32)
    side_names = {"left": 0, "right": 1}
    for episode_index in range(num_episodes):
        record = records[episode_index]
        stable_id_values.append(str(record["stable_id"]).strip())
        collection = str(record.get("collection", record.get("session", ""))).strip()
        object_name = str(record.get("object", record.get("target_object", ""))).strip()
        part_name = str(record.get("part", "")).strip()
        split_name = str(record.get("split", "")).strip()
        side_name = str(record.get("target_side", "")).strip().lower()
        if not collection or not object_name or not split_name or side_name not in side_names:
            raise ValueError(
                f"v3.5 manifest episode {episode_index} must define collection, object, "
                "target_side in {left,right}, and split."
            )
        collections.append(collection)
        objects.append(object_name)
        parts.append(part_name)
        splits.append(split_name)
        included[episode_index] = bool(record.get("include", True))
        manifest_side[episode_index] = side_names[side_name]
        if included[episode_index] and side[episode_index] not in (0, 1):
            raise ValueError(
                f"included v3.5 episode {episode_index} ({stable_id_values[-1]}) has no label-derived target side."
            )
        if included[episode_index] and manifest_side[episode_index] != side[episode_index]:
            raise ValueError(
                f"v3.5 manifest side mismatch for episode {episode_index} ({stable_id_values[-1]}): "
                f"manifest={side_name}, labels={int(side[episode_index])}."
            )

    active_split = str(data_config.memory_manifest_split)
    sampling_allowed = included & (np.asarray(splits, dtype=object) == active_split)
    if not np.any(sampling_allowed):
        raise ValueError(f"v3.5 manifest has no included episodes in active split {active_split!r}.")

    collection_vocab = {name: i for i, name in enumerate(sorted(set(collections)))}
    object_vocab = {name: i for i, name in enumerate(sorted(set(objects)))}
    cells = [(collections[e], objects[e], int(manifest_side[e])) for e in range(num_episodes)]
    cell_vocab = {cell: i for i, cell in enumerate(sorted(set(cells)))}
    if data_config.memory_v35_frozen_population:
        split_counts = {name: splits.count(name) for name in set(splits)}
        expected_split_counts = {"train": _V36_TRAIN_EPISODES, "development": 8, "final_test": 8}
        if num_episodes != _V36_INCLUDED_EPISODES or not np.all(included) or split_counts != expected_split_counts:
            raise ValueError(
                "v36 frozen population must be exactly 70 included converted episodes with "
                f"split counts {expected_split_counts}; found episodes={num_episodes}, "
                f"included={int(included.sum())}, splits={split_counts}."
            )
        if len(cell_vocab) != 8:
            raise ValueError(
                f"v36 frozen population requires exactly eight collection/object/side cells; found {len(cell_vocab)}."
            )
        final_cells = {
            (collections[e], objects[e], int(manifest_side[e]))
            for e in range(num_episodes)
            if splits[e] == "final_test"
        }
        required_final_cells = {
            (collection, object_name, side)
            for collection in _V36_COLLECTION_COUNTS
            for object_name in _V35_OBJECT_PROMPTS
            for side in (0, 1)
        }
        train_cells = {
            (collections[e], objects[e], int(manifest_side[e]))
            for e in range(num_episodes)
            if splits[e] == "train"
        }
        if final_cells != required_final_cells or not required_final_cells <= train_cells:
            raise ValueError(
                "v36 frozen split must have one final-test episode and at least one train episode "
                "in every collection*object*side cell."
            )
    logging.info(
        "v3.5 manifest: %d episodes, split=%s seed=%s, %d active, %d collections, %d cells",
        num_episodes,
        active_split,
        manifest_seed,
        int(sampling_allowed.sum()),
        len(collection_vocab),
        len(cell_vocab),
    )
    return {
        "stable_id": tuple(stable_id_values),
        "collection_name": tuple(collections),
        "object_name": tuple(objects),
        "part_name": tuple(parts),
        "manifest_split": tuple(splits),
        "sampling_allowed": sampling_allowed,
        "collection_id": np.asarray([collection_vocab[x] for x in collections], dtype=np.int32),
        "object_id": np.asarray([object_vocab[x] for x in objects], dtype=np.int32),
        "memory_cell": np.asarray([cell_vocab[x] for x in cells], dtype=np.int32),
        "manifest_side": manifest_side,
        "manifest_d_lo": manifest_d_lo,
        "manifest_d_hi": manifest_d_hi,
    }


def _episode_info_table(
    dataset: Dataset, dataset_meta: lerobot_dataset.LeRobotDatasetMetadata, data_config: "_config.DataConfig"
) -> dict[str, np.ndarray]:
    """Per-episode metadata for memory sequence training:
    * "length": frames in the episode;
    * "switch": the first frame whose (unshifted) task label differs from the episode's
      opening one -- the decision moment (episode length when there is no switch);
    * "side": the answer class from the FINAL subtask label ("open left bin" -> 0,
      "open right bin" -> 1, otherwise -1 = never quizzed);
    * "reveal"/"close": when the answer becomes visible / hidden again, from the optional
      json ({"<episode_index>": reveal | [reveal, close]}), defaults otherwise;
    * label-derived phase bounds (all -1 when the episode lacks a phase, only computed when
      the config names the vocabularies): "evidence_start"/"evidence_end" (first/last frame
      whose label is in `evidence_subtasks`) and "memory_lo"/"memory_hi" (first/last frame
      whose label is in `memory_required_subtasks`).
    """
    cols = _unwrap_lerobot(dataset).hf_dataset.with_format(None)
    task = np.asarray(cols["task_index"], dtype=np.int64)
    episode = np.asarray(cols["episode_index"], dtype=np.int64)

    num_episodes = int(episode.max()) + 1
    starts = np.nonzero(np.append(True, episode[1:] != episode[:-1]))[0]
    ends = np.append(starts[1:], len(episode))
    length = (ends - starts).astype(np.int32)

    side = np.full(num_episodes, -1, dtype=np.int32)
    switch = length.copy()
    for e in range(num_episodes):
        ep_task = task[starts[e] : ends[e]]
        final_task = dataset_meta.tasks[int(ep_task[-1])].lower()
        if "left" in final_task:
            side[e] = 0
        elif "right" in final_task:
            side[e] = 1
        changed = np.nonzero(ep_task != ep_task[0])[0]
        if len(changed) > 0:
            switch[e] = int(changed[0])

    episode_tasks = tuple(task[starts[e] : ends[e]] for e in range(num_episodes))

    reveal = np.full(num_episodes, _DEFAULT_REVEAL_FRAME, dtype=np.int32)
    close = reveal + _DEFAULT_VISIBLE_SPAN
    labeled = 0
    reveal_frames_path = data_config.memory_reveal_frames_path
    if reveal_frames_path is not None and pathlib.Path(reveal_frames_path).exists():
        for key, value in json.loads(pathlib.Path(reveal_frames_path).read_text()).items():
            e = int(key)
            if not 0 <= e < num_episodes:
                raise ValueError(f"reveal_frames json: episode {e} out of range [0, {num_episodes})")
            reveal[e], close[e] = (value, value + _DEFAULT_VISIBLE_SPAN) if np.isscalar(value) else value
            labeled += 1
    logging.info(
        f"episode table: {num_episodes} episodes (len {length.min()}-{length.max()}), "
        f"{int((side >= 0).sum())} sided ({int((side == 0).sum())}L/{int((side == 1).sum())}R), "
        f"reveal frames: {labeled} from json, {num_episodes - labeled} at default {_DEFAULT_REVEAL_FRAME}"
    )
    info = {
        "length": length,
        "switch": switch,
        "side": side,
        "reveal": reveal,
        "close": close,
        "episode_tasks": episode_tasks,
    }
    if data_config.memory_v35_enabled:
        info.update(
            _load_v35_episode_manifest(
                data_config,
                num_episodes=num_episodes,
                side=side,
                episode_length=length,
                episode_tasks=episode_tasks,
                tasks=dataset_meta.tasks,
                prompts=_load_episode_prompts(dataset),
            )
        )

    if data_config.memory_required_subtasks and data_config.evidence_subtasks:
        memory_ids = [i for i, s in dataset_meta.tasks.items() if s in data_config.memory_required_subtasks]
        evidence_ids = [i for i, s in dataset_meta.tasks.items() if s in data_config.evidence_subtasks]
        if not memory_ids or not evidence_ids:
            raise ValueError(
                f"none of memory_required_subtasks={data_config.memory_required_subtasks} or "
                f"evidence_subtasks={data_config.evidence_subtasks} match the dataset's task strings."
            )
        bounds = {
            k: np.full(num_episodes, -1, dtype=np.int32)
            for k in ("evidence_start", "evidence_end", "memory_lo", "memory_hi")
        }
        if data_config.memory_v35_enabled:
            occlusion_ids = [i for i, s in dataset_meta.tasks.items() if s in data_config.memory_occlusion_subtasks]
            execute_ids = [i for i, s in dataset_meta.tasks.items() if s in data_config.memory_execute_subtasks]
            if not occlusion_ids or not execute_ids:
                raise ValueError(
                    f"none of v3.5 occlusion={data_config.memory_occlusion_subtasks} or "
                    f"execute={data_config.memory_execute_subtasks} labels match the dataset task strings."
                )
            bounds.update(
                {
                    "occlusion_lo": np.full(num_episodes, -1, dtype=np.int32),
                    "occlusion_hi": np.full(num_episodes, -1, dtype=np.int32),
                    "execute_start": np.full(num_episodes, -1, dtype=np.int32),
                    "final_e_limit": np.full(num_episodes, -1, dtype=np.int32),
                }
            )
        memory_contiguous = np.ones(num_episodes, dtype=bool)
        for e in range(num_episodes):
            for name, ids in (("evidence", evidence_ids), ("memory", memory_ids)):
                hit = np.nonzero(np.isin(episode_tasks[e], ids))[0]
                if len(hit) > 0:
                    lo_key = "evidence_start" if name == "evidence" else "memory_lo"
                    hi_key = "evidence_end" if name == "evidence" else "memory_hi"
                    bounds[lo_key][e] = int(hit[0])
                    bounds[hi_key][e] = int(hit[-1])
                    if name == "memory" and int(hit[-1]) - int(hit[0]) + 1 != len(hit):
                        # A stray waiting label (e.g. mid-execute) would stretch [lo, hi] over
                        # frames where the answer is visible; endpoints landing there would
                        # grade "memory" the images plainly show. Exclude the episode.
                        memory_contiguous[e] = False
                        logging.warning(
                            f"episode {e}: memory-required labels are not contiguous "
                            f"(span {int(hit[0])}-{int(hit[-1])}, {len(hit)} frames); excluded."
                        )
            if data_config.memory_v35_enabled:
                occlusion_hit = np.nonzero(np.isin(episode_tasks[e], occlusion_ids))[0]
                execute_hit = np.nonzero(np.isin(episode_tasks[e], execute_ids))[0]
                if len(occlusion_hit) > 0:
                    bounds["occlusion_lo"][e] = int(occlusion_hit[0])
                    bounds["occlusion_hi"][e] = int(occlusion_hit[-1])
                if len(execute_hit) > 0:
                    bounds["execute_start"][e] = int(execute_hit[0])
                if bounds["evidence_end"][e] >= 0:
                    bounds["final_e_limit"][e] = bounds["evidence_end"][e] - data_config.memory_e_tail_guard_frames
        usable = (
            (bounds["evidence_start"] >= 0)
            & (bounds["memory_lo"] >= 0)
            & (bounds["evidence_end"] < bounds["memory_lo"])
            & memory_contiguous
        )
        if data_config.memory_v35_enabled:
            occlusion_contiguous = np.asarray(
                [
                    bounds["occlusion_lo"][e] >= 0
                    and bounds["occlusion_hi"][e] - bounds["occlusion_lo"][e] + 1
                    == int(np.isin(episode_tasks[e], occlusion_ids).sum())
                    for e in range(num_episodes)
                ],
                dtype=bool,
            )
            usable &= (
                (bounds["final_e_limit"] >= bounds["evidence_start"])
                & (bounds["occlusion_lo"] > bounds["evidence_end"])
                & (bounds["occlusion_hi"] < bounds["memory_lo"])
                & (bounds["execute_start"] > bounds["memory_hi"])
                & occlusion_contiguous
            )
        for key, value in bounds.items():
            info[key] = np.where(usable, value, -1).astype(np.int32)
        if not usable.all():
            logging.warning(
                f"memory-critical phases missing/malformed in {int((~usable).sum())}/{num_episodes} "
                "episodes; they are excluded from the memory-critical branch."
            )
        if data_config.memory_v35_enabled and data_config.memory_v35_frozen_population:
            _apply_frozen_v35_d_valid(info)
        elif data_config.memory_waiting_max_speed is not None:
            state = np.asarray(cols["state"], dtype=np.float32)
            expected_dim = data_config.memory_waiting_state_dim
            if expected_dim is not None and (state.ndim != 2 or state.shape[1] != expected_dim):
                raise ValueError(f"strict waiting detector expected state shape [N,{expected_dim}], got {state.shape}.")
            _trim_waiting_to_static(
                info,
                state,
                starts,
                ends,
                data_config,
                stride=max(int(data_config.memory_stride_frames), 1),
            )
    return info


def _longest_static_run(window: np.ndarray, max_speed: float, max_excursion: float) -> tuple[int, int] | None:
    """Longest contiguous span of `window` [n, d] that holds still.

    "Still" is two conditions, because either alone admits motion: no frame-to-frame step on any
    dimension may exceed `max_speed` (catches motion onset), and the span's total per-dimension
    excursion must stay within `max_excursion` (catches slow creep that never trips the speed
    test). Returns inclusive [a, b] indices into `window`, or None when no span qualifies.
    """
    if len(window) < 2:
        return None
    speed = np.abs(np.diff(window, axis=0)).max(axis=1)
    # A frame is quiet when the step that ARRIVES at it is small; the first frame has no
    # arriving step, so it borrows the one leaving it. Seeding it True instead would make any
    # window -- including one that moves throughout -- report a spurious static frame at 0.
    quiet = np.concatenate([[speed[0] < max_speed], speed < max_speed])
    best: tuple[int, int] | None = None
    best_len = 0
    start = 0
    while start < len(quiet):
        if not quiet[start]:
            start += 1
            continue
        stop = start
        while stop + 1 < len(quiet) and quiet[stop + 1]:
            stop += 1
        # The run is speed-quiet; walk its left edge in until the excursion also fits.
        left = start
        while left <= stop:
            span = window[left : stop + 1]
            if float((span.max(axis=0) - span.min(axis=0)).max()) <= max_excursion:
                break
            left += 1
        # A single frame carries no evidence of stillness, so runs must span at least two.
        if stop - left + 1 >= 2 and stop - left + 1 > best_len:
            best_len = stop - left + 1
            best = (left, stop)
        start = stop + 1
    return best


def _trim_waiting_to_static(
    info: dict[str, np.ndarray],
    state: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    data_config: "_config.DataConfig",
    stride: int,
) -> None:
    """Shrink [memory_lo, memory_hi] to each episode's stationary core and mark the waiting
    frames it drops (v3.4.1 leak fix 1; see DataConfig.memory_waiting_max_speed).

    Mutates `info` in place: tightens "memory_lo"/"memory_hi", adds "memory_critical_ok"
    (False for episodes whose static core cannot hold a single grid step -- the phase bounds
    stay accurate for the slice dead zone, but no memory-critical endpoint may be placed there),
    and adds "episode_waiting_valid" -- per-episode per-frame booleans that are False exactly on
    waiting-labeled frames outside the static core, so the aux target and the ladder waiting
    mask can drop them.
    """
    max_speed = data_config.memory_waiting_max_speed
    max_excursion = data_config.memory_waiting_max_excursion
    num_episodes = len(info["length"])
    valid = [np.ones(int(info["length"][e]), dtype=bool) for e in range(num_episodes)]
    ok = info["memory_lo"] >= 0
    trimmed = []
    disabled = []
    for e in range(num_episodes):
        lo, hi = int(info["memory_lo"][e]), int(info["memory_hi"][e])
        if lo < 0 or hi < lo:
            continue
        episode_state = state[starts[e] : ends[e]]
        run = _longest_static_run(episode_state[lo : hi + 1], max_speed, max_excursion)
        if run is None:
            info["memory_lo"][e] = -1
            info["memory_hi"][e] = -1
            valid[e][lo : hi + 1] = False
            ok[e] = False
            disabled.append(e)
            continue
        new_lo, new_hi = lo + run[0], lo + run[1]
        valid[e][lo:new_lo] = False
        valid[e][new_hi + 1 : hi + 1] = False
        info["memory_lo"][e] = new_lo
        info["memory_hi"][e] = new_hi
        if new_hi - new_lo + 1 < stride and not data_config.memory_v35_enabled:
            # Too short to place even one grid step: keep the honest phase bounds and label
            # mask, but take the episode out of the memory-critical branch rather than crowding
            # every endpoint onto a handful of frames.  v3.5 is different: raw start residues
            # are searched explicitly, so even a short strict-D core may provide one legal
            # clock-aware sparse point; the per-episode candidate gate decides it later.
            ok[e] = False
            disabled.append(e)
        if (new_lo, new_hi) != (lo, hi):
            trimmed.append((e, lo, hi, new_lo, new_hi))
    info["episode_waiting_valid"] = tuple(valid)
    info["memory_critical_ok"] = ok
    dropped = int(sum((~v).sum() for v in valid))
    total = int(sum(int(info["length"][e]) for e in range(num_episodes)))
    logging.info(
        f"waiting-phase static trim (speed<{max_speed}, excursion<{max_excursion}): "
        f"{len(trimmed)}/{num_episodes} episodes trimmed, {dropped} waiting frames dropped from "
        f"memory supervision ({dropped / max(total, 1):.1%} of all frames); "
        f"memory-critical branch disabled for {disabled}"
    )
    for e, lo, hi, new_lo, new_hi in trimmed:
        if (lo != new_lo) or (hi - new_hi) > 0:
            logging.info(
                f"  episode {e}: waiting [{lo}, {hi}] -> [{new_lo}, {new_hi}] "
                f"(head -{new_lo - lo}, tail -{hi - new_hi})"
            )


def _apply_frozen_v35_d_valid(info: dict[str, np.ndarray]) -> None:
    """Apply pre-split 14D D-valid bounds without reading any sealed episode state.

    The frozen manifest is produced while structural/integrity QA is still allowed. During
    training, final-test observations must remain untouched, so recomputing the detector over
    all 70 state streams is forbidden. Semantic waiting labels remain in ``episode_tasks``;
    this sidecar only controls memory/read eligibility.
    """
    required = ("memory_lo", "memory_hi", "manifest_d_lo", "manifest_d_hi", "length")
    missing = [key for key in required if key not in info]
    if missing:
        raise ValueError(f"v3.5 frozen D_valid application is missing fields: {missing}.")
    semantic_lo = np.asarray(info["memory_lo"], dtype=np.int32).copy()
    semantic_hi = np.asarray(info["memory_hi"], dtype=np.int32).copy()
    d_lo = np.asarray(info["manifest_d_lo"], dtype=np.int32)
    d_hi = np.asarray(info["manifest_d_hi"], dtype=np.int32)
    if semantic_lo.shape != d_lo.shape or semantic_hi.shape != d_hi.shape:
        raise ValueError("v3.5 frozen semantic-D and D_valid arrays have inconsistent shapes.")
    valid: list[np.ndarray] = []
    for episode_index, length in enumerate(np.asarray(info["length"], dtype=np.int32)):
        lo, hi = int(semantic_lo[episode_index]), int(semantic_hi[episode_index])
        core_lo, core_hi = int(d_lo[episode_index]), int(d_hi[episode_index])
        if lo < 0 or hi < lo or not (lo <= core_lo <= core_hi <= hi):
            raise ValueError(
                "v3.5 frozen D_valid must be a nonempty subset of its semantic D phase: "
                f"episode={episode_index}, semantic=[{lo},{hi}], D_valid=[{core_lo},{core_hi}]."
            )
        episode_valid = np.ones(int(length), dtype=bool)
        episode_valid[lo:core_lo] = False
        episode_valid[core_hi + 1 : hi + 1] = False
        valid.append(episode_valid)
    info["memory_lo"] = d_lo.copy()
    info["memory_hi"] = d_hi.copy()
    info["memory_critical_ok"] = np.ones_like(d_lo, dtype=bool)
    info["episode_waiting_valid"] = tuple(valid)
    logging.info("v3.5 applied frozen manifest D_valid bounds without reading sealed state observations")


def _memory_critical_windows(info: dict[str, np.ndarray], data_config: "_config.DataConfig") -> np.ndarray | None:
    """[num_episodes, 4] int32 rows [start_lo, start_hi, memory_lo, memory_hi] for the
    memory-critical branch (see BuildMemorySequence), or None when phases are not configured.
    Rows of -1 disable the branch for that episode."""
    if "evidence_start" not in info:
        return None
    evidence_start = info["evidence_start"]
    start_lo = np.maximum(1, evidence_start - data_config.memory_critical_start_pad)
    window = np.stack([start_lo, evidence_start, info["memory_lo"], info["memory_hi"]], axis=1)
    # A row needs BOTH a usable evidence phase and a waiting phase that can hold an endpoint:
    # memory_critical_endpoint divides by the eligible-step count, so a row with no waiting
    # window must be disabled here rather than reaching it.
    disabled = (evidence_start < 0) | ~info.get("memory_critical_ok", info["memory_lo"] >= 0)
    window[disabled] = -1
    if data_config.memory_v35_enabled:
        required = (
            "final_e_limit",
            "occlusion_lo",
            "occlusion_hi",
            "execute_start",
            "collection_id",
            "object_id",
            "memory_cell",
            "sampling_allowed",
        )
        missing = [key for key in required if key not in info]
        if missing:
            raise ValueError(f"v3.5 phase/manifest table is missing fields: {missing}.")
        allowed = np.asarray(info["sampling_allowed"], dtype=bool)
        bad_training = allowed & disabled
        if np.any(bad_training):
            raise ValueError(
                "v3.5 active-split episodes have unusable E/O/D phases; no silent drop allowed: "
                f"{np.nonzero(bad_training)[0].tolist()}."
            )
        episode_index = np.arange(len(evidence_start), dtype=np.int32)
        marker = np.full(len(evidence_start), _transforms.V35_WINDOW_MARKER_VALUE, dtype=np.int32)
        window = np.column_stack(
            [
                window,
                info["final_e_limit"],
                info["occlusion_lo"],
                info["occlusion_hi"],
                info["execute_start"],
                episode_index,
                info["collection_id"],
                info["object_id"],
                info["memory_cell"],
                marker,
            ]
        )
    return window.astype(np.int32)


@dataclasses.dataclass(frozen=True)
class _SequenceSamplingInfo:
    weights: np.ndarray
    valid_steps: np.ndarray


def _sequence_sampling_info(
    dataset: Dataset,
    dataset_meta: lerobot_dataset.LeRobotDatasetMetadata,
    data_config: "_config.DataConfig",
    max_steps: int,
) -> _SequenceSamplingInfo | None:
    """Per-frame weights over sequence START frames (memory sequence training).

    A sample is a FULL trajectory (start = frame 0), a random contiguous SLICE, or -- when the
    config provides the label-derived phases -- a MEMORY-CRITICAL sample (start shortly before
    the evidence phase; BuildMemorySequence truncates it inside the memory-required phase).
    Branch masses: memory_critical_prob for the memory-critical branch, the rest split by
    memory_slice_prob as before. Slice starts must leave at least memory_min_slice_steps steps
    before the episode end and may not fall in the dead zone where a blank memory never saw
    the answer yet the labels ahead still demand it -- grading that teaches guessing. With
    phases the dead zone is (evidence_start, memory_hi] (from the FIRST evidence frame:
    partial evidence is as ungradable as none); without, the legacy (reveal, switch)
    rule applies. Memory-critical mass is balanced equally over (instruction, side) cells so
    no task or side dominates the waiting supervision.
    """
    inner = _unwrap_lerobot(dataset)
    if inner is None:
        return None
    info = _episode_info_table(dataset, dataset_meta, data_config)
    cols = inner.hf_dataset.with_format(None)
    episode = np.asarray(cols["episode_index"], dtype=np.int64)
    starts = np.nonzero(np.append(True, episode[1:] != episode[:-1]))[0]
    frame = np.arange(len(episode)) - starts[np.searchsorted(starts, np.arange(len(episode)), side="right") - 1]

    # Held-out evaluation episodes (v3.4: one per instruction x side cell) receive ZERO
    # sampling mass in every branch -- full trajectories, slices, and memory-critical starts.
    num_episodes = len(info["length"])
    heldout = np.asarray(sorted(set(data_config.heldout_episodes)), dtype=np.int64)
    if len(heldout) > 0:
        if heldout.min() < 0 or heldout.max() >= num_episodes:
            raise ValueError(f"heldout_episodes {heldout.tolist()} out of range [0, {num_episodes}).")
        logging.info(f"holding out {len(heldout)} episodes from ALL training sampling: {heldout.tolist()}")
    allowed_episode = np.asarray(info.get("sampling_allowed", np.ones(num_episodes, dtype=bool)), dtype=bool)
    if allowed_episode.shape != (num_episodes,):
        raise ValueError(f"sampling_allowed must have shape {(num_episodes,)}, got {allowed_episode.shape}.")
    allowed_episode = allowed_episode & ~np.isin(np.arange(num_episodes), heldout)
    allowed = allowed_episode[episode]

    stride = data_config.memory_stride_frames
    length = info["length"][episode]
    valid_steps = np.minimum((length - frame + stride - 1) // stride, max_steps).astype(np.int32)

    phases = "evidence_start" in info
    window = _memory_critical_windows(info, data_config)
    if phases:
        usable = info["evidence_start"][episode] >= 0
        memory_lo = info["memory_lo"][episode]
        memory_hi = info["memory_hi"][episode]
        # Dead from the first evidence frame on: a slice starting mid-inspection may already
        # have missed the revealing glimpse, yet the waiting labels ahead still grade the side
        # -- partial evidence teaches guessing just like no evidence.
        dead = usable & (frame > info["evidence_start"][episode]) & (frame <= memory_hi)
        in_window = usable & (frame >= window[:, 0][episode]) & (frame <= window[:, 1][episode])
        # the waiting phase must be reachable within the sequence budget
        mc_ok = in_window & (memory_lo - frame <= (max_steps - 1) * stride) & allowed
        critical_family = np.full(len(episode), -1, dtype=np.int8)  # 0 natural, 1 sparse skip-O
        sampled_e_count = np.zeros(len(episode), dtype=np.int32)
        critical_delay = np.full(len(episode), -1, dtype=np.int32)
        # Each memory-critical start truncates at memory_critical_endpoint's DETERMINISTIC
        # step, so its exact valid length is known here and bucket assignment is precise --
        # BuildMemorySequence recomputes the identical endpoint at fetch time. (A per-draw
        # random endpoint would mix bucket lengths inside one batch and trip the homogeneity
        # check in _sequence_bucket_collate_fn.)
        for i in np.nonzero(mc_ok)[0]:
            if data_config.memory_v35_enabled:
                try:
                    layout = _transforms.memory_critical_layout(
                        int(frame[i]),
                        window[episode[i]],
                        stride=stride,
                        lookahead=data_config.subtask_lookahead,
                        num_steps=max_steps,
                    )
                except ValueError:
                    # Individual raw start residues may miss a short D interval.  They receive
                    # zero mass; the per-episode gates below ensure the episode itself cannot
                    # disappear or lose either family silently.
                    mc_ok[i] = False
                    continue
                valid_steps[i] = len(layout.keep_indices)
                critical_family[i] = 1 if layout.sparse_skip_o else 0
                sampled_e_count[i] = layout.sampled_e_count
                critical_delay[i] = layout.n_delay
            else:
                t_q = _transforms.memory_critical_endpoint(
                    int(frame[i]),
                    window[episode[i]],
                    stride=stride,
                    lookahead=data_config.subtask_lookahead,
                    num_steps=max_steps,
                )
                valid_steps[i] = t_q + 1

        if data_config.memory_v35_enabled:
            active_episodes = np.nonzero(allowed_episode)[0]
            missing_any: list[int] = []
            missing_sparse: list[int] = []
            for e in active_episodes:
                episode_candidates = mc_ok & (episode == e)
                if not np.any(episode_candidates):
                    missing_any.append(int(e))
                    continue
                # Prefer >=2 sampled E steps whenever the episode/grid offers them; the hard
                # floor remains one for genuinely short evidence intervals.  Apply this per
                # family: a two-E natural candidate must not accidentally delete an episode's
                # only valid one-E skip-O candidate.
                for family in (0, 1):
                    family_candidates = episode_candidates & (critical_family == family)
                    two_e = family_candidates & (sampled_e_count >= 2)
                    if np.any(two_e):
                        mc_ok[family_candidates & (sampled_e_count < 2)] = False
                remaining = mc_ok & (episode == e)
                if not np.any(remaining):
                    missing_any.append(int(e))
                elif not np.any(remaining & (critical_family == 1)):
                    missing_sparse.append(int(e))
            if missing_any or missing_sparse:
                raise ValueError(
                    "v3.5 active-split episode critical coverage failure (no silent drop): "
                    f"no_E_to_D_candidate={missing_any}, no_skip_O_candidate={missing_sparse}."
                )
            per_episode_e = np.asarray(
                [int(sampled_e_count[mc_ok & (episode == e)].min()) for e in active_episodes], dtype=np.int32
            )
            if np.any(per_episode_e < 1):
                raise ValueError("v3.5 sampled-E hard floor failed after critical candidate filtering.")
            one_e = active_episodes[per_episode_e == 1]
            logging.info(
                "v3.5 critical E coverage: active=%d min/p10/p50/p90/max=%s; one-E episodes=%s",
                len(active_episodes),
                np.percentile(per_episode_e, [0, 10, 50, 90, 100]).tolist(),
                one_e.tolist(),
            )
            for family, name in ((0, "natural"), (1, "skip_o")):
                delay = critical_delay[mc_ok & (critical_family == family)]
                if len(delay) == 0:
                    raise ValueError(f"v3.5 active split has no {name} critical candidates.")
                logging.info(
                    "v3.5 %s candidates=%d n_delay min/p10/p50/p90/max=%s",
                    name,
                    len(delay),
                    np.percentile(delay, [0, 10, 50, 90, 100]).tolist(),
                )
    else:
        dead = (frame > info["reveal"][episode]) & (frame < info["switch"][episode])
        in_window = np.zeros(len(episode), dtype=bool)
        mc_ok = in_window
        critical_family = np.full(len(episode), -1, dtype=np.int8)

    mc_prob = data_config.memory_critical_prob if int(mc_ok.sum()) > 0 else 0.0
    if data_config.memory_v35_enabled and data_config.memory_critical_prob > 0 and mc_prob == 0.0:
        raise ValueError("v3.5 has no eligible memory-critical starts; refusing to disable the branch silently.")
    if data_config.memory_critical_prob > 0 and mc_prob == 0.0:
        logging.warning("memory_critical_prob > 0 but no eligible memory-critical starts; branch disabled.")

    min_frames = data_config.memory_min_slice_steps * stride
    unanchored_d = np.zeros(len(episode), dtype=bool)
    if data_config.memory_v35_enabled:
        # Full and ordinary slice starts can also land on strict D.  Apply the same hard
        # final-E rule as the critical branch before assigning them any probability; relying
        # only on the preprocessing assertion would turn a resample condition into a random
        # mid-run crash.  Dense sequences naturally include their final same-grid eligible E
        # whenever `has_e` is true.
        def grid_intersects(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
            first_k = np.maximum(0, -np.floor_divide(-(lo - frame), stride))
            last_frame = np.minimum(hi, length - 1)
            last_k = np.minimum(max_steps - 1, np.floor_divide(last_frame - frame, stride))
            return first_k <= last_k

        has_d = grid_intersects(info["memory_lo"][episode], info["memory_hi"][episode])
        has_e = grid_intersects(info["evidence_start"][episode], info["final_e_limit"][episode])
        unanchored_d = allowed & has_d & ~has_e
        if np.any(unanchored_d):
            logging.info(
                "v3.5 excluded %d ordinary starts containing strict D without a same-grid final-E anchor",
                int(unanchored_d.sum()),
            )
    # ~in_window (not just ~mc_ok): a window start drawn as a slice would still be truncated
    # by BuildMemorySequence, so window frames belong to the memory-critical branch or nothing
    slice_ok = (frame > 0) & (frame + min_frames <= length) & ~dead & ~in_window & ~unanchored_d & allowed

    slice_prob = data_config.memory_slice_prob
    weights = np.zeros(len(episode), dtype=np.float64)
    full_ok = (frame == 0) & ~unanchored_d & allowed
    n_full = int(full_ok.sum())
    n_slice = int(slice_ok.sum())
    critical_mass = mc_prob
    full_mass = (1.0 - mc_prob) * (1.0 - slice_prob)
    slice_mass = (1.0 - mc_prob) * slice_prob
    if data_config.memory_v35_enabled:
        # Anchor filtering can remove every full (or slice) start.  Preserve the requested
        # critical/non-critical split instead of dropping probability mass and letting the
        # downstream sampler silently renormalize it into a different branch mixture.
        if n_full == 0 and n_slice > 0:
            slice_mass = 1.0 - mc_prob
            full_mass = 0.0
        elif n_slice == 0 and n_full > 0:
            full_mass = 1.0 - mc_prob
            slice_mass = 0.0
        elif n_slice == 0 and n_full == 0:
            if mc_prob <= 0:
                raise ValueError("v3.5 has no anchored full, slice, or memory-critical sequence starts.")
            critical_mass = 1.0
            full_mass = slice_mass = 0.0
    if n_full > 0:
        weights[full_ok] = full_mass / n_full
    if n_slice > 0:
        weights[slice_ok] = slice_mass / n_slice

    n_cells = 0
    if mc_prob > 0:
        mc_episodes = np.unique(episode[mc_ok])
        if data_config.memory_v35_enabled:
            # Equal marginal mass over natural/skip-O x stable manifest cell, then over
            # episodes and starts.  Bucket batching preserves this marginal but does not force
            # exact cell composition inside each individual batch.
            strata: dict[tuple[int, int], dict[int, np.ndarray]] = {}
            for e in mc_episodes:
                for family in (0, 1):
                    idx = np.nonzero(mc_ok & (episode == e) & (critical_family == family))[0]
                    if len(idx):
                        strata.setdefault((family, int(info["memory_cell"][e])), {})[int(e)] = idx
            expected = {
                (family, int(cell)) for family in (0, 1) for cell in np.unique(info["memory_cell"][allowed_episode])
            }
            missing_strata = sorted(expected - set(strata))
            if missing_strata:
                raise ValueError(f"v3.5 natural/skip-O x manifest-cell strata are missing: {missing_strata}.")
            n_cells = len(strata)
            for episode_indices in strata.values():
                for idx in episode_indices.values():
                    weights[idx] = critical_mass / n_cells / len(episode_indices) / len(idx)
        else:
            prompts = _load_episode_prompts(dataset) or ()
            cells: dict[tuple[str, int], list[int]] = {}
            for e in mc_episodes:
                key = (prompts[e] if e < len(prompts) else "", int(info["side"][e]))
                cells.setdefault(key, []).append(int(e))
            n_cells = len(cells)
            for members in cells.values():
                for e in members:
                    idx = np.nonzero(mc_ok & (episode == e))[0]
                    weights[idx] = mc_prob / n_cells / len(members) / len(idx)

    logging.info(
        f"sequence sampling: {n_full} full starts (p={full_mass:.3g}), "
        f"{n_slice} slice starts (p={slice_mass:.3g}), "
        f"{int(mc_ok.sum())} memory-critical starts (p={critical_mass:g}, {n_cells} balance cells), "
        f"{int((~slice_ok & ~mc_ok & (frame > 0)).sum())} frames excluded (dead zone / too close to end)"
    )
    return _SequenceSamplingInfo(weights=weights, valid_steps=valid_steps)


def _validate_sequence_buckets(buckets: Sequence[int], max_steps: int) -> tuple[int, ...]:
    buckets = tuple(int(x) for x in buckets)
    if not buckets:
        return ()
    if any(x <= 0 for x in buckets) or tuple(sorted(set(buckets))) != buckets:
        raise ValueError(f"memory_sequence_buckets must be positive and strictly increasing; got {buckets}.")
    if buckets[-1] != max_steps:
        raise ValueError(
            f"the final memory_sequence_buckets entry must equal memory_seq_steps ({max_steps}); got {buckets[-1]}."
        )
    return buckets


def _sequence_bucket_ids(valid_steps: np.ndarray, buckets: Sequence[int]) -> np.ndarray:
    bucket_array = np.asarray(buckets, dtype=np.int32)
    valid_steps = np.asarray(valid_steps, dtype=np.int32)
    if bucket_array.ndim != 1 or len(bucket_array) == 0:
        raise ValueError("at least one sequence bucket is required.")
    if np.any(valid_steps <= 0):
        raise ValueError("sequence valid lengths must be positive.")
    bucket_ids = np.searchsorted(bucket_array, valid_steps, side="left")
    if np.any(bucket_ids == len(bucket_array)):
        raise ValueError(
            f"sequence length {int(valid_steps.max())} exceeds the largest bucket {int(bucket_array[-1])}."
        )
    return bucket_ids.astype(np.int32)


class SequenceBucketBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Samples homogeneous-shape batches without changing the marginal start distribution.

    A bucket is drawn according to the total original sampling weight assigned to it, then all
    indices in the batch are drawn with replacement from that bucket using their conditional
    weights. Therefore every individual draw still has exactly the same marginal probability
    as WeightedRandomSampler; only within-batch lengths become correlated.
    """

    def __init__(
        self,
        weights: np.ndarray,
        valid_steps: np.ndarray,
        buckets: Sequence[int],
        batch_size: int,
        *,
        generator: torch.Generator,
        num_samples: int | None = None,
    ):
        weights = torch.as_tensor(np.asarray(weights), dtype=torch.float64)
        valid_steps = np.asarray(valid_steps, dtype=np.int32)
        if weights.ndim != 1 or valid_steps.shape != tuple(weights.shape):
            raise ValueError("weights and valid_steps must be one-dimensional arrays with matching lengths.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if torch.any(weights < 0) or not torch.isfinite(weights).all() or float(weights.sum()) <= 0:
            raise ValueError("sampling weights must be finite, nonnegative, and have positive total mass.")

        if not buckets:
            raise ValueError("at least one sequence bucket is required.")
        self.buckets = _validate_sequence_buckets(buckets, int(tuple(buckets)[-1]))
        bucket_ids = _sequence_bucket_ids(valid_steps, self.buckets)
        positive = weights > 0
        self._indices: list[torch.Tensor] = []
        self._conditional_weights: list[torch.Tensor] = []
        masses = []
        active_buckets = []
        for bucket_id, bucket_steps in enumerate(self.buckets):
            indices = torch.from_numpy(np.nonzero((bucket_ids == bucket_id) & np.asarray(positive))[0])
            if len(indices) == 0:
                continue
            bucket_weights = weights[indices]
            mass = bucket_weights.sum()
            if float(mass) <= 0:
                continue
            active_buckets.append(bucket_steps)
            self._indices.append(indices)
            self._conditional_weights.append(bucket_weights)
            masses.append(mass)

        if not masses:
            raise ValueError("no positive-weight sequence starts were assigned to a bucket.")
        self.active_buckets = tuple(active_buckets)
        self._bucket_masses = torch.stack(masses)
        self._batch_size = batch_size
        self._num_samples = len(weights) if num_samples is None else int(num_samples)
        if self._num_samples < batch_size:
            raise ValueError("num_samples must be at least batch_size.")
        self._generator = generator

        total_mass = float(self._bucket_masses.sum())
        logging.info(
            "sequence bucket sampling: "
            + ", ".join(
                f"T{steps}={100 * float(mass) / total_mass:.2f}%"
                for steps, mass in zip(self.active_buckets, self._bucket_masses, strict=True)
            )
        )

    def __iter__(self):
        for _ in range(len(self)):
            bucket_id = int(torch.multinomial(self._bucket_masses, 1, replacement=True, generator=self._generator))
            local = torch.multinomial(
                self._conditional_weights[bucket_id],
                self._batch_size,
                replacement=True,
                generator=self._generator,
            )
            yield self._indices[bucket_id][local].tolist()

    def __len__(self) -> int:
        return self._num_samples // self._batch_size

    def generator_state(self) -> torch.Tensor:
        """Return a copy so a checkpoint cannot race with later sampler draws."""
        return self._generator.get_state().clone()

    def set_generator_state(self, state: torch.Tensor) -> None:
        self._generator.set_state(state)


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(
        f"data_config: repo_id={data_config.repo_id} asset_id={data_config.asset_id} "
        f"norm_stats={sorted(data_config.norm_stats) if data_config.norm_stats else None} "
        f"transforms={[type(t).__name__ for g in (data_config.repack_transforms, data_config.data_transforms, data_config.model_transforms) for t in g.inputs]}"
    )

    if data_config.rlds_data_dir is not None:
        if config.gradient_accumulation_steps != 1:
            raise NotImplementedError("Gradient accumulation is currently supported only by the JAX TorchDataLoader.")
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        exact_resume=getattr(config.model, "memory_v35_enabled", False),
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
    gradient_accumulation_steps: int = 1,
    exact_resume: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    if framework != "jax" and gradient_accumulation_steps != 1:
        raise NotImplementedError("Gradient accumulation is supported only by the JAX trainer.")

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    batch_sampler = None
    collate_fn = _collate_fn
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()
        use_memory = getattr(model_config, "predict_with_memory", False) and data_config.memory_stride_frames > 0
        if shuffle and use_memory:
            dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(
                data_config.repo_id,
                root=data_config.lerobot_dataset_root,
            )
            sampling = _sequence_sampling_info(dataset, dataset_meta, data_config, model_config.memory_seq_steps)
            if sampling is not None:
                generator = torch.Generator()
                generator.manual_seed(seed)
                buckets = _validate_sequence_buckets(data_config.memory_sequence_buckets, model_config.memory_seq_steps)
                if buckets:
                    batch_sampler = SequenceBucketBatchSampler(
                        sampling.weights,
                        sampling.valid_steps,
                        buckets,
                        local_batch_size,
                        generator=generator,
                    )
                    collate_fn = functools.partial(
                        _sequence_bucket_collate_fn,
                        buckets=buckets,
                        max_steps=model_config.memory_seq_steps,
                    )
                else:
                    sampler = torch.utils.data.WeightedRandomSampler(
                        torch.as_tensor(sampling.weights),
                        num_samples=len(sampling.weights),
                        replacement=True,
                        generator=generator,
                    )

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and batch_sampler is None and shuffle),
        sampler=sampler,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
        gradient_accumulation_steps=gradient_accumulation_steps,
        exact_resume=exact_resume,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


_EXACT_RESUME_STATE_SCHEMA_VERSION = 1


def _encode_torch_rng_state(state: torch.Tensor) -> str:
    values = np.asarray(state.cpu(), dtype=np.uint8)
    return base64.b64encode(values.tobytes(order="C")).decode("ascii")


def _decode_torch_rng_state(encoded: str, *, description: str) -> torch.Tensor:
    if not isinstance(encoded, str):
        raise ValueError(f"{description} must be base64 text.")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{description} is not valid base64.") from exc
    if not raw:
        raise ValueError(f"{description} is empty.")
    return torch.from_numpy(np.frombuffer(raw, dtype=np.uint8).copy())


def _capture_numpy_random_state() -> dict[str, typing.Any]:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "algorithm": algorithm,
        "keys_u32_le_base64": base64.b64encode(np.asarray(keys, dtype="<u4").tobytes(order="C")).decode("ascii"),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _decode_numpy_random_state(value: typing.Any) -> tuple[str, np.ndarray, int, int, float]:
    if not isinstance(value, dict) or set(value) != {
        "algorithm",
        "keys_u32_le_base64",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise ValueError("NumPy random state has an invalid schema.")
    try:
        raw = base64.b64decode(value["keys_u32_le_base64"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("NumPy random-state keys are not valid base64.") from exc
    keys = np.frombuffer(raw, dtype="<u4").astype(np.uint32, copy=True)
    if value["algorithm"] != "MT19937" or keys.shape != (624,):
        raise ValueError("Only the canonical NumPy MT19937 module RNG state is supported.")
    position = value["position"]
    has_gauss = value["has_gauss"]
    cached_gaussian = value["cached_gaussian"]
    if not isinstance(position, int) or not 0 <= position <= 624:
        raise ValueError("NumPy random-state position is invalid.")
    if has_gauss not in (0, 1) or not isinstance(cached_gaussian, int | float):
        raise ValueError("NumPy Gaussian-cache state is invalid.")
    return "MT19937", keys, position, has_gauss, float(cached_gaussian)


def _capture_python_random_state() -> dict[str, typing.Any]:
    version, internal, gauss_next = random.getstate()
    return {
        "version": int(version),
        "internal": [int(item) for item in internal],
        "gauss_next": None if gauss_next is None else float(gauss_next),
    }


def _decode_python_random_state(value: typing.Any) -> tuple[int, tuple[int, ...], float | None]:
    if not isinstance(value, dict) or set(value) != {"version", "internal", "gauss_next"}:
        raise ValueError("Python random state has an invalid schema.")
    version = value["version"]
    internal = value["internal"]
    gauss_next = value["gauss_next"]
    if version != 3 or not isinstance(internal, list) or len(internal) != 625:
        raise ValueError("Only the canonical Python random version-3 state is supported.")
    if any(not isinstance(item, int) for item in internal):
        raise ValueError("Python random internal state must contain integers.")
    if gauss_next is not None and not isinstance(gauss_next, int | float):
        raise ValueError("Python random Gaussian-cache state is invalid.")
    return version, tuple(internal), None if gauss_next is None else float(gauss_next)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        batch_sampler: torch.utils.data.Sampler[list[int]] | None = None,
        collate_fn: typing.Callable | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
        gradient_accumulation_steps: int = 1,
        exact_resume: bool = False,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")
        if gradient_accumulation_steps < 1 or local_batch_size % gradient_accumulation_steps != 0:
            raise ValueError(
                f"Local batch size {local_batch_size} must be divisible by positive gradient accumulation steps "
                f"{gradient_accumulation_steps}."
            )
        self._gradient_accumulation_steps = gradient_accumulation_steps
        self._exact_resume = exact_resume
        self._batches_yielded = 0
        self._dataset_size = len(dataset)
        self._local_batch_size = local_batch_size
        self._batch_sampler = batch_sampler
        if exact_resume:
            if num_workers != 0:
                raise ValueError("exact-resume data loading requires num_workers=0 (no worker prefetch).")
            if not isinstance(batch_sampler, SequenceBucketBatchSampler):
                raise ValueError("exact-resume data loading requires SequenceBucketBatchSampler.")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B")
                if gradient_accumulation_steps == 1
                else jax.sharding.PartitionSpec(None, "B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        common_kwargs = {
            "num_workers": num_workers,
            "multiprocessing_context": mp_context,
            "persistent_workers": num_workers > 0,
            "collate_fn": _collate_fn if collate_fn is None else collate_fn,
            "worker_init_fn": _worker_init_fn,
            "generator": generator,
        }
        if batch_sampler is None:
            self._data_loader = torch.utils.data.DataLoader(
                typing.cast(torch.utils.data.Dataset, dataset),
                batch_size=local_batch_size,
                shuffle=(sampler is None and shuffle),
                sampler=sampler,
                drop_last=True,
                **common_kwargs,
            )
        else:
            if sampler is not None or shuffle:
                raise ValueError("batch_sampler is mutually exclusive with sampler and shuffle.")
            self._data_loader = torch.utils.data.DataLoader(
                typing.cast(torch.utils.data.Dataset, dataset),
                batch_sampler=batch_sampler,
                **common_kwargs,
            )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                if self._gradient_accumulation_steps > 1:
                    accumulation_steps = self._gradient_accumulation_steps
                    microbatch_size = next(iter(jax.tree.leaves(batch))).shape[0] // accumulation_steps
                    batch = jax.tree.map(
                        lambda x, steps=accumulation_steps, size=microbatch_size: x.reshape(steps, size, *x.shape[1:]),
                        batch,
                    )
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    result = jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    result = jax.tree.map(torch.as_tensor, batch)
                self._batches_yielded += 1
                yield result

    def _exact_resume_fingerprint(self) -> dict[str, typing.Any]:
        if not isinstance(self._batch_sampler, SequenceBucketBatchSampler):
            raise RuntimeError("exact-resume loader lost its sequence bucket sampler.")
        return {
            "dataset_size": self._dataset_size,
            "local_batch_size": self._local_batch_size,
            "gradient_accumulation_steps": self._gradient_accumulation_steps,
            "sampler": "SequenceBucketBatchSampler",
            "buckets": list(self._batch_sampler.buckets),
            "active_buckets": list(self._batch_sampler.active_buckets),
            "batches_per_sampler_epoch": len(self._batch_sampler),
        }

    def state_dict(self) -> dict[str, typing.Any]:
        """Snapshot every host RNG that can affect a no-worker v3.5 batch.

        The sequence sampler draws one batch at a time. With no worker processes or prefetch,
        its generator state after an accepted update points exactly at the following batch.
        Module-level NumPy/Python/Torch states cover stochastic host transforms.
        """
        if not self._exact_resume:
            raise RuntimeError("exact-resume state is available only for an exact-resume loader.")
        if not isinstance(self._batch_sampler, SequenceBucketBatchSampler):
            raise RuntimeError("exact-resume loader requires SequenceBucketBatchSampler.")
        return {
            "schema_version": _EXACT_RESUME_STATE_SCHEMA_VERSION,
            "fingerprint": self._exact_resume_fingerprint(),
            "batches_yielded": self._batches_yielded,
            "sampler_generator_state_base64": _encode_torch_rng_state(self._batch_sampler.generator_state()),
            "numpy_random_state": _capture_numpy_random_state(),
            "python_random_state": _capture_python_random_state(),
            "torch_cpu_random_state_base64": _encode_torch_rng_state(torch.random.get_rng_state()),
        }

    def load_state_dict(self, state: dict[str, typing.Any]) -> None:
        """Restore before ``iter(loader)`` so no resumed batch is drawn from fresh state."""
        if not self._exact_resume:
            raise RuntimeError("cannot restore exact-resume state into a legacy loader.")
        expected_keys = {
            "schema_version",
            "fingerprint",
            "batches_yielded",
            "sampler_generator_state_base64",
            "numpy_random_state",
            "python_random_state",
            "torch_cpu_random_state_base64",
        }
        if not isinstance(state, dict) or set(state) != expected_keys:
            raise ValueError("exact-resume data iterator state has an invalid schema.")
        if state["schema_version"] != _EXACT_RESUME_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported exact-resume data iterator state version.")
        if state["fingerprint"] != self._exact_resume_fingerprint():
            raise ValueError("exact-resume data iterator fingerprint does not match this loader.")
        batches_yielded = state["batches_yielded"]
        if not isinstance(batches_yielded, int) or batches_yielded < 0:
            raise ValueError("exact-resume batches_yielded must be a nonnegative integer.")

        # Decode and validate everything before mutating process-global RNG state.
        sampler_state = _decode_torch_rng_state(
            state["sampler_generator_state_base64"], description="sampler generator state"
        )
        numpy_state = _decode_numpy_random_state(state["numpy_random_state"])
        python_state = _decode_python_random_state(state["python_random_state"])
        torch_state = _decode_torch_rng_state(
            state["torch_cpu_random_state_base64"], description="Torch CPU random state"
        )
        if not isinstance(self._batch_sampler, SequenceBucketBatchSampler):
            raise RuntimeError("exact-resume loader requires SequenceBucketBatchSampler.")
        self._batch_sampler.set_generator_state(sampler_state)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.random.set_rng_state(torch_state)
        self._batches_yielded = batches_yielded


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


_SEQUENCE_TIME_KEYS = frozenset(
    {
        "image",
        "image_mask",
        "state",
        "actions",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "token_ar_mask",
        "token_loss_mask",
        "token_fast_mask",
        "token_state_mask",
        "tokenized_causal",
        "tokenized_causal_mask",
        "causal_fast_mask",
        "seq_step_mask",
        "seq_block_boundary",
        "seq_probe_labels",
        "seq_probe_mask",
        "seq_probe_visible",
        # v3.4 per-step supervision (seq_state_masked and seq_side_label are per-SEGMENT
        # scalars and deliberately not listed).
        "seq_subtask_class",
        "seq_evidence_mask",
        "seq_waiting_mask",
    }
)


def _sequence_bucket_collate_fn(items, *, buckets: Sequence[int], max_steps: int):
    """Trim a homogeneous bucket batch before stacking and transferring it to JAX."""
    if not items:
        raise ValueError("cannot collate an empty sequence batch.")
    valid_steps = np.asarray([np.count_nonzero(np.asarray(item["seq_step_mask"])) for item in items])
    bucket_ids = _sequence_bucket_ids(valid_steps, buckets)
    if np.any(bucket_ids != bucket_ids[0]):
        raise ValueError(f"sequence bucket batch is not homogeneous: valid lengths {valid_steps.tolist()}.")
    bucket_steps = int(tuple(buckets)[int(bucket_ids[0])])

    def trim_time_axis(x):
        x = np.asarray(x)
        if x.ndim == 0 or x.shape[0] != max_steps:
            raise ValueError(f"expected a temporal leaf with leading length {max_steps}; got shape {x.shape}.")
        return x[:bucket_steps]

    trimmed = []
    for item in items:
        unknown_temporal = [
            key
            for key, value in item.items()
            if key not in _SEQUENCE_TIME_KEYS
            and any(np.asarray(x).ndim > 0 and np.asarray(x).shape[0] == max_steps for x in jax.tree.leaves(value))
        ]
        if unknown_temporal:
            raise ValueError(f"unregistered temporal sequence fields: {unknown_temporal}.")
        trimmed.append(
            {
                key: jax.tree.map(trim_time_axis, value) if key in _SEQUENCE_TIME_KEYS else value
                for key, value in item.items()
            }
        )
    return _collate_fn(trimmed)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    # Fork inherits one numpy RNG state into every worker; without reseeding, transforms that
    # draw from np.random (e.g. the TBPTT fence shift) produce the SAME stream in all workers.
    # torch.initial_seed() is already per-worker (base_seed + worker_id) and derives from the
    # loader's torch.Generator, so numpy stays reproducible under a fixed training seed.
    np.random.seed(torch.initial_seed() % 2**32)


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]

    def state_dict(self) -> dict[str, typing.Any]:
        if not isinstance(self._data_loader, TorchDataLoader):
            raise RuntimeError("exact-resume state is not supported for RLDS data loaders.")
        return self._data_loader.state_dict()

    def load_state_dict(self, state: dict[str, typing.Any]) -> None:
        if not isinstance(self._data_loader, TorchDataLoader):
            raise RuntimeError("exact-resume state is not supported for RLDS data loaders.")
        self._data_loader.load_state_dict(state)
