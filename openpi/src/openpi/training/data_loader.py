from collections.abc import Iterator, Sequence
import dataclasses
import json
import logging
import multiprocessing
import os
import pathlib
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
        # memory plumbing untested on fake data: make the newest half of the writes valid and,
        # when the quiz is enabled, quizzable with valid class labels.
        if observation.memory_write_mask is not None:
            nw = observation.memory_write_mask.shape[0]
            observation = dataclasses.replace(observation, memory_write_mask=jnp.arange(nw) >= nw // 2)
        if observation.memory_probe_labels is not None:
            nw = observation.memory_probe_labels.shape[0]
            observation = dataclasses.replace(
                observation,
                memory_probe_labels=observation.memory_probe_labels % 2,
                memory_probe_mask=observation.memory_write_mask,
                memory_probe_visible=observation.memory_write_mask & (jnp.arange(nw) < 3 * nw // 4),
            )

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def create_torch_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    model_config: _model.BaseModelConfig,
    *,
    window_frames: int | None = None,
) -> Dataset:
    """Create a dataset for training.

    `window_frames` overrides the number of live write-window frames fetched per sample
    (memory co-training with window-size buckets: one dataset per bucket, since lerobot's
    delta_timestamps are fixed at creation).
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    delta_timestamps = {
        key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
    }
    if data_config.subtask_from_task and data_config.subtask_lookahead > 0:
        # Deliver the *future* task_index: the subtask label that conditions this frame's chunk.
        delta_timestamps["task_index"] = [data_config.subtask_lookahead / dataset_meta.fps]
    use_memory = getattr(model_config, "predict_with_memory", False) and data_config.memory_stride_frames > 0
    if use_memory:
        # Memory co-training: also fetch the newest live write-window frames, stacked oldest
        # first with the current frame riding along at offset 0 (split by SplitMemoryWindow).
        live = window_frames if window_frames is not None else model_config.memory_live_writes
        stride = data_config.memory_stride_frames
        offsets = [-(live - j) * stride / dataset_meta.fps for j in range(live)] + [0.0]
        for key in ("image", "left_wrist_image", "right_wrist_image", "state"):
            delta_timestamps[key] = offsets
    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, delta_timestamps=delta_timestamps)

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    if data_config.subtask_from_task:
        dataset = TransformedDataset(dataset, [_transforms.SubtaskFromLeRobotTask(dataset_meta.tasks)])

    if use_memory and getattr(model_config, "memory_probe_weight", 0) > 0:
        side, reveal, close = _episode_quiz_table(dataset, dataset_meta, data_config.memory_reveal_frames_path)
        dataset = TransformedDataset(dataset, [_transforms.MemoryQuizInfo(side, reveal, close)])

    return dataset


# Quiz defaults when no reveal-frames json is provided: the bin contents become visible between
# frames ~200 and ~300 in every bin_memory_banana episode, so 300 is the conservative
# "definitely revealed by now" bound; the bins are closed again by ~450.
_DEFAULT_REVEAL_FRAME = 300
_DEFAULT_VISIBLE_SPAN = 150


def _episode_quiz_table(
    dataset: Dataset, dataset_meta: lerobot_dataset.LeRobotDatasetMetadata, reveal_frames_path: str | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-episode quiz supervision: the answer class from each episode's FINAL subtask label
    ("open left bin" -> 0, "open right bin" -> 1, otherwise -1 = never quizzed) plus the reveal
    and close frames from the optional json ({"<episode_index>": reveal | [reveal, close]}).
    """
    inner = dataset
    while isinstance(inner, TransformedDataset):
        inner = inner._dataset  # noqa: SLF001
    cols = inner.hf_dataset.with_format(None)
    task = np.asarray(cols["task_index"], dtype=np.int64)
    episode = np.asarray(cols["episode_index"], dtype=np.int64)

    num_episodes = int(episode.max()) + 1
    side = np.full(num_episodes, -1, dtype=np.int32)
    last_of_episode = np.nonzero(np.append(episode[1:] != episode[:-1], True))[0]
    for e, last in zip(episode[last_of_episode], last_of_episode, strict=True):
        final_task = dataset_meta.tasks[int(task[last])].lower()
        if "left" in final_task:
            side[e] = 0
        elif "right" in final_task:
            side[e] = 1

    reveal = np.full(num_episodes, _DEFAULT_REVEAL_FRAME, dtype=np.int32)
    close = reveal + _DEFAULT_VISIBLE_SPAN
    labeled = 0
    if reveal_frames_path is not None and pathlib.Path(reveal_frames_path).exists():
        for key, value in json.loads(pathlib.Path(reveal_frames_path).read_text()).items():
            e = int(key)
            if not 0 <= e < num_episodes:
                raise ValueError(f"reveal_frames json: episode {e} out of range [0, {num_episodes})")
            reveal[e], close[e] = (value, value + _DEFAULT_VISIBLE_SPAN) if np.isscalar(value) else value
            labeled += 1
    logging.info(
        f"quiz table: {int((side >= 0).sum())}/{num_episodes} episodes labeled "
        f"({int((side == 0).sum())} left / {int((side == 1).sum())} right), "
        f"reveal frames: {labeled} from json, {num_episodes - labeled} at default {_DEFAULT_REVEAL_FRAME}"
    )
    return side, reveal, close


def _decision_frame_weights(dataset: Dataset, lookahead: int, weight: float) -> np.ndarray | None:
    """Per-frame sampling weights that oversample the subtask decision regions.

    A decision region is the +-lookahead frames around each within-episode task change: the
    frames whose (lookahead-shifted) subtask label cannot be predicted from the current
    observation alone, i.e. the only frames whose CE pressures the memory read.
    """
    inner = dataset
    while isinstance(inner, TransformedDataset):
        inner = inner._dataset  # noqa: SLF001
    if not isinstance(inner, lerobot_dataset.LeRobotDataset):
        return None
    cols = inner.hf_dataset.with_format(None)
    task = np.asarray(cols["task_index"], dtype=np.int64)
    episode = np.asarray(cols["episode_index"], dtype=np.int64)
    weights = np.ones(len(task), dtype=np.float64)
    switches = np.nonzero((task[1:] != task[:-1]) & (episode[1:] == episode[:-1]))[0] + 1
    if len(switches) == 0:
        return None
    for s in switches:
        weights[max(s - lookahead, 0) : s + lookahead] = weight
    region = weights > 1
    logging.info(
        f"decision oversampling: {len(switches)} switches, {int(region.sum())} of {len(weights)} frames "
        f"x{weight:g} -> {weights[region].sum() / weights.sum():.0%} of samples"
    )
    return weights


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
    buckets = tuple(data_config.memory_window_buckets)
    use_buckets = (
        len(buckets) > 0
        and framework == "jax"
        and shuffle
        and getattr(model_config, "predict_with_memory", False)
        and data_config.memory_stride_frames > 0
    )

    def build_one(window_frames: int | None, loader_seed: int, workers: int) -> "TorchDataLoader":
        dataset = create_torch_dataset(data_config, action_horizon, model_config, window_frames=window_frames)
        dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

        # Use TorchDataLoader for both frameworks
        # For PyTorch DDP, create DistributedSampler and divide batch size by world size
        # For JAX, divide by process count
        sampler = None
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
            if shuffle and data_config.memory_decision_oversample > 1 and data_config.subtask_lookahead > 0:
                weights = _decision_frame_weights(
                    dataset, data_config.subtask_lookahead, data_config.memory_decision_oversample
                )
                if weights is not None:
                    generator = torch.Generator()
                    generator.manual_seed(loader_seed)
                    sampler = torch.utils.data.WeightedRandomSampler(
                        torch.as_tensor(weights), num_samples=len(weights), replacement=True, generator=generator
                    )

        logging.info(f"local_batch_size: {local_batch_size}")
        return TorchDataLoader(
            dataset,
            local_batch_size=local_batch_size,
            sharding=None if framework == "pytorch" else sharding,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_batches=num_batches,
            num_workers=workers,
            seed=loader_seed,
            framework=framework,
        )

    if not use_buckets:
        return DataLoaderImpl(data_config, build_one(None, seed, num_workers))

    # Window-size buckets: one loader per bucket (delta_timestamps are fixed at dataset
    # creation), each yielding whole batches of its own shape; a seeded RNG picks the bucket for
    # every batch. The worker budget is split across the buckets -- they all prefetch in
    # parallel, so total decode throughput is what matters.
    workers_each = num_workers // len(buckets) if num_workers > 0 else 0
    loaders = [build_one(window, seed + 1000 * i, workers_each) for i, window in enumerate(buckets)]
    logging.info(f"window buckets: {buckets}, {workers_each} workers each")
    return DataLoaderImpl(data_config, BucketedTorchLoader(loaders, seed=seed))


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
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
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

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
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
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


class BucketedTorchLoader:
    """Round-robins full batches from several TorchDataLoaders (one per window-size bucket).

    Every yielded batch comes entirely from one bucket, so each batch has one static shape and
    jit compiles exactly len(loaders) variants of the train step. The bucket sequence is drawn
    from a seeded RNG (uniform), making it reproducible.
    """

    def __init__(self, loaders: Sequence[TorchDataLoader], *, seed: int = 0):
        self._loaders = list(loaders)
        self._seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self._seed)
        iterators = [iter(loader) for loader in self._loaders]
        while True:
            yield next(iterators[rng.integers(len(iterators))])


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


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
