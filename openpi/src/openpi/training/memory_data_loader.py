"""Episode-sequence data loader for the MAC-style memory model (`Pi0Memory`).

Standard openpi loaders return a single observation + an action chunk from a random point in a random
episode. The memory model instead needs, per training item, N *causal* observation frames spanning
inspect->recall plus a single action chunk at the final frame `t`.

This module wraps the existing transformed per-frame dataset (so all of `YamInputs`, `DeltaActions`,
normalization, and tokenization are reused verbatim): each item samples one episode + a recall-weighted
target `t`, takes `N = linspace(episode_start, t)` frame indices, stacks those transformed observations
to `[N, ...]`, and attaches the action chunk `[H, ad]` at `t`. The collated batch is therefore
observation `[B, N, ...]` + actions `[B, H, ad]`, which `Pi0Memory.compute_loss` unrolls over.
"""

import logging

import jax
import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

logger = logging.getLogger("openpi")

# Keys in a transformed sample that are NOT part of the observation (handled separately / dropped).
_NON_OBS_KEYS = ("actions",)


def _episode_ranges(dataset) -> list[tuple[int, int]]:
    """Return [(from, to), ...] global frame ranges per episode (`to` exclusive)."""
    inner = dataset
    while hasattr(inner, "_dataset"):  # unwrap TransformedDataset / PromptFromLeRobotTask wrappers
        inner = inner._dataset
    edi = getattr(inner, "episode_data_index", None)
    if edi is not None:
        froms = np.asarray(edi["from"]).astype(int).tolist()
        tos = np.asarray(edi["to"]).astype(int).tolist()
        return list(zip(froms, tos, strict=True))
    # Fallback: derive contiguous ranges from per-episode lengths.
    meta = inner.meta
    ranges, start = [], 0
    for i in range(meta.total_episodes):
        length = int(meta.episodes[i]["length"])
        ranges.append((start, start + length))
        start += length
    return ranges


class EpisodeSequenceDataset(_data_loader.Dataset):
    def __init__(
        self,
        transformed_dataset: _data_loader.Dataset,
        episode_ranges: list[tuple[int, int]],
        *,
        n_mem: int,
        action_horizon: int,
        t_sampling: str = "recall_weighted",
        samples_per_episode: int = 50,
    ):
        self._ds = transformed_dataset
        self._eps = episode_ranges
        self._n = n_mem
        self._h = action_horizon
        self._t_sampling = t_sampling
        self._samples_per_episode = samples_per_episode
        if not episode_ranges:
            raise ValueError("EpisodeSequenceDataset got no episodes.")

    def __len__(self) -> int:
        # Inflate so a batch can draw many distinct (episode, t) samples and small datasets still
        # exceed the batch size. __getitem__ maps index -> episode via modulo.
        return len(self._eps) * self._samples_per_episode

    def _sample_t(self, rng: np.random.Generator, lo: int, hi: int) -> int:
        """Sample the action-frame `t` in [lo, hi], weighted toward the recall (back) third."""
        if hi <= lo:
            return lo
        if self._t_sampling == "recall_weighted" and rng.random() < 0.75:
            start = lo + 2 * (hi - lo) // 3  # back third
            return int(rng.integers(start, hi + 1))
        return int(rng.integers(lo, hi + 1))  # uniform over the episode (motor-phase coverage)

    def __getitem__(self, index) -> dict:
        rng = np.random.default_rng()  # entropy-seeded: independent across workers and epochs
        frm, to = self._eps[int(index) % len(self._eps)]
        # Need a full action chunk within the episode for the loss frame.
        hi = max(frm, to - self._h)
        t = self._sample_t(rng, frm, hi)

        idx = np.round(np.linspace(frm, t, self._n)).astype(int)
        idx = np.clip(idx, frm, to - 1)

        frames = [self._ds[int(j)] for j in idx]
        obs_frames = [{k: v for k, v in f.items() if k not in _NON_OBS_KEYS} for f in frames]
        # Stack observation frames along a new leading N axis (handles the nested image/mask dicts).
        item = jax.tree.map(lambda *xs: np.stack(xs, axis=0), *obs_frames)
        # Single action chunk at the final (loss) frame.
        item["actions"] = frames[-1]["actions"]
        return item


def create_memory_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    seed: int = 0,
) -> _data_loader.DataLoader:
    """Mirror of `create_torch_data_loader` that yields episode-sequence batches for the memory model."""
    data_config = config.data.create(config.assets_dirs, config.model)
    n_mem = getattr(config.data, "n_mem", 16)
    t_sampling = getattr(config.data, "t_sampling", "recall_weighted")

    raw = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    episode_ranges = _episode_ranges(raw)
    transformed = _data_loader.transform_dataset(raw, data_config, skip_norm_stats=skip_norm_stats)

    seq_dataset = EpisodeSequenceDataset(
        transformed,
        episode_ranges,
        n_mem=n_mem,
        action_horizon=config.model.action_horizon,
        t_sampling=t_sampling,
    )
    logger.info(f"EpisodeSequenceDataset: {len(episode_ranges)} episodes, n_mem={n_mem}, t_sampling={t_sampling}")

    local_batch_size = config.batch_size // jax.process_count()
    torch_loader = _data_loader.TorchDataLoader(
        seq_dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
    )
    return _data_loader.DataLoaderImpl(data_config, torch_loader)
