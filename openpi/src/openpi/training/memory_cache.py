"""In-RAM cache of the layer-L top-camera hidden states for memory co-training.

Holds one float16 array of shape [num_frames, tokens_per_cam, width] (~24 GB for the YAM
dataset), built by a sequential pass over the dataset with the *current* training parameters and
refreshed in-process every `TrainConfig.memory_cache_refresh_interval` steps. The trainer
gathers each batch's detached write-window slices from it (`Observation.memory_cache_indices`)
and injects them as `Observation.memory_hiddens` before the train step, which bounds the
staleness of the detached writes at one refresh interval.

The pass is dominated by video decoding (3 cameras per frame), so items are built by a torch
DataLoader with parallel workers.
"""

import logging
import multiprocessing
import os
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from openpi.models import model as _model
from openpi.training.data_loader import _collate_fn
from openpi.training.data_loader import _worker_init_fn
import openpi.transforms as _transforms

logger = logging.getLogger("openpi")


class _ExtractionDataset(torch.utils.data.Dataset):
    """Single frames through the exact inference-time input pipeline (no window splitting; no
    subtask/actions, so the tokenizer emits the pure ar=0 context)."""

    def __init__(self, dataset, input_transforms, normalize, model_transforms):
        self._dataset = dataset
        self._input_transforms = input_transforms
        self._normalize = normalize
        self._model_transforms = model_transforms

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict:
        frame = self._dataset[int(index)]
        item = {
            "observation/image": np.asarray(frame["image"]),
            "observation/left_wrist_image": np.asarray(frame["left_wrist_image"]),
            "observation/right_wrist_image": np.asarray(frame["right_wrist_image"]),
            "observation/state": np.asarray(frame["state"]),
        }
        for tf in self._input_transforms:
            item = tf(item)
        item = self._normalize(item)
        for tf in self._model_transforms:
            item = tf(item)
        return item


class MemoryHiddenCache:
    def __init__(self, data_config, model_config, *, batch_size: int = 32, num_workers: int = 8):
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

        self._layer = model_config.memory_layer
        self._batch_size = batch_size
        # spawned workers each re-import the full stack; oversubscribing the CPUs stalls startup
        self._num_workers = max(1, min(num_workers, (os.cpu_count() or num_workers) - 2))
        self._items = _ExtractionDataset(
            lerobot_dataset.LeRobotDataset(data_config.repo_id),
            [t for t in data_config.data_transforms.inputs if not isinstance(t, _transforms.SplitMemoryWindow)],
            _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            list(data_config.model_transforms.inputs),
        )
        self._hiddens: np.ndarray | None = None
        self._extract = None
        self._loader = None

    def __len__(self) -> int:
        return len(self._items)

    def refresh(self, model_def, params) -> None:
        """Recomputes every frame's hidden states with the given (current) parameters."""
        if self._extract is None:
            layer = self._layer

            def extract(params, observation):
                model = nnx.merge(model_def, params)
                return model.extract_topcam_hidden(observation)[layer].astype(jnp.float16)

            self._extract = jax.jit(extract)

        if self._loader is None:
            logger.info(
                f"memory hidden cache: starting {self._num_workers} data workers "
                "(spawn startup re-imports the full stack; expect a quiet minute or two) ..."
            )
            self._loader = torch.utils.data.DataLoader(
                self._items,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                multiprocessing_context=multiprocessing.get_context("spawn") if self._num_workers > 0 else None,
                collate_fn=_collate_fn,
                worker_init_fn=_worker_init_fn,
                persistent_workers=self._num_workers > 0,  # keep the pool alive across refreshes
                drop_last=False,
            )

        num_frames = len(self._items)
        start_time = time.time()
        start = 0
        compiled = False
        logger.info(f"memory hidden cache: extracting {num_frames} frames ...")
        for raw_batch in self._loader:
            n = next(iter(jax.tree.leaves(raw_batch))).shape[0]
            pad = self._batch_size - n  # pad the tail batch to keep one jit shape
            batch = jax.tree.map(lambda x, p=pad: np.concatenate([x, np.repeat(x[-1:], p, 0)]) if p else x, raw_batch)
            if not compiled:
                logger.info("memory hidden cache: first batch arrived, compiling the extraction forward ...")
            hiddens = np.asarray(self._extract(params, _model.Observation.from_dict(batch)))
            if not compiled:
                logger.info("memory hidden cache: compiled, extracting ...")
                compiled = True
            if self._hiddens is None:
                self._hiddens = np.empty((num_frames, *hiddens.shape[1:]), dtype=np.float16)
            self._hiddens[start : start + n] = hiddens[:n]
            start += n
            if start % (self._batch_size * 50) < self._batch_size:
                rate = start / max(time.time() - start_time, 1e-6)
                eta = (num_frames - start) / max(rate, 1e-6)
                logger.info(f"memory hidden cache: {start}/{num_frames} ({rate:.0f} frames/s, eta {eta:.0f}s)")
        logger.info(
            f"memory hidden cache refreshed: {num_frames} frames, {self._hiddens.nbytes / 1e9:.1f} GB, "
            f"{time.time() - start_time:.0f}s"
        )

    def gather(self, indices: np.ndarray) -> np.ndarray:
        """Detached write-window hiddens for a batch of [b, wc] global frame indices."""
        assert self._hiddens is not None, "call refresh() before gather()"
        return self._hiddens[indices].astype(np.float32)
