from collections.abc import Iterator, Sequence
import dataclasses
import functools
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

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
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
    dataset = lerobot_dataset.LeRobotDataset(data_config.repo_id, delta_timestamps=delta_timestamps)

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
        usable = (
            (bounds["evidence_start"] >= 0)
            & (bounds["memory_lo"] >= 0)
            & (bounds["evidence_end"] < bounds["memory_lo"])
            & memory_contiguous
        )
        for key, value in bounds.items():
            info[key] = np.where(usable, value, -1).astype(np.int32)
        if not usable.all():
            logging.warning(
                f"memory-critical phases missing/malformed in {int((~usable).sum())}/{num_episodes} "
                "episodes; they are excluded from the memory-critical branch."
            )
        if data_config.memory_waiting_max_speed is not None:
            _trim_waiting_to_static(
                info,
                np.asarray(cols["state"], dtype=np.float32),
                starts,
                ends,
                data_config,
                stride=max(int(data_config.memory_stride_frames), 1),
            )
    return info


def _longest_static_run(
    window: np.ndarray, max_speed: float, max_excursion: float
) -> tuple[int, int] | None:
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
        if new_hi - new_lo + 1 < stride:
            # Too short to place even one grid step: keep the honest phase bounds and label
            # mask, but take the episode out of the memory-critical branch rather than crowding
            # every endpoint onto a handful of frames.
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
    window[(evidence_start < 0) | ~info.get("memory_critical_ok", info["memory_lo"] >= 0)] = -1
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
    allowed = ~np.isin(episode, heldout)

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
        # Each memory-critical start truncates at memory_critical_endpoint's DETERMINISTIC
        # step, so its exact valid length is known here and bucket assignment is precise --
        # BuildMemorySequence recomputes the identical endpoint at fetch time. (A per-draw
        # random endpoint would mix bucket lengths inside one batch and trip the homogeneity
        # check in _sequence_bucket_collate_fn.)
        for i in np.nonzero(mc_ok)[0]:
            t_q = _transforms.memory_critical_endpoint(
                int(frame[i]),
                window[episode[i]],
                stride=stride,
                lookahead=data_config.subtask_lookahead,
                num_steps=max_steps,
            )
            valid_steps[i] = t_q + 1
    else:
        dead = (frame > info["reveal"][episode]) & (frame < info["switch"][episode])
        in_window = np.zeros(len(episode), dtype=bool)
        mc_ok = in_window

    mc_prob = data_config.memory_critical_prob if int(mc_ok.sum()) > 0 else 0.0
    if data_config.memory_critical_prob > 0 and mc_prob == 0.0:
        logging.warning("memory_critical_prob > 0 but no eligible memory-critical starts; branch disabled.")

    min_frames = data_config.memory_min_slice_steps * stride
    # ~in_window (not just ~mc_ok): a window start drawn as a slice would still be truncated
    # by BuildMemorySequence, so window frames belong to the memory-critical branch or nothing
    slice_ok = (frame > 0) & (frame + min_frames <= length) & ~dead & ~in_window & allowed

    slice_prob = data_config.memory_slice_prob
    weights = np.zeros(len(episode), dtype=np.float64)
    full_ok = (frame == 0) & allowed
    n_full = int(full_ok.sum())
    weights[full_ok] = (1.0 - mc_prob) * (1.0 - slice_prob) / max(n_full, 1)
    if slice_prob > 0 and int(slice_ok.sum()) > 0:
        weights[slice_ok] = (1.0 - mc_prob) * slice_prob / int(slice_ok.sum())

    n_cells = 0
    if mc_prob > 0:
        prompts = _load_episode_prompts(dataset) or ()
        mc_episodes = np.unique(episode[mc_ok])
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
        f"sequence sampling: {n_full} full starts (p={(1 - mc_prob) * (1 - slice_prob):.3g}), "
        f"{int(slice_ok.sum())} slice starts (p={(1 - mc_prob) * slice_prob:.3g}), "
        f"{int(mc_ok.sum())} memory-critical starts (p={mc_prob:g}, {n_cells} balance cells), "
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
            dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
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
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


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
