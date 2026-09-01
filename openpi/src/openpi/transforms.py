from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(self, *, inputs: Sequence[DataTransformFn] = (), outputs: Sequence[DataTransformFn] = ()) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        return jax.tree.map(lambda k: flat_item[k], self.structure)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class InjectPromptFromEpisode(DataTransformFn):
    """Injects each raw item's per-episode high-level prompt (multi-task datasets).

    Runs on raw LeRobot items (needs "episode_index"). `episode_prompts` is indexed by
    episode. A dataset opting into per-episode prompts must cover every episode -- a missing
    or empty entry is a data bug, not a fallback case.
    """

    episode_prompts: tuple[str, ...]

    def __call__(self, data: DataDict) -> DataDict:
        episode = int(np.asarray(data["episode_index"]).item())
        prompt = self.episode_prompts[episode] if episode < len(self.episode_prompts) else ""
        if not prompt:
            raise ValueError(f"episode {episode} has no entry in meta/episode_prompts.json.")
        return {**data, "prompt": np.asarray(prompt)}


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        # Make sure that all the keys in the norm stats are present in the data.
        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks}


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class TokenizeFASTSubtaskInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTSubtaskTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        # The subtask is only present during training. Pop it so no string reaches the batch.
        if (subtask := data.pop("subtask", None)) is not None and not isinstance(subtask, str):
            subtask = subtask.item()

        # Actions stay in the dict: they are still the flow matching target of the action expert.
        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask, fast_mask = self.tokenizer.tokenize(prompt, state, subtask, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
            "token_fast_mask": fast_mask,
        }


def _as_uint8_hwc(image: np.ndarray) -> np.ndarray:
    """LeRobot images may arrive as float32 CHW in [0, 1]; convert to uint8 HWC."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return image


@dataclasses.dataclass(frozen=True)
class MemoryEpisodeInfo(DataTransformFn):
    """Attaches per-episode metadata to each raw LeRobot item (before repack, while
    "episode_index" is still present): "episode_length" always, plus the quiz supervision
    ("quiz_side" / "reveal_frame" / "close_frame") when the side labels are provided, plus the
    memory-critical window ("memory_window" = [start_lo, start_hi, memory_lo, memory_hi], all
    -1 when the episode has no usable phases) when phase tables are provided. Consumed by
    BuildMemorySequence. Built by `data_loader._episode_info_table`."""

    episode_length: np.ndarray
    episode_side: np.ndarray | None = None
    episode_reveal: np.ndarray | None = None
    episode_close: np.ndarray | None = None
    # [num_episodes, 4] int32: memory-critical start window [lo, hi] and the memory-required
    # phase [memory_lo, memory_hi] (frames, inclusive); a row of -1 disables the branch.
    episode_memory_window: np.ndarray | None = None
    # v3.4: the per-episode answer side (0=left, 1=right, -1=unlabeled), attached independently
    # of the legacy quiz plumbing. Consumed by MemoryV34Labels for the probe-ladder labels.
    episode_side_label: np.ndarray | None = None

    def __call__(self, data: DataDict) -> DataDict:
        episode = int(np.asarray(data["episode_index"]).item())
        out = {**data, "episode_length": np.int32(self.episode_length[episode])}
        if self.episode_side is not None:
            out["quiz_side"] = np.int32(self.episode_side[episode])
            out["reveal_frame"] = np.int32(self.episode_reveal[episode])
            out["close_frame"] = np.int32(self.episode_close[episode])
        if self.episode_side_label is not None:
            out["episode_side"] = np.int32(self.episode_side_label[episode])
        if self.episode_memory_window is not None:
            out["memory_window"] = self.episode_memory_window[episode].astype(np.int32)
        return out


@dataclasses.dataclass(frozen=True)
class MemorySequenceSubtasks(DataTransformFn):
    """Per-step (lookahead-shifted) subtask labels for memory sequence training.

    lerobot's __getitem__ can only deliver a SCALAR task_index per item (it calls .item() on
    it), so the per-step labels are looked up from the episodes' task tables instead. Runs on
    raw items (needs episode_index/frame_index); replaces SubtaskFromLeRobotTask for sequence
    configs."""

    stride: int
    steps: int
    lookahead: int
    # per-episode arrays of per-frame task indices (built by data_loader._episode_info_table)
    episode_tasks: tuple
    # dataset_meta.tasks
    tasks: dict[int, str]
    # v3.4.1 leak fix 1: per-episode per-frame booleans, False on waiting-labeled frames whose
    # arm is not actually stationary (data_loader._trim_waiting_to_static). Empty = no trim.
    episode_waiting_valid: tuple = ()

    def __call__(self, data: DataDict) -> DataDict:
        episode = int(np.asarray(data["episode_index"]).item())
        frame = int(np.asarray(data["frame_index"]).item())
        ep_tasks = self.episode_tasks[episode]
        idx = np.minimum(frame + np.arange(self.steps) * self.stride + self.lookahead, len(ep_tasks) - 1)
        # v3.4 also carries the UNSHIFTED per-step labels ("what phase is the observation in"),
        # consumed by MemoryV34Labels for the probe-ladder evidence/waiting frame masks.
        idx_now = np.minimum(frame + np.arange(self.steps) * self.stride, len(ep_tasks) - 1)
        out = {
            **data,
            "subtask": [self.tasks[int(ep_tasks[i])] for i in idx],
            "subtask_now": [self.tasks[int(ep_tasks[i])] for i in idx_now],
        }
        if self.episode_waiting_valid:
            # Same shift convention as the labels they gate: `subtask_valid` follows the
            # lookahead-shifted CE/aux target, `subtask_now_valid` the observation's own phase.
            valid = self.episode_waiting_valid[episode]
            out["subtask_valid"] = valid[idx]
            out["subtask_now_valid"] = valid[idx_now]
        return out


# The first four columns are the frozen v3.3/v3.4 contract.  v3.5 appends metadata to the same
# array so it survives LeRobot's repack step without adding dataset columns or changing dynamic
# timestamp offsets.
V35_MEMORY_WINDOW_SIZE = 13
V35_WINDOW_FINAL_E_LIMIT = 4
V35_WINDOW_OCCLUSION_LO = 5
V35_WINDOW_OCCLUSION_HI = 6
V35_WINDOW_EXECUTE_START = 7
V35_WINDOW_EPISODE_INDEX = 8
V35_WINDOW_COLLECTION_ID = 9
V35_WINDOW_OBJECT_ID = 10
V35_WINDOW_CELL_ID = 11
V35_WINDOW_MARKER = 12
V35_WINDOW_MARKER_VALUE = 35


@dataclasses.dataclass(frozen=True)
class MemoryCriticalLayout:
    """Exact dense/sparse representation selected for one critical start frame."""

    endpoint: int
    keep_indices: np.ndarray
    decay_gap_before: np.ndarray
    sparse_skip_o: bool
    sampled_e_count: int
    n_delay: int


def is_v35_memory_window(window: np.ndarray | None) -> bool:
    return bool(
        window is not None
        and len(window) == V35_MEMORY_WINDOW_SIZE
        and int(window[V35_WINDOW_MARKER]) == V35_WINDOW_MARKER_VALUE
    )


def memory_critical_is_sparse(frame_index: int, window: np.ndarray) -> bool:
    """Stable 50/50 family assignment for v3.5 eligible starts.

    The sampler subsequently balances family x manifest cell x episode.  Assignment is a pure
    function of the raw start and episode window, so the transform can reproduce it in worker
    processes without hidden RNG state.
    """
    if not is_v35_memory_window(window):
        return False
    return (int(frame_index) - int(window[0])) % 2 == 0


def memory_critical_layout(
    frame_index: int, window: np.ndarray, *, stride: int, lookahead: int, num_steps: int
) -> MemoryCriticalLayout:
    """Return the exact v3.5 natural or sparse E->D layout for a critical start.

    Every accepted layout contains the per-grid final eligible E anchor and at least one strict
    D step.  Sparse layouts keep the last one or two E samples plus D samples, and encode the
    omitted valid non-E transitions on the first D step.  Any missing anchor/D candidate raises
    instead of silently yielding a state-invalid training example.
    """
    if not is_v35_memory_window(window):
        endpoint = memory_critical_endpoint(
            frame_index, window, stride=stride, lookahead=lookahead, num_steps=num_steps
        )
        keep = np.arange(endpoint + 1, dtype=np.int32)
        return MemoryCriticalLayout(
            endpoint=endpoint,
            keep_indices=keep,
            decay_gap_before=np.zeros(len(keep), np.int32),
            sparse_skip_o=False,
            sampled_e_count=0,
            n_delay=0,
        )

    evidence_start = int(window[1])
    memory_lo, memory_hi = int(window[2]), int(window[3])
    final_e_limit = int(window[V35_WINDOW_FINAL_E_LIMIT])
    step_frames = int(frame_index) + np.arange(num_steps, dtype=np.int64) * int(stride)

    eligible_e = np.nonzero((step_frames >= evidence_start) & (step_frames <= final_e_limit))[0]
    if len(eligible_e) == 0:
        raise ValueError(
            f"v3.5 critical start {frame_index} has no sampled E frame in "
            f"[{evidence_start}, {final_e_limit}] at stride {stride}."
        )

    in_d = (step_frames >= memory_lo) & (step_frames <= memory_hi)
    decision = np.nonzero(in_d & (step_frames <= memory_hi - lookahead))[0]
    if len(decision) == 0:
        decision = np.nonzero(in_d)[0]
    if len(decision) == 0:
        raise ValueError(
            f"v3.5 critical start {frame_index} has no strict-D grid frame in "
            f"[{memory_lo}, {memory_hi}] at stride {stride}."
        )

    endpoint = int(decision[int(frame_index) % len(decision)])
    final_e = int(eligible_e[-1])
    if endpoint <= final_e:
        raise ValueError(f"v3.5 critical start {frame_index} orders D step {endpoint} before final E step {final_e}.")

    sparse = memory_critical_is_sparse(frame_index, window)
    if not sparse:
        keep = np.arange(endpoint + 1, dtype=np.int32)
        first_d = int(np.nonzero(in_d)[0][0])
        return MemoryCriticalLayout(
            endpoint=endpoint,
            keep_indices=keep,
            decay_gap_before=np.zeros(len(keep), dtype=np.int32),
            sparse_skip_o=False,
            sampled_e_count=len(eligible_e),
            n_delay=first_d - final_e - 1,
        )

    # Keep one or two latest eligible E samples and all strict D samples through the selected
    # endpoint.  O and any pre-D neutral steps are represented by one exact gap before the first
    # D read; subsequent kept D steps are dense and therefore have gap zero.
    keep_e = eligible_e[-2:].astype(np.int32)
    keep_d = np.nonzero(in_d & (np.arange(num_steps) <= endpoint))[0].astype(np.int32)
    if len(keep_d) == 0:
        raise ValueError(f"v3.5 sparse critical start {frame_index} has no D step through endpoint {endpoint}.")
    first_d = int(keep_d[0])
    omitted = np.arange(final_e + 1, first_d, dtype=np.int32)
    occlusion_lo = int(window[V35_WINDOW_OCCLUSION_LO])
    occlusion_hi = int(window[V35_WINDOW_OCCLUSION_HI])
    omitted_frames = step_frames[omitted]
    if np.any((omitted_frames < occlusion_lo) | (omitted_frames > occlusion_hi)):
        # This check deliberately rejects a residue whose next sampled transition still falls
        # in the five-frame semantic-E tail.  Although that tail cannot write, Revision 4
        # permits analytic skipping only across sampled transitions belonging entirely to O.
        # It likewise rejects an unlabeled/reset gap between O and strict D.  Because the
        # episode table derives a contiguous [O_lo, O_hi] directly from current-frame labels,
        # this is available to the sampler before a worker fetches the raw item.
        raise ValueError(
            f"v3.5 sparse critical start {frame_index} skipped sampled frames "
            f"{omitted_frames.tolist()} outside the O-only interval [{occlusion_lo}, {occlusion_hi}]."
        )
    keep = np.concatenate([keep_e, keep_d]).astype(np.int32)
    if np.any(np.diff(keep) <= 0):
        raise ValueError(f"v3.5 sparse critical start {frame_index} produced non-increasing sparse indices.")
    gaps = np.zeros(len(keep), dtype=np.int32)
    gaps[len(keep_e)] = first_d - final_e - 1
    return MemoryCriticalLayout(
        endpoint=endpoint,
        keep_indices=keep,
        decay_gap_before=gaps,
        sparse_skip_o=True,
        # Report commits actually present in the compacted sequence, not all E frames that
        # existed on the discarded dense grid.  The sparse protocol deliberately retains at
        # most the final two anchors.
        sampled_e_count=len(keep_e),
        n_delay=first_d - final_e - 1,
    )


def memory_critical_endpoint(
    frame_index: int, window: np.ndarray, *, stride: int, lookahead: int, num_steps: int
) -> int:
    """The truncation step t_q for a memory-critical sequence starting at `frame_index`.

    DETERMINISTIC per start frame -- the bucket sampler must know each start's exact valid
    length ahead of time, so per-draw randomness would break batch homogeneity (mixed-bucket
    collate error). Endpoint diversity instead comes from stratification: consecutive starts
    in the window cycle through the eligible waiting-phase grid steps, and the start itself is
    drawn uniformly. Eligibility tiers (each may be empty for short/straddled waits):
      1. grid steps whose observation lies in [memory_lo, memory_hi - lookahead], so the
         lookahead-shifted CE target is still a memory-required label;
      2. any grid step inside the waiting phase (waits shorter than the lookahead);
      3. the last grid step before the waiting phase -- observation still neutral, target
         already memory-required.
    """
    if is_v35_memory_window(window):
        # Kept separate from the legacy fallback tiers: a v3.5 D-bearing window without both a
        # final eligible E anchor and a strict D grid point is a data gate failure.
        evidence_start = int(window[1])
        memory_lo, memory_hi = int(window[2]), int(window[3])
        final_e_limit = int(window[V35_WINDOW_FINAL_E_LIMIT])
        step_frames = frame_index + np.arange(num_steps) * stride
        eligible_e = np.nonzero((step_frames >= evidence_start) & (step_frames <= final_e_limit))[0]
        eligible_d = np.nonzero(
            (step_frames >= memory_lo) & (step_frames <= memory_hi) & (step_frames <= memory_hi - lookahead)
        )[0]
        if len(eligible_d) == 0:
            eligible_d = np.nonzero((step_frames >= memory_lo) & (step_frames <= memory_hi))[0]
        if len(eligible_e) == 0 or len(eligible_d) == 0:
            raise ValueError(
                f"v3.5 critical start {frame_index} lacks final-E or strict-D grid coverage "
                f"(E={len(eligible_e)}, D={len(eligible_d)})."
            )
        endpoint = int(eligible_d[frame_index % len(eligible_d)])
        if endpoint <= int(eligible_e[-1]):
            raise ValueError(f"v3.5 critical start {frame_index} has D before its final eligible E anchor.")
        return endpoint

    memory_lo, memory_hi = int(window[2]), int(window[3])
    step_frames = frame_index + np.arange(num_steps) * stride
    in_wait = (step_frames >= memory_lo) & (step_frames <= memory_hi)
    eligible = np.nonzero(in_wait & (step_frames <= memory_hi - lookahead))[0]
    if len(eligible) == 0:
        eligible = np.nonzero(in_wait)[0]
    if len(eligible) == 0:
        eligible = np.nonzero(step_frames < memory_lo)[0][-1:]
    return int(eligible[frame_index % len(eligible)])


@dataclasses.dataclass(frozen=True)
class BuildMemorySequence(DataTransformFn):
    """Builds a sequence training sample from lerobot's stacked step frames (RoboTTT-style).

    The loader delivers, anchored at the sampled base frame: per-camera images and the state at
    the T step frames (base, base+stride, ...), the flat action stream for all T chunks, and
    the per-episode metadata from MemoryEpisodeInfo. This transform:
      * converts the step images to [T, h, w, 3] uint8 (resized later by ResizeImages),
      * reshapes actions to [T, action_horizon, dim],
      * emits "seq_step_mask" (False for steps past the episode end -- lerobot pads by
        repeating the last frame; those steps are loss-masked and their writes are no-ops),
      * MEMORY-CRITICAL samples (start inside the episode's "memory_window", attached by
        MemoryEpisodeInfo): additionally truncates the mask at `memory_critical_endpoint`'s
        deterministic waiting-phase step, so the endpoint's subtask CE can only be solved
        from memory. Determinism per start frame is required for exact bucket assignment;
        endpoint diversity comes from the uniformly drawn start (see the helper),
      * emits "seq_block_boundary": the gradient-block fence, True every `block_steps` steps
        with a fresh random shift per sample (never at step 0). Memory-critical samples get NO
        fence: their entire point is end-to-end credit from the waiting-endpoint CE back to
        the evidence-phase writes, and at <= ~27 valid steps their differentiated chain is no
        longer than a normal sample's 25-step block anyway,
      * emits the per-step quiz supervision when the quiz metadata is present: quizzable =
        a real step at/after the reveal frame AND the reveal happened inside this sequence
        (a slice starting after the reveal never wrote it, so quizzing would teach guessing).

    Inference items (no "frame_index") pass through untouched, so the same transform list
    serves training and serving.
    """

    stride: int
    action_horizon: int
    block_steps: int
    subtask_lookahead: int = 0
    # Required for v3.5's second, item-level proof that every analytically omitted transition
    # is semantic O.  Empty preserves the legacy constructor/output path.
    occlusion_subtasks: tuple[str, ...] = ()

    @staticmethod
    def _compact_sparse_fields(data: DataDict, keep: np.ndarray, num_steps: int) -> None:
        """Compact known T-leading raw fields and pad back to the fixed fetched length.

        LeRobot still fetches the ordinary dense offset grid.  Sparse skip-O is represented
        after fetch, so no per-item dynamic timestamp table or cross-sample state is required.
        Padding repeats the last kept payload only as storage; `seq_step_mask=False` makes those
        steps strict no-ops and every v3.5 supervision mask clears them.
        """
        if len(keep) == 0:
            raise ValueError("cannot compact a sparse memory sequence with no kept steps.")
        padded = np.concatenate([keep, np.full(num_steps - len(keep), keep[-1], dtype=np.int32)])
        for key in (
            "observation/image",
            "observation/left_wrist_image",
            "observation/right_wrist_image",
            "observation/state",
            "actions",
            "subtask_valid",
            "subtask_now_valid",
        ):
            if key in data:
                value = np.asarray(data[key])
                if value.ndim == 0 or value.shape[0] != num_steps:
                    raise ValueError(f"v3.5 sparse field {key!r} does not have leading T={num_steps}: {value.shape}.")
                data[key] = value[padded]
        for key in ("subtask", "subtask_now"):
            if key in data:
                value = data[key]
                if not isinstance(value, list) or len(value) != num_steps:
                    raise ValueError(f"v3.5 sparse field {key!r} must be a list of length {num_steps}.")
                data[key] = [value[int(i)] for i in padded]

    def __call__(self, data: DataDict) -> DataDict:
        if "frame_index" not in data:
            return data
        frame_index = int(np.asarray(data.pop("frame_index")).item())
        data.pop("index", None)
        episode_length = int(np.asarray(data.pop("episode_length")).item())
        window = np.asarray(data.pop("memory_window")) if "memory_window" in data else None

        for key in ("observation/image", "observation/left_wrist_image", "observation/right_wrist_image"):
            data[key] = np.stack([_as_uint8_hwc(frame) for frame in np.asarray(data[key])])
        state = np.asarray(data["observation/state"], dtype=np.float32)
        data["observation/state"] = state
        num_steps = state.shape[0]
        data["actions"] = np.asarray(data["actions"], dtype=np.float32).reshape(num_steps, self.action_horizon, -1)

        step_frames = frame_index + np.arange(num_steps) * self.stride
        data["seq_step_mask"] = step_frames < episode_length

        v35 = is_v35_memory_window(window)
        if v35:
            # Stable manifest metadata is carried in the extended window for every sequence
            # family, including ordinary full/slice samples.
            data["seq_episode_index"] = np.int32(window[V35_WINDOW_EPISODE_INDEX])
            data["seq_collection_id"] = np.int32(window[V35_WINDOW_COLLECTION_ID])
            data["seq_object_id"] = np.int32(window[V35_WINDOW_OBJECT_ID])
            data["seq_memory_cell"] = np.int32(window[V35_WINDOW_CELL_ID])

        memory_critical = window is not None and window[0] >= 0 and window[0] <= frame_index <= window[1]
        if memory_critical:
            if v35:
                layout = memory_critical_layout(
                    frame_index, window, stride=self.stride, lookahead=self.subtask_lookahead, num_steps=num_steps
                )
                if layout.sparse_skip_o:
                    gap_slot = int(np.nonzero(layout.decay_gap_before > 0)[0][0]) if layout.n_delay > 0 else -1
                    if gap_slot >= 0:
                        previous = int(layout.keep_indices[gap_slot - 1])
                        current = int(layout.keep_indices[gap_slot])
                        omitted = np.arange(previous + 1, current, dtype=np.int32)
                        if not np.asarray(data["seq_step_mask"], dtype=bool)[omitted].all():
                            raise ValueError("v3.5 sparse skip interval contains padding/invalid steps.")
                        if not self.occlusion_subtasks:
                            raise ValueError("v3.5 sparse skipping requires explicit occlusion_subtasks validation.")
                        if "subtask_now" not in data:
                            raise ValueError("v3.5 sparse skipping requires unshifted current-frame subtask labels.")
                        allowed_o = set(self.occlusion_subtasks)
                        invalid_labels = [
                            (int(i), str(data["subtask_now"][int(i)]))
                            for i in omitted
                            if str(data["subtask_now"][int(i)]) not in allowed_o
                        ]
                        if invalid_labels:
                            # A distinct reset label is non-O and is rejected here.  Critical
                            # windows never receive stochastic TBPTT/reset fences, so together
                            # these checks prove the omitted span is valid, O-only, non-writing,
                            # and reset-free before compaction.
                            raise ValueError(
                                "v3.5 sparse skip interval must be semantic O-only and reset-free; "
                                f"invalid sampled steps={invalid_labels}."
                            )
                    self._compact_sparse_fields(data, layout.keep_indices, num_steps)
                    step_frames = np.concatenate(
                        [
                            step_frames[layout.keep_indices],
                            np.full(num_steps - len(layout.keep_indices), step_frames[layout.keep_indices[-1]]),
                        ]
                    )
                    data["seq_step_mask"] = np.arange(num_steps) < len(layout.keep_indices)
                    decay_gap = np.zeros(num_steps, dtype=np.int32)
                    decay_gap[: len(layout.decay_gap_before)] = layout.decay_gap_before
                    data["seq_decay_gap_before"] = decay_gap
                else:
                    data["seq_step_mask"] = data["seq_step_mask"] & (np.arange(num_steps) <= layout.endpoint)
                    data["seq_decay_gap_before"] = np.zeros(num_steps, dtype=np.int32)
                data["seq_sparse_skip_o"] = np.bool_(layout.sparse_skip_o)
            else:
                t_q = memory_critical_endpoint(
                    frame_index, window, stride=self.stride, lookahead=self.subtask_lookahead, num_steps=num_steps
                )
                data["seq_step_mask"] = data["seq_step_mask"] & (np.arange(num_steps) <= t_q)
        elif v35:
            data["seq_decay_gap_before"] = np.zeros(num_steps, dtype=np.int32)
            data["seq_sparse_skip_o"] = np.zeros((), dtype=bool)

        if v35:
            valid = np.asarray(data["seq_step_mask"], dtype=bool)
            final_e_limit = int(window[V35_WINDOW_FINAL_E_LIMIT])
            execute_start = int(window[V35_WINDOW_EXECUTE_START])
            occlusion_lo = int(window[V35_WINDOW_OCCLUSION_LO])
            occlusion_hi = int(window[V35_WINDOW_OCCLUSION_HI])
            # Private selectors are consumed by MemoryV34Labels after it forms masks from the
            # unshifted current-frame labels.  They are never exposed as alternative labels.
            data["_v35_write_tail_valid"] = valid & (step_frames <= final_e_limit)
            data["_v35_action_overlaps_execute"] = (
                valid & (step_frames <= execute_start) & (step_frames + self.action_horizon - 1 >= execute_start)
            )
            data["seq_occlusion_mask"] = valid & (step_frames >= occlusion_lo) & (step_frames <= occlusion_hi)
            data["_v35_enabled"] = np.ones((), dtype=bool)

        boundary = np.zeros(num_steps, dtype=bool)
        if self.block_steps > 0 and not memory_critical:
            shift = np.random.randint(self.block_steps)
            boundary = (np.arange(num_steps) > 0) & ((np.arange(num_steps) - shift) % self.block_steps == 0)
        data["seq_block_boundary"] = boundary

        if "quiz_side" in data:
            side = int(np.asarray(data.pop("quiz_side")).item())
            reveal = int(np.asarray(data.pop("reveal_frame")).item())
            close = int(np.asarray(data.pop("close_frame")).item())
            quizzable = data["seq_step_mask"] & (step_frames >= reveal) & (reveal >= frame_index) & (side >= 0)
            data["seq_probe_labels"] = np.full(num_steps, side, dtype=np.int32)
            data["seq_probe_mask"] = quizzable
            data["seq_probe_visible"] = quizzable & (step_frames < close)
        return data


@dataclasses.dataclass(frozen=True)
class MemoryV34Labels(DataTransformFn):
    """Per-segment v3.4 supervision fields (V34_PLAN_final.md 5.1, 5.2, Section 6).

    Runs after BuildMemorySequence on training sequence samples (identified by the per-step
    "subtask" list); inference items pass through untouched. Emits:

      * "seq_subtask_class" [T] int32: the LOOKAHEAD-SHIFTED subtask label's index in
        ``subtask_vocab`` (-1 = unknown) -- the aux-demand CE target (plan 5.1), the same label
        space the causal CE trains on;
      * "seq_evidence_mask"/"seq_waiting_mask" [T] bool from the UNSHIFTED labels
        ("subtask_now"): is the step's OBSERVATION inside the evidence / waiting phase --
        the probe-ladder frame selectors (Section 6 rungs 1 and 4);
      * "seq_side_label" int32: the episode's answer side (0/1; -1 unlabeled), preferring the
        loader-attached "episode_side" and falling back to the side named by any shifted
        waiting label in the window;
      * "seq_state_masked" bool (plan 5.2, when ``state_mask_prob`` > 0): drawn ONCE PER
        SEGMENT, and only for memory-required segments -- windows containing at least one
        valid step whose shifted CE label is a waiting label. Per-frame draws would let the
        model funnel state through the memory (write arm-lean at an unmasked frame, read it at
        a masked one); segment-level masking removes the within-segment funnel.
    """

    subtask_vocab: tuple[str, ...]
    evidence_subtasks: tuple[str, ...]
    memory_required_subtasks: tuple[str, ...]
    state_mask_prob: float = 0.0

    def __call__(self, data: DataDict) -> DataDict:
        subtask = data.get("subtask")
        if not isinstance(subtask, list):
            data.pop("subtask_now", None)
            data.pop("episode_side", None)
            data.pop("subtask_valid", None)
            data.pop("subtask_now_valid", None)
            data.pop("_v35_enabled", None)
            data.pop("_v35_write_tail_valid", None)
            data.pop("_v35_action_overlaps_execute", None)
            return data
        subtask = [str(s) for s in subtask]
        subtask_now = [str(s) for s in data.pop("subtask_now", subtask)]
        if len(subtask_now) != len(subtask):
            raise ValueError(f"subtask/subtask_now length mismatch: {len(subtask)} vs {len(subtask_now)}.")

        vocab = {label: index for index, label in enumerate(self.subtask_vocab)}
        subtask_class = np.asarray([vocab.get(s, -1) for s in subtask], dtype=np.int32)
        evidence = set(self.evidence_subtasks)
        waiting = list(self.memory_required_subtasks)
        evidence_mask = np.asarray([s in evidence for s in subtask_now], dtype=bool)
        waiting_mask = np.asarray([s in waiting for s in subtask_now], dtype=bool)

        # v3.4.1 leak fix 1: withdraw memory supervision from waiting frames whose arm is
        # moving (data_loader._trim_waiting_to_static). The aux target becomes unknown (-1, so
        # the step is masked out of the class-balanced CE) and the frame leaves the ladder's
        # waiting pool. Evidence frames are never gated -- motion there is the demonstration.
        shifted_valid = data.pop("subtask_valid", None)
        now_valid = data.pop("subtask_now_valid", None)
        if shifted_valid is not None:
            shifted_valid = np.asarray(shifted_valid, dtype=bool)
            subtask_class = np.where(shifted_valid, subtask_class, np.int32(-1))
        if now_valid is not None:
            waiting_mask = waiting_mask & np.asarray(now_valid, dtype=bool)

        data["seq_subtask_class"] = subtask_class
        data["seq_evidence_mask"] = evidence_mask
        data["seq_waiting_mask"] = waiting_mask

        side = int(np.asarray(data.pop("episode_side", -1)).item())
        if side < 0:
            for s in subtask:
                if s in waiting:
                    side = waiting.index(s)
                    break
        data["seq_side_label"] = np.int32(side)

        # v3.5 Revision 4 fields are opt-in through the extended memory-window marker.  Legacy
        # configs therefore retain their exact output tree and supervision semantics.
        v35 = bool(np.asarray(data.pop("_v35_enabled", False)).item())
        if v35:
            step_mask = np.asarray(data.get("seq_step_mask", np.ones(len(subtask), dtype=bool)), dtype=bool)
            if step_mask.shape != (len(subtask),):
                raise ValueError(f"v3.5 seq_step_mask must have shape {(len(subtask),)}, got {step_mask.shape}.")
            tail_valid = np.asarray(data.pop("_v35_write_tail_valid"), dtype=bool)
            action_overlap = np.asarray(data.pop("_v35_action_overlaps_execute"), dtype=bool)
            if tail_valid.shape != step_mask.shape or action_overlap.shape != step_mask.shape:
                raise ValueError("v3.5 raw-frame selectors must have the same shape as seq_step_mask.")

            write_mask = evidence_mask & tail_valid & step_mask
            decision_mask = waiting_mask & step_mask
            block_boundary = np.asarray(data.get("seq_block_boundary", np.zeros(len(subtask), dtype=bool)), dtype=bool)
            decay_gap = np.asarray(
                data.get("seq_decay_gap_before", np.zeros(len(subtask), dtype=np.int32)), dtype=np.int32
            )
            if block_boundary.shape != step_mask.shape or decay_gap.shape != step_mask.shape:
                raise ValueError("v3.5 boundary/gap fields must have the same shape as seq_step_mask.")
            if np.any(decay_gap < 0):
                raise ValueError("seq_decay_gap_before cannot be negative.")
            if np.any((decay_gap > 0) & (~step_mask | write_mask)):
                raise ValueError("analytic gaps may precede only valid non-E/read steps.")

            # Read happens before the current transition.  A fence cuts gradient reach at this
            # step's entry but does not erase memory content; the current step's E write becomes
            # available only to later reads.  Analytic skip-O decay is differentiable and does
            # not itself break either state validity or reachability.
            state_valid = np.zeros(len(subtask), dtype=bool)
            credit_reachable = np.zeros(len(subtask), dtype=bool)
            have_state = False
            have_reachable_write = False
            for t in range(len(subtask)):
                if not step_mask[t]:
                    continue
                if block_boundary[t]:
                    have_reachable_write = False
                state_valid[t] = have_state
                credit_reachable[t] = have_reachable_write
                if write_mask[t]:
                    have_state = True
                    have_reachable_write = True

            state_invalid_d = decision_mask & ~state_valid
            if np.any(state_invalid_d):
                raise ValueError(
                    "v3.5 D-bearing sequence lacks a prior final-eligible E anchor at steps "
                    f"{np.nonzero(state_invalid_d)[0].tolist()}; sampler must not silently grade it."
                )

            data["seq_write_mask"] = write_mask
            data["seq_decision_mask"] = decision_mask
            data["seq_read_state_valid"] = state_valid
            data["seq_read_credit_reachable"] = credit_reachable
            data["seq_use_pressure_mask"] = decision_mask & state_valid & action_overlap

            # Padding is a no-op in both dense and sparse layouts, including analytic decay.
            data["seq_decay_gap_before"] = np.where(step_mask, decay_gap, 0).astype(np.int32)
        elif "_v35_write_tail_valid" in data or "_v35_action_overlaps_execute" in data:
            # Private selectors can only be emitted together with the marker; fail closed if a
            # future repack accidentally separates them.
            raise ValueError("v3.5 selectors were present without a v3.5 memory-window marker.")

        if self.state_mask_prob > 0:
            step_mask = np.asarray(data.get("seq_step_mask", np.ones(len(subtask), dtype=bool)))
            shifted_waiting = np.asarray([s in waiting for s in subtask], dtype=bool)
            if shifted_valid is not None:
                # A segment whose only waiting targets were dropped as non-static is no longer
                # memory-required, so it must not draw the plan-5.2 state mask either.
                shifted_waiting = shifted_waiting & shifted_valid
            memory_required_segment = bool(np.any(shifted_waiting & step_mask[: len(shifted_waiting)]))
            drawn = memory_required_segment and (np.random.random() < self.state_mask_prob)
            data["seq_state_masked"] = np.bool_(drawn)
        return data


@dataclasses.dataclass(frozen=True)
class TokenizeMemorySubtaskInputs(DataTransformFn):
    """Tokenizer for the memory co-training layout [images | context | memory | causal].

    Sequence training (per-step subtask list + actions [T, ah, d] present): every step gets the
    ar=0 context and the causal subtask+FAST segment as separate buffers
    (`FASTSubtaskTokenizer.tokenize_split`), stacked to [T, ...].
    At inference it matches TokenizeFASTSubtaskInputs without labels: context tokens only.
    """

    tokenizer: _tokenizer.FASTSubtaskTokenizer
    causal_len: int

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")
        if not isinstance(prompt, str):
            prompt = prompt.item()
        subtask = data.pop("subtask", None)

        state = data["state"]
        if subtask is None:
            # inference: pure ar=0 context, same as the no-label FAST subtask path. The state
            # mask ships along so the v3.4 instruction-only conditioner sees the identical
            # context selection at inference and training.
            tokens, token_mask, ar_mask, loss_mask, fast_mask, state_mask = self.tokenizer.tokenize(
                prompt, state, None, None, return_state_mask=True
            )
            return {
                **data,
                "tokenized_prompt": tokens,
                "tokenized_prompt_mask": token_mask,
                "token_ar_mask": ar_mask,
                "token_loss_mask": loss_mask,
                "token_fast_mask": fast_mask,
                "token_state_mask": state_mask,
            }

        if isinstance(subtask, str):
            subtask = [subtask] * state.shape[0] if state.ndim == 2 else [subtask]
        actions = data["actions"]
        if state.ndim != 2:
            raise ValueError("memory sequence training expects per-step state [T, s]")
        steps = [
            self.tokenizer.tokenize_split(
                prompt, state[k], str(subtask[k]), actions[k], self.causal_len, return_state_mask=True
            )
            for k in range(state.shape[0])
        ]
        context, context_mask, causal, causal_mask, causal_fast, context_state = (
            np.stack(x) for x in zip(*steps, strict=True)
        )
        return {
            **data,
            "tokenized_prompt": context,
            "tokenized_prompt_mask": context_mask,
            # the context is pure ar=0; these exist only to keep the batch structure uniform
            "token_ar_mask": np.zeros(context.shape, dtype=np.int32),
            "token_loss_mask": np.zeros(context.shape, dtype=bool),
            "token_fast_mask": np.zeros(context.shape, dtype=bool),
            "token_state_mask": context_state,
            "tokenized_causal": causal,
            "tokenized_causal_mask": causal_mask,
            "causal_fast_mask": causal_fast,
        }


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class SubtaskFromLeRobotTask(DataTransformFn):
    """Extracts a per-frame subtask string from the current LeRobot dataset task.

    Unlike `PromptFromLeRobotTask`, the result is stored in the "subtask" field, leaving the
    "prompt" field free to carry the high-level task instruction (e.g. via `InjectDefaultPrompt`).
    """

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract subtask without "task_index"')

        # A scalar / length-1 sequence (lookahead task_index), or [T] per-step indices when the
        # loader delivers a memory training sequence.
        indices = np.atleast_1d(np.asarray(data["task_index"]))
        subtasks = []
        for task_index in indices:
            if (subtask := self.tasks.get(int(task_index))) is None:
                raise ValueError(f"task_index={int(task_index)} not found in task mapping: {self.tasks}")
            subtasks.append(subtask)
        return {**data, "subtask": subtasks if len(subtasks) > 1 else subtasks[0]}


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T], selector: at.PyTree[S], fn: Callable[[T, S], T], *, strict: bool = False
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
