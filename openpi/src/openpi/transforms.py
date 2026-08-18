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

    def __call__(self, data: DataDict) -> DataDict:
        episode = int(np.asarray(data["episode_index"]).item())
        out = {**data, "episode_length": np.int32(self.episode_length[episode])}
        if self.episode_side is not None:
            out["quiz_side"] = np.int32(self.episode_side[episode])
            out["reveal_frame"] = np.int32(self.episode_reveal[episode])
            out["close_frame"] = np.int32(self.episode_close[episode])
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

    def __call__(self, data: DataDict) -> DataDict:
        episode = int(np.asarray(data["episode_index"]).item())
        frame = int(np.asarray(data["frame_index"]).item())
        ep_tasks = self.episode_tasks[episode]
        idx = np.minimum(frame + np.arange(self.steps) * self.stride + self.lookahead, len(ep_tasks) - 1)
        return {**data, "subtask": [self.tasks[int(ep_tasks[i])] for i in idx]}


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
        MemoryEpisodeInfo): additionally truncates the mask at a per-draw random step whose
        observation still lies in the memory-required (waiting) phase, so the endpoint's
        subtask CE can only be solved from memory; the endpoint avoids the last
        `subtask_lookahead` waiting frames so its (lookahead-shifted) target stays a
        memory-required label,
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
        data["actions"] = np.asarray(data["actions"], dtype=np.float32).reshape(
            num_steps, self.action_horizon, -1
        )

        step_frames = frame_index + np.arange(num_steps) * self.stride
        data["seq_step_mask"] = step_frames < episode_length

        memory_critical = window is not None and window[0] >= 0 and window[0] <= frame_index <= window[1]
        if memory_critical:
            memory_lo, memory_hi = int(window[2]), int(window[3])
            in_wait = (step_frames >= memory_lo) & (step_frames <= memory_hi)
            eligible = in_wait & (step_frames <= memory_hi - self.subtask_lookahead)
            if not eligible.any():
                eligible = in_wait  # waiting phase shorter than the lookahead
            if not eligible.any():
                # The stride grid straddles a very short waiting phase entirely. End at the
                # last step before it: the observation is still neutral (close/reset) and its
                # lookahead-shifted target is already a memory-required label.
                eligible = np.zeros(num_steps, dtype=bool)
                eligible[np.nonzero(step_frames < memory_lo)[0][-1]] = True
            t_q = np.random.choice(np.nonzero(eligible)[0])
            data["seq_step_mask"] = data["seq_step_mask"] & (np.arange(num_steps) <= t_q)

        boundary = np.zeros(num_steps, dtype=bool)
        if self.block_steps > 0 and not memory_critical:
            shift = np.random.randint(self.block_steps)
            boundary = (np.arange(num_steps) > 0) & ((np.arange(num_steps) - shift) % self.block_steps == 0)
        data["seq_block_boundary"] = boundary

        if "quiz_side" in data:
            side = int(np.asarray(data.pop("quiz_side")).item())
            reveal = int(np.asarray(data.pop("reveal_frame")).item())
            close = int(np.asarray(data.pop("close_frame")).item())
            quizzable = (
                data["seq_step_mask"] & (step_frames >= reveal) & (reveal >= frame_index) & (side >= 0)
            )
            data["seq_probe_labels"] = np.full(num_steps, side, dtype=np.int32)
            data["seq_probe_mask"] = quizzable
            data["seq_probe_visible"] = quizzable & (step_frames < close)
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
            # inference: pure ar=0 context, same as the no-label FAST subtask path
            tokens, token_mask, ar_mask, loss_mask, fast_mask = self.tokenizer.tokenize(prompt, state, None, None)
            return {
                **data,
                "tokenized_prompt": tokens,
                "tokenized_prompt_mask": token_mask,
                "token_ar_mask": ar_mask,
                "token_loss_mask": loss_mask,
                "token_fast_mask": fast_mask,
            }

        if isinstance(subtask, str):
            subtask = [subtask] * state.shape[0] if state.ndim == 2 else [subtask]
        actions = data["actions"]
        if state.ndim != 2:
            raise ValueError("memory sequence training expects per-step state [T, s]")
        steps = [
            self.tokenizer.tokenize_split(prompt, state[k], str(subtask[k]), actions[k], self.causal_len)
            for k in range(state.shape[0])
        ]
        context, context_mask, causal, causal_mask, causal_fast = (np.stack(x) for x in zip(*steps, strict=True))
        return {
            **data,
            "tokenized_prompt": context,
            "tokenized_prompt_mask": context_mask,
            # the context is pure ar=0; these exist only to keep the batch structure uniform
            "token_ar_mask": np.zeros(context.shape, dtype=np.int32),
            "token_loss_mask": np.zeros(context.shape, dtype=bool),
            "token_fast_mask": np.zeros(context.shape, dtype=bool),
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
