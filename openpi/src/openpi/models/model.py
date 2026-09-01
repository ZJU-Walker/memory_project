import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# Type variable for array types (JAX arrays, PyTorch tensors, or numpy arrays)
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# The model always expects these images
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# This may need change if we release a small model.
IMAGE_RESOLUTION = (224, 224)


# Data format
#
# Data transforms produce the model input as a nested dictionary which is later converted
# into `Obesrvation` and `Actions` objects. See below.
#
# In the dictory form, this data should look like:
# {
#     # Observation data.
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB image in [-1, 1] or [0, 255]
#         ...  # Additional camera views
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # True if image is valid
#         ...  # Masks for additional views
#     },
#     "state": float32[*b, s],  # Low-dimensional robot state
#     "tokenized_prompt": int32[*b, l],  # Optional, tokenized language prompt
#     "tokenized_prompt_mask": bool[*b, l],  # Optional, mask for tokenized prompt
#     "token_ar_mask": int32[*b, l],  # Optional, autoregressive mask for FAST model
#     "token_loss_mask": bool[*b, l],  # Optional, loss mask for FAST model
#     "token_fast_mask": bool[*b, l],  # Optional, marks FAST action tokens (subtask co-training)
#
#      # Actions data.
#      "actions": float32[*b ah ad]
# }
# where:
#   *b = batch dimensions
#   h,w = image height/width
#   s = state dimension
#   l = sequence length
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    # Images, in [-1, 1] float32.
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # Image masks, with same keys as images.
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # Low-dimensional robot state.
    state: at.Float[ArrayT, "*b s"]

    # Tokenized prompt.
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # Tokenized prompt mask.
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast model specific fields.

    # Token auto-regressive mask (for FAST autoregressive model).
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token loss mask (for FAST autoregressive model).
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    # Marks the FAST action tokens within the prompt (pi05 subtask co-training). These tokens are a
    # training-time-only CE target for the VLM and are hidden from the flow matching action expert.
    token_fast_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi05 memory co-training fields (Pi0Config.predict_with_memory). Layout per step:
    # [images | context (tokenized_prompt) | memory | causal (subtask+FAST) | action suffix].
    #
    # SEQUENCE TRAINING (RoboTTT-style): a training sample is a contiguous run of T prediction
    # steps from one episode (one step per executed action chunk). All standard fields then
    # carry a leading step axis -- images [b, T, h, w, c], state [b, T, s],
    # tokenized_prompt [b, T, l], tokenized_causal [b, T, cl], actions [b, T, ah, ad] -- which
    # their `*b` shape annotations absorb. The memory starts blank at step 0, is written once
    # per step, and the model predicts (and is graded) at every step. `seq_step_mask` present
    # marks a sequence sample. At inference every field is single-frame as usual.

    # Causal subtask+FAST segment (every valid token is a CE target; left-aligned).
    tokenized_causal: at.Int[ArrayT, "*b cl"] | None = None
    tokenized_causal_mask: at.Bool[ArrayT, "*b cl"] | None = None
    # Marks the FAST branch within the causal segment (hidden from the action expert).
    causal_fast_mask: at.Bool[ArrayT, "*b cl"] | None = None
    # NOTE: in a sequence sample `*b` binds to (batch, T) via the standard fields, so the
    # per-step masks below use their own leading axes ([T] unbatched, [batch, T] collated).
    # True for real steps; False for padding past the episode/slice end (losses masked, writes
    # are exact no-ops on the memory).
    seq_step_mask: at.Bool[ArrayT, "*sb st"] | None = None
    # Gradient-block fence: True at steps where backprop through the memory state is cut at
    # this step's entry (the state content itself always flows through). Never True at step 0.
    seq_block_boundary: at.Bool[ArrayT, "*sb st"] | None = None
    # Quiz-probe supervision per step (train-only; see Pi0Config.memory_probe_weight): answer
    # class (-1 = unknown), quizzable (the reveal has been written into THIS sequence's memory
    # by this step), and answer-still-on-screen (accuracy-split logging only).
    seq_probe_labels: at.Int[ArrayT, "*sb st"] | None = None
    seq_probe_mask: at.Bool[ArrayT, "*sb st"] | None = None
    seq_probe_visible: at.Bool[ArrayT, "*sb st"] | None = None

    # v3.4 fields (V34_PLAN_final.md).
    # Marks the state-digit token positions within tokenized_prompt (same [*b, l] geometry).
    # Used by the input-level state masking (plan 5.2) and the instruction-only conditioner
    # context (plan 5.9). Emitted by the tokenizer for every memory-layout item.
    token_state_mask: at.Bool[ArrayT, "*b l"] | None = None
    # Plan 5.2: True for segments whose state tokens are replaced by the learned null embedding
    # at the input -- sampled ONCE PER SEGMENT in the data pipeline (train-only).
    seq_state_masked: at.Bool[ArrayT, "*sb"] | None = None
    # Plan 5.1: per-step subtask class index in the aux vocabulary (-1 = unknown/unlabeled).
    seq_subtask_class: at.Int[ArrayT, "*sb st"] | None = None
    # Section 6 probe-ladder supervision: the segment's side label (0/1 from the waiting-phase
    # subtask, -1 when the segment has no side-bearing step) and the per-step OBSERVATION-phase
    # masks (unshifted labels: is the frame inside the evidence / waiting phase).
    seq_side_label: at.Int[ArrayT, "*sb"] | None = None
    seq_evidence_mask: at.Bool[ArrayT, "*sb st"] | None = None
    seq_waiting_mask: at.Bool[ArrayT, "*sb st"] | None = None

    # v3.5 Revision 4 current-frame memory protocol.  These are absent for every legacy data
    # config.  Reads occur before the current transition, so `seq_decay_gap_before[..., t]` is
    # applied before reading step t and state-valid/reachable refer only to earlier E commits.
    seq_write_mask: at.Bool[ArrayT, "*sb st"] | None = None
    seq_decision_mask: at.Bool[ArrayT, "*sb st"] | None = None
    seq_occlusion_mask: at.Bool[ArrayT, "*sb st"] | None = None
    seq_read_state_valid: at.Bool[ArrayT, "*sb st"] | None = None
    seq_read_credit_reachable: at.Bool[ArrayT, "*sb st"] | None = None
    seq_decay_gap_before: at.Int[ArrayT, "*sb st"] | None = None
    seq_use_pressure_mask: at.Bool[ArrayT, "*sb st"] | None = None
    # Sparse family and stable manifest metadata are scalar per sampled sequence (batched as
    # [batch]).  `seq_memory_cell` is the stable lexicographic ID of
    # (collection, object, target-side) used for episode-first macro losses.
    seq_sparse_skip_o: at.Bool[ArrayT, "*sb"] | None = None
    seq_episode_index: at.Int[ArrayT, "*sb"] | None = None
    seq_collection_id: at.Int[ArrayT, "*sb"] | None = None
    seq_object_id: at.Int[ArrayT, "*sb"] | None = None
    seq_memory_cell: at.Int[ArrayT, "*sb"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        # Ensure that tokenized_prompt and tokenized_prompt_mask are provided together.
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # If images are uint8, convert them to [-1, 1] float32.
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            token_fast_mask=data.get("token_fast_mask"),
            tokenized_causal=data.get("tokenized_causal"),
            tokenized_causal_mask=data.get("tokenized_causal_mask"),
            causal_fast_mask=data.get("causal_fast_mask"),
            seq_step_mask=data.get("seq_step_mask"),
            seq_block_boundary=data.get("seq_block_boundary"),
            seq_probe_labels=data.get("seq_probe_labels"),
            seq_probe_mask=data.get("seq_probe_mask"),
            seq_probe_visible=data.get("seq_probe_visible"),
            token_state_mask=data.get("token_state_mask"),
            seq_state_masked=data.get("seq_state_masked"),
            seq_subtask_class=data.get("seq_subtask_class"),
            seq_side_label=data.get("seq_side_label"),
            seq_evidence_mask=data.get("seq_evidence_mask"),
            seq_waiting_mask=data.get("seq_waiting_mask"),
            seq_write_mask=data.get("seq_write_mask"),
            seq_decision_mask=data.get("seq_decision_mask"),
            seq_occlusion_mask=data.get("seq_occlusion_mask"),
            seq_read_state_valid=data.get("seq_read_state_valid"),
            seq_read_credit_reachable=data.get("seq_read_credit_reachable"),
            seq_decay_gap_before=data.get("seq_decay_gap_before"),
            seq_use_pressure_mask=data.get("seq_use_pressure_mask"),
            seq_sparse_skip_o=data.get("seq_sparse_skip_o"),
            seq_episode_index=data.get("seq_episode_index"),
            seq_collection_id=data.get("seq_collection_id"),
            seq_object_id=data.get("seq_object_id"),
            seq_memory_cell=data.get("seq_memory_cell"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# Defines the format of the actions. This field is included as "actions" inside the dictionary
# produced by the data transforms.
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for augmax.
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # Back to [-1, 1].
            image = image * 2.0 - 1.0

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    # every other field (tokenized prompt/causal buffers, memory window inputs, ...) passes
    # through unchanged; window_images deliberately skip the augmentation, they emulate past
    # inference-time observations
    return dataclasses.replace(observation, images=out_images, image_masks=out_masks)


def _merge_structural_none_leaves(expected: dict, got: at.Params) -> at.Params:
    """Reinsert `None` leaves that the expected tree records but a restored tree lacks."""
    if not isinstance(expected, dict) or not isinstance(got, dict):
        return got
    merged = dict(got)
    for key, value in expected.items():
        if value is None and key not in merged:
            merged[key] = None
        elif isinstance(value, dict) and key in merged:
            merged[key] = _merge_structural_none_leaves(value, merged[key])
    return merged


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        expected_pure = state.to_pure_dict()
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(expected_pure, params)
        # Orbax drops the structural `None` leaves of bias-free NNX linears on save, so a
        # restored raw params tree is missing exactly those keys; reinsert them where the
        # expected tree records None. Missing real-array keys still fail the strict check.
        params = _merge_structural_none_leaves(expected_pure, params)
        at.check_pytree_equality(expected=expected_pure, got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # If the params were saved with `save_state` during openpi training, every key path will end with "value", which is
    # added by `nnx.State`. We remove the "value" suffix here and always return what NNX calls a "pure dict".
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
