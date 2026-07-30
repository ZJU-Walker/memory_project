import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
import openpi.models.memory as _memory
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # If true, the VLM backbone is additionally supervised with a next-token CE loss on the
    # per-frame subtask text + FAST-tokenized actions (knowledge-insulation-style co-training).
    # Only supported for pi05.
    predict_subtask: bool = False
    # Weight of the token CE loss relative to the flow matching loss.
    ce_loss_weight: float = 1.0

    # If true, subtask decoding can be conditioned on a Titans neural memory
    # (openpi.models.memory): the top-camera hidden states of gemma block `memory_layer` are
    # read from / written to a per-episode memory, and the retrieved tokens are appended to the
    # KV cache after the context text. Adds `Pi0.sample_with_memory`; every existing path is
    # untouched when False (the memory params are not even constructed). Requires
    # predict_subtask.
    predict_with_memory: bool = False
    memory_layer: int = (
        17  # gemma block whose top-camera hidden states feed the memory (last layer) #TODO change layer here
    )
    # Slot budget reserved after the memory block for the causal text (the generated subtask at
    # inference; the subtask + FAST labels at training). The action suffix starts at the static
    # position num_img_tokens + max_token_len + num_memory_tokens + causal_token_len.
    causal_token_len: int = 150
    memory: _memory.MemoryConfig = dataclasses.field(default_factory=_memory.MemoryConfig)
    # Memory training: number of past writes per sample (oldest first; with window buckets this
    # is the LARGEST bucket) and how many of the newest are recomputed live with the current VLM
    # (the rest come detached from the trainer's hidden cache; live == window means all-live and
    # no cache). Gradients flow through the FULL write chain (no truncation): the state
    # recursion's backward only touches the small memory MLP, so its cost is independent of the
    # VLM. What IS expensive is VLM co-adaptation, so only every `memory_grad_every`-th live
    # frame (counted backward from the window end, so the spacing is static under start-padding)
    # keeps activations for a VLM backward; the other live frames run forward-only.
    memory_window: int = 16
    memory_live_writes: int = 4
    memory_grad_every: int = 4
    # Quiz probes ("dense supervision"): at each every-`memory_grad_every`-th write position that
    # the data marks quizzable (a real write at/after the episode's reveal frame), read the
    # memory as it stands, mean-pool the gated read output, and classify the episode's answer
    # (banana side) with a small train-only linear head. Loss contribution:
    # memory_probe_weight * (sum of quiz CEs / number of live quizzes). 0 disables the quiz
    # entirely (the head is not constructed, keeping old checkpoints loadable).
    memory_probe_weight: float = 0.0
    memory_probe_classes: int = 2
    # Gradient-checkpoint interval of the write-chain scan: the fast-weight state is saved every
    # this many writes and the in-between activations are recomputed during backward. Must divide
    # every window length the data can deliver (all bucket sizes).
    memory_remat_chunk: int = 1

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.predict_subtask and not self.pi05:
            raise ValueError("predict_subtask is only supported for pi05.")
        if self.predict_with_memory:
            if not self.predict_subtask:
                raise ValueError("predict_with_memory requires predict_subtask.")
            paligemma_config = _gemma.get_config(self.paligemma_variant)
            if self.memory.d_input != paligemma_config.width or self.memory.d_value != paligemma_config.width:
                raise ValueError(f"memory d_input/d_value must equal the PaliGemma width ({paligemma_config.width}).")
            if not 0 <= self.memory_layer < paligemma_config.depth:
                raise ValueError(f"memory_layer must be in [0, {paligemma_config.depth}).")
            if not 1 <= self.memory_live_writes <= self.memory_window:
                raise ValueError("memory_live_writes must be in [1, memory_window].")
            if not 1 <= self.memory_grad_every <= self.memory_window:
                raise ValueError("memory_grad_every must be in [1, memory_window].")
            if not 1 <= self.memory_remat_chunk <= self.memory_window:
                raise ValueError("memory_remat_chunk must be in [1, memory_window].")
            if self.memory_probe_weight < 0:
                raise ValueError("memory_probe_weight must be >= 0.")
            if self.memory_probe_weight > 0 and self.memory_probe_classes < 2:
                raise ValueError("memory_probe_classes must be >= 2 when the quiz is enabled.")
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                **(
                    {
                        "token_ar_mask": jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                        "token_loss_mask": jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                        "token_fast_mask": jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                    }
                    if self.predict_subtask
                    else {}
                ),
                **(
                    {
                        "tokenized_causal": jax.ShapeDtypeStruct([batch_size, self.causal_token_len], jnp.int32),
                        "tokenized_causal_mask": jax.ShapeDtypeStruct([batch_size, self.causal_token_len], bool),
                        "causal_fast_mask": jax.ShapeDtypeStruct([batch_size, self.causal_token_len], bool),
                        "memory_write_mask": jax.ShapeDtypeStruct([batch_size, self.memory_window], bool),
                        **(
                            {
                                "memory_probe_labels": jax.ShapeDtypeStruct([batch_size, self.memory_window], jnp.int32),
                                "memory_probe_mask": jax.ShapeDtypeStruct([batch_size, self.memory_window], bool),
                                "memory_probe_visible": jax.ShapeDtypeStruct([batch_size, self.memory_window], bool),
                            }
                            if self.memory_probe_weight > 0
                            else {}
                        ),
                        "window_images": {
                            name: jax.ShapeDtypeStruct(
                                [batch_size, self.memory_live_writes, *_model.IMAGE_RESOLUTION, 3], jnp.float32
                            )
                            for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                        },
                        "window_tokens": jax.ShapeDtypeStruct(
                            [batch_size, self.memory_live_writes, self.max_token_len], jnp.int32
                        ),
                        "window_token_masks": jax.ShapeDtypeStruct(
                            [batch_size, self.memory_live_writes, self.max_token_len], bool
                        ),
                        **(
                            {
                                # 256 = SigLIP So400m/14 tokens of the top camera at 224x224
                                "memory_hiddens": jax.ShapeDtypeStruct(
                                    [
                                        batch_size,
                                        self.memory_window - self.memory_live_writes,
                                        256,
                                        _gemma.get_config(self.paligemma_variant).width,
                                    ],
                                    jnp.float32,
                                ),
                                "memory_cache_indices": jax.ShapeDtypeStruct(
                                    [batch_size, self.memory_window - self.memory_live_writes], jnp.int32
                                ),
                            }
                            if self.memory_window > self.memory_live_writes
                            else {}
                        ),
                    }
                    if self.predict_with_memory
                    else {}
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
