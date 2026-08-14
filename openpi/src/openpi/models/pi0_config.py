import dataclasses
from typing import TYPE_CHECKING, Literal

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


MemoryArchitecture = Literal["v3_v31", "v32_layer8_dual_query"]
MemoryWriteSource = Literal["raw_hidden", "post_attention", "query_compressed"]


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # If set, enables train-time RTC action-prefix conditioning. The value is the
    # maximum simulated inference delay, inclusive: D samples delays uniformly
    # from {0, ..., D}. None disables RTC; 0 is a valid no-op configuration.
    simulated_delay: int | None = None
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
    memory_layer: int = 17  # gemma block whose top-camera hidden states feed the memory
    # v3/v3.1 append one retrieved token per top-camera slot after a complete prefix prefill.
    # v3.2 instead stops after block 8, uses independent 16-query cross-attention compressors
    # for read and write, inserts only the 16 retrieved tokens, then continues blocks 9..17.
    memory_architecture: MemoryArchitecture = "v3_v31"
    memory_query_tokens: int = 16
    memory_query_heads: int = 8
    # Representation used by the associative memory write. ``raw_hidden`` preserves the v3
    # behavior (write the layer-``memory_layer`` top-camera states); ``post_attention`` writes
    # the final-normalized outputs of the appended memory-token block (v3.1). Reads always use
    # the raw prefix hidden states selected by ``memory_layer``.
    memory_write_source: MemoryWriteSource = "raw_hidden"
    # Slot budget reserved after the memory block for the causal text (the generated subtask at
    # inference; the subtask + FAST labels at training). The action suffix starts at the static
    # position num_img_tokens + max_token_len + num_memory_tokens + causal_token_len.
    causal_token_len: int = 150
    # Opt-in tensor-core projection for the tied 257k-way vocabulary head. The embedding table
    # remains an FP32 parameter; only decode operands use `dtype`, with FP32 accumulation.
    # Default-off keeps every existing pi0/pi0.5 configuration numerically unchanged.
    bf16_vocab_projection: bool = False
    memory: _memory.MemoryConfig = dataclasses.field(default_factory=_memory.MemoryConfig)
    # Sequence training (RoboTTT-style): a training sample is `memory_seq_steps` consecutive
    # prediction steps from one episode, one step per policy replan
    # (`memory_stride_frames` in the data config). This may be shorter than action_horizon for
    # overlapping RTC chunks. At every step the
    # model reads the memory, predicts subtask+FAST (CE) and actions (flow), then writes the
    # frame. Every step's prefill receives VLM gradients from the losses inside its own
    # gradient block (per-step rematerialization keeps only one step's activations alive at a
    # time, so GPU memory does not grow with sequence length).
    memory_seq_steps: int = 16
    # Gradient-block length in steps: backprop through the memory state is cut every this many
    # steps (per-sample random shift comes from the data; the state CONTENT always flows
    # through, only its gradient is detached -- RoboTTT's TBPTT). 0 or >= memory_seq_steps
    # means never cut.
    memory_block_steps: int = 0
    # Legacy weight for the old probe-training objective. The main v3/v3.1 recipes keep this at
    # zero: the probe must not train the VLM, policy, memory, or its own head through the main
    # optimizer. The field remains loadable so older experiment configs are still understood.
    memory_probe_weight: float = 0.0
    # Opt-in, detached probe metrics. This reuses the checkpoint-compatible probe head but keeps
    # its read/logits outside the backward graph and never adds its CE to the optimized loss.
    # False skips the probe read and classifier compute entirely.
    memory_probe_diagnostic: bool = False
    memory_probe_classes: int = 2

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.predict_subtask and not self.pi05:
            raise ValueError("predict_subtask is only supported for pi05.")
        if self.simulated_delay is not None and not 0 <= self.simulated_delay < self.action_horizon:
            raise ValueError("simulated_delay must be in [0, action_horizon), or None to disable RTC.")
        if self.memory_write_source not in ("raw_hidden", "post_attention", "query_compressed"):
            raise ValueError("unsupported memory_write_source.")
        if self.predict_with_memory:
            if not self.predict_subtask:
                raise ValueError("predict_with_memory requires predict_subtask.")
            paligemma_config = _gemma.get_config(self.paligemma_variant)
            if self.memory.d_input != paligemma_config.width or self.memory.d_value != paligemma_config.width:
                raise ValueError(f"memory d_input/d_value must equal the PaliGemma width ({paligemma_config.width}).")
            if not 0 <= self.memory_layer < paligemma_config.depth:
                raise ValueError(f"memory_layer must be in [0, {paligemma_config.depth}).")
            if self.memory_architecture == "v32_layer8_dual_query":
                if self.memory_layer != 8:
                    raise ValueError("v3.2 requires memory_layer=8.")
                if self.memory_write_source != "query_compressed":
                    raise ValueError("v3.2 requires memory_write_source='query_compressed'.")
                if self.memory_query_tokens != 16:
                    raise ValueError("v3.2 uses exactly 16 read queries and 16 write queries.")
                if self.memory_query_heads < 1 or paligemma_config.width % self.memory_query_heads:
                    raise ValueError("memory_query_heads must divide the PaliGemma width.")
            elif self.memory_architecture == "v3_v31":
                if self.memory_write_source == "query_compressed":
                    raise ValueError("query_compressed writes require the v3.2 architecture.")
            else:
                raise ValueError(f"unsupported memory_architecture: {self.memory_architecture!r}.")
            if self.memory_seq_steps < 1:
                raise ValueError("memory_seq_steps must be >= 1.")
            if self.memory_block_steps < 0:
                raise ValueError("memory_block_steps must be >= 0 (0 = never cut).")
            if self.memory_probe_weight < 0:
                raise ValueError("memory_probe_weight must be >= 0.")
            if self.memory_probe_weight > 0 and self.memory_probe_diagnostic:
                raise ValueError(
                    "memory_probe_diagnostic is detached and cannot be combined with a nonzero memory_probe_weight."
                )
            if self.memory_probe_classes < 2:
                raise ValueError("memory_probe_classes must be >= 2 for checkpoint-compatible probe heads.")
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
        # Sequence training (predict_with_memory): every field carries a leading step axis.
        lead = [batch_size, self.memory_seq_steps] if self.predict_with_memory else [batch_size]
        image_spec = jax.ShapeDtypeStruct([*lead, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct(lead, jnp.bool_)

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
                state=jax.ShapeDtypeStruct([*lead, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([*lead, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([*lead, self.max_token_len], bool),
                **(
                    {
                        "token_ar_mask": jax.ShapeDtypeStruct([*lead, self.max_token_len], jnp.int32),
                        "token_loss_mask": jax.ShapeDtypeStruct([*lead, self.max_token_len], bool),
                        "token_fast_mask": jax.ShapeDtypeStruct([*lead, self.max_token_len], bool),
                    }
                    if self.predict_subtask
                    else {}
                ),
                **(
                    {
                        "tokenized_causal": jax.ShapeDtypeStruct([*lead, self.causal_token_len], jnp.int32),
                        "tokenized_causal_mask": jax.ShapeDtypeStruct([*lead, self.causal_token_len], bool),
                        "causal_fast_mask": jax.ShapeDtypeStruct([*lead, self.causal_token_len], bool),
                        "seq_step_mask": jax.ShapeDtypeStruct(lead, bool),
                        "seq_block_boundary": jax.ShapeDtypeStruct(lead, bool),
                        **(
                            {
                                "seq_probe_labels": jax.ShapeDtypeStruct(lead, jnp.int32),
                                "seq_probe_mask": jax.ShapeDtypeStruct(lead, bool),
                                "seq_probe_visible": jax.ShapeDtypeStruct(lead, bool),
                            }
                            if self.memory_probe_weight > 0 or self.memory_probe_diagnostic
                            else {}
                        ),
                    }
                    if self.predict_with_memory
                    else {}
                ),
            )
        action_spec = jax.ShapeDtypeStruct([*lead, self.action_horizon, self.action_dim], jnp.float32)

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
