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
    # jax.checkpoint_policies name applied to the scanned Gemma and SigLIP blocks. The default
    # fully recomputes each block's forward during backward (minimum memory). "dots_saveable"
    # keeps matmul outputs alive instead, trading one step's activation memory for skipping
    # that recompute -- profitable on large-memory GPUs (H200) under the per-step outer
    # jax.checkpoint of the memory-sequence losses, which bounds what is alive to one step.
    remat_policy: str = "nothing_saveable"
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
    # v3.3: condition the 16 write queries on the layer-8 hidden states of the non-image
    # (instruction + state) prefix tokens through a zero-init cross-attention residual, so the
    # writer can select task-relevant content ("given my task, what is worth remembering?").
    # Zero-init means an initialized v3.3 model computes exactly the unconditioned v3.2 write.
    memory_task_conditioned_write: bool = False
    # Legacy weight for the old probe-training objective. The main v3/v3.1 recipes keep this at
    # zero: the probe must not train the VLM, policy, memory, or its own head through the main
    # optimizer. The field remains loadable so older experiment configs are still understood.
    memory_probe_weight: float = 0.0
    # Opt-in, detached probe metrics. This reuses the checkpoint-compatible probe head but keeps
    # its read/logits outside the backward graph and never adds its CE to the optimized loss.
    # False skips the probe read and classifier compute entirely.
    memory_probe_diagnostic: bool = False
    memory_probe_classes: int = 2

    # ------------------------------------------------------------------------------------------
    # v3.4 (V34_PLAN_final.md). All default-off so every v3.2/v3.3 config and checkpoint replays
    # bit-identically; pi05_yam_mem_v34 turns them on together.
    # ------------------------------------------------------------------------------------------
    # 5.5: cosine (QK-normalized) attention in both memory compressors and the write-query
    # conditioner, with a learned per-head temperature exp(lambda_h) initialized to
    # sqrt(d_head) (unit-vector dot products have std ~1/sqrt(d_head)) and clamped <= 64.
    memory_qk_norm: bool = False
    # 5.5: raw (height, width) of the top camera BEFORE resize_with_pad to 224x224. Determines
    # the static per-patch letterbox validity mask P_valid over the 16x16 SigLIP patch grid:
    # padding patches are -inf-masked out of both compressors' softmaxes, so a letterbox band
    # is mathematically incapable of becoming the write/read attention sink. None disables.
    memory_letterbox_source_hw: tuple[int, int] | None = None
    # 5.3: memory-token query rows in blocks memory_layer+1..end attend ONLY to the memory
    # positions themselves (they stay K/V sources for everyone else). Without this, a memory
    # token's late-block output is an attention summary of images/state -- a readout register
    # -- even with zero injected content.
    memory_blind_tokens: bool = False
    # 5.4: decode the first causal token from the LAST VALID NON-MEMORY prefix position instead
    # of the last memory token's output (which is exactly zero under blinding + a closed gate).
    memory_reseed_ce: bool = False
    # 5.6: how retrieved memory content is injected as the 16 memory tokens.
    #   "gate":     memory_tokens = memory_gate * retrieved (v3.2/v3.3; measured injection RMS
    #               ~62,600x below the residual stream -- numerically invisible).
    #   "tanh_rms": memory_tokens = tanh(w) * retrieved * c / max(rms(retrieved), tau), with w
    #               a zero-init [d_value] parameter (exact-zero start preserved), rms per token
    #               over channels, c the residual-stream RMS measured at the actual v3.4 init,
    #               and tau a floor so weak reads stay weak (non-amplifying).
    memory_injection_mode: Literal["gate", "tanh_rms"] = "gate"
    memory_injection_c: float = 12.4
    memory_injection_tau: float = 0.02
    # 5.9: context rows of the write-query conditioner. "instruction_state" is the v3.3
    # behavior (all non-image prefix rows); "instruction_only" excludes the state-digit token
    # positions -- state encodes phase strongly and the v3.3 writer collapsed to phase.
    memory_conditioner_context: Literal["instruction_state", "instruction_only"] = "instruction_state"
    # 5.2: probability that a memory-required training segment has its state-digit tokens
    # replaced by a learned null embedding at the input, sampled ONCE PER SEGMENT (per-frame
    # masking would let the model funnel state through the memory itself). Everything
    # downstream of the masked input -- h8, read queries, writes, retrieval, CE -- uses the
    # masked view (single-view default).
    memory_state_mask_prob: float = 0.0
    # 5.2 gold-standard variant: the full view drives memory-state evolution (write tokens)
    # while a second state-masked forward produces the CE/flow and their own retrieval.
    # ~2x prefix compute on masked segments.
    memory_state_mask_dual_view: bool = False
    # 5.1: weight of the task-general auxiliary demand -- decode the per-step subtask label
    # from the POST-write memory through a frame-invariant key-space query bank Q_aux
    # (16 x d_key, L2-normalized, consumed via read_key -- never through read()). The head and
    # bank train in the MAIN optimizer: this loss is *supposed* to train the memory.
    memory_aux_loss_weight: float = 0.0
    memory_aux_num_classes: int = 7
    # 5.1 A/B: "key" reads M_t(Q_aux) directly in key space (default -- fully decoupled from
    # the production reader); "hidden" routes a hidden-space bank through project_q, co-training
    # W_Q toward the stored key space.
    memory_aux_query_space: Literal["key", "hidden"] = "key"
    # 5.1 optional episode-vs-reset margin variant (off by default):
    # L_dep = max(0, gamma - [log p(y|M_t) - stop_grad(log p(y|M_0))]).
    memory_aux_margin_weight: float = 0.0
    memory_aux_margin_gamma: float = 1.0
    # Aux-vocab indices of the side-bearing (memory-required) classes, for the per-class-group
    # accuracy split (phase vs side) in logging. Purely diagnostic.
    memory_aux_side_class_ids: tuple[int, ...] = ()
    # Section 6: online probe-ladder heads (rung 1: side from pooled write tokens on evidence
    # frames; rung 4: side from pooled standard-read retrieval on waiting frames). Features are
    # stop-gradient'ed and the heads are updated by a SEPARATE optimizer in train.py, so the
    # probes measure but cannot train or perturb the main model (verified by a bit-identity
    # unit test). Offline episode-split ladder runs live in the diagnostics package.
    memory_ladder_probes: bool = False

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
            if self.memory_task_conditioned_write and self.memory_architecture != "v32_layer8_dual_query":
                raise ValueError("memory_task_conditioned_write requires the v3.2 dual-query architecture.")
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
            v34_features = {
                "memory_qk_norm": self.memory_qk_norm,
                "memory_letterbox_source_hw": self.memory_letterbox_source_hw is not None,
                "memory_blind_tokens": self.memory_blind_tokens,
                "memory_reseed_ce": self.memory_reseed_ce,
                "memory_injection_mode='tanh_rms'": self.memory_injection_mode == "tanh_rms",
                "memory_conditioner_context='instruction_only'": self.memory_conditioner_context
                == "instruction_only",
                "memory_state_mask_prob": self.memory_state_mask_prob > 0,
                "memory_aux_loss_weight": self.memory_aux_loss_weight > 0,
                "memory_ladder_probes": self.memory_ladder_probes,
            }
            if self.memory_architecture != "v32_layer8_dual_query":
                enabled = [name for name, on in v34_features.items() if on]
                if enabled:
                    raise ValueError(f"v3.4 features require the v3.2 dual-query architecture: {enabled}.")
            if self.memory_injection_mode not in ("gate", "tanh_rms"):
                raise ValueError(f"unsupported memory_injection_mode: {self.memory_injection_mode!r}.")
            if self.memory_injection_mode == "tanh_rms" and (
                self.memory_injection_c <= 0 or self.memory_injection_tau <= 0
            ):
                raise ValueError("tanh_rms injection requires positive memory_injection_c and memory_injection_tau.")
            if self.memory_conditioner_context not in ("instruction_state", "instruction_only"):
                raise ValueError(f"unsupported memory_conditioner_context: {self.memory_conditioner_context!r}.")
            if self.memory_conditioner_context == "instruction_only" and not self.memory_task_conditioned_write:
                raise ValueError("memory_conditioner_context='instruction_only' requires memory_task_conditioned_write.")
            if not 0.0 <= self.memory_state_mask_prob <= 1.0:
                raise ValueError("memory_state_mask_prob must be in [0, 1].")
            if self.memory_state_mask_dual_view and self.memory_state_mask_prob == 0:
                raise ValueError("memory_state_mask_dual_view requires memory_state_mask_prob > 0.")
            if self.memory_letterbox_source_hw is not None and (
                len(self.memory_letterbox_source_hw) != 2 or min(self.memory_letterbox_source_hw) <= 0
            ):
                raise ValueError("memory_letterbox_source_hw must be a positive (height, width) pair.")
            if self.memory_aux_loss_weight < 0 or self.memory_aux_margin_weight < 0:
                raise ValueError("aux loss weights must be >= 0.")
            if self.memory_aux_margin_weight > 0 and self.memory_aux_loss_weight == 0:
                raise ValueError("the aux margin variant complements the aux CE; set memory_aux_loss_weight > 0.")
            if self.memory_aux_loss_weight > 0:
                if self.memory_aux_num_classes < 2:
                    raise ValueError("memory_aux_num_classes must be >= 2.")
                if any(not 0 <= c < self.memory_aux_num_classes for c in self.memory_aux_side_class_ids):
                    raise ValueError("memory_aux_side_class_ids must index into [0, memory_aux_num_classes).")
            if self.memory_aux_query_space not in ("key", "hidden"):
                raise ValueError(f"unsupported memory_aux_query_space: {self.memory_aux_query_space!r}.")
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
                        "token_state_mask": jax.ShapeDtypeStruct([*lead, self.max_token_len], bool),
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
                        **(
                            {"seq_state_masked": jax.ShapeDtypeStruct([batch_size], bool)}
                            if self.memory_state_mask_prob > 0
                            else {}
                        ),
                        **(
                            {"seq_subtask_class": jax.ShapeDtypeStruct(lead, jnp.int32)}
                            if self.memory_aux_loss_weight > 0
                            else {}
                        ),
                        **(
                            {
                                "seq_side_label": jax.ShapeDtypeStruct([batch_size], jnp.int32),
                                "seq_evidence_mask": jax.ShapeDtypeStruct(lead, bool),
                                "seq_waiting_mask": jax.ShapeDtypeStruct(lead, bool),
                            }
                            if self.memory_ladder_probes
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
