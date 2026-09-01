import contextlib
import functools
import logging
import math

import augmax
import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import rtc as _rtc
import openpi.models.gemma as _gemma
import openpi.models.memory as _memory
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

PALIGEMMA_EOS_TOKEN = 1


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


def make_memory_step_mask(prefix_mask, prefix_ar, mem_len, causal_len):
    """Attention mask [b, mem, prefix+mem+causal] for the incremental memory-append step: the
    memory tokens attend to the valid ar=0 context (images + prompt/state) and bidirectionally
    to themselves -- never to the causal region (subtask/FAST labels at training, generated
    tokens at inference), so memory cannot launder label information."""
    batch = prefix_mask.shape[0]
    ctx = prefix_mask & (prefix_ar == 0)
    return jnp.concatenate(
        [
            einops.repeat(ctx, "b p -> b m p", m=mem_len),
            jnp.ones((batch, mem_len, mem_len), dtype=bool),
            jnp.zeros((batch, mem_len, causal_len), dtype=bool),
        ],
        axis=-1,
    )


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, "*shape"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "*shape {embedding_dim}"]:
    """Computes sine-cosine embeddings for scalar or token-wise positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "...,j->...j",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


def letterbox_patch_valid(source_hw: tuple[int, int], *, target: int = 224, patch_size: int = 14) -> tuple[bool, ...]:
    """Static per-patch validity mask P_valid for a letterboxed camera (v3.4 plan 5.5).

    Mirrors `image_tools.resize_with_pad` arithmetic exactly (including the int() truncation
    and divmod padding split): a raw ``source_hw`` image is resized to fit ``target x target``
    and padded with black. A SigLIP patch is valid iff its ``patch_size``-pixel cell overlaps
    the real (non-padding) region. Returned flattened row-major over the (target/patch)^2 grid,
    as a hashable tuple so it can live on an nnx.Module as static metadata.

    For the YAM 480x640 top camera at 224x224/14 this marks grid rows 0-1 and 14-15 (the
    letterbox bands) invalid -- the measured v3.3 write-attention sink.
    """
    source_height, source_width = source_hw
    ratio = max(source_width / target, source_height / target)
    resized_height = int(source_height / ratio)
    resized_width = int(source_width / ratio)
    pad_h0, _ = divmod(target - resized_height, 2)
    pad_w0, _ = divmod(target - resized_width, 2)
    if target % patch_size:
        raise ValueError(f"patch size {patch_size} must divide the target resolution {target}.")
    grid = target // patch_size
    valid = []
    for row in range(grid):
        y0, y1 = row * patch_size, (row + 1) * patch_size
        row_valid = y1 > pad_h0 and y0 < pad_h0 + resized_height
        for col in range(grid):
            x0, x1 = col * patch_size, (col + 1) * patch_size
            col_valid = x1 > pad_w0 and x0 < pad_w0 + resized_width
            valid.append(bool(row_valid and col_valid))
    if not any(valid):
        raise ValueError(f"letterbox geometry {source_hw} leaves no valid patch -- refusing an all-masked softmax.")
    return tuple(valid)


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _clip_feature_cotangent(feature: at.Array, limit: float) -> at.Array:
    """Identity forward; cap each example's feature cotangent in the backward pass.

    The v3.5 write/read classifiers consume pooled features outside the memory core API, so
    the core k/v guard cannot bound this direct path into the backbone. This local guard keeps
    the forward feature and head gradients exact while bounding only the gradient entering the
    feature producer.
    """
    return feature


def _clip_feature_cotangent_fwd(feature: at.Array, limit: float):
    del limit
    return feature, None


def _clip_feature_cotangent_bwd(limit: float, _residual, cotangent: at.Array):
    cotangent32 = cotangent.astype(jnp.float32)
    norm = jnp.linalg.norm(cotangent32, axis=-1)
    scale = jnp.minimum(1.0, jnp.asarray(limit, jnp.float32) / (norm + 1e-12))
    return ((cotangent32 * scale[..., None]).astype(cotangent.dtype),)


_clip_feature_cotangent.defvjp(_clip_feature_cotangent_fwd, _clip_feature_cotangent_bwd)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def _side_ce_with_per_term_feature_cap(feature, kernel, bias, label, limit):
    """Cross-entropy whose *unweighted per-term* feature gradient is capped.

    Downstream episode/cell/branch weighting multiplies the already-capped term. Kernel and
    bias gradients remain the exact ordinary cross-entropy gradients.
    """
    logits = feature @ kernel + bias
    ce = -jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), label[:, None], axis=-1)[:, 0]
    return ce, logits


def _side_ce_with_per_term_feature_cap_fwd(feature, kernel, bias, label, limit):
    del limit
    logits = feature @ kernel + bias
    log_probability = jax.nn.log_softmax(logits, axis=-1)
    probability = jnp.exp(log_probability)
    ce = -jnp.take_along_axis(log_probability, label[:, None], axis=-1)[:, 0]
    return (ce, logits), (feature, kernel, label, probability)


def _side_ce_with_per_term_feature_cap_bwd(limit, residual, cotangents):
    feature, kernel, label, probability = residual
    ce_cotangent, logits_cotangent = cotangents
    probability_error = probability - jax.nn.one_hot(label, probability.shape[-1], dtype=jnp.float32)
    raw_feature_grad = probability_error @ kernel.T
    raw_norm = jnp.linalg.norm(raw_feature_grad.astype(jnp.float32), axis=-1)
    cap_scale = jnp.minimum(1.0, jnp.asarray(limit, jnp.float32) / (raw_norm + 1e-12))
    capped_ce_feature_grad = raw_feature_grad * cap_scale[:, None]
    total_logits_cotangent = logits_cotangent + ce_cotangent[:, None] * probability_error
    feature_grad = logits_cotangent @ kernel.T + ce_cotangent[:, None] * capped_ce_feature_grad
    kernel_grad = feature.T @ total_logits_cotangent
    bias_grad = jnp.sum(total_logits_cotangent, axis=0)
    return feature_grad, kernel_grad, bias_grad, None


_side_ce_with_per_term_feature_cap.defvjp(
    _side_ce_with_per_term_feature_cap_fwd,
    _side_ce_with_per_term_feature_cap_bwd,
)


class MemoryQueryCompressor(nnx.Module):
    """Learned query bank that cross-attends to the 256 layer-8 top-camera slots.

    v3.4 (plan 5.5) additions, both default-off so v3.2/v3.3 checkpoints replay bit-exactly:
      * ``qk_norm``: cosine attention -- queries and keys are L2-normalized per head and the
        logits are scaled by a learned per-head temperature ``exp(lambda_h)`` initialized to
        ``sqrt(head_dim)`` (unit-vector dot products have std ~1/sqrt(d), so restoring
        ~unit-std logits needs a MULTIPLIER of sqrt(d), not 1/sqrt(d)) and clamped <= 64 so
        attention cannot re-saturate.
      * ``source_valid``: a static/broadcast per-patch validity mask applied as -inf on the
        logits BEFORE the softmax. Training attention (:meth:`__call__`) and diagnostic
        attention (:meth:`attention_probs`) share :meth:`_attention_logits`, so they are the
        same object by construction -- an invalid patch is mathematically incapable of
        receiving attention in either.
    """

    def __init__(
        self,
        *,
        num_queries: int,
        width: int,
        num_heads: int,
        compute_dtype: jnp.dtype = jnp.float32,
        qk_norm: bool = False,
        rngs: nnx.Rngs,
    ):
        if num_queries < 1 or num_heads < 1 or width % num_heads:
            raise ValueError("query count/heads must be positive and heads must divide width.")
        self.num_queries = num_queries
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.compute_dtype = jnp.dtype(compute_dtype)
        self.qk_norm = qk_norm
        self.query_bank = nnx.Param(
            jax.random.normal(rngs.params(), (num_queries, width), dtype=jnp.float32) / jnp.sqrt(width)
        )
        # Keep FP32 master parameters/optimizer state, but use the configured activation dtype
        # for the four large matrix multiplications. Production pi0.5 uses BF16 here so these
        # projections run on tensor cores; the output is promoted back to FP32 before entering
        # TitansMemory, whose recurrent state and inner update intentionally remain FP32.
        linear = functools.partial(
            nnx.Linear,
            width,
            width,
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
        )
        self.query_proj = linear(rngs=rngs)
        self.key_proj = linear(rngs=rngs)
        self.value_proj = linear(rngs=rngs)
        self.output_proj = linear(rngs=rngs)
        if qk_norm:
            # exp(lambda) = sqrt(head_dim) at init; lambda is learned per head.
            self.logit_scale = nnx.Param(jnp.full((num_heads,), 0.5 * math.log(self.head_dim), dtype=jnp.float32))

    def _attention_logits(
        self,
        q: at.Float[at.Array, "b q h dh"],
        k: at.Float[at.Array, "b n h dh"],
        source_valid: at.Bool[at.Array, " n"] | at.Bool[at.Array, "b n"] | None,
    ) -> at.Float[at.Array, "b h q n"]:
        """Shared FP32 scaled-and-masked logits for __call__ and attention_probs."""
        if self.qk_norm:
            q32 = q.astype(jnp.float32)
            k32 = k.astype(jnp.float32)
            q32 = q32 * jax.lax.rsqrt(jnp.sum(jnp.square(q32), axis=-1, keepdims=True) + 1e-12)
            k32 = k32 * jax.lax.rsqrt(jnp.sum(jnp.square(k32), axis=-1, keepdims=True) + 1e-12)
            scale = jnp.minimum(jnp.exp(self.logit_scale.value), 64.0)
            logits = jnp.einsum("bqhd,bnhd->bhqn", q32, k32, preferred_element_type=jnp.float32)
            logits = logits * scale[None, :, None, None]
        else:
            logits = jnp.einsum("bqhd,bnhd->bhqn", q, k, preferred_element_type=jnp.float32)
            logits = logits * self.head_dim**-0.5
        if source_valid is not None:
            if source_valid.ndim == 1:
                keep = source_valid[None, None, None, :]
            elif source_valid.ndim == 2:
                keep = source_valid[:, None, None, :]
            else:
                raise ValueError(f"source_valid must be [n] or [b, n]; got shape {source_valid.shape}.")
            logits = jnp.where(keep, logits, -jnp.inf)
        return logits

    def __call__(
        self,
        source: at.Float[at.Array, "b n d"],
        queries: at.Float[at.Array, "b q d"] | None = None,
        source_valid: at.Bool[at.Array, " n"] | at.Bool[at.Array, "b n"] | None = None,
    ) -> at.Float[at.Array, "b q d"]:
        """Compress `source` into `num_queries` slots. `queries` overrides the learned bank
        (v3.3 task-conditioned writes); None keeps the unconditioned broadcast bank.
        `source_valid` masks source positions (e.g. letterbox padding patches) out of the
        attention softmax entirely."""
        if source.ndim != 3 or source.shape[-1] != self.width:
            raise ValueError(f"query compressor expects [batch,tokens,{self.width}]; got {source.shape}.")
        batch = source.shape[0]
        if queries is None:
            queries = jnp.broadcast_to(self.query_bank.value[None], (batch, self.num_queries, self.width))
        elif queries.shape != (batch, self.num_queries, self.width):
            raise ValueError(f"queries must have shape {(batch, self.num_queries, self.width)}; got {queries.shape}.")
        q = self.query_proj(queries).reshape(batch, self.num_queries, self.num_heads, self.head_dim)
        k = self.key_proj(source).reshape(batch, source.shape[1], self.num_heads, self.head_dim)
        v = self.value_proj(source).reshape(batch, source.shape[1], self.num_heads, self.head_dim)
        logits = self._attention_logits(q, k, source_valid)
        probs = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        pooled = jnp.einsum("bhqn,bnhd->bqhd", probs, v).reshape(batch, self.num_queries, self.width)
        return self.output_proj(pooled).astype(jnp.float32)

    def attention_probs(
        self,
        source: at.Float[at.Array, "b n d"],
        queries: at.Float[at.Array, "b q d"] | None = None,
        source_valid: at.Bool[at.Array, " n"] | at.Bool[at.Array, "b n"] | None = None,
    ) -> at.Float[at.Array, "b h q n"]:
        """Offline diagnostic view of the query->source attention distribution.

        Recomputes exactly the q/k path of :meth:`__call__` (including an optional
        conditioned query override and the source-validity mask, through the SAME
        :meth:`_attention_logits`) and returns the per-head FP32 softmax weights before they
        are cast to the value dtype, so map mass sums to one per query slot regardless of the
        configured compute dtype.
        """
        if source.ndim != 3 or source.shape[-1] != self.width:
            raise ValueError(f"query compressor expects [batch,tokens,{self.width}]; got {source.shape}.")
        batch = source.shape[0]
        if queries is None:
            queries = jnp.broadcast_to(self.query_bank.value[None], (batch, self.num_queries, self.width))
        q = self.query_proj(queries).reshape(batch, self.num_queries, self.num_heads, self.head_dim)
        k = self.key_proj(source).reshape(batch, source.shape[1], self.num_heads, self.head_dim)
        return jax.nn.softmax(self._attention_logits(q, k, source_valid), axis=-1)


class MemoryQueryConditioner(nnx.Module):
    """Task conditioning for the write query bank (v3.3).

    Cross-attends the learned base write queries to the layer-8 hidden states of the
    non-image prefix tokens (instruction + state) and adds the result as a residual:
    ``Q(I) = Q0 + out_proj(Attn(Q0 -> H_ctx))``. The output projection is ZERO-INITIALIZED,
    so an initialized model writes exactly like the unconditioned v3.2 writer and training
    opens the pathway only as far as the sequence objective finds it useful -- the same
    stability discipline as the zero-init memory content gate. Padding positions are masked
    out of the softmax.
    """

    def __init__(
        self,
        *,
        num_queries: int,
        width: int,
        num_heads: int,
        compute_dtype: jnp.dtype = jnp.float32,
        qk_norm: bool = False,
        rngs: nnx.Rngs,
    ):
        if num_queries < 1 or num_heads < 1 or width % num_heads:
            raise ValueError("query count/heads must be positive and heads must divide width.")
        self.num_queries = num_queries
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.compute_dtype = jnp.dtype(compute_dtype)
        self.qk_norm = qk_norm
        linear = functools.partial(
            nnx.Linear,
            width,
            width,
            use_bias=False,
            dtype=self.compute_dtype,
            param_dtype=jnp.float32,
        )
        self.query_proj = linear(rngs=rngs)
        self.key_proj = linear(rngs=rngs)
        self.value_proj = linear(rngs=rngs)
        self.output_proj = linear(kernel_init=nnx.initializers.zeros_init(), rngs=rngs)
        if qk_norm:
            # v3.4 plan 5.5: cosine attention with a learned per-head temperature, exactly as
            # in MemoryQueryCompressor.
            self.logit_scale = nnx.Param(jnp.full((num_heads,), 0.5 * math.log(self.head_dim), dtype=jnp.float32))

    def __call__(
        self,
        base_queries: at.Float[at.Array, "q d"],
        context: at.Float[at.Array, "b n d"],
        context_mask: at.Bool[at.Array, "b n"],
    ) -> at.Float[at.Array, "b q d"]:
        if base_queries.shape != (self.num_queries, self.width):
            raise ValueError(
                f"base queries must have shape {(self.num_queries, self.width)}; got {base_queries.shape}."
            )
        if context.ndim != 3 or context.shape[-1] != self.width or context.shape[:2] != context_mask.shape:
            raise ValueError(f"context/mask mismatch: {context.shape} vs {context_mask.shape}.")
        batch = context.shape[0]
        queries = jnp.broadcast_to(base_queries[None], (batch, self.num_queries, self.width))
        q = self.query_proj(queries).reshape(batch, self.num_queries, self.num_heads, self.head_dim)
        k = self.key_proj(context).reshape(batch, context.shape[1], self.num_heads, self.head_dim)
        v = self.value_proj(context).reshape(batch, context.shape[1], self.num_heads, self.head_dim)
        if self.qk_norm:
            q32 = q.astype(jnp.float32)
            k32 = k.astype(jnp.float32)
            q32 = q32 * jax.lax.rsqrt(jnp.sum(jnp.square(q32), axis=-1, keepdims=True) + 1e-12)
            k32 = k32 * jax.lax.rsqrt(jnp.sum(jnp.square(k32), axis=-1, keepdims=True) + 1e-12)
            scale = jnp.minimum(jnp.exp(self.logit_scale.value), 64.0)
            logits = jnp.einsum("bqhd,bnhd->bhqn", q32, k32, preferred_element_type=jnp.float32)
            logits = jnp.where(context_mask[:, None, None, :], logits * scale[None, :, None, None], -1e30)
        else:
            logits = jnp.einsum("bqhd,bnhd->bhqn", q, k, preferred_element_type=jnp.float32)
            logits = jnp.where(context_mask[:, None, None, :], logits * self.head_dim**-0.5, -1e30)
        probs = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        pooled = jnp.einsum("bhqn,bnhd->bqhd", probs, v).reshape(batch, self.num_queries, self.width)
        # an all-padding context (never the case in practice) must not leak the uniform softmax
        any_valid = jnp.any(context_mask, axis=-1).astype(jnp.float32)[:, None, None]
        return queries.astype(jnp.float32) + self.output_proj(pooled).astype(jnp.float32) * any_valid


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.simulated_delay = config.simulated_delay
        self.predict_subtask = config.predict_subtask
        self.ce_loss_weight = config.ce_loss_weight
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                decode_dtype=config.dtype if config.bf16_vocab_projection else None,
                adarms=config.pi05,
                remat_policy=config.remat_policy,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
                remat_policy=config.remat_policy,
            )
        )
        fake_image = next(iter(config.fake_obs().images.values()))
        if fake_image.ndim == 5:  # sequence configs carry a step axis; SigLIP sees folded frames
            fake_image = fake_image.reshape(-1, *fake_image.shape[2:])
        img.lazy_init(fake_image, train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        self.predict_with_memory = config.predict_with_memory
        if config.predict_with_memory:
            self.memory = _memory.TitansMemory(config.memory, rngs=rngs)
            # zero-init content gate: an untrained/empty memory injects exactly-zero token content.
            # Kept in the parameter tree even under the v3.4 tanh_rms injection (where it is
            # unused and stays zero) so the tree layout is stable across injection modes.
            self.memory_gate = nnx.Param(jnp.zeros((config.memory.d_value,), dtype=jnp.float32))
            self.memory_layer = config.memory_layer
            self.memory_architecture = config.memory_architecture
            self.memory_write_source = config.memory_write_source
            self.memory_query_tokens = config.memory_query_tokens
            self.causal_token_len = config.causal_token_len
            self.memory_probe_weight = config.memory_probe_weight
            self.memory_probe_diagnostic = config.memory_probe_diagnostic
            # Keep this module in every memory-model parameter tree even when probe computation
            # is disabled. Existing probe-trained v3/v3.1 checkpoints therefore remain strictly
            # loadable; clean no-probe recipes explicitly mask its optimizer updates.
            self.probe_head = nnx.Linear(config.memory.d_value, config.memory_probe_classes, rngs=rngs)
            self.memory_task_conditioned_write = config.memory_task_conditioned_write
            # ---- v3.4 flags (V34_PLAN_final.md); all default to the v3.2/v3.3 behavior ----
            self.memory_injection_mode = config.memory_injection_mode
            self.memory_injection_c = config.memory_injection_c
            self.memory_injection_tau = config.memory_injection_tau
            self.memory_freeze_injection_gate = config.memory_freeze_injection_gate
            self.memory_conditioner_context = config.memory_conditioner_context
            self.memory_blind_tokens = config.memory_blind_tokens
            self.memory_reseed_ce = config.memory_reseed_ce
            self.memory_state_mask_prob = config.memory_state_mask_prob
            self.memory_state_mask_dual_view = config.memory_state_mask_dual_view
            self.memory_aux_loss_weight = config.memory_aux_loss_weight
            self.memory_aux_margin_weight = config.memory_aux_margin_weight
            self.memory_aux_margin_gamma = config.memory_aux_margin_gamma
            self.memory_aux_query_space = config.memory_aux_query_space
            self.memory_aux_side_class_ids = tuple(config.memory_aux_side_class_ids)
            self.memory_ladder_probes = config.memory_ladder_probes
            # v3.5 is entirely opt-in. Keeping these scalar attributes in Python rather than
            # the NNX state means a legacy checkpoint's parameter tree is unchanged.
            self.memory_v35_enabled = config.memory_v35_enabled
            self.memory_write_side_loss_weight = config.memory_write_side_loss_weight
            self.memory_read_side_loss_weight = config.memory_read_side_loss_weight
            self.memory_side_feature_cotangent_clip = config.memory_side_feature_cotangent_clip
            self.memory_num_side_cells = config.memory_num_side_cells
            self.memory_time_consistent_augmentation = config.memory_time_consistent_augmentation
            # Static per-patch letterbox validity over the top camera's SigLIP grid (plan 5.5).
            # SigLIP So400m/14 at 224x224 -> patch 14, 16x16 grid = the 256 h8_top slots.
            self.top_patch_valid = (
                letterbox_patch_valid(tuple(config.memory_letterbox_source_hw))
                if config.memory_letterbox_source_hw is not None
                else None
            )
            if config.memory_injection_mode == "tanh_rms":
                # plan 5.6: memory_tokens = tanh(w) * retrieved * c / max(rms, tau); w zero-init
                # preserves the exact-zero start of the injection.
                initial_w = jnp.arctanh(jnp.asarray(config.memory_injection_gate_init, dtype=jnp.float32))
                self.memory_inject_w = nnx.Param(jnp.full((config.memory.d_value,), initial_w, dtype=jnp.float32))
            if config.memory_blind_tokens:
                # Learned content-free slot embeddings added to the injected memory tokens.
                # This is the plan-5.3 pre-registered register-token fallback, merged into the
                # memory slots: with blinding, a zero-injection memory-token stream is EXACTLY
                # zero through every late block, and each RMSNorm evaluated at zero multiplies
                # the backward by rsqrt(eps)=1e3 -- measured as a 1.6e33 gradient on the
                # injection gate that overflows the global clip norm to inf and freezes ALL
                # training. The slot embeddings carry no memory information (frame- and
                # content-invariant); the content gate's exact-zero start is untouched.
                # Zero-init (v36): RMSNorm renormalizes ANY nonzero slot vector to unit RMS,
                # so a nonzero init at any scale breaks the preregistered step-0 task-health
                # transparency bound (measured 3.1x flow at scales 1.0 through 0.02; 1.02x at
                # exact zero). The v3.4 zero-stream gradient explosion this init once guarded
                # against does not resurface under v3.5's frozen injection gate and cotangent
                # caps: a full-parameter backward through the exactly-zero stream on the real
                # v36 step-0 checkpoint is finite at ordinary scale (global norm 33 vs 47
                # baseline). The embeddings remain trainable and may grow during the pilot.
                self.memory_slot_embedding = nnx.Param(
                    jnp.zeros((config.memory_query_tokens, config.memory.d_value), dtype=jnp.float32)
                )
            if config.memory_state_mask_prob > 0:
                # plan 5.2: learned null embedding substituted for state-digit tokens at the
                # input. Unit-std init (matching embedded-token RMS), NOT zeros: a zero
                # embedding sits at the block-0 RMSNorm singularity, whose rsqrt(eps) backward
                # would inflate the null embedding's gradients ~1e3x and starve everything else
                # through the global clip.
                self.state_null_embedding = nnx.Param(
                    jax.random.normal(rngs.params(), (paligemma_config.width,), dtype=jnp.float32)
                )
            if config.memory_aux_loss_weight > 0:
                # plan 5.1: frame-invariant auxiliary query bank (L2-normalized at use) + head.
                aux_query_dim = (
                    config.memory.d_key if config.memory_aux_query_space == "key" else paligemma_config.width
                )
                self.memory_aux_queries = nnx.Param(
                    jax.random.normal(rngs.params(), (config.memory_query_tokens, aux_query_dim), dtype=jnp.float32)
                )
                self.memory_aux_head = nnx.Linear(config.memory.d_value, config.memory_aux_num_classes, rngs=rngs)
            if config.memory_ladder_probes:
                # Section 6 online rungs: writer content (rung 1) and standard-read retrieval
                # (rung 4). Binary side heads; features are stop-gradient'ed at the loss site
                # and updates are isolated in train.py.
                self.ladder_writer_head = nnx.Linear(paligemma_config.width, 2, rngs=rngs)
                self.ladder_read_head = nnx.Linear(config.memory.d_value, 2, rngs=rngs)
            if config.memory_v35_enabled:
                # These heads are live (features are not detached): L_write teaches the value
                # projection/backbone and L_read teaches the production query/read path. The
                # detached v3.4 ladder remains diagnostic-only.
                self.memory_write_side_head = nnx.Linear(config.memory.d_value, 2, rngs=rngs)
                self.memory_read_side_head = nnx.Linear(config.memory.d_value, 2, rngs=rngs)
            if config.memory_architecture == "v32_layer8_dual_query":
                self.read_query_compressor = MemoryQueryCompressor(
                    num_queries=config.memory_query_tokens,
                    width=paligemma_config.width,
                    num_heads=config.memory_query_heads,
                    compute_dtype=jnp.dtype(config.dtype),
                    qk_norm=config.memory_qk_norm,
                    rngs=rngs,
                )
                self.write_query_compressor = MemoryQueryCompressor(
                    num_queries=config.memory_query_tokens,
                    width=paligemma_config.width,
                    num_heads=config.memory_query_heads,
                    compute_dtype=jnp.dtype(config.dtype),
                    qk_norm=config.memory_qk_norm,
                    rngs=rngs,
                )
                if config.memory_task_conditioned_write:
                    self.write_query_conditioner = MemoryQueryConditioner(
                        num_queries=config.memory_query_tokens,
                        width=paligemma_config.width,
                        num_heads=config.memory_query_heads,
                        compute_dtype=jnp.dtype(config.dtype),
                        qk_norm=config.memory_qk_norm,
                        rngs=rngs,
                    )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    def _select_memory_write_source(
        self,
        raw_hidden: at.Array,
        post_attention: at.Array,
    ) -> at.Array:
        """Selects the configured write representation while keeping all inner math float32."""
        if raw_hidden.shape != post_attention.shape:
            raise ValueError(
                "raw and post-attention memory write sources must have matching shapes; "
                f"got {raw_hidden.shape} and {post_attention.shape}."
            )
        if self.memory_write_source == "raw_hidden":
            return raw_hidden.astype(jnp.float32)
        if self.memory_write_source == "post_attention":
            return post_attention.astype(jnp.float32)
        raise ValueError(f"unsupported memory_write_source: {self.memory_write_source!r}")

    def _v32_empty_cache(self, batch: int, capacity: int, dtype: jnp.dtype) -> _gemma.KVCache:
        config = self.PaliGemma.llm.module.configs[0]
        shape = (config.depth, batch, capacity, config.num_kv_heads, config.head_dim)
        return jnp.zeros(shape, dtype=dtype), jnp.zeros(shape, dtype=dtype)

    def _v32_layer_mask(self, early: at.Array, late: at.Array) -> at.Array:
        """Select early/late key visibility without changing query geometry."""

        if early.shape != late.shape:
            raise ValueError(f"early and late attention masks must match; got {early.shape} and {late.shape}.")
        depth = self.PaliGemma.llm.module.configs[0].depth
        use_late = (jnp.arange(depth) > self.memory_layer).reshape((depth,) + (1,) * early.ndim)
        return jnp.where(use_late, late[None], early[None])

    @staticmethod
    def _pad_attention_columns(mask: at.Array, capacity: int) -> at.Array:
        if mask.shape[-1] > capacity:
            raise ValueError(f"attention mask width {mask.shape[-1]} exceeds cache capacity {capacity}.")
        return jnp.pad(mask, ((0, 0), (0, 0), (0, capacity - mask.shape[-1])))

    def _v32_content_gate(self) -> at.Array:
        """The effective per-channel content scale of the memory injection: the zero-init gate
        (v3.2/v3.3) or tanh(w) under the v3.4 tanh_rms form. Both start exactly zero."""
        if getattr(self, "memory_injection_mode", "gate") == "tanh_rms":
            return jnp.tanh(self.memory_inject_w.value)
        return self.memory_gate.value

    def _v32_inject_memory(self, retrieved: at.Array, gate_value: at.Array | None = None) -> at.Array:
        """Turn raw retrieval into injectable memory-token content (FP32).

        "gate" mode (v3.2/v3.3): gate * retrieved -- measured at ~1/62,600 of residual-stream
        RMS on the v3.3 checkpoint, numerically invisible to blocks 9..17.
        "tanh_rms" mode (v3.4 plan 5.6): tanh(w) * retrieved * c / max(rms(retrieved), tau),
        rms PER TOKEN over channels. The max(rms, tau) floor is non-amplifying: near-zero
        reads (fresh memory) stay near zero instead of being RMS-normalized up to full
        residual scale. `gate_value` overrides the learned scale vector (diagnostics only,
        e.g. probing the gradient pathway while the zero-init scale is still closed).
        """
        gate = self._v32_content_gate() if gate_value is None else gate_value
        if getattr(self, "memory_injection_mode", "gate") == "tanh_rms":
            retrieved32 = retrieved.astype(jnp.float32)
            # sqrt(x + eps^2) rather than sqrt(x): a FRESH memory retrieves exactly zero, where
            # sqrt's derivative is infinite and the zero cotangent from max(rms, tau) would
            # backpropagate 0 * inf = NaN into every memory parameter (the _l2_norm lesson).
            rms = jnp.sqrt(jnp.mean(jnp.square(retrieved32), axis=-1, keepdims=True) + 1e-12)
            normed = retrieved32 * (self.memory_injection_c / jnp.maximum(rms, self.memory_injection_tau))
            return gate * normed
        return gate * retrieved

    def _v35_oracle_injected_content(
        self,
        direction: at.Float[at.Array, "b d"],
        target_rms: float | at.Float[at.Array, " b"],
        *,
        num_slots: int,
    ) -> tuple[at.Float[at.Array, "b q d"], dict[str, at.Array]]:
        """Pin a direct-carry/prototype direction at an explicit injected RMS in FP32.

        This bypasses memory state and query retrieval by construction.  The caller owns the
        condition identity: an episode's final-E ``v_bar`` is direct-carry; the frozen
        requested-side prototype is the correct oracle; the other-side prototype is the donor
        intervention.  All three use this same numerical path, preventing condition-specific
        scale confounds. Invalid directions fail closed to zero and are surfaced in telemetry.
        """
        direction = jnp.asarray(direction)
        if direction.ndim != 2 or direction.shape[1] != self.memory.config.d_value:
            raise ValueError(
                f"v35_oracle_direction must have shape [batch, {self.memory.config.d_value}], got {direction.shape}."
            )
        if not jnp.issubdtype(direction.dtype, jnp.floating):
            raise TypeError(f"v35_oracle_direction must have floating dtype, got {direction.dtype}.")
        batch_size, width = direction.shape
        rms = jnp.asarray(target_rms)
        if not jnp.issubdtype(rms.dtype, jnp.floating):
            raise TypeError(f"v35_oracle_injected_rms must have floating dtype, got {rms.dtype}.")
        if rms.ndim == 0:
            rms = jnp.broadcast_to(rms, (batch_size,))
        elif rms.shape != (batch_size,):
            raise ValueError(
                f"v35_oracle_injected_rms must be a float scalar or shape [batch] ({batch_size},), got {rms.shape}."
            )

        direction32 = direction.astype(jnp.float32)
        rms32 = rms.astype(jnp.float32)
        finite_direction = jnp.all(jnp.isfinite(direction32), axis=-1)
        direction_safe = jnp.where(jnp.isfinite(direction32), direction32, jnp.zeros_like(direction32))
        norm = jnp.linalg.norm(direction_safe, axis=-1)
        valid = finite_direction & jnp.isfinite(rms32) & (rms32 > 0) & (norm >= jnp.asarray(1e-8, jnp.float32))
        safe_rms = jnp.where(jnp.isfinite(rms32) & (rms32 > 0), rms32, jnp.zeros_like(rms32))
        unit = direction_safe / jnp.maximum(norm[:, None], jnp.asarray(1e-8, jnp.float32))
        vector = unit * (safe_rms * jnp.sqrt(jnp.asarray(width, dtype=jnp.float32)))[:, None]
        vector = jnp.where(valid[:, None], vector, jnp.zeros_like(vector))
        content = jnp.broadcast_to(vector[:, None, :], (batch_size, num_slots, width))
        actual_rms = jnp.sqrt(jnp.mean(jnp.square(content), axis=(1, 2)))
        return content.astype(jnp.float32), {
            "v35_oracle_injection_active": jnp.ones((batch_size,), dtype=bool),
            "v35_oracle_injection_valid": valid,
            "v35_oracle_target_rms": safe_rms,
            "v35_oracle_actual_rms": actual_rms.astype(jnp.float32),
        }

    def _v32_top_patch_valid(self, top_token_count: int) -> at.Array | None:
        """The static letterbox validity mask as a device array, or None when not configured."""
        patch_valid = getattr(self, "top_patch_valid", None)
        if patch_valid is None:
            return None
        if len(patch_valid) != top_token_count:
            raise ValueError(
                f"letterbox patch mask covers {len(patch_valid)} patches but the top camera has "
                f"{top_token_count} tokens."
            )
        return jnp.asarray(patch_valid, dtype=bool)

    def _v32_ce_seed_hidden(self, final_prefix: at.Array, base_prefix_mask: at.Array) -> at.Array:
        """First-causal-token seed hidden state (v3.4 plan 5.4): the LAST VALID NON-MEMORY
        prefix position, gathered with the pre-memory 848-wide mask -- never the concatenated
        split mask, whose appended memory positions are all-ones and would select the last
        memory token again. Returns [b, 1, emb]."""
        batch, prefix_len = base_prefix_mask.shape
        if final_prefix.shape[1] < prefix_len:
            raise ValueError(
                f"final_prefix covers {final_prefix.shape[1]} positions but the pre-memory mask is {prefix_len} wide."
            )
        positions = jnp.arange(prefix_len, dtype=jnp.int32)
        last_valid = jnp.max(jnp.where(base_prefix_mask, positions[None], -1), axis=-1)
        last_valid = jnp.maximum(last_valid, 0)  # an all-padding prefix cannot occur in practice
        return final_prefix[jnp.arange(batch), last_valid][:, None]

    def _v32_causal_seed(self, final_prefix: at.Array, base_prefix_mask: at.Array) -> at.Array:
        """The hidden state that predicts causal token 0: reseeded (plan 5.4) or the legacy
        last-memory-token output."""
        if getattr(self, "memory_reseed_ce", False):
            return self._v32_ce_seed_hidden(final_prefix, base_prefix_mask)
        return final_prefix[:, -1:]

    def _v32_apply_state_null(
        self,
        prefix_tokens: at.Array,
        prefix_mask: at.Array,
        token_state_mask: at.Array | None,
        segment_masked: at.Array | None,
    ) -> at.Array:
        """Input-level per-segment state masking (v3.4 plan 5.2): replace the embeddings of the
        state-digit token positions with the learned null embedding for samples whose segment
        drew the mask. A no-op when either input is absent."""
        if token_state_mask is None or segment_masked is None:
            return prefix_tokens
        batch, prefix_len = prefix_mask.shape
        num_img = prefix_len - self.max_token_len
        if token_state_mask.shape != (batch, self.max_token_len):
            raise ValueError(f"token_state_mask must be [batch, {self.max_token_len}]; got {token_state_mask.shape}.")
        replace = token_state_mask & segment_masked[:, None]
        full = jnp.concatenate([jnp.zeros((batch, num_img), dtype=bool), replace], axis=1)
        null = self.state_null_embedding.value.astype(prefix_tokens.dtype)
        return jnp.where(full[..., None], null[None, None, :], prefix_tokens)

    def _v32_prepare_memory_interface(
        self,
        prefix_tokens: at.Array,
        prefix_mask: at.Array,
        prefix_ar: at.Array,
        memory_state: _memory.MemoryState,
        *,
        top_token_count: int,
        zero_read: bool = False,
        gate_value: at.Array | None = None,
        state_token_mask: at.Array | None = None,
        v35_oracle_direction: at.Float[at.Array, "b d"] | None = None,
        v35_oracle_injected_rms: float | at.Float[at.Array, " b"] | None = None,
    ) -> dict[str, at.Array | _gemma.KVCache]:
        """Run only blocks 0..memory_layer and materialize the dual-query interface.

        `gate_value` overrides the learned content gate (diagnostics only, e.g. probing the
        gradient pathway while the zero-init gate is still closed); None uses the parameter.
        `state_token_mask` marks the state-digit positions within the tokenized prompt; it is
        REQUIRED when the conditioner context is 'instruction_only' (v3.4 plan 5.9) so that a
        forgotten call site fails loudly instead of silently running a different writer.
        """

        if self.memory_architecture != "v32_layer8_dual_query":
            raise ValueError("the split layer-8 prefix is only defined for v3.2.")
        oracle_active = v35_oracle_direction is not None or v35_oracle_injected_rms is not None
        if (v35_oracle_direction is None) != (v35_oracle_injected_rms is None):
            raise ValueError("v35_oracle_direction and v35_oracle_injected_rms must be provided together.")
        if oracle_active and not getattr(self, "memory_v35_enabled", False):
            raise ValueError("oracle injection is available only for memory_v35_enabled models.")
        if oracle_active and (zero_read or gate_value is not None):
            raise ValueError("v3.5 oracle injection cannot be combined with zero_read or gate_value.")
        batch, prefix_len = prefix_mask.shape
        mem_len = self.memory_query_tokens
        capacity = prefix_len + mem_len + self.causal_token_len
        depth = self.PaliGemma.llm.module.configs[0].depth
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        cache = self._v32_empty_cache(batch, capacity, prefix_tokens.dtype)

        early_mask = self._pad_attention_columns(make_attn_mask(prefix_mask, prefix_ar), capacity)
        early_active = jnp.arange(depth) <= self.memory_layer
        (h8_all, _), cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=early_mask,
            positions=positions,
            kv_cache=cache,
            cache_position=0,
            active_layers=early_active,
            apply_final_norm=False,
        )
        h8_top = h8_all[:, :top_token_count].astype(jnp.float32)
        source_valid = self._v32_top_patch_valid(top_token_count)
        read_queries = self.read_query_compressor(h8_top, source_valid=source_valid)
        write_queries = None
        if getattr(self, "memory_task_conditioned_write", False):
            # Task conditioning (v3.3): the non-image prefix positions carry the tokenized
            # instruction (and state); their layer-8 hidden states are already computed above.
            num_img = prefix_len - self.max_token_len
            context_mask = prefix_mask[:, num_img:]
            if getattr(self, "memory_conditioner_context", "instruction_state") == "instruction_only":
                # v3.4 plan 5.9: exclude the state-digit rows -- a dedicated phase channel,
                # given that state encodes phase strongly and the v3.3 writer collapsed to it.
                if state_token_mask is None:
                    raise ValueError(
                        "memory_conditioner_context='instruction_only' requires the observation's "
                        "token_state_mask at every call site."
                    )
                if state_token_mask.shape != context_mask.shape:
                    raise ValueError(
                        f"token_state_mask shape {state_token_mask.shape} must match the text rows "
                        f"{context_mask.shape}."
                    )
                context_mask = context_mask & ~state_token_mask
            write_queries = self.write_query_conditioner(
                self.write_query_compressor.query_bank.value,
                h8_all[:, num_img:].astype(jnp.float32),
                context_mask,
            )
        write_tokens = self.write_query_compressor(h8_top, queries=write_queries, source_valid=source_valid)
        retrieved = self.memory.read(memory_state, read_queries)
        if zero_read:
            retrieved = jnp.zeros_like(retrieved)
        if oracle_active:
            injected_content, oracle_aux = self._v35_oracle_injected_content(
                v35_oracle_direction,
                v35_oracle_injected_rms,
                num_slots=mem_len,
            )
        else:
            injected_content = self._v32_inject_memory(retrieved, gate_value).astype(jnp.float32)
            oracle_aux = {
                "v35_oracle_injection_active": jnp.zeros((batch,), dtype=bool),
                "v35_oracle_injection_valid": jnp.zeros((batch,), dtype=bool),
                "v35_oracle_target_rms": jnp.zeros((batch,), dtype=jnp.float32),
                "v35_oracle_actual_rms": jnp.zeros((batch,), dtype=jnp.float32),
            }
        # Keep both sides of the sole mixed-precision boundary visible. Calibration and raw
        # retrieval remain FP32; only the token entering the Transformer is cast.
        injected_pre_cast_rms = jnp.sqrt(jnp.mean(jnp.square(injected_content), axis=(1, 2)))
        injected_post_cast = injected_content.astype(prefix_tokens.dtype)
        injected_post_cast_rms = jnp.sqrt(jnp.mean(jnp.square(injected_post_cast.astype(jnp.float32)), axis=(1, 2)))
        memory_content = injected_content
        slot_embedding = getattr(self, "memory_slot_embedding", None)
        if slot_embedding is not None:
            # Content-free learned slot embeddings (see __init__): they survive zero_read on
            # purpose -- zero_read is the CONTENT ablation; the slots' structural presence is
            # the separate token ablation (plan 5.3 correction).
            memory_content = memory_content + slot_embedding.value[None]
        memory_tokens = memory_content.astype(prefix_tokens.dtype)

        return {
            "cache": cache,
            "h8_all": h8_all,
            "h8_top": h8_top,
            "memory_tokens": memory_tokens,
            "read_queries": read_queries,
            # None when unconditioned; the conditioned [b, q, d] bank otherwise (diagnostics
            # must pass it to attention_probs to see the attention the write actually used)
            "write_queries": write_queries,
            "write_tokens": write_tokens,
            "retrieved": retrieved,
            "injected_pre_cast_rms": injected_pre_cast_rms,
            "injected_post_cast_rms": injected_post_cast_rms,
            **oracle_aux,
            "prefix_mask": prefix_mask,
            "prefix_ar": prefix_ar,
            "capacity": jnp.asarray(capacity, dtype=jnp.int32),
        }

    def _v32_split_late_mask(self, split_mask: at.Array, split_ar: at.Array, prefix_len: int) -> at.Array:
        """The blocks memory_layer+1..end attention mask over [h8_all | memory tokens].

        With ``memory_blind_tokens`` (v3.4 plan 5.3) the memory-token QUERY rows attend only to
        the memory positions themselves (self/mutual 16x16 block -- never fully masked, which
        would NaN the softmax) while remaining K/V sources for every other row: a memory
        token's late-block output becomes a function of retrieved content only, not an
        attention summary of images/state ("readout register" hijack).
        """
        full = make_attn_mask(split_mask, split_ar)
        if getattr(self, "memory_blind_tokens", False):
            split_len = split_mask.shape[1]
            is_memory = jnp.arange(split_len) >= prefix_len
            full = jnp.where(is_memory[None, :, None], is_memory[None, None, :], full)
        return full

    def _v32_prepare_memory_prefix(
        self,
        prefix_tokens: at.Array,
        prefix_mask: at.Array,
        prefix_ar: at.Array,
        memory_state: _memory.MemoryState,
        *,
        top_token_count: int,
        zero_read: bool = False,
        gate_value: at.Array | None = None,
        state_token_mask: at.Array | None = None,
        v35_oracle_direction: at.Float[at.Array, "b d"] | None = None,
        v35_oracle_injected_rms: float | at.Float[at.Array, " b"] | None = None,
    ) -> dict[str, at.Array | _gemma.KVCache]:
        """Run blocks 0..8, form q/z, inject 16 reads, then run blocks 9..17 once.

        `gate_value` overrides the learned content gate (diagnostics only, e.g. probing the
        gradient pathway while the zero-init gate is still closed); None uses the parameter.
        """

        prepared = self._v32_prepare_memory_interface(
            prefix_tokens,
            prefix_mask,
            prefix_ar,
            memory_state,
            top_token_count=top_token_count,
            zero_read=zero_read,
            gate_value=gate_value,
            state_token_mask=state_token_mask,
            v35_oracle_direction=v35_oracle_direction,
            v35_oracle_injected_rms=v35_oracle_injected_rms,
        )
        batch, prefix_len = prefix_mask.shape
        mem_len = self.memory_query_tokens
        capacity = prefix_len + mem_len + self.causal_token_len
        depth = self.PaliGemma.llm.module.configs[0].depth
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        cache = prepared["cache"]
        h8_all = prepared["h8_all"]
        memory_tokens = prepared["memory_tokens"]

        split_tokens = jnp.concatenate([h8_all, memory_tokens], axis=1)
        split_mask = jnp.concatenate([prefix_mask, jnp.ones((batch, mem_len), dtype=bool)], axis=1)
        split_ar = jnp.concatenate([prefix_ar, jnp.zeros((batch, mem_len), dtype=prefix_ar.dtype)], axis=1)
        late_mask = self._pad_attention_columns(self._v32_split_late_mask(split_mask, split_ar, prefix_len), capacity)
        split_positions = jnp.concatenate(
            [positions, prefix_len + jnp.broadcast_to(jnp.arange(mem_len), (batch, mem_len))], axis=1
        )
        late_active = jnp.arange(depth) > self.memory_layer
        (final_prefix, _), cache = self.PaliGemma.llm(
            [split_tokens, None],
            mask=late_mask,
            positions=split_positions,
            kv_cache=cache,
            cache_position=0,
            active_layers=late_active,
        )
        return {
            **prepared,
            "cache": cache,
            "final_prefix": final_prefix,
        }

    def _v32_causal_mask(self, prefix_mask: at.Array, causal_mask: at.Array) -> at.Array:
        batch, causal_len = causal_mask.shape
        mem_len = self.memory_query_tokens
        causal_self = jnp.tril(jnp.ones((causal_len, causal_len), dtype=bool))[None] & causal_mask[:, None, :]
        prefix_rows = einops.repeat(prefix_mask, "b p -> b c p", c=causal_len)
        early = jnp.concatenate(
            [prefix_rows, jnp.zeros((batch, causal_len, mem_len), dtype=bool), causal_self], axis=-1
        )
        late = jnp.concatenate([prefix_rows, jnp.ones((batch, causal_len, mem_len), dtype=bool), causal_self], axis=-1)
        return self._v32_layer_mask(early, late)

    def _v32_step_mask(self, prefix_mask: at.Array, generated_count: at.Array) -> at.Array:
        batch = prefix_mask.shape[0]
        mem_len = self.memory_query_tokens
        gen_valid = jnp.broadcast_to(
            jnp.arange(self.causal_token_len)[None, :] < generated_count, (batch, self.causal_token_len)
        )
        early = jnp.concatenate([prefix_mask, jnp.zeros((batch, mem_len), dtype=bool), gen_valid], axis=1)
        late = jnp.concatenate([prefix_mask, jnp.ones((batch, mem_len), dtype=bool), gen_valid], axis=1)
        return self._v32_layer_mask(early[:, None, :], late[:, None, :])

    def _v32_suffix_mask(
        self, prefix_mask: at.Array, causal_mask: at.Array, suffix_mask: at.Array, suffix_ar: at.Array
    ) -> at.Array:
        batch = prefix_mask.shape[0]
        mem_len = self.memory_query_tokens
        early_view = jnp.concatenate([prefix_mask, jnp.zeros((batch, mem_len), dtype=bool), causal_mask], axis=1)
        late_view = jnp.concatenate([prefix_mask, jnp.ones((batch, mem_len), dtype=bool), causal_mask], axis=1)
        suffix_self = make_attn_mask(suffix_mask, suffix_ar)
        early = jnp.concatenate(
            [einops.repeat(early_view, "b p -> b s p", s=suffix_mask.shape[1]), suffix_self], axis=-1
        )
        late = jnp.concatenate([einops.repeat(late_view, "b p -> b s p", s=suffix_mask.shape[1]), suffix_self], axis=-1)
        return self._v32_layer_mask(early, late)

    def _check_action_prefix_shapes(self, action_prefix: _rtc.ActionPrefix | None, batch_size: int) -> None:
        """Performs static checks that remain safe while sampling is jitted.

        Value bounds are validated before batching by ``rtc.validate_action_prefix``
        at the policy/runtime boundary.
        """
        if action_prefix is None:
            return
        if self.simulated_delay is None:
            raise ValueError("action_prefix requires a checkpoint trained with simulated_delay enabled.")
        expected_actions = (batch_size, self.action_horizon, self.action_dim)
        if action_prefix.actions.shape != expected_actions:
            raise ValueError(
                f"action_prefix.actions must have shape {expected_actions}; got {action_prefix.actions.shape}."
            )
        if action_prefix.delay.shape != (batch_size,) or action_prefix.prefix_length.shape != (batch_size,):
            raise ValueError("action_prefix delay and prefix_length must both have shape [batch].")

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Int[at.Array, "b s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other --> AR mask = 0
            ar_mask.append(0 * input_mask[-1])

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            if obs.token_ar_mask is not None:
                # per-sample AR mask (subtask co-training: causal subtask + FAST branches)
                ar_mask.append(obs.token_ar_mask)
            else:
                # full attention between image and language inputs
                ar_mask.append(0 * obs.tokenized_prompt_mask)
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.concatenate(ar_mask, axis=1).astype(jnp.int32)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Array
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Array | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            if time_emb.ndim == action_tokens.ndim - 1:
                time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            elif time_emb.shape[:-1] == action_tokens.shape[:-1]:
                time_tokens = time_emb
            else:
                raise ValueError(
                    f"timestep embedding must be per-example or per-action; got {time_emb.shape} "
                    f"for actions {action_tokens.shape}."
                )
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"] | dict[str, at.Array]:
        if self.predict_with_memory and observation.seq_step_mask is not None:
            return self._compute_sequence_loss(rng, observation, actions, train=train)

        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        rtc_loss_mask = None
        model_time = time
        if self.simulated_delay is None:
            time_expanded = time[..., None, None]
            x_t = time_expanded * noise + (1 - time_expanded) * actions
        else:
            # Derive the new stream without changing the existing noise/time streams.
            delay_rng = jax.random.fold_in(rng, 0x525443)
            delay = jax.random.randint(delay_rng, batch_shape, 0, self.simulated_delay + 1)  # inclusive maximum
            x_t, model_time, rtc_loss_mask = _rtc.make_noisy_actions(actions, noise, time, delay=delay)
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, model_time)

        if not self.predict_subtask:
            # one big forward pass of prefix + suffix at once
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            suffix_ar = jnp.broadcast_to(suffix_ar_mask.astype(jnp.int32), suffix_mask.shape)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar], axis=1)
            attn_mask = make_attn_mask(input_mask, ar_mask)
            positions = jnp.cumsum(input_mask, axis=1) - 1
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
            return _rtc.renormalize_flow_loss(flow_loss, rtc_loss_mask)

        # Subtask + FAST co-training (knowledge insulation): the VLM backbone is trained by a
        # next-token CE on the subtask + FAST tokens (pass 1); the action expert is trained by flow
        # matching against a stop-gradient'ed prefix (pass 2). With all token masks zero this is
        # mathematically equivalent to the joint pass above.

        # Pass 1: prefix through the VLM expert only, exactly like the prefill in `sample_actions`.
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
        )

        # Next-token CE over the text region of the prefix: position i predicts token i+1.
        text_out = prefix_out[:, -self.max_token_len :]
        logits = self.PaliGemma.llm(text_out[:, :-1], method="decode").astype(jnp.float32)
        targets = observation.tokenized_prompt[:, 1:]
        loss_mask = observation.token_loss_mask[:, 1:]
        token_logp = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), targets[..., None], axis=-1)[..., 0]
        ce_loss = -jnp.sum(token_logp * loss_mask, axis=-1) / jnp.clip(jnp.sum(loss_mask, axis=-1), 1)

        # Pass 2: suffix through the action expert, attending to the cached prefix. The stop
        # gradient insulates the VLM from the flow loss. The FAST branch exists only at training
        # time, so it is hidden from the suffix in both attention and the position offset -- the
        # action expert sees the same prefix geometry as at inference time.
        kv_cache = jax.lax.stop_gradient(kv_cache)
        num_img_tokens = prefix_mask.shape[1] - self.max_token_len
        fast_mask = jnp.concatenate(
            [jnp.zeros((prefix_mask.shape[0], num_img_tokens), dtype=bool), observation.token_fast_mask], axis=1
        )
        suffix_view = prefix_mask & ~fast_mask
        prefix_attn_mask = einops.repeat(suffix_view, "b p -> b s p", s=suffix_tokens.shape[1])
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        positions = jnp.sum(suffix_view, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        return {"flow": _rtc.renormalize_flow_loss(flow_loss, rtc_loss_mask), "ce": ce_loss}

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        action_prefix: _rtc.ActionPrefix | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        self._check_action_prefix_shapes(action_prefix, batch_size)
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            if action_prefix is None:
                model_x_t = x_t
                model_time = jnp.broadcast_to(time, batch_size)
            else:
                model_x_t, model_time = _rtc.condition_action_prefix(x_t, time, action_prefix)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, model_x_t, model_time
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return _rtc.restore_action_prefix(x_0, action_prefix)

    def _decode_subtask(self, preprocessed: _model.Observation, *, stop_token: int, max_decode_steps: int):
        """Prefill + greedy AR decoding of the subtask, using the indexed KV cache.

        Returns the updated (prompt, prompt_mask, ar_mask) text arrays plus everything needed to
        run the flow denoising against the same cache: (kv_cache, prefix_mask, n0, num_img).
        Generated tokens are written both into the text arrays (at each sample's own cursor) and
        into the appended cache slots [prefix_len:] shared across samples; their RoPE positions
        continue each sample's own sequence, so the geometry matches training exactly.
        """
        # embed the images once; only the newest token is embedded per decoding step
        img_tokens = []
        img_masks = []
        for name in preprocessed.images:
            image_tokens, _ = self.PaliGemma.img(preprocessed.images[name], train=False)
            img_tokens.append(image_tokens)
            img_masks.append(einops.repeat(preprocessed.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
        img_tokens = jnp.concatenate(img_tokens, axis=1)
        img_mask = jnp.concatenate(img_masks, axis=1)

        prompt = preprocessed.tokenized_prompt
        prompt_mask = preprocessed.tokenized_prompt_mask
        ar = (
            preprocessed.token_ar_mask
            if preprocessed.token_ar_mask is not None
            else jnp.zeros(prompt.shape, dtype=jnp.int32)
        )
        batch, text_len = prompt.shape
        num_img = img_mask.shape[1]
        prefix_len = num_img + text_len
        batch_idx = jnp.arange(batch)
        n0 = jnp.sum(prompt_mask, axis=-1).astype(jnp.int32)  # [b] first free slot in the text region

        # prefill through the standard path, then pad the cache with slots for the generated tokens
        prefix_tokens = jnp.concatenate([img_tokens, self.PaliGemma.llm(prompt, method="embed")], axis=1)
        prefix_mask = jnp.concatenate([img_mask, prompt_mask], axis=1)
        prefix_ar = jnp.concatenate([0 * img_mask, ar], axis=1).astype(jnp.int32)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=attn_mask, positions=positions)
        kv_cache = jax.tree.map(lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, max_decode_steps), (0, 0), (0, 0))), kv_cache)

        def greedy(hidden):  # [b, emb] -> [b] next token
            logits = self.PaliGemma.llm(hidden[:, None], method="decode")[:, 0]
            return jnp.argmax(logits, axis=-1).astype(prompt.dtype)

        def write(carry, token, k):
            """Appends `token` as the k-th generated token of every unfinished sample."""
            prompt, prompt_mask, ar, done = carry
            idx = jnp.minimum(n0 + k, text_len - 1)
            keep = done | (n0 + k >= text_len)  # already stopped, or the text region is full
            prompt = prompt.at[batch_idx, idx].set(jnp.where(keep, prompt[batch_idx, idx], token))
            prompt_mask = prompt_mask.at[batch_idx, idx].set(jnp.where(keep, prompt_mask[batch_idx, idx], True))  # noqa: FBT003
            ar = ar.at[batch_idx, idx].set(jnp.where(keep, ar[batch_idx, idx], 1))
            done = keep | (token == stop_token) | (token == PALIGEMMA_EOS_TOKEN)
            return prompt, prompt_mask, ar, done

        # the first generated token comes from the prefill output at each sample's last valid position
        token0 = greedy(prefix_out[batch_idx, num_img + n0 - 1])
        written = write((prompt, prompt_mask, ar, jnp.zeros(batch, dtype=bool)), token0, 0)

        def cond(carry):
            _, _, _, done, _, _, k = carry
            return (k < max_decode_steps) & ~jnp.all(done)

        def step(carry):
            prompt, prompt_mask, ar, done, prev, kv_cache, k = carry
            # feed the previous token: its k/v land in cache slot prefix_len + k - 1
            tok_emb = self.PaliGemma.llm(prev[:, None], method="embed")
            pos = (num_img + n0 + k - 1)[:, None]
            gen_valid = jnp.arange(max_decode_steps)[None, :] < k  # cache slots of generated tokens 0..k-1
            step_mask = jnp.concatenate([prefix_mask, jnp.broadcast_to(gen_valid, (batch, max_decode_steps))], axis=1)
            (out, _), kv_cache = self.PaliGemma.llm(
                [tok_emb, None],
                mask=step_mask[:, None, :],
                positions=pos,
                kv_cache=kv_cache,
                cache_position=prefix_len + k - 1,
            )
            token = greedy(out[:, 0])
            prompt, prompt_mask, ar, done = write((prompt, prompt_mask, ar, done), token, k)
            return prompt, prompt_mask, ar, done, token, kv_cache, k + 1

        carry = (*written, token0, kv_cache, jnp.asarray(1, dtype=jnp.int32))
        prompt, prompt_mask, ar, _, prev, kv_cache, k = jax.lax.while_loop(cond, step, carry)

        # commit the last generated token's K/V: the loop writes a token's K/V on the following
        # iteration, so the token that ended decoding (typically the stop terminator) would
        # otherwise leave zeros in its cache slot for the flow suffix to attend to
        tok_emb = self.PaliGemma.llm(prev[:, None], method="embed")
        gen_valid = jnp.arange(max_decode_steps)[None, :] < k
        final_mask = jnp.concatenate([prefix_mask, jnp.broadcast_to(gen_valid, (batch, max_decode_steps))], axis=1)
        _, kv_cache = self.PaliGemma.llm(
            [tok_emb, None],
            mask=final_mask[:, None, :],
            positions=(num_img + n0 + k - 1)[:, None],
            kv_cache=kv_cache,
            cache_position=prefix_len + k - 1,
        )
        return prompt, prompt_mask, ar, kv_cache, prefix_mask, n0, num_img

    def sample_subtask(
        self, observation: _model.Observation, *, stop_token: int, max_decode_steps: int = 10
    ) -> _model.Observation:
        """Greedily decodes the subtask from the VLM backbone and returns the observation with the
        generated tokens appended to the prompt (input/ar masks updated). Per-sample generation
        stops on `stop_token` (the trained subtask terminator "\\n") or the PaliGemma EOS token.
        """
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        prompt, prompt_mask, ar, *_ = self._decode_subtask(
            preprocessed, stop_token=stop_token, max_decode_steps=max_decode_steps
        )
        return observation.replace(tokenized_prompt=prompt, tokenized_prompt_mask=prompt_mask, token_ar_mask=ar)

    def sample_subtask_and_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        stop_token: int,
        max_decode_steps: int = 10,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        action_prefix: _rtc.ActionPrefix | None = None,
    ) -> tuple[_model.Actions, _model.Observation]:
        """Fused inference: one prefill, AR subtask decoding, then flow denoising against the same
        KV cache (the actions are conditioned on the freshly decoded subtask). Returns the actions
        and the observation with the decoded subtask appended to the prompt.
        """
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        prompt, prompt_mask, ar, kv_cache, prefix_mask, n0, num_img = self._decode_subtask(
            preprocessed, stop_token=stop_token, max_decode_steps=max_decode_steps
        )
        batch = prompt.shape[0]
        self._check_action_prefix_shapes(action_prefix, batch)

        # The suffix attends to the valid prefix slots plus each sample's generated cache slots,
        # at positions continuing after them -- the same geometry as compute_loss pass 2.
        gen_len = jnp.sum(prompt_mask, axis=-1).astype(jnp.int32) - n0
        gen_valid = jnp.arange(max_decode_steps)[None, :] < gen_len[:, None]
        suffix_view = jnp.concatenate([prefix_mask, gen_valid], axis=1)
        offset = jnp.sum(suffix_view, axis=-1)

        dt = -1.0 / num_steps
        if noise is None:
            noise = jax.random.normal(rng, (batch, self.action_horizon, self.action_dim))

        def step(carry):
            x_t, time = carry
            if action_prefix is None:
                model_x_t = x_t
                model_time = jnp.broadcast_to(time, batch)
            else:
                model_x_t, model_time = _rtc.condition_action_prefix(x_t, time, action_prefix)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                preprocessed, model_x_t, model_time
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(suffix_view, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = offset[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        x_0 = _rtc.restore_action_prefix(x_0, action_prefix)
        observation = observation.replace(tokenized_prompt=prompt, tokenized_prompt_mask=prompt_mask, token_ar_mask=ar)
        return x_0, observation

    @at.typecheck
    def extract_topcam_hidden(self, observation: _model.Observation) -> at.Float[at.Array, "l b n emb"]:
        """Per-layer hidden states of the top-camera tokens from the VLM prefix forward.

        Runs the same prefill as `sample_actions` with per-layer capture and returns the slice
        belonging to the first camera (`base_0_rgb`, the top camera -- images are embedded first,
        in dict order): [num_layers, b, tokens_per_camera, width]. Entry [L] is the output of
        gemma block L ([-1] = the final hidden state before the output norm). Feed a chosen layer
        into `openpi.models.memory.TitansMemory` to write/read episodic memory.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, _, hidden = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
            return_hidden_states=True,
        )
        num_img_tokens = prefix_mask.shape[1] - self.max_token_len
        tokens_per_cam = num_img_tokens // len(observation.images)
        return hidden[0][:, :, :tokens_per_cam]

    def writer_contribution_step(
        self,
        observation: _model.Observation,
        memory_state: _memory.MemoryState,
        *,
        allow_write: bool = True,
    ) -> tuple[_memory.MemoryState, dict[str, at.Array]]:
        """Run only the v3/v3.1 memory path and diagnose its 256-token write.

        This is an offline diagnostic fast path.  It deliberately stops after the image/context
        prefill, read, and memory-token attention step: it does not decode a subtask or denoise an
        action chunk.  The selected write representation is therefore exactly the same ``h_t``
        (v3) or final ``c18`` (v3.1) that :meth:`sample_with_memory` would write, while avoiding
        all downstream policy compute.

        Per-token errors and fast-weight gradient norms are evaluated against ``memory_state``
        *before* the frame-level write.  ``allow_write=False`` still returns all diagnostics but
        leaves the complete fast state unchanged.
        """
        assert self.predict_with_memory, "the model was not built with predict_with_memory"
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        batch = preprocessed.state.shape[0]

        # This intentionally mirrors sample_with_memory's prefill rather than going through
        # embed_prefix: memory inference has a fixed [images | context text] prefix and needs the
        # selected intermediate Gemma layer for the top-camera tokens.
        img_tokens = []
        img_masks = []
        for name in preprocessed.images:
            image_tokens, _ = self.PaliGemma.img(preprocessed.images[name], train=False)
            img_tokens.append(image_tokens)
            img_masks.append(einops.repeat(preprocessed.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
        img_tokens = jnp.concatenate(img_tokens, axis=1)
        img_mask = jnp.concatenate(img_masks, axis=1)

        prompt = preprocessed.tokenized_prompt
        prompt_mask = preprocessed.tokenized_prompt_mask
        ar = (
            preprocessed.token_ar_mask
            if preprocessed.token_ar_mask is not None
            else jnp.zeros(prompt.shape, dtype=jnp.int32)
        )
        num_img = img_mask.shape[1]
        prefix_len = num_img + prompt.shape[1]
        mem_len = num_img // len(preprocessed.images)
        prefix_tokens = jnp.concatenate([img_tokens, self.PaliGemma.llm(prompt, method="embed")], axis=1)
        prefix_mask = jnp.concatenate([img_mask, prompt_mask], axis=1)
        prefix_ar = jnp.concatenate([0 * img_mask, ar], axis=1).astype(jnp.int32)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache, hidden = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=make_attn_mask(prefix_mask, prefix_ar),
            positions=positions,
            return_hidden_states=True,
        )
        h_t = hidden[0][self.memory_layer][:, :mem_len].astype(jnp.float32)

        retrieved = self.memory.read(memory_state, h_t)
        mem_tokens = (self.memory_gate.value * retrieved).astype(prefix_tokens.dtype)
        # Keep the same cache width and masked causal tail as normal memory inference.  This is
        # cheap relative to the VLM prefill and makes c_t numerically comparable to that path.
        kv_cache = jax.tree.map(
            lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, mem_len + self.causal_token_len), (0, 0), (0, 0))),
            kv_cache,
        )
        mem_positions = prefix_len + jnp.broadcast_to(jnp.arange(mem_len), (batch, mem_len))
        (mem_out, _), _ = self.PaliGemma.llm(
            [mem_tokens, None],
            mask=make_memory_step_mask(prefix_mask, prefix_ar, mem_len, self.causal_token_len),
            positions=mem_positions,
            kv_cache=kv_cache,
            cache_position=prefix_len,
        )
        c_t = mem_out.astype(jnp.float32)
        write_source = self._select_memory_write_source(h_t, c_t)

        token_aux = self.memory.token_write_diagnostics(memory_state, write_source)
        candidate_state, write_aux = self.memory.write(memory_state, write_source)
        new_state = candidate_state if allow_write else memory_state
        aux = {
            **write_aux,
            **token_aux,
            "retrieval_norm": jnp.sqrt(jnp.mean(jnp.square(retrieved.astype(jnp.float32)), axis=(1, 2))),
            "write_source_norm": jnp.sqrt(jnp.mean(jnp.square(write_source), axis=(1, 2))),
            "memory_gate_norm": jnp.broadcast_to(jnp.linalg.norm(self.memory_gate.value), (batch,)),
            "write_occurred": jnp.full((batch,), allow_write, dtype=bool),
        }
        return new_state, aux

    def _final_ct_from_image_embeddings(
        self,
        preprocessed: _model.Observation,
        memory_state: _memory.MemoryState,
        image_embeddings: tuple[at.Array, ...],
        *,
        include_zero_read: bool,
    ) -> tuple[at.Array, at.Array | None, at.Array]:
        """Shared exact inference primal for final-c_t diagnostics.

        ``image_embeddings`` must follow ``preprocessed.images`` order.  SigLIP is kept outside
        this helper so callers can either use its ordinary outputs or substitute patch-space
        interventions.  The prefix is evaluated once even when the zero-read counterfactual is
        requested.
        """
        batch = preprocessed.state.shape[0]
        camera_names = tuple(preprocessed.images)
        if len(image_embeddings) != len(camera_names):
            raise ValueError(f"expected {len(camera_names)} camera embedding tensors; got {len(image_embeddings)}.")
        if not image_embeddings:
            raise ValueError("final-c_t diagnostics require at least one camera embedding tensor.")

        mem_len = image_embeddings[0].shape[1]
        for name, tokens in zip(camera_names, image_embeddings, strict=True):
            if tokens.ndim != 3 or tokens.shape[0] != batch:
                raise ValueError(
                    f"camera {name} patch embeddings must have shape [batch, tokens, width]; got {tokens.shape}."
                )
            if tokens.shape[1] != mem_len:
                raise ValueError("all cameras must produce the same number of SigLIP patch embeddings.")
            if tokens.shape[-1] != self.memory.config.d_input:
                raise ValueError(
                    f"camera {name} patch width must equal memory d_input "
                    f"({self.memory.config.d_input}); got {tokens.shape[-1]}."
                )

        image_mask = jnp.concatenate(
            [
                einops.repeat(preprocessed.image_masks[name], "b -> b s", s=tokens.shape[1])
                for name, tokens in zip(camera_names, image_embeddings, strict=True)
            ],
            axis=1,
        )
        image_tokens = jnp.concatenate(image_embeddings, axis=1)
        prompt = preprocessed.tokenized_prompt
        prompt_mask = preprocessed.tokenized_prompt_mask
        ar = (
            preprocessed.token_ar_mask
            if preprocessed.token_ar_mask is not None
            else jnp.zeros(prompt.shape, dtype=jnp.int32)
        )
        prefix_tokens = jnp.concatenate([image_tokens, self.PaliGemma.llm(prompt, method="embed")], axis=1)
        prefix_mask = jnp.concatenate([image_mask, prompt_mask], axis=1)
        prefix_ar = jnp.concatenate([0 * image_mask, ar], axis=1).astype(jnp.int32)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache, hidden = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=make_attn_mask(prefix_mask, prefix_ar),
            positions=prefix_positions,
            return_hidden_states=True,
        )
        h_t = hidden[0][self.memory_layer][:, :mem_len].astype(jnp.float32)
        retrieved = self.memory.read(memory_state, h_t)

        prefix_len = prefix_tokens.shape[1]
        padded_cache = jax.tree.map(
            lambda x: jnp.pad(
                x,
                ((0, 0), (0, 0), (0, mem_len + self.causal_token_len), (0, 0), (0, 0)),
            ),
            kv_cache,
        )
        memory_attn_mask = make_memory_step_mask(prefix_mask, prefix_ar, mem_len, self.causal_token_len)
        memory_positions = prefix_len + jnp.broadcast_to(jnp.arange(mem_len), (batch, mem_len))

        def run_memory_tokens(retrieved_content):
            memory_tokens = (self.memory_gate.value * retrieved_content).astype(prefix_tokens.dtype)
            (mem_out, _), _ = self.PaliGemma.llm(
                [memory_tokens, None],
                mask=memory_attn_mask,
                positions=memory_positions,
                kv_cache=padded_cache,
                cache_position=prefix_len,
            )
            # Module.__call__ returns `out`, after every transformer block and final_norms.
            return mem_out.astype(jnp.float32)

        final_ct = run_memory_tokens(retrieved)
        zero_read_final_ct = run_memory_tokens(jnp.zeros_like(retrieved)) if include_zero_read else None
        return final_ct, zero_read_final_ct, retrieved

    def final_ct_intervention_step(
        self,
        observation: _model.Observation,
        memory_state: _memory.MemoryState,
        *,
        allow_write: bool = False,
        top_camera_patch_embeddings: at.Float[at.Array, "b n d"] | None = None,
    ) -> tuple[_memory.MemoryState, dict[str, at.Array]]:
        """No-gradient final-c_t forward for patch-space or image-space interventions.

        With no override this is the same final-normalized v3.1 writer primal as inference.
        ``top_camera_patch_embeddings`` can replace the exact SigLIP outputs for the first
        camera.  To test an intervention at the transformed model-image boundary instead, put
        the modified image in ``observation.images['base_0_rgb']`` and omit the override; this
        runs the modified input through SigLIP normally.  Unlike
        :meth:`final_ct_attribution_step`, this method does not build a reverse pass or compute
        a zero-read counterfactual, making it suitable for batched occlusion sweeps.
        """
        assert self.predict_with_memory, "the model was not built with predict_with_memory"
        if self.memory_write_source != "post_attention":
            raise ValueError(
                "final_ct_intervention_step measures the actual writer only when memory_write_source='post_attention'."
            )
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        camera_names = tuple(preprocessed.images)
        if not camera_names or camera_names[0] != "base_0_rgb":
            raise ValueError(
                "final-c_t intervention requires base_0_rgb to be the first/top camera; "
                f"got camera order {camera_names}."
            )

        image_embeddings = []
        for index, name in enumerate(camera_names):
            if index == 0 and top_camera_patch_embeddings is not None:
                image_tokens = jnp.asarray(top_camera_patch_embeddings)
            else:
                image_tokens, _ = self.PaliGemma.img(preprocessed.images[name], train=False)
            image_embeddings.append(image_tokens)
        final_ct, _, retrieved = self._final_ct_from_image_embeddings(
            preprocessed, memory_state, tuple(image_embeddings), include_zero_read=False
        )

        writer_loss = self.memory.surprise(memory_state, final_ct)
        candidate_state, write_aux = self.memory.write(memory_state, final_ct)
        new_state = candidate_state if allow_write else memory_state
        batch = final_ct.shape[0]
        aux = {
            **write_aux,
            "final_ct": final_ct,
            "writer_loss": writer_loss,
            "final_ct_rms": jnp.sqrt(jnp.mean(jnp.square(final_ct), axis=(1, 2))),
            "retrieval_norm": jnp.sqrt(jnp.mean(jnp.square(retrieved.astype(jnp.float32)), axis=(1, 2))),
            "memory_gate_norm": jnp.broadcast_to(jnp.linalg.norm(self.memory_gate.value), (batch,)),
            "top_camera_tokens": jnp.asarray(final_ct.shape[1], dtype=jnp.int32),
            "write_occurred": jnp.full((batch,), allow_write, dtype=bool),
        }
        return new_state, aux

    def writer_echo_factorial_step(
        self,
        observation: _model.Observation,
        memory_state: _memory.MemoryState,
    ) -> tuple[_memory.MemoryState, _memory.MemoryState, dict[str, at.Array]]:
        """Return normal-read and zero-read v3.1 candidate writes from one pre-state.

        This offline diagnostic primitive evaluates the exact final-normalized v3.1 writer
        representation twice while sharing the complete current-observation prefix: once with
        the ordinary retrieved memory and once with the retrieved vectors replaced by zero.
        Both candidate states branch from ``memory_state`` and are returned without selecting or
        committing either branch. A caller can batch different observation/state pairings to
        form a side-effect-free O x M factorial experiment.
        """
        assert self.predict_with_memory, "the model was not built with predict_with_memory"
        if self.memory_write_source != "post_attention":
            raise ValueError("writer_echo_factorial_step is defined for the v3.1 final post_attention writer.")

        preprocessed = _model.preprocess_observation(None, observation, train=False)
        image_embeddings = tuple(
            self.PaliGemma.img(preprocessed.images[name], train=False)[0] for name in preprocessed.images
        )
        normal_ct, zero_read_ct, retrieved = self._final_ct_from_image_embeddings(
            preprocessed,
            memory_state,
            image_embeddings,
            include_zero_read=True,
        )
        assert zero_read_ct is not None
        normal_state, normal_aux = self.memory.write(memory_state, normal_ct)
        zero_state, zero_aux = self.memory.write(memory_state, zero_read_ct)
        aux = {
            "normal_ct": normal_ct,
            "zero_read_ct": zero_read_ct,
            "retrieval_norm": jnp.sqrt(jnp.mean(jnp.square(retrieved.astype(jnp.float32)), axis=(1, 2))),
            **{f"normal_{key}": value for key, value in normal_aux.items()},
            **{f"zero_read_{key}": value for key, value in zero_aux.items()},
        }
        return normal_state, zero_state, aux

    def memory_swap_read_step(
        self,
        observation: _model.Observation,
        memory_state: _memory.MemoryState,
    ) -> dict[str, at.Array]:
        """Return the exact v3.1 retrieved vectors and final c18 without writing.

        This is a read-only offline diagnostic boundary for matched-state interventions.  A
        caller may repeat one observation across different complete fast states and directly
        compare the retrieved tensor that enters the memory-token block, rather than treating
        its scalar norm as evidence of semantic content.
        """
        assert self.predict_with_memory, "the model was not built with predict_with_memory"
        if self.memory_write_source != "post_attention":
            raise ValueError("memory_swap_read_step is defined for the v3.1 final post_attention writer.")

        preprocessed = _model.preprocess_observation(None, observation, train=False)
        image_embeddings = tuple(
            self.PaliGemma.img(preprocessed.images[name], train=False)[0] for name in preprocessed.images
        )
        final_ct, _, retrieved = self._final_ct_from_image_embeddings(
            preprocessed,
            memory_state,
            image_embeddings,
            include_zero_read=False,
        )
        return {
            "retrieved": retrieved.astype(jnp.float32),
            "final_ct": final_ct,
        }

    def writer_echo_factorial_metrics_step(
        self,
        observation: _model.Observation,
        paired_state: _memory.MemoryState,
    ) -> tuple[_memory.MemoryState, dict[str, at.Array]]:
        """Reduce the paired left/right O x M writer experiment entirely on device.

        ``paired_state`` is ``[left_memory, right_memory]``. ``observation`` must have batch
        order ``[left_obs, right_obs, right_obs, left_obs]``. The corresponding memory batch is
        constructed as ``[left_memory, left_memory, right_memory, right_memory]`` so indices
        ``0,2`` are matched O,M and ``1,3`` are observation swaps. For every pairing this method
        evaluates normal and zero reads, reports vector factorial effects, and commits only the
        two matched normal branches for recurrent replay.
        """
        if observation.state.shape[0] != 4:
            raise ValueError("writer echo factorial observations must have batch size 4.")
        first_state_leaf = jax.tree.leaves(paired_state)[0]
        if first_state_leaf.shape[0] != 2:
            raise ValueError("writer echo factorial paired_state must have batch size 2.")

        branch_indices = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
        state4 = jax.tree.map(lambda value: value[branch_indices], paired_state)
        normal_state, zero_state, aux = self.writer_echo_factorial_step(observation, state4)

        def scale_batch(values, scale):
            return jax.tree.map(
                lambda value: value * scale.reshape(scale.shape + (1,) * (value.ndim - 1)),
                values,
            )

        def subtract(left, right):
            return jax.tree.map(lambda x, y: x - y, left, right)

        def add(left, right):
            return jax.tree.map(lambda x, y: x + y, left, right)

        normal_injection = subtract(
            normal_state.momentum,
            scale_batch(state4.momentum, aux["normal_eta"]),
        )
        zero_injection = subtract(
            zero_state.momentum,
            scale_batch(state4.momentum, aux["zero_read_eta"]),
        )
        normal_fast_update = subtract(normal_state.fast_weights, state4.fast_weights)
        zero_fast_update = subtract(zero_state.fast_weights, state4.fast_weights)
        normal_full_update = _memory.MemoryState(normal_fast_update, subtract(normal_state.momentum, state4.momentum))
        zero_full_update = _memory.MemoryState(zero_fast_update, subtract(zero_state.momentum, state4.momentum))

        def take(tree, index):
            return jax.tree.map(lambda value: value[index], tree)

        def tree_dot(left, right):
            return sum(jnp.vdot(x, y).real for x, y in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True))

        def tree_norm(tree):
            return jnp.sqrt(jnp.maximum(tree_dot(tree, tree), 0.0))

        def cosine(left, right):
            return tree_dot(left, right) / jnp.maximum(tree_norm(left) * tree_norm(right), 1e-12)

        metrics: dict[str, list[at.Array]] = {}

        def append_metric(name, value):
            metrics.setdefault(name, []).append(value)

        def reduce_family(name, normal_values, zero_values, original_index, swapped_index):
            a = take(normal_values, original_index)  # A(O, M)
            b = take(zero_values, original_index)  # A(O, 0)
            c = take(normal_values, swapped_index)  # A(O_swap, M)
            d = take(zero_values, swapped_index)  # A(O_swap, 0)
            memory_effect = subtract(a, b)
            observation_effect = subtract(a, c)
            observation_effect_zero = subtract(b, d)
            interaction = add(subtract(a, b), subtract(d, c))
            memory_main = scale_batch(add(subtract(a, b), subtract(c, d)), jnp.asarray(0.5))
            observation_main = scale_batch(add(subtract(a, c), subtract(b, d)), jnp.asarray(0.5))
            base_norm = tree_norm(a)
            memory_norm = tree_norm(memory_effect)
            observation_norm = tree_norm(observation_effect)
            interaction_norm = tree_norm(interaction)
            append_metric(f"{name}_base_norm", base_norm)
            append_metric(f"{name}_memory_effect_norm", memory_norm)
            append_metric(f"{name}_observation_effect_norm", observation_norm)
            append_metric(f"{name}_observation_effect_zero_read_norm", tree_norm(observation_effect_zero))
            append_metric(f"{name}_interaction_norm", interaction_norm)
            append_metric(f"{name}_memory_effect_relative", memory_norm / jnp.maximum(base_norm, 1e-12))
            append_metric(f"{name}_observation_effect_relative", observation_norm / jnp.maximum(base_norm, 1e-12))
            append_metric(f"{name}_interaction_relative", interaction_norm / jnp.maximum(base_norm, 1e-12))
            append_metric(
                f"{name}_memory_to_observation_main_ratio",
                tree_norm(memory_main) / jnp.maximum(tree_norm(observation_main), 1e-12),
            )
            append_metric(f"{name}_normal_vs_zero_cosine", cosine(a, b))
            append_metric(f"{name}_normal_vs_swapped_observation_cosine", cosine(a, c))

        for original_index, swapped_index in ((0, 1), (2, 3)):
            reduce_family("final_ct", aux["normal_ct"], aux["zero_read_ct"], original_index, swapped_index)
            reduce_family("injection", normal_injection, zero_injection, original_index, swapped_index)
            reduce_family("fast_update", normal_fast_update, zero_fast_update, original_index, swapped_index)
            reduce_family("full_update", normal_full_update, zero_full_update, original_index, swapped_index)
            for key in (
                "retrieval_norm",
                "normal_surprise",
                "zero_read_surprise",
                "normal_grad_norm",
                "zero_read_grad_norm",
            ):
                append_metric(key, aux[key][original_index])

        committed_indices = jnp.asarray([0, 2], dtype=jnp.int32)
        committed_state = jax.tree.map(lambda value: value[committed_indices], normal_state)
        return committed_state, {name: jnp.stack(values) for name, values in metrics.items()}

    def final_ct_attribution_step(
        self,
        observation: _model.Observation,
        memory_state: _memory.MemoryState,
        *,
        allow_write: bool = False,
        top_camera_patch_embeddings: at.Float[at.Array, "b n d"] | None = None,
    ) -> tuple[_memory.MemoryState, dict[str, at.Array]]:
        """Attribute the real v3.1 writer objective to top-camera SigLIP tokens.

        The differentiated inputs are the *outputs* of SigLIP for the first camera
        (``base_0_rgb``), not pixels or an internal SigLIP layer.  Consequently the returned
        patch map includes both Gemma passes -- the prefix pass that forms ``h_t`` and the
        incremental memory-token pass -- including all transformer blocks and Gemma's final
        RMSNorm, while deliberately excluding attribution through the image encoder itself.

        ``final_ct`` is the exact normalized ``mem_out`` used by v3.1 inference.  The scalar
        objective for sample ``b`` is the pre-write associative loss

            memory.surprise(memory_state, final_ct)[b].

        A single reverse-mode pass differentiates the sum of these independent per-sample
        losses with respect to ``[B, N, D]`` top-camera patch embeddings.  Because neither the
        transformer nor the memory mixes the batch dimension, the resulting ``[B, N]`` L2 map
        is exactly the per-sample VJP, without constructing a prohibitively large Jacobian.

        The zero-read counterfactual preserves the complete prefix, cache layout, memory-token
        positions, content gate, and final norm; only the retrieved vectors are replaced with
        zeros.  ``allow_write=False`` (the diagnostic default) computes all writer statistics
        but returns ``memory_state`` byte-for-byte unchanged.  Supplying
        ``top_camera_patch_embeddings`` is an optional patch-space intervention hook; omitted,
        the embeddings are produced once by the model's normal SigLIP inference call.

        This diagnostic is intentionally restricted to ``post_attention`` models.  For a v3
        ``raw_hidden`` model, ``final_ct`` is not the configured writer input, so labelling its
        surprise as the actual writer objective would be misleading.
        """
        assert self.predict_with_memory, "the model was not built with predict_with_memory"
        if self.memory_write_source != "post_attention":
            raise ValueError(
                "final_ct_attribution_step measures the actual writer only when memory_write_source='post_attention'."
            )

        preprocessed = _model.preprocess_observation(None, observation, train=False)
        batch = preprocessed.state.shape[0]
        camera_names = tuple(preprocessed.images)
        if not camera_names or camera_names[0] != "base_0_rgb":
            raise ValueError(
                "final-c_t attribution requires base_0_rgb to be the first/top camera; "
                f"got camera order {camera_names}."
            )

        # SigLIP is intentionally outside value_and_grad: the attribution boundary is its
        # exact output patch tensor.  Holding that tensor in float32 gives a stable gradient
        # norm; Gemma immediately casts it to its configured embed dtype, exactly as inference
        # does, so the primal final_c_t is numerically unchanged.
        image_embeddings = []
        inferred_top_embeddings = None
        for index, name in enumerate(camera_names):
            if index == 0 and top_camera_patch_embeddings is not None:
                image_tokens = jnp.asarray(top_camera_patch_embeddings)
            else:
                image_tokens, _ = self.PaliGemma.img(preprocessed.images[name], train=False)
            if index == 0:
                inferred_top_embeddings = image_tokens
            image_embeddings.append(image_tokens)

        assert inferred_top_embeddings is not None
        mem_len = inferred_top_embeddings.shape[1]
        top_embeddings_f32 = inferred_top_embeddings.astype(jnp.float32)
        other_image_embeddings = tuple(image_embeddings[1:])

        def objective(top_embeddings):
            # This is the same two-call inference path as sample_with_memory.  In particular,
            # mem_out is Module.__call__'s `out`, after all blocks and `final_norms`; it is not
            # hidden_states[-1], which is the last block output *before* final RMSNorm.
            all_image_embeddings = (top_embeddings, *other_image_embeddings)
            normal_ct, zero_read_ct, retrieved = self._final_ct_from_image_embeddings(
                preprocessed,
                memory_state,
                all_image_embeddings,
                include_zero_read=True,
            )
            assert zero_read_ct is not None
            writer_loss = self.memory.surprise(memory_state, normal_ct)
            return jnp.sum(writer_loss), (writer_loss, normal_ct, zero_read_ct, retrieved)

        (_, (writer_loss, final_ct, zero_read_ct, retrieved)), top_patch_grad = jax.value_and_grad(
            objective, has_aux=True
        )(top_embeddings_f32)

        # The write happens outside the differentiated objective.  It is the regular v3.1
        # update against the same pre-write state and final_ct; `surprise` in write_aux is thus
        # the same objective returned above.  No state leaf is ever mutated in place.
        candidate_state, write_aux = self.memory.write(memory_state, final_ct)
        new_state = candidate_state if allow_write else memory_state
        top_patch_grad = top_patch_grad.astype(jnp.float32)
        zero_read_delta = final_ct - zero_read_ct
        aux = {
            **write_aux,
            "final_ct": final_ct,
            "writer_loss": writer_loss,
            "writer_loss_top_patch_grad_norm": jnp.linalg.norm(top_patch_grad, axis=-1),
            "final_ct_zero_read_l2": jnp.linalg.norm(zero_read_delta, axis=-1),
            "writer_loss_top_patch_grad_global_norm": jnp.linalg.norm(top_patch_grad, axis=(1, 2)),
            "top_camera_patch_embedding_rms": jnp.sqrt(jnp.mean(jnp.square(top_embeddings_f32), axis=(1, 2))),
            "final_ct_rms": jnp.sqrt(jnp.mean(jnp.square(final_ct), axis=(1, 2))),
            "zero_read_final_ct_rms": jnp.sqrt(jnp.mean(jnp.square(zero_read_ct), axis=(1, 2))),
            "retrieval_norm": jnp.sqrt(jnp.mean(jnp.square(retrieved.astype(jnp.float32)), axis=(1, 2))),
            "memory_gate_norm": jnp.broadcast_to(jnp.linalg.norm(self.memory_gate.value), (batch,)),
            "top_camera_tokens": jnp.asarray(mem_len, dtype=jnp.int32),
            "write_occurred": jnp.full((batch,), allow_write, dtype=bool),
        }
        return new_state, aux

    @contextlib.contextmanager
    def capture_attention(self):
        """Temporarily make the LLM return per-layer attention distributions.

        The flag is a linen dataclass attribute rather than a parameter, so toggling it changes
        no weights and no numerics; it only keeps an extra scan output alive. It is restored on
        exit so a diagnostic can never leave the model in a slower state.
        """

        module = self.PaliGemma.llm.module
        previous = module.return_attn_probs
        object.__setattr__(module, "return_attn_probs", True)
        try:
            yield
        finally:
            object.__setattr__(module, "return_attn_probs", previous)

    def memory_attention_maps(
        self,
        observation: _model.Observation,
        memory_state: _memory.MemoryState,
        *,
        layer: int | None = None,
        head: int | None = None,
        forced_subtask_tokens: at.Int[at.Array, "b cl"] | None = None,
        forced_subtask_mask: at.Bool[at.Array, "b cl"] | None = None,
    ) -> dict[str, at.Array]:
        """Attention from the memory-token and subtask-token rows onto the image patches.

        This is a read-only diagnostic: it performs the same prefill and memory-token extension
        as inference, but denoises no actions, commits no write, and returns the requested
        layer's head-averaged attention distributions.

        Two query groups are returned, matching the two questions the v3.1 experiments ask:

        * ``memory_to_top`` -- for each of the 256 memory-token rows, how much it attends to each
          of the 256 top-camera patch keys. This is what forms ``c_t``, so it shows which image
          regions the *write* is built from.
        * ``subtask_to_top`` / ``subtask_to_memory`` -- for each teacher-forced subtask token,
          its attention onto the top-camera patches and onto the memory block. The latter is the
          share of attention the *decision* spends on retrieved memory rather than current
          vision, which no image-space map alone can establish.

        Rows are full softmax distributions over ALL keys, so the returned per-group maps do not
        each sum to one; ``*_mass`` entries report how much of each row's total attention the
        group captures, which is required to compare groups honestly.

        ``head`` selects a single attention head; the default averages all of them. The average
        is the honest summary only when heads behave alike -- a single head that routes strongly
        to vision or memory is diluted by sink heads that dominate the mean, so a per-head sweep
        is what distinguishes "the model barely looks at the image" from "one head does".
        """

        assert self.predict_with_memory, "the model was not built with predict_with_memory"
        if (forced_subtask_tokens is None) != (forced_subtask_mask is None):
            raise ValueError("forced_subtask_tokens and forced_subtask_mask must be provided together.")
        target_layer = self.memory_layer if layer is None else layer
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        batch = preprocessed.state.shape[0]

        img_tokens = []
        img_masks = []
        for name in preprocessed.images:
            image_tokens, _ = self.PaliGemma.img(preprocessed.images[name], train=False)
            img_tokens.append(image_tokens)
            img_masks.append(einops.repeat(preprocessed.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
        img_tokens = jnp.concatenate(img_tokens, axis=1)
        img_mask = jnp.concatenate(img_masks, axis=1)
        prompt = preprocessed.tokenized_prompt
        prompt_mask = preprocessed.tokenized_prompt_mask
        ar = (
            preprocessed.token_ar_mask
            if preprocessed.token_ar_mask is not None
            else jnp.zeros(prompt.shape, dtype=jnp.int32)
        )
        num_img = img_mask.shape[1]
        prefix_len = num_img + prompt.shape[1]
        mem_len = num_img // len(preprocessed.images)
        causal_len = self.causal_token_len
        if not 0 <= target_layer < self.PaliGemma.llm.module.configs[0].depth:
            raise ValueError(f"layer {target_layer} is outside the model's depth")

        prefix_tokens = jnp.concatenate([img_tokens, self.PaliGemma.llm(prompt, method="embed")], axis=1)
        prefix_mask = jnp.concatenate([img_mask, prompt_mask], axis=1)
        prefix_ar = jnp.concatenate([0 * img_mask, ar], axis=1).astype(jnp.int32)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        # With capture enabled every hidden-state call also yields attention, so unpack the
        # prefill positionally: the prefix's own self-attention is not one of the requested maps.
        prefill = self.PaliGemma.llm(
            [prefix_tokens, None], mask=attn_mask, positions=positions, return_hidden_states=True
        )
        kv_cache, hidden = prefill[1], prefill[2]
        h_t = hidden[0][self.memory_layer][:, :mem_len].astype(jnp.float32)

        mem_tokens = (self.memory_gate.value * self.memory.read(memory_state, h_t)).astype(prefix_tokens.dtype)
        kv_cache = jax.tree.map(
            lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, mem_len + causal_len), (0, 0), (0, 0))), kv_cache
        )
        mem_mask = make_memory_step_mask(prefix_mask, prefix_ar, mem_len, causal_len)
        mem_positions = prefix_len + jnp.broadcast_to(jnp.arange(mem_len), (batch, mem_len))
        memory_pass = self.PaliGemma.llm(
            [mem_tokens, None],
            mask=mem_mask,
            positions=mem_positions,
            kv_cache=kv_cache,
            cache_position=prefix_len,
            return_hidden_states=True,
        )
        if len(memory_pass) != 4:
            raise RuntimeError("memory_attention_maps requires the capture_attention() context")
        kv_cache, mem_attn = memory_pass[1], memory_pass[3]

        # The top camera occupies the first mem_len image-token slots; memory keys sit directly
        # after the whole prefix. Slicing by these offsets keeps the map aligned with the exact
        # 16x16 SigLIP patch grid the renderer expects.
        num_heads = mem_attn.shape[2]
        if head is not None and not 0 <= head < num_heads:
            raise ValueError(f"head {head} is outside the layer's {num_heads} heads")
        memory_rows = mem_attn[target_layer]
        memory_rows = jnp.mean(memory_rows, axis=1) if head is None else memory_rows[:, head]
        # Every row is one softmax over ALL keys, so the blocks below partition its total mass.
        # Camera boundaries follow the order Observation.images was built in, so the split is
        # exact rather than assumed; the residual named below closes the budget to 1.
        camera_names = list(preprocessed.images)
        result = {
            "memory_to_top": memory_rows[:, :, :mem_len],
            "memory_to_memory": memory_rows[:, :, prefix_len : prefix_len + mem_len],
            "memory_to_top_mass": jnp.sum(memory_rows[:, :, :mem_len], axis=-1),
            "memory_to_prefix_mass": jnp.sum(memory_rows[:, :, :prefix_len], axis=-1),
            "memory_to_memory_mass": jnp.sum(memory_rows[:, :, prefix_len : prefix_len + mem_len], axis=-1),
            "memory_to_images_mass": jnp.sum(memory_rows[:, :, :num_img], axis=-1),
            "memory_to_prompt_mass": jnp.sum(memory_rows[:, :, num_img:prefix_len], axis=-1),
            "layer": jnp.asarray(target_layer),
            "top_camera_tokens": jnp.asarray(mem_len),
            "prefix_len": jnp.asarray(prefix_len),
            "num_image_tokens": jnp.asarray(num_img),
            "num_heads": jnp.asarray(num_heads),
            "head": jnp.asarray(-1 if head is None else head),
        }
        for index, name in enumerate(camera_names):
            start = index * mem_len
            result[f"memory_to_camera_{name}_mass"] = jnp.sum(memory_rows[:, :, start : start + mem_len], axis=-1)

        if forced_subtask_tokens is not None:
            gen_tokens = forced_subtask_tokens.astype(prompt.dtype)
            gen_mask = forced_subtask_mask.astype(bool)
            causal_emb = self.PaliGemma.llm(gen_tokens, method="embed")
            causal_rows = jnp.concatenate(
                [
                    einops.repeat(prefix_mask, "b p -> b c p", c=causal_len),
                    jnp.ones((batch, causal_len, mem_len), dtype=bool),
                    jnp.tril(jnp.ones((causal_len, causal_len), dtype=bool))[None] & gen_mask[:, None, :],
                ],
                axis=-1,
            )
            causal_positions = jnp.broadcast_to(
                prefix_len + mem_len + jnp.arange(causal_len)[None], (batch, causal_len)
            )
            causal_pass = self.PaliGemma.llm(
                [causal_emb, None],
                mask=causal_rows,
                positions=causal_positions,
                kv_cache=kv_cache,
                cache_position=prefix_len + mem_len,
                return_hidden_states=True,
            )
            subtask_rows = causal_pass[3][target_layer]
            subtask_rows = jnp.mean(subtask_rows, axis=1) if head is None else subtask_rows[:, head]
            result.update(
                {
                    "subtask_to_top": subtask_rows[:, :, :mem_len],
                    "subtask_to_top_mass": jnp.sum(subtask_rows[:, :, :mem_len], axis=-1),
                    "subtask_to_memory": subtask_rows[:, :, prefix_len : prefix_len + mem_len],
                    "subtask_to_memory_mass": jnp.sum(subtask_rows[:, :, prefix_len : prefix_len + mem_len], axis=-1),
                    "subtask_to_prefix_mass": jnp.sum(subtask_rows[:, :, :prefix_len], axis=-1),
                    "subtask_to_images_mass": jnp.sum(subtask_rows[:, :, :num_img], axis=-1),
                    "subtask_to_prompt_mass": jnp.sum(subtask_rows[:, :, num_img:prefix_len], axis=-1),
                    # Everything after the memory block: the subtask tokens attending to
                    # themselves and their causal predecessors.
                    "subtask_to_causal_mass": jnp.sum(subtask_rows[:, :, prefix_len + mem_len :], axis=-1),
                    "subtask_token_mask": gen_mask,
                }
            )
            for index, name in enumerate(camera_names):
                start = index * mem_len
                result[f"subtask_to_camera_{name}_mass"] = jnp.sum(subtask_rows[:, :, start : start + mem_len], axis=-1)
        return result

    def v35_action_memory_attention_step(
        self,
        observation: _model.Observation,
        memory_state: _memory.MemoryState,
        *,
        action_noise: at.Float[at.Array, "b ah ad"],
        forced_subtask_tokens: at.Int[at.Array, "b cl"],
        forced_subtask_mask: at.Bool[at.Array, "b cl"],
        zero_read: bool = False,
        v35_oracle_direction: at.Float[at.Array, "b d"] | None = None,
        v35_oracle_injected_rms: float | at.Float[at.Array, " b"] | None = None,
        layer: int | None = None,
        head: int | None = None,
    ) -> dict[str, at.Array]:
        """Measure the actual action-expert attention paid to v3.5 memory tokens.

        This is a read-only, one-flow-evaluation diagnostic.  It executes the production
        v3.5 split prefix (including calibrated ``tanh_rms`` injection or the shared oracle
        injection path), teacher-forces the frozen per-frame subtask, and evaluates the action
        suffix at the first denoising point (``time=1``, so ``x_t=action_noise``).  No action is
        integrated and no memory transition is applied.

        The returned mass is averaged over heads and the ``action_horizon`` action-token rows
        at the selected late transformer layer.  ``uniform_baseline`` is derived from the exact
        layer-wise visibility mask for those same rows, not from a hard-coded token count.
        Callers must enter :meth:`capture_attention`; without it this method fails closed.
        """

        if not getattr(self, "memory_v35_enabled", False):
            raise ValueError("v35_action_memory_attention_step requires a v3.5 model.")
        if self.memory_architecture != "v32_layer8_dual_query":
            raise ValueError("v3.5 action-memory attention requires the layer-8 dual-query architecture.")
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        batch = preprocessed.state.shape[0]
        expected_noise = (batch, self.action_horizon, self.action_dim)
        if action_noise.shape != expected_noise:
            raise ValueError(f"action_noise must have shape {expected_noise}; got {action_noise.shape}.")
        expected_causal = (batch, self.causal_token_len)
        if forced_subtask_tokens.shape != expected_causal or forced_subtask_mask.shape != expected_causal:
            raise ValueError(
                "forced subtask buffers must have shape "
                f"{expected_causal}; got {forced_subtask_tokens.shape} and {forced_subtask_mask.shape}."
            )

        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(preprocessed)
        prefix_len = prefix_mask.shape[1]
        num_img = prefix_len - self.max_token_len
        mem_len = self.memory_query_tokens
        prepared = self._v32_prepare_memory_prefix(
            prefix_tokens,
            prefix_mask,
            prefix_ar,
            memory_state,
            top_token_count=num_img // len(preprocessed.images),
            zero_read=zero_read,
            state_token_mask=preprocessed.token_state_mask,
            v35_oracle_direction=v35_oracle_direction,
            v35_oracle_injected_rms=v35_oracle_injected_rms,
        )
        cache = prepared["cache"]
        causal_tokens = forced_subtask_tokens.astype(preprocessed.tokenized_prompt.dtype)
        causal_mask = forced_subtask_mask.astype(bool)
        causal_emb = self.PaliGemma.llm(causal_tokens, method="embed")
        causal_positions = prefix_len + mem_len + jnp.broadcast_to(
            jnp.arange(self.causal_token_len), expected_causal
        )
        (_, _), cache = self.PaliGemma.llm(
            [causal_emb, None],
            mask=self._v32_causal_mask(prefix_mask, causal_mask),
            positions=causal_positions,
            kv_cache=cache,
            cache_position=prefix_len + mem_len,
        )

        flow_time = jnp.ones((batch,), dtype=jnp.float32)
        suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(
            preprocessed, action_noise, flow_time
        )
        suffix_attention_mask = self._v32_suffix_mask(prefix_mask, causal_mask, suffix_mask, suffix_ar)
        suffix_positions = (
            prefix_len
            + mem_len
            + self.causal_token_len
            + jnp.cumsum(suffix_mask, axis=-1)
            - 1
        )
        suffix_pass = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=suffix_attention_mask,
            positions=suffix_positions,
            kv_cache=cache,
            adarms_cond=[None, adarms_cond],
            return_hidden_states=True,
        )
        if len(suffix_pass) != 4:
            raise RuntimeError("v35_action_memory_attention_step requires the capture_attention() context")
        attention = suffix_pass[3]
        depth = attention.shape[0]
        target_layer = depth - 1 if layer is None else layer
        if not self.memory_layer < target_layer < depth:
            raise ValueError(
                f"action-memory attention layer must be a late layer in ({self.memory_layer}, {depth}); "
                f"got {target_layer}."
            )
        num_heads = attention.shape[2]
        if head is not None and not 0 <= head < num_heads:
            raise ValueError(f"head {head} is outside the layer's {num_heads} heads")

        rows = attention[target_layer, :, :, -self.action_horizon :, :]
        rows = jnp.mean(rows, axis=1) if head is None else rows[:, head]
        memory_slice = slice(prefix_len, prefix_len + mem_len)
        per_action_mass = jnp.sum(rows[..., memory_slice], axis=-1)

        visible = suffix_attention_mask[target_layer, :, -self.action_horizon :, :]
        visible_count = jnp.sum(visible, axis=-1)
        memory_visible = jnp.sum(visible[..., memory_slice], axis=-1)
        uniform_per_action = memory_visible / jnp.maximum(visible_count, 1)
        return {
            "action_to_memory_mass": jnp.mean(per_action_mass, axis=-1),
            "action_to_memory_mass_per_action": per_action_mass,
            "uniform_baseline": jnp.mean(uniform_per_action, axis=-1),
            "uniform_baseline_per_action": uniform_per_action,
            "layer": jnp.asarray(target_layer, dtype=jnp.int32),
            "head": jnp.asarray(-1 if head is None else head, dtype=jnp.int32),
            "num_heads": jnp.asarray(num_heads, dtype=jnp.int32),
            "memory_token_count": jnp.asarray(mem_len, dtype=jnp.int32),
            "v35_injected_pre_cast_rms": prepared["injected_pre_cast_rms"],
            "v35_injected_post_cast_rms": prepared["injected_post_cast_rms"],
        }

    def v35_paired_task_health_step(
        self,
        observation: _model.Observation,
        actions: _model.Actions,
        memory_state: _memory.MemoryState,
        *,
        action_noise: at.Float[at.Array, "b ah ad"],
        flow_time: at.Float[at.Array, " b"],
    ) -> dict[str, at.Array]:
        """Evaluate paired source-path and enabled-v3.5 losses with frozen randomness.

        Both branches use the same model tree, preprocessed observation, causal labels,
        ground-truth actions, action noise, and flow timestep.  The source branch runs the
        ordinary prefix followed directly by causal/action tokens; the enabled branch inserts
        the production v3.5 memory interface and reads ``memory_state``.  Neither branch writes
        memory, samples augmentation, draws a random number, or mutates parameters.

        This diagnostic intentionally measures the no-RTC-delay flow objective.  The rung
        protocol freezes that choice together with the frame/noise/time suite so every
        checkpoint is paired against the identical reference.
        """

        if not getattr(self, "memory_v35_enabled", False):
            raise ValueError("v35_paired_task_health_step requires a v3.5 model.")
        if self.memory_architecture != "v32_layer8_dual_query":
            raise ValueError("v3.5 paired task health requires the layer-8 dual-query architecture.")
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        batch = preprocessed.state.shape[0]
        expected_actions = (batch, self.action_horizon, self.action_dim)
        if actions.shape != expected_actions or action_noise.shape != expected_actions:
            raise ValueError(
                f"actions and action_noise must both have shape {expected_actions}; "
                f"got {actions.shape} and {action_noise.shape}."
            )
        if flow_time.shape != (batch,):
            raise ValueError(f"flow_time must have shape ({batch},); got {flow_time.shape}.")
        if preprocessed.tokenized_causal is None or preprocessed.tokenized_causal_mask is None:
            raise ValueError("paired task health requires teacher-forced tokenized_causal labels.")
        causal_tokens = preprocessed.tokenized_causal
        causal_mask = preprocessed.tokenized_causal_mask.astype(bool)
        if causal_tokens.shape != (batch, self.causal_token_len) or causal_mask.shape != causal_tokens.shape:
            raise ValueError("paired task-health causal buffers have the wrong shape.")
        causal_fast = (
            jnp.zeros_like(causal_mask)
            if preprocessed.causal_fast_mask is None
            else preprocessed.causal_fast_mask.astype(bool)
        )

        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(preprocessed)
        prefix_len = prefix_mask.shape[1]
        causal_len = self.causal_token_len
        causal_emb = self.PaliGemma.llm(causal_tokens, method="embed")
        x_t = flow_time[:, None, None] * action_noise + (1.0 - flow_time[:, None, None]) * actions
        target_velocity = action_noise - actions

        def losses_from_cache(
            *,
            cache: _gemma.KVCache,
            final_prefix: at.Array,
            causal_attention_mask: at.Array,
            causal_offset: int,
            suffix_attention_mask: at.Array,
            suffix_offset: int,
        ) -> tuple[at.Array, at.Array]:
            causal_positions = causal_offset + jnp.broadcast_to(jnp.arange(causal_len), causal_tokens.shape)
            (causal_out, _), causal_cache = self.PaliGemma.llm(
                [causal_emb, None],
                mask=causal_attention_mask,
                positions=causal_positions,
                kv_cache=cache,
                cache_position=causal_offset,
            )
            ce_hidden = jnp.concatenate(
                [self._v32_causal_seed(final_prefix, prefix_mask), causal_out[:, :-1]], axis=1
            )
            logits = self.PaliGemma.llm(ce_hidden, method="decode").astype(jnp.float32)
            token_logp = jnp.take_along_axis(
                jax.nn.log_softmax(logits, axis=-1), causal_tokens[..., None], axis=-1
            )[..., 0]
            ce = -jnp.sum(token_logp * causal_mask, axis=-1) / jnp.maximum(jnp.sum(causal_mask, axis=-1), 1)

            suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(
                preprocessed, x_t, flow_time
            )
            suffix_positions = suffix_offset + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=suffix_attention_mask(suffix_mask, suffix_ar),
                positions=suffix_positions,
                kv_cache=jax.lax.stop_gradient(causal_cache),
                adarms_cond=[None, adarms_cond],
            )
            velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :]).astype(jnp.float32)
            flow = jnp.mean(jnp.square(velocity - target_velocity.astype(jnp.float32)), axis=(1, 2))
            return flow, ce

        # Source path: all layers process the ordinary prefix; there is no memory-token block.
        source_capacity = prefix_len + causal_len
        source_cache = self._v32_empty_cache(batch, source_capacity, prefix_tokens.dtype)
        source_prefix_mask = self._pad_attention_columns(make_attn_mask(prefix_mask, prefix_ar), source_capacity)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (source_prefix, _), source_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=source_prefix_mask,
            positions=prefix_positions,
            kv_cache=source_cache,
            cache_position=0,
        )
        source_causal_self = jnp.tril(jnp.ones((causal_len, causal_len), dtype=bool))[None]
        source_causal_mask = jnp.concatenate(
            [
                einops.repeat(prefix_mask, "b p -> b c p", c=causal_len),
                source_causal_self & causal_mask[:, None, :],
            ],
            axis=-1,
        )

        def source_suffix_mask(suffix_mask, suffix_ar):
            visible = jnp.concatenate([prefix_mask, causal_mask & ~causal_fast], axis=1)
            return jnp.concatenate(
                [einops.repeat(visible, "b p -> b s p", s=suffix_mask.shape[1]), make_attn_mask(suffix_mask, suffix_ar)],
                axis=-1,
            )

        source_flow, source_ce = losses_from_cache(
            cache=source_cache,
            final_prefix=source_prefix,
            causal_attention_mask=source_causal_mask,
            causal_offset=prefix_len,
            suffix_attention_mask=source_suffix_mask,
            suffix_offset=prefix_len + causal_len,
        )

        # Enabled path: the exact calibrated split-prefix/memory-token interface used by
        # sequence training and deployment, but no post-read transition.
        top_tokens = (prefix_len - self.max_token_len) // len(preprocessed.images)
        prepared = self._v32_prepare_memory_prefix(
            prefix_tokens,
            prefix_mask,
            prefix_ar,
            memory_state,
            top_token_count=top_tokens,
            state_token_mask=preprocessed.token_state_mask,
        )
        mem_len = self.memory_query_tokens

        def memory_suffix_mask(suffix_mask, suffix_ar):
            return self._v32_suffix_mask(prefix_mask, causal_mask & ~causal_fast, suffix_mask, suffix_ar)

        memory_flow, memory_ce = losses_from_cache(
            cache=prepared["cache"],
            final_prefix=prepared["final_prefix"],
            causal_attention_mask=self._v32_causal_mask(prefix_mask, causal_mask),
            causal_offset=prefix_len + mem_len,
            suffix_attention_mask=memory_suffix_mask,
            suffix_offset=prefix_len + mem_len + causal_len,
        )
        return {
            "source_flow_loss": source_flow,
            "source_subtask_ce": source_ce,
            "memory_flow_loss": memory_flow,
            "memory_subtask_ce": memory_ce,
            "v35_injected_pre_cast_rms": prepared["injected_pre_cast_rms"],
            "v35_injected_post_cast_rms": prepared["injected_post_cast_rms"],
        }

    def prompt_attention_maps(
        self,
        observation: _model.Observation,
        *,
        layer: int | None = None,
        head: int | None = None,
    ) -> dict[str, at.Array]:
        """Attention from every tokenized-prompt row onto the image patches.

        Read-only diagnostic over the plain image+prompt prefix: no memory tokens are
        inserted, nothing is denoised, nothing is written. With :meth:`capture_attention`
        active it returns, for the requested gemma block, each prompt-token row's
        head-averaged attention over the 256 top-camera patch keys plus mass partitions
        over every camera and the text block. Under the v3.2 split architecture the
        prefix pass is bit-identical to inference for blocks ``0..memory_layer`` (memory
        tokens only join after that block), so the default ``layer=memory_layer`` shows
        exactly the attention that shaped the hidden states the memory reads and writes;
        larger layers describe a memory-less forward instead.

        Each row of ``prompt_to_top`` is one softmax over ALL prefix keys, so it does not
        sum to one; the ``*_mass`` entries close the budget. Which rows belong to the
        instruction words versus the discretized state digits is a tokenizer question,
        so callers must slice rows using the tokenizer-provided masks.

        ``layer=-1`` returns the whole depth at once -- every map/mass entry gains a
        leading per-layer axis after batch ([b, depth, ...]). ``head=-1`` does the same
        for the heads of ONE layer ([b, heads, ...]); combining both sweeps is rejected
        to keep the result rank fixed. The capture already materializes everything, so
        neither sweep costs extra forward passes.
        """
        target_layer = self.memory_layer if layer is None else layer
        depth = self.PaliGemma.llm.module.configs[0].depth
        if target_layer != -1 and not 0 <= target_layer < depth:
            raise ValueError(f"layer {target_layer} is outside the model's depth {depth}")
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(preprocessed)
        num_img = prefix_mask.shape[1] - self.max_token_len
        mem_len = num_img // len(preprocessed.images)
        prefix_len = prefix_mask.shape[1]
        attn_mask = make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefill = self.PaliGemma.llm(
            [prefix_tokens, None], mask=attn_mask, positions=positions, return_hidden_states=True
        )
        if len(prefill) != 4:
            raise RuntimeError("prompt_attention_maps requires the capture_attention() context")
        probs = prefill[3]  # [depth, b, heads, T, S]
        num_heads = probs.shape[2]
        if head is not None and head != -1 and not 0 <= head < num_heads:
            raise ValueError(f"head {head} is outside the layer's {num_heads} heads")
        if head == -1 and target_layer == -1:
            raise ValueError("layer=-1 and head=-1 cannot be combined; sweep one axis at a time")
        stack = probs if target_layer == -1 else probs[target_layer][None]  # [groups, b, heads, T, S]
        if head is None:
            grouped = jnp.mean(stack, axis=2)
        elif head == -1:
            grouped = jnp.moveaxis(stack[0], 1, 0)  # heads become the group axis
        else:
            grouped = stack[:, :, head]
        rows = jnp.moveaxis(grouped, 0, 1)  # [b, groups, T, S]
        if target_layer != -1 and head != -1:
            rows = rows[:, 0]
        prompt_rows = rows[..., num_img:prefix_len, :].astype(jnp.float32)
        camera_names = list(preprocessed.images)
        result = {
            "prompt_to_top": prompt_rows[..., :mem_len],
            "prompt_to_images_mass": jnp.sum(prompt_rows[..., :num_img], axis=-1),
            "prompt_to_text_mass": jnp.sum(prompt_rows[..., num_img:prefix_len], axis=-1),
            "prompt_token_mask": prefix_mask[:, num_img:prefix_len],
            "layer": jnp.asarray(target_layer),
            "depth": jnp.asarray(depth),
            "top_camera_tokens": jnp.asarray(mem_len),
            "num_image_tokens": jnp.asarray(num_img),
            "num_heads": jnp.asarray(num_heads),
            "head": jnp.asarray(-1 if head is None else head),
        }
        for index, name in enumerate(camera_names):
            start = index * mem_len
            result[f"prompt_to_camera_{name}_mass"] = jnp.sum(prompt_rows[..., start : start + mem_len], axis=-1)
        return result

    def retrieved_token_ablation_step(
        self,
        observation: _model.Observation,
        memory_state: _memory.MemoryState,
        token_indices: at.Array,
    ) -> dict[str, at.Array]:
        """Causally ablate retrieved-token slots and measure the v3.1 writer change.

        The caller supplies a fixed-size vector of token indices. Index ``-1`` is an exact
        unmodified control; non-negative index ``j`` zeros retrieved row ``j`` before the
        memory gate. The observation prefix and ordinary retrieval are computed once at batch
        size one, then the memory-token pass and associative write are evaluated for every
        intervention from the identical pre-write state. No branch is committed.

        Returned effects are relative to branch zero, which must be the ``-1`` control. This
        same-batch baseline removes batch-shape/BF16 kernel floors while avoiding transfer of
        the multi-million-parameter candidate states to the host.
        """
        assert self.predict_with_memory, "the model was not built with predict_with_memory"
        if self.memory_write_source != "post_attention":
            raise ValueError("retrieved-token ablation is defined for the v3.1 post_attention writer.")
        if observation.state.shape[0] != 1:
            raise ValueError("retrieved-token ablation requires one fixed observation.")
        first_state_leaf = jax.tree.leaves(memory_state)[0]
        if first_state_leaf.shape[0] != 1:
            raise ValueError("retrieved-token ablation requires one fixed pre-write state.")
        token_indices = jnp.asarray(token_indices, dtype=jnp.int32)
        if token_indices.ndim != 1 or token_indices.shape[0] < 2:
            raise ValueError("token_indices must be a rank-1 array containing a control and interventions.")

        preprocessed = _model.preprocess_observation(None, observation, train=False)
        batch = token_indices.shape[0]
        image_embeddings = []
        image_masks = []
        for name in preprocessed.images:
            image_tokens, _ = self.PaliGemma.img(preprocessed.images[name], train=False)
            image_embeddings.append(image_tokens)
            image_masks.append(einops.repeat(preprocessed.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
        mem_len = image_embeddings[0].shape[1]
        image_tokens = jnp.concatenate(image_embeddings, axis=1)
        image_mask = jnp.concatenate(image_masks, axis=1)
        prompt = preprocessed.tokenized_prompt
        prompt_mask = preprocessed.tokenized_prompt_mask
        ar = (
            preprocessed.token_ar_mask
            if preprocessed.token_ar_mask is not None
            else jnp.zeros(prompt.shape, dtype=jnp.int32)
        )
        prefix_tokens = jnp.concatenate([image_tokens, self.PaliGemma.llm(prompt, method="embed")], axis=1)
        prefix_mask = jnp.concatenate([image_mask, prompt_mask], axis=1)
        prefix_ar = jnp.concatenate([0 * image_mask, ar], axis=1).astype(jnp.int32)
        prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache, hidden = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=make_attn_mask(prefix_mask, prefix_ar),
            positions=prefix_positions,
            return_hidden_states=True,
        )
        h_t = hidden[0][self.memory_layer][:, :mem_len].astype(jnp.float32)
        retrieved = self.memory.read(memory_state, h_t)

        slots = jnp.arange(mem_len)[None, :]
        ablated = (token_indices[:, None] >= 0) & (slots == token_indices[:, None])
        retrieved_batch = jnp.broadcast_to(retrieved, (batch, *retrieved.shape[1:]))
        retrieved_batch = jnp.where(ablated[:, :, None], 0.0, retrieved_batch)
        memory_tokens = (self.memory_gate.value * retrieved_batch).astype(prefix_tokens.dtype)

        prefix_len = prefix_tokens.shape[1]
        padded_cache = jax.tree.map(
            lambda x: jnp.broadcast_to(
                jnp.pad(
                    x,
                    ((0, 0), (0, 0), (0, mem_len + self.causal_token_len), (0, 0), (0, 0)),
                ),
                (x.shape[0], batch, x.shape[2] + mem_len + self.causal_token_len, *x.shape[3:]),
            ),
            kv_cache,
        )
        memory_mask = jnp.broadcast_to(
            make_memory_step_mask(prefix_mask, prefix_ar, mem_len, self.causal_token_len),
            (batch, mem_len, prefix_len + mem_len + self.causal_token_len),
        )
        memory_positions = jnp.broadcast_to(prefix_len + jnp.arange(mem_len)[None], (batch, mem_len))
        (final_ct, _), _ = self.PaliGemma.llm(
            [memory_tokens, None],
            mask=memory_mask,
            positions=memory_positions,
            kv_cache=padded_cache,
            cache_position=prefix_len,
        )
        final_ct = final_ct.astype(jnp.float32)

        repeated_state = jax.tree.map(lambda value: jnp.broadcast_to(value, (batch, *value.shape[1:])), memory_state)
        candidate_state, write_aux = self.memory.write(repeated_state, final_ct)
        eta = write_aux["eta"]
        injection = jax.tree.map(
            lambda new, old: new - old * eta.reshape((batch,) + (1,) * (old.ndim - 1)),
            candidate_state.momentum,
            repeated_state.momentum,
        )

        def tree_batch_dot(left, right):
            return sum(
                jnp.sum(x * y, axis=tuple(range(1, x.ndim)))
                for x, y in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
            )

        def tree_batch_effect(tree):
            reference = jax.tree.map(lambda value: value[:1], tree)
            difference = jax.tree.map(lambda value, base: value - base, tree, reference)
            effect = jnp.sqrt(jnp.maximum(tree_batch_dot(difference, difference), 0.0))
            norm = jnp.sqrt(jnp.maximum(tree_batch_dot(tree, tree), 0.0))
            reference_norm = norm[0]
            cosine = tree_batch_dot(tree, reference) / jnp.maximum(norm * reference_norm, 1e-12)
            return effect, effect / jnp.maximum(reference_norm, 1e-12), cosine

        ct_effect, ct_relative, ct_cosine = tree_batch_effect(final_ct)
        injection_effect, injection_relative, injection_cosine = tree_batch_effect(injection)
        return {
            "token_indices": token_indices,
            "retrieved_token_norm": jnp.linalg.norm(retrieved[0], axis=-1),
            "final_ct_effect_l2": ct_effect,
            "final_ct_effect_relative": ct_relative,
            "final_ct_cosine": ct_cosine,
            "injection_effect_l2": injection_effect,
            "injection_effect_relative": injection_relative,
            "injection_cosine": injection_cosine,
            "baseline_final_ct_norm": jnp.linalg.norm(final_ct[0]),
            "baseline_injection_norm": jnp.sqrt(jnp.maximum(tree_batch_dot(injection, injection)[0], 0.0)),
            "retrieval_rms": jnp.sqrt(jnp.mean(jnp.square(retrieved))),
        }

    @staticmethod
    def _v32_memory_scale_metrics(
        prepared: dict[str, at.Array | _gemma.KVCache], *, num_img: int
    ) -> dict[str, at.Array]:
        """RMS measurements shared by the full and early-only memory diagnostics."""

        h8_all = prepared["h8_all"]
        h8_top = prepared["h8_top"]
        retrieved = prepared["retrieved"]
        memory_tokens = prepared["memory_tokens"]
        prefix_mask = prepared["prefix_mask"]
        h8_all_f32 = h8_all.astype(jnp.float32)
        valid = prefix_mask[..., None].astype(jnp.float32)
        hidden_width = h8_all.shape[-1]
        h8_valid_token_count = jnp.sum(prefix_mask, axis=1)
        h8_valid_count = jnp.maximum(h8_valid_token_count * hidden_width, 1)
        context_mask = prefix_mask[:, num_img:]
        context_valid = context_mask[..., None].astype(jnp.float32)
        context_valid_count = jnp.maximum(jnp.sum(context_mask, axis=1) * hidden_width, 1)
        return {
            "h8_all_rms": jnp.sqrt(jnp.mean(jnp.square(h8_all_f32), axis=(1, 2))),
            "h8_valid_rms": jnp.sqrt(jnp.sum(jnp.square(h8_all_f32) * valid, axis=(1, 2)) / h8_valid_count),
            "h8_valid_token_count": h8_valid_token_count,
            "h8_image_rms": jnp.sqrt(jnp.mean(jnp.square(h8_all_f32[:, :num_img]), axis=(1, 2))),
            "h8_context_valid_rms": jnp.sqrt(
                jnp.sum(jnp.square(h8_all_f32[:, num_img:]) * context_valid, axis=(1, 2)) / context_valid_count
            ),
            "h8_top_rms": jnp.sqrt(jnp.mean(jnp.square(h8_top.astype(jnp.float32)), axis=(1, 2))),
            "retrieved_rms": jnp.sqrt(jnp.mean(jnp.square(retrieved.astype(jnp.float32)), axis=(1, 2))),
            "memory_token_rms": jnp.sqrt(jnp.mean(jnp.square(memory_tokens.astype(jnp.float32)), axis=(1, 2))),
        }

    def v32_memory_interface_step(
        self,
        observation: _model.Observation,
        memory_state: _memory.MemoryState,
    ) -> dict[str, at.Array]:
        """CPU-efficient v3.2/v3.3 replay of one frame's memory interface.

        This is the early-only counterpart of :meth:`v32_query_attention_step`: it uses the
        identical preprocessing, cache geometry, blocks ``0..memory_layer``, task-conditioned
        writer (when configured), pre-write read, gate, and RMS definitions. It deliberately
        skips every later transformer block, attention-map recomputation, and per-slot write
        diagnostics. The returned ``write_tokens`` can therefore be passed directly to
        ``memory.write`` by an offline recurrent replay without paying for policy prediction.
        """

        assert self.predict_with_memory, "the model was not built with predict_with_memory"
        if self.memory_architecture != "v32_layer8_dual_query":
            raise ValueError("memory-interface diagnostics are only defined for the v3.2/v3.3 architecture.")
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(preprocessed)
        num_img = prefix_mask.shape[1] - self.max_token_len
        prepared = self._v32_prepare_memory_interface(
            prefix_tokens,
            prefix_mask,
            prefix_ar,
            memory_state,
            top_token_count=num_img // len(preprocessed.images),
            state_token_mask=preprocessed.token_state_mask,
        )
        h8_top = prepared["h8_top"]
        write_keys, write_values = self.memory.project_kv(prepared["write_tokens"])
        result = {
            "read_queries": prepared["read_queries"],
            "write_tokens": prepared["write_tokens"],
            "retrieved": prepared["retrieved"],
            # v3.4 ladder support: the exact key/value pairs the write would store (rung 2's
            # own-key recall reads back read_key(M_t, write_keys) after the caller commits).
            "write_keys": write_keys,
            "write_values": write_values,
            "memory_gate_norm": jnp.broadcast_to(jnp.linalg.norm(self._v32_content_gate()), (h8_top.shape[0],)),
            **self._v32_memory_scale_metrics(prepared, num_img=num_img),
        }
        if getattr(self, "memory_task_conditioned_write", False):
            result["write_queries"] = prepared["write_queries"]
        return result

    def v32_query_attention_step(
        self,
        observation: _model.Observation,
        memory_state: _memory.MemoryState,
    ) -> dict[str, at.Array]:
        """Offline v3.2 diagnostic: dual-query attention maps and read/write tensors for one frame.

        Runs the same split prefix as inference (blocks 0..memory_layer, dual-query read, gated
        inject, blocks memory_layer+1..end) against the *pre-write* ``memory_state`` and returns
        the FP32 query->patch attention maps of both banks together with the tensors the frame
        would read and write.  Nothing is committed: the caller advances state through
        :meth:`sample_with_memory`, whose write consumes exactly the ``write_tokens`` returned
        here (same ``h8_top`` source).
        """
        assert self.predict_with_memory, "the model was not built with predict_with_memory"
        if self.memory_architecture != "v32_layer8_dual_query":
            raise ValueError("query-attention diagnostics are only defined for the v3.2 architecture.")
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(preprocessed)
        num_img = prefix_mask.shape[1] - self.max_token_len
        top_tokens = num_img // len(preprocessed.images)
        prepared = self._v32_prepare_memory_prefix(
            prefix_tokens,
            prefix_mask,
            prefix_ar,
            memory_state,
            top_token_count=top_tokens,
            state_token_mask=preprocessed.token_state_mask,
        )
        h8_top = prepared["h8_top"]
        source_valid = self._v32_top_patch_valid(top_tokens)
        write_tokens = prepared["write_tokens"]
        retrieved = prepared["retrieved"]
        slot_aux = self.memory.token_write_diagnostics(memory_state, write_tokens)
        conditioned = {}
        if getattr(self, "memory_task_conditioned_write", False):
            # v3.3 extras: the instruction-conditioned query bank actually used above, plus the
            # unconditioned-Q0 attention as the within-frame baseline -- their divergence is the
            # direct readout of how much the instruction steers the writer.
            conditioned = {
                "write_queries": prepared["write_queries"],
                "write_attention_base": self.write_query_compressor.attention_probs(h8_top, source_valid=source_valid),
            }
        return {
            "read_attention": self.read_query_compressor.attention_probs(h8_top, source_valid=source_valid),
            "write_attention": self.write_query_compressor.attention_probs(
                h8_top, queries=prepared["write_queries"], source_valid=source_valid
            ),
            "read_queries": prepared["read_queries"],
            "write_tokens": write_tokens,
            "retrieved": retrieved,
            "retrieved_slot_norm": jnp.linalg.norm(retrieved.astype(jnp.float32), axis=-1),
            "write_slot_norm": jnp.linalg.norm(write_tokens.astype(jnp.float32), axis=-1),
            "h8_top_norm": jnp.linalg.norm(h8_top, axis=-1),
            "memory_gate_norm": jnp.broadcast_to(jnp.linalg.norm(self._v32_content_gate()), (h8_top.shape[0],)),
            **self._v32_memory_scale_metrics(prepared, num_img=num_img),
            **conditioned,
            **{f"write_slot_{key}": value for key, value in slot_aux.items()},
        }

    def v33_endpoint_gradient_step(
        self,
        observation: _model.Observation,
        *,
        gate_override: float | None = None,
    ) -> dict[str, at.Array]:
        """The v3.3 gradient-flow check (handoff section 16): does the ENDPOINT's subtask CE
        reach the write tokens of every earlier step?

        Replays a sequence batch deterministically (no augmentation, no flow loss, no TBPTT
        fences -- matching memory-critical samples, which carry none) and returns
        ``g_tau = ||d CE(t_q) / d z_tau^w||`` per step, where t_q is each sample's last valid
        step. The causal order (read -> predict -> write) implies g_{t_q} = 0 exactly; a
        working credit path shows nonzero g_tau back through the evidence phase.

        ``gate_override`` replaces the zero-init content gate with a constant for the probe
        only: at initialization the gate is exactly zero, so the true gradient is trivially
        zero -- overriding verifies the PATHWAY exists; running without it on a trained
        checkpoint verifies the realized gradient.
        """
        assert self.predict_with_memory, "the model was not built with predict_with_memory"
        if self.memory_architecture != "v32_layer8_dual_query":
            raise ValueError("the endpoint gradient probe is only defined for the v3.2/v3.3 interface.")
        if observation.seq_step_mask is None:
            raise ValueError("the endpoint gradient probe needs a sequence batch (seq_step_mask present).")
        b, t = observation.seq_step_mask.shape
        causal_len = observation.tokenized_causal.shape[-1]
        mem_len = self.memory_query_tokens
        gate = (
            None if gate_override is None else jnp.full((self.memory.config.d_value,), gate_override, dtype=jnp.float32)
        )

        def step_first(x):
            return jnp.moveaxis(x, 1, 0)

        xs = {
            "images": {k: step_first(v) for k, v in observation.images.items()},
            "state": step_first(observation.state),
            "tokens": step_first(observation.tokenized_prompt),
            "token_mask": step_first(observation.tokenized_prompt_mask),
            "causal": step_first(observation.tokenized_causal),
            "causal_mask": step_first(observation.tokenized_causal_mask),
            "step_valid": step_first(observation.seq_step_mask),
        }
        if observation.token_state_mask is not None:
            xs["token_state_mask"] = step_first(observation.token_state_mask)

        def endpoint_ce(taps):
            def step(state, x):
                obs_k = _model.Observation(
                    images=x["images"],
                    image_masks={k: jnp.ones(b, dtype=bool) for k in x["images"]},
                    state=x["state"],
                    tokenized_prompt=x["tokens"],
                    tokenized_prompt_mask=x["token_mask"],
                )
                prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(obs_k)
                num_img = prefix_mask.shape[1] - self.max_token_len
                prefix_len = prefix_mask.shape[1]
                prepared = self._v32_prepare_memory_prefix(
                    prefix_tokens,
                    prefix_mask,
                    prefix_ar,
                    state,
                    top_token_count=num_img // len(x["images"]),
                    gate_value=gate,
                    state_token_mask=x.get("token_state_mask"),
                )
                causal_mask_k = x["causal_mask"]
                causal_emb = self.PaliGemma.llm(x["causal"], method="embed")
                causal_positions = jnp.broadcast_to(
                    prefix_len + mem_len + jnp.arange(causal_len)[None], (b, causal_len)
                )
                (causal_out, _), _ = self.PaliGemma.llm(
                    [causal_emb, None],
                    mask=self._v32_causal_mask(prefix_mask, causal_mask_k),
                    positions=causal_positions,
                    kv_cache=prepared["cache"],
                    cache_position=prefix_len + mem_len,
                )
                ce_hidden = jnp.concatenate(
                    [self._v32_causal_seed(prepared["final_prefix"], prefix_mask), causal_out[:, :-1]], axis=1
                )
                logits = self.PaliGemma.llm(ce_hidden, method="decode").astype(jnp.float32)
                token_logp = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), x["causal"][..., None], axis=-1)[
                    ..., 0
                ]
                ce = -jnp.sum(token_logp * causal_mask_k, axis=-1) / jnp.clip(jnp.sum(causal_mask_k, axis=-1), 1)

                write_tokens = prepared["write_tokens"] + x["tap"]
                new_state, _ = self.memory.write(state, write_tokens)
                valid = x["step_valid"]
                state = jax.tree.map(
                    lambda n, o: jnp.where(valid.reshape((b,) + (1,) * (n.ndim - 1)), n, o), new_state, state
                )
                return state, ce * valid.astype(jnp.float32)

            _, ce_steps = jax.lax.scan(
                jax.checkpoint(step, prevent_cse=False), self.memory.init_state(b), {**xs, "tap": taps}
            )
            n_valid = jnp.clip(jnp.sum(xs["step_valid"].astype(jnp.int32), axis=0), 1)
            endpoint = jnp.arange(t)[:, None] == (n_valid - 1)[None, :]
            return jnp.sum(ce_steps * endpoint), ce_steps

        taps = jnp.zeros((t, b, mem_len, self.memory.config.d_input), dtype=jnp.float32)
        grads, ce_steps = jax.grad(endpoint_ce, has_aux=True)(taps)
        return {
            "write_grad_norm": jnp.moveaxis(jnp.linalg.norm(grads, axis=(-2, -1)), 0, 1),
            "ce_per_step": jnp.moveaxis(ce_steps, 0, 1),
            "step_valid": observation.seq_step_mask,
        }

    @staticmethod
    def _v35_inference_mask(
        value: bool | at.Bool[at.Array, " b"] | None,
        *,
        batch_size: int,
        name: str,
    ) -> at.Bool[at.Array, " b"]:
        """Normalize one runtime transition mask without accepting accidental broadcasting."""
        if value is None:
            return jnp.zeros((batch_size,), dtype=bool)
        mask = jnp.asarray(value)
        if mask.dtype != jnp.bool_:
            raise TypeError(f"{name} must have bool dtype, got {mask.dtype}.")
        if mask.ndim == 0:
            return jnp.broadcast_to(mask, (batch_size,))
        if mask.shape != (batch_size,):
            raise ValueError(f"{name} must be a bool scalar or shape [batch] ({batch_size},), got {mask.shape}.")
        return mask

    def _v35_read_geometry(
        self,
        memory_state: _memory.MemoryState,
        read_queries: at.Float[at.Array, "b q d"],
        retrieved: at.Float[at.Array, "b q dv"],
        anchor_key: at.Float[at.Array, "b dk"],
        anchor_value: at.Float[at.Array, "b dv"],
        delay_steps: int | at.Int[at.Array, " b"],
    ) -> dict[str, at.Array]:
        """FP32 D-query alignment against the final eligible-E pooled association.

        ``anchor_value`` is the un-decayed normalized ``v_bar`` from the successful E commit;
        this helper applies the exact fixed-alpha ``rho**delay_steps`` before comparing the
        rank-one anchor prediction with the actual 16 raw reads. Negative/degenerate inputs
        fail closed to zero-valued metrics with ``v35_geometry_valid=False``.
        """
        batch_size, num_queries, _ = read_queries.shape
        anchor_key = jnp.asarray(anchor_key)
        anchor_value = jnp.asarray(anchor_value)
        if anchor_key.shape != (batch_size, self.memory.config.d_key):
            raise ValueError(
                f"v35_anchor_key must have shape [{batch_size}, {self.memory.config.d_key}], got {anchor_key.shape}."
            )
        if anchor_value.shape != (batch_size, self.memory.config.d_value):
            raise ValueError(
                "v35_anchor_value must have shape "
                f"[{batch_size}, {self.memory.config.d_value}], got {anchor_value.shape}."
            )
        if not jnp.issubdtype(anchor_key.dtype, jnp.floating) or not jnp.issubdtype(anchor_value.dtype, jnp.floating):
            raise TypeError("v35 anchor key/value must have floating dtype.")
        delay = jnp.asarray(delay_steps)
        if not jnp.issubdtype(delay.dtype, jnp.integer) or delay.dtype == jnp.bool_:
            raise TypeError(f"v35_anchor_delay_steps must have integer dtype, got {delay.dtype}.")
        if delay.ndim == 0:
            delay = jnp.broadcast_to(delay, (batch_size,))
        elif delay.shape != (batch_size,):
            raise ValueError(
                f"v35_anchor_delay_steps must be an integer scalar or shape [batch] ({batch_size},), got {delay.shape}."
            )

        query_keys = self.memory.project_q(read_queries.astype(jnp.float32))
        h_query = self.memory.hidden_key(memory_state, query_keys).astype(jnp.float32)
        h_anchor = self.memory.hidden_key(memory_state, anchor_key.astype(jnp.float32)[:, None, :])[:, 0].astype(
            jnp.float32
        )
        retrieved32 = retrieved.astype(jnp.float32)
        anchor_value32 = anchor_value.astype(jnp.float32)
        h_anchor_norm_sq = jnp.sum(jnp.square(h_anchor), axis=-1)
        h_query_norm = jnp.linalg.norm(h_query, axis=-1)
        h_anchor_norm = jnp.sqrt(h_anchor_norm_sq)
        dot = jnp.einsum("bqh,bh->bq", h_query, h_anchor, precision=jax.lax.Precision.HIGHEST)
        cosine = dot / jnp.maximum(h_query_norm * h_anchor_norm[:, None], jnp.asarray(1e-12, jnp.float32))
        beta = dot / jnp.maximum(h_anchor_norm_sq[:, None], jnp.asarray(1e-12, jnp.float32))

        delay_valid = delay >= 0
        safe_delay = jnp.maximum(delay, 0).astype(jnp.float32)
        rho = jnp.asarray(1.0 - self.memory.config.alpha_step, dtype=jnp.float32)
        retention = jnp.power(rho, safe_delay)
        retained_value = retention[:, None] * anchor_value32
        predicted = beta[..., None] * retained_value[:, None, :]
        residual = retrieved32 - predicted
        mean_read = jnp.mean(retrieved32, axis=1)
        read_slot_norm = jnp.linalg.norm(retrieved32, axis=-1)
        cancellation_ratio = jnp.linalg.norm(mean_read, axis=-1) / jnp.maximum(
            jnp.mean(read_slot_norm, axis=-1), jnp.asarray(1e-12, jnp.float32)
        )
        beta_mean = jnp.mean(beta, axis=-1)
        reference_sign = jnp.where(beta_mean >= 0, 1.0, -1.0)
        sign_consistency = jnp.mean((beta * reference_sign[:, None] > 0).astype(jnp.float32), axis=-1)
        mean_read_anchor_cosine = jnp.sum(mean_read * anchor_value32, axis=-1) / jnp.maximum(
            jnp.linalg.norm(mean_read, axis=-1) * jnp.linalg.norm(anchor_value32, axis=-1),
            jnp.asarray(1e-12, jnp.float32),
        )
        residual_rms = jnp.sqrt(jnp.mean(jnp.square(residual), axis=(1, 2)))
        actual_rms = jnp.sqrt(jnp.mean(jnp.square(retrieved32), axis=(1, 2)))
        relative_residual = residual_rms / jnp.maximum(actual_rms, jnp.asarray(1e-8, jnp.float32))

        valid = (
            delay_valid
            & jnp.all(jnp.isfinite(anchor_key), axis=-1)
            & jnp.all(jnp.isfinite(anchor_value32), axis=-1)
            & (jnp.linalg.norm(anchor_key.astype(jnp.float32), axis=-1) >= jnp.asarray(1e-8, jnp.float32))
            & (jnp.linalg.norm(anchor_value32, axis=-1) >= jnp.asarray(1e-8, jnp.float32))
            & jnp.isfinite(h_anchor_norm_sq)
            & (h_anchor_norm_sq >= jnp.asarray(self.memory.config.hidden_norm_sq_floor, jnp.float32))
            & jnp.all(jnp.isfinite(h_query), axis=(1, 2))
            & jnp.all(jnp.isfinite(retrieved32), axis=(1, 2))
        )

        def valid_scalar(value):
            return jnp.where(valid, value.astype(jnp.float32), jnp.zeros_like(value, dtype=jnp.float32))

        return {
            "v35_geometry_valid": valid,
            "v35_query_anchor_cosine": jnp.where(valid[:, None], cosine, jnp.zeros_like(cosine)),
            "v35_query_anchor_beta": jnp.where(valid[:, None], beta, jnp.zeros_like(beta)),
            "v35_query_anchor_cosine_mean": valid_scalar(jnp.mean(cosine, axis=-1)),
            "v35_query_anchor_cosine_max": valid_scalar(jnp.max(cosine, axis=-1)),
            "v35_query_low_alignment_fraction": valid_scalar(jnp.mean((cosine <= 0.1).astype(jnp.float32), axis=-1)),
            "v35_query_beta_mean": valid_scalar(beta_mean),
            "v35_query_beta_abs_mean": valid_scalar(jnp.mean(jnp.abs(beta), axis=-1)),
            "v35_query_cancellation_ratio": valid_scalar(cancellation_ratio),
            "v35_query_beta_sign_consistency": valid_scalar(sign_consistency),
            "v35_mean_raw_read_anchor_cosine": valid_scalar(mean_read_anchor_cosine),
            "v35_anchor_predicted_read_residual_rms": valid_scalar(residual_rms),
            "v35_anchor_predicted_read_relative_residual": valid_scalar(relative_residual),
            "v35_anchor_retention": valid_scalar(retention),
            "v35_anchor_delay_steps": jnp.where(valid, delay, jnp.zeros_like(delay)),
            "v35_anchor_hidden_norm_sq": valid_scalar(h_anchor_norm_sq),
        }

    def _v35_inference_transition(
        self,
        memory_state: _memory.MemoryState,
        write_tokens: at.Float[at.Array, "b n d"],
        *,
        transition_valid: bool | at.Bool[at.Array, " b"] | None,
        write_mask: bool | at.Bool[at.Array, " b"] | None,
        write_mode: str,
    ) -> tuple[_memory.MemoryState, dict[str, at.Array]]:
        """Apply the v3.5 E-only inference transition, fail-closed per batch sample.

        This is the runtime counterpart of the sequence scan's current-frame transition:

        * valid + write mask + ``normal``: eligible-E delta commit (including one decay);
        * valid without a normal write: one non-E decay;
        * invalid, omitted masks, or ``frozen``: exact state no-op.

        ``write_mask`` is intentionally not inferred from a predicted subtask or observation.
        Only the trusted episode/phase controller that produced the training-side
        ``seq_write_mask`` may assert it.  Consequently, the legacy default
        ``write_mode='normal'`` is never sufficient to make v3.5 write.
        """
        if write_mode not in ("normal", "frozen", "dynamics_only"):
            raise ValueError(f"unsupported write_mode: {write_mode!r}.")
        batch_size = write_tokens.shape[0]
        transition_valid_mask = self._v35_inference_mask(
            transition_valid, batch_size=batch_size, name="v35_transition_valid"
        )
        write_eligible = self._v35_inference_mask(write_mask, batch_size=batch_size, name="v35_write_mask")

        # Functional candidates are safe to evaluate for the whole batch.  The final tree
        # selection below is authoritative: invalid samples select their original leaves
        # exactly, including momentum and all hidden fast weights.
        write_state, candidate_aux = self.memory.write(memory_state, write_tokens)
        decay_state, _ = self.memory.decay_step(memory_state, write_tokens)
        transition_applied = transition_valid_mask & (write_mode != "frozen")
        commit_requested = transition_applied & write_eligible & (write_mode == "normal")

        def select_state(write_leaf, decay_leaf, old_leaf):
            shape = (batch_size,) + (1,) * (old_leaf.ndim - 1)
            selected_transition = jnp.where(commit_requested.reshape(shape), write_leaf, decay_leaf)
            return jnp.where(transition_applied.reshape(shape), selected_transition, old_leaf)

        new_state = jax.tree.map(select_state, write_state, decay_state, memory_state)
        candidate_commit = candidate_aux.get("commit_applied", jnp.ones((batch_size,), dtype=bool))
        commit_applied = commit_requested & candidate_commit
        decay_only = transition_applied & ~commit_applied
        invalid_write_request = write_eligible & ~transition_valid_mask
        aux = {
            **candidate_aux,
            # Override candidate-only core telemetry with the transition that was actually
            # selected.  The explicit v35 fields make commit/decay/no-op mutually exclusive.
            "commit_applied": commit_applied,
            "write_occurred": commit_applied,
            "v35_transition_valid": transition_valid_mask,
            "v35_write_eligible": write_eligible,
            "v35_transition_applied": transition_applied,
            "v35_commit_requested": commit_requested,
            "v35_commit_applied": commit_applied,
            "v35_decay_only": decay_only,
            "v35_noop": ~transition_applied,
            "v35_invalid_write_request": invalid_write_request,
        }
        return new_state, aux

    def _sample_with_memory_v32(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        memory_state: _memory.MemoryState,
        *,
        stop_token: int,
        max_decode_steps: int,
        num_steps: int | at.Int[at.Array, ""],
        noise: at.Float[at.Array, "b ah ad"] | None,
        action_prefix: _rtc.ActionPrefix | None,
        forced_subtask_tokens: at.Int[at.Array, "b cl"] | None,
        forced_subtask_mask: at.Bool[at.Array, "b cl"] | None,
        zero_read: bool,
        write_mode: str,
        v35_transition_valid: bool | at.Bool[at.Array, " b"] | None,
        v35_write_mask: bool | at.Bool[at.Array, " b"] | None,
        v35_oracle_direction: at.Float[at.Array, "b d"] | None,
        v35_oracle_injected_rms: float | at.Float[at.Array, " b"] | None,
        v35_anchor_key: at.Float[at.Array, "b dk"] | None,
        v35_anchor_value: at.Float[at.Array, "b dv"] | None,
        v35_anchor_delay_steps: int | at.Int[at.Array, " b"] | None,
    ) -> tuple[_model.Actions, _memory.MemoryState, dict[str, at.Array]]:
        """v3.2 inference: layer-8 dual-query read/write with 16 persistent memory tokens.

        ``write_mode`` implements the plan-8.4 three-way retention control:
        "normal" commits the full Titans update, "frozen" returns the input state unchanged
        (M_t = M_{t-1}, S_t = S_{t-1} -- the old ``allow_write=False``), and "dynamics_only"
        applies S_t = eta S_{t-1}, M_t = (1-alpha) M_{t-1} + S_t with the gradient term zeroed.

        In v3.5, ``v35_transition_valid`` and ``v35_write_mask`` additionally gate the current
        sampled transition.  They mirror sequence-training's valid-step and ``seq_write_mask``
        fields.  Both default to false (strict no-op), so an inference caller cannot silently
        write an O/D frame merely by retaining the legacy ``write_mode='normal'`` default.
        """

        preprocessed = _model.preprocess_observation(None, observation, train=False)
        batch = preprocessed.state.shape[0]
        self._check_action_prefix_shapes(action_prefix, batch)
        if (forced_subtask_tokens is None) != (forced_subtask_mask is None):
            raise ValueError("forced_subtask_tokens and forced_subtask_mask must be provided together.")
        if forced_subtask_tokens is not None:
            expected = (batch, self.causal_token_len)
            if forced_subtask_tokens.shape != expected or forced_subtask_mask.shape != expected:
                raise ValueError(
                    "forced subtask buffers must have shape "
                    f"{expected}; got {forced_subtask_tokens.shape} and {forced_subtask_mask.shape}."
                )

        prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(preprocessed)
        prefix_len = prefix_mask.shape[1]
        num_img = prefix_len - self.max_token_len
        top_tokens = num_img // len(preprocessed.images)
        mem_len = self.memory_query_tokens
        gen_base = prefix_len + mem_len
        prepared = self._v32_prepare_memory_prefix(
            prefix_tokens,
            prefix_mask,
            prefix_ar,
            memory_state,
            top_token_count=top_tokens,
            zero_read=zero_read,
            state_token_mask=preprocessed.token_state_mask,
            v35_oracle_direction=v35_oracle_direction,
            v35_oracle_injected_rms=v35_oracle_injected_rms,
        )
        kv_cache = prepared["cache"]
        final_prefix = prepared["final_prefix"]
        write_tokens = prepared["write_tokens"]
        retrieved = prepared["retrieved"]
        anchor_inputs = (v35_anchor_key, v35_anchor_value, v35_anchor_delay_steps)
        if any(value is not None for value in anchor_inputs) and not all(value is not None for value in anchor_inputs):
            raise ValueError("v35_anchor_key, v35_anchor_value, and v35_anchor_delay_steps must be provided together.")
        if all(value is not None for value in anchor_inputs):
            if not getattr(self, "memory_v35_enabled", False):
                raise ValueError("v3.5 anchor geometry is available only for memory_v35_enabled models.")
            geometry_aux = self._v35_read_geometry(
                memory_state,
                prepared["read_queries"],
                retrieved,
                v35_anchor_key,
                v35_anchor_value,
                v35_anchor_delay_steps,
            )
        else:
            geometry_aux = {}

        def finish(actions, tokens, token_mask, *, extra=None):
            if getattr(self, "memory_v35_enabled", False):
                new_state, write_aux = self._v35_inference_transition(
                    memory_state,
                    write_tokens,
                    transition_valid=v35_transition_valid,
                    write_mask=v35_write_mask,
                    write_mode=write_mode,
                )
            else:
                # Preserve the v3.2-v3.4 transition path exactly.  In particular, their
                # historical default remains one normal write per inference call.
                if write_mode == "dynamics_only":
                    candidate_state, write_aux = self.memory.decay_step(memory_state, write_tokens)
                else:
                    candidate_state, write_aux = self.memory.write(memory_state, write_tokens)
                new_state = memory_state if write_mode == "frozen" else candidate_state
                write_aux = {
                    **write_aux,
                    "write_occurred": jnp.full((batch,), write_mode == "normal", dtype=bool),
                }
            aux = {
                **write_aux,
                "tokens": tokens,
                "token_mask": token_mask,
                "retrieval_norm": jnp.sqrt(jnp.mean(jnp.square(retrieved.astype(jnp.float32)), axis=(1, 2))),
                "memory_gate_norm": jnp.broadcast_to(jnp.linalg.norm(self._v32_content_gate()), (batch,)),
                "read_query_norm": jnp.sqrt(
                    jnp.mean(jnp.square(prepared["read_queries"].astype(jnp.float32)), axis=(1, 2))
                ),
                "write_token_norm": jnp.sqrt(jnp.mean(jnp.square(write_tokens.astype(jnp.float32)), axis=(1, 2))),
                "v35_injected_pre_cast_rms": prepared["injected_pre_cast_rms"].astype(jnp.float32),
                "v35_injected_post_cast_rms": prepared["injected_post_cast_rms"].astype(jnp.float32),
                "v35_oracle_injection_active": prepared["v35_oracle_injection_active"],
                "v35_oracle_injection_valid": prepared["v35_oracle_injection_valid"],
                "v35_oracle_target_rms": prepared["v35_oracle_target_rms"],
                "v35_oracle_actual_rms": prepared["v35_oracle_actual_rms"],
                **geometry_aux,
            }
            if extra is not None:
                aux.update(extra)
            return _rtc.restore_action_prefix(actions, action_prefix), new_state, aux

        if forced_subtask_tokens is not None:
            gen_tokens = forced_subtask_tokens.astype(preprocessed.tokenized_prompt.dtype)
            gen_mask = forced_subtask_mask.astype(bool)
            causal_emb = self.PaliGemma.llm(gen_tokens, method="embed")
            causal_positions = jnp.broadcast_to(
                gen_base + jnp.arange(self.causal_token_len)[None], (batch, self.causal_token_len)
            )
            (causal_out, _), kv_cache = self.PaliGemma.llm(
                [causal_emb, None],
                mask=self._v32_causal_mask(prefix_mask, gen_mask),
                positions=causal_positions,
                kv_cache=kv_cache,
                cache_position=gen_base,
            )
            score_hidden = jnp.concatenate(
                [self._v32_causal_seed(final_prefix, prefix_mask), causal_out[:, :-1]], axis=1
            )
            score_logits = self.PaliGemma.llm(score_hidden, method="decode").astype(jnp.float32)
            token_logp = jnp.take_along_axis(jax.nn.log_softmax(score_logits, axis=-1), gen_tokens[..., None], axis=-1)[
                ..., 0
            ]
            total_logp = jnp.sum(token_logp * gen_mask, axis=-1)
            mean_logp = total_logp / jnp.maximum(jnp.sum(gen_mask, axis=-1), 1)

            dt = -1.0 / num_steps
            if noise is None:
                noise = jax.random.normal(rng, (batch, self.action_horizon, self.action_dim))

            def denoise(carry):
                x_t, time = carry
                if action_prefix is None:
                    model_x_t, model_time = x_t, jnp.broadcast_to(time, batch)
                else:
                    model_x_t, model_time = _rtc.condition_action_prefix(x_t, time, action_prefix)
                suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(
                    preprocessed, model_x_t, model_time
                )
                suffix_positions = gen_base + self.causal_token_len + jnp.cumsum(suffix_mask, axis=-1) - 1
                (_, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=self._v32_suffix_mask(prefix_mask, gen_mask, suffix_mask, suffix_ar),
                    positions=suffix_positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )
                return x_t + dt * self.action_out_proj(suffix_out[:, -self.action_horizon :]), time + dt

            def keep_denoising(carry):
                return carry[1] >= -dt / 2

            actions, _ = jax.lax.while_loop(keep_denoising, denoise, (noise, 1.0))
            return finish(
                actions,
                gen_tokens,
                gen_mask,
                extra={"conditioned_subtask_logp": total_logp, "conditioned_subtask_mean_logp": mean_logp},
            )

        def greedy(hidden_vec):
            logits = self.PaliGemma.llm(hidden_vec[:, None], method="decode")[:, 0]
            return jnp.argmax(logits, axis=-1).astype(preprocessed.tokenized_prompt.dtype)

        token0 = greedy(self._v32_causal_seed(final_prefix, prefix_mask)[:, 0])
        gen_tokens = jnp.zeros((batch, self.causal_token_len), dtype=preprocessed.tokenized_prompt.dtype)
        gen_mask = jnp.zeros((batch, self.causal_token_len), dtype=bool)

        def record(tokens, mask, done, token, index):
            tokens = tokens.at[:, index].set(jnp.where(done, tokens[:, index], token))
            mask = mask.at[:, index].set(~done)
            return tokens, mask, done | (token == stop_token) | (token == PALIGEMMA_EOS_TOKEN)

        gen_tokens, gen_mask, done = record(gen_tokens, gen_mask, jnp.zeros(batch, dtype=bool), token0, 0)

        def decode_cond(carry):
            return (carry[-1] < max_decode_steps) & ~jnp.all(carry[2])

        def decode_step(carry):
            tokens, mask, done, previous, cache, index = carry
            token_emb = self.PaliGemma.llm(previous[:, None], method="embed")
            (out, _), cache = self.PaliGemma.llm(
                [token_emb, None],
                mask=self._v32_step_mask(prefix_mask, index),
                positions=jnp.broadcast_to(gen_base + index - 1, (batch, 1)),
                kv_cache=cache,
                cache_position=gen_base + index - 1,
            )
            token = greedy(out[:, 0])
            tokens, mask, done = record(tokens, mask, done, token, index)
            return tokens, mask, done, token, cache, index + 1

        carry = (gen_tokens, gen_mask, done, token0, kv_cache, jnp.asarray(1, dtype=jnp.int32))
        gen_tokens, gen_mask, _, previous, kv_cache, generated = jax.lax.while_loop(decode_cond, decode_step, carry)
        last_emb = self.PaliGemma.llm(previous[:, None], method="embed")
        _, kv_cache = self.PaliGemma.llm(
            [last_emb, None],
            mask=self._v32_step_mask(prefix_mask, generated),
            positions=jnp.broadcast_to(gen_base + generated - 1, (batch, 1)),
            kv_cache=kv_cache,
            cache_position=gen_base + generated - 1,
        )

        causal_live = jnp.arange(self.causal_token_len)[None] < jnp.sum(gen_mask, axis=-1)[:, None]
        dt = -1.0 / num_steps
        if noise is None:
            noise = jax.random.normal(rng, (batch, self.action_horizon, self.action_dim))

        def denoise(carry):
            x_t, time = carry
            if action_prefix is None:
                model_x_t, model_time = x_t, jnp.broadcast_to(time, batch)
            else:
                model_x_t, model_time = _rtc.condition_action_prefix(x_t, time, action_prefix)
            suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(preprocessed, model_x_t, model_time)
            suffix_positions = gen_base + self.causal_token_len + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=self._v32_suffix_mask(prefix_mask, causal_live, suffix_mask, suffix_ar),
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            return x_t + dt * self.action_out_proj(suffix_out[:, -self.action_horizon :]), time + dt

        def keep_denoising(carry):
            return carry[1] >= -dt / 2

        actions, _ = jax.lax.while_loop(keep_denoising, denoise, (noise, 1.0))
        return finish(actions, gen_tokens, gen_mask)

    def sample_with_memory(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        memory_state: _memory.MemoryState,
        *,
        stop_token: int,
        max_decode_steps: int = 10,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        action_prefix: _rtc.ActionPrefix | None = None,
        forced_subtask_tokens: at.Int[at.Array, "b cl"] | None = None,
        forced_subtask_mask: at.Bool[at.Array, "b cl"] | None = None,
        zero_read: bool = False,
        allow_write: bool = True,
        write_mode: str | None = None,
        v35_transition_valid: bool | at.Bool[at.Array, " b"] | None = None,
        v35_write_mask: bool | at.Bool[at.Array, " b"] | None = None,
        v35_oracle_direction: at.Float[at.Array, "b d"] | None = None,
        v35_oracle_injected_rms: float | at.Float[at.Array, " b"] | None = None,
        v35_anchor_key: at.Float[at.Array, "b dk"] | None = None,
        v35_anchor_value: at.Float[at.Array, "b dv"] | None = None,
        v35_anchor_delay_steps: int | at.Int[at.Array, " b"] | None = None,
    ) -> tuple[_model.Actions, _memory.MemoryState, dict[str, at.Array]]:
        """Memory-conditioned fused inference: one prefill + an incremental memory append.

        Static layout: [images 0..num_img | context text (positions unchanged) | memory tokens |
        generated subtask | action suffix]. The prefill is identical to the baseline path and
        also yields the layer-`memory_layer` top-camera hidden states h_t. The memory is read
        with h_t and the retrieved tokens (content-gated, zero-init) are appended to the KV
        cache, producing contextualized memory-token outputs c_t; the subtask is decoded and
        the actions denoised against the extended cache. Only after prediction, the memory is
        written with h_t (v3) or final c18 (v3.1), according to ``memory_write_source``. Returns
        (actions, new_memory_state, aux) with generated tokens/mask and write diagnostics.

        ``forced_subtask_tokens`` is a diagnostics-only teacher-forcing path. It must be a
        left-aligned, causal-token-length buffer accompanied by ``forced_subtask_mask``. The
        returned ``conditioned_subtask_logp`` is the exact summed next-token log probability of
        that complete sequence. ``zero_read`` preserves the memory-token positions but replaces
        retrieved content with zeros. ``allow_write=False`` still computes write diagnostics but
        returns the input state unchanged. These controls make counterfactual evaluations
        side-effect-free without changing normal inference defaults.

        ``write_mode`` (v3.4 plan 8.4) generalizes ``allow_write`` into the three-way retention
        control: "normal" (full Titans update), "frozen" (state unchanged; equals
        ``allow_write=False``), "dynamics_only" (gradient term zeroed: S_t = eta S_{t-1},
        M_t = (1-alpha) M_{t-1} + S_t). When None, it is derived from ``allow_write``.

        v3.5 additionally requires explicit per-sample transition masks.  Set
        ``v35_transition_valid=True`` for a real sampled memory-clock transition, and set
        ``v35_write_mask=True`` only for a manifest-eligible evidence (E) frame; valid O/D
        frames pass a false write mask and therefore decay only.  Both masks default to false,
        making a call with missing phase metadata an exact state no-op.  The masks mirror
        sequence training's valid-step/``seq_write_mask`` contract and are ignored by v3.4.

        Consumer diagnostics may provide ``v35_oracle_direction`` plus an exact target
        ``v35_oracle_injected_rms``. This bypasses state/query retrieval and pins direct-carry,
        correct-prototype, and opposite-donor directions through one shared FP32 path. Passing
        the final successful-E pooled key/value and its non-write delay additionally reports
        per-query hidden-key cosine/beta, cancellation, and anchor-predicted read residuals.
        """
        assert self.predict_with_memory, "the model was not built with predict_with_memory"
        assert max_decode_steps <= self.causal_token_len
        if write_mode is None:
            write_mode = "normal" if allow_write else "frozen"
        if write_mode not in ("normal", "frozen", "dynamics_only"):
            raise ValueError(f"unsupported write_mode: {write_mode!r}.")
        if getattr(self, "memory_architecture", "v3_v31") == "v32_layer8_dual_query":
            return self._sample_with_memory_v32(
                rng,
                observation,
                memory_state,
                stop_token=stop_token,
                max_decode_steps=max_decode_steps,
                num_steps=num_steps,
                noise=noise,
                action_prefix=action_prefix,
                forced_subtask_tokens=forced_subtask_tokens,
                forced_subtask_mask=forced_subtask_mask,
                zero_read=zero_read,
                write_mode=write_mode,
                v35_transition_valid=v35_transition_valid,
                v35_write_mask=v35_write_mask,
                v35_oracle_direction=v35_oracle_direction,
                v35_oracle_injected_rms=v35_oracle_injected_rms,
                v35_anchor_key=v35_anchor_key,
                v35_anchor_value=v35_anchor_value,
                v35_anchor_delay_steps=v35_anchor_delay_steps,
            )
        if any(
            value is not None
            for value in (
                v35_oracle_direction,
                v35_oracle_injected_rms,
                v35_anchor_key,
                v35_anchor_value,
                v35_anchor_delay_steps,
            )
        ):
            raise ValueError("v3.5 oracle/anchor diagnostics require the v3.2 dual-query architecture.")
        if write_mode == "dynamics_only":
            raise ValueError("write_mode='dynamics_only' is only implemented for the v3.2/v3.3/v3.4 interface.")
        # the legacy v3/v3.1 body below branches on allow_write; keep it consistent with an
        # explicitly passed write_mode
        allow_write = write_mode == "normal"
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        batch = preprocessed.state.shape[0]
        self._check_action_prefix_shapes(action_prefix, batch)
        if (forced_subtask_tokens is None) != (forced_subtask_mask is None):
            raise ValueError("forced_subtask_tokens and forced_subtask_mask must be provided together.")
        if forced_subtask_tokens is not None:
            expected = (batch, self.causal_token_len)
            if forced_subtask_tokens.shape != expected or forced_subtask_mask.shape != expected:
                raise ValueError(
                    "forced subtask buffers must have shape "
                    f"{expected}; got {forced_subtask_tokens.shape} and {forced_subtask_mask.shape}."
                )

        # prefill of images + context text, identical to the baseline path, capturing the
        # per-layer hidden states in the same forward
        img_tokens = []
        img_masks = []
        for name in preprocessed.images:
            image_tokens, _ = self.PaliGemma.img(preprocessed.images[name], train=False)
            img_tokens.append(image_tokens)
            img_masks.append(einops.repeat(preprocessed.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
        img_tokens = jnp.concatenate(img_tokens, axis=1)
        img_mask = jnp.concatenate(img_masks, axis=1)

        prompt = preprocessed.tokenized_prompt
        prompt_mask = preprocessed.tokenized_prompt_mask
        ar = (
            preprocessed.token_ar_mask
            if preprocessed.token_ar_mask is not None
            else jnp.zeros(prompt.shape, dtype=jnp.int32)
        )
        num_img = img_mask.shape[1]
        prefix_len = num_img + prompt.shape[1]
        mem_len = num_img // len(preprocessed.images)  # one memory token per top-camera token
        causal_len = self.causal_token_len
        gen_base = prefix_len + mem_len  # first causal cache slot; slot index == position id

        prefix_tokens = jnp.concatenate([img_tokens, self.PaliGemma.llm(prompt, method="embed")], axis=1)
        prefix_mask = jnp.concatenate([img_mask, prompt_mask], axis=1)
        prefix_ar = jnp.concatenate([0 * img_mask, ar], axis=1).astype(jnp.int32)
        attn_mask = make_attn_mask(prefix_mask, prefix_ar)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache, hidden = self.PaliGemma.llm(
            [prefix_tokens, None], mask=attn_mask, positions=positions, return_hidden_states=True
        )
        h_t = hidden[0][self.memory_layer][:, :mem_len].astype(jnp.float32)

        # read M_{t-1} and append the content-gated memory tokens to the cache (their K/V land
        # in slots [prefix_len, prefix_len + mem_len)); the cache is padded once for the memory
        # block plus the whole causal window
        retrieved = self.memory.read(memory_state, h_t)
        if zero_read:
            retrieved = jnp.zeros_like(retrieved)
        mem_tokens = (self.memory_gate.value * retrieved).astype(prefix_tokens.dtype)
        kv_cache = jax.tree.map(
            lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, mem_len + causal_len), (0, 0), (0, 0))), kv_cache
        )
        mem_mask = make_memory_step_mask(prefix_mask, prefix_ar, mem_len, causal_len)
        mem_positions = prefix_len + jnp.broadcast_to(jnp.arange(mem_len), (batch, mem_len))
        (mem_out, _), kv_cache = self.PaliGemma.llm(
            [mem_tokens, None],
            mask=mem_mask,
            positions=mem_positions,
            kv_cache=kv_cache,
            cache_position=prefix_len,
        )
        c_t = mem_out.astype(jnp.float32)

        if forced_subtask_tokens is not None:
            # Teacher-force the complete canonical sequence in one causal extension. Position
            # zero is predicted from the final memory-token output, exactly like free decoding.
            gen_tokens = forced_subtask_tokens.astype(prompt.dtype)
            gen_mask = forced_subtask_mask.astype(bool)
            causal_emb = self.PaliGemma.llm(gen_tokens, method="embed")
            causal_rows = jnp.concatenate(
                [
                    einops.repeat(prefix_mask, "b p -> b c p", c=causal_len),
                    jnp.ones((batch, causal_len, mem_len), dtype=bool),
                    jnp.tril(jnp.ones((causal_len, causal_len), dtype=bool))[None] & gen_mask[:, None, :],
                ],
                axis=-1,
            )
            causal_positions = jnp.broadcast_to(gen_base + jnp.arange(causal_len)[None], (batch, causal_len))
            (causal_out, _), kv_cache = self.PaliGemma.llm(
                [causal_emb, None],
                mask=causal_rows,
                positions=causal_positions,
                kv_cache=kv_cache,
                cache_position=gen_base,
            )

            # Exact full-sequence score: p(token 0) is read from the last memory output, and
            # every later token is read from its teacher-forced predecessor.
            score_hidden = jnp.concatenate([mem_out[:, -1:], causal_out[:, :-1]], axis=1)
            score_logits = self.PaliGemma.llm(score_hidden, method="decode").astype(jnp.float32)
            token_logp = jnp.take_along_axis(jax.nn.log_softmax(score_logits, axis=-1), gen_tokens[..., None], axis=-1)[
                ..., 0
            ]
            conditioned_subtask_logp = jnp.sum(token_logp * gen_mask, axis=-1)
            conditioned_subtask_mean_logp = conditioned_subtask_logp / jnp.maximum(jnp.sum(gen_mask, axis=-1), 1)

            suffix_view = jnp.concatenate([prefix_mask, jnp.ones((batch, mem_len), dtype=bool), gen_mask], axis=1)
            offset = gen_base + causal_len
            dt = -1.0 / num_steps
            if noise is None:
                noise = jax.random.normal(rng, (batch, self.action_horizon, self.action_dim))

            def forced_denoise_step(carry):
                x_t, time = carry
                if action_prefix is None:
                    model_x_t = x_t
                    model_time = jnp.broadcast_to(time, batch)
                else:
                    model_x_t, model_time = _rtc.condition_action_prefix(x_t, time, action_prefix)
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                    preprocessed, model_x_t, model_time
                )
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask = einops.repeat(suffix_view, "b p -> b s p", s=suffix_tokens.shape[1])
                full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
                suffix_positions = offset + jnp.cumsum(suffix_mask, axis=-1) - 1
                (_, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=suffix_positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
                return x_t + dt * v_t, time + dt

            def forced_denoise_cond(carry):
                _, time = carry
                return time >= -dt / 2

            actions, _ = jax.lax.while_loop(forced_denoise_cond, forced_denoise_step, (noise, 1.0))
            actions = _rtc.restore_action_prefix(actions, action_prefix)
            write_source = self._select_memory_write_source(h_t, c_t)
            candidate_state, write_aux = self.memory.write(memory_state, write_source)
            new_state = candidate_state if allow_write else memory_state
            aux = {
                **write_aux,
                "tokens": gen_tokens,
                "token_mask": gen_mask,
                "conditioned_subtask_logp": conditioned_subtask_logp,
                "conditioned_subtask_mean_logp": conditioned_subtask_mean_logp,
                "retrieval_norm": jnp.sqrt(jnp.mean(jnp.square(retrieved.astype(jnp.float32)), axis=(1, 2))),
                "memory_gate_norm": jnp.broadcast_to(jnp.linalg.norm(self.memory_gate.value), (batch,)),
                "write_occurred": jnp.full((batch,), allow_write, dtype=bool),
            }
            return actions, new_state, aux

        def greedy(hidden_vec):  # [b, emb] -> [b] next token
            logits = self.PaliGemma.llm(hidden_vec[:, None], method="decode")[:, 0]
            return jnp.argmax(logits, axis=-1).astype(prompt.dtype)

        def step_mask(k):  # attention columns for one incremental query at decode step k
            gen_valid = jnp.broadcast_to(jnp.arange(causal_len)[None, :] < k, (batch, causal_len))
            return jnp.concatenate([prefix_mask, jnp.ones((batch, mem_len), dtype=bool), gen_valid], axis=1)

        # the first subtask token is read out from the last memory token: position 1223 predicts
        # position 1224 (standard next-token adjacency), memory-conditioned by construction.
        # Note this readout position is untrained until phase I, so the memory path's subtask
        # differs from the baseline until then.
        token0 = greedy(mem_out[:, -1])

        def write_token(gen_tokens, gen_mask, done, token, k):
            """Records `token` as the k-th generated token of every unfinished sample."""
            gen_tokens = gen_tokens.at[:, k].set(jnp.where(done, gen_tokens[:, k], token))
            gen_mask = gen_mask.at[:, k].set(~done)
            done = done | (token == stop_token) | (token == PALIGEMMA_EOS_TOKEN)
            return gen_tokens, gen_mask, done

        gen_tokens = jnp.zeros((batch, causal_len), dtype=prompt.dtype)
        gen_mask = jnp.zeros((batch, causal_len), dtype=bool)
        gen_tokens, gen_mask, done = write_token(gen_tokens, gen_mask, jnp.zeros(batch, dtype=bool), token0, 0)

        def cond(carry):
            _, _, done, _, _, k = carry
            return (k < max_decode_steps) & ~jnp.all(done)

        def step(carry):
            gen_tokens, gen_mask, done, prev, kv_cache, k = carry
            # feed the previous token: its k/v land in cache slot gen_base + k - 1, which is
            # also its position id (the causal window is left-aligned and starts at gen_base)
            tok_emb = self.PaliGemma.llm(prev[:, None], method="embed")
            (out, _), kv_cache = self.PaliGemma.llm(
                [tok_emb, None],
                mask=step_mask(k)[:, None, :],
                positions=jnp.broadcast_to(gen_base + k - 1, (batch, 1)),
                kv_cache=kv_cache,
                cache_position=gen_base + k - 1,
            )
            token = greedy(out[:, 0])
            gen_tokens, gen_mask, done = write_token(gen_tokens, gen_mask, done, token, k)
            return gen_tokens, gen_mask, done, token, kv_cache, k + 1

        carry = (gen_tokens, gen_mask, done, token0, kv_cache, jnp.asarray(1, dtype=jnp.int32))
        gen_tokens, gen_mask, _, prev, kv_cache, k = jax.lax.while_loop(cond, step, carry)

        # commit the last generated token's K/V: the loop writes a token's K/V on the following
        # iteration, so the token that ended decoding (typically the stop terminator) would
        # otherwise leave zeros in its cache slot for the action suffix to attend to
        last_emb = self.PaliGemma.llm(prev[:, None], method="embed")
        _, kv_cache = self.PaliGemma.llm(
            [last_emb, None],
            mask=step_mask(k)[:, None, :],
            positions=jnp.broadcast_to(gen_base + k - 1, (batch, 1)),
            kv_cache=kv_cache,
            cache_position=gen_base + k - 1,
        )

        # denoise, attending to [prefix | memory | generated subtask]; the suffix positions are
        # fully static: the causal window is pre-allocated, so the suffix starts at
        # gen_base + causal_len for every sample
        gen_cols = jnp.arange(causal_len)[None, :] < jnp.sum(gen_mask, axis=-1)[:, None]
        suffix_view = jnp.concatenate([prefix_mask, jnp.ones((batch, mem_len), dtype=bool), gen_cols], axis=1)
        offset = gen_base + causal_len

        dt = -1.0 / num_steps
        if noise is None:
            noise = jax.random.normal(rng, (batch, self.action_horizon, self.action_dim))

        def denoise_step(carry):
            x_t, time = carry
            if action_prefix is None:
                model_x_t = x_t
                model_time = jnp.broadcast_to(time, batch)
            else:
                model_x_t, model_time = _rtc.condition_action_prefix(x_t, time, action_prefix)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                preprocessed, model_x_t, model_time
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(suffix_view, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = offset + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
            return x_t + dt * v_t, time + dt

        def denoise_cond(carry):
            _, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(denoise_cond, denoise_step, (noise, 1.0))
        x_0 = _rtc.restore_action_prefix(x_0, action_prefix)

        # Write only after prediction. Reads always use h_t; v3.1 writes the final-normalized
        # memory-token output.
        write_source = self._select_memory_write_source(h_t, c_t)
        candidate_state, write_aux = self.memory.write(memory_state, write_source)
        new_state = candidate_state if allow_write else memory_state
        aux = {
            **write_aux,
            "tokens": gen_tokens,
            "token_mask": gen_mask,
            "retrieval_norm": jnp.sqrt(jnp.mean(jnp.square(retrieved.astype(jnp.float32)), axis=(1, 2))),
            "memory_gate_norm": jnp.broadcast_to(jnp.linalg.norm(self.memory_gate.value), (batch,)),
            "write_occurred": jnp.full((batch,), allow_write, dtype=bool),
        }
        return x_0, new_state, aux

    def _compute_sequence_loss_v32(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> dict[str, at.Array]:
        """Sequence objective for the layer-8 dual-query v3.2 interface.

        v3.4 additions (V34_PLAN_final.md), each active only when its config flag and data
        fields are present:
          * plan 5.2 -- input-level per-segment state masking: samples whose segment drew the
            mask have their state-digit embeddings replaced by the learned null for EVERY step;
            everything downstream (h8, read queries, writes, retrieval, CE, flow) uses the
            masked view. The dual-view variant instead drives the memory-state evolution
            (write tokens) from the full view while CE/flow/read come from the masked forward.
          * plan 5.4 -- the first causal token is decoded from the last valid non-memory prefix
            position (`_v32_causal_seed`).
          * plan 5.1 -- auxiliary demand: decode the per-step subtask class from the POST-write
            memory through the frame-invariant Q_aux bank (read_key; memory-only by
            construction). Class-balanced macro CE is assembled in train.py from the per-class
            sums returned here. Optional episode-vs-reset margin with a stop-gradient baseline.
          * Section 6 -- online ladder probes: side from stop-gradient'ed pooled write tokens
            (evidence frames) and from stop-gradient'ed pooled standard-read retrieval
            (waiting frames). Head updates are isolated in train.py.
        """

        b, t = observation.seq_step_mask.shape
        ah, ad = actions.shape[-2:]
        aug_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        images = self._augment_sequence_images(aug_rng, observation.images) if train else observation.images
        causal_len = observation.tokenized_causal.shape[-1]
        quiz = (self.memory_probe_weight > 0 or self.memory_probe_diagnostic) and observation.seq_probe_mask is not None
        aux_on = getattr(self, "memory_aux_loss_weight", 0.0) > 0 and observation.seq_subtask_class is not None
        margin_on = aux_on and getattr(self, "memory_aux_margin_weight", 0.0) > 0
        ladder_on = (
            getattr(self, "memory_ladder_probes", False)
            and observation.seq_side_label is not None
            and observation.seq_evidence_mask is not None
            and observation.seq_waiting_mask is not None
        )
        v35_on = getattr(self, "memory_v35_enabled", False)
        if v35_on:
            required = (
                "seq_write_mask",
                "seq_decision_mask",
                "seq_read_state_valid",
                "seq_read_credit_reachable",
                "seq_decay_gap_before",
                "seq_use_pressure_mask",
                "seq_side_label",
                "seq_memory_cell",
            )
            missing = [name for name in required if getattr(observation, name, None) is None]
            if missing:
                raise ValueError(f"v3.5 sequence training is missing required observation fields: {missing}.")
        mask_state = (
            getattr(self, "memory_state_mask_prob", 0.0) > 0
            and observation.seq_state_masked is not None
            and observation.token_state_mask is not None
        )
        dual_view = mask_state and getattr(self, "memory_state_mask_dual_view", False)

        def step_first(x):
            return jnp.moveaxis(x, 1, 0)

        xs = {
            "images": {k: step_first(v) for k, v in images.items()},
            "state": step_first(observation.state),
            "tokens": step_first(observation.tokenized_prompt),
            "token_mask": step_first(observation.tokenized_prompt_mask),
            "causal": step_first(observation.tokenized_causal),
            "causal_mask": step_first(observation.tokenized_causal_mask),
            "causal_fast": step_first(observation.causal_fast_mask),
            "actions": step_first(actions),
            "step_valid": step_first(observation.seq_step_mask),
            "boundary": step_first(observation.seq_block_boundary),
            "noise": jax.random.normal(noise_rng, (t, b, ah, ad)),
            "time": jax.random.beta(time_rng, 1.5, 1, (t, b)) * 0.999 + 0.001,
        }
        if observation.token_state_mask is not None:
            xs["token_state_mask"] = step_first(observation.token_state_mask)
        if self.simulated_delay is not None:
            delay_rng = jax.random.fold_in(rng, 0x525443)
            xs["delay"] = jax.random.randint(delay_rng, (t, b), 0, self.simulated_delay + 1)
        if quiz:
            n_classes = self.probe_head.out_features
            xs["probe_label"] = step_first(jnp.clip(observation.seq_probe_labels, 0, n_classes - 1))
            xs["probe_act"] = step_first(observation.seq_probe_mask)
            xs["probe_vis"] = step_first(observation.seq_probe_visible & observation.seq_probe_mask)
        if aux_on:
            xs["aux_class"] = step_first(observation.seq_subtask_class)
        if ladder_on:
            xs["evidence_mask"] = step_first(observation.seq_evidence_mask)
            xs["waiting_mask"] = step_first(observation.seq_waiting_mask)
        if v35_on:
            xs["write_mask"] = step_first(observation.seq_write_mask)
            xs["decision_mask"] = step_first(observation.seq_decision_mask)
            xs["read_state_valid"] = step_first(observation.seq_read_state_valid)
            xs["read_credit_reachable"] = step_first(observation.seq_read_credit_reachable)
            xs["decay_gap_before"] = step_first(observation.seq_decay_gap_before)
            xs["use_pressure_mask"] = step_first(observation.seq_use_pressure_mask)

        # Per-sample (step-invariant) inputs are closed over rather than scanned.
        segment_masked = observation.seq_state_masked if mask_state else None
        if ladder_on or v35_on:
            side_label = observation.seq_side_label
            side_ok = (side_label >= 0) & (side_label < 2)
            safe_side = jnp.clip(side_label, 0, 1)
        if aux_on:
            aux_classes = self.memory_aux_head.out_features
            # Frame-invariant auxiliary queries (plan 5.1): L2-normalized bank, broadcast per
            # sample; the "hidden" A/B routes a hidden-space bank through project_q instead.
            bank = self.memory_aux_queries.value
            bank = bank * jax.lax.rsqrt(jnp.sum(jnp.square(bank), axis=-1, keepdims=True) + 1e-12)
            bank_b = jnp.broadcast_to(bank[None], (b, *bank.shape))
            if getattr(self, "memory_aux_query_space", "key") == "hidden":
                aux_query_key = self.memory.project_q(bank_b)
            else:
                aux_query_key = bank_b

            def aux_logits_from(state):
                r_aux = self.memory.read_key(state, aux_query_key)
                feats = jnp.mean(r_aux, axis=1)
                feats = feats * jax.lax.rsqrt(jnp.sum(jnp.square(feats), axis=-1, keepdims=True) + 1e-12)
                return self.memory_aux_head(feats).astype(jnp.float32)

            if margin_on:
                # Reset-memory baseline for the margin variant; the ENTIRE M_0 branch is
                # stop-gradient'ed so the objective cannot be gamed by degrading the baseline.
                baseline_logp = jax.lax.stop_gradient(
                    jax.nn.log_softmax(aux_logits_from(self.memory.init_state(b)), axis=-1)
                )

        if v35_on:

            def v35_side_outputs(head, feature, active):
                feature32 = feature.astype(jnp.float32)
                clip_limit = getattr(self, "memory_side_feature_cotangent_clip", None)
                activef = active.astype(jnp.float32)
                if clip_limit is None:
                    head_logits = head(feature32).astype(jnp.float32)
                    ce_value = -jnp.take_along_axis(
                        jax.nn.log_softmax(head_logits, axis=-1), safe_side[:, None], axis=-1
                    )[:, 0]
                else:
                    ce_value, head_logits = _side_ce_with_per_term_feature_cap(
                        feature32,
                        head.kernel.value.astype(jnp.float32),
                        head.bias.value.astype(jnp.float32),
                        safe_side,
                        clip_limit,
                    )

                # This is exactly the unweighted per-term feature gradient that the custom VJP
                # caps before episode/cell reduction and the 0.3 branch weight are applied.
                probability_error = jax.nn.softmax(head_logits, axis=-1) - jax.nn.one_hot(
                    safe_side, 2, dtype=jnp.float32
                )
                feature_grad = probability_error @ head.kernel.value.astype(jnp.float32).T
                feature_grad_norm = jnp.linalg.norm(feature_grad, axis=-1)
                would_bind = (
                    jnp.zeros_like(activef)
                    if clip_limit is None
                    else (feature_grad_norm > clip_limit).astype(jnp.float32) * activef
                )
                return {
                    "ce": ce_value * activef,
                    "correct": (jnp.argmax(head_logits, axis=-1) == safe_side).astype(jnp.float32) * activef,
                    "count": activef,
                    "logits": head_logits,
                    "feature_grad_norm": feature_grad_norm * activef,
                    "feature_clip_would_bind": would_bind,
                }

        def step(carry, x):
            if v35_on:
                state, runtime_state_valid, runtime_credit_reachable = carry
            else:
                state = carry
            boundary_active = x["boundary"] & x["step_valid"] if v35_on else x["boundary"]
            state = jax.tree.map(
                lambda s: jnp.where(boundary_active.reshape((b,) + (1,) * (s.ndim - 1)), jax.lax.stop_gradient(s), s),
                state,
            )
            if v35_on:
                runtime_credit_reachable = jnp.where(
                    boundary_active, jnp.zeros_like(runtime_credit_reachable), runtime_credit_reachable
                )
                # Sparse skip-O semantics: omitted write-free transitions happen before the
                # current read. Invalid/padded gaps fail closed to an exact no-op and are
                # surfaced below rather than silently changing memory state.
                raw_gap = x["decay_gap_before"]
                gap_value_valid = raw_gap >= 0
                gap_apply = x["step_valid"] & gap_value_valid & (raw_gap > 0)
                safe_gap = jnp.where(gap_apply, raw_gap, jnp.zeros_like(raw_gap))
                gap_state, _gap_aux = self.memory.analytic_decay(state, safe_gap)
                state = jax.tree.map(
                    lambda decayed, original: jnp.where(
                        gap_apply.reshape((b,) + (1,) * (decayed.ndim - 1)), decayed, original
                    ),
                    gap_state,
                    state,
                )
            obs_k = _model.Observation(
                images=x["images"],
                image_masks={k: jnp.ones(b, dtype=bool) for k in x["images"]},
                state=x["state"],
                tokenized_prompt=x["tokens"],
                tokenized_prompt_mask=x["token_mask"],
            )
            prefix_tokens, prefix_mask, prefix_ar = self.embed_prefix(obs_k)
            num_img = prefix_mask.shape[1] - self.max_token_len
            top_tokens = num_img // len(x["images"])
            prefix_len = prefix_mask.shape[1]
            mem_len = self.memory_query_tokens
            state_token_mask = x.get("token_state_mask")
            if mask_state:
                masked_prefix_tokens = self._v32_apply_state_null(
                    prefix_tokens, prefix_mask, state_token_mask, segment_masked
                )
            else:
                masked_prefix_tokens = prefix_tokens
            prepared = self._v32_prepare_memory_prefix(
                masked_prefix_tokens,
                prefix_mask,
                prefix_ar,
                state,
                top_token_count=top_tokens,
                state_token_mask=state_token_mask,
            )
            if dual_view:
                # Plan 5.2 gold-standard variant: memory-state evolution from the FULL view
                # (deployment-identical write dynamics); CE/flow and their own retrieval from
                # the masked forward above. Early blocks only -- no second late/causal pass.
                full_prepared = self._v32_prepare_memory_interface(
                    prefix_tokens,
                    prefix_mask,
                    prefix_ar,
                    state,
                    top_token_count=top_tokens,
                    state_token_mask=state_token_mask,
                )
                write_tokens = full_prepared["write_tokens"]
            else:
                write_tokens = prepared["write_tokens"]
            kv_cache = prepared["cache"]
            final_prefix = prepared["final_prefix"]
            read_queries = prepared["read_queries"]

            causal_mask_k = x["causal_mask"]
            causal_emb = self.PaliGemma.llm(x["causal"], method="embed")
            causal_positions = jnp.broadcast_to(prefix_len + mem_len + jnp.arange(causal_len)[None], (b, causal_len))
            (causal_out, _), kv_cache = self.PaliGemma.llm(
                [causal_emb, None],
                mask=self._v32_causal_mask(prefix_mask, causal_mask_k),
                positions=causal_positions,
                kv_cache=kv_cache,
                cache_position=prefix_len + mem_len,
            )
            ce_hidden = jnp.concatenate([self._v32_causal_seed(final_prefix, prefix_mask), causal_out[:, :-1]], axis=1)
            logits = self.PaliGemma.llm(ce_hidden, method="decode").astype(jnp.float32)
            token_logp = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), x["causal"][..., None], axis=-1)[
                ..., 0
            ]
            ce = -jnp.sum(token_logp * causal_mask_k, axis=-1) / jnp.clip(jnp.sum(causal_mask_k, axis=-1), 1)

            time_k = x["time"]
            rtc_loss_mask = None
            model_time = time_k
            if self.simulated_delay is None:
                x_t = time_k[:, None, None] * x["noise"] + (1 - time_k[:, None, None]) * x["actions"]
            else:
                x_t, model_time, rtc_loss_mask = _rtc.make_noisy_actions(
                    x["actions"], x["noise"], time_k, delay=x["delay"]
                )
            u_t = x["noise"] - x["actions"]
            suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(obs_k, x_t, model_time)
            suffix_positions = prefix_len + mem_len + causal_len + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=self._v32_suffix_mask(prefix_mask, causal_mask_k & ~x["causal_fast"], suffix_mask, suffix_ar),
                positions=suffix_positions,
                kv_cache=jax.lax.stop_gradient(kv_cache),
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -ah:])
            flow_tokens = jnp.mean(jnp.square(v_t - u_t), axis=-1)
            flow = jnp.mean(_rtc.renormalize_flow_loss(flow_tokens, rtc_loss_mask), axis=-1)

            valid = x["step_valid"]
            if v35_on:
                transition_valid = valid & gap_value_valid
                write_requested = x["write_mask"] & transition_valid
                # Both candidates start from the exact state that was read above. In delta
                # mode decay_step is observation-independent and cheap; write computes the
                # pooled association needed by L_write even for non-E rows before selection.
                write_state, write_aux = self.memory.write(state, write_tokens)
                decay_state, _ = self.memory.decay_step(state, write_tokens)

                def select_transition(write_leaf, decay_leaf, old_leaf):
                    shape = (b,) + (1,) * (write_leaf.ndim - 1)
                    valid_leaf = transition_valid.reshape(shape)
                    write_leaf_mask = write_requested.reshape(shape)
                    transitioned = jnp.where(write_leaf_mask, write_leaf, decay_leaf)
                    return jnp.where(valid_leaf, transitioned, old_leaf)

                state = jax.tree.map(select_transition, write_state, decay_state, state)
                commit_success = write_requested & write_aux["commit_applied"]
                next_runtime_state_valid = runtime_state_valid | commit_success
                next_runtime_credit_reachable = runtime_credit_reachable | commit_success

                # D losses are legal only when both the sampler and the actual recurrent
                # state agree that a successful E commit precedes this read. Reachability is
                # deliberately telemetry-only and never masks L_read.
                expected_state_valid = x["read_state_valid"]
                effective_read_state_valid = expected_state_valid & runtime_state_valid
                read_active = x["decision_mask"] & transition_valid & effective_read_state_valid & side_ok
                write_active = write_requested & write_aux["commit_applied"] & side_ok
                task_valid = transition_valid & ~(x["decision_mask"] & ~effective_read_state_valid)
                validf = task_valid.astype(jnp.float32)
            else:
                new_state, write_aux = self.memory.write(state, write_tokens)
                state = jax.tree.map(
                    lambda n, o: jnp.where(valid.reshape((b,) + (1,) * (n.ndim - 1)), n, o), new_state, state
                )
                validf = valid.astype(jnp.float32)
            outputs = {
                "ce": ce * validf,
                "flow": flow * validf,
                "valid": validf,
                # Core-steepness telemetry (v34_run1/2 postmortems): the raw inner write
                # gradient norm ramped ~0.5-2.8 (healthy) -> 45-53 before both explosion
                # cycles. Observation only -- stop-gradient keeps it out of the objective.
                # `where` is intentional: multiplication would leave an invalid padded NaN as
                # NaN (IEEE 0 * NaN), poisoning an otherwise valid logging window.
                "write_grad_norm": jnp.where(
                    valid, jax.lax.stop_gradient(write_aux["grad_norm"]), jnp.zeros_like(write_aux["grad_norm"])
                ),
                # Fraction of valid writes whose raw inner gradient is clipped, plus a severe
                # saturation marker (clip factor < .2 means raw norm > 5 at the configured
                # max_grad_norm=1). These are diagnostics only and cannot affect optimization.
                "write_clip_active": jnp.where(
                    valid,
                    jax.lax.stop_gradient((write_aux["clip_factor"] < 1.0).astype(jnp.float32)),
                    0.0,
                ),
                "write_clip_severe": jnp.where(
                    valid if not v35_on else write_requested,
                    jax.lax.stop_gradient((write_aux["clip_factor"] < 0.2).astype(jnp.float32)),
                    0.0,
                ),
            }

            if v35_on:
                write_side = v35_side_outputs(
                    self.memory_write_side_head,
                    write_aux["pooled_value"],
                    write_active,
                )
                # This is the production raw read: mean the 16 FP32 retrievals before tanh-rms
                # pinning or the cast into the Transformer stream.
                raw_read_feature = jnp.mean(prepared["retrieved"].astype(jnp.float32), axis=1)
                read_side = v35_side_outputs(self.memory_read_side_head, raw_read_feature, read_active)
                for prefix, values in (("v35_write", write_side), ("v35_read", read_side)):
                    for name, value in values.items():
                        outputs[f"{prefix}_{name}"] = value

                decision_valid = x["decision_mask"] & transition_valid
                outputs.update(
                    {
                        "v35_write_eligible": write_requested.astype(jnp.float32),
                        "v35_commit_success": commit_success.astype(jnp.float32),
                        "v35_commit_residual_ratio": jnp.where(
                            commit_success,
                            jax.lax.stop_gradient(write_aux["residual_ratio"]),
                            0.0,
                        ),
                        "v35_commit_relative_residual": jnp.where(
                            commit_success,
                            jax.lax.stop_gradient(write_aux["relative_commit_residual"]),
                            0.0,
                        ),
                        "v35_degenerate_write": (write_requested & ~write_aux["commit_applied"]).astype(jnp.float32),
                        "v35_state_invalid_d": (decision_valid & ~effective_read_state_valid).astype(jnp.float32),
                        "v35_state_valid_mismatch": (
                            decision_valid & (expected_state_valid != runtime_state_valid)
                        ).astype(jnp.float32),
                        "v35_reachable": (read_active & runtime_credit_reachable & x["read_credit_reachable"]).astype(
                            jnp.float32
                        ),
                        "v35_reachable_mismatch": (
                            read_active & (runtime_credit_reachable != x["read_credit_reachable"])
                        ).astype(jnp.float32),
                        "v35_invalid_gap": (valid & ~gap_value_valid).astype(jnp.float32),
                        "v35_padding_gap": ((~valid) & (raw_gap != 0)).astype(jnp.float32),
                        "v35_illegal_write_decision_overlap": (
                            transition_valid & x["write_mask"] & x["decision_mask"]
                        ).astype(jnp.float32),
                        "v35_use_pressure": (x["use_pressure_mask"] & read_active).astype(jnp.float32),
                        "v35_raw_read_rms": jnp.sqrt(jnp.mean(jnp.square(raw_read_feature), axis=-1))
                        * read_active.astype(jnp.float32),
                        "v35_injected_pre_cast_rms": prepared["injected_pre_cast_rms"]
                        * transition_valid.astype(jnp.float32),
                        "v35_injected_post_cast_rms": prepared["injected_post_cast_rms"]
                        * transition_valid.astype(jnp.float32),
                        "v35_transition_valid": transition_valid.astype(jnp.float32),
                    }
                )

            if quiz:
                probe_read = self.memory.read(state, read_queries)
                pooled = jnp.mean(self.memory_gate.value * probe_read, axis=1)
                probe_logits = self.probe_head(pooled).astype(jnp.float32)
                if self.memory_probe_diagnostic:
                    probe_logits = jax.lax.stop_gradient(probe_logits)
                probe_logp = jax.nn.log_softmax(probe_logits, axis=-1)
                actf = x["probe_act"].astype(jnp.float32)
                outputs["probe_ce"] = -jnp.take_along_axis(probe_logp, x["probe_label"][:, None], axis=-1)[:, 0] * actf
                outputs["probe_correct"] = (jnp.argmax(probe_logits, axis=-1) == x["probe_label"]).astype(
                    jnp.float32
                ) * actf
                outputs["probe_act"] = actf
                outputs["probe_vis"] = x["probe_vis"].astype(jnp.float32)

            if aux_on:
                # POST-write read (plan 5.1): same-step credit for the writer; on waiting
                # frames the side-bearing label is decodable from M_t only via evidence-phase
                # writes persisting in the fast weights.
                aux_logits = aux_logits_from(state)
                aux_logp = jax.nn.log_softmax(aux_logits, axis=-1)
                label = x["aux_class"]
                label_ok = (label >= 0) & (label < aux_classes)
                safe_label = jnp.clip(label, 0, aux_classes - 1)
                true_logp = jnp.take_along_axis(aux_logp, safe_label[:, None], axis=-1)[:, 0]
                outputs["aux_ce"] = -true_logp
                outputs["aux_correct"] = (jnp.argmax(aux_logits, axis=-1) == safe_label).astype(jnp.float32)
                outputs["aux_valid"] = (label_ok & valid).astype(jnp.float32)
                outputs["aux_label"] = safe_label
                if margin_on:
                    baseline_true = jnp.take_along_axis(baseline_logp, safe_label[:, None], axis=-1)[:, 0]
                    gamma = getattr(self, "memory_aux_margin_gamma", 1.0)
                    outputs["aux_margin"] = jnp.maximum(0.0, gamma - (true_logp - baseline_true))

            if ladder_on:
                # Rung 1 -- writer content on evidence frames; rung 4 -- standard-read
                # retrieval (pre-write M_{t-1}) on waiting frames. Features are STOP-GRADIENTED
                # (the heads observe; they never train the model).
                writer_feats = jnp.mean(write_tokens.astype(jnp.float32), axis=1)
                writer_feats = writer_feats * jax.lax.rsqrt(
                    jnp.sum(jnp.square(writer_feats), axis=-1, keepdims=True) + 1e-12
                )
                writer_logits = self.ladder_writer_head(jax.lax.stop_gradient(writer_feats)).astype(jnp.float32)
                read_feats = jnp.mean(prepared["retrieved"].astype(jnp.float32), axis=1)
                read_feats = read_feats * jax.lax.rsqrt(jnp.sum(jnp.square(read_feats), axis=-1, keepdims=True) + 1e-12)
                read_logits = self.ladder_read_head(jax.lax.stop_gradient(read_feats)).astype(jnp.float32)
                for name, head_logits, frame_mask in (
                    ("ladder_writer", writer_logits, x["evidence_mask"]),
                    ("ladder_read", read_logits, x["waiting_mask"]),
                ):
                    logp = jax.nn.log_softmax(head_logits, axis=-1)
                    active = (frame_mask & valid & side_ok).astype(jnp.float32)
                    outputs[f"{name}_ce"] = -jnp.take_along_axis(logp, safe_side[:, None], axis=-1)[:, 0] * active
                    outputs[f"{name}_correct"] = (jnp.argmax(head_logits, axis=-1) == safe_side).astype(
                        jnp.float32
                    ) * active
                    outputs[f"{name}_count"] = active

            if v35_on:
                return (state, next_runtime_state_valid, next_runtime_credit_reachable), outputs
            return state, outputs

        initial_state = self.memory.init_state(b)
        initial_carry = (
            (initial_state, jnp.zeros((b,), dtype=bool), jnp.zeros((b,), dtype=bool)) if v35_on else initial_state
        )
        _, ys = jax.lax.scan(jax.checkpoint(step, prevent_cse=False), initial_carry, xs)
        write_valid_count = jnp.sum(ys["valid"], axis=0)
        n_valid = jnp.clip(write_valid_count, 1)
        losses = {
            "flow": jnp.sum(ys["flow"], axis=0) / n_valid,
            "ce": jnp.sum(ys["ce"], axis=0) / n_valid,
            # Preserve raw numerators/counts so train.py can pool exactly across unequal
            # sequence lengths, samples, microbatches, and logging windows.
            "write_grad_norm_sum": jnp.sum(ys["write_grad_norm"], axis=0),
            "write_valid_count": write_valid_count,
            "write_clip_count": jnp.sum(ys["write_clip_active"], axis=0),
            "write_severe_clip_count": jnp.sum(ys["write_clip_severe"], axis=0),
            # Per-sample forms remain useful to offline callers; training telemetry uses the
            # exact raw sums above rather than averaging these unequal-denominator ratios.
            "write_grad_norm_mean": jnp.sum(ys["write_grad_norm"], axis=0) / n_valid,
            "write_grad_norm_max": jnp.max(ys["write_grad_norm"], axis=0),
            "write_clip_fraction": jnp.sum(ys["write_clip_active"], axis=0) / n_valid,
            "write_severe_clip_fraction": jnp.sum(ys["write_clip_severe"], axis=0) / n_valid,
        }
        if quiz:
            losses.update(
                {
                    "probe_ce_sum": jnp.sum(ys["probe_ce"], axis=0),
                    "probe_count": jnp.sum(ys["probe_act"], axis=0),
                    "probe_correct": jnp.sum(ys["probe_correct"], axis=0),
                    "probe_count_visible": jnp.sum(ys["probe_vis"], axis=0),
                    "probe_correct_visible": jnp.sum(ys["probe_correct"] * ys["probe_vis"], axis=0),
                    "probe_correct_grid": jnp.moveaxis(ys["probe_correct"], 0, 1),
                    "probe_active_grid": jnp.moveaxis(ys["probe_act"], 0, 1),
                }
            )
        if aux_on:
            # Per-class sums for the class-balanced macro CE (plan 5.1: frequent phase labels
            # must not dominate). train.py divides by the GLOBAL per-class counts so the macro
            # objective stays exact under gradient accumulation.
            aux_classes = self.memory_aux_head.out_features
            class_onehot = jax.nn.one_hot(ys["aux_label"], aux_classes, dtype=jnp.float32)
            weighted = class_onehot * ys["aux_valid"][..., None]
            losses["aux_ce_class_sum"] = jnp.sum(weighted * ys["aux_ce"][..., None], axis=(0, 1))
            losses["aux_count_class"] = jnp.sum(weighted, axis=(0, 1))
            losses["aux_correct_class"] = jnp.sum(weighted * ys["aux_correct"][..., None], axis=(0, 1))
            if margin_on:
                losses["aux_margin_sum"] = jnp.sum(ys["aux_margin"] * ys["aux_valid"])
                losses["aux_margin_count"] = jnp.sum(ys["aux_valid"])
        if ladder_on:
            for name in ("ladder_writer", "ladder_read"):
                losses[f"{name}_ce_sum"] = jnp.sum(ys[f"{name}_ce"])
                losses[f"{name}_correct"] = jnp.sum(ys[f"{name}_correct"])
                losses[f"{name}_count"] = jnp.sum(ys[f"{name}_count"])
        if v35_on:
            num_cells = self.memory_num_side_cells
            memory_cell = observation.seq_memory_cell
            cell_ok = (memory_cell >= 0) & (memory_cell < num_cells)
            safe_cell = jnp.clip(memory_cell, 0, num_cells - 1)
            cell_onehot = jax.nn.one_hot(safe_cell, num_cells, dtype=jnp.float32)

            for name in ("v35_write", "v35_read"):
                frame_count = jnp.sum(ys[f"{name}_count"], axis=0)
                episode_present = (frame_count > 0) & cell_ok
                episode_presentf = episode_present.astype(jnp.float32)
                episode_ce = jnp.sum(ys[f"{name}_ce"], axis=0) / jnp.maximum(frame_count, 1.0)
                mean_logits = jnp.sum(ys[f"{name}_logits"] * ys[f"{name}_count"][..., None], axis=0) / jnp.maximum(
                    frame_count[:, None], 1.0
                )
                episode_correct = (jnp.argmax(mean_logits, axis=-1) == safe_side).astype(jnp.float32) * episode_presentf
                losses[f"{name}_ce_cell_sum"] = jnp.sum(cell_onehot * (episode_ce * episode_presentf)[:, None], axis=0)
                losses[f"{name}_episode_count_cell"] = jnp.sum(cell_onehot * episode_presentf[:, None], axis=0)
                losses[f"{name}_episode_correct_cell"] = jnp.sum(cell_onehot * episode_correct[:, None], axis=0)
                losses[f"{name}_frame_count"] = jnp.sum(frame_count)
                losses[f"{name}_feature_grad_norm_sum"] = jnp.sum(ys[f"{name}_feature_grad_norm"])
                losses[f"{name}_feature_clip_bind_sum"] = jnp.sum(ys[f"{name}_feature_clip_would_bind"])

            commit_count = jnp.sum(ys["v35_commit_success"])
            read_count = jnp.sum(ys["v35_read_count"])
            transition_count = jnp.sum(ys["v35_transition_valid"])
            losses.update(
                {
                    "v35_write_eligible_count": jnp.sum(ys["v35_write_eligible"]),
                    "v35_commit_success_count": commit_count,
                    "v35_degenerate_write_count": jnp.sum(ys["v35_degenerate_write"]),
                    "v35_commit_residual_ratio_sum": jnp.sum(ys["v35_commit_residual_ratio"]),
                    "v35_commit_residual_ratio_max": jnp.max(ys["v35_commit_residual_ratio"]),
                    "v35_commit_relative_residual_sum": jnp.sum(ys["v35_commit_relative_residual"]),
                    "v35_commit_relative_residual_max": jnp.max(ys["v35_commit_relative_residual"]),
                    "v35_state_invalid_d_count": jnp.sum(ys["v35_state_invalid_d"]),
                    "v35_state_valid_mismatch_count": jnp.sum(ys["v35_state_valid_mismatch"]),
                    "v35_reachable_count": jnp.sum(ys["v35_reachable"]),
                    "v35_reachable_mismatch_count": jnp.sum(ys["v35_reachable_mismatch"]),
                    "v35_read_state_valid_count": read_count,
                    "v35_invalid_gap_count": jnp.sum(ys["v35_invalid_gap"]),
                    "v35_padding_gap_count": jnp.sum(ys["v35_padding_gap"]),
                    "v35_illegal_write_decision_overlap_count": jnp.sum(ys["v35_illegal_write_decision_overlap"]),
                    "v35_use_pressure_count": jnp.sum(ys["v35_use_pressure"]),
                    "v35_invalid_cell_count": jnp.sum(~cell_ok),
                    "v35_raw_read_rms_sum": jnp.sum(ys["v35_raw_read_rms"]),
                    "v35_injected_pre_cast_rms_sum": jnp.sum(ys["v35_injected_pre_cast_rms"]),
                    "v35_injected_post_cast_rms_sum": jnp.sum(ys["v35_injected_post_cast_rms"]),
                    "v35_transition_count": transition_count,
                }
            )
        return losses

    def _compute_sequence_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> dict[str, at.Array]:
        """RoboTTT-style sequence loss: one training sample is T consecutive prediction steps
        of one episode (one step per executed action chunk); every field of `observation` and
        `actions` carries a leading step axis.

        Per step, mirroring `sample_with_memory` exactly (same static layout
        [images | context | memory | causal | suffix], same masks and positions):
          1. prefill the step's frame -> live h_t;
          2. read the memory AS IT STANDS (pre-write, the inference order), append the gated
             memory tokens + teacher-forced causal segment to the cache -> next-token CE over
             subtask+FAST, first token from the last memory token's output;
          3. flow matching behind stop_gradient(kv) with per-step independent noise
             (RoboTTT's "sequence action forcing");
          4. write h_t (v3) or final memory-token c18 (v3.1); padding
             steps are exact no-ops on the state;
          5. optional detached diagnostic probe on the post-write state where the data marks the
             step quizzable. In the legacy nonzero-weight mode only, its CE is exposed for the
             caller to add to the objective.

        The scan is rematerialized per step, so only ONE step's VLM activations are ever alive
        -- GPU memory does not grow with T. Gradient blocks: `seq_block_boundary` cuts backprop
        through the memory state at flagged steps (per-sample, data-driven; the state content
        always flows through). Within a block, every step's prefill receives VLM gradients from
        the block's losses; m0 learns through each sequence's first block.
        """
        if getattr(self, "memory_architecture", "v3_v31") == "v32_layer8_dual_query":
            return self._compute_sequence_loss_v32(rng, observation, actions, train=train)
        b, t = observation.seq_step_mask.shape
        ah, ad = actions.shape[-2:]
        aug_rng, noise_rng, time_rng = jax.random.split(rng, 3)

        images = observation.images
        if train:
            images = self._augment_sequence_images(aug_rng, images)

        causal_len = observation.tokenized_causal.shape[-1]
        quiz = (self.memory_probe_weight > 0 or self.memory_probe_diagnostic) and observation.seq_probe_mask is not None

        def step_first(x):  # [b, t, ...] -> [t, b, ...]
            return jnp.moveaxis(x, 1, 0)

        xs = {
            "images": {k: step_first(v) for k, v in images.items()},
            "state": step_first(observation.state),
            "tokens": step_first(observation.tokenized_prompt),
            "token_mask": step_first(observation.tokenized_prompt_mask),
            "causal": step_first(observation.tokenized_causal),
            "causal_mask": step_first(observation.tokenized_causal_mask),
            "causal_fast": step_first(observation.causal_fast_mask),
            "actions": step_first(actions),
            "step_valid": step_first(observation.seq_step_mask),
            "boundary": step_first(observation.seq_block_boundary),
            "noise": jax.random.normal(noise_rng, (t, b, ah, ad)),
            "time": jax.random.beta(time_rng, 1.5, 1, (t, b)) * 0.999 + 0.001,
        }
        if self.simulated_delay is not None:
            delay_rng = jax.random.fold_in(rng, 0x525443)
            xs["delay"] = jax.random.randint(delay_rng, (t, b), 0, self.simulated_delay + 1)  # inclusive maximum
        if quiz:
            n_classes = self.probe_head.out_features
            xs["probe_label"] = step_first(jnp.clip(observation.seq_probe_labels, 0, n_classes - 1))
            xs["probe_act"] = step_first(observation.seq_probe_mask)
            xs["probe_vis"] = step_first(observation.seq_probe_visible & observation.seq_probe_mask)

        def step(state, x):
            # gradient-block fence: cut backprop through the incoming state where flagged; the
            # state content itself passes through unchanged (where() routes the gradient)
            state = jax.tree.map(
                lambda s: jnp.where(x["boundary"].reshape((b,) + (1,) * (s.ndim - 1)), jax.lax.stop_gradient(s), s),
                state,
            )

            # 1. prefill, identical to the inference path
            obs_k = _model.Observation(
                images=x["images"],
                image_masks={k: jnp.ones(b, dtype=bool) for k in x["images"]},
                state=x["state"],
                tokenized_prompt=x["tokens"],
                tokenized_prompt_mask=x["token_mask"],
            )
            prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs_k)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache, hidden = self.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=positions, return_hidden_states=True
            )
            num_img = prefix_mask.shape[1] - self.max_token_len
            mem_len = num_img // len(x["images"])
            prefix_len = prefix_mask.shape[1]
            h_k = hidden[0][self.memory_layer][:, :mem_len].astype(jnp.float32)

            # 2. read the pre-write memory and run [memory | causal] as one cache extension
            retrieved = self.memory.read(state, h_k)
            mem_tokens = (self.memory_gate.value * retrieved).astype(prefix_tokens.dtype)
            causal_emb = self.PaliGemma.llm(x["causal"], method="embed")
            ext_tokens = jnp.concatenate([mem_tokens, causal_emb], axis=1)
            causal_mask_k = x["causal_mask"]
            mem_rows = make_memory_step_mask(prefix_mask, prefix_ar_mask, mem_len, causal_len)
            tri = jnp.tril(jnp.ones((causal_len, causal_len), dtype=bool))
            causal_rows = jnp.concatenate(
                [
                    einops.repeat(prefix_mask, "b p -> b c p", c=causal_len),
                    jnp.ones((b, causal_len, mem_len), dtype=bool),
                    tri[None] & causal_mask_k[:, None, :],
                ],
                axis=-1,
            )
            ext_mask = jnp.concatenate([mem_rows, causal_rows], axis=1)
            ext_positions = jnp.broadcast_to(
                prefix_len + jnp.arange(mem_len + causal_len)[None], (b, mem_len + causal_len)
            )
            kv_cache = jax.tree.map(
                lambda c: jnp.pad(c, ((0, 0), (0, 0), (0, mem_len + causal_len), (0, 0), (0, 0))), kv_cache
            )
            (ext_out, _), kv_cache = self.PaliGemma.llm(
                [ext_tokens, None],
                mask=ext_mask,
                positions=ext_positions,
                kv_cache=kv_cache,
                cache_position=prefix_len,
            )
            mem_out, causal_out = ext_out[:, :mem_len], ext_out[:, mem_len:]
            c_k = mem_out.astype(jnp.float32)
            ce_hidden = jnp.concatenate([mem_out[:, -1:], causal_out[:, :-1]], axis=1)
            logits = self.PaliGemma.llm(ce_hidden, method="decode").astype(jnp.float32)
            token_logp = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), x["causal"][..., None], axis=-1)[
                ..., 0
            ]
            ce = -jnp.sum(token_logp * causal_mask_k, axis=-1) / jnp.clip(jnp.sum(causal_mask_k, axis=-1), 1)

            # 3. flow matching through the action expert only, per-step independent noise
            time_k = x["time"]
            rtc_loss_mask = None
            model_time = time_k
            if self.simulated_delay is None:
                x_t = time_k[:, None, None] * x["noise"] + (1 - time_k[:, None, None]) * x["actions"]
            else:
                x_t, model_time, rtc_loss_mask = _rtc.make_noisy_actions(
                    x["actions"], x["noise"], time_k, delay=x["delay"]
                )
            u_t = x["noise"] - x["actions"]
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(obs_k, x_t, model_time)
            suffix_view = jnp.concatenate(
                [prefix_mask, jnp.ones((b, mem_len), dtype=bool), causal_mask_k & ~x["causal_fast"]], axis=1
            )
            full_attn_mask = jnp.concatenate(
                [
                    einops.repeat(suffix_view, "b p -> b s p", s=suffix_tokens.shape[1]),
                    make_attn_mask(suffix_mask, suffix_ar_mask),
                ],
                axis=-1,
            )
            suffix_positions = prefix_len + mem_len + causal_len + jnp.cumsum(suffix_mask, axis=-1) - 1
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=jax.lax.stop_gradient(kv_cache),
                adarms_cond=[None, adarms_cond],
            )
            v_t = self.action_out_proj(suffix_out[:, -ah:])
            flow_tokens = jnp.mean(jnp.square(v_t - u_t), axis=-1)
            flow = jnp.mean(_rtc.renormalize_flow_loss(flow_tokens, rtc_loss_mask), axis=-1)

            # 4. write AFTER prediction (inference order); padded steps leave the state
            # bit-identical (no decay tick)
            write_source = self._select_memory_write_source(h_k, c_k)
            new_state, _ = self.memory.write(state, write_source)
            valid = x["step_valid"]
            state = jax.tree.map(
                lambda n, o: jnp.where(valid.reshape((b,) + (1,) * (n.ndim - 1)), n, o), new_state, state
            )

            # 5. optional diagnostic probe on the post-write state. Stop the complete probe
            # computation (memory read, content gate, and classifier) so diagnostic logging
            # cannot create a backward path into the policy/memory or the probe head itself.
            # Reading is pure, so the recurrent state is bit-identical with diagnostics on/off.
            if quiz:
                probe_read = self.memory.read(state, h_k)
                pooled = jnp.mean(self.memory_gate.value * probe_read, axis=1)
                probe_logits = self.probe_head(pooled).astype(jnp.float32)
                if self.memory_probe_diagnostic:
                    probe_logits = jax.lax.stop_gradient(probe_logits)
                probe_logp = jax.nn.log_softmax(probe_logits, axis=-1)
                actf = x["probe_act"].astype(jnp.float32)
                probe_ce = -jnp.take_along_axis(probe_logp, x["probe_label"][:, None], axis=-1)[:, 0] * actf
                probe_correct = (jnp.argmax(probe_logits, axis=-1) == x["probe_label"]).astype(jnp.float32) * actf
                probe_vis = x["probe_vis"].astype(jnp.float32)
            else:
                probe_ce = probe_correct = actf = probe_vis = jnp.zeros((b,), dtype=jnp.float32)

            validf = valid.astype(jnp.float32)
            return state, (ce * validf, flow * validf, validf, probe_ce, probe_correct, actf, probe_vis)

        _, ys = jax.lax.scan(jax.checkpoint(step, prevent_cse=False), self.memory.init_state(b), xs)
        ce_steps, flow_steps, valid_steps, probe_ce, probe_cor, probe_act, probe_vis = ys  # each [t, b]
        n_valid = jnp.clip(jnp.sum(valid_steps, axis=0), 1)
        losses = {"flow": jnp.sum(flow_steps, axis=0) / n_valid, "ce": jnp.sum(ce_steps, axis=0) / n_valid}
        if quiz:
            losses.update(
                {
                    "probe_ce_sum": jnp.sum(probe_ce, axis=0),
                    "probe_count": jnp.sum(probe_act, axis=0),
                    "probe_correct": jnp.sum(probe_cor, axis=0),
                    "probe_count_visible": jnp.sum(probe_vis, axis=0),
                    "probe_correct_visible": jnp.sum(probe_cor * probe_vis, axis=0),
                    # per-STEP quiz stats (position 0 = sequence start); the trainer logs one
                    # accuracy scalar per step index
                    "probe_correct_grid": jnp.moveaxis(probe_cor, 0, 1),
                    "probe_active_grid": jnp.moveaxis(probe_act, 0, 1),
                }
            )
        return losses

    def _augment_sequence_images(self, rng: at.KeyArrayLike, images: dict[str, at.Array]) -> dict[str, at.Array]:
        """Train-time sequence augmentation using the single-frame camera policies.

        Legacy configurations sample every frame independently. v3.5 samples once per
        sample/camera and reuses those parameters across time so augmentation cannot create a
        synthetic temporal cue or jitter the evidence-to-decision trajectory.
        """
        out = {}
        for key, image in images.items():
            b, t = image.shape[:2]
            image01 = image / 2.0 + 0.5
            transforms = []
            if "wrist" not in key:
                height, width = image.shape[2:4]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]
            rng, sub_rng = jax.random.split(rng)
            transform = augmax.Chain(*transforms)
            if getattr(self, "memory_time_consistent_augmentation", False):
                sample_rngs = jax.random.split(sub_rng, b)

                def augment_sample(sample_rng, sequence, transform=transform):
                    # A transform is a pure function of (key, image). Broadcasting one key
                    # over this vmap reuses exactly the same sampled transform over T.
                    return jax.vmap(lambda frame: transform(sample_rng, frame))(sequence)

                augmented = jax.vmap(augment_sample)(sample_rngs, image01)
            else:
                # Preserve the exact legacy random-key geometry when v3.5 is disabled.
                flat = image01.reshape(b * t, *image.shape[2:])
                flat = jax.vmap(transform)(jax.random.split(sub_rng, b * t), flat)
                augmented = flat.reshape(image.shape)
            out[key] = (augmented * 2.0 - 1.0).astype(image.dtype)
        return out
