import contextlib
import logging

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


class MemoryQueryCompressor(nnx.Module):
    """Learned query bank that cross-attends to the 256 layer-8 top-camera slots."""

    def __init__(self, *, num_queries: int, width: int, num_heads: int, rngs: nnx.Rngs):
        if num_queries < 1 or num_heads < 1 or width % num_heads:
            raise ValueError("query count/heads must be positive and heads must divide width.")
        self.num_queries = num_queries
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.query_bank = nnx.Param(
            jax.random.normal(rngs.params(), (num_queries, width), dtype=jnp.float32) / jnp.sqrt(width)
        )
        self.query_proj = nnx.Linear(width, width, use_bias=False, rngs=rngs)
        self.key_proj = nnx.Linear(width, width, use_bias=False, rngs=rngs)
        self.value_proj = nnx.Linear(width, width, use_bias=False, rngs=rngs)
        self.output_proj = nnx.Linear(width, width, use_bias=False, rngs=rngs)

    def __call__(self, source: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, "b q d"]:
        if source.ndim != 3 or source.shape[-1] != self.width:
            raise ValueError(f"query compressor expects [batch,tokens,{self.width}]; got {source.shape}.")
        batch = source.shape[0]
        queries = jnp.broadcast_to(self.query_bank.value[None], (batch, self.num_queries, self.width))
        q = self.query_proj(queries).reshape(batch, self.num_queries, self.num_heads, self.head_dim)
        k = self.key_proj(source).reshape(batch, source.shape[1], self.num_heads, self.head_dim)
        v = self.value_proj(source).reshape(batch, source.shape[1], self.num_heads, self.head_dim)
        logits = jnp.einsum("bqhd,bnhd->bhqn", q, k, preferred_element_type=jnp.float32)
        probs = jax.nn.softmax(logits * self.head_dim**-0.5, axis=-1).astype(v.dtype)
        pooled = jnp.einsum("bhqn,bnhd->bqhd", probs, v).reshape(batch, self.num_queries, self.width)
        return self.output_proj(pooled).astype(jnp.float32)


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
                adarms=config.pi05,
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
            # zero-init content gate: an untrained/empty memory injects exactly-zero token content
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
            if config.memory_architecture == "v32_layer8_dual_query":
                self.read_query_compressor = MemoryQueryCompressor(
                    num_queries=config.memory_query_tokens,
                    width=paligemma_config.width,
                    num_heads=config.memory_query_heads,
                    rngs=rngs,
                )
                self.write_query_compressor = MemoryQueryCompressor(
                    num_queries=config.memory_query_tokens,
                    width=paligemma_config.width,
                    num_heads=config.memory_query_heads,
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

    def _v32_prepare_memory_prefix(
        self,
        prefix_tokens: at.Array,
        prefix_mask: at.Array,
        prefix_ar: at.Array,
        memory_state: _memory.MemoryState,
        *,
        top_token_count: int,
        zero_read: bool = False,
    ) -> dict[str, at.Array | _gemma.KVCache]:
        """Run blocks 0..8, form q/z, inject 16 reads, then run blocks 9..17 once."""

        if self.memory_architecture != "v32_layer8_dual_query":
            raise ValueError("the split layer-8 prefix is only defined for v3.2.")
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
        read_queries = self.read_query_compressor(h8_top)
        write_tokens = self.write_query_compressor(h8_top)
        retrieved = self.memory.read(memory_state, read_queries)
        if zero_read:
            retrieved = jnp.zeros_like(retrieved)
        memory_tokens = (self.memory_gate.value * retrieved).astype(prefix_tokens.dtype)

        split_tokens = jnp.concatenate([h8_all, memory_tokens], axis=1)
        split_mask = jnp.concatenate([prefix_mask, jnp.ones((batch, mem_len), dtype=bool)], axis=1)
        split_ar = jnp.concatenate([prefix_ar, jnp.zeros((batch, mem_len), dtype=prefix_ar.dtype)], axis=1)
        late_mask = self._pad_attention_columns(make_attn_mask(split_mask, split_ar), capacity)
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
            "cache": cache,
            "final_prefix": final_prefix,
            "h8_top": h8_top,
            "read_queries": read_queries,
            "write_tokens": write_tokens,
            "retrieved": retrieved,
            "prefix_mask": prefix_mask,
            "prefix_ar": prefix_ar,
            "capacity": jnp.asarray(capacity, dtype=jnp.int32),
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
        allow_write: bool,
    ) -> tuple[_model.Actions, _memory.MemoryState, dict[str, at.Array]]:
        """v3.2 inference: layer-8 dual-query read/write with 16 persistent memory tokens."""

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
        )
        kv_cache = prepared["cache"]
        final_prefix = prepared["final_prefix"]
        write_tokens = prepared["write_tokens"]
        retrieved = prepared["retrieved"]

        def finish(actions, tokens, token_mask, *, extra=None):
            candidate_state, write_aux = self.memory.write(memory_state, write_tokens)
            new_state = candidate_state if allow_write else memory_state
            aux = {
                **write_aux,
                "tokens": tokens,
                "token_mask": token_mask,
                "retrieval_norm": jnp.sqrt(jnp.mean(jnp.square(retrieved.astype(jnp.float32)), axis=(1, 2))),
                "memory_gate_norm": jnp.broadcast_to(jnp.linalg.norm(self.memory_gate.value), (batch,)),
                "read_query_norm": jnp.sqrt(
                    jnp.mean(jnp.square(prepared["read_queries"].astype(jnp.float32)), axis=(1, 2))
                ),
                "write_token_norm": jnp.sqrt(jnp.mean(jnp.square(write_tokens.astype(jnp.float32)), axis=(1, 2))),
                "write_occurred": jnp.full((batch,), allow_write, dtype=bool),
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
            score_hidden = jnp.concatenate([final_prefix[:, -1:], causal_out[:, :-1]], axis=1)
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

        token0 = greedy(final_prefix[:, -1])
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
        """
        assert self.predict_with_memory, "the model was not built with predict_with_memory"
        assert max_decode_steps <= self.causal_token_len
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
                allow_write=allow_write,
            )
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
        """Sequence objective for the layer-8 dual-query v3.2 interface."""

        b, t = observation.seq_step_mask.shape
        ah, ad = actions.shape[-2:]
        aug_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        images = self._augment_sequence_images(aug_rng, observation.images) if train else observation.images
        causal_len = observation.tokenized_causal.shape[-1]
        quiz = (self.memory_probe_weight > 0 or self.memory_probe_diagnostic) and observation.seq_probe_mask is not None

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
        if self.simulated_delay is not None:
            delay_rng = jax.random.fold_in(rng, 0x525443)
            xs["delay"] = jax.random.randint(delay_rng, (t, b), 0, self.simulated_delay + 1)
        if quiz:
            n_classes = self.probe_head.out_features
            xs["probe_label"] = step_first(jnp.clip(observation.seq_probe_labels, 0, n_classes - 1))
            xs["probe_act"] = step_first(observation.seq_probe_mask)
            xs["probe_vis"] = step_first(observation.seq_probe_visible & observation.seq_probe_mask)

        def step(state, x):
            state = jax.tree.map(
                lambda s: jnp.where(x["boundary"].reshape((b,) + (1,) * (s.ndim - 1)), jax.lax.stop_gradient(s), s),
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
            prepared = self._v32_prepare_memory_prefix(
                prefix_tokens, prefix_mask, prefix_ar, state, top_token_count=top_tokens
            )
            kv_cache = prepared["cache"]
            final_prefix = prepared["final_prefix"]
            read_queries = prepared["read_queries"]
            write_tokens = prepared["write_tokens"]

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
            ce_hidden = jnp.concatenate([final_prefix[:, -1:], causal_out[:, :-1]], axis=1)
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

            new_state, _ = self.memory.write(state, write_tokens)
            valid = x["step_valid"]
            state = jax.tree.map(
                lambda n, o: jnp.where(valid.reshape((b,) + (1,) * (n.ndim - 1)), n, o), new_state, state
            )
            if quiz:
                probe_read = self.memory.read(state, read_queries)
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
        ce_steps, flow_steps, valid_steps, probe_ce, probe_cor, probe_act, probe_vis = ys
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
                    "probe_correct_grid": jnp.moveaxis(probe_cor, 0, 1),
                    "probe_active_grid": jnp.moveaxis(probe_act, 0, 1),
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
        """Train-time image augmentation for sequence samples: the [b, T] axes are folded and
        every frame is augmented independently (same transform set as preprocess_observation)."""
        out = {}
        for key, image in images.items():
            b, t = image.shape[:2]
            flat = image.reshape(b * t, *image.shape[2:]) / 2.0 + 0.5
            transforms = []
            if "wrist" not in key:
                height, width = flat.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]
            rng, sub_rng = jax.random.split(rng)
            flat = jax.vmap(augmax.Chain(*transforms))(jax.random.split(sub_rng, b * t), flat)
            out[key] = (flat * 2.0 - 1.0).reshape(image.shape)
        return out
