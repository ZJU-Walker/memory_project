import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
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
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
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
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
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
            self.causal_token_len = config.causal_token_len
            self.memory_grad_every = config.memory_grad_every
            self.memory_remat_chunk = config.memory_remat_chunk
            self.memory_probe_weight = config.memory_probe_weight
            if config.memory_probe_weight > 0:
                # train-only quiz head: classifies the episode's answer from the mean-pooled
                # gated read output at the probe positions (never used at inference)
                self.probe_head = nnx.Linear(config.memory.d_value, config.memory_probe_classes, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

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
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
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
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
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
        if self.predict_with_memory and observation.tokenized_causal is not None:
            return self._compute_loss_with_memory(rng, observation, actions, train=train)

        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)

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

            return jnp.mean(jnp.square(v_t - u_t), axis=-1)

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

        return {"flow": jnp.mean(jnp.square(v_t - u_t), axis=-1), "ce": ce_loss}

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
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
        return x_0

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
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                preprocessed, x_t, jnp.broadcast_to(time, batch)
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
    ) -> tuple[_model.Actions, _memory.MemoryState, dict[str, at.Array]]:
        """Memory-conditioned fused inference: one prefill + an incremental memory append.

        Static layout: [images 0..num_img | context text (positions unchanged) | memory tokens |
        generated subtask | action suffix]. The prefill is identical to the baseline path and
        also yields the layer-`memory_layer` top-camera hidden states h_t. The memory is read
        with h_t and the retrieved tokens (content-gated, zero-init) are appended to the KV
        cache; the subtask is decoded and the actions denoised against the extended cache; the
        memory is written with h_t only after prediction. Returns (actions, new_memory_state,
        aux) with aux carrying the generated tokens/mask and the write diagnostics.
        """
        assert self.predict_with_memory, "the model was not built with predict_with_memory"
        assert max_decode_steps <= self.causal_token_len
        preprocessed = _model.preprocess_observation(None, observation, train=False)
        batch = preprocessed.state.shape[0]

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
        mem_tokens = (self.memory_gate.value * self.memory.read(memory_state, h_t)).astype(prefix_tokens.dtype)
        kv_cache = jax.tree.map(
            lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, mem_len + causal_len), (0, 0), (0, 0))), kv_cache
        )
        mem_mask = make_memory_step_mask(prefix_mask, prefix_ar, mem_len, causal_len)
        mem_positions = prefix_len + jnp.broadcast_to(jnp.arange(mem_len), (batch, mem_len))
        (mem_out, _), kv_cache = self.PaliGemma.llm(
            [mem_tokens, None], mask=mem_mask, positions=mem_positions, kv_cache=kv_cache, cache_position=prefix_len
        )

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
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                preprocessed, x_t, jnp.broadcast_to(time, batch)
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

        # write only after prediction, with the pass-1 hidden states
        new_state, write_aux = self.memory.write(memory_state, h_t)
        aux = {**write_aux, "tokens": gen_tokens, "token_mask": gen_mask}
        return x_0, new_state, aux

    def _compute_loss_with_memory(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> dict[str, at.Array]:
        """Memory co-training loss, mirroring `sample_with_memory` exactly.

        Layout: [images | context | memory | causal (subtask+FAST) | action suffix] with the same
        static positions as inference. The write window is replayed into a fresh memory (detached
        cache hiddens first, then the live window recomputed with the current VLM), the memory is
        read with the live h_t of the current frame, and:
          * CE (VLM + memory + gate): next-token loss over the causal segment, where the first
            token is predicted from the last memory token's output (the inference readout);
          * quiz probes (memory + gate + probe head; VLM at the grad positions): at every
            `memory_grad_every`-th write position that the data marks quizzable, the memory as it
            stands is read with that frame's hiddens and a linear head classifies the episode's
            answer -- dense retrieval supervision inside the window;
          * flow (action expert only, behind stop_gradient(kv)): the suffix attends to
            [context | memory | causal minus FAST] at the static offset.

        Gradients flow through the FULL write chain (no truncation; the state recursion's
        backward only touches the memory MLP, checkpointed every `memory_remat_chunk` writes).
        The VLM keeps activations only for the every-`memory_grad_every`-th live frames (counted
        from the window end); all other live frames run as explicit no-grad prefills, processed
        in sequential groups so only one group's full layer stack is ever materialized.
        """
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch = actions.shape[0]

        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # call 1: prefix prefill, identical to the inference path, yielding the live h_t
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache, hidden = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions, return_hidden_states=True
        )
        num_img = prefix_mask.shape[1] - self.max_token_len
        mem_len = num_img // len(observation.images)
        prefix_len = prefix_mask.shape[1]
        causal_len = observation.tokenized_causal.shape[1]
        h_t = hidden[0][self.memory_layer][:, :mem_len].astype(jnp.float32)

        # live-window hiddens at the given window indices: fold [b, n] into the batch and run
        # the same prefix forward (unaugmented images -- they emulate past inference-time
        # observations)
        def window_hiddens(idx):
            n = len(idx)
            idx = np.asarray(idx)
            w_obs = _model.Observation(
                images={k: v[:, idx].reshape(batch * n, *v.shape[2:]) for k, v in observation.window_images.items()},
                image_masks={k: jnp.ones(batch * n, dtype=bool) for k in observation.window_images},
                state=jnp.zeros((batch * n, observation.state.shape[-1]), observation.state.dtype),
                tokenized_prompt=observation.window_tokens[:, idx].reshape(batch * n, -1),
                tokenized_prompt_mask=observation.window_token_masks[:, idx].reshape(batch * n, -1),
            )
            w_tokens, w_mask, w_ar = self.embed_prefix(w_obs)
            _, _, w_hidden = self.PaliGemma.llm(
                [w_tokens, None],
                mask=make_attn_mask(w_mask, w_ar),
                positions=jnp.cumsum(w_mask, axis=1) - 1,
                return_hidden_states=True,
            )
            return w_hidden[0][self.memory_layer][:, :mem_len].astype(jnp.float32).reshape(batch, n, mem_len, -1)

        wl = observation.window_tokens.shape[1]
        wc = 0 if observation.memory_hiddens is None else observation.memory_hiddens.shape[1]
        num_writes = wc + wl
        # Static every-`memory_grad_every`-th grid, counted backward from the window end. Under
        # start-padding the real writes form a contiguous suffix ending at the newest slot, so
        # counting from the end keeps the grid aligned with them. These positions keep VLM
        # activations for backward (co-adaptation) and are where the quiz probes sit.
        grid = tuple(p for p in range(num_writes) if (num_writes - 1 - p) % self.memory_grad_every == 0)
        grad_live = tuple(p - wc for p in grid if p >= wc)

        # No-grad pass over the whole live window: sequential groups (lax.map) so only one
        # group's full 18-layer hidden stack is ever materialized, under stop_gradient so XLA
        # neither saves activations nor builds a backward -- this is what makes a long all-live
        # window fit in memory. The grad positions are recomputed below WITH activations and
        # overwrite their slots (bit-identical forward values).
        group = self.memory_remat_chunk if wl % self.memory_remat_chunk == 0 else 1

        def group_hiddens(inputs):
            imgs, toks, tmasks = inputs
            w_obs = _model.Observation(
                images={k: v.reshape(batch * group, *v.shape[2:]) for k, v in imgs.items()},
                image_masks={k: jnp.ones(batch * group, dtype=bool) for k in imgs},
                state=jnp.zeros((batch * group, observation.state.shape[-1]), observation.state.dtype),
                tokenized_prompt=toks.reshape(batch * group, -1),
                tokenized_prompt_mask=tmasks.reshape(batch * group, -1),
            )
            w_tokens, w_mask, w_ar = self.embed_prefix(w_obs)
            _, _, w_hidden = self.PaliGemma.llm(
                [w_tokens, None],
                mask=make_attn_mask(w_mask, w_ar),
                positions=jnp.cumsum(w_mask, axis=1) - 1,
                return_hidden_states=True,
            )
            return w_hidden[0][self.memory_layer][:, :mem_len].astype(jnp.float32).reshape(batch, group, mem_len, -1)

        def to_groups(x):  # [b, wl, ...] -> [wl/group, b, group, ...]
            return jnp.moveaxis(x.reshape(batch, wl // group, group, *x.shape[2:]), 0, 1)

        nograd = jax.lax.stop_gradient(
            jax.lax.map(
                group_hiddens,
                (
                    {k: to_groups(v) for k, v in observation.window_images.items()},
                    to_groups(observation.window_tokens),
                    to_groups(observation.window_token_masks),
                ),
            )
        )
        write_live = jnp.moveaxis(nograd, 1, 0).reshape(batch, wl, mem_len, -1)
        if grad_live:
            write_live = write_live.at[:, np.asarray(grad_live)].set(window_hiddens(grad_live))
        if wc > 0:
            write_seq = jnp.concatenate(
                [jax.lax.stop_gradient(observation.memory_hiddens.astype(jnp.float32)), write_live], axis=1
            )
        else:
            write_seq = write_live  # [b, W, mem_len, emb], oldest first

        # Write chain with per-sample validity (episode starts) and FULL BPTT: no stop_gradient
        # anywhere in the state recursion (m0 receives gradient through init_state), gradient-
        # checkpointed every `memory_remat_chunk` writes. The quiz probes run inline: reading is
        # pure (never modifies the state), so the chain is bit-identical with probes on or off.
        quiz = self.memory_probe_weight > 0 and observation.memory_probe_mask is not None
        chunk = self.memory_remat_chunk if num_writes % self.memory_remat_chunk == 0 else 1
        n_chunks = num_writes // chunk
        grid_mask = np.zeros(num_writes, dtype=bool)
        grid_mask[np.asarray(grid)] = True
        if quiz:
            n_classes = self.probe_head.out_features
            labels = jnp.clip(observation.memory_probe_labels, 0, n_classes - 1)
            active = observation.memory_probe_mask & jnp.asarray(grid_mask)[None, :]
            visible = observation.memory_probe_visible & active
        else:
            labels = jnp.zeros((batch, num_writes), dtype=jnp.int32)
            active = jnp.zeros((batch, num_writes), dtype=bool)
            visible = active

        def chain_step(state, xs):
            h, valid, label, act, vis = xs
            new_state, _ = self.memory.write(state, h)
            state = jax.tree.map(
                lambda n, o: jnp.where(valid.reshape(valid.shape + (1,) * (n.ndim - 1)), n, o),
                new_state,
                state,
            )
            if not quiz:
                zero = jnp.zeros((h.shape[0],), dtype=jnp.float32)
                return state, (zero, zero, zero, zero)
            probe_read = self.memory.read(state, h)
            pooled = jnp.mean(self.memory_gate.value * probe_read, axis=1)
            logits = self.probe_head(pooled).astype(jnp.float32)
            logp = jax.nn.log_softmax(logits, axis=-1)
            actf = act.astype(jnp.float32)
            ce = -jnp.take_along_axis(logp, label[:, None], axis=-1)[:, 0]
            correct = (jnp.argmax(logits, axis=-1) == label).astype(jnp.float32)
            return state, (ce * actf, correct * actf, actf, vis.astype(jnp.float32))

        def chunk_body(state, xs_chunk):
            ys = []
            for i in range(chunk):
                state, y = chain_step(state, jax.tree.map(lambda x: x[i], xs_chunk))  # noqa: B023
                ys.append(y)
            return state, jax.tree.map(lambda *z: jnp.stack(z), *ys)

        def to_chunks(x):  # [b, W, ...] -> [n_chunks, chunk, b, ...]
            return jnp.moveaxis(x, 1, 0).reshape(n_chunks, chunk, *x.shape[:1], *x.shape[2:])

        chain_xs = tuple(
            to_chunks(x) for x in (write_seq, observation.memory_write_mask, labels, active, visible)
        )
        memory_state, chain_ys = jax.lax.scan(
            jax.checkpoint(chunk_body, prevent_cse=False), self.memory.init_state(batch), chain_xs
        )
        # per-step quiz stats back to [b, W] (ys arrive as [n_chunks, chunk, b])
        ce_steps, cor_steps, act_steps, vis_steps = (
            jnp.moveaxis(y.reshape(num_writes, batch), 0, 1) for y in chain_ys
        )

        # read with the live h_t and run [memory | causal] as one extension of the cache,
        # exactly like the inference append + teacher-forced decode
        retrieved = self.memory.read(memory_state, h_t)
        mem_tokens = (self.memory_gate.value * retrieved).astype(prefix_tokens.dtype)
        causal_emb = self.PaliGemma.llm(observation.tokenized_causal, method="embed")
        ext_tokens = jnp.concatenate([mem_tokens, causal_emb], axis=1)

        causal_mask = observation.tokenized_causal_mask
        mem_rows = make_memory_step_mask(prefix_mask, prefix_ar_mask, mem_len, causal_len)
        tri = jnp.tril(jnp.ones((causal_len, causal_len), dtype=bool))
        causal_rows = jnp.concatenate(
            [
                einops.repeat(prefix_mask, "b p -> b c p", c=causal_len),
                jnp.ones((batch, causal_len, mem_len), dtype=bool),
                tri[None] & causal_mask[:, None, :],
            ],
            axis=-1,
        )
        ext_mask = jnp.concatenate([mem_rows, causal_rows], axis=1)
        ext_positions = jnp.broadcast_to(
            prefix_len + jnp.arange(mem_len + causal_len)[None], (batch, mem_len + causal_len)
        )
        kv_cache = jax.tree.map(
            lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, mem_len + causal_len), (0, 0), (0, 0))), kv_cache
        )
        (ext_out, _), kv_cache = self.PaliGemma.llm(
            [ext_tokens, None], mask=ext_mask, positions=ext_positions, kv_cache=kv_cache, cache_position=prefix_len
        )
        mem_out, causal_out = ext_out[:, :mem_len], ext_out[:, mem_len:]

        # next-token CE over the causal segment; token 0 is predicted from the memory readout
        ce_hidden = jnp.concatenate([mem_out[:, -1:], causal_out[:, :-1]], axis=1)
        logits = self.PaliGemma.llm(ce_hidden, method="decode").astype(jnp.float32)
        token_logp = jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1), observation.tokenized_causal[..., None], axis=-1
        )[..., 0]
        ce_loss = -jnp.sum(token_logp * causal_mask, axis=-1) / jnp.clip(jnp.sum(causal_mask, axis=-1), 1)

        # flow matching through the action expert only; the suffix sits at the static offset and
        # sees [context | memory | causal minus the FAST branch]
        kv_cache = jax.lax.stop_gradient(kv_cache)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_view = jnp.concatenate(
            [prefix_mask, jnp.ones((batch, mem_len), dtype=bool), causal_mask & ~observation.causal_fast_mask],
            axis=1,
        )
        prefix_attn_mask = einops.repeat(suffix_view, "b p -> b s p", s=suffix_tokens.shape[1])
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
        suffix_positions = prefix_len + mem_len + causal_len + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens],
            mask=full_attn_mask,
            positions=suffix_positions,
            kv_cache=kv_cache,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        losses = {"flow": jnp.mean(jnp.square(v_t - u_t), axis=-1), "ce": ce_loss}
        if quiz:
            # ce/cor steps are pre-masked by `active`; `visible` is a subset of `active`. The
            # grid slices give per-probe-position stats (position index counts from the window
            # start; the trainer logs them per window-size bucket).
            grid_idx = np.asarray(grid)
            losses.update(
                {
                    "probe_ce_sum": jnp.sum(ce_steps, axis=1),
                    "probe_count": jnp.sum(act_steps, axis=1),
                    "probe_correct": jnp.sum(cor_steps, axis=1),
                    "probe_count_visible": jnp.sum(vis_steps, axis=1),
                    "probe_correct_visible": jnp.sum(cor_steps * vis_steps, axis=1),
                    "probe_correct_grid": cor_steps[:, grid_idx],
                    "probe_active_grid": act_steps[:, grid_idx],
                }
            )
        return losses
