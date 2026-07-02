"""Pi0Memory: pi05 with a Titans/MAC-style online episodic neural memory.

This subclasses `Pi0` and reuses its prefix/suffix embedding and Gemma backbone unchanged. The only
additions are a small memory interface (read/write projections + a 2-layer memory MLP whose weights
are updated *online* within an episode) and a "memory token" that is appended to the prefix so the
(frozen-or-trained) action expert can attend to retrieved memory.

Training uses a **causal split-cost unroll** (see memory_network_plan.md):
  - `compute_loss` receives a sequence of N causal observation frames `[B, N, ...]` plus a single
    action chunk `[B, H, ad]` at the final frame `t`. The first N-1 frames are the episode's write
    frames (fixed-stride, ~3 Hz, from the episode start; front-padded to a static length with
    all-False image masks marking the padding -- see memory_data_loader.py).
  - Each write frame performs a **write-only** memory update (cheap: SigLIP `embed_prefix` + an
    inner `jax.grad` surprise-momentum step, no LLM pass). Writes are masked to no-ops on padded
    frames. All write-frame encodings are wrapped in `stop_gradient`, so SigLIP never backprops
    through write frames even when it is trainable (`freeze_vision=False`).
  - **Truncated BPTT**: only the last `mem_bptt_steps` writes are differentiated by the outer loss
    (Python loop); all earlier writes run forward-only inside a `jax.lax.scan` with the carry
    stop-gradded at the boundary. This bounds activation memory independently of episode length.
  - The final frame reads the accumulated memory, appends the memory token to the prefix, and runs
    the single full LLM/action-expert forward to produce the flow-matching loss.

The outer `nnx.value_and_grad` differentiates the last-K writes + final read/forward into the slow
params: `mem_{q,k,v}_proj`, `mem_out_proj`, the learned memory init `M_0`, and the base model
(including SigLIP -- via the final action frame only -- unless the freeze filter excludes it).
"""

import logging

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0 import Pi0
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# Defined at module level so the initializer's function identity is stable across model
# instantiations. Creating it inside __init__ yields a fresh closure each time, which makes the nnx
# graphdef differ between jax.eval_shape and the real init -> pjit out_shardings mismatch at train init.
_MEM_ZERO_INIT = nnx.initializers.zeros_init()


def mem_apply(mem: dict, z: at.Array) -> at.Array:
    """Apply the 2-layer per-example memory MLP. `mem` weights carry a leading batch dim B.

    z: [B, d_mem] -> [B, d_mem]. Each example's output depends only on its own weights (independent
    across the batch), which is what makes the per-example online update well-defined.
    """
    h = jnp.einsum("bi,bih->bh", z, mem["w1"]) + mem["b1"]
    h = nnx.swish(h)
    return jnp.einsum("bh,bho->bo", h, mem["w2"]) + mem["b2"]


class Pi0Memory(Pi0):
    def __init__(self, config: "pi0_config.Pi0MemoryConfig", rngs: nnx.Rngs):
        super().__init__(config, rngs)

        self.embed_dim = _gemma.get_config(config.paligemma_variant).width  # 2048 for gemma_2b
        self.d_mem = config.d_mem
        self.d_hidden = config.d_hidden
        # Online-update hyperparameters (fixed scalars; could become a learned GateMLP later).
        self.mem_eta = config.mem_eta
        self.mem_theta = config.mem_theta
        self.mem_alpha = config.mem_alpha
        # Training-only: how many trailing write steps the outer loss differentiates (TBPTT depth).
        self.mem_bptt_steps = config.mem_bptt_steps

        # Read/write projections (W_Q, W_K, W_V): pooled VLA encoding (2048) -> d_mem.
        self.mem_q_proj = nnx.Linear(self.embed_dim, self.d_mem, rngs=rngs)
        self.mem_k_proj = nnx.Linear(self.embed_dim, self.d_mem, rngs=rngs)
        self.mem_v_proj = nnx.Linear(self.embed_dim, self.d_mem, rngs=rngs)
        # Projects the retrieved memory r_t into a single prefix "memory token". Zero-init the kernel so
        # the token is *exactly* zero at step 0 and training starts as stock pi05 (no base-skill
        # regression from an untrained memory token perturbing the policy). Zero-init does not
        # dead-path training: the kernel's gradient (retrieved memory input x upstream signal) is
        # nonzero, so the token grows as soon as memory content helps the action loss. (A sigmoid gate
        # with a -4 bias was tried here and removed: it scaled every gradient into the memory
        # subsystem by ~0.02 and the memory never trained.)
        self.mem_out_proj = nnx.Linear(self.d_mem, self.embed_dim, kernel_init=_MEM_ZERO_INIT, rngs=rngs)

        # Learned memory initialization M_0 (slow params, trained by the outer loss). Small-random
        # weights so the initial read r_t is small-but-nonzero (keeps gradients alive) and the
        # surprise gradient is non-trivial; biases start at zero.
        key_w1, key_w2 = jax.random.split(rngs.params())
        self.mem_w1 = nnx.Param(0.02 * jax.random.normal(key_w1, (self.d_mem, self.d_hidden)))
        self.mem_b1 = nnx.Param(jnp.zeros((self.d_hidden,)))
        # w2's fan-in is d_hidden, so a flat 0.02 over-scales the layer-2 pre-activations (and shifts
        # the inner-surprise-grad conditioning) as d_hidden grows. Scale by sqrt(4096/d_hidden) so the
        # init matches the original 0.02 at the v1 width (4096) and shrinks for wider memories.
        w2_std = 0.02 * (4096.0 / self.d_hidden) ** 0.5
        self.mem_w2 = nnx.Param(w2_std * jax.random.normal(key_w2, (self.d_hidden, self.d_mem)))
        self.mem_b2 = nnx.Param(jnp.zeros((self.d_mem,)))

    # ------------------------------------------------------------------ memory helpers

    def mem_init(self, batch_size: int) -> dict:
        """Broadcast the learned M_0 params into a per-example online-memory carry."""
        m0 = {
            "w1": self.mem_w1.value,
            "b1": self.mem_b1.value,
            "w2": self.mem_w2.value,
            "b2": self.mem_b2.value,
        }
        return jax.tree.map(lambda p: jnp.broadcast_to(p, (batch_size, *p.shape)), m0)

    def _pool_prefix(self, tokens: at.Array, mask: at.Array) -> at.Array:
        """Masked mean-pool the prefix tokens into a single per-example vector x_t (float32)."""
        w = mask.astype(jnp.float32)[..., None]
        x = (tokens.astype(jnp.float32) * w).sum(axis=1) / jnp.clip(w.sum(axis=1), a_min=1e-6)
        return x

    def _encode(self, obs: _model.Observation) -> at.Array:
        """x_t = pool(embed_prefix(obs)). SigLIP + prompt-embed only (no LLM)."""
        tokens, mask, _ = self.embed_prefix(obs)
        return self._pool_prefix(tokens, mask)

    def _surprise_update(self, mem: dict, surprise: dict, k: at.Array, v: at.Array) -> tuple[dict, dict]:
        """One online write step: G = grad_M ||M(k) - v||^2 ; S = eta*S - theta*G ; M = (1-alpha)*M + S."""

        def mem_loss(m: dict) -> at.Array:
            pred = mem_apply(m, k)
            # Sum over batch (examples are independent) so grad wrt M[b] is exactly d(loss_b)/dM[b].
            return jnp.sum(jnp.mean(jnp.square(pred - v), axis=-1))

        grad = jax.grad(mem_loss)(mem)
        surprise = jax.tree.map(lambda s, g: self.mem_eta * s - self.mem_theta * g, surprise, grad)
        mem = jax.tree.map(lambda m, s: (1.0 - self.mem_alpha) * m + s, mem, surprise)
        return mem, surprise

    def memory_write(self, observation: _model.Observation, mem: dict, surprise: dict) -> tuple[dict, dict]:
        """Public single-step write update for stateful test-time use (one real observation)."""
        observation = _model.preprocess_observation(None, observation, train=False)
        x = self._encode(observation).astype(jnp.float32)
        return self._surprise_update(mem, surprise, self.mem_k_proj(x), self.mem_v_proj(x))

    def _read_memory_token(self, prefix_tokens: at.Array, prefix_mask: at.Array, mem: dict) -> at.Array:
        """r_t = M(W_Q x_t); memory token = mem_out_proj(r_t), cast to the prefix dtype.

        The token is exactly zero at init (zero-init mem_out_proj kernel), so an untrained memory
        cannot disturb the base policy; its magnitude is learned by the action loss.
        """
        x = self._pool_prefix(prefix_tokens, prefix_mask)
        r = mem_apply(mem, self.mem_q_proj(x))
        token = self.mem_out_proj(r)
        return token.astype(prefix_tokens.dtype)[:, None, :]

    @staticmethod
    def _append_memory_token(prefix_tokens, prefix_mask, prefix_ar_mask, mem_token):
        """Append a single memory token to the prefix (always valid, bidirectional / ar=False)."""
        b = prefix_tokens.shape[0]
        prefix_tokens = jnp.concatenate([prefix_tokens, mem_token], axis=1)
        prefix_mask = jnp.concatenate([prefix_mask, jnp.ones((b, 1), dtype=prefix_mask.dtype)], axis=1)
        prefix_ar_mask = jnp.concatenate([prefix_ar_mask, jnp.zeros((1,), dtype=prefix_ar_mask.dtype)], axis=0)
        return prefix_tokens, prefix_mask, prefix_ar_mask

    # ------------------------------------------------------------------ training

    def _masked_write(
        self,
        mem: dict,
        surprise: dict,
        obs_i: _model.Observation,
        rng_i: at.KeyArrayLike,
        valid_i: at.Array,
        *,
        train: bool,
    ) -> tuple[dict, dict]:
        """One training write step, no-op'ed on padded frames.

        `valid_i` is [B] bool derived from the loader's image masks (padded frames are all-False).
        The frame encoding is stop-gradded so SigLIP (and the prompt embedder) never backprop through
        write frames -- vision trains only via the final action frame. Gradients into mem_k/v_proj
        and the memory itself sit after the stop and are unaffected.
        """
        obs_i = _model.preprocess_observation(rng_i, obs_i, train=train)
        x_i = jax.lax.stop_gradient(self._encode(obs_i)).astype(jnp.float32)
        new_mem, new_surprise = self._surprise_update(mem, surprise, self.mem_k_proj(x_i), self.mem_v_proj(x_i))

        def keep(new: at.Array, old: at.Array) -> at.Array:
            return jnp.where(valid_i.reshape((-1,) + (1,) * (new.ndim - 1)), new, old)

        return jax.tree.map(keep, new_mem, mem), jax.tree.map(keep, new_surprise, surprise)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        # observation fields are [B, N, ...] (N causal frames, front-padded to a static length by the
        # memory data loader); actions is [B, H, ad] at the final frame.
        batch_size = observation.state.shape[0]
        n = observation.state.shape[1]  # static under jit; = n_mem_max + 1
        rngs = jax.random.split(rng, n + 1)

        mem = self.mem_init(batch_size)
        surprise = jax.tree.map(jnp.zeros_like, mem)

        # Padding marker from the loader: padded frames carry all-False image masks.
        valid = jnp.stack(list(observation.image_masks.values()), axis=0).any(axis=0)  # [B, N]

        n_write = n - 1
        k = min(self.mem_bptt_steps, n_write)
        n_fwd = n_write - k

        # Phase A (TBPTT): all but the last k writes run forward-only inside a scan (a Python loop
        # over ~100 SigLIP graphs would blow up compile). Stop-gradding the carry at the boundary
        # means the outer grad never unrolls through these steps, so no activations are kept and the
        # unroll's memory footprint is bounded by k, not by episode length.
        if n_fwd > 0:
            xs = (
                jax.tree.map(lambda a: jnp.moveaxis(a[:, :n_fwd], 1, 0), observation),
                rngs[:n_fwd],
                jnp.moveaxis(valid[:, :n_fwd], 1, 0),
            )

            def write_step(carry, xs_i):
                obs_i, rng_i, valid_i = xs_i
                return self._masked_write(*carry, obs_i, rng_i, valid_i, train=train), None

            (mem, surprise), _ = jax.lax.scan(write_step, (mem, surprise), xs)
            mem, surprise = jax.tree.map(jax.lax.stop_gradient, (mem, surprise))

        # Phase B: the last k writes, differentiated end-to-end (the original unrolled loop).
        for i in range(n_fwd, n_write):
            obs_i = jax.tree.map(lambda a, i=i: a[:, i], observation)
            mem, surprise = self._masked_write(mem, surprise, obs_i, rngs[i], valid[:, i], train=train)

        # Final frame t: read memory + single full forward + flow-matching loss.
        obs_t = jax.tree.map(lambda a: a[:, n - 1], observation)
        obs_t = _model.preprocess_observation(rngs[n - 1], obs_t, train=train)
        return self._action_loss(rngs[n], obs_t, actions, mem)

    def _action_loss(
        self, rng: at.KeyArrayLike, obs: _model.Observation, actions: _model.Actions, mem: dict
    ) -> at.Float[at.Array, "*b ah"]:
        noise_rng, time_rng = jax.random.split(rng)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)
        mem_token = self._read_memory_token(prefix_tokens, prefix_mask, mem)
        prefix_tokens, prefix_mask, prefix_ar_mask = self._append_memory_token(
            prefix_tokens, prefix_mask, prefix_ar_mask, mem_token
        )

        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(obs, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        vel_pred = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return jnp.mean(jnp.square(vel_pred - u_t), axis=-1)

    # ------------------------------------------------------------------ inference

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        mem: dict | None = None,
    ) -> _model.Actions:
        """Flow-matching sampling with the memory token appended to the prefix.

        `mem` is the online-memory carry (defaults to the learned M_0). At deployment, the caller
        maintains `(mem, surprise)` across steps via `memory_write` and passes `mem` in here.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))
        if mem is None:
            mem = self.mem_init(batch_size)

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        mem_token = self._read_memory_token(prefix_tokens, prefix_mask, mem)
        prefix_tokens, prefix_mask, prefix_ar_mask = self._append_memory_token(
            prefix_tokens, prefix_mask, prefix_ar_mask, mem_token
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
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
            x_t, time = carry
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
