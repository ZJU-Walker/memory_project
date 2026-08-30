"""Titans-style neural memory (arXiv:2501.00663, "Titans: Learning to Memorize at Test Time").

The memory is the fast weights of a small MLP, one copy per sample. `write` performs one step of
online gradient descent with momentum and forgetting on the associative loss ||M(k) - v||^2
(paper eqs. 11-14), where the key/value/query projections of the current hidden tokens are
    k = L2Norm(SiLU(x W_K)),  q = L2Norm(SiLU(x W_Q)),  v = L2Norm(SiLU(x W_V));
`read` runs the MLP on the query without updating (eq. 15). Keys, queries, and values are all
unit-norm, so both the memory MLP's input scale and the regression-target scale are pinned
regardless of the incoming hidden-state scale: a fresh memory's surprise is exactly 1.0 and
falls toward 0 as content becomes recalled. (Without the value norm, the write budget is spent
on the generic mean component of the targets and nothing frame-specific is stored -- measured.)

There are two disjoint parameter sets:
  * outer (regular nnx params, trained by backprop through writes): W_K / W_V / W_Q, the gate
    head producing (theta, eta, alpha), and the initial fast weights `m0`;
  * inner (`MemoryState`): the per-sample fast weights and their momentum, updated only by
    `write` -- never by the optimizer.

All inner math is float32.

v3.4 additions (V34_PLAN_final.md):
  * Explicit key-space core API (plan 5.10): `project_kv` / `project_q` / `read_key` /
    `write_kv`. The public `read`/`write` are thin wrappers (projection + gates + core), so
    own-key recall, auxiliary-query reads, and synthetic Stage-0 batteries hit exact code
    boundaries instead of double-projecting through the hidden-space interface.
  * `mlp_l2norm` (plan 5.7): unit-L2 normalization of the memory-MLP input and every hidden
    activation (`h_0 = L2Norm(k)`, `h_{l+1} = L2Norm(SiLU(W_l h_l))`, output layer untouched).
    Pins activations O(1) so raw inner gradients are O(1) and the per-write clip stops
    saturating -- the v3.3 replay measured EVERY write saturated (min 3x over the clip, median
    ~500x, worst ~20M x) which made writes constant-size and bloated the fast weights
    exponentially within an episode. With normalization, layer 0 switches from std-1.0 init to
    He (the std-1 compensated lecun shrinkage, which normalization makes moot).
  * `write_kv(..., zero_gradient=True)` / `decay_step` (plan 8.4): the "Dynamics-only"
    retention control -- S_t = eta S_{t-1}, M_t = (1-alpha) M_{t-1} + S_t with the gradient
    term zeroed, isolating passive momentum/forgetting dynamics from new-write interference.
  * Optional drift trust region (`drift_radius`): if ||fast_weights - m0|| exceeds R the drift
    is rescaled to R, direction preserved. A bloat guardrail that does not touch retention.
  * `token_write_diagnostics` is rebuilt on per-token autodiff (plan 5.7): the previous
    analytic backprop hard-coded SiLU-only layers and would silently misreport every
    per-token gradient once L2Norm layers exist. `jax.grad`, one token at a time, stays
    correct under any future MLP change (validated against direct `jax.grad` in tests).
"""

import dataclasses
import functools
import math
from typing import NamedTuple

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


@dataclasses.dataclass(frozen=True)
class MemoryConfig:
    d_input: int = 2048  # gemma_2b width
    d_key: int = 512
    hidden_dims: tuple[int, ...] = (1024, 1024, 1024)
    d_value: int = 2048  # = d_input, so retrieved memories are directly at LLM width

    # Initial gate biases (pre-sigmoid). The gate kernel is zero-init, so until the outer params
    # are trained the gates are exactly these constants.
    theta_bias: float = -2.2  # inner learning rate: sigmoid(-2.2) ~ 0.10
    eta_bias: float = 2.2  # momentum decay: sigmoid(2.2) ~ 0.90
    alpha_bias: float = -4.6  # forgetting rate: sigmoid(-4.6) ~ 0.01

    # Multiplicative intervention on the hidden-token momentum gate. The default returns the
    # learned sigmoid gate unchanged, preserving existing configs and checkpoints bit-for-bit.
    # A zero scale produces an exact zeros_like gate (rather than sigmoid underflow or a very
    # negative bias), while the explicit key-space ``write_kv(..., eta=...)`` API remains an
    # unscaled diagnostic/control boundary.
    eta_scale: float = 1.0

    # Per-sample global-norm clip on the inner gradient. The associative loss (hence gradient)
    # scale follows the value norm, i.e. the incoming hidden-state scale, and a fixed theta
    # diverges to NaN within a few writes once they are mismatched (measured). Below the clip
    # the update is exactly the paper's; above it, the surprise direction is kept and only the
    # step length is bounded.
    max_grad_norm: float = 1.0

    # v3.4 (plan 5.7): unit-L2 normalization inside the memory MLP. False preserves the
    # v3/v3.2/v3.3 forward bit-exactly (their checkpoints replay unchanged).
    mlp_l2norm: bool = False

    # Keep the output layer of a freshly-created per-episode fast state blank, independent of
    # the outer/checkpoint ``m0`` values. The output leaves remain ordinary float32 fast weights
    # and are updated by every inner write; only their episode initializer is fixed. Default-off
    # preserves every existing config and checkpoint exactly.
    blank_initial_output: bool = False

    # v3.4 optional guardrail (plan 5.7): trust region on the per-sample fast-weight drift
    # ||fast_weights - m0||. None disables. When set, a post-update drift exceeding the radius
    # is rescaled onto the sphere (direction preserved) -- bounds bloat without touching the
    # alpha-driven retention dynamics.
    drift_radius: float | None = None

    # Stability guardrail on the OUTER backward pass (v34_run1 postmortem): per-sample clip of
    # the cotangent flowing backward through the recurrent state at each write, i.e. the
    # M_t -> M_{t-1} chain. Outer training can drift the core into a regime where this chain is
    # expansive (v34_run1: ~1.2x per step at ckpt 2750 vs contractive at init, compounding to
    # 1e5+ over a segment and stalling the whole model through the global clip). The clip caps
    # the chain product while preserving its direction -- the long-range credit assignment the
    # plan-5.1 aux demand depends on. Sized so it NEVER binds on healthy chains (measured
    # state-cotangent norms are orders of magnitude below it) and only truncates the unstable
    # tail. None disables (bit-exact backward with pre-fix training).
    state_cotangent_clip: float | None = None

    # Companion guardrail (v34_run2 step-1400 observation): with the state chain capped, the
    # amplified backward escaped through the write's k/v INPUTS into the VLM instead (total
    # grad_norm 48 with the memory group at 3.5). This clips, per sample per write, the
    # cotangent flowing from the write into its projected (k, v) -- i.e. what one write step
    # may send backward toward the write tokens and the tower below them. Same custom-vjp
    # direction-preserving construction. None disables.
    kv_cotangent_clip: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.eta_scale) or not 0.0 <= self.eta_scale <= 1.0:
            raise ValueError(f"eta_scale must be finite and in [0, 1], got {self.eta_scale!r}.")

    @property
    def dims(self) -> tuple[int, ...]:
        """Layer widths of the memory MLP, input to output."""
        return (self.d_key, *self.hidden_dims, self.d_value)


class MemoryState(NamedTuple):
    """Per-sample memory: fast weights of the memory MLP and their momentum (paper's S_t).

    Every leaf is float32 with a leading batch dimension. One state per episode -- create with
    `TitansMemory.init_state` at episode start and thread it through `write` calls.
    """

    fast_weights: dict[str, at.Array]
    momentum: dict[str, at.Array]


def _l2_norm(x: at.Array, eps: float = 1e-6) -> at.Array:
    # rsqrt(sum(x^2) + eps^2) instead of 1 / (norm + eps): the norm's own derivative is
    # x / ||x|| = NaN at exactly-zero inputs, while this form is smooth everywhere.
    return x * jax.lax.rsqrt(jnp.sum(jnp.square(x), axis=-1, keepdims=True) + eps * eps)


def _per_sample(gate: at.Array, leaf: at.Array) -> at.Array:
    """Reshape a [b] gate to broadcast against a [b, ...] weight leaf."""
    return gate.reshape(gate.shape + (1,) * (leaf.ndim - 1))


def _tree_norm(tree) -> at.Array:
    """Per-sample global L2 norm over a [b, ...]-leaved tree."""
    return jnp.sqrt(sum(jnp.sum(jnp.square(g), axis=tuple(range(1, g.ndim))) for g in jax.tree.leaves(tree)))


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _clip_state_cotangent(state_tree, limit: float):
    """Identity forward; backward rescales the per-sample cotangent tree norm to <= limit.

    Applied to the incoming state of every write, so the recurrent backward chain
    M_t -> M_{t-1} passes through exactly one clip per step: any expansive chain product is
    capped at `limit` with its direction preserved. Cotangents of same-step reads join the
    state OUTSIDE the write and are never touched.
    """
    return state_tree


def _clip_state_cotangent_fwd(state_tree, limit: float):
    return state_tree, None


def _clip_state_cotangent_bwd(limit: float, _residual, cotangent):
    scale = jnp.minimum(1.0, limit / (_tree_norm(cotangent) + 1e-12))
    return (jax.tree.map(lambda g: _per_sample(scale, g) * g, cotangent),)


_clip_state_cotangent.defvjp(_clip_state_cotangent_fwd, _clip_state_cotangent_bwd)


class TitansMemory(nnx.Module):
    def __init__(self, config: MemoryConfig, rngs: nnx.Rngs):
        self.config = config
        self.w_k = nnx.Linear(config.d_input, config.d_key, rngs=rngs)
        self.w_v = nnx.Linear(config.d_input, config.d_value, rngs=rngs)
        self.w_q = nnx.Linear(config.d_input, config.d_key, rngs=rngs)

        # Data-dependent gates (theta, eta, alpha), one scalar each per frame. Zero kernel: the
        # gates start as constants and only become data-dependent through outer training. The
        # values are overwritten in place instead of via custom init functions -- nnx stores
        # init fns as static GraphDef attributes, which are compared by identity and would
        # differ between separate constructions (breaking e.g. jit out_shardings matching).
        self.gate = nnx.Linear(config.d_input, 3, rngs=rngs)
        self.gate.kernel.value = jnp.zeros_like(self.gate.kernel.value)
        self.gate.bias.value = jnp.array([config.theta_bias, config.eta_bias, config.alpha_bias], dtype=jnp.float32)

        # Learnable initial fast weights, variance-preserving. Without mlp_l2norm: layer 0 has
        # std 1 (its input is a unit-L2-norm key vector, not a unit-RMS one), hidden layers are
        # He-init for SiLU, and the output layer is zero-init so an unwritten memory reads
        # exactly zero (the read-side integration starts as a no-op). With lecun everywhere the
        # activations shrink ~10x per layer and the memory barely learns (measured). With
        # mlp_l2norm (v3.4, plan 5.7) every hidden activation is renormalized to unit L2, so
        # the std-1.0 compensation is moot and layer 0 uses He like the other hidden layers.
        dims = config.dims
        m0 = {}
        for i in range(len(dims) - 1):
            if i == len(dims) - 2:
                kernel_init = nnx.initializers.zeros_init()
            elif i == 0 and not config.mlp_l2norm:
                kernel_init = nnx.initializers.normal(stddev=1.0)
            else:
                kernel_init = nnx.initializers.he_normal()
            m0[f"w{i}"] = nnx.Param(kernel_init(rngs.params(), (dims[i], dims[i + 1]), jnp.float32))
            m0[f"b{i}"] = nnx.Param(jnp.zeros((dims[i + 1],), dtype=jnp.float32))
        self.m0 = nnx.Dict(**m0)

    @property
    def _num_layers(self) -> int:
        return len(self.config.dims) - 1

    def _initial_fast_leaf(self, name: str) -> at.Array:
        """Effective per-episode initializer for one fast-weight leaf.

        The outer output leaves stay in the parameter/checkpoint tree for compatibility, but
        ``blank_initial_output`` deliberately disconnects them from new episode states. Keeping
        this as the single source of truth also makes the optional trust region use the same
        effective origin as :meth:`init_state`.
        """
        value = self.m0[name].value
        output_layer = self._num_layers - 1
        if self.config.blank_initial_output and name in (f"w{output_layer}", f"b{output_layer}"):
            # zeros_like preserves the parameter leaf's FSDP placement while the explicit dtype
            # keeps the per-sample fast-state contract independent of checkpoint precision.
            return jnp.zeros_like(value, dtype=jnp.float32)
        return value

    def init_state(self, batch_size: int) -> MemoryState:
        """Fresh memory: effective fast-weight initializer broadcast, zero momentum."""
        fast = {}
        for i in range(self._num_layers):
            for name in (f"w{i}", f"b{i}"):
                value = self._initial_fast_leaf(name)
                fast[name] = jnp.broadcast_to(value, (batch_size, *value.shape))
        return MemoryState(fast_weights=fast, momentum=jax.tree.map(jnp.zeros_like, fast))

    def _forward(self, fast_weights: dict[str, at.Array], k: at.Array) -> at.Array:
        """The memory MLP under the given (unbatched) fast weights: [n, d_key] -> [n, d_value].

        With `mlp_l2norm` (plan 5.7) the input and every hidden activation are unit-L2
        normalized (`h_0 = L2Norm(k)`, `h_{l+1} = L2Norm(SiLU(W_l h_l))`); the output layer is
        never normalized so the zero-init "fresh memory reads exactly zero" property survives.
        RMS-norm would NOT work here: it leaves ||h||_2 = sqrt(d) ~ 32 for the 1024-wide
        hiddens, so output-layer gradients stay ~30 and the clip re-saturates.
        """
        l2norm = self.config.mlp_l2norm
        x = _l2_norm(k) if l2norm else k
        for i in range(self._num_layers):
            x = x @ fast_weights[f"w{i}"] + fast_weights[f"b{i}"]
            if i < self._num_layers - 1:
                x = jax.nn.silu(x)
                if l2norm:
                    x = _l2_norm(x)
        return x

    def _keys_values(self, h: at.Array) -> tuple[at.Array, at.Array, at.Array]:
        """Raw float32 tokens x plus the key/value projections (paper eq. 11, unit-norm)."""
        x = h.astype(jnp.float32)
        return x, _l2_norm(jax.nn.silu(self.w_k(x))), _l2_norm(jax.nn.silu(self.w_v(x)))

    # ------------------------------------------------------------------------------------------
    # Explicit key-space core API (plan 5.10). The causal ladder the diagnostics test --
    # projection -> memory core -> reader compatibility -- exists here as exact code boundaries:
    #   writer content   = decode from project_kv outputs (K_t, V_t)
    #   commit           = read_key(M_t, K_t) with the exact keys that participated in the write
    #   standard reader  = read_key(M_t, project_q(h))
    #   aux demand       = read_key(M_t, Q_aux) with a frame-invariant key-space bank
    #   Stage-0          = write_kv -> read_key with synthetic unit (K, V) pairs
    # ------------------------------------------------------------------------------------------

    @at.typecheck
    def project_kv(
        self, h: at.Float[at.Array, "b n d"]
    ) -> tuple[at.Float[at.Array, "b n dk"], at.Float[at.Array, "b n dv"]]:
        """Key/value projections of hidden tokens: k = L2Norm(SiLU(h W_K)), v likewise."""
        _, k, v = self._keys_values(h)
        return k, v

    @at.typecheck
    def project_q(self, h: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, "b n dk"]:
        """Query projection of hidden tokens: q = L2Norm(SiLU(h W_Q))."""
        return _l2_norm(jax.nn.silu(self.w_q(h.astype(jnp.float32))))

    @at.typecheck
    def read_key(
        self, state: MemoryState, q_key: at.Float[at.Array, "b n dk"]
    ) -> at.Float[at.Array, "b n dv"]:
        """Memory-MLP forward on ALREADY-PROJECTED key-space queries (no W_Q projection)."""
        return jax.vmap(self._forward)(state.fast_weights, q_key.astype(jnp.float32))

    def _drift_trust_region(self, fast_weights: dict[str, at.Array]) -> dict[str, at.Array]:
        """Optional guardrail (plan 5.7): rescale ||fast - m0|| onto the drift_radius sphere."""
        radius = self.config.drift_radius
        if radius is None:
            return fast_weights
        m0 = {name: jnp.broadcast_to(self._initial_fast_leaf(name), leaf.shape) for name, leaf in fast_weights.items()}
        drift = jax.tree.map(jnp.subtract, fast_weights, m0)
        drift_norm = _tree_norm(drift)
        # Like the gradient clip, the rescale factor is a stop-gradient optimizer safeguard.
        scale = jax.lax.stop_gradient(jnp.minimum(1.0, radius / (drift_norm + 1e-12)))
        return jax.tree.map(lambda base, d: base + _per_sample(scale, d) * d, m0, drift)

    @at.typecheck
    def write_kv(
        self,
        state: MemoryState,
        k: at.Float[at.Array, "b n dk"],
        v: at.Float[at.Array, "b n dv"],
        theta: at.Float[at.Array, " b"],
        eta: at.Float[at.Array, " b"],
        alpha: at.Float[at.Array, " b"],
        *,
        zero_gradient: bool = False,
    ) -> tuple[MemoryState, dict[str, at.Array]]:
        """One inner update from already-projected key/value pairs (paper eqs. 12-14).

            S_t = eta * S_{t-1} - theta * grad ||M_{t-1}(k) - v||^2      (momentum)
            M_t = (1 - alpha) * M_{t-1} + S_t                            (forgetting)

        The associative loss is AVERAGED over the n tokens (the gradient size must not scale
        with token count) and the gradient is clipped to `max_grad_norm` per sample (global
        norm over all fast weights) before the momentum update.

        ``zero_gradient=True`` is the plan-8.4 "Dynamics-only" retention control: the gates and
        surprise are computed exactly as normal but the gradient term is zero, so
        S_t = eta S_{t-1} and M_t = (1 - alpha) M_{t-1} + S_t -- passive momentum/forgetting
        dynamics with no new content.

        Returns the updated state and per-sample aux: the pre-update prediction error
        ("surprise"), the pre-clip gradient norm, the clip multiplier actually applied, and the
        gates.
        """
        k = k.astype(jnp.float32)
        v = v.astype(jnp.float32)

        if self.config.state_cotangent_clip is not None:
            # Backward-only guardrail on the recurrent chain; the forward values are identical.
            fast, mom = _clip_state_cotangent(
                (state.fast_weights, state.momentum), self.config.state_cotangent_clip
            )
            state = MemoryState(fast_weights=fast, momentum=mom)
        if self.config.kv_cotangent_clip is not None:
            # Backward-only guardrail on what one write may send toward the VLM tokens.
            k, v = _clip_state_cotangent((k, v), self.config.kv_cotangent_clip)

        def loss(fast_weights, k, v):
            # ||M(k) - v||^2 per token (summed over the feature dim, paper eq. 12), averaged
            # over the frame's tokens.
            return jnp.mean(jnp.sum(jnp.square(self._forward(fast_weights, k) - v), axis=-1))

        if zero_gradient:
            surprise = jax.vmap(loss)(state.fast_weights, k, v)
            grads = jax.tree.map(jnp.zeros_like, state.fast_weights)
        else:
            surprise, grads = jax.vmap(jax.value_and_grad(loss))(state.fast_weights, k, v)

        grad_norm = _tree_norm(grads)
        # The clip factor is an optimizer safeguard: outer gradients treat it as a constant
        # (differentiating through sqrt at exactly-zero inner gradients yields inf * 0 = NaN,
        # and near-zero norms would produce exploding second-order terms).
        clip = jax.lax.stop_gradient(jnp.minimum(1.0, self.config.max_grad_norm / (grad_norm + 1e-12)))
        grads = jax.tree.map(lambda g: _per_sample(clip, g) * g, grads)

        momentum = jax.tree.map(lambda s, g: _per_sample(eta, s) * s - _per_sample(theta, g) * g, state.momentum, grads)
        fast_weights = jax.tree.map(lambda w, s: _per_sample(1 - alpha, w) * w + s, state.fast_weights, momentum)
        fast_weights = self._drift_trust_region(fast_weights)
        aux = {
            "surprise": surprise,
            "grad_norm": grad_norm,
            "clip_factor": clip,
            "theta": theta,
            "eta": eta,
            "alpha": alpha,
        }
        return MemoryState(fast_weights, momentum), aux

    @at.typecheck
    def gates(self, h: at.Float[at.Array, "b n d"]) -> tuple[at.Array, at.Array, at.Array]:
        """Per-frame (theta, eta, alpha) from the mean token, exactly as `write` computes them."""
        x = h.astype(jnp.float32)
        return self._gates_from_float_tokens(x)

    def _effective_eta(self, eta: at.Array) -> at.Array:
        """Apply the configured intervention without perturbing either endpoint's numerics."""
        if self.config.eta_scale == 0.0:
            return jnp.zeros_like(eta)
        if self.config.eta_scale == 1.0:
            return eta
        return eta * self.config.eta_scale

    def _gates_from_float_tokens(self, x: at.Array) -> tuple[at.Array, at.Array, at.Array]:
        """Single gate path shared by every hidden-token write/dynamics entry point."""
        g = jax.nn.sigmoid(self.gate(jnp.mean(x, axis=1)))
        return g[:, 0], self._effective_eta(g[:, 1]), g[:, 2]

    @at.typecheck
    def token_write_diagnostics(self, state: MemoryState, h: at.Float[at.Array, "b n d"]) -> dict[str, at.Array]:
        """Measure the individual token contributions to the next associative write.

        This is an offline, read-only diagnostic evaluated against ``state`` *before* the
        write, using exactly the same normalized keys and values as :meth:`write`.  It returns:

        * ``token_error``: ``e_i = ||M(K_i)-V_i||^2``, shape ``[batch, tokens]``;
        * ``token_grad_norm``: ``s_i = ||grad_M e_i||``, shape ``[batch, tokens]``;
        * ``token_mean_loss_grad_norm``: ``s_i / tokens``, the norm of token ``i``'s term in
          the frame-mean gradient used by :meth:`write` (before the common clip/gate scale).

        The per-token gradient is computed by real ``jax.grad`` (plan 5.7): the previous
        analytic backprop assumed SiLU-only layers, and every inserted L2Norm layer adds a
        Jacobian ``(I - x_hat x_hat^T)/||x||`` the analytic path would silently drop --
        corrupting writer-contribution heatmaps while the writer itself works. Tokens are
        processed sequentially (``lax.map``) so only one token's fast-weight gradient is alive
        per sample at a time.

        The write learning-rate gate ``theta`` and global clipping multiply every token's
        current-gradient contribution by the same per-frame scalar, so they do not change the
        relative heatmap.  Momentum and forgetting act on the aggregate state rather than
        selecting tokens.  Individual gradient norms must not be summed to recover the frame
        ``grad_norm`` because different token-gradient vectors can align or cancel.
        """
        _, k, v = self._keys_values(h)

        def per_sample(fast_weights, k_sample, v_sample):
            def one_token(kv):
                k_i, v_i = kv

                def token_error(fw):
                    return jnp.sum(jnp.square(self._forward(fw, k_i[None])[0] - v_i))

                error, grad = jax.value_and_grad(token_error)(fast_weights)
                grad_norm_sq = sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grad))
                return error, jnp.sqrt(grad_norm_sq)

            return jax.lax.map(one_token, (k_sample, v_sample))

        token_error, token_grad_norm = jax.vmap(per_sample)(state.fast_weights, k, v)
        num_tokens = h.shape[1]
        return {
            "token_error": token_error,
            "token_grad_norm": token_grad_norm,
            "token_mean_loss_grad_norm": token_grad_norm / num_tokens,
        }

    @at.typecheck
    def write(self, state: MemoryState, h: at.Float[at.Array, "b n d"]) -> tuple[MemoryState, dict[str, at.Array]]:
        """One associative write of a frame's hidden tokens (paper eqs. 11-14).

        Thin wrapper over the key-space core: project the tokens, compute the data-dependent
        gates from the mean raw token, and delegate to :meth:`write_kv`.
        """
        x, k, v = self._keys_values(h)
        theta, eta, alpha = self._gates_from_float_tokens(x)
        return self.write_kv(state, k, v, theta, eta, alpha)

    @at.typecheck
    def decay_step(
        self, state: MemoryState, h: at.Float[at.Array, "b n d"]
    ) -> tuple[MemoryState, dict[str, at.Array]]:
        """Plan 8.4 "Dynamics-only" step: gates and surprise computed exactly as `write` would,
        but the gradient term is zeroed -- S_t = eta S_{t-1}, M_t = (1-alpha) M_{t-1} + S_t."""
        x, k, v = self._keys_values(h)
        theta, eta, alpha = self._gates_from_float_tokens(x)
        return self.write_kv(state, k, v, theta, eta, alpha, zero_gradient=True)

    @at.typecheck
    def read(self, state: MemoryState, h: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, "b n dv"]:
        """Retrieve without updating (paper eq. 15): M(q) with q = L2Norm(SiLU(x W_Q))."""
        return self.read_key(state, self.project_q(h))

    @at.typecheck
    def surprise(self, state: MemoryState, h: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, " b"]:
        """Prediction error of the current memory on `h`, without writing (equals the `surprise`
        that `write` would report for the same state and input)."""
        _, k, v = self._keys_values(h)
        return jax.vmap(lambda fw, k, v: jnp.mean(jnp.sum(jnp.square(self._forward(fw, k) - v), axis=-1)))(
            state.fast_weights, k, v
        )
