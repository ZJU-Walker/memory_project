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

v3.5 Revision-4 opt-in branch:
  * ``delta_output/pooled_frame`` reduces each frame to one normalized association, decays only
    the output matrix, and applies a differentiable direct rank-one residual assignment.
  * Hidden fast leaves remain fixed; output bias and all momentum remain exact zero. Fixed-alpha
    write-free gaps can therefore be collapsed analytically without changing forward dynamics.
  * Pooling, hidden features, fast state, commits, analytic decay, and raw reads stay FP32.
"""

import dataclasses
import functools
import math
from typing import Literal, NamedTuple

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

    # v3.5 (Revision 4): an explicitly opt-in, one-association-per-frame direct-delta rule.
    # The legacy values remain the defaults so existing configs/checkpoints take exactly the
    # same branch and preserve their forward/backward numerics.
    write_rule: Literal["gradient", "delta_output"] = "gradient"
    association_mode: Literal["tokens", "pooled_frame"] = "tokens"
    delta_rate: float = 1.0

    # Fixed forgetting rate per *sampled memory step* (15 raw frames in the v3.5 pilot).  This
    # is deliberately not produced by the learned gate.  Keeping it static makes a skipped
    # write-free interval exactly collapsible to ``(1 - alpha_step) ** n``.
    alpha_step: float = 0.01

    # A degenerate pooled association or hidden feature must not be made to look valid by the
    # epsilon in L2 normalization / the rank-one denominator.  Such samples receive decay only
    # and expose an explicit telemetry bit.
    association_norm_floor: float = 1e-6
    hidden_norm_sq_floor: float = 1e-6

    def __post_init__(self) -> None:
        if not math.isfinite(self.eta_scale) or not 0.0 <= self.eta_scale <= 1.0:
            raise ValueError(f"eta_scale must be finite and in [0, 1], got {self.eta_scale!r}.")
        if self.write_rule not in ("gradient", "delta_output"):
            raise ValueError(f"Unknown write_rule {self.write_rule!r}.")
        if self.association_mode not in ("tokens", "pooled_frame"):
            raise ValueError(f"Unknown association_mode {self.association_mode!r}.")
        if self.write_rule == "gradient" and self.association_mode != "tokens":
            raise ValueError("write_rule='gradient' requires association_mode='tokens'.")
        if self.write_rule == "delta_output" and self.association_mode != "pooled_frame":
            raise ValueError("write_rule='delta_output' requires association_mode='pooled_frame'.")
        if self.write_rule == "delta_output" and self.drift_radius is not None:
            raise ValueError("drift_radius is incompatible with write_rule='delta_output'.")
        if not math.isfinite(self.delta_rate) or not 0.0 <= self.delta_rate <= 1.0:
            raise ValueError(f"delta_rate must be finite and in [0, 1], got {self.delta_rate!r}.")
        if not math.isfinite(self.alpha_step) or not 0.0 <= self.alpha_step < 1.0:
            raise ValueError(f"alpha_step must be finite and in [0, 1), got {self.alpha_step!r}.")
        if not math.isfinite(self.association_norm_floor) or self.association_norm_floor <= 0.0:
            raise ValueError(
                f"association_norm_floor must be finite and positive, got {self.association_norm_floor!r}."
            )
        if not math.isfinite(self.hidden_norm_sq_floor) or self.hidden_norm_sq_floor <= 0.0:
            raise ValueError(f"hidden_norm_sq_floor must be finite and positive, got {self.hidden_norm_sq_floor!r}.")

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
        blank_output = self.config.blank_initial_output and name in (f"w{output_layer}", f"b{output_layer}")
        # Output bias is not part of the v3.5 fast state.  Enforce the invariant even if a
        # grafted checkpoint contains a nonzero legacy b3 and blank_initial_output was omitted.
        delta_output_bias = self.config.write_rule == "delta_output" and name == f"b{output_layer}"
        if blank_output or delta_output_bias:
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
                if self.config.write_rule == "delta_output":
                    # Checkpoint parameters may be stored in BF16, but v3.5 fast state never is.
                    value = value.astype(jnp.float32)
                fast[name] = jnp.broadcast_to(value, (batch_size, *value.shape))
        return MemoryState(fast_weights=fast, momentum=jax.tree.map(jnp.zeros_like, fast))

    @property
    def _output_weight_name(self) -> str:
        return f"w{self._num_layers - 1}"

    @property
    def _output_bias_name(self) -> str:
        return f"b{self._num_layers - 1}"

    def _hidden(self, fast_weights: dict[str, at.Array], k: at.Array) -> at.Array:
        """FP32 memory hidden features immediately before the output matrix.

        This deliberately lives beside (rather than refactoring) :meth:`_forward`: the legacy
        forward retains its exact operation sequence, while the v3.5 branch gets an explicit
        boundary at which only the final matrix is fast.
        """
        x = k.astype(jnp.float32)
        if self.config.mlp_l2norm:
            x = _l2_norm(x)
        # FP32 dtype alone is not enough on Ampere/Hopper GPUs: default matmul precision is
        # TF32, which breaks the exact-arithmetic fast-weight contract (rank-one commit and
        # retrieval residuals inflate from ~1e-7 to ~1e-3).
        with jax.default_matmul_precision("highest"):
            for i in range(self._num_layers - 1):
                w = fast_weights[f"w{i}"].astype(jnp.float32)
                b = fast_weights[f"b{i}"].astype(jnp.float32)
                x = jax.nn.silu(x @ w + b)
                if self.config.mlp_l2norm:
                    x = _l2_norm(x)
        return x.astype(jnp.float32)

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
        # Raw retrieval shares the fast-weight FP32 exact-arithmetic contract; TF32 (the GPU
        # default for float32 matmuls) must not leak in here.
        with jax.default_matmul_precision("highest"):
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
    def read_key(self, state: MemoryState, q_key: at.Float[at.Array, "b n dk"]) -> at.Float[at.Array, "b n dv"]:
        """Memory-MLP forward on ALREADY-PROJECTED key-space queries (no W_Q projection)."""
        fast_weights = state.fast_weights
        if self.config.write_rule == "delta_output":
            # Raw retrieval is an FP32 contract even when the surrounding model/checkpoint is
            # BF16.  A correctly constructed v3.5 state is already FP32; the casts also make
            # restore boundaries robust without perturbing the legacy branch.
            fast_weights = jax.tree.map(lambda leaf: leaf.astype(jnp.float32), fast_weights)
        return jax.vmap(self._forward)(fast_weights, q_key.astype(jnp.float32))

    @at.typecheck
    def hidden_key(self, state: MemoryState, q_key: at.Float[at.Array, "b n dk"]) -> at.Float[at.Array, "b n dh"]:
        """FP32 pre-output features for key-space query-alignment telemetry."""
        fast_weights = jax.tree.map(lambda leaf: leaf.astype(jnp.float32), state.fast_weights)
        return jax.vmap(self._hidden)(fast_weights, q_key.astype(jnp.float32))

    @at.typecheck
    def pool_kv(
        self,
        k: at.Float[at.Array, "b n dk"],
        v: at.Float[at.Array, "b n dv"],
    ) -> dict[str, at.Array]:
        """Reduce a frame's token associations to one safe, normalized FP32 pair.

        The means are formed independently, then L2-normalized.  Unlike ``_l2_norm`` alone,
        this boundary records the true pre-normalization norms and treats a non-finite or
        near-zero mean as invalid.  Invalid vectors become exact zeros; the commit path uses
        ``association_valid`` to perform decay only rather than manufacture a direction from
        the normalization epsilon.
        """
        if k.shape[1] == 0:
            raise ValueError("Cannot pool an empty token axis.")
        k_mean = jnp.mean(k.astype(jnp.float32), axis=1)
        v_mean = jnp.mean(v.astype(jnp.float32), axis=1)
        k_pre_norm = jnp.sqrt(jnp.sum(jnp.square(k_mean), axis=-1))
        v_pre_norm = jnp.sqrt(jnp.sum(jnp.square(v_mean), axis=-1))
        floor = jnp.asarray(self.config.association_norm_floor, dtype=jnp.float32)
        key_valid = jnp.all(jnp.isfinite(k_mean), axis=-1) & jnp.isfinite(k_pre_norm) & (k_pre_norm >= floor)
        value_valid = jnp.all(jnp.isfinite(v_mean), axis=-1) & jnp.isfinite(v_pre_norm) & (v_pre_norm >= floor)
        k_safe = jnp.where(key_valid[:, None], k_mean, jnp.zeros_like(k_mean))
        v_safe = jnp.where(value_valid[:, None], v_mean, jnp.zeros_like(v_mean))
        pooled_key = _l2_norm(k_safe).astype(jnp.float32)
        pooled_value = _l2_norm(v_safe).astype(jnp.float32)
        return {
            "pooled_key": pooled_key,
            "pooled_value": pooled_value,
            "pooled_key_pre_norm": k_pre_norm.astype(jnp.float32),
            "pooled_value_pre_norm": v_pre_norm.astype(jnp.float32),
            "pooled_key_post_norm": jnp.linalg.norm(pooled_key, axis=-1).astype(jnp.float32),
            "pooled_value_post_norm": jnp.linalg.norm(pooled_value, axis=-1).astype(jnp.float32),
            "pooled_key_valid": key_valid,
            "pooled_value_valid": value_valid,
            "association_valid": key_valid & value_valid,
        }

    def _guard_delta_state(self, state: MemoryState) -> MemoryState:
        """Apply the recurrent cotangent guard, then enforce the v3.5 FP32 boundary."""
        if self.config.state_cotangent_clip is not None:
            fast, momentum = _clip_state_cotangent(
                (state.fast_weights, state.momentum), self.config.state_cotangent_clip
            )
            state = MemoryState(fast_weights=fast, momentum=momentum)
        return MemoryState(
            fast_weights=jax.tree.map(lambda leaf: leaf.astype(jnp.float32), state.fast_weights),
            momentum=jax.tree.map(lambda leaf: leaf.astype(jnp.float32), state.momentum),
        )

    def _canonical_delta_state(self, state: MemoryState, w3: at.Array) -> MemoryState:
        """Build a v3.5 state: hidden leaves fixed, b3 and every momentum leaf exact zero."""
        fast_weights = {}
        for name, leaf in state.fast_weights.items():
            if name == self._output_weight_name:
                fast_weights[name] = w3.astype(jnp.float32)
            elif name == self._output_bias_name:
                fast_weights[name] = jnp.zeros_like(leaf, dtype=jnp.float32)
            else:
                fast_weights[name] = leaf.astype(jnp.float32)
        momentum = jax.tree.map(lambda leaf: jnp.zeros_like(leaf, dtype=jnp.float32), fast_weights)
        return MemoryState(fast_weights=fast_weights, momentum=momentum)

    def _delta_decay_factor(self, n_steps: at.Array) -> at.Array:
        """Fixed-alpha FP32 decay factor, excluded from outer differentiation."""
        rho = jnp.asarray(1.0 - self.config.alpha_step, dtype=jnp.float32)
        return jax.lax.stop_gradient(jnp.power(rho, n_steps.astype(jnp.float32)))

    def analytic_decay(self, state: MemoryState, n_steps: int | at.Array) -> tuple[MemoryState, dict[str, at.Array]]:
        """Collapse ``n_steps`` valid non-write transitions exactly in delta-output mode.

        ``n_steps`` may be an integer scalar or one integer per batch sample.  A Python negative
        value is rejected immediately.  A dynamically traced negative value cannot raise from
        compiled JAX code, so it fails closed to a no-op for that sample and reports
        ``decay_gap_valid=False``.  Callers remain responsible for proving that a skipped span
        contains no write/reset/invalid transition.
        """
        if self.config.write_rule != "delta_output":
            raise ValueError("analytic_decay is valid only for write_rule='delta_output'.")
        gap = jnp.asarray(n_steps)
        if not jnp.issubdtype(gap.dtype, jnp.integer):
            raise TypeError(f"n_steps must have integer dtype, got {gap.dtype}.")
        if gap.ndim == 0:
            if isinstance(n_steps, int) and not isinstance(n_steps, bool) and n_steps < 0:
                raise ValueError(f"n_steps must be non-negative, got {n_steps}.")
            gap = jnp.broadcast_to(gap, (next(iter(state.fast_weights.values())).shape[0],))
        elif gap.shape != (next(iter(state.fast_weights.values())).shape[0],):
            raise ValueError(f"n_steps must be an integer scalar or shape [batch], got shape {gap.shape}.")

        state = self._guard_delta_state(state)
        gap_valid = gap >= 0
        safe_gap = jnp.maximum(gap, jnp.zeros_like(gap))
        decay_factor = self._delta_decay_factor(safe_gap)
        old_w3 = state.fast_weights[self._output_weight_name]
        new_w3 = _per_sample(decay_factor, old_w3) * old_w3
        new_state = self._canonical_delta_state(state, new_w3)
        batch_size = old_w3.shape[0]
        alpha = jnp.full((batch_size,), self.config.alpha_step, dtype=jnp.float32)
        zeros = jnp.zeros((batch_size,), dtype=jnp.float32)
        aux = {
            "decay_steps": gap,
            "decay_gap_valid": gap_valid,
            "decay_factor": decay_factor.astype(jnp.float32),
            "commit_applied": jnp.zeros((batch_size,), dtype=jnp.bool_),
            "surprise": zeros,
            "grad_norm": zeros,
            "clip_factor": jnp.ones((batch_size,), dtype=jnp.float32),
            "theta": zeros,
            "eta": zeros,
            "alpha": alpha,
            "delta_rate": jnp.full((batch_size,), self.config.delta_rate, dtype=jnp.float32),
            "delta_w3_norm": zeros,
            "w3_norm": jnp.linalg.norm(new_w3, axis=(-2, -1)).astype(jnp.float32),
            "w3_maxabs": jnp.max(jnp.abs(new_w3), axis=(-2, -1)).astype(jnp.float32),
        }
        return new_state, aux

    @at.typecheck
    def delta_write_kv(
        self,
        state: MemoryState,
        k: at.Float[at.Array, "b n dk"],
        v: at.Float[at.Array, "b n dv"],
    ) -> tuple[MemoryState, dict[str, at.Array]]:
        """Decay then directly commit one pooled association into the output matrix.

        This is a *transition* API.  A causal sequence caller must perform any current-step
        read before invoking it.  The update is fully differentiable through the pooled K/V,
        hidden features, residual, and incoming w3; only fixed optimizer-like scalars are
        stop-gradient.  Degenerate samples perform the decay but no content update.
        """
        if self.config.write_rule != "delta_output" or self.config.association_mode != "pooled_frame":
            raise ValueError("delta_write_kv requires write_rule='delta_output' and association_mode='pooled_frame'.")
        k = k.astype(jnp.float32)
        v = v.astype(jnp.float32)
        state = self._guard_delta_state(state)
        if self.config.kv_cotangent_clip is not None:
            k, v = _clip_state_cotangent((k, v), self.config.kv_cotangent_clip)

        pooled = self.pool_kv(k, v)
        pooled_key = pooled["pooled_key"]
        pooled_value = pooled["pooled_value"]
        hidden = self.hidden_key(state, pooled_key[:, None, :])[:, 0, :]
        hidden_norm_sq = jnp.sum(jnp.square(hidden), axis=-1)

        batch_size = hidden.shape[0]
        alpha = jnp.full((batch_size,), self.config.alpha_step, dtype=jnp.float32)
        rho = self._delta_decay_factor(jnp.ones((batch_size,), dtype=jnp.int32))
        old_w3 = state.fast_weights[self._output_weight_name].astype(jnp.float32)
        decayed_w3 = _per_sample(rho, old_w3) * old_w3

        # Compute raw telemetry first, then sanitize only the arithmetic feeding a candidate
        # update.  This prevents NaN*False from contaminating an otherwise fail-closed sample.
        raw_prediction = jnp.einsum("bh,bhd->bd", hidden, decayed_w3, precision=jax.lax.Precision.HIGHEST)
        raw_pre_residual = pooled_value - raw_prediction
        state_finite = jnp.all(jnp.isfinite(decayed_w3), axis=(-2, -1))
        hidden_finite = jnp.all(jnp.isfinite(hidden), axis=-1) & jnp.isfinite(hidden_norm_sq)
        residual_finite = jnp.all(jnp.isfinite(raw_pre_residual), axis=-1)
        hidden_valid = hidden_finite & (
            hidden_norm_sq >= jnp.asarray(self.config.hidden_norm_sq_floor, dtype=jnp.float32)
        )
        base_valid = pooled["association_valid"] & state_finite & hidden_valid & residual_finite

        hidden_safe = jnp.where(jnp.isfinite(hidden), hidden, jnp.zeros_like(hidden))
        residual_safe = jnp.where(jnp.isfinite(raw_pre_residual), raw_pre_residual, jnp.zeros_like(raw_pre_residual))
        denominator = jnp.where(hidden_valid, hidden_norm_sq, jnp.ones_like(hidden_norm_sq))
        rate = jax.lax.stop_gradient(jnp.asarray(self.config.delta_rate, dtype=jnp.float32))
        candidate_delta = (
            rate
            * jnp.einsum("bh,bd->bhd", hidden_safe, residual_safe, precision=jax.lax.Precision.HIGHEST)
            / denominator[:, None, None]
        )
        delta_finite = jnp.all(jnp.isfinite(candidate_delta), axis=(-2, -1))
        commit_applied = base_valid & delta_finite
        delta_w3 = jnp.where(commit_applied[:, None, None], candidate_delta, jnp.zeros_like(candidate_delta))
        new_w3 = decayed_w3 + delta_w3
        new_state = self._canonical_delta_state(state, new_w3)

        post_prediction = jnp.einsum("bh,bhd->bd", hidden_safe, new_w3, precision=jax.lax.Precision.HIGHEST)
        post_residual = pooled_value - post_prediction
        pre_residual = jnp.where(residual_finite[:, None], raw_pre_residual, residual_safe)
        pre_residual_norm = jnp.linalg.norm(pre_residual, axis=-1).astype(jnp.float32)
        post_residual_norm = jnp.linalg.norm(post_residual, axis=-1).astype(jnp.float32)
        residual_ratio = post_residual_norm / jnp.maximum(pre_residual_norm, jnp.asarray(1e-12, jnp.float32))
        relative_commit_residual = post_residual_norm / (
            pooled["pooled_value_post_norm"] + jnp.asarray(1e-8, jnp.float32)
        )
        zeros = jnp.zeros((batch_size,), dtype=jnp.float32)
        aux = {
            **pooled,
            "hidden": hidden.astype(jnp.float32),
            "hidden_norm": jnp.sqrt(hidden_norm_sq).astype(jnp.float32),
            "hidden_norm_sq": hidden_norm_sq.astype(jnp.float32),
            "hidden_valid": hidden_valid,
            "state_finite": state_finite,
            "pre_residual": pre_residual.astype(jnp.float32),
            "post_residual": post_residual.astype(jnp.float32),
            "pre_residual_norm": pre_residual_norm,
            "post_residual_norm": post_residual_norm,
            # post/pre verifies the delta-rate identity; post/||v_bar|| is the production
            # mixed-precision commit-quality gate.  Keep both definitions explicit.
            "residual_ratio": residual_ratio.astype(jnp.float32),
            "relative_commit_residual": relative_commit_residual.astype(jnp.float32),
            "commit_applied": commit_applied,
            "surprise": jnp.sum(jnp.square(pre_residual), axis=-1).astype(jnp.float32),
            # Compatibility telemetry: this rule has no inner loss gradient or momentum gate.
            "grad_norm": zeros,
            "clip_factor": jnp.ones((batch_size,), dtype=jnp.float32),
            "theta": jnp.full((batch_size,), self.config.delta_rate, dtype=jnp.float32),
            "eta": zeros,
            "alpha": alpha,
            "delta_rate": jnp.full((batch_size,), self.config.delta_rate, dtype=jnp.float32),
            "decay_factor": rho,
            "delta_w3_norm": jnp.linalg.norm(delta_w3, axis=(-2, -1)).astype(jnp.float32),
            "w3_norm": jnp.linalg.norm(new_w3, axis=(-2, -1)).astype(jnp.float32),
            "w3_maxabs": jnp.max(jnp.abs(new_w3), axis=(-2, -1)).astype(jnp.float32),
        }
        return new_state, aux

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
        if self.config.write_rule == "delta_output":
            # The learned gates are intentionally irrelevant in v3.5.  This compatibility
            # dispatch lets existing key-space callers switch rules through static config while
            # the explicit delta_write_kv API makes the causal read-before-transition boundary
            # visible to new sequence code.
            if zero_gradient:
                return self.analytic_decay(state, 1)
            return self.delta_write_kv(state, k, v)

        k = k.astype(jnp.float32)
        v = v.astype(jnp.float32)

        if self.config.state_cotangent_clip is not None:
            # Backward-only guardrail on the recurrent chain; the forward values are identical.
            fast, mom = _clip_state_cotangent((state.fast_weights, state.momentum), self.config.state_cotangent_clip)
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
        if self.config.write_rule != "gradient" or self.config.association_mode != "tokens":
            raise ValueError("token_write_diagnostics is defined only for the gradient/tokens write rule.")
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
        if self.config.write_rule == "delta_output":
            return self.delta_write_kv(state, k, v)
        theta, eta, alpha = self._gates_from_float_tokens(x)
        return self.write_kv(state, k, v, theta, eta, alpha)

    @at.typecheck
    def decay_step(self, state: MemoryState, h: at.Float[at.Array, "b n d"]) -> tuple[MemoryState, dict[str, at.Array]]:
        """Plan 8.4 "Dynamics-only" step: gates and surprise computed exactly as `write` would,
        but the gradient term is zeroed -- S_t = eta S_{t-1}, M_t = (1-alpha) M_{t-1} + S_t."""
        if self.config.write_rule == "delta_output":
            # h is intentionally unused: fixed-alpha output decay is observation-independent.
            return self.analytic_decay(state, 1)
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
        if self.config.write_rule == "delta_output":
            # Functional transition construction does not mutate ``state``; its aux evaluates
            # the required post-decay/pre-commit residual exactly once, including guards.
            return self.delta_write_kv(state, k, v)[1]["surprise"]
        return jax.vmap(lambda fw, k, v: jnp.mean(jnp.sum(jnp.square(self._forward(fw, k) - v), axis=-1)))(
            state.fast_weights, k, v
        )
