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
"""

import dataclasses
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

    # Per-sample global-norm clip on the inner gradient. The associative loss (hence gradient)
    # scale follows the value norm, i.e. the incoming hidden-state scale, and a fixed theta
    # diverges to NaN within a few writes once they are mismatched (measured). Below the clip
    # the update is exactly the paper's; above it, the surprise direction is kept and only the
    # step length is bounded.
    max_grad_norm: float = 1.0

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

        # Learnable initial fast weights, variance-preserving: layer 0 has std 1 (its input is a
        # unit-L2-norm key vector, not a unit-RMS one), hidden layers are He-init for SiLU, and
        # the output layer is zero-init so an unwritten memory reads exactly zero (the read-side
        # integration starts as a no-op). With lecun everywhere the activations shrink ~10x per
        # layer and the memory barely learns (measured).
        dims = config.dims
        m0 = {}
        for i in range(len(dims) - 1):
            if i == len(dims) - 2:
                kernel_init = nnx.initializers.zeros_init()
            elif i == 0:
                kernel_init = nnx.initializers.normal(stddev=1.0)
            else:
                kernel_init = nnx.initializers.he_normal()
            m0[f"w{i}"] = nnx.Param(kernel_init(rngs.params(), (dims[i], dims[i + 1]), jnp.float32))
            m0[f"b{i}"] = nnx.Param(jnp.zeros((dims[i + 1],), dtype=jnp.float32))
        self.m0 = nnx.Dict(**m0)

    @property
    def _num_layers(self) -> int:
        return len(self.config.dims) - 1

    def init_state(self, batch_size: int) -> MemoryState:
        """Fresh memory: fast weights broadcast from the learnable init, zero momentum."""
        fast = {}
        for i in range(self._num_layers):
            for name in (f"w{i}", f"b{i}"):
                value = self.m0[name].value
                fast[name] = jnp.broadcast_to(value, (batch_size, *value.shape))
        return MemoryState(fast_weights=fast, momentum=jax.tree.map(jnp.zeros_like, fast))

    def _forward(self, fast_weights: dict[str, at.Array], k: at.Array) -> at.Array:
        """The memory MLP under the given (unbatched) fast weights: [n, d_key] -> [n, d_value]."""
        x = k
        for i in range(self._num_layers):
            x = x @ fast_weights[f"w{i}"] + fast_weights[f"b{i}"]
            if i < self._num_layers - 1:
                x = jax.nn.silu(x)
        return x

    def _keys_values(self, h: at.Array) -> tuple[at.Array, at.Array, at.Array]:
        """Raw float32 tokens x plus the key/value projections (paper eq. 11, unit-norm)."""
        x = h.astype(jnp.float32)
        return x, _l2_norm(jax.nn.silu(self.w_k(x))), _l2_norm(jax.nn.silu(self.w_v(x)))

    def _token_error_and_grad_norm(
        self, fast_weights: dict[str, at.Array], k: at.Array, v: at.Array
    ) -> tuple[at.Array, at.Array]:
        """Associative error and fast-weight gradient norm for every token.

        ``fast_weights`` is one sample's unbatched pre-write state and ``k``/``v`` have a
        leading token axis.  For token ``i`` this returns

            e_i = ||M(k_i) - v_i||^2
            s_i = ||d e_i / d fast_weights||_2.

        The gradient norm is computed analytically from the MLP activations.  For a linear
        layer with input ``a`` and output cotangent ``delta``, the per-token kernel gradient is
        the outer product ``a delta^T``; consequently its squared Frobenius norm is
        ``||a||^2 ||delta||^2``.  Adding the bias contribution gives
        ``(||a||^2 + 1) ||delta||^2`` per layer.  This avoids materializing a
        ``[num_tokens, num_fast_parameters]`` Jacobian (more than 1.2B elements for the default
        256-token memory).
        """
        layer_inputs = []
        preactivations = []
        activation = k
        for layer in range(self._num_layers):
            layer_inputs.append(activation)
            preactivation = activation @ fast_weights[f"w{layer}"] + fast_weights[f"b{layer}"]
            preactivations.append(preactivation)
            activation = jax.nn.silu(preactivation) if layer < self._num_layers - 1 else preactivation

        residual = activation - v
        token_error = jnp.sum(jnp.square(residual), axis=-1)
        delta = 2.0 * residual
        token_grad_norm_sq = jnp.zeros_like(token_error)
        for layer in reversed(range(self._num_layers)):
            input_norm_sq = jnp.sum(jnp.square(layer_inputs[layer]), axis=-1)
            delta_norm_sq = jnp.sum(jnp.square(delta), axis=-1)
            token_grad_norm_sq = token_grad_norm_sq + (input_norm_sq + 1.0) * delta_norm_sq
            if layer > 0:
                delta = delta @ fast_weights[f"w{layer}"].T
                previous_preactivation = preactivations[layer - 1]
                sigmoid = jax.nn.sigmoid(previous_preactivation)
                silu_derivative = sigmoid * (1.0 + previous_preactivation * (1.0 - sigmoid))
                delta = delta * silu_derivative

        # Numerical round-off cannot make a sum of squares meaningfully negative, but max(0)
        # prevents an invalid sqrt if a backend ever introduces a tiny negative fused result.
        return token_error, jnp.sqrt(jnp.maximum(token_grad_norm_sq, 0.0))

    @at.typecheck
    def token_write_diagnostics(self, state: MemoryState, h: at.Float[at.Array, "b n d"]) -> dict[str, at.Array]:
        """Measure the 256 individual token contributions to the next associative write.

        This is an offline, read-only diagnostic evaluated against ``state`` *before* the
        write, using exactly the same normalized keys and values as :meth:`write`.  It returns:

        * ``token_error``: ``e_i = ||M(K_i)-V_i||^2``, shape ``[batch, tokens]``;
        * ``token_grad_norm``: ``s_i = ||grad_M e_i||``, shape ``[batch, tokens]``;
        * ``token_mean_loss_grad_norm``: ``s_i / tokens``, the norm of token ``i``'s term in
          the frame-mean gradient used by :meth:`write` (before the common clip/gate scale).

        The write learning-rate gate ``theta`` and global clipping multiply every token's
        current-gradient contribution by the same per-frame scalar, so they do not change the
        relative heatmap.  Momentum and forgetting act on the aggregate state rather than
        selecting tokens.  Individual gradient norms must not be summed to recover the frame
        ``grad_norm`` because different token-gradient vectors can align or cancel.
        """
        _, k, v = self._keys_values(h)
        token_error, token_grad_norm = jax.vmap(self._token_error_and_grad_norm)(state.fast_weights, k, v)
        num_tokens = h.shape[1]
        return {
            "token_error": token_error,
            "token_grad_norm": token_grad_norm,
            "token_mean_loss_grad_norm": token_grad_norm / num_tokens,
        }

    @at.typecheck
    def write(self, state: MemoryState, h: at.Float[at.Array, "b n d"]) -> tuple[MemoryState, dict[str, at.Array]]:
        """One associative write of a frame's hidden tokens (paper eqs. 11-14).

        The frame's n tokens form the gradient mini-batch of a single inner update:
            S_t = eta * S_{t-1} - theta * grad ||M_{t-1}(k) - v||^2      (momentum)
            M_t = (1 - alpha) * M_{t-1} + S_t                            (forgetting)
        The gradient is clipped to `max_grad_norm` per sample (global norm over all fast
        weights) before the momentum update. Returns the updated state and per-sample aux: the
        pre-update prediction error ("surprise"), the pre-clip gradient norm, and the gates.
        """
        x, k, v = self._keys_values(h)
        gates = jax.nn.sigmoid(self.gate(jnp.mean(x, axis=1)))
        theta, eta, alpha = gates[:, 0], gates[:, 1], gates[:, 2]

        def loss(fast_weights, k, v):
            # ||M(k) - v||^2 per token (summed over the feature dim, paper eq. 12), averaged
            # over the frame's tokens.
            return jnp.mean(jnp.sum(jnp.square(self._forward(fast_weights, k) - v), axis=-1))

        surprise, grads = jax.vmap(jax.value_and_grad(loss))(state.fast_weights, k, v)

        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g), axis=tuple(range(1, g.ndim))) for g in jax.tree.leaves(grads)))
        # The clip factor is an optimizer safeguard: outer gradients treat it as a constant
        # (differentiating through sqrt at exactly-zero inner gradients yields inf * 0 = NaN,
        # and near-zero norms would produce exploding second-order terms).
        clip = jax.lax.stop_gradient(jnp.minimum(1.0, self.config.max_grad_norm / (grad_norm + 1e-12)))
        grads = jax.tree.map(lambda g: _per_sample(clip, g) * g, grads)

        momentum = jax.tree.map(lambda s, g: _per_sample(eta, s) * s - _per_sample(theta, g) * g, state.momentum, grads)
        fast_weights = jax.tree.map(lambda w, s: _per_sample(1 - alpha, w) * w + s, state.fast_weights, momentum)
        aux = {"surprise": surprise, "grad_norm": grad_norm, "theta": theta, "eta": eta, "alpha": alpha}
        return MemoryState(fast_weights, momentum), aux

    @at.typecheck
    def read(self, state: MemoryState, h: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, "b n dv"]:
        """Retrieve without updating (paper eq. 15): M(q) with q = L2Norm(SiLU(x W_Q))."""
        q = _l2_norm(jax.nn.silu(self.w_q(h.astype(jnp.float32))))
        return jax.vmap(self._forward)(state.fast_weights, q)

    @at.typecheck
    def surprise(self, state: MemoryState, h: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, " b"]:
        """Prediction error of the current memory on `h`, without writing (equals the `surprise`
        that `write` would report for the same state and input)."""
        _, k, v = self._keys_values(h)
        return jax.vmap(lambda fw, k, v: jnp.mean(jnp.sum(jnp.square(self._forward(fw, k) - v), axis=-1)))(
            state.fast_weights, k, v
        )
