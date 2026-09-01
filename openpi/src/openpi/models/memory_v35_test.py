"""v3.5 Revision-4 tests for pooled-frame, output-only direct-delta memory."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import memory


def _delta_config(**overrides) -> memory.MemoryConfig:
    values = {
        "d_input": 16,
        "d_key": 8,
        "hidden_dims": (12, 10),
        "d_value": 14,
        "mlp_l2norm": True,
        "blank_initial_output": True,
        "write_rule": "delta_output",
        "association_mode": "pooled_frame",
        "delta_rate": 1.0,
        "alpha_step": 0.01,
    }
    values.update(overrides)
    return memory.MemoryConfig(**values)


def _pairs(config: memory.MemoryConfig, *, batch: int = 2, tokens: int = 16, dtype=jnp.float32):
    # A shared component keeps the pooled means safely away from zero while token variation
    # still exercises actual 16-slot pooling.
    k0 = jax.random.normal(jax.random.key(10), (batch, 1, config.d_key), dtype=jnp.float32)
    v0 = jax.random.normal(jax.random.key(11), (batch, 1, config.d_value), dtype=jnp.float32)
    k = k0 + 0.2 * jax.random.normal(jax.random.key(12), (batch, tokens, config.d_key), dtype=jnp.float32)
    v = v0 + 0.2 * jax.random.normal(jax.random.key(13), (batch, tokens, config.d_value), dtype=jnp.float32)
    return k.astype(dtype), v.astype(dtype)


def _nonzero_state(mem: memory.TitansMemory, *, batch: int = 2) -> memory.MemoryState:
    state = mem.init_state(batch)
    fast = dict(state.fast_weights)
    w3_name = mem._output_weight_name  # noqa: SLF001
    b3_name = mem._output_bias_name  # noqa: SLF001
    fast[w3_name] = 0.15 * jax.random.normal(
        jax.random.key(20),
        fast[w3_name].shape,
        dtype=jnp.float32,
    )
    # Dirty values prove that every transition re-establishes the hard invariants.
    fast[b3_name] = jnp.ones_like(fast[b3_name])
    momentum = jax.tree.map(lambda leaf: jnp.full_like(leaf, 0.25), state.momentum)
    return memory.MemoryState(fast, momentum)


def _replace_w3(state: memory.MemoryState, mem: memory.TitansMemory, w3: jax.Array) -> memory.MemoryState:
    fast = dict(state.fast_weights)
    fast[mem._output_weight_name] = w3  # noqa: SLF001
    return memory.MemoryState(fast, state.momentum)


def _relative_error(actual: jax.Array, expected: jax.Array) -> float:
    return float(jnp.linalg.norm(actual - expected) / jnp.maximum(jnp.linalg.norm(expected), 1e-12))


def test_delta_config_is_default_off_and_rejects_illegal_combinations():
    legacy = memory.MemoryConfig()
    assert legacy.write_rule == "gradient"
    assert legacy.association_mode == "tokens"

    with pytest.raises(ValueError, match="requires association_mode='pooled_frame'"):
        _delta_config(association_mode="tokens")
    with pytest.raises(ValueError, match="requires association_mode='tokens'"):
        memory.MemoryConfig(association_mode="pooled_frame")
    with pytest.raises(ValueError, match="drift_radius is incompatible"):
        _delta_config(drift_radius=1.0)
    with pytest.raises(ValueError, match="delta_rate must be finite and in"):
        _delta_config(delta_rate=1.01)
    with pytest.raises(ValueError, match="alpha_step must be finite and in"):
        _delta_config(alpha_step=1.0)
    with pytest.raises(ValueError, match="association_norm_floor must be finite and positive"):
        _delta_config(association_norm_floor=0.0)
    with pytest.raises(ValueError, match="hidden_norm_sq_floor must be finite and positive"):
        _delta_config(hidden_norm_sq_floor=float("nan"))


def test_exact_commit_uses_decay_first_and_enforces_fast_state_invariants():
    config = _delta_config(alpha_step=0.13)
    mem = memory.TitansMemory(config, rngs=nnx.Rngs(0))
    state = _nonzero_state(mem)
    k, v = _pairs(config)
    pooled = mem.pool_kv(k, v)
    hidden = jax.vmap(mem._hidden)(state.fast_weights, pooled["pooled_key"][:, None, :])[:, 0, :]  # noqa: SLF001
    old_w3 = state.fast_weights[mem._output_weight_name]  # noqa: SLF001
    decayed_w3 = jnp.asarray(1.0 - config.alpha_step, dtype=jnp.float32) * old_w3
    expected_pre_residual = pooled["pooled_value"] - jnp.einsum("bh,bhd->bd", hidden, decayed_w3)
    expected_delta = (
        jnp.einsum("bh,bd->bhd", hidden, expected_pre_residual) / jnp.sum(jnp.square(hidden), axis=-1)[:, None, None]
    )

    new_state, aux = mem.delta_write_kv(state, k, v)
    np.testing.assert_allclose(np.asarray(aux["pre_residual"]), np.asarray(expected_pre_residual), rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(new_state.fast_weights[mem._output_weight_name]),  # noqa: SLF001
        np.asarray(decayed_w3 + expected_delta),
        rtol=1e-6,
        atol=1e-7,
    )
    assert np.all(np.asarray(aux["commit_applied"]))
    assert float(jnp.max(aux["post_residual_norm"])) <= 1e-5
    own_key_read = mem.read_key(new_state, aux["pooled_key"][:, None, :])[:, 0, :]
    np.testing.assert_allclose(own_key_read, aux["pooled_value"], rtol=1e-5, atol=1e-5)

    for name in state.fast_weights:
        if name not in (mem._output_weight_name, mem._output_bias_name):  # noqa: SLF001
            np.testing.assert_array_equal(new_state.fast_weights[name], state.fast_weights[name])
    np.testing.assert_array_equal(new_state.fast_weights[mem._output_bias_name], 0.0)  # noqa: SLF001
    for leaf in jax.tree.leaves(new_state.momentum):
        np.testing.assert_array_equal(leaf, 0.0)


def test_half_rate_closes_exactly_half_the_post_decay_residual():
    config = _delta_config(delta_rate=0.5, alpha_step=0.07)
    mem = memory.TitansMemory(config, rngs=nnx.Rngs(1))
    new_state, aux = mem.delta_write_kv(_nonzero_state(mem), *_pairs(config))
    del new_state
    np.testing.assert_allclose(aux["post_residual"], 0.5 * aux["pre_residual"], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(aux["residual_ratio"], 0.5, rtol=2e-6, atol=2e-6)


def test_near_zero_pool_fails_closed_to_decay_only_with_telemetry():
    config = _delta_config(alpha_step=0.2)
    mem = memory.TitansMemory(config, rngs=nnx.Rngs(2))
    state = _nonzero_state(mem)
    k, v = _pairs(config)
    k = k.at[0].set(jnp.zeros_like(k[0]))

    new_state, aux = mem.delta_write_kv(state, k, v)
    assert not bool(aux["pooled_key_valid"][0])
    assert bool(aux["pooled_value_valid"][0])
    assert not bool(aux["association_valid"][0])
    assert not bool(aux["commit_applied"][0])
    np.testing.assert_array_equal(aux["pooled_key"][0], 0.0)
    np.testing.assert_allclose(
        new_state.fast_weights[mem._output_weight_name][0],  # noqa: SLF001
        jnp.asarray(0.8, dtype=jnp.float32) * state.fast_weights[mem._output_weight_name][0],  # noqa: SLF001
        rtol=0.0,
        atol=0.0,
    )
    assert bool(aux["commit_applied"][1])


def test_near_zero_hidden_fails_closed_to_decay_only_with_telemetry():
    config = _delta_config(alpha_step=0.2)
    mem = memory.TitansMemory(config, rngs=nnx.Rngs(21))
    state = _nonzero_state(mem)
    fast = dict(state.fast_weights)
    last_hidden = mem._num_layers - 2  # noqa: SLF001
    fast[f"w{last_hidden}"] = jnp.zeros_like(fast[f"w{last_hidden}"])
    fast[f"b{last_hidden}"] = jnp.zeros_like(fast[f"b{last_hidden}"])
    state = memory.MemoryState(fast, state.momentum)

    new_state, aux = mem.delta_write_kv(state, *_pairs(config))
    np.testing.assert_array_equal(aux["hidden_norm_sq"], 0.0)
    np.testing.assert_array_equal(aux["hidden_valid"], np.zeros((2,), dtype=np.bool_))
    np.testing.assert_array_equal(aux["commit_applied"], np.zeros((2,), dtype=np.bool_))
    np.testing.assert_allclose(
        new_state.fast_weights[mem._output_weight_name],  # noqa: SLF001
        jnp.asarray(0.8, dtype=jnp.float32) * state.fast_weights[mem._output_weight_name],  # noqa: SLF001
        rtol=0.0,
        atol=0.0,
    )


def test_delta_init_pool_hidden_commit_decay_and_raw_read_stay_fp32_under_bf16():
    config = _delta_config()
    mem = memory.TitansMemory(config, rngs=nnx.Rngs(3))
    # Simulate a mixed-precision checkpoint restore.  The v3.5 episode boundary must promote
    # every fast leaf even though the stored outer seed is BF16.
    for name in mem.m0:
        mem.m0[name].value = mem.m0[name].value.astype(jnp.bfloat16)
    state = mem.init_state(2)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(state))
    k, v = _pairs(config, dtype=jnp.bfloat16)

    state, aux = mem.delta_write_kv(state, k, v)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(state))
    for name in (
        "pooled_key",
        "pooled_value",
        "pooled_key_pre_norm",
        "pooled_value_pre_norm",
        "hidden",
        "hidden_norm_sq",
        "pre_residual",
        "post_residual",
        "relative_commit_residual",
        "delta_w3_norm",
        "w3_norm",
    ):
        assert aux[name].dtype == jnp.float32, name
    state, decay_aux = mem.analytic_decay(state, 40)
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(state))
    assert decay_aux["decay_factor"].dtype == jnp.float32
    assert mem.hidden_key(state, k[:, :3]).dtype == jnp.float32
    assert mem.read_key(state, k[:, :3]).dtype == jnp.float32


@pytest.mark.parametrize("n_steps", [0, 1, 40, 100])
def test_dense_and_analytic_decay_match_forward_read_and_gradients(n_steps):
    config = _delta_config(alpha_step=0.01)
    mem = memory.TitansMemory(config, rngs=nnx.Rngs(4))
    base = mem.init_state(2)
    w3_name = mem._output_weight_name  # noqa: SLF001
    w3 = 0.1 * jax.random.normal(
        jax.random.key(30),
        base.fast_weights[w3_name].shape,
        dtype=jnp.float32,
    )
    base = _replace_w3(base, mem, w3)
    q = memory._l2_norm(  # noqa: SLF001
        jax.random.normal(jax.random.key(31), (2, 3, config.d_key), dtype=jnp.float32)
    )
    read_weight = jax.random.normal(jax.random.key(32), (2, 3, config.d_value), dtype=jnp.float32)

    def dense(initial_w3):
        state = _replace_w3(base, mem, initial_w3)
        for _ in range(n_steps):
            state, _ = mem.analytic_decay(state, 1)
        return state

    def analytic(initial_w3):
        return mem.analytic_decay(_replace_w3(base, mem, initial_w3), n_steps)[0]

    dense_state = dense(w3)
    analytic_state = analytic(w3)
    for name in dense_state.fast_weights:
        np.testing.assert_allclose(
            analytic_state.fast_weights[name], dense_state.fast_weights[name], rtol=1e-6, atol=1e-7
        )
    dense_read = mem.read_key(dense_state, q)
    analytic_read = mem.read_key(analytic_state, q)
    assert _relative_error(analytic_read, dense_read) <= 1e-6

    def read_loss(run, initial_w3):
        return jnp.sum(mem.read_key(run(initial_w3), q) * read_weight)

    dense_grad = jax.grad(lambda initial_w3: read_loss(dense, initial_w3))(w3)
    analytic_grad = jax.grad(lambda initial_w3: read_loss(analytic, initial_w3))(w3)
    assert _relative_error(analytic_grad, dense_grad) <= 1e-6


def test_analytic_decay_supports_per_sample_integer_gaps_and_rejects_illegal_use():
    config = _delta_config(alpha_step=0.1)
    mem = memory.TitansMemory(config, rngs=nnx.Rngs(5))
    state = _nonzero_state(mem)
    old_w3 = state.fast_weights[mem._output_weight_name]  # noqa: SLF001
    new_state, aux = mem.analytic_decay(state, jnp.asarray([0, 3], dtype=jnp.int32))
    expected = jnp.asarray([1.0, 0.9**3], dtype=jnp.float32)[:, None, None] * old_w3
    np.testing.assert_allclose(new_state.fast_weights[mem._output_weight_name], expected, rtol=1e-6)  # noqa: SLF001
    np.testing.assert_array_equal(aux["decay_gap_valid"], np.ones((2,), dtype=np.bool_))
    np.testing.assert_array_equal(new_state.fast_weights[mem._output_bias_name], 0.0)  # noqa: SLF001
    for leaf in jax.tree.leaves(new_state.momentum):
        np.testing.assert_array_equal(leaf, 0.0)

    # Production gaps are dynamic scan inputs.  A traced negative cannot raise from XLA, so it
    # must become a flagged no-op rather than accidental growth by rho**negative.
    jitted_decay = jax.jit(lambda current, gaps: mem.analytic_decay(current, gaps))
    traced_state, traced_aux = jitted_decay(state, jnp.asarray([-1, 3], dtype=jnp.int32))
    np.testing.assert_array_equal(traced_aux["decay_gap_valid"], np.asarray([False, True]))
    np.testing.assert_array_equal(
        traced_state.fast_weights[mem._output_weight_name][0],  # noqa: SLF001
        old_w3[0],
    )
    np.testing.assert_allclose(
        traced_state.fast_weights[mem._output_weight_name][1],  # noqa: SLF001
        expected[1],
        rtol=1e-6,
    )

    with pytest.raises(ValueError, match="non-negative"):
        mem.analytic_decay(state, -1)
    with pytest.raises(TypeError, match="integer dtype"):
        mem.analytic_decay(state, 1.5)
    with pytest.raises(ValueError, match=r"scalar or shape \[batch\]"):
        mem.analytic_decay(state, jnp.ones((2, 1), dtype=jnp.int32))
    legacy = memory.TitansMemory(
        memory.MemoryConfig(d_input=16, d_key=8, hidden_dims=(12, 10), d_value=14), rngs=nnx.Rngs(5)
    )
    with pytest.raises(ValueError, match="valid only for write_rule='delta_output'"):
        legacy.analytic_decay(legacy.init_state(2), 1)


def test_existing_write_boundaries_dispatch_to_fixed_delta_and_preserve_cotangent_guards():
    plain = memory.TitansMemory(_delta_config(), rngs=nnx.Rngs(6))
    guarded = memory.TitansMemory(_delta_config(state_cotangent_clip=1e9, kv_cotangent_clip=1e9), rngs=nnx.Rngs(6))
    state = _nonzero_state(plain)
    guarded_state = memory.MemoryState(
        jax.tree.map(jnp.array, state.fast_weights), jax.tree.map(jnp.array, state.momentum)
    )
    k, v = _pairs(plain.config)
    gates = jnp.full((2,), 0.37, dtype=jnp.float32)

    explicit, explicit_aux = plain.delta_write_kv(state, k, v)
    dispatched, dispatched_aux = plain.write_kv(state, k, v, gates, gates, gates)
    jax.tree.map(np.testing.assert_array_equal, explicit, dispatched)
    jax.tree.map(np.testing.assert_array_equal, explicit_aux, dispatched_aux)

    guarded_result, guarded_aux = guarded.delta_write_kv(guarded_state, k, v)
    jax.tree.map(np.testing.assert_array_equal, explicit, guarded_result)
    jax.tree.map(np.testing.assert_array_equal, explicit_aux, guarded_aux)

    q = memory._l2_norm(  # noqa: SLF001
        jax.random.normal(jax.random.key(40), (2, 2, plain.config.d_key), dtype=jnp.float32)
    )

    def loss(mem, source_state, keys, values):
        written, _ = mem.delta_write_kv(source_state, keys, values)
        return jnp.sum(mem.read_key(written, q))

    plain_grads = jax.grad(loss, argnums=(1, 2, 3))(plain, state, k, v)
    guarded_grads = jax.grad(loss, argnums=(1, 2, 3))(guarded, guarded_state, k, v)
    jax.tree.map(np.testing.assert_array_equal, plain_grads, guarded_grads)
    assert float(jnp.linalg.norm(plain_grads[0].fast_weights["w0"])) > 0.0
    assert float(jnp.linalg.norm(plain_grads[1])) > 0.0
    assert float(jnp.linalg.norm(plain_grads[2])) > 0.0
