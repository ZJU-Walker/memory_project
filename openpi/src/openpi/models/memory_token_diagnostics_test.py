import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import memory


def _nontrivial_memory_and_state(*, batch_size: int = 2):
    config = memory.MemoryConfig(d_input=5, d_key=3, hidden_dims=(4, 3), d_value=2)
    module = memory.TitansMemory(config, rngs=nnx.Rngs(7))
    state = module.init_state(batch_size)

    # A freshly initialized memory has a zero output kernel, which intentionally prevents
    # gradients from reaching earlier layers.  Populate every fast-weight leaf so this test
    # exercises the complete analytic backward pass.
    keys = jax.random.split(jax.random.key(19), len(state.fast_weights))
    fast_weights = {
        name: 0.2 * jax.random.normal(key, value.shape)
        for (name, value), key in zip(sorted(state.fast_weights.items()), keys, strict=True)
    }
    state = memory.MemoryState(fast_weights=fast_weights, momentum=state.momentum)
    return module, state


def _tree_global_norm(tree):
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(tree)))


def test_token_write_diagnostics_match_independent_autodiff_for_every_token():
    module, state = _nontrivial_memory_and_state()
    hidden = jax.random.normal(jax.random.key(23), (2, 4, 5))
    diagnostics = module.token_write_diagnostics(state, hidden)
    _, keys, values = module._keys_values(hidden)  # noqa: SLF001

    expected_error = np.empty((2, 4), dtype=np.float32)
    expected_grad_norm = np.empty((2, 4), dtype=np.float32)
    for batch_index in range(2):
        for token_index in range(4):
            key = keys[batch_index, token_index]
            value = values[batch_index, token_index]

            def token_loss(fast_weights, key=key, value=value):
                prediction = module._forward(fast_weights, key[None])[0]  # noqa: SLF001
                return jnp.sum(jnp.square(prediction - value))

            error, grads = jax.value_and_grad(token_loss)(
                jax.tree.map(lambda leaf, batch_index=batch_index: leaf[batch_index], state.fast_weights)
            )
            expected_error[batch_index, token_index] = error
            expected_grad_norm[batch_index, token_index] = _tree_global_norm(grads)

    np.testing.assert_allclose(diagnostics["token_error"], expected_error, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(diagnostics["token_grad_norm"], expected_grad_norm, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(
        diagnostics["token_mean_loss_grad_norm"], expected_grad_norm / hidden.shape[1], rtol=2e-6, atol=2e-6
    )


def test_token_errors_average_to_exact_pre_write_surprise_and_do_not_change_state():
    module, state = _nontrivial_memory_and_state()
    hidden = jax.random.normal(jax.random.key(29), (2, 7, 5))
    before = jax.tree.map(np.asarray, state)

    diagnostics = module.token_write_diagnostics(state, hidden)
    _, write_aux = module.write(state, hidden)

    np.testing.assert_allclose(jnp.mean(diagnostics["token_error"], axis=1), write_aux["surprise"], rtol=1e-6)
    jax.tree.map(np.testing.assert_array_equal, state, before)


def test_blank_memory_token_error_is_uniform_because_values_are_unit_normalized():
    config = memory.MemoryConfig(d_input=5, d_key=3, hidden_dims=(4,), d_value=2)
    module = memory.TitansMemory(config, rngs=nnx.Rngs(37))
    state = module.init_state(1)
    hidden = jax.random.normal(jax.random.key(41), (1, 11, 5))

    diagnostics = module.token_write_diagnostics(state, hidden)

    # m0's output layer is exactly zero and V is L2-normalized.  Except for the 1e-6
    # stabilization term this forces every first-write error to one, irrespective of image
    # content; users must not interpret that first error heatmap as semantic uniformity.
    np.testing.assert_allclose(diagnostics["token_error"], np.ones((1, 11)), rtol=2e-5, atol=2e-5)


def test_token_write_diagnostics_are_jittable_and_preserve_batch_and_grid_axes():
    module, state = _nontrivial_memory_and_state(batch_size=1)
    hidden = jax.random.normal(jax.random.key(31), (1, 256, 5))

    diagnostics = nnx.jit(module.token_write_diagnostics)(state, hidden)

    assert set(diagnostics) == {"token_error", "token_grad_norm", "token_mean_loss_grad_norm"}
    assert all(value.shape == (1, 256) for value in diagnostics.values())
    assert all(value.dtype == jnp.float32 for value in diagnostics.values())
    assert all(np.isfinite(value).all() for value in diagnostics.values())
    np.testing.assert_array_equal(diagnostics["token_mean_loss_grad_norm"], diagnostics["token_grad_norm"] / 256)
