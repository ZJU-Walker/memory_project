import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma


def test_tied_decode_uses_bfloat16_operands_with_float32_output_and_master_weights():
    embedder = gemma.Embedder(vocab_size=8, embed_dim=4)
    hidden = jnp.asarray([[[0.125, -0.25, 0.5, 1.0]]], dtype=jnp.float32)
    variables = embedder.init(jax.random.key(0), hidden, compute_dtype=jnp.bfloat16, method=embedder.decode)

    logits = embedder.apply(variables, hidden, compute_dtype=jnp.bfloat16, method=embedder.decode)
    table = variables["params"]["input_embedding"]
    expected = jnp.dot(
        hidden.astype(jnp.bfloat16),
        table.T.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )

    assert table.dtype == jnp.float32
    assert logits.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(expected))

    def loss(params):
        output = embedder.apply({"params": params}, hidden, compute_dtype=jnp.bfloat16, method=embedder.decode)
        return jnp.mean(jnp.square(output))

    grads = jax.grad(loss)(variables["params"])
    table_grad = grads["input_embedding"]
    assert table_grad.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(table_grad)))
    assert np.any(np.asarray(table_grad) != 0)


def test_tied_decode_default_preserves_original_float32_path():
    embedder = gemma.Embedder(vocab_size=8, embed_dim=4)
    hidden = jnp.asarray([[[0.125, -0.25, 0.5, 1.0]]], dtype=jnp.float32)
    variables = embedder.init(jax.random.key(1), hidden, method=embedder.decode)

    logits = embedder.apply(variables, hidden, method=embedder.decode)
    expected = jnp.dot(hidden, variables["params"]["input_embedding"].T)

    assert logits.dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(expected))
