"""Tests for the read-only v3.2 checkpoint-diagnostic hooks.

These guard the offline evaluation path only: the attention view must share the forward
math bit-for-bit and the per-frame diagnostic step must return exactly the tensors the
inference write would consume, without touching the fast state.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import pi0
from openpi.models.pi0_v32_test import _prepare
from openpi.models.pi0_v32_test import _single_observation
from openpi.models.pi0_v32_test import _TinyV32

_TinyV32.v32_query_attention_step = pi0.Pi0.v32_query_attention_step


@pytest.fixture(scope="module")
def tiny_model():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        # Same vocabulary shrink as pi0_v32_test: these structural tests use token ids <= 8.
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        yield _TinyV32(nnx.Rngs(0))
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def test_attention_probs_shares_the_forward_attention_math(tiny_model):
    source = jax.random.normal(jax.random.key(3), (2, 5, 64), dtype=jnp.float32)
    compressor = tiny_model.read_query_compressor
    probs = compressor.attention_probs(source)
    assert probs.shape == (2, compressor.num_heads, compressor.num_queries, 5)
    assert probs.dtype == jnp.float32
    np.testing.assert_allclose(np.sum(np.asarray(probs), axis=-1), 1.0, rtol=1e-6)
    value = compressor.value_proj(source).reshape(2, 5, compressor.num_heads, compressor.head_dim)
    pooled = jnp.einsum("bhqn,bnhd->bqhd", probs.astype(value.dtype), value)
    recombined = compressor.output_proj(pooled.reshape(2, compressor.num_queries, 64)).astype(jnp.float32)
    np.testing.assert_allclose(np.asarray(recombined), np.asarray(compressor(source)), rtol=1e-6, atol=1e-6)


def test_query_attention_step_matches_inference_tensors_and_is_read_only(tiny_model):
    observation = _single_observation()
    state = tiny_model.memory.init_state(1)
    before = jax.tree.map(np.array, state)
    out = tiny_model.v32_query_attention_step(observation, state)
    jax.tree.map(np.testing.assert_array_equal, before, jax.tree.map(np.asarray, state))

    prepared = _prepare(tiny_model, observation, state)
    np.testing.assert_allclose(
        np.asarray(out["write_tokens"]), np.asarray(prepared["write_tokens"]), rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(np.asarray(out["retrieved"]), np.asarray(prepared["retrieved"]), rtol=1e-6, atol=1e-6)

    for key in ("read_attention", "write_attention"):
        probs = np.asarray(out[key])
        assert probs.shape[-2] == 16
        np.testing.assert_allclose(probs.sum(axis=-1), 1.0, rtol=1e-6)
    assert not np.allclose(np.asarray(out["read_attention"]), np.asarray(out["write_attention"]))
    assert out["write_slot_token_error"].shape == (1, 16)
    assert out["retrieved_slot_norm"].shape == (1, 16)
    assert out["write_slot_norm"].shape == (1, 16)
