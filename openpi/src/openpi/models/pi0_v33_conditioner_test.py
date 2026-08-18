"""Tests for the v3.3 task-conditioned write path and the endpoint gradient probe.

The contract under test: (1) a freshly initialized conditioner is an exact no-op, so v3.3
starts bitwise-identical to v3.2; (2) once its output projection is nonzero, the instruction
tokens change the write and padding never does; (3) the section-16 gradient probe reports
credit flowing from the endpoint CE back to earlier writes, and exactly zero at the endpoint
itself (reads precede writes).
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import memory
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.pi0_v32_test import _prepare
from openpi.models.pi0_v32_test import _single_observation
from openpi.models.pi0_v32_test import _TinyV32


class _TinyV33(_TinyV32):
    v33_endpoint_gradient_step = pi0.Pi0.v33_endpoint_gradient_step

    def __init__(self, rngs: nnx.Rngs):
        super().__init__(rngs)
        self.memory_task_conditioned_write = True
        self.write_query_conditioner = pi0.MemoryQueryConditioner(
            num_queries=16, width=64, num_heads=8, rngs=rngs
        )


@pytest.fixture(scope="module")
def tiny_v33():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        yield _TinyV33(nnx.Rngs(0))
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def test_zero_init_conditioner_is_bitwise_v32(tiny_v33):
    """At initialization the conditioned write must equal the unconditioned one exactly."""
    observation = _single_observation()
    state = tiny_v33.memory.init_state(1)
    conditioned = _prepare(tiny_v33, observation, state)
    tiny_v33.memory_task_conditioned_write = False
    try:
        unconditioned = _prepare(tiny_v33, observation, state)
    finally:
        tiny_v33.memory_task_conditioned_write = True
    np.testing.assert_array_equal(
        np.asarray(conditioned["write_tokens"]), np.asarray(unconditioned["write_tokens"])
    )
    assert unconditioned["write_queries"] is None
    np.testing.assert_array_equal(
        np.asarray(conditioned["write_queries"]),
        np.broadcast_to(np.asarray(tiny_v33.write_query_compressor.query_bank.value)[None], (1, 16, 64)),
    )


def test_conditioner_masks_padding_and_reacts_to_context():
    rngs = nnx.Rngs(0)
    conditioner = pi0.MemoryQueryConditioner(num_queries=4, width=32, num_heads=4, rngs=rngs)
    # open the pathway: the zero-init output projection would hide everything
    conditioner.output_proj.kernel.value = jax.random.normal(jax.random.key(1), (32, 32)) * 0.1
    base = jax.random.normal(jax.random.key(2), (4, 32))
    context = jax.random.normal(jax.random.key(3), (2, 6, 32))
    mask = jnp.asarray([[True] * 4 + [False] * 2, [True] * 6])

    out = conditioner(base, context, mask)
    assert out.shape == (2, 4, 32)
    assert not np.allclose(np.asarray(out[0]), np.asarray(base))  # conditioning changes the bank

    # padded positions must not influence the result
    scrambled = context.at[0, 4:].set(1e3)
    np.testing.assert_allclose(
        np.asarray(conditioner(base, scrambled, mask)[0]), np.asarray(out[0]), rtol=1e-5, atol=1e-5
    )
    # different valid context -> different conditioned queries
    shifted = context.at[0, 0].add(1.0)
    assert not np.allclose(np.asarray(conditioner(base, shifted, mask)[0]), np.asarray(out[0]))
    # an all-padding row degrades to the raw bank rather than NaN
    none_valid = conditioner(base, context, jnp.zeros_like(mask))
    np.testing.assert_allclose(np.asarray(none_valid[1]), np.asarray(base), rtol=1e-6, atol=1e-6)


def test_config_rejects_conditioning_without_v32_architecture():
    with pytest.raises(ValueError, match="memory_task_conditioned_write"):
        pi0_config.Pi0Config(
            pi05=True,
            predict_subtask=True,
            predict_with_memory=True,
            memory_architecture="v3_v31",
            memory_task_conditioned_write=True,
        )


def _sequence_batch(batch: int, steps: int, valid_steps: int):
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=2,
        max_token_len=4,
        predict_subtask=True,
        predict_with_memory=True,
        memory_layer=1,
        memory=memory.MemoryConfig(d_input=64, d_key=8, hidden_dims=(8,), d_value=64),
        memory_seq_steps=steps,
        causal_token_len=2,
    )
    observation = config.fake_obs(batch)
    step_mask = jnp.arange(steps)[None, :] < valid_steps
    causal = jnp.ones((batch, steps, 2), dtype=jnp.int32)
    return dataclasses.replace(
        observation,
        tokenized_prompt_mask=jnp.ones_like(observation.tokenized_prompt_mask),
        seq_step_mask=jnp.broadcast_to(step_mask, (batch, steps)),
        tokenized_causal=causal,
        tokenized_causal_mask=jnp.ones_like(causal, dtype=bool),
    )


def test_endpoint_gradient_probe_flows_backward_not_forward(tiny_v33):
    """g at the endpoint must be exactly zero (write follows the prediction); earlier valid
    steps must receive nonzero credit through the recurrent memory; padded steps none."""
    observation = _sequence_batch(batch=2, steps=5, valid_steps=4)
    out = tiny_v33.v33_endpoint_gradient_step(observation, gate_override=1.0)
    grad = np.asarray(out["write_grad_norm"])
    assert grad.shape == (2, 5)
    np.testing.assert_allclose(grad[:, 3], 0.0, atol=1e-12)  # endpoint's own write: after the CE
    np.testing.assert_allclose(grad[:, 4], 0.0, atol=1e-12)  # padded step
    assert (grad[:, :3] > 0).all(), f"no credit reached earlier writes: {grad}"


def test_endpoint_gradient_probe_zero_gate_kills_credit(tiny_v33):
    """With the content gate forced to zero, no memory reaches the prediction, so the probe
    must report zero gradient everywhere -- the initialization-time signature."""
    observation = _sequence_batch(batch=1, steps=4, valid_steps=3)
    out = tiny_v33.v33_endpoint_gradient_step(observation, gate_override=0.0)
    np.testing.assert_allclose(np.asarray(out["write_grad_norm"]), 0.0, atol=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
