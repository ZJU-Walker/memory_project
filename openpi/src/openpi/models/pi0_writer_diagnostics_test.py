import flax.nnx as nnx
import jax
import numpy as np

from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.rtc_test import _TinyPi05SequenceTrainer
from openpi.shared import nnx_utils


class _TinyWriterContributionModel(_TinyPi05SequenceTrainer):
    writer_contribution_step = pi0.Pi0.writer_contribution_step


def _assert_tree_equal(left, right):
    for left_leaf, right_leaf in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(left_leaf, right_leaf)


def test_writer_contribution_fast_path_is_prewrite_exact_and_can_commit_or_discard():
    model = _TinyWriterContributionModel(nnx.Rngs(0), memory_write_source="post_attention")
    observation = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=2,
        max_token_len=4,
    ).fake_obs(1)
    state0 = model.memory.init_state(1)
    run = nnx_utils.module_jit(model.writer_contribution_step, static_argnames=("allow_write",))

    unchanged, read_only_aux = run(observation, state0, allow_write=False)
    _assert_tree_equal(unchanged, state0)
    np.testing.assert_array_equal(read_only_aux["write_occurred"], [False])
    assert read_only_aux["token_error"].shape == (1, 1)
    assert read_only_aux["token_grad_norm"].shape == (1, 1)
    np.testing.assert_allclose(read_only_aux["surprise"], read_only_aux["token_error"].mean(axis=1))

    changed, write_aux = run(observation, state0, allow_write=True)
    np.testing.assert_array_equal(write_aux["write_occurred"], [True])
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(jax.tree.leaves(changed), jax.tree.leaves(state0), strict=True)
    )

    # Compare against the actual inference path, which appends memory in its own LLM call before
    # causal decoding. (The training helper appends memory+causal tokens in one wider GEMM; it has
    # equivalent masks but legitimately picks a different CUDA reduction at float32 precision.)
    sample = nnx_utils.module_jit(
        model.sample_with_memory,
        static_argnames=("stop_token", "max_decode_steps", "num_steps", "allow_write"),
    )
    _, expected, sample_aux = sample(
        jax.random.key(1),
        observation,
        state0,
        stop_token=1,
        max_decode_steps=1,
        num_steps=2,
        noise=np.zeros((1, model.action_horizon, model.action_dim), dtype=np.float32),
        allow_write=True,
    )
    for actual_leaf, expected_leaf in zip(jax.tree.leaves(changed), jax.tree.leaves(expected), strict=True):
        np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=2e-6, atol=2e-6)
    for name in ("surprise", "grad_norm", "theta", "eta", "alpha", "retrieval_norm", "memory_gate_norm"):
        np.testing.assert_allclose(write_aux[name], sample_aux[name], rtol=2e-6, atol=2e-6)
