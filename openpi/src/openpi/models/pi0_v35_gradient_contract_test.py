"""Gate-C end-to-end gradient contracts for the v3.5 sequence objective."""

# ruff: noqa: SLF001

from collections.abc import Mapping
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models.pi0_v35_test import _TinyV35
from openpi.models.pi0_v35_test import _v35_sequence_observation

_WRITE_HEAD = "memory_write_side_head"
_READ_HEAD = "memory_read_side_head"
_WRITER = "write_query_compressor"
_READER = "read_query_compressor"
_W_Q = "['memory']['w_q']"
_W_K = "['memory']['w_k']"
_W_V = "['memory']['w_v']"


def _path_squared_norms(grads: Any) -> dict[str, float]:
    return {
        jax.tree_util.keystr(path): float(jnp.sum(jnp.square(leaf)))
        for path, leaf in jax.tree_util.tree_leaves_with_path(grads)
    }


def _family_norm(path_norms: Mapping[str, float], fragment: str) -> float:
    matches = [value for path, value in path_norms.items() if fragment in path]
    assert matches, f"No parameter path matched {fragment!r}."
    return sum(matches)


@pytest.fixture(scope="module")
def gate_c_gradients():
    """Trace each branch once, then reuse it across equal-shaped masking cases.

    Keeping the observation dynamic makes the three read cases share one compilation.  The
    fixture exercises the real recurrent Pi0 loss; it does not replace the memory transition
    with a test-only surrogate.
    """
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        model = _TinyV35(nnx.Rngs(350))
        graphdef, params = nnx.split(model)
        actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
        rng = jax.random.key(351)

        def branch_objective(p, observation, branch: str):
            losses = nnx.merge(graphdef, p)._compute_sequence_loss_v32(rng, observation, actions, train=False)
            numerator = jnp.sum(losses[f"v35_{branch}_ce_cell_sum"])
            denominator = jnp.maximum(jnp.sum(losses[f"v35_{branch}_episode_count_cell"]), 1.0)
            diagnostics = {
                "frame_count": losses[f"v35_{branch}_frame_count"],
                "reachable_count": losses["v35_reachable_count"],
                "state_invalid_count": losses["v35_state_invalid_d_count"],
            }
            return numerator / denominator, diagnostics

        write_value_and_grad = jax.jit(
            jax.value_and_grad(lambda p, observation: branch_objective(p, observation, "write"), has_aux=True)
        )
        read_value_and_grad = jax.jit(
            jax.value_and_grad(lambda p, observation: branch_objective(p, observation, "read"), has_aux=True)
        )

        natural = _v35_sequence_observation()
        boundary = _v35_sequence_observation(boundary_at_d=True)
        state_invalid = natural.replace(seq_write_mask=jnp.zeros_like(natural.seq_write_mask))
        padded = natural.replace(
            seq_step_mask=natural.seq_step_mask.at[:, -1].set(False),
            seq_decay_gap_before=natural.seq_decay_gap_before.at[:, -1].set(0),
        )

        cases = {}
        for name, value_and_grad, observation in (
            ("write", write_value_and_grad, natural),
            ("read_reachable", read_value_and_grad, natural),
            ("read_boundary", read_value_and_grad, boundary),
            ("read_state_invalid", read_value_and_grad, state_invalid),
            ("read_padded", read_value_and_grad, padded),
        ):
            (value, diagnostics), grads = value_and_grad(params, observation)
            cases[name] = {
                "value": float(value),
                "diagnostics": {key: float(item) for key, item in diagnostics.items()},
                "path_norms": _path_squared_norms(grads),
            }
        return cases
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def test_lwrite_reaches_writer_value_path_and_write_head(gate_c_gradients):
    case = gate_c_gradients["write"]
    assert np.isfinite(case["value"])
    assert case["diagnostics"]["frame_count"] == 1.0

    # L_write is defined on the committed frame's pooled value.  It must train the dedicated
    # head, V projection, and upstream E-frame writer, without leaking through the read bank.
    for family in (_WRITE_HEAD, _W_V, _WRITER):
        assert _family_norm(case["path_norms"], family) > 0.0, family
    for family in (_READ_HEAD, _READER, _W_Q):
        assert _family_norm(case["path_norms"], family) == 0.0, family


def test_reachable_lread_reaches_consumer_and_preceding_e_writer(gate_c_gradients):
    case = gate_c_gradients["read_reachable"]
    assert np.isfinite(case["value"])
    assert case["diagnostics"]["frame_count"] == 1.0
    assert case["diagnostics"]["reachable_count"] == 1.0

    # The D consumer learns its head/query alignment, and an in-block E->D recurrence assigns
    # credit to both halves of the preceding write association.
    for family in (_READ_HEAD, _READER, _W_Q, _WRITER, _W_K, _W_V):
        assert _family_norm(case["path_norms"], family) > 0.0, family
    assert _family_norm(case["path_norms"], _WRITE_HEAD) == 0.0


def test_boundary_unreachable_lread_trains_consumer_but_not_prior_writer(gate_c_gradients):
    case = gate_c_gradients["read_boundary"]
    assert np.isfinite(case["value"])
    assert case["diagnostics"]["frame_count"] == 1.0
    assert case["diagnostics"]["reachable_count"] == 0.0

    # State validity keeps L_read on.  The TBPTT boundary stops only the carried-state
    # cotangent, so the current D query/head still learn while the unique E writer path is zero.
    for family in (_READ_HEAD, _READER, _W_Q):
        assert _family_norm(case["path_norms"], family) > 0.0, family
    for family in (_WRITER, _W_K, _W_V, _WRITE_HEAD):
        assert _family_norm(case["path_norms"], family) == 0.0, family


@pytest.mark.parametrize(
    ("case_name", "expected_state_invalid"),
    [("read_state_invalid", 1.0), ("read_padded", 0.0)],
)
def test_state_invalid_and_padded_d_steps_have_exactly_zero_gradient(
    gate_c_gradients, case_name, expected_state_invalid
):
    case = gate_c_gradients[case_name]
    assert case["value"] == 0.0
    assert case["diagnostics"]["frame_count"] == 0.0
    assert case["diagnostics"]["state_invalid_count"] == expected_state_invalid
    assert all(value == 0.0 for value in case["path_norms"].values())
