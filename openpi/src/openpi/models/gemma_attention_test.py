"""Attention-probability capture must be a pure diagnostic.

The pi0.5 memory experiments read layer-8 attention to see which image patches a write or a
decoded subtask token actually looks at.  Capturing it must never perturb the model that is
being measured, so these tests pin both the numerical invariance and the correctness of the
returned distributions.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma


def _run(*, return_attn_probs, depth=4, seq=6, causal=True, seed=0):
    config = gemma.get_config("dummy")
    config.depth = depth
    module = gemma.Module(configs=[config], embed_dtype="float32", return_attn_probs=return_attn_probs)
    tokens = jax.random.normal(jax.random.key(seed), (1, seq, config.width)) * 0.1
    positions = jnp.arange(seq)[None]
    mask = jnp.tril(jnp.ones((seq, seq), bool))[None] if causal else jnp.ones((1, seq, seq), bool)

    def call(mod, *args, **kwargs):
        return mod(*args, **kwargs)

    variables = nn.init(call, module)(jax.random.key(0), [tokens], positions, mask, return_hidden_states=True)
    return nn.apply(call, module)(variables, [tokens], positions, mask, return_hidden_states=True), variables


def test_capturing_attention_does_not_change_outputs_or_parameters():
    (baseline_out, _, baseline_hidden), baseline_vars = _run(return_attn_probs=False)
    (probe_out, _, probe_hidden, probs), probe_vars = _run(return_attn_probs=True)

    assert np.allclose(np.asarray(baseline_out[0]), np.asarray(probe_out[0]), atol=1e-6)
    assert np.allclose(np.asarray(baseline_hidden[0]), np.asarray(probe_hidden[0]), atol=1e-6)
    baseline_leaves = jax.tree.leaves_with_path(baseline_vars)
    probe_leaves = jax.tree.leaves_with_path(probe_vars)
    assert [path for path, _ in baseline_leaves] == [path for path, _ in probe_leaves]
    assert probs is not None


def test_attention_rows_are_distributions_over_every_layer_and_head():
    (_, _, _, probs), _ = _run(return_attn_probs=True, depth=5, seq=7)
    array = np.asarray(probs)
    heads = gemma.get_config("dummy").num_heads
    assert array.shape == (5, 1, heads, 7, 7), "expected [depth, batch, head, query, key]"
    assert np.all(array >= 0)
    # Every individual head is its own distribution; averaging heads would also sum to 1 and so
    # would not catch a head axis that was silently collapsed or mis-ordered.
    assert np.allclose(array.sum(axis=-1), 1.0, atol=1e-5)
    assert not np.allclose(array[0, 0, 0], array[0, 0, -1]), "heads must be distinguishable"


def test_masked_positions_receive_no_attention():
    (_, _, _, probs), _ = _run(return_attn_probs=True, seq=6, causal=True)
    array = np.asarray(probs)
    # A causal mask must leave every strictly-upper-triangular entry at exactly zero, otherwise
    # a heatmap would show a query attending to positions it cannot see.
    upper = array[:, 0][:, :, *np.triu_indices(6, 1)]
    assert np.max(upper) == 0.0, "no head may attend to a masked position"


def test_layer_slice_selects_one_layer():
    (_, _, _, probs), _ = _run(return_attn_probs=True, depth=6)
    array = np.asarray(probs)
    assert array.shape[0] == 6
    # Layers must be distinguishable; identical layers would mean the scan axis was collapsed.
    assert not np.allclose(array[0], array[5])


def test_disabled_capture_returns_no_probability_array():
    result, _ = _run(return_attn_probs=False)
    assert len(result) == 3, "hidden-state mode must not grow an attention output when disabled"


@pytest.mark.parametrize("depth", [1, 3])
def test_capture_supports_any_depth(depth):
    (_, _, _, probs), _ = _run(return_attn_probs=True, depth=depth)
    assert np.asarray(probs).shape[0] == depth


def test_partial_early_then_late_stack_is_exactly_one_full_forward():
    config = gemma.get_config("dummy")
    config.depth = 4
    module = gemma.Module(configs=[config], embed_dtype="float32")
    tokens = jax.random.normal(jax.random.key(9), (1, 6, config.width)) * 0.1
    positions = jnp.arange(6)[None]
    mask = jnp.ones((1, 6, 6), dtype=bool)

    def call(mod, *args, **kwargs):
        return mod(*args, **kwargs)

    variables = nn.init(call, module)(jax.random.key(0), [tokens], positions, mask)
    full, _ = nn.apply(call, module)(variables, [tokens], positions, mask)
    empty_cache = (
        jnp.zeros((4, 1, 6, config.num_kv_heads, config.head_dim), dtype=tokens.dtype),
        jnp.zeros((4, 1, 6, config.num_kv_heads, config.head_dim), dtype=tokens.dtype),
    )
    early, early_cache = nn.apply(call, module)(
        variables,
        [tokens],
        positions,
        mask,
        kv_cache=empty_cache,
        cache_position=0,
        active_layers=jnp.asarray([True, True, False, False]),
        apply_final_norm=False,
    )
    split, split_cache = nn.apply(call, module)(
        variables,
        early,
        positions,
        mask,
        kv_cache=early_cache,
        cache_position=0,
        active_layers=jnp.asarray([False, False, True, True]),
    )

    np.testing.assert_array_equal(full[0], split[0])
    assert np.max(np.abs(np.asarray(early_cache[0][:2]))) > 0
    np.testing.assert_array_equal(early_cache[0][2:], 0)
    np.testing.assert_array_equal(split_cache[0][:2], early_cache[0][:2])
    assert np.max(np.abs(np.asarray(split_cache[0][2:]))) > 0
