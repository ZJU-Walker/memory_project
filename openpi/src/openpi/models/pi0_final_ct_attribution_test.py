"""Exactness tests for final-normalized c_t writer attribution.

The production SigLIP encoder emits a 16x16 patch grid.  The tiny test encoder emits one token
per camera so these tests exercise the full two-pass Gemma/memory path without constructing the
large vision model; the token axis remains identical semantically.
"""

import inspect

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.rtc_test import _TinyPi05SequenceTrainer
from openpi.shared import nnx_utils


class _TinyFinalCtModel(_TinyPi05SequenceTrainer):
    _final_ct_from_image_embeddings = pi0.Pi0._final_ct_from_image_embeddings  # noqa: SLF001
    final_ct_intervention_step = pi0.Pi0.final_ct_intervention_step
    final_ct_attribution_step = pi0.Pi0.final_ct_attribution_step
    writer_echo_factorial_step = pi0.Pi0.writer_echo_factorial_step
    writer_echo_factorial_metrics_step = pi0.Pi0.writer_echo_factorial_metrics_step
    memory_swap_read_step = pi0.Pi0.memory_swap_read_step


def _observation(batch_size=1):
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=2,
        max_token_len=4,
    )
    observation = config.fake_obs(batch_size)
    # Break camera symmetry.  The tiny encoder mean-pools each image, so constants are enough
    # to give every camera a distinct, deterministic embedding.
    return observation.replace(
        images={
            name: jnp.full_like(image, -0.75 + 0.5 * index)
            for index, (name, image) in enumerate(observation.images.items())
        }
    )


def _written_state(model, batch_size=1):
    state = model.memory.init_state(batch_size)
    source = jax.random.normal(jax.random.key(17), (batch_size, 3, model.memory.config.d_input))
    # More than one write makes the old-memory read visibly nonzero for arbitrary prefix h_t.
    state, _ = model.memory.write(state, source)
    state, _ = model.memory.write(state, source * 0.5 + 0.1)
    return state


def _assert_tree_exact(actual, expected):
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    assert len(actual_leaves) == len(expected_leaves)
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


@pytest.fixture(scope="module")
def fixture():
    model = _TinyFinalCtModel(nnx.Rngs(0), memory_write_source="post_attention")
    observation = _observation()
    state = _written_state(model)
    attribution = nnx_utils.module_jit(
        model.final_ct_attribution_step,
        static_argnames=("allow_write",),
    )
    intervention = nnx_utils.module_jit(
        model.final_ct_intervention_step,
        static_argnames=("allow_write",),
    )
    return model, observation, state, attribution, intervention


def test_final_ct_attribution_shapes_are_finite_and_nonzero(fixture):
    model, observation, state, attribution, _ = fixture
    unchanged, aux = attribution(observation, state, allow_write=False)

    tokens = int(aux["top_camera_tokens"])
    assert aux["final_ct"].shape == (1, tokens, model.memory.config.d_value)
    assert aux["writer_loss"].shape == (1,)
    assert aux["writer_loss_top_patch_grad_norm"].shape == (1, tokens)
    assert aux["final_ct_zero_read_l2"].shape == (1, tokens)
    for value in aux.values():
        assert np.isfinite(np.asarray(value)).all()
    assert np.all(np.asarray(aux["writer_loss_top_patch_grad_norm"]) >= 0)
    assert np.max(np.asarray(aux["writer_loss_top_patch_grad_norm"])) > 0
    assert np.max(np.asarray(aux["final_ct_zero_read_l2"])) > 0
    np.testing.assert_allclose(aux["writer_loss"], aux["surprise"], rtol=2e-6, atol=2e-6)
    _assert_tree_exact(unchanged, state)
    np.testing.assert_array_equal(aux["write_occurred"], [False])


def test_no_gradient_primal_matches_attribution_and_can_commit(fixture):
    _, observation, state, attribution, intervention = fixture
    unchanged, attributed = attribution(observation, state, allow_write=False)
    unchanged_intervention, primal = intervention(observation, state, allow_write=False)

    _assert_tree_exact(unchanged, state)
    _assert_tree_exact(unchanged_intervention, state)
    np.testing.assert_array_equal(primal["final_ct"], attributed["final_ct"])
    np.testing.assert_array_equal(primal["writer_loss"], attributed["writer_loss"])

    changed, committed = intervention(observation, state, allow_write=True)
    assert any(
        not np.array_equal(np.asarray(actual), np.asarray(before))
        for actual, before in zip(jax.tree.leaves(changed), jax.tree.leaves(state), strict=True)
    )
    np.testing.assert_array_equal(committed["write_occurred"], [True])
    np.testing.assert_array_equal(committed["final_ct"], primal["final_ct"])


def test_writer_echo_factorial_reduces_branches_and_commits_only_matched_normal_cells():
    model = _TinyFinalCtModel(nnx.Rngs(0), memory_write_source="post_attention")
    left = _observation()
    right = _observation().replace(
        images={name: image + 0.25 for name, image in _observation().images.items()},
    )
    observation4 = jax.tree.map(
        lambda *values: None if values[0] is None else jnp.concatenate(values, axis=0),
        left,
        right,
        right,
        left,
        is_leaf=lambda value: value is None,
    )
    paired_state = _written_state(model, batch_size=2)
    run = nnx_utils.module_jit(model.writer_echo_factorial_metrics_step)
    committed, metrics = run(observation4, paired_state)

    assert all(leaf.shape[0] == 2 for leaf in jax.tree.leaves(committed))
    assert metrics
    assert all(value.shape == (2,) for value in metrics.values())
    assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())
    for family in ("final_ct", "injection", "fast_update", "full_update"):
        assert np.all(np.asarray(metrics[f"{family}_memory_effect_relative"]) >= 0)
        assert np.all(np.asarray(metrics[f"{family}_observation_effect_relative"]) >= 0)
        assert np.all(np.asarray(metrics[f"{family}_interaction_relative"]) >= 0)

    indices = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
    state4 = jax.tree.map(lambda value: value[indices], paired_state)
    direct = nnx_utils.module_jit(model.writer_echo_factorial_step)
    normal_state, _, _ = direct(observation4, state4)
    expected = jax.tree.map(lambda value: value[jnp.asarray([0, 2])], normal_state)
    _assert_tree_exact(committed, expected)


def test_memory_swap_read_step_is_read_only_and_returns_full_vectors():
    model = _TinyFinalCtModel(nnx.Rngs(0), memory_write_source="post_attention")
    observation = _observation(batch_size=2)
    state = _written_state(model, batch_size=2)
    before = jax.tree.map(jnp.copy, state)
    run = nnx_utils.module_jit(model.memory_swap_read_step)
    output = run(observation, state)

    assert output["retrieved"].shape[:2] == (2, 1)
    assert output["final_ct"].shape == output["retrieved"].shape
    assert np.isfinite(np.asarray(output["retrieved"])).all()
    assert np.isfinite(np.asarray(output["final_ct"])).all()
    _assert_tree_exact(state, before)


def test_intervention_accepts_batched_occlusions_with_one_repeated_prestate(fixture):
    model, _, state, _, intervention = fixture
    batch_size = 4
    observation = _observation(batch_size)
    # Emulate an occlusion chunk: every counterfactual has a different top-camera input, while
    # all of them branch from the exact same pre-write episodic state.
    top = observation.images["base_0_rgb"]
    levels = jnp.linspace(-1.0, 0.5, batch_size).reshape(batch_size, 1, 1, 1)
    observation = observation.replace(images={**observation.images, "base_0_rgb": top * 0 + levels})
    repeated_state = jax.tree.map(lambda x: jnp.broadcast_to(x, (batch_size, *x.shape[1:])), state)

    unchanged, aux = intervention(observation, repeated_state, allow_write=False)
    assert aux["final_ct"].shape == (batch_size, int(aux["top_camera_tokens"]), model.memory.config.d_value)
    assert aux["writer_loss"].shape == (batch_size,)
    assert aux["surprise"].shape == (batch_size,)
    assert np.isfinite(np.asarray(aux["final_ct"])).all()
    assert np.isfinite(np.asarray(aux["writer_loss"])).all()
    _assert_tree_exact(unchanged, repeated_state)
    np.testing.assert_array_equal(aux["write_occurred"], np.zeros(batch_size, dtype=bool))


def test_override_uses_exact_siglip_output_boundary(fixture):
    model, observation, state, _, intervention = fixture
    preprocessed = model_lib.preprocess_observation(None, observation, train=False)
    inferred, _ = model.PaliGemma.img(preprocessed.images["base_0_rgb"], train=False)

    _, ordinary = intervention(observation, state, allow_write=False)
    _, exact_override = intervention(
        observation,
        state,
        allow_write=False,
        top_camera_patch_embeddings=inferred,
    )
    np.testing.assert_array_equal(exact_override["final_ct"], ordinary["final_ct"])
    np.testing.assert_array_equal(exact_override["writer_loss"], ordinary["writer_loss"])

    _, changed_override = intervention(
        observation,
        state,
        allow_write=False,
        top_camera_patch_embeddings=inferred + 0.25,
    )
    assert not np.array_equal(np.asarray(changed_override["final_ct"]), np.asarray(ordinary["final_ct"]))


def test_final_ct_is_after_last_block_and_final_norm(fixture):
    model, observation, state, _, intervention = fixture
    preprocessed = model_lib.preprocess_observation(None, observation, train=False)
    image_embeddings = tuple(
        model.PaliGemma.img(preprocessed.images[name], train=False)[0] for name in preprocessed.images
    )

    # Reconstruct the memory-token call and retain hidden states.  Gemma documents hidden[-1]
    # as the last block output before final RMSNorm; its ordinary output is after final RMSNorm.
    image_mask = jnp.concatenate(
        [
            jnp.broadcast_to(preprocessed.image_masks[name][:, None], tokens.shape[:2])
            for name, tokens in zip(preprocessed.images, image_embeddings, strict=True)
        ],
        axis=1,
    )
    prompt = preprocessed.tokenized_prompt
    prompt_mask = preprocessed.tokenized_prompt_mask
    ar = preprocessed.token_ar_mask
    if ar is None:
        ar = jnp.zeros_like(prompt)
    prefix_tokens = jnp.concatenate(
        [jnp.concatenate(image_embeddings, axis=1), model.PaliGemma.llm(prompt, method="embed")], axis=1
    )
    prefix_mask = jnp.concatenate([image_mask, prompt_mask], axis=1)
    prefix_ar = jnp.concatenate([jnp.zeros_like(image_mask, dtype=jnp.int32), ar], axis=1)
    _, cache, prefix_hidden = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=pi0.make_attn_mask(prefix_mask, prefix_ar),
        positions=jnp.cumsum(prefix_mask, axis=1) - 1,
        return_hidden_states=True,
    )
    mem_len = image_embeddings[0].shape[1]
    raw_h = prefix_hidden[0][model.memory_layer][:, :mem_len].astype(jnp.float32)
    retrieved = model.memory.read(state, raw_h)
    prefix_len = prefix_tokens.shape[1]
    cache = jax.tree.map(
        lambda x: jnp.pad(
            x,
            ((0, 0), (0, 0), (0, mem_len + model.causal_token_len), (0, 0), (0, 0)),
        ),
        cache,
    )
    memory_tokens = (model.memory_gate.value * retrieved).astype(prefix_tokens.dtype)
    (post_norm, _), _, memory_hidden = model.PaliGemma.llm(
        [memory_tokens, None],
        mask=pi0.make_memory_step_mask(prefix_mask, prefix_ar, mem_len, model.causal_token_len),
        positions=prefix_len + jnp.broadcast_to(jnp.arange(mem_len), (1, mem_len)),
        kv_cache=cache,
        cache_position=prefix_len,
        return_hidden_states=True,
    )
    pre_norm = memory_hidden[0][-1]
    _, aux = intervention(observation, state, allow_write=False)

    np.testing.assert_array_equal(aux["final_ct"], post_norm.astype(jnp.float32))
    assert not np.allclose(np.asarray(post_norm), np.asarray(pre_norm)), (
        "final c_t must be Module output after final RMSNorm, not hidden_states[-1]"
    )


def test_diagnostic_adds_no_parameters_and_preserves_default_inference(fixture):
    model, observation, state, attribution, _ = fixture
    params_before = nnx.state(model, nnx.Param)
    path_before = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(params_before)]

    sample = nnx_utils.module_jit(
        model.sample_with_memory,
        static_argnames=("stop_token", "max_decode_steps", "num_steps"),
    )
    sample_args = {
        "stop_token": 1,
        "max_decode_steps": 1,
        "num_steps": 2,
        "noise": jnp.zeros((1, model.action_horizon, model.action_dim), dtype=jnp.float32),
    }
    actions_before, state_before, aux_before = sample(jax.random.key(9), observation, state, **sample_args)
    attribution(observation, state, allow_write=False)
    actions_after, state_after, aux_after = sample(jax.random.key(9), observation, state, **sample_args)

    params_after = nnx.state(model, nnx.Param)
    path_after = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(params_after)]
    assert path_before == path_after
    _assert_tree_exact(params_after, params_before)
    np.testing.assert_array_equal(actions_after, actions_before)
    _assert_tree_exact(state_after, state_before)
    _assert_tree_exact(aux_after, aux_before)
    assert inspect.signature(model.sample_with_memory).parameters["zero_read"].default is False
    assert inspect.signature(model.sample_with_memory).parameters["allow_write"].default is True


def test_raw_hidden_model_rejects_final_ct_writer_label():
    model = _TinyFinalCtModel(nnx.Rngs(0), memory_write_source="raw_hidden")
    observation = _observation()
    state = model.memory.init_state(1)
    with pytest.raises(ValueError, match="post_attention"):
        model.final_ct_attribution_step(observation, state)
    with pytest.raises(ValueError, match="post_attention"):
        model.final_ct_intervention_step(observation, state)
