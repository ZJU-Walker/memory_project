"""``Pi0.memory_attention_maps`` must observe the model without changing it.

These maps are read as evidence about which image patches a memory write is built from and
whether a decoded subtask token attends to retrieved memory at all, so the diagnostic must not
write memory, must not perturb inference, and must slice the key axis exactly at the boundaries
the renderer assumes.
"""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import memory
from openpi.models import pi0
from openpi.models import pi0_config


def _tiny_config(**overrides):
    base = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        predict_subtask=True,
        predict_with_memory=True,
        memory=memory.MemoryConfig(d_input=64, d_key=8, hidden_dims=(8,), d_value=64),
        memory_layer=2,
        action_horizon=4,
    )
    return dataclasses.replace(base, **overrides) if overrides else base


def _single_step_observation(config):
    observation = config.fake_obs(batch_size=1)
    return jax.tree.map(
        lambda x: x[:, 0] if hasattr(x, "ndim") and x.ndim >= 2 and x.shape[1] == config.memory_seq_steps else x,
        observation,
    )


def _forced_subtask(config, length=3):
    tokens = np.zeros((1, config.causal_token_len), np.int32)
    tokens[0, :length] = np.arange(5, 5 + length)
    mask = np.zeros((1, config.causal_token_len), bool)
    mask[0, :length] = True
    return jnp.asarray(tokens), jnp.asarray(mask)


@pytest.fixture(scope="module")
def fixture():
    config = _tiny_config()
    model = pi0.Pi0(config, rngs=nnx.Rngs(0))
    return config, model, _single_step_observation(config), model.memory.init_state(1)


def test_maps_have_patch_grid_shape_and_are_valid_distributions(fixture):
    config, model, observation, state = fixture
    with model.capture_attention():
        maps = model.memory_attention_maps(observation, state)

    top_tokens = int(maps["top_camera_tokens"])
    memory_to_top = np.asarray(maps["memory_to_top"])
    assert memory_to_top.shape == (1, top_tokens, top_tokens)
    assert np.all(memory_to_top >= 0)
    # Each row is a slice of a full softmax over all keys, so its mass is a fraction, never >1.
    mass = np.asarray(maps["memory_to_top_mass"])
    assert np.all(mass >= 0)
    assert np.all(mass <= 1 + 1e-5)
    assert np.all(np.asarray(maps["memory_to_prefix_mass"]) >= mass - 1e-6), (
        "top-camera attention is a subset of prefix attention"
    )


def test_attention_budget_blocks_partition_the_full_softmax(fixture):
    config, model, observation, state = fixture
    tokens, mask = _forced_subtask(config)
    with model.capture_attention():
        maps = model.memory_attention_maps(
            observation, state, forced_subtask_tokens=tokens, forced_subtask_mask=mask
        )
    cameras = list(observation.images)

    # Memory rows see the prefix plus the memory block and nothing after it.
    per_camera = np.stack([np.asarray(maps[f"memory_to_camera_{name}_mass"]) for name in cameras])
    assert np.allclose(per_camera.sum(axis=0), np.asarray(maps["memory_to_images_mass"]), atol=1e-5), (
        "per-camera masses must sum to the total image mass"
    )
    memory_total = (
        np.asarray(maps["memory_to_images_mass"])
        + np.asarray(maps["memory_to_prompt_mass"])
        + np.asarray(maps["memory_to_memory_mass"])
    )
    assert np.allclose(memory_total, 1.0, atol=1e-4), "memory-row blocks must account for the whole distribution"
    assert np.allclose(
        np.asarray(maps["memory_to_images_mass"]) + np.asarray(maps["memory_to_prompt_mass"]),
        np.asarray(maps["memory_to_prefix_mass"]),
        atol=1e-5,
    )

    # Subtask rows additionally see the causal block, so their budget needs that term to close.
    live = np.asarray(maps["subtask_token_mask"])[0]
    subtask_total = (
        np.asarray(maps["subtask_to_images_mass"])
        + np.asarray(maps["subtask_to_prompt_mass"])
        + np.asarray(maps["subtask_to_memory_mass"])
        + np.asarray(maps["subtask_to_causal_mass"])
    )[0][live]
    assert np.allclose(subtask_total, 1.0, atol=1e-4), "subtask-row blocks must account for the whole distribution"


def test_head_selection_returns_one_head_and_averages_by_default(fixture):
    config, model, observation, state = fixture
    with model.capture_attention():
        averaged = model.memory_attention_maps(observation, state)
        heads = int(averaged["num_heads"])
        per_head = [
            np.asarray(model.memory_attention_maps(observation, state, head=index)["memory_to_top"])
            for index in range(heads)
        ]
        with pytest.raises(ValueError, match="outside the layer"):
            model.memory_attention_maps(observation, state, head=heads)

    assert heads > 1
    assert int(averaged["head"]) == -1
    # The default must be exactly the mean of the individual heads: that identity is what makes a
    # low average interpretable as "diluted by other heads" rather than a different computation.
    assert np.allclose(np.mean(per_head, axis=0), np.asarray(averaged["memory_to_top"]), atol=1e-5)


def test_head_average_can_hide_a_single_specialized_head():
    # A synthetic check on the reduction itself: one head routing hard at a region is pulled
    # toward zero by sink heads, which is precisely why the per-head sweep exists.
    focused = np.zeros(8)
    focused[0] = 0.40
    diluted = float(np.mean(focused))
    assert diluted == pytest.approx(0.05)
    assert focused.max() / max(diluted, 1e-9) == pytest.approx(8.0)


def test_diagnostic_never_writes_memory(fixture):
    config, model, observation, state = fixture
    before = jax.tree.map(np.asarray, state)
    with model.capture_attention():
        model.memory_attention_maps(observation, state)
    after = jax.tree.map(np.asarray, state)
    for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
        assert np.array_equal(a, b), "attention extraction must leave the fast state untouched"


def test_capture_flag_is_restored_and_required(fixture):
    config, model, observation, state = fixture
    module = model.PaliGemma.llm.module
    assert module.return_attn_probs is False
    with model.capture_attention():
        assert module.return_attn_probs is True
    assert module.return_attn_probs is False, "the flag must not leak out of the context"

    with pytest.raises(RuntimeError, match="capture_attention"):
        model.memory_attention_maps(observation, state)


def test_subtask_rows_report_memory_versus_vision_share(fixture):
    config, model, observation, state = fixture
    tokens, mask = _forced_subtask(config)
    with model.capture_attention():
        maps = model.memory_attention_maps(
            observation, state, forced_subtask_tokens=tokens, forced_subtask_mask=mask
        )
    live = np.asarray(maps["subtask_token_mask"])[0]
    memory_share = np.asarray(maps["subtask_to_memory_mass"])[0][live]
    top_share = np.asarray(maps["subtask_to_top_mass"])[0][live]
    assert memory_share.shape[0] == 3
    assert np.all(memory_share >= 0)
    assert np.all(memory_share <= 1 + 1e-5)
    assert np.all(top_share >= 0)
    assert np.all(top_share <= 1 + 1e-5)
    assert np.asarray(maps["subtask_to_memory"]).shape[-1] == int(maps["top_camera_tokens"])


def test_layer_selection_is_validated_and_selects_distinct_layers(fixture):
    config, model, observation, state = fixture
    # ``memory_gate`` is zero-initialized, so in an untrained model every memory token is the
    # same zero vector and all 256 rows are necessarily identical and uniform. Open the gate and
    # vary the images to reproduce the trained-checkpoint regime this diagnostic is used in.
    model.memory_gate.value = jnp.full_like(model.memory_gate.value, 0.5)
    varied = dataclasses.replace(
        observation,
        images={
            name: jax.random.uniform(jax.random.key(index), image.shape, minval=-1.0, maxval=1.0)
            for index, (name, image) in enumerate(observation.images.items())
        },
    )
    written, _ = model.memory.write(state, jax.random.normal(jax.random.key(7), (1, 256, config.memory.d_input)))
    try:
        with model.capture_attention():
            default = np.asarray(model.memory_attention_maps(varied, written)["memory_to_top"])
            other = np.asarray(model.memory_attention_maps(varied, written, layer=0)["memory_to_top"])
            assert default.std() > 0, "an open gate and varied image must produce non-uniform attention"
            assert not np.allclose(default, other), "different layers must give different maps"
            with pytest.raises(ValueError, match="outside the model's depth"):
                model.memory_attention_maps(varied, written, layer=999)
    finally:
        model.memory_gate.value = jnp.zeros_like(model.memory_gate.value)


def test_forced_subtask_arguments_must_be_paired(fixture):
    config, model, observation, state = fixture
    tokens, _ = _forced_subtask(config)
    with model.capture_attention(), pytest.raises(ValueError, match="must be provided together"):
        model.memory_attention_maps(observation, state, forced_subtask_tokens=tokens)
