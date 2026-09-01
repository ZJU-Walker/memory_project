"""Focused model-level contracts for the opt-in v3.5 sequence path."""

# ruff: noqa: SLF001

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import memory
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models.pi0_v32_test import _single_observation
from openpi.models.pi0_v32_test import _TinyV32


class _AugmentationOnly:
    """Bind Pi0's pure augmentation helper without constructing the multi-billion-param model."""

    _augment_sequence_images = pi0.Pi0._augment_sequence_images

    def __init__(self, *, time_consistent: bool):
        self.memory_time_consistent_augmentation = time_consistent


def _repeated_sequence(batch: int = 3, steps: int = 4, size: int = 32):
    base = jax.random.uniform(jax.random.key(7), (size, size, 3), minval=-1.0, maxval=1.0)
    sequence = jnp.broadcast_to(base, (batch, steps, size, size, 3))
    return {
        "base_0_rgb": sequence,
        "left_wrist_0_rgb": sequence,
        "right_wrist_0_rgb": sequence,
    }


def test_v35_sequence_augmentation_reuses_parameters_over_time_and_splits_samples():
    images = _repeated_sequence()
    augmented = _AugmentationOnly(time_consistent=True)._augment_sequence_images(jax.random.key(35), images)

    for value in augmented.values():
        assert value.shape == images["base_0_rgb"].shape
        assert value.dtype == images["base_0_rgb"].dtype
        # Identical source frames under one sample/camera receive exactly one transform.
        np.testing.assert_array_equal(value[:, 1:], jnp.repeat(value[:, :1], value.shape[1] - 1, axis=1))
        # Separate samples have separate keys. Color jitter alone makes this true for wrists;
        # the top camera also independently samples its spatial transform.
        assert any(not np.array_equal(np.asarray(value[0]), np.asarray(value[i])) for i in range(1, value.shape[0]))


def test_legacy_sequence_augmentation_still_samples_each_frame_independently():
    images = _repeated_sequence(batch=1)
    augmented = _AugmentationOnly(time_consistent=False)._augment_sequence_images(jax.random.key(35), images)

    # This guards the default-off compatibility branch and, in particular, its b*T key split.
    for value in augmented.values():
        assert not np.array_equal(np.asarray(value[:, 1:]), np.asarray(value[:, :1]))


def test_side_feature_cotangent_cap_is_forward_identity_and_per_example():
    x = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    weights = jnp.asarray([[100.0, 0.0], [0.03, 0.04]], dtype=jnp.float32)
    limit = 0.1

    forward = pi0._clip_feature_cotangent(x, limit)
    np.testing.assert_array_equal(forward, x)
    grad = jax.grad(lambda value: jnp.sum(pi0._clip_feature_cotangent(value, limit) * weights))(x)
    norms = jnp.linalg.norm(grad, axis=-1)
    np.testing.assert_allclose(norms, jnp.asarray([limit, 0.05]), rtol=1e-6, atol=1e-7)


def test_side_ce_caps_each_unweighted_term_before_global_scaling_and_keeps_head_grad_exact():
    feature = jnp.asarray([[2.0, -1.0, 0.5], [-0.5, 1.5, 2.0]], dtype=jnp.float32)
    kernel = jnp.asarray([[3.0, -2.0], [1.0, 2.5], [-1.5, 1.0]], dtype=jnp.float32)
    bias = jnp.asarray([0.2, -0.1], dtype=jnp.float32)
    label = jnp.asarray([0, 1], dtype=jnp.int32)
    downstream_scale = jnp.asarray([0.1, 2.0], dtype=jnp.float32)
    limit = 0.4

    def loss(feat, weight, offset):
        ce, _ = pi0._side_ce_with_per_term_feature_cap(feat, weight, offset, label, limit)
        return jnp.sum(ce * downstream_scale)

    feature_grad, kernel_grad, bias_grad = jax.grad(loss, argnums=(0, 1, 2))(feature, kernel, bias)
    logits = feature @ kernel + bias
    error = jax.nn.softmax(logits, axis=-1) - jax.nn.one_hot(label, 2)
    raw_feature_grad = error @ kernel.T
    raw_norm = jnp.linalg.norm(raw_feature_grad, axis=-1)
    expected_feature_grad = (
        raw_feature_grad * jnp.minimum(1.0, limit / (raw_norm + 1e-12))[:, None] * downstream_scale[:, None]
    )
    expected_logits_grad = error * downstream_scale[:, None]

    np.testing.assert_allclose(feature_grad, expected_feature_grad, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(kernel_grad, feature.T @ expected_logits_grad, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(bias_grad, jnp.sum(expected_logits_grad, axis=0), rtol=1e-5, atol=1e-6)


def _v35_config(**kwargs):
    fields = {
        "pi05": True,
        "predict_subtask": True,
        "predict_with_memory": True,
        "memory_architecture": "v32_layer8_dual_query",
        "memory_layer": 8,
        "memory_write_source": "query_compressed",
        "memory_injection_mode": "tanh_rms",
        "memory_injection_gate_init": 0.5,
        "memory_freeze_injection_gate": True,
        "memory_v35_enabled": True,
        "memory_write_side_loss_weight": 0.3,
        "memory_read_side_loss_weight": 0.3,
        "memory_time_consistent_augmentation": True,
        "memory": memory.MemoryConfig(
            write_rule="delta_output",
            association_mode="pooled_frame",
            blank_initial_output=True,
            drift_radius=None,
        ),
    }
    fields.update(kwargs)
    return pi0_config.Pi0Config(**fields)


def test_v35_config_accepts_only_the_sealed_core_contract():
    config = _v35_config()
    assert config.memory.write_rule == "delta_output"
    assert config.memory.association_mode == "pooled_frame"

    with pytest.raises(ValueError, match="pooled-frame delta_output"):
        _v35_config(memory=memory.MemoryConfig())
    with pytest.raises(ValueError, match="time-consistent"):
        _v35_config(memory_time_consistent_augmentation=False)


class _TinyV35(_TinyV32):
    capture_attention = pi0.Pi0.capture_attention
    v35_action_memory_attention_step = pi0.Pi0.v35_action_memory_attention_step
    v35_paired_task_health_step = pi0.Pi0.v35_paired_task_health_step
    _v35_oracle_injected_content = pi0.Pi0._v35_oracle_injected_content
    _v35_inference_mask = staticmethod(pi0.Pi0._v35_inference_mask)
    _v35_read_geometry = pi0.Pi0._v35_read_geometry
    _v35_inference_transition = pi0.Pi0._v35_inference_transition

    def __init__(self, rngs: nnx.Rngs):
        super().__init__(rngs)
        self.memory = memory.TitansMemory(
            memory.MemoryConfig(
                d_input=64,
                d_key=8,
                hidden_dims=(8,),
                d_value=64,
                mlp_l2norm=True,
                blank_initial_output=True,
                write_rule="delta_output",
                association_mode="pooled_frame",
                delta_rate=1.0,
                alpha_step=0.01,
            ),
            rngs=rngs,
        )
        self.memory_v35_enabled = True
        self.memory_write_side_loss_weight = 0.3
        self.memory_read_side_loss_weight = 0.3
        self.memory_side_feature_cotangent_clip = 10.0
        self.memory_num_side_cells = 8
        self.memory_time_consistent_augmentation = True
        self.memory_write_side_head = nnx.Linear(64, 2, rngs=rngs)
        self.memory_read_side_head = nnx.Linear(64, 2, rngs=rngs)
        self.memory_ladder_probes = False


@pytest.fixture(scope="module")
def tiny_v35():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        yield _TinyV35(nnx.Rngs(35))
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def _v35_sequence_observation(*, boundary_at_d: bool = False, d_gap: int = 2):
    observation = _single_observation()
    steps = 3

    def repeat_time(value):
        return jnp.repeat(value[:, None], steps, axis=1)

    return observation.replace(
        images={name: repeat_time(image) for name, image in observation.images.items()},
        image_masks={name: repeat_time(mask) for name, mask in observation.image_masks.items()},
        state=repeat_time(observation.state),
        tokenized_prompt=repeat_time(observation.tokenized_prompt),
        tokenized_prompt_mask=repeat_time(observation.tokenized_prompt_mask),
        token_ar_mask=repeat_time(observation.token_ar_mask),
        token_loss_mask=repeat_time(observation.token_loss_mask),
        token_fast_mask=repeat_time(observation.token_fast_mask),
        tokenized_causal=jnp.asarray([[[5, 6], [7, 8], [5, 8]]], dtype=jnp.int32),
        tokenized_causal_mask=jnp.ones((1, steps, 2), dtype=bool),
        causal_fast_mask=jnp.zeros((1, steps, 2), dtype=bool),
        seq_step_mask=jnp.ones((1, steps), dtype=bool),
        seq_block_boundary=jnp.asarray([[False, False, boundary_at_d]]),
        seq_write_mask=jnp.asarray([[True, False, False]]),
        seq_decision_mask=jnp.asarray([[False, False, True]]),
        seq_read_state_valid=jnp.asarray([[False, False, True]]),
        seq_read_credit_reachable=jnp.asarray([[False, False, not boundary_at_d]]),
        seq_decay_gap_before=jnp.asarray([[0, 0, d_gap]], dtype=jnp.int32),
        seq_use_pressure_mask=jnp.asarray([[False, False, True]]),
        seq_occlusion_mask=jnp.asarray([[False, True, False]]),
        seq_sparse_skip_o=jnp.asarray([d_gap > 0]),
        seq_episode_index=jnp.asarray([11], dtype=jnp.int32),
        seq_collection_id=jnp.asarray([1], dtype=jnp.int32),
        seq_object_id=jnp.asarray([0], dtype=jnp.int32),
        seq_memory_cell=jnp.asarray([3], dtype=jnp.int32),
        seq_side_label=jnp.asarray([1], dtype=jnp.int32),
    )


def test_v35_sequence_commits_only_e_reads_d_and_uses_sparse_gap(tiny_v35):
    observation = _v35_sequence_observation()
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    losses = tiny_v35._compute_sequence_loss_v32(jax.random.key(41), observation, actions, train=False)

    np.testing.assert_array_equal(losses["v35_write_eligible_count"], 1.0)
    np.testing.assert_array_equal(losses["v35_commit_success_count"], 1.0)
    np.testing.assert_array_equal(losses["v35_read_state_valid_count"], 1.0)
    np.testing.assert_array_equal(losses["v35_reachable_count"], 1.0)
    np.testing.assert_array_equal(losses["v35_state_invalid_d_count"], 0.0)
    np.testing.assert_array_equal(losses["v35_invalid_gap_count"], 0.0)
    np.testing.assert_array_equal(losses["v35_use_pressure_count"], 1.0)
    np.testing.assert_array_equal(losses["v35_write_episode_count_cell"], jax.nn.one_hot(3, 8))
    np.testing.assert_array_equal(losses["v35_read_episode_count_cell"], jax.nn.one_hot(3, 8))
    assert float(losses["v35_commit_relative_residual_max"]) <= 1e-5
    for key, value in losses.items():
        assert np.isfinite(np.asarray(value)).all(), key


def test_v35_negative_sparse_gap_fails_closed(tiny_v35):
    observation = _v35_sequence_observation(d_gap=-1)
    actions = jnp.zeros((1, 3, 4, 2), dtype=jnp.float32)
    losses = tiny_v35._compute_sequence_loss_v32(jax.random.key(42), observation, actions, train=False)

    np.testing.assert_array_equal(losses["v35_invalid_gap_count"], 1.0)
    np.testing.assert_array_equal(losses["v35_read_state_valid_count"], 0.0)
    np.testing.assert_array_equal(losses["v35_read_episode_count_cell"], jnp.zeros((8,)))


def test_v35_inference_transition_is_per_sample_e_only_and_fail_closed(tiny_v35):
    batch = 4
    seed_tokens = jax.random.normal(jax.random.key(51), (batch, 16, 64))
    write_tokens = jax.random.normal(jax.random.key(52), (batch, 16, 64))
    state, _ = tiny_v35.memory.write(tiny_v35.memory.init_state(batch), seed_tokens)
    write_candidate, write_aux = tiny_v35.memory.write(state, write_tokens)
    decay_candidate, _ = tiny_v35.memory.decay_step(state, write_tokens)
    transition_valid = jnp.asarray([True, True, True, False])
    write_mask = jnp.asarray([True, False, True, True])

    transitioned, aux = tiny_v35._v35_inference_transition(
        state,
        write_tokens,
        transition_valid=transition_valid,
        write_mask=write_mask,
        write_mode="normal",
    )

    def expected_leaf(write_leaf, decay_leaf, old_leaf):
        expected = np.asarray(old_leaf).copy()
        expected[0] = np.asarray(write_leaf)[0]
        expected[1] = np.asarray(decay_leaf)[1]
        expected[2] = np.asarray(write_leaf)[2]
        return expected

    jax.tree.map(
        lambda actual, write, decay, old: np.testing.assert_array_equal(actual, expected_leaf(write, decay, old)),
        transitioned,
        write_candidate,
        decay_candidate,
        state,
    )
    expected_commits = np.asarray([write_aux["commit_applied"][0], False, write_aux["commit_applied"][2], False])
    np.testing.assert_array_equal(aux["write_occurred"], expected_commits)
    np.testing.assert_array_equal(aux["v35_transition_applied"], [True, True, True, False])
    np.testing.assert_array_equal(aux["v35_commit_requested"], [True, False, True, False])
    np.testing.assert_array_equal(aux["v35_commit_applied"], expected_commits)
    np.testing.assert_array_equal(aux["v35_decay_only"], ~expected_commits & transition_valid)
    np.testing.assert_array_equal(aux["v35_noop"], [False, False, False, True])
    np.testing.assert_array_equal(aux["v35_invalid_write_request"], [False, False, False, True])

    # Omitted runtime phase metadata is a strict no-op, even though write_mode is normal.
    frozen, frozen_aux = tiny_v35._v35_inference_transition(
        state, write_tokens, transition_valid=None, write_mask=None, write_mode="normal"
    )
    jax.tree.map(np.testing.assert_array_equal, frozen, state)
    np.testing.assert_array_equal(frozen_aux["v35_noop"], np.ones((batch,), dtype=bool))
    np.testing.assert_array_equal(frozen_aux["write_occurred"], np.zeros((batch,), dtype=bool))


def test_v35_inference_write_mode_remains_an_intervention(tiny_v35):
    batch = 2
    write_tokens = jax.random.normal(jax.random.key(53), (batch, 16, 64))
    state, _ = tiny_v35.memory.write(tiny_v35.memory.init_state(batch), write_tokens * 0.5)
    expected_decay, _ = tiny_v35.memory.decay_step(state, write_tokens)

    dynamics, dynamics_aux = tiny_v35._v35_inference_transition(
        state,
        write_tokens,
        transition_valid=True,
        write_mask=True,
        write_mode="dynamics_only",
    )
    jax.tree.map(np.testing.assert_array_equal, dynamics, expected_decay)
    np.testing.assert_array_equal(dynamics_aux["v35_decay_only"], np.ones((batch,), dtype=bool))
    np.testing.assert_array_equal(dynamics_aux["write_occurred"], np.zeros((batch,), dtype=bool))

    frozen, frozen_aux = tiny_v35._v35_inference_transition(
        state,
        write_tokens,
        transition_valid=True,
        write_mask=True,
        write_mode="frozen",
    )
    jax.tree.map(np.testing.assert_array_equal, frozen, state)
    np.testing.assert_array_equal(frozen_aux["v35_noop"], np.ones((batch,), dtype=bool))


def test_v35_inference_masks_reject_accidental_broadcasting(tiny_v35):
    write_tokens = jnp.zeros((2, 16, 64), dtype=jnp.float32)
    with pytest.raises(ValueError, match="v35_transition_valid"):
        tiny_v35._v35_inference_transition(
            tiny_v35.memory.init_state(2),
            write_tokens,
            transition_valid=jnp.ones((2, 1), dtype=bool),
            write_mask=False,
            write_mode="normal",
        )
    with pytest.raises(TypeError, match="bool dtype"):
        tiny_v35._v35_inference_transition(
            tiny_v35.memory.init_state(2),
            write_tokens,
            transition_valid=True,
            write_mask=jnp.ones((2,), dtype=jnp.int32),
            write_mode="normal",
        )


def test_v35_sample_defaults_to_no_transition_without_phase_metadata(tiny_v35):
    observation = _single_observation()
    state, _ = tiny_v35.memory.write(tiny_v35.memory.init_state(1), jax.random.normal(jax.random.key(54), (1, 16, 64)))
    kwargs = {
        "stop_token": 1,
        "max_decode_steps": 1,
        "num_steps": 1,
        "noise": jnp.zeros((1, 4, 2), dtype=jnp.float32),
        "forced_subtask_tokens": jnp.asarray([[5, 6]], dtype=jnp.int32),
        "forced_subtask_mask": jnp.ones((1, 2), dtype=bool),
    }

    _, unchanged, aux = tiny_v35.sample_with_memory(jax.random.key(55), observation, state, **kwargs)
    jax.tree.map(np.testing.assert_array_equal, unchanged, state)
    np.testing.assert_array_equal(aux["v35_noop"], [True])
    np.testing.assert_array_equal(aux["write_occurred"], [False])

    _, zero_state, zero_aux = tiny_v35.sample_with_memory(
        jax.random.key(55), observation, state, zero_read=True, **kwargs
    )
    jax.tree.map(np.testing.assert_array_equal, zero_state, state)
    np.testing.assert_array_equal(zero_aux["v35_injected_pre_cast_rms"], [0.0])
    np.testing.assert_array_equal(zero_aux["v35_injected_post_cast_rms"], [0.0])

    interface = tiny_v35.v32_memory_interface_step(observation, state)
    expected, expected_aux = tiny_v35.memory.write(state, interface["write_tokens"])
    _, committed, commit_aux = tiny_v35.sample_with_memory(
        jax.random.key(55),
        observation,
        state,
        v35_transition_valid=True,
        v35_write_mask=True,
        **kwargs,
    )
    jax.tree.map(np.testing.assert_array_equal, committed, expected)
    np.testing.assert_array_equal(commit_aux["write_occurred"], expected_aux["commit_applied"])


def test_v35_oracles_share_one_exact_rms_pinning_path_and_fail_closed(tiny_v35):
    directions = jnp.stack([jnp.arange(1, 65, dtype=jnp.float32), -jnp.arange(1, 65, dtype=jnp.float32)])
    targets = jnp.asarray([0.2, 0.35], dtype=jnp.float32)
    injected, aux = tiny_v35._v35_oracle_injected_content(directions, targets, num_slots=16)

    assert injected.dtype == jnp.float32
    assert injected.shape == (2, 16, 64)
    np.testing.assert_allclose(jnp.sqrt(jnp.mean(jnp.square(injected), axis=(1, 2))), targets, rtol=1e-6)
    np.testing.assert_array_equal(aux["v35_oracle_injection_valid"], [True, True])
    # Correct-side and opposite-side donors differ only in direction, never scale.
    np.testing.assert_allclose(injected[1] / targets[1], -(injected[0] / targets[0]), rtol=1e-6, atol=1e-6)

    invalid, invalid_aux = tiny_v35._v35_oracle_injected_content(
        jnp.stack([jnp.zeros(64), jnp.full((64,), jnp.nan)]), targets, num_slots=16
    )
    np.testing.assert_array_equal(invalid, np.zeros((2, 16, 64), dtype=np.float32))
    np.testing.assert_array_equal(invalid_aux["v35_oracle_injection_valid"], [False, False])

    invalid_rms, invalid_rms_aux = tiny_v35._v35_oracle_injected_content(
        directions, jnp.asarray([jnp.nan, -1.0], dtype=jnp.float32), num_slots=16
    )
    np.testing.assert_array_equal(invalid_rms, np.zeros((2, 16, 64), dtype=np.float32))
    np.testing.assert_array_equal(invalid_rms_aux["v35_oracle_injection_valid"], [False, False])
    np.testing.assert_array_equal(invalid_rms_aux["v35_oracle_target_rms"], [0.0, 0.0])


def test_v35_query_geometry_matches_rank_one_anchor_after_exact_decay(tiny_v35):
    write_tokens = jax.random.normal(jax.random.key(61), (1, 16, 64))
    blank = tiny_v35.memory.init_state(1)
    state, write_aux = tiny_v35.memory.write(blank, write_tokens)
    state, _ = tiny_v35.memory.analytic_decay(state, 3)
    read_queries = jax.random.normal(jax.random.key(62), (1, 16, 64))
    retrieved = tiny_v35.memory.read(state, read_queries)
    geometry = tiny_v35._v35_read_geometry(
        state,
        read_queries,
        retrieved,
        write_aux["pooled_key"],
        write_aux["pooled_value"],
        jnp.asarray([3], dtype=jnp.int32),
    )

    np.testing.assert_array_equal(geometry["v35_geometry_valid"], [True])
    assert geometry["v35_query_anchor_cosine"].shape == (1, 16)
    assert geometry["v35_query_anchor_beta"].shape == (1, 16)
    np.testing.assert_allclose(geometry["v35_anchor_retention"], 0.99**3, rtol=1e-6)
    assert float(geometry["v35_anchor_predicted_read_relative_residual"][0]) <= 1e-5
    assert 0.0 <= float(geometry["v35_query_cancellation_ratio"][0]) <= 1.0 + 1e-6
    assert 0.0 <= float(geometry["v35_query_beta_sign_consistency"][0]) <= 1.0

    invalid = tiny_v35._v35_read_geometry(
        state,
        read_queries,
        retrieved,
        write_aux["pooled_key"],
        write_aux["pooled_value"],
        jnp.asarray([-1], dtype=jnp.int32),
    )
    np.testing.assert_array_equal(invalid["v35_geometry_valid"], [False])
    np.testing.assert_array_equal(invalid["v35_query_anchor_beta"], np.zeros((1, 16), dtype=np.float32))


def test_v35_sample_oracle_bypasses_retrieval_at_requested_injected_rms(tiny_v35):
    observation = _single_observation()
    state = tiny_v35.memory.init_state(1)
    kwargs = {
        "stop_token": 1,
        "max_decode_steps": 1,
        "num_steps": 1,
        "noise": jnp.zeros((1, 4, 2), dtype=jnp.float32),
        "forced_subtask_tokens": jnp.asarray([[5, 6]], dtype=jnp.int32),
        "forced_subtask_mask": jnp.ones((1, 2), dtype=bool),
    }
    _, unchanged, aux = tiny_v35.sample_with_memory(
        jax.random.key(63),
        observation,
        state,
        v35_oracle_direction=jnp.ones((1, 64), dtype=jnp.float32),
        v35_oracle_injected_rms=jnp.asarray([0.25], dtype=jnp.float32),
        **kwargs,
    )
    jax.tree.map(np.testing.assert_array_equal, unchanged, state)
    np.testing.assert_array_equal(aux["v35_oracle_injection_active"], [True])
    np.testing.assert_array_equal(aux["v35_oracle_injection_valid"], [True])
    np.testing.assert_allclose(aux["v35_oracle_actual_rms"], 0.25, rtol=1e-6)

    with pytest.raises(ValueError, match="provided together"):
        tiny_v35.sample_with_memory(
            jax.random.key(63),
            observation,
            state,
            v35_oracle_direction=jnp.ones((1, 64), dtype=jnp.float32),
            **kwargs,
        )


def test_v35_action_attention_is_actual_suffix_mass_and_read_only(tiny_v35):
    observation = _single_observation()
    state, _ = tiny_v35.memory.write(
        tiny_v35.memory.init_state(1), jax.random.normal(jax.random.key(70), (1, 16, 64))
    )
    before = jax.tree.map(np.asarray, state)
    kwargs = {
        "action_noise": jax.random.normal(jax.random.key(71), (1, 4, 2)),
        "forced_subtask_tokens": jnp.asarray([[5, 6]], dtype=jnp.int32),
        "forced_subtask_mask": jnp.ones((1, 2), dtype=bool),
    }
    with tiny_v35.capture_attention():
        result = tiny_v35.v35_action_memory_attention_step(observation, state, **kwargs)

    mass = np.asarray(result["action_to_memory_mass"])
    baseline = np.asarray(result["uniform_baseline"])
    assert mass.shape == baseline.shape == (1,)
    assert np.all((mass >= 0.0) & (mass <= 1.0))
    assert np.all((baseline > 0.0) & (baseline < 1.0))
    np.testing.assert_allclose(
        np.mean(np.asarray(result["action_to_memory_mass_per_action"]), axis=-1), mass, atol=1e-6
    )
    np.testing.assert_allclose(
        np.mean(np.asarray(result["uniform_baseline_per_action"]), axis=-1), baseline, atol=1e-6
    )
    jax.tree.map(np.testing.assert_array_equal, before, state)

    with pytest.raises(RuntimeError, match="capture_attention"):
        tiny_v35.v35_action_memory_attention_step(observation, state, **kwargs)


def test_v35_paired_task_health_freezes_inputs_and_does_not_write(tiny_v35):
    observation = _single_observation().replace(
        tokenized_causal=jnp.asarray([[5, 6]], dtype=jnp.int32),
        tokenized_causal_mask=jnp.ones((1, 2), dtype=bool),
        causal_fast_mask=jnp.asarray([[False, True]], dtype=bool),
    )
    state = tiny_v35.memory.init_state(1)
    before = jax.tree.map(np.asarray, state)
    result = tiny_v35.v35_paired_task_health_step(
        observation,
        jnp.zeros((1, 4, 2), dtype=jnp.float32),
        state,
        action_noise=jax.random.normal(jax.random.key(72), (1, 4, 2)),
        flow_time=jnp.asarray([0.4], dtype=jnp.float32),
    )
    for name in (
        "source_flow_loss",
        "source_subtask_ce",
        "memory_flow_loss",
        "memory_subtask_ce",
    ):
        value = np.asarray(result[name])
        assert value.shape == (1,)
        assert np.isfinite(value).all()
        assert np.all(value >= 0.0)
    jax.tree.map(np.testing.assert_array_equal, before, state)
