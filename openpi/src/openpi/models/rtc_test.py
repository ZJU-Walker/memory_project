import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import memory
from openpi.models import model as model_lib
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models import rtc
from openpi.shared import nnx_utils


class _TinyPi05Sampler(nnx.Module):
    """Small action-only π0.5 graph: real Gemma/AdaRMS, no SigLIP."""

    embed_suffix = pi0.Pi0.embed_suffix
    compute_loss = pi0.Pi0.compute_loss
    sample_actions = pi0.Pi0.sample_actions
    _check_action_prefix_shapes = pi0.Pi0._check_action_prefix_shapes  # noqa: SLF001

    def __init__(self, rngs: nnx.Rngs, *, with_image_encoder: bool = False):
        config = gemma.get_config("dummy")
        llm = nnx_bridge.ToNNX(gemma.Module(configs=[config, config], embed_dtype="float32", adarms=True))
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        if with_image_encoder:
            self.PaliGemma = nnx.Dict(llm=llm, img=_TinyImageEncoder(config.width, rngs))
        else:
            self.PaliGemma = nnx.Dict(llm=llm)
        self.action_horizon = 4
        self.action_dim = 2
        self.max_token_len = 4
        self.pi05 = True
        self.simulated_delay = 2
        self.predict_subtask = False
        self.predict_with_memory = False
        self.action_in_proj = nnx.Linear(self.action_dim, config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(config.width, config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(config.width, config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(config.width, self.action_dim, rngs=rngs)

    def embed_prefix(self, obs):
        batch = obs.state.shape[0]
        tokens = jnp.zeros((batch, 1, 64), dtype=jnp.float32)
        mask = jnp.ones((batch, 1), dtype=bool)
        return tokens, mask, jnp.zeros_like(mask, dtype=jnp.int32)


class _TinyImageEncoder(nnx.Module):
    def __init__(self, width: int, rngs: nnx.Rngs):
        self.proj = nnx.Linear(3, width, rngs=rngs)

    def __call__(self, image, *, train: bool = False):
        return self.proj(jnp.mean(image, axis=(1, 2)))[:, None, :], None


class _TinyPi05MemorySampler(_TinyPi05Sampler):
    sample_with_memory = pi0.Pi0.sample_with_memory
    _select_memory_write_source = pi0.Pi0._select_memory_write_source  # noqa: SLF001

    def __init__(self, rngs: nnx.Rngs, *, memory_write_source: pi0_config.MemoryWriteSource = "raw_hidden"):
        super().__init__(rngs, with_image_encoder=True)
        self.predict_with_memory = True
        self.memory_layer = 0
        self.memory_write_source = memory_write_source
        self.causal_token_len = 2
        memory_config = memory.MemoryConfig(d_input=64, d_key=8, hidden_dims=(8,), d_value=64)
        self.memory = memory.TitansMemory(memory_config, rngs=rngs)
        self.memory_gate = nnx.Param(jnp.ones((64,), dtype=jnp.float32))
        self.memory_probe_weight = 0.0


class _TinyPi05SequenceTrainer(_TinyPi05MemorySampler):
    embed_prefix = pi0.Pi0.embed_prefix
    _compute_sequence_loss = pi0.Pi0._compute_sequence_loss  # noqa: SLF001


def _attention_write_representations(model, state, observation, causal_tokens, causal_mask):
    """Recomputes one training-side memory extension and returns its raw/post-attention writes."""
    observation = model_lib.preprocess_observation(None, observation, train=False)
    batch = observation.state.shape[0]
    prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(observation)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache, hidden = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=pi0.make_attn_mask(prefix_mask, prefix_ar),
        positions=positions,
        return_hidden_states=True,
    )
    prefix_len = prefix_mask.shape[1]
    num_img = prefix_len - model.max_token_len
    mem_len = num_img // len(observation.images)
    raw_hidden = hidden[0][model.memory_layer][:, :mem_len].astype(jnp.float32)

    retrieved = model.memory.read(state, raw_hidden)
    mem_tokens = (model.memory_gate.value * retrieved).astype(prefix_tokens.dtype)
    causal_len = causal_tokens.shape[1]
    causal_emb = model.PaliGemma.llm(causal_tokens, method="embed")
    ext_tokens = jnp.concatenate([mem_tokens, causal_emb], axis=1)
    mem_rows = pi0.make_memory_step_mask(prefix_mask, prefix_ar, mem_len, causal_len)
    causal_rows = jnp.concatenate(
        [
            jnp.broadcast_to(prefix_mask[:, None, :], (batch, causal_len, prefix_len)),
            jnp.ones((batch, causal_len, mem_len), dtype=bool),
            jnp.tril(jnp.ones((causal_len, causal_len), dtype=bool))[None] & causal_mask[:, None, :],
        ],
        axis=-1,
    )
    ext_mask = jnp.concatenate([mem_rows, causal_rows], axis=1)
    ext_positions = jnp.broadcast_to(prefix_len + jnp.arange(mem_len + causal_len)[None], (batch, mem_len + causal_len))
    kv_cache = jax.tree.map(lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, mem_len + causal_len), (0, 0), (0, 0))), kv_cache)
    (ext_out, _), _ = model.PaliGemma.llm(
        [ext_tokens, None],
        mask=ext_mask,
        positions=ext_positions,
        kv_cache=kv_cache,
        cache_position=prefix_len,
    )
    return raw_hidden, ext_out[:, :mem_len].astype(jnp.float32)


def _assert_trees_equal(actual, expected):
    jax.tree.map(np.testing.assert_array_equal, actual, expected)


def test_simulated_delay_config_is_inclusive_and_validated():
    assert pi0_config.Pi0Config(action_horizon=5, simulated_delay=0).simulated_delay == 0
    assert pi0_config.Pi0Config(action_horizon=5, simulated_delay=4).simulated_delay == 4
    with pytest.raises(ValueError, match="simulated_delay"):
        pi0_config.Pi0Config(action_horizon=5, simulated_delay=-1)
    with pytest.raises(ValueError, match="simulated_delay"):
        pi0_config.Pi0Config(action_horizon=5, simulated_delay=5)


def test_memory_write_source_config_is_validated_and_defaults_to_v3():
    assert pi0_config.Pi0Config().memory_write_source == "raw_hidden"
    assert pi0_config.Pi0Config(memory_write_source="post_attention").memory_write_source == "post_attention"
    with pytest.raises(ValueError, match="memory_write_source"):
        pi0_config.Pi0Config(memory_write_source="invalid")  # type: ignore[arg-type]


def test_memory_write_modes_select_the_expected_float32_tensor_and_keep_the_parameter_tree():
    raw_model = _TinyPi05MemorySampler(nnx.Rngs(0), memory_write_source="raw_hidden")
    post_model = _TinyPi05MemorySampler(nnx.Rngs(0), memory_write_source="post_attention")
    raw_hidden = jnp.ones((1, 2, 64), dtype=jnp.bfloat16)
    post_attention = jnp.full((1, 2, 64), 2, dtype=jnp.bfloat16)

    raw_write = raw_model._select_memory_write_source(raw_hidden, post_attention)  # noqa: SLF001
    post_write = post_model._select_memory_write_source(raw_hidden, post_attention)  # noqa: SLF001
    assert raw_write.dtype == post_write.dtype == jnp.float32
    np.testing.assert_array_equal(raw_write, raw_hidden.astype(jnp.float32))
    np.testing.assert_array_equal(post_write, post_attention.astype(jnp.float32))
    with pytest.raises(ValueError, match="matching shapes"):
        post_model._select_memory_write_source(raw_hidden, post_attention[:, :1])  # noqa: SLF001

    raw_leaves = jax.tree_util.tree_leaves_with_path(nnx.state(raw_model))
    post_leaves = jax.tree_util.tree_leaves_with_path(nnx.state(post_model))
    assert [jax.tree_util.keystr(path) for path, _ in raw_leaves] == [
        jax.tree_util.keystr(path) for path, _ in post_leaves
    ]
    for (_, raw), (_, post) in zip(raw_leaves, post_leaves, strict=True):
        assert raw.shape == post.shape
        assert raw.dtype == post.dtype
        np.testing.assert_array_equal(raw, post)


def test_post_attention_write_is_insulated_from_teacher_forced_causal_labels():
    model = _TinyPi05SequenceTrainer(nnx.Rngs(0), memory_write_source="post_attention")
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=2,
        max_token_len=4,
        predict_subtask=True,
    )
    observation = config.fake_obs(1)
    state = model.memory.init_state(1)
    causal_mask = jnp.ones((1, model.causal_token_len), dtype=bool)
    causal_a = jnp.array([[5, 6]], dtype=jnp.int32)
    causal_b = jnp.array([[17, 23]], dtype=jnp.int32)

    raw_a, post_a = _attention_write_representations(model, state, observation, causal_a, causal_mask)
    raw_b, post_b = _attention_write_representations(model, state, observation, causal_b, causal_mask)
    np.testing.assert_array_equal(raw_a, raw_b)
    np.testing.assert_array_equal(post_a, post_b)
    assert not np.array_equal(np.asarray(raw_a), np.asarray(post_a))

    state_a, aux_a = model.memory.write(state, model._select_memory_write_source(raw_a, post_a))  # noqa: SLF001
    state_b, aux_b = model.memory.write(state, model._select_memory_write_source(raw_b, post_b))  # noqa: SLF001
    _assert_trees_equal(state_a, state_b)
    _assert_trees_equal(aux_a, aux_b)


def test_train_time_rtc_keeps_prefix_clean_and_renormalizes_suffix():
    actions = jnp.arange(2 * 5 * 2, dtype=jnp.float32).reshape(2, 5, 2)
    noise = -actions - 1
    time = jnp.array([0.25, 0.75], dtype=jnp.float32)
    delay = jnp.array([2, 0], dtype=jnp.int32)

    x_t, token_time, mask = rtc.make_noisy_actions(actions, noise, time, delay=delay)
    np.testing.assert_array_equal(x_t[0, :2], actions[0, :2])
    np.testing.assert_array_equal(token_time[0, :2], 0)
    np.testing.assert_allclose(x_t[0, 2:], 0.25 * noise[0, 2:] + 0.75 * actions[0, 2:])
    np.testing.assert_array_equal(mask, np.array([[False, False, True, True, True], [True] * 5]))

    renormalized = rtc.renormalize_flow_loss(jnp.ones((2, 5)), mask)
    np.testing.assert_array_equal(renormalized[0, :2], 0)
    np.testing.assert_allclose(jnp.mean(renormalized, axis=-1), jnp.ones((2,)))


def test_action_prefix_validation_conditioning_and_exact_restore():
    prefix_actions = jnp.arange(8, dtype=jnp.float32).reshape(1, 4, 2)
    prefix = rtc.ActionPrefix(
        actions=prefix_actions,
        delay=jnp.array([2], dtype=jnp.int32),
        prefix_length=jnp.array([3], dtype=jnp.int32),
    )
    rtc.validate_action_prefix(prefix, action_horizon=4, action_dim=2)

    x_t = jnp.full((1, 4, 2), -5.0)
    conditioned, token_time = rtc.condition_action_prefix(x_t, jnp.asarray(0.5), prefix)
    np.testing.assert_array_equal(conditioned[:, :2], prefix_actions[:, :2])
    np.testing.assert_array_equal(conditioned[:, 2:], x_t[:, 2:])
    np.testing.assert_array_equal(token_time[:, :2], 0)
    np.testing.assert_array_equal(token_time[:, 2:], 0.5)

    restored = rtc.restore_action_prefix(jnp.full_like(x_t, 7.0), prefix)
    np.testing.assert_array_equal(restored[:, :2], prefix_actions[:, :2])
    np.testing.assert_array_equal(restored[:, 2:], 7.0)

    invalid = prefix.replace(delay=jnp.array([4]), prefix_length=jnp.array([3]))
    with pytest.raises(ValueError, match="0 <= delay"):
        rtc.validate_action_prefix(invalid, action_horizon=4, action_dim=2)


def test_tokenwise_timestep_embedding_matches_repeated_scalar_embedding():
    scalar = jnp.array([0.37], dtype=jnp.float32)
    tokenwise = jnp.broadcast_to(scalar[:, None], (1, 4))
    scalar_emb = pi0.posemb_sincos(scalar, 8, min_period=4e-3, max_period=4.0)
    token_emb = pi0.posemb_sincos(tokenwise, 8, min_period=4e-3, max_period=4.0)
    np.testing.assert_allclose(token_emb, jnp.broadcast_to(scalar_emb[:, None], token_emb.shape), rtol=1e-6)


def test_adarms_accepts_per_example_and_per_token_condition_with_same_parameters():
    norm = gemma.RMSNorm()
    x = jnp.ones((2, 4, 8), dtype=jnp.float32)
    per_example = jnp.ones((2, 8), dtype=jnp.float32)
    variables = norm.init(jax.random.key(0), x, per_example)

    per_token = jnp.ones((2, 4, 8), dtype=jnp.float32)
    y_example, gate_example = norm.apply(variables, x, per_example)
    y_token, gate_token = norm.apply(variables, x, per_token)
    assert y_example.shape == y_token.shape == x.shape
    assert gate_example.shape == (2, 1, 8)
    assert gate_token.shape == (2, 4, 8)
    assert variables["params"]["Dense_0"]["kernel"].shape == (8, 24)


def test_pi05_jitted_sampling_uses_tokenwise_adarms_and_restores_prefix_exactly():
    model = _TinyPi05Sampler(nnx.Rngs(0))
    obs = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=2,
        max_token_len=4,
        simulated_delay=2,
    ).fake_obs(1)
    prefix_actions = jnp.arange(8, dtype=jnp.float32).reshape(1, 4, 2)
    prefix = rtc.ActionPrefix(
        actions=prefix_actions,
        delay=jnp.array([2], dtype=jnp.int32),
        prefix_length=jnp.array([3], dtype=jnp.int32),
    )
    sample = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps",))
    actions = sample(
        jax.random.key(1),
        obs,
        num_steps=2,
        noise=jnp.zeros_like(prefix_actions),
        action_prefix=prefix,
    )
    assert actions.shape == prefix_actions.shape
    np.testing.assert_array_equal(actions[:, :2], prefix_actions[:, :2])
    assert np.isfinite(np.asarray(actions)).all()


def test_pi05_jitted_ordinary_training_loss_exercises_rtc_and_is_finite():
    model = _TinyPi05Sampler(nnx.Rngs(0))
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=2,
        max_token_len=4,
        simulated_delay=2,
    )
    obs, actions = config.fake_obs(1), config.fake_act(1)
    rng = jax.random.key(1)
    loss = nnx_utils.module_jit(model.compute_loss)(rng, obs, actions)
    assert loss.shape == (1, 4)
    assert np.isfinite(np.asarray(loss)).all()

    delay = int(jax.random.randint(jax.random.fold_in(rng, 0x525443), (1,), 0, 3)[0])
    np.testing.assert_array_equal(loss[0, :delay], 0)


def test_pi05_jitted_memory_sampling_restores_prefix_exactly():
    model = _TinyPi05MemorySampler(nnx.Rngs(0), memory_write_source="post_attention")
    obs = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=2,
        max_token_len=4,
        simulated_delay=2,
    ).fake_obs(1)
    prefix_actions = jnp.arange(8, dtype=jnp.float32).reshape(1, 4, 2)
    prefix = rtc.ActionPrefix(
        actions=prefix_actions,
        delay=jnp.array([2], dtype=jnp.int32),
        prefix_length=jnp.array([3], dtype=jnp.int32),
    )
    sample = nnx_utils.module_jit(
        model.sample_with_memory,
        static_argnames=("stop_token", "max_decode_steps", "num_steps"),
    )
    state0 = model.memory.init_state(1)
    _, state_without_prefix, aux_without_prefix = sample(
        jax.random.key(1),
        obs,
        state0,
        stop_token=1,
        max_decode_steps=1,
        num_steps=2,
        noise=jnp.zeros_like(prefix_actions),
        action_prefix=None,
    )
    actions, new_state, aux = sample(
        jax.random.key(1),
        obs,
        state0,
        stop_token=1,
        max_decode_steps=1,
        num_steps=2,
        noise=jnp.zeros_like(prefix_actions),
        action_prefix=prefix,
    )
    assert actions.shape == prefix_actions.shape
    np.testing.assert_array_equal(actions[:, :2], prefix_actions[:, :2])
    assert np.isfinite(np.asarray(actions)).all()
    assert new_state.fast_weights
    assert aux["tokens"].shape == (1, 2)
    _assert_trees_equal(new_state, state_without_prefix)
    for key in ("surprise", "grad_norm", "theta", "eta", "alpha", "tokens", "token_mask"):
        np.testing.assert_array_equal(aux[key], aux_without_prefix[key])


def test_pi05_jitted_sequence_loss_exercises_rtc_token_times_and_is_finite():
    model = _TinyPi05SequenceTrainer(nnx.Rngs(0), memory_write_source="post_attention")
    memory_config = memory.MemoryConfig(d_input=64, d_key=8, hidden_dims=(8,), d_value=64)
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=4,
        action_dim=2,
        max_token_len=4,
        simulated_delay=2,
        predict_subtask=True,
        predict_with_memory=True,
        memory=memory_config,
        memory_layer=0,
        causal_token_len=2,
        memory_seq_steps=2,
    )
    obs, actions = config.fake_obs(1), config.fake_act(1)
    compute = nnx_utils.module_jit(model._compute_sequence_loss)  # noqa: SLF001
    losses = compute(jax.random.key(1), obs, actions)
    assert losses["flow"].shape == losses["ce"].shape == (1,)
    assert np.isfinite(np.asarray(losses["flow"])).all()
    assert np.isfinite(np.asarray(losses["ce"])).all()
