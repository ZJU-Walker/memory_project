import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import memory
from openpi.models import pi0
from openpi.models import pi0_config


class _TinyImageEncoder(nnx.Module):
    def __init__(self, width: int, rngs: nnx.Rngs):
        self.proj = nnx.Linear(3, width, rngs=rngs)

    def __call__(self, image, *, train: bool = False):
        del train
        pooled = jnp.mean(image, axis=(1, 2))
        # Four distinct synthetic patch slots keep the test graph tiny while ensuring learned
        # query/key attention (not just the value projection) participates in end-to-end grads.
        offsets = jnp.linspace(-0.3, 0.3, 4, dtype=pooled.dtype)[:, None] * jnp.asarray(
            [1.0, -0.5, 0.25], dtype=pooled.dtype
        )
        return self.proj(pooled[:, None] + offsets[None]), None


class _TinyV32(nnx.Module):
    embed_prefix = pi0.Pi0.embed_prefix
    embed_suffix = pi0.Pi0.embed_suffix
    sample_with_memory = pi0.Pi0.sample_with_memory
    _sample_with_memory_v32 = pi0.Pi0._sample_with_memory_v32  # noqa: SLF001
    _compute_sequence_loss = pi0.Pi0._compute_sequence_loss  # noqa: SLF001
    _compute_sequence_loss_v32 = pi0.Pi0._compute_sequence_loss_v32  # noqa: SLF001
    _check_action_prefix_shapes = pi0.Pi0._check_action_prefix_shapes  # noqa: SLF001
    _v32_empty_cache = pi0.Pi0._v32_empty_cache  # noqa: SLF001
    _v32_layer_mask = pi0.Pi0._v32_layer_mask  # noqa: SLF001
    _pad_attention_columns = staticmethod(pi0.Pi0._pad_attention_columns)  # noqa: SLF001
    _v32_prepare_memory_prefix = pi0.Pi0._v32_prepare_memory_prefix  # noqa: SLF001
    _v32_causal_mask = pi0.Pi0._v32_causal_mask  # noqa: SLF001
    _v32_step_mask = pi0.Pi0._v32_step_mask  # noqa: SLF001
    _v32_suffix_mask = pi0.Pi0._v32_suffix_mask  # noqa: SLF001

    def __init__(self, rngs: nnx.Rngs):
        config = gemma.get_config("dummy")
        config.depth = 2
        llm = nnx_bridge.ToNNX(gemma.Module(configs=[config, config], embed_dtype="float32", adarms=True))
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True])
        self.PaliGemma = nnx.Dict(llm=llm, img=_TinyImageEncoder(config.width, rngs))
        self.action_horizon = 4
        self.action_dim = 2
        self.max_token_len = 4
        self.pi05 = True
        self.simulated_delay = 2
        self.predict_subtask = True
        self.predict_with_memory = True
        self.memory_architecture = "v32_layer8_dual_query"
        self.memory_layer = 0
        self.memory_write_source = "query_compressed"
        self.memory_query_tokens = 16
        self.causal_token_len = 2
        memory_config = memory.MemoryConfig(d_input=64, d_key=8, hidden_dims=(8,), d_value=64)
        self.memory = memory.TitansMemory(memory_config, rngs=rngs)
        self.memory_gate = nnx.Param(jnp.ones((64,), dtype=jnp.float32))
        self.read_query_compressor = pi0.MemoryQueryCompressor(num_queries=16, width=64, num_heads=8, rngs=rngs)
        self.write_query_compressor = pi0.MemoryQueryCompressor(num_queries=16, width=64, num_heads=8, rngs=rngs)
        self.memory_probe_weight = 0.0
        self.memory_probe_diagnostic = False
        self.probe_head = nnx.Linear(64, 2, rngs=rngs)
        self.action_in_proj = nnx.Linear(2, config.width, rngs=rngs)
        self.time_mlp_in = nnx.Linear(config.width, config.width, rngs=rngs)
        self.time_mlp_out = nnx.Linear(config.width, config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(config.width, 2, rngs=rngs)


@pytest.fixture(scope="module")
def tiny_model():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        # These structural tests use token ids <= 8; a tiny embedding avoids retaining several
        # copies of the production 257k-way table while compiling the reverse-mode test.
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        yield _TinyV32(nnx.Rngs(0))
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def _single_observation():
    return (
        pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_horizon=4,
            action_dim=2,
            max_token_len=4,
            predict_subtask=True,
        )
        .fake_obs(1)
        .replace(tokenized_prompt_mask=jnp.ones((1, 4), dtype=bool))
    )


def _prepare(model, observation, state, *, zero_read=False):
    prefix, mask, ar = model.embed_prefix(observation)
    top_tokens = (mask.shape[1] - model.max_token_len) // len(observation.images)
    return model._v32_prepare_memory_prefix(  # noqa: SLF001
        prefix, mask, ar, state, top_token_count=top_tokens, zero_read=zero_read
    )


def test_query_compressor_bfloat16_compute_keeps_fp32_master_and_output():
    compressor = pi0.MemoryQueryCompressor(
        num_queries=4,
        width=64,
        num_heads=8,
        compute_dtype=jnp.bfloat16,
        rngs=nnx.Rngs(123),
    )
    source = jax.random.normal(jax.random.key(1), (2, 16, 64), dtype=jnp.float32)
    output = jax.jit(compressor)(source)

    assert compressor.key_proj.kernel.value.dtype == jnp.float32
    assert compressor.value_proj.kernel.value.dtype == jnp.float32
    assert output.dtype == jnp.float32
    assert output.shape == (2, 4, 64)
    assert np.all(np.isfinite(np.asarray(output)))

    graphdef, params = nnx.split(compressor)

    def loss(p):
        return jnp.mean(jnp.square(nnx.merge(graphdef, p)(source)))

    grads = jax.grad(loss)(params)
    leaves = jax.tree.leaves(grads)
    assert leaves
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
    assert any(np.any(np.asarray(leaf) != 0) for leaf in leaves)


def _sequence_observation():
    observation = _single_observation()
    steps = 2

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
        tokenized_causal=jnp.asarray([[[5, 6], [7, 8]]], dtype=jnp.int32),
        tokenized_causal_mask=jnp.ones((1, steps, 2), dtype=bool),
        causal_fast_mask=jnp.zeros((1, steps, 2), dtype=bool),
        seq_step_mask=jnp.ones((1, steps), dtype=bool),
        seq_block_boundary=jnp.zeros((1, steps), dtype=bool),
    )


def test_v32_config_rejects_wrong_layer_writer_or_query_count():
    memory_config = memory.MemoryConfig(d_input=2048, d_value=2048)
    common = {
        "pi05": True,
        "predict_subtask": True,
        "predict_with_memory": True,
        "memory": memory_config,
        "memory_architecture": "v32_layer8_dual_query",
        "memory_write_source": "query_compressed",
    }
    pi0_config.Pi0Config(memory_layer=8, **common)
    with pytest.raises(ValueError, match="memory_layer=8"):
        pi0_config.Pi0Config(memory_layer=7, **common)
    with pytest.raises(ValueError, match="query_compressed"):
        pi0_config.Pi0Config(memory_layer=8, **(common | {"memory_write_source": "post_attention"}))
    with pytest.raises(ValueError, match="exactly 16"):
        pi0_config.Pi0Config(memory_layer=8, memory_query_tokens=32, **common)


def test_dual_query_banks_are_distinct_and_emit_16_tokens(tiny_model):
    prepared = _prepare(tiny_model, _single_observation(), tiny_model.memory.init_state(1))

    # The tiny image encoder intentionally emits only four slots to keep integration tests cheap.
    # Independently exercise the production source geometry required by v3.2.
    source256 = jax.random.normal(jax.random.key(12), (1, 256, 64))
    read256 = tiny_model.read_query_compressor(source256)
    write256 = tiny_model.write_query_compressor(source256)

    assert prepared["h8_top"].shape == (1, 4, 64)
    assert prepared["read_queries"].shape == prepared["write_tokens"].shape == (1, 16, 64)
    assert read256.shape == write256.shape == (1, 16, 64)
    assert not np.array_equal(np.asarray(prepared["read_queries"]), np.asarray(prepared["write_tokens"]))
    state_paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(nnx.state(tiny_model))]
    assert any("read_query_compressor" in path for path in state_paths)
    assert any("write_query_compressor" in path for path in state_paths)


def test_write_tokens_depend_only_on_current_h8_while_read_uses_prewrite_state(tiny_model):
    model = tiny_model
    observation = _single_observation()
    state0 = model.memory.init_state(1)
    state1, _ = model.memory.write(state0, jnp.full((1, 16, 64), 0.25, dtype=jnp.float32))
    first = _prepare(model, observation, state0)
    second = _prepare(model, observation, state1)

    np.testing.assert_array_equal(first["h8_top"], second["h8_top"])
    np.testing.assert_array_equal(first["read_queries"], second["read_queries"])
    np.testing.assert_array_equal(first["write_tokens"], second["write_tokens"])
    assert not np.array_equal(np.asarray(first["retrieved"]), np.asarray(second["retrieved"]))


def test_retrieved_tokens_exist_only_after_layer8_and_change_the_late_representation(tiny_model):
    model = tiny_model
    observation = _single_observation()
    state, _ = model.memory.write(model.memory.init_state(1), jax.random.normal(jax.random.key(9), (1, 16, 64)))
    normal = _prepare(model, observation, state)
    zeroed = _prepare(model, observation, state, zero_read=True)
    prefix_len = normal["prefix_mask"].shape[1]
    cache_k = np.asarray(normal["cache"][0])

    np.testing.assert_array_equal(cache_k[: model.memory_layer + 1, :, prefix_len : prefix_len + 16], 0)
    assert np.any(cache_k[model.memory_layer + 1 :, :, prefix_len : prefix_len + 16] != 0)
    assert normal["final_prefix"].shape[1] == prefix_len + 16
    assert not np.array_equal(np.asarray(normal["final_prefix"]), np.asarray(zeroed["final_prefix"]))


def test_prediction_is_identical_before_commit_and_only_allow_write_changes_state(tiny_model):
    model = tiny_model
    observation = _single_observation()
    state = model.memory.init_state(1)
    kwargs = {
        "stop_token": 1,
        "max_decode_steps": 1,
        "num_steps": 1,
        "noise": jnp.zeros((1, 4, 2), dtype=jnp.float32),
        "forced_subtask_tokens": jnp.asarray([[5, 6]], dtype=jnp.int32),
        "forced_subtask_mask": jnp.ones((1, 2), dtype=bool),
    }
    actions_frozen, frozen, _ = model.sample_with_memory(
        jax.random.key(1), observation, state, allow_write=False, **kwargs
    )
    actions_written, written, _ = model.sample_with_memory(
        jax.random.key(1), observation, state, allow_write=True, **kwargs
    )

    np.testing.assert_array_equal(actions_frozen, actions_written)
    jax.tree.map(np.testing.assert_array_equal, frozen, state)
    assert any(
        not np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(written), jax.tree.leaves(state), strict=True)
    )


def test_gradients_reach_both_independent_query_banks(tiny_model):
    read_graph, read_params = nnx.split(tiny_model.read_query_compressor)
    write_graph, write_params = nnx.split(tiny_model.write_query_compressor)
    source = jax.random.normal(jax.random.key(13), (1, 7, 64))

    def query_objective(rp, wp):
        read = nnx.merge(read_graph, rp)(source)
        write = nnx.merge(write_graph, wp)(source)
        return jnp.mean(jnp.square(read)) + 0.7 * jnp.mean(jnp.square(write))

    read_grads, write_grads = jax.grad(query_objective, argnums=(0, 1))(read_params, write_params)
    for grads in (read_grads, write_grads):
        per_component = dict.fromkeys(("query_bank", "query_proj", "key_proj", "value_proj", "output_proj"), 0.0)
        for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
            key = jax.tree_util.keystr(path)
            for name in per_component:
                if name in key:
                    per_component[name] += float(jnp.sum(jnp.square(leaf)))
        assert all(value > 0 for value in per_component.values()), per_component


def test_end_to_end_sequence_ce_reaches_queries_and_slow_memory(tiny_model):
    """The real recurrent objective, not an isolated compressor loss, trains the v3.2 interface."""

    graphdef, params = nnx.split(tiny_model)
    observation = _sequence_observation()
    actions = jnp.zeros((1, 2, 4, 2), dtype=jnp.float32)

    def objective(p):
        losses = nnx.merge(graphdef, p)._compute_sequence_loss(  # noqa: SLF001
            jax.random.key(23), observation, actions, train=False
        )
        return jnp.sum(losses["ce"])

    loss, grads = jax.value_and_grad(objective)(params)
    assert bool(jnp.isfinite(loss))
    by_path = {jax.tree_util.keystr(path): leaf for path, leaf in jax.tree_util.tree_leaves_with_path(grads)}
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in by_path.values())

    def family_norm(fragment):
        return sum(float(jnp.sum(jnp.square(leaf))) for path, leaf in by_path.items() if fragment in path)

    # Read and write banks are independently reached through the actual read-before-write
    # recurrence. Titans' read projection, writer K/V projections, and learned initial state
    # must all receive outer-task gradients as required by the architecture contract.
    for fragment in (
        "['PaliGemma']['img']",
        "read_query_compressor",
        "write_query_compressor",
        "['memory']['w_q']",
        "['memory']['w_k']",
        "['memory']['w_v']",
        "['memory']['m0']",
    ):
        assert family_norm(fragment) > 0, fragment

    # Specifically rule out a value-projection-only shortcut: both learned banks and both
    # attention score paths are trained by the recurrent CE objective.
    for bank in ("read_query_compressor", "write_query_compressor"):
        assert family_norm(f"['{bank}']['query_bank']") > 0, bank
        assert family_norm(f"['{bank}']['query_proj']") > 0, bank
        assert family_norm(f"['{bank}']['key_proj']") > 0, bank
