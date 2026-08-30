"""v3.4 memory-core tests (V34_PLAN_final.md 5.7, 5.10, 8.4 and 9.2 e/f/h/i).

Covers:
  * 5.10 refactor equivalence: `read`/`write` bit-identical to the pre-refactor math
    (an inline reference copy of the v3.3 implementation) with `mlp_l2norm=False`;
  * key-space API consistency: read == read_key(project_q), write == write_kv(project_kv);
  * 5.7 unit-L2 MLP: hidden activations pinned to unit norm, zero-init output preserved,
    He layer-0 init, and O(1) raw inner gradients on unit-norm synthetic pairs;
  * 5.7 autodiff token diagnostics: matches direct per-token jax.grad under BOTH forward
    variants (the mandatory "no heatmap is trusted until this passes" test);
  * 8.4 dynamics-only: S_t = eta S_{t-1} and M_t = (1-alpha) M_{t-1} + S_t exactly;
  * drift trust region guardrail.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import memory


def _small_config(**overrides) -> memory.MemoryConfig:
    return memory.MemoryConfig(d_input=32, d_key=12, hidden_dims=(16, 16), d_value=24, **overrides)


def _reference_forward(fast_weights, k):
    """Bit-exact copy of the pre-refactor (v3.3) memory-MLP forward."""
    num_layers = len({name for name in fast_weights if name.startswith("w")})
    x = k
    for i in range(num_layers):
        x = x @ fast_weights[f"w{i}"] + fast_weights[f"b{i}"]
        if i < num_layers - 1:
            x = jax.nn.silu(x)
    return x


def _reference_read(mem: memory.TitansMemory, state, h):
    """Bit-exact copy of the pre-refactor read()."""
    q = memory._l2_norm(jax.nn.silu(mem.w_q(h.astype(jnp.float32))))  # noqa: SLF001
    return jax.vmap(_reference_forward)(state.fast_weights, q)


def _reference_write(mem: memory.TitansMemory, state, h):
    """Bit-exact copy of the pre-refactor write()."""
    x = h.astype(jnp.float32)
    k = memory._l2_norm(jax.nn.silu(mem.w_k(x)))  # noqa: SLF001
    v = memory._l2_norm(jax.nn.silu(mem.w_v(x)))  # noqa: SLF001
    gates = jax.nn.sigmoid(mem.gate(jnp.mean(x, axis=1)))
    theta, eta, alpha = gates[:, 0], gates[:, 1], gates[:, 2]

    def loss(fast_weights, k, v):
        return jnp.mean(jnp.sum(jnp.square(_reference_forward(fast_weights, k) - v), axis=-1))

    surprise, grads = jax.vmap(jax.value_and_grad(loss))(state.fast_weights, k, v)
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g), axis=tuple(range(1, g.ndim))) for g in jax.tree.leaves(grads)))
    clip = jax.lax.stop_gradient(jnp.minimum(1.0, mem.config.max_grad_norm / (grad_norm + 1e-12)))
    grads = jax.tree.map(lambda g: memory._per_sample(clip, g) * g, grads)  # noqa: SLF001
    momentum = jax.tree.map(
        lambda s, g: memory._per_sample(eta, s) * s - memory._per_sample(theta, g) * g,  # noqa: SLF001
        state.momentum,
        grads,
    )
    fast_weights = jax.tree.map(
        lambda w, s: memory._per_sample(1 - alpha, w) * w + s,  # noqa: SLF001
        state.fast_weights,
        momentum,
    )
    aux = {"surprise": surprise, "grad_norm": grad_norm, "theta": theta, "eta": eta, "alpha": alpha}
    return memory.MemoryState(fast_weights, momentum), aux


def _tokens(key, batch=2, n=5, d=32, scale=3.0):
    return scale * jax.random.normal(jax.random.key(key), (batch, n, d), dtype=jnp.float32)


def test_refactor_equivalence_read_write_bit_identical():
    """Plan 9.2(h): the 5.10 core refactor changes NO numerics with mlp_l2norm off."""
    mem = memory.TitansMemory(_small_config(), rngs=nnx.Rngs(0))
    state = mem.init_state(2)
    h = _tokens(1)

    # A written (asymmetric) state is the stronger comparison point than the fresh zero-output one.
    state, _ = mem.write(state, _tokens(2))

    np.testing.assert_array_equal(np.asarray(mem.read(state, h)), np.asarray(_reference_read(mem, state, h)))

    new_state, aux = mem.write(state, h)
    ref_state, ref_aux = _reference_write(mem, state, h)
    jax.tree.map(np.testing.assert_array_equal, tuple(new_state), tuple(ref_state))
    for key in ref_aux:
        np.testing.assert_array_equal(np.asarray(aux[key]), np.asarray(ref_aux[key]))
    # the refactor adds the clip multiplier to aux; nothing else may change
    assert set(aux) == {*ref_aux, "clip_factor"}


def test_key_space_api_matches_hidden_space_wrappers():
    """read == read_key(project_q(h)); write == write_kv(project_kv(h), gates(h))."""
    for l2norm in (False, True):
        mem = memory.TitansMemory(_small_config(mlp_l2norm=l2norm), rngs=nnx.Rngs(0))
        state, _ = mem.write(mem.init_state(2), _tokens(3))
        h = _tokens(4)

        np.testing.assert_array_equal(
            np.asarray(mem.read(state, h)), np.asarray(mem.read_key(state, mem.project_q(h)))
        )
        k, v = mem.project_kv(h)
        theta, eta, alpha = mem.gates(h)
        via_core, aux_core = mem.write_kv(state, k, v, theta, eta, alpha)
        via_wrapper, aux_wrapper = mem.write(state, h)
        jax.tree.map(np.testing.assert_array_equal, tuple(via_core), tuple(via_wrapper))
        for key in aux_core:
            np.testing.assert_array_equal(np.asarray(aux_core[key]), np.asarray(aux_wrapper[key]))
        # projections are unit-L2 per token
        np.testing.assert_allclose(np.linalg.norm(np.asarray(k), axis=-1), 1.0, atol=1e-5)
        np.testing.assert_allclose(np.linalg.norm(np.asarray(v), axis=-1), 1.0, atol=1e-5)


@pytest.mark.parametrize("eta_scale", [-0.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_eta_scale_rejects_nonfinite_and_out_of_range_values(eta_scale):
    with pytest.raises(ValueError, match=r"eta_scale must be finite and in \[0, 1\]"):
        _small_config(eta_scale=eta_scale)


def test_eta_scale_endpoints_and_fractional_scaling_are_exact():
    h = _tokens(40)
    for eta_scale in (0.0, 0.25, 1.0):
        mem = memory.TitansMemory(_small_config(eta_scale=eta_scale), rngs=nnx.Rngs(41))
        raw = jax.nn.sigmoid(mem.gate(jnp.mean(h.astype(jnp.float32), axis=1)))
        theta, eta, alpha = mem.gates(h)
        np.testing.assert_array_equal(np.asarray(theta), np.asarray(raw[:, 0]))
        np.testing.assert_array_equal(np.asarray(alpha), np.asarray(raw[:, 2]))
        if eta_scale == 0.0:
            np.testing.assert_array_equal(np.asarray(eta), np.zeros_like(np.asarray(raw[:, 1])))
        elif eta_scale == 1.0:
            # Default compatibility path returns the sigmoid result without another multiply.
            np.testing.assert_array_equal(np.asarray(eta), np.asarray(raw[:, 1]))
        else:
            np.testing.assert_array_equal(np.asarray(eta), np.asarray(raw[:, 1] * eta_scale))


def test_eta_zero_write_and_decay_match_explicit_eta_zero_core_bit_exactly():
    """Every hidden-token path applies eta_scale, while explicit write_kv stays unscaled."""
    mem = memory.TitansMemory(_small_config(eta_scale=0.0), rngs=nnx.Rngs(42))
    seed_h = _tokens(43)
    seed_k, seed_v = mem.project_kv(seed_h)
    raw_seed_gates = jax.nn.sigmoid(mem.gate(jnp.mean(seed_h.astype(jnp.float32), axis=1)))
    # Seed nonzero momentum through the intentionally explicit/unscaled core boundary.
    state, seed_aux = mem.write_kv(
        mem.init_state(2),
        seed_k,
        seed_v,
        raw_seed_gates[:, 0],
        raw_seed_gates[:, 1],
        raw_seed_gates[:, 2],
    )
    np.testing.assert_array_equal(np.asarray(seed_aux["eta"]), np.asarray(raw_seed_gates[:, 1]))
    assert any(float(jnp.linalg.norm(leaf)) > 0.0 for leaf in jax.tree.leaves(state.momentum))

    h = _tokens(44)
    k, v = mem.project_kv(h)
    theta, eta, alpha = mem.gates(h)
    np.testing.assert_array_equal(np.asarray(eta), 0.0)

    wrapper_state, wrapper_aux = mem.write(state, h)
    core_state, core_aux = mem.write_kv(state, k, v, theta, jnp.zeros_like(eta), alpha)
    jax.tree.map(np.testing.assert_array_equal, wrapper_state, core_state)
    jax.tree.map(np.testing.assert_array_equal, wrapper_aux, core_aux)
    np.testing.assert_array_equal(np.asarray(wrapper_aux["eta"]), 0.0)

    decay_state, decay_aux = mem.decay_step(state, h)
    core_decay_state, core_decay_aux = mem.write_kv(
        state, k, v, theta, jnp.zeros_like(eta), alpha, zero_gradient=True
    )
    jax.tree.map(np.testing.assert_array_equal, decay_state, core_decay_state)
    jax.tree.map(np.testing.assert_array_equal, decay_aux, core_decay_aux)
    np.testing.assert_array_equal(np.asarray(decay_aux["eta"]), 0.0)


def test_eta_scale_changes_no_parameter_key_shape_dtype_or_initial_value():
    default = memory.TitansMemory(_small_config(eta_scale=1.0), rngs=nnx.Rngs(45))
    eta_zero = memory.TitansMemory(_small_config(eta_scale=0.0), rngs=nnx.Rngs(45))
    default_params = nnx.state(default).flat_state()
    eta_zero_params = nnx.state(eta_zero).flat_state()

    assert default_params.keys() == eta_zero_params.keys()
    for path in default_params:
        old = default_params[path].value
        new = eta_zero_params[path].value
        assert old.shape == new.shape, path
        assert old.dtype == new.dtype, path
        np.testing.assert_array_equal(np.asarray(old), np.asarray(new), err_msg=str(path))


def test_l2norm_forward_pins_hidden_activations_and_keeps_zero_fresh_read():
    config = _small_config(mlp_l2norm=True)
    mem = memory.TitansMemory(config, rngs=nnx.Rngs(0))
    state = mem.init_state(2)

    # fresh memory reads exactly zero (zero-init output layer untouched by normalization)
    np.testing.assert_array_equal(np.asarray(mem.read(state, _tokens(5))), 0.0)

    # instrument the forward: the input to the OUTPUT layer must be unit-L2 per token
    fw = {name: np.asarray(leaf[0]) for name, leaf in state.fast_weights.items()}
    k = np.asarray(mem.project_q(_tokens(6))[0])  # any unit key works
    x = k / np.linalg.norm(k, axis=-1, keepdims=True)
    num_layers = mem._num_layers  # noqa: SLF001
    for i in range(num_layers - 1):
        x = np.asarray(jax.nn.silu(x @ fw[f"w{i}"] + fw[f"b{i}"]))
        x = x / np.sqrt(np.sum(np.square(x), axis=-1, keepdims=True) + 1e-12)
        np.testing.assert_allclose(np.linalg.norm(x, axis=-1), 1.0, atol=1e-4)
    out = x @ fw[f"w{num_layers - 1}"] + fw[f"b{num_layers - 1}"]
    np.testing.assert_array_equal(out, 0.0)

    # He layer-0 init when normalized (std ~ sqrt(2 / fan_in), far from the un-normalized 1.0)
    w0 = np.asarray(mem.m0["w0"].value)
    assert abs(w0.std() - np.sqrt(2.0 / w0.shape[0])) < 0.35 * np.sqrt(2.0 / w0.shape[0])


def test_l2norm_raw_gradients_are_order_one_on_unit_pairs():
    """Plan 5.7 acceptance shape at the unit test level: with pinned activations, the raw inner
    gradient on unit-norm targets is O(1), not orders of magnitude above the clip."""
    config = memory.MemoryConfig(
        d_input=2048, d_key=512, hidden_dims=(1024, 1024, 1024), d_value=2048, mlp_l2norm=True
    )
    mem = memory.TitansMemory(config, rngs=nnx.Rngs(0))
    state = mem.init_state(1)
    k = memory._l2_norm(jax.random.normal(jax.random.key(7), (1, 16, 512)))  # noqa: SLF001
    v = memory._l2_norm(jax.random.normal(jax.random.key(8), (1, 16, 2048)))  # noqa: SLF001
    theta = jnp.full((1,), 0.10)
    eta = jnp.full((1,), 0.90)
    alpha = jnp.full((1,), 0.01)
    _, aux = mem.write_kv(state, k, v, theta, eta, alpha)
    grad_norm = float(aux["grad_norm"][0])
    assert 0.05 < grad_norm < 5.0, grad_norm
    # fresh state, zero output: surprise is exactly the unit target norm
    np.testing.assert_allclose(float(aux["surprise"][0]), 1.0, atol=1e-5)


def test_blank_initial_output_ignores_loaded_outer_values_and_stays_fp32():
    """A stale/legacy checkpoint output layer cannot contaminate a fresh episode state."""
    blank = memory.TitansMemory(
        _small_config(mlp_l2norm=True, blank_initial_output=True), rngs=nnx.Rngs(30)
    )
    legacy = memory.TitansMemory(_small_config(mlp_l2norm=True), rngs=nnx.Rngs(30))
    output_layer = blank._num_layers - 1  # noqa: SLF001
    w_name, b_name = f"w{output_layer}", f"b{output_layer}"
    loaded_w = jnp.full(blank.m0[w_name].value.shape, 7.0, dtype=jnp.float32)
    loaded_b = jnp.linspace(-3.0, 3.0, blank.m0[b_name].value.size, dtype=jnp.float32)
    for module in (blank, legacy):
        module.m0[w_name].value = loaded_w
        module.m0[b_name].value = loaded_b

    state = blank.init_state(2)
    np.testing.assert_array_equal(np.asarray(state.fast_weights[w_name]), 0.0)
    np.testing.assert_array_equal(np.asarray(state.fast_weights[b_name]), 0.0)
    np.testing.assert_array_equal(np.asarray(state.momentum[w_name]), 0.0)
    np.testing.assert_array_equal(np.asarray(state.momentum[b_name]), 0.0)
    assert state.fast_weights[w_name].dtype == jnp.float32
    assert state.fast_weights[b_name].dtype == jnp.float32
    # The stored checkpoint leaves are preserved byte-for-byte; only their effective episode
    # initializer is blank. A lower layer still initializes from m0 normally.
    np.testing.assert_array_equal(np.asarray(blank.m0[w_name].value), np.asarray(loaded_w))
    np.testing.assert_array_equal(np.asarray(blank.m0[b_name].value), np.asarray(loaded_b))
    np.testing.assert_array_equal(
        np.asarray(state.fast_weights["w0"][0]), np.asarray(blank.m0["w0"].value)
    )
    q = memory._l2_norm(jax.random.normal(jax.random.key(31), (2, 5, blank.config.d_key)))  # noqa: SLF001
    np.testing.assert_array_equal(np.asarray(blank.read_key(state, q)), 0.0)

    # Default-off is the exact compatibility control: the same loaded leaves are broadcast.
    legacy_state = legacy.init_state(2)
    np.testing.assert_array_equal(np.asarray(legacy_state.fast_weights[w_name][0]), np.asarray(loaded_w))
    np.testing.assert_array_equal(np.asarray(legacy_state.fast_weights[b_name][0]), np.asarray(loaded_b))


def test_blank_initial_output_outer_grads_are_zero_but_fast_output_writes():
    """The outer output seed is dead, while per-episode output leaves remain plastic."""
    config = _small_config(mlp_l2norm=True, blank_initial_output=True)
    mem = memory.TitansMemory(config, rngs=nnx.Rngs(32))
    output_layer = mem._num_layers - 1  # noqa: SLF001
    w_name, b_name = f"w{output_layer}", f"b{output_layer}"
    k = memory._l2_norm(jax.random.normal(jax.random.key(33), (1, 3, config.d_key)))  # noqa: SLF001
    v = memory._l2_norm(jax.random.normal(jax.random.key(34), (1, 3, config.d_value)))  # noqa: SLF001
    theta = jnp.full((1,), 0.10)
    eta = jnp.full((1,), 0.90)
    alpha = jnp.full((1,), 0.01)

    graphdef, params = nnx.split(mem)

    def post_write_error(module_params):
        module = nnx.merge(graphdef, module_params)
        state, _ = module.write_kv(module.init_state(1), k, v, theta, eta, alpha)
        return jnp.sum(jnp.square(module.read_key(state, k) - v))

    grads = jax.grad(post_write_error)(params)
    np.testing.assert_array_equal(np.asarray(grads["m0"][w_name].value), 0.0)
    np.testing.assert_array_equal(np.asarray(grads["m0"][b_name].value), 0.0)
    assert float(jnp.linalg.norm(grads["m0"]["w0"].value)) > 0.0

    state0 = mem.init_state(1)
    initial_error = float(jnp.sum(jnp.square(mem.read_key(state0, k) - v)))

    @jax.jit
    def one_write(state, keys, values):
        return mem.write_kv(state, keys, values, theta, eta, alpha)

    state1, _ = one_write(state0, k, v)
    jax.block_until_ready(state1)
    final_error = float(jnp.sum(jnp.square(mem.read_key(state1, k) - v)))
    assert float(jnp.linalg.norm(state1.fast_weights[w_name])) > 0.0
    assert float(jnp.linalg.norm(state1.fast_weights[b_name])) > 0.0
    assert float(jnp.linalg.norm(state1.momentum[w_name])) > 0.0
    assert final_error < initial_error, (initial_error, final_error)
    # A new segment is blank again; inner plasticity never mutates the outer module.
    np.testing.assert_array_equal(np.asarray(mem.init_state(1).fast_weights[w_name]), 0.0)
    np.testing.assert_array_equal(np.asarray(mem.m0[w_name].value), 0.0)
    np.testing.assert_array_equal(np.asarray(mem.m0[b_name].value), 0.0)


def test_blank_initial_output_trust_region_uses_effective_zero_anchor():
    radius = 0.25
    mem = memory.TitansMemory(
        _small_config(mlp_l2norm=True, blank_initial_output=True, drift_radius=radius), rngs=nnx.Rngs(35)
    )
    output_layer = mem._num_layers - 1  # noqa: SLF001
    w_name, b_name = f"w{output_layer}", f"b{output_layer}"
    # Simulate loading a legacy checkpoint whose dormant outer output is very large.
    mem.m0[w_name].value = jnp.full_like(mem.m0[w_name].value, 10.0)
    mem.m0[b_name].value = jnp.full_like(mem.m0[b_name].value, -10.0)
    base = mem.init_state(1)
    projected_base = mem._drift_trust_region(base.fast_weights)  # noqa: SLF001
    jax.tree.map(np.testing.assert_array_equal, projected_base, base.fast_weights)

    displaced = jax.tree.map(lambda leaf: leaf + jnp.ones_like(leaf), base.fast_weights)
    projected = mem._drift_trust_region(displaced)  # noqa: SLF001
    drift = memory._tree_norm(jax.tree.map(jnp.subtract, projected, base.fast_weights))  # noqa: SLF001
    assert float(drift[0]) <= radius + 1e-5
    # If the trust region accidentally anchored to raw m0, these leaves would be pulled toward
    # +/-10 instead of remaining within radius of the effective blank output.
    assert float(jnp.linalg.norm(projected[w_name])) < radius + 1e-5
    assert float(jnp.linalg.norm(projected[b_name])) < radius + 1e-5


def test_blank_initial_output_preserves_parameter_tree_shape_and_dtype():
    legacy = memory.TitansMemory(_small_config(mlp_l2norm=True), rngs=nnx.Rngs(36))
    blank = memory.TitansMemory(
        _small_config(mlp_l2norm=True, blank_initial_output=True), rngs=nnx.Rngs(36)
    )
    legacy_state = nnx.state(legacy)
    blank_state = nnx.state(blank)
    assert jax.tree.structure(legacy_state) == jax.tree.structure(blank_state)
    for old, new in zip(jax.tree.leaves(legacy_state), jax.tree.leaves(blank_state), strict=True):
        assert old.shape == new.shape
        assert old.dtype == new.dtype


def test_token_write_diagnostics_matches_direct_jax_grad():
    """Plan 5.7 mandatory test: NO writer-contribution heatmap is trusted until the per-token
    diagnostic matches direct jax.grad on a small random memory MLP."""
    for l2norm in (False, True):
        mem = memory.TitansMemory(_small_config(mlp_l2norm=l2norm), rngs=nnx.Rngs(1))
        state, _ = mem.write(mem.init_state(2), _tokens(9))
        h = _tokens(10)
        diag = mem.token_write_diagnostics(state, h)

        _, k, v = mem._keys_values(h)  # noqa: SLF001
        for b in range(h.shape[0]):
            fw = {name: leaf[b] for name, leaf in state.fast_weights.items()}
            for i in range(h.shape[1]):

                def token_error(fast_weights, mlp=mem, k_i=k[b, i], v_i=v[b, i]):
                    pred = mlp._forward(fast_weights, k_i[None])[0]  # noqa: SLF001
                    return jnp.sum(jnp.square(pred - v_i))

                error, grad = jax.value_and_grad(token_error)(fw)
                grad_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree.leaves(grad))))
                np.testing.assert_allclose(float(diag["token_error"][b, i]), float(error), rtol=1e-5)
                np.testing.assert_allclose(float(diag["token_grad_norm"][b, i]), grad_norm, rtol=1e-4)
                np.testing.assert_allclose(
                    float(diag["token_mean_loss_grad_norm"][b, i]), grad_norm / h.shape[1], rtol=1e-4
                )


def test_dynamics_only_step_is_exact():
    """Plan 9.2(i): with the gradient term zeroed, S_t = eta S_{t-1} and
    M_t = (1-alpha) M_{t-1} + S_t exactly."""
    mem = memory.TitansMemory(_small_config(), rngs=nnx.Rngs(2))
    state, _ = mem.write(mem.init_state(2), _tokens(11))
    state, _ = mem.write(state, _tokens(12))  # nonzero momentum
    h = _tokens(13)

    new_state, aux = mem.decay_step(state, h)
    theta, eta, alpha = mem.gates(h)
    for name in state.fast_weights:
        expected_s = memory._per_sample(eta, state.momentum[name]) * state.momentum[name]  # noqa: SLF001
        expected_m = memory._per_sample(1 - alpha, state.fast_weights[name]) * state.fast_weights[name] + expected_s  # noqa: SLF001
        np.testing.assert_array_equal(np.asarray(new_state.momentum[name]), np.asarray(expected_s))
        np.testing.assert_array_equal(np.asarray(new_state.fast_weights[name]), np.asarray(expected_m))
    np.testing.assert_array_equal(np.asarray(aux["grad_norm"]), 0.0)
    # surprise still reports the pre-update prediction error
    np.testing.assert_allclose(np.asarray(aux["surprise"]), np.asarray(mem.surprise(state, h)), rtol=1e-6)
    # theta is unused by the dynamics but still reported, matching write()'s gate computation
    np.testing.assert_array_equal(np.asarray(aux["theta"]), np.asarray(theta))


def test_drift_trust_region_bounds_fast_weight_drift():
    radius = 0.5
    config = _small_config(drift_radius=radius)
    mem = memory.TitansMemory(config, rngs=nnx.Rngs(3))
    state = mem.init_state(1)
    m0 = {name: np.asarray(leaf[0]) for name, leaf in state.fast_weights.items()}
    for i in range(50):
        state, _ = mem.write(state, _tokens(100 + i, batch=1))
        drift = np.sqrt(
            sum(np.sum(np.square(np.asarray(state.fast_weights[name][0]) - m0[name])) for name in m0)
        )
        assert drift <= radius + 1e-4, (i, drift)
    # the guardrail must not silently freeze the memory: some drift accumulates
    assert drift > 0.05 * radius


def test_write_and_read_key_roundtrip_stores_association():
    """Stage-0 style smoke: repeated writes of one unit (k, v) pair converge own-key recall."""
    config = _small_config(mlp_l2norm=True)
    mem = memory.TitansMemory(config, rngs=nnx.Rngs(4))
    state = mem.init_state(1)
    k = memory._l2_norm(jax.random.normal(jax.random.key(20), (1, 1, config.d_key)))  # noqa: SLF001
    v = memory._l2_norm(jax.random.normal(jax.random.key(21), (1, 1, config.d_value)))  # noqa: SLF001
    theta = jnp.full((1,), 0.10)
    eta = jnp.full((1,), 0.90)
    alpha = jnp.full((1,), 0.01)
    initial_error = float(jnp.sum(jnp.square(mem.read_key(state, k) - v)))
    for _ in range(64):
        state, _ = mem.write_kv(state, k, v, theta, eta, alpha)
    final_error = float(jnp.sum(jnp.square(mem.read_key(state, k) - v)))
    assert final_error < 0.2 * initial_error, (initial_error, final_error)


# ---------------------------------------------------------------------------------------------
# state_cotangent_clip (v34_run1 postmortem): backward-only guardrail on the recurrent chain.
# ---------------------------------------------------------------------------------------------


def _chain_read_scalar(mem: memory.TitansMemory, h_steps, weights):
    """Chained writes followed by a per-sample-weighted read scalar (grad path through every
    write's incoming state, i.e. through every cotangent-clip node)."""
    state = mem.init_state(h_steps[0].shape[0])
    for h in h_steps:
        state, _ = mem.write(state, h)
    q = memory._l2_norm(jax.random.normal(jax.random.key(99), h_steps[0].shape[:2] + (mem.config.d_key,)))  # noqa: SLF001
    read = mem.read_key(state, q)
    return jnp.sum(weights[:, None, None] * read)


def test_state_cotangent_clip_forward_is_identity():
    """The clip must not change any forward value, even at a limit that would bind backward."""
    plain = memory.TitansMemory(_small_config(mlp_l2norm=True), rngs=nnx.Rngs(11))
    clipped = memory.TitansMemory(_small_config(mlp_l2norm=True, state_cotangent_clip=1e-6), rngs=nnx.Rngs(11))
    state_p = plain.init_state(2)
    state_c = clipped.init_state(2)
    for step in range(4):
        h = _tokens(500 + step)
        state_p, aux_p = plain.write(state_p, h)
        state_c, aux_c = clipped.write(state_c, h)
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), state_p, state_c)
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), aux_p, aux_c)


def test_state_cotangent_clip_non_binding_backward_is_bit_exact():
    """With a limit far above every chain cotangent, gradients are bit-identical to no clip
    (the backward multiplies by exactly 1.0)."""
    plain = memory.TitansMemory(_small_config(mlp_l2norm=True), rngs=nnx.Rngs(12))
    clipped = memory.TitansMemory(_small_config(mlp_l2norm=True, state_cotangent_clip=1e9), rngs=nnx.Rngs(12))
    h_steps = [_tokens(600 + step) for step in range(6)]
    weights = jnp.asarray([1.0, 2.0])

    def loss(mem):
        return jax.grad(lambda hs: _chain_read_scalar(mem, hs, weights))(h_steps)

    grads_plain = loss(plain)
    grads_clipped = loss(clipped)
    jax.tree.map(
        lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), grads_plain, grads_clipped
    )


def test_kv_cotangent_clip_forward_identity_and_non_binding_backward():
    """The k/v-input cotangent clip: forward identical, backward bit-exact when not binding."""
    plain = memory.TitansMemory(_small_config(mlp_l2norm=True), rngs=nnx.Rngs(14))
    clipped = memory.TitansMemory(
        _small_config(mlp_l2norm=True, state_cotangent_clip=1e9, kv_cotangent_clip=1e9), rngs=nnx.Rngs(14)
    )
    h_steps = [_tokens(800 + step) for step in range(4)]
    weights = jnp.asarray([1.0, 2.0])
    state_p = plain.init_state(2)
    state_c = clipped.init_state(2)
    for h in h_steps:
        state_p, _ = plain.write(state_p, h)
        state_c, _ = clipped.write(state_c, h)
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), state_p, state_c)
    grads_p = jax.grad(lambda hs: _chain_read_scalar(plain, hs, weights))(h_steps)
    grads_c = jax.grad(lambda hs: _chain_read_scalar(clipped, hs, weights))(h_steps)
    jax.tree.map(lambda a, b: np.testing.assert_array_equal(np.asarray(a), np.asarray(b)), grads_p, grads_c)


def test_kv_cotangent_clip_bounds_gradient_into_write_tokens():
    """With a binding kv clip, an arbitrarily amplified downstream loss cannot push more than
    ~limit (times the fixed projection backward) into a step's write tokens."""
    limit = 0.01
    clipped = memory.TitansMemory(_small_config(mlp_l2norm=True, kv_cotangent_clip=limit), rngs=nnx.Rngs(15))
    plain = memory.TitansMemory(_small_config(mlp_l2norm=True), rngs=nnx.Rngs(15))
    h = _tokens(900)
    q = memory._l2_norm(jax.random.normal(jax.random.key(97), (2, 5, 12)))  # noqa: SLF001

    def token_grad(mem, amplification):
        def scalar(h_in):
            state, _ = mem.write(mem.init_state(2), h_in)
            return amplification * jnp.sum(mem.read_key(state, q))

        return jax.grad(scalar)(h)

    # plain: gradient into h scales linearly with the loss amplification (x1000)
    plain_1 = float(jnp.linalg.norm(token_grad(plain, 1.0)))
    plain_1k = float(jnp.linalg.norm(token_grad(plain, 1000.0)))
    np.testing.assert_allclose(plain_1k, 1000.0 * plain_1, rtol=1e-4)
    # clipped: the x1000 amplification is absorbed by the kv clip (gradient into h saturates)
    clipped_1k = float(jnp.linalg.norm(token_grad(clipped, 1000.0)))
    assert clipped_1k < 0.05 * plain_1k, (clipped_1k, plain_1k)


def test_state_cotangent_clip_binds_per_sample_with_direction_preserved():
    """One write: the state cotangent of a sample whose norm exceeds the limit is rescaled to
    exactly limit * direction; a below-limit sample in the same batch is untouched."""
    limit = 0.05
    plain = memory.TitansMemory(_small_config(mlp_l2norm=True), rngs=nnx.Rngs(13))
    clipped = memory.TitansMemory(_small_config(mlp_l2norm=True, state_cotangent_clip=limit), rngs=nnx.Rngs(13))
    h = _tokens(700)
    # sample 0 tiny loss weight (cotangent below the limit), sample 1 amplified far above it
    weights = jnp.asarray([1e-4, 1e4])

    def state_grads(mem):
        base = mem.init_state(2)

        def scalar(fast, mom):
            state, _ = mem.write(memory.MemoryState(fast, mom), h)
            q = memory._l2_norm(jax.random.normal(jax.random.key(98), (2, 5, mem.config.d_key)))  # noqa: SLF001
            return jnp.sum(weights[:, None, None] * mem.read_key(state, q))

        return jax.grad(scalar, argnums=(0, 1))(base.fast_weights, base.momentum)

    g_plain = state_grads(plain)
    g_clipped = state_grads(clipped)
    norms = np.asarray(memory._tree_norm(g_plain))  # noqa: SLF001
    assert norms[0] < limit < norms[1], norms
    expected_scale = np.minimum(1.0, limit / (norms + 1e-12))
    jax.tree.map(
        lambda a, b: np.testing.assert_allclose(
            np.asarray(b), np.asarray(a) * expected_scale.reshape((2,) + (1,) * (a.ndim - 1)), rtol=1e-6
        ),
        g_plain,
        g_clipped,
    )
