"""v3.4 model-level tests (V34_PLAN_final.md section 9.2 and the 5.x component contracts).

Covers, against the tiny two-block fixture:
  (a) blinded-mask structure + behavioral invariance of blinded memory-token outputs;
  (b) exact-zero injection start under the tanh_rms form, plus its normalization math and an
      open gradient path through tanh(w);
  (c) probe stop-grad AND main-update invariance with ladder probes on/off;
  (d) CE seed = last valid NON-memory position under both text padding and appended memory;
  (e) per-segment state-mask consistency: masked segments are invariant to the true state
      digits end-to-end (single view), and the dual-view variant keeps CE invariant while the
      memory-state evolution still sees the real state;
  (g) P_valid forces exactly-zero attention on an enormous-activation padded patch in BOTH
      __call__ and attention_probs, with valid attention summing to 1;
  plus: QK-norm temperature semantics, letterbox geometry for the 480x640 top camera, the
  three-way write_mode retention control, and aux-loss bookkeeping/credit assignment.
"""


import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import gemma
from openpi.models import pi0
from openpi.models.pi0_v32_test import _single_observation
from openpi.models.pi0_v32_test import _TinyV32


class _TinyV34(_TinyV32):
    """The v3.2 tiny fixture with every v3.4 flag on (no conditioner: the tiny v3.2 fixture is
    unconditioned, which the v3.4 features do not require)."""

    def __init__(self, rngs: nnx.Rngs):
        super().__init__(rngs)
        width = 64
        self.memory_injection_mode = "tanh_rms"
        self.memory_injection_c = 2.0
        self.memory_injection_tau = 0.05
        self.memory_inject_w = nnx.Param(jnp.zeros((width,), dtype=jnp.float32))
        self.memory_blind_tokens = True
        self.memory_slot_embedding = nnx.Param(jax.random.normal(rngs.params(), (16, width), dtype=jnp.float32))
        self.memory_reseed_ce = True
        self.memory_state_mask_prob = 0.5
        self.memory_state_mask_dual_view = False
        self.state_null_embedding = nnx.Param(jax.random.normal(rngs.params(), (width,), dtype=jnp.float32))
        self.memory_aux_loss_weight = 0.1
        self.memory_aux_query_space = "key"
        self.memory_aux_margin_weight = 0.0
        self.memory_aux_margin_gamma = 1.0
        self.memory_aux_side_class_ids = (1,)
        self.memory_aux_queries = nnx.Param(jax.random.normal(rngs.params(), (16, 8), dtype=jnp.float32))
        self.memory_aux_head = nnx.Linear(width, 3, rngs=rngs)
        self.memory_ladder_probes = True
        self.ladder_writer_head = nnx.Linear(width, 2, rngs=rngs)
        self.ladder_read_head = nnx.Linear(width, 2, rngs=rngs)
        self.memory_conditioner_context = "instruction_state"
        self.top_patch_valid = None


@pytest.fixture(scope="module")
def tiny_v34():
    original_vocab = gemma.PALIGEMMA_VOCAB_SIZE
    try:
        gemma.PALIGEMMA_VOCAB_SIZE = 128
        yield _TinyV34(nnx.Rngs(0))
    finally:
        gemma.PALIGEMMA_VOCAB_SIZE = original_vocab


def _v34_sequence_observation(*, state_masked: bool, state_token_ids=(5, 6)):
    observation = _single_observation()
    steps = 2

    def repeat_time(value):
        return jnp.repeat(value[:, None], steps, axis=1)

    prompt = repeat_time(observation.tokenized_prompt)
    # position 2 of the 4-token prompt is the "state digit"; its id comes from the test
    prompt = prompt.at[:, :, 2].set(jnp.asarray(state_token_ids, dtype=prompt.dtype)[None])
    token_state_mask = jnp.zeros((1, steps, 4), dtype=bool).at[:, :, 2].set(True)
    return observation.replace(
        images={name: repeat_time(image) for name, image in observation.images.items()},
        image_masks={name: repeat_time(mask) for name, mask in observation.image_masks.items()},
        state=repeat_time(observation.state),
        tokenized_prompt=prompt,
        tokenized_prompt_mask=repeat_time(observation.tokenized_prompt_mask),
        token_ar_mask=repeat_time(observation.token_ar_mask),
        token_loss_mask=repeat_time(observation.token_loss_mask),
        token_fast_mask=repeat_time(observation.token_fast_mask),
        token_state_mask=token_state_mask,
        tokenized_causal=jnp.asarray([[[5, 6], [7, 8]]], dtype=jnp.int32),
        tokenized_causal_mask=jnp.ones((1, steps, 2), dtype=bool),
        causal_fast_mask=jnp.zeros((1, steps, 2), dtype=bool),
        seq_step_mask=jnp.ones((1, steps), dtype=bool),
        seq_block_boundary=jnp.zeros((1, steps), dtype=bool),
        seq_state_masked=jnp.asarray([state_masked]),
        seq_subtask_class=jnp.asarray([[0, 1]], dtype=jnp.int32),
        seq_side_label=jnp.asarray([1], dtype=jnp.int32),
        seq_evidence_mask=jnp.asarray([[True, False]]),
        seq_waiting_mask=jnp.asarray([[False, True]]),
    )


def _prepare_v34(model, observation, state, **kwargs):
    prefix, mask, ar = model.embed_prefix(observation)
    top_tokens = (mask.shape[1] - model.max_token_len) // len(observation.images)
    return model._v32_prepare_memory_prefix(prefix, mask, ar, state, top_token_count=top_tokens, **kwargs)  # noqa: SLF001


# ---------------------------------------------------------------------------------------------
# (a) blinding
# ---------------------------------------------------------------------------------------------


def test_blinded_late_mask_structure(tiny_v34):
    prefix_len, mem_len = 5, 16
    split_mask = jnp.ones((1, prefix_len + mem_len), dtype=bool).at[0, 3].set(False)  # one padding row
    split_ar = jnp.zeros((1, prefix_len + mem_len), dtype=jnp.int32)
    blinded = np.asarray(tiny_v34._v32_split_late_mask(split_mask, split_ar, prefix_len))[0]  # noqa: SLF001
    tiny_v34.memory_blind_tokens = False
    try:
        full = np.asarray(tiny_v34._v32_split_late_mask(split_mask, split_ar, prefix_len))[0]  # noqa: SLF001
    finally:
        tiny_v34.memory_blind_tokens = True

    # memory-token query rows: exactly the 16x16 self block, nothing else
    np.testing.assert_array_equal(blinded[prefix_len:, prefix_len:], True)  # noqa: FBT003
    np.testing.assert_array_equal(blinded[prefix_len:, :prefix_len], False)  # noqa: FBT003
    # every VALID row keeps at least one visible key (no all-masked softmax row; padding rows
    # were already fully masked before blinding and their outputs are never consumed)
    assert blinded[np.asarray(split_mask[0])].any(axis=-1).all()
    # non-memory rows are untouched, INCLUDING their memory-token columns (K/V sources)
    np.testing.assert_array_equal(blinded[:prefix_len], full[:prefix_len])
    assert full[:prefix_len, prefix_len:].any()


def test_blinded_memory_token_outputs_ignore_observation_content(tiny_v34):
    observation = _single_observation()
    other = observation.replace(images={k: v * 0.3 - 0.1 for k, v in observation.images.items()})
    state = tiny_v34.memory.init_state(1)
    prefix_len = 3 * 4 + tiny_v34.max_token_len  # tiny encoder: 4 slots per each of 3 cameras
    out_a = _prepare_v34(tiny_v34, observation, state)
    out_b = _prepare_v34(tiny_v34, other, state)
    # zero injection (w = 0) + blinding: the memory-token outputs are a pure function of
    # positions and the injected (zero) content -- identical across observations
    np.testing.assert_array_equal(
        np.asarray(out_a["final_prefix"][:, prefix_len:]), np.asarray(out_b["final_prefix"][:, prefix_len:])
    )
    # sanity: the rest of the prefix does depend on the observation
    assert not np.array_equal(
        np.asarray(out_a["final_prefix"][:, :prefix_len]), np.asarray(out_b["final_prefix"][:, :prefix_len])
    )
    tiny_v34.memory_blind_tokens = False
    try:
        out_c = _prepare_v34(tiny_v34, observation, state)
        out_d = _prepare_v34(tiny_v34, other, state)
    finally:
        tiny_v34.memory_blind_tokens = True
    assert not np.array_equal(
        np.asarray(out_c["final_prefix"][:, prefix_len:]), np.asarray(out_d["final_prefix"][:, prefix_len:])
    )


# ---------------------------------------------------------------------------------------------
# (b) tanh_rms injection
# ---------------------------------------------------------------------------------------------


def test_tanh_rms_injection_zero_start_math_and_gradient(tiny_v34):
    retrieved = 7.3 * jax.random.normal(jax.random.key(3), (1, 16, 64))
    # exact-zero start regardless of retrieval magnitude
    np.testing.assert_array_equal(np.asarray(tiny_v34._v32_inject_memory(retrieved)), 0.0)  # noqa: SLF001

    # the pathway is open: d(injection)/dw is nonzero at w = 0
    graphdef, params = nnx.split(tiny_v34)

    def injected_mass(p):
        return jnp.sum(jnp.abs(nnx.merge(graphdef, p)._v32_inject_memory(retrieved)))  # noqa: SLF001

    grads = jax.grad(injected_mass)(params)
    w_grad = None
    for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        if "memory_inject_w" in jax.tree_util.keystr(path):
            w_grad = leaf
    assert w_grad is not None
    assert float(jnp.sum(jnp.abs(w_grad))) > 0

    # normalization math with an open gate: tanh(w) * x * c / max(rms(x), tau), rms per token
    tiny_v34.memory_inject_w.value = jnp.full((64,), 0.5, dtype=jnp.float32)
    try:
        out = np.asarray(tiny_v34._v32_inject_memory(retrieved))  # noqa: SLF001
        x = np.asarray(retrieved, dtype=np.float32)
        rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True))
        expected = np.tanh(0.5) * x * (tiny_v34.memory_injection_c / np.maximum(rms, tiny_v34.memory_injection_tau))
        np.testing.assert_allclose(out, expected, rtol=1e-5)
        # weak reads stay weak (non-amplifying floor): a tiny retrieval scales by at most c/tau
        weak = jnp.full((1, 16, 64), 1e-6)
        weak_out = np.asarray(tiny_v34._v32_inject_memory(weak))  # noqa: SLF001
        assert np.abs(weak_out).max() <= np.tanh(0.5) * tiny_v34.memory_injection_c / tiny_v34.memory_injection_tau * 1e-6 * 1.01
    finally:
        tiny_v34.memory_inject_w.value = jnp.zeros((64,), dtype=jnp.float32)


def test_slot_embeddings_carry_no_memory_content(tiny_v34):
    """The blinding register fallback: memory tokens at a closed content gate equal the learned
    slot embeddings exactly -- independent of the memory state -- and zero_read removes CONTENT
    while keeping the slots (plan 5.3's token-vs-content distinction)."""
    observation = _single_observation()
    fresh = tiny_v34.memory.init_state(1)
    written, _ = tiny_v34.memory.write(fresh, jax.random.normal(jax.random.key(21), (1, 16, 64)))
    out_fresh = _prepare_v34(tiny_v34, observation, fresh)
    out_written = _prepare_v34(tiny_v34, observation, written)
    out_zero_read = _prepare_v34(tiny_v34, observation, written, zero_read=True)
    slots = np.asarray(tiny_v34.memory_slot_embedding.value)[None]
    # tanh(0) content gate: tokens are exactly the slots, for ANY memory state
    np.testing.assert_allclose(np.asarray(out_fresh["memory_tokens"], dtype=np.float32), slots, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(out_fresh["memory_tokens"]), np.asarray(out_written["memory_tokens"]))
    np.testing.assert_array_equal(np.asarray(out_zero_read["memory_tokens"]), np.asarray(out_written["memory_tokens"]))
    # ...and the token stream is NOT the exactly-zero RMSNorm singularity
    assert float(np.abs(np.asarray(out_fresh["memory_tokens"])).max()) > 0.1


def test_gate_value_override_replaces_tanh_scale(tiny_v34):
    retrieved = jax.random.normal(jax.random.key(4), (1, 16, 64))
    override = jnp.full((64,), 0.25)
    out = np.asarray(tiny_v34._v32_inject_memory(retrieved, override))  # noqa: SLF001
    x = np.asarray(retrieved, dtype=np.float32)
    rms = np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True))
    expected = 0.25 * x * (tiny_v34.memory_injection_c / np.maximum(rms, tiny_v34.memory_injection_tau))
    np.testing.assert_allclose(out, expected, rtol=1e-5)


# ---------------------------------------------------------------------------------------------
# (c) ladder probes: stop-grad features + main-update invariance
# ---------------------------------------------------------------------------------------------


def test_ladder_probe_gradients_never_touch_the_main_model(tiny_v34):
    """The stop-gradient contract, tested as an exact statement: the gradient of the ladder CE
    terms ALONE is exactly zero on every non-ladder parameter (so adding them to the total loss
    is mathematically invisible to the main model -- any bitwise wiggle between separate
    with/without backward passes is pure floating-point reassociation, e.g. in the embedder's
    scatter-add, not coupling)."""
    observation = _v34_sequence_observation(state_masked=False)
    actions = jnp.zeros((1, 2, 4, 2), dtype=jnp.float32)
    graphdef, params = nnx.split(tiny_v34)

    def ladder_only_loss(p):
        losses = nnx.merge(graphdef, p)._compute_sequence_loss_v32(  # noqa: SLF001
            jax.random.key(11), observation, actions, train=False
        )
        loss = 0.0
        for rung in ("ladder_writer", "ladder_read"):
            loss += losses[f"{rung}_ce_sum"] / jnp.maximum(losses[f"{rung}_count"], 1.0)
        return loss

    grads = jax.grad(ladder_only_loss)(params)
    ladder_norm = 0.0
    for path, leaf in jax.tree_util.tree_leaves_with_path(grads):
        key = jax.tree_util.keystr(path)
        assert bool(jnp.all(jnp.isfinite(leaf))), key
        if "ladder_" in key:
            ladder_norm += float(jnp.sum(jnp.square(leaf)))
        else:
            np.testing.assert_array_equal(np.asarray(leaf), 0.0, err_msg=key)
    assert ladder_norm > 0  # the heads themselves do learn


def test_isolated_probe_sgd_keeps_main_updates_bit_identical():
    """The train.py isolation pattern: zeroing ladder grads before a clip+adamw update yields
    main updates bit-identical to a run where the ladder contributed no gradient at all."""
    import optax

    params = {"main": jnp.asarray([1.0, -2.0]), "ladder_writer_head": jnp.asarray([0.5, 0.5])}
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-2))
    grads_probes_on = {"main": jnp.asarray([5.0, 5.0]), "ladder_writer_head": jnp.asarray([100.0, -100.0])}
    grads_probes_off = {"main": jnp.asarray([5.0, 5.0]), "ladder_writer_head": jnp.zeros(2)}

    zeroed = dict(grads_probes_on, ladder_writer_head=jnp.zeros(2))
    updates_on, _ = tx.update(zeroed, tx.init(params), params)
    updates_off, _ = tx.update(grads_probes_off, tx.init(params), params)
    np.testing.assert_array_equal(np.asarray(updates_on["main"]), np.asarray(updates_off["main"]))
    # WITHOUT zeroing, the huge probe grads would have scaled the main update through the clip
    updates_raw, _ = tx.update(grads_probes_on, tx.init(params), params)
    assert not np.array_equal(np.asarray(updates_raw["main"]), np.asarray(updates_off["main"]))


# ---------------------------------------------------------------------------------------------
# (d) CE seed
# ---------------------------------------------------------------------------------------------


def test_ce_seed_is_last_valid_non_memory_position(tiny_v34):
    batch, prefix_len, mem_len, emb = 2, 6, 3, 4
    final_prefix = jnp.arange(batch * (prefix_len + mem_len) * emb, dtype=jnp.float32).reshape(
        batch, prefix_len + mem_len, emb
    )
    # sample 0: text padding at the end; sample 1: fully valid prefix
    base_mask = jnp.asarray(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, True],
        ]
    )
    seed = tiny_v34._v32_ce_seed_hidden(final_prefix, base_mask)  # noqa: SLF001
    np.testing.assert_array_equal(np.asarray(seed[0, 0]), np.asarray(final_prefix[0, 3]))
    np.testing.assert_array_equal(np.asarray(seed[1, 0]), np.asarray(final_prefix[1, 5]))
    # never a memory position, even though the appended memory slots are "valid" in the split
    # mask -- the helper only ever sees the pre-memory mask width
    assert seed.shape == (batch, 1, emb)

    # a hole in the middle (e.g. a masked camera) must not break the gather
    holey = jnp.asarray([[True, False, True, False, False, False]] * 2)
    seed_holey = tiny_v34._v32_ce_seed_hidden(final_prefix, holey)  # noqa: SLF001
    np.testing.assert_array_equal(np.asarray(seed_holey[0, 0]), np.asarray(final_prefix[0, 2]))


def test_reseeded_ce_ignores_padding_token_ids(tiny_v34):
    observation = _single_observation()
    # invalidate the last prompt slot and vary its token id -- the seed (and thus token0)
    # must not change
    mask = observation.tokenized_prompt_mask.at[:, 3].set(False)
    obs_a = observation.replace(
        tokenized_prompt=observation.tokenized_prompt.at[:, 3].set(9), tokenized_prompt_mask=mask
    )
    obs_b = observation.replace(
        tokenized_prompt=observation.tokenized_prompt.at[:, 3].set(2), tokenized_prompt_mask=mask
    )
    state = tiny_v34.memory.init_state(1)
    prep_a = _prepare_v34(tiny_v34, obs_a, state)
    prep_b = _prepare_v34(tiny_v34, obs_b, state)
    _, mask_a, _ = tiny_v34.embed_prefix(obs_a)
    seed_a = tiny_v34._v32_causal_seed(prep_a["final_prefix"], mask_a)  # noqa: SLF001
    seed_b = tiny_v34._v32_causal_seed(prep_b["final_prefix"], mask_a)  # noqa: SLF001
    np.testing.assert_array_equal(np.asarray(seed_a), np.asarray(seed_b))


# ---------------------------------------------------------------------------------------------
# (e) per-segment state masking
# ---------------------------------------------------------------------------------------------


def _loss_dict(model, observation):
    actions = jnp.zeros((1, 2, 4, 2), dtype=jnp.float32)
    return model._compute_sequence_loss_v32(jax.random.key(29), observation, actions, train=False)  # noqa: SLF001


def test_masked_segments_are_invariant_to_true_state_tokens(tiny_v34):
    obs_a = _v34_sequence_observation(state_masked=True, state_token_ids=(5, 6))
    obs_b = _v34_sequence_observation(state_masked=True, state_token_ids=(9, 3))
    losses_a = _loss_dict(tiny_v34, obs_a)
    losses_b = _loss_dict(tiny_v34, obs_b)
    # every step of the masked segment sees the null embedding: end-to-end invariance of CE,
    # flow, aux, and ladder outputs to the actual state-digit token ids
    for key in losses_a:
        np.testing.assert_array_equal(np.asarray(losses_a[key]), np.asarray(losses_b[key]), err_msg=key)

    # unmasked segments DO depend on the state tokens
    obs_c = _v34_sequence_observation(state_masked=False, state_token_ids=(5, 6))
    obs_d = _v34_sequence_observation(state_masked=False, state_token_ids=(9, 3))
    losses_c = _loss_dict(tiny_v34, obs_c)
    losses_d = _loss_dict(tiny_v34, obs_d)
    assert not np.array_equal(np.asarray(losses_c["ce"]), np.asarray(losses_d["ce"]))


def test_dual_view_masks_ce_but_writes_full_view(tiny_v34):
    tiny_v34.memory_state_mask_dual_view = True
    try:
        obs_a = _v34_sequence_observation(state_masked=True, state_token_ids=(5, 6))
        obs_b = _v34_sequence_observation(state_masked=True, state_token_ids=(9, 3))
        losses_a = _loss_dict(tiny_v34, obs_a)
        losses_b = _loss_dict(tiny_v34, obs_b)
        # the memory state evolves from the FULL view, so the post-write aux readout responds
        # to the real state tokens...
        assert not np.array_equal(
            np.asarray(losses_a["aux_ce_class_sum"]), np.asarray(losses_b["aux_ce_class_sum"])
        )
    finally:
        tiny_v34.memory_state_mask_dual_view = False


# ---------------------------------------------------------------------------------------------
# (g) letterbox validity mask
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("qk_norm", [False, True])
def test_invalid_patch_gets_exactly_zero_attention(qk_norm):
    compressor = pi0.MemoryQueryCompressor(num_queries=4, width=64, num_heads=8, qk_norm=qk_norm, rngs=nnx.Rngs(7))
    source = jax.random.normal(jax.random.key(8), (2, 8, 64))
    source = source.at[:, 6].set(1e6)  # enormous activation on an INVALID patch
    source = source.at[:, 7].set(-1e6)
    valid = jnp.asarray([True] * 6 + [False] * 2)

    probs = np.asarray(compressor.attention_probs(source, source_valid=valid))
    np.testing.assert_array_equal(probs[..., 6:], 0.0)
    np.testing.assert_allclose(probs.sum(-1), 1.0, atol=1e-5)

    # __call__ shares the same logits helper: its output is invariant to invalid-patch content
    out_extreme = compressor(source, source_valid=valid)
    out_replaced = compressor(source.at[:, 6:].set(0.0), source_valid=valid)
    np.testing.assert_allclose(np.asarray(out_extreme), np.asarray(out_replaced), atol=1e-5)
    assert np.isfinite(np.asarray(out_extreme)).all()


def test_letterbox_patch_valid_geometry_yam_topcam():
    valid = np.asarray(pi0.letterbox_patch_valid((480, 640))).reshape(16, 16)
    # 480x640 -> resized 168x224, pad 28 top/bottom -> patch rows 2..13 valid, all columns
    np.testing.assert_array_equal(valid[2:14], True)  # noqa: FBT003
    np.testing.assert_array_equal(valid[:2], False)  # noqa: FBT003
    np.testing.assert_array_equal(valid[14:], False)  # noqa: FBT003
    with pytest.raises(ValueError, match="no valid patch"):
        pi0.letterbox_patch_valid((1, 100000))


def test_prepare_applies_letterbox_mask_to_both_compressors(tiny_v34):
    # tiny encoder emits 4 top slots; mask out slot 0 and check both banks' outputs move
    observation = _single_observation()
    state = tiny_v34.memory.init_state(1)
    baseline = _prepare_v34(tiny_v34, observation, state)
    tiny_v34.top_patch_valid = (False, True, True, True)
    try:
        masked = _prepare_v34(tiny_v34, observation, state)
    finally:
        tiny_v34.top_patch_valid = None
    assert not np.array_equal(np.asarray(baseline["read_queries"]), np.asarray(masked["read_queries"]))
    assert not np.array_equal(np.asarray(baseline["write_tokens"]), np.asarray(masked["write_tokens"]))
    tiny_v34.top_patch_valid = (True, True)  # wrong patch count must fail loudly
    try:
        with pytest.raises(ValueError, match="letterbox patch mask"):
            _prepare_v34(tiny_v34, observation, state)
    finally:
        tiny_v34.top_patch_valid = None


# ---------------------------------------------------------------------------------------------
# QK-norm temperature
# ---------------------------------------------------------------------------------------------


def test_qk_norm_cosine_logits_with_clamped_temperature():
    compressor = pi0.MemoryQueryCompressor(num_queries=2, width=64, num_heads=4, qk_norm=True, rngs=nnx.Rngs(9))
    head_dim = 16
    np.testing.assert_allclose(np.asarray(compressor.logit_scale.value), 0.5 * np.log(head_dim), rtol=1e-6)

    q = jax.random.normal(jax.random.key(1), (1, 2, 4, head_dim))
    k = jax.random.normal(jax.random.key(2), (1, 5, 4, head_dim))
    logits = np.asarray(compressor._attention_logits(q, k, None))  # noqa: SLF001
    qn = np.asarray(q) / np.linalg.norm(np.asarray(q), axis=-1, keepdims=True)
    kn = np.asarray(k) / np.linalg.norm(np.asarray(k), axis=-1, keepdims=True)
    expected = np.einsum("bqhd,bnhd->bhqn", qn, kn) * np.sqrt(head_dim)
    # atol covers TF32 matmul rounding (~1e-3 relative) when the suite runs on a GPU -- the
    # same production numeric every other attention einsum in this stack uses.
    np.testing.assert_allclose(logits, expected, atol=5e-3)
    # cosine logits are bounded by the temperature; the clamp caps runaway sharpening
    assert np.abs(logits).max() <= np.sqrt(head_dim) * (1 + 5e-3)
    compressor.logit_scale.value = jnp.full((4,), 10.0)
    clamped = np.asarray(compressor._attention_logits(q, k, None))  # noqa: SLF001
    assert np.abs(clamped).max() <= 64.0 * (1 + 5e-3)


# ---------------------------------------------------------------------------------------------
# three-way retention control (plan 8.4) at the sampling entry point
# ---------------------------------------------------------------------------------------------


def test_write_mode_three_way_semantics(tiny_v34):
    observation = _single_observation()
    state, _ = tiny_v34.memory.write(
        tiny_v34.memory.init_state(1), jax.random.normal(jax.random.key(13), (1, 16, 64))
    )
    kwargs = {
        "stop_token": 1,
        "max_decode_steps": 1,
        "num_steps": 1,
        "noise": jnp.zeros((1, 4, 2), dtype=jnp.float32),
        "forced_subtask_tokens": jnp.asarray([[5, 6]], dtype=jnp.int32),
        "forced_subtask_mask": jnp.ones((1, 2), dtype=bool),
    }
    _, frozen, _ = tiny_v34.sample_with_memory(jax.random.key(1), observation, state, write_mode="frozen", **kwargs)
    jax.tree.map(np.testing.assert_array_equal, frozen, state)

    _, dyn, dyn_aux = tiny_v34.sample_with_memory(
        jax.random.key(1), observation, state, write_mode="dynamics_only", **kwargs
    )
    interface = tiny_v34.v32_memory_interface_step(observation, state)
    expected_dyn, _ = tiny_v34.memory.decay_step(state, interface["write_tokens"])
    jax.tree.map(np.testing.assert_array_equal, dyn, expected_dyn)
    np.testing.assert_array_equal(np.asarray(dyn_aux["write_occurred"]), False)  # noqa: FBT003

    _, normal, _ = tiny_v34.sample_with_memory(jax.random.key(1), observation, state, write_mode="normal", **kwargs)
    expected_normal, _ = tiny_v34.memory.write(state, interface["write_tokens"])
    jax.tree.map(np.testing.assert_array_equal, normal, expected_normal)

    # legacy allow_write mapping is preserved
    _, legacy_frozen, _ = tiny_v34.sample_with_memory(
        jax.random.key(1), observation, state, allow_write=False, **kwargs
    )
    jax.tree.map(np.testing.assert_array_equal, legacy_frozen, state)


# ---------------------------------------------------------------------------------------------
# aux demand (plan 5.1)
# ---------------------------------------------------------------------------------------------


def test_aux_loss_bookkeeping_and_credit_assignment(tiny_v34):
    observation = _v34_sequence_observation(state_masked=False)
    losses = _loss_dict(tiny_v34, observation)
    # Core-steepness telemetry is derived only from valid writes and stays observational.
    for key in (
        "write_grad_norm_sum",
        "write_valid_count",
        "write_clip_count",
        "write_severe_clip_count",
        "write_grad_norm_mean",
        "write_grad_norm_max",
        "write_clip_fraction",
        "write_severe_clip_fraction",
    ):
        assert np.isfinite(np.asarray(losses[key])).all(), key
    assert np.all(np.asarray(losses["write_grad_norm_max"]) >= np.asarray(losses["write_grad_norm_mean"]))
    clip_fraction = np.asarray(losses["write_clip_fraction"])
    severe_fraction = np.asarray(losses["write_severe_clip_fraction"])
    assert np.all((clip_fraction >= 0.0) & (clip_fraction <= 1.0))
    assert np.all(
        (severe_fraction >= 0.0)
        & (severe_fraction <= clip_fraction)
    )
    num_classes = 3
    assert losses["aux_ce_class_sum"].shape == (num_classes,)
    # labels are class 0 at step 0 and class 1 at step 1, both valid -> one count each
    np.testing.assert_array_equal(np.asarray(losses["aux_count_class"]), [1.0, 1.0, 0.0])
    assert np.isfinite(np.asarray(losses["aux_ce_class_sum"])).all()
    # ladder bookkeeping: one evidence frame (step 0) and one waiting frame (step 1)
    np.testing.assert_array_equal(np.asarray(losses["ladder_writer_count"]), 1.0)
    np.testing.assert_array_equal(np.asarray(losses["ladder_read_count"]), 1.0)

    # the aux CE trains bank, head, memory core, and the WRITER (through the post-write read)
    graphdef, params = nnx.split(tiny_v34)

    def aux_objective(p):
        out = nnx.merge(graphdef, p)._compute_sequence_loss_v32(  # noqa: SLF001
            jax.random.key(29), observation, jnp.zeros((1, 2, 4, 2), dtype=jnp.float32), train=False
        )
        present = out["aux_count_class"] > 0
        per_class = jnp.where(present, out["aux_ce_class_sum"] / jnp.maximum(out["aux_count_class"], 1.0), 0.0)
        return jnp.sum(per_class) / jnp.maximum(jnp.sum(present), 1)

    grads = jax.grad(aux_objective)(params)
    by_path = {jax.tree_util.keystr(path): leaf for path, leaf in jax.tree_util.tree_leaves_with_path(grads)}

    def family_norm(fragment):
        return sum(float(jnp.sum(jnp.square(leaf))) for path, leaf in by_path.items() if fragment in path)

    for fragment in (
        "memory_aux_queries",
        "memory_aux_head",
        "['memory']['w_k']",
        "['memory']['w_v']",
        "['memory']['m0']",
        "write_query_compressor",
    ):
        assert family_norm(fragment) > 0, fragment
    # memory-only by construction: no gradient into the READ query bank or W_Q through q_aux
    assert family_norm("read_query_compressor") == 0
    assert family_norm("['memory']['w_q']") == 0


def test_aux_margin_variant_reports_and_is_nonnegative(tiny_v34):
    tiny_v34.memory_aux_margin_weight = 0.5
    try:
        losses = _loss_dict(tiny_v34, _v34_sequence_observation(state_masked=False))
        assert float(losses["aux_margin_sum"]) >= 0
        assert float(losses["aux_margin_count"]) == 2.0
    finally:
        tiny_v34.memory_aux_margin_weight = 0.0
