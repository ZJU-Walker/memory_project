"""Checks for phase I: the memory co-training loss (`Pi0._compute_loss_with_memory`).

Stage 1 (default) is CPU-safe and torch-free -- runs anywhere:
    uv run python scripts/check_memory_train.py
The centerpiece is the train/inference equivalence test: teacher-forcing the tokens that
`sample_with_memory` generated (same memory state, nonzero gate) through the training-side
forward must reproduce them exactly under argmax -- proving that the masks, positions, and
cache layout of training and inference are identical.

Stage 2 (--real) runs a real batch through the pi05_yam_mem config on iris-hgx-2 (GPU + dataset):
    CUDA_VISIBLE_DEVICES=<free> uv run python scripts/check_memory_train.py --real
"""

import dataclasses
import logging

import einops
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import memory
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import make_attn_mask
from openpi.models.pi0 import make_memory_step_mask


@dataclasses.dataclass
class Args:
    real: bool = False
    config: str = "pi05_yam_mem_v2"
    # checkpoint grafted into the memory model for the stage-2 check
    ckpt: str = "gs://openpi-assets/checkpoints/pi05_base/params"


def _dummy_setup(
    window: int = 4,
    live: int = 1,
    grad_every: int = 2,
    remat_chunk: int = 1,
    probe_weight: float = 0.0,
):
    mem_cfg = memory.MemoryConfig(d_input=64, d_key=16, hidden_dims=(32, 32, 32), d_value=64)
    config = pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        dtype="float32",
        pi05=True,
        predict_subtask=True,
        predict_with_memory=True,
        memory_layer=2,
        causal_token_len=16,
        memory=mem_cfg,
        memory_window=window,
        memory_live_writes=live,
        memory_grad_every=grad_every,
        memory_remat_chunk=remat_chunk,
        memory_probe_weight=probe_weight,
        memory_probe_classes=2,
    )
    model = config.create(jax.random.key(0))
    # nonzero gate so the memory content actually flows. Fresh-model gotcha: the SigLIP head is
    # zero-init, so use an ar=0 prompt to keep the image rows alive (see check_memory_read.py).
    model.memory_gate.value = 0.1 * jax.random.normal(jax.random.key(1), model.memory_gate.value.shape)
    obs = config.fake_obs(1)
    obs = obs.replace(token_ar_mask=jnp.zeros_like(obs.token_ar_mask))
    return config, model, obs


def _window_inputs(config, obs, key, wl: int):
    """Random live-window inputs: per-camera images [1, wl, h, w, c] plus tokenized contexts."""
    keys = jax.random.split(key, wl)
    images = {
        k: jnp.stack([jax.random.uniform(keys[i], v.shape, minval=-1, maxval=1) for i in range(wl)], axis=1)
        for k, v in obs.images.items()
    }
    tokens = jnp.broadcast_to(obs.tokenized_prompt[:, None], (1, wl, config.max_token_len))
    masks = jnp.broadcast_to(obs.tokenized_prompt_mask[:, None], (1, wl, config.max_token_len))
    return images, tokens, masks


def _train_obs(obs, gen_tokens, gen_mask, hiddens, window, write_mask):
    window_images, window_tokens, window_token_masks = window
    return obs.replace(
        tokenized_causal=gen_tokens,
        tokenized_causal_mask=gen_mask,
        causal_fast_mask=jnp.zeros_like(gen_mask),
        memory_hiddens=hiddens,
        memory_cache_indices=jnp.zeros(hiddens.shape[:2], dtype=jnp.int32),
        window_images=window_images,
        window_tokens=window_tokens,
        window_token_masks=window_token_masks,
        memory_write_mask=write_mask,
    )


def _ce_logits(model, observation):
    """The training CE logits, recomputed step-for-step as `_compute_loss_with_memory` does.

    Used as the equivalence oracle: `check_equivalence` ties these logits to the real
    `compute_loss` numerically (identical CE) and to `sample_with_memory` behaviorally
    (argmax reproduces the generated tokens).
    """
    observation = _model.preprocess_observation(None, observation, train=False)
    batch = observation.state.shape[0]
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache, hidden = model.PaliGemma.llm(
        [prefix_tokens, None],
        mask=make_attn_mask(prefix_mask, prefix_ar_mask),
        positions=positions,
        return_hidden_states=True,
    )
    num_img = prefix_mask.shape[1] - model.max_token_len
    mem_len = num_img // len(observation.images)
    prefix_len = prefix_mask.shape[1]
    causal_len = observation.tokenized_causal.shape[1]
    h_t = hidden[0][model.memory_layer][:, :mem_len].astype(jnp.float32)

    def window_hiddens(sl: slice, n: int):
        w_obs = _model.Observation(
            images={k: v[:, sl].reshape(batch * n, *v.shape[2:]) for k, v in observation.window_images.items()},
            image_masks={k: jnp.ones(batch * n, dtype=bool) for k in observation.window_images},
            state=jnp.zeros((batch * n, observation.state.shape[-1]), observation.state.dtype),
            tokenized_prompt=observation.window_tokens[:, sl].reshape(batch * n, -1),
            tokenized_prompt_mask=observation.window_token_masks[:, sl].reshape(batch * n, -1),
        )
        w_tokens, w_mask, w_ar = model.embed_prefix(w_obs)
        _, _, w_hidden = model.PaliGemma.llm(
            [w_tokens, None],
            mask=make_attn_mask(w_mask, w_ar),
            positions=jnp.cumsum(w_mask, axis=1) - 1,
            return_hidden_states=True,
        )
        return w_hidden[0][model.memory_layer][:, :mem_len].astype(jnp.float32).reshape(batch, n, mem_len, -1)

    # single all-live fold + plain python write loop: deliberately independent of the loss's
    # grouped/no-grad passes and chunked scan -- forward values must be identical anyway
    wl = observation.window_tokens.shape[1]
    write_seq = jnp.concatenate(
        [observation.memory_hiddens.astype(jnp.float32), window_hiddens(slice(None), wl)], axis=1
    )

    memory_state = model.memory.init_state(batch)
    for k in range(write_seq.shape[1]):
        new_state, _ = model.memory.write(memory_state, write_seq[:, k])
        valid = observation.memory_write_mask[:, k]
        memory_state = jax.tree.map(
            lambda n, o: jnp.where(valid.reshape(valid.shape + (1,) * (n.ndim - 1)), n, o),  # noqa: B023
            new_state,
            memory_state,
        )

    retrieved = model.memory.read(memory_state, h_t)
    mem_tokens = (model.memory_gate.value * retrieved).astype(prefix_tokens.dtype)
    causal_emb = model.PaliGemma.llm(observation.tokenized_causal, method="embed")
    ext_tokens = jnp.concatenate([mem_tokens, causal_emb], axis=1)
    causal_mask = observation.tokenized_causal_mask
    mem_rows = make_memory_step_mask(prefix_mask, prefix_ar_mask, mem_len, causal_len)
    tri = jnp.tril(jnp.ones((causal_len, causal_len), dtype=bool))
    causal_rows = jnp.concatenate(
        [
            einops.repeat(prefix_mask, "b p -> b c p", c=causal_len),
            jnp.ones((batch, causal_len, mem_len), dtype=bool),
            tri[None] & causal_mask[:, None, :],
        ],
        axis=-1,
    )
    ext_mask = jnp.concatenate([mem_rows, causal_rows], axis=1)
    ext_positions = jnp.broadcast_to(prefix_len + jnp.arange(mem_len + causal_len)[None], (batch, mem_len + causal_len))
    kv_cache = jax.tree.map(lambda x: jnp.pad(x, ((0, 0), (0, 0), (0, mem_len + causal_len), (0, 0), (0, 0))), kv_cache)
    (ext_out, _), _ = model.PaliGemma.llm(
        [ext_tokens, None], mask=ext_mask, positions=ext_positions, kv_cache=kv_cache, cache_position=prefix_len
    )
    mem_out, causal_out = ext_out[:, :mem_len], ext_out[:, mem_len:]
    ce_hidden = jnp.concatenate([mem_out[:, -1:], causal_out[:, :-1]], axis=1)
    return model.PaliGemma.llm(ce_hidden, method="decode").astype(jnp.float32)


def check_equivalence() -> None:
    config, model, obs = _dummy_setup()
    keys = jax.random.split(jax.random.key(2), 4)

    hiddens = jax.random.normal(keys[0], (1, 3, 256, config.memory.d_input)) * 2.0
    window = _window_inputs(config, obs, keys[1], wl=1)
    write_mask = jnp.ones((1, 4), dtype=bool)

    # inference side: replay the same write sequence by hand, then run the fused sampler
    w_single = obs.replace(images={k: v[:, 0] for k, v in window[0].items()})
    h_live = model.extract_topcam_hidden(w_single)[config.memory_layer].astype(jnp.float32)
    state = model.memory.init_state(1)
    for k in range(3):
        state, _ = model.memory.write(state, hiddens[:, k])
    state, _ = model.memory.write(state, h_live)
    _, _, aux = model.sample_with_memory(keys[2], obs, state, stop_token=7, max_decode_steps=6, num_steps=2)
    gen_tokens, gen_mask = aux["tokens"], aux["token_mask"]
    n_gen = int(jnp.sum(gen_mask))
    assert n_gen >= 2, f"want a multi-token generation for a meaningful test, got {n_gen}"

    # training side: teacher-force those very tokens
    train_obs = _train_obs(obs, gen_tokens, gen_mask, hiddens, window, write_mask)
    logits = _ce_logits(model, train_obs)
    pred = jnp.argmax(logits, axis=-1)
    for i in range(n_gen):
        assert int(pred[0, i]) == int(gen_tokens[0, i]), (
            f"train/inference divergence at causal position {i}: "
            f"train argmax {int(pred[0, i])} vs generated {int(gen_tokens[0, i])}"
        )

    # tie the oracle to the real loss: its NLL must equal compute_loss's ce
    logp = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), gen_tokens[..., None], axis=-1)[..., 0]
    ce_ref = float((-jnp.sum(logp * gen_mask, axis=-1) / jnp.clip(jnp.sum(gen_mask, axis=-1), 1))[0])
    actions = jnp.zeros((1, config.action_horizon, config.action_dim))
    losses = model.compute_loss(jax.random.key(9), train_obs, actions, train=False)
    np.testing.assert_allclose(float(losses["ce"][0]), ce_ref, rtol=1e-5)
    print(f"[OK] train == inference: argmax reproduces all {n_gen} generated tokens; ce ties out ({ce_ref:.4f})")


def check_loss_and_grads() -> None:
    config, model, obs = _dummy_setup()
    keys = jax.random.split(jax.random.key(3), 4)
    hiddens = jax.random.normal(keys[0], (1, 3, 256, config.memory.d_input)) * 2.0
    window = _window_inputs(config, obs, keys[1], wl=1)
    gen_tokens = (
        jnp.zeros((1, config.causal_token_len), dtype=jnp.int32).at[0, :5].set(jnp.asarray([11, 12, 13, 14, 7]))
    )
    gen_mask = jnp.zeros((1, config.causal_token_len), dtype=bool).at[0, :5].set(True)
    train_obs = _train_obs(obs, gen_tokens, gen_mask, hiddens, window, jnp.ones((1, 4), dtype=bool))
    actions = jax.random.normal(keys[2], (1, config.action_horizon, config.action_dim)) * 0.1

    losses = model.compute_loss(keys[3], train_obs, actions, train=False)
    assert set(losses) == {"flow", "ce"}
    assert losses["flow"].shape == (1, config.action_horizon)
    assert losses["ce"].shape == (1,)
    assert bool(jnp.all(jnp.isfinite(losses["flow"])))
    assert bool(jnp.isfinite(losses["ce"][0]))
    print(f"[OK] memory loss runs: flow {float(jnp.mean(losses['flow'])):.4f}, ce {float(losses['ce'][0]):.4f}")

    # write-mask invariance: with all writes masked out, the hidden contents must not matter
    obs_a = train_obs.replace(memory_write_mask=jnp.zeros((1, 4), dtype=bool))
    obs_b = obs_a.replace(memory_hiddens=hiddens * 5.0 + 1.0)
    la = model.compute_loss(keys[3], obs_a, actions, train=False)
    lb = model.compute_loss(keys[3], obs_b, actions, train=False)
    np.testing.assert_allclose(np.asarray(la["ce"]), np.asarray(lb["ce"]), atol=1e-6)
    print("[OK] masked writes are fully ignored (episode-start behavior)")

    # gradient routing
    graphdef, params = nnx.split(model)

    def loss_of(params, which):
        m = nnx.merge(graphdef, params)
        return jnp.mean(m.compute_loss(keys[3], train_obs, actions, train=False)[which])

    for which, must_have, must_not_have in (
        ("flow", ["_1", "action_out_proj"], ["memory"]),
        ("ce", ["memory_gate", "'m0'", "'w_k'", "'w_v'"], ["_1", "action_out_proj", "action_in_proj", "time_mlp"]),
    ):
        grads = jax.grad(loss_of)(params, which)
        leaves = jax.tree_util.tree_leaves_with_path(grads)
        assert all(bool(jnp.all(jnp.isfinite(g))) for _, g in leaves)
        norms = {jax.tree_util.keystr(p): float(jnp.linalg.norm(g)) for p, g in leaves}
        for name in must_have:
            assert sum(v for k, v in norms.items() if name in k) > 0, f"{which}: no gradient reaches {name}"
        for name in must_not_have:
            total = sum(v for k, v in norms.items() if name in k)
            assert total == 0, f"{which}: gradient leaked into {name} ({total})"
        if which == "ce":
            vlm = sum(v for k, v in norms.items() if "q_einsum" in k and "_1" not in k)
            assert vlm > 0, "ce gradient does not reach the VLM"
    print("[OK] gradient routing: flow -> action expert only; ce -> VLM + memory + gate only")


def _bump_siglip_head(model) -> None:
    # fresh-model gotcha: the zero-init SigLIP head multiplicatively blocks gradients to input
    # images; perturb it so image-gradient probes can see anything at all
    state = nnx.state(model)
    bumped = []

    def bump(path, leaf):
        ks = jax.tree_util.keystr(path)
        if "head" in ks and "kernel" in ks:
            bumped.append(ks)
            return 0.02 * jax.random.normal(jax.random.key(11), leaf.shape, leaf.dtype)
        return leaf

    nnx.update(model, jax.tree_util.tree_map_with_path(bump, state))
    assert bumped, "did not find the SigLIP head kernel to perturb"


def check_grad_grid_and_full_bptt() -> None:
    # window=4 as [cached0, cached1, live0, live1], grad_every=2: the grid (counted from the
    # window end) is positions {1, 3}, so live1 keeps VLM gradients and live0 runs forward-only.
    config, model, obs = _dummy_setup(window=4, live=2, grad_every=2)
    keys = jax.random.split(jax.random.key(4), 4)
    _bump_siglip_head(model)
    hiddens = jax.random.normal(keys[0], (1, 2, 256, config.memory.d_input)) * 2.0
    window = _window_inputs(config, obs, keys[1], wl=2)
    gen_tokens = jnp.zeros((1, config.causal_token_len), dtype=jnp.int32).at[0, :3].set(jnp.asarray([11, 12, 7]))
    gen_mask = jnp.zeros((1, config.causal_token_len), dtype=bool).at[0, :3].set(True)
    train_obs = _train_obs(obs, gen_tokens, gen_mask, hiddens, window, jnp.ones((1, 4), dtype=bool))
    actions = jnp.zeros((1, config.action_horizon, config.action_dim))
    graphdef, params = nnx.split(model)

    def ce_of_window(wimgs):
        m = nnx.merge(graphdef, params)
        return jnp.mean(m.compute_loss(keys[2], train_obs.replace(window_images=wimgs), actions, train=False)["ce"])

    g = jax.grad(ce_of_window)(train_obs.window_images)
    on_grid = sum(float(jnp.linalg.norm(v[:, 1])) for v in jax.tree.leaves(g))
    off_grid = sum(float(jnp.linalg.norm(v[:, 0])) for v in jax.tree.leaves(g))
    assert on_grid > 0, "the grad-grid live frame received no image gradient"
    assert off_grid == 0, f"an off-grid live frame received image gradient ({off_grid})"
    print(f"[OK] VLM grad grid: on-grid frame grad norm {on_grid:.3g}, off-grid frame exactly 0")

    # full BPTT: with ONLY the oldest write valid, the CE gradient must still reach the memory
    # projections -- it travels through the entire 4-step state recursion (3 masked no-ops)
    deep_obs = train_obs.replace(
        memory_write_mask=jnp.asarray([[True, False, False, False]]),
    )

    def ce_of_params(p):
        return jnp.mean(nnx.merge(graphdef, p).compute_loss(keys[2], deep_obs, actions, train=False)["ce"])

    grads = jax.grad(ce_of_params)(params)
    norms = {jax.tree_util.keystr(p): float(jnp.linalg.norm(g)) for p, g in jax.tree_util.tree_leaves_with_path(grads)}
    for name in ("'w_k'", "'w_v'", "'m0'"):
        assert sum(v for k, v in norms.items() if name in k) > 0, f"full-BPTT gradient does not reach {name}"
    print("[OK] full BPTT: oldest write (depth = window) still gradients w_k/w_v/m0 -- no truncation anywhere")

    # Gradient checkpointing must not change values or gradients. remat_chunk=4 vs 1: the write
    # chain runs as one checkpointed chunk of 4 vs four chunks of 1, while the window no-grad
    # fold keeps the same group size (wl=2 divides neither, both fall back to per-frame groups)
    # -- otherwise a different fold batch size perturbs BLAS reductions at the 1e-6 level.
    _, model2, _ = _dummy_setup(window=4, live=2, grad_every=2, remat_chunk=4)
    nnx.update(model2, nnx.state(model))  # same weights (incl. the SigLIP bump + gate)
    graphdef2, params2 = nnx.split(model2)
    l1 = model.compute_loss(keys[2], train_obs, actions, train=False)
    l2 = nnx.merge(graphdef2, params2).compute_loss(keys[2], train_obs, actions, train=False)
    np.testing.assert_allclose(np.asarray(l1["ce"]), np.asarray(l2["ce"]), rtol=1e-6)

    def ce1_of_params(p):
        return jnp.mean(nnx.merge(graphdef, p).compute_loss(keys[2], train_obs, actions, train=False)["ce"])

    def ce2_of_params(p):
        return jnp.mean(nnx.merge(graphdef2, p).compute_loss(keys[2], train_obs, actions, train=False)["ce"])

    g1 = jax.grad(ce1_of_params)(params)
    g2 = jax.grad(ce2_of_params)(params2)
    for (p1, a), (_, b) in zip(
        jax.tree_util.tree_leaves_with_path(g1), jax.tree_util.tree_leaves_with_path(g2), strict=True
    ):
        # Checkpointing recomputes the forward during backward, so XLA fuses/accumulates in a
        # different order and individual elements wiggle at f32 rounding level (worst on the
        # tied embedder's scatter-adds). Compare per-leaf relative NORM error instead: rounding
        # stays ~1e-4, while a structural difference shows up at O(1).
        a64, b64 = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        err = np.linalg.norm(a64 - b64)
        # absolute floor: some leaves' true gradient is mathematically zero (e.g. attention key
        # biases -- softmax is bias-invariant), leaving pure noise where relative error is
        # meaningless
        assert err <= 1e-3 * np.linalg.norm(a64) + 1e-5, (
            f"remat changed the gradient of {p1}: |diff| {err:.2e} vs |g| {np.linalg.norm(a64):.2e}"
        )
    print("[OK] remat chunking: loss and gradients equal (within f32 rounding) for chunk 1 vs 4")


def check_probes() -> None:
    # window=4 all-live, grad_every=2 -> probe grid positions {1, 3} (from the window end)
    config, model, obs = _dummy_setup(window=4, live=4, grad_every=2, remat_chunk=2, probe_weight=0.5)
    keys = jax.random.split(jax.random.key(5), 4)
    window = _window_inputs(config, obs, keys[0], wl=4)
    hiddens = jnp.zeros((1, 0, 256, config.memory.d_input))
    gen_tokens = jnp.zeros((1, config.causal_token_len), dtype=jnp.int32).at[0, :3].set(jnp.asarray([11, 12, 7]))
    gen_mask = jnp.zeros((1, config.causal_token_len), dtype=bool).at[0, :3].set(True)
    base = _train_obs(obs, gen_tokens, gen_mask, hiddens, window, jnp.ones((1, 4), dtype=bool))
    base = base.replace(memory_hiddens=None, memory_cache_indices=None)
    actions = jnp.zeros((1, config.action_horizon, config.action_dim))

    # quiz supervision: position 0 pre-reveal, the rest quizzable, position 1 still visible
    labels = jnp.ones((1, 4), dtype=jnp.int32)
    probe_mask = jnp.asarray([[False, True, True, True]])
    probe_visible = jnp.asarray([[False, True, False, False]])
    quiz_obs = base.replace(
        memory_probe_labels=labels, memory_probe_mask=probe_mask, memory_probe_visible=probe_visible
    )

    losses = model.compute_loss(keys[1], quiz_obs, actions, train=False)
    # active = grid {1,3} & mask {1,2,3} = {1,3}; visible & active = {1}
    assert float(losses["probe_count"][0]) == 2, losses["probe_count"]
    assert float(losses["probe_count_visible"][0]) == 1, losses["probe_count_visible"]
    assert losses["probe_correct_grid"].shape == (1, 2)
    assert losses["probe_active_grid"].shape == (1, 2)
    np.testing.assert_array_equal(np.asarray(losses["probe_active_grid"]), [[1.0, 1.0]])
    print("[OK] probe schedule: grid {1,3} & quizzable {1,2,3} -> 2 live quizzes, 1 visible")

    # purity: the quiz must not change the flow/ce losses at all (reading is stateless)
    plain = model.compute_loss(
        keys[1],
        base.replace(memory_probe_labels=None, memory_probe_mask=None, memory_probe_visible=None),
        actions,
        train=False,
    )
    np.testing.assert_array_equal(np.asarray(plain["ce"]), np.asarray(losses["ce"]))
    np.testing.assert_array_equal(np.asarray(plain["flow"]), np.asarray(losses["flow"]))
    assert set(plain) == {"flow", "ce"}
    print("[OK] probe purity: flow/ce bit-identical with quizzes on or off")

    # zero content gate -> pooled read is exactly 0 -> logits are the (zero-init) head bias:
    # every quiz CE is ln 2 and argmax picks class 0
    gate_backup = model.memory_gate.value
    model.memory_gate.value = jnp.zeros_like(gate_backup)
    zg = model.compute_loss(keys[1], quiz_obs, actions, train=False)
    np.testing.assert_allclose(float(zg["probe_ce_sum"][0]), 2 * float(jnp.log(2.0)), rtol=1e-6)
    assert float(zg["probe_correct"][0]) == 0  # labels are 1, argmax of [0, 0] is 0
    zg0 = model.compute_loss(keys[1], quiz_obs.replace(memory_probe_labels=jnp.zeros_like(labels)), actions, train=False)
    assert float(zg0["probe_correct"][0]) == 2
    model.memory_gate.value = gate_backup
    print("[OK] probe determinism at zero gate: ce = n*ln2, correctness follows the label")

    # gradient routing: the quiz loss trains head + gate + memory (+ VLM via the on-grid live
    # frame it reads through), and never touches the action expert
    graphdef, params = nnx.split(model)

    def probe_loss_of(p):
        losses = nnx.merge(graphdef, p).compute_loss(keys[1], quiz_obs, actions, train=False)
        return jnp.sum(losses["probe_ce_sum"]) / jnp.maximum(jnp.sum(losses["probe_count"]), 1)

    grads = jax.grad(probe_loss_of)(params)
    norms = {jax.tree_util.keystr(p): float(jnp.linalg.norm(g)) for p, g in jax.tree_util.tree_leaves_with_path(grads)}
    for name in ("probe_head", "memory_gate", "'w_k'", "'w_v'", "'w_q'", "'m0'"):
        assert sum(v for k, v in norms.items() if name in k) > 0, f"probe loss: no gradient reaches {name}"
    for name in ("_1", "action_out_proj", "action_in_proj", "time_mlp"):
        total = sum(v for k, v in norms.items() if name in k)
        assert total == 0, f"probe loss: gradient leaked into {name} ({total})"
    print("[OK] probe gradient routing: head + gate + memory (+VLM), action expert untouched")


def check_transforms() -> None:
    from openpi import transforms as _transforms
    from openpi.models import tokenizer as _tokenizer

    window, live, stride = 4, 2, 25
    split = _transforms.SplitMemoryWindow(window=window, live=live, stride=stride)
    rng = np.random.default_rng(0)
    item = {
        "observation/image": rng.random((live + 1, 3, 480, 640), dtype=np.float32),
        "observation/left_wrist_image": rng.random((live + 1, 3, 480, 640), dtype=np.float32),
        "observation/right_wrist_image": rng.random((live + 1, 3, 480, 640), dtype=np.float32),
        "observation/state": rng.random((live + 1, 14), dtype=np.float32),
        "actions": rng.random((50, 14), dtype=np.float32),
        "frame_index": 60,
        "index": 1060,  # episode starts at global index 1000
    }
    out = split(dict(item))
    assert out["observation/image"].shape == (480, 640, 3)
    assert out["observation/image"].dtype == np.uint8
    assert out["observation/state"].shape == (14,)
    assert out["window_images"]["base_0_rgb"].shape == (live, 224, 224, 3)
    assert float(out["window_images"]["base_0_rgb"].min()) >= -1.0
    assert float(out["window_images"]["base_0_rgb"].max()) <= 1.0
    assert out["window_state"].shape == (live, 14)
    # write offsets are [100, 75, 50, 25] frames; frame 60 -> only the two newest are valid
    np.testing.assert_array_equal(out["memory_write_mask"], [False, False, True, True])
    # detached slots reach before the episode start -> clamped to global index 1000
    np.testing.assert_array_equal(out["memory_cache_indices"], [1000, 1000])
    print("[OK] SplitMemoryWindow: shapes, write mask, episode-clamped cache indices")

    tok = _tokenizer.FASTSubtaskTokenizer(200)
    tokenize = _transforms.TokenizeMemorySubtaskInputs(tok, causal_len=150)
    state = rng.random(14, dtype=np.float32) * 2 - 1
    # smooth action chunks: white noise makes the FAST tokenizer blow past any budget
    t = np.linspace(0, 1, 50)[:, None]
    smooth_actions = (0.3 * np.sin(2 * np.pi * (t + np.linspace(0, 1, 14)[None]))).astype(np.float32)
    train_item = {
        "state": state,
        "actions": smooth_actions,
        "prompt": "find the bin with banana",
        "subtask": "open left bin",
        "window_state": out["window_state"],
    }
    t_out = tokenize(dict(train_item))
    assert t_out["tokenized_prompt"].shape == (200,)
    assert not t_out["token_ar_mask"].any(), "context must be pure ar=0"
    assert t_out["tokenized_causal"].shape == (150,)
    n_causal = int(t_out["tokenized_causal_mask"].sum())
    assert 0 < n_causal < 150
    first_fast = int(np.argmax(t_out["causal_fast_mask"]))
    assert t_out["causal_fast_mask"][first_fast:n_causal].all()
    assert int(t_out["tokenized_causal"][first_fast - 1]) == 108, "subtask terminator '\\n' must precede FAST"
    assert t_out["window_tokens"].shape == (live, 200)
    assert "window_state" not in t_out

    infer_out = tokenize({"state": state, "prompt": "find the bin with banana"})
    assert "tokenized_causal" not in infer_out
    assert infer_out["tokenized_prompt"].shape == (200,)
    print(f"[OK] TokenizeMemorySubtaskInputs: causal segment ({n_causal} tokens), window contexts, inference mode")

    # v2 path: window inferred from the stacked frames (window=0), stride 6, quiz fields.
    # 8 live frames, current frame 130 -> write frames [82..124] (offsets 48..6); with reveal
    # 100 / close 118, quizzable = frames >= 100 -> the newest 5 slots, visible = frames < 118.
    live = 8
    split = _transforms.SplitMemoryWindow(window=0, live=live, stride=6)
    item = {
        "observation/image": rng.random((live + 1, 3, 480, 640), dtype=np.float32),
        "observation/left_wrist_image": rng.random((live + 1, 3, 480, 640), dtype=np.float32),
        "observation/right_wrist_image": rng.random((live + 1, 3, 480, 640), dtype=np.float32),
        "observation/state": rng.random((live + 1, 14), dtype=np.float32),
        "frame_index": 130,
        "index": 2130,
        "quiz_side": np.int32(1),
        "reveal_frame": np.int32(100),
        "close_frame": np.int32(118),
    }
    out = split(dict(item))
    assert out["window_images"]["base_0_rgb"].shape == (live, 224, 224, 3)
    assert "memory_cache_indices" not in out, "all-live window must not emit cache indices"
    write_frames = 130 - np.arange(live, 0, -1) * 6
    np.testing.assert_array_equal(out["memory_write_mask"], np.ones(live, dtype=bool))
    np.testing.assert_array_equal(out["memory_probe_labels"], np.ones(live, dtype=np.int32))
    np.testing.assert_array_equal(out["memory_probe_mask"], write_frames >= 100)
    np.testing.assert_array_equal(out["memory_probe_visible"], (write_frames >= 100) & (write_frames < 118))
    assert int(out["memory_probe_mask"].sum()) == 5
    assert int(out["memory_probe_visible"].sum()) == 3

    # near the episode start the padded slots must be invalid AND unquizzable
    early = dict(item, frame_index=20, index=2020)
    out = split(early)
    np.testing.assert_array_equal(out["memory_write_mask"], 20 - np.arange(live, 0, -1) * 6 >= 0)
    assert not out["memory_probe_mask"][: live - 3].any()
    # side -1 (unlabeled episode) kills every quiz
    out = split(dict(item, quiz_side=np.int32(-1)))
    assert not out["memory_probe_mask"].any()
    print("[OK] SplitMemoryWindow v2: inferred window, quiz mask respects reveal/close/padding/side")

    quiz_info = _transforms.MemoryQuizInfo(
        episode_side=np.asarray([0, 1, -1], dtype=np.int32),
        episode_reveal=np.asarray([300, 250, 300], dtype=np.int32),
        episode_close=np.asarray([450, 400, 450], dtype=np.int32),
    )
    tagged = quiz_info({"episode_index": np.int64(1), "x": 0})
    assert (int(tagged["quiz_side"]), int(tagged["reveal_frame"]), int(tagged["close_frame"])) == (1, 250, 400)
    print("[OK] MemoryQuizInfo: per-episode side/reveal/close attached")


def check_real(args: Args) -> None:
    import time

    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader
    import openpi.training.memory_cache as _memory_cache

    config = _config.get_config(args.config)
    config = dataclasses.replace(config, batch_size=8, num_workers=0, exp_name="check")
    loader = _data_loader.create_data_loader(config, shuffle=True, num_batches=1)
    observation, actions = next(iter(loader))
    print(f"batch loaded: causal {observation.tokenized_causal.shape}, window {observation.window_tokens.shape}")

    # graft the checkpoint into the memory model (missing = fresh memory params)
    model = config.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    loaded = _config.weight_loaders.PartialCheckpointWeightLoader(args.ckpt).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)
    print(f"grafted {args.ckpt}")

    if config.model.memory_window > config.model.memory_live_writes:
        cache = _memory_cache.MemoryHiddenCache(loader.data_config(), config.model)
        graphdef, params = nnx.split(model)
        t0 = time.perf_counter()
        cache.refresh(graphdef, params)
        print(f"cache refresh: {len(cache)} frames in {time.perf_counter() - t0:.0f}s")
        indices = np.asarray(jax.device_get(observation.memory_cache_indices))
        observation = observation.replace(memory_hiddens=jnp.asarray(cache.gather(indices)))

    graphdef, params = nnx.split(model)

    @jax.jit
    def losses_of(params, observation, actions):
        return nnx.merge(graphdef, params).compute_loss(jax.random.key(1), observation, actions, train=True)

    t0 = time.perf_counter()
    losses = jax.block_until_ready(losses_of(params, observation, actions))
    print(f"first loss (incl. compile): {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    losses = jax.block_until_ready(losses_of(params, observation, actions))
    print(
        f"steady loss: {time.perf_counter() - t0:.2f}s | flow {float(jnp.mean(losses['flow'])):.4f} "
        f"ce {float(jnp.mean(losses['ce'])):.4f}"
    )
    if "probe_ce_sum" in losses:
        count = float(jnp.sum(losses["probe_count"]))
        vis = float(jnp.sum(losses["probe_count_visible"]))
        print(
            f"quiz: {count:.0f} live probes ({vis:.0f} visible) | "
            f"loss {float(jnp.sum(losses['probe_ce_sum'])) / max(count, 1):.4f} "
            f"acc {float(jnp.sum(losses['probe_correct'])) / max(count, 1):.2%}"
        )

    @jax.jit
    def grads_of(params, observation, actions):
        def loss_of(p):
            losses = nnx.merge(graphdef, p).compute_loss(jax.random.key(1), observation, actions, train=True)
            loss = jnp.mean(losses["flow"]) + jnp.mean(losses["ce"])
            if "probe_ce_sum" in losses:
                loss += 0.5 * jnp.sum(losses["probe_ce_sum"]) / jnp.maximum(jnp.sum(losses["probe_count"]), 1)
            return loss

        return jax.grad(loss_of)(params)

    grads = jax.block_until_ready(grads_of(params, observation, actions))
    norms = {
        jax.tree_util.keystr(p): float(jnp.linalg.norm(g.astype(jnp.float32)))
        for p, g in jax.tree_util.tree_leaves_with_path(grads)
    }
    for group, match in (
        ("memory", lambda k: "memory" in k),
        ("vlm attn", lambda k: "q_einsum" in k and "_1" not in k),
        ("action expert", lambda k: "_1" in k or "action_out_proj" in k),
    ):
        total = sum(v for k, v in norms.items() if match(k))
        print(f"grad norm [{group}]: {total:.4f}")
        assert total > 0, f"no gradient reaches {group}"
    assert all(np.isfinite(v) for v in norms.values())
    print("[OK] real batch: losses finite, gradients reach memory + VLM + action expert")


def main(args: Args) -> None:
    if args.real:
        # make the cache's progress logs visible (train.py configures logging; this script must too)
        logging.basicConfig(level=logging.INFO, force=True)
    check_transforms()
    check_equivalence()
    check_loss_and_grads()
    check_grad_grid_and_full_bptt()
    check_probes()
    if args.real:
        check_real(args)
    print("\nALL OK")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
