"""Checks for the memory SEQUENCE training loss (`Pi0._compute_sequence_loss`, RoboTTT-style).

Stage 1 (default) is CPU-safe and torch-free -- runs anywhere:
    JAX_PLATFORMS=cpu uv run python scripts/check_memory_train.py
The centerpiece is the per-step train/inference equivalence test: threading
`sample_with_memory` step by step over a sequence and teacher-forcing the tokens it generated
through the training-side forward must reproduce them exactly under argmax -- proving that the
per-step masks, positions, cache layout and the read-before-write order of training and
inference are identical.

Stage 2 (--real) runs a real batch through the current pi05_yam_mem_v31 config on the GPU box:
    uv run python scripts/check_memory_train.py --real
Use `--batch-size 12` only after the default four-sample smoke check succeeds.
"""

import dataclasses
import logging

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
    config: str = "pi05_yam_mem_v31"
    # Dummy/CPU diagnostics can exercise the v3 or v3.1 write representation.
    write_source: pi0_config.MemoryWriteSource = "post_attention"
    # Real diagnostics intentionally default to a small smoke batch; set 12 for the full recipe.
    batch_size: int = 4
    # checkpoint grafted into the memory model for the stage-2 check
    ckpt: str = "gs://openpi-assets/checkpoints/pi05_base/params"


def _dummy_setup(
    seq_steps: int = 4,
    block_steps: int = 0,
    probe_weight: float = 0.0,
    write_source: pi0_config.MemoryWriteSource = "post_attention",
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
        memory_write_source=write_source,
        causal_token_len=16,
        memory=mem_cfg,
        memory_seq_steps=seq_steps,
        memory_block_steps=block_steps,
        memory_probe_weight=probe_weight,
        memory_probe_classes=2,
    )
    model = config.create(jax.random.key(0))
    # nonzero gate so the memory content actually flows. Fresh-model gotcha: the SigLIP head is
    # zero-init, so use an ar=0 prompt to keep the image rows alive (see check_memory_read.py).
    model.memory_gate.value = 0.1 * jax.random.normal(jax.random.key(1), model.memory_gate.value.shape)
    return config, model


def _seq_obs(config, key, batch: int = 1):
    """A structurally-valid sequence observation: random images/states, a broadcast ar=0
    context prompt, empty causal segment, all steps valid, no gradient-block fences."""
    t = config.memory_seq_steps
    keys = jax.random.split(key, 4)
    images = {
        name: jax.random.uniform(k, (batch, t, 224, 224, 3), minval=-1, maxval=1)
        for name, k in zip(
            ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), jax.random.split(keys[0], 3), strict=True
        )
    }
    prompt = jax.random.randint(keys[1], (batch, 1, config.max_token_len), 2, 1000)
    return _model.Observation(
        images=images,
        image_masks={k: jnp.ones((batch, t), dtype=bool) for k in images},
        state=jax.random.uniform(keys[2], (batch, t, config.action_dim), minval=-1, maxval=1),
        tokenized_prompt=jnp.broadcast_to(prompt, (batch, t, config.max_token_len)),
        tokenized_prompt_mask=jnp.ones((batch, t, config.max_token_len), dtype=bool),
        token_ar_mask=jnp.zeros((batch, t, config.max_token_len), dtype=jnp.int32),
        token_loss_mask=jnp.zeros((batch, t, config.max_token_len), dtype=bool),
        token_fast_mask=jnp.zeros((batch, t, config.max_token_len), dtype=bool),
        tokenized_causal=jnp.zeros((batch, t, config.causal_token_len), dtype=jnp.int32),
        tokenized_causal_mask=jnp.zeros((batch, t, config.causal_token_len), dtype=bool),
        causal_fast_mask=jnp.zeros((batch, t, config.causal_token_len), dtype=bool),
        seq_step_mask=jnp.ones((batch, t), dtype=bool),
        seq_block_boundary=jnp.zeros((batch, t), dtype=bool),
    )


def _frame_obs(seq_obs, k: int):
    """The single-frame (inference-style) observation of step k."""
    return _model.Observation(
        images={name: v[:, k] for name, v in seq_obs.images.items()},
        image_masks={name: v[:, k] for name, v in seq_obs.image_masks.items()},
        state=seq_obs.state[:, k],
        tokenized_prompt=seq_obs.tokenized_prompt[:, k],
        tokenized_prompt_mask=seq_obs.tokenized_prompt_mask[:, k],
        token_ar_mask=seq_obs.token_ar_mask[:, k],
        token_loss_mask=seq_obs.token_loss_mask[:, k],
        token_fast_mask=seq_obs.token_fast_mask[:, k],
    )


def _step_logits(model, state, frame_obs, causal, causal_len):
    """Training-side CE logits of ONE step, recomputed independently of the scan (python code
    mirroring `_compute_sequence_loss`'s step body). Returns the final memory-token
    representation so the caller can thread the configured write. Used as the equivalence oracle."""
    observation = _model.preprocess_observation(None, frame_obs, train=False)
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
    h_k = hidden[0][model.memory_layer][:, :mem_len].astype(jnp.float32)

    retrieved = model.memory.read(state, h_k)
    mem_tokens = (model.memory_gate.value * retrieved).astype(prefix_tokens.dtype)
    causal_tokens, causal_mask = causal
    causal_emb = model.PaliGemma.llm(causal_tokens, method="embed")
    ext_tokens = jnp.concatenate([mem_tokens, causal_emb], axis=1)
    mem_rows = make_memory_step_mask(prefix_mask, prefix_ar_mask, mem_len, causal_len)
    tri = jnp.tril(jnp.ones((causal_len, causal_len), dtype=bool))
    causal_rows = jnp.concatenate(
        [
            jnp.broadcast_to(prefix_mask[:, None], (batch, causal_len, prefix_len)),
            jnp.ones((batch, causal_len, mem_len), dtype=bool),
            tri[None] & causal_mask[:, None, :],
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
    mem_out, causal_out = ext_out[:, :mem_len], ext_out[:, mem_len:]
    ce_hidden = jnp.concatenate([mem_out[:, -1:], causal_out[:, :-1]], axis=1)
    return (
        model.PaliGemma.llm(ce_hidden, method="decode").astype(jnp.float32),
        h_k,
        mem_out.astype(jnp.float32),
    )


def check_equivalence(write_source: pi0_config.MemoryWriteSource) -> None:
    config, model = _dummy_setup(seq_steps=3, write_source=write_source)
    keys = jax.random.split(jax.random.key(2), 4)
    seq = _seq_obs(config, keys[0])
    causal_len = config.causal_token_len

    # inference side: thread sample_with_memory over the steps (read -> decode -> write)
    state = model.memory.init_state(1)
    gen_tokens, gen_masks, inference_states = [], [], []
    for k in range(3):
        _, state, aux = model.sample_with_memory(
            keys[1], _frame_obs(seq, k), state, stop_token=7, max_decode_steps=6, num_steps=2
        )
        gen_tokens.append(aux["tokens"])
        gen_masks.append(aux["token_mask"])
        inference_states.append(state)
    n_gen = [int(jnp.sum(m)) for m in gen_masks]
    assert sum(n_gen) >= 4, f"want multi-token generations for a meaningful test, got {n_gen}"

    def to_causal(toks, mask):
        buf = jnp.zeros((1, causal_len), dtype=jnp.int32)
        msk = jnp.zeros((1, causal_len), dtype=bool)
        n = toks.shape[1]
        return buf.at[:, :n].set(toks), msk.at[:, :n].set(mask)

    causal = [to_causal(t, m) for t, m in zip(gen_tokens, gen_masks, strict=True)]

    # oracle side: independent per-step recompute, threading the write exactly like inference
    state = model.memory.init_state(1)
    ce_ref = []
    for k in range(3):
        logits, h_k, c_k = _step_logits(model, state, _frame_obs(seq, k), causal[k], causal_len)
        pred = jnp.argmax(logits, axis=-1)
        for i in range(n_gen[k]):
            assert int(pred[0, i]) == int(gen_tokens[k][0, i]), (
                f"train/inference divergence at step {k} position {i}: "
                f"train argmax {int(pred[0, i])} vs generated {int(gen_tokens[k][0, i])}"
            )
        logp = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), causal[k][0][..., None], axis=-1)[..., 0]
        ce_ref.append(float((-jnp.sum(logp * causal[k][1], axis=-1) / jnp.clip(jnp.sum(causal[k][1], -1), 1))[0]))
        write = model._select_memory_write_source(h_k, c_k)  # noqa: SLF001
        state, _ = model.memory.write(state, write)
        state_differences = []
        for (path, actual), expected in zip(
            jax.tree_util.tree_leaves_with_path(state),
            jax.tree.leaves(inference_states[k]),
            strict=True,
        ):
            actual_host, expected_host = np.asarray(actual), np.asarray(expected)
            state_differences.append((jax.tree_util.keystr(path), float(np.max(np.abs(actual_host - expected_host)))))
            # Inference evaluates the memory rows alone, whereas the training oracle evaluates
            # the same masked rows in a wider [memory|causal] GEMM. CUDA may choose a different
            # reduction/fusion kernel for those shapes, so state leaves can differ by a few f32
            # ULPs even though the masks, argmax tokens and semantics are identical.
            np.testing.assert_allclose(actual_host, expected_host, rtol=2e-5, atol=5e-6)
        worst_path, worst_difference = max(state_differences, key=lambda item: item[1])
        print(f"  step {k}: max state |delta|={worst_difference:.3e} at {worst_path}")

    # the real loss must tie out with the oracle's mean CE
    train_obs = seq.replace(
        tokenized_causal=jnp.stack([c[0] for c in causal], axis=1),
        tokenized_causal_mask=jnp.stack([c[1] for c in causal], axis=1),
    )
    actions = jnp.zeros((1, 3, config.action_horizon, config.action_dim))
    losses = model.compute_loss(jax.random.key(9), train_obs, actions, train=False)
    np.testing.assert_allclose(float(losses["ce"][0]), np.mean(ce_ref), rtol=1e-5)
    print(
        f"[OK] train == inference per step ({write_source}): argmax reproduces {sum(n_gen)} generated tokens "
        f"over 3 steps; memory states match; ce ties out ({np.mean(ce_ref):.4f})"
    )


def check_causal_insulation(write_source: pi0_config.MemoryWriteSource) -> None:
    """Changing teacher-forced labels must not change the memory-token output or write."""
    config, model = _dummy_setup(seq_steps=1, write_source=write_source)
    seq = _seq_obs(config, jax.random.key(12))
    frame = _frame_obs(seq, 0)
    state = model.memory.init_state(1)
    mask = jnp.ones((1, config.causal_token_len), dtype=bool)
    causal_a = (jnp.arange(config.causal_token_len, dtype=jnp.int32)[None] + 3, mask)
    causal_b = (jnp.arange(config.causal_token_len, dtype=jnp.int32)[None] + 103, mask)

    _, h_a, c_a = _step_logits(model, state, frame, causal_a, config.causal_token_len)
    _, h_b, c_b = _step_logits(model, state, frame, causal_b, config.causal_token_len)
    np.testing.assert_array_equal(h_a, h_b)
    np.testing.assert_array_equal(c_a, c_b)
    write_a = model._select_memory_write_source(h_a, c_a)  # noqa: SLF001
    write_b = model._select_memory_write_source(h_b, c_b)  # noqa: SLF001
    state_a, _ = model.memory.write(state, write_a)
    state_b, _ = model.memory.write(state, write_b)
    jax.tree.map(np.testing.assert_array_equal, state_a, state_b)
    print(f"[OK] causal/FAST labels cannot affect {write_source} memory writes")


def check_loss_and_grads(write_source: pi0_config.MemoryWriteSource) -> None:
    config, model = _dummy_setup(seq_steps=4, write_source=write_source)
    keys = jax.random.split(jax.random.key(3), 3)
    seq = _seq_obs(config, keys[0])
    causal = jnp.zeros((1, 4, config.causal_token_len), dtype=jnp.int32)
    causal_mask = jnp.zeros((1, 4, config.causal_token_len), dtype=bool)
    causal = causal.at[:, :, :3].set(jnp.asarray([11, 12, 7]))
    causal_mask = causal_mask.at[:, :, :3].set(True)
    train_obs = seq.replace(tokenized_causal=causal, tokenized_causal_mask=causal_mask)
    actions = jax.random.normal(keys[1], (1, 4, config.action_horizon, config.action_dim)) * 0.1

    losses = model.compute_loss(keys[2], train_obs, actions, train=False)
    assert set(losses) == {"flow", "ce"}
    assert losses["flow"].shape == (1,)
    assert losses["ce"].shape == (1,)
    assert bool(jnp.isfinite(losses["flow"][0]))
    assert bool(jnp.isfinite(losses["ce"][0]))
    print(f"[OK] sequence loss runs: flow {float(losses['flow'][0]):.4f}, ce {float(losses['ce'][0]):.4f}")

    # padded steps must be true no-ops: scrambling an invalid step's inputs changes nothing
    step_mask = jnp.asarray([[True, True, False, True]])
    obs_a = train_obs.replace(seq_step_mask=step_mask)
    scrambled = {
        k: v.at[:, 2].set(jax.random.uniform(keys[1], v.shape[2:], minval=-1, maxval=1))
        for k, v in obs_a.images.items()
    }
    obs_b = obs_a.replace(images=scrambled)
    la = model.compute_loss(keys[2], obs_a, actions, train=False)
    lb = model.compute_loss(keys[2], obs_b, actions, train=False)
    np.testing.assert_allclose(np.asarray(la["ce"]), np.asarray(lb["ce"]), atol=1e-6)
    np.testing.assert_allclose(np.asarray(la["flow"]), np.asarray(lb["flow"]), atol=1e-6)
    print("[OK] padded steps are fully ignored (write no-op + loss masked)")

    # gradient routing (m0 included: full in-block BPTT gives it a direct path)
    graphdef, params = nnx.split(model)

    def loss_of(params, which):
        m = nnx.merge(graphdef, params)
        return jnp.mean(m.compute_loss(keys[2], train_obs, actions, train=False)[which])

    for which, must_have, must_not_have in (
        ("flow", ["_1", "action_out_proj"], ["memory"]),
        (
            "ce",
            ["memory_gate", "'m0'", "'w_k'", "'w_v'", "'w_q'"],
            ["_1", "action_out_proj", "action_in_proj", "time_mlp"],
        ),
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
    print("[OK] gradient routing: flow -> action expert only; ce -> VLM + memory (incl. m0) + gate")


def check_block_fence(write_source: pi0_config.MemoryWriteSource) -> None:
    # batch of 2 identical sequences, CE only at the LAST step; sample 0 has a fence at step 2,
    # sample 1 has none. The last step's loss must gradient the earlier steps' images only up
    # to its own block: sample 0 -> steps {2, 3}, sample 1 -> all steps.
    config, model = _dummy_setup(seq_steps=4, write_source=write_source)
    keys = jax.random.split(jax.random.key(4), 3)

    # fresh-model gotcha: the zero-init SigLIP head multiplicatively blocks gradients to input
    # images; perturb it so the image-gradient probes can see anything at all
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

    seq = _seq_obs(config, keys[0], batch=2)
    causal = jnp.zeros((2, 4, config.causal_token_len), dtype=jnp.int32).at[:, 3, :3].set(jnp.asarray([11, 12, 7]))
    causal_mask = jnp.zeros((2, 4, config.causal_token_len), dtype=bool).at[:, 3, :3].set(True)
    boundary = jnp.zeros((2, 4), dtype=bool).at[0, 2].set(True)
    train_obs = seq.replace(tokenized_causal=causal, tokenized_causal_mask=causal_mask, seq_block_boundary=boundary)
    actions = jnp.zeros((2, 4, config.action_horizon, config.action_dim))
    graphdef, params = nnx.split(model)

    def ce_of_images(images):
        m = nnx.merge(graphdef, params)
        return jnp.sum(m.compute_loss(keys[1], train_obs.replace(images=images), actions, train=False)["ce"])

    g = jax.grad(ce_of_images)(train_obs.images)
    norms = np.zeros((2, 4))
    for v in jax.tree.leaves(g):
        norms += np.asarray(jnp.sqrt(jnp.sum(jnp.square(v), axis=tuple(range(2, v.ndim)))))
    assert norms[0, 0] == 0, f"fenced sample: pre-boundary steps got gradient {norms[0]}"
    assert norms[0, 1] == 0, f"fenced sample: pre-boundary steps got gradient {norms[0]}"
    assert norms[0, 2] > 0, f"fenced sample: in-block steps got no gradient {norms[0]}"
    assert norms[0, 3] > 0, f"fenced sample: in-block steps got no gradient {norms[0]}"
    assert all(norms[1, k] > 0 for k in range(4)), f"unfenced sample: some step got no gradient {norms[1]}"
    print(f"[OK] block fence (per-sample): fenced {np.round(norms[0], 4)} vs unfenced {np.round(norms[1], 4)}")

    # the fence must be invisible to the forward pass: state content crosses it untouched
    no_fence = model.compute_loss(
        keys[1], train_obs.replace(seq_block_boundary=jnp.zeros((2, 4), bool)), actions, train=False
    )
    with_fence = model.compute_loss(keys[1], train_obs, actions, train=False)
    np.testing.assert_array_equal(np.asarray(no_fence["ce"]), np.asarray(with_fence["ce"]))
    np.testing.assert_array_equal(np.asarray(no_fence["flow"]), np.asarray(with_fence["flow"]))
    print("[OK] fences change no forward value: losses bit-identical with fences on/off")


def check_probes(write_source: pi0_config.MemoryWriteSource) -> None:
    config, model = _dummy_setup(seq_steps=4, block_steps=2, probe_weight=0.5, write_source=write_source)
    keys = jax.random.split(jax.random.key(5), 3)
    seq = _seq_obs(config, keys[0])
    causal = jnp.zeros((1, 4, config.causal_token_len), dtype=jnp.int32).at[:, :, :3].set(jnp.asarray([11, 12, 7]))
    causal_mask = jnp.zeros((1, 4, config.causal_token_len), dtype=bool).at[:, :, :3].set(True)
    base = seq.replace(tokenized_causal=causal, tokenized_causal_mask=causal_mask)
    actions = jnp.zeros((1, 4, config.action_horizon, config.action_dim))

    labels = jnp.ones((1, 4), dtype=jnp.int32)
    probe_mask = jnp.asarray([[False, True, True, True]])  # step 0 pre-reveal
    probe_visible = jnp.asarray([[False, True, False, False]])
    quiz_obs = base.replace(seq_probe_labels=labels, seq_probe_mask=probe_mask, seq_probe_visible=probe_visible)

    losses = model.compute_loss(keys[1], quiz_obs, actions, train=False)
    assert float(losses["probe_count"][0]) == 3, losses["probe_count"]
    assert float(losses["probe_count_visible"][0]) == 1, losses["probe_count_visible"]
    assert losses["probe_correct_grid"].shape == (1, 4)
    np.testing.assert_array_equal(np.asarray(losses["probe_active_grid"]), [[0.0, 1.0, 1.0, 1.0]])
    print("[OK] probe schedule: quizzes at every quizzable step (3 live, 1 visible)")

    plain = model.compute_loss(keys[1], base, actions, train=False)
    np.testing.assert_array_equal(np.asarray(plain["ce"]), np.asarray(losses["ce"]))
    np.testing.assert_array_equal(np.asarray(plain["flow"]), np.asarray(losses["flow"]))
    assert set(plain) == {"flow", "ce"}
    print("[OK] probe purity: flow/ce bit-identical with quizzes on or off")

    # zero content gate -> pooled read is exactly 0 -> logits are the (zero-init) head bias:
    # every quiz CE is ln 2 and argmax picks class 0
    gate_backup = model.memory_gate.value
    model.memory_gate.value = jnp.zeros_like(gate_backup)
    zg = model.compute_loss(keys[1], quiz_obs, actions, train=False)
    np.testing.assert_allclose(float(zg["probe_ce_sum"][0]), 3 * float(jnp.log(2.0)), rtol=1e-6)
    assert float(zg["probe_correct"][0]) == 0  # labels are 1, argmax of [0, 0] is 0
    zg0 = model.compute_loss(keys[1], quiz_obs.replace(seq_probe_labels=jnp.zeros_like(labels)), actions, train=False)
    assert float(zg0["probe_correct"][0]) == 3
    model.memory_gate.value = gate_backup
    print("[OK] probe determinism at zero gate: ce = n*ln2, correctness follows the label")

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

    stride, ah, t = 10, 5, 4
    rng = np.random.default_rng(0)
    build = _transforms.BuildMemorySequence(stride=stride, action_horizon=ah, block_steps=2)
    item = {
        "observation/image": rng.random((t, 3, 48, 64), dtype=np.float32),
        "observation/left_wrist_image": rng.random((t, 3, 48, 64), dtype=np.float32),
        "observation/right_wrist_image": rng.random((t, 3, 48, 64), dtype=np.float32),
        "observation/state": rng.random((t, 14), dtype=np.float32),
        "actions": rng.random((t * ah, 14), dtype=np.float32),
        "frame_index": 30,
        "index": 1030,
        "episode_length": 55,
        "quiz_side": np.int32(1),
        "reveal_frame": np.int32(35),
        "close_frame": np.int32(48),
    }
    out = build(dict(item))
    assert out["observation/image"].shape == (t, 48, 64, 3)
    assert out["observation/image"].dtype == np.uint8
    assert out["actions"].shape == (t, ah, 14)
    # step frames 30, 40, 50, 60 with episode length 55 -> the last step is padding
    np.testing.assert_array_equal(out["seq_step_mask"], [True, True, True, False])
    assert not out["seq_block_boundary"][0]
    assert 1 <= int(out["seq_block_boundary"].sum()) <= 2  # block 2 over 4 steps, random shift
    # reveal 35 inside the sequence (>= base 30): quizzes at frames 40, 50; visible < 48 -> 40
    np.testing.assert_array_equal(out["seq_probe_mask"], [False, True, True, False])
    np.testing.assert_array_equal(out["seq_probe_visible"], [False, True, False, False])
    np.testing.assert_array_equal(out["seq_probe_labels"], np.ones(t, dtype=np.int32))

    # a slice starting AFTER the reveal never wrote it -> no quizzes at all
    out = build(dict(item, frame_index=40, index=1040))
    assert not out["seq_probe_mask"].any(), "slice missing the reveal must not be quizzed"
    # unlabeled episode -> no quizzes
    out = build(dict(item, quiz_side=np.int32(-1)))
    assert not out["seq_probe_mask"].any()
    # inference item (no frame_index) passes through untouched
    passthrough = {"observation/state": np.zeros(14), "anything": 3}
    assert build(dict(passthrough)) == passthrough
    print("[OK] BuildMemorySequence: shapes, step mask, fence shift, quiz dead-zone, passthrough")

    info = _transforms.MemoryEpisodeInfo(
        episode_length=np.asarray([100, 200], dtype=np.int32),
        episode_side=np.asarray([0, 1], dtype=np.int32),
        episode_reveal=np.asarray([300, 250], dtype=np.int32),
        episode_close=np.asarray([450, 400], dtype=np.int32),
    )
    tagged = info({"episode_index": np.int64(1)})
    assert (int(tagged["episode_length"]), int(tagged["quiz_side"]), int(tagged["reveal_frame"])) == (200, 1, 250)
    print("[OK] MemoryEpisodeInfo: per-episode length/side/reveal/close attached")

    subtask_tf = _transforms.SubtaskFromLeRobotTask({0: "observe bins", 1: "open left bin"})
    seq_sub = subtask_tf({"task_index": np.asarray([0, 0, 1])})["subtask"]
    assert seq_sub == ["observe bins", "observe bins", "open left bin"]
    single = subtask_tf({"task_index": np.asarray([1])})["subtask"]
    assert single == "open left bin"
    print("[OK] SubtaskFromLeRobotTask: per-step subtask list / single-frame string")

    tok = _tokenizer.FASTSubtaskTokenizer(200)
    tokenize = _transforms.TokenizeMemorySubtaskInputs(tok, causal_len=150)
    state = rng.random((3, 14), dtype=np.float32) * 2 - 1
    # smooth action chunks: white noise makes the FAST tokenizer blow past any budget
    time = np.linspace(0, 1, 50)[:, None]
    smooth = (0.3 * np.sin(2 * np.pi * (time + np.linspace(0, 1, 14)[None]))).astype(np.float32)
    actions = np.stack([smooth, smooth * 0.5, smooth * 0.2])
    t_out = tokenize(
        {
            "state": state,
            "actions": actions,
            "prompt": "find the bin with banana",
            "subtask": ["observe bins", "observe bins", "open left bin"],
        }
    )
    assert t_out["tokenized_prompt"].shape == (3, 200)
    assert not t_out["token_ar_mask"].any(), "context must be pure ar=0"
    assert t_out["tokenized_causal"].shape == (3, 150)
    n_causal = t_out["tokenized_causal_mask"].sum(-1)
    assert (n_causal > 0).all()
    assert (n_causal < 150).all()
    for k in range(3):
        first_fast = int(np.argmax(t_out["causal_fast_mask"][k]))
        assert int(t_out["tokenized_causal"][k, first_fast - 1]) == 108, "subtask terminator '\\n' must precede FAST"

    infer_out = tokenize({"state": state[0], "prompt": "find the bin with banana"})
    assert "tokenized_causal" not in infer_out
    assert infer_out["tokenized_prompt"].shape == (200,)
    print(f"[OK] TokenizeMemorySubtaskInputs: per-step causal segments ({list(n_causal)} tokens), inference mode")


def check_real(args: Args) -> None:
    import time

    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader

    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    config = _config.get_config(args.config)
    if config.model.predict_with_memory and config.model.memory_probe_weight == 0:
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(config.model, memory_probe_diagnostic=True),
        )
    config = dataclasses.replace(config, batch_size=args.batch_size, num_workers=0, exp_name="check")
    loader = _data_loader.create_data_loader(config, shuffle=True, num_batches=1)
    observation, actions = next(iter(loader))
    print(
        f"batch loaded: images {next(iter(observation.images.values())).shape}, "
        f"causal {observation.tokenized_causal.shape}, actions {actions.shape}, "
        f"valid steps/sample {np.asarray(jnp.sum(observation.seq_step_mask, -1))}, "
        f"quizzes/sample {np.asarray(jnp.sum(observation.seq_probe_mask, -1)) if observation.seq_probe_mask is not None else '-'}"
    )

    # graft the checkpoint into the memory model (missing = fresh memory params)
    model = config.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    loaded = _config.weight_loaders.PartialCheckpointWeightLoader(args.ckpt).load(state.to_pure_dict())
    state.replace_by_pure_dict(loaded)
    model = nnx.merge(graphdef, state)
    print(f"grafted {args.ckpt} | write source {config.model.memory_write_source} | batch size {args.batch_size}")

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
            return jnp.mean(losses["flow"]) + jnp.mean(losses["ce"])

        return jax.grad(loss_of)(params)

    t0 = time.perf_counter()
    grads = jax.block_until_ready(grads_of(params, observation, actions))
    print(f"first grads (incl. compile): {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    grads = jax.block_until_ready(grads_of(params, observation, actions))
    print(f"steady grads: {time.perf_counter() - t0:.2f}s")
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
    probe_total = sum(v for k, v in norms.items() if "probe_head" in k)
    print(f"grad norm [detached diagnostic probe head]: {probe_total:.4f}")
    assert probe_total == 0, "detached diagnostic probe leaked into the backward graph"
    assert all(np.isfinite(v) for v in norms.values())
    print("[OK] real batch: losses finite; main gradients unchanged and diagnostic probe detached")


def main(args: Args) -> None:
    if args.real:
        logging.basicConfig(level=logging.INFO, force=True)
    check_transforms()
    check_equivalence(args.write_source)
    check_causal_insulation(args.write_source)
    check_loss_and_grads(args.write_source)
    check_block_fence(args.write_source)
    check_probes(args.write_source)
    if args.real:
        check_real(args)
    print("\nALL OK")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
