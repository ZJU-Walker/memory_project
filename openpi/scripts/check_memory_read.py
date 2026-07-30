"""Checks for Part H: memory-conditioned subtask decoding (`Pi0.sample_with_memory`).

Stage 1 (default) is CPU-safe and torch-free -- runs anywhere, no GPU / checkpoint / dataset:
    uv run python scripts/check_memory_read.py

Stage 2 (--real) needs a GPU, the checkpoint, and the LeRobot dataset (run on iris-hgx-2):
    CUDA_VISIBLE_DEVICES=<free> uv run python scripts/check_memory_read.py --real
It grafts the checkpoint into a memory-enabled model (fresh memory params, zero gate), replays
one training episode threading the memory state, and reports: baseline-vs-memory-path subtask
agreement (characterization -- the generated-token positions moved, so differences are expected
until phase-I training), both subtask timelines, the surprise curve, and timing.

Edit the constants below (or override on the CLI).
"""

import dataclasses
import pathlib
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import memory
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.pi0 import make_memory_step_mask

CKPT = pathlib.Path("/iris/u/kewalk/memory_project/openpi/checkpoints/pi05_yam/pi05_yam_KI/10000")
EPISODE = 0
STRIDE = 10  # evaluate every stride-th frame of the episode
MAX_DECODE_STEPS = 10


@dataclasses.dataclass
class Args:
    real: bool = False  # also run the real-checkpoint episode replay (GPU + dataset)
    ckpt_dir: pathlib.Path = CKPT
    config: str = "pi05_yam"
    episode: int = EPISODE
    stride: int = STRIDE
    layer: int | None = None  # override the config's memory_layer


def _tree_dist(a, b) -> float:
    return float(
        jnp.sqrt(sum(jnp.sum(jnp.square(x - y)) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True)))
    )


def check_mask() -> None:
    # synthetic layout: 3 image slots + 6 text slots (2 context ar0, 2 causal ar1, 2 padding)
    prefix_mask = jnp.array([[True, True, True, True, True, True, True, False, False]])
    prefix_ar = jnp.array([[0, 0, 0, 0, 0, 1, 1, 0, 0]])
    mask = make_memory_step_mask(prefix_mask, prefix_ar, mem_len=2, causal_len=3)
    assert mask.shape == (1, 2, 9 + 2 + 3)
    expected_row = [True] * 5 + [False, False] + [False, False] + [True, True] + [False] * 3
    np.testing.assert_array_equal(np.asarray(mask[0, 0]), expected_row)
    np.testing.assert_array_equal(np.asarray(mask[0, 1]), expected_row)
    print("[OK] memory step mask: attends to ar0 context + itself, never to causal/pad slots")


def check_dummy() -> None:
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
    )
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(1)
    # Fresh-model gotcha (cousin of the adaLN-zero one): the SigLIP head is zero-init, so a
    # fresh model's image tokens are exactly 0 -- and with fake_obs's all-ones (ar=1) text they
    # attend only to each other and stay 0 through every layer, making h/v/surprise all 0. Use
    # an inference-style ar=0 prompt so the image rows pick up signal from the text.
    obs = obs.replace(token_ar_mask=jnp.zeros_like(obs.token_ar_mask))
    h_probe = model.extract_topcam_hidden(obs)[config.memory_layer]
    assert float(jnp.linalg.norm(h_probe)) > 0

    # strict switch: without the flag the memory params are not even constructed, and the
    # baseline fused path still runs
    config_off = dataclasses.replace(config, predict_with_memory=False)
    model_off = config_off.create(jax.random.key(0))
    assert not hasattr(model_off, "memory")
    actions_off, _ = model_off.sample_subtask_and_actions(
        jax.random.key(1), obs, stop_token=7, max_decode_steps=4, num_steps=2
    )
    assert actions_off.shape == (1, config.action_horizon, config.action_dim)
    print("[OK] flag off: no memory params, baseline fused path intact")

    # zero-init gate + fresh memory => the injected token content is exactly zero
    assert bool(jnp.all(model.memory_gate.value == 0))
    state0 = model.memory.init_state(1)
    probe = jax.random.normal(jax.random.key(2), (1, 4, mem_cfg.d_input))
    assert float(jnp.max(jnp.abs(model.memory.read(state0, probe)))) == 0.0
    print("[OK] zero gate + fresh memory: injected content is exactly 0")

    # end-to-end: runs, is deterministic, threads state, and repeated observation gets less
    # surprising through the full pipeline
    noise = jax.random.normal(jax.random.key(3), (1, config.action_horizon, config.action_dim))
    t0 = time.perf_counter()
    actions1, state1, aux1 = model.sample_with_memory(
        jax.random.key(4), obs, state0, stop_token=7, max_decode_steps=4, num_steps=2, noise=noise
    )
    print(f"    sample_with_memory (dummy, cpu): {time.perf_counter() - t0:.1f}s")
    assert actions1.shape == (1, config.action_horizon, config.action_dim)
    assert bool(jnp.all(jnp.isfinite(actions1)))
    assert aux1["tokens"].shape == (1, config.causal_token_len)
    n_gen = int(jnp.sum(aux1["token_mask"]))
    assert 1 <= n_gen <= 4
    left_aligned = jnp.arange(config.causal_token_len)[None] < n_gen
    assert bool(jnp.all(aux1["token_mask"] == left_aligned))
    assert 0.9 < float(aux1["surprise"][0]) < 1.1  # fresh memory, unit-norm values
    assert _tree_dist(state1, state0) > 0

    actions1b, _, aux1b = model.sample_with_memory(
        jax.random.key(4), obs, state0, stop_token=7, max_decode_steps=4, num_steps=2, noise=noise
    )
    assert bool(jnp.all(actions1 == actions1b))
    assert bool(jnp.all(aux1["tokens"] == aux1b["tokens"]))
    print("[OK] sample_with_memory: shapes, finite, left-aligned tokens, deterministic")

    _, state2, aux2 = model.sample_with_memory(
        jax.random.key(5), obs, state1, stop_token=7, max_decode_steps=4, num_steps=2, noise=noise
    )
    assert float(aux2["surprise"][0]) < float(aux1["surprise"][0])
    assert _tree_dist(state2, state1) > 0
    print(
        f"[OK] memory threads through calls: surprise {float(aux1['surprise'][0]):.3f} -> "
        f"{float(aux2['surprise'][0]):.3f} on the repeated observation"
    )


def check_real(args: Args) -> None:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    import openpi.models.tokenizer as _tokenizer
    import openpi.training.checkpoints as _checkpoints
    import openpi.training.config as _config
    import openpi.transforms as _transforms

    cfg = _config.get_config(args.config)
    model_cfg = dataclasses.replace(
        cfg.model,
        predict_with_memory=True,
        **({"memory_layer": args.layer} if args.layer is not None else {}),
    )
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats = _checkpoints.load_norm_stats(args.ckpt_dir / "assets", data_config.asset_id)
    input_transforms = list(data_config.data_transforms.inputs)
    normalize = _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
    model_transforms = list(data_config.model_transforms.inputs)

    pg = _tokenizer.FASTSubtaskTokenizer(cfg.model.max_token_len)._paligemma_tokenizer  # noqa: SLF001
    stop_token = int(pg.encode("placeholder subtask\n")[-1])

    # graft the checkpoint into a memory-enabled model: checkpoint wins on the intersection,
    # the memory params (absent from the checkpoint) keep their fresh init
    def merge(dst: dict, src: dict) -> dict:
        for k, v in src.items():
            dst[k] = merge(dst[k], v) if isinstance(v, dict) else v
        return dst

    model = model_cfg.create(jax.random.key(0))
    params = _model.restore_params(args.ckpt_dir / "params", dtype=jnp.bfloat16)
    graphdef, state = nnx.split(model)
    pure = state.to_pure_dict()
    n_total = len(jax.tree.leaves(pure))
    n_graft = len(jax.tree.leaves(params))
    state.replace_by_pure_dict(merge(pure, params))
    model = nnx.merge(graphdef, state)
    graphdef, state = nnx.split(model)
    print(
        f"loaded {args.ckpt_dir}: grafted {n_graft}/{n_total} param tensors "
        f"(the rest are fresh memory params) | memory_layer={model_cfg.memory_layer}",
        flush=True,
    )

    infer_base = jax.jit(
        lambda s, o: nnx.merge(graphdef, s).sample_subtask(o, stop_token=stop_token, max_decode_steps=MAX_DECODE_STEPS)
    )
    infer_mem = jax.jit(
        lambda s, ms, r, o: nnx.merge(graphdef, s).sample_with_memory(
            r, o, ms, stop_token=stop_token, max_decode_steps=MAX_DECODE_STEPS
        )
    )

    ds = lerobot_dataset.LeRobotDataset(data_config.repo_id)
    ep_from = int(ds.episode_data_index["from"][args.episode])
    ep_to = int(ds.episode_data_index["to"][args.episode])
    ts = list(range(ep_from, ep_to, args.stride))
    tasks = ds.meta.tasks
    print(f"episode {args.episode}: frames {ep_from}..{ep_to} -> {len(ts)} at stride {args.stride}", flush=True)

    def build_item(frame: dict) -> dict:
        item = {
            "observation/image": np.asarray(frame["image"]),
            "observation/left_wrist_image": np.asarray(frame["left_wrist_image"]),
            "observation/right_wrist_image": np.asarray(frame["right_wrist_image"]),
            "observation/state": np.asarray(frame["state"]),
        }
        for tf in input_transforms:
            item = tf(item)
        item = normalize(item)
        for tf in model_transforms:
            item = tf(item)
        return item

    mem_module = nnx.merge(graphdef, state).memory
    mem_state = mem_module.init_state(1)
    labels, base_preds, mem_preds, surprises = [], [], [], []
    t_base = t_mem = 0.0
    t_start = time.perf_counter()
    for i, t in enumerate(ts):
        frame = ds[int(t)]
        labels.append(str(tasks[int(frame["task_index"])]))
        batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *[build_item(frame)])
        obs = _model.Observation.from_dict(batch)

        t0 = time.perf_counter()
        out_obs = infer_base(state, obs)
        n0 = int(np.asarray(batch["tokenized_prompt_mask"]).sum())
        out_n = int(np.asarray(out_obs.tokenized_prompt_mask)[0].sum())
        base_preds.append(pg.decode(np.asarray(out_obs.tokenized_prompt)[0, n0:out_n].tolist()).strip())
        t_base += time.perf_counter() - t0

        t0 = time.perf_counter()
        _, mem_state, aux = infer_mem(state, mem_state, jax.random.fold_in(jax.random.key(0), i), obs)
        tokens = np.asarray(aux["tokens"])[0][np.asarray(aux["token_mask"])[0]]
        mem_preds.append(pg.decode(tokens.tolist()).strip())
        surprises.append(float(aux["surprise"][0]))
        t_mem += time.perf_counter() - t0

        if i == 0:
            print(f"first frame in {time.perf_counter() - t_start:.1f}s (incl. compiles)", flush=True)

    def runs(seq: list[str]) -> str:
        out, start = [], 0
        for i in range(1, len(seq) + 1):
            if i == len(seq) or seq[i] != seq[start]:
                out.append(f"{start}-{i - 1} {seq[start]!r}")
                start = i
        return " | ".join(out)

    agree = float(np.mean([b == m for b, m in zip(base_preds, mem_preds, strict=True)]))
    base_acc = float(np.mean([b == g for b, g in zip(base_preds, labels, strict=True)]))
    mem_acc = float(np.mean([m == g for m, g in zip(mem_preds, labels, strict=True)]))
    print(f"\ngt      : {runs(labels)}")
    print(f"baseline: {runs(base_preds)}")
    print(f"memory  : {runs(mem_preds)}")
    print(
        f"\nagreement memory-vs-baseline: {agree:.1%} | accuracy baseline {base_acc:.1%}, "
        f"memory-path {mem_acc:.1%} (expected to lag until phase-I training)"
    )
    surprises = np.asarray(surprises)
    print(f"surprise: first {surprises[0]:.3f} min {surprises.min():.3f} last {surprises[-1]:.3f}")
    print(f"timing per frame: baseline {t_base / len(ts) * 1e3:.0f} ms, memory path {t_mem / len(ts) * 1e3:.0f} ms")

    out_dir = pathlib.Path(__file__).parent / "eval_results"
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.ckpt_dir.parent.name}_{args.ckpt_dir.name}_ep{args.episode}_L{model_cfg.memory_layer}"
    np.savez(
        out_dir / f"memory_read_{tag}.npz",
        frames=np.asarray(ts),
        labels=np.asarray(labels),
        base_preds=np.asarray(base_preds),
        mem_preds=np.asarray(mem_preds),
        surprise=surprises,
    )
    print(f"saved {out_dir / f'memory_read_{tag}.npz'}")


def main(args: Args) -> None:
    check_mask()
    check_dummy()
    if args.real:
        check_real(args)
    print("\nALL OK")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
