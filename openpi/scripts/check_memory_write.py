"""Checks for the Titans memory module (write path) and the gemma per-layer hidden-state capture.

Stage 1 (default) is CPU-safe and torch-free -- runs anywhere, no GPU / checkpoint / dataset:
    uv run python scripts/check_memory_write.py

Stage 2 (--real) needs a GPU, the checkpoint, and the LeRobot dataset (run on iris-hgx-2):
    CUDA_VISIBLE_DEVICES=<free> uv run python scripts/check_memory_write.py --real
It replays one training episode in time order: extracts the per-layer top-camera hidden states,
prints their norm statistics (the real input scale of the memory), writes them frame-by-frame
into a fresh memory (one per probed layer), and saves a surprise-vs-frame plot with the subtask
boundaries to scripts/eval_results/, plus the raw curves as .npz.

Edit the constants below (or override on the CLI).
"""

import dataclasses
import math
import pathlib
import time

from flax import nnx
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma as _gemma
from openpi.models import memory
from openpi.models import model as _model
from openpi.models import pi0_config

CKPT = pathlib.Path("/iris/u/kewalk/memory_project/openpi/checkpoints/pi05_yam/pi05_yam_KI/10000")
EPISODE = 0  # which training episode to replay
LAYERS = (3, 9, 15, 17)  # gemma blocks to probe (0..17); one fresh memory per layer
STRIDE = 1  # write every stride-th frame
BATCH_SIZE = 16  # frames per extraction forward pass


@dataclasses.dataclass
class Args:
    real: bool = False  # also run the real-checkpoint episode replay (GPU + dataset)
    ckpt_dir: pathlib.Path = CKPT
    config: str = "pi05_yam"
    episode: int = EPISODE
    layers: tuple[int, ...] = LAYERS
    stride: int = STRIDE
    batch_size: int = BATCH_SIZE


def _tree_dist(a, b) -> float:
    return math.sqrt(
        sum(float(jnp.sum(jnp.square(x - y))) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))
    )


def _finite(tree) -> bool:
    return all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(tree))


def _runs(labels: list[str]) -> str:
    """Collapse a per-frame label sequence into 'start-end label | ...' runs."""
    out = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append(f"{start}-{i - 1} {labels[start]!r}")
            start = i
    return " | ".join(out)


def check_memory() -> None:
    cfg = memory.MemoryConfig()
    mem = memory.TitansMemory(cfg, rngs=nnx.Rngs(0))
    keys = jax.random.split(jax.random.key(1), 2)
    h_a = jax.random.normal(keys[0], (1, 256, cfg.d_input)) * 5.0
    h_b = jax.random.normal(keys[1], (1, 256, cfg.d_input)) * 5.0

    # fresh memory: read exactly 0 (zero-init output layer), surprise exactly 1 (unit-norm v)
    state = mem.init_state(1)
    read0 = mem.read(state, h_a)
    assert read0.shape == (1, 256, cfg.d_value)
    assert float(jnp.max(jnp.abs(read0))) == 0.0
    assert abs(float(mem.surprise(state, h_a)[0]) - 1.0) < 1e-3
    print("[OK] fresh memory reads exactly 0, surprise starts at 1.0")

    # keys/values are unit-norm regardless of input scale
    _, k, v = mem._keys_values(h_a * 1000.0)  # noqa: SLF001
    assert float(jnp.max(jnp.abs(jnp.linalg.norm(k, axis=-1) - 1.0))) < 1e-3
    assert float(jnp.max(jnp.abs(jnp.linalg.norm(v, axis=-1) - 1.0))) < 1e-3
    print("[OK] keys/values unit-norm at any input scale")

    # repeated writes memorize the written frame, not unseen content
    st = state
    for _ in range(30):
        st, aux = mem.write(st, h_a)
    s_a, s_b = float(mem.surprise(st, h_a)[0]), float(mem.surprise(st, h_b)[0])
    assert s_a < 0.3, s_a
    assert s_b > 0.7, s_b
    assert _finite(st)
    print(f"[OK] memorization is specific: written A 1.0 -> {s_a:.3f}, unseen B stays {s_b:.3f}")

    # a single write already leaves a usable trace (the robot writes each frame once)
    st1, _ = mem.write(state, h_a)
    s_one = float(mem.surprise(st1, h_a)[0])
    assert s_one < 0.9, s_one
    print(f"[OK] single write retains: A 1.0 -> {s_one:.3f}")

    # the grad clip keeps huge inputs finite
    st_big, _ = mem.write(state, h_a * 1000.0)
    assert _finite(st_big)
    print("[OK] finite at 1000x input scale")

    # batched writes == independent per-sample writes (vmap isolation)
    st_ab, _ = mem.write(mem.init_state(2), jnp.concatenate([h_a, h_b], axis=0))
    st_a, _ = mem.write(mem.init_state(1), h_a)
    st_b, _ = mem.write(mem.init_state(1), h_b)
    jax.tree.map(
        lambda ab, a, b: np.testing.assert_allclose(ab, np.concatenate([a, b]), atol=1e-5),
        st_ab,
        st_a,
        st_b,
    )
    print("[OK] batch writes == independent solo writes")

    # forgetting: alpha ~ 1 (theta, eta ~ 0) wipes the memory in one write
    orig_bias = mem.gate.bias.value
    mem.gate.bias.value = jnp.array([-20.0, -20.0, 20.0])
    wiped, _ = mem.write(st, h_b)
    mem.gate.bias.value = orig_bias
    assert max(float(jnp.max(jnp.abs(x))) for x in jax.tree.leaves(wiped.fast_weights)) < 1e-4
    print("[OK] forgetting gate wipes the memory")

    # momentum: with theta ~ 0 (no new gradient) but eta ~ 0.9, past surprise keeps moving the
    # weights; with eta ~ 0 too, nothing moves
    st1, _ = mem.write(state, h_a)
    mem.gate.bias.value = jnp.array([-20.0, 2.2, -20.0])
    st2, _ = mem.write(st1, h_b)
    moved = _tree_dist(st2.fast_weights, st1.fast_weights)
    mem.gate.bias.value = jnp.array([-20.0, -20.0, -20.0])
    st3, _ = mem.write(st1, h_b)
    frozen = _tree_dist(st3.fast_weights, st1.fast_weights)
    mem.gate.bias.value = orig_bias
    assert moved > 100 * max(frozen, 1e-12), (moved, frozen)
    print(f"[OK] momentum carries surprise: moved {moved:.3g} vs {frozen:.3g} without")

    # outer differentiability: gradients reach every outer param through write chains -- incl.
    # m0 (the old 400m memory runs had a bug where m0 provably received zero gradient)
    graphdef, params = nnx.split(mem)

    def outer_loss(params):
        m = nnx.merge(graphdef, params)
        st = m.init_state(1)
        st, _ = m.write(st, h_a)
        st, _ = m.write(st, h_b)
        return jnp.mean(jnp.square(m.read(st, h_a)))

    grads = jax.grad(outer_loss)(params)
    leaves = jax.tree_util.tree_leaves_with_path(grads)
    assert all(bool(jnp.all(jnp.isfinite(g))) for _, g in leaves)
    norms = {jax.tree_util.keystr(path): float(jnp.linalg.norm(g)) for path, g in leaves}
    for name in ("m0", "w_k", "w_v", "w_q", "gate"):
        total = sum(v for key, v in norms.items() if name in key)
        assert total > 0, f"no outer gradient reaches {name}"
    print("[OK] outer gradients finite and nonzero for m0 / w_k / w_v / w_q / gate")


def check_gemma_capture() -> None:
    config = _gemma.get_config("dummy")
    module = _gemma.Module(configs=[config, config], embed_dtype="float32")
    # gemma.Module defines its own `init` convenience method, which shadows linen's initializer;
    # call the linen one explicitly with the convenience method as the init target.
    variables = nn.Module.init(module, jax.random.key(0), method="init", use_adarms=[False, False])
    b, t = 2, 12
    emb = jax.random.normal(jax.random.key(1), (b, t, config.width)) * 0.3
    positions = jnp.broadcast_to(jnp.arange(t)[None], (b, t))
    mask = jnp.ones((b, t, t), dtype=bool)

    out_ref, cache_ref = module.apply(variables, [emb, None], positions, mask)
    out_cap, cache_cap, hidden = module.apply(variables, [emb, None], positions, mask, return_hidden_states=True)

    assert jnp.array_equal(out_ref[0], out_cap[0])
    assert out_ref[1] is None
    assert out_cap[1] is None
    jax.tree.map(lambda a, c: np.testing.assert_array_equal(a, c), cache_ref, cache_cap)
    print("[OK] capture on/off: main outputs and kv cache bit-identical")

    assert hidden[0].shape == (config.depth, b, t, config.width), hidden[0].shape
    assert hidden[1] is None
    assert not jnp.array_equal(hidden[0][0], hidden[0][-1])
    # the module output is exactly the final norm of the last block's output; on a fresh module
    # the RMSNorm scale is zero-init, so the norm is closed-form
    x = hidden[0][-1].astype(jnp.float32)
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    np.testing.assert_allclose(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)), out_ref[0], atol=1e-5)
    print(f"[OK] hidden states {hidden[0].shape}: entry [-1] reproduces the module output")

    # suffix-style call (action expert only) captures the second expert's stream
    _, _, hidden_s = module.apply(variables, [None, emb], positions, mask, return_hidden_states=True)
    assert hidden_s[0] is None
    assert hidden_s[1].shape == (config.depth, b, t, config.width)
    print("[OK] suffix-only capture works")


def check_pi0_extract() -> None:
    config = pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy", dtype="float32", pi05=True)
    model = config.create(jax.random.key(0))
    obs = config.fake_obs()
    hidden = model.extract_topcam_hidden(obs)
    gemma_cfg = _gemma.get_config("dummy")
    assert hidden.shape == (gemma_cfg.depth, 1, 256, gemma_cfg.width), hidden.shape
    assert _finite(hidden)
    assert float(jnp.max(jnp.abs(hidden))) > 0
    print(f"[OK] extract_topcam_hidden: {hidden.shape} (layers, batch, top-cam tokens, width)")


def check_real(args: Args) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    import matplotlib.pyplot as plt

    import openpi.shared.nnx_utils as nnx_utils
    import openpi.training.checkpoints as _checkpoints
    import openpi.training.config as _config
    import openpi.transforms as _transforms

    cfg = _config.get_config(args.config)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats = _checkpoints.load_norm_stats(args.ckpt_dir / "assets", data_config.asset_id)
    input_transforms = list(data_config.data_transforms.inputs)  # YamInputs, DeltaActions (no-op here)
    normalize = _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
    model_transforms = list(data_config.model_transforms.inputs)  # prompt, resize, tokenize, pad

    ds = lerobot_dataset.LeRobotDataset(data_config.repo_id)
    ep_from = int(ds.episode_data_index["from"][args.episode])
    ep_to = int(ds.episode_data_index["to"][args.episode])
    ts = list(range(ep_from, ep_to, args.stride))
    tasks = ds.meta.tasks
    print(f"episode {args.episode}: frames {ep_from}..{ep_to} -> {len(ts)} at stride {args.stride}", flush=True)

    def build_item(frame: dict) -> dict:
        """The exact inference-time input pipeline (no actions, no subtask -> pure prefix)."""
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

    labels, items = [], []
    for t in ts:
        frame = ds[int(t)]
        labels.append(str(tasks[int(frame["task_index"])]))
        items.append(build_item(frame))
    print(f"subtasks: {_runs(labels)}", flush=True)

    model = cfg.model.load(_model.restore_params(args.ckpt_dir / "params", dtype=jnp.bfloat16))
    graphdef, state = nnx.split(model)
    extract = jax.jit(lambda s, o: nnx.merge(graphdef, s).extract_topcam_hidden(o))
    print(f"loaded {args.ckpt_dir}", flush=True)

    # batched extraction of the selected layers' top-cam hidden states: [T, L, n, d] float16
    sel = jnp.asarray(args.layers)
    t0 = time.perf_counter()
    chunks = []
    for bi in range(0, len(items), args.batch_size):
        chunk = items[bi : bi + args.batch_size]
        pad = args.batch_size - len(chunk)  # pad the tail batch to keep one jit shape
        batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *(chunk + chunk[-1:] * pad))
        hidden = extract(state, _model.Observation.from_dict(batch))  # [depth, B, n, d] bf16
        chunks.append(np.asarray(hidden[sel].astype(jnp.float16))[:, : len(chunk)].transpose(1, 0, 2, 3))
        if bi == 0:
            print(f"first extraction batch in {time.perf_counter() - t0:.1f}s (incl. compile)", flush=True)
    all_h = np.concatenate(chunks, axis=0)
    print(f"extracted hidden states {all_h.shape} in {time.perf_counter() - t0:.1f}s", flush=True)

    # the real input scale of the memory, per layer (calibrates theta / max_grad_norm)
    for li, layer in enumerate(args.layers):
        norms = np.linalg.norm(all_h[:, li].astype(np.float32), axis=-1)
        print(
            f"layer {layer:>2}: token norm mean {norms.mean():8.1f}   "
            f"p5 {np.percentile(norms, 5):8.1f}   p95 {np.percentile(norms, 95):8.1f}"
        )

    # replay the episode into a fresh memory, one per layer (same outer params -> comparable)
    mem = memory.TitansMemory(memory.MemoryConfig(), rngs=nnx.Rngs(0))
    write = nnx_utils.module_jit(mem.write)
    curves: dict[int, np.ndarray] = {}
    for li, layer in enumerate(args.layers):
        st = mem.init_state(1)
        surprises = []
        t1 = time.perf_counter()
        for t in range(all_h.shape[0]):
            st, aux = write(st, jnp.asarray(all_h[t, li][None], dtype=jnp.float32))
            surprises.append(float(aux["surprise"][0]))
        curves[layer] = np.asarray(surprises)
        assert _finite(st), f"non-finite memory state at layer {layer}"
        print(
            f"layer {layer:>2}: {len(surprises)} writes in {time.perf_counter() - t1:.1f}s | "
            f"surprise first {surprises[0]:.3f} min {min(surprises):.3f} last {surprises[-1]:.3f}",
            flush=True,
        )

    # surprise-vs-frame plot with subtask boundaries
    frames = [t - ep_from for t in ts]
    fig, ax = plt.subplots(figsize=(14, 5))
    for layer in args.layers:
        ax.plot(frames, curves[layer], lw=0.9, label=f"layer {layer}")
    prev = labels[0]
    for i in range(1, len(labels)):
        if labels[i] != prev:
            ax.axvline(frames[i], color="k", ls="--", lw=0.8)
            ax.text(
                frames[i], 0.98, f" {labels[i]}", rotation=90, va="top", fontsize=7, transform=ax.get_xaxis_transform()
            )
            prev = labels[i]
    ax.set_xlabel("frame")
    ax.set_ylabel("write surprise (1.0 = unwritten memory)")
    ax.set_title(f"{args.ckpt_dir.parent.name}/{args.ckpt_dir.name} episode {args.episode}: memory write surprise")
    ax.legend()
    fig.tight_layout()

    out_dir = pathlib.Path(__file__).parent / "eval_results"
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.ckpt_dir.parent.name}_{args.ckpt_dir.name}_ep{args.episode}"
    png = out_dir / f"memory_surprise_{tag}.png"
    fig.savefig(png, dpi=140)
    np.savez(
        out_dir / f"memory_surprise_{tag}.npz",
        frames=np.asarray(frames),
        labels=np.asarray(labels),
        **{f"layer_{k}": v for k, v in curves.items()},
    )
    print(f"saved {png}\nsaved {out_dir / f'memory_surprise_{tag}.npz'}")


def main(args: Args) -> None:
    check_memory()
    check_gemma_capture()
    check_pi0_extract()
    if args.real:
        check_real(args)
    print("\nALL OK")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
