"""Extract VLM hidden states for the TOP camera from a pi05 checkpoint, offline.

For every frame of ONE demo (episode) of the LeRobot dataset, runs the pi05 *prefix* forward
pass (SigLIP -> prompt/state embedding -> Gemma-2B VLM) exactly as `Pi0.sample_actions` does
before denoising, and grabs the residual-stream activations of the top camera's 256 patch
tokens at TEST_HIDDEN_LAYER. The action expert never runs.

Relies on the `return_prefix_hiddens` flag added to `gemma.Module.__call__`.
Layer convention: hiddens[k], k in [0, 17] = residual after Gemma block k, pre-final-norm.

Then computes the per-patch surprise e_{t,i} = 1 - cos(h_{t,i}, h_{t-1,i}) for t = 1..T-1,
i = patch 0..255 -> [T-1, 256], and the per-frame score S_t = sum_i softmax(znorm(e_t)/tau)_i
* e_{t,i} -> [T-1] (softmax-weighted patch aggregation; small tau ~ TopK, large tau ~ mean).
Saves to scripts/test_results/: an S_t-vs-time plot (.png) and a composite .mp4 of the
top-cam video with a moving dot marking the current frame's position on the S_t curve.

Usage:
    uv run scripts/extract_hidden.py
"""

import dataclasses
import functools
import logging
import pathlib
import subprocess

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import tqdm
import tyro
from flax import nnx

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader

TEST_HIDDEN_LAYER = 17  # 0..17, hiddens[k] = residual after Gemma block k
DEMO_NUM = 0  # which demo (episode index) of the LeRobot dataset to run
TAU_TEMP = 0.5  # softmax temperature for S_t: small ~ TopK/max, large ~ mean
CKPT = pathlib.Path("checkpoints/pi05_yam/yam_banana_pi05/7000")
NUM_PATCHES = 256  # SigLIP So400m/14 @ 224px -> 16x16 patches; top cam = first image block
BATCH_SIZE = 64


@dataclasses.dataclass
class Args:
    config: str = "pi05_yam"
    ckpt_dir: pathlib.Path = CKPT
    test_hidden_layer: int = TEST_HIDDEN_LAYER
    demo_num: int = DEMO_NUM
    tau_temp: float = TAU_TEMP
    batch_size: int = BATCH_SIZE


def make_probe(model: pi0.Pi0, layer: int):
    """Jitted prefix-only forward -> top-cam activations [B, 256, 2048] f32 at `layer`."""
    graphdef, state = nnx.split(model)

    @jax.jit
    def probe(state, obs: _model.Observation):
        model = nnx.merge(graphdef, state)
        obs = _model.preprocess_observation(None, obs, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
        attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, _, hiddens = model.PaliGemma.llm(
            [prefix_tokens, None], mask=attn_mask, positions=positions, return_prefix_hiddens=True
        )
        return hiddens[layer, :, :NUM_PATCHES, :].astype(jnp.float32)  # [B, 256, 2048]

    return functools.partial(probe, state)


def main(args: Args) -> None:
    cfg = _config.get_config(args.config)

    # Exact training input pipeline, with norm stats from the checkpoint itself.
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats = _checkpoints.load_norm_stats(args.ckpt_dir / "assets", data_config.asset_id)
    data_config = dataclasses.replace(data_config, norm_stats=norm_stats)
    raw_ds = _data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    ds = _data_loader.transform_dataset(raw_ds, data_config)

    start = int(raw_ds.episode_data_index["from"][args.demo_num])
    end = int(raw_ds.episode_data_index["to"][args.demo_num])
    logging.info("Demo %d: frames [%d, %d) -> %d frames", args.demo_num, start, end, end - start)

    model = cfg.model.load(_model.restore_params(args.ckpt_dir / "params", dtype=jnp.bfloat16))
    probe = make_probe(model, args.test_hidden_layer)

    hidden_states, top_frames = [], []
    for i in tqdm.trange(start, end, args.batch_size, unit="batch"):
        frames = [ds[j] for j in range(i, min(i + args.batch_size, end))]
        batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *frames)
        batch.pop("actions", None)
        top_frames.append(batch["image"]["base_0_rgb"])  # uint8 [B, 224, 224, 3], as fed to the model

        hidden_state = np.asarray(probe(_model.Observation.from_dict(batch)))  # [B, 256, 2048]
        if i == start:
            logging.info("hidden_state %s", hidden_state.shape)
        hidden_states.append(hidden_state)

    h = np.concatenate(hidden_states)  # [T, 256, 2048]
    
    # e_{t,i} = 1 - cos(h_{t,i}, h_{t-1,i}), t = 1..T-1: how much patch i's embedding changed since the last frame
    a, b = h[1:], h[:-1]
    e = 1.0 - (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)  # [T-1, 256]

    # S_t = sum_i w_i * e_{t,i} with w = softmax(znorm(e_t) / tau): small tau ~ TopK, large tau ~ mean
    e_norm = (e - e.mean(axis=-1, keepdims=True)) / (e.std(axis=-1, keepdims=True) + 1e-6)
    w = np.exp(e_norm / args.tau_temp)
    w = w / (w.sum(axis=-1, keepdims=True) + 1e-8)
    s = (w * e).sum(axis=-1)  # [T-1]

    # plot S_t vs time and save to scripts/test_results/
    times = np.arange(1, len(s) + 1) / raw_ds.fps  # S_t is the transition into frame t
    out_dir = pathlib.Path(__file__).parent / "test_results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"s_demo{args.demo_num:03d}_layer{args.test_hidden_layer}_tau{args.tau_temp:g}.png"
    plt.figure(figsize=(12, 4))
    plt.plot(times, s, lw=0.8)
    plt.xlabel("time (s)")
    plt.ylabel("S_t")
    plt.title(f"demo {args.demo_num}  layer {args.test_hidden_layer}  tau {args.tau_temp:g}")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    logging.info("saved %s", out)

    # composite mp4: top-cam video sitting exactly over the plot's x-axis, with a time cursor
    frames_u8 = np.concatenate(top_frames)  # [T, 224, 224, 3]
    fig, ax = plt.subplots(figsize=(8.96, 2.56), dpi=100)  # renders to 896 x 256 px
    ax.set_position([112 / 896, 0.24, 672 / 896, 0.70])  # x-axis spans pixels [112, 784): 672 = 3x224
    ax.plot(times, s, lw=0.8)
    ax.set_xlim(0, times[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("S_t")
    fig.canvas.draw()
    plot_img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()  # [256, 896, 3]
    bb = ax.get_window_extent()  # axes bbox in pixels, y measured from bottom
    x0, x1 = round(bb.x0), round(bb.x1)
    plot_h, width = plot_img.shape[:2]
    r0, r1 = plot_h - round(bb.y1), plot_h - round(bb.y0)  # cursor rows: full axes height
    frame_times = np.arange(len(frames_u8)) / raw_ds.fps
    cur = ax.transData.transform(np.column_stack([frame_times, np.zeros_like(frame_times)]))[:, 0]
    cur = np.clip(cur.round().astype(int), x0 + 1, x1 - 2)  # cursor column per frame
    plt.close(fig)

    canvas = np.zeros((672 + plot_h, width, 3), np.uint8)  # 928 x 896, even dims for yuv420p
    mp4 = out_dir / f"s_demo{args.demo_num:03d}_layer{args.test_hidden_layer}_tau{args.tau_temp:g}.mp4"
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{canvas.shape[0]}", "-r", str(raw_ds.fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
        stdin=subprocess.PIPE,
    )
    for j in range(len(frames_u8)):
        canvas[:672, x0:x1] = frames_u8[j].repeat(3, axis=0).repeat(3, axis=1)  # 3x upscale over the x-axis
        panel = plot_img.copy()
        panel[r0:r1, cur[j] - 1 : cur[j] + 2] = (0, 220, 0)  # bright green time cursor
        canvas[672:] = panel
        ffmpeg.stdin.write(canvas.tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    logging.info("saved %s", mp4)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
