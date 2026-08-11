"""Evaluate a memory checkpoint (pi05_yam_mem_*) on a RAW YAM demo, threading the Titans memory
through the episode.

Like eval_yam_subtask_raw.py, the demo's mp4s/npys are read directly and pushed through the
config's own inference-time transform pipeline (SplitMemoryWindow is dataset-only and skipped).
`Pi0.sample_with_memory` runs sequentially every `stride` raw frames (default: the config's
memory_stride_frames, i.e. the training write cadence): each call reads M_{t-1}, decodes the
subtask, denoises the actions, then writes h_t into the memory. Outputs, in scripts/eval_results/:
  1. an mp4 of every raw top-camera frame (30 fps) with the held predicted subtask + surprise
     overlaid and the surprise-vs-frame plot underneath (cursor bar at the current frame, red
     dot flashing on write frames),
  2. a per-joint png of the overlapping predicted action chunks vs the recorded teleop control,
  3. the raw curves as npz and the per-prediction subtasks as txt,
plus the predicted-subtask timeline, the memory-gate norm and per-call latency on stdout.

Run from the repo root on the GPU machine:
    CUDA_VISIBLE_DEVICES=<free> uv run python scripts/eval_yam_mem_subtask_raw.py
"""

import dataclasses
import pathlib
import subprocess
import time

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.transforms as _transforms

CKPT = pathlib.Path("/iris/u/kewalk/memory_project/openpi/checkpoints/pi05_yam_mem_warmup/mem_warmup_v5_layer_8/2000")
RAW_DEMO = pathlib.Path("/iris/u/kewalk/memory_project/data/held_out_eval/demo1")
STRIDE = 0  # frames between predictions; 0 = the config's memory_stride_frames (cadence-matched default)
MAX_DECODE_STEPS = 10
FPS = 30  # recording rate of the raw mp4s
PLOT_H = 320  # pixel height of the surprise plot rendered under the camera frame
LOG_Y = True  # log-scale surprise axis (the interesting structure lives near 0)
FLASH = 6  # frames the write marker stays lit after each prediction
JOINT_NAMES = [f"{arm} {j}" for arm in ("left", "right") for j in (*range(6), "grip")]


@dataclasses.dataclass
class Args:
    ckpt_dir: pathlib.Path = CKPT
    raw_demo: pathlib.Path = RAW_DEMO
    stride: int = STRIDE
    max_decode_steps: int = MAX_DECODE_STEPS
    config: str = "pi05_yam_mem_v3"
    # A/B control: never thread the writes, so every prediction reads the blank (m0) memory.
    # If the subtask timeline matches the normal run, the episode memory contributed nothing.
    ablate_memory: bool = False
    # Second control: thread the writes normally but force the content gate to zero, so the
    # memory tokens are exact zero embeddings -- what the vision path alone predicts through
    # the (degenerate) readout position.
    zero_gate: bool = False


def _read_video_frames(path: pathlib.Path, stride: int) -> tuple[list[np.ndarray], int]:
    """Every stride-th frame of an mp4 as uint8 RGB, plus the total frame count."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if total % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        total += 1
    cap.release()
    return frames, total


def _runs(frames: list[int], labels: list[str]) -> str:
    """Collapse a per-prediction label sequence into 'startframe-endframe label | ...' runs."""
    out = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append(f"{frames[start]}-{frames[i - 1]} {labels[start]!r}")
            start = i
    return " | ".join(out)


def _render_plot(pred_frames: list[int], curve: np.ndarray, total: int, size_hw: tuple[int, int]):
    """The static surprise plot as an RGB array, plus per-raw-frame cursor columns and row range."""
    h, w = size_hw
    fig = plt.Figure(figsize=(w / 100, h / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.step(pred_frames, curve, where="post", lw=0.9, color="tab:blue")
    ax.plot(pred_frames, curve, ".", ms=3, color="tab:blue")
    if LOG_Y:
        ax.set_yscale("log")
    # the trained m0's blank state is not scale-calibrated, so the first write's loss can sit
    # orders of magnitude above the written-state band: scale the axis to the rest of the curve
    if len(curve) > 2:
        lo, hi = float(curve[1:].min()), float(curve[1:].max())
        ax.set_ylim(max(lo * 0.7, 1e-8), hi * 1.5)
        if curve[0] > hi * 1.5:
            ax.text(
                0.99,
                0.97,
                f"first write {curve[0]:.2e} (off scale)",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="tab:red",
            )
    ax.set_xlim(0, total - 1)
    ax.set_xlabel("frame")
    ax.set_ylabel("write surprise")
    ax.set_title("memory write surprise at each prediction")
    fig.tight_layout()
    canvas.draw()
    base = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    # data -> pixel column for every raw frame (only x matters), and the axes' row range
    xs = np.arange(total)
    cols = np.round(ax.transData.transform(np.column_stack([xs, np.full(total, curve.min())]))[:, 0]).astype(int)
    bbox = ax.get_window_extent()
    rows = (int(base.shape[0] - bbox.y1), int(base.shape[0] - bbox.y0))
    return base, np.clip(cols, 0, base.shape[1] - 1), rows


def main(args: Args) -> None:
    from flax import nnx
    import jax
    import jax.numpy as jnp

    cfg = _config.get_config(args.config)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats = _checkpoints.load_norm_stats(args.ckpt_dir / "assets", data_config.asset_id)
    stride = args.stride or data_config.memory_stride_frames
    assert stride > 0, "no stride: set --stride or use a config with memory_stride_frames"

    # Raw demo -> arrays. The top camera is read at stride 1 (the video shows every raw frame);
    # the wrist cameras are only needed at the prediction frames.
    demo = args.raw_demo
    state_raw = np.concatenate(
        [np.load(demo / "left_joint_positions.npy"), np.load(demo / "right_joint_positions.npy")], axis=1
    ).astype(np.float32)
    actions_raw = np.concatenate(
        [np.load(demo / "left_control.npy"), np.load(demo / "right_control.npy")], axis=1
    ).astype(np.float32)
    top, n_top = _read_video_frames(demo / "top_camera_rgb.mp4", 1)
    left, n_left = _read_video_frames(demo / "left_camera_rgb.mp4", stride)
    right, n_right = _read_video_frames(demo / "right_camera_rgb.mp4", stride)
    total = min(len(state_raw), len(actions_raw), n_top, n_left, n_right)
    eval_ts = list(range(0, total, stride))
    print(
        f"{demo}: {total} frames -> {len(eval_ts)} predictions (one write per prediction, stride {stride})", flush=True
    )

    # The exact inference-time input pipeline from the config. BuildMemorySequence is the
    # dataset-side unpacker of lerobot's stacked delta_timestamps frames -- skipped on raw items.
    input_transforms = [
        tf for tf in data_config.data_transforms.inputs if not isinstance(tf, _transforms.BuildMemorySequence)
    ]
    normalize = _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
    model_transforms = list(data_config.model_transforms.inputs)

    unnormalize = _transforms.Unnormalize(
        {"actions": norm_stats["actions"]}, use_quantiles=data_config.use_quantile_norm
    )
    arm_mask = np.asarray(_transforms.make_bool_mask(6, -1, 6, -1))

    def build_item(t: int) -> dict:
        """Model inputs for raw frame t (inference-style: no actions, no window)."""
        item = {
            "observation/image": top[t],
            "observation/left_wrist_image": left[t // stride],
            "observation/right_wrist_image": right[t // stride],
            "observation/state": state_raw[t],
        }
        for tf in input_transforms:
            item = tf(item)
        item = normalize(item)
        for tf in model_transforms:
            item = tf(item)
        return item

    pg = _tokenizer.FASTSubtaskTokenizer(cfg.model.max_token_len)._paligemma_tokenizer  # noqa: SLF001
    # Same terminator the training subtasks were tokenized with (trailing "\n" of the segment).
    stop_token = int(pg.encode("placeholder subtask\n")[-1])

    # float32 restore: the memory's inner gradient descent was built and validated in f32.
    model = cfg.model.load(_model.restore_params(args.ckpt_dir / "params", dtype=jnp.float32))
    gate = np.asarray(model.memory_gate.value)
    print(
        f"loaded {args.ckpt_dir} | memory_gate norm {np.linalg.norm(gate):.4f} "
        f"mean|g| {np.abs(gate).mean():.5f} max|g| {np.abs(gate).max():.5f} (0 = memory content unused)",
        flush=True,
    )
    if args.zero_gate:
        model.memory_gate.value = jnp.zeros_like(model.memory_gate.value)
        print("ZERO-GATE: content gate forced to 0 -- memory tokens are zero embeddings", flush=True)
    graphdef, state = nnx.split(model)

    infer = jax.jit(
        lambda s, ms, rng, o: nnx.merge(graphdef, s).sample_with_memory(
            rng, o, ms, stop_token=stop_token, max_decode_steps=args.max_decode_steps
        )
    )
    mem_state = model.memory.init_state(1)
    if args.ablate_memory:
        print("ABLATION: writes are discarded -- every prediction reads the blank m0 memory", flush=True)

    # Sequential episode replay: the memory state threads from one prediction into the next.
    preds: list[str] = []
    surprise: list[float] = []
    call_ms: list[float] = []
    pred_chunks: list[np.ndarray] = []
    t_start = time.perf_counter()
    for k, t in enumerate(eval_ts):
        item = build_item(t)
        batch = jax.tree.map(lambda x: np.asarray(x)[None], item)
        t0 = time.perf_counter()
        actions, new_state, aux = infer(
            state, mem_state, jax.random.fold_in(jax.random.key(0), k), _model.Observation.from_dict(batch)
        )
        jax.block_until_ready((actions, new_state))
        call_ms.append((time.perf_counter() - t0) * 1e3)
        if not args.ablate_memory:
            mem_state = new_state

        tokens = np.asarray(aux["tokens"])[0]
        mask = np.asarray(aux["token_mask"])[0]
        preds.append(pg.decode(tokens[mask].tolist()).strip())
        surprise.append(float(aux["surprise"][0]))
        delta = unnormalize({"actions": np.asarray(actions)[0, :, :14]})["actions"]
        pred_chunks.append(delta + np.where(arm_mask, state_raw[t], 0.0))  # AbsoluteActions, [horizon, 14]
        if k == 0:
            print(
                f"first call {call_ms[0] / 1e3:.1f}s (incl. compile) | gates theta {np.asarray(aux['theta']).mean():.3f} "
                f"eta {np.asarray(aux['eta']).mean():.3f} alpha {np.asarray(aux['alpha']).mean():.4f}",
                flush=True,
            )
        print(f"[{k:3d}] frame {t:5d}  {call_ms[k]:6.0f} ms  surprise {surprise[k]:.3f}  pred {preds[k]!r}", flush=True)
    curve = np.asarray(surprise)
    steady = np.asarray(call_ms[1:] if len(call_ms) > 1 else call_ms)

    print(f"\npred timeline (frames): {_runs(eval_ts, preds)}")
    print(
        f"latency steady {steady.mean():.0f} ms (p50 {np.percentile(steady, 50):.0f}, p95 {np.percentile(steady, 95):.0f}) "
        f"| surprise first {curve[0]:.3f} min {curve.min():.3f} last {curve[-1]:.3f}",
        flush=True,
    )

    out_dir = pathlib.Path(__file__).parent / "eval_results"
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.ckpt_dir.parent.name}_{args.ckpt_dir.name}_{demo.parent.name}_{demo.name}"
    if args.ablate_memory:
        tag += "_ablate"
    if args.zero_gate:
        tag += "_zerogate"

    np.savez(
        out_dir / f"mem_subtask_{tag}.npz", pred_frames=np.asarray(eval_ts), surprise=curve, call_ms=np.asarray(call_ms)
    )
    with open(out_dir / f"mem_subtask_{tag}.txt", "w") as f:
        f.writelines(f"{t}\t{s:.4f}\t{p}\n" for t, s, p in zip(eval_ts, curve, preds, strict=True))

    # mp4: every raw top-camera frame with the held prediction overlaid, surprise plot + cursor
    # underneath; the red dot flashes on the frames where a prediction + memory write happened.
    frame_h, frame_w = top[0].shape[:2]
    plot, cursor_cols, (row0, row1) = _render_plot(eval_ts, curve, total, (PLOT_H, frame_w))
    mp4 = out_dir / f"mem_subtask_{tag}.mp4"
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{frame_w}x{frame_h + PLOT_H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(mp4),
        ],
        stdin=subprocess.PIPE,
    )
    for i in range(total):
        k = min(i // stride, len(preds) - 1)
        cam = top[i].copy()
        cv2.putText(cam, f"pred: {preds[k]}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 0), 2, cv2.LINE_AA)
        cv2.putText(
            cam,
            f"frame {i}  surprise {curve[k]:.3g}",
            (12, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (235, 235, 60),
            1,
            cv2.LINE_AA,
        )
        if i // stride < len(preds) and i % stride < FLASH:
            cv2.circle(cam, (frame_w - 28, 28), 10, (235, 60, 60), -1)
            cv2.putText(cam, "write", (frame_w - 92, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 60, 60), 1, cv2.LINE_AA)
        panel = plot.copy()
        cv2.line(panel, (int(cursor_cols[i]), row0), (int(cursor_cols[i]), row1), (220, 50, 50), 2)
        ffmpeg.stdin.write(np.vstack([cam, panel]).tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"saved {mp4}")

    # per-joint plot: each predicted action chunk drawn over the frames it targets (consecutive
    # chunks overlap by horizon - stride), against the recorded teleop control in raw units.
    horizon = pred_chunks[0].shape[0]
    fig, axes = plt.subplots(7, 2, figsize=(14, 16), sharex=True)
    for j in range(14):
        ax = axes[j % 7, j // 7]
        ax.plot(np.arange(total), actions_raw[:total, j], lw=0.9, color="black", label="teleop gt")
        for k, t in enumerate(eval_ts):
            ax.plot(
                np.arange(t, t + horizon),
                pred_chunks[k][:, j],
                lw=0.7,
                color="tab:orange",
                alpha=0.7,
                label="pred chunk" if k == 0 else None,
            )
        ax.set_ylabel(JOINT_NAMES[j])
    axes[0, 0].legend()
    axes[6, 0].set_xlabel("frame")
    axes[6, 1].set_xlabel("frame")
    fig.suptitle(f"{demo.parent.name}/{demo.name}: predicted action chunks vs teleop control (raw units)")
    fig.tight_layout()
    png = out_dir / f"mem_joints_{tag}.png"
    fig.savefig(png, dpi=140)
    print(f"saved {png}")
    print(f"total time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
