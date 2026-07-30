"""Evaluate a pi05_yam checkpoint directly on a RAW (unconverted, unlabeled) YAM demo folder.

No LeRobot conversion and no subtask labels needed: the demo's mp4s/npys are read directly and
pushed through the config's own transform pipeline (YamInputs -> DeltaActions -> Normalize with
the training norm stats -> prompt tokenization), i.e. exactly the inference-time input path.
Runs the fused `Pi0.sample_subtask_and_actions` per frame and produces, in scripts/eval_results/:
  1. an mp4 of the top-camera model input with the predicted subtask drawn on each frame,
  2. a per-joint plot of the predicted joint target (unnormalized, state added back -- the exact
     policy output path) vs the recorded teleop control, in raw units,
plus the predicted-subtask run-length timeline on stdout. There is no subtask ground truth here.

Edit the constants below (or override on the CLI), then:
    CUDA_VISIBLE_DEVICES=<free> uv run python scripts/eval_yam_subtask_raw.py
"""

import dataclasses
import pathlib
import subprocess
import time

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import cv2
import matplotlib.pyplot as plt

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.transforms as _transforms

CKPT = pathlib.Path("/iris/u/kewalk/memory_project/openpi/checkpoints/pi05_yam/pi05_yam_KI/10000")
RAW_DEMO = pathlib.Path("/iris/u/kewalk/memory_project/data/held_out_eval/demo1")
STRIDE = 1  # evaluate every stride-th frame of the demo
BATCH_SIZE = 32
MAX_DECODE_STEPS = 10
UPSCALE = 3  # video frames are the 224px model inputs, upscaled by this factor
FPS = 30  # recording rate of the raw mp4s
JOINT_NAMES = [f"{arm} {j}" for arm in ("left", "right") for j in (*range(6), "grip")]


@dataclasses.dataclass
class Args:
    ckpt_dir: pathlib.Path = CKPT
    raw_demo: pathlib.Path = RAW_DEMO
    stride: int = STRIDE
    batch_size: int = BATCH_SIZE
    max_decode_steps: int = MAX_DECODE_STEPS
    config: str = "pi05_yam"


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


def _runs(labels: list[str]) -> str:
    """Collapse a per-frame label sequence into 'start-end label | ...' runs."""
    out = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append(f"{start}-{i - 1} {labels[start]!r}")
            start = i
    return " | ".join(out)


def main(args: Args) -> None:
    from flax import nnx
    import jax
    import jax.numpy as jnp

    cfg = _config.get_config(args.config)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    norm_stats = _checkpoints.load_norm_stats(args.ckpt_dir / "assets", data_config.asset_id)

    # Raw demo -> arrays (same construction as examples/yam/convert_yam_data_to_lerobot.py).
    demo = args.raw_demo
    state_raw = np.concatenate(
        [np.load(demo / "left_joint_positions.npy"), np.load(demo / "right_joint_positions.npy")], axis=1
    ).astype(np.float32)
    actions_raw = np.concatenate(
        [np.load(demo / "left_control.npy"), np.load(demo / "right_control.npy")], axis=1
    ).astype(np.float32)
    top, n_top = _read_video_frames(demo / "top_camera_rgb.mp4", args.stride)
    left, n_left = _read_video_frames(demo / "left_camera_rgb.mp4", args.stride)
    right, n_right = _read_video_frames(demo / "right_camera_rgb.mp4", args.stride)
    total = min(len(state_raw), len(actions_raw), n_top, n_left, n_right)
    eval_ts = list(range(0, total, args.stride))
    print(f"{demo}: {total} frames -> evaluating {len(eval_ts)}", flush=True)

    # The exact inference-time input pipeline from the config (repack is dataset-only, skipped).
    input_transforms = list(data_config.data_transforms.inputs)  # YamInputs, DeltaActions (no-op here)
    normalize = _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
    model_transforms = list(data_config.model_transforms.inputs)  # prompt, resize, tokenize, pad

    # Output path back to raw units, like the policy server: unnormalize the predicted deltas and
    # add the current state back on the arm joints (grippers are trained absolute).
    unnormalize = _transforms.Unnormalize({"actions": norm_stats["actions"]}, use_quantiles=data_config.use_quantile_norm)
    arm_mask = np.asarray(_transforms.make_bool_mask(6, -1, 6, -1))

    def build_item(t: int) -> dict:
        """Model inputs for raw frame t (inference-style: no actions anywhere)."""
        item = {
            "observation/image": top[t // args.stride],
            "observation/left_wrist_image": left[t // args.stride],
            "observation/right_wrist_image": right[t // args.stride],
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

    model = cfg.model.load(_model.restore_params(args.ckpt_dir / "params", dtype=jnp.bfloat16))
    graphdef, state = nnx.split(model)
    print(f"loaded {args.ckpt_dir}", flush=True)

    infer = jax.jit(
        lambda s, rng, o: nnx.merge(graphdef, s).sample_subtask_and_actions(
            rng, o, stop_token=stop_token, max_decode_steps=args.max_decode_steps
        )
    )

    preds: list[str] = []
    gt_joints: list[np.ndarray] = []
    pred_joints: list[np.ndarray] = []
    top_frames: list[np.ndarray] = []
    t_start = time.perf_counter()
    for bi in range(0, len(eval_ts), args.batch_size):
        chunk_ts = eval_ts[bi : bi + args.batch_size]
        pad = args.batch_size - len(chunk_ts)  # pad the tail batch to keep one jit shape
        inputs = [build_item(t) for t in chunk_ts + chunk_ts[-1:] * pad]
        batch = jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs]), *inputs)
        top_frames.extend(batch["image"]["base_0_rgb"][: len(chunk_ts)])  # uint8, before from_dict converts
        actions, out = infer(state, jax.random.fold_in(jax.random.key(0), bi), _model.Observation.from_dict(batch))

        n0 = batch["tokenized_prompt_mask"].sum(-1)
        out_tokens = np.asarray(out.tokenized_prompt)
        out_n = np.asarray(out.tokenized_prompt_mask).sum(-1)
        actions = np.asarray(actions)
        for row, t in enumerate(chunk_ts):
            preds.append(pg.decode(out_tokens[row, n0[row] : out_n[row]].tolist()).strip())
            gt_joints.append(actions_raw[t])  # the teleop control command at frame t
            delta = unnormalize({"actions": actions[row, :, :14]})["actions"]
            pred_joints.append(delta[0] + np.where(arm_mask, state_raw[t], 0.0))  # AbsoluteActions
        if bi == 0:
            print(f"first batch done in {time.perf_counter() - t_start:.1f}s (incl. compile)", flush=True)

    print(f"\npred timeline: {_runs(preds)}")

    out_dir = pathlib.Path(__file__).parent / "eval_results"
    out_dir.mkdir(exist_ok=True)
    tag = f"{args.ckpt_dir.parent.name}_{args.ckpt_dir.name}_{demo.parent.name}_{demo.name}"

    # (1) mp4: top-camera model input with the predicted subtask drawn on every frame.
    size = 224 * UPSCALE
    fps = max(1.0, FPS / args.stride)
    mp4 = out_dir / f"subtask_{tag}.mp4"
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{size}x{size}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
        stdin=subprocess.PIPE,
    )
    for frame, pred in zip(top_frames, preds, strict=True):
        img = frame.repeat(UPSCALE, axis=0).repeat(UPSCALE, axis=1).copy()
        cv2.putText(img, f"pred: {pred}", (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 235, 60), 2, cv2.LINE_AA)
        ffmpeg.stdin.write(img.tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"saved {mp4}")

    # (2) per-joint plot: predicted joint target vs recorded teleop control (raw units).
    gt_j = np.stack(gt_joints)  # [T, 14]
    pred_j = np.stack(pred_joints)
    fig, axes = plt.subplots(7, 2, figsize=(14, 16), sharex=True)
    for j in range(14):
        ax = axes[j % 7, j // 7]
        ax.plot(eval_ts, gt_j[:, j], lw=0.9, label="gt")
        ax.plot(eval_ts, pred_j[:, j], lw=0.9, label="pred")
        ax.set_ylabel(JOINT_NAMES[j])
    axes[0, 0].legend()
    axes[6, 0].set_xlabel("frame")
    axes[6, 1].set_xlabel("frame")
    fig.suptitle(f"{demo.parent.name}/{demo.name}: predicted joint target vs teleop control (raw units)")
    fig.tight_layout()
    png = out_dir / f"joints_{tag}.png"
    fig.savefig(png, dpi=140)
    print(f"saved {png}")
    print(f"total time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
