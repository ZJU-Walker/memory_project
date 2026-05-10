"""Open-loop offline evaluation of pi05_long_task_mem_lora on a recorded demo.

Replays one LONG_TASK_1 demo's recorded observations (top + left RGB streams +
7-d joint state + per-frame instruction) through the trained memory-conditioned
policy step-by-step. Constructs the oracle memory_image (last frame of the
observe segment) for query stages and zeros otherwise. Saves predicted action
chunks plus ground-truth signals to a single npz, prints overall and per-stage
MAE/RMSE/max vs ground truth.

Usage:
    cd /home/kewalk/memory_project
    python eval/offline_eval_long.py                          # default: demo36, ckpt 5000
    python eval/offline_eval_long.py --demo demo37
    python eval/offline_eval_long.py --demo demo36 --ckpt-step 10000
    python eval/offline_eval_long.py --demo demo36 --max-steps 50      # smoke
    python eval/offline_eval_long.py --demo demo36 --no-memory         # ablation

Outputs:
    eval/results_long_<demo>[_nomem].npz
        state         (T, 7)         recorded left_joint_positions
        gt_action     (T, 7)         recorded left_control
        pred_first    (T, 7)         first action of each predicted chunk
        pred_chunks   (T, AH, 7)     full predicted action chunks (AH=50)
        stage_id      (T,)           per-frame stage id (0..7)
"""

import argparse
import json
import pathlib
import time

import imageio.v3 as iio
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

DATASET_ROOT = pathlib.Path("/home/kewalk/memory_project/dataset/LONG_TASK_1")
CKPT_ROOT = pathlib.Path(
    "/home/kewalk/memory_project/openpi/checkpoints/pi05_long_task_mem_lora/long_v1"
)
EVAL_DIR = pathlib.Path("/home/kewalk/memory_project/eval")
CONFIG_NAME = "pi05_long_task_mem_lora"
ACTION_HORIZON = 50
QUERY_STAGES = {3, 4, 5, 6}  # query1..query4
H, W = 480, 640
STAGE_NAMES = [
    ("observe",     [0]),
    ("mix",         [1]),
    ("delay",       [2]),
    ("query1",      [3]),
    ("query2",      [4]),
    ("query3",      [5]),
    ("query4",      [6]),
    ("return_home", [7]),
]


def load_demo(demo_dir: pathlib.Path):
    meta = json.loads((demo_dir / "metadata.json").read_text())
    state = np.load(demo_dir / "left_joint_positions.npy").astype(np.float32)  # (T, 7)
    gt_action = np.load(demo_dir / "left_control.npy").astype(np.float32)       # (T, 7)
    instruction = np.load(demo_dir / "instruction.npy", allow_pickle=True)
    stage_id = np.load(demo_dir / "stage_id.npy").astype(np.int32)
    top = iio.imread(str(demo_dir / "top_camera_rgb.mp4"), plugin="pyav").astype(np.uint8)
    left = iio.imread(str(demo_dir / "left_camera_rgb.mp4"), plugin="pyav").astype(np.uint8)

    T = min(len(state), len(gt_action), len(instruction), len(stage_id), len(top), len(left))
    state = state[:T]
    gt_action = gt_action[:T]
    instruction = instruction[:T]
    stage_id = stage_id[:T]
    top = top[:T]
    left = left[:T]

    observe_seg = next(s for s in meta["segments"] if s["name"] == "observe")
    keyframe_idx = min(observe_seg["end_step"] - 1, T - 1)
    keyframe = top[keyframe_idx].copy()  # (H, W, 3) uint8

    return {
        "T": T,
        "state": state,
        "gt_action": gt_action,
        "instruction": instruction,
        "stage_id": stage_id,
        "top": top,
        "left": left,
        "keyframe": keyframe,
    }


def per_stage_metrics(pred_first: np.ndarray, gt_action: np.ndarray,
                      stage_id: np.ndarray, instruction: np.ndarray):
    """Per-stage MAE/RMSE/max breakdown. Each query stage uses a different
    memory-dependent prompt, so the per-stage rows are also per-prompt rows."""
    abs_err = np.abs(pred_first - gt_action)
    print("\n[error vs gt_action — per stage / per prompt]")
    print(f"  {'stage':<12} {'count':>6}  {'MAE (rad)':>10}  {'RMSE':>8}  {'max':>8}  prompt")
    for name, ids in STAGE_NAMES:
        mask = np.isin(stage_id, ids)
        n = int(mask.sum())
        if n == 0:
            continue
        sub = abs_err[mask]
        mae = float(sub.mean())
        rmse = float(np.sqrt((sub ** 2).mean()))
        mx = float(sub.max())
        # Use the first prompt seen in this stage as the representative — the
        # converter writes the same string across the whole segment.
        idx = int(np.argmax(mask))  # first True index
        prompt = str(instruction[idx])
        if len(prompt) > 70:
            prompt = prompt[:67] + "..."
        print(f"  {name:<12} {n:>6}  {mae:>10.4f}  {rmse:>8.4f}  {mx:>8.4f}  {prompt}")
    print(f"\n  overall MAE (rad)    : {abs_err.mean():.4f}")
    print(f"  overall RMSE (rad)   : {np.sqrt((abs_err**2).mean()):.4f}")
    print(f"  overall max  (rad)   : {abs_err.max():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", default="demo36",
                        help="Demo folder name under dataset/LONG_TASK_1/")
    parser.add_argument("--ckpt-step", type=int, default=5000,
                        help="Checkpoint step to load (e.g. 5000, 10000)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Truncate demo to first N frames (smoke test)")
    parser.add_argument("--no-memory", action="store_true",
                        help="Force memory_image=zeros and has_memory=0 always (ablation).")
    args = parser.parse_args()

    demo_dir = DATASET_ROOT / args.demo
    ckpt_dir = CKPT_ROOT / str(args.ckpt_step)
    suffix = "_nomem" if args.no_memory else ""
    out_path = EVAL_DIR / f"results_long_{args.demo}{suffix}.npz"

    print(f"[load] demo  : {demo_dir}")
    print(f"[load] ckpt  : {ckpt_dir}")
    print(f"[mode] memory: {'OFF (ablation)' if args.no_memory else 'ON (oracle)'}")

    d = load_demo(demo_dir)
    if args.max_steps is not None:
        T = min(args.max_steps, d["T"])
        for k in ("state", "gt_action", "instruction", "stage_id", "top", "left"):
            d[k] = d[k][:T]
        d["T"] = T
    T = d["T"]
    print(f"[load] {T} frames")
    unique_stages, counts = np.unique(d["stage_id"], return_counts=True)
    for s, c in zip(unique_stages, counts):
        print(f"  stage {int(s)}: {int(c)} frames")

    print(f"[load] policy ({CONFIG_NAME})")
    train_config = _config.get_config(CONFIG_NAME)
    policy = _policy_config.create_trained_policy(train_config, ckpt_dir)

    pred_chunks = np.zeros((T, ACTION_HORIZON, 7), dtype=np.float32)
    pred_first = np.zeros((T, 7), dtype=np.float32)

    zero_image = np.zeros((H, W, 3), dtype=np.uint8)

    t0 = time.time()
    for t in range(T):
        in_query = int(d["stage_id"][t]) in QUERY_STAGES
        if args.no_memory:
            in_query = False
        mem_img = d["keyframe"] if in_query else zero_image
        has_mem = np.float32(1.0 if in_query else 0.0)

        obs = {
            "observation/top_image":    d["top"][t],
            "observation/left_image":   d["left"][t],
            "observation/memory_image": mem_img,
            "observation/state":        d["state"][t],
            "observation/has_memory":   np.array([has_mem], dtype=np.float32),
            "prompt":                   str(d["instruction"][t]),
        }
        out = policy.infer(obs)
        chunk = np.asarray(out["actions"])
        if chunk.shape != (ACTION_HORIZON, 7):
            raise RuntimeError(f"Unexpected action shape at t={t}: {chunk.shape}")
        pred_chunks[t] = chunk
        pred_first[t] = chunk[0]

        if (t + 1) % 50 == 0 or t == T - 1:
            elapsed = time.time() - t0
            print(f"  step {t+1}/{T}  elapsed={elapsed:.1f}s  rate={(t+1)/elapsed:.1f} Hz")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        state=d["state"],
        gt_action=d["gt_action"],
        pred_first=pred_first,
        pred_chunks=pred_chunks,
        stage_id=d["stage_id"],
    )
    print(f"[save] {out_path}")

    per_stage_metrics(pred_first, d["gt_action"], d["stage_id"], d["instruction"])


if __name__ == "__main__":
    main()
