"""Open-loop offline evaluation of pi05_yam_cup_lora on a recorded demo.

Replays one demo's recorded observations (3 RGB streams + 7-d joint state)
through the trained policy step-by-step. Saves the predicted action chunks
plus ground-truth signals to a single npz for downstream FK / plot / sim
playback.

Usage:
    cd /home/kewalk/memory_project
    python eval/offline_eval.py                       # default: demo1, ckpt 29999
    python eval/offline_eval.py --demo demo5
    python eval/offline_eval.py --ckpt-step 25000

Outputs:
    eval/results_<demo>.npz
        state         (T, 7)       recorded left_joint_positions
        gt_action     (T, 7)       recorded left_control
        pred_first    (T, 7)       first action of each predicted 10-step chunk
        pred_chunks   (T, 10, 7)   full predicted action chunks
"""

import argparse
import pathlib
import time

import imageio.v3 as iio
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

DATASET_ROOT = pathlib.Path("/home/kewalk/memory_project/dataset/CUP_TASK_0428")
CKPT_ROOT = pathlib.Path("/home/kewalk/memory_project/openpi/checkpoints/pi05_yam_cup_lora/cup_0429_v1")
EVAL_DIR = pathlib.Path("/home/kewalk/memory_project/eval")
PROMPT = "pick up the cup with yellow object inside"
CONFIG_NAME = "pi05_yam_cup_lora"
ACTION_HORIZON = 10


def load_demo(demo_dir: pathlib.Path):
    state = np.load(demo_dir / "left_joint_positions.npy").astype(np.float32)  # (T, 7)
    gt_action = np.load(demo_dir / "left_control.npy").astype(np.float32)       # (T, 7)
    top = iio.imread(str(demo_dir / "top_camera_rgb.mp4"), plugin="pyav").astype(np.uint8)
    left = iio.imread(str(demo_dir / "left_camera_rgb.mp4"), plugin="pyav").astype(np.uint8)
    right = iio.imread(str(demo_dir / "right_camera_rgb.mp4"), plugin="pyav").astype(np.uint8)
    T = min(len(state), len(gt_action), len(top), len(left), len(right))
    return state[:T], gt_action[:T], top[:T], left[:T], right[:T]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", default="demo1", help="Demo folder name under dataset/CUP_TASK_0428/")
    parser.add_argument("--ckpt-step", type=int, default=29999, help="Checkpoint step to load")
    parser.add_argument("--max-steps", type=int, default=None, help="Truncate demo to first N frames")
    args = parser.parse_args()

    demo_dir = DATASET_ROOT / args.demo
    ckpt_dir = CKPT_ROOT / str(args.ckpt_step)
    out_path = EVAL_DIR / f"results_{args.demo}.npz"

    print(f"[load] demo : {demo_dir}")
    print(f"[load] ckpt : {ckpt_dir}")

    state, gt_action, top, left, right = load_demo(demo_dir)
    if args.max_steps is not None:
        T = min(args.max_steps, len(state))
        state, gt_action, top, left, right = state[:T], gt_action[:T], top[:T], left[:T], right[:T]
    T = len(state)
    print(f"[load] {T} frames; cameras 3x{top.shape[1:]}")

    print(f"[load] policy ({CONFIG_NAME})")
    train_config = _config.get_config(CONFIG_NAME)
    policy = _policy_config.create_trained_policy(
        train_config, ckpt_dir, default_prompt=PROMPT
    )

    pred_chunks = np.zeros((T, ACTION_HORIZON, 7), dtype=np.float32)
    pred_first = np.zeros((T, 7), dtype=np.float32)

    t0 = time.time()
    for t in range(T):
        obs = {
            "observation/top_image":   top[t],
            "observation/left_image":  left[t],
            "observation/right_image": right[t],
            "observation/state":       state[t],
            "prompt":                  PROMPT,
        }
        out = policy.infer(obs)
        chunk = np.asarray(out["actions"])  # expect (action_horizon, 7) post YamOutputs
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
        state=state,
        gt_action=gt_action,
        pred_first=pred_first,
        pred_chunks=pred_chunks,
    )
    print(f"[save] {out_path}")

    abs_err = np.abs(pred_first - gt_action)
    print("\n[error vs gt_action — full demo]")
    print(f"  per-joint MAE (rad)  : {np.round(abs_err.mean(axis=0), 4).tolist()}")
    print(f"  per-joint RMSE (rad) : {np.round(np.sqrt((abs_err**2).mean(axis=0)), 4).tolist()}")
    print(f"  per-joint max  (rad) : {np.round(abs_err.max(axis=0), 4).tolist()}")
    print(f"  overall MAE (rad)    : {abs_err.mean():.4f}")


if __name__ == "__main__":
    main()
