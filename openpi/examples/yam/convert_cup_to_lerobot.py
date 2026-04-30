"""
Convert the CUP_TASK_0428 YAM dataset to LeRobot v0.1 format for openpi training.

Source layout (per demo):
    demoN/
        left_camera_rgb.mp4    # 640x480, 30 fps
        right_camera_rgb.mp4
        top_camera_rgb.mp4
        left_joint_positions.npy   # (T, 7) -- joint 7 == gripper
        left_control.npy           # (T, 7) -- action (target joint positions)
        left_gripper_position.npy  # (T, 1) -- redundant, ignored
        left_joint_velocities.npy  # (T, 7) -- ignored
        metadata.json              # {"num_steps": T, "hz": 30, ...}

Output:
    $HF_LEROBOT_HOME/<repo_id>/  (default: ~/.cache/huggingface/lerobot/<repo_id>/)

Usage:
    cd /home/kewalk/memory_project/openpi
    uv run examples/yam/convert_cup_to_lerobot.py \\
        --data-dir /home/kewalk/memory_project/dataset/CUP_TASK_0428
"""

import json
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

REPO_NAME = "kewalk/cup_task_0428"
TASK_PROMPT = "pick up the cup with yellow object inside"
FPS = 30
H, W = 480, 640  # native camera resolution


def _read_video_frames(path: Path) -> np.ndarray:
    """Decode an MP4 to (T, H, W, 3) uint8."""
    frames = iio.imread(str(path), plugin="pyav")  # (T, H, W, 3)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Unexpected frame shape {frames.shape} for {path}")
    return frames.astype(np.uint8)


def _load_demo(demo_dir: Path):
    meta = json.loads((demo_dir / "metadata.json").read_text())
    n_steps = int(meta["num_steps"])

    state = np.load(demo_dir / "left_joint_positions.npy").astype(np.float32)  # (T, 7)
    action = np.load(demo_dir / "left_control.npy").astype(np.float32)         # (T, 7)

    top = _read_video_frames(demo_dir / "top_camera_rgb.mp4")
    left = _read_video_frames(demo_dir / "left_camera_rgb.mp4")
    right = _read_video_frames(demo_dir / "right_camera_rgb.mp4")

    # Trim to the shortest length so all streams align (videos can be off-by-one).
    T = min(n_steps, len(state), len(action), len(top), len(left), len(right))
    if T < n_steps:
        print(f"  [warn] {demo_dir.name}: trimming to T={T} (metadata said {n_steps})")
    return {
        "T": T,
        "state": state[:T],
        "action": action[:T],
        "top": top[:T],
        "left": left[:T],
        "right": right[:T],
    }


def main(data_dir: str, *, repo_id: str = REPO_NAME, push_to_hub: bool = False, limit: int | None = None):
    data_dir = Path(data_dir)
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="yam",
        fps=FPS,
        features={
            "top_image": {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "left_image": {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "right_image": {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    demo_dirs = sorted(
        [p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("demo")],
        key=lambda p: int(p.name.removeprefix("demo")),
    )
    if limit is not None:
        demo_dirs = demo_dirs[:limit]
        print(f"--limit {limit} -> using only {[d.name for d in demo_dirs]}")
    print(f"Found {len(demo_dirs)} demos under {data_dir}")

    for i, demo in enumerate(demo_dirs):
        if not (demo / "write_complete.flag").exists():
            print(f"  [skip] {demo.name}: no write_complete.flag")
            continue

        print(f"[{i+1}/{len(demo_dirs)}] {demo.name}")
        d = _load_demo(demo)
        for t in range(d["T"]):
            dataset.add_frame(
                {
                    "top_image": d["top"][t],
                    "left_image": d["left"][t],
                    "right_image": d["right"][t],
                    "state": d["state"][t],
                    "actions": d["action"][t],
                    "task": TASK_PROMPT,
                }
            )
        dataset.save_episode()

    print(f"Done. Dataset written to {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["yam", "cup_task", "openpi"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
