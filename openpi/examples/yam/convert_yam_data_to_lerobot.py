"""
Convert the bimanual YAM `bin_memory_banana` dataset to LeRobot format.

The raw data lives in one folder per demo, e.g.
    <data_dir>/demo1, <data_dir>/demo2, ...
Each demo folder contains, recorded at ~30 Hz:
    left_joint_positions.npy   (T, 7)  follower state: 6 arm joints + gripper
    right_joint_positions.npy  (T, 7)
    left_control.npy           (T, 7)  leader/teleop command (= action target)
    right_control.npy          (T, 7)
    top_camera_rgb.mp4                 third-person view  (640x480)
    left_camera_rgb.mp4                left wrist view    (640x480)
    right_camera_rgb.mp4               right wrist view   (640x480)
    metadata.json

We build:
    state   = concat(left_joint_positions, right_joint_positions)  -> (T, 14)  actual follower state
    actions = concat(left_control,         right_control)           -> (T, 14)  leader command
    image / left_wrist_image / right_wrist_image = top / left / right cameras

OpenPi assumes proprio is stored in `state` and actions in `actions`.

Usage:
    uv run examples/yam/convert_yam_data_to_lerobot.py \
        --data_dir /iris/u/kewalk/memory_project/data/bin_memory_banana

The resulting dataset is written to $HF_LEROBOT_HOME/<REPO_NAME>.
"""

import pathlib
import re
import shutil

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

REPO_NAME = "yam/bin_memory_banana"  # Name of the output dataset.
PROMPT = "find the bin with banana"  # Single fixed language instruction for all demos.

# Image resolution of the recorded mp4s (height, width, channels).
IMG_SHAPE = (480, 640, 3)
# Combined bimanual state / action dimension: (6 arm joints + 1 gripper) * 2 arms.
DIM = 14
FPS = 30


def _natural_demo_key(p: pathlib.Path) -> int:
    """Sort demo folders numerically (demo1, demo2, ..., demo10) not lexically."""
    m = re.search(r"(\d+)$", p.name)
    return int(m.group(1)) if m else 0


def _read_video_frames(path: pathlib.Path) -> list[np.ndarray]:
    """Read all frames of an mp4 as uint8 RGB (H, W, C) arrays."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # cv2 returns BGR; the model / LeRobot expect RGB.
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def main(
    data_dir: str = "/iris/u/kewalk/memory_project/data/bin_memory_banana",
    *,
    push_to_hub: bool = False,
):
    data_path = pathlib.Path(data_dir)

    # Clean up any existing dataset in the output directory.
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="yam",
        fps=FPS,
        features={
            "image": {
                "dtype": "image",
                "shape": IMG_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "left_wrist_image": {
                "dtype": "image",
                "shape": IMG_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "right_wrist_image": {
                "dtype": "image",
                "shape": IMG_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (DIM,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (DIM,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    demo_dirs = sorted(
        (p for p in data_path.iterdir() if p.is_dir() and p.name.startswith("demo")),
        key=_natural_demo_key,
    )
    print(f"Found {len(demo_dirs)} demos in {data_path}")

    for demo in demo_dirs:
        left_jp = np.load(demo / "left_joint_positions.npy")
        right_jp = np.load(demo / "right_joint_positions.npy")
        left_ctl = np.load(demo / "left_control.npy")
        right_ctl = np.load(demo / "right_control.npy")

        state = np.concatenate([left_jp, right_jp], axis=1).astype(np.float32)  # (T, 14)
        actions = np.concatenate([left_ctl, right_ctl], axis=1).astype(np.float32)  # (T, 14)

        top = _read_video_frames(demo / "top_camera_rgb.mp4")
        left = _read_video_frames(demo / "left_camera_rgb.mp4")
        right = _read_video_frames(demo / "right_camera_rgb.mp4")

        # Guard against off-by-one between proprio and video frame counts.
        T = min(len(state), len(actions), len(top), len(left), len(right))
        if T == 0:
            print(f"  skipping {demo.name}: no frames")
            continue
        print(f"  {demo.name}: {T} frames")

        for t in range(T):
            dataset.add_frame(
                {
                    "image": top[t],
                    "left_wrist_image": left[t],
                    "right_wrist_image": right[t],
                    "state": state[t],
                    "actions": actions[t],
                    "task": PROMPT,
                }
            )
        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub(
            tags=["yam", "bimanual"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
