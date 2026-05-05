"""
Convert the LONG_TASK_1 plate-memory dataset to LeRobot v0.1 format for openpi training.

Layout per demo:
    demoN/
        top_camera_rgb.mp4    # 640x480, 30 fps
        left_camera_rgb.mp4   # used as wrist view
        right_camera_rgb.mp4  # DROPPED — the model ignores live right wrist
        left_joint_positions.npy   # (T, 7)  state
        left_control.npy           # (T, 7)  action
        instruction.npy            # (T,)    str — segment-specific instruction
        stage_id.npy               # (T,)    int  0..7
        metadata.json              # episode_plan + segments

Memory channel:
    For each frame t we attach a memory_image (480, 640, 3) and a has_memory bool:
      - During query stages (stage_id ∈ {3, 4, 5, 6}): memory_image is the
        observe-stage keyframe (last frame of the observe segment), has_memory=True.
      - Otherwise: memory_image is zeros, has_memory=False.
    The training-time YamLongTaskInputs transform routes memory_image into
    pi0.5's right_wrist_0_rgb slot with the per-frame mask = has_memory.

Train split:
    Default range is demos 1..35. demo36 / demo37 are held out for retrieval-method
    comparisons.

Usage:
    cd /home/kewalk/memory_project/openpi
    uv run examples/yam/convert_long_task_to_lerobot.py \\
        --data-dir /home/kewalk/memory_project/dataset/LONG_TASK_1
"""

import json
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

REPO_NAME = "kewalk/long_task_1_mem"
FPS = 30
H, W = 480, 640
QUERY_STAGES = {3, 4, 5, 6}  # query1..query4


def _read_video_frames(path: Path) -> np.ndarray:
    frames = iio.imread(str(path), plugin="pyav")
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Unexpected frame shape {frames.shape} for {path}")
    return frames.astype(np.uint8)


def _load_demo(demo_dir: Path):
    meta = json.loads((demo_dir / "metadata.json").read_text())

    state = np.load(demo_dir / "left_joint_positions.npy").astype(np.float32)
    action = np.load(demo_dir / "left_control.npy").astype(np.float32)
    instruction = np.load(demo_dir / "instruction.npy", allow_pickle=True)
    stage_id = np.load(demo_dir / "stage_id.npy").astype(np.int32)

    top = _read_video_frames(demo_dir / "top_camera_rgb.mp4")
    left = _read_video_frames(demo_dir / "left_camera_rgb.mp4")

    T = min(len(state), len(action), len(top), len(left), len(instruction), len(stage_id))

    # Observe segment is the first segment; its end_step gives us the keyframe index.
    observe_seg = next(s for s in meta["segments"] if s["name"] == "observe")
    keyframe_idx = min(observe_seg["end_step"] - 1, T - 1)
    keyframe = top[keyframe_idx]  # (H, W, 3) uint8

    return {
        "T": T,
        "state": state[:T],
        "action": action[:T],
        "top": top[:T],
        "left": left[:T],
        "stage_id": stage_id[:T],
        "instruction": instruction[:T],
        "keyframe": keyframe,
    }


def main(
    data_dir: str,
    *,
    repo_id: str = REPO_NAME,
    push_to_hub: bool = False,
    limit: int | None = None,
    start_demo: int = 1,
    end_demo: int = 35,
):
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
            "top_image":    {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "left_image":   {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "memory_image": {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "state":        {"dtype": "float32", "shape": (7,), "names": ["state"]},
            "actions":      {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            "has_memory":   {"dtype": "float32", "shape": (1,), "names": ["has_memory"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    demo_dirs = sorted(
        [p for p in data_dir.iterdir() if p.is_dir() and p.name.startswith("demo")],
        key=lambda p: int(p.name.removeprefix("demo")),
    )
    demo_dirs = [p for p in demo_dirs if start_demo <= int(p.name.removeprefix("demo")) <= end_demo]
    if limit is not None:
        demo_dirs = demo_dirs[:limit]
        print(f"--limit {limit} -> using only {[d.name for d in demo_dirs]}")
    print(f"Found {len(demo_dirs)} demos under {data_dir} (range {start_demo}-{end_demo})")

    zero_image = np.zeros((H, W, 3), dtype=np.uint8)

    for i, demo in enumerate(demo_dirs):
        if not (demo / "write_complete.flag").exists():
            print(f"  [skip] {demo.name}: no write_complete.flag")
            continue

        print(f"[{i+1}/{len(demo_dirs)}] {demo.name}")
        d = _load_demo(demo)
        n_query_frames = 0
        for t in range(d["T"]):
            stage = int(d["stage_id"][t])
            if stage in QUERY_STAGES:
                memory_img = d["keyframe"]
                has_mem = 1.0
                n_query_frames += 1
            else:
                memory_img = zero_image
                has_mem = 0.0
            dataset.add_frame(
                {
                    "top_image": d["top"][t],
                    "left_image": d["left"][t],
                    "memory_image": memory_img,
                    "state": d["state"][t],
                    "actions": d["action"][t],
                    "has_memory": np.array([has_mem], dtype=np.float32),
                    "task": str(d["instruction"][t]),
                }
            )
        dataset.save_episode()
        print(f"    T={d['T']}  query_frames={n_query_frames}")

    print(f"Done. Dataset written to {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["yam", "long_task", "openpi", "memory"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
