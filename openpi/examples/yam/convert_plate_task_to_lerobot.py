"""Convert PLATE_TASK_1 demos into the 'plate_task' LeRobot dataset.

Features written per frame:
  - top_image, left_image (videos)
  - state (7,), actions (7,)
  - task: a resolved prompt (e.g. "put banana on the light blue plate") built
    from metadata.json's episode_plan via plate_prompts.detailed_prompt.

Demos 1-35 train, 36-37 held out.

Usage:
    cd /home/kewalk/memory_project/openpi
    uv run examples/yam/convert_plate_task_to_lerobot.py \\
        --data-dir /home/kewalk/memory_project/dataset/PLATE_TASK_1
"""

import gc
import json
import shutil
import sys
from functools import partial
from pathlib import Path

import datasets as hf_datasets
import imageio.v3 as iio
import numpy as np
import tyro

# Swap LeRobot's default libsvtav1 -> libx264 BEFORE importing LeRobotDataset.
from lerobot.common.datasets import video_utils as _lerobot_video_utils
_lerobot_video_utils.encode_video_frames = partial(
    _lerobot_video_utils.encode_video_frames, vcodec="h264", crf=23
)

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import lerobot.common.datasets.lerobot_dataset as _lerobot_dataset_module
_lerobot_dataset_module.encode_video_frames = _lerobot_video_utils.encode_video_frames

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plate_prompts import build_per_frame_prompts  # noqa: E402

REPO_NAME = "kewalk123/plate_task"
FPS = 30
H, W = 480, 640


def _frame_iterator(path: Path):
    for frame in iio.imiter(str(path), plugin="pyav"):
        yield np.asarray(frame, dtype=np.uint8)


def _load_meta(demo_dir: Path):
    meta = json.loads((demo_dir / "metadata.json").read_text())
    state = np.load(demo_dir / "left_joint_positions.npy").astype(np.float32)
    action = np.load(demo_dir / "left_control.npy").astype(np.float32)
    stage_id = np.load(demo_dir / "stage_id.npy").astype(np.int32)
    return meta, state, action, stage_id


def main(
    data_dir: str,
    *,
    repo_id: str = REPO_NAME,
    push_to_hub: bool = False,
    limit: int | None = None,
    start_demo: int = 1,
    end_demo: int = 35,
    image_writer_processes: int = 5,
    image_writer_threads: int = 10,
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
            "top_image":  {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "left_image": {"dtype": "video", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "state":      {"dtype": "float32", "shape": (7,), "names": ["state"]},
            "actions":    {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
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

    for i, demo in enumerate(demo_dirs):
        if not (demo / "write_complete.flag").exists():
            print(f"  [skip] {demo.name}: no write_complete.flag")
            continue

        print(f"[{i+1}/{len(demo_dirs)}] {demo.name}")
        meta, state, action, stage_id = _load_meta(demo)
        T_meta = min(len(state), len(action), len(stage_id))
        prompts = build_per_frame_prompts(meta, stage_id[:T_meta])

        top_iter = _frame_iterator(demo / "top_camera_rgb.mp4")
        left_iter = _frame_iterator(demo / "left_camera_rgb.mp4")

        t = 0
        for t, (top_frame, left_frame) in enumerate(zip(top_iter, left_iter)):
            if t >= T_meta:
                break
            dataset.add_frame(
                {
                    "top_image": top_frame,
                    "left_image": left_frame,
                    "state": state[t],
                    "actions": action[t],
                    "task": prompts[t],
                }
            )
        T_written = t + 1 if t > 0 else 0
        del top_iter, left_iter, meta, state, action, stage_id, prompts
        gc.collect()
        dataset.save_episode()

        # Reset hf_dataset after save_episode — its concatenate_datasets path
        # accumulates every prior episode in RAM and OOMs past demo ~18.
        empty_features = dataset.hf_dataset.features
        dataset.hf_dataset = hf_datasets.Dataset.from_dict(
            {k: [] for k in empty_features},
            features=empty_features,
        )
        gc.collect()
        print(f"    T={T_written}")

    print(f"Done. Dataset written to {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["yam", "plate_task", "openpi"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
