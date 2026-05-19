"""
Convert the LONG_TASK_1 plate-memory dataset to LeRobot v0.1 format for the
no-memory baseline (`pi05_plate_memory_base_lora`).

Layout per demo:
    demoN/
        top_camera_rgb.mp4    # 640x480, 30 fps
        left_camera_rgb.mp4   # used as wrist view
        right_camera_rgb.mp4  # DROPPED — the baseline ignores live right wrist
        left_joint_positions.npy   # (T, 7)  state
        left_control.npy           # (T, 7)  action
        instruction.npy            # (T,)    str — segment-specific instruction
        stage_id.npy               # (T,)    int  0..7
        metadata.json              # episode_plan + segments

Difference vs the memory-conditioned converter (`convert_long_task_to_lerobot.py`
on main): no `memory_image` and no `has_memory` channel are written. The
`right_wrist_0_rgb` slot is still synthesized at training time (as zeros with
`image_mask=False`) inside `PlateMemoryInputs`, so the model sees a missing
right camera rather than data from the real right wrist.

Memory note:
    Frames are streamed via iio.imiter() so peak RAM stays at ~1 frame per video
    instead of ~3 GB per video. Per-episode `dataset.hf_dataset` reset prevents
    LeRobot's `concatenate_datasets` accumulator from growing without bound.

Train split:
    Default range is demos 1..35. demo36 / demo37 are held out for offline eval.

Usage:
    cd /home/kewalk/memory_project/openpi
    uv run examples/yam/convert_plate_memory_to_lerobot.py \\
        --data-dir /home/kewalk/memory_project/dataset/LONG_TASK_1
"""

import gc
import json
import shutil
from pathlib import Path

import datasets as hf_datasets
import imageio.v3 as iio
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

REPO_NAME = "kewalk/plate_memory_base"
DEFAULT_DATA_DIR = "/iris/u/kewalk/memory_project/dataset/LONG_TASK_1"
FPS = 30
H, W = 480, 640


def _frame_iterator(path: Path):
    """Yield frames as (H, W, 3) uint8 arrays one at a time."""
    for frame in iio.imiter(str(path), plugin="pyav"):
        yield np.asarray(frame, dtype=np.uint8)


def _load_meta(demo_dir: Path):
    meta = json.loads((demo_dir / "metadata.json").read_text())
    state = np.load(demo_dir / "left_joint_positions.npy").astype(np.float32)
    action = np.load(demo_dir / "left_control.npy").astype(np.float32)
    instruction = np.load(demo_dir / "instruction.npy", allow_pickle=True)
    stage_id = np.load(demo_dir / "stage_id.npy").astype(np.int32)
    return meta, state, action, instruction, stage_id


def main(
    data_dir: str = DEFAULT_DATA_DIR,
    *,
    repo_id: str = REPO_NAME,
    push_to_hub: bool = False,
    limit: int | None = None,
    start_demo: int = 1,
    end_demo: int = 35,
    image_writer_processes: int = 2,
    image_writer_threads: int = 4,
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
            "top_image":  {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "left_image": {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
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
        meta, state, action, instruction, stage_id = _load_meta(demo)

        T_meta = min(len(state), len(action), len(instruction), len(stage_id))

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
                    "task": str(instruction[t]),
                }
            )
        T_written = t + 1 if t > 0 else 0
        # Drop references and explicitly collect before save_episode flushes
        # writers — avoids holding the previous demo's frames during flush.
        del top_iter, left_iter, meta, state, action, instruction, stage_id
        gc.collect()
        dataset.save_episode()

        # LeRobot's _save_episode_table appends each episode's embedded images
        # into self.hf_dataset via concatenate_datasets — accumulating all prior
        # episodes in RAM and OOMing past demo ~18. The per-episode parquet is
        # already on disk, so reset the in-memory accumulator to keep RAM
        # bounded across many demos.
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
            tags=["yam", "plate_memory", "openpi", "baseline"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
