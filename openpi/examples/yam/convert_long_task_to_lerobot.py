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

Memory note:
    Frames are streamed via iio.imiter() so peak RAM stays at ~1 frame per video
    instead of ~3 GB per video. The keyframe is read separately with index=
    random access. image_writer_processes=2 keeps writer-queue memory bounded.

Train split:
    Default range is demos 1..35. demo36 / demo37 are held out for retrieval-method
    comparisons.

Usage:
    cd /home/kewalk/memory_project/openpi
    uv run examples/yam/convert_long_task_to_lerobot.py \\
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

REPO_NAME = "kewalk/long_task_1_mem"
FPS = 30
H, W = 480, 640
QUERY_STAGES = {3, 4, 5, 6}  # query1..query4


def _read_keyframe(path: Path, index: int) -> np.ndarray:
    """Decode a single frame at `index` without loading the whole video."""
    frame = iio.imread(str(path), plugin="pyav", index=index)
    return np.asarray(frame, dtype=np.uint8)


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
    data_dir: str,
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
            "top_image":    {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "left_image":   {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "memory_image": {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "state":        {"dtype": "float32", "shape": (7,), "names": ["state"]},
            "actions":      {"dtype": "float32", "shape": (7,), "names": ["actions"]},
            "has_memory":   {"dtype": "float32", "shape": (1,), "names": ["has_memory"]},
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

    zero_image = np.zeros((H, W, 3), dtype=np.uint8)

    for i, demo in enumerate(demo_dirs):
        if not (demo / "write_complete.flag").exists():
            print(f"  [skip] {demo.name}: no write_complete.flag")
            continue

        print(f"[{i+1}/{len(demo_dirs)}] {demo.name}")
        meta, state, action, instruction, stage_id = _load_meta(demo)

        # Observe segment first; its end_step-1 indexes the keyframe.
        observe_seg = next(s for s in meta["segments"] if s["name"] == "observe")
        T_meta = min(len(state), len(action), len(instruction), len(stage_id))
        keyframe_idx = min(observe_seg["end_step"] - 1, T_meta - 1)
        keyframe = _read_keyframe(demo / "top_camera_rgb.mp4", keyframe_idx)

        # Stream frames from both videos in lockstep with state/action/etc.
        top_iter = _frame_iterator(demo / "top_camera_rgb.mp4")
        left_iter = _frame_iterator(demo / "left_camera_rgb.mp4")

        n_query_frames = 0
        t = 0
        for t, (top_frame, left_frame) in enumerate(zip(top_iter, left_iter)):
            if t >= T_meta:
                break
            stage = int(stage_id[t])
            if stage in QUERY_STAGES:
                memory_img = keyframe
                has_mem = 1.0
                n_query_frames += 1
            else:
                memory_img = zero_image
                has_mem = 0.0
            dataset.add_frame(
                {
                    "top_image": top_frame,
                    "left_image": left_frame,
                    "memory_image": memory_img,
                    "state": state[t],
                    "actions": action[t],
                    "has_memory": np.array([has_mem], dtype=np.float32),
                    "task": str(instruction[t]),
                }
            )
        T_written = t + 1 if t > 0 else 0
        # Drop references and explicitly collect before save_episode flushes
        # writers — avoids holding the previous demo's frames during flush.
        del top_iter, left_iter, keyframe, meta, state, action, instruction, stage_id
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
        print(f"    T={T_written}  query_frames={n_query_frames}")

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
