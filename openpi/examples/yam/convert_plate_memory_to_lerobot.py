"""
Convert the LONG_TASK_1 plate-memory YAM dataset to LeRobot v0.1 format for openpi training.

Source layout (per demo):
    demoN/
        left_camera_rgb.mp4    # 640x480, ~30 fps
        right_camera_rgb.mp4
        top_camera_rgb.mp4
        left_joint_positions.npy   # (T, 7) -- joint 7 == gripper
        left_control.npy           # (T, 7) -- action (target joint positions)
        instruction.npy            # (T,)   -- per-frame segment instruction (str)
        stage_id.npy               # (T,)   -- per-frame segment id (int)
        metadata.json              # {"num_steps": T, "hz": 30, "segments": [...], ...}

Unlike the cup task, the language instruction varies WITHIN an episode (observe / mix /
delay / query1..4 / return_home). We use `instruction.npy[t]` directly as the LeRobot
`task` field for frame t.

The right camera is still written into the dataset as the real video; disabling it for
the no-right-cam baseline happens inside the policy input transform
(see openpi.policies.plate_memory_policy.PlateMemoryInputs).

Output:
    $HF_LEROBOT_HOME/<repo_id>/  (default: ~/.cache/huggingface/lerobot/<repo_id>/)

Usage:
    cd /iris/u/kewalk/memory_project/openpi
    # smoke test on demo1
    uv run examples/yam/convert_plate_memory_to_lerobot.py --limit 1
    # 35-demo train split (demo36/demo37 held out for offline eval)
    uv run examples/yam/convert_plate_memory_to_lerobot.py --limit 35
"""

import gc
import json
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

REPO_NAME = "kewalk/plate_memory_base"
DEFAULT_DATA_DIR = "/iris/u/kewalk/memory_project/dataset/LONG_TASK_1"
FPS = 30
H, W = 480, 640  # native camera resolution

# Drain the image-writer queue every FLUSH_EVERY frames to keep queue depth (and
# therefore RSS) bounded. At 480x640x3 uint8, 3 cams x 256 frames ≈ 700 MB peak.
FLUSH_EVERY = 256


def _video_frame_count(path: Path) -> int:
    """Frame count from the container header (no full decode)."""
    props = iio.improps(str(path), plugin="pyav")
    if props.shape and props.shape[0] > 0:
        return int(props.shape[0])
    # Fallback: stream-decode if the header didn't expose frame count.
    n = 0
    for _ in iio.imiter(str(path), plugin="pyav"):
        n += 1
    return n


def _load_demo_meta(demo_dir: Path):
    """Load only state / action / instructions / per-camera frame counts.

    Videos are streamed frame-by-frame at write time to bound peak memory.
    """
    meta = json.loads((demo_dir / "metadata.json").read_text())
    n_steps = int(meta["num_steps"])

    state = np.load(demo_dir / "left_joint_positions.npy").astype(np.float32)  # (T, 7)
    action = np.load(demo_dir / "left_control.npy").astype(np.float32)         # (T, 7)
    instructions = np.load(demo_dir / "instruction.npy", allow_pickle=False)    # (T,) <U??

    n_top = _video_frame_count(demo_dir / "top_camera_rgb.mp4")
    n_left = _video_frame_count(demo_dir / "left_camera_rgb.mp4")
    n_right = _video_frame_count(demo_dir / "right_camera_rgb.mp4")

    T = min(n_steps, len(state), len(action), len(instructions), n_top, n_left, n_right)
    if T < n_steps:
        print(f"  [warn] {demo_dir.name}: trimming to T={T} (metadata said {n_steps})")
    return {
        "T": T,
        "state": state[:T],
        "action": action[:T],
        "instructions": instructions[:T],
    }


def main(
    data_dir: str = DEFAULT_DATA_DIR,
    *,
    repo_id: str = REPO_NAME,
    push_to_hub: bool = False,
    limit: int | None = None,
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
            "top_image": {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "left_image": {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "right_image": {"dtype": "image", "shape": (H, W, 3), "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        # Threads-only writer (in-process queue). Subprocess mode used an unbounded
        # multiprocessing.JoinableQueue that filled faster than disk could drain
        # it -> RSS reached 40+ GB by demo 15. We also throttle below by calling
        # _wait_image_writer() every FLUSH_EVERY frames so the queue stays bounded.
        image_writer_threads=4,
        image_writer_processes=0,
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
        d = _load_demo_meta(demo)
        T = d["T"]

        # Stream the three camera videos frame-by-frame so we never hold a full
        # decoded video in memory at once (~2.6 GB / video).
        top_iter = iio.imiter(str(demo / "top_camera_rgb.mp4"), plugin="pyav")
        left_iter = iio.imiter(str(demo / "left_camera_rgb.mp4"), plugin="pyav")
        right_iter = iio.imiter(str(demo / "right_camera_rgb.mp4"), plugin="pyav")

        for t in range(T):
            top_frame = next(top_iter)
            left_frame = next(left_iter)
            right_frame = next(right_iter)
            dataset.add_frame(
                {
                    "top_image": np.asarray(top_frame, dtype=np.uint8),
                    "left_image": np.asarray(left_frame, dtype=np.uint8),
                    "right_image": np.asarray(right_frame, dtype=np.uint8),
                    "state": d["state"][t],
                    "actions": d["action"][t],
                    "task": str(d["instructions"][t]),
                }
            )
            # Periodic drain so the writer queue can't grow without bound.
            if (t + 1) % FLUSH_EVERY == 0:
                dataset.image_writer.wait_until_done()

        # Drop iterators (and any pyav decoder state they hold) before save_episode.
        del top_iter, left_iter, right_iter
        dataset.save_episode()

        # LeRobot's save_episode does `concatenate_datasets([hf_dataset, ep_dataset])`,
        # which keeps every embedded image (~150 KB/frame x 3 cams) in RAM forever.
        # Across 35 demos this hits ~80 GB RSS. The on-disk parquet shard is already
        # written by save_episode -> reset the in-memory copy to an empty dataset so
        # peak RAM stays per-episode bounded.
        dataset.hf_dataset = dataset.create_hf_dataset()
        del d
        gc.collect()

    print(f"Done. Dataset written to {output_path}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["yam", "plate_memory", "openpi"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
