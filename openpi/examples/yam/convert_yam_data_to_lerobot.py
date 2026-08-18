"""
Convert the bimanual YAM `bin_memory_banana` dataset, with per-frame subtask labels, to LeRobot format.

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
    subtask_labels.json                from examples/yam/label_subtasks_gui.py:
                                       [{"task": <subtask>, "start": <frame>, "end": <frame>}, ...]
                                       contiguous segments tiling the episode

We build:
    state   = concat(left_joint_positions, right_joint_positions)  -> (T, 14)  actual follower state
    actions = concat(left_control,         right_control)           -> (T, 14)  leader command
    image / left_wrist_image / right_wrist_image = top / left / right cameras

The per-frame LeRobot `task` field carries the *subtask* (stored as task_index + meta/tasks.jsonl).
The high-level prompt is NOT stored per frame: single-source datasets get a constant prompt at
training time (InjectDefaultPrompt); multi-source datasets store one instruction per episode in
a `meta/episode_prompts.json` sidecar ({"<episode_index>": "<instruction>"}), read at training
time by the data loader when `prompt_from_episode_meta` is set.
Demos without a complete subtask_labels.json are skipped.

Usage (single source, constant prompt injected at training time):
    uv run examples/yam/convert_yam_data_to_lerobot.py \
        --data_dir /iris/u/kewalk/memory_project/data/bin_memory_banana

Usage (multiple sources with per-source instructions, one combined dataset):
    uv run examples/yam/convert_yam_data_to_lerobot.py \
        --data_dirs /iris/u/kewalk/memory_project/data/0816_banana \
                    /iris/u/kewalk/memory_project/data/0816_grey_box \
        --instructions "find the banana" "find the grey pepper box" \
        --repo_name yam/bin_memory_0816_subtask

The resulting dataset is written to $HF_LEROBOT_HOME/<REPO_NAME>.
"""

import json
import pathlib
import re
import shutil

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

REPO_NAME = "yam/bin_memory_banana_subtask"  # Name of the output dataset.

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


def _load_frame_subtasks(demo: pathlib.Path, num_frames: int) -> list[str] | None:
    """Per-frame subtask strings from the labeler's json, or None if missing/incomplete."""
    label_file = demo / "subtask_labels.json"
    if not label_file.exists():
        return None
    labels: list[str] = []
    for seg in json.loads(label_file.read_text()):
        if seg["start"] != len(labels):  # segments must tile the episode contiguously from frame 0
            return None
        labels.extend([seg["task"]] * (seg["end"] - seg["start"] + 1))
    # Labels index the top mp4; num_frames may be trimmed slightly shorter (proprio/video off-by-ones).
    return labels[:num_frames] if len(labels) >= num_frames else None


def main(
    data_dir: str = "/iris/u/kewalk/memory_project/data/bin_memory_banana",
    *,
    data_dirs: list[str] | None = None,
    instructions: list[str] | None = None,
    repo_name: str = REPO_NAME,
    push_to_hub: bool = False,
):
    """Convert one or more raw demo folders into a single LeRobot dataset.

    `data_dirs` (with optional matching `instructions`) overrides `data_dir`. Sources are
    converted in the given order, so episode indices are contiguous per source. When
    `instructions` is provided, meta/episode_prompts.json records each episode's high-level
    instruction for per-episode prompt injection at training time.
    """
    source_paths = [pathlib.Path(d) for d in (data_dirs if data_dirs else [data_dir])]
    if instructions is not None and len(instructions) != len(source_paths):
        raise ValueError(f"got {len(instructions)} instructions for {len(source_paths)} data dirs.")

    # Clean up any existing dataset in the output directory.
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
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

    episode_prompts: dict[str, str] = {}
    for source_index, data_path in enumerate(source_paths):
        demo_dirs = sorted(
            (p for p in data_path.iterdir() if p.is_dir() and p.name.startswith("demo")),
            key=_natural_demo_key,
        )
        print(f"Found {len(demo_dirs)} demos in {data_path}")

        for demo in demo_dirs:
            if not (demo / "subtask_labels.json").exists():
                print(f"  skipping {demo.name}: no subtask_labels.json")
                continue

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
            num_frames = min(len(state), len(actions), len(top), len(left), len(right))
            if num_frames == 0:
                print(f"  skipping {demo.name}: no frames")
                continue

            subtasks = _load_frame_subtasks(demo, num_frames)
            if subtasks is None:
                print(f"  skipping {demo.name}: incomplete subtask_labels.json")
                continue
            print(f"  {demo.name}: {num_frames} frames, {len(set(subtasks))} subtasks", flush=True)

            for t in range(num_frames):
                dataset.add_frame(
                    {
                        "image": top[t],
                        "left_wrist_image": left[t],
                        "right_wrist_image": right[t],
                        "state": state[t],
                        "actions": actions[t],
                        "task": subtasks[t],
                    }
                )
            if instructions is not None:
                episode_prompts[str(dataset.num_episodes)] = instructions[source_index]
            dataset.save_episode()

    if instructions is not None:
        prompts_file = output_path / "meta" / "episode_prompts.json"
        prompts_file.write_text(json.dumps(episode_prompts, indent=2))
        print(f"wrote {len(episode_prompts)} episode prompts to {prompts_file}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["yam", "bimanual"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
