"""
Convert bimanual YAM demonstrations, with per-frame subtask labels, to LeRobot format.

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
Legacy directory mode skips demos without a complete ``subtask_labels.json``.  Manifest mode is
fail-closed: every included episode must have complete labels, an instruction, and the expected
stream length before an existing output is touched.

Usage (single source, constant prompt injected at training time):
    uv run examples/yam/convert_yam_data_to_lerobot.py \
        --data_dir /iris/u/kewalk/memory_project/data/bin_memory_banana \
        --task-vocabulary "open both lids" "wait; target bin is left" "open left bin" \
                          "close both lids and reset arms" "inspect both bins" \
                          "open right bin" "wait; target bin is right"

Usage (multiple sources with per-source instructions, one combined dataset):
    uv run examples/yam/convert_yam_data_to_lerobot.py \
        --data_dirs /iris/u/kewalk/memory_project/data/0816_banana \
                    /iris/u/kewalk/memory_project/data/0816_grey_box \
        --instructions "find the banana" "find the grey pepper box" \
        --repo_name yam/bin_memory_0816_subtask

Usage (ordered per-episode manifest; recommended for mixed-prompt collections):
    uv run examples/yam/convert_yam_data_to_lerobot.py \
        --episode-manifest ../data/0816_0830_episode_manifest_v1.json \
        --repo-name yam/bin_memory_0816_0830_v35_subtask

The resulting dataset is written to $HF_LEROBOT_HOME/<REPO_NAME>.
"""

import hashlib
import json
import pathlib
import re
import shutil
from typing import Any

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
import tyro

REPO_NAME = "yam/bin_memory_banana_subtask"  # Name of the output dataset.

# Image resolution of the recorded mp4s (height, width, channels).
IMG_SHAPE = (480, 640, 3)
# Combined bimanual state / action dimension: (6 arm joints + 1 gripper) * 2 arms.
DIM = 14
FPS = 30

_ARRAY_STREAMS = (
    "left_joint_positions.npy",
    "right_joint_positions.npy",
    "left_control.npy",
    "right_control.npy",
)
_VIDEO_STREAMS = (
    "top_camera_rgb.mp4",
    "left_camera_rgb.mp4",
    "right_camera_rgb.mp4",
)


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


def _load_frame_subtasks(
    demo: pathlib.Path, num_frames: int, label_filename: str = "subtask_labels.json"
) -> list[str] | None:
    """Per-frame subtask strings from the labeler's json, or None if missing/incomplete."""
    label_file = demo / label_filename
    if not label_file.exists():
        return None
    labels: list[str] = []
    try:
        segments = json.loads(label_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(segments, list) or not segments:
        return None
    for seg in segments:
        if not isinstance(seg, dict) or not {"task", "start", "end"}.issubset(seg):
            return None
        if not isinstance(seg["task"], str) or not isinstance(seg["start"], int) or not isinstance(seg["end"], int):
            return None
        if seg["start"] != len(labels):  # segments must tile the episode contiguously from frame 0
            return None
        if seg["end"] < seg["start"]:
            return None
        labels.extend([seg["task"]] * (seg["end"] - seg["start"] + 1))
    # Labels index the top mp4; num_frames may be trimmed slightly shorter (proprio/video off-by-ones).
    return labels[:num_frames] if len(labels) >= num_frames else None


def _validate_task_vocabulary(task_vocabulary: list[str] | None) -> tuple[str, ...] | None:
    """Validate an optional fixed task vocabulary without changing its order or spelling."""
    if task_vocabulary is None:
        return None
    if not task_vocabulary:
        raise ValueError("task_vocabulary must contain at least one task when supplied.")
    if any(not task.strip() for task in task_vocabulary):
        raise ValueError("task_vocabulary entries must be nonempty.")
    if len(set(task_vocabulary)) != len(task_vocabulary):
        raise ValueError("task_vocabulary entries must be unique.")
    return tuple(task_vocabulary)


def _validate_raw_label_tasks(
    demo: pathlib.Path, task_vocabulary: tuple[str, ...], label_filename: str = "subtask_labels.json"
) -> None:
    """Fail if any task string in a raw label file is outside the fixed vocabulary."""
    label_file = demo / label_filename
    allowed_tasks = set(task_vocabulary)
    raw_tasks = [segment["task"] for segment in json.loads(label_file.read_text())]
    unknown_tasks = list(dict.fromkeys(task for task in raw_tasks if task not in allowed_tasks))
    if unknown_tasks:
        raise ValueError(f"{label_file} contains tasks outside task_vocabulary: {unknown_tasks}")


def _register_task_vocabulary(dataset: LeRobotDataset, task_vocabulary: tuple[str, ...] | None) -> None:
    """Pre-register task IDs in the caller-specified order."""
    if task_vocabulary is not None:
        for task in task_vocabulary:
            dataset.meta.add_task(task)


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _strict_source_record(spec: dict[str, Any]) -> dict[str, Any]:
    demo = pathlib.Path(spec["raw_dir"])
    return {
        "stable_id": spec["stable_id"],
        "raw_dir": str(demo),
        "manifest_raw_dir": spec["manifest_raw_dir"],
        "manifest_position": spec["manifest_position"],
        "instruction": spec["instruction"],
        "target_side": spec.get("target_side"),
        "collection": spec.get("collection"),
        "raw_demo": demo.name,
        "metadata_counter": spec.get("metadata_counter"),
        "timestamp": spec.get("timestamp"),
        "num_frames": spec["num_frames"],
        "stream_lengths": spec["stream_lengths"],
        "label_file": str(spec.get("label_file") or "subtask_labels.json"),
        "label_sha256": spec["label_sha256"],
        "memory_waiting_core": spec.get("memory_waiting_core"),
    }


def _validate_resume_prefix(
    output_path: pathlib.Path,
    conversion_specs: list[dict[str, Any]],
    task_vocabulary: tuple[str, ...] | None,
) -> tuple[int, int]:
    """Validate a fully committed episode prefix and return (next episode, frame count)."""
    if not output_path.is_dir():
        raise FileNotFoundError(f"resume output does not exist: {output_path}")
    info = json.loads((output_path / "meta" / "info.json").read_text())
    resume_from = int(info["total_episodes"])
    if resume_from <= 0 or resume_from >= len(conversion_specs):
        raise ValueError(
            f"resume requires a non-empty incomplete prefix; found {resume_from}/{len(conversion_specs)} episodes"
        )
    expected_frames = sum(int(spec["num_frames"]) for spec in conversion_specs[:resume_from])
    if int(info["total_frames"]) != expected_frames:
        raise ValueError(
            f"resume prefix frame mismatch: info has {info['total_frames']}, manifest prefix has {expected_frames}"
        )
    if info.get("splits") != {"train": f"0:{resume_from}"}:
        raise ValueError(f"resume prefix has unexpected split metadata: {info.get('splits')}")

    tasks = _read_jsonl(output_path / "meta" / "tasks.jsonl")
    observed_vocabulary = tuple(row["task"] for row in sorted(tasks, key=lambda row: row["task_index"]))
    if task_vocabulary is not None and observed_vocabulary != task_vocabulary:
        raise ValueError(f"resume task vocabulary mismatch: observed={observed_vocabulary}, expected={task_vocabulary}")
    episodes = _read_jsonl(output_path / "meta" / "episodes.jsonl")
    if len(episodes) != resume_from:
        raise ValueError(f"resume prefix has {len(episodes)} episode metadata rows, expected {resume_from}")
    for episode_index, (row, spec) in enumerate(zip(episodes, conversion_specs[:resume_from], strict=True)):
        if row.get("episode_index") != episode_index or row.get("length") != spec["num_frames"]:
            raise ValueError(f"resume prefix episode metadata mismatch at index {episode_index}: {row}")

    parquet_files = sorted(output_path.glob("data/chunk-*/episode_*.parquet"))
    expected_names = [f"episode_{index:06d}.parquet" for index in range(resume_from)]
    if [path.name for path in parquet_files] != expected_names:
        raise ValueError("resume prefix parquet files are not the exact contiguous committed prefix")

    temporary_images = output_path / "images"
    if temporary_images.is_dir():
        image_files = [path for path in temporary_images.rglob("*") if path.is_file()]
        expected_episode_dir = f"episode_{resume_from:06d}"
        if any(expected_episode_dir not in path.parts for path in image_files):
            raise ValueError(f"resume found temporary images outside {expected_episode_dir}")
        shutil.rmtree(temporary_images)
        print(f"removed {len(image_files)} uncommitted temporary images for episode {resume_from}")
    return resume_from, expected_frames


def _open_dataset_for_append(repo_name: str, output_path: pathlib.Path) -> LeRobotDataset:
    """Open existing metadata for append without materializing all image Parquets in memory."""
    dataset = LeRobotDataset.__new__(LeRobotDataset)
    dataset.meta = LeRobotDatasetMetadata(repo_id=repo_name, root=output_path)
    dataset.repo_id = dataset.meta.repo_id
    dataset.root = dataset.meta.root
    dataset.revision = None
    dataset.tolerance_s = 1e-4
    dataset.image_writer = None
    dataset.episodes = None
    dataset.hf_dataset = dataset.create_hf_dataset()
    dataset.image_transforms = None
    dataset.delta_timestamps = None
    dataset.delta_indices = None
    dataset.episode_data_index = None
    dataset.video_backend = None
    dataset.episode_buffer = dataset.create_episode_buffer()
    dataset.start_image_writer(num_processes=0, num_threads=8)
    return dataset


def _video_frame_count(path: pathlib.Path) -> int:
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def _episode_stream_lengths(demo: pathlib.Path) -> dict[str, int]:
    """Lengths of exactly the streams consumed by the converter."""
    lengths: dict[str, int] = {}
    for filename in _ARRAY_STREAMS:
        path = demo / filename
        if not path.is_file():
            raise ValueError(f"{demo}: missing required stream {filename}")
        lengths[filename] = len(np.load(path, mmap_mode="r"))
    for filename in _VIDEO_STREAMS:
        path = demo / filename
        if not path.is_file():
            raise ValueError(f"{demo}: missing required stream {filename}")
        lengths[filename] = _video_frame_count(path)
        if lengths[filename] <= 0:
            raise ValueError(f"{demo}: video {filename} has no frames")
    return lengths


def _resolve_manifest_path(manifest_path: pathlib.Path, payload: dict[str, Any], raw_dir: str) -> pathlib.Path:
    path = pathlib.Path(raw_dir)
    if path.is_absolute():
        return path.resolve()
    root_value = payload.get("raw_root", ".")
    root = pathlib.Path(root_value)
    if not root.is_absolute():
        root = manifest_path.parent / root
    return (root / path).resolve()


def _load_episode_manifest(path: pathlib.Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load ordered included episodes and resolve their raw paths.

    The manifest preserves episode identity across mixed directories and explicit exclusions.
    Relative ``raw_dir`` values are resolved below ``raw_root`` (itself relative to the manifest).
    """
    manifest_path = path.resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError(f"{manifest_path}: expected schema_version=1")
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError(f"{manifest_path}: episodes must be a nonempty list")

    included: list[dict[str, Any]] = []
    seen_raw_dirs: set[pathlib.Path] = set()
    seen_stable_ids: set[str] = set()
    for position, raw in enumerate(raw_episodes):
        if not isinstance(raw, dict) or "raw_dir" not in raw:
            raise ValueError(f"{manifest_path}: episode {position} has no raw_dir")
        if not raw.get("include", True):
            if not str(raw.get("exclude_reason") or "").strip():
                raise ValueError(f"{manifest_path}: excluded episode {position} needs exclude_reason")
            continue
        demo = _resolve_manifest_path(manifest_path, payload, str(raw["raw_dir"]))
        stable_id = str(raw.get("stable_id") or raw["raw_dir"]).strip()
        instruction = str(raw.get("instruction") or "").strip()
        label_filename = str(raw.get("label_file") or "subtask_labels.json").strip()
        if not stable_id:
            raise ValueError(f"{manifest_path}: episode {position} has an empty stable_id")
        if not instruction:
            raise ValueError(f"{manifest_path}: included {stable_id} has no instruction")
        if not label_filename or pathlib.Path(label_filename).name != label_filename:
            raise ValueError(f"{manifest_path}: {stable_id} has invalid label_file {label_filename!r}")
        if demo in seen_raw_dirs:
            raise ValueError(f"{manifest_path}: duplicate raw_dir {demo}")
        if stable_id in seen_stable_ids:
            raise ValueError(f"{manifest_path}: duplicate stable_id {stable_id!r}")
        seen_raw_dirs.add(demo)
        seen_stable_ids.add(stable_id)
        included.append(
            {
                **raw,
                "raw_dir": demo,
                "manifest_raw_dir": str(raw["raw_dir"]),
                "stable_id": stable_id,
                "instruction": instruction,
                "label_file": label_filename,
                "manifest_position": position,
            }
        )
    return payload, included


def _preflight_manifest(
    manifest_path: pathlib.Path,
    payload: dict[str, Any],
    episodes: list[dict[str, Any]],
    task_vocabulary: tuple[str, ...] | None,
) -> tuple[list[dict[str, Any]], int]:
    """Validate all included episodes before output deletion or video decoding."""
    expected = payload.get("expected", {})
    expected_episodes = expected.get("included_episodes")
    if expected_episodes is not None and int(expected_episodes) != len(episodes):
        raise ValueError(f"{manifest_path}: expected {expected_episodes} included episodes, found {len(episodes)}")
    for raw_index, stable_id in expected.get("episode_index_to_stable_id", {}).items():
        index = int(raw_index)
        if index < 0 or index >= len(episodes):
            raise ValueError(f"{manifest_path}: asserted episode index {index} is out of range")
        if episodes[index]["stable_id"] != stable_id:
            raise ValueError(
                f"{manifest_path}: episode {index} is {episodes[index]['stable_id']!r}, expected {stable_id!r}"
            )

    total_frames = 0
    checked: list[dict[str, Any]] = []
    waiting_config = payload.get("memory_waiting_core_config", {})
    waiting_max_step = float(waiting_config.get("max_state_step", 4e-3))
    waiting_max_excursion = float(waiting_config.get("max_state_excursion", 2e-2))
    waiting_stride = int(waiting_config.get("stride_frames", 15))
    for episode in episodes:
        demo = episode["raw_dir"]
        stable_id = episode["stable_id"]
        if not demo.is_dir():
            raise ValueError(f"{manifest_path}: {stable_id} raw directory does not exist: {demo}")
        label_file = demo / episode["label_file"]
        if not label_file.is_file():
            raise ValueError(f"{manifest_path}: {stable_id} missing {label_file.name}")
        lengths = _episode_stream_lengths(demo)
        label_sha256 = _sha256(label_file)
        declared_label_sha256 = episode.get("label_sha256")
        if declared_label_sha256 is not None and declared_label_sha256 != label_sha256:
            raise ValueError(f"{manifest_path}: {stable_id} label SHA256 differs from the approved manifest")
        declared_stream_sha256 = episode.get("raw_stream_sha256")
        if declared_stream_sha256 is not None:
            expected_streams = {*_ARRAY_STREAMS, *_VIDEO_STREAMS}
            if set(declared_stream_sha256) != expected_streams:
                raise ValueError(f"{manifest_path}: {stable_id} approved stream hash inventory is incomplete")
            for filename in (*_ARRAY_STREAMS, *_VIDEO_STREAMS):
                if _sha256(demo / filename) != declared_stream_sha256[filename]:
                    raise ValueError(f"{manifest_path}: {stable_id}/{filename} SHA256 differs from approval")
        num_frames = min(lengths.values())
        expected_frames = episode.get("expected_num_frames")
        if expected_frames is not None and int(expected_frames) != num_frames:
            raise ValueError(
                f"{manifest_path}: {stable_id} expected {expected_frames} frames, found {num_frames}: {lengths}"
            )
        subtasks = _load_frame_subtasks(demo, num_frames, episode["label_file"])
        if subtasks is None:
            raise ValueError(f"{manifest_path}: {stable_id} labels do not cover frames 0..{num_frames - 1}")
        segments = json.loads(label_file.read_text())
        if segments[-1]["end"] != num_frames - 1:
            raise ValueError(
                f"{manifest_path}: {stable_id} label end {segments[-1]['end']} != shortest stream end {num_frames - 1}"
            )
        if task_vocabulary is not None:
            _validate_raw_label_tasks(demo, task_vocabulary, episode["label_file"])
        target_side = episode.get("target_side")
        if target_side is not None:
            if target_side not in ("left", "right"):
                raise ValueError(f"{manifest_path}: {stable_id} has invalid target_side {target_side!r}")
            expected_wait = f"wait; target bin is {target_side}"
            expected_execute = f"open {target_side} bin"
            if expected_wait not in subtasks or subtasks[-1] != expected_execute:
                raise ValueError(f"{manifest_path}: {stable_id} labels disagree with target_side={target_side}")
            if expected.get("require_exact_five_phase_schema"):
                expected_phases = [
                    "open both lids",
                    "inspect both bins",
                    "close both lids and reset arms",
                    expected_wait,
                    expected_execute,
                ]
                actual_phases = [segment["task"] for segment in segments]
                if actual_phases != expected_phases:
                    raise ValueError(
                        f"{manifest_path}: {stable_id} expected five phases {expected_phases}, found {actual_phases}"
                    )
        waiting_core = episode.get("memory_waiting_core")
        if expected.get("require_memory_waiting_core") and waiting_core is None:
            raise ValueError(f"{manifest_path}: {stable_id} has no memory_waiting_core")
        waiting_audit = None
        if waiting_core is not None:
            start, end = int(waiting_core["start"]), int(waiting_core["end"])
            if start < 0 or end < start or end >= num_frames:
                raise ValueError(f"{manifest_path}: {stable_id} has invalid memory_waiting_core {start}..{end}")
            if target_side is None:
                raise ValueError(f"{manifest_path}: {stable_id} waiting core requires target_side")
            expected_wait = f"wait; target bin is {target_side}"
            if any(task != expected_wait for task in subtasks[start : end + 1]):
                raise ValueError(f"{manifest_path}: {stable_id} memory_waiting_core extends outside {expected_wait!r}")
            if expected.get("require_semantic_wait_equals_core"):
                wait_frames = [index for index, task in enumerate(subtasks) if task == expected_wait]
                if not wait_frames or wait_frames != list(range(start, end + 1)):
                    raise ValueError(
                        f"{manifest_path}: {stable_id} semantic wait does not exactly equal "
                        f"memory_waiting_core {start}..{end}"
                    )
            state = np.concatenate(
                [
                    np.load(demo / "left_joint_positions.npy", mmap_mode="r"),
                    np.load(demo / "right_joint_positions.npy", mmap_mode="r"),
                ],
                axis=1,
            )[:num_frames]
            core_state = state[start : end + 1]
            max_step = float(np.abs(np.diff(core_state, axis=0)).max()) if len(core_state) > 1 else float("inf")
            max_excursion = float((core_state.max(axis=0) - core_state.min(axis=0)).max())
            if max_step >= waiting_max_step or max_excursion > waiting_max_excursion:
                raise ValueError(
                    f"{manifest_path}: {stable_id} waiting core motion failed: "
                    f"step={max_step}, excursion={max_excursion}"
                )
            eligible = end - start + 1 >= waiting_stride
            declared_eligible = waiting_core.get("eligible_at_stride")
            if declared_eligible is not None and bool(declared_eligible) != eligible:
                raise ValueError(f"{manifest_path}: {stable_id} waiting eligibility disagrees with core length")
            waiting_audit = {
                "start": start,
                "end": end,
                "length": end - start + 1,
                "max_14d_step": max_step,
                "max_14d_excursion": max_excursion,
                "eligible_at_stride": eligible,
                "stride_frames": waiting_stride,
            }
        checked.append(
            {
                **episode,
                "num_frames": num_frames,
                "stream_lengths": lengths,
                "label_sha256": label_sha256,
                "memory_waiting_core": waiting_audit,
            }
        )
        total_frames += num_frames

    expected_frames = expected.get("included_frames")
    if expected_frames is not None and int(expected_frames) != total_frames:
        raise ValueError(f"{manifest_path}: expected {expected_frames} included frames, found {total_frames}")
    return checked, total_frames


def main(
    data_dir: str = "/iris/u/kewalk/memory_project/data/bin_memory_banana",
    *,
    data_dirs: list[str] | None = None,
    instructions: list[str] | None = None,
    task_vocabulary: list[str] | None = None,
    episode_manifest: str | None = None,
    repo_name: str = REPO_NAME,
    overwrite: bool = False,
    resume: bool = False,
    preflight_only: bool = False,
    push_to_hub: bool = False,
):
    """Convert one or more raw demo folders into a single LeRobot dataset.

    `episode_manifest` overrides directory mode and supplies the exact output order, prompt,
    source identity, and optional expected frame count for each episode.  It is the recommended
    mode for mixed-prompt collections.  `task_vocabulary`, when supplied, fixes task indices to
    the given order and rejects labels outside that vocabulary; manifest mode can carry the same
    list and verifies that a CLI override is identical.
    """
    manifest_path: pathlib.Path | None = None
    manifest_payload: dict[str, Any] | None = None
    strict_manifest_mode = episode_manifest is not None
    conversion_specs: list[dict[str, Any]] = []
    expected_total_frames: int | None = None

    if strict_manifest_mode:
        if data_dirs is not None or instructions is not None:
            raise ValueError("episode_manifest cannot be combined with data_dirs or instructions")
        manifest_path = pathlib.Path(episode_manifest).resolve()
        manifest_payload, conversion_specs = _load_episode_manifest(manifest_path)
        manifest_vocabulary = manifest_payload.get("task_vocabulary")
        if (
            manifest_vocabulary is not None
            and task_vocabulary is not None
            and list(manifest_vocabulary) != list(task_vocabulary)
        ):
            raise ValueError("CLI task_vocabulary does not exactly match the episode manifest")
        vocabulary_source = task_vocabulary if task_vocabulary is not None else manifest_vocabulary
        validated_task_vocabulary = _validate_task_vocabulary(vocabulary_source)
        conversion_specs, expected_total_frames = _preflight_manifest(
            manifest_path, manifest_payload, conversion_specs, validated_task_vocabulary
        )
        print(
            f"Manifest preflight passed: {len(conversion_specs)} episodes / "
            f"{expected_total_frames} frames from {manifest_path}"
        )
    else:
        source_paths = [pathlib.Path(d) for d in (data_dirs if data_dirs else [data_dir])]
        if instructions is not None and len(instructions) != len(source_paths):
            raise ValueError(f"got {len(instructions)} instructions for {len(source_paths)} data dirs.")
        validated_task_vocabulary = _validate_task_vocabulary(task_vocabulary)
        for source_index, data_path in enumerate(source_paths):
            if not data_path.is_dir():
                raise ValueError(f"data directory does not exist: {data_path}")
            demo_dirs = sorted(
                (p for p in data_path.iterdir() if p.is_dir() and p.name.startswith("demo")),
                key=_natural_demo_key,
            )
            print(f"Found {len(demo_dirs)} demos in {data_path}")
            for demo in demo_dirs:
                label_file = demo / "subtask_labels.json"
                if not label_file.exists():
                    print(f"  skipping {demo.name}: no subtask_labels.json")
                    continue
                if validated_task_vocabulary is not None:
                    _validate_raw_label_tasks(demo, validated_task_vocabulary)
                conversion_specs.append(
                    {
                        "raw_dir": demo,
                        "label_file": "subtask_labels.json",
                        "instruction": instructions[source_index] if instructions is not None else None,
                        "stable_id": f"{data_path.name}/{demo.name}",
                    }
                )

    if preflight_only:
        if not strict_manifest_mode:
            raise ValueError("preflight_only requires episode_manifest")
        return

    if resume and not strict_manifest_mode:
        raise ValueError("resume requires episode_manifest")
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")

    # No existing output is touched until every strict-manifest episode passes preflight.
    output_path = HF_LEROBOT_HOME / repo_name
    resume_from = 0
    converted_frames = 0
    if resume:
        resume_from, converted_frames = _validate_resume_prefix(
            output_path, conversion_specs, validated_task_vocabulary
        )
        dataset = _open_dataset_for_append(repo_name, output_path)
        if dataset.num_episodes != resume_from:
            raise RuntimeError(f"loaded resume dataset has {dataset.num_episodes} episodes, expected {resume_from}")
        # Thread-only writes avoid subprocess memory multiplication while keeping recovery practical.
        print(f"resuming at episode {resume_from} / frame {converted_frames} with 8 thread-only image writers")
    elif output_path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output_path}; pass --overwrite to replace it")
        shutil.rmtree(output_path)
    if not resume:
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
        _register_task_vocabulary(dataset, validated_task_vocabulary)

    if strict_manifest_mode:
        episode_prompts = {str(index): str(spec["instruction"]) for index, spec in enumerate(conversion_specs)}
        episode_sources = {str(index): _strict_source_record(spec) for index, spec in enumerate(conversion_specs)}
        memory_waiting_cores = {
            str(index): spec["memory_waiting_core"]
            for index, spec in enumerate(conversion_specs)
            if spec.get("memory_waiting_core") is not None
        }
    else:
        episode_prompts: dict[str, str] = {}
        episode_sources: dict[str, dict[str, Any]] = {}
        memory_waiting_cores: dict[str, dict[str, Any]] = {}
    for spec_index, spec in enumerate(conversion_specs):
        if spec_index < resume_from:
            continue
        demo = pathlib.Path(spec["raw_dir"])
        label_filename = str(spec.get("label_file") or "subtask_labels.json")
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
            if strict_manifest_mode:
                raise RuntimeError(f"{spec['stable_id']}: decoded zero frames after a passing preflight")
            print(f"  skipping {demo.name}: no frames")
            continue
        if strict_manifest_mode and num_frames != spec["num_frames"]:
            raise RuntimeError(
                f"{spec['stable_id']}: decoded {num_frames} frames, preflight found {spec['num_frames']}"
            )

        subtasks = _load_frame_subtasks(demo, num_frames, label_filename)
        if subtasks is None:
            if strict_manifest_mode:
                raise RuntimeError(f"{spec['stable_id']}: labels became incomplete after preflight")
            print(f"  skipping {demo.name}: incomplete {label_filename}")
            continue
        print(
            f"  [{dataset.num_episodes:03d}] {spec['stable_id']}: {num_frames} frames, {len(set(subtasks))} subtasks",
            flush=True,
        )

        output_episode_index = int(dataset.num_episodes)
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
        instruction = spec.get("instruction")
        if instruction is not None:
            episode_prompts[str(output_episode_index)] = str(instruction)
        dataset.save_episode()
        converted_frames += num_frames

        if strict_manifest_mode and output_episode_index != spec_index:
            raise RuntimeError(
                f"episode identity drift while converting {spec['stable_id']}: "
                f"dataset index={output_episode_index}, manifest index={spec_index}"
            )

    if strict_manifest_mode:
        assert manifest_path is not None
        assert manifest_payload is not None
        if len(episode_sources) != len(conversion_specs) or converted_frames != expected_total_frames:
            raise RuntimeError(
                f"conversion count mismatch: got {len(episode_sources)} episodes / {converted_frames} frames"
            )
        if len(episode_prompts) != len(conversion_specs):
            raise RuntimeError(
                f"prompt count mismatch: got {len(episode_prompts)} for {len(conversion_specs)} episodes"
            )
        _atomic_json(output_path / "meta" / "episode_prompts.json", episode_prompts)
        _atomic_json(output_path / "meta" / "episode_sources.json", episode_sources)
        if memory_waiting_cores:
            _atomic_json(output_path / "meta" / "memory_waiting_cores.json", memory_waiting_cores)
        _atomic_json(
            output_path / "meta" / "conversion_report.json",
            {
                "schema_version": 1,
                "repo_name": repo_name,
                "episode_manifest": str(manifest_path),
                "episode_manifest_sha256": _sha256(manifest_path),
                "review_status": manifest_payload.get("review_status"),
                "included_episodes": len(episode_sources),
                "included_frames": converted_frames,
                "excluded_episodes": [item for item in manifest_payload["episodes"] if not item.get("include", True)],
                "task_vocabulary": list(validated_task_vocabulary or ()),
            },
        )
        print(
            f"wrote strict provenance for {len(episode_sources)} episodes / "
            f"{converted_frames} frames to {output_path / 'meta'}"
        )
    elif episode_prompts:
        _atomic_json(output_path / "meta" / "episode_prompts.json", episode_prompts)
        print(f"wrote {len(episode_prompts)} episode prompts to {output_path / 'meta' / 'episode_prompts.json'}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["yam", "bimanual"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
