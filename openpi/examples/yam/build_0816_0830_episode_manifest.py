"""Build the fail-closed episode manifest for the 0816 + 0830 v3.5 dataset.

The first 60 output indices intentionally retain the 0816 ordering used by v3.4.  The two
0830 directories contain mixed prompts, so their per-demo prompt and target side come from
``0830_episode_manifest_v1.json`` rather than from the directory name.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

TASK_VOCABULARY = (
    "open both lids",
    "wait; target bin is left",
    "open left bin",
    "close both lids and reset arms",
    "inspect both bins",
    "open right bin",
    "wait; target bin is right",
)
SOURCE_SPECS = (
    ("0816_banana", "find the banana"),
    ("0816_grey_box", "find the grey pepper box"),
)
EXPECTED_0816_EPISODES_PER_SOURCE = 30
WAITING_MAX_STATE_STEP = 4e-3
WAITING_MAX_STATE_EXCURSION = 2e-2
WAITING_STRIDE_FRAMES = 15
EXPECTED_INDEX_IDENTITIES = {
    "15": "0816_banana/demo16",
    "29": "0816_banana/demo30",
    "44": "0816_grey_box/demo15",
    "59": "0816_grey_box/demo30",
    "60": "0830_bin_part1/demo1",
    "75": "0830_bin_part1/demo16",
    "76": "0830_bin_part2/demo1",
    "88": "0830_bin_part2/demo13",
    "89": "0830_bin_part2/demo15",
}
ARRAY_STREAMS = (
    "left_joint_positions.npy",
    "right_joint_positions.npy",
    "left_control.npy",
    "right_control.npy",
)
VIDEO_STREAMS = (
    "top_camera_rgb.mp4",
    "left_camera_rgb.mp4",
    "right_camera_rgb.mp4",
)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _natural_demo_key(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else 0


def _video_length(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if length <= 0:
        raise ValueError(f"video has no frames: {path}")
    return length


def _stream_lengths(demo: Path) -> dict[str, int]:
    lengths = {filename: len(np.load(demo / filename, mmap_mode="r")) for filename in ARRAY_STREAMS}
    lengths.update({filename: _video_length(demo / filename) for filename in VIDEO_STREAMS})
    return lengths


def _load_complete_segments(demo: Path, label_file: str, num_frames: int) -> list[dict[str, Any]]:
    path = demo / label_file
    if not path.is_file():
        raise ValueError(f"missing label file: {path}")
    segments = json.loads(path.read_text())
    if not isinstance(segments, list) or not segments:
        raise ValueError(f"empty/invalid labels: {path}")
    expected_start = 0
    for index, segment in enumerate(segments):
        if segment.get("start") != expected_start or segment.get("end", -1) < expected_start:
            raise ValueError(f"non-contiguous segment {index} in {path}")
        if segment.get("task") not in TASK_VOCABULARY:
            raise ValueError(f"unknown task {segment.get('task')!r} in {path}")
        expected_start = int(segment["end"]) + 1
    if expected_start != num_frames:
        raise ValueError(f"{path} covers {expected_start} frames; shortest stream has {num_frames}")
    return segments


def _target_side(segments: list[dict[str, Any]], path: Path) -> str:
    final = segments[-1]["task"]
    if final == "open left bin":
        side = "left"
    elif final == "open right bin":
        side = "right"
    else:
        raise ValueError(f"{path}: final task is not a sided execute phase: {final!r}")
    if f"wait; target bin is {side}" not in {segment["task"] for segment in segments}:
        raise ValueError(f"{path}: wait and execute side are inconsistent")
    return side


def _metadata_fields(demo: Path) -> dict[str, Any]:
    metadata = json.loads((demo / "metadata.json").read_text())
    return {
        "metadata_counter": metadata.get("episode_counter"),
        "timestamp": metadata.get("start_datetime"),
    }


def _included_record(
    workspace: Path,
    demo: Path,
    instruction: str,
    label_file: str,
    extra: dict[str, Any] | None = None,
    memory_waiting_core: dict[str, Any] | None = None,
) -> dict[str, Any]:
    lengths = _stream_lengths(demo)
    num_frames = min(lengths.values())
    segments = _load_complete_segments(demo, label_file, num_frames)
    derived_side = _target_side(segments, demo / label_file)
    expected_phase_tasks = [
        "open both lids",
        "inspect both bins",
        "close both lids and reset arms",
        f"wait; target bin is {derived_side}",
        f"open {derived_side} bin",
    ]
    actual_phase_tasks = [segment["task"] for segment in segments]
    if actual_phase_tasks != expected_phase_tasks:
        raise ValueError(f"{demo}: expected exact five-phase schema {expected_phase_tasks}, found {actual_phase_tasks}")
    if extra is not None and extra.get("target_side", derived_side) != derived_side:
        raise ValueError(f"{demo}: manifest target_side={extra['target_side']} disagrees with labels={derived_side}")
    wait_task = f"wait; target bin is {derived_side}"
    wait_segments = [segment for segment in segments if segment["task"] == wait_task]
    if len(wait_segments) != 1:
        raise ValueError(f"{demo}: expected exactly one {wait_task!r} segment")
    wait_segment = wait_segments[0]
    if memory_waiting_core is None:
        memory_waiting_core = {"start": wait_segment["start"], "end": wait_segment["end"]}
    core_start = int(memory_waiting_core["start"])
    core_end = int(memory_waiting_core["end"])
    if core_start < wait_segment["start"] or core_end > wait_segment["end"]:
        raise ValueError(f"{demo}: memory waiting core is outside the semantic wait segment")
    memory_waiting_core = {
        "start": core_start,
        "end": core_end,
        "eligible_at_stride": core_end - core_start + 1 >= WAITING_STRIDE_FRAMES,
    }
    raw_dir = str(demo.relative_to(workspace))
    collection = demo.parent.name
    stable_id = f"{collection}/{demo.name}"
    return {
        "stable_id": stable_id,
        "raw_dir": raw_dir,
        "collection": collection,
        "instruction": instruction,
        "target_side": derived_side,
        "include": True,
        "label_file": label_file,
        "expected_num_frames": num_frames,
        "memory_waiting_core": memory_waiting_core,
        **_metadata_fields(demo),
        **(extra or {}),
    }


def build_manifest(
    workspace: Path,
    fresh_manifest_path: Path,
    waiting_audit_path: Path,
    label_file_0816: str,
    label_file_0830: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    waiting_audit = json.loads(waiting_audit_path.read_text())
    waiting_by_stable_id = {
        f"{episode['source']}/{episode['demo_name']}": episode for episode in waiting_audit["episodes"]
    }
    if len(waiting_by_stable_id) != 60:
        raise ValueError(f"{waiting_audit_path}: expected 60 unique 0816 episodes")
    for collection, instruction in SOURCE_SPECS:
        source = workspace / "data" / collection
        demos = sorted(
            (path for path in source.iterdir() if path.is_dir() and path.name.startswith("demo")),
            key=_natural_demo_key,
        )
        if len(demos) != EXPECTED_0816_EPISODES_PER_SOURCE:
            raise ValueError(f"{source}: expected 30 demos, found {len(demos)}")
        for demo in demos:
            stable_id = f"{collection}/{demo.name}"
            audit_episode = waiting_by_stable_id.get(stable_id)
            if audit_episode is None:
                raise ValueError(f"{waiting_audit_path}: missing {stable_id}")
            records.append(
                _included_record(
                    workspace,
                    demo,
                    instruction,
                    label_file_0816,
                    memory_waiting_core=audit_episode["strict_static_core_14d"],
                )
            )

    fresh_payload = json.loads(fresh_manifest_path.read_text())
    seen_fresh: set[str] = set()
    for fresh in fresh_payload["episodes"]:
        raw_dir = str(fresh["raw_dir"])
        if raw_dir in seen_fresh:
            raise ValueError(f"duplicate fresh raw_dir {raw_dir}")
        seen_fresh.add(raw_dir)
        demo = workspace / raw_dir
        stable_id = f"{demo.parent.name}/{demo.name}"
        if not fresh.get("include", True):
            records.append(
                {
                    "stable_id": stable_id,
                    "raw_dir": raw_dir,
                    "collection": demo.parent.name,
                    "include": False,
                    "exclude_reason": fresh["exclude_reason"],
                    **_metadata_fields(demo),
                }
            )
            continue
        records.append(
            _included_record(
                workspace,
                demo,
                str(fresh["instruction"]),
                label_file_0830,
                {
                    "banana_side": fresh["banana_side"],
                    "grey_box_side": fresh["grey_box_side"],
                    "target_side": fresh["target_side"],
                },
            )
        )

    included = [record for record in records if record["include"]]
    for raw_index, stable_id in EXPECTED_INDEX_IDENTITIES.items():
        index = int(raw_index)
        if included[index]["stable_id"] != stable_id:
            raise ValueError(f"episode {index} identity drift: {included[index]['stable_id']} != {stable_id}")
    prompt_counts = {
        prompt: sum(record["instruction"] == prompt for record in included)
        for prompt in ("find the banana", "find the grey pepper box")
    }
    side_counts = {side: sum(record["target_side"] == side for record in included) for side in ("left", "right")}
    return {
        "schema_version": 1,
        "created": "2026-08-30",
        "dataset_version": "v35",
        "review_status": "assistant_validated_pending_user",
        "raw_root": "..",
        "task_vocabulary": list(TASK_VOCABULARY),
        "memory_waiting_core_config": {
            "state_dimensions": 14,
            "max_state_step": WAITING_MAX_STATE_STEP,
            "max_state_excursion": WAITING_MAX_STATE_EXCURSION,
            "stride_frames": WAITING_STRIDE_FRAMES,
            "source_0816": str(waiting_audit_path),
            "source_0830": "strict semantic wait segments from 0830_memory_labels_v1",
        },
        "source_order": [
            "0816_banana",
            "0816_grey_box",
            "0830_bin_part1",
            "0830_bin_part2",
        ],
        "expected": {
            "included_episodes": len(included),
            "included_frames": sum(int(record["expected_num_frames"]) for record in included),
            "require_memory_waiting_core": True,
            "require_semantic_wait_equals_core": True,
            "require_exact_five_phase_schema": True,
            "prompt_counts": prompt_counts,
            "target_side_counts": side_counts,
            "episode_index_to_stable_id": EXPECTED_INDEX_IDENTITIES,
        },
        "episodes": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-manifest", type=Path, default=Path("../data/0830_episode_manifest_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("../data/0816_0830_episode_manifest_v1.json"))
    parser.add_argument(
        "--waiting-audit",
        type=Path,
        default=Path("diagnostic_outputs/0816_memory_label_audit_v1/audit.json"),
    )
    parser.add_argument("--label-file-0816", default="subtask_labels.json")
    parser.add_argument("--label-file-0830", default="subtask_labels.json")
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[3]
    fresh_manifest = args.fresh_manifest.resolve()
    waiting_audit = args.waiting_audit.resolve()
    output = args.output.resolve()
    payload = build_manifest(
        workspace,
        fresh_manifest,
        waiting_audit,
        args.label_file_0816,
        args.label_file_0830,
    )
    _atomic_json(output, payload)
    expected = payload["expected"]
    print(f"wrote {expected['included_episodes']} episodes / {expected['included_frames']} frames to {output}")


if __name__ == "__main__":
    main()
