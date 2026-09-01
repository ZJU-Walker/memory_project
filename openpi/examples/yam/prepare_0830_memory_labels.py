"""Prepare five-phase labels and review grids for the two 0830 raw collections.

The automatic result is first saved as subtask_labels_autogen.json.  With --write-final it is
also atomically installed as subtask_labels.json.  The generated grids and canonical browser
labeler remain the final human review surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np

TOP_VIDEO = "top_camera_rgb.mp4"
FINAL_LABEL = "subtask_labels.json"
AUTOGEN_LABEL = "subtask_labels_autogen.json"
BACKUP_LABEL = "subtask_labels_pre_0830_autogen_v1.json"
LEFT_BASKET = (190, 300, 190, 310)
RIGHT_BASKET = (320, 430, 190, 300)
VISIBLE_THRESHOLD = 0.20
MAX_STATE_STEP = 4e-3
MAX_STATE_EXCURSION = 2e-2
DELTAS = (-10, -5, -1, 0, 1, 5, 10)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _natural_key(path: Path) -> int:
    match = re.search(r"(\d+)$", path.name)
    return int(match.group(1)) if match else 0


def _white_fraction(image: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x0, x1, y0, y1 = box
    region = image[y0:y1, x0:x1].astype(np.int16)
    lo, hi = region.min(axis=2), region.max(axis=2)
    return float(((lo > 140) & (hi - lo < 60)).mean())


def _runs(mask: np.ndarray, bridge: int = 0, minimum: int = 1) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], np.asarray(mask, dtype=bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    raw = [(int(edges[i]), int(edges[i + 1] - 1)) for i in range(0, len(edges), 2)]
    merged: list[tuple[int, int]] = []
    for start, end in raw:
        if merged and start - merged[-1][1] - 1 <= bridge:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return [(a, b) for a, b in merged if b - a + 1 >= minimum]


def _longest_static_run(state: np.ndarray) -> tuple[int, int] | None:
    if len(state) < 2:
        return None
    speed = np.abs(np.diff(state, axis=0)).max(axis=1)
    quiet = np.concatenate([[speed[0] < MAX_STATE_STEP], speed < MAX_STATE_STEP])
    best = None
    best_length = 0
    start = 0
    while start < len(quiet):
        if not quiet[start]:
            start += 1
            continue
        stop = start
        while stop + 1 < len(quiet) and quiet[stop + 1]:
            stop += 1
        left = start
        while left <= stop:
            span = state[left : stop + 1]
            if float((span.max(axis=0) - span.min(axis=0)).max()) <= MAX_STATE_EXCURSION:
                break
            left += 1
        if stop - left + 1 >= 2 and stop - left + 1 > best_length:
            best = (left, stop)
            best_length = stop - left + 1
        start = stop + 1
    return best


def _read_scores(demo: Path) -> tuple[int, float, np.ndarray]:
    cap = cv2.VideoCapture(str(demo / TOP_VIDEO))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    scores = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        scores.append((_white_fraction(frame, LEFT_BASKET), _white_fraction(frame, RIGHT_BASKET)))
    cap.release()
    if not scores:
        raise ValueError(f"{demo}: top video has no frames")
    return len(scores), fps, np.asarray(scores, dtype=np.float32)


def _stream_lengths(demo: Path) -> dict[str, int]:
    result = {"metadata": int(json.loads((demo / "metadata.json").read_text())["num_steps"])}
    for name in (
        "left_joint_positions.npy",
        "right_joint_positions.npy",
        "left_control.npy",
        "right_control.npy",
        "left_joint_velocities.npy",
        "right_joint_velocities.npy",
        "left_gripper_position.npy",
        "right_gripper_position.npy",
    ):
        result[name] = len(np.load(demo / name, mmap_mode="r"))
    for name in ("top_camera_rgb.mp4", "left_camera_rgb.mp4", "right_camera_rgb.mp4"):
        cap = cv2.VideoCapture(str(demo / name))
        result[name] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
    return result


def _phase_labels(side: str) -> tuple[str, ...]:
    return (
        "open both lids",
        "inspect both bins",
        "close both lids and reset arms",
        f"wait; target bin is {side}",
        f"open {side} bin",
    )


def _propose(workspace: Path, spec: dict[str, Any]) -> dict[str, Any]:
    demo = workspace / spec["raw_dir"]
    lengths = _stream_lengths(demo)
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{demo}: stream mismatch {lengths}")
    num_frames, fps, scores = _read_scores(demo)
    if num_frames != next(iter(lengths.values())):
        raise ValueError(f"{demo}: decoded frames do not match streams")

    visible_runs = _runs(scores.min(axis=1) > VISIBLE_THRESHOLD, minimum=20)
    if not visible_runs:
        raise ValueError(f"{demo}: no both-visible run")
    visible_run = max(visible_runs, key=lambda item: item[1] - item[0])
    inspect_start, visible_end = visible_run

    state = np.concatenate(
        [np.load(demo / "left_joint_positions.npy"), np.load(demo / "right_joint_positions.npy")],
        axis=1,
    ).astype(np.float32)[:num_frames]
    step = np.concatenate([[0.0], np.abs(np.diff(state, axis=0)).max(axis=1)])
    motion_runs = _runs(step > MAX_STATE_STEP, bridge=3, minimum=3)
    close_run = next((run for run in motion_runs if run[0] <= visible_end <= run[1]), None)
    if close_run is None:
        raise ValueError(f"{demo}: no closing run contains visible end {visible_end}")
    close_start = close_run[0]

    target_side = str(spec["target_side"])
    detected_side = "left" if scores[-1, 0] > scores[-1, 1] else "right"
    if detected_side != target_side:
        raise ValueError(f"{demo}: final side {detected_side} != manifest {target_side}")
    side_index = 0 if target_side == "left" else 1
    terminal_runs = [
        run
        for run in _runs(scores[:, side_index] > VISIBLE_THRESHOLD, bridge=8, minimum=10)
        if run[0] > visible_end + 30
    ]
    if not terminal_runs:
        raise ValueError(f"{demo}: no terminal open run")
    terminal_run = max(terminal_runs, key=lambda item: item[1])
    if terminal_run[1] < num_frames - 10:
        raise ValueError(f"{demo}: terminal open does not reach tail")

    search_start = close_run[1] + 1
    static = _longest_static_run(state[search_start : terminal_run[0]])
    if static is None:
        raise ValueError(f"{demo}: no strict waiting plateau")
    wait_start, wait_end = search_start + static[0], search_start + static[1]
    execute_start = wait_end + 1
    wait_state = state[wait_start : wait_end + 1]
    wait_step = float(np.abs(np.diff(wait_state, axis=0)).max()) if len(wait_state) > 1 else math.inf
    wait_excursion = float((wait_state.max(axis=0) - wait_state.min(axis=0)).max())
    if wait_step >= MAX_STATE_STEP or wait_excursion > MAX_STATE_EXCURSION:
        raise ValueError(f"{demo}: waiting motion invariant failed")

    left_base = np.median(state[wait_start : wait_end + 1, :6], axis=0)
    right_base = np.median(state[wait_start : wait_end + 1, 7:13], axis=0)
    left_departure = float(np.abs(state[execute_start:, :6] - left_base).max())
    right_departure = float(np.abs(state[execute_start:, 7:13] - right_base).max())
    motion_side = "left" if left_departure > right_departure else "right"
    if motion_side != target_side:
        raise ValueError(f"{demo}: active arm {motion_side} != target {target_side}")

    names = _phase_labels(target_side)
    segments = [
        {"task": names[0], "start": 0, "end": inspect_start - 1},
        {"task": names[1], "start": inspect_start, "end": close_start - 1},
        {"task": names[2], "start": close_start, "end": wait_start - 1},
        {"task": names[3], "start": wait_start, "end": wait_end},
        {"task": names[4], "start": execute_start, "end": num_frames - 1},
    ]
    return {
        **spec,
        "num_frames": num_frames,
        "fps_header": fps,
        "segments": segments,
        "boundaries": {
            "inspect_start": inspect_start,
            "close_start": close_start,
            "wait_start": wait_start,
            "execute_start": execute_start,
            "visible_run": list(visible_run),
            "terminal_open_run": list(terminal_run),
        },
        "audit": {
            "wait_length": wait_end - wait_start + 1,
            "wait_max_14d_step": wait_step,
            "wait_max_14d_excursion": wait_excursion,
            "left_execute_departure": left_departure,
            "right_execute_departure": right_departure,
            "detected_final_side": detected_side,
            "detected_motion_side": motion_side,
        },
        "source_hashes": {
            name: _sha256(demo / name)
            for name in ("metadata.json", TOP_VIDEO, "left_joint_positions.npy", "right_joint_positions.npy")
        },
    }


def _read_frame(path: Path, index: int, size: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    index = int(np.clip(index, 0, count - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return cv2.resize(frame, size)


def _boundary_sheet(workspace: Path, records: list[dict[str, Any]], boundary: str, output: Path) -> None:
    rows = []
    for record in records:
        demo = workspace / record["raw_dir"]
        center = int(record["boundaries"][boundary])
        tiles = []
        for delta in DELTAS:
            index = int(np.clip(center + delta, 0, record["num_frames"] - 1))
            tile = _read_frame(demo / TOP_VIDEO, index, (240, 180))
            cv2.rectangle(tile, (0, 0), (239, 27), (0, 0, 0), -1)
            cv2.putText(
                tile,
                f"{demo.name} f{index} ({delta:+d})",
                (4, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)
        rows.append(np.concatenate(tiles, axis=1))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 93])


def _object_sheet(workspace: Path, records: list[dict[str, Any]], output: Path) -> None:
    tiles = []
    for record in records:
        demo = workspace / record["raw_dir"]
        inspect = record["segments"][1]
        index = (inspect["start"] + inspect["end"]) // 2
        tile = _read_frame(demo / TOP_VIDEO, index, (400, 300))
        cv2.rectangle(tile, (0, 0), (399, 42), (0, 0, 0), -1)
        prompt = "banana" if record["instruction"] == "find the banana" else "grey"
        cv2.putText(
            tile,
            f"{demo.name} f{index} {prompt} target={record['target_side'][0].upper()}",
            (5, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(tile)
    rows = []
    for start in range(0, len(tiles), 4):
        row = tiles[start : start + 4]
        row += [np.zeros_like(tiles[0])] * (4 - len(row))
        rows.append(np.concatenate(row, axis=1))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.concatenate(rows, axis=0), [cv2.IMWRITE_JPEG_QUALITY, 94])


def _write_artifact_manifest(output: Path) -> None:
    files = sorted(path for path in output.rglob("*") if path.is_file() and path.name != "artifact_manifest.json")
    _atomic_json(
        output / "artifact_manifest.json",
        {
            "schema_version": "openpi.diagnostic_artifact_manifest.v1",
            "files": [
                {
                    "path": str(path.relative_to(output)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in files
            ],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("../data/0830_episode_manifest_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostic_outputs/0830_memory_labels_v1"))
    parser.add_argument("--write-final", action="store_true")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[3]
    manifest = args.manifest if args.manifest.is_absolute() else (Path.cwd() / args.manifest).resolve()
    output = args.output_dir if args.output_dir.is_absolute() else (Path.cwd() / args.output_dir).resolve()
    payload = json.loads(manifest.read_text())
    specs = payload["episodes"]
    if len({item["raw_dir"] for item in specs}) != len(specs):
        raise ValueError("duplicate raw_dir in manifest")

    records = []
    excluded = []
    for spec in specs:
        demo = workspace / spec["raw_dir"]
        if not demo.is_dir():
            raise ValueError(f"missing demo {demo}")
        if not spec["include"]:
            excluded.append(spec)
            continue
        record = _propose(workspace, spec)
        records.append(record)
        _atomic_json(demo / AUTOGEN_LABEL, record["segments"])
        if args.write_final:
            final = demo / FINAL_LABEL
            if final.exists() and not (demo / BACKUP_LABEL).exists():
                (demo / BACKUP_LABEL).write_bytes(final.read_bytes())
            _atomic_json(final, record["segments"])

    collections: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        collections.setdefault(Path(record["raw_dir"]).parent.name, []).append(record)
    for name, collection_records in collections.items():
        collection_records.sort(key=lambda item: _natural_key(Path(item["raw_dir"])))
        root = output / name
        _object_sheet(workspace, collection_records, root / "object_prompt_target.jpg")
        for boundary in ("inspect_start", "close_start", "wait_start", "execute_start"):
            _boundary_sheet(workspace, collection_records, boundary, root / f"{boundary}.jpg")

    prompts = [record["instruction"] for record in records]
    sides = [record["target_side"] for record in records]
    report = {
        "schema_version": 1,
        "manifest": str(manifest),
        "write_final": args.write_final,
        "review_status": "assistant_reviewed_pending_user",
        "included_episodes": len(records),
        "included_frames": sum(record["num_frames"] for record in records),
        "prompt_counts": {name: prompts.count(name) for name in sorted(set(prompts))},
        "target_side_counts": {name: sides.count(name) for name in sorted(set(sides))},
        "excluded": excluded,
        "episodes": records,
    }
    _atomic_json(output / "report.json", report)
    _atomic_json(
        output / "review_ledger.json",
        {
            record["raw_dir"]: {
                "status": "pending_user_review",
                "reviewer": None,
                "reviewed_at": None,
                "auto_boundaries": record["boundaries"],
                "manual_boundaries": None,
                "prompt": record["instruction"],
                "target_side": record["target_side"],
            }
            for record in records
        },
    )
    _write_artifact_manifest(output)
    print(
        f"prepared {len(records)} episodes / {report['included_frames']} frames; "
        f"excluded {len(excluded)}; write_final={args.write_final}"
    )
    print(f"report: {output / 'report.json'}")


if __name__ == "__main__":
    main()
