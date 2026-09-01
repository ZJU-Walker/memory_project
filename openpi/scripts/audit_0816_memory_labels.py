#!/usr/bin/env python3
"""Read-only audit of the 60-demo 0816 memory labels.

This reproduces the v3.4.1 production waiting-core rule on the raw 14-D
follower state.  It deliberately never rewrites ``subtask_labels.json``.
Semantic review recommendations and kinematic/static-core measurements are
kept as separate fields in the output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

import cv2
import numpy as np

OPENPI_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = OPENPI_ROOT.parent
DEFAULT_DATA_ROOT = WORKSPACE_ROOT / "data"
DEFAULT_OUTPUT_DIR = OPENPI_ROOT / "diagnostic_outputs" / "0816_memory_label_audit_v1"
DEFAULT_REVIEW_FILE = OPENPI_ROOT / "scripts" / "0816_inspect_start_review_v1.json"

MAX_SPEED = 4e-3
MAX_EXCURSION = 2e-2
MEMORY_STRIDE = 15
ARM_DIMS = tuple(range(6)) + tuple(range(7, 13))
GRIPPER_DIMS = (6, 13)

EXPECTED_TRIMMED_EPISODES = (
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    15,
    16,
    19,
    21,
    22,
    27,
    28,
    29,
    31,
    36,
    38,
    40,
    41,
    42,
    43,
    44,
    45,
    47,
    53,
    54,
    55,
    56,
    57,
    59,
)
EXPECTED_DROPPED_WAITING_FRAMES = 317
EXPECTED_EP26_CURRENT_FILE_SHA256 = "37ef382af4a963a79ad628823c1923193ec12673d6ab614df6bbe87593587d7f"
EXPECTED_EP26_PRE_FIX_BACKUP_FILE_SHA256 = "4376faa46cf4d43670a3208e5076c3b9fadbfc0598096c355d2fde832f697637"
EXPECTED_EP26_PRE_FIX_SEMANTIC_SHA256 = "5af3be3d282d0ddd158eb13fcd68d59fd336de255c71485728566f836c48479f"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_sha256(labels: list[dict[str, Any]]) -> str:
    canonical = json.dumps(labels, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _episode_specs(data_root: Path) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for episode_index in range(60):
        is_banana = episode_index < 30
        source_name = "0816_banana" if is_banana else "0816_grey_box"
        source_episode = episode_index + 1 if is_banana else episode_index - 29
        specs.append(
            {
                "episode_index": episode_index,
                "source": source_name,
                "source_episode": source_episode,
                "demo_name": f"demo{source_episode}",
                "prompt": "find the banana" if is_banana else "find the grey pepper box",
                "demo_dir": data_root / source_name / f"demo{source_episode}",
            }
        )
    return specs


def _longest_static_run(
    window: np.ndarray, max_speed: float = MAX_SPEED, max_excursion: float = MAX_EXCURSION
) -> tuple[int, int] | None:
    """Exact copy of the v3.4.1 production rule; returns inclusive bounds."""
    if len(window) < 2:
        return None
    speed = np.abs(np.diff(window, axis=0)).max(axis=1)
    quiet = np.concatenate([[speed[0] < max_speed], speed < max_speed])
    best: tuple[int, int] | None = None
    best_len = 0
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
            span = window[left : stop + 1]
            if float((span.max(axis=0) - span.min(axis=0)).max()) <= max_excursion:
                break
            left += 1
        if stop - left + 1 >= 2 and stop - left + 1 > best_len:
            best_len = stop - left + 1
            best = (left, stop)
        start = stop + 1
    return best


def _run_algorithm_self_checks() -> list[str]:
    quiet_then_jump = np.asarray([[0.0], [0.0], [0.1], [0.1], [0.1]], dtype=np.float64)
    assert _longest_static_run(quiet_then_jump) == (0, 1)

    slow_creep = np.arange(9, dtype=np.float64)[:, None] * 0.003
    run = _longest_static_run(slow_creep)
    assert run == (2, 8), run
    span = slow_creep[run[0] : run[1] + 1]
    assert float(np.abs(np.diff(span, axis=0)).max()) < MAX_SPEED
    assert float(np.ptp(span, axis=0).max()) <= MAX_EXCURSION

    specs = _episode_specs(Path("/data"))
    assert specs[0]["demo_dir"] == Path("/data/0816_banana/demo1")
    assert specs[29]["demo_dir"] == Path("/data/0816_banana/demo30")
    assert specs[30]["demo_dir"] == Path("/data/0816_grey_box/demo1")
    assert specs[59]["demo_dir"] == Path("/data/0816_grey_box/demo30")
    return [
        "production static-run boundary semantics",
        "slow-creep excursion trimming",
        "fixed 30+30 episode ordering",
    ]


def _video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    count = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if count <= 0:
        raise RuntimeError(f"invalid frame count {count}: {path}")
    return count


def _load_review_file(path: Path, specs: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "openpi.0816.inspect_start_review.v1":
        raise ValueError(f"unexpected inspect review schema in {path}")
    rows = payload.get("reviews", [])
    by_episode = {int(row["episode_index"]): row for row in rows}
    if set(by_episode) != set(range(60)) or len(rows) != 60:
        raise ValueError("inspect review must contain exactly one row for every episode 0..59")
    for spec in specs:
        row = by_episode[spec["episode_index"]]
        expected = (spec["source"], spec["demo_name"])
        actual = (row.get("source"), row.get("demo_name"))
        if actual != expected:
            raise ValueError(f"inspect review identity mismatch for episode {spec['episode_index']}: {actual}")
        if row.get("both_objects_visible") not in (True, False):
            raise ValueError(f"inspect review is not resolved for episode {spec['episode_index']}")
    metadata = {key: value for key, value in payload.items() if key != "reviews"}
    return by_episode, metadata


def _load_labels(path: Path, authoritative_length: int) -> list[dict[str, Any]]:
    labels = json.loads(path.read_text())
    if not labels:
        raise ValueError(f"empty labels: {path}")
    cursor = 0
    for segment in labels:
        if set(segment) != {"task", "start", "end"}:
            raise ValueError(f"unexpected label fields in {path}: {segment}")
        if int(segment["start"]) != cursor or int(segment["end"]) < cursor:
            raise ValueError(f"labels do not tile contiguously in {path}: {segment}")
        cursor = int(segment["end"]) + 1
    if cursor < authoritative_length:
        raise ValueError(f"labels end at {cursor}, before usable stream length {authoritative_length}: {path}")
    return labels


def _task_bounds(labels: list[dict[str, Any]], prefix: str) -> tuple[int, int]:
    segments = [segment for segment in labels if str(segment["task"]).startswith(prefix)]
    if not segments:
        raise ValueError(f"missing task prefix {prefix!r}")
    lo = int(segments[0]["start"])
    hi = int(segments[-1]["end"])
    cursor = lo
    for segment in segments:
        if int(segment["start"]) != cursor:
            raise ValueError(f"non-contiguous segments for {prefix!r}: {segments}")
        cursor = int(segment["end"]) + 1
    return lo, hi


def _motion_metrics(window: np.ndarray, dims: tuple[int, ...], origin: np.ndarray) -> dict[str, float]:
    selected = np.asarray(window[:, dims], dtype=np.float64)
    selected_origin = np.asarray(origin[list(dims)], dtype=np.float64)
    max_step = float(np.abs(np.diff(selected, axis=0)).max()) if len(selected) >= 2 else 0.0
    return {
        "max_abs_step": max_step,
        "max_per_dim_excursion": float(np.ptp(selected, axis=0).max()),
        "max_abs_displacement_from_wait_start": float(np.abs(selected - selected_origin).max()),
    }


def _white_fraction(frame: np.ndarray, roi: tuple[int, int, int, int]) -> float:
    x0, x1, y0, y1 = roi
    patch = frame[y0:y1, x0:x1]
    lo = patch.min(axis=2)
    hi = patch.max(axis=2)
    return float(np.mean((lo > 140) & ((hi - lo) < 60)))


def _read_requested_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    wanted = sorted(set(indices))
    if not wanted:
        return {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    wanted_set = set(wanted)
    result: dict[int, np.ndarray] = {}
    frame_index = 0
    last = wanted[-1]
    while frame_index <= last:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index in wanted_set:
            result[frame_index] = frame
        frame_index += 1
    cap.release()
    missing = sorted(wanted_set - set(result))
    if missing:
        raise RuntimeError(f"failed to decode requested frames {missing} from {path}")
    return result


def _tile(frame: np.ndarray, caption: str, width: int, height: int, caption_height: int = 32) -> np.ndarray:
    image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height + caption_height, width, 3), dtype=np.uint8)
    canvas[caption_height:] = image
    cv2.putText(
        canvas,
        caption,
        (5, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return canvas


def _save_contact_sheet(tiles: list[np.ndarray], columns: int, path: Path) -> None:
    if not tiles:
        raise ValueError(f"no tiles for {path}")
    rows: list[np.ndarray] = []
    blank = np.zeros_like(tiles[0])
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        row.extend([blank.copy() for _ in range(columns - len(row))])
        rows.append(np.concatenate(row, axis=1))
    sheet = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), sheet):
        raise RuntimeError(f"failed to write {path}")


def _semantic_fix(episode_index: int, labels: list[dict[str, Any]], demo_dir: Path) -> dict[str, Any] | None:
    if episode_index != 26:
        return None
    backup_path = demo_dir / "subtask_labels_pre_v35.json"
    if not backup_path.is_file():
        raise FileNotFoundError(f"ep26 semantic-fix backup is missing: {backup_path}")
    backup_labels = json.loads(backup_path.read_text())
    if backup_labels[:-1] != labels[:-1]:
        raise ValueError("ep26 backup/current labels differ outside the intended final segment")
    backup_last = backup_labels[-1]
    current_last = labels[-1]
    expected_bounds = (595, 931)
    if (
        backup_last.get("task") != "wait; target bin is right"
        or current_last.get("task") != "open right bin"
        or (backup_last.get("start"), backup_last.get("end")) != expected_bounds
        or (current_last.get("start"), current_last.get("end")) != expected_bounds
    ):
        raise ValueError("ep26 backup/current labels do not encode the verified one-segment semantic fix")
    backup_semantic_sha256 = _semantic_sha256(backup_labels)
    if backup_semantic_sha256 != EXPECTED_EP26_PRE_FIX_SEMANTIC_SHA256:
        raise ValueError(f"ep26 backup does not match pinned pre-v3.5 labels: {backup_semantic_sha256}")
    backup_file_sha256 = _sha256_file(backup_path)
    current_file_sha256 = _sha256_file(demo_dir / "subtask_labels.json")
    if backup_file_sha256 != EXPECTED_EP26_PRE_FIX_BACKUP_FILE_SHA256:
        raise ValueError(f"ep26 backup byte hash drifted: {backup_file_sha256}")
    if current_file_sha256 != EXPECTED_EP26_CURRENT_FILE_SHA256:
        raise ValueError(f"ep26 current-label byte hash drifted: {current_file_sha256}")
    return {
        "status": "applied_with_backup",
        "issue": "final open-right phase is mislabeled as a second wait-right segment",
        "applied_change": {
            "start": 595,
            "end": 931,
            "old_task": "wait; target bin is right",
            "new_task": "open right bin",
        },
        "backup_path": str(backup_path.resolve()),
        "backup_file_sha256": backup_file_sha256,
        "current_file_sha256": current_file_sha256,
        "backup_semantic_sha256": backup_semantic_sha256,
        "current_semantic_sha256": _semantic_sha256(labels),
        "backup_segments": backup_labels,
        "current_segments": labels,
        "modified_by_this_audit": False,
    }


def _analyse_episode(spec: dict[str, Any]) -> dict[str, Any]:
    demo_dir = Path(spec["demo_dir"])
    required = [
        "left_joint_positions.npy",
        "right_joint_positions.npy",
        "left_control.npy",
        "right_control.npy",
        "top_camera_rgb.mp4",
        "left_camera_rgb.mp4",
        "right_camera_rgb.mp4",
        "subtask_labels.json",
    ]
    missing = [name for name in required if not (demo_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing files in {demo_dir}: {missing}")

    left_state = np.load(demo_dir / "left_joint_positions.npy", mmap_mode="r")
    right_state = np.load(demo_dir / "right_joint_positions.npy", mmap_mode="r")
    left_action = np.load(demo_dir / "left_control.npy", mmap_mode="r")
    right_action = np.load(demo_dir / "right_control.npy", mmap_mode="r")
    if left_state.ndim != 2 or right_state.ndim != 2 or left_state.shape[1] != 7 or right_state.shape[1] != 7:
        raise ValueError(f"unexpected follower-state shape in {demo_dir}")

    stream_lengths = {
        "left_joint_positions": len(left_state),
        "right_joint_positions": len(right_state),
        "left_control": len(left_action),
        "right_control": len(right_action),
        "top_camera_rgb": _video_frame_count(demo_dir / "top_camera_rgb.mp4"),
        "left_camera_rgb": _video_frame_count(demo_dir / "left_camera_rgb.mp4"),
        "right_camera_rgb": _video_frame_count(demo_dir / "right_camera_rgb.mp4"),
    }
    authoritative_length = min(stream_lengths.values())
    label_path = demo_dir / "subtask_labels.json"
    labels = _load_labels(label_path, authoritative_length)
    wait_lo, wait_hi_raw = _task_bounds(labels, "wait; target bin is ")
    wait_hi = min(wait_hi_raw, authoritative_length - 1)
    inspect_lo, _ = _task_bounds(labels, "inspect both bins")
    if inspect_lo >= authoritative_length or wait_lo >= authoritative_length or wait_hi < wait_lo:
        raise ValueError(f"phase bounds exceed usable streams in {demo_dir}")

    state = np.concatenate(
        [np.asarray(left_state[:authoritative_length]), np.asarray(right_state[:authoritative_length])], axis=1
    ).astype(np.float32, copy=False)
    waiting = state[wait_lo : wait_hi + 1]
    full_run = _longest_static_run(waiting)
    if full_run is None:
        raise ValueError(f"no 14-D strict static core in {demo_dir}")
    core_lo, core_hi = wait_lo + full_run[0], wait_lo + full_run[1]

    arm_run = _longest_static_run(waiting[:, ARM_DIMS])
    if arm_run is None:
        arm_core: dict[str, int] | None = None
    else:
        arm_core = {"start": wait_lo + arm_run[0], "end": wait_lo + arm_run[1], "length": arm_run[1] - arm_run[0] + 1}

    core = state[core_lo : core_hi + 1]
    original_metrics = {
        "arm": _motion_metrics(waiting, ARM_DIMS, state[wait_lo]),
        "gripper": _motion_metrics(waiting, GRIPPER_DIMS, state[wait_lo]),
    }
    core_metrics = {
        "arm": _motion_metrics(core, ARM_DIMS, state[wait_lo]),
        "gripper": _motion_metrics(core, GRIPPER_DIMS, state[wait_lo]),
    }
    core_max_step = max(core_metrics["arm"]["max_abs_step"], core_metrics["gripper"]["max_abs_step"])
    core_max_excursion = max(
        core_metrics["arm"]["max_per_dim_excursion"], core_metrics["gripper"]["max_per_dim_excursion"]
    )
    assert core_max_step < MAX_SPEED
    assert core_max_excursion <= MAX_EXCURSION + 1e-12

    return {
        "episode_index": spec["episode_index"],
        "source": spec["source"],
        "source_episode": spec["source_episode"],
        "demo_name": spec["demo_name"],
        "prompt": spec["prompt"],
        "demo_path": str(demo_dir.resolve()),
        "stream_lengths": stream_lengths,
        "authoritative_length": authoritative_length,
        "raw_label_length": int(labels[-1]["end"]) + 1,
        "raw_label_sha256": _sha256_file(label_path),
        "raw_segments": labels,
        "inspect_start": inspect_lo,
        "original_wait": {"start": wait_lo, "end": wait_hi, "length": wait_hi - wait_lo + 1},
        "strict_static_core_14d": {"start": core_lo, "end": core_hi, "length": core_hi - core_lo + 1},
        "trim": {
            "head": core_lo - wait_lo,
            "tail": wait_hi - core_hi,
            "dropped": (core_lo - wait_lo) + (wait_hi - core_hi),
        },
        "changed": (core_lo, core_hi) != (wait_lo, wait_hi),
        "eligible_at_stride_15": core_hi - core_lo + 1 >= MEMORY_STRIDE,
        "arm_only_static_core_12d": arm_core,
        "motion_metrics": {"original_wait": original_metrics, "strict_static_core_14d": core_metrics},
        "recommended_semantic_fix": _semantic_fix(spec["episode_index"], labels, demo_dir),
    }


def _boundary_indices(row: dict[str, Any]) -> list[tuple[str, int]]:
    wait = row["original_wait"]
    core = row["strict_static_core_14d"]
    return [
        ("wait_lo", wait["start"]),
        ("core_lo", core["start"]),
        ("core_lo+5", min(core["start"] + 5, core["end"])),
        ("core_hi-5", max(core["start"], core["end"] - 5)),
        ("core_hi", core["end"]),
        ("wait_hi", wait["end"]),
    ]


def _render_review_artifacts(
    episodes: list[dict[str, Any]], review_by_episode: dict[int, dict[str, Any]], output_dir: Path
) -> dict[str, Any]:
    inspect_tiles: list[np.ndarray] = []
    boundary_rows: list[np.ndarray] = []
    boundary_page_map: list[dict[str, Any]] = []
    ep26_fix_tiles: list[np.ndarray] = []
    left_roi = (190, 300, 190, 310)
    right_roi = (320, 430, 190, 300)

    for row in episodes:
        top_path = Path(row["demo_path"]) / "top_camera_rgb.mp4"
        requests = [row["inspect_start"]]
        if row["changed"]:
            requests.extend(frame for _, frame in _boundary_indices(row))
        if row["episode_index"] == 26:
            requests.extend([541, 594, 595, 600, 900, 931])
        frames = _read_requested_frames(top_path, requests)

        inspect_frame = frames[row["inspect_start"]]
        inspect_sha = hashlib.sha256(inspect_frame.tobytes()).hexdigest()
        review = review_by_episode[row["episode_index"]]
        row["inspect_start_visibility"] = {
            "both_objects_visible": bool(review["both_objects_visible"]),
            "review_note": review.get("note", ""),
            "decision_source": "versioned visual-review manifest; image metrics are diagnostic only",
            "decoded_bgr_sha256": inspect_sha,
            "diagnostic_white_fraction": {
                "left_fixed_roi": _white_fraction(inspect_frame, left_roi),
                "right_fixed_roi": _white_fraction(inspect_frame, right_roi),
                "roi_format": "[x0, x1, y0, y1]",
                "left_roi": list(left_roi),
                "right_roi": list(right_roi),
                "warning": "uncalibrated trace metric; not an object-visibility classifier",
            },
        }
        visibility_status = "PASS" if review["both_objects_visible"] else "REVIEW"
        caption = (
            f"ep{row['episode_index']:02d} {row['source'].replace('0816_', '')} "
            f"f{row['inspect_start']} {visibility_status}"
        )
        inspect_tiles.append(_tile(inspect_frame, caption, 256, 192, 28))

        if row["changed"]:
            tiles = []
            for name, frame_index in _boundary_indices(row):
                caption = f"ep{row['episode_index']:02d} {name} f{frame_index}"
                tiles.append(_tile(frames[frame_index], caption, 224, 168, 30))
            boundary_rows.append(np.concatenate(tiles, axis=1))

        if row["episode_index"] == 26:
            fix_frames = [
                ("current_wait_lo", 541),
                ("current_wait_hi", 594),
                ("current_open_lo", 595),
                ("current_open_lo+5", 600),
                ("current_open", 900),
                ("current_open_hi", 931),
            ]
            ep26_fix_tiles = [
                _tile(frames[frame_index], f"ep26 {name} f{frame_index}", 224, 168, 30)
                for name, frame_index in fix_frames
            ]

    inspect_path = output_dir / "inspect_start_all60.png"
    _save_contact_sheet(inspect_tiles, 6, inspect_path)

    boundary_dir = output_dir / "waiting_boundary_sheets"
    boundary_dir.mkdir(parents=True, exist_ok=True)
    rows_per_page = 8
    changed_rows = [row for row in episodes if row["changed"]]
    for page_index, start in enumerate(range(0, len(boundary_rows), rows_per_page), start=1):
        page_rows = boundary_rows[start : start + rows_per_page]
        width = page_rows[0].shape[1]
        if len(page_rows) < rows_per_page:
            blank = np.zeros_like(page_rows[0])
            page_rows.extend([blank.copy() for _ in range(rows_per_page - len(page_rows))])
        page = np.concatenate(page_rows, axis=0)
        assert page.shape[1] == width
        path = boundary_dir / f"waiting_boundaries_{page_index:02d}.png"
        if not cv2.imwrite(str(path), page):
            raise RuntimeError(f"failed to write {path}")
        page_episodes = [row["episode_index"] for row in changed_rows[start : start + rows_per_page]]
        boundary_page_map.append({"file": str(path.relative_to(output_dir)), "episodes": page_episodes})

    ep26_fix_path = output_dir / "ep26_semantic_fix_applied.png"
    _save_contact_sheet(ep26_fix_tiles, 6, ep26_fix_path)

    return {
        "inspect_contact_sheet": str(inspect_path.relative_to(output_dir)),
        "boundary_contact_sheets": boundary_page_map,
        "ep26_semantic_fix_sheet": str(ep26_fix_path.relative_to(output_dir)),
    }


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# 0816 memory-label audit v1",
        "",
        "This audit itself is read-only: it did not modify any raw label. The 14-D strict static core is a kinematic training boundary, not a replacement for semantic phase labels.",
        "",
        "## Summary",
        "",
        f"- Episodes audited: **{summary['episodes']}**",
        f"- Strict-core changes: **{summary['trimmed_episodes']}/60**",
        f"- Waiting frames outside strict cores: **{summary['dropped_waiting_frames']}**",
        f"- Eligible at stride 15: **{summary['eligible_at_stride_15']}/60**",
        f"- Inspect-start frames with both objects visible: **{summary['inspect_both_objects_visible']}/60**",
        f"- Semantic fix already applied with verified backup: episodes **{summary['semantic_fix_applied_with_backup_episodes']}**",
        "",
        "The fixed-ROI whiteness values in `audit.json` are uncalibrated trace metrics only. Visibility decisions come from the versioned visual-review manifest and the exact-frame contact sheet.",
        "",
        "## Episode table",
        "",
        "| ep | raw demo | wait | strict 14-D core | trim H/T | stride15 | arm-only core | arm step/excursion (wait) | gripper step/excursion (wait) | inspect |",
        "|---:|:---|:---|:---|:---|:---:|:---|:---|:---|:---:|",
    ]
    for row in report["episodes"]:
        wait = row["original_wait"]
        core = row["strict_static_core_14d"]
        arm_core = row["arm_only_static_core_12d"]
        arm_core_text = "none" if arm_core is None else f"{arm_core['start']}-{arm_core['end']}"
        arm = row["motion_metrics"]["original_wait"]["arm"]
        grip = row["motion_metrics"]["original_wait"]["gripper"]
        inspect = "yes" if row["inspect_start_visibility"]["both_objects_visible"] else "NO"
        lines.append(
            f"| {row['episode_index']} | {row['source']}/{row['demo_name']} | {wait['start']}-{wait['end']} | "
            f"{core['start']}-{core['end']} | {row['trim']['head']}/{row['trim']['tail']} | "
            f"{'yes' if row['eligible_at_stride_15'] else 'NO'} | {arm_core_text} | "
            f"{arm['max_abs_step']:.5f}/{arm['max_per_dim_excursion']:.5f} | "
            f"{grip['max_abs_step']:.5f}/{grip['max_per_dim_excursion']:.5f} | {inspect} |"
        )
    lines.extend(
        [
            "",
            "## Review artifacts",
            "",
            f"- Inspect-start contact sheet: `{report['artifacts']['inspect_contact_sheet']}`",
        ]
    )
    lines.extend(
        f"- `{page['file']}`: episodes {page['episodes']}" for page in report["artifacts"]["boundary_contact_sheets"]
    )
    lines.extend(
        [
            "",
            "## Episode 26",
            "",
            "Episode 26 (`0816_banana/demo27`) was corrected before this audit: frames 595-931 now read `open right bin`. The pre-v3.5 duplicate-wait label is preserved as `subtask_labels_pre_v35.json`; `audit.json` verifies its pinned semantic hash and records both file hashes. This audit did not apply or rewrite that correction.",
            f"Review sheet: `{report['artifacts']['ep26_semantic_fix_sheet']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_artifact_manifest(output_dir: Path) -> None:
    paths = sorted(p for p in output_dir.rglob("*") if p.is_file() and p.name != "artifact_manifest.json")
    rows = [
        {
            "path": str(path.relative_to(output_dir)),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in paths
    ]
    payload = {"schema_version": "openpi.diagnostic_artifact_manifest.v1", "files": rows}
    (output_dir / "artifact_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def run_audit(data_root: Path, output_dir: Path, review_file: Path, *, allow_baseline_drift: bool) -> dict[str, Any]:
    self_checks = _run_algorithm_self_checks()
    specs = _episode_specs(data_root)
    review_by_episode, review_metadata = _load_review_file(review_file, specs)
    protected_label_paths = [Path(spec["demo_dir"]) / "subtask_labels.json" for spec in specs]
    protected_label_paths.append(data_root / "0816_banana" / "demo27" / "subtask_labels_pre_v35.json")
    label_hashes_before = {str(path.resolve()): _sha256_file(path) for path in protected_label_paths}
    episodes = [_analyse_episode(spec) for spec in specs]

    trimmed = tuple(row["episode_index"] for row in episodes if row["changed"])
    dropped = sum(row["trim"]["dropped"] for row in episodes)
    ineligible = [row["episode_index"] for row in episodes if not row["eligible_at_stride_15"]]
    if not allow_baseline_drift:
        assert trimmed == EXPECTED_TRIMMED_EPISODES, trimmed
        assert dropped == EXPECTED_DROPPED_WAITING_FRAMES, dropped
        assert ineligible == [22], ineligible

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite versioned audit directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        artifacts = _render_review_artifacts(episodes, review_by_episode, staging_dir)
        label_hashes_after = {str(path.resolve()): _sha256_file(path) for path in protected_label_paths}
        assert label_hashes_before == label_hashes_after, "a raw label changed while the audit was running"
        self_checks.extend(
            [
                "all 60 canonical raw labels and the ep26 pre-v3.5 backup remained byte-identical",
                "all strict cores satisfy step < 4e-3 and excursion <= 0.02",
                "inspect review manifest covers the fixed 30+30 episode ordering",
                "ep26 backup semantically matches the pinned duplicate-wait pre-v3.5 label",
            ]
        )
        summary = {
            "episodes": len(episodes),
            "trimmed_episodes": len(trimmed),
            "trimmed_episode_indices": list(trimmed),
            "head_trimmed_episodes": [row["episode_index"] for row in episodes if row["trim"]["head"] > 0],
            "tail_trimmed_episodes": [row["episode_index"] for row in episodes if row["trim"]["tail"] > 0],
            "dropped_waiting_frames": dropped,
            "eligible_at_stride_15": sum(row["eligible_at_stride_15"] for row in episodes),
            "ineligible_at_stride_15_episode_indices": ineligible,
            "inspect_both_objects_visible": sum(
                row["inspect_start_visibility"]["both_objects_visible"] for row in episodes
            ),
            "inspect_not_both_visible_episode_indices": [
                row["episode_index"] for row in episodes if not row["inspect_start_visibility"]["both_objects_visible"]
            ],
            "semantic_fix_recommendation_episodes": [],
            "semantic_fix_applied_with_backup_episodes": [
                row["episode_index"]
                for row in episodes
                if row["recommended_semantic_fix"] is not None
                and row["recommended_semantic_fix"]["status"] == "applied_with_backup"
            ],
            "semantic_labels_modified_by_this_audit": False,
        }
        report = {
            "schema_version": "openpi.0816_memory_label_audit.v1",
            "audit_date": "2026-08-30",
            "mode": "read_only",
            "episode_ordering": [
                "episodes 0-29 = 0816_banana/demo1-demo30",
                "episodes 30-59 = 0816_grey_box/demo1-demo30",
            ],
            "static_core_config": {
                "state_dimensions": 14,
                "dimension_order": "left_joint_positions[0:7] + right_joint_positions[0:7]",
                "max_abs_step": MAX_SPEED,
                "max_abs_step_comparator": "strictly less than",
                "max_per_dimension_excursion": MAX_EXCURSION,
                "max_excursion_comparator": "less than or equal",
                "memory_stride_frames": MEMORY_STRIDE,
                "implementation_reference": str((OPENPI_ROOT / "src/openpi/training/data_loader.py").resolve())
                + ":412-520",
            },
            "inspect_review": {
                **review_metadata,
                "manifest_path": str(review_file.resolve()),
                "manifest_sha256": _sha256_file(review_file),
                "decision_warning": "fixed-ROI whiteness is diagnostic only, not an object detector",
            },
            "summary": summary,
            "self_checks_passed": self_checks,
            "artifacts": artifacts,
            "episodes": episodes,
        }
        (staging_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        (staging_dir / "report.md").write_text(_markdown_report(report))
        _write_artifact_manifest(staging_dir)
        staging_dir.replace(output_dir)
        return report
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--inspect-review", type=Path, default=DEFAULT_REVIEW_FILE)
    parser.add_argument(
        "--allow-baseline-drift",
        action="store_true",
        help="report changed inputs without enforcing the pinned v1 counts (38 trims, 317 frames, ep22 only)",
    )
    parser.add_argument("--self-check-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_check_only:
        checks = _run_algorithm_self_checks()
        print(json.dumps({"status": "pass", "checks": checks}, indent=2))
        return
    report = run_audit(
        args.data_root,
        args.output_dir,
        args.inspect_review,
        allow_baseline_drift=args.allow_baseline_drift,
    )
    print(json.dumps({"status": "pass", "output_dir": str(args.output_dir), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
