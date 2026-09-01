#!/usr/bin/env python3
# ruff: noqa: SLF001
"""Generate read-only-canonical v3.5 static-wait label overlays for 0816.

The canonical ``subtask_labels.json`` files are inputs only.  Each generated
``subtask_labels_v35_staticwait.json`` keeps the same five task strings and
episode coverage, but assigns the strict 14-D stationary core to ``wait``:

* close/reset ends at ``strict_core.start - 1``;
* wait is exactly the inclusive strict core;
* execute starts at ``strict_core.end + 1``.

All 60 overlays are staged and validated before per-file atomic publication.
If publication fails, newly published overlays are rolled back.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import cv2
import numpy as np

from scripts import audit_0816_memory_labels as audit

DEFAULT_OUTPUT_DIR = audit.OPENPI_ROOT / "diagnostic_outputs" / "0816_v35_staticwait_overlay_v1"
DEFAULT_VISUAL_REVIEW_FILE = audit.OPENPI_ROOT / "scripts" / "0816_v35_staticwait_visual_review_v1.json"
OVERLAY_NAME = "subtask_labels_v35_staticwait.json"
SCHEMA_VERSION = "openpi.0816_v35_staticwait_overlay.v1"
EXPECTED_CHANGED_EPISODES = audit.EXPECTED_TRIMMED_EPISODES
HELDOUT_EPISODES = (15, 29, 44, 59)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _side_from_wait(task: str) -> str:
    prefix = "wait; target bin is "
    if not task.startswith(prefix):
        raise ValueError(f"not a wait-side task: {task!r}")
    side = task.removeprefix(prefix)
    if side not in {"left", "right"}:
        raise ValueError(f"unexpected target side: {side!r}")
    return side


def _validate_five_phase_labels(labels: list[dict[str, Any]], *, context: str) -> str:
    if len(labels) != 5:
        raise ValueError(f"{context}: expected exactly five segments, got {len(labels)}")
    expected_fixed = (
        "open both lids",
        "inspect both bins",
        "close both lids and reset arms",
    )
    tasks = tuple(segment["task"] for segment in labels)
    if tasks[:3] != expected_fixed:
        raise ValueError(f"{context}: unexpected first three phases: {tasks[:3]}")
    side = _side_from_wait(tasks[3])
    if tasks[4] != f"open {side} bin":
        raise ValueError(f"{context}: wait side {side!r} does not match execute task {tasks[4]!r}")
    cursor = 0
    for segment in labels:
        if set(segment) != {"task", "start", "end"}:
            raise ValueError(f"{context}: unexpected fields: {segment}")
        if segment["start"] != cursor or segment["end"] < segment["start"]:
            raise ValueError(f"{context}: segments do not tile contiguously: {segment}")
        cursor = segment["end"] + 1
    return side


def _build_overlay(
    canonical: list[dict[str, Any]], core_start: int, core_end: int, *, context: str
) -> list[dict[str, Any]]:
    _validate_five_phase_labels(canonical, context=f"{context} canonical")
    old_wait = canonical[3]
    if not (old_wait["start"] <= core_start <= core_end <= old_wait["end"]):
        raise ValueError(f"{context}: strict core {core_start}-{core_end} is outside canonical wait {old_wait}")
    overlay = copy.deepcopy(canonical)
    overlay[2]["end"] = core_start - 1
    overlay[3]["start"] = core_start
    overlay[3]["end"] = core_end
    overlay[4]["start"] = core_end + 1
    _validate_five_phase_labels(overlay, context=f"{context} overlay")
    if [segment["task"] for segment in overlay] != [segment["task"] for segment in canonical]:
        raise AssertionError(f"{context}: overlay changed the task vocabulary or order")
    if overlay[0]["start"] != canonical[0]["start"] or overlay[-1]["end"] != canonical[-1]["end"]:
        raise AssertionError(f"{context}: overlay changed episode coverage")
    return overlay


def _state_for_row(row: dict[str, Any]) -> np.ndarray:
    demo_dir = Path(row["demo_path"])
    left = np.load(demo_dir / "left_joint_positions.npy", mmap_mode="r")
    right = np.load(demo_dir / "right_joint_positions.npy", mmap_mode="r")
    length = row["authoritative_length"]
    return np.concatenate([np.asarray(left[:length]), np.asarray(right[:length])], axis=1).astype(
        np.float32, copy=False
    )


def _direct_motion_metrics(window: np.ndarray, dims: tuple[int, ...]) -> dict[str, float]:
    selected = window[:, dims]
    return {
        "max_abs_step": float(np.abs(np.diff(selected, axis=0)).max()) if len(selected) >= 2 else 0.0,
        "max_per_dim_excursion": float(np.ptp(selected, axis=0).max()),
    }


def _optional_chunk_metrics(state: np.ndarray, start: int, end: int, origin: np.ndarray) -> dict[str, Any] | None:
    if end < start:
        return None
    window = state[start : end + 1]
    return {
        "start": start,
        "end": end,
        "length": end - start + 1,
        "arm": audit._motion_metrics(window, audit.ARM_DIMS, origin),
        "gripper": audit._motion_metrics(window, audit.GRIPPER_DIMS, origin),
    }


def _analyse_overlay(row: dict[str, Any]) -> dict[str, Any]:
    context = f"episode {row['episode_index']} {row['source']}/{row['demo_name']}"
    canonical = row["raw_segments"]
    side = _validate_five_phase_labels(canonical, context=f"{context} canonical")
    core = row["strict_static_core_14d"]
    overlay = _build_overlay(canonical, core["start"], core["end"], context=context)
    overlay_side = _validate_five_phase_labels(overlay, context=f"{context} overlay")
    assert overlay_side == side

    state = _state_for_row(row)
    wait_window = state[overlay[3]["start"] : overlay[3]["end"] + 1]
    full_metrics = _direct_motion_metrics(wait_window, tuple(range(14)))
    arm_metrics = _direct_motion_metrics(wait_window, audit.ARM_DIMS)
    gripper_metrics = _direct_motion_metrics(wait_window, audit.GRIPPER_DIMS)
    if not full_metrics["max_abs_step"] < audit.MAX_SPEED:
        raise ValueError(f"{context}: overlay wait violates strict step threshold: {full_metrics}")
    if not full_metrics["max_per_dim_excursion"] <= audit.MAX_EXCURSION:
        raise ValueError(f"{context}: overlay wait violates excursion threshold: {full_metrics}")

    old_wait = row["original_wait"]
    head = core["start"] - old_wait["start"]
    tail = old_wait["end"] - core["end"]
    arm_core = row["arm_only_static_core_12d"]
    if arm_core is None:
        arm_trim = old_wait["length"]
        gripper_extra_head = 0
        gripper_extra_tail = 0
    else:
        arm_trim = (arm_core["start"] - old_wait["start"]) + (old_wait["end"] - arm_core["end"])
        gripper_extra_head = max(0, core["start"] - arm_core["start"])
        gripper_extra_tail = max(0, arm_core["end"] - core["end"])
    gripper_extra_trim = gripper_extra_head + gripper_extra_tail
    full_trim = head + tail
    canonical_wait_metrics = row["motion_metrics"]["original_wait"]
    arm_passes_original_wait = (
        canonical_wait_metrics["arm"]["max_abs_step"] < audit.MAX_SPEED
        and canonical_wait_metrics["arm"]["max_per_dim_excursion"] <= audit.MAX_EXCURSION
    )
    gripper_passes_original_wait = (
        canonical_wait_metrics["gripper"]["max_abs_step"] < audit.MAX_SPEED
        and canonical_wait_metrics["gripper"]["max_per_dim_excursion"] <= audit.MAX_EXCURSION
    )
    overlay_bytes = _json_bytes(overlay)
    return {
        "episode_index": row["episode_index"],
        "source": row["source"],
        "source_episode": row["source_episode"],
        "demo_name": row["demo_name"],
        "demo_path": row["demo_path"],
        "prompt": row["prompt"],
        "target_side": side,
        "is_heldout": row["episode_index"] in HELDOUT_EPISODES,
        "authoritative_length": row["authoritative_length"],
        "canonical_label_length": row["raw_label_length"],
        "canonical_label_path": str(Path(row["demo_path"]) / "subtask_labels.json"),
        "canonical_label_sha256": row["raw_label_sha256"],
        "overlay_label_path": str(Path(row["demo_path"]) / OVERLAY_NAME),
        "overlay_label_sha256": _sha256_bytes(overlay_bytes),
        "canonical_segments": canonical,
        "overlay_segments": overlay,
        "canonical_wait": old_wait,
        "strict_wait": {"start": core["start"], "end": core["end"], "length": core["length"]},
        "reassigned_frames": {
            "wait_head_to_close_reset": head,
            "wait_tail_to_execute": tail,
            "total": full_trim,
        },
        "strict_wait_motion": {
            "full_14d": full_metrics,
            "arm_12d": arm_metrics,
            "gripper_2d": gripper_metrics,
        },
        "reassigned_chunk_motion": {
            "head_to_close_reset": _optional_chunk_metrics(
                state, old_wait["start"], core["start"] - 1, state[old_wait["start"]]
            ),
            "tail_to_execute": _optional_chunk_metrics(
                state, core["end"] + 1, old_wait["end"], state[old_wait["start"]]
            ),
        },
        "gripper_boundary_attribution": {
            "arm_only_static_core": arm_core,
            "arm_only_trim_frames": arm_trim,
            "gripper_additional_head_frames": gripper_extra_head,
            "gripper_additional_tail_frames": gripper_extra_tail,
            "gripper_additional_trim_frames": gripper_extra_trim,
            "original_wait_arm_passes_both_thresholds": arm_passes_original_wait,
            "original_wait_gripper_passes_both_thresholds": gripper_passes_original_wait,
            "gripper_only_static_violation": arm_passes_original_wait and not gripper_passes_original_wait,
            "gripper_dominant_trim": gripper_extra_trim > arm_trim,
        },
        "semantic_risk_flags": {
            "head_frames_reclassified_as_close_reset": head > 0,
            "tail_frames_reclassified_as_execute": tail > 0,
            "execute_start_is_kinematic_not_manually_semantic": tail > 0,
            "wait_shorter_than_stride_15": core["length"] < audit.MEMORY_STRIDE,
        },
        "task_strings_and_order_unchanged": True,
        "episode_coverage_unchanged": True,
        "overlay_bytes": overlay_bytes,
    }


def _load_visual_review(path: Path, rows: list[dict[str, Any]]) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != "openpi.0816_v35_staticwait_visual_review.v1":
        raise ValueError(f"unexpected visual-review schema: {path}")
    reviews = payload.get("reviews", [])
    by_episode = {int(review["episode_index"]): review for review in reviews}
    if len(reviews) != 60 or set(by_episode) != set(range(60)):
        raise ValueError("visual review must contain exactly one row for episodes 0..59")
    for row in rows:
        review = by_episode[row["episode_index"]]
        if (review.get("source"), review.get("demo_name")) != (row["source"], row["demo_name"]):
            raise ValueError(f"visual-review identity mismatch for episode {row['episode_index']}")
        if review.get("objects_hidden_through_wait") not in (True, False):
            raise ValueError(f"unresolved wait visibility review for episode {row['episode_index']}")
    metadata = {key: value for key, value in payload.items() if key != "reviews"}
    return by_episode, metadata


def _review_indices(row: dict[str, Any]) -> list[tuple[str, int]]:
    canonical = row["canonical_segments"]
    overlay = row["overlay_segments"]
    return [
        ("old_close_hi", canonical[2]["end"]),
        ("new_close_hi", overlay[2]["end"]),
        ("new_wait_lo", overlay[3]["start"]),
        ("new_wait_hi", overlay[3]["end"]),
        ("new_exec_lo", overlay[4]["start"]),
        ("old_exec_lo", canonical[4]["start"]),
    ]


def _render_review_pages(rows: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    page_rows: list[np.ndarray] = []
    gripper_rows: list[np.ndarray] = []
    gripper_episode_indices: list[int] = []
    for row in rows:
        top_path = Path(row["demo_path"]) / "top_camera_rgb.mp4"
        indices = _review_indices(row)
        frames = audit._read_requested_frames(top_path, [frame for _, frame in indices])
        tiles = [
            audit._tile(
                frames[frame_index],
                f"ep{row['episode_index']:02d} {name} f{frame_index}",
                224,
                168,
                30,
            )
            for name, frame_index in indices
        ]
        rendered_row = np.concatenate(tiles, axis=1)
        page_rows.append(rendered_row)
        if row["gripper_boundary_attribution"]["gripper_additional_trim_frames"] > 0:
            gripper_rows.append(rendered_row)
            gripper_episode_indices.append(row["episode_index"])

    review_dir = output_dir / "overlay_boundary_review"
    review_dir.mkdir(parents=True, exist_ok=True)
    rows_per_page = 8
    page_map = []
    for page_index, start in enumerate(range(0, len(page_rows), rows_per_page), start=1):
        current = page_rows[start : start + rows_per_page]
        blank = np.zeros_like(current[0])
        current.extend(blank.copy() for _ in range(rows_per_page - len(current)))
        page = np.concatenate(current, axis=0)
        path = review_dir / f"all60_overlay_boundaries_{page_index:02d}.png"
        if not cv2.imwrite(str(path), page):
            raise RuntimeError(f"failed to write {path}")
        page_map.append(
            {
                "file": str(path.relative_to(output_dir)),
                "episodes": [row["episode_index"] for row in rows[start : start + rows_per_page]],
            }
        )

    gripper_path = output_dir / "gripper_boundary_review.png"
    audit._save_contact_sheet(gripper_rows, 1, gripper_path)
    return {
        "all60_boundary_pages": page_map,
        "gripper_boundary_page": {
            "file": str(gripper_path.relative_to(output_dir)),
            "episodes": gripper_episode_indices,
        },
    }


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "overlay_bytes"}


def _summary(rows: list[dict[str, Any]], review_by_episode: dict[int, dict[str, Any]] | None) -> dict[str, Any]:
    changed = [row["episode_index"] for row in rows if row["reassigned_frames"]["total"] > 0]
    head = [row["episode_index"] for row in rows if row["reassigned_frames"]["wait_head_to_close_reset"] > 0]
    tail = [row["episode_index"] for row in rows if row["reassigned_frames"]["wait_tail_to_execute"] > 0]
    gripper_extra = [
        row["episode_index"]
        for row in rows
        if row["gripper_boundary_attribution"]["gripper_additional_trim_frames"] > 0
    ]
    gripper_only = [
        row["episode_index"] for row in rows if row["gripper_boundary_attribution"]["gripper_only_static_violation"]
    ]
    gripper_dominant = [
        row["episode_index"] for row in rows if row["gripper_boundary_attribution"]["gripper_dominant_trim"]
    ]
    return {
        "episodes": len(rows),
        "overlays_with_changed_boundaries": len(changed),
        "changed_episode_indices": changed,
        "head_relabel_episode_indices": head,
        "head_frames_relabelled_wait_to_close_reset": sum(
            row["reassigned_frames"]["wait_head_to_close_reset"] for row in rows
        ),
        "tail_relabel_episode_indices": tail,
        "tail_frames_relabelled_wait_to_execute": sum(row["reassigned_frames"]["wait_tail_to_execute"] for row in rows),
        "total_frames_removed_from_wait": sum(row["reassigned_frames"]["total"] for row in rows),
        "stride15_ineligible_episode_indices": [
            row["episode_index"] for row in rows if row["semantic_risk_flags"]["wait_shorter_than_stride_15"]
        ],
        "gripper_additional_boundary_episode_indices": gripper_extra,
        "gripper_only_static_violation_episode_indices": gripper_only,
        "gripper_dominant_trim_episode_indices": gripper_dominant,
        "heldout_episode_indices": list(HELDOUT_EPISODES),
        "heldout_overlay_hashes": {
            str(row["episode_index"]): row["overlay_label_sha256"] for row in rows if row["is_heldout"]
        },
        "manual_visual_no_leak_passes": (
            sum(bool(review["objects_hidden_through_wait"]) for review in review_by_episode.values())
            if review_by_episode is not None
            else None
        ),
        "canonical_labels_modified_by_generator": False,
    }


def _markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# 0816 v3.5 static-wait overlays",
        "",
        "The 60 canonical `subtask_labels.json` files were read-only inputs. Each overlay keeps exactly five segments, unchanged task strings/order, unchanged episode coverage, and replaces the semantic wait bounds with the production 14-D strict static core.",
        "",
        "## Result",
        "",
        f"- Overlay files: **{summary['episodes']}/60**",
        f"- Changed boundaries: **{summary['overlays_with_changed_boundaries']}/60**",
        f"- Frames moved from wait to close/reset: **{summary['head_frames_relabelled_wait_to_close_reset']}** across {summary['head_relabel_episode_indices']}",
        f"- Frames moved from wait to execute: **{summary['tail_frames_relabelled_wait_to_execute']}** across {summary['tail_relabel_episode_indices']}",
        f"- Total removed from wait: **{summary['total_frames_removed_from_wait']}**",
        f"- Wait shorter than stride 15: **{summary['stride15_ineligible_episode_indices']}**",
        f"- Manual top-camera no-visible-object review: **{summary['manual_visual_no_leak_passes']}/60**",
        "",
        "Every overlay wait independently passed `max(abs(diff(state_14d))) < 0.004` and per-dimension excursion `<= 0.02` on production-equivalent float32 state.",
        "",
        "## Gripper attribution",
        "",
        f"- Gripper adds boundary trimming beyond the 12-D arm-only core: {summary['gripper_additional_boundary_episode_indices']}",
        f"- Arm passes the entire canonical wait while gripper alone violates: {summary['gripper_only_static_violation_episode_indices']}",
        f"- Gripper-attributed extra frames exceed arm-attributed trim: {summary['gripper_dominant_trim_episode_indices']}",
        "",
        "These are operational attributions from the difference between the 12-D arm-only and 14-D cores, not causal claims about annotation intent.",
        "",
        "## Semantic risks",
        "",
        "- Head trimming relabels detected residual motion as `close both lids and reset arms`. This is consistent with the requested static wait, but it extends a semantic phase without manual action-onset labels.",
        "- Tail trimming relabels frames as `open <side> bin` from the first frame after the kinematic core. Those frames are guaranteed to be outside strict wait, but the kinematic boundary may precede a human semantic execute onset.",
        "- Episode 22 has a valid 12-frame static wait overlay but remains ineligible for the stride-15 memory-critical branch.",
        "- Episodes 12, 16, 29, and 36 have an arm-static canonical wait whose adjustment is caused only by gripper motion under the full 14-D rule.",
        "- Episode 26 is already canonical-correct (`open right bin` at 595-931); its overlay is unchanged at wait 541-594 and no duplicate-wait fix is performed here.",
        "",
        "## Per-episode boundaries and hashes",
        "",
        "| ep | raw demo | side | canonical wait | overlay wait | H→close | T→execute | stride15 | gripper extra | overlay SHA256 |",
        "|---:|:---|:---:|:---|:---|---:|---:|:---:|---:|:---|",
    ]
    for row in report["episodes"]:
        canonical = row["canonical_wait"]
        strict = row["strict_wait"]
        reassigned = row["reassigned_frames"]
        grip = row["gripper_boundary_attribution"]
        lines.append(
            f"| {row['episode_index']} | {row['source']}/{row['demo_name']} | {row['target_side']} | "
            f"{canonical['start']}-{canonical['end']} | {strict['start']}-{strict['end']} | "
            f"{reassigned['wait_head_to_close_reset']} | {reassigned['wait_tail_to_execute']} | "
            f"{'NO' if row['semantic_risk_flags']['wait_shorter_than_stride_15'] else 'yes'} | "
            f"{grip['gripper_additional_trim_frames']} | `{row['overlay_label_sha256']}` |"
        )
    lines.extend(["", "## Review grids", ""])
    lines.extend(
        f"- `{page['file']}`: episodes {page['episodes']}"
        for page in report["review_artifacts"]["all60_boundary_pages"]
    )
    gripper = report["review_artifacts"]["gripper_boundary_page"]
    lines.append(f"- `{gripper['file']}`: gripper-sensitive episodes {gripper['episodes']}")
    lines.append("")
    return "\n".join(lines)


def _artifact_manifest(output_dir: Path) -> None:
    paths = sorted(p for p in output_dir.rglob("*") if p.is_file() and p.name != "artifact_manifest.json")
    payload = {
        "schema_version": "openpi.diagnostic_artifact_manifest.v1",
        "files": [
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": audit._sha256_file(path),
            }
            for path in paths
        ],
    }
    (output_dir / "artifact_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")


def _prepare_overlay_temp_files(rows: list[dict[str, Any]]) -> dict[Path, Path]:
    staged: dict[Path, Path] = {}
    try:
        for row in rows:
            destination = Path(row["overlay_label_path"])
            if destination.exists():
                raise FileExistsError(f"refusing to overwrite existing overlay: {destination}")
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{OVERLAY_NAME}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as stream:
                stream.write(row["overlay_bytes"])
                stream.flush()
                os.fsync(stream.fileno())
                staged[destination] = Path(stream.name)
        return staged
    except BaseException:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        raise


def _publish_overlay_temp_files(staged: dict[Path, Path]) -> list[Path]:
    published: list[Path] = []
    try:
        for destination, temporary in staged.items():
            os.replace(temporary, destination)
            published.append(destination)
        return published
    except BaseException:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        for destination in published:
            destination.unlink(missing_ok=True)
        raise


def _canonical_hashes(specs: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str((Path(spec["demo_dir"]) / "subtask_labels.json").resolve()): audit._sha256_file(
            Path(spec["demo_dir"]) / "subtask_labels.json"
        )
        for spec in specs
    }


def generate(
    data_root: Path,
    output_dir: Path,
    *,
    visual_review_file: Path | None,
    publish_overlays: bool,
) -> dict[str, Any]:
    audit._run_algorithm_self_checks()
    specs = audit._episode_specs(data_root)
    canonical_hashes_before = _canonical_hashes(specs)
    audit_rows = [audit._analyse_episode(spec) for spec in specs]
    rows = [_analyse_overlay(row) for row in audit_rows]
    changed = tuple(row["episode_index"] for row in rows if row["reassigned_frames"]["total"] > 0)
    if changed != EXPECTED_CHANGED_EPISODES:
        raise AssertionError(f"unexpected changed overlays: {changed}")

    review_by_episode: dict[int, dict[str, Any]] | None = None
    review_metadata: dict[str, Any] | None = None
    if visual_review_file is not None:
        review_by_episode, review_metadata = _load_visual_review(visual_review_file, rows)
    if publish_overlays and review_by_episode is None:
        raise ValueError("publishing requires a complete versioned visual-review manifest")

    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite versioned output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_output = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    staged_overlays: dict[Path, Path] = {}
    published: list[Path] = []
    try:
        review_artifacts = _render_review_pages(rows, staging_output)
        for row in rows:
            if review_by_episode is None:
                row["manual_visual_no_leak_review"] = {
                    "status": "pending",
                    "objects_hidden_through_wait": None,
                }
            else:
                review = review_by_episode[row["episode_index"]]
                row["manual_visual_no_leak_review"] = {
                    "status": "pass" if review["objects_hidden_through_wait"] else "fail",
                    "objects_hidden_through_wait": bool(review["objects_hidden_through_wait"]),
                    "note": review.get("note", ""),
                }
        summary = _summary(rows, review_by_episode)
        report = {
            "schema_version": SCHEMA_VERSION,
            "generation_date": "2026-08-30",
            "mode": "published" if publish_overlays else "preview_only",
            "overlay_filename": OVERLAY_NAME,
            "rewrite_rule": {
                "close_reset_end": "strict_wait.start - 1",
                "wait_bounds": "inclusive strict 14-D static core",
                "execute_start": "strict_wait.end + 1",
            },
            "strict_wait_config": {
                "state_dimensions": 14,
                "dtype": "float32",
                "max_abs_step": audit.MAX_SPEED,
                "max_abs_step_comparator": "strictly less than",
                "max_per_dim_excursion": audit.MAX_EXCURSION,
                "max_excursion_comparator": "less than or equal",
            },
            "visual_review": {
                "status": "complete" if review_metadata is not None else "pending",
                "metadata": review_metadata,
                "manifest_path": str(visual_review_file.resolve()) if visual_review_file is not None else None,
                "manifest_sha256": audit._sha256_file(visual_review_file) if visual_review_file is not None else None,
                "warning": "manual top-camera boundary review; no uncalibrated whiteness detector is used",
            },
            "summary": summary,
            "canonical_hashes_before": canonical_hashes_before,
            "canonical_hashes_after": None,
            "review_artifacts": review_artifacts,
            "episodes": [_public_row(row) for row in rows],
        }

        if publish_overlays:
            staged_overlays = _prepare_overlay_temp_files(rows)
            published = _publish_overlay_temp_files(staged_overlays)
            for row in rows:
                destination = Path(row["overlay_label_path"])
                if audit._sha256_file(destination) != row["overlay_label_sha256"]:
                    raise AssertionError(f"published overlay hash mismatch: {destination}")

        canonical_hashes_after = _canonical_hashes(specs)
        if canonical_hashes_after != canonical_hashes_before:
            raise AssertionError("a canonical label changed during overlay generation")
        report["canonical_hashes_after"] = canonical_hashes_after
        (staging_output / "overlay_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
        (staging_output / "report.md").write_text(_markdown_report(report))
        _artifact_manifest(staging_output)
        staging_output.replace(output_dir)
        return report
    except BaseException:
        for temporary in staged_overlays.values():
            temporary.unlink(missing_ok=True)
        for destination in published:
            destination.unlink(missing_ok=True)
        shutil.rmtree(staging_output, ignore_errors=True)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=audit.DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--visual-review", type=Path, default=DEFAULT_VISUAL_REVIEW_FILE)
    parser.add_argument(
        "--preview-only",
        action="store_true",
        help="render review grids and a pending report without writing any overlay into raw demo directories",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    review_file = None if args.preview_only else args.visual_review
    report = generate(
        args.data_root,
        args.output_dir,
        visual_review_file=review_file,
        publish_overlays=not args.preview_only,
    )
    print(json.dumps({"status": "pass", "output_dir": str(args.output_dir), "summary": report["summary"]}, indent=2))


if __name__ == "__main__":
    main()
