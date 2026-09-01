"""Build versioned 0816 labels whose semantic wait is exactly the strict 14-D static core.

Canonical ``subtask_labels.json`` files are never overwritten.  The v3.5 converter reads the
separate ``subtask_labels_v35_staticwait.json`` overlay so every frame labeled wait satisfies
the production motion gate.  Frames trimmed from the head remain close/reset; frames trimmed
from the tail become execute.  This is deliberately conservative for leak prevention and is
surfaced for manual boundary review in the existing 0816 audit sheets.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

CANONICAL_LABEL = "subtask_labels.json"
OVERLAY_LABEL = "subtask_labels_v35_staticwait.json"
MAX_STATE_STEP = 4e-3
MAX_STATE_EXCURSION = 2e-2
STRIDE_FRAMES = 15


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _expected_tasks(side: str) -> list[str]:
    return [
        "open both lids",
        "inspect both bins",
        "close both lids and reset arms",
        f"wait; target bin is {side}",
        f"open {side} bin",
    ]


def build_overlay(canonical: list[dict[str, Any]], core_start: int, core_end: int) -> list[dict[str, Any]]:
    if len(canonical) != 5:
        raise ValueError(f"expected exactly five canonical segments, found {len(canonical)}")
    wait_task = str(canonical[3]["task"])
    if wait_task == "wait; target bin is left":
        side = "left"
    elif wait_task == "wait; target bin is right":
        side = "right"
    else:
        raise ValueError(f"segment four is not a sided wait: {wait_task!r}")
    tasks = [str(segment["task"]) for segment in canonical]
    if tasks != _expected_tasks(side):
        raise ValueError(f"canonical labels do not have the exact five-phase {side} schema: {tasks}")
    if canonical[0]["start"] != 0:
        raise ValueError("canonical labels do not start at frame zero")
    for previous, current in itertools.pairwise(canonical):
        if current["start"] != previous["end"] + 1:
            raise ValueError("canonical labels are not contiguous")
    original_wait = canonical[3]
    if not (original_wait["start"] <= core_start <= core_end <= original_wait["end"]):
        raise ValueError(
            f"strict core {core_start}..{core_end} is outside canonical wait "
            f"{original_wait['start']}..{original_wait['end']}"
        )
    if core_start <= canonical[2]["start"] or core_end >= canonical[4]["end"]:
        raise ValueError("strict core would make close/reset or execute empty")
    return [
        dict(canonical[0]),
        dict(canonical[1]),
        {**canonical[2], "end": core_start - 1},
        {**canonical[3], "start": core_start, "end": core_end},
        {**canonical[4], "start": core_end + 1},
    ]


def _validate_static_wait(demo: Path, overlay: list[dict[str, Any]]) -> dict[str, Any]:
    wait = overlay[3]
    state = np.concatenate(
        [
            np.load(demo / "left_joint_positions.npy"),
            np.load(demo / "right_joint_positions.npy"),
        ],
        axis=1,
    ).astype(np.float32)
    waiting = state[wait["start"] : wait["end"] + 1]
    max_step = float(np.abs(np.diff(waiting, axis=0)).max()) if len(waiting) > 1 else float("inf")
    max_excursion = float((waiting.max(axis=0) - waiting.min(axis=0)).max())
    if max_step >= MAX_STATE_STEP or max_excursion > MAX_STATE_EXCURSION:
        raise ValueError(f"{demo}: overlay wait motion failed: step={max_step}, excursion={max_excursion}")
    return {
        "start": wait["start"],
        "end": wait["end"],
        "length": len(waiting),
        "max_14d_step": max_step,
        "max_14d_excursion": max_excursion,
        "eligible_at_stride_15": len(waiting) >= STRIDE_FRAMES,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("diagnostic_outputs/0816_memory_label_audit_v1/audit.json"),
    )
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    audit_path = args.audit.resolve()
    audit = json.loads(audit_path.read_text())
    if len(audit["episodes"]) != 60:
        raise ValueError(f"{audit_path}: expected 60 episodes")
    canonical_before = {}
    records = []
    for episode in audit["episodes"]:
        demo = Path(episode["demo_path"])
        canonical_path = demo / CANONICAL_LABEL
        canonical_sha = _sha256(canonical_path)
        if canonical_sha != episode["raw_label_sha256"]:
            raise ValueError(
                f"{canonical_path}: canonical hash drifted from audit {episode['raw_label_sha256']} to {canonical_sha}"
            )
        canonical_before[str(canonical_path)] = canonical_sha
        canonical = json.loads(canonical_path.read_text())
        core = episode["strict_static_core_14d"]
        overlay = build_overlay(canonical, int(core["start"]), int(core["end"]))
        motion = _validate_static_wait(demo, overlay)
        output = demo / OVERLAY_LABEL
        encoded = json.dumps(overlay, indent=2) + "\n"
        if output.exists() and output.read_text() != encoded and not args.overwrite:
            raise FileExistsError(f"{output} differs; pass --overwrite to replace it")
        if args.write:
            _atomic_json(output, overlay)
            if output.read_text() != encoded:
                raise AssertionError(f"atomic write verification failed: {output}")
        records.append(
            {
                "episode_index": episode["episode_index"],
                "stable_id": f"{episode['source']}/{episode['demo_name']}",
                "demo_path": str(demo),
                "canonical_label": CANONICAL_LABEL,
                "canonical_sha256": canonical_sha,
                "overlay_label": OVERLAY_LABEL,
                "overlay_sha256": _sha256(output) if args.write else hashlib.sha256(encoded.encode()).hexdigest(),
                "original_wait": episode["original_wait"],
                "strict_wait": motion,
                "reassigned_to_close_reset": {
                    "start": canonical[3]["start"],
                    "end": core["start"] - 1,
                    "frames": core["start"] - canonical[3]["start"],
                },
                "reassigned_to_execute": {
                    "start": core["end"] + 1,
                    "end": canonical[3]["end"],
                    "frames": canonical[3]["end"] - core["end"],
                },
            }
        )

    for raw_path, before_sha in canonical_before.items():
        if _sha256(Path(raw_path)) != before_sha:
            raise AssertionError(f"canonical label changed while generating overlays: {raw_path}")
    eligible = sum(record["strict_wait"]["eligible_at_stride_15"] for record in records)
    report = {
        "schema_version": 1,
        "status": "written" if args.write else "dry_run_passed",
        "audit": str(audit_path),
        "canonical_labels_modified": False,
        "overlay_label": OVERLAY_LABEL,
        "episodes": len(records),
        "wait_frames_reassigned": sum(
            record["reassigned_to_close_reset"]["frames"] + record["reassigned_to_execute"]["frames"]
            for record in records
        ),
        "eligible_at_stride_15": eligible,
        "ineligible_at_stride_15": len(records) - eligible,
        "semantic_policy": {
            "head_trim": "reassign from wait to close both lids and reset arms",
            "tail_trim": "reassign from wait to sided execute",
            "reason": "conservative v3.5 no-motion wait; canonical labels remain unchanged for historical runs",
        },
        "episodes_detail": records,
    }
    report_path = audit_path.parent / "staticwait_overlay.json"
    _atomic_json(report_path, report)
    print(
        f"{report['status']}: {len(records)} overlays; reassigned "
        f"{report['wait_frames_reassigned']} wait frames; report={report_path}"
    )


if __name__ == "__main__":
    main()
