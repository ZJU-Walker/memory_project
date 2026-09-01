"""Move each 0831 episode's inspect_start to full object visibility.

The autogen rule starts "inspect both bins" when both bin interiors first exceed a fixed
20% white fraction, which can land mid-lid-swing with the objects still partly occluded.
This refinement keeps every other boundary and moves inspect_start to the first frame where
BOTH sides' white fractions have saturated at >= SATURATION of their own within-window
plateau (95th percentile) and stay there for HOLD frames, i.e. the lids are fully open.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import prepare_0830_memory_labels as prep

SATURATION = 0.90
HOLD = 5
MIN_INSPECT = 15
ROOT = Path(__file__).resolve().parents[3]


def refine(demo: Path) -> dict:
    labels = json.loads((demo / prep.FINAL_LABEL).read_text())
    tasks = [seg["task"] for seg in labels]
    assert tasks[0] == "open both lids" and tasks[1] == "inspect both bins", demo
    old_start, close_start = labels[1]["start"], labels[2]["start"]
    _, _, scores = prep._read_scores(demo)
    window = scores[old_start:close_start]
    plateau = np.percentile(window, 95, axis=0)
    ok = np.all(window >= SATURATION * plateau[None, :], axis=1)
    sustained = [i for i in range(len(ok) - HOLD + 1) if ok[i : i + HOLD].all()]
    latest_allowed = close_start - MIN_INSPECT - old_start
    candidates = [i for i in sustained if i <= latest_allowed]
    shift = candidates[0] if candidates else (sustained[0] if sustained else 0)
    shift = min(shift, max(latest_allowed, 0))
    new_start = old_start + shift
    labels[0]["end"] = new_start - 1
    labels[1]["start"] = new_start
    payload = json.dumps(labels, indent=2) + "\n"
    (demo / prep.FINAL_LABEL).write_text(payload)
    (demo / prep.AUTOGEN_LABEL).write_text(payload)
    return {
        "demo": demo.name,
        "old_inspect_start": old_start,
        "new_inspect_start": new_start,
        "shift_frames": shift,
        "inspect_len": close_start - new_start,
        "plateau_left": float(plateau[0]),
        "plateau_right": float(plateau[1]),
    }


def main() -> None:
    manifest = json.loads((ROOT / "data/0831_episode_manifest_v1.json").read_text())
    rows = [refine(ROOT / spec["raw_dir"]) for spec in manifest["episodes"]]
    shifts = [r["shift_frames"] for r in rows]
    lens = [r["inspect_len"] for r in rows]
    print(f"refined {len(rows)} episodes; shift frames min={min(shifts)} median={int(np.median(shifts))} max={max(shifts)}")
    print(f"inspect length after: min={min(lens)} median={int(np.median(lens))} max={max(lens)}")
    for r in rows:
        if r["shift_frames"] == 0 or r["inspect_len"] < MIN_INSPECT:
            print("  review:", r)
    (ROOT / "openpi/examples/yam/diagnostic_outputs/0831_memory_labels_v1/inspect_refinement.json").write_text(
        json.dumps(rows, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
