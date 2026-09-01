"""First-round 5-phase relabeling of the June-30 banana demos to match the 0816 vocabulary.

Splits each demo's old ``observe bins`` window into ``open both lids`` / ``inspect both bins`` /
``close both lids and reset arms``, keeping the terminal ``open left/right bin`` unchanged.

``inspect both bins`` is defined VISUALLY, per the labeling requirement "bins fully open and
both objects observable": a per-frame whiteness detector over the two fixed bin regions of the
top camera (the white baskets are only visible when their wooden board lids are off) marks a
frame open when both regions exceed WHITE_FRAC; the inspect segment is the longest contiguous
both-open run (dips shorter than GAP_CLOSE frames, e.g. an arm shadow, are bridged).  Motion
segmentation was tried first and rejected: the demoer replaces the boards quickly mid-window
and then rests, which fooled every joint-speed cue; the pixel cue matches frame inspection.

Originals are backed up once per demo as ``subtask_labels_3task_backup.json``.  Demos with a
short (<MIN_INSPECT) or ambiguous open run are flagged NEEDS_REVIEW for the Streamlit pass.

Usage: python examples/yam/autolabel_banana0630_subtasks.py [--dry-run]
"""

import argparse
import json
import pathlib

import cv2
import numpy as np

DATA_ROOT = pathlib.Path("/iris/u/kewalk/memory_project/data/bin_memory_banana")
L_BOX = (190, 300, 190, 310)   # x0, x1, y0, y1 in the 640x480 top camera
R_BOX = (320, 430, 190, 300)
WHITE_FRAC = 0.33              # both regions above this = bins fully open
GAP_CLOSE = 10                 # bridge sub-10-frame dips (arm shadows)
MIN_INSPECT = 20               # frames; shorter runs get flagged

OPEN_LABEL = "open both lids"
INSPECT_LABEL = "inspect both bins"
CLOSE_LABEL = "close both lids and reset arms"


def _whiteness(img, box):
    x0, x1, y0, y1 = box
    region = img[y0:y1, x0:x1].astype(np.int16)
    lo, hi = region.min(axis=2), region.max(axis=2)
    return float(((lo > 140) & (hi - lo < 60)).mean())


def segment(demo: pathlib.Path):
    backup = demo / "subtask_labels_3task_backup.json"
    source = backup if backup.exists() else demo / "subtask_labels.json"
    labels = json.loads(source.read_text())
    if [seg["task"] for seg in labels][:1] != ["observe bins"]:
        return None, labels, "unexpected first segment"
    observe_end = labels[0]["end"]

    cap = cv2.VideoCapture(str(demo / "top_camera_rgb.mp4"))
    both_open = np.zeros(observe_end + 1, dtype=bool)
    for t in range(observe_end + 1):
        ok, img = cap.read()
        if not ok:
            cap.release()
            return None, labels, f"failed to decode frame {t}"
        both_open[t] = _whiteness(img, L_BOX) > WHITE_FRAC and _whiteness(img, R_BOX) > WHITE_FRAC
    cap.release()

    # bridge short dips, then take the longest contiguous open run
    padded = np.concatenate([[False], both_open, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    runs = [(int(edges[i]), int(edges[i + 1]) - 1) for i in range(0, len(edges), 2)]
    merged = []
    for run in runs:
        if merged and run[0] - merged[-1][1] - 1 <= GAP_CLOSE:
            merged[-1] = (merged[-1][0], run[1])
        else:
            merged.append(run)
    flags = []
    if not merged:
        return None, labels, "bins never detected fully open"
    lengths = [end - start + 1 for start, end in merged]
    start, end = merged[int(np.argmax(lengths))]
    if max(lengths) < MIN_INSPECT:
        flags.append(f"open run only {max(lengths)}f")
    others = [ln for ln in sorted(lengths, reverse=True)[1:] if ln >= MIN_INSPECT]
    if others:
        flags.append(f"second open run of {others[0]}f")
    if start < 30:
        flags.append("open run starts suspiciously early")
    if end > observe_end - 10:
        flags.append("open run touches the observe end")

    new = [
        {"task": OPEN_LABEL, "start": 0, "end": start - 1},
        {"task": INSPECT_LABEL, "start": start, "end": end},
        {"task": CLOSE_LABEL, "start": end + 1, "end": int(observe_end)},
        *labels[1:],
    ]
    return new, labels, "; ".join(flags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    demos = sorted((p for p in DATA_ROOT.iterdir() if p.name.startswith("demo")),
                   key=lambda p: int(p.name[4:]))
    review = 0
    for demo in demos:
        new, old, note = segment(demo)
        if new is None:
            print(f"{demo.name:>7}: SKIP ({note})")
            continue
        spans = {seg["task"]: (seg["start"], seg["end"]) for seg in new}
        o, i, c = spans[OPEN_LABEL], spans[INSPECT_LABEL], spans[CLOSE_LABEL]
        tag = f"  NEEDS_REVIEW: {note}" if note else ""
        if note:
            review += 1
        print(f"{demo.name:>7}: open 0-{o[1]:<4} inspect {i[0]}-{i[1]:<4} "
              f"({i[1]-i[0]+1:>3}f) close {c[0]}-{c[1]:<4} | {new[-1]['task']}{tag}")
        if not args.dry_run:
            backup = demo / "subtask_labels_3task_backup.json"
            if not backup.exists():
                backup.write_text(json.dumps(old, indent=4))
            (demo / "subtask_labels.json").write_text(json.dumps(new, indent=4))
    print(f"\n{len(demos)} demos, {review} flagged NEEDS_REVIEW"
          + (" (dry run, nothing written)" if args.dry_run else "; originals backed up as subtask_labels_3task_backup.json"))


if __name__ == "__main__":
    main()
