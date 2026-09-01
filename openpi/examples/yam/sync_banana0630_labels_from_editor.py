"""Sync boundary edits from the subtask_label_editor back to the raw June-30 demos.

The browser editor (Qwen3-VL/qwen-vl-finetune/scripts/subtask_label_editor.py) edits the
combined ``subtask_labels.json`` in the editor-facing dataset view; this script writes each
episode's segments back to ``demoN/subtask_labels.json`` (episode_{i:06d}.mp4 <-> demo{i+1}).

Usage: python examples/yam/sync_banana0630_labels_from_editor.py [--dry-run]
"""

import argparse
import json
import pathlib

EDITOR_LABELS = pathlib.Path(
    "/iris/projects/humanoid/ke/relabel_0630_banana/0630_banana_mem/videos/chunk-000/subtask_labels.json")
RAW_ROOT = pathlib.Path("/iris/u/kewalk/memory_project/data/bin_memory_banana")
EXPECTED_TASKS = ("open both lids", "inspect both bins", "close both lids and reset arms")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    combined = json.loads(EDITOR_LABELS.read_text())
    changed = 0
    for key, segs in sorted(combined.items()):
        index = int(key.split("_")[1].split(".")[0])
        demo = RAW_ROOT / f"demo{index + 1}"
        tasks = tuple(seg["task"] for seg in segs)
        if tasks[:3] != EXPECTED_TASKS or len(segs) != 4:
            raise ValueError(f"{key}: unexpected task structure {tasks}")
        for prev, cur in zip(segs, segs[1:]):
            if cur["start"] != prev["end"] + 1:
                raise ValueError(f"{key}: segments not contiguous at {cur}")
        current = json.loads((demo / "subtask_labels.json").read_text())
        if current == segs:
            continue
        changed += 1
        print(f"{demo.name:>7}: inspect {segs[1]['start']}-{segs[1]['end']} "
              f"(was {current[1]['start']}-{current[1]['end']})")
        if not args.dry_run:
            (demo / "subtask_labels.json").write_text(json.dumps(segs, indent=4))
    print(f"{changed} demos updated" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
