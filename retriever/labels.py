"""Generate training/eval JSONL samples for the retriever.

Each sample is one (instruction, current-frame) anchor with:
  - positives: ground-truth keyframes the retriever should rank high
  - hard_negatives: strictly past keyframes that look plausibly relevant

Unified positive rule (current frame is in the candidate pool):
  - query1..4 anchor  -> positives = all observe-segment keyframes
  - any other stage   -> positives = [anchor_frame_id]   (self-retrieval)

Strict causal masking: every candidate (positive or negative) has
frame_id <= anchor_frame_id. Enforced here and again in dataset.py.

Usage:
    python -m retriever.labels --features-dir retriever/cache/features \
                               --metadata-root dataset/PLATE_TASK_1 \
                               --out retriever/cache/labels
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

from retriever.memory_bank import MemoryBank

# Plate-memory stage IDs
OBSERVE = 0
MIX = 1
DELAY = 2
QUERY_STAGES = (3, 4, 5, 6)
RETURN_HOME = 7

ANCHORS_PER_SEGMENT = 2
N_HARD_NEGS = 4
TRAIN_DEMOS = list(range(1, 36))   # 1..35
EVAL_DEMOS = [36, 37]


def _build_samples_for_demo(
    demo: int,
    meta: dict,
    bank: MemoryBank,
    rng: random.Random,
) -> list[dict]:
    """Generate anchor samples for one demo."""
    samples: list[dict] = []

    # frame_id -> bank index, sorted
    kf_ids = bank.frame_ids.tolist()
    kf_stage = bank.stage_ids.tolist()
    by_stage: dict[int, list[int]] = {}
    for fid, sid in zip(kf_ids, kf_stage):
        by_stage.setdefault(int(sid), []).append(int(fid))

    observe_kfs = by_stage.get(OBSERVE, [])
    mix_kfs = by_stage.get(MIX, [])
    mix_second_half = mix_kfs[len(mix_kfs) // 2 :] if mix_kfs else []

    # Build segment list with start/end for anchor sampling, and a per-frame stage map.
    frame_stage = {fid: sid for fid, sid in zip(kf_ids, kf_stage)}

    for seg in meta["segments"]:
        sid = int(seg["stage_id"])
        start = int(seg["start_step"])
        end = int(seg["end_step"])
        seg_kfs = [f for f in kf_ids if start <= f < end]
        if not seg_kfs:
            continue

        # 2 random anchors per segment + 1 extra at start for queries.
        n = ANCHORS_PER_SEGMENT
        anchors = list(rng.sample(seg_kfs, min(n, len(seg_kfs))))
        if sid in QUERY_STAGES:
            anchors.append(seg_kfs[0])
        # dedupe while preserving order
        seen = set()
        anchors = [a for a in anchors if not (a in seen or seen.add(a))]

        for anchor in anchors:
            anchor_idx = kf_ids.index(anchor)
            past_kfs = [f for f in kf_ids if f <= anchor]  # causal

            # Positives
            if sid in QUERY_STAGES:
                positives = [f for f in observe_kfs if f <= anchor]
            else:
                positives = [anchor]
            if not positives:
                continue

            # Hard negatives — must be past, must not be a positive, must not be the anchor.
            pos_set = set(positives)
            if sid in QUERY_STAGES:
                # disturbed-layout frames from second half of mix + preceding query stages
                pool = [f for f in mix_second_half if f <= anchor and f not in pos_set]
                for prev_q in QUERY_STAGES:
                    if prev_q >= sid:
                        break
                    pool += [f for f in by_stage.get(prev_q, []) if f <= anchor and f not in pos_set]
            else:
                # non-query anchor: any past keyframe that isn't the anchor itself
                pool = [f for f in past_kfs if f != anchor]
            rng.shuffle(pool)
            hard_negs = pool[:N_HARD_NEGS]

            samples.append({
                "demo": f"demo{demo}",
                "anchor_frame_id": int(anchor),
                "anchor_prompt": bank.prompts[anchor_idx],
                "stage_id": sid,
                "positives": [int(x) for x in positives],
                "strong_positives": [],
                "hard_negatives": [int(x) for x in hard_negs],
            })

    return samples


def build(
    features_dir: pathlib.Path,
    metadata_root: pathlib.Path,
    out_dir: pathlib.Path,
    seed: int = 0,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    def run(demos: list[int], out_path: pathlib.Path) -> int:
        total = 0
        with out_path.open("w") as f:
            for d in demos:
                bank_path = features_dir / f"demo{d}.npz"
                meta_path = metadata_root / f"demo{d}" / "metadata.json"
                if not bank_path.exists() or not meta_path.exists():
                    continue
                bank = MemoryBank.load(bank_path)
                meta = json.loads(meta_path.read_text())
                samples = _build_samples_for_demo(d, meta, bank, rng)
                for s in samples:
                    f.write(json.dumps(s) + "\n")
                total += len(samples)
        return total

    n_train = run(TRAIN_DEMOS, out_dir / "samples.jsonl")
    n_eval = run(EVAL_DEMOS, out_dir / "samples_eval.jsonl")
    print(f"[labels] train={n_train}  eval={n_eval}  out={out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", default="retriever/cache/features", type=pathlib.Path)
    p.add_argument("--metadata-root", default="dataset/PLATE_TASK_1", type=pathlib.Path)
    p.add_argument("--out", default="retriever/cache/labels", type=pathlib.Path)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    build(args.features_dir, args.metadata_root, args.out, args.seed)


if __name__ == "__main__":
    main()
