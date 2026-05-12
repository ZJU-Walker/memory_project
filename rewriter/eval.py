"""End-to-end eval: retriever top-1 -> Gemini rewriter -> compare to ground truth.

Usage:
    python -m rewriter.eval --ckpt retriever/runs/v1_0511/best.pt
    python -m rewriter.eval --ckpt ... --limit 4 --no-cache
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Pull in the canonical-prompt oracle from openpi.
sys.path.append(str(pathlib.Path("openpi/examples/yam").resolve()))
from plate_prompts import detailed_prompt  # noqa: E402

from retriever.dataset import RetrieverDataset
from retriever.eval import STAGE_NAMES, score_batch
from retriever.model import Retriever
from retriever.viz import _video_frames
from rewriter.rewriter import rewrite
from rewriter.vocabulary import is_in_vocabulary

DATASET_ROOT = pathlib.Path("/home/kewalk/memory_project/dataset/PLATE_TASK_1")
QUERY_STAGES = {3, 4, 5, 6}


def _load_meta(demo: str) -> dict:
    return json.loads((DATASET_ROOT / demo / "metadata.json").read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=pathlib.Path)
    p.add_argument("--features-dir", default="retriever/cache/features", type=pathlib.Path)
    p.add_argument("--labels", default="retriever/cache/labels/samples_eval.jsonl", type=pathlib.Path)
    p.add_argument("--text-emb", default="retriever/cache/text_emb.npz", type=pathlib.Path)
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--out", default=None, type=pathlib.Path,
                   help="output JSON (default: <ckpt parent>/rewriter_eval.json)")
    args = p.parse_args()

    ds = RetrieverDataset(args.labels, args.features_dir, args.text_emb)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = Retriever().to(args.device)
    state = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    n = len(ds) if args.limit is None else min(args.limit, len(ds))
    print(f"[rewriter.eval] {n} samples, model={args.model}, ckpt={args.ckpt}")

    results = []
    for i, batch in enumerate(tqdm(loader, total=n)):
        if i >= n:
            break
        sid = int(batch["stage_id"][0])
        demo = batch["demo"][0]
        anchor_fid = int(batch["anchor_frame_id"][0])
        anchor_prompt = ds.samples[i]["anchor_prompt"]
        cand_fids = batch["cand_frame_ids"][0].numpy()
        valid_mask = batch["valid_mask"][0].numpy()

        with torch.no_grad():
            scores = score_batch(model, batch, args.device).cpu().numpy()[0]
        scores_masked = np.where(valid_mask, scores, -np.inf)
        top1_idx = int(np.argmax(scores_masked))
        top1_fid = int(cand_fids[top1_idx])

        video = _video_frames(demo)
        image = np.asarray(video[top1_fid], dtype=np.uint8)

        rewritten = rewrite(image, anchor_prompt, model=args.model)

        meta = _load_meta(demo)
        expected = detailed_prompt(meta, sid)
        # query-stage object extraction (for object-only accuracy)
        expected_object = (
            meta["episode_plan"]["queries"][sid - 3]["correct_object"].replace("_", " ")
            if sid in QUERY_STAGES else None
        )

        rewritten_norm = rewritten.strip().rstrip(".")
        expected_norm = expected.strip().rstrip(".")
        exact = rewritten_norm.lower() == expected_norm.lower()
        object_ok = (
            (expected_object.lower() in rewritten.lower())
            if expected_object is not None else None
        )
        vocab_clean = is_in_vocabulary(rewritten)

        results.append({
            "demo": demo,
            "stage_id": sid,
            "stage_name": STAGE_NAMES.get(sid, f"stage{sid}"),
            "anchor_frame_id": anchor_fid,
            "top1_frame_id": top1_fid,
            "anchor_prompt": anchor_prompt,
            "expected": expected,
            "rewritten": rewritten,
            "exact_match": bool(exact),
            "object_match": object_ok,
            "vocab_clean": bool(vocab_clean),
        })

    # Aggregate metrics.
    by_stage: dict[str, list[dict]] = collections.defaultdict(list)
    for r in results:
        by_stage[r["stage_name"]].append(r)
    query_rs = [r for r in results if r["stage_id"] in QUERY_STAGES]
    nonquery_rs = [r for r in results if r["stage_id"] not in QUERY_STAGES]

    def agg(rs: list[dict]) -> dict:
        if not rs:
            return {"n": 0}
        n = len(rs)
        out = {"n": n,
               "exact_match": sum(r["exact_match"] for r in rs) / n,
               "vocab_clean": sum(r["vocab_clean"] for r in rs) / n}
        obj_rs = [r for r in rs if r["object_match"] is not None]
        if obj_rs:
            out["object_match"] = sum(r["object_match"] for r in obj_rs) / len(obj_rs)
        return out

    report = {
        "overall": agg(results),
        "query": agg(query_rs),
        "non_query": agg(nonquery_rs),
        "per_stage": {s: agg(rs) for s, rs in by_stage.items()},
        "errors": [r for r in results if not r["exact_match"]],
    }

    def fmt(d):
        parts = [f"exact={d.get('exact_match', float('nan')):.3f}"]
        if "object_match" in d:
            parts.append(f"obj={d['object_match']:.3f}")
        parts.append(f"vocab={d.get('vocab_clean', float('nan')):.3f}")
        parts.append(f"n={d['n']}")
        return "  ".join(parts)

    print("\n=== Rewriter Eval ===")
    print(f"  [overall  ] {fmt(report['overall'])}")
    print(f"  [query    ] {fmt(report['query'])}")
    print(f"  [non_query] {fmt(report['non_query'])}")
    print("  per stage:")
    for name in ["observe", "mix", "delay", "query1", "query2", "query3", "query4", "return_home"]:
        if name in report["per_stage"]:
            print(f"    {name:13s} {fmt(report['per_stage'][name])}")

    if report["errors"]:
        print(f"\n  {len(report['errors'])} mismatches (first 5):")
        for r in report["errors"][:5]:
            print(f"    {r['demo']} {r['stage_name']:13s}  expected: {r['expected']!r}")
            print(f"    {' '*len(r['demo'])} {' '*13}  got:      {r['rewritten']!r}")

    if args.out is None:
        args.out = args.ckpt.parent / "rewriter_eval.json"
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n[rewriter.eval] saved {args.out}")


if __name__ == "__main__":
    main()
