"""Evaluate the trained retriever on held-out demos.

Reports Recall@K (K=1,3,5), MRR, and per-stage breakdown. Splits anchors into
two retrieval regimes:
  - memory-query  (stage_id in {3,4,5,6}, positive = observe-segment keyframes)
  - self-retrieval (other stages, positive = current frame)

Usage:
    python -m retriever.eval --ckpt retriever/runs/v1/best.pt \\
                             --features-dir retriever/cache/features \\
                             --labels retriever/cache/labels/samples_eval.jsonl

    python -m retriever.eval --smoke --ckpt retriever/runs/smoke/best.pt
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from retriever.dataset import RetrieverDataset
from retriever.model import Retriever
from retriever.train import M_TOPK, NEG_INF, _select_topm

QUERY_STAGES = {3, 4, 5, 6}
STAGE_NAMES = {0: "observe", 1: "mix", 2: "delay", 3: "query1", 4: "query2",
               5: "query3", 6: "query4", 7: "return_home"}


@torch.no_grad()
def score_batch(model: Retriever, batch: dict, device: str) -> torch.Tensor:
    """Return final scores (B, N_MAX). Invalid slots are -inf."""
    text_emb = batch["text_emb"].to(device)
    cur_global = batch["cur_global"].to(device)
    cur_patches = batch["cur_patches"].to(device)
    cand_global = batch["cand_global"].to(device)
    cand_patches = batch["cand_patches"].to(device)
    dt = batch["dt"].to(device)
    valid = batch["valid_mask"].to(device)
    forced = batch["forced_mask"].to(device)

    B, N, _ = cand_global.shape
    coarse = model.coarse(text_emb, cur_global, cand_global, dt)
    # At eval we do NOT use the labeled forced mask to bias selection — pretend we don't
    # know the labels. Just pick the top-M by coarse.
    idx_m = _select_topm(coarse, torch.zeros_like(forced), valid, M_TOPK)
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, M_TOPK)
    cand_patches_m = cand_patches[b_idx, idx_m]
    dt_m = dt[b_idx, idx_m]
    rerank = model.reranker(text_emb, cur_patches, cand_patches_m, dt_m)
    combined_m = coarse.gather(1, idx_m) + rerank                              # (B, M)

    # Scatter back to the full (B, N) candidate dim. Non-rerank candidates keep their
    # coarse-only score (they're already lower-ranked by definition).
    scores = coarse.clone()
    scores.scatter_(1, idx_m, combined_m)
    scores = scores.masked_fill(~valid, NEG_INF)
    return scores


def rank_metrics(scores: torch.Tensor, pos_mask: torch.Tensor) -> dict:
    """Recall@{1,3,5} and MRR for one batch. Anchors with no positives are skipped."""
    has_pos = pos_mask.any(dim=-1)
    if not has_pos.any():
        return {"recall@1": float("nan"), "recall@3": float("nan"),
                "recall@5": float("nan"), "mrr": float("nan"), "n": 0}
    s = scores[has_pos]
    p = pos_mask[has_pos]

    # rank of each candidate = number of candidates with strictly greater score, plus 1
    # for ties we use stable sort by index (consistent)
    order = s.argsort(dim=-1, descending=True)  # (B', N)
    # rank of each item = position in `order`
    ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(s.size(1), device=s.device).unsqueeze(0).expand_as(order))
    # best positive rank per row
    big = order.size(1) + 1
    pos_rank = torch.where(p, ranks, torch.full_like(ranks, big)).min(dim=-1).values  # (B',)
    pos_rank = pos_rank + 1  # 1-indexed
    return {
        "recall@1": float((pos_rank <= 1).float().mean()),
        "recall@3": float((pos_rank <= 3).float().mean()),
        "recall@5": float((pos_rank <= 5).float().mean()),
        "mrr": float((1.0 / pos_rank.float()).mean()),
        "n": int(p.size(0)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=pathlib.Path)
    p.add_argument("--features-dir", default="retriever/cache/features", type=pathlib.Path)
    p.add_argument("--labels", default="retriever/cache/labels/samples_eval.jsonl", type=pathlib.Path)
    p.add_argument("--text-emb", default="retriever/cache/text_emb.npz", type=pathlib.Path)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, type=pathlib.Path)
    p.add_argument("--smoke", action="store_true",
                   help="If samples_eval.jsonl is empty, fall back to samples.jsonl")
    args = p.parse_args()

    labels_path = args.labels
    if args.smoke and (not labels_path.exists() or labels_path.stat().st_size == 0):
        fallback = labels_path.parent / "samples.jsonl"
        print(f"[eval] {labels_path} empty, falling back to {fallback}")
        labels_path = fallback

    ds = RetrieverDataset(labels_path, args.features_dir, args.text_emb)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = Retriever().to(args.device)
    state = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()
    print(f"[eval] {len(ds)} samples from {labels_path}, ckpt={args.ckpt}")

    # Accumulate scores+positives by regime and by stage.
    all_scores: list[torch.Tensor] = []
    all_pos: list[torch.Tensor] = []
    all_stage: list[int] = []
    for batch in loader:
        scores = score_batch(model, batch, args.device).cpu()
        pos = batch["positives_mask"].cpu()
        all_scores.append(scores)
        all_pos.append(pos)
        all_stage.extend([int(s) for s in batch["stage_id"]])
    scores = torch.cat(all_scores, dim=0)
    pos = torch.cat(all_pos, dim=0)
    stage = torch.tensor(all_stage)

    is_query = torch.tensor([s in QUERY_STAGES for s in all_stage])
    report: dict = {"overall": rank_metrics(scores, pos)}
    if is_query.any():
        report["memory_query"] = rank_metrics(scores[is_query], pos[is_query])
    if (~is_query).any():
        report["self_retrieval"] = rank_metrics(scores[~is_query], pos[~is_query])
    per_stage: dict[str, dict] = {}
    for sid in sorted(set(all_stage)):
        m = torch.tensor([s == sid for s in all_stage])
        per_stage[STAGE_NAMES.get(sid, f"stage{sid}")] = rank_metrics(scores[m], pos[m])
    report["per_stage"] = per_stage

    # Print and save
    def fmt(d):
        return (f"R@1={d['recall@1']:.3f}  R@3={d['recall@3']:.3f}  "
                f"R@5={d['recall@5']:.3f}  MRR={d['mrr']:.3f}  n={d['n']}")
    print("\n=== Eval ===")
    for k in ("overall", "memory_query", "self_retrieval"):
        if k in report:
            print(f"  [{k:15s}] {fmt(report[k])}")
    print("  per stage:")
    for name, d in per_stage.items():
        print(f"    {name:13s} {fmt(d)}")

    if args.out is None:
        args.out = args.ckpt.parent / "eval.json"
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\n[eval] saved {args.out}")


if __name__ == "__main__":
    main()
