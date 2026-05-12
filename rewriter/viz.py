"""Per-demo visualization of the retriever -> rewriter pipeline.

For each eval anchor in the chosen demo, draws a figure with:
    [ anchor frame | top-1 retrieved frame ]
plus a caption block showing original prompt, rewritten prompt, and
(for query stages) the ground-truth expected prompt.

Usage:
    python -m rewriter.viz --ckpt retriever/runs/v1_0511/best.pt --demo demo36
    python -m rewriter.viz --ckpt ... --demo demo36 --limit 4
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# plate_prompts oracle
sys.path.append(str(pathlib.Path("openpi/examples/yam").resolve()))
from plate_prompts import detailed_prompt  # noqa: E402

from retriever.dataset import RetrieverDataset
from retriever.eval import STAGE_NAMES, score_batch
from retriever.model import Retriever
from retriever.viz import _video_frames, _draw_panel, _wrap
from rewriter.rewriter import rewrite
from rewriter.vocabulary import is_in_vocabulary

DATASET_ROOT = pathlib.Path("/home/kewalk/memory_project/dataset/PLATE_TASK_1")
QUERY_STAGES = {3, 4, 5, 6}


def _meta(demo: str) -> dict:
    return json.loads((DATASET_ROOT / demo / "metadata.json").read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=pathlib.Path)
    p.add_argument("--demo", required=True, help="e.g. demo36 or demo37")
    p.add_argument("--features-dir", default="retriever/cache/features", type=pathlib.Path)
    p.add_argument("--labels", default="retriever/cache/labels/samples_eval.jsonl", type=pathlib.Path)
    p.add_argument("--text-emb", default="retriever/cache/text_emb.npz", type=pathlib.Path)
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--limit", type=int, default=None,
                   help="Max anchors to render (default: all matching demo)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, type=pathlib.Path,
                   help="Output dir (default: <ckpt parent>/rewriter_viz/<demo>)")
    args = p.parse_args()

    out_dir = args.out if args.out else args.ckpt.parent / "rewriter_viz" / args.demo
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = RetrieverDataset(args.labels, args.features_dir, args.text_emb)
    # Restrict to the chosen demo, then keep one anchor per stage — the
    # earliest one (closest to that stage's start_step).
    demo_idx_all = [i for i, s in enumerate(ds.samples) if s["demo"] == args.demo]
    if not demo_idx_all:
        raise SystemExit(f"No eval samples for {args.demo!r} in {args.labels}")
    by_stage_earliest: dict[int, int] = {}
    for i in demo_idx_all:
        s = ds.samples[i]
        sid = int(s["stage_id"])
        fid = int(s["anchor_frame_id"])
        if sid not in by_stage_earliest or fid < int(ds.samples[by_stage_earliest[sid]]["anchor_frame_id"]):
            by_stage_earliest[sid] = i
    demo_idx = [by_stage_earliest[sid] for sid in sorted(by_stage_earliest)]
    if args.limit is not None:
        demo_idx = demo_idx[: args.limit]
    print(f"[rewriter.viz] {args.demo}: {len(demo_idx)} anchors "
          f"(one per stage, from {len(demo_idx_all)} eval samples), out={out_dir}")

    model = Retriever().to(args.device)
    state = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    video = _video_frames(args.demo)
    meta = _meta(args.demo)
    loader = DataLoader(torch.utils.data.Subset(ds, demo_idx), batch_size=1, shuffle=False)

    for batch, i in zip(loader, demo_idx):
        sid = int(batch["stage_id"][0])
        anchor_fid = int(batch["anchor_frame_id"][0])
        original_prompt = ds.samples[i]["anchor_prompt"]
        cand_fids = batch["cand_frame_ids"][0].numpy()
        valid_mask = batch["valid_mask"][0].numpy()

        with torch.no_grad():
            scores = score_batch(model, batch, args.device).cpu().numpy()[0]
        scores_masked = np.where(valid_mask, scores, -np.inf)
        top1_idx = int(np.argmax(scores_masked))
        top1_fid = int(cand_fids[top1_idx])
        top1_score = float(scores[top1_idx])

        anchor_img = video[anchor_fid]
        retrieved_img = video[top1_fid]

        rewritten = rewrite(retrieved_img, original_prompt, model=args.model)
        expected = detailed_prompt(meta, sid) if sid in QUERY_STAGES else None
        vocab_ok = is_in_vocabulary(rewritten)

        # Figure
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0))
        stage_name = STAGE_NAMES.get(sid, f"stage{sid}")
        fig.suptitle(f"{args.demo}  |  {stage_name}  |  anchor frame {anchor_fid}", fontsize=11)
        _draw_panel(axes[0], anchor_img, f"anchor frame ({anchor_fid})", border="anchor")
        retrieved_label = f"retrieved top-1 (frame {top1_fid}, score {top1_score:.2f})"
        _draw_panel(axes[1], retrieved_img, retrieved_label, border=None)

        # Caption block
        lines = [
            f"original:  {_wrap(original_prompt, width=110)}",
            f"rewritten: {_wrap(rewritten, width=110)}",
        ]
        if expected is not None:
            ok = rewritten.strip().rstrip(".").lower() == expected.strip().rstrip(".").lower()
            tag = "MATCH" if ok else "MISMATCH"
            lines.append(f"expected:  {_wrap(expected, width=110)}   [{tag}]")
        if not vocab_ok:
            lines.append("(!) vocabulary check: FAILED")

        plt.tight_layout(rect=(0, 0.30, 1, 0.94))
        fig.text(0.02, 0.02, "\n".join(lines), fontsize=8.5, family="monospace",
                 verticalalignment="bottom")

        fname = f"{stage_name}_a{anchor_fid:05d}.png"
        fig.savefig(out_dir / fname, dpi=110)
        plt.close(fig)
        print(f"  {stage_name:13s}  frame {anchor_fid:5d}  -> top1={top1_fid:5d}  "
              f"rewrite={rewritten!r}")

    print(f"[rewriter.viz] saved {len(demo_idx)} figures to {out_dir}")


if __name__ == "__main__":
    main()
