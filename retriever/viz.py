"""Visualize what the retriever returned for each eval anchor.

For each sample in samples_eval.jsonl, plots a row of images:
    [ anchor (current frame) | top-1 | top-2 | ... | top-k ]
with the prompt as title and labels:
    - rank + score under each retrieved frame
    - green border = labeled positive
    - red border   = labeled hard negative
    - blue dashed  = the anchor / current frame
Saves one .png per anchor under <out>/viz/.

Usage:
    python -m retriever.viz --ckpt retriever/runs/v1_0511/best.pt --topk 5
    python -m retriever.viz --ckpt ... --max-per-stage 2
"""

from __future__ import annotations

import argparse
import json
import pathlib
from functools import lru_cache

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from retriever.dataset import RetrieverDataset
from retriever.eval import STAGE_NAMES, score_batch
from retriever.model import Retriever

DATASET_ROOT = pathlib.Path("/home/kewalk/memory_project/dataset/PLATE_TASK_1")


@lru_cache(maxsize=4)
def _video_frames(demo: str) -> np.ndarray:
    """Decode the top-camera mp4 once per demo. (T, H, W, 3) uint8."""
    path = DATASET_ROOT / demo / "top_camera_rgb.mp4"
    return iio.imread(str(path), plugin="pyav")


def _draw_panel(ax, img: np.ndarray, title: str, border: str | None):
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    if border is not None:
        color, style = {
            "anchor": ("tab:blue", "--"),
            "positive": ("tab:green", "-"),
            "negative": ("tab:red", "-"),
        }[border]
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3.0)
            spine.set_linestyle(style)


def _wrap(text: str, width: int = 70) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=pathlib.Path)
    p.add_argument("--features-dir", default="retriever/cache/features", type=pathlib.Path)
    p.add_argument("--labels", default="retriever/cache/labels/samples_eval.jsonl", type=pathlib.Path)
    p.add_argument("--text-emb", default="retriever/cache/text_emb.npz", type=pathlib.Path)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max-per-stage", type=int, default=None,
                   help="Limit how many viz to generate per stage (default: all)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, type=pathlib.Path,
                   help="output dir (default: <ckpt parent>/viz)")
    args = p.parse_args()

    out_dir = args.out if args.out is not None else args.ckpt.parent / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = RetrieverDataset(args.labels, args.features_dir, args.text_emb)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model = Retriever().to(args.device)
    state = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    model.load_state_dict(state["model"])
    model.eval()

    print(f"[viz] {len(ds)} samples, topk={args.topk}, out={out_dir}")
    per_stage_count: dict[int, int] = {}

    for i, batch in enumerate(loader):
        sid = int(batch["stage_id"][0])
        per_stage_count[sid] = per_stage_count.get(sid, 0)
        if args.max_per_stage is not None and per_stage_count[sid] >= args.max_per_stage:
            continue
        per_stage_count[sid] += 1

        demo = batch["demo"][0]
        anchor_fid = int(batch["anchor_frame_id"][0])
        prompt = ds.samples[i]["anchor_prompt"]
        cand_fids = batch["cand_frame_ids"][0].numpy()        # (N,)
        pos_mask = batch["positives_mask"][0].numpy()         # (N,)
        forced_mask = batch["forced_mask"][0].numpy()         # (N,)
        valid_mask = batch["valid_mask"][0].numpy()           # (N,)

        with torch.no_grad():
            scores = score_batch(model, batch, args.device).cpu().numpy()[0]  # (N,)

        # Top-K valid by score
        scores_masked = np.where(valid_mask, scores, -np.inf)
        order = np.argsort(-scores_masked)
        topk_idx = order[: args.topk]

        # Decode the frames we need
        video = _video_frames(demo)
        anchor_img = video[anchor_fid]

        # Plot
        ncols = 1 + args.topk
        fig, axes = plt.subplots(1, ncols, figsize=(2.5 * ncols, 3.0))
        stage_name = STAGE_NAMES.get(sid, f"stage{sid}")
        fig.suptitle(f"{demo}  |  {stage_name}  |  anchor frame {anchor_fid}\n"
                     f"prompt: {_wrap(prompt)}", fontsize=9)

        _draw_panel(axes[0], anchor_img, f"anchor (frame {anchor_fid})", border="anchor")

        for rank, ci in enumerate(topk_idx):
            cand_fid = int(cand_fids[ci])
            if cand_fid < 0:
                axes[1 + rank].axis("off")
                continue
            cand_img = video[cand_fid]
            label = []
            if pos_mask[ci]:
                label.append("POS")
                border = "positive"
            elif forced_mask[ci]:
                label.append("hard-neg")
                border = "negative"
            else:
                border = None
            title = f"#{rank+1}  frame {cand_fid}\nscore {scores[ci]:.2f}"
            if label:
                title += f"  [{'|'.join(label)}]"
            _draw_panel(axes[1 + rank], cand_img, title, border=border)

        plt.tight_layout(rect=(0, 0, 1, 0.92))
        fname = f"{demo}_{stage_name}_a{anchor_fid:05d}.png"
        fig.savefig(out_dir / fname, dpi=110)
        plt.close(fig)

    n_made = sum(per_stage_count.values())
    print(f"[viz] saved {n_made} figures to {out_dir}")


if __name__ == "__main__":
    main()
