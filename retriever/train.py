"""Train the coarse retriever + attention reranker jointly.

Per anchor forward pass:
  1. Coarse score over all causal candidates (variable N, padded to N_MAX, masked).
  2. Select top-M by coarse score; force-include labeled positives and hard-negs.
  3. Rerank only those M candidates.
  4. Compute multi-positive InfoNCE on combined logits (coarse + rerank) over M.
  5. Aux coarse loss (weight 0.1) on raw coarse logits over the full causal set.

Usage:
    python -m retriever.train \\
        --features-dir retriever/cache/features \\
        --labels retriever/cache/labels/samples.jsonl \\
        --epochs 50 --batch-size 8 --lr 3e-4 \\
        --out retriever/runs/v1

    python -m retriever.train --smoke --out retriever/runs/smoke
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from retriever.dataset import RetrieverDataset
from retriever.model import Retriever

M_TOPK = 32              # candidates fed to the reranker
NEG_INF = -1e9
AUX_COARSE_WEIGHT = 0.1


def _select_topm(coarse_logits: torch.Tensor, forced_mask: torch.Tensor, valid_mask: torch.Tensor, M: int) -> torch.Tensor:
    """Pick M indices per batch row: force-include `forced_mask`, fill remainder with top coarse logits.

    Returns idx (B, M) into the candidate dim. Padded slots (where fewer than M valid
    candidates exist) are filled by repeating the first valid index — the reranker
    will see them anyway, and `valid_mask` gated downstream prevents leakage.
    """
    B, N = coarse_logits.shape
    masked_logits = coarse_logits.masked_fill(~valid_mask, NEG_INF)
    # Boost forced positions so they are guaranteed to be in the top-M.
    boosted = masked_logits + forced_mask.float() * 1e6
    idx = boosted.topk(M, dim=-1).indices  # (B, M)
    return idx


def loss_and_metrics(model: Retriever, batch: dict, device: str) -> tuple[torch.Tensor, dict]:
    text_emb = batch["text_emb"].to(device)
    cur_global = batch["cur_global"].to(device)
    cur_patches = batch["cur_patches"].to(device)
    cand_global = batch["cand_global"].to(device)
    cand_patches = batch["cand_patches"].to(device)
    dt = batch["dt"].to(device)
    valid = batch["valid_mask"].to(device)
    pos = batch["positives_mask"].to(device)
    forced = batch["forced_mask"].to(device)

    B, N, _ = cand_global.shape

    # --- Coarse ---
    coarse = model.coarse(text_emb, cur_global, cand_global, dt)             # (B, N)

    # --- Top-M selection (with forced-include) ---
    idx_m = _select_topm(coarse, forced, valid, M_TOPK)                       # (B, M)
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, M_TOPK)
    cand_patches_m = cand_patches[b_idx, idx_m]                               # (B, M, P, D)
    dt_m = dt[b_idx, idx_m]                                                   # (B, M)
    valid_m = valid[b_idx, idx_m]                                             # (B, M)
    pos_m = pos[b_idx, idx_m]                                                 # (B, M)
    coarse_m = coarse.gather(1, idx_m)                                        # (B, M)

    # --- Rerank ---
    rerank = model.reranker(text_emb, cur_patches, cand_patches_m, dt_m)      # (B, M)

    combined = coarse_m + rerank
    combined = combined.masked_fill(~valid_m, NEG_INF)

    # Multi-positive InfoNCE on combined.
    log_probs = F.log_softmax(combined, dim=-1)
    has_pos = pos_m.any(dim=-1)
    if has_pos.any():
        pos_count = pos_m.float().sum(-1).clamp(min=1)
        sample_loss = -(log_probs * pos_m.float()).sum(-1) / pos_count
        loss_main = sample_loss[has_pos].mean()
    else:
        loss_main = combined.new_zeros(())

    # Aux coarse loss over full causal set.
    coarse_full = coarse.masked_fill(~valid, NEG_INF)
    log_probs_c = F.log_softmax(coarse_full, dim=-1)
    has_pos_full = pos.any(dim=-1)
    if has_pos_full.any():
        pos_count_full = pos.float().sum(-1).clamp(min=1)
        sample_loss_c = -(log_probs_c * pos.float()).sum(-1) / pos_count_full
        loss_aux = sample_loss_c[has_pos_full].mean()
    else:
        loss_aux = coarse_full.new_zeros(())

    loss = loss_main + AUX_COARSE_WEIGHT * loss_aux

    # Quick metric: recall@1 on combined (within top-M).
    top1 = combined.argmax(-1)
    top1_pos = pos_m.gather(1, top1.unsqueeze(-1)).squeeze(-1)
    r1 = top1_pos.float()[has_pos].mean() if has_pos.any() else combined.new_zeros(())

    return loss, {
        "loss": float(loss.detach()),
        "loss_main": float(loss_main.detach()),
        "loss_aux": float(loss_aux.detach()),
        "r1": float(r1.detach()),
        "n_pos_samples": int(has_pos.sum()),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", default="retriever/cache/features", type=pathlib.Path)
    p.add_argument("--labels", default="retriever/cache/labels/samples.jsonl", type=pathlib.Path)
    p.add_argument("--text-emb", default="retriever/cache/text_emb.npz", type=pathlib.Path)
    p.add_argument("--out", default="retriever/runs/v1", type=pathlib.Path)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--smoke", action="store_true", help="1 epoch, batch 2, runs quickly")
    p.add_argument("--wandb-project", default="retriever")
    p.add_argument("--wandb-name", default=None, help="run name (default: --out leaf)")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    if args.smoke:
        args.epochs = 1
        args.batch_size = 2

    args.out.mkdir(parents=True, exist_ok=True)

    # Init wandb: resume if wandb_id.txt exists in args.out, else fresh.
    wandb_id_path = args.out / "wandb_id.txt"
    if args.no_wandb:
        wandb.init(mode="disabled")
    elif wandb_id_path.exists():
        wandb.init(
            id=wandb_id_path.read_text().strip(),
            resume="must",
            project=args.wandb_project,
        )
    else:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or args.out.name,
            config={k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()},
        )
        if wandb.run is not None and wandb.run.id:
            wandb_id_path.write_text(wandb.run.id)
    ds = RetrieverDataset(args.labels, args.features_dir, args.text_emb)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    print(f"[train] {len(ds)} samples, batch={args.batch_size}, epochs={args.epochs}, device={args.device}")

    model = Retriever().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = args.out / "train_log.jsonl"
    best_r1 = -1.0
    global_step = 0
    with log_path.open("w") as logf:
        epoch_bar = tqdm(range(args.epochs), desc="epoch", position=0, leave=True)
        for epoch in epoch_bar:
            t0 = time.time()
            tot = {"loss": 0.0, "loss_main": 0.0, "loss_aux": 0.0, "r1": 0.0, "n": 0}
            model.train()
            step_bar = tqdm(loader, desc=f"e{epoch:03d}", position=1, leave=False)
            for batch in step_bar:
                opt.zero_grad()
                loss, m = loss_and_metrics(model, batch, args.device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot["loss"] += m["loss"]; tot["loss_main"] += m["loss_main"]
                tot["loss_aux"] += m["loss_aux"]; tot["r1"] += m["r1"]; tot["n"] += 1
                step_bar.set_postfix(loss=f"{m['loss']:.3f}", r1=f"{m['r1']:.2f}")
                wandb.log({
                    "step/loss": m["loss"], "step/loss_main": m["loss_main"],
                    "step/loss_aux": m["loss_aux"], "step/r1": m["r1"],
                    "epoch": epoch,
                }, step=global_step)
                global_step += 1
            step_bar.close()
            n = max(tot["n"], 1)
            log = {
                "epoch": epoch,
                "loss": tot["loss"] / n,
                "loss_main": tot["loss_main"] / n,
                "loss_aux": tot["loss_aux"] / n,
                "r1": tot["r1"] / n,
                "elapsed_s": time.time() - t0,
            }
            logf.write(json.dumps(log) + "\n"); logf.flush()
            epoch_bar.set_postfix(loss=f"{log['loss']:.3f}", main=f"{log['loss_main']:.3f}",
                                  aux=f"{log['loss_aux']:.3f}", r1=f"{log['r1']:.3f}",
                                  best=f"{max(best_r1, log['r1']):.3f}")
            wandb.log({
                "epoch/loss": log["loss"], "epoch/loss_main": log["loss_main"],
                "epoch/loss_aux": log["loss_aux"], "epoch/r1": log["r1"],
                "epoch/elapsed_s": log["elapsed_s"], "epoch_num": epoch,
            }, step=global_step)

            if log["r1"] > best_r1:
                best_r1 = log["r1"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "r1": best_r1},
                           args.out / "best.pt")

    torch.save({"model": model.state_dict(), "epoch": args.epochs - 1},
               args.out / "last.pt")
    wandb.run.summary["best_r1"] = best_r1
    wandb.finish()
    print(f"[train] done. best r1={best_r1:.3f}. ckpts in {args.out}")


if __name__ == "__main__":
    main()
