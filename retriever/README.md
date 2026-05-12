# Keyframe Retriever

Coarse-to-fine attention retriever over per-episode memory banks.

Design: `/home/kewalk/.claude/plans/ok-now-lets-work-dazzling-milner.md`

## Pipeline

```text
features.py   →  cache/features/demoN.npz   (frozen SigLIP, JAX, one-shot)
labels.py     →  cache/labels/*.jsonl       (anchors + positives + hard negs)
train.py      →  runs/<exp>/best.pt         (joint coarse + reranker, PyTorch)
eval.py       →  runs/<exp>/eval.json       (Recall@K / MRR / per-stage)
```

All commands assume:

```bash
source /home/kewalk/memory_project/openpi/.venv/bin/activate
cd /home/kewalk/memory_project
```

## 1. Build the feature cache

```bash
# All 37 demos (~10 min after JIT compile, ~5 GB on disk)
python -m retriever.features --demos 1-37

# Or a subset (test/iterate)
python -m retriever.features --demos 1
python -m retriever.features --demos 1,5,36,37
```

Outputs `retriever/cache/features/demoN.npz`. Skips demos whose cache already exists.

## 2. Generate labels

```bash
python -m retriever.labels \
    --features-dir retriever/cache/features \
    --metadata-root dataset/PLATE_TASK_1 \
    --out retriever/cache/labels
```

Outputs `samples.jsonl` (demos 1-35) and `samples_eval.jsonl` (demos 36-37).

## 3. Train

```bash
# Full run
python -m retriever.train \
    --features-dir retriever/cache/features \
    --labels retriever/cache/labels/samples.jsonl \
    --epochs 50 --batch-size 8 --lr 3e-4 \
    --out retriever/runs/v1

# Smoke (1 epoch, batch 2)
python -m retriever.train --smoke --out retriever/runs/smoke
```

Saves `best.pt` (best train R@1) and `last.pt`. Log at `runs/<exp>/train_log.jsonl`.

**wandb**: enabled by default to project `retriever`, run name = `--out` leaf
(e.g. `v1`). Re-running with the same `--out` resumes the same wandb run
(id in `runs/<exp>/wandb_id.txt`).

```bash
# Custom project / run name
python -m retriever.train --wandb-project memory_retriever --wandb-name lr3e4_b8 ...

# Disable entirely
python -m retriever.train --no-wandb ...
```

Logged keys: `step/{loss,loss_main,loss_aux,r1}` per batch and
`epoch/{loss,loss_main,loss_aux,r1,elapsed_s}` per epoch.

## 4. Eval

```bash
# Held-out eval (demos 36-37)
python -m retriever.eval --ckpt retriever/runs/v1_0511/best.pt

# Smoke (falls back to samples.jsonl if eval set is empty)
python -m retriever.eval --smoke --ckpt retriever/runs/smoke/best.pt
```

Prints overall / memory-query / self-retrieval / per-stage breakdown, writes
`runs/<exp>/eval.json`.

## End-to-end smoke

```bash
python -m retriever.features --demos 1
python -m retriever.labels --features-dir retriever/cache/features \
                           --metadata-root dataset/PLATE_TASK_1 \
                           --out retriever/cache/labels
python -m retriever.train --smoke --out retriever/runs/smoke
python -m retriever.eval --smoke --ckpt retriever/runs/smoke/best.pt
```
