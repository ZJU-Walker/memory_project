# v34_run5_eta0 pilot launch provenance

Captured after all code/test changes and the finalized four-H100 smoke, immediately before the
fresh 2,500-step pilot. The worktree is intentionally dirty; `source_snapshot.tar.gz`,
`tracked_changes.diff`, `git_status.txt`, and `key_files.sha256` are the authoritative source
record for this launch.

## Intervention

`pi05_yam_mem_v34_run5_eta0` is a fresh official `pi05_base` run. Relative to run4, the only
effective model setting change is `MemoryConfig.eta_scale=0.0` (plus config/experiment identity).
The current clipped write still applies, but the previous write's momentum is not carried into
the next write. Aux weight 0.1, theta, alpha, state/KV clips, data, optimizer, batch/FSDP, and seed
remain unchanged. This run must never restore a run4 TrainState.

`eta_scale` is static GraphDef configuration and is not stored in checkpoint arrays. Training,
resume, evaluation, and serving must explicitly use `pi05_yam_mem_v34_run5_eta0`.

## Evidence before pilot

- Fixed-K/V raw checkpoint-2250 replay: eta~0.90053 produced 9/179 gradients above 5
  (max 11.1229); eta=0 produced 0/179 (max 3.5689). Severe events were late-sequence, supporting
  previous-write momentum as the amplifier.
- One-H100 production-geometry Stage 0 (`17024084.62`) completed 0:0; all repeated,
  orthogonal, rank-1, and conflicting-input batteries passed with eta=0.
- Final CPU verification: 108 passed, 1 known long generic debug-training test deselected;
  subsequent watcher + retention focused suite: 45 passed. Ruff, Python compilation, launcher
  syntax, and `git diff --check` passed.
- Four-H100 smoke (`17024084.65`) completed 0:0 at 121/121 and atomically finalized checkpoint
  120. At step 120: loss=5.6853, CE=4.1028, aux=1.7373, write mean=1.4948, true window
  max=4.2983, severe fraction=0, memory-grad=0.3979. Every watcher row had no warning or
  violation; there was no OOM, NaN, CUDA/NCCL failure, or traceback.
- Selective Orbax restore of smoke checkpoint 120 reports TrainState step 121. Raw and EMA
  `memory/m0/{w3,b3}` plus their Adam mu/nu are exact finite FP32 zeros. Raw/EMA w2 controls are
  nonzero and differ (L2 45.18172567 / 45.19276189; diff L2 0.65188377), proving the restore is
  reading real, distinct state.

## Known behavioral limitation

Matched checkpoint-2250 heldout retention counterfactuals covered episodes {15,29,44,59} and
showed no eta=0 accuracy regression, but normal and zero-read accuracy were both 0.75. This
reveals shortcut/non-memory dominance and does not prove causal memory usefulness. The fresh
pilot is therefore a numerical-stability experiment first; causal reset/swap/retention gates
remain co-primary before claiming useful episodic memory.

## Monitoring

The pilot uses a fresh empty metrics log and a watcher bound to its exact numeric Slurm child
step. The watcher fails closed on missing, malformed, or nonfinite telemetry; retries failed
`scancel` calls without broadening the target; and treats a single 5--7.5% severe-clip window as
a warning unless repeated or corroborated by another gradient metric. It immediately stops on
severe>=7.5%, write mean>3, memory-grad>5, or nonfinite telemetry.
