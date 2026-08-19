#!/bin/bash
# v3.3 write-token side probe: do the write tokens ENCODE the target side, and does the
# instruction steer that code? Diagnostic-only (no gradient touches the model).
#
# Usage on a compute node you own:
#   srun --jobid=<JOBID> --overlap bash diagnostic_outputs/adv/run_write_token_probe.sh [CKPT]
cd /iris/u/kewalk/memory_project/openpi || exit 1
export HOME=/iris/u/kewalk
export PATH="/iris/u/kewalk/.local/bin:$PATH"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7

CKPT="${1:-6500}"
DATA=/iris/u/kewalk/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask

# Smoke first: 8 episodes, so a broken path fails in ~4 min rather than ~40.
uv run python scripts/v33_write_token_probe.py \
  --checkpoint "checkpoints/pi05_yam_mem_v33/v33_run1/${CKPT}" \
  --dataset_root "$DATA" \
  --output_dir "diagnostic_outputs/v33_write_probe/${CKPT}_smoke" \
  --max_episodes 8 || exit 1
echo "=== SMOKE OK, running all 60 episodes ==="

# Full run: every usable episode, leave-one-episode-out over all four cells.
exec uv run python scripts/v33_write_token_probe.py \
  --checkpoint "checkpoints/pi05_yam_mem_v33/v33_run1/${CKPT}" \
  --dataset_root "$DATA" \
  --output_dir "diagnostic_outputs/v33_write_probe/${CKPT}"
