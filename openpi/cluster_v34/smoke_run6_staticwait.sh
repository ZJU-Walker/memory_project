#!/bin/bash
# Integration smoke for v34_run6 (run5 + the v3.4.1 static-waiting trim).
# Validates that the trimmed phase table, the new per-step validity fields, and the repack
# structure survive real training steps.
#
# The trim is a DATA change, so device topology is irrelevant to what is under test. Override
# for whatever allocation is free, e.g. a single-GPU side job:
#   SMOKE_FSDP=1 SMOKE_BATCH=4 bash cluster_v34/smoke_run6_staticwait.sh
set -euo pipefail
set -x

cd /iris/u/kewalk/memory_project/openpi
export HOME=/iris/u/kewalk
export HF_HOME=/iris/u/kewalk/.cache/huggingface
export OPENPI_DATA_HOME=/iris/u/kewalk/.cache/openpi
export OPENPI_JAX_CACHE_DIR=/iris/u/kewalk/.cache/jax
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92

exec .venv/bin/python scripts/train.py pi05_yam_mem_v34_run6_staticwait \
  --exp-name v34_run6_staticwait_smoke \
  "$@" \
  --overwrite \
  --no-wandb-enabled \
  --num-train-steps 31 \
  --log-interval 10 \
  --save-interval 250 \
  --batch-size "${SMOKE_BATCH:-12}" \
  --gradient-accumulation-steps 1 \
  --fsdp-devices "${SMOKE_FSDP:-4}" \
  --seed 42
