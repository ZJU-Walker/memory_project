#!/bin/bash
# Fresh four-H100 integration smoke for the isolated eta_scale=0 run5 configuration.
set -euo pipefail
set -x

cd /iris/u/kewalk/memory_project/openpi
export HOME=/iris/u/kewalk
export HF_HOME=/iris/u/kewalk/.cache/huggingface
export OPENPI_DATA_HOME=/iris/u/kewalk/.cache/openpi
export OPENPI_JAX_CACHE_DIR=/iris/u/kewalk/.cache/jax
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92

exec .venv/bin/python scripts/train.py pi05_yam_mem_v34_run5_eta0 \
  --exp-name v34_run5_eta0_smoke \
  "$@" \
  --no-overwrite \
  --no-wandb-enabled \
  --num-train-steps 121 \
  --log-interval 10 \
  --save-interval 250 \
  --batch-size 12 \
  --gradient-accumulation-steps 1 \
  --fsdp-devices 4 \
  --seed 42
