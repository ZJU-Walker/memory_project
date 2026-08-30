#!/bin/bash
# v34_run5: fresh official pi05 base with run4's blank output and eta_scale=0; all other
# training settings are identical to run4. Pass --resume explicitly to continue this run.
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
  --exp-name v34_run5_eta0 \
  "$@" \
  --no-overwrite \
  --num-train-steps 20000 \
  --log-interval 100 \
  --save-interval 250 \
  --keep-period 5000 \
  --batch-size 12 \
  --gradient-accumulation-steps 1 \
  --fsdp-devices 4 \
  --seed 42
