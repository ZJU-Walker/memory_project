#!/bin/bash
# v34_run6: v34_run5 (blank episode output + eta_scale=0) with the v3.4.1 waiting-leak fix 1 --
# the memory-required phase is trimmed to each episode's stationary core, so memory-critical
# endpoints can no longer land on frames where the arm is already moving toward a bin
# (17.1% of endpoints before the trim, 0.0% after). Every model/optimizer setting matches run5;
# the training DATA is the only variable. Pass --resume explicitly to continue this run.
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
  --exp-name v34_run6_staticwait \
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
