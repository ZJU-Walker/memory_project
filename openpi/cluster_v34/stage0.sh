#!/bin/bash
# v3.4 Stage-0 synthetic memory-core battery on the training hardware (plan 9, mandatory).
set -x
cd /iris/u/kewalk/memory_project/openpi || exit 1
export HOME=/iris/u/kewalk
export HF_HOME=/iris/u/kewalk/.cache/huggingface
export OPENPI_DATA_HOME=/iris/u/kewalk/.cache/openpi
export CUDA_VISIBLE_DEVICES=0
exec .venv/bin/python scripts/v34_stage0_memory_core.py
