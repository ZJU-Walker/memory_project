#!/usr/bin/env bash

# Portable v3.5 runtime environment. Source this file from any cluster; every mutable
# cache/output path is derived from the copied memory_project directory.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "source cluster_v35/env.sh instead of executing it" >&2
  exit 2
fi

v35_env_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export MEMORY_PROJECT_ROOT="$(cd -- "${v35_env_dir}/../.." && pwd -P)"

export UV_CACHE_DIR="${MEMORY_PROJECT_ROOT}/v35/cache/uv"
export HF_HOME="${MEMORY_PROJECT_ROOT}/v35/cache/huggingface"
export HF_LEROBOT_HOME="${MEMORY_PROJECT_ROOT}/data/lerobot"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export OPENPI_DATA_HOME="${MEMORY_PROJECT_ROOT}/v35/cache/openpi"
export OPENPI_JAX_CACHE_DIR="${MEMORY_PROJECT_ROOT}/v35/cache/jax"
export TMPDIR="${MEMORY_PROJECT_ROOT}/v35/tmp"
export WANDB_DIR="${MEMORY_PROJECT_ROOT}/v35/wandb"
# Online is the portable default: a new cluster may populate its project-local caches from
# the official sources. Set V35_OFFLINE=1 only after all required artifacts are present.
v35_offline="${V35_OFFLINE:-0}"
if [[ "${v35_offline}" != "0" && "${v35_offline}" != "1" ]]; then
  echo "V35_OFFLINE must be 0 or 1, got: ${v35_offline}" >&2
  return 2
fi
export HF_HUB_OFFLINE="${v35_offline}"
export TRANSFORMERS_OFFLINE="${v35_offline}"
export HF_DATASETS_OFFLINE="${v35_offline}"

mkdir -p \
  "${MEMORY_PROJECT_ROOT}/v35/assets" \
  "${MEMORY_PROJECT_ROOT}/v35/checkpoints" \
  "${MEMORY_PROJECT_ROOT}/v35/diagnostics" \
  "${UV_CACHE_DIR}" \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${OPENPI_DATA_HOME}" \
  "${OPENPI_JAX_CACHE_DIR}" \
  "${TMPDIR}" \
  "${WANDB_DIR}"

unset v35_env_dir v35_offline
