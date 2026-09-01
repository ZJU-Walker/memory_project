#!/usr/bin/env bash
set -euo pipefail

v35_launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=env.sh
source "${v35_launcher_dir}/env.sh"

v35_python="${V35_PYTHON:-${MEMORY_PROJECT_ROOT}/openpi/.venv/bin/python}"
if [[ ! -x "${v35_python}" ]]; then
  echo "v3.5 Python is not executable: ${v35_python}" >&2
  echo "Create the environment with uv, or set V35_PYTHON to this cluster's interpreter." >&2
  exit 2
fi

cd -- "${MEMORY_PROJECT_ROOT}/openpi"
exec "${v35_python}" scripts/compute_norm_stats.py \
  --config-name pi05_yam_mem_v35 \
  --manifest-sha256 9085fe50d7b02ea65930f3647ce0413e0583a66d430484e06c60812c52af8442
