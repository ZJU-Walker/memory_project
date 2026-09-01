#!/usr/bin/env bash
set -euo pipefail

v35_launcher_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=env.sh
source "${v35_launcher_dir}/env.sh"

v35_python="${V35_PYTHON:-${MEMORY_PROJECT_ROOT}/openpi/.venv/bin/python}"
if [[ ! -x "${v35_python}" ]]; then
  echo "v3.5 Python is not executable: ${v35_python}" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.92}"
cd -- "${MEMORY_PROJECT_ROOT}/openpi"
exec "${v35_python}" scripts/v35_step0_bootstrap.py "$@"
