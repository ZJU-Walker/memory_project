#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 [--dry-run] [--verify-dataset] USER@HOST:/absolute/path/to/memory_project/" >&2
}

v35_sync_dry_run=0
v35_sync_verify_dataset=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run)
      v35_sync_dry_run=1
      shift
      ;;
    --verify-dataset)
      v35_sync_verify_dataset=1
      shift
      ;;
    --)
      shift
      break
      ;;
    -*)
      usage
      exit 2
      ;;
    *)
      break
      ;;
  esac
done
if [[ "$#" -ne 1 || -z "${1}" ]]; then
  usage
  exit 2
fi
v35_sync_destination="$1"

v35_sync_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
v35_sync_root="$(cd -- "${v35_sync_dir}/../.." && pwd -P)"
v35_sync_manifest="${v35_sync_root}/data/0830_0831_episode_manifest_v36_frozen.json"
v35_sync_dataset="${v35_sync_root}/data/lerobot/yam/bin_memory_0830_0831_v36_subtask"
v35_sync_inventory="${v35_sync_root}/data/0830_0831_v36_dataset_tree_inventory.json"
v35_sync_norm="${v35_sync_root}/v35/assets/pi05_yam_0830_0831_v36/yam/bin_memory_0830_0831_v36_subtask/norm_stats.json"
v35_sync_norm_provenance="${v35_sync_root}/v35/assets/pi05_yam_0830_0831_v36/yam/bin_memory_0830_0831_v36_subtask/norm_stats_provenance.json"

if [[ ! -d "${v35_sync_dataset}" || ! -f "${v35_sync_norm}" || ! -f "${v35_sync_norm_provenance}" || ! -f "${v35_sync_manifest}" || ! -f "${v35_sync_inventory}" ]]; then
  echo "portable v3.5 data/norm layout is incomplete under ${v35_sync_root}" >&2
  exit 2
fi

verify_sha256() {
  local v35_sync_file="$1"
  local v35_sync_expected="$2"
  local v35_sync_label="$3"
  local v35_sync_actual
  v35_sync_actual="$(sha256sum "${v35_sync_file}")"
  v35_sync_actual="${v35_sync_actual%% *}"
  if [[ "${v35_sync_actual}" != "${v35_sync_expected}" ]]; then
    echo "${v35_sync_label} SHA-256 mismatch: expected ${v35_sync_expected}, got ${v35_sync_actual}" >&2
    exit 2
  fi
}

verify_sha256 "${v35_sync_manifest}" "9085fe50d7b02ea65930f3647ce0413e0583a66d430484e06c60812c52af8442" "frozen manifest"
verify_sha256 "${v35_sync_inventory}" "193561bbde0fc5de586140e8cf4a9432b6d4f8b590176e9bd85519170d08d172" "dataset inventory"
verify_sha256 "${v35_sync_norm}" "5535ea95ad7ed1edc399ba47e278285a1fdec02a589451cec0dc9003d458519c" "train-only norm stats"
verify_sha256 "${v35_sync_norm_provenance}" "36e0ab51d53272038e1b204c752b30fa6ab000096bff9ff6dccd605166188c58" "norm provenance"

v35_sync_options=(
  -a
  --partial
  --info=progress2
  --exclude=/openpi/.venv/
  --exclude=/openpi/checkpoints/
  --exclude=/openpi/diagnostic_checkpoints/
  --exclude=/v35/cache/
  --exclude=/v35/checkpoints/
  --exclude=/v35/tmp/
  --exclude=/v35/wandb/
)
if [[ "${v35_sync_dry_run}" -eq 1 ]]; then
  v35_sync_options+=(--dry-run)
fi

# Deliberately no --delete: source uploads, destination experiments, and historical runs are
# never removed by this helper. Downloadable caches and cluster-local checkpoints are omitted;
# frozen data, norm assets, gate evidence, code, and manifests are transferred.
rsync "${v35_sync_options[@]}" "${v35_sync_root}/" "${v35_sync_destination}"

if [[ "${v35_sync_verify_dataset}" -eq 1 && "${v35_sync_dry_run}" -eq 0 ]]; then
  if [[ ! "${v35_sync_destination}" =~ ^([A-Za-z0-9._@-]+):(/[A-Za-z0-9._/-]+/?)$ ]]; then
    echo "--verify-dataset requires a simple USER@HOST:/absolute/path destination" >&2
    exit 2
  fi
  v35_sync_host="${BASH_REMATCH[1]}"
  v35_sync_remote_root="${BASH_REMATCH[2]%/}"
  # The verifier is stdlib-only and hashes all 202 destination files against the canonical
  # inventory. It is deliberately optional because reading the full 58.5 GB tree is costly.
  ssh -- "${v35_sync_host}" python3 \
    "${v35_sync_remote_root}/openpi/scripts/relocate_v35_dataset.py" verify \
    --tree "${v35_sync_remote_root}/data/lerobot/yam/bin_memory_0830_0831_v36_subtask" \
    --inventory "${v35_sync_remote_root}/data/0830_0831_v36_dataset_tree_inventory.json"
fi
