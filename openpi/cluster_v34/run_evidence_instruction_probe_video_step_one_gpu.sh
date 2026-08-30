#!/bin/bash
set -euo pipefail

mode="${1:?usage: $0 smoke|full NUMERIC_CHECKPOINT_STEP}"
step="${2:?usage: $0 smoke|full NUMERIC_CHECKPOINT_STEP}"
if [[ "$mode" != "smoke" && "$mode" != "full" ]]; then
    echo "mode must be smoke or full" >&2
    exit 2
fi
if [[ ! "$step" =~ ^[0-9]+$ ]]; then
    echo "checkpoint step must be numeric" >&2
    exit 2
fi

job_id=17099350
serve_step="${job_id}.37"
node=iris-hgx-2
repo=/iris/u/kewalk/memory_project/openpi
dataset=/iris/u/kewalk/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask
archive_checkpoint="$repo/diagnostic_checkpoints/v34_run5_eta0_resume_copies/$step"
live_checkpoint="$repo/checkpoints/pi05_yam_mem_v34_run5_eta0/v34_run5_eta0/$step"
if [[ -f "$archive_checkpoint/_CHECKPOINT_METADATA" && -f "$archive_checkpoint.manifest.json" ]]; then
    checkpoint="$archive_checkpoint"
elif [[ -f "$live_checkpoint/_CHECKPOINT_METADATA" ]]; then
    checkpoint="$live_checkpoint"
else
    echo "no finalized archive or live checkpoint found for step $step" >&2
    exit 1
fi
probe_artifact="$repo/diagnostic_outputs/v34_fixed_writer_probe/full_${job_id}/$step/raw"
snapshot="$repo/cluster_v34/provenance/v34_run5_eta0_main_launch/source_snapshot.tar.gz"
snapshot_sha=ebb8aa0a9ebf6d1ea9af932cd818b090d02f38a4a61a191c78ac8127f13fbf16
holder_pid=3035649
serve_pid=3559973
serve_uuid=GPU-dcaeae50-e4f3-ee9f-02ab-936c7c2b36ba
probe_uuid=GPU-52661869-33d0-aea6-7bb3-d9d119b05dc3
output_root="$repo/diagnostic_outputs/v34_heldout_evidence_instruction_dual_head/${mode}_${job_id}/$step"
lock_dir="$repo/diagnostic_outputs/.v34_gpu1_${job_id}.lock"

serve_state=$(sacct -j "$serve_step" -n -P --format=State | sed -n '1{s/[[:space:]]*$//;p}')
if [[ "$serve_state" != "RUNNING" ]]; then
    echo "refusing launch: expected $serve_step RUNNING, got $serve_state" >&2
    exit 1
fi
active_steps=$(squeue --steps -h -j "$job_id" -o '%i|%j')
unexpected_steps=$(printf '%s\n' "$active_steps" | awk -F'|' -v keep="$serve_step" '$1 !~ /\.(batch|extern)$/ && $1 != keep')
if [[ -n "$unexpected_steps" ]]; then
    echo "refusing launch: unexpected active steps" >&2
    printf '%s\n' "$unexpected_steps" >&2
    exit 1
fi

mkdir "$lock_dir"
holder_paused=false

signal_holder() {
    local signal="$1"
    ssh "$node" bash -s -- "$holder_pid" "$signal" <<'REMOTE'
set -euo pipefail
pid="$1"
signal="$2"
expected_python=/iris/u/kewalk/openpi_trossen/.venv/bin/python3
expected_cwd=/iris/u/kewalk/openpi_trossen
expected_arg=cluster_scripts/train_hs.py
[[ -r "/proc/$pid/cmdline" ]]
[[ "$(stat -c %u "/proc/$pid")" == 24706 ]]
[[ "$(readlink -f "/proc/$pid/cwd")" == "$expected_cwd" ]]
mapfile -d "" argv < "/proc/$pid/cmdline"
[[ "${#argv[@]}" == 2 ]]
[[ "${argv[0]}" == "$expected_python" ]]
[[ "${argv[1]}" == "$expected_arg" ]]
kill "-$signal" "$pid"
for _ in 1 2 3 4 5; do
    state=$(awk '/^State:/ {print $2}' "/proc/$pid/status")
    if [[ "$signal" == STOP && "$state" == T ]] || [[ "$signal" == CONT && "$state" != T ]]; then
        exit 0
    fi
    sleep 0.1
done
exit 1
REMOTE
}

cleanup() {
    status=$?
    if [[ "$holder_paused" == true ]]; then
        signal_holder CONT || status=99
    fi
    rmdir "$lock_dir" 2>/dev/null || true
    exit "$status"
}
trap cleanup EXIT INT TERM HUP

[[ -d "$checkpoint" ]]
[[ -f "$checkpoint/_CHECKPOINT_METADATA" ]]
if [[ "$checkpoint" == "$archive_checkpoint" ]]; then
    [[ -f "$checkpoint.manifest.json" ]]
fi
[[ -f "$probe_artifact/COMPLETE" ]]
(cd "$probe_artifact" && sha256sum -c COMPLETE)
[[ ! -e "$output_root/raw" ]]
[[ "$(sha256sum "$snapshot" | awk '{print $1}')" == "$snapshot_sha" ]]

gpu_apps=$(ssh "$node" "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader")
printf '%s\n' "$gpu_apps" | grep -Fx "$serve_uuid, $holder_pid" >/dev/null
printf '%s\n' "$gpu_apps" | grep -Fx "$serve_uuid, $serve_pid" >/dev/null
if [[ "$(printf '%s\n' "$gpu_apps" | grep -Fc "$probe_uuid, $holder_pid")" != 1 ]]; then
    echo "refusing launch: probe GPU is not holder-only" >&2
    printf '%s\n' "$gpu_apps" >&2
    exit 1
fi

signal_holder STOP
holder_paused=true
gpu_apps=$(ssh "$node" "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader")
printf '%s\n' "$gpu_apps" | grep -Fx "$serve_uuid, $serve_pid" >/dev/null
if [[ "$(printf '%s\n' "$gpu_apps" | grep -Fc "$probe_uuid, $holder_pid")" != 1 ]] || \
   [[ "$(printf '%s\n' "$gpu_apps" | grep -Fc "$probe_uuid")" != 1 ]]; then
    echo "refusing launch: probe GPU has a process other than the stopped holder" >&2
    printf '%s\n' "$gpu_apps" >&2
    exit 1
fi
[[ "$(ssh "$node" "awk '/^State:/ {print \$2}' /proc/$holder_pid/status")" == T ]]

srun \
    --overlap --exact \
    --jobid="$job_id" \
    --nodes=1 --nodelist="$node" \
    --ntasks=1 --cpus-per-task=16 --cpu-bind=cores \
    --gpus-per-task=2 --gpu-bind=map_gpu:1 \
    --mem=400G --time=03:00:00 \
    --kill-on-bad-exit=1 \
    --job-name="v34-dual-head-prompt-${mode}-${step}" \
    bash -s -- "$repo" "$dataset" "$checkpoint" "$probe_artifact" "$snapshot" "$snapshot_sha" "$output_root" "$mode" "$probe_uuid" <<'REMOTE'
set -euo pipefail
repo="$1"
dataset="$2"
checkpoint="$3"
probe_artifact="$4"
snapshot="$5"
snapshot_sha="$6"
output_root="$7"
mode="$8"
probe_uuid="$9"

[[ "$(nvidia-smi --query-gpu=uuid --format=csv,noheader)" == "$probe_uuid" ]]
[[ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" == "3035649" ]]
[[ "$(awk '/^State:/ {print $2}' /proc/3035649/status)" == T ]]
[[ "$(sha256sum "$snapshot" | awk '{print $1}')" == "$snapshot_sha" ]]
[[ -f "$probe_artifact/COMPLETE" ]]
(cd "$probe_artifact" && sha256sum -c COMPLETE)

source_root=$(mktemp -d "/tmp/v34-evidence-prompt-${SLURM_JOB_ID}.${SLURM_STEP_ID}.XXXXXX")
cleanup_source() {
    find "$source_root" -depth -type f -delete
    find "$source_root" -depth -type d -empty -delete
}
trap cleanup_source EXIT
tar -xzf "$snapshot" -C "$source_root"
export HF_HOME=/iris/u/kewalk/.cache/huggingface
export OPENPI_DATA_HOME=/iris/u/kewalk/.cache/openpi
export OPENPI_JAX_CACHE_DIR=/iris/u/kewalk/.cache/jax
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export V34_RUN5_SOURCE_ROOT="$source_root"
export PYTHONPATH="$source_root/src"

cd "$repo"
extra=()
if [[ "$mode" == smoke ]]; then
    extra+=(--smoke-only)
fi
.venv/bin/python -u scripts/v34_heldout_evidence_instruction_probe_video.py \
    --checkpoint "$checkpoint" \
    --dataset-root "$dataset" \
    --probe-artifact-dir "$probe_artifact" \
    --output-dir "$output_root" \
    --config pi05_yam_mem_v34_run5_eta0 \
    --parameter-source raw \
    --batch-size 8 \
    "${extra[@]}" </dev/null
[[ -f "$output_root/raw/COMPLETE" ]]
(cd "$output_root/raw" && sha256sum -c COMPLETE)
REMOTE

signal_holder CONT
holder_paused=false
rmdir "$lock_dir"
trap - EXIT INT TERM HUP
