#!/usr/bin/env bash

# Watch the formal v3.1 H100 run from the Slurm login node.  The watchdog only
# restarts after the training process has disappeared twice in succession, and
# gives up after a bounded number of failures so that a deterministic bug does
# not create an infinite restart loop.  Create STOP_FILE to disable relaunches.

set -u

JOB_ID="${JOB_ID:-16721231}"
COMPUTE_HOST="${COMPUTE_HOST:-iris-hgx-1}"
REPO="${REPO:-/iris/u/kewalk/memory_project/openpi}"
EXP_NAME="${EXP_NAME:-attnwrite_base_s10_d6_t60_b20-40-60_tb25_bs12_seed42}"
TRAIN_LOG="${TRAIN_LOG:-${REPO}/training_logs/v31_t60_buckets_seed42.log}"
WATCH_LOG="${WATCH_LOG:-${REPO}/training_logs/v31_t60_buckets_seed42.watchdog.log}"
STOP_FILE="${STOP_FILE:-${REPO}/training_logs/v31_t60_buckets_seed42.stop}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REPO}/checkpoints/pi05_yam_mem_v31/${EXP_NAME}}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STALL_SECONDS="${STALL_SECONDS:-1800}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
FINAL_STEP="${FINAL_STEP:-20000}"

restarts=0

log() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"${WATCH_LOG}"
}

allocation_is_running() {
    squeue -h -j "${JOB_ID}" -t R -o '%i' 2>/dev/null | grep -qx "${JOB_ID}"
}

training_is_running() {
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${COMPUTE_HOST}" \
        "pgrep -f 'scripts/train.py pi05_yam_mem_v31 --exp-name ${EXP_NAME}' >/dev/null" \
        2>/dev/null
}

training_has_nonfinite_metrics() {
    tail -n 250 "${TRAIN_LOG}" 2>/dev/null \
        | grep -Eiq 'Step [0-9]+:.*(ce_loss|flow_loss|grad_norm|loss|param_norm|probe_loss)=(nan|[-+]?inf)(,|$)'
}

training_log_is_stale() {
    modified="$(stat -c '%Y' "${TRAIN_LOG}" 2>/dev/null || printf '0')"
    now="$(date +%s)"
    ((modified > 0 && now - modified > STALL_SECONDS))
}

stop_training() {
    reason="$1"
    log "stopping unhealthy training process: ${reason}"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${COMPUTE_HOST}" \
        "ps -eo pid=,comm=,args= | awk '\$2 ~ /python/ && index(\$0, \"scripts/train.py pi05_yam_mem_v31 --exp-name ${EXP_NAME}\") { print \$1 }' | xargs -r kill -TERM" \
        >>"${WATCH_LOG}" 2>&1 || true
}

latest_checkpoint_step() {
    find "${CHECKPOINT_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
        | awk '/^[0-9]+$/ { if ($1 > max) max=$1 } END { print max+0 }'
}

launch_training() {
    log "launching retry $((restarts + 1))/${MAX_RESTARTS} from latest checkpoint"
    srun --jobid="${JOB_ID}" --overlap --nodes=1 --ntasks=1 bash -lc "
        cd '${REPO}'
        exec env CUDA_VISIBLE_DEVICES=0,1 \
            XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
            HF_HOME=/iris/u/kewalk/.cache/huggingface \
            HF_HUB_OFFLINE=1 \
            TRANSFORMERS_OFFLINE=1 \
            OPENPI_DATA_HOME=/iris/u/kewalk/.cache/openpi \
            .venv/bin/python -u scripts/train.py pi05_yam_mem_v31 \
            --exp-name '${EXP_NAME}' \
            --batch-size 12 \
            --gradient-accumulation-steps 6 \
            --fsdp-devices 2 \
            --seed 42 \
            --resume \
            >> '${TRAIN_LOG}' 2>&1
    " >>"${WATCH_LOG}" 2>&1
    status=$?
    restarts=$((restarts + 1))
    log "retry ${restarts} exited with status ${status}"
}

mkdir -p "$(dirname "${WATCH_LOG}")"
log "watchdog started for Slurm job ${JOB_ID}; stop file: ${STOP_FILE}"

while allocation_is_running; do
    if [[ -e "${STOP_FILE}" ]]; then
        log "stop file present; exiting without relaunch"
        exit 0
    fi

    if training_is_running; then
        if training_has_nonfinite_metrics; then
            stop_training "non-finite scalar metric"
            sleep 60
            continue
        fi
        if training_log_is_stale; then
            stop_training "no log progress for more than ${STALL_SECONDS} seconds"
            sleep 60
            continue
        fi
        sleep "${POLL_SECONDS}"
        continue
    fi

    # Avoid racing a process that is between Slurm launch and Python startup.
    sleep 20
    if training_is_running; then
        continue
    fi

    step="$(latest_checkpoint_step)"
    if ((step >= FINAL_STEP)); then
        log "training complete at checkpoint ${step}"
        exit 0
    fi
    if ((restarts >= MAX_RESTARTS)); then
        log "maximum restart count reached at checkpoint ${step}; manual intervention required"
        exit 1
    fi

    log "training process missing at checkpoint ${step}"
    launch_training
done

log "Slurm allocation ${JOB_ID} is no longer running; exiting"
