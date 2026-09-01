#!/bin/bash
# Banana-0630 transfer probe: fan 9 checkpoints across 4 GPUs of allocation 17074121.
# usage: run_banana0630_eval.sh [smoke|full]
set -euo pipefail
mode="${1:?usage: $0 smoke|full}"
job_id=17074121
node=iris-hgx-1
repo=/iris/u/kewalk/memory_project/openpi
ckroot=$repo/checkpoints/pi05_yam_mem_v34_run5_eta0/v34_run5_eta0
pilot=$repo/diagnostic_checkpoints/v34_run5_eta0_pilot_copies
resume=$repo/diagnostic_checkpoints/v34_run5_eta0_resume_copies
art99=$repo/diagnostic_outputs/v34_fixed_writer_probe/full_17099350
art74=$repo/diagnostic_outputs/v34_fixed_writer_probe/full_17074121
outroot=$repo/diagnostic_outputs/v34_banana0630_writer_probe
snapshot=$repo/cluster_v34/provenance/v34_run5_eta0_main_launch/source_snapshot.tar.gz

worker() {
    local name="$1"; shift
    srun --overlap --exact --jobid="$job_id" \
        --nodes=1 --nodelist="$node" \
        --ntasks=1 --cpus-per-task=12 --cpu-bind=cores \
        --gpus-per-task=1 --mem=150G --time=05:00:00 \
        --kill-on-bad-exit=1 --job-name="$name" \
        bash -s -- "$repo" "$snapshot" "$@" <<'REMOTE'
set -euo pipefail
repo="$1"; snapshot="$2"; shift 2
echo "[worker] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} tasks=$#"
source_root=$(mktemp -d "/tmp/v34-ban-${SLURM_JOB_ID}.${SLURM_STEP_ID}.XXXXXX")
trap 'find "$source_root" -depth -delete' EXIT
tar -xzf "$snapshot" -C "$source_root"
export HF_HOME=/iris/u/kewalk/.cache/huggingface
export OPENPI_DATA_HOME=/iris/u/kewalk/.cache/openpi
export OPENPI_JAX_CACHE_DIR=/iris/u/kewalk/.cache/jax
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export V34_RUN5_SOURCE_ROOT="$source_root"
export PYTHONPATH="$source_root/src"
cd "$repo"
for task in "$@"; do
    IFS='|' read -r step ckpt artifact extra <<< "$task"
    echo "=== [worker] checkpoint $step ==="
    .venv/bin/python -u scripts/v34_banana0630_writer_probe_eval.py \
        --checkpoint "$ckpt" \
        --probe-artifact-dir "$artifact" \
        --output-dir "$repo/diagnostic_outputs/v34_banana0630_writer_probe/$step" \
        --parameter-source raw \
        --batch-size 8 --frame-stride 4 $extra </dev/null
done
REMOTE
}

if [[ "$mode" == smoke ]]; then
    rm -rf "$outroot/2500smoke"
    worker v34-ban-smoke "2500smoke|$pilot/2500|$art99/2500/raw|--limit-episodes 2"
    echo "=== smoke done ==="
    exit 0
fi

worker v34-ban-w1 "500|$pilot/500|$art99/500/raw|" "9500|$resume/9500|$art99/9500/raw|" &
worker v34-ban-w2 "1000|$pilot/1000|$art99/1000/raw|" "10000|$ckroot/10000|$art99/10000/raw|" "15750|$ckroot/15750|$art74/15750/raw|" &
worker v34-ban-w3 "2500|$pilot/2500|$art99/2500/raw|" "11000|$resume/11000|$art99/11000/raw|" &
worker v34-ban-w4 "5000|$ckroot/5000|$art99/5000/raw|" "14750|$resume/14750|$art99/14750/raw|" &
wait
echo "=== all banana0630 workers done ==="
