#!/bin/bash
# Banana-0630 probe on the RELABELED "inspect both bins" windows (job 17130107, 4 GPUs).
# usage: run_eval0830.sh stageA|stageB
#   stageA: 9-checkpoint inspect-window eval (fresh+online heads), 4 parallel workers
#   stageB: full-episode per-frame scores at ckpt 15750 for the score videos, 4 workers
set -euo pipefail
mode="${1:?usage: $0 stageA|stageB}"
job_id=17130107
node=iris-hgx-1
repo=/iris/u/kewalk/memory_project/openpi
ckroot=$repo/checkpoints/pi05_yam_mem_v34_run5_eta0/v34_run5_eta0
pilot=$repo/diagnostic_checkpoints/v34_run5_eta0_pilot_copies
resume=$repo/diagnostic_checkpoints/v34_run5_eta0_resume_copies
art99=$repo/diagnostic_outputs/v34_fixed_writer_probe/full_17099350
art74=$repo/diagnostic_outputs/v34_fixed_writer_probe/full_17074121
snapshot=$repo/cluster_v34/provenance/v34_run5_eta0_main_launch/source_snapshot.tar.gz

worker() {
    local name="$1" gpu="$2"; shift 2
    srun --overlap --exact --jobid="$job_id" \
        --nodes=1 --nodelist="$node" \
        --ntasks=1 --cpus-per-task=12 --cpu-bind=cores \
        --gpus-per-task=4 --mem=150G --time=05:00:00 \
        --kill-on-bad-exit=1 --job-name="$name" \
        bash -s -- "$repo" "$snapshot" "$gpu" "$@"
}

read -r -d '' REMOTE <<'RS' || true
set -euo pipefail
repo="$1"; snapshot="$2"; pin_gpu="$3"; shift 3
export CUDA_VISIBLE_DEVICES="$pin_gpu"
echo "[worker] pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES tasks=$#"
source_root=$(mktemp -d "/tmp/v34-e830-${SLURM_JOB_ID}.${SLURM_STEP_ID}.XXXXXX")
trap 'find "$source_root" -depth -delete' EXIT
tar -xzf "$snapshot" -C "$source_root"
export HF_HOME=/iris/u/kewalk/.cache/huggingface OPENPI_DATA_HOME=/iris/u/kewalk/.cache/openpi
export OPENPI_JAX_CACHE_DIR=/iris/u/kewalk/.cache/jax HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false V34_RUN5_SOURCE_ROOT="$source_root" PYTHONPATH="$source_root/src"
cd "$repo"
for task in "$@"; do
    IFS='|' read -r outdir ckpt artifact extra <<< "$task"
    echo "=== [worker] $outdir ==="
    .venv/bin/python -u scripts/v34_eval0830_writer_probe.py \
        --checkpoint "$ckpt" \
        --probe-artifact-dir "$artifact" \
        --output-dir "$repo/diagnostic_outputs/$outdir" \
        --parameter-source raw --batch-size 8 $extra </dev/null
done
RS

if [[ "$mode" == stageA ]]; then
    O=v34_eval0830_writer_probe
    worker v34-e830-w1 0 "$O/500|$pilot/500|$art99/500/raw|" "$O/9500|$resume/9500|$art99/9500/raw|" <<< "$REMOTE" &
    worker v34-e830-w2 1 "$O/1000|$pilot/1000|$art99/1000/raw|" "$O/10000|$ckroot/10000|$art99/10000/raw|" "$O/15750|$ckroot/15750|$art74/15750/raw|" <<< "$REMOTE" &
    worker v34-e830-w3 2 "$O/2500|$pilot/2500|$art99/2500/raw|" "$O/11000|$resume/11000|$art99/11000/raw|" <<< "$REMOTE" &
    worker v34-e830-w4 3 "$O/5000|$ckroot/5000|$art99/5000/raw|" "$O/14750|$resume/14750|$art99/14750/raw|" <<< "$REMOTE" &
    wait
    echo "=== stageA done ==="
else
    B="--full-frame-scores"
    O=v34_eval0830_writer_probe_full15750
    worker v34-e830f-w1 0 "$O|$ckroot/15750|$art74/15750/raw|$B --episodes 0,1,2" <<< "$REMOTE" &
    worker v34-e830f-w2 1 "$O|$ckroot/15750|$art74/15750/raw|$B --episodes 3,4,5" <<< "$REMOTE" &
    worker v34-e830f-w3 2 "$O|$ckroot/15750|$art74/15750/raw|$B --episodes 6,7" <<< "$REMOTE" &
    worker v34-e830f-w4 3 "$O|$ckroot/15750|$art74/15750/raw|$B --episodes 8,9" <<< "$REMOTE" &
    wait
    echo "=== stageB done ==="
fi
