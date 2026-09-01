#!/bin/bash
# Extend the 0830 eval to the new latest checkpoint 18000: fixed probe -> eval -> stage B videos.
set -euo pipefail
job_id=17130107; node=iris-hgx-1
repo=/iris/u/kewalk/memory_project/openpi
dataset=/iris/u/kewalk/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask
ckpt=$repo/checkpoints/pi05_yam_mem_v34_run5_eta0/v34_run5_eta0/18000
art=$repo/diagnostic_outputs/v34_fixed_writer_probe/full_${job_id}/18000
snapshot=$repo/cluster_v34/provenance/v34_run5_eta0_main_launch/source_snapshot.tar.gz
[[ -f "$ckpt/_CHECKPOINT_METADATA" ]]

worker() {
    local name="$1" gpu="$2"; shift 2
    srun --overlap --exact --jobid="$job_id" --nodes=1 --nodelist="$node" \
        --ntasks=1 --cpus-per-task=12 --cpu-bind=cores --gpus-per-task=4 \
        --mem=150G --time=04:00:00 --kill-on-bad-exit=1 --job-name="$name" \
        bash -s -- "$repo" "$dataset" "$snapshot" "$ckpt" "$art" "$gpu" "$@"
}
read -r -d '' REMOTE <<'RS' || true
set -euo pipefail
repo="$1"; dataset="$2"; snapshot="$3"; ckpt="$4"; art="$5"; pin="$6"; stage="$7"; extra="${8:-}"
export CUDA_VISIBLE_DEVICES="$pin"
echo "[${stage}] pinned GPU $pin"
source_root=$(mktemp -d "/tmp/v34-e18-${SLURM_JOB_ID}.${SLURM_STEP_ID}.XXXXXX")
trap 'find "$source_root" -depth -delete' EXIT
tar -xzf "$snapshot" -C "$source_root"
export HF_HOME=/iris/u/kewalk/.cache/huggingface OPENPI_DATA_HOME=/iris/u/kewalk/.cache/openpi
export OPENPI_JAX_CACHE_DIR=/iris/u/kewalk/.cache/jax HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false V34_RUN5_SOURCE_ROOT="$source_root" PYTHONPATH="$source_root/src"
cd "$repo"
case "$stage" in
  fixedprobe)
    .venv/bin/python -u scripts/v34_fixed_writer_probe_eval.py \
        --checkpoint "$ckpt" --dataset-root "$dataset" --output-dir "$art" \
        --config pi05_yam_mem_v34_run5_eta0 --parameter-source raw \
        --batch-size 8 --seed 3401 --null-repeats 8 </dev/null
    (cd "$art/raw" && sha256sum -c COMPLETE >/dev/null && echo "fixed-probe@18000 verified") ;;
  eval)
    .venv/bin/python -u scripts/v34_eval0830_writer_probe.py \
        --checkpoint "$ckpt" --probe-artifact-dir "$art/raw" \
        --output-dir "$repo/diagnostic_outputs/v34_eval0830_writer_probe/18000" \
        --parameter-source raw --batch-size 8 </dev/null ;;
  stageB)
    .venv/bin/python -u scripts/v34_eval0830_writer_probe.py \
        --checkpoint "$ckpt" --probe-artifact-dir "$art/raw" \
        --output-dir "$repo/diagnostic_outputs/v34_eval0830_writer_probe_full18000" \
        --parameter-source raw --batch-size 8 --full-frame-scores $extra </dev/null ;;
esac
RS
echo "=== phase 1: fixed probe @18000 ==="
worker v34-fixw18000 0 fixedprobe <<< "$REMOTE"
echo "=== phase 2: eval0830 @18000 ==="
worker v34-e830-18000 0 eval <<< "$REMOTE"
echo "=== phase 3: stage B full-frame @18000 ==="
worker v34-e830f18-w1 0 stageB "--episodes 0,1,2" <<< "$REMOTE" &
worker v34-e830f18-w2 1 stageB "--episodes 3,4,5" <<< "$REMOTE" &
worker v34-e830f18-w3 2 stageB "--episodes 6,7" <<< "$REMOTE" &
worker v34-e830f18-w4 3 stageB "--episodes 8,9" <<< "$REMOTE" &
wait
echo "=== all 18000 phases done ==="
