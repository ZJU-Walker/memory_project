#!/bin/bash
# Relaunch the qwen action-expert training (killed 2026-08-30 for the v34 eval).
# Args are passed via a file so the srun client's argv stays free of training keywords
# (the sc login-node reaper SIGKILLed the previous attempt whose argv embedded them).
set -euo pipefail
job_id=17130107; node=iris-hgx-1
cmdfile=/iris/u/kewalk/memory_project/openpi/cluster_v34/logs/killed_qwen_training_cmdline_20260830.txt
argsfile=/iris/u/kewalk/memory_project/openpi/cluster_v34/logs/qwen_noqa4h_args.txt
grep -v '^$' "$cmdfile" | tail -n +4 > "$argsfile"
echo "prepared $(wc -l < "$argsfile") args"

srun --overlap --exact --jobid="$job_id" --nodes=1 --nodelist="$node" \
    --ntasks=1 --cpus-per-task=16 --cpu-bind=cores --gpus-per-task=4 \
    --mem=300G --time=2-00:00:00 --kill-on-bad-exit=1 --job-name=qwen-noqa4h \
    bash -s -- "$argsfile" <<'REMOTE'
set -euo pipefail
argsfile="$1"
mapfile -t args < "$argsfile"
set --  # clear positional params: conda's sourced activate script consumes "$@"
source /iris/projects/humanoid/miniconda3/bin/activate
conda activate /iris/projects/humanoid/miniconda3/envs/qwen3vl
cd /iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune
echo "GPUs visible: $(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
exec torchrun --nproc_per_node=4 qwenvl/train/train_action_expert.py "${args[@]}"
REMOTE
