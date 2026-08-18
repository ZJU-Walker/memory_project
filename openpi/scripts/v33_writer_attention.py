"""Driver for the v3.3 writer-attention diagnostic (see openpi.diagnostics.v33_writer_attention).

Usage:
    uv run scripts/v33_writer_attention.py \
        --checkpoint checkpoints/pi05_yam_mem_v33/v33_run1/2750 \
        --dataset_root ~/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask \
        --output_dir diagnostic_outputs/v33_writer_attention/2750
"""

import tyro

from openpi.diagnostics.v33_writer_attention import Options
from openpi.diagnostics.v33_writer_attention import V33WriterAttentionRunner

if __name__ == "__main__":
    V33WriterAttentionRunner(tyro.cli(Options)).run()
