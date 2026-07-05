"""Serve the v1 memory run (`yam_banana_memory_400m_v1`) -- 3-camera memory encoding.

The v1 run was launched before the memory path was restricted to the top camera, so its
mem_q/k/v/out projections were trained against the 3-camera pooled encoding. This wrapper serves
those checkpoints faithfully by forcing ``mem_camera=all`` (write keys AND the read query then pool
all cameras + prompt, exactly as during v1 training). For v2+ checkpoints (top-camera-only memory)
use ``serve_policy_memory_v2.py`` instead -- serving a checkpoint with the wrong encoding does not
crash, it just silently mismatches what the memory was trained on.

Usage (defaults to the final v1 checkpoint; pass --policy.dir for an intermediate step):
    uv run scripts/serve_policy_memory_v1.py
    uv run scripts/serve_policy_memory_v1.py --policy.dir=checkpoints/pi05_yam_memory/yam_banana_memory_400m_v1/5000
"""

import logging

import tyro

from serve_policy_memory import Args, Checkpoint, main

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(
        tyro.cli(
            Args,
            default=Args(
                policy=Checkpoint(
                    config="pi05_yam_memory",
                    dir="checkpoints/pi05_yam_memory/yam_banana_memory_400m_v1/29999",
                ),
                mem_camera="all",  # v1 trained with 3-camera pooled memory encoding
            ),
        )
    )
