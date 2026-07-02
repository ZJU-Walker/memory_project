"""Serve the v2 memory run (`yam_banana_memory_400m_v2`) -- top-camera-only memory encoding.

v2 checkpoints were trained with the memory path restricted to the top camera + instruction
(``mem_camera="base_0_rgb"``, the training config's default), so no override is needed -- this
wrapper just bakes in the v2 checkpoint directory. For the older v1 run (3-camera memory encoding)
use ``serve_policy_memory_v1.py``, which forces ``mem_camera=all`` to match how v1 was trained.

Usage (defaults to the final v2 checkpoint; pass --policy.dir for an intermediate step):
    uv run scripts/serve_policy_memory_v2.py
    uv run scripts/serve_policy_memory_v2.py --policy.dir=checkpoints/pi05_yam_memory/yam_banana_memory_400m_v2/5000
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
                    dir="checkpoints/pi05_yam_memory/yam_banana_memory_400m_v2/29999",
                ),
                # mem_camera left empty: the training config's top-camera-only encoding applies.
            ),
        )
    )
