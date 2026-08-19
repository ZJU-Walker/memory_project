"""CLI: do the v3.3 write tokens encode the target side? (diagnostic-only linear probe)

Example::

    uv run python scripts/v33_write_token_probe.py \
      --checkpoint checkpoints/pi05_yam_mem_v33/v33_run1/6250 \
      --dataset_root ~/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask \
      --output_dir diagnostic_outputs/v33_write_probe/6250
"""

import argparse
from pathlib import Path

from openpi.diagnostics import v33_write_token_probe as probe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--config", default="pi05_yam_mem_v33")
    parser.add_argument("--episode_indices", type=int, nargs="*", default=[])
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=0,
        help="cap the episode count (0 = all); useful for a fast smoke run",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    options = probe.Options(
        checkpoint=args.checkpoint,
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        config=args.config,
        episode_indices=tuple(args.episode_indices),
        stride=args.stride,
        max_episodes=args.max_episodes,
        seed=args.seed,
    )
    probe.WriteTokenProbeRunner(options).run()


if __name__ == "__main__":
    main()
