"""Re-render per-slot (16-tile) attention grids from a finished writer-attention run.

CPU only -- reads the saved ``episode_*_maps.npz`` files and re-decodes the camera frames from
the dataset, so no checkpoint or GPU is needed. New runs of scripts/v33_writer_attention.py
render the TRUE-variant grid automatically; use this to backfill old runs or to render other
variants (cf / base / read).

Usage:
    uv run scripts/v33_render_slot_grids.py \
        --run_dir diagnostic_outputs/v33_writer_attention/2750 \
        --dataset_root ~/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask \
        --variant true
"""

import dataclasses
from pathlib import Path
import re

import tyro

from openpi.diagnostics import v33_writer_attention as wa


@dataclasses.dataclass(frozen=True)
class Options:
    run_dir: Path
    dataset_root: Path
    variant: str = "true"
    fps: float = 4.0


def main(options: Options) -> None:
    npz_paths = sorted(options.run_dir.glob("episode_*_maps.npz"))
    if not npz_paths:
        raise FileNotFoundError(f"no episode_*_maps.npz under {options.run_dir}")
    for npz_path in npz_paths:
        match = re.fullmatch(r"episode_(\d+)_maps\.npz", npz_path.name)
        episode = int(match.group(1))
        suffix = "_slots" if options.variant == "true" else f"_slots_{options.variant}"
        output = npz_path.with_name(f"episode_{episode:03d}{suffix}.mp4")
        encoder = wa.slot_grid_from_npz(
            npz_path, options.dataset_root, episode, output, variant=options.variant, fps=options.fps
        )
        print(f"{output.name}: {encoder}")


if __name__ == "__main__":
    main(tyro.cli(Options))
