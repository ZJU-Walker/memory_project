"""Re-render per-slot (16-tile) attention grids from a finished writer-attention run.

CPU only -- reads the saved ``episode_*_maps.npz`` files and re-decodes the camera frames from
the dataset, so no checkpoint or GPU is needed. New runs of scripts/v33_writer_attention.py
render the TRUE-variant grid in both styles automatically; use this to backfill old runs or to
render other variants (cf / base / read).

Two styles (see openpi.diagnostics.v33_writer_attention.SLOT_GRID_STYLES):
  --style v32    per-frame per-slot max normalization, JET, intensity-scaled alpha -- the v3.2
                 convention; every tile's peak saturates red, so a slot's attention SHAPE is
                 obvious at a glance. Purely relative: says nothing about magnitude over time.
  --style video  one fixed scale per slot across the episode, inferno, flat alpha -- frames are
                 comparable, but because attention is very peaked most patches render near
                 black and a sharp map can look like an untouched image.

Usage:
    uv run scripts/v33_render_slot_grids.py \
        --run_dirs diagnostic_outputs/v33_writer_attention/4250 \
        --dataset_root ~/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask \
        --style v32
"""

import dataclasses
from pathlib import Path
import re

import tyro

from openpi.diagnostics import v33_writer_attention as wa


@dataclasses.dataclass(frozen=True)
class Options:
    run_dirs: tuple[Path, ...]
    dataset_root: Path
    variant: str = "true"
    fps: float = 4.0
    # "v32": per-frame per-slot max + JET + intensity alpha (readable at a glance, purely
    # relative); "video": fixed per-slot scale across the episode (comparable across frames).
    style: str = "v32"
    overwrite: bool = False


def main(options: Options) -> None:
    if options.style not in wa.SLOT_GRID_STYLES:
        raise ValueError(f"style must be one of {wa.SLOT_GRID_STYLES}; got {options.style!r}")
    for run_dir in options.run_dirs:
        npz_paths = sorted(run_dir.glob("episode_*_maps.npz"))
        if not npz_paths:
            raise FileNotFoundError(f"no episode_*_maps.npz under {run_dir}")
        for npz_path in npz_paths:
            match = re.fullmatch(r"episode_(\d+)_maps\.npz", npz_path.name)
            episode = int(match.group(1))
            parts = ["_slots"]
            if options.variant != "true":
                parts.append(f"_{options.variant}")
            if options.style != "video":
                parts.append(f"_{options.style}")
            output = npz_path.with_name(f"episode_{episode:03d}{''.join(parts)}.mp4")
            if output.exists():
                if not options.overwrite:
                    print(f"{output.name}: exists, skipping (pass --overwrite to replace)")
                    continue
                output.unlink()
            encoder = wa.slot_grid_from_npz(
                npz_path,
                options.dataset_root,
                episode,
                output,
                variant=options.variant,
                fps=options.fps,
                style=options.style,
            )
            print(f"{run_dir.name}/{output.name}: {encoder}")


if __name__ == "__main__":
    main(tyro.cli(Options))
