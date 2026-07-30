"""Render one contact sheet per episode so the bins-open ("reveal") frames can be labeled fast.

Each sheet tiles every --step-th top-camera frame of an episode with its frame number. Look for
the first tile where the bin contents are visible, note the frame number, and put it into
assets/pi05_yam/reveal_frames.json as {"<episode_index>": reveal_frame} or, to also label when
the bins are closed again (enables the exact visible-vs-hidden quiz accuracy split),
{"<episode_index>": [reveal_frame, close_frame]}. Episodes missing from the json fall back to
the conservative defaults (reveal 300, close 450).

Needs the dataset + torch, so run on iris-hgx-2:
    uv run python scripts/make_reveal_sheets.py
Sheets land in scripts/eval_results/reveal_sheets/ plus a reveal_frames.template.json prefilled
with the defaults (copy it to assets/pi05_yam/reveal_frames.json and edit).
"""

import dataclasses
import json
import pathlib

import cv2
import numpy as np
import tyro


@dataclasses.dataclass
class Args:
    repo_id: str = "yam/bin_memory_banana_subtask"
    step: int = 15  # frames between tiles (0.5 s at 30 fps)
    tile_width: int = 212
    columns: int = 8
    out_dir: pathlib.Path = pathlib.Path(__file__).parent / "eval_results" / "reveal_sheets"


def main(args: Args) -> None:
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

    dataset = lerobot_dataset.LeRobotDataset(args.repo_id)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    num_episodes = dataset.meta.total_episodes

    for episode in range(num_episodes):
        start = int(dataset.episode_data_index["from"][episode])
        end = int(dataset.episode_data_index["to"][episode])
        tiles = []
        for idx in range(start, end, args.step):
            frame = np.asarray(dataset[idx]["image"])
            if frame.ndim == 3 and frame.shape[0] == 3:  # CHW float [0,1] -> HWC uint8
                frame = np.transpose(frame, (1, 2, 0))
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            scale = args.tile_width / frame.shape[1]
            tile = cv2.resize(frame, (args.tile_width, int(frame.shape[0] * scale)))
            label = f"{idx - start}"
            cv2.putText(tile, label, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(tile, label, (4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 60), 1, cv2.LINE_AA)
            tiles.append(tile)

        rows = []
        for r in range(0, len(tiles), args.columns):
            row = tiles[r : r + args.columns]
            row += [np.zeros_like(tiles[0])] * (args.columns - len(row))
            rows.append(np.concatenate(row, axis=1))
        sheet = np.concatenate(rows, axis=0)
        out = args.out_dir / f"episode_{episode:03d}.png"
        cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        print(f"episode {episode:2d}: {end - start} frames -> {out}")

    template = {str(e): 300 for e in range(num_episodes)}
    template_path = args.out_dir / "reveal_frames.template.json"
    template_path.write_text(json.dumps(template, indent=2))
    print(f"\ntemplate written to {template_path}")
    print("label the reveal frames, then copy to assets/pi05_yam/reveal_frames.json")


if __name__ == "__main__":
    main(tyro.cli(Args))
