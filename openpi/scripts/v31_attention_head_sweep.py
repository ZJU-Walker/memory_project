"""Per-head attention budget for one layer of the v3.1 memory model.

Block averages over heads answer "does the model look at the image" only when heads behave
alike. Transformers routinely dedicate most heads to attention sinks while one or two do the
task-relevant routing, and averaging hides exactly that. This sweep reports, for every head, how
its attention divides across the key blocks, so a diluted average can be told apart from a model
that genuinely ignores vision or memory.

It reuses the attention replay runner, so episode loading, transforms, and the write cadence are
identical to the heatmap diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import numpy as np

from openpi.diagnostics import attention_replay as _attention
from openpi.diagnostics import writer_contribution as _writer

SCHEMA_VERSION = "openpi.v31.attention_head_sweep.v1"


def _blocks(attention: dict[str, Any], prefix: str, live: np.ndarray | None) -> dict[str, float]:
    out = {}
    for key, value in attention.items():
        if not key.startswith(f"{prefix}_to_") or not key.endswith("_mass"):
            continue
        array = np.asarray(value[0], dtype=np.float64)
        selected = array if live is None else array[live]
        out[key[len(prefix) + 4 : -5]] = float(np.mean(selected)) if selected.size else 0.0
    return out


def run(options: _attention.AttentionRunOptions, *, max_frames: int) -> dict[str, Any]:
    runner = _attention.AttentionReplayRunner(options)
    episodes = []
    for source in runner.sources:
        frames = []
        state = runner.model.memory.init_state(1)
        for policy_step, sample in enumerate(_iter_frames(runner, source, max_frames)):
            observation, _ = runner._transform_observation(  # noqa: SLF001
                source, sample["raw_frame"], sample["top"], sample["left"], sample["right"], sample["state"]
            )
            per_head = {}
            with runner.model.capture_attention():
                heads = int(
                    jax.device_get(
                        runner._attention_step(  # noqa: SLF001
                            observation,
                            state,
                            layer=runner.layer,
                            forced_subtask_tokens=runner._tokens,  # noqa: SLF001
                            forced_subtask_mask=runner._mask,  # noqa: SLF001
                        )["num_heads"]
                    )
                )
                for head in range(heads):
                    attention = jax.device_get(
                        runner._attention_step(  # noqa: SLF001
                            observation,
                            state,
                            layer=runner.layer,
                            head=head,
                            forced_subtask_tokens=runner._tokens,  # noqa: SLF001
                            forced_subtask_mask=runner._mask,  # noqa: SLF001
                        )
                    )
                    live = np.asarray(attention["subtask_token_mask"][0], dtype=bool)
                    per_head[head] = {
                        "memory": _blocks(attention, "memory", None),
                        "subtask": _blocks(attention, "subtask", live),
                    }
            frames.append({"raw_frame": sample["raw_frame"], "policy_step": policy_step, "heads": per_head})
            state, _ = runner._write_step(observation, state, allow_write=True)  # noqa: SLF001
            state = jax.device_get(state)
            print(f"[{source.episode_id}] frame {sample['raw_frame']} heads={heads}", flush=True)
        episodes.append(
            {"episode_id": source.episode_id, "ground_truth_side": source.ground_truth_side, "frames": frames}
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_path": str(options.checkpoint),
        "config": options.config,
        "layer": runner.layer,
        "subtask": options.subtask,
        "effective_stride": runner.stride,
        "max_frames": max_frames,
        "interpretation_warnings": [
            "Per-head masses partition each head's own softmax and sum to 1 within a head.",
            "A head's share is not its importance: the output projection reweights heads, so a "
            "head with high visual mass may still contribute little to the residual stream.",
        ],
        "episodes": episodes,
    }
    options.output_dir.mkdir(parents=True, exist_ok=False)
    _writer._write_json(options.output_dir / "head_sweep.json", manifest)  # noqa: SLF001
    return manifest


def _iter_frames(runner, source, max_frames):
    """Yield decoded stride-aligned frames, reusing the runner's LeRobot parquet reader."""
    import pyarrow.parquet as pq

    columns = ["image", "left_wrist_image", "right_wrist_image", "state", "frame_index", "task_index"]
    parquet = pq.ParquetFile(source.path)
    emitted = 0
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        for row in batch.to_pylist():
            raw_frame = row["frame_index"]
            if raw_frame % runner.stride:
                continue
            yield {
                "raw_frame": raw_frame,
                "top": runner._decode_inline_image(row["image"], field="image", raw_frame=raw_frame),  # noqa: SLF001
                "left": runner._decode_inline_image(  # noqa: SLF001
                    row["left_wrist_image"], field="left_wrist_image", raw_frame=raw_frame
                ),
                "right": runner._decode_inline_image(  # noqa: SLF001
                    row["right_wrist_image"], field="right_wrist_image", raw_frame=raw_frame
                ),
                "state": np.asarray(row["state"], dtype=np.float32),
            }
            emitted += 1
            if emitted >= max_frames:
                return


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Per-head attention budget sweep")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="pi05_yam_mem_v31")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-indices", type=_writer._parse_episode_indices, required=True)  # noqa: SLF001
    parser.add_argument("--layer", type=int)
    parser.add_argument("--subtask", default=_attention.CANONICAL_SUBTASKS[0])
    parser.add_argument("--max-frames", type=int, default=12)
    namespace = parser.parse_args(argv)
    options = _attention.AttentionRunOptions(
        checkpoint=namespace.checkpoint,
        output_dir=namespace.output_dir,
        dataset_root=namespace.dataset_root,
        episode_indices=tuple(namespace.episode_indices),
        config=namespace.config,
        layer=namespace.layer,
        subtask=namespace.subtask,
        render_video=False,
    )
    manifest = run(options, max_frames=namespace.max_frames)
    print(json.dumps({"episodes": len(manifest["episodes"]), "layer": manifest["layer"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
