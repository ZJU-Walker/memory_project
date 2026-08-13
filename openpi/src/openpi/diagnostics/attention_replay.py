"""Offline replay of layer-8 attention maps over a v3.1 memory episode.

This answers a different question from the writer-contribution diagnostics.  Those measure how
much each patch token *changes the fast weights*; this measures where the transformer *looks*.
Two query groups are replayed at the normal write cadence:

* ``memory_to_top`` -- the 256 memory-token rows attending to the 256 top-camera patches.  These
  rows produce ``c_t``, so the map shows which image regions the write representation is built
  from.
* ``subtask_to_top`` -- the live (non-padding) rows of a teacher-forced canonical subtask
  attending to the same patches.  This is sequence-level exploratory routing, not a causal
  attribution of the left/right decision token.  The accompanying ``*_mass`` scalars record
  how much of each live row's total attention goes to memory versus current vision.

Attention rows are softmax distributions over *all* keys, so a group's map does not sum to one.
The recorded mass fractions are what make groups comparable, and they are written to CSV
alongside the heatmaps rather than being folded into the colours.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import dataclasses
import json
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.diagnostics import token_heatmap
from openpi.diagnostics import writer_contribution as _writer
from openpi.shared import nnx_utils

SCHEMA_VERSION = "openpi.v31.attention_replay.v1"
QUERY_GROUPS = ("memory_to_top", "subtask_to_top")
# The canonical strings the policy is trained to emit; scoring the decision's attention requires
# teacher-forcing a complete sequence rather than one tokenizer-dependent token.
CANONICAL_SUBTASKS = ("open left bin", "open right bin")


def _tokenize_canonical_subtask(tokenizer: Any, causal_len: int, text: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Tokenize one canonical answer without ever promoting tokenizer padding to live rows."""

    tokens, mask = tokenizer.tokenize(f"{text}\n")
    tokens = np.asarray(tokens, dtype=np.int32)
    mask = np.asarray(mask, dtype=bool)
    if tokens.shape != (causal_len,) or mask.shape != (causal_len,):
        raise ValueError(
            "canonical subtask tokenizer must return two causal-length vectors; "
            f"got {tokens.shape} and {mask.shape} for causal_len={causal_len}."
        )
    if not np.any(mask):
        raise ValueError("canonical subtask tokenizer returned no live tokens.")
    if np.any(mask[np.count_nonzero(mask) :]):
        raise ValueError("canonical subtask tokenizer mask must be left-aligned.")
    return jnp.asarray(tokens[None]), jnp.asarray(mask[None])


@dataclasses.dataclass(frozen=True)
class AttentionRunOptions(_writer.RunOptions):
    layer: int | None = None
    subtask: str = CANONICAL_SUBTASKS[0]
    # Reduce the 256 memory-token query rows to one map per frame. "mean" shows the average
    # region the write attends to; "max" surfaces a region any single row focuses on.
    row_reduction: str = "mean"
    # Attention dumps large sink mass onto letterbox padding, which is not part of the camera
    # image; fitting the colour scale on content only keeps the scene structure legible.
    exclude_letterbox_padding: bool = True
    # Render the absolute map plus a sink-normalized view. Per-frame z-scoring is omitted here:
    # its within-frame mean is itself dominated by the sinks, so it does not remove them.
    heatmap_scale_modes: tuple[str, ...] = ("video", "per_token_history")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.layer is not None and self.layer < 0:
            raise ValueError("layer must be non-negative")
        if self.row_reduction not in ("mean", "max"):
            raise ValueError("row_reduction must be 'mean' or 'max'")
        if not self.subtask.strip():
            raise ValueError("subtask must be a non-empty canonical string")


@dataclasses.dataclass(frozen=True)
class _AttentionFrame:
    raw_frame: int
    policy_step: int
    model_image_rgb: np.ndarray
    raw_image_rgb: np.ndarray
    maps: dict[str, np.ndarray]
    scalar: dict[str, float]
    phase: str = ""


class AttentionReplayRunner(_writer.WriterContributionRunner):
    """Replay a checkpoint over recorded episodes, recording layer attention per write."""

    def __init__(self, options: AttentionRunOptions):
        super().__init__(options)
        self.options: AttentionRunOptions = options
        depth = self.model.PaliGemma.llm.module.configs[0].depth
        self.layer = self.model.memory_layer if options.layer is None else options.layer
        if not 0 <= self.layer < depth:
            raise ValueError(f"layer {self.layer} is outside the model depth {depth}")
        self._tokens, self._mask = self._tokenize_subtask(options.subtask)
        # Attention capture is a linen attribute, so it must be enabled around every traced call;
        # jitting the bound method keeps the replay at inference speed.
        # ``layer``/``head`` index stacked arrays and bounds-check in Python, so both must be
        # static; a per-head sweep therefore compiles one executable per head.
        self._attention_step = nnx_utils.module_jit(self.model.memory_attention_maps, static_argnames=("layer", "head"))
        self._write_step = nnx_utils.module_jit(self.model.writer_contribution_step, static_argnames=("allow_write",))

    def _tokenize_subtask(self, text: str) -> tuple[jnp.ndarray, jnp.ndarray]:
        from openpi.models import tokenizer as _tokenizer

        causal_len = self.model.causal_token_len
        return _tokenize_canonical_subtask(_tokenizer.PaligemmaTokenizer(causal_len), causal_len, text)

    def _measure_attention(self, observation, memory_state) -> dict[str, np.ndarray]:
        with self.model.capture_attention():
            maps = self._attention_step(
                observation,
                memory_state,
                layer=self.layer,
                forced_subtask_tokens=self._tokens,
                forced_subtask_mask=self._mask,
            )
        return jax.device_get(maps)

    @staticmethod
    def _progress_detail(frame) -> str:
        return (
            f"mem->top={frame.scalar['memory_to_top_mass']:.4f} "
            f"subtask->mem={frame.scalar['subtask_to_memory_mass']:.4f}"
        )

    def _reduce_rows(self, rows: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """Collapse a [rows, 256] attention block into one 256-token map."""

        if rows.size == 0:
            return np.zeros(token_heatmap.TOKEN_COUNT, dtype=np.float32)
        if self.options.row_reduction == "max":
            return rows.max(axis=0).astype(np.float32)
        if weights is not None:
            total = float(np.sum(weights))
            if total > 0:
                return (rows * weights[:, None]).sum(axis=0).astype(np.float32) / total
        return rows.mean(axis=0).astype(np.float32)

    def _evaluate_frame(
        self,
        source: _writer.EpisodeSource,
        *,
        raw_frame: int,
        policy_step: int,
        top_rgb: np.ndarray,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        robot_state: np.ndarray,
        memory_state: object,
        phase: str = "",
    ) -> tuple[object, _AttentionFrame]:
        observation, model_image = self._transform_observation(
            source, raw_frame, top_rgb, left_rgb, right_rgb, robot_state
        )
        attention = self._measure_attention(observation, memory_state)

        memory_rows = np.asarray(attention["memory_to_top"][0], dtype=np.float32)
        live = np.asarray(attention["subtask_token_mask"][0], dtype=bool)
        subtask_rows = np.asarray(attention["subtask_to_top"][0], dtype=np.float32)[live]
        maps = {
            "memory_to_top": self._reduce_rows(memory_rows),
            "subtask_to_top": self._reduce_rows(subtask_rows),
        }
        for name, values in maps.items():
            if values.shape != (token_heatmap.TOKEN_COUNT,):
                raise ValueError(f"{name} must reduce to a 256-token grid, got {values.shape}")

        subtask_memory_mass = np.asarray(attention["subtask_to_memory_mass"][0], dtype=np.float64)[live]
        subtask_top_mass = np.asarray(attention["subtask_to_top_mass"][0], dtype=np.float64)[live]
        scalar = {
            "memory_to_top_mass": float(np.mean(np.asarray(attention["memory_to_top_mass"][0], dtype=np.float64))),
            "memory_to_memory_mass": float(
                np.mean(np.asarray(attention["memory_to_memory_mass"][0], dtype=np.float64))
            ),
            "subtask_to_top_mass": float(np.mean(subtask_top_mass)) if subtask_top_mass.size else 0.0,
            "subtask_to_memory_mass": float(np.mean(subtask_memory_mass)) if subtask_memory_mass.size else 0.0,
            "subtask_live_tokens": float(np.count_nonzero(live)),
        }
        # Every remaining key block, so the recorded budget adds to 1 and no attention is
        # silently unaccounted for. Memory rows cannot see the causal block at all.
        for key, values in attention.items():
            if not key.endswith("_mass") or key in scalar:
                continue
            array = np.asarray(values[0], dtype=np.float64)
            selected = array[live] if key.startswith("subtask_") else array
            scalar[key] = float(np.mean(selected)) if selected.size else 0.0
        # Advance the episode exactly as normal inference would, so frame t+1 sees the state a
        # real rollout would have. Attention measurement itself never writes.
        memory_state, _ = self._write_step(observation, memory_state, allow_write=True)
        memory_state = jax.device_get(memory_state)
        return memory_state, _AttentionFrame(
            raw_frame=raw_frame,
            policy_step=policy_step,
            model_image_rgb=model_image,
            raw_image_rgb=np.array(top_rgb, copy=True),
            maps=maps,
            scalar=scalar,
            phase=phase,
        )

    def _save_attention_episode(
        self,
        source: _writer.EpisodeSource,
        measured: Sequence[_AttentionFrame],
        total_raw_frames: int,
        episode_dir: Path,
    ) -> dict[str, Any]:
        episode_dir.mkdir(parents=True, exist_ok=False)
        stacked = {name: np.stack([frame.maps[name] for frame in measured]) for name in QUERY_GROUPS}
        scalar_names = tuple(measured[0].scalar)
        np.savez_compressed(
            episode_dir / "attention.npz",
            raw_frame=np.asarray([frame.raw_frame for frame in measured], dtype=np.int64),
            policy_step=np.asarray([frame.policy_step for frame in measured], dtype=np.int64),
            task=np.asarray([frame.phase for frame in measured], dtype=np.str_),
            **stacked,
            **{name: np.asarray([frame.scalar[name] for frame in measured], dtype=np.float32) for name in scalar_names},
        )

        with (episode_dir / "frame_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            fieldnames = [
                "raw_frame",
                "policy_step",
                "task",
                *scalar_names,
                *(f"{name}_{stat}" for name in QUERY_GROUPS for stat in ("max", "top_10pct_mass_fraction")),
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for frame in measured:
                row: dict[str, Any] = {
                    "raw_frame": frame.raw_frame,
                    "policy_step": frame.policy_step,
                    "task": frame.phase,
                    **frame.scalar,
                }
                for name, values in frame.maps.items():
                    stats = token_heatmap.metric_statistics(values)
                    row[f"{name}_max"] = stats["max"]
                    row[f"{name}_top_10pct_mass_fraction"] = stats["top_10pct_mass_fraction"]
                writer.writerow(row)

        videos = {}
        if self.options.render_video:
            fps = self.options.video_fps or source.control_hz / self.stride
            for name in QUERY_GROUPS:
                frames = [
                    token_heatmap.TokenMetricFrame(
                        raw_frame=frame.raw_frame,
                        policy_step=frame.policy_step,
                        model_image_rgb=frame.model_image_rgb,
                        raw_image_rgb=frame.raw_image_rgb,
                        token_values=frame.maps[name],
                        write_count=frame.policy_step + 1,
                        phase=frame.phase,
                        timestamp_s=frame.raw_frame / source.control_hz,
                        metadata={
                            "episode_id": source.episode_id,
                            "layer": self.layer,
                            "query_group": name,
                            "row_reduction": self.options.row_reduction,
                        },
                    )
                    for frame in measured
                ]
                for scale_mode in self.options.heatmap_scale_modes:
                    if scale_mode == "video":
                        video_key = name
                        mode_colormap = self.options.colormap
                    elif scale_mode == "per_token_history":
                        # Attention sinks are positionally fixed and nearly constant, so
                        # subtracting each token's own episode baseline cancels them and leaves
                        # only what actually changes with the scene.
                        video_key = f"{name}_sink_normalized"
                        mode_colormap = "coolwarm"
                    else:
                        video_key = f"{name}_perframe_zscore"
                        mode_colormap = "coolwarm"
                    videos[video_key] = token_heatmap.export_heatmap_video(
                        frames,
                        episode_dir / "heatmaps" / video_key,
                        metric_name=f"{name}_L{self.layer}",
                        fps=fps,
                        normalization=token_heatmap.NormalizationSpec(
                            lower_percentile=0.0,
                            upper_percentile=self.options.upper_percentile,
                            anchor_zero=True,
                            exclude_letterbox_padding=self.options.exclude_letterbox_padding,
                        ),
                        scale_mode=scale_mode,
                        zscore_range=self.options.zscore_range,
                        alpha=self.options.alpha,
                        colormap=mode_colormap,
                        video_encoder=_writer.WRITER_VIDEO_ENCODER,
                    )

        summary = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": source.episode_id,
            "path": str(source.path),
            "ground_truth_side": source.ground_truth_side,
            "control_hz": source.control_hz,
            "layer": self.layer,
            "subtask": self.options.subtask,
            "row_reduction": self.options.row_reduction,
            "excluded_letterbox_padding_from_scale": self.options.exclude_letterbox_padding,
            "letterbox_padding_mass_fraction": {
                name: float(
                    np.mean(
                        [
                            token_heatmap.raw_projection_statistics(
                                frame.maps[name], token_heatmap.letterbox_geometry(*frame.raw_image_rgb.shape[:2])
                            )["letterbox_padding_mass_fraction"]
                            for frame in measured
                        ]
                    )
                )
                for name in QUERY_GROUPS
            },
            "total_raw_frames": total_raw_frames,
            "sampled_write_frames": len(measured),
            "stride": self.stride,
            "first_raw_frame": measured[0].raw_frame,
            "last_raw_frame": measured[-1].raw_frame,
            "attention_mass": {
                name: {
                    "mean": float(np.mean([frame.scalar[name] for frame in measured])),
                    "min": float(np.min([frame.scalar[name] for frame in measured])),
                    "max": float(np.max([frame.scalar[name] for frame in measured])),
                }
                for name in scalar_names
            },
            "videos": {name: str(Path("heatmaps") / name / details["video"]) for name, details in videos.items()},
        }
        _writer._write_json(episode_dir / "summary.json", summary)  # noqa: SLF001
        return summary

    def run(self) -> dict[str, Any]:
        start = time.monotonic()
        self.options.output_dir.mkdir(parents=True, exist_ok=False)
        episodes_dir = self.options.output_dir / "episodes"
        episodes_dir.mkdir()
        summaries = []
        for source in self.sources:
            measured, total = self._measure_episode(source)
            summaries.append(
                self._save_attention_episode(
                    source,
                    measured,
                    total,
                    episodes_dir / _writer._safe_name(source.episode_id),  # noqa: SLF001
                )
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_path": str(self.options.checkpoint),
            "config": self.options.config,
            "layer": self.layer,
            "subtask": self.options.subtask,
            "row_reduction": self.options.row_reduction,
            "configured_stride": self.configured_stride,
            "effective_stride": self.stride,
            "episode_indices": list(self.options.episode_indices),
            "memory_write_source": self.train_config.model.memory_write_source,
            "interpretation_warnings": [
                "Attention rows are softmax distributions over ALL keys; a per-group map does not "
                "sum to one. Use the recorded *_mass columns to compare groups.",
                "Attention shows where the model looks, not what it stores: a patch can be "
                "attended to without changing the fast weights, and vice versa.",
                "subtask_* aggregates live rows of a teacher-forced complete answer. It is not "
                "a causal attribution of the first left/right decision token.",
                "Head-averaged within the layer; an individual head may be sharper than this mean.",
            ],
            "episodes": summaries,
            "elapsed_s": time.monotonic() - start,
        }
        _writer._write_json(self.options.output_dir / "run_manifest.json", manifest)  # noqa: SLF001
        return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay layer attention maps over v3.1 memory episodes")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="pi05_yam_mem_v31")
    parser.add_argument("--output-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", dest="episode_paths", type=Path, action="append")
    source.add_argument("--episode-manifest", type=Path)
    source.add_argument("--dataset-root", type=Path)
    parser.add_argument("--episode-indices", type=_writer._parse_episode_indices)  # noqa: SLF001
    parser.add_argument("--stride", type=int, help="raw-frame cadence; default is the config value")
    parser.add_argument("--layer", type=int, help="transformer layer to read; default is memory_layer (8)")
    parser.add_argument("--subtask", default=CANONICAL_SUBTASKS[0], choices=CANONICAL_SUBTASKS)
    parser.add_argument("--row-reduction", default="mean", choices=("mean", "max"))
    parser.add_argument(
        "--include-letterbox-padding",
        action="store_true",
        help="fit the colour scale over padding tokens too; by default attention sinks on the "
        "black letterbox bars are excluded so real camera content sets the scale",
    )
    parser.add_argument("--video-fps", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--colormap", choices=token_heatmap.SUPPORTED_COLORMAPS, default="inferno")
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _build_parser().parse_args(argv)
    options = AttentionRunOptions(
        checkpoint=namespace.checkpoint,
        output_dir=namespace.output_dir,
        episode_paths=tuple(namespace.episode_paths or ()),
        episode_manifest=namespace.episode_manifest,
        dataset_root=namespace.dataset_root,
        episode_indices=tuple(namespace.episode_indices or ()),
        config=namespace.config,
        stride=namespace.stride,
        video_fps=namespace.video_fps,
        max_frames=namespace.max_frames,
        render_video=not namespace.skip_video,
        colormap=namespace.colormap,
        alpha=namespace.alpha,
        upper_percentile=namespace.upper_percentile,
        layer=namespace.layer,
        subtask=namespace.subtask,
        row_reduction=namespace.row_reduction,
        exclude_letterbox_padding=not namespace.include_letterbox_padding,
    )
    manifest = AttentionReplayRunner(options).run()
    print(json.dumps(_writer._jsonable(manifest), indent=2, sort_keys=True), flush=True)  # noqa: SLF001
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
