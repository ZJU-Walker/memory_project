"""v3.1 memory-to-memory attention and retrieved-token causal ablation replay."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import csv
import dataclasses
import json
import math
from pathlib import Path
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.diagnostics import token_heatmap
from openpi.diagnostics import writer_contribution as _writer
from openpi.shared import nnx_utils

SCHEMA_VERSION = "openpi.v31.memory_to_memory_replay.v1"
ATTENTION_MAPS = (
    "query_memory_mass",
    "key_incoming_absolute",
    "key_incoming_conditional",
    "same_slot_diagonal",
)
CAUSAL_MAPS = ("final_ct_ablation_relative", "injection_ablation_relative")


def _parse_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 0 for item in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected distinct comma-separated non-negative integers")
    return values


@dataclasses.dataclass(frozen=True)
class RunOptions(_writer.RunOptions):
    layer: int = 8
    ablation_raw_frames: tuple[int, ...] = (350, 360, 370, 380)
    ablation_chunk_size: int = 8
    exclude_letterbox_padding: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.layer < 0:
            raise ValueError("layer must be non-negative")
        if not self.ablation_raw_frames or any(frame < 0 for frame in self.ablation_raw_frames):
            raise ValueError("ablation_raw_frames must be non-empty and non-negative")
        if len(set(self.ablation_raw_frames)) != len(self.ablation_raw_frames):
            raise ValueError("ablation_raw_frames must be distinct")
        if self.ablation_chunk_size < 1 or token_heatmap.TOKEN_COUNT % self.ablation_chunk_size:
            raise ValueError("ablation_chunk_size must be a positive divisor of 256")


@dataclasses.dataclass(frozen=True)
class _Frame:
    raw_frame: int
    policy_step: int
    model_image_rgb: np.ndarray
    raw_image_rgb: np.ndarray
    attention_maps: dict[str, np.ndarray]
    scalar: dict[str, float]
    phase: str = ""
    attention_matrix: np.ndarray | None = None
    causal_maps: dict[str, np.ndarray] | None = None
    retrieved_token_norm: np.ndarray | None = None


def matrix_statistics(matrix: np.ndarray) -> dict[str, float]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (token_heatmap.TOKEN_COUNT, token_heatmap.TOKEN_COUNT):
        raise ValueError(f"memory attention matrix must be [256,256], got {matrix.shape}")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("memory attention matrix must be finite and non-negative")
    total = float(np.sum(matrix))
    diagonal = float(np.trace(matrix))
    incoming = np.sum(matrix, axis=0)
    incoming_distribution = incoming / max(float(np.sum(incoming)), 1e-12)
    positive = incoming_distribution[incoming_distribution > 0]
    entropy = -float(np.sum(positive * np.log(positive))) / math.log(token_heatmap.TOKEN_COUNT)
    conditional_rows = matrix / np.maximum(np.sum(matrix, axis=1, keepdims=True), 1e-12)
    row_entropy = -np.sum(
        np.where(conditional_rows > 0, conditional_rows * np.log(np.maximum(conditional_rows, 1e-30)), 0.0),
        axis=1,
    ) / math.log(token_heatmap.TOKEN_COUNT)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    singular_distribution = singular_values / max(float(np.sum(singular_values)), 1e-12)
    singular_positive = singular_distribution[singular_distribution > 0]
    effective_rank = float(np.exp(-np.sum(singular_positive * np.log(singular_positive))))
    return {
        "mean_memory_mass_per_query": total / token_heatmap.TOKEN_COUNT,
        "diagonal_share_of_memory_mass": diagonal / max(total, 1e-12),
        "uniform_diagonal_share": 1.0 / token_heatmap.TOKEN_COUNT,
        "incoming_normalized_entropy": entropy,
        "incoming_effective_tokens": float(np.exp(entropy * math.log(token_heatmap.TOKEN_COUNT))),
        "mean_conditional_row_entropy": float(np.mean(row_entropy)),
        "effective_rank": effective_rank,
    }


class MemoryToMemoryReplayRunner(_writer.WriterContributionRunner):
    def __init__(self, options: RunOptions):
        super().__init__(options)
        self.options: RunOptions = options
        depth = self.model.PaliGemma.llm.module.configs[0].depth
        if not 0 <= options.layer < depth:
            raise ValueError(f"layer {options.layer} is outside depth {depth}")
        self._attention_step = nnx_utils.module_jit(self.model.memory_attention_maps, static_argnames=("layer", "head"))
        self._ablation_step = nnx_utils.module_jit(self.model.retrieved_token_ablation_step)
        self._write_step = nnx_utils.module_jit(self.model.writer_contribution_step, static_argnames=("allow_write",))

    @staticmethod
    def _progress_detail(frame) -> str:
        detail = (
            f"mem-mass={frame.scalar['mean_memory_mass_per_query']:.4f} "
            f"diag-share={frame.scalar['diagonal_share_of_memory_mass']:.4f}"
        )
        return detail + (" causal-ablation" if frame.causal_maps is not None else "")

    def _measure_attention(self, observation, memory_state) -> np.ndarray:
        with self.model.capture_attention():
            output = self._attention_step(observation, memory_state, layer=self.options.layer, head=None)
        matrix = np.asarray(jax.device_get(output["memory_to_memory"])[0], dtype=np.float32)
        if matrix.shape != (token_heatmap.TOKEN_COUNT, token_heatmap.TOKEN_COUNT):
            raise ValueError(f"expected memory matrix [256,256], got {matrix.shape}")
        return matrix

    def _measure_causal_ablation(self, observation, memory_state) -> tuple[dict[str, np.ndarray], np.ndarray]:
        values = {name: np.zeros(token_heatmap.TOKEN_COUNT, dtype=np.float32) for name in CAUSAL_MAPS}
        retrieved_token_norm = None
        for start in range(0, token_heatmap.TOKEN_COUNT, self.options.ablation_chunk_size):
            indices = np.concatenate(
                [
                    np.asarray([-1], dtype=np.int32),
                    np.arange(start, start + self.options.ablation_chunk_size, dtype=np.int32),
                ]
            )
            output = jax.device_get(self._ablation_step(observation, memory_state, jnp.asarray(indices)))
            if int(np.asarray(output["token_indices"])[0]) != -1:
                raise RuntimeError("causal ablation batch lost its unmodified control")
            for floor_name in ("final_ct_effect_l2", "injection_effect_l2"):
                floor = float(np.asarray(output[floor_name])[0])
                if floor != 0.0:
                    raise RuntimeError(f"same-batch no-op floor for {floor_name} is not exact zero: {floor}")
            values["final_ct_ablation_relative"][start : start + self.options.ablation_chunk_size] = np.asarray(
                output["final_ct_effect_relative"], dtype=np.float32
            )[1:]
            values["injection_ablation_relative"][start : start + self.options.ablation_chunk_size] = np.asarray(
                output["injection_effect_relative"], dtype=np.float32
            )[1:]
            if retrieved_token_norm is None:
                retrieved_token_norm = np.asarray(output["retrieved_token_norm"], dtype=np.float32)
        assert retrieved_token_norm is not None
        if not all(
            np.all(np.isfinite(array)) and np.all(array >= 0) for array in (*values.values(), retrieved_token_norm)
        ):
            raise FloatingPointError("causal ablation produced invalid values")
        return values, retrieved_token_norm

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
    ) -> tuple[object, _Frame]:
        observation, model_image = self._transform_observation(
            source, raw_frame, top_rgb, left_rgb, right_rgb, robot_state
        )
        matrix = self._measure_attention(observation, memory_state)
        query_mass = np.sum(matrix, axis=1)
        incoming = np.mean(matrix, axis=0)
        incoming_conditional = incoming / max(float(np.sum(incoming)), 1e-12)
        maps = {
            "query_memory_mass": query_mass.astype(np.float32),
            "key_incoming_absolute": incoming.astype(np.float32),
            "key_incoming_conditional": incoming_conditional.astype(np.float32),
            "same_slot_diagonal": np.diag(matrix).astype(np.float32),
        }
        selected = raw_frame in self.options.ablation_raw_frames
        causal_maps, retrieved_norm = (
            self._measure_causal_ablation(observation, memory_state) if selected else (None, None)
        )
        scalar = matrix_statistics(matrix)
        memory_state, _ = self._write_step(observation, memory_state, allow_write=True)
        memory_state = jax.device_get(memory_state)
        return memory_state, _Frame(
            raw_frame=raw_frame,
            policy_step=policy_step,
            model_image_rgb=model_image,
            raw_image_rgb=np.array(top_rgb, copy=True),
            attention_maps=maps,
            scalar=scalar,
            phase=phase,
            attention_matrix=matrix if selected else None,
            causal_maps=causal_maps,
            retrieved_token_norm=retrieved_norm,
        )

    def _token_frames(
        self,
        source: _writer.EpisodeSource,
        measured: Sequence[_Frame],
        name: str,
        *,
        causal: bool,
    ):
        selected = [frame for frame in measured if frame.causal_maps is not None] if causal else list(measured)
        return [
            token_heatmap.TokenMetricFrame(
                raw_frame=frame.raw_frame,
                policy_step=frame.policy_step,
                model_image_rgb=frame.model_image_rgb,
                raw_image_rgb=frame.raw_image_rgb,
                token_values=(frame.causal_maps if causal else frame.attention_maps)[name],  # type: ignore[index]
                write_count=frame.policy_step,
                phase=frame.phase,
                timestamp_s=frame.raw_frame / source.control_hz,
                metadata={
                    "episode_id": source.episode_id,
                    "layer": self.options.layer,
                    "metric": name,
                    "slot_alignment_not_pixel_causality": True,
                },
            )
            for frame in selected
        ]

    def _save_episode(
        self,
        source: _writer.EpisodeSource,
        measured: Sequence[_Frame],
        total_raw_frames: int,
        episode_dir: Path,
    ) -> dict[str, Any]:
        episode_dir.mkdir(parents=True, exist_ok=False)
        selected = [frame for frame in measured if frame.attention_matrix is not None]
        if {frame.raw_frame for frame in selected} != set(self.options.ablation_raw_frames):
            raise ValueError(
                f"episode {source.episode_id} did not contain every requested ablation frame; "
                f"got {[frame.raw_frame for frame in selected]}"
            )
        np.savez_compressed(
            episode_dir / "attention_maps.npz",
            raw_frame=np.asarray([frame.raw_frame for frame in measured], dtype=np.int64),
            **{name: np.stack([frame.attention_maps[name] for frame in measured]) for name in ATTENTION_MAPS},
            **{name: np.asarray([frame.scalar[name] for frame in measured]) for name in measured[0].scalar},
        )
        np.savez_compressed(
            episode_dir / "selected_full_matrices.npz",
            raw_frame=np.asarray([frame.raw_frame for frame in selected], dtype=np.int64),
            memory_to_memory=np.stack([frame.attention_matrix for frame in selected]),
        )
        np.savez_compressed(
            episode_dir / "selected_causal_ablation.npz",
            raw_frame=np.asarray([frame.raw_frame for frame in selected], dtype=np.int64),
            **{name: np.stack([frame.causal_maps[name] for frame in selected]) for name in CAUSAL_MAPS},  # type: ignore[index]
            retrieved_token_norm=np.stack([frame.retrieved_token_norm for frame in selected]),
        )
        with (episode_dir / "frame_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            fieldnames = ["raw_frame", "policy_step", "task", *measured[0].scalar, "causal_ablation"]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for frame in measured:
                writer.writerow(
                    {
                        "raw_frame": frame.raw_frame,
                        "policy_step": frame.policy_step,
                        "task": frame.phase,
                        **frame.scalar,
                        "causal_ablation": frame.causal_maps is not None,
                    }
                )

        videos = {}
        if self.options.render_video:
            normalization = token_heatmap.NormalizationSpec(
                lower_percentile=0.0,
                upper_percentile=self.options.upper_percentile,
                anchor_zero=True,
                exclude_letterbox_padding=self.options.exclude_letterbox_padding,
            )
            attention_semantics = {
                "query_memory_mass": "absolute attention mass each memory-query row routes to all 256 memory keys",
                "key_incoming_absolute": "mean absolute attention received by each old-memory key across 256 queries",
                "key_incoming_conditional": "old-memory key share conditional on attention entering the memory block",
                "same_slot_diagonal": "absolute same-slot memory-query to memory-key diagonal attention",
            }
            for name in ATTENTION_MAPS:
                details = token_heatmap.export_heatmap_video(
                    self._token_frames(source, measured, name, causal=False),
                    episode_dir / "heatmaps" / name,
                    metric_name=f"{name}_L{self.options.layer}",
                    metric_semantics=attention_semantics[name],
                    fps=self.options.video_fps or source.control_hz / self.stride,
                    normalization=normalization,
                    alpha=self.options.alpha,
                    colormap=self.options.colormap,
                    video_encoder=_writer.WRITER_VIDEO_ENCODER,
                )
                videos[name] = str(Path("heatmaps") / name / details["video"])
            causal_semantics = {
                "final_ct_ablation_relative": "relative final-c18 change caused by zeroing one retrieved-memory slot",
                "injection_ablation_relative": "relative associative-injection change caused by zeroing one retrieved-memory slot",
            }
            for name in CAUSAL_MAPS:
                details = token_heatmap.export_heatmap_video(
                    self._token_frames(source, measured, name, causal=True),
                    episode_dir / "heatmaps" / name,
                    metric_name=name,
                    metric_semantics=causal_semantics[name],
                    fps=1.0,
                    normalization=normalization,
                    alpha=self.options.alpha,
                    colormap=self.options.colormap,
                    video_encoder=_writer.WRITER_VIDEO_ENCODER,
                )
                videos[name] = str(Path("heatmaps") / name / details["video"])

        selected_statistics = {
            str(frame.raw_frame): {
                **frame.scalar,
                **{
                    name: token_heatmap.metric_statistics(frame.causal_maps[name])  # type: ignore[index]
                    for name in CAUSAL_MAPS
                },
            }
            for frame in selected
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": source.episode_id,
            "ground_truth_side": source.ground_truth_side,
            "layer": self.options.layer,
            "head_reduction": "mean across all attention heads",
            "total_raw_frames": total_raw_frames,
            "sampled_frames": len(measured),
            "ablation_raw_frames": list(self.options.ablation_raw_frames),
            "selected_statistics": selected_statistics,
            "videos": videos,
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
                self._save_episode(
                    source,
                    measured,
                    total,
                    episodes_dir / _writer._safe_name(source.episode_id),  # noqa: SLF001
                )
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "checkpoint": str(self.options.checkpoint),
            "checkpoint_params_hash": _writer.v31_pi0._path_hash(self.options.checkpoint / "params"),  # noqa: SLF001
            "config": self.options.config,
            "layer": self.options.layer,
            "episode_indices": list(self.options.episode_indices),
            "stride": self.stride,
            "ablation_raw_frames": list(self.options.ablation_raw_frames),
            "ablation_chunk_size": self.options.ablation_chunk_size,
            "elapsed_s": time.monotonic() - start,
            "interpretation_warnings": [
                "Attention is routing evidence, not value-weighted causality.",
                "The 16x16 overlays are memory-token slot alignment, not strict raw-pixel attribution.",
                "Raw overlays crop letterbox padding, but every full 256-slot array is retained in NPZ.",
                "Causal maps zero one retrieved slot at a time; effects are nonlinear and not additive.",
                "Head-averaged matrices can dilute specialized individual heads.",
                "This run uses the existing v3.1 checkpoint-2500 and does not evaluate a trained v3.2 model.",
            ],
            "episodes": summaries,
        }
        _writer._write_json(self.options.output_dir / "run_manifest.json", manifest)  # noqa: SLF001
        return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v3.1 memory-to-memory attention and causal slot ablation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-indices", type=_writer._parse_episode_indices, default=(0, 2))  # noqa: SLF001
    parser.add_argument("--config", default="pi05_yam_mem_v31")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--layer", type=int, default=8)
    parser.add_argument("--ablation-raw-frames", type=_parse_ints, default=(350, 360, 370, 380))
    parser.add_argument("--ablation-chunk-size", type=int, default=8)
    parser.add_argument("--video-fps", type=float)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--colormap", choices=token_heatmap.SUPPORTED_COLORMAPS, default="inferno")
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _parser().parse_args(argv)
    options = RunOptions(
        checkpoint=namespace.checkpoint,
        dataset_root=namespace.dataset_root,
        output_dir=namespace.output_dir,
        episode_indices=tuple(namespace.episode_indices),
        config=namespace.config,
        stride=namespace.stride,
        layer=namespace.layer,
        ablation_raw_frames=tuple(namespace.ablation_raw_frames),
        ablation_chunk_size=namespace.ablation_chunk_size,
        video_fps=namespace.video_fps,
        render_video=not namespace.skip_video,
        colormap=namespace.colormap,
        alpha=namespace.alpha,
        upper_percentile=namespace.upper_percentile,
    )
    manifest = MemoryToMemoryReplayRunner(options).run()
    print(json.dumps(_writer._jsonable(manifest), indent=2, sort_keys=True), flush=True)  # noqa: SLF001
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
