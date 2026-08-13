"""Paired v3.1 writer echo factorial on an existing checkpoint.

For each matched left/right policy step this runner evaluates, from identical pre-write state,
the four cells A(O,M), A(O,0), A(O_swap,M), and A(O_swap,0). A is primarily the exact clipped
associative injection S+ - eta*S; final c_t, fast-weight update, and full-state update are also
reported. Only the matched A(O,M) branch advances each episode's recurrent state.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import csv
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.diagnostics import v31
from openpi.diagnostics import writer_contribution as _writer
from openpi.shared import nnx_utils

SCHEMA_VERSION = "openpi.v31.writer_echo_factorial.v1"


@dataclasses.dataclass(frozen=True)
class EchoRunOptions:
    checkpoint: Path
    dataset_root: Path
    output_dir: Path
    left_episode: int = 0
    right_episode: int = 2
    config: str = "pi05_yam_mem_v31"
    stride: int | None = None
    max_steps: int | None = None
    reveal_frame: int = 300
    close_frame: int = 450

    def __post_init__(self) -> None:
        for name in ("checkpoint", "dataset_root", "output_dir"):
            object.__setattr__(self, name, getattr(self, name).expanduser().resolve())
        if self.left_episode < 0 or self.right_episode < 0 or self.left_episode == self.right_episode:
            raise ValueError("left/right episode indices must be distinct non-negative integers")
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 0 <= self.reveal_frame < self.close_frame:
            raise ValueError("require 0 <= reveal_frame < close_frame")


@dataclasses.dataclass(frozen=True)
class ReplayFrame:
    raw_frame: int
    phase: str
    observation: Any
    raw_state: np.ndarray


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("artifact contains non-finite float")
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_directory(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(v31.sha256_file(file_path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _phase(raw_frame: int, *, reveal: int, close: int) -> str:
    if raw_frame < reveal:
        return "pre_reveal"
    if raw_frame < close:
        return "visible"
    return "post_close"


def _batched_observation(left: ReplayFrame, right: ReplayFrame):
    # Required core order: [left obs, right obs, right obs, left obs].
    observations = (left.observation, right.observation, right.observation, left.observation)
    return jax.tree.map(
        lambda *values: None if values[0] is None else jnp.concatenate(values, axis=0),
        *observations,
        is_leaf=lambda value: value is None,
    )


def summarize_rows(rows: list[dict[str, Any]], metric_names: tuple[str, ...]) -> dict[str, Any]:
    """Aggregate scalar metrics overall and by phase/side without weighting episodes by length."""
    summary: dict[str, Any] = {}
    groups = {"overall": rows}
    for side in ("left", "right"):
        groups[side] = [row for row in rows if row["memory_side"] == side]
    for phase in ("pre_reveal", "visible", "post_close"):
        groups[phase] = [row for row in rows if row["phase"] == phase]
    for group_name, group in groups.items():
        if not group:
            continue
        summary[group_name] = {"count": len(group)}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in group], dtype=np.float64)
            summary[group_name][metric] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return summary


class WriterEchoFactorialRunner:
    def __init__(self, options: EchoRunOptions):
        self.options = options
        if options.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite output directory: {options.output_dir}")
        base_options = _writer.RunOptions(
            checkpoint=options.checkpoint,
            dataset_root=options.dataset_root,
            output_dir=options.output_dir,
            episode_indices=(options.left_episode, options.right_episode),
            config=options.config,
            stride=options.stride,
            render_video=False,
        )
        self.base = _writer.WriterContributionRunner(base_options)
        by_side = {source.ground_truth_side: source for source in self.base.sources}
        if set(by_side) != {"left", "right"}:
            raise ValueError(f"selected episodes must provide one left and one right side; got {sorted(by_side)}")
        self.left_source = by_side["left"]
        self.right_source = by_side["right"]
        self._step = nnx_utils.module_jit(self.base.model.writer_echo_factorial_metrics_step)

    def _load_frames(self, source: _writer.EpisodeSource, *, max_raw_frame: int | None = None) -> list[ReplayFrame]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyarrow is required for LeRobot factorial replay") from exc

        parquet = pq.ParquetFile(source.path)
        columns = ["image", "left_wrist_image", "right_wrist_image", "state", "frame_index", "task_index"]
        missing = set(columns) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"episode {source.episode_id!r} lacks columns {sorted(missing)}")
        frames: list[ReplayFrame] = []
        expected_frame = 0
        for batch in parquet.iter_batches(batch_size=100, columns=columns):
            for row in batch.to_pylist():
                raw_frame = row["frame_index"]
                if raw_frame != expected_frame:
                    raise ValueError(
                        f"episode {source.episode_id!r} frame_index must be contiguous; "
                        f"expected {expected_frame}, got {raw_frame!r}"
                    )
                expected_frame += 1
                if max_raw_frame is not None and raw_frame > max_raw_frame:
                    return frames
                if raw_frame % self.base.stride:
                    continue
                state = np.asarray(row["state"], dtype=np.float32)
                if state.shape != (14,) or not np.all(np.isfinite(state)):
                    raise ValueError(f"invalid state at {source.episode_id} frame {raw_frame}")
                images = {
                    name: self.base._decode_inline_image(row[field], field=field, raw_frame=raw_frame)  # noqa: SLF001
                    for name, field in (
                        ("top", "image"),
                        ("left", "left_wrist_image"),
                        ("right", "right_wrist_image"),
                    )
                }
                observation, _ = self.base._transform_observation(  # noqa: SLF001
                    source,
                    raw_frame,
                    images["top"],
                    images["left"],
                    images["right"],
                    state,
                )
                frames.append(
                    ReplayFrame(
                        raw_frame=raw_frame,
                        phase=_phase(
                            raw_frame,
                            reveal=self.options.reveal_frame,
                            close=self.options.close_frame,
                        ),
                        observation=observation,
                        raw_state=np.array(state, copy=True),
                    )
                )
        return frames

    def run(self) -> Path:
        left_frames = self._load_frames(self.left_source)
        right_frames = self._load_frames(self.right_source)
        common_steps = min(len(left_frames), len(right_frames))
        if self.options.max_steps is not None:
            common_steps = min(common_steps, self.options.max_steps)
        if common_steps < 1:
            raise ValueError("paired episodes contain no common sampled steps")

        paired_state = self.base.model.memory.init_state(2)
        rows: list[dict[str, Any]] = []
        metric_names: tuple[str, ...] | None = None
        for policy_step in range(common_steps):
            left = left_frames[policy_step]
            right = right_frames[policy_step]
            if left.raw_frame != right.raw_frame:
                raise ValueError(
                    f"paired raw frames differ at policy step {policy_step}: {left.raw_frame} vs {right.raw_frame}"
                )
            paired_state, metrics = self._step(_batched_observation(left, right), paired_state)
            metrics = jax.device_get(metrics)
            if metric_names is None:
                metric_names = tuple(sorted(metrics))
            for side_index, side in enumerate(("left", "right")):
                row: dict[str, Any] = {
                    "policy_step": policy_step,
                    "raw_frame": left.raw_frame,
                    "phase": left.phase,
                    "memory_side": side,
                    "observation_side": side,
                    "swapped_observation_side": "right" if side == "left" else "left",
                }
                for name in metric_names:
                    value = float(np.asarray(metrics[name])[side_index])
                    if not math.isfinite(value):
                        raise FloatingPointError(f"non-finite {name} at step {policy_step}/{side}")
                    row[name] = value
                rows.append(row)
            print(
                f"step {policy_step + 1}/{common_steps} raw={left.raw_frame} "
                f"injection memory/obs ratio L={rows[-2]['injection_memory_to_observation_main_ratio']:.4f} "
                f"R={rows[-1]['injection_memory_to_observation_main_ratio']:.4f}",
                flush=True,
            )

        assert metric_names is not None
        self.options.output_dir.mkdir(parents=True, exist_ok=False)
        with (self.options.output_dir / "factorial_steps.csv").open("w", newline="", encoding="utf-8") as stream:
            fieldnames = [
                "policy_step",
                "raw_frame",
                "phase",
                "memory_side",
                "observation_side",
                "swapped_observation_side",
                *metric_names,
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        summary = summarize_rows(rows, metric_names)
        _write_json(self.options.output_dir / "summary.json", summary)
        _write_json(
            self.options.output_dir / "run_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "checkpoint": str(self.options.checkpoint),
                "checkpoint_params_sha256": _sha256_directory(self.options.checkpoint / "params"),
                "config": self.options.config,
                "memory_write_source": self.base.train_config.model.memory_write_source,
                "left_episode": self.left_source.episode_id,
                "right_episode": self.right_source.episode_id,
                "stride": self.base.stride,
                "common_steps": common_steps,
                "restore_dtype": "float32",
                "phase_boundaries": {
                    "reveal_frame": self.options.reveal_frame,
                    "close_frame": self.options.close_frame,
                    "status": "operator-supplied/default heuristic; reveal annotation file is absent",
                },
                "branch_order": ["O_left,M_left", "O_right,M_left", "O_right,M_right", "O_left,M_right"],
                "definitions": {
                    "A": "exact clipped associative injection S_plus - eta*S from memory.write",
                    "memory_effect": "A(O,M)-A(O,zero_read)",
                    "observation_effect": "A(O,M)-A(O_swap,M)",
                    "interaction": "A(O,M)-A(O,0)-A(O_swap,M)+A(O_swap,0)",
                    "commit": "only matched normal-read A(O,M) advances each recurrent episode state",
                },
                "warnings": [
                    "This is a v3.1 checkpoint-2500 diagnostic, not a trained v3.2 result.",
                    "O_swap changes the complete current observation (three cameras and robot state), not only banana pixels.",
                    "Factorial effect vectors need not be orthogonal; norm ratios are descriptive and do not sum to 100%.",
                    "Checkpoint static writer provenance is legacy/unverified; runtime config is explicitly pi05_yam_mem_v31.",
                ],
            },
        )
        return self.options.output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="v3.1 A(O,M) writer echo factorial on checkpoint-2500")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left-episode", type=int, default=0)
    parser.add_argument("--right-episode", type=int, default=2)
    parser.add_argument("--config", default="pi05_yam_mem_v31")
    parser.add_argument("--stride", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--reveal-frame", type=int, default=300)
    parser.add_argument("--close-frame", type=int, default=450)
    args = parser.parse_args(argv)
    output = WriterEchoFactorialRunner(EchoRunOptions(**vars(args))).run()
    print(f"writer echo factorial complete: {output}")
