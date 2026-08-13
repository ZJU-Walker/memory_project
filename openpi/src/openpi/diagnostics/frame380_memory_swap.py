"""Exact frame-380 v3.1 complete-memory swap on checkpoint-2500.

The two recurrent memories are replayed normally through raw frame 370. At raw frame 380,
each fixed current observation is evaluated against both complete pre-write states with the
same random seed/noise and with writes disabled. The artifact records full retrieved vectors,
canonical left/right sequence margins, free-decoded subtasks, and robot-space action chunks.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from openpi.diagnostics import v31_pi0
from openpi.diagnostics import writer_echo_factorial as _echo
from openpi.models import tokenizer as _tokenizer
from openpi.shared import nnx_utils
from openpi.shared import normalize as _normalize
import openpi.transforms as _transforms

SCHEMA_VERSION = "openpi.v31.frame380_memory_swap.v1"


@dataclasses.dataclass(frozen=True)
class RunOptions:
    checkpoint: Path
    dataset_root: Path
    output_dir: Path
    target_raw_frame: int = 380
    left_episode: int = 0
    right_episode: int = 2
    config: str = "pi05_yam_mem_v31"
    stride: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("checkpoint", "dataset_root", "output_dir"):
            object.__setattr__(self, name, getattr(self, name).expanduser().resolve())
        if self.target_raw_frame < 0 or self.stride <= 0 or self.target_raw_frame % self.stride:
            raise ValueError("target_raw_frame must be non-negative and divisible by stride")
        if self.left_episode < 0 or self.right_episode < 0 or self.left_episode == self.right_episode:
            raise ValueError("left/right episode indices must be distinct non-negative integers")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must fit uint32")


def _strict_json(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("artifact contains non-finite float")
        return value
    if isinstance(value, np.generic):
        return _strict_json(value.item())
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_strict_json(item) for item in value]
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_strict_json(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _batch_observations(*observations):
    return jax.tree.map(
        lambda *values: None if values[0] is None else jnp.concatenate(values, axis=0),
        *observations,
        is_leaf=lambda value: value is None,
    )


def _take_state(state, index: int):
    return jax.tree.map(lambda value: value[index : index + 1], state)


def vector_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("vector comparison requires same-shaped finite arrays")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    difference = float(np.linalg.norm(left - right))
    denominator = max(left_norm * right_norm, 1e-12)
    return {
        "left_norm": left_norm,
        "right_norm": right_norm,
        "difference_l2": difference,
        "difference_relative_to_mean_norm": difference / max(0.5 * (left_norm + right_norm), 1e-12),
        "cosine": float(np.vdot(left, right).real / denominator),
    }


def action_motion(actions: np.ndarray, raw_state: np.ndarray) -> dict[str, Any]:
    """Describe which seven-joint arm moves more; this is not a task-stage oracle."""
    actions = np.asarray(actions, dtype=np.float64)
    raw_state = np.asarray(raw_state, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 14 or raw_state.shape != (14,):
        raise ValueError(f"expected actions [H,14] and state [14], got {actions.shape}/{raw_state.shape}")
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(raw_state)):
        raise ValueError("action motion inputs must be finite")
    displacement = actions - raw_state[None, :]
    trajectory = [float(np.sqrt(np.mean(np.square(displacement[:, arm])))) for arm in (slice(0, 7), slice(7, 14))]
    endpoint = [float(np.linalg.norm(displacement[-1, arm])) for arm in (slice(0, 7), slice(7, 14))]
    margin = trajectory[0] - trajectory[1]
    return {
        "left_arm_trajectory_rms": trajectory[0],
        "right_arm_trajectory_rms": trajectory[1],
        "left_arm_endpoint_l2": endpoint[0],
        "right_arm_endpoint_l2": endpoint[1],
        "trajectory_left_minus_right": margin,
        "trajectory_normalized_margin": margin / max(trajectory[0] + trajectory[1], 1e-12),
        "larger_motion_arm": "left" if margin > 0 else "right" if margin < 0 else "tie",
    }


class Frame380MemorySwapRunner:
    def __init__(self, options: RunOptions):
        self.options = options
        if options.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite output directory: {options.output_dir}")
        echo_options = _echo.EchoRunOptions(
            checkpoint=options.checkpoint,
            dataset_root=options.dataset_root,
            output_dir=options.output_dir,
            left_episode=options.left_episode,
            right_episode=options.right_episode,
            config=options.config,
            stride=options.stride,
        )
        self.echo = _echo.WriterEchoFactorialRunner(echo_options)
        self.read_step = nnx_utils.module_jit(self.echo.base.model.memory_swap_read_step)
        self.write_step = self.echo.base._step  # noqa: SLF001
        self.evaluator = self._make_evaluator()

    def _make_evaluator(self) -> v31_pi0.Pi0SnapshotEvaluator:
        train_config = self.echo.base.train_config
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
        if data_config.asset_id is None:
            raise ValueError("YAM swap diagnostics require a normalization-stat asset_id")
        norm_stats = _normalize.load(self.options.checkpoint / "assets" / data_config.asset_id)
        decode_tokenizer = _tokenizer.FASTSubtaskTokenizer(train_config.model.max_token_len)._paligemma_tokenizer  # noqa: SLF001
        stop_token = int(decode_tokenizer.encode("placeholder subtask\n")[-1])
        output_transforms = [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
        return v31_pi0.Pi0SnapshotEvaluator(
            self.echo.base.model,
            decode_tokenizer=decode_tokenizer,
            stop_token=stop_token,
            output_transforms=output_transforms,
            configured_write_every_frames=self.echo.base.configured_stride,
            control_hz=30.0,
            action_side_config=None,
        )

    @staticmethod
    def _diagnostic_observation(frame: _echo.ReplayFrame, *, episode_id: str) -> v31_pi0.Pi0DiagnosticObservation:
        return v31_pi0.Pi0DiagnosticObservation(
            model_observation=frame.observation,
            transformed_state=np.asarray(frame.observation.state[0]),
            raw_state=np.array(frame.raw_state, copy=True),
            episode_id=episode_id,
            observation_id=f"{episode_id}:frame:{frame.raw_frame}",
            raw_frame=frame.raw_frame,
            policy_step=frame.raw_frame // 10,
            phase="visible",
            wall_time_s=frame.raw_frame / 30.0,
        )

    def run(self) -> Path:
        left_frames = self.echo._load_frames(  # noqa: SLF001
            self.echo.left_source, max_raw_frame=self.options.target_raw_frame
        )
        right_frames = self.echo._load_frames(  # noqa: SLF001
            self.echo.right_source, max_raw_frame=self.options.target_raw_frame
        )
        if [frame.raw_frame for frame in left_frames] != [frame.raw_frame for frame in right_frames]:
            raise ValueError("paired episode sample frames differ before target")
        if not left_frames or left_frames[-1].raw_frame != self.options.target_raw_frame:
            raise ValueError("target frame was not loaded from both episodes")

        paired_state = self.echo.base.model.memory.init_state(2)
        for left, right in zip(left_frames[:-1], right_frames[:-1], strict=True):
            paired_state, _ = self.write_step(
                _batch_observations(left.observation, right.observation),
                paired_state,
                allow_write=True,
            )
        writes = len(left_frames) - 1
        if writes != self.options.target_raw_frame // self.options.stride:
            raise RuntimeError(f"expected {self.options.target_raw_frame // self.options.stride} writes, got {writes}")

        states = {
            "left_memory": v31_pi0._make_fast_state(_take_state(paired_state, 0), writes=writes),  # noqa: SLF001
            "right_memory": v31_pi0._make_fast_state(_take_state(paired_state, 1), writes=writes),  # noqa: SLF001
        }
        target_frames = {"left_observation": left_frames[-1], "right_observation": right_frames[-1]}

        events: dict[str, Any] = {}
        action_arrays: dict[str, np.ndarray] = {}
        read_arrays: dict[str, np.ndarray] = {}
        comparisons: dict[str, Any] = {}
        for observation_name, frame in target_frames.items():
            episode_id = (
                self.echo.left_source.episode_id
                if observation_name.startswith("left")
                else self.echo.right_source.episode_id
            )
            diagnostic_observation = self._diagnostic_observation(frame, episode_id=episode_id)
            repeated_observation = _batch_observations(frame.observation, frame.observation)
            paired_for_read = jax.tree.map(
                lambda left, right: jnp.concatenate([left, right], axis=0),
                states["left_memory"].memory_state,
                states["right_memory"].memory_state,
            )
            read_output = jax.device_get(self.read_step(repeated_observation, paired_for_read))
            for state_index, state_name in enumerate(("left_memory", "right_memory")):
                branch = f"{observation_name}__{state_name}"
                read_arrays[f"{branch}__retrieved"] = np.asarray(read_output["retrieved"])[state_index]
                read_arrays[f"{branch}__final_ct"] = np.asarray(read_output["final_ct"])[state_index]
                result = self.evaluator.evaluate_snapshot(
                    diagnostic_observation,
                    states[state_name],
                    None,
                    allow_write=False,
                    seed=self.options.seed,
                )
                action = np.asarray(result.pre_rtc_action)
                action_arrays[branch] = action
                event = result.to_event_dict(test_name="frame380_memory_swap", branch=branch)
                event["action_motion"] = action_motion(action, frame.raw_state)
                events[branch] = event

            left_branch = f"{observation_name}__left_memory"
            right_branch = f"{observation_name}__right_memory"
            comparisons[observation_name] = {
                "retrieved": vector_comparison(
                    read_arrays[f"{left_branch}__retrieved"], read_arrays[f"{right_branch}__retrieved"]
                ),
                "final_ct": vector_comparison(
                    read_arrays[f"{left_branch}__final_ct"], read_arrays[f"{right_branch}__final_ct"]
                ),
                "actions": vector_comparison(action_arrays[left_branch], action_arrays[right_branch]),
                "delta_lr_left_memory": events[left_branch]["delta_lr"],
                "delta_lr_right_memory": events[right_branch]["delta_lr"],
                "delta_lr_swap_effect": events[left_branch]["delta_lr"] - events[right_branch]["delta_lr"],
                "decoded_subtask_left_memory": events[left_branch]["decoded_subtask"],
                "decoded_subtask_right_memory": events[right_branch]["decoded_subtask"],
                "action_motion_left_memory": events[left_branch]["action_motion"],
                "action_motion_right_memory": events[right_branch]["action_motion"],
            }

        self.options.output_dir.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(self.options.output_dir / "retrieved_and_final_ct.npz", **read_arrays)
        np.savez_compressed(self.options.output_dir / "actions.npz", **action_arrays)
        _write_json(
            self.options.output_dir / "results.json",
            {
                "schema_version": SCHEMA_VERSION,
                "events": events,
                "comparisons": comparisons,
            },
        )
        _write_json(
            self.options.output_dir / "run_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "checkpoint": str(self.options.checkpoint),
                "checkpoint_params_sha256": _echo._sha256_directory(self.options.checkpoint / "params"),  # noqa: SLF001
                "config": self.options.config,
                "target_raw_frame": self.options.target_raw_frame,
                "pre_write_frames": list(range(0, self.options.target_raw_frame, self.options.stride)),
                "pre_write_count": writes,
                "left_episode": self.echo.left_source.episode_id,
                "right_episode": self.echo.right_source.episode_id,
                "seed": self.options.seed,
                "allow_write_at_target": False,
                "rtc": "none",
                "restore_dtype": "float32",
                "warnings": [
                    "Frame 380 is still task_index=0/observe bins in both demonstrations.",
                    "Its 50-step action horizon ends near raw frame 429, before the demonstrated side-opening stage.",
                    "Larger-motion arm is descriptive at this frame and is not a validated bin-side classifier.",
                    "Checkpoint static writer provenance is legacy/unverified; runtime config is explicitly v3.1.",
                ],
            },
        )
        return self.options.output_dir


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="frame-380 v3.1 complete-memory causal swap")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-raw-frame", type=int, default=380)
    parser.add_argument("--left-episode", type=int, default=0)
    parser.add_argument("--right-episode", type=int, default=2)
    parser.add_argument("--config", default="pi05_yam_mem_v31")
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    output = Frame380MemorySwapRunner(RunOptions(**vars(parser.parse_args(argv)))).run()
    print(f"frame-380 memory swap complete: {output}")
