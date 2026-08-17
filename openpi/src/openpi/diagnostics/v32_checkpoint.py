"""Offline v3.2 checkpoint diagnostics in one replay pass per episode.

For every scheduled memory frame this runner measures, against the identical pre-write state:

1. the zero-read counterfactual (handoff section 12.6): free-decoded subtask, denoised actions,
   and teacher-forced ground-truth subtask log-probability with the normal read versus zeroed
   retrieved memory;
2. the dual-query attention maps (12.1-12.2): all 16 read-query and 16 write-query
   query->patch distributions over the layer-8 top-camera slots, rendered as H.264 videos;
3. write/retrieval statistics over time (12.3-12.4): write gate/surprise/per-slot error,
   cross-slot selectivity, retrieval norms and step-to-step similarity, and fast-weight drift.

The committed write is exactly the normal-path `sample_with_memory` write, once per frame; every
counterfactual runs with ``allow_write=False`` so replay dynamics match deployment.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.diagnostics import token_heatmap
from openpi.diagnostics import writer_contribution as _wc
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import nnx_utils
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
import openpi.transforms as _transforms

SCHEMA_VERSION = "openpi.v32.checkpoint_diagnostics.v1"
_EPS = 1e-12


@dataclasses.dataclass(frozen=True)
class Options:
    checkpoint: Path
    config: str
    dataset_root: Path
    episode_indices: tuple[int, ...]
    output_dir: Path
    stride: int | None = None
    anchor_frame: int | None = None
    max_frames: int | None = None
    num_denoise_steps: int = 10
    max_decode_steps: int = 10
    video_fps: float = 5.0
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", Path(self.checkpoint).expanduser().resolve())
        object.__setattr__(self, "dataset_root", Path(self.dataset_root).expanduser().resolve())
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        object.__setattr__(self, "episode_indices", tuple(self.episode_indices))
        if not self.episode_indices:
            raise ValueError("provide at least one --episode-indices entry")
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if self.num_denoise_steps <= 0 or self.max_decode_steps <= 0:
            raise ValueError("denoise/decode step counts must be positive")
        if not math.isfinite(self.video_fps) or self.video_fps <= 0:
            raise ValueError("video_fps must be finite and positive")


def _tree_sq_sum(tree: Any) -> jnp.ndarray:
    return sum(jnp.vdot(leaf, leaf) for leaf in jax.tree.leaves(tree))


def _tree_dot(a: Any, b: Any) -> jnp.ndarray:
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True))


def _tree_diff(a: Any, b: Any) -> Any:
    return jax.tree.map(lambda x, y: x - y, a, b)


@jax.jit
def _state_step_metrics(new_state: Any, prev_state: Any, init_state: Any) -> dict[str, jnp.ndarray]:
    out = {}
    for name in ("fast_weights", "momentum"):
        new = getattr(new_state, name)
        prev = getattr(prev_state, name)
        init = getattr(init_state, name)
        norm = jnp.sqrt(_tree_sq_sum(new))
        prev_norm = jnp.sqrt(_tree_sq_sum(prev))
        out[f"{name}_norm"] = norm
        out[f"{name}_step_delta"] = jnp.sqrt(_tree_sq_sum(_tree_diff(new, prev)))
        out[f"{name}_drift_from_init"] = jnp.sqrt(_tree_sq_sum(_tree_diff(new, init)))
        out[f"{name}_cosine_prev"] = _tree_dot(new, prev) / jnp.maximum(norm * prev_norm, _EPS)
    return out


@jax.jit
def _state_anchor_metrics(state: Any, anchor: Any) -> dict[str, jnp.ndarray]:
    out = {}
    for name in ("fast_weights", "momentum"):
        current = getattr(state, name)
        reference = getattr(anchor, name)
        norm = jnp.sqrt(_tree_sq_sum(current))
        ref_norm = jnp.sqrt(_tree_sq_sum(reference))
        out[f"{name}_drift_from_anchor"] = jnp.sqrt(_tree_sq_sum(_tree_diff(current, reference)))
        out[f"{name}_cosine_anchor"] = _tree_dot(current, reference) / jnp.maximum(norm * ref_norm, _EPS)
    return out


def _align_params(expected: Any, got: Any) -> Any:
    """Project restored params onto the abstract state, reinserting dropped ``None`` slots.

    flax 0.10 keeps a ``bias: None`` entry in module state for ``nnx.Linear(use_bias=False)``
    (the v3.2 query compressors), which the checkpoint writer cannot serialize, so restored
    v3.2 params fail ``BaseModelConfig.load``'s strict structure check.  This walk keeps only
    keys the model expects (mirroring ``remove_extra_params``, which would strip the ``None``
    slots again) and fills a key with ``None`` only when the abstract state holds exactly
    ``None`` there; genuinely missing arrays still fail the strict check.
    """
    if not isinstance(expected, dict) or not isinstance(got, dict):
        return got
    aligned = {}
    for key, expected_child in expected.items():
        if key in got:
            aligned[key] = _align_params(expected_child, got[key])
        elif expected_child is None:
            aligned[key] = None
    return aligned


def _entropy_stats(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-slot normalized entropy and effective token count for [slots, tokens] maps."""
    clipped = np.clip(probs.astype(np.float64), _EPS, 1.0)
    clipped /= clipped.sum(axis=-1, keepdims=True)
    entropy = -np.sum(clipped * np.log(clipped), axis=-1)
    return entropy / math.log(probs.shape[-1]), np.exp(entropy)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("summary values must be finite")
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list | np.ndarray):
        return [_jsonable(item) for item in np.asarray(value).tolist()] if isinstance(value, np.ndarray) else [
            _jsonable(item) for item in value
        ]
    raise TypeError(f"cannot serialize {type(value).__name__}")


@dataclasses.dataclass
class _FrameRecord:
    raw_frame: int
    policy_step: int
    gt_subtask: str
    decoded_normal: str
    decoded_zero: str
    model_image_rgb: np.ndarray
    read_attention: np.ndarray  # [heads, slots, tokens] fp32
    write_attention: np.ndarray
    retrieved: np.ndarray  # [slots, width]
    write_tokens: np.ndarray
    actions_normal: np.ndarray  # [horizon, action_dim] model space
    actions_zero: np.ndarray
    robot_actions_normal: np.ndarray  # [horizon, robot_dims]
    robot_actions_zero: np.ndarray
    scalar: dict[str, float]


class V32CheckpointRunner:
    """Checkpoint-backed single-pass LeRobot replay for the v3.2 dual-query interface."""

    def __init__(self, options: Options):
        self.options = options
        self.sources = _wc._load_lerobot_sources(options.dataset_root, options.episode_indices)  # noqa: SLF001
        if options.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite output directory: {options.output_dir}")
        if not (options.checkpoint / "params").is_dir():
            raise FileNotFoundError(f"checkpoint has no params item: {options.checkpoint}")

        self.train_config = _config.get_config(options.config)
        model_config = self.train_config.model
        if getattr(model_config, "memory_architecture", "v3_v31") != "v32_layer8_dual_query":
            raise ValueError("v3.2 checkpoint diagnostics require memory_architecture='v32_layer8_dual_query'")
        data_config = self.train_config.data.create(self.train_config.assets_dirs, model_config)
        if data_config.asset_id is None:
            raise ValueError("diagnostics require a normalization-stat asset_id")
        configured_stride = int(data_config.memory_stride_frames)
        self.stride = configured_stride if options.stride is None else options.stride
        norm_stats = _normalize.load(options.checkpoint / "assets" / data_config.asset_id)

        raw_params = _model.restore_params(options.checkpoint / "params", dtype=jnp.float32)
        abstract_state = nnx.state(nnx.eval_shape(model_config.create, jax.random.key(0))).to_pure_dict()
        self.model = model_config.load(_align_params(abstract_state, raw_params), remove_extra_params=False)
        self.tokenizer = _tokenizer.FASTSubtaskTokenizer(model_config.max_token_len)._paligemma_tokenizer  # noqa: SLF001
        self.stop_token = int(self.tokenizer.encode("placeholder subtask\n")[-1])

        input_transforms = [
            transform
            for transform in data_config.data_transforms.inputs
            if not isinstance(transform, _transforms.BuildMemorySequence)
        ]
        self.input_transform = _transforms.compose(
            [
                *input_transforms,
                _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )
        self.output_transform = _transforms.compose(
            [
                *data_config.model_transforms.outputs,
                _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.data_transforms.outputs,
            ]
        )
        self._qstep = nnx_utils.module_jit(self.model.v32_query_attention_step)
        self._sample = nnx_utils.module_jit(
            self.model.sample_with_memory,
            static_argnames=("stop_token", "max_decode_steps", "num_steps", "zero_read", "allow_write"),
        )

    def _forced_buffers(self, text: str) -> tuple[jnp.ndarray, jnp.ndarray]:
        tokens = self.tokenizer.encode(text.strip() + "\n")
        if not 0 < len(tokens) <= self.model.causal_token_len:
            raise ValueError(f"subtask {text!r} has invalid token count {len(tokens)}")
        padded = np.zeros((1, self.model.causal_token_len), dtype=np.int32)
        mask = np.zeros_like(padded, dtype=bool)
        padded[0, : len(tokens)] = tokens
        mask[0, : len(tokens)] = True
        return jnp.asarray(padded), jnp.asarray(mask)

    def _decode_tokens(self, aux: dict[str, Any]) -> str:
        tokens = np.asarray(aux["tokens"])[0]
        mask = np.asarray(aux["token_mask"])[0]
        return self.tokenizer.decode(tokens[mask].tolist()).strip()

    def _transform_observation(
        self, top_rgb: np.ndarray, left_rgb: np.ndarray, right_rgb: np.ndarray, state: np.ndarray
    ) -> tuple[_model.Observation, np.ndarray, np.ndarray]:
        transformed = self.input_transform(
            {
                "observation/image": top_rgb,
                "observation/left_wrist_image": left_rgb,
                "observation/right_wrist_image": right_rgb,
                "observation/state": state,
            }
        )
        model_image = np.asarray(transformed["image"]["base_0_rgb"])
        if model_image.shape != (224, 224, 3) or model_image.dtype != np.uint8:
            raise ValueError(f"invalid transformed top image {model_image.shape}/{model_image.dtype}")
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
        return (
            _model.Observation.from_dict(batched),
            np.array(model_image, copy=True),
            np.asarray(transformed["state"], dtype=np.float32),
        )

    def _robot_actions(self, model_actions: np.ndarray, transformed_state: np.ndarray) -> np.ndarray:
        output = self.output_transform(
            {"state": np.array(transformed_state, copy=True), "actions": np.asarray(model_actions)[0]}
        )
        actions = np.asarray(output["actions"], dtype=np.float64)
        if actions.ndim != 2 or not np.all(np.isfinite(actions)):
            raise ValueError(f"robot-space actions must be a finite rank-2 array, got {actions.shape}")
        return actions

    def _sample_variant(
        self,
        key: jnp.ndarray,
        observation: _model.Observation,
        memory_state: Any,
        noise: jnp.ndarray,
        *,
        zero_read: bool,
        allow_write: bool,
        forced: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ):
        forced_tokens, forced_mask = forced if forced is not None else (None, None)
        return self._sample(
            key,
            observation,
            memory_state,
            stop_token=self.stop_token,
            max_decode_steps=self.options.max_decode_steps,
            num_steps=self.options.num_denoise_steps,
            noise=noise,
            forced_subtask_tokens=forced_tokens,
            forced_subtask_mask=forced_mask,
            zero_read=zero_read,
            allow_write=allow_write,
        )

    def _replay_episode(self, source: _wc.EpisodeSource) -> list[_FrameRecord]:
        import pyarrow.parquet as pq

        columns = ["image", "left_wrist_image", "right_wrist_image", "state", "frame_index", "task_index"]
        table = pq.read_table(source.path, columns=columns)
        rows = table.to_pylist()
        if not rows:
            raise ValueError(f"episode {source.episode_id!r} has no rows")

        init_state = self.model.memory.init_state(1)
        memory_state = init_state
        anchor_state = None
        records: list[_FrameRecord] = []
        started = time.monotonic()
        for row in rows:
            raw_frame = int(row["frame_index"])
            if raw_frame % self.stride:
                continue
            top_rgb = _wc.WriterContributionRunner._decode_inline_image(  # noqa: SLF001
                row["image"], field="image", raw_frame=raw_frame
            )
            left_rgb = _wc.WriterContributionRunner._decode_inline_image(  # noqa: SLF001
                row["left_wrist_image"], field="left_wrist_image", raw_frame=raw_frame
            )
            right_rgb = _wc.WriterContributionRunner._decode_inline_image(  # noqa: SLF001
                row["right_wrist_image"], field="right_wrist_image", raw_frame=raw_frame
            )
            state = np.asarray(row["state"], dtype=np.float32)
            task_index = int(row["task_index"])
            if not 0 <= task_index < len(source.task_names):
                raise ValueError(f"frame {raw_frame} task_index {task_index} outside task table")
            gt_subtask = source.task_names[task_index]

            observation, model_image, transformed_state = self._transform_observation(
                top_rgb, left_rgb, right_rgb, state
            )
            key = jax.random.key(np.uint32(_wc_seed(self.options.seed, source.episode_id, raw_frame)))
            noise = jax.random.normal(key, (1, self.model.action_horizon, self.model.action_dim), dtype=jnp.float32)

            qaux = self._qstep(observation, memory_state)
            actions_normal, next_state, aux_normal = self._sample_variant(
                key, observation, memory_state, noise, zero_read=False, allow_write=True
            )
            actions_zero, _, aux_zero = self._sample_variant(
                key, observation, memory_state, noise, zero_read=True, allow_write=False
            )
            forced = self._forced_buffers(gt_subtask)
            _, _, aux_forced_normal = self._sample_variant(
                key, observation, memory_state, noise, zero_read=False, allow_write=False, forced=forced
            )
            _, _, aux_forced_zero = self._sample_variant(
                key, observation, memory_state, noise, zero_read=True, allow_write=False, forced=forced
            )
            drift = _state_step_metrics(next_state, memory_state, init_state)
            anchor_frame = self.options.anchor_frame
            if anchor_frame is not None and anchor_state is None and raw_frame >= anchor_frame:
                anchor_state = next_state
            anchored = _state_anchor_metrics(next_state, anchor_state) if anchor_state is not None else {}

            qaux, aux_normal, aux_zero, aux_forced_normal, aux_forced_zero, drift, anchored = jax.device_get(
                (qaux, aux_normal, aux_zero, aux_forced_normal, aux_forced_zero, drift, anchored)
            )
            actions_normal, actions_zero = jax.device_get((actions_normal, actions_zero))
            memory_state = next_state

            robot_normal = self._robot_actions(actions_normal, transformed_state)
            robot_zero = self._robot_actions(actions_zero, transformed_state)
            read_probs = np.asarray(qaux["read_attention"], dtype=np.float32)[0]
            write_probs = np.asarray(qaux["write_attention"], dtype=np.float32)[0]
            read_entropy, _ = _entropy_stats(read_probs.mean(axis=0))
            write_entropy, _ = _entropy_stats(write_probs.mean(axis=0))
            write_slot_norm = np.asarray(qaux["write_slot_norm"], dtype=np.float64)[0]
            action_diff = np.asarray(actions_normal, dtype=np.float64) - np.asarray(actions_zero, dtype=np.float64)
            robot_diff = robot_normal - robot_zero
            half = robot_diff.shape[1] // 2

            scalar = {
                "action_mse_model": float(np.mean(action_diff**2)),
                "action_mse_robot": float(np.mean(robot_diff**2)),
                "action_maxabs_robot": float(np.max(np.abs(robot_diff))),
                "action_mse_robot_left": float(np.mean(robot_diff[:, :half] ** 2)),
                "action_mse_robot_right": float(np.mean(robot_diff[:, half:] ** 2)),
                "gt_logp_normal": float(np.asarray(aux_forced_normal["conditioned_subtask_logp"])[0]),
                "gt_logp_zero": float(np.asarray(aux_forced_zero["conditioned_subtask_logp"])[0]),
                "gt_mean_logp_normal": float(np.asarray(aux_forced_normal["conditioned_subtask_mean_logp"])[0]),
                "gt_mean_logp_zero": float(np.asarray(aux_forced_zero["conditioned_subtask_mean_logp"])[0]),
                "retrieval_norm": float(np.asarray(aux_normal["retrieval_norm"])[0]),
                "read_query_norm": float(np.asarray(aux_normal["read_query_norm"])[0]),
                "write_token_norm": float(np.asarray(aux_normal["write_token_norm"])[0]),
                "memory_gate_norm": float(np.asarray(aux_normal["memory_gate_norm"])[0]),
                "eta": float(np.asarray(aux_normal["eta"])[0]),
                "theta": float(np.asarray(aux_normal["theta"])[0]),
                "alpha": float(np.asarray(aux_normal["alpha"])[0]),
                "surprise": float(np.asarray(aux_normal["surprise"])[0]),
                "grad_norm": float(np.asarray(aux_normal["grad_norm"])[0]),
                "write_slot_norm_cv": float(np.std(write_slot_norm) / max(np.mean(write_slot_norm), _EPS)),
                "write_slot_error_cv": float(
                    np.std(np.asarray(qaux["write_slot_token_error"])[0])
                    / max(np.mean(np.asarray(qaux["write_slot_token_error"])[0]), _EPS)
                ),
                "read_entropy_mean": float(np.mean(read_entropy)),
                "write_entropy_mean": float(np.mean(write_entropy)),
                **{f"drift_{name}": float(np.asarray(value)) for name, value in drift.items()},
                **{f"anchor_{name}": float(np.asarray(value)) for name, value in anchored.items()},
            }
            record = _FrameRecord(
                raw_frame=raw_frame,
                policy_step=len(records),
                gt_subtask=gt_subtask,
                decoded_normal=self._decode_tokens(aux_normal),
                decoded_zero=self._decode_tokens(aux_zero),
                model_image_rgb=model_image,
                read_attention=read_probs,
                write_attention=write_probs,
                retrieved=np.asarray(qaux["retrieved"], dtype=np.float32)[0],
                write_tokens=np.asarray(qaux["write_tokens"], dtype=np.float32)[0],
                actions_normal=np.asarray(actions_normal, dtype=np.float32)[0],
                actions_zero=np.asarray(actions_zero, dtype=np.float32)[0],
                robot_actions_normal=robot_normal.astype(np.float32),
                robot_actions_zero=robot_zero.astype(np.float32),
                scalar=scalar,
            )
            records.append(record)
            print(
                f"[{source.episode_id}] frame {raw_frame} step {record.policy_step} "
                f"mse_robot={scalar['action_mse_robot']:.5g} dlogp={scalar['gt_logp_normal'] - scalar['gt_logp_zero']:+.4g} "
                f"decoded={record.decoded_normal!r} zero={record.decoded_zero!r} gt={gt_subtask!r} "
                f"eta={scalar['eta']:.4g} surprise={scalar['surprise']:.4g} "
                f"({time.monotonic() - started:.0f}s)",
                flush=True,
            )
            if self.options.max_frames is not None and len(records) >= self.options.max_frames:
                break
        if not records:
            raise ValueError(f"episode {source.episode_id!r} yielded no sampled frames")
        return records

    # ---------------------------------------------------------------- rendering

    def _attention_video(self, records: list[_FrameRecord], bank: str, path: Path, episode_id: str) -> None:
        frames = []
        for record in records:
            probs = getattr(record, f"{bank}_attention").mean(axis=0)  # [slots, tokens]
            entropy, effective = _entropy_stats(probs)
            tiles = []
            for slot in range(probs.shape[0]):
                grid = probs[slot].reshape(token_heatmap.TOKEN_GRID_SIZE, token_heatmap.TOKEN_GRID_SIZE)
                heat = grid / max(float(grid.max()), _EPS)
                heat224 = cv2.resize(heat.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST)
                color = cv2.applyColorMap((heat224 * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
                weight = (0.6 * heat224)[..., None]
                tile = (record.model_image_rgb.astype(np.float32) * (1 - weight) + color * weight).astype(np.uint8)
                cv2.putText(
                    tile,
                    f"s{slot} eff={effective[slot]:.0f} H={entropy[slot]:.2f}",
                    (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (245, 245, 245),
                    1,
                )
                tiles.append(tile)
            grid_rows = [np.concatenate(tiles[index * 4 : index * 4 + 4], axis=1) for index in range(4)]
            body = np.concatenate(grid_rows, axis=0)
            header = np.full((44, body.shape[1], 3), 20, dtype=np.uint8)
            cv2.putText(
                header,
                f"{bank} queries -> layer-8 top patches | {episode_id} | raw {record.raw_frame} "
                f"step {record.policy_step} | mean H {np.mean(entropy):.2f}",
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                1,
            )
            frames.append(np.concatenate([header, body], axis=0))
        encoder = token_heatmap.encode_mp4(frames, path, self.options.video_fps, encoder="imageio")
        if "libx264" not in encoder:
            raise RuntimeError(f"attention video was not H.264-encoded: {encoder}")

    def _curves_figure(self, records: list[_FrameRecord], path: Path, episode_id: str) -> None:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt

        frames = [record.raw_frame for record in records]

        def series(name: str) -> np.ndarray:
            return np.asarray([record.scalar.get(name, np.nan) for record in records], dtype=np.float64)

        figure, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
        figure.suptitle(f"v3.2 checkpoint diagnostics | {episode_id}")

        axis = axes[0][0]
        axis.plot(frames, series("action_mse_robot"), label="robot-action MSE (normal vs zero-read)")
        axis.plot(frames, series("action_mse_robot_left"), label="left arm", alpha=0.6)
        axis.plot(frames, series("action_mse_robot_right"), label="right arm", alpha=0.6)
        axis.set_yscale("log")
        axis.set_title("zero-read action divergence")
        axis.legend(fontsize=8)

        axis = axes[0][1]
        axis.plot(frames, series("gt_logp_normal"), label="GT subtask logp (normal)")
        axis.plot(frames, series("gt_logp_zero"), label="GT subtask logp (zero-read)")
        mismatch = [record.decoded_normal != record.decoded_zero for record in records]
        for frame, differs in zip(frames, mismatch, strict=True):
            if differs:
                axis.axvline(frame, color="red", alpha=0.15)
        axis.set_title("teacher-forced GT log-prob (red: free decodes differ)")
        axis.legend(fontsize=8)

        axis = axes[1][0]
        axis.plot(frames, series("retrieval_norm"), label="retrieval RMS")
        axis.plot(frames, series("write_token_norm"), label="write-token RMS")
        cosines = [np.nan]
        for previous, current in itertools.pairwise(records):
            a, b = previous.retrieved.ravel(), current.retrieved.ravel()
            cosines.append(float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), _EPS)))
        twin = axis.twinx()
        twin.plot(frames, cosines, color="green", alpha=0.6, label="cos(retrieved_t, retrieved_t-1)")
        twin.set_ylim(-1.05, 1.05)
        axis.set_title("retrieval / write magnitude over time")
        axis.legend(fontsize=8, loc="upper left")
        twin.legend(fontsize=8, loc="lower right")

        axis = axes[1][1]
        axis.plot(frames, series("eta"), label="eta")
        axis.plot(frames, series("theta"), label="theta")
        axis.plot(frames, series("surprise"), label="surprise")
        axis.plot(frames, series("write_slot_norm_cv"), label="write slot-norm CV")
        axis.set_yscale("log")
        axis.set_title("write gates / selectivity")
        axis.legend(fontsize=8)

        axis = axes[2][0]
        axis.plot(frames, series("read_entropy_mean"), label="read maps")
        axis.plot(frames, series("write_entropy_mean"), label="write maps")
        axis.set_ylim(0, 1.02)
        axis.set_title("mean normalized attention entropy (1 = uniform)")
        axis.set_xlabel("raw frame")
        axis.legend(fontsize=8)

        axis = axes[2][1]
        axis.plot(frames, series("drift_fast_weights_norm"), label="||M_t||")
        axis.plot(frames, series("drift_fast_weights_step_delta"), label="||M_t - M_t-1||")
        axis.plot(frames, series("drift_fast_weights_drift_from_init"), label="||M_t - M_0||")
        anchored = series("anchor_fast_weights_drift_from_anchor")
        if np.any(np.isfinite(anchored)):
            axis.plot(frames, anchored, label="||M_t - M_anchor||")
        axis.set_yscale("log")
        axis.set_title("fast-weight drift")
        axis.set_xlabel("raw frame")
        axis.legend(fontsize=8)

        if self.options.anchor_frame is not None:
            for axis in axes.ravel():
                axis.axvline(self.options.anchor_frame, color="black", linestyle="--", alpha=0.4)
        figure.tight_layout()
        figure.savefig(path, dpi=130)
        plt.close(figure)

    # ---------------------------------------------------------------- outputs

    def _save_episode(self, source: _wc.EpisodeSource, records: list[_FrameRecord]) -> dict[str, Any]:
        episode_dir = self.options.output_dir / source.episode_id
        episode_dir.mkdir(parents=True, exist_ok=False)

        np.savez_compressed(
            episode_dir / "arrays.npz",
            raw_frames=np.asarray([record.raw_frame for record in records], dtype=np.int32),
            model_images=np.stack([record.model_image_rgb for record in records]),
            read_attention=np.stack([record.read_attention for record in records]).astype(np.float16),
            write_attention=np.stack([record.write_attention for record in records]).astype(np.float16),
            retrieved=np.stack([record.retrieved for record in records]).astype(np.float16),
            write_tokens=np.stack([record.write_tokens for record in records]).astype(np.float16),
            actions_normal=np.stack([record.actions_normal for record in records]),
            actions_zero=np.stack([record.actions_zero for record in records]),
            robot_actions_normal=np.stack([record.robot_actions_normal for record in records]),
            robot_actions_zero=np.stack([record.robot_actions_zero for record in records]),
        )
        self._attention_video(records, "read", episode_dir / "read_attention.mp4", source.episode_id)
        self._attention_video(records, "write", episode_dir / "write_attention.mp4", source.episode_id)
        self._curves_figure(records, episode_dir / "curves.png", source.episode_id)

        anchor = self.options.anchor_frame

        def aggregate(names: list[str], selector) -> dict[str, float]:
            chosen = [record for record in records if selector(record)]
            return {
                name: float(np.mean([record.scalar[name] for record in chosen])) for name in names if chosen
            }

        divergence_names = [
            "action_mse_robot",
            "action_mse_robot_left",
            "action_mse_robot_right",
            "gt_logp_normal",
            "gt_logp_zero",
            "retrieval_norm",
            "eta",
            "surprise",
            "write_slot_norm_cv",
            "read_entropy_mean",
            "write_entropy_mean",
        ]
        summary = {
            "schema": SCHEMA_VERSION,
            "episode_id": source.episode_id,
            "ground_truth_side": source.ground_truth_side,
            "checkpoint": self.options.checkpoint,
            "config": self.options.config,
            "stride": self.stride,
            "anchor_frame": anchor,
            "policy_steps": len(records),
            "decoded_mismatch_fraction": float(
                np.mean([record.decoded_normal != record.decoded_zero for record in records])
            ),
            "decoded_normal_matches_gt_fraction": float(
                np.mean([record.decoded_normal == record.gt_subtask.strip() for record in records])
            ),
            "decoded_zero_matches_gt_fraction": float(
                np.mean([record.decoded_zero == record.gt_subtask.strip() for record in records])
            ),
            "mean_all": aggregate(divergence_names, lambda record: True),
            "frames": [
                {
                    "raw_frame": record.raw_frame,
                    "gt_subtask": record.gt_subtask,
                    "decoded_normal": record.decoded_normal,
                    "decoded_zero": record.decoded_zero,
                    **record.scalar,
                }
                for record in records
            ],
        }
        if anchor is not None:
            summary["mean_pre_anchor"] = aggregate(divergence_names, lambda record: record.raw_frame < anchor)
            summary["mean_post_anchor"] = aggregate(divergence_names, lambda record: record.raw_frame >= anchor)
            summary["decoded_mismatch_fraction_post_anchor"] = float(
                np.mean(
                    [record.decoded_normal != record.decoded_zero for record in records if record.raw_frame >= anchor]
                    or [0.0]
                )
            )
        (episode_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
        return summary

    def run(self) -> None:
        self.options.output_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "schema": SCHEMA_VERSION,
            "options": dataclasses.asdict(self.options),
            "episodes": [source.episode_id for source in self.sources],
        }
        (self.options.output_dir / "run.json").write_text(
            json.dumps(_jsonable(manifest), indent=2) + "\n", encoding="utf-8"
        )
        for source in self.sources:
            print(f"=== replaying {source.episode_id} (side={source.ground_truth_side}) ===", flush=True)
            records = self._replay_episode(source)
            summary = self._save_episode(source, records)
            print(
                f"=== {source.episode_id}: {summary['policy_steps']} steps, "
                f"decode mismatch {summary['decoded_mismatch_fraction']:.2%}, "
                f"normal-vs-GT {summary['decoded_normal_matches_gt_fraction']:.2%}, "
                f"zero-vs-GT {summary['decoded_zero_matches_gt_fraction']:.2%} ===",
                flush=True,
            )


def _wc_seed(seed: int, episode_id: str, raw_frame: int) -> int:
    import zlib

    return (zlib.crc32(f"{seed}:{episode_id}:{raw_frame}".encode()) & 0x7FFFFFFF) or 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="pi05_yam_mem_v32")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--episode-indices", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=None, help="default: the config's memory stride")
    parser.add_argument("--anchor-frame", type=int, default=None, help="raw frame for anchored drift/aggregates")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--num-denoise-steps", type=int, default=10)
    parser.add_argument("--max-decode-steps", type=int, default=10)
    parser.add_argument("--video-fps", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args(argv)
    V32CheckpointRunner(
        Options(
            checkpoint=arguments.checkpoint,
            config=arguments.config,
            dataset_root=arguments.dataset_root,
            episode_indices=tuple(arguments.episode_indices),
            output_dir=arguments.output_dir,
            stride=arguments.stride,
            anchor_frame=arguments.anchor_frame,
            max_frames=arguments.max_frames,
            num_denoise_steps=arguments.num_denoise_steps,
            max_decode_steps=arguments.max_decode_steps,
            video_fps=arguments.video_fps,
            seed=arguments.seed,
        )
    ).run()
    return 0
