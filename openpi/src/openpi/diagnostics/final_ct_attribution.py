"""Offline end-to-end attribution of the final v3.1 writer representation.

The two always-recorded maps answer deliberately different questions:

* ``writer_surprise_sensitivity_top_siglip_output_slot`` is the L2 norm of the
  derivative of the *pre-write associative writer loss* with respect to each
  top-camera SigLIP output token.  It is a globally contextualized image-encoder
  output-slot sensitivity map through every Gemma block, not raw-pixel causality or
  the magnitude of the clipped fast-state update.
* ``final_ct_zero_read_output_slot_l2`` compares final ``c_t`` with the normal memory
  read to final ``c_t`` with a zero retrieval.  Its 256 entries index contextual
  memory-token *output slots*; the grid is slot-aligned and is not pixel attribution.

Optionally, selected raw frames receive a causal 14x14 model-image occlusion sweep.
All 256 counterfactuals branch from the identical pre-write MemoryState and are
discarded.  Only the unmodified baseline write advances the episode.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import dataclasses
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import cv2
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

from openpi.diagnostics import token_heatmap
from openpi.diagnostics import v31
from openpi.diagnostics import v31_pi0
from openpi.diagnostics import writer_contribution as _writer
from openpi.models import model as _model
from openpi.shared import nnx_utils

SCHEMA_VERSION = "openpi.v31.final_ct_attribution.v1"
CORE_API = "Pi0.final_ct_attribution_step.v1"
INTERVENTION_API = "Pi0.final_ct_intervention_step.v1"
RESTORE_DTYPE = "float32"
VIDEO_ENCODER = "opencv"

SIGLIP_OUTPUT_SLOT_METRIC = "writer_surprise_sensitivity_top_siglip_output_slot"
OUTPUT_SLOT_METRIC = "final_ct_zero_read_output_slot_l2"
PRIMARY_METRICS = (SIGLIP_OUTPUT_SLOT_METRIC, OUTPUT_SLOT_METRIC)
OCCLUSION_METRICS = (
    "occlusion_writer_loss_abs_delta",
    "occlusion_final_ct_rms_delta",
    "occlusion_fast_update_relative_l2",
    "occlusion_full_state_update_relative_l2",
    "occlusion_full_update_cosine",
)

_CORE_MAP_KEYS = {
    SIGLIP_OUTPUT_SLOT_METRIC: "writer_loss_top_patch_grad_norm",
    OUTPUT_SLOT_METRIC: "final_ct_zero_read_l2",
}
_CORE_REQUIRED_KEYS = (
    "final_ct",
    "writer_loss",
    "writer_loss_top_patch_grad_norm",
    "final_ct_zero_read_l2",
    "write_occurred",
)
_CORE_SCALAR_KEYS = (
    "writer_loss",
    "writer_loss_top_patch_grad_global_norm",
    "top_camera_patch_embedding_rms",
    "final_ct_rms",
    "zero_read_final_ct_rms",
    "retrieval_norm",
    "memory_gate_norm",
    "surprise",
    "grad_norm",
    "theta",
    "eta",
    "alpha",
    "top_camera_tokens",
)
_METRIC_SPACE = {
    SIGLIP_OUTPUT_SLOT_METRIC: (
        "top-camera SigLIP output token slot (spatially aligned row-major 16x16, but globally contextualized; "
        "not raw-image-patch causality)"
    ),
    OUTPUT_SLOT_METRIC: "final c_t output slot (row-major slot alignment; not pixel attribution)",
    "occlusion_writer_loss_abs_delta": "occluded input top-camera patch",
    "occlusion_final_ct_rms_delta": "occluded input top-camera patch",
    "occlusion_fast_update_relative_l2": "occluded input top-camera patch",
    "occlusion_full_state_update_relative_l2": "occluded input top-camera patch",
    "occlusion_full_update_cosine": "occluded input top-camera patch",
}
_METRIC_SEMANTICS = {
    SIGLIP_OUTPUT_SLOT_METRIC: (
        "L2 norm of the gradient of the reverse-mode primal's pre-write mean associative surprise with respect "
        "to one globally contextualized top-camera SigLIP output token; primal numerical floors versus the "
        "ordinary committed intervention writer are recorded per frame"
    ),
    OUTPUT_SLOT_METRIC: (
        "L2 difference between normal-read and zero-read final c_t at one contextual memory-token output slot"
    ),
    "occlusion_writer_loss_abs_delta": (
        "absolute change in pre-write writer surprise after blacking one 14x14 normalized model-image patch"
    ),
    "occlusion_final_ct_rms_delta": (
        "RMS change over every final-c_t slot and channel after blacking one 14x14 model-image patch"
    ),
    "occlusion_fast_update_relative_l2": (
        "L2 difference between occluded and baseline fast-weight update vectors, divided by baseline update L2"
    ),
    "occlusion_full_state_update_relative_l2": (
        "L2 difference between occluded and baseline full MemoryState update vectors (fast weights plus momentum), "
        "divided by baseline update L2"
    ),
    "occlusion_full_update_cosine": "cosine between occluded and baseline full MemoryState update vectors",
}


@dataclasses.dataclass(frozen=True)
class FinalCtRunOptions(_writer.RunOptions):
    """Replay settings; empty ``gradient_raw_frames`` means every cadence frame."""

    gradient_raw_frames: tuple[int, ...] = ()
    occlusion_raw_frames: tuple[int, ...] = ()
    occlusion_batch_size: int = 1
    metrics: tuple[str, ...] = PRIMARY_METRICS
    # The absolute view is primary. History is explicitly labelled as a relative,
    # episode-fitted visualization that can amplify tiny residual variation.
    heatmap_scale_modes: tuple[str, ...] = ("video", "per_token_history")
    exclude_letterbox_padding: bool = True

    def __post_init__(self) -> None:
        # _writer validates the shared source, renderer, and cadence settings. It does
        # not know our metric names, so temporarily present its supported defaults.
        requested_metrics = tuple(self.metrics)
        object.__setattr__(self, "metrics", _writer.DEFAULT_METRICS)
        super().__post_init__()
        object.__setattr__(self, "metrics", requested_metrics)
        unknown = set(requested_metrics) - {*PRIMARY_METRICS, *OCCLUSION_METRICS}
        if not requested_metrics or unknown or len(set(requested_metrics)) != len(requested_metrics):
            raise ValueError(
                "metrics must be a unique non-empty selection from "
                f"{(*PRIMARY_METRICS, *OCCLUSION_METRICS)}; unknown={sorted(unknown)}"
            )
        for name, frames in (
            ("gradient_raw_frames", self.gradient_raw_frames),
            ("occlusion_raw_frames", self.occlusion_raw_frames),
        ):
            values = tuple(frames)
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
                raise ValueError(f"{name} must contain non-negative integers")
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be strictly increasing and unique")
            object.__setattr__(self, name, values)
        if self.occlusion_batch_size <= 0:
            raise ValueError("occlusion_batch_size must be positive")
        if self.occlusion_raw_frames:
            # A default occlusion request should produce its causal maps without the
            # caller having to repeat a long metric list. The signed cosine remains
            # tabular/NPZ-only because the shared heatmap renderer is non-negative.
            object.__setattr__(
                self,
                "metrics",
                tuple(dict.fromkeys((*requested_metrics, *OCCLUSION_METRICS[:-1]))),
            )


@dataclasses.dataclass
class _FinalCtFrame:
    raw_frame: int
    policy_step: int
    model_image_rgb: np.ndarray
    raw_image_rgb: np.ndarray
    maps: dict[str, np.ndarray]
    scalar: dict[str, float]
    phase: str = ""
    occlusion: dict[str, np.ndarray] | None = None


def _finite_map(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (token_heatmap.TOKEN_COUNT,):
        raise ValueError(f"{name} must have shape (256,), got {array.shape}")
    if array.dtype.kind != "f" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite floating-point values")
    if np.any(array < 0):
        raise ValueError(f"{name} must be non-negative")
    return np.asarray(array, dtype=np.float32)


def _finite_scalar(value: Any, *, name: str) -> float:
    array = np.asarray(value)
    if array.shape not in ((), (1,)) or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be one numeric scalar, got {array.shape}/{array.dtype}")
    result = float(array.reshape(-1)[0])
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _tree_repeat_batch(tree: Any, repeats: int) -> Any:
    """Repeat a batch-one pytree without flattening or aliasing its semantic fields."""

    if repeats <= 0:
        raise ValueError("repeats must be positive")

    def repeat(leaf):
        array = jnp.asarray(leaf)
        if array.ndim == 0 or array.shape[0] != 1:
            raise ValueError(f"counterfactual tree leaf must have leading batch size 1, got {array.shape}")
        return jnp.repeat(array, repeats, axis=0)

    return jax.tree.map(repeat, tree)


def _tree_exactly_equal(left: Any, right: Any) -> bool:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    return len(left_leaves) == len(right_leaves) and all(
        np.array_equal(np.asarray(left_leaf), np.asarray(right_leaf))
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True)
    )


def _tree_update_geometry(branch_state: Any, baseline_state: Any, pre_state: Any) -> dict[str, np.ndarray]:
    """Compare branch and baseline state-update vectors without materializing them."""

    branch_fast = branch_state.fast_weights
    branch_momentum = branch_state.momentum
    base_fast = baseline_state.fast_weights
    base_momentum = baseline_state.momentum
    pre_fast = pre_state.fast_weights
    pre_momentum = pre_state.momentum
    batch = next(iter(branch_fast.values())).shape[0]
    for name, tree in (
        ("branch fast weights", branch_fast),
        ("branch momentum", branch_momentum),
        ("baseline fast weights", base_fast),
        ("baseline momentum", base_momentum),
        ("pre-state fast weights", pre_fast),
        ("pre-state momentum", pre_momentum),
    ):
        if any(np.asarray(leaf).shape[0] != batch for leaf in tree.values()):
            raise ValueError(f"{name} does not use the shared counterfactual batch size {batch}")

    def accum(branch_tree, base_tree, pre_tree):
        branch_delta_norm2 = np.zeros(batch, dtype=np.float64)
        difference_norm2 = np.zeros(batch, dtype=np.float64)
        dot = np.zeros(batch, dtype=np.float64)
        baseline_norm2 = np.zeros(batch, dtype=np.float64)
        for key in sorted(base_tree):
            branch_delta = np.asarray(branch_tree[key], dtype=np.float64) - np.asarray(pre_tree[key], dtype=np.float64)
            baseline_delta = np.asarray(base_tree[key], dtype=np.float64) - np.asarray(pre_tree[key], dtype=np.float64)
            axes = tuple(range(1, branch_delta.ndim))
            branch_delta_norm2 += np.sum(np.square(branch_delta), axis=axes)
            difference_norm2 += np.sum(np.square(branch_delta - baseline_delta), axis=axes)
            dot += np.sum(branch_delta * baseline_delta, axis=axes)
            baseline_norm2 += np.sum(np.square(baseline_delta), axis=axes)
        return branch_delta_norm2, difference_norm2, dot, baseline_norm2

    fast_branch2, fast_diff2, fast_dot, fast_base2 = accum(branch_fast, base_fast, pre_fast)
    mom_branch2, mom_diff2, mom_dot, mom_base2 = accum(branch_momentum, base_momentum, pre_momentum)
    full_branch2 = fast_branch2 + mom_branch2
    full_diff2 = fast_diff2 + mom_diff2
    full_dot = fast_dot + mom_dot
    full_base2 = fast_base2 + mom_base2
    epsilon = np.finfo(np.float64).eps
    return {
        "fast_relative_l2": np.sqrt(fast_diff2) / np.maximum(np.sqrt(fast_base2), epsilon),
        "full_relative_l2": np.sqrt(full_diff2) / np.maximum(np.sqrt(full_base2), epsilon),
        "full_cosine": full_dot / np.maximum(np.sqrt(full_branch2 * full_base2), epsilon),
        "baseline_fast_update_l2": np.sqrt(fast_base2),
        "baseline_full_update_l2": np.sqrt(full_base2),
    }


def _git_head(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _source_hashes(repo_root: Path) -> dict[str, Any]:
    """Hash bytes actually executed, including dirty tracked and new untracked files."""

    relative_paths = (
        "openpi/src/openpi/diagnostics/final_ct_attribution.py",
        "openpi/scripts/v31_final_ct_attribution.py",
        "openpi/src/openpi/models/pi0.py",
        "openpi/src/openpi/models/memory.py",
        "openpi/src/openpi/diagnostics/writer_contribution.py",
        "openpi/src/openpi/diagnostics/token_heatmap.py",
    )
    files = {}
    aggregate = hashlib.sha256()
    for relative in relative_paths:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required diagnostic source is missing: {path}")
        digest = v31.sha256_file(path)
        files[relative] = digest
        aggregate.update(relative.encode())
        aggregate.update(b"\0")
        aggregate.update(digest.encode())
        aggregate.update(b"\0")
    return {"algorithm": "sha256", "aggregate": aggregate.hexdigest(), "files": files}


def _runtime_environment() -> dict[str, Any]:
    devices = [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
            "process_index": int(device.process_index),
        }
        for device in jax.devices()
    ]
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "jax_backend": jax.default_backend(),
        "jax_process_index": jax.process_index(),
        "jax_process_count": jax.process_count(),
        "devices": devices,
    }


def _validate_lerobot_row_identity(
    *,
    raw_frame: Any,
    previous_raw_frame: int,
    episode_index: Any | None,
    expected_episode_index: int | None,
) -> int:
    """Require canonical contiguous row order and, when recorded, episode identity."""

    if isinstance(raw_frame, bool) or not isinstance(raw_frame, int) or raw_frame < 0:
        raise ValueError(f"invalid LeRobot frame_index: {raw_frame!r}")
    expected_raw_frame = previous_raw_frame + 1
    if raw_frame != expected_raw_frame:
        raise ValueError(
            "LeRobot frame_index must be contiguous, strictly increasing, and unique from zero; "
            f"expected {expected_raw_frame}, got {raw_frame}"
        )
    if episode_index is not None:
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ValueError(f"invalid LeRobot episode_index: {episode_index!r}")
        if expected_episode_index is None:
            raise ValueError("cannot validate parquet episode_index against a noncanonical source episode ID")
        if episode_index != expected_episode_index:
            raise ValueError(
                f"LeRobot row episode_index {episode_index} does not match selected episode {expected_episode_index}"
            )
    return raw_frame


class FinalCtAttributionRunner(_writer.WriterContributionRunner):
    """Checkpoint-backed replay that preserves the deployed recurrent write order."""

    def __init__(self, options: FinalCtRunOptions):
        super().__init__(options)
        self.options: FinalCtRunOptions = options
        self._attribution_step = nnx_utils.module_jit(
            self.model.final_ct_attribution_step, static_argnames=("allow_write",)
        )
        self._intervention_step = nnx_utils.module_jit(
            self.model.final_ct_intervention_step, static_argnames=("allow_write",)
        )
        self.gemma_depth = int(self.model.PaliGemma.llm.module.configs[0].depth)
        if self.gemma_depth != 18:
            raise ValueError(f"v3.1 final-c_t diagnostics require the expected 18-block Gemma, got {self.gemma_depth}")
        if int(self.model.memory_layer) != 8:
            raise ValueError(
                f"v3.1 final-c_t diagnostics require memory_layer=8 for h_t, got {self.model.memory_layer}"
            )
        # Use one no-gradient primal for every committed recurrent write. Mixing this
        # with the reverse-mode attribution executable creates a real BF16/compiler
        # numerical floor that makes exact no-op padding occlusions look nonzero.
        self._advance_step = self._intervention_step
        self._requested = set(options.gradient_raw_frames) | set(options.occlusion_raw_frames)
        self._occlusion_requested = set(options.occlusion_raw_frames)

    def _should_measure(self, raw_frame: int) -> bool:
        return not self.options.gradient_raw_frames or raw_frame in self._requested

    def _occlusion_sweep(
        self,
        observation: _model.Observation,
        pre_state: Any,
        baseline_state: Any,
        baseline_aux: Mapping[str, Any],
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        top = np.asarray(observation.images["base_0_rgb"])
        if top.shape != (1, token_heatmap.MODEL_IMAGE_SIZE, token_heatmap.MODEL_IMAGE_SIZE, 3):
            raise ValueError(f"expected normalized batch-one 224x224 base_0_rgb, got {top.shape}")
        if top.dtype.kind != "f" or not np.all(np.isfinite(top)):
            raise ValueError("normalized base_0_rgb must contain finite floating-point values")

        recurrent_baseline_ct = np.asarray(baseline_aux["final_ct"][0], dtype=np.float32)
        recurrent_baseline_loss = _finite_scalar(baseline_aux["writer_loss"], name="baseline writer_loss")
        if recurrent_baseline_ct.shape[:1] != (token_heatmap.TOKEN_COUNT,):
            raise ValueError(f"baseline final_ct must have 256 output slots, got {recurrent_baseline_ct.shape}")

        loss_delta = np.empty(token_heatmap.TOKEN_COUNT, dtype=np.float32)
        ct_delta = np.empty(token_heatmap.TOKEN_COUNT, dtype=np.float32)
        fast_relative = np.empty(token_heatmap.TOKEN_COUNT, dtype=np.float32)
        full_relative = np.empty(token_heatmap.TOKEN_COUNT, dtype=np.float32)
        cosine = np.empty(token_heatmap.TOKEN_COUNT, dtype=np.float32)
        baseline_norms: dict[str, float] | None = None
        batch_baseline_ct_floor = 0.0
        batch_baseline_loss_floor = 0.0
        patch_size = token_heatmap.PATCH_SIZE

        for start in range(0, token_heatmap.TOKEN_COUNT, self.options.occlusion_batch_size):
            indices = np.arange(start, min(start + self.options.occlusion_batch_size, token_heatmap.TOKEN_COUNT))
            images = np.repeat(top, len(indices), axis=0)
            for branch, patch_index in enumerate(indices):
                row, column = divmod(int(patch_index), token_heatmap.TOKEN_GRID_SIZE)
                images[
                    branch,
                    row * patch_size : (row + 1) * patch_size,
                    column * patch_size : (column + 1) * patch_size,
                    :,
                ] = -1.0
            baseline_observation = _tree_repeat_batch(observation, len(indices))
            branch_observation = baseline_observation.replace(
                images={**baseline_observation.images, "base_0_rgb": jnp.asarray(images)}
            )
            branch_pre = _tree_repeat_batch(pre_state, len(indices))
            # Match both API and batch shape. In mixed precision, comparing against
            # the attribution executable or a differently batched executable creates
            # a nonzero compiler floor even when an occluded padding patch is already
            # exactly -1 and therefore changes no input byte.
            if len(indices) == 1:
                # Default H100-safe mode: the recurrent baseline was already
                # produced by this exact batch-one intervention executable.
                batch_baseline_state, batch_baseline_aux = baseline_state, baseline_aux
            else:
                batch_baseline_state, batch_baseline_aux = self._intervention_step(
                    baseline_observation, branch_pre, allow_write=True
                )
            # Functional candidate writes are returned for comparison and immediately
            # discarded. They never become the recurrent episode state.
            branch_state, branch_aux = self._intervention_step(branch_observation, branch_pre, allow_write=True)
            batch_baseline_state, batch_baseline_aux, branch_state, branch_aux = jax.device_get(
                (batch_baseline_state, batch_baseline_aux, branch_state, branch_aux)
            )
            batch_baseline_loss = np.asarray(batch_baseline_aux["writer_loss"], dtype=np.float64)
            batch_baseline_ct = np.asarray(batch_baseline_aux["final_ct"], dtype=np.float32)
            branch_loss = np.asarray(branch_aux["writer_loss"], dtype=np.float64)
            branch_ct = np.asarray(branch_aux["final_ct"], dtype=np.float32)
            expected_ct_shape = (
                len(indices),
                *recurrent_baseline_ct.shape,
            )
            if (
                branch_loss.shape != (len(indices),)
                or batch_baseline_loss.shape != (len(indices),)
                or branch_ct.shape != expected_ct_shape
                or batch_baseline_ct.shape != expected_ct_shape
            ):
                raise ValueError(
                    "invalid matched intervention shapes: "
                    f"baseline_loss={batch_baseline_loss.shape}, branch_loss={branch_loss.shape}, "
                    f"baseline_ct={batch_baseline_ct.shape}, branch_ct={branch_ct.shape}"
                )
            geometry = _tree_update_geometry(branch_state, batch_baseline_state, branch_pre)
            if baseline_norms is None:
                baseline_norms = {
                    "baseline_fast_update_l2": float(geometry["baseline_fast_update_l2"][0]),
                    "baseline_full_update_l2": float(geometry["baseline_full_update_l2"][0]),
                }
            batch_baseline_ct_floor = max(
                batch_baseline_ct_floor,
                float(
                    np.max(
                        np.sqrt(
                            np.mean(
                                np.square(batch_baseline_ct - recurrent_baseline_ct[None]),
                                axis=(1, 2),
                                dtype=np.float64,
                            )
                        )
                    )
                ),
            )
            batch_baseline_loss_floor = max(
                batch_baseline_loss_floor,
                float(np.max(np.abs(batch_baseline_loss - recurrent_baseline_loss))),
            )
            loss_delta[indices] = np.abs(branch_loss - batch_baseline_loss)
            ct_delta[indices] = np.sqrt(
                np.mean(np.square(branch_ct - batch_baseline_ct), axis=(1, 2), dtype=np.float64)
            )
            fast_relative[indices] = np.asarray(geometry["fast_relative_l2"], dtype=np.float32)
            full_relative[indices] = np.asarray(geometry["full_relative_l2"], dtype=np.float32)
            cosine[indices] = np.clip(np.asarray(geometry["full_cosine"], dtype=np.float32), -1.0, 1.0)

        assert baseline_norms is not None
        baseline_norms.update(
            {
                "occlusion_batch_baseline_final_ct_rms_floor": batch_baseline_ct_floor,
                "occlusion_batch_baseline_writer_loss_abs_floor": batch_baseline_loss_floor,
            }
        )
        maps = {
            "occlusion_writer_loss_abs_delta": loss_delta,
            "occlusion_final_ct_rms_delta": ct_delta,
            "occlusion_fast_update_relative_l2": fast_relative,
            "occlusion_full_state_update_relative_l2": full_relative,
            # Renderer requires non-negative values. Persist the signed cosine in NPZ/CSV,
            # but do not offer it as a heatmap metric.
            "occlusion_full_update_cosine": cosine,
        }
        for name in OCCLUSION_METRICS[:-1]:
            _finite_map(maps[name], name=name)
        if not np.all(np.isfinite(cosine)):
            raise ValueError("occlusion_full_update_cosine contains non-finite values")
        return maps, baseline_norms

    def _evaluate_attribution_frame(
        self,
        source: _writer.EpisodeSource,
        *,
        raw_frame: int,
        policy_step: int,
        top_rgb: np.ndarray,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        robot_state: np.ndarray,
        memory_state: Any,
        phase: str = "",
    ) -> tuple[Any, _FinalCtFrame | None]:
        observation, model_image = self._transform_observation(
            source, raw_frame, top_rgb, left_rgb, right_rgb, robot_state
        )
        if not self._should_measure(raw_frame):
            next_state, _ = self._advance_step(observation, memory_state, allow_write=True)
            return jax.device_get(next_state), None

        pre_state = memory_state
        attribution_state, attribution_aux = self._attribution_step(observation, pre_state, allow_write=False)
        baseline_state, baseline_aux = self._intervention_step(observation, pre_state, allow_write=True)
        attribution_state, attribution_aux, baseline_state, baseline_aux, pre_state_host = jax.device_get(
            (attribution_state, attribution_aux, baseline_state, baseline_aux, pre_state)
        )
        if not _tree_exactly_equal(attribution_state, pre_state_host):
            raise RuntimeError("read-only attribution changed the complete pre-write MemoryState")
        missing = set(_CORE_REQUIRED_KEYS) - set(attribution_aux)
        if missing:
            raise KeyError(f"{CORE_API} omitted required aux keys: {sorted(missing)}")
        baseline_required = {"final_ct", "writer_loss", "surprise", "write_occurred"}
        missing_baseline = baseline_required - set(baseline_aux)
        if missing_baseline:
            raise KeyError(f"{INTERVENTION_API} omitted required aux keys: {sorted(missing_baseline)}")
        if bool(np.asarray(attribution_aux["write_occurred"])[0]):
            raise RuntimeError("read-only attribution unexpectedly reported a committed write")
        if not bool(np.asarray(baseline_aux["write_occurred"])[0]):
            raise RuntimeError("intervention baseline did not commit its one normal recurrent write")

        attribution_ct = np.asarray(attribution_aux["final_ct"])
        baseline_ct = np.asarray(baseline_aux["final_ct"])
        expected_prefix = (1, token_heatmap.TOKEN_COUNT)
        if (
            attribution_ct.ndim != 3
            or attribution_ct.shape[:2] != expected_prefix
            or baseline_ct.shape != attribution_ct.shape
        ):
            raise ValueError(
                f"attribution/intervention final_ct must share shape [1,256,D], got "
                f"{attribution_ct.shape}/{baseline_ct.shape}"
            )
        if (
            attribution_ct.dtype.kind != "f"
            or baseline_ct.dtype.kind != "f"
            or not np.all(np.isfinite(attribution_ct))
            or not np.all(np.isfinite(baseline_ct))
        ):
            raise ValueError("attribution/intervention final_ct must contain finite floating-point values")
        maps = {
            name: _finite_map(np.asarray(attribution_aux[key])[0], name=name) for name, key in _CORE_MAP_KEYS.items()
        }
        # Overlapping scalars come from the ordinary no-gradient intervention
        # executable, because that exact primal produces the recurrent state and the
        # occlusion baseline. Only reverse-mode-specific quantities come from attr.
        scalar = {
            name: _finite_scalar(baseline_aux[name], name=name) for name in _CORE_SCALAR_KEYS if name in baseline_aux
        }
        for name in (
            "writer_loss_top_patch_grad_global_norm",
            "top_camera_patch_embedding_rms",
            "zero_read_final_ct_rms",
        ):
            if name in attribution_aux:
                scalar[name] = _finite_scalar(attribution_aux[name], name=name)
        attribution_loss = _finite_scalar(attribution_aux["writer_loss"], name="attribution writer_loss")
        attribution_surprise = _finite_scalar(attribution_aux["surprise"], name="attribution surprise")
        if "writer_loss" not in scalar or "surprise" not in scalar:
            raise KeyError("intervention baseline aux must include writer_loss and surprise")
        if not math.isclose(scalar["writer_loss"], scalar["surprise"], rel_tol=2e-5, abs_tol=2e-5):
            raise RuntimeError("baseline writer_loss does not equal the actual pre-write associative surprise")
        if not math.isclose(attribution_loss, attribution_surprise, rel_tol=2e-5, abs_tol=2e-5):
            raise RuntimeError("attribution writer_loss does not equal its pre-write associative surprise")
        scalar.update(
            {
                "primal_final_ct_rms_floor": float(
                    np.sqrt(np.mean(np.square(attribution_ct - baseline_ct), dtype=np.float64))
                ),
                "primal_writer_loss_abs_floor": abs(attribution_loss - scalar["writer_loss"]),
            }
        )
        occlusion = None
        if raw_frame in self._occlusion_requested:
            occlusion, occlusion_scalar = self._occlusion_sweep(
                observation, pre_state_host, baseline_state, baseline_aux
            )
            scalar.update(occlusion_scalar)
        return baseline_state, _FinalCtFrame(
            raw_frame=raw_frame,
            policy_step=policy_step,
            model_image_rgb=model_image,
            raw_image_rgb=np.array(top_rgb, copy=True),
            maps=maps,
            scalar=scalar,
            phase=phase,
            occlusion=occlusion,
        )

    @staticmethod
    def _validate_frame_order(frames: Sequence[_FinalCtFrame]) -> None:
        if not frames:
            raise ValueError("episode yielded no selected attribution frames")
        raw = [frame.raw_frame for frame in frames]
        steps = [frame.policy_step for frame in frames]
        if any(right <= left for left, right in itertools.pairwise(raw)):
            raise ValueError("selected raw_frame values must be strictly increasing and unique")
        if any(right <= left for left, right in itertools.pairwise(steps)):
            raise ValueError("selected policy_step values must be strictly increasing and unique")

    def _measure_raw_episode(self, source: _writer.EpisodeSource) -> tuple[list[_FinalCtFrame], int, int]:
        files = _writer._required_episode_files(source)  # noqa: SLF001
        state = _writer._load_states(files, source.episode_id)  # noqa: SLF001
        total = min(
            len(state),
            *(_writer._video_frame_count(files[name]) for name in ("top", "left", "right")),  # noqa: SLF001
        )
        captures = {name: cv2.VideoCapture(str(files[name])) for name in ("top", "left", "right")}
        if not all(capture.isOpened() for capture in captures.values()):
            for capture in captures.values():
                capture.release()
            raise ValueError(f"could not open all videos for episode {source.episode_id!r}")
        memory_state = self.model.memory.init_state(1)
        measured: list[_FinalCtFrame] = []
        policy_step = 0
        try:
            for raw_frame in range(total):
                decoded = {name: capture.read() for name, capture in captures.items()}
                if not all(ok for ok, _ in decoded.values()):
                    raise ValueError(f"video decode ended early at frame {raw_frame}")
                if raw_frame % self.stride:
                    continue
                rgb = {name: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for name, (_, frame) in decoded.items()}
                memory_state, frame = self._evaluate_attribution_frame(
                    source,
                    raw_frame=raw_frame,
                    policy_step=policy_step,
                    top_rgb=rgb["top"],
                    left_rgb=rgb["left"],
                    right_rgb=rgb["right"],
                    robot_state=state[raw_frame],
                    memory_state=memory_state,
                )
                if frame is not None:
                    measured.append(frame)
                    print(
                        f"[{source.episode_id}] raw={raw_frame} step={policy_step} "
                        f"writer_loss={frame.scalar['writer_loss']:.5g} "
                        f"occlusion={frame.occlusion is not None}",
                        flush=True,
                    )
                policy_step += 1
                if self.options.max_frames is not None and policy_step >= self.options.max_frames:
                    break
        finally:
            for capture in captures.values():
                capture.release()
        self._validate_requested_frames(source, measured)
        self._validate_frame_order(measured)
        return measured, total, policy_step

    def _measure_lerobot_episode(self, source: _writer.EpisodeSource) -> tuple[list[_FinalCtFrame], int, int]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyarrow is required for inline-parquet replay") from exc
        parquet = pq.ParquetFile(source.path)
        schema_names = set(parquet.schema_arrow.names)
        columns = ["image", "left_wrist_image", "right_wrist_image", "state", "frame_index", "task_index"]
        missing = set(columns) - schema_names
        if missing:
            raise ValueError(f"LeRobot episode lacks columns {sorted(missing)}")
        if "episode_index" in schema_names:
            columns.append("episode_index")
        total = parquet.metadata.num_rows
        memory_state = self.model.memory.init_state(1)
        measured: list[_FinalCtFrame] = []
        policy_step = 0
        previous_raw_frame = -1
        try:
            expected_episode_index = int(source.episode_id.removeprefix("episode_"))
        except ValueError:
            expected_episode_index = None
        stop = False
        for batch in parquet.iter_batches(batch_size=100, columns=columns):
            for row in batch.to_pylist():
                raw_frame = row["frame_index"]
                previous_raw_frame = _validate_lerobot_row_identity(
                    raw_frame=raw_frame,
                    previous_raw_frame=previous_raw_frame,
                    episode_index=row.get("episode_index"),
                    expected_episode_index=expected_episode_index,
                )
                if raw_frame % self.stride:
                    continue
                robot_state = np.asarray(row["state"], dtype=np.float32)
                if robot_state.shape != (14,) or not np.all(np.isfinite(robot_state)):
                    raise ValueError(f"invalid state at raw frame {raw_frame}")
                images = {
                    name: self._decode_inline_image(row[field], field=field, raw_frame=raw_frame)
                    for name, field in (
                        ("top", "image"),
                        ("left", "left_wrist_image"),
                        ("right", "right_wrist_image"),
                    )
                }
                task_index = row["task_index"]
                phase = (
                    source.task_names[task_index]
                    if isinstance(task_index, int) and 0 <= task_index < len(source.task_names)
                    else ""
                )
                memory_state, frame = self._evaluate_attribution_frame(
                    source,
                    raw_frame=raw_frame,
                    policy_step=policy_step,
                    top_rgb=images["top"],
                    left_rgb=images["left"],
                    right_rgb=images["right"],
                    robot_state=robot_state,
                    memory_state=memory_state,
                    phase=phase,
                )
                if frame is not None:
                    measured.append(frame)
                    print(
                        f"[{source.episode_id}] raw={raw_frame} step={policy_step} task={phase!r} "
                        f"writer_loss={frame.scalar['writer_loss']:.5g} occlusion={frame.occlusion is not None}",
                        flush=True,
                    )
                policy_step += 1
                if self.options.max_frames is not None and policy_step >= self.options.max_frames:
                    stop = True
                    break
            if stop:
                break
        self._validate_requested_frames(source, measured)
        self._validate_frame_order(measured)
        return measured, total, policy_step

    def _validate_requested_frames(self, source: _writer.EpisodeSource, measured: Sequence[_FinalCtFrame]) -> None:
        if not self._requested:
            return
        found = {frame.raw_frame for frame in measured}
        missing = self._requested - found
        if missing:
            raise ValueError(
                f"episode {source.episode_id!r} did not encounter requested raw frames {sorted(missing)} "
                f"at effective stride {self.stride} (or before max_frames)"
            )

    def _measure_episode(self, source: _writer.EpisodeSource) -> tuple[list[_FinalCtFrame], int, int]:
        if source.source_format == "raw_yam":
            return self._measure_raw_episode(source)
        if source.source_format == "lerobot_inline_parquet":
            return self._measure_lerobot_episode(source)
        raise AssertionError(f"unhandled episode source format {source.source_format!r}")

    def _video_frames(self, source: _writer.EpisodeSource, measured: Sequence[_FinalCtFrame], metric: str):
        selected = [
            frame for frame in measured if metric in frame.maps or (frame.occlusion and metric in frame.occlusion)
        ]
        result = []
        for frame in selected:
            values = frame.maps.get(metric) if metric in frame.maps else frame.occlusion[metric]  # type: ignore[index]
            result.append(
                token_heatmap.TokenMetricFrame(
                    raw_frame=frame.raw_frame,
                    policy_step=frame.policy_step,
                    model_image_rgb=frame.model_image_rgb,
                    raw_image_rgb=frame.raw_image_rgb,
                    token_values=values,
                    write_count=frame.policy_step + 1,
                    phase=frame.phase,
                    timestamp_s=frame.raw_frame / source.control_hz,
                    metadata={
                        "episode_id": source.episode_id,
                        "metric_space": _METRIC_SPACE[metric],
                        "absolute_metric_is_primary": True,
                    },
                )
            )
        return result

    def _save_episode(
        self,
        source: _writer.EpisodeSource,
        measured: Sequence[_FinalCtFrame],
        total_raw_frames: int,
        episode_dir: Path,
        replayed_write_frames: int | None = None,
    ) -> dict[str, Any]:
        self._validate_frame_order(measured)
        episode_dir.mkdir(parents=True, exist_ok=False)
        raw_frame = np.asarray([frame.raw_frame for frame in measured], dtype=np.int64)
        policy_step = np.asarray([frame.policy_step for frame in measured], dtype=np.int64)
        minimum_replayed_writes = int(policy_step[-1]) + 1
        if replayed_write_frames is None:
            replayed_write_frames = minimum_replayed_writes
        if replayed_write_frames < minimum_replayed_writes:
            raise ValueError(
                f"replayed_write_frames {replayed_write_frames} cannot precede the last selected policy step "
                f"{int(policy_step[-1])}"
            )
        scalar_names = tuple(sorted({name for frame in measured for name in frame.scalar}))
        common_scalar_names = tuple(sorted(set.intersection(*(set(frame.scalar) for frame in measured))))
        arrays: dict[str, np.ndarray] = {
            "raw_frame": raw_frame,
            "policy_step": policy_step,
            "task": np.asarray([frame.phase for frame in measured], dtype=np.str_),
            **{name: np.stack([frame.maps[name] for frame in measured]) for name in PRIMARY_METRICS},
        }
        for name in common_scalar_names:
            values = np.asarray([frame.scalar[name] for frame in measured], dtype=np.float32)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"scalar artifact {name} contains non-finite values")
            arrays[name] = values
        occlusion_frames = [frame for frame in measured if frame.occlusion is not None]
        arrays["occlusion_raw_frame"] = np.asarray([frame.raw_frame for frame in occlusion_frames], dtype=np.int64)
        arrays["occlusion_policy_step"] = np.asarray([frame.policy_step for frame in occlusion_frames], dtype=np.int64)
        for name in OCCLUSION_METRICS:
            arrays[name] = (
                np.stack([frame.occlusion[name] for frame in occlusion_frames])  # type: ignore[index]
                if occlusion_frames
                else np.empty((0, token_heatmap.TOKEN_COUNT), dtype=np.float32)
            )
        optional_scalar_names = tuple(sorted(set(scalar_names) - set(common_scalar_names)))
        for name in optional_scalar_names:
            if any(name not in frame.scalar for frame in occlusion_frames):
                raise ValueError(f"optional scalar {name!r} is not present on every occlusion frame")
            values = np.asarray([frame.scalar[name] for frame in occlusion_frames], dtype=np.float32)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"occlusion scalar artifact {name} contains non-finite values")
            arrays[name] = values
        np.savez_compressed(episode_dir / "contributions.npz", **arrays)

        with (episode_dir / "frame_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            stats_names = (
                "mean",
                "std",
                "coefficient_of_variation",
                "normalized_entropy",
                "effective_token_count",
                "top_10pct_mass_fraction",
            )
            fieldnames = [
                "raw_frame",
                "policy_step",
                "task",
                "has_occlusion",
                *scalar_names,
                *(f"{name}_{stat}" for name in (*PRIMARY_METRICS, *OCCLUSION_METRICS[:-1]) for stat in stats_names),
                "occlusion_full_update_cosine_mean",
                "occlusion_full_update_cosine_min",
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for frame in measured:
                row: dict[str, Any] = {
                    "raw_frame": frame.raw_frame,
                    "policy_step": frame.policy_step,
                    "task": frame.phase,
                    "has_occlusion": frame.occlusion is not None,
                    **frame.scalar,
                }
                available = dict(frame.maps)
                if frame.occlusion:
                    available.update(frame.occlusion)
                for name in (*PRIMARY_METRICS, *OCCLUSION_METRICS[:-1]):
                    if name in available:
                        for stat, value in token_heatmap.metric_statistics(available[name]).items():
                            if stat in stats_names:
                                row[f"{name}_{stat}"] = value
                if frame.occlusion:
                    cosine = frame.occlusion["occlusion_full_update_cosine"]
                    row["occlusion_full_update_cosine_mean"] = float(np.mean(cosine))
                    row["occlusion_full_update_cosine_min"] = float(np.min(cosine))
                writer.writerow(row)

        videos: dict[str, str] = {}
        if self.options.render_video:
            fps = self.options.video_fps or source.control_hz / self.stride
            for metric in self.options.metrics:
                if metric == "occlusion_full_update_cosine":
                    continue
                frames = self._video_frames(source, measured, metric)
                if not frames:
                    continue
                for scale_mode in self.options.heatmap_scale_modes:
                    suffix = "absolute" if scale_mode == "video" else "relative_history_AMPLIFIES_SMALL_VARIATION"
                    video_key = f"{metric}__{suffix}"
                    details = token_heatmap.export_heatmap_video(
                        frames,
                        episode_dir / "heatmaps" / video_key,
                        metric_name=f"{metric} [{_METRIC_SPACE[metric]}]",
                        fps=fps,
                        normalization=token_heatmap.NormalizationSpec(
                            lower_percentile=0.0,
                            upper_percentile=self.options.upper_percentile,
                            anchor_zero=True,
                            exclude_letterbox_padding=self.options.exclude_letterbox_padding,
                        ),
                        alpha=self.options.alpha,
                        colormap=self.options.colormap if scale_mode == "video" else "coolwarm",
                        video_encoder=VIDEO_ENCODER,
                        scale_mode=scale_mode,
                        zscore_range=self.options.zscore_range,
                    )
                    heatmap_dir = episode_dir / "heatmaps" / video_key
                    heatmap_manifest_path = heatmap_dir / "manifest.json"
                    heatmap_manifest = dict(_writer._strict_json(heatmap_manifest_path))  # noqa: SLF001
                    heatmap_manifest.update(
                        {
                            "parent_schema_version": SCHEMA_VERSION,
                            "metric_semantics": _METRIC_SEMANTICS[metric],
                            "metric_space": _METRIC_SPACE[metric],
                            "absolute_metric_is_primary": True,
                            "relative_view_warning": (
                                None
                                if scale_mode == "video"
                                else "Episode-fitted relative colors erase absolute magnitude and may amplify tiny residual variation."
                            ),
                        }
                    )
                    _writer._write_json(heatmap_manifest_path, heatmap_manifest)  # noqa: SLF001
                    videos[video_key] = str(Path("heatmaps") / video_key / str(details["video"]))

        aggregate = {
            name: {
                "metric_space": _METRIC_SPACE[name],
                "mean": float(np.mean(np.stack([frame.maps[name] for frame in measured]))),
                "max": float(np.max(np.stack([frame.maps[name] for frame in measured]))),
            }
            for name in PRIMARY_METRICS
        }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": source.episode_id,
            "path": str(source.path),
            "source_format": source.source_format,
            "ground_truth_side": source.ground_truth_side,
            "control_hz": source.control_hz,
            "total_raw_frames": total_raw_frames,
            "sampled_write_frames": replayed_write_frames,
            "replayed_write_frames": replayed_write_frames,
            "attribution_frames": len(measured),
            "occlusion_frames": len(occlusion_frames),
            "stride": self.stride,
            "first_raw_frame": int(raw_frame[0]),
            "last_raw_frame": int(raw_frame[-1]),
            "aggregate": aggregate,
            "videos": videos,
        }
        _writer._write_json(episode_dir / "summary.json", summary)  # noqa: SLF001
        return summary

    def run(self) -> dict[str, Any]:
        start = time.monotonic()
        self.options.output_dir.mkdir(parents=True, exist_ok=False)
        episodes_dir = self.options.output_dir / "episodes"
        episodes_dir.mkdir()
        repo_root = Path(__file__).parents[4]
        initial_state = self.model.memory.init_state(1)
        inner_dtypes = sorted({str(np.asarray(leaf).dtype) for leaf in jax.tree.leaves(initial_state)})
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git_head": _git_head(repo_root),
            "code_revision": v31.current_code_revision(repo_root),
            "executed_source_content": _source_hashes(repo_root),
            "runtime_environment": _runtime_environment(),
            "checkpoint_path": str(self.options.checkpoint),
            "checkpoint_params_hash": v31_pi0._path_hash(self.options.checkpoint / "params"),  # noqa: SLF001
            "checkpoint_static_config_provenance_verified": self.provenance.verified,
            "checkpoint_metadata_path": self.provenance.metadata_path,
            "checkpoint_config_name": self.provenance.config_name,
            "checkpoint_memory_write_source": self.provenance.memory_write_source,
            "config": self.options.config,
            "restore_dtype": RESTORE_DTYPE,
            "inner_memory_state_dtypes": inner_dtypes,
            "memory_write_source": self.train_config.model.memory_write_source,
            "memory_layer": self.train_config.model.memory_layer,
            "gemma_depth": self.gemma_depth,
            "final_ct_capture": "ordinary Gemma output after all 18 blocks + final_norms",
            "configured_stride": self.configured_stride,
            "effective_stride": self.stride,
            "gradient_raw_frames": list(self.options.gradient_raw_frames),
            "gradient_frame_selection": "all cadence frames"
            if not self.options.gradient_raw_frames
            else "explicit raw frames plus all occlusion raw frames",
            "occlusion_raw_frames": list(self.options.occlusion_raw_frames),
            "occlusion_batch_size": self.options.occlusion_batch_size,
            "metrics": list(self.options.metrics),
            "heatmap_scale_modes": list(self.options.heatmap_scale_modes),
            "render_video": self.options.render_video,
            "video_fps_override": self.options.video_fps,
            "video_encoder_requested": VIDEO_ENCODER if self.options.render_video else None,
            "max_frames": self.options.max_frames,
            "source_manifest": None if self.options.episode_manifest is None else str(self.options.episode_manifest),
            "source_manifest_hash": None
            if self.options.episode_manifest is None
            else v31.sha256_file(self.options.episode_manifest),
            "dataset_root": None if self.options.dataset_root is None else str(self.options.dataset_root),
            "episode_indices": list(self.options.episode_indices),
            "episodes": [
                {
                    "episode_id": source.episode_id,
                    "path": str(source.path),
                    "source_format": source.source_format,
                    "control_hz": source.control_hz,
                    "ground_truth_side": source.ground_truth_side,
                }
                for source in self.sources
            ],
            "core_api": CORE_API,
            "intervention_api": INTERVENTION_API,
            "objective_semantics": {
                "writer_loss": "mean over 256 tokens of ||M_(t-1)(K(c_t,i)) - V(c_t,i)||^2",
                "selected_write_tensor": "final c_t after all Gemma blocks and final output norm",
                "final_ct_capture": "ordinary Gemma output after all 18 blocks + final_norms; memory_layer=8 is used only to form the read query h_t",
                "measurement_state": "pre-write M_(t-1), including fast_weights and momentum",
                "gradient_map": "L2 norm of d(writer_loss)/d(top-camera SigLIP output token); the slots are spatially aligned but globally contextualized, so this is writer-surprise sensitivity, not raw-patch causality or actual clipped update magnitude",
                "gradient_primal_floor": "primal_final_ct_rms_floor and primal_writer_loss_abs_floor compare the reverse-mode attribution primal with the ordinary no-gradient intervention primal that is actually committed",
                "zero_read_map": "per-final-c_t-output-slot L2(normal retrieval minus zero retrieval); slot-aligned, not pixel attribution",
                "occlusion": "set one exact 14x14 patch of normalized 224x224 base_0_rgb to -1; all branches and a same-shaped unmodified baseline use the identical pre-state and the same intervention executable; discard branches and commit the batch-one unmodified intervention baseline once",
                "occlusion_update_comparison": "fast-only and full-state (fast_weights+momentum) differences compare candidate update vectors; cosine is signed",
                "write_order": "run reverse-mode attribution read-only against pre-state, then commit exactly one ordinary intervention baseline write; unmeasured cadence writes use that same intervention executable",
                "token_layout": token_heatmap.TOKEN_LAYOUT,
            },
            "interpretation_warnings": [
                "The gradient heatmap indexes globally contextualized SigLIP output token slots. It is writer-surprise sensitivity through final c_t, not raw-pixel causality or actual update magnitude; the write clip factor is stop-gradient.",
                "Gradient values are local to the reverse-mode primal. Per-frame primal_*_floor scalars quantify its mixed-precision/compiler numerical difference from the ordinary intervention primal that advances memory.",
                "The zero-read 16x16 map indexes contextual final-c_t output slots and must not be read as a camera-pixel heatmap.",
                "Occlusion is the numerical intervention on the actual candidate fast-state update; padding patches may be unchanged because they are already black.",
                "Absolute videos and raw contributions.npz values drive interpretation.",
                "Relative/history videos fit an episode baseline, erase absolute magnitude, and can amplify negligible residual variation; their filenames are explicitly labelled.",
            ],
        }
        _writer._write_json(self.options.output_dir / "run_manifest.json", manifest)  # noqa: SLF001

        summaries = []
        for source in self.sources:
            measured, total, replayed_writes = self._measure_episode(source)
            summaries.append(
                self._save_episode(
                    source,
                    measured,
                    total,
                    episodes_dir / _writer._safe_name(source.episode_id),  # noqa: SLF001
                    replayed_write_frames=replayed_writes,
                )
            )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "episode_count": len(summaries),
            "attribution_frames": sum(item["attribution_frames"] for item in summaries),
            "occlusion_frames": sum(item["occlusion_frames"] for item in summaries),
            "elapsed_s": time.monotonic() - start,
            "episodes": summaries,
        }
        _writer._write_json(self.options.output_dir / "summary.json", summary)  # noqa: SLF001
        return summary


def _parse_raw_frames(value: str) -> tuple[int, ...]:
    try:
        frames = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("raw frames must be comma-separated integers") from exc
    if not frames or any(frame < 0 for frame in frames) or tuple(sorted(set(frames))) != frames:
        raise argparse.ArgumentTypeError("raw frames must be a strictly increasing unique non-negative list")
    return frames


def _parse_metrics(value: str) -> tuple[str, ...]:
    metrics = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = {*PRIMARY_METRICS, *OCCLUSION_METRICS}
    if not metrics or set(metrics) - allowed or len(set(metrics)) != len(metrics):
        raise argparse.ArgumentTypeError(f"metrics must be a unique comma list from {sorted(allowed)}")
    return metrics


def _parse_scale_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(modes) - set(token_heatmap.SCALE_MODES)
    if not modes or unknown or len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError(
            f"heatmap scale modes must be a unique comma list from {token_heatmap.SCALE_MODES}"
        )
    return modes


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline end-to-end final-c_t writer attribution replay")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="pi05_yam_mem_v31")
    parser.add_argument("--output-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", dest="episode_paths", type=Path, action="append")
    source.add_argument("--episode-manifest", type=Path)
    source.add_argument("--dataset-root", type=Path)
    parser.add_argument("--episode-indices", type=_writer._parse_episode_indices)  # noqa: SLF001
    parser.add_argument("--stride", type=int)
    parser.add_argument("--gradient-raw-frames", type=_parse_raw_frames, default=())
    parser.add_argument("--occlusion-raw-frames", type=_parse_raw_frames, default=())
    parser.add_argument("--occlusion-batch-size", type=int, default=1)
    parser.add_argument("--metrics", type=_parse_metrics, default=PRIMARY_METRICS)
    parser.add_argument("--video-fps", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--colormap", choices=("inferno", "magma", "turbo", "viridis"), default="inferno")
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    parser.add_argument("--heatmap-scale-modes", type=_parse_scale_modes, default=("video", "per_token_history"))
    parser.add_argument("--zscore-range", type=float, default=3.0)
    parser.add_argument(
        "--include-letterbox-padding-in-scale",
        action="store_true",
        help="fit absolute color scales on padding tokens too (raw NPZ always retains them)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _build_parser().parse_args(argv)
    options = FinalCtRunOptions(
        checkpoint=namespace.checkpoint,
        output_dir=namespace.output_dir,
        episode_paths=tuple(namespace.episode_paths or ()),
        episode_manifest=namespace.episode_manifest,
        dataset_root=namespace.dataset_root,
        episode_indices=tuple(namespace.episode_indices or ()),
        config=namespace.config,
        stride=namespace.stride,
        gradient_raw_frames=namespace.gradient_raw_frames,
        occlusion_raw_frames=namespace.occlusion_raw_frames,
        occlusion_batch_size=namespace.occlusion_batch_size,
        metrics=namespace.metrics,
        video_fps=namespace.video_fps,
        max_frames=namespace.max_frames,
        render_video=not namespace.skip_video,
        colormap=namespace.colormap,
        alpha=namespace.alpha,
        upper_percentile=namespace.upper_percentile,
        heatmap_scale_modes=namespace.heatmap_scale_modes,
        zscore_range=namespace.zscore_range,
        exclude_letterbox_padding=not namespace.include_letterbox_padding_in_scale,
    )
    summary = FinalCtAttributionRunner(options).run()
    print(json.dumps(_writer._jsonable(summary), indent=2, sort_keys=True), flush=True)  # noqa: SLF001
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
