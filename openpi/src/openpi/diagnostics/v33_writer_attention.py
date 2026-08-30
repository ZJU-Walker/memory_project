"""v3.3 writer-attention diagnostic: where do the write tokens look, and does the instruction move them?

Replays memory episodes at the training write cadence against a v3.3 checkpoint, carrying the
real memory state (each frame's committed write is exactly the ``write_tokens`` the model would
write). Per frame it records three write-attention variants over the 256 layer-8 top-camera
patches:

* TRUE     -- the task-conditioned bank under the episode's real instruction;
* CF       -- the same frame under the counterfactual instruction (the other task);
* BASE(Q0) -- the unconditioned v3.2-style bank, the within-frame baseline.

TRUE-vs-CF divergence is the direct readout of the v3.3 conditioner (handoff section 17.3): if
the instruction steers the writer, the two maps must differ, most meaningfully during the
evidence phase. TRUE-vs-BASE measures how far training has moved conditioning overall.

Before any replay the runner prints the two zero-init pathway scalars from the raw checkpoint
(conditioner ``output_proj`` kernel norm and the memory content-gate norm): if the first is
still ~0 the factorial cannot show anything, and knowing that costs seconds, not a rollout.

Outputs per episode: an H.264 MP4 (panels: top camera | TRUE | CF | |TRUE-CF|, video-fitted
color scales), an NPZ with the head-averaged per-slot maps, and per-frame metrics inside the
run-level JSON summary (per-phase aggregates are also printed as a table). The metrics include
the writer's fast-weight gradient norm before its inner clip, the resulting clip factor, and
whether that frame saturated the clip. These are distinct from the outer optimizer gradient
norm printed by ``scripts/train.py``. It also measures layer-8, raw-retrieval, and actual
gated-injection RMS so a fixed read multiplier can be calibrated without conflating the raw
retrieval with the current 2048-channel content gate.
"""

# ruff: noqa: SLF001, I001 - this diagnostic deliberately reuses the private v3.1/v3.2 replay
# machinery (episode sources, inline-image decode, checkpoint alignment); those helpers ARE the
# shared implementation, re-exporting them publicly for one consumer would be churn. And the
# import ORDER is load-bearing: pyarrow.parquet must load before any openpi module -- with the
# openpi stack (jax/flax/sentencepiece/...) resident first, every parquet read of this
# dataset's inline-image files segfaults on iris workstations (exit 139, no traceback), the
# same environment-quirk class as the pinned torch-before-tensorflow order in data_loader.

from __future__ import annotations

import pyarrow.parquet as pq

import dataclasses
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

from openpi import transforms as _transforms
from openpi.diagnostics import token_heatmap
from openpi.diagnostics import writer_contribution as _wc
from openpi.diagnostics.v32_checkpoint import _align_params
from openpi.models import model as _model
from openpi.shared import nnx_utils
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

SCHEMA_VERSION = "openpi.v33.writer_attention.v2"
_EPS = 1e-12


@dataclasses.dataclass(frozen=True)
class Options:
    checkpoint: Path
    dataset_root: Path
    output_dir: Path
    config: str = "pi05_yam_mem_v33"
    # Empty tuple selects one usable episode per (instruction, side) cell automatically.
    episode_indices: tuple[int, ...] = ()
    stride: int | None = None
    video_fps: float = 4.0

    def __post_init__(self) -> None:
        for name in ("checkpoint", "dataset_root", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        object.__setattr__(self, "episode_indices", tuple(self.episode_indices))
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be positive")
        if not math.isfinite(self.video_fps) or self.video_fps <= 0:
            raise ValueError("video_fps must be finite and positive")


def _param(raw: Any, *path: str) -> np.ndarray:
    """Descend a restored pure-dict checkpoint, unwrapping trailing ``{"value": array}``."""
    node = raw
    for key in path:
        if not isinstance(node, dict) or key not in node:
            raise KeyError(f"checkpoint is missing parameter path {'/'.join(path)} (stopped at {key!r})")
        node = node[key]
    if isinstance(node, dict) and set(node) == {"value"}:
        node = node["value"]
    if isinstance(node, dict):
        raise KeyError(f"parameter path {'/'.join(path)} does not end at an array leaf")
    return np.asarray(node)


def pathway_scalars(raw_params: Any) -> dict[str, float]:
    """The two zero-init pathway norms that gate what this diagnostic can show."""
    out_kernel = _param(raw_params, "write_query_conditioner", "output_proj", "kernel")
    gate = _param(raw_params, "memory_gate")
    return {
        "conditioner_output_proj_norm": float(np.linalg.norm(out_kernel)),
        "memory_gate_norm": float(np.linalg.norm(gate)),
    }


def writer_clip_statistics(grad_norms: np.ndarray, max_grad_norm: float) -> dict[str, float]:
    """Summarize the pre-clip fast-weight gradient norms from ``TitansMemory.write``."""
    values = np.asarray(grad_norms, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"grad_norms must be a non-empty vector, got {values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("grad_norms must contain finite nonnegative values")
    if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be finite and positive")
    clip_factors = np.minimum(1.0, max_grad_norm / (values + _EPS))
    return {
        "write_grad_norm_preclip_min": float(np.min(values)),
        "write_grad_norm_preclip_median": float(np.median(values)),
        "write_grad_norm_preclip_mean": float(np.mean(values)),
        "write_grad_norm_preclip_p95": float(np.percentile(values, 95.0)),
        "write_grad_norm_preclip_max": float(np.max(values)),
        "write_clip_factor_median": float(np.median(clip_factors)),
        "write_clip_saturation_fraction": float(np.mean(values > max_grad_norm)),
    }


def retrieval_scale_statistics(
    h8_valid_rms: np.ndarray,
    h8_valid_token_count: np.ndarray,
    h8_image_rms: np.ndarray,
    h8_context_valid_rms: np.ndarray,
    retrieved_rms: np.ndarray,
    memory_token_rms: np.ndarray,
) -> dict[str, float | int | None]:
    """Summarize the scale bridge from raw retrieval to the layer-8 residual stream.

    ``retrieval_match_c`` would make ``c * retrieved`` match valid layer-8 prefix RMS.
    Near-zero reads are excluded from ratios instead of becoming epsilon-divided outliers.
    The caller selects carried-state rather than fresh-memory frames when that distinction is
    important; a trained checkpoint's learnable initial memory need not read exactly zero.
    """
    h8 = np.asarray(h8_valid_rms, dtype=np.float64)
    counts = np.asarray(h8_valid_token_count, dtype=np.float64)
    image = np.asarray(h8_image_rms, dtype=np.float64)
    context = np.asarray(h8_context_valid_rms, dtype=np.float64)
    retrieved = np.asarray(retrieved_rms, dtype=np.float64)
    injected = np.asarray(memory_token_rms, dtype=np.float64)
    inputs = (h8, counts, image, context, retrieved, injected)
    if h8.ndim != 1 or h8.size == 0 or any(values.shape != h8.shape for values in inputs[1:]):
        raise ValueError(
            "all retrieval-scale inputs must be equal non-empty vectors; "
            f"got {[values.shape for values in inputs]}"
        )
    if not all(np.all(np.isfinite(values)) and np.all(values >= 0) for values in inputs):
        raise ValueError("RMS inputs must contain finite nonnegative values")
    if np.any(counts <= 0):
        raise ValueError("h8_valid_token_count must be positive")

    nonzero_retrieval = retrieved > _EPS
    nonzero_injection = injected > _EPS
    valid_h8 = h8 > _EPS
    c_values = h8[nonzero_retrieval] / retrieved[nonzero_retrieval]
    c_extra_values = h8[nonzero_injection] / injected[nonzero_injection]
    injected_ratios = injected[valid_h8] / h8[valid_h8]
    h8_energy_rms = float(np.sqrt(np.sum(np.square(h8) * counts) / np.sum(counts)))
    retrieved_energy_rms = float(np.sqrt(np.mean(np.square(retrieved))))
    injected_energy_rms = float(np.sqrt(np.mean(np.square(injected))))
    out: dict[str, float | int | None] = {
        "retrieval_scale_frame_count": int(h8.size),
        "h8_valid_rms_median": float(np.median(h8)),
        "h8_valid_rms_energy": h8_energy_rms,
        "h8_image_rms_median": float(np.median(image)),
        "h8_context_valid_rms_median": float(np.median(context)),
        "retrieved_rms_median": float(np.median(retrieved)),
        "retrieved_rms_energy": retrieved_energy_rms,
        "memory_token_rms_median": float(np.median(injected)),
        "memory_token_rms_energy": injected_energy_rms,
        "retrieval_zero_fraction": float(np.mean(~nonzero_retrieval)),
        "retrieval_match_c_count": int(c_values.size),
        "retrieval_match_c_median": None,
        "retrieval_match_c_p05": None,
        "retrieval_match_c_p95": None,
        "retrieval_match_c_energy": (
            h8_energy_rms / retrieved_energy_rms if retrieved_energy_rms > _EPS else None
        ),
        "gated_match_c_extra_median": None,
        "gated_match_c_extra_p05": None,
        "gated_match_c_extra_p95": None,
        "gated_match_c_extra_energy": h8_energy_rms / injected_energy_rms if injected_energy_rms > _EPS else None,
        "memory_token_to_h8_ratio_median": None,
    }
    if c_values.size:
        out.update(
            {
                "retrieval_match_c_median": float(np.median(c_values)),
                "retrieval_match_c_p05": float(np.percentile(c_values, 5.0)),
                "retrieval_match_c_p95": float(np.percentile(c_values, 95.0)),
            }
        )
    if c_extra_values.size:
        out.update(
            {
                "gated_match_c_extra_median": float(np.median(c_extra_values)),
                "gated_match_c_extra_p05": float(np.percentile(c_extra_values, 5.0)),
                "gated_match_c_extra_p95": float(np.percentile(c_extra_values, 95.0)),
            }
        )
    if injected_ratios.size:
        out["memory_token_to_h8_ratio_median"] = float(np.median(injected_ratios))
    return out


def head_mean(attention: np.ndarray) -> np.ndarray:
    """[b, h, q, n] -> head-averaged [q, n] for batch row 0; rows stay softmax distributions."""
    attention = np.asarray(attention, dtype=np.float64)
    if attention.ndim != 4 or attention.shape[0] < 1:
        raise ValueError(f"attention must be [b, h, q, n], got {attention.shape}")
    return attention[0].mean(axis=0)


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-slot Jensen-Shannon divergence (nats) between [q, n] row distributions."""
    p = np.asarray(p, dtype=np.float64) + _EPS
    q = np.asarray(q, dtype=np.float64) + _EPS
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b), axis=-1)  # noqa: E731
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def left_half_mass(slot_maps: np.ndarray) -> float:
    """Fraction of total attention mass on the left half of the 16x16 patch grid."""
    maps = np.asarray(slot_maps, dtype=np.float64)
    grid = maps.reshape(*maps.shape[:-1], 16, 16)
    left = grid[..., :8].sum()
    total = grid.sum()
    return float(left / max(total, _EPS))


def slot_entropy(slot_maps: np.ndarray) -> float:
    """Mean entropy (nats) of the per-slot attention distributions; max is ln(256) ~ 5.55."""
    maps = np.asarray(slot_maps, dtype=np.float64) + _EPS
    maps = maps / maps.sum(axis=-1, keepdims=True)
    return float(-(maps * np.log(maps)).sum(axis=-1).mean())


def _read_tasks(dataset_root: Path) -> dict[int, str]:
    tasks = {}
    with open(dataset_root / "meta" / "tasks.jsonl") as f:
        for line in f:
            row = json.loads(line)
            tasks[int(row["task_index"])] = str(row["task"])
    return tasks


def _read_prompts(dataset_root: Path) -> dict[int, str]:
    raw = json.loads((dataset_root / "meta" / "episode_prompts.json").read_text())
    prompts = {int(k): str(v) for k, v in raw.items()}
    if len(set(prompts.values())) != 2:
        raise ValueError(
            f"the counterfactual prompt is only defined for exactly two distinct instructions; "
            f"got {sorted(set(prompts.values()))}"
        )
    return prompts


@dataclasses.dataclass(frozen=True)
class _EpisodePlan:
    episode: int
    prompt: str
    counterfactual: str
    side: str
    evidence: tuple[int, int]
    memory: tuple[int, int]
    length: int


def _episode_phases(task_ids: np.ndarray, tasks: dict[int, str], data_config: Any) -> tuple | None:
    labels = [tasks[int(t)] for t in task_ids]
    evidence = [i for i, s in enumerate(labels) if s in data_config.evidence_subtasks]
    memory = [i for i, s in enumerate(labels) if s in data_config.memory_required_subtasks]
    if not evidence or not memory or evidence[-1] >= memory[0]:
        return None
    return (evidence[0], evidence[-1]), (memory[0], memory[-1])


def _plan_episodes(options: Options, data_config: Any) -> list[_EpisodePlan]:
    tasks = _read_tasks(options.dataset_root)
    prompts = _read_prompts(options.dataset_root)
    first, second = sorted(set(prompts.values()))
    other = {first: second, second: first}

    candidates = options.episode_indices or tuple(sorted(prompts))
    plans: list[_EpisodePlan] = []
    seen_cells: set[tuple[str, str]] = set()
    for episode in candidates:
        source = _wc._load_lerobot_sources(options.dataset_root, [episode])[0]
        table = pq.read_table(source.path, columns=["task_index"])
        task_ids = np.asarray(table["task_index"])
        phases = _episode_phases(task_ids, tasks, data_config)
        if phases is None:
            if options.episode_indices:
                raise ValueError(f"episode {episode} has no usable evidence/wait phases")
            continue
        final = tasks[int(task_ids[-1])].lower()
        side = "left" if "left" in final else ("right" if "right" in final else "?")
        cell = (prompts[episode], side)
        if not options.episode_indices and cell in seen_cells:
            continue
        seen_cells.add(cell)
        plans.append(
            _EpisodePlan(
                episode=episode,
                prompt=prompts[episode],
                counterfactual=other[prompts[episode]],
                side=side,
                evidence=phases[0],
                memory=phases[1],
                length=len(task_ids),
            )
        )
        if not options.episode_indices and len(seen_cells) == 4:
            break
    if not plans:
        raise ValueError("no usable episodes found")
    return plans


def _slot_scales(slot_maps: np.ndarray) -> list[token_heatmap.ColorScale]:
    """One video-level color scale per slot. Every slot's attention row is a softmax
    distribution (total mass 1 by construction), so per-slot normalization is the honest
    display: slots differ in SHAPE, not in total mass."""
    maps = np.asarray(slot_maps, dtype=np.float64)
    if maps.ndim != 3 or maps.shape[1:] != (16, 256):
        raise ValueError(f"slot maps must be [frames, 16, 256], got {maps.shape}")
    scales = []
    for slot in range(16):
        vmax = float(np.percentile(maps[:, slot], 99.0))
        scales.append(
            token_heatmap.ColorScale(
                vmin=0.0,
                vmax=max(vmax, float(_EPS)),
                lower_percentile=0.0,
                upper_percentile=99.0,
                anchor_zero=True,
            )
        )
    return scales


_SLOT_GRID_HEADER = 28

# Rendering styles for the per-slot grid. They answer different questions and are NOT
# interchangeable:
#
# "video" -- one fixed scale per slot across the whole episode (99th percentile), inferno,
#   flat alpha. Frames are comparable to each other, so "attention got stronger/weaker" is
#   readable. Cost: attention here is extremely peaked (the top patch often holds ~35-40% of
#   a frame's mass), so against a video-wide ceiling ~96% of patches render below 2%
#   brightness -- a genuinely sharp map can look like an almost untouched image.
#
# "v32" -- the v3.2 convention (diagnostics/v32_checkpoint.py `_attention_video`): each tile
#   normalized by ITS OWN per-frame maximum, JET colormap, opacity proportional to intensity.
#   Every tile's peak therefore saturates red in every frame, which makes the SHAPE of each
#   slot's attention obvious at a glance. Cost: it is purely relative -- a weak diffuse frame
#   and a razor-sharp one look equally red, so it says nothing about magnitude across time.
#
# Read "v32" for locating what a slot looks at; read "video" for whether that changed.
SLOT_GRID_STYLES = ("video", "v32")


def _v32_style_tile(image: np.ndarray, slot_map: np.ndarray) -> np.ndarray:
    """One tile in the v3.2 convention: per-frame max normalization, JET, intensity alpha.

    Mirrors `v32_checkpoint.V32CheckpointRunner._attention_video` exactly so the two
    generations' videos are visually comparable.
    """
    grid = np.asarray(slot_map, dtype=np.float64).reshape(token_heatmap.TOKEN_GRID_SIZE, token_heatmap.TOKEN_GRID_SIZE)
    heat = grid / max(float(grid.max()), _EPS)
    heat224 = cv2.resize(heat.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap((heat224 * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    weight = (0.6 * heat224)[..., None]
    return (image.astype(np.float32) * (1 - weight) + color * weight).astype(np.uint8)


def slot_grid_frame(
    image: np.ndarray,
    slot_maps: np.ndarray,
    scales: list[token_heatmap.ColorScale] | None,
    label: str,
    *,
    style: str = "video",
) -> np.ndarray:
    """Compose one 4x4 grid frame (16 write slots) over the same camera image: a header bar
    plus 16 per-slot 224x224 overlays, slot index and (in v32 style) the slot's entropy and
    effective token count stamped in each tile. See SLOT_GRID_STYLES for the normalizations."""
    maps = np.asarray(slot_maps, dtype=np.float64)
    if maps.shape != (16, 256):
        raise ValueError(f"per-frame slot maps must be [16, 256], got {maps.shape}")
    if style not in SLOT_GRID_STYLES:
        raise ValueError(f"unsupported slot grid style {style!r}; choose one of {SLOT_GRID_STYLES}")
    if style == "video" and (scales is None or len(scales) != 16):
        raise ValueError("style='video' needs one color scale per slot")

    normalized = maps / np.maximum(maps.sum(axis=-1, keepdims=True), _EPS)
    clipped = np.clip(normalized, _EPS, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=-1)

    rows = []
    for row_index in range(4):
        tiles = []
        for column in range(4):
            slot = 4 * row_index + column
            if style == "v32":
                tile = _v32_style_tile(image, maps[slot])
                text = f"s{slot} eff={np.exp(entropy[slot]):.0f} H={entropy[slot]:.2f}"
            else:
                tile, _ = token_heatmap.heatmap_overlay(
                    image, maps[slot], scales[slot], scale_mode="video", draw_grid=False
                )
                text = f"s{slot}"
            cv2.putText(tile, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
            tiles.append(tile)
        rows.append(np.concatenate(tiles, axis=1))
    grid = np.concatenate(rows, axis=0)
    bar = np.zeros((_SLOT_GRID_HEADER, grid.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, f"[{style}] {label}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([bar, grid], axis=0)


def render_slot_grid(
    images: list[np.ndarray],
    slot_maps: np.ndarray,
    labels: list[str],
    path: Path,
    *,
    fps: float = 4.0,
    style: str = "video",
) -> str:
    """Encode the per-slot 4x4 grid video; returns the encoder identifier."""
    maps = np.asarray(slot_maps, dtype=np.float64)
    if len(images) != maps.shape[0] or len(labels) != maps.shape[0]:
        raise ValueError(f"images/labels/maps disagree: {len(images)}/{len(labels)}/{maps.shape[0]}")
    scales = _slot_scales(maps) if style == "video" else None
    frames = [
        slot_grid_frame(image, maps[index], scales, labels[index], style=style) for index, image in enumerate(images)
    ]
    return token_heatmap.encode_mp4(frames, path, fps)


def slot_grid_from_npz(
    npz_path: Path,
    dataset_root: Path,
    episode: int,
    output_path: Path,
    *,
    variant: str = "true",
    fps: float = 4.0,
    style: str = "video",
) -> str:
    """Re-render the per-slot grid from a saved maps NPZ -- CPU only, no model or GPU.

    The camera frames are re-decoded from the dataset and letterboxed with the shared
    ``raw_top_camera_to_model_rgb`` geometry, so tiles align with the training-time patches.
    """
    data = np.load(npz_path)
    if variant not in data:
        raise KeyError(f"{npz_path} has no variant {variant!r}; available: {sorted(data.keys())}")
    frame_indices = np.asarray(data["frames"], dtype=np.int64)
    maps = np.asarray(data[variant], dtype=np.float64)
    tasks = _read_tasks(dataset_root)
    source = _wc._load_lerobot_sources(dataset_root, [episode])[0]
    rows = pq.read_table(source.path, columns=["image", "frame_index", "task_index"]).to_pylist()
    by_frame = {int(row["frame_index"]): row for row in rows}
    images, labels = [], []
    for frame in frame_indices:
        row = by_frame[int(frame)]
        raw = _wc.WriterContributionRunner._decode_inline_image(row["image"], field="image", raw_frame=int(frame))
        model_rgb, _ = token_heatmap.raw_top_camera_to_model_rgb(raw)
        images.append(model_rgb)
        labels.append(f"ep{episode} f{int(frame)} {variant.upper()} | {tasks[int(row['task_index'])]}"[:70])
    return render_slot_grid(images, maps, labels, output_path, fps=fps, style=style)


class V33WriterAttentionRunner:
    def __init__(self, options: Options):
        init_started = time.monotonic()

        def stage(message: str) -> None:
            print(f"[startup +{time.monotonic() - init_started:.1f}s] {message}", flush=True)

        self.options = options
        if options.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite output directory: {options.output_dir}")
        if not (options.checkpoint / "params").is_dir():
            raise FileNotFoundError(f"checkpoint has no params item: {options.checkpoint}")

        stage("building config and tokenizers")
        self.train_config = _config.get_config(options.config)
        model_config = self.train_config.model
        if not getattr(model_config, "memory_task_conditioned_write", False):
            raise ValueError("the writer-attention factorial requires memory_task_conditioned_write=True")
        self.data_config = self.train_config.data.create(self.train_config.assets_dirs, model_config)
        self.stride = int(self.data_config.memory_stride_frames) if options.stride is None else options.stride

        # Norm stats: prefer the checkpoint's own assets (exact training-time stats), fall back
        # to the repo assets dir the config points at.
        candidates = [
            options.checkpoint / "assets" / self.data_config.asset_id,
            self.train_config.assets_dirs / self.data_config.asset_id,
        ]
        norm_stats = None
        for candidate in candidates:
            if candidate.is_dir():
                norm_stats = _normalize.load(candidate)
                break
        if norm_stats is None:
            raise FileNotFoundError(f"no norm stats under any of: {[str(c) for c in candidates]}")

        stage("restoring FP32 checkpoint parameters")
        raw_params = _model.restore_params(options.checkpoint / "params", dtype=jnp.float32)
        stage("checkpoint parameters restored; building abstract model state")
        self.scalars = pathway_scalars(raw_params)
        abstract_state = nnx.state(nnx.eval_shape(model_config.create, jax.random.key(0))).to_pure_dict()
        stage("aligning checkpoint tree and assembling model")
        self.model = model_config.load(_align_params(abstract_state, raw_params), remove_extra_params=False)

        input_transforms = [
            transform
            for transform in self.data_config.data_transforms.inputs
            if not isinstance(transform, _transforms.BuildMemorySequence)
        ]
        self.input_transform = _transforms.compose(
            [
                *input_transforms,
                _transforms.Normalize(norm_stats, use_quantiles=self.data_config.use_quantile_norm),
                *self.data_config.model_transforms.inputs,
            ]
        )
        stage("freezing diagnostic callables")
        self._qstep = nnx_utils.module_jit(self.model.v32_query_attention_step)
        self._write = nnx_utils.module_jit(self.model.memory.write)
        self.max_grad_norm = float(self.model.memory.config.max_grad_norm)
        stage("runner initialization complete")

    def _observation(self, row: dict[str, Any], raw_frame: int, prompt: str) -> tuple[_model.Observation, np.ndarray]:
        decode = _wc.WriterContributionRunner._decode_inline_image
        transformed = self.input_transform(
            {
                "observation/image": decode(row["image"], field="image", raw_frame=raw_frame),
                "observation/left_wrist_image": decode(
                    row["left_wrist_image"], field="left_wrist_image", raw_frame=raw_frame
                ),
                "observation/right_wrist_image": decode(
                    row["right_wrist_image"], field="right_wrist_image", raw_frame=raw_frame
                ),
                "observation/state": np.asarray(row["state"], dtype=np.float32),
                "prompt": prompt,
            }
        )
        model_image = np.array(np.asarray(transformed["image"]["base_0_rgb"]), copy=True)
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
        return _model.Observation.from_dict(batched), model_image

    def _replay(self, plan: _EpisodePlan, tasks: dict[int, str]) -> dict[str, Any]:
        source = _wc._load_lerobot_sources(self.options.dataset_root, [plan.episode])[0]
        columns = ["image", "left_wrist_image", "right_wrist_image", "state", "frame_index", "task_index"]
        rows = pq.read_table(source.path, columns=columns).to_pylist()

        memory_state = self.model.memory.init_state(1)
        frames: list[dict[str, Any]] = []
        maps: dict[str, list[np.ndarray]] = {"true": [], "cf": [], "base": [], "read": []}
        images: list[np.ndarray] = []
        started = time.monotonic()
        for row in rows:
            raw_frame = int(row["frame_index"])
            if raw_frame % self.stride or raw_frame > plan.memory[1]:
                continue
            observation_true, model_image = self._observation(row, raw_frame, plan.prompt)
            observation_cf, _ = self._observation(row, raw_frame, plan.counterfactual)
            out_true = self._qstep(observation_true, memory_state)
            out_cf = self._qstep(observation_cf, memory_state)

            true_map = head_mean(out_true["write_attention"])
            cf_map = head_mean(out_cf["write_attention"])
            base_map = head_mean(out_true["write_attention_base"])
            read_map = head_mean(out_true["read_attention"])
            query_true = np.asarray(out_true["write_queries"], dtype=np.float64)[0]
            query_cf = np.asarray(out_cf["write_queries"], dtype=np.float64)[0]
            query_base = np.asarray(self.model.write_query_compressor.query_bank.value, dtype=np.float64)
            js = js_divergence(true_map, cf_map)
            next_memory_state, write_aux = self._write(memory_state, out_true["write_tokens"])
            write_grad_norm = float(np.asarray(write_aux["grad_norm"])[0])
            write_clip_factor = min(1.0, self.max_grad_norm / (write_grad_norm + _EPS))
            h8_all_rms = float(np.asarray(out_true["h8_all_rms"])[0])
            h8_valid_rms = float(np.asarray(out_true["h8_valid_rms"])[0])
            h8_valid_token_count = int(np.asarray(out_true["h8_valid_token_count"])[0])
            h8_image_rms = float(np.asarray(out_true["h8_image_rms"])[0])
            h8_context_valid_rms = float(np.asarray(out_true["h8_context_valid_rms"])[0])
            h8_top_rms = float(np.asarray(out_true["h8_top_rms"])[0])
            retrieved_rms = float(np.asarray(out_true["retrieved_rms"])[0])
            memory_token_rms = float(np.asarray(out_true["memory_token_rms"])[0])

            label = tasks[int(row["task_index"])]
            phase = (
                "evidence"
                if plan.evidence[0] <= raw_frame <= plan.evidence[1]
                else "waiting"
                if raw_frame >= plan.memory[0]
                else "retention"
                if raw_frame > plan.evidence[1]
                else "approach"
            )
            frames.append(
                {
                    "frame": raw_frame,
                    "writes_before": len(frames),
                    "phase": phase,
                    "task": label,
                    "js_mean": float(js.mean()),
                    "js_max": float(js.max()),
                    "delta_query_cf": float(np.linalg.norm(query_true - query_cf)),
                    "delta_query_base": float(np.linalg.norm(query_true - query_base)),
                    "delta_write_tokens_cf": float(
                        np.linalg.norm(
                            np.asarray(out_true["write_tokens"], dtype=np.float64)
                            - np.asarray(out_cf["write_tokens"], dtype=np.float64)
                        )
                    ),
                    "left_mass_true": left_half_mass(true_map),
                    "left_mass_cf": left_half_mass(cf_map),
                    "left_mass_read": left_half_mass(read_map),
                    "entropy_true": slot_entropy(true_map),
                    "write_norm": float(np.asarray(out_true["write_slot_norm"]).mean()),
                    # ``grad_norm`` is the fast-weight MLP gradient after the 16-token loss
                    # mean and before the per-sample inner clip. It is not the outer training
                    # gradient norm. The clip is saturated exactly when this exceeds the
                    # configured threshold (1.0 in v3.3).
                    "write_grad_norm_preclip": write_grad_norm,
                    "write_clip_factor": write_clip_factor,
                    "write_clip_saturated": write_grad_norm > self.max_grad_norm,
                    "write_surprise": float(np.asarray(write_aux["surprise"])[0]),
                    "write_theta": float(np.asarray(write_aux["theta"])[0]),
                    "write_eta": float(np.asarray(write_aux["eta"])[0]),
                    "write_alpha": float(np.asarray(write_aux["alpha"])[0]),
                    # Scale bridge for a fixed read-injection multiplier. ``h8_all_rms`` is
                    # the literal static-shape tensor (including prompt padding), while
                    # ``h8_valid_rms`` excludes masked padding and defines the reported c.
                    "h8_all_rms": h8_all_rms,
                    "h8_valid_rms": h8_valid_rms,
                    "h8_valid_token_count": h8_valid_token_count,
                    "h8_image_rms": h8_image_rms,
                    "h8_context_valid_rms": h8_context_valid_rms,
                    "h8_top_rms": h8_top_rms,
                    "retrieved_rms": retrieved_rms,
                    "memory_token_rms": memory_token_rms,
                    "retrieval_match_c": h8_valid_rms / retrieved_rms if retrieved_rms > _EPS else None,
                    "memory_token_to_h8_ratio": memory_token_rms / h8_valid_rms if h8_valid_rms > _EPS else None,
                }
            )
            for key, value in (("true", true_map), ("cf", cf_map), ("base", base_map), ("read", read_map)):
                maps[key].append(value.astype(np.float16))
            images.append(model_image)
            memory_state = next_memory_state
        if not frames:
            raise ValueError(f"episode {plan.episode}: no frames on the stride grid before the wait end")
        print(
            f"episode {plan.episode} ({plan.prompt} / {plan.side}): {len(frames)} frames replayed "
            f"in {time.monotonic() - started:.1f}s"
        )
        return {"plan": plan, "frames": frames, "maps": maps, "images": images}

    def _render(self, replay: dict[str, Any], path: Path) -> str:
        plan: _EpisodePlan = replay["plan"]
        summed = {key: [m.astype(np.float64).sum(axis=0) for m in replay["maps"][key]] for key in ("true", "cf")}
        diffs = [np.abs(a - b) for a, b in zip(summed["true"], summed["cf"], strict=True)]

        def scale_for(series: list[np.ndarray]) -> token_heatmap.ColorScale:
            values = np.concatenate([s.ravel() for s in series])
            vmax = float(np.percentile(values, 99.0))
            return token_heatmap.ColorScale(
                vmin=0.0,
                vmax=max(vmax, float(_EPS)),
                lower_percentile=0.0,
                upper_percentile=99.0,
                anchor_zero=True,
            )

        attention_scale = scale_for(summed["true"] + summed["cf"])
        diff_scale = scale_for(diffs)
        video_frames = []
        for index, (image, frame) in enumerate(zip(replay["images"], replay["frames"], strict=True)):
            panels = [image]
            for values, scale in (
                (summed["true"][index], attention_scale),
                (summed["cf"][index], attention_scale),
                (diffs[index], diff_scale),
            ):
                overlay, _ = token_heatmap.heatmap_overlay(image, values, scale, scale_mode="video")
                panels.append(overlay)
            composite = np.concatenate(panels, axis=1)
            for column, text in enumerate(
                (
                    f"f{frame['frame']} {frame['phase']}",
                    f"TRUE: {plan.prompt.split()[-1]}",
                    f"CF: {plan.counterfactual.split()[-1]}",
                    f"|TRUE-CF| js={frame['js_mean']:.3f}",
                )
            ):
                cv2.putText(
                    composite,
                    text,
                    (column * 224 + 4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            video_frames.append(composite)
        return token_heatmap.encode_mp4(video_frames, path, self.options.video_fps)

    def _phase_table(self, all_frames: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        table: dict[str, dict[str, Any]] = {}
        for phase in ("approach", "evidence", "retention", "waiting"):
            rows = [f for f in all_frames if f["phase"] == phase]
            if not rows:
                continue
            carried_rows = [f for f in rows if f["writes_before"] > 0]
            scale_rows = carried_rows or rows
            table[phase] = {
                "frames": len(rows),
                "js_mean": float(np.mean([f["js_mean"] for f in rows])),
                "js_max": float(np.max([f["js_max"] for f in rows])),
                "delta_query_cf_mean": float(np.mean([f["delta_query_cf"] for f in rows])),
                "delta_query_base_mean": float(np.mean([f["delta_query_base"] for f in rows])),
                "left_mass_true_mean": float(np.mean([f["left_mass_true"] for f in rows])),
                "entropy_true_mean": float(np.mean([f["entropy_true"] for f in rows])),
                **writer_clip_statistics(
                    np.asarray([f["write_grad_norm_preclip"] for f in rows]), self.max_grad_norm
                ),
                **retrieval_scale_statistics(
                    np.asarray([f["h8_valid_rms"] for f in scale_rows]),
                    np.asarray([f["h8_valid_token_count"] for f in scale_rows]),
                    np.asarray([f["h8_image_rms"] for f in scale_rows]),
                    np.asarray([f["h8_context_valid_rms"] for f in scale_rows]),
                    np.asarray([f["retrieved_rms"] for f in scale_rows]),
                    np.asarray([f["memory_token_rms"] for f in scale_rows]),
                ),
                "retrieval_scale_carried_only": bool(carried_rows),
            }
        return table

    def run(self) -> dict[str, Any]:
        print(f"pathway scalars: {self.scalars}")
        tasks = _read_tasks(self.options.dataset_root)
        plans = _plan_episodes(self.options, self.data_config)
        print(f"replaying {len(plans)} episodes: {[(p.episode, p.prompt, p.side) for p in plans]}")

        self.options.output_dir.mkdir(parents=True)
        report: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "checkpoint": str(self.options.checkpoint),
            "stride": self.stride,
            "inner_clip_max_grad_norm": self.max_grad_norm,
            "pathway_scalars": self.scalars,
            "episodes": [],
        }
        every_frame: list[dict[str, Any]] = []
        for plan in plans:
            replay = self._replay(plan, tasks)
            stem = f"episode_{plan.episode:03d}"
            encoder = self._render(replay, self.options.output_dir / f"{stem}.mp4")
            slot_labels = [
                f"ep{plan.episode} f{frame['frame']} {frame['phase']} | TRUE: {plan.prompt}"
                for frame in replay["frames"]
            ]
            # Both normalizations: "v32" is readable at a glance (per-frame per-slot max, JET),
            # "video" is comparable across frames. See SLOT_GRID_STYLES.
            slot_maps = np.stack(replay["maps"]["true"]).astype(np.float64)
            slot_encoder = {
                style: render_slot_grid(
                    replay["images"],
                    slot_maps,
                    slot_labels,
                    self.options.output_dir
                    / (f"{stem}_slots.mp4" if style == "video" else f"{stem}_slots_{style}.mp4"),
                    fps=self.options.video_fps,
                    style=style,
                )
                for style in SLOT_GRID_STYLES
            }
            np.savez_compressed(
                self.options.output_dir / f"{stem}_maps.npz",
                frames=np.asarray([f["frame"] for f in replay["frames"]], dtype=np.int32),
                **{key: np.stack(value) for key, value in replay["maps"].items()},
            )
            report["episodes"].append(
                {
                    "episode": plan.episode,
                    "prompt": plan.prompt,
                    "counterfactual": plan.counterfactual,
                    "side": plan.side,
                    "evidence": list(plan.evidence),
                    "memory": list(plan.memory),
                    "video_encoder": encoder,
                    "slot_video_encoder": slot_encoder,
                    "phase_table": self._phase_table(replay["frames"]),
                    "frames": replay["frames"],
                }
            )
            every_frame.extend(replay["frames"])

        report["phase_table"] = self._phase_table(every_frame)
        report["writer_clip_summary"] = writer_clip_statistics(
            np.asarray([frame["write_grad_norm_preclip"] for frame in every_frame]), self.max_grad_norm
        )
        carried_frames = [frame for frame in every_frame if frame["writes_before"] > 0]
        scale_summary_frames = carried_frames or every_frame
        report["retrieval_scale_summary"] = retrieval_scale_statistics(
            np.asarray([frame["h8_valid_rms"] for frame in scale_summary_frames]),
            np.asarray([frame["h8_valid_token_count"] for frame in scale_summary_frames]),
            np.asarray([frame["h8_image_rms"] for frame in scale_summary_frames]),
            np.asarray([frame["h8_context_valid_rms"] for frame in scale_summary_frames]),
            np.asarray([frame["retrieved_rms"] for frame in scale_summary_frames]),
            np.asarray([frame["memory_token_rms"] for frame in scale_summary_frames]),
        )
        report["retrieval_scale_summary"]["retrieval_scale_carried_only"] = bool(carried_frames)
        (self.options.output_dir / "summary.json").write_text(json.dumps(report, indent=2))
        print(
            f"\n{'phase':>10} {'frames':>7} {'js_mean':>9} {'js_max':>9} {'dQ_cf':>9} {'dQ_base':>9} {'left':>6} {'H':>6}"
        )
        for phase, row in report["phase_table"].items():
            print(
                f"{phase:>10} {row['frames']:>7} {row['js_mean']:>9.4f} {row['js_max']:>9.4f} "
                f"{row['delta_query_cf_mean']:>9.3f} {row['delta_query_base_mean']:>9.3f} "
                f"{row['left_mass_true_mean']:>6.3f} {row['entropy_true_mean']:>6.3f}"
            )
        print(
            f"\ninner writer clip (threshold={self.max_grad_norm:g}; grad norms are pre-clip)\n"
            f"{'phase':>10} {'frames':>7} {'g_min':>10} {'g_p50':>10} {'g_p95':>10} "
            f"{'g_max':>10} {'sat_frac':>9} {'clip_p50':>10}"
        )
        clip_rows = {"all": {"frames": len(every_frame), **report["writer_clip_summary"]}, **report["phase_table"]}
        for phase, row in clip_rows.items():
            print(
                f"{phase:>10} {row['frames']:>7} {row['write_grad_norm_preclip_min']:>10.3g} "
                f"{row['write_grad_norm_preclip_median']:>10.3g} "
                f"{row['write_grad_norm_preclip_p95']:>10.3g} "
                f"{row['write_grad_norm_preclip_max']:>10.3g} "
                f"{row['write_clip_saturation_fraction']:>9.3f} "
                f"{row['write_clip_factor_median']:>10.3g}"
            )
        print(
            "\nread-injection RMS (first pre-write frame excluded when carried reads exist)\n"
            f"{'phase':>10} {'used':>7} {'h8_p50':>10} {'image':>10} {'context':>10} "
            f"{'ret_p50':>10} {'inj_p50':>10}"
        )
        scale_rows = {
            "all": report["retrieval_scale_summary"],
            **report["phase_table"],
        }
        for phase, row in scale_rows.items():
            print(
                f"{phase:>10} {row['retrieval_scale_frame_count']:>7} {row['h8_valid_rms_median']:>10.3g} "
                f"{row['h8_image_rms_median']:>10.3g} {row['h8_context_valid_rms_median']:>10.3g} "
                f"{row['retrieved_rms_median']:>10.3g} {row['memory_token_rms_median']:>10.3g}"
            )
        print(
            "\nread-injection constants (c_raw replaces gate; c_extra multiplies gated retrieval)\n"
            f"{'phase':>10} {'used':>7} {'c_p05':>10} {'c_p50':>10} {'c_p95':>10} "
            f"{'c_energy':>10} {'c_extra':>10} {'inj/h8':>10}"
        )
        for phase, row in scale_rows.items():
            c50 = row["retrieval_match_c_median"]
            c05 = row["retrieval_match_c_p05"]
            c95 = row["retrieval_match_c_p95"]
            c_energy = row["retrieval_match_c_energy"]
            c_extra = row["gated_match_c_extra_median"]
            injected_ratio = row["memory_token_to_h8_ratio_median"]
            print(
                f"{phase:>10} {row['retrieval_scale_frame_count']:>7} "
                f"{c05 if c05 is not None else float('nan'):>10.3g} "
                f"{c50 if c50 is not None else float('nan'):>10.3g} "
                f"{c95 if c95 is not None else float('nan'):>10.3g} "
                f"{c_energy if c_energy is not None else float('nan'):>10.3g} "
                f"{c_extra if c_extra is not None else float('nan'):>10.3g} "
                f"{injected_ratio if injected_ratio is not None else float('nan'):>10.3g}"
            )
        print(f"wrote {self.options.output_dir}/summary.json")
        return report
