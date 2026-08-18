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
run-level JSON summary (per-phase aggregates are also printed as a table).
"""

# ruff: noqa: SLF001 - this diagnostic deliberately reuses the private v3.1/v3.2 replay
# machinery (episode sources, inline-image decode, checkpoint alignment); those helpers ARE the
# shared implementation, re-exporting them publicly for one consumer would be churn.

from __future__ import annotations

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

SCHEMA_VERSION = "openpi.v33.writer_attention.v1"
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
    import pyarrow.parquet as pq

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


class V33WriterAttentionRunner:
    def __init__(self, options: Options):
        self.options = options
        if options.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite output directory: {options.output_dir}")
        if not (options.checkpoint / "params").is_dir():
            raise FileNotFoundError(f"checkpoint has no params item: {options.checkpoint}")

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

        raw_params = _model.restore_params(options.checkpoint / "params", dtype=jnp.float32)
        self.scalars = pathway_scalars(raw_params)
        abstract_state = nnx.state(nnx.eval_shape(model_config.create, jax.random.key(0))).to_pure_dict()
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
        self._qstep = nnx_utils.module_jit(self.model.v32_query_attention_step)
        self._write = nnx_utils.module_jit(self.model.memory.write)

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
        import pyarrow.parquet as pq

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
                }
            )
            for key, value in (("true", true_map), ("cf", cf_map), ("base", base_map), ("read", read_map)):
                maps[key].append(value.astype(np.float16))
            images.append(model_image)
            memory_state = self._write(memory_state, out_true["write_tokens"])[0]
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

    @staticmethod
    def _phase_table(all_frames: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        table: dict[str, dict[str, float]] = {}
        for phase in ("approach", "evidence", "retention", "waiting"):
            rows = [f for f in all_frames if f["phase"] == phase]
            if not rows:
                continue
            table[phase] = {
                "frames": len(rows),
                "js_mean": float(np.mean([f["js_mean"] for f in rows])),
                "js_max": float(np.max([f["js_max"] for f in rows])),
                "delta_query_cf_mean": float(np.mean([f["delta_query_cf"] for f in rows])),
                "delta_query_base_mean": float(np.mean([f["delta_query_base"] for f in rows])),
                "left_mass_true_mean": float(np.mean([f["left_mass_true"] for f in rows])),
                "entropy_true_mean": float(np.mean([f["entropy_true"] for f in rows])),
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
            "pathway_scalars": self.scalars,
            "episodes": [],
        }
        every_frame: list[dict[str, Any]] = []
        for plan in plans:
            replay = self._replay(plan, tasks)
            stem = f"episode_{plan.episode:03d}"
            encoder = self._render(replay, self.options.output_dir / f"{stem}.mp4")
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
                    "phase_table": self._phase_table(replay["frames"]),
                    "frames": replay["frames"],
                }
            )
            every_frame.extend(replay["frames"])

        report["phase_table"] = self._phase_table(every_frame)
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
        print(f"wrote {self.options.output_dir}/summary.json")
        return report
