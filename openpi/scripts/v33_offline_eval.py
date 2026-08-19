"""Offline v3.3 eval on labeled dataset episodes: predicted subtask vs ground truth, with the
zero-read memory ablation, rendered to MP4.

Unlike the raw-demo eval scripts, this runs on LeRobot episodes that carry per-frame subtask
labels, so every prediction is SCORED, not just displayed. For each episode it replays frames at
the training write cadence, threading the real Titans memory (each step reads M_{t-1}, decodes
the subtask, then writes), and at every step also decodes a counterfactual with ``zero_read``:
the identical frame and cache with the retrieved memory zeroed, committed nowhere.

That pairing is the actual test of the memory. During the waiting phase the answer is not
visible, so:

* NORMAL correct and ZERO-READ wrong  => the answer came from memory (what v3.3 wants);
* both correct                        => the model does not need memory here (shortcut or
                                         leakage from the current frame);
* both wrong                          => it has not learned the phase at all.

Outputs per episode in ``--output_dir``:
  * ``episode_XXX_eval.mp4`` -- every frame of the top camera at 30 fps with the held
    prediction, the ground-truth label, the zero-read counterfactual, phase, and a running
    waiting-phase scoreboard burned in (H.264);
  * ``episode_XXX.json`` -- per-prediction records (labels, both decodes, side extraction,
    memory-state norms, latency);
and a run-level ``summary.json`` + printed table with per-phase accuracy for both conditions.

Usage (GPU, forward only):
    uv run scripts/v33_offline_eval.py \
        --checkpoint checkpoints/pi05_yam_mem_v33/v33_run1/6250 \
        --dataset_root ~/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask \
        --output_dir diagnostic_outputs/v33_offline_eval/6250
"""

# ruff: noqa: SLF001, I001 - reuses the private v3.1/v3.2 replay helpers, and pyarrow.parquet
# must import before the openpi stack (see v33_writer_attention for the segfault details).

from __future__ import annotations

import pyarrow.parquet as pq

import dataclasses
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
import tyro

from openpi import transforms as _transforms
from openpi.diagnostics import token_heatmap
from openpi.diagnostics import v33_writer_attention as _wa
from openpi.diagnostics import writer_contribution as _wc
from openpi.diagnostics.v32_checkpoint import _align_params
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import nnx_utils
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

SCHEMA_VERSION = "openpi.v33.offline_eval.v1"
VIDEO_FPS = 30.0


@dataclasses.dataclass(frozen=True)
class Options:
    checkpoint: Path
    dataset_root: Path
    output_dir: Path
    config: str = "pi05_yam_mem_v33"
    # Empty selects one usable episode per (instruction, side) cell.
    episode_indices: tuple[int, ...] = ()
    stride: int | None = None
    max_decode_steps: int = 12
    num_denoise_steps: int = 10
    seed: int = 0
    # Render every raw frame (30 fps, prediction held between steps). False renders only the
    # prediction frames, which is much faster but choppy.
    full_rate_video: bool = True


def _side_of(label: str) -> str:
    """left/right if the label commits to a side, else '-'."""
    low = label.lower()
    if "left" in low:
        return "left"
    if "right" in low:
        return "right"
    return "-"


def _phase_of(frame: int, evidence: tuple[int, int], memory: tuple[int, int]) -> str:
    if frame < evidence[0]:
        return "approach"
    if frame <= evidence[1]:
        return "evidence"
    if frame < memory[0]:
        return "retention"
    if frame <= memory[1]:
        return "waiting"
    return "execute"


class V33OfflineEval:
    def __init__(self, options: Options):
        self.options = options
        if options.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite output directory: {options.output_dir}")
        if not (options.checkpoint / "params").is_dir():
            raise FileNotFoundError(f"checkpoint has no params item: {options.checkpoint}")

        self.train_config = _config.get_config(options.config)
        model_config = self.train_config.model
        self.data_config = self.train_config.data.create(self.train_config.assets_dirs, model_config)
        self.stride = int(self.data_config.memory_stride_frames) if options.stride is None else options.stride

        norm_stats = None
        for candidate in (
            options.checkpoint / "assets" / self.data_config.asset_id,
            self.train_config.assets_dirs / self.data_config.asset_id,
        ):
            if candidate.is_dir():
                norm_stats = _normalize.load(candidate)
                break
        if norm_stats is None:
            raise FileNotFoundError("no norm stats found for this config/checkpoint")

        import flax.nnx as nnx
        import jax
        import jax.numpy as jnp

        raw_params = _model.restore_params(options.checkpoint / "params", dtype=jnp.float32)
        self.scalars = _wa.pathway_scalars(raw_params)
        abstract_state = nnx.state(nnx.eval_shape(model_config.create, jax.random.key(0))).to_pure_dict()
        self.model = model_config.load(_align_params(abstract_state, raw_params), remove_extra_params=False)
        self.model.eval()

        self.tokenizer = _tokenizer.FASTSubtaskTokenizer(model_config.max_token_len)._paligemma_tokenizer
        self.stop_token = int(self.tokenizer.encode("placeholder subtask\n")[-1])

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
        self._sample = nnx_utils.module_jit(
            self.model.sample_with_memory,
            static_argnames=("stop_token", "max_decode_steps", "num_steps", "zero_read", "allow_write"),
        )

    def _observation(self, row: dict[str, Any], frame: int, prompt: str):
        import jax
        import jax.numpy as jnp

        decode = _wc.WriterContributionRunner._decode_inline_image
        transformed = self.input_transform(
            {
                "observation/image": decode(row["image"], field="image", raw_frame=frame),
                "observation/left_wrist_image": decode(
                    row["left_wrist_image"], field="left_wrist_image", raw_frame=frame
                ),
                "observation/right_wrist_image": decode(
                    row["right_wrist_image"], field="right_wrist_image", raw_frame=frame
                ),
                "observation/state": np.asarray(row["state"], dtype=np.float32),
                "prompt": prompt,
            }
        )
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
        return _model.Observation.from_dict(batched)

    def _decode(self, aux: dict[str, Any]) -> str:
        tokens = np.asarray(aux["tokens"])[0]
        mask = np.asarray(aux["token_mask"])[0]
        return self.tokenizer.decode(tokens[mask].tolist()).strip()

    def _forced_buffers(self, text: str):
        import jax.numpy as jnp

        tokens = self.tokenizer.encode(text.strip() + "\n")
        if not 0 < len(tokens) <= self.model.causal_token_len:
            raise ValueError(f"subtask {text!r} has invalid token count {len(tokens)}")
        padded = np.zeros((1, self.model.causal_token_len), dtype=np.int32)
        mask = np.zeros_like(padded, dtype=bool)
        padded[0, : len(tokens)] = tokens
        mask[0, : len(tokens)] = True
        return jnp.asarray(padded), jnp.asarray(mask)

    def _side_logp_margins(self, observation, memory_state, key) -> dict[str, float]:
        """Teacher-forced log-probability of "left" vs "right" waiting labels, with and without
        the memory read.

        The retrieval is multiplied by a small content gate, so a greedy decode can be identical
        while the memory still shifts the logits. The margin (logp[true side] - logp[other
        side]) with the read minus the same margin without it isolates exactly how much the
        memory contributes to the side decision, below the argmax threshold.
        """
        margins = {}
        for name, zero_read in (("read", False), ("zero", True)):
            logps = {}
            for side in ("left", "right"):
                tokens, mask = self._forced_buffers(f"wait; target bin is {side}")
                _, _, aux = self._sample(
                    key,
                    observation,
                    memory_state,
                    stop_token=self.stop_token,
                    max_decode_steps=1,
                    num_steps=self.options.num_denoise_steps,
                    forced_subtask_tokens=tokens,
                    forced_subtask_mask=mask,
                    zero_read=zero_read,
                    allow_write=False,
                )
                logps[side] = float(np.asarray(aux["conditioned_subtask_logp"])[0])
            margins[f"logp_left_{name}"] = logps["left"]
            margins[f"logp_right_{name}"] = logps["right"]
        margins["side_margin_read"] = margins["logp_left_read"] - margins["logp_right_read"]
        margins["side_margin_zero"] = margins["logp_left_zero"] - margins["logp_right_zero"]
        # how much the memory read moves the left-vs-right decision, in nats
        margins["memory_margin_shift"] = margins["side_margin_read"] - margins["side_margin_zero"]
        return {k: round(v, 4) for k, v in margins.items()}

    def _donor_state(self, plan: _wa._EpisodePlan, plans: list[_wa._EpisodePlan], tasks: dict[int, str]):
        """Build the end-of-evidence memory state of an episode with the OPPOSITE side, for the
        swap condition. Writes are committed exactly as in the normal path, then frozen."""
        donor = next((p for p in plans if p.side != plan.side and p.side in ("left", "right")), None)
        if donor is None:
            return None, "-"
        source = _wc._load_lerobot_sources(self.options.dataset_root, [donor.episode])[0]
        columns = ["image", "left_wrist_image", "right_wrist_image", "state", "frame_index", "task_index"]
        by_frame = {int(row["frame_index"]): row for row in pq.read_table(source.path, columns=columns).to_pylist()}
        import jax

        state = self.model.memory.init_state(1)
        key = jax.random.key(self.options.seed + 1)
        # replay the donor up to the end of its evidence phase: the memory then holds its answer
        for frame in sorted(by_frame):
            if frame % self.stride or frame > donor.evidence[1]:
                continue
            observation = self._observation(by_frame[frame], frame, donor.prompt)
            key, step_key = jax.random.split(key)
            _, state, _ = self._sample(
                step_key,
                observation,
                state,
                stop_token=self.stop_token,
                max_decode_steps=1,
                num_steps=self.options.num_denoise_steps,
                zero_read=False,
                allow_write=True,
            )
        print(f"  swap donor: episode {donor.episode} (side {donor.side}) replayed to frame {donor.evidence[1]}")
        return state, donor.side

    def _run_episode(
        self,
        plan: _wa._EpisodePlan,
        tasks: dict[int, str],
        donor_state: Any = None,
        donor_side: str = "-",
    ) -> dict[str, Any]:
        import jax

        source = _wc._load_lerobot_sources(self.options.dataset_root, [plan.episode])[0]
        columns = ["image", "left_wrist_image", "right_wrist_image", "state", "frame_index", "task_index"]
        rows = pq.read_table(source.path, columns=columns).to_pylist()
        by_frame = {int(row["frame_index"]): row for row in rows}

        memory_state = self.model.memory.init_state(1)
        rng = jax.random.key(self.options.seed)
        records: list[dict[str, Any]] = []
        started = time.monotonic()
        for frame in sorted(by_frame):
            if frame % self.stride:
                continue
            row = by_frame[frame]
            observation = self._observation(row, frame, plan.prompt)
            rng, key_normal, key_zero, key_swap = jax.random.split(rng, 4)

            step_started = time.monotonic()
            # NORMAL: reads the live memory and commits its write (this advances the episode).
            _, new_state, aux = self._sample(
                key_normal,
                observation,
                memory_state,
                stop_token=self.stop_token,
                max_decode_steps=self.options.max_decode_steps,
                num_steps=self.options.num_denoise_steps,
                zero_read=False,
                allow_write=True,
            )
            # ZERO-READ counterfactual: identical frame/state, retrieval zeroed, nothing written.
            _, _, aux_zero = self._sample(
                key_zero,
                observation,
                memory_state,
                stop_token=self.stop_token,
                max_decode_steps=self.options.max_decode_steps,
                num_steps=self.options.num_denoise_steps,
                zero_read=True,
                allow_write=False,
            )
            # SWAP: the same frame reading a DONOR episode's memory (opposite side). Zero-read
            # only removes the memory; the swap replaces it with a contradicting one, which is
            # the stronger test -- a model using memory must follow the donor.
            aux_swap = None
            if donor_state is not None:
                _, _, aux_swap = self._sample(
                    key_swap,
                    observation,
                    donor_state,
                    stop_token=self.stop_token,
                    max_decode_steps=self.options.max_decode_steps,
                    num_steps=self.options.num_denoise_steps,
                    zero_read=False,
                    allow_write=False,
                )
            latency = time.monotonic() - step_started

            truth = tasks[int(row["task_index"])]
            predicted = self._decode(aux)
            predicted_zero = self._decode(aux_zero)
            predicted_swap = self._decode(aux_swap) if aux_swap is not None else ""
            record = {
                "frame": frame,
                "phase": _phase_of(frame, plan.evidence, plan.memory),
                "truth": truth,
                "pred": predicted,
                "pred_zero_read": predicted_zero,
                "correct": predicted == truth,
                "correct_zero_read": predicted_zero == truth,
                "truth_side": _side_of(truth),
                "pred_side": _side_of(predicted),
                "pred_zero_side": _side_of(predicted_zero),
                "latency_s": round(latency, 3),
            }
            if aux_swap is not None:
                record.update(
                    {
                        "pred_swap": predicted_swap,
                        "pred_swap_side": _side_of(predicted_swap),
                        "swap_donor_side": donor_side,
                        # follows_swap: did contradicting memory flip the answer to the donor's
                        # side? The signature of a model that actually reads its memory.
                        "follows_swap": _side_of(predicted_swap) == donor_side and donor_side != "-",
                    }
                )
            # Sub-threshold influence: the retrieval is gated by a small scalar, so the greedy
            # decode can be identical while the memory still moves the logits. Compare the
            # forced log-probability of both side answers with and without the read.
            record.update(self._side_logp_margins(observation, memory_state, key_normal))
            records.append(record)
            memory_state = new_state
        elapsed = time.monotonic() - started
        print(
            f"episode {plan.episode} ({plan.prompt} / target {plan.side}): "
            f"{len(records)} predictions in {elapsed:.1f}s",
            flush=True,
        )
        return {"plan": plan, "records": records, "by_frame": by_frame}

    @staticmethod
    def _phase_scores(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
        out: dict[str, dict[str, float]] = {}
        for phase in ("approach", "evidence", "retention", "waiting", "execute"):
            rows = [r for r in records if r["phase"] == phase]
            if not rows:
                continue
            sided = [r for r in rows if r["truth_side"] != "-"]
            swapped = [r for r in rows if "follows_swap" in r and r.get("swap_donor_side", "-") != "-"]
            shifts = [r["memory_margin_shift"] for r in rows if "memory_margin_shift" in r]
            out[phase] = {
                "n": len(rows),
                "exact": float(np.mean([r["correct"] for r in rows])),
                "exact_zero_read": float(np.mean([r["correct_zero_read"] for r in rows])),
                "side_acc": float(np.mean([r["pred_side"] == r["truth_side"] for r in sided]))
                if sided
                else float("nan"),
                "side_acc_zero_read": float(np.mean([r["pred_zero_side"] == r["truth_side"] for r in sided]))
                if sided
                else float("nan"),
                # fraction of steps where a contradicting donor memory flipped the answer to it
                "follows_swap": float(np.mean([r["follows_swap"] for r in swapped])) if swapped else float("nan"),
                # mean |shift| of the left-vs-right logit margin caused by the read (nats):
                # nonzero here with zero accuracy change = sub-threshold memory influence
                "abs_margin_shift": float(np.mean(np.abs(shifts))) if shifts else float("nan"),
            }
        return out

    def _render(self, result: dict[str, Any], path: Path) -> str:
        plan: _wa._EpisodePlan = result["plan"]
        records = result["records"]
        by_frame = result["by_frame"]
        by_pred = {r["frame"]: r for r in records}
        last_frame = max(r["frame"] for r in records)
        frames_to_render = (
            [f for f in sorted(by_frame) if f <= last_frame]
            if self.options.full_rate_video
            else [r["frame"] for r in records]
        )

        video_frames = []
        held: dict[str, Any] | None = None
        waiting_hits = waiting_total = waiting_zero_hits = 0
        for frame in frames_to_render:
            if frame in by_pred:
                held = by_pred[frame]
                if held["phase"] == "waiting":
                    waiting_total += 1
                    waiting_hits += int(held["correct"])
                    waiting_zero_hits += int(held["correct_zero_read"])
            raw = _wc.WriterContributionRunner._decode_inline_image(
                by_frame[frame]["image"], field="image", raw_frame=frame
            )
            model_rgb, _ = token_heatmap.raw_top_camera_to_model_rgb(raw)
            canvas = cv2.resize(model_rgb, (672, 672), interpolation=cv2.INTER_LINEAR)
            panel = np.zeros((256, 672, 3), dtype=np.uint8)

            truth = held["truth"] if held else "-"
            pred = held["pred"] if held else "-"
            pred_zero = held["pred_zero_read"] if held else "-"
            phase = held["phase"] if held else "-"
            ok = bool(held and held["correct"])
            ok_zero = bool(held and held["correct_zero_read"])
            lines = [
                (f"ep{plan.episode}  f{frame}  [{phase}]  {plan.prompt}  (target: {plan.side})", (235, 235, 235)),
                (f"truth : {truth}", (235, 235, 235)),
                (f"pred  : {pred}", (90, 235, 90) if ok else (90, 90, 245)),
                (f"no-mem: {pred_zero}", (90, 235, 90) if ok_zero else (90, 90, 245)),
            ]
            if held and "pred_swap" in held:
                lines.append(
                    (
                        f"swap  : {held['pred_swap']}  (donor side {held['swap_donor_side']})",
                        (235, 160, 60) if held["follows_swap"] else (170, 170, 170),
                    )
                )
            if held and "memory_margin_shift" in held:
                lines.append(
                    (
                        f"memory shifts L-vs-R margin by {held['memory_margin_shift']:+.3f} nats",
                        (200, 200, 235),
                    )
                )
            if waiting_total:
                lines.append(
                    (
                        f"waiting so far: memory {waiting_hits}/{waiting_total}   "
                        f"zero-read {waiting_zero_hits}/{waiting_total}",
                        (235, 210, 90),
                    )
                )
            for index, (text, color) in enumerate(lines):
                cv2.putText(
                    panel, text[:78], (10, 26 + index * 33), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA
                )
            video_frames.append(np.concatenate([canvas, panel], axis=0))

        fps = VIDEO_FPS if self.options.full_rate_video else VIDEO_FPS / self.stride
        return token_heatmap.encode_mp4(video_frames, path, fps)

    def run(self) -> dict[str, Any]:
        print(f"pathway scalars: {self.scalars}", flush=True)
        tasks = _wa._read_tasks(self.options.dataset_root)
        plans = _wa._plan_episodes(
            _wa.Options(
                checkpoint=self.options.checkpoint,
                dataset_root=self.options.dataset_root,
                output_dir=self.options.output_dir,
                config=self.options.config,
                episode_indices=self.options.episode_indices,
            ),
            self.data_config,
        )
        print(f"evaluating {len(plans)} episodes: {[(p.episode, p.prompt, p.side) for p in plans]}", flush=True)
        self.options.output_dir.mkdir(parents=True)

        report: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "checkpoint": str(self.options.checkpoint),
            "stride": self.stride,
            "pathway_scalars": self.scalars,
            "episodes": [],
        }
        every: list[dict[str, Any]] = []
        for plan in plans:
            donor_state, donor_side = self._donor_state(plan, plans, tasks)
            result = self._run_episode(plan, tasks, donor_state=donor_state, donor_side=donor_side)
            stem = f"episode_{plan.episode:03d}"
            encoder = self._render(result, self.options.output_dir / f"{stem}_eval.mp4")
            entry = {
                "episode": plan.episode,
                "prompt": plan.prompt,
                "side": plan.side,
                "evidence": list(plan.evidence),
                "memory": list(plan.memory),
                "video_encoder": encoder,
                "phase_scores": self._phase_scores(result["records"]),
                "records": result["records"],
            }
            (self.options.output_dir / f"{stem}.json").write_text(json.dumps(entry, indent=2))
            report["episodes"].append({k: v for k, v in entry.items() if k != "records"})
            every.extend(result["records"])

        report["phase_scores"] = self._phase_scores(every)
        (self.options.output_dir / "summary.json").write_text(json.dumps(report, indent=2))

        print(
            f"\n{'phase':>10} {'n':>5} {'exact':>8} {'exact(0read)':>13} {'side':>8} "
            f"{'side(0read)':>12} {'follows_swap':>13} {'|margin shift|':>15}"
        )
        for phase, row in report["phase_scores"].items():
            print(
                f"{phase:>10} {row['n']:>5} {row['exact']:>8.2f} {row['exact_zero_read']:>13.2f} "
                f"{row['side_acc']:>8.2f} {row['side_acc_zero_read']:>12.2f} "
                f"{row['follows_swap']:>13.2f} {row['abs_margin_shift']:>15.4f}"
            )
        waiting = report["phase_scores"].get("waiting")
        if waiting:
            delta = waiting["exact"] - waiting["exact_zero_read"]
            print(
                f"\nWAITING-PHASE MEMORY EFFECT: {delta:+.2f} exact-match "
                f"({waiting['exact']:.2f} with memory vs {waiting['exact_zero_read']:.2f} with the read zeroed)"
            )
            print("  > 0 means the answer came from memory; ~0 means the prediction did not need it.")
        print(f"wrote {self.options.output_dir}/summary.json", flush=True)
        return report


if __name__ == "__main__":
    V33OfflineEval(tyro.cli(Options)).run()
