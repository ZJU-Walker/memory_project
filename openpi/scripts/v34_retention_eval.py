"""v3.4 three-way retention control (V34_PLAN_final.md 8.4) on the held-out episodes.

For each held-out episode the memory is built NORMALLY through the evidence phase (up to the
closure frame = last evidence frame), then the occlusion (retention + waiting) is crossed
under three write regimes:

  A. Frozen        (write_mode="frozen"):        M_t = M_{t-1}, S_t = S_{t-1}
  B. Dynamics-only (write_mode="dynamics_only"): S_t = eta S_{t-1}, M_t = (1-alpha) M_{t-1} + S_t
  C. Normal        (write_mode="normal"):        full Titans update

At every waiting-phase grid frame each branch scores the teacher-forced left-vs-right waiting
labels (margins in nats) with the branch's own pre-write state; branch C additionally records
the zero-read margins (the consumption control).

Interpretation table (plan 8.4):
    Frozen fail,               Normal fail -> reader/query drift (cross-check ladder rung 2)
    Frozen pass, Dyn fail,     Normal fail -> passive dynamics destroy retention (alpha/eta)
    Frozen pass, Dyn pass,     Normal fail -> NEW-WRITE INTERFERENCE: the v3.4.1 trigger
    all pass                                -> retention healthy

Run:
    uv run scripts/v34_retention_eval.py \\
        --checkpoint checkpoints/pi05_yam_mem_v34/<exp>/<step> \\
        --dataset_root ~/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask \\
        --output_dir diagnostic_outputs/v34_retention/<step> \\
        --config pi05_yam_mem_v34

Always pass the checkpoint's exact config identity. In particular, run5 checkpoints require
``--config pi05_yam_mem_v34_run5_eta0`` because eta_scale is not stored in checkpoint arrays.
"""

# ruff: noqa: SLF001, I001 - the pyarrow-before-openpi import order is load-bearing
import pyarrow.parquet as pq

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import tyro

import v33_offline_eval as _base
import openpi.diagnostics.v33_write_token_probe as _probe
import openpi.diagnostics.writer_contribution as _wc

BRANCHES = ("frozen", "dynamics_only", "normal")
DEFAULT_HELDOUT = (15, 29, 44, 59)


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint: Path
    dataset_root: Path
    output_dir: Path
    # Required because MemoryConfig fields such as eta_scale and blank_initial_output are static
    # GraphDef semantics, not checkpoint arrays. A run5 checkpoint loaded with the v34 default
    # would restore successfully while silently evaluating eta~0.9 instead of eta=0.
    config: str = tyro.MISSING
    episodes: tuple[int, ...] = DEFAULT_HELDOUT
    pass_threshold: float = 0.9  # waiting-frame side accuracy for a branch to "pass"
    num_denoise_steps: int = 3  # actions are not evaluated here; keep the chunk cheap
    seed: int = 0

    def __post_init__(self) -> None:
        if self.config is tyro.MISSING or not self.config.strip():
            raise ValueError("--config is required because checkpoint arrays do not encode static memory semantics.")


class V34RetentionEval(_base.V33OfflineEval):
    def __init__(self, args: Args):
        super().__init__(
            _base.Options(
                checkpoint=args.checkpoint,
                dataset_root=args.dataset_root,
                output_dir=args.output_dir,
                config=args.config,
                episode_indices=args.episodes,
                num_denoise_steps=args.num_denoise_steps,
                seed=args.seed,
            )
        )
        self.args = args

    def _advance(self, key, observation, state, write_mode):
        """One production step under the given write regime. The write tokens come from the
        prefix interface and never depend on the causal region, so a minimal one-token forced
        decode is enough to advance the state exactly as deployment would."""
        import jax.numpy as jnp

        forced = np.zeros((1, self.model.causal_token_len), dtype=np.int32)
        forced_mask = np.zeros_like(forced, dtype=bool)
        forced[0, 0] = self.stop_token
        forced_mask[0, 0] = True
        _, new_state, _ = self._sample(
            key,
            observation,
            state,
            stop_token=self.stop_token,
            max_decode_steps=1,
            num_steps=self.args.num_denoise_steps,
            forced_subtask_tokens=jnp.asarray(forced),
            forced_subtask_mask=jnp.asarray(forced_mask),
            zero_read=False,
            write_mode=write_mode,
        )
        return new_state

    def run(self) -> dict[str, Any]:
        import jax

        tasks = _base._wa._read_tasks(self.options.dataset_root)
        plans = _probe._plan_all_episodes(self.options, self.data_config)
        wanted = set(self.args.episodes)
        plans = [plan for plan in plans if plan.episode in wanted]
        found = [plan.episode for plan in plans]
        if set(found) != wanted or len(found) != len(wanted):
            raise ValueError(
                "retention evaluation must cover every requested episode exactly once: "
                f"requested={sorted(wanted)}, found={sorted(found)}"
            )
        rng = jax.random.key(self.args.seed)
        report: dict[str, Any] = {
            "episodes": [],
            "checkpoint": str(self.options.checkpoint),
            "config": self.args.config,
            # eta_scale is static GraphDef state and is not encoded in checkpoint arrays. Record
            # it explicitly so a numerically valid report cannot silently use run4 semantics.
            "memory_eta_scale": float(self.model.memory.config.eta_scale),
        }

        for plan in plans:
            source = _wc._load_lerobot_sources(self.options.dataset_root, [plan.episode])[0]
            rows = pq.read_table(
                source.path,
                columns=["image", "left_wrist_image", "right_wrist_image", "state", "frame_index", "task_index"],
            ).to_pylist()
            grid = [row for row in rows if int(row["frame_index"]) % self.stride == 0]
            closure = plan.evidence[1]
            truth = plan.side
            other = "right" if truth == "left" else "left"
            print(
                f"episode {plan.episode} ({plan.prompt} / {truth}): closure frame {closure}, "
                f"waiting {plan.memory}",
                flush=True,
            )

            # Phase 1: build the memory normally through the evidence phase.
            state = self.model.memory.init_state(1)
            for row in grid:
                frame = int(row["frame_index"])
                if frame > closure:
                    break
                rng, key = jax.random.split(rng)
                observation = self._observation(row, frame, plan.prompt)
                state = self._advance(key, observation, state, "normal")

            # Phase 2: cross the occlusion under the three regimes.
            branch_states = dict.fromkeys(BRANCHES, state)
            waiting_rows: list[dict[str, Any]] = []
            for row in grid:
                frame = int(row["frame_index"])
                if frame <= closure or frame > plan.memory[1]:
                    continue
                rng, key = jax.random.split(rng)
                observation = self._observation(row, frame, plan.prompt)
                in_waiting = plan.memory[0] <= frame <= plan.memory[1]
                record: dict[str, Any] = {"frame": frame, "in_waiting": in_waiting}
                for branch in BRANCHES:
                    if in_waiting:
                        margins = self._side_logp_margins(observation, branch_states[branch], key)
                        margin_true = (
                            margins["side_margin_read"] if truth == "left" else -margins["side_margin_read"]
                        )
                        record[branch] = {
                            "margin_true_minus_other": round(margin_true, 4),
                            "correct": margin_true > 0,
                        }
                        if branch == "normal":
                            zero_margin = (
                                margins["side_margin_zero"] if truth == "left" else -margins["side_margin_zero"]
                            )
                            record["zero_read"] = {
                                "margin_true_minus_other": round(zero_margin, 4),
                                "correct": zero_margin > 0,
                            }
                            record["memory_margin_shift"] = margins["memory_margin_shift"]
                            record["memory_margin_shift_true_minus_other"] = (
                                margins["memory_margin_shift"]
                                if truth == "left"
                                else -margins["memory_margin_shift"]
                            )
                    branch_states[branch] = self._advance(key, observation, branch_states[branch], branch)
                if in_waiting:
                    waiting_rows.append(record)

            summary = {"episode": plan.episode, "prompt": plan.prompt, "side": truth, "other": other}
            for branch in BRANCHES:
                correct = [r[branch]["correct"] for r in waiting_rows]
                margins = [r[branch]["margin_true_minus_other"] for r in waiting_rows]
                summary[branch] = {
                    "waiting_frames": len(correct),
                    "accuracy": float(np.mean(correct)) if correct else None,
                    "mean_margin": float(np.mean(margins)) if margins else None,
                }
            zero_correct = [r["zero_read"]["correct"] for r in waiting_rows if "zero_read" in r]
            summary["zero_read_accuracy"] = float(np.mean(zero_correct)) if zero_correct else None
            aligned_shifts = [r["memory_margin_shift_true_minus_other"] for r in waiting_rows]
            summary["mean_memory_margin_shift_true_minus_other"] = (
                float(np.mean(aligned_shifts)) if aligned_shifts else None
            )
            summary["frames"] = waiting_rows
            report["episodes"].append(summary)
            print(
                "  waiting accuracy: "
                + ", ".join(f"{b}={summary[b]['accuracy']}" for b in BRANCHES)
                + f", zero_read={summary['zero_read_accuracy']}",
                flush=True,
            )

        threshold = self.args.pass_threshold
        aggregate = {}
        for branch in BRANCHES:
            values = [e[branch]["accuracy"] for e in report["episodes"] if e[branch]["accuracy"] is not None]
            aggregate[branch] = {
                "accuracy": float(np.mean(values)) if values else None,
                "passes": bool(values and np.mean(values) >= threshold),
            }
        zero_values = [e["zero_read_accuracy"] for e in report["episodes"] if e["zero_read_accuracy"] is not None]
        aggregate["zero_read"] = {"accuracy": float(np.mean(zero_values)) if zero_values else None}
        normal_accuracy = aggregate["normal"]["accuracy"]
        zero_accuracy = aggregate["zero_read"]["accuracy"]
        aggregate["normal_minus_zero_read_accuracy"] = (
            float(normal_accuracy - zero_accuracy) if normal_accuracy is not None and zero_accuracy is not None else None
        )
        report["aggregate"] = aggregate
        report["v341_trigger_new_write_interference"] = bool(
            aggregate["frozen"]["passes"]
            and aggregate["dynamics_only"]["passes"]
            and not aggregate["normal"]["passes"]
        )
        print("\n================ THREE-WAY RETENTION (plan 8.4) ================")
        for branch in BRANCHES:
            entry = aggregate[branch]
            print(f"  {branch:>14}: accuracy {entry['accuracy']}  -> {'PASS' if entry['passes'] else 'fail'}")
        print(f"  v3.4.1 selective-write trigger (Dyn pass AND Normal fail): "
              f"{report['v341_trigger_new_write_interference']}")
        self.options.output_dir.mkdir(parents=True, exist_ok=True)
        (self.options.output_dir / "retention_report.json").write_text(json.dumps(report, indent=2))
        del tasks
        return report


def main(args: Args) -> None:
    V34RetentionEval(args).run()


if __name__ == "__main__":
    main(tyro.cli(Args))
