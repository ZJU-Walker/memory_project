"""Render stride-1 free decodes with full online memory for the four run5 heldouts.

Each episode starts from exactly one fresh M0.  Frames are then consumed strictly in parquet
order with batch size one.  Every raw frame reads the current memory and greedily decodes the
subtask.  Frames ``0, 15, 30, ...`` then commit the normal fast-memory write/update; all other
frames use ``write_mode='frozen'`` and must return a bit-exact unchanged state.  The returned
state is threaded to the next raw frame in both cases.  This preserves run5's 15-frame memory
write cadence while still visualizing a model decode on every frame.

The output is intentionally separate from the reset-memory diagnostic.  Every MP4 overlays the
free model decode, the observation-time GT label at ``t``, and the deliberately look-ahead
training label at ``min(t+15, T-1)`` on the exact three 224x224 model-input camera tensors.

Example (run from the OpenPI repository root):

    V34_RUN5_SOURCE_ROOT=/tmp/v34_run5_source \
    PYTHONPATH=/tmp/v34_run5_source/src \
    .venv/bin/python scripts/v34_heldout_online_memory_video.py \
      --checkpoint diagnostic_checkpoints/v34_run5_eta0_pilot_copies/2500 \
      --dataset-root /iris/u/kewalk/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask \
      --output-dir diagnostic_outputs/v34_heldout_online_memory/2500/raw \
      --config pi05_yam_mem_v34_run5_eta0 --parameter-source raw \
      --num-shards 2 --shard-id 0

Use ``--smoke-only`` with a brand-new output directory before the full two-GPU render.
"""

# ruff: noqa: SLF001, I001 - shared audited internals and pyarrow-before-JAX import order are intentional.
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import time
from typing import Any

# Import the audited helper first: it deliberately imports pyarrow before OpenPI/JAX.
import v34_heldout_free_decode_video as fresh

import jax
import jax.numpy as jnp
import numpy as np


SCHEMA_VERSION = "openpi.v34.heldout_scheduled_memory_video.v1"
MEMORY_WRITE_STRIDE = 15
SMOKE_FRAMES_PER_EPISODE = MEMORY_WRITE_STRIDE + 2
EXPECTED_WRITE_COUNTS = {15: 59, 29: 59, 44: 58, 59: 50}
# Normal and frozen write modes are separate static XLA executables. Their FP32 retrieval-RMS
# reductions need not be bit-identical on GPU even though tokens and the input state are exact.
# This matches the repository's established within-run BF16/XLA replay tolerance.
CROSS_STATIC_MODE_RETRIEVAL_ATOL = 1e-5
FINITE_AUX_KEYS = (
    "retrieval_norm",
    "surprise",
    "grad_norm",
    "clip_factor",
    "theta",
    "eta",
    "alpha",
    "memory_gate_norm",
    "read_query_norm",
    "write_token_norm",
)


def _is_scheduled_write(frame: int) -> bool:
    if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
        raise ValueError(f"frame must be a nonnegative integer, got {frame!r}")
    return frame % MEMORY_WRITE_STRIDE == 0


def _expected_write_count(frame_count: int) -> int:
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError(f"frame_count must be a positive integer, got {frame_count!r}")
    return (frame_count + MEMORY_WRITE_STRIDE - 1) // MEMORY_WRITE_STRIDE


def _cross_static_retrieval_comparison(normal: float, frozen: float) -> dict[str, float | bool]:
    normal = float(normal)
    frozen = float(frozen)
    if not math.isfinite(normal) or not math.isfinite(frozen) or normal < 0.0 or frozen < 0.0:
        raise ValueError(f"retrieval controls must be finite and nonnegative, got {normal}/{frozen}")
    difference = abs(normal - frozen)
    return {
        "normal": normal,
        "frozen": frozen,
        "absolute_difference": difference,
        "absolute_tolerance": CROSS_STATIC_MODE_RETRIEVAL_ATOL,
        "passes": difference <= CROSS_STATIC_MODE_RETRIEVAL_ATOL,
    }


@dataclasses.dataclass(frozen=True)
class Args(fresh.Args):
    """Pinned online-memory arguments; recurrence requires a single row per call."""

    batch_size: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.batch_size != 1:
            raise ValueError("online-memory recurrence requires --batch-size 1")


def _parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parameter-source", choices=("raw", "ema"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return Args(**vars(parser.parse_args(argv)))


class HeldoutOnlineMemoryVideo(fresh.HeldoutFreeDecodeVideo):
    """The full read/write/update counterpart to the independent-M0 diagnostic."""

    def __init__(self, args: Args):
        super().__init__(args)
        self.args = args
        if int(self.base.data_config.memory_stride_frames) != MEMORY_WRITE_STRIDE:
            raise ValueError(
                f"run5 memory write stride changed: expected {MEMORY_WRITE_STRIDE}, "
                f"got {self.base.data_config.memory_stride_frames}"
            )
        self._first_frame_contract: dict[str, Any] | None = None

    @staticmethod
    def _state_max_abs(state: Any) -> float:
        leaves = jax.tree.leaves(state)
        if not leaves:
            raise ValueError("memory state has no leaves")
        finite = jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves])
        if not bool(np.asarray(jnp.all(finite))):
            raise RuntimeError("memory state contains NaN or infinity")
        maxima = jnp.stack([jnp.max(jnp.abs(leaf.astype(jnp.float32))) for leaf in leaves])
        value = float(np.asarray(jnp.max(maxima)))
        if not math.isfinite(value):
            raise RuntimeError("memory state max-abs is not finite")
        return value

    @staticmethod
    def _state_bit_exact_equal(left: Any, right: Any) -> bool:
        left_leaves, left_structure = jax.tree_util.tree_flatten(left)
        right_leaves, right_structure = jax.tree_util.tree_flatten(right)
        if left_structure != right_structure or len(left_leaves) != len(right_leaves) or not left_leaves:
            return False
        comparisons = []
        for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
            if left_leaf.shape != right_leaf.shape or left_leaf.dtype != right_leaf.dtype:
                return False
            if left_leaf.dtype != jnp.float32:
                raise TypeError(f"memory state leaf dtype must be float32, got {left_leaf.dtype}")
            left_bits = jax.lax.bitcast_convert_type(left_leaf, jnp.uint32)
            right_bits = jax.lax.bitcast_convert_type(right_leaf, jnp.uint32)
            comparisons.append(jnp.all(left_bits == right_bits))
        return bool(np.asarray(jnp.all(jnp.stack(comparisons))))

    @staticmethod
    def _finite_aux_values(aux: dict[str, Any], *, frame: int) -> dict[str, float]:
        numeric_aux = {}
        for key in FINITE_AUX_KEYS:
            values = np.asarray(aux.get(key), dtype=np.float64)
            if values.shape != (1,) or not np.all(np.isfinite(values)):
                raise RuntimeError(f"frame {frame}: invalid {key}: shape/value={values}")
            numeric_aux[key] = float(values[0])
        if numeric_aux["retrieval_norm"] < 0.0:
            raise RuntimeError(f"frame {frame}: retrieval_norm is negative: {numeric_aux['retrieval_norm']}")
        if numeric_aux["eta"] != 0.0:
            raise RuntimeError(f"frame {frame}: run5 eta intervention is not exact zero: {numeric_aux['eta']}")
        return numeric_aux

    def _assert_online_inference_contract(
        self,
        input_state: Any,
        returned_state: Any,
        aux: dict[str, Any],
        *,
        frame: int,
        commit_write: bool,
    ) -> dict[str, Any]:
        writes = np.asarray(aux.get("write_occurred"))
        if writes.shape != (1,) or bool(writes[0]) != commit_write:
            raise RuntimeError(
                f"frame {frame}: write_occurred mismatch; expected {commit_write}, got {writes}"
            )
        numeric_aux = self._finite_aux_values(aux, frame=frame)
        if frame == 0 and abs(numeric_aux["retrieval_norm"]) > fresh.RETRIEVAL_ZERO_TOL:
            raise RuntimeError(
                f"frame 0: blank run5 M0 retrieval is not zero: {numeric_aux['retrieval_norm']}"
            )
        input_max_abs = self._state_max_abs(input_state)
        returned_max_abs = self._state_max_abs(returned_state)
        state_diff = self._state_diff(returned_state, input_state)
        if not math.isfinite(state_diff):
            raise RuntimeError(f"frame {frame}: state difference is not finite: {state_diff}")
        if commit_write and state_diff <= 0.0:
            raise RuntimeError(
                f"frame {frame}: write_occurred=True but returned memory did not change; max abs diff={state_diff}"
            )
        frozen_bit_exact = not commit_write and self._state_bit_exact_equal(returned_state, input_state)
        if not commit_write and (state_diff != 0.0 or not frozen_bit_exact):
            raise RuntimeError(
                f"frame {frame}: frozen read/decode did not return bit-exact state; max abs diff={state_diff}"
            )
        return {
            "batch_size": 1,
            "frame": frame,
            "frame_is_scheduled_write": _is_scheduled_write(frame),
            "commit_write_this_call": commit_write,
            "write_mode": "normal" if commit_write else "frozen",
            "write_occurred": commit_write,
            **numeric_aux,
            "input_state_max_abs": input_max_abs,
            "returned_state_max_abs": returned_max_abs,
            "returned_state_max_abs_difference": state_diff,
            "returned_state_changed": commit_write,
            "frozen_state_bit_exact_unchanged": frozen_bit_exact,
        }

    def _infer_online(
        self,
        observation: Any,
        memory_state: Any,
        *,
        frame: int,
        commit_write: bool | None = None,
        record_first_frame: bool = True,
    ) -> tuple[Any, Any, dict[str, Any], dict[str, Any]]:
        batch = int(observation.state.shape[0])
        if batch != 1:
            raise ValueError(f"online recurrence requires batch size one, got {batch}")
        noise = jnp.zeros(
            (1, self.base.model.action_horizon, self.base.model.action_dim),
            dtype=jnp.float32,
        )
        scheduled_write = _is_scheduled_write(frame)
        if commit_write is None:
            commit_write = scheduled_write
        elif commit_write != scheduled_write and commit_write:
            raise ValueError(f"frame {frame}: cannot commit an off-schedule write")
        write_mode = "normal" if commit_write else "frozen"
        output = self.base._sample(
            jax.random.key(np.uint32(self.args.seed)),
            observation,
            memory_state,
            stop_token=self.base.stop_token,
            max_decode_steps=fresh.PRODUCTION_MAX_DECODE_STEPS,
            num_steps=fresh.PRODUCTION_NUM_STEPS,
            noise=noise,
            action_prefix=None,
            forced_subtask_tokens=None,
            forced_subtask_mask=None,
            zero_read=False,
            allow_write=commit_write,
            write_mode=write_mode,
        )
        jax.block_until_ready(output)
        actions, returned_state, aux = output
        contract = self._assert_online_inference_contract(
            memory_state,
            returned_state,
            aux,
            frame=frame,
            commit_write=commit_write,
        )
        if record_first_frame and self._first_frame_contract is None:
            self._first_frame_contract = contract
        return actions, returned_state, aux, contract

    def _infer_frozen_control(
        self,
        observation: Any,
        memory_state: Any,
        *,
        frame: int,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Smoke-only control: same prestate/read/decode, but do not commit the final write."""
        actions, returned_state, aux, _contract = self._infer_online(
            observation,
            memory_state,
            frame=frame,
            commit_write=False,
            record_first_frame=False,
        )
        return actions, returned_state, aux

    def _reserve_output(self, payloads: dict[int, dict[str, Any]]) -> Path:
        output = self.args.artifact_dir
        output.mkdir(parents=False, exist_ok=False)
        incomplete = output / "INCOMPLETE.json"
        value = {
            "schema": SCHEMA_VERSION,
            "status": "reserved; online inference/rendering not complete",
            "checkpoint": str(self.args.checkpoint),
            "checkpoint_step": fresh.CHECKPOINT_STEP,
            "internal_train_step": fresh.INTERNAL_TRAIN_STEP,
            "parameter_source": self.args.parameter_source,
            "episodes": list(payloads),
            "shard": {"num_shards": self.args.num_shards, "shard_id": self.args.shard_id},
            "memory_mode": "online_read_every_frame_scheduled_write_stride15",
        }
        incomplete.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        return incomplete

    def _common_summary(self, payloads: dict[int, dict[str, Any]]) -> dict[str, Any]:
        result = super()._common_summary(payloads)
        result["schema"] = SCHEMA_VERSION
        result["source_provenance"] = {
            "exact_run5_openpi_source": self.base.run5_source_provenance,
            "online_diagnostic_script": fresh.causal._file_identity(Path(__file__).resolve()),
            "shared_render_and_runtime_script": fresh.causal._file_identity(Path(fresh.__file__).resolve()),
        }
        result["inference_contract"] = {
            "frame_stride": 1,
            "batch_size": 1,
            "episode_memory_initialization": "exactly one model.memory.init_state(1) at frame 0",
            "strict_frame_order": True,
            "read_timing": "every raw frame t reads the current state before decoding",
            "write_schedule": "commit normal write/update iff frame_index % 15 == 0",
            "write_timing": "scheduled frame t commits after its read/decode",
            "frozen_timing": (
                "all off-schedule frames read/decode with write_mode=frozen and return bit-exact unchanged state"
            ),
            "memory_write_stride_frames": MEMORY_WRITE_STRIDE,
            "online_history_accumulated": True,
            "each_frame_processed_once": True,
            "returned_state_threaded": True,
            "allow_write": "true only on frame_index % 15 == 0; false otherwise",
            "write_mode": "normal on scheduled writes; frozen otherwise",
            "zero_read": False,
            "num_steps": fresh.PRODUCTION_NUM_STEPS,
            "max_decode_steps": fresh.PRODUCTION_MAX_DECODE_STEPS,
            "decoder": "free greedy sample_with_memory",
            "action_noise": "all-zero float32 tensor",
            "rng_seed": self.args.seed,
            "per_frame_assertions": (
                "all read/writer-candidate/gate auxiliaries and state leaves are finite; eta is exactly zero; "
                "scheduled writes occur and change state; frozen frames do not write and return identical state"
            ),
            "cadence_note": "decode/read stride is 1 raw frame; committed memory-write stride matches run5 at 15",
            "smoke_cross_static_mode_control": {
                "scope": "scheduled write frames only, from the exact same input state",
                "exact_invariants": "decoded token ids/masks/status and frozen returned memory state",
                "retrieval_norm_invariant": (
                    "absolute difference between normal/frozen static XLA executables is at most 1e-5"
                ),
                "retrieval_norm_absolute_tolerance": CROSS_STATIC_MODE_RETRIEVAL_ATOL,
                "justification": (
                    "FP32 retrieval RMS is a reduction reached through separately compiled normal/frozen "
                    "BF16/XLA graphs; the repository uses the same 1e-5 within-run replay tolerance"
                ),
            },
            "first_frame_assertions": self._first_frame_contract,
        }
        return result

    def run_smoke(self) -> dict[str, Any]:
        started = time.monotonic()
        payloads = self._load_payloads()
        incomplete = self._reserve_output(payloads)
        checks = []
        reference_m0 = self.base.model.memory.init_state(1)
        self._state_max_abs(reference_m0)
        for episode, payload in payloads.items():
            rows = payload["rows"]
            count = min(SMOKE_FRAMES_PER_EPISODE, len(rows))
            memory_state = self.base.model.memory.init_state(1)
            if self._state_diff(memory_state, reference_m0) != 0.0:
                raise RuntimeError(f"episode {episode}: newly reset M0 differs from the reference M0")
            initial_max_abs = self._state_max_abs(memory_state)
            episode_retrievals = []
            write_count = 0
            for frame in range(count):
                row = rows[frame]
                if int(row["frame_index"]) != frame:
                    raise RuntimeError(f"episode {episode}: smoke recurrence lost strict frame order")
                observation = self._observations([(row, frame, payload["plan"].prompt)])
                scheduled_write = _is_scheduled_write(frame)
                frozen_state = None
                frozen_aux = None
                if scheduled_write:
                    _frozen_actions, frozen_state, frozen_aux = self._infer_frozen_control(
                        observation,
                        memory_state,
                        frame=frame,
                    )
                _actions, returned_state, aux, contract = self._infer_online(
                    observation,
                    memory_state,
                    frame=frame,
                )
                decoded, _tokens, _masks, statuses = self._decode_batch(aux)
                normal_retrieval = float(np.asarray(aux["retrieval_norm"])[0])
                retrieval_control = None
                if scheduled_write:
                    if frozen_aux is None or frozen_state is None:
                        raise AssertionError("scheduled smoke write lacks its frozen same-prestate control")
                    _frozen_decoded, frozen_tokens, frozen_masks, frozen_statuses = self._decode_batch(frozen_aux)
                    normal_tokens = np.asarray(aux["tokens"])
                    normal_masks = np.asarray(aux["token_mask"], dtype=bool)
                    if not self._same_tokens(
                        normal_tokens[0], normal_masks[0], frozen_tokens[0], frozen_masks[0]
                    ):
                        raise RuntimeError(f"episode {episode} frame {frame}: normal/frozen tokens differ")
                    if statuses != frozen_statuses:
                        raise RuntimeError(f"episode {episode} frame {frame}: normal/frozen decode statuses differ")
                    frozen_retrieval = float(np.asarray(frozen_aux["retrieval_norm"])[0])
                    retrieval_control = _cross_static_retrieval_comparison(
                        normal_retrieval,
                        frozen_retrieval,
                    )
                    print(
                        f"[smoke ep{episode} f{frame}] cross-static retrieval control: "
                        f"normal={retrieval_control['normal']:.9g} "
                        f"frozen={retrieval_control['frozen']:.9g} "
                        f"abs_diff={retrieval_control['absolute_difference']:.9g} "
                        f"tol={retrieval_control['absolute_tolerance']:.9g}",
                        flush=True,
                    )
                    if not bool(retrieval_control["passes"]):
                        raise RuntimeError(
                            f"episode {episode} frame {frame}: cross-static normal/frozen retrieval "
                            f"difference exceeds tolerance: {json.dumps(retrieval_control, sort_keys=True)}"
                        )
                    if self._state_diff(frozen_state, memory_state) != 0.0:
                        raise RuntimeError(
                            f"episode {episode} frame {frame}: frozen control is not the identical prestate"
                        )
                if frame == 0 and abs(normal_retrieval) > fresh.RETRIEVAL_ZERO_TOL:
                    raise RuntimeError(
                        f"episode {episode}: blank M0 frame-0 retrieval is not zero: {normal_retrieval}"
                    )
                episode_retrievals.append(normal_retrieval)
                write_count += int(scheduled_write)
                checks.append(
                    {
                        "episode": episode,
                        "frame": frame,
                        "decoded_subtask": decoded[0],
                        **statuses[0],
                        **contract,
                        "same_prestate_normal_vs_frozen_control_run": scheduled_write,
                        "normal_vs_frozen_tokens_equal": True if scheduled_write else None,
                        "cross_static_normal_vs_frozen_retrieval": retrieval_control,
                        "frozen_state_unchanged": (True if not scheduled_write else None),
                        "cumulative_write_count": write_count,
                    }
                )
                memory_state = returned_state
            expected_writes = _expected_write_count(count)
            if write_count != expected_writes:
                raise RuntimeError(
                    f"episode {episode}: smoke write count {write_count} != expected {expected_writes}"
                )
            final_max_abs = self._state_max_abs(memory_state)
            if self._state_diff(memory_state, self.base.model.memory.init_state(1)) <= 0.0:
                raise RuntimeError(f"episode {episode}: smoke state did not depart from M0")
            if count > 1 and not any(value > fresh.RETRIEVAL_ZERO_TOL for value in episode_retrievals[1:]):
                raise RuntimeError(f"episode {episode}: post-write retrieval never became nonzero in smoke prefix")
            checks[-1]["episode_initial_state_max_abs"] = initial_max_abs
            checks[-1]["episode_final_state_max_abs"] = final_max_abs

        report = {
            **self._common_summary(payloads),
            "mode": "smoke-only",
            "status": "pass",
            "frames_per_episode": SMOKE_FRAMES_PER_EPISODE,
            "frame_checks": checks,
            "checks": {
                "episode_M0_initialized_once": True,
                "strict_sequential_state_threading": True,
                "read_enabled": True,
                "scheduled_write_update_enabled": True,
                "write_schedule_f0_f15": True,
                "all_and_only_scheduled_writes_occurred": True,
                "all_off_schedule_frames_frozen_unchanged": True,
                "all_retrieval_norms_finite": True,
                "scheduled_write_states_changed": True,
                "due_frame_normal_vs_frozen_same_decode": True,
                "due_frame_cross_static_retrieval_within_absolute_tolerance": True,
                "cross_static_retrieval_absolute_tolerance": CROSS_STATIC_MODE_RETRIEVAL_ATOL,
                "frozen_control_state_unchanged": True,
                "frame0_blank_M0_retrieval_zero": True,
                "post_write_retrieval_became_nonzero": True,
                "identical_M0_reset_per_episode": True,
            },
            "elapsed_seconds": time.monotonic() - started,
        }
        identity = fresh._write_json_atomic(self.args.artifact_dir / "smoke.json", report)
        fresh._write_complete_atomic(self.args.artifact_dir, {"smoke.json": identity})
        incomplete.unlink()
        print(json.dumps(report["checks"], indent=2), flush=True)
        return report

    def _episode_video(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        plan = payload["plan"]
        episode = int(plan.episode)
        source = payload["source"]
        fps = float(source.control_hz)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"episode {episode} has invalid source fps {fps!r}")
        rows = payload["rows"]
        task_names = payload["task_names"]
        video_name = f"episode_{episode:06d}_online_memory_free_decode.mp4"
        jsonl_name = f"episode_{episode:06d}_online_memory_frames.jsonl"
        video = fresh._AtomicMp4Writer(self.args.artifact_dir / video_name, fps)
        jsonl_path = self.args.artifact_dir / jsonl_name
        jsonl_temporary = jsonl_path.with_name(f".{jsonl_path.name}.tmp")
        if jsonl_path.exists() or jsonl_temporary.exists():
            video.abort()
            raise FileExistsError(f"refusing to overwrite frame records: {jsonl_path}")

        counts: dict[str, int] = {}
        decode_status_counts = {"terminated": 0, "truncated": 0}
        decoded_runs: list[dict[str, Any]] = []
        previous_decoded: str | None = None
        run_start = 0
        retrieval_norms: list[float] = []
        state_update_diffs: list[float] = []
        committed_update_diffs: list[float] = []
        frozen_state_diffs: list[float] = []
        write_count = 0
        memory_state = self.base.model.memory.init_state(1)
        initial_state_max_abs = self._state_max_abs(memory_state)
        started = time.monotonic()
        try:
            with jsonl_temporary.open("w", encoding="utf-8") as jsonl:
                for frame, row in enumerate(rows):
                    if int(row["frame_index"]) != frame:
                        raise RuntimeError(
                            f"episode {episode}: recurrence requires frame {frame}, got {row['frame_index']}"
                        )
                    observation = self._observations([(row, frame, plan.prompt)])
                    _actions, returned_state, aux, contract = self._infer_online(
                        observation,
                        memory_state,
                        frame=frame,
                    )
                    decoded, tokens, masks, statuses = self._decode_batch(aux)
                    prediction = decoded[0]
                    decode_status = statuses[0]
                    target_frame, target_task = fresh._training_target(rows, frame, fresh.TRAIN_TARGET_LOOKAHEAD)
                    gt_now = task_names[int(row["task_index"])]
                    gt_target = task_names[target_task]
                    images = fresh._processed_images(observation, 0)
                    retrieval_norm = float(contract["retrieval_norm"])
                    state_update_diff = float(contract["returned_state_max_abs_difference"])
                    scheduled_write = _is_scheduled_write(frame)
                    write_count_after = write_count + int(scheduled_write)
                    if scheduled_write:
                        memory_caption = (
                            f"ONLINE MEMORY - COMMIT WRITE #{write_count_after} (f{frame}, cadence=15): "
                            f"read prior state RMS={retrieval_norm:.4g} -> decode -> NORMAL write/update; "
                            f"surprise={contract['surprise']:.4g} clip={contract['clip_factor']:.4g} "
                            f"theta/eta/alpha={contract['theta']:.4g}/{contract['eta']:.4g}/"
                            f"{contract['alpha']:.4g}; state delta={state_update_diff:.4g} -> thread M_t"
                        )
                    else:
                        next_write = frame + (MEMORY_WRITE_STRIDE - frame % MEMORY_WRITE_STRIDE)
                        memory_caption = (
                            f"ONLINE MEMORY - READ ONLY (f{frame}; next write=f{next_write}): "
                            f"read RMS={retrieval_norm:.4g} -> decode -> FROZEN / NO COMMITTED UPDATE; "
                            f"state delta={state_update_diff:.4g} (exact unchanged) -> thread same state. "
                            f"Writer values are candidate diagnostics only: surprise={contract['surprise']:.4g}, "
                            f"clip={contract['clip_factor']:.4g}, theta/eta/alpha={contract['theta']:.4g}/"
                            f"{contract['eta']:.4g}/{contract['alpha']:.4g}"
                        )
                    rendered = fresh._render_frame(
                        images,
                        episode=episode,
                        frame=frame,
                        fps=fps,
                        prompt=plan.prompt,
                        decoded=prediction,
                        decoded_token_count=int(decode_status["token_count"]),
                        decoded_terminated=bool(decode_status["terminated"]),
                        decoded_truncated=bool(decode_status["truncated"]),
                        gt_now=gt_now,
                        target_frame=target_frame,
                        gt_train_target=gt_target,
                        parameter_source=self.args.parameter_source,
                        config=self.args.config,
                        memory_caption=memory_caption,
                    )
                    video.write(rendered)
                    record = {
                        "episode": episode,
                        "frame": frame,
                        "time_seconds": frame / fps,
                        "fps": fps,
                        "prompt": plan.prompt,
                        "heldout_side": plan.side,
                        "decoded_subtask": prediction,
                        "decoded_side": fresh.causal._phase_side(prediction),
                        "decoded_token_ids": tokens[0][masks[0]].tolist(),
                        "decoded_token_count": decode_status["token_count"],
                        "decoded_terminated": decode_status["terminated"],
                        "decoded_truncated": decode_status["truncated"],
                        "decode_termination_token": decode_status["termination_token"],
                        "gt_now": gt_now,
                        "gt_now_task_index": int(row["task_index"]),
                        "gt_training_target_frame": target_frame,
                        "gt_training_target": gt_target,
                        "gt_training_target_task_index": target_task,
                        "processed_camera_sha256": {
                            key: fresh.causal._array_sha256(image) for key, image in images.items()
                        },
                        "memory": {
                            "mode": "online_read_every_frame_scheduled_write_stride15",
                            "episode_M0_initialized_once": True,
                            "state_before_frame": "M0" if frame == 0 else f"returned state from frame {frame - 1}",
                            "frames_accumulated_before_read": frame,
                            "writes_accumulated_before_read": (
                                frame + MEMORY_WRITE_STRIDE - 1
                            )
                            // MEMORY_WRITE_STRIDE,
                            "online_history_accumulated": True,
                            "each_frame_processed_once": True,
                            "raw_frame_decode_read_stride": 1,
                            "memory_write_stride": MEMORY_WRITE_STRIDE,
                            "training_memory_step_stride": 15,
                            "write_cadence_matches_training": True,
                            "strict_frame_order": True,
                            "read_enabled": True,
                            "zero_read": False,
                            "retrieval_norm": retrieval_norm,
                            "scheduled_write_frame": scheduled_write,
                            "writes_enabled_this_frame": scheduled_write,
                            "write_mode": "normal" if scheduled_write else "frozen",
                            "write_occurred": scheduled_write,
                            "write_count_after_frame": write_count_after,
                            "candidate_writer_diagnostics": {
                                "note": (
                                    "committed on this scheduled write frame"
                                    if scheduled_write
                                    else "computed by the model but NOT committed on this frozen frame"
                                ),
                                "surprise": contract["surprise"],
                                "grad_norm": contract["grad_norm"],
                                "clip_factor": contract["clip_factor"],
                                "theta": contract["theta"],
                                "eta": contract["eta"],
                                "alpha": contract["alpha"],
                                "memory_gate_norm": contract["memory_gate_norm"],
                                "read_query_norm": contract["read_query_norm"],
                                "write_token_norm": contract["write_token_norm"],
                            },
                            "state_changed_after_write": scheduled_write,
                            "frozen_state_bit_exact_unchanged": contract["frozen_state_bit_exact_unchanged"],
                            "returned_state_max_abs_difference": state_update_diff,
                            "input_state_max_abs": contract["input_state_max_abs"],
                            "returned_state_max_abs": contract["returned_state_max_abs"],
                            "returned_state_threaded_to_next_frame": frame + 1 < len(rows),
                        },
                        "inference": {
                            "parameter_source": self.args.parameter_source,
                            "config": self.args.config,
                            "checkpoint_step": fresh.CHECKPOINT_STEP,
                            "num_steps": fresh.PRODUCTION_NUM_STEPS,
                            "max_decode_steps": fresh.PRODUCTION_MAX_DECODE_STEPS,
                            "action_noise": "zeros",
                        },
                    }
                    jsonl.write(
                        json.dumps(fresh.causal._strict_json(record), sort_keys=True, allow_nan=False) + "\n"
                    )
                    memory_state = returned_state
                    retrieval_norms.append(retrieval_norm)
                    state_update_diffs.append(state_update_diff)
                    if scheduled_write:
                        committed_update_diffs.append(state_update_diff)
                    else:
                        frozen_state_diffs.append(state_update_diff)
                    write_count = write_count_after
                    counts[prediction] = counts.get(prediction, 0) + 1
                    decode_status_counts["terminated" if decode_status["terminated"] else "truncated"] += 1
                    if previous_decoded is None:
                        previous_decoded = prediction
                        run_start = frame
                    elif prediction != previous_decoded:
                        decoded_runs.append(
                            {"start": run_start, "end": frame - 1, "decoded_subtask": previous_decoded}
                        )
                        run_start = frame
                        previous_decoded = prediction
                    print(f"[ep{episode}] online frame {frame}/{len(rows) - 1}", flush=True)
                if previous_decoded is not None:
                    decoded_runs.append(
                        {"start": run_start, "end": len(rows) - 1, "decoded_subtask": previous_decoded}
                    )
                jsonl.flush()
                os.fsync(jsonl.fileno())
            os.replace(jsonl_temporary, jsonl_path)
            video_identity = video.close()
        except BaseException:
            video.abort()
            raise

        expected_writes = _expected_write_count(len(rows))
        registered_expected = EXPECTED_WRITE_COUNTS.get(episode)
        if registered_expected is None or expected_writes != registered_expected:
            raise RuntimeError(
                f"episode {episode}: expected write-count registry/formula mismatch: "
                f"registry={registered_expected}, formula={expected_writes}"
            )
        if write_count != expected_writes:
            raise RuntimeError(f"episode {episode}: expected {expected_writes} writes, observed {write_count}")
        if any(value != 0.0 for value in frozen_state_diffs):
            raise RuntimeError(f"episode {episode}: at least one frozen frame changed memory")
        if len(rows) > 1 and not any(value > fresh.RETRIEVAL_ZERO_TOL for value in retrieval_norms[1:]):
            raise RuntimeError(f"episode {episode}: post-write retrieval never became nonzero")
        if video.frames != len(rows):
            raise RuntimeError(f"episode {episode}: rendered frame count mismatch: {video.frames} != {len(rows)}")
        final_state_max_abs = self._state_max_abs(memory_state)
        jsonl_identity = fresh.causal._file_identity(jsonl_path)
        episode_summary = {
            "episode": episode,
            "prompt": plan.prompt,
            "heldout_side": plan.side,
            "source_parquet": str(source.path),
            "fps": fps,
            "frame_count": len(rows),
            "expected_frames": [0, len(rows) - 1],
            "label_runs": payload["label_runs"],
            "decoded_counts": counts,
            "decode_status_counts": decode_status_counts,
            "decoded_runs": decoded_runs,
            "online_memory": {
                "M0_initializations": 1,
                "strict_order": True,
                "read_count": len(rows),
                "write_count": write_count,
                "expected_write_count": expected_writes,
                "frozen_read_count": len(rows) - write_count,
                "write_schedule": "frame_index % 15 == 0",
                "state_thread_count": max(0, len(rows) - 1),
                "retrieval_norm_min": min(retrieval_norms),
                "retrieval_norm_max": max(retrieval_norms),
                "retrieval_norm_nonzero_frames": sum(
                    value > fresh.RETRIEVAL_ZERO_TOL for value in retrieval_norms
                ),
                "state_difference_all_frames_min": min(state_update_diffs),
                "state_difference_all_frames_max": max(state_update_diffs),
                "committed_update_max_abs_difference_min": min(committed_update_diffs),
                "committed_update_max_abs_difference_max": max(committed_update_diffs),
                "frozen_state_max_abs_difference_max": max(frozen_state_diffs, default=0.0),
                "initial_state_max_abs": initial_state_max_abs,
                "final_state_max_abs": final_state_max_abs,
            },
            "video": video_name,
            "frame_records": jsonl_name,
            "elapsed_seconds": time.monotonic() - started,
        }
        return episode_summary, {video_name: video_identity, jsonl_name: jsonl_identity}

    def run(self) -> dict[str, Any]:
        if self.args.smoke_only:
            return self.run_smoke()
        started = time.monotonic()
        payloads = self._load_payloads()
        incomplete = self._reserve_output(payloads)
        episodes = []
        artifacts: dict[str, dict[str, Any]] = {}
        for episode in self.args.episodes:
            episode_summary, episode_artifacts = self._episode_video(payloads[episode])
            episodes.append(episode_summary)
            artifacts.update(episode_artifacts)

        summary = {
            **self._common_summary(payloads),
            "mode": "production-online-memory-scheduled-writes",
            "status": "complete",
            "episode_results": episodes,
            "total_frames": sum(item["frame_count"] for item in episodes),
            "total_reads": sum(item["online_memory"]["read_count"] for item in episodes),
            "total_writes": sum(item["online_memory"]["write_count"] for item in episodes),
            "decode_status_counts": {
                status: sum(item["decode_status_counts"][status] for item in episodes)
                for status in ("terminated", "truncated")
            },
            "elapsed_seconds": time.monotonic() - started,
            "artifacts": artifacts,
        }
        summary_identity = fresh._write_json_atomic(self.args.artifact_dir / "summary.json", summary)
        artifacts["summary.json"] = summary_identity
        fresh._write_complete_atomic(self.args.artifact_dir, artifacts)
        incomplete.unlink()
        print(
            json.dumps(
                {
                    "status": "complete",
                    "memory_mode": "online_read_every_frame_scheduled_write_stride15",
                    "episodes": list(self.args.episodes),
                    "total_frames": summary["total_frames"],
                    "output": str(self.args.artifact_dir),
                },
                indent=2,
            ),
            flush=True,
        )
        return summary


def main(argv: list[str] | None = None) -> None:
    HeldoutOnlineMemoryVideo(_parse_args(argv)).run()


if __name__ == "__main__":
    main()
