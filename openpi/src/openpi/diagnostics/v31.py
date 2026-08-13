"""Shared, policy-independent diagnostics for the pi0.5 memory v3.1 experiments.

This module deliberately knows nothing about JAX, Pi0, YAM hardware, or websocket
serving.  A model-specific adapter implements :class:`EvaluationAdapter`; the
experiment runners then provide matched counterfactual branching, state-isolation
checks, freeze semantics, artifact schemas, and execution interlocks.

The generic framework never sends an action to hardware.  In real-time mode it may
authorize one explicitly confirmed oracle condition, but the model/robot integration
must still implement the final operator interlock and actuation separately.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
import copy
import csv
import dataclasses
import hashlib
import importlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

SCHEMA_VERSION = "openpi.v31.diagnostics.v1"
CANONICAL_LEFT_SUBTASK = "open left bin"
CANONICAL_RIGHT_SUBTASK = "open right bin"
ACTIVE_ORACLE_CONFIRMATION = "CONFIRM_ACTIVE_V31_ORACLE"

Side = Literal["left", "right"]
ActionSide = Literal["left", "right", "undetermined"]
HardwareMode = Literal["offline", "realtime"]
CounterfactualMode = Literal["shadow", "active"]
DiagnosticLevel = Literal["basic", "tokens"]
TestName = Literal["oracle", "state_swap", "freeze", "temporal"]


def _check_finite(value: float | None, name: str) -> None:
    if value is not None and not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite when present, got {value!r}")


def _copy_array(value: np.ndarray | Sequence[float] | None, name: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite numeric values")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def derive_seed(seed: int, *parts: object) -> int:
    """Derive a reproducible uint32 seed without depending on Python's randomized hash."""
    payload = "\0".join([str(seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big", signed=False)


def _stable_result_id(fields: Mapping[str, Any]) -> str:
    payload = json.dumps(_jsonable(fields), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def _jsonable(value: Any) -> Any:
    """Converts a value to strict JSON primitives; dense arrays require artifact files."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value: {value!r}")
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        raise TypeError("dense arrays must be written to NPZ and referenced from JSON")
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    raise TypeError(f"cannot encode {type(value).__name__} as diagnostic JSON")


@dataclasses.dataclass(frozen=True)
class CanonicalScore:
    """Autoregressive sequence score for one exact canonical subtask string."""

    text: str
    total_logp: float
    token_count: int
    mean_logp: float | None = None

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("canonical score text must be non-empty")
        if self.token_count <= 0:
            raise ValueError("canonical score token_count must be positive")
        _check_finite(self.total_logp, "total_logp")
        if self.total_logp > 1e-6:
            raise ValueError("canonical total_logp must be <= 0")
        expected = float(self.total_logp) / self.token_count
        if self.mean_logp is None:
            object.__setattr__(self, "mean_logp", expected)
        else:
            _check_finite(self.mean_logp, "mean_logp")
            if not math.isclose(float(self.mean_logp), expected, rel_tol=1e-6, abs_tol=1e-7):
                raise ValueError("mean_logp must equal total_logp / token_count")


@dataclasses.dataclass(frozen=True)
class ActionSideMetric:
    """Task-configured action-side classification; never assumes a generic robot axis."""

    side: ActionSide = "undetermined"
    metric: float | None = None
    axis_name: str | None = None
    coordinate_frame: str | None = None
    decision_window: str | None = None
    pass_threshold: float | None = None
    details: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.side not in ("left", "right", "undetermined"):
            raise ValueError(f"invalid action side: {self.side!r}")
        _check_finite(self.metric, "action-side metric")
        _check_finite(self.pass_threshold, "action-side pass threshold")
        object.__setattr__(self, "details", dict(self.details))
        _jsonable(self.details)


@dataclasses.dataclass(frozen=True)
class DiagnosticResult:
    """One deterministic evaluation from a caller-owned fast-state snapshot.

    ``next_fast_state`` is an opaque runtime value.  It is excluded from JSON/CSV
    artifacts and equality because complete state persistence belongs to the adapter.
    All action arrays are defensive, read-only copies.
    """

    episode_id: str
    observation_id: str
    raw_frame: int
    policy_step: int
    phase: str
    seed: int
    left_score: CanonicalScore
    right_score: CanonicalScore
    decoded_subtask: str
    fast_state_hash_before: str
    fast_state_hash_after: str
    write_count_before: int
    write_count_after: int
    ground_truth_side: Side | None = None
    forced_subtask: str | None = None
    zero_read: bool = False
    allow_write: bool = False
    pre_rtc_action: np.ndarray | None = None
    committed_action: np.ndarray | None = None
    pre_rtc_side: ActionSideMetric = dataclasses.field(default_factory=ActionSideMetric)
    committed_side: ActionSideMetric = dataclasses.field(default_factory=ActionSideMetric)
    memory_read_norm: float | None = None
    memory_gate_norm: float | None = None
    surprise: float | None = None
    write_due: bool = False
    write_occurred: bool = False
    rtc_metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    configured_write_every_frames: int | None = None
    actual_write_interval_s: float | None = None
    action_horizon: int | None = None
    control_hz: float | None = None
    wall_time_s: float | None = None
    latency_ms: float | None = None
    warnings: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    token_diagnostics: Mapping[str, np.ndarray] = dataclasses.field(default_factory=dict, repr=False, compare=False)
    next_fast_state: Any = dataclasses.field(default=None, repr=False, compare=False)
    schema_version: str = SCHEMA_VERSION
    result_id: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported diagnostic schema: {self.schema_version!r}")
        if not self.episode_id or not self.observation_id:
            raise ValueError("episode_id and observation_id must be non-empty")
        if self.raw_frame < 0 or self.policy_step < 0:
            raise ValueError("raw_frame and policy_step must be non-negative")
        if self.ground_truth_side not in (None, "left", "right"):
            raise ValueError(f"invalid ground-truth side: {self.ground_truth_side!r}")
        if self.left_score.text != CANONICAL_LEFT_SUBTASK:
            raise ValueError(f"left score must use exact text {CANONICAL_LEFT_SUBTASK!r}")
        if self.right_score.text != CANONICAL_RIGHT_SUBTASK:
            raise ValueError(f"right score must use exact text {CANONICAL_RIGHT_SUBTASK!r}")
        if self.write_count_before < 0 or self.write_count_after < 0:
            raise ValueError("write counts must be non-negative")
        if not self.fast_state_hash_before or not self.fast_state_hash_after:
            raise ValueError("fast-state hashes must be non-empty")
        if self.write_occurred and not self.allow_write:
            raise ValueError("a result cannot write when allow_write=False")
        if not self.write_occurred and self.write_count_after != self.write_count_before:
            raise ValueError("write count changed even though write_occurred=False")
        if self.write_occurred and self.write_count_after != self.write_count_before + 1:
            raise ValueError("one diagnostic policy step must commit exactly one write")
        if self.configured_write_every_frames is not None and self.configured_write_every_frames <= 0:
            raise ValueError("configured write cadence must be positive")
        if self.action_horizon is not None and self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.control_hz is not None and self.control_hz <= 0:
            raise ValueError("control_hz must be positive")
        for name in (
            "memory_read_norm",
            "memory_gate_norm",
            "surprise",
            "actual_write_interval_s",
            "wall_time_s",
            "latency_ms",
        ):
            _check_finite(getattr(self, name), name)
        for name in (
            "memory_read_norm",
            "memory_gate_norm",
            "surprise",
            "actual_write_interval_s",
            "wall_time_s",
            "latency_ms",
        ):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be non-negative")
        object.__setattr__(self, "pre_rtc_action", _copy_array(self.pre_rtc_action, "pre_rtc_action"))
        object.__setattr__(self, "committed_action", _copy_array(self.committed_action, "committed_action"))
        object.__setattr__(self, "rtc_metadata", dict(self.rtc_metadata))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        _jsonable(self.rtc_metadata)
        _jsonable(self.diagnostics)
        object.__setattr__(self, "warnings", tuple(str(item) for item in self.warnings))
        token_values = {
            str(name): _copy_array(value, f"token_diagnostics[{name!r}]")
            for name, value in self.token_diagnostics.items()
        }
        object.__setattr__(self, "token_diagnostics", token_values)
        if not self.result_id:
            object.__setattr__(
                self,
                "result_id",
                _stable_result_id(
                    {
                        "episode": self.episode_id,
                        "observation": self.observation_id,
                        "policy_step": self.policy_step,
                        "seed": self.seed,
                        "forced_subtask": self.forced_subtask,
                        "zero_read": self.zero_read,
                        "allow_write": self.allow_write,
                        "state": self.fast_state_hash_before,
                    }
                ),
            )

    @property
    def delta_lr(self) -> float:
        """Positive values favor the exact canonical left subtask."""
        return float(self.left_score.total_logp - self.right_score.total_logp)

    @property
    def delta_lr_length_normalized(self) -> float:
        return float(self.left_score.mean_logp - self.right_score.mean_logp)  # type: ignore[operator]

    @property
    def signed_margin(self) -> float | None:
        if self.ground_truth_side is None:
            return None
        return self.delta_lr if self.ground_truth_side == "left" else -self.delta_lr

    def to_event_dict(
        self,
        *,
        test_name: str,
        branch: str,
        action_file: str | None = None,
        token_file: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Returns the strict, array-free event representation."""
        event = {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "test": test_name,
            "branch": branch,
            "episode_id": self.episode_id,
            "ground_truth_side": self.ground_truth_side,
            "observation_id": self.observation_id,
            "raw_frame": self.raw_frame,
            "policy_step": self.policy_step,
            "phase": self.phase,
            "wall_time_s": self.wall_time_s,
            "seed": self.seed,
            "forced_subtask": self.forced_subtask,
            "zero_read": self.zero_read,
            "allow_write": self.allow_write,
            "left_score": dataclasses.asdict(self.left_score),
            "right_score": dataclasses.asdict(self.right_score),
            "delta_lr": self.delta_lr,
            "delta_lr_length_normalized": self.delta_lr_length_normalized,
            "signed_margin": self.signed_margin,
            "decoded_subtask": self.decoded_subtask,
            "pre_rtc_action_side": dataclasses.asdict(self.pre_rtc_side),
            "committed_action_side": dataclasses.asdict(self.committed_side),
            "actions_file": action_file,
            "memory_read_norm": self.memory_read_norm,
            "memory_gate_norm": self.memory_gate_norm,
            "surprise": self.surprise,
            "write_due": self.write_due,
            "write_occurred": self.write_occurred,
            "write_count_before": self.write_count_before,
            "write_count_after": self.write_count_after,
            "fast_state_hash_before": self.fast_state_hash_before,
            "fast_state_hash_after": self.fast_state_hash_after,
            "configured_write_every_frames": self.configured_write_every_frames,
            "actual_write_interval_s": self.actual_write_interval_s,
            "action_horizon": self.action_horizon,
            "control_hz": self.control_hz,
            "rtc_metadata": self.rtc_metadata,
            "latency_ms": self.latency_ms,
            "warnings": self.warnings,
            "diagnostics": self.diagnostics,
            "token_diagnostics_file": token_file,
            "extra": dict(extra or {}),
        }
        return _jsonable(event)


@runtime_checkable
class EvaluationAdapter(Protocol):
    """Model-specific operations needed by all generic diagnostic runners."""

    def clone_fast_state(self, state: object) -> object:
        """Return an isolated, complete clone, including momentum and counters."""

    def fast_state_hash(self, state: object) -> str:
        """Return a deterministic hash over the complete next-step state."""

    def write_count(self, state: object) -> int:
        """Return the number of committed writes represented by ``state``."""

    def evaluate_snapshot(
        self,
        observation: object,
        fast_state: object,
        rtc_state: object | None,
        *,
        forced_subtask: str | None = None,
        zero_read: bool = False,
        allow_write: bool = False,
        seed: int = 0,
    ) -> DiagnosticResult:
        """Read/predict, optionally write, and return the new state without mutating inputs."""


def evaluate_snapshot(
    evaluator: EvaluationAdapter,
    observation: object,
    fast_state: object,
    rtc_state: object | None,
    *,
    forced_subtask: str | None = None,
    zero_read: bool = False,
    allow_write: bool = False,
    seed: int = 0,
) -> DiagnosticResult:
    """Evaluate one isolated branch and enforce the shared counterfactual contract.

    Caller-owned state, observation, and RTC inputs are never passed directly to
    model code.  Hashing before and after catches shallow state-clone aliases and
    accidental branch mutation.  A no-write branch must leave both the complete
    state hash and write count unchanged.
    """
    if forced_subtask not in (None, CANONICAL_LEFT_SUBTASK, CANONICAL_RIGHT_SUBTASK):
        raise ValueError(
            "forced_subtask must be None or one of the exact canonical strings: "
            f"{CANONICAL_LEFT_SUBTASK!r}, {CANONICAL_RIGHT_SUBTASK!r}"
        )
    original_hash = evaluator.fast_state_hash(fast_state)
    original_count = evaluator.write_count(fast_state)
    branch_state = evaluator.clone_fast_state(fast_state)
    if evaluator.fast_state_hash(branch_state) != original_hash:
        raise RuntimeError("clone_fast_state changed the state contents")
    try:
        branch_observation = copy.deepcopy(observation)
        branch_rtc_state = copy.deepcopy(rtc_state)
    except Exception as exc:
        raise TypeError(
            "diagnostic observation and RTC state must be cloneable so counterfactual branches remain isolated"
        ) from exc

    start = time.monotonic()
    result = evaluator.evaluate_snapshot(
        branch_observation,
        branch_state,
        branch_rtc_state,
        forced_subtask=forced_subtask,
        zero_read=zero_read,
        allow_write=allow_write,
        seed=seed,
    )
    wrapper_latency_ms = (time.monotonic() - start) * 1000
    if not isinstance(result, DiagnosticResult):
        raise TypeError(f"adapter returned {type(result).__name__}, expected DiagnosticResult")
    if evaluator.fast_state_hash(fast_state) != original_hash or evaluator.write_count(fast_state) != original_count:
        raise RuntimeError("evaluate_snapshot mutated the caller-owned fast state")
    if result.fast_state_hash_before != original_hash or result.write_count_before != original_count:
        raise RuntimeError("adapter reported a before-state that does not match the supplied snapshot")
    if (
        result.forced_subtask != forced_subtask
        or result.zero_read != zero_read
        or result.allow_write != allow_write
        or result.seed != seed
    ):
        raise RuntimeError("adapter result does not match the requested counterfactual controls")
    if result.next_fast_state is None:
        raise RuntimeError("adapter must return the complete next_fast_state in DiagnosticResult")
    after_hash = evaluator.fast_state_hash(result.next_fast_state)
    after_count = evaluator.write_count(result.next_fast_state)
    if result.fast_state_hash_after != after_hash or result.write_count_after != after_count:
        raise RuntimeError("adapter-reported after-state does not match next_fast_state")
    if not allow_write and (after_hash != original_hash or after_count != original_count or result.write_occurred):
        raise RuntimeError("read-only counterfactual changed the fast state")
    if not result.write_occurred and after_hash != original_hash:
        raise RuntimeError("adapter changed fast state while reporting write_occurred=False")
    if result.latency_ms is None:
        result = dataclasses.replace(result, latency_ms=wrapper_latency_ms)
    return result


@dataclasses.dataclass(frozen=True)
class ReplayStep:
    observation: object = dataclasses.field(repr=False, compare=False)
    observation_id: str
    raw_frame: int
    policy_step: int
    phase: str
    rtc_state: object | None = dataclasses.field(default=None, repr=False, compare=False)
    wall_time_s: float | None = None
    write_due: bool = True

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise ValueError("observation_id must be non-empty")
        if self.raw_frame < 0 or self.policy_step < 0:
            raise ValueError("raw_frame and policy_step must be non-negative")
        _check_finite(self.wall_time_s, "wall_time_s")
        if self.wall_time_s is not None and self.wall_time_s < 0:
            raise ValueError("wall_time_s must be non-negative")


@dataclasses.dataclass(frozen=True)
class ReplayEpisode:
    episode_id: str
    ground_truth_side: Side
    steps: tuple[ReplayStep, ...]
    initial_fast_state: object = dataclasses.field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if self.ground_truth_side not in ("left", "right"):
            raise ValueError(f"invalid episode side: {self.ground_truth_side!r}")
        object.__setattr__(self, "steps", tuple(self.steps))
        if not self.steps:
            raise ValueError("replay episode must contain at least one step")
        frames = [step.raw_frame for step in self.steps]
        policy_steps = [step.policy_step for step in self.steps]
        if frames != sorted(frames) or policy_steps != sorted(policy_steps):
            raise ValueError("replay steps must be ordered by raw_frame and policy_step")
        if len(set(frames)) != len(frames) or len(set(policy_steps)) != len(policy_steps):
            raise ValueError("raw_frame and policy_step values must be unique within an episode")


@dataclasses.dataclass(frozen=True)
class EpisodeAnnotation:
    reveal_frame: int
    close_frame: int
    decision_start_frame: int | None = None
    decision_end_frame: int | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.reveal_frame <= self.close_frame:
            raise ValueError("annotation must satisfy 0 <= reveal_frame <= close_frame")
        if self.decision_start_frame is not None and self.decision_start_frame < self.close_frame:
            raise ValueError("decision_start_frame cannot precede close_frame")
        if self.decision_end_frame is not None and (
            self.decision_start_frame is None or self.decision_end_frame < self.decision_start_frame
        ):
            raise ValueError("decision_end_frame requires and cannot precede decision_start_frame")


@dataclasses.dataclass(frozen=True)
class FastStateSnapshot:
    state: object = dataclasses.field(repr=False, compare=False)
    source_episode: str
    ground_truth_side: Side | None
    phase: str
    write_count: int
    state_hash: str

    def __post_init__(self) -> None:
        if not self.source_episode or not self.state_hash:
            raise ValueError("snapshot source_episode and state_hash must be non-empty")
        if self.ground_truth_side not in (None, "left", "right"):
            raise ValueError(f"invalid snapshot side: {self.ground_truth_side!r}")
        if self.write_count < 0:
            raise ValueError("snapshot write_count must be non-negative")


@runtime_checkable
class OfflineReplaySource(Protocol):
    """Repository-specific loader boundary for fixed-observation replay."""

    def load_episodes(
        self, episode_manifest: str | Path, annotations: str | Path
    ) -> tuple[Sequence[ReplayEpisode], Mapping[str, EpisodeAnnotation]]:
        """Load recorded observations; counterfactual actions must never alter later inputs."""


@dataclasses.dataclass(frozen=True)
class ExecutionSafety:
    hardware_mode: HardwareMode = "offline"
    counterfactual_mode: CounterfactualMode = "shadow"
    execute_actions: bool = False
    require_operator_confirmation: bool = False
    operator_confirmed: bool = False
    active_forced_side: Side | None = None


def validate_execution_safety(safety: ExecutionSafety, enabled_tests: Sequence[str]) -> None:
    """Fail closed around real-time counterfactuals and active oracle execution."""
    tests = tuple(enabled_tests)
    unknown = set(tests) - {"oracle", "state_swap", "freeze", "temporal"}
    if not tests or unknown:
        raise ValueError(f"unknown diagnostic tests: {sorted(unknown)}")
    if len(set(tests)) != len(tests):
        raise ValueError("diagnostic tests must not contain duplicates")
    if not safety.execute_actions and safety.active_forced_side is not None:
        raise PermissionError("a forced active side is only valid when execute_actions=True")
    if safety.hardware_mode == "offline":
        if safety.execute_actions or safety.counterfactual_mode != "shadow":
            raise PermissionError("offline diagnostics cannot execute actions or enter active mode")
        return
    if safety.hardware_mode != "realtime":
        raise ValueError(f"invalid hardware_mode: {safety.hardware_mode!r}")
    if not safety.execute_actions:
        if safety.counterfactual_mode != "shadow":
            raise PermissionError("non-executing real-time diagnostics must remain shadow-only")
        return
    if safety.counterfactual_mode != "active":
        raise PermissionError("execute_actions requires counterfactual_mode='active'")
    if tests != ("oracle",):
        raise PermissionError("only a single active oracle test may execute counterfactual actions")
    if safety.active_forced_side not in ("left", "right"):
        raise PermissionError("active oracle execution requires exactly one forced left/right condition")
    if not safety.require_operator_confirmation or not safety.operator_confirmed:
        raise PermissionError("active oracle execution requires the explicit operator confirmation interlock")


@dataclasses.dataclass(frozen=True)
class ExperimentRecord:
    branch: str
    result: DiagnosticResult
    valid: bool = True
    invalid_reason: str | None = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.branch:
            raise ValueError("record branch must be non-empty")
        if self.valid and self.invalid_reason is not None:
            raise ValueError("valid experiment records cannot have invalid_reason")
        if not self.valid and not self.invalid_reason:
            raise ValueError("invalid experiment records require invalid_reason")
        object.__setattr__(self, "metadata", dict(self.metadata))
        _jsonable(self.metadata)


@dataclasses.dataclass(frozen=True)
class ExperimentReport:
    test_name: TestName
    records: tuple[ExperimentRecord, ...]
    valid: bool
    interpretation: str
    invalid_reason: str | None = None
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.test_name not in ("oracle", "state_swap", "freeze", "temporal"):
            raise ValueError(f"unknown experiment test: {self.test_name!r}")
        if not self.interpretation:
            raise ValueError("report interpretation must be non-empty")
        if self.valid and self.invalid_reason is not None:
            raise ValueError("valid reports cannot have invalid_reason")
        if not self.valid and not self.invalid_reason:
            raise ValueError("invalid reports require invalid_reason")
        if self.valid and not self.records:
            raise ValueError("valid reports must contain at least one record")
        _jsonable(self.metadata)


def _with_step_identity(result: DiagnosticResult, episode: ReplayEpisode, step: ReplayStep) -> DiagnosticResult:
    if result.observation_id != step.observation_id:
        raise RuntimeError(
            f"adapter observation_id {result.observation_id!r} does not match replay {step.observation_id!r}"
        )
    if (
        result.episode_id != episode.episode_id
        or result.raw_frame != step.raw_frame
        or result.policy_step != step.policy_step
    ):
        raise RuntimeError("adapter result identity does not match replay step")
    return dataclasses.replace(
        result,
        phase=step.phase,
        ground_truth_side=episode.ground_truth_side,
        wall_time_s=step.wall_time_s,
        write_due=step.write_due,
    )


def run_oracle_test(
    evaluator: EvaluationAdapter,
    episode: ReplayEpisode,
    step: ReplayStep,
    fast_state: object,
    *,
    seed: int = 0,
    safety: ExecutionSafety | None = None,
) -> ExperimentReport:
    """Run matched forced-left/right semantic branches without memory writes."""
    safety = safety or ExecutionSafety()
    validate_execution_safety(safety, ("oracle",))
    sides: tuple[Side, ...]
    if safety.execute_actions:
        assert safety.active_forced_side is not None
        sides = (safety.active_forced_side,)
    else:
        sides = ("left", "right")
    records = []
    for side in sides:
        forced = CANONICAL_LEFT_SUBTASK if side == "left" else CANONICAL_RIGHT_SUBTASK
        result = evaluate_snapshot(
            evaluator,
            step.observation,
            fast_state,
            step.rtc_state,
            forced_subtask=forced,
            allow_write=False,
            seed=seed,
        )
        result = _with_step_identity(result, episode, step)
        records.append(
            ExperimentRecord(
                branch=f"forced_{side}",
                result=result,
                metadata={"forced_side": side, "execution_authorized": safety.execute_actions},
            )
        )
    broker_evaluated = all(
        result.rtc_metadata.get("broker_simulated") is True or result.rtc_metadata.get("broker_available") is True
        for result in (record.result for record in records)
    )
    if len(records) == 1:
        result = records[0].result
        expected_side = sides[0]
        if result.pre_rtc_side.side == "undetermined":
            interpretation = "invalid_or_undetermined"
            valid = False
            invalid_reason = "task-configured pre-RTC action-side metric is unavailable"
        elif result.pre_rtc_side.side != expected_side:
            interpretation = "active_oracle_action_conditioning_failure"
            valid = True
            invalid_reason = None
        elif not broker_evaluated:
            interpretation = "active_oracle_action_conditioning_pass_rtc_not_evaluated"
            valid = True
            invalid_reason = None
        elif result.committed_side.side == "undetermined":
            interpretation = "invalid_or_undetermined"
            valid = False
            invalid_reason = "task-configured committed/post-RTC action-side metric is unavailable"
        elif result.committed_side.side != expected_side:
            interpretation = "active_oracle_rtc_broker_failure"
            valid = True
            invalid_reason = None
        else:
            interpretation = "active_oracle_pass"
            valid = True
            invalid_reason = None
    else:
        left, right = (record.result for record in records)
        pre = (left.pre_rtc_side.side, right.pre_rtc_side.side)
        post = (left.committed_side.side, right.committed_side.side)
        if "undetermined" in pre:
            interpretation = "invalid_or_undetermined"
            valid = False
            invalid_reason = "task-configured pre-RTC action-side metric is unavailable"
        elif pre != ("left", "right"):
            interpretation = "action_conditioning_failure"
            valid = True
            invalid_reason = None
        elif not broker_evaluated:
            interpretation = "action_conditioning_pass_rtc_not_evaluated"
            valid = True
            invalid_reason = None
        elif "undetermined" in post:
            interpretation = "invalid_or_undetermined"
            valid = False
            invalid_reason = "task-configured committed/post-RTC action-side metric is unavailable"
        elif post != ("left", "right"):
            interpretation = "rtc_broker_failure"
            valid = True
            invalid_reason = None
        else:
            interpretation = "pass"
            valid = True
            invalid_reason = None
    return ExperimentReport(
        test_name="oracle",
        records=tuple(records),
        valid=valid,
        interpretation=interpretation,
        invalid_reason=invalid_reason,
        metadata={
            "seed": seed,
            "shadow_only": not safety.execute_actions,
            "broker_evaluated": broker_evaluated,
            "expected_side": sides[0] if len(records) == 1 else None,
        },
    )


def _validate_snapshot(evaluator: EvaluationAdapter, snapshot: FastStateSnapshot) -> None:
    if evaluator.fast_state_hash(snapshot.state) != snapshot.state_hash:
        raise ValueError(f"snapshot {snapshot.source_episode!r} hash does not match its state")
    if evaluator.write_count(snapshot.state) != snapshot.write_count:
        raise ValueError(f"snapshot {snapshot.source_episode!r} write count does not match its state")


def run_state_swap_test(
    evaluator: EvaluationAdapter,
    observations: Mapping[str, tuple[ReplayEpisode, ReplayStep]],
    snapshots: Mapping[str, FastStateSnapshot],
    *,
    seed: int = 0,
    include_zero_read: bool = True,
    safety: ExecutionSafety | None = None,
) -> ExperimentReport:
    """Evaluate fixed observations under matched left/right/m0 complete states."""
    safety = safety or ExecutionSafety()
    validate_execution_safety(safety, ("state_swap",))
    if safety.execute_actions:
        raise PermissionError("state-swap counterfactuals are shadow-only")
    required = {"left_state", "right_state", "m0"}
    missing = required - snapshots.keys()
    if missing:
        raise ValueError(f"state swap is missing snapshots: {sorted(missing)}")
    if not observations:
        raise ValueError("state swap requires at least one fixed observation")
    if snapshots["left_state"].ground_truth_side != "left":
        raise ValueError("left_state snapshot must come from a left-banana episode")
    if snapshots["right_state"].ground_truth_side != "right":
        raise ValueError("right_state snapshot must come from a right-banana episode")
    if snapshots["m0"].ground_truth_side is not None or snapshots["m0"].write_count != 0:
        raise ValueError("m0 snapshot must be an unconditioned zero-write reset state")
    for snapshot in snapshots.values():
        _validate_snapshot(evaluator, snapshot)

    records: list[ExperimentRecord] = []
    swap_effects: dict[str, float] = {}
    for observation_name, (episode, step) in observations.items():
        margins: dict[str, float] = {}
        for state_name in ("left_state", "right_state", "m0"):
            snapshot = snapshots[state_name]
            result = evaluate_snapshot(
                evaluator,
                step.observation,
                snapshot.state,
                step.rtc_state,
                allow_write=False,
                seed=seed,
            )
            result = _with_step_identity(result, episode, step)
            margins[state_name] = result.delta_lr
            records.append(
                ExperimentRecord(
                    branch=f"{observation_name}__{state_name}",
                    result=result,
                    metadata={
                        "observation_variant": observation_name,
                        "state_variant": state_name,
                        "state_source_episode": snapshot.source_episode,
                        "state_ground_truth_side": snapshot.ground_truth_side,
                        "state_phase": snapshot.phase,
                        "snapshot_hash": snapshot.state_hash,
                        "snapshot_write_count": snapshot.write_count,
                    },
                )
            )
        swap_effects[observation_name] = margins["left_state"] - margins["right_state"]
        if include_zero_read:
            snapshot = snapshots["m0"]
            result = evaluate_snapshot(
                evaluator,
                step.observation,
                snapshot.state,
                step.rtc_state,
                zero_read=True,
                allow_write=False,
                seed=seed,
            )
            result = _with_step_identity(result, episode, step)
            records.append(
                ExperimentRecord(
                    branch=f"{observation_name}__zero_read",
                    result=result,
                    metadata={
                        "observation_variant": observation_name,
                        "state_variant": "zero_read",
                        "state_source_episode": snapshot.source_episode,
                        "snapshot_hash": snapshot.state_hash,
                        "snapshot_write_count": snapshot.write_count,
                    },
                )
            )

    pair_records = {
        (record.metadata["observation_variant"], record.metadata["state_variant"]): record.result for record in records
    }
    pre_actions_consistent = all(
        pair_records[(obs_name, "left_state")].pre_rtc_side.side == "left"
        and pair_records[(obs_name, "right_state")].pre_rtc_side.side == "right"
        for obs_name in observations
    )
    broker_evaluated = all(
        result.rtc_metadata.get("broker_simulated") is True or result.rtc_metadata.get("broker_available") is True
        for result in pair_records.values()
    )
    committed_actions_consistent = not broker_evaluated or all(
        pair_records[(obs_name, "left_state")].committed_side.side == "left"
        and pair_records[(obs_name, "right_state")].committed_side.side == "right"
        for obs_name in observations
    )
    decoded_flip = all(
        pair_records[(obs_name, "left_state")].decoded_subtask == CANONICAL_LEFT_SUBTASK
        and pair_records[(obs_name, "right_state")].decoded_subtask == CANONICAL_RIGHT_SUBTASK
        for obs_name in observations
    )
    margins_flip = all(
        pair_records[(obs_name, "left_state")].delta_lr > 0 and pair_records[(obs_name, "right_state")].delta_lr < 0
        for obs_name in observations
    )
    pre_action_sides_available = all(
        pair_records[(obs_name, state_name)].pre_rtc_side.side != "undetermined"
        for obs_name in observations
        for state_name in ("left_state", "right_state")
    )
    pre_actions_ignore_state = pre_action_sides_available and all(
        pair_records[(obs_name, "left_state")].pre_rtc_side.side
        == pair_records[(obs_name, "right_state")].pre_rtc_side.side
        for obs_name in observations
    )
    effects = tuple(swap_effects.values())
    m0_right_favoring = all(pair_records[(obs_name, "m0")].delta_lr < 0 for obs_name in observations)
    zero_right_favoring = include_zero_read and all(
        pair_records[(obs_name, "zero_read")].delta_lr < 0 for obs_name in observations
    )
    left_write_count = snapshots["left_state"].write_count
    right_write_count = snapshots["right_state"].write_count
    write_count_difference = abs(left_write_count - right_write_count)
    write_counts_matched = write_count_difference == 0
    if not write_counts_matched:
        # A state-content intervention is not isolated when state age also changes.
        # Preserve the measurements, but never promote this comparison to a causal pass.
        interpretation = "write_age_confounded"
    elif margins_flip and decoded_flip and pre_actions_consistent and committed_actions_consistent:
        interpretation = "strong_causal_pass"
    elif effects and all(effect > 0 for effect in effects):
        interpretation = "partial_pass"
    elif m0_right_favoring or zero_right_favoring:
        interpretation = "learned_right_prior"
    elif effects and all(abs(effect) <= 1e-8 for effect in effects):
        interpretation = "policy_ignores_memory"
    elif any(abs(effect) > 1e-8 for effect in effects) and pre_actions_ignore_state:
        interpretation = "decoder_uses_memory_actions_ignore_it"
    else:
        interpretation = "mixed_or_inconclusive"
    return ExperimentReport(
        test_name="state_swap",
        records=tuple(records),
        valid=True,
        interpretation=interpretation,
        metadata={
            "swap_effects": swap_effects,
            "left_right_write_counts_matched": write_counts_matched,
            "left_state_write_count": left_write_count,
            "right_state_write_count": right_write_count,
            "write_count_difference": write_count_difference,
            "m0_right_favoring": m0_right_favoring,
            "zero_read_right_favoring": zero_right_favoring,
            "broker_evaluated": broker_evaluated,
            "shadow_only": True,
        },
    )


def _aligned_freeze_frames(episode: ReplayEpisode, annotation: EpisodeAnnotation) -> tuple[int, int]:
    visible_writes = [
        step.raw_frame
        for step in episode.steps
        if step.write_due and annotation.reveal_frame <= step.raw_frame <= annotation.close_frame
    ]
    if not visible_writes:
        raise ValueError(
            "no scheduled write contains the reveal before the close boundary; freeze comparison is invalid"
        )
    reveal_write = visible_writes[0]
    through_close = [
        step.raw_frame for step in episode.steps if step.write_due and step.raw_frame <= annotation.close_frame
    ]
    if not through_close:
        raise ValueError("no scheduled write exists at or before the close boundary")
    return reveal_write, through_close[-1]


def _decision_results(records: Sequence[ExperimentRecord], annotation: EpisodeAnnotation) -> list[DiagnosticResult]:
    selected = [record.result for record in records if record.result.phase == "decision"]
    if selected:
        return selected
    if annotation.decision_start_frame is None:
        return []
    end = annotation.decision_end_frame if annotation.decision_end_frame is not None else math.inf
    return [record.result for record in records if annotation.decision_start_frame <= record.result.raw_frame <= end]


def run_freeze_test(
    evaluator: EvaluationAdapter,
    episode: ReplayEpisode,
    annotation: EpisodeAnnotation,
    *,
    seed: int = 0,
    safety: ExecutionSafety | None = None,
) -> ExperimentReport:
    """Replay normal/freeze-after-reveal/freeze-after-close against fixed observations."""
    safety = safety or ExecutionSafety()
    validate_execution_safety(safety, ("freeze",))
    if safety.execute_actions:
        raise PermissionError("freeze counterfactuals are shadow-only")
    try:
        reveal_write_frame, close_write_frame = _aligned_freeze_frames(episode, annotation)
    except ValueError as exc:
        return ExperimentReport(
            test_name="freeze",
            records=(),
            valid=False,
            interpretation="invalid_boundary",
            invalid_reason=str(exc),
            metadata={"episode_id": episode.episode_id},
        )

    variants = {
        "normal": None,
        "freeze_after_reveal": reveal_write_frame,
        "freeze_after_close": close_write_frame,
    }
    all_records: list[ExperimentRecord] = []
    by_variant: dict[str, list[ExperimentRecord]] = {}
    for variant, last_permitted_frame in variants.items():
        state = evaluator.clone_fast_state(episode.initial_fast_state)
        variant_records = []
        for step in episode.steps:
            allow_write = step.write_due and (last_permitted_frame is None or step.raw_frame <= last_permitted_frame)
            step_seed = derive_seed(seed, episode.episode_id, step.policy_step)
            result = evaluate_snapshot(
                evaluator,
                step.observation,
                state,
                step.rtc_state,
                allow_write=allow_write,
                seed=step_seed,
            )
            result = _with_step_identity(result, episode, step)
            record = ExperimentRecord(
                branch=variant,
                result=result,
                metadata={
                    "variant": variant,
                    "last_permitted_write_frame": last_permitted_frame,
                    "annotation_reveal_frame": annotation.reveal_frame,
                    "annotation_close_frame": annotation.close_frame,
                },
            )
            variant_records.append(record)
            all_records.append(record)
            state = result.next_fast_state
        by_variant[variant] = variant_records

    decision_means: dict[str, float | None] = {}
    for variant, records in by_variant.items():
        decision = _decision_results(records, annotation)
        margins = [result.signed_margin for result in decision if result.signed_margin is not None]
        decision_means[variant] = float(np.mean(margins)) if margins else None
    if any(value is None for value in decision_means.values()):
        return ExperimentReport(
            test_name="freeze",
            records=tuple(all_records),
            valid=False,
            interpretation="invalid_or_undetermined",
            invalid_reason="no annotated decision observations were available for every variant",
            metadata={
                "reveal_write_frame": reveal_write_frame,
                "close_write_frame": close_write_frame,
                "decision_signed_margin_mean": decision_means,
            },
        )
    normal = float(decision_means["normal"])  # type: ignore[arg-type]
    reveal = float(decision_means["freeze_after_reveal"])  # type: ignore[arg-type]
    close = float(decision_means["freeze_after_close"])  # type: ignore[arg-type]
    first_post_reveal: dict[str, float | None] = {}
    for variant, records in by_variant.items():
        after_reveal_write = [
            record.result.signed_margin
            for record in records
            if record.result.raw_frame > reveal_write_frame and record.result.signed_margin is not None
        ]
        first_post_reveal[variant] = after_reveal_write[0] if after_reveal_write else None
    normal_post_reveal = [
        record.result.signed_margin
        for record in by_variant["normal"]
        if record.result.raw_frame > reveal_write_frame and record.result.signed_margin is not None
    ]
    final_write_counts = {
        variant: records[-1].result.write_count_after if records else 0 for variant, records in by_variant.items()
    }
    if reveal <= 0 and close > 0:
        interpretation = "late_integration"
    elif normal < 0 and (reveal > 0 or close > 0) and any(margin > 0 for margin in normal_post_reveal):
        interpretation = "overwrite_or_interference"
    elif (
        normal <= 0
        and reveal <= 0
        and close <= 0
        and all(value is not None and value <= 0 for value in first_post_reveal.values())
    ):
        interpretation = "never_stored_or_used"
    elif normal >= max(reveal, close):
        interpretation = "no_observed_interference"
    else:
        interpretation = "mixed_or_inconclusive"
    return ExperimentReport(
        test_name="freeze",
        records=tuple(all_records),
        valid=True,
        interpretation=interpretation,
        metadata={
            "reveal_write_frame": reveal_write_frame,
            "close_write_frame": close_write_frame,
            "decision_signed_margin_mean": decision_means,
            "first_post_reveal_signed_margin": first_post_reveal,
            "final_write_count": final_write_counts,
            "shadow_only": True,
        },
    )


def run_temporal_test(
    evaluator: EvaluationAdapter,
    episode: ReplayEpisode,
    annotation: EpisodeAnnotation,
    *,
    seed: int = 0,
    safety: ExecutionSafety | None = None,
) -> ExperimentReport:
    """Log the pre-write policy signal and post-prediction write transition at every step."""
    safety = safety or ExecutionSafety()
    validate_execution_safety(safety, ("temporal",))
    if safety.execute_actions:
        raise PermissionError("generic temporal diagnostics never actuate hardware")
    state = evaluator.clone_fast_state(episode.initial_fast_state)
    records = []
    for step in episode.steps:
        step_seed = derive_seed(seed, episode.episode_id, step.policy_step)
        result = evaluate_snapshot(
            evaluator,
            step.observation,
            state,
            step.rtc_state,
            allow_write=step.write_due,
            seed=step_seed,
        )
        result = _with_step_identity(result, episode, step)
        records.append(
            ExperimentRecord(
                branch="normal",
                result=result,
                metadata={
                    "annotation_reveal_frame": annotation.reveal_frame,
                    "annotation_close_frame": annotation.close_frame,
                },
            )
        )
        state = result.next_fast_state

    reveal_write_frames = [
        step.raw_frame
        for step in episode.steps
        if step.write_due and annotation.reveal_frame <= step.raw_frame <= annotation.close_frame
    ]
    if not reveal_write_frames:
        return ExperimentReport(
            test_name="temporal",
            records=tuple(records),
            valid=False,
            interpretation="invalid_boundary",
            invalid_reason="no scheduled write contains the reveal before the close boundary",
            metadata={
                "shadow_only": True,
                "episode_id": episode.episode_id,
                "ground_truth_side": episode.ground_truth_side,
                "reveal_write_frame": None,
            },
        )
    # Predictions read M_(t-1) and write only afterward, so the first policy signal
    # that can depend on this reveal write is strictly after its scheduled frame.
    reveal_write_frame = reveal_write_frames[0]
    margins = [(record.result.raw_frame, record.result.signed_margin) for record in records]
    after_reveal = [margin for frame, margin in margins if frame > reveal_write_frame and margin is not None]
    after_close = [margin for frame, margin in margins if frame >= annotation.close_frame and margin is not None]
    after_reveal_delta_lr = [
        record.result.delta_lr for record in records if record.result.raw_frame > reveal_write_frame
    ]
    after_close_delta_lr = [
        record.result.delta_lr for record in records if record.result.raw_frame >= annotation.close_frame
    ]
    if (
        after_reveal
        and any(margin > 0 for margin in after_reveal)
        and after_close
        and any(margin < 0 for margin in after_close)
    ):
        interpretation = "learned_then_lost"
    elif after_reveal and all(margin <= 0 for margin in after_reveal):
        interpretation = "never_learned"
    elif after_close_delta_lr and all(delta < 0 for delta in after_close_delta_lr):
        # A single episode can establish right-favoring output, but not distinguish a
        # causal right-state response from a cross-episode right attractor.  The latter
        # is inferred only in the aggregate summary when both ground-truth sides agree.
        interpretation = "post_close_right_favoring"
    else:
        interpretation = "mixed_or_inconclusive"
    return ExperimentReport(
        test_name="temporal",
        records=tuple(records),
        valid=True,
        interpretation=interpretation,
        metadata={
            "shadow_only": True,
            "episode_id": episode.episode_id,
            "ground_truth_side": episode.ground_truth_side,
            "reveal_write_frame": reveal_write_frame,
            "after_reveal_delta_lr_first": after_reveal_delta_lr[0] if after_reveal_delta_lr else None,
            "after_reveal_delta_lr_last": after_reveal_delta_lr[-1] if after_reveal_delta_lr else None,
            "after_reveal_delta_lr_drift": (
                after_reveal_delta_lr[-1] - after_reveal_delta_lr[0] if len(after_reveal_delta_lr) >= 2 else None
            ),
            "after_close_delta_lr_mean": float(np.mean(after_close_delta_lr)) if after_close_delta_lr else None,
        },
    )


@dataclasses.dataclass(frozen=True)
class RunManifest:
    """Self-contained provenance and safety record written before diagnostic events."""

    run_id: str
    created_at_utc: str
    code_revision: str
    checkpoint_path: str
    checkpoint_hash: str
    loaded_config: Mapping[str, Any]
    overrides: Mapping[str, Any]
    episode_manifest: str | None
    episode_manifest_hash: str | None
    episode_version: str | None
    annotations_path: str | None
    annotations_hash: str | None
    annotation_version: str | None
    seeds: tuple[int, ...]
    configured_write_every_frames: int
    effective_write_every_frames: int
    action_horizon: int
    control_hz: float
    rtc_settings: Mapping[str, Any]
    hardware_mode: HardwareMode
    enabled_tests: tuple[TestName, ...]
    diagnostic_level: DiagnosticLevel
    counterfactual_mode: CounterfactualMode
    execute_actions: bool
    require_operator_confirmation: bool
    operator_confirmed: bool
    active_forced_side: Side | None = None
    warnings: tuple[str, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported manifest schema: {self.schema_version!r}")
        for name in ("run_id", "created_at_utc", "code_revision", "checkpoint_path", "checkpoint_hash"):
            if not getattr(self, name):
                raise ValueError(f"manifest {name} must be non-empty")
        if self.configured_write_every_frames <= 0 or self.effective_write_every_frames <= 0:
            raise ValueError("configured and effective write cadence must be positive")
        _check_finite(self.control_hz, "control_hz")
        if self.action_horizon <= 0 or self.control_hz <= 0:
            raise ValueError("action horizon and control frequency must be positive")
        if self.diagnostic_level not in ("basic", "tokens"):
            raise ValueError(f"invalid diagnostic_level: {self.diagnostic_level!r}")
        if not self.seeds:
            raise ValueError("manifest must record at least one seed")
        object.__setattr__(self, "loaded_config", copy.deepcopy(dict(self.loaded_config)))
        object.__setattr__(self, "overrides", copy.deepcopy(dict(self.overrides)))
        object.__setattr__(self, "rtc_settings", copy.deepcopy(dict(self.rtc_settings)))
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "enabled_tests", tuple(self.enabled_tests))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        _jsonable(self.loaded_config)
        _jsonable(self.overrides)
        _jsonable(self.rtc_settings)
        safety = ExecutionSafety(
            hardware_mode=self.hardware_mode,
            counterfactual_mode=self.counterfactual_mode,
            execute_actions=self.execute_actions,
            require_operator_confirmation=self.require_operator_confirmation,
            operator_confirmed=self.operator_confirmed,
            active_forced_side=self.active_forced_side,
        )
        validate_execution_safety(safety, self.enabled_tests)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(dataclasses.asdict(self))


@runtime_checkable
class StateSerializer(Protocol):
    def save_state(self, path: Path, state: object) -> None:
        """Persist a complete state without pickle or lossy summaries."""


def _temporal_cross_episode_summary(reports: Sequence[ExperimentReport]) -> dict[str, Any]:
    """Infer a shared right bias only from raw Delta_LR on both episode sides.

    A signed margin cannot be used here: negative signed margin means *incorrect* for
    either side, not specifically right-favoring.  Conversely, raw ``Delta_LR < 0``
    always favors the right canonical string.  We require valid temporal reports from
    both ground-truth sides before assigning the cross-episode category.
    """
    temporal = [report for report in reports if report.test_name == "temporal" and report.valid]
    rows = []
    for report in temporal:
        metadata = report.metadata
        rows.append(
            {
                "episode_id": metadata.get("episode_id"),
                "ground_truth_side": metadata.get("ground_truth_side"),
                "after_close_delta_lr_mean": metadata.get("after_close_delta_lr_mean"),
                "after_reveal_delta_lr_drift": metadata.get("after_reveal_delta_lr_drift"),
            }
        )
    sides = {row["ground_truth_side"] for row in rows if row["ground_truth_side"] in ("left", "right")}
    if sides != {"left", "right"}:
        return {
            "valid": False,
            "interpretation": "insufficient_both_sides",
            "invalid_reason": "valid temporal reports must include both left- and right-banana episodes",
            "episode_count": len(rows),
            "ground_truth_sides": sorted(sides),
            "criterion": "raw after-close Delta_LR < 0 on every episode side",
        }
    raw_values = [row["after_close_delta_lr_mean"] for row in rows]
    if any(value is None or not math.isfinite(float(value)) for value in raw_values):
        return {
            "valid": False,
            "interpretation": "insufficient_raw_delta_lr",
            "invalid_reason": "one or more temporal reports lack a finite after-close raw Delta_LR mean",
            "episode_count": len(rows),
            "ground_truth_sides": sorted(sides),
            "criterion": "raw after-close Delta_LR < 0 on every episode side",
        }
    right_favoring = all(float(value) < 0 for value in raw_values)
    drifts = [row["after_reveal_delta_lr_drift"] for row in rows]
    finite_drifts = [float(value) for value in drifts if value is not None and math.isfinite(float(value))]
    increasingly_negative = len(finite_drifts) == len(rows) and all(value < 0 for value in finite_drifts)
    return {
        "valid": True,
        "interpretation": "right_attractor_or_prior" if right_favoring else "no_shared_right_attractor",
        "invalid_reason": None,
        "episode_count": len(rows),
        "ground_truth_sides": sorted(sides),
        "all_episode_sides_right_favoring": right_favoring,
        "all_episode_sides_increasingly_negative": increasingly_negative,
        "criterion": "raw after-close Delta_LR < 0 on every episode side",
        "episodes": rows,
    }


_CSV_COLUMNS = (
    "schema_version",
    "result_id",
    "test",
    "branch",
    "valid",
    "invalid_reason",
    "report_valid",
    "report_invalid_reason",
    "record_valid",
    "record_invalid_reason",
    "episode_id",
    "ground_truth_side",
    "observation_id",
    "raw_frame",
    "policy_step",
    "phase",
    "seed",
    "forced_subtask",
    "zero_read",
    "allow_write",
    "left_total_logp",
    "right_total_logp",
    "left_mean_logp",
    "right_mean_logp",
    "delta_lr",
    "delta_lr_length_normalized",
    "signed_margin",
    "decoded_subtask",
    "pre_rtc_action_side",
    "committed_action_side",
    "actions_file",
    "write_due",
    "write_occurred",
    "write_count_before",
    "write_count_after",
    "fast_state_hash_before",
    "fast_state_hash_after",
    "memory_read_norm",
    "memory_gate_norm",
    "surprise",
    "latency_ms",
    "token_diagnostics_file",
    "metadata_json",
)


class ArtifactWriter:
    """Writes strict JSONL/CSV/NPZ artifacts and aggregate summaries."""

    def __init__(self, output_dir: str | Path, manifest: RunManifest):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        for directory in ("actions", "states", "token_diagnostics", "plots"):
            (self.output_dir / directory).mkdir()
        self.manifest = manifest
        self._reports: list[ExperimentReport] = []
        self._events: list[dict[str, Any]] = []
        self._finalized = False
        self._csv_rows: dict[str, list[dict[str, Any]]] = {
            "oracle": [],
            "state_swap": [],
            "freeze": [],
            "temporal": [],
        }
        self._write_json("run_manifest.json", manifest.to_dict())
        (self.output_dir / "events.jsonl").touch()

    def _write_json(self, relative_path: str, value: Mapping[str, Any]) -> None:
        path = self.output_dir / relative_path
        path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def save_state(self, name: str, state: object, serializer: StateSerializer) -> str:
        if self._finalized:
            raise RuntimeError("cannot add state artifacts after finalization")
        safe_name = _safe_filename(name)
        relative = Path("states") / f"{safe_name}.npz"
        destination = self.output_dir / relative
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite state artifact: {relative}")
        serializer.save_state(destination, state)
        if not destination.is_file():
            raise RuntimeError("state serializer did not create the requested state artifact")
        return relative.as_posix()

    def add_report(self, report: ExperimentReport) -> None:
        if self._finalized:
            raise RuntimeError("cannot add reports after finalization")
        if report.test_name not in self.manifest.enabled_tests:
            raise ValueError(f"application returned disabled diagnostic test {report.test_name!r}")
        if self.manifest.diagnostic_level != "tokens" and any(
            record.result.token_diagnostics for record in report.records
        ):
            raise ValueError("token diagnostics were produced without diagnostic_level='tokens'")
        self._reports.append(report)
        for record in report.records:
            index = len(self._events)
            stem = f"{index:06d}_{_safe_filename(report.test_name)}_{_safe_filename(record.branch)}"
            action_ref = None
            arrays = {}
            if record.result.pre_rtc_action is not None:
                arrays["pre_rtc_action"] = record.result.pre_rtc_action
            if record.result.committed_action is not None:
                arrays["committed_action"] = record.result.committed_action
            if arrays:
                relative = Path("actions") / f"{stem}.npz"
                np.savez_compressed(self.output_dir / relative, **arrays)
                action_ref = relative.as_posix()
            token_ref = None
            if record.result.token_diagnostics:
                relative = Path("token_diagnostics") / f"{stem}.npz"
                np.savez_compressed(self.output_dir / relative, **record.result.token_diagnostics)
                token_ref = relative.as_posix()
            event = record.result.to_event_dict(
                test_name=report.test_name,
                branch=record.branch,
                action_file=action_ref,
                token_file=token_ref,
                extra={
                    **record.metadata,
                    "record_valid": record.valid,
                    "record_invalid_reason": record.invalid_reason,
                    "report_valid": report.valid,
                    "report_invalid_reason": report.invalid_reason,
                    "report_interpretation": report.interpretation,
                },
            )
            self._events.append(event)
            with (self.output_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
            self._csv_rows[report.test_name].append(
                {
                    "schema_version": record.result.schema_version,
                    "result_id": record.result.result_id,
                    "test": report.test_name,
                    "branch": record.branch,
                    "valid": report.valid and record.valid,
                    "invalid_reason": report.invalid_reason or record.invalid_reason,
                    "report_valid": report.valid,
                    "report_invalid_reason": report.invalid_reason,
                    "record_valid": record.valid,
                    "record_invalid_reason": record.invalid_reason,
                    "episode_id": record.result.episode_id,
                    "ground_truth_side": record.result.ground_truth_side,
                    "observation_id": record.result.observation_id,
                    "raw_frame": record.result.raw_frame,
                    "policy_step": record.result.policy_step,
                    "phase": record.result.phase,
                    "seed": record.result.seed,
                    "forced_subtask": record.result.forced_subtask,
                    "zero_read": record.result.zero_read,
                    "allow_write": record.result.allow_write,
                    "left_total_logp": record.result.left_score.total_logp,
                    "right_total_logp": record.result.right_score.total_logp,
                    "left_mean_logp": record.result.left_score.mean_logp,
                    "right_mean_logp": record.result.right_score.mean_logp,
                    "delta_lr": record.result.delta_lr,
                    "delta_lr_length_normalized": record.result.delta_lr_length_normalized,
                    "signed_margin": record.result.signed_margin,
                    "decoded_subtask": record.result.decoded_subtask,
                    "pre_rtc_action_side": record.result.pre_rtc_side.side,
                    "committed_action_side": record.result.committed_side.side,
                    "actions_file": action_ref,
                    "write_due": record.result.write_due,
                    "write_occurred": record.result.write_occurred,
                    "write_count_before": record.result.write_count_before,
                    "write_count_after": record.result.write_count_after,
                    "fast_state_hash_before": record.result.fast_state_hash_before,
                    "fast_state_hash_after": record.result.fast_state_hash_after,
                    "memory_read_norm": record.result.memory_read_norm,
                    "memory_gate_norm": record.result.memory_gate_norm,
                    "surprise": record.result.surprise,
                    "latency_ms": record.result.latency_ms,
                    "token_diagnostics_file": token_ref,
                    "metadata_json": json.dumps(_jsonable(record.metadata), sort_keys=True),
                }
            )

    def finalize(self) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("diagnostic artifacts have already been finalized")
        for test_name, rows in self._csv_rows.items():
            path = self.output_dir / f"{test_name}_results.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=_CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
        report_valid = sum(report.valid for report in self._reports)
        record_valid = sum(report.valid and record.valid for report in self._reports for record in report.records)
        interpretation_counts = Counter(report.interpretation for report in self._reports)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.manifest.run_id,
            "report_counts": {
                "total": len(self._reports),
                "valid": report_valid,
                "invalid": len(self._reports) - report_valid,
            },
            "record_counts": {
                "total": sum(len(report.records) for report in self._reports),
                "valid": record_valid,
                "invalid": sum(len(report.records) for report in self._reports) - record_valid,
            },
            "interpretation_counts": dict(sorted(interpretation_counts.items())),
            "temporal_cross_episode": _temporal_cross_episode_summary(self._reports),
            "reports": [
                {
                    "test": report.test_name,
                    "valid": report.valid,
                    "invalid_reason": report.invalid_reason,
                    "interpretation": report.interpretation,
                    "record_count": len(report.records),
                    "metadata": report.metadata,
                }
                for report in self._reports
            ],
        }
        self._write_json("summary.json", summary)
        self._write_plots()
        self._finalized = True
        return _jsonable(summary)

    def _write_plots(self) -> None:
        try:
            import matplotlib as mpl

            mpl.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover - dependency is present in the dev environment.
            raise RuntimeError("matplotlib is required to create diagnostic plots") from exc

        def save_plot(filename: str, xlabel: str, ylabel: str, points: Sequence[tuple[float, float, str]]) -> None:
            figure, axis = plt.subplots(figsize=(8, 4.5))
            if points:
                groups: dict[str, list[tuple[float, float]]] = {}
                for x, y, label in points:
                    groups.setdefault(label, []).append((x, y))
                for label, values in sorted(groups.items()):
                    values.sort()
                    axis.plot([x for x, _ in values], [y for _, y in values], marker=".", label=label)
                axis.axhline(0.0, color="black", linewidth=0.8)
                if len(groups) <= 12:
                    axis.legend(fontsize=7)
            else:
                axis.text(0.5, 0.5, "No valid data", ha="center", va="center", transform=axis.transAxes)
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            figure.tight_layout()
            figure.savefig(self.output_dir / "plots" / filename, dpi=140)
            plt.close(figure)

        temporal_points = [
            (
                float(event["policy_step"]),
                float(event["signed_margin"]),
                f"{event['episode_id']}:{event['branch']}",
            )
            for event in self._events
            if event["test"] == "temporal" and event["signed_margin"] is not None
        ]
        save_plot("margin_over_time.png", "Policy step", "Signed canonical margin", temporal_points)
        write_points = [
            (
                float(event["write_count_before"]),
                float(event["signed_margin"]),
                f"{event['episode_id']}:{event['branch']}",
            )
            for event in self._events
            if event["test"] == "temporal" and event["signed_margin"] is not None
        ]
        save_plot("margin_vs_write_count.png", "Write count before prediction", "Signed canonical margin", write_points)

        swap_points = []
        for report in self._reports:
            if report.test_name != "state_swap":
                continue
            for index, (name, effect) in enumerate(sorted(report.metadata.get("swap_effects", {}).items())):
                swap_points.append((float(index), float(effect), str(name)))
        save_plot(
            "state_swap_effect.png", "Observation index", "Delta_LR(left state) - Delta_LR(right state)", swap_points
        )


def _safe_filename(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    safe = safe.strip("._")
    if not safe:
        raise ValueError(f"cannot make a safe artifact name from {value!r}")
    return safe[:120]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_code_revision(cwd: str | Path | None = None) -> str:
    """Return git revision plus a dirty suffix; never silently label dirty code clean."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return revision + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclasses.dataclass(frozen=True)
class DiagnosticRunRequest:
    mode: HardwareMode
    checkpoint: str
    tests: tuple[TestName, ...]
    output_dir: Path
    seed: int
    diagnostic_level: DiagnosticLevel
    adapter: str
    config: str = "pi05_yam_mem_v31"
    episodes: str | None = None
    annotations: str | None = None
    robot_config: str | None = None
    write_every_frames: int | None = None
    action_horizon: int | None = None
    counterfactual_mode: CounterfactualMode = "shadow"
    forced_subtask_side: Side | None = None
    execute_actions: bool = False
    require_operator_confirmation: bool = False
    operator_confirmation_token: str | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint or not self.config or not self.adapter:
            raise ValueError("checkpoint, config, and adapter must be non-empty")
        object.__setattr__(self, "tests", tuple(self.tests))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.diagnostic_level not in ("basic", "tokens"):
            raise ValueError(f"invalid diagnostic_level: {self.diagnostic_level!r}")
        if self.write_every_frames is not None and self.write_every_frames <= 0:
            raise ValueError("write_every_frames must be positive")
        if self.action_horizon is not None and self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

    @property
    def safety(self) -> ExecutionSafety:
        return ExecutionSafety(
            hardware_mode=self.mode,
            counterfactual_mode=self.counterfactual_mode,
            execute_actions=self.execute_actions,
            require_operator_confirmation=self.require_operator_confirmation,
            operator_confirmed=self.operator_confirmation_token == ACTIVE_ORACLE_CONFIRMATION,
            active_forced_side=self.forced_subtask_side,
        )


@runtime_checkable
class DiagnosticApplication(Protocol):
    """Pi0 integration boundary used by the thin CLI."""

    def build_manifest(self, request: DiagnosticRunRequest) -> RunManifest:
        """Load checkpoint/runtime config and report effective settings and provenance."""

    def run(self, request: DiagnosticRunRequest, writer: ArtifactWriter) -> Iterable[ExperimentReport]:
        """Run requested diagnostics using recorded inputs or shadow real-time observations."""


def _load_application(specification: str, request: DiagnosticRunRequest) -> DiagnosticApplication:
    if ":" not in specification:
        raise ValueError("--adapter must use 'python.module:factory' syntax")
    module_name, attribute_name = specification.split(":", 1)
    try:
        factory: Callable[[DiagnosticRunRequest], DiagnosticApplication] = getattr(
            importlib.import_module(module_name), attribute_name
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(f"could not load diagnostic adapter {specification!r}") from exc
    application = factory(request)
    if not isinstance(application, DiagnosticApplication):
        raise TypeError("diagnostic adapter factory did not return a DiagnosticApplication")
    return application


def _parse_tests(value: str) -> tuple[TestName, ...]:
    tests = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = set(tests) - {"oracle", "state_swap", "freeze", "temporal"}
    if not tests or unknown:
        raise argparse.ArgumentTypeError(f"tests must be a non-empty comma list; unknown={sorted(unknown)}")
    if len(set(tests)) != len(tests):
        raise argparse.ArgumentTypeError("tests must not contain duplicates")
    return tests  # type: ignore[return-value]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shared pi0.5 memory v3.1 diagnostic runner")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("offline", "realtime"):
        subparser = subparsers.add_parser(mode)
        subparser.add_argument("--checkpoint", required=True)
        subparser.add_argument(
            "--config",
            default="pi05_yam_mem_v31",
            help="Explicit static model config; checkpoint parameters alone do not encode v3/v3.1 writer semantics",
        )
        subparser.add_argument(
            "--adapter",
            required=True,
            help="Model-specific factory as python.module:factory (the generic runner cannot infer Pi0 checkpoint IO)",
        )
        subparser.add_argument("--tests", type=_parse_tests, required=True)
        subparser.add_argument("--seed", type=int, default=0)
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--diagnostic-level", choices=("basic", "tokens"), default="basic")
        subparser.add_argument("--write-every-frames", type=int)
        subparser.add_argument("--action-horizon", type=int)
        subparser.add_argument("--counterfactual-mode", choices=("shadow", "active"), default="shadow")
        subparser.add_argument("--forced-subtask", dest="forced_subtask_side", choices=("left", "right"))
        subparser.add_argument("--execute-actions", action="store_true")
        subparser.add_argument("--require-operator-confirmation", action="store_true")
        subparser.add_argument("--operator-confirmation-token")
    subparsers.choices["offline"].add_argument("--episodes", required=True)
    subparsers.choices["offline"].add_argument("--annotations", required=True)
    subparsers.choices["realtime"].add_argument("--robot-config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _build_parser().parse_args(argv)
    request = DiagnosticRunRequest(
        mode=namespace.mode,
        checkpoint=namespace.checkpoint,
        config=namespace.config,
        tests=namespace.tests,
        output_dir=namespace.output_dir,
        seed=namespace.seed,
        diagnostic_level=namespace.diagnostic_level,
        adapter=namespace.adapter,
        episodes=getattr(namespace, "episodes", None),
        annotations=getattr(namespace, "annotations", None),
        robot_config=getattr(namespace, "robot_config", None),
        write_every_frames=namespace.write_every_frames,
        action_horizon=namespace.action_horizon,
        counterfactual_mode=namespace.counterfactual_mode,
        forced_subtask_side=namespace.forced_subtask_side,
        execute_actions=namespace.execute_actions,
        require_operator_confirmation=namespace.require_operator_confirmation,
        operator_confirmation_token=namespace.operator_confirmation_token,
    )
    validate_execution_safety(request.safety, request.tests)
    application = _load_application(request.adapter, request)
    manifest = application.build_manifest(request)
    manifest_controls = (
        manifest.hardware_mode,
        manifest.enabled_tests,
        manifest.diagnostic_level,
        manifest.counterfactual_mode,
        manifest.execute_actions,
        manifest.require_operator_confirmation,
        manifest.operator_confirmed,
        manifest.active_forced_side,
    )
    request_controls = (
        request.mode,
        request.tests,
        request.diagnostic_level,
        request.counterfactual_mode,
        request.execute_actions,
        request.require_operator_confirmation,
        request.safety.operator_confirmed,
        request.forced_subtask_side,
    )
    if manifest_controls != request_controls:
        raise RuntimeError("adapter manifest safety/configuration controls do not match the validated CLI request")
    if manifest.loaded_config.get("name") != request.config:
        raise RuntimeError("adapter manifest loaded_config.name does not match the requested static config")
    writer = ArtifactWriter(request.output_dir, manifest)
    for report in application.run(request, writer):
        writer.add_report(report)
    writer.finalize()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
