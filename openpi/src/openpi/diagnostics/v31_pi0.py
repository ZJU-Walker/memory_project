"""Concrete Pi0.5/YAM adapter for the v3.1 memory diagnostic framework.

The adapter intentionally supports fixed-observation offline replay only.  The generic
framework contains the real-time safety/interlock contract, but this module never opens robot
hardware or sends an actuator command.  That makes oracle and counterfactual branches safe to
run on recorded YAM demonstrations.

Episode manifest schema (JSON)::

    {
      "version": "yam-eval-v1",
      "control_hz": 30,
      "action_side": {                 # optional; otherwise action side is undetermined
        "axis_index": 0,
        "axis_name": "configured axis",
        "coordinate_frame": "configured frame",
        "window_start": 0,
        "window_end": 50,
        "aggregation": "endpoint",     # endpoint | mean | endpoint_delta
        "positive_side": "right",
        "threshold": 0.05
      },
      "episodes": [
        {"episode_id": "demo1", "path": "/abs/path/demo1", "ground_truth_side": "left"}
      ]
    }

Annotation schema (JSON)::

    {
      "version": "manual-v1",
      "episodes": {
        "demo1": {
          "reveal_frame": 300,
          "close_frame": 450,
          "decision_start_frame": 491,
          "decision_end_frame": 940
        }
      }
    }

Canonical subtask sequence scores teacher-force the complete strings including their training
terminator (``"open left bin\\n"`` and ``"open right bin\\n"``).  All compared branches reuse
the exact same observation, fast state, flow noise, RTC prefix, checkpoint, and decode settings.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import enum
import hashlib
import json
import logging
import pathlib
import re
import time
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.diagnostics import v31
from openpi.models import memory as _memory
from openpi.models import memory_diagnostics as _memory_diagnostics
from openpi.models import model as _model
from openpi.models import rtc as _rtc
from openpi.models import tokenizer as _tokenizer
from openpi.shared import nnx_utils
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
import openpi.transforms as _transforms

_EPISODE_MANIFEST_VERSION = "yam-eval-v1"
_ANNOTATION_VERSION = "manual-v1"
_EXPECTED_MEMORY_WRITE_SOURCE = "post_attention"


def _strict_int(value: Any, *, name: str, minimum: int | None = None) -> int:
    """Parse a JSON integer without silently truncating floats or accepting booleans."""

    if isinstance(value, bool) or not isinstance(value, int | np.integer):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _finite_float(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"{name} must be numeric, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _nonempty_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a non-empty string")
    return value


@dataclasses.dataclass(frozen=True)
class Pi0FastState:
    """Complete recurrent state needed to reproduce the next memory read/write."""

    memory_state: _memory.MemoryState = dataclasses.field(repr=False, compare=False)
    writes: int
    state_hash: str
    last_write_raw_frame: int | None = None
    last_write_wall_time_s: float | None = None


@dataclasses.dataclass(frozen=True)
class Pi0DiagnosticObservation:
    model_observation: _model.Observation = dataclasses.field(repr=False, compare=False)
    transformed_state: np.ndarray = dataclasses.field(repr=False, compare=False)
    # Robot-space state before normalization/padding. Action-side rules are defined in
    # robot coordinates, whereas ``transformed_state`` is only for reversing DeltaActions.
    raw_state: np.ndarray = dataclasses.field(repr=False, compare=False)
    episode_id: str
    observation_id: str
    raw_frame: int
    policy_step: int
    phase: str
    wall_time_s: float | None


@dataclasses.dataclass(frozen=True)
class Pi0RtcState:
    action_prefix: _rtc.ActionPrefix = dataclasses.field(repr=False, compare=False)
    metadata: Mapping[str, Any] = dataclasses.field(default_factory=dict)


def _validate_rtc_state(state: Pi0RtcState, *, action_horizon: int, action_dim: int) -> None:
    """Validate values and dtypes that the shared RTC shape/bounds check intentionally leaves generic."""

    _rtc.validate_action_prefix(state.action_prefix, action_horizon=action_horizon, action_dim=action_dim)
    actions = np.asarray(state.action_prefix.actions)
    delay = np.asarray(state.action_prefix.delay)
    prefix_length = np.asarray(state.action_prefix.prefix_length)
    if actions.dtype.kind != "f" or not np.all(np.isfinite(actions)):
        raise ValueError("RTC action prefix must contain only finite floating-point actions")
    if delay.dtype.kind not in "iu" or prefix_length.dtype.kind not in "iu":
        raise TypeError("RTC delay and prefix_length must use integer dtypes")
    if not isinstance(state.metadata, Mapping):
        raise TypeError("RTC metadata must be a mapping")


@dataclasses.dataclass(frozen=True)
class ActionSideConfig:
    """Robot/task-owned rule for interpreting an action trajectory's side."""

    axis_index: int
    axis_name: str
    coordinate_frame: str
    window_start: int
    window_end: int
    aggregation: str
    positive_side: v31.Side
    threshold: float

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None, *, horizon: int) -> ActionSideConfig | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("action_side must be a JSON object")
        horizon = _strict_int(horizon, name="action_horizon", minimum=1)
        required = {"axis_index", "axis_name", "coordinate_frame", "positive_side", "threshold"}
        missing = required - value.keys()
        if missing:
            raise ValueError(f"action_side is missing required fields: {sorted(missing)}")
        result = cls(
            axis_index=_strict_int(value["axis_index"], name="action_side.axis_index", minimum=0),
            axis_name=_nonempty_string(value["axis_name"], name="action_side.axis_name"),
            coordinate_frame=_nonempty_string(value["coordinate_frame"], name="action_side.coordinate_frame"),
            window_start=_strict_int(value.get("window_start", 0), name="action_side.window_start", minimum=0),
            window_end=_strict_int(value.get("window_end", horizon), name="action_side.window_end", minimum=1),
            aggregation=_nonempty_string(value.get("aggregation", "endpoint"), name="action_side.aggregation"),
            positive_side=_nonempty_string(value["positive_side"], name="action_side.positive_side"),  # type: ignore[arg-type]
            threshold=_finite_float(value["threshold"], name="action_side.threshold", minimum=0.0),
        )
        if not 0 <= result.window_start < result.window_end <= horizon:
            raise ValueError("action_side window must satisfy 0 <= start < end <= action_horizon")
        if result.aggregation not in ("endpoint", "mean", "endpoint_delta"):
            raise ValueError("action_side.aggregation must be endpoint, mean, or endpoint_delta")
        if result.positive_side not in ("left", "right"):
            raise ValueError("action_side.positive_side must be left or right")
        return result

    def classify(self, actions: np.ndarray, initial_state: np.ndarray) -> v31.ActionSideMetric:
        actions = np.asarray(actions)
        if actions.ndim != 2 or self.axis_index >= actions.shape[1]:
            return v31.ActionSideMetric(
                details={"reason": f"actions shape {actions.shape} has no configured axis {self.axis_index}"}
            )
        if actions.shape[0] < self.window_end:
            return v31.ActionSideMetric(
                details={
                    "reason": (
                        f"actions horizon {actions.shape[0]} is shorter than configured window end {self.window_end}"
                    )
                }
            )
        values = actions[self.window_start : self.window_end, self.axis_index]
        if values.dtype.kind not in "fiu" or not np.all(np.isfinite(values)):
            return v31.ActionSideMetric(details={"reason": "configured action-side window is not finite numeric data"})
        if self.aggregation == "endpoint":
            metric = float(values[-1])
        elif self.aggregation == "mean":
            metric = float(np.mean(values))
        else:
            initial_state = np.asarray(initial_state)
            if initial_state.ndim != 1 or self.axis_index >= initial_state.shape[0]:
                return v31.ActionSideMetric(details={"reason": "initial state lacks configured action-side axis"})
            if initial_state.dtype.kind not in "fiu" or not np.isfinite(initial_state[self.axis_index]):
                return v31.ActionSideMetric(details={"reason": "initial action-side state is not finite numeric data"})
            metric = float(values[-1] - initial_state[self.axis_index])
        if metric > self.threshold:
            side = self.positive_side
        elif metric < -self.threshold:
            side = "left" if self.positive_side == "right" else "right"
        else:
            side = "undetermined"
        return v31.ActionSideMetric(
            side=side,
            metric=metric,
            axis_name=self.axis_name,
            coordinate_frame=self.coordinate_frame,
            decision_window=f"[{self.window_start},{self.window_end})/{self.aggregation}",
            pass_threshold=self.threshold,
            details={"axis_index": self.axis_index, "positive_side": self.positive_side},
        )


def _complete_state_hash(
    memory_state: _memory.MemoryState,
    writes: int,
    last_write_raw_frame: int | None,
    last_write_wall_time_s: float | None,
) -> str:
    writes = _strict_int(writes, name="fast-state writes", minimum=0)
    if last_write_raw_frame is not None:
        last_write_raw_frame = _strict_int(last_write_raw_frame, name="fast-state last_write_raw_frame", minimum=0)
    if last_write_wall_time_s is not None:
        last_write_wall_time_s = _finite_float(
            last_write_wall_time_s, name="fast-state last_write_wall_time_s", minimum=0.0
        )
    payload = {
        "memory": _memory_diagnostics.memory_state_hash(memory_state),
        "writes": writes,
        "last_write_raw_frame": last_write_raw_frame,
        "last_write_wall_time_s": last_write_wall_time_s,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _make_fast_state(
    memory_state: _memory.MemoryState,
    *,
    writes: int,
    last_write_raw_frame: int | None = None,
    last_write_wall_time_s: float | None = None,
) -> Pi0FastState:
    return Pi0FastState(
        memory_state=memory_state,
        writes=writes,
        state_hash=_complete_state_hash(memory_state, writes, last_write_raw_frame, last_write_wall_time_s),
        last_write_raw_frame=last_write_raw_frame,
        last_write_wall_time_s=last_write_wall_time_s,
    )


class Pi0SnapshotEvaluator(v31.EvaluationAdapter):
    """Deterministic, functional evaluator over a loaded Pi0 memory model."""

    def __init__(
        self,
        model,
        *,
        decode_tokenizer,
        stop_token: int,
        output_transforms: Sequence[_transforms.DataTransformFn],
        configured_write_every_frames: int,
        control_hz: float,
        action_side_config: ActionSideConfig | None,
        max_decode_steps: int = 10,
        num_denoise_steps: int = 10,
        diagnostic_level: v31.DiagnosticLevel = "basic",
    ) -> None:
        if getattr(model, "memory_write_source", None) != "post_attention":
            raise ValueError("v3.1 diagnostics require memory_write_source='post_attention'")
        configured_write_every_frames = _strict_int(
            configured_write_every_frames, name="configured_write_every_frames", minimum=1
        )
        control_hz = _finite_float(control_hz, name="control_hz")
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        max_decode_steps = _strict_int(max_decode_steps, name="max_decode_steps", minimum=1)
        num_denoise_steps = _strict_int(num_denoise_steps, name="num_denoise_steps", minimum=1)
        if max_decode_steps > model.causal_token_len:
            raise ValueError(f"max_decode_steps={max_decode_steps} exceeds causal_token_len={model.causal_token_len}")
        self._model = model
        self._decode_tokenizer = decode_tokenizer
        self._stop_token = int(stop_token)
        self._output_transform = _transforms.compose(output_transforms)
        self._configured_write_every_frames = configured_write_every_frames
        self._control_hz = control_hz
        self._action_side_config = action_side_config
        self._max_decode_steps = max_decode_steps
        self._num_denoise_steps = num_denoise_steps
        self._diagnostic_level = diagnostic_level
        self._sample = nnx_utils.module_jit(
            model.sample_with_memory,
            static_argnames=(
                "stop_token",
                "max_decode_steps",
                "num_steps",
                "zero_read",
                "allow_write",
            ),
        )
        self._canonical_tokens = {
            text: self._tokenize_canonical(text) for text in (v31.CANONICAL_LEFT_SUBTASK, v31.CANONICAL_RIGHT_SUBTASK)
        }

    def initial_state(self) -> Pi0FastState:
        return _make_fast_state(self._model.memory.init_state(1), writes=0)

    def clone_fast_state(self, state: object) -> Pi0FastState:
        state = self._require_state(state)
        source_hash = self.fast_state_hash(state)
        if source_hash != state.state_hash:
            raise RuntimeError("fast state changed after its cached hash was recorded")
        cloned = _memory_diagnostics.clone_memory_state(state.memory_state, backend="preserve")
        result = _make_fast_state(
            cloned,
            writes=state.writes,
            last_write_raw_frame=state.last_write_raw_frame,
            last_write_wall_time_s=state.last_write_wall_time_s,
        )
        if result.state_hash != source_hash:
            raise RuntimeError("fast-state clone changed its complete hash")
        return result

    def fast_state_hash(self, state: object) -> str:
        state = self._require_state(state)
        return _complete_state_hash(
            state.memory_state,
            state.writes,
            state.last_write_raw_frame,
            state.last_write_wall_time_s,
        )

    def write_count(self, state: object) -> int:
        state = self._require_state(state)
        return _strict_int(state.writes, name="fast-state writes", minimum=0)

    @staticmethod
    def _require_state(state: object) -> Pi0FastState:
        if not isinstance(state, Pi0FastState):
            raise TypeError(f"expected Pi0FastState, got {type(state).__name__}")
        return state

    def _tokenize_canonical(self, text: str) -> tuple[jax.Array, jax.Array, int]:
        tokens = self._decode_tokenizer.encode(text.strip() + "\n")
        if not 0 < len(tokens) <= self._model.causal_token_len:
            raise ValueError(f"canonical subtask {text!r} has invalid token count {len(tokens)}")
        padded = np.zeros((1, self._model.causal_token_len), dtype=np.int32)
        mask = np.zeros_like(padded, dtype=bool)
        padded[0, : len(tokens)] = tokens
        mask[0, : len(tokens)] = True
        return jnp.asarray(padded), jnp.asarray(mask), len(tokens)

    def _run(
        self,
        observation: Pi0DiagnosticObservation,
        state: Pi0FastState,
        *,
        seed: int,
        forced_subtask: str | None,
        zero_read: bool,
        allow_write: bool,
        action_prefix: _rtc.ActionPrefix | None,
    ):
        forced_tokens = forced_mask = None
        if forced_subtask is not None:
            forced_tokens, forced_mask, _ = self._canonical_tokens[forced_subtask]
        key = jax.random.key(np.uint32(seed))
        noise = jax.random.normal(key, (1, self._model.action_horizon, self._model.action_dim), dtype=jnp.float32)
        output = self._sample(
            key,
            observation.model_observation,
            state.memory_state,
            stop_token=self._stop_token,
            max_decode_steps=self._max_decode_steps,
            num_steps=self._num_denoise_steps,
            noise=noise,
            action_prefix=action_prefix,
            forced_subtask_tokens=forced_tokens,
            forced_subtask_mask=forced_mask,
            zero_read=zero_read,
            allow_write=allow_write,
        )
        jax.block_until_ready(output)
        return output

    def _robot_actions(self, model_actions: Any, observation: Pi0DiagnosticObservation) -> np.ndarray:
        output = {
            "state": np.array(observation.transformed_state, copy=True),
            "actions": np.asarray(model_actions)[0],
        }
        transformed = self._output_transform(output)
        actions = np.asarray(transformed["actions"], dtype=np.float64)
        if actions.ndim != 2 or not np.all(np.isfinite(actions)):
            raise ValueError(f"diagnostic action output must be a finite rank-2 array, got {actions.shape}")
        return actions

    def _action_side(self, actions: np.ndarray, observation: Pi0DiagnosticObservation) -> v31.ActionSideMetric:
        if self._action_side_config is None:
            return v31.ActionSideMetric(
                details={"reason": "episode manifest does not define a task/robot action_side rule"}
            )
        return self._action_side_config.classify(actions, np.asarray(observation.raw_state))

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
    ) -> v31.DiagnosticResult:
        if not isinstance(observation, Pi0DiagnosticObservation):
            raise TypeError(f"expected Pi0DiagnosticObservation, got {type(observation).__name__}")
        state = self._require_state(fast_state)
        state_hash_before = self.fast_state_hash(state)
        if state_hash_before != state.state_hash:
            raise ValueError("fast state contents do not match its recorded complete hash")
        rtc = None
        if rtc_state is not None:
            if not isinstance(rtc_state, Pi0RtcState):
                raise TypeError(f"expected Pi0RtcState, got {type(rtc_state).__name__}")
            rtc = rtc_state
            _validate_rtc_state(
                rtc,
                action_horizon=self._model.action_horizon,
                action_dim=self._model.action_dim,
            )

        start = time.monotonic()
        # The two exact sequence-score calls differ only in the canonical causal tokens.  They
        # are read-only and use the same state/observation/noise controls.
        scored = {}
        for text in (v31.CANONICAL_LEFT_SUBTASK, v31.CANONICAL_RIGHT_SUBTASK):
            scored[text] = self._run(
                observation,
                state,
                seed=seed,
                forced_subtask=text,
                zero_read=zero_read,
                allow_write=False,
                action_prefix=None,
            )

        # Pre-RTC prediction.  A forced branch can reuse its scoring call; free decoding needs
        # one matched call of its own.
        if forced_subtask is None:
            pre_output = self._run(
                observation,
                state,
                seed=seed,
                forced_subtask=None,
                zero_read=zero_read,
                allow_write=allow_write and rtc is None,
                action_prefix=None,
            )
        elif allow_write and rtc is None:
            pre_output = self._run(
                observation,
                state,
                seed=seed,
                forced_subtask=forced_subtask,
                zero_read=zero_read,
                allow_write=True,
                action_prefix=None,
            )
        else:
            pre_output = scored[forced_subtask]

        if rtc is None:
            committed_output = pre_output
            rtc_metadata: dict[str, Any] = {"rtc_applied": False, "broker_simulated": False}
            warnings = ("RTC/broker state unavailable; committed action equals the pre-RTC action",)
        else:
            committed_output = self._run(
                observation,
                state,
                seed=seed,
                forced_subtask=forced_subtask,
                zero_read=zero_read,
                allow_write=allow_write,
                action_prefix=rtc.action_prefix,
            )
            rtc_metadata = {"rtc_applied": True, "broker_simulated": False, **dict(rtc.metadata)}
            warnings = ("model RTC prefix was applied; asynchronous broker scheduling itself was not replayed",)

        model_actions_pre, _, pre_aux = pre_output
        model_actions_committed, next_memory_state, aux = committed_output
        pre_actions = self._robot_actions(model_actions_pre, observation)
        committed_actions = self._robot_actions(model_actions_committed, observation)

        tokens = np.asarray(aux["tokens"])[0]
        token_mask = np.asarray(aux["token_mask"])[0]
        decoded = forced_subtask or self._decode_tokenizer.decode(tokens[token_mask].tolist()).strip()

        left_aux = scored[v31.CANONICAL_LEFT_SUBTASK][2]
        right_aux = scored[v31.CANONICAL_RIGHT_SUBTASK][2]
        left_count = self._canonical_tokens[v31.CANONICAL_LEFT_SUBTASK][2]
        right_count = self._canonical_tokens[v31.CANONICAL_RIGHT_SUBTASK][2]
        left_total = float(np.asarray(left_aux["conditioned_subtask_logp"])[0])
        right_total = float(np.asarray(right_aux["conditioned_subtask_logp"])[0])

        wrote = bool(np.asarray(aux["write_occurred"])[0])
        next_writes = state.writes + int(wrote)
        next_state = _make_fast_state(
            next_memory_state,
            writes=next_writes,
            last_write_raw_frame=observation.raw_frame if wrote else state.last_write_raw_frame,
            last_write_wall_time_s=observation.wall_time_s if wrote else state.last_write_wall_time_s,
        )
        actual_interval = None
        if wrote and state.last_write_wall_time_s is not None and observation.wall_time_s is not None:
            actual_interval = observation.wall_time_s - state.last_write_wall_time_s
        latency_ms = (time.monotonic() - start) * 1000

        return v31.DiagnosticResult(
            episode_id=observation.episode_id,
            observation_id=observation.observation_id,
            raw_frame=observation.raw_frame,
            policy_step=observation.policy_step,
            phase=observation.phase,
            seed=seed,
            left_score=v31.CanonicalScore(v31.CANONICAL_LEFT_SUBTASK, left_total, left_count),
            right_score=v31.CanonicalScore(v31.CANONICAL_RIGHT_SUBTASK, right_total, right_count),
            decoded_subtask=decoded,
            fast_state_hash_before=state_hash_before,
            fast_state_hash_after=next_state.state_hash,
            write_count_before=state.writes,
            write_count_after=next_writes,
            forced_subtask=forced_subtask,
            zero_read=zero_read,
            allow_write=allow_write,
            pre_rtc_action=pre_actions,
            committed_action=committed_actions,
            pre_rtc_side=self._action_side(pre_actions, observation),
            committed_side=self._action_side(committed_actions, observation),
            memory_read_norm=float(np.asarray(aux["retrieval_norm"])[0]),
            memory_gate_norm=float(np.asarray(aux["memory_gate_norm"])[0]),
            surprise=float(np.asarray(aux["surprise"])[0]),
            write_occurred=wrote,
            rtc_metadata=rtc_metadata,
            configured_write_every_frames=self._configured_write_every_frames,
            actual_write_interval_s=actual_interval,
            action_horizon=self._model.action_horizon,
            control_hz=self._control_hz,
            wall_time_s=observation.wall_time_s,
            latency_ms=latency_ms,
            warnings=warnings,
            diagnostics={
                "inner_grad_norm": float(np.asarray(aux["grad_norm"])[0]),
                "theta": float(np.asarray(aux["theta"])[0]),
                "eta": float(np.asarray(aux["eta"])[0]),
                "alpha": float(np.asarray(aux["alpha"])[0]),
                "pre_rtc_surprise": float(np.asarray(pre_aux["surprise"])[0]),
                "diagnostic_level": self._diagnostic_level,
            },
            next_fast_state=next_state,
        )

    def advance_state(self, observation: Pi0DiagnosticObservation, state: Pi0FastState, *, seed: int) -> Pi0FastState:
        """One normal prediction/write used only to build a pre-decision snapshot."""
        _, next_memory, aux = self._run(
            observation,
            state,
            seed=seed,
            forced_subtask=None,
            zero_read=False,
            allow_write=True,
            action_prefix=None,
        )
        if not bool(np.asarray(aux["write_occurred"])[0]):
            raise RuntimeError("snapshot builder requested a write but the model did not write")
        return _make_fast_state(
            next_memory,
            writes=state.writes + 1,
            last_write_raw_frame=observation.raw_frame,
            last_write_wall_time_s=observation.wall_time_s,
        )


def _read_json(path: str | pathlib.Path, *, name: str) -> Mapping[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = item
        return result

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON number {value}")

    value = json.loads(
        pathlib.Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _validate_document_version(document: Mapping[str, Any], *, name: str, expected: str) -> None:
    version = document.get("version")
    if version != expected:
        raise ValueError(f"unsupported {name} version {version!r}; expected {expected!r}")


def _require_control_hz(manifest: Mapping[str, Any]) -> float:
    """Return an explicit, strictly positive control rate from an episode manifest."""

    if "control_hz" not in manifest:
        raise ValueError("episode manifest must explicitly provide control_hz")
    control_hz = _finite_float(manifest["control_hz"], name="episode manifest control_hz")
    if control_hz <= 0:
        raise ValueError("episode manifest control_hz must be positive")
    return control_hz


@dataclasses.dataclass(frozen=True)
class CheckpointStaticProvenance:
    """Static model semantics attested by Orbax checkpoint custom metadata."""

    verified: bool
    metadata_path: str
    config_name: str | None = None
    memory_write_source: str | None = None


def _read_checkpoint_static_provenance(
    checkpoint: str | pathlib.Path,
    *,
    expected_config_name: str,
    expected_memory_write_source: str = _EXPECTED_MEMORY_WRITE_SOURCE,
) -> CheckpointStaticProvenance:
    """Strictly parse and verify config semantics from ``_CHECKPOINT_METADATA``.

    Legacy checkpoints with an intact ``custom_metadata`` object but neither static
    field are accepted as explicitly unverified. Partially populated, malformed, or
    contradictory provenance fails closed.
    """

    metadata_path = pathlib.Path(checkpoint) / "_CHECKPOINT_METADATA"
    document = _read_json(metadata_path, name="Orbax checkpoint metadata")
    if "custom_metadata" not in document:
        raise ValueError("Orbax checkpoint metadata is missing custom_metadata")
    custom_metadata = document["custom_metadata"]
    if not isinstance(custom_metadata, Mapping):
        raise TypeError("Orbax checkpoint custom_metadata must be a JSON object")
    has_config = "config_name" in custom_metadata
    has_write_source = "memory_write_source" in custom_metadata
    if not has_config and not has_write_source:
        return CheckpointStaticProvenance(verified=False, metadata_path=str(metadata_path))
    if has_config != has_write_source:
        raise ValueError("Orbax checkpoint custom_metadata must provide config_name and memory_write_source together")
    config_name = _nonempty_string(custom_metadata["config_name"], name="checkpoint config_name")
    memory_write_source = _nonempty_string(
        custom_metadata["memory_write_source"], name="checkpoint memory_write_source"
    )
    if config_name != expected_config_name:
        raise ValueError(
            f"checkpoint config_name {config_name!r} does not match requested config {expected_config_name!r}"
        )
    if memory_write_source != expected_memory_write_source:
        raise ValueError(
            "checkpoint memory_write_source "
            f"{memory_write_source!r} does not match required {expected_memory_write_source!r}"
        )
    return CheckpointStaticProvenance(
        verified=True,
        metadata_path=str(metadata_path),
        config_name=config_name,
        memory_write_source=memory_write_source,
    )


def _video_length(path: pathlib.Path) -> int:
    import cv2

    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count <= 0:
        raise ValueError(f"could not determine video length: {path}")
    return count


def _read_selected_frames(path: pathlib.Path, selected: Sequence[int]) -> dict[int, np.ndarray]:
    import cv2

    selected_set = {int(index) for index in selected}
    frames: dict[int, np.ndarray] = {}
    capture = cv2.VideoCapture(str(path))
    index = 0
    while selected_set:
        ok, frame = capture.read()
        if not ok:
            break
        if index in selected_set:
            frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            selected_set.remove(index)
        index += 1
    capture.release()
    if selected_set:
        raise ValueError(f"video {path} is missing requested frames {sorted(selected_set)[:5]}")
    return frames


class RawYamReplaySource(v31.OfflineReplaySource):
    def __init__(
        self,
        evaluator: Pi0SnapshotEvaluator,
        *,
        input_transforms: Sequence[_transforms.DataTransformFn],
        effective_stride: int,
    ) -> None:
        effective_stride = _strict_int(effective_stride, name="effective_stride", minimum=1)
        self._evaluator = evaluator
        self._input_transform = _transforms.compose(input_transforms)
        self._effective_stride = effective_stride
        self.manifest_data: Mapping[str, Any] | None = None
        self.annotation_data: Mapping[str, Any] | None = None

    def load_episodes(
        self, episode_manifest: str | pathlib.Path, annotations: str | pathlib.Path
    ) -> tuple[Sequence[v31.ReplayEpisode], Mapping[str, v31.EpisodeAnnotation]]:
        manifest = _read_json(episode_manifest, name="episode manifest")
        annotation_document = _read_json(annotations, name="annotations")
        _validate_document_version(manifest, name="episode manifest", expected=_EPISODE_MANIFEST_VERSION)
        _require_control_hz(manifest)
        _validate_document_version(annotation_document, name="annotation", expected=_ANNOTATION_VERSION)
        entries = manifest.get("episodes")
        if not isinstance(entries, list) or not entries:
            raise ValueError("episode manifest must contain a non-empty 'episodes' list")
        raw_annotations = annotation_document.get("episodes")
        if not isinstance(raw_annotations, Mapping):
            raise ValueError("annotations must contain an 'episodes' mapping")

        parsed_annotations = {}
        episodes = []
        seen_episode_ids = set()
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("every episode manifest entry must be an object")
            missing_entry_fields = {"episode_id", "path", "ground_truth_side"} - entry.keys()
            if missing_entry_fields:
                raise ValueError(f"episode manifest entry is missing fields: {sorted(missing_entry_fields)}")
            episode_id = _nonempty_string(entry["episode_id"], name="episode_id")
            if episode_id in seen_episode_ids:
                raise ValueError(f"episode manifest contains duplicate episode_id {episode_id!r}")
            seen_episode_ids.add(episode_id)
            if episode_id not in raw_annotations:
                raise ValueError(f"annotations are missing episode {episode_id!r}")
            raw_annotation = raw_annotations[episode_id]
            if not isinstance(raw_annotation, Mapping):
                raise ValueError(f"annotation {episode_id!r} must be an object")
            missing_annotation_fields = {"reveal_frame", "close_frame"} - raw_annotation.keys()
            if missing_annotation_fields:
                raise ValueError(f"annotation {episode_id!r} is missing fields: {sorted(missing_annotation_fields)}")
            annotation = v31.EpisodeAnnotation(
                reveal_frame=_strict_int(
                    raw_annotation["reveal_frame"], name=f"annotation {episode_id!r} reveal_frame", minimum=0
                ),
                close_frame=_strict_int(
                    raw_annotation["close_frame"], name=f"annotation {episode_id!r} close_frame", minimum=0
                ),
                decision_start_frame=(
                    None
                    if raw_annotation.get("decision_start_frame") is None
                    else _strict_int(
                        raw_annotation["decision_start_frame"],
                        name=f"annotation {episode_id!r} decision_start_frame",
                        minimum=0,
                    )
                ),
                decision_end_frame=(
                    None
                    if raw_annotation.get("decision_end_frame") is None
                    else _strict_int(
                        raw_annotation["decision_end_frame"],
                        name=f"annotation {episode_id!r} decision_end_frame",
                        minimum=0,
                    )
                ),
                version=_ANNOTATION_VERSION,
            )
            parsed_annotations[episode_id] = annotation
            episodes.append(self._load_episode(entry, annotation))
        self.manifest_data = manifest
        self.annotation_data = annotation_document
        return tuple(episodes), parsed_annotations

    def _load_episode(self, entry: Mapping[str, Any], annotation: v31.EpisodeAnnotation) -> v31.ReplayEpisode:
        episode_id = _nonempty_string(entry["episode_id"], name="episode_id")
        side = _nonempty_string(entry["ground_truth_side"], name=f"episode {episode_id!r} ground_truth_side")
        if side not in ("left", "right"):
            raise ValueError(f"episode {episode_id!r} ground_truth_side must be left or right")
        directory = (
            pathlib.Path(_nonempty_string(entry["path"], name=f"episode {episode_id!r} path")).expanduser().resolve()
        )
        required = {
            "top": directory / "top_camera_rgb.mp4",
            "left": directory / "left_camera_rgb.mp4",
            "right": directory / "right_camera_rgb.mp4",
            "left_state": directory / "left_joint_positions.npy",
            "right_state": directory / "right_joint_positions.npy",
        }
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"episode {episode_id!r} is missing files: {missing}")
        arm_states = {
            side_name: np.load(required[f"{side_name}_state"], allow_pickle=False) for side_name in ("left", "right")
        }
        for side_name, arm_state in arm_states.items():
            if arm_state.ndim != 2 or arm_state.shape[1] != 7:
                raise ValueError(
                    f"episode {episode_id!r} {side_name} joint state must have shape [T, 7], got {arm_state.shape}"
                )
            if not np.all(np.isfinite(arm_state)):
                raise ValueError(f"episode {episode_id!r} {side_name} joint state contains non-finite values")
        if arm_states["left"].shape[0] != arm_states["right"].shape[0]:
            raise ValueError(
                f"episode {episode_id!r} left/right joint-state lengths differ: "
                f"{arm_states['left'].shape[0]} != {arm_states['right'].shape[0]}"
            )
        state = np.concatenate([arm_states["left"], arm_states["right"]], axis=1).astype(np.float32)
        total = min(len(state), *(_video_length(required[name]) for name in ("top", "left", "right")))
        if total < 1:
            raise ValueError(f"episode {episode_id!r} has no synchronized state/video frames")
        annotated_frames = {
            "reveal_frame": annotation.reveal_frame,
            "close_frame": annotation.close_frame,
            "decision_start_frame": annotation.decision_start_frame,
            "decision_end_frame": annotation.decision_end_frame,
        }
        out_of_range = {name: frame for name, frame in annotated_frames.items() if frame is not None and frame >= total}
        if out_of_range:
            raise ValueError(
                f"episode {episode_id!r} annotations exceed its last usable frame {total - 1}: {out_of_range}"
            )
        frames = tuple(range(0, total, self._effective_stride))
        videos = {name: _read_selected_frames(required[name], frames) for name in ("top", "left", "right")}
        steps = []
        for policy_step, raw_frame in enumerate(frames):
            if raw_frame < annotation.reveal_frame:
                phase = "pre_reveal"
            elif raw_frame < annotation.close_frame:
                phase = "visible"
            elif (
                annotation.decision_start_frame is not None
                and raw_frame >= annotation.decision_start_frame
                and (annotation.decision_end_frame is None or raw_frame <= annotation.decision_end_frame)
            ):
                phase = "decision"
            elif annotation.decision_end_frame is not None and raw_frame > annotation.decision_end_frame:
                phase = "post_decision"
            else:
                phase = "post_close"
            raw_item = {
                "observation/image": videos["top"][raw_frame],
                "observation/left_wrist_image": videos["left"][raw_frame],
                "observation/right_wrist_image": videos["right"][raw_frame],
                "observation/state": state[raw_frame],
            }
            transformed = self._input_transform(raw_item)
            batch = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
            observation = Pi0DiagnosticObservation(
                model_observation=_model.Observation.from_dict(batch),
                transformed_state=np.asarray(transformed["state"]),
                raw_state=np.array(state[raw_frame], copy=True),
                episode_id=episode_id,
                observation_id=f"{episode_id}:frame:{raw_frame}",
                raw_frame=raw_frame,
                policy_step=policy_step,
                phase=phase,
                wall_time_s=raw_frame / self._evaluator._control_hz,  # noqa: SLF001
            )
            steps.append(
                v31.ReplayStep(
                    observation=observation,
                    observation_id=observation.observation_id,
                    raw_frame=raw_frame,
                    policy_step=policy_step,
                    phase=phase,
                    wall_time_s=observation.wall_time_s,
                    write_due=True,
                )
            )
        return v31.ReplayEpisode(
            episode_id=episode_id,
            ground_truth_side=side,  # type: ignore[arg-type]
            steps=tuple(steps),
            initial_fast_state=self._evaluator.initial_state(),
        )


class Pi0StateSerializer(v31.StateSerializer):
    def save_state(self, path: pathlib.Path, state: object) -> None:
        if not isinstance(state, Pi0FastState):
            raise TypeError(f"expected Pi0FastState, got {type(state).__name__}")
        actual_hash = _complete_state_hash(
            state.memory_state,
            state.writes,
            state.last_write_raw_frame,
            state.last_write_wall_time_s,
        )
        if state.state_hash != actual_hash:
            raise ValueError("refusing to save a fast state whose contents do not match its recorded hash")
        snapshot = _memory_diagnostics.create_memory_snapshot(
            state.memory_state,
            writes=state.writes,
            metadata={
                "complete_state_hash": actual_hash,
                "last_write_raw_frame": state.last_write_raw_frame,
                "last_write_wall_time_s": state.last_write_wall_time_s,
            },
            backend="numpy",
        )
        _memory_diagnostics.save_memory_snapshot(path, snapshot)

    def load_state(self, path: str | pathlib.Path, *, backend: Literal["numpy", "jax"] = "jax") -> Pi0FastState:
        """Restore a saved complete state and verify adapter metadata against its contents."""

        snapshot = _memory_diagnostics.load_memory_snapshot(path, backend=backend)
        required_metadata = {"complete_state_hash", "last_write_raw_frame", "last_write_wall_time_s"}
        missing = required_metadata - snapshot.metadata.keys()
        if missing:
            raise ValueError(f"Pi0 fast-state snapshot metadata is missing fields: {sorted(missing)}")
        recorded_hash = snapshot.metadata["complete_state_hash"]
        if not isinstance(recorded_hash, str):
            raise TypeError("Pi0 fast-state complete_state_hash metadata must be a string")
        state = _make_fast_state(
            snapshot.state,
            writes=snapshot.writes,
            last_write_raw_frame=snapshot.metadata["last_write_raw_frame"],
            last_write_wall_time_s=snapshot.metadata["last_write_wall_time_s"],
        )
        if state.state_hash != recorded_hash:
            raise ValueError(
                "Pi0 fast-state complete hash does not match the restored memory, write count, or write metadata"
            )
        return state


def _path_hash(path: str | pathlib.Path) -> str:
    root = pathlib.Path(path).resolve()
    digest = hashlib.sha256()
    files = [root] if root.is_file() else sorted(item for item in root.rglob("*") if item.is_file())
    for item in files:
        relative = item.name if root.is_file() else item.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _json_config(value: Any) -> Any:
    if value is tyro.MISSING:
        return "<MISSING>"
    if value is None or isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, re.Pattern):
        return value.pattern
    if dataclasses.is_dataclass(value):
        return {field.name: _json_config(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_config(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_config(item) for item in value]
    return repr(value)


class Pi0V31DiagnosticApplication(v31.DiagnosticApplication):
    def __init__(self, request: v31.DiagnosticRunRequest) -> None:
        if request.diagnostic_level == "tokens":
            raise NotImplementedError(
                "the concrete Pi0 adapter does not yet produce token diagnostics; use --diagnostic-level basic"
            )
        if request.mode != "offline":
            raise NotImplementedError(
                "the concrete Pi0 adapter is offline-only and never opens robot hardware; "
                "use a separately reviewed shadow-stream adapter for realtime mode"
            )
        if request.episodes is None or request.annotations is None:
            raise ValueError("offline Pi0 diagnostics require --episodes and --annotations")
        config_name = str(getattr(request, "config", "pi05_yam_mem_v31"))
        self._train_config = _config.get_config(config_name)
        if self._train_config.model.memory_write_source != _EXPECTED_MEMORY_WRITE_SOURCE:
            raise ValueError("Pi0 v3.1 diagnostics refuse configs whose writer is not post_attention")
        self._checkpoint = pathlib.Path(request.checkpoint).expanduser().resolve()
        if not (self._checkpoint / "params").is_dir():
            raise FileNotFoundError(f"checkpoint has no params item: {self._checkpoint}")
        self._checkpoint_provenance = _read_checkpoint_static_provenance(
            self._checkpoint,
            expected_config_name=config_name,
        )

        manifest_data = _read_json(request.episodes, name="episode manifest")
        _validate_document_version(manifest_data, name="episode manifest", expected=_EPISODE_MANIFEST_VERSION)
        control_hz = _require_control_hz(manifest_data)
        data_config = self._train_config.data.create(self._train_config.assets_dirs, self._train_config.model)
        configured_stride = _strict_int(data_config.memory_stride_frames, name="configured memory stride", minimum=1)
        effective_stride = (
            configured_stride
            if request.write_every_frames is None
            else _strict_int(request.write_every_frames, name="write_every_frames", minimum=1)
        )
        requested_horizon = (
            None
            if request.action_horizon is None
            else _strict_int(request.action_horizon, name="action_horizon", minimum=1)
        )
        if requested_horizon is not None and requested_horizon != self._train_config.model.action_horizon:
            logging.warning(
                "--action-horizon=%d differs from fixed checkpoint horizon %d; model horizon remains unchanged",
                requested_horizon,
                self._train_config.model.action_horizon,
            )

        if data_config.asset_id is None:
            raise ValueError("Pi0 YAM diagnostics require a normalization-stat asset_id")
        norm_stats = _normalize.load(self._checkpoint / "assets" / data_config.asset_id)
        model = self._train_config.model.load(_model.restore_params(self._checkpoint / "params", dtype=jnp.float32))
        tokenizer = _tokenizer.FASTSubtaskTokenizer(self._train_config.model.max_token_len)._paligemma_tokenizer  # noqa: SLF001
        stop_token = int(tokenizer.encode("placeholder subtask\n")[-1])
        input_transforms = [
            transform
            for transform in data_config.data_transforms.inputs
            if not isinstance(transform, _transforms.BuildMemorySequence)
        ]
        input_transforms = [
            *input_transforms,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
        output_transforms = [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
        action_side = ActionSideConfig.from_mapping(
            manifest_data.get("action_side"), horizon=self._train_config.model.action_horizon
        )
        self.evaluator = Pi0SnapshotEvaluator(
            model,
            decode_tokenizer=tokenizer,
            stop_token=stop_token,
            output_transforms=output_transforms,
            configured_write_every_frames=configured_stride,
            control_hz=control_hz,
            action_side_config=action_side,
            diagnostic_level=request.diagnostic_level,
        )
        self._source = RawYamReplaySource(
            self.evaluator,
            input_transforms=input_transforms,
            effective_stride=effective_stride,
        )
        self._episodes, self._annotations = self._source.load_episodes(request.episodes, request.annotations)
        self._manifest_data = manifest_data
        self._configured_stride = configured_stride
        self._effective_stride = effective_stride
        self._control_hz = control_hz
        self._serializer = Pi0StateSerializer()

    def build_manifest(self, request: v31.DiagnosticRunRequest) -> v31.RunManifest:
        warnings = []
        if self._effective_stride != self._configured_stride:
            warnings.append(
                f"cadence override {self._effective_stride} differs from configured {self._configured_stride} frames"
            )
        if self._manifest_data.get("action_side") is None:
            warnings.append("no action_side rule: automatic action-side results will be undetermined")
        if not self._checkpoint_provenance.verified:
            warnings.append(
                "checkpoint custom_metadata lacks config_name/memory_write_source; "
                "checkpoint static-config provenance is unverified"
            )
        return v31.RunManifest(
            run_id=request.output_dir.name,
            created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            code_revision=v31.current_code_revision(pathlib.Path(__file__).parents[3]),
            checkpoint_path=str(self._checkpoint),
            # Only inference params determine the served policy. Hashing train_state would add
            # tens of GB of unrelated optimizer data and make diagnostics unnecessarily slow.
            checkpoint_hash=_path_hash(self._checkpoint / "params"),
            loaded_config=_json_config(self._train_config),
            overrides={
                "write_every_frames": request.write_every_frames,
                "action_horizon": request.action_horizon,
            },
            episode_manifest=str(request.episodes),
            episode_manifest_hash=v31.sha256_file(request.episodes),
            episode_version=str(self._manifest_data.get("version", "unspecified")),
            annotations_path=str(request.annotations),
            annotations_hash=v31.sha256_file(request.annotations),
            annotation_version=str(self._source.annotation_data.get("version", "unspecified")),
            seeds=(request.seed,),
            configured_write_every_frames=self._configured_stride,
            effective_write_every_frames=self._effective_stride,
            action_horizon=self._train_config.model.action_horizon,
            control_hz=self._control_hz,
            rtc_settings={
                "training_simulated_delay_inclusive": self._train_config.model.simulated_delay,
                "offline_prefixes_available": False,
                "checkpoint_static_config_provenance_verified": self._checkpoint_provenance.verified,
                "checkpoint_metadata_path": self._checkpoint_provenance.metadata_path,
                "checkpoint_config_name": self._checkpoint_provenance.config_name,
                "checkpoint_memory_write_source": self._checkpoint_provenance.memory_write_source,
            },
            hardware_mode=request.mode,
            enabled_tests=request.tests,
            diagnostic_level=request.diagnostic_level,
            counterfactual_mode=request.counterfactual_mode,
            execute_actions=request.execute_actions,
            require_operator_confirmation=request.require_operator_confirmation,
            operator_confirmed=request.safety.operator_confirmed,
            active_forced_side=request.forced_subtask_side,
            warnings=tuple(warnings),
        )

    def _decision_step(self, episode: v31.ReplayEpisode) -> v31.ReplayStep | None:
        decision = [step for step in episode.steps if step.phase == "decision"]
        if not decision:
            return None
        return decision[0]

    def _state_before(self, episode: v31.ReplayEpisode, target: v31.ReplayStep, *, seed: int) -> Pi0FastState:
        state = self.evaluator.clone_fast_state(episode.initial_fast_state)
        for step in episode.steps:
            if step.policy_step >= target.policy_step:
                break
            state = self.evaluator.advance_state(
                step.observation, state, seed=v31.derive_seed(seed, episode.episode_id, step.policy_step)
            )
        return state

    def run(self, request: v31.DiagnosticRunRequest, writer: v31.ArtifactWriter) -> Sequence[v31.ExperimentReport]:
        reports = []
        decision_steps = {episode.episode_id: self._decision_step(episode) for episode in self._episodes}
        decision_states: dict[str, Pi0FastState] = {}
        state_artifacts: dict[str, str] = {}

        def decision_state(episode: v31.ReplayEpisode) -> Pi0FastState:
            target = decision_steps[episode.episode_id]
            if target is None:
                raise ValueError(f"episode {episode.episode_id!r} has no sampled decision observation")
            if episode.episode_id not in decision_states:
                decision_states[episode.episode_id] = self._state_before(episode, target, seed=request.seed)
                state_artifacts[episode.episode_id] = writer.save_state(
                    f"{episode.episode_id}_pre_decision",
                    decision_states[episode.episode_id],
                    self._serializer,
                )
            return decision_states[episode.episode_id]

        if "oracle" in request.tests:
            for episode in self._episodes:
                step = decision_steps[episode.episode_id]
                if step is None:
                    reports.append(
                        v31.ExperimentReport(
                            test_name="oracle",
                            records=(),
                            valid=False,
                            interpretation="invalid_missing_decision",
                            invalid_reason=(
                                f"episode {episode.episode_id!r} has no sampled observation inside its decision window"
                            ),
                            metadata={"episode_id": episode.episode_id},
                        )
                    )
                    continue
                report = v31.run_oracle_test(
                    self.evaluator,
                    episode,
                    step,
                    decision_state(episode),
                    seed=request.seed,
                    safety=request.safety,
                )
                reports.append(
                    dataclasses.replace(
                        report,
                        metadata={**report.metadata, "state_artifact": state_artifacts[episode.episode_id]},
                    )
                )

        if "state_swap" in request.tests:
            left_candidates = [
                episode
                for episode in self._episodes
                if episode.ground_truth_side == "left" and decision_steps[episode.episode_id] is not None
            ]
            right_candidates = [
                episode
                for episode in self._episodes
                if episode.ground_truth_side == "right" and decision_steps[episode.episode_id] is not None
            ]
            # With one write per loaded step, the decision policy-step index is the exact
            # pre-decision write count. Choose the closest left/right pair before replaying.
            matched_pair = min(
                ((left, right) for left in left_candidates for right in right_candidates),
                key=lambda pair: abs(
                    decision_steps[pair[0].episode_id].policy_step  # type: ignore[union-attr]
                    - decision_steps[pair[1].episode_id].policy_step  # type: ignore[union-attr]
                ),
                default=None,
            )
            left, right = matched_pair if matched_pair is not None else (None, None)
            if left is None or right is None:
                reports.append(
                    v31.ExperimentReport(
                        test_name="state_swap",
                        records=(),
                        valid=False,
                        interpretation="invalid_missing_side",
                        invalid_reason="state swap requires at least one left and one right episode",
                    )
                )
            else:
                left_state, right_state = decision_state(left), decision_state(right)
                m0 = self.evaluator.initial_state()
                snapshots = {
                    "left_state": v31.FastStateSnapshot(
                        left_state, left.episode_id, "left", "pre_decision", left_state.writes, left_state.state_hash
                    ),
                    "right_state": v31.FastStateSnapshot(
                        right_state,
                        right.episode_id,
                        "right",
                        "pre_decision",
                        right_state.writes,
                        right_state.state_hash,
                    ),
                    "m0": v31.FastStateSnapshot(m0, "learned_m0", None, "reset", 0, m0.state_hash),
                }
                m0_artifact = writer.save_state("learned_m0", m0, self._serializer)
                observations = {
                    "left_observation": (left, decision_steps[left.episode_id]),
                    "right_observation": (right, decision_steps[right.episode_id]),
                }
                report = v31.run_state_swap_test(
                    self.evaluator,
                    observations,  # type: ignore[arg-type]
                    snapshots,
                    seed=request.seed,
                    include_zero_read=True,
                    safety=request.safety,
                )
                reports.append(
                    dataclasses.replace(
                        report,
                        metadata={
                            **report.metadata,
                            "selected_left_episode": left.episode_id,
                            "selected_right_episode": right.episode_id,
                            "write_count_difference": abs(left_state.writes - right_state.writes),
                            "state_artifacts": {
                                "left_state": state_artifacts[left.episode_id],
                                "right_state": state_artifacts[right.episode_id],
                                "m0": m0_artifact,
                            },
                        },
                    )
                )

        if "freeze" in request.tests:
            reports.extend(
                v31.run_freeze_test(
                    self.evaluator,
                    episode,
                    self._annotations[episode.episode_id],
                    seed=request.seed,
                    safety=request.safety,
                )
                for episode in self._episodes
            )

        if "temporal" in request.tests:
            reports.extend(
                v31.run_temporal_test(
                    self.evaluator,
                    episode,
                    self._annotations[episode.episode_id],
                    seed=request.seed,
                    safety=request.safety,
                )
                for episode in self._episodes
            )
        return tuple(reports)


def create_application(request: v31.DiagnosticRunRequest) -> Pi0V31DiagnosticApplication:
    """CLI adapter factory: ``--adapter openpi.diagnostics.v31_pi0:create_application``."""
    return Pi0V31DiagnosticApplication(request)
