from collections import deque
import logging
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


logger = logging.getLogger(__name__)


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0


class RealtimeActionChunkBroker(_base_policy.BasePolicy):
    """Runs action-chunk inference in the background for real-time chunking.

    The wrapped policy returns a full action chunk under ``"actions"`` while
    this broker returns one action step from that chunk on every ``infer``
    call. After ``steps_between_inference`` actions, a single background
    worker requests the next chunk. Actions from the old chunk continue to be
    returned while that request is in flight.

    Each policy request carries an ``action_prefix`` dictionary with:

    * ``actions``: the unexecuted suffix of the old chunk, shifted to index 0
      and right-padded with zeros to ``action_horizon``;
    * ``delay``: the estimated number of old actions that will execute while
      the request is in flight; and
    * ``prefix_length``: the number of valid (non-padding) prefix actions.

    The policy is expected to hard-condition the first ``delay`` actions of
    its new chunk on this prefix. When the new chunk arrives, this broker skips
    exactly the number of new-chunk steps that were executed from the old
    chunk during inference, keeping the control stream continuous.

    Action chunks may be a single ndarray or a nested dm-tree structure whose
    leaves are ndarrays with ``action_horizon`` as their leading dimension.
    For the first request, the zero prefix shape is derived from
    ``observation[state_key]``; its structure must therefore match the raw
    action structure expected by the wrapped policy.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        action_horizon: int,
        steps_between_inference: int,
        initial_delay_steps: int = 1,
        delay_tolerance_steps: int = 3,
        max_async_delay_steps: int = 15,
        delay_buffer_size: int = 8,
        state_key: str = "observation/state",
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if steps_between_inference <= 0:
            raise ValueError("steps_between_inference must be positive")
        if steps_between_inference > action_horizon:
            raise ValueError(
                f"steps_between_inference ({steps_between_inference}) cannot exceed action_horizon ({action_horizon})"
            )
        if initial_delay_steps < 0:
            raise ValueError("initial_delay_steps must be non-negative")
        if delay_tolerance_steps < 0:
            raise ValueError("delay_tolerance_steps must be non-negative")
        if max_async_delay_steps <= 0:
            raise ValueError("max_async_delay_steps must be positive")
        if initial_delay_steps > max_async_delay_steps:
            raise ValueError("initial_delay_steps cannot exceed max_async_delay_steps")
        if delay_buffer_size <= 0:
            raise ValueError("delay_buffer_size must be positive")
        if not state_key:
            raise ValueError("state_key must be non-empty")

        self._policy = policy
        self._action_horizon = action_horizon
        self._steps_between_inference = steps_between_inference
        self._initial_delay_steps = initial_delay_steps
        self._delay_tolerance_steps = delay_tolerance_steps
        self._max_async_delay_steps = max_async_delay_steps
        self._delay_buffer_size = delay_buffer_size
        self._state_key = state_key

        self._condition = threading.Condition()
        self._thread: Optional[threading.Thread] = None
        self._closed = False
        self._error: Optional[BaseException] = None

        self._chunk: Optional[Dict] = None  # noqa: UP006
        self._step = 0
        self._inference_backstop_step: Optional[int] = None
        self._request_in_flight = False
        self._pending_request: Optional[Tuple[Dict, int]] = None  # noqa: UP006
        self._delay_history = deque([initial_delay_steps], maxlen=delay_buffer_size)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """Return the next action step, waiting only at the safety backstop."""
        self._ensure_initialized(obs)

        with self._condition:
            while True:
                self._raise_if_unavailable_locked()

                # Schedule from the observation supplied *after* the configured
                # number of old actions has completed and before returning the
                # next old action. Keeping the observation in an explicit
                # handoff makes request timing independent of which thread wins
                # the condition-variable wake-up race.
                if not self._request_in_flight and self._step >= self._steps_between_inference:
                    self._queue_inference_locked(obs)

                # Do not consume more old actions than the delay used to train
                # and condition the replacement chunk (plus a small tolerance).
                if self._inference_backstop_step is not None and self._step >= self._inference_backstop_step:
                    self._condition.wait()
                    continue

                if self._chunk is not None and self._step < self._action_horizon:
                    action = _slice_action_step(self._chunk, self._step)
                    self._step += 1
                    self._condition.notify_all()
                    return action

                # The old chunk is exhausted. This can happen when the
                # configured absolute delay limit reaches the horizon.
                self._condition.notify_all()
                self._condition.wait()

    @override
    def reset(self) -> None:
        """Stop background work, reset the wrapped policy, and start fresh."""
        # Waiting without a timeout is intentional: reusing the wrapped policy
        # while its previous infer call is still running would be unsafe.
        self._stop_async_thread(join_timeout_s=None)
        self._policy.reset()
        with self._condition:
            self._thread = None
            self._closed = False
            self._error = None
            self._chunk = None
            self._step = 0
            self._inference_backstop_step = None
            self._request_in_flight = False
            self._pending_request = None
            self._delay_history = deque([self._initial_delay_steps], maxlen=self._delay_buffer_size)

    def close(self, timeout_s: Optional[float] = 5.0) -> None:
        """Stop the background worker.

        The wrapped policy owns its transport and is not closed by this
        method. A daemon worker whose underlying ``infer`` cannot be cancelled
        may outlive the timeout, but it cannot publish another chunk after the
        broker is closed.
        """
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s must be non-negative or None")
        self._stop_async_thread(join_timeout_s=timeout_s)

    def _ensure_initialized(self, obs: Dict) -> None:  # noqa: UP006
        with self._condition:
            self._raise_if_unavailable_locked()
            if self._chunk is not None:
                return

        if self._state_key not in obs:
            raise KeyError(
                f"observation is missing {self._state_key!r}, which is required to construct "
                "the initial RTC action prefix"
            )
        initial_obs = dict(obs)
        initial_obs["action_prefix"] = _make_zero_action_prefix(
            obs[self._state_key],
            horizon=self._action_horizon,
        )
        initial_chunk = self._policy.infer(initial_obs)

        with self._condition:
            # close() may have been called while the initial synchronous
            # request was in flight.
            self._raise_if_unavailable_locked()
            self._set_chunk_locked(initial_chunk)
            self._start_async_thread_locked()
            self._condition.notify_all()

    def _inference_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed or self._error is not None or self._pending_request is not None
                )
                if self._closed or self._error is not None:
                    return

                request = self._pending_request
                if request is None:
                    continue
                obs, start_step = request
                self._pending_request = None

            try:
                new_chunk = self._policy.infer(obs)
            except BaseException as exc:
                with self._condition:
                    self._error = exc
                    self._inference_backstop_step = None
                    self._request_in_flight = False
                    self._condition.notify_all()
                return

            with self._condition:
                if self._closed:
                    return

                executed_during_inference = self._step - start_step
                try:
                    self._set_chunk_locked(new_chunk)
                except BaseException as exc:
                    self._error = exc
                    self._inference_backstop_step = None
                    self._request_in_flight = False
                    self._condition.notify_all()
                    return
                # New-chunk index i corresponds to old-prefix index i. Skip
                # actions already executed while inference was running.
                self._step = executed_during_inference
                self._inference_backstop_step = None
                self._request_in_flight = False
                self._delay_history.append(executed_during_inference)
                self._condition.notify_all()

    def _queue_inference_locked(self, obs: Dict) -> None:  # noqa: UP006
        if self._chunk is None:
            raise RuntimeError("cannot schedule RTC inference without an action chunk")

        start_step = self._step
        remaining_actions = max(0, self._action_horizon - start_step)

        # The maximum of a short recent history is deliberately conservative:
        # underestimating delay breaks the hard prefix, while a modest
        # overestimate merely copies more known actions.
        delay_estimate = max(self._delay_history)
        assumed_delay = min(delay_estimate, self._max_async_delay_steps, remaining_actions)
        backstop_delay = min(
            delay_estimate + self._delay_tolerance_steps,
            self._max_async_delay_steps,
            remaining_actions,
        )

        request_obs = dict(obs)
        request_obs["action_prefix"] = _make_action_prefix(
            self._chunk["actions"],
            start=start_step,
            delay_steps=assumed_delay,
            horizon=self._action_horizon,
        )
        self._inference_backstop_step = start_step + backstop_delay
        self._request_in_flight = True
        self._pending_request = (request_obs, start_step)
        self._condition.notify_all()

    def _set_chunk_locked(self, chunk: Dict) -> None:  # noqa: UP006
        if not isinstance(chunk, dict) or "actions" not in chunk:
            raise ValueError("policy inference result must be a dictionary containing 'actions'")
        _validate_action_chunk(chunk["actions"], horizon=self._action_horizon)
        self._chunk = chunk

    def _start_async_thread_locked(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._inference_loop,
            name="openpi-rtc-inference",
            daemon=True,
        )
        self._thread.start()

    def _stop_async_thread(self, *, join_timeout_s: Optional[float]) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            thread = self._thread

        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=join_timeout_s)
            if thread.is_alive():
                logger.warning("Timed out waiting for RTC background inference to stop")
            else:
                with self._condition:
                    if self._thread is thread:
                        self._thread = None

    def _raise_if_unavailable_locked(self) -> None:
        if self._error is not None:
            raise self._error
        if self._closed:
            raise RuntimeError("RealtimeActionChunkBroker is closed")


def _validate_action_chunk(actions: object, *, horizon: int) -> None:
    leaves = tree.flatten(actions)
    if not leaves:
        raise ValueError("policy action chunk must contain at least one array")
    for leaf in leaves:
        array = np.asarray(leaf)
        if array.ndim == 0 or array.shape[0] != horizon:
            raise ValueError(
                "every action leaf must have action_horizon as its leading dimension; "
                f"expected {horizon}, got shape {array.shape}"
            )


def _slice_action_step(chunk: Dict, step: int) -> Dict:  # noqa: UP006
    result = dict(chunk)
    result["actions"] = tree.map_structure(lambda value: np.asarray(value)[step, ...], chunk["actions"])
    return result


def _make_action_prefix(actions: object, *, start: int, delay_steps: int, horizon: int) -> Dict:  # noqa: UP006
    if start < 0 or start > horizon:
        raise ValueError(f"start must be between 0 and {horizon}, got {start}")
    _validate_action_chunk(actions, horizon=horizon)
    prefix_length = horizon - start

    def pad_prefix(value: object) -> np.ndarray:
        array = np.asarray(value)
        padded = np.zeros_like(array)
        if prefix_length:
            padded[:prefix_length, ...] = array[start:, ...]
        return padded

    return {
        "actions": tree.map_structure(pad_prefix, actions),
        "delay": min(max(0, delay_steps), prefix_length),
        "prefix_length": prefix_length,
    }


def _make_zero_action_prefix(action_template: object, *, horizon: int) -> Dict:  # noqa: UP006
    def zero_action(value: object) -> np.ndarray:
        array = np.asarray(value)
        return np.zeros((horizon, *array.shape), dtype=array.dtype)

    actions = tree.map_structure(zero_action, action_template)
    if not tree.flatten(actions):
        raise ValueError("initial action template must contain at least one array")
    return {
        "actions": actions,
        "delay": 0,
        "prefix_length": 0,
    }
