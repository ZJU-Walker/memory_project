import copy
import queue
import threading
import time

import numpy as np
import pytest

from openpi_client import action_chunk_broker


def test_realtime_broker_sends_zero_then_right_padded_action_prefix() -> None:
    policy = _QueuedPolicy([_array_chunk(0, horizon=5)])
    broker = _broker(
        policy,
        action_horizon=5,
        steps_between_inference=2,
        initial_delay_steps=1,
    )

    try:
        _assert_array_action(broker.infer(_array_obs(0)), 0)
        _assert_array_action(broker.infer(_array_obs(1)), 1)
        assert len(policy.requests) == 1

        # The request is handed off from obs2, after actions 0 and 1 have
        # completed and before old-chunk action 2 is consumed.
        _assert_array_action(broker.infer(_array_obs(2)), 2)
        policy.wait_for_requests(2)
        assert policy.requests[0]["step"] == 0
        assert policy.requests[1]["step"] == 2

        initial_prefix = policy.requests[0]["action_prefix"]
        np.testing.assert_array_equal(initial_prefix["actions"], np.zeros((5, 2), dtype=np.float32))
        assert initial_prefix["actions"].dtype == np.float32
        assert initial_prefix["delay"] == 0
        assert initial_prefix["prefix_length"] == 0

        prefix = policy.requests[1]["action_prefix"]
        np.testing.assert_array_equal(prefix["actions"], _repeated_actions([2, 3, 4, 0, 0]))
        assert prefix["delay"] == 1
        assert prefix["prefix_length"] == 3
    finally:
        policy.enqueue(_array_chunk(10, horizon=5))
        broker.close()


def test_realtime_broker_supports_nested_action_structures() -> None:
    policy = _QueuedPolicy([_nested_chunk(0, horizon=4)])
    broker = _broker(policy, action_horizon=4, steps_between_inference=2)

    try:
        action = broker.infer(_nested_obs(0))
        np.testing.assert_array_equal(action["actions"]["arm"], np.full(2, 0, dtype=np.float32))
        np.testing.assert_array_equal(action["actions"]["gripper"], np.full(1, 100, dtype=np.float32))
        broker.infer(_nested_obs(1))
        broker.infer(_nested_obs(2))
        policy.wait_for_requests(2)
        assert policy.requests[1]["step"] == 2

        initial = policy.requests[0]["action_prefix"]["actions"]
        assert initial["arm"].shape == (4, 2)
        assert initial["gripper"].shape == (4, 1)

        prefix = policy.requests[1]["action_prefix"]
        np.testing.assert_array_equal(prefix["actions"]["arm"], _repeated_actions([2, 3, 0, 0]))
        np.testing.assert_array_equal(
            prefix["actions"]["gripper"],
            np.asarray([[102], [103], [0], [0]], dtype=np.float32),
        )
        assert prefix["prefix_length"] == 2
    finally:
        policy.enqueue(_nested_chunk(10, horizon=4))
        broker.close()


def test_realtime_broker_preserves_steps_executed_during_inference() -> None:
    policy = _QueuedPolicy([_array_chunk(0, horizon=5)])
    broker = _broker(policy, action_horizon=5, steps_between_inference=3)

    try:
        _assert_array_action(broker.infer(_array_obs(0)), 0)
        _assert_array_action(broker.infer(_array_obs(1)), 1)
        _assert_array_action(broker.infer(_array_obs(2)), 2)
        assert len(policy.requests) == 1

        _assert_array_action(broker.infer(_array_obs(3)), 3)
        policy.wait_for_requests(2)
        assert policy.requests[1]["step"] == 3
        next_chunk = _array_chunk(10, horizon=5)
        policy.enqueue(next_chunk)
        _wait_for_chunk_and_step(broker, next_chunk, step=1)

        # Index 0 of the replacement chunk corresponds to the old action that
        # ran during inference, so the next returned action must be index 1.
        _assert_array_action(broker.infer(_array_obs(4)), 11)
    finally:
        broker.close()


def test_realtime_broker_uses_conservative_observed_delay() -> None:
    policy = _QueuedPolicy([_array_chunk(0, horizon=8)])
    broker = _broker(
        policy,
        action_horizon=8,
        steps_between_inference=3,
        initial_delay_steps=1,
        delay_tolerance_steps=3,
    )

    try:
        for step in range(3):
            _assert_array_action(broker.infer(_array_obs(step)), step)
        assert len(policy.requests) == 1

        # Two actions execute during the first background inference.
        _assert_array_action(broker.infer(_array_obs(3)), 3)
        policy.wait_for_requests(2)
        assert policy.requests[1]["step"] == 3
        _assert_array_action(broker.infer(_array_obs(4)), 4)
        next_chunk = _array_chunk(10, horizon=8)
        policy.enqueue(next_chunk)
        _wait_for_chunk_and_step(broker, next_chunk, step=2)

        # Move the replacement chunk to its inference trigger.
        _assert_array_action(broker.infer(_array_obs(5)), 12)
        _assert_array_action(broker.infer(_array_obs(6)), 13)
        policy.wait_for_requests(3)
        assert policy.requests[2]["step"] == 6
        assert policy.requests[2]["action_prefix"]["delay"] == 2
    finally:
        policy.enqueue(_array_chunk(20, horizon=8))
        broker.close()


def test_realtime_broker_clips_delay_to_remaining_prefix() -> None:
    policy = _QueuedPolicy([_array_chunk(0, horizon=4)])
    broker = _broker(
        policy,
        action_horizon=4,
        steps_between_inference=2,
        initial_delay_steps=4,
        max_async_delay_steps=4,
    )

    try:
        broker.infer(_array_obs(0))
        broker.infer(_array_obs(1))
        broker.infer(_array_obs(2))
        policy.wait_for_requests(2)
        assert policy.requests[1]["step"] == 2
        prefix = policy.requests[1]["action_prefix"]
        assert prefix["prefix_length"] == 2
        assert prefix["delay"] == 2
    finally:
        policy.enqueue(_array_chunk(10, horizon=4))
        broker.close()


def test_realtime_broker_blocks_at_relative_backstop() -> None:
    policy = _QueuedPolicy([_array_chunk(0, horizon=6)])
    broker = _broker(
        policy,
        action_horizon=6,
        steps_between_inference=2,
        initial_delay_steps=1,
        delay_tolerance_steps=1,
        max_async_delay_steps=6,
    )
    blocked_result = queue.Queue()

    try:
        _assert_array_action(broker.infer(_array_obs(0)), 0)
        _assert_array_action(broker.infer(_array_obs(1)), 1)
        _assert_array_action(broker.infer(_array_obs(2)), 2)
        policy.wait_for_requests(2)
        assert policy.requests[1]["step"] == 2
        _assert_array_action(broker.infer(_array_obs(3)), 3)

        thread = threading.Thread(target=_infer_into_queue, args=(broker, _array_obs(4), blocked_result))
        thread.start()
        time.sleep(0.05)
        assert blocked_result.empty()

        policy.enqueue(_array_chunk(10, horizon=6))
        _assert_array_action(_get_thread_result(thread, blocked_result), 12)
    finally:
        policy.enqueue(_array_chunk(20, horizon=6))
        broker.close()


def test_realtime_broker_surfaces_background_errors() -> None:
    policy = _QueuedPolicy([_array_chunk(0, horizon=3), RuntimeError("boom")])
    broker = _broker(policy, action_horizon=3, steps_between_inference=1)

    try:
        broker.infer(_array_obs(0))
        broker.infer(_array_obs(1))
        policy.wait_for_requests(2)
        assert policy.requests[1]["step"] == 1
        _wait_for_error(broker)

        with pytest.raises(RuntimeError, match="boom"):
            broker.infer(_array_obs(2))
    finally:
        broker.close()


def test_realtime_broker_surfaces_invalid_background_chunks() -> None:
    policy = _QueuedPolicy([_array_chunk(0, horizon=3), {"actions": np.zeros((2, 2))}])
    broker = _broker(policy, action_horizon=3, steps_between_inference=1)

    try:
        broker.infer(_array_obs(0))
        broker.infer(_array_obs(1))
        policy.wait_for_requests(2)
        assert policy.requests[1]["step"] == 1
        _wait_for_error(broker)

        with pytest.raises(ValueError, match="leading dimension"):
            broker.infer(_array_obs(2))
    finally:
        broker.close()


def test_realtime_broker_reset_stops_worker_and_restarts_with_zero_prefix() -> None:
    policy = _QueuedPolicy([_array_chunk(0, horizon=4), _array_chunk(20, horizon=4)])
    broker = _broker(policy, action_horizon=4, steps_between_inference=3)

    _assert_array_action(broker.infer(_array_obs(0)), 0)
    broker.reset()
    assert policy.reset_count == 1

    try:
        _assert_array_action(broker.infer(_array_obs(1)), 20)
        assert len(policy.requests) == 2
        assert policy.requests[1]["action_prefix"]["delay"] == 0
        assert policy.requests[1]["action_prefix"]["prefix_length"] == 0
    finally:
        broker.close()

    with pytest.raises(RuntimeError, match="closed"):
        broker.infer(_array_obs(2))


def test_realtime_broker_validates_configuration_and_chunk_shape() -> None:
    policy = _QueuedPolicy([])
    with pytest.raises(ValueError, match="cannot exceed"):
        _broker(policy, action_horizon=2, steps_between_inference=3)
    with pytest.raises(ValueError, match="initial_delay_steps"):
        _broker(
            policy,
            action_horizon=4,
            steps_between_inference=2,
            initial_delay_steps=3,
            max_async_delay_steps=2,
        )

    policy.enqueue({"actions": np.zeros((3, 2), dtype=np.float32)})
    broker = _broker(policy, action_horizon=4, steps_between_inference=2)
    try:
        with pytest.raises(ValueError, match="leading dimension"):
            broker.infer(_array_obs(0))
    finally:
        broker.close()


class _QueuedPolicy:
    def __init__(self, responses) -> None:
        self._responses = queue.Queue()
        for response in responses:
            self.enqueue(response)
        self.requests = []
        self.reset_count = 0
        self._condition = threading.Condition()

    def infer(self, obs):
        with self._condition:
            self.requests.append(copy.deepcopy(obs))
            self._condition.notify_all()

        response = self._responses.get(timeout=2.0)
        if isinstance(response, BaseException):
            raise response
        return response

    def reset(self) -> None:
        self.reset_count += 1

    def enqueue(self, response) -> None:
        self._responses.put(response)

    def wait_for_requests(self, count: int) -> None:
        with self._condition:
            assert self._condition.wait_for(lambda: len(self.requests) >= count, timeout=2.0)


def _broker(
    policy,
    *,
    action_horizon: int,
    steps_between_inference: int,
    initial_delay_steps: int = 1,
    delay_tolerance_steps: int = 3,
    max_async_delay_steps: int = 15,
):
    return action_chunk_broker.RealtimeActionChunkBroker(
        policy,
        action_horizon=action_horizon,
        steps_between_inference=steps_between_inference,
        initial_delay_steps=initial_delay_steps,
        delay_tolerance_steps=delay_tolerance_steps,
        max_async_delay_steps=max_async_delay_steps,
        delay_buffer_size=4,
    )


def _array_chunk(start: int, *, horizon: int):
    return {
        "actions": _repeated_actions(range(start, start + horizon)),
        "metadata": {"chunk_start": start},
    }


def _nested_chunk(start: int, *, horizon: int):
    return {
        "actions": {
            "arm": _repeated_actions(range(start, start + horizon)),
            "gripper": np.asarray(range(start + 100, start + 100 + horizon), dtype=np.float32)[:, None],
        }
    }


def _array_obs(step: int):
    return {
        "step": step,
        "observation/state": np.full(2, step, dtype=np.float32),
    }


def _nested_obs(step: int):
    return {
        "step": step,
        "observation/state": {
            "arm": np.full(2, step, dtype=np.float32),
            "gripper": np.full(1, step + 100, dtype=np.float32),
        },
    }


def _repeated_actions(values):
    return np.repeat(np.asarray(list(values), dtype=np.float32)[:, None], 2, axis=1)


def _assert_array_action(action, value: int) -> None:
    np.testing.assert_array_equal(action["actions"], np.full(2, value, dtype=np.float32))


def _infer_into_queue(broker, obs, output) -> None:
    try:
        output.put(broker.infer(obs))
    except BaseException as exc:
        output.put(exc)


def _get_thread_result(thread, output):
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    result = output.get_nowait()
    if isinstance(result, BaseException):
        raise result
    return result


def _wait_for_chunk_and_step(broker, expected_chunk, *, step: int) -> None:
    with broker._condition:
        assert broker._condition.wait_for(
            lambda: broker._chunk is expected_chunk and broker._step == step,
            timeout=2.0,
        )


def _wait_for_error(broker) -> None:
    with broker._condition:
        assert broker._condition.wait_for(lambda: broker._error is not None, timeout=2.0)
