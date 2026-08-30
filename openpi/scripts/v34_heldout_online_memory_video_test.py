from __future__ import annotations

# The diagnostic intentionally exposes script-private helpers for fail-closed tests.
# ruff: noqa: SLF001
from pathlib import Path
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v34_heldout_online_memory_video as online
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


def _args(**overrides) -> online.Args:
    values = {
        "checkpoint": Path("checkpoint/2500"),
        "dataset_root": Path("dataset"),
        "output_dir": Path("output"),
        "config": online.fresh.causal.RUN5_CONFIG,
        "parameter_source": "raw",
    }
    values.update(overrides)
    return online.Args(**values)


def _aux(*, write: bool, retrieval: float) -> dict[str, jnp.ndarray]:
    result = {
        key: jnp.asarray([0.25], dtype=jnp.float32)
        for key in online.FINITE_AUX_KEYS
    }
    result["retrieval_norm"] = jnp.asarray([retrieval], dtype=jnp.float32)
    result["eta"] = jnp.asarray([0.0], dtype=jnp.float32)
    result["write_occurred"] = jnp.asarray([write])
    result["tokens"] = jnp.asarray([[10, 99, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=jnp.int32)
    result["token_mask"] = jnp.asarray(
        [[True, True, False, False, False, False, False, False, False, False]]
    )
    return result


def _fake_runner() -> tuple[online.HeldoutOnlineMemoryVideo, list[dict[str, object]]]:
    runner = object.__new__(online.HeldoutOnlineMemoryVideo)
    runner.args = SimpleNamespace(seed=0)
    runner._first_frame_contract = None
    calls: list[dict[str, object]] = []

    def fake_sample(_key, observation, state, **kwargs):
        calls.append(kwargs)
        assert observation.state.shape[0] == 1
        write = kwargs["write_mode"] == "normal"
        assert kwargs["allow_write"] is write
        next_state = {"m": state["m"] + 1.0} if write else state
        retrieval = float(np.asarray(state["m"])[0, 0])
        return (
            jnp.zeros((1, 3, 4), dtype=jnp.float32),
            next_state,
            _aux(write=write, retrieval=retrieval),
        )

    runner.base = SimpleNamespace(
        model=SimpleNamespace(action_horizon=3, action_dim=4),
        stop_token=99,
        _sample=fake_sample,
        _state_max_abs_diff=lambda left, right: float(
            np.max(np.abs(np.asarray(left["m"]) - np.asarray(right["m"])))
        ),
    )
    return runner, calls


def test_args_force_batch_one_and_preserve_two_gpu_shards() -> None:
    assert _args().batch_size == 1
    assert _args(num_shards=2, shard_id=0).episodes == (15, 44)
    assert _args(num_shards=2, shard_id=1).episodes == (29, 59)
    with pytest.raises(ValueError, match="requires --batch-size 1"):
        _args(batch_size=2)


def test_stride15_schedule_and_registered_heldout_write_counts() -> None:
    assert [frame for frame in range(32) if online._is_scheduled_write(frame)] == [0, 15, 30]
    assert online._expected_write_count(1) == 1
    assert online._expected_write_count(15) == 1
    assert online._expected_write_count(16) == 2
    assert online._expected_write_count(876) == 59
    assert online._expected_write_count(863) == 58
    assert online._expected_write_count(740) == 50
    assert online.EXPECTED_WRITE_COUNTS == {15: 59, 29: 59, 44: 58, 59: 50}
    with pytest.raises(ValueError, match="nonnegative"):
        online._is_scheduled_write(-1)
    with pytest.raises(ValueError, match="positive"):
        online._expected_write_count(0)


def test_cross_static_retrieval_control_uses_recorded_absolute_1e5_tolerance() -> None:
    tolerance = online.CROSS_STATIC_MODE_RETRIEVAL_ATOL
    assert tolerance == 1e-5
    inside = online._cross_static_retrieval_comparison(0.0, tolerance * 0.999)
    boundary = online._cross_static_retrieval_comparison(0.0, tolerance)
    outside_value = float(np.nextafter(np.float64(tolerance), np.float64(np.inf)))
    outside = online._cross_static_retrieval_comparison(0.0, outside_value)
    assert inside == {
        "normal": 0.0,
        "frozen": tolerance * 0.999,
        "absolute_difference": tolerance * 0.999,
        "absolute_tolerance": tolerance,
        "passes": True,
    }
    assert boundary["passes"] is True
    assert boundary["absolute_difference"] == tolerance
    assert outside["passes"] is False
    assert outside["absolute_difference"] > tolerance
    with pytest.raises(ValueError, match="finite and nonnegative"):
        online._cross_static_retrieval_comparison(float("nan"), 0.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        online._cross_static_retrieval_comparison(-1e-6, 0.0)


def test_scheduled_inference_reads_every_frame_writes_only_due_and_threads_state() -> None:
    runner, calls = _fake_runner()
    observation = SimpleNamespace(state=jnp.zeros((1, 14), dtype=jnp.float32))
    state0 = {"m": jnp.zeros((1, 2), dtype=jnp.float32)}

    _actions, state1, aux0, contract0 = runner._infer_online(observation, state0, frame=0)
    assert calls[-1]["zero_read"] is False
    assert calls[-1]["allow_write"] is True
    assert calls[-1]["write_mode"] == "normal"
    assert contract0["write_occurred"] is True
    assert contract0["returned_state_changed"] is True
    assert contract0["retrieval_norm"] == 0.0
    assert np.array_equal(np.asarray(state1["m"]), np.ones((1, 2), dtype=np.float32))

    _actions, state2, aux1, contract1 = runner._infer_online(observation, state1, frame=1)
    assert calls[-1]["zero_read"] is False
    assert calls[-1]["allow_write"] is False
    assert calls[-1]["write_mode"] == "frozen"
    assert contract1["write_occurred"] is False
    assert contract1["frozen_state_bit_exact_unchanged"] is True
    assert contract1["retrieval_norm"] == 1.0
    assert runner._state_bit_exact_equal(state2, state1)

    _actions, state15, aux15, contract15 = runner._infer_online(observation, state2, frame=15)
    assert calls[-1]["allow_write"] is True
    assert calls[-1]["write_mode"] == "normal"
    assert contract15["write_occurred"] is True
    assert np.array_equal(np.asarray(state15["m"]), np.full((1, 2), 2.0, dtype=np.float32))
    assert bool(np.asarray(aux0["write_occurred"])[0])
    assert not bool(np.asarray(aux1["write_occurred"])[0])
    assert bool(np.asarray(aux15["write_occurred"])[0])

    with pytest.raises(ValueError, match="off-schedule"):
        runner._infer_online(observation, state2, frame=2, commit_write=True)


def test_frozen_same_prestate_control_matches_decode_read_and_never_changes_state() -> None:
    runner, _calls = _fake_runner()
    observation = SimpleNamespace(state=jnp.zeros((1, 14), dtype=jnp.float32))
    prestate = {"m": jnp.ones((1, 2), dtype=jnp.float32)}
    _a, frozen, frozen_aux = runner._infer_frozen_control(observation, prestate, frame=15)
    _a, normal, normal_aux, _contract = runner._infer_online(observation, prestate, frame=15)
    assert runner._state_bit_exact_equal(frozen, prestate)
    assert not runner._state_bit_exact_equal(normal, prestate)
    assert np.array_equal(np.asarray(frozen_aux["tokens"]), np.asarray(normal_aux["tokens"]))
    assert np.array_equal(np.asarray(frozen_aux["token_mask"]), np.asarray(normal_aux["token_mask"]))
    assert float(np.asarray(frozen_aux["retrieval_norm"])[0]) == float(
        np.asarray(normal_aux["retrieval_norm"])[0]
    )


def test_contract_rejects_wrong_write_state_change_nonfinite_aux_and_nonzero_eta() -> None:
    runner, _calls = _fake_runner()
    state0 = {"m": jnp.zeros((1, 2), dtype=jnp.float32)}
    state1 = {"m": jnp.ones((1, 2), dtype=jnp.float32)}

    with pytest.raises(RuntimeError, match="write_occurred mismatch"):
        runner._assert_online_inference_contract(
            state0,
            state0,
            _aux(write=False, retrieval=0.0),
            frame=0,
            commit_write=True,
        )
    with pytest.raises(RuntimeError, match="did not change"):
        runner._assert_online_inference_contract(
            state0,
            state0,
            _aux(write=True, retrieval=0.0),
            frame=0,
            commit_write=True,
        )
    with pytest.raises(RuntimeError, match="bit-exact"):
        runner._assert_online_inference_contract(
            state0,
            state1,
            _aux(write=False, retrieval=0.0),
            frame=1,
            commit_write=False,
        )
    bad_eta = _aux(write=False, retrieval=0.0)
    bad_eta["eta"] = jnp.asarray([1e-6], dtype=jnp.float32)
    with pytest.raises(RuntimeError, match="eta intervention"):
        runner._assert_online_inference_contract(
            state0,
            state0,
            bad_eta,
            frame=1,
            commit_write=False,
        )
    bad_retrieval = _aux(write=False, retrieval=float("nan"))
    with pytest.raises(RuntimeError, match="invalid retrieval_norm"):
        runner._assert_online_inference_contract(
            state0,
            state0,
            bad_retrieval,
            frame=1,
            commit_write=False,
        )


def test_bit_exact_state_check_distinguishes_positive_and_negative_zero() -> None:
    positive = {"m": jnp.asarray([[0.0]], dtype=jnp.float32)}
    negative = {"m": jnp.asarray([[-0.0]], dtype=jnp.float32)}
    assert online.HeldoutOnlineMemoryVideo._state_bit_exact_equal(positive, positive)
    assert not online.HeldoutOnlineMemoryVideo._state_bit_exact_equal(positive, negative)
