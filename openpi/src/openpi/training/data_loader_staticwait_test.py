"""Tests for the v3.4.1 static-waiting trim (leak fix 1).

Run: .venv/bin/python -m pytest src/openpi/training/data_loader_staticwait_test.py -q
"""

# ruff: noqa: SLF001, PT018 - these tests intentionally exercise the private trim helpers.

import dataclasses

import numpy as np
import pytest

from openpi import transforms as _transforms
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

# --------------------------------------------------------------------------------------
# _longest_static_run
# --------------------------------------------------------------------------------------


def test_static_run_all_quiet_returns_whole_window():
    window = np.zeros((40, 3), dtype=np.float32)
    assert _data_loader._longest_static_run(window, 1e-3, 1e-2) == (0, 39)


def test_static_run_trims_head_and_tail_motion():
    window = np.zeros((60, 2), dtype=np.float32)
    window[:10, 0] = np.linspace(0.5, 0.0, 10)  # settling into the pose
    window[45:, 0] = np.linspace(0.0, 0.6, 15)  # leaving toward the bin
    a, b = _data_loader._longest_static_run(window, 4e-3, 2e-2)
    assert a >= 9 and b <= 45
    assert b - a + 1 >= 30


def test_static_run_picks_longest_of_two_quiet_spans():
    window = np.zeros((100, 1), dtype=np.float32)
    window[20:24, 0] = np.linspace(0, 0.4, 4)  # a burst splits the window
    window[24:, 0] = 0.4
    a, b = _data_loader._longest_static_run(window, 4e-3, 2e-2)
    assert (a, b) == (24, 99)


def test_static_run_rejects_slow_creep_that_never_trips_speed():
    """A drift of 1e-3/frame never exceeds max_speed but accumulates past the excursion cap;
    the run must be shortened even though every individual step is quiet."""
    window = (np.arange(100, dtype=np.float32) * 1e-3)[:, None]
    a, b = _data_loader._longest_static_run(window, 4e-3, 2e-2)
    assert b - a + 1 <= 21
    span = window[a : b + 1]
    assert float(span.max() - span.min()) <= 2e-2


def test_static_run_returns_none_when_nothing_qualifies():
    window = (np.arange(30, dtype=np.float32) * 0.5)[:, None]
    assert _data_loader._longest_static_run(window, 1e-3, 1e-2) is None


def test_static_run_empty_window():
    assert _data_loader._longest_static_run(np.zeros((0, 3), np.float32), 1e-3, 1e-2) is None


# --------------------------------------------------------------------------------------
# _trim_waiting_to_static
# --------------------------------------------------------------------------------------


def _fake_info(lengths, waiting_bounds):
    return {
        "length": np.asarray(lengths, dtype=np.int32),
        "memory_lo": np.asarray([b[0] for b in waiting_bounds], dtype=np.int32),
        "memory_hi": np.asarray([b[1] for b in waiting_bounds], dtype=np.int32),
    }


def _cfg(**kw):
    return dataclasses.replace(
        _config.DataConfig(memory_waiting_max_speed=4e-3, memory_waiting_max_excursion=2e-2), **kw
    )


def test_trim_marks_only_nonstatic_waiting_frames_invalid():
    n = 100
    state = np.zeros((n, 2), dtype=np.float32)
    state[80:, 0] = np.linspace(0, 0.5, 20)  # motion starts at 80, inside the waiting phase
    info = _fake_info([n], [(40, 95)])
    _data_loader._trim_waiting_to_static(info, state, np.asarray([0]), np.asarray([n]), _cfg(), stride=15)

    assert int(info["memory_lo"][0]) == 40
    assert int(info["memory_hi"][0]) < 82
    valid = info["episode_waiting_valid"][0]
    assert valid[:40].all(), "frames before the waiting phase are untouched"
    assert valid[40:80].all(), "static waiting frames stay valid"
    assert not valid[85:96].any(), "moving waiting frames are dropped"
    assert valid[96:].all(), "frames after the waiting phase are untouched"


def test_trim_disables_branch_when_static_core_is_shorter_than_a_grid_step():
    n = 60
    state = np.zeros((n, 2), dtype=np.float32)
    state[30:, 0] = np.linspace(0, 0.5, 30)
    info = _fake_info([n], [(20, 55)])
    _data_loader._trim_waiting_to_static(info, state, np.asarray([0]), np.asarray([n]), _cfg(), stride=15)
    assert not bool(info["memory_critical_ok"][0])
    # the phase bounds stay accurate (the slice dead zone still needs them) and the honest
    # label mask survives even though the memory-critical branch is off for this episode
    assert int(info["memory_lo"][0]) == 20
    assert int(info["memory_hi"][0]) < 32
    assert info["episode_waiting_valid"][0][20:30].all()
    assert not info["episode_waiting_valid"][0][35:56].any()


def test_disabled_episode_yields_a_fully_disabled_memory_critical_window():
    """Regression: an episode taken out of the branch must have its whole window row zeroed.
    Leaving start_lo/evidence_start live while memory_lo is -1 sends memory_critical_endpoint
    into an empty eligible set, where the modulo divides by zero."""
    n = 60
    state = np.zeros((n, 2), dtype=np.float32)
    state[30:, 0] = np.linspace(0, 0.5, 30)
    info = _fake_info([n], [(20, 55)])
    info["evidence_start"] = np.asarray([5], dtype=np.int32)
    info["evidence_end"] = np.asarray([15], dtype=np.int32)
    cfg = _cfg(memory_critical_start_pad=4)
    _data_loader._trim_waiting_to_static(info, state, np.asarray([0]), np.asarray([n]), cfg, stride=15)
    assert not bool(info["memory_critical_ok"][0])
    window = _data_loader._memory_critical_windows(info, cfg)
    np.testing.assert_array_equal(window[0], [-1, -1, -1, -1])


def test_eligible_episode_keeps_its_window_and_places_static_endpoints():
    n = 200
    state = np.zeros((n, 2), dtype=np.float32)
    state[150:, 0] = np.linspace(0, 0.5, 50)  # motion late in the waiting phase
    info = _fake_info([n], [(100, 180)])
    info["evidence_start"] = np.asarray([40], dtype=np.int32)
    info["evidence_end"] = np.asarray([60], dtype=np.int32)
    cfg = _cfg(memory_critical_start_pad=20)
    _data_loader._trim_waiting_to_static(info, state, np.asarray([0]), np.asarray([n]), cfg, stride=15)
    assert bool(info["memory_critical_ok"][0])
    window = _data_loader._memory_critical_windows(info, cfg)[0]
    for start in range(int(window[0]), int(window[1]) + 1):
        step = _transforms.memory_critical_endpoint(start, window, stride=15, lookahead=15, num_steps=40)
        frame = min(start + step * 15, n - 1)
        assert float(np.abs(state[frame] - state[100]).max()) <= 2e-2, f"endpoint {frame} is in motion"


def test_trim_leaves_episodes_without_a_waiting_phase_alone():
    n = 50
    state = np.zeros((n, 2), dtype=np.float32)
    info = _fake_info([n], [(-1, -1)])
    _data_loader._trim_waiting_to_static(info, state, np.asarray([0]), np.asarray([n]), _cfg(), stride=15)
    assert int(info["memory_lo"][0]) == -1
    assert info["episode_waiting_valid"][0].all()


def test_trim_is_a_noop_on_an_already_static_phase():
    n = 80
    state = np.zeros((n, 3), dtype=np.float32)
    info = _fake_info([n], [(20, 70)])
    _data_loader._trim_waiting_to_static(info, state, np.asarray([0]), np.asarray([n]), _cfg(), stride=15)
    assert (int(info["memory_lo"][0]), int(info["memory_hi"][0])) == (20, 70)
    assert info["episode_waiting_valid"][0].all()


# --------------------------------------------------------------------------------------
# label plumbing
# --------------------------------------------------------------------------------------

VOCAB = ("open both lids", "wait; target bin is left", "inspect both bins", "open left bin")


def _labels(**kw):
    return _transforms.MemoryV34Labels(
        subtask_vocab=VOCAB,
        evidence_subtasks=("inspect both bins",),
        memory_required_subtasks=("wait; target bin is left",),
        **kw,
    )


def test_invalid_waiting_steps_lose_the_aux_target_and_the_waiting_mask():
    wait, evid = "wait; target bin is left", "inspect both bins"
    data = {
        "subtask": [evid, wait, wait, wait],
        "subtask_now": [evid, wait, wait, wait],
        "subtask_valid": np.asarray([True, True, False, True]),
        "subtask_now_valid": np.asarray([True, False, False, True]),
        "episode_side": np.int32(0),
    }
    out = _labels()(dict(data))
    np.testing.assert_array_equal(out["seq_subtask_class"], [2, 1, -1, 1])
    np.testing.assert_array_equal(out["seq_waiting_mask"], [False, False, False, True])
    # evidence frames are never gated by motion
    np.testing.assert_array_equal(out["seq_evidence_mask"], [True, False, False, False])
    assert "subtask_valid" not in out and "subtask_now_valid" not in out


def test_labels_unchanged_when_no_validity_supplied():
    wait = "wait; target bin is left"
    data = {"subtask": [wait, wait], "subtask_now": [wait, wait], "episode_side": np.int32(0)}
    out = _labels()(dict(data))
    np.testing.assert_array_equal(out["seq_subtask_class"], [1, 1])
    np.testing.assert_array_equal(out["seq_waiting_mask"], [True, True])


def test_state_mask_not_drawn_when_every_waiting_target_was_dropped():
    wait = "wait; target bin is left"
    data = {
        "subtask": [wait, wait],
        "subtask_now": [wait, wait],
        "subtask_valid": np.asarray([False, False]),
        "subtask_now_valid": np.asarray([False, False]),
        "episode_side": np.int32(0),
        "seq_step_mask": np.asarray([True, True]),
    }
    out = _labels(state_mask_prob=1.0)(dict(data))
    assert not bool(out["seq_state_masked"]), "a segment with no valid waiting target is not memory-required"


def test_sequence_subtasks_emits_shifted_and_unshifted_validity():
    tasks = {0: "open both lids", 1: "wait; target bin is left"}
    ep_tasks = np.asarray([0] * 10 + [1] * 10)
    valid = np.ones(20, dtype=bool)
    valid[15:] = False  # the tail of the waiting phase is moving
    tf = _transforms.MemorySequenceSubtasks(
        stride=5, steps=3, lookahead=5, episode_tasks=(ep_tasks,), tasks=tasks, episode_waiting_valid=(valid,)
    )
    out = tf({"episode_index": np.int32(0), "frame_index": np.int32(5)})
    # observations at frames 5, 10, 15; shifted targets at 10, 15, 19 (clipped)
    np.testing.assert_array_equal(out["subtask_now_valid"], [True, True, False])
    np.testing.assert_array_equal(out["subtask_valid"], [True, False, False])


def test_run6_config_enables_the_trim_and_run5_does_not():
    run6 = _config.get_config("pi05_yam_mem_v34_run6_staticwait")
    run5 = _config.get_config("pi05_yam_mem_v34_run5_eta0")
    assert run6.data.base_config.memory_waiting_max_speed == 4e-3
    assert run6.data.base_config.memory_waiting_max_excursion == 0.02
    assert run5.data.base_config.memory_waiting_max_speed is None
    # the only intended difference is the waiting trim
    assert run6.model == run5.model
    assert run6.batch_size == run5.batch_size
    assert run6.data.base_config.heldout_episodes == run5.data.base_config.heldout_episodes


if __name__ == "__main__":
    pytest.main([__file__, "-q"])


def test_inference_items_drop_the_validity_fields():
    """Non-sequence (inference) items take the early return; the trim's per-step flags must not
    survive it -- an unregistered field with a leading length of max_steps trips the sequence
    collate guard."""
    out = _labels()({"state": np.zeros(14), "subtask_valid": np.ones(3, bool), "subtask_now_valid": np.ones(3, bool)})
    assert "subtask_valid" not in out
    assert "subtask_now_valid" not in out
