"""Tests for the v3.3 transform pieces: per-episode prompt injection and the memory-critical
endpoint truncation in BuildMemorySequence."""

import numpy as np
import pytest

import openpi.transforms as _transforms


def _raw_sequence_item(*, frame_index: int, steps: int = 12, window=None):
    item = {
        "frame_index": np.int64(frame_index),
        "index": np.int64(0),
        "episode_length": np.int32(200),
        "observation/image": np.zeros((steps, 4, 4, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.zeros((steps, 4, 4, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.zeros((steps, 4, 4, 3), dtype=np.uint8),
        "observation/state": np.zeros((steps, 14), dtype=np.float32),
        "actions": np.zeros((steps, 3 * 14), dtype=np.float32),
    }
    if window is not None:
        item["memory_window"] = np.asarray(window, dtype=np.int32)
    return item


def _build(**kwargs) -> _transforms.BuildMemorySequence:
    defaults = {"stride": 15, "action_horizon": 3, "block_steps": 4, "subtask_lookahead": 5}
    return _transforms.BuildMemorySequence(**{**defaults, **kwargs})


def test_prompt_from_episode_injects_and_rejects_missing():
    inject = _transforms.InjectPromptFromEpisode(("find the banana", "find the grey pepper box"))
    out = inject({"episode_index": np.int64(1)})
    assert str(out["prompt"]) == "find the grey pepper box"
    with pytest.raises(ValueError, match="episode 2"):
        inject({"episode_index": np.int64(2)})


def test_normal_sample_is_untouched_and_fenced():
    out = _build()(_raw_sequence_item(frame_index=0, window=[30, 60, 140, 159]))
    # start outside the window: full episode-length mask, fences present, key consumed
    np.testing.assert_array_equal(out["seq_step_mask"], np.arange(12) * 15 < 200)
    assert out["seq_block_boundary"].any()
    assert not out["seq_block_boundary"][0]
    assert "memory_window" not in out


def test_memory_critical_sample_truncates_inside_wait_and_carries_no_fence():
    # start 45, grid 45+15k: only k=7 (frame 150) lies in [memory_lo=140, memory_hi-lookahead=154]
    out = _build()(_raw_sequence_item(frame_index=45, window=[30, 60, 140, 159]))
    np.testing.assert_array_equal(out["seq_step_mask"], np.arange(12) <= 7)
    assert not out["seq_block_boundary"].any()


def test_short_wait_falls_back_to_ignoring_the_lookahead():
    # wait spans 13 frames < lookahead 15 (the shortest wait in the 0816 data): tier-2 rule
    out = _build(subtask_lookahead=15)(_raw_sequence_item(frame_index=45, window=[30, 60, 147, 159]))
    np.testing.assert_array_equal(out["seq_step_mask"], np.arange(12) <= 7)  # frame 150 in [147, 159]


def test_straddled_wait_ends_on_the_last_neutral_step_before_it():
    # the stride grid skips [151, 158] entirely (150 < 151, 165 > 158): tier-3 rule ends at
    # k=7 (frame 150), whose lookahead target is already a memory-required label
    out = _build()(_raw_sequence_item(frame_index=45, window=[30, 60, 151, 158]))
    np.testing.assert_array_equal(out["seq_step_mask"], np.arange(12) <= 7)
    assert not out["seq_block_boundary"].any()


def test_endpoints_are_deterministic_per_start_and_stratified_across_starts():
    """The same start must always truncate identically (exact bucket assignment depends on
    it), while consecutive starts cycle through the eligible waiting-phase endpoints."""
    build = _build(subtask_lookahead=0)
    lengths = {}
    for start in range(30, 38):
        out = build(_raw_sequence_item(frame_index=start, window=[30, 60, 140, 199]))
        again = build(_raw_sequence_item(frame_index=start, window=[30, 60, 140, 199]))
        np.testing.assert_array_equal(out["seq_step_mask"], again["seq_step_mask"])
        lengths[start] = int(out["seq_step_mask"].sum())
    assert len(set(lengths.values())) > 1, f"no endpoint diversity across starts: {lengths}"
    # every endpoint observation lies in the waiting phase [140, 199]
    for start, n in lengths.items():
        assert 140 <= start + (n - 1) * 15 <= 199


def test_disabled_window_row_means_normal_sample():
    out = _build()(_raw_sequence_item(frame_index=45, window=[-1, -1, -1, -1]))
    np.testing.assert_array_equal(out["seq_step_mask"], 45 + np.arange(12) * 15 < 200)
