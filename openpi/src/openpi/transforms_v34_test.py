"""v3.4 data-transform tests: MemoryV34Labels (plan 5.1/5.2/Section 6 supervision fields)."""

import numpy as np

from openpi import transforms

_VOCAB = (
    "open both lids",
    "wait; target bin is left",
    "open left bin",
    "close both lids and reset arms",
    "inspect both bins",
    "open right bin",
    "wait; target bin is right",
)


def _labels(state_mask_prob=0.0):
    return transforms.MemoryV34Labels(
        subtask_vocab=_VOCAB,
        evidence_subtasks=("inspect both bins",),
        memory_required_subtasks=("wait; target bin is left", "wait; target bin is right"),
        state_mask_prob=state_mask_prob,
    )


def _segment(shifted, now, *, episode_side=1, step_mask=None):
    data = {
        "subtask": list(shifted),
        "subtask_now": list(now),
        "episode_side": np.int32(episode_side),
    }
    if step_mask is not None:
        data["seq_step_mask"] = np.asarray(step_mask)
    return data


def test_labels_classes_masks_and_side():
    shifted = ["inspect both bins", "close both lids and reset arms", "wait; target bin is right", "mystery"]
    now = ["open both lids", "inspect both bins", "close both lids and reset arms", "wait; target bin is right"]
    out = _labels()(_segment(shifted, now, episode_side=1))
    np.testing.assert_array_equal(out["seq_subtask_class"], [4, 3, 6, -1])
    # phase masks come from the UNSHIFTED labels (what the observation shows)
    np.testing.assert_array_equal(out["seq_evidence_mask"], [False, True, False, False])
    np.testing.assert_array_equal(out["seq_waiting_mask"], [False, False, False, True])
    assert int(out["seq_side_label"]) == 1
    assert "subtask_now" not in out
    assert "episode_side" not in out
    assert "seq_state_masked" not in out  # prob 0 -> field absent


def test_side_falls_back_to_the_window_waiting_label():
    shifted = ["wait; target bin is left"] * 2
    now = ["close both lids and reset arms"] * 2
    out = _labels()(_segment(shifted, now, episode_side=-1))
    assert int(out["seq_side_label"]) == 0


def test_state_mask_only_on_memory_required_segments():
    rng = np.random.RandomState(0)
    np.random.seed(0)
    labels = _labels(state_mask_prob=1.0)
    # a segment whose valid steps include a waiting CE target IS eligible
    eligible = labels(
        _segment(
            ["inspect both bins", "wait; target bin is left"],
            ["open both lids", "inspect both bins"],
            step_mask=[True, True],
        )
    )
    assert bool(eligible["seq_state_masked"])
    # the waiting target being MASKED OUT (past the truncation) removes eligibility
    truncated = labels(
        _segment(
            ["inspect both bins", "wait; target bin is left"],
            ["open both lids", "inspect both bins"],
            step_mask=[True, False],
        )
    )
    assert not bool(truncated["seq_state_masked"])
    # no waiting label anywhere -> never masked
    neutral = labels(
        _segment(["open both lids", "inspect both bins"], ["open both lids", "open both lids"], step_mask=[True, True])
    )
    assert not bool(neutral["seq_state_masked"])
    del rng


def test_state_mask_probability_is_per_segment():
    np.random.seed(1234)
    labels = _labels(state_mask_prob=0.5)
    seg = _segment(
        ["wait; target bin is right"] * 3,
        ["close both lids and reset arms"] * 3,
        step_mask=[True, True, True],
    )
    draws = [bool(labels(dict(seg))["seq_state_masked"]) for _ in range(400)]
    rate = np.mean(draws)
    assert 0.4 < rate < 0.6, rate


def test_inference_items_pass_through_untouched():
    labels = _labels(state_mask_prob=1.0)
    data = {"prompt": "find the banana", "subtask_now": ["x"], "episode_side": np.int32(1)}
    out = labels(dict(data))
    assert "seq_subtask_class" not in out
    assert "subtask_now" not in out
    assert "episode_side" not in out
