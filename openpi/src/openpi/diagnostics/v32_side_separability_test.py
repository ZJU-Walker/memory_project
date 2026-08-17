"""Tests for the v3.2 side-separability metrics on synthetic token trajectories."""

import numpy as np
import pytest

from openpi.diagnostics import v32_side_separability as sep


@pytest.fixture
def synthetic_pair() -> tuple[np.ndarray, np.ndarray]:
    """Two [57, 16, 32] episodes: identical until step 16, side-shifted afterwards."""
    rng = np.random.default_rng(0)
    drift = 0.02 * np.cumsum(rng.normal(size=(57, 1, 32)), axis=0)
    base = (rng.normal(size=32) + drift).repeat(16, axis=1)
    jitter = 0.05 * rng.normal(size=(2, 57, 16, 32))
    side_axis = rng.normal(size=32)
    shift = np.zeros((57, 1, 1))
    shift[16:] = 1.0
    left = base + jitter[0] + 0.8 * shift * side_axis
    right = base + jitter[1] - 0.8 * shift * side_axis
    return left.astype(np.float32), right.astype(np.float32)


def test_side_signal_emerges_at_reveal_and_persists(synthetic_pair):
    left, right = synthetic_pair
    result = sep.analyze_stage(left, right)
    phases = result["per_phase"]
    assert phases["pre_reveal"]["separation_ratio"] < 2.0
    assert phases["reveal"]["separation_ratio"] > 5.0
    assert phases["pre_reveal"]["decode_from_reveal"] <= 0.6
    for name in ("reveal", "post_closure", "decision_closed"):
        assert phases[name]["decode_from_reveal"] == 1.0
    proj_left = np.asarray(result["lr_projection_left"])
    proj_right = np.asarray(result["lr_projection_right"])
    assert np.all(proj_left[16:] > 0)
    assert np.all(proj_right[16:] < 0)


def test_no_signal_when_episodes_share_content():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(57, 1, 32)).repeat(16, axis=1)
    left = (base + 0.05 * rng.normal(size=(57, 16, 32))).astype(np.float32)
    right = (base + 0.05 * rng.normal(size=(57, 16, 32))).astype(np.float32)
    result = sep.analyze_stage(left, right)
    for phase in result["per_phase"].values():
        assert phase["separation_ratio"] < 2.5


def test_slot_uniformity_is_one_for_identical_slots():
    tokens = np.ones((3, 16, 8), dtype=np.float32)
    np.testing.assert_allclose(sep.slot_uniformity(tokens), 1.0, atol=1e-6)
