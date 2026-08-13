import numpy as np
import pytest

from openpi.diagnostics import frame380_memory_swap


def test_vector_comparison_exact_cases():
    same = frame380_memory_swap.vector_comparison(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert same["difference_l2"] == 0
    assert same["cosine"] == pytest.approx(1.0)

    orthogonal = frame380_memory_swap.vector_comparison(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert orthogonal["difference_l2"] == pytest.approx(np.sqrt(2))
    assert orthogonal["cosine"] == pytest.approx(0.0)


def test_action_motion_identifies_larger_arm_and_rejects_shapes():
    state = np.zeros(14)
    actions = np.zeros((50, 14))
    actions[:, :7] = 0.2
    actions[:, 7:] = 0.1
    result = frame380_memory_swap.action_motion(actions, state)
    assert result["larger_motion_arm"] == "left"
    assert result["trajectory_left_minus_right"] > 0

    with pytest.raises(ValueError, match="expected actions"):
        frame380_memory_swap.action_motion(np.zeros((50, 13)), state)
