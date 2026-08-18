"""Tests for the pure-math helpers of the v3.3 writer-attention diagnostic."""

# ruff: noqa: SLF001 - the private planning helpers are part of the contract under test.

import numpy as np
import pytest

from openpi.diagnostics import v33_writer_attention as wa


def test_js_divergence_bounds():
    p = np.full((3, 256), 1.0 / 256)
    np.testing.assert_allclose(wa.js_divergence(p, p), 0.0, atol=1e-12)
    a = np.zeros((1, 256))
    b = np.zeros((1, 256))
    a[0, 0] = 1.0
    b[0, 255] = 1.0
    np.testing.assert_allclose(wa.js_divergence(a, b), np.log(2.0), rtol=1e-6)


def test_js_divergence_normalizes_unnormalized_rows():
    rng = np.random.default_rng(0)
    p = rng.random((4, 256))
    np.testing.assert_allclose(wa.js_divergence(p, 3.0 * p), 0.0, atol=1e-9)


def test_left_half_mass_splits_the_grid():
    maps = np.zeros((2, 256))
    grid = maps.reshape(2, 16, 16)
    grid[:, :, :8] = 1.0  # all mass in the left half
    assert wa.left_half_mass(maps) == pytest.approx(1.0)
    grid[:, :, 8:] = 1.0  # now uniform
    assert wa.left_half_mass(maps) == pytest.approx(0.5)


def test_slot_entropy_uniform_is_log_n():
    maps = np.full((16, 256), 1.0 / 256)
    assert wa.slot_entropy(maps) == pytest.approx(np.log(256.0), rel=1e-6)
    peaked = np.zeros((1, 256))
    peaked[0, 7] = 1.0
    assert wa.slot_entropy(peaked) == pytest.approx(0.0, abs=1e-6)


def test_head_mean_shape_contract():
    attention = np.random.default_rng(1).random((2, 8, 16, 256))
    out = wa.head_mean(attention)
    assert out.shape == (16, 256)
    np.testing.assert_allclose(out, attention[0].mean(axis=0))
    with pytest.raises(ValueError, match="attention must be"):
        wa.head_mean(attention[0])


def test_param_unwraps_orbax_value_leaves():
    raw = {"write_query_conditioner": {"output_proj": {"kernel": {"value": np.ones((2, 2))}}}}
    np.testing.assert_array_equal(wa._param(raw, "write_query_conditioner", "output_proj", "kernel"), np.ones((2, 2)))
    with pytest.raises(KeyError, match="missing parameter path"):
        wa._param(raw, "write_query_conditioner", "missing")


def test_pathway_scalars_reports_both_norms():
    raw = {
        "write_query_conditioner": {"output_proj": {"kernel": {"value": np.zeros((4, 4))}}},
        "memory_gate": {"value": np.full((16, 8), 3.0)},
    }
    scalars = wa.pathway_scalars(raw)
    assert scalars["conditioner_output_proj_norm"] == pytest.approx(0.0)
    assert scalars["memory_gate_norm"] == pytest.approx(np.sqrt(16 * 8 * 9.0))


def test_slot_grid_frame_places_each_slot_tile():
    """16 per-slot tiles in a 4x4 grid under a header bar; a slot's hot patch must light up
    inside that slot's tile and nowhere else in its row/column neighborhood."""
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    maps = np.zeros((16, 256))
    maps[:, 0] = 1e-6  # keep every slot's scale finite
    maps[5, 137] = 1.0  # slot 5 (grid row 1, column 1), patch (8, 9)
    scales = wa._slot_scales(maps[None])
    frame = wa.slot_grid_frame(image, maps, scales, "label")
    assert frame.shape == (wa._SLOT_GRID_HEADER + 4 * 224, 4 * 224, 3)
    assert frame.dtype == np.uint8

    def tile(row, column):
        top = wa._SLOT_GRID_HEADER + row * 224
        return frame[top : top + 224, column * 224 : (column + 1) * 224]

    # patch (8, 9) of slot 5's tile: pixels [8*14:9*14, 9*14:10*14]
    hot = tile(1, 1)[8 * 14 : 9 * 14, 9 * 14 : 10 * 14]
    cold = tile(1, 2)[8 * 14 : 9 * 14, 9 * 14 : 10 * 14]
    assert hot.astype(int).sum() > cold.astype(int).sum()
    with pytest.raises(ValueError, match="per-frame slot maps"):
        wa.slot_grid_frame(image, maps[:4], scales, "label")


class _PhaseConfig:
    evidence_subtasks = ("inspect both bins",)
    memory_required_subtasks = ("wait; target bin is left", "wait; target bin is right")


def test_episode_phases_derivation():
    tasks = {
        0: "open both lids",
        1: "inspect both bins",
        2: "close both lids and reset arms",
        3: "wait; target bin is left",
    }
    ids = np.array([0] * 5 + [1] * 3 + [2] * 4 + [3] * 2)
    evidence, memory = wa._episode_phases(ids, tasks, _PhaseConfig())
    assert evidence == (5, 7)
    assert memory == (12, 13)
    assert wa._episode_phases(np.array([0, 0, 2]), tasks, _PhaseConfig()) is None  # no phases
    assert wa._episode_phases(np.array([3, 1]), tasks, _PhaseConfig()) is None  # wait before evidence
