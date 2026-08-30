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


def test_writer_clip_statistics_uses_preclip_threshold():
    stats = wa.writer_clip_statistics(np.array([0.5, 2.0, 30.0, 100.0]), max_grad_norm=1.0)
    assert stats["write_grad_norm_preclip_min"] == pytest.approx(0.5)
    assert stats["write_grad_norm_preclip_median"] == pytest.approx(16.0)
    assert stats["write_grad_norm_preclip_p95"] == pytest.approx(89.5)
    assert stats["write_grad_norm_preclip_max"] == pytest.approx(100.0)
    assert stats["write_clip_saturation_fraction"] == pytest.approx(0.75)
    assert stats["write_clip_factor_median"] == pytest.approx((1.0 / 2.0 + 1.0 / 30.0) / 2.0)


def test_writer_clip_statistics_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="non-empty vector"):
        wa.writer_clip_statistics(np.empty(0), max_grad_norm=1.0)
    with pytest.raises(ValueError, match="finite nonnegative"):
        wa.writer_clip_statistics(np.array([1.0, np.nan]), max_grad_norm=1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        wa.writer_clip_statistics(np.array([1.0]), max_grad_norm=0.0)


def test_retrieval_scale_statistics_excludes_blank_reads():
    stats = wa.retrieval_scale_statistics(
        h8_valid_rms=np.array([2.0, 2.0, 3.0]),
        h8_valid_token_count=np.array([4, 4, 8]),
        h8_image_rms=np.array([2.0, 2.0, 2.5]),
        h8_context_valid_rms=np.array([2.0, 2.0, 4.0]),
        retrieved_rms=np.array([0.0, 0.5, 1.5]),
        memory_token_rms=np.array([0.0, 0.01, 0.03]),
    )
    assert stats["h8_valid_rms_median"] == pytest.approx(2.0)
    assert stats["retrieved_rms_median"] == pytest.approx(0.5)
    assert stats["retrieval_zero_fraction"] == pytest.approx(1.0 / 3.0)
    assert stats["retrieval_match_c_count"] == 2
    assert stats["retrieval_match_c_median"] == pytest.approx(3.0)
    assert stats["retrieval_match_c_p05"] == pytest.approx(2.1)
    assert stats["retrieval_match_c_p95"] == pytest.approx(3.9)
    assert stats["memory_token_to_h8_ratio_median"] == pytest.approx(0.005)
    expected_h8_energy = np.sqrt((2.0**2 * 4 + 2.0**2 * 4 + 3.0**2 * 8) / 16)
    expected_retrieved_energy = np.sqrt((0.0**2 + 0.5**2 + 1.5**2) / 3)
    assert stats["retrieval_match_c_energy"] == pytest.approx(expected_h8_energy / expected_retrieved_energy)


def test_retrieval_scale_statistics_all_blank_has_no_c():
    stats = wa.retrieval_scale_statistics(
        np.ones(2), np.ones(2), np.ones(2), np.ones(2), np.zeros(2), np.zeros(2)
    )
    assert stats["retrieval_match_c_count"] == 0
    assert stats["retrieval_match_c_median"] is None
    assert stats["retrieval_match_c_p05"] is None
    assert stats["retrieval_match_c_p95"] is None


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


def test_v32_style_tile_matches_the_v32_convention():
    """The v32 style must reproduce v32_checkpoint._attention_video's math exactly: per-frame
    per-slot max normalization, JET, opacity = 0.6 * intensity. A uniform map and a peaked map
    both saturate their own peak -- that relativeness IS the convention."""
    import cv2

    image = np.full((224, 224, 3), 100, dtype=np.uint8)
    slot_map = np.zeros(256)
    slot_map[137] = 1.0  # patch (8, 9)
    slot_map[0] = 0.5

    tile = wa._v32_style_tile(image, slot_map)
    grid = slot_map.reshape(16, 16)
    heat = grid / grid.max()
    heat224 = cv2.resize(heat.astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST)
    color = cv2.applyColorMap((heat224 * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    weight = (0.6 * heat224)[..., None]
    expected = (image.astype(np.float32) * (1 - weight) + color * weight).astype(np.uint8)
    np.testing.assert_array_equal(tile, expected)

    # the peak patch is JET-red (red channel dominates, blue suppressed) and visibly shifted
    # from the raw image; an unattended patch keeps the raw image untouched
    peak = tile[8 * 14 + 7, 9 * 14 + 7]
    assert peak[0] > peak[2], f"peak should be red-dominant, got RGB {peak}"
    assert peak[0] > image[0, 0, 0], "peak should be brighter in red than the raw image"
    np.testing.assert_array_equal(tile[15 * 14 + 7, 15 * 14 + 7], image[0, 0])

    # scaling the whole map changes nothing (purely relative)
    np.testing.assert_array_equal(wa._v32_style_tile(image, slot_map * 17.0), tile)


def test_slot_grid_frame_style_validation_and_shape():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    maps = np.full((16, 256), 1.0 / 256)
    frame = wa.slot_grid_frame(image, maps, None, "label", style="v32")
    assert frame.shape == (wa._SLOT_GRID_HEADER + 4 * 224, 4 * 224, 3)
    with pytest.raises(ValueError, match="unsupported slot grid style"):
        wa.slot_grid_frame(image, maps, None, "label", style="rainbow")
    with pytest.raises(ValueError, match="needs one color scale per slot"):
        wa.slot_grid_frame(image, maps, None, "label", style="video")


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
