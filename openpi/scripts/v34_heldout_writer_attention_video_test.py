"""Focused fail-closed tests for the every-frame heldout writer-attention video."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

# The diagnostic intentionally exposes script-private pure helpers for contract tests.
# ruff: noqa: SLF001
_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS))
try:
    import v34_heldout_writer_attention_video as attention
finally:
    sys.path.remove(str(_SCRIPTS))


def _args(**overrides) -> attention.Args:
    values = {
        "checkpoint": Path("checkpoint/11000"),
        "dataset_root": Path("dataset"),
        "output_dir": Path("output/11000"),
        "config": attention.RUN5_CONFIG,
        "parameter_source": "raw",
    }
    values.update(overrides)
    return attention.Args(**values)


def _uniform_attention() -> tuple[np.ndarray, np.ndarray]:
    valid = attention._expected_source_valid()
    values = np.zeros((1, attention.NUM_HEADS, attention.NUM_SLOTS, attention.NUM_PATCHES), dtype=np.float32)
    values[..., valid] = 1.0 / valid.sum()
    return values, valid


def test_args_are_generic_numeric_but_run5_and_parameter_source_are_explicit() -> None:
    args = _args()
    assert args.checkpoint_step == 11000
    assert args.episodes == attention.HELDOUT_EPISODES
    assert args.artifact_dir.name == "raw"
    assert _args(parameter_source="ema", smoke_only=True).episodes == (attention.SMOKE_EPISODE,)
    for overrides, match in (
        ({"checkpoint": Path("checkpoint/latest")}, "numeric"),
        ({"config": "pi05_yam_mem_v34"}, "pinned"),
        ({"parameter_source": "both"}, "raw or ema"),
    ):
        with pytest.raises(ValueError, match=match):
            _args(**overrides)


def test_commit_schedule_includes_frame_zero_and_smoke_exercises_two_commits() -> None:
    assert attention._is_scheduled_commit(0)
    assert attention._is_scheduled_commit(15)
    assert not attention._is_scheduled_commit(14)
    assert attention._expected_commit_count(attention.SMOKE_FRAME_COUNT) == 2
    assert attention.FULL_TOTAL_FRAME_COUNT == 3355
    assert [attention._expected_commit_count(count) for count in attention.EXPECTED_FRAME_COUNTS.values()] == [
        attention.EXPECTED_COMMIT_COUNTS[episode] for episode in attention.HELDOUT_EPISODES
    ]
    with pytest.raises(ValueError, match="nonnegative"):
        attention._is_scheduled_commit(-1)
    with pytest.raises(ValueError, match="positive"):
        attention._expected_commit_count(0)


def test_unmodified_qstep_timing_and_call_accounting_are_explicit() -> None:
    assert attention.QSTEP_EXECUTABLE_DESCRIPTION == (
        "unmodified nnx_utils.module_jit(model.v32_query_attention_step)"
    )
    source = Path(attention.__file__).read_text(encoding="utf-8")
    assert "self._qstep = nnx_utils.module_jit(self.runtime.model.v32_query_attention_step)" in source

    assert attention._qstep_call_accounting(17, smoke_only=True) == {
        "recurrence_read_attention_calls": 17,
        "recurrence_candidate_writer_calls": 17,
        "smoke_fresh_M0_control_qstep_calls": 1,
        "total_qstep_calls_including_smoke_controls": 18,
    }
    assert attention._qstep_call_accounting(876, smoke_only=False)[
        "total_qstep_calls_including_smoke_controls"
    ] == 876

    timings = np.full(attention.SMOKE_FRAME_COUNT, 0.25, dtype=np.float64)
    timings[0] = 99.0
    timing = attention._smoke_timing_gate(timings)
    assert timing["compile_excluded_frame"] == 0
    assert timing["steady_timed_frames"] == list(range(1, attention.SMOKE_FRAME_COUNT))
    assert timing["steady_unmodified_qstep_seconds_mean"] == pytest.approx(0.25)
    assert timing["extrapolated_full_qstep_seconds_from_mean"] == pytest.approx(
        0.25 * attention.FULL_TOTAL_FRAME_COUNT
    )
    assert timing["status"] == "pass"
    with pytest.raises(RuntimeError, match="invalid smoke qstep timings"):
        attention._smoke_timing_gate(timings[:-1])


def test_source_valid_is_exact_letterbox_rows_two_through_thirteen() -> None:
    valid = attention._validate_source_valid(attention._expected_source_valid())
    grid = valid.reshape(attention.GRID_SIZE, attention.GRID_SIZE)
    assert not grid[:2].any()
    assert grid[2:14].all()
    assert not grid[14:].any()
    assert int(valid.sum()) == attention.VALID_PATCH_COUNT
    broken = valid.copy()
    broken[0] = True
    with pytest.raises(ValueError, match="letterbox-valid"):
        attention._validate_source_valid(broken)


def test_attention_reduction_preserves_probability_and_rejects_padding_mass() -> None:
    values, valid = _uniform_attention()
    slots, mean_map = attention._head_mean_per_slot(values, valid)
    assert slots.shape == (attention.NUM_SLOTS, attention.NUM_PATCHES)
    assert mean_map.shape == (attention.NUM_PATCHES,)
    np.testing.assert_allclose(slots.sum(axis=-1), 1.0, atol=1e-6)
    assert float(mean_map.sum()) == pytest.approx(1.0)
    assert float(mean_map[~valid].max()) == 0.0

    padding = values.copy()
    padding[..., 0] = 1e-4
    first_valid = int(np.flatnonzero(valid)[0])
    padding[..., first_valid] -= 1e-4
    with pytest.raises(RuntimeError, match="invalid patches"):
        attention._head_mean_per_slot(padding, valid)
    with pytest.raises(ValueError, match="shape"):
        attention._head_mean_per_slot(values[:, :1], valid)
    nan_values = values.copy()
    nan_values[..., first_valid] = np.nan
    with pytest.raises(FloatingPointError, match="NaN"):
        attention._head_mean_per_slot(nan_values, valid)


def test_one_hot_patch_layout_is_row_major_without_transpose_or_flip() -> None:
    row, column = 3, 11
    one_hot = np.zeros(attention.NUM_PATCHES, dtype=np.float32)
    one_hot[row * attention.GRID_SIZE + column] = 1.0
    grid = attention.token_heatmap.token_grid(one_hot)
    assert np.unravel_index(int(np.argmax(grid)), grid.shape) == (row, column)
    assert grid[column, row] == 0.0
    assert "row-major" in attention.token_heatmap.TOKEN_LAYOUT
    assert "no transpose or flip" in attention.token_heatmap.TOKEN_LAYOUT


def test_global_scale_uses_all_selected_valid_maps_and_render_has_exact_geometry() -> None:
    values, valid = _uniform_attention()
    slots, mean_map = attention._head_mean_per_slot(values, valid)
    result_a = SimpleNamespace(episode=15, mean_maps=np.stack([mean_map, mean_map * 1.5]))
    result_b = SimpleNamespace(episode=29, mean_maps=np.stack([mean_map * 0.5]))
    scale = attention._fit_global_scale([result_a, result_b], valid)
    expected = np.percentile(
        np.concatenate([result_a.mean_maps[:, valid].ravel(), result_b.mean_maps[:, valid].ravel()]),
        attention.GLOBAL_SCALE_PERCENTILE,
    )
    assert scale.vmax == pytest.approx(expected)
    assert scale.excluded_letterbox_padding

    image = np.full((attention.token_heatmap.MODEL_IMAGE_SIZE, attention.token_heatmap.MODEL_IMAGE_SIZE, 3), 80, dtype=np.uint8)
    rendered = attention._render_main_frame(
        image,
        mean_map,
        scale,
        episode=15,
        frame=15,
        frame_count=17,
        fps=30.0,
        prompt="find the banana",
        side="left",
        gt="open both lids",
        checkpoint_step=11000,
        parameter_source="raw",
        scheduled_commit=True,
    )
    assert rendered.shape == (attention.MAIN_HEIGHT, attention.MAIN_WIDTH, 3)
    assert rendered.dtype == np.uint8
    expected_left = np.repeat(
        np.repeat(image, attention.MAIN_DISPLAY_SCALE, axis=0), attention.MAIN_DISPLAY_SCALE, axis=1
    )
    np.testing.assert_array_equal(rendered[attention.MAIN_HEADER_HEIGHT :, : attention.MAIN_PANEL_SIZE], expected_left)
    assert not np.array_equal(rendered[attention.MAIN_HEADER_HEIGHT :, attention.MAIN_PANEL_SIZE :], expected_left)


def test_npz_roundtrip_enforces_schema_mask_sums_and_schedule(tmp_path: Path) -> None:
    values, valid = _uniform_attention()
    slots, mean_map = attention._head_mean_per_slot(values, valid)
    frame_count = 17
    path = tmp_path / "maps.npz"
    identity = attention._write_npz_atomic(
        path,
        {
            "frame_index": np.arange(frame_count, dtype=np.int32),
            "task_index": np.zeros(frame_count, dtype=np.int16),
            "scheduled_commit": np.asarray(
                [attention._is_scheduled_commit(frame) for frame in range(frame_count)], dtype=bool
            ),
            "source_valid": valid,
            "attention_head_mean_slots": np.broadcast_to(slots, (frame_count, *slots.shape)).astype(np.float16),
            "attention_mean_heads_slots": np.broadcast_to(mean_map, (frame_count, *mean_map.shape)).astype(np.float16),
            "retrieval_rms": np.zeros(frame_count, dtype=np.float32),
            "write_token_rms": np.ones(frame_count, dtype=np.float32),
        },
    )
    assert identity["sha256"]
    checks = attention._validate_saved_npz(path, frame_count=frame_count, source_valid=valid)
    assert checks["schema_exact"]
    assert checks["slot_shape"] == [frame_count, attention.NUM_SLOTS, attention.NUM_PATCHES]


def test_private_stage_only_publishes_complete_directory_and_cleans_failure(tmp_path: Path) -> None:
    final = tmp_path / "raw"
    with attention._Stage(final) as stage:
        artifact = stage.stage_dir / "artifact.txt"
        artifact.write_text("evidence\n", encoding="utf-8")
        stage.publish({"artifact.txt": attention.fixed.causal._file_identity(artifact)})
    assert final.is_dir()
    assert (final / "COMPLETE").read_text().endswith("  artifact.txt\n")
    assert not (final / "INCOMPLETE.json").exists()
    assert not list(tmp_path.glob(".raw.staging.*"))

    failed = tmp_path / "ema"
    with pytest.raises(RuntimeError, match="injected"), attention._Stage(failed):
        raise RuntimeError("injected failure")
    assert not failed.exists()
    assert not list(tmp_path.glob(".ema.staging.*"))
