import dataclasses
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from openpi.diagnostics import final_ct_attribution as attribution
from openpi.diagnostics import token_heatmap
from openpi.diagnostics import writer_contribution as _writer
from openpi.models import memory as _memory
from openpi.models import model as _model


def _options(tmp_path: Path, **changes) -> attribution.FinalCtRunOptions:
    base = attribution.FinalCtRunOptions(
        checkpoint=tmp_path / "checkpoint",
        output_dir=tmp_path / "output",
        episode_paths=(tmp_path / "demo",),
        render_video=False,
    )
    return dataclasses.replace(base, **changes)


def _state(value: float, batch: int = 1) -> _memory.MemoryState:
    fast = {"w0": jnp.full((batch, 2), value, dtype=jnp.float32)}
    momentum = {"w0": jnp.full((batch, 2), value * 2, dtype=jnp.float32)}
    return _memory.MemoryState(fast_weights=fast, momentum=momentum)


def _observation() -> _model.Observation:
    return _model.Observation(
        images={"base_0_rgb": jnp.zeros((1, 224, 224, 3), dtype=jnp.float32)},
        image_masks={"base_0_rgb": jnp.ones((1,), dtype=bool)},
        state=jnp.zeros((1, 14), dtype=jnp.float32),
    )


def _frame(step: int, *, occlusion: bool = False) -> attribution._FinalCtFrame:
    sensitivity = np.linspace(0.0, 1.0 + step, token_heatmap.TOKEN_COUNT, dtype=np.float32)
    zero_read = np.linspace(1.0, 2.0 + step, token_heatmap.TOKEN_COUNT, dtype=np.float32)
    scalar = {"writer_loss": 0.5 + step, "surprise": 0.5 + step, "final_ct_rms": 2.0}
    occlusion_maps = None
    if occlusion:
        occlusion_maps = {
            "occlusion_writer_loss_abs_delta": np.full(256, 0.1, dtype=np.float32),
            "occlusion_final_ct_rms_delta": np.full(256, 0.2, dtype=np.float32),
            "occlusion_fast_update_relative_l2": np.full(256, 0.3, dtype=np.float32),
            "occlusion_full_state_update_relative_l2": np.full(256, 0.4, dtype=np.float32),
            "occlusion_full_update_cosine": np.full(256, 0.9, dtype=np.float32),
        }
        scalar.update({"baseline_fast_update_l2": 3.0, "baseline_full_update_l2": 4.0})
    return attribution._FinalCtFrame(  # noqa: SLF001
        raw_frame=step * 10,
        policy_step=step,
        model_image_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        raw_image_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        maps={
            attribution.SIGLIP_OUTPUT_SLOT_METRIC: sensitivity,
            attribution.OUTPUT_SLOT_METRIC: zero_read,
        },
        scalar=scalar,
        phase="observe bins",
        occlusion=occlusion_maps,
    )


def test_options_validate_selected_frames_and_enable_occlusion_artifacts_by_default(tmp_path: Path):
    options = _options(tmp_path, gradient_raw_frames=(0, 20), occlusion_raw_frames=(20,))
    assert options.occlusion_batch_size == 1
    assert options.gradient_raw_frames == (0, 20)
    assert options.occlusion_raw_frames == (20,)
    assert all(name in options.metrics for name in attribution.OCCLUSION_METRICS[:-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        _options(tmp_path, gradient_raw_frames=(20, 0))
    with pytest.raises(ValueError, match="strictly increasing"):
        _options(tmp_path, occlusion_raw_frames=(10, 10))


def test_fake_core_attribution_is_validated_and_commits_one_baseline_write(tmp_path: Path):
    runner = object.__new__(attribution.FinalCtAttributionRunner)
    runner.options = _options(tmp_path)
    runner._requested = set()  # noqa: SLF001
    runner._occlusion_requested = set()  # noqa: SLF001
    observation = _observation()
    pre_state = _state(0.0)
    next_state = _state(1.0)
    model_image = np.zeros((224, 224, 3), dtype=np.uint8)
    runner._transform_observation = lambda *args: (observation, model_image)  # noqa: SLF001

    attribution_aux = {
        "final_ct": jnp.ones((1, 256, 4), dtype=jnp.float32),
        "writer_loss": jnp.asarray([0.30], dtype=jnp.float32),
        "surprise": jnp.asarray([0.30], dtype=jnp.float32),
        "writer_loss_top_patch_grad_norm": jnp.ones((1, 256), dtype=jnp.float32),
        "final_ct_zero_read_l2": jnp.full((1, 256), 2.0, dtype=jnp.float32),
        "write_occurred": jnp.asarray([False]),
        "final_ct_rms": jnp.asarray([1.0], dtype=jnp.float32),
        "top_camera_tokens": jnp.asarray(256, dtype=jnp.int32),
    }
    baseline_aux = {
        "final_ct": jnp.full((1, 256, 4), 1.125, dtype=jnp.float32),
        "writer_loss": jnp.asarray([0.25], dtype=jnp.float32),
        "surprise": jnp.asarray([0.25], dtype=jnp.float32),
        "write_occurred": jnp.asarray([True]),
        "final_ct_rms": jnp.asarray([1.125], dtype=jnp.float32),
        "top_camera_tokens": jnp.asarray(256, dtype=jnp.int32),
    }

    def fake_attribution(_observation, _pre_state, *, allow_write):
        assert allow_write is False
        return _pre_state, attribution_aux

    def fake_intervention(_observation, _pre_state, *, allow_write):
        assert allow_write is True
        return next_state, baseline_aux

    runner._attribution_step = fake_attribution  # noqa: SLF001
    runner._intervention_step = fake_intervention  # noqa: SLF001
    actual_state, frame = runner._evaluate_attribution_frame(  # noqa: SLF001
        _writer.EpisodeSource("demo", tmp_path / "demo", 30.0),
        raw_frame=0,
        policy_step=0,
        top_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        left_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        right_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        robot_state=np.zeros(14, dtype=np.float32),
        memory_state=pre_state,
    )
    assert frame is not None
    np.testing.assert_array_equal(actual_state.fast_weights["w0"], next_state.fast_weights["w0"])
    assert frame.maps[attribution.SIGLIP_OUTPUT_SLOT_METRIC].shape == (256,)
    assert frame.maps[attribution.OUTPUT_SLOT_METRIC].shape == (256,)
    assert frame.scalar["writer_loss"] == pytest.approx(0.25)
    assert frame.scalar["primal_writer_loss_abs_floor"] == pytest.approx(0.05)
    assert frame.scalar["primal_final_ct_rms_floor"] == pytest.approx(0.125)


def test_occlusion_branches_share_prestate_and_are_not_committed(tmp_path: Path):
    runner = object.__new__(attribution.FinalCtAttributionRunner)
    runner.options = _options(tmp_path, occlusion_raw_frames=(0,), occlusion_batch_size=31)
    observation = _observation()
    image = observation.images["base_0_rgb"].at[:, :14, :14, :].set(-1.0)
    observation = observation.replace(images={"base_0_rgb": image})
    pre_state = _state(0.0)
    baseline_state = _memory.MemoryState(
        fast_weights={"w0": jnp.full((1, 2), 2.0, dtype=jnp.float32)},
        momentum={"w0": jnp.full((1, 2), 3.0, dtype=jnp.float32)},
    )
    before = np.array(pre_state.fast_weights["w0"], copy=True)
    batch_sizes = []

    def fake_intervention(branch_observation, branch_pre, *, allow_write):
        assert allow_write is True
        batch = branch_observation.state.shape[0]
        batch_sizes.append(batch)
        np.testing.assert_array_equal(branch_pre.fast_weights["w0"], np.zeros((batch, 2), dtype=np.float32))
        images = np.asarray(branch_observation.images["base_0_rgb"])
        black_pixels = np.count_nonzero(images == -1.0, axis=(1, 2, 3))
        signal = jnp.asarray(black_pixels / (14 * 14 * 3), dtype=jnp.float32)
        branch_state = _memory.MemoryState(
            fast_weights={"w0": jnp.ones((batch, 2), dtype=jnp.float32) + signal[:, None]},
            momentum={"w0": jnp.full((batch, 2), 2.0, dtype=jnp.float32) + signal[:, None]},
        )
        return branch_state, {
            "writer_loss": signal,
            "final_ct": jnp.broadcast_to(signal[:, None, None], (batch, 256, 4)),
        }

    runner._intervention_step = fake_intervention  # noqa: SLF001
    maps, scalar = runner._occlusion_sweep(  # noqa: SLF001
        observation,
        pre_state,
        baseline_state,
        {"writer_loss": jnp.asarray([1.0]), "final_ct": jnp.ones((1, 256, 4))},
    )
    np.testing.assert_array_equal(pre_state.fast_weights["w0"], before)
    assert sum(batch_sizes) == 512
    assert max(batch_sizes) == 31
    # Patch zero is already exactly black: the same-executable baseline makes
    # this counterfactual an exact numerical no-op, with no compiler floor.
    for name in attribution.OCCLUSION_METRICS[:-1]:
        assert maps[name][0] == 0.0
    assert maps["occlusion_full_update_cosine"][0] == pytest.approx(1.0)
    np.testing.assert_allclose(maps["occlusion_writer_loss_abs_delta"][1:], 1.0)
    np.testing.assert_allclose(maps["occlusion_final_ct_rms_delta"][1:], 1.0)
    assert np.all(maps["occlusion_fast_update_relative_l2"][1:] > 0)
    assert np.all(np.isfinite(maps["occlusion_full_state_update_relative_l2"]))
    assert np.all((maps["occlusion_full_update_cosine"] >= -1.0) & (maps["occlusion_full_update_cosine"] <= 1.0))
    assert scalar["baseline_fast_update_l2"] > 0
    assert scalar["baseline_full_update_l2"] > scalar["baseline_fast_update_l2"]
    assert scalar["occlusion_batch_baseline_final_ct_rms_floor"] == 0.0
    assert scalar["occlusion_batch_baseline_writer_loss_abs_floor"] == 0.0


def test_artifact_schema_distinguishes_siglip_slots_ct_slots_and_causal_occlusions(tmp_path: Path):
    runner = object.__new__(attribution.FinalCtAttributionRunner)
    runner.options = _options(tmp_path, occlusion_raw_frames=(0,))
    runner.stride = 10
    source = _writer.EpisodeSource("demo", tmp_path / "demo", 30.0, "left")
    frames = [_frame(0, occlusion=True), _frame(1)]

    summary = runner._save_episode(  # noqa: SLF001
        source,
        frames,
        20,
        tmp_path / "artifacts",
        replayed_write_frames=7,
    )
    arrays = np.load(tmp_path / "artifacts" / "contributions.npz", allow_pickle=False)
    assert arrays[attribution.SIGLIP_OUTPUT_SLOT_METRIC].shape == (2, 256)
    assert arrays[attribution.OUTPUT_SLOT_METRIC].shape == (2, 256)
    assert arrays["occlusion_writer_loss_abs_delta"].shape == (1, 256)
    assert arrays["occlusion_full_update_cosine"].shape == (1, 256)
    assert arrays["baseline_fast_update_l2"].shape == (1,)
    assert not any(
        np.issubdtype(arrays[name].dtype, np.floating) and np.any(~np.isfinite(arrays[name])) for name in arrays.files
    )
    assert summary["attribution_frames"] == 2
    assert summary["occlusion_frames"] == 1
    assert summary["sampled_write_frames"] == 7
    assert summary["replayed_write_frames"] == 7
    document = json.loads((tmp_path / "artifacts" / "summary.json").read_text(encoding="utf-8"))
    assert "globally contextualized" in document["aggregate"][attribution.SIGLIP_OUTPUT_SLOT_METRIC]["metric_space"]
    assert "not pixel attribution" in document["aggregate"][attribution.OUTPUT_SLOT_METRIC]["metric_space"]


def test_selected_frame_order_must_be_strict_and_unique():
    valid = [_frame(0), _frame(1)]
    attribution.FinalCtAttributionRunner._validate_frame_order(valid)  # noqa: SLF001
    duplicate = [_frame(0), dataclasses.replace(_frame(1), raw_frame=0)]
    with pytest.raises(ValueError, match="raw_frame"):
        attribution.FinalCtAttributionRunner._validate_frame_order(duplicate)  # noqa: SLF001


def test_lerobot_row_identity_requires_contiguous_frames_and_matching_episode():
    previous = attribution._validate_lerobot_row_identity(  # noqa: SLF001
        raw_frame=0,
        previous_raw_frame=-1,
        episode_index=7,
        expected_episode_index=7,
    )
    assert previous == 0
    assert (
        attribution._validate_lerobot_row_identity(  # noqa: SLF001
            raw_frame=1,
            previous_raw_frame=previous,
            episode_index=7,
            expected_episode_index=7,
        )
        == 1
    )
    with pytest.raises(ValueError, match="contiguous"):
        attribution._validate_lerobot_row_identity(  # noqa: SLF001
            raw_frame=2,
            previous_raw_frame=0,
            episode_index=7,
            expected_episode_index=7,
        )
    with pytest.raises(ValueError, match="does not match"):
        attribution._validate_lerobot_row_identity(  # noqa: SLF001
            raw_frame=1,
            previous_raw_frame=0,
            episode_index=8,
            expected_episode_index=7,
        )
