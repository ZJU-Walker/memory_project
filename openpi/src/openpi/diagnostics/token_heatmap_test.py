import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from openpi.diagnostics import token_heatmap


def _frame(values, *, raw_frame=0, policy_step=0, raw_image_rgb=None, geometry=None):
    return token_heatmap.TokenMetricFrame(
        raw_frame=raw_frame,
        policy_step=policy_step,
        model_image_rgb=np.full((224, 224, 3), 32, dtype=np.uint8),
        token_values=np.asarray(values, dtype=np.float32),
        raw_image_rgb=raw_image_rgb,
        letterbox_geometry=geometry,
        write_count=policy_step + 1,
        phase="visible",
    )


def test_token_grid_is_siglip_row_major_without_transpose_or_flip():
    values = np.arange(256, dtype=np.float32)
    grid = token_heatmap.token_grid(values)
    assert grid.shape == (16, 16)
    assert grid[0, 0] == 0
    assert grid[0, 15] == 15
    assert grid[1, 0] == 16
    assert grid[15, 15] == 255


def test_raw_top_camera_uses_exact_model_letterbox_and_reports_geometry():
    raw = np.zeros((100, 200, 3), dtype=np.uint8)
    raw[..., 0] = 123
    model_rgb, geometry = token_heatmap.raw_top_camera_to_model_rgb(raw)
    assert geometry == token_heatmap.LetterboxGeometry(
        source_height=100,
        source_width=200,
        resized_height=112,
        resized_width=224,
        pad_top=56,
        pad_bottom=56,
        pad_left=0,
        pad_right=0,
    )
    assert model_rgb.shape == (224, 224, 3)
    assert np.all(model_rgb[:56] == 0)
    assert np.all(model_rgb[56:168, :, 0] == 123)
    assert np.all(model_rgb[168:] == 0)


def test_raw_frame_geometry_is_inferred_and_mismatch_is_rejected():
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    frame = _frame(np.zeros(256), raw_image_rgb=raw)
    assert frame.letterbox_geometry == token_heatmap.LetterboxGeometry(
        source_height=480,
        source_width=640,
        resized_height=168,
        resized_width=224,
        pad_top=28,
        pad_bottom=28,
        pad_left=0,
        pad_right=0,
    )
    wrong = token_heatmap.letterbox_geometry(640, 480)
    with pytest.raises(ValueError, match="does not match raw_image"):
        _frame(np.zeros(256), raw_image_rgb=raw, geometry=wrong)
    with pytest.raises(ValueError, match="requires raw_image"):
        _frame(np.zeros(256), geometry=frame.letterbox_geometry)


def test_normalized_model_image_conversion_does_not_resize():
    image = np.zeros((1, 224, 224, 3), dtype=np.float32)
    image[..., 0] = -1
    image[..., 1] = 0
    image[..., 2] = 1
    rgb = token_heatmap.normalized_model_top_camera_to_rgb(image)
    assert tuple(rgb[0, 0]) == (0, 128, 255)
    with pytest.raises(ValueError, match="shape"):
        token_heatmap.normalized_model_top_camera_to_rgb(np.zeros((100, 200, 3), dtype=np.float32))


def test_token_values_reject_bad_shape_negative_and_nonfinite():
    with pytest.raises(ValueError, match="shape"):
        token_heatmap.token_grid(np.zeros(255))
    bad = np.zeros(256)
    bad[2] = -1
    with pytest.raises(ValueError, match="non-negative"):
        token_heatmap.token_grid(bad)
    bad[2] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        token_heatmap.token_grid(bad)


def test_video_wide_scale_is_zero_anchored_and_constant_safe():
    scale = token_heatmap.fit_color_scale([_frame(np.ones(256) * 2), _frame(np.arange(256))])
    assert scale.vmin == 0
    assert 2 < scale.vmax <= 255
    zero_scale = token_heatmap.fit_color_scale([_frame(np.zeros(256))])
    assert (zero_scale.vmin, zero_scale.vmax) == (0, 1)


def test_overlay_preserves_patch_orientation():
    values = np.zeros(256)
    values[0] = 1
    frame = _frame(values)
    scale = token_heatmap.ColorScale(vmin=0, vmax=1, lower_percentile=0, upper_percentile=100, anchor_zero=True)
    overlay, normalized = token_heatmap.heatmap_overlay(frame.model_image_rgb, values, scale, alpha=1, draw_grid=False)
    assert normalized[0, 0] == 1
    assert normalized[15, 15] == 0
    assert not np.array_equal(overlay[5, 5], overlay[-5, -5])
    assert np.all(overlay[:14, :14] == overlay[0, 0])
    assert np.all(overlay[-14:, -14:] == overlay[-1, -1])


def test_model_grid_projects_to_raw_camera_without_padding_or_orientation_error():
    geometry = token_heatmap.letterbox_geometry(480, 640)
    normalized = np.zeros((16, 16), dtype=np.float32)
    normalized[2, 0] = 1.0  # First visible model patch -> raw top-left.
    normalized[13, 15] = 0.5  # Last visible model patch -> raw bottom-right.
    projected = token_heatmap.project_normalized_grid_to_raw(normalized, geometry)
    assert projected.shape == (480, 640)
    assert projected[0, 0] == 1
    assert projected[39, 39] == 1
    assert projected[41, 20] == 0
    assert projected[-1, -1] == 0.5

    padding_only = np.zeros((16, 16), dtype=np.float32)
    padding_only[0, 0] = 1
    assert np.all(token_heatmap.project_normalized_grid_to_raw(padding_only, geometry) == 0)


def test_portrait_projection_crops_horizontal_padding_without_transpose():
    geometry = token_heatmap.letterbox_geometry(640, 480)
    assert (geometry.pad_left, geometry.pad_right) == (28, 28)
    normalized = np.zeros((16, 16), dtype=np.float32)
    normalized[0, 2] = 1
    normalized[15, 13] = 0.5
    projected = token_heatmap.project_normalized_grid_to_raw(normalized, geometry)
    assert projected.shape == (640, 480)
    assert projected[0, 0] == 1
    assert projected[-1, -1] == 0.5
    padding_only = np.zeros((16, 16), dtype=np.float32)
    padding_only[0, 0] = 1
    assert np.all(token_heatmap.project_normalized_grid_to_raw(padding_only, geometry) == 0)


def test_raw_projection_statistics_account_for_letterbox_area():
    geometry = token_heatmap.letterbox_geometry(480, 640)
    stats = token_heatmap.raw_projection_statistics(np.ones(256), geometry)
    assert stats["visible_content_mass_fraction"] == pytest.approx(0.75)
    assert stats["letterbox_padding_mass_fraction"] == pytest.approx(0.25)


def test_uniform_contribution_statistics_show_all_tokens_effective():
    stats = token_heatmap.metric_statistics(np.ones(256))
    assert stats["coefficient_of_variation"] == 0
    assert stats["normalized_entropy"] == pytest.approx(1)
    assert stats["effective_token_count"] == pytest.approx(256)
    assert stats["top_10pct_mass_fraction"] == pytest.approx(26 / 256)


def test_mp4_encoder_falls_back_to_opencv(monkeypatch, tmp_path):
    frame = np.zeros((10, 12, 3), dtype=np.uint8)

    def fail_imageio(frames, path, fps):
        raise RuntimeError("no ffmpeg")

    def fake_opencv(frames, path, fps):
        path.write_bytes(b"fake mp4")
        return "opencv:mp4v"

    monkeypatch.setattr(token_heatmap, "_encode_imageio", fail_imageio)
    monkeypatch.setattr(token_heatmap, "_encode_opencv", fake_opencv)
    path = tmp_path / "video.mp4"
    assert token_heatmap.encode_mp4([frame], path, 3) == "opencv:mp4v"
    assert path.read_bytes() == b"fake mp4"


def test_mp4_encoder_can_force_opencv_without_touching_imageio(monkeypatch, tmp_path):
    frame = np.zeros((10, 12, 3), dtype=np.uint8)

    def forbidden_imageio(frames, path, fps):
        raise AssertionError("forced OpenCV mode must not initialize imageio/ffmpeg")

    def fake_opencv(frames, path, fps):
        path.write_bytes(b"opencv only")
        return "opencv:mp4v"

    monkeypatch.setattr(token_heatmap, "_encode_imageio", forbidden_imageio)
    monkeypatch.setattr(token_heatmap, "_encode_opencv", fake_opencv)
    path = tmp_path / "video.mp4"
    assert token_heatmap.encode_mp4([frame], path, 3, encoder="opencv") == "opencv:mp4v"
    assert path.read_bytes() == b"opencv only"


def test_padding_masked_scale_ignores_letterbox_sinks():
    # A 640x480 frame pads rows 0-1 and 14-15. Attention parks huge "sink" mass there, which
    # would otherwise set vmax and crush all real scene structure into the bottom colour steps.
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    geometry = token_heatmap.letterbox_geometry(480, 640)
    values = np.full(256, 1.0)
    grid = values.reshape(16, 16)
    grid[0, :] = 1000.0  # padding sink
    grid[8, 8] = 5.0  # the real hotspot on camera content
    frame = _frame(grid.reshape(256), raw_image_rgb=raw, geometry=geometry)

    including = token_heatmap.fit_color_scale([frame], token_heatmap.NormalizationSpec(upper_percentile=100.0))
    excluding = token_heatmap.fit_color_scale(
        [frame], token_heatmap.NormalizationSpec(upper_percentile=100.0, exclude_letterbox_padding=True)
    )
    assert including.vmax == pytest.approx(1000.0)
    assert excluding.vmax == pytest.approx(5.0), "the scale must be set by camera content, not the sink"
    assert excluding.excluded_letterbox_padding is True

    # The content hotspot goes from invisible to full scale; the sink merely clips.
    assert token_heatmap.normalize_token_values(grid.reshape(256), including)[8, 8] < 0.01
    assert token_heatmap.normalize_token_values(grid.reshape(256), excluding)[8, 8] == pytest.approx(1.0)
    assert token_heatmap.normalize_token_values(grid.reshape(256), excluding)[0, 0] == pytest.approx(1.0)


def test_padding_masked_scale_requires_known_geometry():
    plain = _frame(np.ones(256))
    with pytest.raises(ValueError, match="letterbox_geometry"):
        token_heatmap.fit_color_scale([plain], token_heatmap.NormalizationSpec(exclude_letterbox_padding=True))


def test_padding_mass_statistics_still_report_the_excluded_region():
    geometry = token_heatmap.letterbox_geometry(480, 640)
    grid = np.zeros((16, 16))
    grid[0, :] = 3.0
    grid[8, 8] = 1.0
    stats = token_heatmap.raw_projection_statistics(grid.reshape(256), geometry)
    # Excluding padding from the colour scale must not hide how much mass lives there.
    assert stats["letterbox_padding_mass_fraction"] > 0.9


def _history_frames(values_per_frame):
    return [_frame(values, raw_frame=10 * i, policy_step=i) for i, values in enumerate(values_per_frame)]


def test_token_history_cancels_static_layout_and_isolates_an_event():
    # A border token is persistently hot and one interior token spikes in a single frame. The
    # per-frame view cannot separate them; the history view must flag only the spike.
    baseline = np.full(256, 1.0)
    baseline[0] = 50.0  # static hot border token, hot in EVERY frame
    frames_values = [baseline.copy() for _ in range(8)]
    frames_values[5][200] = 40.0  # one-off event
    frames = _history_frames(frames_values)
    history = token_heatmap.fit_token_history(frames)

    event_z = token_heatmap.history_token_z(frames_values[5], history)
    assert event_z[200] > 2.0, "the one-off event must stand out against its own baseline"
    assert abs(event_z[0]) < 0.5, "a persistently hot token is typical for itself and must not stand out"

    quiet_z = token_heatmap.history_token_z(frames_values[0], history)
    assert abs(quiet_z[0]) < 0.5
    assert np.max(np.abs(quiet_z)) < 2.0, "a frame with no event must have no strong hotspot"


def test_token_history_is_invariant_to_overall_write_magnitude():
    # Frame-level decay must not colour the map: rescaling a whole frame changes no token's
    # share of that write, so an identical spatial pattern must score identically at any
    # magnitude. This is the property that makes late, tiny writes readable.
    rng = np.random.default_rng(0)
    pattern = rng.uniform(1.0, 5.0, size=256)
    jitter = rng.uniform(0.9, 1.1, size=(6, 256))
    decayed = pattern[None] * jitter * np.power(10.0, -np.arange(6))[:, None]
    history = token_heatmap.fit_token_history(_history_frames(decayed))

    scaled_early = token_heatmap.history_token_z(decayed[0], history)
    # The same frame scaled down by 10^5 must produce the same scores.
    assert np.allclose(scaled_early, token_heatmap.history_token_z(decayed[0] * 1e-5, history), atol=1e-9)


def test_token_history_neutralizes_degenerate_and_validates_inputs():
    constant = [np.full(256, 3.0) for _ in range(4)]
    history = token_heatmap.fit_token_history(_history_frames(constant))
    grid = token_heatmap.history_token_values(constant[0], history)
    assert np.allclose(grid, 0.5), "zero-variance tokens must render neutral, not as hotspots"

    with pytest.raises(ValueError, match="at least one"):
        token_heatmap.fit_token_history([])
    with pytest.raises(ValueError, match="requires a fitted TokenHistory"):
        token_heatmap.heatmap_overlay(
            np.full((224, 224, 3), 32, dtype=np.uint8),
            np.ones(256),
            None,
            scale_mode="per_token_history",
        )


def test_export_per_token_history_documents_baseline_and_saves_raw_z(monkeypatch, tmp_path):
    monkeypatch.setattr(token_heatmap, "encode_mp4", lambda frames, path, fps, *, encoder="auto": "test:encoder")
    values = [np.full(256, 2.0) for _ in range(5)]
    values[3][100] = 20.0
    output = tmp_path / "history"
    manifest = token_heatmap.export_heatmap_video(
        _history_frames(values),
        output,
        metric_name="token_error",
        fps=3,
        colormap="coolwarm",
        scale_mode="per_token_history",
    )
    assert manifest["scale_mode"] == "per_token_history"
    assert manifest["normalization"]["token_history"]["frame_count"] == 5
    assert "SHARE" in manifest["normalization"]["scope"]
    assert manifest["normalization"]["token_history"]["detrend"].startswith("per-frame log-median")

    stored = np.load(output / manifest["records"][3]["data"])
    assert stored["history_z"].shape == (16, 16)
    assert stored["history_z"].reshape(256)[100] > 1.5


class _FakeVideoWriter:
    """OpenCV writer stub whose avc1 codec is unavailable, matching cluster nodes."""

    def __init__(self, path, fourcc, fps, size):
        self._path = Path(path)
        self._opened = fourcc == cv2.VideoWriter_fourcc(*"mp4v")

    def isOpened(self):  # noqa: N802 - OpenCV API name
        return self._opened

    def write(self, frame):
        with self._path.open("ab") as stream:
            stream.write(frame.tobytes())

    def release(self):
        pass


def test_opencv_mp4v_is_never_delivered_without_h264_reencode(monkeypatch, tmp_path):
    frame = np.zeros((10, 12, 3), dtype=np.uint8)
    reencoded = []
    monkeypatch.setattr(cv2, "VideoWriter", _FakeVideoWriter)
    monkeypatch.setattr(token_heatmap, "_reencode_h264_in_place", reencoded.append)
    path = tmp_path / "video.mp4"
    assert token_heatmap.encode_mp4([frame], path, 3, encoder="opencv") == "opencv:mp4v+ffmpeg:libx264"
    # encode_mp4 hands the encoder its temporary file; the destination appears on atomic rename.
    assert len(reencoded) == 1
    assert reencoded[0].parent == path.parent
    assert path.is_file()

    def failing_reencode(target):
        raise RuntimeError("no ffmpeg binary")

    monkeypatch.setattr(token_heatmap, "_reencode_h264_in_place", failing_reencode)
    with pytest.raises(RuntimeError, match="H.264"):
        token_heatmap.encode_mp4([frame], tmp_path / "video2.mp4", 3, encoder="opencv")
    assert not (tmp_path / "video2.mp4").exists()


def test_export_writes_png_npz_manifest_and_video(monkeypatch, tmp_path):
    def fake_encode(frames, path, fps, *, encoder="auto"):
        assert encoder == "auto"
        assert len(frames) == 2
        assert frames[0].shape[0] % 2 == frames[0].shape[1] % 2 == 0
        path.write_bytes(b"video")
        return "test:encoder"

    monkeypatch.setattr(token_heatmap, "encode_mp4", fake_encode)
    frames = [
        _frame(np.arange(256), raw_frame=10, policy_step=1),
        _frame(np.arange(256)[::-1], raw_frame=20, policy_step=2),
    ]
    output = tmp_path / "heatmaps"
    manifest = token_heatmap.export_heatmap_video(frames, output, metric_name="gradient norm", fps=3)
    assert manifest["frame_count"] == 2
    assert manifest["token_layout"] == token_heatmap.TOKEN_LAYOUT
    assert (output / "gradient_norm_heatmap.mp4").read_bytes() == b"video"
    assert len(list((output / "frames").glob("*.png"))) == 2
    data_files = list((output / "data").glob("*.npz"))
    assert len(data_files) == 2
    with np.load(data_files[0], allow_pickle=False) as data:
        assert data["token_grid"].shape == (16, 16)
        assert data["model_image_rgb"].shape == (224, 224, 3)
    disk_manifest = json.loads((output / "manifest.json").read_text())
    assert disk_manifest["encoder"] == "test:encoder"
    assert disk_manifest["normalization"]["scope"] == "one fixed scale across the entire video"


def test_export_raw_overlay_preserves_model_grid_and_raw_projection(monkeypatch, tmp_path):
    def fake_encode(frames, path, fps, *, encoder="auto"):
        assert encoder == "auto"
        # 640x480 raw input is displayed at 448x336, preserving 4:3 rather than stretching square.
        assert frames[0].shape[0] < frames[0].shape[1]
        path.write_bytes(b"video")
        return "test:encoder"

    monkeypatch.setattr(token_heatmap, "encode_mp4", fake_encode)
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    values = np.arange(256, dtype=np.float32)
    output = tmp_path / "raw_heatmaps"
    manifest = token_heatmap.export_heatmap_video(
        [_frame(values, raw_image_rgb=raw)], output, metric_name="gradient norm", fps=3
    )
    record = manifest["records"][0]
    assert record["rendered_on_raw_image"] is True
    assert record["letterbox_geometry"]["pad_top"] == 28
    data_path = output / record["data"]
    with np.load(data_path, allow_pickle=False) as data:
        assert data["raw_image_rgb"].shape == (480, 640, 3)
        assert data["token_grid"].shape == (16, 16)
        assert data["projected_normalized_raw_map"].shape == (480, 640)
        np.testing.assert_array_equal(data["letterbox_padding"], [28, 28, 0, 0])


def test_export_refuses_to_overwrite_existing_directory(tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        token_heatmap.export_heatmap_video([_frame(np.zeros(256))], output, metric_name="error", fps=3)


def test_export_rejects_mixed_raw_and_model_fallback_before_writing(tmp_path):
    frames = [
        _frame(np.zeros(256)),
        _frame(np.zeros(256), raw_image_rgb=np.zeros((224, 224, 3), dtype=np.uint8)),
    ]
    output = tmp_path / "mixed"
    with pytest.raises(ValueError, match="consistently"):
        token_heatmap.export_heatmap_video(frames, output, metric_name="error", fps=3)
    assert not output.exists()


def test_export_sanitizes_metric_filename_and_validates_fps_before_writing(monkeypatch, tmp_path):
    def fake_encode(frames, path, fps, *, encoder="auto"):
        path.write_bytes(b"video")
        return "test:encoder"

    monkeypatch.setattr(token_heatmap, "encode_mp4", fake_encode)
    output = tmp_path / "safe"
    token_heatmap.export_heatmap_video([_frame(np.zeros(256))], output, metric_name="../gradient norm", fps=3)
    assert (output / "gradient_norm_heatmap.mp4").is_file()

    invalid_output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="fps"):
        token_heatmap.export_heatmap_video([_frame(np.zeros(256))], invalid_output, metric_name="error", fps=0)
    assert not invalid_output.exists()


def test_zscore_maps_constant_frame_to_neutral_midpoint():
    grid = token_heatmap.zscore_token_values(np.full(256, 100.0))
    assert grid.shape == (16, 16)
    assert np.allclose(grid, 0.5)
    with pytest.raises(ValueError, match="z_range"):
        token_heatmap.zscore_token_values(np.arange(256), 0.0)


def test_zscore_highlights_hot_token_regardless_of_absolute_magnitude():
    values = np.full(256, 100.0)
    values[37] = 200.0
    flat = token_heatmap.zscore_token_values(values, 3.0).reshape(256)
    rest = np.delete(flat, 37)
    assert flat[37] > 0.99
    assert np.all(rest < 0.5)
    assert np.all(rest > 0.4)


def test_export_per_frame_zscore_is_magnitude_invariant_and_documents_mode(monkeypatch, tmp_path):
    def fake_encode(frames, path, fps, *, encoder="auto"):
        path.write_bytes(b"video")
        return "test:encoder"

    monkeypatch.setattr(token_heatmap, "encode_mp4", fake_encode)
    pattern = np.ones(256)
    pattern[5] = 2.0
    # Same relative pattern six orders of magnitude apart: the blank-memory transient scenario.
    frames = [
        _frame(pattern * 1000.0, raw_frame=0, policy_step=0),
        _frame(pattern * 0.001, raw_frame=10, policy_step=1),
    ]
    output = tmp_path / "zscore"
    manifest = token_heatmap.export_heatmap_video(
        frames,
        output,
        metric_name="token_error",
        fps=3,
        colormap="coolwarm",
        scale_mode="per_frame_zscore",
        zscore_range=3.0,
    )
    assert manifest["scale_mode"] == "per_frame_zscore"
    assert manifest["normalization"]["z_range"] == 3.0
    assert "vmax" not in manifest["normalization"]
    grids = []
    for data_file in sorted((output / "data").glob("*.npz")):
        with np.load(data_file, allow_pickle=False) as data:
            grids.append(np.asarray(data["normalized_grid"]))
    assert np.allclose(grids[0], grids[1], atol=1e-6)
    assert grids[0].reshape(256)[5] > 0.9


def test_video_scale_mode_requires_fitted_scale_and_rejects_unknown_mode():
    values = np.arange(256, dtype=np.float64)
    with pytest.raises(ValueError, match="ColorScale"):
        token_heatmap.heatmap_overlay(np.full((224, 224, 3), 32, dtype=np.uint8), values, None)
    scale = token_heatmap.fit_color_scale([_frame(values)])
    with pytest.raises(ValueError, match="scale_mode"):
        token_heatmap.heatmap_overlay(np.full((224, 224, 3), 32, dtype=np.uint8), values, scale, scale_mode="per_video")
