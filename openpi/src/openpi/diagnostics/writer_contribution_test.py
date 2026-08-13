import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from openpi.diagnostics import token_heatmap
from openpi.diagnostics import writer_contribution as contribution


def _options(tmp_path: Path, **changes) -> contribution.RunOptions:
    base = contribution.RunOptions(
        checkpoint=tmp_path / "checkpoint",
        output_dir=tmp_path / "output",
        episode_paths=(tmp_path / "demo1",),
        render_video=False,
    )
    return dataclasses.replace(base, **changes)


def test_run_options_require_exactly_one_source_and_validate_metrics(tmp_path: Path):
    with pytest.raises(ValueError, match="exactly one source"):
        contribution.RunOptions(checkpoint=tmp_path / "c", output_dir=tmp_path / "o")
    with pytest.raises(ValueError, match="exactly one source"):
        contribution.RunOptions(
            checkpoint=tmp_path / "c",
            output_dir=tmp_path / "o",
            episode_paths=(tmp_path / "demo",),
            dataset_root=tmp_path / "dataset",
            episode_indices=(0,),
        )
    with pytest.raises(ValueError, match="episode-indices"):
        contribution.RunOptions(
            checkpoint=tmp_path / "c",
            output_dir=tmp_path / "o",
            dataset_root=tmp_path / "dataset",
        )
    with pytest.raises(ValueError, match="unsupported metrics"):
        _options(tmp_path, metrics=("not_a_metric",))


def test_lerobot_metadata_resolves_inline_parquet_episodes_and_sides(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "fps": 30,
        "total_episodes": 3,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "image": {"dtype": "image"},
            "left_wrist_image": {"dtype": "image"},
            "right_wrist_image": {"dtype": "image"},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (root / "meta" / "tasks.jsonl").write_text(
        "\n".join(
            [
                '{"task_index":0,"task":"observe bins"}',
                '{"task_index":1,"task":"open left bin"}',
                '{"task_index":2,"task":"open right bin"}',
            ]
        ),
        encoding="utf-8",
    )
    (root / "meta" / "episodes.jsonl").write_text(
        "\n".join(
            [
                '{"episode_index":0,"tasks":["observe bins","open left bin"],"length":2}',
                '{"episode_index":1,"tasks":["observe bins","open left bin"],"length":2}',
                '{"episode_index":2,"tasks":["observe bins","open right bin"],"length":2}',
            ]
        ),
        encoding="utf-8",
    )
    for index in (0, 2):
        (root / "data" / "chunk-000" / f"episode_{index:06d}.parquet").touch()

    options = contribution.RunOptions(
        checkpoint=tmp_path / "checkpoint",
        output_dir=tmp_path / "output",
        dataset_root=root,
        episode_indices=(0, 2),
        render_video=False,
    )
    sources = contribution.load_episode_sources(options)
    assert [source.episode_id for source in sources] == ["episode_000000", "episode_000002"]
    assert [source.ground_truth_side for source in sources] == ["left", "right"]
    assert all(source.source_format == "lerobot_inline_parquet" for source in sources)
    assert sources[0].task_names == ("observe bins", "open left bin", "open right bin")
    assert sources[0].control_hz == 30


def test_episode_artifacts_preserve_all_raw_token_arrays_and_spatial_summaries(tmp_path: Path):
    runner = object.__new__(contribution.WriterContributionRunner)
    runner.options = _options(tmp_path)
    runner.stride = 10
    source = contribution.EpisodeSource("demo1", tmp_path / "demo1", 30.0, "left")
    frames = []
    for step in range(2):
        error = np.linspace(0.0, 1.0 + step, token_heatmap.TOKEN_COUNT, dtype=np.float32)
        grad = np.linspace(1.0, 2.0 + step, token_heatmap.TOKEN_COUNT, dtype=np.float32)
        frames.append(
            contribution._MeasuredFrame(  # noqa: SLF001
                raw_frame=step * 10,
                policy_step=step,
                model_image_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
                raw_image_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
                metrics={
                    "token_error": error,
                    "token_grad_norm": grad,
                    "token_mean_loss_grad_norm": grad / token_heatmap.TOKEN_COUNT,
                },
                scalar={
                    "surprise": float(error.mean()),
                    "grad_norm": 2.0,
                    "theta": 0.1,
                    "eta": 0.9,
                    "alpha": 0.01,
                    "retrieval_norm": 0.0,
                    "write_source_norm": 1.0,
                    "memory_gate_norm": 1.0,
                },
                phase="observe bins",
            )
        )

    summary = runner._save_episode(source, frames, 20, tmp_path / "artifacts")  # noqa: SLF001
    arrays = np.load(tmp_path / "artifacts" / "contributions.npz", allow_pickle=False)
    assert arrays["token_error"].shape == (2, 256)
    assert arrays["token_grad_norm"].shape == (2, 256)
    assert arrays["task"].tolist() == ["observe bins", "observe bins"]
    assert summary["sampled_write_frames"] == 2
    assert 0 < summary["aggregate"]["token_grad_norm"]["per_frame_effective_token_count_mean"] <= 256


def test_writer_renderer_forces_in_process_opencv_encoder(monkeypatch, tmp_path: Path):
    runner = object.__new__(contribution.WriterContributionRunner)
    runner.options = _options(tmp_path, render_video=True, metrics=("token_error",))
    runner.stride = 10
    source = contribution.EpisodeSource("demo1", tmp_path / "demo1", 30.0)
    error = np.ones(token_heatmap.TOKEN_COUNT, dtype=np.float32)
    frame = contribution._MeasuredFrame(  # noqa: SLF001
        raw_frame=0,
        policy_step=0,
        model_image_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
        raw_image_rgb=np.zeros((480, 640, 3), dtype=np.uint8),
        metrics={
            "token_error": error,
            "token_grad_norm": error,
            "token_mean_loss_grad_norm": error / token_heatmap.TOKEN_COUNT,
        },
        scalar={
            "surprise": 1.0,
            "grad_norm": 1.0,
            "theta": 0.1,
            "eta": 0.9,
            "alpha": 0.01,
            "retrieval_norm": 0.0,
            "write_source_norm": 1.0,
            "memory_gate_norm": 1.0,
        },
    )

    def fake_export(frames, output_dir, **kwargs):
        assert len(frames) == 1
        assert kwargs["video_encoder"] == "opencv"
        return {"video": "token_error_heatmap.mp4", "encoder": "opencv:mp4v"}

    monkeypatch.setattr(token_heatmap, "export_heatmap_video", fake_export)
    summary = runner._save_episode(source, [frame], 1, tmp_path / "artifacts")  # noqa: SLF001
    assert summary["videos"]["token_error"].endswith("token_error_heatmap.mp4")


def test_inline_image_decoder_fails_closed_and_round_trips_png():
    import cv2

    rgb = np.zeros((12, 16, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    decoded = contribution.WriterContributionRunner._decode_inline_image(  # noqa: SLF001
        {"bytes": encoded.tobytes(), "path": "frame.png"}, field="image", raw_frame=0
    )
    np.testing.assert_array_equal(decoded, rgb)
    with pytest.raises(ValueError, match="inline image struct"):
        contribution.WriterContributionRunner._decode_inline_image(  # noqa: SLF001
            {"path": "frame.png"}, field="image", raw_frame=0
        )
