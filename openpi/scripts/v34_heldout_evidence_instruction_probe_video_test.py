"""Focused pure-contract tests for evidence-only dual-head instruction videos."""

from __future__ import annotations

# The diagnostic intentionally exposes script-private helpers for contract tests.
# ruff: noqa: SLF001
from pathlib import Path
import sys

import numpy as np
import pytest

_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS))
try:
    import v34_heldout_evidence_instruction_probe_video as video
finally:
    sys.path.remove(str(_SCRIPTS))


def _args(**overrides) -> video.Args:
    values = {
        "checkpoint": Path("checkpoint/14750"),
        "dataset_root": Path("dataset"),
        "probe_artifact_dir": Path("probe/14750/raw"),
        "output_dir": Path("output/14750"),
        "config": video.RUN5_CONFIG,
        "parameter_source": "raw",
    }
    values.update(overrides)
    return video.Args(**values)


def _spec(prompt: str, side: str) -> video.fixed.EpisodeSpec:
    return video.fixed.EpisodeSpec(
        episode=15,
        prompt=prompt,
        side=side,
        label=video.fixed.SIDE_TO_LABEL[side],
        length=100,
        parquet=Path("episode_000015.parquet"),
        evidence_frames=(20, 21, 22),
        approach_frames=(17, 18, 19),
        heldout=True,
    )


def test_args_are_explicit_numeric_and_output_is_namespaced() -> None:
    args = _args()
    assert args.checkpoint_step == 14750
    assert args.artifact_dir.name == "raw"
    assert args.episodes == video.HELDOUT_EPISODES
    assert _args(smoke_only=True).episodes == (video.HELDOUT_EPISODES[0],)
    for overrides, match in (
        ({"checkpoint": Path("checkpoint/latest")}, "numeric"),
        ({"config": "pi05_yam_mem_v34"}, "pinned"),
        ({"parameter_source": "both"}, "raw or ema"),
        ({"batch_size": 0}, "batch-size"),
        ({"batch_size": 65}, "batch-size"),
    ):
        with pytest.raises(ValueError, match=match):
            _args(**overrides)


def test_native_and_switched_change_only_prompt_and_target_side() -> None:
    for prompt, side in (
        (video.fixed.PROMPTS[0], "left"),
        (video.fixed.PROMPTS[0], "right"),
        (video.fixed.PROMPTS[1], "left"),
        (video.fixed.PROMPTS[1], "right"),
    ):
        native, switched = video._conditions(_spec(prompt, side))
        assert native.name == "native"
        assert native.prompt == prompt
        assert native.target_side == side
        assert switched.name == "switched"
        assert switched.prompt != prompt
        assert {switched.prompt, prompt} == set(video.fixed.PROMPTS)
        assert switched.target_side == video._opposite_side(side)


def test_prefix_uses_no_future_evidence_and_updates_1_through_n() -> None:
    model = video.fixed.ProbeModel(
        mean=np.zeros(2, dtype=np.float64),
        scale=np.ones(2, dtype=np.float64),
        weights=np.asarray([1.0, -0.5, 0.25], dtype=np.float64),
    )
    features = np.asarray([[1.0, 0.0], [3.0, 2.0], [-1.0, 2.0]], dtype=np.float32)
    scores, counts = video.base._evidence_prefix_scores(
        features,
        np.ones(3, dtype=bool),
        model,
    )
    assert counts.tolist() == [1, 2, 3]
    assert scores.tolist() == pytest.approx([1.25, 1.75, 7.0 / 12.0])
    changed_future = features.copy()
    changed_future[2] = [1000.0, -1000.0]
    changed_scores, _ = video.base._evidence_prefix_scores(
        changed_future,
        np.ones(3, dtype=bool),
        model,
    )
    np.testing.assert_array_equal(changed_scores[:2], scores[:2])


def test_checkpoint_online_affine_and_prefix_are_causal_and_reproducible() -> None:
    head = video.OnlineWriterHead(
        kernel=np.asarray([[1.0, 3.0], [-2.0, 1.0]], dtype=np.float32),
        bias=np.asarray([0.25, -0.5], dtype=np.float32),
    )
    features = np.asarray([[1.0, 0.0], [3.0, 2.0], [-1.0, 2.0]], dtype=np.float32)
    prefixes = video._evidence_prefix_features(features)
    np.testing.assert_array_equal(
        prefixes,
        np.asarray([[1.0, 0.0], [2.0, 1.0], [1.0, 4.0 / 3.0]], dtype=np.float32),
    )
    expected_logits = features @ head.kernel + head.bias
    np.testing.assert_array_equal(head.logits(features), expected_logits)
    np.testing.assert_array_equal(
        head.scores(features),
        (expected_logits[:, 1] - expected_logits[:, 0]).astype(np.float64),
    )
    assert head.scores(features)[0] == pytest.approx(float(expected_logits[0, 1] - expected_logits[0, 0]))
    assert video.base._prediction(float(head.scores(features)[0])) == "right"

    changed_future = features.copy()
    changed_future[2] = [1000.0, -1000.0]
    changed_prefixes = video._evidence_prefix_features(changed_future)
    np.testing.assert_array_equal(changed_prefixes[:2], prefixes[:2])
    np.testing.assert_array_equal(
        head.scores(changed_prefixes[:2]),
        head.scores(prefixes[:2]),
    )
    assert video.ONLINE_RUNTIME_PARITY_ATOL == 2e-3
    assert video.ONLINE_FINAL_LOGIT_ATOL == 2e-3


def test_checkpoint_online_head_extraction_unwraps_exact_kernel_and_bias() -> None:
    class Param:
        def __init__(self, value: np.ndarray):
            self.value = value

    class Module:
        kernel = Param(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        bias = Param(np.asarray([0.5, -0.25], dtype=np.float32))

    class Runtime:
        _online_writer_head = Module()

    extracted = video._extract_online_writer_head(Runtime())
    np.testing.assert_array_equal(extracted.kernel, Module.kernel.value)
    np.testing.assert_array_equal(extracted.bias, Module.bias.value)
    assert extracted.feature_width == 2
    with pytest.raises(ValueError, match=r"\[width, 2\]"):
        video.OnlineWriterHead(
            kernel=np.zeros((2, 3), dtype=np.float32),
            bias=np.zeros(2, dtype=np.float32),
        )


def test_online_report_native_logits_have_exact_heldout_coverage_and_sign() -> None:
    scores = {15: -1.5, 29: 0.2, 44: -0.1, 59: 0.1}
    rows = [
        {
            "episode": episode,
            "truth_side": video.EXPECTED_CELLS[episode][1],
            "evidence_logit_right_minus_left": score,
            "evidence_prediction": video.base._prediction(score),
        }
        for episode, score in scores.items()
    ]
    report = {
        "stored_online_writer_head": {
            "parameter_source": "raw",
            "heldout_4": {"episodes": rows},
        }
    }
    assert video._online_report_logits(report, parameter_source="raw") == scores
    rows[0]["evidence_prediction"] = "right"
    with pytest.raises(ValueError, match="prediction"):
        video._online_report_logits(report, parameter_source="raw")


def test_paired_aggregate_counts_cover_both_heads() -> None:
    pairs = [
        {
            "heads": {
                "fresh_fixed_56_demo": {
                    "native_final_correct": index < 2,
                    "switched_final_correct": index < 2,
                    "shift_toward_switched_target": True,
                    "instruction_switch_changed_prediction": False,
                },
                "checkpoint_online_stored": {
                    "native_final_correct": True,
                    "switched_final_correct": index == 0,
                    "shift_toward_switched_target": index < 3,
                    "instruction_switch_changed_prediction": index == 0,
                },
            }
        }
        for index in range(4)
    ]
    aggregates = video._paired_effect_aggregates(pairs)
    fresh = aggregates["fresh_fixed_56_demo"]
    online = aggregates["checkpoint_online_stored"]
    assert (
        fresh["native_correct_count"],
        fresh["switched_correct_count"],
        fresh["shift_toward_switched_target_count"],
        fresh["prediction_flip_count"],
    ) == (2, 2, 4, 0)
    assert (
        online["native_correct_count"],
        online["switched_correct_count"],
        online["shift_toward_switched_target_count"],
        online["prediction_flip_count"],
    ) == (4, 1, 3, 1)


def test_render_has_exact_geometry_and_preserves_uncovered_pixels() -> None:
    image = np.arange(video.MODEL_IMAGE_SIZE * video.MODEL_IMAGE_SIZE * 3, dtype=np.uint32)
    image = (image.reshape(video.MODEL_IMAGE_SIZE, video.MODEL_IMAGE_SIZE, 3) % 256).astype(np.uint8)
    condition = video.Condition("switched", video.fixed.PROMPTS[1], "right")
    rendered = video._render_frame(
        image,
        episode=15,
        frame=269,
        evidence_index=1,
        evidence_total=58,
        condition=condition,
        current_score=1.25,
        prefix_score=-0.75,
        online_current_score=-0.1,
        online_prefix_score=0.2,
        checkpoint_step=14750,
        parameter_source="raw",
    )
    assert rendered.shape == (video.CANVAS_HEIGHT, video.CANVAS_WIDTH, 3)
    assert rendered.dtype == np.uint8
    enlarged = np.repeat(
        np.repeat(image, video.DISPLAY_SCALE, axis=0),
        video.DISPLAY_SCALE,
        axis=1,
    )
    np.testing.assert_array_equal(
        rendered[video.HEADER_HEIGHT + 100 : -4, 4:-4],
        enlarged[100:-4, 4:-4],
    )
    with pytest.raises(ValueError, match="outside"):
        video._render_frame(
            image,
            episode=15,
            frame=269,
            evidence_index=0,
            evidence_total=58,
            condition=condition,
            current_score=1.25,
            prefix_score=-0.75,
            online_current_score=-0.1,
            online_prefix_score=0.2,
            checkpoint_step=14750,
            parameter_source="raw",
        )


def test_render_labels_both_heads_on_the_same_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    labels: list[str] = []

    def record_text(_canvas, text, _origin, **_kwargs) -> None:
        labels.append(text)

    monkeypatch.setattr(video.base, "_put_text", record_text)
    video._render_frame(
        np.zeros((video.MODEL_IMAGE_SIZE, video.MODEL_IMAGE_SIZE, 3), dtype=np.uint8),
        episode=15,
        frame=269,
        evidence_index=1,
        evidence_total=58,
        condition=video.Condition("native", video.fixed.PROMPTS[0], "left"),
        current_score=-1.25,
        prefix_score=-0.75,
        online_current_score=0.1,
        online_prefix_score=0.2,
        checkpoint_step=14750,
        parameter_source="raw",
    )
    assert any("FRESH FIXED" in label for label in labels)
    assert any("CHECKPOINT ONLINE" in label for label in labels)
    assert any("FRESH   frame" in label for label in labels)
    assert any("ONLINE  frame" in label for label in labels)


def test_expected_full_design_is_exactly_eight_videos() -> None:
    pairs = [
        (episode, condition.name)
        for episode in video.HELDOUT_EPISODES
        for condition in video._conditions(
            video.fixed.EpisodeSpec(
                episode=episode,
                prompt=video.fixed.PROMPTS[0] if episode < 30 else video.fixed.PROMPTS[1],
                side="left" if episode in (15, 44) else "right",
                label=0 if episode in (15, 44) else 1,
                length=100,
                parquet=Path(f"episode_{episode:06d}.parquet"),
                evidence_frames=(20,),
                approach_frames=(19,),
                heldout=True,
            )
        )
    ]
    assert len(pairs) == 8
    assert len(set(pairs)) == 8
    assert {condition for _episode, condition in pairs} == set(video.CONDITIONS)
