# ruff: noqa: SLF001

import dataclasses
import json
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from openpi.diagnostics import v31
from openpi.diagnostics import v31_pi0
from openpi.models import memory
from openpi.models import memory_diagnostics
from openpi.models import rtc
from openpi.policies import yam_policy
from openpi.shared import normalize
import openpi.transforms as transforms


def _memory_state() -> memory.MemoryState:
    return memory.MemoryState(
        fast_weights={
            "w0": np.arange(12, dtype=np.float32).reshape(1, 3, 4),
            "b0": np.arange(4, dtype=np.float32).reshape(1, 4),
        },
        momentum={
            "w0": np.full((1, 3, 4), 0.25, dtype=np.float32),
            "b0": np.full((1, 4), -0.5, dtype=np.float32),
        },
    )


def _observation(*, transformed_state: np.ndarray, raw_state: np.ndarray) -> v31_pi0.Pi0DiagnosticObservation:
    return v31_pi0.Pi0DiagnosticObservation(
        model_observation=None,  # type: ignore[arg-type]
        transformed_state=transformed_state,
        raw_state=raw_state,
        episode_id="demo1",
        observation_id="demo1:frame:40",
        raw_frame=40,
        policy_step=4,
        phase="decision",
        wall_time_s=4 / 3,
    )


def _action_side_mapping(**overrides):
    value = {
        "axis_index": 0,
        "axis_name": "left shoulder",
        "coordinate_frame": "joint radians",
        "window_start": 0,
        "window_end": 5,
        "aggregation": "endpoint_delta",
        "positive_side": "right",
        "threshold": 0.1,
    }
    value.update(overrides)
    return value


def test_yam_inverse_transforms_and_action_side_use_raw_robot_units():
    raw_state = np.linspace(10.0, 23.0, 14, dtype=np.float32)
    norm_stats = {
        "state": normalize.NormStats(
            mean=np.linspace(5.0, 18.0, 14, dtype=np.float32),
            std=np.full(14, 2.0, dtype=np.float32),
        ),
        "actions": normalize.NormStats(
            mean=np.zeros(14, dtype=np.float32),
            std=np.ones(14, dtype=np.float32),
        ),
    }
    transformed_state = transforms.Normalize(norm_stats)({"state": np.array(raw_state, copy=True)})["state"]
    transformed_state = transforms.PadStatesAndActions(32)({"state": transformed_state})["state"]

    evaluator = object.__new__(v31_pi0.Pi0SnapshotEvaluator)
    evaluator._output_transform = transforms.compose(
        [
            transforms.Unnormalize(norm_stats),
            transforms.AbsoluteActions(transforms.make_bool_mask(6, -1, 6, -1)),
            yam_policy.YamOutputs(),
        ]
    )
    evaluator._action_side_config = v31_pi0.ActionSideConfig.from_mapping(_action_side_mapping(), horizon=5)
    observation = _observation(transformed_state=transformed_state, raw_state=raw_state)

    model_actions = np.zeros((1, 5, 32), dtype=np.float32)
    model_actions[0, -1, 0] = 0.25
    robot_actions = evaluator._robot_actions(model_actions, observation)

    assert robot_actions.shape == (5, 14)
    np.testing.assert_allclose(robot_actions[:, 1:6], np.broadcast_to(raw_state[None, 1:6], (5, 5)), rtol=1e-6)
    np.testing.assert_allclose(robot_actions[:, 7:13], np.broadcast_to(raw_state[None, 7:13], (5, 6)), rtol=1e-6)
    np.testing.assert_allclose(robot_actions[:, [6, 13]], 0.0, atol=1e-7)
    metric = evaluator._action_side(robot_actions, observation)
    assert metric.side == "right"
    assert metric.metric == pytest.approx(0.25, rel=1e-5)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"axis_index": True}, "must be an integer"),
        ({"axis_index": 0.5}, "must be an integer"),
        ({"window_end": 6}, "window must satisfy"),
        ({"positive_side": "up"}, "must be left or right"),
        ({"threshold": float("nan")}, "must be finite"),
        ({"axis_name": ""}, "non-empty string"),
    ],
)
def test_action_side_config_rejects_ambiguous_or_invalid_schema(override, error):
    with pytest.raises((TypeError, ValueError), match=error):
        v31_pi0.ActionSideConfig.from_mapping(_action_side_mapping(**override), horizon=5)


def test_action_side_short_or_nonfinite_trajectory_is_undetermined():
    config = v31_pi0.ActionSideConfig.from_mapping(_action_side_mapping(), horizon=5)
    assert config is not None

    short = config.classify(np.ones((4, 14), dtype=np.float32), np.zeros(14, dtype=np.float32))
    assert short.side == "undetermined"
    assert "shorter" in short.details["reason"]

    nonfinite = np.ones((5, 14), dtype=np.float32)
    nonfinite[-1, 0] = np.nan
    invalid = config.classify(nonfinite, np.zeros(14, dtype=np.float32))
    assert invalid.side == "undetermined"
    assert "not finite" in invalid.details["reason"]


def test_complete_fast_state_hash_covers_weights_momentum_counter_and_write_metadata():
    evaluator = object.__new__(v31_pi0.Pi0SnapshotEvaluator)
    baseline = v31_pi0._make_fast_state(_memory_state(), writes=2, last_write_raw_frame=20, last_write_wall_time_s=0.5)
    clone = evaluator.clone_fast_state(baseline)

    assert evaluator.fast_state_hash(clone) == baseline.state_hash
    assert evaluator.write_count(clone) == 2
    memory_diagnostics.assert_memory_states_equal(clone.memory_state, baseline.memory_state)
    memory_diagnostics.assert_memory_states_isolated(clone.memory_state, baseline.memory_state)

    changed_momentum = memory_diagnostics.clone_memory_state(baseline.memory_state)
    changed_momentum.momentum["w0"][0, 0, 0] += 1
    assert (
        v31_pi0._make_fast_state(changed_momentum, writes=2).state_hash
        != v31_pi0._make_fast_state(baseline.memory_state, writes=2).state_hash
    )
    assert (
        v31_pi0._make_fast_state(baseline.memory_state, writes=3).state_hash
        != v31_pi0._make_fast_state(baseline.memory_state, writes=2).state_hash
    )
    assert (
        v31_pi0._make_fast_state(
            baseline.memory_state, writes=2, last_write_raw_frame=21, last_write_wall_time_s=0.5
        ).state_hash
        != baseline.state_hash
    )

    clone.memory_state.fast_weights["b0"][0, 0] += 1
    assert evaluator.fast_state_hash(clone) != clone.state_hash
    with pytest.raises(RuntimeError, match="cached hash"):
        evaluator.clone_fast_state(clone)


@pytest.mark.parametrize("writes", [-1, True, 1.5])
def test_fast_state_rejects_invalid_write_counter(writes):
    with pytest.raises((TypeError, ValueError), match="writes"):
        v31_pi0._make_fast_state(_memory_state(), writes=writes)


def test_pi0_state_serializer_round_trip_and_semantic_hash_validation(tmp_path):
    serializer = v31_pi0.Pi0StateSerializer()
    state = v31_pi0._make_fast_state(_memory_state(), writes=7, last_write_raw_frame=70, last_write_wall_time_s=7 / 3)
    path = tmp_path / "states" / "pre_decision.npz"
    serializer.save_state(path, state)
    restored = serializer.load_state(path, backend="numpy")

    assert restored.writes == 7
    assert restored.last_write_raw_frame == 70
    assert restored.last_write_wall_time_s == pytest.approx(7 / 3)
    assert restored.state_hash == state.state_hash
    memory_diagnostics.assert_memory_states_equal(restored.memory_state, state.memory_state)
    memory_diagnostics.assert_memory_states_isolated(restored.memory_state, state.memory_state)

    stale = dataclasses.replace(state, state_hash="0" * 64)
    with pytest.raises(ValueError, match="recorded hash"):
        serializer.save_state(tmp_path / "stale.npz", stale)

    inconsistent = memory_diagnostics.create_memory_snapshot(
        state.memory_state,
        writes=state.writes,
        metadata={
            "complete_state_hash": "0" * 64,
            "last_write_raw_frame": state.last_write_raw_frame,
            "last_write_wall_time_s": state.last_write_wall_time_s,
        },
    )
    inconsistent_path = memory_diagnostics.save_memory_snapshot(tmp_path / "inconsistent.npz", inconsistent)
    with pytest.raises(ValueError, match="complete hash"):
        serializer.load_state(inconsistent_path)


@pytest.mark.parametrize("defect", ["bounds", "float_delay", "nonfinite_action"])
def test_invalid_rtc_prefix_is_rejected_before_model_execution(defect):
    evaluator = object.__new__(v31_pi0.Pi0SnapshotEvaluator)
    evaluator._model = SimpleNamespace(action_horizon=5, action_dim=2)
    state = v31_pi0._make_fast_state(_memory_state(), writes=0)
    observation = _observation(transformed_state=np.zeros(2), raw_state=np.zeros(2))
    actions = np.zeros((1, 5, 2), dtype=np.float32)
    delay = np.asarray([2], dtype=np.int32)
    prefix_length = np.asarray([4], dtype=np.int32)
    if defect == "bounds":
        delay = np.asarray([5], dtype=np.int32)
    elif defect == "float_delay":
        delay = np.asarray([2.5], dtype=np.float32)
    else:
        actions[0, 0, 0] = np.nan
    invalid_prefix = rtc.ActionPrefix(
        actions=actions,
        delay=delay,
        prefix_length=prefix_length,
    )

    with pytest.raises(
        (TypeError, ValueError),
        match="0 <= delay <= prefix_length|integer dtype|finite values|finite floating-point",
    ):
        evaluator.evaluate_snapshot(observation, state, v31_pi0.Pi0RtcState(invalid_prefix))


def _write_documents(tmp_path, manifest, annotations):
    manifest_path = tmp_path / "episodes.json"
    annotations_path = tmp_path / "annotations.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    return manifest_path, annotations_path


def _valid_documents():
    return (
        {
            "version": "yam-eval-v1",
            "control_hz": 30.0,
            "episodes": [{"episode_id": "demo1", "path": "/recordings/demo1", "ground_truth_side": "left"}],
        },
        {
            "version": "manual-v1",
            "episodes": {
                "demo1": {
                    "reveal_frame": 30,
                    "close_frame": 45,
                    "decision_start_frame": 50,
                    "decision_end_frame": 80,
                }
            },
        },
    )


def test_manifest_and_annotation_schema_accepts_one_strict_valid_episode(monkeypatch, tmp_path):
    manifest, annotations = _valid_documents()
    manifest_path, annotations_path = _write_documents(tmp_path, manifest, annotations)
    source = v31_pi0.RawYamReplaySource(None, input_transforms=(), effective_stride=10)  # type: ignore[arg-type]
    monkeypatch.setattr(
        v31_pi0.RawYamReplaySource, "_load_episode", lambda self, entry, annotation: entry["episode_id"]
    )

    episodes, parsed = source.load_episodes(manifest_path, annotations_path)

    assert episodes == ("demo1",)
    assert parsed["demo1"] == v31.EpisodeAnnotation(30, 45, 50, 80, "manual-v1")
    assert source.manifest_data == manifest
    assert source.annotation_data == annotations


class _ReplayEvaluator:
    _control_hz = 30.0

    @staticmethod
    def initial_state():
        return "initial-fast-state"


def _replay_transform(item):
    return {
        "image": {
            "base_0_rgb": item["observation/image"],
            "left_wrist_0_rgb": item["observation/left_wrist_image"],
            "right_wrist_0_rgb": item["observation/right_wrist_image"],
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        },
        "state": item["observation/state"],
        "tokenized_prompt": np.zeros(4, dtype=np.int32),
        "tokenized_prompt_mask": np.ones(4, dtype=bool),
    }


def _raw_episode_files(tmp_path, *, left=None, right=None):
    directory = tmp_path / "demo1"
    directory.mkdir()
    for name in ("top_camera_rgb.mp4", "left_camera_rgb.mp4", "right_camera_rgb.mp4"):
        (directory / name).touch()
    if left is None:
        left = np.arange(100 * 7, dtype=np.float32).reshape(100, 7)
    if right is None:
        right = np.arange(100 * 7, dtype=np.float32).reshape(100, 7) + 1000
    np.save(directory / "left_joint_positions.npy", left)
    np.save(directory / "right_joint_positions.npy", right)
    return directory, left, right


def _stub_videos(monkeypatch, *, length=100):
    monkeypatch.setattr(v31_pi0, "_video_length", lambda path: length)
    monkeypatch.setattr(
        v31_pi0,
        "_read_selected_frames",
        lambda path, selected: {index: np.zeros((2, 2, 3), dtype=np.uint8) for index in selected},
    )


def test_raw_replay_preserves_joint_units_and_decision_window(monkeypatch, tmp_path):
    directory, left, right = _raw_episode_files(tmp_path)
    _stub_videos(monkeypatch)
    source = v31_pi0.RawYamReplaySource(_ReplayEvaluator(), input_transforms=[_replay_transform], effective_stride=10)
    annotation = v31.EpisodeAnnotation(20, 40, 60, 70, "manual-v1")

    episode = source._load_episode(
        {"episode_id": "demo1", "path": str(directory), "ground_truth_side": "left"}, annotation
    )

    assert [step.phase for step in episode.steps] == [
        "pre_reveal",
        "pre_reveal",
        "visible",
        "visible",
        "post_close",
        "post_close",
        "decision",
        "decision",
        "post_decision",
        "post_decision",
    ]
    frame_20 = episode.steps[2].observation
    np.testing.assert_array_equal(frame_20.raw_state, np.concatenate([left[20], right[20]]))
    assert frame_20.wall_time_s == pytest.approx(2 / 3)
    assert episode.initial_fast_state == "initial-fast-state"


@pytest.mark.parametrize("defect", ["wrong_width", "nonfinite", "length_mismatch", "annotation_out_of_range"])
def test_raw_replay_rejects_malformed_robot_state_or_annotation(monkeypatch, tmp_path, defect):
    left = np.zeros((100, 7), dtype=np.float32)
    right = np.zeros((100, 7), dtype=np.float32)
    if defect == "wrong_width":
        left = np.zeros((100, 6), dtype=np.float32)
    elif defect == "nonfinite":
        left[3, 2] = np.nan
    elif defect == "length_mismatch":
        right = np.zeros((99, 7), dtype=np.float32)
    directory, _, _ = _raw_episode_files(tmp_path, left=left, right=right)
    _stub_videos(monkeypatch)
    source = v31_pi0.RawYamReplaySource(_ReplayEvaluator(), input_transforms=[_replay_transform], effective_stride=10)
    decision_end = 100 if defect == "annotation_out_of_range" else 70
    annotation = v31.EpisodeAnnotation(20, 40, 60, decision_end, "manual-v1")

    with pytest.raises(ValueError, match="shape|non-finite|lengths differ|exceed"):
        source._load_episode({"episode_id": "demo1", "path": str(directory), "ground_truth_side": "left"}, annotation)


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda manifest, annotations: manifest.update(version="legacy"), "episode manifest version"),
        (lambda manifest, annotations: manifest.pop("control_hz"), "explicitly provide control_hz"),
        (lambda manifest, annotations: annotations.pop("version"), "annotation version"),
        (
            lambda manifest, annotations: manifest["episodes"].append(dict(manifest["episodes"][0])),
            "duplicate episode_id",
        ),
        (
            lambda manifest, annotations: annotations["episodes"]["demo1"].update(reveal_frame=30.5),
            "must be an integer",
        ),
    ],
)
def test_manifest_and_annotation_schema_rejects_ambiguous_documents(monkeypatch, tmp_path, mutate, error):
    manifest, annotations = _valid_documents()
    mutate(manifest, annotations)
    manifest_path, annotations_path = _write_documents(tmp_path, manifest, annotations)
    source = v31_pi0.RawYamReplaySource(None, input_transforms=(), effective_stride=10)  # type: ignore[arg-type]
    monkeypatch.setattr(v31_pi0.RawYamReplaySource, "_load_episode", lambda self, entry, annotation: None)

    with pytest.raises((TypeError, ValueError), match=error):
        source.load_episodes(manifest_path, annotations_path)


def test_json_reader_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"version":"yam-eval-v1","version":"other"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        v31_pi0._read_json(duplicate, name="episode manifest")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"control_hz":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON number"):
        v31_pi0._read_json(nonfinite, name="episode manifest")


@pytest.mark.parametrize("value", [0, -1, True, "30", float("nan"), float("inf")])
def test_episode_manifest_control_hz_must_be_explicit_positive_and_finite(value):
    with pytest.raises((TypeError, ValueError), match="control_hz"):
        v31_pi0._require_control_hz({"control_hz": value})
    with pytest.raises(ValueError, match="explicitly provide control_hz"):
        v31_pi0._require_control_hz({})


def _write_checkpoint_metadata(directory, custom_metadata):
    directory.mkdir()
    (directory / "_CHECKPOINT_METADATA").write_text(json.dumps({"custom_metadata": custom_metadata}), encoding="utf-8")
    return directory


def test_checkpoint_static_provenance_accepts_legacy_unverified_and_matching_verified(tmp_path):
    legacy = _write_checkpoint_metadata(tmp_path / "legacy", {})
    result = v31_pi0._read_checkpoint_static_provenance(legacy, expected_config_name="pi05_yam_mem_v31")
    assert result.verified is False
    assert result.config_name is None
    assert result.memory_write_source is None

    verified = _write_checkpoint_metadata(
        tmp_path / "verified",
        {"config_name": "pi05_yam_mem_v31", "memory_write_source": "post_attention"},
    )
    result = v31_pi0._read_checkpoint_static_provenance(verified, expected_config_name="pi05_yam_mem_v31")
    assert result.verified is True
    assert result.config_name == "pi05_yam_mem_v31"
    assert result.memory_write_source == "post_attention"


@pytest.mark.parametrize(
    ("custom_metadata", "error"),
    [
        ({"config_name": "pi05_yam_mem_v31"}, "provide config_name and memory_write_source together"),
        (
            {"config_name": "wrong_config", "memory_write_source": "post_attention"},
            "does not match requested config",
        ),
        (
            {"config_name": "pi05_yam_mem_v31", "memory_write_source": "pre_attention"},
            "does not match required",
        ),
        ({"config_name": True, "memory_write_source": "post_attention"}, "non-empty string"),
    ],
)
def test_checkpoint_static_provenance_rejects_partial_or_contradictory_fields(tmp_path, custom_metadata, error):
    checkpoint = _write_checkpoint_metadata(tmp_path / "checkpoint", custom_metadata)
    with pytest.raises((TypeError, ValueError), match=error):
        v31_pi0._read_checkpoint_static_provenance(checkpoint, expected_config_name="pi05_yam_mem_v31")


@pytest.mark.parametrize(
    "payload",
    [
        "{",
        '{"custom_metadata":[],"custom_metadata":{}}',
        '{"custom_metadata":{"noise":NaN}}',
        '{"item_handlers":{}}',
        '{"custom_metadata":[]}',
    ],
)
def test_checkpoint_static_provenance_rejects_malformed_or_corrupt_metadata(tmp_path, payload):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "_CHECKPOINT_METADATA").write_text(payload, encoding="utf-8")
    with pytest.raises((TypeError, ValueError)):
        v31_pi0._read_checkpoint_static_provenance(checkpoint, expected_config_name="pi05_yam_mem_v31")


def test_pi0_application_rejects_unimplemented_token_diagnostics_before_loading(tmp_path):
    request = v31.DiagnosticRunRequest(
        mode="offline",
        checkpoint=str(tmp_path / "missing-checkpoint"),
        tests=("temporal",),
        output_dir=tmp_path / "output",
        seed=0,
        diagnostic_level="tokens",
        adapter="openpi.diagnostics.v31_pi0:create_application",
        episodes=str(tmp_path / "episodes.json"),
        annotations=str(tmp_path / "annotations.json"),
    )
    with pytest.raises(NotImplementedError, match="does not yet produce token diagnostics"):
        v31_pi0.Pi0V31DiagnosticApplication(request)


def test_build_manifest_marks_legacy_checkpoint_static_provenance_unverified(monkeypatch, tmp_path):
    @dataclasses.dataclass(frozen=True)
    class FakeModelConfig:
        action_horizon: int = 50
        simulated_delay: tuple[int, int] = (0, 6)
        memory_write_source: str = "post_attention"

    @dataclasses.dataclass(frozen=True)
    class FakeTrainConfig:
        name: str = "pi05_yam_mem_v31"
        model: FakeModelConfig = dataclasses.field(default_factory=FakeModelConfig)

    checkpoint = tmp_path / "checkpoint" / "2500"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params" / "tiny").write_bytes(b"params")
    manifest_path, annotations_path = _write_documents(tmp_path, *_valid_documents())
    provenance = v31_pi0.CheckpointStaticProvenance(
        verified=False,
        metadata_path=str(checkpoint / "_CHECKPOINT_METADATA"),
    )
    application = object.__new__(v31_pi0.Pi0V31DiagnosticApplication)
    application._effective_stride = 10
    application._configured_stride = 10
    application._manifest_data = {"version": "yam-eval-v1", "control_hz": 30.0}
    application._checkpoint_provenance = provenance
    application._checkpoint = checkpoint
    application._train_config = FakeTrainConfig()
    application._source = SimpleNamespace(annotation_data={"version": "manual-v1"})
    application._control_hz = 30.0
    monkeypatch.setattr(v31, "current_code_revision", lambda path: "test-revision")
    request = v31.DiagnosticRunRequest(
        mode="offline",
        checkpoint=str(checkpoint),
        tests=("temporal",),
        output_dir=tmp_path / "output",
        seed=0,
        diagnostic_level="basic",
        adapter="openpi.diagnostics.v31_pi0:create_application",
        episodes=str(manifest_path),
        annotations=str(annotations_path),
    )

    manifest = application.build_manifest(request)

    assert manifest.rtc_settings["checkpoint_static_config_provenance_verified"] is False
    assert manifest.rtc_settings["checkpoint_config_name"] is None
    assert any("static-config provenance is unverified" in warning for warning in manifest.warnings)


def test_json_config_serializes_tyro_missing_deterministically():
    import tyro

    assert v31_pi0._json_config(tyro.MISSING) == "<MISSING>"


@pytest.mark.parametrize("stride", [0, -1, True, 1.5])
def test_replay_source_requires_positive_integral_stride(stride):
    with pytest.raises((TypeError, ValueError), match="effective_stride"):
        v31_pi0.RawYamReplaySource(None, input_transforms=(), effective_stride=stride)  # type: ignore[arg-type]


def test_serializer_can_restore_jax_backend(tmp_path):
    serializer = v31_pi0.Pi0StateSerializer()
    state = v31_pi0._make_fast_state(_memory_state(), writes=0)
    path = tmp_path / "state.npz"
    serializer.save_state(path, state)

    restored = serializer.load_state(path, backend="jax")

    assert all(isinstance(value, jax.Array) for value in jax.tree.leaves(restored.memory_state))
    assert restored.state_hash == state.state_hash
