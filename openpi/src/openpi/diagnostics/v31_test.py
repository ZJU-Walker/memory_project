from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from openpi.diagnostics import v31


@dataclasses.dataclass
class _FakeState:
    value: float
    momentum: float
    writes: int = 0


@dataclasses.dataclass(frozen=True)
class _FakeObservation:
    episode_id: str
    observation_id: str
    raw_frame: int
    policy_step: int
    phase: str
    observation_bias: float = 0.0


class _FakeAdapter:
    def __init__(self, *, mutate_input: bool = False, action_metric: bool = True):
        self.mutate_input = mutate_input
        self.action_metric = action_metric
        self.calls: list[dict] = []

    def clone_fast_state(self, state: object) -> object:
        return copy.deepcopy(state)

    def fast_state_hash(self, state: object) -> str:
        assert isinstance(state, _FakeState)
        payload = f"{state.value:.17g}|{state.momentum:.17g}|{state.writes}".encode()
        return hashlib.sha256(payload).hexdigest()

    def write_count(self, state: object) -> int:
        assert isinstance(state, _FakeState)
        return state.writes

    def evaluate_snapshot(
        self,
        observation: object,
        fast_state: object,
        rtc_state: object | None,
        *,
        forced_subtask: str | None = None,
        zero_read: bool = False,
        allow_write: bool = False,
        seed: int = 0,
    ) -> v31.DiagnosticResult:
        assert isinstance(observation, _FakeObservation)
        assert isinstance(fast_state, _FakeState)
        before_hash = self.fast_state_hash(fast_state)
        before_count = fast_state.writes
        read = 0.0 if zero_read else fast_state.value
        margin = observation.observation_bias + read
        if forced_subtask == v31.CANONICAL_LEFT_SUBTASK:
            action = -1.0
            decoded = forced_subtask
        elif forced_subtask == v31.CANONICAL_RIGHT_SUBTASK:
            action = 1.0
            decoded = forced_subtask
        else:
            action = -1.0 if margin > 0 else 1.0
            decoded = v31.CANONICAL_LEFT_SUBTASK if margin > 0 else v31.CANONICAL_RIGHT_SUBTASK
        # Deterministic noise depends only on the explicit seed.
        trajectory = np.full((3, 2), action + np.random.default_rng(seed).normal(scale=1e-4), dtype=np.float32)
        committed = np.array(trajectory, copy=True)
        if rtc_state in ("collapse", "broker-collapse"):
            committed[:] = 1.0
        next_state = copy.deepcopy(fast_state)
        if allow_write:
            reveal = 0.0
            if observation.phase == "visible":
                reveal = 2.0 if "left" in observation.episode_id else -2.0
            # After close, normal writes slowly drive the state toward the right.
            interference = -1.0 if observation.phase in ("post_close", "decision") else 0.0
            next_state.momentum = 0.5 * next_state.momentum + reveal + interference
            next_state.value += next_state.momentum
            next_state.writes += 1
        if self.mutate_input:
            fast_state.value += 999
        metric = (
            v31.ActionSideMetric(
                side="left" if action < 0 else "right",
                metric=float(action),
                axis_name="fake-x",
                coordinate_frame="fake-world",
                decision_window="all",
                pass_threshold=0.0,
            )
            if self.action_metric
            else v31.ActionSideMetric()
        )
        committed_metric = (
            v31.ActionSideMetric(
                side="left" if committed[-1, 0] < 0 else "right",
                metric=float(committed[-1, 0]),
                axis_name="fake-x",
                coordinate_frame="fake-world",
                decision_window="all",
                pass_threshold=0.0,
            )
            if self.action_metric
            else v31.ActionSideMetric()
        )
        self.calls.append(
            {
                "observation": observation.observation_id,
                "input_hash": before_hash,
                "forced_subtask": forced_subtask,
                "zero_read": zero_read,
                "allow_write": allow_write,
                "seed": seed,
            }
        )
        left_logp = -float(np.logaddexp(0.0, -margin))
        right_logp = -float(np.logaddexp(0.0, margin))
        return v31.DiagnosticResult(
            episode_id=observation.episode_id,
            observation_id=observation.observation_id,
            raw_frame=observation.raw_frame,
            policy_step=observation.policy_step,
            phase=observation.phase,
            seed=seed,
            left_score=v31.CanonicalScore(v31.CANONICAL_LEFT_SUBTASK, total_logp=left_logp, token_count=3),
            right_score=v31.CanonicalScore(v31.CANONICAL_RIGHT_SUBTASK, total_logp=right_logp, token_count=3),
            decoded_subtask=decoded,
            forced_subtask=forced_subtask,
            zero_read=zero_read,
            allow_write=allow_write,
            pre_rtc_action=trajectory,
            committed_action=committed,
            pre_rtc_side=metric,
            committed_side=committed_metric,
            memory_read_norm=abs(read),
            memory_gate_norm=1.0,
            surprise=abs(read - observation.observation_bias),
            fast_state_hash_before=before_hash,
            fast_state_hash_after=self.fast_state_hash(next_state),
            write_count_before=before_count,
            write_count_after=next_state.writes,
            write_due=True,
            write_occurred=allow_write,
            rtc_metadata={
                "state": rtc_state if isinstance(rtc_state, str | int | float | bool | type(None)) else "structured",
                "broker_simulated": isinstance(rtc_state, str) and rtc_state.startswith("broker-"),
            },
            configured_write_every_frames=10,
            action_horizon=3,
            control_hz=30.0,
            token_diagnostics={"token_error": np.arange(4, dtype=np.float32)},
            next_fast_state=next_state,
        )


class _AliasingAdapter(_FakeAdapter):
    def clone_fast_state(self, state: object) -> object:
        return state


class _InputMutatingAdapter(_FakeAdapter):
    def evaluate_snapshot(self, observation, fast_state, rtc_state, **kwargs):
        object.__setattr__(observation, "observation_bias", 0.75)
        if isinstance(rtc_state, dict):
            rtc_state["mutated"] = True
        return super().evaluate_snapshot(observation, fast_state, rtc_state, **kwargs)


def _step(
    episode: str,
    policy_step: int,
    phase: str,
    *,
    write_due: bool = True,
    observation_bias: float = 0.0,
    rtc_state=None,
) -> v31.ReplayStep:
    raw_frame = policy_step * 10
    obs = _FakeObservation(episode, f"{episode}:{raw_frame}", raw_frame, policy_step, phase, observation_bias)
    return v31.ReplayStep(
        observation=obs,
        observation_id=obs.observation_id,
        raw_frame=raw_frame,
        policy_step=policy_step,
        phase=phase,
        rtc_state=rtc_state,
        wall_time_s=policy_step / 3,
        write_due=write_due,
    )


def _episode(episode_id: str = "left_episode") -> v31.ReplayEpisode:
    return v31.ReplayEpisode(
        episode_id=episode_id,
        ground_truth_side="left" if "left" in episode_id else "right",
        steps=(
            _step(episode_id, 0, "pre_reveal"),
            _step(episode_id, 1, "visible"),
            _step(episode_id, 2, "visible"),
            _step(episode_id, 3, "post_close"),
            _step(episode_id, 4, "decision"),
        ),
        initial_fast_state=_FakeState(0.0, 0.0),
    )


def _manifest() -> v31.RunManifest:
    return v31.RunManifest(
        run_id="fake-run",
        created_at_utc="2026-08-12T00:00:00Z",
        code_revision="abc123-dirty",
        checkpoint_path="/checkpoint/2500",
        checkpoint_hash="sha256:test",
        loaded_config={"name": "pi05_yam_mem_v31", "memory_write_source": "post_attention"},
        overrides={},
        episode_manifest="episodes.json",
        episode_manifest_hash="sha256:episodes",
        episode_version="v1",
        annotations_path="annotations.json",
        annotations_hash="sha256:annotations",
        annotation_version="v1",
        seeds=(0,),
        configured_write_every_frames=10,
        effective_write_every_frames=10,
        action_horizon=50,
        control_hz=30.0,
        rtc_settings={"max_delay": 6},
        hardware_mode="offline",
        enabled_tests=("oracle", "state_swap", "freeze", "temporal"),
        diagnostic_level="tokens",
        counterfactual_mode="shadow",
        execute_actions=False,
        require_operator_confirmation=False,
        operator_confirmed=False,
    )


def test_canonical_score_and_diagnostic_result_validate_schema_and_copy_arrays():
    with pytest.raises(ValueError, match="token_count"):
        v31.CanonicalScore(v31.CANONICAL_LEFT_SUBTASK, -1, 0)
    with pytest.raises(ValueError, match="mean_logp"):
        v31.CanonicalScore(v31.CANONICAL_LEFT_SUBTASK, -2, 2, mean_logp=-3)
    with pytest.raises(ValueError, match="<= 0"):
        v31.CanonicalScore(v31.CANONICAL_LEFT_SUBTASK, 0.1, 2)

    adapter = _FakeAdapter()
    state = _FakeState(2.0, 0.0)
    result = v31.evaluate_snapshot(adapter, _step("left_episode", 4, "decision").observation, state, None, seed=7)
    assert result.delta_lr == 2.0
    assert result.delta_lr_length_normalized == pytest.approx(2 / 3)
    assert result.fast_state_hash_before == result.fast_state_hash_after
    assert result.write_count_before == result.write_count_after == 0
    assert not result.pre_rtc_action.flags.writeable
    event = result.to_event_dict(test_name="temporal", branch="normal")
    assert "pre_rtc_action" not in event
    assert event["actions_file"] is None


def test_evaluate_snapshot_is_deterministic_and_isolates_caller_state():
    adapter = _FakeAdapter()
    state = _FakeState(1.5, 0.2, writes=3)
    obs = _step("left_episode", 4, "decision").observation
    first = v31.evaluate_snapshot(adapter, obs, state, None, seed=123)
    second = v31.evaluate_snapshot(adapter, obs, state, None, seed=123)
    assert first.result_id == second.result_id
    np.testing.assert_array_equal(first.pre_rtc_action, second.pre_rtc_action)
    assert state == _FakeState(1.5, 0.2, writes=3)
    assert adapter.calls[0] == adapter.calls[1]

    with pytest.raises(RuntimeError, match="caller-owned"):
        v31.evaluate_snapshot(_AliasingAdapter(mutate_input=True), obs, state, None, seed=123)


def test_evaluate_snapshot_defensively_clones_observation_and_rtc_state():
    observation = _step("left_episode", 1, "visible", observation_bias=0.25).observation
    rtc_state = {"prefix": np.arange(4, dtype=np.float32), "mutated": False}
    result = v31.evaluate_snapshot(_InputMutatingAdapter(), observation, _FakeState(0, 0), rtc_state, seed=1)
    assert observation.observation_bias == 0.25
    assert rtc_state["mutated"] is False
    np.testing.assert_array_equal(rtc_state["prefix"], np.arange(4, dtype=np.float32))
    assert result.delta_lr == pytest.approx(0.75)


def test_read_only_branch_rejects_silent_state_change():
    class BadAdapter(_FakeAdapter):
        def evaluate_snapshot(self, *args, **kwargs):
            result = super().evaluate_snapshot(*args, **kwargs)
            next_state = copy.deepcopy(result.next_fast_state)
            next_state.value += 1
            return dataclasses.replace(
                result,
                fast_state_hash_after=self.fast_state_hash(next_state),
                next_fast_state=next_state,
            )

    with pytest.raises(RuntimeError, match="read-only counterfactual"):
        v31.evaluate_snapshot(
            BadAdapter(),
            _step("left_episode", 0, "pre_reveal").observation,
            _FakeState(0, 0),
            None,
            allow_write=False,
        )


def test_oracle_test_is_matched_read_only_and_classifies_action_vs_rtc():
    episode = _episode()
    step = episode.steps[-1]
    state = _FakeState(1.0, 0.0, writes=4)
    adapter = _FakeAdapter()
    report = v31.run_oracle_test(adapter, episode, step, state, seed=9)
    assert report.valid
    assert report.interpretation == "action_conditioning_pass_rtc_not_evaluated"
    assert report.metadata["broker_evaluated"] is False
    assert [record.branch for record in report.records] == ["forced_left", "forced_right"]
    assert {call["input_hash"] for call in adapter.calls} == {adapter.fast_state_hash(state)}
    assert {call["seed"] for call in adapter.calls} == {9}
    assert all(not record.result.write_occurred for record in report.records)

    collapsed = dataclasses.replace(step, rtc_state="broker-collapse")
    report = v31.run_oracle_test(_FakeAdapter(), episode, collapsed, state)
    assert report.interpretation == "rtc_broker_failure"
    assert report.metadata["broker_evaluated"] is True
    broker_pass = dataclasses.replace(step, rtc_state="broker-pass")
    report = v31.run_oracle_test(_FakeAdapter(), episode, broker_pass, state)
    assert report.interpretation == "pass"
    undetermined = v31.run_oracle_test(_FakeAdapter(action_metric=False), episode, step, state)
    assert not undetermined.valid
    assert undetermined.interpretation == "invalid_or_undetermined"


def test_state_swap_varies_only_complete_state_and_reports_effect():
    adapter = _FakeAdapter()
    left = _episode("left_episode")
    right = _episode("right_episode")
    left_state = _FakeState(2.0, 0.7, writes=5)
    right_state = _FakeState(-2.0, -0.7, writes=5)
    m0 = _FakeState(-0.5, 0.0, writes=0)
    snapshots = {
        name: v31.FastStateSnapshot(
            state=state,
            source_episode=name,
            ground_truth_side=("left" if name == "left_state" else "right" if name == "right_state" else None),
            phase="decision",
            write_count=state.writes,
            state_hash=adapter.fast_state_hash(state),
        )
        for name, state in (("left_state", left_state), ("right_state", right_state), ("m0", m0))
    }
    report = v31.run_state_swap_test(
        adapter,
        {"left_obs": (left, left.steps[-1]), "right_obs": (right, right.steps[-1])},
        snapshots,
        seed=4,
    )
    assert report.interpretation == "strong_causal_pass"
    assert report.metadata["swap_effects"] == {"left_obs": 4.0, "right_obs": 4.0}
    assert report.metadata["broker_evaluated"] is False
    assert len(report.records) == 8
    assert left_state == _FakeState(2.0, 0.7, writes=5)
    assert right_state == _FakeState(-2.0, -0.7, writes=5)
    zero = [record for record in report.records if record.metadata["state_variant"] == "zero_read"]
    assert all(record.result.memory_read_norm == 0 for record in zero)


def test_state_swap_never_reports_causal_pass_when_write_age_is_unmatched():
    adapter = _FakeAdapter()
    left = _episode("left_episode")
    right = _episode("right_episode")
    left_state = _FakeState(2.0, 0.7, writes=5)
    right_state = _FakeState(-2.0, -0.7, writes=3)
    m0 = _FakeState(0.0, 0.0, writes=0)
    snapshots = {
        "left_state": v31.FastStateSnapshot(
            left_state, "left_history", "left", "decision", left_state.writes, adapter.fast_state_hash(left_state)
        ),
        "right_state": v31.FastStateSnapshot(
            right_state,
            "right_history",
            "right",
            "decision",
            right_state.writes,
            adapter.fast_state_hash(right_state),
        ),
        "m0": v31.FastStateSnapshot(m0, "learned_m0", None, "reset", 0, adapter.fast_state_hash(m0)),
    }
    report = v31.run_state_swap_test(
        adapter,
        {"left_obs": (left, left.steps[-1]), "right_obs": (right, right.steps[-1])},
        snapshots,
    )
    # The underlying margins, decoded subtasks, and actions meet the strong-pass
    # conditions, so this specifically exercises the write-age confound guard.
    assert report.metadata["swap_effects"] == {"left_obs": 4.0, "right_obs": 4.0}
    assert report.interpretation == "write_age_confounded"
    assert report.metadata["left_right_write_counts_matched"] is False
    assert report.metadata["left_state_write_count"] == 5
    assert report.metadata["right_state_write_count"] == 3
    assert report.metadata["write_count_difference"] == 2


def test_state_swap_validates_snapshot_meaning_and_reports_learned_right_prior():
    adapter = _FakeAdapter()
    left = _episode("left_episode")
    right = _episode("right_episode")
    same_right_state = _FakeState(-2.0, 0.0, writes=5)
    m0 = _FakeState(-0.5, 0.0, writes=0)
    snapshots = {
        "left_state": v31.FastStateSnapshot(
            same_right_state,
            "left_history",
            "left",
            "decision",
            5,
            adapter.fast_state_hash(same_right_state),
        ),
        "right_state": v31.FastStateSnapshot(
            same_right_state,
            "right_history",
            "right",
            "decision",
            5,
            adapter.fast_state_hash(same_right_state),
        ),
        "m0": v31.FastStateSnapshot(m0, "learned_m0", None, "reset", 0, adapter.fast_state_hash(m0)),
    }
    report = v31.run_state_swap_test(
        adapter,
        {"left_obs": (left, left.steps[-1]), "right_obs": (right, right.steps[-1])},
        snapshots,
    )
    assert report.interpretation == "learned_right_prior"
    assert report.metadata["m0_right_favoring"] is True

    wrong = {**snapshots, "left_state": dataclasses.replace(snapshots["left_state"], ground_truth_side="right")}
    with pytest.raises(ValueError, match="left_state"):
        v31.run_state_swap_test(adapter, {"left_obs": (left, left.steps[-1])}, wrong)


def test_freeze_boundaries_are_aligned_to_scheduled_writes_and_isolated():
    adapter = _FakeAdapter()
    episode = _episode()
    annotation = v31.EpisodeAnnotation(reveal_frame=5, close_frame=25, decision_start_frame=40)
    report = v31.run_freeze_test(adapter, episode, annotation, seed=0)
    assert report.valid
    assert report.metadata["reveal_write_frame"] == 10
    assert report.metadata["close_write_frame"] == 20
    counts = {
        branch: max(record.result.write_count_after for record in report.records if record.branch == branch)
        for branch in ("normal", "freeze_after_reveal", "freeze_after_close")
    }
    assert counts == {"normal": 5, "freeze_after_reveal": 2, "freeze_after_close": 3}
    # Each variant begins from the same complete blank state, rather than a previous branch's output.
    initial_hashes = [
        record.result.fast_state_hash_before for record in report.records if record.result.policy_step == 0
    ]
    assert len(set(initial_hashes)) == 1


def test_freeze_marks_missing_reveal_write_invalid_instead_of_freezing_empty_state():
    episode = _episode()
    no_reveal_write = dataclasses.replace(
        episode,
        steps=tuple(
            dataclasses.replace(step, write_due=False) if step.phase == "visible" else step for step in episode.steps
        ),
    )
    report = v31.run_freeze_test(
        _FakeAdapter(),
        no_reveal_write,
        v31.EpisodeAnnotation(reveal_frame=5, close_frame=25, decision_start_frame=40),
    )
    assert not report.valid
    assert report.interpretation == "invalid_boundary"
    assert report.records == ()


def test_temporal_logs_prewrite_hash_then_threads_postwrite_state():
    adapter = _FakeAdapter()
    episode = _episode()
    report = v31.run_temporal_test(
        adapter,
        episode,
        v31.EpisodeAnnotation(reveal_frame=5, close_frame=25, decision_start_frame=40),
        seed=99,
    )
    assert len(report.records) == len(episode.steps)
    for previous, current in zip(report.records, report.records[1:], strict=False):
        assert previous.result.fast_state_hash_after == current.result.fast_state_hash_before
        assert previous.result.write_count_after == current.result.write_count_before
    assert [record.result.write_count_before for record in report.records] == [0, 1, 2, 3, 4]
    observed_seeds = [record.result.seed for record in report.records]
    assert len(set(observed_seeds)) == len(episode.steps)
    second = v31.run_temporal_test(
        _FakeAdapter(),
        episode,
        v31.EpisodeAnnotation(reveal_frame=5, close_frame=25, decision_start_frame=40),
        seed=99,
    )
    assert [record.result.seed for record in second.records] == observed_seeds


@pytest.mark.parametrize(
    ("safety", "tests"),
    [
        (v31.ExecutionSafety(hardware_mode="realtime", counterfactual_mode="active"), ("oracle",)),
        (
            v31.ExecutionSafety(
                hardware_mode="realtime", counterfactual_mode="active", execute_actions=True, active_forced_side="left"
            ),
            ("oracle",),
        ),
        (
            v31.ExecutionSafety(
                hardware_mode="realtime",
                counterfactual_mode="active",
                execute_actions=True,
                require_operator_confirmation=True,
                operator_confirmed=True,
                active_forced_side="left",
            ),
            ("oracle", "temporal"),
        ),
        (v31.ExecutionSafety(hardware_mode="offline", execute_actions=True), ("oracle",)),
    ],
)
def test_execution_interlock_rejects_unsafe_counterfactuals(safety, tests):
    with pytest.raises(PermissionError):
        v31.validate_execution_safety(safety, tests)


def test_active_oracle_requires_one_confirmed_side_and_never_runs_other_branch():
    safety = v31.ExecutionSafety(
        hardware_mode="realtime",
        counterfactual_mode="active",
        execute_actions=True,
        require_operator_confirmation=True,
        operator_confirmed=True,
        active_forced_side="left",
    )
    adapter = _FakeAdapter()
    episode = _episode()
    report = v31.run_oracle_test(adapter, episode, episode.steps[-1], _FakeState(0, 0), safety=safety)
    assert len(report.records) == 1
    assert report.records[0].branch == "forced_left"
    assert report.records[0].metadata["execution_authorized"] is True
    assert report.interpretation == "active_oracle_action_conditioning_pass_rtc_not_evaluated"

    broker_step = dataclasses.replace(episode.steps[-1], rtc_state="broker-pass")
    report = v31.run_oracle_test(_FakeAdapter(), episode, broker_step, _FakeState(0, 0), safety=safety)
    assert report.interpretation == "active_oracle_pass"

    collapsed_step = dataclasses.replace(episode.steps[-1], rtc_state="broker-collapse")
    report = v31.run_oracle_test(_FakeAdapter(), episode, collapsed_step, _FakeState(0, 0), safety=safety)
    assert report.interpretation == "active_oracle_rtc_broker_failure"

    report = v31.run_oracle_test(
        _FakeAdapter(action_metric=False), episode, broker_step, _FakeState(0, 0), safety=safety
    )
    assert not report.valid
    assert report.interpretation == "invalid_or_undetermined"

    class WrongPreRtcSide(_FakeAdapter):
        def evaluate_snapshot(self, *args, **kwargs):
            result = super().evaluate_snapshot(*args, **kwargs)
            return dataclasses.replace(result, pre_rtc_side=dataclasses.replace(result.pre_rtc_side, side="right"))

    report = v31.run_oracle_test(WrongPreRtcSide(), episode, broker_step, _FakeState(0, 0), safety=safety)
    assert report.interpretation == "active_oracle_action_conditioning_failure"


def test_temporal_right_favoring_uses_raw_delta_and_attractor_requires_both_sides():
    annotation = v31.EpisodeAnnotation(reveal_frame=5, close_frame=25, decision_start_frame=40)
    left = v31.run_temporal_test(_FakeAdapter(), _episode("left_episode"), annotation)
    right = v31.run_temporal_test(_FakeAdapter(), _episode("right_episode"), annotation)
    assert right.interpretation == "post_close_right_favoring"
    assert right.metadata["after_close_delta_lr_mean"] < 0
    assert all(record.result.signed_margin >= 0 for record in right.records)

    ordinary = v31._temporal_cross_episode_summary((left, right))  # noqa: SLF001
    assert ordinary["interpretation"] == "no_shared_right_attractor"

    biased_left = dataclasses.replace(
        left,
        metadata={
            **left.metadata,
            "after_close_delta_lr_mean": -1.0,
            "after_reveal_delta_lr_drift": -0.5,
        },
    )
    aggregate = v31._temporal_cross_episode_summary((biased_left, right))  # noqa: SLF001
    assert aggregate["interpretation"] == "right_attractor_or_prior"
    assert aggregate["all_episode_sides_right_favoring"] is True
    assert aggregate["ground_truth_sides"] == ["left", "right"]


def test_temporal_marks_missing_reveal_write_invalid_but_retains_step_records():
    episode = _episode()
    no_reveal_write = dataclasses.replace(
        episode,
        steps=tuple(
            dataclasses.replace(step, write_due=False) if step.phase == "visible" else step for step in episode.steps
        ),
    )
    report = v31.run_temporal_test(
        _FakeAdapter(),
        no_reveal_write,
        v31.EpisodeAnnotation(reveal_frame=5, close_frame=25, decision_start_frame=40),
    )
    assert not report.valid
    assert report.interpretation == "invalid_boundary"
    assert len(report.records) == len(episode.steps)


def test_artifacts_are_self_contained_strict_and_keep_dense_arrays_out_of_json(tmp_path: Path):
    writer = v31.ArtifactWriter(tmp_path / "run", _manifest())
    episode = _episode()
    reports = [
        v31.run_oracle_test(_FakeAdapter(), episode, episode.steps[-1], _FakeState(1, 0)),
        v31.run_temporal_test(
            _FakeAdapter(),
            episode,
            v31.EpisodeAnnotation(reveal_frame=5, close_frame=25, decision_start_frame=40),
        ),
    ]
    for report in reports:
        writer.add_report(report)
    invalid = v31.ExperimentReport(
        test_name="freeze",
        records=(),
        valid=False,
        interpretation="invalid_boundary",
        invalid_reason="fake missing reveal",
    )
    writer.add_report(invalid)
    summary = writer.finalize()

    expected = {
        "run_manifest.json",
        "events.jsonl",
        "summary.json",
        "oracle_results.csv",
        "state_swap_results.csv",
        "freeze_results.csv",
        "temporal_results.csv",
        "actions",
        "states",
        "token_diagnostics",
        "plots",
    }
    assert expected <= {path.name for path in writer.output_dir.iterdir()}
    assert {path.name for path in (writer.output_dir / "plots").iterdir()} == {
        "margin_over_time.png",
        "margin_vs_write_count.png",
        "state_swap_effect.png",
    }
    events = [json.loads(line) for line in (writer.output_dir / "events.jsonl").read_text().splitlines()]
    assert events
    assert all(event["schema_version"] == v31.SCHEMA_VERSION for event in events)
    assert all("pre_rtc_action" not in event and event["actions_file"] for event in events)
    for event in events:
        arrays = np.load(writer.output_dir / event["actions_file"], allow_pickle=False)
        assert set(arrays.files) == {"pre_rtc_action", "committed_action"}
        token_arrays = np.load(writer.output_dir / event["token_diagnostics_file"], allow_pickle=False)
        assert token_arrays["token_error"].shape == (4,)
    assert summary["report_counts"] == {"total": 3, "valid": 2, "invalid": 1}
    assert summary["reports"][-1]["invalid_reason"] == "fake missing reveal"


def test_artifact_directory_must_not_preexist(tmp_path: Path):
    path = tmp_path / "existing"
    path.mkdir()
    with pytest.raises(FileExistsError):
        v31.ArtifactWriter(path, _manifest())


def test_invalid_report_is_invalid_in_event_and_csv_and_basic_mode_rejects_token_arrays(tmp_path: Path):
    episode = _episode()
    invalid = v31.run_oracle_test(_FakeAdapter(action_metric=False), episode, episode.steps[-1], _FakeState(0, 0))
    assert not invalid.valid
    assert invalid.records
    writer = v31.ArtifactWriter(tmp_path / "invalid-run", _manifest())
    writer.add_report(invalid)
    summary = writer.finalize()
    assert summary["record_counts"] == {"total": 2, "valid": 0, "invalid": 2}
    events = [json.loads(line) for line in (writer.output_dir / "events.jsonl").read_text().splitlines()]
    assert all(event["extra"]["report_valid"] is False for event in events)
    with (writer.output_dir / "oracle_results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["valid"] == "False" and row["report_valid"] == "False" for row in rows)
    assert all(row["invalid_reason"] for row in rows)

    basic_manifest = dataclasses.replace(_manifest(), diagnostic_level="basic")
    basic_writer = v31.ArtifactWriter(tmp_path / "basic-run", basic_manifest)
    temporal = v31.run_temporal_test(
        _FakeAdapter(),
        episode,
        v31.EpisodeAnnotation(reveal_frame=5, close_frame=25, decision_start_frame=40),
    )
    with pytest.raises(ValueError, match="diagnostic_level='tokens'"):
        basic_writer.add_report(temporal)


def test_cli_safety_fails_before_loading_adapter_or_creating_artifacts(tmp_path: Path):
    output = tmp_path / "must-not-exist"
    with pytest.raises(PermissionError):
        v31.main(
            [
                "realtime",
                "--checkpoint",
                "checkpoint",
                "--robot-config",
                "robot.json",
                "--adapter",
                "does.not.exist:factory",
                "--tests",
                "state_swap",
                "--counterfactual-mode",
                "active",
                "--execute-actions",
                "--require-operator-confirmation",
                "--operator-confirmation-token",
                v31.ACTIVE_ORACLE_CONFIRMATION,
                "--forced-subtask",
                "left",
                "--output-dir",
                str(output),
            ]
        )
    assert not output.exists()


def test_cli_defaults_static_config_to_v31():
    parser = v31._build_parser()  # noqa: SLF001
    args = parser.parse_args(
        [
            "offline",
            "--checkpoint",
            "checkpoint",
            "--episodes",
            "episodes.json",
            "--annotations",
            "annotations.json",
            "--adapter",
            "fake:factory",
            "--tests",
            "temporal",
            "--output-dir",
            "output",
        ]
    )
    assert args.config == "pi05_yam_mem_v31"

    request = v31.DiagnosticRunRequest(
        mode="offline",
        checkpoint="checkpoint",
        tests=("temporal",),
        output_dir=Path("output"),
        seed=0,
        diagnostic_level="basic",
        adapter="fake:factory",
    )
    assert request.config == "pi05_yam_mem_v31"


def test_duplicate_tests_and_incomplete_active_manifest_fail_closed():
    with pytest.raises(argparse.ArgumentTypeError, match="duplicates"):
        v31._parse_tests("oracle,oracle")  # noqa: SLF001

    manifest_kwargs = dataclasses.asdict(_manifest())
    manifest_kwargs.pop("active_forced_side")
    manifest_kwargs.update(
        {
            "hardware_mode": "realtime",
            "enabled_tests": ("oracle",),
            "counterfactual_mode": "active",
            "execute_actions": True,
            "require_operator_confirmation": True,
            "operator_confirmed": True,
        }
    )
    with pytest.raises(PermissionError, match="forced left/right"):
        v31.RunManifest(**manifest_kwargs)
    active = v31.RunManifest(**manifest_kwargs, active_forced_side="left")
    assert active.active_forced_side == "left"
