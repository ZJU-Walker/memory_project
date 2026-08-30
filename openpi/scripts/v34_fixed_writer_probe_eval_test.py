from __future__ import annotations

# The evaluator intentionally exposes script-private pure helpers for fail-closed tests.
# ruff: noqa: SLF001
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pytest

_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS))
try:
    import v34_fixed_writer_probe_eval as probe
finally:
    sys.path.remove(str(_SCRIPTS))


def _args(**overrides) -> probe.Args:
    values = {
        "checkpoint": Path("checkpoint/2500"),
        "dataset_root": Path("dataset"),
        "output_dir": Path("output/2500"),
    }
    values.update(overrides)
    return probe.Args(**values)


def _spec(
    episode: int,
    prompt: str,
    side: str,
    *,
    heldout: bool,
    evidence: tuple[int, ...] = (4, 5),
    approach: tuple[int, ...] = (2, 3),
) -> probe.EpisodeSpec:
    return probe.EpisodeSpec(
        episode=episode,
        prompt=prompt,
        side=side,
        label=probe.SIDE_TO_LABEL[side],
        length=10,
        parquet=Path(f"episode_{episode:06d}.parquet"),
        evidence_frames=evidence,
        approach_frames=approach,
        heldout=heldout,
    )


def _all_specs() -> list[probe.EpisodeSpec]:
    banana_left = {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15}
    specs = []
    for episode in range(60):
        prompt = probe.PROMPTS[0] if episode < 30 else probe.PROMPTS[1]
        side = (
            ("left" if episode in banana_left else "right")
            if episode < 30
            else ("left" if episode < 45 else "right")
        )
        specs.append(
            _spec(
                episode,
                prompt,
                side,
                heldout=episode in set(probe.HELDOUT_EPISODES),
            )
        )
    return specs


def test_args_pin_protocol_and_separate_raw_ema_namespaces() -> None:
    assert _args().parameter_source == "raw"
    assert _args().artifact_dir.name == "raw"
    assert _args(parameter_source="ema").artifact_dir.name == "ema"
    assert _args(smoke_only=True).smoke_only
    invalid = [
        ({"checkpoint": Path("checkpoint/latest")}, "numeric"),
        ({"config": "pi05_yam_mem_v34"}, "pinned"),
        ({"parameter_source": "both"}, "raw or ema"),
        ({"batch_size": 0}, "batch-size"),
        ({"batch_size": 65}, "batch-size"),
        ({"seed": 0}, "seed"),
        ({"null_repeats": 2}, "null-repeats"),
    ]
    for overrides, match in invalid:
        with pytest.raises(ValueError, match=match):
            _args(**overrides)


def test_online_pool_matches_the_exact_pi0_writer_head_formula() -> None:
    tokens = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5) - 7.0
    expected = jnp.mean(jnp.asarray(tokens).astype(jnp.float32), axis=1)
    expected = expected * jax.lax.rsqrt(
        jnp.sum(jnp.square(expected), axis=-1, keepdims=True) + 1e-12
    )
    actual = probe._online_pool(tokens)
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1.0, atol=1e-6)
    states = probe._l2_rows(np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(states[0], [0.6, 0.8], atol=1e-7)
    np.testing.assert_array_equal(states[1], [0.0, 0.0])
def test_episode_feature_is_mean_of_normalized_frames_without_second_norm() -> None:
    specs = [
        _spec(0, probe.PROMPTS[0], "left", heldout=False, evidence=(4, 5), approach=(2, 3)),
        _spec(1, probe.PROMPTS[0], "right", heldout=False, evidence=(4, 5), approach=(2, 3)),
    ]
    frame_episode = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    frame_phase = np.asarray([0, 0, 1, 1, 0, 0, 1, 1])
    feature = np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    frame_arrays = {
        "frame_episode": frame_episode,
        "frame_phase": frame_phase,
        "frame_writer": feature,
        "frame_value": feature,
        "frame_key": feature,
        "frame_state": feature,
    }
    output = probe._episode_feature_arrays(specs, frame_arrays)
    np.testing.assert_array_equal(output["episode_writer_evidence"][0], [0.5, 0.5])
    assert np.linalg.norm(output["episode_writer_evidence"][0]) == pytest.approx(np.sqrt(0.5))


def test_exact_design_builds_14_balanced_folds_and_never_fits_heldout() -> None:
    specs = _all_specs()
    train, folds = probe._validate_design(specs)
    assert len(train) == 56
    assert len(folds) == 14
    assert not set(train) & set(probe.HELDOUT_EPISODES)
    seen = []
    for fold in folds:
        assert len(fold.train_indices) == 52
        assert len(fold.test_indices) == 4
        assert not set(fold.train_indices) & set(probe.HELDOUT_EPISODES)
        assert not set(fold.test_indices) & set(probe.HELDOUT_EPISODES)
        cells = {specs[index].cell for index in fold.test_indices}
        assert len(cells) == 4
        seen.extend(fold.test_indices)
    assert sorted(seen) == sorted(train)

    broken = list(specs)
    broken[15] = _spec(15, probe.PROMPTS[0], "left", heldout=False)
    with pytest.raises(ValueError, match="heldout set changed"):
        probe._validate_design(broken)


def test_task_rows_select_all_unshifted_evidence_and_equal_immediate_window() -> None:
    tasks = [0, 0, 0, 0, 4, 4, 4, 3, 3, 3]
    table = pa.table(
        {
            "frame_index": list(range(10)),
            "episode_index": [9] * 10,
            "task_index": tasks,
        }
    )
    evidence, approach = probe._validate_task_rows(
        table,
        episode=9,
        expected_length=10,
        task_names=tuple(str(index) for index in range(7)),
    )
    assert evidence == (4, 5, 6)
    assert approach == (1, 2, 3)

    broken = pa.table(
        {
            "frame_index": list(range(10)),
            "episode_index": [9] * 10,
            "task_index": [0, 0, 4, 4, 0, 4, 3, 3, 3, 3],
        }
    )
    with pytest.raises(ValueError, match="contiguous"):
        probe._validate_task_rows(
            broken,
            episode=9,
            expected_length=10,
            task_names=tuple(str(index) for index in range(7)),
        )


def test_probe_is_zero_initialized_fresh_deterministic_and_metrics_are_exact() -> None:
    features = np.asarray([[-3.0], [-2.0], [-1.0], [-0.5], [0.5], [1.0], [2.0], [3.0]])
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    first = probe._fit_probe(features, labels)
    second = probe._fit_probe(features, labels)
    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.weights[0] > 0
    scores = first.scores(features)
    metrics = probe._binary_metrics(labels, scores)
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["auc"] == 1.0
    assert metrics["truth_margin"] > 0
    assert metrics["log_loss"] < math_log_2()


def math_log_2() -> float:
    return float(np.log(2.0))


def test_oof_applies_the_evidence_trained_head_and_standardizer_to_approach() -> None:
    specs = _all_specs()
    _train, folds = probe._validate_design(specs)
    labels = np.asarray([spec.label for spec in specs])
    evidence = (2.0 * labels - 1.0)[:, None]
    approach = np.zeros_like(evidence)
    result = probe._oof_probe(evidence, approach, labels, folds)
    assert result["evidence_metrics"]["accuracy"] == 1.0
    assert result["evidence_metrics"]["auc"] == 1.0
    assert result["approach_same_head_metrics"]["accuracy"] == 0.5
    assert result["paired_evidence_minus_approach"]["mean"] > 0
    assert all(record["zero_initialized"] for record in result["folds"])


def test_within_prompt_null_is_deterministic_and_preserves_each_prompt_balance() -> None:
    specs = _all_specs()
    train, _folds = probe._validate_design(specs)
    labels = np.asarray([spec.label for spec in specs])
    prompts = [spec.prompt for spec in specs]
    first = probe._within_prompt_shuffles(labels, prompts, train)
    second = probe._within_prompt_shuffles(labels, prompts, train)
    assert len(first) == probe.DEFAULT_NULL_REPEATS
    for left, right in zip(first, second, strict=True):
        np.testing.assert_array_equal(left, right)
        for prompt in probe.PROMPTS:
            indices = [index for index in train if prompts[index] == prompt]
            assert sorted(left[indices].tolist()) == sorted(labels[indices].tolist())
    assert any(not np.array_equal(labels[list(train)], item[list(train)]) for item in first)


def test_checkpoint_origin_accepts_general_numeric_live_steps_and_manifest_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    live = repo / "checkpoints/pi05_yam_mem_v34_run5_eta0/v34_run5_eta0/9500"
    live.mkdir(parents=True)
    info = {"step_label": 9500}
    result = probe._validate_checkpoint_origin(live, repo, info)
    assert result["origin_kind"] == "exact live run5 checkpoint root"

    archive = repo / "diagnostic_checkpoints/v34_run5_eta0_copies/2500"
    archive.mkdir(parents=True)
    calls = []

    def fake_manifest(checkpoint, live_step, checkpoint_info):
        calls.append((checkpoint, live_step, checkpoint_info))
        return {"validated": True}

    monkeypatch.setattr(probe.causal, "_validate_archive_snapshot_manifest", fake_manifest)
    archived = probe._validate_checkpoint_origin(archive, repo, {"step_label": 2500})
    assert archived["archive_snapshot_manifest"] == {"validated": True}
    assert calls[0][1].name == "2500"

    outside = repo / "other/2500"
    with pytest.raises(ValueError, match="exact run5 live root"):
        probe._validate_checkpoint_origin(outside, repo, {"step_label": 2500})


def test_memory_independence_smoke_checks_writer_key_value_and_discards_state() -> None:
    class Runtime:
        @staticmethod
        def _write(state, write_tokens):
            del write_tokens
            return state + 1.0, {}

        @staticmethod
        def _interface(observation, state):
            del observation, state
            base = jnp.ones((2, 3, 4), dtype=jnp.float32)
            return {"write_tokens": base, "write_keys": base * 2, "write_values": base * 3}

        @staticmethod
        def state_max_abs_difference(left, right):
            return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))

    fresh = Runtime._interface(None, jnp.zeros((2, 1)))
    result = probe._memory_independence_smoke(
        Runtime(), None, fresh, jnp.zeros((2, 1), dtype=jnp.float32)
    )
    assert result["scratch_written_state_max_abs_change_from_m0"] == 1.0
    assert result["fresh_vs_nonempty_max_abs_difference"] == {
        "writer": 0.0,
        "key": 0.0,
        "value": 0.0,
    }
    assert result["scratch_written_state_discarded"]


def test_runtime_jits_bound_methods_but_calls_online_head_module_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Memory:
        def write(self):
            return None

    class Model:
        def __init__(self) -> None:
            self.memory = Memory()
            self.ladder_writer_head = object()

        def v32_memory_interface_step(self):
            return None

    seen = []

    def fake_module_jit(method):
        seen.append(method)
        return ("jitted", method)

    monkeypatch.setattr(probe.causal.nnx_utils, "module_jit", fake_module_jit)
    model = Model()
    interface, write, head = probe._bind_runtime_callables(model)
    assert interface[0] == "jitted"
    assert write[0] == "jitted"
    assert len(seen) == 2
    assert head is model.ladder_writer_head


def test_fixed_smoke_subset_is_balanced_and_uses_one_matched_pair() -> None:
    specs = _all_specs()
    smoke = [spec for spec in specs if spec.episode in probe.SMOKE_EPISODES]
    assert {spec.cell for spec in smoke} == {
        (prompt, side) for prompt in probe.PROMPTS for side in probe.SIDE_TO_LABEL
    }
    for spec in smoke:
        selected = probe._selected_frames(spec, smoke_only=True)
        assert len(selected) == 2
        assert selected[0][0] == "approach"
        assert selected[1][0] == "evidence"
        offset = len(spec.evidence_frames) // 2
        assert selected[0][1] == spec.approach_frames[offset]
        assert selected[1][1] == spec.evidence_frames[offset]


def test_atomic_output_never_overwrites_and_complete_binds_both_artifacts(tmp_path: Path) -> None:
    destination = tmp_path / "raw"
    with probe._AtomicOutput(destination) as output:
        features = output.write_npz({"x": np.arange(3, dtype=np.int64)})
        report = output.write_report({"ok": True})
        output.complete({"features.npz": features, "report.json": report})
    assert destination.is_dir()
    assert (destination / "features.npz").is_file()
    assert (destination / "report.json").is_file()
    complete = (destination / "COMPLETE").read_text()
    assert features["sha256"] in complete
    assert report["sha256"] in complete
    with pytest.raises(FileExistsError, match="overwrite"), probe._AtomicOutput(destination):
        pass

    failed = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="deliberate"), probe._AtomicOutput(failed):
        raise RuntimeError("deliberate")
    assert not failed.exists()
