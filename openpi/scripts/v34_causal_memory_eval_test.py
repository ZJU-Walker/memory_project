from __future__ import annotations

# The evaluator intentionally exposes script-private pure helpers for focused fail-closed tests.
# ruff: noqa: SLF001
import copy
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v34_causal_memory_eval as causal
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


_SIDE_BY_EPISODE = {15: "left", 29: "right", 44: "left", 59: "right"}
_PROMPT_BY_EPISODE = {
    15: "find the banana",
    29: "find the banana",
    44: "find the grey pepper box",
    59: "find the grey pepper box",
}


def _args(**overrides) -> causal.Args:
    values = {
        "checkpoint": Path("checkpoint"),
        "dataset_root": Path("dataset"),
        "output_dir": Path("output"),
        "config": causal.RUN5_CONFIG,
        "parameter_source": "raw",
    }
    values.update(overrides)
    return causal.Args(**values)


def _run5_semantics() -> tuple[SimpleNamespace, SimpleNamespace]:
    model = SimpleNamespace(
        memory=SimpleNamespace(eta_scale=0.0, blank_initial_output=True),
        memory_architecture="v32_layer8_dual_query",
    )
    data = SimpleNamespace(
        heldout_episodes=causal.EXPECTED_HELDOUT,
        memory_stride_frames=15,
    )
    return model, data


def _plan(
    episode: int,
    prompt: str,
    side: str,
    vector: tuple[int, int, int, int, int],
) -> SimpleNamespace:
    evidence_start, evidence_end, memory_start, memory_end, length = vector
    return SimpleNamespace(
        episode=episode,
        prompt=prompt,
        side=side,
        evidence=(evidence_start, evidence_end),
        memory=(memory_start, memory_end),
        length=length,
    )


def _donor_plans() -> tuple[list[SimpleNamespace], list[SimpleNamespace]]:
    banana = (100, 130, 500, 550, 875)
    pepper_left = (200, 230, 520, 580, 863)
    pepper_right = (300, 330, 500, 560, 740)
    recipients = [
        _plan(15, "find the banana", "left", banana),
        _plan(29, "find the banana", "right", banana),
        _plan(44, "find the grey pepper box", "left", pepper_left),
        _plan(59, "find the grey pepper box", "right", pepper_right),
    ]
    donors = [
        _plan(13, "find the banana", "left", banana),
        _plan(18, "find the banana", "right", banana),
        _plan(31, "find the grey pepper box", "left", pepper_left),
        _plan(46, "find the grey pepper box", "right", pepper_left),
        _plan(42, "find the grey pepper box", "left", pepper_right),
        _plan(54, "find the grey pepper box", "right", pepper_right),
        # A closer numeric episode id must not beat the matching prompt/side/vector constraints.
        _plan(1, "wrong prompt", "left", banana),
        _plan(2, "wrong prompt", "right", banana),
    ]
    return recipients + donors, recipients


def _margin(side: str, truth_aligned: float) -> dict[str, float | bool]:
    difference = causal._truth_sign(side) * truth_aligned
    return {
        "logp_left": -0.5 * difference,
        "logp_right": 0.5 * difference,
        "right_minus_left": difference,
        "truth_aligned": truth_aligned,
        "correct": truth_aligned > 0.0,
    }


def _condition_margins(side: str) -> dict[str, dict[str, float | bool]]:
    aligned = {
        "normal": 0.60,
        "normal_zero_read": 0.0,
        "reset": 0.0,
        "never_write": 0.0,
        "pre_evidence_frozen": 0.05,
        "frozen_after_evidence": 0.50,
        "dynamics_only": 0.45,
        "same_side_frozen_swap": 0.50,
        "opposite_frozen_swap": -0.50,
        "heldout_opposite_frozen_swap": -0.50,
    }
    assert set(aligned) == set(causal.CONDITIONS)
    return {condition: _margin(side, value) for condition, value in aligned.items()}


def _action_conditions(side: str) -> dict[str, dict[str, float]]:
    aligned = {
        "normal": 0.30,
        "normal_zero_read": 0.0,
        "reset": 0.0,
        "never_write": 0.0,
        "pre_evidence_frozen": 0.05,
        "frozen_after_evidence": 0.25,
        "dynamics_only": 0.23,
        "same_side_frozen_swap": 0.25,
        "opposite_frozen_swap": -0.25,
        "heldout_opposite_frozen_swap": -0.25,
    }
    assert set(aligned) == set(causal.CONDITIONS)
    sign = causal._truth_sign(side)
    return {
        condition: {
            "right_minus_left": sign * value,
            "truth_aligned_score": value,
        }
        for condition, value in aligned.items()
    }


def _valid_records(*, full: bool) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for episode in causal.EXPECTED_HELDOUT:
        side = _SIDE_BY_EPISODE[episode]
        for frame in causal.EXPECTED_WAIT_FRAMES[episode]:
            record: dict[str, object] = {
                "episode": episode,
                "prompt": _PROMPT_BY_EPISODE[episode],
                "truth_side": side,
                "truth_sign": causal._truth_sign(side),
                "frame": frame,
                "writes_before_current_frame": frame // 15,
                "token_noise_sha256": hashlib.sha256(
                    f"token-{episode}-{frame}".encode()
                ).hexdigest(),
                "same_side_donor_episode": causal.EXPECTED_DONORS["same_side"][episode],
                "same_side_donor_side": side,
                "opposite_side_donor_episode": causal.EXPECTED_DONORS["opposite_side"][episode],
                "opposite_side_donor_side": "right" if side == "left" else "left",
                "heldout_opposite_donor_episode": causal.EXPECTED_HELDOUT_OPPOSITE_DONORS[
                    episode
                ],
                "heldout_opposite_donor_side": "right" if side == "left" else "left",
                "margins": _condition_margins(side),
            }
            if full and frame in causal.EXPECTED_ACTION_FRAMES[episode]:
                demonstrated_score = causal.EXPECTED_DEMONSTRATED_ACTION_SCORES[(episode, frame)]
                record["ground_truth_action"] = {
                    "right_minus_left": demonstrated_score,
                    "truth_aligned_score": causal._truth_aligned(side, demonstrated_score),
                }
                record["actions"] = [
                    {
                        "noise_replicate": replicate,
                        "noise_sha256": hashlib.sha256(
                            f"action-{episode}-{frame}-{replicate}".encode()
                        ).hexdigest(),
                        "conditions": _action_conditions(side),
                        "reset_invariant_max_abs_robot_action": 0.0,
                    }
                    for replicate in causal.ACTION_NOISE_REPLICATES
                ]
            records.append(record)
    return records


def _set_margin(record: dict[str, object], condition: str, value: float) -> None:
    side = str(record["truth_side"])
    margins = record["margins"]
    assert isinstance(margins, dict)
    margins[condition] = _margin(side, value)


def test_args_accept_only_the_preregistered_run5_surface() -> None:
    assert _args(parameter_source="raw").parameter_source == "raw"
    assert _args(parameter_source="ema", mode="full").parameter_source == "ema"

    invalid = [
        ({"config": "pi05_yam_mem_v34"}, "pinned"),
        ({"parameter_source": "optimizer"}, "raw or ema"),
        ({"mode": "action"}, "token or full"),
        ({"episodes": (15, 29, 44)}, "exact heldouts"),
        ({"episodes": (59, 44, 29, 15)}, "exact heldouts"),
        ({"seed": 1}, "seed 0"),
        ({"action_noise_replicates": (0, 1)}, "exact action-noise replicates"),
        ({"action_noise_replicates": (0, 1, 999)}, "exact action-noise replicates"),
        ({"action_noise_replicates": (0, 0, 2)}, "exact action-noise replicates"),
        ({"action_denoise_steps": 9}, "action-denoise-steps 10"),
        ({"action_denoise_steps": 11}, "action-denoise-steps 10"),
    ]
    for overrides, match in invalid:
        with pytest.raises(ValueError, match=match):
            _args(**overrides)


def test_parameter_source_paths_are_explicit_and_fail_closed() -> None:
    checkpoint = Path("/tmp/checkpoint")
    assert causal._parameter_restore_path(checkpoint, "raw") == checkpoint / "train_state"
    assert causal._parameter_restore_path(checkpoint, "ema") == checkpoint / "params"
    with pytest.raises(ValueError, match="unknown parameter source"):
        causal._parameter_restore_path(checkpoint, "optimizer")


def test_checkpoint_origin_and_internal_train_step_are_bound_to_run5(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    live = repo / "checkpoints/pi05_yam_mem_v34_run5_eta0/v34_run5_eta0/500"
    archive = repo / "diagnostic_checkpoints/v34_run5_eta0_read_only/1000"
    wrong_run = repo / "diagnostic_checkpoints/v34_run4_read_only/500"
    pilot_log = repo / causal.PILOT_LOG
    pilot_log.parent.mkdir(parents=True)
    pilot_log.write_text(
        "\n".join(
            (
                "Training config: name=pi05_yam_mem_v34_run5_eta0 "
                "exp_name=v34_run5_eta0 eta_scale=0.0",
                f"root_directory={live.parent}",
                f"Finished saving checkpoint (finalized tmp dir) to `{live}`",
                f"Finished saving checkpoint (finalized tmp dir) to `{live.parent / '1000'}`",
            )
        ),
        encoding="utf-8",
    )
    checkpoint_metadata = {"init_timestamp_nsecs": 654321, "commit_timestamp_nsecs": 654399}
    archive.mkdir(parents=True)
    control_files = (
        "_CHECKPOINT_METADATA",
        "params/_METADATA",
        "params/_sharding",
        "params/manifest.ocdbt",
        "train_state/_METADATA",
        "train_state/_sharding",
        "train_state/manifest.ocdbt",
    )
    for relative in control_files:
        path = archive / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        contents = (
            json.dumps(checkpoint_metadata)
            if relative == "_CHECKPOINT_METADATA"
            else f"control:{relative}"
        )
        path.write_text(contents, encoding="utf-8")
    archive_manifest = {
        "schema": "openpi-checkpoint-snapshot-v1",
        "source": str(live.parent / "1000"),
        "destination": str(archive),
        "step": "1000",
        "source_metadata": {
            "device": 1,
            "inode": 2,
            "size": 3,
            "mtime_ns": 4,
            "ctime_ns": 5,
        },
        "checkpoint_metadata": checkpoint_metadata,
        "commit_timestamp_nsecs": checkpoint_metadata["commit_timestamp_nsecs"],
        "files": [
            {
                "path": relative,
                "size": (archive / relative).stat().st_size,
                "sha256": causal._sha256_file(archive / relative),
            }
            for relative in control_files
        ],
    }
    manifest_path = archive.parent / "1000.manifest.json"
    manifest_path.write_text(json.dumps(archive_manifest), encoding="utf-8")
    live_info = {"init_timestamp_nsecs": 123456, "commit_timestamp_nsecs": 123499}
    archive_info = {
        "init_timestamp_nsecs": 654321,
        "commit_timestamp_nsecs": 654399,
    }

    live_identity = causal._validate_checkpoint_origin(
        live.resolve(), repo.resolve(), live_info
    )
    archive_identity = causal._validate_checkpoint_origin(
        archive.resolve(), repo.resolve(), archive_info
    )
    assert live_identity["origin_kind"] == "live run5 checkpoint root"
    assert archive_identity["origin_kind"] == "run5-labelled diagnostic archive root"
    with pytest.raises(ValueError, match="exact run5 live root|run5-labelled"):
        causal._validate_checkpoint_origin(wrong_run.resolve(), repo.resolve(), live_info)
    with pytest.raises(ValueError, match="timestamps differ"):
        causal._validate_checkpoint_origin(archive.resolve(), repo.resolve(), live_info)

    archive_manifest["step"] = 1000
    manifest_path.write_text(json.dumps(archive_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong step"):
        causal._validate_checkpoint_origin(archive.resolve(), repo.resolve(), archive_info)
    archive_manifest["step"] = "1000"
    archive_manifest["source"] = str(live.parent / "999")
    manifest_path.write_text(json.dumps(archive_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong source"):
        causal._validate_checkpoint_origin(archive.resolve(), repo.resolve(), archive_info)

    intermediary = repo / causal.PILOT_HARDLINK_ARCHIVE / "1000"
    for relative in control_files:
        path = intermediary / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((archive / relative).read_bytes())
    intermediary_stat = intermediary.stat()
    archive_manifest["source"] = str(intermediary)
    archive_manifest["source_metadata"] = {
        "device": intermediary_stat.st_dev,
        "inode": intermediary_stat.st_ino,
        "size": intermediary_stat.st_size,
        "mtime_ns": intermediary_stat.st_mtime_ns,
        "ctime_ns": intermediary_stat.st_ctime_ns,
    }
    manifest_path.write_text(json.dumps(archive_manifest), encoding="utf-8")
    chained_identity = causal._validate_checkpoint_origin(
        archive.resolve(), repo.resolve(), archive_info
    )
    assert (
        chained_identity["archive_snapshot_manifest"]["source_kind"]
        == "exact run5 read-only hardlink intermediary"
    )
    (intermediary / "params/_METADATA").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="intermediary size mismatch"):
        causal._validate_checkpoint_origin(archive.resolve(), repo.resolve(), archive_info)

    assert causal._validate_checkpoint_train_step(live, 501) == {
        "checkpoint_manager_step_label": 500,
        "internal_train_state_step": 501,
    }
    with pytest.raises(ValueError, match="label/internal TrainState step mismatch"):
        causal._validate_checkpoint_train_step(live, 500)
    with pytest.raises(ValueError, match="not numeric"):
        causal._validate_checkpoint_train_step(live.with_name("final"), 501)

    transforms = causal._raw_parameter_transforms()
    assert set(transforms) == {r"params/(.*)"}


def test_train_step_reader_opens_only_the_scalar_ocdbt_leaf(tmp_path: Path) -> None:
    source = tmp_path / "train_state"
    source.mkdir()
    spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "ocdbt",
            "base": {"driver": "file", "path": str(source)},
            "path": "step",
        },
    }
    step_store = causal.ts.open(
        spec,
        create=True,
        dtype=causal.ts.int32,
        shape=(),
    ).result()
    step_store.write(np.asarray(501, dtype=np.int32)).result()
    assert causal._restore_train_step(source) == 501

    wrong_shape = tmp_path / "wrong_shape"
    wrong_shape.mkdir()
    wrong_store = causal.ts.open(
        {
            **spec,
            "kvstore": {**spec["kvstore"], "base": {"driver": "file", "path": str(wrong_shape)}},
        },
        create=True,
        dtype=causal.ts.int32,
        shape=(1,),
    ).result()
    wrong_store.write(np.asarray([501], dtype=np.int32)).result()
    with pytest.raises(ValueError, match="not scalar"):
        causal._restore_train_step(wrong_shape)


def test_static_run5_semantics_pin_eta_blank_output_architecture_and_data() -> None:
    model, data = _run5_semantics()
    causal._validate_run5_semantics(
        causal.RUN5_CONFIG, model, data, causal.EXPECTED_EMA_DECAY
    )

    cases = [
        ("wrong", model, data, "config="),
        (
            causal.RUN5_CONFIG,
            SimpleNamespace(
                memory=SimpleNamespace(eta_scale=0.1, blank_initial_output=True),
                memory_architecture="v32_layer8_dual_query",
            ),
            data,
            "eta_scale",
        ),
        (
            causal.RUN5_CONFIG,
            SimpleNamespace(
                memory=SimpleNamespace(eta_scale=0.0, blank_initial_output=False),
                memory_architecture="v32_layer8_dual_query",
            ),
            data,
            "blank_initial_output",
        ),
        (
            causal.RUN5_CONFIG,
            SimpleNamespace(
                memory=SimpleNamespace(eta_scale=0.0, blank_initial_output=True),
                memory_architecture="other",
            ),
            data,
            "memory_architecture",
        ),
        (
            causal.RUN5_CONFIG,
            model,
            SimpleNamespace(heldout_episodes=(15, 29), memory_stride_frames=15),
            "heldout_episodes",
        ),
        (
            causal.RUN5_CONFIG,
            model,
            SimpleNamespace(heldout_episodes=causal.EXPECTED_HELDOUT, memory_stride_frames=10),
            "memory_stride_frames",
        ),
    ]
    for config_name, bad_model, bad_data, match in cases:
        with pytest.raises(ValueError, match=match):
            causal._validate_run5_semantics(
                config_name, bad_model, bad_data, causal.EXPECTED_EMA_DECAY
            )
    with pytest.raises(ValueError, match="ema_decay"):
        causal._validate_run5_semantics(causal.RUN5_CONFIG, model, data, None)
    with pytest.raises(ValueError, match="ema_decay"):
        causal._validate_run5_semantics(causal.RUN5_CONFIG, model, data, 0.99)


def test_donor_maps_are_exact_nonheldout_and_same_prompt() -> None:
    all_plans, recipients = _donor_plans()
    result = causal._build_donor_maps(all_plans, recipients, stride=15)

    assert result["same_side"] == {15: 13, 29: 18, 44: 31, 59: 54}
    assert result["opposite_side"] == {15: 18, 29: 13, 44: 46, 59: 42}
    assert causal.EXPECTED_HELDOUT_OPPOSITE_DONORS == {15: 29, 29: 15, 44: 59, 59: 44}
    heldout = set(causal.EXPECTED_HELDOUT)
    recipients_by_episode = {plan.episode: plan for plan in recipients}
    for episode in causal.EXPECTED_HELDOUT:
        for kind in ("same_side", "opposite_side"):
            donor_episode = result[kind][episode]
            selection = result["selection"][episode][kind]
            assert donor_episode not in heldout
            assert selection["training_exposed"] is True
            assert selection["prompt"] == recipients_by_episode[episode].prompt
            if kind == "same_side":
                assert selection["side"] == recipients_by_episode[episode].side
            else:
                assert selection["side"] != recipients_by_episode[episode].side


def test_donor_maps_reject_wrong_prompt_only_and_duplicate_ids() -> None:
    all_plans, recipients = _donor_plans()
    no_banana_left = [
        plan
        for plan in all_plans
        if not (
            plan.episode not in causal.EXPECTED_HELDOUT
            and plan.prompt == "find the banana"
            and plan.side == "left"
        )
    ]
    with pytest.raises(ValueError, match="no nonheldout"):
        causal._build_donor_maps(no_banana_left, recipients, stride=15)

    with pytest.raises(ValueError, match="duplicate episode"):
        causal._build_donor_maps([*all_plans, all_plans[-1]], recipients, stride=15)
    with pytest.raises(ValueError, match="stride 15"):
        causal._build_donor_maps(all_plans, recipients, stride=10)


def test_truth_alignment_and_side_token_prefix_mask() -> None:
    assert causal._truth_sign("left") == -1
    assert causal._truth_sign("right") == 1
    assert causal._truth_aligned("left", -0.3) == pytest.approx(0.3)
    assert causal._truth_aligned("right", 0.3) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="left/right"):
        causal._truth_sign("center")

    left, right, index = causal._side_prefix_masks([10, 20, 30, 40], [10, 20, 31, 40])
    assert index == 2
    assert left == right == [True, True, True, False]
    with pytest.raises(ValueError, match="equal token lengths"):
        causal._side_prefix_masks([1], [1, 2])
    with pytest.raises(ValueError, match="exactly one token"):
        causal._side_prefix_masks([1, 2], [1, 2])
    with pytest.raises(ValueError, match="exactly one token"):
        causal._side_prefix_masks([1, 2, 3], [9, 2, 8])


def test_action_motion_excludes_both_grippers() -> None:
    state = np.zeros(14, dtype=np.float64)
    gripper_only = np.zeros((5, 14), dtype=np.float64)
    gripper_only[:, 6] = 1000.0
    gripper_only[:, 13] = -1000.0
    metric = causal._action_motion(gripper_only, state)
    assert metric["right_minus_left"] == 0.0
    assert metric["left_arm_trajectory_rms"] == 0.0
    assert metric["right_arm_trajectory_rms"] == 0.0
    assert metric["larger_motion_arm"] == "tie"

    right_arm = gripper_only.copy()
    right_arm[:, 7:13] = 0.2
    metric = causal._action_motion(right_arm, state)
    assert metric["right_minus_left"] == pytest.approx(0.2)
    assert causal._truth_aligned("right", float(metric["right_minus_left"])) == pytest.approx(0.2)

    left_arm = gripper_only.copy()
    left_arm[:, :6] = 0.3
    metric = causal._action_motion(left_arm, state)
    assert metric["right_minus_left"] == pytest.approx(-0.3)
    assert causal._truth_aligned("left", float(metric["right_minus_left"])) == pytest.approx(0.3)


def test_demonstrated_action_calibration_is_exact_and_truth_material() -> None:
    expected = {
        (15, 540): -0.3441188637378251,
        (29, 525): 0.09268175840937752,
        (29, 540): 0.2828443328407406,
        (44, 555): -0.1986549176675482,
        (44, 570): -0.4509864556453942,
        (59, 525): 0.07312115284674275,
        (59, 540): 0.25985896347967274,
        (59, 555): 0.4813865098501708,
    }
    assert expected == causal.EXPECTED_DEMONSTRATED_ACTION_SCORES
    for (episode, frame), score in expected.items():
        causal._validate_demonstrated_action_score(episode, frame, score)
        aligned = causal._truth_aligned(_SIDE_BY_EPISODE[episode], score)
        assert aligned > causal.ACTION_SCORE_EPS
        with pytest.raises(ValueError, match="calibration changed"):
            causal._validate_demonstrated_action_score(episode, frame, score + 1e-8)
    with pytest.raises(ValueError, match="not pre-registered"):
        causal._validate_demonstrated_action_score(15, 555, 0.2)


def test_schedule_helpers_preserve_prewrite_grid_and_action_registration() -> None:
    unordered = [{"frame_index": frame} for frame in (31, 30, 0, 15, 45)]
    scheduled = causal._scheduled_grid_rows(
        unordered,
        15,
        after_exclusive=0,
        through_inclusive=31,
    )
    assert [row["frame_index"] for row in scheduled] == [15, 30]
    with pytest.raises(ValueError, match="duplicate frame_index"):
        causal._scheduled_grid_rows(
            [*unordered, {"frame_index": 15}],
            15,
            through_inclusive=45,
        )

    target = 540
    dense_rows = [{"frame_index": frame} for frame in range(target + 1)]
    before_target = causal._scheduled_grid_rows(
        dense_rows,
        15,
        through_inclusive=target - 1,
    )
    assert len(before_target) == target // 15 == 36
    assert before_target[-1]["frame_index"] == 525
    assert all(row["frame_index"] != target for row in before_target)

    execute_start = {15: 560, 29: 558, 44: 590, 59: 560}
    for episode in causal.EXPECTED_HELDOUT:
        side = _SIDE_BY_EPISODE[episode]
        waiting_label = f"wait; target bin is {side}"
        execute_label = f"open {side} bin"
        tasks = {0: waiting_label, 1: execute_label}
        last = max(causal.EXPECTED_WAIT_FRAMES[episode]) + 49
        rows = [
            {
                "frame_index": frame,
                "task_index": int(frame >= execute_start[episode]),
            }
            for frame in range(last + 1)
        ]
        plan = SimpleNamespace(episode=episode, side=side)
        assert causal._action_eligible_frames(
            plan,
            rows,
            tasks,
            stride=15,
            horizon=50,
        ) == causal.EXPECTED_ACTION_FRAMES[episode]


def test_strict_json_rejects_nonfinite_and_unsupported_values() -> None:
    converted = causal._strict_json({1: (np.int64(2), np.float32(3.5), True, None)})
    assert converted == {"1": [2, 3.5, True, None]}
    for value in (float("nan"), float("inf"), np.float64("-inf")):
        with pytest.raises(FloatingPointError, match="nonfinite"):
            causal._strict_json({"value": value})
    with pytest.raises(TypeError, match=r"cannot serialize .*Path"):
        causal._strict_json(Path("not-json"))


def test_complete_parameter_tree_identity_hashes_every_byte() -> None:
    first = {
        "z": np.asarray([1, 2, 3], dtype=np.int16),
        "nested": {"a": np.asarray([[4.0, 5.0]], dtype=np.float32), "none": None},
    }
    reordered = {"nested": first["nested"], "z": first["z"]}
    identity = causal._parameter_tree_identity(first)
    assert identity == causal._parameter_tree_identity(reordered)
    assert identity["bytes_hashed"] == first["z"].nbytes + first["nested"]["a"].nbytes
    assert identity["array_leaves_hashed"] == 2

    changed = copy.deepcopy(first)
    changed["z"][-1] = 9
    assert causal._parameter_tree_identity(changed)["sha256"] != identity["sha256"]
    with pytest.raises(TypeError, match="object dtype"):
        causal._parameter_tree_identity({"bad": np.asarray([object()], dtype=object)})
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        causal._parameter_tree_identity({"bad": np.asarray([1.0, np.nan])})
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        causal._parameter_tree_identity(
            {"bad_bfloat16": np.asarray([1.0, np.inf], dtype=causal.jnp.bfloat16)}
        )
    with pytest.raises(TypeError, match="nonnumeric dtype"):
        causal._parameter_tree_identity({"bad_text": np.asarray(["not-a-parameter"])})


def test_tokenizer_assets_are_pinned_before_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paligemma = tmp_path / "paligemma_tokenizer.model"
    paligemma.write_bytes(b"pinned-paligemma")
    fast = tmp_path / causal.EXPECTED_FAST_TOKENIZER_COMMIT
    fast.mkdir()
    (fast / "tokenizer.json").write_bytes(b"pinned-fast")
    monkeypatch.setattr(
        causal,
        "EXPECTED_PALIGEMMA_TOKENIZER_SHA256",
        hashlib.sha256(paligemma.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        causal,
        "EXPECTED_FAST_TOKENIZER_FILE_SHA256",
        {"tokenizer.json": hashlib.sha256(b"pinned-fast").hexdigest()},
    )

    identity = causal._validate_tokenizer_asset_files(paligemma, fast)
    assert identity["all_expected_hashes_match"] is True
    (fast / "tokenizer.json").write_bytes(b"changed-fast")
    with pytest.raises(ValueError, match="FAST tokenizer snapshot"):
        causal._validate_tokenizer_asset_files(paligemma, fast)
    paligemma.write_bytes(b"changed-paligemma")
    with pytest.raises(ValueError, match="PaliGemma tokenizer hash mismatch"):
        causal._validate_tokenizer_asset_files(paligemma, fast)


def test_launch_hash_parser_and_directory_identity(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    launch = repo / causal.LAUNCH_PROVENANCE
    launch.mkdir(parents=True)
    lines = []
    for index, relative in enumerate(causal.LAUNCH_BOUND_FILES):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"source-{index}\n"
        path.write_text(payload, encoding="utf-8")
        lines.append(f"{hashlib.sha256(payload.encode()).hexdigest()}  ./{relative}")
    (launch / "key_files.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for index, relative in enumerate(causal.LAUNCH_SNAPSHOT_BOUND_FILES):
        path = repo / relative
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"snapshot-source-{index}\n", encoding="utf-8")
    with tarfile.open(launch / "source_snapshot.tar.gz", "w:gz") as archive:
        for relative in causal.LAUNCH_SNAPSHOT_BOUND_FILES:
            archive.add(repo / relative, arcname=relative)
    manifest_entries = []
    for name in ("key_files.sha256", "source_snapshot.tar.gz"):
        path = launch / name
        manifest_entries.append(
            f"{causal._sha256_file(path)}  {path.relative_to(repo)}"
        )
    (launch / "manifest.sha256").write_text(
        "\n".join(manifest_entries) + "\n", encoding="utf-8"
    )

    provenance = causal._validate_launch_provenance(repo)
    assert provenance["all_launch_bound_hashes_match"] is True
    assert set(provenance["launch_manifest"]["verified_artifact_hashes"]) == {
        str(causal.LAUNCH_PROVENANCE / "key_files.sha256"),
        str(causal.LAUNCH_PROVENANCE / "source_snapshot.tar.gz"),
    }
    assert set(provenance["launch_bound_current_hashes"]) == set(causal.LAUNCH_BOUND_FILES)
    assert set(provenance["snapshot_bound_current_hashes"]) == set(
        causal.LAUNCH_SNAPSHOT_BOUND_FILES
    )
    identity = causal._directory_identity(launch)
    assert identity["files"]["key_files.sha256"]["sha256"] == causal._sha256_file(
        launch / "key_files.sha256"
    )

    snapshot_path = launch / "source_snapshot.tar.gz"
    original_snapshot = snapshot_path.read_bytes()
    with tarfile.open(snapshot_path, "w:gz") as archive:
        for relative in causal.LAUNCH_SNAPSHOT_BOUND_FILES:
            archive.add(repo / relative, arcname=relative)
        duplicate = causal.LAUNCH_SNAPSHOT_BOUND_FILES[0]
        archive.add(repo / duplicate, arcname=duplicate)
    manifest_entries[-1] = (
        f"{causal._sha256_file(snapshot_path)}  {snapshot_path.relative_to(repo)}"
    )
    (launch / "manifest.sha256").write_text(
        "\n".join(manifest_entries) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="duplicate file member"):
        causal._validate_launch_provenance(repo)

    snapshot_path.write_bytes(original_snapshot)
    manifest_entries[-1] = (
        f"{causal._sha256_file(snapshot_path)}  {snapshot_path.relative_to(repo)}"
    )
    (launch / "manifest.sha256").write_text(
        "\n".join(manifest_entries) + "\n", encoding="utf-8"
    )
    (launch / "key_files.sha256").write_text("malformed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="launch manifest hash mismatch"):
        causal._validate_launch_provenance(repo)


def test_record_schema_requires_exact_wait_grid() -> None:
    valid = _valid_records(full=False)
    causal._validate_record_schema(valid, "token")

    missing = copy.deepcopy(valid)
    missing.pop()
    with pytest.raises(ValueError, match="record grid"):
        causal._validate_record_schema(missing, "token")

    duplicate = copy.deepcopy(valid)
    duplicate.append(copy.deepcopy(duplicate[0]))
    with pytest.raises(ValueError, match="duplicate episode/frame"):
        causal._validate_record_schema(duplicate, "token")


def test_record_schema_rejects_internally_plausible_but_wrong_episode_metadata() -> None:
    records = _valid_records(full=False)
    corrupted = copy.deepcopy(records)
    corrupted[0]["opposite_side_donor_episode"] = causal.EXPECTED_DONORS["same_side"][15]
    with pytest.raises(ValueError, match="wrong opposite-side donor metadata"):
        causal._validate_record_schema(corrupted, "token")

    corrupted_heldout = copy.deepcopy(records)
    corrupted_heldout[0]["heldout_opposite_donor_episode"] = 59
    with pytest.raises(ValueError, match="wrong heldout-opposite donor metadata"):
        causal._validate_record_schema(corrupted_heldout, "token")


def test_full_record_schema_requires_exact_action_frame_replicate_product() -> None:
    valid = _valid_records(full=True)
    causal._validate_record_schema(valid, "full")

    expected = {
        (episode, frame, replicate)
        for episode in causal.EXPECTED_HELDOUT
        for frame in causal.EXPECTED_ACTION_FRAMES[episode]
        for replicate in causal.ACTION_NOISE_REPLICATES
    }
    actual = {
        (int(record["episode"]), int(record["frame"]), int(sample["noise_replicate"]))
        for record in valid
        for sample in record.get("actions", [])
    }
    assert actual == expected

    missing_sample = copy.deepcopy(valid)
    next(record for record in missing_sample if "actions" in record)["actions"].pop()
    with pytest.raises(ValueError, match="action.*replicate|replicate.*action"):
        causal._validate_record_schema(missing_sample, "full")

    action_on_wrong_frame = copy.deepcopy(valid)
    source = next(record for record in action_on_wrong_frame if "actions" in record)
    target = next(record for record in action_on_wrong_frame if "actions" not in record)
    target["actions"] = copy.deepcopy(source["actions"])
    with pytest.raises(ValueError, match="action"):
        causal._validate_record_schema(action_on_wrong_frame, "full")


def test_full_record_schema_rejects_bogus_replicate_999() -> None:
    records = _valid_records(full=True)
    first_action_record = next(record for record in records if "actions" in record)
    first_action_record["actions"][0]["noise_replicate"] = 999
    with pytest.raises(ValueError, match="replicate"):
        causal._acceptance_summary(records, "full")


def test_synthetic_valid_token_and_full_acceptance() -> None:
    token = causal._acceptance_summary(_valid_records(full=False), "token")
    assert token["reset_zero_never_token_invariant_pass"] is True
    assert token["causal_token_pass"] is True
    assert token["causal_action_pass"] is None
    assert token["source_behavioral_influence_evidence_pass"] is None
    assert token["proof_claimed"] is False

    full = causal._acceptance_summary(_valid_records(full=True), "full")
    assert full["causal_token_pass"] is True
    assert full["causal_action_pass"] is True
    assert full["source_behavioral_influence_evidence_pass"] is True
    assert full["proof_claimed"] is False


def test_sensitivity_controls_are_reported_but_do_not_redefine_primary_pass() -> None:
    records = _valid_records(full=True)
    for record in records:
        _set_margin(record, "dynamics_only", -0.30)
        _set_margin(record, "heldout_opposite_frozen_swap", 0.30)
        side = str(record["truth_side"])
        sign = causal._truth_sign(side)
        for sample in record.get("actions", []):
            sample["conditions"]["dynamics_only"] = {
                "right_minus_left": sign * -0.30,
                "truth_aligned_score": -0.30,
            }
            sample["conditions"]["heldout_opposite_frozen_swap"] = {
                "right_minus_left": sign * 0.30,
                "truth_aligned_score": 0.30,
            }

    summary = causal._acceptance_summary(records, "full")
    assert summary["causal_token_pass"] is True
    assert summary["causal_action_pass"] is True
    assert summary["source_behavioral_influence_evidence_pass"] is True
    for episode in causal.EXPECTED_HELDOUT:
        token = summary["token_episode_checks"][episode]
        action = summary["action"]["episodes"][episode]
        assert token["passes"] is True
        assert token["sensitivity_controls_pass"] is False
        assert action["passes"] is True
        assert action["sensitivity_controls_pass"] is False


def test_token_outlier_cannot_hide_low_frame_coverage() -> None:
    records = _valid_records(full=False)
    episode_records = [record for record in records if record["episode"] == 59]
    assert len(episode_records) == 4
    for record in episode_records:
        for condition, value in {
            "normal": -0.10,
            "pre_evidence_frozen": 0.0,
            "frozen_after_evidence": -0.10,
            "dynamics_only": -0.10,
            "same_side_frozen_swap": -0.10,
            "opposite_frozen_swap": 0.10,
        }.items():
            _set_margin(record, condition, value)
    outlier = episode_records[0]
    for condition, value in {
        "normal": 4.0,
        "pre_evidence_frozen": 0.0,
        "frozen_after_evidence": 4.0,
        "dynamics_only": 4.0,
        "same_side_frozen_swap": 4.0,
        "opposite_frozen_swap": -4.0,
    }.items():
        _set_margin(outlier, condition, value)

    summary = causal._acceptance_summary(records, "token")
    assert summary["token_episode_checks"][59]["mean_truth_aligned_margin"]["normal"] > 0.0
    criteria = summary["token_episode_checks"][59]["primary_criteria"]
    assert criteria["normal_favors_truth"]["directional_fraction"] < 0.75
    assert summary["token_episode_checks"][59]["passes"] is False
    assert summary["causal_token_pass"] is False


def test_reset_zero_never_invariant_failure_cannot_pass() -> None:
    records = _valid_records(full=False)
    margins = records[0]["margins"]
    assert isinstance(margins, dict)
    margins["reset"] = _margin(str(records[0]["truth_side"]), 1e-3)
    summary = causal._acceptance_summary(records, "token")
    assert summary["reset_zero_never_token_invariant_pass"] is False
    assert summary["causal_token_pass"] is False
