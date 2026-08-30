from __future__ import annotations

# The evaluator intentionally exposes script-private pure helpers for fail-closed tests.
# ruff: noqa: SLF001
import copy
import hashlib
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v34_waiting_camera_shortcut_eval as camera
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


_SIDE = {15: "left", 29: "right", 44: "left", 59: "right"}
_PROMPT = {
    15: "find the banana",
    29: "find the banana",
    44: "find the grey pepper box",
    59: "find the grey pepper box",
}


def _args(**overrides) -> camera.Args:
    values = {
        "checkpoint": Path("checkpoint/2500"),
        "dataset_root": Path("dataset"),
        "output_dir": Path("output"),
        "parent_report": Path("parent/report.json"),
        "config": camera.causal.RUN5_CONFIG,
        "parameter_source": "raw",
    }
    values.update(overrides)
    return camera.Args(**values)


def _observation(levels: dict[str, float], *, state: float = 1.0, instruction: int = 10):
    model = camera.causal._model
    images = {
        camera.CAMERA_KEYS[name]: jnp.full((1, 2, 2, 3), levels[name], dtype=jnp.float32)
        for name in camera.CAMERAS
    }
    masks = {key: jnp.ones((1,), dtype=bool) for key in images}
    return model.Observation(
        images=images,
        image_masks=masks,
        state=jnp.full((1, 14), state, dtype=jnp.float32),
        tokenized_prompt=jnp.asarray([[instruction, 7, 0]], dtype=jnp.int32),
        tokenized_prompt_mask=jnp.asarray([[True, True, False]]),
        token_ar_mask=jnp.zeros((1, 3), dtype=jnp.int32),
        token_state_mask=jnp.asarray([[False, True, False]]),
    )


def _margin(side: str, truth_aligned: float) -> dict[str, float | bool]:
    raw = camera.causal._truth_sign(side) * truth_aligned
    return {
        "logp_left": -0.5 * raw,
        "logp_right": 0.5 * raw,
        "right_minus_left": raw,
        "truth_aligned": truth_aligned,
        "correct": truth_aligned > 0.0,
    }


def _condition_margins(side: str, contributions: dict[str, float]) -> dict[str, object]:
    native = 0.9
    return {
        condition: _margin(
            side,
            native - sum(contributions[name] for name in camera.SUBSET_BY_CONDITION[condition]),
        )
        for condition in camera.CONDITIONS
    }


def _records(contributions: dict[str, float]) -> list[dict[str, object]]:
    records = []
    for episode in camera.causal.EXPECTED_HELDOUT:
        side = _SIDE[episode]
        for frame in camera.causal.EXPECTED_WAIT_FRAMES[episode]:
            margins = _condition_margins(side, contributions)
            records.append(
                {
                    "episode": episode,
                    "prompt": _PROMPT[episode],
                    "truth_side": side,
                    "truth_sign": camera.causal._truth_sign(side),
                    "frame": frame,
                    "opposite_episode": camera.bundle.OPPOSITE_EPISODE[episode],
                    "opposite_side": "right" if side == "left" else "left",
                    "opposite_frame": camera.bundle.EXPECTED_OPPOSITE_FRAME[(episode, frame)],
                    "token_noise_sha256": hashlib.sha256(
                        f"noise-{episode}-{frame}".encode()
                    ).hexdigest(),
                    "observation_identity": {
                        condition: {
                            "sha256": hashlib.sha256(
                                f"obs-{episode}-{frame}-{condition}".encode()
                            ).hexdigest(),
                            "fields": {},
                        }
                        for condition in camera.CONDITIONS
                    },
                    "margins": margins,
                    "effects": camera._camera_effects(margins),
                    "historical_parent_comparison": {
                        "max_abs_margin_difference": 0.0,
                        "margin_absolute_differences": {
                            label: dict.fromkeys(
                                (
                                    "logp_left",
                                    "logp_right",
                                    "right_minus_left",
                                    "truth_aligned",
                                ),
                                0.0,
                            )
                            for label in ("native", "all_cameras")
                        },
                        "observation_identities_equal": {
                            "native": True,
                            "all_cameras": True,
                        },
                        "token_noise_identity_equal": True,
                        "identity_contract_passes": True,
                        "numeric_comparison_is_gating": False,
                        "non_gating_reason": "separate GPU process",
                    },
                    "within_run_endpoint_replay": {
                        "tolerance": camera.WITHIN_RUN_REPLAY_TOL,
                        "max_abs_margin_difference": 0.0,
                        "margin_absolute_differences": {
                            label: dict.fromkeys(
                                (
                                    "logp_left",
                                    "logp_right",
                                    "right_minus_left",
                                    "truth_aligned",
                                ),
                                0.0,
                            )
                            for label in ("reset_native", "reset_swap_all_cameras")
                        },
                        "passes": True,
                    },
                }
            )
    return records


def _parent_summary() -> dict[str, object]:
    return {
        "schema": camera.bundle.SCHEMA_VERSION,
        "classification": "reset-memory side shortcut localized to current images",
        "direct_reset_observation_side_shortcut_evidence_pass": True,
        "image_channel_localized": True,
        "parameter_source": "raw",
    }


def test_args_are_pinned_to_parent_run5_step2500_source_and_seed() -> None:
    assert _args(parameter_source="raw").checkpoint.name == "2500"
    assert _args(parameter_source="ema").parameter_source == "ema"
    invalid = [
        ({"checkpoint": Path("checkpoint/2000")}, "checkpoint 2500"),
        ({"config": "pi05_yam_mem_v34"}, "pinned"),
        ({"parameter_source": "optimizer"}, "raw or ema"),
        ({"parent_report": Path("parent/result.json")}, "report.json"),
        ({"seed": 1}, "seed 0"),
    ]
    for overrides, match in invalid:
        with pytest.raises(ValueError, match=match):
            _args(**overrides)


def test_factorial_is_exact_three_camera_power_set_in_stable_order() -> None:
    assert len(camera.SUBSETS) == 8
    assert len(set(camera.SUBSETS)) == 8
    assert camera.SUBSETS[0] == ()
    assert camera.SUBSETS[-1] == camera.CAMERAS
    assert camera._condition(("right_wrist", "top")) == "reset_swap_top_right_wrist"
    with pytest.raises(ValueError, match="unknown camera"):
        camera._canonical_subset(("side_camera",))


def test_variants_swap_only_selected_image_and_mask_and_keep_context_exact() -> None:
    recipient = _observation({"top": 1.0, "left_wrist": 2.0, "right_wrist": 3.0})
    donor = _observation(
        {"top": 11.0, "left_wrist": 12.0, "right_wrist": 13.0},
        state=9.0,
        instruction=10,
    )
    variants = camera._camera_variants(recipient, donor)
    assert tuple(variants) == camera.CONDITIONS
    for condition, variant in variants.items():
        camera.bundle._assert_same_state_context(recipient, variant)
        selected = camera.SUBSET_BY_CONDITION[condition]
        for name, key in camera.CAMERA_KEYS.items():
            expected = donor if name in selected else recipient
            assert camera._camera_equal(variant, expected, key)

    vacuous = _observation({"top": 1.0, "left_wrist": 12.0, "right_wrist": 13.0})
    with pytest.raises(RuntimeError, match="top intervention is vacuous"):
        camera._camera_variants(recipient, vacuous)

    wrong = recipient.replace(
        images={"unexpected": recipient.images["base_0_rgb"]},
        image_masks={"unexpected": recipient.image_masks["base_0_rgb"]},
    )
    with pytest.raises(RuntimeError, match="camera model contract changed"):
        camera._camera_variants(wrong, donor)

    invalid_masks = dict(donor.image_masks)
    invalid_masks["left_wrist_0_rgb"] = jnp.zeros((1,), dtype=bool)
    masked_donor = donor.replace(image_masks=invalid_masks)
    with pytest.raises(RuntimeError, match="left_wrist camera is not valid"):
        camera._camera_variants(recipient, masked_donor)


def test_shapley_effects_are_exact_for_single_camera_signal_and_sum_to_bundle() -> None:
    margins = _condition_margins(
        "left", {"top": 1.8, "left_wrist": 0.0, "right_wrist": 0.0}
    )
    effects = camera._camera_effects(margins)
    assert effects["bundle_drop_toward_donor"] == pytest.approx(1.8)
    assert effects["camera"]["top"]["shapley_drop_toward_donor"] == pytest.approx(1.8)
    assert effects["camera"]["left_wrist"]["shapley_drop_toward_donor"] == pytest.approx(0.0)
    assert effects["camera"]["right_wrist"]["shapley_drop_toward_donor"] == pytest.approx(0.0)
    assert effects["shapley_efficiency_residual"] == pytest.approx(0.0, abs=1e-12)
    assert len(effects["camera"]["top"]["edges"]) == 4


def test_acceptance_separates_factorial_contribution_sufficiency_and_rescue() -> None:
    top_only = camera._acceptance_summary(
        _records({"top": 1.8, "left_wrist": 0.0, "right_wrist": 0.0}),
        _parent_summary(),
    )
    assert top_only["bundle_shortcut_replicated"] is True
    assert top_only["historical_parent_endpoint_comparison"][
        "numeric_comparison_is_gating"
    ] is False
    assert top_only["historical_parent_endpoint_comparison"][
        "all_observation_and_noise_identities_match"
    ] is True
    assert top_only["within_run_endpoint_replay_invariant"]["all_12_cells_pass"] is True
    assert top_only["factorially_implicated_cameras"] == ["top"]
    assert top_only["camera"]["top"]["factorial_directional_contribution"] is True
    assert top_only["camera"]["top"]["isolated_swap_sufficient_for_donor_side"] is True
    assert (
        top_only["camera"]["top"]
        ["native_camera_rescues_recipient_side_against_other_donor_cameras"]
        is True
    )
    assert top_only["camera"]["left_wrist"]["factorial_directional_contribution"] is False

    # Three additive weak channels jointly flip the label.  Shapley attribution detects each
    # contribution while correctly refusing to call any single-camera swap sufficient or any
    # one native camera a rescue against the other two donor cameras.
    additive = camera._acceptance_summary(
        _records({"top": 0.6, "left_wrist": 0.6, "right_wrist": 0.6}),
        _parent_summary(),
    )
    assert additive["factorially_implicated_cameras"] == list(camera.CAMERAS)
    for name in camera.CAMERAS:
        result = additive["camera"][name]
        assert result["factorial_directional_contribution"] is True
        assert result["isolated_swap_sufficient_for_donor_side"] is False
        assert result["native_camera_rescues_recipient_side_against_other_donor_cameras"] is False


def test_record_validation_rejects_schema_nonfinite_effect_and_replay_corruption() -> None:
    records = _records({"top": 1.8, "left_wrist": 0.0, "right_wrist": 0.0})
    camera._validate_records(records)

    bad = copy.deepcopy(records)
    bad[0]["margins"].pop("reset_swap_top")
    with pytest.raises(ValueError, match="margin condition mismatch"):
        camera._validate_records(bad)

    bad = copy.deepcopy(records)
    bad[0]["margins"]["reset_swap_top"]["logp_left"] = np.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        camera._validate_records(bad)

    bad = copy.deepcopy(records)
    bad[0]["effects"]["camera"]["top"]["shapley_drop_toward_donor"] += 0.1
    with pytest.raises(ValueError, match="effect inconsistency"):
        camera._validate_records(bad)

    bad = copy.deepcopy(records)
    bad[0]["historical_parent_comparison"]["identity_contract_passes"] = False
    with pytest.raises(ValueError, match="identity pass flag is inconsistent"):
        camera._validate_records(bad)

    bad = copy.deepcopy(records)
    bad[0]["historical_parent_comparison"]["margin_absolute_differences"]["native"][
        "logp_left"
    ] = 0.2
    with pytest.raises(ValueError, match="maximum mismatch"):
        camera._validate_records(bad)

    bad = copy.deepcopy(records)
    bad[0]["within_run_endpoint_replay"]["passes"] = False
    with pytest.raises(ValueError, match="pass flag is inconsistent"):
        camera._validate_records(bad)


def test_parent_comparison_binds_inputs_but_does_not_gate_cross_process_numerics() -> None:
    side = "left"
    parent = {
        "margins": {
            "reset_full": _margin(side, 0.9),
            "reset_opposite_image_only": _margin(side, -0.9),
        },
        "observation_identity": {
            "reset_full": {"sha256": "a" * 64, "fields": {}},
            "reset_opposite_image_only": {"sha256": "b" * 64, "fields": {}},
        },
        "token_noise_sha256": "c" * 64,
    }
    margins = {
        "reset_native": _margin(side, 0.9),
        "reset_swap_all_cameras": _margin(side, -0.9),
    }
    identities = {
        "reset_native": parent["observation_identity"]["reset_full"],
        "reset_swap_all_cameras": parent["observation_identity"][
            "reset_opposite_image_only"
        ],
    }
    comparison = camera._historical_parent_comparison(parent, "c" * 64, identities, margins)
    assert comparison["identity_contract_passes"] is True
    assert comparison["numeric_comparison_is_gating"] is False

    wrong_identity = copy.deepcopy(identities)
    wrong_identity["reset_native"] = {"sha256": "d" * 64, "fields": {}}
    assert not camera._historical_parent_comparison(
        parent, "c" * 64, wrong_identity, margins
    )["identity_contract_passes"]
    assert not camera._historical_parent_comparison(
        parent, "e" * 64, identities, margins
    )["identity_contract_passes"]
    changed = copy.deepcopy(margins)
    changed["reset_native"]["logp_left"] += 0.5
    changed_comparison = camera._historical_parent_comparison(
        parent, "c" * 64, identities, changed
    )
    assert changed_comparison["identity_contract_passes"] is True
    assert changed_comparison["max_abs_margin_difference"] == pytest.approx(0.5)


def test_within_run_endpoint_replay_keeps_strict_numeric_invariant() -> None:
    margins = {
        "reset_native": _margin("left", 0.9),
        "reset_swap_all_cameras": _margin("left", -0.9),
    }
    exact = camera._within_run_endpoint_replay(margins, copy.deepcopy(margins))
    assert exact["passes"] is True
    assert exact["max_abs_margin_difference"] == 0.0

    changed = copy.deepcopy(margins)
    changed["reset_native"]["logp_left"] += 1e-3
    replay = camera._within_run_endpoint_replay(margins, changed)
    assert replay["passes"] is False
    assert replay["max_abs_margin_difference"] == pytest.approx(1e-3)


def test_completed_parent_reader_requires_exact_checksum_and_no_incomplete(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    report_path = parent / "report.json"
    encoded = b'{"schema":"test"}\n'
    report_path.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    (parent / "COMPLETE").write_text(f"{digest}  report.json\n", encoding="utf-8")
    report, identity = camera._read_completed_parent(report_path)
    assert report == {"schema": "test"}
    assert identity["report"]["sha256"] == digest

    (parent / "INCOMPLETE.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="parent audit is incomplete"):
        camera._read_completed_parent(report_path)
    (parent / "INCOMPLETE.json").unlink()
    (parent / "COMPLETE").write_text(f"{'0' * 64}  report.json\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="COMPLETE marker mismatch"):
        camera._read_completed_parent(report_path)


def test_output_reservation_and_completion_are_fail_closed(tmp_path: Path) -> None:
    failed = tmp_path / "failed"
    incomplete = camera._reserve_output(failed, {"schema": "test", "status": "reserved"})
    assert incomplete.is_file()
    with pytest.raises(FileExistsError):
        camera._reserve_output(failed, {})
    with pytest.raises(FloatingPointError):
        camera._finalize_report(failed, {"bad": float("nan")})
    assert incomplete.is_file()
    assert not (failed / "COMPLETE").exists()
    assert not (failed / "report.json").exists()

    completed = tmp_path / "completed"
    camera._reserve_output(completed, {"schema": "test", "status": "reserved"})
    identities = camera._finalize_report(completed, {"schema": "test", "value": 1})
    assert not (completed / "INCOMPLETE.json").exists()
    assert (completed / "COMPLETE").is_file()
    assert identities["report"] == camera.causal._file_identity(completed / "report.json")
    report, _identity = camera._read_completed_parent(completed / "report.json")
    assert report == {"schema": "test", "value": 1}
