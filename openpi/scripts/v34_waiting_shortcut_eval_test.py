from __future__ import annotations

# The evaluator intentionally exposes script-private pure helpers for focused fail-closed tests.
# ruff: noqa: SLF001
import copy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v34_waiting_shortcut_eval as shortcut
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


_SIDE = {15: "left", 29: "right", 44: "left", 59: "right"}
_PROMPT = {
    15: "find the banana",
    29: "find the banana",
    44: "find the grey pepper box",
    59: "find the grey pepper box",
}


def _args(**overrides) -> shortcut.Args:
    values = {
        "checkpoint": Path("checkpoint/2500"),
        "dataset_root": Path("dataset"),
        "output_dir": Path("output"),
        "config": shortcut.causal.RUN5_CONFIG,
        "parameter_source": "raw",
    }
    values.update(overrides)
    return shortcut.Args(**values)


def _margin(side: str, truth_aligned: float) -> dict[str, float | bool]:
    raw = shortcut.causal._truth_sign(side) * truth_aligned
    return {
        "logp_left": -0.5 * raw,
        "logp_right": 0.5 * raw,
        "right_minus_left": raw,
        "truth_aligned": truth_aligned,
        "correct": truth_aligned > 0.0,
    }


def _records(channel: str = "none") -> list[dict[str, object]]:
    if channel not in ("none", "image", "state", "both"):
        raise ValueError(channel)
    rows = []
    for episode in shortcut.causal.EXPECTED_HELDOUT:
        side = _SIDE[episode]
        prompt = _PROMPT[episode]
        counterfactual = (
            "find the grey pepper box" if prompt == "find the banana" else "find the banana"
        )
        for frame in shortcut.causal.EXPECTED_WAIT_FRAMES[episode]:
            if channel == "none":
                aligned = dict.fromkeys(shortcut.CONDITIONS, 0.0)
            else:
                aligned = {
                    "reset_full": 0.60,
                    "reset_image_null": 0.0 if channel in ("image", "both") else 0.60,
                    "reset_state_null": 0.0 if channel in ("state", "both") else 0.60,
                    "reset_both_null": 0.0,
                    "reset_full_counterfactual_task": 0.60,
                    "reset_both_null_counterfactual_task": 0.0,
                    "reset_opposite_full_obs": -0.60,
                    "reset_opposite_image_only": -0.60 if channel in ("image", "both") else 0.60,
                    "reset_opposite_state_only": -0.60 if channel in ("state", "both") else 0.60,
                }
            margins = {
                condition: _margin(side, value) for condition, value in aligned.items()
            }
            effects = {
                "full_minus_image_null_truth_aligned": (
                    margins["reset_full"]["truth_aligned"]
                    - margins["reset_image_null"]["truth_aligned"]
                ),
                "full_minus_state_null_truth_aligned": (
                    margins["reset_full"]["truth_aligned"]
                    - margins["reset_state_null"]["truth_aligned"]
                ),
                "full_minus_both_null_truth_aligned": (
                    margins["reset_full"]["truth_aligned"]
                    - margins["reset_both_null"]["truth_aligned"]
                ),
                "full_minus_opposite_full_truth_aligned": (
                    margins["reset_full"]["truth_aligned"]
                    - margins["reset_opposite_full_obs"]["truth_aligned"]
                ),
                "full_minus_opposite_image_truth_aligned": (
                    margins["reset_full"]["truth_aligned"]
                    - margins["reset_opposite_image_only"]["truth_aligned"]
                ),
                "full_minus_opposite_state_truth_aligned": (
                    margins["reset_full"]["truth_aligned"]
                    - margins["reset_opposite_state_only"]["truth_aligned"]
                ),
            }
            rows.append(
                {
                    "episode": episode,
                    "prompt": prompt,
                    "counterfactual_prompt": counterfactual,
                    "truth_side": side,
                    "truth_sign": shortcut.causal._truth_sign(side),
                    "frame": frame,
                    "opposite_episode": shortcut.OPPOSITE_EPISODE[episode],
                    "opposite_side": "right" if side == "left" else "left",
                    "opposite_frame": shortcut.EXPECTED_OPPOSITE_FRAME[(episode, frame)],
                    "token_noise_sha256": hashlib.sha256(
                        f"{episode}-{frame}".encode()
                    ).hexdigest(),
                    "observation_identity": {
                        condition: {
                            "sha256": hashlib.sha256(
                                f"obs-{episode}-{frame}-{condition}".encode()
                            ).hexdigest(),
                            "fields": {},
                        }
                        for condition in shortcut.CONDITIONS
                    },
                    "margins": margins,
                    "effects": effects,
                }
            )
    return rows


def _observation(image: float, state: float, instruction: int, state_token: int):
    model = shortcut.causal._model
    return model.Observation(
        images={"base_0_rgb": jnp.full((1, 2, 2, 3), image, dtype=jnp.float32)},
        image_masks={"base_0_rgb": jnp.ones((1,), dtype=bool)},
        state=jnp.full((1, 14), state, dtype=jnp.float32),
        tokenized_prompt=jnp.asarray([[instruction, state_token, 0]], dtype=jnp.int32),
        tokenized_prompt_mask=jnp.asarray([[True, True, False]]),
        token_ar_mask=jnp.zeros((1, 3), dtype=jnp.int32),
        token_state_mask=jnp.asarray([[False, True, False]]),
    )


def _refresh_effects(record: dict[str, object]) -> None:
    margins = record["margins"]
    record["effects"] = {
        "full_minus_image_null_truth_aligned": (
            margins["reset_full"]["truth_aligned"]
            - margins["reset_image_null"]["truth_aligned"]
        ),
        "full_minus_state_null_truth_aligned": (
            margins["reset_full"]["truth_aligned"]
            - margins["reset_state_null"]["truth_aligned"]
        ),
        "full_minus_both_null_truth_aligned": (
            margins["reset_full"]["truth_aligned"]
            - margins["reset_both_null"]["truth_aligned"]
        ),
        "full_minus_opposite_full_truth_aligned": (
            margins["reset_full"]["truth_aligned"]
            - margins["reset_opposite_full_obs"]["truth_aligned"]
        ),
        "full_minus_opposite_image_truth_aligned": (
            margins["reset_full"]["truth_aligned"]
            - margins["reset_opposite_image_only"]["truth_aligned"]
        ),
        "full_minus_opposite_state_truth_aligned": (
            margins["reset_full"]["truth_aligned"]
            - margins["reset_opposite_state_only"]["truth_aligned"]
        ),
    }


def _valid_variants() -> dict[str, object]:
    full = _observation(1.0, 1.0, 10, 1)
    state_null = _observation(1.0, 0.0, 10, 0)
    full_cf = _observation(1.0, 1.0, 20, 1)
    state_null_cf = _observation(1.0, 0.0, 20, 0)
    opposite_full = _observation(2.0, 2.0, 10, 2)
    opposite_image = _observation(2.0, 1.0, 10, 1)
    opposite_state = _observation(1.0, 2.0, 10, 2)
    return {
        "reset_full": full,
        "reset_image_null": shortcut._null_images(full),
        "reset_state_null": state_null,
        "reset_both_null": shortcut._null_images(state_null),
        "reset_full_counterfactual_task": full_cf,
        "reset_both_null_counterfactual_task": shortcut._null_images(state_null_cf),
        "reset_opposite_full_obs": opposite_full,
        "reset_opposite_image_only": opposite_image,
        "reset_opposite_state_only": opposite_state,
    }


def test_args_are_pinned_to_run5_step2500_and_raw_or_ema() -> None:
    assert _args(parameter_source="raw").checkpoint.name == "2500"
    assert _args(parameter_source="ema").parameter_source == "ema"
    invalid = [
        ({"checkpoint": Path("checkpoint/2000")}, "checkpoint 2500"),
        ({"checkpoint": Path("checkpoint/latest")}, "checkpoint 2500"),
        ({"config": "pi05_yam_mem_v34"}, "pinned"),
        ({"parameter_source": "optimizer"}, "raw or ema"),
        ({"seed": 1}, "seed 0"),
    ]
    for overrides, match in invalid:
        with pytest.raises(ValueError, match=match):
            _args(**overrides)


def test_normalized_rank_pairing_is_exact_and_dataset_drift_fails() -> None:
    assert shortcut._validate_pairing() == shortcut.EXPECTED_OPPOSITE_FRAME
    assert shortcut._rank_paired_frames(shortcut.causal.EXPECTED_WAIT_FRAMES) == {
        (15, 540): 525,
        (29, 510): 540,
        (29, 525): 540,
        (29, 540): 540,
        (44, 525): 510,
        (44, 540): 525,
        (44, 555): 540,
        (44, 570): 555,
        (59, 510): 525,
        (59, 525): 540,
        (59, 540): 555,
        (59, 555): 570,
    }
    changed = dict(shortcut.causal.EXPECTED_WAIT_FRAMES)
    changed[15] = (525, 540)
    assert shortcut._rank_paired_frames(changed) != shortcut.EXPECTED_OPPOSITE_FRAME
    with pytest.raises(ValueError, match="exact heldout"):
        shortcut._rank_paired_frames({15: (540,)})


def test_canonical_state_is_equal_episode_weight_not_frame_weight() -> None:
    payloads = {}
    constants = {15: 0.0, 29: 2.0, 44: 4.0, 59: 10.0}
    for episode, constant in constants.items():
        payloads[episode] = {
            "by_frame": {
                frame: {"state": np.full((14,), constant, dtype=np.float32)}
                for frame in shortcut.causal.EXPECTED_WAIT_FRAMES[episode]
            }
        }
    canonical, provenance = shortcut._canonical_waiting_state(payloads)
    assert np.array_equal(canonical, np.full((14,), 4.0, dtype=np.float32))
    assert provenance["side_balance"] == {"left_episode_count": 2, "right_episode_count": 2}
    assert provenance["instruction_balance"] == {
        "find the banana": 2,
        "find the grey pepper box": 2,
    }
    assert provenance["canonical_identity"] == shortcut._array_identity(canonical)

    bad = copy.deepcopy(payloads)
    bad[15]["by_frame"][540]["state"] = np.full((13,), 0.0, dtype=np.float32)
    with pytest.raises(ValueError, match="invalid canonical state source"):
        shortcut._canonical_waiting_state(bad)
    bad = copy.deepcopy(payloads)
    bad[15]["by_frame"][540]["state"][0] = np.nan
    with pytest.raises(ValueError, match="invalid canonical state source"):
        shortcut._canonical_waiting_state(bad)


def test_image_null_is_exact_nonmutating_and_variant_contract_is_factorial() -> None:
    variants = _valid_variants()
    original = variants["reset_full"]
    original_image = np.asarray(original.images["base_0_rgb"]).copy()
    shortcut._validate_variant_contract(variants)
    assert np.array_equal(np.asarray(original.images["base_0_rgb"]), original_image)
    for condition in (
        "reset_image_null",
        "reset_both_null",
        "reset_both_null_counterfactual_task",
    ):
        observation = variants[condition]
        assert not np.any(np.asarray(observation.images["base_0_rgb"]))
        assert not np.any(np.asarray(observation.image_masks["base_0_rgb"]))

    bad = dict(variants)
    bad["reset_opposite_image_only"] = _observation(2.0, 2.0, 10, 2)
    with pytest.raises(RuntimeError, match="state/text context"):
        shortcut._validate_variant_contract(bad)
    bad = dict(variants)
    bad.pop("reset_state_null")
    with pytest.raises(RuntimeError, match="variant set mismatch"):
        shortcut._validate_variant_contract(bad)


def test_observation_and_tree_identities_bind_values_shapes_and_dtypes() -> None:
    first = _observation(1.0, 1.0, 10, 1)
    same = _observation(1.0, 1.0, 10, 1)
    changed = _observation(1.0, 1.1, 10, 1)
    assert shortcut._observation_identity(first) == shortcut._observation_identity(same)
    assert shortcut._observation_identity(first)["sha256"] != shortcut._observation_identity(
        changed
    )["sha256"]
    assert shortcut._tree_identity({"a": jnp.zeros((2,)), "b": jnp.ones((1,))})[
        "leaf_count"
    ] == 2
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        shortcut._array_identity(np.asarray([np.inf]))


def test_acceptance_localizes_image_or_state_only_from_real_swaps() -> None:
    image = shortcut._acceptance_summary(_records("image"))
    assert image["direct_reset_observation_side_shortcut_evidence_pass"] is True
    assert image["image_channel_localized"] is True
    assert image["state_channel_localized"] is False
    assert "images" in image["classification"]
    assert image["null_ablation_corroboration"][
        "removing_images_reduces_recipient_margin"
    ]["passes"] is True

    state = shortcut._acceptance_summary(_records("state"))
    assert state["direct_reset_observation_side_shortcut_evidence_pass"] is True
    assert state["image_channel_localized"] is False
    assert state["state_channel_localized"] is True
    assert "state" in state["classification"]

    both = shortcut._acceptance_summary(_records("both"))
    assert both["image_channel_localized"] is True
    assert both["state_channel_localized"] is True


def test_no_accuracy_or_swap_direction_means_no_shortcut_claim() -> None:
    report = shortcut._acceptance_summary(_records("none"))
    assert report["direct_reset_observation_side_shortcut_evidence_pass"] is False
    assert report["image_channel_localized"] is False
    assert report["state_channel_localized"] is False
    assert report["proof_claimed"] is False
    assert report["classification"].startswith("no primary")


def test_null_ablation_alone_cannot_trigger_primary_shortcut_gate() -> None:
    records = _records("none")
    for record in records:
        side = str(record["truth_side"])
        record["margins"]["reset_full"] = _margin(side, 0.60)
        record["margins"]["reset_image_null"] = _margin(side, 0.0)
        record["margins"]["reset_state_null"] = _margin(side, 0.0)
        record["margins"]["reset_both_null"] = _margin(side, 0.0)
        # Real opposite-side swaps do not move toward the donor.
        record["margins"]["reset_opposite_full_obs"] = _margin(side, 0.60)
        record["margins"]["reset_opposite_image_only"] = _margin(side, 0.60)
        record["margins"]["reset_opposite_state_only"] = _margin(side, 0.60)
        _refresh_effects(record)
    report = shortcut._acceptance_summary(records)
    assert report["null_ablation_corroboration"][
        "removing_both_reduces_recipient_margin"
    ]["passes"] is True
    assert report["direct_reset_observation_side_shortcut_evidence_pass"] is False


def test_both_null_same_prompt_invariant_fails_closed() -> None:
    records = _records("none")
    target = next(record for record in records if record["episode"] == 15)
    target["margins"]["reset_both_null"] = _margin("left", 0.1)
    _refresh_effects(target)
    with pytest.raises(RuntimeError, match="both-null same-instruction invariant failed"):
        shortcut._acceptance_summary(records)


def test_record_schema_rejects_wrong_pair_nonfinite_and_missing_condition() -> None:
    records = _records("none")
    records[0]["opposite_frame"] = 999
    with pytest.raises(ValueError, match="wrong opposite frame"):
        shortcut._validate_records(records)

    records = _records("none")
    records[0]["margins"].pop("reset_full")
    with pytest.raises(ValueError, match="condition schema mismatch"):
        shortcut._validate_records(records)

    records = _records("none")
    records[0]["margins"]["reset_full"]["logp_left"] = np.nan
    with pytest.raises(FloatingPointError, match="nonfinite margin"):
        shortcut._validate_records(records)


def test_balanced_gate_uses_equal_episode_macro_not_unequal_frame_count() -> None:
    records = _records("none")
    # Episode means are [1, 1, -1, -1], so the cell macro is exactly zero even though the
    # unequal per-episode frame counts would produce a different frame-weighted mean.
    values = {15: 1.0, 29: 1.0, 44: -1.0, 59: -1.0}
    gate = shortcut._balanced_gate(records, lambda row: values[int(row["episode"])], 0.0)
    assert gate["cell_macro_mean"] == 0.0
    assert gate["episode_directional_fraction"] == 0.5
    assert gate["passes"] is False


def test_counterfactual_task_sensitivity_is_reported_but_not_called_shortcut() -> None:
    records = _records("none")
    for record in records:
        side = str(record["truth_side"])
        # Use a fixed raw task effect.  Within each supplied both-null prompt, left and right
        # cells remain identical, preserving the input-elimination invariant.
        native_raw = 0.30 if record["prompt"] == "find the banana" else -0.20
        counter_raw = -0.20 if record["prompt"] == "find the banana" else 0.30
        sign = shortcut.causal._truth_sign(side)
        record["margins"]["reset_both_null"] = _margin(side, sign * native_raw)
        record["margins"]["reset_both_null_counterfactual_task"] = _margin(
            side, sign * counter_raw
        )
        _refresh_effects(record)
    report = shortcut._acceptance_summary(records)
    assert report["task_instruction_sensitivity_controls"][
        "both_null_task_sensitivity_abs"
    ]["passes"] is True
    assert report["direct_reset_observation_side_shortcut_evidence_pass"] is False


def test_compose_row_selects_only_requested_image_and_state_channels() -> None:
    row = {
        "image": b"top",
        "left_wrist_image": b"left",
        "right_wrist_image": b"right",
        "state": np.ones((14,), dtype=np.float32),
        "task_index": 7,
    }
    state = np.arange(14, dtype=np.float32)
    composed = shortcut._compose_row(row, state)
    assert set(composed) == {
        "image",
        "left_wrist_image",
        "right_wrist_image",
        "state",
    }
    assert composed["image"] == b"top"
    assert np.array_equal(composed["state"], state)
    state[0] = -1
    assert composed["state"][0] == 0


def test_pair_plans_must_be_same_instruction_and_opposite_side_contract() -> None:
    plans = {
        15: SimpleNamespace(prompt="find the banana", side="left"),
        29: SimpleNamespace(prompt="find the banana", side="right"),
    }
    assert plans[15].prompt == plans[29].prompt
    assert plans[15].side != plans[29].side
