"""Read-only reset-memory audit for current-frame side shortcuts in run5 checkpoint 2500.

This diagnostic answers a narrower question than ``v34_causal_memory_eval.py``: can the model
predict the waiting-side label from the *current* waiting observation when episodic memory is
freshly reset?  Every score is evaluated at the exact pre-registered held-out waiting frames,
with the same reset ``M0`` and explicit RNG/noise tensor across all interventions.

The four requested channel views are:

* ``reset_full`` -- native images, state, and task instruction;
* ``reset_image_null`` -- all camera pixels zero and every camera mask false;
* ``reset_state_null`` -- state replaced by one fixed, cell-balanced waiting reset pose;
* ``reset_both_null`` -- both interventions together.

Null inputs can be out of distribution, so they are corroborating controls only.  Primary
localization uses in-distribution, same-instruction observations from the paired held-out episode
with the opposite side: full-observation, image-only, and state-only swaps.  The remaining two
conditions replace the task instruction with the other instruction seen during training while
holding either the native observation or both-null observation fixed.  These are task-sensitivity
controls, not side-shortcut evidence by themselves.

The checkpoint is never modified and no memory writes are committed.  Output uses the same
fail-closed reservation, checkpoint/source/config validation, raw-vs-EMA restore semantics, and
offline tokenizer resolution as the causal-memory audit.  A crash leaves ``INCOMPLETE.json``;
only a fully validated report receives a checksummed ``COMPLETE`` marker.
"""

# ruff: noqa: SLF001, I001 - this evidence script intentionally reuses audited private helpers.
from __future__ import annotations

import pyarrow.parquet as pq  # noqa: F401 - must precede the OpenPI/JAX import stack.

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

import v34_causal_memory_eval as causal


SCHEMA_VERSION = "openpi.v34.waiting_shortcut_eval.v1"
CHECKPOINT_STEP = 2500
INTERNAL_TRAIN_STEP = 2501
CONDITIONS = (
    "reset_full",
    "reset_image_null",
    "reset_state_null",
    "reset_both_null",
    "reset_full_counterfactual_task",
    "reset_both_null_counterfactual_task",
    "reset_opposite_full_obs",
    "reset_opposite_image_only",
    "reset_opposite_state_only",
)
OPPOSITE_EPISODE = causal.EXPECTED_HELDOUT_OPPOSITE_DONORS
EXPECTED_OPPOSITE_FRAME = {
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
EFFECT_EPS = 0.05
MIN_DIRECTIONAL_COVERAGE = 0.75
INVARIANT_TOL = 1e-5


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint: Path
    dataset_root: Path
    output_dir: Path
    config: str
    parameter_source: Literal["raw", "ema"]
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", Path(self.checkpoint).expanduser().resolve())
        object.__setattr__(self, "dataset_root", Path(self.dataset_root).expanduser().resolve())
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        if self.config != causal.RUN5_CONFIG:
            raise ValueError(
                f"waiting shortcut audit is pinned to --config {causal.RUN5_CONFIG!r}; "
                f"got {self.config!r}"
            )
        if self.parameter_source not in ("raw", "ema"):
            raise ValueError("--parameter-source must be raw or ema")
        if self.checkpoint.name != str(CHECKPOINT_STEP):
            raise ValueError(
                f"waiting shortcut audit is pinned to checkpoint {CHECKPOINT_STEP}; "
                f"got {self.checkpoint.name!r}"
            )
        if self.seed != 0:
            raise ValueError("the pre-registered waiting shortcut audit requires --seed 0")


def _parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parameter-source", choices=("raw", "ema"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    return Args(**vars(parser.parse_args(argv)))


def _array_identity(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("cannot identify an object-dtype array")
    try:
        finite = np.isfinite(array)
    except TypeError as exc:
        raise TypeError(f"cannot identify nonnumeric array dtype {array.dtype}") from exc
    if not np.all(finite):
        raise FloatingPointError("array identity input contains NaN/Inf")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(b"\0")
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(memoryview(contiguous.view(np.uint8).reshape(-1)))
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": digest.hexdigest(),
    }


def _tree_identity(tree: Any) -> dict[str, Any]:
    leaves = jax.tree.leaves(tree)
    if not leaves:
        raise ValueError("tree identity requires at least one leaf")
    digest = hashlib.sha256()
    identities = []
    for index, leaf in enumerate(leaves):
        identity = _array_identity(leaf)
        digest.update(str(index).encode())
        digest.update(b"\0")
        digest.update(identity["sha256"].encode())
        identities.append(identity)
    return {"sha256": digest.hexdigest(), "leaf_count": len(leaves), "leaves": identities}


def _rank_paired_frames(wait_frames: dict[int, tuple[int, ...]]) -> dict[tuple[int, int], int]:
    """Pair opposite-side frames by normalized rank, with a singleton placed at the midpoint."""
    if set(wait_frames) != set(causal.EXPECTED_HELDOUT):
        raise ValueError("paired-frame construction requires the exact heldout episodes")
    result: dict[tuple[int, int], int] = {}
    for episode in causal.EXPECTED_HELDOUT:
        frames = tuple(wait_frames[episode])
        donor_episode = OPPOSITE_EPISODE[episode]
        donors = tuple(wait_frames[donor_episode])
        if not frames or not donors:
            raise ValueError("waiting frame lists must be nonempty")
        for index, frame in enumerate(frames):
            position = index / (len(frames) - 1) if len(frames) > 1 else 0.5
            donor_index = min(
                range(len(donors)),
                key=lambda candidate: (
                    abs((candidate / (len(donors) - 1) if len(donors) > 1 else 0.5) - position),
                    donors[candidate],
                ),
            )
            result[(episode, frame)] = donors[donor_index]
    return result


def _validate_pairing() -> dict[tuple[int, int], int]:
    derived = _rank_paired_frames(causal.EXPECTED_WAIT_FRAMES)
    if derived != EXPECTED_OPPOSITE_FRAME:
        raise ValueError(
            "opposite-side waiting-frame pairing changed: "
            f"expected {EXPECTED_OPPOSITE_FRAME}, got {derived}"
        )
    return derived


def _canonical_waiting_state(payloads: dict[int, dict[str, Any]]) -> tuple[np.ndarray, dict[str, Any]]:
    """Return an equal-episode-weight canonical waiting reset pose.

    Unequal numbers of waiting frames must not weight one prompt/side cell more heavily.  We
    first mean within each of the four heldout episodes, then mean those four vectors equally.
    The heldout grid has two instructions x two sides exactly once each.
    """
    episode_means: dict[int, np.ndarray] = {}
    source_hashes: dict[int, dict[str, Any]] = {}
    shape: tuple[int, ...] | None = None
    for episode in causal.EXPECTED_HELDOUT:
        payload = payloads[episode]
        frames = causal.EXPECTED_WAIT_FRAMES[episode]
        states = []
        for frame in frames:
            if frame not in payload["by_frame"]:
                raise ValueError(f"canonical state source is missing ep{episode} frame {frame}")
            state = np.asarray(payload["by_frame"][frame]["state"], dtype=np.float64)
            if shape is None:
                shape = state.shape
            if state.shape != shape or state.shape != (14,) or not np.all(np.isfinite(state)):
                raise ValueError(f"invalid canonical state source ep{episode} f{frame}: {state.shape}")
            states.append(state)
        stack = np.stack(states)
        episode_means[episode] = np.mean(stack, axis=0)
        source_hashes[episode] = _array_identity(stack)
    canonical = np.mean(np.stack([episode_means[episode] for episode in causal.EXPECTED_HELDOUT]), axis=0)
    if canonical.shape != (14,) or not np.all(np.isfinite(canonical)):
        raise ValueError("canonical waiting state is invalid")
    return canonical.astype(np.float32), {
        "construction": (
            "arithmetic mean within each exact heldout waiting grid, then equal arithmetic mean "
            "over the four prompt-by-side cells"
        ),
        "episode_order": list(causal.EXPECTED_HELDOUT),
        "source_frame_stack_identity": source_hashes,
        "per_episode_mean": {
            episode: episode_means[episode].tolist() for episode in causal.EXPECTED_HELDOUT
        },
        "canonical_value": canonical.tolist(),
        "canonical_identity": _array_identity(canonical.astype(np.float32)),
        "side_balance": {"left_episode_count": 2, "right_episode_count": 2},
        "instruction_balance": {"find the banana": 2, "find the grey pepper box": 2},
    }


def _compose_row(image_row: dict[str, Any], state: Any) -> dict[str, Any]:
    return {
        "image": image_row["image"],
        "left_wrist_image": image_row["left_wrist_image"],
        "right_wrist_image": image_row["right_wrist_image"],
        "state": np.array(state, dtype=np.float32, copy=True),
    }


def _null_images(observation: Any) -> Any:
    if set(observation.images) != set(observation.image_masks) or not observation.images:
        raise ValueError("observation image/mask keys are missing or inconsistent")
    images = {name: jnp.zeros_like(value) for name, value in observation.images.items()}
    masks = {name: jnp.zeros_like(observation.image_masks[name], dtype=bool) for name in images}
    return observation.replace(images=images, image_masks=masks)


def _observation_identity(observation: Any) -> dict[str, Any]:
    fields = {
        "images": {name: _array_identity(value) for name, value in observation.images.items()},
        "image_masks": {
            name: _array_identity(value.astype(jnp.int8))
            for name, value in observation.image_masks.items()
        },
        "state": _array_identity(observation.state),
        "tokenized_prompt": _array_identity(observation.tokenized_prompt),
        "tokenized_prompt_mask": _array_identity(observation.tokenized_prompt_mask.astype(jnp.int8)),
        "token_ar_mask": _array_identity(observation.token_ar_mask),
        "token_state_mask": _array_identity(observation.token_state_mask.astype(jnp.int8)),
    }
    digest = hashlib.sha256(
        json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"sha256": digest, "fields": fields}


def _arrays_equal(left: Any, right: Any) -> bool:
    return np.array_equal(np.asarray(left), np.asarray(right))


def _assert_same_images(left: Any, right: Any) -> None:
    if tuple(left.images) != tuple(right.images) or tuple(left.image_masks) != tuple(right.image_masks):
        raise RuntimeError("variant camera ordering/keys differ")
    for name in left.images:
        if not _arrays_equal(left.images[name], right.images[name]):
            raise RuntimeError(f"variant image differs unexpectedly: {name}")
        if not _arrays_equal(left.image_masks[name], right.image_masks[name]):
            raise RuntimeError(f"variant image mask differs unexpectedly: {name}")


def _assert_same_state_context(left: Any, right: Any) -> None:
    for field in (
        "state",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "token_ar_mask",
        "token_state_mask",
    ):
        if not _arrays_equal(getattr(left, field), getattr(right, field)):
            raise RuntimeError(f"variant state/text context differs unexpectedly: {field}")


def _validate_variant_contract(variants: dict[str, Any]) -> None:
    if set(variants) != set(CONDITIONS):
        raise RuntimeError(f"variant set mismatch: expected {CONDITIONS}, got {tuple(variants)}")
    full = variants["reset_full"]
    image_null = variants["reset_image_null"]
    state_null = variants["reset_state_null"]
    both_null = variants["reset_both_null"]
    full_cf = variants["reset_full_counterfactual_task"]
    both_cf = variants["reset_both_null_counterfactual_task"]
    opposite_full = variants["reset_opposite_full_obs"]
    opposite_image = variants["reset_opposite_image_only"]
    opposite_state = variants["reset_opposite_state_only"]

    _assert_same_state_context(full, image_null)
    _assert_same_images(full, state_null)
    _assert_same_images(image_null, both_null)
    _assert_same_state_context(state_null, both_null)
    _assert_same_images(full, full_cf)
    if not _arrays_equal(full.state, full_cf.state):
        raise RuntimeError("counterfactual-task control changed continuous state")
    if _arrays_equal(full.tokenized_prompt, full_cf.tokenized_prompt):
        raise RuntimeError("counterfactual-task control did not change prompt tokens")
    _assert_same_images(both_null, both_cf)
    if not _arrays_equal(both_null.state, both_cf.state):
        raise RuntimeError("both-null task control changed continuous state")
    if _arrays_equal(both_null.tokenized_prompt, both_cf.tokenized_prompt):
        raise RuntimeError("both-null task control did not change prompt tokens")
    _assert_same_images(opposite_full, opposite_image)
    _assert_same_state_context(full, opposite_image)
    _assert_same_images(full, opposite_state)
    _assert_same_state_context(opposite_full, opposite_state)

    if not any(bool(np.asarray(mask).any()) for mask in full.image_masks.values()):
        raise RuntimeError("native full observation has no valid camera")
    for condition in ("reset_image_null", "reset_both_null", "reset_both_null_counterfactual_task"):
        observation = variants[condition]
        for name in observation.images:
            if np.any(np.asarray(observation.images[name])) or np.any(
                np.asarray(observation.image_masks[name])
            ):
                raise RuntimeError(f"{condition} is not an exact image-null intervention")


def _mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("mean requires nonempty finite values")
    return float(sum(values) / len(values))


def _balanced_gate(
    records: list[dict[str, Any]], value_fn, threshold: float
) -> dict[str, Any]:
    values = [float(value_fn(record)) for record in records]
    if not all(math.isfinite(value) for value in values):
        raise FloatingPointError("gate received NaN/Inf")
    episode_means = {
        episode: _mean(
            [float(value_fn(record)) for record in records if int(record["episode"]) == episode]
        )
        for episode in causal.EXPECTED_HELDOUT
    }
    frame_fraction = _mean([float(value > threshold) for value in values])
    episode_fraction = _mean(
        [float(value > threshold) for value in episode_means.values()]
    )
    macro_mean = _mean(list(episode_means.values()))
    return {
        "threshold": threshold,
        "frame_values": values,
        "episode_mean": episode_means,
        "cell_macro_mean": macro_mean,
        "frame_directional_fraction": frame_fraction,
        "episode_directional_fraction": episode_fraction,
        "required_directional_fraction": MIN_DIRECTIONAL_COVERAGE,
        "passes": (
            macro_mean > threshold
            and frame_fraction >= MIN_DIRECTIONAL_COVERAGE
            and episode_fraction >= MIN_DIRECTIONAL_COVERAGE
        ),
    }


def _margin(record: dict[str, Any], condition: str) -> float:
    return float(record["margins"][condition]["truth_aligned"])


def _raw_margin(record: dict[str, Any], condition: str) -> float:
    return float(record["margins"][condition]["right_minus_left"])


def _validate_records(records: list[dict[str, Any]]) -> None:
    expected_count = sum(len(frames) for frames in causal.EXPECTED_WAIT_FRAMES.values())
    if len(records) != expected_count:
        raise ValueError(f"expected {expected_count} records, got {len(records)}")
    seen = set()
    for record in records:
        expected_record_fields = {
            "episode",
            "prompt",
            "counterfactual_prompt",
            "truth_side",
            "truth_sign",
            "frame",
            "opposite_episode",
            "opposite_side",
            "opposite_frame",
            "token_noise_sha256",
            "observation_identity",
            "margins",
            "effects",
        }
        if set(record) != expected_record_fields:
            raise ValueError(
                "record schema mismatch: "
                f"missing={sorted(expected_record_fields - set(record))}, "
                f"extra={sorted(set(record) - expected_record_fields)}"
            )
        episode = int(record["episode"])
        frame = int(record["frame"])
        key = (episode, frame)
        if key in seen or frame not in causal.EXPECTED_WAIT_FRAMES.get(episode, ()):
            raise ValueError(f"unexpected/duplicate record {key}")
        seen.add(key)
        if int(record["opposite_episode"]) != OPPOSITE_EPISODE[episode]:
            raise ValueError(f"wrong opposite episode for {key}")
        if int(record["opposite_frame"]) != EXPECTED_OPPOSITE_FRAME[key]:
            raise ValueError(f"wrong opposite frame for {key}")
        expected_prompt, expected_side = causal.EXPECTED_CELLS[episode]
        other_prompts = {
            prompt for prompt, _side in causal.EXPECTED_CELLS.values()
        } - {expected_prompt}
        if (
            record["prompt"] != expected_prompt
            or record["truth_side"] != expected_side
            or int(record["truth_sign"]) != causal._truth_sign(expected_side)
            or record["counterfactual_prompt"] not in other_prompts
        ):
            raise ValueError(f"prompt/truth schema mismatch for {key}")
        expected_opposite_side = "right" if expected_side == "left" else "left"
        if record["opposite_side"] != expected_opposite_side:
            raise ValueError(f"wrong opposite side for {key}")
        identities = record["observation_identity"]
        if not isinstance(identities, dict) or set(identities) != set(CONDITIONS):
            raise ValueError(f"observation identity schema mismatch for {key}")
        for condition, identity in identities.items():
            if (
                not isinstance(identity, dict)
                or set(identity) != {"sha256", "fields"}
                or not causal._is_sha256(identity["sha256"])
                or not isinstance(identity["fields"], dict)
            ):
                raise ValueError(f"invalid observation identity for {key}/{condition}")
        if set(record["margins"]) != set(CONDITIONS):
            raise ValueError(f"condition schema mismatch for {key}")
        for condition in CONDITIONS:
            margin = record["margins"][condition]
            if set(margin) != {
                "logp_left",
                "logp_right",
                "right_minus_left",
                "truth_aligned",
                "correct",
            }:
                raise ValueError(f"margin schema mismatch for {key}/{condition}")
            numeric = [float(margin[name]) for name in ("logp_left", "logp_right", "right_minus_left", "truth_aligned")]
            if not all(math.isfinite(value) for value in numeric):
                raise FloatingPointError(f"nonfinite margin for {key}/{condition}")
            difference = numeric[1] - numeric[0]
            if not math.isclose(difference, numeric[2], rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"right-minus-left inconsistency for {key}/{condition}")
            expected_aligned = causal._truth_aligned(str(record["truth_side"]), numeric[2])
            if not math.isclose(expected_aligned, numeric[3], rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"truth alignment inconsistency for {key}/{condition}")
        if not causal._is_sha256(record["token_noise_sha256"]):
            raise ValueError(f"invalid matched-noise hash for {key}")
        expected_effects = {
            "full_minus_image_null_truth_aligned": (
                _margin(record, "reset_full") - _margin(record, "reset_image_null")
            ),
            "full_minus_state_null_truth_aligned": (
                _margin(record, "reset_full") - _margin(record, "reset_state_null")
            ),
            "full_minus_both_null_truth_aligned": (
                _margin(record, "reset_full") - _margin(record, "reset_both_null")
            ),
            "full_minus_opposite_full_truth_aligned": (
                _margin(record, "reset_full") - _margin(record, "reset_opposite_full_obs")
            ),
            "full_minus_opposite_image_truth_aligned": (
                _margin(record, "reset_full")
                - _margin(record, "reset_opposite_image_only")
            ),
            "full_minus_opposite_state_truth_aligned": (
                _margin(record, "reset_full")
                - _margin(record, "reset_opposite_state_only")
            ),
        }
        if not isinstance(record["effects"], dict) or set(record["effects"]) != set(
            expected_effects
        ):
            raise ValueError(f"effect schema mismatch for {key}")
        for name, expected in expected_effects.items():
            actual = float(record["effects"][name])
            if not math.isfinite(actual) or not math.isclose(
                actual, expected, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError(f"effect inconsistency for {key}/{name}")


def _both_null_invariant(records: list[dict[str, Any]]) -> dict[str, Any]:
    """With image/state fixed, equal instructions must produce equal logits across cells."""
    max_abs = 0.0
    comparisons = []
    for prompt in sorted({str(record["prompt"]) for record in records}):
        subset = [record for record in records if record["prompt"] == prompt]
        for condition in ("reset_both_null",):
            values = [_raw_margin(record, condition) for record in subset]
            spread = max(values) - min(values)
            max_abs = max(max_abs, abs(spread))
            comparisons.append({"prompt": prompt, "condition": condition, "spread": spread})
    # The counterfactual prompt reverses task names; group by the actually supplied prompt.
    for supplied_prompt in sorted({str(record["counterfactual_prompt"]) for record in records}):
        subset = [record for record in records if record["counterfactual_prompt"] == supplied_prompt]
        values = [_raw_margin(record, "reset_both_null_counterfactual_task") for record in subset]
        spread = max(values) - min(values)
        max_abs = max(max_abs, abs(spread))
        comparisons.append(
            {
                "prompt": supplied_prompt,
                "condition": "reset_both_null_counterfactual_task",
                "spread": spread,
            }
        )
    return {
        "tolerance": INVARIANT_TOL,
        "max_abs_within_supplied_prompt_spread": max_abs,
        "comparisons": comparisons,
        "passes": max_abs <= INVARIANT_TOL,
    }


def _condition_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for condition in CONDITIONS:
        episode_margin = {
            episode: _mean(
                [_margin(row, condition) for row in records if int(row["episode"]) == episode]
            )
            for episode in causal.EXPECTED_HELDOUT
        }
        episode_accuracy = {
            episode: _mean(
                [
                    float(row["margins"][condition]["correct"])
                    for row in records
                    if int(row["episode"]) == episode
                ]
            )
            for episode in causal.EXPECTED_HELDOUT
        }
        result[condition] = {
            "episode_mean_truth_aligned_margin": episode_margin,
            "cell_macro_mean_truth_aligned_margin": _mean(list(episode_margin.values())),
            "episode_accuracy": episode_accuracy,
            "cell_macro_accuracy": _mean(list(episode_accuracy.values())),
        }
    return result


def _acceptance_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    _validate_records(records)
    both_null = _both_null_invariant(records)
    if not both_null["passes"]:
        raise RuntimeError(
            "both-null same-instruction invariant failed: "
            f"{both_null['max_abs_within_supplied_prompt_spread']}"
        )

    primary = {
        "native_reset_favors_recipient_side": _balanced_gate(
            records, lambda row: _margin(row, "reset_full"), 0.0
        ),
        "opposite_full_observation_favors_donor_side": _balanced_gate(
            records, lambda row: -_margin(row, "reset_opposite_full_obs"), 0.0
        ),
        "full_observation_swap_moves_toward_donor": _balanced_gate(
            records,
            lambda row: _margin(row, "reset_full")
            - _margin(row, "reset_opposite_full_obs"),
            EFFECT_EPS,
        ),
    }
    direct_shortcut = all(gate["passes"] for gate in primary.values())

    image_localization_gates = {
        "opposite_images_favor_donor_with_recipient_state": _balanced_gate(
            records, lambda row: -_margin(row, "reset_opposite_image_only"), 0.0
        ),
        "image_swap_moves_toward_donor": _balanced_gate(
            records,
            lambda row: _margin(row, "reset_full")
            - _margin(row, "reset_opposite_image_only"),
            EFFECT_EPS,
        ),
    }
    state_localization_gates = {
        "opposite_state_favors_donor_with_recipient_images": _balanced_gate(
            records, lambda row: -_margin(row, "reset_opposite_state_only"), 0.0
        ),
        "state_swap_moves_toward_donor": _balanced_gate(
            records,
            lambda row: _margin(row, "reset_full")
            - _margin(row, "reset_opposite_state_only"),
            EFFECT_EPS,
        ),
    }
    image_localized = direct_shortcut and all(
        gate["passes"] for gate in image_localization_gates.values()
    )
    state_localized = direct_shortcut and all(
        gate["passes"] for gate in state_localization_gates.values()
    )

    null_corroboration = {
        "removing_images_reduces_recipient_margin": _balanced_gate(
            records,
            lambda row: _margin(row, "reset_full") - _margin(row, "reset_image_null"),
            EFFECT_EPS,
        ),
        "canonicalizing_state_reduces_recipient_margin": _balanced_gate(
            records,
            lambda row: _margin(row, "reset_full") - _margin(row, "reset_state_null"),
            EFFECT_EPS,
        ),
        "removing_both_reduces_recipient_margin": _balanced_gate(
            records,
            lambda row: _margin(row, "reset_full") - _margin(row, "reset_both_null"),
            EFFECT_EPS,
        ),
    }
    task_controls = {
        "native_observation_task_sensitivity_abs": _balanced_gate(
            records,
            lambda row: abs(
                _raw_margin(row, "reset_full")
                - _raw_margin(row, "reset_full_counterfactual_task")
            ),
            EFFECT_EPS,
        ),
        "both_null_task_sensitivity_abs": _balanced_gate(
            records,
            lambda row: abs(
                _raw_margin(row, "reset_both_null")
                - _raw_margin(row, "reset_both_null_counterfactual_task")
            ),
            EFFECT_EPS,
        ),
    }

    if not direct_shortcut:
        classification = "no primary reset-memory current-observation side-shortcut evidence"
    elif image_localized and state_localized:
        classification = "reset-memory side shortcut; both image and state channels localize"
    elif image_localized:
        classification = "reset-memory side shortcut localized to current images"
    elif state_localized:
        classification = "reset-memory side shortcut localized to current state"
    else:
        classification = "reset-memory side shortcut present but isolated channel is mixed/unlocalized"

    return {
        "criteria_version": "run5-step2500-waiting-shortcut-preregistered-v1",
        "effect_epsilon_nats": EFFECT_EPS,
        "minimum_directional_coverage": MIN_DIRECTIONAL_COVERAGE,
        "both_null_input_elimination_invariant": both_null,
        "condition_summary": _condition_summary(records),
        "primary_reset_observation_gates": primary,
        "direct_reset_observation_side_shortcut_evidence_pass": direct_shortcut,
        "image_localization_gates": image_localization_gates,
        "image_channel_localized": image_localized,
        "state_localization_gates": state_localization_gates,
        "state_channel_localized": state_localized,
        "null_ablation_corroboration": null_corroboration,
        "task_instruction_sensitivity_controls": task_controls,
        "classification": classification,
        "proof_claimed": False,
        "interpretation_guardrails": [
            "Null-ablation drops are corroboration only because absent cameras/canonical state are distribution shifts.",
            "Primary localization requires same-instruction, opposite-side, in-distribution observation swaps.",
            "Task-instruction sensitivity is expected task conditioning and is never side-shortcut evidence by itself.",
            "This audit localizes current-frame information under reset memory; it does not test closed-loop success.",
            "Raw and EMA reports are separate; a cross-source claim requires concordant classifications.",
        ],
    }


class WaitingShortcutEval:
    def __init__(self, args: Args):
        self.args = args
        base_args = causal.Args(
            checkpoint=args.checkpoint,
            dataset_root=args.dataset_root,
            output_dir=args.output_dir,
            config=args.config,
            parameter_source=args.parameter_source,
            mode="token",
            seed=args.seed,
        )
        self.base = causal.CausalMemoryEval(base_args)
        step_identity = self.base.parameter_provenance["train_state_step_identity"]
        if step_identity != {
            "checkpoint_manager_step_label": CHECKPOINT_STEP,
            "internal_train_state_step": INTERNAL_TRAIN_STEP,
        }:
            raise ValueError(f"unexpected checkpoint step identity: {step_identity}")
        self.pairing = _validate_pairing()

    def _variants(
        self,
        plan: Any,
        recipient_row: dict[str, Any],
        recipient_frame: int,
        opposite_row: dict[str, Any],
        opposite_frame: int,
        canonical_state: np.ndarray,
    ) -> dict[str, Any]:
        prompt = str(plan.prompt)
        counterfactual = str(plan.counterfactual)
        full, _ = self.base._observation(recipient_row, recipient_frame, prompt)
        state_null, _ = self.base._observation(
            _compose_row(recipient_row, canonical_state), recipient_frame, prompt
        )
        full_cf, _ = self.base._observation(recipient_row, recipient_frame, counterfactual)
        state_null_cf, _ = self.base._observation(
            _compose_row(recipient_row, canonical_state), recipient_frame, counterfactual
        )
        opposite_full, _ = self.base._observation(opposite_row, opposite_frame, prompt)
        opposite_image, _ = self.base._observation(
            _compose_row(opposite_row, recipient_row["state"]), recipient_frame, prompt
        )
        opposite_state, _ = self.base._observation(
            _compose_row(recipient_row, opposite_row["state"]), recipient_frame, prompt
        )
        variants = {
            "reset_full": full,
            "reset_image_null": _null_images(full),
            "reset_state_null": state_null,
            "reset_both_null": _null_images(state_null),
            "reset_full_counterfactual_task": full_cf,
            "reset_both_null_counterfactual_task": _null_images(state_null_cf),
            "reset_opposite_full_obs": opposite_full,
            "reset_opposite_image_only": opposite_image,
            "reset_opposite_state_only": opposite_state,
        }
        _validate_variant_contract(variants)
        return variants

    def _provenance(
        self,
        plans: list[Any],
        payloads: dict[int, dict[str, Any]],
        canonical_provenance: dict[str, Any],
        reset_identity: dict[str, Any],
    ) -> dict[str, Any]:
        repo = self.base.repo
        source_paths = (
            Path(__file__).resolve(),
            (repo / "scripts/v34_causal_memory_eval.py").resolve(),
        )
        dataset_paths = (
            self.args.dataset_root / "meta/tasks.jsonl",
            self.args.dataset_root / "meta/episode_prompts.json",
            self.args.dataset_root / "meta/info.json",
            self.args.dataset_root / "meta/episodes.jsonl",
        )
        return {
            "checkpoint": self.base.checkpoint_info,
            "checkpoint_origin": self.base.checkpoint_origin,
            "run5_launch_provenance": self.base.launch_provenance,
            "config": self.args.config,
            "parameter_source": self.base.parameter_provenance,
            "memory_eta_scale": float(self.base.model.memory.config.eta_scale),
            "blank_initial_output": bool(self.base.model.memory.config.blank_initial_output),
            "dataset_root": str(self.args.dataset_root),
            "normalization_asset_identity": self.base.norm_asset_identity,
            "tokenizer_asset_provenance": self.base.tokenizer_asset_provenance,
            "dataset_metadata_identity": {
                str(path): causal._file_identity(path) for path in dataset_paths
            },
            "heldout_parquet_identity": {
                episode: payloads[episode]["source_identity"]
                for episode in causal.EXPECTED_HELDOUT
            },
            "diagnostic_source_identity": {
                str(path): causal._file_identity(path) for path in source_paths
            },
            "heldout_cells": [
                {
                    "episode": int(plan.episode),
                    "prompt": str(plan.prompt),
                    "counterfactual_prompt": str(plan.counterfactual),
                    "side": str(plan.side),
                }
                for plan in plans
            ],
            "expected_wait_frames": causal.EXPECTED_WAIT_FRAMES,
            "opposite_episode_map": OPPOSITE_EPISODE,
            "opposite_frame_map": {
                f"ep{episode}_f{frame}": donor_frame
                for (episode, frame), donor_frame in sorted(self.pairing.items())
            },
            "opposite_frame_pairing_rule": (
                "same-instruction reciprocal heldout episode; nearest normalized waiting-grid rank, "
                "singleton recipient/donor rank fixed at 0.5; lower frame breaks ties"
            ),
            "canonical_state": canonical_provenance,
            "reset_memory_identity": reset_identity,
            "conditions": {
                "reset_full": "native images/state/task with freshly initialized M0",
                "reset_image_null": "pixels exactly zero and all camera masks false; native state/task",
                "reset_state_null": "fixed cell-balanced canonical waiting state; native images/task",
                "reset_both_null": "image-null plus fixed canonical state; native task",
                "reset_full_counterfactual_task": "native observation with the other in-dataset task prompt",
                "reset_both_null_counterfactual_task": "both-null observation with the other in-dataset task prompt",
                "reset_opposite_full_obs": "same-task opposite-side heldout images and state",
                "reset_opposite_image_only": "same-task opposite-side images; recipient state",
                "reset_opposite_state_only": "recipient images; same-task opposite-side state",
            },
            "write_timing": "fresh M0 at every score; allow_write=False/write_mode=frozen; zero committed writes",
            "matched_rng": (
                "one explicit fold-in key/noise tensor per recipient episode/frame reused by all conditions; "
                "teacher-forced token logp is deterministic with respect to the action noise"
            ),
            "seed": self.args.seed,
            "network_mode": "HF_HUB_OFFLINE=TRANSFORMERS_OFFLINE=HF_DATASETS_OFFLINE=1",
            "score": "D=log p(right)-log p(left), yD truth aligned to recipient side",
        }

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        self.args.output_dir.mkdir(parents=False, exist_ok=False)
        incomplete = self.args.output_dir / "INCOMPLETE.json"
        incomplete.write_text(
            json.dumps(
                causal._strict_json(
                    {
                        "schema": SCHEMA_VERSION,
                        "checkpoint": self.base.checkpoint_info,
                        "parameter_source": self.args.parameter_source,
                        "status": "reserved; shortcut replay/provenance not yet complete",
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

        plans, _snapshot_plans, _donor_maps, payloads = self.base._load_episode_payloads()
        canonical_state, canonical_provenance = _canonical_waiting_state(payloads)
        reset = self.base.model.memory.init_state(1)
        reset_identity = _tree_identity(reset)
        if self.base._state_max_abs_diff(reset, self.base.model.memory.init_state(1)) != 0.0:
            raise RuntimeError("fresh M0 is not deterministic")
        provenance = self._provenance(
            plans, payloads, canonical_provenance, reset_identity
        )
        incomplete.write_text(
            json.dumps(
                causal._strict_json({"schema": SCHEMA_VERSION, "provenance": provenance}),
                indent=2,
            ),
            encoding="utf-8",
        )

        by_plan = {int(plan.episode): plan for plan in plans}
        if tuple(sorted(by_plan)) != causal.EXPECTED_HELDOUT:
            raise ValueError(f"unexpected heldout plan set: {sorted(by_plan)}")
        expected_prompts = {cell[0] for cell in causal.EXPECTED_CELLS.values()}
        records = []
        for episode in causal.EXPECTED_HELDOUT:
            plan = by_plan[episode]
            if str(plan.counterfactual) not in expected_prompts - {str(plan.prompt)}:
                raise ValueError(
                    f"ep{episode} counterfactual prompt is not the other in-dataset task: "
                    f"{plan.counterfactual!r}"
                )
            opposite_episode = OPPOSITE_EPISODE[episode]
            opposite_plan = by_plan[opposite_episode]
            if opposite_plan.prompt != plan.prompt or opposite_plan.side == plan.side:
                raise ValueError(f"ep{episode} opposite control is not same-task/opposite-side")
            payload = payloads[episode]
            opposite_payload = payloads[opposite_episode]
            for frame in causal.EXPECTED_WAIT_FRAMES[episode]:
                opposite_frame = self.pairing[(episode, frame)]
                recipient_row = payload["by_frame"][frame]
                opposite_row = opposite_payload["by_frame"][opposite_frame]
                variants = self._variants(
                    plan,
                    recipient_row,
                    frame,
                    opposite_row,
                    opposite_frame,
                    canonical_state,
                )
                key, noise = self.base._noise(episode, frame, 0)
                margins = {
                    condition: self.base._score_side(
                        key,
                        noise,
                        variants[condition],
                        reset,
                        str(plan.side),
                        zero_read=False,
                    )
                    for condition in CONDITIONS
                }
                if _tree_identity(reset)["sha256"] != reset_identity["sha256"]:
                    raise RuntimeError("reset M0 changed during frozen shortcut scoring")
                record = {
                    "episode": episode,
                    "prompt": str(plan.prompt),
                    "counterfactual_prompt": str(plan.counterfactual),
                    "truth_side": str(plan.side),
                    "truth_sign": causal._truth_sign(str(plan.side)),
                    "frame": frame,
                    "opposite_episode": opposite_episode,
                    "opposite_side": str(opposite_plan.side),
                    "opposite_frame": opposite_frame,
                    "token_noise_sha256": causal._array_sha256(noise),
                    "observation_identity": {
                        condition: _observation_identity(variants[condition])
                        for condition in CONDITIONS
                    },
                    "margins": margins,
                    "effects": {
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
                    },
                }
                records.append(record)
                print(
                    f"ep{episode} f{frame} reset_full={margins['reset_full']['truth_aligned']:+.4f} "
                    f"image_null={margins['reset_image_null']['truth_aligned']:+.4f} "
                    f"state_null={margins['reset_state_null']['truth_aligned']:+.4f} "
                    f"both_null={margins['reset_both_null']['truth_aligned']:+.4f} "
                    f"opp_full={margins['reset_opposite_full_obs']['truth_aligned']:+.4f} "
                    f"opp_img={margins['reset_opposite_image_only']['truth_aligned']:+.4f} "
                    f"opp_state={margins['reset_opposite_state_only']['truth_aligned']:+.4f}",
                    flush=True,
                )

        acceptance = _acceptance_summary(records)
        report = causal._strict_json(
            {
                "schema": SCHEMA_VERSION,
                "provenance": provenance,
                "records": records,
                "acceptance": acceptance,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        temporary = self.args.output_dir / ".report.json.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.args.output_dir / "report.json")
        complete = self.args.output_dir / "COMPLETE"
        complete.write_text(
            f"{hashlib.sha256(encoded.encode()).hexdigest()}  report.json\n",
            encoding="utf-8",
        )
        with complete.open("rb") as handle:
            os.fsync(handle.fileno())
        incomplete.unlink()
        print(json.dumps(acceptance, indent=2), flush=True)
        print(f"wrote {self.args.output_dir / 'report.json'}", flush=True)
        return report


def main(argv: list[str] | None = None) -> None:
    WaitingShortcutEval(_parse_args(argv)).run()


if __name__ == "__main__":
    main()
