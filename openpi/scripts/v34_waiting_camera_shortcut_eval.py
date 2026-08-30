"""Read-only per-camera localization of the run5 waiting-observation shortcut.

This is a confirmatory extension of ``v34_waiting_shortcut_eval.py``.  It requires a completed,
checksummed parent report for the same checkpoint and parameter source whose primary result
localized the reset-memory shortcut to the image bundle.  At the same exact 12 held-out waiting
frames, it evaluates the full 2^3 factorial of same-instruction, opposite-side camera swaps:
top/base, left wrist, and right wrist, alone and in every combination.

Every variant keeps the recipient state and text context exactly fixed, uses the same paired donor
frame, starts from the same fresh M0, commits no writes, and reuses one explicit RNG/noise tensor.
The native and all-camera endpoints must replay within the same process and executable invocation.
The historical parent values are linked by exact input/provenance identities and compared
descriptively, not treated as bitwise GPU reference values.  A crash or failed invariant leaves
``INCOMPLETE.json``; only a fully validated report receives a checksummed ``COMPLETE`` marker.
Raw and EMA parameter sources are always separate runs and separate reports.
"""

# ruff: noqa: SLF001, I001 - evidence script intentionally reuses audited private helpers.
from __future__ import annotations

import pyarrow.parquet as pq  # noqa: F401 - must precede the OpenPI/JAX import stack.

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

import v34_waiting_shortcut_eval as bundle


causal = bundle.causal
SCHEMA_VERSION = "openpi.v34.waiting_camera_shortcut_eval.v2"
CHECKPOINT_STEP = bundle.CHECKPOINT_STEP
INTERNAL_TRAIN_STEP = bundle.INTERNAL_TRAIN_STEP
EFFECT_EPS = bundle.EFFECT_EPS
MIN_DIRECTIONAL_COVERAGE = bundle.MIN_DIRECTIONAL_COVERAGE
WITHIN_RUN_REPLAY_TOL = 1e-5

CAMERAS = ("top", "left_wrist", "right_wrist")
CAMERA_KEYS = {
    "top": "base_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}
SUBSETS = (
    (),
    ("top",),
    ("left_wrist",),
    ("right_wrist",),
    ("top", "left_wrist"),
    ("top", "right_wrist"),
    ("left_wrist", "right_wrist"),
    CAMERAS,
)
CONDITION_BY_SUBSET = {
    (): "reset_native",
    ("top",): "reset_swap_top",
    ("left_wrist",): "reset_swap_left_wrist",
    ("right_wrist",): "reset_swap_right_wrist",
    ("top", "left_wrist"): "reset_swap_top_left_wrist",
    ("top", "right_wrist"): "reset_swap_top_right_wrist",
    ("left_wrist", "right_wrist"): "reset_swap_left_right_wrist",
    CAMERAS: "reset_swap_all_cameras",
}
SUBSET_BY_CONDITION = {condition: subset for subset, condition in CONDITION_BY_SUBSET.items()}
CONDITIONS = tuple(CONDITION_BY_SUBSET[subset] for subset in SUBSETS)


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint: Path
    dataset_root: Path
    output_dir: Path
    parent_report: Path
    config: str
    parameter_source: Literal["raw", "ema"]
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("checkpoint", "dataset_root", "output_dir", "parent_report"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        if self.config != causal.RUN5_CONFIG:
            raise ValueError(
                f"camera shortcut audit is pinned to --config {causal.RUN5_CONFIG!r}; "
                f"got {self.config!r}"
            )
        if self.parameter_source not in ("raw", "ema"):
            raise ValueError("--parameter-source must be raw or ema")
        if self.checkpoint.name != str(CHECKPOINT_STEP):
            raise ValueError(
                f"camera shortcut audit is pinned to checkpoint {CHECKPOINT_STEP}; "
                f"got {self.checkpoint.name!r}"
            )
        if self.parent_report.name != "report.json":
            raise ValueError("--parent-report must name the completed parent's report.json")
        if self.seed != 0:
            raise ValueError("the pre-registered camera shortcut audit requires --seed 0")


def _parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parameter-source", choices=("raw", "ema"), required=True)
    parser.add_argument("--seed", type=int, default=0)
    return Args(**vars(parser.parse_args(argv)))


def _canonical_subset(cameras: Any) -> tuple[str, ...]:
    selected = set(cameras)
    if not selected.issubset(CAMERAS):
        raise ValueError(f"unknown camera subset: {sorted(selected)}")
    return tuple(camera for camera in CAMERAS if camera in selected)


def _condition(cameras: Any) -> str:
    return CONDITION_BY_SUBSET[_canonical_subset(cameras)]


def _read_completed_parent(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(path).expanduser().resolve()
    parent_dir = path.parent
    incomplete = parent_dir / "INCOMPLETE.json"
    complete = parent_dir / "COMPLETE"
    if incomplete.exists():
        raise RuntimeError(f"parent audit is incomplete: {incomplete}")
    if not path.is_file() or not complete.is_file():
        raise FileNotFoundError("parent report and COMPLETE marker are both required")
    encoded = path.read_bytes()
    report_sha = hashlib.sha256(encoded).hexdigest()
    expected_line = f"{report_sha}  report.json"
    lines = complete.read_text(encoding="utf-8").splitlines()
    if lines != [expected_line]:
        raise RuntimeError(
            f"parent COMPLETE marker mismatch: expected {[expected_line]!r}, got {lines!r}"
        )
    try:
        report = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid parent JSON: {path}") from exc
    identity = {
        "report": causal._file_identity(path),
        "complete": causal._file_identity(complete),
    }
    return report, identity


def _validate_parent_report(
    report: dict[str, Any], args: Args, base: Any
) -> dict[tuple[int, int], dict[str, Any]]:
    expected_fields = {"schema", "provenance", "records", "acceptance", "elapsed_seconds"}
    if not isinstance(report, dict) or set(report) != expected_fields:
        raise ValueError("parent report top-level schema mismatch")
    if report["schema"] != bundle.SCHEMA_VERSION:
        raise ValueError(f"wrong parent schema: {report['schema']!r}")
    bundle._validate_records(report["records"])
    recomputed = causal._strict_json(bundle._acceptance_summary(report["records"]))
    if recomputed != report["acceptance"]:
        raise RuntimeError("parent acceptance does not recompute exactly from its records")
    if not bool(report["acceptance"]["direct_reset_observation_side_shortcut_evidence_pass"]):
        raise RuntimeError("parent report did not establish a reset-memory observation shortcut")
    if not bool(report["acceptance"]["image_channel_localized"]):
        raise RuntimeError("parent report did not localize the shortcut to the image bundle")

    provenance = report["provenance"]
    required = {
        "checkpoint",
        "parameter_source",
        "config",
        "dataset_root",
        "expected_wait_frames",
        "opposite_episode_map",
        "opposite_frame_map",
        "heldout_parquet_identity",
        "diagnostic_source_identity",
    }
    if not isinstance(provenance, dict) or not required.issubset(provenance):
        raise ValueError("parent provenance is incomplete")
    expected_wait_frames = causal._strict_json(causal.EXPECTED_WAIT_FRAMES)
    expected_opposite_episodes = causal._strict_json(bundle.OPPOSITE_EPISODE)
    expected_opposite_frames = {
        f"ep{episode}_f{frame}": donor
        for (episode, frame), donor in sorted(bundle.EXPECTED_OPPOSITE_FRAME.items())
    }
    comparisons = {
        "checkpoint": (provenance["checkpoint"], base.checkpoint_info),
        "parameter_source": (provenance["parameter_source"], base.parameter_provenance),
        "config": (provenance["config"], args.config),
        "dataset_root": (provenance["dataset_root"], str(args.dataset_root)),
        "expected_wait_frames": (provenance["expected_wait_frames"], expected_wait_frames),
        "opposite_episode_map": (
            provenance["opposite_episode_map"],
            expected_opposite_episodes,
        ),
        "opposite_frame_map": (provenance["opposite_frame_map"], expected_opposite_frames),
    }
    for name, (actual, expected) in comparisons.items():
        if actual != expected:
            raise RuntimeError(f"parent {name} does not match this camera audit")
    if provenance["parameter_source"].get("parameter_source") != args.parameter_source:
        raise RuntimeError("parent raw/EMA source does not match --parameter-source")
    for source, identity in provenance["diagnostic_source_identity"].items():
        if causal._file_identity(Path(source)) != identity:
            raise RuntimeError(f"parent-bound diagnostic source changed: {source}")

    by_cell = {}
    for record in report["records"]:
        key = (int(record["episode"]), int(record["frame"]))
        if key in by_cell:
            raise ValueError(f"duplicate parent cell: {key}")
        by_cell[key] = record
    expected_cells = {
        (episode, frame)
        for episode, frames in causal.EXPECTED_WAIT_FRAMES.items()
        for frame in frames
    }
    if set(by_cell) != expected_cells:
        raise ValueError("parent cell grid differs from the exact 12-frame registration")
    return by_cell


def _validate_parent_payloads(
    parent_report: dict[str, Any], payloads: dict[int, dict[str, Any]]
) -> None:
    actual = causal._strict_json(
        {
            episode: payloads[episode]["source_identity"]
            for episode in causal.EXPECTED_HELDOUT
        }
    )
    expected = parent_report["provenance"]["heldout_parquet_identity"]
    if actual != expected:
        raise RuntimeError("heldout parquet identities differ from the completed parent audit")


def _camera_equal(left: Any, right: Any, key: str) -> bool:
    return bundle._arrays_equal(left.images[key], right.images[key]) and bundle._arrays_equal(
        left.image_masks[key], right.image_masks[key]
    )


def _swap_cameras(recipient: Any, donor: Any, cameras: Any) -> Any:
    selected = _canonical_subset(cameras)
    expected_keys = tuple(CAMERA_KEYS[camera] for camera in CAMERAS)
    for label, observation in (("recipient", recipient), ("donor", donor)):
        if tuple(observation.images) != expected_keys or tuple(observation.image_masks) != expected_keys:
            raise RuntimeError(
                f"{label} camera order/keys changed: expected {expected_keys}, "
                f"got {tuple(observation.images)}/{tuple(observation.image_masks)}"
            )
    images = dict(recipient.images)
    masks = dict(recipient.image_masks)
    for camera in selected:
        key = CAMERA_KEYS[camera]
        images[key] = donor.images[key]
        masks[key] = donor.image_masks[key]
    return recipient.replace(images=images, image_masks=masks)


def _camera_variants(recipient: Any, donor: Any) -> dict[str, Any]:
    expected_keys = tuple(CAMERA_KEYS[camera] for camera in CAMERAS)
    if tuple(recipient.images) != expected_keys or tuple(donor.images) != expected_keys:
        raise RuntimeError("camera model contract changed")
    for camera, key in CAMERA_KEYS.items():
        if (
            np.asarray(recipient.images[key]).shape != np.asarray(donor.images[key]).shape
            or np.asarray(recipient.images[key]).dtype != np.asarray(donor.images[key]).dtype
            or np.asarray(recipient.image_masks[key]).shape
            != np.asarray(donor.image_masks[key]).shape
        ):
            raise RuntimeError(f"paired {camera} image/mask tensor contract changed")
        if not np.all(np.asarray(recipient.image_masks[key])) or not np.all(
            np.asarray(donor.image_masks[key])
        ):
            raise RuntimeError(f"paired {camera} camera is not valid in both observations")
        if _camera_equal(recipient, donor, key):
            raise RuntimeError(f"paired {camera} intervention is vacuous")

    variants = {
        CONDITION_BY_SUBSET[subset]: _swap_cameras(recipient, donor, subset)
        for subset in SUBSETS
    }
    if set(variants) != set(CONDITIONS):
        raise RuntimeError("camera factorial condition set changed")
    for condition, variant in variants.items():
        subset = SUBSET_BY_CONDITION[condition]
        bundle._assert_same_state_context(recipient, variant)
        for camera, key in CAMERA_KEYS.items():
            expected = donor if camera in subset else recipient
            if not _camera_equal(variant, expected, key):
                raise RuntimeError(f"{condition} violates the {camera} swap contract")
    return variants


def _mean(values: list[float]) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("mean requires nonempty finite values")
    return float(sum(values) / len(values))


def _margin(record: dict[str, Any], condition: str) -> float:
    return float(record["margins"][condition]["truth_aligned"])


def _camera_effects(margins: dict[str, dict[str, Any]]) -> dict[str, Any]:
    aligned = {
        subset: float(margins[CONDITION_BY_SUBSET[subset]]["truth_aligned"])
        for subset in SUBSETS
    }
    if not all(math.isfinite(value) for value in aligned.values()):
        raise FloatingPointError("camera effects received NaN/Inf")
    camera_effects = {}
    shapley_sum = 0.0
    for camera in CAMERAS:
        edges = []
        for background in SUBSETS:
            if camera in background:
                continue
            swapped = _canonical_subset((*background, camera))
            weight = {0: 1.0 / 3.0, 1: 1.0 / 6.0, 2: 1.0 / 3.0}[len(background)]
            drop = aligned[background] - aligned[swapped]
            edges.append(
                {
                    "background_condition": CONDITION_BY_SUBSET[background],
                    "swapped_condition": CONDITION_BY_SUBSET[swapped],
                    "shapley_weight": weight,
                    "truth_aligned_drop_toward_donor": drop,
                }
            )
        if len(edges) != 4 or not math.isclose(
            sum(float(edge["shapley_weight"]) for edge in edges), 1.0, abs_tol=1e-12
        ):
            raise RuntimeError("invalid three-camera Shapley edge construction")
        shapley = sum(
            float(edge["shapley_weight"])
            * float(edge["truth_aligned_drop_toward_donor"])
            for edge in edges
        )
        singleton = (camera,)
        all_without = _canonical_subset(item for item in CAMERAS if item != camera)
        camera_effects[camera] = {
            "isolated_swap_drop_toward_donor": aligned[()] - aligned[singleton],
            "restore_native_from_all_donor_drop_reversed": aligned[all_without] - aligned[CAMERAS],
            "factorial_edge_mean_drop_toward_donor": _mean(
                [float(edge["truth_aligned_drop_toward_donor"]) for edge in edges]
            ),
            "shapley_drop_toward_donor": shapley,
            "edges": edges,
        }
        shapley_sum += shapley

    pair_interactions = {}
    for left, right in itertools.combinations(CAMERAS, 2):
        pair = _canonical_subset((left, right))
        joint_drop = aligned[()] - aligned[pair]
        isolated_sum = (aligned[()] - aligned[(left,)]) + (aligned[()] - aligned[(right,)])
        pair_interactions[f"{left}+{right}"] = {
            "joint_drop_toward_donor": joint_drop,
            "sum_of_isolated_drops": isolated_sum,
            "joint_minus_isolated_sum": joint_drop - isolated_sum,
        }

    total = aligned[()] - aligned[CAMERAS]
    return {
        "bundle_drop_toward_donor": total,
        "camera": camera_effects,
        "pair_interactions_at_native_background": pair_interactions,
        "shapley_sum": shapley_sum,
        "shapley_efficiency_residual": shapley_sum - total,
    }


def _historical_parent_comparison(
    parent: dict[str, Any],
    noise_sha256: str,
    identities: dict[str, dict[str, Any]],
    margins: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Bind current endpoints to the parent inputs, without assuming cross-process bit identity.

    GPU BF16/XLA inference is not bitwise stable across separately compiled processes.  The
    numeric deltas are therefore descriptive.  Exact observation/noise identity remains a hard
    contract, while the parent conclusion is re-established by the current run's aggregate gates.
    """
    comparisons = {
        "native": ("reset_native", "reset_full"),
        "all_cameras": ("reset_swap_all_cameras", "reset_opposite_image_only"),
    }
    differences = {}
    max_abs = 0.0
    identities_equal = {}
    for label, (condition, parent_condition) in comparisons.items():
        field_differences = {}
        for field in ("logp_left", "logp_right", "right_minus_left", "truth_aligned"):
            difference = abs(
                float(margins[condition][field])
                - float(parent["margins"][parent_condition][field])
            )
            field_differences[field] = difference
            max_abs = max(max_abs, difference)
        differences[label] = field_differences
        identities_equal[label] = (
            identities[condition] == parent["observation_identity"][parent_condition]
        )
    noise_equal = noise_sha256 == parent["token_noise_sha256"]
    return {
        "max_abs_margin_difference": max_abs,
        "margin_absolute_differences": differences,
        "observation_identities_equal": identities_equal,
        "token_noise_identity_equal": noise_equal,
        "identity_contract_passes": all(identities_equal.values()) and noise_equal,
        "numeric_comparison_is_gating": False,
        "non_gating_reason": (
            "separate GPU/XLA compilations are not bitwise reference executions; exact replay is "
            "tested within this process and the bundle conclusion is re-tested on all 12 cells"
        ),
    }


def _within_run_endpoint_replay(
    margins: dict[str, dict[str, Any]], replayed: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    endpoints = ("reset_native", "reset_swap_all_cameras")
    if set(replayed) != set(endpoints):
        raise ValueError(f"within-run endpoint replay set changed: {tuple(replayed)}")
    differences = {}
    max_abs = 0.0
    for condition in endpoints:
        field_differences = {}
        for field in ("logp_left", "logp_right", "right_minus_left", "truth_aligned"):
            difference = abs(float(margins[condition][field]) - float(replayed[condition][field]))
            field_differences[field] = difference
            max_abs = max(max_abs, difference)
        if bool(margins[condition]["correct"]) != bool(replayed[condition]["correct"]):
            raise RuntimeError(f"within-run endpoint correctness changed for {condition}")
        differences[condition] = field_differences
    return {
        "tolerance": WITHIN_RUN_REPLAY_TOL,
        "max_abs_margin_difference": max_abs,
        "margin_absolute_differences": differences,
        "passes": max_abs <= WITHIN_RUN_REPLAY_TOL,
    }


def _numeric_close(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def _validate_difference_table(
    table: Any, expected_labels: set[str], *, context: str
) -> float:
    expected_fields = {"logp_left", "logp_right", "right_minus_left", "truth_aligned"}
    if (
        not isinstance(table, dict)
        or set(table) != expected_labels
        or any(
            not isinstance(values, dict) or set(values) != expected_fields
            for values in table.values()
        )
    ):
        raise ValueError(f"{context} differences mismatch")
    flat = [float(value) for values in table.values() for value in values.values()]
    if not all(math.isfinite(value) and value >= 0.0 for value in flat):
        raise FloatingPointError(f"invalid {context} difference")
    return max(flat)


def _validate_records(records: list[dict[str, Any]]) -> None:
    expected_grid = {
        (episode, frame)
        for episode, frames in causal.EXPECTED_WAIT_FRAMES.items()
        for frame in frames
    }
    actual_grid = [(int(record["episode"]), int(record["frame"])) for record in records]
    if len(actual_grid) != len(set(actual_grid)) or set(actual_grid) != expected_grid:
        raise ValueError("camera records differ from the exact 12-frame registration")
    expected_fields = {
        "episode",
        "prompt",
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
        "historical_parent_comparison",
        "within_run_endpoint_replay",
    }
    margin_fields = {
        "logp_left",
        "logp_right",
        "right_minus_left",
        "truth_aligned",
        "correct",
    }
    for record in records:
        if set(record) != expected_fields:
            raise ValueError("camera record schema mismatch")
        episode = int(record["episode"])
        frame = int(record["frame"])
        key = (episode, frame)
        prompt, side = causal.EXPECTED_CELLS[episode]
        opposite_episode = bundle.OPPOSITE_EPISODE[episode]
        expected_opposite_side = "right" if side == "left" else "left"
        if (
            record["prompt"] != prompt
            or record["truth_side"] != side
            or int(record["truth_sign"]) != causal._truth_sign(side)
            or int(record["opposite_episode"]) != opposite_episode
            or record["opposite_side"] != expected_opposite_side
            or int(record["opposite_frame"]) != bundle.EXPECTED_OPPOSITE_FRAME[key]
        ):
            raise ValueError(f"camera record truth/pairing mismatch for {key}")
        if not causal._is_sha256(record["token_noise_sha256"]):
            raise ValueError(f"invalid noise identity for {key}")
        if set(record["observation_identity"]) != set(CONDITIONS):
            raise ValueError(f"observation identity condition mismatch for {key}")
        for condition, identity in record["observation_identity"].items():
            if (
                not isinstance(identity, dict)
                or set(identity) != {"sha256", "fields"}
                or not causal._is_sha256(identity["sha256"])
                or not isinstance(identity["fields"], dict)
            ):
                raise ValueError(f"invalid observation identity for {key}/{condition}")
        if set(record["margins"]) != set(CONDITIONS):
            raise ValueError(f"margin condition mismatch for {key}")
        for condition, margin in record["margins"].items():
            if set(margin) != margin_fields:
                raise ValueError(f"margin schema mismatch for {key}/{condition}")
            numeric = {
                name: float(margin[name])
                for name in margin_fields
                if name != "correct"
            }
            if not all(math.isfinite(value) for value in numeric.values()):
                raise FloatingPointError(f"nonfinite margin for {key}/{condition}")
            difference = numeric["logp_right"] - numeric["logp_left"]
            aligned = causal._truth_aligned(side, difference)
            if not _numeric_close(difference, numeric["right_minus_left"]):
                raise ValueError(f"right-minus-left inconsistency for {key}/{condition}")
            if not _numeric_close(aligned, numeric["truth_aligned"]):
                raise ValueError(f"truth alignment inconsistency for {key}/{condition}")
            if bool(margin["correct"]) != (aligned > 0.0):
                raise ValueError(f"correctness inconsistency for {key}/{condition}")
        expected_effects = _camera_effects(record["margins"])
        if causal._strict_json(record["effects"]) != causal._strict_json(expected_effects):
            raise ValueError(f"camera effect inconsistency for {key}")
        residual = float(record["effects"]["shapley_efficiency_residual"])
        if abs(residual) > 1e-6:
            raise RuntimeError(f"Shapley efficiency invariant failed for {key}: {residual}")
        parent_comparison = record["historical_parent_comparison"]
        parent_fields = {
            "max_abs_margin_difference",
            "margin_absolute_differences",
            "observation_identities_equal",
            "token_noise_identity_equal",
            "identity_contract_passes",
            "numeric_comparison_is_gating",
            "non_gating_reason",
        }
        if not isinstance(parent_comparison, dict) or set(parent_comparison) != parent_fields:
            raise ValueError(f"historical parent comparison schema mismatch for {key}")
        parent_max = _validate_difference_table(
            parent_comparison["margin_absolute_differences"],
            {"native", "all_cameras"},
            context=f"historical parent comparison for {key}",
        )
        if not _numeric_close(parent_comparison["max_abs_margin_difference"], parent_max):
            raise ValueError(f"historical parent comparison maximum mismatch for {key}")
        identity_equal = parent_comparison["observation_identities_equal"]
        if not isinstance(identity_equal, dict) or set(identity_equal) != {"native", "all_cameras"}:
            raise ValueError(f"historical parent observation identity mismatch for {key}")
        expected_identity_pass = all(bool(value) for value in identity_equal.values()) and bool(
            parent_comparison["token_noise_identity_equal"]
        )
        if bool(parent_comparison["identity_contract_passes"]) != expected_identity_pass:
            raise ValueError(f"historical parent identity pass flag is inconsistent for {key}")
        if not expected_identity_pass:
            raise RuntimeError(f"historical parent input identity contract failed for {key}")
        if bool(parent_comparison["numeric_comparison_is_gating"]):
            raise ValueError(f"historical cross-process numeric comparison became gating for {key}")
        if not isinstance(parent_comparison["non_gating_reason"], str) or not parent_comparison[
            "non_gating_reason"
        ]:
            raise ValueError(f"historical parent non-gating rationale missing for {key}")

        replay = record["within_run_endpoint_replay"]
        replay_fields = {
            "tolerance",
            "max_abs_margin_difference",
            "margin_absolute_differences",
            "passes",
        }
        if not isinstance(replay, dict) or set(replay) != replay_fields:
            raise ValueError(f"within-run endpoint replay schema mismatch for {key}")
        if not _numeric_close(replay["tolerance"], WITHIN_RUN_REPLAY_TOL, tolerance=0.0):
            raise ValueError(f"within-run endpoint replay tolerance changed for {key}")
        replay_max = _validate_difference_table(
            replay["margin_absolute_differences"],
            {"reset_native", "reset_swap_all_cameras"},
            context=f"within-run endpoint replay for {key}",
        )
        if not _numeric_close(replay["max_abs_margin_difference"], replay_max):
            raise ValueError(f"within-run endpoint replay maximum mismatch for {key}")
        expected_replay_pass = replay_max <= WITHIN_RUN_REPLAY_TOL
        if bool(replay["passes"]) != expected_replay_pass:
            raise ValueError(f"within-run endpoint replay pass flag is inconsistent for {key}")
        if not expected_replay_pass:
            raise RuntimeError(f"within-run endpoint replay invariant failed for {key}")


def _balanced_gate(records: list[dict[str, Any]], value_fn, threshold: float) -> dict[str, Any]:
    return bundle._balanced_gate(records, value_fn, threshold)


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


def _acceptance_summary(
    records: list[dict[str, Any]], parent_summary: dict[str, Any]
) -> dict[str, Any]:
    _validate_records(records)
    primary = {
        "native_reset_favors_recipient_side": _balanced_gate(
            records, lambda row: _margin(row, _condition(())), 0.0
        ),
        "all_camera_swap_favors_donor_side": _balanced_gate(
            records, lambda row: -_margin(row, _condition(CAMERAS)), 0.0
        ),
        "all_camera_swap_moves_toward_donor": _balanced_gate(
            records,
            lambda row: float(row["effects"]["bundle_drop_toward_donor"]),
            EFFECT_EPS,
        ),
    }
    bundle_replicated = all(gate["passes"] for gate in primary.values())

    camera_results = {}
    implicated = []
    for camera in CAMERAS:
        singleton = _condition((camera,))
        all_without = _condition(item for item in CAMERAS if item != camera)
        gates = {
            "isolated_swap_moves_toward_donor": _balanced_gate(
                records,
                lambda row, c=camera: float(
                    row["effects"]["camera"][c]["isolated_swap_drop_toward_donor"]
                ),
                EFFECT_EPS,
            ),
            "isolated_swap_favors_donor": _balanced_gate(
                records, lambda row, c=singleton: -_margin(row, c), 0.0
            ),
            "restoring_only_this_native_camera_moves_toward_recipient": _balanced_gate(
                records,
                lambda row, c=camera: float(
                    row["effects"]["camera"][c][
                        "restore_native_from_all_donor_drop_reversed"
                    ]
                ),
                EFFECT_EPS,
            ),
            "keeping_only_this_camera_native_favors_recipient": _balanced_gate(
                records, lambda row, c=all_without: _margin(row, c), 0.0
            ),
            "shapley_directional_contribution": _balanced_gate(
                records,
                lambda row, c=camera: float(
                    row["effects"]["camera"][c]["shapley_drop_toward_donor"]
                ),
                EFFECT_EPS,
            ),
        }
        directional = bundle_replicated and gates["shapley_directional_contribution"]["passes"]
        isolated_sufficient = bundle_replicated and all(
            gates[name]["passes"]
            for name in ("isolated_swap_moves_toward_donor", "isolated_swap_favors_donor")
        )
        native_rescue = bundle_replicated and all(
            gates[name]["passes"]
            for name in (
                "restoring_only_this_native_camera_moves_toward_recipient",
                "keeping_only_this_camera_native_favors_recipient",
            )
        )
        camera_results[camera] = {
            "camera_key": CAMERA_KEYS[camera],
            "gates": gates,
            "factorial_directional_contribution": directional,
            "isolated_swap_sufficient_for_donor_side": isolated_sufficient,
            "native_camera_rescues_recipient_side_against_other_donor_cameras": native_rescue,
        }
        if directional:
            implicated.append(camera)

    historical_parent_max = max(
        float(row["historical_parent_comparison"]["max_abs_margin_difference"])
        for row in records
    )
    within_run_replay_max = max(
        float(row["within_run_endpoint_replay"]["max_abs_margin_difference"])
        for row in records
    )
    if not bundle_replicated:
        classification = "completed parent image shortcut did not replicate in camera factorial"
    elif not implicated:
        classification = "image-bundle shortcut replicated; no camera clears factorial attribution gate"
    else:
        classification = (
            "image-bundle shortcut replicated; factorial directional contribution from "
            + ", ".join(implicated)
        )
    return {
        "criteria_version": "run5-step2500-waiting-camera-factorial-preregistered-v2",
        "effect_epsilon_nats": EFFECT_EPS,
        "minimum_directional_coverage": MIN_DIRECTIONAL_COVERAGE,
        "parent_bundle_prerequisite": parent_summary,
        "historical_parent_endpoint_comparison": {
            "numeric_comparison_is_gating": False,
            "max_abs_margin_difference": historical_parent_max,
            "all_observation_and_noise_identities_match": all(
                row["historical_parent_comparison"]["identity_contract_passes"]
                for row in records
            ),
        },
        "within_run_endpoint_replay_invariant": {
            "tolerance": WITHIN_RUN_REPLAY_TOL,
            "max_abs_margin_difference": within_run_replay_max,
            "all_12_cells_pass": all(
                row["within_run_endpoint_replay"]["passes"] for row in records
            ),
        },
        "condition_summary": _condition_summary(records),
        "bundle_replication_gates": primary,
        "bundle_shortcut_replicated": bundle_replicated,
        "camera": camera_results,
        "factorially_implicated_cameras": implicated,
        "classification": classification,
        "proof_claimed": False,
        "interpretation_guardrails": [
            "A positive Shapley gate is interaction-aware directional attribution, not proof of a unique visual cue.",
            "Isolated-swap sufficiency and native-camera rescue are reported separately from factorial attribution.",
            "Opposite-side donor episodes can differ in pose, lighting, object placement, and episode identity beyond side.",
            "The four episode cells are balanced but small; frame rows within an episode are correlated.",
            "Fresh M0 and frozen writes isolate current observation influence; this does not test memory or closed-loop success.",
            "Historical parent log-probability deltas are descriptive because separately compiled BF16 GPU executions are not bitwise references.",
            "Exact endpoint repeatability is enforced within one process; the parent bundle conclusion is re-tested over all 12 current cells.",
            "Raw and EMA reports are separate; a cross-source conclusion requires concordant camera roles.",
        ],
    }


def _write_json_fsync(path: Path, value: Any) -> None:
    encoded = json.dumps(causal._strict_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _reserve_output(output_dir: Path, value: Any) -> Path:
    output_dir.mkdir(parents=False, exist_ok=False)
    incomplete = output_dir / "INCOMPLETE.json"
    _write_json_fsync(incomplete, value)
    return incomplete


def _finalize_report(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(
        causal._strict_json(report), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    temporary_report = output_dir / ".report.json.tmp"
    with temporary_report.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    report_path = output_dir / "report.json"
    os.replace(temporary_report, report_path)
    complete_text = f"{hashlib.sha256(encoded.encode()).hexdigest()}  report.json\n"
    temporary_complete = output_dir / ".COMPLETE.tmp"
    with temporary_complete.open("w", encoding="utf-8") as handle:
        handle.write(complete_text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_complete, output_dir / "COMPLETE")
    (output_dir / "INCOMPLETE.json").unlink()
    return {
        "report": causal._file_identity(report_path),
        "complete": causal._file_identity(output_dir / "COMPLETE"),
    }


class WaitingCameraShortcutEval:
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
        self.pairing = bundle._validate_pairing()
        self.parent_report, self.parent_identity = _read_completed_parent(args.parent_report)
        self.parent_by_cell = _validate_parent_report(self.parent_report, args, self.base)

    def _provenance(
        self,
        plans: list[Any],
        payloads: dict[int, dict[str, Any]],
        reset_identity: dict[str, Any],
    ) -> dict[str, Any]:
        repo = self.base.repo
        source_paths = (
            Path(__file__).resolve(),
            Path(bundle.__file__).resolve(),
            (repo / "scripts/v34_causal_memory_eval.py").resolve(),
        )
        dataset_paths = (
            self.args.dataset_root / "meta/tasks.jsonl",
            self.args.dataset_root / "meta/episode_prompts.json",
            self.args.dataset_root / "meta/info.json",
            self.args.dataset_root / "meta/episodes.jsonl",
        )
        parent_acceptance = self.parent_report["acceptance"]
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
            "parent_waiting_shortcut_report": {
                **self.parent_identity,
                "schema": self.parent_report["schema"],
                "classification": parent_acceptance["classification"],
                "direct_reset_observation_side_shortcut_evidence_pass": parent_acceptance[
                    "direct_reset_observation_side_shortcut_evidence_pass"
                ],
                "image_channel_localized": parent_acceptance["image_channel_localized"],
                "state_channel_localized": parent_acceptance["state_channel_localized"],
            },
            "heldout_cells": [
                {
                    "episode": int(plan.episode),
                    "prompt": str(plan.prompt),
                    "side": str(plan.side),
                }
                for plan in plans
            ],
            "expected_wait_frames": causal.EXPECTED_WAIT_FRAMES,
            "opposite_episode_map": bundle.OPPOSITE_EPISODE,
            "opposite_frame_map": {
                f"ep{episode}_f{frame}": donor
                for (episode, frame), donor in sorted(self.pairing.items())
            },
            "camera_keys": CAMERA_KEYS,
            "factorial_conditions": {
                condition: list(SUBSET_BY_CONDITION[condition]) for condition in CONDITIONS
            },
            "intervention": (
                "same-instruction reciprocal heldout opposite-side image+mask swaps; recipient state "
                "and all token/text context fixed exactly"
            ),
            "reset_memory_identity": reset_identity,
            "write_timing": (
                "fresh M0 at every score; allow_write=False/write_mode=frozen; zero committed writes"
            ),
            "matched_rng": (
                "one explicit fold-in key/noise tensor per recipient episode/frame reused by all "
                "eight camera conditions and the same-process native/all endpoint replay"
            ),
            "endpoint_replay_semantics": {
                "historical_parent": (
                    "exact checkpoint/source/observation/noise identity bridge plus descriptive "
                    "numeric deltas; cross-process GPU values are not a bitwise gate"
                ),
                "within_run": (
                    "native and all-camera conditions rescored with the same executable, M0, key, "
                    f"and noise; max absolute logp/margin tolerance {WITHIN_RUN_REPLAY_TOL}"
                ),
                "conclusion_replication": (
                    "the completed parent's bundle-level conclusion must independently clear the "
                    "current run's balanced 12-cell gates"
                ),
            },
            "runtime_numeric_context": {
                "jax_version": str(bundle.jax.__version__),
                "devices": [
                    {
                        "id": int(device.id),
                        "platform": str(device.platform),
                        "device_kind": str(device.device_kind),
                        "process_index": int(device.process_index),
                    }
                    for device in bundle.jax.devices()
                ],
                "environment": {
                    name: os.environ.get(name)
                    for name in (
                        "CUDA_VISIBLE_DEVICES",
                        "JAX_DEFAULT_MATMUL_PRECISION",
                        "NVIDIA_TF32_OVERRIDE",
                        "XLA_FLAGS",
                    )
                },
            },
            "seed": self.args.seed,
            "network_mode": "HF_HUB_OFFLINE=TRANSFORMERS_OFFLINE=HF_DATASETS_OFFLINE=1",
            "score": "D=log p(right)-log p(left), yD truth aligned to recipient side",
        }

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        incomplete = _reserve_output(
            self.args.output_dir,
            {
                "schema": SCHEMA_VERSION,
                "checkpoint": self.base.checkpoint_info,
                "parameter_source": self.args.parameter_source,
                "parent_report": self.parent_identity,
                "status": "reserved; per-camera replay/provenance not yet complete",
            },
        )
        plans, _snapshot_plans, _donor_maps, payloads = self.base._load_episode_payloads()
        _validate_parent_payloads(self.parent_report, payloads)
        reset = self.base.model.memory.init_state(1)
        reset_identity = bundle._tree_identity(reset)
        if self.base._state_max_abs_diff(reset, self.base.model.memory.init_state(1)) != 0.0:
            raise RuntimeError("fresh M0 is not deterministic")
        provenance = self._provenance(plans, payloads, reset_identity)
        _write_json_fsync(incomplete, {"schema": SCHEMA_VERSION, "provenance": provenance})

        by_plan = {int(plan.episode): plan for plan in plans}
        if tuple(sorted(by_plan)) != causal.EXPECTED_HELDOUT:
            raise ValueError(f"unexpected heldout plan set: {sorted(by_plan)}")
        records = []
        for episode in causal.EXPECTED_HELDOUT:
            plan = by_plan[episode]
            opposite_episode = bundle.OPPOSITE_EPISODE[episode]
            opposite_plan = by_plan[opposite_episode]
            if opposite_plan.prompt != plan.prompt or opposite_plan.side == plan.side:
                raise ValueError(f"ep{episode} donor is not same-instruction/opposite-side")
            for frame in causal.EXPECTED_WAIT_FRAMES[episode]:
                opposite_frame = self.pairing[(episode, frame)]
                recipient_row = payloads[episode]["by_frame"][frame]
                opposite_row = payloads[opposite_episode]["by_frame"][opposite_frame]
                recipient, _ = self.base._observation(recipient_row, frame, str(plan.prompt))
                donor, _ = self.base._observation(
                    opposite_row, opposite_frame, str(plan.prompt)
                )
                variants = _camera_variants(recipient, donor)
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
                replayed_endpoints = {
                    condition: self.base._score_side(
                        key,
                        noise,
                        variants[condition],
                        reset,
                        str(plan.side),
                        zero_read=False,
                    )
                    for condition in ("reset_native", "reset_swap_all_cameras")
                }
                within_run_replay = _within_run_endpoint_replay(margins, replayed_endpoints)
                if not within_run_replay["passes"]:
                    raise RuntimeError(
                        f"ep{episode} f{frame} failed same-process endpoint replay: "
                        f"{within_run_replay}"
                    )
                if bundle._tree_identity(reset)["sha256"] != reset_identity["sha256"]:
                    raise RuntimeError("reset M0 changed during frozen camera scoring")
                noise_sha256 = causal._array_sha256(noise)
                identities = {
                    condition: bundle._observation_identity(variants[condition])
                    for condition in CONDITIONS
                }
                parent = self.parent_by_cell[(episode, frame)]
                parent_comparison = _historical_parent_comparison(
                    parent, noise_sha256, identities, margins
                )
                if not parent_comparison["identity_contract_passes"]:
                    raise RuntimeError(
                        f"ep{episode} f{frame} differs from the historical parent's exact "
                        f"observation/noise inputs: {parent_comparison}"
                    )
                effects = _camera_effects(margins)
                record = {
                    "episode": episode,
                    "prompt": str(plan.prompt),
                    "truth_side": str(plan.side),
                    "truth_sign": causal._truth_sign(str(plan.side)),
                    "frame": frame,
                    "opposite_episode": opposite_episode,
                    "opposite_side": str(opposite_plan.side),
                    "opposite_frame": opposite_frame,
                    "token_noise_sha256": noise_sha256,
                    "observation_identity": identities,
                    "margins": margins,
                    "effects": effects,
                    "historical_parent_comparison": parent_comparison,
                    "within_run_endpoint_replay": within_run_replay,
                }
                records.append(record)
                print(
                    f"ep{episode} f{frame} native={margins['reset_native']['truth_aligned']:+.4f} "
                    f"top={margins['reset_swap_top']['truth_aligned']:+.4f} "
                    f"left={margins['reset_swap_left_wrist']['truth_aligned']:+.4f} "
                    f"right={margins['reset_swap_right_wrist']['truth_aligned']:+.4f} "
                    f"all={margins['reset_swap_all_cameras']['truth_aligned']:+.4f}",
                    flush=True,
                )

        parent_summary = {
            "schema": self.parent_report["schema"],
            "classification": self.parent_report["acceptance"]["classification"],
            "direct_reset_observation_side_shortcut_evidence_pass": True,
            "image_channel_localized": True,
            "parameter_source": self.args.parameter_source,
        }
        acceptance = _acceptance_summary(records, parent_summary)
        report = causal._strict_json(
            {
                "schema": SCHEMA_VERSION,
                "provenance": provenance,
                "records": records,
                "acceptance": acceptance,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        identities = _finalize_report(self.args.output_dir, report)
        print(json.dumps(acceptance, indent=2), flush=True)
        print(f"wrote {identities['report']['resolved_path']}", flush=True)
        return report


def main(argv: list[str] | None = None) -> None:
    WaitingCameraShortcutEval(_parse_args(argv)).run()


if __name__ == "__main__":
    main()
