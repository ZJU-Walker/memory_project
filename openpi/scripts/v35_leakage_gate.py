"""Reduce frozen per-episode v3.5 features into the preregistered Gate-B decision.

The feature collector is deliberately separate.  This CPU-only reducer accepts a canonical
feature-envelope plus its hash-linked NPZ and the frozen schema-v2 episode manifest.  The NPZ
must contain exactly one already-aggregated row for every one of the 54 training episodes;
development and final-test rows are rejected.

Two and only two probes are gating: final images+state, and fresh-step-0 k_bar+v_bar.  Both
use the same deterministic five-fold episode OOF protocol.  Null labels are permuted inside
collection*object strata and the complete fit/preprocess/predict pipeline is rerun for every
permutation.  Nine other modality probes are reported descriptively and never enter Gate B.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
from pathlib import Path
import sys
from typing import Any

import numpy as np
from numpy import typing as npt

from openpi.shared import project_paths

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import v35_gate_artifacts as artifacts  # noqa: E402

FEATURE_SCHEMA_VERSION = "openpi.v35.leakage-features.v1"
PREPROCESSING_SCHEMA_VERSION = "openpi.v35.leakage-preprocessing.v1"
DECISION_SCHEMA_VERSION = "openpi.v35.leakage-gate-decision.v1"
CRITERIA_VERSION = "openpi.v35.gate-b.rev5.v1"
AGGREGATION_PROTOCOL = "episode_mean_final_d_valid_after_frozen_final_preprocessing.v1"
FOLD_PROTOCOL = "five_fold_sha256_ranked_collection_object_episode_oof.v1"
PRIMARY_FEATURES = (
    "primary_final_images_state",
    "primary_step0_kbar_vbar",
)
DESCRIPTIVE_FEATURES = (
    "descriptive_top_only",
    "descriptive_left_wrist_only",
    "descriptive_right_wrist_only",
    "descriptive_all_images_only",
    "descriptive_state_only",
    "descriptive_prompt_only",
    "descriptive_layer8_only",
    "descriptive_kbar_only",
    "descriptive_vbar_only",
)
ALL_FEATURES = PRIMARY_FEATURES + DESCRIPTIVE_FEATURES
PRIMARY_LABELS = {
    "primary_final_images_state": "joint_final_images_plus_robot_state",
    "primary_step0_kbar_vbar": "fresh_step0_kbar_plus_vbar",
}
PERMUTATIONS_MIN = 1_000
PERMUTATION_SEED = 35
FOLD_SEED = 35
FOLD_COUNT = 5
RIDGE_LAMBDA = 1.0
BONFERRONI_ALPHA = 0.0125
NULL_PERCENTILE = 98.75
POOLED_BALANCED_ACCURACY_MAX = 0.62
BOOTSTRAP_REPEATS = 2_000
BOOTSTRAP_SEED = 3501
_STD_FLOOR = 1e-6

FloatMatrix = npt.NDArray[np.float32]
IntVector = npt.NDArray[np.int64]


class LeakageGateError(artifacts.GateArtifactError):
    """Raised when Gate B cannot make an authenticated decision."""


def _resolve_project_cli_path(path: Path, *, name: str) -> Path:
    """Resolve one production CLI path inside the portable project root."""
    raw = Path(path)
    if raw.is_absolute():
        raise LeakageGateError(f"{name} must be relative to memory_project, got {str(raw)!r}")
    if ".." in raw.parts:
        raise LeakageGateError(f"{name} must not escape memory_project, got {str(raw)!r}")
    try:
        return project_paths.project_path(raw)
    except project_paths.ProjectRootError as exc:
        raise LeakageGateError(f"invalid {name}: {exc}") from exc


@dataclasses.dataclass(frozen=True)
class LeakageDataset:
    stable_ids: tuple[str, ...]
    collections: tuple[str, ...]
    objects: tuple[str, ...]
    labels: IntVector
    features: Mapping[str, FloatMatrix]
    feature_envelope_id: str
    feature_envelope_sha256: str
    feature_npz_sha256: str
    preprocessing_artifact_id: str
    preprocessing_artifact_sha256: str
    preprocessing_protocol_sha256: str
    initialization_parameter_tree_sha256: str
    initialization_manifest_sha256: str
    official_base_source_tree_sha256: str
    feature_protocol_sha256: str


@dataclasses.dataclass(frozen=True)
class _PreparedFold:
    train_indices: IntVector
    test_indices: IntVector
    prediction_operator: np.ndarray


def _text_vector(array: npt.NDArray[Any], name: str) -> tuple[str, ...]:
    if array.ndim != 1 or array.dtype.kind not in ("S", "U"):
        raise LeakageGateError(f"{name} must be a one-dimensional string array")
    values: list[str] = []
    for raw in array:
        if isinstance(raw, bytes):
            try:
                value = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise LeakageGateError(f"{name} contains invalid UTF-8") from exc
        else:
            value = str(raw)
        if not value or value != value.strip():
            raise LeakageGateError(f"{name} contains an empty or noncanonical stable ID")
        values.append(value)
    return tuple(values)


def _feature_matrix(array: npt.NDArray[Any], name: str, rows: int) -> FloatMatrix:
    if array.dtype != np.dtype(np.float32):
        raise LeakageGateError(f"{name} must be float32; got {array.dtype}")
    if array.ndim != 2 or array.shape[0] != rows or array.shape[1] < 1:
        raise LeakageGateError(f"{name} must have shape ({rows}, feature_dim>=1); got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise LeakageGateError(f"{name} contains NaN or infinite values")
    return np.asarray(array, dtype=np.float32)


def load_feature_dataset(
    feature_envelope_path: Path,
    *,
    manifest: artifacts.FrozenManifest,
) -> LeakageDataset:
    envelope_path = Path(feature_envelope_path)
    envelope = artifacts.load_canonical_envelope(envelope_path, schema_version=FEATURE_SCHEMA_VERSION)
    payload = artifacts.require_exact_keys(
        "leakage feature payload",
        envelope["payload"],
        {
            "aggregation_protocol",
            "completed_updates",
            "development_or_final_test_accessed",
            "episode_manifest_sha256",
            "feature_npz",
            "feature_protocol_sha256",
            "initialization_manifest",
            "initialization_parameter_tree_sha256",
            "preprocessing_artifact",
            "population_split",
            "split_assignment_sha256",
        },
    )
    if payload["episode_manifest_sha256"] != manifest.sha256:
        raise LeakageGateError("feature artifact was not extracted from the supplied frozen manifest")
    if payload["split_assignment_sha256"] != manifest.split_assignment_sha256:
        raise LeakageGateError("feature artifact split-assignment hash does not match the frozen manifest")
    if payload["aggregation_protocol"] != AGGREGATION_PROTOCOL:
        raise LeakageGateError(f"feature aggregation must be {AGGREGATION_PROTOCOL!r}")
    if payload["population_split"] != "train" or payload["development_or_final_test_accessed"] is not False:
        raise LeakageGateError("Gate-B features must be train-only while development and final test remain untouched")
    if payload["completed_updates"] != 0:
        raise LeakageGateError("fresh-v3.5 kbar/vbar Gate-B features must come from completed_updates=0")
    initialization_sha256 = artifacts.require_sha256(
        "initialization_parameter_tree_sha256", payload["initialization_parameter_tree_sha256"]
    )
    initialization_identity = artifacts.load_initialization_identity(
        owner_path=envelope_path,
        descriptor=payload["initialization_manifest"],
        manifest=manifest,
        expected_parameter_tree_sha256=initialization_sha256,
    )
    preprocessing_path, preprocessing_file_sha256 = artifacts.resolve_hashed_relative_file(
        owner_path=envelope_path,
        descriptor=payload["preprocessing_artifact"],
        descriptor_name="preprocessing_artifact",
    )
    preprocessing = artifacts.load_canonical_envelope(
        preprocessing_path,
        schema_version=PREPROCESSING_SCHEMA_VERSION,
    )
    preprocessing_payload = artifacts.require_exact_keys(
        "preprocessing payload",
        preprocessing["payload"],
        {
            "episode_manifest_sha256",
            "final_preprocessing",
            "image_preprocessing_sha256",
            "norm_stats_sha256",
            "protocol_sha256",
            "split_assignment_sha256",
            "status",
        },
    )
    if (
        preprocessing_payload["episode_manifest_sha256"] != manifest.sha256
        or preprocessing_payload["split_assignment_sha256"] != manifest.split_assignment_sha256
        or preprocessing_payload["status"] != "frozen"
        or preprocessing_payload["final_preprocessing"] is not True
    ):
        raise LeakageGateError("leakage preprocessing artifact is not the frozen final train protocol")
    for name in ("image_preprocessing_sha256", "norm_stats_sha256", "protocol_sha256"):
        artifacts.require_sha256(f"preprocessing.{name}", preprocessing_payload[name])
    if preprocessing_payload["norm_stats_sha256"] != initialization_identity.artifact_hashes.get("norm_stats_sha256"):
        raise LeakageGateError("leakage preprocessing norm hash does not match the authenticated initialization")
    feature_protocol_sha256 = artifacts.require_sha256("feature_protocol_sha256", payload["feature_protocol_sha256"])
    npz_path, npz_sha256 = artifacts.resolve_hashed_relative_file(
        owner_path=envelope_path,
        descriptor=payload["feature_npz"],
        descriptor_name="feature_npz",
    )

    required_keys = {"episode_stable_id", *ALL_FEATURES}
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys != required_keys:
                raise LeakageGateError(
                    f"feature NPZ keys mismatch: missing={sorted(required_keys - keys)}, "
                    f"extra={sorted(keys - required_keys)}"
                )
            stable_ids = _text_vector(archive["episode_stable_id"], "episode_stable_id")
            if len(stable_ids) != 54 or len(set(stable_ids)) != 54:
                raise LeakageGateError("feature NPZ must contain exactly 54 unique episode rows")
            feature_arrays = {
                name: _feature_matrix(archive[name], name, len(stable_ids)).copy() for name in ALL_FEATURES
            }
    except (OSError, ValueError) as exc:
        if isinstance(exc, LeakageGateError):
            raise
        raise LeakageGateError(f"cannot load feature NPZ {npz_path}: {exc}") from exc

    train_by_id = {episode.stable_id: episode for episode in manifest.split("train")}
    expected_stable_ids = tuple(episode.stable_id for episode in manifest.split("train"))
    if stable_ids != expected_stable_ids:
        missing = sorted(set(expected_stable_ids) - set(stable_ids))
        foreign = sorted(set(stable_ids) - set(expected_stable_ids))
        raise LeakageGateError(
            "feature rows must equal the 54 training episodes in frozen manifest order; "
            f"missing={missing}, non_train_or_unknown={foreign}"
        )
    ordered_episodes = tuple(train_by_id[stable_id] for stable_id in stable_ids)
    labels = np.asarray([episode.target_side for episode in ordered_episodes], dtype=np.int64)
    if np.bincount(labels, minlength=2).min() < 1:
        raise LeakageGateError("training feature population must contain both target sides")
    return LeakageDataset(
        stable_ids=stable_ids,
        collections=tuple(episode.collection for episode in ordered_episodes),
        objects=tuple(episode.object_name for episode in ordered_episodes),
        labels=labels,
        features=feature_arrays,
        feature_envelope_id=envelope["artifact_id"],
        feature_envelope_sha256=artifacts.sha256_bytes(envelope_path.read_bytes()),
        feature_npz_sha256=npz_sha256,
        preprocessing_artifact_id=preprocessing["artifact_id"],
        preprocessing_artifact_sha256=preprocessing_file_sha256,
        preprocessing_protocol_sha256=preprocessing_payload["protocol_sha256"],
        initialization_parameter_tree_sha256=initialization_sha256,
        initialization_manifest_sha256=initialization_identity.file_sha256,
        official_base_source_tree_sha256=initialization_identity.official_source_tree_sha256,
        feature_protocol_sha256=feature_protocol_sha256,
    )


def _stable_rank(stable_id: str, *, seed: int) -> bytes:
    return hashlib.sha256(f"{FOLD_PROTOCOL}|{seed}|{stable_id}".encode()).digest()


def _fold_ids(dataset: LeakageDataset) -> IntVector:
    """Assign folds within collection*object without consulting side labels."""
    by_stratum: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (collection, object_name) in enumerate(zip(dataset.collections, dataset.objects, strict=True)):
        by_stratum[(collection, object_name)].append(index)
    folds = np.full(len(dataset.stable_ids), -1, dtype=np.int64)
    for indices in by_stratum.values():
        ordered = sorted(indices, key=lambda index: _stable_rank(dataset.stable_ids[index], seed=FOLD_SEED))
        for rank, index in enumerate(ordered):
            folds[index] = rank % FOLD_COUNT
    if np.any(folds < 0) or set(folds.tolist()) != set(range(FOLD_COUNT)):
        raise LeakageGateError("deterministic five-fold assignment did not populate every fold")
    return folds


def _fit_ridge_scores(train_x: FloatMatrix, train_y: IntVector, test_x: FloatMatrix) -> np.ndarray:
    if set(train_y.tolist()) != {0, 1}:
        raise LeakageGateError("an OOF training fold lost one target side")
    mean = np.mean(train_x, axis=0, dtype=np.float64)
    std = np.std(train_x, axis=0, dtype=np.float64)
    std = np.where(std >= _STD_FLOOR, std, 1.0)
    standardized_train = (train_x.astype(np.float64) - mean) / std
    standardized_test = (test_x.astype(np.float64) - mean) / std
    train_design = np.concatenate((standardized_train, np.ones((len(train_x), 1))), axis=1)
    test_design = np.concatenate((standardized_test, np.ones((len(test_x), 1))), axis=1)
    targets = train_y.astype(np.float64) * 2.0 - 1.0

    # Use the smaller exact ridge system.  The intercept is included in the design; its small
    # fixed penalty avoids a singular constant-feature corner without data-dependent tuning.
    if train_design.shape[1] <= train_design.shape[0]:
        gram = train_design.T @ train_design
        gram.flat[:: gram.shape[0] + 1] += RIDGE_LAMBDA
        coefficients = np.linalg.solve(gram, train_design.T @ targets)
    else:
        kernel = train_design @ train_design.T
        kernel.flat[:: kernel.shape[0] + 1] += RIDGE_LAMBDA
        coefficients = train_design.T @ np.linalg.solve(kernel, targets)
    return np.asarray(test_design @ coefficients, dtype=np.float64)


def _oof_scores(features: FloatMatrix, labels: IntVector, folds: IntVector) -> np.ndarray:
    return _prepared_oof_scores(_prepare_oof(features, folds), labels)


def _prepare_oof(features: FloatMatrix, folds: IntVector) -> tuple[_PreparedFold, ...]:
    """Factor the label-independent part of every frozen fold exactly once.

    Fold assignment and train-only standardization depend only on frozen manifest fields and
    features.  Ridge coefficients and predictions remain label-dependent and are recomputed
    from each permuted training-label vector via the exact linear solve operator below.
    """
    prepared: list[_PreparedFold] = []
    for fold in range(FOLD_COUNT):
        test_mask = folds == fold
        train_mask = ~test_mask
        if not np.any(test_mask):
            raise LeakageGateError(f"OOF fold {fold} is empty")
        train_indices = np.flatnonzero(train_mask).astype(np.int64)
        test_indices = np.flatnonzero(test_mask).astype(np.int64)
        train_x = features[train_indices]
        test_x = features[test_indices]
        mean = np.mean(train_x, axis=0, dtype=np.float64)
        std = np.std(train_x, axis=0, dtype=np.float64)
        std = np.where(std >= _STD_FLOOR, std, 1.0)
        standardized_train = (train_x.astype(np.float64) - mean) / std
        standardized_test = (test_x.astype(np.float64) - mean) / std
        train_design = np.concatenate((standardized_train, np.ones((len(train_x), 1))), axis=1)
        test_design = np.concatenate((standardized_test, np.ones((len(test_x), 1))), axis=1)
        if train_design.shape[1] <= train_design.shape[0]:
            gram = train_design.T @ train_design
            gram.flat[:: gram.shape[0] + 1] += RIDGE_LAMBDA
            operator = test_design @ np.linalg.solve(gram, train_design.T)
        else:
            kernel = train_design @ train_design.T
            kernel.flat[:: kernel.shape[0] + 1] += RIDGE_LAMBDA
            operator = test_design @ train_design.T @ np.linalg.solve(kernel, np.eye(len(train_x)))
        prepared.append(
            _PreparedFold(
                train_indices=train_indices,
                test_indices=test_indices,
                prediction_operator=np.asarray(operator, dtype=np.float64),
            )
        )
    return tuple(prepared)


def _prepared_oof_scores(prepared: Sequence[_PreparedFold], labels: IntVector) -> np.ndarray:
    scores = np.full(len(labels), np.nan, dtype=np.float64)
    for fold in prepared:
        train_labels = labels[fold.train_indices]
        if set(train_labels.tolist()) != {0, 1}:
            raise LeakageGateError("an OOF training fold lost one target side")
        targets = train_labels.astype(np.float64) * 2.0 - 1.0
        scores[fold.test_indices] = fold.prediction_operator @ targets
    if not np.all(np.isfinite(scores)):
        raise LeakageGateError("OOF pipeline produced non-finite scores")
    return scores


def _balanced_accuracy(labels: IntVector, predictions: IntVector) -> tuple[float, tuple[float, float]]:
    recalls: list[float] = []
    for side in (0, 1):
        mask = labels == side
        if not np.any(mask):
            raise LeakageGateError("balanced accuracy is undefined for a single-side population")
        recalls.append(float(np.mean(predictions[mask] == labels[mask])))
    return float(np.mean(recalls)), (recalls[0], recalls[1])


def _collection_statistic(
    *,
    labels: IntVector,
    predictions: IntVector,
    collections: Sequence[str],
    objects: Sequence[str],
    collection: str,
) -> tuple[float, dict[str, Any]]:
    object_metrics: dict[str, Any] = {}
    values: list[float] = []
    for object_name in sorted(set(objects)):
        mask = np.asarray(
            [c == collection and o == object_name for c, o in zip(collections, objects, strict=True)],
            dtype=bool,
        )
        if not np.any(mask):
            raise LeakageGateError(f"collection/object stratum {(collection, object_name)!r} is empty")
        balanced, recalls = _balanced_accuracy(labels[mask], predictions[mask])
        object_metrics[object_name] = {
            "episode_count": int(np.sum(mask)),
            "balanced_accuracy": balanced,
            "left_recall": recalls[0],
            "right_recall": recalls[1],
        }
        values.append(balanced)
    return float(np.mean(values)), object_metrics


def _metric_bundle(dataset: LeakageDataset, labels: IntVector, scores: np.ndarray) -> dict[str, Any]:
    predictions = (scores >= 0.0).astype(np.int64)
    pooled_balanced, pooled_recalls = _balanced_accuracy(labels, predictions)
    by_collection: dict[str, Any] = {}
    for collection in sorted(set(dataset.collections)):
        statistic, per_object = _collection_statistic(
            labels=labels,
            predictions=predictions,
            collections=dataset.collections,
            objects=dataset.objects,
            collection=collection,
        )
        by_collection[collection] = {
            "object_macro_balanced_accuracy": statistic,
            "objects": per_object,
        }
    return {
        "episode_count": len(labels),
        "pooled_balanced_accuracy": pooled_balanced,
        "pooled_left_recall": pooled_recalls[0],
        "pooled_right_recall": pooled_recalls[1],
        "collections": by_collection,
    }


def _permuted_labels(dataset: LeakageDataset, *, repeats: int, seed: int) -> tuple[IntVector, ...]:
    if repeats < PERMUTATIONS_MIN:
        raise LeakageGateError(f"Gate B requires at least {PERMUTATIONS_MIN} full-pipeline permutations")
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (collection, object_name) in enumerate(zip(dataset.collections, dataset.objects, strict=True)):
        groups[(collection, object_name)].append(index)
    for stratum, indices in groups.items():
        if set(dataset.labels[indices].tolist()) != {0, 1}:
            raise LeakageGateError(f"permutation stratum {stratum!r} does not contain both sides")
    rng = np.random.default_rng(seed)
    outputs: list[IntVector] = []
    for _ in range(repeats):
        shuffled = dataset.labels.copy()
        for indices in groups.values():
            shuffled[indices] = rng.permutation(shuffled[indices])
        outputs.append(shuffled)
    return tuple(outputs)


def _bootstrap_ci(
    dataset: LeakageDataset,
    labels: IntVector,
    predictions: IntVector,
    *,
    collection: str | None,
) -> list[float]:
    groups: dict[tuple[str, str, int], np.ndarray] = {}
    for selected_collection in sorted(set(dataset.collections)):
        if collection is not None and selected_collection != collection:
            continue
        for object_name in sorted(set(dataset.objects)):
            for side in (0, 1):
                indices = np.asarray(
                    [
                        index
                        for index, (item_collection, item_object) in enumerate(
                            zip(dataset.collections, dataset.objects, strict=True)
                        )
                        if item_collection == selected_collection
                        and item_object == object_name
                        and labels[index] == side
                    ],
                    dtype=np.int64,
                )
                if not len(indices):
                    raise LeakageGateError("cannot bootstrap an empty collection/object/side stratum")
                groups[(selected_collection, object_name, side)] = indices
    rng = np.random.default_rng(BOOTSTRAP_SEED + (0 if collection is None else sum(map(ord, collection))))
    values = np.empty(BOOTSTRAP_REPEATS, dtype=np.float64)
    for repeat in range(BOOTSTRAP_REPEATS):
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in groups.values()])
        if collection is None:
            values[repeat] = _balanced_accuracy(labels[sampled], predictions[sampled])[0]
        else:
            sampled_collections = tuple(dataset.collections[index] for index in sampled)
            sampled_objects = tuple(dataset.objects[index] for index in sampled)
            values[repeat] = _collection_statistic(
                labels=labels[sampled],
                predictions=predictions[sampled],
                collections=sampled_collections,
                objects=sampled_objects,
                collection=collection,
            )[0]
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def evaluate_gate_b(
    dataset: LeakageDataset,
    *,
    permutations: int = PERMUTATIONS_MIN,
    permutation_seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    """Run the frozen Gate-B family and return an unhashed decision payload."""
    if permutation_seed != PERMUTATION_SEED:
        raise LeakageGateError(f"Gate B permutation seed is frozen at {PERMUTATION_SEED}")
    folds = _fold_ids(dataset)
    fold_assignment = [
        {"fold": int(fold), "stable_id": stable_id} for stable_id, fold in zip(dataset.stable_ids, folds, strict=True)
    ]
    shuffled_labels = _permuted_labels(dataset, repeats=permutations, seed=permutation_seed)
    primary_results: dict[str, Any] = {}
    family_tests: list[dict[str, Any]] = []

    for feature_name in PRIMARY_FEATURES:
        prepared = _prepare_oof(dataset.features[feature_name], folds)
        observed_scores = _prepared_oof_scores(prepared, dataset.labels)
        observed = _metric_bundle(dataset, dataset.labels, observed_scores)
        null_by_collection = {
            collection: np.empty(permutations, dtype=np.float64) for collection in sorted(set(dataset.collections))
        }
        for repeat, null_labels in enumerate(shuffled_labels):
            # Fixed folds depend only on manifest fields; every permutation still repeats
            # standardization, fitting, and held-out prediction from the shuffled labels.
            null_scores = _prepared_oof_scores(prepared, null_labels)
            null_predictions = (null_scores >= 0.0).astype(np.int64)
            for collection, values in null_by_collection.items():
                values[repeat] = _collection_statistic(
                    labels=null_labels,
                    predictions=null_predictions,
                    collections=dataset.collections,
                    objects=dataset.objects,
                    collection=collection,
                )[0]

        collection_results: dict[str, Any] = {}
        predictions = (observed_scores >= 0.0).astype(np.int64)
        for collection, null_values in null_by_collection.items():
            statistic = observed["collections"][collection]["object_macro_balanced_accuracy"]
            exceedances = int(np.sum(null_values >= statistic))
            p_perm = float((1 + exceedances) / (permutations + 1))
            null_cutoff = float(np.percentile(null_values, NULL_PERCENTILE, method="higher"))
            passes = bool(p_perm > BONFERRONI_ALPHA and statistic <= null_cutoff)
            collection_results[collection] = {
                **observed["collections"][collection],
                "confidence_interval_95": _bootstrap_ci(
                    dataset,
                    dataset.labels,
                    predictions,
                    collection=collection,
                ),
                "null_98_75_percentile_higher": null_cutoff,
                "null_mean": float(np.mean(null_values)),
                "permutation_exceedances": exceedances,
                "permutation_p_value_plus_one": p_perm,
                "passes_family_corrected_test": passes,
            }
            family_tests.append(
                {
                    "probe": PRIMARY_LABELS[feature_name],
                    "collection": collection,
                    "observed_statistic": statistic,
                    "permutation_p_value": p_perm,
                    "passes": passes,
                }
            )

        point_pass = bool(observed["pooled_balanced_accuracy"] <= POOLED_BALANCED_ACCURACY_MAX)
        primary_results[feature_name] = {
            "gating_name": PRIMARY_LABELS[feature_name],
            "pooled": {
                "balanced_accuracy": observed["pooled_balanced_accuracy"],
                "left_recall": observed["pooled_left_recall"],
                "right_recall": observed["pooled_right_recall"],
                "confidence_interval_95": _bootstrap_ci(
                    dataset,
                    dataset.labels,
                    predictions,
                    collection=None,
                ),
                "maximum": POOLED_BALANCED_ACCURACY_MAX,
                "passes_point_gate": point_pass,
            },
            "collections": collection_results,
        }

    descriptive: dict[str, Any] = {}
    for feature_name in DESCRIPTIVE_FEATURES:
        scores = _prepared_oof_scores(_prepare_oof(dataset.features[feature_name], folds), dataset.labels)
        metrics = _metric_bundle(dataset, dataset.labels, scores)
        predictions = (scores >= 0.0).astype(np.int64)
        metrics["pooled_confidence_interval_95"] = _bootstrap_ci(
            dataset,
            dataset.labels,
            predictions,
            collection=None,
        )
        for collection in sorted(set(dataset.collections)):
            metrics["collections"][collection]["confidence_interval_95"] = _bootstrap_ci(
                dataset,
                dataset.labels,
                predictions,
                collection=collection,
            )
        descriptive[feature_name] = metrics

    if len(family_tests) != 4:
        raise AssertionError("the preregistered family must contain exactly four tests")
    family_pass = all(test["passes"] for test in family_tests)
    pooled_pass = all(result["pooled"]["passes_point_gate"] for result in primary_results.values())
    status = "pass" if family_pass and pooled_pass else "fail"
    return {
        "status": status,
        "criteria_version": CRITERIA_VERSION,
        "population": {
            "split": "train",
            "episode_count": len(dataset.stable_ids),
            "stable_ids": list(dataset.stable_ids),
            "collections": {
                collection: sum(item == collection for item in dataset.collections)
                for collection in sorted(set(dataset.collections))
            },
        },
        "protocol": {
            "primary_probe_count": 2,
            "primary_features": [PRIMARY_LABELS[name] for name in PRIMARY_FEATURES],
            "descriptive_features": list(DESCRIPTIVE_FEATURES),
            "fold_protocol": FOLD_PROTOCOL,
            "fold_count": FOLD_COUNT,
            "fold_assignment_sha256": artifacts.sha256_bytes(artifacts.canonical_json_bytes(fold_assignment)),
            "fold_episode_counts": [int(np.sum(folds == fold)) for fold in range(FOLD_COUNT)],
            "fold_seed": FOLD_SEED,
            "ridge_lambda": RIDGE_LAMBDA,
            "fold_feature_standardization": "fit_on_training_episodes_only",
            "permutation_count": permutations,
            "permutation_seed": permutation_seed,
            "permutation_strata": "collection*object",
            "permutation_pipeline": "complete_oof_refit_and_predict_for_every_permutation_no_fixed_prediction_shuffle",
            "permutation_implementation": (
                "cache_label_independent_train_only_standardization_and_exact_ridge_factorization; "
                "recompute_coefficients_and_predictions_from_every_permuted_training-label vector"
            ),
            "permutation_p_value": "(1+count(T_perm>=T_observed))/(B+1)",
            "null_percentile": NULL_PERCENTILE,
            "null_quantile_method": "higher",
            "bonferroni_family_size": 4,
            "per_test_alpha": BONFERRONI_ALPHA,
            "pooled_balanced_accuracy_max": POOLED_BALANCED_ACCURACY_MAX,
            "confidence_intervals_are_gating": False,
            "bootstrap_repeats": BOOTSTRAP_REPEATS,
        },
        "primary": primary_results,
        "four_test_family": family_tests,
        "descriptive": descriptive,
        "gates": {
            "four_test_family_pass": family_pass,
            "two_pooled_point_gates_pass": pooled_pass,
            "passes": status == "pass",
        },
        "decision": (
            "natural_branch_may_continue" if status == "pass" else "stop_natural_branch_recollect_or_name_new_branch"
        ),
    }


def build_decision_artifact(
    dataset: LeakageDataset,
    *,
    manifest: artifacts.FrozenManifest,
    permutations: int = PERMUTATIONS_MIN,
    permutation_seed: int = PERMUTATION_SEED,
) -> dict[str, Any]:
    gate = evaluate_gate_b(dataset, permutations=permutations, permutation_seed=permutation_seed)
    payload = {
        **gate,
        "provenance": {
            "episode_manifest_sha256": manifest.sha256,
            "split_assignment_sha256": manifest.split_assignment_sha256,
            "feature_envelope_id": dataset.feature_envelope_id,
            "feature_envelope_sha256": dataset.feature_envelope_sha256,
            "feature_npz_sha256": dataset.feature_npz_sha256,
            "preprocessing_artifact_id": dataset.preprocessing_artifact_id,
            "preprocessing_artifact_sha256": dataset.preprocessing_artifact_sha256,
            "preprocessing_protocol_sha256": dataset.preprocessing_protocol_sha256,
            "initialization_parameter_tree_sha256": dataset.initialization_parameter_tree_sha256,
            "initialization_manifest_sha256": dataset.initialization_manifest_sha256,
            "official_base_source_tree_sha256": dataset.official_base_source_tree_sha256,
            "feature_protocol_sha256": dataset.feature_protocol_sha256,
        },
    }
    return artifacts.artifact_envelope(DECISION_SCHEMA_VERSION, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Every path argument must be relative to memory_project; absolute paths and '..' are rejected.",
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Frozen schema-v2 70-episode manifest.")
    parser.add_argument("--manifest-sha256", required=True, help="Expected exact manifest byte hash.")
    parser.add_argument("--features", type=Path, required=True, help="Canonical leakage-feature envelope.")
    parser.add_argument("--output", type=Path, required=True, help="New canonical Gate-B decision JSON.")
    parser.add_argument(
        "--permutations",
        type=int,
        default=PERMUTATIONS_MIN,
        help=f"Full-pipeline within-stratum permutations (minimum {PERMUTATIONS_MIN}).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        project_paths.configure_v35_runtime_environment()
        manifest_path = _resolve_project_cli_path(args.manifest, name="manifest path")
        features_path = _resolve_project_cli_path(args.features, name="features path")
        output_path = _resolve_project_cli_path(args.output, name="output path")
        manifest = artifacts.load_frozen_manifest(manifest_path, expected_sha256=args.manifest_sha256)
        dataset = load_feature_dataset(features_path, manifest=manifest)
        decision = build_decision_artifact(
            dataset,
            manifest=manifest,
            permutations=args.permutations,
        )
        artifacts.write_canonical_envelope(
            output_path,
            decision,
            schema_version=DECISION_SCHEMA_VERSION,
        )
    except (
        LeakageGateError,
        artifacts.GateArtifactError,
        project_paths.ProjectRootError,
        FileExistsError,
        OSError,
    ) as exc:
        parser.error(str(exc))
    print(f"Gate B: {decision['payload']['status']} ({decision['artifact_id']}); wrote {args.output}")


if __name__ == "__main__":
    main()
