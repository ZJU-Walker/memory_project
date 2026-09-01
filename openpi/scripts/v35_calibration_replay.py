"""Reproducible fresh-step-0 preflight and replay collection for v3.5 calibration.

The calibration is circular unless it has a special preflight: ordinary training correctly
refuses to start before ``c`` and ``tau`` are frozen, while those values must be measured from
the exact fresh model that training will use.  This script provides that narrow bypass without
weakening the training lock.

Two CLI stages are intentionally separate:

``preflight``
    Validate the frozen 54-train-episode manifest and train-only normalization protocol, then
    instantiate the registered v3.5 model with the training seed.  Shared leaves are loaded
    through the audited official-base loader, ordinary frozen leaves are cast exactly as in
    training, and the shared ``parameter_tree_sha256`` helper hashes the actual post-load,
    post-cast step-0 tree.  No calibration artifact is required and no optimizer update runs.

``seal``
    Validate exactly one complete replay shard for each of the 54 frozen train stable IDs and
    emit the strict NPZ consumed by ``v35_injection_calibration.py``.  Every shard is bound to
    the preflight tree, official-base source, replay protocol/code, manifest, and dataset
    protocol.  Missing data, duplicate IDs, placeholder dtypes, or mismatched provenance fail
    before an output is written.

This module also exposes :func:`measure_episode_replay`, the model-level collector used by an
external dataset runner.  Dataset-specific frame selection stays external until a frozen
frame-selection artifact exists; silently inventing evidence/decision frames here would make
the calibration irreproducible.  The preflight artifact enumerates those remaining inputs.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import io
import json
from pathlib import Path
import re
from typing import Any
import zipfile

import numpy as np
from numpy import typing as npt

from openpi.shared import project_paths

PREFLIGHT_SCHEMA = "openpi.v35.calibration-preflight.v1"
EPISODE_REPLAY_SCHEMA = "openpi.v35.calibration-episode-replay.v1"
TRAIN_EPISODE_COUNT = 54
OFFICIAL_BASE_URI = "gs://openpi-assets/checkpoints/pi05_base/params"
NORM_PROVENANCE_SCHEMA_VERSION = 2
NORMALIZATION_PROTOCOL = "raw-train-rows-delta-action-horizon-v1"
TRAIN_STORAGE_ROOT_CONTRACT = "memory_project-relative-v1"
TRAIN_STORAGE_SCOPE = "selected train episode parquet, optional videos, plus structural meta files"
LOW_COSINE_MAX = 0.10
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NPZ_KEY_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
_STORAGE_EPISODE_RE = re.compile(r"(?:^|/)episode_(\d{6})\.(parquet|mp4)$")
_QUERY_KINDS = frozenset(("low_cos_query", "synthetic_orthogonal_query"))

# This object, rather than an informal prose comment, is the frozen semantic protocol.  Its
# canonical hash is stored in every artifact and replay shard.
REPLAY_PROTOCOL: dict[str, Any] = {
    "schema_version": 1,
    "population": {
        "episode_count": TRAIN_EPISODE_COUNT,
        "split": "train",
        "membership": "exact included train stable IDs from the byte-frozen manifest",
    },
    "clock": {
        "unit": "one valid sampled transition (15 raw frames)",
        "n_delay": "number of O-only transitions after the E commit and before the D read",
        "decay": "analytic (1-alpha_step)**n_delay on the FP32 delta-output w3",
    },
    "evidence": {
        "frame": "frozen final eligible E frame after the five-raw-frame tail guard",
        "commit": "one E-only pooled-frame direct delta commit from a fresh episode state",
        "clean_raw_retrieved": (
            "all 16 real D query-bank FP32 slots read from M_E before delay decay; production pinning "
            "is applied independently per slot before any aggregation"
        ),
        "mixed_precision_noise": "FP32 post-commit residual v_bar_E-M_E(k_bar_E) returned by the commit",
    },
    "decision": {
        "frame": "frozen strict stationary-D frame reached after the natural O gap",
        "layer8_residual": (
            "per-channel RMS over valid pre-memory layer-8 prefix rows at D; its RMS over channels "
            "equals the valid-token layer-8 residual RMS"
        ),
    },
    "query_noise": {
        "natural": (
            "individual FP32 raw read slots whose cos(h(q),h(k_bar_E)) <= 0.10, with the exact "
            "transformed observation and memory-clock gap recorded"
        ),
        "fallback": "pre-registered synthetic orthogonal-query controls, explicitly labeled; never zero memory",
        "required": "at least one query control and one post-commit residual per episode",
    },
    "augmentation": "disabled",
    "arithmetic": "memory state, commit, hidden-key alignment, raw reads, and exported vectors are FP32",
    "gate": "raw memory_inject_w from the actual step-0 tree; tanh must be 0.5 channelwise",
}

Float32Array = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.integer[Any]]


class ReplayPreflightError(ValueError):
    """Raised when a calibration preflight or replay collection cannot be authenticated."""


@dataclasses.dataclass(frozen=True)
class FrozenDataProtocol:
    production: bool
    manifest_path: Path
    manifest_sha256: str
    split_seed: int
    train_stable_ids: tuple[str, ...]
    train_episode_indices: tuple[int, ...]
    train_episode_frame_counts: tuple[int, ...]
    dataset_repo_id: str
    dataset_num_episodes: int
    dataset_num_frames: int
    dataset_protocol_sha256: str
    train_storage_root: Path
    train_storage_sha256: str
    train_storage_file_count: int
    norm_provenance_path: Path
    norm_provenance_sha256: str
    norm_stats_path: Path
    norm_stats_sha256: str


@dataclasses.dataclass(frozen=True)
class Step0Identity:
    actual_parameter_tree_sha256: str
    official_base_source_sha256: str
    target_schema_sha256: str
    graft_manifest_sha256: str
    raw_gate: Float32Array


@dataclasses.dataclass(frozen=True)
class ReplayBindings:
    step0_parameter_tree_sha256: str
    official_base_source_sha256: str
    replay_protocol_sha256: str
    collector_source_sha256: str
    manifest_sha256: str
    dataset_protocol_sha256: str


@dataclasses.dataclass(frozen=True)
class QueryControlFrame:
    frame_index: int
    n_delay: int
    observation: Any


@dataclasses.dataclass(frozen=True)
class EpisodeReplayRecord:
    stable_id: str
    split: str
    evidence_frame_index: int
    decision_frame_index: int
    n_delay: int
    evidence_observation_sha256: str
    decision_observation_sha256: str
    clean_raw_retrieved: Float32Array
    layer8_residual: Float32Array
    mixed_precision_noise_raw_retrieved: Float32Array
    query_noise_raw_retrieved: Float32Array
    query_noise_cosine: Float32Array
    query_noise_kind: tuple[str, ...]
    query_noise_frame_index: IntArray
    query_noise_observation_sha256: tuple[str, ...]
    commit_relative_residual: float
    commit_applied: bool
    bindings: ReplayBindings


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def replay_protocol_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(REPLAY_PROTOCOL))


def collector_source_sha256() -> str:
    return sha256_bytes(Path(__file__).read_bytes())


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ReplayPreflightError(f"{name} must be a lower-case 64-character SHA256 digest")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReplayPreflightError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_strict_json(path: Path) -> tuple[dict[str, Any], bytes]:
    path = Path(path)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReplayPreflightError(f"cannot read {path}: {exc}") from exc
    try:
        value = json.loads(payload, object_pairs_hook=_strict_object)
    except json.JSONDecodeError as exc:
        raise ReplayPreflightError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReplayPreflightError(f"{path} must contain a JSON object")
    return value, payload


def _envelope(payload: dict[str, Any], *, id_prefix: str) -> dict[str, Any]:
    digest = sha256_bytes(canonical_json_bytes(payload))
    return {
        "artifact_sha256": digest,
        f"{id_prefix}_id": f"sha256:{digest}",
        "hash_scope": "SHA256 of canonical_json($.payload)",
        "payload": payload,
    }


def _verify_envelope(artifact: Mapping[str, Any], *, id_prefix: str) -> bool:
    payload = artifact.get("payload")
    digest = artifact.get("artifact_sha256")
    if not isinstance(payload, dict) or not isinstance(digest, str):
        return False
    actual = sha256_bytes(canonical_json_bytes(payload))
    return digest == actual and artifact.get(f"{id_prefix}_id") == f"sha256:{actual}"


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_json_bytes(value) + b"\n"
    try:
        with path.open("xb") as stream:
            stream.write(encoded)
    except FileExistsError as exc:
        raise ReplayPreflightError(f"refusing to overwrite existing artifact: {path}") from exc


def canonical_npz_bytes(arrays: Mapping[str, npt.NDArray[Any]]) -> bytes:
    """Serialize a pickle-free, byte-reproducible NPZ.

    ``numpy.savez`` embeds ZIP member timestamps.  Fixed metadata and sorted keys make this
    producer byte-identical across reruns with identical arrays.
    """
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as archive:
        for key in sorted(arrays):
            if _NPZ_KEY_RE.fullmatch(key) is None:
                raise ReplayPreflightError(f"invalid canonical NPZ key {key!r}")
            array = np.asarray(arrays[key])
            if array.dtype.hasobject:
                raise ReplayPreflightError(f"canonical NPZ key {key!r} has forbidden object dtype")
            npy = io.BytesIO()
            np.lib.format.write_array(npy, array, version=(1, 0), allow_pickle=False)
            info = zipfile.ZipInfo(filename=f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o600 << 16
            archive.writestr(info, npy.getvalue())
    return output.getvalue()


def _write_npz_once(path: Path, arrays: Mapping[str, npt.NDArray[Any]]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_npz_bytes(arrays)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise ReplayPreflightError(f"refusing to overwrite existing NPZ: {path}") from exc
    return sha256_bytes(payload)


def _valid_positive_int(value: Any, *, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ReplayPreflightError(f"{name} must be a positive integer")
    return value


def _validate_recorded_train_storage(
    storage: Any,
    *,
    train_episode_indices: tuple[int, ...],
    production: bool,
) -> tuple[Path, str, int]:
    """Validate the immutable storage-seal declaration without opening media bytes."""
    if not isinstance(storage, dict):
        raise ReplayPreflightError("production norm provenance requires a train_storage seal")
    if production:
        if "root" in storage:
            raise ReplayPreflightError("production train_storage must not contain the legacy machine-local root")
        if storage.get("root_contract") != TRAIN_STORAGE_ROOT_CONTRACT:
            raise ReplayPreflightError(
                f"production train_storage.root_contract must be {TRAIN_STORAGE_ROOT_CONTRACT!r}"
            )
        root_relative = storage.get("root_relative")
        if not isinstance(root_relative, str) or not root_relative:
            raise ReplayPreflightError("production train_storage.root_relative must be a project-relative directory")
        if Path(root_relative).is_absolute() or Path(root_relative).as_posix() != root_relative:
            raise ReplayPreflightError(
                "production train_storage.root_relative must use canonical POSIX relative syntax"
            )
        try:
            root = project_paths.project_path(root_relative)
            canonical_relative = project_paths.project_relative_path(root).as_posix()
        except project_paths.ProjectRootError as exc:
            raise ReplayPreflightError(f"invalid production train_storage.root_relative: {exc}") from exc
        if canonical_relative != root_relative or not root.is_dir():
            raise ReplayPreflightError(
                f"production train_storage.root_relative is missing or non-canonical: {root_relative!r}"
            )
    else:
        # This compatibility branch exists only for isolated synthetic tests.  Production
        # callers default to the portable schema above and cannot opt into it accidentally.
        root_value = storage.get("root")
        if not isinstance(root_value, str) or not root_value:
            raise ReplayPreflightError("non-production legacy train_storage.root must be an absolute directory")
        root = Path(root_value).expanduser().resolve()
        if not Path(root_value).is_absolute() or str(root) != root_value or not root.is_dir():
            raise ReplayPreflightError(f"legacy train_storage.root is missing or not canonical: {root_value!r}")
    if storage.get("selected_episode_indices") != list(train_episode_indices):
        raise ReplayPreflightError("train_storage episode selection is not the frozen train split")
    if storage.get("scope") != TRAIN_STORAGE_SCOPE:
        raise ReplayPreflightError("train_storage has the wrong scope")
    records = storage.get("files")
    if not isinstance(records, list) or not records:
        raise ReplayPreflightError("train_storage.files must be a non-empty file-record list")

    paths: list[str] = []
    data_seen: set[int] = set()
    video_seen: set[int] = set()
    has_meta = False
    selected = set(train_episode_indices)
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise ReplayPreflightError("train_storage file records require exactly path/size/sha256")
        relative = record["path"]
        size = record["size"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or type(size) is not int
            or size < 0
        ):
            raise ReplayPreflightError("train_storage contains an invalid relative path or file size")
        _require_sha256("train_storage file SHA256", record["sha256"])
        paths.append(relative)
        if relative.startswith("meta/"):
            has_meta = True
            continue
        match = _STORAGE_EPISODE_RE.search(relative)
        if match is None:
            raise ReplayPreflightError(f"train_storage has an unrecognized episode file: {relative}")
        episode_index = int(match.group(1))
        if episode_index not in selected:
            raise ReplayPreflightError("train_storage seal contains held-out episode media")
        extension = match.group(2)
        if relative.startswith("data/") and extension == "parquet":
            data_seen.add(episode_index)
        elif relative.startswith("videos/") and extension == "mp4":
            video_seen.add(episode_index)
        else:
            raise ReplayPreflightError(f"train_storage file is in the wrong storage tree: {relative}")
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ReplayPreflightError("train_storage file records must be uniquely path-sorted")
    if data_seen != selected or video_seen not in (set(), selected) or not has_meta:
        raise ReplayPreflightError(
            "train_storage must cover every train parquet, either zero or every train video, and structural meta"
        )
    aggregate = sha256_bytes(canonical_json_bytes(records))
    if storage.get("sha256") != aggregate:
        raise ReplayPreflightError("train_storage aggregate SHA256 is invalid")
    return root, aggregate, len(records)


def validate_frozen_data_protocol(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    norm_provenance_path: Path,
    norm_stats_path: Path,
    expected_repo_id: str,
    expected_split_seed: int,
    production: bool = True,
) -> FrozenDataProtocol:
    """Authenticate the manifest and train-only dataset/norm selection without loading images."""
    resolved_manifest_path = Path(manifest_path).resolve()
    resolved_norm_provenance_path = Path(norm_provenance_path).resolve()
    resolved_norm_stats_path = Path(norm_stats_path).resolve()
    if production:
        for name, path in (
            ("manifest_path", resolved_manifest_path),
            ("norm_provenance_path", resolved_norm_provenance_path),
            ("norm_stats_path", resolved_norm_stats_path),
        ):
            try:
                project_paths.project_relative_path(path)
            except project_paths.ProjectRootError as exc:
                raise ReplayPreflightError(f"production {name} must be inside memory_project: {exc}") from exc
    manifest_sha256 = _require_sha256("manifest_sha256", manifest_sha256)
    manifest, manifest_bytes = _read_strict_json(resolved_manifest_path)
    actual_manifest_sha = sha256_bytes(manifest_bytes)
    if actual_manifest_sha != manifest_sha256:
        raise ReplayPreflightError(f"manifest SHA256 mismatch: expected {manifest_sha256}, found {actual_manifest_sha}")
    _valid_positive_int(manifest.get("schema_version"), name="manifest schema_version")
    manifest_seed = manifest.get("split_seed")
    if manifest_seed is None and isinstance(manifest.get("split"), dict):
        manifest_seed = manifest["split"].get("seed")
    if type(expected_split_seed) is not int or manifest_seed != expected_split_seed:
        raise ReplayPreflightError(
            f"manifest split seed mismatch: expected {expected_split_seed}, found {manifest_seed!r}"
        )
    entries = manifest.get("episodes")
    if not isinstance(entries, list) or not entries:
        raise ReplayPreflightError("manifest must contain a non-empty episodes list")

    stable_ids: set[str] = set()
    converted_indices: set[int] = set()
    train: list[tuple[int, str]] = []
    for record in entries:
        if not isinstance(record, dict):
            raise ReplayPreflightError("manifest episode entries must be objects")
        stable_id = str(record.get("stable_id", "")).strip()
        if not stable_id or stable_id in stable_ids:
            raise ReplayPreflightError(f"manifest stable_id is empty or duplicated: {stable_id!r}")
        stable_ids.add(stable_id)
        include = record.get("include", True)
        if type(include) is not bool:
            raise ReplayPreflightError(f"manifest include flag for {stable_id!r} must be boolean")
        index_value = record.get("episode_index", record.get("lerobot_episode_index"))
        if index_value is not None:
            if type(index_value) is not int or index_value < 0 or index_value in converted_indices:
                raise ReplayPreflightError(f"invalid/duplicate converted episode index for {stable_id!r}")
            converted_indices.add(index_value)
        if include and record.get("split") == "train":
            if index_value is None:
                raise ReplayPreflightError(f"included train episode {stable_id!r} has no converted index")
            train.append((index_value, stable_id))
    train.sort()
    if len(train) != TRAIN_EPISODE_COUNT:
        raise ReplayPreflightError(
            f"frozen manifest must contain exactly {TRAIN_EPISODE_COUNT} included train episodes; got {len(train)}"
        )
    train_indices = tuple(index for index, _ in train)
    train_ids = tuple(stable_id for _, stable_id in train)

    provenance, provenance_bytes = _read_strict_json(resolved_norm_provenance_path)
    if provenance.get("schema_version") != NORM_PROVENANCE_SCHEMA_VERSION:
        raise ReplayPreflightError(f"norm provenance requires schema_version={NORM_PROVENANCE_SCHEMA_VERSION}")
    if provenance.get("status") != "complete" or provenance.get("repo_id") != expected_repo_id:
        raise ReplayPreflightError("norm provenance status/repo_id does not match the registered dataset")
    manifest_info = provenance.get("manifest")
    selection = provenance.get("selection")
    computation = provenance.get("computation")
    norm_info = provenance.get("norm_stats")
    if not all(isinstance(value, dict) for value in (manifest_info, selection, computation, norm_info)):
        raise ReplayPreflightError("norm provenance is missing manifest/selection/computation/norm_stats objects")
    if (
        manifest_info.get("sha256") != manifest_sha256
        or manifest_info.get("active_split") != "train"
        or manifest_info.get("split_seed") != expected_split_seed
    ):
        raise ReplayPreflightError("norm provenance is not bound to the frozen train manifest")
    if production:
        if "path" in manifest_info:
            raise ReplayPreflightError("production norm provenance must not contain a machine-local manifest.path")
        manifest_relative = manifest_info.get("path_relative")
        if not isinstance(manifest_relative, str) or not manifest_relative:
            raise ReplayPreflightError("production norm provenance requires manifest.path_relative")
        if Path(manifest_relative).is_absolute() or Path(manifest_relative).as_posix() != manifest_relative:
            raise ReplayPreflightError("manifest.path_relative must use canonical POSIX relative syntax")
        try:
            recorded_manifest = project_paths.project_path(manifest_relative)
            canonical_manifest_relative = project_paths.project_relative_path(recorded_manifest).as_posix()
        except project_paths.ProjectRootError as exc:
            raise ReplayPreflightError(f"invalid manifest.path_relative: {exc}") from exc
        if canonical_manifest_relative != manifest_relative or recorded_manifest != resolved_manifest_path:
            raise ReplayPreflightError("manifest.path_relative does not identify the supplied frozen manifest")
    frame_counts = selection.get("selected_episode_frame_counts")
    dataset_num_episodes = selection.get("dataset_num_episodes")
    dataset_num_frames = selection.get("dataset_num_frames")
    if (
        type(dataset_num_episodes) is not int
        or dataset_num_episodes != len(converted_indices)
        or converted_indices != set(range(dataset_num_episodes))
        or type(dataset_num_frames) is not int
        or dataset_num_frames <= 0
        or selection.get("selected_num_episodes") != TRAIN_EPISODE_COUNT
        or selection.get("selected_episode_indices") != list(train_indices)
        or selection.get("selected_stable_ids") != list(train_ids)
        or not isinstance(frame_counts, list)
        or len(frame_counts) != TRAIN_EPISODE_COUNT
        or any(type(count) is not int or count <= 0 for count in frame_counts)
        or selection.get("selected_num_frames") != sum(frame_counts)
    ):
        raise ReplayPreflightError("norm provenance selection is not exactly the frozen 54-episode train split")
    dataset_protocol_sha = _require_sha256(
        "dataset_episode_frame_protocol_sha256",
        selection.get("dataset_episode_frame_protocol_sha256"),
    )
    train_storage_root, train_storage_sha, train_storage_file_count = _validate_recorded_train_storage(
        provenance.get("train_storage"), train_episode_indices=train_indices, production=production
    )
    if computation.get("protocol") != NORMALIZATION_PROTOCOL:
        raise ReplayPreflightError(f"norm computation protocol must be exactly {NORMALIZATION_PROTOCOL!r}")
    requested_batch_size = computation.get("requested_batch_size")
    if (
        type(requested_batch_size) is not int
        or requested_batch_size <= 0
        or computation.get("processed_base_rows") != sum(frame_counts)
        or computation.get("drop_last_rows") != 0
        or computation.get("num_batches_including_partial_final_batch")
        != (sum(frame_counts) + requested_batch_size - 1) // requested_batch_size
        or norm_info.get("file") != resolved_norm_stats_path.name
    ):
        raise ReplayPreflightError("norm computation did not process every selected train row exactly once")
    try:
        norm_bytes = resolved_norm_stats_path.read_bytes()
    except OSError as exc:
        raise ReplayPreflightError(f"cannot read norm stats {resolved_norm_stats_path}: {exc}") from exc
    norm_sha = sha256_bytes(norm_bytes)
    if norm_info.get("sha256") != norm_sha:
        raise ReplayPreflightError("norm_stats.json bytes do not match norm provenance")
    return FrozenDataProtocol(
        production=production,
        manifest_path=resolved_manifest_path,
        manifest_sha256=manifest_sha256,
        split_seed=expected_split_seed,
        train_stable_ids=train_ids,
        train_episode_indices=train_indices,
        train_episode_frame_counts=tuple(frame_counts),
        dataset_repo_id=expected_repo_id,
        dataset_num_episodes=dataset_num_episodes,
        dataset_num_frames=dataset_num_frames,
        dataset_protocol_sha256=dataset_protocol_sha,
        train_storage_root=train_storage_root,
        train_storage_sha256=train_storage_sha,
        train_storage_file_count=train_storage_file_count,
        norm_provenance_path=resolved_norm_provenance_path,
        norm_provenance_sha256=sha256_bytes(provenance_bytes),
        norm_stats_path=resolved_norm_stats_path,
        norm_stats_sha256=norm_sha,
    )


def validate_actual_dataset_contract(
    *,
    data_config: Any,
    model_config: Any,
    action_horizon: int,
    expected: FrozenDataProtocol,
) -> str:
    """Run the production loader contract and authenticate the live dataset row protocol.

    ``create_torch_dataset`` is intentional here.  For the registered memory model it calls
    ``data_loader._load_v35_episode_manifest`` through the ordinary episode-table path, which
    validates the frozen collection/part/object/prompt schema, label bytes, independent
    ``d_valid`` and ``e_visibility`` provenance, and deterministic split algorithm.  This
    preflight does not carry a second, weaker definition of those fields.
    """
    import openpi.training.data_loader as data_loader

    if data_config.repo_id != expected.dataset_repo_id:
        raise ReplayPreflightError(
            f"live data config repo_id mismatch: expected {expected.dataset_repo_id!r}, found {data_config.repo_id!r}"
        )
    dataset = data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    return _validate_actual_dataset_rows(dataset, expected=expected)


def _validate_actual_dataset_rows(dataset: Any, *, expected: FrozenDataProtocol) -> str:
    """Authenticate raw episode rows after production dataset construction succeeds."""
    inner = dataset
    visited: set[int] = set()
    while not hasattr(inner, "hf_dataset"):
        identity = id(inner)
        if identity in visited or not hasattr(inner, "_dataset"):
            raise ReplayPreflightError("v3.5 calibration requires a real LeRobot dataset")
        visited.add(identity)
        inner = inner._dataset  # noqa: SLF001
    columns = inner.hf_dataset.with_format(None)
    if "episode_index" not in columns.column_names:
        raise ReplayPreflightError("live v3.5 dataset is missing its episode_index column")
    episode = np.asarray(columns["episode_index"])
    if episode.ndim != 1 or len(episode) != len(dataset) or not np.issubdtype(episode.dtype, np.integer):
        raise ReplayPreflightError(
            "live v3.5 episode_index must be one integer per dataset row; "
            f"got shape={episode.shape}, rows={len(dataset)}, dtype={episode.dtype}"
        )
    episode = episode.astype(np.int64, copy=False)
    if np.any(episode < 0):
        raise ReplayPreflightError("live v3.5 episode_index contains a negative value")
    unique = np.unique(episode)
    expected_indices = np.arange(expected.dataset_num_episodes, dtype=np.int64)
    if not np.array_equal(unique, expected_indices) or len(episode) != expected.dataset_num_frames:
        raise ReplayPreflightError(
            "live dataset episode/frame population differs from norm provenance: "
            f"episodes={unique.tolist()}, frames={len(episode)}"
        )

    manifest, _ = _read_strict_json(expected.manifest_path)
    records: dict[int, dict[str, Any]] = {}
    for record in manifest.get("episodes", []):
        if not isinstance(record, dict):
            raise ReplayPreflightError("manifest episode entries must be objects")
        index = record.get("episode_index", record.get("lerobot_episode_index"))
        if index is None:
            continue
        if type(index) is not int or index in records:
            raise ReplayPreflightError("manifest has an invalid or duplicate converted episode index")
        records[index] = record
    if set(records) != set(expected_indices.tolist()):
        raise ReplayPreflightError("manifest converted episodes differ from the live dataset population")

    episode_protocol = [
        {
            "episode_index": index,
            "stable_id": str(records[index]["stable_id"]).strip(),
            "split": records[index]["split"],
            "include": records[index].get("include", True),
            "frame_count": int(np.count_nonzero(episode == index)),
        }
        for index in sorted(records)
    ]
    actual_protocol_sha = sha256_bytes(
        canonical_json_bytes(
            {
                "manifest_sha256": expected.manifest_sha256,
                "episodes": episode_protocol,
                "train_storage_sha256": expected.train_storage_sha256,
            }
        )
    )
    if actual_protocol_sha != expected.dataset_protocol_sha256:
        raise ReplayPreflightError(
            "live dataset episode/frame protocol hash differs from train-only norm provenance: "
            f"expected {expected.dataset_protocol_sha256}, found {actual_protocol_sha}"
        )
    live_train = tuple(
        (item["episode_index"], item["stable_id"], item["frame_count"])
        for item in episode_protocol
        if item["include"] is True and item["split"] == "train"
    )
    expected_train = tuple(
        zip(
            expected.train_episode_indices,
            expected.train_stable_ids,
            expected.train_episode_frame_counts,
            strict=True,
        )
    )
    if live_train != expected_train:
        raise ReplayPreflightError("live dataset train membership/frame counts differ from norm provenance")
    return actual_protocol_sha


def validate_production_train_storage(config: Any, *, expected: FrozenDataProtocol) -> str:
    """Use training's authoritative byte-level storage verifier on the preflight inputs."""
    # Script-local import resolves to openpi/scripts/train.py under the supported CLI launch.
    import train as train_script

    provenance, _ = _read_strict_json(expected.norm_provenance_path)
    actual = train_script._validate_v35_train_storage_seal(  # noqa: SLF001
        config,
        provenance,
        selected_episode_indices=list(expected.train_episode_indices),
    )
    if actual != expected.train_storage_sha256:
        raise ReplayPreflightError("production train-storage verifier returned a digest different from norm provenance")
    return actual


def _verify_graft_manifest(path: Path) -> dict[str, Any]:
    graft, _ = _read_strict_json(path)
    recorded = graft.get("manifest_sha256")
    unsigned = {key: value for key, value in graft.items() if key != "manifest_sha256"}
    actual = sha256_bytes(canonical_json_bytes(unsigned))
    if recorded != actual:
        raise ReplayPreflightError("audited official-base graft manifest self-hash is invalid")
    tree_hashes = graft.get("tree_hashes")
    if not isinstance(tree_hashes, dict):
        raise ReplayPreflightError("audited graft manifest has no tree_hashes object")
    _require_sha256("official source tree SHA256", tree_hashes.get("source_sha256"))
    _require_sha256("target schema SHA256", tree_hashes.get("target_schema_sha256"))
    return graft


def _initialize_actual_step0_state(config: Any, *, graft_manifest_path: Path) -> tuple[Any, Step0Identity]:
    """Return the exact train step-0 state and its audited identity.

    This is shared by calibration preflight and the write-once step-0 bootstrap.  Keeping one
    implementation is important: both artifacts must describe the same post-load, post-cast
    parameter tree.  The caller owns the returned (potentially device-resident) state.  This
    function deliberately bypasses only the c/tau training lock and never constructs a data
    loader or performs an optimizer update.
    """
    import jax

    # Script-local import resolves to openpi/scripts/train.py both under CLI and pytest.
    import train as train_script

    import openpi.training.sharding as sharding
    import openpi.training.weight_loaders as weight_loaders

    if getattr(config.model, "memory_v35_calibrated", False):
        raise ReplayPreflightError("preflight must use the registered uncalibrated v3.5 model config")
    loader = config.weight_loader
    if not isinstance(loader, weight_loaders.AuditedPartialCheckpointWeightLoader):
        raise ReplayPreflightError("preflight requires AuditedPartialCheckpointWeightLoader")
    if loader.params_path != OFFICIAL_BASE_URI or loader.manifest_output_path != str(graft_manifest_path):
        raise ReplayPreflightError("preflight loader is not bound to the official base and requested graft manifest")

    # This is the same split used by train.main.  Calling init_train_state directly is the
    # narrow calibration bypass: it performs no readiness check and no optimizer update.
    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    state, _ = train_script.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    train_script._validate_v35_initialized_gate(config, state.params)  # noqa: SLF001
    actual_tree_sha = weight_loaders.parameter_tree_sha256(state.params.to_pure_dict())

    gate_leaves = state.params.filter(train_script.MEMORY_INJECT_GATE_FILTER).flat_state()
    if len(gate_leaves) != 1:
        raise ReplayPreflightError(f"expected one memory_inject_w leaf, found {len(gate_leaves)}")
    raw_gate = np.asarray(jax.device_get(next(iter(gate_leaves.values())).value))
    if raw_gate.dtype != np.float32 or raw_gate.ndim != 1 or not np.all(np.isfinite(raw_gate)):
        raise ReplayPreflightError("actual step-0 memory_inject_w is not one finite FP32 vector")

    graft = _verify_graft_manifest(graft_manifest_path)
    identity = Step0Identity(
        actual_parameter_tree_sha256=actual_tree_sha,
        official_base_source_sha256=graft["tree_hashes"]["source_sha256"],
        target_schema_sha256=graft["tree_hashes"]["target_schema_sha256"],
        graft_manifest_sha256=graft["manifest_sha256"],
        raw_gate=np.asarray(raw_gate, dtype=np.float32),
    )
    return state, identity


def _initialize_actual_step0(config: Any, *, graft_manifest_path: Path) -> Step0Identity:
    """Run the exact train initialization path, deliberately bypassing only the c/tau lock."""

    state, identity = _initialize_actual_step0_state(config, graft_manifest_path=graft_manifest_path)
    # Materialize no other full tree: parameter_tree_sha256 already hashes one leaf at a time.
    del state
    return identity


def make_preflight_artifact(
    *,
    config_name: str,
    config_seed: int,
    alpha_step: float,
    data: FrozenDataProtocol,
    step0: Step0Identity,
    controls_file: str,
    controls_sha256: str,
) -> dict[str, Any]:
    if len(data.train_stable_ids) != TRAIN_EPISODE_COUNT:
        raise ReplayPreflightError("preflight artifact requires exactly 54 train stable IDs")
    for name, value in (
        ("actual step-0 tree", step0.actual_parameter_tree_sha256),
        ("official base source", step0.official_base_source_sha256),
        ("target schema", step0.target_schema_sha256),
        ("graft manifest", step0.graft_manifest_sha256),
        ("controls", controls_sha256),
    ):
        _require_sha256(name, value)
    gate = np.asarray(step0.raw_gate)
    if gate.dtype != np.float32 or gate.ndim != 1 or not gate.size:
        raise ReplayPreflightError("preflight raw gate must be a non-empty FP32 vector")
    effective = np.tanh(gate.astype(np.float32)).astype(np.float32)
    if not np.allclose(effective, np.float32(0.5), rtol=0.0, atol=1e-6):
        raise ReplayPreflightError("preflight raw gate does not produce tanh(w)=0.5 channelwise")
    protocol_sha = replay_protocol_sha256()
    source_sha = collector_source_sha256()
    if data.production:
        path_fields = {
            "path_contract": TRAIN_STORAGE_ROOT_CONTRACT,
            "manifest_path_relative": project_paths.project_relative_path(data.manifest_path).as_posix(),
            "norm_provenance_path_relative": project_paths.project_relative_path(data.norm_provenance_path).as_posix(),
            "norm_stats_path_relative": project_paths.project_relative_path(data.norm_stats_path).as_posix(),
        }
    else:
        path_fields = {
            "manifest_path": str(data.manifest_path),
            "norm_provenance_path": str(data.norm_provenance_path),
            "norm_stats_path": str(data.norm_stats_path),
        }
    payload = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "init_complete_replay_required",
        "config": {
            "name": config_name,
            "seed": config_seed,
            "alpha_step": float(np.float32(alpha_step)),
            "official_base_uri": OFFICIAL_BASE_URI,
            "initialization_rng": "_, init_rng = jax.random.split(jax.random.key(config.seed))",
            "parameter_state": "post audited load, post training frozen-leaf cast, pre optimizer update",
        },
        "initialization": {
            "actual_step0_parameter_tree_sha256": step0.actual_parameter_tree_sha256,
            "official_base_source_tree_sha256": step0.official_base_source_sha256,
            "target_schema_sha256": step0.target_schema_sha256,
            "graft_manifest_sha256": step0.graft_manifest_sha256,
            "controls_file": controls_file,
            "controls_sha256": controls_sha256,
            "gate_channel_count": int(gate.size),
            "raw_gate_sha256": sha256_bytes(gate.tobytes(order="C")),
            "effective_gate_min": float(np.min(effective)),
            "effective_gate_max": float(np.max(effective)),
        },
        "data": {
            "production": data.production,
            "dataset_repo_id": data.dataset_repo_id,
            "dataset_num_episodes": data.dataset_num_episodes,
            "dataset_num_frames": data.dataset_num_frames,
            "dataset_episode_frame_protocol_sha256": data.dataset_protocol_sha256,
            "train_storage_sha256": data.train_storage_sha256,
            "train_storage_file_count": data.train_storage_file_count,
            "manifest_sha256": data.manifest_sha256,
            "norm_provenance_sha256": data.norm_provenance_sha256,
            "norm_stats_sha256": data.norm_stats_sha256,
            "split": "train",
            "split_seed": data.split_seed,
            "train_episode_count": TRAIN_EPISODE_COUNT,
            "train_episode_indices": list(data.train_episode_indices),
            "train_stable_ids": list(data.train_stable_ids),
            **path_fields,
        },
        "replay": {
            "protocol": REPLAY_PROTOCOL,
            "protocol_sha256": protocol_sha,
            "collector_source_file": Path(__file__).name,
            "collector_source_sha256": source_sha,
            "remaining_external_artifacts": [
                "the converted LeRobot dataset whose episode/frame protocol matches the recorded dataset hash",
                "a frozen per-episode frame-selection artifact naming final eligible E, strict stationary D, and query-control frames",
                "54 complete episode replay shards produced from this exact step-0 tree with augmentation disabled",
                "pre-registered synthetic orthogonal-query controls for any episode with no real query slot at cosine <= 0.10",
            ],
        },
    }
    return _envelope(payload, id_prefix="preflight")


def read_preflight(path: Path) -> dict[str, Any]:
    artifact, _ = _read_strict_json(path)
    if not _verify_envelope(artifact, id_prefix="preflight"):
        raise ReplayPreflightError("preflight payload hash/ID is invalid")
    payload = artifact["payload"]
    if payload.get("schema_version") != PREFLIGHT_SCHEMA or payload.get("status") != "init_complete_replay_required":
        raise ReplayPreflightError("preflight has the wrong schema or status")
    if payload.get("replay", {}).get("protocol_sha256") != replay_protocol_sha256():
        raise ReplayPreflightError("preflight replay protocol does not match this collector")
    if payload.get("replay", {}).get("collector_source_sha256") != collector_source_sha256():
        raise ReplayPreflightError("preflight was produced by different collector source bytes")
    data = payload.get("data")
    if not isinstance(data, dict) or type(data.get("production")) is not bool:
        raise ReplayPreflightError("preflight must explicitly identify production versus synthetic data")
    legacy_path_fields = {"manifest_path", "norm_provenance_path", "norm_stats_path"}
    relative_path_fields = {
        "manifest_path_relative",
        "norm_provenance_path_relative",
        "norm_stats_path_relative",
    }
    if data["production"]:
        if legacy_path_fields & data.keys():
            raise ReplayPreflightError("production preflight contains forbidden machine-local paths")
        if data.get("path_contract") != TRAIN_STORAGE_ROOT_CONTRACT:
            raise ReplayPreflightError("production preflight has the wrong project-relative path contract")
        for field in relative_path_fields:
            value = data.get(field)
            if (
                not isinstance(value, str)
                or not value
                or Path(value).is_absolute()
                or Path(value).as_posix() != value
                or ".." in Path(value).parts
            ):
                raise ReplayPreflightError(f"production preflight has invalid {field}")
    else:
        if relative_path_fields & data.keys() or "path_contract" in data:
            raise ReplayPreflightError("synthetic preflight mixes production and legacy path contracts")
        for field in legacy_path_fields:
            value = data.get(field)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ReplayPreflightError(f"synthetic preflight requires an absolute legacy {field}")
    return artifact


def bindings_from_preflight(artifact: Mapping[str, Any]) -> ReplayBindings:
    if not _verify_envelope(artifact, id_prefix="preflight"):
        raise ReplayPreflightError("cannot derive bindings from an invalid preflight")
    payload = artifact["payload"]
    initialization = payload["initialization"]
    data = payload["data"]
    replay = payload["replay"]
    return ReplayBindings(
        step0_parameter_tree_sha256=_require_sha256(
            "step0 parameter tree", initialization["actual_step0_parameter_tree_sha256"]
        ),
        official_base_source_sha256=_require_sha256(
            "official base source", initialization["official_base_source_tree_sha256"]
        ),
        replay_protocol_sha256=_require_sha256("replay protocol", replay["protocol_sha256"]),
        collector_source_sha256=_require_sha256("collector source", replay["collector_source_sha256"]),
        manifest_sha256=_require_sha256("manifest", data["manifest_sha256"]),
        dataset_protocol_sha256=_require_sha256("dataset protocol", data["dataset_episode_frame_protocol_sha256"]),
    )


def _float32_vector(value: Any, *, name: str, channels: int | None = None) -> Float32Array:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 1 or not array.size:
        raise ReplayPreflightError(f"{name} must be one non-empty FP32 vector")
    if channels is not None and array.shape != (channels,):
        raise ReplayPreflightError(f"{name} must have shape ({channels},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ReplayPreflightError(f"{name} contains NaN or infinite values")
    return array


def _float32_clean_slots(value: Any) -> Float32Array:
    array = np.asarray(value)
    if array.dtype != np.float32 or array.ndim != 2 or array.shape[0] != 16 or not array.shape[1]:
        raise ReplayPreflightError(
            f"clean_raw_retrieved must preserve all real FP32 slots with shape (16,D); got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ReplayPreflightError("clean_raw_retrieved contains NaN or infinite values")
    return array


def validate_episode_record(record: EpisodeReplayRecord) -> int:
    if not record.stable_id.strip() or record.split != "train":
        raise ReplayPreflightError("episode replay requires a non-empty stable ID and split='train'")
    if (
        type(record.evidence_frame_index) is not int
        or type(record.decision_frame_index) is not int
        or record.evidence_frame_index < 0
        or record.decision_frame_index <= record.evidence_frame_index
    ):
        raise ReplayPreflightError("episode replay has invalid evidence/decision frame indices")
    if type(record.n_delay) is not int or record.n_delay < 0:
        raise ReplayPreflightError("episode replay n_delay must be a non-negative integer")
    for name, digest in (
        ("evidence observation", record.evidence_observation_sha256),
        ("decision observation", record.decision_observation_sha256),
        ("step0 parameter tree", record.bindings.step0_parameter_tree_sha256),
        ("official base source", record.bindings.official_base_source_sha256),
        ("replay protocol", record.bindings.replay_protocol_sha256),
        ("collector source", record.bindings.collector_source_sha256),
        ("manifest", record.bindings.manifest_sha256),
        ("dataset protocol", record.bindings.dataset_protocol_sha256),
    ):
        _require_sha256(name, digest)
    clean = _float32_clean_slots(record.clean_raw_retrieved)
    channels = clean.shape[1]
    if float(np.sqrt(np.mean(np.square(clean, dtype=np.float32), dtype=np.float32))) <= 1e-12:
        raise ReplayPreflightError("clean_raw_retrieved is exact zero and cannot be a committed clean read")
    residual = _float32_vector(record.layer8_residual, name="layer8_residual", channels=channels)
    if np.any(residual < 0) or float(np.linalg.norm(residual)) <= 1e-12:
        raise ReplayPreflightError("layer8_residual must be a nonzero per-channel RMS profile")
    _float32_vector(
        record.mixed_precision_noise_raw_retrieved,
        name="mixed_precision_noise_raw_retrieved",
        channels=channels,
    )
    query = np.asarray(record.query_noise_raw_retrieved)
    if query.dtype != np.float32 or query.ndim != 2 or query.shape[0] < 1 or query.shape[1] != channels:
        raise ReplayPreflightError(f"query_noise_raw_retrieved must have FP32 shape (K,{channels}) with K>=1")
    if not np.all(np.isfinite(query)):
        raise ReplayPreflightError("query_noise_raw_retrieved contains NaN or infinite values")
    k = query.shape[0]
    cosine = np.asarray(record.query_noise_cosine)
    if cosine.dtype != np.float32 or cosine.shape != (k,) or not np.all(np.isfinite(cosine)):
        raise ReplayPreflightError("query_noise_cosine must be one finite FP32 value per query control")
    if np.any(cosine < -1.0) or np.any(cosine > LOW_COSINE_MAX):
        raise ReplayPreflightError("query noise controls must satisfy cosine in [-1, 0.10]")
    if len(record.query_noise_kind) != k or set(record.query_noise_kind) - _QUERY_KINDS:
        raise ReplayPreflightError("query_noise_kind must explicitly label every natural/synthetic control")
    frames = np.asarray(record.query_noise_frame_index)
    if not np.issubdtype(frames.dtype, np.integer) or frames.shape != (k,) or np.any(frames < -1):
        raise ReplayPreflightError("query_noise_frame_index must contain K integers >= -1")
    if len(record.query_noise_observation_sha256) != k:
        raise ReplayPreflightError("query noise observation hashes must have length K")
    for index, (kind, frame, digest) in enumerate(
        zip(record.query_noise_kind, frames, record.query_noise_observation_sha256, strict=True)
    ):
        _require_sha256(f"query observation {index}", digest)
        if (kind == "synthetic_orthogonal_query") != (int(frame) == -1):
            raise ReplayPreflightError("synthetic controls require frame=-1; real low-cos controls require a frame")
    if type(record.commit_applied) is not bool or not record.commit_applied:
        raise ReplayPreflightError("episode replay requires an actually applied direct E commit")
    if not np.isfinite(record.commit_relative_residual) or record.commit_relative_residual < 0:
        raise ReplayPreflightError("commit_relative_residual must be finite and non-negative")
    return channels


def episode_record_arrays(record: EpisodeReplayRecord) -> dict[str, npt.NDArray[Any]]:
    validate_episode_record(record)
    return {
        "schema_version": np.asarray(EPISODE_REPLAY_SCHEMA),
        "stable_id": np.asarray(record.stable_id),
        "split": np.asarray(record.split),
        "evidence_frame_index": np.asarray(record.evidence_frame_index, dtype=np.int64),
        "decision_frame_index": np.asarray(record.decision_frame_index, dtype=np.int64),
        "n_delay": np.asarray(record.n_delay, dtype=np.int32),
        "evidence_observation_sha256": np.asarray(record.evidence_observation_sha256),
        "decision_observation_sha256": np.asarray(record.decision_observation_sha256),
        "clean_raw_retrieved": np.asarray(record.clean_raw_retrieved, dtype=np.float32),
        "layer8_residual": np.asarray(record.layer8_residual, dtype=np.float32),
        "mixed_precision_noise_raw_retrieved": np.asarray(record.mixed_precision_noise_raw_retrieved, dtype=np.float32),
        "query_noise_raw_retrieved": np.asarray(record.query_noise_raw_retrieved, dtype=np.float32),
        "query_noise_cosine": np.asarray(record.query_noise_cosine, dtype=np.float32),
        "query_noise_kind": np.asarray(record.query_noise_kind),
        "query_noise_frame_index": np.asarray(record.query_noise_frame_index, dtype=np.int64),
        "query_noise_observation_sha256": np.asarray(record.query_noise_observation_sha256),
        "commit_relative_residual": np.asarray(record.commit_relative_residual, dtype=np.float32),
        "commit_applied": np.asarray(record.commit_applied, dtype=np.bool_),
        "step0_parameter_tree_sha256": np.asarray(record.bindings.step0_parameter_tree_sha256),
        "official_base_source_sha256": np.asarray(record.bindings.official_base_source_sha256),
        "replay_protocol_sha256": np.asarray(record.bindings.replay_protocol_sha256),
        "collector_source_sha256": np.asarray(record.bindings.collector_source_sha256),
        "manifest_sha256": np.asarray(record.bindings.manifest_sha256),
        "dataset_protocol_sha256": np.asarray(record.bindings.dataset_protocol_sha256),
    }


def write_episode_replay_shard(path: Path, record: EpisodeReplayRecord) -> str:
    """Write one complete, deterministic replay shard; partial records have no representation."""
    return _write_npz_once(path, episode_record_arrays(record))


def _npz_text_scalar(archive: Any, key: str) -> str:
    array = np.asarray(archive[key])
    if array.size != 1 or array.dtype.kind not in ("S", "U"):
        raise ReplayPreflightError(f"replay shard {key} must be one string scalar")
    value = array.reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _npz_int_scalar(archive: Any, key: str) -> int:
    array = np.asarray(archive[key])
    if array.size != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ReplayPreflightError(f"replay shard {key} must be one integer scalar")
    return int(array.reshape(-1)[0])


def load_episode_replay_shard(path: Path) -> tuple[EpisodeReplayRecord, str]:
    path = Path(path)
    try:
        payload = path.read_bytes()
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            expected = {
                "schema_version",
                "stable_id",
                "split",
                "evidence_frame_index",
                "decision_frame_index",
                "n_delay",
                "evidence_observation_sha256",
                "decision_observation_sha256",
                "clean_raw_retrieved",
                "layer8_residual",
                "mixed_precision_noise_raw_retrieved",
                "query_noise_raw_retrieved",
                "query_noise_cosine",
                "query_noise_kind",
                "query_noise_frame_index",
                "query_noise_observation_sha256",
                "commit_relative_residual",
                "commit_applied",
                "step0_parameter_tree_sha256",
                "official_base_source_sha256",
                "replay_protocol_sha256",
                "collector_source_sha256",
                "manifest_sha256",
                "dataset_protocol_sha256",
            }
            if set(archive.files) != expected:
                raise ReplayPreflightError(
                    f"replay shard keys differ from schema: missing={sorted(expected - set(archive.files))}, "
                    f"extra={sorted(set(archive.files) - expected)}"
                )
            if _npz_text_scalar(archive, "schema_version") != EPISODE_REPLAY_SCHEMA:
                raise ReplayPreflightError("replay shard schema_version is invalid")
            kinds = tuple(str(value) for value in np.asarray(archive["query_noise_kind"]).tolist())
            observation_hashes = tuple(
                str(value) for value in np.asarray(archive["query_noise_observation_sha256"]).tolist()
            )
            commit_array = np.asarray(archive["commit_applied"])
            if commit_array.size != 1 or commit_array.dtype != np.bool_:
                raise ReplayPreflightError("replay shard commit_applied must be one bool scalar")
            residual_scalar = np.asarray(archive["commit_relative_residual"])
            if residual_scalar.size != 1 or not np.issubdtype(residual_scalar.dtype, np.floating):
                raise ReplayPreflightError("replay shard commit_relative_residual must be one float scalar")
            record = EpisodeReplayRecord(
                stable_id=_npz_text_scalar(archive, "stable_id"),
                split=_npz_text_scalar(archive, "split"),
                evidence_frame_index=_npz_int_scalar(archive, "evidence_frame_index"),
                decision_frame_index=_npz_int_scalar(archive, "decision_frame_index"),
                n_delay=_npz_int_scalar(archive, "n_delay"),
                evidence_observation_sha256=_npz_text_scalar(archive, "evidence_observation_sha256"),
                decision_observation_sha256=_npz_text_scalar(archive, "decision_observation_sha256"),
                clean_raw_retrieved=np.asarray(archive["clean_raw_retrieved"]),
                layer8_residual=np.asarray(archive["layer8_residual"]),
                mixed_precision_noise_raw_retrieved=np.asarray(archive["mixed_precision_noise_raw_retrieved"]),
                query_noise_raw_retrieved=np.asarray(archive["query_noise_raw_retrieved"]),
                query_noise_cosine=np.asarray(archive["query_noise_cosine"]),
                query_noise_kind=kinds,
                query_noise_frame_index=np.asarray(archive["query_noise_frame_index"]),
                query_noise_observation_sha256=observation_hashes,
                commit_relative_residual=float(residual_scalar.reshape(-1)[0]),
                commit_applied=bool(commit_array.reshape(-1)[0]),
                bindings=ReplayBindings(
                    step0_parameter_tree_sha256=_npz_text_scalar(archive, "step0_parameter_tree_sha256"),
                    official_base_source_sha256=_npz_text_scalar(archive, "official_base_source_sha256"),
                    replay_protocol_sha256=_npz_text_scalar(archive, "replay_protocol_sha256"),
                    collector_source_sha256=_npz_text_scalar(archive, "collector_source_sha256"),
                    manifest_sha256=_npz_text_scalar(archive, "manifest_sha256"),
                    dataset_protocol_sha256=_npz_text_scalar(archive, "dataset_protocol_sha256"),
                ),
            )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ReplayPreflightError):
            raise
        raise ReplayPreflightError(f"cannot load replay shard {path}: {exc}") from exc
    validate_episode_record(record)
    return record, sha256_bytes(payload)


def _tree_array_sha256(tree: Any) -> str:
    """Hash a transformed observation by deterministic pytree leaf position/spec/value."""
    import jax

    digest = hashlib.sha256()
    digest.update(b"openpi-v35-transformed-observation-v1")
    leaves = jax.tree.leaves(tree)
    for index, leaf in enumerate(leaves):
        array = np.asarray(jax.device_get(leaf))
        if array.dtype.hasobject:
            raise ReplayPreflightError("transformed observation contains object dtype")
        for field in (
            str(index).encode(),
            str(array.dtype).encode(),
            canonical_json_bytes(list(array.shape)),
            np.ascontiguousarray(array).tobytes(order="C"),
        ):
            digest.update(len(field).to_bytes(8, "big"))
            digest.update(field)
    return digest.hexdigest()


def verify_replay_model(model: Any, expected_step0_sha256: str) -> str:
    """Hash a live model with the shared train helper before collecting any episode shard."""
    import flax.nnx as nnx

    import openpi.training.weight_loaders as weight_loaders

    expected = _require_sha256("expected step0 tree", expected_step0_sha256)
    actual = weight_loaders.parameter_tree_sha256(nnx.state(model).to_pure_dict())
    if actual != expected:
        raise ReplayPreflightError("live replay model is not the authenticated fresh step-0 parameter tree")
    return actual


def _prepare_early_interface(model: Any, observation: Any, memory_state: Any) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    import openpi.models.model as model_lib

    # Collector observations arrive as host numpy leaves (int64 token ids included); the
    # model methods are typed for JAX arrays, and jnp.asarray also applies the training
    # pipeline's x64-disabled int32 canonicalization.
    observation = jax.tree.map(jnp.asarray, observation)
    preprocessed = model_lib.preprocess_observation(None, observation, train=False)
    # Evidence/decision calls pass batch 1; the query-control path legitimately passes the
    # frozen batch of up to 8. Reject only unbatched [h,w,c] inputs.
    reference = next(iter(preprocessed.images.values()))
    if reference.ndim != 4 or reference.shape[0] < 1:
        raise ReplayPreflightError(
            f"calibration replay requires batched [b,h,w,c] observations, got images {reference.shape}"
        )
    prefix_tokens, prefix_mask, prefix_ar = model.embed_prefix(preprocessed)
    num_img = prefix_mask.shape[1] - model.max_token_len
    return model._v32_prepare_memory_interface(  # noqa: SLF001
        prefix_tokens,
        prefix_mask,
        prefix_ar,
        memory_state,
        top_token_count=num_img // len(preprocessed.images),
        state_token_mask=preprocessed.token_state_mask,
    )


def measure_episode_replay(
    *,
    model: Any,
    bindings: ReplayBindings,
    stable_id: str,
    evidence_frame_index: int,
    evidence_observation: Any,
    decision_frame_index: int,
    decision_observation: Any,
    n_delay: int,
    query_control_frames: Sequence[QueryControlFrame],
) -> EpisodeReplayRecord:
    """Measure one real episode from an already authenticated live fresh-step-0 model.

    Call :func:`verify_replay_model` once immediately before the episode loop.  This function
    never invents frames or synthetic controls: callers must obtain them from a frozen frame
    selection.  If the supplied real frames yield no slot at cosine <= 0.10, collection fails
    and the caller must provide the separately preregistered synthetic-control artifact.
    """
    import jax
    import jax.numpy as jnp

    if type(n_delay) is not int or n_delay < 0:
        raise ReplayPreflightError("n_delay must be a non-negative integer")
    state0 = model.memory.init_state(1)
    evidence = _prepare_early_interface(model, evidence_observation, state0)
    write_tokens = evidence["write_tokens"]
    write_keys, write_values = model.memory.project_kv(write_tokens)
    pooled = model.memory.pool_kv(write_keys, write_values)
    state_e, commit = model.memory.write(state0, write_tokens)
    commit_applied = bool(np.asarray(jax.device_get(commit["commit_applied"]))[0])
    if not commit_applied:
        raise ReplayPreflightError(f"episode {stable_id!r} evidence commit was degenerate/not applied")
    stored_hidden = model.memory.hidden_key(state_e, pooled["pooled_key"][:, None, :])[:, 0, :]

    # Preserve the real 16-slot D query bank.  Reading it from post-E state before the O decay
    # gives the clean signal; the reducer later applies analytic decay and the exact per-slot
    # production pin.  Averaging these reads here would be statistically wrong.
    clean_decision = _prepare_early_interface(model, decision_observation, state_e)
    clean = clean_decision["retrieved"]
    state_d, decay_aux = model.memory.analytic_decay(state_e, n_delay)
    if not bool(np.asarray(jax.device_get(decay_aux["decay_gap_valid"]))[0]):
        raise ReplayPreflightError("analytic decision gap was rejected by the memory core")
    decision = _prepare_early_interface(model, decision_observation, state_d)
    h8 = decision["h8_all"].astype(jnp.float32)
    valid = decision["prefix_mask"].astype(jnp.float32)[..., None]
    valid_tokens = jnp.maximum(jnp.sum(valid, axis=1), 1.0)
    residual_profile = jnp.sqrt(jnp.sum(jnp.square(h8) * valid, axis=1) / valid_tokens)

    query_reads: list[np.ndarray] = []
    query_cosines: list[np.float32] = []
    query_frames: list[int] = []
    query_hashes: list[str] = []
    stored_norm = jnp.linalg.norm(stored_hidden, axis=-1, keepdims=True)
    for control in query_control_frames:
        if type(control.frame_index) is not int or type(control.n_delay) is not int or control.n_delay < 0:
            raise ReplayPreflightError("query control frame/gap metadata is invalid")
        control_state, control_decay = model.memory.analytic_decay(state_e, control.n_delay)
        if not bool(np.asarray(jax.device_get(control_decay["decay_gap_valid"]))[0]):
            raise ReplayPreflightError("query-control analytic gap was rejected")
        prepared = _prepare_early_interface(model, control.observation, control_state)
        query_keys = model.memory.project_q(prepared["read_queries"])
        query_hidden = model.memory.hidden_key(control_state, query_keys)
        cosine = jnp.sum(query_hidden * stored_hidden[:, None, :], axis=-1) / jnp.maximum(
            jnp.linalg.norm(query_hidden, axis=-1) * stored_norm,
            1e-12,
        )
        cosine_host = np.asarray(jax.device_get(cosine), dtype=np.float32)[0]
        retrieved_host = np.asarray(jax.device_get(prepared["retrieved"]), dtype=np.float32)[0]
        selected = np.nonzero(cosine_host <= LOW_COSINE_MAX)[0]
        observation_sha = _tree_array_sha256(control.observation)
        for slot in selected:
            query_reads.append(retrieved_host[slot])
            query_cosines.append(np.float32(cosine_host[slot]))
            query_frames.append(control.frame_index)
            query_hashes.append(observation_sha)
    if not query_reads:
        raise ReplayPreflightError(
            f"episode {stable_id!r} has no real low-cos query slot; a preregistered synthetic "
            "orthogonal-query control shard is required"
        )

    clean_host = np.asarray(jax.device_get(clean), dtype=np.float32)[0]
    residual_host = np.asarray(jax.device_get(residual_profile), dtype=np.float32)[0]
    mixed_host = np.asarray(jax.device_get(commit["post_residual"]), dtype=np.float32)[0]
    relative = float(np.asarray(jax.device_get(commit["relative_commit_residual"]))[0])
    record = EpisodeReplayRecord(
        stable_id=stable_id,
        split="train",
        evidence_frame_index=evidence_frame_index,
        decision_frame_index=decision_frame_index,
        n_delay=n_delay,
        evidence_observation_sha256=_tree_array_sha256(evidence_observation),
        decision_observation_sha256=_tree_array_sha256(decision_observation),
        clean_raw_retrieved=clean_host,
        layer8_residual=residual_host,
        mixed_precision_noise_raw_retrieved=mixed_host,
        query_noise_raw_retrieved=np.stack(query_reads).astype(np.float32),
        query_noise_cosine=np.asarray(query_cosines, dtype=np.float32),
        query_noise_kind=tuple("low_cos_query" for _ in query_reads),
        query_noise_frame_index=np.asarray(query_frames, dtype=np.int64),
        query_noise_observation_sha256=tuple(query_hashes),
        commit_relative_residual=relative,
        commit_applied=commit_applied,
        bindings=bindings,
    )
    validate_episode_record(record)
    return record


def _load_controls(preflight_path: Path, artifact: Mapping[str, Any]) -> tuple[Float32Array, str]:
    initialization = artifact["payload"]["initialization"]
    controls_path = preflight_path.parent / initialization["controls_file"]
    try:
        payload = controls_path.read_bytes()
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != {
                "memory_inject_w",
                "step0_parameter_tree_sha256",
                "official_base_source_sha256",
            }:
                raise ReplayPreflightError("step0 controls NPZ has unexpected keys")
            gate = np.asarray(archive["memory_inject_w"])
            step0 = _npz_text_scalar(archive, "step0_parameter_tree_sha256")
            official = _npz_text_scalar(archive, "official_base_source_sha256")
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ReplayPreflightError):
            raise
        raise ReplayPreflightError(f"cannot load step0 controls {controls_path}: {exc}") from exc
    digest = sha256_bytes(payload)
    if digest != initialization["controls_sha256"]:
        raise ReplayPreflightError("step0 controls bytes do not match preflight")
    if step0 != initialization["actual_step0_parameter_tree_sha256"]:
        raise ReplayPreflightError("step0 controls carry a different parameter-tree hash")
    if official != initialization["official_base_source_tree_sha256"]:
        raise ReplayPreflightError("step0 controls carry a different official-base source hash")
    return _float32_vector(gate, name="memory_inject_w"), digest


def seal_replay_npz(*, preflight_path: Path, shards_dir: Path, output_path: Path) -> str:
    """Seal exactly 74 authenticated episode shards into the strict calibration input NPZ."""
    artifact = read_preflight(preflight_path)
    payload = artifact["payload"]
    expected_bindings = bindings_from_preflight(artifact)
    gate, controls_sha = _load_controls(Path(preflight_path), artifact)
    expected_ids = tuple(payload["data"]["train_stable_ids"])
    if len(expected_ids) != TRAIN_EPISODE_COUNT or len(set(expected_ids)) != TRAIN_EPISODE_COUNT:
        raise ReplayPreflightError("preflight train membership is not exactly 74 unique stable IDs")
    paths = sorted(Path(shards_dir).glob("*.npz"))
    if len(paths) != TRAIN_EPISODE_COUNT:
        raise ReplayPreflightError(
            f"replay sealing requires exactly {TRAIN_EPISODE_COUNT} NPZ shards; found {len(paths)}"
        )
    by_id: dict[str, tuple[EpisodeReplayRecord, str]] = {}
    channels: int | None = None
    for path in paths:
        record, digest = load_episode_replay_shard(path)
        if record.stable_id in by_id:
            raise ReplayPreflightError(f"duplicate replay stable ID {record.stable_id!r}")
        if record.bindings != expected_bindings:
            raise ReplayPreflightError(f"replay shard {record.stable_id!r} has mismatched provenance bindings")
        record_channels = validate_episode_record(record)
        if channels is None:
            channels = record_channels
        elif channels != record_channels:
            raise ReplayPreflightError("replay shards disagree on retrieved/residual channel width")
        by_id[record.stable_id] = (record, digest)
    if set(by_id) != set(expected_ids):
        raise ReplayPreflightError(
            f"replay membership differs from preflight: missing={sorted(set(expected_ids) - set(by_id))}, "
            f"extra={sorted(set(by_id) - set(expected_ids))}"
        )
    if channels != gate.size:
        raise ReplayPreflightError(f"replay channel width {channels} does not match gate width {gate.size}")

    ordered = [by_id[stable_id][0] for stable_id in expected_ids]
    clean = np.stack([record.clean_raw_retrieved for record in ordered]).astype(np.float32)
    residual = np.stack([record.layer8_residual for record in ordered]).astype(np.float32)
    delays = np.asarray([record.n_delay for record in ordered], dtype=np.int32)
    noise: list[np.ndarray] = []
    noise_episode: list[int] = []
    noise_kind: list[str] = []
    noise_cosine: list[np.float32] = []
    for episode_index, record in enumerate(ordered):
        for vector, cosine, kind in zip(
            record.query_noise_raw_retrieved,
            record.query_noise_cosine,
            record.query_noise_kind,
            strict=True,
        ):
            noise.append(vector)
            noise_episode.append(episode_index)
            noise_kind.append(kind)
            noise_cosine.append(np.float32(cosine))
        noise.append(record.mixed_precision_noise_raw_retrieved)
        noise_episode.append(episode_index)
        noise_kind.append("mixed_precision_residual")
        noise_cosine.append(np.float32(np.nan))

    arrays: dict[str, npt.NDArray[Any]] = {
        # Exact strict inputs consumed by v35_injection_calibration.py.
        "episode_stable_id": np.asarray(expected_ids),
        "episode_split": np.asarray(["train"] * TRAIN_EPISODE_COUNT),
        "clean_raw_retrieved": clean,
        "layer8_residual": residual,
        "n_delay": delays,
        "alpha_step": np.asarray(payload["config"]["alpha_step"], dtype=np.float32),
        "memory_inject_w": gate,
        "noise_raw_retrieved": np.stack(noise).astype(np.float32),
        "noise_episode_index": np.asarray(noise_episode, dtype=np.int32),
        "noise_kind": np.asarray(noise_kind),
        "noise_query_cosine": np.asarray(noise_cosine, dtype=np.float32),
        "source_sha256": np.asarray(expected_bindings.step0_parameter_tree_sha256),
        "dataset_sha256": np.asarray(expected_bindings.dataset_protocol_sha256),
        "split_sha256": np.asarray(expected_bindings.manifest_sha256),
        # Transitive provenance committed by input_npz_sha256 in the calibration artifact.
        "official_base_source_sha256": np.asarray(expected_bindings.official_base_source_sha256),
        "replay_protocol_sha256": np.asarray(expected_bindings.replay_protocol_sha256),
        "collector_source_sha256": np.asarray(expected_bindings.collector_source_sha256),
        "preflight_sha256": np.asarray(artifact["artifact_sha256"]),
        "controls_sha256": np.asarray(controls_sha),
        "episode_shard_sha256": np.asarray([by_id[stable_id][1] for stable_id in expected_ids]),
        "evidence_frame_index": np.asarray([record.evidence_frame_index for record in ordered], dtype=np.int64),
        "decision_frame_index": np.asarray([record.decision_frame_index for record in ordered], dtype=np.int64),
        "evidence_observation_sha256": np.asarray([record.evidence_observation_sha256 for record in ordered]),
        "decision_observation_sha256": np.asarray([record.decision_observation_sha256 for record in ordered]),
        "commit_relative_residual": np.asarray(
            [record.commit_relative_residual for record in ordered], dtype=np.float32
        ),
        "commit_applied": np.asarray([record.commit_applied for record in ordered], dtype=np.bool_),
    }
    return _write_npz_once(output_path, arrays)


def _resolve_project_cli_path(path: Path, *, name: str) -> Path:
    """Resolve one production CLI path without accepting a machine-local spelling."""
    raw = Path(path)
    if raw.is_absolute():
        raise ReplayPreflightError(f"production {name} must be relative to memory_project, got {str(raw)!r}")
    if ".." in raw.parts:
        raise ReplayPreflightError(f"production {name} must not escape memory_project, got {str(raw)!r}")
    try:
        return project_paths.project_path(raw)
    except project_paths.ProjectRootError as exc:
        raise ReplayPreflightError(f"invalid production {name}: {exc}") from exc


def run_preflight(
    *,
    config_name: str,
    manifest_path: Path,
    manifest_sha256: str,
    assets_dir: Path,
    output_dir: Path,
    fsdp_devices: int | None = None,
) -> Path:
    """Validate frozen inputs and instantiate the exact audited fresh training tree."""
    project_paths.configure_v35_runtime_environment()
    manifest_path = _resolve_project_cli_path(manifest_path, name="manifest path")
    assets_dir = _resolve_project_cli_path(assets_dir, name="assets directory")
    output_dir = _resolve_project_cli_path(output_dir, name="output directory")
    if output_dir.exists():
        raise ReplayPreflightError(f"refusing to reuse preflight output directory: {output_dir}")

    # Keep the repository's pinned Torch-before-config import order.  Importing config first
    # may initialize TensorFlow before Torch and has caused interpreter crashes on the cluster.
    # isort: off
    import openpi.training.data_loader as _data_loader  # noqa: F401
    import openpi.training.config as config_lib
    # isort: on

    import openpi.training.weight_loaders as weight_loaders

    config = config_lib.get_config(config_name)
    if config_name != "pi05_yam_mem_v35" or not getattr(config.model, "memory_v35_enabled", False):
        raise ReplayPreflightError("calibration preflight is restricted to registered pi05_yam_mem_v35")
    if getattr(config.model, "memory_v35_calibrated", False):
        raise ReplayPreflightError("registered preflight config must remain calibration-locked/uncalibrated")
    if not isinstance(config.weight_loader, weight_loaders.AuditedPartialCheckpointWeightLoader):
        raise ReplayPreflightError("registered v3.5 config no longer uses the audited partial loader")
    if config.weight_loader.params_path != OFFICIAL_BASE_URI:
        raise ReplayPreflightError("registered v3.5 config no longer points to the official Pi0.5 base")

    base = config.data.base_config
    if base is None:
        raise ReplayPreflightError("registered v3.5 data factory has no base config")
    bound_base = dataclasses.replace(
        base,
        memory_episode_manifest_path=str(manifest_path),
        memory_episode_manifest_sha256=manifest_sha256,
        memory_manifest_split="train",
    )
    bound_assets = dataclasses.replace(config.data.assets, assets_dir=str(assets_dir))
    bound_factory = dataclasses.replace(config.data, base_config=bound_base, assets=bound_assets)
    asset_id = bound_assets.asset_id or bound_factory.repo_id
    norm_dir = Path(bound_assets.assets_dir) / str(asset_id)
    data = validate_frozen_data_protocol(
        manifest_path=Path(manifest_path),
        manifest_sha256=manifest_sha256,
        norm_provenance_path=norm_dir / "norm_stats_provenance.json",
        norm_stats_path=norm_dir / "norm_stats.json",
        expected_repo_id=bound_factory.repo_id,
        expected_split_seed=bound_base.memory_manifest_split_seed,
    )
    # Constructing the ordinary dataset is the authoritative schema check.  In particular,
    # an older pending manifest cannot pass by merely presenting 74 strings and a norm hash.
    live_data_config = bound_factory.create(config.assets_dirs, config.model)
    validate_actual_dataset_contract(
        data_config=live_data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        expected=data,
    )
    storage_config = dataclasses.replace(config, data=bound_factory)
    validate_production_train_storage(storage_config, expected=data)
    output_dir.mkdir(parents=True)

    graft_path = output_dir / "initialization_graft_manifest.json"
    bound_loader = dataclasses.replace(config.weight_loader, manifest_output_path=str(graft_path))
    config = dataclasses.replace(
        config,
        data=bound_factory,
        weight_loader=bound_loader,
        exp_name="calibration_preflight",
        checkpoint_base_dir=str(output_dir.parent),
        fsdp_devices=config.fsdp_devices if fsdp_devices is None else fsdp_devices,
        wandb_enabled=False,
        resume=False,
        overwrite=False,
    )
    step0 = _initialize_actual_step0(config, graft_manifest_path=graft_path)
    controls_name = "step0_controls.npz"
    controls_sha = _write_npz_once(
        output_dir / controls_name,
        {
            "memory_inject_w": step0.raw_gate,
            "step0_parameter_tree_sha256": np.asarray(step0.actual_parameter_tree_sha256),
            "official_base_source_sha256": np.asarray(step0.official_base_source_sha256),
        },
    )
    artifact = make_preflight_artifact(
        config_name=config.name,
        config_seed=config.seed,
        alpha_step=config.model.memory.alpha_step,
        data=data,
        step0=step0,
        controls_file=controls_name,
        controls_sha256=controls_sha,
    )
    preflight_path = output_dir / "preflight.json"
    _write_json_once(preflight_path, artifact)
    return preflight_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Every path argument must be relative to memory_project; absolute paths and '..' are rejected.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="Create audited fresh-step-0 preflight artifacts.")
    preflight.add_argument("--config-name", default="pi05_yam_mem_v35")
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.add_argument("--manifest-sha256", required=True)
    preflight.add_argument(
        "--assets-dir",
        type=Path,
        required=True,
        help="Base assets directory containing <asset_id>/norm_stats*.json.",
    )
    preflight.add_argument("--output-dir", type=Path, required=True)
    preflight.add_argument("--fsdp-devices", type=int)

    seal = commands.add_parser("seal", help="Seal 74 authenticated replay shards into calibration NPZ.")
    seal.add_argument("--preflight", type=Path, required=True)
    seal.add_argument("--shards-dir", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        project_paths.configure_v35_runtime_environment()
        if args.command == "preflight":
            path = run_preflight(
                config_name=args.config_name,
                manifest_path=args.manifest,
                manifest_sha256=args.manifest_sha256,
                assets_dir=args.assets_dir,
                output_dir=args.output_dir,
                fsdp_devices=args.fsdp_devices,
            )
            print(f"wrote authenticated preflight: {path}")
        else:
            preflight_path = _resolve_project_cli_path(args.preflight, name="preflight path")
            shards_dir = _resolve_project_cli_path(args.shards_dir, name="shards directory")
            output_path = _resolve_project_cli_path(args.output, name="output path")
            digest = seal_replay_npz(
                preflight_path=preflight_path,
                shards_dir=shards_dir,
                output_path=output_path,
            )
            print(f"wrote calibration replay NPZ: {args.output} (sha256:{digest})")
    except (ReplayPreflightError, project_paths.ProjectRootError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
