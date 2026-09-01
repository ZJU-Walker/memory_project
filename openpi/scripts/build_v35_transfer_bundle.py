"""Build and verify a portable, byte-exact v3.5 dataset transfer bundle.

The bundle lives at ``DATASET_ROOT/v35_training_bundle``.  It deliberately does not live
under ``data/``, ``videos/``, or ``meta/``: production v3.5 norm provenance seals those
storage trees, so adding transfer metadata there after normalization would invalidate the
seal.  Only explicitly enumerated JSON provenance, frozen label JSON, and a generated README
are copied.  Dataset parquet, videos, and raw numpy streams are never opened or copied.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import tempfile
from typing import Any
import uuid

BUNDLE_NAME = "v35_training_bundle"
TRANSFER_MANIFEST_NAME = "TRANSFER_MANIFEST.json"
README_NAME = "README.md"
RESTORE_PREFIX = PurePosixPath("restore_tree/project_root")
SCHEMA_VERSION = "openpi.v35.training-transfer-bundle.v1"
DATASET_REPO_ID = "yam/bin_memory_0830_0831_v36_subtask"
EXPECTED_SPLITS = {"train": 54, "development": 8, "final_test": 8}
EXPECTED_INCLUDED_EPISODES = 70
EXPECTED_RAW_RECORDS = 71
EXPECTED_SPLIT_SEED = 36
APPROVAL_PROVENANCE_FILENAMES = (
    "approval_ledger.json",
    "conversion_report_preapproval.json",
    "parquet_payload_inventory.json",
    "release.json",
)
DEFAULT_FROZEN_MANIFEST = PurePosixPath("data/0830_0831_episode_manifest_v36_frozen.json")
DEFAULT_APPROVED_MANIFEST = PurePosixPath("data/0830_0831_episode_manifest_v36_approved.json")
DEFAULT_APPROVAL_PROVENANCE_DIR = PurePosixPath("openpi/diagnostic_outputs/0830_0831_dataset_v36_approval")
DEFAULT_APPROVED_VERIFICATION = PurePosixPath(
    "openpi/diagnostic_outputs/0830_0831_dataset_v36_approved_verification.json"
)
DEFAULT_NORM_RESTORE_DIR = PurePosixPath("v35/assets/pi05_yam_0830_0831_v36/yam/bin_memory_0830_0831_v36_subtask")
PROJECT_DATASET_PREFIX = PurePosixPath("data/lerobot")
TRAIN_STORAGE_ROOT_CONTRACT = "memory_project-relative-v1"
TRAIN_STORAGE_SCOPE = "selected train episode parquet, optional videos, plus structural meta files"
NORM_COMPUTATION_PROTOCOL = "raw-train-rows-delta-action-horizon-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ALLOWED_PAYLOAD_SUFFIXES = frozenset((".json", ".md"))
FORBIDDEN_STORAGE_PARTS = frozenset(("videos",))
FORBIDDEN_STORAGE_SUFFIXES = frozenset((".arrow", ".avi", ".mkv", ".mov", ".mp4", ".npy", ".npz", ".parquet"))


class BundleError(ValueError):
    """Raised when a source or materialized transfer bundle is not fail-closed valid."""


@dataclasses.dataclass(frozen=True)
class BundleInputs:
    dataset_root: Path
    source_project_root: Path
    frozen_manifest: Path
    approved_manifest: Path
    approval_provenance_dir: Path
    approved_verification: Path
    dataset_repo_id: str = DATASET_REPO_ID
    norm_dir: Path | None = None
    norm_restore_dir: PurePosixPath = DEFAULT_NORM_RESTORE_DIR


@dataclasses.dataclass(frozen=True)
class CopyItem:
    source: Path
    destination: PurePosixPath
    role: str
    size: int
    sha256: str

    def record(self) -> dict[str, Any]:
        return {
            "path": self.destination.as_posix(),
            "role": self.role,
            "sha256": self.sha256,
            "size": self.size,
        }


@dataclasses.dataclass(frozen=True)
class BundlePlan:
    dataset_root: Path
    bundle_root: Path
    copy_items: tuple[CopyItem, ...]
    readme_bytes: bytes
    transfer_manifest: Mapping[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical JSON encoding used by the transfer manifest."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise BundleError(f"{name} must be a lower-case 64-character SHA256 digest")
    return value


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"{name} must be a JSON object: {path}")
    return value


def _resolve_existing_file(path: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise BundleError(f"{name} must be a non-symlink regular file: {resolved}")
    return resolved


def _resolve_existing_directory(path: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise BundleError(f"{name} must be a non-symlink directory: {resolved}")
    return resolved


def _safe_relative_path(value: str | PurePosixPath, *, name: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise BundleError(f"{name} must be a normalized relative POSIX path: {value!r}")
    return path


def _project_relative(path: Path, project_root: Path, *, name: str) -> PurePosixPath:
    try:
        relative = path.relative_to(project_root)
    except ValueError as exc:
        raise BundleError(f"{name} must be contained by source project root {project_root}: {path}") from exc
    return _safe_relative_path(PurePosixPath(*relative.parts), name=f"{name} project-relative path")


def _path_from_json(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise BundleError(f"{name} must be a nonempty path string")
    return Path(value).expanduser().resolve()


def _validate_historical_dataset_reference(value: Any, *, dataset_root: Path, name: str) -> None:
    """Accept an immutable pre-relocation root only when its dataset basename still matches.

    The release records predate the project-local copy and intentionally remain byte-exact.
    Their payload inventories and structural-meta hashes authenticate the dataset contents;
    the historical absolute root is therefore an identity hint, not a runtime location.
    """

    historical_root = _path_from_json(value, name=name)
    if historical_root.name != dataset_root.name:
        raise BundleError(f"{name} names a different dataset: {historical_root.name!r}")


def _validate_record_population(raw: Mapping[str, Any], *, name: str) -> list[dict[str, Any]]:
    episodes = raw.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != EXPECTED_RAW_RECORDS:
        raise BundleError(f"{name} must contain exactly {EXPECTED_RAW_RECORDS} raw episode records")
    if any(not isinstance(record, dict) for record in episodes):
        raise BundleError(f"{name} episode records must all be JSON objects")
    typed = list(episodes)
    stable_ids = [record.get("stable_id") for record in typed]
    if any(not isinstance(stable_id, str) or not stable_id for stable_id in stable_ids):
        raise BundleError(f"{name} contains a missing or invalid stable_id")
    if len(set(stable_ids)) != EXPECTED_RAW_RECORDS:
        raise BundleError(f"{name} stable_ids must be unique")
    included = [record for record in typed if bool(record.get("include", True))]
    if len(included) != EXPECTED_INCLUDED_EPISODES:
        raise BundleError(f"{name} must contain exactly {EXPECTED_INCLUDED_EPISODES} included episodes")
    return typed


def _validate_frozen_manifest(
    path: Path,
    *,
    source_project_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    raw = _load_json_object(path, name="frozen manifest")
    if (
        raw.get("schema_version") != 2
        or raw.get("dataset_version") != "v36"
        or raw.get("review_status") != "frozen"
        or raw.get("split_seed") != EXPECTED_SPLIT_SEED
    ):
        raise BundleError("frozen manifest must be the schema-v2 frozen v36 seed-36 manifest")
    records = _validate_record_population(raw, name="frozen manifest")
    included = [record for record in records if bool(record.get("include", True))]
    indices = [record.get("episode_index") for record in included]
    if any(type(index) is not int for index in indices) or sorted(indices) != list(range(EXPECTED_INCLUDED_EPISODES)):
        raise BundleError("frozen manifest included episode_index values must be exactly 0..89")
    split_counts = dict(Counter(record.get("split") for record in included))
    if split_counts != EXPECTED_SPLITS:
        raise BundleError(f"frozen manifest split must be exactly 54/8/8; found {split_counts}")

    raw_root_value = raw.get("raw_root")
    if not isinstance(raw_root_value, str) or not raw_root_value:
        raise BundleError("frozen manifest raw_root must be a nonempty path string")
    raw_root = Path(raw_root_value)
    if not raw_root.is_absolute():
        raw_root = path.parent / raw_root
    if raw_root.resolve() != source_project_root:
        raise BundleError(
            "frozen manifest raw_root must resolve to source_project_root so the unchanged restore tree remains valid"
        )
    return raw, included, split_counts


def _validate_sidecar(
    *,
    manifest_path: Path,
    descriptor: Any,
    descriptor_name: str,
    role: str,
    expected_status: str,
    included_ids: set[str],
) -> tuple[Path, str]:
    if not isinstance(descriptor, dict):
        raise BundleError(f"frozen manifest is missing {descriptor_name} descriptor")
    file_name = descriptor.get("report_file")
    if not isinstance(file_name, str) or Path(file_name).name != file_name:
        raise BundleError(f"{descriptor_name}.report_file must be a basename")
    expected_sha256 = _require_sha256(f"{descriptor_name}.report_sha256", descriptor.get("report_sha256"))
    path = _resolve_existing_file(manifest_path.parent / file_name, name=descriptor_name)
    if _sha256_file(path) != expected_sha256:
        raise BundleError(f"{descriptor_name} bytes do not match the hash pinned by the frozen manifest")
    raw = _load_json_object(path, name=descriptor_name)
    if raw.get("status") != expected_status:
        raise BundleError(f"{descriptor_name} must have status={expected_status!r}")
    if role in ("e_visibility_sidecar", "d_valid_sidecar"):
        entries = raw.get("episodes")
        if not isinstance(entries, list) or len(entries) != EXPECTED_INCLUDED_EPISODES:
            raise BundleError(f"{descriptor_name} must cover exactly 90 episodes")
        sidecar_ids = {
            entry.get("stable_id")
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("stable_id"), str)
        }
        if sidecar_ids != included_ids:
            raise BundleError(f"{descriptor_name} stable IDs do not exactly match the frozen included population")
    return path, expected_sha256


def _validate_approved_manifest(
    path: Path,
    *,
    frozen: Mapping[str, Any],
    frozen_records: Sequence[Mapping[str, Any]],
    source_project_root: Path,
) -> tuple[dict[str, Any], str]:
    raw = _load_json_object(path, name="approved manifest")
    if (
        raw.get("schema_version") != 1
        or raw.get("dataset_version") != "v36"
        or raw.get("review_status") != "user_approved"
    ):
        raise BundleError("approved manifest must be the schema-v1 user-approved v3.5 manifest")
    records = _validate_record_population(raw, name="approved manifest")
    digest = _sha256_file(path)
    source_descriptor = frozen.get("source_approved_manifest")
    if (
        not isinstance(source_descriptor, dict)
        or source_descriptor.get("file") != path.name
        or source_descriptor.get("sha256") != digest
    ):
        raise BundleError("frozen manifest does not hash-pin the supplied approved manifest")

    approved_root_value = raw.get("raw_root")
    if not isinstance(approved_root_value, str) or not approved_root_value:
        raise BundleError("approved manifest raw_root must be a nonempty path string")
    approved_root = Path(approved_root_value)
    if not approved_root.is_absolute():
        approved_root = path.parent / approved_root
    if approved_root.resolve() != source_project_root:
        raise BundleError("approved manifest raw_root does not resolve to source_project_root")

    approved_by_id = {str(record["stable_id"]): record for record in records}
    frozen_by_id = {str(record["stable_id"]): record for record in frozen_records}
    if approved_by_id.keys() != frozen_by_id.keys():
        raise BundleError("approved and frozen manifest populations differ")
    identity_fields = ("include", "raw_dir", "label_file", "label_sha256")
    for stable_id, frozen_record in frozen_by_id.items():
        approved_record = approved_by_id[stable_id]
        if any(approved_record.get(field) != frozen_record.get(field) for field in identity_fields):
            raise BundleError(f"approved/frozen label identity differs for {stable_id}")
    return raw, digest


def _validate_approval_ledger(
    path: Path,
    *,
    frozen: Mapping[str, Any],
    approved_path: Path,
    approved: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    raw = _load_json_object(path, name="approval ledger")
    entries = raw.get("entries")
    if (
        raw.get("schema_version") != "openpi.dataset_approval_ledger.v1"
        or raw.get("reviewer") != "user"
        or not isinstance(entries, list)
        or len(entries) != EXPECTED_RAW_RECORDS
        or any(not isinstance(entry, dict) for entry in entries)
    ):
        raise BundleError("approval ledger is not a complete 71-record user approval")
    if any(entry.get("decision") not in ("user_approved", "user_approved_exclusion") for entry in entries):
        raise BundleError("approval ledger contains a non-approved decision")
    if _path_from_json(raw.get("approved_manifest"), name="approval ledger approved_manifest") != approved_path:
        raise BundleError("approval ledger points at a different approved manifest")
    digest = _sha256_file(path)
    if frozen.get("approval_ledger_sha256") != digest:
        raise BundleError("frozen manifest does not hash-pin the supplied approval ledger")

    approved_by_id = {str(record["stable_id"]): record for record in approved["episodes"]}
    ledger_by_id = {str(entry.get("stable_id", "")): entry for entry in entries}
    if ledger_by_id.keys() != approved_by_id.keys() or "" in ledger_by_id:
        raise BundleError("approval ledger stable IDs do not match the approved manifest")
    for stable_id, record in approved_by_id.items():
        entry = ledger_by_id[stable_id]
        if entry.get("label_sha256") != record.get("label_sha256") or bool(entry.get("include")) != bool(
            record.get("include", True)
        ):
            raise BundleError(f"approval ledger label identity differs for {stable_id}")
    return raw, digest


def _validate_parquet_inventory(path: Path, *, dataset_root: Path) -> tuple[dict[str, Any], str]:
    raw = _load_json_object(path, name="parquet payload inventory")
    records = raw.get("files")
    if (
        raw.get("schema_version") != "openpi.parquet_payload_inventory.v1"
        or raw.get("file_count") != EXPECTED_INCLUDED_EPISODES
        or not isinstance(records, list)
        or len(records) != EXPECTED_INCLUDED_EPISODES
    ):
        raise BundleError("parquet payload inventory must describe exactly 90 files")
    _validate_historical_dataset_reference(
        raw.get("dataset"), dataset_root=dataset_root, name="parquet inventory dataset"
    )
    seen: set[str] = set()
    total_bytes = 0
    for record in records:
        if not isinstance(record, dict):
            raise BundleError("parquet payload inventory records must be objects")
        relative = _safe_relative_path(record.get("path", ""), name="parquet inventory path")
        if relative.suffix != ".parquet" or relative.parts[0] != "data" or relative.as_posix() in seen:
            raise BundleError("parquet inventory contains an invalid or duplicate data path")
        seen.add(relative.as_posix())
        size = record.get("size_bytes")
        if type(size) is not int or size <= 0:
            raise BundleError("parquet inventory sizes must be positive integers")
        _require_sha256("parquet inventory file hash", record.get("sha256"))
        total_bytes += size
    if raw.get("total_bytes") != total_bytes:
        raise BundleError("parquet payload inventory total_bytes is inconsistent")
    return raw, _sha256_file(path)


def _validate_dataset_meta_hashes(
    meta_hashes: Any,
    *,
    dataset_root: Path,
) -> dict[str, str]:
    if not isinstance(meta_hashes, dict) or not meta_hashes:
        raise BundleError("release verification must hash-pin structural dataset meta files")
    validated: dict[str, str] = {}
    for value, expected in sorted(meta_hashes.items()):
        relative = _safe_relative_path(value, name="release meta path")
        if BUNDLE_NAME in relative.parts:
            raise BundleError("release meta inventory must exclude the transfer bundle subtree")
        digest = _require_sha256(f"meta_sha256[{value!r}]", expected)
        path = _resolve_existing_file(dataset_root / "meta" / Path(*relative.parts), name="dataset meta file")
        try:
            path.relative_to(dataset_root / "meta")
        except ValueError as exc:
            raise BundleError(f"dataset meta path escapes meta/: {value}") from exc
        if _sha256_file(path) != digest:
            raise BundleError(f"dataset meta hash mismatch: {value}")
        validated[value] = digest
    return validated


def _validate_release_provenance(
    *,
    release_path: Path,
    conversion_path: Path,
    inventory_path: Path,
    verification_path: Path,
    approved_path: Path,
    approved_sha256: str,
    ledger_path: Path,
    ledger_sha256: str,
    dataset_root: Path,
    dataset_repo_id: str,
) -> None:
    conversion = _load_json_object(conversion_path, name="preapproval conversion report")
    if (
        conversion.get("repo_name") != dataset_repo_id
        or conversion.get("included_episodes") != EXPECTED_INCLUDED_EPISODES
    ):
        raise BundleError("preapproval conversion report does not describe the expected 90-episode dataset")

    inventory, inventory_sha256 = _validate_parquet_inventory(inventory_path, dataset_root=dataset_root)
    verification = _load_json_object(verification_path, name="approved verification")
    verification_sha256 = _sha256_file(verification_path)
    release = _load_json_object(release_path, name="dataset release")
    if (
        release.get("schema_version") != "openpi.dataset_release.v1"
        or release.get("status") != "user_approved_data_ready_for_training"
        or release.get("episodes") != EXPECTED_INCLUDED_EPISODES
    ):
        raise BundleError("dataset release is not the approved 90-episode release")
    _validate_historical_dataset_reference(release.get("dataset"), dataset_root=dataset_root, name="release dataset")
    expected_links = (
        ("approved_manifest", approved_path, "approved_manifest_sha256", approved_sha256),
        ("approval_ledger", ledger_path, "approval_ledger_sha256", ledger_sha256),
        ("verification", verification_path, "verification_sha256", verification_sha256),
        ("parquet_payload_inventory", inventory_path, "parquet_payload_inventory_sha256", inventory_sha256),
    )
    for path_key, expected_path, hash_key, expected_hash in expected_links:
        if _path_from_json(release.get(path_key), name=f"release {path_key}") != expected_path:
            raise BundleError(f"release {path_key} points at the wrong file")
        if release.get(hash_key) != expected_hash:
            raise BundleError(f"release {hash_key} does not match the supplied bytes")
    if (
        release.get("parquet_payload_files") != inventory["file_count"]
        or release.get("parquet_payload_bytes") != inventory["total_bytes"]
    ):
        raise BundleError("release parquet counts do not match the payload inventory")
    if (
        verification.get("schema_version") != 2
        or verification.get("status") != "passed"
        or verification.get("episodes") != EXPECTED_INCLUDED_EPISODES
        or verification.get("manifest_sha256") != approved_sha256
        or _path_from_json(verification.get("manifest"), name="verification manifest") != approved_path
    ):
        raise BundleError("approved verification is not bound to the supplied manifest and dataset")
    _validate_historical_dataset_reference(
        verification.get("dataset"), dataset_root=dataset_root, name="verification dataset"
    )
    release_meta = _validate_dataset_meta_hashes(release.get("meta_sha256"), dataset_root=dataset_root)
    if verification.get("meta_sha256") != release_meta:
        raise BundleError("approved verification and release meta hashes differ")


def _validate_norm_assets(
    norm_dir: Path,
    *,
    frozen_sha256: str,
    included_records: Sequence[Mapping[str, Any]],
    dataset_repo_id: str,
) -> tuple[Path, Path]:
    norm_stats = _resolve_existing_file(norm_dir / "norm_stats.json", name="v3.5 norm stats")
    provenance_path = _resolve_existing_file(
        norm_dir / "norm_stats_provenance.json",
        name="v3.5 norm provenance",
    )
    provenance = _load_json_object(provenance_path, name="v3.5 norm provenance")
    train_records = sorted(
        (record for record in included_records if record.get("split") == "train"),
        key=lambda record: int(record["episode_index"]),
    )
    expected_indices = [int(record["episode_index"]) for record in train_records]
    expected_ids = [str(record["stable_id"]) for record in train_records]
    manifest_info = provenance.get("manifest")
    selection = provenance.get("selection")
    computation = provenance.get("computation")
    norm_info = provenance.get("norm_stats")
    storage = provenance.get("train_storage")
    if not all(isinstance(value, dict) for value in (manifest_info, selection, computation, norm_info, storage)):
        raise BundleError("norm provenance is missing manifest/selection/computation/norm_stats/train_storage objects")
    if (
        provenance.get("schema_version") != 2
        or provenance.get("status") != "complete"
        or provenance.get("repo_id") != dataset_repo_id
        or manifest_info.get("sha256") != frozen_sha256
        or manifest_info.get("path_relative") != DEFAULT_FROZEN_MANIFEST.as_posix()
        or "path" in manifest_info
        or manifest_info.get("active_split") != "train"
        or manifest_info.get("split_seed") != EXPECTED_SPLIT_SEED
        or selection.get("dataset_num_episodes") != EXPECTED_INCLUDED_EPISODES
        or selection.get("selected_num_episodes") != EXPECTED_SPLITS["train"]
        or selection.get("selected_episode_indices") != expected_indices
        or selection.get("selected_stable_ids") != expected_ids
    ):
        raise BundleError("norm provenance is not bound to the exact frozen 54-episode train split")
    frame_counts = selection.get("selected_episode_frame_counts")
    if (
        not isinstance(frame_counts, list)
        or len(frame_counts) != EXPECTED_SPLITS["train"]
        or any(type(count) is not int or count <= 0 for count in frame_counts)
        or selection.get("selected_num_frames") != sum(frame_counts)
        or computation.get("processed_base_rows") != sum(frame_counts)
        or computation.get("protocol") != NORM_COMPUTATION_PROTOCOL
        or computation.get("drop_last_rows") != 0
    ):
        raise BundleError("norm provenance does not cover every selected train frame exactly once")
    requested_batch_size = computation.get("requested_batch_size")
    if type(requested_batch_size) is not int or requested_batch_size <= 0:
        raise BundleError("norm provenance requested_batch_size must be a positive integer")
    expected_batches = (sum(frame_counts) + requested_batch_size - 1) // requested_batch_size
    if computation.get("num_batches_including_partial_final_batch") != expected_batches:
        raise BundleError("norm provenance batch count is inconsistent")
    if norm_info.get("file") != norm_stats.name or norm_info.get("sha256") != _sha256_file(norm_stats):
        raise BundleError("norm_stats.json bytes do not match norm provenance")

    expected_root_relative = (PROJECT_DATASET_PREFIX / PurePosixPath(dataset_repo_id)).as_posix()
    if (
        "root" in storage
        or storage.get("root_contract") != TRAIN_STORAGE_ROOT_CONTRACT
        or storage.get("root_relative") != expected_root_relative
        or storage.get("scope") != TRAIN_STORAGE_SCOPE
    ):
        raise BundleError("norm train_storage does not use the exact portable project-relative contract")
    if storage.get("selected_episode_indices") != expected_indices:
        raise BundleError("norm train-storage membership is not the frozen train split")
    storage_files = storage.get("files")
    if not isinstance(storage_files, list) or not storage_files:
        raise BundleError("norm train_storage.files must be a nonempty file-record list")
    storage_paths: set[str] = set()
    for record in storage_files:
        if not isinstance(record, dict):
            raise BundleError("norm train-storage records must be JSON objects")
        relative = _safe_relative_path(record.get("path", ""), name="norm train-storage path")
        if BUNDLE_NAME in relative.parts:
            raise BundleError("norm train-storage inventory must exclude the transfer bundle subtree")
        if relative.as_posix() in storage_paths:
            raise BundleError("norm train-storage inventory contains duplicate paths")
        storage_paths.add(relative.as_posix())
        if type(record.get("size")) is not int or record["size"] <= 0:
            raise BundleError("norm train-storage sizes must be positive integers")
        _require_sha256("norm train-storage file hash", record.get("sha256"))
    storage_sha256 = _sha256_bytes(canonical_json_bytes(storage_files))
    if storage.get("sha256") != storage_sha256:
        raise BundleError("norm train-storage aggregate SHA256 is invalid")
    return norm_stats, provenance_path


def _copy_item(source: Path, destination: PurePosixPath, *, role: str) -> CopyItem:
    source = _resolve_existing_file(source, name=role)
    destination = _safe_relative_path(destination, name=f"{role} destination")
    if destination.suffix not in ALLOWED_PAYLOAD_SUFFIXES:
        raise BundleError(f"{role} destination has a forbidden payload type: {destination}")
    if destination.suffix in FORBIDDEN_STORAGE_SUFFIXES or FORBIDDEN_STORAGE_PARTS.intersection(destination.parts):
        raise BundleError(f"{role} would copy raw storage into the transfer bundle: {destination}")
    return CopyItem(
        source=source,
        destination=destination,
        role=role,
        size=source.stat().st_size,
        sha256=_sha256_file(source),
    )


def _readme_bytes(
    *,
    frozen_destination: PurePosixPath,
    frozen_sha256: str,
    dataset_repo_id: str,
    norm_restore_dir: PurePosixPath | None,
) -> bytes:
    norm_text: str
    if norm_restore_dir is None:
        norm_text = (
            "Norm assets were not present when this bundle was built. Before training, run the v3.5 train-only "
            "normalization job and rebuild the bundle with `--norm-dir`.\n"
        )
    else:
        repo_parts = PurePosixPath(dataset_repo_id).parts
        if tuple(norm_restore_dir.parts[-len(repo_parts) :]) != repo_parts:
            raise BundleError("norm restore directory must end with the dataset repo_id path")
        assets_dir = PurePosixPath(*norm_restore_dir.parts[: -len(repo_parts)])
        norm_text = (
            "Norm assets are included at "
            f"`<PROJECT_ROOT>/{norm_restore_dir.as_posix()}`. Set `AssetsConfig.assets_dir` to "
            f"`<PROJECT_ROOT>/{assets_dir.as_posix()}` and leave `asset_id` unset (or set it to "
            f"`{dataset_repo_id}`). The copied schema-v2 provenance is byte-exact and portable: keep "
            f"`train_storage.root_relative=data/lerobot/{dataset_repo_id}` and "
            "`train_storage.root_contract=memory_project-relative-v1` unchanged.\n"
        )
    text = f"""# v3.5 training transfer bundle

This directory makes the LeRobot dataset self-contained for transfer. It contains only frozen
labels, manifests, approval/release provenance, and optional train-only norm assets. It contains
no parquet, videos, raw state/action arrays, or links. `TRANSFER_MANIFEST.json` is canonical JSON
and hashes every payload file except itself.

Population: 70 included episodes; train/development/final_test = 54/8/8.
Frozen manifest SHA256: `{frozen_sha256}`.

## Verify

From a checkout containing `openpi/scripts/build_v35_transfer_bundle.py`:

```bash
python3 openpi/scripts/build_v35_transfer_bundle.py verify --bundle-root <DATASET_ROOT>/{BUNDLE_NAME}
```

## Use directly or restore

The unchanged frozen manifest and labels already form a consistent relative tree under
`restore_tree/project_root`. You may point the training config directly at
`<DATASET_ROOT>/{BUNDLE_NAME}/{frozen_destination.as_posix()}`, or copy the restore tree into a
clean project root:

```bash
cp -a <DATASET_ROOT>/{BUNDLE_NAME}/restore_tree/project_root/. <PROJECT_ROOT>/
```

Set `memory_episode_manifest_path` to `<PROJECT_ROOT>/{frozen_destination.relative_to(RESTORE_PREFIX).as_posix()}`
and `memory_episode_manifest_sha256` to `{frozen_sha256}`. Keep `memory_manifest_split_seed=36`,
`memory_manifest_split="train"`, and `memory_v35_frozen_population=True`.

Place the uploaded dataset exactly at `<PROJECT_ROOT>/data/lerobot/{dataset_repo_id}`. The v3.5
configuration resolves it from the project root and does not depend on a machine-local
Hugging Face cache. Set `MEMORY_PROJECT_ROOT=<PROJECT_ROOT>` only when source-tree discovery is
not available on the destination cluster.

{norm_text}
Within the dataset, do not move this bundle into its internal `data/`, `videos/`, or `meta/`
directories; those storage trees are covered by the train-storage seal. The schema-v2 manifest
and norm provenance use project-relative paths and must not be rebased after transfer.
"""
    return text.encode("utf-8")


def _resolve_input_path(project_root: Path, value: Path | None, default: PurePosixPath) -> Path:
    if value is None:
        return project_root / Path(*default.parts)
    return value if value.is_absolute() else project_root / value


def _validate_inputs(inputs: BundleInputs) -> BundleInputs:
    dataset_root = _resolve_existing_directory(inputs.dataset_root, name="dataset root")
    source_project_root = _resolve_existing_directory(inputs.source_project_root, name="source project root")
    repo_path = _safe_relative_path(inputs.dataset_repo_id, name="dataset repo_id")
    expected_dataset_root = (
        source_project_root / Path(*PROJECT_DATASET_PREFIX.parts) / Path(*repo_path.parts)
    ).resolve()
    if dataset_root != expected_dataset_root:
        raise BundleError(
            f"dataset root must use the portable project layout {expected_dataset_root}; found {dataset_root}"
        )
    if not (dataset_root / "meta").is_dir():
        raise BundleError(f"dataset root has no structural meta/ directory: {dataset_root}")
    frozen_manifest = _resolve_existing_file(inputs.frozen_manifest, name="frozen manifest")
    approved_manifest = _resolve_existing_file(inputs.approved_manifest, name="approved manifest")
    approval_dir = _resolve_existing_directory(inputs.approval_provenance_dir, name="approval provenance directory")
    approved_verification = _resolve_existing_file(inputs.approved_verification, name="approved verification")
    for path, name in (
        (frozen_manifest, "frozen manifest"),
        (approved_manifest, "approved manifest"),
        (approval_dir, "approval provenance directory"),
        (approved_verification, "approved verification"),
    ):
        _project_relative(path, source_project_root, name=name)
    norm_dir = None
    if inputs.norm_dir is not None:
        norm_dir = _resolve_existing_directory(inputs.norm_dir, name="norm asset directory")
    return BundleInputs(
        dataset_root=dataset_root,
        source_project_root=source_project_root,
        frozen_manifest=frozen_manifest,
        approved_manifest=approved_manifest,
        approval_provenance_dir=approval_dir,
        approved_verification=approved_verification,
        dataset_repo_id=inputs.dataset_repo_id,
        norm_dir=norm_dir,
        norm_restore_dir=_safe_relative_path(inputs.norm_restore_dir, name="norm restore directory"),
    )


def plan_bundle(inputs: BundleInputs) -> BundlePlan:
    """Validate every source and return the deterministic copy plan without writing."""

    inputs = _validate_inputs(inputs)
    frozen, included, split_counts = _validate_frozen_manifest(
        inputs.frozen_manifest,
        source_project_root=inputs.source_project_root,
    )
    frozen_records = list(frozen["episodes"])
    approved, approved_sha256 = _validate_approved_manifest(
        inputs.approved_manifest,
        frozen=frozen,
        frozen_records=frozen_records,
        source_project_root=inputs.source_project_root,
    )
    approval_paths = {
        name: _resolve_existing_file(inputs.approval_provenance_dir / name, name=name)
        for name in APPROVAL_PROVENANCE_FILENAMES
    }
    _ledger, ledger_sha256 = _validate_approval_ledger(
        approval_paths["approval_ledger.json"],
        frozen=frozen,
        approved_path=inputs.approved_manifest,
        approved=approved,
    )
    _validate_release_provenance(
        release_path=approval_paths["release.json"],
        conversion_path=approval_paths["conversion_report_preapproval.json"],
        inventory_path=approval_paths["parquet_payload_inventory.json"],
        verification_path=inputs.approved_verification,
        approved_path=inputs.approved_manifest,
        approved_sha256=approved_sha256,
        ledger_path=approval_paths["approval_ledger.json"],
        ledger_sha256=ledger_sha256,
        dataset_root=inputs.dataset_root,
        dataset_repo_id=inputs.dataset_repo_id,
    )

    bundle_root = inputs.dataset_root / BUNDLE_NAME
    included_ids = {str(record["stable_id"]) for record in included}
    sidecar_specs = (
        ("block_confound_audit", "block-confound sidecar", "block_confound_sidecar", "pass"),
        ("e_visibility_review", "E-visibility sidecar", "e_visibility_sidecar", "user_approved"),
        ("d_valid_sidecar", "D-valid sidecar", "d_valid_sidecar", "complete"),
    )
    sidecars: list[tuple[Path, str]] = []
    for descriptor_key, descriptor_name, role, status in sidecar_specs:
        path, _digest = _validate_sidecar(
            manifest_path=inputs.frozen_manifest,
            descriptor=frozen.get(descriptor_key),
            descriptor_name=descriptor_name,
            role=role,
            expected_status=status,
            included_ids=included_ids,
        )
        sidecars.append((path, role))

    items: list[CopyItem] = []

    def add_project_file(path: Path, *, role: str) -> None:
        if bundle_root == path or bundle_root in path.parents:
            raise BundleError(f"refusing to inventory the transfer bundle as its own source: {path}")
        relative = _project_relative(path, inputs.source_project_root, name=role)
        items.append(_copy_item(path, RESTORE_PREFIX / relative, role=role))

    add_project_file(inputs.frozen_manifest, role="frozen_manifest")
    for path, role in sidecars:
        add_project_file(path, role=role)
    add_project_file(inputs.approved_manifest, role="approved_manifest")
    for name in APPROVAL_PROVENANCE_FILENAMES:
        add_project_file(approval_paths[name], role=f"approval_provenance:{name}")
    add_project_file(inputs.approved_verification, role="approved_verification")

    frozen_raw_root = Path(str(frozen["raw_root"]))
    if not frozen_raw_root.is_absolute():
        frozen_raw_root = inputs.frozen_manifest.parent / frozen_raw_root
    frozen_raw_root = frozen_raw_root.resolve()
    for record in sorted(included, key=lambda value: int(value["episode_index"])):
        stable_id = str(record["stable_id"])
        raw_dir = _safe_relative_path(record.get("raw_dir", ""), name=f"{stable_id} raw_dir")
        label_file = record.get("label_file")
        if not isinstance(label_file, str) or Path(label_file).name != label_file or not label_file.endswith(".json"):
            raise BundleError(f"{stable_id} label_file must be a JSON basename")
        source = _resolve_existing_file(
            frozen_raw_root / Path(*raw_dir.parts) / label_file,
            name=f"label for {stable_id}",
        )
        try:
            source.relative_to(frozen_raw_root)
        except ValueError as exc:
            raise BundleError(f"label source escapes raw_root for {stable_id}") from exc
        expected_sha256 = _require_sha256(f"{stable_id} label_sha256", record.get("label_sha256"))
        label_bytes = source.read_bytes()
        if _sha256_bytes(label_bytes) != expected_sha256:
            raise BundleError(f"label bytes do not match the frozen hash for {stable_id}: {source}")
        try:
            labels = json.loads(label_bytes, object_pairs_hook=_reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BundleError(f"label bytes are not valid JSON for {stable_id}: {exc}") from exc
        if not isinstance(labels, list):
            raise BundleError(f"label JSON must be a list for {stable_id}")
        destination = RESTORE_PREFIX / raw_dir / label_file
        items.append(_copy_item(source, destination, role="episode_label"))

    norm_restore_dir: PurePosixPath | None = None
    if inputs.norm_dir is not None:
        norm_stats, norm_provenance = _validate_norm_assets(
            inputs.norm_dir,
            frozen_sha256=_sha256_file(inputs.frozen_manifest),
            included_records=included,
            dataset_repo_id=inputs.dataset_repo_id,
        )
        norm_restore_dir = inputs.norm_restore_dir
        for path, role in ((norm_stats, "train_only_norm_stats"), (norm_provenance, "train_only_norm_provenance")):
            items.append(_copy_item(path, RESTORE_PREFIX / norm_restore_dir / path.name, role=role))

    destinations = [item.destination.as_posix() for item in items]
    if len(destinations) != len(set(destinations)):
        duplicates = sorted(path for path, count in Counter(destinations).items() if count > 1)
        raise BundleError(f"copy plan has destination collisions: {duplicates}")
    if sum(item.role == "episode_label" for item in items) != EXPECTED_INCLUDED_EPISODES:
        raise BundleError("copy plan does not contain exactly 90 label files")
    if any(item.source == bundle_root or bundle_root in item.source.parents for item in items):
        raise BundleError("copy plan recursively consumes an existing transfer bundle")

    frozen_destination = next(item.destination for item in items if item.role == "frozen_manifest")
    frozen_sha256 = _sha256_file(inputs.frozen_manifest)
    readme_bytes = _readme_bytes(
        frozen_destination=frozen_destination,
        frozen_sha256=frozen_sha256,
        dataset_repo_id=inputs.dataset_repo_id,
        norm_restore_dir=norm_restore_dir,
    )
    file_records = [item.record() for item in items]
    file_records.append(
        {
            "path": README_NAME,
            "role": "restore_instructions",
            "sha256": _sha256_bytes(readme_bytes),
            "size": len(readme_bytes),
        }
    )
    file_records.sort(key=lambda record: record["path"])
    transfer_manifest: dict[str, Any] = {
        "dataset_repo_id": inputs.dataset_repo_id,
        "files": file_records,
        "frozen_manifest": {
            "path": frozen_destination.as_posix(),
            "sha256": frozen_sha256,
        },
        "norm_assets": {
            "included": norm_restore_dir is not None,
            "restore_directory": None if norm_restore_dir is None else (RESTORE_PREFIX / norm_restore_dir).as_posix(),
        },
        "payload": {
            "byte_count": sum(record["size"] for record in file_records),
            "file_count": len(file_records),
            "transfer_manifest_self_included": False,
        },
        "population": {
            "included_episodes": EXPECTED_INCLUDED_EPISODES,
            "raw_records": EXPECTED_RAW_RECORDS,
            "split_seed": EXPECTED_SPLIT_SEED,
            "splits": split_counts,
        },
        "restore_prefix": RESTORE_PREFIX.as_posix(),
        "schema_version": SCHEMA_VERSION,
    }
    return BundlePlan(
        dataset_root=inputs.dataset_root,
        bundle_root=bundle_root,
        copy_items=tuple(sorted(items, key=lambda item: item.destination.as_posix())),
        readme_bytes=readme_bytes,
        transfer_manifest=transfer_manifest,
    )


def verify_bundle(bundle_root: Path) -> dict[str, Any]:
    """Verify a materialized bundle without reading dataset data/videos/meta storage."""

    bundle_root = _resolve_existing_directory(bundle_root, name="transfer bundle")
    manifest_path = _resolve_existing_file(bundle_root / TRANSFER_MANIFEST_NAME, name="transfer manifest")
    raw_bytes = manifest_path.read_bytes()
    manifest = _load_json_object(manifest_path, name="transfer manifest")
    if raw_bytes != canonical_json_bytes(manifest) + b"\n":
        raise BundleError("TRANSFER_MANIFEST.json is not in canonical byte representation")
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("restore_prefix") != RESTORE_PREFIX.as_posix():
        raise BundleError("transfer manifest has the wrong schema or restore prefix")
    population = manifest.get("population")
    if not isinstance(population, dict) or population.get("splits") != EXPECTED_SPLITS:
        raise BundleError("transfer manifest population is not the frozen 54/8/8 split")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise BundleError("transfer manifest must contain a nonempty file list")
    listed: set[str] = set()
    role_counts: Counter[str] = Counter()
    byte_count = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "role", "sha256", "size"}:
            raise BundleError("transfer manifest file records must contain path/role/size/sha256 exactly")
        relative = _safe_relative_path(record["path"], name="transfer payload path")
        if relative.as_posix() == TRANSFER_MANIFEST_NAME or relative.as_posix() in listed:
            raise BundleError("transfer manifest contains itself or a duplicate payload path")
        if relative.suffix not in ALLOWED_PAYLOAD_SUFFIXES or relative.suffix in FORBIDDEN_STORAGE_SUFFIXES:
            raise BundleError(f"transfer manifest contains a forbidden payload type: {relative}")
        if FORBIDDEN_STORAGE_PARTS.intersection(relative.parts):
            raise BundleError(f"transfer manifest contains raw video storage: {relative}")
        path = bundle_root / Path(*relative.parts)
        resolved = _resolve_existing_file(path, name="transfer payload")
        try:
            resolved.relative_to(bundle_root)
        except ValueError as exc:
            raise BundleError(f"transfer payload escapes bundle root: {relative}") from exc
        size = record["size"]
        if type(size) is not int or size < 0 or path.stat().st_size != size:
            raise BundleError(f"transfer payload size mismatch: {relative}")
        expected_sha256 = _require_sha256(f"transfer payload hash for {relative}", record["sha256"])
        if _sha256_file(path) != expected_sha256:
            raise BundleError(f"transfer payload hash mismatch: {relative}")
        listed.add(relative.as_posix())
        role_counts[str(record["role"])] += 1
        byte_count += size
    if role_counts["episode_label"] != EXPECTED_INCLUDED_EPISODES:
        raise BundleError("transfer bundle must contain exactly 90 byte-exact label files")
    if role_counts["frozen_manifest"] != 1 or role_counts["approved_manifest"] != 1:
        raise BundleError("transfer bundle is missing the frozen or approved manifest")
    if sum(role_counts[role] for role in ("block_confound_sidecar", "e_visibility_sidecar", "d_valid_sidecar")) != 3:
        raise BundleError("transfer bundle is missing one or more frozen sidecars")
    if role_counts["train_only_norm_stats"] != role_counts["train_only_norm_provenance"]:
        raise BundleError("transfer bundle must include both norm files or neither")
    norm_included = bool(manifest.get("norm_assets", {}).get("included"))
    if norm_included != (role_counts["train_only_norm_stats"] == 1):
        raise BundleError("transfer manifest norm_assets flag is inconsistent")

    actual: set[str] = set()
    for path in bundle_root.rglob("*"):
        if path.is_symlink():
            raise BundleError(f"transfer bundle contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(bundle_root).as_posix()
        if relative != TRANSFER_MANIFEST_NAME:
            actual.add(relative)
    if actual != listed:
        raise BundleError(
            f"transfer bundle payload inventory differs from disk: missing={sorted(listed - actual)}, "
            f"unexpected={sorted(actual - listed)}"
        )
    payload = manifest.get("payload")
    if (
        not isinstance(payload, dict)
        or payload.get("file_count") != len(records)
        or payload.get("byte_count") != byte_count
        or payload.get("transfer_manifest_self_included") is not False
    ):
        raise BundleError("transfer manifest payload summary is inconsistent")
    return {
        "bundle_root": str(bundle_root),
        "file_count": len(records),
        "label_count": role_counts["episode_label"],
        "norm_assets_included": norm_included,
        "payload_bytes": byte_count,
        "transfer_manifest_sha256": _sha256_bytes(raw_bytes),
    }


def build_bundle(plan: BundlePlan, *, overwrite: bool = False) -> dict[str, Any]:
    """Materialize a validated plan transactionally and verify the completed bundle."""

    target = plan.bundle_root
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing transfer bundle: {target}")
    if target.parent != plan.dataset_root or target.name != BUNDLE_NAME:
        raise BundleError(f"unsafe transfer bundle target: {target}")
    stage = Path(tempfile.mkdtemp(prefix=f".{BUNDLE_NAME}.stage-", dir=plan.dataset_root))
    backup: Path | None = None
    try:
        for item in plan.copy_items:
            destination = stage / Path(*item.destination.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item.source, destination, follow_symlinks=False)
            if (
                destination.is_symlink()
                or destination.stat().st_size != item.size
                or _sha256_file(destination) != item.sha256
            ):
                raise BundleError(f"post-copy verification failed for {item.destination}")
        (stage / README_NAME).write_bytes(plan.readme_bytes)
        (stage / TRANSFER_MANIFEST_NAME).write_bytes(canonical_json_bytes(plan.transfer_manifest) + b"\n")
        verify_bundle(stage)
        if target.exists():
            backup = plan.dataset_root / f".{BUNDLE_NAME}.backup-{uuid.uuid4().hex}"
            os.replace(target, backup)
        try:
            os.replace(stage, target)
        except Exception:
            if backup is not None and backup.exists() and not target.exists():
                os.replace(backup, target)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
        return verify_bundle(target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if backup is not None and backup.exists():
            if not target.exists():
                os.replace(backup, target)
            else:
                shutil.rmtree(backup)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="validate sources and build the transfer bundle")
    build.add_argument("--dataset-root", type=Path, required=True)
    build.add_argument("--source-project-root", type=Path, required=True)
    build.add_argument("--frozen-manifest", type=Path)
    build.add_argument("--approved-manifest", type=Path)
    build.add_argument("--approval-provenance-dir", type=Path)
    build.add_argument("--approved-verification", type=Path)
    build.add_argument("--dataset-repo-id", default=DATASET_REPO_ID)
    build.add_argument("--norm-dir", type=Path)
    build.add_argument("--norm-restore-dir", type=PurePosixPath, default=DEFAULT_NORM_RESTORE_DIR)
    build.add_argument("--dry-run", action="store_true")
    build.add_argument("--overwrite", action="store_true")
    verify = subparsers.add_parser("verify", help="verify a materialized transfer bundle")
    verify.add_argument("--bundle-root", type=Path, required=True)
    return parser


def _plan_from_args(args: argparse.Namespace) -> BundlePlan:
    project_root = args.source_project_root.expanduser().resolve()
    norm_dir = None
    if args.norm_dir is not None:
        norm_dir = args.norm_dir if args.norm_dir.is_absolute() else project_root / args.norm_dir
    inputs = BundleInputs(
        dataset_root=args.dataset_root,
        source_project_root=project_root,
        frozen_manifest=_resolve_input_path(project_root, args.frozen_manifest, DEFAULT_FROZEN_MANIFEST),
        approved_manifest=_resolve_input_path(project_root, args.approved_manifest, DEFAULT_APPROVED_MANIFEST),
        approval_provenance_dir=_resolve_input_path(
            project_root,
            args.approval_provenance_dir,
            DEFAULT_APPROVAL_PROVENANCE_DIR,
        ),
        approved_verification=_resolve_input_path(
            project_root,
            args.approved_verification,
            DEFAULT_APPROVED_VERIFICATION,
        ),
        dataset_repo_id=args.dataset_repo_id,
        norm_dir=norm_dir,
        norm_restore_dir=args.norm_restore_dir,
    )
    return plan_bundle(inputs)


def main() -> None:
    args = _build_parser().parse_args()
    if args.command == "verify":
        result = verify_bundle(args.bundle_root)
    else:
        plan = _plan_from_args(args)
        if args.dry_run:
            manifest_bytes = canonical_json_bytes(plan.transfer_manifest) + b"\n"
            result = {
                "bundle_root": str(plan.bundle_root),
                "dry_run": True,
                "file_count": plan.transfer_manifest["payload"]["file_count"],
                "label_count": EXPECTED_INCLUDED_EPISODES,
                "norm_assets_included": plan.transfer_manifest["norm_assets"]["included"],
                "payload_bytes": plan.transfer_manifest["payload"]["byte_count"],
                "transfer_manifest_sha256": _sha256_bytes(manifest_bytes),
            }
        else:
            result = build_bundle(plan, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
