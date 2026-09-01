"""Produce the authenticated, train-only feature artifact consumed by Gate B.

This collector is intentionally stricter than a generic representation dump.  It reads only
the 54 training parquet files selected by the frozen schema-v2 manifest, and within each file
uses only the inclusive ``d_valid`` interval.  Images and the 14-D state pass through the
registered v3.5 inference preprocessing with augmentation disabled.  The model features come
from the exact fresh-step-0 v3.5 parameter tree and the production layer-8 memory interface.

The output is a deterministic NPZ plus canonical JSON envelopes accepted by
``v35_leakage_gate.py``.  All command-line paths are relative to ``memory_project``.  The
collector also authenticates the actual official-base and step-0 parameter trees, the frozen
manifest, schema-v2 norm provenance, the selected-train storage seal, and the fixed feature and
preprocessing protocols before it reads a pixel or robot-state value.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import sys
import tempfile
from typing import Any, Protocol
import zipfile

import numpy as np
from numpy import typing as npt

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import v35_gate_artifacts as artifacts  # noqa: E402
import v35_leakage_gate as gate  # noqa: E402

from openpi.shared import project_paths  # noqa: E402

CONFIG_NAME = "pi05_yam_mem_v35"
OFFICIAL_BASE_URI = "gs://openpi-assets/checkpoints/pi05_base/params"
OFFICIAL_BASE_PARAMS = PurePosixPath("v35/cache/openpi/openpi-assets/checkpoints/pi05_base/params")
FROZEN_MANIFEST = project_paths.V35_FROZEN_MANIFEST
FROZEN_MANIFEST_SHA256 = "9085fe50d7b02ea65930f3647ce0413e0583a66d430484e06c60812c52af8442"
DATASET_ROOT = project_paths.V35_DATASET_DIR
NORM_DIR = project_paths.V35_ASSETS_DIR / project_paths.V35_REPO_ID
NORM_STATS_FILE = "norm_stats.json"
NORM_PROVENANCE_FILE = "norm_stats_provenance.json"
NORM_COMPUTATION_PROTOCOL = "raw-train-rows-delta-action-horizon-v1"
NORM_SCOPE = "v3.5 manifest-selected included training episodes only"
ROOT_CONTRACT = "memory_project-relative-v1"
STORAGE_SCOPE = "selected train episode parquet, optional videos, plus structural meta files"
EXPECTED_DATASET_FRAMES = 55_980
EXPECTED_TRAIN_FRAMES = 42_725
EXPECTED_INITIALIZATION_SEED = 42
EXPECTED_IMAGE_KEYS = ("image", "left_wrist_image", "right_wrist_image")
PARQUET_COLUMNS = (*EXPECTED_IMAGE_KEYS, "state", "frame_index", "episode_index", "task_index")
EXPECTED_PROMPTS = {
    "banana": "find the banana",
    "grey_pepper_box": "find the grey pepper box",
}
EXPECTED_WAIT_TASKS = {
    0: "wait; target bin is left",
    1: "wait; target bin is right",
}
NPZ_MEMBER_ORDER = ("episode_stable_id", *gate.ALL_FEATURES)
NPZ_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


IMAGE_PREPROCESSING_SPEC: Mapping[str, Any] = {
    "schema": "openpi.v35.leakage-image-preprocessing.v1",
    "source": "converted parquet inline RGB only",
    "decode": "PIL.Image.open(bytes).convert('RGB')",
    "resize": "openpi_client.image_tools.resize_with_pad(height=224,width=224,method=PIL.BILINEAR)",
    "scale": "uint8/255*2-1 float32",
    "augmentation": False,
    "camera_order": ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"],
}
PREPROCESSING_PROTOCOL_SPEC: Mapping[str, Any] = {
    "schema": "openpi.v35.leakage-final-preprocessing.v1",
    "registered_config": CONFIG_NAME,
    "model_preprocess_train": False,
    "image": IMAGE_PREPROCESSING_SPEC,
    "state": {
        "dimensions": 14,
        "normalization": "(state-train54_mean)/(train54_std+1e-6)",
        "norm_computation_protocol": NORM_COMPUTATION_PROTOCOL,
        "state_masking": False,
    },
    "prompt": "manifest episode prompt tokenized by registered FASTSubtaskTokenizer inference path",
}
FEATURE_PROTOCOL_SPEC: Mapping[str, Any] = {
    "schema": "openpi.v35.leakage-feature-production.v1",
    "population": {
        "split": "train",
        "episode_count": 54,
        "development_or_final_observation_access": False,
        "row_order": "frozen manifest episode_index order",
    },
    "frames": {
        "selector": "inclusive manifest episode.d_valid[start,end]",
        "validation": "contiguous frame_index, matching episode_index, exact side-specific wait task",
        "aggregation": gate.AGGREGATION_PROTOCOL,
    },
    "fresh_model": {
        "config": CONFIG_NAME,
        "completed_updates": 0,
        "official_source_uri": OFFICIAL_BASE_URI,
        "parameter_identity": "actual official-base tree and actual fresh step-0 tree SHA256",
    },
    "features": {
        "primary_final_images_state": "mean-D concat(flatten(top,left_wrist,right_wrist), normalized_14d_state)",
        "primary_step0_kbar_vbar": "mean-D concat(production pool_kv.pooled_key,pool_kv.pooled_value)",
        "descriptive_top_only": "mean-D flatten(top)",
        "descriptive_left_wrist_only": "mean-D flatten(left_wrist)",
        "descriptive_right_wrist_only": "mean-D flatten(right_wrist)",
        "descriptive_all_images_only": "mean-D concat(flatten(top,left_wrist,right_wrist))",
        "descriptive_state_only": "mean-D normalized_14d_state",
        "descriptive_prompt_only": "mean-D concat(nonstate_valid_token_ids,nonstate_valid_mask)",
        "descriptive_layer8_only": "mean-D mean over 16 production layer8 write tokens",
        "descriptive_kbar_only": "mean-D production pool_kv.pooled_key",
        "descriptive_vbar_only": "mean-D production pool_kv.pooled_value",
    },
    "numerics": {
        "model_interface": "production v32_memory_interface_step with blank FP32 fast state",
        "episode_accumulation": "raw-frame order float64 sum then one float32 cast",
        "stored_dtype": "float32",
        "finite_required": True,
    },
}


def _canonical_sha256(value: Any) -> str:
    return artifacts.sha256_bytes(artifacts.canonical_json_bytes(value))


# These literals make protocol changes deliberate.  Import fails if somebody edits a spec
# without also reviewing and updating its frozen digest.
FROZEN_IMAGE_PREPROCESSING_SHA256 = "5a28e445efda185b58c96ce0601f1e14886818dcbdcad0fd33e5e1ebf1ac5006"
FROZEN_PREPROCESSING_PROTOCOL_SHA256 = "35484fd1e1fac9dac9f9b2a62fb9a0748f289b0e6af6b8fd68f31b3220724c19"
FROZEN_FEATURE_PROTOCOL_SHA256 = "f755bc6fa5314aabd6f1985126890e7eace0b66e853a9022354ebd9d8404ba86"


class LeakageFeatureError(artifacts.GateArtifactError):
    """Raised when trusted Gate-B feature production cannot be proven."""


@dataclasses.dataclass(frozen=True)
class EpisodeSource:
    stable_id: str
    episode_index: int
    collection: str
    object_name: str
    target_side: int
    d_start: int
    d_end: int
    expected_frames: int
    prompt: str

    @property
    def d_count(self) -> int:
        return self.d_end - self.d_start + 1


@dataclasses.dataclass(frozen=True)
class AuthenticatedInputs:
    manifest: artifacts.FrozenManifest
    manifest_raw: Mapping[str, Any]
    episodes: tuple[EpisodeSource, ...]
    dataset_root: Path
    norm_dir: Path
    norm_stats_sha256: str
    norm_provenance_sha256: str
    train_storage_sha256: str
    dataset_protocol_sha256: str
    initialization_path: Path
    initialization_raw: Mapping[str, Any]
    initialization_sha256: str
    initialization_parameter_tree_sha256: str
    official_base_source_tree_sha256: str
    graft_path: Path
    graft_sha256: str
    step0_params_path: Path


@dataclasses.dataclass(frozen=True)
class EpisodeFeatureRow:
    stable_id: str
    frame_count: int
    features: Mapping[str, npt.NDArray[np.float32]]


class EpisodeExtractor(Protocol):
    def extract_episode(self, episode: EpisodeSource) -> EpisodeFeatureRow: ...


def _assert_frozen_protocol_hashes() -> None:
    expected = {
        "image preprocessing": (IMAGE_PREPROCESSING_SPEC, FROZEN_IMAGE_PREPROCESSING_SHA256),
        "final preprocessing": (PREPROCESSING_PROTOCOL_SPEC, FROZEN_PREPROCESSING_PROTOCOL_SHA256),
        "feature production": (FEATURE_PROTOCOL_SPEC, FROZEN_FEATURE_PROTOCOL_SHA256),
    }
    for name, (spec, digest) in expected.items():
        actual = _canonical_sha256(spec)
        if actual != digest:
            raise LeakageFeatureError(f"frozen {name} protocol hash mismatch: declared {digest}, computed {actual}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise LeakageFeatureError(f"duplicate JSON key {key!r}")
        output[key] = value
    return output


def _load_json_object(path: Path, *, name: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LeakageFeatureError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise LeakageFeatureError(f"{name} must be a JSON object")
    return value, raw


def _relative_project_path(value: str | PurePosixPath, *, name: str, must_exist: bool = True) -> Path:
    relative = PurePosixPath(value)
    if not relative.parts or relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise LeakageFeatureError(f"{name} must be a normalized memory_project-relative POSIX path")
    path = project_paths.project_path(relative)
    if must_exist and not path.exists():
        raise LeakageFeatureError(f"{name} does not exist: {relative.as_posix()}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise LeakageFeatureError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _self_hashed_json(raw: Mapping[str, Any], *, hash_key: str, name: str) -> str:
    digest = artifacts.require_sha256(f"{name}.{hash_key}", raw.get(hash_key))
    unhashed = dict(raw)
    del unhashed[hash_key]
    actual = _canonical_sha256(unhashed)
    if actual != digest:
        raise LeakageFeatureError(f"{name} self-hash mismatch: expected {digest}, computed {actual}")
    return digest


def _manifest_episode_sources(
    manifest: artifacts.FrozenManifest,
    manifest_raw: Mapping[str, Any],
) -> tuple[EpisodeSource, ...]:
    records = manifest_raw.get("episodes")
    if not isinstance(records, list):
        raise LeakageFeatureError("frozen manifest is missing episodes")
    by_index = {
        record.get("episode_index", record.get("lerobot_episode_index")): record
        for record in records
        if isinstance(record, dict) and bool(record.get("include", True))
    }
    sources: list[EpisodeSource] = []
    for episode in manifest.split("train"):
        record = by_index.get(episode.episode_index)
        if not isinstance(record, dict) or record.get("stable_id") != episode.stable_id:
            raise LeakageFeatureError(f"manifest raw record mismatch for training episode {episode.stable_id!r}")
        d_valid = record.get("d_valid")
        if not isinstance(d_valid, dict):
            raise LeakageFeatureError(f"training episode {episode.stable_id!r} is missing embedded d_valid")
        d_start, d_end = d_valid.get("start"), d_valid.get("end")
        expected_frames = record.get("expected_num_frames")
        if (
            d_valid.get("detector") != "14d-max-step-lt-0.004-max-excursion-lte-0.02-v1"
            or d_valid.get("state_dim") != 14
            or type(d_start) is not int
            or type(d_end) is not int
            or type(expected_frames) is not int
            or not (0 <= d_start <= d_end < expected_frames)
        ):
            raise LeakageFeatureError(f"training episode {episode.stable_id!r} has invalid final d_valid interval")
        prompt = record.get("prompt")
        if prompt != EXPECTED_PROMPTS[episode.object_name]:
            raise LeakageFeatureError(f"training episode {episode.stable_id!r} has a noncanonical prompt")
        sources.append(
            EpisodeSource(
                stable_id=episode.stable_id,
                episode_index=episode.episode_index,
                collection=episode.collection,
                object_name=episode.object_name,
                target_side=episode.target_side,
                d_start=d_start,
                d_end=d_end,
                expected_frames=expected_frames,
                prompt=prompt,
            )
        )
    if len(sources) != 54 or tuple(item.stable_id for item in sources) != tuple(
        item.stable_id for item in manifest.split("train")
    ):
        raise LeakageFeatureError("feature population is not exactly train54 in frozen manifest order")
    return tuple(sources)


def _validate_storage_seal(
    provenance: Mapping[str, Any],
    *,
    dataset_root: Path,
    train_episodes: Sequence[EpisodeSource],
) -> tuple[str, str]:
    storage = provenance.get("train_storage")
    if not isinstance(storage, dict) or set(storage) != {
        "root_contract",
        "root_relative",
        "sha256",
        "files",
        "selected_episode_indices",
        "scope",
    }:
        raise LeakageFeatureError("norm provenance is missing train_storage")
    expected_root = DATASET_ROOT.as_posix()
    expected_indices = [episode.episode_index for episode in train_episodes]
    if (
        storage.get("root_contract") != ROOT_CONTRACT
        or storage.get("root_relative") != expected_root
        or "root" in storage
        or storage.get("scope") != STORAGE_SCOPE
        or storage.get("selected_episode_indices") != expected_indices
        or dataset_root.resolve() != project_paths.project_path(DATASET_ROOT)
    ):
        raise LeakageFeatureError("norm train_storage does not use the frozen project-relative contract")
    files = storage.get("files")
    if not isinstance(files, list) or not files:
        raise LeakageFeatureError("norm train_storage.files must be a nonempty file list")
    aggregate = _canonical_sha256(files)
    if storage.get("sha256") != aggregate:
        raise LeakageFeatureError("norm train_storage aggregate SHA256 is invalid")

    train_indices = set(expected_indices)
    episode_filename = re.compile(r"episode_(\d{6})\.(parquet|mp4)")
    seen_paths: set[str] = set()
    ordered_paths: list[str] = []
    for offset, record in enumerate(files):
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise LeakageFeatureError(f"train_storage.files[{offset}] is not canonical")
        relative_value = record.get("path")
        if not isinstance(relative_value, str):
            raise LeakageFeatureError(f"train_storage.files[{offset}].path is invalid")
        relative = PurePosixPath(relative_value)
        if (
            not relative.parts
            or relative.is_absolute()
            or any(part in ("", ".", "..") for part in relative.parts)
            or relative_value in seen_paths
        ):
            raise LeakageFeatureError(f"train_storage.files[{offset}].path is unsafe or duplicated")
        seen_paths.add(relative_value)
        ordered_paths.append(relative_value)
        artifacts.require_sha256(f"train_storage.files[{offset}].sha256", record.get("sha256"))
        if type(record.get("size")) is not int or record["size"] < 0:
            raise LeakageFeatureError(f"train_storage.files[{offset}].size is invalid")
        if relative.parts[0] == "data":
            match = episode_filename.fullmatch(relative.name)
            if relative.parts[:2] != ("data", "chunk-000") or match is None or match.group(2) != "parquet":
                raise LeakageFeatureError(f"noncanonical episode parquet path {relative_value!r}")
            episode_index = int(match.group(1))
            if episode_index not in train_indices:
                raise LeakageFeatureError("train storage seal contains a development/final parquet")
        elif relative.parts[0] == "videos":
            match = episode_filename.fullmatch(relative.name)
            if len(relative.parts) < 4 or relative.parts[1] != "chunk-000" or match is None or match.group(2) != "mp4":
                raise LeakageFeatureError(f"noncanonical episode video path {relative_value!r}")
            if int(match.group(1)) not in train_indices:
                raise LeakageFeatureError("train storage seal contains a development/final video")
        elif relative.parts[0] == "meta":
            if "v35_training_bundle" in relative.parts:
                raise LeakageFeatureError("portable transfer-bundle files are outside the train storage seal")
        else:
            raise LeakageFeatureError(f"train storage seal contains an out-of-scope path {relative_value!r}")
    if ordered_paths != sorted(ordered_paths):
        raise LeakageFeatureError("train storage seal files are not in canonical project-relative order")
    expected_parquet = {f"data/chunk-000/episode_{episode.episode_index:06d}.parquet" for episode in train_episodes}
    present_parquet = {path for path in seen_paths if path.startswith("data/") and path.endswith(".parquet")}
    if present_parquet != expected_parquet:
        raise LeakageFeatureError("train storage seal does not contain exactly the train54 parquet files")
    required_meta = {
        "meta/episode_prompts.json",
        "meta/episodes.jsonl",
        "meta/info.json",
        "meta/tasks.jsonl",
    }
    if not required_meta.issubset(seen_paths):
        raise LeakageFeatureError(
            f"train storage seal is missing structural metadata: {sorted(required_meta - seen_paths)}"
        )
    return aggregate, provenance["selection"]["dataset_episode_frame_protocol_sha256"]


def _validate_norm_provenance(
    *,
    norm_dir: Path,
    manifest: artifacts.FrozenManifest,
    manifest_raw: Mapping[str, Any],
    episodes: Sequence[EpisodeSource],
    dataset_root: Path,
) -> tuple[str, str, str, str]:
    norm_path = norm_dir / NORM_STATS_FILE
    provenance_path = norm_dir / NORM_PROVENANCE_FILE
    provenance, provenance_bytes = _load_json_object(provenance_path, name="norm provenance")
    norm_sha256 = _sha256_file(norm_path)
    manifest_info = provenance.get("manifest")
    selection = provenance.get("selection")
    computation = provenance.get("computation")
    norm_info = provenance.get("norm_stats")
    if not all(isinstance(item, dict) for item in (manifest_info, selection, computation, norm_info)):
        raise LeakageFeatureError("norm provenance sections must be JSON objects")
    expected_ids = [episode.stable_id for episode in episodes]
    expected_indices = [episode.episode_index for episode in episodes]
    expected_excluded = sorted(
        episode.episode_index for split in ("development", "final_test") for episode in manifest.split(split)
    )
    expected_frame_counts = [episode.expected_frames for episode in episodes]
    frame_counts = selection.get("selected_episode_frame_counts")
    requested_batch_size = computation.get("requested_batch_size")
    if (
        provenance.get("schema_version") != 2
        or provenance.get("status") != "complete"
        or provenance.get("repo_id") != project_paths.V35_REPO_ID
        or provenance.get("scope") != NORM_SCOPE
        or manifest_info.get("path_relative") != FROZEN_MANIFEST.as_posix()
        or "path" in manifest_info
        or manifest_info.get("sha256") != manifest.sha256
        or manifest_info.get("schema_version") != 2
        or manifest_info.get("active_split") != "train"
        or manifest_info.get("split_seed") != 36
        or selection.get("dataset_num_episodes") != 70
        or selection.get("dataset_num_frames") != EXPECTED_DATASET_FRAMES
        or selection.get("selected_num_episodes") != 54
        or selection.get("selected_num_frames") != EXPECTED_TRAIN_FRAMES
        or selection.get("selected_episode_indices") != expected_indices
        or selection.get("selected_stable_ids") != expected_ids
        or selection.get("excluded_episode_indices") != expected_excluded
        or frame_counts != expected_frame_counts
        or selection.get("selected_num_frames") != sum(frame_counts)
        or computation.get("protocol") != NORM_COMPUTATION_PROTOCOL
        or computation.get("processed_base_rows") != sum(frame_counts)
        or computation.get("drop_last_rows") != 0
        or type(requested_batch_size) is not int
        or requested_batch_size <= 0
        or computation.get("num_batches_including_partial_final_batch")
        != (EXPECTED_TRAIN_FRAMES + requested_batch_size - 1) // requested_batch_size
        or norm_info.get("file") != NORM_STATS_FILE
        or norm_info.get("sha256") != norm_sha256
    ):
        raise LeakageFeatureError("norm provenance does not exactly bind the frozen train54 protocol")
    dataset_protocol_sha256 = artifacts.require_sha256(
        "dataset_episode_frame_protocol_sha256", selection.get("dataset_episode_frame_protocol_sha256")
    )
    storage_sha256, storage_protocol_sha256 = _validate_storage_seal(
        provenance,
        dataset_root=dataset_root,
        train_episodes=episodes,
    )
    if storage_protocol_sha256 != dataset_protocol_sha256:
        raise LeakageFeatureError("train storage and norm dataset protocol hashes disagree")
    raw_records = manifest_raw.get("episodes")
    if not isinstance(raw_records, list):
        raise LeakageFeatureError("frozen manifest is missing raw episode records")
    included_by_index = {
        record.get("episode_index", record.get("lerobot_episode_index")): record
        for record in raw_records
        if isinstance(record, dict) and bool(record.get("include", True))
    }
    episode_protocol = []
    for episode in manifest.episodes:
        record = included_by_index.get(episode.episode_index)
        frame_count = record.get("expected_num_frames") if isinstance(record, dict) else None
        if type(frame_count) is not int or frame_count <= 0 or record.get("stable_id") != episode.stable_id:
            raise LeakageFeatureError(f"manifest frame protocol is invalid for {episode.stable_id!r}")
        episode_protocol.append(
            {
                "episode_index": episode.episode_index,
                "stable_id": episode.stable_id,
                "split": episode.split,
                "include": True,
                "frame_count": frame_count,
            }
        )
    recomputed_dataset_protocol_sha256 = _canonical_sha256(
        {
            "manifest_sha256": manifest.sha256,
            "episodes": episode_protocol,
            "train_storage_sha256": storage_sha256,
        }
    )
    if dataset_protocol_sha256 != recomputed_dataset_protocol_sha256:
        raise LeakageFeatureError("norm dataset episode/frame/storage protocol SHA256 is not reproducible")
    return norm_sha256, artifacts.sha256_bytes(provenance_bytes), storage_sha256, dataset_protocol_sha256


def _validate_initialization(
    *,
    initialization_path: Path,
    manifest: artifacts.FrozenManifest,
    norm_stats_sha256: str,
    norm_provenance_sha256: str,
    train_storage_sha256: str,
    actual_step0_sha256: str,
    actual_official_base_sha256: str,
) -> tuple[dict[str, Any], str, Path, str]:
    identity, identity_bytes = _load_json_object(initialization_path, name="initialization identity")
    _self_hashed_json(identity, hash_key="identity_sha256", name="initialization identity")
    artifact_hashes = identity.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise LeakageFeatureError("initialization identity is missing artifact_hashes")
    if (
        identity.get("format_version") != 2
        or identity.get("config_name") != CONFIG_NAME
        or identity.get("official_source_uri") != OFFICIAL_BASE_URI
        or identity.get("initialization_seed") != EXPECTED_INITIALIZATION_SEED
        or identity.get("step0_checkpoint") != 0
        or identity.get("actual_step0_parameter_tree_sha256") != actual_step0_sha256
        or identity.get("source_tree_sha256") != actual_official_base_sha256
        or artifact_hashes.get("episode_manifest_sha256") != manifest.sha256
        or artifact_hashes.get("norm_stats_sha256") != norm_stats_sha256
        or artifact_hashes.get("norm_stats_provenance_sha256") != norm_provenance_sha256
        or artifact_hashes.get("train_storage_sha256") != train_storage_sha256
    ):
        raise LeakageFeatureError("initialization identity does not bind the authenticated fresh v3.5 inputs")

    graft_name = identity.get("graft_manifest_file")
    if (
        not isinstance(graft_name, str)
        or not graft_name
        or Path(graft_name).name != graft_name
        or graft_name != "initialization_graft_manifest.json"
    ):
        raise LeakageFeatureError("initialization identity has a noncanonical graft manifest filename")
    graft_path = initialization_path.parent / graft_name
    graft, graft_bytes = _load_json_object(graft_path, name="initialization graft manifest")
    graft_sha256 = artifacts.sha256_bytes(graft_bytes)
    graft_self_sha256 = _self_hashed_json(graft, hash_key="manifest_sha256", name="initialization graft manifest")
    tree_hashes = graft.get("tree_hashes")
    if (
        not isinstance(tree_hashes, dict)
        or graft.get("format_version") != 1
        or graft.get("loader") != "AuditedPartialCheckpointWeightLoader"
        or graft.get("checkpoint_root") != OFFICIAL_BASE_URI
        or tree_hashes.get("source_sha256") != actual_official_base_sha256
        or identity.get("graft_manifest_sha256") != graft_self_sha256
        or identity.get("source_tree_sha256") != tree_hashes.get("source_sha256")
        or artifact_hashes.get("initialization_graft_manifest_file_sha256") != graft_sha256
        or artifact_hashes.get("initialization_graft_manifest_self_sha256") != graft_self_sha256
    ):
        raise LeakageFeatureError("audited graft manifest does not authenticate the actual official base tree")
    return dict(identity), artifacts.sha256_bytes(identity_bytes), graft_path, graft_sha256


def authenticate_inputs(
    *,
    step0_params_relative: str | PurePosixPath,
    initialization_relative: str | PurePosixPath,
    manifest_sha256: str = FROZEN_MANIFEST_SHA256,
    parameter_tree_hasher: Callable[[Path], str] | None = None,
) -> AuthenticatedInputs:
    """Authenticate every producer input before any observation-column read."""

    _assert_frozen_protocol_hashes()
    if manifest_sha256 != FROZEN_MANIFEST_SHA256:
        raise LeakageFeatureError("the production manifest SHA256 is frozen; arbitrary manifest hashes are rejected")
    manifest_path = _relative_project_path(FROZEN_MANIFEST, name="frozen manifest")
    dataset_root = _relative_project_path(DATASET_ROOT, name="project-local dataset")
    norm_dir = _relative_project_path(NORM_DIR, name="v3.5 norm directory")
    step0_params_path = _relative_project_path(step0_params_relative, name="step-0 params")
    initialization_path = _relative_project_path(initialization_relative, name="initialization identity")
    official_base_path = _relative_project_path(OFFICIAL_BASE_PARAMS, name="official-base params")
    if not all(path.is_dir() for path in (dataset_root, norm_dir, step0_params_path, official_base_path)):
        raise LeakageFeatureError("dataset, norm, official-base params, and step-0 params must be directories")
    if not initialization_path.is_file():
        raise LeakageFeatureError("initialization identity must be a regular file")

    manifest = artifacts.load_frozen_manifest(manifest_path, expected_sha256=manifest_sha256)
    manifest_raw, _ = _load_json_object(manifest_path, name="frozen manifest")
    episodes = _manifest_episode_sources(manifest, manifest_raw)
    norm_stats_sha256, norm_provenance_sha256, storage_sha256, dataset_protocol_sha256 = _validate_norm_provenance(
        norm_dir=norm_dir,
        manifest=manifest,
        manifest_raw=manifest_raw,
        episodes=episodes,
        dataset_root=dataset_root,
    )
    if parameter_tree_hasher is None:
        parameter_tree_hasher = _parameter_tree_sha256
    actual_official_base_sha256 = parameter_tree_hasher(official_base_path)
    actual_step0_sha256 = parameter_tree_hasher(step0_params_path)
    identity, identity_sha256, graft_path, graft_sha256 = _validate_initialization(
        initialization_path=initialization_path,
        manifest=manifest,
        norm_stats_sha256=norm_stats_sha256,
        norm_provenance_sha256=norm_provenance_sha256,
        train_storage_sha256=storage_sha256,
        actual_step0_sha256=actual_step0_sha256,
        actual_official_base_sha256=actual_official_base_sha256,
    )
    return AuthenticatedInputs(
        manifest=manifest,
        manifest_raw=manifest_raw,
        episodes=episodes,
        dataset_root=dataset_root,
        norm_dir=norm_dir,
        norm_stats_sha256=norm_stats_sha256,
        norm_provenance_sha256=norm_provenance_sha256,
        train_storage_sha256=storage_sha256,
        dataset_protocol_sha256=dataset_protocol_sha256,
        initialization_path=initialization_path,
        initialization_raw=identity,
        initialization_sha256=identity_sha256,
        initialization_parameter_tree_sha256=actual_step0_sha256,
        official_base_source_tree_sha256=actual_official_base_sha256,
        graft_path=graft_path,
        graft_sha256=graft_sha256,
        step0_params_path=step0_params_path,
    )


def _parameter_tree_sha256(params_path: Path) -> str:
    # Imported lazily so validation/unit tests that inject a hasher stay CPU-only and do not
    # initialize JAX before the portable runtime environment is installed.
    from openpi.models import model as model_lib
    from openpi.training import weight_loaders

    params = model_lib.restore_params(params_path, restore_type=np.ndarray)
    return weight_loaders.parameter_tree_sha256(params)


def collect_feature_arrays(
    inputs: AuthenticatedInputs,
    extractor: EpisodeExtractor,
) -> dict[str, npt.NDArray[Any]]:
    """Collect one validated, already-aggregated row per frozen training episode."""

    rows: list[EpisodeFeatureRow] = []
    for episode in inputs.episodes:
        row = extractor.extract_episode(episode)
        if row.stable_id != episode.stable_id or row.frame_count != episode.d_count:
            raise LeakageFeatureError(
                f"episode extractor returned wrong identity/count for {episode.stable_id!r}: "
                f"{row.stable_id!r}, {row.frame_count}"
            )
        if set(row.features) != set(gate.ALL_FEATURES):
            raise LeakageFeatureError(
                f"episode {episode.stable_id!r} feature keys mismatch: "
                f"missing={sorted(set(gate.ALL_FEATURES) - set(row.features))}, "
                f"extra={sorted(set(row.features) - set(gate.ALL_FEATURES))}"
            )
        for name, value in row.features.items():
            array = np.asarray(value)
            if array.dtype != np.dtype(np.float32) or array.ndim != 1 or not array.size:
                raise LeakageFeatureError(f"{episode.stable_id} {name} must be a nonempty float32 vector")
            if not np.all(np.isfinite(array)):
                raise LeakageFeatureError(f"{episode.stable_id} {name} contains non-finite values")
        rows.append(row)
    if len(rows) != 54:
        raise LeakageFeatureError(f"collector must produce exactly 54 episode rows, got {len(rows)}")

    arrays: dict[str, npt.NDArray[Any]] = {
        "episode_stable_id": np.asarray([row.stable_id for row in rows]),
    }
    for name in gate.ALL_FEATURES:
        dimensions = {row.features[name].shape for row in rows}
        if len(dimensions) != 1:
            raise LeakageFeatureError(f"feature {name} changes dimension across episodes: {sorted(dimensions)}")
        arrays[name] = np.stack([row.features[name] for row in rows]).astype(np.float32, copy=False)
    return arrays


def _canonical_npz_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{name}.npy", date_time=NPZ_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    info.comment = b""
    info.extra = b""
    return info


def _validate_npz_arrays(arrays: Mapping[str, npt.NDArray[Any]]) -> None:
    if tuple(arrays) != NPZ_MEMBER_ORDER:
        raise LeakageFeatureError(f"canonical NPZ member order must be {NPZ_MEMBER_ORDER}")
    stable = np.asarray(arrays["episode_stable_id"])
    if stable.ndim != 1 or stable.dtype.kind != "U" or len(stable) != 54 or len(set(stable.tolist())) != 54:
        raise LeakageFeatureError("episode_stable_id must be exactly 54 unique canonical Unicode strings")
    for name in gate.ALL_FEATURES:
        array = np.asarray(arrays[name])
        if array.dtype != np.dtype(np.float32) or array.ndim != 2 or array.shape[0] != 54 or array.shape[1] < 1:
            raise LeakageFeatureError(f"{name} must be float32 [54,feature_dim]")
        if not array.flags.c_contiguous or not np.all(np.isfinite(array)):
            raise LeakageFeatureError(f"{name} must be C-contiguous and finite")


def write_canonical_npz(path: Path, arrays: Mapping[str, npt.NDArray[Any]]) -> str:
    """Write a deterministic ZIP_STORED NPZ without timestamps or pickle payloads."""

    ordered = {name: np.ascontiguousarray(arrays[name]) for name in NPZ_MEMBER_ORDER}
    _validate_npz_arrays(ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, mode="w", allowZip64=True) as archive:
            for name, array in ordered.items():
                with archive.open(_canonical_npz_info(name), mode="w", force_zip64=True) as stream:
                    np.lib.format.write_array(stream, array, version=(2, 0), allow_pickle=False)
        validate_canonical_npz(temporary)
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise LeakageFeatureError(f"refusing to overwrite existing feature NPZ {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(path)


def validate_canonical_npz(path: Path) -> None:
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            infos = archive.infolist()
            if [info.filename for info in infos] != [f"{name}.npy" for name in NPZ_MEMBER_ORDER]:
                raise LeakageFeatureError("feature NPZ has noncanonical member names/order")
            for info in infos:
                if (
                    info.date_time != NPZ_TIMESTAMP
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.create_system != 3
                    or info.external_attr != 0o600 << 16
                    or info.comment
                    or info.extra
                ):
                    raise LeakageFeatureError(f"feature NPZ member {info.filename!r} has noncanonical ZIP metadata")
                with archive.open(info, mode="r") as member:
                    if np.lib.format.read_magic(member) != (2, 0):
                        raise LeakageFeatureError(
                            f"feature NPZ member {info.filename!r} must use the frozen NPY v2.0 header"
                        )
            with np.load(path, allow_pickle=False) as archive_arrays:
                arrays = {name: archive_arrays[name] for name in NPZ_MEMBER_ORDER}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, LeakageFeatureError):
            raise
        raise LeakageFeatureError(f"cannot validate feature NPZ {path}: {exc}") from exc
    _validate_npz_arrays(arrays)


def _write_bytes_once(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(contents)
    except FileExistsError as exc:
        raise LeakageFeatureError(f"refusing to overwrite existing artifact {path}") from exc


def emit_feature_artifacts(
    *,
    inputs: AuthenticatedInputs,
    arrays: Mapping[str, npt.NDArray[Any]],
    output_envelope: Path,
) -> Path:
    """Emit reducer-compatible immutable snapshots, deterministic NPZ, and envelope."""

    _assert_frozen_protocol_hashes()
    output_envelope = output_envelope.resolve()
    try:
        output_envelope.relative_to(project_paths.memory_project_root())
    except ValueError as exc:
        raise LeakageFeatureError("output envelope must remain inside memory_project") from exc
    if output_envelope.suffix != ".json":
        raise LeakageFeatureError("output envelope must use a .json filename")
    output_envelope.parent.mkdir(parents=True, exist_ok=True)
    npz_path = output_envelope.with_suffix(".npz")
    preprocessing_path = output_envelope.with_name(f"{output_envelope.stem}_preprocessing.json")
    initialization_snapshot = output_envelope.with_name(f"{output_envelope.stem}_initialization_manifest.json")
    graft_snapshot = output_envelope.with_name(f"{output_envelope.stem}_initialization_graft_manifest.json")
    for path in (output_envelope, npz_path, preprocessing_path, initialization_snapshot, graft_snapshot):
        if path.exists():
            raise LeakageFeatureError(f"refusing to overwrite existing artifact {path}")

    if (
        _sha256_file(inputs.initialization_path) != inputs.initialization_sha256
        or _sha256_file(inputs.graft_path) != inputs.graft_sha256
    ):
        raise LeakageFeatureError("initialization identity or audited graft changed during feature extraction")

    _write_bytes_once(initialization_snapshot, inputs.initialization_path.read_bytes())
    _write_bytes_once(graft_snapshot, inputs.graft_path.read_bytes())
    npz_sha256 = write_canonical_npz(npz_path, arrays)
    preprocessing = artifacts.artifact_envelope(
        gate.PREPROCESSING_SCHEMA_VERSION,
        {
            "episode_manifest_sha256": inputs.manifest.sha256,
            "final_preprocessing": True,
            "image_preprocessing_sha256": FROZEN_IMAGE_PREPROCESSING_SHA256,
            "norm_stats_sha256": inputs.norm_stats_sha256,
            "protocol_sha256": FROZEN_PREPROCESSING_PROTOCOL_SHA256,
            "split_assignment_sha256": inputs.manifest.split_assignment_sha256,
            "status": "frozen",
        },
    )
    artifacts.write_canonical_envelope(
        preprocessing_path,
        preprocessing,
        schema_version=gate.PREPROCESSING_SCHEMA_VERSION,
    )
    payload = {
        "aggregation_protocol": gate.AGGREGATION_PROTOCOL,
        "completed_updates": 0,
        "development_or_final_test_accessed": False,
        "episode_manifest_sha256": inputs.manifest.sha256,
        "feature_npz": {"path": npz_path.name, "sha256": npz_sha256},
        "feature_protocol_sha256": FROZEN_FEATURE_PROTOCOL_SHA256,
        "initialization_manifest": {
            "path": initialization_snapshot.name,
            "sha256": _sha256_file(initialization_snapshot),
        },
        "initialization_parameter_tree_sha256": inputs.initialization_parameter_tree_sha256,
        "preprocessing_artifact": {
            "path": preprocessing_path.name,
            "sha256": _sha256_file(preprocessing_path),
        },
        "population_split": "train",
        "split_assignment_sha256": inputs.manifest.split_assignment_sha256,
    }
    envelope = artifacts.artifact_envelope(gate.FEATURE_SCHEMA_VERSION, payload)
    artifacts.write_canonical_envelope(output_envelope, envelope, schema_version=gate.FEATURE_SCHEMA_VERSION)
    # Load through the consumer before declaring success.  This catches accidental producer /
    # reducer contract drift immediately, while the immutable source files are still present.
    gate.load_feature_dataset(output_envelope, manifest=inputs.manifest)
    return output_envelope


class ProductionEpisodeExtractor:
    """H100-backed exact final-D extractor over the project-local converted parquet."""

    def __init__(self, inputs: AuthenticatedInputs, *, batch_size: int):
        if batch_size <= 0:
            raise LeakageFeatureError("batch_size must be positive")
        self.inputs = inputs
        self.batch_size = batch_size
        self._train_indices = {episode.episode_index for episode in inputs.episodes}
        self._task_names = self._load_task_names(inputs.dataset_root)
        self._verify_storage_files()
        self._initialize_runtime()

    @staticmethod
    def _load_task_names(dataset_root: Path) -> dict[int, str]:
        names: dict[int, str] = {}
        try:
            lines = (dataset_root / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise LeakageFeatureError(f"cannot read tasks.jsonl: {exc}") from exc
        for line in lines:
            if not line.strip():
                continue
            value = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            index, task = value.get("task_index"), value.get("task")
            if type(index) is not int or not isinstance(task, str) or index in names:
                raise LeakageFeatureError("tasks.jsonl contains an invalid or duplicate task")
            names[index] = task
        if set(names.values()) != {
            "open both lids",
            "inspect both bins",
            "close both lids and reset arms",
            "wait; target bin is left",
            "wait; target bin is right",
            "open left bin",
            "open right bin",
        }:
            raise LeakageFeatureError("tasks.jsonl does not contain the frozen seven-task vocabulary")
        return names

    def _verify_storage_files(self) -> None:
        provenance, _ = _load_json_object(
            self.inputs.norm_dir / NORM_PROVENANCE_FILE,
            name="norm provenance",
        )
        storage = provenance["train_storage"]
        for record in storage["files"]:
            relative = PurePosixPath(record["path"])
            path = self.inputs.dataset_root.joinpath(*relative.parts)
            if not path.is_file() or path.is_symlink() or path.stat().st_size != record["size"]:
                raise LeakageFeatureError(f"sealed train storage file is missing or changed: {relative.as_posix()}")
            if _sha256_file(path) != record["sha256"]:
                raise LeakageFeatureError(f"sealed train storage SHA256 changed: {relative.as_posix()}")

    def _initialize_runtime(self) -> None:
        import jax
        import jax.numpy as jnp

        from openpi.models import model as model_lib
        from openpi.shared import nnx_utils
        from openpi.shared import normalize as normalize_lib
        from openpi.training import config as config_lib
        from openpi.training import weight_loaders
        import openpi.transforms as transforms

        config = config_lib.get_config(CONFIG_NAME)
        if (
            config.name != CONFIG_NAME
            or getattr(config.model, "memory_architecture", None) != "v32_layer8_dual_query"
            or not getattr(config.model, "memory_v35_enabled", False)
            or config.model.memory_query_tokens != 16
            or config.model.memory.association_mode != "pooled_frame"
            or config.model.memory.write_rule != "delta_output"
        ):
            raise LeakageFeatureError("registered v3.5 model no longer matches the frozen feature protocol")
        if (
            _sha256_file(self.inputs.norm_dir / NORM_STATS_FILE) != self.inputs.norm_stats_sha256
            or _sha256_file(self.inputs.norm_dir / NORM_PROVENANCE_FILE) != self.inputs.norm_provenance_sha256
        ):
            raise LeakageFeatureError("v3.5 norm stats or provenance changed after authentication")
        params = model_lib.restore_params(self.inputs.step0_params_path, restore_type=np.ndarray)
        if weight_loaders.parameter_tree_sha256(params) != self.inputs.initialization_parameter_tree_sha256:
            raise LeakageFeatureError("step-0 params changed between authentication and model load")
        model = config.model.load(params)
        model.eval()
        data_config = config.data.create(config.assets_dirs, config.model)
        norm_stats = normalize_lib.load(self.inputs.norm_dir)
        self._data_input_transform = transforms.compose(data_config.data_transforms.inputs)
        self._normalize = transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
        self._model_transform = transforms.compose(data_config.model_transforms.inputs)
        self._interface = nnx_utils.module_jit(model.v32_memory_interface_step)
        self._pool_kv = nnx_utils.module_jit(model.memory.pool_kv)
        self._memory_state = model.memory.init_state(self.batch_size)
        if any(np.dtype(leaf.dtype) != np.dtype(np.float32) for leaf in jax.tree.leaves(self._memory_state)):
            raise LeakageFeatureError("fresh v3.5 fast-weight and momentum state must be entirely float32")
        self._model_lib = model_lib
        self._jax = jax
        self._jnp = jnp

    @staticmethod
    def _decode_image(value: Any, *, field: str, frame: int) -> np.ndarray:
        from PIL import Image

        payload = value.get("bytes") if isinstance(value, dict) else value
        if not isinstance(payload, bytes):
            raise LeakageFeatureError(f"{field} frame {frame} does not contain inline image bytes")
        try:
            with Image.open(io.BytesIO(payload)) as image:
                decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except (OSError, ValueError) as exc:
            raise LeakageFeatureError(f"cannot decode {field} frame {frame}: {exc}") from exc
        if decoded.shape != (480, 640, 3):
            raise LeakageFeatureError(f"{field} frame {frame} has unexpected decoded shape {decoded.shape}")
        return decoded

    def _parquet_path(self, episode: EpisodeSource) -> Path:
        if episode.episode_index not in self._train_indices:
            raise LeakageFeatureError("development/final parquet observation access is forbidden")
        return self.inputs.dataset_root / "data" / "chunk-000" / f"episode_{episode.episode_index:06d}.parquet"

    def _read_final_d_rows(self, episode: EpisodeSource) -> list[dict[str, Any]]:
        import pyarrow.parquet as pq

        path = self._parquet_path(episode)
        try:
            table = pq.read_table(
                path,
                columns=list(PARQUET_COLUMNS),
                filters=[
                    ("frame_index", ">=", episode.d_start),
                    ("frame_index", "<=", episode.d_end),
                ],
            )
        except Exception as exc:
            raise LeakageFeatureError(f"cannot read final-D rows for {episode.stable_id}: {exc}") from exc
        rows = table.to_pylist()
        frame_indices = [int(row["frame_index"]) for row in rows]
        if frame_indices != list(range(episode.d_start, episode.d_end + 1)):
            raise LeakageFeatureError(f"{episode.stable_id} final-D frame indices are incomplete or reordered")
        wait_task = EXPECTED_WAIT_TASKS[episode.target_side]
        for row in rows:
            if int(row["episode_index"]) != episode.episode_index:
                raise LeakageFeatureError(f"{episode.stable_id} parquet contains a foreign episode_index")
            task_index = int(row["task_index"])
            if self._task_names.get(task_index) != wait_task:
                raise LeakageFeatureError(f"{episode.stable_id} d_valid row is not the exact side-specific wait task")
            state = np.asarray(row["state"], dtype=np.float32)
            if state.shape != (14,) or not np.all(np.isfinite(state)):
                raise LeakageFeatureError(f"{episode.stable_id} final-D state is not finite 14-D float32")
        return rows

    def _observation(self, row: Mapping[str, Any], episode: EpisodeSource) -> tuple[Any, np.ndarray]:
        frame = int(row["frame_index"])
        transformed = self._data_input_transform(
            {
                "observation/image": self._decode_image(row["image"], field="image", frame=frame),
                "observation/left_wrist_image": self._decode_image(
                    row["left_wrist_image"], field="left_wrist_image", frame=frame
                ),
                "observation/right_wrist_image": self._decode_image(
                    row["right_wrist_image"], field="right_wrist_image", frame=frame
                ),
                "observation/state": np.asarray(row["state"], dtype=np.float32),
                "prompt": episode.prompt,
            }
        )
        transformed = self._normalize(transformed)
        normalized_state = np.asarray(transformed["state"], dtype=np.float32)
        if normalized_state.shape != (14,) or not np.all(np.isfinite(normalized_state)):
            raise LeakageFeatureError(f"{episode.stable_id} normalized state is not finite 14-D")
        transformed = self._model_transform(transformed)
        batched = self._jax.tree.map(lambda value: self._jnp.asarray(value)[None, ...], transformed)
        return self._model_lib.Observation.from_dict(batched), normalized_state

    def _extract_batch(
        self, observations: Sequence[Any], normalized_states: Sequence[np.ndarray]
    ) -> dict[str, np.ndarray]:
        real_count = len(observations)
        if not 0 < real_count <= self.batch_size:
            raise LeakageFeatureError("internal final-D batch is empty or oversized")
        padded_observations = list(observations) + [observations[-1]] * (self.batch_size - real_count)
        observation = self._jax.tree.map(
            lambda *values: self._jnp.concatenate(values, axis=0),
            *padded_observations,
        )
        interface = self._interface(observation, self._memory_state)
        pooled = self._pool_kv(interface["write_keys"], interface["write_values"])
        key = np.asarray(self._jax.device_get(pooled["pooled_key"][:real_count]), dtype=np.float32)
        value = np.asarray(self._jax.device_get(pooled["pooled_value"][:real_count]), dtype=np.float32)
        valid = np.asarray(self._jax.device_get(pooled["association_valid"][:real_count]), dtype=bool)
        if not np.all(valid):
            raise LeakageFeatureError("fresh step-0 final-D pooling produced an invalid kbar/vbar association")
        write_tokens = np.asarray(
            self._jax.device_get(interface["write_tokens"][:real_count]),
            dtype=np.float32,
        )
        images = {
            name: np.asarray(self._jax.device_get(observation.images[name][:real_count]), dtype=np.float32)
            for name in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        }
        tokens = np.asarray(self._jax.device_get(observation.tokenized_prompt[:real_count]), dtype=np.int32)
        token_valid = np.asarray(self._jax.device_get(observation.tokenized_prompt_mask[:real_count]), dtype=bool)
        state_mask = np.asarray(self._jax.device_get(observation.token_state_mask[:real_count]), dtype=bool)
        prompt_valid = token_valid & ~state_mask
        prompt = np.concatenate(
            [np.where(prompt_valid, tokens, 0).astype(np.float32), prompt_valid.astype(np.float32)],
            axis=-1,
        )
        states = np.stack(normalized_states).astype(np.float32, copy=False)
        flat_images = [images[name].reshape(real_count, -1) for name in images]
        all_images = np.concatenate(flat_images, axis=-1)
        outputs = {
            "primary_final_images_state": np.concatenate([all_images, states], axis=-1),
            "primary_step0_kbar_vbar": np.concatenate([key, value], axis=-1),
            "descriptive_top_only": flat_images[0],
            "descriptive_left_wrist_only": flat_images[1],
            "descriptive_right_wrist_only": flat_images[2],
            "descriptive_all_images_only": all_images,
            "descriptive_state_only": states,
            "descriptive_prompt_only": prompt,
            "descriptive_layer8_only": np.mean(write_tokens, axis=1, dtype=np.float32),
            "descriptive_kbar_only": key,
            "descriptive_vbar_only": value,
        }
        for name, array in outputs.items():
            if array.dtype != np.dtype(np.float32) or not np.all(np.isfinite(array)):
                raise LeakageFeatureError(f"production batch feature {name} is not finite float32")
        return outputs

    def extract_episode(self, episode: EpisodeSource) -> EpisodeFeatureRow:
        if episode.episode_index not in self._train_indices:
            raise LeakageFeatureError("extract_episode accepts training episodes only")
        rows = self._read_final_d_rows(episode)
        sums: dict[str, np.ndarray] = {}
        for start in range(0, len(rows), self.batch_size):
            observations: list[Any] = []
            states: list[np.ndarray] = []
            for row in rows[start : start + self.batch_size]:
                observation, state = self._observation(row, episode)
                observations.append(observation)
                states.append(state)
            batch = self._extract_batch(observations, states)
            for name, values in batch.items():
                if name not in sums:
                    sums[name] = np.zeros(values.shape[1], dtype=np.float64)
                # Keep the aggregate invariant to GPU batch size: add each raw frame in fixed
                # frame-index order rather than reducing the batch on device.
                for value in values:
                    sums[name] += value.astype(np.float64)
        features = {name: np.asarray(total / len(rows), dtype=np.float32) for name, total in sums.items()}
        return EpisodeFeatureRow(stable_id=episode.stable_id, frame_count=len(rows), features=features)


def produce(
    *,
    step0_params_relative: str | PurePosixPath,
    initialization_relative: str | PurePosixPath,
    output_relative: str | PurePosixPath,
    batch_size: int,
) -> Path:
    project_paths.configure_v35_runtime_environment()
    inputs = authenticate_inputs(
        step0_params_relative=step0_params_relative,
        initialization_relative=initialization_relative,
    )
    extractor = ProductionEpisodeExtractor(inputs, batch_size=batch_size)
    arrays = collect_feature_arrays(inputs, extractor)
    output = _relative_project_path(output_relative, name="output envelope", must_exist=False)
    return emit_feature_artifacts(inputs=inputs, arrays=arrays, output_envelope=output)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step0-params",
        required=True,
        help="memory_project-relative path to the fresh completed_updates=0 params directory.",
    )
    parser.add_argument(
        "--initialization-manifest",
        required=True,
        help="memory_project-relative path to the matching run initialization_manifest.json.",
    )
    parser.add_argument(
        "--output",
        default="v35/diagnostics/gate_b/leakage_features.json",
        help="memory_project-relative output envelope (created exclusively; never overwritten).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Fixed H100 inference batch size.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = produce(
        step0_params_relative=args.step0_params,
        initialization_relative=args.initialization_manifest,
        output_relative=args.output,
        batch_size=args.batch_size,
    )
    print(output.relative_to(project_paths.memory_project_root()).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
