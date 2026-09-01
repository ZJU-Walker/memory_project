"""Produce the fail-closed v3.5 Gate-A data decision and its detailed audit.

The production audit has a deliberately narrow access boundary:

* all 71 raw records are authenticated from the frozen manifest and its hashed structural
  sidecars (90 converted episodes plus the approved exclusion);
* only the 74 training parquet files are opened, and only their scalar identity/task columns
  are decoded;
* development/final-test parquet, image, state, action, and video payloads are never opened.

Both outputs are canonical self-hashed JSON and are created exactly once.  Every path stored in
them is relative to ``memory_project`` so the complete project can be copied to another cluster.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np

# Import the data loader (and therefore Torch) before the training config.  Importing
# config first can initialize JAX/XLA before Torch in a fresh interpreter and has
# produced a deterministic native crash on the production hosts.
# isort: off
from openpi.training import data_loader
from openpi import transforms
from openpi.shared import project_paths
from openpi.training import config as train_config
# isort: on

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import v35_calibration_replay as replay  # noqa: E402
import v35_collect_calibration_replay as collector  # noqa: E402
import v35_gate_artifacts as artifacts  # noqa: E402

DECISION_SCHEMA_VERSION = "openpi.v35.data-gate-decision.v1"
REPORT_SCHEMA_VERSION = "openpi.v35.data-gate-report.v1"
CRITERIA_VERSION = "openpi.v35.gate-a.rev5.v1"
CONFIG_NAME = "pi05_yam_mem_v35"
RAW_EPISODES = 71
INCLUDED_EPISODES = 70
TRAIN_EPISODES = 54
DEVELOPMENT_EPISODES = 8
FINAL_TEST_EPISODES = 8

REQUIRED_CHECKS = frozenset(
    {
        "block_confound_audit_pass",
        "every_train_d_window_has_final_eligible_e_anchor",
        "every_train_episode_has_eligible_sampled_e_step",
        "every_train_episode_has_skip_o_d_candidate",
        "manual_review_complete",
        "stable_id_mapping_exact",
        "stable_split_frozen",
        "successful_commit_accounting_complete",
        "task_vocabulary_exact",
        "train_only_normalization",
        "zero_state_invalid_d_loss_steps",
    }
)

_EXPECTED_PHASES = (
    "open both lids",
    "inspect both bins",
    "close both lids and reset arms",
    "wait; target bin is {side}",
    "open {side} bin",
)
_DATA_PARQUET_RE = re.compile(r"data/chunk-\d{3}/episode_(\d{6})\.parquet")
_EPISODE_PAYLOAD_RE = re.compile(r"(?:^|/)episode_(\d{6})\.(?:parquet|mp4)$")


class DataGateError(artifacts.GateArtifactError):
    """Raised before either Gate-A artifact is written."""


@dataclasses.dataclass(frozen=True)
class Candidate:
    family: str
    start_frame: int
    sampled_e_count: int
    n_delay: int
    commit_frames: tuple[int, ...]
    d_frames: tuple[int, ...]
    use_pressure_frames: tuple[int, ...]
    state_invalid_d_steps: int
    credit_reachable_d_steps: int


@dataclasses.dataclass(frozen=True)
class CalibrationDatasetIdentity:
    kind: str
    artifact_id: str
    file_sha256: str
    dataset_protocol_sha256: str
    successful_commit_metrics: tuple[Mapping[str, Any], ...]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DataGateError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, *, name: str) -> tuple[dict[str, Any], bytes]:
    try:
        encoded = Path(path).read_bytes()
        value = json.loads(encoded, object_pairs_hook=_strict_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataGateError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DataGateError(f"{name} must be a JSON object")
    return value, encoded


def _read_jsonl(path: Path, *, name: str) -> list[dict[str, Any]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        values = [json.loads(line, object_pairs_hook=_strict_object) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise DataGateError(f"cannot read {name} {path}: {exc}") from exc
    if any(not isinstance(value, dict) for value in values):
        raise DataGateError(f"{name} must contain only JSON objects")
    return values


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            while block := stream.read(chunk_size):
                digest.update(block)
    except OSError as exc:
        raise DataGateError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _project_relative(path: Path, *, name: str) -> str:
    try:
        return project_paths.project_relative_path(path).as_posix()
    except project_paths.ProjectRootError as exc:
        raise DataGateError(f"{name} must be inside memory_project: {path}") from exc


def _canonical_custom_envelope(value: Mapping[str, Any], *, id_key: str) -> bool:
    if set(value) != {"artifact_sha256", id_key, "hash_scope", "payload"}:
        return False
    payload = value.get("payload")
    if not isinstance(payload, dict):
        return False
    digest = artifacts.sha256_bytes(artifacts.canonical_json_bytes(payload))
    return (
        value.get("artifact_sha256") == digest
        and value.get(id_key) == f"sha256:{digest}"
        and value.get("hash_scope") == "SHA256 of canonical_json($.payload)"
    )


def _load_calibration_dataset_identity(
    path: Path,
    *,
    protocol: replay.FrozenDataProtocol,
    manifest: artifacts.FrozenManifest,
) -> CalibrationDatasetIdentity:
    """Authenticate the sealed calibration artifact; a preflight alone cannot pass Gate A."""
    raw, encoded = _read_json(path, name="calibration dataset descriptor")
    if encoded != artifacts.canonical_json_bytes(raw) + b"\n":
        raise DataGateError("calibration dataset descriptor is not canonical JSON")
    payload = raw.get("payload")
    schema = payload.get("schema_version") if isinstance(payload, dict) else None

    if schema == replay.PREFLIGHT_SCHEMA:
        raise DataGateError(
            "a calibration preflight has no successful commits; passing Gate A requires the sealed "
            "injection-calibration artifact"
        )

    if schema == "openpi.v35.injection-calibration.v1":
        if not _canonical_custom_envelope(raw, id_key="calibration_id"):
            raise DataGateError("calibration artifact payload hash/ID is invalid")
        if payload.get("status") != "pass" or payload.get("gates", {}).get("passes") is not True:
            raise DataGateError("calibration descriptor is not a passing calibration artifact")
        population = payload.get("population")
        provenance = payload.get("provenance")
        if not isinstance(population, dict) or not isinstance(provenance, dict):
            raise DataGateError("calibration artifact is missing population/provenance")
        if (
            population.get("split") != "train"
            or population.get("episode_count") != TRAIN_EPISODES
            or population.get("stable_ids") != list(protocol.train_stable_ids)
            or provenance.get("split_sha256") != manifest.sha256
            or provenance.get("dataset_sha256") != protocol.dataset_protocol_sha256
        ):
            raise DataGateError("calibration artifact is stale or is not the exact frozen train-54 dataset")
        if population.get("clean_slots_per_episode") != 16:
            raise DataGateError("calibration artifact did not retain all 16 clean committed-read slots")
        required_provenance_hashes = {
            "collector_source_sha256",
            "dataset_sha256",
            "input_npz_sha256",
            "observed_membership_sha256",
            "official_base_source_sha256",
            "preflight_sha256",
            "replay_protocol_sha256",
            "source_sha256",
            "split_sha256",
        }
        if not required_provenance_hashes <= set(provenance) or any(
            not isinstance(provenance[name], str) or artifacts.SHA256_RE.fullmatch(provenance[name]) is None
            for name in required_provenance_hashes
        ):
            raise DataGateError("calibration artifact is missing a required replay/provenance SHA256")
        npz_keys = provenance.get("npz_keys")
        required_npz_keys = {
            "clean_raw_retrieved",
            "episode_split",
            "episode_stable_id",
            "layer8_residual",
            "n_delay",
        }
        if not isinstance(npz_keys, list) or not required_npz_keys <= set(npz_keys):
            raise DataGateError("calibration artifact does not identify its clean commit/residual replay arrays")
        statistics = payload.get("statistics")
        episode_metrics = statistics.get("episode_metrics") if isinstance(statistics, dict) else None
        if not isinstance(episode_metrics, list) or len(episode_metrics) != TRAIN_EPISODES:
            raise DataGateError("calibration artifact does not contain 54 episode commit/read metrics")
        expected_metric_keys = {
            "clean_injected_to_residual_rms",
            "clean_episode_raw_rms",
            "clean_slot_raw_rms_p50",
            "decayed_injected_to_residual_rms",
            "decayed_retained_amplitude",
            "n_delay",
            "stable_id",
        }
        if [item.get("stable_id") if isinstance(item, dict) else None for item in episode_metrics] != list(
            protocol.train_stable_ids
        ):
            raise DataGateError("calibration commit/read metric membership/order differs from frozen train-54")
        for item in episode_metrics:
            if not isinstance(item, dict) or set(item) != expected_metric_keys:
                raise DataGateError("calibration episode commit/read metric has the wrong schema")
            positive = (
                "clean_injected_to_residual_rms",
                "clean_episode_raw_rms",
                "clean_slot_raw_rms_p50",
                "decayed_injected_to_residual_rms",
                "decayed_retained_amplitude",
            )
            if any(
                isinstance(item[name], bool)
                or not isinstance(item[name], int | float)
                or not math.isfinite(float(item[name]))
                or float(item[name]) <= 0.0
                for name in positive
            ):
                raise DataGateError(
                    f"calibration has nonfinite/degenerate commit/read evidence for {item['stable_id']}"
                )
            if type(item["n_delay"]) is not int or item["n_delay"] < 0:
                raise DataGateError(f"calibration has invalid n_delay for {item['stable_id']}")
        return CalibrationDatasetIdentity(
            kind="injection_calibration",
            artifact_id=str(raw["calibration_id"]),
            file_sha256=artifacts.sha256_bytes(encoded),
            dataset_protocol_sha256=str(provenance["dataset_sha256"]),
            successful_commit_metrics=tuple(episode_metrics),
        )
    raise DataGateError("descriptor must be a sealed passing v3.5 injection-calibration artifact")


def _segments_from_label(
    *, manifest_path: Path, raw_manifest: dict[str, Any], record: dict[str, Any]
) -> tuple[dict[str, Any], ...]:
    try:
        label_path = data_loader._v35_resolve_label_path(manifest_path, raw_manifest, record)  # noqa: SLF001
    except ValueError as exc:
        raise DataGateError(str(exc)) from exc
    expected_sha = artifacts.require_sha256(f"{record['stable_id']} label_sha256", record.get("label_sha256"))
    if _sha256_file(label_path) != expected_sha:
        raise DataGateError(f"frozen structural label hash changed for {record['stable_id']}")
    try:
        value = json.loads(label_path.read_text(encoding="utf-8"), object_pairs_hook=_strict_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataGateError(f"cannot parse structural label {label_path}: {exc}") from exc
    if not isinstance(value, list) or len(value) != 5 or any(not isinstance(item, dict) for item in value):
        raise DataGateError(f"{record['stable_id']} must have exactly five structural label segments")
    side = str(record.get("target_side", "")).strip().lower()
    expected_tasks = tuple(task.format(side=side) for task in _EXPECTED_PHASES)
    if tuple(item.get("task") for item in value) != expected_tasks:
        raise DataGateError(f"{record['stable_id']} has a noncanonical/side-inconsistent phase vocabulary")
    next_start = 0
    for segment in value:
        start, end = segment.get("start"), segment.get("end")
        if type(start) is not int or type(end) is not int or start != next_start or end < start:
            raise DataGateError(f"{record['stable_id']} structural labels are not contiguous")
        next_start = end + 1
    if next_start != record.get("expected_num_frames"):
        raise DataGateError(f"{record['stable_id']} structural labels do not cover the frozen frame count")
    return tuple(value)


def _load_and_validate_manifest_structure(
    manifest_path: Path, *, manifest_sha256: str, data_config: train_config.DataConfig
) -> tuple[artifacts.FrozenManifest, dict[str, Any], dict[int, dict[str, Any]], dict[int, tuple[dict[str, Any], ...]]]:
    manifest = artifacts.load_frozen_manifest(manifest_path, expected_sha256=manifest_sha256)
    raw, encoded = _read_json(manifest_path, name="frozen manifest")
    if artifacts.sha256_bytes(encoded) != manifest.sha256:
        raise DataGateError("frozen manifest changed while Gate A was running")
    entries = raw.get("episodes")
    if not isinstance(entries, list) or len(entries) != RAW_EPISODES:
        raise DataGateError("Gate A requires exactly 71 raw manifest records")
    included = [record for record in entries if isinstance(record, dict) and record.get("include", True) is True]
    excluded = [record for record in entries if isinstance(record, dict) and record.get("include", True) is False]
    if len(included) != INCLUDED_EPISODES or len(excluded) != 1:
        raise DataGateError("Gate A requires exactly 90 included records and one approved exclusion")
    if (
        excluded[0].get("stable_id") != "0830_bin_part2/demo14"
        or excluded[0].get("approval_status") != "user_approved_exclusion"
        or not str(excluded[0].get("exclude_reason", "")).strip()
    ):
        raise DataGateError("the sole raw exclusion is not the approved incomplete part2/demo14")
    if raw.get("task_vocabulary") != list(data_config.memory_subtask_vocab):
        raise DataGateError("frozen manifest task vocabulary/order differs from registered v3.5")
    if (
        raw.get("review_status") != "frozen"
        or raw.get("approval", {}).get("reviewer") != "user"
        or any(record.get("approval_status") != "user_approved" for record in included)
    ):
        raise DataGateError("the 71-record manifest is not completely user approved/frozen")
    source = raw.get("source_approved_manifest")
    if not isinstance(source, dict) or set(source) != {"file", "sha256"}:
        raise DataGateError("frozen manifest is missing its approved-manifest descriptor")
    source_name = source.get("file")
    if not isinstance(source_name, str) or Path(source_name).name != source_name:
        raise DataGateError("approved-manifest descriptor must be a sibling basename")
    approved_path = manifest_path.parent / source_name
    if _sha256_file(approved_path) != artifacts.require_sha256("approved manifest", source.get("sha256")):
        raise DataGateError("user-approved source manifest bytes changed")
    approved, _ = _read_json(approved_path, name="approved manifest")
    if (
        approved.get("schema_version") != 1
        or approved.get("dataset_version") != "v36"
        or approved.get("review_status") != "user_approved"
        or approved.get("approval", {}).get("reviewer") != "user"
        or len(approved.get("episodes", [])) != RAW_EPISODES
    ):
        raise DataGateError("approved source manifest no longer carries the complete user sign-off")

    try:
        visibility = data_loader._v35_load_hashed_sidecar(  # noqa: SLF001
            manifest_path, raw.get("e_visibility_review"), name="E-visibility review"
        )
        d_valid = data_loader._v35_load_hashed_sidecar(  # noqa: SLF001
            manifest_path, raw.get("d_valid_sidecar"), name="D_valid sidecar"
        )
        block = data_loader._v35_load_hashed_sidecar(  # noqa: SLF001
            manifest_path, raw.get("block_confound_audit"), name="block-confound audit"
        )
    except ValueError as exc:
        raise DataGateError(str(exc)) from exc
    visibility_by_id = {item.get("stable_id"): item.get("e_visibility") for item in visibility.get("episodes", [])}
    d_valid_by_id = {item.get("stable_id"): item.get("d_valid") for item in d_valid.get("episodes", [])}
    if (
        visibility.get("schema_version") != "openpi.v36.e-visibility-review.v1"
        or visibility.get("status") != "user_approved"
        or len(visibility_by_id) != INCLUDED_EPISODES
        or d_valid.get("schema_version") != "openpi.v36.d-valid.v1"
        or d_valid.get("status") != "complete"
        or len(d_valid_by_id) != INCLUDED_EPISODES
    ):
        raise DataGateError("hashed E-visibility/D-valid sidecars do not cover the exact converted population")
    expected_block = data_loader._v35_block_confound_summary(entries)  # noqa: SLF001
    audit_fields = [
        {
            key: record.get(key)
            for key in ("stable_id", "include", "collection", "part", "object", "target_side", "timestamp")
        }
        for record in entries
        if bool(record.get("include", True))
    ]
    audit_fields_sha256 = hashlib.sha256(
        data_loader._v35_canonical_json(audit_fields).encode("utf-8")  # noqa: SLF001
    ).hexdigest()
    block_descriptor = raw.get("block_confound_audit")
    if (
        block.get("schema_version") != "openpi.v36.block-confound-audit.v1"
        or block.get("status") != "pass"
        or block.get("manifest_fields_only") is not True
        or block.get("manifest_fields_sha256") != audit_fields_sha256
        or not isinstance(block_descriptor, dict)
        or block_descriptor.get("manifest_fields_sha256") != audit_fields_sha256
        or block.get("summary") != expected_block
        or expected_block.get("pass") is not True
    ):
        raise DataGateError("0830 block-confound audit does not reproduce from frozen manifest fields")
    try:
        expected_splits = data_loader._v35_expected_frozen_splits(entries, seed=36)  # noqa: SLF001
    except ValueError as exc:
        raise DataGateError(str(exc)) from exc
    if any(record.get("split") != expected_splits.get(record.get("stable_id")) for record in included):
        raise DataGateError("split assignment does not reproduce the seeded manifest-only algorithm")

    by_index: dict[int, dict[str, Any]] = {}
    segments: dict[int, tuple[dict[str, Any], ...]] = {}
    for record in included:
        index = record.get("episode_index")
        if type(index) is not int or index in by_index:
            raise DataGateError("included manifest has invalid/duplicate converted episode index")
        stable_id = str(record["stable_id"])
        if record.get("e_visibility") != visibility_by_id.get(stable_id):
            raise DataGateError(f"E-visibility sidecar mismatch for {stable_id}")
        if record.get("d_valid") != d_valid_by_id.get(stable_id):
            raise DataGateError(f"D-valid sidecar mismatch for {stable_id}")
        e_visibility = record["e_visibility"]
        d_record = record["d_valid"]
        max_step = d_record.get("max_14d_step")
        max_excursion = d_record.get("max_14d_excursion")
        if (
            e_visibility.get("manual_reviewed") is not True
            or e_visibility.get("both_objects_visible") is not True
            or d_record.get("detector") != data_loader._V35_D_VALID_DETECTOR  # noqa: SLF001
            or d_record.get("state_dim") != 14
            or type(d_record.get("eligible_at_stride_15")) is not bool
            or isinstance(max_step, bool)
            or not isinstance(max_step, int | float)
            or not math.isfinite(float(max_step))
            or isinstance(max_excursion, bool)
            or not isinstance(max_excursion, int | float)
            or not math.isfinite(float(max_excursion))
            or float(max_step) >= float(data_config.memory_waiting_max_speed)
            or float(max_excursion) > float(data_config.memory_waiting_max_excursion)
        ):
            raise DataGateError(f"manual E/D leak-control provenance fails for {stable_id}")
        if d_record.get("source_left_joint_sha256") != record.get("raw_stream_sha256", {}).get(
            "left_joint_positions.npy"
        ) or d_record.get("source_right_joint_sha256") != record.get("raw_stream_sha256", {}).get(
            "right_joint_positions.npy"
        ):
            raise DataGateError(f"D-valid arm-stationarity hashes do not bind the raw joints for {stable_id}")
        episode_segments = _segments_from_label(manifest_path=manifest_path, raw_manifest=raw, record=record)
        evidence_segment = episode_segments[1]
        waiting_segment = episode_segments[3]
        d_start, d_end = d_record.get("start"), d_record.get("end")
        if (
            e_visibility.get("first_valid_visible_frame") != evidence_segment["start"]
            or type(e_visibility.get("last_clean_visible_frame")) is not int
            or e_visibility["last_clean_visible_frame"] < int(evidence_segment["end"]) - 5
            or type(d_start) is not int
            or type(d_end) is not int
            or not int(waiting_segment["start"]) <= d_start <= d_end <= int(waiting_segment["end"])
        ):
            raise DataGateError(f"structural E anchor or D-valid interval is inconsistent for {stable_id}")
        by_index[index] = record
        segments[index] = episode_segments
    if set(by_index) != set(range(INCLUDED_EPISODES)):
        raise DataGateError("stable-ID mapping is not contiguous over converted episode indices 0..89")
    return manifest, raw, by_index, segments


def _storage_record_map(
    provenance: Mapping[str, Any], *, dataset_root: Path, train_indices: set[int]
) -> dict[str, dict[str, Any]]:
    storage = provenance.get("train_storage")
    if not isinstance(storage, dict):
        raise DataGateError("norm provenance has no train_storage object")
    if storage.get("root_relative") != _project_relative(dataset_root, name="dataset root"):
        raise DataGateError("norm provenance train_storage.root_relative identifies another dataset")
    records = storage.get("files")
    if not isinstance(records, list):
        raise DataGateError("norm provenance train_storage.files must be a list")
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size"}:
            raise DataGateError("train-storage records require exactly path/sha256/size")
        relative = record.get("path")
        size = record.get("size")
        if (
            not isinstance(relative, str)
            or not relative
            or relative in result
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or type(size) is not int
            or size < 0
        ):
            raise DataGateError("train-storage record has an invalid/duplicate relative path or size")
        artifacts.require_sha256(f"train storage {relative}", record.get("sha256"))
        match = _EPISODE_PAYLOAD_RE.search(relative)
        if match is not None and int(match.group(1)) not in train_indices:
            raise DataGateError(f"train-storage seal names a held-out payload: {relative}")
        path = dataset_root / relative
        try:
            stat = path.stat()
        except OSError as exc:
            raise DataGateError(f"missing train-storage file {relative}: {exc}") from exc
        if not path.is_file() or stat.st_size != size or _sha256_file(path) != record["sha256"]:
            raise DataGateError(f"train-storage file bytes changed: {relative}")
        result[relative] = record
    return result


def _validate_dataset_metadata(
    *,
    dataset_root: Path,
    storage: Mapping[str, Mapping[str, Any]],
    records: Mapping[int, Mapping[str, Any]],
    segments: Mapping[int, Sequence[Mapping[str, Any]]],
    expected_vocabulary: Sequence[str],
) -> dict[int, str]:
    required = {
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episode_sources.json",
        "meta/episode_prompts.json",
        "meta/conversion_report.json",
    }
    if not required <= set(storage):
        raise DataGateError(f"train-storage seal is missing structural metadata: {sorted(required - set(storage))}")
    info, _ = _read_json(dataset_root / "meta/info.json", name="dataset info")
    task_rows = _read_jsonl(dataset_root / "meta/tasks.jsonl", name="dataset task vocabulary")
    episode_rows = _read_jsonl(dataset_root / "meta/episodes.jsonl", name="dataset episode table")
    sources, _ = _read_json(dataset_root / "meta/episode_sources.json", name="episode source mapping")
    prompts, _ = _read_json(dataset_root / "meta/episode_prompts.json", name="episode prompt mapping")
    conversion, _ = _read_json(dataset_root / "meta/conversion_report.json", name="conversion report")
    tasks = {row.get("task_index"): row.get("task") for row in task_rows}
    expected_frames = sum(int(record["expected_num_frames"]) for record in records.values())
    if (
        sorted(tasks) != list(range(len(expected_vocabulary)))
        or [tasks[index] for index in sorted(tasks)] != list(expected_vocabulary)
        or info.get("total_episodes") != INCLUDED_EPISODES
        or info.get("total_frames") != expected_frames
        or info.get("total_tasks") != len(expected_vocabulary)
        or len(episode_rows) != INCLUDED_EPISODES
        or set(sources) != {str(index) for index in range(INCLUDED_EPISODES)}
        or set(prompts) != set(sources)
        or conversion.get("repo_name") != project_paths.V35_REPO_ID
        or conversion.get("review_status") != "user_approved"
        or conversion.get("included_episodes") != INCLUDED_EPISODES
        or conversion.get("included_frames") != expected_frames
        or conversion.get("task_vocabulary") != list(expected_vocabulary)
    ):
        raise DataGateError("converted dataset population/task vocabulary is not the frozen 70/7 contract")
    by_episode_row = {row.get("episode_index"): row for row in episode_rows}
    if set(by_episode_row) != set(records):
        raise DataGateError("episodes.jsonl does not map every converted episode exactly once")
    for index, record in records.items():
        stable_id = str(record["stable_id"])
        source = sources[str(index)]
        row = by_episode_row[index]
        expected_tasks = {str(segment["task"]) for segment in segments[index]}
        if (
            source.get("stable_id") != stable_id
            or source.get("manifest_raw_dir") != record.get("raw_dir")
            or source.get("label_sha256") != record.get("label_sha256")
            or source.get("num_frames") != record.get("expected_num_frames")
            or source.get("target_side") != record.get("target_side")
            or prompts[str(index)] != record.get("prompt")
            or row.get("length") != record.get("expected_num_frames")
            or set(row.get("tasks", [])) != expected_tasks
        ):
            raise DataGateError(f"converted stable-ID/source/prompt mapping mismatch for {stable_id}")
    return {int(index): str(task) for index, task in tasks.items()}


def _summary(values: Sequence[int | float]) -> dict[str, int | float]:
    if not values:
        return {"count": 0, "max": 0, "min": 0, "p10": 0, "p50": 0, "p90": 0}
    array = np.asarray(values)
    quantiles = np.quantile(array, [0.1, 0.5, 0.9], method="linear")

    def scalar(value: np.generic | int | float) -> int | float:
        number = float(value)
        return int(number) if number.is_integer() else number

    return {
        "count": int(array.size),
        "max": scalar(np.max(array)),
        "min": scalar(np.min(array)),
        "p10": scalar(quantiles[0]),
        "p50": scalar(quantiles[1]),
        "p90": scalar(quantiles[2]),
    }


def _histogram(values: Sequence[int]) -> dict[str, int]:
    counts: dict[int, int] = defaultdict(int)
    for value in values:
        counts[int(value)] += 1
    return {str(value): counts[value] for value in sorted(counts)}


def _require_candidate_coverage(candidates: Sequence[Candidate], *, stable_id: str) -> None:
    if not candidates:
        raise DataGateError(f"{stable_id} has zero final-E-to-D critical candidates")
    if not any(candidate.family == "skip_o" for candidate in candidates):
        raise DataGateError(f"{stable_id} has zero skip-O D candidates")
    if any(candidate.sampled_e_count < 1 for candidate in candidates):
        raise DataGateError(f"{stable_id} retained a zero-E critical candidate")


def _task_names(columns: collector.EpisodeColumns, tasks: Mapping[int, str], *, stable_id: str) -> np.ndarray:
    try:
        return np.asarray([tasks[int(index)] for index in columns.task_index], dtype=object)
    except KeyError as exc:
        raise DataGateError(f"{stable_id} parquet references unknown task index {exc.args[0]}") from exc


def _phase_bounds(task_names: np.ndarray, task: str, *, stable_id: str) -> tuple[int, int]:
    hits = np.nonzero(task_names == task)[0]
    if not len(hits) or not np.array_equal(hits, np.arange(hits[0], hits[-1] + 1)):
        raise DataGateError(f"{stable_id} has missing/noncontiguous phase {task!r}")
    return int(hits[0]), int(hits[-1])


def _candidate_from_layout(
    *,
    layout: transforms.MemoryCriticalLayout,
    start: int,
    stride: int,
    e_start: int,
    final_e_limit: int,
    d_lo: int,
    d_hi: int,
    execute_start: int,
    action_horizon: int,
) -> Candidate:
    frames = start + layout.keep_indices.astype(np.int64) * stride
    commit = tuple(int(frame) for frame in frames if e_start <= frame <= final_e_limit)
    d_frames = tuple(int(frame) for frame in frames if d_lo <= frame <= d_hi)
    state = False
    invalid = 0
    reachable = 0
    for frame in frames:
        if d_lo <= frame <= d_hi:
            invalid += int(not state)
            reachable += int(state)
        if e_start <= frame <= final_e_limit:
            state = True
    use_pressure = tuple(
        frame for frame in d_frames if frame <= execute_start and frame + action_horizon - 1 >= execute_start
    )
    return Candidate(
        family="skip_o" if layout.sparse_skip_o else "natural",
        start_frame=start,
        sampled_e_count=layout.sampled_e_count,
        n_delay=layout.n_delay,
        commit_frames=commit,
        d_frames=d_frames,
        use_pressure_frames=use_pressure,
        state_invalid_d_steps=invalid,
        credit_reachable_d_steps=reachable,
    )


def _critical_candidates(
    *,
    record: Mapping[str, Any],
    task_names: np.ndarray,
    data_config: train_config.DataConfig,
    max_steps: int,
    action_horizon: int,
    collection_id: int,
    object_id: int,
    cell_id: int,
) -> tuple[list[Candidate], dict[str, int]]:
    stable_id = str(record["stable_id"])
    side = str(record["target_side"])
    e_start, e_end = _phase_bounds(task_names, "inspect both bins", stable_id=stable_id)
    o_lo, o_hi = _phase_bounds(task_names, "close both lids and reset arms", stable_id=stable_id)
    semantic_d_lo, semantic_d_hi = _phase_bounds(task_names, f"wait; target bin is {side}", stable_id=stable_id)
    execute_start, _ = _phase_bounds(task_names, f"open {side} bin", stable_id=stable_id)
    visibility = record["e_visibility"]
    d_valid = record["d_valid"]
    final_e_limit = e_end - data_config.memory_e_tail_guard_frames
    if (
        visibility.get("first_valid_visible_frame") != e_start
        or int(visibility.get("last_clean_visible_frame", -1)) < final_e_limit
    ):
        raise DataGateError(f"{stable_id} final eligible E anchor is not covered by both-object visibility")
    d_lo, d_hi = int(d_valid["start"]), int(d_valid["end"])
    if not semantic_d_lo <= d_lo <= d_hi <= semantic_d_hi or np.any(
        task_names[d_lo : d_hi + 1] != f"wait; target bin is {side}"
    ):
        raise DataGateError(f"{stable_id} D_valid is not a strict subset of its side-bearing wait phase")
    if not (e_start <= final_e_limit < o_lo <= o_hi < d_lo <= d_hi < execute_start):
        raise DataGateError(f"{stable_id} E/O/D/execute phases violate the frozen temporal order")

    stride = data_config.memory_stride_frames
    start_lo = max(1, e_start - data_config.memory_critical_start_pad)
    window = np.asarray(
        [
            start_lo,
            e_start,
            d_lo,
            d_hi,
            final_e_limit,
            o_lo,
            o_hi,
            execute_start,
            int(record["episode_index"]),
            collection_id,
            object_id,
            cell_id,
            transforms.V35_WINDOW_MARKER_VALUE,
        ],
        dtype=np.int32,
    )
    candidates: list[Candidate] = []
    for start in range(start_lo, e_start + 1):
        if d_lo - start > (max_steps - 1) * stride:
            continue
        try:
            layout = transforms.memory_critical_layout(
                start,
                window,
                stride=stride,
                lookahead=data_config.subtask_lookahead,
                num_steps=max_steps,
            )
        except ValueError:
            continue
        candidates.append(
            _candidate_from_layout(
                layout=layout,
                start=start,
                stride=stride,
                e_start=e_start,
                final_e_limit=final_e_limit,
                d_lo=d_lo,
                d_hi=d_hi,
                execute_start=execute_start,
                action_horizon=action_horizon,
            )
        )
    # This is the same per-family preference implemented in _sequence_sampling_info: when a
    # family offers any two-E start, one-E starts in that family receive zero mass.
    retained: list[Candidate] = []
    for family in ("natural", "skip_o"):
        family_candidates = [candidate for candidate in candidates if candidate.family == family]
        if any(candidate.sampled_e_count >= 2 for candidate in family_candidates):
            family_candidates = [candidate for candidate in family_candidates if candidate.sampled_e_count >= 2]
        retained.extend(family_candidates)
    grid_e = np.arange(
        ((e_start + stride - 1) // stride) * stride,
        final_e_limit + 1,
        stride,
        dtype=np.int64,
    )
    grid_d = np.arange(((d_lo + stride - 1) // stride) * stride, d_hi + 1, stride, dtype=np.int64)
    if not len(grid_e) or not len(grid_d):
        raise DataGateError(f"{stable_id} has zero stride-aligned E or strict-D steps")
    return retained, {
        "d_valid_raw_frames": d_hi - d_lo + 1,
        "e_to_d_n_delay_grid0": int((grid_d[0] - grid_e[-1]) // stride - 1),
        "eligible_raw_e_frames": final_e_limit - e_start + 1,
        "eligible_sampled_e_steps_grid0": len(grid_e),
        "evidence_end": e_end,
        "evidence_start": e_start,
        "execute_start": execute_start,
        "final_eligible_e_frame_grid0": int(grid_e[-1]),
        "final_eligible_e_limit": final_e_limit,
        "occlusion_end": o_hi,
        "occlusion_start": o_lo,
        "strict_d_end": d_hi,
        "strict_d_start": d_lo,
    }


def _ordinary_window_accounting(
    *,
    length: int,
    phase: Mapping[str, int],
    data_config: train_config.DataConfig,
    max_steps: int,
    block_steps: int,
) -> dict[str, int | float]:
    stride = data_config.memory_stride_frames
    min_frames = data_config.memory_min_slice_steps * stride
    start_lo = max(1, phase["evidence_start"] - data_config.memory_critical_start_pad)
    d_steps = reachable_steps = invalid_steps = accepted_d_windows = excluded_unanchored = 0
    for start in range(length):
        frames = start + np.arange(max_steps, dtype=np.int64) * stride
        frames = frames[frames < length]
        has_d = np.any((frames >= phase["strict_d_start"]) & (frames <= phase["strict_d_end"]))
        has_e = np.any((frames >= phase["evidence_start"]) & (frames <= phase["final_eligible_e_limit"]))
        unanchored = bool(has_d and not has_e)
        if unanchored:
            excluded_unanchored += 1
        in_window = start_lo <= start <= phase["evidence_start"]
        dead = phase["evidence_start"] < start <= phase["strict_d_end"]
        full_ok = start == 0 and not unanchored
        slice_ok = start > 0 and start + min_frames <= length and not dead and not in_window and not unanchored
        if not has_d or not (full_ok or slice_ok):
            continue
        accepted_d_windows += 1
        d_positions = np.nonzero((frames >= phase["strict_d_start"]) & (frames <= phase["strict_d_end"]))[0]
        e_positions = np.nonzero((frames >= phase["evidence_start"]) & (frames <= phase["final_eligible_e_limit"]))[0]
        if not len(e_positions) or int(e_positions[-1]) >= int(d_positions[0]):
            invalid_steps += len(d_positions)
            continue
        d_steps += len(d_positions)
        if block_steps <= 0:
            reachable_steps += len(d_positions)
            continue
        for shift in range(block_steps):
            reachable = False
            for step in range(len(frames)):
                if step > 0 and (step - shift) % block_steps == 0:
                    reachable = False
                if step in e_positions:
                    reachable = True
                if step in d_positions:
                    reachable_steps += int(reachable)
        d_steps += len(d_positions) * (block_steps - 1)
    return {
        "accepted_d_windows": accepted_d_windows,
        "credit_reachable_d_fraction_over_uniform_boundary_shift": (
            float(reachable_steps / d_steps) if d_steps else 1.0
        ),
        "d_steps_over_uniform_boundary_shift": d_steps,
        "excluded_unanchored_d_starts": excluded_unanchored,
        "state_invalid_d_steps_after_sampler_filter": invalid_steps,
    }


def _audit_train_episodes(
    *,
    records: Mapping[int, Mapping[str, Any]],
    manifest: artifacts.FrozenManifest,
    protocol: replay.FrozenDataProtocol,
    storage: Mapping[str, Mapping[str, Any]],
    tasks: Mapping[int, str],
    data_config: train_config.DataConfig,
    max_steps: int,
    action_horizon: int,
    block_steps: int,
    read_episode_columns: Callable[[Path, int], collector.EpisodeColumns],
    dataset_root: Path,
) -> list[dict[str, Any]]:
    train = tuple(episode for episode in manifest.episodes if episode.split == "train")
    if tuple(episode.episode_index for episode in train) != protocol.train_episode_indices:
        raise DataGateError("norm provenance train ordering differs from frozen manifest ordering")
    collection_vocab = {
        name: index for index, name in enumerate(sorted({item.collection for item in manifest.episodes}))
    }
    object_vocab = {name: index for index, name in enumerate(sorted({item.object_name for item in manifest.episodes}))}
    cells = sorted({(item.collection, item.object_name, item.target_side) for item in manifest.episodes})
    cell_vocab = {cell: index for index, cell in enumerate(cells)}
    parquet_records: dict[int, Mapping[str, Any]] = {}
    for relative, record in storage.items():
        match = _DATA_PARQUET_RE.fullmatch(relative)
        if match is not None:
            parquet_records[int(match.group(1))] = record

    output: list[dict[str, Any]] = []
    for episode in train:
        record = records[episode.episode_index]
        parquet = parquet_records.get(episode.episode_index)
        if parquet is None:
            raise DataGateError(f"train-storage seal has no parquet for {episode.stable_id}")
        parquet_path = dataset_root / str(parquet["path"])
        # The physical file was authenticated above.  The production reader requests exactly
        # episode_index/frame_index/index/task_index; no image, state, action, or media column.
        try:
            columns = collector._validate_columns(  # noqa: SLF001
                read_episode_columns(parquet_path, episode.episode_index),
                episode_index=episode.episode_index,
                frame_count=int(record["expected_num_frames"]),
            )
        except (collector.CollectionError, OSError, ValueError) as exc:
            raise DataGateError(f"cannot validate train scalar columns for {episode.stable_id}: {exc}") from exc
        task_names = _task_names(columns, tasks, stable_id=episode.stable_id)
        phase_names = [task.format(side="right" if episode.target_side else "left") for task in _EXPECTED_PHASES]
        for name in phase_names:
            _phase_bounds(task_names, name, stable_id=episode.stable_id)
        # Compare every scalar task row to the separately hash-authenticated source labels.
        rebuilt: list[str] = []
        for segment in _segments_from_label(
            manifest_path=manifest.path,
            raw_manifest=_read_json(manifest.path, name="frozen manifest")[0],
            record=dict(record),
        ):
            rebuilt.extend([str(segment["task"])] * (int(segment["end"]) - int(segment["start"]) + 1))
        if rebuilt != task_names.tolist():
            raise DataGateError(f"converted train task rows differ from frozen label bytes for {episode.stable_id}")

        candidates, phase = _critical_candidates(
            record=record,
            task_names=task_names,
            data_config=data_config,
            max_steps=max_steps,
            action_horizon=action_horizon,
            collection_id=collection_vocab[episode.collection],
            object_id=object_vocab[episode.object_name],
            cell_id=cell_vocab[(episode.collection, episode.object_name, episode.target_side)],
        )
        natural = [candidate for candidate in candidates if candidate.family == "natural"]
        skip = [candidate for candidate in candidates if candidate.family == "skip_o"]
        _require_candidate_coverage(candidates, stable_id=episode.stable_id)
        state_invalid = sum(candidate.state_invalid_d_steps for candidate in candidates)
        if state_invalid:
            raise DataGateError(f"{episode.stable_id} retained {state_invalid} state-invalid critical D steps")
        critical_d = sum(len(candidate.d_frames) for candidate in candidates)
        critical_reachable = sum(candidate.credit_reachable_d_steps for candidate in candidates)
        if critical_reachable != critical_d:
            raise DataGateError(f"{episode.stable_id} critical D supervision is not credit reachable")
        ordinary = _ordinary_window_accounting(
            length=int(record["expected_num_frames"]),
            phase=phase,
            data_config=data_config,
            max_steps=max_steps,
            block_steps=block_steps,
        )
        if ordinary["state_invalid_d_steps_after_sampler_filter"] != 0:
            raise DataGateError(f"{episode.stable_id} ordinary sampler retained a state-invalid D step")
        commit_occurrences = sum(len(candidate.commit_frames) for candidate in candidates)
        if commit_occurrences <= 0:
            raise DataGateError(f"{episode.stable_id} has no accounted successful direct-commit opportunity")
        unique_commits = sorted({frame for candidate in candidates for frame in candidate.commit_frames})
        use_frames = sorted({frame for candidate in candidates for frame in candidate.use_pressure_frames})
        output.append(
            {
                "cell": {
                    "collection": episode.collection,
                    "object": episode.object_name,
                    "part": episode.part,
                    "target_side": "right" if episode.target_side else "left",
                },
                "critical_candidates": {
                    "natural_count": len(natural),
                    "natural_n_delay": _summary([candidate.n_delay for candidate in natural]),
                    "natural_n_delay_histogram": _histogram([candidate.n_delay for candidate in natural]),
                    "skip_o_count": len(skip),
                    "skip_o_n_delay": _summary([candidate.n_delay for candidate in skip]),
                    "skip_o_n_delay_histogram": _histogram([candidate.n_delay for candidate in skip]),
                    "total_count": len(candidates),
                },
                "eligible_e": {
                    "minimum_sampled_steps_across_retained_candidates": min(
                        candidate.sampled_e_count for candidate in candidates
                    ),
                    "raw_frames": phase["eligible_raw_e_frames"],
                    "sampled_steps_grid_origin_0": phase["eligible_sampled_e_steps_grid0"],
                    "single_sampled_step_grid_origin_0": phase["eligible_sampled_e_steps_grid0"] == 1,
                },
                "ordinary_sampler": ordinary,
                "phase_bounds": phase,
                "sidecar_origin0_stride_eligibility": bool(record["d_valid"].get("eligible_at_stride_15")),
                "read_accounting": {
                    "critical_credit_reachable_d_fraction": float(critical_reachable / critical_d),
                    "critical_d_step_occurrences": critical_d,
                    "state_invalid_d_loss_steps": state_invalid
                    + int(ordinary["state_invalid_d_steps_after_sampler_filter"]),
                },
                "stable_id": episode.stable_id,
                "successful_commit_accounting": {
                    "direct_commit_occurrences": commit_occurrences,
                    "unique_eligible_commit_frames": unique_commits,
                },
                "use_pressure": {
                    "candidate_step_occurrences": sum(len(candidate.use_pressure_frames) for candidate in candidates),
                    "unique_raw_frames": use_frames,
                    "unique_step_count": len(use_frames),
                },
            }
        )
    return output


def _aggregate_episodes(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_collection: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_cell: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for episode in episodes:
        cell = episode["cell"]
        by_collection[str(cell["collection"])].append(episode)
        by_cell[(str(cell["collection"]), str(cell["object"]), str(cell["target_side"]))].append(episode)

    def group_summary(members: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "eligible_raw_e_frames": _summary([item["eligible_e"]["raw_frames"] for item in members]),
            "eligible_sampled_e_steps_grid_origin_0": _summary(
                [item["eligible_e"]["sampled_steps_grid_origin_0"] for item in members]
            ),
            "episode_count": len(members),
            "natural_candidate_count": sum(item["critical_candidates"]["natural_count"] for item in members),
            "skip_o_candidate_count": sum(item["critical_candidates"]["skip_o_count"] for item in members),
            "successful_direct_commit_occurrences": sum(
                item["successful_commit_accounting"]["direct_commit_occurrences"] for item in members
            ),
            "successful_calibration_commit_read_episodes": sum(
                item["successful_commit_accounting"].get("calibration_nonzero_finite_commit_read") is True
                for item in members
            ),
            "use_pressure_candidate_step_occurrences": sum(
                item["use_pressure"]["candidate_step_occurrences"] for item in members
            ),
            "use_pressure_unique_step_count": sum(item["use_pressure"]["unique_step_count"] for item in members),
        }

    natural_delays = [
        int(delay)
        for item in episodes
        for delay, count in item["critical_candidates"]["natural_n_delay_histogram"].items()
        for _ in range(int(count))
    ]
    skip_delays = [
        int(delay)
        for item in episodes
        for delay, count in item["critical_candidates"]["skip_o_n_delay_histogram"].items()
        for _ in range(int(count))
    ]
    grid_delays = [int(item["phase_bounds"]["e_to_d_n_delay_grid0"]) for item in episodes]
    return {
        "by_cell": {
            f"{collection}/{object_name}/{side}": group_summary(members)
            for (collection, object_name, side), members in sorted(by_cell.items())
        },
        "by_collection": {name: group_summary(members) for name, members in sorted(by_collection.items())},
        "e_to_d_n_delay_grid_origin_0": _summary(grid_delays),
        "natural_candidate_n_delay": {
            "histogram": _histogram(natural_delays),
            "summary": _summary(natural_delays),
        },
        "single_sampled_e_step_stable_ids": [
            item["stable_id"] for item in episodes if item["eligible_e"]["single_sampled_step_grid_origin_0"]
        ],
        "skip_o_candidate_n_delay": {
            "histogram": _histogram(skip_delays),
            "summary": _summary(skip_delays),
        },
    }


def _registered_data_config(config_name: str) -> tuple[train_config.TrainConfig, train_config.DataConfig]:
    config = train_config.get_config(config_name)
    factory = config.data
    if factory.base_config is None:
        raise DataGateError("registered v3.5 data factory has no base configuration")
    base = dataclasses.replace(
        factory.base_config,
        repo_id=factory.repo_id,
        asset_id=factory.assets.asset_id or factory.repo_id,
    )
    expected_dataset = project_paths.project_path(project_paths.V35_DATASET_DIR)
    expected_manifest = project_paths.project_path(project_paths.V35_FROZEN_MANIFEST)
    if (
        config.name != CONFIG_NAME
        or base.repo_id != project_paths.V35_REPO_ID
        or not base.memory_v35_enabled
        or not base.memory_v35_frozen_population
        or base.memory_manifest_split != "train"
        or base.memory_manifest_split_seed != 36
        or Path(str(base.lerobot_dataset_root)).resolve() != expected_dataset
        or Path(str(base.memory_episode_manifest_path)).resolve() != expected_manifest
        or base.memory_episode_manifest_sha256 != "9085fe50d7b02ea65930f3647ce0413e0583a66d430484e06c60812c52af8442"
        or config.model.memory_seq_steps != 40
        or config.model.memory_block_steps != 25
        or config.model.action_horizon != 50
    ):
        raise DataGateError("registered pi05_yam_mem_v35 data/sampler contract changed")
    return config, base


def build_gate_artifacts(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    dataset_root: Path,
    norm_provenance_path: Path,
    norm_stats_path: Path,
    calibration_descriptor_path: Path,
    detail_report_path: Path,
    data_config: train_config.DataConfig,
    action_horizon: int,
    max_steps: int,
    block_steps: int,
    read_episode_columns: Callable[[Path, int], collector.EpisodeColumns] = collector._read_parquet_columns,  # noqa: SLF001
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build, but do not write, the report and authorization-consumed decision."""
    manifest_path = Path(manifest_path).resolve()
    dataset_root = Path(dataset_root).resolve()
    norm_provenance_path = Path(norm_provenance_path).resolve()
    norm_stats_path = Path(norm_stats_path).resolve()
    calibration_descriptor_path = Path(calibration_descriptor_path).resolve()
    detail_report_path = Path(detail_report_path).resolve()
    for name, path in (
        ("manifest", manifest_path),
        ("dataset root", dataset_root),
        ("norm provenance", norm_provenance_path),
        ("norm stats", norm_stats_path),
        ("calibration descriptor", calibration_descriptor_path),
        ("detail report", detail_report_path),
    ):
        _project_relative(path, name=name)

    manifest, _, records, segments = _load_and_validate_manifest_structure(
        manifest_path, manifest_sha256=manifest_sha256, data_config=data_config
    )
    try:
        protocol = replay.validate_frozen_data_protocol(
            manifest_path=manifest_path,
            manifest_sha256=manifest.sha256,
            norm_provenance_path=norm_provenance_path,
            norm_stats_path=norm_stats_path,
            expected_repo_id=project_paths.V35_REPO_ID,
            expected_split_seed=36,
            production=True,
        )
    except replay.ReplayPreflightError as exc:
        raise DataGateError(f"invalid train-only normalization/data protocol: {exc}") from exc
    if protocol.train_storage_root.resolve() != dataset_root:
        raise DataGateError("norm provenance train storage root differs from the registered project dataset")
    provenance, _ = _read_json(norm_provenance_path, name="norm provenance")
    storage = _storage_record_map(
        provenance, dataset_root=dataset_root, train_indices=set(protocol.train_episode_indices)
    )
    tasks = _validate_dataset_metadata(
        dataset_root=dataset_root,
        storage=storage,
        records=records,
        segments=segments,
        expected_vocabulary=data_config.memory_subtask_vocab,
    )
    calibration = _load_calibration_dataset_identity(calibration_descriptor_path, protocol=protocol, manifest=manifest)
    if calibration.dataset_protocol_sha256 != protocol.dataset_protocol_sha256:
        raise DataGateError("calibration descriptor dataset protocol differs from current frozen data")
    episodes = _audit_train_episodes(
        records=records,
        manifest=manifest,
        protocol=protocol,
        storage=storage,
        tasks=tasks,
        data_config=data_config,
        max_steps=max_steps,
        action_horizon=action_horizon,
        block_steps=block_steps,
        read_episode_columns=read_episode_columns,
        dataset_root=dataset_root,
    )
    calibration_by_id = {str(item["stable_id"]): item for item in calibration.successful_commit_metrics}
    enriched_episodes: list[dict[str, Any]] = []
    for episode in episodes:
        stable_id = str(episode["stable_id"])
        metric = calibration_by_id.get(stable_id)
        if metric is None or metric["n_delay"] != episode["phase_bounds"]["e_to_d_n_delay_grid0"]:
            raise DataGateError(f"calibration commit/read clock differs from current Gate-A data for {stable_id}")
        enriched_episodes.append(
            {
                **episode,
                "successful_commit_accounting": {
                    **episode["successful_commit_accounting"],
                    "calibration_clean_episode_raw_rms": metric["clean_episode_raw_rms"],
                    "calibration_clean_injected_to_residual_rms": metric["clean_injected_to_residual_rms"],
                    "calibration_clean_slot_raw_rms_p50": metric["clean_slot_raw_rms_p50"],
                    "calibration_nonzero_finite_commit_read": True,
                },
            }
        )
    episodes = enriched_episodes
    minimum_e = min(item["eligible_e"]["minimum_sampled_steps_across_retained_candidates"] for item in episodes)
    state_invalid = sum(item["read_accounting"]["state_invalid_d_loss_steps"] for item in episodes)
    final_anchors = sum(bool(item["critical_candidates"]["total_count"]) for item in episodes)
    skip_episodes = sum(bool(item["critical_candidates"]["skip_o_count"]) for item in episodes)
    commit_episodes = sum(
        item["successful_commit_accounting"]["calibration_nonzero_finite_commit_read"] is True for item in episodes
    )
    if (
        len(episodes) != TRAIN_EPISODES
        or minimum_e < 1
        or state_invalid != 0
        or final_anchors != TRAIN_EPISODES
        or skip_episodes != TRAIN_EPISODES
        or commit_episodes != TRAIN_EPISODES
    ):
        raise DataGateError("training episode E/skip-O/state/commit accounting is incomplete")

    dataset_identity = {
        "calibration_dataset_protocol_sha256": calibration.dataset_protocol_sha256,
        "episode_manifest_sha256": manifest.sha256,
        "norm_computation_protocol": replay.NORMALIZATION_PROTOCOL,
        "norm_stats_provenance_sha256": protocol.norm_provenance_sha256,
        "norm_stats_sha256": protocol.norm_stats_sha256,
        "split_assignment_sha256": manifest.split_assignment_sha256,
        "train_storage_sha256": protocol.train_storage_sha256,
    }
    split_counts = {name: len(manifest.split(name)) for name in ("train", "development", "final_test")}
    report_payload = {
        "access_audit": {
            "development_payload_files_opened": 0,
            "final_test_payload_files_opened": 0,
            "heldout_access": "frozen manifest, hashed structural sidecars/labels, and global structural meta only",
            "train_parquet_columns_decoded": ["episode_index", "frame_index", "index", "task_index"],
            "train_parquet_files_opened": TRAIN_EPISODES,
            "train_storage_files_hash_verified": len(storage),
        },
        "aggregates": _aggregate_episodes(episodes),
        "calibration_descriptor": {
            "artifact_id": calibration.artifact_id,
            "kind": calibration.kind,
            "path_relative": _project_relative(calibration_descriptor_path, name="calibration descriptor"),
            "sha256": calibration.file_sha256,
        },
        "criteria_version": CRITERIA_VERSION,
        "dataset_identity": dataset_identity,
        "episode_manifest_id": f"sha256:{manifest.sha256}",
        "normalization": {
            "active_split": "train",
            "computation_protocol": replay.NORMALIZATION_PROTOCOL,
            "norm_provenance_path_relative": _project_relative(norm_provenance_path, name="norm provenance"),
            "norm_stats_path_relative": _project_relative(norm_stats_path, name="norm stats"),
            "selected_episode_count": TRAIN_EPISODES,
            "train_storage_root_relative": _project_relative(dataset_root, name="dataset root"),
        },
        "population": {
            "development": split_counts["development"],
            "excluded_raw": RAW_EPISODES - INCLUDED_EPISODES,
            "final_test": split_counts["final_test"],
            "included": INCLUDED_EPISODES,
            "raw_records": RAW_EPISODES,
            "train": split_counts["train"],
        },
        "protocol": {
            "action_horizon_raw_frames": action_horizon,
            "e_tail_guard_raw_frames": data_config.memory_e_tail_guard_frames,
            "max_sequence_steps": max_steps,
            "natural_skip_o_target_mass": {"natural": 0.5, "skip_o": 0.5},
            "read_credit_reachable": "reported descriptively; state-valid is the hard gate",
            "sampled_stride_raw_frames": data_config.memory_stride_frames,
            "successful_commit_count_definition": (
                "one nonzero finite clean committed read plus aligned layer-8 residual ratio per train episode "
                "from the sealed passing injection-calibration artifact; sampler opportunities are also reported"
            ),
        },
        "status": "pass",
        "training_episodes": episodes,
    }
    report = artifacts.artifact_envelope(REPORT_SCHEMA_VERSION, report_payload)
    report_bytes = artifacts.canonical_json_bytes(report) + b"\n"
    detail_descriptor = {
        "artifact_id": report["artifact_id"],
        "path_relative": _project_relative(detail_report_path, name="detail report"),
        "sha256": artifacts.sha256_bytes(report_bytes),
    }
    checks = dict.fromkeys(sorted(REQUIRED_CHECKS), True)
    decision_payload = {
        "checks": checks,
        "criteria_version": CRITERIA_VERSION,
        "dataset_identity": dataset_identity,
        "detail_report": detail_descriptor,
        "episode_manifest_sha256": manifest.sha256,
        "final_test_accessed": False,
        "population": {
            "development": DEVELOPMENT_EPISODES,
            "final_test": FINAL_TEST_EPISODES,
            "total": INCLUDED_EPISODES,
            "train": TRAIN_EPISODES,
        },
        "split_assignment_sha256": manifest.split_assignment_sha256,
        "status": "pass",
        "training_counts": {
            "episodes_with_final_eligible_e_anchor": final_anchors,
            "episodes_with_skip_o_d_candidate": skip_episodes,
            "episodes_with_successful_commit_accounting": commit_episodes,
            "minimum_eligible_sampled_e_steps": minimum_e,
            "state_invalid_d_loss_steps": state_invalid,
            "training_episodes": len(episodes),
        },
    }
    decision = artifacts.artifact_envelope(DECISION_SCHEMA_VERSION, decision_payload)
    return decision, report


def write_gate_artifacts(
    *, decision_path: Path, detail_report_path: Path, decision: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    decision_path = Path(decision_path)
    detail_report_path = Path(detail_report_path)
    if decision_path.resolve() == detail_report_path.resolve():
        raise DataGateError("decision and detail report paths must differ")
    if decision_path.exists() or detail_report_path.exists():
        raise DataGateError("refusing to overwrite an existing Gate-A decision or detail report")
    artifacts.write_canonical_envelope(detail_report_path, report, schema_version=REPORT_SCHEMA_VERSION)
    artifacts.write_canonical_envelope(decision_path, decision, schema_version=DECISION_SCHEMA_VERSION)


def _default_norm_dir() -> Path:
    return project_paths.project_path(project_paths.V35_ASSETS_DIR / project_paths.V35_REPO_ID)


def _parser() -> argparse.ArgumentParser:
    norm_dir = _default_norm_dir()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default=CONFIG_NAME)
    parser.add_argument("--manifest", type=Path, default=project_paths.V35_FROZEN_MANIFEST)
    parser.add_argument(
        "--manifest-sha256",
        default="9085fe50d7b02ea65930f3647ce0413e0583a66d430484e06c60812c52af8442",
    )
    parser.add_argument("--dataset-root", type=Path, default=project_paths.V35_DATASET_DIR)
    parser.add_argument(
        "--norm-provenance",
        type=Path,
        default=project_paths.project_relative_path(norm_dir / "norm_stats_provenance.json"),
    )
    parser.add_argument(
        "--norm-stats", type=Path, default=project_paths.project_relative_path(norm_dir / "norm_stats.json")
    )
    parser.add_argument("--calibration-descriptor", type=Path, required=True)
    parser.add_argument("--detail-report", type=Path, default=Path("v35/diagnostics/gate_a/data_gate_report.json"))
    parser.add_argument("--output", type=Path, default=Path("v35/diagnostics/gate_a/data_gate_decision.json"))
    return parser


def _resolve_cli_path(path: Path, *, name: str) -> Path:
    if path.is_absolute() or ".." in path.parts:
        raise DataGateError(f"{name} must be a confined memory_project-relative path")
    try:
        return project_paths.project_path(path)
    except project_paths.ProjectRootError as exc:
        raise DataGateError(f"invalid {name}: {exc}") from exc


def main() -> None:
    args = _parser().parse_args()
    project_paths.configure_v35_runtime_environment()
    config, data_config = _registered_data_config(args.config_name)
    manifest_path = _resolve_cli_path(args.manifest, name="manifest")
    dataset_root = _resolve_cli_path(args.dataset_root, name="dataset root")
    norm_provenance_path = _resolve_cli_path(args.norm_provenance, name="norm provenance")
    norm_stats_path = _resolve_cli_path(args.norm_stats, name="norm stats")
    calibration_descriptor_path = _resolve_cli_path(args.calibration_descriptor, name="calibration descriptor")
    detail_report_path = _resolve_cli_path(args.detail_report, name="detail report")
    output_path = _resolve_cli_path(args.output, name="decision output")
    if output_path.exists() or detail_report_path.exists():
        raise DataGateError("refusing to overwrite an existing Gate-A decision or detail report")
    decision, report = build_gate_artifacts(
        manifest_path=manifest_path,
        manifest_sha256=args.manifest_sha256,
        dataset_root=dataset_root,
        norm_provenance_path=norm_provenance_path,
        norm_stats_path=norm_stats_path,
        calibration_descriptor_path=calibration_descriptor_path,
        detail_report_path=detail_report_path,
        data_config=data_config,
        action_horizon=config.model.action_horizon,
        max_steps=config.model.memory_seq_steps,
        block_steps=config.model.memory_block_steps,
    )
    write_gate_artifacts(
        decision_path=output_path,
        detail_report_path=detail_report_path,
        decision=decision,
        report=report,
    )
    print(
        json.dumps(
            {
                "decision": _project_relative(output_path, name="decision output"),
                "decision_artifact_id": decision["artifact_id"],
                "detail_report": _project_relative(detail_report_path, name="detail report"),
                "detail_report_artifact_id": report["artifact_id"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
