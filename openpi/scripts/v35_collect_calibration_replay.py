"""Freeze and collect the train-54 real-sequence replay used by v3.5 calibration.

This is the dataset runner deliberately left outside :mod:`v35_calibration_replay`.  It has
three fail-closed stages:

``select``
    Authenticate the fresh-step-0 preflight and freeze one canonical frame-selection artifact.
    Only the 54 parquet files named by the preflight's train split are opened.  Development and
    final-test parquet/image payloads are never inspected.  The selected evidence frame is the
    last stride-15 E frame before the five-frame tail guard, the decision frame is the first
    stride-15 frame in the independent stationary-D interval, and every intervening sampled
    frame is preregistered as a possible real low-cosine query control.

``collect``
    Reconstruct the registered model from the official Pi0.5 base and the registered seed,
    authenticate the live post-load/post-cast parameter-tree hash against the preflight, then
    collect a disjoint ordinal shard of episodes.  The clean read always preserves all 16
    decision slots.  If none of the preregistered real slots has cosine <= 0.10, a deterministic
    hidden-space orthogonal control is measured against the same nonzero post-commit memory and
    explicitly labelled synthetic.

``seal``
    Validate all 54 episode shards, call the existing strict reducer-input sealer, and write a
    canonical receipt binding the selection artifact, every shard, and the final NPZ.

All CLI paths are relative to ``memory_project``.  The script never writes into the dataset.
Parallel collection is supported with ``--num-shards``/``--shard-index``; one process per GPU is
the intended H100 launch pattern.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import gc
import hashlib
import io
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import sys
from typing import Any

import numpy as np
from numpy import typing as npt

from openpi.shared import project_paths

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v35_calibration_replay as replay
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


SELECTION_SCHEMA = "openpi.v35.calibration-frame-selection.v1"
COLLECTION_RECEIPT_SCHEMA = "openpi.v35.calibration-collection-receipt.v1"
STRIDE_FRAMES = 15
E_TAIL_GUARD_FRAMES = 5
EVIDENCE_TASK = "inspect both bins"
OCCLUSION_TASK = "close both lids and reset arms"
_SIDE_TASKS = {
    "left": "wait; target bin is left",
    "right": "wait; target bin is right",
}
_PARQUET_RE = re.compile(r"data/chunk-\d{3}/episode_(\d{6})\.parquet")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

# The complete algorithm is hashed into the selection artifact.  In particular, real query
# candidates are chosen without model outputs, so a later low-cosine search cannot cherry-pick
# frames.  The synthetic method is likewise fixed before any episode is replayed.
SELECTION_PROTOCOL: dict[str, Any] = {
    "schema_version": 1,
    "population": "exact ordered train stable IDs from authenticated calibration preflight",
    "payload_access": "open selected train parquet scalar columns only; never open development/final parquet",
    "clock": {
        "stride_raw_frames": STRIDE_FRAMES,
        "grid_origin": 0,
        "n_delay": "(decision_frame-final_eligible_e_frame)/15 - 1",
    },
    "evidence": {
        "task": EVIDENCE_TASK,
        "tail_guard_raw_frames": E_TAIL_GUARD_FRAMES,
        "selection": "largest grid frame <= semantic_E_end-5, within E, and <= manual last-clean-visible",
    },
    "decision": "smallest grid frame inside independent d_valid inclusive interval with side-specific D task",
    "real_query_controls": (
        "every grid frame strictly after evidence and through decision inclusive; retain every slot whose "
        "hidden-key cosine to stored evidence hidden key is <=0.10"
    ),
    "synthetic_fallback": {
        "when": "no preregistered real query slot has cosine <=0.10",
        "gap": "episode decision n_delay",
        "algorithm": (
            "choose the least-absolute coordinate of normalized stored hidden h; Gram-Schmidt that canonical "
            "basis vector against h in FP32; read the actual nonzero delayed w3 directly with the result"
        ),
        "label": "synthetic_orthogonal_query",
    },
    "augmentation": "disabled",
    "arithmetic": "memory/commit/hidden alignment/raw reads/synthetic Gram-Schmidt are FP32",
}


class CollectionError(ValueError):
    """Raised when train-only selection or collection cannot be authenticated."""


@dataclasses.dataclass(frozen=True)
class EpisodeColumns:
    episode_index: npt.NDArray[np.integer[Any]]
    frame_index: npt.NDArray[np.integer[Any]]
    dataset_index: npt.NDArray[np.integer[Any]]
    task_index: npt.NDArray[np.integer[Any]]


@dataclasses.dataclass(frozen=True)
class LoadedFrame:
    episode_index: int
    frame_index: int
    dataset_index: int
    task_index: int
    image: np.ndarray
    left_wrist_image: np.ndarray
    right_wrist_image: np.ndarray
    state: npt.NDArray[np.float32]


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


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def selection_protocol_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(SELECTION_PROTOCOL))


def collector_source_sha256() -> str:
    return sha256_file(Path(__file__))


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise CollectionError(f"{name} must be a lower-case 64-character SHA256 digest")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CollectionError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        encoded = Path(path).read_bytes()
        value = json.loads(encoded, object_pairs_hook=_strict_object)
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectionError(f"cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CollectionError(f"{path} must contain one JSON object")
    return value, encoded


def _envelope(payload: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    digest = sha256_bytes(canonical_json_bytes(payload))
    return {
        "artifact_sha256": digest,
        f"{prefix}_id": f"sha256:{digest}",
        "hash_scope": "SHA256 of canonical_json($.payload)",
        "payload": payload,
    }


def _verify_envelope(value: Mapping[str, Any], *, prefix: str) -> bool:
    payload = value.get("payload")
    if not isinstance(payload, dict):
        return False
    digest = sha256_bytes(canonical_json_bytes(payload))
    return value.get("artifact_sha256") == digest and value.get(f"{prefix}_id") == f"sha256:{digest}"


def _write_json_once(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(canonical_json_bytes(value) + b"\n")
    except FileExistsError as exc:
        raise CollectionError(f"refusing to overwrite existing artifact: {path}") from exc


def _resolve_project_path(path: Path, *, name: str) -> Path:
    raw = Path(path)
    if raw.is_absolute() or ".." in raw.parts:
        raise CollectionError(f"{name} must be a confined path relative to memory_project")
    try:
        return project_paths.project_path(raw)
    except project_paths.ProjectRootError as exc:
        raise CollectionError(f"invalid {name}: {exc}") from exc


def _preflight_path(preflight: Mapping[str, Any], field: str) -> Path:
    data = preflight["payload"]["data"]
    if data.get("production") is not True:
        raise CollectionError("the end-to-end collector is restricted to a production project-relative preflight")
    relative = data.get(field)
    if not isinstance(relative, str):
        raise CollectionError(f"production preflight is missing {field}")
    try:
        return project_paths.project_path(relative)
    except project_paths.ProjectRootError as exc:
        raise CollectionError(f"invalid preflight {field}: {exc}") from exc


def _manifest_records(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    entries = manifest.get("episodes")
    if not isinstance(entries, list):
        raise CollectionError("frozen manifest has no episodes list")
    records: dict[str, dict[str, Any]] = {}
    for raw in entries:
        if not isinstance(raw, dict):
            raise CollectionError("frozen manifest episode is not an object")
        stable_id = str(raw.get("stable_id", "")).strip()
        if not stable_id or stable_id in records:
            raise CollectionError(f"frozen manifest has an empty/duplicate stable ID: {stable_id!r}")
        records[stable_id] = raw
    return records


def _storage_maps(
    provenance: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    storage = provenance.get("train_storage")
    if not isinstance(storage, dict) or storage.get("root_contract") != replay.TRAIN_STORAGE_ROOT_CONTRACT:
        raise CollectionError("norm provenance does not use the portable train-storage contract")
    records = storage.get("files")
    if not isinstance(records, list):
        raise CollectionError("norm provenance train_storage has no file records")
    parquets: dict[int, dict[str, Any]] = {}
    by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "size", "sha256"}:
            raise CollectionError("train_storage file records require exactly path/size/sha256")
        relative = record["path"]
        if not isinstance(relative, str) or relative in by_path:
            raise CollectionError("train_storage has a non-string or duplicate path")
        _require_sha256("train_storage file", record["sha256"])
        by_path[relative] = record
        match = _PARQUET_RE.fullmatch(relative)
        if match is not None:
            episode = int(match.group(1))
            if episode in parquets:
                raise CollectionError(f"train_storage has duplicate parquet for episode {episode}")
            parquets[episode] = record
    return parquets, by_path


def _validate_columns(columns: EpisodeColumns, *, episode_index: int, frame_count: int) -> EpisodeColumns:
    arrays = tuple(np.asarray(value) for value in dataclasses.astuple(columns))
    if any(array.ndim != 1 or len(array) != frame_count for array in arrays):
        raise CollectionError(f"episode {episode_index} scalar parquet columns do not have {frame_count} rows")
    if any(not np.issubdtype(array.dtype, np.integer) for array in arrays):
        raise CollectionError(f"episode {episode_index} scalar parquet columns must all be integers")
    episode, frame, dataset, task = (array.astype(np.int64, copy=False) for array in arrays)
    if not np.all(episode == episode_index):
        raise CollectionError(f"episode {episode_index} parquet carries a different episode_index")
    if not np.array_equal(frame, np.arange(frame_count, dtype=np.int64)):
        raise CollectionError(f"episode {episode_index} frame_index is not canonical 0..N-1")
    if len(np.unique(dataset)) != frame_count or np.any(np.diff(dataset) != 1):
        raise CollectionError(f"episode {episode_index} global dataset index is not contiguous/unique")
    if np.any(task < 0):
        raise CollectionError(f"episode {episode_index} has a negative task_index")
    return EpisodeColumns(episode, frame, dataset, task)


def _episode_selection(
    *,
    record: Mapping[str, Any],
    columns: EpisodeColumns,
    tasks: Mapping[int, str],
    parquet_record: Mapping[str, Any],
) -> dict[str, Any]:
    stable_id = str(record["stable_id"])
    episode_index = record.get("episode_index")
    frame_count = record.get("expected_num_frames")
    if type(episode_index) is not int or type(frame_count) is not int or frame_count <= 0:
        raise CollectionError(f"{stable_id} has invalid episode_index/expected_num_frames")
    columns = _validate_columns(columns, episode_index=episode_index, frame_count=frame_count)
    try:
        task_names = np.asarray([tasks[int(index)] for index in columns.task_index], dtype=object)
    except KeyError as exc:
        raise CollectionError(f"{stable_id} parquet references unknown task index {exc.args[0]}") from exc

    evidence = np.nonzero(task_names == EVIDENCE_TASK)[0]
    if not len(evidence) or not np.array_equal(evidence, np.arange(evidence[0], evidence[-1] + 1)):
        raise CollectionError(f"{stable_id} does not have one contiguous evidence phase")
    visibility = record.get("e_visibility")
    if not isinstance(visibility, dict) or visibility.get("manual_reviewed") is not True:
        raise CollectionError(f"{stable_id} has no approved E-visibility record")
    e_start, e_end = int(evidence[0]), int(evidence[-1])
    if visibility.get("first_valid_visible_frame") != e_start or visibility.get("both_objects_visible") is not True:
        raise CollectionError(f"{stable_id} evidence start is not the approved both-object-visible frame")
    last_clean = visibility.get("last_clean_visible_frame")
    if type(last_clean) is not int:
        raise CollectionError(f"{stable_id} has no integer last-clean-visible frame")
    eligible_limit = min(e_end - E_TAIL_GUARD_FRAMES, last_clean)
    evidence_grid = columns.frame_index[
        (columns.frame_index % STRIDE_FRAMES == 0)
        & (columns.frame_index >= e_start)
        & (columns.frame_index <= eligible_limit)
    ]
    if not len(evidence_grid):
        raise CollectionError(f"{stable_id} has no sampled E frame after the tail guard")
    evidence_frame = int(evidence_grid[-1])
    if task_names[evidence_frame] != EVIDENCE_TASK:
        raise CollectionError(f"{stable_id} selected evidence frame is not E")

    target_side = str(record.get("target_side", "")).strip().lower()
    if target_side not in _SIDE_TASKS:
        raise CollectionError(f"{stable_id} has invalid target_side {target_side!r}")
    prompt = str(record.get("prompt", "")).strip()
    if not prompt:
        raise CollectionError(f"{stable_id} has an empty high-level prompt")
    d_valid = record.get("d_valid")
    if not isinstance(d_valid, dict):
        raise CollectionError(f"{stable_id} has no independent d_valid record")
    d_start, d_end = d_valid.get("start"), d_valid.get("end")
    if type(d_start) is not int or type(d_end) is not int or not 0 <= d_start <= d_end < frame_count:
        raise CollectionError(f"{stable_id} has invalid d_valid bounds")
    decision_grid = columns.frame_index[
        (columns.frame_index % STRIDE_FRAMES == 0) & (columns.frame_index >= d_start) & (columns.frame_index <= d_end)
    ]
    expected_d_task = _SIDE_TASKS[target_side]
    decision_grid = decision_grid[task_names[decision_grid] == expected_d_task]
    if not len(decision_grid):
        raise CollectionError(f"{stable_id} has no stride-15 stationary-D frame with the expected side task")
    decision_frame = int(decision_grid[0])
    if decision_frame <= evidence_frame or (decision_frame - evidence_frame) % STRIDE_FRAMES:
        raise CollectionError(f"{stable_id} evidence-to-decision clock is not a positive stride-15 interval")
    n_delay = (decision_frame - evidence_frame) // STRIDE_FRAMES - 1
    query_frames = list(range(evidence_frame + STRIDE_FRAMES, decision_frame + 1, STRIDE_FRAMES))
    if len(query_frames) != n_delay + 1:
        raise CollectionError(f"{stable_id} query-control clock is inconsistent")

    def frame_ref(frame: int) -> dict[str, int]:
        return {"frame_index": frame, "dataset_index": int(columns.dataset_index[frame])}

    query_controls = [
        {
            **frame_ref(frame),
            "n_delay": (frame - evidence_frame) // STRIDE_FRAMES - 1,
        }
        for frame in query_frames
    ]
    return {
        "stable_id": stable_id,
        "split": "train",
        "episode_index": episode_index,
        "frame_count": frame_count,
        "target_side": target_side,
        "prompt": prompt,
        "parquet_path_relative": parquet_record["path"],
        "parquet_sha256": parquet_record["sha256"],
        "evidence": {
            "semantic_start": e_start,
            "semantic_end": e_end,
            "eligible_limit": eligible_limit,
            **frame_ref(evidence_frame),
        },
        "decision": {
            "d_valid_start": d_start,
            "d_valid_end": d_end,
            "n_delay": n_delay,
            **frame_ref(decision_frame),
        },
        "query_controls": query_controls,
    }


def build_frame_selection_artifact(
    *,
    preflight: Mapping[str, Any],
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    tasks: Mapping[int, str],
    read_episode_columns: Callable[[Path, int], EpisodeColumns],
    dataset_root: Path,
    structural_meta_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build the canonical selection, calling ``read_episode_columns`` for train episodes only."""
    if not replay._verify_envelope(preflight, id_prefix="preflight"):  # noqa: SLF001
        raise CollectionError("invalid calibration preflight envelope")
    payload = preflight["payload"]
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("train_episode_count") != replay.TRAIN_EPISODE_COUNT:
        raise CollectionError("preflight does not name exactly 54 train episodes")
    train_ids = data.get("train_stable_ids")
    train_indices = data.get("train_episode_indices")
    if (
        not isinstance(train_ids, list)
        or not isinstance(train_indices, list)
        or len(train_ids) != replay.TRAIN_EPISODE_COUNT
        or len(train_indices) != replay.TRAIN_EPISODE_COUNT
        or len(set(train_ids)) != replay.TRAIN_EPISODE_COUNT
    ):
        raise CollectionError("preflight train membership is malformed")
    records = _manifest_records(manifest)
    parquets, _ = _storage_maps(provenance)
    selections: list[dict[str, Any]] = []
    for expected_index, stable_id in zip(train_indices, train_ids, strict=True):
        record = records.get(stable_id)
        if (
            not isinstance(record, dict)
            or record.get("split") != "train"
            or record.get("include", True) is not True
            or record.get("episode_index") != expected_index
        ):
            raise CollectionError(f"preflight train episode {stable_id!r} disagrees with the frozen manifest")
        parquet_record = parquets.get(expected_index)
        if parquet_record is None:
            raise CollectionError(f"train storage has no parquet for {stable_id!r}")
        parquet_path = dataset_root / parquet_record["path"]
        columns = read_episode_columns(parquet_path, expected_index)
        selections.append(
            _episode_selection(record=record, columns=columns, tasks=tasks, parquet_record=parquet_record)
        )

    bindings = dataclasses.asdict(replay.bindings_from_preflight(preflight))
    selection_payload = {
        "schema_version": SELECTION_SCHEMA,
        "status": "complete_train54_only",
        "preflight": {
            "artifact_sha256": preflight["artifact_sha256"],
            "preflight_id": preflight["preflight_id"],
        },
        "bindings": bindings,
        "source": {
            "file": Path(__file__).name,
            "sha256": collector_source_sha256(),
        },
        "protocol": SELECTION_PROTOCOL,
        "protocol_sha256": selection_protocol_sha256(),
        "data": {
            "root_contract": replay.TRAIN_STORAGE_ROOT_CONTRACT,
            "dataset_root_relative": project_paths.project_relative_path(dataset_root).as_posix(),
            "dataset_repo_id": data["dataset_repo_id"],
            "manifest_sha256": data["manifest_sha256"],
            "dataset_protocol_sha256": data["dataset_episode_frame_protocol_sha256"],
            "train_storage_sha256": data["train_storage_sha256"],
            "structural_meta_sha256": dict(sorted(structural_meta_sha256.items())),
            "heldout_payload_access_count": 0,
        },
        "selection": {
            "episode_count": replay.TRAIN_EPISODE_COUNT,
            "episodes": selections,
        },
    }
    return _envelope(selection_payload, prefix="selection")


def _read_parquet_columns(path: Path, _: int) -> EpisodeColumns:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise CollectionError("pyarrow is required to read the converted LeRobot parquet") from exc
    try:
        table = pq.read_table(path, columns=["episode_index", "frame_index", "index", "task_index"])
    except (OSError, ValueError) as exc:
        raise CollectionError(f"cannot read scalar columns from {path}: {exc}") from exc
    return EpisodeColumns(
        episode_index=np.asarray(table["episode_index"].to_numpy(zero_copy_only=False)),
        frame_index=np.asarray(table["frame_index"].to_numpy(zero_copy_only=False)),
        dataset_index=np.asarray(table["index"].to_numpy(zero_copy_only=False)),
        task_index=np.asarray(table["task_index"].to_numpy(zero_copy_only=False)),
    )


def _load_tasks(path: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line, object_pairs_hook=_strict_object)
            index, task = value.get("task_index"), value.get("task")
            if type(index) is not int or not isinstance(task, str) or index in tasks:
                raise CollectionError("tasks.jsonl contains an invalid/duplicate task record")
            tasks[index] = task
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectionError(f"cannot parse {path}: {exc}") from exc
    if not tasks:
        raise CollectionError("tasks.jsonl is empty")
    return tasks


def _verify_storage_file(dataset_root: Path, record: Mapping[str, Any]) -> str:
    path = dataset_root / str(record["path"])
    try:
        stat = path.stat()
    except OSError as exc:
        raise CollectionError(f"missing train-storage file {path}: {exc}") from exc
    if not path.is_file() or stat.st_size != record["size"]:
        raise CollectionError(f"train-storage size/type mismatch for {path}")
    actual = sha256_file(path)
    if actual != record["sha256"]:
        raise CollectionError(f"train-storage SHA256 mismatch for {path}")
    return actual


def freeze_frame_selection(*, preflight_path: Path, output_path: Path) -> dict[str, Any]:
    preflight = replay.read_preflight(preflight_path)
    data = preflight["payload"]["data"]
    if data.get("dataset_repo_id") != project_paths.V35_REPO_ID:
        raise CollectionError("preflight repo ID is not the registered v3.5 dataset")
    manifest_path = _preflight_path(preflight, "manifest_path_relative")
    provenance_path = _preflight_path(preflight, "norm_provenance_path_relative")
    manifest, manifest_bytes = _read_json(manifest_path)
    provenance, provenance_bytes = _read_json(provenance_path)
    if sha256_bytes(manifest_bytes) != data["manifest_sha256"]:
        raise CollectionError("frozen manifest bytes changed after preflight")
    if sha256_bytes(provenance_bytes) != data["norm_provenance_sha256"]:
        raise CollectionError("norm provenance bytes changed after preflight")
    dataset_root = project_paths.project_path(project_paths.V35_DATASET_DIR)
    expected_root = provenance.get("train_storage", {}).get("root_relative")
    if expected_root != project_paths.V35_DATASET_DIR.as_posix():
        raise CollectionError("norm provenance identifies a different project-local dataset root")
    _, by_path = _storage_maps(provenance)
    structural_names = ("meta/info.json", "meta/tasks.jsonl", "meta/episode_prompts.json")
    structural_hashes: dict[str, str] = {}
    for relative in structural_names:
        record = by_path.get(relative)
        if record is None:
            raise CollectionError(f"train-storage seal is missing structural file {relative}")
        structural_hashes[relative] = _verify_storage_file(dataset_root, record)
    tasks = _load_tasks(dataset_root / "meta/tasks.jsonl")

    # Hash each train parquet immediately before its scalar columns are read.  The callback is
    # invoked exclusively by the preflight's 74-item loop, so held-out parquet paths are never
    # resolved or opened by this stage.
    parquets, _ = _storage_maps(provenance)

    def authenticated_reader(path: Path, episode_index: int) -> EpisodeColumns:
        record = parquets.get(episode_index)
        if record is None or path != dataset_root / record["path"]:
            raise CollectionError("selection attempted to read a parquet outside the train seal")
        _verify_storage_file(dataset_root, record)
        return _read_parquet_columns(path, episode_index)

    artifact = build_frame_selection_artifact(
        preflight=preflight,
        manifest=manifest,
        provenance=provenance,
        tasks=tasks,
        read_episode_columns=authenticated_reader,
        dataset_root=dataset_root,
        structural_meta_sha256=structural_hashes,
    )
    _write_json_once(output_path, artifact)
    return artifact


def read_frame_selection(path: Path, *, preflight: Mapping[str, Any]) -> dict[str, Any]:
    artifact, _ = _read_json(path)
    if not _verify_envelope(artifact, prefix="selection"):
        raise CollectionError("frame-selection artifact hash/ID is invalid")
    payload = artifact["payload"]
    if payload.get("schema_version") != SELECTION_SCHEMA or payload.get("status") != "complete_train54_only":
        raise CollectionError("frame-selection artifact has the wrong schema/status")
    if payload.get("protocol") != SELECTION_PROTOCOL or payload.get("protocol_sha256") != selection_protocol_sha256():
        raise CollectionError("frame-selection protocol differs from this collector")
    source = payload.get("source")
    if not isinstance(source, dict) or source.get("sha256") != collector_source_sha256():
        raise CollectionError("frame selection was produced by different collector source bytes")
    if payload.get("preflight", {}).get("artifact_sha256") != preflight.get("artifact_sha256"):
        raise CollectionError("frame selection is bound to a different preflight")
    if payload.get("bindings") != dataclasses.asdict(replay.bindings_from_preflight(preflight)):
        raise CollectionError("frame selection has different replay bindings")
    data = payload.get("data")
    if (
        not isinstance(data, dict)
        or data.get("root_contract") != replay.TRAIN_STORAGE_ROOT_CONTRACT
        or data.get("dataset_root_relative") != project_paths.V35_DATASET_DIR.as_posix()
        or data.get("heldout_payload_access_count") != 0
    ):
        raise CollectionError("frame selection has the wrong dataset/held-out access contract")
    episodes = payload.get("selection", {}).get("episodes")
    expected_ids = preflight["payload"]["data"]["train_stable_ids"]
    expected_indices = preflight["payload"]["data"]["train_episode_indices"]
    if (
        not isinstance(episodes, list)
        or len(episodes) != replay.TRAIN_EPISODE_COUNT
        or [entry.get("stable_id") for entry in episodes] != expected_ids
        or [entry.get("episode_index") for entry in episodes] != expected_indices
        or any(entry.get("split") != "train" for entry in episodes)
    ):
        raise CollectionError("frame selection membership/order differs from preflight train-54")
    return artifact


def _decode_image(value: Any, *, name: str) -> np.ndarray:
    from PIL import Image

    encoded: bytes | None = None
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, bytes | bytearray | memoryview):
            encoded = bytes(raw)
        elif value.get("path") is not None:
            raise CollectionError(f"{name} unexpectedly references external image path {value['path']!r}")
    elif isinstance(value, bytes | bytearray | memoryview):
        encoded = bytes(value)
    if not encoded:
        raise CollectionError(f"{name} is not an embedded nonempty image")
    try:
        with Image.open(io.BytesIO(encoded)) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except OSError as exc:
        raise CollectionError(f"cannot decode {name}: {exc}") from exc


def _load_selected_frames(parquet_path: Path, episode: Mapping[str, Any]) -> dict[int, LoadedFrame]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise CollectionError("pyarrow is required to read calibration frames") from exc
    expected: dict[int, int] = {
        episode["evidence"]["frame_index"]: episode["evidence"]["dataset_index"],
        episode["decision"]["frame_index"]: episode["decision"]["dataset_index"],
    }
    expected.update({control["frame_index"]: control["dataset_index"] for control in episode["query_controls"]})
    columns = [
        "episode_index",
        "frame_index",
        "index",
        "task_index",
        "image",
        "left_wrist_image",
        "right_wrist_image",
        "state",
    ]
    found: dict[int, LoadedFrame] = {}
    try:
        batches = pq.ParquetFile(parquet_path).iter_batches(batch_size=8, columns=columns)
        for batch in batches:
            values = batch.to_pydict()
            for row, raw_frame in enumerate(values["frame_index"]):
                frame = int(raw_frame)
                if frame not in expected:
                    continue
                loaded = LoadedFrame(
                    episode_index=int(values["episode_index"][row]),
                    frame_index=frame,
                    dataset_index=int(values["index"][row]),
                    task_index=int(values["task_index"][row]),
                    image=_decode_image(values["image"][row], name=f"frame {frame} image"),
                    left_wrist_image=_decode_image(
                        values["left_wrist_image"][row], name=f"frame {frame} left wrist image"
                    ),
                    right_wrist_image=_decode_image(
                        values["right_wrist_image"][row], name=f"frame {frame} right wrist image"
                    ),
                    state=np.asarray(values["state"][row], dtype=np.float32),
                )
                if frame in found:
                    raise CollectionError(f"parquet contains duplicate selected frame {frame}")
                found[frame] = loaded
            if len(found) == len(expected):
                break
    except (OSError, ValueError) as exc:
        if isinstance(exc, CollectionError):
            raise
        raise CollectionError(f"cannot read selected frames from {parquet_path}: {exc}") from exc
    if set(found) != set(expected):
        raise CollectionError(f"parquet is missing selected frames {sorted(set(expected) - set(found))}")
    for frame, item in found.items():
        if (
            item.episode_index != episode["episode_index"]
            or item.dataset_index != expected[frame]
            or item.state.shape != (14,)
            or not np.all(np.isfinite(item.state))
        ):
            raise CollectionError(f"selected frame identity/state mismatch at raw frame {frame}")
    return found


def _bind_registered_config(preflight: Mapping[str, Any], *, fsdp_devices: int) -> Any:
    # Preserve the repository's Torch-before-config import order.
    # isort: off
    import openpi.training.data_loader as _data_loader  # noqa: F401
    import openpi.training.config as config_lib
    # isort: on

    config_name = preflight["payload"]["config"]["name"]
    config = config_lib.get_config(config_name)
    if config_name != "pi05_yam_mem_v35" or config.seed != preflight["payload"]["config"]["seed"]:
        raise CollectionError("registered config name/seed differs from preflight")
    if (
        config.weight_loader.params_path != replay.OFFICIAL_BASE_URI
        or preflight["payload"]["config"].get("official_base_uri") != replay.OFFICIAL_BASE_URI
        or getattr(config.model, "memory_v35_calibrated", False)
    ):
        raise CollectionError("registered replay config is not the uncalibrated official Pi0.5 fresh base")
    if float(np.float32(config.model.memory.alpha_step)) != preflight["payload"]["config"].get("alpha_step"):
        raise CollectionError("registered alpha_step differs from the authenticated preflight")
    manifest_path = _preflight_path(preflight, "manifest_path_relative")
    norm_stats_path = _preflight_path(preflight, "norm_stats_path_relative")
    repo_parts = PurePosixPath(config.data.repo_id).parts
    assets_dir = norm_stats_path.parent
    for _ in repo_parts:
        assets_dir = assets_dir.parent
    base = config.data.base_config
    if base is None:
        raise CollectionError("registered v3.5 data factory has no base config")
    base = dataclasses.replace(
        base,
        memory_episode_manifest_path=str(manifest_path),
        memory_episode_manifest_sha256=preflight["payload"]["data"]["manifest_sha256"],
        memory_manifest_split="train",
    )
    factory = dataclasses.replace(
        config.data,
        base_config=base,
        assets=dataclasses.replace(config.data.assets, assets_dir=str(assets_dir)),
    )
    graft_path = Path(preflight["_path"]).parent / "initialization_graft_manifest.json"
    loader = dataclasses.replace(config.weight_loader, manifest_output_path=str(graft_path))
    return dataclasses.replace(
        config,
        data=factory,
        weight_loader=loader,
        fsdp_devices=fsdp_devices,
        wandb_enabled=False,
        resume=False,
        overwrite=False,
    )


def _initialize_authenticated_model(config: Any, preflight: Mapping[str, Any]) -> Any:
    import flax.nnx as nnx
    import jax
    import train as train_script

    import openpi.training.sharding as sharding

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    state, _ = train_script.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    train_script._validate_v35_initialized_gate(config, state.params)  # noqa: SLF001
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    replay.verify_replay_model(model, preflight["payload"]["initialization"]["actual_step0_parameter_tree_sha256"])
    graft = replay._verify_graft_manifest(Path(config.weight_loader.manifest_output_path))  # noqa: SLF001
    initialization = preflight["payload"]["initialization"]
    if (
        graft["tree_hashes"]["source_sha256"] != initialization["official_base_source_tree_sha256"]
        or graft["tree_hashes"]["target_schema_sha256"] != initialization["target_schema_sha256"]
        or graft["manifest_sha256"] != initialization["graft_manifest_sha256"]
    ):
        raise CollectionError("reconstructed official-base graft differs from authenticated preflight")
    del state
    gc.collect()
    return model


def _make_observation_transform(config: Any) -> Callable[[LoadedFrame, str], Any]:
    import openpi.transforms as transforms

    base = config.data.base_config
    if base is None:
        raise CollectionError("registered v3.5 data factory has no base config")
    inference_base = dataclasses.replace(
        base,
        subtask_from_task=False,
        subtask_lookahead=0,
        memory_stride_frames=0,
        memory_v35_enabled=False,
        memory_v35_frozen_population=False,
    )
    factory = dataclasses.replace(config.data, base_config=inference_base)
    data_config = factory.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise CollectionError("registered replay transform could not load frozen train-54 norm stats")
    transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )

    def make(frame: LoadedFrame, prompt: str) -> Any:
        raw = {
            "image": frame.image,
            "left_wrist_image": frame.left_wrist_image,
            "right_wrist_image": frame.right_wrist_image,
            "state": frame.state.copy(),
            # Required by the registered repack, but ignored by the inference tokenizer and
            # Observation.  Keep a deterministic finite correctly-shaped placeholder.
            "actions": np.zeros((config.model.action_horizon, 14), dtype=np.float32),
            "prompt": prompt,
        }
        return _observation_from_transformed(transform(raw))

    return make


def _observation_from_transformed(value: Any) -> Any:
    """Build one unbatched Observation from a single-frame transform output.

    A single-frame transform output carries scalar Python/numpy bools (e.g. image masks);
    Observation's typed fields require arrays, so coerce every leaf first — the same
    boundary the rung collector crosses via its pre-from_dict asarray map.
    """
    import jax

    import openpi.models.model as model_lib

    value = jax.tree.map(np.asarray, value)
    observation = model_lib.Observation.from_dict(value)
    return jax.tree.map(lambda leaf: np.asarray(leaf), observation)


def _batch_observations(observations: Sequence[Any]) -> Any:
    import jax

    if not observations:
        raise CollectionError("cannot batch zero observations")
    return jax.tree.map(lambda *leaves: np.stack(leaves, axis=0), *observations)


def _synthetic_orthogonal_control(
    stored_hidden: npt.NDArray[np.float32],
    delayed_w3: npt.NDArray[np.float32],
) -> tuple[npt.NDArray[np.float32], np.float32, npt.NDArray[np.float32]]:
    """Return deterministic orthogonal hidden, its cosine, and the actual delayed raw read."""
    hidden = np.asarray(stored_hidden)
    w3 = np.asarray(delayed_w3)
    if hidden.dtype != np.float32 or hidden.ndim != 1 or not np.all(np.isfinite(hidden)):
        raise CollectionError("stored hidden key must be one finite FP32 vector")
    if w3.dtype != np.float32 or w3.ndim != 2 or w3.shape[0] != hidden.size or not np.all(np.isfinite(w3)):
        raise CollectionError("delayed output fast weight must be finite FP32 [hidden,channel]")
    if float(np.sqrt(np.sum(np.square(w3), dtype=np.float32), dtype=np.float32)) <= 1e-12:
        raise CollectionError("synthetic fallback must read the actual nonzero delayed memory, not zero memory")
    norm = np.sqrt(np.sum(np.square(hidden), dtype=np.float32), dtype=np.float32)
    if not np.isfinite(norm) or norm <= np.float32(1e-12):
        raise CollectionError("cannot construct an orthogonal control from a zero stored hidden key")
    unit = hidden / norm
    coordinate = int(np.argmin(np.abs(unit)))
    basis = np.zeros_like(unit)
    basis[coordinate] = np.float32(1.0)
    projection = np.sum(basis * unit, dtype=np.float32)
    orthogonal = basis - projection * unit
    orthogonal_norm = np.sqrt(np.sum(np.square(orthogonal), dtype=np.float32), dtype=np.float32)
    if not np.isfinite(orthogonal_norm) or orthogonal_norm <= np.float32(1e-12):
        raise CollectionError("canonical Gram-Schmidt orthogonal control is degenerate")
    orthogonal = np.asarray(orthogonal / orthogonal_norm, dtype=np.float32)
    cosine = np.float32(
        np.sum(orthogonal * hidden, dtype=np.float32)
        / np.maximum(
            np.sqrt(np.sum(np.square(orthogonal), dtype=np.float32), dtype=np.float32) * norm,
            np.float32(1e-12),
        )
    )
    raw_read = np.asarray(orthogonal @ w3, dtype=np.float32)
    if not np.all(np.isfinite(raw_read)) or not -1.0 <= float(cosine) <= replay.LOW_COSINE_MAX:
        raise CollectionError(f"synthetic orthogonal control failed its cosine/read contract: cosine={cosine}")
    return orthogonal, cosine, raw_read


def _collect_episode(
    *,
    model: Any,
    bindings: replay.ReplayBindings,
    selection_sha256: str,
    episode: Mapping[str, Any],
    observations: Mapping[int, Any],
    query_batch_size: int,
) -> replay.EpisodeReplayRecord:
    import jax
    import jax.numpy as jnp

    stable_id = episode["stable_id"]
    evidence_frame = episode["evidence"]["frame_index"]
    decision_frame = episode["decision"]["frame_index"]
    n_delay = episode["decision"]["n_delay"]
    evidence_observation = _batch_observations([observations[evidence_frame]])
    decision_observation = _batch_observations([observations[decision_frame]])

    state0 = model.memory.init_state(1)
    evidence = replay._prepare_early_interface(model, evidence_observation, state0)  # noqa: SLF001
    write_keys, write_values = model.memory.project_kv(evidence["write_tokens"])
    pooled = model.memory.pool_kv(write_keys, write_values)
    state_e, commit = model.memory.write(state0, evidence["write_tokens"])
    commit_applied = bool(np.asarray(jax.device_get(commit["commit_applied"]))[0])
    if not commit_applied:
        raise CollectionError(f"episode {stable_id!r} evidence direct commit was not applied")
    stored_hidden = model.memory.hidden_key(state_e, pooled["pooled_key"][:, None, :])[:, 0, :]

    clean_decision = replay._prepare_early_interface(model, decision_observation, state_e)  # noqa: SLF001
    clean_host = np.asarray(jax.device_get(clean_decision["retrieved"]), dtype=np.float32)[0]
    state_d, decay_aux = model.memory.analytic_decay(state_e, n_delay)
    if not bool(np.asarray(jax.device_get(decay_aux["decay_gap_valid"]))[0]):
        raise CollectionError(f"episode {stable_id!r} decision decay gap was rejected")
    decision = replay._prepare_early_interface(model, decision_observation, state_d)  # noqa: SLF001
    h8 = decision["h8_all"].astype(jnp.float32)
    valid = decision["prefix_mask"].astype(jnp.float32)[..., None]
    residual_profile = jnp.sqrt(jnp.sum(jnp.square(h8) * valid, axis=1) / jnp.maximum(jnp.sum(valid, axis=1), 1.0))
    residual_host = np.asarray(jax.device_get(residual_profile), dtype=np.float32)[0]

    query_reads: list[np.ndarray] = []
    query_cosines: list[np.float32] = []
    query_frames: list[int] = []
    query_hashes: list[str] = []
    stored_norm = jnp.linalg.norm(stored_hidden, axis=-1, keepdims=True)
    controls = episode["query_controls"]
    if type(query_batch_size) is not int or query_batch_size <= 0:
        raise CollectionError("query_batch_size must be a positive integer")
    for start in range(0, len(controls), query_batch_size):
        chunk = controls[start : start + query_batch_size]
        chunk_observations = [observations[item["frame_index"]] for item in chunk]
        observation_batch = _batch_observations(chunk_observations)
        batch_count = len(chunk)
        state_batch = jax.tree.map(
            lambda leaf, count=batch_count: jnp.broadcast_to(leaf, (count, *leaf.shape[1:])), state_e
        )
        gaps = jnp.asarray([item["n_delay"] for item in chunk], dtype=jnp.int32)
        control_state, control_aux = model.memory.analytic_decay(state_batch, gaps)
        if not np.all(np.asarray(jax.device_get(control_aux["decay_gap_valid"]))):
            raise CollectionError(f"episode {stable_id!r} query-control decay gap was rejected")
        prepared = replay._prepare_early_interface(model, observation_batch, control_state)  # noqa: SLF001
        query_keys = model.memory.project_q(prepared["read_queries"])
        query_hidden = model.memory.hidden_key(control_state, query_keys)
        cosine = jnp.sum(query_hidden * stored_hidden, axis=-1) / jnp.maximum(
            jnp.linalg.norm(query_hidden, axis=-1) * stored_norm,
            1e-12,
        )
        cosine_host = np.asarray(jax.device_get(cosine), dtype=np.float32)
        retrieved_host = np.asarray(jax.device_get(prepared["retrieved"]), dtype=np.float32)
        for local, item in enumerate(chunk):
            selected_slots = np.nonzero(cosine_host[local] <= replay.LOW_COSINE_MAX)[0]
            observation_sha = replay._tree_array_sha256(chunk_observations[local])  # noqa: SLF001
            for slot in selected_slots:
                query_reads.append(retrieved_host[local, slot])
                query_cosines.append(np.float32(cosine_host[local, slot]))
                query_frames.append(item["frame_index"])
                query_hashes.append(observation_sha)

    query_kind: tuple[str, ...]
    if query_reads:
        query_kind = tuple("low_cos_query" for _ in query_reads)
    else:
        hidden_host = np.asarray(jax.device_get(stored_hidden), dtype=np.float32)[0]
        w3_name = model.memory._output_weight_name  # noqa: SLF001
        w3_host = np.asarray(jax.device_get(state_d.fast_weights[w3_name]), dtype=np.float32)[0]
        _, cosine, raw_read = _synthetic_orthogonal_control(hidden_host, w3_host)
        query_reads = [raw_read]
        query_cosines = [cosine]
        query_frames = [-1]
        query_hashes = [
            sha256_bytes(
                canonical_json_bytes(
                    {
                        "algorithm": SELECTION_PROTOCOL["synthetic_fallback"]["algorithm"],
                        "selection_sha256": selection_sha256,
                        "stable_id": stable_id,
                        "n_delay": n_delay,
                    }
                )
            )
        ]
        query_kind = ("synthetic_orthogonal_query",)

    mixed_host = np.asarray(jax.device_get(commit["post_residual"]), dtype=np.float32)[0]
    relative = float(np.asarray(jax.device_get(commit["relative_commit_residual"]))[0])
    record = replay.EpisodeReplayRecord(
        stable_id=stable_id,
        split="train",
        evidence_frame_index=evidence_frame,
        decision_frame_index=decision_frame,
        n_delay=n_delay,
        evidence_observation_sha256=replay._tree_array_sha256(evidence_observation),  # noqa: SLF001
        decision_observation_sha256=replay._tree_array_sha256(decision_observation),  # noqa: SLF001
        clean_raw_retrieved=clean_host,
        layer8_residual=residual_host,
        mixed_precision_noise_raw_retrieved=mixed_host,
        query_noise_raw_retrieved=np.stack(query_reads).astype(np.float32),
        query_noise_cosine=np.asarray(query_cosines, dtype=np.float32),
        query_noise_kind=query_kind,
        query_noise_frame_index=np.asarray(query_frames, dtype=np.int64),
        query_noise_observation_sha256=tuple(query_hashes),
        commit_relative_residual=relative,
        commit_applied=commit_applied,
        bindings=bindings,
    )
    replay.validate_episode_record(record)
    if record.clean_raw_retrieved.shape[0] != 16:
        raise CollectionError(f"episode {stable_id!r} did not preserve exactly 16 clean decision slots")
    return record


def _validate_existing_shard(
    path: Path,
    *,
    episode: Mapping[str, Any],
    bindings: replay.ReplayBindings,
) -> bool:
    if not path.exists():
        return False
    record, _ = replay.load_episode_replay_shard(path)
    if (
        record.stable_id != episode["stable_id"]
        or record.bindings != bindings
        or record.evidence_frame_index != episode["evidence"]["frame_index"]
        or record.decision_frame_index != episode["decision"]["frame_index"]
        or record.n_delay != episode["decision"]["n_delay"]
    ):
        raise CollectionError(f"existing shard {path} does not match its frozen selection")
    allowed_frames = {item["frame_index"] for item in episode["query_controls"]}
    for kind, frame in zip(record.query_noise_kind, record.query_noise_frame_index, strict=True):
        if kind == "low_cos_query" and int(frame) not in allowed_frames:
            raise CollectionError(f"existing shard {path} contains an unregistered real query frame")
    return True


def collect_shard(
    *,
    preflight_path: Path,
    selection_path: Path,
    shards_dir: Path,
    shard_index: int,
    num_shards: int,
    query_batch_size: int,
    fsdp_devices: int,
) -> tuple[int, int]:
    if type(num_shards) is not int or num_shards <= 0:
        raise CollectionError("num_shards must be a positive integer")
    if type(shard_index) is not int or not 0 <= shard_index < num_shards:
        raise CollectionError("shard_index must lie in [0,num_shards)")
    if type(fsdp_devices) is not int or fsdp_devices <= 0:
        raise CollectionError("fsdp_devices must be a positive integer")
    preflight = replay.read_preflight(preflight_path)
    preflight = dict(preflight)
    preflight["_path"] = str(preflight_path)
    selection = read_frame_selection(selection_path, preflight=preflight)
    episodes = selection["payload"]["selection"]["episodes"]
    bindings = replay.bindings_from_preflight(preflight)
    shards_dir.mkdir(parents=True, exist_ok=True)
    assigned = [(ordinal, episode) for ordinal, episode in enumerate(episodes) if ordinal % num_shards == shard_index]
    pending = [
        (ordinal, episode)
        for ordinal, episode in assigned
        if not _validate_existing_shard(shards_dir / f"episode_{ordinal:03d}.npz", episode=episode, bindings=bindings)
    ]
    if not pending:
        return len(assigned), 0

    config = _bind_registered_config(preflight, fsdp_devices=fsdp_devices)
    model = _initialize_authenticated_model(config, preflight)
    make_observation = _make_observation_transform(config)
    dataset_root = project_paths.project_path(project_paths.V35_DATASET_DIR)
    written = 0
    for ordinal, episode in pending:
        parquet = dataset_root / episode["parquet_path_relative"]
        if sha256_file(parquet) != episode["parquet_sha256"]:
            raise CollectionError(f"train parquet bytes changed before replay: {episode['stable_id']}")
        frames = _load_selected_frames(parquet, episode)
        observations = {frame: make_observation(value, episode["prompt"]) for frame, value in frames.items()}
        record = _collect_episode(
            model=model,
            bindings=bindings,
            selection_sha256=selection["artifact_sha256"],
            episode=episode,
            observations=observations,
            query_batch_size=query_batch_size,
        )
        replay.write_episode_replay_shard(shards_dir / f"episode_{ordinal:03d}.npz", record)
        written += 1
    return len(assigned), written


def seal_collection(
    *,
    preflight_path: Path,
    selection_path: Path,
    shards_dir: Path,
    output_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    preflight = replay.read_preflight(preflight_path)
    selection = read_frame_selection(selection_path, preflight=preflight)
    episodes = selection["payload"]["selection"]["episodes"]
    bindings = replay.bindings_from_preflight(preflight)
    shard_records: list[dict[str, Any]] = []
    synthetic_episode_count = 0
    natural_query_slot_count = 0
    for ordinal, episode in enumerate(episodes):
        path = shards_dir / f"episode_{ordinal:03d}.npz"
        if not _validate_existing_shard(path, episode=episode, bindings=bindings):
            raise CollectionError(f"missing replay shard for {episode['stable_id']}")
        record, digest = replay.load_episode_replay_shard(path)
        synthetic = "synthetic_orthogonal_query" in record.query_noise_kind
        synthetic_episode_count += int(synthetic)
        natural_query_slot_count += sum(kind == "low_cos_query" for kind in record.query_noise_kind)
        shard_records.append(
            {
                "ordinal": ordinal,
                "stable_id": record.stable_id,
                "file": path.name,
                "sha256": digest,
                "clean_slot_count": int(record.clean_raw_retrieved.shape[0]),
                "query_control_count": len(record.query_noise_kind),
                "synthetic_fallback": synthetic,
            }
        )
    output_sha = replay.seal_replay_npz(
        preflight_path=preflight_path,
        shards_dir=shards_dir,
        output_path=output_path,
    )
    payload = {
        "schema_version": COLLECTION_RECEIPT_SCHEMA,
        "status": "complete_train54_replay",
        "preflight_sha256": preflight["artifact_sha256"],
        "selection_sha256": selection["artifact_sha256"],
        "selection_protocol_sha256": selection_protocol_sha256(),
        "collector_source_sha256": collector_source_sha256(),
        "replay_collector_source_sha256": replay.collector_source_sha256(),
        "bindings": dataclasses.asdict(bindings),
        "output": {
            "file": output_path.name,
            "sha256": output_sha,
            "format": "strict NPZ accepted by v35_injection_calibration.py",
        },
        "summary": {
            "episode_count": len(shard_records),
            "clean_slots_per_episode": 16,
            "natural_low_cos_query_slot_count": natural_query_slot_count,
            "synthetic_fallback_episode_count": synthetic_episode_count,
            "heldout_payload_access_count": 0,
        },
        "episode_shards": shard_records,
    }
    receipt = _envelope(payload, prefix="collection")
    _write_json_once(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="Freeze train-only frame/query selection.")
    select.add_argument("--preflight", type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)

    collect = commands.add_parser("collect", help="Collect one deterministic ordinal GPU shard.")
    collect.add_argument("--preflight", type=Path, required=True)
    collect.add_argument("--selection", type=Path, required=True)
    collect.add_argument("--shards-dir", type=Path, required=True)
    collect.add_argument("--shard-index", type=int, default=0)
    collect.add_argument("--num-shards", type=int, default=1)
    collect.add_argument("--query-batch-size", type=int, default=8)
    collect.add_argument("--fsdp-devices", type=int, default=1)

    seal = commands.add_parser("seal", help="Seal 54 episode shards and their provenance receipt.")
    seal.add_argument("--preflight", type=Path, required=True)
    seal.add_argument("--selection", type=Path, required=True)
    seal.add_argument("--shards-dir", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    seal.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        project_paths.configure_v35_runtime_environment()
        if args.command == "select":
            preflight_path = _resolve_project_path(args.preflight, name="preflight")
            output_path = _resolve_project_path(args.output, name="selection output")
            artifact = freeze_frame_selection(preflight_path=preflight_path, output_path=output_path)
            print(f"wrote train-54 frame selection: {output_path} (sha256:{artifact['artifact_sha256']})")
        elif args.command == "collect":
            preflight_path = _resolve_project_path(args.preflight, name="preflight")
            selection_path = _resolve_project_path(args.selection, name="selection")
            shards_dir = _resolve_project_path(args.shards_dir, name="shards directory")
            assigned, written = collect_shard(
                preflight_path=preflight_path,
                selection_path=selection_path,
                shards_dir=shards_dir,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                query_batch_size=args.query_batch_size,
                fsdp_devices=args.fsdp_devices,
            )
            print(
                f"collection shard {args.shard_index}/{args.num_shards}: assigned={assigned}, newly_written={written}"
            )
        else:
            preflight_path = _resolve_project_path(args.preflight, name="preflight")
            selection_path = _resolve_project_path(args.selection, name="selection")
            shards_dir = _resolve_project_path(args.shards_dir, name="shards directory")
            output_path = _resolve_project_path(args.output, name="sealed replay output")
            receipt_path = _resolve_project_path(args.receipt, name="collection receipt")
            receipt = seal_collection(
                preflight_path=preflight_path,
                selection_path=selection_path,
                shards_dir=shards_dir,
                output_path=output_path,
                receipt_path=receipt_path,
            )
            print(f"wrote sealed replay and receipt: sha256:{receipt['artifact_sha256']}")
    except (CollectionError, replay.ReplayPreflightError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
