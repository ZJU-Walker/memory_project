"""Verify the converted 0816 + 0830 v3.5 LeRobot dataset against its raw manifest.

This is intentionally independent of the converter's in-memory counters.  It reads every
numeric parquet row and checks episode identity, lengths, task indices, state, actions, prompts,
and the strict 14-D waiting cores against the raw sources.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

EXPECTED_EPISODES = 90
EXPECTED_FRAMES = 77_445
EXPECTED_TASKS = 7
EXPECTED_FPS = 30
EXPECTED_MAX_STATE_STEP = 4e-3
EXPECTED_MAX_STATE_EXCURSION = 2e-2
EXPECTED_STRIDE_FRAMES = 15
EXPECTED_ELIGIBLE_AT_STRIDE = 89
EXPECTED_IMAGE_SAMPLES = 1080
EXPECTED_TASK_VOCABULARY = (
    "open both lids",
    "wait; target bin is left",
    "open left bin",
    "close both lids and reset arms",
    "inspect both bins",
    "open right bin",
    "wait; target bin is right",
)
EXPECTED_PROMPT_COUNTS = {"find the banana": 44, "find the grey pepper box": 46}
EXPECTED_SIDE_COUNTS = {"left": 45, "right": 45}
EXPECTED_COLLECTION_COUNTS = {
    "0816_banana": 30,
    "0816_grey_box": 30,
    "0830_bin_part1": 16,
    "0830_bin_part2": 14,
}
EXPECTED_SOURCE_ORDER = list(EXPECTED_COLLECTION_COUNTS)
EXPECTED_EXCLUDED_STABLE_IDS = {"0830_bin_part2/demo14"}
EXPECTED_INDEX_ANCHORS = {
    "15": "0816_banana/demo16",
    "29": "0816_banana/demo30",
    "44": "0816_grey_box/demo15",
    "59": "0816_grey_box/demo30",
    "60": "0830_bin_part1/demo1",
    "75": "0830_bin_part1/demo16",
    "76": "0830_bin_part2/demo1",
    "88": "0830_bin_part2/demo13",
    "89": "0830_bin_part2/demo15",
}
ARRAY_STREAMS = (
    "left_joint_positions.npy",
    "right_joint_positions.npy",
    "left_control.npy",
    "right_control.npy",
)
VIDEO_TO_IMAGE_COLUMN = {
    "top_camera_rgb.mp4": "image",
    "left_camera_rgb.mp4": "left_wrist_image",
    "right_camera_rgb.mp4": "right_wrist_image",
}
IMAGE_COLUMNS = tuple(VIDEO_TO_IMAGE_COLUMN.values())
NUMERIC_COLUMNS = (
    "episode_index",
    "frame_index",
    "index",
    "timestamp",
    "task_index",
    "state",
    "actions",
)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _resolve_raw_dir(manifest_path: Path, manifest: dict[str, Any], value: str) -> Path:
    raw_dir = Path(value)
    if raw_dir.is_absolute():
        return raw_dir.resolve()
    root = Path(manifest.get("raw_root", "."))
    if not root.is_absolute():
        root = manifest_path.parent / root
    return (root / raw_dir).resolve()


def _validate_five_phase_labels(
    label_path: Path,
    num_frames: int,
    target_side: str,
    task_to_index: dict[str, int],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    _require(target_side in {"left", "right"}, f"{label_path}: invalid target side {target_side!r}")
    segments = json.loads(label_path.read_text())
    _require(isinstance(segments, list), f"{label_path}: labels must be a list")
    _require(len(segments) == 5, f"{label_path}: expected exactly five phases, found {len(segments)}")
    expected_tasks = [
        "open both lids",
        "inspect both bins",
        "close both lids and reset arms",
        f"wait; target bin is {target_side}",
        f"open {target_side} bin",
    ]
    actual_tasks = [segment.get("task") for segment in segments]
    _require(
        actual_tasks == expected_tasks,
        f"{label_path}: expected phase order {expected_tasks}, found {actual_tasks}",
    )
    expected = np.full(num_frames, -1, dtype=np.int64)
    cursor = 0
    for phase_index, segment in enumerate(segments):
        _require(
            set(segment) == {"task", "start", "end"},
            f"{label_path}: phase {phase_index} has unexpected fields {sorted(segment)}",
        )
        _require(
            isinstance(segment["start"], int) and not isinstance(segment["start"], bool),
            f"{label_path}: phase {phase_index} start is not an integer",
        )
        _require(
            isinstance(segment["end"], int) and not isinstance(segment["end"], bool),
            f"{label_path}: phase {phase_index} end is not an integer",
        )
        _require(segment["start"] == cursor, f"{label_path}: non-contiguous labels at frame {cursor}")
        _require(segment["end"] >= cursor, f"{label_path}: empty/reversed phase {phase_index}")
        stop = int(segment["end"]) + 1
        _require(stop <= num_frames, f"{label_path}: phase {phase_index} extends past frame {num_frames - 1}")
        expected[cursor:stop] = task_to_index[segment["task"]]
        cursor = stop
    _require(cursor == num_frames, f"{label_path}: expected {num_frames} labeled frames, found {cursor}")
    _require(not np.any(expected < 0), f"{label_path}: one or more frames have no task")
    return segments, expected


def _parquet_path(dataset: Path, episode_index: int) -> Path:
    matches = list(dataset.glob(f"data/chunk-*/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise AssertionError(f"episode {episode_index}: expected one parquet, found {matches}")
    return matches[0]


def _expected_arrow_types() -> dict[str, pa.DataType]:
    image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    return {
        "image": image_type,
        "left_wrist_image": image_type,
        "right_wrist_image": image_type,
        "state": pa.list_(pa.float32(), 14),
        "actions": pa.list_(pa.float32(), 14),
        "timestamp": pa.float32(),
        "frame_index": pa.int64(),
        "episode_index": pa.int64(),
        "index": pa.int64(),
        "task_index": pa.int64(),
    }


def _verify_parquet_schema(parquet: pq.ParquetFile, stable_id: str) -> None:
    schema = parquet.schema_arrow
    expected = _expected_arrow_types()
    _require(
        set(schema.names) == set(expected),
        f"{stable_id}: parquet columns {schema.names} do not match expected {list(expected)}",
    )
    for name, expected_type in expected.items():
        actual_type = schema.field(name).type
        _require(actual_type == expected_type, f"{stable_id}: {name} type {actual_type} != {expected_type}")


def _verify_image_metadata(parquet: pq.ParquetFile, stable_id: str) -> None:
    image_leaves = {
        "image.bytes",
        "image.path",
        "left_wrist_image.bytes",
        "left_wrist_image.path",
        "right_wrist_image.bytes",
        "right_wrist_image.path",
    }
    for row_group_index in range(parquet.metadata.num_row_groups):
        row_group = parquet.metadata.row_group(row_group_index)
        observed = set()
        for column_index in range(row_group.num_columns):
            column = row_group.column(column_index)
            if column.path_in_schema not in image_leaves:
                continue
            observed.add(column.path_in_schema)
            if column.num_values != row_group.num_rows:
                raise AssertionError(
                    f"{stable_id}: {column.path_in_schema} has {column.num_values} values for {row_group.num_rows} rows"
                )
            if column.statistics is not None and column.statistics.null_count != 0:
                raise AssertionError(f"{stable_id}: {column.path_in_schema} contains nulls")
        if observed != image_leaves:
            raise AssertionError(f"{stable_id}: image columns missing from row group {row_group_index}")


def _row_group_samples(parquet: pq.ParquetFile, frame_indices: list[int]) -> dict[int, list[tuple[int, int]]]:
    remaining = set(frame_indices)
    grouped: dict[int, list[tuple[int, int]]] = {}
    offset = 0
    for row_group_index in range(parquet.metadata.num_row_groups):
        rows = parquet.metadata.row_group(row_group_index).num_rows
        hits = sorted(index for index in remaining if offset <= index < offset + rows)
        if hits:
            grouped[row_group_index] = [(index, index - offset) for index in hits]
            remaining.difference_update(hits)
        offset += rows
    _require(not remaining, f"parquet does not contain requested sample frames {sorted(remaining)}")
    return grouped


def _output_image_samples(
    parquet: pq.ParquetFile,
    frame_indices: list[int],
    stable_id: str,
) -> dict[str, dict[int, np.ndarray]]:
    result: dict[str, dict[int, np.ndarray]] = {column: {} for column in IMAGE_COLUMNS}
    for row_group_index, samples in _row_group_samples(parquet, frame_indices).items():
        table = parquet.read_row_group(row_group_index, columns=list(IMAGE_COLUMNS))
        for frame_index, local_index in samples:
            for column_name in IMAGE_COLUMNS:
                record = table[column_name][local_index].as_py()
                _require(isinstance(record, dict), f"{stable_id}: {column_name}[{frame_index}] is not an image struct")
                encoded = record.get("bytes")
                path = record.get("path")
                _require(
                    isinstance(encoded, bytes) and len(encoded) > 0,
                    f"{stable_id}: {column_name}[{frame_index}] has empty image bytes",
                )
                _require(
                    path == f"frame_{frame_index:06d}.png",
                    f"{stable_id}: {column_name}[{frame_index}] path is {path!r}",
                )
                with Image.open(io.BytesIO(encoded)) as image:
                    _require(image.format == "PNG", f"{stable_id}: {column_name}[{frame_index}] is {image.format}")
                    _require(image.mode == "RGB", f"{stable_id}: {column_name}[{frame_index}] mode is {image.mode}")
                    _require(
                        image.size == (640, 480),
                        f"{stable_id}: {column_name}[{frame_index}] size is {image.size}",
                    )
                    image.load()
                    pixels = np.asarray(image, dtype=np.uint8).copy()
                result[column_name][frame_index] = pixels
    return result


def _raw_video_samples(path: Path, frame_indices: list[int], stable_id: str) -> tuple[dict[int, np.ndarray], int]:
    wanted = set(frame_indices)
    samples: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    _require(cap.isOpened(), f"{stable_id}: cannot open raw video {path}")
    frame_index = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        _require(
            bgr.shape == (480, 640, 3) and bgr.dtype == np.uint8,
            f"{stable_id}: {path.name} frame {frame_index} has shape/dtype {bgr.shape}/{bgr.dtype}",
        )
        if frame_index in wanted:
            samples[frame_index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frame_index += 1
    cap.release()
    missing = wanted - set(samples)
    _require(not missing, f"{stable_id}: {path.name} is missing sample frames {sorted(missing)}")
    return samples, frame_index


def _verify_image_samples(
    parquet: pq.ParquetFile,
    raw_dir: Path,
    sample_indices: list[int],
    expected_stream_lengths: dict[str, int],
    stable_id: str,
    digest: Any,
) -> tuple[int, dict[str, int]]:
    output = _output_image_samples(parquet, sample_indices, stable_id)
    decoded = 0
    decoded_lengths: dict[str, int] = {}
    for video_name, column_name in VIDEO_TO_IMAGE_COLUMN.items():
        raw, decoded_length = _raw_video_samples(raw_dir / video_name, sample_indices, stable_id)
        decoded_lengths[video_name] = decoded_length
        _require(
            decoded_length == int(expected_stream_lengths[video_name]),
            f"{stable_id}: decoded {video_name} length {decoded_length} != provenance "
            f"{expected_stream_lengths[video_name]}",
        )
        for frame_index in sample_indices:
            actual = output[column_name][frame_index]
            expected = raw[frame_index]
            _require(
                np.array_equal(actual, expected),
                f"{stable_id}: {column_name} frame {frame_index} differs pixelwise from {video_name}",
            )
            digest.update(actual.tobytes())
            decoded += 1
    return decoded, decoded_lengths


def _validate_numeric_stats(
    stats: dict[str, Any],
    values_by_name: dict[str, np.ndarray],
    num_frames: int,
    stable_id: str,
) -> None:
    for name, values in values_by_name.items():
        feature_stats = stats.get(name)
        _require(isinstance(feature_stats, dict), f"{stable_id}: episodes_stats missing {name}")
        _require(feature_stats.get("count") == [num_frames], f"{stable_id}: {name} stats count is wrong")
        expected_values = {
            "min": values.min(axis=0),
            "max": values.max(axis=0),
            "mean": values.mean(axis=0),
            "std": values.std(axis=0),
        }
        for statistic, expected in expected_values.items():
            actual = np.asarray(feature_stats.get(statistic))
            _require(
                actual.shape == np.asarray(expected).shape,
                f"{stable_id}: {name}.{statistic} shape {actual.shape} != {np.asarray(expected).shape}",
            )
            _require(np.all(np.isfinite(actual)), f"{stable_id}: {name}.{statistic} contains non-finite values")
            _require(
                np.allclose(actual, expected, rtol=1e-6, atol=1e-7),
                f"{stable_id}: {name}.{statistic} differs from parquet values",
            )


def _validate_image_stats(stats: dict[str, Any], num_frames: int, stable_id: str) -> None:
    for name in IMAGE_COLUMNS:
        feature_stats = stats.get(name)
        _require(isinstance(feature_stats, dict), f"{stable_id}: episodes_stats missing {name}")
        count = feature_stats.get("count")
        _require(
            isinstance(count, list) and len(count) == 1 and 0 < int(count[0]) <= num_frames,
            f"{stable_id}: {name} stats count is invalid: {count}",
        )
        for statistic in ("min", "max", "mean", "std"):
            value = np.asarray(feature_stats.get(statistic), dtype=np.float64)
            _require(value.shape == (3, 1, 1), f"{stable_id}: {name}.{statistic} shape is {value.shape}")
            _require(np.all(np.isfinite(value)), f"{stable_id}: {name}.{statistic} contains non-finite values")
        minimum = np.asarray(feature_stats["min"])
        maximum = np.asarray(feature_stats["max"])
        _require(np.all(minimum >= 0) and np.all(maximum <= 1), f"{stable_id}: {name} stats leave [0,1]")
        _require(np.all(minimum <= maximum), f"{stable_id}: {name} stats min exceeds max")


def _validate_info(info: dict[str, Any]) -> None:
    expected_top_level = {
        "codebase_version": "v2.1",
        "robot_type": "yam",
        "total_episodes": EXPECTED_EPISODES,
        "total_frames": EXPECTED_FRAMES,
        "total_tasks": EXPECTED_TASKS,
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": EXPECTED_FPS,
        "splits": {"train": "0:90"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    _require(
        set(info) == {*expected_top_level, "features"},
        f"info.json top-level fields are {sorted(info)}",
    )
    for name, expected_value in expected_top_level.items():
        _require(info.get(name) == expected_value, f"info.json {name}={info.get(name)!r}, expected {expected_value!r}")
    features = info.get("features", {})
    expected_features = {
        "image": ("image", [480, 640, 3], ["height", "width", "channel"]),
        "left_wrist_image": ("image", [480, 640, 3], ["height", "width", "channel"]),
        "right_wrist_image": ("image", [480, 640, 3], ["height", "width", "channel"]),
        "state": ("float32", [14], ["state"]),
        "actions": ("float32", [14], ["actions"]),
        "timestamp": ("float32", [1], None),
        "frame_index": ("int64", [1], None),
        "episode_index": ("int64", [1], None),
        "index": ("int64", [1], None),
        "task_index": ("int64", [1], None),
    }
    _require(set(features) == set(expected_features), f"info.json feature keys are {sorted(features)}")
    for name, (dtype, shape, names) in expected_features.items():
        _require(set(features[name]) == {"dtype", "shape", "names"}, f"info.json {name} fields drifted")
        _require(features[name].get("dtype") == dtype, f"info.json {name} dtype != {dtype}")
        _require(features[name].get("shape") == shape, f"info.json {name} shape != {shape}")
        _require(features[name].get("names") == names, f"info.json {name} names != {names}")


def _run_self_checks() -> list[str]:
    _require(len(EXPECTED_TASK_VOCABULARY) == EXPECTED_TASKS, "task count constant disagrees with vocabulary")
    _require(len(set(EXPECTED_TASK_VOCABULARY)) == EXPECTED_TASKS, "task vocabulary contains duplicates")
    _require(
        EXPECTED_EPISODES * 4 * len(IMAGE_COLUMNS) == EXPECTED_IMAGE_SAMPLES,
        "image sample cardinality invariant failed",
    )
    vocabulary = {task: index for index, task in enumerate(EXPECTED_TASK_VOCABULARY)}
    labels = [
        {"task": "open both lids", "start": 0, "end": 1},
        {"task": "inspect both bins", "start": 2, "end": 3},
        {"task": "close both lids and reset arms", "start": 4, "end": 5},
        {"task": "wait; target bin is left", "start": 6, "end": 7},
        {"task": "open left bin", "start": 8, "end": 9},
    ]
    expected = np.asarray([0, 0, 4, 4, 3, 3, 1, 1, 2, 2], dtype=np.int64)
    # Exercise the same construction without a temporary file.
    observed = np.concatenate(
        [np.full(segment["end"] - segment["start"] + 1, vocabulary[segment["task"]]) for segment in labels]
    )
    _require(np.array_equal(observed, expected), "internal task-index self-check failed")
    _require(pa.list_(pa.float32(), 14).list_size == 14, "internal Arrow fixed-size-list self-check failed")
    return ["pinned dataset constants", "task-index phase expansion", "Arrow float32[14] schema"]


def verify(dataset: Path, manifest_path: Path) -> dict[str, Any]:
    self_checks = _run_self_checks()
    _require(dataset.is_dir(), f"dataset directory does not exist: {dataset}")
    _require(manifest_path.is_file(), f"manifest does not exist: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    _require(manifest.get("schema_version") == 1, "manifest schema_version != 1")
    _require(manifest.get("dataset_version") == "v35", "manifest dataset_version != v35")
    _require(manifest.get("source_order") == EXPECTED_SOURCE_ORDER, "manifest source_order drifted")
    raw_episodes = manifest.get("episodes")
    _require(isinstance(raw_episodes, list), "manifest episodes must be a list")
    included_with_positions = [
        (position, episode) for position, episode in enumerate(raw_episodes) if episode.get("include", True)
    ]
    included = [episode for _, episode in included_with_positions]
    excluded = [episode for episode in raw_episodes if not episode.get("include", True)]
    _require(len(included) == EXPECTED_EPISODES, f"manifest includes {len(included)} episodes, expected 90")
    _require(len(excluded) == 1, f"manifest excludes {len(excluded)} episodes, expected exactly one")
    _require(
        {episode.get("stable_id") for episode in excluded} == EXPECTED_EXCLUDED_STABLE_IDS,
        f"manifest excluded IDs are {[episode.get('stable_id') for episode in excluded]}",
    )
    approved = manifest.get("review_status") == "user_approved"
    if approved:
        approval = manifest.get("approval")
        _require(isinstance(approval, dict), "approved manifest has no approval record")
        _require(approval.get("reviewer") == "user", "approved manifest reviewer is not user")
        _require(bool(approval.get("approved_at")), "approved manifest has no approval timestamp")
        _require(
            all(episode.get("approval_status") == "user_approved" for episode in included),
            "not every included episode is user_approved",
        )
        _require(
            excluded[0].get("approval_status") == "user_approved_exclusion",
            "excluded demo14 is not explicitly user-approved for exclusion",
        )
    expected = manifest["expected"]
    _require(int(expected.get("included_episodes", -1)) == EXPECTED_EPISODES, "manifest expected episodes != 90")
    _require(int(expected.get("included_frames", -1)) == EXPECTED_FRAMES, "manifest expected frames != 77445")
    _require(expected.get("prompt_counts") == EXPECTED_PROMPT_COUNTS, "manifest prompt counts drifted")
    _require(expected.get("target_side_counts") == EXPECTED_SIDE_COUNTS, "manifest side counts drifted")
    _require(
        expected.get("episode_index_to_stable_id") == EXPECTED_INDEX_ANCHORS,
        "manifest episode identity anchors drifted",
    )
    for flag in (
        "require_memory_waiting_core",
        "require_semantic_wait_equals_core",
        "require_exact_five_phase_schema",
    ):
        _require(expected.get(flag) is True, f"manifest expected.{flag} is not true")

    vocabulary = manifest["task_vocabulary"]
    _require(tuple(vocabulary) == EXPECTED_TASK_VOCABULARY, "manifest task vocabulary/order drifted")
    task_to_index = {task: index for index, task in enumerate(vocabulary)}
    _require(len(task_to_index) == EXPECTED_TASKS, "manifest task vocabulary contains duplicates")
    waiting_config = manifest.get("memory_waiting_core_config", {})
    _require(waiting_config.get("state_dimensions") == 14, "manifest waiting state_dimensions != 14")
    _require(
        float(waiting_config.get("max_state_step", math.inf)) == EXPECTED_MAX_STATE_STEP,
        "manifest max_state_step != 0.004",
    )
    _require(
        float(waiting_config.get("max_state_excursion", math.inf)) == EXPECTED_MAX_STATE_EXCURSION,
        "manifest max_state_excursion != 0.02",
    )
    _require(
        int(waiting_config.get("stride_frames", -1)) == EXPECTED_STRIDE_FRAMES,
        "manifest stride_frames != 15",
    )
    waiting_audit_path = Path(str(waiting_config.get("source_0816", ""))).resolve()
    _require(waiting_audit_path.is_file(), f"0816 waiting audit does not exist: {waiting_audit_path}")
    waiting_audit = json.loads(waiting_audit_path.read_text())
    _require(
        waiting_audit.get("schema_version") == "openpi.0816_memory_label_audit.v1",
        "0816 waiting audit schema_version drifted",
    )
    audit_static_config = waiting_audit.get("static_core_config", {})
    expected_audit_static_config = {
        "state_dimensions": 14,
        "max_abs_step": EXPECTED_MAX_STATE_STEP,
        "max_abs_step_comparator": "strictly less than",
        "max_per_dimension_excursion": EXPECTED_MAX_STATE_EXCURSION,
        "max_excursion_comparator": "less than or equal",
        "memory_stride_frames": EXPECTED_STRIDE_FRAMES,
    }
    for field, expected_value in expected_audit_static_config.items():
        _require(
            audit_static_config.get(field) == expected_value,
            f"0816 waiting audit static_core_config.{field} drifted",
        )
    waiting_audit_by_id = {
        f"{episode['source']}/{episode['demo_name']}": episode for episode in waiting_audit.get("episodes", [])
    }
    _require(len(waiting_audit_by_id) == 60, "0816 waiting audit does not contain 60 unique episodes")

    meta_dir = dataset / "meta"
    required_meta = {
        "info": meta_dir / "info.json",
        "tasks": meta_dir / "tasks.jsonl",
        "episodes": meta_dir / "episodes.jsonl",
        "episodes_stats": meta_dir / "episodes_stats.jsonl",
        "prompts": meta_dir / "episode_prompts.json",
        "sources": meta_dir / "episode_sources.json",
        "waiting_cores": meta_dir / "memory_waiting_cores.json",
        "conversion_report": meta_dir / "conversion_report.json",
    }
    for name, path in required_meta.items():
        _require(path.is_file(), f"dataset is incomplete: missing {name} metadata at {path}")
    _require(
        {path.name for path in meta_dir.iterdir() if path.is_file()} == {path.name for path in required_meta.values()},
        "dataset meta file inventory contains missing or unexpected files",
    )
    meta_hashes_before = {name: _sha256(path) for name, path in required_meta.items()}
    info = json.loads(required_meta["info"].read_text())
    tasks = _jsonl(required_meta["tasks"])
    episodes_meta = _jsonl(required_meta["episodes"])
    episodes_stats = _jsonl(required_meta["episodes_stats"])
    prompts = json.loads(required_meta["prompts"].read_text())
    sources = json.loads(required_meta["sources"].read_text())
    waiting_cores = json.loads(required_meta["waiting_cores"].read_text())
    conversion_report = json.loads(required_meta["conversion_report"].read_text())

    _validate_info(info)
    expected_indices = {str(index) for index in range(EXPECTED_EPISODES)}
    for row_index, row in enumerate(tasks):
        _require(set(row) == {"task_index", "task"}, f"tasks.jsonl row {row_index} fields drifted")
    _require(
        [(row.get("task_index"), row.get("task")) for row in tasks] == list(enumerate(vocabulary)),
        "tasks.jsonl does not exactly match the pinned task vocabulary",
    )
    _require(len(episodes_meta) == EXPECTED_EPISODES, "episodes.jsonl does not contain 90 rows")
    _require(len(episodes_stats) == EXPECTED_EPISODES, "episodes_stats.jsonl does not contain 90 rows")
    for row_index, row in enumerate(episodes_stats):
        _require(
            set(row) == {"episode_index", "stats"},
            f"episodes_stats.jsonl row {row_index} fields drifted",
        )
    stats_by_episode = {int(row["episode_index"]): row["stats"] for row in episodes_stats}
    _require(
        len(stats_by_episode) == EXPECTED_EPISODES and set(stats_by_episode) == set(range(EXPECTED_EPISODES)),
        "episodes_stats.jsonl has duplicate or missing episode indices",
    )
    _require(set(prompts) == expected_indices, "episode_prompts.json keys are not exactly 0..89")
    _require(set(sources) == expected_indices, "episode_sources.json keys are not exactly 0..89")
    _require(set(waiting_cores) == expected_indices, "memory_waiting_cores.json keys are not exactly 0..89")
    _require(
        set(conversion_report)
        == {
            "schema_version",
            "repo_name",
            "episode_manifest",
            "episode_manifest_sha256",
            "review_status",
            "included_episodes",
            "included_frames",
            "excluded_episodes",
            "task_vocabulary",
        },
        f"conversion_report fields drifted: {sorted(conversion_report)}",
    )
    _require(conversion_report.get("schema_version") == 1, "conversion_report schema_version != 1")
    _require(
        conversion_report.get("episode_manifest_sha256") == manifest_sha256,
        "conversion_report manifest SHA does not match the supplied manifest",
    )
    _require(
        Path(str(conversion_report.get("episode_manifest", ""))).resolve() == manifest_path,
        "conversion_report points to a different manifest path",
    )
    _require(
        int(conversion_report.get("included_episodes", -1)) == EXPECTED_EPISODES,
        "conversion_report included_episodes != 90",
    )
    _require(
        int(conversion_report.get("included_frames", -1)) == EXPECTED_FRAMES,
        "conversion_report included_frames != 77445",
    )
    _require(
        tuple(conversion_report.get("task_vocabulary", ())) == EXPECTED_TASK_VOCABULARY,
        "conversion_report task vocabulary drifted",
    )
    _require(conversion_report.get("excluded_episodes") == excluded, "conversion_report excluded episodes drifted")
    _require(
        conversion_report.get("review_status") == manifest.get("review_status"),
        "conversion_report review_status differs from manifest",
    )
    expected_repo_name = f"{dataset.parent.name}/{dataset.name}"
    _require(
        conversion_report.get("repo_name") == expected_repo_name,
        f"conversion_report repo_name {conversion_report.get('repo_name')!r} != {expected_repo_name!r}",
    )

    parquet_files = list(dataset.glob("data/chunk-*/episode_*.parquet"))
    _require(len(parquet_files) == EXPECTED_EPISODES, f"expected 90 parquet files, found {len(parquet_files)}")

    total_rows = 0
    prompt_counts: dict[str, int] = {}
    side_counts = {"left": 0, "right": 0}
    collection_counts: dict[str, int] = {}
    task_frame_counts = dict.fromkeys(vocabulary, 0)
    eligible_count = 0
    episode_results: list[dict[str, Any]] = []
    decoded_image_samples = 0
    numeric_digest = hashlib.sha256()
    sampled_image_digest = hashlib.sha256()
    raw_source_digest = hashlib.sha256()
    global_index_offset = 0
    for episode_index, (manifest_position, spec) in enumerate(included_with_positions):
        key = str(episode_index)
        stable_id = str(spec["stable_id"])
        _require(spec.get("raw_dir") == f"data/{stable_id}", f"{stable_id}: manifest raw_dir is not canonical")
        expected_label_file = (
            "subtask_labels_v35_staticwait.json" if stable_id.startswith("0816_") else "subtask_labels.json"
        )
        _require(
            spec.get("label_file") == expected_label_file,
            f"{stable_id}: label_file={spec.get('label_file')!r}, expected {expected_label_file!r}",
        )
        raw_dir = _resolve_raw_dir(manifest_path, manifest, spec["raw_dir"])
        label_path = raw_dir / spec.get("label_file", "subtask_labels.json")
        num_frames = int(spec["expected_num_frames"])
        _require(num_frames > 0, f"{stable_id}: expected_num_frames must be positive")
        _require(label_path.is_file(), f"{stable_id}: missing label file {label_path}")
        label_sha256 = _sha256(label_path)
        if approved:
            _require(
                spec.get("label_sha256") == label_sha256,
                f"{stable_id}: approved label SHA256 differs from current label",
            )
        source = sources[key]
        expected_source_fields = {
            "stable_id": stable_id,
            "manifest_raw_dir": str(spec["raw_dir"]),
            "manifest_position": manifest_position,
            "instruction": spec["instruction"],
            "target_side": spec["target_side"],
            "collection": spec["collection"],
            "raw_demo": raw_dir.name,
            "metadata_counter": spec.get("metadata_counter"),
            "timestamp": spec.get("timestamp"),
            "num_frames": num_frames,
            "label_file": spec.get("label_file", "subtask_labels.json"),
            "label_sha256": label_sha256,
        }
        _require(
            set(source)
            == {
                *expected_source_fields,
                "raw_dir",
                "stream_lengths",
                "memory_waiting_core",
            },
            f"{stable_id}: episode_sources fields drifted: {sorted(source)}",
        )
        for field, expected_value in expected_source_fields.items():
            _require(
                source.get(field) == expected_value,
                f"{stable_id}: episode_sources.{field}={source.get(field)!r}, expected {expected_value!r}",
            )
        _require(Path(source.get("raw_dir", "")).resolve() == raw_dir, f"{stable_id}: source raw_dir drifted")
        _require(prompts[key] == spec["instruction"], f"{stable_id}: prompt sidecar differs from manifest")
        episode_meta = episodes_meta[episode_index]
        _require(
            set(episode_meta) == {"episode_index", "tasks", "length"},
            f"{stable_id}: episodes.jsonl fields drifted: {sorted(episode_meta)}",
        )
        _require(episode_meta.get("episode_index") == episode_index, f"{stable_id}: episodes.jsonl index drifted")
        _require(episode_meta.get("length") == num_frames, f"{stable_id}: episodes.jsonl length drifted")

        parquet_path = _parquet_path(dataset, episode_index)
        parquet = pq.ParquetFile(parquet_path)
        _verify_parquet_schema(parquet, stable_id)
        _verify_image_metadata(parquet, stable_id)
        _require(parquet.metadata.num_rows == num_frames, f"{stable_id}: parquet metadata row count drifted")
        table = pq.read_table(parquet_path, columns=list(NUMERIC_COLUMNS))
        _require(table.num_rows == num_frames, f"{stable_id}: parquet has {table.num_rows} rows, expected {num_frames}")
        for column_name in NUMERIC_COLUMNS:
            _require(table[column_name].null_count == 0, f"{stable_id}: {column_name} contains nulls")
        parquet_episode = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        global_index = np.asarray(table["index"].to_numpy(), dtype=np.int64)
        timestamp = np.asarray(table["timestamp"].to_numpy(), dtype=np.float32)
        task_index = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
        state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        actions = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        _require(state.shape == (num_frames, 14), f"{stable_id}: state shape is {state.shape}")
        _require(actions.shape == (num_frames, 14), f"{stable_id}: actions shape is {actions.shape}")
        _require(np.all(np.isfinite(state)), f"{stable_id}: state contains non-finite values")
        _require(np.all(np.isfinite(actions)), f"{stable_id}: actions contain non-finite values")
        _require(np.all(parquet_episode == episode_index), f"{stable_id}: wrong episode_index in parquet")
        _require(
            np.array_equal(frame_index, np.arange(num_frames, dtype=np.int64)),
            f"{stable_id}: frame_index is not 0..{num_frames - 1}",
        )
        expected_global_index = np.arange(global_index_offset, global_index_offset + num_frames, dtype=np.int64)
        _require(
            np.array_equal(global_index, expected_global_index),
            f"{stable_id}: global index does not start at {global_index_offset}",
        )
        expected_timestamp = np.arange(num_frames, dtype=np.float32) / np.float32(EXPECTED_FPS)
        _require(np.array_equal(timestamp, expected_timestamp), f"{stable_id}: timestamp != frame_index / 30")
        segments, expected_tasks = _validate_five_phase_labels(
            label_path,
            num_frames,
            str(spec["target_side"]),
            task_to_index,
        )
        if not np.array_equal(task_index, expected_tasks):
            mismatch = int(np.flatnonzero(task_index != expected_tasks)[0])
            raise AssertionError(f"{stable_id}: task_index mismatch at frame {mismatch}")
        expected_episode_tasks = {segment["task"] for segment in segments}
        actual_episode_tasks = episode_meta.get("tasks")
        _require(
            isinstance(actual_episode_tasks, list) and len(actual_episode_tasks) == len(set(actual_episode_tasks)),
            f"{stable_id}: episodes.jsonl tasks must be a duplicate-free list",
        )
        _require(
            set(actual_episode_tasks) == expected_episode_tasks,
            f"{stable_id}: episodes.jsonl tasks {actual_episode_tasks} != {sorted(expected_episode_tasks)}",
        )

        raw_arrays: dict[str, np.ndarray] = {}
        actual_stream_lengths: dict[str, int] = {}
        raw_stream_sha256: dict[str, str] = {}
        raw_signatures_before: dict[str, tuple[int, int]] = {}
        for filename in (*ARRAY_STREAMS, *VIDEO_TO_IMAGE_COLUMN):
            path = raw_dir / filename
            _require(path.is_file(), f"{stable_id}: missing raw stream {path}")
            stat = path.stat()
            raw_signatures_before[filename] = (stat.st_size, stat.st_mtime_ns)
            raw_stream_sha256[filename] = _sha256(path)
            raw_source_digest.update(stable_id.encode())
            raw_source_digest.update(filename.encode())
            raw_source_digest.update(raw_stream_sha256[filename].encode())
        if approved:
            _require(
                spec.get("raw_stream_sha256") == raw_stream_sha256,
                f"{stable_id}: approved raw stream SHA256 inventory differs from current raw data",
            )
        for filename in ARRAY_STREAMS:
            array = np.load(raw_dir / filename, mmap_mode="r")
            _require(array.ndim == 2 and array.shape[1] == 7, f"{stable_id}: {filename} shape is {array.shape}")
            actual_stream_lengths[filename] = len(array)
            raw_arrays[filename] = np.asarray(array[:num_frames])
            _require(len(array) >= num_frames, f"{stable_id}: {filename} is shorter than {num_frames}")
            _require(np.all(np.isfinite(raw_arrays[filename])), f"{stable_id}: {filename} contains non-finite values")
        raw_state_native = np.concatenate(
            [raw_arrays["left_joint_positions.npy"], raw_arrays["right_joint_positions.npy"]], axis=1
        )
        raw_state = raw_state_native.astype(np.float32)
        raw_actions = np.concatenate([raw_arrays["left_control.npy"], raw_arrays["right_control.npy"]], axis=1).astype(
            np.float32
        )
        _require(np.array_equal(state, raw_state), f"{stable_id}: state differs from raw source")
        _require(np.array_equal(actions, raw_actions), f"{stable_id}: actions differ from raw source")

        core = waiting_cores[key]
        manifest_core = spec["memory_waiting_core"]
        _require(
            set(core)
            == {
                "start",
                "end",
                "length",
                "max_14d_step",
                "max_14d_excursion",
                "eligible_at_stride",
                "stride_frames",
            },
            f"{stable_id}: memory_waiting_cores fields drifted: {sorted(core)}",
        )
        _require(
            set(manifest_core) == {"start", "end", "eligible_at_stride"},
            f"{stable_id}: manifest memory_waiting_core fields drifted: {sorted(manifest_core)}",
        )
        _require(source.get("memory_waiting_core") == core, f"{stable_id}: source/core sidecars disagree")
        core_start = int(core.get("start", -1))
        core_end = int(core.get("end", -1))
        _require(core_start == int(manifest_core["start"]), f"{stable_id}: core start differs from manifest")
        _require(core_end == int(manifest_core["end"]), f"{stable_id}: core end differs from manifest")
        _require(0 <= core_start <= core_end < num_frames, f"{stable_id}: invalid core {core_start}..{core_end}")
        core_length = core_end - core_start + 1
        eligible = core_length >= EXPECTED_STRIDE_FRAMES
        _require(core.get("length") == core_length, f"{stable_id}: core length field is wrong")
        _require(core.get("stride_frames") == EXPECTED_STRIDE_FRAMES, f"{stable_id}: core stride field != 15")
        _require(core.get("eligible_at_stride") is eligible, f"{stable_id}: core eligibility is wrong")
        _require(
            manifest_core.get("eligible_at_stride") is eligible,
            f"{stable_id}: manifest core eligibility is wrong",
        )
        wait_task_index = task_to_index[f"wait; target bin is {spec['target_side']}"]
        wait_frames = np.flatnonzero(task_index == wait_task_index)
        _require(
            np.array_equal(wait_frames, np.arange(core_start, core_end + 1)),
            f"{stable_id}: semantic wait does not exactly equal strict core",
        )
        _require(core_length >= 2, f"{stable_id}: strict core has fewer than two frames")
        core_state = raw_state_native[core_start : core_end + 1]
        max_step = float(np.abs(np.diff(core_state, axis=0)).max())
        max_excursion = float((core_state.max(axis=0) - core_state.min(axis=0)).max())
        _require(max_step < EXPECTED_MAX_STATE_STEP, f"{stable_id}: core step {max_step} >= 0.004")
        _require(max_excursion <= EXPECTED_MAX_STATE_EXCURSION, f"{stable_id}: core excursion {max_excursion} > 0.02")
        _require(
            math.isclose(float(core.get("max_14d_step", math.inf)), max_step, rel_tol=0, abs_tol=1e-7),
            f"{stable_id}: core max_14d_step field differs from recomputation",
        )
        _require(
            math.isclose(float(core.get("max_14d_excursion", math.inf)), max_excursion, rel_tol=0, abs_tol=1e-7),
            f"{stable_id}: core max_14d_excursion field differs from recomputation",
        )
        if episode_index < 60:
            audited = waiting_audit_by_id.get(stable_id)
            _require(audited is not None, f"{stable_id}: missing from the pinned 0816 waiting audit")
            audited_core = audited["strict_static_core_14d"]
            _require(
                set(audited_core) == {"start", "end", "length"},
                f"{stable_id}: pinned audit strict core fields drifted",
            )
            _require(
                (core_start, core_end) == (int(audited_core["start"]), int(audited_core["end"])),
                f"{stable_id}: manifest core differs from the pinned 0816 audit",
            )
            _require(audited_core["length"] == core_length, f"{stable_id}: pinned audit core length is wrong")

        sample_indices = sorted({0, core_start, core_end, num_frames - 1})
        image_count, decoded_video_lengths = _verify_image_samples(
            parquet,
            raw_dir,
            sample_indices,
            source.get("stream_lengths", {}),
            stable_id,
            sampled_image_digest,
        )
        decoded_image_samples += image_count
        actual_stream_lengths.update(decoded_video_lengths)
        _require(
            source.get("stream_lengths") == actual_stream_lengths,
            f"{stable_id}: provenance stream_lengths={source.get('stream_lengths')} != {actual_stream_lengths}",
        )
        _require(min(actual_stream_lengths.values()) == num_frames, f"{stable_id}: shortest raw stream != {num_frames}")
        for filename, signature in raw_signatures_before.items():
            stat = (raw_dir / filename).stat()
            _require(
                (stat.st_size, stat.st_mtime_ns) == signature,
                f"{stable_id}: raw stream changed during verification: {filename}",
            )
            _require(
                _sha256(raw_dir / filename) == raw_stream_sha256[filename],
                f"{stable_id}: raw stream byte hash changed during verification: {filename}",
            )
        _require(_sha256(label_path) == label_sha256, f"{stable_id}: label changed during verification")

        values_by_name = {
            "state": state,
            "actions": actions,
            "timestamp": timestamp[:, None],
            "frame_index": frame_index[:, None],
            "episode_index": parquet_episode[:, None],
            "index": global_index[:, None],
            "task_index": task_index[:, None],
        }
        episode_stats = stats_by_episode[episode_index]
        _require(
            set(episode_stats) == {*NUMERIC_COLUMNS, *IMAGE_COLUMNS},
            f"{stable_id}: episodes_stats feature fields drifted: {sorted(episode_stats)}",
        )
        _validate_numeric_stats(episode_stats, values_by_name, num_frames, stable_id)
        _validate_image_stats(episode_stats, num_frames, stable_id)

        prompt_counts[spec["instruction"]] = prompt_counts.get(spec["instruction"], 0) + 1
        side_counts[spec["target_side"]] += 1
        collection = spec["collection"]
        collection_counts[collection] = collection_counts.get(collection, 0) + 1
        for task, index in task_to_index.items():
            task_frame_counts[task] += int(np.count_nonzero(task_index == index))
        eligible_count += int(eligible)
        total_rows += num_frames
        numeric_digest.update(state.tobytes())
        numeric_digest.update(actions.tobytes())
        numeric_digest.update(task_index.tobytes())
        numeric_digest.update(global_index.tobytes())
        numeric_digest.update(timestamp.tobytes())
        episode_results.append(
            {
                "episode_index": episode_index,
                "stable_id": stable_id,
                "frames": num_frames,
                "instruction": spec["instruction"],
                "target_side": spec["target_side"],
                "waiting_core": core,
                "image_sample_frames": sample_indices,
                "image_samples_verified": image_count,
                "label_sha256": label_sha256,
                "raw_stream_sha256": raw_stream_sha256,
            }
        )
        global_index_offset += num_frames

    _require(total_rows == EXPECTED_FRAMES, f"verified rows {total_rows} != 77445")
    _require(global_index_offset == EXPECTED_FRAMES, "global index final offset != 77445")
    _require(prompt_counts == EXPECTED_PROMPT_COUNTS, f"verified prompt counts drifted: {prompt_counts}")
    _require(side_counts == EXPECTED_SIDE_COUNTS, f"verified side counts drifted: {side_counts}")
    _require(
        collection_counts == EXPECTED_COLLECTION_COUNTS,
        f"verified collection counts drifted: {collection_counts}",
    )
    _require(sum(task_frame_counts.values()) == EXPECTED_FRAMES, "task frame counts do not sum to 77445")
    _require(eligible_count == EXPECTED_ELIGIBLE_AT_STRIDE, f"eligible core count {eligible_count} != 89")
    _require(
        decoded_image_samples == EXPECTED_IMAGE_SAMPLES,
        f"verified {decoded_image_samples} boundary images, expected 1080",
    )
    for key, stable_id in expected["episode_index_to_stable_id"].items():
        _require(sources[key]["stable_id"] == stable_id, f"episode identity anchor {key} drifted")
    _require(_sha256(manifest_path) == manifest_sha256, "manifest changed during verification")
    meta_hashes_after = {name: _sha256(path) for name, path in required_meta.items()}
    _require(meta_hashes_after == meta_hashes_before, "dataset metadata changed during verification")
    return {
        "schema_version": 2,
        "status": "passed",
        "dataset": str(dataset),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "waiting_audit_0816": str(waiting_audit_path),
        "waiting_audit_0816_sha256": _sha256(waiting_audit_path),
        "episodes": len(included),
        "frames": total_rows,
        "tasks": len(vocabulary),
        "prompts": prompt_counts,
        "target_sides": side_counts,
        "collections": collection_counts,
        "task_frame_counts": task_frame_counts,
        "memory_waiting_eligible_at_stride_15": eligible_count,
        "memory_waiting_ineligible_at_stride_15": len(included) - eligible_count,
        "numeric_state_action_task_sha256": numeric_digest.hexdigest(),
        "raw_source_inventory_sha256": raw_source_digest.hexdigest(),
        "sampled_image_pixels_sha256": sampled_image_digest.hexdigest(),
        "decoded_image_samples": decoded_image_samples,
        "image_sample_policy": "all 90 episodes x [0, core.start, core.end, last] x 3 cameras; exact raw RGB parity",
        "self_checks_passed": self_checks,
        "meta_sha256": {path.name: _sha256(path) for path in sorted(meta_dir.iterdir()) if path.is_file()},
        "episode_results": episode_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("diagnostic_outputs/0816_0830_dataset_v35_verification.json"),
    )
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    if args.self_check_only:
        print(json.dumps({"status": "passed", "self_checks_passed": _run_self_checks()}, indent=2))
        return
    if args.dataset is None or args.manifest is None:
        parser.error("--dataset and --manifest are required unless --self-check-only is used")
    result = verify(args.dataset.resolve(), args.manifest.resolve())
    output = args.output.resolve()
    _atomic_json(output, result)
    print(f"PASS: {result['episodes']} episodes / {result['frames']} frames / {result['tasks']} tasks; report={output}")


if __name__ == "__main__":
    main()
