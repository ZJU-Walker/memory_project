"""Fast offline replay for spatial v3.1 memory-writer contribution diagnostics.

This runner is intentionally separate from the causal Tests 1--4 framework.  It answers a
different, local question: for each top-camera patch token in one scheduled frame write, how large
is its individual associative error and its individual fast-weight gradient norm?  Replay performs
one VLM prefill/read/memory-token append per sampled frame, measures the 256 contributions against
the pre-write state, then commits the normal *single frame-level* memory update.  It never decodes a
subtask and never runs action denoising.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import csv
import dataclasses
import json
import math
from pathlib import Path
import time
from typing import Any

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from openpi.diagnostics import token_heatmap
from openpi.diagnostics import v31
from openpi.diagnostics import v31_pi0
from openpi.models import model as _model
from openpi.shared import nnx_utils
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
import openpi.transforms as _transforms

SCHEMA_VERSION = "openpi.v31.writer_contribution.v1"
SUPPORTED_METRICS = (
    "token_error",
    "token_grad_norm",
    "token_mean_loss_grad_norm",
)
DEFAULT_METRICS = ("token_grad_norm", "token_error")
WRITER_VIDEO_ENCODER = "opencv"


@dataclasses.dataclass(frozen=True)
class EpisodeSource:
    episode_id: str
    path: Path
    control_hz: float
    ground_truth_side: str | None = None
    source_format: str = "raw_yam"
    task_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id must be non-empty")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0:
            raise ValueError("control_hz must be finite and positive")
        if self.ground_truth_side not in (None, "left", "right"):
            raise ValueError("ground_truth_side must be left, right, or absent")
        if self.source_format not in ("raw_yam", "lerobot_inline_parquet"):
            raise ValueError(f"unsupported episode source_format: {self.source_format!r}")
        object.__setattr__(self, "path", self.path.expanduser().resolve())
        object.__setattr__(self, "task_names", tuple(self.task_names))


@dataclasses.dataclass(frozen=True)
class RunOptions:
    checkpoint: Path
    output_dir: Path
    episode_paths: tuple[Path, ...] = ()
    episode_manifest: Path | None = None
    dataset_root: Path | None = None
    episode_indices: tuple[int, ...] = ()
    config: str = "pi05_yam_mem_v31"
    stride: int | None = None
    metrics: tuple[str, ...] = DEFAULT_METRICS
    video_fps: float | None = None
    max_frames: int | None = None
    render_video: bool = True
    colormap: str = "inferno"
    alpha: float = 0.58
    upper_percentile: float = 99.0
    # Absolute errors/grad norms decay by orders of magnitude within an episode (the blank-memory
    # transient dominates a fixed video scale), so by default render each metric both ways: the
    # absolute video-wide scale and a per-frame z-score view that keeps late-episode structure
    # visible. The z-score view uses a diverging colormap centered on the frame mean.
    heatmap_scale_modes: tuple[str, ...] = ("video", "per_frame_zscore", "per_token_history")
    zscore_range: float = 3.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", self.checkpoint.expanduser().resolve())
        object.__setattr__(self, "output_dir", self.output_dir.expanduser().resolve())
        object.__setattr__(self, "episode_paths", tuple(path.expanduser().resolve() for path in self.episode_paths))
        object.__setattr__(
            self,
            "episode_manifest",
            None if self.episode_manifest is None else self.episode_manifest.expanduser().resolve(),
        )
        object.__setattr__(
            self,
            "dataset_root",
            None if self.dataset_root is None else self.dataset_root.expanduser().resolve(),
        )
        object.__setattr__(self, "episode_indices", tuple(self.episode_indices))
        source_count = (
            int(bool(self.episode_paths)) + int(self.episode_manifest is not None) + int(self.dataset_root is not None)
        )
        if source_count != 1:
            raise ValueError(
                "provide exactly one source: --episode (repeatable), --episode-manifest, or --dataset-root"
            )
        if self.dataset_root is not None:
            if not self.episode_indices:
                raise ValueError("--dataset-root requires a non-empty --episode-indices list")
            if any(isinstance(index, bool) or index < 0 for index in self.episode_indices):
                raise ValueError("episode indices must be non-negative integers")
            if len(set(self.episode_indices)) != len(self.episode_indices):
                raise ValueError("episode indices must not contain duplicates")
        elif self.episode_indices:
            raise ValueError("episode_indices are only valid with dataset_root")
        if not self.config:
            raise ValueError("config must be non-empty")
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.video_fps is not None and (not math.isfinite(self.video_fps) or self.video_fps <= 0):
            raise ValueError("video_fps must be finite and positive")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if not self.metrics or len(set(self.metrics)) != len(self.metrics):
            raise ValueError("metrics must be non-empty and contain no duplicates")
        unknown = set(self.metrics) - set(SUPPORTED_METRICS)
        if unknown:
            raise ValueError(f"unsupported metrics: {sorted(unknown)}")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if not 0.0 < self.upper_percentile <= 100.0:
            raise ValueError("upper_percentile must lie in (0, 100]")
        object.__setattr__(self, "heatmap_scale_modes", tuple(self.heatmap_scale_modes))
        if not self.heatmap_scale_modes or len(set(self.heatmap_scale_modes)) != len(self.heatmap_scale_modes):
            raise ValueError("heatmap_scale_modes must be non-empty and contain no duplicates")
        unknown_modes = set(self.heatmap_scale_modes) - set(token_heatmap.SCALE_MODES)
        if unknown_modes:
            raise ValueError(f"unsupported heatmap scale modes: {sorted(unknown_modes)}")
        if not math.isfinite(self.zscore_range) or self.zscore_range <= 0:
            raise ValueError("zscore_range must be finite and positive")


def _strict_json(path: Path) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"{path} contains duplicate JSON key {key!r}")
            output[key] = value
        return output

    def reject_nonfinite(value: str) -> None:
        raise ValueError(f"{path} contains non-finite JSON number {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite_positive(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _direct_episode_source(path: Path) -> EpisodeSource:
    metadata_path = path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"raw episode lacks metadata.json (needed for control_hz): {path}")
    metadata = _strict_json(metadata_path)
    hz = metadata.get("hz")
    if hz is None:
        raise ValueError(f"raw episode metadata lacks hz: {metadata_path}")
    return EpisodeSource(path.name, path, _finite_positive(hz, name=f"{metadata_path} hz"))


def load_episode_sources(options: RunOptions) -> tuple[EpisodeSource, ...]:
    """Load either repeatable raw directories or the existing strict ``yam-eval-v1`` manifest."""

    if options.episode_paths:
        sources = tuple(_direct_episode_source(path) for path in options.episode_paths)
    elif options.episode_manifest is not None:
        document = _strict_json(options.episode_manifest)
        if document.get("version") != "yam-eval-v1":
            raise ValueError("episode manifest version must be 'yam-eval-v1'")
        control_hz = _finite_positive(document.get("control_hz"), name="episode manifest control_hz")
        entries = document.get("episodes")
        if not isinstance(entries, list) or not entries:
            raise ValueError("episode manifest must contain a non-empty episodes list")
        parsed = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                raise TypeError(f"episode manifest entry {index} must be an object")
            episode_id = entry.get("episode_id")
            path = entry.get("path")
            side = entry.get("ground_truth_side")
            if not isinstance(episode_id, str) or not episode_id.strip():
                raise TypeError(f"episode manifest entry {index} has invalid episode_id")
            if not isinstance(path, str) or not path.strip():
                raise TypeError(f"episode manifest entry {index} has invalid path")
            parsed.append(EpisodeSource(episode_id, Path(path), control_hz, side))
        sources = tuple(parsed)
    else:
        assert options.dataset_root is not None
        sources = _load_lerobot_sources(options.dataset_root, options.episode_indices)
    ids = [source.episode_id for source in sources]
    if len(set(ids)) != len(ids):
        raise ValueError("episode IDs must be unique")
    return sources


def _read_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(value)
    return tuple(records)


def _load_lerobot_sources(dataset_root: Path, episode_indices: Sequence[int]) -> tuple[EpisodeSource, ...]:
    """Resolve v2.1 LeRobot episodes whose three PNG images are inline parquet structs."""

    info_path = dataset_root / "meta" / "info.json"
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    for path in (info_path, tasks_path, episodes_path):
        if not path.is_file():
            raise FileNotFoundError(f"LeRobot dataset is missing metadata: {path}")
    info = _strict_json(info_path)
    fps = _finite_positive(info.get("fps"), name="LeRobot info fps")
    total_episodes = info.get("total_episodes")
    chunks_size = info.get("chunks_size")
    data_path = info.get("data_path")
    if isinstance(total_episodes, bool) or not isinstance(total_episodes, int) or total_episodes <= 0:
        raise ValueError("LeRobot total_episodes must be a positive integer")
    if isinstance(chunks_size, bool) or not isinstance(chunks_size, int) or chunks_size <= 0:
        raise ValueError("LeRobot chunks_size must be a positive integer")
    if not isinstance(data_path, str) or not data_path:
        raise ValueError("LeRobot info data_path must be a non-empty format string")
    features = info.get("features")
    if not isinstance(features, Mapping):
        raise ValueError("LeRobot info must define features")
    for name in ("image", "left_wrist_image", "right_wrist_image"):
        feature = features.get(name)
        if not isinstance(feature, Mapping) or feature.get("dtype") != "image":
            raise ValueError(f"LeRobot feature {name!r} must be an inline image feature")

    tasks = {}
    for record in _read_jsonl(tasks_path):
        index, name = record.get("task_index"), record.get("task")
        if isinstance(index, bool) or not isinstance(index, int) or not isinstance(name, str) or not name:
            raise ValueError(f"invalid LeRobot task record in {tasks_path}")
        if index in tasks:
            raise ValueError(f"duplicate LeRobot task_index {index}")
        tasks[index] = name
    if not tasks or sorted(tasks) != list(range(max(tasks) + 1)):
        raise ValueError("LeRobot task indices must be contiguous from zero")
    task_names = tuple(tasks[index] for index in range(len(tasks)))

    episode_metadata = {}
    for record in _read_jsonl(episodes_path):
        index = record.get("episode_index")
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError(f"invalid episode_index in {episodes_path}")
        if index in episode_metadata:
            raise ValueError(f"duplicate episode_index {index} in {episodes_path}")
        episode_metadata[index] = record

    sources = []
    for index in episode_indices:
        if index < 0 or index >= total_episodes or index not in episode_metadata:
            raise ValueError(f"LeRobot episode index {index} is outside available metadata [0, {total_episodes})")
        try:
            relative = data_path.format(episode_chunk=index // chunks_size, episode_index=index)
        except (KeyError, ValueError) as exc:
            raise ValueError(f"could not format LeRobot data_path {data_path!r}") from exc
        parquet = dataset_root / relative
        if not parquet.is_file():
            raise FileNotFoundError(f"LeRobot episode parquet is missing: {parquet}")
        listed_tasks = episode_metadata[index].get("tasks")
        side = None
        if isinstance(listed_tasks, list):
            open_sides = {
                name.split()[1]
                for name in listed_tasks
                if isinstance(name, str) and name in ("open left bin", "open right bin")
            }
            if len(open_sides) == 1:
                side = open_sides.pop()
        sources.append(
            EpisodeSource(
                episode_id=f"episode_{index:06d}",
                path=parquet,
                control_hz=fps,
                ground_truth_side=side,
                source_format="lerobot_inline_parquet",
                task_names=task_names,
            )
        )
    return tuple(sources)


def _required_episode_files(source: EpisodeSource) -> dict[str, Path]:
    files = {
        "top": source.path / "top_camera_rgb.mp4",
        "left": source.path / "left_camera_rgb.mp4",
        "right": source.path / "right_camera_rgb.mp4",
        "left_state": source.path / "left_joint_positions.npy",
        "right_state": source.path / "right_joint_positions.npy",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"episode {source.episode_id!r} is missing files: {missing}")
    return files


def _video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count <= 0:
        raise ValueError(f"could not determine video frame count: {path}")
    return count


def _load_states(files: Mapping[str, Path], episode_id: str) -> np.ndarray:
    left = np.load(files["left_state"], allow_pickle=False)
    right = np.load(files["right_state"], allow_pickle=False)
    for side, values in (("left", left), ("right", right)):
        if values.ndim != 2 or values.shape[1] != 7:
            raise ValueError(f"episode {episode_id!r} {side} state must have shape [T, 7], got {values.shape}")
        if values.dtype.kind not in "fiu" or not np.all(np.isfinite(values)):
            raise ValueError(f"episode {episode_id!r} {side} state must be finite numeric data")
    if len(left) != len(right):
        raise ValueError(f"episode {episode_id!r} left/right state lengths differ")
    return np.concatenate([left, right], axis=1).astype(np.float32)


def _safe_name(value: str) -> str:
    result = "".join(character if character.isalnum() or character in "._-" else "_" for character in value)
    result = result.strip("._")
    if not result:
        raise ValueError(f"cannot form a safe filename from {value!r}")
    return result[:120]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON values must be finite")
        return value
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(item) for item in value]
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclasses.dataclass
class _MeasuredFrame:
    raw_frame: int
    policy_step: int
    model_image_rgb: np.ndarray
    raw_image_rgb: np.ndarray
    metrics: dict[str, np.ndarray]
    scalar: dict[str, float]
    phase: str = ""


class WriterContributionRunner:
    """Checkpoint-backed, single-pass raw-YAM replay."""

    def __init__(self, options: RunOptions):
        self.options = options
        self.sources = load_episode_sources(options)
        if options.output_dir.exists():
            raise FileExistsError(f"refusing to overwrite output directory: {options.output_dir}")
        if not (options.checkpoint / "params").is_dir():
            raise FileNotFoundError(f"checkpoint has no params item: {options.checkpoint}")

        self.train_config = _config.get_config(options.config)
        if not self.train_config.model.predict_with_memory:
            raise ValueError("writer contribution diagnostics require a memory-enabled config")
        if self.train_config.model.memory_write_source != "post_attention":
            raise ValueError("v3.1 writer contribution diagnostics require memory_write_source='post_attention'")
        self.provenance = v31_pi0._read_checkpoint_static_provenance(  # noqa: SLF001
            options.checkpoint,
            expected_config_name=options.config,
        )
        data_config = self.train_config.data.create(self.train_config.assets_dirs, self.train_config.model)
        configured_stride = int(data_config.memory_stride_frames)
        self.stride = configured_stride if options.stride is None else options.stride
        if self.stride <= 0:
            raise ValueError("effective stride must be positive")
        if data_config.asset_id is None:
            raise ValueError("YAM writer diagnostics require a normalization-stat asset_id")
        norm_stats = _normalize.load(options.checkpoint / "assets" / data_config.asset_id)
        self.model = self.train_config.model.load(
            _model.restore_params(options.checkpoint / "params", dtype=jnp.float32)
        )
        input_transforms = [
            transform
            for transform in data_config.data_transforms.inputs
            if not isinstance(transform, _transforms.BuildMemorySequence)
        ]
        self.input_transform = _transforms.compose(
            [
                *input_transforms,
                _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )
        self._step = nnx_utils.module_jit(
            self.model.writer_contribution_step,
            static_argnames=("allow_write",),
        )
        self.configured_stride = configured_stride

    @staticmethod
    def _progress_detail(frame) -> str:
        """Per-frame progress text. Subclasses measuring other quantities override this."""
        return f"error={frame.scalar['surprise']:.5g} aggregate_grad={frame.scalar['grad_norm']:.5g}"

    def _transform_observation(
        self,
        source: EpisodeSource,
        raw_frame: int,
        top_rgb: np.ndarray,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        state: np.ndarray,
    ) -> tuple[_model.Observation, np.ndarray]:
        transformed = self.input_transform(
            {
                "observation/image": top_rgb,
                "observation/left_wrist_image": left_rgb,
                "observation/right_wrist_image": right_rgb,
                "observation/state": state,
            }
        )
        model_image = np.asarray(transformed["image"]["base_0_rgb"])
        if model_image.shape != (224, 224, 3) or model_image.dtype != np.uint8:
            raise ValueError(
                f"episode {source.episode_id!r} frame {raw_frame} produced invalid model top image "
                f"{model_image.shape}/{model_image.dtype}"
            )
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
        return _model.Observation.from_dict(batched), np.array(model_image, copy=True)

    def _evaluate_frame(
        self,
        source: EpisodeSource,
        *,
        raw_frame: int,
        policy_step: int,
        top_rgb: np.ndarray,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray,
        robot_state: np.ndarray,
        memory_state: object,
        phase: str = "",
    ) -> tuple[object, _MeasuredFrame]:
        observation, model_image = self._transform_observation(
            source,
            raw_frame,
            top_rgb,
            left_rgb,
            right_rgb,
            robot_state,
        )
        memory_state, aux = self._step(observation, memory_state, allow_write=True)
        memory_state, aux = jax.device_get((memory_state, aux))
        metrics = {name: np.asarray(aux[name][0], dtype=np.float32) for name in SUPPORTED_METRICS}
        if any(values.shape != (token_heatmap.TOKEN_COUNT,) for values in metrics.values()):
            shapes = {name: values.shape for name, values in metrics.items()}
            raise ValueError(f"expected a 16x16/256-token writer grid, got {shapes}")
        surprise = float(np.asarray(aux["surprise"])[0])
        mean_error = float(np.mean(metrics["token_error"], dtype=np.float64))
        if not np.isclose(surprise, mean_error, rtol=2e-5, atol=2e-5):
            raise RuntimeError(
                f"mean per-token error does not match the frame associative loss: {mean_error} != {surprise}"
            )
        scalar = {
            name: float(np.asarray(aux[name])[0])
            for name in (
                "surprise",
                "grad_norm",
                "theta",
                "eta",
                "alpha",
                "retrieval_norm",
                "write_source_norm",
                "memory_gate_norm",
            )
        }
        measured = _MeasuredFrame(
            raw_frame=raw_frame,
            policy_step=policy_step,
            model_image_rgb=model_image,
            raw_image_rgb=np.array(top_rgb, copy=True),
            metrics=metrics,
            scalar=scalar,
            phase=phase,
        )
        return memory_state, measured

    def _measure_raw_episode(self, source: EpisodeSource) -> tuple[list[_MeasuredFrame], int]:
        files = _required_episode_files(source)
        state = _load_states(files, source.episode_id)
        total = min(len(state), *(_video_frame_count(files[name]) for name in ("top", "left", "right")))
        captures = {name: cv2.VideoCapture(str(files[name])) for name in ("top", "left", "right")}
        if not all(capture.isOpened() for capture in captures.values()):
            for capture in captures.values():
                capture.release()
            raise ValueError(f"could not open all three videos for episode {source.episode_id!r}")

        memory_state = self.model.memory.init_state(1)
        measured: list[_MeasuredFrame] = []
        try:
            for raw_frame in range(total):
                decoded = {name: capture.read() for name, capture in captures.items()}
                if not all(ok for ok, _ in decoded.values()):
                    raise ValueError(f"video decode ended early in episode {source.episode_id!r} at frame {raw_frame}")
                if raw_frame % self.stride:
                    continue
                rgb = {name: cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for name, (_, frame) in decoded.items()}
                memory_state, frame = self._evaluate_frame(
                    source,
                    raw_frame=raw_frame,
                    policy_step=len(measured),
                    top_rgb=rgb["top"],
                    left_rgb=rgb["left"],
                    right_rgb=rgb["right"],
                    robot_state=state[raw_frame],
                    memory_state=memory_state,
                )
                measured.append(frame)
                print(
                    f"[{source.episode_id}] frame {raw_frame}/{total - 1} write {len(measured)} "
                    f"{self._progress_detail(frame)}",
                    flush=True,
                )
                if self.options.max_frames is not None and len(measured) >= self.options.max_frames:
                    break
        finally:
            for capture in captures.values():
                capture.release()
        if not measured:
            raise ValueError(f"episode {source.episode_id!r} yielded no sampled frames")
        return measured, total

    @staticmethod
    def _decode_inline_image(value: Any, *, field: str, raw_frame: int) -> np.ndarray:
        if not isinstance(value, Mapping) or not isinstance(value.get("bytes"), bytes):
            raise ValueError(f"LeRobot {field} frame {raw_frame} is not an inline image struct")
        decoded = cv2.imdecode(np.frombuffer(value["bytes"], dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError(f"could not decode LeRobot {field} frame {raw_frame}")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    def _measure_lerobot_episode(self, source: EpisodeSource) -> tuple[list[_MeasuredFrame], int]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - pyarrow is part of the training environment.
            raise RuntimeError("pyarrow is required for inline-parquet LeRobot replay") from exc

        parquet = pq.ParquetFile(source.path)
        required = {"image", "left_wrist_image", "right_wrist_image", "state", "frame_index", "task_index"}
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"LeRobot episode {source.episode_id!r} lacks parquet columns {sorted(missing)}")
        total = parquet.metadata.num_rows
        memory_state = self.model.memory.init_state(1)
        measured: list[_MeasuredFrame] = []
        columns = ["image", "left_wrist_image", "right_wrist_image", "state", "frame_index", "task_index"]
        for batch in parquet.iter_batches(batch_size=100, columns=columns):
            for row in batch.to_pylist():
                raw_frame = row["frame_index"]
                if isinstance(raw_frame, bool) or not isinstance(raw_frame, int) or raw_frame < 0:
                    raise ValueError(f"invalid frame_index in episode {source.episode_id!r}: {raw_frame!r}")
                if raw_frame % self.stride:
                    continue
                robot_state = np.asarray(row["state"], dtype=np.float32)
                if robot_state.shape != (14,) or not np.all(np.isfinite(robot_state)):
                    raise ValueError(f"invalid state at episode {source.episode_id!r} frame {raw_frame}")
                images = {
                    name: self._decode_inline_image(row[field], field=field, raw_frame=raw_frame)
                    for name, field in (
                        ("top", "image"),
                        ("left", "left_wrist_image"),
                        ("right", "right_wrist_image"),
                    )
                }
                task_index = row["task_index"]
                phase = ""
                if isinstance(task_index, int) and 0 <= task_index < len(source.task_names):
                    phase = source.task_names[task_index]
                memory_state, frame = self._evaluate_frame(
                    source,
                    raw_frame=raw_frame,
                    policy_step=len(measured),
                    top_rgb=images["top"],
                    left_rgb=images["left"],
                    right_rgb=images["right"],
                    robot_state=robot_state,
                    memory_state=memory_state,
                    phase=phase,
                )
                measured.append(frame)
                print(
                    f"[{source.episode_id}] frame {raw_frame}/{total - 1} write {len(measured)} "
                    f"task={phase!r} {self._progress_detail(frame)}",
                    flush=True,
                )
                if self.options.max_frames is not None and len(measured) >= self.options.max_frames:
                    break
            if self.options.max_frames is not None and len(measured) >= self.options.max_frames:
                break
        if not measured:
            raise ValueError(f"episode {source.episode_id!r} yielded no sampled frames")
        return measured, total

    def _measure_episode(self, source: EpisodeSource) -> tuple[list[_MeasuredFrame], int]:
        if source.source_format == "raw_yam":
            return self._measure_raw_episode(source)
        if source.source_format == "lerobot_inline_parquet":
            return self._measure_lerobot_episode(source)
        raise AssertionError(f"unhandled episode source format {source.source_format!r}")

    def _save_episode(
        self,
        source: EpisodeSource,
        measured: Sequence[_MeasuredFrame],
        total_raw_frames: int,
        episode_dir: Path,
    ) -> dict[str, Any]:
        episode_dir.mkdir(parents=True, exist_ok=False)
        stacked_metrics = {name: np.stack([frame.metrics[name] for frame in measured]) for name in SUPPORTED_METRICS}
        scalar_names = tuple(measured[0].scalar)
        np.savez_compressed(
            episode_dir / "contributions.npz",
            raw_frame=np.asarray([frame.raw_frame for frame in measured], dtype=np.int64),
            policy_step=np.asarray([frame.policy_step for frame in measured], dtype=np.int64),
            task=np.asarray([frame.phase for frame in measured], dtype=np.str_),
            **stacked_metrics,
            **{name: np.asarray([frame.scalar[name] for frame in measured], dtype=np.float32) for name in scalar_names},
        )

        with (episode_dir / "frame_summary.csv").open("w", newline="", encoding="utf-8") as stream:
            fieldnames = [
                "raw_frame",
                "policy_step",
                "task",
                *scalar_names,
                *(
                    f"{name}_{stat}"
                    for name in SUPPORTED_METRICS
                    for stat in (
                        "mean",
                        "std",
                        "coefficient_of_variation",
                        "normalized_entropy",
                        "effective_token_count",
                        "top_10pct_mass_fraction",
                    )
                ),
            ]
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for frame in measured:
                row: dict[str, Any] = {
                    "raw_frame": frame.raw_frame,
                    "policy_step": frame.policy_step,
                    "task": frame.phase,
                    **frame.scalar,
                }
                for name, values in frame.metrics.items():
                    stats = token_heatmap.metric_statistics(values)
                    for stat in (
                        "mean",
                        "std",
                        "coefficient_of_variation",
                        "normalized_entropy",
                        "effective_token_count",
                        "top_10pct_mass_fraction",
                    ):
                        row[f"{name}_{stat}"] = stats[stat]
                writer.writerow(row)

        videos = {}
        if self.options.render_video:
            fps = self.options.video_fps or source.control_hz / self.stride
            for metric_name in self.options.metrics:
                frames = []
                for frame in measured:
                    kwargs: dict[str, Any] = {}
                    # token_heatmap v1 originally rendered on the exact 224x224 model input;
                    # newer versions additionally inverse-map the letterbox onto the raw image.
                    fields = {field.name for field in dataclasses.fields(token_heatmap.TokenMetricFrame)}
                    if "raw_image_rgb" in fields:
                        kwargs["raw_image_rgb"] = frame.raw_image_rgb
                    if "letterbox_geometry" in fields:
                        kwargs["letterbox_geometry"] = token_heatmap.letterbox_geometry(
                            frame.raw_image_rgb.shape[0], frame.raw_image_rgb.shape[1]
                        )
                    frames.append(
                        token_heatmap.TokenMetricFrame(
                            raw_frame=frame.raw_frame,
                            policy_step=frame.policy_step,
                            model_image_rgb=frame.model_image_rgb,
                            token_values=frame.metrics[metric_name],
                            write_count=frame.policy_step + 1,
                            phase=frame.phase,
                            timestamp_s=frame.raw_frame / source.control_hz,
                            metadata={
                                "episode_id": source.episode_id,
                                "write_source": "post_attention",
                                "pre_write_measurement": True,
                            },
                            **kwargs,
                        )
                    )
                for scale_mode in self.options.heatmap_scale_modes:
                    if scale_mode == "video":
                        video_key = metric_name
                        mode_colormap = self.options.colormap
                    elif scale_mode == "per_token_history":
                        video_key = f"{metric_name}_token_history"
                        mode_colormap = "coolwarm"
                    else:
                        video_key = f"{metric_name}_perframe_zscore"
                        mode_colormap = "coolwarm"
                    videos[video_key] = token_heatmap.export_heatmap_video(
                        frames,
                        episode_dir / "heatmaps" / video_key,
                        metric_name=metric_name,
                        fps=fps,
                        normalization=token_heatmap.NormalizationSpec(
                            lower_percentile=0.0,
                            upper_percentile=self.options.upper_percentile,
                            anchor_zero=True,
                        ),
                        alpha=self.options.alpha,
                        colormap=mode_colormap,
                        # This runner has already initialized multithreaded JAX.  OpenCV encodes
                        # in-process, avoiding imageio-ffmpeg's unsafe os.fork() after JAX startup.
                        video_encoder=WRITER_VIDEO_ENCODER,
                        scale_mode=scale_mode,
                        zscore_range=self.options.zscore_range,
                    )

        aggregate = {}
        for name, values in stacked_metrics.items():
            per_frame = [token_heatmap.metric_statistics(row) for row in values]
            aggregate[name] = {
                "global": token_heatmap.metric_statistics(values.reshape(-1, token_heatmap.TOKEN_COUNT).mean(axis=0)),
                "per_frame_effective_token_count_mean": float(
                    np.mean([stats["effective_token_count"] for stats in per_frame])
                ),
                "per_frame_normalized_entropy_mean": float(
                    np.mean([stats["normalized_entropy"] for stats in per_frame])
                ),
                "per_frame_top_10pct_mass_fraction_mean": float(
                    np.mean([stats["top_10pct_mass_fraction"] for stats in per_frame])
                ),
            }
        summary = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": source.episode_id,
            "path": str(source.path),
            "source_format": source.source_format,
            "ground_truth_side": source.ground_truth_side,
            "control_hz": source.control_hz,
            "total_raw_frames": total_raw_frames,
            "sampled_write_frames": len(measured),
            "stride": self.stride,
            "first_raw_frame": measured[0].raw_frame,
            "last_raw_frame": measured[-1].raw_frame,
            "aggregate": aggregate,
            "videos": {name: str(Path("heatmaps") / name / details["video"]) for name, details in videos.items()},
        }
        _write_json(episode_dir / "summary.json", summary)
        return summary

    def run(self) -> dict[str, Any]:
        start = time.monotonic()
        self.options.output_dir.mkdir(parents=True, exist_ok=False)
        episodes_dir = self.options.output_dir / "episodes"
        episodes_dir.mkdir()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_revision": v31.current_code_revision(Path(__file__).parents[3]),
            "checkpoint_path": str(self.options.checkpoint),
            "checkpoint_params_hash": v31_pi0._path_hash(self.options.checkpoint / "params"),  # noqa: SLF001
            "checkpoint_static_config_provenance_verified": self.provenance.verified,
            "checkpoint_metadata_path": self.provenance.metadata_path,
            "config": self.options.config,
            "memory_write_source": self.train_config.model.memory_write_source,
            "memory_layer": self.train_config.model.memory_layer,
            "configured_stride": self.configured_stride,
            "effective_stride": self.stride,
            "metrics": list(self.options.metrics),
            "render_video": self.options.render_video,
            "heatmap_scale_modes": list(self.options.heatmap_scale_modes),
            "zscore_range": self.options.zscore_range,
            "video_encoder_requested": WRITER_VIDEO_ENCODER if self.options.render_video else None,
            "video_fps_override": self.options.video_fps,
            "max_frames": self.options.max_frames,
            "source_manifest": None if self.options.episode_manifest is None else str(self.options.episode_manifest),
            "source_manifest_hash": (
                None if self.options.episode_manifest is None else v31.sha256_file(self.options.episode_manifest)
            ),
            "dataset_root": None if self.options.dataset_root is None else str(self.options.dataset_root),
            "episode_indices": list(self.options.episode_indices),
            "episodes": [
                {
                    "episode_id": source.episode_id,
                    "path": str(source.path),
                    "source_format": source.source_format,
                    "control_hz": source.control_hz,
                    "ground_truth_side": source.ground_truth_side,
                }
                for source in self.sources
            ],
            "method": {
                "measurement_state": "pre_write M_(t-1)",
                "selected_write_tensor": "post-attention c_t",
                "update_semantics": "one frame-level mean loss over 256 tokens, not 256 sequential writes",
                "policy_compute_skipped": ["subtask decoding", "canonical scoring", "action denoising"],
                "token_layout": token_heatmap.TOKEN_LAYOUT,
            },
            "interpretation_warnings": [
                "Individual token-gradient norms do not add to the aggregate frame gradient because vectors can align or cancel.",
                "The frame update averages 256 token losses; theta and clipping scale their aggregate, not 256 separate writes.",
                "At an exactly zero-output blank memory, normalized values make token_error approximately one everywhere.",
                "v3.1 c_t tokens are contextualized by attention, so the grid is slot-aligned attribution, not strict pixel causality.",
                "'video'-mode colors use one fixed video-wide zero-anchored scale; raw arrays remain in contributions.npz.",
                "'per_frame_zscore'-mode colors standardize each frame by its own mean/std: they show relative within-frame structure only and amplify noise when the frame is near-uniform -- read the printed per-frame CV before trusting a hotspot.",
            ],
        }
        _write_json(self.options.output_dir / "run_manifest.json", manifest)

        episode_summaries = []
        for source in self.sources:
            measured, total = self._measure_episode(source)
            episode_summaries.append(
                self._save_episode(
                    source,
                    measured,
                    total,
                    episodes_dir / _safe_name(source.episode_id),
                )
            )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "episode_count": len(episode_summaries),
            "sampled_write_frames": sum(item["sampled_write_frames"] for item in episode_summaries),
            "elapsed_s": time.monotonic() - start,
            "episodes": episode_summaries,
        }
        _write_json(self.options.output_dir / "summary.json", summary)
        return summary


def _parse_metrics(value: str) -> tuple[str, ...]:
    metrics = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(metrics) - set(SUPPORTED_METRICS)
    if not metrics or unknown or len(set(metrics)) != len(metrics):
        raise argparse.ArgumentTypeError(
            f"metrics must be a unique comma list from {SUPPORTED_METRICS}; unknown={sorted(unknown)}"
        )
    return metrics


def _parse_scale_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = set(modes) - set(token_heatmap.SCALE_MODES)
    if not modes or unknown or len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError(
            f"heatmap scale modes must be a unique comma list from {token_heatmap.SCALE_MODES}; "
            f"unknown={sorted(unknown)}"
        )
    return modes


def _parse_episode_indices(value: str) -> tuple[int, ...]:
    parts = tuple(item.strip() for item in value.split(",") if item.strip())
    try:
        indices = tuple(int(item) for item in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("episode indices must be comma-separated integers") from exc
    if not indices or any(index < 0 for index in indices) or len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("episode indices must be a unique non-empty list of non-negative integers")
    return indices


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fast offline 16x16 v3.1 memory-writer contribution replay",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="pi05_yam_mem_v31")
    parser.add_argument("--output-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--episode", dest="episode_paths", type=Path, action="append")
    source.add_argument("--episode-manifest", type=Path)
    source.add_argument(
        "--dataset-root",
        type=Path,
        help="LeRobot v2.1 root with inline image bytes in per-episode parquet files",
    )
    parser.add_argument(
        "--episode-indices",
        type=_parse_episode_indices,
        help="comma-separated LeRobot episode indices, required with --dataset-root (for example 0,2)",
    )
    parser.add_argument("--stride", type=int, help="raw-frame write stride; default is the config cadence")
    parser.add_argument("--metrics", type=_parse_metrics, default=DEFAULT_METRICS)
    parser.add_argument("--video-fps", type=float, help="default preserves real time: control_hz / stride")
    parser.add_argument("--max-frames", type=int, help="smoke/debug limit per episode")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--colormap", choices=("inferno", "magma", "turbo", "viridis"), default="inferno")
    parser.add_argument("--alpha", type=float, default=0.58)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    parser.add_argument(
        "--heatmap-scale-modes",
        type=_parse_scale_modes,
        default=("video", "per_frame_zscore", "per_token_history"),
        help=(
            "comma list from video,per_frame_zscore,per_token_history; 'video' is one absolute scale for the "
            "whole clip, 'per_frame_zscore' standardizes every frame by its own mean/std so late-episode "
            "structure stays visible after the blank-memory transient, and 'per_token_history' scores every "
            "token against its own frame-detrended baseline across the episode so static scene layout cancels "
            "and per-write events stand out"
        ),
    )
    parser.add_argument(
        "--zscore-range",
        type=float,
        default=3.0,
        help="color-clip bound in per-frame standard deviations for per_frame_zscore videos",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    namespace = _build_parser().parse_args(argv)
    options = RunOptions(
        checkpoint=namespace.checkpoint,
        output_dir=namespace.output_dir,
        episode_paths=tuple(namespace.episode_paths or ()),
        episode_manifest=namespace.episode_manifest,
        dataset_root=namespace.dataset_root,
        episode_indices=tuple(namespace.episode_indices or ()),
        config=namespace.config,
        stride=namespace.stride,
        metrics=namespace.metrics,
        video_fps=namespace.video_fps,
        max_frames=namespace.max_frames,
        render_video=not namespace.skip_video,
        colormap=namespace.colormap,
        alpha=namespace.alpha,
        upper_percentile=namespace.upper_percentile,
        heatmap_scale_modes=namespace.heatmap_scale_modes,
        zscore_range=namespace.zscore_range,
    )
    summary = WriterContributionRunner(options).run()
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
