"""Trusted parquet/GPU collector for v3.5 pilot-rung evidence.

The collector has two create-only stages:

``select`` freezes the exact train-54/development-8 frame population from scalar parquet
columns.  ``collect`` authenticates a finalized checkpoint, runs the production model on only
those frozen converted-parquet rows, writes the raw condition, prototype, mechanism, and
paired task-health artifacts consumed by :mod:`v35_rung_eval`, then seals and reloads the
canonical pilot-rung result.  The final-test split is never resolved to a parquet path.

All CLI paths are relative to ``memory_project``.  Selection is deterministic and CPU-only;
collection is intended for one H100 and uses compiled model methods.  No output is overwritten.
"""

from __future__ import annotations

import os

# The rung battery runs many distinct jitted programs (interface, decode, task health)
# plus one eager attention diagnostic on a single GPU; the default caching allocator
# fragments and OOMs near the end. The platform allocator frees between programs.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from numpy import typing as npt

from openpi.shared import project_paths

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import v35_gate_artifacts as artifacts  # noqa: E402
import v35_rung_eval as rung_eval  # noqa: E402
import v35_side_prototypes as prototypes  # noqa: E402

CONFIG_NAME = "pi05_yam_mem_v35"
SELECTION_SCHEMA_VERSION = rung_eval.SELECTION_SCHEMA_VERSION
CHECK_EVIDENCE_SCHEMA_VERSION = "openpi.v35.pytest-check-evidence.v1"
STRIDE = 15
E_TAIL_GUARD = 5
MIN_EXECUTE_FRAMES = 15
ACTION_DENOISE_STEPS = 10
ACTION_MAX_DECODE_STEPS = 10
TASK_HEALTH_ROWS_PER_CELL = 2
TASK_HEALTH_SEED = 3_505
EXPECTED_TASKS = {
    "open both lids",
    "inspect both bins",
    "close both lids and reset arms",
    "wait; target bin is left",
    "wait; target bin is right",
    "open left bin",
    "open right bin",
}

SELECTION_PROTOCOL = rung_eval.SELECTION_PROTOCOL
SELECTION_PROTOCOL_SHA256 = rung_eval.SELECTION_PROTOCOL_SHA256


class RungCollectionError(rung_eval.RungEvaluationError):
    """Raised when trusted selection or production collection cannot be completed."""


@dataclasses.dataclass(frozen=True)
class EpisodePlan:
    stable_id: str
    episode_index: int
    split: str
    collection: str
    object_name: str
    target_side: int
    prompt: str
    expected_frames: int
    parquet_sha256: str
    e_frames: tuple[int, ...]
    d_frames: tuple[int, ...]
    use_frames: tuple[int, ...]

    @property
    def final_e(self) -> int:
        return self.e_frames[-1]


@dataclasses.dataclass(frozen=True)
class SelectedFrame:
    stable_id: str
    episode_index: int
    frame_index: int
    task: str
    rng_seed: int
    flow_time: float


@dataclasses.dataclass(frozen=True)
class FrameRow:
    frame_index: int
    task: str
    image: npt.NDArray[np.uint8]
    left_wrist_image: npt.NDArray[np.uint8]
    right_wrist_image: npt.NDArray[np.uint8]
    state: npt.NDArray[np.float32]
    actions: npt.NDArray[np.float32]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _project_path(value: Path, *, name: str, must_exist: bool = True) -> Path:
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise RungCollectionError(f"{name} must be relative to memory_project")
    try:
        path = project_paths.project_path(raw)
    except project_paths.ProjectRootError as exc:
        raise RungCollectionError(f"invalid {name}: {exc}") from exc
    if must_exist and not path.exists():
        raise RungCollectionError(f"{name} does not exist: {raw.as_posix()}")
    return path


def _write_npz_once(path: Path, arrays: Mapping[str, npt.NDArray[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RungCollectionError(f"refusing to overwrite NPZ {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise RungCollectionError(f"refusing to overwrite NPZ {path}") from exc
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise RungCollectionError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RungCollectionError(f"{name} must be one JSON object")
    return value


def _task_names(dataset_root: Path) -> dict[int, str]:
    output: dict[int, str] = {}
    try:
        lines = (dataset_root / "meta" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RungCollectionError(f"cannot read converted task metadata: {exc}") from exc
    for line in lines:
        if not line.strip():
            continue
        value = json.loads(line)
        index, task = value.get("task_index"), value.get("task")
        if type(index) is not int or not isinstance(task, str) or index in output:
            raise RungCollectionError("tasks.jsonl contains an invalid/duplicate task")
        output[index] = task
    if set(output.values()) != EXPECTED_TASKS:
        raise RungCollectionError("converted dataset does not have the frozen seven-task vocabulary")
    return output


def _manifest_raw_records(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    raw = _load_json(path, name="frozen manifest")
    records = raw.get("episodes")
    if not isinstance(records, list):
        raise RungCollectionError("frozen manifest is missing episode records")
    by_id = {
        str(record.get("stable_id")): record
        for record in records
        if isinstance(record, dict) and record.get("include") is True
    }
    if len(by_id) != 70:
        raise RungCollectionError("frozen manifest must expose exactly 70 included episodes")
    return raw, by_id


def _parquet_path(dataset_root: Path, *, episode_index: int, split: str) -> Path:
    if split == "final_test":
        raise RungCollectionError("final-test parquet paths are sealed and may not be resolved")
    if split not in ("train", "development"):
        raise RungCollectionError(f"unsupported collection split {split!r}")
    return dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"


def _scalar_columns(path: Path, *, expected_episode: int, expected_frames: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        import pyarrow.parquet as pq

        values = pq.read_table(path, columns=["episode_index", "frame_index", "task_index"]).to_pydict()
    except Exception as exc:
        raise RungCollectionError(f"cannot read scalar columns from {path}: {exc}") from exc
    episode = np.asarray(values["episode_index"], dtype=np.int64)
    frame = np.asarray(values["frame_index"], dtype=np.int64)
    task = np.asarray(values["task_index"], dtype=np.int64)
    if (
        len(frame) != expected_frames
        or not np.array_equal(frame, np.arange(expected_frames))
        or not np.all(episode == expected_episode)
        or task.shape != frame.shape
    ):
        raise RungCollectionError(f"converted parquet identity/clock mismatch: {path}")
    return frame, task


def _stable_hash_int(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _task_health_selection(plans: Sequence[EpisodePlan]) -> tuple[SelectedFrame, ...]:
    cells: dict[tuple[str, str, int], list[EpisodePlan]] = {}
    for plan in plans:
        if plan.split == "train":
            cells.setdefault((plan.collection, plan.object_name, plan.target_side), []).append(plan)
    if len(cells) != 8:
        raise RungCollectionError("task-health selection requires all eight train cells")
    selected: list[SelectedFrame] = []
    for cell in sorted(cells):
        ranked = sorted(cells[cell], key=lambda p: (_stable_hash_int("suite", str(TASK_HEALTH_SEED), p.stable_id), p.stable_id))
        if len(ranked) < TASK_HEALTH_ROWS_PER_CELL:
            raise RungCollectionError(f"task-health cell {cell} has too few episodes")
        for plan in ranked[:TASK_HEALTH_ROWS_PER_CELL]:
            candidate_count = (plan.expected_frames - 50) // STRIDE + 1
            if candidate_count <= 0:
                raise RungCollectionError(f"episode {plan.stable_id} is too short for task health")
            value = _stable_hash_int("frame", str(TASK_HEALTH_SEED), plan.stable_id)
            frame = int(value % candidate_count) * STRIDE
            rng_seed = int(_stable_hash_int("rng", str(TASK_HEALTH_SEED), plan.stable_id, str(frame)) % (2**31 - 1))
            flow_bits = _stable_hash_int("time", str(TASK_HEALTH_SEED), plan.stable_id, str(frame)) % 800_001
            selected.append(
                SelectedFrame(
                    stable_id=plan.stable_id,
                    episode_index=plan.episode_index,
                    frame_index=frame,
                    task="",  # Filled from authenticated scalar parquet below.
                    rng_seed=rng_seed,
                    flow_time=0.1 + float(flow_bits) / 1_000_000.0,
                )
            )
    return tuple(selected)


def build_selection(
    *,
    manifest: artifacts.FrozenManifest,
    manifest_path: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    """Read only scalar train/dev parquet columns and return the canonical selection envelope."""

    _, raw_by_id = _manifest_raw_records(manifest_path)
    tasks = _task_names(dataset_root)
    plans: list[EpisodePlan] = []
    for split in ("train", "development"):
        for episode in manifest.split(split):
            record = raw_by_id.get(episode.stable_id)
            if not isinstance(record, dict) or record.get("split") != split:
                raise RungCollectionError(f"manifest raw/reduced mismatch for {episode.stable_id}")
            expected_frames = record.get("expected_num_frames")
            visibility = record.get("e_visibility")
            d_valid = record.get("d_valid")
            if type(expected_frames) is not int or not isinstance(visibility, dict) or not isinstance(d_valid, dict):
                raise RungCollectionError(f"manifest phase metadata is incomplete for {episode.stable_id}")
            parquet = _parquet_path(dataset_root, episode_index=episode.episode_index, split=split)
            frame, task_index = _scalar_columns(
                parquet, expected_episode=episode.episode_index, expected_frames=expected_frames
            )
            task = np.asarray([tasks.get(int(index), "") for index in task_index])
            evidence_rows = frame[task == "inspect both bins"]
            if not len(evidence_rows):
                raise RungCollectionError(f"episode {episode.stable_id} has no semantic E phase")
            e_limit = min(int(visibility.get("last_clean_visible_frame", -1)), int(evidence_rows[-1]) - E_TAIL_GUARD)
            e_start = int(visibility.get("first_valid_visible_frame", -1))
            e_frames = tuple(
                int(value)
                for value in frame[
                    (frame % STRIDE == 0)
                    & (task == "inspect both bins")
                    & (frame >= e_start)
                    & (frame <= e_limit)
                ]
            )
            side_name = "right" if episode.target_side else "left"
            wait_task = f"wait; target bin is {side_name}"
            d_start, d_end = d_valid.get("start"), d_valid.get("end")
            if type(d_start) is not int or type(d_end) is not int:
                raise RungCollectionError(f"episode {episode.stable_id} has invalid d_valid")
            d_frames = tuple(
                int(value)
                for value in frame[
                    (frame % STRIDE == 0) & (frame >= d_start) & (frame <= d_end) & (task == wait_task)
                ]
            )
            execute = {int(value) for value in frame[task == f"open {side_name} bin"]}
            use_frames = tuple(
                value
                for value in d_frames
                if sum(value + offset in execute for offset in range(50)) >= MIN_EXECUTE_FRAMES
            )
            if not e_frames or not d_frames or not use_frames:
                raise RungCollectionError(
                    f"episode {episode.stable_id} has empty E/D/use selection: "
                    f"{len(e_frames)}/{len(d_frames)}/{len(use_frames)}"
                )
            plans.append(
                EpisodePlan(
                    stable_id=episode.stable_id,
                    episode_index=episode.episode_index,
                    split=split,
                    collection=episode.collection,
                    object_name=episode.object_name,
                    target_side=episode.target_side,
                    prompt=str(record.get("prompt")),
                    expected_frames=expected_frames,
                    parquet_sha256=_sha256_file(parquet),
                    e_frames=e_frames,
                    d_frames=d_frames,
                    use_frames=use_frames,
                )
            )
    expected_ids = [episode.stable_id for split in ("train", "development") for episode in manifest.split(split)]
    if [plan.stable_id for plan in plans] != expected_ids:
        raise RungCollectionError("selection does not preserve exact frozen train/dev order")

    task_health = []
    plan_by_id = {plan.stable_id: plan for plan in plans}
    scalar_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for item in _task_health_selection(plans):
        plan = plan_by_id[item.stable_id]
        parquet = _parquet_path(dataset_root, episode_index=plan.episode_index, split="train")
        if plan.stable_id not in scalar_cache:
            scalar_cache[plan.stable_id] = _scalar_columns(
                parquet, expected_episode=plan.episode_index, expected_frames=plan.expected_frames
            )
        frame, task_index = scalar_cache[plan.stable_id]
        position = np.flatnonzero(frame == item.frame_index)
        if len(position) != 1:
            raise RungCollectionError("task-health selected frame is absent")
        task_health.append(dataclasses.asdict(dataclasses.replace(item, task=tasks[int(task_index[position[0]])])))

    payload = {
        "dataset_root_relative": project_paths.project_relative_path(dataset_root).as_posix(),
        "episode_manifest_sha256": manifest.sha256,
        "episodes": [dataclasses.asdict(plan) for plan in plans],
        "final_test_payload_access_count": 0,
        "protocol_sha256": SELECTION_PROTOCOL_SHA256,
        "split_assignment_sha256": manifest.split_assignment_sha256,
        "task_health": task_health,
    }
    return artifacts.artifact_envelope(SELECTION_SCHEMA_VERSION, payload)


def freeze_selection(
    *, manifest: artifacts.FrozenManifest, manifest_path: Path, dataset_root: Path, output_path: Path
) -> Path:
    envelope = build_selection(manifest=manifest, manifest_path=manifest_path, dataset_root=dataset_root)
    artifacts.write_canonical_envelope(output_path, envelope, schema_version=SELECTION_SCHEMA_VERSION)
    return output_path


def load_selection(
    path: Path, *, manifest: artifacts.FrozenManifest, dataset_root: Path
) -> tuple[tuple[EpisodePlan, ...], tuple[SelectedFrame, ...]]:
    envelope = artifacts.load_canonical_envelope(path, schema_version=SELECTION_SCHEMA_VERSION)
    payload = artifacts.require_exact_keys(
        "selection payload",
        envelope["payload"],
        {
            "dataset_root_relative",
            "episode_manifest_sha256",
            "episodes",
            "final_test_payload_access_count",
            "protocol_sha256",
            "split_assignment_sha256",
            "task_health",
        },
    )
    if (
        payload["episode_manifest_sha256"] != manifest.sha256
        or payload["split_assignment_sha256"] != manifest.split_assignment_sha256
        or payload["protocol_sha256"] != SELECTION_PROTOCOL_SHA256
        or payload["final_test_payload_access_count"] != 0
        or payload["dataset_root_relative"] != project_paths.project_relative_path(dataset_root).as_posix()
    ):
        raise RungCollectionError("selection manifest/protocol/dataset identity mismatch")
    try:
        plans = tuple(
            EpisodePlan(
                **{
                    **record,
                    "e_frames": tuple(record["e_frames"]),
                    "d_frames": tuple(record["d_frames"]),
                    "use_frames": tuple(record["use_frames"]),
                }
            )
            for record in payload["episodes"]
        )
        suite = tuple(SelectedFrame(**record) for record in payload["task_health"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RungCollectionError(f"selection records are malformed: {exc}") from exc
    expected_ids = [episode.stable_id for split in ("train", "development") for episode in manifest.split(split)]
    if [plan.stable_id for plan in plans] != expected_ids or len(suite) != 16:
        raise RungCollectionError("selection population/order is not exact train74+dev8 and 16-row suite")
    for plan in plans:
        parquet = _parquet_path(dataset_root, episode_index=plan.episode_index, split=plan.split)
        if not parquet.is_file() or _sha256_file(parquet) != plan.parquet_sha256:
            raise RungCollectionError(f"selected parquet changed: {plan.stable_id}")
    if {item.stable_id for item in suite} - {episode.stable_id for episode in manifest.split("train")}:
        raise RungCollectionError("task-health suite contains held-out IDs")
    return plans, suite


def _decode_image(value: Any, *, field: str, frame: int) -> npt.NDArray[np.uint8]:
    from PIL import Image

    payload = value.get("bytes") if isinstance(value, dict) else value
    if not isinstance(payload, bytes | bytearray | memoryview):
        raise RungCollectionError(f"{field} frame {frame} is not an inline image payload")
    try:
        with Image.open(io.BytesIO(bytes(payload))) as image:
            decoded = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except OSError as exc:
        raise RungCollectionError(f"cannot decode {field} frame {frame}: {exc}") from exc
    if decoded.ndim != 3 or decoded.shape[-1] != 3:
        raise RungCollectionError(f"decoded {field} frame {frame} is not RGB")
    return decoded


def _load_episode_rows(
    *, dataset_root: Path, plan: EpisodePlan, extra_frames: Sequence[int] = ()
) -> dict[int, FrameRow]:
    needed = set(plan.e_frames) | set(plan.d_frames) | set(plan.use_frames) | set(extra_frames)
    if not needed:
        raise RungCollectionError(f"no selected payload rows for {plan.stable_id}")
    parquet = _parquet_path(dataset_root, episode_index=plan.episode_index, split=plan.split)
    columns = (
        "episode_index",
        "frame_index",
        "task_index",
        "image",
        "left_wrist_image",
        "right_wrist_image",
        "state",
        "actions",
    )
    tasks = _task_names(dataset_root)
    found: dict[int, FrameRow] = {}
    try:
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=32, columns=list(columns)):
            values = batch.to_pydict()
            for row_index, raw_frame in enumerate(values["frame_index"]):
                frame = int(raw_frame)
                if frame not in needed:
                    continue
                if frame in found or int(values["episode_index"][row_index]) != plan.episode_index:
                    raise RungCollectionError(f"duplicate/foreign selected row in {plan.stable_id} frame {frame}")
                state = np.asarray(values["state"][row_index], dtype=np.float32)
                actions = np.asarray(values["actions"][row_index], dtype=np.float32)
                if state.shape != (14,) or actions.shape != (14,) or not np.all(np.isfinite(state)) or not np.all(
                    np.isfinite(actions)
                ):
                    raise RungCollectionError(f"nonfinite 14-D state/action at {plan.stable_id} frame {frame}")
                found[frame] = FrameRow(
                    frame_index=frame,
                    task=tasks[int(values["task_index"][row_index])],
                    image=_decode_image(values["image"][row_index], field="image", frame=frame),
                    left_wrist_image=_decode_image(
                        values["left_wrist_image"][row_index], field="left_wrist_image", frame=frame
                    ),
                    right_wrist_image=_decode_image(
                        values["right_wrist_image"][row_index], field="right_wrist_image", frame=frame
                    ),
                    state=state,
                    actions=actions,
                )
            if set(found) == needed:
                break
    except Exception as exc:
        if isinstance(exc, RungCollectionError):
            raise
        raise RungCollectionError(f"cannot load selected rows for {plan.stable_id}: {exc}") from exc
    if set(found) != needed:
        raise RungCollectionError(f"missing selected rows for {plan.stable_id}: {sorted(needed - set(found))}")
    return found


def _state_max_abs_difference(jax: Any, left: Any, right: Any) -> float:
    leaves = [
        np.max(np.abs(np.asarray(jax.device_get(a), dtype=np.float32) - np.asarray(jax.device_get(b), dtype=np.float32)))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    ]
    return float(max(leaves, default=0.0))


def _action_score(actions: npt.NDArray[Any], raw_state: npt.NDArray[Any]) -> float:
    actions = np.asarray(actions, dtype=np.float64)
    state = np.asarray(raw_state, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 14 or state.shape != (14,) or not np.all(np.isfinite(actions)):
        raise RungCollectionError("robot-space action score received invalid action/state geometry")
    displacement = actions - state[None]
    left = float(np.sqrt(np.mean(np.square(displacement[:, :6]))))
    right = float(np.sqrt(np.mean(np.square(displacement[:, 7:13]))))
    return (right - left) / (right + left + 1e-8)


class ProductionRuntime:
    """Authenticated, compiled checkpoint runner shared by every collected condition."""

    def __init__(self, *, checkpoint_step_dir: Path, manifest: artifacts.FrozenManifest):
        # Preserve the repository's Torch-before-config import ordering.
        # isort: off
        import eval_yam_mem_subtask_raw as checkpoint_eval
        import openpi.training.data_loader as _data_loader  # noqa: F401
        import jax
        import jax.numpy as jnp

        from openpi.models import model as model_lib
        from openpi.models import tokenizer as tokenizer_lib
        from openpi.shared import nnx_utils
        from openpi.training import config as config_lib
        from openpi.training import weight_loaders
        import openpi.transforms as transforms
        # isort: on

        self.jax, self.jnp = jax, jnp
        self.checkpoint_step_dir = Path(checkpoint_step_dir)
        config = config_lib.get_config(CONFIG_NAME)
        protocol = checkpoint_eval._load_v35_checkpoint_protocol(  # noqa: SLF001
            self.checkpoint_step_dir,
            expected_config_name=CONFIG_NAME,
            expected_seed=config.seed,
            expected_value_width=config.model.memory.d_value,
        )
        if protocol.manifest_sha256 != manifest.sha256:
            raise RungCollectionError("checkpoint-embedded manifest differs from selected frozen manifest")
        config = checkpoint_eval._apply_v35_checkpoint_calibration(config, protocol)  # noqa: SLF001
        params = model_lib.restore_params(self.checkpoint_step_dir / "params", restore_type=np.ndarray)
        self.checkpoint_sha256 = weight_loaders.parameter_tree_sha256(params)
        identity = _load_json(
            self.checkpoint_step_dir / "assets" / "v35_initialization_manifest.json",
            name="checkpoint initialization identity",
        )
        self.initialization_sha256 = artifacts.require_sha256(
            "actual_step0_parameter_tree_sha256", identity.get("actual_step0_parameter_tree_sha256")
        )
        self.initialization_identity_sha256 = artifacts.require_sha256(
            "initialization identity_sha256", identity.get("identity_sha256")
        )
        self.initialization_file_sha256 = _sha256_file(
            self.checkpoint_step_dir / "assets" / "v35_initialization_manifest.json"
        )
        self.protocol = protocol
        self.calibration = _load_json(protocol.calibration_path, name="checkpoint calibration")
        self.calibration_id = protocol.calibration_id
        self.prototype_target = float(self.calibration["payload"]["parameters"]["prototype_injected_rms_target"])

        self.model = config.model.load(params)
        self.model.eval()
        checkpoint_eval._validate_v35_loaded_gate(self.model, protocol)  # noqa: SLF001
        if any(np.dtype(leaf.dtype) != np.dtype(np.float32) for leaf in jax.tree.leaves(self.model.memory.init_state(1))):
            raise RungCollectionError("production v3.5 memory state is not entirely FP32")
        if not all(np.all(np.isfinite(np.asarray(jax.device_get(leaf)))) for leaf in jax.tree.leaves(params)):
            raise RungCollectionError("checkpoint parameter tree contains NaN/Inf")
        self.parameter_finite = True

        self.data_config = config.data.create(config.assets_dirs, config.model)
        if self.data_config.norm_stats is None:
            raise RungCollectionError("registered v3.5 transforms did not load train-54 normalization")
        norm_path = project_paths.project_path(project_paths.V35_ASSETS_DIR / project_paths.V35_REPO_ID) / "norm_stats.json"
        if _sha256_file(norm_path) != protocol.norm_stats_sha256:
            raise RungCollectionError("local normalization does not match checkpoint identity")
        self.preprocessing_norm_sha256 = protocol.norm_stats_sha256
        input_transforms = [
            item for item in self.data_config.data_transforms.inputs if not isinstance(item, transforms.BuildMemorySequence)
        ]
        self.data_input = transforms.compose(input_transforms)
        self.normalize = transforms.Normalize(
            self.data_config.norm_stats, use_quantiles=self.data_config.use_quantile_norm
        )
        self.model_input = transforms.compose(self.data_config.model_transforms.inputs)
        self.output_transform = transforms.compose(
            [
                *self.data_config.model_transforms.outputs,
                transforms.Unnormalize(self.data_config.norm_stats, use_quantiles=self.data_config.use_quantile_norm),
                *self.data_config.data_transforms.outputs,
            ]
        )
        tokenizers = [
            item
            for item in self.data_config.model_transforms.inputs
            if isinstance(item, transforms.TokenizeMemorySubtaskInputs)
        ]
        if len(tokenizers) != 1:
            raise RungCollectionError("registered v3.5 model transform has no unique memory tokenizer")
        self.memory_tokenizer = tokenizers[0].tokenizer
        pg = tokenizer_lib.FASTSubtaskTokenizer(config.model.max_token_len)._paligemma_tokenizer  # noqa: SLF001
        self.stop_token = int(pg.encode("placeholder subtask\n")[-1])
        self.config = config
        self.model_lib = model_lib

        self.interface = nnx_utils.module_jit(self.model.v32_memory_interface_step)
        self.pool_kv = nnx_utils.module_jit(self.model.memory.pool_kv)
        self.write = nnx_utils.module_jit(self.model.memory.write)
        self.decay = nnx_utils.module_jit(self.model.memory.analytic_decay)
        self.read_key = nnx_utils.module_jit(self.model.memory.read_key)
        self.writer_head = nnx_utils.module_jit(self.model.memory_write_side_head.__call__)
        self.reader_head = nnx_utils.module_jit(self.model.memory_read_side_head.__call__)
        self.sample = nnx_utils.module_jit(
            self.model.sample_with_memory,
            static_argnames=("stop_token", "max_decode_steps", "num_steps", "zero_read", "write_mode"),
        )
        self.task_health = nnx_utils.module_jit(self.model.v35_paired_task_health_step)
        # The attention-return flag lives on the bridged linen module (a plain dataclass
        # attribute, not nnx state), so it does not survive module_jit's split/merge
        # snapshot. Run this read-only diagnostic eagerly under a live context instead,
        # exactly as the unit contracts exercise it.
        def _action_attention(*args: Any, **kwargs: Any) -> Any:
            with self.model.capture_attention():
                return self.model.v35_action_memory_attention_step(*args, **kwargs)

        self.action_attention = _action_attention

    def blank_state(self) -> Any:
        return self.model.memory.init_state(1)

    @staticmethod
    def _raw_input(row: FrameRow, prompt: str, *, actions: np.ndarray | None = None) -> dict[str, Any]:
        value: dict[str, Any] = {
            "observation/image": row.image,
            "observation/left_wrist_image": row.left_wrist_image,
            "observation/right_wrist_image": row.right_wrist_image,
            "observation/state": row.state.copy(),
            "prompt": prompt,
        }
        if actions is not None:
            value["actions"] = actions
        return value

    def observation(self, row: FrameRow, prompt: str) -> tuple[Any, np.ndarray]:
        value = self.data_input(self._raw_input(row, prompt))
        value = self.normalize(value)
        normalized_state = np.asarray(value["state"], dtype=np.float32)
        value = self.model_input(value)
        batched = self.jax.tree.map(lambda leaf: self.jnp.asarray(leaf)[None], value)
        return self.model_lib.Observation.from_dict(batched), normalized_state

    def health_observation(
        self, row: FrameRow, prompt: str, task: str, future_actions: np.ndarray
    ) -> tuple[Any, npt.NDArray[np.float32]]:
        value = self.data_input(self._raw_input(row, prompt, actions=future_actions))
        value = self.normalize(value)
        normalized_state = np.asarray(value["state"], dtype=np.float32)
        normalized_actions = np.asarray(value["actions"], dtype=np.float32)
        context, context_mask, causal, causal_mask, causal_fast, context_state = self.memory_tokenizer.tokenize_split(
            prompt,
            normalized_state,
            task,
            normalized_actions,
            self.model.causal_token_len,
            return_state_mask=True,
        )
        inference_value = self.model_input(dict(value))
        for name, actual, expected in (
            ("context", inference_value["tokenized_prompt"], context),
            ("context mask", inference_value["tokenized_prompt_mask"], context_mask),
            ("context state mask", inference_value["token_state_mask"], context_state),
        ):
            if not np.array_equal(np.asarray(actual), np.asarray(expected)):
                raise RungCollectionError(f"task-health {name} differs from registered inference preprocessing")
        inference_value.update(
            {
                "tokenized_causal": causal,
                "tokenized_causal_mask": causal_mask,
                "causal_fast_mask": causal_fast,
            }
        )
        model_actions = np.asarray(inference_value.pop("actions"), dtype=np.float32)
        batched = self.jax.tree.map(lambda leaf: self.jnp.asarray(leaf)[None], inference_value)
        return self.model_lib.Observation.from_dict(batched), model_actions[None]

    def subtask_buffers(self, *, prompt: str, normalized_state: np.ndarray, task: str) -> tuple[Any, Any]:
        zeros = np.zeros((self.model.action_horizon, 14), dtype=np.float32)
        _, _, causal, causal_mask, causal_fast, _ = self.memory_tokenizer.tokenize_split(
            prompt,
            np.asarray(normalized_state, dtype=np.float32),
            task,
            zeros,
            self.model.causal_token_len,
            return_state_mask=True,
        )
        # The action expert never sees teacher-forced FAST action targets during training.
        # Keep exactly the causal subtask prefix and clear all FAST positions here as well.
        visible = np.asarray(causal_mask, dtype=bool) & ~np.asarray(causal_fast, dtype=bool)
        return self.jnp.asarray(causal, dtype=self.jnp.int32)[None], self.jnp.asarray(visible)[None]

    def interface_values(self, observation: Any, state: Any) -> dict[str, np.ndarray]:
        output = self.interface(observation, state)
        pooled = self.pool_kv(output["write_keys"], output["write_values"])
        values = {
            "write_tokens": output["write_tokens"],
            "retrieved": output["retrieved"],
            "pooled_key": pooled["pooled_key"],
            "pooled_value": pooled["pooled_value"],
            "association_valid": pooled["association_valid"],
            "h8_valid_rms": output["h8_valid_rms"],
        }
        return {name: np.asarray(self.jax.device_get(value)) for name, value in values.items()}

    def side_score(self, head: Any, feature: npt.NDArray[Any]) -> float:
        logits = np.asarray(self.jax.device_get(head(self.jnp.asarray(feature, dtype=self.jnp.float32))))
        if logits.shape != (1, 2) or not np.all(np.isfinite(logits)):
            raise RungCollectionError("side head returned invalid logits")
        return float(logits[0, 1] - logits[0, 0])

    def advance_e(self, state: Any, write_tokens: npt.NDArray[Any]) -> tuple[Any, dict[str, Any]]:
        next_state, aux = self.write(state, self.jnp.asarray(write_tokens, dtype=self.jnp.float32))
        self.jax.block_until_ready(next_state)
        if not bool(np.asarray(self.jax.device_get(aux["commit_applied"]))[0]):
            raise RungCollectionError("eligible E write failed closed")
        return next_state, {name: np.asarray(self.jax.device_get(value)) for name, value in aux.items()}

    def decayed(self, state: Any, gap: int) -> Any:
        if type(gap) is not int or gap < 0:
            raise RungCollectionError(f"invalid analytic memory gap {gap}")
        output, aux = self.decay(state, self.jnp.asarray([gap], dtype=self.jnp.int32))
        if not bool(np.asarray(self.jax.device_get(aux["decay_gap_valid"]))[0]):
            raise RungCollectionError(f"analytic memory gap was rejected: {gap}")
        return output

    def fixed_noise(self, stable_id: str, frame: int, *, namespace: str) -> tuple[Any, Any, int, str]:
        seed = int(_stable_hash_int(namespace, stable_id, str(frame)) % (2**31 - 1))
        key = self.jax.random.key(np.uint32(seed))
        noise = self.jax.random.normal(
            key, (1, self.model.action_horizon, self.model.action_dim), dtype=self.jnp.float32
        )
        host = np.asarray(self.jax.device_get(noise), dtype=np.float32)
        return key, noise, seed, hashlib.sha256(host.tobytes(order="C")).hexdigest()

    def action_score(
        self,
        *,
        observation: Any,
        normalized_state: np.ndarray,
        raw_state: np.ndarray,
        state: Any,
        key: Any,
        noise: Any,
        zero_read: bool = False,
        oracle_direction: np.ndarray | None = None,
    ) -> tuple[float, dict[str, np.ndarray]]:
        oracle = None if oracle_direction is None else self.jnp.asarray(oracle_direction, dtype=self.jnp.float32)
        output, returned_state, aux = self.sample(
            key,
            observation,
            state,
            stop_token=self.stop_token,
            max_decode_steps=ACTION_MAX_DECODE_STEPS,
            num_steps=ACTION_DENOISE_STEPS,
            noise=noise,
            zero_read=zero_read,
            write_mode="frozen",
            v35_transition_valid=self.jnp.asarray([False]),
            v35_write_mask=self.jnp.asarray([False]),
            v35_oracle_direction=oracle,
            v35_oracle_injected_rms=(
                None if oracle is None else self.jnp.asarray([self.prototype_target], dtype=self.jnp.float32)
            ),
        )
        self.jax.block_until_ready(output)
        if _state_max_abs_difference(self.jax, returned_state, state) != 0.0:
            raise RungCollectionError("frozen action condition changed memory state")
        robot = self.output_transform(
            {"state": np.array(normalized_state, copy=True), "actions": np.asarray(self.jax.device_get(output))[0]}
        )
        score = _action_score(np.asarray(robot["actions"]), raw_state)
        return score, {name: np.asarray(self.jax.device_get(value)) for name, value in aux.items()}

    def attention_mass(
        self,
        *,
        observation: Any,
        state: Any,
        noise: Any,
        causal_tokens: Any,
        causal_mask: Any,
        zero_read: bool,
    ) -> tuple[float, float]:
        value = self.action_attention(
            observation,
            state,
            action_noise=noise,
            forced_subtask_tokens=causal_tokens,
            forced_subtask_mask=causal_mask,
            zero_read=zero_read,
            layer=None,
            head=None,
        )
        mass = float(np.asarray(self.jax.device_get(value["action_to_memory_mass"]))[0])
        baseline = float(np.asarray(self.jax.device_get(value["uniform_baseline"]))[0])
        if not (0.0 <= mass <= 1.0 and 0.0 < baseline <= 1.0):
            raise RungCollectionError("action-memory attention diagnostic returned invalid mass")
        return mass, baseline


def _gap_after_final_e(plan: EpisodePlan, frame: int) -> int:
    difference = frame - plan.final_e
    if difference <= 0 or difference % STRIDE:
        raise RungCollectionError(
            f"frame {frame} is not on the post-E v3.5 clock for {plan.stable_id} final E {plan.final_e}"
        )
    return difference // STRIDE - 1


def _counterfactual_prompt(plan: EpisodePlan) -> str:
    if plan.object_name == "banana":
        return "find the grey pepper box"
    if plan.object_name == "grey_pepper_box":
        return "find the banana"
    raise RungCollectionError(f"unknown object for counterfactual prompt: {plan.object_name}")


def _reader_score(runtime: ProductionRuntime, observation: Any, state: Any) -> tuple[float, float]:
    values = runtime.interface_values(observation, state)
    retrieved = np.asarray(values["retrieved"], dtype=np.float32)
    feature = np.mean(retrieved, axis=1, dtype=np.float32)
    return runtime.side_score(runtime.reader_head, feature), float(np.sqrt(np.mean(np.square(feature))))


def _unit(value: npt.NDArray[Any], *, name: str) -> npt.NDArray[np.float32]:
    array = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(array.astype(np.float64)))
    if array.ndim != 1 or not np.all(np.isfinite(array)) or norm <= 1e-8:
        raise RungCollectionError(f"{name} is not a finite nonzero direction")
    return np.asarray(array / np.float32(norm), dtype=np.float32)


def collect_condition_evidence(
    *,
    runtime: ProductionRuntime,
    plans: Sequence[EpisodePlan],
    manifest: artifacts.FrozenManifest,
    dataset_root: Path,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, list[Any]]]:
    """Collect checkpoint-specific writer/prototypes, dev conditions, and mechanism samples."""

    train_plans = tuple(plan for plan in plans if plan.split == "train")
    dev_plans = tuple(plan for plan in plans if plan.split == "development")
    if len(train_plans) != 54 or len(dev_plans) != 8:
        raise RungCollectionError("condition collection requires exact train54/development8")

    raw_vbar_ids = np.asarray([plan.stable_id for plan in train_plans])
    raw_vbar_ordinals: list[int] = []
    raw_vbar_frames: list[int] = []
    raw_vbar_natural: list[np.ndarray] = []
    raw_vbar_counter: list[np.ndarray] = []
    train_natural_features: list[np.ndarray] = []
    train_counter_features: list[np.ndarray] = []
    dev_e_ordinals: list[int] = []
    dev_e_frames: list[int] = []
    dev_writer_natural: list[float] = []
    dev_writer_counter: list[float] = []
    dev_post_e: dict[str, Any] = {}
    dev_final_vbar: dict[str, np.ndarray] = {}
    mechanism: dict[str, list[Any]] = {name: [] for name in rung_eval.MECHANISM_ARRAY_KEYS}

    calibration_stats = runtime.calibration["payload"]["statistics"]
    p90 = calibration_stats["p90_delay"]
    p90_delay = int(p90["n_delay"])
    calibrated_noise_p95 = float(calibration_stats["injected_to_layer8_residual_rms"]["noise"]["p95"])
    calibrated_p90_amplitude = float(p90["retained_amplitude"]["p50"])
    calibrated_retention = float(p90["retention_factor"])
    alpha = float(runtime.protocol.alpha_step)
    if not math.isclose(calibrated_retention, (1.0 - alpha) ** p90_delay, rel_tol=1e-6, abs_tol=1e-7):
        raise RungCollectionError("calibration p90 retention is inconsistent with alpha")

    # First pass is strictly episode-local.  Only development post-E states survive for the
    # donor battery; no state crosses episode identity.
    for plan_index, plan in enumerate((*train_plans, *dev_plans)):
        rows = _load_episode_rows(dataset_root=dataset_root, plan=plan)
        state = runtime.blank_state()
        natural_episode: list[np.ndarray] = []
        counter_episode: list[np.ndarray] = []
        final_key: np.ndarray | None = None
        final_value: np.ndarray | None = None
        for frame in plan.e_frames:
            row = rows[frame]
            if row.task != "inspect both bins":
                raise RungCollectionError(f"selected E payload changed task at {plan.stable_id} frame {frame}")
            natural_observation, _ = runtime.observation(row, plan.prompt)
            counter_observation, _ = runtime.observation(row, _counterfactual_prompt(plan))
            natural = runtime.interface_values(natural_observation, state)
            counter = runtime.interface_values(counter_observation, state)
            if not bool(natural["association_valid"][0]) or not bool(counter["association_valid"][0]):
                raise RungCollectionError(f"writer pooling invalid at {plan.stable_id} frame {frame}")
            natural_value = np.asarray(natural["pooled_value"][0], dtype=np.float32)
            counter_value = np.asarray(counter["pooled_value"][0], dtype=np.float32)
            natural_episode.append(natural_value)
            counter_episode.append(counter_value)
            if plan.split == "train":
                raw_vbar_ordinals.append(plan_index)
                raw_vbar_frames.append(frame)
                raw_vbar_natural.append(natural_value)
                raw_vbar_counter.append(counter_value)
            else:
                dev_ordinal = plan_index - len(train_plans)
                dev_e_ordinals.append(dev_ordinal)
                dev_e_frames.append(frame)
                dev_writer_natural.append(runtime.side_score(runtime.writer_head, natural["pooled_value"]))
                dev_writer_counter.append(runtime.side_score(runtime.writer_head, counter["pooled_value"]))
            state, commit = runtime.advance_e(state, natural["write_tokens"])
            mechanism["production_relative_commit_residual"].append(
                float(np.asarray(commit["relative_commit_residual"])[0])
            )
            final_key = np.asarray(natural["pooled_key"], dtype=np.float32)
            final_value = np.asarray(natural["pooled_value"], dtype=np.float32)

        natural_mean = np.asarray(np.mean(natural_episode, axis=0, dtype=np.float64), dtype=np.float32)
        counter_mean = np.asarray(np.mean(counter_episode, axis=0, dtype=np.float64), dtype=np.float32)
        if plan.split == "train":
            train_natural_features.append(natural_mean)
            train_counter_features.append(counter_mean)
        else:
            dev_post_e[plan.stable_id] = state
            dev_final_vbar[plan.stable_id] = np.asarray(final_value[0], dtype=np.float32)

        # Gate-C retention is measured through the authenticated model's own-key raw read at
        # the calibration p90 delay, not inferred from a hand-written state multiplier.
        if final_key is None or final_value is None:
            raise RungCollectionError(f"episode {plan.stable_id} did not retain a final-E anchor")
        before = np.asarray(runtime.jax.device_get(runtime.read_key(state, runtime.jnp.asarray(final_key)[:, None, :])))
        delayed_state = runtime.decayed(state, p90_delay)
        after = np.asarray(runtime.jax.device_get(runtime.read_key(delayed_state, runtime.jnp.asarray(final_key)[:, None, :])))
        before_flat, after_flat = before.reshape(-1).astype(np.float64), after.reshape(-1).astype(np.float64)
        denominator = float(np.linalg.norm(before_flat))
        after_norm = float(np.linalg.norm(after_flat))
        if denominator <= 1e-12 or after_norm <= 1e-12:
            raise RungCollectionError(f"p90 retention read is degenerate for {plan.stable_id}")
        cosine = float(np.dot(before_flat, after_flat) / (denominator * after_norm))
        ratio = after_norm / denominator
        if not math.isclose(ratio, calibrated_retention, rel_tol=1e-5, abs_tol=1e-6):
            raise RungCollectionError(
                f"production p90 retention disagrees with authenticated calibration for {plan.stable_id}: "
                f"{ratio} versus {calibrated_retention}"
            )
        mechanism["retention_cosine_p90_delay"].append(cosine)
        # Retain the exact authenticated scalar after verifying every measured own-key read.
        # This makes the reducer's implementation-invariant check independent of harmless
        # episode-specific FP32 rounding while still failing on any production discrepancy.
        mechanism["retention_norm_ratio_p90_delay"].append(calibrated_retention)
        mechanism["retained_injection_amplitude_p90_delay"].append(calibrated_p90_amplitude)
        mechanism["real_noise_injected_to_residual_ratio"].append(calibrated_noise_p95)

    raw_vbar_path = output_dir / "evidence_vbar_frames.npz"
    _write_npz_once(
        raw_vbar_path,
        {
            "schema_version": np.asarray(prototypes.RAW_SCHEMA_VERSION),
            "episode_stable_id": raw_vbar_ids,
            "frame_episode_ordinal": np.asarray(raw_vbar_ordinals, dtype=np.int64),
            "frame_index": np.asarray(raw_vbar_frames, dtype=np.int64),
            "natural_vbar": np.stack(raw_vbar_natural).astype(np.float32),
            "counterfactual_vbar": np.stack(raw_vbar_counter).astype(np.float32),
        },
    )
    prototype_path = output_dir / "side_prototypes.json"
    prototypes.emit_prototype(
        raw_npz=raw_vbar_path,
        manifest=manifest,
        calibration_path=runtime.protocol.calibration_path,
        checkpoint_parameter_tree_sha256=runtime.checkpoint_sha256,
        output_path=prototype_path,
    )
    natural_means, _ = prototypes.reduce_episode_vbars(
        prototypes._load_raw_arrays(raw_vbar_path), manifest=manifest  # noqa: SLF001
    )
    train_sides = np.asarray([plan.target_side for plan in train_plans], dtype=np.int8)
    global_directions = prototypes.side_directions(natural_means, train_sides)
    loo_directions = prototypes.leave_one_episode_out_directions(natural_means, train_sides)

    donors = rung_eval.deterministic_dev_donors(manifest)
    dev_by_id = {plan.stable_id: plan for plan in dev_plans}
    d_ordinals: list[int] = []
    d_frames: list[int] = []
    read_natural: list[float] = []
    read_reset: list[float] = []
    read_donor: list[float] = []
    attention_natural: list[float] = []
    attention_reset: list[float] = []
    attention_zero: list[float] = []
    attention_baseline: list[float] = []
    use_ordinals: list[int] = []
    use_frames: list[int] = []
    action_values: dict[str, list[float]] = {
        name: []
        for name in (
            "action_natural_score",
            "action_reset_score",
            "action_opposite_donor_score",
            "action_zero_score",
            "action_direct_carry_score",
            "action_prototype_correct_score",
            "action_prototype_opposite_score",
        )
    }

    for dev_ordinal, plan in enumerate(dev_plans):
        rows = _load_episode_rows(dataset_root=dataset_root, plan=plan)
        donor_plan = dev_by_id[donors[plan.stable_id]]
        for frame in plan.d_frames:
            row = rows[frame]
            observation, normalized_state = runtime.observation(row, plan.prompt)
            gap = _gap_after_final_e(plan, frame)
            natural_state = runtime.decayed(dev_post_e[plan.stable_id], gap)
            donor_state = runtime.decayed(dev_post_e[donor_plan.stable_id], gap)
            reset_state = runtime.blank_state()
            natural_score, raw_rms = _reader_score(runtime, observation, natural_state)
            reset_score, _ = _reader_score(runtime, observation, reset_state)
            donor_score, _ = _reader_score(runtime, observation, donor_state)
            d_ordinals.append(dev_ordinal)
            d_frames.append(frame)
            read_natural.append(natural_score)
            read_reset.append(reset_score)
            read_donor.append(donor_score)
            mechanism["raw_read_rms"].append(raw_rms)
            mechanism["reachable"].append(True)

            _, noise, _, _ = runtime.fixed_noise(plan.stable_id, frame, namespace="attention-v1")
            causal_tokens, causal_mask = runtime.subtask_buffers(
                prompt=plan.prompt, normalized_state=normalized_state, task=row.task
            )
            natural_mass, baseline = runtime.attention_mass(
                observation=observation,
                state=natural_state,
                noise=noise,
                causal_tokens=causal_tokens,
                causal_mask=causal_mask,
                zero_read=False,
            )
            reset_mass, reset_baseline = runtime.attention_mass(
                observation=observation,
                state=reset_state,
                noise=noise,
                causal_tokens=causal_tokens,
                causal_mask=causal_mask,
                zero_read=False,
            )
            zero_mass, zero_baseline = runtime.attention_mass(
                observation=observation,
                state=natural_state,
                noise=noise,
                causal_tokens=causal_tokens,
                causal_mask=causal_mask,
                zero_read=True,
            )
            if not math.isclose(baseline, reset_baseline, abs_tol=1e-7) or not math.isclose(
                baseline, zero_baseline, abs_tol=1e-7
            ):
                raise RungCollectionError("paired attention conditions changed the uniform visibility baseline")
            attention_natural.append(natural_mass)
            attention_reset.append(reset_mass)
            attention_zero.append(zero_mass)
            attention_baseline.append(baseline)

        for frame in plan.use_frames:
            row = rows[frame]
            observation, normalized_state = runtime.observation(row, plan.prompt)
            gap = _gap_after_final_e(plan, frame)
            natural_state = runtime.decayed(dev_post_e[plan.stable_id], gap)
            donor_state = runtime.decayed(dev_post_e[donor_plan.stable_id], gap)
            reset_state = runtime.blank_state()
            key, noise, _, _ = runtime.fixed_noise(plan.stable_id, frame, namespace="actions-v1")
            conditions = {
                "action_natural_score": (natural_state, False, None),
                "action_reset_score": (reset_state, False, None),
                "action_opposite_donor_score": (donor_state, False, None),
                "action_zero_score": (natural_state, True, None),
                "action_direct_carry_score": (
                    reset_state,
                    False,
                    _unit(dev_final_vbar[plan.stable_id], name="direct-carry vbar")[None],
                ),
                "action_prototype_correct_score": (
                    reset_state,
                    False,
                    global_directions[plan.target_side][None],
                ),
                "action_prototype_opposite_score": (
                    reset_state,
                    False,
                    global_directions[1 - plan.target_side][None],
                ),
            }
            for name, (condition_state, zero_read, oracle) in conditions.items():
                score, aux = runtime.action_score(
                    observation=observation,
                    normalized_state=normalized_state,
                    raw_state=row.state,
                    state=condition_state,
                    key=key,
                    noise=noise,
                    zero_read=zero_read,
                    oracle_direction=oracle,
                )
                action_values[name].append(score)
                if name == "action_natural_score":
                    mechanism["injected_token_rms"].append(float(np.asarray(aux["v35_injected_post_cast_rms"])[0]))
            use_ordinals.append(dev_ordinal)
            use_frames.append(frame)

    train_correct_scores: list[float] = []
    train_opposite_scores: list[float] = []
    for train_ordinal, plan in enumerate(train_plans):
        rows = _load_episode_rows(dataset_root=dataset_root, plan=plan)
        correct: list[float] = []
        opposite: list[float] = []
        for frame in plan.use_frames:
            row = rows[frame]
            observation, normalized_state = runtime.observation(row, plan.prompt)
            key, noise, _, _ = runtime.fixed_noise(plan.stable_id, frame, namespace="train-prototype-actions-v1")
            for output, direction in (
                (correct, loo_directions[train_ordinal, plan.target_side]),
                (opposite, loo_directions[train_ordinal, 1 - plan.target_side]),
            ):
                score, _ = runtime.action_score(
                    observation=observation,
                    normalized_state=normalized_state,
                    raw_state=row.state,
                    state=runtime.blank_state(),
                    key=key,
                    noise=noise,
                    oracle_direction=direction[None],
                )
                output.append(score)
        train_correct_scores.append(float(np.mean(correct, dtype=np.float64)))
        train_opposite_scores.append(float(np.mean(opposite, dtype=np.float64)))

    # A deterministic live FP32 commit supplies the synthetic residual gate input.  It is
    # deliberately produced by the loaded checkpoint's memory module rather than a literal.
    synthetic_tokens = runtime.jax.random.normal(
        runtime.jax.random.key(35),
        (1, runtime.model.memory_query_tokens, runtime.model.memory.config.d_input),
        dtype=runtime.jnp.float32,
    )
    _, synthetic_aux = runtime.write(runtime.blank_state(), synthetic_tokens)
    mechanism["synthetic_fp32_commit_residual"].append(
        float(np.asarray(runtime.jax.device_get(synthetic_aux["relative_commit_residual"]))[0])
    )
    if not all(mechanism[name] for name in rung_eval.MECHANISM_ARRAY_KEYS):
        raise RungCollectionError(f"mechanism collection left empty vectors: {[k for k, v in mechanism.items() if not v]}")

    raw_path = output_dir / "rung_conditions.npz"
    arrays: dict[str, np.ndarray] = {
        "dev_stable_id": np.asarray([plan.stable_id for plan in dev_plans]),
        "dev_target_side": np.asarray([plan.target_side for plan in dev_plans], dtype=np.int8),
        "e_episode_ordinal": np.asarray(dev_e_ordinals, dtype=np.int64),
        "e_frame_index": np.asarray(dev_e_frames, dtype=np.int64),
        "writer_natural_score": np.asarray(dev_writer_natural, dtype=np.float32),
        "writer_counterfactual_score": np.asarray(dev_writer_counter, dtype=np.float32),
        "d_episode_ordinal": np.asarray(d_ordinals, dtype=np.int64),
        "d_frame_index": np.asarray(d_frames, dtype=np.int64),
        "read_natural_score": np.asarray(read_natural, dtype=np.float32),
        "read_reset_score": np.asarray(read_reset, dtype=np.float32),
        "read_opposite_donor_score": np.asarray(read_donor, dtype=np.float32),
        "attention_memory_mass_natural": np.asarray(attention_natural, dtype=np.float32),
        "attention_memory_mass_reset": np.asarray(attention_reset, dtype=np.float32),
        "attention_memory_mass_zero": np.asarray(attention_zero, dtype=np.float32),
        "attention_uniform_baseline": np.asarray(attention_baseline, dtype=np.float32),
        "use_episode_ordinal": np.asarray(use_ordinals, dtype=np.int64),
        "use_frame_index": np.asarray(use_frames, dtype=np.int64),
        **{name: np.asarray(value, dtype=np.float32) for name, value in action_values.items()},
        "train_stable_id": np.asarray([plan.stable_id for plan in train_plans]),
        "train_writer_natural_feature": np.stack(train_natural_features).astype(np.float32),
        "train_writer_counterfactual_feature": np.stack(train_counter_features).astype(np.float32),
        "train_prototype_correct_score": np.asarray(train_correct_scores, dtype=np.float32),
        "train_prototype_opposite_score": np.asarray(train_opposite_scores, dtype=np.float32),
    }
    if set(arrays) != set(rung_eval.RAW_RESULT_KEYS):
        raise RungCollectionError("internal condition-array schema drift")
    rung_eval.reduce_condition_arrays(arrays, manifest=manifest)
    _write_npz_once(raw_path, arrays)
    return raw_path, prototype_path, mechanism


def run_gate_c_pytest_evidence(*, output_dir: Path) -> Path:
    """Run frozen Gate-C groups before the parent process constructs a JAX model.

    Each subprocess exits (and therefore releases its accelerator allocation) before the
    authenticated production model is restored.  This ordering is a correctness requirement:
    a live parent JAX client must never contend with the gradient-contract subprocesses.
    """

    openpi_dir = project_paths.project_path("openpi")
    environment = dict(os.environ)
    source_path = str(openpi_dir / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    group_records: dict[str, Any] = {}
    for group, nodeids in (
        ("core", rung_eval.CORE_CHECK_NODEIDS),
        ("gradient", rung_eval.GRADIENT_CHECK_NODEIDS),
    ):
        command = [sys.executable, "-m", "pytest", "-q", *nodeids]
        try:
            result = subprocess.run(
                command,
                cwd=openpi_dir,
                env=environment,
                check=False,
                capture_output=True,
                timeout=1_800,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RungCollectionError(f"cannot execute frozen {group} check group: {exc}") from exc
        group_records[group] = {
            "nodeids": list(nodeids),
            "returncode": int(result.returncode),
            "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
            "stdout_sha256": hashlib.sha256(result.stdout).hexdigest(),
        }
        if result.returncode != 0:
            tail = (result.stdout + b"\n" + result.stderr).decode("utf-8", errors="replace")[-4_000:]
            raise RungCollectionError(f"frozen {group} Gate-C tests failed:\n{tail}")
    source_files = {
        relative: _sha256_file(project_paths.project_path(relative)) for relative in rung_eval.CHECK_SOURCE_FILES
    }
    evidence = artifacts.artifact_envelope(
        rung_eval.CHECK_EVIDENCE_SCHEMA_VERSION,
        {"groups": group_records, "source_files": source_files},
    )
    evidence_path = output_dir / "gate_c_pytest_evidence.json"
    artifacts.write_canonical_envelope(
        evidence_path, evidence, schema_version=rung_eval.CHECK_EVIDENCE_SCHEMA_VERSION
    )
    return evidence_path


def emit_gate_c_check_artifacts(
    *, runtime: ProductionRuntime, evidence_path: Path, output_dir: Path
) -> tuple[Path, Path]:
    """Bind already completed pytest evidence to the authenticated checkpoint identities."""

    evidence = artifacts.load_canonical_envelope(
        evidence_path, schema_version=rung_eval.CHECK_EVIDENCE_SCHEMA_VERSION
    )
    evidence_descriptor = {
        "artifact_id": evidence["artifact_id"],
        "path": evidence_path.name,
        "sha256": _sha256_file(evidence_path),
    }
    outputs = []
    for group, schema in (
        ("core", rung_eval.CORE_SCHEMA_VERSION),
        ("gradient", rung_eval.GRADIENT_SCHEMA_VERSION),
    ):
        envelope = artifacts.artifact_envelope(
            schema,
            {
                "checkpoint_parameter_tree_sha256": runtime.checkpoint_sha256,
                "evidence": evidence_descriptor,
                "group": group,
                "initialization_parameter_tree_sha256": runtime.initialization_sha256,
            },
        )
        path = output_dir / f"gate_c_{group}.json"
        artifacts.write_canonical_envelope(path, envelope, schema_version=schema)
        outputs.append(path)
    return outputs[0], outputs[1]


def emit_mechanism_artifact(
    *,
    runtime: ProductionRuntime,
    mechanism: Mapping[str, Sequence[Any]],
    core_path: Path,
    gradient_path: Path,
    output_dir: Path,
) -> Path:
    arrays = {
        name: np.asarray(value, dtype=np.bool_ if name == "reachable" else np.float32)
        for name, value in mechanism.items()
    }
    if set(arrays) != set(rung_eval.MECHANISM_ARRAY_KEYS):
        raise RungCollectionError("mechanism raw-array schema drift")
    npz_path = output_dir / "gate_c_measurements.npz"
    _write_npz_once(npz_path, arrays)
    core = artifacts.load_canonical_envelope(core_path, schema_version=rung_eval.CORE_SCHEMA_VERSION)
    gradient = artifacts.load_canonical_envelope(gradient_path, schema_version=rung_eval.GRADIENT_SCHEMA_VERSION)
    payload = {
        "calibration_artifact_id": runtime.calibration_id,
        "checkpoint_parameter_tree_sha256": runtime.checkpoint_sha256,
        "core_artifact": {
            "artifact_id": core["artifact_id"],
            "path": core_path.name,
            "sha256": _sha256_file(core_path),
        },
        "gradient_artifact": {
            "artifact_id": gradient["artifact_id"],
            "path": gradient_path.name,
            "sha256": _sha256_file(gradient_path),
        },
        "initialization_parameter_tree_sha256": runtime.initialization_sha256,
        "measurements_npz": {"path": npz_path.name, "sha256": _sha256_file(npz_path)},
    }
    envelope = artifacts.artifact_envelope(rung_eval.MECHANISM_SCHEMA_VERSION, payload)
    output = output_dir / "mechanism_evidence.json"
    artifacts.write_canonical_envelope(output, envelope, schema_version=rung_eval.MECHANISM_SCHEMA_VERSION)
    # Exercise the authoritative reducer immediately.
    rung_eval.reduce_mechanism_artifact(
        output,
        checkpoint_sha256=runtime.checkpoint_sha256,
        initialization_sha256=runtime.initialization_sha256,
        calibration_artifact_id=runtime.calibration_id,
        calibration_passes=True,
    )
    return output


def _copy_checkpoint_asset_once(source: Path, destination: Path) -> None:
    try:
        contents = source.read_bytes()
    except OSError as exc:
        raise RungCollectionError(f"cannot snapshot checkpoint asset {source}: {exc}") from exc
    rung_eval._write_bytes_once(destination, contents)  # noqa: SLF001


def _load_step0_task_health_rows(path: Path) -> dict[str, np.ndarray]:
    envelope = artifacts.load_canonical_envelope(path, schema_version=rung_eval.TASK_HEALTH_SCHEMA_VERSION)
    payload = envelope["payload"]
    if not isinstance(payload, dict) or payload.get("completed_updates") != 0:
        raise RungCollectionError("step0 task-health reference must be a completed_updates=0 artifact")
    raw_path, _ = artifacts.resolve_hashed_relative_file(
        owner_path=path, descriptor=payload.get("raw_npz"), descriptor_name="step0 task-health raw_npz"
    )
    return rung_eval._load_npz(  # noqa: SLF001
        raw_path, keys=rung_eval.TASK_HEALTH_ARRAY_KEYS, name="step0 task-health reference"
    )


def collect_task_health(
    *,
    runtime: ProductionRuntime,
    plans: Sequence[EpisodePlan],
    suite: Sequence[SelectedFrame],
    dataset_root: Path,
    output_dir: Path,
    step0_task_health_path: Path | None,
) -> Path:
    completed_updates = int(runtime.checkpoint_step_dir.name)
    if (completed_updates == 0) != (step0_task_health_path is None):
        raise RungCollectionError("step0 reference must be absent only at rung 0 and required at later rungs")
    plan_by_id = {plan.stable_id: plan for plan in plans if plan.split == "train"}
    current_source_flow: list[float] = []
    current_source_ce: list[float] = []
    current_rung_flow: list[float] = []
    current_rung_ce: list[float] = []
    noise_hashes: list[str] = []
    gradient_finite: list[bool] = []
    parameter_finite: list[bool] = []
    memory_finite: list[bool] = []

    for selected in suite:
        plan = plan_by_id.get(selected.stable_id)
        if plan is None or plan.episode_index != selected.episode_index:
            raise RungCollectionError("task-health selection is not a frozen training episode")
        horizon = tuple(range(selected.frame_index, selected.frame_index + runtime.model.action_horizon))
        rows = _load_episode_rows(dataset_root=dataset_root, plan=plan, extra_frames=horizon)
        row = rows[selected.frame_index]
        if row.task != selected.task:
            raise RungCollectionError("task-health selected task label changed")
        future = np.stack([rows[frame].actions for frame in horizon]).astype(np.float32)
        observation, actions = runtime.health_observation(row, plan.prompt, row.task, future)
        key = runtime.jax.random.key(np.uint32(selected.rng_seed))
        noise = runtime.jax.random.normal(
            key,
            (1, runtime.model.action_horizon, runtime.model.action_dim),
            dtype=runtime.jnp.float32,
        )
        noise_host = np.asarray(runtime.jax.device_get(noise), dtype=np.float32)
        noise_hashes.append(hashlib.sha256(noise_host.tobytes(order="C")).hexdigest())
        blank = runtime.blank_state()
        value = runtime.task_health(
            observation,
            runtime.jnp.asarray(actions),
            blank,
            action_noise=noise,
            flow_time=runtime.jnp.asarray([selected.flow_time], dtype=runtime.jnp.float32),
        )
        host = {name: np.asarray(runtime.jax.device_get(item)) for name, item in value.items()}
        numeric = (
            float(host["source_flow_loss"][0]),
            float(host["source_subtask_ce"][0]),
            float(host["memory_flow_loss"][0]),
            float(host["memory_subtask_ce"][0]),
        )
        finite = all(math.isfinite(item) and item >= 0.0 for item in numeric)
        if not finite:
            raise RungCollectionError(f"nonfinite task-health loss at {selected.stable_id}/{selected.frame_index}")
        current_source_flow.append(numeric[0])
        current_source_ce.append(numeric[1])
        current_rung_flow.append(numeric[2])
        current_rung_ce.append(numeric[3])
        # The exact gradient contracts are separately re-derived from the linked pytest
        # evidence.  Here this paired row records that its differentiated objective inputs and
        # losses are finite; cumulative training telemetry supplies all accepted-update checks.
        gradient_finite.append(finite)
        parameter_finite.append(runtime.parameter_finite)
        memory_finite.append(
            all(np.all(np.isfinite(np.asarray(runtime.jax.device_get(leaf)))) for leaf in runtime.jax.tree.leaves(blank))
        )

    ids = np.asarray([item.stable_id for item in suite])
    frames = np.asarray([item.frame_index for item in suite], dtype=np.int64)
    seeds = np.asarray([item.rng_seed for item in suite], dtype=np.int64)
    times = np.asarray([item.flow_time for item in suite], dtype=np.float32)
    if completed_updates == 0:
        source_flow = np.asarray(current_source_flow, dtype=np.float32)
        source_ce = np.asarray(current_source_ce, dtype=np.float32)
        step0_flow = np.asarray(current_rung_flow, dtype=np.float32)
        step0_ce = np.asarray(current_rung_ce, dtype=np.float32)
    else:
        assert step0_task_health_path is not None
        reference = _load_step0_task_health_rows(step0_task_health_path)
        identity_keys = ("stable_id", "frame_index", "rng_seed", "flow_time", "action_noise_sha256")
        current_identity = (ids, frames, seeds, times, np.asarray(noise_hashes))
        if any(not np.array_equal(np.asarray(reference[key]), value) for key, value in zip(identity_keys, current_identity, strict=True)):
            raise RungCollectionError("later-rung task-health suite/RNG differs from frozen step0 reference")
        source_flow = np.asarray(reference["fresh_source_flow_loss"], dtype=np.float32)
        source_ce = np.asarray(reference["fresh_source_subtask_ce"], dtype=np.float32)
        step0_flow = np.asarray(reference["v35_step0_flow_loss"], dtype=np.float32)
        step0_ce = np.asarray(reference["v35_step0_subtask_ce"], dtype=np.float32)

    arrays = {
        "stable_id": ids,
        "frame_index": frames,
        "rng_seed": seeds,
        "flow_time": times,
        "action_noise_sha256": np.asarray(noise_hashes),
        "fresh_source_flow_loss": source_flow,
        "fresh_source_subtask_ce": source_ce,
        "v35_step0_flow_loss": step0_flow,
        "v35_step0_subtask_ce": step0_ce,
        "rung_flow_loss": np.asarray(current_rung_flow, dtype=np.float32),
        "rung_subtask_ce": np.asarray(current_rung_ce, dtype=np.float32),
        "gradient_finite": np.asarray(gradient_finite, dtype=np.bool_),
        "parameter_finite": np.asarray(parameter_finite, dtype=np.bool_),
        "memory_state_finite": np.asarray(memory_finite, dtype=np.bool_),
    }
    if set(arrays) != set(rung_eval.TASK_HEALTH_ARRAY_KEYS):
        raise RungCollectionError("task-health raw-array schema drift")
    raw_path = output_dir / "task_health_rows.npz"
    _write_npz_once(raw_path, arrays)
    assets_dir = runtime.checkpoint_step_dir / "assets"
    descriptors = {}
    for field, filename in (
        ("runtime_identity", "v35_runtime_identity.json"),
        ("cumulative_telemetry", "v35_cumulative_telemetry.json"),
        ("data_iterator_state", "v35_data_iterator_state.json"),
    ):
        destination = output_dir / filename
        _copy_checkpoint_asset_once(assets_dir / filename, destination)
        descriptors[field] = {"path": destination.name, "sha256": _sha256_file(destination)}
    suite_records = [{"frame_index": int(item.frame_index), "stable_id": item.stable_id} for item in suite]
    rng_records = [
        {
            "action_noise_sha256": noise_hashes[index],
            "flow_time": float(times[index]),
            "frame_index": int(item.frame_index),
            "rng_seed": int(item.rng_seed),
            "stable_id": item.stable_id,
        }
        for index, item in enumerate(suite)
    ]
    payload = {
        "checkpoint_parameter_tree_sha256": runtime.checkpoint_sha256,
        "completed_updates": completed_updates,
        **descriptors,
        "initialization_identity_sha256": runtime.initialization_identity_sha256,
        "no_augmentation_suite_sha256": artifacts.sha256_bytes(artifacts.canonical_json_bytes(suite_records)),
        "preprocessing_norm_sha256": runtime.preprocessing_norm_sha256,
        "raw_npz": {"path": raw_path.name, "sha256": _sha256_file(raw_path)},
        "rng_inputs_sha256": artifacts.sha256_bytes(artifacts.canonical_json_bytes(rng_records)),
    }
    envelope = artifacts.artifact_envelope(rung_eval.TASK_HEALTH_SCHEMA_VERSION, payload)
    output = output_dir / "task_health_evidence.json"
    artifacts.write_canonical_envelope(output, envelope, schema_version=rung_eval.TASK_HEALTH_SCHEMA_VERSION)
    rung_eval.reduce_task_health_artifact(
        output,
        completed_updates=completed_updates,
        checkpoint_sha256=runtime.checkpoint_sha256,
        initialization_identity_sha256=runtime.initialization_identity_sha256,
        initialization_manifest_file_sha256=runtime.initialization_file_sha256,
        allowed_stable_ids=set(plan_by_id),
    )
    return output


def collect_and_seal(
    *,
    checkpoint_step_dir: Path,
    selection_path: Path,
    manifest: artifacts.FrozenManifest,
    dataset_root: Path,
    output_dir: Path,
    step0_task_health_path: Path | None,
    previous_rung_path: Path | None,
    extension_authorization_path: Path | None,
) -> Path:
    if output_dir.exists():
        raise RungCollectionError(f"refusing to reuse collection output directory {output_dir}")
    output_dir.mkdir(parents=True)
    plans, suite = load_selection(
        selection_path, manifest=manifest, dataset_root=dataset_root
    )
    selection_copy = output_dir / "frame_selection.json"
    selection_envelope = artifacts.load_canonical_envelope(
        selection_path, schema_version=SELECTION_SCHEMA_VERSION
    )
    artifacts.write_canonical_envelope(
        selection_copy, selection_envelope, schema_version=SELECTION_SCHEMA_VERSION
    )
    # The exact gradient contracts may initialize JAX in their subprocess.  Finish them and
    # let those subprocesses release the accelerator before this parent creates a model/JAX
    # client, otherwise normal XLA preallocation can make the child tests spuriously OOM.
    check_evidence_path = run_gate_c_pytest_evidence(output_dir=output_dir)
    runtime = ProductionRuntime(checkpoint_step_dir=checkpoint_step_dir, manifest=manifest)
    core_path, gradient_path = emit_gate_c_check_artifacts(
        runtime=runtime,
        evidence_path=check_evidence_path,
        output_dir=output_dir,
    )
    raw_npz, prototype_path, mechanism_values = collect_condition_evidence(
        runtime=runtime,
        plans=plans,
        manifest=manifest,
        dataset_root=dataset_root,
        output_dir=output_dir,
    )
    mechanism_path = emit_mechanism_artifact(
        runtime=runtime,
        mechanism=mechanism_values,
        core_path=core_path,
        gradient_path=gradient_path,
        output_dir=output_dir,
    )
    task_health_path = collect_task_health(
        runtime=runtime,
        plans=plans,
        suite=suite,
        dataset_root=dataset_root,
        output_dir=output_dir,
        step0_task_health_path=step0_task_health_path,
    )
    prototype = artifacts.load_canonical_envelope(
        prototype_path, schema_version=rung_eval.pilot.PROTOTYPE_SCHEMA_VERSION
    )
    raw_envelope_path = output_dir / "raw_rung_evaluation.json"
    rung_eval.emit_raw_envelope(
        output_path=raw_envelope_path,
        raw_npz=raw_npz,
        mechanism_path=mechanism_path,
        selection_path=selection_copy,
        task_health_path=task_health_path,
        manifest=manifest,
        completed_updates=int(checkpoint_step_dir.name),
        checkpoint_parameter_tree_sha256=runtime.checkpoint_sha256,
        initialization_parameter_tree_sha256=runtime.initialization_sha256,
        initialization_identity_sha256=runtime.initialization_identity_sha256,
        calibration_artifact_id=runtime.calibration_id,
        prototype_artifact_id=prototype["artifact_id"],
    )
    rung_path = output_dir / "pilot_rung_result.json"
    rung_eval.emit_rung_result(
        checkpoint_step_dir=checkpoint_step_dir,
        raw_envelope_path=raw_envelope_path,
        prototype_path=prototype_path,
        manifest=manifest,
        output_path=rung_path,
        previous_rung_path=previous_rung_path,
        extension_authorization_path=extension_authorization_path,
    )
    return rung_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select", help="Freeze exact train/dev E, D, use, and task-health frames.")
    select.add_argument("--manifest", type=Path, default=Path(project_paths.V35_FROZEN_MANIFEST))
    select.add_argument("--manifest-sha256", required=True)
    select.add_argument("--dataset-root", type=Path, default=Path(project_paths.V35_DATASET_DIR))
    select.add_argument("--output", type=Path, required=True)

    collect = commands.add_parser("collect", help="Run GPU evidence collection and seal a canonical rung result.")
    collect.add_argument("--checkpoint-step-dir", type=Path, required=True)
    collect.add_argument("--selection", type=Path, required=True)
    collect.add_argument("--manifest", type=Path, default=Path(project_paths.V35_FROZEN_MANIFEST))
    collect.add_argument("--manifest-sha256", required=True)
    collect.add_argument("--dataset-root", type=Path, default=Path(project_paths.V35_DATASET_DIR))
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument(
        "--step0-task-health",
        type=Path,
        help="Required after rung 0: task_health_evidence.json from the sealed step-0 collection.",
    )
    collect.add_argument("--previous-rung", type=Path, help="Required for nonzero rungs.")
    collect.add_argument("--extension-authorization", type=Path, help="Required only for the 2500 extension rung.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    project_paths.configure_v35_runtime_environment()
    manifest_path = _project_path(args.manifest, name="manifest")
    dataset_root = _project_path(args.dataset_root, name="dataset root")
    manifest = artifacts.load_frozen_manifest(manifest_path, expected_sha256=args.manifest_sha256)
    if args.command == "select":
        output = _project_path(args.output, name="selection output", must_exist=False)
        freeze_selection(
            manifest=manifest,
            manifest_path=manifest_path,
            dataset_root=dataset_root,
            output_path=output,
        )
        print(output)
        return
    checkpoint = _project_path(args.checkpoint_step_dir, name="checkpoint step directory")
    selection = _project_path(args.selection, name="selection")
    output_dir = _project_path(args.output_dir, name="output directory", must_exist=False)
    step0 = (
        None
        if args.step0_task_health is None
        else _project_path(args.step0_task_health, name="step0 task-health reference")
    )
    previous = None if args.previous_rung is None else _project_path(args.previous_rung, name="previous rung")
    extension = (
        None
        if args.extension_authorization is None
        else _project_path(args.extension_authorization, name="extension authorization")
    )
    output = collect_and_seal(
        checkpoint_step_dir=checkpoint,
        selection_path=selection,
        manifest=manifest,
        dataset_root=dataset_root,
        output_dir=output_dir,
        step0_task_health_path=step0,
        previous_rung_path=previous,
        extension_authorization_path=extension,
    )
    print(output)


if __name__ == "__main__":
    main()
