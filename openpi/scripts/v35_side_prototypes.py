"""Seal checkpoint-specific, train-only v3.5 side prototypes.

The input is a raw feature NPZ containing every eligible evidence-frame ``v_bar`` for the
frozen 54-episode training split.  This reducer performs the episode-first aggregation itself,
constructs the two unit directions, and emits the exact artifact consumed by
``v35_pilot_gate.load_rung_result``.  Development and final-test IDs are rejected before any
array is reduced.

All CLI paths are confined to ``memory_project`` and outputs are create-only.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
import sys
import tempfile
from typing import Any
import zipfile

import numpy as np
from numpy import typing as npt

from openpi.shared import project_paths

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import v35_gate_artifacts as artifacts  # noqa: E402
import v35_pilot_gate as pilot  # noqa: E402

RAW_SCHEMA_VERSION = "openpi.v35.evidence-vbar-frames.v1"
RAW_KEYS = (
    "schema_version",
    "episode_stable_id",
    "frame_episode_ordinal",
    "frame_index",
    "natural_vbar",
    "counterfactual_vbar",
)
PROTOTYPE_KEYS = ("episode_stable_id", "episode_mean_vbar", "left_direction", "right_direction")
NPZ_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


class SidePrototypeError(artifacts.GateArtifactError):
    """Raised when prototype inputs are incomplete, non-training, or mutable."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise SidePrototypeError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _project_path(value: Path, *, name: str, must_exist: bool = True) -> Path:
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise SidePrototypeError(f"{name} must be relative to memory_project")
    try:
        path = project_paths.project_path(raw)
    except project_paths.ProjectRootError as exc:
        raise SidePrototypeError(f"invalid {name}: {exc}") from exc
    if must_exist and not path.exists():
        raise SidePrototypeError(f"{name} does not exist: {raw.as_posix()}")
    return path


def _text_scalar(array: npt.NDArray[Any], *, name: str) -> str:
    value = np.asarray(array)
    if value.size != 1 or value.dtype.kind not in ("S", "U"):
        raise SidePrototypeError(f"{name} must be one string scalar")
    item = value.reshape(-1)[0]
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def _load_raw_arrays(path: Path) -> dict[str, npt.NDArray[Any]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(RAW_KEYS):
                raise SidePrototypeError(
                    f"raw vbar NPZ keys mismatch: missing={sorted(set(RAW_KEYS) - set(archive.files))}, "
                    f"extra={sorted(set(archive.files) - set(RAW_KEYS))}"
                )
            arrays = {name: np.asarray(archive[name]) for name in RAW_KEYS}
    except (OSError, ValueError) as exc:
        if isinstance(exc, SidePrototypeError):
            raise
        raise SidePrototypeError(f"cannot load raw vbar NPZ {path}: {exc}") from exc
    if _text_scalar(arrays["schema_version"], name="schema_version") != RAW_SCHEMA_VERSION:
        raise SidePrototypeError(f"raw vbar NPZ must use {RAW_SCHEMA_VERSION}")
    return arrays


def reduce_episode_vbars(
    arrays: Mapping[str, npt.NDArray[Any]],
    *,
    manifest: artifacts.FrozenManifest,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Return natural/counterfactual episode means in exact frozen train order."""

    expected_ids = [episode.stable_id for episode in manifest.split("train")]
    ids_array = np.asarray(arrays["episode_stable_id"])
    ids = [item.decode() if isinstance(item, bytes) else str(item) for item in ids_array]
    if ids_array.ndim != 1 or ids_array.dtype.kind not in ("S", "U") or ids != expected_ids:
        raise SidePrototypeError("raw vbar episode IDs must exactly match frozen train order")

    ordinal = np.asarray(arrays["frame_episode_ordinal"])
    frame = np.asarray(arrays["frame_index"])
    natural = np.asarray(arrays["natural_vbar"])
    counter = np.asarray(arrays["counterfactual_vbar"])
    rows = len(ordinal)
    if (
        not np.issubdtype(ordinal.dtype, np.integer)
        or ordinal.shape != (rows,)
        or not np.issubdtype(frame.dtype, np.integer)
        or frame.shape != (rows,)
        or natural.dtype != np.dtype(np.float32)
        or counter.dtype != np.dtype(np.float32)
        or natural.ndim != 2
        or counter.shape != natural.shape
        or natural.shape[0] != rows
        or natural.shape[1] < 1
        or rows < len(expected_ids)
        or not np.all(np.isfinite(natural))
        or not np.all(np.isfinite(counter))
    ):
        raise SidePrototypeError("raw vbar frame arrays must be aligned finite FP32 matrices")
    if np.any(ordinal < 0) or np.any(ordinal >= len(expected_ids)) or np.any(frame < 0):
        raise SidePrototypeError("raw vbar frame ordinals/indices are out of range")
    ordering = np.lexsort((frame.astype(np.int64), ordinal.astype(np.int64)))
    if not np.array_equal(ordering, np.arange(rows)):
        raise SidePrototypeError("raw vbar rows must be sorted by episode ordinal then raw frame")

    natural_means: list[npt.NDArray[np.float32]] = []
    counter_means: list[npt.NDArray[np.float32]] = []
    for episode_ordinal in range(len(expected_ids)):
        selected = ordinal == episode_ordinal
        if not np.any(selected):
            raise SidePrototypeError(f"training episode {expected_ids[episode_ordinal]!r} has zero eligible E vbars")
        episode_frames = frame[selected]
        if len({int(value) for value in episode_frames}) != len(episode_frames):
            raise SidePrototypeError(f"training episode {expected_ids[episode_ordinal]!r} repeats an E frame")
        # Float64 accumulation makes the episode mean invariant to collector batch size.  The
        # only stored representation is the explicit FP32 cast required by the prototype spec.
        natural_means.append(np.asarray(np.mean(natural[selected], axis=0, dtype=np.float64), dtype=np.float32))
        counter_means.append(np.asarray(np.mean(counter[selected], axis=0, dtype=np.float64), dtype=np.float32))
    return np.stack(natural_means), np.stack(counter_means)


def side_directions(
    episode_means: npt.NDArray[np.float32],
    target_sides: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.float32]:
    """Episode-macro side means followed by one L2 normalization per side."""

    means = np.asarray(episode_means)
    sides = np.asarray(target_sides)
    if means.dtype != np.dtype(np.float32) or means.ndim != 2 or sides.shape != (len(means),):
        raise SidePrototypeError("episode means/sides have invalid dtype or shape")
    output = []
    for side in (0, 1):
        selected = means[sides == side]
        if not len(selected):
            raise SidePrototypeError(f"prototype population has no side-{side} episodes")
        direction = np.mean(selected, axis=0, dtype=np.float32)
        norm = float(np.linalg.norm(direction.astype(np.float64)))
        if not np.isfinite(norm) or norm <= np.finfo(np.float32).tiny:
            raise SidePrototypeError(f"side-{side} prototype is degenerate")
        output.append(np.asarray(direction / np.float32(norm), dtype=np.float32))
    return np.stack(output)


def leave_one_episode_out_directions(
    episode_means: npt.NDArray[np.float32],
    target_sides: npt.NDArray[np.integer[Any]],
) -> npt.NDArray[np.float32]:
    """Return ``[episode, requested_side, channel]`` train-only LOO directions."""

    means = np.asarray(episode_means)
    sides = np.asarray(target_sides)
    outputs = np.empty((len(means), 2, means.shape[1]), dtype=np.float32)
    for index in range(len(means)):
        keep = np.arange(len(means)) != index
        outputs[index] = side_directions(means[keep], sides[keep])
    return outputs


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{name}.npy", date_time=NPZ_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    return info


def _write_npz_once(path: Path, arrays: Mapping[str, npt.NDArray[Any]]) -> str:
    if tuple(arrays) != PROTOTYPE_KEYS:
        raise SidePrototypeError(f"prototype NPZ member order must be {PROTOTYPE_KEYS}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SidePrototypeError(f"refusing to overwrite prototype NPZ {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    Path(temporary_name).unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(temporary_name, mode="x", allowZip64=True) as archive:
            for name in PROTOTYPE_KEYS:
                with archive.open(_zip_info(name), mode="w", force_zip64=True) as stream:
                    np.lib.format.write_array(stream, np.ascontiguousarray(arrays[name]), version=(2, 0), allow_pickle=False)
        try:
            Path(temporary_name).replace(path)
        except FileExistsError as exc:
            raise SidePrototypeError(f"refusing to overwrite prototype NPZ {path}") from exc
    finally:
        Path(temporary_name).unlink(missing_ok=True)
    return _sha256_file(path)


def _calibration_identity(path: Path) -> tuple[str, float]:
    try:
        raw = path.read_bytes()
        import json

        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise SidePrototypeError(f"cannot read calibration {path}: {exc}") from exc
    payload = value.get("payload") if isinstance(value, dict) else None
    if not isinstance(payload, dict):
        raise SidePrototypeError("calibration has no payload")
    digest = artifacts.sha256_bytes(artifacts.canonical_json_bytes(payload))
    calibration_id = f"sha256:{digest}"
    if (
        value.get("artifact_sha256") != digest
        or value.get("calibration_id") != calibration_id
        or payload.get("schema_version") != "openpi.v35.injection-calibration.v1"
        or payload.get("status") != "pass"
        or payload.get("gates", {}).get("passes") is not True
    ):
        raise SidePrototypeError("calibration envelope/hash/status is invalid")
    target = payload.get("parameters", {}).get("prototype_injected_rms_target")
    if isinstance(target, bool) or not isinstance(target, int | float) or not np.isfinite(target) or target <= 0:
        raise SidePrototypeError("calibration has no positive prototype injected-RMS target")
    return calibration_id, float(target)


def emit_prototype(
    *,
    raw_npz: Path,
    manifest: artifacts.FrozenManifest,
    calibration_path: Path,
    checkpoint_parameter_tree_sha256: str,
    output_path: Path,
) -> Path:
    """Reduce raw E features and create the reducer-compatible prototype artifact."""

    artifacts.require_sha256("checkpoint_parameter_tree_sha256", checkpoint_parameter_tree_sha256)
    arrays = _load_raw_arrays(raw_npz)
    means, _ = reduce_episode_vbars(arrays, manifest=manifest)
    train = manifest.split("train")
    sides = np.asarray([episode.target_side for episode in train], dtype=np.int8)
    directions = side_directions(means, sides)
    calibration_id, target = _calibration_identity(calibration_path)

    output_path = Path(output_path)
    if output_path.exists():
        raise SidePrototypeError(f"refusing to overwrite prototype artifact {output_path}")
    raw_npz = Path(raw_npz)
    if raw_npz.resolve().parent != output_path.resolve().parent:
        raise SidePrototypeError("raw evidence NPZ and prototype artifact must share one immutable directory")
    raw_npz_sha256 = _sha256_file(raw_npz)
    npz_path = output_path.with_suffix(".npz")
    npz_sha256 = _write_npz_once(
        npz_path,
        {
            "episode_stable_id": np.asarray([episode.stable_id for episode in train]),
            "episode_mean_vbar": means,
            "left_direction": directions[0],
            "right_direction": directions[1],
        },
    )
    payload = {
        "calibration_artifact_id": calibration_id,
        "checkpoint_parameter_tree_sha256": checkpoint_parameter_tree_sha256,
        "construction_protocol": pilot.PROTOTYPE_CONSTRUCTION,
        "directions_npz": {"path": npz_path.name, "sha256": npz_sha256},
        "episode_manifest_sha256": manifest.sha256,
        "prototype_injected_rms_target": target,
        "source_evidence_npz": {"path": raw_npz.name, "sha256": raw_npz_sha256},
        "split_assignment_sha256": manifest.split_assignment_sha256,
        "training_stable_ids": [episode.stable_id for episode in train],
    }
    envelope = artifacts.artifact_envelope(pilot.PROTOTYPE_SCHEMA_VERSION, payload)
    artifacts.write_canonical_envelope(output_path, envelope, schema_version=pilot.PROTOTYPE_SCHEMA_VERSION)
    return output_path


def _checkpoint_hash(params_path: Path) -> str:
    from openpi.models import model as model_lib
    from openpi.training import weight_loaders

    params = model_lib.restore_params(params_path, restore_type=np.ndarray)
    return weight_loaders.parameter_tree_sha256(params)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-npz", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(project_paths.V35_FROZEN_MANIFEST))
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--checkpoint-params", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    project_paths.configure_v35_runtime_environment()
    manifest_path = _project_path(args.manifest, name="manifest")
    raw_path = _project_path(args.raw_npz, name="raw NPZ")
    calibration_path = _project_path(args.calibration, name="calibration")
    params_path = _project_path(args.checkpoint_params, name="checkpoint params")
    output_path = _project_path(args.output, name="output", must_exist=False)
    manifest = artifacts.load_frozen_manifest(manifest_path, expected_sha256=args.manifest_sha256)
    emit_prototype(
        raw_npz=raw_path,
        manifest=manifest,
        calibration_path=calibration_path,
        checkpoint_parameter_tree_sha256=_checkpoint_hash(params_path),
        output_path=output_path,
    )
    print(output_path)


if __name__ == "__main__":
    main()
