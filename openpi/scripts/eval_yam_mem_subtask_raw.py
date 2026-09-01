"""Evaluate a memory checkpoint (pi05_yam_mem_*) on a RAW YAM demo, threading the Titans memory
through exactly one episode.

Like eval_yam_subtask_raw.py, the demo's mp4s/npys are read directly and pushed through the
config's own inference-time transform pipeline (SplitMemoryWindow is dataset-only and skipped).
`Pi0.sample_with_memory` runs sequentially every `stride` raw frames (default: the config's
memory_stride_frames, i.e. the training memory clock). Legacy v3-v3.4 checkpoints retain their
historical every-call write behavior. v3.5 instead requires a SHA-pinned episode manifest and
SHA-pinned semantic-label sidecar: every valid sampled step transitions, but only tail-guarded
evidence (E) frames receive `v35_write_mask=True`; O/D frames decay only. v3.5 is sealed to one
step per exactly 15 raw frames and the memory is freshly reset at this episode's start.

Outputs go to ``scripts/eval_results`` for legacy models and to the project-local
``v35/diagnostics/eval_yam_mem_subtask_raw`` directory for v3.5:
  1. an mp4 of every raw top-camera frame (30 fps) with the held predicted subtask + surprise
     overlaid and the surprise-vs-frame plot underneath (cursor bar at the current frame, red
     dot flashing on write frames),
  2. a per-joint png of the overlapping predicted action chunks vs the recorded teleop control,
  3. the raw curves as npz and the per-prediction subtasks as txt,
plus the predicted-subtask timeline, the memory-gate norm and per-call latency on stdout.

Run from the repo root on the GPU machine:
    CUDA_VISIBLE_DEVICES=<free> uv run python scripts/eval_yam_mem_subtask_raw.py
"""

import dataclasses
import hashlib
import json
import pathlib
import re
import subprocess
import time
from typing import Any

import matplotlib as mpl
import numpy as np

# Arrow must load before OpenPI/JAX in the installed native stack (the same ordering enforced
# by the repository-wide pytest bootstrap); otherwise this standalone CLI can exit in libarrow.
import pyarrow.parquet as _pyarrow_parquet  # noqa: F401

mpl.use("Agg")
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.shared.project_paths as _project_paths
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

CKPT = _project_paths.project_path("openpi/checkpoints/pi05_yam_mem_warmup/mem_warmup_v5_layer_8/2000")
RAW_DEMO = _project_paths.project_path("data/held_out_eval/demo1")
STRIDE = 0  # frames between predictions; 0 = the config's memory_stride_frames (cadence-matched default)
MAX_DECODE_STEPS = 10
FPS = 30  # recording rate of the raw mp4s
PLOT_H = 320  # pixel height of the surprise plot rendered under the camera frame
LOG_Y = True  # log-scale surprise axis (the interesting structure lives near 0)
FLASH = 6  # frames the write marker stays lit after each prediction
JOINT_NAMES = [f"{arm} {j}" for arm in ("left", "right") for j in (*range(6), "grip")]
V35_STRIDE_FRAMES = 15
V35_E_TAIL_GUARD_FRAMES = 5
V35_MANIFEST_SCHEMA_VERSION = _data_loader._V35_MANIFEST_SCHEMA_VERSION  # noqa: SLF001
V35_SPLIT_ALGORITHM = _data_loader._V35_SPLIT_ALGORITHM  # noqa: SLF001
V35_SPLIT_ALGORITHM_SHA256 = _data_loader._V35_SPLIT_ALGORITHM_SHA256  # noqa: SLF001
V35_D_VALID_DETECTOR = _data_loader._V35_D_VALID_DETECTOR  # noqa: SLF001
V35_OBJECT_PROMPTS = _data_loader._V35_OBJECT_PROMPTS  # noqa: SLF001


@dataclasses.dataclass
class Args:
    ckpt_dir: pathlib.Path = CKPT
    raw_demo: pathlib.Path = RAW_DEMO
    stride: int = STRIDE
    max_decode_steps: int = MAX_DECODE_STEPS
    config: str = "pi05_yam_mem_v3"
    # v3.5 resolves this from DataConfig by default and requires the corresponding SHA-256.
    # The manifest record must in turn pin its semantic label sidecar with `label_sha256`.
    manifest: pathlib.Path | None = None
    manifest_sha256: str | None = None
    # A/B control: never thread the writes, so every prediction reads the blank (m0) memory.
    # If the subtask timeline matches the normal run, the episode memory contributed nothing.
    ablate_memory: bool = False
    # Second control: thread the writes normally but force the content gate to zero, so the
    # memory tokens are exact zero embeddings -- what the vision path alone predicts through
    # the (degenerate) readout position.
    zero_gate: bool = False


@dataclasses.dataclass(frozen=True)
class V35EpisodeProtocol:
    """Trusted current-frame transition schedule for one raw episode."""

    stable_id: str
    split: str
    collection: str
    part: str
    object_name: str
    prompt: str
    manifest_path: pathlib.Path
    manifest_sha256: str
    label_path: pathlib.Path
    label_sha256: str
    expected_num_frames: int
    phase_by_frame: tuple[str, ...]
    write_frames: tuple[int, ...]
    d_valid_start: int
    d_valid_end: int


@dataclasses.dataclass(frozen=True)
class V35CheckpointProtocol:
    """Authenticated immutable v3.5 calibration embedded in one checkpoint."""

    assets_dir: pathlib.Path
    calibration_path: pathlib.Path
    calibration_id: str
    manifest_path: pathlib.Path
    manifest_sha256: str
    memory_injection_c: float
    memory_injection_tau: float
    alpha_step: float
    gate_target: float
    gate_atol: float
    raw_gate_sha256: str
    norm_stats_sha256: str


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"v3.5 protocol JSON contains duplicate key {key!r}.")
        result[key] = value
    return result


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalize_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} is required for trusted v3.5 inference.")
    digest = value.removeprefix("sha256:")
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"{name} must be 64 lower-case hexadecimal characters.")
    return digest


def _strict_int(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _read_json(path: pathlib.Path, *, name: str) -> Any:
    try:
        return json.loads(path.read_text(), object_pairs_hook=_strict_json_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {path}: {exc}") from exc


def _canonical_json_bytes(value: Any, *, ensure_ascii: bool) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=ensure_ascii,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"v3.5 provenance is not canonical-JSON serializable: {exc}") from exc


def _load_self_hashed_json(path: pathlib.Path, *, hash_key: str, name: str) -> dict[str, Any]:
    value = _read_json(path, name=name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object: {path}.")
    recorded = value.get(hash_key)
    unsigned = {key: item for key, item in value.items() if key != hash_key}
    actual = hashlib.sha256(_canonical_json_bytes(unsigned, ensure_ascii=True)).hexdigest()
    if recorded != actual:
        raise ValueError(f"{name} self-hash is invalid: {path}.")
    return value


def _positive_finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a positive finite number.")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return result


def _load_v35_checkpoint_protocol(
    ckpt_dir: pathlib.Path,
    *,
    expected_config_name: str,
    expected_seed: int,
    expected_value_width: int,
) -> V35CheckpointProtocol:
    """Authenticate the checkpoint-owned calibration before constructing the model."""
    assets = ckpt_dir.expanduser().resolve() / "assets"
    paths = {
        "calibration": assets / "v35_calibration_artifact.json",
        "manifest": assets / "v35_episode_manifest.json",
        "norm_provenance": assets / "v35_norm_stats_provenance.json",
        "graft": assets / "v35_initialization_graft_manifest.json",
        "identity": assets / "v35_initialization_manifest.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"v3.5 checkpoint is missing embedded provenance assets: {missing}.")

    calibration = _read_json(paths["calibration"], name="v3.5 calibration artifact")
    if not isinstance(calibration, dict) or not isinstance(calibration.get("payload"), dict):
        raise ValueError("v3.5 calibration artifact must contain a payload object.")
    payload = calibration["payload"]
    calibration_digest = hashlib.sha256(_canonical_json_bytes(payload, ensure_ascii=False)).hexdigest()
    calibration_id = f"sha256:{calibration_digest}"
    if calibration.get("artifact_sha256") != calibration_digest or calibration.get("calibration_id") != calibration_id:
        raise ValueError("v3.5 calibration artifact payload hash/ID is invalid.")
    if payload.get("schema_version") != "openpi.v35.injection-calibration.v1" or payload.get("status") != "pass":
        raise ValueError("v3.5 calibration artifact has the wrong schema or is not passing.")

    identity = _load_self_hashed_json(
        paths["identity"], hash_key="identity_sha256", name="v3.5 initialization identity"
    )
    graft = _load_self_hashed_json(paths["graft"], hash_key="manifest_sha256", name="v3.5 graft manifest")
    artifact_hashes = identity.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict):
        raise ValueError("v3.5 initialization identity is missing artifact_hashes.")
    manifest_sha256 = _sha256(paths["manifest"])
    embedded_hashes = {
        "calibration_artifact_sha256": _sha256(paths["calibration"]),
        "episode_manifest_sha256": manifest_sha256,
        "norm_stats_provenance_sha256": _sha256(paths["norm_provenance"]),
        "initialization_graft_manifest_file_sha256": _sha256(paths["graft"]),
        "initialization_graft_manifest_self_sha256": graft["manifest_sha256"],
    }
    if any(artifact_hashes.get(key) != value for key, value in embedded_hashes.items()):
        raise ValueError("v3.5 checkpoint provenance assets do not match initialization identity hashes.")

    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("v3.5 calibration artifact is missing parameters.")
    injection_c = _positive_finite_float(parameters.get("memory_injection_c"), name="memory_injection_c")
    injection_tau = _positive_finite_float(parameters.get("memory_injection_tau"), name="memory_injection_tau")
    alpha_step = _positive_finite_float(parameters.get("alpha_step"), name="alpha_step")
    expected_calibration = {
        "alpha_step": alpha_step,
        "memory_injection_c": injection_c,
        "memory_injection_tau": injection_tau,
    }
    if (
        identity.get("format_version") != 2
        or identity.get("config_name") != expected_config_name
        or identity.get("official_source_uri") != "gs://openpi-assets/checkpoints/pi05_base/params"
        or identity.get("initialization_seed") != expected_seed
        or identity.get("calibration_id") != calibration_id
        or identity.get("memory_calibration") != expected_calibration
    ):
        raise ValueError("v3.5 checkpoint initialization identity/config/calibration binding is invalid.")

    provenance = payload.get("provenance")
    tree_hashes = graft.get("tree_hashes")
    if not isinstance(provenance, dict) or not isinstance(tree_hashes, dict):
        raise ValueError("v3.5 calibration/graft provenance is incomplete.")
    calibration_step0_sha256 = _normalize_sha256(
        provenance.get("source_sha256"), name="calibration provenance source_sha256"
    )
    calibration_base_sha256 = _normalize_sha256(
        provenance.get("official_base_source_sha256"),
        name="calibration provenance official_base_source_sha256",
    )
    identity_step0_sha256 = _normalize_sha256(
        identity.get("actual_step0_parameter_tree_sha256"),
        name="initialization identity actual_step0_parameter_tree_sha256",
    )
    graft_base_sha256 = _normalize_sha256(tree_hashes.get("source_sha256"), name="graft manifest source_sha256")
    if (
        calibration_step0_sha256 != identity_step0_sha256
        or calibration_base_sha256 != graft_base_sha256
        or provenance.get("split_sha256") != manifest_sha256
        or identity.get("graft_manifest_sha256") != graft.get("manifest_sha256")
    ):
        raise ValueError("v3.5 calibration does not bind the authenticated fresh-base initialization and manifest.")

    manifest = _read_json(paths["manifest"], name="embedded v3.5 episode manifest")
    episodes = manifest.get("episodes") if isinstance(manifest, dict) else None
    if not isinstance(episodes, list):
        raise ValueError("embedded v3.5 episode manifest must contain an episodes list.")
    train_ids = {
        str(record.get("stable_id", "")).strip()
        for record in episodes
        if isinstance(record, dict) and record.get("include") is True and record.get("split") == "train"
    }
    population = payload.get("population")
    artifact_ids = population.get("stable_ids") if isinstance(population, dict) else None
    if (
        not isinstance(population, dict)
        or population.get("split") != "train"
        or population.get("episode_count") != 54
        or not isinstance(artifact_ids, list)
        or len(artifact_ids) != 54
        or len(set(artifact_ids)) != 54
        or set(artifact_ids) != train_ids
    ):
        raise ValueError("v3.5 calibration membership is not exactly the frozen 54-episode training split.")
    membership = [{"split": "train", "stable_id": stable_id} for stable_id in artifact_ids]
    membership_sha256 = hashlib.sha256(_canonical_json_bytes(membership, ensure_ascii=False)).hexdigest()
    if provenance.get("observed_membership_sha256") != membership_sha256:
        raise ValueError("v3.5 calibration membership hash is invalid.")

    gate = payload.get("gate")
    gates = payload.get("gates")
    raw_gate_sha256 = gate.get("raw_w_sha256") if isinstance(gate, dict) else None
    gate_target = gate.get("target_effective_tanh_gate") if isinstance(gate, dict) else None
    gate_atol = gate.get("required_atol") if isinstance(gate, dict) else None
    if (
        not isinstance(gate, dict)
        or not isinstance(gates, dict)
        or gate_target != 0.5
        or gate.get("open_channel_count") != expected_value_width
        or isinstance(gate_atol, bool)
        or not isinstance(gate_atol, int | float)
        or not np.isfinite(gate_atol)
        or gate_atol < 0
        or re.fullmatch(r"[0-9a-f]{64}", str(raw_gate_sha256)) is None
        or identity.get("memory_inject_w_sha256") != raw_gate_sha256
        or gates.get("passes") is not True
        or gates.get("all_episodes_train") is not True
        or gates.get("fixed_effective_gate_is_0_5") is not True
    ):
        raise ValueError("v3.5 calibration gate/population invariants are not passing.")

    norm_provenance = _read_json(paths["norm_provenance"], name="v3.5 norm-stats provenance")
    norm_manifest = norm_provenance.get("manifest") if isinstance(norm_provenance, dict) else None
    norm_selection = norm_provenance.get("selection") if isinstance(norm_provenance, dict) else None
    norm_stats = norm_provenance.get("norm_stats") if isinstance(norm_provenance, dict) else None
    norm_storage = norm_provenance.get("train_storage") if isinstance(norm_provenance, dict) else None
    if (
        not isinstance(norm_provenance, dict)
        or norm_provenance.get("status") != "complete"
        or not isinstance(norm_manifest, dict)
        or norm_manifest.get("sha256") != manifest_sha256
        or not isinstance(norm_selection, dict)
        or provenance.get("dataset_sha256") != norm_selection.get("dataset_episode_frame_protocol_sha256")
        or not isinstance(norm_stats, dict)
        or norm_stats.get("sha256") != artifact_hashes.get("norm_stats_sha256")
        or not isinstance(norm_storage, dict)
        or norm_storage.get("sha256") != artifact_hashes.get("train_storage_sha256")
    ):
        raise ValueError("v3.5 norm/calibration/manifest provenance binding is invalid.")

    return V35CheckpointProtocol(
        assets_dir=assets,
        calibration_path=paths["calibration"],
        calibration_id=calibration_id,
        manifest_path=paths["manifest"],
        manifest_sha256=manifest_sha256,
        memory_injection_c=injection_c,
        memory_injection_tau=injection_tau,
        alpha_step=alpha_step,
        gate_target=float(gate_target),
        gate_atol=float(gate_atol),
        raw_gate_sha256=str(raw_gate_sha256),
        norm_stats_sha256=str(artifact_hashes["norm_stats_sha256"]),
    )


def _apply_v35_checkpoint_calibration(cfg: _config.TrainConfig, protocol: V35CheckpointProtocol) -> _config.TrainConfig:
    memory_config = dataclasses.replace(cfg.model.memory, alpha_step=protocol.alpha_step)
    model_config = dataclasses.replace(
        cfg.model,
        memory=memory_config,
        memory_injection_c=protocol.memory_injection_c,
        memory_injection_tau=protocol.memory_injection_tau,
        memory_v35_calibrated=True,
        memory_v35_calibration_id=protocol.calibration_id,
        memory_v35_calibration_path=str(protocol.calibration_path),
    )
    return dataclasses.replace(cfg, model=model_config)


def _validate_v35_norm_asset(protocol: V35CheckpointProtocol, *, asset_id: str) -> None:
    norm_path = protocol.assets_dir / asset_id / "norm_stats.json"
    if not norm_path.is_file() or _sha256(norm_path) != protocol.norm_stats_sha256:
        raise ValueError("v3.5 checkpoint norm_stats.json does not match initialization identity.")


def _validate_v35_loaded_gate(model: Any, protocol: V35CheckpointProtocol) -> None:
    raw_gate = np.asarray(model.memory_inject_w.value)
    if raw_gate.dtype != np.float32 or hashlib.sha256(raw_gate.tobytes(order="C")).hexdigest() != (
        protocol.raw_gate_sha256
    ):
        raise ValueError("v3.5 checkpoint memory_inject_w dtype/hash does not match calibration identity.")
    effective = np.tanh(raw_gate.astype(np.float32)).astype(np.float32)
    if not np.all(np.isfinite(effective)) or not np.allclose(
        effective, protocol.gate_target, rtol=0.0, atol=protocol.gate_atol
    ):
        raise ValueError("v3.5 checkpoint effective injection gate does not match calibration.")


def _resolve_manifest_raw_dir(
    manifest_path: pathlib.Path, manifest: dict[str, Any], record: dict[str, Any]
) -> pathlib.Path:
    raw_dir_value = record.get("raw_dir")
    if not isinstance(raw_dir_value, str) or not raw_dir_value.strip():
        raise ValueError("v3.5 manifest episode is missing raw_dir.")
    raw_dir = pathlib.Path(raw_dir_value)
    if raw_dir.is_absolute():
        raise ValueError("v3.5 manifest raw_dir must be relative to raw_root.")
    stable_id = record.get("stable_id")
    if not isinstance(stable_id, str) or not raw_dir.as_posix().rstrip("/").endswith(stable_id):
        raise ValueError(f"v3.5 manifest raw_dir does not identify stable_id {stable_id!r}.")
    raw_root_value = manifest.get("raw_root")
    if not isinstance(raw_root_value, str) or not raw_root_value.strip():
        raise ValueError("v3.5 manifest raw_root must be a non-empty path string.")
    raw_root = pathlib.Path(raw_root_value)
    if not raw_root.is_absolute():
        raw_root = manifest_path.parent / raw_root
    raw_root = raw_root.resolve()
    resolved = (raw_root / raw_dir).resolve()
    try:
        resolved.relative_to(raw_root)
    except ValueError as exc:
        raise ValueError(f"v3.5 manifest raw_dir escapes raw_root for {stable_id!r}.") from exc
    return resolved


def _v35_expected_frozen_splits(records: list[dict[str, Any]], *, seed: int) -> dict[str, str]:
    """Call the production loader's manifest-field-only 74/8/8 split implementation."""
    return _data_loader._v35_expected_frozen_splits(records, seed=seed)  # noqa: SLF001


def _validate_v35_frozen_population(
    manifest: dict[str, Any],
    *,
    manifest_path: pathlib.Path,
    split_seed: int,
    subtask_vocabulary: tuple[str, ...],
) -> None:
    """Validate canonical population fields and reproduce every frozen split assignment."""
    if manifest.get("schema_version") != V35_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"v3.5 frozen manifest requires schema_version={V35_MANIFEST_SCHEMA_VERSION}.")
    if manifest.get("dataset_version") != "v36" or manifest.get("review_status") != "frozen":
        raise ValueError("v3.5 manifest must have dataset_version='v35' and review_status='frozen'.")
    if manifest.get("split_algorithm") != V35_SPLIT_ALGORITHM:
        raise ValueError(f"v3.5 manifest must use split_algorithm={V35_SPLIT_ALGORITHM!r}.")
    if manifest.get("split_algorithm_sha256") != V35_SPLIT_ALGORITHM_SHA256:
        raise ValueError("v3.5 manifest split algorithm specification hash is invalid.")
    if len(subtask_vocabulary) != 7 or manifest.get("task_vocabulary") != list(subtask_vocabulary):
        raise ValueError("v3.5 frozen manifest task_vocabulary/order does not match the configured seven tasks.")

    block_audit = manifest.get("block_confound_audit")
    report_file = block_audit.get("report_file") if isinstance(block_audit, dict) else None
    report_sha256 = block_audit.get("report_sha256") if isinstance(block_audit, dict) else None
    if (
        not isinstance(block_audit, dict)
        or block_audit.get("status") != "pass"
        or block_audit.get("manifest_fields_only") is not True
        or not isinstance(report_file, str)
        or not report_file.strip()
        or not isinstance(report_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", report_sha256) is None
    ):
        raise ValueError("v3.5 frozen manifest is missing a passing 0830 object/side block-confound audit.")
    report_path = pathlib.Path(report_file)
    if report_path.is_absolute() or ".." in report_path.parts:
        raise ValueError("v3.5 block-confound report_file must be relative to the manifest directory.")
    report_path = manifest_path.parent / report_path
    if not report_path.is_file() or _sha256(report_path) != report_sha256:
        raise ValueError("v3.5 block-confound audit report bytes do not match their manifest hash.")

    records = manifest["episodes"]
    stable_ids: set[str] = set()
    excluded: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"manifest.episodes[{index}] must be an object.")
        stable_id = record.get("stable_id")
        if not isinstance(stable_id, str) or not stable_id.strip() or stable_id in stable_ids:
            raise ValueError(f"manifest.episodes[{index}] has a missing or duplicate stable_id {stable_id!r}.")
        stable_ids.add(stable_id)
        if record.get("include") is not True:
            excluded.append(record)
            continue
        collection = record.get("collection")
        part = record.get("part", "")
        object_name = record.get("object")
        prompt = record.get("prompt")
        if collection not in ("0830", "0831"):
            raise ValueError(f"v3.5 included episode {stable_id!r} has invalid collection {collection!r}.")
        if (collection == "0830" and part not in ("part1", "part2")) or (
            collection == "0816" and part not in ("", None)
        ):
            raise ValueError(f"v3.5 included episode {stable_id!r} has invalid part {part!r} for {collection}.")
        if object_name not in V35_OBJECT_PROMPTS or prompt != V35_OBJECT_PROMPTS[object_name]:
            raise ValueError(f"v3.5 included episode {stable_id!r} has a noncanonical object/prompt pair.")
        if record.get("target_side") not in ("left", "right"):
            raise ValueError(f"v3.5 included episode {stable_id!r} has invalid target_side.")
        if str(record.get("exclude_reason", "")).strip():
            raise ValueError(f"v3.5 included episode {stable_id!r} has a nonempty exclude_reason.")

    if len(records) != 71 or len(excluded) != 1:
        raise ValueError("v3.5 frozen provenance must contain 90 included episodes plus one excluded raw episode.")
    excluded_record = excluded[0]
    if (
        excluded_record.get("stable_id") != "0830_bin_part2/demo14"
        or not str(excluded_record.get("exclude_reason", "")).strip()
        or not str(excluded_record.get("raw_dir", "")).strip()
    ):
        raise ValueError("v3.5 frozen provenance must retain excluded 0830_bin_part2/demo14 with reason/raw_dir.")
    expected_splits = _v35_expected_frozen_splits(records, seed=split_seed)
    wrong = {
        str(record["stable_id"]): (expected_splits[str(record["stable_id"])], record.get("split"))
        for record in records
        if record.get("include") is True and record.get("split") != expected_splits[str(record["stable_id"])]
    }
    if wrong:
        raise ValueError(f"v3.5 manifest split assignments do not reproduce the frozen algorithm: {wrong}.")


def _load_v35_episode_protocol(
    *,
    manifest_path: pathlib.Path,
    manifest_sha256: str | None,
    raw_demo: pathlib.Path,
    expected_split_seed: int | None,
    subtask_vocabulary: tuple[str, ...],
    evidence_subtasks: tuple[str, ...],
    occlusion_subtasks: tuple[str, ...],
    decision_subtasks: tuple[str, ...],
    execute_subtasks: tuple[str, ...],
    stride: int,
    tail_guard: int,
) -> V35EpisodeProtocol:
    """Build a manifest-owned E-only schedule without consulting model outputs.

    This deliberately requires both a frozen manifest digest and a per-record label digest.
    A manifest that merely points at a mutable label file is not trusted phase metadata.
    """
    if stride != V35_STRIDE_FRAMES:
        raise ValueError(
            f"v3.5 pilot requires exactly {V35_STRIDE_FRAMES} raw frames per transition; got stride={stride}."
        )
    if tail_guard != V35_E_TAIL_GUARD_FRAMES:
        raise ValueError(
            f"v3.5 pilot requires the sealed {V35_E_TAIL_GUARD_FRAMES}-raw-frame E tail guard; got {tail_guard}."
        )
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise ValueError(f"v3.5 manifest does not exist: {manifest_path}")
    expected_manifest_sha = _normalize_sha256(manifest_sha256, name="manifest_sha256")
    actual_manifest_sha = _sha256(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError(
            f"v3.5 manifest SHA-256 mismatch: expected {expected_manifest_sha}, found {actual_manifest_sha}."
        )
    manifest = _read_json(manifest_path, name="v3.5 manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("episodes"), list):
        raise ValueError("v3.5 manifest must be an object containing an episodes list.")
    _strict_int(manifest.get("schema_version"), name="manifest.schema_version", minimum=1)
    split_seed = _strict_int(manifest.get("split_seed"), name="manifest.split_seed")
    if split_seed != 36:
        raise ValueError(f"v3.5 frozen split algorithm requires split_seed=36; found {split_seed}.")
    if expected_split_seed is not None and split_seed != expected_split_seed:
        raise ValueError(f"v3.5 manifest split_seed mismatch: expected {expected_split_seed}, found {split_seed}.")
    _validate_v35_frozen_population(
        manifest,
        manifest_path=manifest_path,
        split_seed=split_seed,
        subtask_vocabulary=subtask_vocabulary,
    )

    demo = raw_demo.expanduser().resolve()
    matches: list[dict[str, Any]] = []
    for index, raw_record in enumerate(manifest["episodes"]):
        if not isinstance(raw_record, dict):
            raise ValueError(f"manifest.episodes[{index}] must be an object.")
        if _resolve_manifest_raw_dir(manifest_path, manifest, raw_record) == demo:
            matches.append(raw_record)
    if len(matches) != 1:
        raise ValueError(f"raw demo {demo} must match exactly one v3.5 manifest record; found {len(matches)}.")
    record = matches[0]
    if record.get("include") is not True:
        raise ValueError(f"v3.5 manifest record for {demo} is not explicitly included.")
    stable_id = record.get("stable_id")
    split = record.get("split")
    collection = record.get("collection")
    part = record.get("part", "")
    object_name = record.get("object")
    prompt = record.get("prompt")
    target_side = record.get("target_side")
    if not isinstance(stable_id, str) or not stable_id.strip():
        raise ValueError("v3.5 manifest record is missing stable_id.")
    if split not in ("train", "development", "final_test"):
        raise ValueError(f"v3.5 manifest record {stable_id!r} has an unfrozen/invalid split {split!r}.")
    if collection not in ("0830", "0831"):
        raise ValueError(f"v3.5 manifest record {stable_id!r} has invalid collection {collection!r}.")
    if (collection == "0830" and part not in ("part1", "part2")) or (collection == "0831" and part not in ("", None)):
        raise ValueError(f"v3.5 manifest record {stable_id!r} has invalid part {part!r} for {collection}.")
    if object_name not in V35_OBJECT_PROMPTS or prompt != V35_OBJECT_PROMPTS[object_name]:
        raise ValueError(f"v3.5 manifest record {stable_id!r} has a noncanonical object/prompt pair.")
    if target_side not in ("left", "right"):
        raise ValueError(f"v3.5 manifest record {stable_id!r} has invalid target_side {target_side!r}.")

    expected_num_frames = _strict_int(
        record.get("expected_num_frames"), name=f"manifest episode {stable_id}.expected_num_frames", minimum=1
    )
    label_name = record.get("label_file")
    if not isinstance(label_name, str) or pathlib.Path(label_name).name != label_name:
        raise ValueError(f"v3.5 manifest record {stable_id!r} label_file must be one relative filename.")
    label_path = (demo / label_name).resolve()
    if label_path.parent != demo or not label_path.is_file():
        raise ValueError(f"v3.5 label sidecar does not exist inside the raw demo: {label_path}")
    expected_label_sha = _normalize_sha256(record.get("label_sha256"), name=f"{stable_id}.label_sha256")
    actual_label_sha = _sha256(label_path)
    if actual_label_sha != expected_label_sha:
        raise ValueError(
            f"v3.5 label SHA-256 mismatch for {stable_id}: expected {expected_label_sha}, found {actual_label_sha}."
        )

    segments = _read_json(label_path, name=f"{stable_id} semantic label sidecar")
    if not isinstance(segments, list) or len(segments) != 5:
        raise ValueError(f"{stable_id} must have exactly five semantic label segments.")
    phase_by_frame: list[str] = []
    expected_start = 0
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"{stable_id} label segment {index} must be an object.")
        task = segment.get("task")
        if not isinstance(task, str) or not task:
            raise ValueError(f"{stable_id} label segment {index} has no task.")
        start = _strict_int(segment.get("start"), name=f"{stable_id}.segments[{index}].start")
        end = _strict_int(segment.get("end"), name=f"{stable_id}.segments[{index}].end")
        if start != expected_start or end < start:
            raise ValueError(
                f"{stable_id} label segments must be gap-free, non-overlapping, and inclusive; "
                f"segment {index} is [{start}, {end}], expected start {expected_start}."
            )
        phase_by_frame.extend([task] * (end - start + 1))
        expected_start = end + 1
    if len(phase_by_frame) != expected_num_frames:
        raise ValueError(
            f"{stable_id} label coverage has {len(phase_by_frame)} frames, expected {expected_num_frames}."
        )

    evidence = set(evidence_subtasks)
    occlusion = set(occlusion_subtasks)
    decision = set(decision_subtasks)
    execute = set(execute_subtasks)
    if not evidence or not occlusion or not decision or not execute:
        raise ValueError("v3.5 phase vocabularies must all be non-empty.")
    tasks = [str(segment["task"]) for segment in segments]
    exact_tasks = [
        "open both lids",
        "inspect both bins",
        "close both lids and reset arms",
        f"wait; target bin is {target_side}",
        f"open {target_side} bin",
    ]
    if tasks != exact_tasks:
        raise ValueError(f"{stable_id} does not have the exact side-consistent five-phase schema: {tasks}.")
    if tasks[1] not in evidence or tasks[2] not in occlusion or tasks[3] not in decision or tasks[4] not in execute:
        raise ValueError(f"{stable_id} labels do not follow the sealed A/E/O/D/X phase order: {tasks}.")
    if tasks[0] in evidence | occlusion | decision | execute:
        raise ValueError(f"{stable_id} approach phase overlaps a memory phase vocabulary: {tasks[0]!r}.")
    if target_side not in tasks[3].lower() or target_side not in tasks[4].lower():
        raise ValueError(f"{stable_id} target_side={target_side!r} disagrees with D/X labels {tasks[3:5]}.")

    e_start = _strict_int(segments[1]["start"], name=f"{stable_id}.evidence_start")
    e_end = _strict_int(segments[1]["end"], name=f"{stable_id}.evidence_end")
    visibility = record.get("e_visibility")
    if not isinstance(visibility, dict):
        raise ValueError(f"v3.5 manifest record {stable_id!r} is missing manual E-visibility provenance.")
    contact_sheet_sha256 = visibility.get("contact_sheet_sha256")
    last_clean = visibility.get("last_clean_visible_frame")
    if (
        visibility.get("manual_reviewed") is not True
        or visibility.get("both_objects_visible") is not True
        or visibility.get("first_valid_visible_frame") != e_start
        or not isinstance(last_clean, int)
        or isinstance(last_clean, bool)
        or last_clean < e_end - tail_guard
        or not isinstance(contact_sheet_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", contact_sheet_sha256) is None
    ):
        raise ValueError(f"v3.5 manifest record {stable_id!r} has invalid manual E-visibility/anchor provenance.")
    final_e_limit = e_end - tail_guard
    write_frames = tuple(frame for frame in range(0, expected_num_frames, stride) if e_start <= frame <= final_e_limit)
    if not write_frames:
        raise ValueError(f"{stable_id} has zero eligible sampled E frames after the tail guard.")

    d_valid = record.get("d_valid")
    if not isinstance(d_valid, dict):
        raise ValueError(f"v3.5 manifest record {stable_id!r} is missing its independent d_valid sidecar.")
    if d_valid.get("detector") != V35_D_VALID_DETECTOR or d_valid.get("state_dim") != 14:
        raise ValueError(f"v3.5 manifest record {stable_id!r} has the wrong d_valid detector provenance.")
    core_start = _strict_int(d_valid.get("start"), name=f"{stable_id}.d_valid.start")
    core_end = _strict_int(d_valid.get("end"), name=f"{stable_id}.d_valid.end")
    d_start = _strict_int(segments[3]["start"], name=f"{stable_id}.decision_start")
    d_end = _strict_int(segments[3]["end"], name=f"{stable_id}.decision_end")
    if not d_start <= core_start <= core_end <= d_end:
        raise ValueError(
            f"{stable_id} independent d_valid [{core_start}, {core_end}] lies outside semantic D [{d_start}, {d_end}]."
        )

    return V35EpisodeProtocol(
        stable_id=stable_id.strip(),
        split=split,
        collection=collection,
        part="" if part is None else part,
        object_name=object_name,
        prompt=prompt,
        manifest_path=manifest_path,
        manifest_sha256=actual_manifest_sha,
        label_path=label_path,
        label_sha256=actual_label_sha,
        expected_num_frames=expected_num_frames,
        phase_by_frame=tuple(phase_by_frame),
        write_frames=write_frames,
        d_valid_start=core_start,
        d_valid_end=core_end,
    )


def _read_video_frames(path: pathlib.Path, stride: int) -> tuple[list[np.ndarray], int]:
    """Every stride-th frame of an mp4 as uint8 RGB, plus the total frame count."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if total % stride == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        total += 1
    cap.release()
    return frames, total


def _runs(frames: list[int], labels: list[str]) -> str:
    """Collapse a per-prediction label sequence into 'startframe-endframe label | ...' runs."""
    out = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append(f"{frames[start]}-{frames[i - 1]} {labels[start]!r}")
            start = i
    return " | ".join(out)


def _render_plot(pred_frames: list[int], curve: np.ndarray, total: int, size_hw: tuple[int, int]):
    """The static surprise plot as an RGB array, plus per-raw-frame cursor columns and row range."""
    h, w = size_hw
    fig = plt.Figure(figsize=(w / 100, h / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.step(pred_frames, curve, where="post", lw=0.9, color="tab:blue")
    ax.plot(pred_frames, curve, ".", ms=3, color="tab:blue")
    if LOG_Y:
        ax.set_yscale("log")
    # the trained m0's blank state is not scale-calibrated, so the first write's loss can sit
    # orders of magnitude above the written-state band: scale the axis to the rest of the curve
    if len(curve) > 2:
        lo, hi = float(curve[1:].min()), float(curve[1:].max())
        ax.set_ylim(max(lo * 0.7, 1e-8), hi * 1.5)
        if curve[0] > hi * 1.5:
            ax.text(
                0.99,
                0.97,
                f"first write {curve[0]:.2e} (off scale)",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="tab:red",
            )
    ax.set_xlim(0, total - 1)
    ax.set_xlabel("frame")
    ax.set_ylabel("write surprise")
    ax.set_title("memory write surprise at each prediction")
    fig.tight_layout()
    canvas.draw()
    base = np.asarray(canvas.buffer_rgba())[..., :3].copy()
    # data -> pixel column for every raw frame (only x matters), and the axes' row range
    xs = np.arange(total)
    cols = np.round(ax.transData.transform(np.column_stack([xs, np.full(total, curve.min())]))[:, 0]).astype(int)
    bbox = ax.get_window_extent()
    rows = (int(base.shape[0] - bbox.y1), int(base.shape[0] - bbox.y0))
    return base, np.clip(cols, 0, base.shape[1] - 1), rows


def main(args: Args) -> None:
    from flax import nnx
    import jax
    import jax.numpy as jnp

    cfg = _config.get_config(args.config)
    checkpoint_protocol: V35CheckpointProtocol | None = None
    if bool(getattr(cfg.model, "memory_v35_enabled", False)):
        _project_paths.configure_v35_runtime_environment()
        checkpoint_protocol = _load_v35_checkpoint_protocol(
            args.ckpt_dir,
            expected_config_name=cfg.name,
            expected_seed=cfg.seed,
            expected_value_width=cfg.model.memory.d_value,
        )
        # Registered c/tau are deliberately unsafe construction placeholders. Inference gets
        # its numeric model config only from checkpoint-owned, hash-authenticated calibration.
        cfg = _apply_v35_checkpoint_calibration(cfg, checkpoint_protocol)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    if checkpoint_protocol is not None:
        if not data_config.asset_id:
            raise ValueError("v3.5 checkpoint has no normalization asset_id.")
        _validate_v35_norm_asset(checkpoint_protocol, asset_id=data_config.asset_id)
    norm_stats = _checkpoints.load_norm_stats(args.ckpt_dir / "assets", data_config.asset_id)
    stride = args.stride or data_config.memory_stride_frames
    assert stride > 0, "no stride: set --stride or use a config with memory_stride_frames"
    model_v35 = bool(getattr(cfg.model, "memory_v35_enabled", False))
    data_v35 = bool(getattr(data_config, "memory_v35_enabled", False))
    if model_v35 != data_v35:
        raise ValueError(f"model/data v3.5 enablement mismatch: model={model_v35}, data={data_v35}.")

    protocol: V35EpisodeProtocol | None = None
    if model_v35:
        if checkpoint_protocol is None:
            raise AssertionError("v3.5 checkpoint protocol was not authenticated.")
        if stride != V35_STRIDE_FRAMES or int(data_config.memory_stride_frames) != V35_STRIDE_FRAMES:
            raise ValueError(
                f"v3.5 inference is sealed to stride={V35_STRIDE_FRAMES}; "
                f"requested={stride}, configured={data_config.memory_stride_frames}."
            )
        manifest_value = args.manifest or getattr(data_config, "memory_episode_manifest_path", None)
        if manifest_value is None:
            raise ValueError("v3.5 inference requires a frozen episode manifest path.")
        manifest_path = pathlib.Path(manifest_value).expanduser().resolve()
        if not manifest_path.is_file() or _sha256(manifest_path) != checkpoint_protocol.manifest_sha256:
            raise ValueError("v3.5 evaluation manifest is not byte-identical to the checkpoint-owned manifest.")
        requested_digest = args.manifest_sha256 or getattr(data_config, "memory_episode_manifest_sha256", None)
        if requested_digest is not None and _normalize_sha256(requested_digest, name="manifest_sha256") != (
            checkpoint_protocol.manifest_sha256
        ):
            raise ValueError("v3.5 requested/configured manifest SHA-256 disagrees with checkpoint provenance.")
        protocol = _load_v35_episode_protocol(
            manifest_path=manifest_path,
            manifest_sha256=checkpoint_protocol.manifest_sha256,
            raw_demo=args.raw_demo,
            expected_split_seed=getattr(data_config, "memory_manifest_split_seed", None),
            subtask_vocabulary=tuple(data_config.memory_subtask_vocab),
            evidence_subtasks=tuple(data_config.evidence_subtasks),
            occlusion_subtasks=tuple(data_config.memory_occlusion_subtasks),
            decision_subtasks=tuple(data_config.memory_required_subtasks),
            execute_subtasks=tuple(data_config.memory_execute_subtasks),
            stride=stride,
            tail_guard=int(data_config.memory_e_tail_guard_frames),
        )

    # Raw demo -> arrays. The top camera is read at stride 1 (the video shows every raw frame);
    # the wrist cameras are only needed at the prediction frames.
    demo = args.raw_demo
    state_raw = np.concatenate(
        [np.load(demo / "left_joint_positions.npy"), np.load(demo / "right_joint_positions.npy")], axis=1
    ).astype(np.float32)
    actions_raw = np.concatenate(
        [np.load(demo / "left_control.npy"), np.load(demo / "right_control.npy")], axis=1
    ).astype(np.float32)
    top, n_top = _read_video_frames(demo / "top_camera_rgb.mp4", 1)
    left, n_left = _read_video_frames(demo / "left_camera_rgb.mp4", stride)
    right, n_right = _read_video_frames(demo / "right_camera_rgb.mp4", stride)
    stream_lengths = {
        "state": len(state_raw),
        "actions": len(actions_raw),
        "top": n_top,
        "left": n_left,
        "right": n_right,
    }
    if protocol is not None and any(length != protocol.expected_num_frames for length in stream_lengths.values()):
        raise ValueError(
            f"{protocol.stable_id} raw stream lengths must all equal manifest expected_num_frames "
            f"{protocol.expected_num_frames}; got {stream_lengths}."
        )
    total = min(stream_lengths.values())
    eval_ts = list(range(0, total, stride))
    if protocol is None:
        schedule_summary = f"one legacy write per prediction, stride {stride}"
    else:
        schedule_summary = (
            f"v3.5 clock stride {stride}, {len(protocol.write_frames)} eligible E commits, "
            f"manifest {protocol.manifest_sha256[:12]}, labels {protocol.label_sha256[:12]}"
        )
    print(f"{demo}: {total} frames -> {len(eval_ts)} predictions ({schedule_summary})", flush=True)

    # The exact inference-time input pipeline from the config. BuildMemorySequence is the
    # dataset-side unpacker of lerobot's stacked delta_timestamps frames -- skipped on raw items.
    input_transforms = [
        tf for tf in data_config.data_transforms.inputs if not isinstance(tf, _transforms.BuildMemorySequence)
    ]
    normalize = _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
    model_transforms = list(data_config.model_transforms.inputs)

    unnormalize = _transforms.Unnormalize(
        {"actions": norm_stats["actions"]}, use_quantiles=data_config.use_quantile_norm
    )
    arm_mask = np.asarray(_transforms.make_bool_mask(6, -1, 6, -1))

    def build_item(t: int) -> dict:
        """Model inputs for raw frame t (inference-style: no actions, no window)."""
        item = {
            "observation/image": top[t],
            "observation/left_wrist_image": left[t // stride],
            "observation/right_wrist_image": right[t // stride],
            "observation/state": state_raw[t],
        }
        for tf in input_transforms:
            item = tf(item)
        item = normalize(item)
        for tf in model_transforms:
            item = tf(item)
        return item

    pg = _tokenizer.FASTSubtaskTokenizer(cfg.model.max_token_len)._paligemma_tokenizer  # noqa: SLF001
    # Same terminator the training subtasks were tokenized with (trailing "\n" of the segment).
    stop_token = int(pg.encode("placeholder subtask\n")[-1])

    # float32 restore: the memory's inner gradient descent was built and validated in f32.
    model = cfg.model.load(_model.restore_params(args.ckpt_dir / "params", dtype=jnp.float32))
    if checkpoint_protocol is not None:
        _validate_v35_loaded_gate(model, checkpoint_protocol)
        effective_gate = np.tanh(np.asarray(model.memory_inject_w.value, dtype=np.float32))
        print(
            f"loaded authenticated v3.5 {args.ckpt_dir} | calibration {checkpoint_protocol.calibration_id} "
            f"c={checkpoint_protocol.memory_injection_c:.6g} tau={checkpoint_protocol.memory_injection_tau:.6g} "
            f"alpha={checkpoint_protocol.alpha_step:.6g} effective gate "
            f"[{effective_gate.min():.6g}, {effective_gate.max():.6g}]",
            flush=True,
        )
    else:
        gate = np.asarray(model.memory_gate.value)
        print(
            f"loaded {args.ckpt_dir} | memory_gate norm {np.linalg.norm(gate):.4f} "
            f"mean|g| {np.abs(gate).mean():.5f} max|g| {np.abs(gate).max():.5f} (0 = memory content unused)",
            flush=True,
        )
    if args.zero_gate and not model_v35:
        model.memory_gate.value = jnp.zeros_like(model.memory_gate.value)
        print("ZERO-GATE: content gate forced to 0 -- memory tokens are zero embeddings", flush=True)
    elif args.zero_gate:
        print("ZERO-INJECTION: v3.5 raw retrieval is forced to zero; the frozen tanh gate is untouched", flush=True)
    graphdef, state = nnx.split(model)

    if protocol is None:
        # Preserve the v3-v3.4 call signature and historical every-frame writer exactly.
        infer = jax.jit(
            lambda s, ms, rng, o: nnx.merge(graphdef, s).sample_with_memory(
                rng, o, ms, stop_token=stop_token, max_decode_steps=args.max_decode_steps
            )
        )
    else:
        infer = jax.jit(
            lambda s, ms, rng, o, transition_valid, write_mask, anchor_key, anchor_value, anchor_delay: nnx.merge(
                graphdef, s
            ).sample_with_memory(
                rng,
                o,
                ms,
                stop_token=stop_token,
                max_decode_steps=args.max_decode_steps,
                zero_read=args.zero_gate,
                v35_transition_valid=transition_valid,
                v35_write_mask=write_mask,
                v35_anchor_key=anchor_key,
                v35_anchor_value=anchor_value,
                v35_anchor_delay_steps=anchor_delay,
            )
        )
    # One invocation evaluates exactly one raw demo/episode. Reset before frame zero even if
    # callers reuse main() in-process; state is never carried across episode identities.
    mem_state = model.memory.init_state(1)
    print("memory reset at episode start", flush=True)
    if args.ablate_memory:
        print("ABLATION: writes are discarded -- every prediction reads the blank m0 memory", flush=True)

    # Sequential episode replay: the memory state threads from one prediction into the next.
    preds: list[str] = []
    surprise: list[float] = []
    call_ms: list[float] = []
    pred_chunks: list[np.ndarray] = []
    write_occurred: list[bool] = []
    transition_kind: list[str] = []
    decision_valid: list[bool] = []
    geometry_valid: list[bool] = []
    query_cosine_slots: list[np.ndarray] = []
    query_beta_slots: list[np.ndarray] = []
    query_cosine_mean: list[float] = []
    query_cosine_max: list[float] = []
    query_low_alignment_fraction: list[float] = []
    query_beta_mean: list[float] = []
    query_beta_abs_mean: list[float] = []
    query_beta_sign_consistency: list[float] = []
    query_cancellation: list[float] = []
    mean_raw_read_anchor_cosine: list[float] = []
    anchor_retention: list[float] = []
    anchor_delay_reported: list[int] = []
    query_residual_rms: list[float] = []
    query_relative_residual: list[float] = []
    anchor_hidden_norm_sq: list[float] = []
    injected_pre_cast_rms: list[float] = []
    injected_post_cast_rms: list[float] = []
    anchor_key = np.zeros((1, model.memory.config.d_key), dtype=np.float32)
    anchor_value = np.zeros((1, model.memory.config.d_value), dtype=np.float32)
    anchor_delay = np.zeros((1,), dtype=np.int32)
    anchor_available = False
    t_start = time.perf_counter()
    write_frame_set = set() if protocol is None else set(protocol.write_frames)
    for k, t in enumerate(eval_ts):
        item = build_item(t)
        batch = jax.tree.map(lambda x: np.asarray(x)[None], item)
        t0 = time.perf_counter()
        observation = _model.Observation.from_dict(batch)
        if protocol is None:
            actions, new_state, aux = infer(state, mem_state, jax.random.fold_in(jax.random.key(0), k), observation)
            expected_write = True
            phase = "legacy"
        else:
            expected_write = t in write_frame_set
            phase = protocol.phase_by_frame[t]
            transition_valid = jnp.ones((), dtype=bool)
            actions, new_state, aux = infer(
                state,
                mem_state,
                jax.random.fold_in(jax.random.key(0), k),
                observation,
                transition_valid,
                jnp.asarray(expected_write, dtype=bool),
                jnp.asarray(anchor_key),
                jnp.asarray(anchor_value),
                jnp.asarray(anchor_delay),
            )
        jax.block_until_ready((actions, new_state))
        call_ms.append((time.perf_counter() - t0) * 1e3)
        actual_write = bool(np.asarray(aux["write_occurred"])[0])
        if protocol is not None:
            pre_cast_rms = float(np.asarray(aux["v35_injected_pre_cast_rms"])[0])
            post_cast_rms = float(np.asarray(aux["v35_injected_post_cast_rms"])[0])
            if args.zero_gate and pre_cast_rms != 0.0:
                raise RuntimeError(f"v3.5 zero-injection contract failed at frame {t}: pre-cast RMS={pre_cast_rms!r}.")
            invalid_request = bool(np.asarray(aux["v35_invalid_write_request"])[0])
            transitioned = bool(np.asarray(aux["v35_transition_applied"])[0])
            if invalid_request or not transitioned or actual_write != expected_write:
                raise RuntimeError(
                    f"v3.5 transition contract failed at frame {t}: expected_write={expected_write}, "
                    f"actual_write={actual_write}, transitioned={transitioned}, invalid_request={invalid_request}."
                )
            kind = "commit" if actual_write else "decay"
        else:
            pre_cast_rms = float("nan")
            post_cast_rms = float("nan")
            kind = "commit" if actual_write else "frozen"
        write_occurred.append(actual_write)
        transition_kind.append(kind)
        injected_pre_cast_rms.append(pre_cast_rms)
        injected_post_cast_rms.append(post_cast_rms)
        decision_valid.append(protocol is not None and protocol.d_valid_start <= t <= protocol.d_valid_end)
        if protocol is None:
            geometry_valid.append(False)
            query_cosine_slots.append(np.full((model.memory_query_tokens,), np.nan, dtype=np.float32))
            query_beta_slots.append(np.full((model.memory_query_tokens,), np.nan, dtype=np.float32))
            query_cosine_mean.append(float("nan"))
            query_cosine_max.append(float("nan"))
            query_low_alignment_fraction.append(float("nan"))
            query_beta_mean.append(float("nan"))
            query_beta_abs_mean.append(float("nan"))
            query_beta_sign_consistency.append(float("nan"))
            query_cancellation.append(float("nan"))
            mean_raw_read_anchor_cosine.append(float("nan"))
            anchor_retention.append(float("nan"))
            anchor_delay_reported.append(-1)
            query_residual_rms.append(float("nan"))
            query_relative_residual.append(float("nan"))
            anchor_hidden_norm_sq.append(float("nan"))
        else:
            geometry_ok = bool(np.asarray(aux["v35_geometry_valid"])[0])
            geometry_valid.append(geometry_ok)
            query_cosine_slots.append(
                np.asarray(aux["v35_query_anchor_cosine"])[0]
                if geometry_ok
                else np.full((model.memory_query_tokens,), np.nan, dtype=np.float32)
            )
            query_beta_slots.append(
                np.asarray(aux["v35_query_anchor_beta"])[0]
                if geometry_ok
                else np.full((model.memory_query_tokens,), np.nan, dtype=np.float32)
            )
            query_cosine_mean.append(
                float(np.asarray(aux["v35_query_anchor_cosine_mean"])[0]) if geometry_ok else float("nan")
            )
            query_cosine_max.append(
                float(np.asarray(aux["v35_query_anchor_cosine_max"])[0]) if geometry_ok else float("nan")
            )
            query_low_alignment_fraction.append(
                float(np.asarray(aux["v35_query_low_alignment_fraction"])[0]) if geometry_ok else float("nan")
            )
            query_beta_mean.append(float(np.asarray(aux["v35_query_beta_mean"])[0]) if geometry_ok else float("nan"))
            query_beta_abs_mean.append(
                float(np.asarray(aux["v35_query_beta_abs_mean"])[0]) if geometry_ok else float("nan")
            )
            query_beta_sign_consistency.append(
                float(np.asarray(aux["v35_query_beta_sign_consistency"])[0]) if geometry_ok else float("nan")
            )
            query_cancellation.append(
                float(np.asarray(aux["v35_query_cancellation_ratio"])[0]) if geometry_ok else float("nan")
            )
            mean_raw_read_anchor_cosine.append(
                float(np.asarray(aux["v35_mean_raw_read_anchor_cosine"])[0]) if geometry_ok else float("nan")
            )
            anchor_retention.append(float(np.asarray(aux["v35_anchor_retention"])[0]) if geometry_ok else float("nan"))
            anchor_delay_reported.append(int(np.asarray(aux["v35_anchor_delay_steps"])[0]) if geometry_ok else -1)
            query_residual_rms.append(
                float(np.asarray(aux["v35_anchor_predicted_read_residual_rms"])[0]) if geometry_ok else float("nan")
            )
            query_relative_residual.append(
                float(np.asarray(aux["v35_anchor_predicted_read_relative_residual"])[0])
                if geometry_ok
                else float("nan")
            )
            anchor_hidden_norm_sq.append(
                float(np.asarray(aux["v35_anchor_hidden_norm_sq"])[0]) if geometry_ok else float("nan")
            )
            # Reset ablation intentionally discards every candidate transition.  Do not
            # fabricate a carried E anchor for its later query telemetry: those later reads
            # see m0, not the candidate state produced here.
            if actual_write and not args.ablate_memory:
                anchor_key = np.asarray(aux["pooled_key"], dtype=np.float32)
                anchor_value = np.asarray(aux["pooled_value"], dtype=np.float32)
                anchor_delay.fill(0)
                anchor_available = True
            elif anchor_available and not args.ablate_memory:
                # The current read saw the pre-transition delay recorded above; this valid
                # non-write transition contributes to the following sampled read.
                anchor_delay += 1
        if not args.ablate_memory:
            mem_state = new_state

        tokens = np.asarray(aux["tokens"])[0]
        mask = np.asarray(aux["token_mask"])[0]
        preds.append(pg.decode(tokens[mask].tolist()).strip())
        surprise.append(float(aux["surprise"][0]))
        delta = unnormalize({"actions": np.asarray(actions)[0, :, :14]})["actions"]
        pred_chunks.append(delta + np.where(arm_mask, state_raw[t], 0.0))  # AbsoluteActions, [horizon, 14]
        if k == 0:
            print(
                f"first call {call_ms[0] / 1e3:.1f}s (incl. compile) | gates theta {np.asarray(aux['theta']).mean():.3f} "
                f"eta {np.asarray(aux['eta']).mean():.3f} alpha {np.asarray(aux['alpha']).mean():.4f}",
                flush=True,
            )
        print(
            f"[{k:3d}] frame {t:5d}  {call_ms[k]:6.0f} ms  {kind:6s}  phase {phase!r}  "
            f"surprise {surprise[k]:.3f}  qcos {query_cosine_mean[k]:.3f}  "
            f"beta {query_beta_mean[k]:.3f}  pred {preds[k]!r}",
            flush=True,
        )
    curve = np.asarray(surprise)
    steady = np.asarray(call_ms[1:] if len(call_ms) > 1 else call_ms)

    print(f"\npred timeline (frames): {_runs(eval_ts, preds)}")
    print(
        f"latency steady {steady.mean():.0f} ms (p50 {np.percentile(steady, 50):.0f}, p95 {np.percentile(steady, 95):.0f}) "
        f"| surprise first {curve[0]:.3f} min {curve.min():.3f} last {curve[-1]:.3f}",
        flush=True,
    )

    out_dir = (
        _project_paths.project_path(_project_paths.V35_DIAGNOSTICS_DIR) / "eval_yam_mem_subtask_raw"
        if model_v35
        else pathlib.Path(__file__).parent / "eval_results"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.ckpt_dir.parent.name}_{args.ckpt_dir.name}_{demo.parent.name}_{demo.name}"
    if args.ablate_memory:
        tag += "_ablate"
    if args.zero_gate:
        tag += "_zerogate"

    np.savez(
        out_dir / f"mem_subtask_{tag}.npz",
        pred_frames=np.asarray(eval_ts),
        surprise=curve,
        call_ms=np.asarray(call_ms),
        write_occurred=np.asarray(write_occurred, dtype=bool),
        transition_kind=np.asarray(transition_kind),
        injected_pre_cast_rms=np.asarray(injected_pre_cast_rms),
        injected_post_cast_rms=np.asarray(injected_post_cast_rms),
        decision_valid=np.asarray(decision_valid, dtype=bool),
        geometry_valid=np.asarray(geometry_valid, dtype=bool),
        query_anchor_cosine_slots=np.asarray(query_cosine_slots),
        query_anchor_beta_slots=np.asarray(query_beta_slots),
        query_anchor_cosine_mean=np.asarray(query_cosine_mean),
        query_anchor_cosine_max=np.asarray(query_cosine_max),
        query_low_alignment_fraction=np.asarray(query_low_alignment_fraction),
        query_anchor_beta_mean=np.asarray(query_beta_mean),
        query_anchor_beta_abs_mean=np.asarray(query_beta_abs_mean),
        query_anchor_beta_sign_consistency=np.asarray(query_beta_sign_consistency),
        query_cancellation_ratio=np.asarray(query_cancellation),
        mean_raw_read_anchor_cosine=np.asarray(mean_raw_read_anchor_cosine),
        anchor_retention=np.asarray(anchor_retention),
        anchor_delay_steps=np.asarray(anchor_delay_reported, dtype=np.int32),
        anchor_predicted_read_residual_rms=np.asarray(query_residual_rms),
        anchor_predicted_read_relative_residual=np.asarray(query_relative_residual),
        anchor_hidden_norm_sq=np.asarray(anchor_hidden_norm_sq),
        stable_id="" if protocol is None else protocol.stable_id,
        split="" if protocol is None else protocol.split,
        collection="" if protocol is None else protocol.collection,
        part="" if protocol is None else protocol.part,
        object_name="" if protocol is None else protocol.object_name,
        prompt="" if protocol is None else protocol.prompt,
        manifest_sha256="" if protocol is None else protocol.manifest_sha256,
        label_sha256="" if protocol is None else protocol.label_sha256,
        memory_stride_frames=stride,
    )
    with open(out_dir / f"mem_subtask_{tag}.txt", "w") as f:
        f.writelines(
            f"{t}\t{s:.4f}\t{kind}\t{p}\n" for t, s, kind, p in zip(eval_ts, curve, transition_kind, preds, strict=True)
        )

    # mp4: every raw top-camera frame with the held prediction overlaid, surprise plot + cursor
    # underneath; the red dot flashes on the frames where a prediction + memory write happened.
    frame_h, frame_w = top[0].shape[:2]
    plot, cursor_cols, (row0, row1) = _render_plot(eval_ts, curve, total, (PLOT_H, frame_w))
    mp4 = out_dir / f"mem_subtask_{tag}.mp4"
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{frame_w}x{frame_h + PLOT_H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(mp4),
        ],
        stdin=subprocess.PIPE,
    )
    for i in range(total):
        k = min(i // stride, len(preds) - 1)
        cam = top[i].copy()
        cv2.putText(cam, f"pred: {preds[k]}", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 0), 2, cv2.LINE_AA)
        cv2.putText(
            cam,
            f"frame {i}  surprise {curve[k]:.3g}",
            (12, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (235, 235, 60),
            1,
            cv2.LINE_AA,
        )
        if i // stride < len(preds) and write_occurred[i // stride] and i % stride < FLASH:
            cv2.circle(cam, (frame_w - 28, 28), 10, (235, 60, 60), -1)
            cv2.putText(cam, "write", (frame_w - 92, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 60, 60), 1, cv2.LINE_AA)
        panel = plot.copy()
        cv2.line(panel, (int(cursor_cols[i]), row0), (int(cursor_cols[i]), row1), (220, 50, 50), 2)
        ffmpeg.stdin.write(np.vstack([cam, panel]).tobytes())
    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"saved {mp4}")

    # per-joint plot: each predicted action chunk drawn over the frames it targets (consecutive
    # chunks overlap by horizon - stride), against the recorded teleop control in raw units.
    horizon = pred_chunks[0].shape[0]
    fig, axes = plt.subplots(7, 2, figsize=(14, 16), sharex=True)
    for j in range(14):
        ax = axes[j % 7, j // 7]
        ax.plot(np.arange(total), actions_raw[:total, j], lw=0.9, color="black", label="teleop gt")
        for k, t in enumerate(eval_ts):
            ax.plot(
                np.arange(t, t + horizon),
                pred_chunks[k][:, j],
                lw=0.7,
                color="tab:orange",
                alpha=0.7,
                label="pred chunk" if k == 0 else None,
            )
        ax.set_ylabel(JOINT_NAMES[j])
    axes[0, 0].legend()
    axes[6, 0].set_xlabel("frame")
    axes[6, 1].set_xlabel("frame")
    fig.suptitle(f"{demo.parent.name}/{demo.name}: predicted action chunks vs teleop control (raw units)")
    fig.tight_layout()
    png = out_dir / f"mem_joints_{tag}.png"
    fig.savefig(png, dpi=140)
    print(f"saved {png}")
    print(f"total time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
