import dataclasses
import functools
import hashlib
import json
import logging
import os
import pathlib
import platform
import re
import tempfile
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
from jax.experimental import checkify
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.project_paths as project_paths
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.v35_authorization as _v35_authorization
import openpi.training.weight_loaders as _weight_loaders

_V35_CHECKPOINT_STEPS_BY_TARGET: dict[int, tuple[int, ...]] = {
    1_000: (250, 500, 1_000),
    2_500: (250, 500, 1_000, 2_500),
    10_000: (250, 500, 1_000, 2_500, 5_000, 10_000),
}
_V35_CHECKPOINT_PROVENANCE_FILENAMES = {
    "calibration": "v35_calibration_artifact.json",
    "episode_manifest": "v35_episode_manifest.json",
    "norm_provenance": "v35_norm_stats_provenance.json",
    "graft_manifest": "v35_initialization_graft_manifest.json",
    "initialization_identity": "v35_initialization_manifest.json",
}
_V35_PILOT_AUTHORIZATION_FILENAME = _v35_authorization.PILOT_AUTHORIZATION_FILENAME
_V35_CONTINUATION_AUTHORIZATION_FILENAME = _v35_authorization.CONTINUATION_AUTHORIZATION_FILENAME
_V35_VALIDATED_STORAGE_SEALS: set[tuple[str, str, tuple[tuple[str, int, int], ...]]] = set()
_V35_RUNTIME_GUARD_NAMES = (
    "write_eligible_count_mismatch",
    "commit_count_mismatch",
    "write_feature_term_count_mismatch",
    "read_state_valid_count_mismatch",
    "read_feature_term_count_mismatch",
    "transition_count_mismatch",
    "write_episode_cell_count_mismatch",
    "read_episode_cell_count_mismatch",
    "degenerate_write",
    "state_invalid_decision",
    "state_valid_mismatch",
    "credit_reachable_mismatch",
    "invalid_decay_gap",
    "nonzero_padding_gap",
    "write_decision_overlap",
    "invalid_memory_cell",
    "invalid_side_label",
    "semantic_control_on_padding",
)


def _configure_v35_runtime_environment(config: _config.TrainConfig) -> None:
    """Fail closed unless every v3.5 runtime path belongs to this project copy."""

    if not getattr(config.model, "memory_v35_enabled", False):
        return
    try:
        project_paths.configure_v35_runtime_environment()
        project_paths.validate_executing_openpi_checkout()
    except project_paths.ProjectRootError as exc:
        raise ValueError(str(exc)) from exc


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
    allow_new_run_from_bootstrap_zero: bool = False,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    wandb_id_path = ckpt_dir / "wandb_id.txt"
    if resuming and wandb_id_path.is_file():
        run_id = wandb_id_path.read_text().strip()
        if not run_id:
            raise ValueError(f"W&B run ID is empty: {wandb_id_path}")
        wandb.init(id=run_id, resume="must", project=config.project_name)
    elif not resuming or allow_new_run_from_bootstrap_zero:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        with wandb_id_path.open("x", encoding="utf-8") as stream:
            stream.write(str(wandb.run.id))
            stream.flush()
            os.fsync(stream.fileno())
    else:
        raise FileNotFoundError(f"Resuming checkpoint has no W&B run ID: {wandb_id_path}")

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _log_training_identity(config: _config.TrainConfig) -> None:
    """Make single-variable memory interventions unambiguous in every startup log."""
    memory_config = getattr(config.model, "memory", None)
    logging.info(
        "Training config: name=%s exp_name=%s eta_scale=%s",
        config.name,
        config.exp_name,
        getattr(memory_config, "eta_scale", None),
    )
    if getattr(config.model, "memory_v35_enabled", False):
        logging.info(
            "v3.5 semantic_training_config_sha256=%s",
            _v35_authorization.semantic_training_config_sha256(config),
        )


def _validate_v35_training_ready(config: _config.TrainConfig) -> None:
    """Refuse a v3.5 optimizer run whose fresh-base/calibration contract is incomplete."""
    model_config = config.model
    if not getattr(model_config, "memory_v35_enabled", False):
        return
    if not getattr(model_config, "memory_v35_calibrated", False):
        raise ValueError(
            "v3.5 training is locked until train-only injection calibration is frozen; set "
            "memory_v35_calibrated=True, memory_v35_calibration_id, memory_injection_c, and "
            "memory_injection_tau from the calibration artifact."
        )
    official_base = "gs://openpi-assets/checkpoints/pi05_base/params"
    if not isinstance(config.weight_loader, _weight_loaders.AuditedPartialCheckpointWeightLoader) or (
        config.weight_loader.params_path != official_base
    ):
        raise ValueError(f"v3.5 must use audited shared-parameter initialization from {official_base!r}.")
    if config.ema_decay is not None:
        raise ValueError("v3.5 uses raw parameters as primary and requires ema_decay=None.")
    if not getattr(model_config, "memory_freeze_injection_gate", False):
        raise ValueError("v3.5 requires the calibrated injection gate to remain frozen.")
    if not getattr(config.data.base_config, "memory_v35_frozen_population", False):
        raise ValueError("v3.5 training requires the frozen 70-episode Gate-A population lock.")
    if config.num_workers != 0:
        raise ValueError("v3.5 exact continuation requires num_workers=0 so no batch can be prefetched.")
    if not config.data.base_config.memory_sequence_buckets:
        raise ValueError("v3.5 exact continuation requires the stateful sequence-bucket sampler.")
    _load_and_validate_v35_calibration_artifact(config)


def _weight_loader_for_run(config: _config.TrainConfig) -> _weight_loaders.WeightLoader:
    """Bind the audited initialization manifest to this launch's checkpoint directory."""
    loader = config.weight_loader
    if isinstance(loader, _weight_loaders.AuditedPartialCheckpointWeightLoader):
        manifest_path = loader.manifest_output_path
        if manifest_path is None:
            manifest_path = str(config.checkpoint_dir / "initialization_graft_manifest.json")
            loader = dataclasses.replace(loader, manifest_output_path=manifest_path)
    return loader


def _cast_frozen_params(config: _config.TrainConfig, params: nnx.State) -> nnx.State:
    """Cast ordinary frozen leaves to BF16 while keeping the v3.5 gate exactly FP32."""
    cast_filter = config.freeze_filter
    if getattr(config.model, "memory_v35_enabled", False):
        cast_filter = nnx.All(cast_filter, nnx.Not(MEMORY_INJECT_GATE_FILTER))
    return nnx_utils.state_map(params, cast_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))


def _validate_v35_initialized_gate(config: _config.TrainConfig, params: nnx.State) -> None:
    """Fail closed unless the initialized v3.5 injection gate matches calibration exactly."""
    if not getattr(config.model, "memory_v35_enabled", False):
        return
    gate_leaves = params.filter(MEMORY_INJECT_GATE_FILTER).flat_state()
    if len(gate_leaves) != 1:
        raise ValueError(f"v3.5 expected exactly one memory_inject_w leaf, found {len(gate_leaves)}.")
    gate_w = np.asarray(jax.device_get(next(iter(gate_leaves.values())).value))
    if gate_w.dtype != np.float32:
        raise ValueError(f"v3.5 memory_inject_w must remain float32, got {gate_w.dtype}.")
    expected = np.float32(getattr(config.model, "memory_injection_gate_init", 0.5))
    effective = np.tanh(gate_w.astype(np.float32)).astype(np.float32)
    if not np.all(np.isfinite(effective)) or not np.allclose(effective, expected, rtol=0.0, atol=1e-6):
        raise ValueError(
            "v3.5 memory_inject_w does not match the frozen calibrated gate: "
            f"expected tanh(w)={expected}, observed range=[{effective.min()}, {effective.max()}]."
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _calibration_canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _v35_norm_artifact_paths(config: _config.TrainConfig) -> tuple[pathlib.Path, pathlib.Path]:
    data_factory = config.data
    asset_id = data_factory.assets.asset_id or data_factory.repo_id
    assets_dir = pathlib.Path(data_factory.assets.assets_dir or config.assets_dirs)
    norm_dir = assets_dir / asset_id
    return norm_dir / "norm_stats.json", norm_dir / "norm_stats_provenance.json"


def _load_self_hashed_json(
    path: pathlib.Path,
    *,
    hash_key: str,
    description: str,
) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {path}")
    recorded_hash = value.get(hash_key)
    unsigned = {key: item for key, item in value.items() if key != hash_key}
    actual_hash = _sha256_bytes(_canonical_json(unsigned).encode("utf-8"))
    if recorded_hash != actual_hash:
        raise ValueError(f"{description} self-hash is invalid: {path}")
    return value


def _stream_file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_v35_train_storage_seal(
    config: _config.TrainConfig,
    provenance: dict[str, Any],
    *,
    selected_episode_indices: list[int],
) -> str | None:
    """Recompute the train-only storage seal without opening held-out episode media."""
    if not getattr(config.data.base_config, "memory_v35_frozen_population", False):
        return None
    storage = provenance.get("train_storage")
    if not isinstance(storage, dict):
        raise ValueError("v3.5 production norm provenance requires a train_storage seal.")
    if "root" in storage:
        raise ValueError("v3.5 train_storage must not embed an absolute machine-local root.")
    if storage.get("root_contract") != "memory_project-relative-v1":
        raise ValueError("v3.5 train_storage requires the memory-project-relative-v1 root contract.")
    root_value = storage.get("root_relative")
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("v3.5 train_storage.root_relative must be a project-relative dataset directory.")
    root = project_paths.project_path(root_value)
    configured_root = pathlib.Path(config.data.base_config.lerobot_dataset_root).expanduser().resolve()
    if root != configured_root or root != project_paths.project_path(project_paths.V35_DATASET_DIR):
        raise ValueError(
            "v3.5 train_storage root does not match the registered project-local dataset: "
            f"recorded={root}, configured={configured_root}."
        )
    if not root.is_dir():
        raise ValueError(f"v3.5 train_storage dataset directory is missing: {root}.")
    if storage.get("selected_episode_indices") != selected_episode_indices:
        raise ValueError("v3.5 train_storage episode selection does not match the frozen train split.")
    if storage.get("scope") != "selected train episode parquet, optional videos, plus structural meta files":
        raise ValueError("v3.5 train_storage has the wrong portable storage scope.")

    selected = set(selected_episode_indices)
    episode_pattern = re.compile(r"episode_(\d{6})\.(?:parquet|mp4)$")
    files: list[pathlib.Path] = []
    data_seen: set[int] = set()
    video_seen: set[int] = set()
    storage_directories = ["data"]
    if (root / "videos").is_dir():
        storage_directories.append("videos")
    for directory_name in storage_directories:
        directory = root / directory_name
        if not directory.is_dir():
            raise ValueError(f"v3.5 dataset is missing required {directory_name}/ storage under {root}.")
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            match = episode_pattern.search(path.name)
            if match is None:
                raise ValueError(f"v3.5 dataset has an unrecognized episode storage filename: {path}.")
            episode_index = int(match.group(1))
            if episode_index not in selected:
                # Deliberately do not open held-out parquet/video content.
                continue
            files.append(path)
            (data_seen if directory_name == "data" else video_seen).add(episode_index)
    missing_data = sorted(selected - data_seen)
    missing_video = sorted(selected - video_seen) if "videos" in storage_directories else []
    if missing_data or missing_video:
        raise ValueError(
            "v3.5 train-only storage seal cannot find every selected episode: "
            f"missing_data={missing_data}, missing_video={missing_video}."
        )
    meta = root / "meta"
    if not meta.is_dir():
        raise ValueError(f"v3.5 dataset is missing structural meta/ storage under {root}.")
    files.extend(
        path for path in meta.rglob("*") if path.is_file() and "v35_training_bundle" not in path.relative_to(meta).parts
    )

    ordered_paths = sorted(set(files), key=lambda item: item.relative_to(root).as_posix())
    recorded_records = storage.get("files")
    if not isinstance(recorded_records, list):
        raise ValueError("v3.5 train_storage.files must be a file-record list.")
    aggregate = _sha256_bytes(_canonical_json(recorded_records).encode("utf-8"))
    if storage.get("sha256") != aggregate:
        raise ValueError("v3.5 train_storage aggregate SHA256 is invalid.")
    stat_signature = tuple(
        (
            path.relative_to(root).as_posix(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in ordered_paths
    )
    recorded_paths_and_sizes = [
        (record.get("path"), record.get("size")) if isinstance(record, dict) else (None, None)
        for record in recorded_records
    ]
    if recorded_paths_and_sizes != [(path, size) for path, size, _ in stat_signature]:
        raise ValueError("v3.5 train_storage file list/size/SHA256 no longer matches the dataset.")
    cache_key = (str(root), aggregate, stat_signature)
    if cache_key in _V35_VALIDATED_STORAGE_SEALS:
        return aggregate
    actual_records = [
        {"path": relative, "size": size, "sha256": _stream_file_sha256(root / relative)}
        for relative, size, _ in stat_signature
    ]
    if recorded_records != actual_records:
        raise ValueError("v3.5 train_storage file list/size/SHA256 no longer matches the dataset.")
    _V35_VALIDATED_STORAGE_SEALS.add(cache_key)
    return aggregate


def _v35_recorded_train_storage_sha256(config: _config.TrainConfig) -> str | None:
    _, provenance_path = _v35_norm_artifact_paths(config)
    try:
        provenance = json.loads(provenance_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read v3.5 norm provenance {provenance_path}: {exc}") from exc
    storage = provenance.get("train_storage") if isinstance(provenance, dict) else None
    return storage.get("sha256") if isinstance(storage, dict) else None


def _load_and_validate_v35_calibration_artifact(config: _config.TrainConfig) -> dict[str, Any]:
    """Authenticate the train-only calibration and prove that the launch uses its values."""
    model_config = config.model
    artifact_path_value = getattr(model_config, "memory_v35_calibration_path", None)
    if not artifact_path_value:
        raise ValueError("v3.5 requires a calibration artifact path.")
    artifact_path = pathlib.Path(artifact_path_value)
    if not artifact_path.is_file():
        raise ValueError(f"v3.5 calibration artifact does not exist: {artifact_path}")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read v3.5 calibration artifact {artifact_path}: {exc}") from exc
    if not isinstance(artifact, dict) or not isinstance(artifact.get("payload"), dict):
        raise ValueError("v3.5 calibration artifact must contain a payload object.")
    payload = artifact["payload"]
    digest = hashlib.sha256(_calibration_canonical_json(payload).encode("utf-8")).hexdigest()
    expected_id = f"sha256:{digest}"
    if artifact.get("artifact_sha256") != digest or artifact.get("calibration_id") != expected_id:
        raise ValueError("v3.5 calibration artifact payload hash/ID is invalid.")
    if getattr(model_config, "memory_v35_calibration_id", None) != expected_id:
        raise ValueError("v3.5 config calibration ID does not match its artifact.")
    if payload.get("schema_version") != "openpi.v35.injection-calibration.v1" or payload.get("status") != "pass":
        raise ValueError("v3.5 calibration artifact has the wrong schema or is not passing.")

    data_config = getattr(config.data, "base_config", None)
    if data_config is None:
        raise ValueError("v3.5 requires an explicit base data config for manifest verification.")
    manifest_sha256 = getattr(data_config, "memory_episode_manifest_sha256", None)
    if (
        not isinstance(manifest_sha256, str)
        or len(manifest_sha256) != 64
        or any(char not in "0123456789abcdef" for char in manifest_sha256)
    ):
        raise ValueError("v3.5 requires the frozen episode manifest SHA256 in its data config.")
    manifest_path = pathlib.Path(data_config.memory_episode_manifest_path)
    if not manifest_path.is_file():
        raise ValueError(f"v3.5 episode manifest does not exist: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
        raise ValueError("v3.5 episode manifest bytes do not match the configured SHA256.")
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"v3.5 episode manifest is invalid JSON: {exc}") from exc
    episodes = manifest.get("episodes") if isinstance(manifest, dict) else None
    if not isinstance(episodes, list):
        raise ValueError("v3.5 episode manifest must contain an episodes list.")
    train_ids = {
        str(record.get("stable_id", "")).strip()
        for record in episodes
        if isinstance(record, dict) and bool(record.get("include", True)) and record.get("split") == "train"
    }
    population = payload.get("population", {})
    artifact_ids = population.get("stable_ids")
    if (
        population.get("split") != "train"
        or population.get("episode_count") != 54
        or not isinstance(artifact_ids, list)
        or len(artifact_ids) != 54
        or len(set(artifact_ids)) != 54
        or set(artifact_ids) != train_ids
    ):
        raise ValueError("v3.5 calibration membership is not exactly the frozen 54-episode training split.")
    provenance = payload.get("provenance", {})
    if provenance.get("split_sha256") != manifest_sha256:
        raise ValueError("v3.5 calibration split hash does not match the frozen episode manifest.")
    membership = [{"split": "train", "stable_id": stable_id} for stable_id in artifact_ids]
    membership_sha256 = hashlib.sha256(_calibration_canonical_json(membership).encode("utf-8")).hexdigest()
    if provenance.get("observed_membership_sha256") != membership_sha256:
        raise ValueError("v3.5 calibration membership hash is invalid.")

    dataset_protocol_sha256 = _validate_v35_norm_stats_provenance(
        config,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
    )
    if provenance.get("dataset_sha256") != dataset_protocol_sha256:
        raise ValueError("v3.5 calibration dataset hash does not match train-only norm provenance.")

    parameters = payload.get("parameters", {})
    if (
        # The calibration artifact records the FP32 runtime alpha (the memory core computes
        # decay in float32); compare in that exact precision rather than float64.
        np.float32(parameters.get("alpha_step", float("nan"))) != np.float32(model_config.memory.alpha_step)
        or float(parameters.get("memory_injection_c", float("nan"))) != float(model_config.memory_injection_c)
        or float(parameters.get("memory_injection_tau", float("nan"))) != float(model_config.memory_injection_tau)
    ):
        raise ValueError("v3.5 c/tau/alpha config does not exactly match the calibration artifact.")
    gate = payload.get("gate", {})
    if (
        gate.get("target_effective_tanh_gate") != 0.5
        or gate.get("open_channel_count") != model_config.memory.d_value
        or not payload.get("gates", {}).get("passes", False)
        or not payload.get("gates", {}).get("all_episodes_train", False)
        or not payload.get("gates", {}).get("fixed_effective_gate_is_0_5", False)
    ):
        raise ValueError("v3.5 calibration gate/population invariants are not passing.")
    return artifact


def _validate_v35_norm_stats_provenance(
    config: _config.TrainConfig,
    *,
    manifest: dict[str, Any],
    manifest_sha256: str,
) -> str:
    """Reject stale/all-episode normalization assets before constructing the data loader."""
    data_factory = config.data
    data_config = data_factory.base_config
    norm_path, provenance_path = _v35_norm_artifact_paths(config)
    norm_dir = norm_path.parent
    if not provenance_path.is_file() or not norm_path.is_file():
        raise ValueError(f"v3.5 requires committed train-only norm stats and provenance in {norm_dir}.")
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read v3.5 norm provenance {provenance_path}: {exc}") from exc
    episodes = manifest["episodes"]
    converted_records = [
        record
        for record in episodes
        if isinstance(record, dict) and record.get("episode_index", record.get("lerobot_episode_index")) is not None
    ]
    train_records = sorted(
        (
            record
            for record in converted_records
            if bool(record.get("include", True)) and record.get("split") == "train"
        ),
        key=lambda record: int(record.get("episode_index", record.get("lerobot_episode_index"))),
    )
    expected_indices = [
        int(record.get("episode_index", record.get("lerobot_episode_index"))) for record in train_records
    ]
    expected_ids = [str(record["stable_id"]).strip() for record in train_records]
    manifest_info = provenance.get("manifest", {})
    selection = provenance.get("selection", {})
    computation = provenance.get("computation", {})
    norm_info = provenance.get("norm_stats", {})
    frame_counts = selection.get("selected_episode_frame_counts")
    expected_manifest_relative = project_paths.project_relative_path(
        data_config.memory_episode_manifest_path
    ).as_posix()
    if (
        provenance.get("schema_version") != 2
        or provenance.get("status") != "complete"
        or provenance.get("repo_id") != data_factory.repo_id
        or manifest_info.get("sha256") != manifest_sha256
        or manifest_info.get("path_relative") != expected_manifest_relative
        or "path" in manifest_info
        or manifest_info.get("active_split") != "train"
        or manifest_info.get("split_seed") != data_config.memory_manifest_split_seed
        or selection.get("dataset_num_episodes") != len(converted_records)
        or selection.get("selected_num_episodes") != len(expected_ids)
        or selection.get("selected_episode_indices") != expected_indices
        or selection.get("selected_stable_ids") != expected_ids
        or not isinstance(frame_counts, list)
        or len(frame_counts) != len(expected_ids)
        or any(not isinstance(count, int) or count <= 0 for count in frame_counts)
        or selection.get("selected_num_frames") != sum(frame_counts)
        or not isinstance(selection.get("dataset_episode_frame_protocol_sha256"), str)
        or len(selection.get("dataset_episode_frame_protocol_sha256")) != 64
        or any(char not in "0123456789abcdef" for char in selection.get("dataset_episode_frame_protocol_sha256", ""))
        or computation.get("protocol") != "raw-train-rows-delta-action-horizon-v1"
        or computation.get("processed_base_rows") != sum(frame_counts)
        or computation.get("drop_last_rows") != 0
        or norm_info.get("file") != "norm_stats.json"
    ):
        raise ValueError("v3.5 norm provenance does not exactly describe the frozen train-only split.")
    expected_batches = (sum(frame_counts) + computation.get("requested_batch_size", 0) - 1) // max(
        computation.get("requested_batch_size", 0), 1
    )
    if computation.get("num_batches_including_partial_final_batch") != expected_batches:
        raise ValueError("v3.5 norm provenance has an inconsistent processed batch count.")
    if hashlib.sha256(norm_path.read_bytes()).hexdigest() != norm_info.get("sha256"):
        raise ValueError("v3.5 norm_stats.json bytes do not match their provenance SHA256.")
    _validate_v35_train_storage_seal(
        config,
        provenance,
        selected_episode_indices=expected_indices,
    )
    return selection["dataset_episode_frame_protocol_sha256"]


def _write_json_once(path: pathlib.Path, payload: dict[str, Any]) -> None:
    # The initialization identity is an immutable-stage artifact: consumers
    # (v35_prepare_pilot's _load_immutable_json) accept only canonical compact sorted
    # JSON with exactly one trailing newline, so the producer must emit those bytes.
    serialized = _canonical_json(payload) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == serialized:
            return
        raise FileExistsError(f"refusing to overwrite a different initialization identity: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _write_v35_initialization_identity(config: _config.TrainConfig, params: nnx.State) -> pathlib.Path | None:
    """Bind official source provenance and the actual post-cast step-0 tree to this run."""
    if not getattr(config.model, "memory_v35_enabled", False):
        return None
    checkpoint_dir = pathlib.Path(config.checkpoint_dir)
    graft_path = checkpoint_dir / "initialization_graft_manifest.json"
    if not graft_path.is_file():
        raise FileNotFoundError(f"v3.5 audited loader did not produce its graft manifest: {graft_path}")
    graft = _load_self_hashed_json(
        graft_path,
        hash_key="manifest_sha256",
        description="v3.5 graft manifest",
    )
    recorded_graft_hash = graft["manifest_sha256"]

    gate_leaf = next(iter(params.filter(MEMORY_INJECT_GATE_FILTER).flat_state().values()))
    raw_gate = np.asarray(jax.device_get(gate_leaf.value), dtype=np.float32)
    effective_gate = np.tanh(raw_gate)
    actual_step0_hash = _weight_loaders.parameter_tree_sha256(params.to_pure_dict())
    calibration = _load_and_validate_v35_calibration_artifact(config)
    calibration_payload = calibration["payload"]
    if calibration_payload["provenance"].get("source_sha256") != actual_step0_hash:
        raise ValueError("v3.5 calibration was not produced from this exact fresh step-0 parameter tree.")
    raw_gate_hash = hashlib.sha256(raw_gate.tobytes(order="C")).hexdigest()
    if calibration_payload["gate"].get("raw_w_sha256") != raw_gate_hash:
        raise ValueError("v3.5 calibration raw gate hash does not match initialized memory_inject_w.")
    semantic_config_sha256 = _v35_authorization.semantic_training_config_sha256(config)
    run_id_sha256 = _v35_authorization.run_id_sha256(
        config_name=config.name,
        experiment_name=config.exp_name,
        initialization_seed=config.seed,
        initialization_parameter_tree_sha256=actual_step0_hash,
        calibration_artifact_id=calibration["calibration_id"],
        semantic_config_sha256=semantic_config_sha256,
    )

    calibration_path = pathlib.Path(config.model.memory_v35_calibration_path)
    episode_manifest_path = pathlib.Path(config.data.base_config.memory_episode_manifest_path)
    norm_stats_path, norm_provenance_path = _v35_norm_artifact_paths(config)
    payload: dict[str, Any] = {
        "format_version": 2,
        "config_name": config.name,
        "experiment_name": config.exp_name,
        "official_source_uri": config.weight_loader.params_path,
        "initialization_seed": config.seed,
        "graft_manifest_file": graft_path.name,
        "graft_manifest_sha256": recorded_graft_hash,
        "source_tree_sha256": graft["tree_hashes"]["source_sha256"],
        "target_schema_sha256": graft["tree_hashes"]["target_schema_sha256"],
        "actual_step0_parameter_tree_sha256": actual_step0_hash,
        "calibration_id": calibration["calibration_id"],
        "run_id_sha256": run_id_sha256,
        "semantic_training_config_sha256": semantic_config_sha256,
        "step0_checkpoint": 0,
        "memory_inject_w_dtype": str(np.asarray(jax.device_get(gate_leaf.value)).dtype),
        "memory_inject_w_sha256": raw_gate_hash,
        "effective_gate_min": float(np.min(effective_gate)),
        "effective_gate_max": float(np.max(effective_gate)),
        "memory_calibration": {
            # Record the FP32 runtime alpha so plain-equality checks against the
            # calibration artifact's fp32-recorded value hold exactly.
            "alpha_step": float(np.float32(config.model.memory.alpha_step)),
            "memory_injection_c": float(config.model.memory_injection_c),
            "memory_injection_tau": float(config.model.memory_injection_tau),
        },
        "artifact_hashes": {
            "calibration_artifact_sha256": _sha256_bytes(calibration_path.read_bytes()),
            "episode_manifest_sha256": _sha256_bytes(episode_manifest_path.read_bytes()),
            "norm_stats_provenance_sha256": _sha256_bytes(norm_provenance_path.read_bytes()),
            "norm_stats_sha256": _sha256_bytes(norm_stats_path.read_bytes()),
            "train_storage_sha256": _v35_recorded_train_storage_sha256(config),
            "initialization_graft_manifest_file_sha256": _sha256_bytes(graft_path.read_bytes()),
            "initialization_graft_manifest_self_sha256": recorded_graft_hash,
        },
        "checkpoint_protocol": {
            "allowed_continuation_targets": sorted(_V35_CHECKPOINT_STEPS_BY_TARGET),
            "checkpoint_steps_by_target": {
                str(target): list(steps) for target, steps in _V35_CHECKPOINT_STEPS_BY_TARGET.items()
            },
            "keep_period": 250,
            "step_labels": "completed_optimizer_updates",
        },
    }
    payload["identity_sha256"] = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    identity_path = checkpoint_dir / "initialization_manifest.json"
    _write_json_once(identity_path, payload)
    return identity_path


def _validate_v35_checkpoint_protocol(
    config: _config.TrainConfig,
    *,
    resuming: bool,
    latest_step: int | None = None,
) -> None:
    """Enforce the preregistered pilot/extension/full-budget rung schedule."""
    if not getattr(config.model, "memory_v35_enabled", False):
        return
    target = config.num_train_steps
    expected_steps = _V35_CHECKPOINT_STEPS_BY_TARGET.get(target)
    if expected_steps is None:
        raise ValueError(
            "v3.5 num_train_steps must be one of the frozen continuation targets "
            f"{tuple(_V35_CHECKPOINT_STEPS_BY_TARGET)}; got {target}."
        )
    if not config.checkpoint_by_completed_updates or tuple(config.checkpoint_steps) != expected_steps:
        raise ValueError(
            f"v3.5 target {target} requires exact completed-update checkpoint_steps={expected_steps}; "
            f"got {tuple(config.checkpoint_steps)}."
        )
    if config.keep_period != 250:
        raise ValueError("v3.5 requires keep_period=250 so every frozen rung remains retained.")
    if not resuming:
        if target != 1_000:
            raise ValueError("a fresh v3.5 run must execute the frozen 1,000-update pilot first.")
        return
    if latest_step is None:
        raise ValueError("v3.5 resume validation requires the latest checkpoint step.")
    allowed_resume_sources = {
        # A live checkpoint's own metadata cannot authorize its mutable params/optimizer.
        # Resume only externally sealed source rungs named by the launch authorization.
        1_000: (0,),
        2_500: (1_000,),
        10_000: (1_000, 2_500),
    }[target]
    if latest_step not in allowed_resume_sources:
        raise ValueError(
            f"v3.5 target {target} may resume only from authorization-linked source rungs "
            f"{allowed_resume_sources}; latest checkpoint is {latest_step}. Intermediate crash resumes "
            "require separately sealed external rung/hash evidence."
        )


def _v35_provenance_source_paths(
    config: _config.TrainConfig,
    identity_path: pathlib.Path,
) -> dict[str, pathlib.Path]:
    _, norm_provenance_path = _v35_norm_artifact_paths(config)
    sources = {
        _V35_CHECKPOINT_PROVENANCE_FILENAMES["calibration"]: pathlib.Path(config.model.memory_v35_calibration_path),
        _V35_CHECKPOINT_PROVENANCE_FILENAMES["episode_manifest"]: pathlib.Path(
            config.data.base_config.memory_episode_manifest_path
        ),
        _V35_CHECKPOINT_PROVENANCE_FILENAMES["norm_provenance"]: norm_provenance_path,
        _V35_CHECKPOINT_PROVENANCE_FILENAMES["graft_manifest"]: pathlib.Path(config.checkpoint_dir)
        / "initialization_graft_manifest.json",
        _V35_CHECKPOINT_PROVENANCE_FILENAMES["initialization_identity"]: identity_path,
    }
    if config.v35_pilot_authorization_path is not None:
        sources[_V35_PILOT_AUTHORIZATION_FILENAME] = project_paths.project_path(config.v35_pilot_authorization_path)
    return sources


def _validate_v35_root_identity(
    config: _config.TrainConfig,
    identity_path: pathlib.Path,
) -> dict[str, Any]:
    """Authenticate the immutable root identity against this launch and its current artifacts."""
    if not identity_path.is_file():
        raise FileNotFoundError(f"v3.5 initialization identity is missing: {identity_path}")
    identity = _load_self_hashed_json(
        identity_path,
        hash_key="identity_sha256",
        description="v3.5 initialization identity",
    )
    expected_protocol = {
        "allowed_continuation_targets": sorted(_V35_CHECKPOINT_STEPS_BY_TARGET),
        "checkpoint_steps_by_target": {
            str(target): list(steps) for target, steps in _V35_CHECKPOINT_STEPS_BY_TARGET.items()
        },
        "keep_period": 250,
        "step_labels": "completed_optimizer_updates",
    }
    expected_memory_calibration = {
        "alpha_step": float(np.float32(config.model.memory.alpha_step)),
        "memory_injection_c": float(config.model.memory_injection_c),
        "memory_injection_tau": float(config.model.memory_injection_tau),
    }
    expected_semantic_config_sha256 = _v35_authorization.semantic_training_config_sha256(config)
    calibration = _load_and_validate_v35_calibration_artifact(config)
    graft_path = pathlib.Path(config.checkpoint_dir) / "initialization_graft_manifest.json"
    graft = _load_self_hashed_json(
        graft_path,
        hash_key="manifest_sha256",
        description="v3.5 graft manifest",
    )
    calibration_path = pathlib.Path(config.model.memory_v35_calibration_path)
    episode_manifest_path = pathlib.Path(config.data.base_config.memory_episode_manifest_path)
    norm_stats_path, norm_provenance_path = _v35_norm_artifact_paths(config)
    expected_artifact_hashes = {
        "calibration_artifact_sha256": _sha256_bytes(calibration_path.read_bytes()),
        "episode_manifest_sha256": _sha256_bytes(episode_manifest_path.read_bytes()),
        "norm_stats_provenance_sha256": _sha256_bytes(norm_provenance_path.read_bytes()),
        "norm_stats_sha256": _sha256_bytes(norm_stats_path.read_bytes()),
        "train_storage_sha256": _v35_recorded_train_storage_sha256(config),
        "initialization_graft_manifest_file_sha256": _sha256_bytes(graft_path.read_bytes()),
        "initialization_graft_manifest_self_sha256": graft["manifest_sha256"],
    }
    expected_run_id_sha256 = _v35_authorization.run_id_sha256(
        config_name=config.name,
        experiment_name=config.exp_name,
        initialization_seed=config.seed,
        initialization_parameter_tree_sha256=identity.get("actual_step0_parameter_tree_sha256"),
        calibration_artifact_id=calibration["calibration_id"],
        semantic_config_sha256=expected_semantic_config_sha256,
    )
    if (
        identity.get("format_version") != 2
        or identity.get("config_name") != config.name
        or identity.get("experiment_name") != config.exp_name
        or identity.get("official_source_uri") != config.weight_loader.params_path
        or identity.get("initialization_seed") != config.seed
        or identity.get("calibration_id") != calibration["calibration_id"]
        or identity.get("run_id_sha256") != expected_run_id_sha256
        or identity.get("semantic_training_config_sha256") != expected_semantic_config_sha256
        or identity.get("memory_calibration") != expected_memory_calibration
        or identity.get("artifact_hashes") != expected_artifact_hashes
        or identity.get("checkpoint_protocol") != expected_protocol
    ):
        raise ValueError("v3.5 root initialization identity does not match the current config/artifacts.")
    return identity


def _snapshot_v35_checkpoint_provenance(
    config: _config.TrainConfig,
    identity_path: pathlib.Path,
) -> dict[str, bytes]:
    """Return the authenticated immutable byte payload copied into every checkpoint."""
    _validate_v35_root_identity(config, identity_path)
    sources = _v35_provenance_source_paths(config, identity_path)
    try:
        return {name: path.read_bytes() for name, path in sources.items()}
    except OSError as exc:
        raise ValueError(f"cannot snapshot v3.5 checkpoint provenance: {exc}") from exc


def _validate_v35_resume_checkpoint_assets(
    config: _config.TrainConfig,
    *,
    checkpoint_step: int,
    identity_path: pathlib.Path,
) -> dict[str, bytes]:
    """Require the latest checkpoint to embed byte-identical launch provenance."""
    expected = _snapshot_v35_checkpoint_provenance(config, identity_path)
    # Checkpoint 0 is finalized before Gate A/B/C can consume it and before their reducer can
    # issue the pilot authorization.  The authorization is therefore authenticated externally
    # when training resumes from 0, then included in the returned snapshot so checkpoint 250+
    # embeds it.  Every other rung must already contain it byte-for-byte.
    embedded_expected = dict(expected)
    if checkpoint_step == 0:
        embedded_expected.pop(_V35_PILOT_AUTHORIZATION_FILENAME, None)
    assets_dir = pathlib.Path(config.checkpoint_dir) / str(checkpoint_step) / "assets"
    if checkpoint_step == 0:
        optional_pilot = assets_dir / _V35_PILOT_AUTHORIZATION_FILENAME
        external_pilot = expected.get(_V35_PILOT_AUTHORIZATION_FILENAME)
        if optional_pilot.is_file() and (
            external_pilot is None or optional_pilot.read_bytes() != external_pilot
        ):
            raise ValueError("v3.5 checkpoint 0 embeds a pilot authorization different from the validated external one.")
    for name, expected_bytes in embedded_expected.items():
        embedded_path = assets_dir / name
        if not embedded_path.is_file():
            raise FileNotFoundError(f"v3.5 checkpoint {checkpoint_step} is missing embedded provenance asset {name}.")
        if embedded_path.read_bytes() != expected_bytes:
            raise ValueError(
                f"v3.5 checkpoint {checkpoint_step} provenance asset {name} is not byte-identical "
                "to the authenticated run root."
            )
    return expected


def _step_labels_and_save_decision(
    config: _config.TrainConfig, *, loop_step: int, start_step: int
) -> tuple[int, bool, int]:
    """Return metric label, save decision, and checkpoint label for one completed update."""
    completed_step = loop_step + 1
    if config.checkpoint_by_completed_updates:
        should_save = (
            completed_step in config.checkpoint_steps
            if config.checkpoint_steps
            else completed_step % config.save_interval == 0 or completed_step == config.num_train_steps
        )
        return completed_step, should_save, completed_step
    should_save = (loop_step % config.save_interval == 0 and loop_step > start_step) or (
        loop_step == config.num_train_steps - 1
    )
    return loop_step, should_save, loop_step


_V35_CUMULATIVE_TELEMETRY_SCHEMA_VERSION = 1
_V35_CUMULATIVE_COUNT_KEYS = (
    "accepted_update_count",
    "finite_accepted_update_count",
    "pre_shared_severe_clip_count",
    "pre_shared_update_count",
    "write_feature_cap_bind_numerator",
    "write_feature_cap_bind_denominator",
    "read_feature_cap_bind_numerator",
    "read_feature_cap_bind_denominator",
)


def _new_v35_cumulative_telemetry() -> dict[str, int | float]:
    return {
        "schema_version": _V35_CUMULATIVE_TELEMETRY_SCHEMA_VERSION,
        **dict.fromkeys(_V35_CUMULATIVE_COUNT_KEYS, 0),
        "pre_shared_grad_norm_max": 0.0,
    }


def _validate_v35_cumulative_telemetry(telemetry: dict[str, Any], *, completed_updates: int) -> dict[str, int | float]:
    expected_keys = {
        "schema_version",
        *_V35_CUMULATIVE_COUNT_KEYS,
        "pre_shared_grad_norm_max",
    }
    if not isinstance(telemetry, dict) or set(telemetry) != expected_keys:
        raise ValueError("v3.5 cumulative telemetry has an invalid schema.")
    if telemetry["schema_version"] != _V35_CUMULATIVE_TELEMETRY_SCHEMA_VERSION:
        raise ValueError("v3.5 cumulative telemetry has an unsupported schema version.")
    for key in _V35_CUMULATIVE_COUNT_KEYS:
        value = telemetry[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"v3.5 cumulative telemetry {key} must be a nonnegative integer.")
    grad_norm_max = telemetry["pre_shared_grad_norm_max"]
    if not isinstance(grad_norm_max, int | float) or not np.isfinite(grad_norm_max) or grad_norm_max < 0:
        raise ValueError("v3.5 cumulative pre_shared_grad_norm_max must be finite and nonnegative.")
    for numerator, denominator in (
        ("pre_shared_severe_clip_count", "pre_shared_update_count"),
        ("write_feature_cap_bind_numerator", "write_feature_cap_bind_denominator"),
        ("read_feature_cap_bind_numerator", "read_feature_cap_bind_denominator"),
    ):
        if telemetry[numerator] > telemetry[denominator]:
            raise ValueError(f"v3.5 cumulative telemetry {numerator} exceeds {denominator}.")
    for key in ("accepted_update_count", "finite_accepted_update_count", "pre_shared_update_count"):
        if telemetry[key] != completed_updates:
            raise ValueError(
                f"v3.5 cumulative telemetry {key}={telemetry[key]} does not match "
                f"completed updates {completed_updates}."
            )
    return telemetry


def _restore_and_validate_v35_authorized_source_checkpoint(
    config: _config.TrainConfig,
    *,
    checkpoint_manager: Any,
    checkpoint_step: int,
    state_shape: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    source_authorization: _v35_authorization.AuthorizationRecord,
) -> tuple[training_utils.TrainState, dict[str, Any], str]:
    """Restore and externally authenticate every mutable component of a v3.5 source rung."""

    train_state = _checkpoints.restore_state(
        checkpoint_manager,
        state_shape,
        data_loader,
        step=checkpoint_step,
    )
    jax.block_until_ready(train_state)
    if int(train_state.step) != checkpoint_step:
        raise ValueError(
            "v3.5 restored optimizer-update count does not match its checkpoint label: "
            f"state.step={int(train_state.step)}, checkpoint={checkpoint_step}."
        )
    cumulative_telemetry = _checkpoints.restore_v35_runtime_state(
        config.checkpoint_dir,
        checkpoint_step,
        data_loader,
    )
    _validate_v35_cumulative_telemetry(cumulative_telemetry, completed_updates=checkpoint_step)
    _validate_v35_initialized_gate(config, train_state.params)

    step_dir = pathlib.Path(config.checkpoint_dir) / str(checkpoint_step)
    assets_dir = step_dir / "assets"
    parameter_tree_sha256 = _weight_loaders.parameter_tree_sha256(train_state.params.to_pure_dict())
    try:
        runtime_hashes = {
            "runtime_identity_sha256": _sha256_bytes(
                (assets_dir / _checkpoints.V35_RUNTIME_IDENTITY_FILENAME).read_bytes()
            ),
            "cumulative_telemetry_sha256": _sha256_bytes(
                (assets_dir / _checkpoints.V35_CUMULATIVE_TELEMETRY_FILENAME).read_bytes()
            ),
            "data_iterator_state_sha256": _sha256_bytes(
                (assets_dir / _checkpoints.V35_DATA_ITERATOR_STATE_FILENAME).read_bytes()
            ),
        }
    except OSError as exc:
        raise ValueError(f"cannot hash restored v3.5 runtime assets: {exc}") from exc
    optimizer_state_sha256 = _checkpoints.v35_checkpoint_component_tree_sha256(step_dir / "train_state")
    _v35_authorization.validate_live_source_checkpoint_binding(
        source_authorization,
        completed_updates=checkpoint_step,
        parameter_tree_sha256=parameter_tree_sha256,
        optimizer_state_sha256=optimizer_state_sha256,
        **runtime_hashes,
    )
    return train_state, cumulative_telemetry, parameter_tree_sha256


def _v35_integer_metric(info: dict[str, at.Array], key: str) -> int:
    if key not in info:
        raise ValueError(f"accepted v3.5 update is missing cumulative telemetry metric {key}.")
    value = float(np.sum(np.asarray(jax.device_get(info[key]), dtype=np.float64)))
    rounded = round(value)
    if not np.isfinite(value) or value < 0 or not np.isclose(value, rounded, rtol=0.0, atol=1e-5):
        raise ValueError(f"accepted v3.5 update has invalid count metric {key}={value}.")
    return int(rounded)


def _accumulate_v35_cumulative_telemetry(
    telemetry: dict[str, int | float],
    info: dict[str, at.Array],
    *,
    completed_updates: int,
) -> None:
    """Commit host Gate-D counters only after checkify accepted the optimizer update."""
    update_count = _v35_integer_metric(info, "diagnostic/v35_pre_shared_clip_update_count")
    if update_count != 1:
        raise ValueError(f"each accepted v3.5 optimizer update must report update_count=1, got {update_count}.")
    severe = _v35_integer_metric(info, "diagnostic/v35_pre_shared_clip_severe_count")
    write_bind = _v35_integer_metric(info, "diagnostic/v35_write_feature_clip_bind_sum")
    write_terms = _v35_integer_metric(info, "diagnostic/v35_write_feature_term_count")
    read_bind = _v35_integer_metric(info, "diagnostic/v35_read_feature_clip_bind_sum")
    read_terms = _v35_integer_metric(info, "diagnostic/v35_read_feature_term_count")
    grad_norm_max = float(
        np.max(np.asarray(jax.device_get(info["diagnostic/v35_pre_shared_clip_grad_norm_max"]), dtype=np.float64))
    )
    if not np.isfinite(grad_norm_max) or grad_norm_max < 0:
        raise ValueError("accepted v3.5 update has non-finite pre-shared gradient norm telemetry.")

    telemetry["accepted_update_count"] += 1
    # The update reaches this point only after the checkified finite/runtime guard accepted it.
    telemetry["finite_accepted_update_count"] += 1
    telemetry["pre_shared_severe_clip_count"] += severe
    telemetry["pre_shared_update_count"] += update_count
    telemetry["write_feature_cap_bind_numerator"] += write_bind
    telemetry["write_feature_cap_bind_denominator"] += write_terms
    telemetry["read_feature_cap_bind_numerator"] += read_bind
    telemetry["read_feature_cap_bind_denominator"] += read_terms
    telemetry["pre_shared_grad_norm_max"] = max(float(telemetry["pre_shared_grad_norm_max"]), grad_norm_max)
    _validate_v35_cumulative_telemetry(telemetry, completed_updates=completed_updates)


_PER_POSITION_METRIC_SUFFIX = re.compile(r"_p\d+$")


def _is_per_position_metric(key: str) -> bool:
    """Only suppress expanded vector entries such as ``..._p17`` from console output."""
    return _PER_POSITION_METRIC_SUFFIX.search(key) is not None


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _pad_probe_grids(correct_grid: at.Array, active_grid: at.Array, max_steps: int) -> tuple[at.Array, at.Array]:
    if correct_grid.shape != active_grid.shape or correct_grid.ndim != 1:
        raise ValueError("probe correct/active grids must be matching one-dimensional arrays.")
    if correct_grid.shape[0] > max_steps:
        raise ValueError(f"probe grid length {correct_grid.shape[0]} exceeds configured maximum {max_steps}.")
    pad = max_steps - correct_grid.shape[0]
    return jnp.pad(correct_grid, (0, pad)), jnp.pad(active_grid, (0, pad))


def _aux_macro_ce(class_ce_sum: at.Array, class_count: at.Array) -> at.Array:
    """Class-balanced macro CE (v3.4 plan 5.1): mean over PRESENT classes of the per-class mean
    CE, so frequent phase labels cannot dominate and re-reward the phase-only representation."""
    present = class_count > 0
    per_class = jnp.where(present, class_ce_sum / jnp.maximum(class_count, 1.0), 0.0)
    return jnp.sum(per_class) / jnp.maximum(jnp.sum(present.astype(jnp.float32)), 1.0)


def _v35_cell_macro_ce(cell_ce_sum: at.Array, cell_episode_count: at.Array) -> at.Array:
    """Episode-first, then equal-present-cell macro CE for v3.5 side supervision."""
    present = cell_episode_count > 0
    per_cell = jnp.where(present, cell_ce_sum / jnp.maximum(cell_episode_count, 1.0), 0.0)
    return jnp.sum(per_cell) / jnp.maximum(jnp.sum(present.astype(jnp.float32)), 1.0)


def _v35_loss_info(chunked_loss: dict[str, at.Array]) -> dict[str, at.Array]:
    """Keep v3.5 numerators and denominators explicit for exact logging-window pooling."""
    info = {}
    for branch in ("write", "read"):
        prefix = f"v35_{branch}"
        info[f"diagnostic/{prefix}_episode_correct"] = chunked_loss[f"{prefix}_episode_correct_cell"]
        info[f"diagnostic/{prefix}_episode_count"] = chunked_loss[f"{prefix}_episode_count_cell"]
        info[f"diagnostic/{prefix}_feature_grad_norm_sum"] = chunked_loss[f"{prefix}_feature_grad_norm_sum"]
        info[f"diagnostic/{prefix}_feature_clip_bind_sum"] = chunked_loss[f"{prefix}_feature_clip_bind_sum"]
        info[f"diagnostic/{prefix}_feature_term_count"] = chunked_loss[f"{prefix}_frame_count"]
    for key in (
        "v35_write_eligible_count",
        "v35_commit_success_count",
        "v35_degenerate_write_count",
        "v35_commit_residual_ratio_sum",
        "v35_commit_residual_ratio_max",
        "v35_commit_relative_residual_sum",
        "v35_commit_relative_residual_max",
        "v35_state_invalid_d_count",
        "v35_state_valid_mismatch_count",
        "v35_reachable_count",
        "v35_reachable_mismatch_count",
        "v35_read_state_valid_count",
        "v35_invalid_gap_count",
        "v35_padding_gap_count",
        "v35_illegal_write_decision_overlap_count",
        "v35_use_pressure_count",
        "v35_invalid_cell_count",
        "v35_raw_read_rms_sum",
        "v35_injected_pre_cast_rms_sum",
        "v35_injected_post_cast_rms_sum",
        "v35_transition_count",
    ):
        info[f"diagnostic/{key}"] = chunked_loss[key]
    return info


def _v35_runtime_guard_vector(
    config: _config.TrainConfig,
    observation: _model.Observation,
    loss_info: dict[str, at.Array],
) -> at.Array:
    """Return one fail-closed bit for every v3.5 runtime accounting invariant.

    The sampler supplies the objective denominators, while the recurrent model reports what
    actually committed and read.  A degenerate association, malformed gap, invalid side/cell,
    or recurrent-state disagreement can otherwise silently reduce the implemented denominator
    and let an invalid optimizer update look healthy.  Counts are therefore reconciled on the
    complete effective batch (including every gradient-accumulation microbatch).
    """
    required_fields = (
        "seq_step_mask",
        "seq_write_mask",
        "seq_decision_mask",
        "seq_read_state_valid",
        "seq_read_credit_reachable",
        "seq_decay_gap_before",
        "seq_use_pressure_mask",
        "seq_memory_cell",
        "seq_side_label",
    )
    missing = [name for name in required_fields if getattr(observation, name) is None]
    if missing:
        raise ValueError(f"v3.5 runtime guard requires observation fields: {missing}.")

    step = observation.seq_step_mask
    write = observation.seq_write_mask
    decision = observation.seq_decision_mask
    state_valid = observation.seq_read_state_valid
    credit_reachable = observation.seq_read_credit_reachable
    gap = observation.seq_decay_gap_before
    use_pressure = observation.seq_use_pressure_mask
    cell = observation.seq_memory_cell
    side = observation.seq_side_label

    if write.shape != step.shape or decision.shape != step.shape or state_valid.shape != step.shape:
        raise ValueError("v3.5 runtime masks must have identical [..., T] shapes.")
    if credit_reachable.shape != step.shape or gap.shape != step.shape or use_pressure.shape != step.shape:
        raise ValueError("v3.5 runtime reachability/gap/use-pressure fields must match seq_step_mask.")
    if cell.shape != step.shape[:-1] or side.shape != step.shape[:-1]:
        raise ValueError("v3.5 cell and side fields must have the sample-leading shape of seq_step_mask.")

    def count(mask: at.Array) -> at.Array:
        return jnp.sum(mask.astype(jnp.float32))

    expected_write_frames = count(write & step)
    expected_read_frames = count(decision & state_valid & step)
    expected_transitions = count(step)
    write_episode = jnp.any(write & step, axis=-1)
    read_episode = jnp.any(decision & state_valid & step, axis=-1)

    num_cells = config.model.memory_num_side_cells
    safe_cell = jnp.clip(cell, 0, num_cells - 1)
    cell_onehot = jax.nn.one_hot(safe_cell, num_cells, dtype=jnp.float32)
    sample_axes = tuple(range(cell_onehot.ndim - 1))

    def episode_count_by_cell(present: at.Array) -> at.Array:
        return jnp.sum(cell_onehot * present[..., None].astype(jnp.float32), axis=sample_axes)

    expected_write_episodes = episode_count_by_cell(write_episode)
    expected_read_episodes = episode_count_by_cell(read_episode)

    def metric(name: str) -> at.Array:
        key = f"diagnostic/{name}"
        if key not in loss_info:
            raise ValueError(f"v3.5 runtime guard is missing model metric {key!r}.")
        return loss_info[key]

    telemetry_nonzero = (
        "v35_degenerate_write_count",
        "v35_state_invalid_d_count",
        "v35_state_valid_mismatch_count",
        "v35_reachable_mismatch_count",
        "v35_invalid_gap_count",
        "v35_padding_gap_count",
        "v35_illegal_write_decision_overlap_count",
        "v35_invalid_cell_count",
    )
    telemetry_violations = [metric(name) != 0 for name in telemetry_nonzero]

    invalid_side = count((side < 0) | (side >= 2)) != 0
    semantic_control_on_padding = count((write | decision | use_pressure) & ~step) != 0
    violations = (
        metric("v35_write_eligible_count") != expected_write_frames,
        metric("v35_commit_success_count") != expected_write_frames,
        metric("v35_write_feature_term_count") != expected_write_frames,
        metric("v35_read_state_valid_count") != expected_read_frames,
        metric("v35_read_feature_term_count") != expected_read_frames,
        metric("v35_transition_count") != expected_transitions,
        jnp.any(metric("v35_write_episode_count") != expected_write_episodes),
        jnp.any(metric("v35_read_episode_count") != expected_read_episodes),
        *telemetry_violations,
        invalid_side,
        semantic_control_on_padding,
    )
    if len(violations) != len(_V35_RUNTIME_GUARD_NAMES):
        raise AssertionError("v3.5 runtime guard names and predicates are out of sync.")
    return jnp.stack([jnp.asarray(value, dtype=bool) for value in violations])


def _check_v35_runtime_guard(violations: at.Array) -> None:
    """Emit named checkify assertions for a runtime-guard predicate vector."""
    if violations.shape != (len(_V35_RUNTIME_GUARD_NAMES),):
        raise ValueError(
            "v3.5 runtime guard vector has the wrong shape: "
            f"expected {(len(_V35_RUNTIME_GUARD_NAMES),)}, found {violations.shape}."
        )
    for index, name in enumerate(_V35_RUNTIME_GUARD_NAMES):
        checkify.check(~violations[index], f"v3.5 runtime invariant failed: {name}")


def _checked_v35_train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Functionalized per-update v3.5 guard used only by the official training loop."""
    new_state, info = train_step(config, rng, state, batch)
    info = dict(info)
    _check_v35_runtime_guard(info.pop("_v35_runtime_guard"))
    return new_state, info


def _aux_group_metrics(chunked_loss: dict[str, at.Array], side_class_ids: tuple[int, ...]) -> dict[str, at.Array]:
    """Per-class-group (phase vs side-bearing) accuracy numerators/denominators for logging."""
    class_count = chunked_loss["aux_count_class"]
    class_correct = chunked_loss["aux_correct_class"]
    num_classes = class_count.shape[0]
    side = jnp.zeros((num_classes,), dtype=bool)
    if side_class_ids:
        side = side.at[jnp.asarray(side_class_ids, dtype=jnp.int32)].set(True)
    return {
        "diagnostic/aux_side_correct": jnp.sum(jnp.where(side, class_correct, 0.0)),
        "diagnostic/aux_side_count": jnp.sum(jnp.where(side, class_count, 0.0)),
        "diagnostic/aux_phase_correct": jnp.sum(jnp.where(side, 0.0, class_correct)),
        "diagnostic/aux_phase_count": jnp.sum(jnp.where(side, 0.0, class_count)),
    }


def _write_diagnostic_sums(chunked_loss: dict[str, at.Array]) -> dict[str, at.Array]:
    """Unreduced write telemetry with a shared valid-write denominator.

    Keeping numerators/counts intact here lets `_reduce_infos` pool exactly across samples,
    unequal sequence buckets, microbatches, and optimizer updates.
    """
    return {
        "diagnostic/write_inner_grad_sum": jnp.sum(chunked_loss["write_grad_norm_sum"]),
        "diagnostic/write_inner_valid_count": jnp.sum(chunked_loss["write_valid_count"]),
        "diagnostic/write_inner_clip_count": jnp.sum(chunked_loss["write_clip_count"]),
        "diagnostic/write_inner_severe_clip_count": jnp.sum(chunked_loss["write_severe_clip_count"]),
        "diagnostic/write_inner_grad_max": jnp.max(chunked_loss["write_grad_norm_max"]),
    }


_LADDER_RUNGS = ("ladder_writer", "ladder_read")

# v3.4 Section 6: the online probe-ladder heads. Their gradients are removed from the main
# optimizer path entirely (a probe must not scale main-model updates through the global clip
# norm) and applied by a separate constant-LR SGD in train_step.
LADDER_PROBE_FILTER = nnx_utils.PathRegex(r".*ladder_(writer|read)_head.*")

# Every parameter on the memory path: the Titans core (memory/*), the read/write query
# compressors and conditioner, and the v3.2+/v3.4 interface params (memory_inject_w,
# memory_gate, memory_aux_*, memory_slot_embedding, state_null_embedding). Used by the
# optional `memory_grad_clip` group pre-clip in train_step.
MEMORY_PATH_FILTER = nnx_utils.PathRegex(r".*(memory|query_compressor|query_conditioner|state_null_embedding).*")
MEMORY_INJECT_GATE_FILTER = nnx_utils.PathRegex(r".*memory_inject_w.*")


def _reduce_infos(infos: list[dict[str, at.Array]]) -> dict[str, np.ndarray]:
    stacked_infos = common_utils.stack_forest(infos)
    reduced = jax.device_get(jax.tree.map(lambda x: jnp.mean(x, axis=0), stacked_infos))
    norm_count_key = "_expensive_norm_count"
    if norm_count_key in reduced:
        # Parameter/gradient norms traverse the multi-billion-parameter tree. They are sampled
        # once per logging window inside train_step, so reduce them by their explicit count
        # rather than diluting the single sample by the number of updates in the window.
        count = np.sum(jax.device_get(stacked_infos[norm_count_key]), axis=0)
        for key in ("grad_norm", "param_norm"):
            reduced[key] = np.sum(jax.device_get(stacked_infos[key]), axis=0) / np.maximum(count, 1)
        reduced.pop(norm_count_key)
    write_count_key = "diagnostic/write_inner_valid_count"
    if write_count_key in reduced:
        # Exact pooled ratios. Averaging per-sample/per-update means would over-weight short
        # sequences and sparse buckets.
        count = np.sum(jax.device_get(stacked_infos[write_count_key]), axis=0)
        grad_sum = np.sum(jax.device_get(stacked_infos["diagnostic/write_inner_grad_sum"]), axis=0)
        clip_count = np.sum(jax.device_get(stacked_infos["diagnostic/write_inner_clip_count"]), axis=0)
        severe_count = np.sum(jax.device_get(stacked_infos["diagnostic/write_inner_severe_clip_count"]), axis=0)
        reduced.update(
            {
                "diagnostic/write_inner_grad_norm": grad_sum / np.maximum(count, 1),
                "diagnostic/write_inner_clip_fraction": clip_count / np.maximum(count, 1),
                "diagnostic/write_inner_severe_clip_fraction": severe_count / np.maximum(count, 1),
            }
        )
        for key in (
            write_count_key,
            "diagnostic/write_inner_grad_sum",
            "diagnostic/write_inner_clip_count",
            "diagnostic/write_inner_severe_clip_count",
        ):
            reduced.pop(key)
    # This metric is already a max over sequence position and batch inside each optimizer
    # update. Preserve its meaning across the logging window instead of averaging those maxima.
    window_max_key = "diagnostic/write_inner_grad_max"
    if window_max_key in reduced:
        reduced[window_max_key] = np.max(jax.device_get(stacked_infos[window_max_key]), axis=0)
    probe_count_key = "diagnostic/probe_count"
    if probe_count_key in reduced:
        # Diagnostic accuracies are ratios over every live probe in the entire log window, not
        # an unweighted mean of per-batch ratios (which biases sparse/zero-probe buckets).
        count = np.sum(jax.device_get(stacked_infos[probe_count_key]), axis=0)
        correct = np.sum(jax.device_get(stacked_infos["diagnostic/probe_correct"]), axis=0)
        visible_count = np.sum(jax.device_get(stacked_infos["diagnostic/probe_visible_count"]), axis=0)
        visible_correct = np.sum(jax.device_get(stacked_infos["diagnostic/probe_visible_correct"]), axis=0)
        loss_numerator = np.sum(jax.device_get(stacked_infos["diagnostic/probe_loss_numerator"]), axis=0)
        reduced.update(
            {
                "diagnostic/probe_loss": loss_numerator / np.maximum(count, 1),
                "diagnostic/probe_accuracy": correct / np.maximum(count, 1),
                "diagnostic/probe_accuracy_visible": visible_correct / np.maximum(visible_count, 1),
                "diagnostic/probe_accuracy_hidden": (correct - visible_correct) / np.maximum(count - visible_count, 1),
            }
        )
        for key in (
            "diagnostic/probe_correct",
            "diagnostic/probe_visible_count",
            "diagnostic/probe_visible_correct",
            "diagnostic/probe_loss_numerator",
        ):
            reduced.pop(key)

    grid_correct_key = "diagnostic/probe_correct_grid"
    if grid_correct_key in reduced:
        correct_grid = np.sum(jax.device_get(stacked_infos[grid_correct_key]), axis=0)
        active_grid = np.sum(jax.device_get(stacked_infos["diagnostic/probe_active_grid"]), axis=0)
        reduced.pop(grid_correct_key)
        reduced.pop("diagnostic/probe_active_grid")
        reduced["diagnostic/probe_accuracy_by_step"] = correct_grid / np.maximum(active_grid, 1)

    if "diagnostic/v35_write_eligible_count" in reduced:

        def total(key):
            return np.sum(jax.device_get(stacked_infos[key]), axis=0)

        eligible = total("diagnostic/v35_write_eligible_count")
        committed = total("diagnostic/v35_commit_success_count")
        read_count = total("diagnostic/v35_read_state_valid_count")
        transition_count = total("diagnostic/v35_transition_count")
        reduced.update(
            {
                "diagnostic/v35_commit_success_fraction": committed / np.maximum(eligible, 1),
                "diagnostic/v35_degenerate_write_fraction": total("diagnostic/v35_degenerate_write_count")
                / np.maximum(eligible, 1),
                "diagnostic/v35_commit_residual_ratio": total("diagnostic/v35_commit_residual_ratio_sum")
                / np.maximum(committed, 1),
                "diagnostic/v35_commit_relative_residual": total("diagnostic/v35_commit_relative_residual_sum")
                / np.maximum(committed, 1),
                "diagnostic/v35_reachable_fraction": total("diagnostic/v35_reachable_count")
                / np.maximum(read_count, 1),
                "diagnostic/v35_raw_read_rms": total("diagnostic/v35_raw_read_rms_sum") / np.maximum(read_count, 1),
                "diagnostic/v35_injected_pre_cast_rms": total("diagnostic/v35_injected_pre_cast_rms_sum")
                / np.maximum(transition_count, 1),
                "diagnostic/v35_injected_post_cast_rms": total("diagnostic/v35_injected_post_cast_rms_sum")
                / np.maximum(transition_count, 1),
                "diagnostic/v35_commit_residual_ratio_max": np.max(
                    jax.device_get(stacked_infos["diagnostic/v35_commit_residual_ratio_max"]), axis=0
                ),
                "diagnostic/v35_commit_relative_residual_max": np.max(
                    jax.device_get(stacked_infos["diagnostic/v35_commit_relative_residual_max"]), axis=0
                ),
            }
        )
        for branch in ("write", "read"):
            term_count = total(f"diagnostic/v35_{branch}_feature_term_count")
            reduced[f"diagnostic/v35_{branch}_feature_grad_norm"] = total(
                f"diagnostic/v35_{branch}_feature_grad_norm_sum"
            ) / np.maximum(term_count, 1)
            reduced[f"diagnostic/v35_{branch}_feature_clip_bind_fraction"] = total(
                f"diagnostic/v35_{branch}_feature_clip_bind_sum"
            ) / np.maximum(term_count, 1)
        for key in (
            "diagnostic/v35_commit_residual_ratio_sum",
            "diagnostic/v35_commit_relative_residual_sum",
            "diagnostic/v35_raw_read_rms_sum",
            "diagnostic/v35_injected_pre_cast_rms_sum",
            "diagnostic/v35_injected_post_cast_rms_sum",
            "diagnostic/v35_write_feature_grad_norm_sum",
            "diagnostic/v35_write_feature_clip_bind_sum",
            "diagnostic/v35_write_feature_term_count",
            "diagnostic/v35_read_feature_grad_norm_sum",
            "diagnostic/v35_read_feature_clip_bind_sum",
            "diagnostic/v35_read_feature_term_count",
        ):
            reduced.pop(key)

    severe_key = "diagnostic/v35_pre_shared_clip_severe_count"
    if severe_key in reduced:
        severe_count = np.sum(jax.device_get(stacked_infos[severe_key]), axis=0)
        update_count = np.sum(jax.device_get(stacked_infos["diagnostic/v35_pre_shared_clip_update_count"]), axis=0)
        grad_norm_sum = np.sum(jax.device_get(stacked_infos["diagnostic/v35_pre_shared_clip_grad_norm_sum"]), axis=0)
        grad_norm_max = np.max(jax.device_get(stacked_infos["diagnostic/v35_pre_shared_clip_grad_norm_max"]), axis=0)
        reduced.update(
            {
                severe_key: severe_count,
                "diagnostic/v35_pre_shared_clip_update_count": update_count,
                "diagnostic/v35_pre_shared_clip_severe_fraction": severe_count / np.maximum(update_count, 1),
                "diagnostic/v35_pre_shared_clip_grad_norm": grad_norm_sum / np.maximum(update_count, 1),
                "diagnostic/v35_pre_shared_clip_grad_norm_max": grad_norm_max,
            }
        )
        reduced.pop("diagnostic/v35_pre_shared_clip_grad_norm_sum")

    # v3.4 aux/ladder accuracies: any (X_correct, X_count) pair becomes a window-exact ratio.
    for key in [k for k in reduced if k.endswith("_count") and k.replace("_count", "_correct") in reduced]:
        correct_key = key.replace("_count", "_correct")
        count = np.sum(jax.device_get(stacked_infos[key]), axis=0)
        correct = np.sum(jax.device_get(stacked_infos[correct_key]), axis=0)
        reduced[key.replace("_count", "_accuracy")] = correct / np.maximum(count, 1)
        reduced.pop(key)
        reduced.pop(correct_key)
    return reduced


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert ordinary frozen params to bfloat16. The calibrated v3.5 injection gate is a
        # numerical control parameter and must remain exactly float32.
        params = _cast_frozen_params(config, params)

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(_weight_loader_for_run(config), train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        if isinstance(chunked_loss, dict):
            # Subtask co-training: combine the flow and (weighted) token CE losses, log both.
            loss = jnp.mean(chunked_loss["flow"]) + model.ce_loss_weight * jnp.mean(chunked_loss["ce"])
            info = {"flow_loss": jnp.mean(chunked_loss["flow"]), "ce_loss": jnp.mean(chunked_loss["ce"])}
            if "write_grad_norm_sum" in chunked_loss:
                # Core-steepness telemetry (v34 postmortems): healthy ~0.5-3; ramping toward
                # ~50 preceded both explosion cycles by several hundred steps.
                info.update(_write_diagnostic_sums(chunked_loss))
            if "probe_ce_sum" in chunked_loss:
                # Probe outputs are logged under an explicitly diagnostic namespace. Detached
                # diagnostic mode has weight zero and therefore cannot affect the main loss.
                count = jnp.sum(chunked_loss["probe_count"])
                correct = jnp.sum(chunked_loss["probe_correct"])
                vis_count = jnp.sum(chunked_loss["probe_count_visible"])
                vis_correct = jnp.sum(chunked_loss["probe_correct_visible"])
                probe_loss_numerator = jnp.sum(chunked_loss["probe_ce_sum"])
                if model.memory_probe_weight > 0:
                    loss += model.memory_probe_weight * probe_loss_numerator / jnp.maximum(count, 1)
                info.update(
                    {
                        "diagnostic/probe_loss_numerator": probe_loss_numerator,
                        "diagnostic/probe_count": count,
                        "diagnostic/probe_correct": correct,
                        "diagnostic/probe_visible_count": vis_count,
                        "diagnostic/probe_visible_correct": vis_correct,
                    }
                )
                # Keep the per-position numerator and denominator separate and pad both to the
                # configured maximum sequence length. Bucket batches have different static T;
                # padding an already-divided accuracy would incorrectly count absent positions
                # as errors and would also make stack_forest fail across bucket shapes.
                correct_grid = jnp.sum(chunked_loss["probe_correct_grid"], axis=0)
                active_grid = jnp.sum(chunked_loss["probe_active_grid"], axis=0)
                correct_grid, active_grid = _pad_probe_grids(correct_grid, active_grid, config.model.memory_seq_steps)
                info.update(
                    {
                        "diagnostic/probe_correct_grid": correct_grid,
                        "diagnostic/probe_active_grid": active_grid,
                    }
                )
            if "aux_ce_class_sum" in chunked_loss:
                # v3.4 plan 5.1: class-balanced macro CE, trained by the MAIN optimizer.
                aux_loss = _aux_macro_ce(chunked_loss["aux_ce_class_sum"], chunked_loss["aux_count_class"])
                loss += model.memory_aux_loss_weight * aux_loss
                info["aux_loss"] = aux_loss
                info.update(_aux_group_metrics(chunked_loss, model.memory_aux_side_class_ids))
                if "aux_margin_sum" in chunked_loss:
                    aux_margin = chunked_loss["aux_margin_sum"] / jnp.maximum(chunked_loss["aux_margin_count"], 1.0)
                    loss += model.memory_aux_margin_weight * aux_margin
                    info["aux_margin"] = aux_margin
            if "v35_write_ce_cell_sum" in chunked_loss:
                write_side_loss = _v35_cell_macro_ce(
                    chunked_loss["v35_write_ce_cell_sum"],
                    chunked_loss["v35_write_episode_count_cell"],
                )
                read_side_loss = _v35_cell_macro_ce(
                    chunked_loss["v35_read_ce_cell_sum"],
                    chunked_loss["v35_read_episode_count_cell"],
                )
                loss += model.memory_write_side_loss_weight * write_side_loss
                loss += model.memory_read_side_loss_weight * read_side_loss
                info["v35_write_side_loss"] = write_side_loss
                info["v35_read_side_loss"] = read_side_loss
                info.update(_v35_loss_info(chunked_loss))
            if "ladder_writer_ce_sum" in chunked_loss:
                # Section 6 online rungs: features are stop-gradient'ed inside the model, so
                # this term reaches ONLY the ladder heads -- whose grads train_step removes
                # from the main optimizer path and applies with the isolated probe SGD.
                for rung in _LADDER_RUNGS:
                    rung_loss = chunked_loss[f"{rung}_ce_sum"] / jnp.maximum(chunked_loss[f"{rung}_count"], 1.0)
                    loss += rung_loss
                    info[f"diagnostic/{rung}_loss"] = rung_loss
                    info[f"diagnostic/{rung}_correct"] = chunked_loss[f"{rung}_correct"]
                    info[f"diagnostic/{rung}_count"] = chunked_loss[f"{rung}_count"]
            if observation.seq_step_mask is not None:
                info.update(
                    sequence_bucket_steps=jnp.asarray(observation.seq_step_mask.shape[1], dtype=jnp.float32),
                    sequence_valid_fraction=jnp.mean(observation.seq_step_mask),
                )
            return loss, info
        return jnp.mean(chunked_loss), {}

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    if config.gradient_accumulation_steps == 1:
        # Keep the original full-batch path unchanged. Besides avoiding extra overhead on
        # H200, this preserves the exact pre-accumulation random-number and reduction order.
        (loss, loss_info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
            model, train_rng, observation, actions
        )
    else:
        accumulation_steps = config.gradient_accumulation_steps
        if observation.state.shape[0] != accumulation_steps or actions.shape[0] != accumulation_steps:
            raise ValueError(
                "Accumulated training batches must have leading shape "
                f"[{accumulation_steps}, microbatch]; got state {observation.state.shape} and actions {actions.shape}."
            )

        # Probe CE is a ratio over every live quiz in the effective B12 batch. Dividing each
        # B4 numerator by its own count and then averaging would change the objective, so use
        # the data-only full-batch denominator for every microbatch contribution.
        global_probe_count = (
            None if observation.seq_probe_mask is None else jnp.sum(observation.seq_probe_mask.astype(jnp.float32))
        )

        # v3.4 objectives are ratios with data-only denominators too: compute the GLOBAL
        # per-class counts (aux macro CE) and per-rung frame counts (ladder) from the full
        # accumulated batch so summing microbatch contributions reproduces the exact
        # full-batch objective.
        aux_class_count_global = None
        if observation.seq_subtask_class is not None:
            num_aux_classes = getattr(config.model, "memory_aux_num_classes", 0)
            aux_cls = observation.seq_subtask_class
            aux_valid = (aux_cls >= 0) & (aux_cls < num_aux_classes) & observation.seq_step_mask
            aux_onehot = jax.nn.one_hot(jnp.clip(aux_cls, 0, num_aux_classes - 1), num_aux_classes)
            aux_class_count_global = jnp.sum(aux_onehot * aux_valid[..., None].astype(jnp.float32), axis=(0, 1, 2))
            aux_margin_count_global = jnp.sum(aux_valid.astype(jnp.float32))
        ladder_count_global = None
        if observation.seq_side_label is not None and observation.seq_evidence_mask is not None:
            side_ok = (observation.seq_side_label >= 0) & (observation.seq_side_label < 2)
            ladder_count_global = {
                "ladder_writer": jnp.sum(
                    (observation.seq_evidence_mask & observation.seq_step_mask & side_ok[..., None]).astype(jnp.float32)
                ),
                "ladder_read": jnp.sum(
                    (observation.seq_waiting_mask & observation.seq_step_mask & side_ok[..., None]).astype(jnp.float32)
                ),
            }
        v35_cell_count_global = None
        if getattr(config.model, "memory_v35_enabled", False):
            if (
                observation.seq_memory_cell is None
                or observation.seq_side_label is None
                or observation.seq_write_mask is None
                or observation.seq_decision_mask is None
                or observation.seq_read_state_valid is None
            ):
                raise ValueError("v3.5 accumulation requires cell, side, write, decision, and state-valid fields.")
            num_cells = config.model.memory_num_side_cells
            cell = observation.seq_memory_cell
            cell_ok = (cell >= 0) & (cell < num_cells)
            safe_cell = jnp.clip(cell, 0, num_cells - 1)
            cell_onehot = jax.nn.one_hot(safe_cell, num_cells, dtype=jnp.float32)
            side_ok = (observation.seq_side_label >= 0) & (observation.seq_side_label < 2)
            write_episode = jnp.any(observation.seq_write_mask & observation.seq_step_mask, axis=-1)
            read_episode = jnp.any(
                observation.seq_decision_mask & observation.seq_read_state_valid & observation.seq_step_mask,
                axis=-1,
            )

            def episode_count_by_cell(present):
                weight = (present & side_ok & cell_ok).astype(jnp.float32)
                return jnp.sum(cell_onehot * weight[..., None], axis=(0, 1))

            v35_cell_count_global = {
                "write": episode_count_by_cell(write_episode),
                "read": episode_count_by_cell(read_episode),
            }

        def microbatch_loss_fn(model, rng, micro_observation, micro_actions):
            chunked_loss = model.compute_loss(rng, micro_observation, micro_actions, train=True)
            if not isinstance(chunked_loss, dict):
                return jnp.mean(chunked_loss) / accumulation_steps, {}

            flow_loss = jnp.mean(chunked_loss["flow"])
            ce_loss = jnp.mean(chunked_loss["ce"])
            loss = (flow_loss + model.ce_loss_weight * ce_loss) / accumulation_steps
            # These are additive contributions to the metrics of the effective global batch.
            info = {"flow_loss": flow_loss / accumulation_steps, "ce_loss": ce_loss / accumulation_steps}
            if "write_grad_norm_sum" in chunked_loss:
                info.update(_write_diagnostic_sums(chunked_loss))
            if "probe_ce_sum" in chunked_loss:
                if global_probe_count is None:
                    raise ValueError("Probe losses require observation.seq_probe_mask.")
                count = jnp.sum(chunked_loss["probe_count"])
                correct = jnp.sum(chunked_loss["probe_correct"])
                vis_count = jnp.sum(chunked_loss["probe_count_visible"])
                vis_correct = jnp.sum(chunked_loss["probe_correct_visible"])
                probe_loss_numerator = jnp.sum(chunked_loss["probe_ce_sum"])
                if model.memory_probe_weight > 0:
                    loss += model.memory_probe_weight * probe_loss_numerator / jnp.maximum(global_probe_count, 1)
                info.update(
                    {
                        "diagnostic/probe_loss_numerator": probe_loss_numerator,
                        "diagnostic/probe_count": count,
                        "diagnostic/probe_correct": correct,
                        "diagnostic/probe_visible_count": vis_count,
                        "diagnostic/probe_visible_correct": vis_correct,
                    }
                )
                correct_grid = jnp.sum(chunked_loss["probe_correct_grid"], axis=0)
                active_grid = jnp.sum(chunked_loss["probe_active_grid"], axis=0)
                correct_grid, active_grid = _pad_probe_grids(correct_grid, active_grid, config.model.memory_seq_steps)
                info.update(
                    {
                        "diagnostic/probe_correct_grid": correct_grid,
                        "diagnostic/probe_active_grid": active_grid,
                    }
                )
            if "aux_ce_class_sum" in chunked_loss:
                if aux_class_count_global is None:
                    raise ValueError("aux losses require observation.seq_subtask_class.")
                present = aux_class_count_global > 0
                per_class = jnp.where(
                    present, chunked_loss["aux_ce_class_sum"] / jnp.maximum(aux_class_count_global, 1.0), 0.0
                )
                aux_contrib = jnp.sum(per_class) / jnp.maximum(jnp.sum(present.astype(jnp.float32)), 1.0)
                loss += model.memory_aux_loss_weight * aux_contrib
                info["aux_loss"] = aux_contrib  # additive: sums to the exact global macro CE
                info.update(_aux_group_metrics(chunked_loss, model.memory_aux_side_class_ids))
                if "aux_margin_sum" in chunked_loss:
                    aux_margin = chunked_loss["aux_margin_sum"] / jnp.maximum(aux_margin_count_global, 1.0)
                    loss += model.memory_aux_margin_weight * aux_margin
                    info["aux_margin"] = aux_margin
            if "v35_write_ce_cell_sum" in chunked_loss:
                if v35_cell_count_global is None:
                    raise ValueError("v3.5 losses require global effective-batch cell counts.")

                def v35_micro_contribution(branch):
                    global_count = v35_cell_count_global[branch]
                    present = global_count > 0
                    per_cell = jnp.where(
                        present,
                        chunked_loss[f"v35_{branch}_ce_cell_sum"] / jnp.maximum(global_count, 1.0),
                        0.0,
                    )
                    return jnp.sum(per_cell) / jnp.maximum(jnp.sum(present.astype(jnp.float32)), 1.0)

                write_side_loss = v35_micro_contribution("write")
                read_side_loss = v35_micro_contribution("read")
                loss += model.memory_write_side_loss_weight * write_side_loss
                loss += model.memory_read_side_loss_weight * read_side_loss
                info["v35_write_side_loss"] = write_side_loss
                info["v35_read_side_loss"] = read_side_loss
                info.update(_v35_loss_info(chunked_loss))
            if "ladder_writer_ce_sum" in chunked_loss:
                if ladder_count_global is None:
                    raise ValueError("ladder probe losses require the seq_side/evidence/waiting fields.")
                for rung in _LADDER_RUNGS:
                    rung_loss = chunked_loss[f"{rung}_ce_sum"] / jnp.maximum(ladder_count_global[rung], 1.0)
                    loss += rung_loss
                    info[f"diagnostic/{rung}_loss"] = rung_loss
                    info[f"diagnostic/{rung}_correct"] = chunked_loss[f"{rung}_correct"]
                    info[f"diagnostic/{rung}_count"] = chunked_loss[f"{rung}_count"]
            if micro_observation.seq_step_mask is not None:
                info.update(
                    sequence_bucket_steps=jnp.asarray(
                        micro_observation.seq_step_mask.shape[1] / accumulation_steps, dtype=jnp.float32
                    ),
                    sequence_valid_fraction=jnp.mean(micro_observation.seq_step_mask) / accumulation_steps,
                )
            return loss, info

        value_and_grad = nnx.value_and_grad(microbatch_loss_fn, argnums=diff_state, has_aux=True)
        # Seed the carry with microbatch zero, then use a real XLA loop for the remainder.
        # A Python loop would inline one complete VLM forward/backward graph per microbatch;
        # on 80GB H100s that made B2x6 *larger* than B4x3. `fori_loop` compiles one reusable
        # body and keeps only one microbatch's activations live at a time.
        first_observation = jax.tree.map(lambda x: x[0], observation)
        (loss, loss_info), grads = value_and_grad(
            model,
            jax.random.fold_in(train_rng, 0),
            first_observation,
            actions[0],
        )

        def accumulate_microbatch(microbatch_index, carry):
            accumulated_loss, accumulated_info, accumulated_grads = carry
            micro_observation = jax.tree.map(lambda x: x[microbatch_index], observation)
            (micro_loss, micro_info), micro_grads = value_and_grad(
                model,
                jax.random.fold_in(train_rng, microbatch_index),
                micro_observation,
                actions[microbatch_index],
            )
            accumulated_info = jax.tree.map(jnp.add, accumulated_info, micro_info)
            for max_key in (
                "diagnostic/write_inner_grad_max",
                "diagnostic/v35_commit_residual_ratio_max",
                "diagnostic/v35_commit_relative_residual_max",
            ):
                if max_key in accumulated_info:
                    # Max-reduced leaves must not be summed by the generic tree reduction.
                    accumulated_info[max_key] = jnp.maximum(carry[1][max_key], micro_info[max_key])
            return (
                accumulated_loss + micro_loss,
                accumulated_info,
                jax.tree.map(jnp.add, accumulated_grads, micro_grads),
            )

        loss, loss_info, grads = jax.lax.fori_loop(
            1,
            accumulation_steps,
            accumulate_microbatch,
            (loss, loss_info, grads),
        )
    if getattr(config.model, "memory_v35_enabled", False):
        # This private vector is consumed by the checkified official training wrapper and is
        # removed before logging.  Computing it from the full effective batch makes the same
        # contract exact with or without gradient accumulation.
        loss_info["_v35_runtime_guard"] = _v35_runtime_guard_vector(config, observation, loss_info)
    diagnostic_only_probe = (
        getattr(config.model, "predict_with_memory", False) and getattr(config.model, "memory_probe_weight", 0) == 0
    )
    if diagnostic_only_probe:
        # Keep the probe leaves in the optimizer tree so probe-trained checkpoints retain an
        # identical TrainState structure, but guarantee that neither diagnostics nor stale
        # restored moments can update the compatibility head.
        probe_filter = nnx_utils.PathRegex(r".*probe_head.*")
        grads = nnx_utils.state_map(
            grads, probe_filter, lambda variable: variable.replace(jnp.zeros_like(variable.value))
        )

    # v3.5 calibration freezes the effective tanh injection gate at 0.5. Enforce this here in
    # addition to the recipe's freeze_filter so a custom launch cannot mutate it through Adam
    # weight decay or restored optimizer moments.
    frozen_injection_gate = getattr(config.model, "memory_freeze_injection_gate", False)
    if frozen_injection_gate:
        injection_gate_filter = nnx_utils.PathRegex(r".*memory_inject_w.*")
        grads = nnx_utils.state_map(
            grads,
            injection_gate_filter,
            lambda variable: variable.replace(jnp.zeros_like(variable.value)),
        )

    # v3.4 Section 6: the probe ladder gets an ISOLATED optimizer. The ladder-head grads are
    # extracted, then zeroed out of the main path BEFORE tx.update -- so they contribute
    # nothing to the global clip norm or the Adam state -- and applied afterwards as a plain
    # constant-LR SGD step. With probe features stop-gradient'ed in the model, one main-model
    # update is bit-identical with the probes enabled or disabled (unit-tested).
    ladder_isolated = getattr(config.model, "memory_ladder_probes", False)
    if ladder_isolated:
        ladder_grads = grads.filter(LADDER_PROBE_FILTER)
        grads = nnx_utils.state_map(
            grads, LADDER_PROBE_FILTER, lambda variable: variable.replace(jnp.zeros_like(variable.value))
        )

    # v34_run1 postmortem: pre-clip the memory-path gradient group to its own norm budget
    # BEFORE the shared global clip. The recurrent memory backward can spike orders of
    # magnitude above the rest of the model; without this, one bad chain scales EVERY
    # parameter's update toward zero through the global clip (the observed explosion/collapse
    # limit cycle). The group clip preserves the memory gradient's direction and leaves all
    # non-memory gradients untouched.
    if config.memory_grad_clip is not None:
        memory_norm = optax.global_norm(grads.filter(MEMORY_PATH_FILTER))
        memory_scale = jnp.minimum(1.0, config.memory_grad_clip / (memory_norm + 1e-12))
        grads = nnx_utils.state_map(
            grads,
            MEMORY_PATH_FILTER,
            # flax None-bias slots survive into the grads State as None-valued leaves.
            lambda variable: variable if variable.value is None else variable.replace(variable.value * memory_scale),
        )
        loss_info["memory_grad_norm"] = memory_norm

    # Gate-D stability is defined on every optimizer update, after branch-local feature and
    # memory-group caps but before AdamW's shared global clip. Sampling this norm only on log
    # steps cannot establish the preregistered <=1% severe-update rate.
    v35_pre_shared_grad_norm = None
    if getattr(config.model, "memory_v35_enabled", False):
        shared_clip_threshold = getattr(config.optimizer, "clip_gradient_norm", None)
        if shared_clip_threshold is None or shared_clip_threshold <= 0:
            raise ValueError("v3.5 requires an optimizer with a positive shared clip_gradient_norm.")
        v35_pre_shared_grad_norm = optax.global_norm(grads)
        severe = (v35_pre_shared_grad_norm > 10.0 * shared_clip_threshold).astype(jnp.float32)
        loss_info.update(
            {
                "diagnostic/v35_pre_shared_clip_grad_norm_sum": v35_pre_shared_grad_norm,
                "diagnostic/v35_pre_shared_clip_grad_norm_max": v35_pre_shared_grad_norm,
                "diagnostic/v35_pre_shared_clip_severe_count": severe,
                "diagnostic/v35_pre_shared_clip_update_count": jnp.asarray(1.0, jnp.float32),
            }
        )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    if diagnostic_only_probe:
        updates = nnx_utils.state_map(
            updates, probe_filter, lambda variable: variable.replace(jnp.zeros_like(variable.value))
        )
    if frozen_injection_gate:
        updates = nnx_utils.state_map(
            updates,
            injection_gate_filter,
            lambda variable: variable.replace(jnp.zeros_like(variable.value)),
        )
    if ladder_isolated:
        # Erase the weight-decay-only AdamW update on the ladder leaves, then apply the SGD.
        updates = nnx_utils.state_map(
            updates, LADDER_PROBE_FILTER, lambda variable: variable.replace(jnp.zeros_like(variable.value))
        )
    new_params = optax.apply_updates(params, updates)
    if ladder_isolated:
        ladder_new = jax.tree.map(
            lambda p, g: p - config.probe_lr * g, params.filter(LADDER_PROBE_FILTER), ladder_grads
        )
        new_params = nnx.State.merge(new_params.filter(nnx.Not(LADDER_PROBE_FILTER)), ladder_new)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        ema_params = jax.tree.map(
            lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
        )
        if diagnostic_only_probe:
            # Probe-trained checkpoints can contain different raw and EMA probe values. Preserve
            # both exactly: allowing the saved/inference EMA head to drift toward the frozen raw
            # head would still mutate the diagnostic across resumed no-probe training.
            ema_params = nnx.State.merge(
                ema_params.filter(nnx.Not(probe_filter)),
                state.ema_params.filter(probe_filter),
            )
        if frozen_injection_gate:
            ema_params = nnx.State.merge(
                ema_params.filter(nnx.Not(injection_gate_filter)),
                state.ema_params.filter(injection_gate_filter),
            )
        new_state = dataclasses.replace(
            new_state,
            ema_params=ema_params,
        )

    # These full-tree diagnostics do not affect optimization and are only consumed every
    # log_interval steps. Avoid paying their bandwidth/collective cost on the other updates.
    expensive_norm_active = jnp.equal(jnp.mod(state.step, config.log_interval), 0)

    def compute_expensive_norms(_):
        kernel_params = nnx.state(
            model,
            nnx.All(
                nnx.Param,
                nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
                lambda _, x: x.value.ndim > 1,
            ),
        )
        grad_norm = v35_pre_shared_grad_norm if v35_pre_shared_grad_norm is not None else optax.global_norm(grads)
        return grad_norm, optax.global_norm(kernel_params)

    grad_norm, param_norm = jax.lax.cond(
        expensive_norm_active,
        compute_expensive_norms,
        lambda _: (jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32)),
        operand=None,
    )
    info = {
        "loss": loss,
        **loss_info,
        "grad_norm": grad_norm,
        "param_norm": param_norm,
        "_expensive_norm_count": expensive_norm_active.astype(jnp.float32),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    _configure_v35_runtime_environment(config)
    logging.info(f"Running on: {platform.node()}")
    _log_training_identity(config)

    _validate_v35_training_ready(config)
    v35_enabled = getattr(config.model, "memory_v35_enabled", False)
    v35_pilot_authorization: _v35_authorization.AuthorizationRecord | None = None
    if v35_enabled:
        v35_pilot_authorization = _v35_authorization.load_and_validate_pilot_authorization(config)
        if not config.resume:
            raise ValueError(
                "v3.5 optimizer training must --resume the finalized completed-update-0 checkpoint; "
                "create it with scripts/v35_step0_bootstrap.py after train-54 calibration."
            )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    microbatch_size = config.batch_size // config.gradient_accumulation_steps
    if microbatch_size % jax.device_count() != 0:
        raise ValueError(
            f"Microbatch size {microbatch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # Cluster jobs can inherit an AFS home directory even though the working tree and caches
    # live on Iris.  Allow launchers to choose the cache location without mutating HOME.
    jax_cache_dir = os.environ.get("OPENPI_JAX_CACHE_DIR")
    if jax_cache_dir is None:
        if v35_enabled:
            jax_cache_dir = str(project_paths.project_path(project_paths.JAX_CACHE_DIR))
        else:
            jax_cache_dir = str(epath.Path("~/.cache/jax").expanduser())
    jax.config.update("jax_compilation_cache_dir", jax_cache_dir)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(sharding.DATA_AXIS)
        if config.gradient_accumulation_steps == 1
        else jax.sharding.PartitionSpec(None, sharding.DATA_AXIS),
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
        allow_step_zero_resume=v35_enabled,
    )
    v35_identity_path = pathlib.Path(config.checkpoint_dir) / "initialization_manifest.json"
    v35_provenance_assets: dict[str, bytes] | None = None
    v35_continuation_authorization: _v35_authorization.AuthorizationRecord | None = None
    v35_source_authorization: _v35_authorization.AuthorizationRecord | None = None
    v35_embedded_continuation_bytes: bytes | None = None
    if v35_enabled:
        latest_step = int(checkpoint_manager.latest_step()) if resuming else None
        _validate_v35_checkpoint_protocol(config, resuming=resuming, latest_step=latest_step)
        if resuming:
            v35_provenance_assets = _validate_v35_resume_checkpoint_assets(
                config,
                checkpoint_step=latest_step,
                identity_path=v35_identity_path,
            )
            if config.num_train_steps > 1_000:
                if v35_pilot_authorization is None:
                    raise AssertionError("v3.5 continuation is missing its validated pilot authorization")
                v35_continuation_authorization = _v35_authorization.load_and_validate_continuation_authorization(
                    config,
                    pilot_authorization=v35_pilot_authorization,
                    latest_checkpoint_step=latest_step,
                )
                embedded_continuation_path = (
                    pathlib.Path(config.checkpoint_dir)
                    / str(latest_step)
                    / "assets"
                    / _V35_CONTINUATION_AUTHORIZATION_FILENAME
                )
                if embedded_continuation_path.is_file():
                    v35_embedded_continuation_bytes = embedded_continuation_path.read_bytes()
                v35_provenance_assets[_V35_CONTINUATION_AUTHORIZATION_FILENAME] = (
                    v35_continuation_authorization.path.read_bytes()
                )
            v35_source_authorization = (
                v35_pilot_authorization
                if latest_step == 0
                else v35_continuation_authorization
            )
            if v35_source_authorization is None:
                raise ValueError(
                    "v3.5 resume source is not linked by the configured authorization; intermediate "
                    "checkpoints require separately sealed external rung/hash evidence"
                )
    init_wandb(
        config,
        resuming=resuming,
        enabled=config.wandb_enabled,
        allow_new_run_from_bootstrap_zero=v35_enabled and resuming and latest_step == 0,
    )

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    n_params = sum(x.size for x in jax.tree.leaves(train_state.params))
    logging.info(f"Initialized train state: {n_params / 1e9:.2f}B params")

    v35_cumulative_telemetry = _new_v35_cumulative_telemetry() if v35_enabled else None
    v35_restored_parameter_tree_sha256: str | None = None
    if resuming:
        if v35_enabled:
            if v35_source_authorization is None:
                raise AssertionError("v3.5 resume is missing its source authorization")
            train_state, v35_cumulative_telemetry, v35_restored_parameter_tree_sha256 = (
                _restore_and_validate_v35_authorized_source_checkpoint(
                    config,
                    checkpoint_manager=checkpoint_manager,
                    checkpoint_step=latest_step,
                    state_shape=train_state,
                    data_loader=data_loader,
                    source_authorization=v35_source_authorization,
                )
            )
        else:
            train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
    _validate_v35_initialized_gate(config, train_state.params)

    initialization_manifest_path = None
    if v35_enabled:
        initialization_manifest_path = (
            v35_identity_path if resuming else _write_v35_initialization_identity(config, train_state.params)
        )
        if initialization_manifest_path is None or not initialization_manifest_path.is_file():
            raise FileNotFoundError("v3.5 initialization identity is missing.")
        if not resuming:
            v35_provenance_assets = _snapshot_v35_checkpoint_provenance(
                config,
                initialization_manifest_path,
            )
        initialization_identity = _validate_v35_root_identity(config, initialization_manifest_path)
        if v35_pilot_authorization is None:
            raise AssertionError("v3.5 run is missing its validated pilot authorization")
        actual_step0_parameter_tree_sha256 = (
            v35_restored_parameter_tree_sha256
            if resuming and latest_step == 0
            else _weight_loaders.parameter_tree_sha256(train_state.params.to_pure_dict())
            if not resuming
            else None
        )
        _v35_authorization.validate_pilot_run_binding(
            config,
            v35_pilot_authorization,
            initialization_identity=initialization_identity,
            actual_parameter_tree_sha256=actual_step0_parameter_tree_sha256,
        )
        if v35_continuation_authorization is not None:
            if v35_restored_parameter_tree_sha256 is None:
                raise AssertionError("v3.5 continuation is missing its restored parameter identity")
            _v35_authorization.validate_continuation_checkpoint_binding(
                v35_continuation_authorization,
                latest_checkpoint_step=latest_step,
                actual_parameter_tree_sha256=v35_restored_parameter_tree_sha256,
                embedded_authorization_bytes=v35_embedded_continuation_bytes,
            )

    # Revision-5 checkpoint labels are completed-update counts. A fresh step-0 snapshot is the
    # immutable source/task-health reference; rung 250 therefore means exactly 250 optimizer
    # updates, not the legacy off-by-one convention (loop index 250 after 251 updates).
    start_step = int(train_state.step)
    if config.checkpoint_by_completed_updates and not resuming and start_step == 0:
        v35_runtime_kwargs = {}
        if v35_enabled:
            if v35_cumulative_telemetry is None:
                raise AssertionError("v3.5 step-0 checkpoint requires cumulative telemetry.")
            v35_runtime_kwargs["v35_cumulative_telemetry"] = v35_cumulative_telemetry
        _checkpoints.save_state(
            checkpoint_manager,
            train_state,
            data_loader,
            0,
            provenance_assets=v35_provenance_assets,
            **v35_runtime_kwargs,
        )

    # v3.5 restores sampler and transform RNG before this iterator is constructed. With
    # num_workers=0, `next` consumes exactly one visible batch and no future batch is prefetched.
    data_iter = iter(data_loader)
    batch = next(data_iter)
    batch_mb = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(batch)) / 1e6
    logging.info(
        f"Initialized data loader: {len(jax.tree.leaves(batch))} arrays, {batch_mb:.0f} MB/effective batch; "
        f"global_batch={config.batch_size}, microbatch={microbatch_size}, "
        f"gradient_accumulation_steps={config.gradient_accumulation_steps}"
    )

    # Log images only for a fresh run. On resume W&B rejects a new step-0 record, and staging
    # these device arrays needlessly fragments the very tight 80GB H100 allocator before the
    # first accumulated update.
    if not resuming:
        log_images = batch[0].images
        if config.gradient_accumulation_steps > 1:
            log_images = jax.tree.map(lambda x: x.reshape(config.batch_size, *x.shape[2:]), log_images)
        images_to_log = [
            wandb.Image(
                np.concatenate(
                    [np.array(img[i, 0] if img.ndim == 5 else img[i]) for img in log_images.values()], axis=1
                )
            )
            for i in range(min(5, len(next(iter(log_images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)

    v35_runtime_guard = getattr(config.model, "memory_v35_enabled", False)
    if v35_runtime_guard:
        # `checkify` carries device-side assertion state out of the compiled update.  The host
        # throws before accepting/donating the candidate state, so an invalid update can never
        # be logged or checkpointed.  Legacy recipes retain their original compiled signature.
        checked_step = checkify.checkify(
            functools.partial(_checked_v35_train_step, config),
            errors=checkify.user_checks,
        )
        ptrain_step = jax.jit(
            checked_step,
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(None, (train_state_sharding, replicated_sharding)),
            donate_argnums=(1,),
        )
    else:
        ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,),
        )

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            if v35_runtime_guard:
                runtime_error, candidate = ptrain_step(train_rng, train_state, batch)
                runtime_error.throw()
                train_state, info = candidate
            else:
                train_state, info = ptrain_step(train_rng, train_state, batch)
        metric_step, should_save, checkpoint_step = _step_labels_and_save_decision(
            config, loop_step=step, start_step=start_step
        )
        if v35_runtime_guard:
            if v35_cumulative_telemetry is None:
                raise AssertionError("v3.5 accepted update is missing its cumulative telemetry ledger.")
            _accumulate_v35_cumulative_telemetry(
                v35_cumulative_telemetry,
                info,
                completed_updates=metric_step,
            )
        infos.append(info)
        if metric_step % config.log_interval == 0:
            reduced = _reduce_infos(infos)
            reduced_info = {}
            for k, v in reduced.items():
                if np.ndim(v) == 1:  # per-step quiz accuracy -> one scalar per step index
                    reduced_info.update({f"{k}_p{i}": float(x) for i, x in enumerate(v)})
                else:
                    reduced_info[f"{k}"] = float(v)
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items() if not _is_per_position_metric(k))
            label = "Completed update" if config.checkpoint_by_completed_updates else "Step"
            pbar.write(f"{label} {metric_step}: {info_str}")
            wandb.log(reduced_info, step=metric_step)
            infos = []

        # Save v3.5 at the accepted-update boundary before drawing another batch. Since the
        # loader has no workers/prefetch, its snapshotted RNG points exactly to the next batch.
        if should_save and v35_runtime_guard:
            _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                checkpoint_step,
                provenance_assets=v35_provenance_assets,
                v35_cumulative_telemetry=v35_cumulative_telemetry,
            )
        if not v35_runtime_guard or metric_step < config.num_train_steps:
            batch = next(data_iter)
        if should_save and not v35_runtime_guard:
            # Preserve legacy ordering, which historically fetched the next batch before save.
            _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                checkpoint_step,
                provenance_assets=v35_provenance_assets,
            )

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
