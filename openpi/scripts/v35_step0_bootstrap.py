"""Create and finalize the immutable v3.5 completed-update-0 checkpoint.

This CLI closes the intentional prelaunch dependency cycle without weakening ``train.py``:

1. ``initialize`` loads the official Pi0.5 base through the audited graft, freshly
   initializes every v3.5-only leaf, initializes Adam at step zero, and saves an unfinalized
   raw state.  It does not construct a dataset/loader and cannot fetch a batch.
2. Calibration replay is collected and the train-54 calibration artifact is sealed.
3. ``finalize`` authenticates that calibration against the raw tree and frozen dataset,
   creates the production loader solely to snapshot its initial sampler/RNG state, and writes
   checkpoint ``0`` plus the format-2 initialization identity.  It never constructs an
   iterator and therefore consumes no batch.
4. Gate A, Gate B, step-0 Gate C/task health, and the pilot authorization are produced from
   that finalized checkpoint.  Ordinary optimizer training can then start only by resuming it.

Every CLI path is relative to ``memory_project``.  Both phases refuse to overwrite output.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import sys
from types import SimpleNamespace
from typing import Any

from openpi.shared import project_paths

# Install project-local HF/OpenPI/JAX/temp paths before importing libraries that can bind cache
# locations at import time.  This makes the CLI standalone on a freshly synced cluster; the
# official base may still be downloaded from its gs:// URI into this project-local cache.
project_paths.configure_v35_runtime_environment()
project_paths.validate_executing_openpi_checkout()

from flax import nnx  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import train as train_script  # noqa: E402
import v35_calibration_replay as calibration_replay  # noqa: E402

import openpi.training.checkpoints as checkpoints  # noqa: E402
import openpi.training.config as training_config  # noqa: E402
import openpi.training.data_loader as data_loader  # noqa: E402
import openpi.training.sharding as sharding  # noqa: E402
import openpi.training.v35_authorization as authorization  # noqa: E402
import openpi.training.weight_loaders as weight_loaders  # noqa: E402

CONFIG_NAME = "pi05_yam_mem_v35"
OFFICIAL_BASE_URI = authorization.OFFICIAL_BASE_URI
PROVISIONAL_SCHEMA_VERSION = "openpi.v35.step0-bootstrap-provisional.v1"
PROVISIONAL_FILENAME = "step0_bootstrap_provisional.json"
RAW_CHECKPOINT_DIRNAME = "bootstrap_raw_state"
FINAL_CHECKPOINT_STEP = 0
_SHA256_CHARS = frozenset("0123456789abcdef")


class Step0BootstrapError(ValueError):
    """Raised when write-once step-0 construction cannot be authenticated."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _canonical_payload_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)[:-1]).hexdigest()


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in _SHA256_CHARS for char in value):
        raise Step0BootstrapError(f"{name} must be a lower-case 64-character SHA256")
    return value


def _relative_path(value: str | PurePosixPath, *, name: str, must_exist: bool = False) -> Path:
    relative = PurePosixPath(value)
    if not relative.parts or relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise Step0BootstrapError(f"{name} must be a normalized memory_project-relative POSIX path")
    path = project_paths.project_path(relative)
    if must_exist and not path.exists():
        raise Step0BootstrapError(f"{name} does not exist: {relative.as_posix()}")
    return path


def _project_relative(path: Path) -> str:
    try:
        return project_paths.project_relative_path(path).as_posix()
    except project_paths.ProjectRootError as exc:
        raise Step0BootstrapError(str(exc)) from exc


def _write_once(path: Path, contents: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(contents)
    except FileExistsError as exc:
        raise Step0BootstrapError(f"refusing to overwrite existing step-0 artifact: {path}") from exc


def _load_canonical_self_hashed(path: Path, *, hash_key: str, name: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise Step0BootstrapError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise Step0BootstrapError(f"{name} must be canonical JSON with one trailing newline")
    recorded = _require_sha256(f"{name}.{hash_key}", value.get(hash_key))
    unsigned = {key: item for key, item in value.items() if key != hash_key}
    if recorded != _canonical_payload_sha256(unsigned):
        raise Step0BootstrapError(f"{name} self-hash is invalid")
    return value, raw


def _checkpoint_root(config: training_config.TrainConfig) -> Path:
    root = Path(config.checkpoint_dir).resolve()
    _project_relative(root)
    return root


def _base_config(*, experiment_name: str, fsdp_devices: int | None) -> training_config.TrainConfig:
    if not experiment_name.strip() or PurePosixPath(experiment_name).name != experiment_name:
        raise Step0BootstrapError("experiment name must be one nonempty path component")
    config = training_config.get_config(CONFIG_NAME)
    if not getattr(config.model, "memory_v35_enabled", False):
        raise Step0BootstrapError("registered v3.5 config no longer enables memory_v35")
    loader = config.weight_loader
    if not isinstance(loader, weight_loaders.AuditedPartialCheckpointWeightLoader):
        raise Step0BootstrapError("registered v3.5 config no longer uses the audited partial loader")
    if loader.params_path != OFFICIAL_BASE_URI:
        raise Step0BootstrapError("registered v3.5 config no longer names the official Pi0.5 base")
    config = dataclasses.replace(
        config,
        exp_name=experiment_name,
        fsdp_devices=config.fsdp_devices if fsdp_devices is None else fsdp_devices,
        overwrite=False,
        resume=False,
    )
    root = _checkpoint_root(config)
    loader = dataclasses.replace(loader, manifest_output_path=str(root / "initialization_graft_manifest.json"))
    return dataclasses.replace(config, weight_loader=loader)


def _calibrated_config(
    *,
    experiment_name: str,
    calibration_relative: str | PurePosixPath,
    fsdp_devices: int | None,
) -> training_config.TrainConfig:
    config = _base_config(experiment_name=experiment_name, fsdp_devices=fsdp_devices)
    calibration_path = _relative_path(calibration_relative, name="calibration artifact", must_exist=True)
    try:
        artifact = json.loads(calibration_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise Step0BootstrapError(f"cannot read calibration artifact: {exc}") from exc
    payload = artifact.get("payload") if isinstance(artifact, dict) else None
    parameters = payload.get("parameters") if isinstance(payload, dict) else None
    if not isinstance(parameters, dict):
        raise Step0BootstrapError("calibration artifact is missing payload.parameters")
    calibration_id = artifact.get("calibration_id")
    if not isinstance(calibration_id, str):
        raise Step0BootstrapError("calibration artifact is missing calibration_id")
    try:
        c = float(parameters["memory_injection_c"])
        tau = float(parameters["memory_injection_tau"])
    except (KeyError, TypeError, ValueError) as exc:
        raise Step0BootstrapError("calibration artifact has invalid c/tau") from exc
    if not np.isfinite(c) or not np.isfinite(tau):
        raise Step0BootstrapError("calibration c/tau must be finite")
    model = dataclasses.replace(
        config.model,
        memory_v35_calibrated=True,
        memory_v35_calibration_id=calibration_id,
        memory_v35_calibration_path=str(calibration_path),
        memory_injection_c=c,
        memory_injection_tau=tau,
    )
    return dataclasses.replace(config, model=model)


def _dataset_identity(config: training_config.TrainConfig) -> dict[str, Any]:
    manifest_path = Path(config.data.base_config.memory_episode_manifest_path)
    norm_stats_path, norm_provenance_path = train_script._v35_norm_artifact_paths(config)  # noqa: SLF001
    try:
        norm_provenance = json.loads(norm_provenance_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise Step0BootstrapError(f"cannot read train-54 norm provenance: {exc}") from exc
    storage = norm_provenance.get("train_storage") if isinstance(norm_provenance, dict) else None
    if not isinstance(storage, dict):
        raise Step0BootstrapError("train-54 norm provenance is missing train_storage")
    return {
        "episode_manifest_path_relative": _project_relative(manifest_path),
        "episode_manifest_sha256": _sha256_bytes(manifest_path.read_bytes()),
        "norm_stats_path_relative": _project_relative(norm_stats_path),
        "norm_stats_sha256": _sha256_bytes(norm_stats_path.read_bytes()),
        "norm_stats_provenance_path_relative": _project_relative(norm_provenance_path),
        "norm_stats_provenance_sha256": _sha256_bytes(norm_provenance_path.read_bytes()),
        "train_storage_sha256": _require_sha256("train_storage.sha256", storage.get("sha256")),
    }


def _make_provisional_manifest(
    *,
    config: training_config.TrainConfig,
    step0: calibration_replay.Step0Identity,
    graft_path: Path,
) -> dict[str, Any]:
    root = _checkpoint_root(config)
    payload: dict[str, Any] = {
        "schema_version": PROVISIONAL_SCHEMA_VERSION,
        "status": "initialized_unfinalized",
        "config_name": config.name,
        "experiment_name": config.exp_name,
        "initialization_seed": config.seed,
        "completed_optimizer_updates": 0,
        "data_loader_constructed": False,
        "data_batches_consumed": 0,
        "official_source_uri": OFFICIAL_BASE_URI,
        "official_source_tree_sha256": step0.official_base_source_sha256,
        "target_schema_sha256": step0.target_schema_sha256,
        "actual_step0_parameter_tree_sha256": step0.actual_parameter_tree_sha256,
        "graft_manifest": {
            "path_relative": _project_relative(graft_path),
            "file_sha256": _sha256_bytes(graft_path.read_bytes()),
            "manifest_sha256": step0.graft_manifest_sha256,
        },
        "raw_checkpoint_path_relative": _project_relative(root / RAW_CHECKPOINT_DIRNAME / "0"),
        "registered_uncalibrated_semantic_config_sha256": authorization.semantic_training_config_sha256(config),
        "dataset_identity": _dataset_identity(config),
    }
    payload["provisional_identity_sha256"] = _canonical_payload_sha256(payload)
    return payload


class _NoDatasetBootstrapLoader:
    """The raw-state save callback needs only this normalization-free descriptor."""

    @staticmethod
    def data_config() -> SimpleNamespace:
        return SimpleNamespace(norm_stats=None, asset_id=None)


def _validate_calibrated_static_model(config: training_config.TrainConfig, state: Any) -> None:
    """Prove restored params are paired with the sealed calibration's static GraphDef."""

    model = nnx.merge(state.model_def, state.params)
    try:
        actual_c = float(model.memory_injection_c)
        actual_tau = float(model.memory_injection_tau)
    finally:
        del model
    if actual_c != float(config.model.memory_injection_c) or actual_tau != float(config.model.memory_injection_tau):
        raise Step0BootstrapError("restored step-0 GraphDef does not carry the sealed calibration c/tau")


def initialize(*, experiment_name: str, fsdp_devices: int | None = None) -> Path:
    """Write an unfinalized exact step-0 state without constructing a data loader."""

    project_paths.configure_v35_runtime_environment()
    config = _base_config(experiment_name=experiment_name, fsdp_devices=fsdp_devices)
    if getattr(config.model, "memory_v35_calibrated", False):
        raise Step0BootstrapError("initialize requires the registered uncalibrated v3.5 config")
    root = _checkpoint_root(config)
    if root.exists():
        raise Step0BootstrapError(f"refusing to reuse existing experiment directory: {_project_relative(root)}")
    root.mkdir(parents=True)
    graft_path = root / "initialization_graft_manifest.json"
    state, step0 = calibration_replay._initialize_actual_step0_state(  # noqa: SLF001
        config,
        graft_manifest_path=graft_path,
    )
    if int(state.step) != 0:
        raise Step0BootstrapError("fresh initialization unexpectedly has a nonzero optimizer step")
    provisional = _make_provisional_manifest(config=config, step0=step0, graft_path=graft_path)
    provisional_path = root / PROVISIONAL_FILENAME
    provisional_bytes = _canonical_json_bytes(provisional)
    _write_once(provisional_path, provisional_bytes)

    raw_dir = root / RAW_CHECKPOINT_DIRNAME
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        raw_dir,
        keep_period=None,
        overwrite=False,
        resume=False,
        allow_step_zero_resume=True,
    )
    if resuming:
        raise AssertionError("new raw step-0 directory unexpectedly entered resume mode")
    checkpoints.save_state(
        manager,
        state,
        _NoDatasetBootstrapLoader(),
        0,
        provenance_assets={PROVISIONAL_FILENAME: provisional_bytes},
    )
    manager.wait_until_finished()
    if tuple(manager.all_steps()) != (0,):
        raise Step0BootstrapError("raw bootstrap checkpoint did not commit exactly step 0")
    return provisional_path


def _authenticate_provisional(config: training_config.TrainConfig) -> tuple[dict[str, Any], bytes]:
    root = _checkpoint_root(config)
    path = root / PROVISIONAL_FILENAME
    provisional, raw = _load_canonical_self_hashed(
        path,
        hash_key="provisional_identity_sha256",
        name="step-0 provisional identity",
    )
    required = {
        "schema_version",
        "status",
        "config_name",
        "experiment_name",
        "initialization_seed",
        "completed_optimizer_updates",
        "data_loader_constructed",
        "data_batches_consumed",
        "official_source_uri",
        "official_source_tree_sha256",
        "target_schema_sha256",
        "actual_step0_parameter_tree_sha256",
        "graft_manifest",
        "raw_checkpoint_path_relative",
        "registered_uncalibrated_semantic_config_sha256",
        "dataset_identity",
        "provisional_identity_sha256",
    }
    if set(provisional) != required:
        raise Step0BootstrapError("step-0 provisional identity has an unexpected schema")
    if (
        provisional["schema_version"] != PROVISIONAL_SCHEMA_VERSION
        or provisional["status"] != "initialized_unfinalized"
        or provisional["config_name"] != config.name
        or provisional["experiment_name"] != config.exp_name
        or provisional["initialization_seed"] != config.seed
        or provisional["completed_optimizer_updates"] != 0
        or provisional["data_loader_constructed"] is not False
        or provisional["data_batches_consumed"] != 0
        or provisional["official_source_uri"] != OFFICIAL_BASE_URI
    ):
        raise Step0BootstrapError("step-0 provisional identity does not describe this zero-update run")
    for name in (
        "official_source_tree_sha256",
        "target_schema_sha256",
        "actual_step0_parameter_tree_sha256",
    ):
        _require_sha256(name, provisional[name])

    uncalibrated = _base_config(experiment_name=config.exp_name, fsdp_devices=config.fsdp_devices)
    if provisional["registered_uncalibrated_semantic_config_sha256"] != authorization.semantic_training_config_sha256(
        uncalibrated
    ):
        raise Step0BootstrapError("registered uncalibrated config changed after initialize")
    expected_dataset = _dataset_identity(config)
    if provisional["dataset_identity"] != expected_dataset:
        raise Step0BootstrapError("dataset/norm identity changed after initialize")
    graft = provisional["graft_manifest"]
    if not isinstance(graft, dict) or set(graft) != {"path_relative", "file_sha256", "manifest_sha256"}:
        raise Step0BootstrapError("provisional graft descriptor is invalid")
    graft_path = _relative_path(graft["path_relative"], name="graft manifest", must_exist=True)
    if graft_path != root / "initialization_graft_manifest.json":
        raise Step0BootstrapError("provisional graft path is not the experiment graft")
    verified_graft = calibration_replay._verify_graft_manifest(graft_path)  # noqa: SLF001
    if (
        graft["file_sha256"] != _sha256_bytes(graft_path.read_bytes())
        or graft["manifest_sha256"] != verified_graft["manifest_sha256"]
        or provisional["official_source_tree_sha256"] != verified_graft["tree_hashes"]["source_sha256"]
        or provisional["target_schema_sha256"] != verified_graft["tree_hashes"]["target_schema_sha256"]
    ):
        raise Step0BootstrapError("audited graft changed after initialize")
    expected_raw = root / RAW_CHECKPOINT_DIRNAME / "0"
    if _relative_path(provisional["raw_checkpoint_path_relative"], name="raw checkpoint", must_exist=True) != expected_raw:
        raise Step0BootstrapError("provisional raw checkpoint path is invalid")
    embedded = expected_raw / "assets" / PROVISIONAL_FILENAME
    if not embedded.is_file() or embedded.read_bytes() != raw:
        raise Step0BootstrapError("raw checkpoint does not embed the byte-exact provisional identity")
    return provisional, raw


def finalize(
    *,
    experiment_name: str,
    calibration_relative: str | PurePosixPath,
    fsdp_devices: int | None = None,
) -> Path:
    """Seal the final identity and exact-resume step-0 checkpoint without drawing a batch."""

    project_paths.configure_v35_runtime_environment()
    config = _calibrated_config(
        experiment_name=experiment_name,
        calibration_relative=calibration_relative,
        fsdp_devices=fsdp_devices,
    )
    root = _checkpoint_root(config)
    final_identity_path = root / "initialization_manifest.json"
    final_checkpoint_path = root / "0"
    if final_identity_path.exists() or final_checkpoint_path.exists():
        raise Step0BootstrapError("refusing to overwrite an existing finalized step-0 run")
    provisional, _ = _authenticate_provisional(config)

    # This authenticates the calibration envelope, c/tau/alpha, gate, train-54 membership,
    # manifest, norm provenance, and selected storage seal.  No launch authorization is read.
    train_script._validate_v35_training_ready(config)  # noqa: SLF001

    mesh = sharding.make_mesh(config.fsdp_devices)
    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)
    state_shape, _ = train_script.init_train_state(config, init_rng, mesh, resume=True)
    raw_dir = root / RAW_CHECKPOINT_DIRNAME
    raw_manager, resuming = checkpoints.initialize_checkpoint_dir(
        raw_dir,
        keep_period=None,
        overwrite=False,
        resume=True,
        allow_step_zero_resume=True,
    )
    if not resuming or raw_manager.latest_step() != 0:
        raise Step0BootstrapError("raw bootstrap state is not exactly completed update 0")
    state = checkpoints.restore_state(raw_manager, state_shape, None, step=0)
    jax.block_until_ready(state)
    if int(state.step) != 0:
        raise Step0BootstrapError("restored bootstrap state has a nonzero optimizer step")
    actual_tree_sha256 = weight_loaders.parameter_tree_sha256(state.params.to_pure_dict())
    if actual_tree_sha256 != provisional["actual_step0_parameter_tree_sha256"]:
        raise Step0BootstrapError("restored raw parameter tree differs from provisional initialization")
    # Orbax restores dynamic params/optimizer leaves into the calibrated target structure.  Pin
    # the static GraphDef/optimizer controls to that target explicitly, then prove the graph
    # carries the sealed c/tau rather than initialize's placeholders.
    state = dataclasses.replace(
        state,
        model_def=state_shape.model_def,
        tx=state_shape.tx,
        ema_decay=state_shape.ema_decay,
    )
    _validate_calibrated_static_model(config, state)
    train_script._validate_v35_initialized_gate(config, state.params)  # noqa: SLF001

    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    initial_loader_state = loader.state_dict()
    if initial_loader_state.get("batches_yielded") != 0:
        raise Step0BootstrapError("finalization loader consumed a batch before its initial snapshot")

    # Complete every check that does not depend on the final identity before writing it.  A
    # failed loader or nonempty checkpoint-root precheck therefore cannot strand an identity
    # that a later finalize invocation is forbidden to replace.
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        root,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
        allow_step_zero_resume=True,
    )
    if resuming or tuple(manager.all_steps()):
        raise Step0BootstrapError("experiment root unexpectedly contains a finalized checkpoint")

    # The format-2 writer additionally proves the calibration source tree and raw gate hash.
    identity_path = train_script._write_v35_initialization_identity(config, state.params)  # noqa: SLF001
    if identity_path != final_identity_path:
        raise Step0BootstrapError("format-2 identity writer selected an unexpected path")
    train_script._validate_v35_root_identity(config, identity_path)  # noqa: SLF001

    # Pilot authorization is necessarily produced after Gate A/B/C consume this checkpoint.
    # It is excluded from checkpoint 0 only; train.py authenticates it externally on resume
    # and embeds it in every subsequently saved rung.
    preauthorization_config = dataclasses.replace(config, v35_pilot_authorization_path=None)
    provenance = train_script._snapshot_v35_checkpoint_provenance(  # noqa: SLF001
        preauthorization_config,
        identity_path,
    )
    telemetry = train_script._new_v35_cumulative_telemetry()  # noqa: SLF001
    train_script._validate_v35_cumulative_telemetry(telemetry, completed_updates=0)  # noqa: SLF001

    checkpoints.save_state(
        manager,
        state,
        loader,
        FINAL_CHECKPOINT_STEP,
        provenance_assets=provenance,
        v35_cumulative_telemetry=telemetry,
    )
    manager.wait_until_finished()
    if tuple(manager.all_steps()) != (0,):
        raise Step0BootstrapError("final completed-update-0 checkpoint did not commit")

    # Authenticate the just-written exact-resume state using a second freshly constructed
    # loader.  Loading the state is allowed; constructing an iterator is intentionally absent.
    verification_loader = data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    restored_telemetry = checkpoints.restore_v35_runtime_state(root, 0, verification_loader)
    if restored_telemetry != telemetry or verification_loader.state_dict() != initial_loader_state:
        raise Step0BootstrapError("final checkpoint does not restore the exact initial loader/RNG boundary")
    return identity_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initialize_parser = commands.add_parser("initialize", help="Create raw audited zero-update state.")
    initialize_parser.add_argument("--experiment-name", required=True)
    initialize_parser.add_argument("--fsdp-devices", type=int)
    finalize_parser = commands.add_parser("finalize", help="Seal calibration-bound checkpoint 0.")
    finalize_parser.add_argument("--experiment-name", required=True)
    finalize_parser.add_argument(
        "--calibration",
        required=True,
        help="memory_project-relative passing train-54 calibration artifact",
    )
    finalize_parser.add_argument("--fsdp-devices", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "initialize":
        output = initialize(experiment_name=args.experiment_name, fsdp_devices=args.fsdp_devices)
    else:
        output = finalize(
            experiment_name=args.experiment_name,
            calibration_relative=args.calibration,
            fsdp_devices=args.fsdp_devices,
        )
    print(_project_relative(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
