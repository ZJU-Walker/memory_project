from __future__ import annotations

import asyncio
import collections.abc
import concurrent.futures as futures
import dataclasses
import hashlib
import json
import logging
import pathlib
from typing import Any, Protocol

from etils import epath
import jax
import orbax.checkpoint as ocp
import orbax.checkpoint.future as future

from openpi.shared import array_typing as at
import openpi.shared.normalize as _normalize
import openpi.training.data_loader as _data_loader
import openpi.training.utils as training_utils

V35_DATA_ITERATOR_STATE_FILENAME = "v35_data_iterator_state.json"
V35_CUMULATIVE_TELEMETRY_FILENAME = "v35_cumulative_telemetry.json"
V35_RUNTIME_IDENTITY_FILENAME = "v35_runtime_identity.json"
_V35_INITIALIZATION_IDENTITY_FILENAME = "v35_initialization_manifest.json"


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _sha256(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def v35_checkpoint_component_tree_sha256(path: epath.Path | str) -> str:
    """Hash one immutable Orbax checkpoint component using the rung-evidence protocol.

    The digest covers every relative file name, byte size, and file digest.  It deliberately
    rejects symlinks so a live launch cannot validate one target and restore another.  This is
    the same representation sealed as ``optimizer_state_sha256`` by the v3.5 rung reducer.
    """

    root = pathlib.Path(str(path))
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"v3.5 checkpoint directory component is missing or unsafe: {root}")
    entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    if any(item.is_symlink() for item in entries):
        raise ValueError(f"v3.5 checkpoint directory component contains a symlink: {root}")
    files = [item for item in entries if item.is_file()]
    if not files:
        raise ValueError(f"v3.5 checkpoint directory component contains no regular files: {root}")
    inventory = [
        {
            "path": item.relative_to(root).as_posix(),
            "sha256": _sha256(item.read_bytes()),
            "size": item.stat().st_size,
        }
        for item in files
    ]
    canonical = json.dumps(
        inventory,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256(canonical)


def _snapshot_v35_runtime_assets(
    data_loader: _data_loader.DataLoader,
    cumulative_telemetry: collections.abc.Mapping[str, Any],
    *,
    completed_updates: int,
    run_initialization_identity: bytes,
) -> dict[str, bytes]:
    """Create canonical, mutually bound bytes before Orbax starts its async save."""
    if completed_updates < 0:
        raise ValueError("v3.5 completed-update count must be nonnegative.")
    data_state_bytes = _canonical_json_bytes(data_loader.state_dict())
    telemetry_bytes = _canonical_json_bytes(dict(cumulative_telemetry))
    unsigned_identity = {
        "format_version": 1,
        "completed_updates": completed_updates,
        "run_initialization_identity_sha256": _sha256(run_initialization_identity),
        "data_iterator_state_file": V35_DATA_ITERATOR_STATE_FILENAME,
        "data_iterator_state_sha256": _sha256(data_state_bytes),
        "cumulative_telemetry_file": V35_CUMULATIVE_TELEMETRY_FILENAME,
        "cumulative_telemetry_sha256": _sha256(telemetry_bytes),
    }
    identity = dict(unsigned_identity)
    identity["identity_sha256"] = _sha256(_canonical_json_bytes(unsigned_identity)[:-1])
    return {
        V35_DATA_ITERATOR_STATE_FILENAME: data_state_bytes,
        V35_CUMULATIVE_TELEMETRY_FILENAME: telemetry_bytes,
        V35_RUNTIME_IDENTITY_FILENAME: _canonical_json_bytes(identity),
    }


def _write_provenance_assets(directory: epath.Path, assets: collections.abc.Mapping[str, bytes]) -> None:
    """Write already-snapshotted provenance without text decoding or reserialization."""
    for name, contents in assets.items():
        (directory / name).write_bytes(contents)


def initialize_checkpoint_dir(
    checkpoint_dir: epath.Path | str,
    *,
    keep_period: int | None,
    overwrite: bool,
    resume: bool,
    allow_step_zero_resume: bool = False,
) -> tuple[ocp.CheckpointManager, bool]:
    checkpoint_dir = epath.Path(checkpoint_dir).resolve()
    resuming = False
    if checkpoint_dir.exists():
        if overwrite:
            checkpoint_dir.rmtree()
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Wiped checkpoint directory {checkpoint_dir}")
        elif resume:
            resuming = True
        else:
            raise FileExistsError(
                f"Checkpoint directory {checkpoint_dir} already exists. Use --overwrite or --resume "
                "to indicate how to handle it."
            )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    mngr = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers={
            "assets": CallbackHandler(),
            "train_state": ocp.PyTreeCheckpointHandler(),
            "params": ocp.PyTreeCheckpointHandler(),
        },
        options=ocp.CheckpointManagerOptions(
            max_to_keep=1,
            keep_period=keep_period,
            create=False,
            async_options=ocp.AsyncOptions(timeout_secs=7200),
        ),
    )

    # Special case: the checkpoint directory exists and the user requests to resume training, but the training run did
    # not get to the first checkpoint saved. In this case, we don't actually want the train script to try and restore a
    # checkpoint, since it will fail.
    empty_or_initialization_only = tuple(mngr.all_steps()) == () or (
        tuple(mngr.all_steps()) == (0,) and not allow_step_zero_resume
    )
    if resuming and empty_or_initialization_only:
        logging.info("Checkpoint directory exists, but does not contain any checkpoints. Aborting resume.")
        resuming = False

    return mngr, resuming


def save_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int,
    *,
    initialization_manifest_path: epath.Path | str | None = None,
    provenance_assets: collections.abc.Mapping[str, bytes] | None = None,
    v35_cumulative_telemetry: collections.abc.Mapping[str, Any] | None = None,
):
    # Snapshot provenance before Orbax starts its asynchronous save. This prevents a mutable
    # source file from changing between the checkpoint request and the assets callback.
    frozen_provenance_assets = {name: bytes(contents) for name, contents in (provenance_assets or {}).items()}
    for name in frozen_provenance_assets:
        if not name or epath.Path(name).name != name:
            raise ValueError(f"checkpoint provenance asset must be a plain filename, got {name!r}")

    frozen_runtime_assets: dict[str, bytes] = {}
    if v35_cumulative_telemetry is not None:
        try:
            run_initialization_identity = frozen_provenance_assets[_V35_INITIALIZATION_IDENTITY_FILENAME]
        except KeyError as exc:
            raise ValueError(
                "v3.5 runtime state requires the authenticated initialization identity provenance asset."
            ) from exc
        frozen_runtime_assets = _snapshot_v35_runtime_assets(
            data_loader,
            v35_cumulative_telemetry,
            completed_updates=step,
            run_initialization_identity=run_initialization_identity,
        )
        collisions = set(frozen_runtime_assets) & set(frozen_provenance_assets)
        if collisions:
            raise ValueError(f"v3.5 runtime asset names collide with provenance: {sorted(collisions)}")

    def save_assets(directory: epath.Path):
        # Save the normalization stats.
        data_config = data_loader.data_config()
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(directory / data_config.asset_id, norm_stats)
        if initialization_manifest_path is not None:
            source = epath.Path(initialization_manifest_path)
            if not source.is_file():
                raise FileNotFoundError(f"initialization manifest disappeared before checkpoint save: {source}")
            (directory / "v35_initialization_manifest.json").write_text(source.read_text())
        _write_provenance_assets(directory, frozen_provenance_assets)
        _write_provenance_assets(directory, frozen_runtime_assets)

    # Split params that can be used for inference into a separate item.
    with at.disable_typechecking():
        train_state, params = _split_params(state)
    items = {
        "assets": save_assets,
        "train_state": train_state,
        "params": {"params": params},
    }
    checkpoint_manager.save(step, items)


def restore_state(
    checkpoint_manager: ocp.CheckpointManager,
    state: training_utils.TrainState,
    data_loader: _data_loader.DataLoader,
    step: int | None = None,
) -> training_utils.TrainState:
    del data_loader

    with at.disable_typechecking():
        # Split params that can be used for inference into a separate item.
        train_state, params = _split_params(state)
        restored = checkpoint_manager.restore(
            step,
            items={
                "train_state": train_state,
                "params": {"params": params},
            },
        )
    return _merge_params(restored["train_state"], restored["params"])


def restore_v35_runtime_state(
    checkpoint_dir: epath.Path | str,
    checkpoint_step: int,
    data_loader: _data_loader.DataLoader,
) -> dict[str, Any]:
    """Authenticate and restore exact data continuation plus cumulative Gate-D telemetry."""
    assets_dir = epath.Path(checkpoint_dir) / str(checkpoint_step) / "assets"
    paths = {
        name: assets_dir / name
        for name in (
            V35_DATA_ITERATOR_STATE_FILENAME,
            V35_CUMULATIVE_TELEMETRY_FILENAME,
            V35_RUNTIME_IDENTITY_FILENAME,
            _V35_INITIALIZATION_IDENTITY_FILENAME,
        )
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"v3.5 checkpoint {checkpoint_step} is missing exact-resume assets: {sorted(missing)}")
    try:
        identity_bytes = paths[V35_RUNTIME_IDENTITY_FILENAME].read_bytes()
        identity = json.loads(identity_bytes)
        data_state_bytes = paths[V35_DATA_ITERATOR_STATE_FILENAME].read_bytes()
        data_state = json.loads(data_state_bytes)
        telemetry_bytes = paths[V35_CUMULATIVE_TELEMETRY_FILENAME].read_bytes()
        telemetry = json.loads(telemetry_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read v3.5 checkpoint runtime state: {exc}") from exc
    if not isinstance(identity, dict) or not isinstance(data_state, dict) or not isinstance(telemetry, dict):
        raise ValueError("v3.5 checkpoint runtime assets must be JSON objects.")
    if identity_bytes != _canonical_json_bytes(identity):
        raise ValueError("v3.5 runtime identity is not canonical JSON.")
    if data_state_bytes != _canonical_json_bytes(data_state):
        raise ValueError("v3.5 data iterator state is not canonical JSON.")
    if telemetry_bytes != _canonical_json_bytes(telemetry):
        raise ValueError("v3.5 cumulative telemetry is not canonical JSON.")

    recorded_identity_sha256 = identity.get("identity_sha256")
    unsigned_identity = {key: value for key, value in identity.items() if key != "identity_sha256"}
    expected_identity_sha256 = _sha256(_canonical_json_bytes(unsigned_identity)[:-1])
    expected_identity = {
        "format_version": 1,
        "completed_updates": checkpoint_step,
        "run_initialization_identity_sha256": _sha256(paths[_V35_INITIALIZATION_IDENTITY_FILENAME].read_bytes()),
        "data_iterator_state_file": V35_DATA_ITERATOR_STATE_FILENAME,
        "data_iterator_state_sha256": _sha256(data_state_bytes),
        "cumulative_telemetry_file": V35_CUMULATIVE_TELEMETRY_FILENAME,
        "cumulative_telemetry_sha256": _sha256(telemetry_bytes),
    }
    if recorded_identity_sha256 != expected_identity_sha256 or unsigned_identity != expected_identity:
        raise ValueError("v3.5 checkpoint runtime identity/hash binding is invalid.")

    # The loader must still have no iterator: this restore installs the exact sampler and
    # transform RNG boundary from which the caller constructs the first resumed iterator.
    data_loader.load_state_dict(data_state)
    return telemetry


def load_norm_stats(assets_dir: epath.Path | str, asset_id: str) -> dict[str, _normalize.NormStats] | None:
    norm_stats_dir = epath.Path(assets_dir) / asset_id
    norm_stats = _normalize.load(norm_stats_dir)
    logging.info(f"Loaded norm stats from {norm_stats_dir}")
    return norm_stats


class Callback(Protocol):
    def __call__(self, directory: epath.Path) -> None: ...


class CallbackHandler(ocp.AsyncCheckpointHandler):
    """A CheckpointHandler for calling an arbitrary function asynchronously. Only for saving, not for restoring."""

    def save(self, directory: epath.Path, args: CallbackSave):
        if jax.process_index() == 0:
            args.callback(directory)

    async def async_save(self, directory: epath.Path, args: CallbackSave) -> list[futures.Future]:
        return [future.CommitFutureAwaitingContractedSignals(asyncio.to_thread(self.save, directory, args))]

    def restore(self, *args, **kwargs):
        raise NotImplementedError("CallbackHandler does not support restore")


@ocp.args.register_with_handler(CallbackHandler, for_save=True)
@dataclasses.dataclass
class CallbackSave(ocp.args.CheckpointArgs):
    callback: Callback


@ocp.args.register_with_handler(CallbackHandler, for_restore=True)
class CallbackRestore(ocp.args.CheckpointArgs): ...


def _split_params(state: training_utils.TrainState) -> tuple[training_utils.TrainState, at.Params]:
    if state.ema_params is not None:
        params = state.ema_params
        train_state = dataclasses.replace(state, ema_params=None)
    else:
        params = state.params
        train_state = dataclasses.replace(state, params={})
    return train_state, params


def _merge_params(train_state: training_utils.TrainState, params: dict[str, at.Params]) -> training_utils.TrainState:
    # Revert the logic inside `_split_params`. Assumes that existence of `params` means that EMA params were used during the split.
    if train_state.params:
        return dataclasses.replace(train_state, ema_params=params["params"])
    return dataclasses.replace(train_state, params=params["params"])
