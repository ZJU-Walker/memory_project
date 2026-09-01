"""Fail-closed runtime validation for v3.5 launch authorizations.

The expensive Gate-A/B/C/D reducers live in :mod:`scripts.v35_training_authorization`.
Training consumes only their canonical, self-hashed result.  Every machine-local path in
an authorization is relative to ``memory_project`` so the complete experiment can be
copied to another cluster without changing an authenticated byte.
"""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import enum
import hashlib
from importlib import metadata as importlib_metadata
import json
import pathlib
import platform
import re
from typing import Any

import tyro

import openpi.shared.nnx_utils as nnx_utils
import openpi.shared.project_paths as project_paths

AUTHORIZATION_SCHEMA_VERSION = "openpi.v35.training-authorization.v2"
SEMANTIC_CONFIG_SCHEMA_VERSION = "openpi.v35.semantic-training-config.v2"
FROZEN_SOURCE_IDENTITY_SCHEMA_VERSION = "openpi.v35.frozen-source-identity.v1"
PILOT_AUTHORIZATION_KIND = "pilot_1000"
CONTINUATION_AUTHORIZATION_KIND = "continuation"
OFFICIAL_BASE_URI = "gs://openpi-assets/checkpoints/pi05_base/params"
PILOT_AUTHORIZATION_FILENAME = "v35_pilot_authorization.json"
CONTINUATION_AUTHORIZATION_FILENAME = "v35_continuation_authorization.json"

# These are the only top-level TrainConfig fields omitted from the semantic identity.
# They select a frozen rung/launch mode or point at the authorization that permits that
# mode; none changes the optimizer update computed from a given batch and state.
SEMANTIC_CONFIG_EXCLUDED_FIELDS = frozenset(
    {
        "checkpoint_steps",
        "num_train_steps",
        "resume",
        "v35_continuation_authorization_path",
        "v35_pilot_authorization_path",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SEMANTIC_SOURCE_PROTOCOL = "openpi.v35.training-and-gate-source-tree.v2"
_SEMANTIC_SOURCE_STATIC_FILES = (
    "openpi/pyproject.toml",
    "openpi/uv.lock",
    "openpi/scripts/compute_norm_stats.py",
    "openpi/scripts/eval_yam_mem_subtask_raw.py",
    "openpi/scripts/train.py",
)
_SEMANTIC_RUNTIME_DISTRIBUTIONS = (
    "datasets",
    "einops",
    "flax",
    "jax",
    "jaxlib",
    "jaxtyping",
    "lerobot",
    "numpy",
    "optax",
    "orbax-checkpoint",
    "scipy",
    "torch",
    "transformers",
    "wandb",
)


class V35AuthorizationError(ValueError):
    """Raised when an authorization is missing, mutable, or bound to another run."""


@dataclasses.dataclass(frozen=True)
class AuthorizationRecord:
    path: pathlib.Path
    file_sha256: str
    artifact_id: str
    payload: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class FrozenSourceIdentityRecord:
    path: pathlib.Path
    file_sha256: str
    artifact_id: str
    source_identity: Mapping[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise V35AuthorizationError(f"{name} must be a lower-case 64-character SHA256 digest")
    return value


def require_artifact_id(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise V35AuthorizationError(f"{name} must have the form sha256:<lower-case digest>")
    require_sha256(name, value.removeprefix("sha256:"))
    return value


def _project_portable_absolute(path: pathlib.Path) -> dict[str, str]:
    try:
        relative = project_paths.project_relative_path(path).as_posix()
    except project_paths.ProjectRootError as exc:
        raise V35AuthorizationError(f"v3.5 semantic config contains an out-of-project local path: {path}") from exc
    return {"__memory_project_relative_path__": relative}


def _semantic_value(value: Any) -> Any:
    """Convert config state to a type-preserving, portable JSON value."""

    if value is tyro.MISSING:
        return {"__tyro_missing__": "propagating"}
    if isinstance(value, nnx_utils.PathRegex):
        pattern = value.pattern
        if not isinstance(pattern, re.Pattern):
            raise V35AuthorizationError("PathRegex must contain a compiled regular expression")
        return {
            "__path_regex__": pattern.pattern,
            "flags": pattern.flags,
            "separator": value.sep,
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = {field.name: _semantic_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
        return {
            "__dataclass__": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": fields,
        }
    if isinstance(value, pathlib.PurePath):
        if value.is_absolute():
            return _project_portable_absolute(pathlib.Path(value))
        if ".." in value.parts:
            raise V35AuthorizationError(f"v3.5 semantic config path escapes memory_project: {value}")
        return {"__relative_path__": pathlib.PurePosixPath(*value.parts).as_posix()}
    if isinstance(value, re.Pattern):
        return {"__regex__": value.pattern, "flags": value.flags}
    if isinstance(value, enum.Enum):
        return {
            "__enum__": f"{type(value).__module__}.{type(value).__qualname__}",
            "value": _semantic_value(value.value),
        }
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise V35AuthorizationError("v3.5 semantic config mappings must use string keys")
        return {key: _semantic_value(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple | list):
        return [_semantic_value(item) for item in value]
    if value is None or type(value) in (bool, int, float, str):
        if isinstance(value, float) and not (-float("inf") < value < float("inf")):
            raise V35AuthorizationError("v3.5 semantic config contains a non-finite float")
        if isinstance(value, str) and pathlib.PurePath(value).is_absolute():
            return _project_portable_absolute(pathlib.Path(value))
        return value
    raise V35AuthorizationError(
        f"v3.5 semantic config contains an unsupported value {type(value).__module__}.{type(value).__qualname__}"
    )


def semantic_source_identity() -> dict[str, Any]:
    """Hash the exact source bytes that define v3.5 training and its launch gates.

    Source ancestry, rather than ``MEMORY_PROJECT_ROOT``, identifies the code that Python is
    actually executing. Test fixtures may relocate data roots, while a production copied
    checkout still binds the executable source tree byte-for-byte.
    """

    module_path = pathlib.Path(__file__).resolve()
    try:
        source_root = project_paths.discover_memory_project_root(source_file=module_path, cwd=module_path.parent)
        module_path.relative_to(source_root)
    except (project_paths.ProjectRootError, ValueError) as exc:
        raise V35AuthorizationError("cannot bind v3.5 semantic identity to its executing source checkout") from exc
    selected = set(_SEMANTIC_SOURCE_STATIC_FILES)
    selected.update(
        path.relative_to(source_root).as_posix() for path in (source_root / "openpi/src/openpi").rglob("*.py")
    )
    selected.update(
        path.relative_to(source_root).as_posix() for path in (source_root / "openpi/scripts").glob("v35_*.py")
    )
    files: dict[str, str] = {}
    for relative in sorted(selected):
        unresolved = source_root / pathlib.PurePosixPath(relative)
        path = unresolved.resolve()
        try:
            path.relative_to(source_root)
            payload = path.read_bytes()
        except (ValueError, OSError) as exc:
            raise V35AuthorizationError(f"cannot hash required v3.5 source file {relative!r}") from exc
        if not path.is_file() or unresolved.is_symlink():
            raise V35AuthorizationError(f"required v3.5 source file is not a regular file: {relative!r}")
        files[relative] = sha256_bytes(payload)
    runtime = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "distributions": {},
    }
    for distribution in _SEMANTIC_RUNTIME_DISTRIBUTIONS:
        try:
            runtime["distributions"][distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise V35AuthorizationError(f"required v3.5 runtime distribution is missing: {distribution}") from exc
    unsigned = {
        "protocol": _SEMANTIC_SOURCE_PROTOCOL,
        "selection": {
            "package_glob": "openpi/src/openpi/**/*.py",
            "v35_script_glob": "openpi/scripts/v35_*.py",
            "static_files": list(_SEMANTIC_SOURCE_STATIC_FILES),
        },
        "files": files,
        "runtime": runtime,
    }
    return {**unsigned, "aggregate_sha256": sha256_bytes(canonical_json_bytes(unsigned))}


def frozen_source_identity_envelope() -> dict[str, Any]:
    """Return the create-once source/runtime identity frozen before any pilot stage.

    The source identity is deliberately separate from the calibrated semantic config: it can
    be sealed before calibration exists, then checked before every producer/skip and by the
    eventual training authorization.
    """

    payload = {
        "schema_version": FROZEN_SOURCE_IDENTITY_SCHEMA_VERSION,
        "source_identity": semantic_source_identity(),
        "status": "frozen_before_pre_pilot",
    }
    return {
        "artifact_id": f"sha256:{sha256_bytes(canonical_json_bytes(payload))}",
        "payload": payload,
        "schema_version": FROZEN_SOURCE_IDENTITY_SCHEMA_VERSION,
    }


def load_and_validate_frozen_source_identity(path: pathlib.Path) -> FrozenSourceIdentityRecord:
    """Authenticate a run's source snapshot and require this process to match it exactly."""

    path = pathlib.Path(path)
    project_paths.project_relative_path(path)
    try:
        raw_bytes = path.read_bytes()
        value = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise V35AuthorizationError(f"cannot read frozen v3.5 source identity {path}: {exc}") from exc
    value = _exact_mapping(
        "frozen source identity envelope",
        value,
        {"artifact_id", "payload", "schema_version"},
    )
    payload = _exact_mapping(
        "frozen source identity payload",
        value["payload"],
        {"schema_version", "source_identity", "status"},
    )
    artifact_id = f"sha256:{sha256_bytes(canonical_json_bytes(payload))}"
    if (
        value["schema_version"] != FROZEN_SOURCE_IDENTITY_SCHEMA_VERSION
        or payload["schema_version"] != FROZEN_SOURCE_IDENTITY_SCHEMA_VERSION
        or payload["status"] != "frozen_before_pre_pilot"
        or value["artifact_id"] != artifact_id
        or raw_bytes != canonical_json_bytes(value) + b"\n"
    ):
        raise V35AuthorizationError("frozen v3.5 source identity has an invalid schema, hash, or encoding")
    source_identity = payload["source_identity"]
    if not isinstance(source_identity, Mapping):
        raise V35AuthorizationError("frozen v3.5 source identity payload is not an object")
    current = semantic_source_identity()
    if source_identity != current:
        frozen_aggregate = source_identity.get("aggregate_sha256")
        current_aggregate = current.get("aggregate_sha256")
        raise V35AuthorizationError(
            "v3.5 source/runtime identity changed after pre-pilot freeze; use a new experiment "
            f"(frozen={frozen_aggregate}, current={current_aggregate})"
        )
    return FrozenSourceIdentityRecord(
        path=path.resolve(),
        file_sha256=sha256_bytes(raw_bytes),
        artifact_id=artifact_id,
        source_identity=dict(source_identity),
    )


def semantic_training_config_payload(config: Any) -> dict[str, Any]:
    """Return the exact optimizer semantics, excluding only frozen target/auth controls."""

    if not dataclasses.is_dataclass(config) or isinstance(config, type):
        raise V35AuthorizationError("semantic training config requires a dataclass instance")
    fields = {
        field.name: _semantic_value(getattr(config, field.name))
        for field in dataclasses.fields(config)
        if field.name not in SEMANTIC_CONFIG_EXCLUDED_FIELDS
    }
    return {
        "schema_version": SEMANTIC_CONFIG_SCHEMA_VERSION,
        "excluded_top_level_fields": sorted(SEMANTIC_CONFIG_EXCLUDED_FIELDS),
        "train_config_type": f"{type(config).__module__}.{type(config).__qualname__}",
        "fields": fields,
        "source_identity": semantic_source_identity(),
    }


def semantic_training_config_sha256(config: Any) -> str:
    return sha256_bytes(canonical_json_bytes(semantic_training_config_payload(config)))


def run_id_sha256(
    *,
    config_name: str,
    experiment_name: str,
    initialization_seed: int,
    initialization_parameter_tree_sha256: str,
    calibration_artifact_id: str,
    semantic_config_sha256: str,
) -> str:
    """Return the stable same-run identity used by launch and Gate-D rung artifacts."""

    if not config_name or not experiment_name or type(initialization_seed) is not int:
        raise V35AuthorizationError("run identity requires config, experiment, and integer seed")
    require_sha256("initialization_parameter_tree_sha256", initialization_parameter_tree_sha256)
    require_artifact_id("calibration_artifact_id", calibration_artifact_id)
    require_sha256("semantic_config_sha256", semantic_config_sha256)
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": "openpi.v35.run-identity.v1",
                "config_name": config_name,
                "experiment_name": experiment_name,
                "initialization_seed": initialization_seed,
                "initialization_parameter_tree_sha256": initialization_parameter_tree_sha256,
                "calibration_artifact_id": calibration_artifact_id,
                "semantic_training_config_sha256": semantic_config_sha256,
            }
        )
    )


def authorization_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "artifact_id": f"sha256:{sha256_bytes(canonical_json_bytes(payload))}",
        "payload": payload,
    }


def write_authorization_once(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    """Write a canonical authorization without ever replacing an existing decision."""

    expected = authorization_envelope(value.get("payload", {}))
    if dict(value) != expected:
        raise V35AuthorizationError("refusing to write an invalid authorization envelope")
    path = pathlib.Path(path)
    project_paths.project_relative_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(canonical_json_bytes(value) + b"\n")
    except FileExistsError:
        if path.read_bytes() != canonical_json_bytes(value) + b"\n":
            raise V35AuthorizationError(f"refusing to overwrite a different authorization: {path}") from None


def _exact_mapping(name: str, value: Any, keys: set[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise V35AuthorizationError(f"{name} must be an object")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise V35AuthorizationError(f"{name} keys mismatch: missing={missing}, extra={extra}")
    return value


def load_authorization(path: pathlib.Path, *, expected_kind: str) -> AuthorizationRecord:
    path = pathlib.Path(path)
    project_paths.project_relative_path(path)
    try:
        raw_bytes = path.read_bytes()
        value = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise V35AuthorizationError(f"cannot read v3.5 authorization {path}: {exc}") from exc
    value = _exact_mapping("authorization envelope", value, {"artifact_id", "payload", "schema_version"})
    payload = value["payload"]
    if value["schema_version"] != AUTHORIZATION_SCHEMA_VERSION or not isinstance(payload, Mapping):
        raise V35AuthorizationError("v3.5 authorization has the wrong schema or payload type")
    artifact_id = f"sha256:{sha256_bytes(canonical_json_bytes(payload))}"
    if value["artifact_id"] != artifact_id:
        raise V35AuthorizationError("v3.5 authorization self-hash is invalid")
    if raw_bytes != canonical_json_bytes(value) + b"\n":
        raise V35AuthorizationError("v3.5 authorization is not canonical JSON")
    if payload.get("authorization_kind") != expected_kind or payload.get("status") != "pass":
        raise V35AuthorizationError(
            f"expected a passing {expected_kind!r} authorization, got "
            f"kind={payload.get('authorization_kind')!r}, status={payload.get('status')!r}"
        )
    return AuthorizationRecord(
        path=path.resolve(),
        file_sha256=sha256_bytes(raw_bytes),
        artifact_id=artifact_id,
        payload=dict(payload),
    )


def _configured_project_relative_path(config: Any, field_name: str) -> pathlib.Path:
    value = getattr(config, field_name, None)
    if not isinstance(value, str) or not value.strip():
        raise V35AuthorizationError(f"v3.5 requires --{field_name.replace('_', '-')}")
    relative = pathlib.PurePath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise V35AuthorizationError(f"{field_name} must be memory_project-relative, got {value!r}")
    return project_paths.project_path(relative)


def _file_sha256(path: pathlib.Path, *, name: str) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise V35AuthorizationError(f"cannot read {name} {path}: {exc}") from exc


def _load_json(path: pathlib.Path, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise V35AuthorizationError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise V35AuthorizationError(f"{name} must be a JSON object: {path}")
    return value


def _norm_paths(config: Any) -> tuple[pathlib.Path, pathlib.Path]:
    data_factory = config.data
    asset_id = data_factory.assets.asset_id or data_factory.repo_id
    assets_dir = pathlib.Path(data_factory.assets.assets_dir or config.assets_dirs)
    norm_dir = assets_dir / asset_id
    return norm_dir / "norm_stats.json", norm_dir / "norm_stats_provenance.json"


def _validate_evidence_descriptor(name: str, value: Any) -> None:
    descriptor = _exact_mapping(
        f"gate_evidence.{name}",
        value,
        {"artifact_id", "criteria_version", "path_relative", "sha256"},
    )
    require_artifact_id(f"gate_evidence.{name}.artifact_id", descriptor["artifact_id"])
    require_sha256(f"gate_evidence.{name}.sha256", descriptor["sha256"])
    relative = descriptor["path_relative"]
    if not isinstance(relative, str) or not relative:
        raise V35AuthorizationError(f"gate_evidence.{name}.path_relative must be non-empty")
    path = project_paths.project_path(relative)
    raw_bytes = path.read_bytes()
    if sha256_bytes(raw_bytes) != descriptor["sha256"]:
        raise V35AuthorizationError(f"gate_evidence.{name} file SHA256 changed: {path}")
    try:
        envelope = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise V35AuthorizationError(f"gate_evidence.{name} is invalid JSON: {path}") from exc
    if not isinstance(envelope, Mapping) or envelope.get("artifact_id") != descriptor["artifact_id"]:
        raise V35AuthorizationError(f"gate_evidence.{name} artifact ID changed: {path}")


_SOURCE_CHECKPOINT_KEYS = {
    "completed_updates",
    "cumulative_telemetry_sha256",
    "data_iterator_state_sha256",
    "optimizer_state_sha256",
    "parameter_tree_sha256",
    "rung_artifact_id",
    "rung_file_sha256",
    "rung_path_relative",
    "runtime_identity_sha256",
}


def _validate_source_checkpoint_descriptor(
    name: str,
    value: Any,
    *,
    expected_completed_updates: int | None = None,
    expected_rung_descriptor: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Authenticate the external rung that freezes every live resume component."""

    source = _exact_mapping(name, value, _SOURCE_CHECKPOINT_KEYS)
    completed_updates = source["completed_updates"]
    if type(completed_updates) is not int or completed_updates < 0:
        raise V35AuthorizationError(f"{name}.completed_updates must be a nonnegative integer")
    if expected_completed_updates is not None and completed_updates != expected_completed_updates:
        raise V35AuthorizationError(
            f"{name} completed update {completed_updates} differs from authorized source {expected_completed_updates}"
        )
    for key in (
        "cumulative_telemetry_sha256",
        "data_iterator_state_sha256",
        "optimizer_state_sha256",
        "parameter_tree_sha256",
        "rung_file_sha256",
        "runtime_identity_sha256",
    ):
        require_sha256(f"{name}.{key}", source[key])
    require_artifact_id(f"{name}.rung_artifact_id", source["rung_artifact_id"])
    relative = source["rung_path_relative"]
    if not isinstance(relative, str) or not relative:
        raise V35AuthorizationError(f"{name}.rung_path_relative must be non-empty")
    rung_path = project_paths.project_path(relative)
    try:
        rung_bytes = rung_path.read_bytes()
        rung = json.loads(rung_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise V35AuthorizationError(f"cannot read {name} rung {rung_path}: {exc}") from exc
    if sha256_bytes(rung_bytes) != source["rung_file_sha256"]:
        raise V35AuthorizationError(f"{name} rung file SHA256 changed: {rung_path}")
    if not isinstance(rung, Mapping) or rung.get("artifact_id") != source["rung_artifact_id"]:
        raise V35AuthorizationError(f"{name} rung artifact ID changed: {rung_path}")
    payload = rung.get("payload")
    checkpoint = payload.get("checkpoint") if isinstance(payload, Mapping) else None
    provenance = payload.get("run_provenance") if isinstance(payload, Mapping) else None
    expected_checkpoint = {
        "completed_updates": completed_updates,
        "parameter_tree_sha256": source["parameter_tree_sha256"],
    }
    expected_provenance = {
        "cumulative_telemetry_sha256": source["cumulative_telemetry_sha256"],
        "data_iterator_state_sha256": source["data_iterator_state_sha256"],
        "optimizer_state_sha256": source["optimizer_state_sha256"],
        "runtime_identity_sha256": source["runtime_identity_sha256"],
    }
    if (
        checkpoint != expected_checkpoint
        or not isinstance(provenance, Mapping)
        or any(provenance.get(key) != digest for key, digest in expected_provenance.items())
    ):
        raise V35AuthorizationError(f"{name} does not match its sealed rung checkpoint/runtime identity")
    if expected_rung_descriptor is not None:
        expected = {
            "artifact_id": expected_rung_descriptor.get("artifact_id"),
            "path_relative": expected_rung_descriptor.get("path_relative"),
            "sha256": expected_rung_descriptor.get("sha256"),
        }
        actual = {
            "artifact_id": source["rung_artifact_id"],
            "path_relative": relative,
            "sha256": source["rung_file_sha256"],
        }
        if actual != expected:
            raise V35AuthorizationError(f"{name} does not reference the authorization-linked rung")
    return source


def _validate_prepilot_source_identity_descriptor(value: Any) -> FrozenSourceIdentityRecord:
    descriptor = _exact_mapping(
        "prepilot_source_identity",
        value,
        {"artifact_id", "path_relative", "sha256", "source_aggregate_sha256"},
    )
    require_artifact_id("prepilot_source_identity.artifact_id", descriptor["artifact_id"])
    require_sha256("prepilot_source_identity.sha256", descriptor["sha256"])
    require_sha256(
        "prepilot_source_identity.source_aggregate_sha256",
        descriptor["source_aggregate_sha256"],
    )
    relative = descriptor["path_relative"]
    if not isinstance(relative, str) or not relative:
        raise V35AuthorizationError("prepilot_source_identity.path_relative must be non-empty")
    record = load_and_validate_frozen_source_identity(project_paths.project_path(relative))
    if (
        record.artifact_id != descriptor["artifact_id"]
        or record.file_sha256 != descriptor["sha256"]
        or record.source_identity.get("aggregate_sha256") != descriptor["source_aggregate_sha256"]
    ):
        raise V35AuthorizationError("prepilot source identity descriptor changed after authorization")
    return record


def _validate_common_pilot_payload(config: Any, payload: Mapping[str, Any]) -> None:
    required = {
        "authorization_kind",
        "authorized_target_completed_updates",
        "calibration_identity",
        "config_identity",
        "dataset_identity",
        "final_test_remains_sealed",
        "gate_evidence",
        "initialization_identity",
        "prepilot_source_identity",
        "protocols",
        "run_identity",
        "semantic_config_schema_version",
        "semantic_training_config_sha256",
        "source_checkpoint",
        "status",
    }
    _exact_mapping("pilot authorization payload", payload, required)
    # Check the create-once prepilot source snapshot before recomputing the semantic config,
    # so source/runtime drift is reported as the violated temporal freeze contract.
    _validate_prepilot_source_identity_descriptor(payload["prepilot_source_identity"])
    if payload["authorized_target_completed_updates"] != 1_000 or payload["final_test_remains_sealed"] is not True:
        raise V35AuthorizationError("pilot authorization must authorize only 1,000 updates with final-test sealed")
    if payload["semantic_config_schema_version"] != SEMANTIC_CONFIG_SCHEMA_VERSION:
        raise V35AuthorizationError("pilot authorization uses the wrong semantic-config schema")
    semantic_sha256 = semantic_training_config_sha256(config)
    if payload["semantic_training_config_sha256"] != semantic_sha256:
        raise V35AuthorizationError("pilot authorization is bound to a different semantic training config")

    config_identity = _exact_mapping(
        "config_identity", payload["config_identity"], {"config_name", "experiment_name", "initialization_seed"}
    )
    if config_identity != {
        "config_name": config.name,
        "experiment_name": config.exp_name,
        "initialization_seed": config.seed,
    }:
        raise V35AuthorizationError("pilot authorization is bound to a different config/experiment/seed")

    data_config = config.data.base_config
    manifest_path = pathlib.Path(data_config.memory_episode_manifest_path)
    norm_stats_path, norm_provenance_path = _norm_paths(config)
    norm_provenance = _load_json(norm_provenance_path, name="v3.5 norm provenance")
    storage = norm_provenance.get("train_storage")
    computation = norm_provenance.get("computation")
    if not isinstance(storage, Mapping) or not isinstance(computation, Mapping):
        raise V35AuthorizationError("v3.5 norm provenance is missing train_storage/computation")
    dataset_identity = _exact_mapping(
        "dataset_identity",
        payload["dataset_identity"],
        {
            "episode_manifest_sha256",
            "norm_computation_protocol",
            "norm_stats_provenance_sha256",
            "norm_stats_sha256",
            "split_assignment_sha256",
            "train_storage_sha256",
        },
    )
    require_sha256("dataset_identity.split_assignment_sha256", dataset_identity["split_assignment_sha256"])
    expected_dataset = {
        "episode_manifest_sha256": _file_sha256(manifest_path, name="frozen episode manifest"),
        "norm_computation_protocol": computation.get("protocol"),
        "norm_stats_provenance_sha256": _file_sha256(norm_provenance_path, name="norm provenance"),
        "norm_stats_sha256": _file_sha256(norm_stats_path, name="norm stats"),
        "split_assignment_sha256": dataset_identity["split_assignment_sha256"],
        "train_storage_sha256": storage.get("sha256"),
    }
    if dict(dataset_identity) != expected_dataset:
        raise V35AuthorizationError("pilot authorization dataset/norm identity no longer matches production assets")
    if dataset_identity["episode_manifest_sha256"] != data_config.memory_episode_manifest_sha256:
        raise V35AuthorizationError("pilot authorization manifest hash differs from the registered config")
    if dataset_identity["norm_computation_protocol"] != "raw-train-rows-delta-action-horizon-v1":
        raise V35AuthorizationError("pilot authorization is not bound to the frozen raw-train norm protocol")

    calibration_path = pathlib.Path(config.model.memory_v35_calibration_path)
    calibration = _load_json(calibration_path, name="v3.5 calibration artifact")
    calibration_payload = calibration.get("payload")
    if not isinstance(calibration_payload, Mapping):
        raise V35AuthorizationError("v3.5 calibration artifact is missing its payload")
    calibration_provenance = calibration_payload.get("provenance")
    if not isinstance(calibration_provenance, Mapping):
        raise V35AuthorizationError("v3.5 calibration artifact is missing provenance")
    calibration_identity = _exact_mapping(
        "calibration_identity",
        payload["calibration_identity"],
        {
            "artifact_id",
            "collector_source_sha256",
            "dataset_protocol_sha256",
            "file_sha256",
            "preflight_sha256",
            "replay_protocol_sha256",
        },
    )
    expected_calibration = {
        "artifact_id": config.model.memory_v35_calibration_id,
        "collector_source_sha256": calibration_provenance.get("collector_source_sha256"),
        "dataset_protocol_sha256": calibration_provenance.get("dataset_sha256"),
        "file_sha256": _file_sha256(calibration_path, name="calibration artifact"),
        "preflight_sha256": calibration_provenance.get("preflight_sha256"),
        "replay_protocol_sha256": calibration_provenance.get("replay_protocol_sha256"),
    }
    if dict(calibration_identity) != expected_calibration:
        raise V35AuthorizationError("pilot authorization calibration/protocol identity changed")
    for key, value in calibration_identity.items():
        if key == "artifact_id":
            require_artifact_id(f"calibration_identity.{key}", value)
        else:
            require_sha256(f"calibration_identity.{key}", value)

    initialization = _exact_mapping(
        "initialization_identity",
        payload["initialization_identity"],
        {
            "initialization_identity_sha256",
            "initialization_manifest_file_sha256",
            "official_source_tree_sha256",
            "official_source_uri",
            "parameter_tree_sha256",
        },
    )
    if initialization["official_source_uri"] != OFFICIAL_BASE_URI:
        raise V35AuthorizationError("pilot authorization does not use the official Pi0.5 base URI")
    for key in (
        "initialization_identity_sha256",
        "initialization_manifest_file_sha256",
        "official_source_tree_sha256",
        "parameter_tree_sha256",
    ):
        require_sha256(f"initialization_identity.{key}", initialization[key])
    if getattr(config.weight_loader, "params_path", None) != OFFICIAL_BASE_URI:
        raise V35AuthorizationError("current v3.5 config does not use the official Pi0.5 base URI")

    run = _exact_mapping("run_identity", payload["run_identity"], {"run_id_sha256"})
    expected_run_id = run_id_sha256(
        config_name=config.name,
        experiment_name=config.exp_name,
        initialization_seed=config.seed,
        initialization_parameter_tree_sha256=initialization["parameter_tree_sha256"],
        calibration_artifact_id=calibration_identity["artifact_id"],
        semantic_config_sha256=semantic_sha256,
    )
    if run.get("run_id_sha256") != expected_run_id:
        raise V35AuthorizationError("pilot authorization run identity is invalid")
    protocols = _exact_mapping(
        "protocols",
        payload["protocols"],
        {
            "calibration_schema_version",
            "checkpoint_labels",
            "gate_a_criteria_version",
            "gate_b_criteria_version",
            "step0_criteria_version",
        },
    )
    if (
        protocols["calibration_schema_version"] != "openpi.v35.injection-calibration.v1"
        or protocols["checkpoint_labels"] != "completed_optimizer_updates"
    ):
        raise V35AuthorizationError("pilot authorization uses a non-production calibration/checkpoint protocol")
    evidence = _exact_mapping("gate_evidence", payload["gate_evidence"], {"gate_a", "gate_b", "step0"})
    for name, descriptor in evidence.items():
        _validate_evidence_descriptor(name, descriptor)
        if descriptor["criteria_version"] != protocols[f"{name}_criteria_version"]:
            raise V35AuthorizationError(f"gate_evidence.{name} criteria version mismatch")
    source = _validate_source_checkpoint_descriptor(
        "source_checkpoint",
        payload["source_checkpoint"],
        expected_completed_updates=0,
        expected_rung_descriptor=evidence["step0"],
    )
    if source["parameter_tree_sha256"] != initialization["parameter_tree_sha256"]:
        raise V35AuthorizationError("pilot source checkpoint parameters differ from authorized initialization")


def load_and_validate_pilot_authorization(config: Any) -> AuthorizationRecord:
    """Authenticate the static pilot permission against the current portable config/assets."""

    path = _configured_project_relative_path(config, "v35_pilot_authorization_path")
    record = load_authorization(path, expected_kind=PILOT_AUTHORIZATION_KIND)
    _validate_common_pilot_payload(config, record.payload)
    return record


def validate_pilot_run_binding(
    config: Any,
    authorization: AuthorizationRecord,
    *,
    initialization_identity: Mapping[str, Any],
    actual_parameter_tree_sha256: str | None,
) -> None:
    """Bind an initialized/restored TrainState to the exact prelaunch decision."""

    expected = authorization.payload["initialization_identity"]
    if actual_parameter_tree_sha256 is not None:
        require_sha256("actual_parameter_tree_sha256", actual_parameter_tree_sha256)
        if actual_parameter_tree_sha256 != expected["parameter_tree_sha256"]:
            raise V35AuthorizationError("actual v3.5 step-0 parameter tree differs from the authorized tree")
    expected_fields = {
        "actual_step0_parameter_tree_sha256": expected["parameter_tree_sha256"],
        "calibration_id": authorization.payload["calibration_identity"]["artifact_id"],
        "config_name": config.name,
        "experiment_name": config.exp_name,
        "identity_sha256": expected["initialization_identity_sha256"],
        "official_source_uri": OFFICIAL_BASE_URI,
        "run_id_sha256": authorization.payload["run_identity"]["run_id_sha256"],
        "semantic_training_config_sha256": authorization.payload["semantic_training_config_sha256"],
        "source_tree_sha256": expected["official_source_tree_sha256"],
        "step0_checkpoint": 0,
    }
    mismatches = {
        key: (initialization_identity.get(key), value)
        for key, value in expected_fields.items()
        if initialization_identity.get(key) != value
    }
    if mismatches:
        raise V35AuthorizationError(f"run initialization identity differs from pilot authorization: {mismatches}")


def validate_live_source_checkpoint_binding(
    authorization: AuthorizationRecord,
    *,
    completed_updates: int,
    parameter_tree_sha256: str,
    optimizer_state_sha256: str,
    runtime_identity_sha256: str,
    cumulative_telemetry_sha256: str,
    data_iterator_state_sha256: str,
) -> None:
    """Bind a restored live checkpoint to the authorization's externally sealed rung.

    This validator is intentionally usable only at the exact source rung.  A checkpoint's own
    metadata cannot authorize its mutable parameter or optimizer bytes, so later crash-resume
    checkpoints require a new externally sealed rung/hash rather than a self-asserted receipt.
    """

    if authorization.payload.get("authorization_kind") == PILOT_AUTHORIZATION_KIND:
        evidence = authorization.payload.get("gate_evidence")
        expected_rung = evidence.get("step0") if isinstance(evidence, Mapping) else None
    elif authorization.payload.get("authorization_kind") == CONTINUATION_AUTHORIZATION_KIND:
        expected_rung = None
    else:
        raise V35AuthorizationError("live checkpoint binding requires a pilot or continuation authorization")
    source = _validate_source_checkpoint_descriptor(
        "source_checkpoint",
        authorization.payload.get("source_checkpoint"),
        expected_completed_updates=completed_updates,
        expected_rung_descriptor=expected_rung,
    )
    actual = {
        "cumulative_telemetry_sha256": cumulative_telemetry_sha256,
        "data_iterator_state_sha256": data_iterator_state_sha256,
        "optimizer_state_sha256": optimizer_state_sha256,
        "parameter_tree_sha256": parameter_tree_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
    }
    for name, digest in actual.items():
        require_sha256(f"live_checkpoint.{name}", digest)
    mismatches = {
        name: {"actual": digest, "authorized": source[name]}
        for name, digest in actual.items()
        if source[name] != digest
    }
    if mismatches:
        raise V35AuthorizationError(f"live v3.5 source checkpoint differs from its sealed rung: {mismatches}")


def _validate_linked_authorization_descriptor(
    name: str,
    descriptor: Any,
    *,
    expected: AuthorizationRecord | None = None,
    expected_kind: str = PILOT_AUTHORIZATION_KIND,
) -> AuthorizationRecord:
    descriptor = _exact_mapping(name, descriptor, {"artifact_id", "path_relative", "sha256"})
    relative = descriptor["path_relative"]
    if not isinstance(relative, str) or not relative:
        raise V35AuthorizationError(f"{name}.path_relative must be non-empty")
    record = load_authorization(
        project_paths.project_path(relative),
        expected_kind=(expected.payload["authorization_kind"] if expected is not None else expected_kind),
    )
    if record.artifact_id != descriptor["artifact_id"] or record.file_sha256 != descriptor["sha256"]:
        raise V35AuthorizationError(f"{name} linked authorization identity changed")
    if expected is not None and record != expected:
        raise V35AuthorizationError(f"{name} does not reference the configured pilot authorization")
    return record


def load_and_validate_continuation_authorization(
    config: Any,
    *,
    pilot_authorization: AuthorizationRecord,
    latest_checkpoint_step: int,
) -> AuthorizationRecord:
    """Authenticate a Gate-D permission for the requested 2.5k/10k continuation."""

    path = _configured_project_relative_path(config, "v35_continuation_authorization_path")
    record = load_authorization(path, expected_kind=CONTINUATION_AUTHORIZATION_KIND)
    payload = _exact_mapping(
        "continuation authorization payload",
        record.payload,
        {
            "authorization_kind",
            "authorized_target_completed_updates",
            "calibration_identity",
            "config_identity",
            "final_test_remains_sealed",
            "gate_d",
            "initialization_identity",
            "pilot_authorization",
            "prior_continuation_authorization",
            "run_identity",
            "semantic_config_schema_version",
            "semantic_training_config_sha256",
            "source_checkpoint",
            "status",
        },
    )
    target = payload["authorized_target_completed_updates"]
    if target not in (2_500, 10_000) or target != config.num_train_steps:
        raise V35AuthorizationError("continuation authorization target differs from the requested frozen target")
    if payload["final_test_remains_sealed"] is not True:
        raise V35AuthorizationError("continuation authorization accessed the sealed final-test split")
    semantic_sha256 = semantic_training_config_sha256(config)
    if (
        payload["semantic_config_schema_version"] != SEMANTIC_CONFIG_SCHEMA_VERSION
        or payload["semantic_training_config_sha256"] != semantic_sha256
    ):
        raise V35AuthorizationError("continuation authorization is bound to a different semantic config")
    _validate_linked_authorization_descriptor(
        "pilot_authorization", payload["pilot_authorization"], expected=pilot_authorization
    )
    for name in ("config_identity", "run_identity", "initialization_identity", "calibration_identity"):
        if payload[name] != pilot_authorization.payload[name]:
            raise V35AuthorizationError(f"continuation {name} differs from the pilot authorization")

    gate_d = _exact_mapping(
        "gate_d",
        payload["gate_d"],
        {
            "action",
            "artifact_id",
            "criteria_version",
            "endpoint_completed_updates",
            "file_sha256",
            "outcome",
            "path_relative",
        },
    )
    endpoint = gate_d["endpoint_completed_updates"]
    if target == 2_500:
        expected_gate = (1_000, "inconclusive", "extend_same_run_once_to_2500")
    else:
        expected_gate = (endpoint, "pass", "continue_to_fixed_10000_budget")
        if endpoint not in (1_000, 2_500):
            raise V35AuthorizationError("10k continuation requires a passing 1k or 2.5k Gate-D endpoint")
    if (endpoint, gate_d["outcome"], gate_d["action"]) != expected_gate:
        raise V35AuthorizationError("Gate-D outcome/action does not authorize the requested continuation")
    require_artifact_id("gate_d.artifact_id", gate_d["artifact_id"])
    require_sha256("gate_d.file_sha256", gate_d["file_sha256"])
    decision_path = project_paths.project_path(gate_d["path_relative"])
    decision_bytes = decision_path.read_bytes()
    if sha256_bytes(decision_bytes) != gate_d["file_sha256"]:
        raise V35AuthorizationError("Gate-D decision file changed after authorization")
    try:
        decision = json.loads(decision_bytes)
    except json.JSONDecodeError as exc:
        raise V35AuthorizationError("Gate-D decision is invalid JSON") from exc
    if not isinstance(decision, Mapping) or decision.get("artifact_id") != gate_d["artifact_id"]:
        raise V35AuthorizationError("Gate-D decision artifact ID changed after authorization")

    source = _validate_source_checkpoint_descriptor(
        "source_checkpoint",
        payload["source_checkpoint"],
        expected_completed_updates=endpoint,
    )
    decision_payload = decision.get("payload")
    provenance = decision_payload.get("provenance") if isinstance(decision_payload, Mapping) else None
    rung_artifacts = provenance.get("rung_artifacts") if isinstance(provenance, Mapping) else None
    endpoint_evidence = None
    if isinstance(rung_artifacts, list):
        endpoint_evidence = next(
            (
                item
                for item in rung_artifacts
                if isinstance(item, Mapping) and item.get("completed_updates") == endpoint
            ),
            None,
        )
    if endpoint_evidence is None or any(
        endpoint_evidence.get(key) != source[source_key]
        for key, source_key in (
            ("artifact_id", "rung_artifact_id"),
            ("file_sha256", "rung_file_sha256"),
            ("checkpoint_parameter_tree_sha256", "parameter_tree_sha256"),
        )
    ):
        raise V35AuthorizationError("continuation source checkpoint differs from its Gate-D endpoint rung")
    if latest_checkpoint_step != endpoint:
        raise V35AuthorizationError(
            "latest checkpoint is not the authorization-linked source rung; intermediate crash resumes "
            "require a separately sealed external rung/hash and are forbidden by this launch contract"
        )

    prior = payload["prior_continuation_authorization"]
    if target == 10_000 and endpoint == 2_500:
        if prior is None:
            raise V35AuthorizationError("a 2.5k pass must retain the authenticated one-time extension authorization")
        prior_record = _validate_linked_authorization_descriptor(
            "prior_continuation_authorization",
            prior,
            expected_kind=CONTINUATION_AUTHORIZATION_KIND,
        )
        if (
            prior_record.payload.get("authorized_target_completed_updates") != 2_500
            or prior_record.payload.get("run_identity") != payload["run_identity"]
        ):
            raise V35AuthorizationError("prior one-time extension authorization belongs to another run")
    elif prior is not None:
        raise V35AuthorizationError("prior continuation authorization is allowed only after a 2.5k endpoint")
    return record


def validate_continuation_checkpoint_binding(
    authorization: AuthorizationRecord,
    *,
    latest_checkpoint_step: int,
    actual_parameter_tree_sha256: str,
    embedded_authorization_bytes: bytes | None,
) -> None:
    """Bind Gate D to the exact externally sealed source checkpoint.

    Merely embedding an authorization in a later checkpoint cannot authenticate that later
    checkpoint's independently mutable parameters and optimizer state.  Such a crash resume
    therefore needs its own sealed external rung and is not accepted here.
    """

    source = authorization.payload["source_checkpoint"]
    source_step = source["completed_updates"]
    if latest_checkpoint_step != source_step:
        raise V35AuthorizationError(
            "continuation may resume only its authorization-linked source rung; a later intermediate "
            "checkpoint requires separately sealed external rung evidence"
        )
    if embedded_authorization_bytes is not None and embedded_authorization_bytes != authorization.path.read_bytes():
        raise V35AuthorizationError("source checkpoint embeds a different continuation authorization")
    if actual_parameter_tree_sha256 != source["parameter_tree_sha256"]:
        raise V35AuthorizationError("Gate-D decision parameter hash differs from the restored source checkpoint")
