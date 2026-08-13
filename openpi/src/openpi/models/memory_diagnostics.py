"""Host-side utilities for inspecting, snapshotting, and forking fast memory state.

These helpers deliberately live outside :mod:`openpi.models.memory`: they do not
participate in the model PyTree, change ``MemoryState``, or alter the online write
rule.  They are intended for evaluation interventions and diagnostics, where a
single memory state must be copied into independent counterfactual branches.

Hashes and snapshots cover *both* ``fast_weights`` and ``momentum``.  Hashing is
bit-exact and canonical across NumPy and JAX arrays of the same dtype, shape, and
contents.  Snapshot metadata is restricted to JSON values so an NPZ never needs
pickle to load.
"""

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models.memory import MemoryState

SNAPSHOT_SCHEMA = "openpi.fast_memory_state"
SNAPSHOT_VERSION = 1
_HASH_ALGORITHM = "sha256"
_MANIFEST_KEY = "__manifest__"

ArrayBackend = Literal["preserve", "numpy", "jax"]


@dataclasses.dataclass(frozen=True)
class MemoryStateSnapshot:
    """An integrity-protected envelope around a cloned fast-memory state.

    The dataclass itself is frozen and ``metadata`` is a private JSON round-trip
    copy.  The state is cloned on creation and again on every restore, so neither
    the caller's source state nor any restored branch aliases it.  JAX arrays are
    immutable; direct mutation of a snapshot's NumPy leaves is detected by
    :func:`validate_memory_snapshot` through the stored digest.
    """

    state: MemoryState
    writes: int
    metadata: dict[str, Any]
    state_hash: str
    snapshot_hash: str
    schema: str = SNAPSHOT_SCHEMA
    version: int = SNAPSHOT_VERSION


def validate_memory_state(state: MemoryState, *, require_finite: bool = False) -> None:
    """Validate the structural invariants expected by the fast-memory writer.

    Each group must be a non-empty string-keyed mapping.  Momentum must have
    exactly one leaf for each fast-weight leaf with the same shape and dtype,
    and all leaves must share a non-empty leading batch dimension.
    """

    if not isinstance(state, MemoryState):
        raise TypeError(f"state must be MemoryState, got {type(state).__name__}")

    groups = {"fast_weights": state.fast_weights, "momentum": state.momentum}
    for group_name, group in groups.items():
        if not isinstance(group, Mapping) or not group:
            raise ValueError(f"MemoryState.{group_name} must be a non-empty mapping")
        invalid_keys = [key for key in group if not isinstance(key, str) or not key]
        if invalid_keys:
            raise ValueError(f"MemoryState.{group_name} contains non-string or empty keys: {invalid_keys!r}")

    fast_keys = set(state.fast_weights)
    momentum_keys = set(state.momentum)
    if fast_keys != momentum_keys:
        missing = sorted(fast_keys - momentum_keys)
        extra = sorted(momentum_keys - fast_keys)
        raise ValueError(f"momentum keys must match fast_weights; missing={missing}, extra={extra}")

    batch_size = None
    for name in sorted(fast_keys):
        fast = _host_array(state.fast_weights[name], path=f"fast_weights/{name}")
        momentum = _host_array(state.momentum[name], path=f"momentum/{name}")
        if fast.ndim < 1:
            raise ValueError(f"fast_weights/{name} must have a leading batch dimension")
        if fast.shape != momentum.shape:
            raise ValueError(f"shape mismatch for {name}: fast_weights={fast.shape}, momentum={momentum.shape}")
        if _canonical_dtype_name(fast.dtype) != _canonical_dtype_name(momentum.dtype):
            raise ValueError(f"dtype mismatch for {name}: fast_weights={fast.dtype}, momentum={momentum.dtype}")
        if _canonical_dtype_name(fast.dtype) != "float32":
            raise TypeError(f"fast-memory leaves must be float32; {name} has dtype {fast.dtype}")
        if batch_size is None:
            batch_size = fast.shape[0]
            if batch_size < 1:
                raise ValueError("fast-memory batch dimension must be non-empty")
        elif fast.shape[0] != batch_size:
            raise ValueError(
                f"inconsistent fast-memory batch dimension: expected {batch_size}, "
                f"fast_weights/{name} has {fast.shape[0]}"
            )
        if require_finite and (not np.all(np.isfinite(fast)) or not np.all(np.isfinite(momentum))):
            raise ValueError(f"non-finite values in fast-memory leaf {name}")


def clone_memory_state(state: MemoryState, *, backend: ArrayBackend = "preserve") -> MemoryState:
    """Return a complete, branch-safe copy of ``state``.

    ``backend='preserve'`` retains NumPy versus JAX on each leaf.  A JAX array
    may share an internal immutable device buffer, which is safe because JAX
    arrays cannot be mutated; NumPy leaves always receive distinct storage.
    """

    validate_memory_state(state)
    _validate_backend(backend)
    return MemoryState(
        fast_weights={name: _clone_array(value, backend) for name, value in state.fast_weights.items()},
        momentum={name: _clone_array(value, backend) for name, value in state.momentum.items()},
    )


def fork_memory_state(
    state: MemoryState, branch_count: int, *, backend: ArrayBackend = "preserve"
) -> tuple[MemoryState, ...]:
    """Clone ``state`` into mutually isolated counterfactual branches."""

    if isinstance(branch_count, bool) or not isinstance(branch_count, int) or branch_count < 1:
        raise ValueError(f"branch_count must be a positive integer, got {branch_count!r}")
    branches = tuple(clone_memory_state(state, backend=backend) for _ in range(branch_count))
    for left_index, left in enumerate(branches):
        for right in branches[left_index + 1 :]:
            assert_memory_states_isolated(left, right)
    return branches


def memory_state_hash(state: MemoryState) -> str:
    """Return a deterministic SHA-256 digest of the complete fast-memory state."""

    validate_memory_state(state)
    digest = hashlib.sha256()
    _hash_field(digest, "schema", SNAPSHOT_SCHEMA)
    _hash_field(digest, "state_hash_version", str(SNAPSHOT_VERSION))
    for group_name, group in (("fast_weights", state.fast_weights), ("momentum", state.momentum)):
        _hash_field(digest, "group", group_name)
        _hash_field(digest, "leaf_count", str(len(group)))
        for name in sorted(group):
            array = _canonical_host_array(group[name], path=f"{group_name}/{name}")
            _hash_field(digest, "name", name)
            _hash_field(digest, "dtype", _canonical_dtype_name(array.dtype))
            _hash_field(digest, "shape", json.dumps(array.shape, separators=(",", ":")))
            _hash_bytes(digest, array.tobytes(order="C"))
    return digest.hexdigest()


def memory_states_equal(left: MemoryState, right: MemoryState) -> bool:
    """Return whether two states are structurally and bit-for-bit identical."""

    try:
        return memory_state_hash(left) == memory_state_hash(right)
    except (TypeError, ValueError):
        return False


def assert_memory_states_equal(left: MemoryState, right: MemoryState) -> None:
    """Raise ``AssertionError`` with a useful first difference if states differ."""

    validate_memory_state(left)
    validate_memory_state(right)
    if memory_states_equal(left, right):
        return
    difference = _first_state_difference(left, right)
    raise AssertionError(f"fast-memory states differ: {difference}")


def memory_states_share_mutable_storage(left: MemoryState, right: MemoryState) -> bool:
    """Return whether any NumPy leaves across the two states overlap storage.

    JAX device arrays are immutable and therefore cannot create mutation leaks
    between counterfactual branches even if XLA reuses a backing buffer.
    """

    validate_memory_state(left)
    validate_memory_state(right)
    left_leaves = [*left.fast_weights.values(), *left.momentum.values()]
    right_leaves = [*right.fast_weights.values(), *right.momentum.values()]
    for left_leaf in left_leaves:
        if not isinstance(left_leaf, np.ndarray):
            continue
        for right_leaf in right_leaves:
            if isinstance(right_leaf, np.ndarray) and np.shares_memory(left_leaf, right_leaf):
                return True
    return False


def assert_memory_states_isolated(left: MemoryState, right: MemoryState) -> None:
    """Assert that no mutable array storage is shared between two states."""

    if memory_states_share_mutable_storage(left, right):
        raise AssertionError("fast-memory states share mutable NumPy storage")


def create_memory_snapshot(
    state: MemoryState,
    *,
    writes: int,
    metadata: Mapping[str, Any] | None = None,
    backend: ArrayBackend = "preserve",
) -> MemoryStateSnapshot:
    """Create an integrity-protected snapshot without retaining mutable aliases."""

    writes = _validate_writes(writes)
    metadata_copy = _canonical_metadata(metadata)
    state_copy = clone_memory_state(state, backend=backend)
    state_digest = memory_state_hash(state_copy)
    return MemoryStateSnapshot(
        state=state_copy,
        writes=writes,
        metadata=metadata_copy,
        state_hash=state_digest,
        snapshot_hash=_snapshot_hash(state_digest, writes, metadata_copy),
    )


def validate_memory_snapshot(snapshot: MemoryStateSnapshot, *, require_finite: bool = False) -> None:
    """Validate schema, state contents, and both snapshot integrity digests."""

    if not isinstance(snapshot, MemoryStateSnapshot):
        raise TypeError(f"snapshot must be MemoryStateSnapshot, got {type(snapshot).__name__}")
    if snapshot.schema != SNAPSHOT_SCHEMA or snapshot.version != SNAPSHOT_VERSION:
        raise ValueError(
            f"unsupported memory snapshot schema/version: {snapshot.schema!r} v{snapshot.version}; "
            f"expected {SNAPSHOT_SCHEMA!r} v{SNAPSHOT_VERSION}"
        )
    writes = _validate_writes(snapshot.writes)
    metadata = _canonical_metadata(snapshot.metadata)
    validate_memory_state(snapshot.state, require_finite=require_finite)
    actual_state_hash = memory_state_hash(snapshot.state)
    if snapshot.state_hash != actual_state_hash:
        raise ValueError(
            f"memory snapshot state hash mismatch: stored={snapshot.state_hash}, actual={actual_state_hash}"
        )
    actual_snapshot_hash = _snapshot_hash(actual_state_hash, writes, metadata)
    if snapshot.snapshot_hash != actual_snapshot_hash:
        raise ValueError(
            "memory snapshot envelope hash mismatch "
            f"(writes or metadata changed): stored={snapshot.snapshot_hash}, actual={actual_snapshot_hash}"
        )


def restore_memory_snapshot(
    snapshot: MemoryStateSnapshot, *, backend: ArrayBackend = "preserve"
) -> tuple[MemoryState, int, dict[str, Any]]:
    """Restore an independent state plus its write counter and metadata copy."""

    _validate_backend(backend)
    validate_memory_snapshot(snapshot)
    return (
        clone_memory_state(snapshot.state, backend=backend),
        snapshot.writes,
        _canonical_metadata(snapshot.metadata),
    )


def fork_memory_snapshot(
    snapshot: MemoryStateSnapshot, branch_count: int, *, backend: ArrayBackend = "preserve"
) -> tuple[tuple[MemoryState, int, dict[str, Any]], ...]:
    """Restore a snapshot into mutually isolated counterfactual branches."""

    validate_memory_snapshot(snapshot)
    if isinstance(branch_count, bool) or not isinstance(branch_count, int) or branch_count < 1:
        raise ValueError(f"branch_count must be a positive integer, got {branch_count!r}")
    branches = tuple(restore_memory_snapshot(snapshot, backend=backend) for _ in range(branch_count))
    states = tuple(branch[0] for branch in branches)
    for left_index, left in enumerate(states):
        for right in states[left_index + 1 :]:
            assert_memory_states_isolated(left, right)
    return branches


def save_memory_snapshot(
    path: str | os.PathLike[str], snapshot: MemoryStateSnapshot, *, overwrite: bool = False
) -> pathlib.Path:
    """Atomically save ``snapshot`` as a pickle-free, manifest-backed NPZ file."""

    validate_memory_snapshot(snapshot)
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"memory snapshot already exists: {target}")

    arrays: dict[str, np.ndarray] = {}
    leaf_manifest = []
    array_index = 0
    for group_name, group in (("fast_weights", snapshot.state.fast_weights), ("momentum", snapshot.state.momentum)):
        for name in sorted(group):
            npz_key = f"array_{array_index:06d}"
            array = _canonical_host_array(group[name], path=f"{group_name}/{name}")
            arrays[npz_key] = array
            leaf_manifest.append(
                {
                    "group": group_name,
                    "name": name,
                    "npz_key": npz_key,
                    "dtype": _canonical_dtype_name(array.dtype),
                    "shape": list(array.shape),
                }
            )
            array_index += 1

    manifest = {
        "schema": snapshot.schema,
        "version": snapshot.version,
        "hash_algorithm": _HASH_ALGORITHM,
        "state_hash": snapshot.state_hash,
        "snapshot_hash": snapshot.snapshot_hash,
        "writes": snapshot.writes,
        "metadata": snapshot.metadata,
        "leaves": leaf_manifest,
    }
    manifest_bytes = _canonical_json_bytes(manifest)
    arrays[_MANIFEST_KEY] = np.frombuffer(manifest_bytes, dtype=np.uint8).copy()

    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as temp:
            temporary_name = temp.name
            np.savez_compressed(temp, **arrays)
            temp.flush()
            os.fsync(temp.fileno())
        if overwrite:
            os.replace(temporary_name, target)
            temporary_name = None
        else:
            # Hard-link publication is atomic and refuses to replace a file that
            # appeared after the preflight existence check.
            os.link(temporary_name, target)
        return target
    finally:
        if temporary_name is not None:
            pathlib.Path(temporary_name).unlink(missing_ok=True)


def load_memory_snapshot(
    path: str | os.PathLike[str], *, backend: Literal["numpy", "jax"] = "numpy"
) -> MemoryStateSnapshot:
    """Load and integrity-check a snapshot, returning independent array storage."""

    if backend not in ("numpy", "jax"):
        raise ValueError(f"load backend must be 'numpy' or 'jax', got {backend!r}")
    source = pathlib.Path(path)
    with np.load(source, allow_pickle=False) as archive:
        if len(archive.files) != len(set(archive.files)):
            raise ValueError("memory snapshot contains duplicate NPZ array names")
        if _MANIFEST_KEY not in archive.files:
            raise ValueError(f"memory snapshot is missing {_MANIFEST_KEY!r}")
        manifest_array = archive[_MANIFEST_KEY]
        if manifest_array.dtype != np.uint8 or manifest_array.ndim != 1:
            raise ValueError("memory snapshot manifest must be a one-dimensional uint8 array")
        try:
            manifest = json.loads(manifest_array.tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("memory snapshot manifest is not valid UTF-8 JSON") from error
        _validate_manifest(manifest)

        expected_npz_keys = {_MANIFEST_KEY}
        groups: dict[str, dict[str, Any]] = {"fast_weights": {}, "momentum": {}}
        seen_paths = set()
        for leaf in manifest["leaves"]:
            _validate_leaf_manifest(leaf)
            group_name = leaf["group"]
            name = leaf["name"]
            npz_key = leaf["npz_key"]
            path_key = (group_name, name)
            if path_key in seen_paths:
                raise ValueError(f"duplicate memory snapshot leaf: {group_name}/{name}")
            if npz_key in expected_npz_keys:
                raise ValueError(f"duplicate or reserved NPZ key in manifest: {npz_key}")
            seen_paths.add(path_key)
            expected_npz_keys.add(npz_key)
            if npz_key not in archive.files:
                raise ValueError(f"memory snapshot is missing declared array {npz_key!r}")
            array = np.array(archive[npz_key], copy=True, order="C")
            if list(array.shape) != leaf["shape"]:
                raise ValueError(
                    f"shape mismatch for {group_name}/{name}: manifest={leaf['shape']}, array={list(array.shape)}"
                )
            if _canonical_dtype_name(array.dtype) != leaf["dtype"]:
                raise ValueError(
                    f"dtype mismatch for {group_name}/{name}: manifest={leaf['dtype']}, array={array.dtype}"
                )
            groups[group_name][name] = jnp.asarray(array) if backend == "jax" else array

        extra_keys = set(archive.files) - expected_npz_keys
        if extra_keys:
            raise ValueError(f"memory snapshot contains undeclared arrays: {sorted(extra_keys)}")

    snapshot = MemoryStateSnapshot(
        state=MemoryState(fast_weights=groups["fast_weights"], momentum=groups["momentum"]),
        writes=manifest["writes"],
        metadata=_canonical_metadata(manifest["metadata"]),
        state_hash=manifest["state_hash"],
        snapshot_hash=manifest["snapshot_hash"],
        schema=manifest["schema"],
        version=manifest["version"],
    )
    validate_memory_snapshot(snapshot)
    return snapshot


def _host_array(value: Any, *, path: str) -> np.ndarray:
    if not isinstance(value, np.ndarray | jax.Array):
        raise TypeError(f"{path} must be a NumPy or JAX array, got {type(value).__name__}")
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError(f"{path} cannot have object dtype")
    return array


def _canonical_host_array(value: Any, *, path: str) -> np.ndarray:
    array = _host_array(value, path=path)
    dtype = array.dtype.newbyteorder("<")
    return np.ascontiguousarray(array.astype(dtype, copy=False))


def _canonical_dtype_name(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    return dtype.name


def _clone_array(value: Any, backend: ArrayBackend):
    host_copy = np.array(_host_array(value, path="state leaf"), copy=True, order="C")
    if backend == "numpy" or (backend == "preserve" and isinstance(value, np.ndarray)):
        return host_copy
    if backend == "jax" or (backend == "preserve" and isinstance(value, jax.Array)):
        return jnp.asarray(host_copy)
    raise AssertionError(f"unreachable backend: {backend}")


def _validate_backend(backend: ArrayBackend) -> None:
    if backend not in ("preserve", "numpy", "jax"):
        raise ValueError(f"backend must be 'preserve', 'numpy', or 'jax', got {backend!r}")


def _validate_writes(writes: int) -> int:
    if isinstance(writes, bool) or not isinstance(writes, int | np.integer) or writes < 0:
        raise ValueError(f"writes must be a non-negative integer, got {writes!r}")
    return int(writes)


def _canonical_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError(f"metadata must be a mapping, got {type(metadata).__name__}")
    if any(not isinstance(key, str) for key in metadata):
        raise TypeError("metadata keys must be strings")
    try:
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError("metadata must contain only finite, JSON-serializable values") from error
    if not isinstance(decoded, dict):
        raise TypeError("metadata must encode to a JSON object")
    return decoded


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _snapshot_hash(state_hash: str, writes: int, metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    _hash_field(digest, "schema", SNAPSHOT_SCHEMA)
    _hash_field(digest, "version", str(SNAPSHOT_VERSION))
    _hash_field(digest, "state_hash", state_hash)
    _hash_field(digest, "writes", str(writes))
    _hash_bytes(digest, _canonical_json_bytes(metadata))
    return digest.hexdigest()


def _hash_field(digest: Any, label: str, value: str) -> None:
    _hash_bytes(digest, label.encode("utf-8"))
    _hash_bytes(digest, value.encode("utf-8"))


def _hash_bytes(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _first_state_difference(left: MemoryState, right: MemoryState) -> str:
    for group_name in ("fast_weights", "momentum"):
        left_group = getattr(left, group_name)
        right_group = getattr(right, group_name)
        if set(left_group) != set(right_group):
            return f"{group_name} keys differ: {sorted(left_group)} != {sorted(right_group)}"
        for name in sorted(left_group):
            left_array = _host_array(left_group[name], path=f"{group_name}/{name}")
            right_array = _host_array(right_group[name], path=f"{group_name}/{name}")
            if left_array.shape != right_array.shape:
                return f"{group_name}/{name} shape differs: {left_array.shape} != {right_array.shape}"
            if _canonical_dtype_name(left_array.dtype) != _canonical_dtype_name(right_array.dtype):
                return f"{group_name}/{name} dtype differs: {left_array.dtype} != {right_array.dtype}"
            left_bytes = _canonical_host_array(left_array, path=f"{group_name}/{name}").tobytes(order="C")
            right_bytes = _canonical_host_array(right_array, path=f"{group_name}/{name}").tobytes(order="C")
            if left_bytes != right_bytes:
                return f"{group_name}/{name} contents differ"
    return "canonical hashes differ"


def _validate_manifest(manifest: Any) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("memory snapshot manifest must be a JSON object")
    required = {
        "schema",
        "version",
        "hash_algorithm",
        "state_hash",
        "snapshot_hash",
        "writes",
        "metadata",
        "leaves",
    }
    missing = required - manifest.keys()
    extra = manifest.keys() - required
    if missing or extra:
        raise ValueError(f"memory snapshot manifest fields mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    if manifest["schema"] != SNAPSHOT_SCHEMA or manifest["version"] != SNAPSHOT_VERSION:
        raise ValueError(f"unsupported memory snapshot schema/version: {manifest['schema']!r} v{manifest['version']}")
    if manifest["hash_algorithm"] != _HASH_ALGORITHM:
        raise ValueError(f"unsupported memory snapshot hash algorithm: {manifest['hash_algorithm']!r}")
    _validate_writes(manifest["writes"])
    _canonical_metadata(manifest["metadata"])
    if not isinstance(manifest["leaves"], list) or not manifest["leaves"]:
        raise ValueError("memory snapshot manifest must declare at least one array leaf")
    for hash_name in ("state_hash", "snapshot_hash"):
        value = manifest[hash_name]
        if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"manifest {hash_name} must be a lowercase SHA-256 hex digest")


def _validate_leaf_manifest(leaf: Any) -> None:
    required = {"group", "name", "npz_key", "dtype", "shape"}
    if not isinstance(leaf, dict) or set(leaf) != required:
        raise ValueError(f"invalid memory snapshot leaf manifest: {leaf!r}")
    if leaf["group"] not in ("fast_weights", "momentum"):
        raise ValueError(f"invalid memory snapshot leaf group: {leaf['group']!r}")
    if not isinstance(leaf["name"], str) or not leaf["name"]:
        raise ValueError("memory snapshot leaf name must be a non-empty string")
    if not isinstance(leaf["npz_key"], str) or not leaf["npz_key"].startswith("array_"):
        raise ValueError(f"invalid memory snapshot NPZ key: {leaf['npz_key']!r}")
    if not isinstance(leaf["dtype"], str) or not leaf["dtype"]:
        raise ValueError("memory snapshot leaf dtype must be a non-empty string")
    if (
        not isinstance(leaf["shape"], list)
        or not leaf["shape"]
        or any(isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in leaf["shape"])
    ):
        raise ValueError(f"invalid memory snapshot leaf shape: {leaf['shape']!r}")
