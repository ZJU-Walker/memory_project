from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import json
import logging
import os
import pathlib
import re
import tempfile
from typing import Any, Protocol, runtime_checkable

import flax.traverse_util
import jax
import numpy as np
import orbax.checkpoint as ocp

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PartialCheckpointWeightLoader(WeightLoader):
    """Loads a checkpoint into a model that has extra, newly added parameters.

    Everything present in the checkpoint is loaded 1:1; every parameter the checkpoint does not
    contain (e.g. the Titans memory subsystem of `predict_with_memory` models when starting from
    a pre-memory checkpoint) keeps its fresh initialization.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class AuditedPartialCheckpointWeightLoader(PartialCheckpointWeightLoader):
    """Strict official-params initialization for models with explicitly fresh leaves.

    Unlike the legacy permissive partial loader, every source/target leaf must belong to one
    allowlisted outcome and shared leaves must match shape and dtype exactly. This loader reads
    the ordinary standalone ``params`` artifact; it never restores optimizer, EMA, or raw
    train-state data.
    """

    matched_allowlist: tuple[str, ...] = ()
    fresh_init_allowlist: tuple[str, ...] = ()
    ignored_source_allowlist: tuple[str, ...] = ()
    manifest_output_path: str | None = None

    def load(self, params: at.Params) -> at.Params:
        return self.load_with_manifest(params).params

    def load_with_manifest(self, params: at.Params) -> AuditedGraftResult:
        _validate_allowlist_configuration(self)
        if self.manifest_output_path is None:
            raise AuditedGraftError("audited partial loading requires manifest_output_path")
        restored_path = download.maybe_download(self.params_path)
        source_params = _model.restore_params(restored_path, restore_type=np.ndarray)
        result = _audit_and_graft(
            source_params,
            params,
            self,
            pathlib.Path(os.fspath(restored_path)),
            checkpoint_root_label=self.params_path,
            parameter_source=f"standalone params artifact: {self.params_path}",
            standalone_params_present=True,
        )
        _write_manifest(pathlib.Path(self.manifest_output_path), result.manifest)
        logger.info(
            "Audited partial initialization complete: matched=%d fresh=%d ignored=%d source_sha256=%s "
            "graft_sha256=%s",
            len(result.manifest.matched),
            len(result.manifest.fresh_initialized),
            len(result.manifest.ignored_source),
            result.manifest.source_tree_sha256,
            result.manifest.graft_tree_sha256,
        )
        return result


class AuditedGraftError(ValueError):
    """Raised when an audited graft cannot prove that every parameter leaf was handled explicitly."""


@dataclasses.dataclass(frozen=True)
class GraftLeafRecord:
    """JSON-safe provenance for one source or target parameter leaf."""

    path: str
    action: str
    shape: tuple[int, ...]
    dtype: str
    value_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "action": self.action,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "value_sha256": self.value_sha256,
        }


@dataclasses.dataclass(frozen=True)
class ConditionalResetRecord:
    """The fully recorded decision for the optional memory-injection-gate reset."""

    path: str
    closed_abs_gate_lt: float
    closed_fraction_threshold: float
    observed_closed_fraction: float
    reset_gate_value: float
    reset_parameter_value: float
    applied: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class GraftManifest:
    """Stable, serializable description of an audited raw-parameter graft.

    ``source_tree_sha256`` hashes every raw parameter value in ``train_state`` (including
    explicitly ignored source leaves). ``target_schema_sha256`` hashes target paths, shapes, and
    dtypes. ``graft_tree_sha256`` hashes grafted/reset values and records fresh-init leaves by
    schema and action, because their eventual random values do not exist when training passes a
    ``jax.ShapeDtypeStruct`` target tree to the loader.
    """

    format_version: int
    loader: str
    checkpoint_root: str
    parameter_source: str
    standalone_params_present: bool
    matched_allowlist: tuple[str, ...]
    fresh_init_allowlist: tuple[str, ...]
    ignored_source_allowlist: tuple[str, ...]
    source_tree_sha256: str
    target_schema_sha256: str
    graft_tree_sha256: str
    matched: tuple[GraftLeafRecord, ...]
    fresh_initialized: tuple[GraftLeafRecord, ...]
    ignored_source: tuple[GraftLeafRecord, ...]
    reset: tuple[GraftLeafRecord, ...]
    conditional_reset: ConditionalResetRecord | None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "format_version": self.format_version,
            "loader": self.loader,
            "checkpoint_root": self.checkpoint_root,
            "parameter_source": self.parameter_source,
            "standalone_params_present": self.standalone_params_present,
            "allowlists": {
                "matched": list(self.matched_allowlist),
                "fresh_init": list(self.fresh_init_allowlist),
                "ignored_source": list(self.ignored_source_allowlist),
            },
            "tree_hashes": {
                "source_sha256": self.source_tree_sha256,
                "target_schema_sha256": self.target_schema_sha256,
                "graft_sha256": self.graft_tree_sha256,
            },
            "counts": {
                "matched": len(self.matched),
                "fresh_initialized": len(self.fresh_initialized),
                "ignored_source": len(self.ignored_source),
                "reset": len(self.reset),
            },
            "leaves": {
                "matched": [leaf.to_dict() for leaf in self.matched],
                "fresh_initialized": [leaf.to_dict() for leaf in self.fresh_initialized],
                "ignored_source": [leaf.to_dict() for leaf in self.ignored_source],
                "reset": [leaf.to_dict() for leaf in self.reset],
            },
            "conditional_reset": None if self.conditional_reset is None else self.conditional_reset.to_dict(),
        }
        payload["manifest_sha256"] = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
        return payload


@dataclasses.dataclass(frozen=True)
class AuditedGraftResult:
    params: at.Params
    manifest: GraftManifest


@dataclasses.dataclass(frozen=True)
class AuditedRawCheckpointWeightLoader(WeightLoader):
    """Fail-closed graft of raw model parameters from an OpenPI Orbax training checkpoint.

    This loader is intentionally disabled by default. ``checkpoint_path`` must name a checkpoint
    *step root* containing ``train_state/``; passing the sibling standalone ``params/`` item is
    rejected. The loader restores only ``train_state/params/*/value`` through explicit Orbax
    transforms, so it cannot silently select the standalone EMA parameters written by
    ``training.checkpoints.save_state``.

    Every leaf must belong to exactly one structural outcome and be admitted by the corresponding
    full-match regular-expression allowlist:

    * source and target: ``matched_allowlist`` and strict shape/dtype equality;
    * target only: ``fresh_init_allowlist``;
    * source only: ``ignored_source_allowlist``.

    Patterns that match no leaf are rejected to catch stale or misspelled provenance rules. The
    existing ``PartialCheckpointWeightLoader`` remains the permissive legacy API and is not used by
    this class.
    """

    checkpoint_path: str
    enabled: bool = False
    matched_allowlist: tuple[str, ...] = ()
    fresh_init_allowlist: tuple[str, ...] = ()
    ignored_source_allowlist: tuple[str, ...] = ()
    manifest_output_path: str | None = None
    memory_inject_w_path: str | None = None
    reset_memory_inject_w_if_closed_fraction_gt: float | None = None
    memory_inject_w_closed_abs_gate_lt: float = 0.1
    memory_inject_w_reset_gate_value: float = 0.5

    def load(self, params: at.Params) -> at.Params:
        return self.load_with_manifest(params).params

    def load_with_manifest(self, params: at.Params) -> AuditedGraftResult:
        if not self.enabled:
            raise AuditedGraftError(
                "AuditedRawCheckpointWeightLoader is default-off; set enabled=True only after freezing allowlists"
            )
        _validate_loader_configuration(self)
        checkpoint_root = _resolve_checkpoint_root(self.checkpoint_path)
        source_params = _restore_raw_train_state_params(checkpoint_root / "train_state")
        result = _audit_and_graft(source_params, params, self, checkpoint_root)
        if self.manifest_output_path is not None:
            _write_manifest(pathlib.Path(self.manifest_output_path), result.manifest)
        logger.info(
            "Audited raw graft complete: matched=%d fresh=%d ignored=%d reset=%d source_sha256=%s graft_sha256=%s",
            len(result.manifest.matched),
            len(result.manifest.fresh_initialized),
            len(result.manifest.ignored_source),
            len(result.manifest.reset),
            result.manifest.source_tree_sha256,
            result.manifest.graft_tree_sha256,
        )
        return result


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _validate_allowlist_configuration(loader: Any) -> None:
    for name, patterns in (
        ("matched_allowlist", loader.matched_allowlist),
        ("fresh_init_allowlist", loader.fresh_init_allowlist),
        ("ignored_source_allowlist", loader.ignored_source_allowlist),
    ):
        for pattern in patterns:
            if not pattern:
                raise AuditedGraftError(f"{name} contains an empty regex")
            try:
                re.compile(pattern)
            except re.error as exc:
                raise AuditedGraftError(f"invalid regex in {name}: {pattern!r}: {exc}") from exc


def _validate_loader_configuration(loader: AuditedRawCheckpointWeightLoader) -> None:
    _validate_allowlist_configuration(loader)

    reset_fields = (
        loader.memory_inject_w_path,
        loader.reset_memory_inject_w_if_closed_fraction_gt,
    )
    if any(field is None for field in reset_fields) and any(field is not None for field in reset_fields):
        raise AuditedGraftError(
            "memory_inject_w_path and reset_memory_inject_w_if_closed_fraction_gt must be configured together"
        )
    if loader.reset_memory_inject_w_if_closed_fraction_gt is not None and not (
        0.0 <= loader.reset_memory_inject_w_if_closed_fraction_gt <= 1.0
    ):
        raise AuditedGraftError("closed-fraction reset threshold must be in [0, 1]")
    if not (0.0 < loader.memory_inject_w_closed_abs_gate_lt < 1.0):
        raise AuditedGraftError("memory_inject_w_closed_abs_gate_lt must be in (0, 1)")
    if not (-1.0 < loader.memory_inject_w_reset_gate_value < 1.0):
        raise AuditedGraftError("memory_inject_w_reset_gate_value must be in (-1, 1)")


def _resolve_checkpoint_root(checkpoint_path: str) -> pathlib.Path:
    downloaded = download.maybe_download(checkpoint_path)
    root = pathlib.Path(os.fspath(downloaded)).resolve()
    if root.name in {"params", "train_state"}:
        raise AuditedGraftError(f"checkpoint_path must be the checkpoint step root, not its {root.name!r} item: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint step root does not exist: {root}")
    raw_item = root / "train_state"
    if not raw_item.is_dir():
        raise AuditedGraftError(f"checkpoint step root has no Orbax train_state item: {raw_item}")
    return root


def _restore_raw_train_state_params(train_state_path: pathlib.Path) -> at.Params:
    """Restore all and only ``train_state/params/*/value`` leaves as NumPy arrays."""
    with ocp.PyTreeCheckpointer() as checkpointer:
        metadata = checkpointer.metadata(train_state_path)
        # Orbax 0.11 returns a ``TreeMetadata`` wrapper. It intentionally behaves like its
        # underlying tree but does not inherit ``collections.abc.Mapping``; use the documented
        # ``tree`` property before validating the serialized train-state structure. Keeping the
        # Mapping fallback also supports older Orbax releases.
        metadata_tree = getattr(metadata, "tree", metadata)
        if not isinstance(metadata_tree, Mapping) or "params" not in metadata_tree:
            raise AuditedGraftError(f"Orbax train_state metadata has no raw params subtree: {train_state_path}")

        flat_metadata = flax.traverse_util.flatten_dict(metadata_tree["params"])
        if not flat_metadata:
            raise AuditedGraftError(f"Orbax train_state raw params subtree is empty: {train_state_path}")
        malformed = [_path_string(path) for path in flat_metadata if not path or path[-1] != "value"]
        if malformed:
            raise AuditedGraftError(
                "raw train_state parameter metadata must end in '/value'; malformed leaves: " + _format_paths(malformed)
            )

        pure_metadata = flax.traverse_util.unflatten_dict(
            {path[:-1]: leaf_metadata for path, leaf_metadata in flat_metadata.items()}
        )
        item = {"params": pure_metadata}
        restore_args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item)
        restored = checkpointer.restore(
            train_state_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=restore_args,
                transforms={r"params/(.*)": ocp.RestoreTransform(original_key=r"params/\1/value")},
                transforms_default_to_original=False,
            ),
        )

    if not isinstance(restored, Mapping) or set(restored) != {"params"}:
        raise AuditedGraftError("explicit raw-parameter Orbax restore returned an unexpected tree")
    expected_paths = set(flax.traverse_util.flatten_dict(pure_metadata))
    restored_paths = set(flax.traverse_util.flatten_dict(restored["params"]))
    if restored_paths != expected_paths:
        raise AuditedGraftError(
            "explicit raw-parameter Orbax restore changed leaf paths: "
            f"missing={_format_paths(_display_paths(expected_paths - restored_paths))}; "
            f"extra={_format_paths(_display_paths(restored_paths - expected_paths))}"
        )
    return restored["params"]


def _audit_and_graft(
    source_params: at.Params,
    target_params: at.Params,
    loader: Any,
    checkpoint_root: pathlib.Path,
    *,
    checkpoint_root_label: str | None = None,
    parameter_source: str = "train_state/params/*/value (raw; standalone params item never restored)",
    standalone_params_present: bool | None = None,
) -> AuditedGraftResult:
    flat_source = flax.traverse_util.flatten_dict(source_params)
    flat_target = flax.traverse_util.flatten_dict(target_params)
    _validate_flat_paths(flat_source, tree_name="source")
    _validate_flat_paths(flat_target, tree_name="target")

    source_paths = set(flat_source)
    target_paths = set(flat_target)
    matched_paths = source_paths & target_paths
    fresh_paths = target_paths - source_paths
    ignored_paths = source_paths - target_paths

    _require_allowlisted("matched", matched_paths, loader.matched_allowlist)
    _require_allowlisted("fresh-init", fresh_paths, loader.fresh_init_allowlist)
    _require_allowlisted("ignored-source", ignored_paths, loader.ignored_source_allowlist)

    for path in sorted(matched_paths, key=_path_string):
        source_shape, source_dtype = _leaf_spec(flat_source[path])
        target_shape, target_dtype = _leaf_spec(flat_target[path])
        if source_shape != target_shape:
            raise AuditedGraftError(
                f"shape mismatch for {_path_string(path)!r}: source={source_shape}, target={target_shape}"
            )
        if source_dtype != target_dtype:
            raise AuditedGraftError(
                f"dtype mismatch for {_path_string(path)!r}: source={source_dtype}, target={target_dtype}; "
                "audited grafts never cast silently"
            )

    reset_path: tuple[str, ...] | None = None
    reset_value: np.ndarray | None = None
    reset_record: ConditionalResetRecord | None = None
    memory_inject_w_path = getattr(loader, "memory_inject_w_path", None)
    if memory_inject_w_path is not None:
        reset_path = _parse_exact_path(memory_inject_w_path)
        if reset_path not in matched_paths:
            raise AuditedGraftError(
                "configured memory_inject_w_path must identify exactly one source-and-target leaf: "
                f"{memory_inject_w_path!r}"
            )
        source_gate = _as_parameter_array(flat_source[reset_path], path=reset_path)
        gate = np.tanh(source_gate.astype(np.float64))
        closed_abs_gate_lt = loader.memory_inject_w_closed_abs_gate_lt
        reset_threshold = loader.reset_memory_inject_w_if_closed_fraction_gt
        reset_gate_value = loader.memory_inject_w_reset_gate_value
        closed_fraction = float(np.mean(np.abs(gate) < closed_abs_gate_lt))
        applied = closed_fraction > reset_threshold
        reset_parameter_value = float(np.arctanh(reset_gate_value))
        if applied:
            reset_value = np.full(source_gate.shape, reset_parameter_value, dtype=source_gate.dtype)
        reset_record = ConditionalResetRecord(
            path=memory_inject_w_path,
            closed_abs_gate_lt=closed_abs_gate_lt,
            closed_fraction_threshold=reset_threshold,
            observed_closed_fraction=closed_fraction,
            reset_gate_value=reset_gate_value,
            reset_parameter_value=reset_parameter_value,
            applied=applied,
        )

    flat_result: dict[tuple[str, ...], Any] = {}
    for path in target_paths:
        if path in fresh_paths:
            flat_result[path] = flat_target[path]
        elif path == reset_path and reset_value is not None:
            flat_result[path] = reset_value
        else:
            flat_result[path] = flat_source[path]

    effective_reset_paths = set() if reset_value is None else {reset_path}
    ordinary_matched_paths = matched_paths - effective_reset_paths
    matched_records = tuple(
        _leaf_record(path, "grafted_raw", flat_source[path])
        for path in sorted(ordinary_matched_paths, key=_path_string)
    )
    fresh_records = tuple(
        _leaf_record(path, "fresh_init", flat_target[path], include_value=False)
        for path in sorted(fresh_paths, key=_path_string)
    )
    ignored_records = tuple(
        _leaf_record(path, "ignored_source", flat_source[path]) for path in sorted(ignored_paths, key=_path_string)
    )
    reset_records = (
        ()
        if reset_value is None or reset_path is None
        else (_leaf_record(reset_path, "conditional_reset", reset_value),)
    )

    manifest = GraftManifest(
        format_version=1,
        loader=type(loader).__name__,
        checkpoint_root=checkpoint_root_label or str(checkpoint_root),
        parameter_source=parameter_source,
        standalone_params_present=(
            (checkpoint_root / "params").is_dir()
            if standalone_params_present is None
            else standalone_params_present
        ),
        matched_allowlist=loader.matched_allowlist,
        fresh_init_allowlist=loader.fresh_init_allowlist,
        ignored_source_allowlist=loader.ignored_source_allowlist,
        source_tree_sha256=_value_tree_hash(flat_source),
        target_schema_sha256=_schema_tree_hash(flat_target),
        graft_tree_sha256=_graft_tree_hash(flat_result, fresh_paths),
        matched=matched_records,
        fresh_initialized=fresh_records,
        ignored_source=ignored_records,
        reset=reset_records,
        conditional_reset=reset_record,
    )
    return AuditedGraftResult(flax.traverse_util.unflatten_dict(flat_result), manifest)


def _require_allowlisted(category: str, paths: set[tuple[str, ...]], regexes: tuple[str, ...]) -> None:
    compiled = tuple(re.compile(regex) for regex in regexes)
    unexpected = [path for path in paths if not any(pattern.fullmatch(_path_string(path)) for pattern in compiled)]
    if unexpected:
        raise AuditedGraftError(f"unexpected {category} leaves: {_format_paths(_display_paths(unexpected))}")

    unused = [
        regex
        for regex, pattern in zip(regexes, compiled, strict=True)
        if not any(pattern.fullmatch(_path_string(path)) for path in paths)
    ]
    if unused:
        raise AuditedGraftError(f"{category} allowlist patterns matched no leaves: {unused!r}")


def _parse_exact_path(path: str) -> tuple[str, ...]:
    pieces = tuple(path.split("/"))
    if not pieces or any(not piece for piece in pieces):
        raise AuditedGraftError(f"invalid exact parameter path: {path!r}")
    return pieces


def _validate_flat_paths(flat_tree: Mapping[tuple[str, ...], Any], *, tree_name: str) -> None:
    for path in flat_tree:
        if not path or any(not isinstance(piece, str) or not piece or "/" in piece for piece in path):
            raise AuditedGraftError(f"{tree_name} has a path that cannot be represented canonically: {path!r}")


def _path_string(path: tuple[str, ...]) -> str:
    return "/".join(path)


def _display_paths(paths) -> list[str]:
    return sorted(_path_string(path) for path in paths)


def _format_paths(paths: list[str], *, limit: int = 12) -> str:
    if not paths:
        return "[]"
    displayed = paths[:limit]
    suffix = "" if len(paths) <= limit else f", ... (+{len(paths) - limit})"
    return "[" + ", ".join(repr(path) for path in displayed) + suffix + "]"


def _leaf_spec(value: Any) -> tuple[tuple[int, ...], str]:
    # NNX preserves ``bias=None`` for bias-free Linear modules in the parameter-state
    # PyTree.  It is a structural sentinel, not a trainable array, but it must remain in the
    # partial tree so ``replace_by_pure_dict`` sees the exact target structure.  Give it an
    # explicit schema identity rather than silently dropping it from the graft audit.
    if value is None:
        return (), "none"
    if not hasattr(value, "shape") or not hasattr(value, "dtype"):
        raise AuditedGraftError(f"parameter leaf has no shape/dtype: {type(value).__name__}")
    shape = tuple(int(dim) for dim in value.shape)
    try:
        dtype = np.dtype(value.dtype)
    except TypeError as exc:
        raise AuditedGraftError(f"parameter leaf has unsupported dtype: {value.dtype!r}") from exc
    if dtype.hasobject:
        raise AuditedGraftError("object-dtype parameter leaves are forbidden")
    # ``dtype.str`` is not an identity-preserving representation for NumPy extension dtypes:
    # ml_dtypes.bfloat16, for example, reports ``<V2`` and would collide with an actual two-byte
    # void dtype. ``str(dtype)`` preserves the bfloat16 name while retaining explicit byte order
    # for non-native-endian built-in dtypes.
    return shape, str(dtype)


def _as_parameter_array(value: Any, *, path: tuple[str, ...]) -> np.ndarray:
    if isinstance(value, jax.ShapeDtypeStruct):
        raise AuditedGraftError(f"source parameter {_path_string(path)!r} was restored as schema, not an array")
    array = np.asarray(value)
    _leaf_spec(array)
    return array


def _leaf_value_hash(value: Any) -> str:
    if value is None:
        digest = hashlib.sha256()
        _hash_field(digest, b"none")
        _hash_field(digest, b"[]")
        _hash_field(digest, b"structural-none")
        return digest.hexdigest()
    array = _canonical_parameter_array(value)
    _, dtype = _leaf_spec(array)
    digest = hashlib.sha256()
    _hash_field(digest, dtype.encode())
    _hash_field(digest, json.dumps(list(array.shape), separators=(",", ":")).encode())
    _hash_field(digest, array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_parameter_array(value: Any) -> np.ndarray:
    if isinstance(value, jax.ShapeDtypeStruct):
        raise AuditedGraftError("cannot value-hash a ShapeDtypeStruct")
    array = np.ascontiguousarray(np.asarray(value))
    _, dtype = _leaf_spec(array)
    parsed_dtype = np.dtype(dtype)
    if parsed_dtype.itemsize > 1 and parsed_dtype.byteorder == ">":
        array = array.byteswap().view(parsed_dtype.newbyteorder("<"))
    return array


def _leaf_record(path: tuple[str, ...], action: str, value: Any, *, include_value: bool = True) -> GraftLeafRecord:
    try:
        shape, dtype = _leaf_spec(value)
    except AuditedGraftError as exc:
        raise AuditedGraftError(f"parameter leaf {_path_string(path)!r}: {exc}") from exc
    return GraftLeafRecord(
        path=_path_string(path),
        action=action,
        shape=shape,
        dtype=dtype,
        value_sha256=_leaf_value_hash(value) if include_value else None,
    )


def _schema_tree_hash(flat_tree: Mapping[tuple[str, ...], Any]) -> str:
    digest = hashlib.sha256()
    _hash_field(digest, b"openpi-parameter-schema-v1")
    for path in sorted(flat_tree, key=_path_string):
        shape, dtype = _leaf_spec(flat_tree[path])
        _hash_field(digest, _path_string(path).encode())
        _hash_field(digest, dtype.encode())
        _hash_field(digest, json.dumps(list(shape), separators=(",", ":")).encode())
    return digest.hexdigest()


def _value_tree_hash(flat_tree: Mapping[tuple[str, ...], Any]) -> str:
    digest = hashlib.sha256()
    _hash_field(digest, b"openpi-parameter-values-v1")
    for path in sorted(flat_tree, key=_path_string):
        _hash_field(digest, _path_string(path).encode())
        _hash_field(digest, _leaf_value_hash(flat_tree[path]).encode())
    return digest.hexdigest()


def parameter_tree_sha256(params: at.Params) -> str:
    """Hash an actual parameter tree by canonical path, shape, dtype, and value.

    Leaves are materialized and hashed one at a time by ``_leaf_value_hash``. This bounds host
    memory to the largest individual leaf even when ``params`` is a multi-billion-parameter
    sharded step-0 state.
    """
    flat_params = flax.traverse_util.flatten_dict(params)
    _validate_flat_paths(flat_params, tree_name="parameter")
    return _value_tree_hash(flat_params)


def _graft_tree_hash(flat_tree: Mapping[tuple[str, ...], Any], fresh_paths: set[tuple[str, ...]]) -> str:
    digest = hashlib.sha256()
    _hash_field(digest, b"openpi-audited-graft-v1")
    for path in sorted(flat_tree, key=_path_string):
        _hash_field(digest, _path_string(path).encode())
        if path in fresh_paths:
            shape, dtype = _leaf_spec(flat_tree[path])
            _hash_field(digest, b"fresh-init-schema")
            _hash_field(digest, dtype.encode())
            _hash_field(digest, json.dumps(list(shape), separators=(",", ":")).encode())
        else:
            _hash_field(digest, b"loaded-value")
            _hash_field(digest, _leaf_value_hash(flat_tree[path]).encode())
    return digest.hexdigest()


def _hash_field(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _write_manifest(path: pathlib.Path, manifest: GraftManifest) -> None:
    # The graft manifest is an immutable-stage artifact: consumers (v35_prepare_pilot's
    # _load_immutable_json) accept only canonical compact sorted JSON with exactly one
    # trailing newline, so the producer must emit those exact bytes.
    payload = _canonical_json(manifest.to_dict()) + "\n"
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return
        raise FileExistsError(f"refusing to overwrite a different graft manifest: {path}")

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
