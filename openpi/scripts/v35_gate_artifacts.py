"""Shared fail-closed artifact helpers for the v3.5 release-gate reducers.

The reducers intentionally consume sealed artifacts rather than importing the training or
evaluation stack.  This module owns the canonical JSON envelope and the small, read-only
view of the frozen episode manifest that both reducers need.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
from pathlib import Path
import re
from typing import Any

SHA256_RE = re.compile(r"[0-9a-f]{64}")
CANONICAL_COLLECTIONS = frozenset(("0830", "0831"))
CANONICAL_SPLITS = frozenset(("train", "development", "final_test"))
V35_SPLIT_SEED = 36
V35_SPLIT_ALGORITHM = "openpi.v36.sha256-ranked-manifest-fields.v1"
V35_SPLIT_ALGORITHM_SPEC = (
    "seed=36; sort stable_id; rank by sha256(algorithm,stage,seed,stable_id,part,object,target_side); "
    "select one final-test episode per collection*object*side cell; then one development episode per "
    "collection*object*side cell while preserving one train episode in every 0830 part*object*side "
    "and 0831 object*side cell; assign every other included episode to train"
)
V35_SPLIT_ALGORITHM_SHA256 = hashlib.sha256(V35_SPLIT_ALGORITHM_SPEC.encode()).hexdigest()
V36_COLLECTION_COUNTS = {"0830": 30, "0831": 40}
V36_TRAIN_EPISODES = 54
V35_OBJECTS = frozenset(("banana", "grey_pepper_box"))


class GateArtifactError(ValueError):
    """Raised when a release-gate input is incomplete, mutable, or inconsistent."""


@dataclasses.dataclass(frozen=True)
class ManifestEpisode:
    stable_id: str
    episode_index: int
    collection: str
    object_name: str
    part: str
    target_side: int
    split: str


@dataclasses.dataclass(frozen=True)
class FrozenManifest:
    path: Path
    sha256: str
    split_assignment_sha256: str
    episodes: tuple[ManifestEpisode, ...]

    def split(self, name: str) -> tuple[ManifestEpisode, ...]:
        return tuple(episode for episode in self.episodes if episode.split == name)


@dataclasses.dataclass(frozen=True)
class InitializationIdentity:
    file_sha256: str
    identity_sha256: str
    parameter_tree_sha256: str
    official_source_uri: str
    official_source_tree_sha256: str
    memory_inject_w_sha256: str
    config_name: str
    initialization_seed: int
    artifact_hashes: Mapping[str, Any]


def canonical_json_bytes(value: Any) -> bytes:
    """Return the canonical UTF-8 encoding used by every v3.5 gate artifact."""
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
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise GateArtifactError(f"{name} must be a lower-case 64-character SHA256 digest")
    return value


def require_artifact_id(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        raise GateArtifactError(f"{name} must have the form sha256:<lower-case digest>")
    require_sha256(name, value.removeprefix("sha256:"))
    return value


def artifact_envelope(schema_version: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    payload_copy = dict(payload)
    digest = sha256_bytes(canonical_json_bytes(payload_copy))
    return {
        "schema_version": schema_version,
        "artifact_id": f"sha256:{digest}",
        "payload": payload_copy,
    }


def verify_envelope(value: Any, *, schema_version: str) -> bool:
    if not isinstance(value, dict) or set(value) != {"schema_version", "artifact_id", "payload"}:
        return False
    if value.get("schema_version") != schema_version or not isinstance(value.get("payload"), dict):
        return False
    digest = sha256_bytes(canonical_json_bytes(value["payload"]))
    return value.get("artifact_id") == f"sha256:{digest}"


def load_canonical_envelope(path: Path, *, schema_version: str) -> dict[str, Any]:
    """Load an exact canonical envelope, rejecting whitespace and duplicate-key ambiguity."""
    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
        value = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise GateArtifactError(f"cannot read {path}: {exc}") from exc
    if not verify_envelope(value, schema_version=schema_version):
        raise GateArtifactError(f"{path} is not a valid {schema_version} envelope")
    if raw_bytes != canonical_json_bytes(value) + b"\n":
        raise GateArtifactError(f"{path} is not in the canonical JSON byte representation")
    return value


def write_canonical_envelope(path: Path, value: Mapping[str, Any], *, schema_version: str) -> None:
    if not verify_envelope(value, schema_version=schema_version):
        raise GateArtifactError("refusing to write an invalid release-gate envelope")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(value) + b"\n")


def require_exact_keys(name: str, value: Any, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GateArtifactError(f"{name} must be an object")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise GateArtifactError(f"{name} keys mismatch: missing={missing}, extra={extra}")
    return value


def resolve_hashed_relative_file(
    *,
    owner_path: Path,
    descriptor: Any,
    descriptor_name: str,
) -> tuple[Path, str]:
    descriptor = require_exact_keys(descriptor_name, descriptor, {"path", "sha256"})
    relative = descriptor["path"]
    if not isinstance(relative, str) or not relative.strip():
        raise GateArtifactError(f"{descriptor_name}.path must be a non-empty relative path")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise GateArtifactError(f"{descriptor_name}.path must not be absolute or escape its artifact directory")
    expected_sha256 = require_sha256(f"{descriptor_name}.sha256", descriptor["sha256"])
    resolved = owner_path.parent / relative_path
    try:
        actual_sha256 = sha256_bytes(resolved.read_bytes())
    except OSError as exc:
        raise GateArtifactError(f"cannot read linked {descriptor_name} file {resolved}: {exc}") from exc
    if actual_sha256 != expected_sha256:
        raise GateArtifactError(f"{descriptor_name} SHA256 mismatch: expected {expected_sha256}, found {actual_sha256}")
    return resolved, actual_sha256


def load_initialization_identity(
    *,
    owner_path: Path,
    descriptor: Any,
    manifest: FrozenManifest,
    expected_parameter_tree_sha256: str,
    calibration_id: str | None = None,
    calibration_file_sha256: str | None = None,
    calibration_raw_gate_sha256: str | None = None,
    calibration_parameters: Mapping[str, float] | None = None,
) -> InitializationIdentity:
    """Authenticate the initialization identity emitted by ``scripts/train.py``."""
    identity_path, file_sha256 = resolve_hashed_relative_file(
        owner_path=owner_path,
        descriptor=descriptor,
        descriptor_name="initialization_manifest",
    )
    try:
        raw = json.loads(identity_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise GateArtifactError(f"cannot read initialization identity {identity_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise GateArtifactError("initialization identity must be a JSON object")
    identity_sha256 = require_sha256("initialization identity_sha256", raw.get("identity_sha256"))
    unhashed = dict(raw)
    del unhashed["identity_sha256"]
    actual_identity_sha256 = hashlib.sha256(
        json.dumps(
            unhashed,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    if identity_sha256 != actual_identity_sha256:
        raise GateArtifactError("initialization identity self-hash is invalid")
    expected_parameter_tree_sha256 = require_sha256(
        "expected initialization parameter tree", expected_parameter_tree_sha256
    )
    official_uri = "gs://openpi-assets/checkpoints/pi05_base/params"
    if (
        raw.get("format_version") != 2
        or raw.get("official_source_uri") != official_uri
        or raw.get("step0_checkpoint") != 0
        or raw.get("actual_step0_parameter_tree_sha256") != expected_parameter_tree_sha256
    ):
        raise GateArtifactError("initialization identity is not the official fresh Pi0.5 step-0 tree")
    artifact_hashes = raw.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict) or artifact_hashes.get("episode_manifest_sha256") != manifest.sha256:
        raise GateArtifactError("initialization identity is not bound to the frozen episode manifest")
    source_tree_sha256 = require_sha256("initialization source_tree_sha256", raw.get("source_tree_sha256"))
    gate_sha256 = require_sha256("initialization memory_inject_w_sha256", raw.get("memory_inject_w_sha256"))
    config_name = raw.get("config_name")
    initialization_seed = raw.get("initialization_seed")
    if not isinstance(config_name, str) or not config_name or type(initialization_seed) is not int:
        raise GateArtifactError("initialization identity is missing config name or initialization seed")
    if calibration_id is not None:
        require_artifact_id("calibration_id", calibration_id)
        if raw.get("calibration_id") != calibration_id:
            raise GateArtifactError("initialization identity calibration ID mismatch")
        if artifact_hashes.get("calibration_artifact_sha256") != calibration_file_sha256:
            raise GateArtifactError("initialization identity calibration file hash mismatch")
        if gate_sha256 != calibration_raw_gate_sha256:
            raise GateArtifactError("initialization identity raw memory gate hash mismatch")
        recorded_parameters = raw.get("memory_calibration")
        if not isinstance(recorded_parameters, dict) or calibration_parameters is None:
            raise GateArtifactError("initialization identity is missing calibrated memory parameters")
        if any(
            float(recorded_parameters.get(name, float("nan"))) != float(value)
            for name, value in calibration_parameters.items()
        ):
            raise GateArtifactError("initialization identity c/tau/alpha do not match calibration")
    return InitializationIdentity(
        file_sha256=file_sha256,
        identity_sha256=identity_sha256,
        parameter_tree_sha256=expected_parameter_tree_sha256,
        official_source_uri=official_uri,
        official_source_tree_sha256=source_tree_sha256,
        memory_inject_w_sha256=gate_sha256,
        config_name=config_name,
        initialization_seed=initialization_seed,
        artifact_hashes=dict(artifact_hashes),
    )


def _target_side(value: Any, stable_id: str) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("left", "right"):
            return int(normalized == "right")
    if type(value) is int and value in (0, 1):
        return value
    raise GateArtifactError(f"manifest episode {stable_id!r} has invalid target_side")


def _split_assignment_digest(episodes: Sequence[ManifestEpisode]) -> str:
    assignment = [
        {
            "collection": episode.collection,
            "episode_index": episode.episode_index,
            "object": episode.object_name,
            "part": episode.part,
            "split": episode.split,
            "stable_id": episode.stable_id,
            "target_side": episode.target_side,
        }
        for episode in sorted(episodes, key=lambda item: item.episode_index)
    ]
    return sha256_bytes(canonical_json_bytes(assignment))


def _split_rank(record: Mapping[str, Any], *, stage: str) -> str:
    fields = (
        V35_SPLIT_ALGORITHM,
        stage,
        str(V35_SPLIT_SEED),
        str(record["stable_id"]),
        str(record.get("part", "")),
        str(record["object"]),
        str(record["target_side"]),
    )
    return hashlib.sha256("\0".join(fields).encode()).hexdigest()


def _guard_cell(record: Mapping[str, Any]) -> tuple[str, ...]:
    if str(record["collection"]) == "0830":
        return ("0830", str(record.get("part", "")), str(record["object"]), str(record["target_side"]))
    return (str(record["collection"]), str(record["object"]), str(record["target_side"]))


def _expected_frozen_splits(records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    by_collection = {
        collection: [record for record in records if record["collection"] == collection]
        for collection in CANONICAL_COLLECTIONS
    }
    if {key: len(value) for key, value in by_collection.items()} != V36_COLLECTION_COUNTS:
        raise GateArtifactError(f"frozen v36 manifest requires exactly {V36_COLLECTION_COUNTS} included episodes")
    cells = {
        (collection, object_name, side): []
        for collection in CANONICAL_COLLECTIONS
        for object_name in V35_OBJECTS
        for side in ("left", "right")
    }
    for record in records:
        cell = (str(record["collection"]), str(record["object"]), str(record["target_side"]))
        if cell not in cells:
            raise GateArtifactError(f"frozen v36 record has invalid collection/object/side cell {cell!r}")
        cells[cell].append(record)
    if any(not values for values in cells.values()):
        raise GateArtifactError("frozen manifest is missing a collection*object*side cell")
    final_ids: set[str] = set()
    remaining: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for cell, values in sorted(cells.items()):
        selected = min(values, key=lambda record: _split_rank(record, stage="final_test"))
        final_ids.add(str(selected["stable_id"]))
        remaining[cell] = [record for record in values if record is not selected]
    development_ids: set[str] = set()
    for cell in sorted(cells):
        pool = remaining[cell]
        guard_counts: dict[tuple[str, ...], int] = {}
        for record in pool:
            guard_counts[_guard_cell(record)] = guard_counts.get(_guard_cell(record), 0) + 1
        candidates = [record for record in pool if guard_counts[_guard_cell(record)] >= 2]
        if not candidates:
            raise GateArtifactError("frozen split cannot preserve train coverage while choosing development")
        selected = min(candidates, key=lambda record: _split_rank(record, stage="development"))
        development_ids.add(str(selected["stable_id"]))
        remaining[cell] = [record for record in pool if record is not selected]
    expected: dict[str, str] = {}
    for record in records:
        stable_id = str(record["stable_id"])
        expected[stable_id] = (
            "final_test" if stable_id in final_ids else "development" if stable_id in development_ids else "train"
        )
    train_guards: dict[tuple[str, ...], int] = {}
    for record in records:
        if expected[str(record["stable_id"])] == "train":
            train_guards[_guard_cell(record)] = train_guards.get(_guard_cell(record), 0) + 1
    all_guards = {_guard_cell(record) for record in records}
    if sum(split == "train" for split in expected.values()) != V36_TRAIN_EPISODES or any(
        train_guards.get(guard, 0) < 1 for guard in all_guards
    ):
        raise GateArtifactError("frozen split algorithm did not preserve the required 54 train episodes/cell coverage")
    return expected


def load_frozen_manifest(path: Path, *, expected_sha256: str) -> FrozenManifest:
    """Load the schema-v2 70-episode manifest without exposing final-test observations."""
    path = Path(path)
    expected_sha256 = require_sha256("episode_manifest_sha256", expected_sha256)
    try:
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise GateArtifactError(f"cannot read frozen episode manifest {path}: {exc}") from exc
    actual_sha256 = sha256_bytes(raw_bytes)
    if actual_sha256 != expected_sha256:
        raise GateArtifactError(f"episode manifest SHA256 mismatch: expected {expected_sha256}, found {actual_sha256}")
    if not isinstance(raw, dict) or raw.get("schema_version") != 2 or not isinstance(raw.get("episodes"), list):
        raise GateArtifactError("release gates require the frozen schema_version=2 episode manifest")
    if (
        raw.get("dataset_version") != "v36"
        or raw.get("review_status") != "frozen"
        or raw.get("split_seed") != V35_SPLIT_SEED
        or raw.get("split_algorithm") != V35_SPLIT_ALGORITHM
        or raw.get("split_algorithm_sha256") != V35_SPLIT_ALGORITHM_SHA256
    ):
        raise GateArtifactError("release gates require the frozen v3.5 review and SHA-ranked split declaration")

    episodes: list[ManifestEpisode] = []
    stable_ids: set[str] = set()
    episode_indices: set[int] = set()
    normalized_records: list[dict[str, Any]] = []
    for offset, record in enumerate(raw["episodes"]):
        if not isinstance(record, dict):
            raise GateArtifactError(f"manifest episode entry {offset} must be an object")
        include = record.get("include", True)
        if type(include) is not bool:
            raise GateArtifactError(f"manifest episode entry {offset} has a non-boolean include flag")
        if not include:
            continue
        stable_id = record.get("stable_id")
        if not isinstance(stable_id, str) or not stable_id.strip() or stable_id != stable_id.strip():
            raise GateArtifactError(f"manifest episode entry {offset} has invalid stable_id")
        if stable_id in stable_ids:
            raise GateArtifactError(f"manifest has duplicate stable_id {stable_id!r}")
        stable_ids.add(stable_id)
        index_value = record.get("episode_index", record.get("lerobot_episode_index"))
        if type(index_value) is not int or index_value < 0:
            raise GateArtifactError(f"included manifest episode {stable_id!r} has invalid episode_index")
        if index_value in episode_indices:
            raise GateArtifactError(f"manifest has duplicate episode_index {index_value}")
        episode_indices.add(index_value)
        collection = record.get("collection")
        object_name = record.get("object", record.get("target_object"))
        part = record.get("part", "")
        split = record.get("split")
        if collection not in CANONICAL_COLLECTIONS:
            raise GateArtifactError(f"manifest episode {stable_id!r} has noncanonical collection {collection!r}")
        if not isinstance(object_name, str) or not object_name.strip() or object_name != object_name.strip():
            raise GateArtifactError(f"manifest episode {stable_id!r} has invalid object")
        if object_name not in V35_OBJECTS:
            raise GateArtifactError(f"manifest episode {stable_id!r} has noncanonical object {object_name!r}")
        if (
            not isinstance(part, str)
            or (collection == "0830" and part not in ("part1", "part2"))
            or (collection == "0831" and part != "")
        ):
            raise GateArtifactError(f"manifest episode {stable_id!r} has invalid part {part!r}")
        if split not in CANONICAL_SPLITS:
            raise GateArtifactError(f"manifest episode {stable_id!r} has invalid split {split!r}")
        side = _target_side(record.get("target_side"), stable_id)
        side_name = "right" if side else "left"
        episodes.append(
            ManifestEpisode(
                stable_id=stable_id,
                episode_index=index_value,
                collection=collection,
                object_name=object_name,
                part=part,
                target_side=side,
                split=split,
            )
        )
        normalized_records.append(
            {
                "stable_id": stable_id,
                "collection": collection,
                "object": object_name,
                "part": part,
                "target_side": side_name,
                "split": split,
            }
        )

    if len(episodes) != 70 or episode_indices != set(range(70)):
        raise GateArtifactError(
            "frozen v3.5 manifest must contain exactly 70 included episodes indexed contiguously from 0"
        )
    split_counts = {split: sum(episode.split == split for episode in episodes) for split in CANONICAL_SPLITS}
    expected_counts = {"train": 54, "development": 8, "final_test": 8}
    if split_counts != expected_counts:
        raise GateArtifactError(f"frozen v3.5 split counts must be {expected_counts}; found {split_counts}")
    expected_splits = _expected_frozen_splits(normalized_records)
    wrong_splits = {
        record["stable_id"]: (expected_splits[record["stable_id"]], record["split"])
        for record in normalized_records
        if record["split"] != expected_splits[record["stable_id"]]
    }
    if wrong_splits:
        raise GateArtifactError(f"manifest split assignments do not reproduce the frozen algorithm: {wrong_splits}")
    objects = {episode.object_name for episode in episodes}
    cells = {(episode.collection, episode.object_name, episode.target_side) for episode in episodes}
    if len(objects) != 2 or len(cells) != 8:
        raise GateArtifactError(
            "frozen v3.5 manifest must contain two objects and all eight collection/object/side cells"
        )
    for collection in CANONICAL_COLLECTIONS:
        for object_name in objects:
            for side in (0, 1):
                if not any(
                    episode.collection == collection
                    and episode.object_name == object_name
                    and episode.target_side == side
                    and episode.split == "train"
                    for episode in episodes
                ):
                    raise GateArtifactError(f"training split is empty for cell {(collection, object_name, side)!r}")

    ordered = tuple(sorted(episodes, key=lambda item: item.episode_index))
    return FrozenManifest(
        path=path,
        sha256=actual_sha256,
        split_assignment_sha256=_split_assignment_digest(ordered),
        episodes=ordered,
    )
