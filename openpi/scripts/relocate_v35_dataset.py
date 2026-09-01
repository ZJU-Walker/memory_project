"""Copy the frozen v3.5 LeRobot dataset into the portable project tree.

This is intentionally a copy-only tool.  It never renames, removes, chmods, or
writes anything below the source tree.  Existing destination payloads are only
accepted when their byte count and SHA256 match the source; mismatches and
unexpected files fail closed.  Interrupted files use a deterministic sidecar
name and resume only after their entire prefix is compared with the source.

The production layout is::

    <memory_project>/data/lerobot/yam/bin_memory_0830_0831_v36_subtask

Set ``HF_LEROBOT_HOME=<memory_project>/data/lerobot`` before importing LeRobot.
The unchanged repo ID then resolves directly to the copied directory.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import stat
import sys
from typing import Any
import uuid

SCHEMA_VERSION = "openpi.v35.dataset-tree-inventory.v1"
DATASET_REPO_ID = "yam/bin_memory_0830_0831_v36_subtask"
DATASET_RELATIVE_PATH = PurePosixPath("data/lerobot") / DATASET_REPO_ID
INVENTORY_RELATIVE_PATH = PurePosixPath("data/0830_0831_v36_dataset_tree_inventory.json")
PARTIAL_SUFFIX = ".v35-relocation.part"
CHUNK_SIZE = 8 * 1024 * 1024
SHA256_RE = re.compile(r"[0-9a-f]{64}")

# Identity of the user-approved, independently audited one-folder upload.
EXPECTED_FILE_COUNT = 202
EXPECTED_DIRECTORY_COUNT = 108  # Does not include the dataset root itself.
EXPECTED_TOTAL_BYTES = 58_534_560_294
EXPECTED_TRANSFER_MANIFEST_SHA256 = "51fd7899988bc6502bd21d3e1d372f6bdf47c2c434f4dbafe8375143618eab86"
EXPECTED_EPISODES = 70
EXPECTED_FRAMES = 55_980
EXPECTED_TASKS = 7
REQUIRED_FILES = frozenset(
    {
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl",
        "meta/episode_prompts.json",
        "meta/memory_waiting_cores.json",
        "v35_training_bundle/README.md",
        "v35_training_bundle/TRANSFER_MANIFEST.json",
        (
            "v35_training_bundle/restore_tree/project_root/data/"
            "0830_0831_episode_manifest_v36_frozen.json"
        ),
    }
)


class RelocationError(RuntimeError):
    """Raised when relocation cannot remain byte-exact and fail-closed."""


@dataclasses.dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> FileIdentity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            mode=value.st_mode,
            size=value.st_size,
            mtime_ns=value.st_mtime_ns,
        )


@dataclasses.dataclass(frozen=True)
class ScannedFile:
    relative_path: PurePosixPath
    absolute_path: Path
    identity: FileIdentity


@dataclasses.dataclass(frozen=True)
class TreeScan:
    root: Path
    directories: tuple[PurePosixPath, ...]
    files: tuple[ScannedFile, ...]

    @property
    def total_bytes(self) -> int:
        return sum(item.identity.size for item in self.files)


@dataclasses.dataclass(frozen=True)
class FileRecord:
    path: PurePosixPath
    size: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {"path": self.path.as_posix(), "sha256": self.sha256, "size": self.size}


@dataclasses.dataclass(frozen=True)
class TreeInventory:
    dataset_repo_id: str
    directories: tuple[PurePosixPath, ...]
    files: tuple[FileRecord, ...]

    @property
    def total_bytes(self) -> int:
        return sum(item.size for item in self.files)

    @property
    def tree_sha256(self) -> str:
        tree = {
            "directories": [path.as_posix() for path in self.directories],
            "files": [item.as_dict() for item in self.files],
        }
        return _sha256_bytes(_canonical_json_bytes(tree))

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_repo_id": self.dataset_repo_id,
            "directories": [path.as_posix() for path in self.directories],
            "directory_count": len(self.directories),
            "file_count": len(self.files),
            "files": [item.as_dict() for item in self.files],
            "schema_version": SCHEMA_VERSION,
            "total_bytes": self.total_bytes,
            "tree_sha256": self.tree_sha256,
        }

    def canonical_bytes(self) -> bytes:
        return _canonical_json_bytes(self.as_dict()) + b"\n"


def memory_project_root() -> Path:
    """Return the source checkout root without relying on the current directory."""

    root = Path(__file__).resolve().parents[2]
    if not (root / "openpi" / "pyproject.toml").is_file():
        raise RelocationError(f"could not discover memory_project from script location: {root}")
    return root


def default_destination() -> Path:
    return memory_project_root().joinpath(*DATASET_RELATIVE_PATH.parts)


def default_inventory_path() -> Path:
    return memory_project_root().joinpath(*INVENTORY_RELATIVE_PATH.parts)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _path_lstat(path: Path, *, role: str) -> os.stat_result:
    try:
        return path.lstat()
    except FileNotFoundError as exc:
        raise RelocationError(f"{role} disappeared: {path}") from exc


def _require_plain_directory(path: Path, *, role: str) -> os.stat_result:
    value = _path_lstat(path, role=role)
    if stat.S_ISLNK(value.st_mode):
        raise RelocationError(f"{role} must not be a symlink: {path}")
    if not stat.S_ISDIR(value.st_mode):
        raise RelocationError(f"{role} is not a directory: {path}")
    return value


def _require_plain_file(path: Path, *, role: str) -> os.stat_result:
    value = _path_lstat(path, role=role)
    if stat.S_ISLNK(value.st_mode):
        raise RelocationError(f"{role} must not be a symlink: {path}")
    if not stat.S_ISREG(value.st_mode):
        raise RelocationError(f"{role} is not a regular file: {path}")
    return value


def _relative_posix(root: Path, path: Path) -> PurePosixPath:
    relative = path.relative_to(root)
    return PurePosixPath(*relative.parts)


def scan_tree(root: Path) -> TreeScan:
    """Scan a tree without following links and reject every special node."""

    root = Path(root).expanduser()
    _require_plain_directory(root, role="tree root")
    root = root.absolute()
    directories: list[PurePosixPath] = []
    files: list[ScannedFile] = []

    for current_text, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        current = Path(current_text)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            path = current / name
            value = _require_plain_directory(path, role="tree directory")
            if value.st_nlink < 1:
                raise RelocationError(f"tree directory has an invalid link count: {path}")
            directories.append(_relative_posix(root, path))
        for name in file_names:
            path = current / name
            value = _require_plain_file(path, role="tree file")
            files.append(
                ScannedFile(
                    relative_path=_relative_posix(root, path),
                    absolute_path=path,
                    identity=FileIdentity.from_stat(value),
                )
            )

    directories.sort(key=PurePosixPath.as_posix)
    files.sort(key=lambda item: item.relative_path.as_posix())
    return TreeScan(root=root, directories=tuple(directories), files=tuple(files))


def _open_readonly(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RelocationError(f"could not open regular file without following links: {path}: {exc}") from exc
    value = os.fstat(descriptor)
    if not stat.S_ISREG(value.st_mode):
        os.close(descriptor)
        raise RelocationError(f"opened payload is not a regular file: {path}")
    return descriptor


def _identity_from_open_file(descriptor: int) -> FileIdentity:
    return FileIdentity.from_stat(os.fstat(descriptor))


def _same_payload_identity(left: FileIdentity, right: FileIdentity) -> bool:
    return (
        left.device == right.device
        and left.inode == right.inode
        and left.mode == right.mode
        and left.size == right.size
        and left.mtime_ns == right.mtime_ns
    )


def _verify_path_identity(path: Path, expected: FileIdentity, *, role: str) -> None:
    actual = FileIdentity.from_stat(_require_plain_file(path, role=role))
    if not _same_payload_identity(actual, expected):
        raise RelocationError(f"{role} changed while relocation was running: {path}")


def hash_file_stable(path: Path, *, expected_size: int | None = None) -> tuple[str, FileIdentity]:
    """Hash a regular file and prove its identity stayed stable during the read."""

    descriptor = _open_readonly(path)
    try:
        before = _identity_from_open_file(descriptor)
        if expected_size is not None and before.size != expected_size:
            raise RelocationError(f"file size mismatch for {path}: expected {expected_size}, got {before.size}")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            while chunk := stream.read(CHUNK_SIZE):
                digest.update(chunk)
        after = _identity_from_open_file(descriptor)
    finally:
        os.close(descriptor)
    if not _same_payload_identity(before, after):
        raise RelocationError(f"file changed while it was being hashed: {path}")
    _verify_path_identity(path, after, role="hashed file")
    return digest.hexdigest(), after


def _validate_v35_identity(scan: TreeScan) -> None:
    file_paths = {item.relative_path.as_posix() for item in scan.files}
    missing = sorted(REQUIRED_FILES - file_paths)
    if missing:
        raise RelocationError(f"v3.5 dataset is missing required files: {missing}")
    if len(scan.files) != EXPECTED_FILE_COUNT:
        raise RelocationError(
            f"v3.5 dataset file count mismatch: expected {EXPECTED_FILE_COUNT}, got {len(scan.files)}"
        )
    if len(scan.directories) != EXPECTED_DIRECTORY_COUNT:
        raise RelocationError(
            "v3.5 dataset directory count mismatch: "
            f"expected {EXPECTED_DIRECTORY_COUNT}, got {len(scan.directories)}"
        )
    if scan.total_bytes != EXPECTED_TOTAL_BYTES:
        raise RelocationError(
            f"v3.5 dataset byte count mismatch: expected {EXPECTED_TOTAL_BYTES}, got {scan.total_bytes}"
        )

    transfer = scan.root / "v35_training_bundle" / "TRANSFER_MANIFEST.json"
    transfer_sha256, _ = hash_file_stable(transfer)
    if transfer_sha256 != EXPECTED_TRANSFER_MANIFEST_SHA256:
        raise RelocationError(
            "v3.5 transfer manifest mismatch: "
            f"expected {EXPECTED_TRANSFER_MANIFEST_SHA256}, got {transfer_sha256}"
        )

    info_path = scan.root / "meta" / "info.json"
    try:
        info = json.loads(info_path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise RelocationError(f"could not read LeRobot metadata: {info_path}: {exc}") from exc
    expected_info = {
        "robot_type": "yam",
        "total_episodes": EXPECTED_EPISODES,
        "total_frames": EXPECTED_FRAMES,
        "total_tasks": EXPECTED_TASKS,
        "total_videos": 0,
    }
    actual_info = {key: info.get(key) for key in expected_info}
    if actual_info != expected_info:
        raise RelocationError(f"LeRobot metadata identity mismatch: expected {expected_info}, got {actual_info}")


def build_inventory(root: Path, *, enforce_v35_identity: bool = True) -> TreeInventory:
    scan = scan_tree(root)
    if enforce_v35_identity:
        _validate_v35_identity(scan)
    records: list[FileRecord] = []
    for index, item in enumerate(scan.files, start=1):
        print(f"hash source [{index}/{len(scan.files)}] {item.relative_path.as_posix()}", flush=True)
        digest, identity = hash_file_stable(item.absolute_path, expected_size=item.identity.size)
        if not _same_payload_identity(identity, item.identity):
            raise RelocationError(f"source changed after its structural scan: {item.absolute_path}")
        records.append(FileRecord(path=item.relative_path, size=item.identity.size, sha256=digest))
    final_scan = scan_tree(scan.root)
    _require_same_structure(scan, final_scan, role="source")
    if enforce_v35_identity:
        _validate_v35_identity(final_scan)
    return TreeInventory(
        dataset_repo_id=DATASET_REPO_ID,
        directories=scan.directories,
        files=tuple(records),
    )


def _parse_inventory_path(value: object, *, role: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise RelocationError(f"{role} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or path.as_posix() != value:
        raise RelocationError(f"{role} is not a canonical confined POSIX path: {value!r}")
    return path


def inventory_from_dict(raw: object) -> TreeInventory:
    if not isinstance(raw, dict):
        raise RelocationError("inventory must be a JSON object")
    expected_keys = {
        "dataset_repo_id",
        "directories",
        "directory_count",
        "file_count",
        "files",
        "schema_version",
        "total_bytes",
        "tree_sha256",
    }
    if set(raw) != expected_keys:
        raise RelocationError(f"inventory keys mismatch: expected {sorted(expected_keys)}, got {sorted(raw)}")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise RelocationError(f"unsupported inventory schema: {raw['schema_version']!r}")
    if raw["dataset_repo_id"] != DATASET_REPO_ID:
        raise RelocationError(f"inventory repo ID mismatch: {raw['dataset_repo_id']!r}")

    raw_directories = raw["directories"]
    raw_files = raw["files"]
    if not isinstance(raw_directories, list) or not isinstance(raw_files, list):
        raise RelocationError("inventory directories and files must be lists")
    directories = tuple(_parse_inventory_path(item, role="inventory directory") for item in raw_directories)
    if list(directories) != sorted(directories, key=PurePosixPath.as_posix) or len(set(directories)) != len(
        directories
    ):
        raise RelocationError("inventory directories must be unique and sorted")

    records: list[FileRecord] = []
    for item in raw_files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
            raise RelocationError("inventory file records must contain exactly path, sha256, and size")
        path = _parse_inventory_path(item["path"], role="inventory file")
        size = item["size"]
        sha256 = item["sha256"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise RelocationError(f"inventory file has invalid size: {item!r}")
        if not isinstance(sha256, str) or SHA256_RE.fullmatch(sha256) is None:
            raise RelocationError(f"inventory file has invalid SHA256: {item!r}")
        records.append(FileRecord(path=path, size=size, sha256=sha256))
    files = tuple(records)
    if list(files) != sorted(files, key=lambda item: item.path.as_posix()) or len({item.path for item in files}) != len(
        files
    ):
        raise RelocationError("inventory files must be unique and sorted")

    inventory = TreeInventory(dataset_repo_id=DATASET_REPO_ID, directories=directories, files=files)
    expected_scalars = {
        "directory_count": len(directories),
        "file_count": len(files),
        "total_bytes": inventory.total_bytes,
        "tree_sha256": inventory.tree_sha256,
    }
    actual_scalars = {key: raw[key] for key in expected_scalars}
    if actual_scalars != expected_scalars:
        raise RelocationError(f"inventory summary mismatch: expected {expected_scalars}, got {actual_scalars}")
    return inventory


def load_inventory(path: Path) -> TreeInventory:
    path = Path(path).expanduser()
    _require_plain_file(path, role="inventory")
    raw_bytes = path.read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise RelocationError(f"inventory is not valid JSON: {path}: {exc}") from exc
    inventory = inventory_from_dict(raw)
    if raw_bytes != inventory.canonical_bytes():
        raise RelocationError(f"inventory is not canonical JSON: {path}")
    return inventory


def write_inventory(path: Path, inventory: TreeInventory) -> None:
    """Create an inventory atomically; accept an existing byte-identical one."""

    path = Path(path).expanduser()
    payload = inventory.canonical_bytes()
    if path.exists() or path.is_symlink():
        _require_plain_file(path, role="existing inventory")
        if path.read_bytes() != payload:
            raise RelocationError(f"refusing to overwrite a different inventory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as exc:
        _require_plain_file(path, role="concurrent inventory")
        if path.read_bytes() != payload:
            raise RelocationError(f"refusing to overwrite a concurrently created inventory: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _partial_path(destination: Path) -> Path:
    return destination.parent / f".{destination.name}{PARTIAL_SUFFIX}"


def _require_same_structure(before: TreeScan, after: TreeScan, *, role: str) -> None:
    if before.directories != after.directories:
        raise RelocationError(f"{role} directory inventory changed during relocation")
    before_files = [(item.relative_path, item.identity.size) for item in before.files]
    after_files = [(item.relative_path, item.identity.size) for item in after.files]
    if before_files != after_files:
        raise RelocationError(f"{role} file inventory changed during relocation")


def _expected_partial_paths(file_paths: Iterable[PurePosixPath]) -> set[PurePosixPath]:
    result: set[PurePosixPath] = set()
    for relative in file_paths:
        name = f".{relative.name}{PARTIAL_SUFFIX}"
        result.add(relative.parent / name)
    return result


def _audit_partial_destination(destination: Path, source_scan: TreeScan) -> None:
    if not destination.exists() and not destination.is_symlink():
        return
    scan = scan_tree(destination)
    expected_directories = set(source_scan.directories)
    actual_directories = set(scan.directories)
    if not actual_directories.issubset(expected_directories):
        extras = sorted(actual_directories - expected_directories, key=PurePosixPath.as_posix)
        raise RelocationError(f"destination contains unexpected directories: {extras}")
    expected_files = {item.relative_path for item in source_scan.files}
    allowed_files = expected_files | _expected_partial_paths(expected_files)
    actual_files = {item.relative_path for item in scan.files}
    extras = actual_files - allowed_files
    if extras:
        raise RelocationError(
            f"destination contains unexpected files: {sorted(extras, key=PurePosixPath.as_posix)}"
        )


def _mkdir_plain(path: Path) -> None:
    if path.exists() or path.is_symlink():
        _require_plain_directory(path, role="destination directory")
        return
    try:
        path.mkdir()
    except FileExistsError:
        _require_plain_directory(path, role="concurrently created destination directory")


def _prepare_destination(destination: Path, directories: Sequence[PurePosixPath]) -> None:
    destination = Path(destination).expanduser().absolute()
    missing: list[Path] = []
    current = destination
    while not current.exists() and not current.is_symlink():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    _require_plain_directory(current, role="destination ancestor")
    for path in reversed(missing):
        _mkdir_plain(path)
    _require_plain_directory(destination, role="destination root")
    for relative in sorted(directories, key=lambda value: (len(value.parts), value.as_posix())):
        _mkdir_plain(destination.joinpath(*relative.parts))


def _validate_source_destination(source: Path, destination: Path) -> tuple[Path, Path]:
    source = Path(source).expanduser().absolute()
    destination = Path(destination).expanduser().absolute()
    source_resolved = source.resolve(strict=True)
    destination_resolved = destination.resolve(strict=False)
    if source_resolved == destination_resolved:
        raise RelocationError("source and destination must be different directories")
    if source_resolved in destination_resolved.parents:
        raise RelocationError("destination must not be nested below the source")
    if destination_resolved in source_resolved.parents:
        raise RelocationError("source must not be nested below the destination")
    return source, destination


def _read_exact(descriptor: int, count: int, *, path: Path) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            raise RelocationError(f"file ended before its recorded size: {path}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _compare_prefix(source_descriptor: int, partial_descriptor: int, length: int, *, path: Path) -> Any:
    digest = hashlib.sha256()
    remaining = length
    os.lseek(source_descriptor, 0, os.SEEK_SET)
    os.lseek(partial_descriptor, 0, os.SEEK_SET)
    while remaining:
        count = min(CHUNK_SIZE, remaining)
        source_chunk = _read_exact(source_descriptor, count, path=path)
        partial_chunk = _read_exact(partial_descriptor, count, path=path)
        if source_chunk != partial_chunk:
            raise RelocationError(f"resumable partial does not match the source prefix: {path}")
        digest.update(source_chunk)
        remaining -= count
    return digest


def _nearest_existing_directory(path: Path) -> Path:
    current = path
    while not current.exists() and not current.is_symlink():
        if current.parent == current:
            break
        current = current.parent
    _require_plain_directory(current, role="destination filesystem ancestor")
    return current


def _remaining_copy_bytes(destination: Path, source_scan: TreeScan) -> int:
    remaining = 0
    for source in source_scan.files:
        final = destination.joinpath(*source.relative_path.parts)
        partial = _partial_path(final)
        if final.exists() or final.is_symlink():
            _require_plain_file(final, role="existing destination file")
            continue
        if partial.exists() or partial.is_symlink():
            value = _require_plain_file(partial, role="resumable partial")
            if value.st_size > source.identity.size:
                raise RelocationError(f"resumable partial is larger than its source: {partial}")
            remaining += source.identity.size - value.st_size
        else:
            remaining += source.identity.size
    return remaining


def _copy_or_resume_file(
    source: ScannedFile,
    destination: Path,
    *,
    expected: FileRecord | None,
) -> tuple[FileRecord, FileIdentity, FileIdentity]:
    final = destination.joinpath(*source.relative_path.parts)
    partial = _partial_path(final)
    if final.exists() or final.is_symlink():
        _require_plain_file(final, role="existing destination file")
        source_digest, source_identity = hash_file_stable(
            source.absolute_path, expected_size=source.identity.size
        )
        destination_digest, destination_identity = hash_file_stable(final, expected_size=source.identity.size)
        record = FileRecord(path=source.relative_path, size=source.identity.size, sha256=source_digest)
        _validate_record(record, expected=expected, destination_digest=destination_digest)
        if partial.exists() or partial.is_symlink():
            _require_plain_file(partial, role="stale resumable partial")
            partial_digest, _ = hash_file_stable(partial, expected_size=source.identity.size)
            if partial_digest != source_digest:
                raise RelocationError(f"stale resumable partial conflicts with completed file: {partial}")
            partial.unlink()
        return record, source_identity, destination_identity

    if partial.exists() or partial.is_symlink():
        partial_stat = _require_plain_file(partial, role="resumable partial")
        if partial_stat.st_size > source.identity.size:
            raise RelocationError(f"resumable partial is larger than its source: {partial}")
        partial_descriptor = os.open(
            partial,
            os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
    else:
        partial_descriptor = os.open(
            partial,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )

    source_descriptor = _open_readonly(source.absolute_path)
    try:
        source_before = _identity_from_open_file(source_descriptor)
        if not _same_payload_identity(source_before, source.identity):
            raise RelocationError(f"source changed before copy: {source.absolute_path}")
        partial_before = _identity_from_open_file(partial_descriptor)
        if not stat.S_ISREG(partial_before.mode):
            raise RelocationError(f"resumable partial is not a regular file: {partial}")
        digest = _compare_prefix(source_descriptor, partial_descriptor, partial_before.size, path=partial)
        os.lseek(partial_descriptor, 0, os.SEEK_END)
        remaining = source_before.size - partial_before.size
        while remaining:
            chunk = os.read(source_descriptor, min(CHUNK_SIZE, remaining))
            if not chunk:
                raise RelocationError(f"source ended early during copy: {source.absolute_path}")
            view = memoryview(chunk)
            while view:
                written = os.write(partial_descriptor, view)
                if written <= 0:
                    raise RelocationError(f"could not make progress writing partial: {partial}")
                view = view[written:]
            digest.update(chunk)
            remaining -= len(chunk)
        os.fsync(partial_descriptor)
        os.fchmod(partial_descriptor, stat.S_IMODE(source_before.mode))
        source_after = _identity_from_open_file(source_descriptor)
        partial_after = _identity_from_open_file(partial_descriptor)
    finally:
        os.close(source_descriptor)
        os.close(partial_descriptor)
    if not _same_payload_identity(source_before, source_after):
        raise RelocationError(f"source changed while it was copied: {source.absolute_path}")
    if partial_after.size != source_before.size:
        raise RelocationError(f"resumable partial has the wrong completed size: {partial}")
    source_digest = digest.hexdigest()
    partial_digest, _ = hash_file_stable(partial, expected_size=source_before.size)
    record = FileRecord(path=source.relative_path, size=source_before.size, sha256=source_digest)
    _validate_record(record, expected=expected, destination_digest=partial_digest)
    try:
        os.link(partial, final, follow_symlinks=False)
    except FileExistsError as exc:
        raise RelocationError(f"refusing to overwrite a concurrently created destination: {final}") from exc
    partial.unlink()
    destination_digest, destination_identity = hash_file_stable(final, expected_size=source_before.size)
    if destination_digest != source_digest:
        raise RelocationError(f"destination changed while finalizing: {final}")
    return record, source_after, destination_identity


def _validate_record(record: FileRecord, *, expected: FileRecord | None, destination_digest: str) -> None:
    if expected is not None and record != expected:
        raise RelocationError(
            f"source does not match expected inventory for {record.path}: expected {expected}, got {record}"
        )
    if destination_digest != record.sha256:
        raise RelocationError(
            f"destination SHA256 mismatch for {record.path}: expected {record.sha256}, got {destination_digest}"
        )


def _inventory_matches_scan(inventory: TreeInventory, scan: TreeScan, *, role: str) -> None:
    if inventory.directories != scan.directories:
        raise RelocationError(f"{role} directories do not match the expected inventory")
    expected = [(item.path, item.size) for item in inventory.files]
    actual = [(item.relative_path, item.identity.size) for item in scan.files]
    if expected != actual:
        raise RelocationError(f"{role} files do not match the expected inventory")


def copy_dataset(
    source: Path,
    destination: Path,
    *,
    expected_inventory: TreeInventory | None = None,
    inventory_output: Path | None = None,
    enforce_v35_identity: bool = True,
    dry_run: bool = False,
) -> TreeInventory | None:
    """Copy a dataset with resume, no-overwrite, and end-to-end verification."""

    source, destination = _validate_source_destination(source, destination)
    source_scan = scan_tree(source)
    if enforce_v35_identity:
        _validate_v35_identity(source_scan)
    if expected_inventory is not None:
        _inventory_matches_scan(expected_inventory, source_scan, role="source")
    _audit_partial_destination(destination, source_scan)
    if inventory_output is not None:
        output = Path(inventory_output).expanduser().absolute().resolve(strict=False)
        for tree, role in ((source, "source"), (destination, "destination")):
            resolved_tree = tree.resolve(strict=False)
            if output == resolved_tree or resolved_tree in output.parents:
                raise RelocationError(f"inventory output must stay outside the {role} dataset tree: {output}")
    remaining_bytes = _remaining_copy_bytes(destination, source_scan)
    free_bytes = shutil.disk_usage(_nearest_existing_directory(destination)).free
    if free_bytes < remaining_bytes:
        raise RelocationError(
            f"destination filesystem has insufficient free space: need {remaining_bytes}, available {free_bytes}"
        )
    if dry_run:
        print(
            f"dry-run passed: {len(source_scan.files)} files, {source_scan.total_bytes} bytes; "
            f"remaining={remaining_bytes}, free={free_bytes}, destination={destination} (no writes performed)"
        )
        return None

    _prepare_destination(destination, source_scan.directories)
    expected_by_path = {} if expected_inventory is None else {item.path: item for item in expected_inventory.files}
    records: list[FileRecord] = []
    source_identities: dict[PurePosixPath, FileIdentity] = {}
    destination_identities: dict[PurePosixPath, FileIdentity] = {}
    for index, item in enumerate(source_scan.files, start=1):
        print(f"copy/verify [{index}/{len(source_scan.files)}] {item.relative_path.as_posix()}", flush=True)
        record, source_identity, destination_identity = _copy_or_resume_file(
            item,
            destination,
            expected=expected_by_path.get(item.relative_path),
        )
        records.append(record)
        source_identities[item.relative_path] = source_identity
        destination_identities[item.relative_path] = destination_identity

    inventory = TreeInventory(
        dataset_repo_id=DATASET_REPO_ID,
        directories=source_scan.directories,
        files=tuple(records),
    )
    if expected_inventory is not None and inventory != expected_inventory:
        raise RelocationError("completed source inventory does not match the supplied inventory")

    final_source_scan = scan_tree(source)
    final_destination_scan = scan_tree(destination)
    _require_same_structure(source_scan, final_source_scan, role="source")
    _inventory_matches_scan(inventory, final_destination_scan, role="destination")
    if enforce_v35_identity:
        _validate_v35_identity(final_source_scan)
        _validate_v35_identity(final_destination_scan)
    for item in final_source_scan.files:
        _verify_path_identity(item.absolute_path, source_identities[item.relative_path], role="source file")
    for item in final_destination_scan.files:
        _verify_path_identity(
            item.absolute_path,
            destination_identities[item.relative_path],
            role="destination file",
        )
    if inventory_output is not None:
        write_inventory(inventory_output, inventory)
    print(
        f"copy verified: {len(inventory.files)} files, {inventory.total_bytes} bytes, "
        f"tree_sha256={inventory.tree_sha256}",
        flush=True,
    )
    return inventory


def verify_tree(tree: Path, inventory: TreeInventory, *, enforce_v35_identity: bool = True) -> None:
    scan = scan_tree(tree)
    if enforce_v35_identity:
        _validate_v35_identity(scan)
    _inventory_matches_scan(inventory, scan, role="verified tree")
    for index, item in enumerate(scan.files, start=1):
        expected = inventory.files[index - 1]
        print(f"verify [{index}/{len(scan.files)}] {item.relative_path.as_posix()}", flush=True)
        digest, identity = hash_file_stable(item.absolute_path, expected_size=expected.size)
        if digest != expected.sha256:
            raise RelocationError(
                f"verified tree SHA256 mismatch for {expected.path}: expected {expected.sha256}, got {digest}"
            )
        if not _same_payload_identity(identity, item.identity):
            raise RelocationError(f"verified tree changed after structural scan: {item.absolute_path}")
    final_scan = scan_tree(scan.root)
    _require_same_structure(scan, final_scan, role="verified tree")
    if enforce_v35_identity:
        _validate_v35_identity(final_scan)
    print(
        f"tree verified: {len(inventory.files)} files, {inventory.total_bytes} bytes, "
        f"tree_sha256={inventory.tree_sha256}",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    inventory_parser = commands.add_parser("inventory", help="hash a source tree and write a canonical inventory")
    inventory_parser.add_argument("--source", type=Path, required=True)
    inventory_parser.add_argument("--output", type=Path, default=None)

    copy_parser = commands.add_parser("copy", help="copy or resume the frozen dataset and verify every file")
    copy_parser.add_argument("--source", type=Path, required=True)
    copy_parser.add_argument("--destination", type=Path, default=None)
    copy_parser.add_argument("--expected-inventory", type=Path, default=None)
    copy_parser.add_argument("--inventory-output", type=Path, default=None)
    copy_parser.add_argument("--dry-run", action="store_true")

    verify_parser = commands.add_parser("verify", help="verify a copied tree against its canonical inventory")
    verify_parser.add_argument("--tree", type=Path, default=None)
    verify_parser.add_argument("--inventory", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inventory":
            inventory = build_inventory(args.source)
            output = default_inventory_path() if args.output is None else args.output
            write_inventory(output, inventory)
            print(f"inventory written: {output} ({inventory.tree_sha256})")
            return 0
        if args.command == "copy":
            destination = default_destination() if args.destination is None else args.destination
            expected = None if args.expected_inventory is None else load_inventory(args.expected_inventory)
            output = default_inventory_path() if args.inventory_output is None else args.inventory_output
            copy_dataset(
                args.source,
                destination,
                expected_inventory=expected,
                inventory_output=None if args.dry_run else output,
                dry_run=args.dry_run,
            )
            return 0
        if args.command == "verify":
            tree = default_destination() if args.tree is None else args.tree
            inventory_path = default_inventory_path() if args.inventory is None else args.inventory
            verify_tree(tree, load_inventory(inventory_path))
            return 0
    except RelocationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
