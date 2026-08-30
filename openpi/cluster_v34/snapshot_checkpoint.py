"""Safely publish an independent byte-copy snapshot of a finalized Orbax step.

The utility is intentionally independent of training and Orbax code.  A
snapshot is copied into a private staging directory, verified from the
destination, and published without replacing an existing directory.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
from typing import Any

_METADATA = "_CHECKPOINT_METADATA"
_REQUIRED_ITEMS = ("params", "train_state", "assets")
_TEMP_NAME = re.compile(r"(?:^|[._-])(tmp|temp|staging)(?:$|[._-])", re.IGNORECASE)
_STEP_NAME = re.compile(r"[0-9]+\Z", re.ASCII)
_CHUNK_SIZE = 1024 * 1024
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100
_MANIFEST_SCHEMA = "openpi-checkpoint-snapshot-v1"
_RENAME_UNSUPPORTED_ERRNOS = {errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL}


class SnapshotError(RuntimeError):
    """An input or filesystem condition made snapshotting unsafe."""


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolved(path: Path, *, label: str) -> Path:
    try:
        return path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SnapshotError(f"{label} does not resolve: {path}") from exc


def _lstat(path: Path, *, label: str) -> os.stat_result:
    try:
        result = path.lstat()
    except OSError as exc:
        raise SnapshotError(f"cannot inspect {label}: {path}") from exc
    if stat.S_ISLNK(result.st_mode):
        raise SnapshotError(f"symlink is not allowed for {label}: {path}")
    return result


def _require_directory(path: Path, *, label: str) -> os.stat_result:
    info = _lstat(path, label=label)
    if not stat.S_ISDIR(info.st_mode):
        raise SnapshotError(f"{label} is not a directory: {path}")
    return info


def _temporary_name(name: str) -> bool:
    return bool(_TEMP_NAME.search(name)) or name.lower().startswith("tmp")


def _open_source(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(path, flags)
    except OSError as exc:
        raise SnapshotError(f"cannot open source file: {path}") from exc


def _source_stat_key(info: os.stat_result) -> tuple[int, int, int, int, int]:
    """The identity and mutation fields checked around every source read."""
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _require_regular(info: os.stat_result, *, path: Path) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise SnapshotError(f"special file is not allowed: {path}")


def _safe_destination_mode(source_mode: int) -> int:
    # Checkpoint data is evidence, not an executable output.  Do not copy
    # write bits, and ensure validation can read the resulting file.
    return (stat.S_IMODE(source_mode) & ~0o222) | 0o444


def _copy_file(
    source_file: Path,
    destination_file: Path,
    expected_source: os.stat_result | None = None,
) -> dict[str, int | str]:
    """Copy one source file through an O_NOFOLLOW fd and verify its identity."""
    source_fd = _open_source(source_file)
    destination_fd: int | None = None
    try:
        before = os.fstat(source_fd)
        _require_regular(before, path=source_file)
        try:
            path_before = os.stat(source_file, follow_symlinks=False)
        except OSError as exc:
            raise SnapshotError(f"source disappeared before copy: {source_file}") from exc
        if expected_source is not None and _source_stat_key(before) != _source_stat_key(expected_source):
            raise SnapshotError(f"source changed before copy: {source_file}")
        if _source_stat_key(path_before) != _source_stat_key(before):
            raise SnapshotError(f"source changed before copy: {source_file}")

        destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        try:
            destination_fd = os.open(destination_file, destination_flags, _safe_destination_mode(before.st_mode))
            os.fchmod(destination_fd, _safe_destination_mode(before.st_mode))
        except OSError as exc:
            raise SnapshotError(f"cannot create snapshot file: {destination_file}") from exc

        digest = hashlib.sha256()
        size = 0
        while True:
            try:
                chunk = os.read(source_fd, _CHUNK_SIZE)
            except OSError as exc:
                raise SnapshotError(f"cannot read source file: {source_file}") from exc
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
            view = memoryview(chunk)
            while view:
                try:
                    written = os.write(destination_fd, view)
                except OSError as exc:
                    raise SnapshotError(f"cannot write snapshot file: {destination_file}") from exc
                if written <= 0:
                    raise SnapshotError(f"short write for snapshot file: {destination_file}")
                view = view[written:]
        try:
            os.fsync(destination_fd)
            after = os.fstat(source_fd)
            path_after = os.stat(source_file, follow_symlinks=False)
        except OSError as exc:
            raise SnapshotError(f"cannot verify source file after copy: {source_file}") from exc
        if _source_stat_key(after) != _source_stat_key(before) or _source_stat_key(path_after) != _source_stat_key(after):
            raise SnapshotError(f"source changed during copy: {source_file}")
        if size != before.st_size:
            raise SnapshotError(f"source size changed during copy: {source_file}")
        return {
            "path": source_file.name,
            "size": size,
            "sha256": digest.hexdigest(),
        }
    finally:
        if destination_fd is not None:
            with contextlib.suppress(OSError):
                os.close(destination_fd)
        with contextlib.suppress(OSError):
            os.close(source_fd)


def _scan_tree(source: Path) -> tuple[dict[str, os.stat_result], set[str]]:
    """Enumerate a checkpoint, rejecting links/special files/temp directories."""
    files: dict[str, os.stat_result] = {}
    directories: set[str] = {""}

    def visit(directory: Path, relative: str) -> None:
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise SnapshotError(f"cannot enumerate checkpoint: {directory}") from exc
        for entry in entries:
            child = Path(entry.path)
            child_relative = f"{relative}/{entry.name}" if relative else entry.name
            info = _lstat(child, label="checkpoint entry")
            if stat.S_ISDIR(info.st_mode):
                if _temporary_name(entry.name):
                    raise SnapshotError(f"temporary directory in checkpoint: {child}")
                directories.add(child_relative)
                visit(child, child_relative)
            elif stat.S_ISREG(info.st_mode):
                files[child_relative] = info
            else:
                raise SnapshotError(f"special file is not allowed: {child}")

    visit(source, "")
    return files, directories


def _read_committed_metadata(source: Path) -> tuple[dict[str, Any], int]:
    metadata_path = source / _METADATA
    source_fd = _open_source(metadata_path)
    try:
        before = os.fstat(source_fd)
        _require_regular(before, path=metadata_path)
        data = bytearray()
        while True:
            chunk = os.read(source_fd, _CHUNK_SIZE)
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(source_fd)
        if _source_stat_key(before) != _source_stat_key(after):
            raise SnapshotError(f"source metadata changed while reading: {metadata_path}")
    except OSError as exc:
        raise SnapshotError(f"cannot read {_METADATA}: {metadata_path}") from exc
    finally:
        with contextlib.suppress(OSError):
            os.close(source_fd)
    try:
        metadata = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"invalid {_METADATA}: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise SnapshotError(f"invalid {_METADATA}: expected an object")
    commit_timestamp = metadata.get("commit_timestamp_nsecs")
    if type(commit_timestamp) is not int or commit_timestamp <= 0:
        raise SnapshotError(f"{_METADATA} commit timestamp must be a positive integer")
    return metadata, commit_timestamp


def _check_source(source: Path, manager_root: Path) -> tuple[dict[str, os.stat_result], set[str], dict[str, Any], int, os.stat_result]:
    if not _STEP_NAME.fullmatch(source.name):
        raise SnapshotError(f"source basename must be a numeric finalized step: {source.name}")
    source_info = _require_directory(source, label="source")
    _require_directory(manager_root, label="manager root")
    try:
        siblings = list(os.scandir(manager_root))
    except OSError as exc:
        raise SnapshotError(f"cannot enumerate manager root: {manager_root}") from exc
    for entry in siblings:
        if entry.name == source.name:
            continue
        step_prefix = (
            entry.name.startswith(f"{source.name}.")
            or entry.name.startswith(f"{source.name}_")
            or entry.name.startswith(f"{source.name}-")
        )
        if step_prefix and _temporary_name(entry.name):
            raise SnapshotError(f"temporary step sibling is present: {entry.name}")

    files, directories = _scan_tree(source)
    for item in _REQUIRED_ITEMS:
        _require_directory(source / item, label=f"required item {item}")
    metadata, commit_timestamp = _read_committed_metadata(source)
    return files, directories, metadata, commit_timestamp, source_info


def _copy_tree(
    source: Path,
    staging: Path,
    source_files: dict[str, os.stat_result],
    source_directories: set[str],
) -> list[dict[str, int | str]]:
    copied_files: list[dict[str, int | str]] = []
    try:
        for relative in sorted(source_directories - {""}):
            (staging / relative).mkdir(mode=0o700, exist_ok=False)
        for relative in sorted(source_files):
            destination_file = staging / relative
            copied = _copy_file(source / relative, destination_file, source_files[relative])
            copied["path"] = relative
            copied["mode"] = stat.S_IMODE(_lstat(destination_file, label="snapshot file").st_mode)
            copied_files.append(copied)
        return copied_files
    except SnapshotError:
        raise
    except OSError as exc:
        raise SnapshotError(f"could not create snapshot: {exc}") from exc


def _hash_file(path: Path) -> tuple[int, str]:
    fd = _open_source(path)
    digest = hashlib.sha256()
    size = 0
    try:
        info = os.fstat(fd)
        _require_regular(info, path=path)
        while True:
            chunk = os.read(fd, _CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    except OSError as exc:
        raise SnapshotError(f"cannot hash snapshot file: {path}") from exc
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
    return size, digest.hexdigest()


def _validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise SnapshotError("manifest must be an object")
    required = {
        "schema",
        "step",
        "source",
        "destination",
        "source_metadata",
        "checkpoint_metadata",
        "commit_timestamp_nsecs",
        "files",
    }
    if set(manifest) != required:
        raise SnapshotError("manifest schema is not exact")
    if manifest["schema"] != _MANIFEST_SCHEMA or not isinstance(manifest["step"], str) or not _STEP_NAME.fullmatch(manifest["step"]):
        raise SnapshotError("manifest step or schema is invalid")
    timestamp = manifest["commit_timestamp_nsecs"]
    if type(timestamp) is not int or timestamp <= 0:
        raise SnapshotError("manifest commit timestamp must be a positive integer")
    source_metadata = manifest["source_metadata"]
    if not isinstance(source_metadata, dict) or set(source_metadata) != {"device", "inode", "size", "mtime_ns", "ctime_ns"}:
        raise SnapshotError("manifest source metadata is invalid")
    if any(type(source_metadata[key]) is not int or source_metadata[key] < 0 for key in source_metadata):
        raise SnapshotError("manifest source metadata values are invalid")
    if not isinstance(manifest["source"], str) or not manifest["source"]:
        raise SnapshotError("manifest source is invalid")
    if not isinstance(manifest["destination"], str) or not manifest["destination"]:
        raise SnapshotError("manifest destination is invalid")
    if not isinstance(manifest["checkpoint_metadata"], dict):
        raise SnapshotError("manifest checkpoint metadata is invalid")
    files = manifest["files"]
    if not isinstance(files, list):
        raise SnapshotError("manifest files must be a list")
    seen: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "size", "sha256"}:
            raise SnapshotError("manifest file entry is invalid")
        path = item["path"]
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or path.endswith("/")
            or "\\" in path
            or "//" in path
        ):
            raise SnapshotError("manifest file path is invalid")
        parts = Path(path).parts
        if any(part in {"", ".", ".."} for part in parts) or path in seen:
            raise SnapshotError("manifest file paths are invalid")
        seen.add(path)
        if type(item["size"]) is not int or item["size"] < 0:
            raise SnapshotError("manifest file size is invalid")
        if not isinstance(item["sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise SnapshotError("manifest file digest is invalid")
    if [item["path"] for item in files] != sorted(seen):
        raise SnapshotError("manifest file entries are not sorted")
    return manifest


def _validate_destination(destination: Path, manifest: dict[str, Any]) -> None:
    _validate_manifest(manifest)
    expected = {item["path"]: item for item in manifest["files"]}
    actual_files, _ = _scan_tree(destination)
    if set(actual_files) != set(expected):
        raise SnapshotError("published snapshot file set differs from manifest")
    for relative, expected_item in expected.items():
        info = _lstat(destination / relative, label="published snapshot file")
        size, digest = _hash_file(destination / relative)
        if size != expected_item["size"] or digest != expected_item["sha256"]:
            raise SnapshotError(f"published snapshot content differs from manifest: {relative}")
        if info.st_nlink != 1:
            raise SnapshotError(f"published snapshot file is shared: {relative}")


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(directory, flags)
    except OSError as exc:
        raise SnapshotError(f"cannot open directory for fsync: {directory}") from exc
    try:
        os.fsync(fd)
    except OSError as exc:
        raise SnapshotError(f"cannot fsync directory: {directory}") from exc
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename a directory/file only when destination is absent."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = libc.renameat2
    except (AttributeError, OSError):
        renameat2 = None
    if renameat2 is not None:
        renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        renameat2.restype = ctypes.c_int
        result = renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number not in _RENAME_UNSUPPORTED_ERRNOS:
            error = OSError(error_number, os.strerror(error_number), destination)
            raise SnapshotError(f"no-replace publication failed (errno={error_number}): {destination}") from error

    # Portable fallback: directories are claimed with mkdir (inherently
    # no-replace), then each staged child is moved into the claimed directory.
    # A regular file can be claimed atomically with a same-filesystem hard
    # link, followed by unlinking the staging name.  Both paths are under the
    # destination root, so the link does not cross devices.
    try:
        source_info = _lstat(source, label="publication staging path")
        if stat.S_ISDIR(source_info.st_mode):
            destination.mkdir(mode=stat.S_IMODE(source_info.st_mode), exist_ok=False)
            for child in sorted(source.iterdir(), key=lambda item: item.name):
                os.rename(child, destination / child.name)
            source.rmdir()
        elif stat.S_ISREG(source_info.st_mode):
            os.link(source, destination, follow_symlinks=False)
            source.unlink()
        else:
            raise SnapshotError(f"publication staging path is not regular: {source}")
    except FileExistsError:
        raise SnapshotError(f"final path already exists: {destination}") from None
    except OSError as exc:
        raise SnapshotError(f"could not publish without replacement: {destination}") from exc


def _write_exclusive(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    created = False
    success = False
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SnapshotError(f"manifest already exists or cannot be created: {path}") from exc
    created = True
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise SnapshotError(f"short write for manifest: {path}")
            view = view[written:]
        os.fsync(fd)
        success = True
    except OSError as exc:
        raise SnapshotError(f"cannot write manifest: {path}") from exc
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
        if created and not success:
            # A failed manifest write must not leave a partial staging file.
            # The final adjacent manifest is never removed by this routine.
            with contextlib.suppress(OSError):
                if path.name.endswith(".manifest.staging"):
                    path.unlink()


def _read_manifest(path: Path) -> dict[str, Any]:
    fd = _open_source(path)
    data = bytearray()
    try:
        while True:
            chunk = os.read(fd, _CHUNK_SIZE)
            if not chunk:
                break
            data.extend(chunk)
    except OSError as exc:
        raise SnapshotError(f"cannot read manifest: {path}") from exc
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
    try:
        return _validate_manifest(json.loads(bytes(data).decode("utf-8")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError(f"invalid manifest: {path}") from exc


def make_snapshot(source: Path, destination_root: Path, manager_root: Path | None = None) -> dict[str, Any]:
    """Create and atomically publish an independent checkpoint snapshot."""
    source_input = Path(source).expanduser()
    destination_input = Path(destination_root).expanduser()
    manager_input = Path(manager_root).expanduser() if manager_root else None
    if source_input.is_symlink() or destination_input.is_symlink() or (manager_input is not None and manager_input.is_symlink()):
        raise SnapshotError("source, manager-root, and destination-root symlinks are not allowed")
    source_path = _resolved(source_input, label="source")
    manager_path = _resolved(manager_input, label="manager root") if manager_input else source_path.parent
    if source_path == manager_path or not _is_within(source_path, manager_path):
        raise SnapshotError("source must be inside the manager root")
    destination_path = destination_input.resolve(strict=False)
    if destination_path == manager_path or _is_within(destination_path, manager_path):
        raise SnapshotError("destination-root must be outside the manager root")
    if _is_within(manager_path, destination_path) or _is_within(source_path, destination_path):
        raise SnapshotError("destination-root must not be an ancestor of manager or source")
    if destination_path.exists():
        destination_info = _require_directory(destination_path, label="destination root")
    else:
        try:
            destination_path.mkdir(parents=True, mode=0o700)
        except OSError as exc:
            raise SnapshotError(f"cannot create destination root: {destination_path}") from exc
        destination_info = _require_directory(destination_path, label="destination root")
    if destination_info.st_uid != os.getuid() or stat.S_IMODE(destination_info.st_mode) & 0o022:
        raise SnapshotError("destination root must be user-owned and not group/world writable")

    source_files, source_directories, checkpoint_metadata, commit_timestamp, source_info = _check_source(source_path, manager_path)
    step = source_path.name
    final_path = destination_path / step
    manifest_path = destination_path / f"{step}.manifest.json"
    staging_path = destination_path / f".{step}.staging"
    manifest_staging_path = destination_path / f".{step}.manifest.staging"
    lock_path = destination_path / f".{step}.lock"
    protected_paths = (final_path, manifest_path, staging_path, manifest_staging_path, lock_path)
    if any(os.path.lexists(path) for path in protected_paths):
        raise SnapshotError("final, manifest, staging, or lock path already exists")

    lock_fd: int | None = None
    lock_created = False
    staging_created = False
    manifest_staging_created = False
    try:
        try:
            lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
        except OSError as exc:
            raise SnapshotError(f"per-step lock already exists: {lock_path}") from exc
        lock_created = True
        try:
            os.write(lock_fd, f"pid={os.getpid()}\n".encode())
            os.fsync(lock_fd)
        except OSError as exc:
            raise SnapshotError(f"cannot initialize lock: {lock_path}") from exc

        try:
            staging_path.mkdir(mode=0o700, exist_ok=False)
        except OSError as exc:
            raise SnapshotError(f"cannot create staging path: {staging_path}") from exc
        staging_created = True
        copied_files = _copy_tree(source_path, staging_path, source_files, source_directories)
        current_source_files, current_source_directories = _scan_tree(source_path)
        if set(current_source_files) != set(source_files) or current_source_directories != source_directories:
            raise SnapshotError("source tree changed while snapshot was being created")
        if any(_source_stat_key(current_source_files[path]) != _source_stat_key(source_files[path]) for path in source_files):
            raise SnapshotError("source changed while snapshot was being created")
        staged_files, staged_directories = _scan_tree(staging_path)
        if set(staged_files) != set(source_files):
            raise SnapshotError("staged snapshot file set differs from source")
        expected_directories = {relative for relative in source_directories if relative}
        if staged_directories != {"", *expected_directories}:
            raise SnapshotError("staged snapshot directory set differs from source")
        if len(copied_files) != len(source_files):
            raise SnapshotError("staged snapshot file count differs from source")

        manifest = _validate_manifest(
            {
                "schema": _MANIFEST_SCHEMA,
                "step": step,
                "source": str(source_path),
                "destination": str(final_path),
                "source_metadata": {
                    "device": source_info.st_dev,
                    "inode": source_info.st_ino,
                    "size": source_info.st_size,
                    "mtime_ns": source_info.st_mtime_ns,
                    "ctime_ns": source_info.st_ctime_ns,
                },
                "checkpoint_metadata": checkpoint_metadata,
                "commit_timestamp_nsecs": commit_timestamp,
                "files": sorted(
                    ({"path": item["path"], "size": item["size"], "sha256": item["sha256"]} for item in copied_files),
                    key=lambda item: item["path"],
                ),
            }
        )
        manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        _fsync_directory(staging_path)
        _write_exclusive(manifest_staging_path, manifest_bytes)
        manifest_staging_created = True
        _fsync_directory(destination_path)
        if any(os.path.lexists(path) for path in (final_path, manifest_path)):
            raise SnapshotError("final or manifest appeared during snapshot creation")
        _rename_noreplace(staging_path, final_path)
        staging_created = False
        _rename_noreplace(manifest_staging_path, manifest_path)
        manifest_staging_created = False
        _fsync_directory(destination_path)
        published_manifest = _read_manifest(manifest_path)
        _validate_destination(final_path, published_manifest)
        return manifest
    except Exception:
        if staging_created:
            with contextlib.suppress(OSError):
                shutil.rmtree(staging_path)
        if manifest_staging_created:
            with contextlib.suppress(OSError):
                manifest_staging_path.unlink()
        raise
    finally:
        if lock_fd is not None:
            with contextlib.suppress(OSError):
                os.close(lock_fd)
        if lock_created:
            with contextlib.suppress(OSError):
                lock_path.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="finalized numeric Orbax step directory")
    parser.add_argument("--destination-root", type=Path, required=True, help="exclusive root outside manager/source")
    parser.add_argument("--manager-root", type=Path, help="optional checkpoint manager root (defaults to source parent)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        manifest = make_snapshot(args.source, args.destination_root, args.manager_root)
    except (SnapshotError, OSError, ValueError) as exc:
        print(f"snapshot failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
