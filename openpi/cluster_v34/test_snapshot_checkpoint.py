from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path

import pytest

import cluster_v34.snapshot_checkpoint as snapshot
from cluster_v34.snapshot_checkpoint import SnapshotError
from cluster_v34.snapshot_checkpoint import main
from cluster_v34.snapshot_checkpoint import make_snapshot


def _checkpoint(root: Path, step: str = "42", *, commit_timestamp: object = 123) -> Path:
    source = root / "manager" / step
    for item in ("params", "train_state", "assets"):
        (source / item).mkdir(parents=True)
    (source / "params" / "weights").write_bytes(b"weights")
    (source / "train_state" / "state").write_bytes(b"state")
    (source / "assets" / "vocab").write_bytes(b"assets")
    metadata = {"commit_timestamp_nsecs": commit_timestamp, "custom_metadata": {"test": "yes"}}
    (source / "_CHECKPOINT_METADATA").write_text(json.dumps(metadata), encoding="utf-8")
    return source


def _destination_files(destination: Path) -> dict[str, bytes]:
    return {str(path.relative_to(destination)): path.read_bytes() for path in destination.rglob("*") if path.is_file()}


def test_happy_path_has_independent_files_and_strict_adjacent_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = _checkpoint(tmp_path)
    destination_root = tmp_path / "snapshots"

    assert main(["--source", str(source), "--destination-root", str(destination_root)]) == 0
    output = json.loads(capsys.readouterr().out)
    destination = destination_root / "42"
    manifest_path = destination_root / "42.manifest.json"
    assert Path(output["destination"]) == destination.resolve()
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == output
    assert manifest["commit_timestamp_nsecs"] == 123
    assert manifest["checkpoint_metadata"]["custom_metadata"] == {"test": "yes"}
    assert [entry["path"] for entry in manifest["files"]] == sorted(entry["path"] for entry in manifest["files"])
    for relative in ("_CHECKPOINT_METADATA", "params/weights", "train_state/state", "assets/vocab"):
        source_stat = (source / relative).stat()
        destination_stat = (destination / relative).stat()
        assert (source_stat.st_dev, source_stat.st_ino) != (destination_stat.st_dev, destination_stat.st_ino)
        assert destination_stat.st_nlink == 1
        assert destination_stat.st_size == source_stat.st_size
        assert hashlib.sha256((destination / relative).read_bytes()).hexdigest() == next(
            entry["sha256"] for entry in manifest["files"] if entry["path"] == relative
        )


def test_destination_mutation_does_not_affect_source(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    destination = Path(make_snapshot(source, tmp_path / "snapshots")["destination"])
    source_file = source / "params" / "weights"
    source_before = source_file.read_bytes()
    os.chmod(destination / "params" / "weights", 0o644)
    (destination / "params" / "weights").write_bytes(b"changed destination")
    assert source_file.read_bytes() == source_before


@pytest.mark.parametrize("existing_name", ["42", "42.manifest.json", ".42.staging", ".42.manifest.staging", ".42.lock"])
def test_existing_final_manifest_staging_or_lock_is_refused(tmp_path: Path, existing_name: str) -> None:
    source = _checkpoint(tmp_path)
    destination_root = tmp_path / "snapshots"
    destination_root.mkdir()
    existing = destination_root / existing_name
    if existing_name.endswith(".staging") or existing_name == "42":
        existing.mkdir()
    else:
        existing.write_text("existing", encoding="utf-8")
    with pytest.raises(SnapshotError):
        make_snapshot(source, destination_root)
    assert existing.exists()


@pytest.mark.parametrize("commit_timestamp", [None, 0, -1, 1.5, "123", True])
def test_malformed_commit_timestamp_is_refused(tmp_path: Path, commit_timestamp: object) -> None:
    source = _checkpoint(tmp_path, commit_timestamp=commit_timestamp)
    with pytest.raises(SnapshotError):
        make_snapshot(source, tmp_path / "snapshots")


def test_missing_required_item_or_temporary_directory_is_refused(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    (source / "assets" / "vocab").unlink()
    (source / "assets").rmdir()
    with pytest.raises(SnapshotError):
        make_snapshot(source, tmp_path / "snapshots")

    source = _checkpoint(tmp_path / "second")
    (source / "params" / "tmp").mkdir()
    with pytest.raises(SnapshotError):
        make_snapshot(source, tmp_path / "second-snapshots")


def test_destination_equal_inside_or_ancestor_is_refused(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path)
    with pytest.raises(SnapshotError):
        make_snapshot(source, source.parent)
    with pytest.raises(SnapshotError):
        make_snapshot(source, source / "nested")
    with pytest.raises(SnapshotError):
        make_snapshot(source, tmp_path)


def test_copy_failure_cleans_only_staging_and_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _checkpoint(tmp_path)
    destination_root = tmp_path / "snapshots"
    original_copy = snapshot._copy_file  # noqa: SLF001

    def fail_copy(*args: object, **kwargs: object) -> dict[str, int | str]:
        raise SnapshotError("injected copy failure")

    monkeypatch.setattr(snapshot, "_copy_file", fail_copy)
    with pytest.raises(SnapshotError):
        make_snapshot(source, destination_root)
    assert not (destination_root / "42").exists()
    assert not (destination_root / ".42.staging").exists()
    assert not (destination_root / ".42.manifest.staging").exists()
    assert not (destination_root / ".42.lock").exists()
    monkeypatch.setattr(snapshot, "_copy_file", original_copy)
    assert _destination_files(source) == {
        "_CHECKPOINT_METADATA": (source / "_CHECKPOINT_METADATA").read_bytes(),
        "params/weights": b"weights",
        "train_state/state": b"state",
        "assets/vocab": b"assets",
    }


def test_source_mutation_during_copy_is_refused_and_cleaned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _checkpoint(tmp_path)
    target = source / "params" / "weights"
    original_copy = snapshot._copy_file  # noqa: SLF001
    mutated = False

    def mutate_copy(source_file: Path, destination_file: Path, expected: os.stat_result | None = None) -> dict[str, int | str]:
        nonlocal mutated
        if not mutated and source_file == target:
            mutated = True
            target.write_bytes(b"mutated while copying")
        return original_copy(source_file, destination_file, expected)

    monkeypatch.setattr(snapshot, "_copy_file", mutate_copy)
    with pytest.raises(SnapshotError):
        make_snapshot(source, tmp_path / "snapshots")
    assert not (tmp_path / "snapshots" / "42").exists()
    assert not (tmp_path / "snapshots" / ".42.staging").exists()
    assert not (tmp_path / "snapshots" / ".42.lock").exists()


def test_source_deletion_after_publication_still_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _checkpoint(tmp_path)
    original_validate = snapshot._validate_destination  # noqa: SLF001
    deleted = False

    def delete_source_after_publish(destination: Path, manifest: dict[str, object]) -> None:
        nonlocal deleted
        if not deleted:
            deleted = True
            for child in sorted(source.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            source.rmdir()
        original_validate(destination, manifest)

    monkeypatch.setattr(snapshot, "_validate_destination", delete_source_after_publish)
    result = make_snapshot(source, tmp_path / "snapshots")
    assert deleted
    assert Path(result["destination"]).exists()
    assert Path(result["destination"] + ".manifest.json").exists()


def test_final_race_cannot_overwrite_existing_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _checkpoint(tmp_path)
    destination_root = tmp_path / "snapshots"
    original_publish = snapshot._rename_noreplace  # noqa: SLF001
    raced = False

    def race_publish(staging: Path, final: Path) -> None:
        nonlocal raced
        if not raced and final.name == "42":
            raced = True
            final.mkdir()
            (final / "sentinel").write_text("keep", encoding="utf-8")
        original_publish(staging, final)

    monkeypatch.setattr(snapshot, "_rename_noreplace", race_publish)
    with pytest.raises(SnapshotError):
        make_snapshot(source, destination_root)
    assert (destination_root / "42" / "sentinel").read_text(encoding="utf-8") == "keep"
    assert not (destination_root / ".42.staging").exists()
    assert not (destination_root / ".42.lock").exists()


@pytest.mark.parametrize("unsupported_errno", [errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EINVAL])
def test_rename_noreplace_falls_back_for_unsupported_errno(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsupported_errno: int
) -> None:
    source = tmp_path / "staging"
    source.mkdir()
    (source / "payload").write_text("payload", encoding="utf-8")
    destination = tmp_path / "final"

    class FakeRename:
        argtypes: object
        restype: object

        def __call__(self, *_args: object) -> int:
            snapshot.ctypes.set_errno(unsupported_errno)
            return -1

    class FakeLibc:
        renameat2 = FakeRename()

    monkeypatch.setattr(snapshot.ctypes, "CDLL", lambda *_args, **_kwargs: FakeLibc())
    snapshot._rename_noreplace(source, destination)  # noqa: SLF001
    assert (destination / "payload").read_text(encoding="utf-8") == "payload"
    assert not source.exists()


def test_rename_noreplace_does_not_fall_back_for_eexist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "staging"
    source.mkdir()
    (source / "payload").write_text("payload", encoding="utf-8")
    destination = tmp_path / "final"
    destination.mkdir()
    (destination / "sentinel").write_text("keep", encoding="utf-8")

    class FakeRename:
        argtypes: object
        restype: object

        def __call__(self, *_args: object) -> int:
            snapshot.ctypes.set_errno(errno.EEXIST)
            return -1

    class FakeLibc:
        renameat2 = FakeRename()

    monkeypatch.setattr(snapshot.ctypes, "CDLL", lambda *_args, **_kwargs: FakeLibc())
    with pytest.raises(SnapshotError, match=r"errno=17"):
        snapshot._rename_noreplace(source, destination)  # noqa: SLF001
    assert (destination / "sentinel").read_text(encoding="utf-8") == "keep"
    assert source.exists()


def test_rename_noreplace_file_fallback_is_atomic_and_no_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "manifest.staging"
    source.write_text("manifest", encoding="utf-8")
    destination = tmp_path / "manifest.json"

    class FakeRename:
        argtypes: object
        restype: object

        def __call__(self, *_args: object) -> int:
            snapshot.ctypes.set_errno(errno.EOPNOTSUPP)
            return -1

    class FakeLibc:
        renameat2 = FakeRename()

    monkeypatch.setattr(snapshot.ctypes, "CDLL", lambda *_args, **_kwargs: FakeLibc())
    snapshot._rename_noreplace(source, destination)  # noqa: SLF001
    assert destination.read_text(encoding="utf-8") == "manifest"
    assert not source.exists()
    assert destination.stat().st_nlink == 1

    source = tmp_path / "manifest.staging-2"
    source.write_text("new", encoding="utf-8")

    class ExistingRename:
        argtypes: object
        restype: object

        def __call__(self, *_args: object) -> int:
            snapshot.ctypes.set_errno(errno.EEXIST)
            return -1

    class ExistingLibc:
        renameat2 = ExistingRename()

    monkeypatch.setattr(snapshot.ctypes, "CDLL", lambda *_args, **_kwargs: ExistingLibc())
    with pytest.raises(SnapshotError, match=r"errno=17"):
        snapshot._rename_noreplace(source, destination)  # noqa: SLF001
    assert source.read_text(encoding="utf-8") == "new"
    assert destination.read_text(encoding="utf-8") == "manifest"


def test_non_numeric_step_is_refused(tmp_path: Path) -> None:
    source = _checkpoint(tmp_path, step="42.tmp")
    with pytest.raises(SnapshotError):
        make_snapshot(source, tmp_path / "snapshots")
