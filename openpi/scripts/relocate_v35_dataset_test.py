# ruff: noqa: PLR2004

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import sys

import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import relocate_v35_dataset as relocation
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


def _source_fixture(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    payloads = {
        "data/chunk-000/episode_000000.parquet": b"episode-zero" * 101,
        "data/chunk-000/episode_000001.parquet": b"episode-one" * 79,
        "meta/info.json": b'{"total_episodes":2}\n',
        "meta/tasks.jsonl": b'{"task":"test"}\n',
        "v35_training_bundle/README.md": b"fixture\n",
    }
    for relative, payload in payloads.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (source / "empty-directory").mkdir()
    return source


def _inventory(source: Path) -> relocation.TreeInventory:
    return relocation.build_inventory(source, enforce_v35_identity=False)


def _partial(destination_file: Path) -> Path:
    return destination_file.parent / f".{destination_file.name}{relocation.PARTIAL_SUFFIX}"


def test_inventory_is_canonical_deterministic_and_portable(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    first = _inventory(source)
    second = _inventory(source)
    path = tmp_path / "inventory.json"

    relocation.write_inventory(path, first)
    relocation.write_inventory(path, second)

    assert first == second
    assert relocation.load_inventory(path) == first
    assert path.read_bytes() == first.canonical_bytes()
    raw = json.loads(path.read_bytes())
    assert raw["dataset_repo_id"] == relocation.DATASET_REPO_ID
    assert raw["tree_sha256"] == first.tree_sha256
    assert str(source) not in path.read_text(encoding="utf-8")


def test_copy_is_exact_and_source_is_never_modified(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    destination = tmp_path / "portable" / "data" / "lerobot" / relocation.DATASET_REPO_ID
    inventory_path = tmp_path / "portable" / "data" / "inventory.json"
    before = {
        item.relative_path: (item.identity.mode, item.identity.mtime_ns, item.absolute_path.read_bytes())
        for item in relocation.scan_tree(source).files
    }

    actual = relocation.copy_dataset(
        source,
        destination,
        inventory_output=inventory_path,
        enforce_v35_identity=False,
    )

    assert actual is not None
    assert relocation.load_inventory(inventory_path) == actual
    relocation.verify_tree(destination, actual, enforce_v35_identity=False)
    assert relocation.scan_tree(source).directories == relocation.scan_tree(destination).directories
    assert not list(destination.rglob(f"*{relocation.PARTIAL_SUFFIX}"))
    after_scan = relocation.scan_tree(source)
    for item in after_scan.files:
        mode, mtime_ns, payload = before[item.relative_path]
        assert item.identity.mode == mode
        assert item.identity.mtime_ns == mtime_ns
        assert item.absolute_path.read_bytes() == payload


def test_copy_resumes_matching_partial_and_skips_matching_final(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    destination = tmp_path / "destination"
    first_source = source / "data/chunk-000/episode_000000.parquet"
    second_source = source / "data/chunk-000/episode_000001.parquet"
    first_destination = destination / "data/chunk-000/episode_000000.parquet"
    second_destination = destination / "data/chunk-000/episode_000001.parquet"
    first_destination.parent.mkdir(parents=True)
    first_destination.write_bytes(first_source.read_bytes())
    partial = _partial(second_destination)
    second_bytes = second_source.read_bytes()
    partial.write_bytes(second_bytes[:137])

    inventory = relocation.copy_dataset(source, destination, enforce_v35_identity=False)

    assert inventory is not None
    assert first_destination.read_bytes() == first_source.read_bytes()
    assert second_destination.read_bytes() == second_bytes
    assert not partial.exists()
    relocation.verify_tree(destination, inventory, enforce_v35_identity=False)


@pytest.mark.parametrize("kind", ["completed", "partial"])
def test_copy_refuses_mismatched_existing_payload_without_overwrite(tmp_path: Path, kind: str) -> None:
    source = _source_fixture(tmp_path)
    destination = tmp_path / "destination"
    source_file = source / "data/chunk-000/episode_000000.parquet"
    destination_file = destination / "data/chunk-000/episode_000000.parquet"
    destination_file.parent.mkdir(parents=True)
    bad = b"x" * len(source_file.read_bytes())
    target = destination_file if kind == "completed" else _partial(destination_file)
    target.write_bytes(bad if kind == "completed" else bad[:211])

    with pytest.raises(relocation.RelocationError, match="mismatch|does not match"):
        relocation.copy_dataset(source, destination, enforce_v35_identity=False)

    assert target.read_bytes() == (bad if kind == "completed" else bad[:211])
    assert source_file.read_bytes() != bad


def test_copy_refuses_unexpected_destination_file_before_writing(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    destination = tmp_path / "destination"
    unexpected = destination / "foreign.txt"
    unexpected.parent.mkdir(parents=True)
    unexpected.write_text("do not touch", encoding="utf-8")

    with pytest.raises(relocation.RelocationError, match="unexpected files"):
        relocation.copy_dataset(source, destination, enforce_v35_identity=False)

    assert unexpected.read_text(encoding="utf-8") == "do not touch"
    assert len(relocation.scan_tree(destination).files) == 1


def test_copy_dry_run_performs_no_destination_writes(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    destination = tmp_path / "absent" / "dataset"

    result = relocation.copy_dataset(source, destination, enforce_v35_identity=False, dry_run=True)

    assert result is None
    assert not destination.exists()


def test_expected_inventory_detects_source_mutation(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    expected = _inventory(source)
    mutated = source / "meta/tasks.jsonl"
    original = mutated.read_bytes()
    mutated.write_bytes(b"x" * len(original))

    with pytest.raises(relocation.RelocationError, match="expected inventory"):
        relocation.copy_dataset(
            source,
            tmp_path / "destination",
            expected_inventory=expected,
            enforce_v35_identity=False,
        )


def test_inventory_refuses_noncanonical_or_overwrite(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    inventory = _inventory(source)
    path = tmp_path / "inventory.json"
    path.write_text(json.dumps(inventory.as_dict(), indent=2), encoding="utf-8")

    with pytest.raises(relocation.RelocationError, match="not canonical"):
        relocation.load_inventory(path)
    with pytest.raises(relocation.RelocationError, match="refusing to overwrite"):
        relocation.write_inventory(path, inventory)


def test_verify_detects_destination_corruption(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    destination = tmp_path / "destination"
    inventory = relocation.copy_dataset(source, destination, enforce_v35_identity=False)
    assert inventory is not None
    target = destination / "meta/tasks.jsonl"
    payload = target.read_bytes()
    target.write_bytes(b"z" * len(payload))

    with pytest.raises(relocation.RelocationError, match="SHA256 mismatch"):
        relocation.verify_tree(destination, inventory, enforce_v35_identity=False)


def test_source_and_destination_symlinks_fail_closed(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    source_link = source / "meta/link.json"
    source_link.symlink_to(outside)

    with pytest.raises(relocation.RelocationError, match="symlink"):
        relocation.copy_dataset(source, tmp_path / "destination", enforce_v35_identity=False)

    source_link.unlink()
    destination = tmp_path / "destination"
    destination.symlink_to(source, target_is_directory=True)
    with pytest.raises(relocation.RelocationError, match="different|nested|symlink"):
        relocation.copy_dataset(source, destination, enforce_v35_identity=False)


def test_default_paths_are_project_relative_and_match_runtime_contract() -> None:
    root = relocation.memory_project_root()

    assert relocation.default_destination() == root / "data/lerobot/yam/bin_memory_0830_0831_v36_subtask"
    assert relocation.default_inventory_path() == root / "data/0830_0831_v36_dataset_tree_inventory.json"
    assert not relocation.DATASET_RELATIVE_PATH.is_absolute()
    assert ".." not in relocation.DATASET_RELATIVE_PATH.parts


def test_fixture_hash_expectation_is_meaningful(tmp_path: Path) -> None:
    source = _source_fixture(tmp_path)
    inventory = _inventory(source)
    target = source / "meta/info.json"
    expected = next(item for item in inventory.files if item.path.as_posix() == "meta/info.json")

    assert expected.sha256 == hashlib.sha256(target.read_bytes()).hexdigest()
    assert stat.S_ISREG(target.stat().st_mode)
    assert os.path.commonpath((source, target)) == str(source)
