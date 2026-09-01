# ruff: noqa: PLR2004

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import build_v35_transfer_bundle as transfer
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


def _json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _fixture(tmp_path: Path, *, with_norm: bool = True) -> tuple[transfer.BundleInputs, list[tuple[Path, bytes]]]:
    project = tmp_path / "project"
    dataset = project / transfer.PROJECT_DATASET_PREFIX / transfer.DATASET_REPO_ID
    historical_dataset = tmp_path / "legacy-huggingface-cache" / transfer.DATASET_REPO_ID
    data_dir = project / "data"
    approval_dir = project / transfer.DEFAULT_APPROVAL_PROVENANCE_DIR
    verification_path = project / transfer.DEFAULT_APPROVED_VERIFICATION
    dataset_meta = dataset / "meta"
    data_dir.mkdir(parents=True)
    approval_dir.mkdir(parents=True)
    dataset_meta.mkdir(parents=True)

    records: list[dict[str, object]] = []
    label_sources: list[tuple[Path, bytes]] = []
    for index in range(transfer.EXPECTED_RAW_RECORDS):
        included = index < transfer.EXPECTED_INCLUDED_EPISODES
        stable_id = f"fixture/demo{index:03d}"
        raw_dir = f"data/fixture/demo{index:03d}"
        label_path = project / raw_dir / "labels.json"
        if included:
            label_bytes = (
                json.dumps(
                    [{"end": phase, "start": phase, "task": f"phase {phase}"} for phase in range(5)],
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
            label_path.parent.mkdir(parents=True)
            label_path.write_bytes(label_bytes)
            label_sha256 = hashlib.sha256(label_bytes).hexdigest()
            label_sources.append((label_path, label_bytes))
        else:
            label_sha256 = "f" * 64
        record: dict[str, object] = {
            "include": included,
            "label_file": "labels.json",
            "label_sha256": label_sha256,
            "raw_dir": raw_dir,
            "stable_id": stable_id,
        }
        if included:
            record["episode_index"] = index
            if index < 54:
                record["split"] = "train"
            elif index < 62:
                record["split"] = "development"
            else:
                record["split"] = "final_test"
        records.append(record)

    approved_path = project / transfer.DEFAULT_APPROVED_MANIFEST
    approved = {
        "dataset_version": "v36",
        "episodes": records,
        "raw_root": "..",
        "review_status": "user_approved",
        "schema_version": 1,
    }
    approved_sha256 = _json(approved_path, approved)

    ledger_path = approval_dir / "approval_ledger.json"
    ledger = {
        "approved_manifest": str(approved_path.resolve()),
        "entries": [
            {
                "decision": "user_approved" if record["include"] else "user_approved_exclusion",
                "include": record["include"],
                "label_sha256": record["label_sha256"],
                "stable_id": record["stable_id"],
            }
            for record in records
        ],
        "reviewer": "user",
        "schema_version": "openpi.dataset_approval_ledger.v1",
    }
    ledger_sha256 = _json(ledger_path, ledger)

    included_ids = [str(record["stable_id"]) for record in records if record["include"]]
    block_path = data_dir / "frozen_block.json"
    visibility_path = data_dir / "frozen_visibility.json"
    d_valid_path = data_dir / "frozen_d_valid.json"
    block_sha256 = _json(block_path, {"status": "pass"})
    visibility_sha256 = _json(
        visibility_path,
        {"episodes": [{"stable_id": stable_id} for stable_id in included_ids], "status": "user_approved"},
    )
    d_valid_sha256 = _json(
        d_valid_path,
        {"episodes": [{"stable_id": stable_id} for stable_id in included_ids], "status": "complete"},
    )
    frozen_path = project / transfer.DEFAULT_FROZEN_MANIFEST
    frozen = {
        "approval_ledger_sha256": ledger_sha256,
        "block_confound_audit": {"report_file": block_path.name, "report_sha256": block_sha256},
        "d_valid_sidecar": {"report_file": d_valid_path.name, "report_sha256": d_valid_sha256},
        "dataset_version": "v36",
        "e_visibility_review": {"report_file": visibility_path.name, "report_sha256": visibility_sha256},
        "episodes": records,
        "raw_root": "..",
        "review_status": "frozen",
        "schema_version": 2,
        "source_approved_manifest": {"file": approved_path.name, "sha256": approved_sha256},
        "split_seed": 36,
    }
    frozen_sha256 = _json(frozen_path, frozen)

    info_path = dataset_meta / "info.json"
    info_sha256 = _json(info_path, {"total_episodes": 70})
    meta_sha256 = {"info.json": info_sha256}
    inventory_path = approval_dir / "parquet_payload_inventory.json"
    inventory_files = [
        {
            "path": f"data/chunk-000/episode_{index:06d}.parquet",
            "sha256": _digest_text(f"parquet-{index}"),
            "size_bytes": index + 1,
        }
        for index in range(70)
    ]
    inventory = {
        "dataset": str(historical_dataset.resolve()),
        "file_count": 70,
        "files": inventory_files,
        "schema_version": "openpi.parquet_payload_inventory.v1",
        "total_bytes": sum(record["size_bytes"] for record in inventory_files),
    }
    inventory_sha256 = _json(inventory_path, inventory)
    _json(
        approval_dir / "conversion_report_preapproval.json",
        {"included_episodes": 70, "repo_name": transfer.DATASET_REPO_ID, "schema_version": 1},
    )
    verification = {
        "dataset": str(historical_dataset.resolve()),
        "episodes": 70,
        "manifest": str(approved_path.resolve()),
        "manifest_sha256": approved_sha256,
        "meta_sha256": meta_sha256,
        "schema_version": 2,
        "status": "passed",
    }
    verification_sha256 = _json(verification_path, verification)
    release = {
        "approval_ledger": str(ledger_path.resolve()),
        "approval_ledger_sha256": ledger_sha256,
        "approved_manifest": str(approved_path.resolve()),
        "approved_manifest_sha256": approved_sha256,
        "dataset": str(historical_dataset.resolve()),
        "episodes": 70,
        "meta_sha256": meta_sha256,
        "parquet_payload_bytes": inventory["total_bytes"],
        "parquet_payload_files": 70,
        "parquet_payload_inventory": str(inventory_path.resolve()),
        "parquet_payload_inventory_sha256": inventory_sha256,
        "schema_version": "openpi.dataset_release.v1",
        "status": "user_approved_data_ready_for_training",
        "verification": str(verification_path.resolve()),
        "verification_sha256": verification_sha256,
    }
    _json(approval_dir / "release.json", release)

    norm_dir: Path | None = None
    if with_norm:
        norm_dir = project / transfer.DEFAULT_NORM_RESTORE_DIR
        norm_stats_path = norm_dir / "norm_stats.json"
        norm_sha256 = _json(norm_stats_path, {"state": {"mean": [0.0], "std": [1.0]}})
        train_records = records[:54]
        storage_files = sorted(
            [
                {
                    "path": f"data/chunk-000/episode_{index:06d}.parquet",
                    "sha256": _digest_text(f"train-parquet-{index}"),
                    "size": index + 1,
                }
                for index in range(54)
            ]
            + [
                {
                    "path": f"videos/chunk-000/camera/episode_{index:06d}.mp4",
                    "sha256": _digest_text(f"train-video-{index}"),
                    "size": index + 1,
                }
                for index in range(54)
            ]
            + [{"path": "meta/info.json", "sha256": info_sha256, "size": info_path.stat().st_size}],
            key=lambda record: record["path"],
        )
        frame_counts = [2] * 54
        provenance = {
            "schema_version": 2,
            "computation": {
                "drop_last_rows": 0,
                "num_batches_including_partial_final_batch": 7,
                "processed_base_rows": sum(frame_counts),
                "protocol": transfer.NORM_COMPUTATION_PROTOCOL,
                "requested_batch_size": 16,
            },
            "manifest": {
                "active_split": "train",
                "path_relative": transfer.DEFAULT_FROZEN_MANIFEST.as_posix(),
                "sha256": frozen_sha256,
                "split_seed": 36,
            },
            "norm_stats": {"file": "norm_stats.json", "sha256": norm_sha256},
            "repo_id": transfer.DATASET_REPO_ID,
            "selection": {
                "dataset_num_episodes": 70,
                "selected_episode_frame_counts": frame_counts,
                "selected_episode_indices": list(range(54)),
                "selected_num_episodes": 54,
                "selected_num_frames": sum(frame_counts),
                "selected_stable_ids": [str(record["stable_id"]) for record in train_records],
            },
            "status": "complete",
            "train_storage": {
                "files": storage_files,
                "root_contract": transfer.TRAIN_STORAGE_ROOT_CONTRACT,
                "root_relative": f"data/lerobot/{transfer.DATASET_REPO_ID}",
                "scope": transfer.TRAIN_STORAGE_SCOPE,
                "selected_episode_indices": list(range(54)),
                "sha256": hashlib.sha256(transfer.canonical_json_bytes(storage_files)).hexdigest(),
            },
        }
        _json(norm_dir / "norm_stats_provenance.json", provenance)

    return (
        transfer.BundleInputs(
            dataset_root=dataset,
            source_project_root=project,
            frozen_manifest=frozen_path,
            approved_manifest=approved_path,
            approval_provenance_dir=approval_dir,
            approved_verification=verification_path,
            norm_dir=norm_dir,
        ),
        label_sources,
    )


def test_builds_byte_exact_self_contained_bundle_and_refuses_overwrite(tmp_path: Path) -> None:
    inputs, labels = _fixture(tmp_path)
    plan = transfer.plan_bundle(inputs)
    assert not plan.bundle_root.exists()
    assert plan.transfer_manifest["population"]["splits"] == transfer.EXPECTED_SPLITS
    assert plan.transfer_manifest["norm_assets"]["included"] is True

    result = transfer.build_bundle(plan)
    assert result["label_count"] == 70
    assert result["norm_assets_included"] is True
    bundle_root = inputs.dataset_root / transfer.BUNDLE_NAME
    assert bundle_root.parent == inputs.dataset_root
    assert not (inputs.dataset_root / "meta" / transfer.BUNDLE_NAME).exists()
    transfer_manifest_path = bundle_root / transfer.TRANSFER_MANIFEST_NAME
    transfer_manifest = json.loads(transfer_manifest_path.read_bytes())
    assert transfer_manifest_path.read_bytes() == transfer.canonical_json_bytes(transfer_manifest) + b"\n"
    assert transfer_manifest["norm_assets"]["restore_directory"] == (
        "restore_tree/project_root/v35/assets/pi05_yam_0830_0831_v36/yam/bin_memory_0830_0831_v36_subtask"
    )
    assert [record["path"] for record in transfer_manifest["files"]] == sorted(
        record["path"] for record in transfer_manifest["files"]
    )
    copied_label = bundle_root / "restore_tree/project_root/data/fixture/demo000/labels.json"
    assert copied_label.read_bytes() == labels[0][1]
    assert copied_label.stat().st_ino != labels[0][0].stat().st_ino
    assert not any(path.suffix in transfer.FORBIDDEN_STORAGE_SUFFIXES for path in bundle_root.rglob("*"))
    readme = (bundle_root / transfer.README_NAME).read_text(encoding="utf-8")
    assert f"<PROJECT_ROOT>/data/lerobot/{transfer.DATASET_REPO_ID}" in readme
    assert "train_storage.root_relative=data/lerobot/" in readme
    assert "canonical absolute path" not in readme
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        transfer.build_bundle(plan)


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        (None, "schema_version", 1, "exact frozen 54-episode train split"),
        ("manifest", "path", "/legacy/project/manifest.json", "exact frozen 54-episode train split"),
        ("manifest", "path_relative", "data/wrong.json", "exact frozen 54-episode train split"),
        ("computation", "protocol", "legacy", "every selected train frame"),
        ("train_storage", "root", "/legacy/cache/dataset", "portable project-relative contract"),
        ("train_storage", "root_contract", "legacy", "portable project-relative contract"),
        ("train_storage", "root_relative", "data/lerobot/yam/wrong", "portable project-relative contract"),
        ("train_storage", "scope", "parquet only", "portable project-relative contract"),
    ],
)
def test_norm_provenance_portable_contract_fails_closed(
    tmp_path: Path,
    section: str | None,
    key: str,
    value: object,
    match: str,
) -> None:
    inputs, _labels = _fixture(tmp_path)
    assert inputs.norm_dir is not None
    provenance_path = inputs.norm_dir / "norm_stats_provenance.json"
    provenance = json.loads(provenance_path.read_bytes())
    target = provenance if section is None else provenance[section]
    target[key] = value
    _json(provenance_path, provenance)
    with pytest.raises(transfer.BundleError, match=match):
        transfer.plan_bundle(inputs)


def test_cli_dry_run_validates_without_writing(tmp_path: Path) -> None:
    inputs, _labels = _fixture(tmp_path, with_norm=False)
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPTS_DIR / "build_v35_transfer_bundle.py"),
            "build",
            "--dataset-root",
            str(inputs.dataset_root),
            "--source-project-root",
            str(inputs.source_project_root),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["dry_run"] is True
    assert result["label_count"] == 70
    assert result["norm_assets_included"] is False
    assert not (inputs.dataset_root / transfer.BUNDLE_NAME).exists()


def test_source_label_hash_and_split_counts_fail_closed(tmp_path: Path) -> None:
    inputs, labels = _fixture(tmp_path / "label", with_norm=False)
    labels[0][0].write_bytes(labels[0][1] + b" ")
    with pytest.raises(transfer.BundleError, match="label bytes do not match"):
        transfer.plan_bundle(inputs)

    inputs, _labels = _fixture(tmp_path / "split", with_norm=False)
    frozen = json.loads(inputs.frozen_manifest.read_bytes())
    frozen["episodes"][0]["split"] = "development"
    _json(inputs.frozen_manifest, frozen)
    with pytest.raises(transfer.BundleError, match="54/8/8"):
        transfer.plan_bundle(inputs)


def test_verifier_rejects_tampered_payload(tmp_path: Path) -> None:
    inputs, _labels = _fixture(tmp_path, with_norm=False)
    plan = transfer.plan_bundle(inputs)
    transfer.build_bundle(plan)
    copied_label = (
        inputs.dataset_root / transfer.BUNDLE_NAME / "restore_tree/project_root/data/fixture/demo000/labels.json"
    )
    copied_label.write_bytes(copied_label.read_bytes() + b" ")
    with pytest.raises(transfer.BundleError, match="size mismatch|hash mismatch"):
        transfer.verify_bundle(inputs.dataset_root / transfer.BUNDLE_NAME)
