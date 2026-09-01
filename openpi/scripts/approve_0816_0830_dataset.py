"""Freeze the user-approved 0816+0830 labels and promote the verified dataset metadata.

Promotion binds every included label and all seven raw streams to an approved manifest, updates
only conversion_report.json, and then expects the full verifier to be rerun.  Finalization also
seals every Parquet payload file by path, size, and SHA-256.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

STREAMS = (
    "left_joint_positions.npy",
    "right_joint_positions.npy",
    "left_control.npy",
    "right_control.npy",
    "top_camera_rgb.mp4",
    "left_camera_rgb.mp4",
    "right_camera_rgb.mp4",
)
EXPECTED_EPISODES = 90
EXPECTED_FRAMES = 77_445
EXPECTED_TASKS = 7
EXPECTED_META_FILES = {
    "conversion_report.json",
    "episode_prompts.json",
    "episode_sources.json",
    "episodes.jsonl",
    "episodes_stats.jsonl",
    "info.json",
    "memory_waiting_cores.json",
    "tasks.jsonl",
}
APPROVAL_MESSAGE = "ok i think this label is correct and ready for training"


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        raise AssertionError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _seal_payload(dataset: Path, output_path: Path) -> dict[str, Any]:
    files = sorted((dataset / "data").glob("chunk-*/episode_*.parquet"))
    _require(len(files) == EXPECTED_EPISODES, f"Parquet payload count is {len(files)}, expected 90")
    records = []
    for expected_episode, path in enumerate(files):
        expected_name = f"episode_{expected_episode:06d}.parquet"
        _require(path.name == expected_name, f"unexpected Parquet order/name: {path}")
        records.append(
            {
                "path": str(path.relative_to(dataset)),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": "openpi.parquet_payload_inventory.v1",
        "dataset": str(dataset),
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }
    _atomic_json(output_path, payload)
    return payload


def _resolve_raw_dir(manifest_path: Path, manifest: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    root = Path(manifest.get("raw_root", "."))
    if not root.is_absolute():
        root = manifest_path.parent / root
    return (root / path).resolve()


def _current_hashes(
    manifest_path: Path,
    manifest: dict[str, Any],
    episode: dict[str, Any],
) -> tuple[str | None, dict[str, str]]:
    raw_dir = _resolve_raw_dir(manifest_path, manifest, str(episode["raw_dir"]))
    _require(raw_dir.is_dir(), f"missing raw directory: {raw_dir}")
    streams = {}
    for filename in STREAMS:
        path = raw_dir / filename
        _require(path.is_file(), f"missing raw stream: {path}")
        streams[filename] = _sha256(path)
    if not episode.get("include", True):
        return None, streams
    label_path = raw_dir / str(episode.get("label_file", "subtask_labels.json"))
    _require(label_path.is_file(), f"missing approved label: {label_path}")
    return _sha256(label_path), streams


def _load_candidate_contract(
    manifest_path: Path,
    verification_path: Path,
    dataset: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text())
    verification = json.loads(verification_path.read_text())
    _require(verification.get("status") == "passed", "candidate verification did not pass")
    _require(verification.get("episodes") == EXPECTED_EPISODES, "candidate verification episodes != 90")
    _require(verification.get("frames") == EXPECTED_FRAMES, "candidate verification frames != 77445")
    _require(verification.get("tasks") == EXPECTED_TASKS, "candidate verification tasks != 7")
    _require(
        verification.get("manifest_sha256") == _sha256(manifest_path),
        "candidate verification is not bound to the pending manifest",
    )
    _require(Path(str(verification.get("dataset", ""))).resolve() == dataset, "verification dataset path drifted")
    _require(
        set(verification.get("meta_sha256", {})) == EXPECTED_META_FILES,
        "candidate verification metadata inventory is not the exact expected 8-file set",
    )
    for filename, expected_hash in verification.get("meta_sha256", {}).items():
        path = dataset / "meta" / filename
        _require(path.is_file(), f"candidate dataset is missing {path}")
        _require(_sha256(path) == expected_hash, f"candidate metadata changed after verification: {filename}")
    return manifest, verification


def prepare(
    candidate_manifest_path: Path,
    candidate_verification_path: Path,
    dataset: Path,
    approved_manifest_path: Path,
    approval_ledger_path: Path,
    artifact_paths: list[Path],
) -> None:
    pending, verification = _load_candidate_contract(candidate_manifest_path, candidate_verification_path, dataset)
    _require(pending.get("review_status") == "assistant_validated_pending_user", "candidate is not pending review")
    by_stable_id = {row["stable_id"]: row for row in verification["episode_results"]}
    _require(len(by_stable_id) == EXPECTED_EPISODES, "candidate verification has duplicate/missing episodes")

    approved_at = datetime.now().astimezone().isoformat(timespec="seconds")
    approved = copy.deepcopy(pending)
    ledger_entries = []
    included_count = 0
    for episode in approved["episodes"]:
        stable_id = str(episode["stable_id"])
        label_hash, stream_hashes = _current_hashes(candidate_manifest_path, pending, episode)
        if episode.get("include", True):
            verified = by_stable_id.get(stable_id)
            _require(verified is not None, f"candidate verification is missing {stable_id}")
            _require(verified["label_sha256"] == label_hash, f"{stable_id}: label changed after verification")
            _require(
                verified["raw_stream_sha256"] == stream_hashes,
                f"{stable_id}: raw stream changed after verification",
            )
            decision = "user_approved"
            included_count += 1
            episode["label_sha256"] = label_hash
        else:
            _require(stable_id == "0830_bin_part2/demo14", f"unexpected excluded episode {stable_id}")
            decision = "user_approved_exclusion"
        episode["raw_stream_sha256"] = stream_hashes
        episode["approval_status"] = decision
        ledger_entries.append(
            {
                "stable_id": stable_id,
                "include": bool(episode.get("include", True)),
                "decision": decision,
                "reviewer": "user",
                "approved_at": approved_at,
                "label_file": episode.get("label_file"),
                "label_sha256": label_hash,
                "raw_stream_sha256": stream_hashes,
            }
        )
    _require(included_count == EXPECTED_EPISODES, f"approved included count is {included_count}")

    artifacts = []
    for path in artifact_paths:
        resolved = path.resolve()
        _require(resolved.is_file(), f"missing manual-review artifact: {resolved}")
        artifacts.append({"path": str(resolved), "sha256": _sha256(resolved)})
    approved["review_status"] = "user_approved"
    approved["approval"] = {
        "reviewer": "user",
        "approved_at": approved_at,
        "approval_message": APPROVAL_MESSAGE,
        "candidate_manifest": str(candidate_manifest_path),
        "candidate_manifest_sha256": _sha256(candidate_manifest_path),
        "candidate_verification": str(candidate_verification_path),
        "candidate_verification_sha256": _sha256(candidate_verification_path),
        "candidate_numeric_state_action_task_sha256": verification["numeric_state_action_task_sha256"],
        "candidate_raw_source_inventory_sha256": verification["raw_source_inventory_sha256"],
        "candidate_sampled_image_pixels_sha256": verification["sampled_image_pixels_sha256"],
        "manual_review_artifacts": artifacts,
    }
    approved["expected"]["require_content_hashes"] = True
    approved["expected"]["require_user_approval"] = True
    _atomic_json(approved_manifest_path, approved)
    _atomic_json(
        approval_ledger_path,
        {
            "schema_version": "openpi.dataset_approval_ledger.v1",
            "reviewer": "user",
            "approved_at": approved_at,
            "approval_message": APPROVAL_MESSAGE,
            "approved_manifest": str(approved_manifest_path),
            "entries": ledger_entries,
        },
    )
    print(f"prepared user-approved manifest with {included_count} included episodes: {approved_manifest_path}")


def promote(
    candidate_manifest_path: Path,
    candidate_verification_path: Path,
    dataset: Path,
    approved_manifest_path: Path,
    approval_ledger_path: Path,
    approval_dir: Path,
) -> None:
    pending, _ = _load_candidate_contract(candidate_manifest_path, candidate_verification_path, dataset)
    approved = json.loads(approved_manifest_path.read_text())
    ledger = json.loads(approval_ledger_path.read_text())
    _require(approved.get("review_status") == "user_approved", "manifest is not user-approved")
    _require(len(ledger.get("entries", [])) == len(approved["episodes"]), "approval ledger is incomplete")
    for episode in approved["episodes"]:
        label_hash, stream_hashes = _current_hashes(approved_manifest_path, approved, episode)
        _require(episode.get("raw_stream_sha256") == stream_hashes, f"{episode['stable_id']}: raw hash drift")
        if episode.get("include", True):
            _require(episode.get("label_sha256") == label_hash, f"{episode['stable_id']}: label hash drift")

    conversion_report_path = dataset / "meta" / "conversion_report.json"
    report = json.loads(conversion_report_path.read_text())
    _require(
        report.get("episode_manifest_sha256") == _sha256(candidate_manifest_path),
        "dataset was not converted from the expected pending manifest",
    )
    _require(report.get("review_status") == pending.get("review_status"), "candidate review status drifted")
    approval_dir.mkdir(parents=True, exist_ok=True)
    backup = approval_dir / "conversion_report_preapproval.json"
    if not backup.exists():
        _atomic_json(backup, report)
    report["episode_manifest"] = str(approved_manifest_path)
    report["episode_manifest_sha256"] = _sha256(approved_manifest_path)
    report["review_status"] = "user_approved"
    report["excluded_episodes"] = [episode for episode in approved["episodes"] if not episode.get("include", True)]
    _atomic_json(conversion_report_path, report)
    print(f"promoted conversion_report.json to approved manifest {approved_manifest_path}")


def finalize(
    dataset: Path,
    approved_manifest_path: Path,
    approval_ledger_path: Path,
    final_verification_path: Path,
    payload_inventory_path: Path,
    release_path: Path,
) -> None:
    manifest = json.loads(approved_manifest_path.read_text())
    verification = json.loads(final_verification_path.read_text())
    _require(manifest.get("review_status") == "user_approved", "approved manifest status drifted")
    _require(verification.get("status") == "passed", "approved verification did not pass")
    _require(
        verification.get("manifest_sha256") == _sha256(approved_manifest_path),
        "approved verification manifest SHA drifted",
    )
    _require(Path(str(verification.get("dataset", ""))).resolve() == dataset, "approved dataset path drifted")
    _require(verification.get("episodes") == EXPECTED_EPISODES, "approved episodes != 90")
    _require(verification.get("frames") == EXPECTED_FRAMES, "approved frames != 77445")
    _require(verification.get("tasks") == EXPECTED_TASKS, "approved tasks != 7")
    _require(
        set(verification.get("meta_sha256", {})) == EXPECTED_META_FILES,
        "approved verification metadata inventory is not the exact expected 8-file set",
    )
    approval = manifest.get("approval", {})
    digest_bindings = {
        "numeric_state_action_task_sha256": "candidate_numeric_state_action_task_sha256",
        "raw_source_inventory_sha256": "candidate_raw_source_inventory_sha256",
        "sampled_image_pixels_sha256": "candidate_sampled_image_pixels_sha256",
    }
    for verification_key, approval_key in digest_bindings.items():
        _require(
            verification.get(verification_key) == approval.get(approval_key),
            f"approved verification {verification_key} differs from the reviewed candidate",
        )
    payload = _seal_payload(dataset, payload_inventory_path)
    _atomic_json(
        release_path,
        {
            "schema_version": "openpi.dataset_release.v1",
            "status": "user_approved_data_ready_for_training",
            "dataset": str(dataset),
            "approved_manifest": str(approved_manifest_path),
            "approved_manifest_sha256": _sha256(approved_manifest_path),
            "approval_ledger": str(approval_ledger_path),
            "approval_ledger_sha256": _sha256(approval_ledger_path),
            "verification": str(final_verification_path),
            "verification_sha256": _sha256(final_verification_path),
            "parquet_payload_inventory": str(payload_inventory_path),
            "parquet_payload_inventory_sha256": _sha256(payload_inventory_path),
            "parquet_payload_files": payload["file_count"],
            "parquet_payload_bytes": payload["total_bytes"],
            "episodes": EXPECTED_EPISODES,
            "frames": EXPECTED_FRAMES,
            "tasks": EXPECTED_TASKS,
            "numeric_state_action_task_sha256": verification["numeric_state_action_task_sha256"],
            "raw_source_inventory_sha256": verification["raw_source_inventory_sha256"],
            "sampled_image_pixels_sha256": verification["sampled_image_pixels_sha256"],
            "meta_sha256": verification["meta_sha256"],
        },
    )
    print(f"finalized approved dataset release: {release_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "promote", "finalize"))
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-verification", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--approved-manifest", type=Path, required=True)
    parser.add_argument("--approval-ledger", type=Path, required=True)
    parser.add_argument("--approval-dir", type=Path, required=True)
    parser.add_argument("--final-verification", type=Path)
    parser.add_argument("--payload-inventory", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--artifact", action="append", type=Path, default=[])
    args = parser.parse_args()
    candidate_manifest = args.candidate_manifest.resolve()
    candidate_verification = args.candidate_verification.resolve()
    dataset = args.dataset.resolve()
    approved_manifest = args.approved_manifest.resolve()
    approval_ledger = args.approval_ledger.resolve()
    approval_dir = args.approval_dir.resolve()
    if args.mode == "prepare":
        prepare(
            candidate_manifest,
            candidate_verification,
            dataset,
            approved_manifest,
            approval_ledger,
            args.artifact,
        )
    elif args.mode == "promote":
        promote(
            candidate_manifest,
            candidate_verification,
            dataset,
            approved_manifest,
            approval_ledger,
            approval_dir,
        )
    else:
        _require(args.final_verification is not None, "finalize requires --final-verification")
        _require(args.payload_inventory is not None, "finalize requires --payload-inventory")
        _require(args.release is not None, "finalize requires --release")
        finalize(
            dataset,
            approved_manifest,
            approval_ledger,
            args.final_verification.resolve(),
            args.payload_inventory.resolve(),
            args.release.resolve(),
        )


if __name__ == "__main__":
    main()
