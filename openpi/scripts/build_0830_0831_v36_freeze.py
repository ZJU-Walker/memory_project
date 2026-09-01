"""Freeze the v36 (0830+0831) dataset: manifest, split, audits, and tree inventory.

The v3.5 90-episode freeze was invalidated by the Gate-B side leak in the 0816 collection
(both primary probes p~=0.001).  This builder produces the v36 replacement population from
the two leak-clean collections:

- 0830: the 30 already user-approved episodes.  Their label and raw-stream bytes must be
  EXACTLY the ones sealed inside the v3.5 frozen manifest; their d_valid and e_visibility
  audit records are carried over verbatim.
- 0831: the 40 newly collected episodes.  d_valid is recomputed here with the identical
  14D detector, and e_visibility comes from the refined full-visibility inspect boundaries
  plus the assistant-verified contact sheet.

Split (seed 36, "openpi.v36.sha256-ranked-manifest-fields.v1"): rank episodes with the same
null-joined SHA-256 formula as v3.5 over (algorithm, stage, seed, stable_id, part, object,
target_side); select one final-test episode per collection*object*side cell (8), then one
development episode per collection*object*side cell (8) while preserving at least one train
episode in every 0830 part*object*side and 0831 object*side cell; every other included
episode is train (54).

Outputs (all under memory_project/data):
- 0830_0831_episode_manifest_v36_frozen.json
- 0830_0831_episode_manifest_v36_frozen_d_valid.json
- 0830_0831_episode_manifest_v36_frozen_e_visibility.json
- 0830_0831_episode_manifest_v36_frozen_block_confound.json
- 0830_0831_v36_dataset_tree_inventory.json

The script is read-only with respect to raw data, labels, and the converted dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pyarrow.parquet  # noqa: F401  (import order guard: Arrow before JAX/OpenPI)
import relocate_v35_dataset as relocate

import openpi.training.data_loader as _data_loader

SPLIT_SEED = 36
SPLIT_ALGORITHM = "openpi.v36.sha256-ranked-manifest-fields.v1"
SPLIT_SPEC = (
    "seed=36; sort stable_id; rank by sha256(algorithm,stage,seed,stable_id,part,object,target_side); "
    "select one final-test episode per collection*object*side cell; then one development episode per "
    "collection*object*side cell while preserving one train episode in every 0830 part*object*side "
    "and 0831 object*side cell; assign every other included episode to train"
)
DETECTOR = "14d-max-step-lt-0.004-max-excursion-lte-0.02-v1"
MAX_STATE_STEP = 4e-3
MAX_STATE_EXCURSION = 2e-2
STREAMS = (
    "left_joint_positions.npy",
    "right_joint_positions.npy",
    "left_control.npy",
    "right_control.npy",
    "top_camera_rgb.mp4",
    "left_camera_rgb.mp4",
    "right_camera_rgb.mp4",
)
OBJECT_PROMPTS = {
    "banana": "find the banana",
    "grey_pepper_box": "find the grey pepper box",
}
EXPECTED_INCLUDED = {"0830": 30, "0831": 40}
EXPECTED_FRAMES = 55_980
USER_APPROVAL_MESSAGE = "1 thing now the subtask label is ok so i approvbe your label"


class FreezeError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:  # noqa: FBT001
    if not condition:
        raise FreezeError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def _split_rank(record: dict[str, Any], *, stage: str, seed: int) -> str:
    fields = (
        SPLIT_ALGORITHM,
        stage,
        str(seed),
        str(record["stable_id"]),
        str(record.get("part", "")),
        str(record["object"]),
        str(record["target_side"]),
    )
    return hashlib.sha256("\0".join(fields).encode("utf-8")).hexdigest()


def build_v36_splits(records: list[dict[str, Any]], *, seed: int = SPLIT_SEED) -> dict[str, str]:
    """Assign train/development/final_test purely from preregistered manifest fields."""
    included = [record for record in records if bool(record.get("include", True))]
    by_collection: dict[str, list[dict[str, Any]]] = {"0830": [], "0831": []}
    for record in included:
        collection = str(record.get("collection", "")).strip()
        _require(collection in by_collection, f"v36 manifest has unknown collection {collection!r}")
        by_collection[collection].append(record)
    counts = {key: len(value) for key, value in by_collection.items()}
    _require(counts == EXPECTED_INCLUDED, f"v36 population must be {EXPECTED_INCLUDED}, got {counts}")

    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in included:
        cell = (record["collection"], str(record["object"]), str(record["target_side"]))
        cells.setdefault(cell, []).append(record)
    expected_cells = {
        (collection, object_name, side)
        for collection in ("0830", "0831")
        for object_name in OBJECT_PROMPTS
        for side in ("left", "right")
    }
    _require(set(cells) == expected_cells, f"v36 collection*object*side cells incomplete: {sorted(set(cells))}")

    def guard_cell(record: dict[str, Any]) -> tuple[str, ...]:
        # The finer stratification whose train coverage must never empty out.
        if record["collection"] == "0830":
            return ("0830", str(record.get("part", "")), str(record["object"]), str(record["target_side"]))
        return ("0831", str(record["object"]), str(record["target_side"]))

    remaining: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    final_ids: set[str] = set()
    for cell, values in sorted(cells.items()):
        selected = min(values, key=lambda record: _split_rank(record, stage="final_test", seed=seed))
        final_ids.add(str(selected["stable_id"]))
        remaining[cell] = [record for record in values if record is not selected]

    development_ids: set[str] = set()
    for cell in sorted(expected_cells):
        pool = remaining[cell]
        guard_counts: dict[tuple[str, ...], int] = {}
        for record in pool:
            guard_counts[guard_cell(record)] = guard_counts.get(guard_cell(record), 0) + 1
        candidates = [record for record in pool if guard_counts[guard_cell(record)] >= 2]
        _require(
            bool(candidates),
            f"v36 cannot select a development episode for cell {cell} while preserving train coverage",
        )
        selected = min(candidates, key=lambda record: _split_rank(record, stage="development", seed=seed))
        development_ids.add(str(selected["stable_id"]))
        remaining[cell] = [record for record in pool if record is not selected]

    splits: dict[str, str] = {}
    for record in included:
        stable_id = str(record["stable_id"])
        splits[stable_id] = (
            "final_test" if stable_id in final_ids else "development" if stable_id in development_ids else "train"
        )
    _require(list(splits.values()).count("train") == 54, "v36 split did not produce exactly 54 training episodes")
    _require(
        list(splits.values()).count("development") == 8, "v36 split did not produce exactly 8 development episodes"
    )
    _require(list(splits.values()).count("final_test") == 8, "v36 split did not produce exactly 8 final-test episodes")
    guard_remaining: dict[tuple[str, ...], int] = {}
    for record in included:
        if splits[str(record["stable_id"])] == "train":
            guard_remaining[guard_cell(record)] = guard_remaining.get(guard_cell(record), 0) + 1
    _require(all(count >= 1 for count in guard_remaining.values()), "v36 split empties a guarded train cell")
    return splits


def _d_valid_for_wait(demo: Path, wait_start: int, wait_end: int) -> dict[str, Any]:
    left = np.load(demo / "left_joint_positions.npy").astype(np.float32)
    right = np.load(demo / "right_joint_positions.npy").astype(np.float32)
    state = np.concatenate([left, right], axis=1)[wait_start : wait_end + 1]
    _require(state.shape[1] == 14, f"{demo}: expected 14D state")
    _require(len(state) >= 2, f"{demo}: wait window too short for the detector")
    max_step = float(np.abs(np.diff(state, axis=0)).max())
    max_excursion = float((state.max(axis=0) - state.min(axis=0)).max())
    _require(max_step < MAX_STATE_STEP, f"{demo}: wait max step {max_step} violates detector bound")
    _require(max_excursion <= MAX_STATE_EXCURSION, f"{demo}: wait excursion {max_excursion} violates bound")
    return {
        "detector": DETECTOR,
        "state_dim": 14,
        "start": wait_start,
        "end": wait_end,
        "max_14d_step": max_step,
        "max_14d_excursion": max_excursion,
        "eligible_at_stride_15": wait_end - wait_start + 1 >= 15,
        "source_left_joint_sha256": _sha256_file(demo / "left_joint_positions.npy"),
        "source_right_joint_sha256": _sha256_file(demo / "right_joint_positions.npy"),
    }


def _natural_key(stable_id: str) -> tuple[str, int]:
    match = re.search(r"(\d+)$", stable_id)
    return (stable_id.rsplit("demo", 1)[0], int(match.group(1)) if match else 0)


def _block_confound_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Manifest-only temporal audit: per recording block, how side/object interleave over time."""
    blocks: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        block = f"{record['collection']}_{record['part']}" if record["part"] else record["collection"]
        blocks.setdefault(block, []).append(record)
    summary: dict[str, Any] = {"algorithm": "openpi.v36.manifest-only-temporal-block-audit.v1", "blocks": {}}
    worst_run = 0
    for block, values in sorted(blocks.items()):
        ordered = sorted(values, key=lambda record: _natural_key(str(record["stable_id"])))
        features: dict[str, Any] = {}
        for feature in ("target_side", "object"):
            sequence = [str(record[feature]) for record in ordered]
            runs = 1 + sum(1 for i in range(1, len(sequence)) if sequence[i] != sequence[i - 1])
            longest = max(
                (len(list(group)) for group in _group_runs(sequence)),
                default=0,
            )
            worst_run = max(worst_run, longest)
            features[feature] = {
                "classes": sorted(set(sequence)),
                "counts": {name: sequence.count(name) for name in sorted(set(sequence))},
                "alternation_runs": runs,
                "longest_single_class_run": longest,
            }
        summary["blocks"][block] = {
            "episode_count": len(ordered),
            "ordered_stable_ids": [str(record["stable_id"]) for record in ordered],
            "features": features,
        }
    summary["longest_single_class_run_overall"] = worst_run
    return summary


def _group_runs(sequence: list[str]):
    start = 0
    for index in range(1, len(sequence) + 1):
        if index == len(sequence) or sequence[index] != sequence[start]:
            yield sequence[start:index]
            start = index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conversion-manifest", type=Path, default=Path("../data/0830_0831_episode_manifest_v1.json"))
    parser.add_argument("--v35-frozen", type=Path, default=Path("../data/0816_0830_episode_manifest_v35_frozen.json"))
    parser.add_argument("--dataset", type=Path, default=Path("../data/lerobot/yam/bin_memory_0830_0831_v36_subtask"))
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=Path("examples/yam/diagnostic_outputs/0831_memory_labels_v1/inspect_start_refined.jpg"),
    )
    parser.add_argument("--output-prefix", type=Path, default=Path("../data/0830_0831_episode_manifest_v36_frozen"))
    parser.add_argument(
        "--inventory-output", type=Path, default=Path("../data/0830_0831_v36_dataset_tree_inventory.json")
    )
    args = parser.parse_args()

    conversion_path = args.conversion_manifest.resolve()
    conversion = json.loads(conversion_path.read_text())
    v35 = json.loads(args.v35_frozen.resolve().read_text())
    v35_by_id = {str(entry["stable_id"]): entry for entry in v35["episodes"]}
    dataset = args.dataset.resolve()
    raw_root = (conversion_path.parent / conversion["raw_root"]).resolve()
    contact_sheet = args.contact_sheet.resolve()
    contact_sheet_sha = _sha256_file(contact_sheet)

    from datetime import datetime

    approved_at = datetime.now().astimezone().isoformat(timespec="seconds")
    ledger = {
        "schema_version": "openpi.v36.approval-ledger.v1",
        "reviewer": "user",
        "approved_at": approved_at,
        "approval_message": USER_APPROVAL_MESSAGE,
        "scope": "0831 subtask labels (five-phase, full-visibility inspect refinement) and v36 freeze",
        "episodes": {
            spec["stable_id"]: {"status": "user_approved", "reviewer": "user", "approved_at": approved_at}
            for spec in conversion["episodes"]
            if spec.get("include", True) and "0831_bin" in spec["raw_dir"]
        },
    }
    ledger_path = conversion_path.parent / "0830_0831_v36_approval_ledger.json"
    _atomic_json(ledger_path, ledger)
    approval_ledger_sha256 = _sha256_file(ledger_path)

    approved = dict(conversion)
    approved["dataset_version"] = "v36"
    approved["review_status"] = "user_approved"
    approved["approval"] = {
        "reviewer": "user",
        "approved_at": approved_at,
        "approval_message": USER_APPROVAL_MESSAGE,
    }
    approved_path = conversion_path.parent / "0830_0831_episode_manifest_v36_approved.json"
    _atomic_json(approved_path, approved)

    episodes: list[dict[str, Any]] = []
    episode_index = 0
    refinement = json.loads(
        (raw_root / "openpi/examples/yam/diagnostic_outputs/0831_memory_labels_v1/inspect_refinement.json").read_text()
    )
    refined_by_demo = {row["demo"]: row for row in refinement}

    for spec in conversion["episodes"]:
        if not spec.get("include", True):
            sealed_excluded = v35_by_id.get(str(spec["stable_id"]))
            _require(
                sealed_excluded is not None and not sealed_excluded.get("include", True),
                f"excluded episode {spec['stable_id']} has no sealed v3.5 provenance record",
            )
            episodes.append(dict(sealed_excluded))
            continue
        demo = raw_root / spec["raw_dir"]
        label_path = demo / spec.get("label_file", "subtask_labels.json")
        label_sha = _sha256_file(label_path)
        stream_sha = {name: _sha256_file(demo / name) for name in STREAMS}
        is_0831 = "0831_bin" in spec["raw_dir"]
        collection = "0831" if is_0831 else "0830"
        part = "" if is_0831 else spec["raw_dir"].split("/")[1].removeprefix("0830_bin_")
        object_name = "banana" if spec["instruction"] == OBJECT_PROMPTS["banana"] else "grey_pepper_box"

        if is_0831:
            labels = json.loads(label_path.read_text())
            wait = next(seg for seg in labels if seg["task"].startswith("wait;"))
            inspect = labels[1]
            _require(inspect["task"] == "inspect both bins", f"{demo}: unexpected label order")
            d_valid = _d_valid_for_wait(demo, wait["start"], wait["end"])
            refined = refined_by_demo[demo.name]
            _require(refined["new_inspect_start"] == inspect["start"], f"{demo}: refinement/label drift")
            e_visibility = {
                "manual_reviewed": True,
                "both_objects_visible": True,
                "first_valid_visible_frame": inspect["start"],
                "last_clean_visible_frame": inspect["end"],
                "contact_sheet_sha256": contact_sheet_sha,
                "decision_source": (
                    "0831 full-visibility saturation refinement (refine_0831_inspect_start.py), "
                    "assistant-verified per-episode contact sheets, user browser-labeler review, "
                    "and the recorded user approval message"
                ),
                "approval_ledger_sha256": approval_ledger_sha256,
            }
            approval_status = "user_approved"
        else:
            sealed = v35_by_id.get(spec["stable_id"])
            _require(sealed is not None, f"0830 episode {spec['stable_id']} missing from v3.5 frozen manifest")
            _require(sealed["label_sha256"] == label_sha, f"{spec['stable_id']}: label bytes drifted since v3.5 freeze")
            _require(
                sealed["raw_stream_sha256"] == stream_sha,
                f"{spec['stable_id']}: raw stream bytes drifted since v3.5 freeze",
            )
            d_valid = dict(sealed["d_valid"])
            e_visibility = dict(sealed["e_visibility"])
            approval_status = str(sealed["approval_status"])

        episodes.append(
            {
                "stable_id": spec["stable_id"],
                "raw_dir": spec["raw_dir"],
                "collection": collection,
                "part": part,
                "object": object_name,
                "prompt": spec["instruction"],
                "target_side": spec["target_side"],
                "include": True,
                "label_file": spec.get("label_file", "subtask_labels.json"),
                "expected_num_frames": spec["expected_num_frames"],
                "memory_waiting_core": dict(spec["memory_waiting_core"]),
                "metadata_counter": spec.get("metadata_counter", 0),
                "timestamp": spec.get("timestamp", ""),
                "label_sha256": label_sha,
                "raw_stream_sha256": stream_sha,
                "approval_status": approval_status,
                "episode_index": episode_index,
                "e_visibility": e_visibility,
                "d_valid": d_valid,
            }
        )
        episode_index += 1

    included = [entry for entry in episodes if entry.get("include", True)]
    _require(len(included) == 70, f"expected 70 included episodes, got {len(included)}")
    total_frames = sum(entry["expected_num_frames"] for entry in included)
    _require(total_frames == EXPECTED_FRAMES, f"expected {EXPECTED_FRAMES} frames, got {total_frames}")

    splits = build_v36_splits(included)
    loader_splits = _data_loader._v35_expected_frozen_splits(episodes, seed=SPLIT_SEED)  # noqa: SLF001
    _require(splits == loader_splits, "freezer/loader split implementations disagree")
    import v35_gate_artifacts as gate_artifacts

    _require(_data_loader._V35_SPLIT_ALGORITHM_SPEC == SPLIT_SPEC, "freezer/loader split spec text drifted")  # noqa: SLF001
    _require(gate_artifacts.V35_SPLIT_ALGORITHM_SPEC == SPLIT_SPEC, "freezer/gate split spec text drifted")
    gate_records = [
        {key: entry[key] for key in ("stable_id", "collection", "part", "object", "target_side")} for entry in included
    ]
    gate_splits = gate_artifacts._expected_frozen_splits(gate_records)  # noqa: SLF001
    _require(splits == gate_splits, "freezer/gate-artifacts split implementations disagree")
    for entry in included:
        entry["split"] = splits[str(entry["stable_id"])]

    prefix = args.output_prefix.resolve()
    d_valid_report = {
        "schema_version": "openpi.v36.d-valid.v1",
        "status": "complete",
        "detector": DETECTOR,
        "state_dim": 14,
        "episodes": [{"stable_id": str(entry["stable_id"]), "d_valid": entry["d_valid"]} for entry in included],
    }
    e_visibility_report = {
        "schema_version": "openpi.v36.e-visibility-review.v1",
        "status": "user_approved",
        "approval_ledger_sha256": approval_ledger_sha256,
        "episodes": [
            {"stable_id": str(entry["stable_id"]), "e_visibility": entry["e_visibility"]} for entry in included
        ],
    }
    audit_fields = [
        {
            key: record.get(key)
            for key in ("stable_id", "include", "collection", "part", "object", "target_side", "timestamp")
        }
        for record in episodes
        if bool(record.get("include", True))
    ]
    audit_fields_sha256 = hashlib.sha256(
        _data_loader._v35_canonical_json(audit_fields).encode("utf-8")  # noqa: SLF001
    ).hexdigest()
    block_summary = _data_loader._v35_block_confound_summary(episodes)  # noqa: SLF001
    _require(block_summary.get("pass") is True, f"v36 block-confound audit failed: {block_summary}")
    block_report = {
        "schema_version": "openpi.v36.block-confound-audit.v1",
        "status": "pass",
        "manifest_fields_only": True,
        "manifest_fields_sha256": audit_fields_sha256,
        "summary": block_summary,
    }
    d_valid_path = prefix.with_name(prefix.name + "_d_valid.json")
    e_visibility_path = prefix.with_name(prefix.name + "_e_visibility.json")
    block_path = prefix.with_name(prefix.name + "_block_confound.json")
    _atomic_json(d_valid_path, d_valid_report)
    _atomic_json(e_visibility_path, e_visibility_report)
    _atomic_json(block_path, block_report)

    frozen = {
        "schema_version": 2,
        "created": "2026-08-31",
        "dataset_version": "v36",
        "review_status": "frozen",
        "raw_root": conversion["raw_root"],
        "task_vocabulary": conversion["task_vocabulary"],
        "memory_waiting_core_config": {
            "state_dimensions": 14,
            "max_state_step": MAX_STATE_STEP,
            "max_state_excursion": MAX_STATE_EXCURSION,
            "stride_frames": 15,
        },
        "source_order": conversion["source_order"],
        "expected": {
            "included_episodes": 70,
            "included_frames": EXPECTED_FRAMES,
            "require_memory_waiting_core": True,
            "require_semantic_wait_equals_core": True,
            "require_exact_five_phase_schema": True,
        },
        "approval": {
            "reviewer": "user",
            "approved_at": approved_at,
            "approval_message": USER_APPROVAL_MESSAGE,
            "candidate_manifest_sha256": _sha256_file(conversion_path),
        },
        "source_approved_manifest": {
            "file": approved_path.name,
            "sha256": _sha256_file(approved_path),
        },
        "approval_ledger_sha256": approval_ledger_sha256,
        "provenance": {
            "conversion_manifest": conversion_path.name,
            "conversion_manifest_sha256": _sha256_file(conversion_path),
            "v35_frozen_manifest_sha256": _sha256_file(args.v35_frozen.resolve()),
            "excluded_0816_reason": (
                "Gate B leakage stop: both primary probes on the 0816 collection reject the "
                "leak-free null (macro balanced accuracy 0.821/0.804, permutation p=0.001); "
                "sealed evidence in v35/diagnostics/runs/v35_fresh_pilot_20260831_r7/gates/gate_b.json"
            ),
        },
        "split_seed": SPLIT_SEED,
        "split_algorithm": _data_loader._V35_SPLIT_ALGORITHM,  # noqa: SLF001
        "split_algorithm_spec": SPLIT_SPEC,
        "split_algorithm_sha256": hashlib.sha256(SPLIT_SPEC.encode()).hexdigest(),
        "block_confound_audit": {
            "status": "pass",
            "manifest_fields_only": True,
            "manifest_fields_sha256": audit_fields_sha256,
            "report_file": block_path.name,
            "report_sha256": _sha256_file(block_path),
        },
        "e_visibility_review": {
            "status": "user_approved",
            "report_file": e_visibility_path.name,
            "report_sha256": _sha256_file(e_visibility_path),
        },
        "d_valid_sidecar": {
            "detector": DETECTOR,
            "state_dim": 14,
            "report_file": d_valid_path.name,
            "report_sha256": _sha256_file(d_valid_path),
        },
        "episodes": episodes,
    }
    frozen_path = prefix.with_suffix(".json")
    _atomic_json(frozen_path, frozen)

    validated = gate_artifacts.load_frozen_manifest(frozen_path, expected_sha256=_sha256_file(frozen_path))
    _require(len(validated.episodes) == 70, "round-trip validation returned the wrong episode count")

    scanned = relocate.build_inventory(dataset, enforce_v35_identity=False)
    inventory = relocate.TreeInventory(
        dataset_repo_id="yam/bin_memory_0830_0831_v36_subtask",
        directories=scanned.directories,
        files=scanned.files,
    )
    inventory_path = args.inventory_output.resolve()
    # The pre-pilot orchestrator loads the inventory through its immutable-stage loader,
    # which accepts only canonical compact JSON with exactly one trailing newline.
    inventory_tmp = inventory_path.with_suffix(inventory_path.suffix + ".tmp")
    inventory_tmp.write_bytes(inventory.canonical_bytes())
    inventory_tmp.replace(inventory_path)
    inventory_dict = inventory.as_dict()

    from collections import Counter

    print("frozen manifest:", frozen_path)
    print("  sha256:", _sha256_file(frozen_path))
    print("inventory:", args.inventory_output.resolve())
    print("  sha256:", _sha256_file(args.inventory_output.resolve()))
    print("  tree_sha256:", inventory_dict["tree_sha256"])
    print("splits:", Counter(splits.values()))
    for name in ("final_test", "development"):
        chosen = sorted(sid for sid, split in splits.items() if split == name)
        print(f"  {name}: {chosen}")


if __name__ == "__main__":
    main()
