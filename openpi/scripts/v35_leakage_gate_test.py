import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v35_gate_artifacts as artifacts
    import v35_leakage_gate as leakage
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


def _manifest(tmp_path: Path) -> tuple[Path, str]:
    episodes = []
    for part, repeats in (("part1", (4, 4, 4, 4)), ("part2", (4, 3, 4, 3))):
        demo = 1
        for (object_name, side), count in zip(
            (("banana", "left"), ("banana", "right"), ("grey_pepper_box", "left"), ("grey_pepper_box", "right")),
            repeats,
            strict=True,
        ):
            for _ in range(count):
                episodes.append(
                    {
                        "stable_id": f"0830_bin_{part}/demo{demo}",
                        "episode_index": len(episodes),
                        "collection": "0830",
                        "object": object_name,
                        "part": part,
                        "target_side": side,
                        "split": "train",
                        "include": True,
                    }
                )
                demo += 1
    demo = 1
    for (object_name, side), count in zip(
        (("banana", "left"), ("banana", "right"), ("grey_pepper_box", "left"), ("grey_pepper_box", "right")),
        (10, 10, 10, 10),
        strict=True,
    ):
        for _ in range(count):
            episodes.append(
                {
                    "stable_id": f"0831_bin/demo{demo}",
                    "episode_index": len(episodes),
                    "collection": "0831",
                    "object": object_name,
                    "part": "",
                    "target_side": side,
                    "split": "train",
                    "include": True,
                }
            )
            demo += 1
    expected = artifacts._expected_frozen_splits(episodes)  # noqa: SLF001
    for record in episodes:
        record["split"] = expected[record["stable_id"]]
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "dataset_version": "v36",
                "episodes": episodes,
                "review_status": "frozen",
                "schema_version": 2,
                "split_algorithm": artifacts.V35_SPLIT_ALGORITHM,
                "split_algorithm_sha256": artifacts.V35_SPLIT_ALGORITHM_SHA256,
                "split_seed": 36,
            }
        ),
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _features(
    tmp_path: Path,
    manifest: artifacts.FrozenManifest,
    *,
    leaking: bool = False,
    replace_last_with_final: bool = False,
) -> Path:
    train = manifest.split("train")
    stable_ids = [episode.stable_id for episode in train]
    if replace_last_with_final:
        stable_ids[-1] = manifest.split("final_test")[0].stable_id
    labels = np.asarray([episode.target_side for episode in train], dtype=np.float32)
    arrays: dict[str, np.ndarray] = {"episode_stable_id": np.asarray(stable_ids)}
    for feature_name in leakage.ALL_FEATURES:
        if leaking and feature_name == "primary_final_images_state":
            arrays[feature_name] = np.stack((labels, 1.0 - labels), axis=1).astype(np.float32)
        else:
            arrays[feature_name] = np.zeros((54, 2), dtype=np.float32)
    npz_path = tmp_path / "features.npz"
    np.savez(npz_path, **arrays)
    initialization_payload = {
        "actual_step0_parameter_tree_sha256": "4" * 64,
        "artifact_hashes": {"episode_manifest_sha256": manifest.sha256, "norm_stats_sha256": "9" * 64},
        "config_name": "pi05_yam_mem_v35",
        "format_version": 2,
        "initialization_seed": 35,
        "memory_inject_w_sha256": "6" * 64,
        "official_source_uri": "gs://openpi-assets/checkpoints/pi05_base/params",
        "source_tree_sha256": "7" * 64,
        "step0_checkpoint": 0,
    }
    initialization_payload["identity_sha256"] = hashlib.sha256(
        json.dumps(initialization_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    initialization_path = tmp_path / "initialization_manifest.json"
    initialization_path.write_text(json.dumps(initialization_payload), encoding="utf-8")
    preprocessing = artifacts.artifact_envelope(
        leakage.PREPROCESSING_SCHEMA_VERSION,
        {
            "episode_manifest_sha256": manifest.sha256,
            "final_preprocessing": True,
            "image_preprocessing_sha256": "8" * 64,
            "norm_stats_sha256": "9" * 64,
            "protocol_sha256": "a" * 64,
            "split_assignment_sha256": manifest.split_assignment_sha256,
            "status": "frozen",
        },
    )
    preprocessing_path = tmp_path / "preprocessing.json"
    artifacts.write_canonical_envelope(
        preprocessing_path,
        preprocessing,
        schema_version=leakage.PREPROCESSING_SCHEMA_VERSION,
    )
    payload = {
        "aggregation_protocol": leakage.AGGREGATION_PROTOCOL,
        "completed_updates": 0,
        "development_or_final_test_accessed": False,
        "episode_manifest_sha256": manifest.sha256,
        "feature_npz": {
            "path": npz_path.name,
            "sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        },
        "feature_protocol_sha256": "3" * 64,
        "initialization_manifest": {
            "path": initialization_path.name,
            "sha256": hashlib.sha256(initialization_path.read_bytes()).hexdigest(),
        },
        "initialization_parameter_tree_sha256": "4" * 64,
        "preprocessing_artifact": {
            "path": preprocessing_path.name,
            "sha256": hashlib.sha256(preprocessing_path.read_bytes()).hexdigest(),
        },
        "population_split": "train",
        "split_assignment_sha256": manifest.split_assignment_sha256,
    }
    envelope = artifacts.artifact_envelope(leakage.FEATURE_SCHEMA_VERSION, payload)
    path = tmp_path / "features.json"
    artifacts.write_canonical_envelope(path, envelope, schema_version=leakage.FEATURE_SCHEMA_VERSION)
    return path


def _dataset(tmp_path: Path, *, leaking: bool = False) -> tuple[artifacts.FrozenManifest, leakage.LeakageDataset]:
    manifest_path, manifest_sha = _manifest(tmp_path)
    manifest = artifacts.load_frozen_manifest(manifest_path, expected_sha256=manifest_sha)
    feature_path = _features(tmp_path, manifest, leaking=leaking)
    return manifest, leakage.load_feature_dataset(feature_path, manifest=manifest)


def test_gate_b_runs_exact_four_test_family_and_passes_noninformative_features(tmp_path: Path) -> None:
    manifest, dataset = _dataset(tmp_path)

    decision = leakage.build_decision_artifact(dataset, manifest=manifest, permutations=1_000)
    payload = decision["payload"]

    assert artifacts.verify_envelope(decision, schema_version=leakage.DECISION_SCHEMA_VERSION)
    assert payload["status"] == "pass"
    assert payload["protocol"]["primary_probe_count"] == 2
    assert payload["protocol"]["permutation_count"] == 1_000
    assert payload["protocol"]["permutation_strata"] == "collection*object"
    assert len(payload["four_test_family"]) == 4
    assert set(payload["descriptive"]) == set(leakage.DESCRIPTIVE_FEATURES)
    assert all(item["permutation_p_value"] > 0.0125 for item in payload["four_test_family"])
    assert all(result["pooled"]["balanced_accuracy"] <= 0.62 for result in payload["primary"].values())


def test_primary_side_feature_fails_both_statistical_and_pooled_protection(tmp_path: Path) -> None:
    manifest, dataset = _dataset(tmp_path, leaking=True)

    decision = leakage.build_decision_artifact(dataset, manifest=manifest, permutations=1_000)
    leaked = decision["payload"]["primary"]["primary_final_images_state"]

    assert decision["payload"]["status"] == "fail"
    assert leaked["pooled"]["balanced_accuracy"] == 1.0
    assert not leaked["pooled"]["passes_point_gate"]
    assert not all(result["passes_family_corrected_test"] for result in leaked["collections"].values())


def test_gate_b_rejects_short_null_and_any_nontrain_episode(tmp_path: Path) -> None:
    manifest, dataset = _dataset(tmp_path)
    with pytest.raises(leakage.LeakageGateError, match="at least 1000"):
        leakage.evaluate_gate_b(dataset, permutations=999)

    foreign_dir = tmp_path / "foreign"
    foreign_dir.mkdir()
    manifest_path = foreign_dir / "manifest.json"
    manifest_path.write_bytes(manifest.path.read_bytes())
    copied_manifest = artifacts.load_frozen_manifest(manifest_path, expected_sha256=manifest.sha256)
    feature_path = _features(foreign_dir, copied_manifest, replace_last_with_final=True)
    with pytest.raises(leakage.LeakageGateError, match="non_train_or_unknown"):
        leakage.load_feature_dataset(feature_path, manifest=copied_manifest)


def test_cached_label_independent_factorization_matches_literal_fold_refits(tmp_path: Path) -> None:
    _, dataset = _dataset(tmp_path)
    rng = np.random.default_rng(7)
    features = rng.normal(size=(54, 9)).astype(np.float32)
    folds = leakage._fold_ids(dataset)  # noqa: SLF001
    literal = np.full(54, np.nan)
    for fold in range(leakage.FOLD_COUNT):
        test = folds == fold
        train = ~test
        literal[test] = leakage._fit_ridge_scores(  # noqa: SLF001
            features[train], dataset.labels[train], features[test]
        )

    cached = leakage._prepared_oof_scores(  # noqa: SLF001
        leakage._prepare_oof(features, folds),  # noqa: SLF001
        dataset.labels,
    )

    np.testing.assert_allclose(cached, literal, rtol=1e-10, atol=1e-10)


def test_feature_envelope_and_npz_hashes_fail_closed(tmp_path: Path) -> None:
    manifest_path, manifest_sha = _manifest(tmp_path)
    manifest = artifacts.load_frozen_manifest(manifest_path, expected_sha256=manifest_sha)
    feature_path = _features(tmp_path, manifest)
    npz_path = tmp_path / "features.npz"
    with npz_path.open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(artifacts.GateArtifactError, match="SHA256 mismatch"):
        leakage.load_feature_dataset(feature_path, manifest=manifest)


@pytest.mark.parametrize("bad_path", [Path("../foreign.json"), Path("/iris/u/user/foreign.json")])
def test_production_cli_paths_reject_unconfined_paths(bad_path: Path) -> None:
    with pytest.raises(leakage.LeakageGateError, match="memory_project"):
        leakage._resolve_project_cli_path(bad_path, name="features path")  # noqa: SLF001
