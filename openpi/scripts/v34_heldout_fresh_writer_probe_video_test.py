"""Focused fail-closed tests for heldout fresh-writer-probe videos."""

from __future__ import annotations

# The diagnostic intentionally exposes script-private pure helpers for contract tests.
# ruff: noqa: SLF001
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_SCRIPTS = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS))
try:
    import v34_heldout_fresh_writer_probe_video as video
finally:
    sys.path.remove(str(_SCRIPTS))


def _args(**overrides) -> video.Args:
    values = {
        "checkpoint": Path("checkpoint/11000"),
        "dataset_root": Path("dataset"),
        "probe_artifact_dir": Path("probe/11000/raw"),
        "output_dir": Path("output/11000"),
        "config": video.RUN5_CONFIG,
        "parameter_source": "raw",
    }
    values.update(overrides)
    return video.Args(**values)


def _spec(
    episode: int,
    prompt: str,
    side: str,
    *,
    heldout: bool,
    evidence: tuple[int, ...] = (4, 5),
) -> video.fixed.EpisodeSpec:
    return video.fixed.EpisodeSpec(
        episode=episode,
        prompt=prompt,
        side=side,
        label=video.fixed.SIDE_TO_LABEL[side],
        length=10,
        parquet=Path(f"episode_{episode:06d}.parquet"),
        evidence_frames=evidence,
        approach_frames=(2, 3),
        heldout=heldout,
    )


def _all_specs() -> list[video.fixed.EpisodeSpec]:
    banana_left = {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15}
    specs = []
    for episode in range(60):
        prompt = video.fixed.PROMPTS[0] if episode < 30 else video.fixed.PROMPTS[1]
        side = (
            ("left" if episode in banana_left else "right")
            if episode < 30
            else ("left" if episode < 45 else "right")
        )
        specs.append(
            _spec(
                episode,
                prompt,
                side,
                heldout=episode in set(video.HELDOUT_EPISODES),
            )
        )
    return specs


def _write_probe_artifact(
    root: Path,
    dataset: Path,
    specs: list[video.fixed.EpisodeSpec],
    parameter_identity: dict[str, object],
    dataset_provenance: dict[str, object],
    source_provenance: dict[str, object],
    norm_provenance: dict[str, object],
    tokenizer_provenance: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    root.mkdir(parents=True)
    labels = np.asarray([spec.label for spec in specs], dtype=np.int8)
    # Four dimensions avoid a degenerate standardizer while keeping the fixture small.
    writer = np.stack(
        [
            np.asarray(
                [
                    2.0 * label - 1.0,
                    (episode % 7) / 7.0,
                    (episode % 3) / 3.0,
                    1.0,
                ],
                dtype=np.float32,
            )
            for episode, label in enumerate(labels)
        ]
    )
    train, _folds = video.fixed._validate_design(specs)
    head = video.fixed._fit_probe(writer[np.asarray(train)], labels[np.asarray(train)])
    heldout_scores = head.scores(writer[np.asarray(video.HELDOUT_EPISODES)])
    np.savez(
        root / "features.npz",
        episode_ids=np.arange(60, dtype=np.int64),
        episode_labels=labels,
        episode_heldout=np.asarray([spec.heldout for spec in specs], dtype=bool),
        episode_evidence_counts=np.asarray([len(spec.evidence_frames) for spec in specs], dtype=np.int64),
        episode_writer_evidence=writer,
    )
    report = {
        "schema_version": video.fixed.SCHEMA_VERSION,
        "status": "complete",
        "checkpoint_step_label": 11000,
        "config": video.RUN5_CONFIG,
        "parameter_source": "raw",
        "design": {"episodes": 60, "heldout_episodes": list(video.HELDOUT_EPISODES)},
        "extraction_contract": {
            "mode": "full",
            "fresh_independent_m0_per_frame_row": True,
            "writes_during_production_extraction": False,
            "returned_state_threaded": False,
            "feature_formula": "float32 L2(mean_slots(tokens)), epsilon=1e-12",
            "episode_formula": "mean(per-frame feature), no second normalization",
            "evidence_selector": "unshifted parquet task_index == 4",
        },
        "probe_protocol": {
            "fit_unit": "one equal-weight episode feature",
            "feature_aggregation": "mean of per-frame features; no second normalization",
            "classifier": "binary logistic regression, right=1, left=0",
            "fresh_initialization": "all weights and bias exactly zero for every fold/source/checkpoint",
            "standardization": "evidence-fit episodes only; reused unchanged for matched approach",
            "l2": video.fixed.PROBE_L2,
            "steps": video.fixed.PROBE_STEPS,
            "learning_rate": video.fixed.PROBE_LR,
            "heldout_excluded_from_every_fit": True,
            "train_episode_count": video.EXPECTED_TRAIN_EPISODES,
        },
        "parameter_provenance": parameter_identity,
        "source_provenance": source_provenance,
        "normalization_asset_provenance": norm_provenance,
        "tokenizer_asset_provenance": tokenizer_provenance,
        "dataset_provenance": {
            "dataset_root": str(dataset.resolve()),
            "metadata_file_identities": {
                name: video.fixed.causal._file_identity(dataset / "meta" / name)
                for name in ("info.json", "tasks.jsonl", "episodes.jsonl", "episode_prompts.json")
            },
            "all_frame_episode_task_protocol_sha256": dataset_provenance[
                "all_frame_episode_task_protocol_sha256"
            ],
            "parquet_storage_metadata": dataset_provenance["parquet_storage_metadata"],
        },
        "fresh_probe_streams": {
            "writer": {
                "fit_all_56_test_exact_heldout_4": {
                    "episodes": [
                        {
                            "episode": episode,
                            "evidence_logit_right_minus_left": float(score),
                        }
                        for episode, score in zip(
                            video.HELDOUT_EPISODES, heldout_scores, strict=True
                        )
                    ]
                }
            }
        },
    }
    (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
    lines = []
    for name in ("features.npz", "report.json"):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (root / "COMPLETE").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return writer, heldout_scores


def _refresh_complete(root: Path) -> None:
    lines = [
        f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}"
        for name in ("features.npz", "report.json")
    ]
    (root / "COMPLETE").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_args_are_explicit_generic_numeric_and_separate_artifact_namespaces() -> None:
    args = _args()
    assert args.checkpoint_step == 11000
    assert args.episodes == video.HELDOUT_EPISODES
    assert args.artifact_dir.name == "raw"
    assert _args(parameter_source="ema", smoke_only=True).episodes == (video.SMOKE_EPISODE,)
    invalid = [
        ({"checkpoint": Path("checkpoint/latest")}, "numeric"),
        ({"config": "pi05_yam_mem_v34"}, "pinned"),
        ({"parameter_source": "both"}, "raw or ema"),
        ({"batch_size": 0}, "batch-size"),
        ({"batch_size": 65}, "batch-size"),
    ]
    for overrides, match in invalid:
        with pytest.raises(ValueError, match=match):
            _args(**overrides)


def test_evidence_prefix_is_na_before_updates_only_evidence_and_freezes_after() -> None:
    model = video.fixed.ProbeModel(
        mean=np.zeros(2, dtype=np.float64),
        scale=np.ones(2, dtype=np.float64),
        weights=np.asarray([1.0, -0.5, 0.25], dtype=np.float64),
    )
    features = np.asarray(
        [[100.0, 0.0], [1.0, 0.0], [3.0, 2.0], [-100.0, 0.0]], dtype=np.float32
    )
    mask = np.asarray([False, True, True, False])
    scores, counts = video._evidence_prefix_scores(features, mask, model)
    assert np.isnan(scores[0])
    assert counts.tolist() == [0, 1, 2, 2]
    assert scores[1] == pytest.approx(1.25)
    assert scores[2] == pytest.approx(1.75)
    assert scores[3] == pytest.approx(scores[2])
    assert video._prediction(-1e-9) == "left"
    assert video._prediction(0.0) == "left"
    assert video._prediction(1e-9) == "right"


def test_phase_category_is_direct_and_marks_matched_approach_window() -> None:
    spec = _spec(
        15,
        video.fixed.PROMPTS[0],
        "left",
        heldout=True,
        evidence=(4, 5),
    )
    assert video._phase_category(spec, 2, 0) == "MATCHED PRE-EVIDENCE"
    assert video._phase_category(spec, 4, 4) == "EVIDENCE"
    assert video._phase_category(spec, 6, 3) == "RESET"
    assert video._phase_category(spec, 6, 1) == "WAIT"
    assert video._phase_category(spec, 6, 5) == "EXECUTE"
    assert video._phase_category(spec, 0, 0) == "OPEN / PRE-EVIDENCE"


def test_manifest_is_exact_hash_checked_and_path_safe(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    for name, payload in (("features.npz", b"features"), ("report.json", b"{}")):
        (artifact / name).write_bytes(payload)
    (artifact / "COMPLETE").write_text(
        "\n".join(
            f"{hashlib.sha256((artifact / name).read_bytes()).hexdigest()}  {name}"
            for name in ("features.npz", "report.json")
        )
        + "\n",
        encoding="utf-8",
    )
    identities = video._manifest_identities(
        artifact, expected_names={"features.npz", "report.json"}
    )
    assert set(identities) == {"features.npz", "report.json", "COMPLETE"}
    (artifact / "report.json").write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        video._manifest_identities(artifact, expected_names={"features.npz", "report.json"})

    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    (unsafe / "COMPLETE").write_text(f"{'0' * 64}  ../escape\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe"):
        video._manifest_identities(unsafe, expected_names={"features.npz", "report.json"})


def test_probe_bundle_refits_exact_all56_and_reproduces_report(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    for name in ("info.json", "tasks.jsonl", "episodes.jsonl", "episode_prompts.json"):
        (dataset / "meta" / name).write_text(f"{name}\n", encoding="utf-8")
    specs = _all_specs()
    parameter_identity = {
        "parameter_source": "raw",
        "train_state_step_identity": {
            "checkpoint_manager_step_label": 11000,
            "internal_train_state_step": 11001,
        },
        "restored_parameter_tree_identity": {"sha256": "a" * 64},
    }
    dataset_provenance = {
        "all_frame_episode_task_protocol_sha256": "b" * 64,
        "parquet_storage_metadata": [
            {
                "episode": episode,
                "resolved_path": str((dataset / f"episode_{episode:06d}.parquet").resolve()),
                "bytes": 1000 + episode,
                "mtime_ns": 2000 + episode,
                "parquet_rows": specs[episode].length,
            }
            for episode in range(60)
        ],
    }
    source_provenance = {
        "launch_snapshot_identity": {"bytes": 42, "sha256": "c" * 64},
    }
    norm_provenance = {
        "files": {"norm_stats.json": {"bytes": 12, "sha256": "d" * 64}},
    }
    tokenizer_provenance = {
        "all_expected_hashes_match": True,
        "fast_expected_commit": "commit",
        "fast_snapshot": {
            "files": {"tokenizer.json": {"bytes": 13, "sha256": "e" * 64}},
        },
        "paligemma_model": {"bytes": 14, "sha256": "f" * 64},
    }
    artifact = tmp_path / "probe" / "11000" / "raw"
    writer, heldout_scores = _write_probe_artifact(
        artifact,
        dataset,
        specs,
        parameter_identity,
        dataset_provenance,
        source_provenance,
        norm_provenance,
        tokenizer_provenance,
    )
    args = _args(
        dataset_root=dataset,
        probe_artifact_dir=artifact,
        checkpoint=tmp_path / "checkpoint" / "11000",
        output_dir=tmp_path / "output" / "11000",
    )
    runtime = SimpleNamespace(
        parameter_provenance=parameter_identity,
        source_provenance=source_provenance,
        norm_provenance=norm_provenance,
        tokenizer_provenance=tokenizer_provenance,
    )
    bundle = video._load_probe_bundle(
        args, runtime, specs, dataset_provenance
    )
    assert len(bundle.train_indices) == 56
    assert not set(bundle.train_indices) & set(video.HELDOUT_EPISODES)
    np.testing.assert_array_equal(bundle.episode_writer_evidence, writer)
    np.testing.assert_allclose(
        [bundle.heldout_report_logits[episode] for episode in video.HELDOUT_EPISODES],
        heldout_scores,
        atol=0.0,
    )

    bad = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    bad["probe_protocol"]["heldout_excluded_from_every_fit"] = False
    (artifact / "report.json").write_text(json.dumps(bad), encoding="utf-8")
    _refresh_complete(artifact)
    with pytest.raises(ValueError, match="fitting contract"):
        video._load_probe_bundle(args, runtime, specs, dataset_provenance)


def test_probe_bundle_rejects_dataset_source_norm_and_tokenizer_mismatches(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "meta").mkdir(parents=True)
    for name in ("info.json", "tasks.jsonl", "episodes.jsonl", "episode_prompts.json"):
        (dataset / "meta" / name).write_text(f"{name}\n", encoding="utf-8")
    specs = _all_specs()
    parameter_identity = {
        "parameter_source": "raw",
        "train_state_step_identity": {
            "checkpoint_manager_step_label": 11000,
            "internal_train_state_step": 11001,
        },
        "restored_parameter_tree_identity": {"sha256": "a" * 64},
    }
    dataset_provenance = {
        "all_frame_episode_task_protocol_sha256": "b" * 64,
        "parquet_storage_metadata": [{"episode": episode} for episode in range(60)],
    }
    source_provenance = {"launch_snapshot_identity": {"sha256": "c" * 64}}
    norm_provenance = {"files": {"norm_stats.json": {"sha256": "d" * 64}}}
    tokenizer_provenance = {
        "all_expected_hashes_match": True,
        "fast_expected_commit": "commit",
        "fast_snapshot": {"files": {"tokenizer.json": {"sha256": "e" * 64}}},
        "paligemma_model": {"sha256": "f" * 64},
    }
    artifact = tmp_path / "probe" / "11000" / "raw"
    _write_probe_artifact(
        artifact,
        dataset,
        specs,
        parameter_identity,
        dataset_provenance,
        source_provenance,
        norm_provenance,
        tokenizer_provenance,
    )
    args = _args(
        dataset_root=dataset,
        probe_artifact_dir=artifact,
        checkpoint=tmp_path / "checkpoint" / "11000",
        output_dir=tmp_path / "output" / "11000",
    )
    runtime = SimpleNamespace(
        parameter_provenance=parameter_identity,
        source_provenance=source_provenance,
        norm_provenance=norm_provenance,
        tokenizer_provenance=tokenizer_provenance,
    )

    bad_dataset = copy.deepcopy(dataset_provenance)
    bad_dataset["all_frame_episode_task_protocol_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="all_frame_episode_task_protocol_sha256"):
        video._load_probe_bundle(args, runtime, specs, bad_dataset)
    bad_storage = copy.deepcopy(dataset_provenance)
    bad_storage["parquet_storage_metadata"][0]["episode"] = 999
    with pytest.raises(ValueError, match="parquet_storage_metadata"):
        video._load_probe_bundle(args, runtime, specs, bad_storage)

    original_report = json.loads((artifact / "report.json").read_text(encoding="utf-8"))
    mutations = (
        (
            lambda report: report["source_provenance"]["launch_snapshot_identity"].__setitem__(
                "sha256", "0" * 64
            ),
            "source snapshots",
        ),
        (
            lambda report: report["normalization_asset_provenance"]["files"].__setitem__(
                "norm_stats.json", {"sha256": "0" * 64}
            ),
            "normalization asset identities",
        ),
        (
            lambda report: report["tokenizer_asset_provenance"]["paligemma_model"].__setitem__(
                "sha256", "0" * 64
            ),
            "tokenizer identities",
        ),
    )
    for mutate, match in mutations:
        changed = copy.deepcopy(original_report)
        mutate(changed)
        (artifact / "report.json").write_text(json.dumps(changed), encoding="utf-8")
        _refresh_complete(artifact)
        with pytest.raises(ValueError, match=match):
            video._load_probe_bundle(args, runtime, specs, dataset_provenance)


def test_render_is_exact_geometry_and_preserves_uncovered_top_pixels() -> None:
    image = np.arange(video.MODEL_IMAGE_SIZE * video.MODEL_IMAGE_SIZE * 3, dtype=np.uint32)
    image = (image.reshape(video.MODEL_IMAGE_SIZE, video.MODEL_IMAGE_SIZE, 3) % 256).astype(np.uint8)
    rendered = video._render_frame(
        image,
        episode=15,
        frame=269,
        full_frame_count=876,
        fps=30.0,
        prompt="find the banana",
        truth_side="left",
        gt_subtask=video.EVIDENCE_LABEL,
        phase_category="EVIDENCE",
        instantaneous_score=-2.0,
        cumulative_score=-1.0,
        evidence_seen=1,
        evidence_total=58,
        is_evidence=True,
        checkpoint_step=11000,
        parameter_source="raw",
    )
    assert rendered.shape == (video.CANVAS_HEIGHT, video.CANVAS_WIDTH, 3)
    assert rendered.dtype == np.uint8
    enlarged = np.repeat(
        np.repeat(image, video.DISPLAY_SCALE, axis=0), video.DISPLAY_SCALE, axis=1
    )
    # Header, badge, and three-pixel border intentionally modify the image.  Interior pixels
    # below the badge must remain the exact nearest-neighbour model input.
    np.testing.assert_array_equal(
        rendered[video.HEADER_HEIGHT + 100 : -4, 4:-4],
        enlarged[100:-4, 4:-4],
    )
