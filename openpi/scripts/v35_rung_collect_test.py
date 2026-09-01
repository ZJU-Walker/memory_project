from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import subprocess
import sys

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v35_gate_artifacts as artifacts
    import v35_rung_collect as collect
    import v35_rung_eval as rung
    from v35_rung_eval_test import _manifest
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


def _raw_manifest(path: Path, manifest: artifacts.FrozenManifest) -> None:
    records = []
    for episode in manifest.episodes:
        side = "right" if episode.target_side else "left"
        records.append(
            {
                "stable_id": episode.stable_id,
                "include": True,
                "split": episode.split,
                "expected_num_frames": 200,
                "prompt": "find the banana"
                if episode.object_name == "banana"
                else "find the grey pepper box",
                "e_visibility": {
                    "first_valid_visible_frame": 30,
                    "last_clean_visible_frame": 60,
                },
                "d_valid": {"start": 120, "end": 149},
                "target_side": side,
            }
        )
    path.write_text(json.dumps({"episodes": records}), encoding="utf-8")


def test_selection_never_resolves_final_test_and_freezes_nonempty_exact_clock(tmp_path: Path, monkeypatch) -> None:
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    _raw_manifest(manifest_path, manifest)
    dataset_root = tmp_path / "dataset"
    opened: list[int] = []
    by_index = {episode.episode_index: episode for episode in manifest.episodes}

    def scalar(path: Path, *, expected_episode: int, expected_frames: int):
        del path
        opened.append(expected_episode)
        episode = by_index[expected_episode]
        frame = np.arange(expected_frames, dtype=np.int64)
        side = "right" if episode.target_side else "left"
        names = np.asarray(
            [
                "open both lids"
                if value < 15
                else "inspect both bins"
                if value < 75
                else "close both lids and reset arms"
                if value < 120
                else f"wait; target bin is {side}"
                if value < 150
                else f"open {side} bin"
                for value in frame
            ]
        )
        reverse = {name: index for index, name in enumerate(sorted(collect.EXPECTED_TASKS))}
        return frame, np.asarray([reverse[name] for name in names], dtype=np.int64)

    monkeypatch.setattr(collect, "_scalar_columns", scalar)
    monkeypatch.setattr(collect, "_sha256_file", lambda _: "a" * 64)
    monkeypatch.setattr(
        collect,
        "_task_names",
        lambda _: dict(enumerate(sorted(collect.EXPECTED_TASKS))),
    )
    monkeypatch.setattr(collect.project_paths, "project_relative_path", lambda _: PurePosixPath("data/fake"))
    envelope = collect.build_selection(
        manifest=manifest, manifest_path=manifest_path, dataset_root=dataset_root
    )

    payload = envelope["payload"]
    assert len(payload["episodes"]) == 62
    assert len(payload["task_health"]) == 16
    assert payload["final_test_payload_access_count"] == 0
    assert set(opened) == {episode.episode_index for split in ("train", "development") for episode in manifest.split(split)}
    assert not set(opened) & {episode.episode_index for episode in manifest.split("final_test")}
    assert all(record["e_frames"] == (30, 45, 60) for record in payload["episodes"])
    assert all(record["d_frames"] == (120, 135) for record in payload["episodes"])
    assert all(record["use_frames"] for record in payload["episodes"])


def _check_evidence(tmp_path: Path, *, returncode: int) -> tuple[Path, dict]:
    source_files = {
        relative: hashlib.sha256(collect.project_paths.project_path(relative).read_bytes()).hexdigest()
        for relative in rung.CHECK_SOURCE_FILES
    }
    evidence_path = tmp_path / f"evidence-{returncode}.json"
    evidence = artifacts.artifact_envelope(
        rung.CHECK_EVIDENCE_SCHEMA_VERSION,
        {
            "groups": {
                "core": {
                    "nodeids": list(rung.CORE_CHECK_NODEIDS),
                    "returncode": returncode,
                    "stderr_sha256": "1" * 64,
                    "stdout_sha256": "2" * 64,
                },
                "gradient": {
                    "nodeids": list(rung.GRADIENT_CHECK_NODEIDS),
                    "returncode": 0,
                    "stderr_sha256": "3" * 64,
                    "stdout_sha256": "4" * 64,
                },
            },
            "source_files": source_files,
        },
    )
    artifacts.write_canonical_envelope(
        evidence_path, evidence, schema_version=rung.CHECK_EVIDENCE_SCHEMA_VERSION
    )
    return evidence_path, evidence


def test_check_artifact_is_derived_from_exact_node_group_and_rejects_failed_evidence(tmp_path: Path) -> None:
    checkpoint, initialization = "5" * 64, "6" * 64
    evidence_path, evidence = _check_evidence(tmp_path, returncode=0)
    core_path = tmp_path / "core.json"
    core = artifacts.artifact_envelope(
        rung.CORE_SCHEMA_VERSION,
        {
            "checkpoint_parameter_tree_sha256": checkpoint,
            "evidence": {
                "artifact_id": evidence["artifact_id"],
                "path": evidence_path.name,
                "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            },
            "group": "core",
            "initialization_parameter_tree_sha256": initialization,
        },
    )
    artifacts.write_canonical_envelope(core_path, core, schema_version=rung.CORE_SCHEMA_VERSION)
    checks = rung._load_check_results(  # noqa: SLF001
        core_path,
        {"artifact_id": core["artifact_id"], "path": core_path.name, "sha256": hashlib.sha256(core_path.read_bytes()).hexdigest()},
        name="core_artifact",
        schema=rung.CORE_SCHEMA_VERSION,
        expected_checks=rung.CORE_CHECKS,
        checkpoint_sha256=checkpoint,
        initialization_sha256=initialization,
    )
    assert checks == dict.fromkeys(rung.CORE_CHECKS, True)

    failed_path, failed = _check_evidence(tmp_path, returncode=1)
    failed_core_path = tmp_path / "failed-core.json"
    failed_core = artifacts.artifact_envelope(
        rung.CORE_SCHEMA_VERSION,
        {
            **core["payload"],
            "evidence": {
                "artifact_id": failed["artifact_id"],
                "path": failed_path.name,
                "sha256": hashlib.sha256(failed_path.read_bytes()).hexdigest(),
            },
        },
    )
    artifacts.write_canonical_envelope(
        failed_core_path, failed_core, schema_version=rung.CORE_SCHEMA_VERSION
    )
    with pytest.raises(rung.RungEvaluationError, match="did not pass"):
        rung._load_check_results(  # noqa: SLF001
            failed_core_path,
            {
                "artifact_id": failed_core["artifact_id"],
                "path": failed_core_path.name,
                "sha256": hashlib.sha256(failed_core_path.read_bytes()).hexdigest(),
            },
            name="core_artifact",
            schema=rung.CORE_SCHEMA_VERSION,
            expected_checks=rung.CORE_CHECKS,
            checkpoint_sha256=checkpoint,
            initialization_sha256=initialization,
        )


def test_cli_requires_explicit_subcommand_and_collect_output() -> None:
    parser = collect._parser()  # noqa: SLF001
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "collect",
            "--checkpoint-step-dir",
            "v35/checkpoints/run/0",
            "--selection",
            "v35/diagnostics/rung_selection.json",
            "--manifest-sha256",
            "a" * 64,
            "--output-dir",
            "v35/diagnostics/rungs/0",
        ]
    )
    assert args.command == "collect"
    assert args.output_dir == Path("v35/diagnostics/rungs/0")


def test_collector_import_stays_accelerator_cold_before_gate_c_subprocesses() -> None:
    scripts = Path(__file__).parent.resolve()
    source = scripts.parent / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((str(source), str(scripts)))
    code = (
        "import sys; import v35_rung_collect; "
        "assert 'jax' not in sys.modules; "
        "assert 'openpi.models.model' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=environment,
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, (result.stdout + result.stderr).decode(errors="replace")
