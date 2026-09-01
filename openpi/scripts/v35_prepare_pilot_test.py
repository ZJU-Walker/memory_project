from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import v35_prepare_pilot as prepare

from openpi.shared import project_paths
from openpi.training import v35_authorization as authorization
from openpi.training import weight_loaders


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "memory_project"
    (root / "openpi/src/openpi").mkdir(parents=True)
    (root / "openpi/pyproject.toml").touch()
    monkeypatch.setenv(project_paths.MEMORY_PROJECT_ROOT_ENV, str(root))
    return root


def _envelope(payload: dict, *, schema_version: str = "test.v1") -> bytes:
    encoded = json.dumps(payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    value = {
        "artifact_id": f"sha256:{hashlib.sha256(encoded).hexdigest()}",
        "payload": payload,
        "schema_version": schema_version,
    }
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    )


def test_plan_uses_exact_cli_order_portable_layout_and_gpu_ownership(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    layout = prepare.Layout.create(experiment="pilot_01")
    stages = prepare.build_stages(
        layout=layout,
        manifest_sha256=prepare.FROZEN_MANIFEST_SHA256,
        python="python",
        gpus=("0", "1", "2", "3"),
        fsdp_devices=4,
        query_batch_size=8,
        leakage_batch_size=8,
    )

    assert [stage.name for stage in stages] == [
        "bootstrap initialize",
        "calibration preflight",
        "train-54 replay selection",
        "parallel replay collection",
        "replay seal and receipt",
        "injection calibration",
        "bootstrap finalize",
        "Gate A",
        "Gate-B feature production",
        "Gate-B reducer",
        "rung selection",
        "rung 0 collect and seal",
    ]
    assert layout.checkpoint_relative.as_posix() == "v35/checkpoints/pi05_yam_mem_v35/pilot_01"
    workers = stages[3].commands
    assert len(workers) == 4
    assert [command.env["CUDA_VISIBLE_DEVICES"] for command in workers] == ["0", "1", "2", "3"]
    assert all(command.argv[command.argv.index("--fsdp-devices") + 1] == "1" for command in workers)
    assert stages[0].commands[0].env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert stages[1].commands[0].env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert stages[8].commands[0].env["CUDA_VISIBLE_DEVICES"] == "0"
    assert stages[11].commands[0].env["CUDA_VISIBLE_DEVICES"] == "0"
    rung_command = stages[11].commands[0].argv
    assert rung_command[rung_command.index("--checkpoint-step-dir") + 1] == (
        "v35/checkpoints/pi05_yam_mem_v35/pilot_01/0"
    )
    for stage in stages:
        for command in stage.commands:
            for index, value in enumerate(command.argv[:-1]):
                if (value.startswith("--") and "dir" in value) or value in {
                    "--manifest",
                    "--dataset-root",
                    "--preflight",
                    "--selection",
                    "--output",
                    "--receipt",
                    "--calibration",
                    "--features",
                    "--step0-params",
                    "--initialization-manifest",
                    "--checkpoint-step-dir",
                    "--detail-report",
                    "--norm-stats",
                    "--norm-provenance",
                    "--calibration-descriptor",
                }:
                    candidate = command.argv[index + 1]
                    assert not candidate.startswith(str(root))


@pytest.mark.parametrize(
    ("query_batch_size", "leakage_batch_size", "message"),
    [(4, 8, "query-batch-size=8"), (8, 16, "leakage-batch-size=8")],
)
def test_sealed_pipeline_rejects_batch_protocol_drift(
    tmp_path,
    monkeypatch,
    query_batch_size,
    leakage_batch_size,
    message,
):
    _project(tmp_path, monkeypatch)
    layout = prepare.Layout.create(experiment="pilot")
    with pytest.raises(prepare.PreparePilotError, match=message):
        prepare.build_stages(
            layout=layout,
            manifest_sha256=prepare.FROZEN_MANIFEST_SHA256,
            python="python",
            gpus=("0", "1", "2", "3"),
            fsdp_devices=4,
            query_batch_size=query_batch_size,
            leakage_batch_size=leakage_batch_size,
        )


class _RecordingRunner:
    def __init__(self):
        self.commands = []

    def run(self, command):
        self.commands.append(command)
        return ""

    def run_parallel(self, commands):
        self.commands.extend(commands)


def test_resume_rejects_source_or_runtime_drift_before_any_command(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    layout = prepare.Layout.create(experiment="pilot")
    frozen = {
        "aggregate_sha256": "a" * 64,
        "files": {"openpi/src/openpi/example.py": "b" * 64},
        "protocol": "openpi.v35.training-and-gate-source-tree.v2",
        "runtime": {"distributions": {}, "python_implementation": "CPython", "python_version": "3.11.0"},
        "selection": {},
    }
    monkeypatch.setattr(authorization, "semantic_source_identity", lambda: frozen)
    prepare._freeze_or_validate_source_identity(layout, existing_lineage=False)  # noqa: SLF001

    changed = {**frozen, "aggregate_sha256": "c" * 64}
    monkeypatch.setattr(authorization, "semantic_source_identity", lambda: changed)
    runner = _RecordingRunner()
    with pytest.raises(prepare.PreparePilotError, match="use a new experiment"):
        prepare.prepare(
            layout=layout,
            manifest_sha256=prepare.FROZEN_MANIFEST_SHA256,
            gpus=("0", "1", "2", "3"),
            fsdp_devices=4,
            query_batch_size=8,
            leakage_batch_size=8,
            resume=True,
            plan_only=False,
            runner=runner,
            python="python",
        )
    assert runner.commands == []


class _NoRun:
    def run(self, command):  # pragma: no cover - failure explains an orchestration regression
        raise AssertionError(command)


def test_resume_validates_complete_group_and_rejects_tamper_or_partial(tmp_path, monkeypatch, capsys):
    _project(tmp_path, monkeypatch)
    output = project_paths.project_path("v35/diagnostics/runs/pilot/gate.json")
    output.parent.mkdir(parents=True)
    output.write_bytes(_envelope({"status": "pass"}))
    group = prepare.ImmutableGroup("gate", (output,), output)
    stage = prepare.Stage("gate", (), group)
    prepare._run_immutable_stage(stage, resume=True, runner=_NoRun())  # noqa: SLF001
    assert "skip validated" in capsys.readouterr().out

    value = json.loads(output.read_bytes())
    value["payload"]["status"] = "fail"
    output.write_bytes(json.dumps(value, separators=(",", ":"), sort_keys=True).encode() + b"\n")
    with pytest.raises(prepare.PreparePilotError, match="payload hash"):
        prepare._run_immutable_stage(stage, resume=True, runner=_NoRun())  # noqa: SLF001

    second = output.with_name("second.json")
    partial = prepare.Stage("pair", (), prepare.ImmutableGroup("pair", (output, second)))
    with pytest.raises(prepare.PreparePilotError, match="partial create-only"):
        prepare._run_immutable_stage(partial, resume=True, runner=_NoRun())  # noqa: SLF001


class _ReplayRunner:
    def __init__(self, layout: prepare.Layout):
        self.layout = layout
        self.commands = ()

    def run_parallel(self, commands):
        self.commands = tuple(commands)
        root = self.layout.absolute(self.layout.replay_shards)
        root.mkdir(parents=True, exist_ok=True)
        for path in prepare._replay_paths(self.layout):  # noqa: SLF001
            if not path.exists():
                path.write_bytes(b"immutable-shard")


def test_resume_parallel_replay_fills_only_missing_shards_one_process_per_gpu(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    layout = prepare.Layout.create(experiment="pilot")
    stages = prepare.build_stages(
        layout=layout,
        manifest_sha256=prepare.FROZEN_MANIFEST_SHA256,
        python="python",
        gpus=("0", "1", "2", "3"),
        fsdp_devices=4,
        query_batch_size=8,
        leakage_batch_size=8,
    )
    replay_stage = stages[3]
    sealed = stages[4].outputs
    assert sealed is not None
    first = prepare._replay_paths(layout)[0]  # noqa: SLF001
    first.parent.mkdir(parents=True)
    first.write_bytes(b"already-complete")
    runner = _ReplayRunner(layout)

    prepare._run_replay_stage(  # noqa: SLF001
        replay_stage,
        layout=layout,
        resume=True,
        sealed_group=sealed,
        runner=runner,
    )

    assert len(runner.commands) == 4
    assert [command.env["CUDA_VISIBLE_DEVICES"] for command in runner.commands] == ["0", "1", "2", "3"]
    assert first.read_bytes() == b"already-complete"
    assert all(path.is_file() for path in prepare._replay_paths(layout))  # noqa: SLF001


class _FakeProcess:
    def __init__(self, returncode: int | None):
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


def test_parallel_runner_terminates_peers_on_first_failure(tmp_path, monkeypatch):
    failed = _FakeProcess(7)
    running = _FakeProcess(None)
    processes = iter((failed, running))
    monkeypatch.setattr(prepare.subprocess, "Popen", lambda *args, **kwargs: next(processes))
    runner = prepare.CommandRunner(cwd=tmp_path, base_env={})
    commands = (prepare.Command(("worker-0",)), prepare.Command(("worker-1",)))

    with pytest.raises(prepare.PreparePilotError, match=r"failed \(7\)"):
        runner.run_parallel(commands)
    assert running.terminated
    assert not running.killed


def test_bootstrap_preflight_crosscheck_rejects_tree_mismatch(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    layout = prepare.Layout.create(experiment="pilot")
    provisional = layout.absolute(layout.provisional)
    preflight = layout.absolute(layout.preflight)
    provisional.parent.mkdir(parents=True)
    preflight.parent.mkdir(parents=True)

    provisional_payload = {
        "actual_step0_parameter_tree_sha256": "a" * 64,
        "official_source_tree_sha256": "b" * 64,
        "target_schema_sha256": "c" * 64,
        "graft_manifest": {"manifest_sha256": "d" * 64},
        "initialization_seed": 42,
        "official_source_uri": "gs://openpi-assets/checkpoints/pi05_base/params",
    }
    provisional_payload["provisional_identity_sha256"] = hashlib.sha256(
        json.dumps(provisional_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    provisional.write_bytes(
        json.dumps(provisional_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    )
    preflight.write_bytes(
        _envelope(
            {
                "config": {
                    "seed": 42,
                    "official_base_uri": "gs://openpi-assets/checkpoints/pi05_base/params",
                },
                "initialization": {
                    "actual_step0_parameter_tree_sha256": "f" * 64,
                    "official_base_source_tree_sha256": "b" * 64,
                    "target_schema_sha256": "c" * 64,
                    "graft_manifest_sha256": "d" * 64,
                },
            }
        )
    )
    with pytest.raises(prepare.PreparePilotError, match="actual step-0 parameter tree"):
        prepare._validate_bootstrap_preflight_binding(layout)  # noqa: SLF001


def test_bootstrap_preflight_crosscheck_rejects_dataset_mismatch(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    layout = prepare.Layout.create(experiment="pilot")
    provisional = layout.absolute(layout.provisional)
    preflight = layout.absolute(layout.preflight)
    norm_provenance = layout.absolute(layout.norm_provenance)
    provisional.parent.mkdir(parents=True)
    preflight.parent.mkdir(parents=True)
    norm_provenance.parent.mkdir(parents=True)
    norm_provenance.write_bytes(
        json.dumps(
            {"selection": {"dataset_episode_frame_protocol_sha256": "e" * 64}},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    dataset_identity = {
        "episode_manifest_path_relative": layout.manifest_relative.as_posix(),
        "episode_manifest_sha256": "1" * 64,
        "norm_stats_path_relative": layout.norm_stats.as_posix(),
        "norm_stats_sha256": "2" * 64,
        "norm_stats_provenance_path_relative": layout.norm_provenance.as_posix(),
        "norm_stats_provenance_sha256": "3" * 64,
        "train_storage_sha256": "4" * 64,
    }
    provisional_payload = {
        "actual_step0_parameter_tree_sha256": "a" * 64,
        "official_source_tree_sha256": "b" * 64,
        "target_schema_sha256": "c" * 64,
        "graft_manifest": {"manifest_sha256": "d" * 64},
        "initialization_seed": 42,
        "official_source_uri": "gs://openpi-assets/checkpoints/pi05_base/params",
        "dataset_identity": dataset_identity,
    }
    provisional_payload["provisional_identity_sha256"] = hashlib.sha256(
        json.dumps(provisional_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    provisional.write_bytes(
        json.dumps(provisional_payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    )
    preflight.write_bytes(
        _envelope(
            {
                "config": {
                    "seed": 42,
                    "official_base_uri": "gs://openpi-assets/checkpoints/pi05_base/params",
                },
                "initialization": {
                    "actual_step0_parameter_tree_sha256": "a" * 64,
                    "official_base_source_tree_sha256": "b" * 64,
                    "target_schema_sha256": "c" * 64,
                    "graft_manifest_sha256": "d" * 64,
                },
                "data": {
                    "production": True,
                    "path_contract": "memory_project-relative-v1",
                    "dataset_repo_id": project_paths.V35_REPO_ID,
                    "manifest_path_relative": layout.manifest_relative.as_posix(),
                    "manifest_sha256": "1" * 64,
                    "norm_stats_path_relative": layout.norm_stats.as_posix(),
                    "norm_stats_sha256": "2" * 64,
                    "norm_provenance_path_relative": layout.norm_provenance.as_posix(),
                    "norm_provenance_sha256": "3" * 64,
                    "train_storage_sha256": "f" * 64,
                    "dataset_episode_frame_protocol_sha256": "e" * 64,
                    "split": "train",
                    "split_seed": 36,
                    "train_episode_count": 54,
                },
            }
        )
    )

    with pytest.raises(prepare.PreparePilotError, match="train storage SHA256"):
        prepare._validate_bootstrap_preflight_binding(layout)  # noqa: SLF001


def test_gate_b_fail_stops_before_development_rung(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    layout = prepare.Layout.create(experiment="pilot")
    gate_b = layout.absolute(layout.gate_b)
    gate_b.parent.mkdir(parents=True)
    gate_b.write_bytes(
        _envelope(
            {
                "status": "fail",
                "gates": {"passes": False},
                "decision": "stop_natural_branch_recollect_or_name_new_branch",
            },
            schema_version=prepare.GATE_B_SCHEMA_VERSION,
        )
    )

    with pytest.raises(prepare.PreparePilotError, match="before any development-set"):
        prepare._validate_gate_b_pass(layout)  # noqa: SLF001


def test_sealed_layout_rejects_custom_dataset_split_brain(tmp_path, monkeypatch):
    _project(tmp_path, monkeypatch)
    layout = prepare.Layout.create(
        experiment="pilot",
        dataset_root=prepare.PurePosixPath("data/lerobot/custom"),
    )

    with pytest.raises(prepare.PreparePilotError, match="registered manifest, dataset root"):
        prepare._validate_sealed_layout(layout)  # noqa: SLF001


def test_inventory_hashes_train_and_dev_but_never_opens_final_payload(tmp_path, monkeypatch):
    root = _project(tmp_path, monkeypatch)
    layout = prepare.Layout.create(
        experiment="pilot",
        dataset_root=prepare.PurePosixPath("data/lerobot/yam/bin_memory_0830_0831_v36_subtask"),
    )
    dataset = layout.absolute(layout.dataset_relative)
    records = []
    episodes = []
    for index in range(70):
        split = "train" if index < 54 else "development" if index < 62 else "final_test"
        episodes.append({"episode_index": index, "include": True, "split": split})
        relative = f"data/chunk-000/episode_{index:06d}.parquet"
        payload = f"payload-{index:06d}".encode()
        records.append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)})
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    for index in range(8):
        relative = f"meta/fixture_{index:03d}.json"
        payload = f"meta-{index:03d}".encode()
        path = dataset / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        records.append({"path": relative, "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)})

    manifest_path = root / "data/manifest.json"
    manifest_path.write_bytes(
        json.dumps({"episodes": episodes}, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True).encode()
        + b"\n"
    )
    inventory_path = root / "data/inventory.json"
    inventory = {
        "dataset_repo_id": project_paths.V35_REPO_ID,
        "directories": [],
        "directory_count": 108,
        "file_count": 78,
        "files": records,
        "schema_version": "openpi.v35.dataset-tree-inventory.v1",
        "total_bytes": 41_632_438_815,
        "tree_sha256": prepare.FROZEN_DATASET_TREE_SHA256,
    }
    inventory_path.write_bytes(
        json.dumps(inventory, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
    )

    hashed_paths = []
    sha256_file = prepare._sha256_file  # noqa: SLF001

    def recording_sha256(path):
        hashed_paths.append(path)
        return sha256_file(path)

    monkeypatch.setattr(prepare, "_sha256_file", recording_sha256)
    prepare._validate_dataset_inventory(  # noqa: SLF001
        layout,
        inventory_path=inventory_path,
        manifest_path=manifest_path,
        verify_bytes=True,
    )
    final_names = {f"episode_{index:06d}.parquet" for index in range(62, 70)}
    assert not any(path.name in final_names for path in hashed_paths)
    changed = dataset / "data/chunk-000/episode_000054.parquet"
    changed.write_bytes(b"corrupt-000054")
    assert changed.stat().st_size == records[54]["size"]
    with pytest.raises(prepare.PreparePilotError, match="SHA256 mismatch"):
        prepare._validate_dataset_inventory(  # noqa: SLF001
            layout,
            inventory_path=inventory_path,
            manifest_path=manifest_path,
            verify_bytes=True,
        )


def _real_graft_manifest(tmp_path):
    root = tmp_path / "ckpt" / "1000"
    (root / "train_state").mkdir(parents=True)
    loader = weight_loaders.AuditedRawCheckpointWeightLoader(
        checkpoint_path=str(root),
        enabled=True,
        matched_allowlist=(r"model/.*",),
    )
    result = weight_loaders._audit_and_graft(  # noqa: SLF001
        {"model": {"backbone": np.asarray([1.0], dtype=np.float32)}},
        {"model": {"backbone": np.zeros((1,), dtype=np.float32)}},
        loader,
        root,
    )
    return result.manifest


def test_written_graft_manifest_passes_immutable_stage_loader(tmp_path):
    manifest = _real_graft_manifest(tmp_path)
    manifest_path = tmp_path / "initialization_graft_manifest.json"

    weight_loaders._write_manifest(manifest_path, manifest)  # noqa: SLF001

    raw = manifest_path.read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\n\n")
    loaded = prepare._load_immutable_json(manifest_path)  # noqa: SLF001
    assert loaded["manifest_sha256"] == manifest.to_dict()["manifest_sha256"]


def test_pretty_printed_graft_manifest_is_rejected_by_immutable_stage_loader(tmp_path):
    manifest = _real_graft_manifest(tmp_path)
    manifest_path = tmp_path / "initialization_graft_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )

    with pytest.raises(prepare.PreparePilotError, match="not canonical"):
        prepare._load_immutable_json(manifest_path)  # noqa: SLF001


def test_non_finite_json_constant_is_rejected_cleanly(tmp_path):
    path = tmp_path / "attack.json"
    path.write_bytes(b'{"a":NaN}\n')
    with pytest.raises(prepare.PreparePilotError, match="non-finite"):
        prepare._load_immutable_json(path)  # noqa: SLF001
    path.write_bytes(b'{"a":-Infinity}\n')
    with pytest.raises(prepare.PreparePilotError, match="non-finite"):
        prepare._load_immutable_json(path)  # noqa: SLF001


def test_written_initialization_identity_passes_immutable_stage_loader(tmp_path):
    import train as train_script

    payload = {"format_version": 2, "initialization_seed": 42}
    payload["identity_sha256"] = hashlib.sha256(train_script._canonical_json(payload).encode()).hexdigest()  # noqa: SLF001
    identity_path = tmp_path / "initialization_manifest.json"

    train_script._write_json_once(identity_path, payload)  # noqa: SLF001

    loaded = prepare._load_immutable_json(identity_path)  # noqa: SLF001
    assert loaded["identity_sha256"] == payload["identity_sha256"]
    train_script._write_json_once(identity_path, payload)  # noqa: SLF001


def test_frozen_norm_provenance_loader_accepts_pretty_frozen_bytes(tmp_path):
    path = tmp_path / "norm_stats_provenance.json"
    path.write_text(json.dumps({"selection": {"episodes": 54}}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    value = prepare._load_frozen_json(path)  # noqa: SLF001
    assert value["selection"] == {"episodes": 54}

    path.write_bytes(b'{"a": 1, "a": 2}\n')
    with pytest.raises(prepare.PreparePilotError, match="duplicate"):
        prepare._load_frozen_json(path)  # noqa: SLF001
