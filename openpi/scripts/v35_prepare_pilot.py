"""Prepare and authenticate a v3.5 pilot without running an optimizer update.

This is a portable orchestration layer over the existing write-once v3.5 CLIs.  It owns no
scientific reducer logic: each stage is executed by its production CLI, and the final command
is ``v35_train.py --verify-only``.  All generated paths are below
``v35/diagnostics/runs/<experiment>`` except the finalized checkpoint, whose registered home is
``v35/checkpoints/pi05_yam_mem_v35/<experiment>``.

Normal mode requires a completely new experiment.  ``--resume`` validates and skips complete
immutable output groups.  A partially written create-only group fails closed; the sole
exception is replay shards, because their producer authenticates every existing episode shard
and creates only missing shards.  One replay process is launched per requested GPU.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shlex
import subprocess
import sys
import time
from typing import Any

from openpi.shared import project_paths

CONFIG_NAME = "pi05_yam_mem_v35"
EXPECTED_REPLAY_SHARDS = 54
FROZEN_MANIFEST_SHA256 = "9085fe50d7b02ea65930f3647ce0413e0583a66d430484e06c60812c52af8442"
FROZEN_DATASET_INVENTORY_SHA256 = "193561bbde0fc5de586140e8cf4a9432b6d4f8b590176e9bd85519170d08d172"
FROZEN_DATASET_TREE_SHA256 = "43cf91f5a18ec411d26f926a5d3ace3d8a235f0e8201b9c3ef816f094d145e93"
FROZEN_NORM_STATS_SHA256 = "5535ea95ad7ed1edc399ba47e278285a1fdec02a589451cec0dc9003d458519c"
FROZEN_NORM_PROVENANCE_SHA256 = "36e0ab51d53272038e1b204c752b30fa6ab000096bff9ff6dccd605166188c58"
DEFAULT_MANIFEST = PurePosixPath(project_paths.V35_FROZEN_MANIFEST)
DEFAULT_DATASET_ROOT = PurePosixPath(project_paths.V35_DATASET_DIR)
DEFAULT_ASSETS_DIR = PurePosixPath(project_paths.V35_ASSETS_DIR)
DEFAULT_NORM_DIR = DEFAULT_ASSETS_DIR / PurePosixPath(project_paths.V35_REPO_ID)
DEFAULT_DATASET_INVENTORY = PurePosixPath("data/0830_0831_v36_dataset_tree_inventory.json")
GATE_B_SCHEMA_VERSION = "openpi.v35.leakage-gate-decision.v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EXPERIMENT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class PreparePilotError(RuntimeError):
    """Raised before an unsafe, ambiguous, or incomplete orchestration action."""


@dataclasses.dataclass(frozen=True)
class Command:
    argv: tuple[str, ...]
    env: Mapping[str, str] = dataclasses.field(default_factory=dict)
    capture_stdout: bool = False

    def display(self) -> str:
        prefix = " ".join(f"{name}={shlex.quote(value)}" for name, value in sorted(self.env.items()))
        command = shlex.join(self.argv)
        return f"{prefix} {command}" if prefix else command


@dataclasses.dataclass(frozen=True)
class ImmutableGroup:
    name: str
    paths: tuple[Path, ...]
    primary_json: Path | None = None

    def presence(self) -> tuple[bool, ...]:
        return tuple(path.exists() for path in self.paths)


@dataclasses.dataclass(frozen=True)
class Stage:
    name: str
    commands: tuple[Command, ...]
    outputs: ImmutableGroup | None
    parallel: bool = False


@dataclasses.dataclass(frozen=True)
class Layout:
    experiment: str
    run_relative: PurePosixPath
    checkpoint_relative: PurePosixPath
    manifest_relative: PurePosixPath
    dataset_relative: PurePosixPath
    assets_relative: PurePosixPath
    norm_relative: PurePosixPath

    @classmethod
    def create(
        cls,
        *,
        experiment: str,
        manifest: PurePosixPath = DEFAULT_MANIFEST,
        dataset_root: PurePosixPath = DEFAULT_DATASET_ROOT,
        assets_dir: PurePosixPath = DEFAULT_ASSETS_DIR,
    ) -> Layout:
        _require_experiment(experiment)
        manifest = _normalized_relative(manifest, name="manifest")
        dataset_root = _normalized_relative(dataset_root, name="dataset root")
        assets_dir = _normalized_relative(assets_dir, name="assets directory")
        return cls(
            experiment=experiment,
            run_relative=PurePosixPath("v35/diagnostics/runs") / experiment,
            checkpoint_relative=PurePosixPath("v35/checkpoints") / CONFIG_NAME / experiment,
            manifest_relative=manifest,
            dataset_relative=dataset_root,
            assets_relative=assets_dir,
            norm_relative=assets_dir / PurePosixPath(project_paths.V35_REPO_ID),
        )

    def absolute(self, relative: PurePosixPath) -> Path:
        return project_paths.project_path(relative)

    @property
    def calibration_root(self) -> PurePosixPath:
        return self.run_relative / "calibration"

    @property
    def preflight_dir(self) -> PurePosixPath:
        return self.calibration_root / "preflight"

    @property
    def preflight(self) -> PurePosixPath:
        return self.preflight_dir / "preflight.json"

    @property
    def calibration_selection(self) -> PurePosixPath:
        return self.calibration_root / "frame_selection.json"

    @property
    def replay_shards(self) -> PurePosixPath:
        return self.calibration_root / "replay_shards"

    @property
    def replay_npz(self) -> PurePosixPath:
        return self.calibration_root / "train54_replay.npz"

    @property
    def replay_receipt(self) -> PurePosixPath:
        return self.calibration_root / "collection_receipt.json"

    @property
    def calibration(self) -> PurePosixPath:
        return self.calibration_root / "injection_calibration.json"

    @property
    def gates_root(self) -> PurePosixPath:
        return self.run_relative / "gates"

    @property
    def gate_a_report(self) -> PurePosixPath:
        return self.gates_root / "gate_a_detail_report.json"

    @property
    def gate_a(self) -> PurePosixPath:
        return self.gates_root / "gate_a.json"

    @property
    def leakage_features(self) -> PurePosixPath:
        return self.gates_root / "gate_b_features.json"

    @property
    def gate_b(self) -> PurePosixPath:
        return self.gates_root / "gate_b.json"

    @property
    def rung_selection(self) -> PurePosixPath:
        return self.run_relative / "rung_selection.json"

    @property
    def rung0_dir(self) -> PurePosixPath:
        return self.run_relative / "rungs/0"

    @property
    def rung0(self) -> PurePosixPath:
        return self.rung0_dir / "pilot_rung_result.json"

    @property
    def pilot_authorization(self) -> PurePosixPath:
        return self.run_relative / "pilot_authorization.json"

    @property
    def prepilot_source_identity(self) -> PurePosixPath:
        return self.run_relative / "prepilot_source_identity.json"

    @property
    def norm_stats(self) -> PurePosixPath:
        return self.norm_relative / "norm_stats.json"

    @property
    def norm_provenance(self) -> PurePosixPath:
        return self.norm_relative / "norm_stats_provenance.json"

    @property
    def provisional(self) -> PurePosixPath:
        return self.checkpoint_relative / "step0_bootstrap_provisional.json"

    @property
    def initialization(self) -> PurePosixPath:
        return self.checkpoint_relative / "initialization_manifest.json"

    @property
    def checkpoint0(self) -> PurePosixPath:
        return self.checkpoint_relative / "0"


class CommandRunner:
    """Subprocess adapter kept injectable for focused orchestration tests."""

    def __init__(self, *, cwd: Path, base_env: Mapping[str, str]):
        self.cwd = cwd
        self.base_env = dict(base_env)

    def _environment(self, command: Command) -> dict[str, str]:
        return {**self.base_env, **command.env}

    def run(self, command: Command) -> str:
        try:
            result = subprocess.run(
                command.argv,
                cwd=self.cwd,
                env=self._environment(command),
                check=True,
                text=True,
                stdout=subprocess.PIPE if command.capture_stdout else None,
            )
        except subprocess.CalledProcessError as exc:
            raise PreparePilotError(f"stage command failed ({exc.returncode}): {command.display()}") from exc
        return result.stdout or ""

    def run_parallel(self, commands: Sequence[Command]) -> None:
        processes: list[tuple[Command, subprocess.Popen[str]]] = []
        try:
            for command in commands:
                process = subprocess.Popen(
                    command.argv,
                    cwd=self.cwd,
                    env=self._environment(command),
                    text=True,
                )
                processes.append((command, process))
            pending = set(range(len(processes)))
            while pending:
                completed: list[int] = []
                for index in pending:
                    command, process = processes[index]
                    returncode = process.poll()
                    if returncode is None:
                        continue
                    completed.append(index)
                    if returncode:
                        for other_index in pending:
                            other_process = processes[other_index][1]
                            if other_index != index and other_process.poll() is None:
                                other_process.terminate()
                        self._reap(processes)
                        raise PreparePilotError(
                            f"parallel replay collection failed ({returncode}): {command.display()}"
                        )
                pending.difference_update(completed)
                if pending:
                    time.sleep(0.1)
        except BaseException:
            for _, process in processes:
                if process.poll() is None:
                    process.terminate()
            self._reap(processes)
            raise

    @staticmethod
    def _reap(processes: Sequence[tuple[Command, subprocess.Popen[str]]]) -> None:
        for _, process in processes:
            if process.poll() is not None:
                continue
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _validate_sealed_layout(layout: Layout) -> None:
    if (
        layout.manifest_relative != DEFAULT_MANIFEST
        or layout.dataset_relative != DEFAULT_DATASET_ROOT
        or layout.assets_relative != DEFAULT_ASSETS_DIR
    ):
        raise PreparePilotError(
            "the sealed production pipeline requires the registered manifest, dataset root, and assets directory"
        )


def _require_experiment(value: str) -> str:
    if _EXPERIMENT_RE.fullmatch(value) is None or value in (".", ".."):
        raise PreparePilotError(
            "experiment name must be one portable component containing only letters, digits, '.', '_' or '-'"
        )
    return value


def _normalized_relative(value: str | PurePosixPath, *, name: str) -> PurePosixPath:
    relative = PurePosixPath(value)
    if (
        not relative.parts
        or relative.is_absolute()
        or any(part in ("", ".", "..") for part in relative.parts)
        or relative.as_posix() != str(value)
    ):
        raise PreparePilotError(f"{name} must be a normalized memory_project-relative POSIX path")
    return relative


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                digest.update(block)
    except OSError as exc:
        raise PreparePilotError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _reject_json_constant(name: str) -> Any:
    # Without this hook json.loads accepts NaN/Infinity, which _canonical_json
    # (allow_nan=False) later re-raises as an uncaught ValueError instead of a
    # clean fail-closed decision.
    raise PreparePilotError(f"JSON contains forbidden non-finite constant {name!r}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PreparePilotError(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def _canonical_json(value: Any, *, ensure_ascii: bool) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=ensure_ascii,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _load_immutable_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_strict_object, parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparePilotError(f"cannot read immutable JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreparePilotError(f"immutable JSON must contain an object: {path}")
    if raw not in (
        _canonical_json(value, ensure_ascii=False) + b"\n",
        _canonical_json(value, ensure_ascii=True) + b"\n",
    ):
        raise PreparePilotError(f"immutable JSON is not canonical with one trailing newline: {path}")
    payload = value.get("payload")
    if isinstance(payload, dict):
        digest = hashlib.sha256(_canonical_json(payload, ensure_ascii=False)).hexdigest()
        alternate = hashlib.sha256(_canonical_json(payload, ensure_ascii=True)).hexdigest()
        valid_digests = {digest, alternate}
        if "artifact_id" in value:
            if value.get("artifact_id") not in {f"sha256:{item}" for item in valid_digests}:
                raise PreparePilotError(f"immutable envelope payload hash is invalid: {path}")
        elif "artifact_sha256" in value:
            recorded = value.get("artifact_sha256")
            id_values = [item for key, item in value.items() if key.endswith("_id")]
            if recorded not in valid_digests or f"sha256:{recorded}" not in id_values:
                raise PreparePilotError(f"immutable custom envelope payload hash is invalid: {path}")
    else:
        for key in ("provisional_identity_sha256", "identity_sha256", "manifest_sha256"):
            if key not in value:
                continue
            unsigned = {name: item for name, item in value.items() if name != key}
            digests = {
                hashlib.sha256(_canonical_json(unsigned, ensure_ascii=False)).hexdigest(),
                hashlib.sha256(_canonical_json(unsigned, ensure_ascii=True)).hexdigest(),
            }
            if value[key] not in digests:
                raise PreparePilotError(f"immutable {key} is invalid: {path}")
            break
    return value


def _load_frozen_json(path: Path) -> dict[str, Any]:
    """Load a byte-hashed frozen JSON file without imposing a new encoding.

    The approved episode manifest predates the canonical single-line artifact envelope
    convention and is intentionally pretty-printed.  Its exact SHA256 is authenticated by
    ``_validate_inputs``; reformatting it would invalidate every downstream dataset binding.
    Duplicate keys and non-object roots remain forbidden.
    """

    try:
        value = json.loads(path.read_bytes(), object_pairs_hook=_strict_object, parse_constant=_reject_json_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparePilotError(f"cannot read frozen JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreparePilotError(f"frozen JSON must contain an object: {path}")
    return value


def _authorization_module():
    # Keep the orchestration CLI lightweight through --help/--plan input validation.  The
    # authorization module imports JAX-backed helpers and is needed only when a real run freezes
    # or revalidates its executable source/runtime identity.
    import openpi.training.v35_authorization as authorization

    return authorization


def _validate_frozen_source_identity(layout: Layout) -> Any:
    """Require the current source/runtime bytes to equal this run's create-once snapshot."""

    try:
        return _authorization_module().load_and_validate_frozen_source_identity(
            layout.absolute(layout.prepilot_source_identity)
        )
    except (OSError, ValueError) as exc:
        raise PreparePilotError(str(exc)) from exc


def _freeze_or_validate_source_identity(layout: Layout, *, existing_lineage: bool) -> Any:
    """Create the run snapshot before stage 1, or authenticate it before a resume."""

    authorization = _authorization_module()
    path = layout.absolute(layout.prepilot_source_identity)
    if not path.exists():
        if existing_lineage:
            raise PreparePilotError(
                "pre-pilot lineage predates the frozen source/runtime identity; use a new experiment"
            )
        value = authorization.frozen_source_identity_envelope()
        encoded = authorization.canonical_json_bytes(value) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError:
            # A concurrent creator is acceptable only if it committed the identical current
            # identity; the loader below performs the complete canonical/hash/current check.
            pass
    return _validate_frozen_source_identity(layout)


def _validate_linked_files(value: Any, *, owner: Path) -> None:
    if isinstance(value, list):
        for item in value:
            _validate_linked_files(item, owner=owner)
        return
    if not isinstance(value, dict):
        return
    sha = value.get("sha256")
    relative: str | None = None
    base: Path | None = None
    if isinstance(sha, str) and isinstance(value.get("path_relative"), str):
        relative = value["path_relative"]
        try:
            base = project_paths.memory_project_root()
        except project_paths.ProjectRootError as exc:
            raise PreparePilotError(str(exc)) from exc
    elif isinstance(sha, str) and isinstance(value.get("path"), str):
        relative = value["path"]
        base = owner.parent
    if relative is not None and base is not None:
        candidate = PurePosixPath(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise PreparePilotError(f"linked artifact path escapes its immutable owner: {relative!r}")
        linked = base.joinpath(*candidate.parts)
        if not linked.is_file() or _sha256_file(linked) != sha:
            raise PreparePilotError(f"linked immutable artifact is missing or changed: {linked}")
    for item in value.values():
        _validate_linked_files(item, owner=owner)


def _validate_known_links(value: Mapping[str, Any], *, owner: Path) -> None:
    payload = value.get("payload")
    if not isinstance(payload, dict):
        return
    schema = payload.get("schema_version")
    if schema == "openpi.v35.calibration-preflight.v1":
        initialization = payload.get("initialization")
        if not isinstance(initialization, dict):
            raise PreparePilotError("calibration preflight is missing initialization bindings")
        controls = initialization.get("controls_file")
        controls_sha256 = initialization.get("controls_sha256")
        if not isinstance(controls, str) or not isinstance(controls_sha256, str):
            raise PreparePilotError("calibration preflight controls descriptor is invalid")
        controls_path = owner.parent / controls
        if not controls_path.is_file() or _sha256_file(controls_path) != controls_sha256:
            raise PreparePilotError("calibration preflight controls file is missing or changed")
    if schema == "openpi.v35.calibration-collection-receipt.v1":
        output = payload.get("output")
        shards = payload.get("episode_shards")
        if not isinstance(output, dict) or not isinstance(shards, list):
            raise PreparePilotError("calibration collection receipt is missing output/shard bindings")
        output_path = owner.parent / str(output.get("file", ""))
        if not output_path.is_file() or _sha256_file(output_path) != output.get("sha256"):
            raise PreparePilotError("sealed calibration replay differs from its receipt")
        shard_root = owner.parent / "replay_shards"
        for record in shards:
            if not isinstance(record, dict):
                raise PreparePilotError("calibration collection receipt has a malformed shard record")
            shard_path = shard_root / str(record.get("file", ""))
            if not shard_path.is_file() or _sha256_file(shard_path) != record.get("sha256"):
                raise PreparePilotError(f"calibration replay shard differs from its receipt: {shard_path}")


def _validate_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise PreparePilotError(f"immutable output is missing or empty: {path}")
    if path.suffix == ".json":
        value = _load_immutable_json(path)
        _validate_linked_files(value, owner=path)
        _validate_known_links(value, owner=path)


def _validate_group(group: ImmutableGroup) -> None:
    if not all(group.presence()):
        raise PreparePilotError(f"immutable output group {group.name!r} is incomplete")
    for path in group.paths:
        if path.is_dir():
            if not any(path.iterdir()):
                raise PreparePilotError(f"immutable output directory is empty: {path}")
        else:
            _validate_file(path)
    if group.primary_json is not None:
        _validate_file(group.primary_json)


def _validate_gate_b_pass(layout: Layout) -> None:
    value = _load_immutable_json(layout.absolute(layout.gate_b))
    payload = value.get("payload")
    gates = payload.get("gates") if isinstance(payload, dict) else None
    if (
        value.get("schema_version") != GATE_B_SCHEMA_VERSION
        or not isinstance(gates, dict)
        or payload.get("status") != "pass"
        or gates.get("passes") is not True
        or payload.get("decision") != "natural_branch_may_continue"
    ):
        raise PreparePilotError(
            "Gate B did not pass; stop this natural branch before any development-set rung collection"
        )


def _group_state(group: ImmutableGroup) -> str:
    presence = group.presence()
    if all(presence):
        return "complete"
    if any(presence):
        return "partial"
    return "missing"


def _script(python: str, name: str, *args: str, env: Mapping[str, str] | None = None) -> Command:
    return Command((python, f"scripts/{name}", *args), env={} if env is None else dict(env))


def _rung0_files(layout: Layout) -> tuple[Path, ...]:
    names = (
        "frame_selection.json",
        "gate_c_pytest_evidence.json",
        "gate_c_core.json",
        "gate_c_gradient.json",
        "evidence_vbar_frames.npz",
        "side_prototypes.npz",
        "side_prototypes.json",
        "rung_conditions.npz",
        "gate_c_measurements.npz",
        "mechanism_evidence.json",
        "task_health_rows.npz",
        "v35_runtime_identity.json",
        "v35_cumulative_telemetry.json",
        "v35_data_iterator_state.json",
        "task_health_evidence.json",
        "raw_rung_evaluation.json",
        "pilot_rung_result.json",
    )
    root = layout.absolute(layout.rung0_dir)
    return (root, *(root / name for name in names))


def build_stages(
    *,
    layout: Layout,
    manifest_sha256: str,
    python: str,
    gpus: Sequence[str],
    fsdp_devices: int,
    query_batch_size: int,
    leakage_batch_size: int,
) -> tuple[Stage, ...]:
    """Return the exact static command graph up to the dynamic config-hash stage."""

    if _SHA256_RE.fullmatch(manifest_sha256) is None:
        raise PreparePilotError("manifest SHA256 must be lower-case 64-character hex")
    if not gpus or len(set(gpus)) != len(gpus):
        raise PreparePilotError("--gpus must name one or more unique devices")
    if fsdp_devices != len(gpus):
        raise PreparePilotError("--fsdp-devices must equal the number of visible --gpus for this sealed pipeline")
    if query_batch_size != 8:
        raise PreparePilotError("the sealed v3.5 replay protocol requires --query-batch-size=8")
    if leakage_batch_size != 8:
        raise PreparePilotError("the sealed v3.5 Gate-B protocol requires --leakage-batch-size=8")
    manifest = layout.manifest_relative.as_posix()
    dataset = layout.dataset_relative.as_posix()
    visible = ",".join(gpus)
    shared_env = {"CUDA_VISIBLE_DEVICES": visible}
    single_gpu_env = {"CUDA_VISIBLE_DEVICES": gpus[0]}
    fsdp = str(fsdp_devices)
    checkpoint = layout.absolute(layout.checkpoint_relative)
    preflight_dir = layout.absolute(layout.preflight_dir)
    feature = layout.absolute(layout.leakage_features)
    feature_companions = (
        feature,
        feature.with_suffix(".npz"),
        feature.with_name(f"{feature.stem}_preprocessing.json"),
        feature.with_name(f"{feature.stem}_initialization_manifest.json"),
        feature.with_name(f"{feature.stem}_initialization_graft_manifest.json"),
    )
    worker_commands = tuple(
        _script(
            python,
            "v35_collect_calibration_replay.py",
            "collect",
            "--preflight",
            layout.preflight.as_posix(),
            "--selection",
            layout.calibration_selection.as_posix(),
            "--shards-dir",
            layout.replay_shards.as_posix(),
            "--shard-index",
            str(index),
            "--num-shards",
            str(len(gpus)),
            "--query-batch-size",
            str(query_batch_size),
            "--fsdp-devices",
            "1",
            env={"CUDA_VISIBLE_DEVICES": gpu},
        )
        for index, gpu in enumerate(gpus)
    )
    return (
        Stage(
            "bootstrap initialize",
            (
                _script(
                    python,
                    "v35_step0_bootstrap.py",
                    "initialize",
                    "--experiment-name",
                    layout.experiment,
                    "--fsdp-devices",
                    fsdp,
                    env=shared_env,
                ),
            ),
            ImmutableGroup(
                "bootstrap initialize",
                (
                    checkpoint,
                    layout.absolute(layout.provisional),
                    checkpoint / "initialization_graft_manifest.json",
                    checkpoint / "bootstrap_raw_state/0",
                ),
                layout.absolute(layout.provisional),
            ),
        ),
        Stage(
            "calibration preflight",
            (
                _script(
                    python,
                    "v35_calibration_replay.py",
                    "preflight",
                    "--config-name",
                    CONFIG_NAME,
                    "--manifest",
                    manifest,
                    "--manifest-sha256",
                    manifest_sha256,
                    "--assets-dir",
                    layout.assets_relative.as_posix(),
                    "--output-dir",
                    layout.preflight_dir.as_posix(),
                    "--fsdp-devices",
                    fsdp,
                    env=shared_env,
                ),
            ),
            ImmutableGroup(
                "calibration preflight",
                (
                    preflight_dir,
                    layout.absolute(layout.preflight),
                    preflight_dir / "step0_controls.npz",
                    preflight_dir / "initialization_graft_manifest.json",
                ),
                layout.absolute(layout.preflight),
            ),
        ),
        Stage(
            "train-54 replay selection",
            (
                _script(
                    python,
                    "v35_collect_calibration_replay.py",
                    "select",
                    "--preflight",
                    layout.preflight.as_posix(),
                    "--output",
                    layout.calibration_selection.as_posix(),
                    env=shared_env,
                ),
            ),
            ImmutableGroup(
                "train-54 replay selection",
                (layout.absolute(layout.calibration_selection),),
                layout.absolute(layout.calibration_selection),
            ),
        ),
        Stage("parallel replay collection", worker_commands, None, parallel=True),
        Stage(
            "replay seal and receipt",
            (
                _script(
                    python,
                    "v35_collect_calibration_replay.py",
                    "seal",
                    "--preflight",
                    layout.preflight.as_posix(),
                    "--selection",
                    layout.calibration_selection.as_posix(),
                    "--shards-dir",
                    layout.replay_shards.as_posix(),
                    "--output",
                    layout.replay_npz.as_posix(),
                    "--receipt",
                    layout.replay_receipt.as_posix(),
                    env=shared_env,
                ),
            ),
            ImmutableGroup(
                "replay seal and receipt",
                (layout.absolute(layout.replay_npz), layout.absolute(layout.replay_receipt)),
                layout.absolute(layout.replay_receipt),
            ),
        ),
        Stage(
            "injection calibration",
            (
                _script(
                    python,
                    "v35_injection_calibration.py",
                    "--input",
                    layout.replay_npz.as_posix(),
                    "--output",
                    layout.calibration.as_posix(),
                    env=shared_env,
                ),
            ),
            ImmutableGroup(
                "injection calibration",
                (layout.absolute(layout.calibration),),
                layout.absolute(layout.calibration),
            ),
        ),
        Stage(
            "bootstrap finalize",
            (
                _script(
                    python,
                    "v35_step0_bootstrap.py",
                    "finalize",
                    "--experiment-name",
                    layout.experiment,
                    "--calibration",
                    layout.calibration.as_posix(),
                    "--fsdp-devices",
                    fsdp,
                    env=shared_env,
                ),
            ),
            ImmutableGroup(
                "bootstrap finalize",
                (
                    layout.absolute(layout.initialization),
                    layout.absolute(layout.checkpoint0),
                    layout.absolute(layout.checkpoint0) / "assets/v35_runtime_identity.json",
                    layout.absolute(layout.checkpoint0) / "assets/v35_cumulative_telemetry.json",
                    layout.absolute(layout.checkpoint0) / "assets/v35_data_iterator_state.json",
                ),
                layout.absolute(layout.initialization),
            ),
        ),
        Stage(
            "Gate A",
            (
                _script(
                    python,
                    "v35_data_gate.py",
                    "--config-name",
                    CONFIG_NAME,
                    "--manifest",
                    manifest,
                    "--manifest-sha256",
                    manifest_sha256,
                    "--dataset-root",
                    dataset,
                    "--norm-provenance",
                    layout.norm_provenance.as_posix(),
                    "--norm-stats",
                    layout.norm_stats.as_posix(),
                    "--calibration-descriptor",
                    layout.calibration.as_posix(),
                    "--detail-report",
                    layout.gate_a_report.as_posix(),
                    "--output",
                    layout.gate_a.as_posix(),
                    env=shared_env,
                ),
            ),
            ImmutableGroup(
                "Gate A",
                (layout.absolute(layout.gate_a_report), layout.absolute(layout.gate_a)),
                layout.absolute(layout.gate_a),
            ),
        ),
        Stage(
            "Gate-B feature production",
            (
                _script(
                    python,
                    "v35_leakage_features.py",
                    "--step0-params",
                    (layout.checkpoint0 / "params").as_posix(),
                    "--initialization-manifest",
                    layout.initialization.as_posix(),
                    "--output",
                    layout.leakage_features.as_posix(),
                    "--batch-size",
                    str(leakage_batch_size),
                    env=single_gpu_env,
                ),
            ),
            ImmutableGroup("Gate-B feature production", feature_companions, feature),
        ),
        Stage(
            "Gate-B reducer",
            (
                _script(
                    python,
                    "v35_leakage_gate.py",
                    "--manifest",
                    manifest,
                    "--manifest-sha256",
                    manifest_sha256,
                    "--features",
                    layout.leakage_features.as_posix(),
                    "--output",
                    layout.gate_b.as_posix(),
                    env=shared_env,
                ),
            ),
            ImmutableGroup("Gate-B reducer", (layout.absolute(layout.gate_b),), layout.absolute(layout.gate_b)),
        ),
        Stage(
            "rung selection",
            (
                _script(
                    python,
                    "v35_rung_collect.py",
                    "select",
                    "--manifest",
                    manifest,
                    "--manifest-sha256",
                    manifest_sha256,
                    "--dataset-root",
                    dataset,
                    "--output",
                    layout.rung_selection.as_posix(),
                    env=shared_env,
                ),
            ),
            ImmutableGroup(
                "rung selection",
                (layout.absolute(layout.rung_selection),),
                layout.absolute(layout.rung_selection),
            ),
        ),
        Stage(
            "rung 0 collect and seal",
            (
                _script(
                    python,
                    "v35_rung_collect.py",
                    "collect",
                    "--checkpoint-step-dir",
                    layout.checkpoint0.as_posix(),
                    "--selection",
                    layout.rung_selection.as_posix(),
                    "--manifest",
                    manifest,
                    "--manifest-sha256",
                    manifest_sha256,
                    "--dataset-root",
                    dataset,
                    "--output-dir",
                    layout.rung0_dir.as_posix(),
                    env=single_gpu_env,
                ),
            ),
            ImmutableGroup("rung 0 collect and seal", _rung0_files(layout), layout.absolute(layout.rung0)),
        ),
    )


def semantic_hash_command(*, layout: Layout, python: str, gpus: Sequence[str], fsdp_devices: int) -> Command:
    return dataclasses.replace(
        _script(
            python,
            "v35_train.py",
            "--experiment-name",
            layout.experiment,
            "--calibration",
            layout.calibration.as_posix(),
            "--fsdp-devices",
            str(fsdp_devices),
            "--print-semantic-config-sha256",
            env={"CUDA_VISIBLE_DEVICES": ",".join(gpus)},
        ),
        capture_stdout=True,
    )


def authorization_command(
    *,
    layout: Layout,
    manifest_sha256: str,
    semantic_sha256: str,
    initialization_seed: int,
    python: str,
    gpus: Sequence[str],
) -> Command:
    return _script(
        python,
        "v35_training_authorization.py",
        "pilot",
        "--manifest",
        layout.manifest_relative.as_posix(),
        "--manifest-sha256",
        manifest_sha256,
        "--gate-a",
        layout.gate_a.as_posix(),
        "--gate-b",
        layout.gate_b.as_posix(),
        "--step0-rung",
        layout.rung0.as_posix(),
        "--prepilot-source-identity",
        layout.prepilot_source_identity.as_posix(),
        "--norm-stats",
        layout.norm_stats.as_posix(),
        "--norm-provenance",
        layout.norm_provenance.as_posix(),
        "--config-name",
        CONFIG_NAME,
        "--experiment-name",
        layout.experiment,
        "--initialization-seed",
        str(initialization_seed),
        "--semantic-training-config-sha256",
        semantic_sha256,
        "--output",
        layout.pilot_authorization.as_posix(),
        env={"CUDA_VISIBLE_DEVICES": ",".join(gpus)},
    )


def verify_command(*, layout: Layout, python: str, gpus: Sequence[str], fsdp_devices: int) -> Command:
    return _script(
        python,
        "v35_train.py",
        "--experiment-name",
        layout.experiment,
        "--calibration",
        layout.calibration.as_posix(),
        "--pilot-authorization",
        layout.pilot_authorization.as_posix(),
        "--target",
        "1000",
        "--fsdp-devices",
        str(fsdp_devices),
        "--verify-only",
        env={"CUDA_VISIBLE_DEVICES": ",".join(gpus)},
    )


def _run_immutable_stage(stage: Stage, *, resume: bool, runner: CommandRunner) -> None:
    if stage.outputs is None:
        raise AssertionError("immutable stage has no output group")
    state = _group_state(stage.outputs)
    if state == "complete":
        if not resume:
            raise PreparePilotError(f"{stage.name} already exists; use --resume to authenticate and skip it")
        _validate_group(stage.outputs)
        print(f"[skip validated] {stage.name}")
        return
    if state == "partial":
        raise PreparePilotError(
            f"{stage.name} has a partial create-only output group; refusing overwrite. "
            "Preserve it for diagnosis and restart with a new experiment name."
        )
    print(f"[run] {stage.name}")
    for command in stage.commands:
        runner.run(command)
    _validate_group(stage.outputs)


def _replay_paths(layout: Layout) -> tuple[Path, ...]:
    root = layout.absolute(layout.replay_shards)
    return tuple(root / f"episode_{ordinal:03d}.npz" for ordinal in range(EXPECTED_REPLAY_SHARDS))


def _run_replay_stage(
    stage: Stage,
    *,
    layout: Layout,
    resume: bool,
    sealed_group: ImmutableGroup,
    runner: CommandRunner,
) -> None:
    sealed_state = _group_state(sealed_group)
    if sealed_state == "complete":
        if not resume:
            raise PreparePilotError("sealed replay already exists; use --resume")
        _validate_group(sealed_group)
        print("[skip validated] parallel replay collection (sealed replay exists)")
        return
    if sealed_state == "partial":
        raise PreparePilotError("replay seal/receipt is partial; refusing to mutate its source shards")
    root = layout.absolute(layout.replay_shards)
    existing = tuple(path for path in _replay_paths(layout) if path.exists())
    unexpected = tuple(root.glob("episode_*.npz")) if root.is_dir() else ()
    expected_names = {path.name for path in _replay_paths(layout)}
    if any(path.name not in expected_names for path in unexpected):
        raise PreparePilotError("replay shard directory contains an unexpected episode_*.npz file")
    if (root.exists() or existing) and not resume:
        raise PreparePilotError("replay shard output already exists; use --resume to validate/fill it")
    for path in existing:
        _validate_file(path)
    print(f"[run parallel] {stage.name}: existing={len(existing)}, GPUs={len(stage.commands)}")
    runner.run_parallel(stage.commands)
    paths = _replay_paths(layout)
    if not all(path.is_file() for path in paths):
        missing = [path.name for path in paths if not path.is_file()]
        raise PreparePilotError(f"parallel replay collection left missing shards: {missing}")
    for path in paths:
        _validate_file(path)


def _parse_semantic_hash(stdout: str) -> str:
    matches = [line.strip() for line in stdout.splitlines() if _SHA256_RE.fullmatch(line.strip())]
    if len(matches) != 1:
        raise PreparePilotError("semantic-config command did not emit exactly one SHA256 line")
    return matches[0]


def _initialization_seed(layout: Layout) -> int:
    identity = _load_immutable_json(layout.absolute(layout.initialization))
    seed = identity.get("initialization_seed")
    if type(seed) is not int:
        raise PreparePilotError("finalized initialization identity has no integer initialization_seed")
    return seed


def _validate_bootstrap_preflight_binding(layout: Layout) -> None:
    """Reject divergent deterministic initializations before any 74-episode replay."""

    provisional = _load_immutable_json(layout.absolute(layout.provisional))
    preflight = _load_immutable_json(layout.absolute(layout.preflight))
    payload = preflight.get("payload")
    initialization = payload.get("initialization") if isinstance(payload, dict) else None
    config = payload.get("config") if isinstance(payload, dict) else None
    if not isinstance(initialization, dict) or not isinstance(config, dict):
        raise PreparePilotError("calibration preflight is missing initialization/config identity")
    comparisons = {
        "actual step-0 parameter tree": (
            provisional.get("actual_step0_parameter_tree_sha256"),
            initialization.get("actual_step0_parameter_tree_sha256"),
        ),
        "official base source tree": (
            provisional.get("official_source_tree_sha256"),
            initialization.get("official_base_source_tree_sha256"),
        ),
        "target schema": (
            provisional.get("target_schema_sha256"),
            initialization.get("target_schema_sha256"),
        ),
        "graft manifest": (
            provisional.get("graft_manifest", {}).get("manifest_sha256")
            if isinstance(provisional.get("graft_manifest"), dict)
            else None,
            initialization.get("graft_manifest_sha256"),
        ),
        "initialization seed": (provisional.get("initialization_seed"), config.get("seed")),
        "official base URI": (provisional.get("official_source_uri"), config.get("official_base_uri")),
    }
    mismatches = {name: values for name, values in comparisons.items() if values[0] != values[1]}
    if mismatches:
        raise PreparePilotError(f"bootstrap/preflight initialization identity mismatch: {mismatches}")

    provisional_data = provisional.get("dataset_identity")
    preflight_data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(provisional_data, dict) or not isinstance(preflight_data, dict):
        raise PreparePilotError("bootstrap/preflight is missing dataset identity")
    # The portable norm provenance is a frozen historical artifact whose representation
    # predates the canonical single-line envelope convention; _validate_inputs already
    # authenticated its exact bytes against FROZEN_NORM_PROVENANCE_SHA256.
    norm_provenance = _load_frozen_json(layout.absolute(layout.norm_provenance))
    selection = norm_provenance.get("selection")
    if not isinstance(selection, dict):
        raise PreparePilotError("frozen norm provenance is missing selection identity")
    data_comparisons = {
        "manifest relative path": (
            provisional_data.get("episode_manifest_path_relative"),
            preflight_data.get("manifest_path_relative"),
        ),
        "manifest SHA256": (
            provisional_data.get("episode_manifest_sha256"),
            preflight_data.get("manifest_sha256"),
        ),
        "norm stats relative path": (
            provisional_data.get("norm_stats_path_relative"),
            preflight_data.get("norm_stats_path_relative"),
        ),
        "norm stats SHA256": (
            provisional_data.get("norm_stats_sha256"),
            preflight_data.get("norm_stats_sha256"),
        ),
        "norm provenance relative path": (
            provisional_data.get("norm_stats_provenance_path_relative"),
            preflight_data.get("norm_provenance_path_relative"),
        ),
        "norm provenance SHA256": (
            provisional_data.get("norm_stats_provenance_sha256"),
            preflight_data.get("norm_provenance_sha256"),
        ),
        "train storage SHA256": (
            provisional_data.get("train_storage_sha256"),
            preflight_data.get("train_storage_sha256"),
        ),
        "dataset episode/frame protocol SHA256": (
            selection.get("dataset_episode_frame_protocol_sha256"),
            preflight_data.get("dataset_episode_frame_protocol_sha256"),
        ),
    }
    data_mismatches = {name: values for name, values in data_comparisons.items() if values[0] != values[1]}
    if data_mismatches:
        raise PreparePilotError(f"bootstrap/preflight dataset identity mismatch: {data_mismatches}")
    if (
        preflight_data.get("production") is not True
        or preflight_data.get("path_contract") != "memory_project-relative-v1"
        or preflight_data.get("dataset_repo_id") != project_paths.V35_REPO_ID
        or preflight_data.get("split") != "train"
        or preflight_data.get("split_seed") != 36
        or preflight_data.get("train_episode_count") != 54
    ):
        raise PreparePilotError("calibration preflight does not use the frozen production train-54 data contract")


def _validate_dataset_inventory(
    layout: Layout,
    *,
    inventory_path: Path,
    manifest_path: Path,
    verify_bytes: bool,
) -> None:
    """Authenticate every train/development payload and every runtime metadata file.

    Final-test parquet bytes remain sealed. Their expected records stay bound by the frozen
    inventory hash, but this pre-pilot path neither opens nor hashes those payloads.
    """

    inventory = _load_immutable_json(inventory_path)
    expected_inventory_keys = {
        "dataset_repo_id",
        "directories",
        "directory_count",
        "file_count",
        "files",
        "schema_version",
        "total_bytes",
        "tree_sha256",
    }
    if set(inventory) != expected_inventory_keys:
        raise PreparePilotError("dataset inventory has unexpected top-level keys")
    if (
        inventory.get("schema_version") != "openpi.v35.dataset-tree-inventory.v1"
        or inventory.get("dataset_repo_id") != project_paths.V35_REPO_ID
        or inventory.get("tree_sha256") != FROZEN_DATASET_TREE_SHA256
        or inventory.get("file_count") != 78
        or inventory.get("total_bytes") != 41_632_438_815
    ):
        raise PreparePilotError("dataset inventory does not describe the frozen v3.5 dataset")
    records = inventory.get("files")
    if not isinstance(records, list) or len(records) != 78:
        raise PreparePilotError("dataset inventory must contain exactly 78 file records")

    # The exact manifest bytes were already checked against FROZEN_MANIFEST_SHA256 by
    # _validate_inputs.  Preserve its approved pretty-printed representation.
    manifest = _load_frozen_json(manifest_path)
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise PreparePilotError("frozen manifest has no episode records")
    selected_indices: set[int] = set()
    final_indices: set[int] = set()
    for episode in episodes:
        if not isinstance(episode, dict) or episode.get("include") is not True:
            continue
        index = episode.get("episode_index")
        split = episode.get("split")
        if type(index) is not int or index < 0:
            raise PreparePilotError("frozen manifest has an invalid included episode index")
        if split in ("train", "development"):
            selected_indices.add(index)
        elif split == "final_test":
            final_indices.add(index)
        else:
            raise PreparePilotError("frozen manifest has an invalid included split")
    if len(selected_indices) != 62 or len(final_indices) != 8 or selected_indices & final_indices:
        raise PreparePilotError("frozen manifest no longer has the sealed 54/8/8 payload split")

    expected_selected = {f"data/chunk-000/episode_{index:06d}.parquet" for index in selected_indices}
    expected_final = {f"data/chunk-000/episode_{index:06d}.parquet" for index in final_indices}
    seen: set[str] = set()
    verified_selected: set[str] = set()
    dataset_root = layout.absolute(layout.dataset_relative).resolve()
    for record in records:
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "size"}:
            raise PreparePilotError("dataset inventory contains a malformed file record")
        relative = record["path"]
        digest = record["sha256"]
        size = record["size"]
        if (
            not isinstance(relative, str)
            or PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or relative in seen
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or type(size) is not int
            or size < 0
        ):
            raise PreparePilotError("dataset inventory contains an invalid or duplicate file record")
        seen.add(relative)
        is_parquet = relative.startswith("data/") and relative.endswith(".parquet")
        if is_parquet and relative not in expected_selected and relative not in expected_final:
            raise PreparePilotError(f"dataset inventory contains an unknown episode payload: {relative}")
        must_hash = relative in expected_selected or relative.startswith("meta/")
        must_stat = must_hash or relative in expected_final
        if not must_stat:
            continue
        candidate = (dataset_root / PurePosixPath(relative)).resolve()
        try:
            candidate.relative_to(dataset_root)
        except ValueError as exc:
            raise PreparePilotError(f"dataset inventory path resolves outside the dataset: {relative}") from exc
        try:
            actual_size = candidate.stat().st_size
        except OSError as exc:
            raise PreparePilotError(f"required frozen dataset file is missing: {candidate}") from exc
        if not candidate.is_file() or actual_size != size:
            raise PreparePilotError(f"frozen dataset file size mismatch: {candidate}")
        if verify_bytes and must_hash and _sha256_file(candidate) != digest:
            raise PreparePilotError(f"frozen dataset file SHA256 mismatch: {candidate}")
        if relative in expected_selected:
            verified_selected.add(relative)
    if verified_selected != expected_selected:
        missing = sorted(expected_selected - verified_selected)
        raise PreparePilotError(f"dataset inventory is missing train/development payloads: {missing}")


def _validate_inputs(
    layout: Layout,
    *,
    expected_manifest_sha256: str | None,
    verify_dataset_bytes: bool = True,
) -> str:
    manifest = layout.absolute(layout.manifest_relative)
    inventory = layout.absolute(DEFAULT_DATASET_INVENTORY)
    for name, path in (
        ("manifest", manifest),
        ("dataset inventory", inventory),
        ("dataset root", layout.absolute(layout.dataset_relative)),
        ("norm stats", layout.absolute(layout.norm_stats)),
        ("norm provenance", layout.absolute(layout.norm_provenance)),
    ):
        if not path.exists():
            raise PreparePilotError(f"required {name} is missing: {path}")
    actual = _sha256_file(manifest)
    if actual != FROZEN_MANIFEST_SHA256:
        raise PreparePilotError(
            f"frozen production manifest SHA256 mismatch: expected {FROZEN_MANIFEST_SHA256}, found {actual}"
        )
    if expected_manifest_sha256 is not None and expected_manifest_sha256 != FROZEN_MANIFEST_SHA256:
        raise PreparePilotError(f"manifest SHA256 mismatch: expected {expected_manifest_sha256}, found {actual}")
    inventory_sha256 = _sha256_file(inventory)
    if inventory_sha256 != FROZEN_DATASET_INVENTORY_SHA256:
        raise PreparePilotError(
            "frozen dataset inventory SHA256 mismatch: "
            f"expected {FROZEN_DATASET_INVENTORY_SHA256}, found {inventory_sha256}"
        )
    _validate_dataset_inventory(
        layout,
        inventory_path=inventory,
        manifest_path=manifest,
        verify_bytes=verify_dataset_bytes,
    )
    expected_norm_hashes = {
        layout.absolute(layout.norm_stats): FROZEN_NORM_STATS_SHA256,
        layout.absolute(layout.norm_provenance): FROZEN_NORM_PROVENANCE_SHA256,
    }
    for path, expected in expected_norm_hashes.items():
        found = _sha256_file(path)
        if found != expected:
            raise PreparePilotError(
                f"frozen norm artifact SHA256 mismatch for {path}: expected {expected}, found {found}"
            )
    return actual


def _ensure_resume_lineage(stages: Sequence[Stage], *, resume: bool) -> None:
    groups = [stage.outputs for stage in stages if stage.outputs is not None]
    present_later = False
    for group in reversed(groups):
        assert group is not None
        state = _group_state(group)
        if state != "missing":
            present_later = True
        elif present_later and resume:
            raise PreparePilotError(
                f"resume lineage is missing upstream immutable group {group.name!r} while later outputs exist"
            )


def _print_plan(
    *,
    stages: Sequence[Stage],
    semantic: Command,
    authorization: Command,
    verify: Command,
) -> None:
    for index, stage in enumerate(stages, start=1):
        status = "dynamic" if stage.outputs is None else _group_state(stage.outputs)
        print(f"{index:02d}. {stage.name} [{status}]")
        for command in stage.commands:
            print(f"    {command.display()}")
    print(f"{len(stages) + 1:02d}. semantic training-config SHA256 [read-only]")
    print(f"    {semantic.display()}")
    print(f"{len(stages) + 2:02d}. pilot authorization [dynamic SHA256]")
    print(f"    {authorization.display()}")
    print(f"{len(stages) + 3:02d}. final pre-pilot verification [read-only, no training]")
    print(f"    {verify.display()}")


def prepare(
    *,
    layout: Layout,
    manifest_sha256: str,
    gpus: Sequence[str],
    fsdp_devices: int,
    query_batch_size: int,
    leakage_batch_size: int,
    resume: bool,
    plan_only: bool,
    runner: CommandRunner,
    python: str,
) -> None:
    stages = build_stages(
        layout=layout,
        manifest_sha256=manifest_sha256,
        python=python,
        gpus=gpus,
        fsdp_devices=fsdp_devices,
        query_batch_size=query_batch_size,
        leakage_batch_size=leakage_batch_size,
    )
    semantic = semantic_hash_command(layout=layout, python=python, gpus=gpus, fsdp_devices=fsdp_devices)
    placeholder_auth = authorization_command(
        layout=layout,
        manifest_sha256=manifest_sha256,
        semantic_sha256="<computed-semantic-config-sha256>",
        initialization_seed=42,
        python=python,
        gpus=gpus,
    )
    verify = verify_command(layout=layout, python=python, gpus=gpus, fsdp_devices=fsdp_devices)
    if plan_only:
        _print_plan(stages=stages, semantic=semantic, authorization=placeholder_auth, verify=verify)
        return
    existing_lineage = (
        layout.absolute(layout.run_relative).exists() or layout.absolute(layout.checkpoint_relative).exists()
    )
    if not resume:
        for name, relative in (
            ("diagnostic run", layout.run_relative),
            ("checkpoint run", layout.checkpoint_relative),
        ):
            if layout.absolute(relative).exists():
                raise PreparePilotError(f"{name} already exists; choose a new experiment or use --resume")
    _freeze_or_validate_source_identity(layout, existing_lineage=existing_lineage)
    authorization_group = ImmutableGroup(
        "pilot authorization",
        (layout.absolute(layout.pilot_authorization),),
        layout.absolute(layout.pilot_authorization),
    )
    if resume and _group_state(authorization_group) != "missing":
        incomplete = [
            stage.outputs.name
            for stage in stages
            if stage.outputs is not None and _group_state(stage.outputs) != "complete"
        ]
        if incomplete:
            raise PreparePilotError(
                f"pilot authorization exists but its upstream immutable lineage is incomplete: {incomplete}"
            )
    existing_shards = tuple(path for path in _replay_paths(layout) if path.exists())
    if existing_shards:
        required = (stages[0].outputs, stages[1].outputs, stages[2].outputs)
        if any(group is None or _group_state(group) != "complete" for group in required):
            raise PreparePilotError("replay shards exist without complete bootstrap/preflight/selection lineage")
    _ensure_resume_lineage(stages, resume=resume)
    sealed_group = next(stage.outputs for stage in stages if stage.name == "replay seal and receipt")
    assert sealed_group is not None
    for stage in stages:
        _validate_frozen_source_identity(layout)
        if stage.parallel:
            _run_replay_stage(
                stage,
                layout=layout,
                resume=resume,
                sealed_group=sealed_group,
                runner=runner,
            )
        else:
            _run_immutable_stage(stage, resume=resume, runner=runner)
            if stage.name == "calibration preflight":
                _validate_bootstrap_preflight_binding(layout)
            elif stage.name == "Gate-B reducer":
                _validate_gate_b_pass(layout)

    _validate_frozen_source_identity(layout)
    auth_state = _group_state(authorization_group)
    if auth_state == "complete":
        if not resume:
            raise PreparePilotError("pilot authorization already exists; use --resume")
        _validate_group(authorization_group)
        print("[skip validated] pilot authorization")
    else:
        if auth_state == "partial":
            raise PreparePilotError("pilot authorization output is partial")
        semantic_sha256 = _parse_semantic_hash(runner.run(semantic))
        _validate_frozen_source_identity(layout)
        command = authorization_command(
            layout=layout,
            manifest_sha256=manifest_sha256,
            semantic_sha256=semantic_sha256,
            initialization_seed=_initialization_seed(layout),
            python=python,
            gpus=gpus,
        )
        print("[run] semantic config identity and pilot authorization")
        runner.run(command)
        _validate_group(authorization_group)

    _validate_frozen_source_identity(layout)
    print("[verify-only] complete pilot launch contract; optimizer is not invoked")
    runner.run(verify)


def _gpu_list(value: str | None) -> tuple[str, ...]:
    raw = value if value is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if not values or len(values) != len(set(values)) or any(any(char.isspace() for char in item) for item in values):
        raise PreparePilotError("--gpus must be a comma-separated list of unique CUDA device identifiers")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST.as_posix())
    parser.add_argument("--manifest-sha256", help="Optional expected manifest hash; actual bytes are always hashed.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT.as_posix())
    parser.add_argument("--assets-dir", default=DEFAULT_ASSETS_DIR.as_posix())
    parser.add_argument(
        "--gpus",
        help="Comma-separated devices; defaults to CUDA_VISIBLE_DEVICES or 0. Replay uses one process per entry.",
    )
    parser.add_argument(
        "--fsdp-devices",
        type=int,
        help="Bootstrap/finalize device count; must equal the visible GPU count (default: GPU count).",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=8,
        help="Frozen replay batch size; production accepts only 8 so resumed shards cannot mix protocols.",
    )
    parser.add_argument(
        "--leakage-batch-size",
        type=int,
        default=8,
        help="Frozen Gate-B feature batch size; production accepts only 8.",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Validate/skip complete artifacts and fill missing shards."
    )
    parser.add_argument(
        "--plan", action="store_true", help="Print exact commands and current artifact states; do nothing."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        project_paths.configure_v35_runtime_environment()
        project_paths.validate_executing_openpi_checkout()
        layout = Layout.create(
            experiment=args.experiment_name,
            manifest=_normalized_relative(args.manifest, name="manifest"),
            dataset_root=_normalized_relative(args.dataset_root, name="dataset root"),
            assets_dir=_normalized_relative(args.assets_dir, name="assets directory"),
        )
        _validate_sealed_layout(layout)
        gpus = _gpu_list(args.gpus)
        fsdp_devices = len(gpus) if args.fsdp_devices is None else args.fsdp_devices
        manifest_sha256 = _validate_inputs(
            layout,
            expected_manifest_sha256=args.manifest_sha256,
            verify_dataset_bytes=not args.plan,
        )
        openpi_dir = project_paths.project_path(project_paths.OPENPI_DIR)
        base_env = dict(os.environ)
        base_env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        runner = CommandRunner(cwd=openpi_dir, base_env=base_env)
        prepare(
            layout=layout,
            manifest_sha256=manifest_sha256,
            gpus=gpus,
            fsdp_devices=fsdp_devices,
            query_batch_size=args.query_batch_size,
            leakage_batch_size=args.leakage_batch_size,
            resume=args.resume,
            plan_only=args.plan,
            runner=runner,
            python=sys.executable,
        )
    except (PreparePilotError, project_paths.ProjectRootError, OSError, ValueError) as exc:
        _parser().error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
