"""Read-only causal-memory audit for the v34 run5 eta=0 pilot.

This diagnostic is intentionally separate from numerical-safety monitoring.  It evaluates the
four training-held-out episodes under matched, functional interventions at every waiting frame:

* production-normal memory (pre-write at the current frame);
* the same normal state with retrieval content zeroed;
* a freshly reset memory and a never-written memory (an exact run5 invariant);
* memory frozen immediately before and immediately after the evidence phase;
* post-evidence dynamics-only and normal-write branches; and
* matched non-held-out, same-instruction same-side and opposite-side post-evidence swaps; and
* the reciprocal same-instruction heldout opposite-side swap as a separate sensitivity control.

The token statistic is D = log p(right) - log p(left), scored only through the single differing
side token in the canonical waiting labels.  Every reported headline margin is truth aligned:
``y * D`` with y=+1 for right and y=-1 for left.  ``--mode full`` additionally samples robot-space
action chunks with identical explicit diffusion noise in every intervention and reports
S = RMS(right-arm displacement) - RMS(left-arm displacement).

The checkpoint is never modified.  ``ema`` restores the standalone ``params`` item; ``raw``
restores only ``train_state/params/*/value`` through an explicit Orbax transform (never optimizer
or EMA state).  The output directory is exclusively reserved with an ``INCOMPLETE`` marker before
inference; the report and action-array artifact are atomically installed inside it, and only then
is a checksummed ``COMPLETE`` marker written. Existing destinations and destinations inside the
checkpoint are rejected.

Example (one H100; always pass the static run5 config explicitly):

    .venv/bin/python scripts/v34_causal_memory_eval.py \
      --checkpoint diagnostic_checkpoints/v34_run5_eta0_pilot_copies/500 \
      --dataset-root /iris/u/kewalk/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask \
      --output-dir diagnostic_outputs/v34_run5_causal/500/raw \
      --config pi05_yam_mem_v34_run5_eta0 --parameter-source raw --mode token

``token`` is the cheap checkpoint-500/1000 audit.  ``full`` is the final checkpoint audit and
uses three fixed action-noise replicates on a pre-registered subset whose 50-step demonstrated
chunk contains at least 15 execute-phase frames.
"""

# ruff: noqa: SLF001, I001 - pyarrow must precede the openpi/JAX stack for this dataset.
from __future__ import annotations

import pyarrow.parquet as pq

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Literal

# Evidence-grade evaluation must not silently fetch a different tokenizer/config revision.
# These are set before importing any OpenPI/Transformers module; a missing cache fails closed.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import flax.nnx as nnx
import flax.traverse_util as traverse_util
from huggingface_hub import snapshot_download
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import tensorstore as ts

from openpi import transforms as _transforms
import openpi.diagnostics.v33_write_token_probe as _probe
import openpi.diagnostics.v33_writer_attention as _wa
import openpi.diagnostics.writer_contribution as _wc
from openpi.diagnostics.v32_checkpoint import _align_params
from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.shared import nnx_utils
from openpi.shared import normalize as _normalize
from openpi.training import config as _config


SCHEMA_VERSION = "openpi.v34.causal_memory_eval.v1"
RUN5_CONFIG = "pi05_yam_mem_v34_run5_eta0"
EXPECTED_HELDOUT = (15, 29, 44, 59)
EXPECTED_CELLS = {
    15: ("find the banana", "left"),
    29: ("find the banana", "right"),
    44: ("find the grey pepper box", "left"),
    59: ("find the grey pepper box", "right"),
}
EXPECTED_DONORS = {
    "same_side": {15: 13, 29: 18, 44: 31, 59: 54},
    "opposite_side": {15: 18, 29: 13, 44: 46, 59: 42},
}
EXPECTED_HELDOUT_OPPOSITE_DONORS = {15: 29, 29: 15, 44: 59, 59: 44}
EXPECTED_WAIT_FRAMES = {
    15: (540,),
    29: (510, 525, 540),
    44: (525, 540, 555, 570),
    59: (510, 525, 540, 555),
}
EXPECTED_ACTION_FRAMES = {
    15: (540,),
    29: (525, 540),
    44: (555, 570),
    59: (525, 540, 555),
}
EXPECTED_DEMONSTRATED_ACTION_SCORES = {
    (15, 540): -0.3441188637378251,
    (29, 525): 0.09268175840937752,
    (29, 540): 0.2828443328407406,
    (44, 555): -0.1986549176675482,
    (44, 570): -0.4509864556453942,
    (59, 525): 0.07312115284674275,
    (59, 540): 0.25985896347967274,
    (59, 555): 0.4813865098501708,
}
CONDITIONS = (
    "normal",
    "normal_zero_read",
    "reset",
    "never_write",
    "pre_evidence_frozen",
    "frozen_after_evidence",
    "dynamics_only",
    "same_side_frozen_swap",
    "opposite_frozen_swap",
    "heldout_opposite_frozen_swap",
)
ACTION_NOISE_REPLICATES = (0, 1, 2)
RESET_TOL = 1e-5
TOKEN_EFFECT_EPS = 0.05
ACTION_EFFECT_EPS = 0.02
MIN_DIRECTIONAL_COVERAGE = 0.75
MIN_EXECUTE_FRAMES_IN_ACTION_CHUNK = 15
ACTION_SCORE_EPS = 0.05
ACTION_DENOISE_STEPS = 10
ACTION_MAX_DECODE_STEPS = 10
EXPECTED_EMA_DECAY = 0.999
EXPECTED_PALIGEMMA_TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
EXPECTED_FAST_TOKENIZER_COMMIT = "ec4d7aa71691cac0b8bed6942be45684db2110f4"
EXPECTED_FAST_TOKENIZER_FILE_SHA256 = {
    "processing_action_tokenizer.py": "6f021ca1f4c1b194ab6fa399d80baf3d642eadb17efb8f73301e4ac401522c20",
    "processor_config.json": "f40cfbb1020858fe1d48c0f946b0c1315a90d6e84aa82710036f24f4c167706a",
    "special_tokens_map.json": "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356",
    "tokenizer.json": "6507dd709287fd018882120c0071787f1f62bad9f180f1e8c5235bda1b71fa78",
    "tokenizer_config.json": "b4030e2a13a0dea22e99d54c086fb320c71e66ad034ac4eba4301a0a27d5e5cd",
}
LAUNCH_PROVENANCE = Path("cluster_v34/provenance/v34_run5_eta0_main_launch")
PILOT_LOG = Path("cluster_v34/logs/v34_run5_eta0-pilot-parent17024084.err")
PILOT_HARDLINK_ARCHIVE = Path(
    "diagnostic_checkpoints/v34_run5_eta0_pilot_hardlinks_read_only"
)
LAUNCH_BOUND_FILES = (
    "src/openpi/models/memory.py",
    "src/openpi/models/pi0.py",
    "src/openpi/training/config.py",
)
LAUNCH_SNAPSHOT_BOUND_FILES = (
    "scripts/train.py",
    "scripts/serve_yam_memory.py",
    "src/openpi/models/gemma.py",
    "src/openpi/models/gemma_fast.py",
    "src/openpi/models/lora.py",
    "src/openpi/models/memory.py",
    "src/openpi/models/model.py",
    "src/openpi/models/pi0.py",
    "src/openpi/models/pi0_config.py",
    "src/openpi/models/rtc.py",
    "src/openpi/models/siglip.py",
    "src/openpi/models/tokenizer.py",
    "src/openpi/models/utils/fsq_tokenizer.py",
    "src/openpi/models/vit.py",
    "src/openpi/policies/yam_policy.py",
    "src/openpi/shared/image_tools.py",
    "src/openpi/shared/download.py",
    "src/openpi/shared/nnx_utils.py",
    "src/openpi/shared/normalize.py",
    "src/openpi/training/checkpoints.py",
    "src/openpi/training/utils.py",
    "src/openpi/transforms.py",
)


@dataclasses.dataclass(frozen=True)
class Args:
    checkpoint: Path
    dataset_root: Path
    output_dir: Path
    config: str
    parameter_source: Literal["raw", "ema"]
    mode: Literal["token", "full"] = "token"
    episodes: tuple[int, ...] = EXPECTED_HELDOUT
    seed: int = 0
    action_noise_replicates: tuple[int, ...] = ACTION_NOISE_REPLICATES
    action_denoise_steps: int = ACTION_DENOISE_STEPS

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint", Path(self.checkpoint).expanduser().resolve())
        object.__setattr__(self, "dataset_root", Path(self.dataset_root).expanduser().resolve())
        object.__setattr__(self, "output_dir", Path(self.output_dir).expanduser().resolve())
        object.__setattr__(self, "episodes", tuple(self.episodes))
        object.__setattr__(self, "action_noise_replicates", tuple(self.action_noise_replicates))
        if self.config != RUN5_CONFIG:
            raise ValueError(f"causal audit is pinned to --config {RUN5_CONFIG!r}; got {self.config!r}")
        if self.parameter_source not in ("raw", "ema"):
            raise ValueError("--parameter-source must be raw or ema")
        if self.mode not in ("token", "full"):
            raise ValueError("--mode must be token or full")
        if self.episodes != EXPECTED_HELDOUT:
            raise ValueError(
                f"causal audit requires exact heldouts {EXPECTED_HELDOUT}; got {self.episodes}"
            )
        if self.seed != 0:
            raise ValueError("the pre-registered causal audit requires --seed 0")
        if self.action_noise_replicates != ACTION_NOISE_REPLICATES:
            raise ValueError(
                "the pre-registered causal audit requires exact action-noise replicates "
                f"{ACTION_NOISE_REPLICATES}; got {self.action_noise_replicates}"
            )
        if self.action_denoise_steps != ACTION_DENOISE_STEPS:
            raise ValueError(
                "production-parity causal audit requires "
                f"--action-denoise-steps {ACTION_DENOISE_STEPS}"
            )


def _parse_args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--parameter-source", choices=("raw", "ema"), required=True)
    parser.add_argument("--mode", choices=("token", "full"), default="token")
    parser.add_argument("--episodes", type=int, nargs="+", default=list(EXPECTED_HELDOUT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-noise-replicates", type=int, nargs="+", default=list(ACTION_NOISE_REPLICATES)
    )
    parser.add_argument("--action-denoise-steps", type=int, default=10)
    values = vars(parser.parse_args(argv))
    values["episodes"] = tuple(values["episodes"])
    values["action_noise_replicates"] = tuple(values["action_noise_replicates"])
    return Args(**values)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "resolved_path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _directory_identity(path: Path) -> dict[str, Any]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise FileNotFoundError(f"identity directory contains no files: {path}")
    return {
        "resolved_path": str(path.resolve()),
        "files": {
            str(item.relative_to(path)): {
                "bytes": item.stat().st_size,
                "sha256": _sha256_file(item),
            }
            for item in files
        },
    }


def _validate_tokenizer_asset_files(
    paligemma_path: Path, fast_snapshot_path: Path
) -> dict[str, Any]:
    paligemma_identity = _file_identity(paligemma_path)
    if paligemma_identity["sha256"] != EXPECTED_PALIGEMMA_TOKENIZER_SHA256:
        raise ValueError(
            "cached PaliGemma tokenizer hash mismatch: expected "
            f"{EXPECTED_PALIGEMMA_TOKENIZER_SHA256}, got {paligemma_identity['sha256']}"
        )
    fast_identity = _directory_identity(fast_snapshot_path)
    actual_fast_hashes = {
        name: identity["sha256"] for name, identity in fast_identity["files"].items()
    }
    if actual_fast_hashes != EXPECTED_FAST_TOKENIZER_FILE_SHA256:
        raise ValueError(
            "cached FAST tokenizer snapshot files differ from the pinned commit: "
            f"expected {EXPECTED_FAST_TOKENIZER_FILE_SHA256}, got {actual_fast_hashes}"
        )
    return {
        "paligemma_model": paligemma_identity,
        "fast_snapshot": fast_identity,
        "fast_expected_commit": EXPECTED_FAST_TOKENIZER_COMMIT,
        "all_expected_hashes_match": True,
    }


def _resolve_tokenizer_assets() -> dict[str, Any]:
    """Resolve only pinned local tokenizer assets; never invoke a network fallback."""
    openpi_cache = Path(os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser().resolve()
    paligemma_path = openpi_cache / "big_vision/paligemma_tokenizer.model"
    if not paligemma_path.is_file():
        raise FileNotFoundError(
            "pinned PaliGemma tokenizer is absent from the local OpenPI cache; "
            f"refusing a network download: {paligemma_path}"
        )
    fast_snapshot_path = Path(
        snapshot_download(
            "physical-intelligence/fast",
            revision="main",
            local_files_only=True,
        )
    ).resolve()
    if fast_snapshot_path.name != EXPECTED_FAST_TOKENIZER_COMMIT:
        raise ValueError(
            "cached physical-intelligence/fast main revision changed: expected "
            f"{EXPECTED_FAST_TOKENIZER_COMMIT}, got {fast_snapshot_path.name}"
        )
    return _validate_tokenizer_asset_files(paligemma_path, fast_snapshot_path)


def _verify_sha256_manifest(
    manifest_path: Path, repo: Path, allowed_directory: Path
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"SHA256 manifest is missing: {manifest_path}")
    recorded: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        try:
            valid_digest = len(fields[0]) == 64 and int(fields[0], 16) >= 0
        except (IndexError, ValueError):
            valid_digest = False
        if len(fields) != 2 or not valid_digest:
            raise ValueError(f"malformed SHA256 manifest line: {line!r}")
        relative = fields[1].strip().removeprefix("*")
        if relative in recorded:
            raise ValueError(f"duplicate SHA256 manifest entry: {relative}")
        candidate = (repo / relative).resolve()
        if not _is_relative_to(candidate, allowed_directory.resolve()):
            raise ValueError(f"SHA256 manifest entry escapes launch provenance: {relative}")
        recorded[relative] = fields[0].lower()
    if not recorded:
        raise ValueError(f"SHA256 manifest is empty: {manifest_path}")
    verified = {}
    for relative, expected in recorded.items():
        path = (repo / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"launch manifest artifact is missing: {path}")
        actual = _sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"launch manifest hash mismatch for {relative}: expected {expected}, got {actual}"
            )
        verified[relative] = actual
    return {
        "manifest_identity": _file_identity(manifest_path),
        "verified_artifact_hashes": verified,
        "scope": "every artifact listed by the launch manifest; local manifest is not signed",
    }


def _validate_launch_provenance(repo: Path) -> dict[str, Any]:
    launch_dir = repo / LAUNCH_PROVENANCE
    launch_manifest = _verify_sha256_manifest(
        launch_dir / "manifest.sha256", repo, launch_dir
    )
    for required_name in ("key_files.sha256", "source_snapshot.tar.gz"):
        required_entry = str(LAUNCH_PROVENANCE / required_name)
        if required_entry not in launch_manifest["verified_artifact_hashes"]:
            raise ValueError(f"run5 launch manifest does not bind {required_entry}")
    key_path = launch_dir / "key_files.sha256"
    if not key_path.is_file():
        raise FileNotFoundError(f"run5 launch provenance is missing: {key_path}")
    recorded: dict[str, str] = {}
    for line in key_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split(maxsplit=1)
        try:
            valid_digest = len(fields[0]) == 64 and int(fields[0], 16) >= 0
        except (IndexError, ValueError):
            valid_digest = False
        if len(fields) != 2 or not valid_digest:
            raise ValueError(f"malformed run5 key hash line: {line!r}")
        relative = fields[1].strip().removeprefix("./")
        if relative in recorded:
            raise ValueError(f"duplicate run5 key hash entry: {relative}")
        recorded[relative] = fields[0].lower()
    matched = {}
    for relative in LAUNCH_BOUND_FILES:
        if relative not in recorded:
            raise ValueError(f"run5 launch provenance does not bind {relative}")
        current_path = repo / relative
        current = _sha256_file(current_path)
        expected = recorded[relative]
        if current != expected:
            raise ValueError(
                f"current static GraphDef source differs from run5 launch: {relative}: "
                f"expected {expected}, got {current}"
            )
        matched[relative] = current
    snapshot_path = launch_dir / "source_snapshot.tar.gz"
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"run5 source snapshot is missing: {snapshot_path}")
    snapshot_matched = {}
    with tarfile.open(snapshot_path, "r:gz") as archive:
        members = {}
        for member in archive.getmembers():
            if not member.isfile():
                continue
            if member.name in members:
                raise ValueError(f"duplicate file member in run5 source snapshot: {member.name}")
            members[member.name] = member
        for relative in LAUNCH_SNAPSHOT_BOUND_FILES:
            if relative not in members:
                raise ValueError(f"run5 source snapshot does not contain {relative}")
            handle = archive.extractfile(members[relative])
            if handle is None:
                raise ValueError(f"cannot read {relative} from run5 source snapshot")
            expected = hashlib.sha256(handle.read()).hexdigest()
            current = _sha256_file(repo / relative)
            if current != expected:
                raise ValueError(
                    f"current inference source differs from run5 launch snapshot: {relative}: "
                    f"expected {expected}, got {current}"
                )
            snapshot_matched[relative] = current
    return {
        "directory": str(launch_dir.resolve()),
        "launch_manifest": launch_manifest,
        "key_files_sha256": _sha256_file(key_path),
        "launch_bound_current_hashes": matched,
        "source_snapshot_identity": _file_identity(snapshot_path),
        "snapshot_bound_current_hashes": snapshot_matched,
        "all_launch_bound_hashes_match": True,
    }


def _checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"checkpoint step directory not found: {checkpoint}")
    required_dirs = (checkpoint / "params", checkpoint / "train_state", checkpoint / "assets")
    missing = [str(path) for path in required_dirs if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"checkpoint is incomplete; missing directories: {missing}")
    metadata_path = checkpoint / "_CHECKPOINT_METADATA"
    if not metadata_path.is_file() or metadata_path.stat().st_size == 0:
        raise FileNotFoundError(f"finalized checkpoint metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    init_ns = metadata.get("init_timestamp_nsecs")
    commit_ns = metadata.get("commit_timestamp_nsecs")
    if not isinstance(init_ns, int) or not isinstance(commit_ns, int) or commit_ns < init_ns:
        raise ValueError(f"checkpoint metadata has invalid init/commit timestamps: {metadata}")
    for item in ("params", "train_state"):
        for name in ("_METADATA", "manifest.ocdbt"):
            path = checkpoint / item / name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"checkpoint item is not finalized: {path}")
    if not checkpoint.name.isdigit():
        raise ValueError(f"checkpoint step directory must have a numeric name: {checkpoint.name!r}")
    return {
        "step_dir": str(checkpoint),
        "step_label": int(checkpoint.name),
        "init_timestamp_nsecs": init_ns,
        "commit_timestamp_nsecs": commit_ns,
        "checkpoint_metadata_sha256": _sha256_file(metadata_path),
    }


def _validate_archive_snapshot_manifest(
    checkpoint: Path, live_step: Path, checkpoint_info: dict[str, Any]
) -> dict[str, Any]:
    manifest_path = checkpoint.parent / f"{checkpoint.name}.manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"archived run5 checkpoint requires its copy manifest: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_manifest_fields = {
        "schema",
        "step",
        "source",
        "destination",
        "source_metadata",
        "checkpoint_metadata",
        "commit_timestamp_nsecs",
        "files",
    }
    if not isinstance(manifest, dict) or set(manifest) != required_manifest_fields:
        raise ValueError("checkpoint snapshot manifest schema is not exact")
    if manifest.get("schema") != "openpi-checkpoint-snapshot-v1":
        raise ValueError(f"unsupported checkpoint snapshot manifest schema: {manifest.get('schema')!r}")
    if type(manifest.get("step")) is not str or manifest.get("step") != checkpoint.name:
        raise ValueError(f"checkpoint snapshot manifest has wrong step: {manifest.get('step')!r}")
    raw_source = manifest.get("source")
    raw_destination = manifest.get("destination")
    if not isinstance(raw_source, str) or not Path(raw_source).is_absolute():
        raise ValueError(f"checkpoint snapshot manifest has invalid source: {raw_source!r}")
    if not isinstance(raw_destination, str) or not Path(raw_destination).is_absolute():
        raise ValueError(
            f"checkpoint snapshot manifest has invalid destination: {raw_destination!r}"
        )
    source = Path(raw_source).resolve()
    repo = live_step.parents[3]
    hardlink_step = (repo / PILOT_HARDLINK_ARCHIVE / checkpoint.name).resolve()
    if raw_source == str(live_step):
        source_kind = "exact finalized live run5 step"
    elif raw_source == str(hardlink_step):
        source_kind = "exact run5 read-only hardlink intermediary"
    else:
        raise ValueError(f"checkpoint snapshot manifest has wrong source: {raw_source!r}")
    if raw_destination != str(checkpoint):
        raise ValueError(
            f"checkpoint snapshot manifest has wrong destination: {manifest.get('destination')!r}"
        )
    source_metadata = manifest.get("source_metadata")
    source_metadata_fields = {"device", "inode", "size", "mtime_ns", "ctime_ns"}
    if not isinstance(source_metadata, dict) or set(source_metadata) != source_metadata_fields:
        raise ValueError("checkpoint snapshot manifest has invalid source metadata")
    if any(
        isinstance(source_metadata[field], bool)
        or not isinstance(source_metadata[field], int)
        or source_metadata[field] < 0
        for field in source_metadata_fields
    ):
        raise ValueError("checkpoint snapshot manifest has invalid source metadata values")
    checkpoint_metadata = json.loads(
        (checkpoint / "_CHECKPOINT_METADATA").read_text(encoding="utf-8")
    )
    if manifest.get("checkpoint_metadata") != checkpoint_metadata:
        raise ValueError("checkpoint snapshot manifest metadata differs from _CHECKPOINT_METADATA")
    if (
        manifest.get("commit_timestamp_nsecs") != checkpoint_info["commit_timestamp_nsecs"]
        or checkpoint_metadata.get("init_timestamp_nsecs")
        != checkpoint_info["init_timestamp_nsecs"]
    ):
        raise ValueError("checkpoint snapshot manifest timestamps differ from checkpoint metadata")

    files = manifest.get("files")
    if not isinstance(files, list):
        raise ValueError("checkpoint snapshot manifest files must be a list")
    declared: dict[str, dict[str, Any]] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("checkpoint snapshot manifest contains a non-object file entry")
        if set(entry) != {"path", "size", "sha256"}:
            raise ValueError("checkpoint snapshot manifest file entry schema is not exact")
        relative = entry.get("path")
        digest = entry.get("sha256")
        size = entry.get("size")
        if not isinstance(relative, str) or not relative or relative in declared:
            raise ValueError(f"invalid/duplicate checkpoint snapshot file path: {relative!r}")
        path = (checkpoint / relative).resolve()
        if not _is_relative_to(path, checkpoint):
            raise ValueError(f"checkpoint snapshot file escapes checkpoint: {relative!r}")
        if not _is_sha256(digest) or isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"invalid checkpoint snapshot identity for {relative!r}")
        declared[relative] = {"sha256": digest, "size": size}
    if list(declared) != sorted(declared):
        raise ValueError("checkpoint snapshot manifest file entries are not sorted")
    actual_paths = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in actual_paths):
        raise ValueError("checkpoint snapshot contains a symlinked file")
    actual_relatives = {str(path.relative_to(checkpoint)) for path in actual_paths}
    if actual_relatives != set(declared):
        raise ValueError(
            "checkpoint snapshot file set differs from manifest: "
            f"missing={sorted(set(declared) - actual_relatives)}, "
            f"extra={sorted(actual_relatives - set(declared))}"
        )
    for relative, identity in declared.items():
        actual_size = (checkpoint / relative).stat().st_size
        if actual_size != identity["size"]:
            raise ValueError(
                f"checkpoint snapshot size mismatch for {relative}: "
                f"expected {identity['size']}, got {actual_size}"
            )
    control_paths = {
        "_CHECKPOINT_METADATA",
        "params/_METADATA",
        "params/_sharding",
        "params/manifest.ocdbt",
        "train_state/_METADATA",
        "train_state/_sharding",
        "train_state/manifest.ocdbt",
    }
    control_paths.update(relative for relative in declared if relative.startswith("assets/"))
    missing_controls = control_paths - set(declared)
    if missing_controls:
        raise ValueError(f"checkpoint snapshot manifest lacks control files: {sorted(missing_controls)}")
    verified_controls = {}
    for relative in sorted(control_paths):
        actual = _sha256_file(checkpoint / relative)
        expected = declared[relative]["sha256"]
        if actual != expected:
            raise ValueError(
                f"checkpoint snapshot control hash mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )
        verified_controls[relative] = actual
    intermediary_identity = None
    if source_kind == "exact run5 read-only hardlink intermediary":
        if not source.is_dir():
            raise FileNotFoundError(f"checkpoint snapshot intermediary is missing: {source}")
        source_stat = source.stat()
        # ``st_dev`` differs across the login/compute-node NFS mount namespaces.  The stable
        # inode/timestamps/size plus exact file-set, per-file sizes, and control hashes are bound.
        stable_source_metadata = {
            "inode": source_stat.st_ino,
            "size": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
            "ctime_ns": source_stat.st_ctime_ns,
        }
        expected_stable_metadata = {
            key: source_metadata[key] for key in stable_source_metadata
        }
        if stable_source_metadata != expected_stable_metadata:
            raise ValueError(
                "checkpoint snapshot intermediary identity differs from copy-time metadata"
            )
        source_paths = sorted(path for path in source.rglob("*") if path.is_file())
        if any(path.is_symlink() for path in source_paths):
            raise ValueError("checkpoint snapshot intermediary contains a symlinked file")
        source_relatives = {str(path.relative_to(source)) for path in source_paths}
        if source_relatives != set(declared):
            raise ValueError("checkpoint snapshot intermediary file set differs from manifest")
        for relative, identity in declared.items():
            if (source / relative).stat().st_size != identity["size"]:
                raise ValueError(
                    f"checkpoint snapshot intermediary size mismatch for {relative}"
                )
        source_control_hashes = {}
        for relative in sorted(control_paths):
            actual = _sha256_file(source / relative)
            if actual != declared[relative]["sha256"]:
                raise ValueError(
                    f"checkpoint snapshot intermediary control hash mismatch for {relative}"
                )
            source_control_hashes[relative] = actual
        intermediary_identity = {
            "resolved_path": str(source),
            "copy_time_source_metadata": source_metadata,
            "current_stable_source_metadata": stable_source_metadata,
            "device_id_note": (
                "copy-time device id is recorded but not compared because NFS mount namespaces "
                "report different st_dev values on login and compute nodes"
            ),
            "all_files_exist_and_sizes_match": True,
            "verified_control_file_hashes": source_control_hashes,
        }
    return {
        "snapshot_manifest_identity": _file_identity(manifest_path),
        "source_kind": source_kind,
        "source_intermediary_identity": intermediary_identity,
        "declared_file_count": len(declared),
        "declared_total_bytes": sum(identity["size"] for identity in declared.values()),
        "all_declared_files_exist_and_sizes_match": True,
        "independently_verified_control_file_hashes": verified_controls,
        "large_array_hashes": (
            "declared by the copy manifest; restored selected parameter source is independently "
            "finite-checked and byte-hashed by this evaluator"
        ),
    }


def _validate_checkpoint_origin(
    checkpoint: Path, repo: Path, checkpoint_info: dict[str, Any]
) -> dict[str, Any]:
    live_root = (repo / "checkpoints/pi05_yam_mem_v34_run5_eta0/v34_run5_eta0").resolve()
    archive_root = (repo / "diagnostic_checkpoints").resolve()
    if _is_relative_to(checkpoint, live_root):
        kind = "live run5 checkpoint root"
        root = live_root
        archive_manifest = None
    elif _is_relative_to(checkpoint, archive_root) and any(
        "v34_run5_eta0" in part for part in checkpoint.relative_to(archive_root).parts[:-1]
    ):
        kind = "run5-labelled diagnostic archive root"
        root = archive_root
        archive_manifest = _validate_archive_snapshot_manifest(
            checkpoint, live_root / checkpoint.name, checkpoint_info
        )
    else:
        raise ValueError(
            "checkpoint must come from the exact run5 live root or a run5-labelled directory "
            f"under {archive_root}; got {checkpoint}"
        )
    pilot_log = repo / PILOT_LOG
    if not pilot_log.is_file():
        raise FileNotFoundError(f"run5 pilot log is missing: {pilot_log}")
    log_text = pilot_log.read_text(encoding="utf-8", errors="strict")
    recorded_live_step = live_root / checkpoint.name
    required_log_evidence = (
        "Training config: name=pi05_yam_mem_v34_run5_eta0 exp_name=v34_run5_eta0 eta_scale=0.0",
        f"root_directory={live_root}",
        f"Finished saving checkpoint (finalized tmp dir) to `{recorded_live_step}`",
    )
    missing_evidence = [text for text in required_log_evidence if text not in log_text]
    if missing_evidence:
        raise ValueError(
            "checkpoint metadata/path are not bound to the recorded run5 pilot log; "
            f"missing evidence: {missing_evidence}"
        )
    return {
        "origin_kind": kind,
        "accepted_root": str(root),
        "resolved_step_path": str(checkpoint),
        "recorded_live_step_path": str(recorded_live_step),
        "archive_snapshot_manifest": archive_manifest,
        "pilot_log_identity_at_evaluation": _file_identity(pilot_log),
        "pilot_log_evidence": list(required_log_evidence),
        "binding_scope": (
            "exact live/archive path and finalized-step evidence from the run5 pilot log; archived "
            "copies additionally require their full-copy manifest; internal TrainState-step/config/"
            "source validation is always required"
        ),
    }


def _preflight_output(output_dir: Path, checkpoint: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_dir}")
    if _is_relative_to(output_dir, checkpoint):
        raise ValueError(f"output directory must be outside checkpoint step: {checkpoint}")
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.is_dir():
        raise NotADirectoryError(parent)
    fd, probe = tempfile.mkstemp(prefix=".v34-causal-write-probe-", dir=parent)
    os.close(fd)
    Path(probe).unlink()
    return parent


def _parameter_restore_path(checkpoint: Path, source: str) -> Path:
    if source == "ema":
        return checkpoint / "params"
    if source == "raw":
        return checkpoint / "train_state"
    raise ValueError(f"unknown parameter source {source!r}")


def _raw_parameter_transforms() -> dict[str, ocp.RestoreTransform]:
    return {r"params/(.*)": ocp.RestoreTransform(original_key=r"params/\1/value")}


def _restore_raw_pure_params(source_path: Path, target_params: dict[str, Any]) -> dict[str, Any]:
    item = {"params": target_params}
    restore_args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item)
    with ocp.PyTreeCheckpointer() as checkpointer:
        restored = checkpointer.restore(
            source_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=restore_args,
                transforms=_raw_parameter_transforms(),
                transforms_default_to_original=False,
            ),
        )

    def check_shape(actual, target):
        if np.shape(actual) != np.shape(target):
            raise ValueError(f"raw parameter shape mismatch: {np.shape(actual)} != {np.shape(target)}")
        return actual

    return jax.tree.map(check_shape, restored["params"], target_params)


def _restore_train_step(source_path: Path) -> int:
    """Read only TrainState.step directly from its scalar OCDBT/Zarr leaf.

    Orbax 0.11's apparently one-leaf PyTree restore first materializes/probes the complete 36 GB
    TrainState in this checkpoint.  Opening the exact ``step`` TensorStore key avoids that unsafe
    behavior and has been smoke-tested against run5 checkpoints 120, 250, and 500.
    """
    source_path = source_path.resolve()
    if not source_path.is_dir():
        raise FileNotFoundError(f"train_state source does not exist: {source_path}")
    spec = {
        "driver": "zarr",
        "kvstore": {
            "driver": "ocdbt",
            "base": {"driver": "file", "path": str(source_path)},
            "path": "step",
        },
        "recheck_cached_data": False,
        "recheck_cached_metadata": False,
        "fill_missing_data_reads": False,
    }
    step_store = ts.open(spec, open=True).result()
    if tuple(step_store.shape) != ():
        raise ValueError(f"stored TrainState step is not scalar: shape={tuple(step_store.shape)}")
    step_array = np.asarray(step_store.read().result())
    if step_array.shape != () or step_array.dtype != np.dtype(np.int32):
        raise ValueError(
            "restored TrainState step is not the required int32 scalar: "
            f"shape={step_array.shape}, dtype={step_array.dtype}"
        )
    return int(step_array.item())


def _validate_checkpoint_train_step(checkpoint: Path, train_step: int) -> dict[str, int]:
    if not checkpoint.name.isdigit():
        raise ValueError(f"checkpoint label is not numeric: {checkpoint.name!r}")
    label = int(checkpoint.name)
    expected = label + 1
    if train_step != expected:
        raise ValueError(
            f"checkpoint label/internal TrainState step mismatch: label={label}, "
            f"expected internal step={expected}, got {train_step}"
        )
    return {"checkpoint_manager_step_label": label, "internal_train_state_step": train_step}


def _parameter_tree_identity(params: dict[str, Any]) -> dict[str, Any]:
    """Hash every byte in the restored raw or EMA parameter tree, in canonical path order."""
    digest = hashlib.sha256()
    flat = traverse_util.flatten_dict(params, sep="/")
    total_bytes = 0
    leaf_count = 0
    for path, value in sorted(flat.items()):
        digest.update(path.encode())
        digest.update(b"\0")
        if value is None:
            digest.update(b"<none>")
            continue
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"parameter leaf {path} has object dtype")
        try:
            finite = np.isfinite(array)
        except TypeError as exc:
            raise TypeError(f"parameter leaf {path} has nonnumeric dtype {array.dtype}") from exc
        if not np.all(finite):
            raise FloatingPointError(f"parameter leaf {path} contains NaN/Inf")
        digest.update(str(array.shape).encode())
        digest.update(b"\0")
        digest.update(str(array.dtype).encode())
        digest.update(b"\0")
        contiguous = np.ascontiguousarray(array)
        byte_view = contiguous.view(np.uint8).reshape(-1)
        digest.update(memoryview(byte_view))
        total_bytes += int(byte_view.size)
        leaf_count += 1
    return {
        "sha256": digest.hexdigest(),
        "bytes_hashed": total_bytes,
        "array_leaves_hashed": leaf_count,
        "scope": "complete restored parameter tree; canonical paths, shapes, dtypes, and all bytes",
    }


def _source_metadata(source_path: Path) -> dict[str, Any]:
    if not source_path.is_dir():
        raise FileNotFoundError(f"parameter source does not exist: {source_path}")
    entries = {}
    for name in ("_METADATA", "_sharding", "manifest.ocdbt"):
        path = source_path / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"parameter source metadata missing: {path}")
        entries[name] = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    return {"resolved_path": str(source_path), "metadata_files": entries}


def _validate_run5_semantics(
    config_name: str, model_config: Any, data_config: Any, ema_decay: float | None
) -> None:
    failures = []
    if config_name != RUN5_CONFIG:
        failures.append(f"config={config_name!r}")
    if float(model_config.memory.eta_scale) != 0.0:
        failures.append(f"eta_scale={model_config.memory.eta_scale!r}")
    if not bool(model_config.memory.blank_initial_output):
        failures.append("blank_initial_output=False")
    if getattr(model_config, "memory_architecture", None) != "v32_layer8_dual_query":
        failures.append(f"memory_architecture={getattr(model_config, 'memory_architecture', None)!r}")
    if tuple(data_config.heldout_episodes) != EXPECTED_HELDOUT:
        failures.append(f"heldout_episodes={tuple(data_config.heldout_episodes)!r}")
    if int(data_config.memory_stride_frames) != 15:
        failures.append(f"memory_stride_frames={data_config.memory_stride_frames!r}")
    if ema_decay is None or not math.isclose(
        float(ema_decay), EXPECTED_EMA_DECAY, rel_tol=0.0, abs_tol=0.0
    ):
        failures.append(f"ema_decay={ema_decay!r}")
    if failures:
        raise ValueError("run5 static semantics mismatch: " + ", ".join(failures))


def _load_model(
    checkpoint: Path, train_config: Any, parameter_source: Literal["raw", "ema"]
) -> tuple[Any, dict[str, Any]]:
    model_config = train_config.model
    abstract = nnx.state(nnx.eval_shape(model_config.create, jax.random.key(0))).to_pure_dict()
    source_path = _parameter_restore_path(checkpoint, parameter_source)
    source_provenance = _source_metadata(source_path)
    if parameter_source == "ema":
        loaded = _model.restore_params(source_path, restore_type=np.ndarray)
        restore_semantics = "standalone EMA params item"
    else:
        loaded = _restore_raw_pure_params(source_path, abstract)
        restore_semantics = (
            "train_state params/*/value only via explicit Orbax transform; optimizer and EMA excluded"
        )
    train_step = _restore_train_step(checkpoint / "train_state")
    step_identity = _validate_checkpoint_train_step(checkpoint, train_step)
    train_state_storage_identity = _source_metadata(checkpoint / "train_state")
    fingerprint = _parameter_tree_identity(loaded)
    aligned = _align_params(abstract, loaded)
    model = model_config.load(aligned, remove_extra_params=False)
    model.eval()
    source_provenance.update(
        {
            "parameter_source": parameter_source,
            "restore_semantics": restore_semantics,
            "train_state_step_identity": step_identity,
            "train_state_storage_metadata": train_state_storage_identity,
            "restored_parameter_tree_identity": fingerprint,
        }
    )
    return model, source_provenance


def _truth_sign(side: str) -> int:
    if side == "right":
        return 1
    if side == "left":
        return -1
    raise ValueError(f"side must be left/right, got {side!r}")


def _truth_aligned(side: str, right_minus_left: float) -> float:
    return float(_truth_sign(side) * right_minus_left)


def _side_prefix_masks(left_tokens: list[int], right_tokens: list[int]) -> tuple[list[bool], list[bool], int]:
    if len(left_tokens) != len(right_tokens):
        raise ValueError("canonical left/right sequences must have equal token lengths")
    differences = [index for index, pair in enumerate(zip(left_tokens, right_tokens, strict=True)) if pair[0] != pair[1]]
    if len(differences) != 1:
        raise ValueError(f"canonical side sequences must differ at exactly one token, got {differences}")
    index = differences[0]
    if left_tokens[:index] != right_tokens[:index]:
        raise AssertionError("canonical side prefixes differ before the side token")
    mask = [position <= index for position in range(len(left_tokens))]
    return mask.copy(), mask.copy(), index


def _plan_match_vector(plan: Any) -> tuple[int, int, int, int, int]:
    return (
        int(plan.evidence[0]),
        int(plan.evidence[1]),
        int(plan.memory[0]),
        int(plan.memory[1]),
        int(plan.length),
    )


def _build_donor_maps(
    all_plans: list[Any], recipient_plans: list[Any], stride: int
) -> dict[str, Any]:
    """Pre-register matched, training-exposed same/opposite donors for each heldout target.

    A donor is the non-heldout episode minimizing L1 distance in
    ``[evidence_start, evidence_end, wait_start, wait_end, episode_length]`` within the required
    instruction/side cell; ties break by episode id. Using the same non-heldout pool for both
    side interventions avoids heldout-vs-training-exposure asymmetry. The exact selected IDs are
    pinned so a dataset or annotation change fails rather than silently changing the experiment.
    """
    if stride != 15:
        raise ValueError(f"donor pre-registration requires stride 15, got {stride}")
    by_episode = {int(plan.episode): plan for plan in all_plans}
    if len(by_episode) != len(all_plans):
        raise ValueError("episode plans contain duplicate episode ids")
    recipients = {int(plan.episode): plan for plan in recipient_plans}
    if tuple(sorted(recipients)) != EXPECTED_HELDOUT:
        raise ValueError(
            f"donor maps require exact heldout recipients {EXPECTED_HELDOUT}, got {sorted(recipients)}"
        )
    heldout = set(EXPECTED_HELDOUT)
    maps: dict[str, dict[int, int]] = {"same_side": {}, "opposite_side": {}}
    selection: dict[int, dict[str, Any]] = {}
    for episode in EXPECTED_HELDOUT:
        target = recipients[episode]
        if target.side not in ("left", "right"):
            raise ValueError(f"recipient episode {episode} has invalid side {target.side!r}")
        target_vector = _plan_match_vector(target)
        selection[episode] = {}
        wanted_sides = {
            "same_side": target.side,
            "opposite_side": "right" if target.side == "left" else "left",
        }
        for kind, wanted_side in wanted_sides.items():
            candidates = [
                plan
                for plan in all_plans
                if int(plan.episode) not in heldout
                and plan.prompt == target.prompt
                and plan.side == wanted_side
            ]
            if not candidates:
                raise ValueError(
                    f"episode {episode} has no nonheldout {kind} donor for "
                    f"prompt={target.prompt!r}, side={wanted_side!r}"
                )

            def rank(plan, reference=target_vector):
                vector = _plan_match_vector(plan)
                distance = sum(
                    abs(left - right) for left, right in zip(vector, reference, strict=True)
                )
                return (distance, int(plan.episode))

            donor = min(candidates, key=rank)
            donor_episode = int(donor.episode)
            distance = int(rank(donor)[0])
            maps[kind][episode] = donor_episode
            selection[episode][kind] = {
                "episode": donor_episode,
                "prompt": donor.prompt,
                "side": donor.side,
                "target_match_vector": target_vector,
                "donor_match_vector": _plan_match_vector(donor),
                "l1_frame_distance": distance,
                "tie_break": "lowest episode id",
                "training_exposed": True,
            }
    for kind, expected in EXPECTED_DONORS.items():
        if maps[kind] != expected:
            raise ValueError(
                f"pre-registered {kind} donor map changed: expected {expected}, got {maps[kind]}"
            )
    return {**maps, "selection": selection}


def _phase_side(label: str) -> str:
    lower = label.lower()
    if "left" in lower:
        return "left"
    if "right" in lower:
        return "right"
    return "-"


def _scheduled_grid_rows(
    rows: list[dict[str, Any]],
    stride: int,
    *,
    after_exclusive: int = -1,
    through_inclusive: int,
) -> list[dict[str, Any]]:
    if stride <= 0:
        raise ValueError("stride must be positive")
    ordered = sorted(rows, key=lambda row: int(row["frame_index"]))
    frames = [int(row["frame_index"]) for row in ordered]
    if len(frames) != len(set(frames)):
        raise ValueError("episode rows contain duplicate frame_index values")
    return [
        row
        for row in ordered
        if after_exclusive < int(row["frame_index"]) <= through_inclusive
        and int(row["frame_index"]) % stride == 0
    ]


def _validate_episode_truth(plan: Any, rows: list[dict[str, Any]], tasks: dict[int, str], stride: int) -> tuple[int, ...]:
    waiting = [
        row
        for row in rows
        if int(row["frame_index"]) % stride == 0
        and plan.memory[0] <= int(row["frame_index"]) <= plan.memory[1]
    ]
    frames = tuple(int(row["frame_index"]) for row in waiting)
    if frames != EXPECTED_WAIT_FRAMES[int(plan.episode)]:
        raise ValueError(
            f"episode {plan.episode} waiting grid changed: expected {EXPECTED_WAIT_FRAMES[int(plan.episode)]}, got {frames}"
        )
    sides = {_phase_side(tasks[int(row["task_index"])]) for row in waiting}
    if sides != {plan.side}:
        raise ValueError(f"episode {plan.episode} waiting truth mismatch: plan={plan.side}, labels={sides}")
    final_side = _phase_side(tasks[int(rows[-1]["task_index"])])
    if final_side != plan.side:
        raise ValueError(f"episode {plan.episode} final side {final_side} != waiting side {plan.side}")
    return frames


def _action_motion(actions: np.ndarray, raw_state: np.ndarray) -> dict[str, float | str]:
    actions = np.asarray(actions, dtype=np.float64)
    raw_state = np.asarray(raw_state, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 14 or raw_state.shape != (14,):
        raise ValueError(f"expected actions [H,14] and state [14], got {actions.shape}/{raw_state.shape}")
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(raw_state)):
        raise FloatingPointError("nonfinite robot action/state")
    displacement = actions - raw_state[None, :]
    # YAM state/action order is [left 6 arm joints, left gripper, right 6 arm joints,
    # right gripper].  Gripper opening is not a side-choice statistic and is excluded.
    left = float(np.sqrt(np.mean(np.square(displacement[:, :6]))))
    right = float(np.sqrt(np.mean(np.square(displacement[:, 7:13]))))
    left_endpoint = float(np.linalg.norm(displacement[-1, :6]))
    right_endpoint = float(np.linalg.norm(displacement[-1, 7:13]))
    score = right - left
    return {
        "left_arm_trajectory_rms": left,
        "right_arm_trajectory_rms": right,
        "right_minus_left": score,
        "left_arm_endpoint_l2": left_endpoint,
        "right_arm_endpoint_l2": right_endpoint,
        "larger_motion_arm": "right" if score > 0 else "left" if score < 0 else "tie",
    }


def _action_eligible_frames(
    plan: Any, rows: list[dict[str, Any]], tasks: dict[int, str], *, stride: int, horizon: int
) -> tuple[int, ...]:
    if stride != 15:
        raise ValueError(f"action frame pre-registration requires stride 15, got {stride}")
    by_frame = {int(row["frame_index"]): row for row in rows}
    execute_frames = {
        frame
        for frame, row in by_frame.items()
        if _phase_side(tasks[int(row["task_index"])]) == plan.side
        and tasks[int(row["task_index"])].lower().startswith("open ")
        and " bin" in tasks[int(row["task_index"])].lower()
    }
    eligible = []
    for frame in EXPECTED_WAIT_FRAMES[int(plan.episode)]:
        covered = sum((frame + offset) in execute_frames for offset in range(horizon))
        if covered >= MIN_EXECUTE_FRAMES_IN_ACTION_CHUNK:
            eligible.append(frame)
    result = tuple(eligible)
    if result != EXPECTED_ACTION_FRAMES[int(plan.episode)]:
        raise ValueError(
            f"episode {plan.episode} action frame pre-registration changed: "
            f"expected {EXPECTED_ACTION_FRAMES[int(plan.episode)]}, got {result}"
        )
    return result


def _validate_demonstrated_action_score(episode: int, frame: int, score: float) -> None:
    key = (episode, frame)
    if key not in EXPECTED_DEMONSTRATED_ACTION_SCORES:
        raise ValueError(f"action score is not pre-registered for ep{episode} f{frame}")
    expected = EXPECTED_DEMONSTRATED_ACTION_SCORES[key]
    if not math.isclose(score, expected, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"demonstrated action calibration changed at ep{episode} f{frame}: "
            f"expected {expected}, got {score}"
        )


def _strict_json(value: Any) -> Any:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FloatingPointError(f"report contains nonfinite float: {value}")
        return value
    if isinstance(value, np.generic):
        return _strict_json(value.item())
    if isinstance(value, dict):
        return {str(key): _strict_json(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_strict_json(item) for item in value]
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _array_sha256(value: Any) -> str:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("cannot hash object array")
    if np.issubdtype(array.dtype, np.inexact) and not np.all(np.isfinite(array)):
        raise FloatingPointError("cannot hash nonfinite array")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode())
    digest.update(b"\0")
    digest.update(str(contiguous.dtype).encode())
    digest.update(b"\0")
    digest.update(memoryview(contiguous.view(np.uint8).reshape(-1)))
    return digest.hexdigest()


def _array_difference(left: Any, right: Any) -> dict[str, float]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError(f"array difference shape mismatch: {left_array.shape} != {right_array.shape}")
    difference = left_array - right_array
    if not np.all(np.isfinite(difference)):
        raise FloatingPointError("array difference is nonfinite")
    return {
        "max_abs": float(np.max(np.abs(difference))),
        "rms": float(np.sqrt(np.mean(np.square(difference)))),
        "l2": float(np.linalg.norm(difference)),
    }


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if not arrays:
        raise ValueError("refusing to write an empty action artifact")
    for name, value in arrays.items():
        if not name or not isinstance(name, str):
            raise ValueError("NPZ artifact keys must be nonempty strings")
        array = np.asarray(value)
        if np.issubdtype(array.dtype, np.inexact) and not np.all(np.isfinite(array)):
            raise FloatingPointError(f"artifact array {name!r} is nonfinite")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return _file_identity(path)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    result = float(np.mean(np.asarray(values, dtype=np.float64)))
    if not math.isfinite(result):
        raise FloatingPointError("nonfinite aggregate")
    return result


def _episode_means(records: list[dict[str, Any]], value_fn) -> dict[int, float]:
    result = {}
    for episode in EXPECTED_HELDOUT:
        values = [float(value_fn(row)) for row in records if int(row["episode"]) == episode]
        result[episode] = _mean(values)
    return result


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _validate_record_schema(
    records: list[dict[str, Any]],
    mode: str,
    expected_replicates: tuple[int, ...] = ACTION_NOISE_REPLICATES,
) -> None:
    """Reject missing, duplicated, nonfinite, or exploratory cells before acceptance."""
    if mode not in ("token", "full"):
        raise ValueError(mode)
    expected_grid = {
        (episode, frame)
        for episode, frames in EXPECTED_WAIT_FRAMES.items()
        for frame in frames
    }
    actual_grid = [(int(row["episode"]), int(row["frame"])) for row in records]
    if len(actual_grid) != len(set(actual_grid)):
        raise ValueError("records contain duplicate episode/frame cells")
    if set(actual_grid) != expected_grid:
        raise ValueError(
            f"record grid differs from pre-registration: missing={sorted(expected_grid - set(actual_grid))}, "
            f"extra={sorted(set(actual_grid) - expected_grid)}"
        )
    required_margin_fields = {
        "logp_left",
        "logp_right",
        "right_minus_left",
        "truth_aligned",
        "correct",
    }
    for row in records:
        episode = int(row["episode"])
        frame = int(row["frame"])
        truth = str(row["truth_side"])
        expected_prompt, expected_truth = EXPECTED_CELLS[episode]
        if str(row.get("prompt")) != expected_prompt or truth != expected_truth:
            raise ValueError(
                f"ep{episode} f{frame} heldout cell mismatch: "
                f"expected {(expected_prompt, expected_truth)}, got {(row.get('prompt'), truth)}"
            )
        if int(row.get("truth_sign", 0)) != _truth_sign(truth):
            raise ValueError(f"ep{episode} f{frame} has wrong truth_sign")
        if int(row.get("writes_before_current_frame", -1)) != frame // 15:
            raise ValueError(f"ep{episode} f{frame} has wrong pre-write count")
        expected_same = EXPECTED_DONORS["same_side"][episode]
        expected_opposite = EXPECTED_DONORS["opposite_side"][episode]
        if (
            int(row.get("same_side_donor_episode", -1)) != expected_same
            or str(row.get("same_side_donor_side")) != truth
        ):
            raise ValueError(f"ep{episode} f{frame} has wrong same-side donor metadata")
        other_side = "right" if truth == "left" else "left"
        if (
            int(row.get("opposite_side_donor_episode", -1)) != expected_opposite
            or str(row.get("opposite_side_donor_side")) != other_side
        ):
            raise ValueError(f"ep{episode} f{frame} has wrong opposite-side donor metadata")
        if (
            int(row.get("heldout_opposite_donor_episode", -1))
            != EXPECTED_HELDOUT_OPPOSITE_DONORS[episode]
            or str(row.get("heldout_opposite_donor_side")) != other_side
        ):
            raise ValueError(f"ep{episode} f{frame} has wrong heldout-opposite donor metadata")
        if not _is_sha256(row.get("token_noise_sha256")):
            raise ValueError(f"ep{episode} f{frame} has no valid matched token-noise digest")
        margins = row.get("margins")
        if not isinstance(margins, dict) or set(margins) != set(CONDITIONS):
            raise ValueError(f"ep{episode} f{frame} has wrong condition set")
        for condition, values in margins.items():
            if not isinstance(values, dict) or set(values) != required_margin_fields:
                raise ValueError(f"ep{episode} f{frame} {condition} has wrong margin fields")
            numeric = {name: float(values[name]) for name in required_margin_fields - {"correct"}}
            if not all(math.isfinite(value) for value in numeric.values()):
                raise FloatingPointError(f"ep{episode} f{frame} {condition} has nonfinite margin")
            difference = numeric["logp_right"] - numeric["logp_left"]
            if not math.isclose(difference, numeric["right_minus_left"], abs_tol=1e-6):
                raise ValueError(f"ep{episode} f{frame} {condition} logp difference is inconsistent")
            aligned = _truth_aligned(truth, difference)
            if not math.isclose(aligned, numeric["truth_aligned"], abs_tol=1e-6):
                raise ValueError(f"ep{episode} f{frame} {condition} truth alignment is inconsistent")
            if bool(values["correct"]) != (aligned > 0.0):
                raise ValueError(f"ep{episode} f{frame} {condition} correctness is inconsistent")

        eligible = frame in EXPECTED_ACTION_FRAMES[episode]
        if mode == "token":
            if row.get("actions") or "ground_truth_action" in row:
                raise ValueError("token mode must not contain action samples")
            continue
        samples = row.get("actions", [])
        if not eligible:
            if samples or "ground_truth_action" in row:
                raise ValueError(f"noneligible ep{episode} f{frame} contains action data")
            continue
        if "ground_truth_action" not in row:
            raise ValueError(f"eligible ep{episode} f{frame} lacks demonstrated action calibration")
        ground_truth_score = float(row["ground_truth_action"]["right_minus_left"])
        _validate_demonstrated_action_score(episode, frame, ground_truth_score)
        actual_replicates = [int(sample["noise_replicate"]) for sample in samples]
        if tuple(actual_replicates) != expected_replicates:
            raise ValueError(
                f"ep{episode} f{frame} action replicates must be exact ordered "
                f"{expected_replicates}, got {actual_replicates}"
            )
        for sample in samples:
            if not _is_sha256(sample.get("noise_sha256")):
                raise ValueError(f"ep{episode} f{frame} action sample lacks a valid noise digest")
            condition_values = sample.get("conditions")
            if not isinstance(condition_values, dict) or set(condition_values) != set(CONDITIONS):
                raise ValueError(f"ep{episode} f{frame} action sample has wrong condition set")
            for condition, values in condition_values.items():
                for name in ("right_minus_left", "truth_aligned_score"):
                    value = float(values[name])
                    if not math.isfinite(value):
                        raise FloatingPointError(
                            f"ep{episode} f{frame} action {condition}/{name} is nonfinite"
                        )
                aligned = _truth_aligned(truth, float(values["right_minus_left"]))
                if not math.isclose(aligned, float(values["truth_aligned_score"]), abs_tol=1e-6):
                    raise ValueError(
                        f"ep{episode} f{frame} action {condition} truth alignment is inconsistent"
                    )
            reset_error = float(sample["reset_invariant_max_abs_robot_action"])
            if not math.isfinite(reset_error) or reset_error < 0.0:
                raise FloatingPointError(f"ep{episode} f{frame} has invalid reset invariant")


def _coverage_check(values: list[float], threshold: float) -> dict[str, Any]:
    mean = _mean(values)
    coverage = _mean([float(value > threshold) for value in values])
    return {
        "values": values,
        "mean": mean,
        "threshold": threshold,
        "directional_fraction": coverage,
        "required_directional_fraction": MIN_DIRECTIONAL_COVERAGE,
        "passes": mean > threshold and coverage >= MIN_DIRECTIONAL_COVERAGE,
    }


def _acceptance_summary(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    """Conservative, pre-registered, cell-balanced causal-influence criteria.

    No result is labelled proof: this is open-loop influence evidence on four heldout episodes.
    A primary criterion must clear both its episode mean and >=75% of paired frame/noise cells,
    preventing a single large outlier from hiding mostly wrong directions. Dynamics-only and the
    reciprocal-heldout swap are reported separately as sensitivity controls and do not redefine
    the exposure-matched primary estimand.
    """
    _validate_record_schema(records, mode)

    reset_logp_max = max(
        abs(float(row["margins"][left][field]) - float(row["margins"][right][field]))
        for row in records
        for left, right in (("normal_zero_read", "reset"), ("reset", "never_write"))
        for field in ("logp_left", "logp_right", "right_minus_left")
    )
    reset_token_invariant = reset_logp_max <= RESET_TOL

    def margin(row, condition):
        return float(row["margins"][condition]["truth_aligned"])

    def raw_margin(row, condition):
        return float(row["margins"][condition]["right_minus_left"])

    token_episode: dict[int, dict[str, Any]] = {}
    for episode in EXPECTED_HELDOUT:
        subset = [row for row in records if int(row["episode"]) == episode]
        means = {condition: _mean([margin(row, condition) for row in subset]) for condition in CONDITIONS}
        vectors = {
            condition: [margin(row, condition) for row in subset] for condition in CONDITIONS
        }
        paired_vectors = {
            "normal_minus_zero": [
                margin(row, "normal") - margin(row, "normal_zero_read") for row in subset
            ],
            "normal_minus_reset": [margin(row, "normal") - margin(row, "reset") for row in subset],
            "normal_abs_confidence_minus_reset": [
                abs(raw_margin(row, "normal")) - abs(raw_margin(row, "reset")) for row in subset
            ],
            "post_evidence_minus_pre_evidence": [
                margin(row, "frozen_after_evidence") - margin(row, "pre_evidence_frozen")
                for row in subset
            ],
            "post_evidence_minus_reset": [
                margin(row, "frozen_after_evidence") - margin(row, "reset") for row in subset
            ],
            "post_evidence_abs_confidence_minus_reset": [
                abs(raw_margin(row, "frozen_after_evidence")) - abs(raw_margin(row, "reset"))
                for row in subset
            ],
            "same_side_minus_opposite_swap": [
                margin(row, "same_side_frozen_swap") - margin(row, "opposite_frozen_swap")
                for row in subset
            ],
            "opposite_effect_minus_same_episode_effect": [
                (
                    margin(row, "frozen_after_evidence")
                    - margin(row, "opposite_frozen_swap")
                    - abs(
                        margin(row, "frozen_after_evidence")
                        - margin(row, "same_side_frozen_swap")
                    )
                )
                for row in subset
            ],
        }
        primary_criteria = {
            "normal_favors_truth": _coverage_check(vectors["normal"], 0.0),
            "frozen_native_favors_truth": _coverage_check(vectors["frozen_after_evidence"], 0.0),
            "same_side_swap_favors_truth": _coverage_check(vectors["same_side_frozen_swap"], 0.0),
            "opposite_swap_favors_donor": _coverage_check(
                [-value for value in vectors["opposite_frozen_swap"]], 0.0
            ),
            "normal_read_beats_zero": _coverage_check(
                paired_vectors["normal_minus_zero"], TOKEN_EFFECT_EPS
            ),
            "normal_read_beats_reset": _coverage_check(
                paired_vectors["normal_minus_reset"], TOKEN_EFFECT_EPS
            ),
            "normal_confidence_exceeds_reset": _coverage_check(
                paired_vectors["normal_abs_confidence_minus_reset"], TOKEN_EFFECT_EPS
            ),
            "evidence_writes_help": _coverage_check(
                paired_vectors["post_evidence_minus_pre_evidence"], TOKEN_EFFECT_EPS
            ),
            "post_evidence_beats_reset": _coverage_check(
                paired_vectors["post_evidence_minus_reset"], TOKEN_EFFECT_EPS
            ),
            "post_evidence_confidence_exceeds_reset": _coverage_check(
                paired_vectors["post_evidence_abs_confidence_minus_reset"], TOKEN_EFFECT_EPS
            ),
            "same_side_beats_opposite_swap": _coverage_check(
                paired_vectors["same_side_minus_opposite_swap"], TOKEN_EFFECT_EPS
            ),
            "side_swap_effect_dominates_episode_swap_effect": _coverage_check(
                paired_vectors["opposite_effect_minus_same_episode_effect"], TOKEN_EFFECT_EPS
            ),
        }
        sensitivity_controls = {
            "dynamics_only_favors_truth": _coverage_check(vectors["dynamics_only"], 0.0),
            "heldout_opposite_swap_favors_donor": _coverage_check(
                [-value for value in vectors["heldout_opposite_frozen_swap"]], 0.0
            ),
        }
        token_episode[episode] = {
            "mean_truth_aligned_margin": means,
            "paired_vectors": paired_vectors,
            "primary_criteria": primary_criteria,
            "sensitivity_controls": sensitivity_controls,
            "passes": all(entry["passes"] for entry in primary_criteria.values()),
            "sensitivity_controls_pass": all(
                entry["passes"] for entry in sensitivity_controls.values()
            ),
        }
    token_pass = reset_token_invariant and all(entry["passes"] for entry in token_episode.values())
    token_sensitivity_pass = all(
        entry["sensitivity_controls_pass"] for entry in token_episode.values()
    )

    condition_summary = {}
    for condition in CONDITIONS:
        per_episode = _episode_means(records, lambda row, c=condition: margin(row, c))
        per_episode_accuracy = {
            episode: _mean(
                [
                    float(row["margins"][condition]["correct"])
                    for row in records
                    if int(row["episode"]) == episode
                ]
            )
            for episode in EXPECTED_HELDOUT
        }
        condition_summary[condition] = {
            "episode_mean_truth_aligned_margin": per_episode,
            "cell_macro_mean_margin": _mean(list(per_episode.values())),
            "episode_accuracy": per_episode_accuracy,
            "cell_macro_accuracy": _mean(list(per_episode_accuracy.values())),
        }

    reset_episode_raw = _episode_means(records, lambda row: raw_margin(row, "reset"))
    reset_bias = {
        "episode_mean_right_minus_left": reset_episode_raw,
        "cell_macro_right_minus_left": _mean(list(reset_episode_raw.values())),
        "cell_macro_abs_right_minus_left": _mean([abs(value) for value in reset_episode_raw.values()]),
        "note": "descriptive only; paired confidence/effect gates above determine acceptance",
    }

    action_report: dict[str, Any] | None = None
    action_pass: bool | None = None
    if mode == "full":
        action_episode = {}
        action_reset_max = 0.0
        for row in records:
            for sample in row.get("actions", []):
                action_reset_max = max(
                    action_reset_max,
                    float(sample["reset_invariant_max_abs_robot_action"]),
                )
        reset_action_invariant = action_reset_max <= RESET_TOL
        for episode in EXPECTED_HELDOUT:
            samples = [
                sample
                for row in records
                if int(row["episode"]) == episode
                for sample in row.get("actions", [])
            ]
            if not samples:
                raise ValueError(f"full mode has no action samples for episode {episode}")
            scores = {
                condition: [
                    float(sample["conditions"][condition]["truth_aligned_score"])
                    for sample in samples
                ]
                for condition in CONDITIONS
            }
            raw_scores = {
                condition: [
                    float(sample["conditions"][condition]["right_minus_left"])
                    for sample in samples
                ]
                for condition in CONDITIONS
            }
            paired_vectors = {
                "normal_minus_zero": [
                    left - right
                    for left, right in zip(
                        scores["normal"], scores["normal_zero_read"], strict=True
                    )
                ],
                "normal_minus_reset": [
                    left - right
                    for left, right in zip(scores["normal"], scores["reset"], strict=True)
                ],
                "normal_abs_confidence_minus_reset": [
                    abs(left) - abs(right)
                    for left, right in zip(raw_scores["normal"], raw_scores["reset"], strict=True)
                ],
                "post_evidence_minus_pre_evidence": [
                    left - right
                    for left, right in zip(
                        scores["frozen_after_evidence"], scores["pre_evidence_frozen"], strict=True
                    )
                ],
                "post_evidence_minus_reset": [
                    left - right
                    for left, right in zip(
                        scores["frozen_after_evidence"], scores["reset"], strict=True
                    )
                ],
                "post_evidence_abs_confidence_minus_reset": [
                    abs(left) - abs(right)
                    for left, right in zip(
                        raw_scores["frozen_after_evidence"], raw_scores["reset"], strict=True
                    )
                ],
                "same_side_minus_opposite_swap": [
                    left - right
                    for left, right in zip(
                        scores["same_side_frozen_swap"], scores["opposite_frozen_swap"], strict=True
                    )
                ],
                "opposite_effect_minus_same_episode_effect": [
                    native - opposite - abs(native - same)
                    for native, same, opposite in zip(
                        scores["frozen_after_evidence"],
                        scores["same_side_frozen_swap"],
                        scores["opposite_frozen_swap"],
                        strict=True,
                    )
                ],
            }
            primary_criteria = {
                "normal_favors_truth_materially": _coverage_check(
                    scores["normal"], ACTION_SCORE_EPS
                ),
                "frozen_native_favors_truth_materially": _coverage_check(
                    scores["frozen_after_evidence"], ACTION_SCORE_EPS
                ),
                "same_side_swap_favors_truth_materially": _coverage_check(
                    scores["same_side_frozen_swap"], ACTION_SCORE_EPS
                ),
                "opposite_swap_favors_donor_materially": _coverage_check(
                    [-value for value in scores["opposite_frozen_swap"]], ACTION_SCORE_EPS
                ),
                "normal_read_beats_zero": _coverage_check(
                    paired_vectors["normal_minus_zero"], ACTION_EFFECT_EPS
                ),
                "normal_read_beats_reset": _coverage_check(
                    paired_vectors["normal_minus_reset"], ACTION_EFFECT_EPS
                ),
                "normal_confidence_exceeds_reset": _coverage_check(
                    paired_vectors["normal_abs_confidence_minus_reset"], ACTION_EFFECT_EPS
                ),
                "evidence_writes_help": _coverage_check(
                    paired_vectors["post_evidence_minus_pre_evidence"], ACTION_EFFECT_EPS
                ),
                "post_evidence_beats_reset": _coverage_check(
                    paired_vectors["post_evidence_minus_reset"], ACTION_EFFECT_EPS
                ),
                "post_evidence_confidence_exceeds_reset": _coverage_check(
                    paired_vectors["post_evidence_abs_confidence_minus_reset"], ACTION_EFFECT_EPS
                ),
                "same_side_beats_opposite_swap": _coverage_check(
                    paired_vectors["same_side_minus_opposite_swap"], 2.0 * ACTION_SCORE_EPS
                ),
                "side_swap_effect_dominates_episode_swap_effect": _coverage_check(
                    paired_vectors["opposite_effect_minus_same_episode_effect"], ACTION_EFFECT_EPS
                ),
            }
            sensitivity_controls = {
                "dynamics_only_favors_truth_materially": _coverage_check(
                    scores["dynamics_only"], ACTION_SCORE_EPS
                ),
                "heldout_opposite_swap_favors_donor_materially": _coverage_check(
                    [-value for value in scores["heldout_opposite_frozen_swap"]],
                    ACTION_SCORE_EPS,
                ),
            }
            action_episode[episode] = {
                "samples": len(samples),
                "mean_truth_aligned_score": {
                    condition: _mean(values) for condition, values in scores.items()
                },
                "paired_vectors": paired_vectors,
                "primary_criteria": primary_criteria,
                "sensitivity_controls": sensitivity_controls,
                "passes": all(entry["passes"] for entry in primary_criteria.values()),
                "sensitivity_controls_pass": all(
                    entry["passes"] for entry in sensitivity_controls.values()
                ),
            }
        action_pass = reset_action_invariant and all(entry["passes"] for entry in action_episode.values())
        action_sensitivity_pass = all(
            entry["sensitivity_controls_pass"] for entry in action_episode.values()
        )
        action_report = {
            "score_definition": (
                "RMS(right six arm-joint displacement) - RMS(left six arm-joint displacement); "
                "grippers excluded"
            ),
            "material_effect_threshold": ACTION_SCORE_EPS,
            "paired_effect_threshold": ACTION_EFFECT_EPS,
            "minimum_directional_coverage": MIN_DIRECTIONAL_COVERAGE,
            "reset_invariant_tolerance": RESET_TOL,
            "reset_invariant_max_abs_robot_action": action_reset_max,
            "reset_invariant_pass": reset_action_invariant,
            "episodes": action_episode,
            "passes": action_pass,
            "sensitivity_controls_pass": action_sensitivity_pass,
            "acceptance_scope": (
                "passes uses exposure-matched primary criteria; dynamics-only and reciprocal-heldout "
                "branches are separately reported sensitivity controls"
            ),
        }

    if mode == "token":
        classification = (
            "token-level matched-intervention influence criteria pass; action behavior not run"
            if token_pass
            else "token-level matched-intervention influence criteria do not pass"
        )
        behavioral_evidence: bool | None = None
    else:
        behavioral_evidence = bool(token_pass and action_pass)
        if behavioral_evidence:
            classification = "source-level matched token-and-action causal-influence evidence passes"
        elif token_pass:
            classification = "token influence pattern only; action behavior does not pass"
        else:
            classification = "matched causal-influence criteria do not pass"

    return {
        "criteria_version": "run5-preregistered-v2",
        "token_effect_epsilon_nats": TOKEN_EFFECT_EPS,
        "minimum_directional_coverage": MIN_DIRECTIONAL_COVERAGE,
        "reset_tolerance": RESET_TOL,
        "reset_zero_never_logp_max_abs": reset_logp_max,
        "reset_zero_never_token_invariant_pass": reset_token_invariant,
        "condition_summary": condition_summary,
        "reset_side_bias": reset_bias,
        "token_episode_checks": token_episode,
        "causal_token_pass": token_pass,
        "token_sensitivity_controls_pass": token_sensitivity_pass,
        "action": action_report,
        "causal_action_pass": action_pass,
        "source_behavioral_influence_evidence_pass": behavioral_evidence,
        "proof_claimed": False,
        "classification": classification,
        "cross_source_requirement": (
            "Robust run5 influence evidence requires separate raw and EMA full reports both to pass; "
            "this report never pools parameter sources."
        ),
        "limitations": [
            "Same/opposite donors are matched nonheldout episodes; this controls side specificity but donor "
            "histories can still differ in visual content from each heldout recipient.",
            "Open-loop action sign is a calibrated heldout statistic, not closed-loop task success.",
            "Checkpoint-500/1000 token mode is trend evidence only; it cannot satisfy the action-level endpoint.",
            "This audit reports causal influence under interventions; it does not claim a complete proof of "
            "closed-loop episodic-memory necessity or sufficiency.",
        ],
    }


class CausalMemoryEval:
    def __init__(self, args: Args):
        self.args = args
        self.repo = Path(__file__).resolve().parents[1]
        self.launch_provenance = _validate_launch_provenance(self.repo)
        self.checkpoint_info = _checkpoint_metadata(args.checkpoint)
        self.checkpoint_origin = _validate_checkpoint_origin(
            args.checkpoint, self.repo, self.checkpoint_info
        )
        _preflight_output(args.output_dir, args.checkpoint)
        if not args.dataset_root.is_dir():
            raise FileNotFoundError(f"dataset root not found: {args.dataset_root}")

        self.train_config = _config.get_config(args.config)
        self.tokenizer_asset_provenance = _resolve_tokenizer_assets()
        self.data_config = self.train_config.data.create(self.train_config.assets_dirs, self.train_config.model)
        _validate_run5_semantics(
            args.config,
            self.train_config.model,
            self.data_config,
            self.train_config.ema_decay,
        )
        self.stride = int(self.data_config.memory_stride_frames)

        asset_id = self.data_config.asset_id
        if asset_id is None:
            raise ValueError("run5 data config has no normalization asset id")
        norm_path = args.checkpoint / "assets" / asset_id
        if not norm_path.is_dir():
            raise FileNotFoundError(f"checkpoint-local normalization stats missing: {norm_path}")
        self.norm_asset_identity = _directory_identity(norm_path)
        self.norm_stats = _normalize.load(norm_path)

        self.model, self.parameter_provenance = _load_model(
            args.checkpoint, self.train_config, args.parameter_source
        )
        if float(self.model.memory.config.eta_scale) != 0.0:
            raise ValueError(f"loaded GraphDef has eta_scale={self.model.memory.config.eta_scale}, expected 0")
        if not bool(self.model.memory.config.blank_initial_output):
            raise ValueError("loaded GraphDef does not enforce blank_initial_output")

        input_transforms = [
            transform
            for transform in self.data_config.data_transforms.inputs
            if not isinstance(transform, _transforms.BuildMemorySequence)
        ]
        self.input_transform = _transforms.compose(
            [
                *input_transforms,
                _transforms.Normalize(self.norm_stats, use_quantiles=self.data_config.use_quantile_norm),
                *self.data_config.model_transforms.inputs,
            ]
        )
        self.output_transform = _transforms.compose(
            [
                *self.data_config.model_transforms.outputs,
                _transforms.Unnormalize(self.norm_stats, use_quantiles=self.data_config.use_quantile_norm),
                *self.data_config.data_transforms.outputs,
            ]
        )
        fast_snapshot_path = self.tokenizer_asset_provenance["fast_snapshot"]["resolved_path"]
        self.tokenizer = _tokenizer.FASTSubtaskTokenizer(
            self.train_config.model.max_token_len,
            fast_tokenizer_path=fast_snapshot_path,
        )._paligemma_tokenizer
        self.stop_token = int(self.tokenizer.encode("placeholder subtask\n")[-1])
        self.side_buffers, self.side_token_provenance = self._make_side_buffers()
        self.stop_buffers = self._make_forced_buffers([self.stop_token], [True])
        self._sample = nnx_utils.module_jit(
            self.model.sample_with_memory,
            static_argnames=(
                "stop_token",
                "max_decode_steps",
                "num_steps",
                "zero_read",
                "allow_write",
                "write_mode",
            ),
        )

    def _make_forced_buffers(self, tokens: list[int], live: list[bool]):
        if len(tokens) != len(live) or len(tokens) > self.model.causal_token_len:
            raise ValueError("invalid forced buffers")
        padded = np.zeros((1, self.model.causal_token_len), dtype=np.int32)
        mask = np.zeros_like(padded, dtype=bool)
        padded[0, : len(tokens)] = tokens
        mask[0, : len(live)] = live
        return jnp.asarray(padded), jnp.asarray(mask)

    def _make_side_buffers(self):
        texts = {
            "left": "wait; target bin is left\n",
            "right": "wait; target bin is right\n",
        }
        tokens = {side: list(self.tokenizer.encode(text)) for side, text in texts.items()}
        left_mask, right_mask, differing = _side_prefix_masks(tokens["left"], tokens["right"])
        buffers = {
            "left": self._make_forced_buffers(tokens["left"], left_mask),
            "right": self._make_forced_buffers(tokens["right"], right_mask),
        }
        return buffers, {
            "texts": texts,
            "token_ids": tokens,
            "token_count": len(tokens["left"]),
            "side_token_index": differing,
            "score_mask_includes_through_side_token_only": True,
        }

    def _observation(self, row: dict[str, Any], frame: int, prompt: str):
        decode = _wc.WriterContributionRunner._decode_inline_image
        transformed = self.input_transform(
            {
                "observation/image": decode(row["image"], field="image", raw_frame=frame),
                "observation/left_wrist_image": decode(
                    row["left_wrist_image"], field="left_wrist_image", raw_frame=frame
                ),
                "observation/right_wrist_image": decode(
                    row["right_wrist_image"], field="right_wrist_image", raw_frame=frame
                ),
                "observation/state": np.asarray(row["state"], dtype=np.float32),
                "prompt": prompt,
            }
        )
        transformed_state = np.asarray(transformed["state"], dtype=np.float32)
        batched = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], transformed)
        return _model.Observation.from_dict(batched), transformed_state

    def _noise(self, episode: int, frame: int, replicate: int):
        key = jax.random.key(np.uint32(self.args.seed))
        for value in (episode, frame, replicate):
            key = jax.random.fold_in(key, np.uint32(value))
        noise = jax.random.normal(
            key,
            (1, self.model.action_horizon, self.model.action_dim),
            dtype=jnp.float32,
        )
        return key, noise

    def _sample_frozen(
        self,
        key,
        noise,
        observation,
        state,
        *,
        zero_read: bool,
        forced: tuple[Any, Any] | None,
        num_steps: int,
        max_decode_steps: int,
    ):
        tokens, mask = forced if forced is not None else (None, None)
        output = self._sample(
            key,
            observation,
            state,
            stop_token=self.stop_token,
            max_decode_steps=max_decode_steps,
            num_steps=num_steps,
            noise=noise,
            action_prefix=None,
            forced_subtask_tokens=tokens,
            forced_subtask_mask=mask,
            zero_read=zero_read,
            allow_write=False,
            write_mode="frozen",
        )
        jax.block_until_ready(output)
        actions, returned_state, aux = output
        if bool(np.asarray(aux["write_occurred"])[0]):
            raise RuntimeError("counterfactual scoring unexpectedly committed a write")
        if self._state_max_abs_diff(returned_state, state) != 0.0:
            raise RuntimeError("frozen counterfactual returned a changed memory state")
        return actions, returned_state, aux

    def _advance(self, key, noise, observation, state, write_mode: str):
        tokens, mask = self.stop_buffers
        output = self._sample(
            key,
            observation,
            state,
            stop_token=self.stop_token,
            max_decode_steps=1,
            num_steps=1,
            noise=noise,
            action_prefix=None,
            forced_subtask_tokens=tokens,
            forced_subtask_mask=mask,
            zero_read=False,
            allow_write=write_mode == "normal",
            write_mode=write_mode,
        )
        jax.block_until_ready(output)
        _actions, new_state, aux = output
        occurred = bool(np.asarray(aux["write_occurred"])[0])
        if occurred != (write_mode == "normal"):
            raise RuntimeError(f"write_occurred mismatch for {write_mode}: {occurred}")
        if write_mode == "frozen" and self._state_max_abs_diff(new_state, state) != 0.0:
            raise RuntimeError("frozen replay changed memory state")
        return new_state

    def _score_side(self, key, noise, observation, state, truth: str, *, zero_read: bool):
        logps = {}
        for side in ("left", "right"):
            _actions, _returned, aux = self._sample_frozen(
                key,
                noise,
                observation,
                state,
                zero_read=zero_read,
                forced=self.side_buffers[side],
                num_steps=1,
                max_decode_steps=1,
            )
            value = float(np.asarray(aux["conditioned_subtask_logp"])[0])
            if not math.isfinite(value):
                raise FloatingPointError(f"nonfinite conditioned logp for {side}")
            logps[side] = value
        difference = logps["right"] - logps["left"]
        aligned = _truth_aligned(truth, difference)
        return {
            "logp_left": logps["left"],
            "logp_right": logps["right"],
            "right_minus_left": difference,
            "truth_aligned": aligned,
            "correct": aligned > 0.0,
        }

    def _decode(self, aux: dict[str, Any]) -> str:
        tokens = np.asarray(aux["tokens"])[0]
        mask = np.asarray(aux["token_mask"])[0]
        return self.tokenizer.decode(tokens[mask].tolist()).strip()

    def _robot_actions(self, model_actions: Any, transformed_state: np.ndarray) -> np.ndarray:
        output = self.output_transform(
            {
                "state": np.array(transformed_state, copy=True),
                "actions": np.asarray(model_actions)[0],
            }
        )
        actions = np.asarray(output["actions"], dtype=np.float64)
        if actions.shape != (self.model.action_horizon, 14) or not np.all(np.isfinite(actions)):
            raise FloatingPointError(f"invalid robot-space actions: {actions.shape}")
        return actions

    def _action_variant(
        self,
        key,
        noise,
        observation,
        transformed_state,
        raw_state,
        state,
        truth,
        *,
        zero_read: bool,
    ):
        actions, _returned, aux = self._sample_frozen(
            key,
            noise,
            observation,
            state,
            zero_read=zero_read,
            forced=None,
            num_steps=self.args.action_denoise_steps,
            max_decode_steps=ACTION_MAX_DECODE_STEPS,
        )
        model_array = np.asarray(actions)[0]
        if not np.all(np.isfinite(model_array)):
            raise FloatingPointError("nonfinite model-space actions")
        robot = self._robot_actions(actions, transformed_state)
        metric = _action_motion(robot, raw_state)
        metric["truth_aligned_score"] = _truth_aligned(truth, float(metric["right_minus_left"]))
        return {
            "decoded_subtask": self._decode(aux),
            **metric,
        }, model_array, robot

    @staticmethod
    def _state_max_abs_diff(left, right) -> float:
        leaves = [
            jnp.max(jnp.abs(a.astype(jnp.float32) - b.astype(jnp.float32)))
            for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
        ]
        return float(np.asarray(jnp.max(jnp.stack(leaves))))

    def _load_episode_payloads(self):
        all_plans = _probe._plan_all_episodes(
            SimpleNamespace(dataset_root=self.args.dataset_root), self.data_config
        )
        by_episode = {int(plan.episode): plan for plan in all_plans}
        if len(by_episode) != len(all_plans):
            raise ValueError("all-episode plan contains duplicate ids")
        recipient_plans = [by_episode[episode] for episode in self.args.episodes if episode in by_episode]
        if tuple(int(plan.episode) for plan in recipient_plans) != EXPECTED_HELDOUT:
            raise ValueError(
                f"failed to resolve exact heldouts: {[plan.episode for plan in recipient_plans]}"
            )
        donor_maps = _build_donor_maps(all_plans, recipient_plans, self.stride)
        selected_episodes = set(EXPECTED_HELDOUT)
        for kind in ("same_side", "opposite_side"):
            selected_episodes.update(donor_maps[kind].values())
        snapshot_plans = [by_episode[episode] for episode in sorted(selected_episodes)]
        tasks = _wa._read_tasks(self.args.dataset_root)
        sources = _wc._load_lerobot_sources(
            self.args.dataset_root, [plan.episode for plan in snapshot_plans]
        )
        columns = [
            "image",
            "left_wrist_image",
            "right_wrist_image",
            "state",
            "actions",
            "frame_index",
            "task_index",
        ]
        payloads = {}
        for plan, source in zip(snapshot_plans, sources, strict=True):
            rows = pq.read_table(source.path, columns=columns).to_pylist()
            episode = int(plan.episode)
            if episode in set(EXPECTED_HELDOUT):
                _validate_episode_truth(plan, rows, tasks, self.stride)
                action_frames = _action_eligible_frames(
                    plan,
                    rows,
                    tasks,
                    stride=self.stride,
                    horizon=self.model.action_horizon,
                )
            else:
                action_frames = ()
            payloads[int(plan.episode)] = {
                "plan": plan,
                "rows": rows,
                "by_frame": {int(row["frame_index"]): row for row in rows},
                "action_frames": action_frames,
                "source_path": str(source.path),
                "source_identity": _file_identity(source.path),
                "is_heldout_recipient": episode in set(EXPECTED_HELDOUT),
            }
        return recipient_plans, snapshot_plans, donor_maps, payloads

    def _build_snapshots(self, plans, payloads):
        snapshots = {}
        for plan in plans:
            episode = int(plan.episode)
            grid = _scheduled_grid_rows(
                payloads[episode]["rows"],
                self.stride,
                through_inclusive=int(plan.evidence[1]),
            )
            state = self.model.memory.init_state(1)
            pre_evidence = None
            evidence_grid_frames = []
            for row in grid:
                frame = int(row["frame_index"])
                if frame >= plan.evidence[0] and pre_evidence is None:
                    pre_evidence = state
                if plan.evidence[0] <= frame <= plan.evidence[1]:
                    evidence_grid_frames.append(frame)
                observation, _ = self._observation(row, frame, plan.prompt)
                key, noise = self._noise(episode, frame, 0)
                state = self._advance(key, noise, observation, state, "normal")
            if pre_evidence is None or not evidence_grid_frames:
                raise ValueError(f"episode {episode} has no evidence frame on stride grid")
            snapshots[episode] = {
                "initial": self.model.memory.init_state(1),
                "pre_evidence": pre_evidence,
                "post_evidence": state,
                "evidence_grid_frames": tuple(evidence_grid_frames),
                "writes_through_evidence": len(grid),
            }
            if self._state_max_abs_diff(snapshots[episode]["initial"], self.model.memory.init_state(1)) != 0.0:
                raise RuntimeError("fresh memory state is not deterministic")
        return snapshots

    def _ground_truth_action(self, payload, frame: int, truth: str):
        by_frame = payload["by_frame"]
        future = []
        for offset in range(self.model.action_horizon):
            target = frame + offset
            if target not in by_frame:
                raise ValueError(f"ground-truth action horizon leaves episode at frame {frame}")
            future.append(np.asarray(by_frame[target]["actions"], dtype=np.float64))
        actions = np.stack(future)
        raw_state = np.asarray(by_frame[frame]["state"], dtype=np.float64)
        metric = _action_motion(actions, raw_state)
        episode = int(payload["plan"].episode)
        _validate_demonstrated_action_score(
            episode, frame, float(metric["right_minus_left"])
        )
        aligned = _truth_aligned(truth, float(metric["right_minus_left"]))
        if aligned <= ACTION_SCORE_EPS:
            raise ValueError(
                f"pre-registered action frame {frame} has non-material/wrong demonstrated score {aligned}"
            )
        metric["truth_aligned_score"] = aligned
        return metric, actions

    def _provenance(self, plans, donor_maps, payloads, snapshots) -> dict[str, Any]:
        repo = self.repo
        source_files = [
            Path(__file__).resolve(),
            repo / "scripts/serve_yam_memory.py",
            repo / "scripts/train.py",
            repo / "src/openpi/training/config.py",
            repo / "src/openpi/transforms.py",
            repo / "src/openpi/models/pi0.py",
            repo / "src/openpi/models/memory.py",
            repo / "src/openpi/models/model.py",
            repo / "src/openpi/models/tokenizer.py",
            repo / "src/openpi/models/pi0_config.py",
            repo / "src/openpi/policies/yam_policy.py",
            repo / "src/openpi/shared/normalize.py",
            repo / "src/openpi/shared/download.py",
            repo / "src/openpi/training/checkpoints.py",
            repo / "src/openpi/training/utils.py",
            repo / "src/openpi/diagnostics/v32_checkpoint.py",
            repo / "src/openpi/diagnostics/v33_write_token_probe.py",
            repo / "src/openpi/diagnostics/v33_writer_attention.py",
            repo / "src/openpi/diagnostics/writer_contribution.py",
        ]
        dataset_files = [
            self.args.dataset_root / "meta/tasks.jsonl",
            self.args.dataset_root / "meta/episode_prompts.json",
            self.args.dataset_root / "meta/info.json",
            self.args.dataset_root / "meta/episodes.jsonl",
        ]
        return {
            "checkpoint": self.checkpoint_info,
            "checkpoint_origin": self.checkpoint_origin,
            "run5_launch_provenance": self.launch_provenance,
            "config": self.args.config,
            "ema_decay": float(self.train_config.ema_decay),
            "memory_eta_scale": float(self.model.memory.config.eta_scale),
            "blank_initial_output": bool(self.model.memory.config.blank_initial_output),
            "parameter_source": self.parameter_provenance,
            "dataset_root": str(self.args.dataset_root),
            "normalization_asset_identity": self.norm_asset_identity,
            "tokenizer_asset_provenance": self.tokenizer_asset_provenance,
            "dataset_metadata_identity": {
                str(path): _file_identity(path) for path in dataset_files
            },
            "source_identity": {str(path): _file_identity(path) for path in source_files},
            "heldout_episodes": list(self.args.episodes),
            "heldout_cells": [
                {"episode": int(plan.episode), "prompt": plan.prompt, "side": plan.side}
                for plan in plans
            ],
            "same_instruction_same_side_donor_map": donor_maps["same_side"],
            "same_instruction_opposite_side_donor_map": donor_maps["opposite_side"],
            "heldout_reciprocal_opposite_side_donor_map": EXPECTED_HELDOUT_OPPOSITE_DONORS,
            "donor_selection": donor_maps["selection"],
            "selected_episode_parquet_identity": {
                episode: payloads[episode]["source_identity"] for episode in sorted(payloads)
            },
            "snapshot_replay": {
                episode: {
                    "writes_through_evidence": snapshots[episode]["writes_through_evidence"],
                    "evidence_grid_frames": snapshots[episode]["evidence_grid_frames"],
                }
                for episode in sorted(snapshots)
            },
            "stride": self.stride,
            "mode": self.args.mode,
            "network_mode": "HF_HUB_OFFLINE=TRANSFORMERS_OFFLINE=HF_DATASETS_OFFLINE=1",
            "seed": self.args.seed,
            "action_noise_replicates": list(self.args.action_noise_replicates),
            "action_denoise_steps": self.args.action_denoise_steps,
            "action_max_decode_steps": ACTION_MAX_DECODE_STEPS,
            "production_action_max_decode_steps": 10,
            "action_horizon": self.model.action_horizon,
            "action_dim_model": self.model.action_dim,
            "side_scoring": self.side_token_provenance,
            "interventions": {
                "normal": "production state before the current waiting-frame write",
                "normal_zero_read": "same normal state/current observation; retrieved content set to zero; no commit",
                "reset": "fresh run5 init_state M0; read enabled; no commit",
                "never_write": "episode state with no writes from frame zero; exactly M0; no commit",
                "pre_evidence_frozen": "normal replay stopped immediately before first evidence-grid write",
                "frozen_after_evidence": "normal replay through last evidence-grid write, then no state dynamics",
                "dynamics_only": "post-evidence state with eta=0 momentum/alpha decay but zero new gradient",
                "same_side_frozen_swap": (
                    "matched nonheldout same-instruction/same-side post-evidence frozen state"
                ),
                "opposite_frozen_swap": "same-instruction opposite-side post-evidence frozen state",
                "heldout_opposite_frozen_swap": (
                    "same-instruction opposite-side post-evidence frozen state from the paired "
                    "heldout recipient; sensitivity control separate from matched nonheldout donors"
                ),
            },
            "write_timing": "all scores/actions use the pre-write state at the current waiting frame",
            "matched_noise": "one explicit normal-noise tensor per episode/frame/replicate reused by every condition",
            "action_frame_rule": (
                f"waiting grid frames whose demonstrated H={self.model.action_horizon} chunk contains at least "
                f"{MIN_EXECUTE_FRAMES_IN_ACTION_CHUNK} execute-phase frames"
            ),
            "expected_wait_frames": EXPECTED_WAIT_FRAMES,
            "expected_action_frames": EXPECTED_ACTION_FRAMES,
            "expected_demonstrated_action_right_minus_left": {
                f"ep{episode}_f{frame}": score
                for (episode, frame), score in EXPECTED_DEMONSTRATED_ACTION_SCORES.items()
            },
        }

    def run(self) -> dict[str, Any]:
        started = time.monotonic()
        # Reserve the destination before replay so another process can never race us into
        # overwriting it. A crash leaves INCOMPLETE.json, never a plausible completed report.
        self.args.output_dir.mkdir(parents=False, exist_ok=False)
        incomplete = self.args.output_dir / "INCOMPLETE.json"
        incomplete.write_text(
            json.dumps(
                _strict_json(
                    {
                        "schema": SCHEMA_VERSION,
                        "checkpoint": self.checkpoint_info,
                        "parameter_source": self.args.parameter_source,
                        "status": "reserved; replay/provenance not yet complete",
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )

        plans, snapshot_plans, donor_maps, payloads = self._load_episode_payloads()
        snapshots = self._build_snapshots(snapshot_plans, payloads)
        provenance = self._provenance(plans, donor_maps, payloads, snapshots)
        incomplete.write_text(
            json.dumps(_strict_json({"schema": SCHEMA_VERSION, "provenance": provenance}), indent=2),
            encoding="utf-8",
        )

        records = []
        action_artifacts: dict[str, np.ndarray] = {}
        for plan in plans:
            episode = int(plan.episode)
            payload = payloads[episode]
            snapshot = snapshots[episode]
            normal = snapshot["post_evidence"]
            dynamics = snapshot["post_evidence"]
            frozen = snapshot["post_evidence"]
            reset = snapshot["initial"]
            never_write = self.model.memory.init_state(1)
            if self._state_max_abs_diff(reset, never_write) != 0.0:
                raise RuntimeError("reset and never-written state differ before replay")
            same_donor_episode = donor_maps["same_side"][episode]
            opposite_donor_episode = donor_maps["opposite_side"][episode]
            heldout_opposite_episode = EXPECTED_HELDOUT_OPPOSITE_DONORS[episode]
            same_donor_state = snapshots[same_donor_episode]["post_evidence"]
            opposite_donor_state = snapshots[opposite_donor_episode]["post_evidence"]
            heldout_opposite_state = snapshots[heldout_opposite_episode]["post_evidence"]
            same_donor_plan = payloads[same_donor_episode]["plan"]
            opposite_donor_plan = payloads[opposite_donor_episode]["plan"]
            heldout_opposite_plan = payloads[heldout_opposite_episode]["plan"]
            grid = _scheduled_grid_rows(
                payload["rows"],
                self.stride,
                after_exclusive=int(plan.evidence[1]),
                through_inclusive=int(plan.memory[1]),
            )
            normal_write_count = int(snapshot["writes_through_evidence"])
            for row in grid:
                frame = int(row["frame_index"])
                expected_writes_before = len(
                    _scheduled_grid_rows(
                        payload["rows"],
                        self.stride,
                        through_inclusive=frame - 1,
                    )
                )
                if normal_write_count != expected_writes_before:
                    raise RuntimeError(
                        f"pre-write timing mismatch ep{episode} f{frame}: "
                        f"state has {normal_write_count} writes, expected {expected_writes_before}"
                    )
                observation, transformed_state = self._observation(row, frame, plan.prompt)
                key, noise = self._noise(episode, frame, 0)
                in_waiting = plan.memory[0] <= frame <= plan.memory[1]
                if in_waiting:
                    states = {
                        "normal": (normal, False),
                        "normal_zero_read": (normal, True),
                        "reset": (reset, False),
                        "never_write": (never_write, False),
                        "pre_evidence_frozen": (snapshot["pre_evidence"], False),
                        "frozen_after_evidence": (frozen, False),
                        "dynamics_only": (dynamics, False),
                        "same_side_frozen_swap": (same_donor_state, False),
                        "opposite_frozen_swap": (opposite_donor_state, False),
                        "heldout_opposite_frozen_swap": (heldout_opposite_state, False),
                    }
                    margins = {
                        condition: self._score_side(
                            key,
                            noise,
                            observation,
                            state,
                            plan.side,
                            zero_read=zero_read,
                        )
                        for condition, (state, zero_read) in states.items()
                    }
                    record: dict[str, Any] = {
                        "episode": episode,
                        "prompt": plan.prompt,
                        "truth_side": plan.side,
                        "truth_sign": _truth_sign(plan.side),
                        "frame": frame,
                        "writes_before_current_frame": normal_write_count,
                        "same_side_donor_episode": same_donor_episode,
                        "same_side_donor_side": same_donor_plan.side,
                        "opposite_side_donor_episode": opposite_donor_episode,
                        "opposite_side_donor_side": opposite_donor_plan.side,
                        "heldout_opposite_donor_episode": heldout_opposite_episode,
                        "heldout_opposite_donor_side": heldout_opposite_plan.side,
                        "token_noise_sha256": _array_sha256(noise),
                        "margins": margins,
                        "effects": {
                            "normal_minus_zero": (
                                margins["normal"]["truth_aligned"]
                                - margins["normal_zero_read"]["truth_aligned"]
                            ),
                            "normal_minus_reset": (
                                margins["normal"]["truth_aligned"] - margins["reset"]["truth_aligned"]
                            ),
                            "post_evidence_minus_pre_evidence": (
                                margins["frozen_after_evidence"]["truth_aligned"]
                                - margins["pre_evidence_frozen"]["truth_aligned"]
                            ),
                            "post_evidence_minus_reset": (
                                margins["frozen_after_evidence"]["truth_aligned"]
                                - margins["reset"]["truth_aligned"]
                            ),
                            "native_frozen_minus_opposite_swap": (
                                margins["frozen_after_evidence"]["truth_aligned"]
                                - margins["opposite_frozen_swap"]["truth_aligned"]
                            ),
                            "same_side_minus_opposite_swap": (
                                margins["same_side_frozen_swap"]["truth_aligned"]
                                - margins["opposite_frozen_swap"]["truth_aligned"]
                            ),
                            "native_frozen_minus_heldout_opposite_swap": (
                                margins["frozen_after_evidence"]["truth_aligned"]
                                - margins["heldout_opposite_frozen_swap"]["truth_aligned"]
                            ),
                        },
                    }
                    if self.args.mode == "full" and frame in set(payload["action_frames"]):
                        ground_truth_metric, ground_truth_actions = self._ground_truth_action(
                            payload, frame, plan.side
                        )
                        record["ground_truth_action"] = ground_truth_metric
                        action_samples = []
                        raw_state = np.asarray(row["state"], dtype=np.float64)
                        frame_prefix = f"ep{episode:03d}_f{frame:04d}"
                        action_artifacts[f"{frame_prefix}__ground_truth_robot_actions"] = (
                            ground_truth_actions
                        )
                        action_artifacts[f"{frame_prefix}__raw_state"] = raw_state
                        action_artifacts[f"{frame_prefix}__transformed_state"] = transformed_state
                        for replicate in self.args.action_noise_replicates:
                            action_key, action_noise = self._noise(episode, frame, replicate)
                            condition_metrics = {}
                            robot_arrays = {}
                            model_arrays = {}
                            sample_prefix = f"{frame_prefix}_r{replicate:02d}"
                            noise_array = np.asarray(action_noise)[0]
                            action_artifacts[f"{sample_prefix}__matched_model_noise"] = noise_array
                            for condition, (state, zero_read) in states.items():
                                metric, model_array, robot = self._action_variant(
                                    action_key,
                                    action_noise,
                                    observation,
                                    transformed_state,
                                    raw_state,
                                    state,
                                    plan.side,
                                    zero_read=zero_read,
                                )
                                condition_metrics[condition] = metric
                                model_arrays[condition] = model_array
                                robot_arrays[condition] = robot
                                action_artifacts[f"{sample_prefix}__{condition}__model_actions"] = (
                                    model_array
                                )
                                action_artifacts[f"{sample_prefix}__{condition}__robot_actions"] = robot
                            reset_max = max(
                                _array_difference(robot_arrays[left], robot_arrays[right])["max_abs"]
                                for left, right in (
                                    ("normal_zero_read", "reset"),
                                    ("reset", "never_write"),
                                )
                            )
                            if reset_max > RESET_TOL:
                                raise RuntimeError(
                                    f"reset/zero/no-write action invariant failed at ep{episode} f{frame}: {reset_max}"
                                )
                            action_samples.append(
                                {
                                    "noise_replicate": replicate,
                                    "noise_sha256": _array_sha256(noise_array),
                                    "conditions": condition_metrics,
                                    "reset_invariant_max_abs_robot_action": reset_max,
                                    "paired_robot_action_differences": {
                                        "normal_vs_zero": _array_difference(
                                            robot_arrays["normal"], robot_arrays["normal_zero_read"]
                                        ),
                                        "normal_vs_reset": _array_difference(
                                            robot_arrays["normal"], robot_arrays["reset"]
                                        ),
                                        "native_vs_pre_evidence": _array_difference(
                                            robot_arrays["frozen_after_evidence"],
                                            robot_arrays["pre_evidence_frozen"],
                                        ),
                                        "native_vs_same_side": _array_difference(
                                            robot_arrays["frozen_after_evidence"],
                                            robot_arrays["same_side_frozen_swap"],
                                        ),
                                        "native_vs_opposite_side": _array_difference(
                                            robot_arrays["frozen_after_evidence"],
                                            robot_arrays["opposite_frozen_swap"],
                                        ),
                                        "native_vs_heldout_opposite_side": _array_difference(
                                            robot_arrays["frozen_after_evidence"],
                                            robot_arrays["heldout_opposite_frozen_swap"],
                                        ),
                                        "same_side_vs_opposite_side": _array_difference(
                                            robot_arrays["same_side_frozen_swap"],
                                            robot_arrays["opposite_frozen_swap"],
                                        ),
                                    },
                                    "model_action_sha256": {
                                        condition: _array_sha256(value)
                                        for condition, value in model_arrays.items()
                                    },
                                    "robot_action_sha256": {
                                        condition: _array_sha256(value)
                                        for condition, value in robot_arrays.items()
                                    },
                                }
                            )
                        record["actions"] = action_samples
                    records.append(record)
                    print(
                        f"ep{episode} f{frame}: normal={margins['normal']['truth_aligned']:+.4f} "
                        f"zero={margins['normal_zero_read']['truth_aligned']:+.4f} "
                        f"frozen={margins['frozen_after_evidence']['truth_aligned']:+.4f} "
                        f"same={margins['same_side_frozen_swap']['truth_aligned']:+.4f} "
                        f"opposite={margins['opposite_frozen_swap']['truth_aligned']:+.4f} "
                        f"heldout_opposite={margins['heldout_opposite_frozen_swap']['truth_aligned']:+.4f}",
                        flush=True,
                    )

                # Advance only after scoring: every recorded intervention saw M_{t-1}.
                normal = self._advance(key, noise, observation, normal, "normal")
                dynamics = self._advance(key, noise, observation, dynamics, "dynamics_only")
                frozen = self._advance(key, noise, observation, frozen, "frozen")
                normal_write_count += 1

        expected_record_count = sum(len(value) for value in EXPECTED_WAIT_FRAMES.values())
        if len(records) != expected_record_count:
            raise RuntimeError(f"expected {expected_record_count} waiting records, got {len(records)}")
        acceptance = _acceptance_summary(records, self.args.mode)
        if not acceptance["reset_zero_never_token_invariant_pass"]:
            raise RuntimeError(
                "reset/zero/no-write logp invariant failed: "
                f"{acceptance['reset_zero_never_logp_max_abs']}"
            )
        artifact_identity: dict[str, Any] | None = None
        if self.args.mode == "full":
            artifact_path = self.args.output_dir / "action_arrays.npz"
            artifact_identity = _write_npz_atomic(artifact_path, action_artifacts)
            artifact_identity.update(
                {
                    "array_count": len(action_artifacts),
                    "arrays": {
                        name: {
                            "shape": list(np.asarray(value).shape),
                            "dtype": str(np.asarray(value).dtype),
                            "sha256": _array_sha256(value),
                        }
                        for name, value in sorted(action_artifacts.items())
                    },
                }
            )
        elif action_artifacts:
            raise RuntimeError("token mode unexpectedly accumulated action artifacts")
        report = _strict_json(
            {
                "schema": SCHEMA_VERSION,
                "provenance": provenance,
                "records": records,
                "acceptance": acceptance,
                "artifacts": {"action_arrays": artifact_identity},
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        temporary_report = self.args.output_dir / ".report.json.tmp"
        with temporary_report.open("w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_report, self.args.output_dir / "report.json")
        complete_lines = [f"{hashlib.sha256(encoded.encode()).hexdigest()}  report.json"]
        if artifact_identity is not None:
            complete_lines.append(f"{artifact_identity['sha256']}  action_arrays.npz")
        complete_path = self.args.output_dir / "COMPLETE"
        complete_path.write_text("\n".join(complete_lines) + "\n", encoding="utf-8")
        with complete_path.open("rb") as handle:
            os.fsync(handle.fileno())
        incomplete.unlink()
        print(json.dumps(acceptance, indent=2), flush=True)
        print(f"wrote {self.args.output_dir / 'report.json'}", flush=True)
        return report


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    CausalMemoryEval(args).run()


if __name__ == "__main__":
    main()
