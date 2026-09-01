"""Portable local-path contract for the self-contained memory project.

All machine-local v3.5 inputs and outputs live below one ``memory_project``
directory.  The directory is discovered from this source checkout by default
and can be relocated explicitly with ``MEMORY_PROJECT_ROOT``.  Remote object
identifiers such as the official Pi0.5 ``gs://`` URI are deliberately outside
this contract.
"""

from __future__ import annotations

from collections.abc import Iterator
import os
import pathlib

MEMORY_PROJECT_ROOT_ENV = "MEMORY_PROJECT_ROOT"

# Checked-in/project-relative layout.  Keep these values relative so a copied
# memory_project directory has no embedded source-cluster location.
OPENPI_DIR = pathlib.PurePosixPath("openpi")
DATA_DIR = pathlib.PurePosixPath("data")
LEROBOT_HOME = DATA_DIR / "lerobot"
V35_ROOT = pathlib.PurePosixPath("v35")
CACHE_DIR = V35_ROOT / "cache"
UV_CACHE_DIR = CACHE_DIR / "uv"
HUGGINGFACE_HOME = CACHE_DIR / "huggingface"
HF_DATASETS_CACHE = HUGGINGFACE_HOME / "datasets"
OPENPI_DATA_HOME = CACHE_DIR / "openpi"
JAX_CACHE_DIR = CACHE_DIR / "jax"
TMP_DIR = V35_ROOT / "tmp"
WANDB_DIR = V35_ROOT / "wandb"

V35_REPO_ID = "yam/bin_memory_0830_0831_v36_subtask"
V35_DATASET_DIR = LEROBOT_HOME / V35_REPO_ID
V35_FROZEN_MANIFEST = DATA_DIR / "0830_0831_episode_manifest_v36_frozen.json"
V35_ASSETS_ROOT = V35_ROOT / "assets"
V35_ASSETS_DIR = V35_ASSETS_ROOT / "pi05_yam_0830_0831_v36"
V35_CHECKPOINTS_DIR = V35_ROOT / "checkpoints"
V35_DIAGNOSTICS_DIR = V35_ROOT / "diagnostics"


class ProjectRootError(ValueError):
    """Raised when a project-root override or project-relative path is invalid."""


def _candidate_roots(start: pathlib.Path) -> Iterator[pathlib.Path]:
    """Yield ``start`` (or its parent for a file) and all of its parents."""

    current = start.resolve()
    if current.is_file():
        current = current.parent
    yield current
    yield from current.parents


def _looks_like_memory_project(path: pathlib.Path) -> bool:
    """Return whether ``path`` has the source layout of this project."""

    return (path / "openpi" / "pyproject.toml").is_file() and (path / "openpi" / "src" / "openpi").is_dir()


def discover_memory_project_root(
    *,
    source_file: pathlib.Path | None = None,
    cwd: pathlib.Path | None = None,
) -> pathlib.Path:
    """Discover the source checkout's ``memory_project`` directory.

    Source-file ancestry is authoritative.  The current directory is only a
    fallback for editable/packaged executions whose installed module no longer
    sits below the checkout.
    """

    source = pathlib.Path(__file__) if source_file is None else pathlib.Path(source_file)
    working_dir = pathlib.Path.cwd() if cwd is None else pathlib.Path(cwd)
    seen: set[pathlib.Path] = set()
    for anchor in (source, working_dir):
        for candidate in _candidate_roots(anchor):
            if candidate in seen:
                continue
            seen.add(candidate)
            if _looks_like_memory_project(candidate):
                return candidate
    raise ProjectRootError(
        f"could not discover memory_project from source {source.resolve()} or cwd {working_dir.resolve()}; "
        f"set {MEMORY_PROJECT_ROOT_ENV} to the copied project directory"
    )


def memory_project_root() -> pathlib.Path:
    """Return the portable project root, honoring ``MEMORY_PROJECT_ROOT``."""

    override = os.environ.get(MEMORY_PROJECT_ROOT_ENV)
    if override is None:
        return discover_memory_project_root()
    if not override.strip():
        raise ProjectRootError(f"{MEMORY_PROJECT_ROOT_ENV} must not be empty")
    root = pathlib.Path(override).expanduser()
    if not root.is_absolute():
        root = pathlib.Path.cwd() / root
    root = root.resolve()
    if not _looks_like_memory_project(root):
        raise ProjectRootError(
            f"{MEMORY_PROJECT_ROOT_ENV}={override!r} is not a memory_project root: "
            "expected openpi/pyproject.toml and openpi/src/openpi"
        )
    return root


def project_path(relative_path: str | pathlib.PurePath) -> pathlib.Path:
    """Resolve a confined project-relative path against ``memory_project``.

    Absolute paths and parent traversal are rejected so configuration cannot
    accidentally reintroduce a machine-local dependency outside the synced
    directory.
    """

    relative = pathlib.PurePath(relative_path)
    if relative.is_absolute():
        raise ProjectRootError(f"project path must be relative, got {str(relative)!r}")
    if ".." in relative.parts:
        raise ProjectRootError(f"project path must not escape memory_project, got {str(relative)!r}")
    root = memory_project_root()
    candidate = root.joinpath(*relative.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ProjectRootError(f"project path resolves outside memory_project: {str(relative)!r}") from exc
    return candidate


def project_relative_path(path: str | pathlib.Path) -> pathlib.PurePosixPath:
    """Return the canonical project-relative spelling of an in-project path."""

    candidate = pathlib.Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = pathlib.Path.cwd() / candidate
    candidate = candidate.resolve()
    root = memory_project_root()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ProjectRootError(f"path is outside memory_project: {candidate}") from exc
    return pathlib.PurePosixPath(*relative.parts)


def validate_executing_openpi_checkout() -> pathlib.Path:
    """Require the imported OpenPI package to come from this project copy.

    A foreign editable virtual environment can otherwise execute launch scripts from one
    checkout while importing ``openpi`` from another, defeating project-relative provenance.
    """

    root = memory_project_root()
    actual = pathlib.Path(__file__).resolve()
    expected = (root / OPENPI_DIR / "src/openpi/shared/project_paths.py").resolve()
    if actual != expected:
        raise ProjectRootError(
            "the active Python imports openpi from a different checkout: "
            f"expected {expected}, found {actual}; recreate the editable environment for this memory_project"
        )
    return root


def v35_runtime_environment() -> dict[str, str]:
    """Return the environment mapping that keeps v3.5 runtime state in-project."""

    return {
        MEMORY_PROJECT_ROOT_ENV: str(memory_project_root()),
        "UV_CACHE_DIR": str(project_path(UV_CACHE_DIR)),
        "HF_HOME": str(project_path(HUGGINGFACE_HOME)),
        "HF_LEROBOT_HOME": str(project_path(LEROBOT_HOME)),
        "HF_DATASETS_CACHE": str(project_path(HF_DATASETS_CACHE)),
        "OPENPI_DATA_HOME": str(project_path(OPENPI_DATA_HOME)),
        "OPENPI_JAX_CACHE_DIR": str(project_path(JAX_CACHE_DIR)),
        "TMPDIR": str(project_path(TMP_DIR)),
        "WANDB_DIR": str(project_path(WANDB_DIR)),
    }


def configure_v35_runtime_environment() -> dict[str, str]:
    """Install the portable v3.5 environment and create its writable directories.

    An inherited machine-local cache setting is treated as a configuration error instead of
    being silently overwritten.  This matters for preprocessing utilities as well as training:
    tokenizer or checkpoint downloads can otherwise escape the project before the model starts.
    """

    expected = v35_runtime_environment()
    scheduler_scratch = {"TMPDIR", "WANDB_DIR"}
    for name, expected_value in expected.items():
        current = os.environ.get(name)
        if (
            name not in scheduler_scratch
            and current is not None
            and pathlib.Path(current).expanduser().resolve() != pathlib.Path(expected_value).resolve()
        ):
            raise ProjectRootError(
                f"v3.5 portable runtime path mismatch for {name}: expected {expected_value}, found {current}"
            )
        os.environ[name] = expected_value
    for relative in (
        V35_ASSETS_ROOT,
        V35_CHECKPOINTS_DIR,
        V35_DIAGNOSTICS_DIR,
        CACHE_DIR,
        UV_CACHE_DIR,
        TMP_DIR,
        WANDB_DIR,
    ):
        project_path(relative).mkdir(parents=True, exist_ok=True)
    return expected
