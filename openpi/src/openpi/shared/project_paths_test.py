import os
import pathlib

import pytest

import openpi.shared.project_paths as project_paths


def _project_fixture(root: pathlib.Path) -> pathlib.Path:
    (root / "openpi" / "src" / "openpi").mkdir(parents=True)
    (root / "openpi" / "pyproject.toml").touch()
    return root


def test_source_discovery_is_independent_of_current_working_directory(tmp_path: pathlib.Path) -> None:
    root = _project_fixture(tmp_path / "copy" / "memory_project")
    source = root / "openpi" / "src" / "openpi" / "shared" / "project_paths.py"
    source.parent.mkdir(parents=True)
    source.touch()

    assert project_paths.discover_memory_project_root(source_file=source, cwd=tmp_path) == root


def test_environment_override_relocates_every_v35_local_path(monkeypatch, tmp_path: pathlib.Path) -> None:
    root = _project_fixture(tmp_path / "other_cluster" / "memory_project")
    monkeypatch.setenv(project_paths.MEMORY_PROJECT_ROOT_ENV, str(root))

    assert project_paths.project_path(project_paths.V35_DATASET_DIR) == (
        root / "data/lerobot/yam/bin_memory_0830_0831_v36_subtask"
    )
    assert project_paths.project_path(project_paths.V35_FROZEN_MANIFEST) == (
        root / "data/0830_0831_episode_manifest_v36_frozen.json"
    )
    assert project_paths.project_path(project_paths.V35_ASSETS_DIR) == root / "v35/assets/pi05_yam_0830_0831_v36"
    assert project_paths.project_path(project_paths.V35_CHECKPOINTS_DIR) == root / "v35/checkpoints"
    assert project_paths.project_path(project_paths.V35_DIAGNOSTICS_DIR) == root / "v35/diagnostics"

    environment = project_paths.v35_runtime_environment()
    assert environment == {
        "MEMORY_PROJECT_ROOT": str(root),
        "UV_CACHE_DIR": str(root / "v35/cache/uv"),
        "HF_HOME": str(root / "v35/cache/huggingface"),
        "HF_LEROBOT_HOME": str(root / "data/lerobot"),
        "HF_DATASETS_CACHE": str(root / "v35/cache/huggingface/datasets"),
        "OPENPI_DATA_HOME": str(root / "v35/cache/openpi"),
        "OPENPI_JAX_CACHE_DIR": str(root / "v35/cache/jax"),
        "TMPDIR": str(root / "v35/tmp"),
        "WANDB_DIR": str(root / "v35/wandb"),
    }


@pytest.mark.parametrize("path", ["../foreign", "data/../../foreign", "/iris/u/user/data"])
def test_project_path_rejects_escape_and_absolute_paths(monkeypatch, tmp_path: pathlib.Path, path: str) -> None:
    root = _project_fixture(tmp_path / "memory_project")
    monkeypatch.setenv(project_paths.MEMORY_PROJECT_ROOT_ENV, str(root))

    with pytest.raises(project_paths.ProjectRootError):
        project_paths.project_path(path)


def test_project_path_rejects_symlink_escape(monkeypatch, tmp_path: pathlib.Path) -> None:
    root = _project_fixture(tmp_path / "memory_project")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (root / "escape").symlink_to(foreign, target_is_directory=True)
    monkeypatch.setenv(project_paths.MEMORY_PROJECT_ROOT_ENV, str(root))

    with pytest.raises(project_paths.ProjectRootError, match="resolves outside memory_project"):
        project_paths.project_path("escape/artifact.json")


def test_project_relative_path_round_trips_and_rejects_outside(monkeypatch, tmp_path: pathlib.Path) -> None:
    root = _project_fixture(tmp_path / "memory_project")
    monkeypatch.setenv(project_paths.MEMORY_PROJECT_ROOT_ENV, str(root))
    absolute = root / "v35/assets/norm_stats.json"

    relative = project_paths.project_relative_path(absolute)

    assert relative == pathlib.PurePosixPath("v35/assets/norm_stats.json")
    assert project_paths.project_path(relative) == absolute
    with pytest.raises(project_paths.ProjectRootError, match="outside memory_project"):
        project_paths.project_relative_path(tmp_path / "other_cluster/file.json")


def test_invalid_override_fails_closed(monkeypatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.setenv(project_paths.MEMORY_PROJECT_ROOT_ENV, str(tmp_path / "not_a_project"))

    with pytest.raises(project_paths.ProjectRootError, match="not a memory_project root"):
        project_paths.memory_project_root()


def test_executing_checkout_rejects_foreign_editable_environment(monkeypatch, tmp_path: pathlib.Path) -> None:
    root = _project_fixture(tmp_path / "memory_project")
    monkeypatch.setenv(project_paths.MEMORY_PROJECT_ROOT_ENV, str(root))

    with pytest.raises(project_paths.ProjectRootError, match="different checkout"):
        project_paths.validate_executing_openpi_checkout()


def test_configure_runtime_creates_only_project_local_directories(monkeypatch, tmp_path: pathlib.Path) -> None:
    root = _project_fixture(tmp_path / "memory_project")
    monkeypatch.setenv(project_paths.MEMORY_PROJECT_ROOT_ENV, str(root))
    expected = project_paths.v35_runtime_environment()
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "scheduler-tmp"))

    environment = project_paths.configure_v35_runtime_environment()

    assert environment == expected
    assert all(os.environ.get(name) == value for name, value in environment.items())
    for relative in (
        project_paths.V35_ASSETS_ROOT,
        project_paths.V35_CHECKPOINTS_DIR,
        project_paths.V35_DIAGNOSTICS_DIR,
        project_paths.CACHE_DIR,
        project_paths.UV_CACHE_DIR,
        project_paths.TMP_DIR,
        project_paths.WANDB_DIR,
    ):
        assert project_paths.project_path(relative).is_dir()


def test_configure_runtime_rejects_inherited_foreign_cache(monkeypatch, tmp_path: pathlib.Path) -> None:
    root = _project_fixture(tmp_path / "memory_project")
    monkeypatch.setenv(project_paths.MEMORY_PROJECT_ROOT_ENV, str(root))
    for name, value in project_paths.v35_runtime_environment().items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OPENPI_DATA_HOME", str(tmp_path / "foreign-cache"))

    with pytest.raises(project_paths.ProjectRootError, match="OPENPI_DATA_HOME"):
        project_paths.configure_v35_runtime_environment()


def test_checked_in_contract_values_are_relative() -> None:
    for path in (
        project_paths.OPENPI_DIR,
        project_paths.DATA_DIR,
        project_paths.LEROBOT_HOME,
        project_paths.V35_ROOT,
        project_paths.CACHE_DIR,
        project_paths.UV_CACHE_DIR,
        project_paths.V35_DATASET_DIR,
        project_paths.V35_FROZEN_MANIFEST,
        project_paths.V35_ASSETS_DIR,
        project_paths.V35_CHECKPOINTS_DIR,
        project_paths.V35_DIAGNOSTICS_DIR,
        project_paths.WANDB_DIR,
    ):
        assert not path.is_absolute()
        assert ".." not in path.parts
