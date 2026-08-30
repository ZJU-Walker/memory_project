from pathlib import Path
import sys

import pytest
import tyro

# v34_retention_eval intentionally uses script-local imports when invoked as
# ``python scripts/v34_retention_eval.py``. Mirror that entry-point import path here.
_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v34_retention_eval as retention
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


def _paths() -> dict[str, Path]:
    return {
        "checkpoint": Path("checkpoint"),
        "dataset_root": Path("dataset"),
        "output_dir": Path("output"),
    }


def test_retention_config_is_required_for_direct_construction() -> None:
    with pytest.raises(ValueError, match="--config is required"):
        retention.Args(**_paths())

    args = retention.Args(**_paths(), config="pi05_yam_mem_v34_run5_eta0")
    assert args.config == "pi05_yam_mem_v34_run5_eta0"


def test_retention_cli_help_marks_config_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        tyro.cli(retention.Args, args=["--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--config STR" in help_text
    assert "(required)" in help_text


def test_retention_cli_rejects_missing_config_before_main(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        tyro.cli(
            retention.Args,
            args=[
                "--checkpoint",
                "checkpoint",
                "--dataset-root",
                "dataset",
                "--output-dir",
                "output",
            ],
        )

    assert exit_info.value.code == 2
    assert "--config" in capsys.readouterr().err
