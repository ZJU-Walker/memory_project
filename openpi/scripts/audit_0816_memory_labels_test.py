# ruff: noqa: SLF001

import json
from pathlib import Path

import numpy as np

from scripts import audit_0816_memory_labels as audit


def test_longest_static_run_uses_strict_step_and_bounded_excursion() -> None:
    window = np.asarray([[0.0], [0.003], [0.006], [0.009], [0.012], [0.015], [0.018], [0.021], [0.024]])

    assert audit._longest_static_run(window) == (2, 8)

    exact_threshold_step = np.asarray([[0.0], [audit.MAX_SPEED], [audit.MAX_SPEED]])
    assert audit._longest_static_run(exact_threshold_step) is None


def test_episode_ordering_is_fixed_30_plus_30() -> None:
    specs = audit._episode_specs(Path("/raw"))

    assert len(specs) == 60
    assert specs[0]["demo_dir"] == Path("/raw/0816_banana/demo1")
    assert specs[29]["demo_dir"] == Path("/raw/0816_banana/demo30")
    assert specs[30]["demo_dir"] == Path("/raw/0816_grey_box/demo1")
    assert specs[59]["demo_dir"] == Path("/raw/0816_grey_box/demo30")


def test_algorithm_self_checks() -> None:
    assert len(audit._run_algorithm_self_checks()) == 3


def test_ep26_fix_and_backup_are_semantically_pinned() -> None:
    demo_dir = audit.DEFAULT_DATA_ROOT / "0816_banana" / "demo27"
    current = json.loads((demo_dir / "subtask_labels.json").read_text())

    result = audit._semantic_fix(26, current, demo_dir)

    assert result is not None
    assert result["status"] == "applied_with_backup"
    assert result["current_file_sha256"] == audit.EXPECTED_EP26_CURRENT_FILE_SHA256
    assert result["backup_file_sha256"] == audit.EXPECTED_EP26_PRE_FIX_BACKUP_FILE_SHA256
    assert result["backup_semantic_sha256"] == audit.EXPECTED_EP26_PRE_FIX_SEMANTIC_SHA256
    assert result["backup_segments"][-1]["task"] == "wait; target bin is right"
    assert result["current_segments"][-1]["task"] == "open right bin"
