# ruff: noqa: SLF001

import pytest

from scripts import generate_0816_v35_staticwait_overlays as generator


def _canonical(side: str = "left") -> list[dict[str, object]]:
    return [
        {"task": "open both lids", "start": 0, "end": 9},
        {"task": "inspect both bins", "start": 10, "end": 19},
        {"task": "close both lids and reset arms", "start": 20, "end": 29},
        {"task": f"wait; target bin is {side}", "start": 30, "end": 49},
        {"task": f"open {side} bin", "start": 50, "end": 69},
    ]


def test_build_overlay_moves_only_three_phase_boundaries() -> None:
    canonical = _canonical()

    overlay = generator._build_overlay(canonical, 34, 44, context="test")

    assert canonical[2]["end"] == 29
    assert canonical[3] == {"task": "wait; target bin is left", "start": 30, "end": 49}
    assert overlay[2]["end"] == 33
    assert overlay[3] == {"task": "wait; target bin is left", "start": 34, "end": 44}
    assert overlay[4]["start"] == 45
    assert [segment["task"] for segment in overlay] == [segment["task"] for segment in canonical]
    assert overlay[0]["start"] == 0
    assert overlay[-1]["end"] == 69


def test_side_mismatch_fails_closed() -> None:
    labels = _canonical()
    labels[-1]["task"] = "open right bin"

    with pytest.raises(ValueError, match="does not match"):
        generator._validate_five_phase_labels(labels, context="test")


def test_core_outside_canonical_wait_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside canonical wait"):
        generator._build_overlay(_canonical(), 29, 44, context="test")
