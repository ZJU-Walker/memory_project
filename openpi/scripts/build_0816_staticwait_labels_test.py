import pytest

from scripts import build_0816_staticwait_labels as build


def _canonical(side: str = "left") -> list[dict]:
    return [
        {"task": "open both lids", "start": 0, "end": 9},
        {"task": "inspect both bins", "start": 10, "end": 19},
        {"task": "close both lids and reset arms", "start": 20, "end": 29},
        {"task": f"wait; target bin is {side}", "start": 30, "end": 49},
        {"task": f"open {side} bin", "start": 50, "end": 69},
    ]


def test_build_overlay_reassigns_wait_edges_without_gaps():
    overlay = build.build_overlay(_canonical(), 34, 43)
    assert overlay[2]["end"] == 33
    assert overlay[3] == {"task": "wait; target bin is left", "start": 34, "end": 43}
    assert overlay[4]["start"] == 44
    assert [segment["start"] for segment in overlay] == [0, 10, 20, 34, 44]
    assert [segment["end"] for segment in overlay] == [9, 19, 33, 43, 69]


def test_build_overlay_rejects_mixed_side_schema():
    canonical = _canonical()
    canonical[-1]["task"] = "open right bin"
    with pytest.raises(ValueError, match="exact five-phase"):
        build.build_overlay(canonical, 34, 43)
