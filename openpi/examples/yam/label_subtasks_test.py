"""Tests for the YAM subtask labeler's label-file logic.

These cover the contract the converter depends on: segments tile the episode contiguously
from frame 0, and a saved file is only "complete" when it covers every frame.
"""

import importlib.util
import json
import pathlib
import sys

import pytest

_HERE = pathlib.Path(__file__).parent


def _load(module_name: str):
    """Import a sibling script by path (``examples/`` is not an importable package)."""
    spec = importlib.util.spec_from_file_location(module_name, _HERE / f"{module_name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


lab = _load("label_subtasks")

SUBTASKS = ["observe bins", "open left bin", "open right bin"]


def _load_converter():
    """The converter needs lerobot/cv2; skip rather than fail where they are absent."""
    try:
        return _load("convert_yam_data_to_lerobot")
    except ImportError as e:
        pytest.skip(f"converter not importable: {e}")
        raise


@pytest.fixture
def demo(tmp_path: pathlib.Path) -> pathlib.Path:
    d = tmp_path / "demo1"
    d.mkdir()
    return d


def test_natural_demo_order_is_numeric(tmp_path: pathlib.Path):
    for name in ("demo1", "demo2", "demo10", "demo20", "notademo"):
        (tmp_path / name).mkdir()
    assert [p.name for p in lab.find_demos(tmp_path)] == ["demo1", "demo2", "demo10", "demo20"]


def test_contiguous_full_coverage_has_no_problems():
    segments = [
        {"task": "observe bins", "start": 0, "end": 490},
        {"task": "open left bin", "start": 491, "end": 939},
    ]
    assert lab.validate_segments(segments, 940, SUBTASKS) == []


def test_gap_between_segments_is_reported():
    segments = [
        {"task": "observe bins", "start": 0, "end": 100},
        {"task": "open left bin", "start": 105, "end": 939},
    ]
    problems = lab.validate_segments(segments, 940, SUBTASKS)
    assert len(problems) == 1
    assert "expected 101" in problems[0]


def test_uncovered_tail_is_reported():
    segments = [{"task": "observe bins", "start": 0, "end": 500}]
    problems = lab.validate_segments(segments, 940, SUBTASKS)
    assert problems == ["episode not covered: labeled 501 of 940 frames"]


def test_unknown_subtask_is_reported():
    segments = [{"task": "wander around", "start": 0, "end": 939}]
    problems = lab.validate_segments(segments, 940, SUBTASKS)
    assert any("unknown subtask" in p for p in problems)


def test_save_roundtrip_marks_complete(demo: pathlib.Path):
    segments = [
        {"task": "observe bins", "start": 0, "end": 490},
        {"task": "open right bin", "start": 491, "end": 939},
    ]
    result = lab.save_segments(demo, segments, 940, SUBTASKS)
    assert result["complete"]
    assert not result["problems"]
    assert json.loads((demo / lab.LABEL_FILE).read_text()) == segments
    assert lab.load_segments(demo) == segments


def test_partial_save_is_written_but_not_complete(demo: pathlib.Path):
    segments = [{"task": "observe bins", "start": 0, "end": 100}]
    result = lab.save_segments(demo, segments, 940, SUBTASKS)
    assert not result["complete"]
    assert (demo / lab.LABEL_FILE).exists()  # partial progress is still persisted


def test_saving_empty_segments_removes_the_label_file(demo: pathlib.Path):
    (demo / lab.LABEL_FILE).write_text("[]")
    result = lab.save_segments(demo, [], 940, SUBTASKS)
    assert result["ok"]
    assert not (demo / lab.LABEL_FILE).exists()


def _memory_segments(side: str, num_frames: int = 940) -> list[dict]:
    """A well-formed five-phase episode for the given side."""
    labels = lab.memory_subtasks(side)
    bounds = [(0, 199), (200, 399), (400, 599), (600, 799), (800, num_frames - 1)]
    return [
        {"task": task, "start": s, "end": e} for task, (s, e) in zip(labels, bounds, strict=True)
    ]


def test_memory_vocabulary_is_seven_labels_five_per_episode():
    assert lab.all_memory_subtasks() == [
        "open both lids",
        "inspect both bins",
        "close both lids and reset arms",
        "wait; target bin is left",
        "wait; target bin is right",
        "open left bin",
        "open right bin",
    ]
    assert lab.memory_subtasks("left") == [
        "open both lids",
        "inspect both bins",
        "close both lids and reset arms",
        "wait; target bin is left",
        "open left bin",
    ]


@pytest.mark.parametrize("side", ["left", "right"])
def test_well_formed_memory_episode_validates(side: str):
    segments = _memory_segments(side)
    assert lab.validate_segments(segments, 940, lab.all_memory_subtasks()) == []


def test_mixed_side_episode_is_rejected():
    """wait-left followed by open-right corrupts the memory supervision."""
    segments = _memory_segments("left")
    segments[4] = {**segments[4], "task": "open right bin"}
    problems = lab.validate_segments(segments, 940, lab.all_memory_subtasks())
    assert any("mixes sides" in p for p in problems)


def test_out_of_order_phases_are_rejected():
    segments = _memory_segments("left")
    segments[1], segments[2] = (
        {**segments[2], "start": segments[1]["start"], "end": segments[1]["end"]},
        {**segments[1], "start": segments[2]["start"], "end": segments[2]["end"]},
    )
    problems = lab.validate_segments(segments, 940, lab.all_memory_subtasks())
    assert any("backwards in phase order" in p for p in problems)


def test_side_inference_and_phase_rules_ignore_custom_vocabularies():
    """A non-memory label set must not trip the phase checks."""
    segments = [{"task": "observe bins", "start": 0, "end": 939}]
    assert lab.validate_phase_schema(segments) == []
    assert lab.LabelerState.side_of_segments(segments) is None
    assert lab.LabelerState.side_of_segments(_memory_segments("right")) == "right"


def test_memory_labels_survive_the_converter(demo: pathlib.Path):
    convert = _load_converter()
    num_frames = 940
    lab.save_segments(demo, _memory_segments("right", num_frames), num_frames, lab.all_memory_subtasks())
    frame_tasks = convert._load_frame_subtasks(demo, num_frames)  # noqa: SLF001 - contract under test
    assert frame_tasks is not None
    assert frame_tasks[600] == "wait; target bin is right"
    assert frame_tasks[800] == "open right bin"
    assert len(set(frame_tasks)) == 5


def test_saved_labels_are_accepted_by_the_converter_loader(demo: pathlib.Path):
    """The whole point of this tool: the converter must accept what it writes."""
    convert = _load_converter()
    num_frames = 940
    segments = [
        {"task": "observe bins", "start": 0, "end": 490},
        {"task": "open left bin", "start": 491, "end": num_frames - 1},
    ]
    lab.save_segments(demo, segments, num_frames, SUBTASKS)
    frame_tasks = convert._load_frame_subtasks(demo, num_frames)  # noqa: SLF001 - that is the contract under test
    assert frame_tasks is not None
    assert len(frame_tasks) == num_frames
    assert frame_tasks[0] == "observe bins"
    assert frame_tasks[490] == "observe bins"
    assert frame_tasks[491] == "open left bin"
    assert frame_tasks[-1] == "open left bin"
