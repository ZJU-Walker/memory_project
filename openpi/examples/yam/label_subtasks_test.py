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


def test_review_session_can_hide_explicitly_excluded_demo(tmp_path: pathlib.Path):
    (tmp_path / "demo1").mkdir()
    (tmp_path / "demo14").mkdir()
    state = lab.LabelerState(
        data_dir=tmp_path,
        subtasks=SUBTASKS,
        excluded_demos=frozenset({"demo14"}),
    )
    assert [demo["name"] for demo in state.demo_summaries()] == ["demo1"]


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


def test_versioned_label_overlay_does_not_replace_canonical(demo: pathlib.Path):
    canonical = [{"task": "observe bins", "start": 0, "end": 9}]
    overlay = [{"task": "open left bin", "start": 0, "end": 9}]
    (demo / lab.LABEL_FILE).write_text(json.dumps(canonical))

    result = lab.save_segments(demo, overlay, 10, SUBTASKS, "subtask_labels_v35.json")

    assert result["complete"]
    assert lab.load_segments(demo) == canonical
    assert lab.load_segments(demo, "subtask_labels_v35.json") == overlay


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
    return [{"task": task, "start": s, "end": e} for task, (s, e) in zip(labels, bounds, strict=True)]


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


@pytest.mark.parametrize(
    ("vocabulary", "message"),
    [
        ([], "at least one task"),
        (["open both bins", "   "], "nonempty"),
        (["open both bins", "open both bins"], "unique"),
    ],
)
def test_converter_rejects_invalid_task_vocabulary(vocabulary: list[str], message: str):
    convert = _load_converter()
    with pytest.raises(ValueError, match=message):
        convert._validate_task_vocabulary(vocabulary)  # noqa: SLF001 - CLI validation contract


def test_converter_preregisters_task_vocabulary_before_saving(
    demo: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    convert = _load_converter()
    vocabulary = ["open both bins", "inspect both bins", "close and reset"]
    (demo / "subtask_labels.json").write_text(json.dumps([{"task": "inspect both bins", "start": 0, "end": 0}]))
    events = []

    class FakeMeta:
        def add_task(self, task: str) -> None:
            events.append(("add_task", task))

    class FakeDataset:
        def __init__(self):
            self.meta = FakeMeta()
            self.num_episodes = 0

        def add_frame(self, frame: dict) -> None:
            events.append(("add_frame", frame["task"]))

        def save_episode(self) -> None:
            events.append(("save_episode", None))
            self.num_episodes += 1

    dataset = FakeDataset()
    monkeypatch.setattr(convert.LeRobotDataset, "create", lambda **_: dataset)
    monkeypatch.setattr(convert, "HF_LEROBOT_HOME", tmp_path / "lerobot")
    monkeypatch.setattr(convert.np, "load", lambda _: convert.np.zeros((1, 7), dtype=convert.np.float32))
    monkeypatch.setattr(
        convert,
        "_read_video_frames",
        lambda _: [convert.np.zeros(convert.IMG_SHAPE, dtype=convert.np.uint8)],
    )

    convert.main(data_dir=str(demo.parent), repo_name="yam/test", task_vocabulary=vocabulary)

    assert events[:3] == [("add_task", task) for task in vocabulary]
    assert events[3:] == [("add_frame", "inspect both bins"), ("save_episode", None)]


def test_converter_rejects_unknown_raw_task_before_creating_dataset(
    demo: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    convert = _load_converter()
    (demo / "subtask_labels.json").write_text(
        json.dumps(
            [
                {"task": "open both bins", "start": 0, "end": 9},
                {"task": "typo outside converted frames", "start": 10, "end": 19},
            ]
        )
    )

    def fail_create(**_):
        raise AssertionError("dataset creation must happen after raw-label validation")

    monkeypatch.setattr(convert.LeRobotDataset, "create", fail_create)
    with pytest.raises(ValueError, match="typo outside converted frames"):
        convert.main(data_dir=str(demo.parent), task_vocabulary=["open both bins"])


def test_converter_omitted_task_vocabulary_is_a_noop():
    convert = _load_converter()
    assert convert._validate_task_vocabulary(None) is None  # noqa: SLF001 - CLI validation contract


def _write_manifest(path: pathlib.Path, demo: pathlib.Path, *, target_side: str = "left") -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "raw_root": ".",
                "task_vocabulary": ["wait; target bin is left", "open left bin"],
                "expected": {
                    "included_episodes": 1,
                    "included_frames": 2,
                    "episode_index_to_stable_id": {"0": "collection/demo1"},
                },
                "episodes": [
                    {
                        "stable_id": "collection/demo1",
                        "raw_dir": demo.name,
                        "include": True,
                        "instruction": "find the banana",
                        "target_side": target_side,
                        "expected_num_frames": 2,
                    },
                    {
                        "stable_id": "collection/demo2",
                        "raw_dir": "demo2",
                        "include": False,
                        "exclude_reason": "intentionally incomplete",
                    },
                ],
            }
        )
    )


def test_converter_manifest_preflight_is_ordered_and_fail_closed(
    demo: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    convert = _load_converter()
    labels = [
        {"task": "wait; target bin is left", "start": 0, "end": 0},
        {"task": "open left bin", "start": 1, "end": 1},
    ]
    (demo / "subtask_labels.json").write_text(json.dumps(labels))
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, demo)

    payload, episodes = convert._load_episode_manifest(manifest)  # noqa: SLF001
    monkeypatch.setattr(
        convert,
        "_episode_stream_lengths",
        lambda _: {"test_stream": 2},
    )
    checked, total = convert._preflight_manifest(  # noqa: SLF001
        manifest, payload, episodes, ("wait; target bin is left", "open left bin")
    )

    assert total == 2
    assert [episode["stable_id"] for episode in checked] == ["collection/demo1"]
    assert checked[0]["raw_dir"] == demo.resolve()
    assert checked[0]["num_frames"] == 2


def test_converter_manifest_rejects_target_side_label_mismatch(
    demo: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    convert = _load_converter()
    (demo / "subtask_labels.json").write_text(
        json.dumps(
            [
                {"task": "wait; target bin is left", "start": 0, "end": 0},
                {"task": "open left bin", "start": 1, "end": 1},
            ]
        )
    )
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, demo, target_side="right")
    payload, episodes = convert._load_episode_manifest(manifest)  # noqa: SLF001
    monkeypatch.setattr(
        convert,
        "_episode_stream_lengths",
        lambda _: {"test_stream": 2},
    )

    with pytest.raises(ValueError, match="disagree with target_side=right"):
        convert._preflight_manifest(  # noqa: SLF001
            manifest,
            payload,
            episodes,
            ("wait; target bin is left", "open left bin"),
        )


def test_converter_manifest_writes_prompt_and_source_provenance(
    demo: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    convert = _load_converter()
    labels = [
        {"task": "wait; target bin is left", "start": 0, "end": 0},
        {"task": "open left bin", "start": 1, "end": 1},
    ]
    (demo / "subtask_labels.json").write_text(json.dumps(labels))
    manifest = tmp_path / "manifest.json"
    _write_manifest(manifest, demo)
    stream_lengths = {"test_stream": 2}
    monkeypatch.setattr(convert, "_episode_stream_lengths", lambda _: stream_lengths)
    monkeypatch.setattr(convert, "HF_LEROBOT_HOME", tmp_path / "lerobot")
    monkeypatch.setattr(convert.np, "load", lambda *_, **__: convert.np.zeros((2, 7), dtype=convert.np.float32))
    monkeypatch.setattr(
        convert,
        "_read_video_frames",
        lambda _: [convert.np.zeros(convert.IMG_SHAPE, dtype=convert.np.uint8)] * 2,
    )

    class FakeMeta:
        def add_task(self, _: str) -> None:
            pass

    class FakeDataset:
        def __init__(self):
            self.meta = FakeMeta()
            self.num_episodes = 0
            self.frames = []

        def add_frame(self, frame: dict) -> None:
            self.frames.append(frame)

        def save_episode(self) -> None:
            self.num_episodes += 1

    dataset = FakeDataset()
    monkeypatch.setattr(convert.LeRobotDataset, "create", lambda **_: dataset)

    convert.main(episode_manifest=str(manifest), repo_name="yam/test-manifest")

    meta = tmp_path / "lerobot" / "yam" / "test-manifest" / "meta"
    assert json.loads((meta / "episode_prompts.json").read_text()) == {"0": "find the banana"}
    sources = json.loads((meta / "episode_sources.json").read_text())
    assert sources["0"]["stable_id"] == "collection/demo1"
    assert sources["0"]["target_side"] == "left"
    assert sources["0"]["num_frames"] == 2
    assert dataset.num_episodes == 1
    assert len(dataset.frames) == 2
