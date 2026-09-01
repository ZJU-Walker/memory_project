"""Revision-4 tests for v3.5 E/O/D masks, sparse clock, and manifest sampling."""

# ruff: noqa: SLF001, I001 - private loader helpers are the protocol under test; import torch
# backed data_loader before config, matching the established loader test ordering.

import hashlib
import json
import pathlib
import random
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

import openpi.training.data_loader as _data_loader
import openpi.training.config as _config
import openpi.transforms as _transforms

OPEN = "open both lids"
INSPECT = "inspect both bins"
CLOSE = "close both lids and reset arms"
WAIT_L = "wait; target bin is left"
WAIT_R = "wait; target bin is right"
EXEC_L = "open left bin"
EXEC_R = "open right bin"
VOCAB = {0: OPEN, 1: INSPECT, 2: CLOSE, 3: WAIT_L, 4: WAIT_R, 5: EXEC_L, 6: EXEC_R}
IDS = {value: key for key, value in VOCAB.items()}


class _RandomHostTransformDataset:
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return {
            "sample": np.asarray(
                [
                    int(index),
                    np.random.randint(0, 2**24),
                    random.randrange(2**24),
                    int(torch.randint(0, 2**24, ()).item()),
                ],
                dtype=np.int64,
            )
        }


def _exact_resume_test_loader(seed: int) -> _data_loader.TorchDataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = _data_loader.SequenceBucketBatchSampler(
        np.ones(12, dtype=np.float64),
        np.full(12, 4, dtype=np.int32),
        (4,),
        2,
        generator=generator,
        num_samples=12,
    )
    return _data_loader.TorchDataLoader(
        _RandomHostTransformDataset(),
        local_batch_size=2,
        batch_sampler=sampler,
        num_workers=0,
        seed=seed,
        framework="pytorch",
        exact_resume=True,
    )


def test_exact_resume_next_batch_and_all_host_rng_states_are_byte_identical():
    old_numpy = np.random.get_state()
    old_python = random.getstate()
    old_torch = torch.random.get_rng_state()
    try:
        np.random.seed(3501)
        random.seed(3502)
        torch.manual_seed(3503)
        uninterrupted_loader = _exact_resume_test_loader(3504)
        uninterrupted = iter(uninterrupted_loader)
        next(uninterrupted)
        next(uninterrupted)
        checkpoint_state = uninterrupted_loader.state_dict()
        # The state must be portable canonical-JSON input, not a pickle or process object.
        json.dumps(checkpoint_state, sort_keys=True, separators=(",", ":"), allow_nan=False)

        expected_next = next(uninterrupted)["sample"]
        expected_after = uninterrupted_loader.state_dict()

        resumed_loader = _exact_resume_test_loader(9999)
        resumed_loader.load_state_dict(checkpoint_state)
        actual_next = next(iter(resumed_loader))["sample"]

        torch.testing.assert_close(actual_next, expected_next, rtol=0, atol=0)
        assert resumed_loader.state_dict() == expected_after
    finally:
        np.random.set_state(old_numpy)
        random.setstate(old_python)
        torch.random.set_rng_state(old_torch)


def test_exact_resume_fails_closed_with_workers_or_non_stateful_sampler():
    dataset = _RandomHostTransformDataset()
    with pytest.raises(ValueError, match="num_workers=0"):
        _data_loader.TorchDataLoader(
            dataset,
            local_batch_size=2,
            num_workers=1,
            framework="pytorch",
            exact_resume=True,
        )
    with pytest.raises(ValueError, match="SequenceBucketBatchSampler"):
        _data_loader.TorchDataLoader(
            dataset,
            local_batch_size=2,
            num_workers=0,
            framework="pytorch",
            exact_resume=True,
        )


def test_project_local_lerobot_root_is_passed_to_metadata_and_dataset(monkeypatch, tmp_path: pathlib.Path):
    calls = {}

    class _Metadata:
        fps = 30
        tasks: ClassVar[dict] = {}

        def __init__(self, repo_id, *, root=None):
            calls["metadata"] = (repo_id, root)

    class _Dataset:
        def __init__(self, repo_id, *, root=None, delta_timestamps=None):
            calls["dataset"] = (repo_id, root, delta_timestamps)

    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDatasetMetadata", _Metadata)
    monkeypatch.setattr(_data_loader.lerobot_dataset, "LeRobotDataset", _Dataset)
    data_config = _config.DataConfig(
        repo_id="yam/portable",
        lerobot_dataset_root=str(tmp_path / "data/lerobot/yam/portable"),
    )

    result = _data_loader.create_torch_dataset(
        data_config,
        action_horizon=3,
        model_config=SimpleNamespace(predict_with_memory=False),
    )

    assert isinstance(result, _Dataset)
    expected_root = str(tmp_path / "data/lerobot/yam/portable")
    assert calls["metadata"] == ("yam/portable", expected_root)
    assert calls["dataset"] == (
        "yam/portable",
        expected_root,
        {"actions": [0.0, 1 / 30, 2 / 30]},
    )


def _window(
    *,
    start_lo=1,
    evidence_start=30,
    decision_lo=100,
    decision_hi=129,
    final_e_limit=54,
    occlusion_lo=60,
    occlusion_hi=99,
    execute_start=130,
    episode=0,
    collection=0,
    object_id=0,
    cell=0,
):
    return np.asarray(
        [
            start_lo,
            evidence_start,
            decision_lo,
            decision_hi,
            final_e_limit,
            occlusion_lo,
            occlusion_hi,
            execute_start,
            episode,
            collection,
            object_id,
            cell,
            _transforms.V35_WINDOW_MARKER_VALUE,
        ],
        dtype=np.int32,
    )


def _raw_label(frame: int) -> str:
    if frame < 30:
        return OPEN
    if frame <= 59:
        return INSPECT
    if frame < 100:
        return CLOSE
    if frame <= 129:
        return WAIT_R
    return EXEC_R


def _sequence_item(*, frame=1, stride=15, steps=14, action_horizon=15, window=None):
    frames = frame + np.arange(steps) * stride
    now = [_raw_label(int(x)) for x in frames]
    # Deliberately unrelated look-ahead targets: v3.5 eligibility must use `subtask_now` only.
    shifted = [WAIT_R] * steps
    return {
        "frame_index": np.int64(frame),
        "index": np.int64(0),
        "episode_length": np.int32(180),
        "memory_window": _window() if window is None else window,
        "observation/image": np.zeros((steps, 2, 2, 3), dtype=np.uint8),
        "observation/left_wrist_image": np.zeros((steps, 2, 2, 3), dtype=np.uint8),
        "observation/right_wrist_image": np.zeros((steps, 2, 2, 3), dtype=np.uint8),
        "observation/state": np.zeros((steps, 14), dtype=np.float32),
        "actions": np.zeros((steps, action_horizon * 14), dtype=np.float32),
        "subtask": shifted,
        "subtask_now": now,
        "subtask_valid": np.ones(steps, dtype=bool),
        "subtask_now_valid": np.ones(steps, dtype=bool),
        "episode_side": np.int32(1),
    }


def _labels():
    return _transforms.MemoryV34Labels(
        subtask_vocab=tuple(VOCAB.values()),
        evidence_subtasks=(INSPECT,),
        memory_required_subtasks=(WAIT_L, WAIT_R),
    )


def test_sparse_masks_use_current_frame_tail_guard_read_before_write_and_padding():
    build = _transforms.BuildMemorySequence(stride=15, action_horizon=15, block_steps=4, occlusion_subtasks=(CLOSE,))
    out = _labels()(build(_sequence_item()))

    assert bool(out["seq_sparse_skip_o"])
    np.testing.assert_array_equal(out["seq_step_mask"][:6], [True, True, True, True, False, False])
    # Raw frames kept are E=31,46 and D=106,121.  The final five raw E frames are ineligible
    # even though their current semantic label remains inspect.
    np.testing.assert_array_equal(out["seq_write_mask"][:6], [True, True, False, False, False, False])
    np.testing.assert_array_equal(out["seq_decision_mask"][:6], [False, False, True, True, False, False])
    # Read-before-write: the first E read is invalid, the second sees the first commit, and D
    # sees the final eligible E commit.  No TBPTT fence exists in a critical window.
    np.testing.assert_array_equal(out["seq_read_state_valid"][:6], [False, True, True, True, False, False])
    np.testing.assert_array_equal(out["seq_read_credit_reachable"][:6], [False, True, True, True, False, False])
    # Five omitted valid non-E transitions occur before the first D read; padding carries zero.
    np.testing.assert_array_equal(out["seq_decay_gap_before"][:6], [0, 0, 3, 0, 0, 0])
    # D@121 owns an action chunk [121,135] overlapping execute onset 130; D@106 does not.
    np.testing.assert_array_equal(out["seq_use_pressure_mask"][:6], [False, False, False, True, False, False])
    assert int(out["seq_episode_index"]) == 0
    assert int(out["seq_memory_cell"]) == 0


def test_strict_decision_validity_removes_moving_wait_without_rewriting_semantic_label():
    item = _sequence_item()
    # Raw D samples are dense indices 7 and 8; invalidate the latter in the strict 14D sidecar.
    item["subtask_now_valid"][8] = False
    out = _labels()(
        _transforms.BuildMemorySequence(stride=15, action_horizon=15, block_steps=4, occlusion_subtasks=(CLOSE,))(item)
    )
    np.testing.assert_array_equal(out["seq_waiting_mask"][:4], [False, False, True, False])
    np.testing.assert_array_equal(out["seq_decision_mask"][:4], [False, False, True, False])
    assert out["subtask"][3] == WAIT_R, "strict D validity must not rewrite the semantic subtask"
    assert not out["seq_use_pressure_mask"].any()


def test_state_valid_survives_tbptt_fence_while_credit_reach_does_not():
    data = {
        "subtask": [INSPECT, CLOSE, WAIT_L],
        "subtask_now": [INSPECT, CLOSE, WAIT_L],
        "subtask_valid": np.ones(3, dtype=bool),
        "subtask_now_valid": np.ones(3, dtype=bool),
        "episode_side": np.int32(0),
        "seq_step_mask": np.ones(3, dtype=bool),
        "seq_block_boundary": np.asarray([False, False, True]),
        "seq_decay_gap_before": np.zeros(3, dtype=np.int32),
        "_v35_enabled": np.ones((), dtype=bool),
        "_v35_write_tail_valid": np.ones(3, dtype=bool),
        "_v35_action_overlaps_execute": np.zeros(3, dtype=bool),
    }
    out = _labels()(data)
    np.testing.assert_array_equal(out["seq_read_state_valid"], [False, True, True])
    np.testing.assert_array_equal(out["seq_read_credit_reachable"], [False, True, False])
    assert out["seq_decision_mask"][2]
    assert out["seq_read_state_valid"][2]


def test_long_delay_sparse_layout_is_exact_and_natural_layout_is_dense():
    window = _window(
        start_lo=1,
        evidence_start=31,
        decision_lo=300,
        decision_hi=329,
        final_e_limit=55,
        occlusion_lo=56,
        occlusion_hi=299,
        execute_start=330,
    )
    sparse = _transforms.memory_critical_layout(1, window, stride=15, lookahead=0, num_steps=40)
    assert sparse.sparse_skip_o
    assert sparse.n_delay == 16
    assert sparse.sampled_e_count == 2
    np.testing.assert_array_equal(sparse.keep_indices, [2, 3, 20, 21])
    np.testing.assert_array_equal(sparse.decay_gap_before, [0, 0, 16, 0])

    natural = _transforms.memory_critical_layout(2, window, stride=15, lookahead=0, num_steps=40)
    assert not natural.sparse_skip_o
    assert natural.n_delay == 16
    assert natural.endpoint == 20
    assert len(natural.keep_indices) == 21
    assert not natural.decay_gap_before.any()


def test_sparse_layout_filters_tail_e_residue_and_non_o_gap_before_sampler_weighting():
    # start=13 samples E@43, then semantic-tail E@58 before O begins at 60.  It is a sparse
    # family residue, but that tail transition is not legal to omit analytically.
    with pytest.raises(ValueError, match="outside the O-only interval"):
        _transforms.memory_critical_layout(13, _window(), stride=15, lookahead=0, num_steps=14)

    # Even with a clean E anchor, an unlabeled/reset gap after O cannot be represented as
    # skip-O decay.  Here sampled frame 91 lies beyond the declared O interval and before D.
    with pytest.raises(ValueError, match="outside the O-only interval"):
        _transforms.memory_critical_layout(1, _window(occlusion_hi=80), stride=15, lookahead=0, num_steps=14)


def test_sparse_build_rejects_non_o_or_reset_label_inside_declared_o_bounds():
    item = _sequence_item()
    item["subtask_now"][5] = "memory reset"
    build = _transforms.BuildMemorySequence(stride=15, action_horizon=15, block_steps=4, occlusion_subtasks=(CLOSE,))
    with pytest.raises(ValueError, match="O-only and reset-free"):
        build(item)


class _StubHF:
    def __init__(self, cols):
        self._cols = cols

    def with_format(self, _fmt):
        return self._cols


class _StubLeRobot:
    def __init__(self, episodes, root):
        self.hf_dataset = _StubHF(
            {
                "task_index": [task for episode in episodes for task in episode],
                "episode_index": [index for index, episode in enumerate(episodes) for _ in episode],
                "state": np.concatenate([np.zeros((len(episode), 14), np.float32) for episode in episodes]),
            }
        )
        self.root = root


class _StubMeta:
    tasks = VOCAB


def _episode(side: str, *, wait_frames=30):
    wait = WAIT_L if side == "left" else WAIT_R
    execute = EXEC_L if side == "left" else EXEC_R
    phases = ((OPEN, 30), (INSPECT, 30), (CLOSE, 240), (wait, wait_frames), (execute, 30))
    return [IDS[label] for label, count in phases for _ in range(count)]


def _write_manifest(path, records, *, seed=36):
    path.write_text(json.dumps({"schema_version": 1, "split_seed": seed, "episodes": records}))
    return path


def _manifest_record(index, side, split, *, collection="0816", object_name="banana"):
    return {
        "episode_index": index,
        "stable_id": f"{collection}/demo{index + 1}",
        "collection": collection,
        "object": object_name,
        "target_side": side,
        "split": split,
        "include": True,
    }


def _v35_config(manifest, **overrides):
    values = {
        "memory_stride_frames": 15,
        "memory_slice_prob": 0.5,
        "memory_min_slice_steps": 2,
        "subtask_lookahead": 15,
        "evidence_subtasks": (INSPECT,),
        "memory_required_subtasks": (WAIT_L, WAIT_R),
        "memory_critical_prob": 0.5,
        "memory_critical_start_pad": 30,
        "memory_waiting_max_speed": 4e-3,
        "memory_waiting_max_excursion": 2e-2,
        "memory_v35_enabled": True,
        "memory_e_tail_guard_frames": 5,
        "memory_occlusion_subtasks": (CLOSE,),
        "memory_execute_subtasks": (EXEC_L, EXEC_R),
        "memory_sparse_skip_o_prob": 0.5,
        "memory_waiting_state_dim": 14,
        "memory_episode_manifest_path": str(manifest),
        "memory_manifest_split": "train",
        "memory_manifest_split_seed": 36,
    }
    values.update(overrides)
    return _config.DataConfig(**values)


def test_manifest_split_is_stable_and_sampler_balances_natural_skip_by_cell(tmp_path, monkeypatch):
    episodes = [_episode("left"), _episode("right"), _episode("left")]
    stub = _StubLeRobot(episodes, tmp_path)
    monkeypatch.setattr(_data_loader, "_unwrap_lerobot", lambda _dataset: stub)
    manifest = _write_manifest(
        tmp_path / "manifest.json",
        [
            _manifest_record(0, "left", "train"),
            _manifest_record(1, "right", "train"),
            _manifest_record(2, "left", "development", collection="0830_part1", object_name="greybox"),
        ],
    )
    config = _v35_config(manifest)
    info = _data_loader._episode_info_table(stub, _StubMeta(), config)
    np.testing.assert_array_equal(info["sampling_allowed"], [True, True, False])
    assert info["stable_id"] == ("0816/demo1", "0816/demo2", "0830_part1/demo3")
    assert info["memory_cell"][0] != info["memory_cell"][1]

    sampling = _data_loader._sequence_sampling_info(stub, _StubMeta(), config, max_steps=40)
    offsets = np.concatenate([[0], np.cumsum(info["length"])[:-1]])
    assert sampling.weights[offsets[2] :].sum() == 0.0, "non-active manifest splits must get zero mass"
    # Episode 0 start@13 has an otherwise valid E->D grid but its first omitted transition is
    # semantic-tail E@58, before O starts at 60.  It must be filtered in sampler construction,
    # not survive until a worker transform raises.
    assert sampling.weights[13] == 0.0
    assert sampling.weights.sum() == pytest.approx(1.0)

    windows = _data_loader._memory_critical_windows(info, config)
    family_mass = {False: 0.0, True: 0.0}
    for episode_index in (0, 1):
        base = int(offsets[episode_index])
        for frame in range(int(windows[episode_index, 0]), int(windows[episode_index, 1]) + 1):
            weight = float(sampling.weights[base + frame])
            if weight:
                family_mass[_transforms.memory_critical_is_sparse(frame, windows[episode_index])] += weight
    assert family_mass[False] == pytest.approx(0.25)
    assert family_mass[True] == pytest.approx(0.25)


def test_short_strict_d_gets_clock_aware_candidate_instead_of_silent_drop(tmp_path, monkeypatch):
    episodes = [_episode("left", wait_frames=5)]
    stub = _StubLeRobot(episodes, tmp_path)
    monkeypatch.setattr(_data_loader, "_unwrap_lerobot", lambda _dataset: stub)
    manifest = _write_manifest(tmp_path / "short.json", [_manifest_record(0, "left", "train")])
    sampling = _data_loader._sequence_sampling_info(stub, _StubMeta(), _v35_config(manifest), max_steps=40)
    assert sampling.weights.sum() == pytest.approx(1.0)
    assert np.count_nonzero(sampling.weights) > 1


def test_full_and_slice_d_windows_without_same_grid_e_anchor_are_resampled(tmp_path, monkeypatch):
    # E raw frames are [31,46], hence tail-guarded eligibility is [31,41].  Starts at raw
    # residues 0 (full@0 and slice@15) miss E entirely but do intersect the later strict D;
    # other residues provide valid natural and sparse critical candidates for the episode.
    phases = ((OPEN, 31), (INSPECT, 16), (CLOSE, 240), (WAIT_L, 30), (EXEC_L, 30))
    episodes = [[IDS[label] for label, count in phases for _ in range(count)]]
    stub = _StubLeRobot(episodes, tmp_path)
    monkeypatch.setattr(_data_loader, "_unwrap_lerobot", lambda _dataset: stub)
    manifest = _write_manifest(tmp_path / "unaligned.json", [_manifest_record(0, "left", "train")])
    config = _v35_config(manifest, memory_critical_start_pad=5)
    sampling = _data_loader._sequence_sampling_info(stub, _StubMeta(), config, max_steps=40)

    assert sampling.weights[0] == 0.0
    assert sampling.weights[15] == 0.0
    assert sampling.weights.sum() == pytest.approx(1.0)


def test_missing_d_candidate_fails_closed(tmp_path, monkeypatch):
    episodes = [_episode("left", wait_frames=1)]
    stub = _StubLeRobot(episodes, tmp_path)
    monkeypatch.setattr(_data_loader, "_unwrap_lerobot", lambda _dataset: stub)
    manifest = _write_manifest(tmp_path / "bad.json", [_manifest_record(0, "left", "train")])
    with pytest.raises(ValueError, match="unusable E/O/D|no_E_to_D_candidate"):
        _data_loader._sequence_sampling_info(stub, _StubMeta(), _v35_config(manifest), max_steps=40)


def test_manifest_seed_and_coverage_fail_closed(tmp_path, monkeypatch):
    episodes = [_episode("left"), _episode("right")]
    stub = _StubLeRobot(episodes, tmp_path)
    monkeypatch.setattr(_data_loader, "_unwrap_lerobot", lambda _dataset: stub)
    wrong_seed = _write_manifest(
        tmp_path / "wrong_seed.json",
        [_manifest_record(0, "left", "train"), _manifest_record(1, "right", "train")],
        seed=34,
    )
    with pytest.raises(ValueError, match="split_seed mismatch"):
        _data_loader._episode_info_table(stub, _StubMeta(), _v35_config(wrong_seed))

    missing = _write_manifest(tmp_path / "missing.json", [_manifest_record(0, "left", "train")])
    with pytest.raises(ValueError, match="manifest/dataset episode mismatch"):
        _data_loader._episode_info_table(stub, _StubMeta(), _v35_config(missing))


def test_manifest_allows_excluded_raw_episode_without_converted_index(tmp_path, monkeypatch):
    episodes = [_episode("left")]
    stub = _StubLeRobot(episodes, tmp_path)
    monkeypatch.setattr(_data_loader, "_unwrap_lerobot", lambda _dataset: stub)
    excluded = {
        "stable_id": "0830_part2/demo14",
        "collection": "0830_part2",
        "object": "unknown",
        "target_side": "right",
        "split": "excluded",
        "include": False,
        "exclude_reason": "incomplete terminal phase",
    }
    manifest = _write_manifest(tmp_path / "with_excluded.json", [_manifest_record(0, "left", "train"), excluded])
    info = _data_loader._episode_info_table(stub, _StubMeta(), _v35_config(manifest))
    assert info["stable_id"] == ("0816/demo1",)


def _frozen_population_records():
    records = []
    cells = [(object_name, side) for object_name in ("banana", "grey_pepper_box") for side in ("left", "right")]
    for part, demos in (("part1", range(1, 17)), ("part2", [*range(1, 14), 15])):
        for offset, demo in enumerate(demos):
            object_name, side = cells[offset % len(cells)]
            records.append(
                {
                    "stable_id": f"0830_bin_{part}/demo{demo}",
                    "collection": "0830",
                    "part": part,
                    "object": object_name,
                    "target_side": side,
                    "include": True,
                }
            )
    for demo in range(1, 41):
        object_name, side = cells[(demo - 1) % len(cells)]
        records.append(
            {
                "stable_id": f"0831_bin/demo{demo}",
                "collection": "0831",
                "part": "",
                "object": object_name,
                "target_side": side,
                "include": True,
            }
        )
    records.append(
        {
            "stable_id": "0830_bin_part2/demo14",
            "raw_dir": "data/0830_bin_part2/demo14",
            "include": False,
            "exclude_reason": "no terminal execute phase",
        }
    )
    return records


def test_frozen_split_is_reproduced_from_manifest_fields_and_preserves_every_cell():
    records = _frozen_population_records()
    first = _data_loader._v35_expected_frozen_splits(records, seed=36)
    second = _data_loader._v35_expected_frozen_splits(list(reversed(records)), seed=36)
    assert first == second, "source file ordering must not influence the preregistered split"
    assert list(first.values()).count("train") == 54
    assert list(first.values()).count("development") == 8
    assert list(first.values()).count("final_test") == 8

    included = [record for record in records if record.get("include", True)]
    final_cells = {
        (record["collection"], record["object"], record["target_side"])
        for record in included
        if first[record["stable_id"]] == "final_test"
    }
    development_cells = {
        (record["collection"], record["object"], record["target_side"])
        for record in included
        if first[record["stable_id"]] == "development"
    }
    train_guards = {
        _data_loader._v36_split_guard_cell(record)
        for record in included
        if first[record["stable_id"]] == "train"
    }
    all_guards = {_data_loader._v36_split_guard_cell(record) for record in included}
    assert len(final_cells) == 8
    assert len(development_cells) == 8
    assert train_guards == all_guards, "every guarded cell must keep at least one train episode"


def test_frozen_d_valid_sidecar_masks_semantic_wait_without_observation_access():
    info = {
        "length": np.asarray([30], dtype=np.int32),
        "memory_lo": np.asarray([10], dtype=np.int32),
        "memory_hi": np.asarray([20], dtype=np.int32),
        "manifest_d_lo": np.asarray([12], dtype=np.int32),
        "manifest_d_hi": np.asarray([18], dtype=np.int32),
    }
    _data_loader._apply_frozen_v35_d_valid(info)
    assert (int(info["memory_lo"][0]), int(info["memory_hi"][0])) == (12, 18)
    expected = np.ones(30, dtype=bool)
    expected[10:12] = False
    expected[19:21] = False
    np.testing.assert_array_equal(info["episode_waiting_valid"][0], expected)

    info["manifest_d_hi"] = np.asarray([21], dtype=np.int32)
    with pytest.raises(ValueError, match="subset of its semantic D phase"):
        _data_loader._apply_frozen_v35_d_valid(info)


def test_frozen_record_provenance_binds_prompt_labels_and_independent_d_valid(tmp_path):
    raw_dir = tmp_path / "raw" / "0816_banana" / "demo1"
    raw_dir.mkdir(parents=True)
    segments = [
        {"task": OPEN, "start": 0, "end": 1},
        {"task": INSPECT, "start": 2, "end": 3},
        {"task": CLOSE, "start": 4, "end": 5},
        {"task": WAIT_L, "start": 6, "end": 7},
        {"task": EXEC_L, "start": 8, "end": 9},
    ]
    label_path = raw_dir / "subtask_labels.json"
    label_path.write_text(json.dumps(segments))
    record = {
        "episode_index": 0,
        "stable_id": "0816_banana/demo1",
        "collection": "0816",
        "part": "",
        "object": "banana",
        "prompt": "find the banana",
        "target_side": "left",
        "split": "train",
        "include": True,
        "raw_dir": "raw/0816_banana/demo1",
        "label_file": label_path.name,
        "label_sha256": hashlib.sha256(label_path.read_bytes()).hexdigest(),
        "expected_num_frames": 10,
        "e_visibility": {
            "manual_reviewed": True,
            "both_objects_visible": True,
            "first_valid_visible_frame": 2,
            "last_clean_visible_frame": 3,
            "contact_sheet_sha256": "a" * 64,
        },
        "d_valid": {
            "start": 6,
            "end": 7,
            "state_dim": 14,
            "detector": _data_loader._V35_D_VALID_DETECTOR,
        },
    }
    raw = {
        "dataset_version": "v36",
        "review_status": "frozen",
        "raw_root": ".",
        "split_algorithm": _data_loader._V35_SPLIT_ALGORITHM,
        "split_algorithm_sha256": _data_loader._V35_SPLIT_ALGORITHM_SHA256,
    }
    episode_tasks = (np.asarray([IDS[segment["task"]] for segment in segments for _ in range(2)]),)
    d_lo, d_hi = _data_loader._validate_v35_frozen_record_provenance(
        manifest_path=tmp_path / "manifest.json",
        raw=raw,
        records={0: record},
        num_episodes=1,
        episode_length=np.asarray([10], dtype=np.int32),
        episode_tasks=episode_tasks,
        tasks=VOCAB,
        prompts=("find the banana",),
        visibility_records={record["stable_id"]: record["e_visibility"]},
        d_valid_records={record["stable_id"]: record["d_valid"]},
    )
    np.testing.assert_array_equal(d_lo, [6])
    np.testing.assert_array_equal(d_hi, [7])

    record["label_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="label bytes/hash mismatch"):
        _data_loader._validate_v35_frozen_record_provenance(
            manifest_path=tmp_path / "manifest.json",
            raw=raw,
            records={0: record},
            num_episodes=1,
            episode_length=np.asarray([10], dtype=np.int32),
            episode_tasks=episode_tasks,
            tasks=VOCAB,
            prompts=("find the banana",),
            visibility_records={record["stable_id"]: record["e_visibility"]},
            d_valid_records={record["stable_id"]: record["d_valid"]},
        )


def test_legacy_defaults_and_label_tree_are_unchanged():
    config = _config.DataConfig()
    assert not config.memory_v35_enabled
    assert config.memory_e_tail_guard_frames == 0
    out = _labels()(
        {
            "subtask": [INSPECT, WAIT_L],
            "subtask_now": [INSPECT, WAIT_L],
            "episode_side": np.int32(0),
            "seq_step_mask": np.asarray([True, True]),
        }
    )
    assert "seq_write_mask" not in out
    assert "seq_decay_gap_before" not in out
    np.testing.assert_array_equal(out["seq_evidence_mask"], [True, False])
    np.testing.assert_array_equal(out["seq_waiting_mask"], [False, True])


def test_v35_config_seals_the_fifteen_raw_frame_clock(tmp_path):
    manifest = tmp_path / "placeholder.json"
    with pytest.raises(ValueError, match="memory_stride_frames=15"):
        _v35_config(manifest, memory_stride_frames=14)

    with pytest.raises(ValueError, match="disjoint evidence and occlusion"):
        _v35_config(manifest, memory_occlusion_subtasks=(INSPECT,))
