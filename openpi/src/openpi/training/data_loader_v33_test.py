"""Tests for the v3.3 label-derived phase table and memory-critical sequence sampling.

The lerobot dataset is stubbed at the `_unwrap_lerobot` seam: these tests exercise exactly the
frame math (phase bounds, dead zones, branch masses, cell balance, bucket lengths) that the
real loader applies to the 0816 dataset.
"""

# ruff: noqa: SLF001, I001 - private sampler helpers ARE the contract under test; and
# data_loader (torch) must import before config (tensorflow) or the interpreter segfaults,
# same pinned order as data_loader_test.py.

import json

import numpy as np
import pytest

import openpi.training.data_loader as _data_loader
import openpi.training.config as _config
import openpi.transforms as _transforms

OPEN, INSPECT, CLOSE = "open both lids", "inspect both bins", "close both lids and reset arms"
WAIT_L, WAIT_R = "wait; target bin is left", "wait; target bin is right"
EXEC_L, EXEC_R = "open left bin", "open right bin"
VOCAB = {0: OPEN, 1: INSPECT, 2: CLOSE, 3: WAIT_L, 4: WAIT_R, 5: EXEC_L, 6: EXEC_R}
_IDS = {v: k for k, v in VOCAB.items()}

# Frame layout used by every synthetic episode (length 200):
#   open 0-59 | inspect 60-89 | close 90-139 | wait 140-159 | execute 160-199
_PHASES = ((OPEN, 60), (INSPECT, 30), (CLOSE, 50), ("wait", 20), ("exec", 40))


def _episode(side: str) -> list[int]:
    tasks = []
    for name, span in _PHASES:
        label = {"wait": WAIT_L if side == "left" else WAIT_R, "exec": EXEC_L if side == "left" else EXEC_R}.get(
            name, name
        )
        tasks.extend([_IDS[label]] * span)
    return tasks


class _StubHF:
    def __init__(self, cols):
        self._cols = cols

    def with_format(self, _fmt):
        return self._cols


class _StubLeRobot:
    def __init__(self, episodes: list[list[int]], root):
        self.hf_dataset = _StubHF(
            {
                "task_index": [t for ep in episodes for t in ep],
                "episode_index": [e for e, ep in enumerate(episodes) for _ in ep],
            }
        )
        self.root = root


class _StubMeta:
    tasks = VOCAB


@pytest.fixture
def four_cell_dataset(tmp_path, monkeypatch):
    """banana-L, banana-R, grey-L, grey-R -- one episode per balance cell."""
    episodes = [_episode("left"), _episode("right"), _episode("left"), _episode("right")]
    stub = _StubLeRobot(episodes, tmp_path)
    (tmp_path / "meta").mkdir()
    prompts = {0: "find the banana", 1: "find the banana", 2: "find the grey pepper box", 3: "find the grey pepper box"}
    (tmp_path / "meta" / "episode_prompts.json").write_text(json.dumps({str(k): v for k, v in prompts.items()}))
    monkeypatch.setattr(_data_loader, "_unwrap_lerobot", lambda _d: stub)
    return stub


def _data_config(**overrides) -> _config.DataConfig:
    return _config.DataConfig(
        memory_stride_frames=15,
        memory_slice_prob=0.5,
        memory_min_slice_steps=2,
        subtask_lookahead=5,
        evidence_subtasks=(INSPECT,),
        memory_required_subtasks=(WAIT_L, WAIT_R),
        memory_critical_prob=0.5,
        memory_critical_start_pad=30,
        prompt_from_episode_meta=True,
        **overrides,
    )


def test_phase_table_derives_label_bounds(four_cell_dataset):
    info = _data_loader._episode_info_table(four_cell_dataset, _StubMeta(), _data_config())
    np.testing.assert_array_equal(info["evidence_start"], [60] * 4)
    np.testing.assert_array_equal(info["evidence_end"], [89] * 4)
    np.testing.assert_array_equal(info["memory_lo"], [140] * 4)
    np.testing.assert_array_equal(info["memory_hi"], [159] * 4)
    np.testing.assert_array_equal(info["side"], [0, 1, 0, 1])

    windows = _data_loader._memory_critical_windows(info, _data_config())
    np.testing.assert_array_equal(windows, [[30, 60, 140, 159]] * 4)


def test_episode_without_wait_phase_is_excluded(tmp_path, monkeypatch):
    broken = [_IDS[OPEN]] * 100 + [_IDS[EXEC_L]] * 100  # no inspect, no wait
    stub = _StubLeRobot([_episode("left"), broken], tmp_path)
    monkeypatch.setattr(_data_loader, "_unwrap_lerobot", lambda _d: stub)
    info = _data_loader._episode_info_table(stub, _StubMeta(), _data_config())
    assert info["evidence_start"][0] == 60
    assert info["evidence_start"][1] == -1
    windows = _data_loader._memory_critical_windows(info, _data_config())
    assert (windows[1] == -1).all()


def test_sampler_branch_masses_dead_zone_and_cell_balance(four_cell_dataset):
    config = _data_config()
    sampling = _data_loader._sequence_sampling_info(four_cell_dataset, _StubMeta(), config, max_steps=40)
    weights = sampling.weights.reshape(4, 200)

    # branch masses: 25% full trajectories, 25% slices, 50% memory-critical
    assert weights[:, 0].sum() == pytest.approx(0.25)
    in_window = np.zeros(200, dtype=bool)
    in_window[30:61] = True
    assert weights[:, in_window].sum() == pytest.approx(0.5)
    assert weights.sum() == pytest.approx(1.0)

    # every (instruction, side) cell holds exactly a quarter of the memory-critical mass
    for e in range(4):
        assert weights[e, in_window].sum() == pytest.approx(0.125)

    # the dead zone (evidence_end, memory_hi] never starts a sequence: a blank memory there
    # would be graded on side labels it can never answer
    assert weights[:, 90:160].sum() == 0.0
    # execution starts are ordinary slices again
    assert weights[:, 160:170].sum() > 0.0

    # memory-critical starts are bucketed at their deterministic endpoint, not episode end
    steps = sampling.valid_steps.reshape(4, 200)
    assert steps[0, 30] == 9  # start 30: the single eligible wait step is frame 150 (k=8)
    assert steps[0, 60] == 7  # start 60: frame 150 again (k=6)
    assert steps[0, 0] == np.ceil(200 / 15)  # full trajectories still run to the episode end


def test_sampler_and_transform_agree_on_every_memory_critical_length(four_cell_dataset):
    """THE bucket-homogeneity invariant behind the collate error seen at launch: for every
    window start, the sampler's valid_steps must equal the steps BuildMemorySequence actually
    leaves after truncation -- otherwise a truncated sample lands in a longer bucket's batch
    and _sequence_bucket_collate_fn raises 'sequence bucket batch is not homogeneous'."""
    config = _data_config()
    sampling = _data_loader._sequence_sampling_info(four_cell_dataset, _StubMeta(), config, max_steps=40)
    info = _data_loader._episode_info_table(four_cell_dataset, _StubMeta(), config)
    windows = _data_loader._memory_critical_windows(info, config)
    build = _transforms.BuildMemorySequence(
        stride=15, action_horizon=3, block_steps=4, subtask_lookahead=config.subtask_lookahead
    )
    steps = sampling.valid_steps.reshape(4, 200)
    weights = sampling.weights.reshape(4, 200)
    for e in range(4):
        for f in range(int(windows[e, 0]), int(windows[e, 1]) + 1):
            assert weights[e, f] > 0
            item = {
                "frame_index": np.int64(f),
                "index": np.int64(0),
                "episode_length": np.int32(200),
                "memory_window": windows[e],
                "observation/image": np.zeros((40, 2, 2, 3), dtype=np.uint8),
                "observation/left_wrist_image": np.zeros((40, 2, 2, 3), dtype=np.uint8),
                "observation/right_wrist_image": np.zeros((40, 2, 2, 3), dtype=np.uint8),
                "observation/state": np.zeros((40, 14), dtype=np.float32),
                "actions": np.zeros((40, 3 * 14), dtype=np.float32),
            }
            out = build(item)
            assert int(out["seq_step_mask"].sum()) == steps[e, f], (e, f, steps[e, f])


def test_sampler_without_phase_config_keeps_legacy_behavior(four_cell_dataset):
    config = _config.DataConfig(memory_stride_frames=15, memory_slice_prob=0.5, memory_min_slice_steps=2)
    sampling = _data_loader._sequence_sampling_info(four_cell_dataset, _StubMeta(), config, max_steps=40)
    weights = sampling.weights.reshape(4, 200)
    assert weights[:, 0].sum() == pytest.approx(0.5)  # legacy full/slice split only
    assert weights.sum() == pytest.approx(1.0)
    # legacy dead zone: (reveal=300 default, switch) -- switch is the first label change (60);
    # nothing here is excluded beyond end-proximity, so slices exist in the old dead-zone span
    assert weights[:, 100:160].sum() > 0.0


def test_memory_critical_prob_with_no_usable_episode_disables_branch(tmp_path, monkeypatch):
    broken = [_IDS[OPEN]] * 100 + [_IDS[EXEC_L]] * 100
    stub = _StubLeRobot([broken, list(broken)], tmp_path)
    monkeypatch.setattr(_data_loader, "_unwrap_lerobot", lambda _d: stub)
    sampling = _data_loader._sequence_sampling_info(stub, _StubMeta(), _data_config(), max_steps=40)
    assert sampling.weights.sum() == pytest.approx(1.0)  # full+slice mass renormalized to 1


def test_load_episode_prompts_reads_sidecar(four_cell_dataset):
    prompts = _data_loader._load_episode_prompts(four_cell_dataset)
    assert prompts == (
        "find the banana",
        "find the banana",
        "find the grey pepper box",
        "find the grey pepper box",
    )


def test_v33_registry_configs_resolve():
    v33 = _config.get_config("pi05_yam_mem_v33")
    assert v33.model.memory_task_conditioned_write
    assert v33.model.memory_block_steps == 25  # normal samples keep TBPTT fences
    base = v33.data.base_config if hasattr(v33.data, "base_config") else None
    assert base.memory_critical_prob == 0.5
    assert base.prompt_from_episode_meta
    assert base.memory_required_subtasks == (WAIT_L, WAIT_R)
    assert _config.get_config("pi05_yam_0816").data.base_config.prompt_from_episode_meta
