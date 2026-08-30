from __future__ import annotations

# The evaluator intentionally exposes script-private pure helpers for focused fail-closed tests.
# ruff: noqa: SLF001
from pathlib import Path
import sys
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v34_heldout_free_decode_video as video
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


def _args(**overrides) -> video.Args:
    values = {
        "checkpoint": Path("checkpoint/2500"),
        "dataset_root": Path("dataset"),
        "output_dir": Path("output"),
        "config": video.causal.RUN5_CONFIG,
        "parameter_source": "raw",
    }
    values.update(overrides)
    return video.Args(**values)


def _rows(task_indices: tuple[int, ...] = (0, 0, 1, 1)) -> list[dict[str, object]]:
    return [
        {
            "frame_index": frame,
            "episode_index": 15,
            "task_index": task,
            "state": np.zeros(14, dtype=np.float32),
        }
        for frame, task in enumerate(task_indices)
    ]


def test_args_pin_checkpoint_config_source_seed_and_shard_output() -> None:
    assert _args().episodes == video.HELDOUT_EPISODES
    assert _args(parameter_source="ema").parameter_source == "ema"
    sharded = _args(num_shards=2, shard_id=1)
    assert sharded.episodes == (29, 59)
    assert sharded.artifact_dir.name == "shard_01_of_02"

    invalid = [
        ({"checkpoint": Path("checkpoint/2499")}, "checkpoint 2500"),
        ({"config": "pi05_yam_mem_v34_run4"}, "pinned"),
        ({"parameter_source": "optimizer"}, "raw or ema"),
        ({"batch_size": 0}, "batch-size"),
        ({"batch_size": 65}, "batch-size"),
        ({"num_shards": 0}, "num-shards"),
        ({"num_shards": 5}, "num-shards"),
        ({"num_shards": 2, "shard_id": 2}, "shard-id"),
        ({"seed": 1}, "seed 0"),
    ]
    for overrides, match in invalid:
        with pytest.raises(ValueError, match=match):
            _args(**overrides)


def test_deterministic_sharding_balances_the_requested_two_gpu_split() -> None:
    assert video._episodes_for_shard(1, 0) == (15, 29, 44, 59)
    assert video._episodes_for_shard(2, 0) == (15, 44)
    assert video._episodes_for_shard(2, 1) == (29, 59)
    assert video._episodes_for_shard(4, 0) == (15,)
    assert video._episodes_for_shard(4, 3) == (59,)
    with pytest.raises(ValueError, match="shard_id"):
        video._episodes_for_shard(2, 2)


def test_rows_require_stride_one_identity_valid_tasks_and_contiguous_label_blocks() -> None:
    rows = _rows()
    runs = video._validate_rows(
        rows,
        episode=15,
        expected_length=4,
        task_names=("open", "wait"),
    )
    assert runs == [
        {"start": 0, "end": 1, "task_index": 0, "label": "open"},
        {"start": 2, "end": 3, "task_index": 1, "label": "wait"},
    ]
    assert video._boundary_frames(rows) == (0, 1, 2, 3)

    noncontiguous_frame = _rows()
    noncontiguous_frame[2]["frame_index"] = 3
    with pytest.raises(ValueError, match="contiguous from zero"):
        video._validate_rows(
            noncontiguous_frame,
            episode=15,
            expected_length=4,
            task_names=("open", "wait"),
        )

    with pytest.raises(ValueError, match="not contiguous blocks"):
        video._validate_rows(
            _rows((0, 1, 0, 0)),
            episode=15,
            expected_length=4,
            task_names=("open", "wait"),
        )

    bad_task = _rows()
    bad_task[0]["task_index"] = 2
    with pytest.raises(ValueError, match="invalid task_index"):
        video._validate_rows(
            bad_task,
            episode=15,
            expected_length=4,
            task_names=("open", "wait"),
        )

    with pytest.raises(ValueError, match="frame count mismatch"):
        video._validate_rows(
            rows,
            episode=15,
            expected_length=5,
            task_names=("open", "wait"),
        )


def test_training_target_is_exact_t_plus_15_with_end_clamp() -> None:
    rows = _rows(tuple(0 if frame < 17 else 1 for frame in range(20)))
    assert video._training_target(rows, 0, 15) == (15, 0)
    assert video._training_target(rows, 2, 15) == (17, 1)
    assert video._training_target(rows, 10, 15) == (19, 1)
    with pytest.raises(ValueError, match="outside"):
        video._training_target(rows, 20, 15)


def test_fixed_batch_padding_repeats_only_the_last_live_entry() -> None:
    padded, live = video._pad_batch(["a", "b", "c"], 5)
    assert live == 3
    assert padded == ["a", "b", "c", "c", "c"]
    with pytest.raises(ValueError, match="nonempty"):
        video._pad_batch([], 5)
    with pytest.raises(ValueError, match="no larger"):
        video._pad_batch([1, 2], 1)


def test_processed_camera_tiles_are_exact_and_render_does_not_modify_them() -> None:
    normalized = {
        "base_0_rgb": np.full((1, 224, 224, 3), -1.0, dtype=np.float32),
        "left_wrist_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.float32),
        "right_wrist_0_rgb": np.ones((1, 224, 224, 3), dtype=np.float32),
    }
    observation = SimpleNamespace(
        images=normalized,
        image_masks={key: np.ones((1,), dtype=bool) for key in normalized},
    )
    images = video._processed_images(observation, 0)
    assert np.all(images["base_0_rgb"] == 0)
    assert np.all(images["left_wrist_0_rgb"] == 128)
    assert np.all(images["right_wrist_0_rgb"] == 255)

    frame = video._render_frame(
        images,
        episode=15,
        frame=7,
        fps=30.0,
        prompt="find the banana",
        decoded="wait; target bin is left",
        decoded_token_count=7,
        decoded_terminated=True,
        decoded_truncated=False,
        gt_now="close both lids and reset arms",
        target_frame=22,
        gt_train_target="wait; target bin is left",
        parameter_source="raw",
        config=video.causal.RUN5_CONFIG,
    )
    assert frame.shape == (video.CANVAS_HEIGHT, video.CANVAS_WIDTH, 3)
    assert frame.dtype == np.uint8
    image_row = frame[video.HEADER_HEIGHT : video.HEADER_HEIGHT + video.DISPLAY_IMAGE_SIZE]
    for index, (_label, key) in enumerate(video.CAMERA_KEYS):
        x0 = index * video.DISPLAY_IMAGE_SIZE
        displayed = image_row[:, x0 : x0 + video.DISPLAY_IMAGE_SIZE]
        assert np.array_equal(displayed[::2, ::2], images[key])
        assert np.array_equal(displayed[1::2, 1::2], images[key])

    uint8_observation = SimpleNamespace(
        images={key: value.copy() for key, value in images.items()},
        image_masks={key: np.ones((1,), dtype=bool) for key in images},
    )
    uint8_observation.images = {key: value[None] for key, value in uint8_observation.images.items()}
    copied = video._processed_images(uint8_observation, 0)
    assert all(np.array_equal(copied[key], images[key]) for key in images)


def test_infer_uses_fresh_m0_zero_noise_and_frozen_write_contract() -> None:
    runner = object.__new__(video.HeldoutFreeDecodeVideo)
    runner.args = SimpleNamespace(seed=0)
    runner._first_batch_contract = None
    calls = []

    class FakeMemory:
        @staticmethod
        def init_state(batch: int):
            return {"m": jnp.zeros((batch, 2), dtype=jnp.float32)}

    def fake_sample(key, observation, state, **kwargs):
        del key
        calls.append(kwargs)
        batch = observation.state.shape[0]
        assert np.array_equal(np.asarray(kwargs["noise"]), np.zeros((batch, 3, 4), dtype=np.float32))
        return (
            jnp.zeros((batch, 3, 4), dtype=jnp.float32),
            state,
            {
                "write_occurred": jnp.zeros((batch,), dtype=bool),
                "retrieval_norm": jnp.zeros((batch,), dtype=jnp.float32),
                "tokens": jnp.ones((batch, 10), dtype=jnp.int32),
                "token_mask": jnp.ones((batch, 10), dtype=bool),
            },
        )

    runner.base = SimpleNamespace(
        model=SimpleNamespace(memory=FakeMemory(), action_horizon=3, action_dim=4),
        stop_token=99,
        _sample=fake_sample,
        _state_max_abs_diff=lambda left, right: float(np.max(np.abs(np.asarray(left["m"]) - np.asarray(right["m"])))),
    )
    observation = SimpleNamespace(state=jnp.zeros((2, 14), dtype=jnp.float32))
    runner._infer(observation, num_steps=1, zero_read=False)
    assert calls[0]["max_decode_steps"] == 10
    assert calls[0]["num_steps"] == 1
    assert calls[0]["zero_read"] is False
    assert calls[0]["allow_write"] is False
    assert calls[0]["write_mode"] == "frozen"
    assert runner._first_batch_contract == {
        "batch_size": 2,
        "write_occurred_all_false": True,
        "retrieval_norm_max_abs": 0.0,
        "returned_state_max_abs_difference": 0.0,
    }


def test_inference_contract_fails_on_write_read_or_returned_state() -> None:
    runner = object.__new__(video.HeldoutFreeDecodeVideo)
    runner.base = SimpleNamespace(
        _state_max_abs_diff=lambda left, right: float(np.max(np.abs(np.asarray(left) - np.asarray(right))))
    )
    state = np.zeros((2, 2), dtype=np.float32)
    good = {
        "write_occurred": np.zeros(2, dtype=bool),
        "retrieval_norm": np.zeros(2, dtype=np.float32),
    }
    assert runner._assert_inference_contract(state, state.copy(), good, batch=2)["write_occurred_all_false"]

    bad_write = {**good, "write_occurred": np.asarray([False, True])}
    with pytest.raises(RuntimeError, match="write_occurred"):
        runner._assert_inference_contract(state, state.copy(), bad_write, batch=2)

    bad_read = {**good, "retrieval_norm": np.asarray([0.0, 1e-3])}
    with pytest.raises(RuntimeError, match="retrieval is not zero"):
        runner._assert_inference_contract(state, state.copy(), bad_read, batch=2)

    with pytest.raises(RuntimeError, match="changed memory state"):
        runner._assert_inference_contract(state, np.ones_like(state), good, batch=2)


def test_decode_status_distinguishes_stop_termination_from_limit_truncation() -> None:
    runner = object.__new__(video.HeldoutFreeDecodeVideo)
    runner.base = SimpleNamespace(
        stop_token=99,
        tokenizer=SimpleNamespace(decode=lambda tokens: " ".join(map(str, tokens))),
    )
    tokens = np.asarray(
        [
            [10, 99, 0, 0, 0, 0, 0, 0, 0, 0],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ],
        dtype=np.int32,
    )
    masks = np.asarray(
        [
            [True, True, False, False, False, False, False, False, False, False],
            [True] * 10,
        ]
    )
    decoded, returned_tokens, returned_masks, statuses = runner._decode_batch({"tokens": tokens, "token_mask": masks})
    assert decoded == ["10 99", "10 11 12 13 14 15 16 17 18 19"]
    assert np.array_equal(returned_tokens, tokens)
    assert np.array_equal(returned_masks, masks)
    assert statuses == [
        {
            "token_count": 2,
            "terminated": True,
            "truncated": False,
            "termination_token": "stop/newline",
        },
        {
            "token_count": 10,
            "terminated": False,
            "truncated": True,
            "termination_token": None,
        },
    ]

    with pytest.raises(RuntimeError, match="stopped before its limit"):
        runner._decode_batch(
            {
                "tokens": np.asarray([[10, 11, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32),
                "token_mask": np.asarray([[True, True, False, False, False, False, False, False, False, False]]),
            }
        )


def test_exact_run5_source_root_is_required(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(video.RUN5_SOURCE_ENV, raising=False)
    with pytest.raises(RuntimeError, match=video.RUN5_SOURCE_ENV):
        video._validate_run5_source_root(tmp_path)


def test_atomic_json_and_complete_hashes_and_refuses_overwrite(tmp_path: Path) -> None:
    report = tmp_path / "summary.json"
    identity = video._write_json_atomic(report, {"ok": True})
    assert report.is_file()
    assert identity["sha256"] == video.causal._sha256_file(report)
    assert not (tmp_path / ".summary.json.tmp").exists()
    with pytest.raises(FileExistsError, match="overwrite"):
        video._write_json_atomic(report, {"ok": False})

    video._write_complete_atomic(tmp_path, {"summary.json": identity})
    complete = tmp_path / "COMPLETE"
    assert complete.read_text(encoding="utf-8") == f"{identity['sha256']}  summary.json\n"
    with pytest.raises(FileExistsError, match="overwrite"):
        video._write_complete_atomic(tmp_path, {"summary.json": identity})


@pytest.mark.skipif(video.shutil.which("ffmpeg") is None, reason="ffmpeg is unavailable")
def test_atomic_mp4_writer_closes_pipe_and_installs_nonempty_h264(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mp4"
    writer = video._AtomicMp4Writer(path, 30.0)
    writer.write(np.zeros((video.CANVAS_HEIGHT, video.CANVAS_WIDTH, 3), dtype=np.uint8))
    identity = writer.close()
    assert writer.frames == 1
    assert path.is_file()
    assert path.stat().st_size > 0
    assert identity["sha256"] == video.causal._sha256_file(path)
    assert not (tmp_path / ".tiny.mp4.tmp").exists()
