import dataclasses
import hashlib
from pathlib import Path
import sys
from typing import NamedTuple

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v35_calibration_replay as replay
    import v35_collect_calibration_replay as collector
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


def _fake_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "memory_project"
    (root / "openpi" / "src" / "openpi").mkdir(parents=True)
    (root / "openpi" / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    monkeypatch.setenv("MEMORY_PROJECT_ROOT", str(root))
    return root


def _bindings() -> replay.ReplayBindings:
    return replay.ReplayBindings(
        step0_parameter_tree_sha256="1" * 64,
        official_base_source_sha256="2" * 64,
        replay_protocol_sha256=replay.replay_protocol_sha256(),
        collector_source_sha256=replay.collector_source_sha256(),
        manifest_sha256="3" * 64,
        dataset_protocol_sha256="4" * 64,
    )


def _preflight() -> dict:
    bindings = _bindings()
    train_ids = [f"train-{index:03d}" for index in range(54)]
    payload = {
        "schema_version": replay.PREFLIGHT_SCHEMA,
        "status": "init_complete_replay_required",
        "initialization": {
            "actual_step0_parameter_tree_sha256": bindings.step0_parameter_tree_sha256,
            "official_base_source_tree_sha256": bindings.official_base_source_sha256,
        },
        "data": {
            "production": True,
            "dataset_repo_id": "yam/bin_memory_0830_0831_v36_subtask",
            "dataset_episode_frame_protocol_sha256": bindings.dataset_protocol_sha256,
            "train_storage_sha256": "5" * 64,
            "manifest_sha256": bindings.manifest_sha256,
            "train_episode_count": 54,
            "train_episode_indices": list(range(54)),
            "train_stable_ids": train_ids,
        },
        "replay": {
            "protocol_sha256": bindings.replay_protocol_sha256,
            "collector_source_sha256": bindings.collector_source_sha256,
        },
    }
    return replay._envelope(payload, id_prefix="preflight")  # noqa: SLF001


def _manifest() -> dict:
    records = []
    for index in range(70):
        train = index < 54
        side = "left" if index % 2 == 0 else "right"
        records.append(
            {
                "stable_id": f"train-{index:03d}" if train else f"heldout-{index:03d}",
                "episode_index": index,
                "include": True,
                "split": "train" if train else ("development" if index < 62 else "final_test"),
                "expected_num_frames": 91,
                "target_side": side,
                "prompt": "find the banana",
                "e_visibility": {
                    "manual_reviewed": True,
                    "both_objects_visible": True,
                    "first_valid_visible_frame": 15,
                    "last_clean_visible_frame": 29,
                },
                "d_valid": {"start": 60, "end": 74},
            }
        )
    return {"schema_version": 2, "episodes": records}


def _provenance() -> dict:
    files = [
        {
            "path": f"data/chunk-000/episode_{index:06d}.parquet",
            "size": 1,
            "sha256": hashlib.sha256(f"episode-{index}".encode()).hexdigest(),
        }
        for index in range(54)
    ]
    return {
        "train_storage": {
            "root_contract": replay.TRAIN_STORAGE_ROOT_CONTRACT,
            "files": files,
        }
    }


def _columns(index: int, *, d_task: int | None = None) -> collector.EpisodeColumns:
    frame = np.arange(91, dtype=np.int64)
    task = np.zeros(91, dtype=np.int64)
    task[15:30] = 4
    task[30:60] = 3
    task[60:75] = (1 if index % 2 == 0 else 6) if d_task is None else d_task
    task[75:] = 2 if index % 2 == 0 else 5
    return collector.EpisodeColumns(
        episode_index=np.full(91, index, dtype=np.int64),
        frame_index=frame,
        dataset_index=np.arange(index * 100, index * 100 + 91, dtype=np.int64),
        task_index=task,
    )


def test_selection_reads_only_preflight_train_parquets_and_freezes_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fake_project(tmp_path, monkeypatch)
    dataset_root = root / "data" / "lerobot" / "yam" / "bin_memory_0830_0831_v36_subtask"
    dataset_root.mkdir(parents=True)
    opened: list[int] = []

    def reader(path: Path, index: int) -> collector.EpisodeColumns:
        assert path.name == f"episode_{index:06d}.parquet"
        opened.append(index)
        return _columns(index)

    artifact = collector.build_frame_selection_artifact(
        preflight=_preflight(),
        manifest=_manifest(),
        provenance=_provenance(),
        tasks={
            0: "open both lids",
            1: "wait; target bin is left",
            2: "open left bin",
            3: "close both lids and reset arms",
            4: "inspect both bins",
            5: "open right bin",
            6: "wait; target bin is right",
        },
        read_episode_columns=reader,
        dataset_root=dataset_root,
        structural_meta_sha256={"meta/info.json": "a" * 64},
    )

    assert opened == list(range(54))
    assert all(index < 54 for index in opened)
    assert collector._verify_envelope(artifact, prefix="selection")  # noqa: SLF001
    assert artifact["payload"]["data"]["heldout_payload_access_count"] == 0
    episodes = artifact["payload"]["selection"]["episodes"]
    assert len(episodes) == 54
    assert episodes[0]["evidence"]["frame_index"] == 15
    assert episodes[0]["decision"]["frame_index"] == 60
    assert episodes[0]["decision"]["n_delay"] == 2
    assert [item["frame_index"] for item in episodes[0]["query_controls"]] == [30, 45, 60]
    assert [item["n_delay"] for item in episodes[0]["query_controls"]] == [0, 1, 2]


def test_episode_selection_fails_when_stationary_d_task_has_wrong_side() -> None:
    record = _manifest()["episodes"][0]
    parquet_record = _provenance()["train_storage"]["files"][0]

    with pytest.raises(collector.CollectionError, match="no stride-15 stationary-D"):
        collector._episode_selection(  # noqa: SLF001
            record=record,
            columns=_columns(0, d_task=6),
            tasks={
                0: "open both lids",
                1: "wait; target bin is left",
                2: "open left bin",
                3: "close both lids and reset arms",
                4: "inspect both bins",
                5: "open right bin",
                6: "wait; target bin is right",
            },
            parquet_record=parquet_record,
        )


def test_synthetic_fallback_is_deterministic_orthogonal_and_reads_real_w3() -> None:
    hidden = np.asarray([0.8, -0.2, 0.5, 0.1], dtype=np.float32)
    w3 = np.asarray(
        [
            [0.4, 0.2, -0.1],
            [0.1, -0.3, 0.2],
            [0.5, 0.1, 0.4],
            [-0.2, 0.6, 0.3],
        ],
        dtype=np.float32,
    )

    orthogonal, cosine, raw = collector._synthetic_orthogonal_control(hidden, w3)  # noqa: SLF001
    second = collector._synthetic_orthogonal_control(hidden, w3)  # noqa: SLF001

    assert orthogonal.dtype == np.float32
    assert raw.dtype == np.float32
    assert abs(float(cosine)) < 1e-6
    np.testing.assert_allclose(raw, orthogonal @ w3, rtol=0, atol=0)
    for left, right in zip((orthogonal, cosine, raw), second, strict=True):
        np.testing.assert_array_equal(left, right)
    with pytest.raises(collector.CollectionError, match="nonzero delayed memory"):
        collector._synthetic_orthogonal_control(hidden, np.zeros_like(w3))  # noqa: SLF001


def test_episode_collector_preserves_16_slots_and_emits_synthetic_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    import jax.numpy as jnp

    class _State(NamedTuple):
        fast_weights: dict[str, jnp.ndarray]
        momentum: dict[str, jnp.ndarray]

    class _Memory:
        _output_weight_name = "w3"

        def init_state(self, batch_size: int) -> _State:
            w3 = jnp.zeros((batch_size, 2, 4), dtype=jnp.float32)
            return _State({"w3": w3}, {"w3": jnp.zeros_like(w3)})

        def project_kv(self, tokens: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            batch = tokens.shape[0]
            return (
                jnp.broadcast_to(jnp.asarray([[[1.0, 0.0]]], dtype=jnp.float32), (batch, 1, 2)),
                jnp.ones((batch, 1, 4), dtype=jnp.float32),
            )

        def pool_kv(self, keys: jnp.ndarray, values: jnp.ndarray) -> dict[str, jnp.ndarray]:
            return {"pooled_key": keys[:, 0], "pooled_value": values[:, 0]}

        def write(self, state: _State, _tokens: jnp.ndarray) -> tuple[_State, dict[str, jnp.ndarray]]:
            w3 = jnp.broadcast_to(
                jnp.asarray(
                    [
                        [0.5, 0.4, 0.3, 0.2],
                        [0.1, -0.2, 0.4, 0.6],
                    ],
                    dtype=jnp.float32,
                ),
                state.fast_weights["w3"].shape,
            )
            batch = w3.shape[0]
            return _State({"w3": w3}, {"w3": jnp.zeros_like(w3)}), {
                "commit_applied": jnp.ones((batch,), dtype=jnp.bool_),
                "post_residual": jnp.full((batch, 4), 0.005, dtype=jnp.float32),
                "relative_commit_residual": jnp.full((batch,), 1e-4, dtype=jnp.float32),
            }

        def hidden_key(self, state: _State, keys: jnp.ndarray) -> jnp.ndarray:
            del state
            # Stored and natural query hiddens are deliberately aligned, forcing the fallback.
            batch, slots = keys.shape[:2]
            return jnp.broadcast_to(jnp.asarray([1.0, 0.0], dtype=jnp.float32), (batch, slots, 2))

        def analytic_decay(self, state: _State, gaps: int | jnp.ndarray) -> tuple[_State, dict[str, jnp.ndarray]]:
            gap = jnp.asarray(gaps, dtype=jnp.int32)
            if gap.ndim == 0:
                gap = jnp.broadcast_to(gap, (state.fast_weights["w3"].shape[0],))
            factor = jnp.power(jnp.float32(0.99), gap.astype(jnp.float32))
            w3 = factor[:, None, None] * state.fast_weights["w3"]
            return _State({"w3": w3}, {"w3": jnp.zeros_like(w3)}), {
                "decay_gap_valid": jnp.ones(gap.shape, dtype=jnp.bool_)
            }

        def project_q(self, queries: jnp.ndarray) -> jnp.ndarray:
            return queries

    class _Model:
        memory = _Memory()

    def prepare(_model: _Model, observation: jnp.ndarray, state: _State) -> dict[str, jnp.ndarray]:
        batch = observation.shape[0]
        read_queries = jnp.broadcast_to(jnp.asarray([1.0, 0.0], dtype=jnp.float32), (batch, 16, 2))
        retrieved = jnp.einsum("bsh,bhd->bsd", read_queries, state.fast_weights["w3"])
        return {
            "write_tokens": jnp.ones((batch, 1, 3), dtype=jnp.float32),
            "read_queries": read_queries,
            "retrieved": retrieved,
            "h8_all": jnp.full((batch, 3, 4), 0.25, dtype=jnp.float32),
            "prefix_mask": jnp.ones((batch, 3), dtype=jnp.bool_),
        }

    monkeypatch.setattr(replay, "_prepare_early_interface", prepare)
    episode = {
        "stable_id": "train-000",
        "evidence": {"frame_index": 15},
        "decision": {"frame_index": 60, "n_delay": 2},
        "query_controls": [
            {"frame_index": 30, "n_delay": 0},
            {"frame_index": 45, "n_delay": 1},
            {"frame_index": 60, "n_delay": 2},
        ],
    }
    observations = {frame: np.asarray([frame, 1], dtype=np.int32) for frame in (15, 30, 45, 60)}

    record = collector._collect_episode(  # noqa: SLF001
        model=_Model(),
        bindings=_bindings(),
        selection_sha256="a" * 64,
        episode=episode,
        observations=observations,
        query_batch_size=2,
    )

    assert record.clean_raw_retrieved.shape == (16, 4)
    assert record.query_noise_raw_retrieved.shape == (1, 4)
    assert record.query_noise_kind == ("synthetic_orthogonal_query",)
    assert record.query_noise_frame_index.tolist() == [-1]
    assert abs(float(record.query_noise_cosine[0])) < 1e-6


def test_selection_reader_rejects_source_or_payload_tampering(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _fake_project(tmp_path, monkeypatch)
    dataset_root = root / "data" / "lerobot" / "yam" / "bin_memory_0830_0831_v36_subtask"
    dataset_root.mkdir(parents=True)
    preflight = _preflight()
    artifact = collector.build_frame_selection_artifact(
        preflight=preflight,
        manifest=_manifest(),
        provenance=_provenance(),
        tasks={
            0: "open both lids",
            1: "wait; target bin is left",
            2: "open left bin",
            3: "close both lids and reset arms",
            4: "inspect both bins",
            5: "open right bin",
            6: "wait; target bin is right",
        },
        read_episode_columns=lambda _path, index: _columns(index),
        dataset_root=dataset_root,
        structural_meta_sha256={},
    )
    selection = tmp_path / "selection.json"
    collector._write_json_once(selection, artifact)  # noqa: SLF001
    assert (
        collector.read_frame_selection(selection, preflight=preflight)["artifact_sha256"] == artifact["artifact_sha256"]
    )

    changed = dataclasses.replace(_columns(0), frame_index=np.arange(91, dtype=np.int64))
    assert changed.frame_index[-1] == 90
    raw = selection.read_text(encoding="utf-8").replace("complete_train54_only", "tampered_train74_only")
    selection.write_text(raw, encoding="utf-8")
    with pytest.raises(collector.CollectionError, match="hash/ID"):
        collector.read_frame_selection(selection, preflight=preflight)


def test_single_frame_observation_coerces_scalar_masks_and_batches():
    import v35_collect_calibration_replay as collect

    value = {
        "image": {"base_0_rgb": np.zeros((4, 4, 3), dtype=np.float32)},
        "image_mask": {"base_0_rgb": np.bool_(True)},  # noqa: FBT003
        "state": np.zeros((14,), dtype=np.float32),
        "tokenized_prompt": np.zeros((8,), dtype=np.int32),
        "tokenized_prompt_mask": np.ones((8,), dtype=bool),
    }

    observation = collect._observation_from_transformed(value)  # noqa: SLF001

    mask = observation.image_masks["base_0_rgb"]
    assert isinstance(mask, np.ndarray)
    assert mask.shape == ()
    batched = collect._batch_observations([observation, observation])  # noqa: SLF001
    assert batched.image_masks["base_0_rgb"].shape == (2,)
    assert batched.images["base_0_rgb"].shape == (2, 4, 4, 3)
