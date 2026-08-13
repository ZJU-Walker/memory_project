import dataclasses
import json

import jax
import numpy as np
import pytest

from openpi.models import memory
from openpi.models import memory_diagnostics as diagnostics


def _numpy_state() -> memory.MemoryState:
    return memory.MemoryState(
        fast_weights={
            "w0": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            "b0": np.arange(6, dtype=np.float32).reshape(2, 3),
        },
        momentum={
            "w0": np.linspace(-1, 1, 12, dtype=np.float32).reshape(2, 2, 3),
            "b0": np.full((2, 3), 0.25, dtype=np.float32),
        },
    )


def _rewrite_npz(path, *, mutate_manifest=None, mutate_arrays=None):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    if mutate_manifest is not None:
        manifest = json.loads(arrays["__manifest__"].tobytes().decode("utf-8"))
        mutate_manifest(manifest)
        arrays["__manifest__"] = np.frombuffer(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8"), dtype=np.uint8
        ).copy()
    if mutate_arrays is not None:
        mutate_arrays(arrays)
    with path.open("wb") as output:
        np.savez_compressed(output, **arrays)


def test_clone_and_fork_are_complete_equal_and_mutably_isolated():
    original = _numpy_state()
    clone = diagnostics.clone_memory_state(original)

    diagnostics.assert_memory_states_equal(original, clone)
    diagnostics.assert_memory_states_isolated(original, clone)
    assert set(clone.fast_weights) == set(clone.momentum) == {"w0", "b0"}

    clone.fast_weights["w0"][0, 0, 0] = 999
    clone.momentum["b0"][1, 2] = -999
    assert original.fast_weights["w0"][0, 0, 0] == 0
    assert original.momentum["b0"][1, 2] == 0.25
    assert not diagnostics.memory_states_equal(original, clone)

    branches = diagnostics.fork_memory_state(original, 3)
    for index, branch in enumerate(branches):
        diagnostics.assert_memory_states_equal(original, branch)
        for other in branches[index + 1 :]:
            diagnostics.assert_memory_states_isolated(branch, other)
    branches[0].fast_weights["b0"][0, 0] = 123
    assert branches[1].fast_weights["b0"][0, 0] != 123
    assert original.fast_weights["b0"][0, 0] != 123


def test_isolation_detects_cross_leaf_numpy_aliases():
    original = _numpy_state()
    aliased = memory.MemoryState(
        fast_weights={"w0": original.fast_weights["w0"], "b0": original.fast_weights["b0"]},
        momentum={"w0": original.momentum["w0"], "b0": original.momentum["b0"]},
    )
    assert diagnostics.memory_states_share_mutable_storage(original, aliased)
    with pytest.raises(AssertionError, match="share mutable"):
        diagnostics.assert_memory_states_isolated(original, aliased)


def test_hash_is_deterministic_across_mapping_order_and_jax_numpy_backends():
    numpy_state = _numpy_state()
    reverse_state = memory.MemoryState(
        fast_weights=dict(reversed(numpy_state.fast_weights.items())),
        momentum=dict(reversed(numpy_state.momentum.items())),
    )
    jax_state = diagnostics.clone_memory_state(numpy_state, backend="jax")

    expected = diagnostics.memory_state_hash(numpy_state)
    assert diagnostics.memory_state_hash(reverse_state) == expected
    assert diagnostics.memory_state_hash(jax_state) == expected
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(jax_state))

    changed_momentum = diagnostics.clone_memory_state(numpy_state)
    changed_momentum.momentum["w0"][0, 0, 0] += 1
    assert diagnostics.memory_state_hash(changed_momentum) != expected


@pytest.mark.parametrize(
    ("state", "error"),
    [
        (memory.MemoryState({}, {}), "non-empty"),
        (
            memory.MemoryState(
                fast_weights={"w": np.zeros((1, 2), np.float32)},
                momentum={"other": np.zeros((1, 2), np.float32)},
            ),
            "keys must match",
        ),
        (
            memory.MemoryState(
                fast_weights={"w": np.zeros((1, 2), np.float32)},
                momentum={"w": np.zeros((1, 3), np.float32)},
            ),
            "shape mismatch",
        ),
        (
            memory.MemoryState(
                fast_weights={"w": np.zeros((1, 2), np.float32)},
                momentum={"w": np.zeros((1, 2), np.float64)},
            ),
            "dtype mismatch",
        ),
        (
            memory.MemoryState(
                fast_weights={"w": np.zeros((1, 2), np.int32)},
                momentum={"w": np.zeros((1, 2), np.int32)},
            ),
            "float32",
        ),
        (
            memory.MemoryState(
                fast_weights={"w": np.zeros((1, 2), np.float64)},
                momentum={"w": np.zeros((1, 2), np.float64)},
            ),
            "float32",
        ),
    ],
)
def test_validate_memory_state_rejects_malformed_state(state, error):
    with pytest.raises((TypeError, ValueError), match=error):
        diagnostics.validate_memory_state(state)


def test_finite_validation_is_optional_for_forensic_snapshots():
    state = _numpy_state()
    state.fast_weights["b0"][0, 0] = np.nan
    diagnostics.validate_memory_state(state)
    with pytest.raises(ValueError, match="non-finite"):
        diagnostics.validate_memory_state(state, require_finite=True)


def test_snapshot_clones_inputs_hashes_counter_and_metadata_and_restores_copies():
    state = _numpy_state()
    metadata = {"episode": 7, "intervention": "online", "nested": {"enabled": True}}
    snapshot = diagnostics.create_memory_snapshot(state, writes=11, metadata=metadata)
    diagnostics.validate_memory_snapshot(snapshot)

    state.fast_weights["w0"][0, 0, 0] = 999
    metadata["nested"]["enabled"] = False
    assert snapshot.state.fast_weights["w0"][0, 0, 0] == 0
    assert snapshot.metadata["nested"]["enabled"] is True

    branch_a, writes_a, metadata_a = diagnostics.restore_memory_snapshot(snapshot)
    branch_b, writes_b, metadata_b = diagnostics.restore_memory_snapshot(snapshot)
    assert writes_a == writes_b == 11
    assert metadata_a == metadata_b == snapshot.metadata
    diagnostics.assert_memory_states_equal(branch_a, branch_b)
    diagnostics.assert_memory_states_isolated(branch_a, branch_b)

    branch_a.fast_weights["w0"][0, 0, 0] = -123
    metadata_a["nested"]["enabled"] = False
    assert branch_b.fast_weights["w0"][0, 0, 0] == 0
    assert snapshot.state.fast_weights["w0"][0, 0, 0] == 0
    assert metadata_b["nested"]["enabled"] is True

    with pytest.raises(ValueError, match="envelope hash mismatch"):
        diagnostics.validate_memory_snapshot(dataclasses.replace(snapshot, writes=12))
    changed_metadata = {**snapshot.metadata, "episode": 8}
    with pytest.raises(ValueError, match="envelope hash mismatch"):
        diagnostics.validate_memory_snapshot(dataclasses.replace(snapshot, metadata=changed_metadata))

    tampered_state = diagnostics.clone_memory_state(snapshot.state)
    tampered_state.momentum["w0"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="state hash mismatch"):
        diagnostics.validate_memory_snapshot(dataclasses.replace(snapshot, state=tampered_state))


def test_snapshot_forks_clone_state_and_nested_metadata():
    snapshot = diagnostics.create_memory_snapshot(_numpy_state(), writes=5, metadata={"nested": {"branch": "baseline"}})
    branches = diagnostics.fork_memory_snapshot(snapshot, 2)
    first_state, first_writes, first_metadata = branches[0]
    second_state, second_writes, second_metadata = branches[1]

    assert first_writes == second_writes == 5
    diagnostics.assert_memory_states_isolated(first_state, second_state)
    first_state.fast_weights["b0"][0, 0] = 88
    first_metadata["nested"]["branch"] = "intervention"
    assert second_state.fast_weights["b0"][0, 0] != 88
    assert second_metadata["nested"]["branch"] == "baseline"
    assert snapshot.metadata["nested"]["branch"] == "baseline"


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_npz_round_trip_has_manifest_and_validates_all_state(backend, tmp_path):
    snapshot = diagnostics.create_memory_snapshot(
        _numpy_state(), writes=4, metadata={"episode_id": "ep-004", "tags": ["reveal", "hidden"]}
    )
    path = diagnostics.save_memory_snapshot(tmp_path / "states" / "reveal.npz", snapshot)

    with np.load(path, allow_pickle=False) as archive:
        assert "__manifest__" in archive.files
        manifest = json.loads(archive["__manifest__"].tobytes().decode("utf-8"))
        assert manifest["schema"] == diagnostics.SNAPSHOT_SCHEMA
        assert manifest["version"] == diagnostics.SNAPSHOT_VERSION
        assert manifest["writes"] == 4
        assert {leaf["group"] for leaf in manifest["leaves"]} == {"fast_weights", "momentum"}
        assert len(manifest["leaves"]) == 4

    loaded = diagnostics.load_memory_snapshot(path, backend=backend)
    diagnostics.assert_memory_states_equal(snapshot.state, loaded.state)
    assert loaded.writes == snapshot.writes
    assert loaded.metadata == snapshot.metadata
    assert loaded.state_hash == snapshot.state_hash
    assert loaded.snapshot_hash == snapshot.snapshot_hash
    expected_type = np.ndarray if backend == "numpy" else jax.Array
    assert all(isinstance(leaf, expected_type) for leaf in jax.tree.leaves(loaded.state))


def test_npz_load_rejects_array_and_envelope_tampering(tmp_path):
    snapshot = diagnostics.create_memory_snapshot(_numpy_state(), writes=3, metadata={"phase": "hidden"})
    array_path = diagnostics.save_memory_snapshot(tmp_path / "array_tamper.npz", snapshot)

    def mutate_first_array(arrays):
        array_key = next(key for key in arrays if key.startswith("array_"))
        arrays[array_key].flat[0] += 1

    _rewrite_npz(array_path, mutate_arrays=mutate_first_array)
    with pytest.raises(ValueError, match="state hash mismatch"):
        diagnostics.load_memory_snapshot(array_path)

    envelope_path = diagnostics.save_memory_snapshot(tmp_path / "envelope_tamper.npz", snapshot)
    _rewrite_npz(envelope_path, mutate_manifest=lambda manifest: manifest.__setitem__("writes", 9))
    with pytest.raises(ValueError, match="envelope hash mismatch"):
        diagnostics.load_memory_snapshot(envelope_path)


def test_save_refuses_overwrite_by_default_and_can_replace_explicitly(tmp_path):
    first = diagnostics.create_memory_snapshot(_numpy_state(), writes=1)
    second = diagnostics.create_memory_snapshot(_numpy_state(), writes=2)
    path = diagnostics.save_memory_snapshot(tmp_path / "state.npz", first)

    with pytest.raises(FileExistsError):
        diagnostics.save_memory_snapshot(path, second)
    assert diagnostics.load_memory_snapshot(path).writes == 1

    diagnostics.save_memory_snapshot(path, second, overwrite=True)
    assert diagnostics.load_memory_snapshot(path).writes == 2


def test_metadata_and_branch_count_validation():
    state = _numpy_state()
    with pytest.raises(ValueError, match="positive integer"):
        diagnostics.fork_memory_state(state, 0)
    with pytest.raises(ValueError, match="non-negative integer"):
        diagnostics.create_memory_snapshot(state, writes=True)
    with pytest.raises(ValueError, match="JSON-serializable"):
        diagnostics.create_memory_snapshot(state, writes=0, metadata={"bad": np.nan})
    with pytest.raises(TypeError, match="keys must be strings"):
        diagnostics.create_memory_snapshot(state, writes=0, metadata={1: "bad"})
