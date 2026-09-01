from pathlib import Path
import sys

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v35_gate_artifacts as artifacts
    import v35_side_prototypes as prototypes
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


def _manifest(tmp_path: Path) -> artifacts.FrozenManifest:
    episodes = tuple(
        artifacts.ManifestEpisode(
            stable_id=f"train/{index:02d}",
            episode_index=index,
            collection="0831",
            object_name="banana",
            part="",
            target_side=index % 2,
            split="train",
        )
        for index in range(54)
    ) + tuple(
        artifacts.ManifestEpisode(
            stable_id=f"final/{index}",
            episode_index=74 + index,
            collection="0830",
            object_name="banana",
            part="part2",
            target_side=index % 2,
            split="final_test",
        )
        for index in range(8)
    )
    return artifacts.FrozenManifest(
        path=tmp_path / "manifest.json",
        sha256="1" * 64,
        split_assignment_sha256="2" * 64,
        episodes=episodes,
    )


def _arrays(manifest: artifacts.FrozenManifest) -> dict[str, np.ndarray]:
    ids = [episode.stable_id for episode in manifest.split("train")]
    ordinal = np.repeat(np.arange(54, dtype=np.int16), 2)
    frame = np.tile(np.asarray([15, 30], dtype=np.int32), 54) + np.repeat(np.arange(54, dtype=np.int32) * 60, 2)
    side = np.asarray([1.0 if manifest.split("train")[i].target_side else -1.0 for i in ordinal], dtype=np.float32)
    natural = np.stack([side, np.ones_like(side)], axis=1).astype(np.float32)
    return {
        "schema_version": np.asarray(prototypes.RAW_SCHEMA_VERSION),
        "episode_stable_id": np.asarray(ids),
        "frame_episode_ordinal": ordinal,
        "frame_index": frame,
        "natural_vbar": natural,
        "counterfactual_vbar": -natural,
    }


def test_episode_first_prototypes_and_loo_exclude_current_episode(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    natural, counter = prototypes.reduce_episode_vbars(_arrays(manifest), manifest=manifest)
    assert natural.shape == counter.shape == (54, 2)
    sides = np.asarray([episode.target_side for episode in manifest.split("train")], dtype=np.int8)
    directions = prototypes.side_directions(natural, sides)
    assert directions.dtype == np.float32
    assert np.allclose(np.linalg.norm(directions, axis=1), 1.0)

    changed = natural.copy()
    changed[0] = np.asarray([-1000.0, 500.0], dtype=np.float32)
    loo = prototypes.leave_one_episode_out_directions(changed, sides)
    expected_left = prototypes.side_directions(changed[1:], sides[1:])[0]
    assert np.allclose(loo[0, 0], expected_left)


def test_prototype_reducer_rejects_final_test_identity_and_zero_episode(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    arrays = _arrays(manifest)
    contaminated = dict(arrays)
    ids = arrays["episode_stable_id"].copy()
    ids[-1] = "final/0"
    contaminated["episode_stable_id"] = ids
    with pytest.raises(prototypes.SidePrototypeError, match="train order"):
        prototypes.reduce_episode_vbars(contaminated, manifest=manifest)

    missing = dict(arrays)
    keep = arrays["frame_episode_ordinal"] != 4
    for key in ("frame_episode_ordinal", "frame_index", "natural_vbar", "counterfactual_vbar"):
        missing[key] = arrays[key][keep]
    with pytest.raises(prototypes.SidePrototypeError, match="zero eligible"):
        prototypes.reduce_episode_vbars(missing, manifest=manifest)
