# ruff: noqa: SLF001

import json
import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.training import weight_loaders


def _checkpoint_root(tmp_path: pathlib.Path) -> pathlib.Path:
    root = tmp_path / "1000"
    (root / "train_state").mkdir(parents=True)
    return root


def _loader(root: pathlib.Path, **kwargs) -> weight_loaders.AuditedRawCheckpointWeightLoader:
    return weight_loaders.AuditedRawCheckpointWeightLoader(
        checkpoint_path=str(root),
        enabled=True,
        matched_allowlist=(r"model/.*",),
        **kwargs,
    )


def _audit(
    tmp_path: pathlib.Path,
    source: dict,
    target: dict,
    **kwargs,
) -> weight_loaders.AuditedGraftResult:
    root = _checkpoint_root(tmp_path)
    loader = _loader(root, **kwargs)
    return weight_loaders._audit_and_graft(source, target, loader, root)


def test_audited_loader_is_default_off() -> None:
    loader = weight_loaders.AuditedRawCheckpointWeightLoader(checkpoint_path="does-not-exist")

    with pytest.raises(weight_loaders.AuditedGraftError, match="default-off"):
        loader.load({"model": {"kernel": np.zeros((1,), dtype=np.float32)}})


def test_orbax_restore_uses_raw_train_state_and_never_standalone_ema(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = {"model": {"kernel": np.asarray([1.0, 2.0], dtype=np.float32)}}
    ema = {"model": {"kernel": np.asarray([91.0, 92.0], dtype=np.float32)}}
    root = tmp_path / "1000"
    (root / "train_state").mkdir(parents=True)
    (root / "params").mkdir()
    calls: list[tuple[str, pathlib.Path]] = []

    class FakeTreeMetadata:
        """Orbax 0.11 metadata is indexable but is deliberately not a Mapping."""

        def __init__(self, tree):
            self.tree = tree

    class FakeCheckpointer:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def metadata(self, path):
            calls.append(("metadata", pathlib.Path(path)))
            return FakeTreeMetadata(
                {
                    "params": {
                        "model": {"kernel": {"value": jax.ShapeDtypeStruct(raw["model"]["kernel"].shape, np.float32)}}
                    },
                    "ema_params": {"must_not_restore": jax.ShapeDtypeStruct(ema["model"]["kernel"].shape, np.float32)},
                }
            )

        def restore(self, path, args):
            calls.append(("restore", pathlib.Path(path)))
            assert args.transforms_default_to_original is False
            assert set(args.transforms) == {r"params/(.*)"}
            assert args.transforms[r"params/(.*)"].original_key == r"params/\1/value"
            return {"params": raw}

    monkeypatch.setattr(weight_loaders.ocp, "PyTreeCheckpointer", FakeCheckpointer)
    manifest_path = tmp_path / "graft_manifest.json"
    loader = _loader(root, manifest_output_path=str(manifest_path))

    result = loader.load_with_manifest({"model": {"kernel": np.zeros((2,), dtype=np.float32)}})

    np.testing.assert_array_equal(result.params["model"]["kernel"], raw["model"]["kernel"])
    assert not np.array_equal(result.params["model"]["kernel"], ema["model"]["kernel"])
    assert result.manifest.parameter_source.startswith("train_state/params/*/value")
    assert result.manifest.standalone_params_present
    assert calls == [("metadata", root / "train_state"), ("restore", root / "train_state")]
    serialized = json.loads(manifest_path.read_text())
    assert serialized["counts"] == {
        "fresh_initialized": 0,
        "ignored_source": 0,
        "matched": 1,
        "reset": 0,
    }
    assert len(serialized["manifest_sha256"]) == 64


def test_direct_train_state_or_standalone_params_path_is_rejected(tmp_path: pathlib.Path) -> None:
    root = tmp_path / "1000"
    (root / "train_state").mkdir(parents=True)
    (root / "params").mkdir()

    for item_name in ("train_state", "params"):
        loader = weight_loaders.AuditedRawCheckpointWeightLoader(
            checkpoint_path=str(root / item_name),
            enabled=True,
            matched_allowlist=(r"model/.*",),
        )
        with pytest.raises(weight_loaders.AuditedGraftError, match="checkpoint step root"):
            loader.load({"model": {"kernel": np.zeros((1,), dtype=np.float32)}})


def test_explicit_matched_fresh_and_ignored_allowlists(tmp_path: pathlib.Path) -> None:
    source = {
        "model": {"backbone": np.asarray([1.0, 2.0], dtype=np.float32)},
        "old": {"head": np.asarray([3.0], dtype=np.float32)},
    }
    fresh_shape = jax.ShapeDtypeStruct((2, 2), np.float32)
    target = {
        "model": {
            "backbone": jax.ShapeDtypeStruct((2,), np.float32),
            "write_head": fresh_shape,
        }
    }
    root = _checkpoint_root(tmp_path)
    loader = weight_loaders.AuditedRawCheckpointWeightLoader(
        checkpoint_path=str(root),
        enabled=True,
        matched_allowlist=(r"model/backbone",),
        fresh_init_allowlist=(r"model/write_head",),
        ignored_source_allowlist=(r"old/head",),
    )

    result = weight_loaders._audit_and_graft(source, target, loader, root)

    np.testing.assert_array_equal(result.params["model"]["backbone"], source["model"]["backbone"])
    assert result.params["model"]["write_head"] is fresh_shape
    assert [leaf.path for leaf in result.manifest.matched] == ["model/backbone"]
    assert [leaf.path for leaf in result.manifest.fresh_initialized] == ["model/write_head"]
    assert [leaf.path for leaf in result.manifest.ignored_source] == ["old/head"]
    assert result.manifest.fresh_initialized[0].value_sha256 is None
    json.dumps(result.manifest.to_dict(), allow_nan=False)


def test_structural_none_fresh_leaf_is_explicitly_audited_and_hashable(tmp_path: pathlib.Path) -> None:
    root = _checkpoint_root(tmp_path)
    loader = weight_loaders.AuditedRawCheckpointWeightLoader(
        checkpoint_path=str(root),
        enabled=True,
        matched_allowlist=(r"model/backbone",),
        fresh_init_allowlist=(r"model/new_head",),
    )

    result = weight_loaders._audit_and_graft(
        {"model": {"backbone": np.zeros((1,), dtype=np.float32)}},
        {"model": {"backbone": np.zeros((1,), dtype=np.float32), "new_head": None}},
        loader,
        root,
    )

    assert result.params["model"]["new_head"] is None
    assert result.manifest.fresh_initialized[0].path == "model/new_head"
    assert result.manifest.fresh_initialized[0].shape == ()
    assert result.manifest.fresh_initialized[0].dtype == "none"
    assert result.manifest.fresh_initialized[0].value_sha256 is None
    assert weight_loaders.parameter_tree_sha256(result.params)


def test_malformed_fresh_leaf_error_names_exact_path(tmp_path: pathlib.Path) -> None:
    root = _checkpoint_root(tmp_path)
    loader = weight_loaders.AuditedRawCheckpointWeightLoader(
        checkpoint_path=str(root),
        enabled=True,
        matched_allowlist=(r"model/backbone",),
        fresh_init_allowlist=(r"model/new_head",),
    )

    with pytest.raises(weight_loaders.AuditedGraftError, match="parameter leaf 'model/new_head'.*object"):
        weight_loaders._audit_and_graft(
            {"model": {"backbone": np.zeros((1,), dtype=np.float32)}},
            {"model": {"backbone": np.zeros((1,), dtype=np.float32), "new_head": object()}},
            loader,
            root,
        )


@pytest.mark.parametrize(
    ("source", "target", "loader_kwargs", "match"),
    [
        (
            {"model": {"backbone": np.zeros((1,), dtype=np.float32)}, "old": {"extra": np.zeros((1,))}},
            {"model": {"backbone": np.zeros((1,), dtype=np.float32)}},
            {},
            "unexpected ignored-source leaves",
        ),
        (
            {"model": {"backbone": np.zeros((1,), dtype=np.float32)}},
            {
                "model": {
                    "backbone": np.zeros((1,), dtype=np.float32),
                    "new_head": np.zeros((1,), dtype=np.float32),
                }
            },
            {},
            "unexpected fresh-init leaves",
        ),
        (
            {"model": {"backbone": np.zeros((1,), dtype=np.float32)}},
            {"model": {"backbone": np.zeros((1,), dtype=np.float32)}},
            {"matched_allowlist": ()},
            "unexpected matched leaves",
        ),
        (
            {"model": {"backbone": np.zeros((1,), dtype=np.float32)}},
            {"model": {"backbone": np.zeros((1,), dtype=np.float32)}},
            {"fresh_init_allowlist": (r"model/typo_head",)},
            "fresh-init allowlist patterns matched no leaves",
        ),
    ],
)
def test_unexpected_or_stale_allowlist_rules_fail_closed(
    tmp_path: pathlib.Path,
    source: dict,
    target: dict,
    loader_kwargs: dict,
    match: str,
) -> None:
    root = _checkpoint_root(tmp_path)
    base_kwargs = {"matched_allowlist": (r"model/.*",)}
    base_kwargs.update(loader_kwargs)
    loader = weight_loaders.AuditedRawCheckpointWeightLoader(checkpoint_path=str(root), enabled=True, **base_kwargs)

    with pytest.raises(weight_loaders.AuditedGraftError, match=match):
        weight_loaders._audit_and_graft(source, target, loader, root)


@pytest.mark.parametrize(
    ("source_leaf", "target_leaf", "match"),
    [
        (np.zeros((2,), dtype=np.float32), np.zeros((3,), dtype=np.float32), "shape mismatch"),
        (np.zeros((2,), dtype=np.float16), np.zeros((2,), dtype=np.float32), "dtype mismatch"),
    ],
)
def test_matched_shape_and_dtype_must_be_exact(
    tmp_path: pathlib.Path, source_leaf: np.ndarray, target_leaf: np.ndarray, match: str
) -> None:
    with pytest.raises(weight_loaders.AuditedGraftError, match=match):
        _audit(
            tmp_path,
            {"model": {"backbone": source_leaf}},
            {"model": {"backbone": target_leaf}},
        )


def test_bfloat16_dtype_identity_is_preserved_in_audit_and_hash(tmp_path: pathlib.Path) -> None:
    source = {"model": {"backbone": np.asarray([1.0], dtype=jnp.bfloat16)}}
    target = {"model": {"backbone": jax.ShapeDtypeStruct((1,), jnp.bfloat16)}}

    result = _audit(tmp_path, source, target)

    assert result.manifest.matched[0].dtype == "bfloat16"
    assert result.manifest.source_tree_sha256


def test_conditional_memory_inject_w_reset_is_audited(tmp_path: pathlib.Path) -> None:
    source = {
        "model": {
            "backbone": np.asarray([1.0], dtype=np.float32),
            "memory_inject_w": np.zeros((4,), dtype=np.float32),
        }
    }
    target = {
        "model": {
            "backbone": np.zeros((1,), dtype=np.float32),
            "memory_inject_w": np.zeros((4,), dtype=np.float32),
        }
    }

    result = _audit(
        tmp_path,
        source,
        target,
        memory_inject_w_path="model/memory_inject_w",
        reset_memory_inject_w_if_closed_fraction_gt=0.5,
    )

    expected = np.full((4,), np.arctanh(0.5), dtype=np.float32)
    np.testing.assert_allclose(result.params["model"]["memory_inject_w"], expected)
    assert result.manifest.conditional_reset is not None
    assert result.manifest.conditional_reset.applied
    assert result.manifest.conditional_reset.observed_closed_fraction == 1.0
    assert [leaf.path for leaf in result.manifest.reset] == ["model/memory_inject_w"]
    assert [leaf.path for leaf in result.manifest.matched] == ["model/backbone"]


def test_conditional_memory_inject_w_keeps_an_open_raw_gate(tmp_path: pathlib.Path) -> None:
    open_parameter = np.full((4,), np.arctanh(0.5), dtype=np.float32)
    source = {"model": {"memory_inject_w": open_parameter}}
    target = {"model": {"memory_inject_w": np.zeros((4,), dtype=np.float32)}}

    result = _audit(
        tmp_path,
        source,
        target,
        memory_inject_w_path="model/memory_inject_w",
        reset_memory_inject_w_if_closed_fraction_gt=0.5,
    )

    np.testing.assert_array_equal(result.params["model"]["memory_inject_w"], open_parameter)
    assert result.manifest.conditional_reset is not None
    assert not result.manifest.conditional_reset.applied
    assert not result.manifest.reset
    assert [leaf.path for leaf in result.manifest.matched] == ["model/memory_inject_w"]


def test_tree_hashes_are_order_stable_and_fresh_values_are_schema_only(tmp_path: pathlib.Path) -> None:
    source_a = {
        "model": {
            "z": np.asarray([3.0], dtype=np.float32),
            "a": np.asarray([1.0, 2.0], dtype=np.float32),
        }
    }
    source_b = {
        "model": {
            "a": np.asarray([1.0, 2.0], dtype=np.float32),
            "z": np.asarray([3.0], dtype=np.float32),
        }
    }
    target_a = {
        "model": {
            "z": np.zeros((1,), dtype=np.float32),
            "a": np.zeros((2,), dtype=np.float32),
            "new_head": np.asarray([11.0], dtype=np.float32),
        }
    }
    target_b = {
        "model": {
            "new_head": np.asarray([99.0], dtype=np.float32),
            "a": np.zeros((2,), dtype=np.float32),
            "z": np.zeros((1,), dtype=np.float32),
        }
    }
    first = _audit(tmp_path / "first", source_a, target_a, fresh_init_allowlist=(r"model/new_head",))
    second = _audit(tmp_path / "second", source_b, target_b, fresh_init_allowlist=(r"model/new_head",))

    assert first.manifest.source_tree_sha256 == second.manifest.source_tree_sha256
    assert first.manifest.target_schema_sha256 == second.manifest.target_schema_sha256
    assert first.manifest.graft_tree_sha256 == second.manifest.graft_tree_sha256

    changed = _audit(
        tmp_path / "changed",
        {"model": {"a": np.asarray([1.0, 9.0], dtype=np.float32), "z": np.asarray([3.0], dtype=np.float32)}},
        target_a,
        fresh_init_allowlist=(r"model/new_head",),
    )
    assert changed.manifest.source_tree_sha256 != first.manifest.source_tree_sha256
    assert changed.manifest.graft_tree_sha256 != first.manifest.graft_tree_sha256


def test_manifest_write_is_idempotent_but_refuses_different_content(tmp_path: pathlib.Path) -> None:
    result = _audit(
        tmp_path / "source",
        {"model": {"backbone": np.asarray([1.0], dtype=np.float32)}},
        {"model": {"backbone": np.zeros((1,), dtype=np.float32)}},
    )
    manifest_path = tmp_path / "manifest.json"

    weight_loaders._write_manifest(manifest_path, result.manifest)
    weight_loaders._write_manifest(manifest_path, result.manifest)

    changed = _audit(
        tmp_path / "other",
        {"model": {"backbone": np.asarray([2.0], dtype=np.float32)}},
        {"model": {"backbone": np.zeros((1,), dtype=np.float32)}},
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        weight_loaders._write_manifest(manifest_path, changed.manifest)


def test_partial_checkpoint_loader_retains_legacy_permissive_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = {
        "model": {
            "shared": np.asarray([1.0], dtype=np.float16),
            "source_only": np.asarray([2.0], dtype=np.float32),
        }
    }
    target = {
        "model": {
            "shared": np.asarray([0.0], dtype=np.float32),
            "fresh": np.asarray([3.0], dtype=np.float32),
        }
    }
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)

    result = weight_loaders.PartialCheckpointWeightLoader("legacy").load(target)

    assert set(result["model"]) == {"shared", "fresh"}
    np.testing.assert_array_equal(result["model"]["shared"], np.asarray([1.0], dtype=np.float32))
    np.testing.assert_array_equal(result["model"]["fresh"], target["model"]["fresh"])


def test_audited_partial_loader_uses_standalone_params_and_records_manifest(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = {"model": {"shared": np.asarray([1.0], dtype=np.float32)}}
    target = {
        "model": {
            "shared": jax.ShapeDtypeStruct((1,), np.float32),
            "memory_head": jax.ShapeDtypeStruct((2,), np.float32),
        }
    }
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: tmp_path / "official-params")
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *_args, **_kwargs: source)
    manifest_path = tmp_path / "initialization.json"
    loader = weight_loaders.AuditedPartialCheckpointWeightLoader(
        "gs://official/params",
        matched_allowlist=(r"model/shared",),
        fresh_init_allowlist=(r"model/memory_head",),
        manifest_output_path=str(manifest_path),
    )

    result = loader.load_with_manifest(target)

    np.testing.assert_array_equal(result.params["model"]["shared"], source["model"]["shared"])
    assert result.params["model"]["memory_head"] is target["model"]["memory_head"]
    payload = json.loads(manifest_path.read_text())
    assert payload["parameter_source"] == "standalone params artifact: gs://official/params"
    assert payload["checkpoint_root"] == "gs://official/params"
    assert payload["counts"]["matched"] == 1
    assert payload["counts"]["fresh_initialized"] == 1


def test_audited_partial_loader_rejects_missing_or_unexpected_shared_leaf(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: tmp_path / "official-params")
    monkeypatch.setattr(
        weight_loaders._model,
        "restore_params",
        lambda *_args, **_kwargs: {"model": {"unexpected_memory_leaf": np.ones((1,), dtype=np.float32)}},
    )
    loader = weight_loaders.AuditedPartialCheckpointWeightLoader(
        "gs://official/params",
        matched_allowlist=(r"model/shared",),
        fresh_init_allowlist=(r"model/memory_.*",),
        manifest_output_path=str(tmp_path / "manifest.json"),
    )

    with pytest.raises(weight_loaders.AuditedGraftError, match="unexpected matched leaves"):
        loader.load({"model": {"unexpected_memory_leaf": np.zeros((1,), dtype=np.float32)}})

    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda *_args, **_kwargs: {})
    with pytest.raises(weight_loaders.AuditedGraftError, match="matched allowlist patterns matched no leaves"):
        loader.load({"model": {"shared": np.zeros((1,), dtype=np.float32)}})
