# ruff: noqa: SLF001

import copy
import importlib.util
import pathlib
import sys
import types
from unittest import mock

import numpy as np
import pytest

import openpi.models.rtc as _rtc


def _module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _load_server_module() -> types.ModuleType:
    """Loads only the server class, stubbing checkpoint/server dependencies unused by these tests."""
    model_module = _module("openpi.models.model")
    tokenizer_module = _module("openpi.models.tokenizer")
    models_package = _module(
        "openpi.models",
        model=model_module,
        rtc=_rtc,
        tokenizer=tokenizer_module,
    )
    models_package.__path__ = []
    policy_module = _module("openpi.policies.policy", Policy=type("Policy", (), {}))
    policies_package = _module("openpi.policies", policy=policy_module)
    policies_package.__path__ = []
    websocket_module = _module("openpi.serving.websocket_policy_server")
    serving_package = _module("openpi.serving", websocket_policy_server=websocket_module)
    serving_package.__path__ = []
    checkpoints_module = _module("openpi.training.checkpoints")
    config_module = _module("openpi.training.config")
    training_package = _module(
        "openpi.training",
        checkpoints=checkpoints_module,
        config=config_module,
    )
    training_package.__path__ = []
    download_module = _module("openpi.shared.download")
    nnx_utils_module = _module("openpi.shared.nnx_utils")
    shared_package = _module(
        "openpi.shared",
        download=download_module,
        nnx_utils=nnx_utils_module,
    )
    shared_package.__path__ = []
    transforms_module = _module("openpi.transforms")
    openpi_package = _module(
        "openpi",
        models=models_package,
        policies=policies_package,
        serving=serving_package,
        shared=shared_package,
        training=training_package,
        transforms=transforms_module,
    )
    openpi_package.__path__ = []
    stubs = {
        "tyro": _module("tyro"),
        "openpi": openpi_package,
        "openpi.models": models_package,
        "openpi.models.model": model_module,
        "openpi.models.rtc": _rtc,
        "openpi.models.tokenizer": tokenizer_module,
        "openpi.policies": policies_package,
        "openpi.policies.policy": policy_module,
        "openpi.serving": serving_package,
        "openpi.serving.websocket_policy_server": websocket_module,
        "openpi.shared": shared_package,
        "openpi.shared.download": download_module,
        "openpi.shared.nnx_utils": nnx_utils_module,
        "openpi.training": training_package,
        "openpi.training.checkpoints": checkpoints_module,
        "openpi.training.config": config_module,
        "openpi.transforms": transforms_module,
    }

    path = pathlib.Path(__file__).with_name("serve_yam_memory.py")
    spec = importlib.util.spec_from_file_location("_serve_yam_memory_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {**stubs, spec.name: module}):
        spec.loader.exec_module(module)
    return module


serve_yam_memory = _load_server_module()


def _policy(*, simulated_delay: int | None = 2) -> serve_yam_memory.MemoryPolicy:
    policy = object.__new__(serve_yam_memory.MemoryPolicy)
    policy._action_horizon = 4
    policy._action_dim = 3
    policy._raw_action_dim = 2
    policy._simulated_delay = simulated_delay

    def transform(inputs: dict) -> dict:
        # Stand in for the server's DeltaActions -> Normalize -> PadStatesAndActions path.
        state = np.asarray(inputs["observation/state"])
        actions = (np.asarray(inputs["actions"]) - state[None, :]) / 2.0
        return {"actions": np.pad(actions, ((0, 0), (0, 1)))}

    policy._input_transform = transform
    return policy


def _prefix() -> dict:
    return {
        "actions": np.asarray([[2.0, 6.0], [4.0, 8.0], [6.0, 10.0], [0.0, 0.0]], dtype=np.float32),
        "delay": 2,
        "prefix_length": 3,
    }


def test_server_metadata_exposes_v31_config_and_write_source() -> None:
    train_config = types.SimpleNamespace(
        name="pi05_yam_mem_v31",
        model=types.SimpleNamespace(
            memory_architecture="v3_v31",
            memory_write_source="post_attention",
            action_horizon=50,
        ),
        # Runtime semantics must override stale or user-supplied policy metadata.
        policy_metadata={"config_name": "stale", "memory_write_source": "stale", "custom": "kept"},
    )
    data_config = types.SimpleNamespace(memory_stride_frames=10)

    metadata = serve_yam_memory._build_server_metadata(train_config, data_config, simulated_delay=6)

    assert metadata == {
        "config_name": "pi05_yam_mem_v31",
        "memory_architecture": "v3_v31",
        "memory_write_source": "post_attention",
        "memory_query_tokens": None,
        "action_horizon": 50,
        "rtc_enabled": True,
        "rtc_max_delay": 6,
        "rtc_delay_semantics": "inclusive_max",
        "memory_stride_frames": 10,
        "custom": "kept",
    }


def test_server_metadata_marks_legacy_write_source_and_disabled_rtc() -> None:
    train_config = types.SimpleNamespace(
        name="legacy_memory_config",
        model=types.SimpleNamespace(action_horizon=50),
        policy_metadata=None,
    )
    data_config = types.SimpleNamespace(memory_stride_frames=6)

    metadata = serve_yam_memory._build_server_metadata(train_config, data_config, simulated_delay=None)

    assert metadata["memory_write_source"] == "raw_hidden"
    assert metadata["memory_architecture"] == "v3_v31"
    assert metadata["rtc_enabled"] is False
    assert metadata["rtc_max_delay"] is None


def test_server_metadata_exposes_v32_dual_query_contract() -> None:
    train_config = types.SimpleNamespace(
        name="pi05_yam_mem_v32",
        model=types.SimpleNamespace(
            memory_architecture="v32_layer8_dual_query",
            memory_write_source="query_compressed",
            memory_query_tokens=16,
            action_horizon=50,
        ),
        policy_metadata=None,
    )
    metadata = serve_yam_memory._build_server_metadata(
        train_config, types.SimpleNamespace(memory_stride_frames=10), simulated_delay=6
    )

    assert metadata["memory_architecture"] == "v32_layer8_dual_query"
    assert metadata["memory_write_source"] == "query_compressed"
    assert metadata["memory_query_tokens"] == 16
    assert metadata["rtc_enabled"] is True
    assert metadata["rtc_max_delay"] == 6


def test_prepare_action_prefix_transforms_and_batches_without_mutating_request() -> None:
    policy = _policy()
    prefix = _prefix()
    original = copy.deepcopy(prefix)

    result = policy._prepare_action_prefix(
        {"observation/state": np.asarray([2.0, 4.0], dtype=np.float32)},
        prefix,
    )

    assert isinstance(result, _rtc.ActionPrefix)
    assert result.actions.shape == (1, 4, 3)
    np.testing.assert_allclose(
        result.actions,
        np.asarray([[[0.0, 1.0, 0.0], [1.0, 2.0, 0.0], [2.0, 3.0, 0.0], [-1.0, -2.0, 0.0]]]),
    )
    np.testing.assert_array_equal(result.delay, np.asarray([2], dtype=np.int32))
    np.testing.assert_array_equal(result.prefix_length, np.asarray([3], dtype=np.int32))
    np.testing.assert_array_equal(prefix["actions"], original["actions"])
    assert prefix["delay"] == original["delay"]
    assert prefix["prefix_length"] == original["prefix_length"]


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("delay", 1.0, "integer scalar"),
        ("delay", True, "integer scalar"),
        ("delay", 4, "0 <= delay"),
        ("prefix_length", 5, "0 <= delay"),
        ("delay", 3, "inclusive RTC maximum"),
    ],
)
def test_prepare_action_prefix_rejects_invalid_bounds_and_scalars(field: str, value: object, match: str) -> None:
    prefix = _prefix()
    prefix[field] = value

    with pytest.raises(ValueError, match=match):
        _policy()._prepare_action_prefix(
            {"observation/state": np.asarray([2.0, 4.0], dtype=np.float32)},
            prefix,
        )


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((3, 2), dtype=np.float32),
        np.full((4, 2), np.nan, dtype=np.float32),
        np.full((4, 2), "not-an-action", dtype=object),
    ],
)
def test_prepare_action_prefix_rejects_bad_action_payloads(actions: np.ndarray) -> None:
    prefix = _prefix()
    prefix["actions"] = actions

    with pytest.raises(ValueError, match="action_prefix.actions"):
        _policy()._prepare_action_prefix(
            {"observation/state": np.asarray([2.0, 4.0], dtype=np.float32)},
            prefix,
        )


def test_prepare_action_prefix_requires_rtc_enabled_checkpoint() -> None:
    with pytest.raises(ValueError, match="configured with simulated_delay"):
        _policy(simulated_delay=None)._prepare_action_prefix(
            {"observation/state": np.asarray([2.0, 4.0], dtype=np.float32)},
            _prefix(),
        )
