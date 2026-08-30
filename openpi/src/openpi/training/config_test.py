import dataclasses

import pytest

from openpi.training import config as _config
from openpi.training import weight_loaders

_PI05_BASE_PARAMS = "gs://openpi-assets/checkpoints/pi05_base/params"


def test_yam_v3_and_v31_start_from_the_same_pi05_base_parameters() -> None:
    v3 = _config.get_config("pi05_yam_mem_v3")
    v31 = _config.get_config("pi05_yam_mem_v31")

    for train_config in (v3, v31):
        assert isinstance(train_config.weight_loader, weight_loaders.PartialCheckpointWeightLoader)
        assert train_config.weight_loader.params_path == _PI05_BASE_PARAMS
        assert train_config.model.memory_seq_steps == 60
        assert train_config.model.memory_probe_weight == 0.0
        assert train_config.model.memory_probe_diagnostic is False
        assert train_config.data.base_config.memory_sequence_buckets == (20, 40, 60)

    # The controlled writer ablation must remain identical in every other field.
    normalized_v31 = dataclasses.replace(
        v31,
        name=v3.name,
        model=dataclasses.replace(v31.model, memory_write_source="raw_hidden"),
        # v3.1 intentionally checkpoints every 500 steps after the first step-1000 save;
        # checkpoint cadence is operational state, not part of the writer ablation.
        save_interval=v3.save_interval,
    )
    assert normalized_v31 == v3


def test_yam_v32_is_a_fresh_base_initialized_dual_query_experiment() -> None:
    v32 = _config.get_config("pi05_yam_mem_v32")

    assert isinstance(v32.weight_loader, weight_loaders.PartialCheckpointWeightLoader)
    assert v32.weight_loader.params_path == _PI05_BASE_PARAMS
    assert v32.model.memory_architecture == "v32_layer8_dual_query"
    assert v32.model.memory_layer == 8
    assert v32.model.memory_write_source == "query_compressed"
    assert v32.model.memory_query_tokens == 16
    assert v32.model.memory_query_heads == 8
    assert v32.model.max_token_len == 80
    assert v32.model.causal_token_len == 128
    assert v32.model.bf16_vocab_projection is True
    assert v32.model.memory_seq_steps == 40
    assert v32.model.memory_probe_weight == 0.0
    assert v32.model.memory_probe_diagnostic is False
    assert v32.gradient_accumulation_steps == 1
    assert v32.save_interval == 250
    assert v32.data.base_config.memory_stride_frames == 15
    assert v32.data.base_config.memory_min_slice_steps == 14
    assert v32.data.base_config.memory_sequence_buckets == (14, 27, 40)
    last_observation_frame = (v32.model.memory_seq_steps - 1) * v32.data.base_config.memory_stride_frames
    assert last_observation_frame == 585
    assert last_observation_frame + v32.model.action_horizon - 1 == 634
    assert not v32.freeze_filter(("PaliGemma", "img", "encoder", "kernel"), object())
    assert v32.freeze_filter(("memory", "gate", "kernel"), object())
    assert not v32.freeze_filter(("read_query_compressor", "key_proj", "kernel"), object())

    for config_name in ("pi05_yam_mem_v3", "pi05_yam_mem_v31"):
        assert _config.get_config(config_name).model.bf16_vocab_projection is False
    assert _config.get_config("pi05_yam_mem_v2").model.max_token_len == 200


def test_gradient_accumulation_is_opt_in_and_validated() -> None:
    config = _config.get_config("pi05_yam_mem_v31")
    assert config.gradient_accumulation_steps == 1
    assert dataclasses.replace(config, gradient_accumulation_steps=3).batch_size == 12

    with pytest.raises(ValueError, match="must be positive"):
        dataclasses.replace(config, gradient_accumulation_steps=0)
    with pytest.raises(ValueError, match="must be divisible"):
        dataclasses.replace(config, gradient_accumulation_steps=5)


def test_v34_run4_differs_only_by_identity_and_blank_memory_output() -> None:
    v34 = _config.get_config("pi05_yam_mem_v34")
    run4 = _config.get_config("pi05_yam_mem_v34_run4")

    assert isinstance(run4.weight_loader, weight_loaders.PartialCheckpointWeightLoader)
    assert run4.weight_loader.params_path == _PI05_BASE_PARAMS
    assert v34.model.memory.blank_initial_output is False
    assert run4.model.memory.blank_initial_output is True
    assert not run4.freeze_filter(("memory", "m0", "w3"), object())
    normalized = dataclasses.replace(
        run4,
        name=v34.name,
        model=dataclasses.replace(
            run4.model,
            memory=dataclasses.replace(run4.model.memory, blank_initial_output=False),
        ),
    )
    assert normalized == v34


def test_v34_run5_differs_from_run4_only_by_identity_and_eta_scale() -> None:
    run4 = _config.get_config("pi05_yam_mem_v34_run4")
    run5 = _config.get_config("pi05_yam_mem_v34_run5_eta0")

    assert isinstance(run5.weight_loader, weight_loaders.PartialCheckpointWeightLoader)
    assert run5.weight_loader.params_path == _PI05_BASE_PARAMS
    assert run5.model.memory.blank_initial_output is True
    assert run4.model.memory.eta_scale == 1.0
    assert run5.model.memory.eta_scale == 0.0
    normalized = dataclasses.replace(
        run5,
        name=run4.name,
        model=dataclasses.replace(
            run5.model,
            memory=dataclasses.replace(run5.model.memory, eta_scale=run4.model.memory.eta_scale),
        ),
    )
    assert normalized == run4
