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


def test_gradient_accumulation_is_opt_in_and_validated() -> None:
    config = _config.get_config("pi05_yam_mem_v31")
    assert config.gradient_accumulation_steps == 1
    assert dataclasses.replace(config, gradient_accumulation_steps=3).batch_size == 12

    with pytest.raises(ValueError, match="must be positive"):
        dataclasses.replace(config, gradient_accumulation_steps=0)
    with pytest.raises(ValueError, match="must be divisible"):
        dataclasses.replace(config, gradient_accumulation_steps=5)
