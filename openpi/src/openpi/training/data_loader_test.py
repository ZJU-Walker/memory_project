# ruff: noqa: I001, SLF001

import dataclasses
import functools

import jax
import numpy as np
import pytest
import torch

from openpi.training import data_loader as _data_loader
from openpi.models import pi0_config
from openpi.training import config as _config


def _sequence_item(valid_steps: int, *, max_steps: int = 60, offset: int = 0) -> dict:
    step = np.arange(max_steps) + offset
    mask = np.arange(max_steps) < valid_steps
    return {
        "image": {"base_0_rgb": np.broadcast_to(step[:, None, None, None], (max_steps, 2, 2, 3))},
        "image_mask": {"base_0_rgb": np.ones(max_steps, dtype=bool)},
        "state": np.broadcast_to(step[:, None], (max_steps, 3)),
        "actions": np.broadcast_to(step[:, None, None], (max_steps, 5, 3)),
        "tokenized_prompt": np.broadcast_to(step[:, None], (max_steps, 4)),
        "tokenized_prompt_mask": np.ones((max_steps, 4), dtype=bool),
        "token_ar_mask": np.zeros((max_steps, 4), dtype=np.int32),
        "token_loss_mask": np.zeros((max_steps, 4), dtype=bool),
        "token_fast_mask": np.zeros((max_steps, 4), dtype=bool),
        "tokenized_causal": np.broadcast_to(step[:, None], (max_steps, 6)),
        "tokenized_causal_mask": np.ones((max_steps, 6), dtype=bool),
        "causal_fast_mask": np.zeros((max_steps, 6), dtype=bool),
        "seq_step_mask": mask,
        "seq_block_boundary": np.zeros(max_steps, dtype=bool),
        "seq_probe_labels": np.zeros(max_steps, dtype=np.int32),
        "seq_probe_mask": mask,
        "seq_probe_visible": mask,
        # v3.4 fields: per-step supervision is trimmed with the time axis; the per-SEGMENT
        # scalars (seq_state_masked, seq_side_label) are deliberately absent from the
        # registered temporal keys and pass through the collate untouched.
        "token_state_mask": np.zeros((max_steps, 4), dtype=bool),
        "seq_subtask_class": np.zeros(max_steps, dtype=np.int32),
        "seq_evidence_mask": mask,
        "seq_waiting_mask": np.zeros(max_steps, dtype=bool),
        "seq_state_masked": np.bool_(),
        "seq_side_label": np.int32(1),
    }


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


@pytest.mark.parametrize(
    ("valid_steps", "expected"),
    [(1, 0), (20, 0), (21, 1), (40, 1), (41, 2), (60, 2)],
)
def test_sequence_bucket_ids_round_up_without_truncating(valid_steps: int, expected: int):
    result = _data_loader._sequence_bucket_ids(np.asarray([valid_steps]), (20, 40, 60))
    np.testing.assert_array_equal(result, [expected])


@pytest.mark.parametrize("buckets", [(0, 60), (40, 20, 60), (20, 20, 60), (20, 40)])
def test_sequence_bucket_validation_rejects_invalid_config(buckets: tuple[int, ...]):
    with pytest.raises(ValueError, match="memory_sequence_buckets|final"):
        _data_loader._validate_sequence_buckets(buckets, 60)


def test_sequence_bucket_sampler_is_homogeneous_deterministic_and_preserves_marginals():
    weights = np.asarray([0.0, 0.05, 0.15, 0.30, 0.50])
    valid_steps = np.asarray([10, 20, 35, 45, 60])

    def make_sampler(seed: int):
        generator = torch.Generator().manual_seed(seed)
        return _data_loader.SequenceBucketBatchSampler(
            weights,
            valid_steps,
            (20, 40, 60),
            batch_size=8,
            generator=generator,
            num_samples=16_000,
        )

    batches = list(make_sampler(7))
    assert batches == list(make_sampler(7))
    assert batches != list(make_sampler(8))
    assert all(
        len(set(_data_loader._sequence_bucket_ids(valid_steps[batch], (20, 40, 60)).tolist())) == 1 for batch in batches
    )

    draws = np.asarray([index for batch in batches for index in batch])
    assert 0 not in draws
    empirical = np.bincount(draws, minlength=len(weights)) / len(draws)
    np.testing.assert_allclose(empirical, weights, atol=0.02)


def test_sequence_bucket_collate_crops_every_temporal_field_and_preserves_valid_prefix():
    items = [_sequence_item(13), _sequence_item(20, offset=100)]
    batch = _data_loader._sequence_bucket_collate_fn(items, buckets=(20, 40, 60), max_steps=60)

    for key in _data_loader._SEQUENCE_TIME_KEYS:
        assert all(x.shape[1] == 20 for x in jax.tree.leaves(batch[key]))
    assert batch["actions"].shape == (2, 20, 5, 3)
    np.testing.assert_array_equal(batch["state"][0, :, 0], np.arange(20))
    np.testing.assert_array_equal(batch["state"][1, :, 0], np.arange(20) + 100)
    assert np.all(batch["seq_step_mask"][0, :13])
    assert not np.any(batch["seq_step_mask"][0, 13:])


def test_sequence_bucket_collate_rejects_mixed_bucket_batch():
    with pytest.raises(ValueError, match="not homogeneous"):
        _data_loader._sequence_bucket_collate_fn(
            [_sequence_item(20), _sequence_item(21)], buckets=(20, 40, 60), max_steps=60
        )


def test_torch_data_loader_accepts_bucket_batch_sampler():
    class SequenceDataset:
        def __init__(self):
            self.items = [_sequence_item(10), _sequence_item(20), _sequence_item(21), _sequence_item(40)]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    loader = _data_loader.TorchDataLoader(
        SequenceDataset(),
        local_batch_size=2,
        batch_sampler=[[0, 1], [2, 3]],
        collate_fn=functools.partial(_data_loader._sequence_bucket_collate_fn, buckets=(20, 40, 60), max_steps=60),
        num_batches=2,
    )
    batches = list(loader)
    assert [batch["actions"].shape for batch in batches] == [(2, 20, 5, 3), (2, 40, 5, 3)]


def test_torch_data_loader_preserves_effective_batch_and_adds_microbatch_axis():
    class SequenceDataset:
        def __len__(self):
            return 12

        def __getitem__(self, index):
            return _sequence_item(20, offset=index * 100)

    loader = _data_loader.TorchDataLoader(
        SequenceDataset(),
        local_batch_size=12,
        batch_sampler=[list(range(12))],
        collate_fn=functools.partial(_data_loader._sequence_bucket_collate_fn, buckets=(20, 40, 60), max_steps=60),
        num_batches=1,
        gradient_accumulation_steps=3,
    )
    (batch,) = list(loader)

    assert all(x.shape[:2] == (3, 4) for x in jax.tree.leaves(batch))
    assert batch["actions"].shape == (3, 4, 20, 5, 3)
    # Reshaping must preserve the exact B12 sample order rather than resampling B4 three times.
    np.testing.assert_array_equal(batch["state"][:, :, 0, 0].reshape(-1), np.arange(12) * 100)


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
