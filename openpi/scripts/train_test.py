# ruff: noqa: I001, SLF001

import dataclasses
import os
import pathlib

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

os.environ["JAX_PLATFORMS"] = "cpu"

from . import train
from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import utils as training_utils


class _ToyModel(_model.BaseModel):
    """Deterministic model for exact full-batch/accumulated-update comparisons."""

    def __init__(self):
        super().__init__(action_dim=1, action_horizon=1, max_token_len=1)
        self.kernel = nnx.Param(jnp.asarray([[0.7]], dtype=jnp.float32))
        self.ce_loss_weight = 0.3
        self.memory_probe_weight = 0.5

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, train
        prediction = observation.state[..., 0] * self.kernel.value[0, 0]
        target = actions[..., 0, 0]
        flow = jnp.mean(jnp.square(prediction - target), axis=1)
        ce = jnp.mean(jnp.square(prediction + 0.25 - target), axis=1)

        probe_mask = observation.seq_probe_mask.astype(jnp.float32)
        probe_target = observation.seq_probe_labels.astype(jnp.float32)
        probe_ce = jnp.square(prediction - probe_target) * probe_mask
        probe_correct = (jnp.abs(prediction - probe_target) < 0.5).astype(jnp.float32) * probe_mask
        visible = observation.seq_probe_visible.astype(jnp.float32) * probe_mask
        return {
            "flow": flow,
            "ce": ce,
            "probe_ce_sum": jnp.sum(probe_ce, axis=1),
            "probe_count": jnp.sum(probe_mask, axis=1),
            "probe_correct": jnp.sum(probe_correct, axis=1),
            "probe_count_visible": jnp.sum(visible, axis=1),
            "probe_correct_visible": jnp.sum(probe_correct * visible, axis=1),
            "probe_correct_grid": probe_correct,
            "probe_active_grid": probe_mask,
        }

    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        return observation.state[..., :1, None]


def _toy_train_state(config: _config.TrainConfig) -> training_utils.TrainState:
    model = _ToyModel()
    params = nnx.state(model)
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule)
    trainable_params = params.filter(config.trainable_filter)
    return training_utils.TrainState(
        step=jnp.asarray(7),
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(trainable_params),
        ema_decay=config.ema_decay,
        ema_params=params,
    )


def _toy_batch(batch_size: int = 12, sequence_steps: int = 4):
    state = jnp.linspace(-1.0, 1.0, batch_size * sequence_steps).reshape(batch_size, sequence_steps, 1)
    actions = (0.4 * state + 0.1)[..., None]
    step = jnp.arange(sequence_steps)[None]
    sample = jnp.arange(batch_size)[:, None]
    probe_mask = (step + sample) % 3 != 0
    observation = _model.Observation(
        images={},
        image_masks={},
        state=state,
        seq_step_mask=jnp.ones((batch_size, sequence_steps), dtype=bool),
        seq_block_boundary=jnp.zeros((batch_size, sequence_steps), dtype=bool),
        seq_probe_labels=((step + sample) % 2).astype(jnp.int32),
        seq_probe_mask=probe_mask,
        seq_probe_visible=probe_mask & (step < 2),
    )
    return observation, actions


def test_probe_grid_metrics_stack_across_sequence_buckets():
    correct20, active20 = train._pad_probe_grids(jnp.asarray([1.0] * 20), jnp.asarray([2.0] * 20), 60)
    correct40, active40 = train._pad_probe_grids(jnp.asarray([3.0] * 40), jnp.asarray([4.0] * 40), 60)
    correct60, active60 = train._pad_probe_grids(jnp.asarray([5.0] * 60), jnp.asarray([10.0] * 60), 60)

    reduced = train._reduce_infos(
        [
            {"probe_correct_grid": correct20, "probe_active_grid": active20},
            {"probe_correct_grid": correct40, "probe_active_grid": active40},
            {"probe_correct_grid": correct60, "probe_active_grid": active60},
        ]
    )

    assert set(reduced) == {"probe_acc_grid"}
    np.testing.assert_allclose(reduced["probe_acc_grid"][:20], 9 / 16, rtol=1e-6)
    np.testing.assert_allclose(reduced["probe_acc_grid"][20:40], 8 / 14, rtol=1e-6)
    np.testing.assert_allclose(reduced["probe_acc_grid"][40:], 0.5, rtol=1e-6)


def test_pad_probe_grids_rejects_overlong_metric():
    with pytest.raises(ValueError, match="exceeds configured maximum"):
        train._pad_probe_grids(jnp.ones(61), jnp.ones(61), 60)


@pytest.mark.parametrize("accumulation_steps", [3, 6])
def test_gradient_accumulation_matches_one_effective_full_batch_update(accumulation_steps: int):
    base_config = dataclasses.replace(
        _config.get_config("debug"),
        batch_size=12,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0e6),
        ema_decay=0.9,
    )
    accumulated_config = dataclasses.replace(base_config, gradient_accumulation_steps=accumulation_steps)
    full_batch = _toy_batch()
    microbatch_size = base_config.batch_size // accumulation_steps
    accumulated_batch = jax.tree.map(lambda x: x.reshape(accumulation_steps, microbatch_size, *x.shape[1:]), full_batch)

    full_state, full_info = train.train_step(
        base_config, jax.random.key(123), _toy_train_state(base_config), full_batch
    )
    accumulated_state, accumulated_info = train.train_step(
        accumulated_config,
        jax.random.key(123),
        _toy_train_state(accumulated_config),
        accumulated_batch,
    )

    assert int(full_state.step) == int(accumulated_state.step) == 8
    for full, accumulated in zip(
        jax.tree.leaves(full_state.params), jax.tree.leaves(accumulated_state.params), strict=True
    ):
        np.testing.assert_allclose(full, accumulated, rtol=2e-6, atol=2e-6)
    for full, accumulated in zip(
        jax.tree.leaves(full_state.opt_state), jax.tree.leaves(accumulated_state.opt_state), strict=True
    ):
        np.testing.assert_allclose(full, accumulated, rtol=2e-6, atol=2e-6)
    for full, accumulated in zip(
        jax.tree.leaves(full_state.ema_params), jax.tree.leaves(accumulated_state.ema_params), strict=True
    ):
        np.testing.assert_allclose(full, accumulated, rtol=2e-6, atol=2e-6)
    assert set(full_info) == set(accumulated_info)
    for key in full_info:
        np.testing.assert_allclose(full_info[key], accumulated_info[key], rtol=2e-6, atol=2e-6, err_msg=key)


@pytest.mark.parametrize("config_name", ["debug"])
def test_train(tmp_path: pathlib.Path, config_name: str):
    config = dataclasses.replace(
        _config._CONFIGS_DICT[config_name],
        batch_size=2,
        checkpoint_base_dir=str(tmp_path / "checkpoint"),
        exp_name="test",
        overwrite=False,
        resume=False,
        num_train_steps=2,
        log_interval=1,
    )
    train.main(config)

    # test resuming
    config = dataclasses.replace(config, resume=True, num_train_steps=4)
    train.main(config)
