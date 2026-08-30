# ruff: noqa: I001, SLF001

import dataclasses
import os
import pathlib

# The backend must be selected before importing Flax/JAX; setting it afterwards is not an
# effective CPU interlock and can make this test module unexpectedly claim a training GPU.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from . import train
from openpi.models import model as _model
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import utils as training_utils


def test_training_identity_log_includes_config_and_effective_eta_scale(caplog):
    config = dataclasses.replace(
        _config.get_config("pi05_yam_mem_v34_run5_eta0"),
        exp_name="v34_run5_eta0_smoke",
    )
    with caplog.at_level("INFO"):
        train._log_training_identity(config)
    assert "name=pi05_yam_mem_v34_run5_eta0" in caplog.text
    assert "exp_name=v34_run5_eta0_smoke" in caplog.text
    assert "eta_scale=0.0" in caplog.text


def test_console_metric_filter_keeps_aux_phase_but_hides_only_position_suffixes():
    assert not train._is_per_position_metric("diagnostic/aux_phase_accuracy")
    assert not train._is_per_position_metric("diagnostic/some_prefix_metric")
    assert train._is_per_position_metric("diagnostic/probe_accuracy_by_step_p0")
    assert train._is_per_position_metric("diagnostic/probe_accuracy_by_step_p39")


class _ToyModel(_model.BaseModel):
    """Deterministic model for exact full-batch/accumulated-update comparisons."""

    def __init__(self, *, memory_probe_weight: float = 0.5, memory_probe_diagnostic: bool = False):
        super().__init__(action_dim=1, action_horizon=1, max_token_len=1)
        self.kernel = nnx.Param(jnp.asarray([[0.7]], dtype=jnp.float32))
        self.probe_head = nnx.Param(jnp.asarray([[0.9]], dtype=jnp.float32))
        self.ce_loss_weight = 0.3
        self.memory_probe_weight = memory_probe_weight
        self.memory_probe_diagnostic = memory_probe_diagnostic

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, train
        prediction = observation.state[..., 0] * self.kernel.value[0, 0]
        target = actions[..., 0, 0]
        flow = jnp.mean(jnp.square(prediction - target), axis=1)
        ce = jnp.mean(jnp.square(prediction + 0.25 - target), axis=1)

        # Deterministic synthetic write telemetry exercises the exact same sum/count/max
        # reducers as the production sequence model, including gradient accumulation.
        valid = observation.seq_step_mask
        synthetic_write_norm = jnp.abs(observation.state[..., 0])
        write_metrics = {
            "write_grad_norm_sum": jnp.sum(jnp.where(valid, synthetic_write_norm, 0.0), axis=1),
            "write_valid_count": jnp.sum(valid.astype(jnp.float32), axis=1),
            "write_clip_count": jnp.sum((valid & (synthetic_write_norm > 0.5)).astype(jnp.float32), axis=1),
            "write_severe_clip_count": jnp.sum(
                (valid & (synthetic_write_norm > 0.9)).astype(jnp.float32), axis=1
            ),
            "write_grad_norm_max": jnp.max(jnp.where(valid, synthetic_write_norm, 0.0), axis=1),
        }

        if self.memory_probe_weight == 0 and not self.memory_probe_diagnostic:
            return {"flow": flow, "ce": ce, **write_metrics}

        probe_prediction = prediction * self.probe_head.value[0, 0]
        if self.memory_probe_diagnostic:
            probe_prediction = jax.lax.stop_gradient(probe_prediction)
        probe_mask = observation.seq_probe_mask.astype(jnp.float32)
        probe_target = observation.seq_probe_labels.astype(jnp.float32)
        probe_ce = jnp.square(probe_prediction - probe_target) * probe_mask
        probe_correct = (jnp.abs(probe_prediction - probe_target) < 0.5).astype(jnp.float32) * probe_mask
        visible = observation.seq_probe_visible.astype(jnp.float32) * probe_mask
        return {
            "flow": flow,
            "ce": ce,
            **write_metrics,
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
    model = _ToyModel(
        memory_probe_weight=config.model.memory_probe_weight,
        memory_probe_diagnostic=config.model.memory_probe_diagnostic,
    )
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
            {
                "diagnostic/probe_correct_grid": correct20,
                "diagnostic/probe_active_grid": active20,
            },
            {
                "diagnostic/probe_correct_grid": correct40,
                "diagnostic/probe_active_grid": active40,
            },
            {
                "diagnostic/probe_correct_grid": correct60,
                "diagnostic/probe_active_grid": active60,
            },
        ]
    )

    assert set(reduced) == {"diagnostic/probe_accuracy_by_step"}
    np.testing.assert_allclose(reduced["diagnostic/probe_accuracy_by_step"][:20], 9 / 16, rtol=1e-6)
    np.testing.assert_allclose(reduced["diagnostic/probe_accuracy_by_step"][20:40], 8 / 14, rtol=1e-6)
    np.testing.assert_allclose(reduced["diagnostic/probe_accuracy_by_step"][40:], 0.5, rtol=1e-6)


def test_expensive_norm_metrics_are_count_weighted_across_log_window():
    reduced = train._reduce_infos(
        [
            {
                "loss": jnp.asarray(1.0),
                "grad_norm": jnp.asarray(9.0),
                "param_norm": jnp.asarray(12.0),
                "_expensive_norm_count": jnp.asarray(1.0),
            },
            {
                "loss": jnp.asarray(3.0),
                "grad_norm": jnp.asarray(0.0),
                "param_norm": jnp.asarray(0.0),
                "_expensive_norm_count": jnp.asarray(0.0),
            },
            {
                "loss": jnp.asarray(5.0),
                "grad_norm": jnp.asarray(0.0),
                "param_norm": jnp.asarray(0.0),
                "_expensive_norm_count": jnp.asarray(0.0),
            },
        ]
    )

    assert reduced["loss"] == pytest.approx(3.0)
    assert reduced["grad_norm"] == pytest.approx(9.0)
    assert reduced["param_norm"] == pytest.approx(12.0)
    assert "_expensive_norm_count" not in reduced


def test_write_inner_metrics_pool_valid_writes_and_keep_true_window_max():
    reduced = train._reduce_infos(
        [
            {
                "diagnostic/write_inner_grad_sum": jnp.asarray(4.0),
                "diagnostic/write_inner_valid_count": jnp.asarray(2.0),
                "diagnostic/write_inner_clip_count": jnp.asarray(1.0),
                "diagnostic/write_inner_severe_clip_count": jnp.asarray(0.0),
                "diagnostic/write_inner_grad_max": jnp.asarray(3.0),
            },
            {
                "diagnostic/write_inner_grad_sum": jnp.asarray(36.0),
                "diagnostic/write_inner_valid_count": jnp.asarray(9.0),
                "diagnostic/write_inner_clip_count": jnp.asarray(6.0),
                "diagnostic/write_inner_severe_clip_count": jnp.asarray(3.0),
                "diagnostic/write_inner_grad_max": jnp.asarray(11.0),
            },
        ]
    )

    assert reduced["diagnostic/write_inner_grad_norm"] == pytest.approx(40 / 11)
    assert reduced["diagnostic/write_inner_clip_fraction"] == pytest.approx(7 / 11)
    assert reduced["diagnostic/write_inner_severe_clip_fraction"] == pytest.approx(3 / 11)
    assert reduced["diagnostic/write_inner_grad_max"] == pytest.approx(11.0)


def test_diagnostic_probe_metrics_are_count_weighted_and_namespaced():
    reduced = train._reduce_infos(
        [
            {
                "diagnostic/probe_loss_numerator": jnp.asarray(0.0),
                "diagnostic/probe_count": jnp.asarray(0.0),
                "diagnostic/probe_correct": jnp.asarray(0.0),
                "diagnostic/probe_visible_count": jnp.asarray(0.0),
                "diagnostic/probe_visible_correct": jnp.asarray(0.0),
            },
            {
                "diagnostic/probe_loss_numerator": jnp.asarray(2.0),
                "diagnostic/probe_count": jnp.asarray(4.0),
                "diagnostic/probe_correct": jnp.asarray(3.0),
                "diagnostic/probe_visible_count": jnp.asarray(1.0),
                "diagnostic/probe_visible_correct": jnp.asarray(1.0),
            },
        ]
    )

    assert not any(key.startswith("probe_") for key in reduced)
    assert reduced["diagnostic/probe_count"] == 2.0  # mean live probes per optimizer step
    assert reduced["diagnostic/probe_loss"] == 0.5
    assert reduced["diagnostic/probe_accuracy"] == 0.75
    assert reduced["diagnostic/probe_accuracy_visible"] == 1.0
    assert reduced["diagnostic/probe_accuracy_hidden"] == 2 / 3


def test_pad_probe_grids_rejects_overlong_metric():
    with pytest.raises(ValueError, match="exceeds configured maximum"):
        train._pad_probe_grids(jnp.ones(61), jnp.ones(61), 60)


def test_detached_diagnostic_probe_does_not_change_total_loss_or_main_update():
    debug = _config.get_config("debug_mem")
    common_model = dataclasses.replace(debug.model, memory_probe_weight=0.0)
    common = dataclasses.replace(
        debug,
        model=dataclasses.replace(common_model, memory_probe_diagnostic=False),
        batch_size=12,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0e6),
        ema_decay=0.9,
    )
    diagnostic = dataclasses.replace(
        common,
        model=dataclasses.replace(common.model, memory_probe_diagnostic=True),
    )
    batch = _toy_batch()

    plain_state = _toy_train_state(common)
    diagnostic_state = _toy_train_state(diagnostic)
    legacy_probe_state = _toy_train_state(dataclasses.replace(common, model=debug.model))
    assert jax.tree.structure(plain_state.opt_state) == jax.tree.structure(legacy_probe_state.opt_state)

    # Simulate a resumed probe-trained checkpoint, whose saved EMA probe can legitimately differ
    # from its raw train parameter. No-probe continuation must preserve both representations.
    probe_filter = nnx_utils.PathRegex(r".*probe_head.*")

    def offset_probe_ema(state):
        return dataclasses.replace(
            state,
            ema_params=nnx_utils.state_map(
                state.ema_params,
                probe_filter,
                lambda variable: variable.replace(variable.value + 0.25),
            ),
        )

    plain_state = offset_probe_ema(plain_state)
    diagnostic_state = offset_probe_ema(diagnostic_state)
    plain_next, plain_info = train.train_step(common, jax.random.key(123), plain_state, batch)
    diagnostic_next, diagnostic_info = train.train_step(diagnostic, jax.random.key(123), diagnostic_state, batch)

    np.testing.assert_array_equal(plain_info["loss"], diagnostic_info["loss"])
    np.testing.assert_array_equal(plain_info["flow_loss"], diagnostic_info["flow_loss"])
    np.testing.assert_array_equal(plain_info["ce_loss"], diagnostic_info["ce_loss"])
    assert not any(key.startswith("diagnostic/probe") for key in plain_info)
    assert "diagnostic/probe_loss_numerator" in diagnostic_info
    assert "probe_loss" not in diagnostic_info

    # Identical post-update parameters/optimizer state prove that the diagnostic auxiliary
    # outputs did not alter any gradient consumed by the main optimizer.
    for plain, with_diagnostic in zip(
        jax.tree.leaves(plain_next.params), jax.tree.leaves(diagnostic_next.params), strict=True
    ):
        np.testing.assert_array_equal(plain, with_diagnostic)
    for plain, with_diagnostic in zip(
        jax.tree.leaves(plain_next.opt_state), jax.tree.leaves(diagnostic_next.opt_state), strict=True
    ):
        np.testing.assert_array_equal(plain, with_diagnostic)
    for plain, with_diagnostic in zip(
        jax.tree.leaves(plain_next.ema_params), jax.tree.leaves(diagnostic_next.ema_params), strict=True
    ):
        np.testing.assert_array_equal(plain, with_diagnostic)
    np.testing.assert_array_equal(plain_state.params["probe_head"].value, plain_next.params["probe_head"].value)
    np.testing.assert_array_equal(
        diagnostic_state.params["probe_head"].value, diagnostic_next.params["probe_head"].value
    )
    np.testing.assert_array_equal(plain_state.ema_params["probe_head"].value, plain_next.ema_params["probe_head"].value)
    np.testing.assert_array_equal(
        diagnostic_state.ema_params["probe_head"].value,
        diagnostic_next.ema_params["probe_head"].value,
    )


@pytest.mark.parametrize("accumulation_steps", [3, 6])
def test_gradient_accumulation_matches_one_effective_full_batch_update(accumulation_steps: int):
    base_config = dataclasses.replace(
        _config.get_config("debug"),
        batch_size=12,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0e6),
        ema_decay=0.9,
    )
    accumulated_config = dataclasses.replace(base_config, gradient_accumulation_steps=accumulation_steps)
    observation, actions = _toy_batch()
    # Heterogeneous valid lengths catch the old bug where per-sequence fractions were
    # averaged and short sequences received the same weight as long ones.
    lengths = (jnp.arange(base_config.batch_size) % 4) + 1
    heterogeneous_valid = jnp.arange(observation.seq_step_mask.shape[1])[None, :] < lengths[:, None]
    observation = dataclasses.replace(observation, seq_step_mask=heterogeneous_valid)
    full_batch = (observation, actions)
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
    assert float(full_info["diagnostic/write_inner_grad_max"]) == pytest.approx(1.0)
    assert float(accumulated_info["diagnostic/write_inner_valid_count"]) == pytest.approx(30.0)


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


class _ToyV34Model(_model.BaseModel):
    """Toy model emitting the v3.4 aux/ladder loss keys so train_step's plumbing -- macro CE,
    isolated ladder SGD, metric namespacing -- is exercised exactly as the real model would."""

    def __init__(self, *, ladder_scale: float, aux_weight: float):
        super().__init__(action_dim=1, action_horizon=1, max_token_len=1)
        self.kernel = nnx.Param(jnp.asarray([[0.7]], dtype=jnp.float32))
        # Name matches train.MEMORY_PATH_FILTER; zero-init so every pre-existing expected value
        # (pooled etc.) is unchanged while the param still receives a nonzero gradient.
        self.memory_gain = nnx.Param(jnp.asarray([[0.0]], dtype=jnp.float32))
        self.ladder_writer_head = nnx.Param(jnp.asarray([[0.5]], dtype=jnp.float32))
        self.ladder_read_head = nnx.Param(jnp.asarray([[0.25]], dtype=jnp.float32))
        self.ce_loss_weight = 0.3
        self.memory_probe_weight = 0.0
        self.memory_probe_diagnostic = False
        self.memory_aux_loss_weight = aux_weight
        self.memory_aux_margin_weight = 0.0
        self.memory_aux_side_class_ids = (1,)
        self.ladder_scale = ladder_scale

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, train
        prediction = observation.state[..., 0] * (self.kernel.value[0, 0] + self.memory_gain.value[0, 0])
        target = actions[..., 0, 0]
        flow = jnp.mean(jnp.square(prediction - target), axis=1)
        ce = jnp.mean(jnp.square(prediction + 0.25 - target), axis=1)
        pooled = jnp.mean(prediction)
        losses = {
            "flow": flow,
            "ce": ce,
            # three aux classes with counts (2, 1, 0): macro CE = (sum0/2 + sum1/1) / 2
            "aux_ce_class_sum": jnp.stack([2.0 * jnp.square(pooled), jnp.square(pooled - 1.0), 0.0 * pooled]),
            "aux_count_class": jnp.asarray([2.0, 1.0, 0.0]),
            "aux_correct_class": jnp.asarray([1.0, 0.0, 0.0]),
        }
        blocked = jax.lax.stop_gradient(pooled)
        for name, head in (("ladder_writer", self.ladder_writer_head), ("ladder_read", self.ladder_read_head)):
            head_out = head.value[0, 0] * blocked
            losses[f"{name}_ce_sum"] = self.ladder_scale * jnp.square(head_out - 1.0)
            losses[f"{name}_count"] = jnp.asarray(1.0)
            losses[f"{name}_correct"] = jnp.asarray(0.0)
        return losses

    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        return observation.state[..., :1, None]


def _v34_toy_setup(*, ladder_scale: float, aux_weight: float = 0.5):
    v34 = _config.get_config("pi05_yam_mem_v34")
    config = dataclasses.replace(
        v34,
        exp_name="toy",
        batch_size=4,
        gradient_accumulation_steps=1,
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        probe_lr=0.05,
        ema_decay=None,
        freeze_filter=nnx.Nothing,
    )
    model = _ToyV34Model(ladder_scale=ladder_scale, aux_weight=aux_weight)
    params = nnx.state(model)
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule)
    state = training_utils.TrainState(
        step=jnp.asarray(3),
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params.filter(config.trainable_filter)),
        ema_decay=None,
        ema_params=None,
    )
    batch_state = jnp.linspace(-0.4, 1.0, 4 * 2).reshape(4, 2, 1)  # nonzero mean: aux grads live
    observation = _model.Observation(
        images={},
        image_masks={},
        state=batch_state,
        seq_step_mask=jnp.ones((4, 2), dtype=bool),
        seq_block_boundary=jnp.zeros((4, 2), dtype=bool),
    )
    actions = (0.4 * batch_state + 0.1)[..., None]
    return config, state, (observation, actions)


def test_v34_ladder_isolation_and_probe_sgd_inside_train_step():
    """Huge ladder-head gradients must not perturb the main update (they are removed before
    the global clip), the ladder heads must move by exactly -probe_lr * grad, and their Adam
    moments must stay zero."""
    rng = jax.random.key(0)
    config_huge, state_huge, batch = _v34_toy_setup(ladder_scale=1e6)
    config_zero, state_zero, _ = _v34_toy_setup(ladder_scale=0.0)

    new_huge, info_huge = train.train_step(config_huge, rng, state_huge, batch)
    new_zero, info_zero = train.train_step(config_zero, rng, state_zero, batch)

    np.testing.assert_array_equal(
        np.asarray(new_huge.params["kernel"].value), np.asarray(new_zero.params["kernel"].value)
    )
    np.testing.assert_array_equal(np.asarray(info_huge["grad_norm"]), np.asarray(info_zero["grad_norm"]))

    # the ladder heads take exactly one isolated SGD step: d/dW [s*(W*p - 1)^2] = 2s(W*p-1)p
    observation, _ = batch
    model = nnx.merge(state_huge.model_def, state_huge.params)
    pooled = float(jnp.mean(observation.state[..., 0] * 0.7))
    for name, w0 in (("ladder_writer_head", 0.5), ("ladder_read_head", 0.25)):
        grad = 2.0 * 1e6 * (w0 * pooled - 1.0) * pooled
        expected = w0 - config_huge.probe_lr * grad
        np.testing.assert_allclose(float(new_huge.params[name].value[0, 0]), expected, rtol=1e-5)
    del model

    # ladder Adam moments untouched (grads were zeroed before tx.update)
    for path, leaf in jax.tree_util.tree_leaves_with_path(new_huge.opt_state):
        if "ladder_" in jax.tree_util.keystr(path):
            np.testing.assert_array_equal(np.asarray(leaf), 0.0)

    # metric plumbing: rung losses and correct/count reach the info dict
    for rung in ("ladder_writer", "ladder_read"):
        assert f"diagnostic/{rung}_loss" in info_huge
        assert f"diagnostic/{rung}_count" in info_huge


def test_v34_aux_macro_ce_reaches_loss_and_kernel():
    rng = jax.random.key(1)
    config_on, state_on, batch = _v34_toy_setup(ladder_scale=0.0, aux_weight=0.5)
    config_off, state_off, _ = _v34_toy_setup(ladder_scale=0.0, aux_weight=0.0)

    new_on, info_on = train.train_step(config_on, rng, state_on, batch)
    new_off, info_off = train.train_step(config_off, rng, state_off, batch)

    observation, _ = batch
    pooled = float(jnp.mean(observation.state[..., 0] * 0.7))
    expected_macro = ((2.0 * pooled**2) / 2.0 + (pooled - 1.0) ** 2) / 2.0
    np.testing.assert_allclose(float(info_on["aux_loss"]), expected_macro, rtol=1e-5)
    # the aux CE trains the main model: the kernel's Adam first moment (= the clipped gradient
    # on a fresh optimizer) differs with the aux weight on vs off. (The parameter itself moves
    # identically on the very first step -- fresh-moment Adam is sign-normalized -- so the
    # moment, not the update, is the right witness.)
    assert float(info_on["loss"]) != float(info_off["loss"])
    moment_on = moment_off = None
    for path, leaf in jax.tree_util.tree_leaves_with_path(new_on.opt_state):
        if "kernel" in jax.tree_util.keystr(path):
            moment_on = np.asarray(leaf)
            break
    for path, leaf in jax.tree_util.tree_leaves_with_path(new_off.opt_state):
        if "kernel" in jax.tree_util.keystr(path):
            moment_off = np.asarray(leaf)
            break
    assert moment_on is not None
    assert not np.array_equal(moment_on, moment_off)
    # class-group accuracy metrics flow to reduction
    assert float(info_on["diagnostic/aux_side_count"]) == 1.0
    assert float(info_on["diagnostic/aux_phase_count"]) == 2.0


def test_v34_memory_grad_group_clip():
    """The memory-path group pre-clip (v34_run1 postmortem) zeroes/rescales ONLY the
    memory-path gradients before the shared global clip, logs their norm, and is a bit-exact
    no-op when it does not bind."""
    rng = jax.random.key(2)

    # clip = 0: memory-path grads are erased before tx.update -> their Adam moments stay
    # exactly zero, while the non-memory kernel still trains.
    config, state, batch = _v34_toy_setup(ladder_scale=0.0)
    config_zero = dataclasses.replace(config, memory_grad_clip=0.0)
    new_zero, info_zero = train.train_step(config_zero, rng, state, batch)
    assert float(info_zero["memory_grad_norm"]) > 0.0
    kernel_moments = memory_moments = 0
    for path, leaf in jax.tree_util.tree_leaves_with_path(new_zero.opt_state):
        key = jax.tree_util.keystr(path)
        if "memory_gain" in key:
            np.testing.assert_array_equal(np.asarray(leaf), 0.0)
            memory_moments += 1
        elif "'kernel'" in key:
            kernel_moments += 1
            assert np.any(np.asarray(leaf) != 0.0)
    assert memory_moments > 0
    assert kernel_moments > 0

    # a clip far above the actual norm is bit-exact with the clip disabled
    _, state_loose, _ = _v34_toy_setup(ladder_scale=0.0)
    _, state_off, _ = _v34_toy_setup(ladder_scale=0.0)
    new_loose, info_loose = train.train_step(
        dataclasses.replace(config, memory_grad_clip=1e9), rng, state_loose, batch
    )
    new_off, info_off = train.train_step(dataclasses.replace(config, memory_grad_clip=None), rng, state_off, batch)
    assert "memory_grad_norm" in info_loose
    assert "memory_grad_norm" not in info_off
    for name in ("kernel", "memory_gain"):
        np.testing.assert_array_equal(
            np.asarray(new_loose.params[name].value), np.asarray(new_off.params[name].value)
        )
