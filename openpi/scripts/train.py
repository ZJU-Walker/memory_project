import dataclasses
import functools
import logging
import os
import platform
import re
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _log_training_identity(config: _config.TrainConfig) -> None:
    """Make single-variable memory interventions unambiguous in every startup log."""
    memory_config = getattr(config.model, "memory", None)
    logging.info(
        "Training config: name=%s exp_name=%s eta_scale=%s",
        config.name,
        config.exp_name,
        getattr(memory_config, "eta_scale", None),
    )


_PER_POSITION_METRIC_SUFFIX = re.compile(r"_p\d+$")


def _is_per_position_metric(key: str) -> bool:
    """Only suppress expanded vector entries such as ``..._p17`` from console output."""
    return _PER_POSITION_METRIC_SUFFIX.search(key) is not None


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _pad_probe_grids(correct_grid: at.Array, active_grid: at.Array, max_steps: int) -> tuple[at.Array, at.Array]:
    if correct_grid.shape != active_grid.shape or correct_grid.ndim != 1:
        raise ValueError("probe correct/active grids must be matching one-dimensional arrays.")
    if correct_grid.shape[0] > max_steps:
        raise ValueError(f"probe grid length {correct_grid.shape[0]} exceeds configured maximum {max_steps}.")
    pad = max_steps - correct_grid.shape[0]
    return jnp.pad(correct_grid, (0, pad)), jnp.pad(active_grid, (0, pad))


def _aux_macro_ce(class_ce_sum: at.Array, class_count: at.Array) -> at.Array:
    """Class-balanced macro CE (v3.4 plan 5.1): mean over PRESENT classes of the per-class mean
    CE, so frequent phase labels cannot dominate and re-reward the phase-only representation."""
    present = class_count > 0
    per_class = jnp.where(present, class_ce_sum / jnp.maximum(class_count, 1.0), 0.0)
    return jnp.sum(per_class) / jnp.maximum(jnp.sum(present.astype(jnp.float32)), 1.0)


def _aux_group_metrics(chunked_loss: dict[str, at.Array], side_class_ids: tuple[int, ...]) -> dict[str, at.Array]:
    """Per-class-group (phase vs side-bearing) accuracy numerators/denominators for logging."""
    class_count = chunked_loss["aux_count_class"]
    class_correct = chunked_loss["aux_correct_class"]
    num_classes = class_count.shape[0]
    side = jnp.zeros((num_classes,), dtype=bool)
    if side_class_ids:
        side = side.at[jnp.asarray(side_class_ids, dtype=jnp.int32)].set(True)
    return {
        "diagnostic/aux_side_correct": jnp.sum(jnp.where(side, class_correct, 0.0)),
        "diagnostic/aux_side_count": jnp.sum(jnp.where(side, class_count, 0.0)),
        "diagnostic/aux_phase_correct": jnp.sum(jnp.where(side, 0.0, class_correct)),
        "diagnostic/aux_phase_count": jnp.sum(jnp.where(side, 0.0, class_count)),
    }


def _write_diagnostic_sums(chunked_loss: dict[str, at.Array]) -> dict[str, at.Array]:
    """Unreduced write telemetry with a shared valid-write denominator.

    Keeping numerators/counts intact here lets `_reduce_infos` pool exactly across samples,
    unequal sequence buckets, microbatches, and optimizer updates.
    """
    return {
        "diagnostic/write_inner_grad_sum": jnp.sum(chunked_loss["write_grad_norm_sum"]),
        "diagnostic/write_inner_valid_count": jnp.sum(chunked_loss["write_valid_count"]),
        "diagnostic/write_inner_clip_count": jnp.sum(chunked_loss["write_clip_count"]),
        "diagnostic/write_inner_severe_clip_count": jnp.sum(chunked_loss["write_severe_clip_count"]),
        "diagnostic/write_inner_grad_max": jnp.max(chunked_loss["write_grad_norm_max"]),
    }


_LADDER_RUNGS = ("ladder_writer", "ladder_read")

# v3.4 Section 6: the online probe-ladder heads. Their gradients are removed from the main
# optimizer path entirely (a probe must not scale main-model updates through the global clip
# norm) and applied by a separate constant-LR SGD in train_step.
LADDER_PROBE_FILTER = nnx_utils.PathRegex(r".*ladder_(writer|read)_head.*")

# Every parameter on the memory path: the Titans core (memory/*), the read/write query
# compressors and conditioner, and the v3.2+/v3.4 interface params (memory_inject_w,
# memory_gate, memory_aux_*, memory_slot_embedding, state_null_embedding). Used by the
# optional `memory_grad_clip` group pre-clip in train_step.
MEMORY_PATH_FILTER = nnx_utils.PathRegex(r".*(memory|query_compressor|query_conditioner|state_null_embedding).*")


def _reduce_infos(infos: list[dict[str, at.Array]]) -> dict[str, np.ndarray]:
    stacked_infos = common_utils.stack_forest(infos)
    reduced = jax.device_get(jax.tree.map(lambda x: jnp.mean(x, axis=0), stacked_infos))
    norm_count_key = "_expensive_norm_count"
    if norm_count_key in reduced:
        # Parameter/gradient norms traverse the multi-billion-parameter tree. They are sampled
        # once per logging window inside train_step, so reduce them by their explicit count
        # rather than diluting the single sample by the number of updates in the window.
        count = np.sum(jax.device_get(stacked_infos[norm_count_key]), axis=0)
        for key in ("grad_norm", "param_norm"):
            reduced[key] = np.sum(jax.device_get(stacked_infos[key]), axis=0) / np.maximum(count, 1)
        reduced.pop(norm_count_key)
    write_count_key = "diagnostic/write_inner_valid_count"
    if write_count_key in reduced:
        # Exact pooled ratios. Averaging per-sample/per-update means would over-weight short
        # sequences and sparse buckets.
        count = np.sum(jax.device_get(stacked_infos[write_count_key]), axis=0)
        grad_sum = np.sum(jax.device_get(stacked_infos["diagnostic/write_inner_grad_sum"]), axis=0)
        clip_count = np.sum(jax.device_get(stacked_infos["diagnostic/write_inner_clip_count"]), axis=0)
        severe_count = np.sum(
            jax.device_get(stacked_infos["diagnostic/write_inner_severe_clip_count"]), axis=0
        )
        reduced.update(
            {
                "diagnostic/write_inner_grad_norm": grad_sum / np.maximum(count, 1),
                "diagnostic/write_inner_clip_fraction": clip_count / np.maximum(count, 1),
                "diagnostic/write_inner_severe_clip_fraction": severe_count / np.maximum(count, 1),
            }
        )
        for key in (
            write_count_key,
            "diagnostic/write_inner_grad_sum",
            "diagnostic/write_inner_clip_count",
            "diagnostic/write_inner_severe_clip_count",
        ):
            reduced.pop(key)
    # This metric is already a max over sequence position and batch inside each optimizer
    # update. Preserve its meaning across the logging window instead of averaging those maxima.
    window_max_key = "diagnostic/write_inner_grad_max"
    if window_max_key in reduced:
        reduced[window_max_key] = np.max(jax.device_get(stacked_infos[window_max_key]), axis=0)
    probe_count_key = "diagnostic/probe_count"
    if probe_count_key in reduced:
        # Diagnostic accuracies are ratios over every live probe in the entire log window, not
        # an unweighted mean of per-batch ratios (which biases sparse/zero-probe buckets).
        count = np.sum(jax.device_get(stacked_infos[probe_count_key]), axis=0)
        correct = np.sum(jax.device_get(stacked_infos["diagnostic/probe_correct"]), axis=0)
        visible_count = np.sum(jax.device_get(stacked_infos["diagnostic/probe_visible_count"]), axis=0)
        visible_correct = np.sum(jax.device_get(stacked_infos["diagnostic/probe_visible_correct"]), axis=0)
        loss_numerator = np.sum(jax.device_get(stacked_infos["diagnostic/probe_loss_numerator"]), axis=0)
        reduced.update(
            {
                "diagnostic/probe_loss": loss_numerator / np.maximum(count, 1),
                "diagnostic/probe_accuracy": correct / np.maximum(count, 1),
                "diagnostic/probe_accuracy_visible": visible_correct / np.maximum(visible_count, 1),
                "diagnostic/probe_accuracy_hidden": (correct - visible_correct) / np.maximum(count - visible_count, 1),
            }
        )
        for key in (
            "diagnostic/probe_correct",
            "diagnostic/probe_visible_count",
            "diagnostic/probe_visible_correct",
            "diagnostic/probe_loss_numerator",
        ):
            reduced.pop(key)

    grid_correct_key = "diagnostic/probe_correct_grid"
    if grid_correct_key in reduced:
        correct_grid = np.sum(jax.device_get(stacked_infos[grid_correct_key]), axis=0)
        active_grid = np.sum(jax.device_get(stacked_infos["diagnostic/probe_active_grid"]), axis=0)
        reduced.pop(grid_correct_key)
        reduced.pop("diagnostic/probe_active_grid")
        reduced["diagnostic/probe_accuracy_by_step"] = correct_grid / np.maximum(active_grid, 1)

    # v3.4 aux/ladder accuracies: any (X_correct, X_count) pair becomes a window-exact ratio.
    for key in [k for k in reduced if k.endswith("_count") and k.replace("_count", "_correct") in reduced]:
        correct_key = key.replace("_count", "_correct")
        count = np.sum(jax.device_get(stacked_infos[key]), axis=0)
        correct = np.sum(jax.device_get(stacked_infos[correct_key]), axis=0)
        reduced[key.replace("_count", "_accuracy")] = correct / np.maximum(count, 1)
        reduced.pop(key)
        reduced.pop(correct_key)
    return reduced


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        if isinstance(chunked_loss, dict):
            # Subtask co-training: combine the flow and (weighted) token CE losses, log both.
            loss = jnp.mean(chunked_loss["flow"]) + model.ce_loss_weight * jnp.mean(chunked_loss["ce"])
            info = {"flow_loss": jnp.mean(chunked_loss["flow"]), "ce_loss": jnp.mean(chunked_loss["ce"])}
            if "write_grad_norm_sum" in chunked_loss:
                # Core-steepness telemetry (v34 postmortems): healthy ~0.5-3; ramping toward
                # ~50 preceded both explosion cycles by several hundred steps.
                info.update(_write_diagnostic_sums(chunked_loss))
            if "probe_ce_sum" in chunked_loss:
                # Probe outputs are logged under an explicitly diagnostic namespace. Detached
                # diagnostic mode has weight zero and therefore cannot affect the main loss.
                count = jnp.sum(chunked_loss["probe_count"])
                correct = jnp.sum(chunked_loss["probe_correct"])
                vis_count = jnp.sum(chunked_loss["probe_count_visible"])
                vis_correct = jnp.sum(chunked_loss["probe_correct_visible"])
                probe_loss_numerator = jnp.sum(chunked_loss["probe_ce_sum"])
                if model.memory_probe_weight > 0:
                    loss += model.memory_probe_weight * probe_loss_numerator / jnp.maximum(count, 1)
                info.update(
                    {
                        "diagnostic/probe_loss_numerator": probe_loss_numerator,
                        "diagnostic/probe_count": count,
                        "diagnostic/probe_correct": correct,
                        "diagnostic/probe_visible_count": vis_count,
                        "diagnostic/probe_visible_correct": vis_correct,
                    }
                )
                # Keep the per-position numerator and denominator separate and pad both to the
                # configured maximum sequence length. Bucket batches have different static T;
                # padding an already-divided accuracy would incorrectly count absent positions
                # as errors and would also make stack_forest fail across bucket shapes.
                correct_grid = jnp.sum(chunked_loss["probe_correct_grid"], axis=0)
                active_grid = jnp.sum(chunked_loss["probe_active_grid"], axis=0)
                correct_grid, active_grid = _pad_probe_grids(correct_grid, active_grid, config.model.memory_seq_steps)
                info.update(
                    {
                        "diagnostic/probe_correct_grid": correct_grid,
                        "diagnostic/probe_active_grid": active_grid,
                    }
                )
            if "aux_ce_class_sum" in chunked_loss:
                # v3.4 plan 5.1: class-balanced macro CE, trained by the MAIN optimizer.
                aux_loss = _aux_macro_ce(chunked_loss["aux_ce_class_sum"], chunked_loss["aux_count_class"])
                loss += model.memory_aux_loss_weight * aux_loss
                info["aux_loss"] = aux_loss
                info.update(_aux_group_metrics(chunked_loss, model.memory_aux_side_class_ids))
                if "aux_margin_sum" in chunked_loss:
                    aux_margin = chunked_loss["aux_margin_sum"] / jnp.maximum(chunked_loss["aux_margin_count"], 1.0)
                    loss += model.memory_aux_margin_weight * aux_margin
                    info["aux_margin"] = aux_margin
            if "ladder_writer_ce_sum" in chunked_loss:
                # Section 6 online rungs: features are stop-gradient'ed inside the model, so
                # this term reaches ONLY the ladder heads -- whose grads train_step removes
                # from the main optimizer path and applies with the isolated probe SGD.
                for rung in _LADDER_RUNGS:
                    rung_loss = chunked_loss[f"{rung}_ce_sum"] / jnp.maximum(chunked_loss[f"{rung}_count"], 1.0)
                    loss += rung_loss
                    info[f"diagnostic/{rung}_loss"] = rung_loss
                    info[f"diagnostic/{rung}_correct"] = chunked_loss[f"{rung}_correct"]
                    info[f"diagnostic/{rung}_count"] = chunked_loss[f"{rung}_count"]
            if observation.seq_step_mask is not None:
                info.update(
                    sequence_bucket_steps=jnp.asarray(observation.seq_step_mask.shape[1], dtype=jnp.float32),
                    sequence_valid_fraction=jnp.mean(observation.seq_step_mask),
                )
            return loss, info
        return jnp.mean(chunked_loss), {}

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    if config.gradient_accumulation_steps == 1:
        # Keep the original full-batch path unchanged. Besides avoiding extra overhead on
        # H200, this preserves the exact pre-accumulation random-number and reduction order.
        (loss, loss_info), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
            model, train_rng, observation, actions
        )
    else:
        accumulation_steps = config.gradient_accumulation_steps
        if observation.state.shape[0] != accumulation_steps or actions.shape[0] != accumulation_steps:
            raise ValueError(
                "Accumulated training batches must have leading shape "
                f"[{accumulation_steps}, microbatch]; got state {observation.state.shape} and actions {actions.shape}."
            )

        # Probe CE is a ratio over every live quiz in the effective B12 batch. Dividing each
        # B4 numerator by its own count and then averaging would change the objective, so use
        # the data-only full-batch denominator for every microbatch contribution.
        global_probe_count = (
            None if observation.seq_probe_mask is None else jnp.sum(observation.seq_probe_mask.astype(jnp.float32))
        )

        # v3.4 objectives are ratios with data-only denominators too: compute the GLOBAL
        # per-class counts (aux macro CE) and per-rung frame counts (ladder) from the full
        # accumulated batch so summing microbatch contributions reproduces the exact
        # full-batch objective.
        aux_class_count_global = None
        if observation.seq_subtask_class is not None:
            num_aux_classes = getattr(config.model, "memory_aux_num_classes", 0)
            aux_cls = observation.seq_subtask_class
            aux_valid = (aux_cls >= 0) & (aux_cls < num_aux_classes) & observation.seq_step_mask
            aux_onehot = jax.nn.one_hot(jnp.clip(aux_cls, 0, num_aux_classes - 1), num_aux_classes)
            aux_class_count_global = jnp.sum(aux_onehot * aux_valid[..., None].astype(jnp.float32), axis=(0, 1, 2))
            aux_margin_count_global = jnp.sum(aux_valid.astype(jnp.float32))
        ladder_count_global = None
        if observation.seq_side_label is not None and observation.seq_evidence_mask is not None:
            side_ok = (observation.seq_side_label >= 0) & (observation.seq_side_label < 2)
            ladder_count_global = {
                "ladder_writer": jnp.sum(
                    (observation.seq_evidence_mask & observation.seq_step_mask & side_ok[..., None]).astype(jnp.float32)
                ),
                "ladder_read": jnp.sum(
                    (observation.seq_waiting_mask & observation.seq_step_mask & side_ok[..., None]).astype(jnp.float32)
                ),
            }

        def microbatch_loss_fn(model, rng, micro_observation, micro_actions):
            chunked_loss = model.compute_loss(rng, micro_observation, micro_actions, train=True)
            if not isinstance(chunked_loss, dict):
                return jnp.mean(chunked_loss) / accumulation_steps, {}

            flow_loss = jnp.mean(chunked_loss["flow"])
            ce_loss = jnp.mean(chunked_loss["ce"])
            loss = (flow_loss + model.ce_loss_weight * ce_loss) / accumulation_steps
            # These are additive contributions to the metrics of the effective global batch.
            info = {"flow_loss": flow_loss / accumulation_steps, "ce_loss": ce_loss / accumulation_steps}
            if "write_grad_norm_sum" in chunked_loss:
                info.update(_write_diagnostic_sums(chunked_loss))
            if "probe_ce_sum" in chunked_loss:
                if global_probe_count is None:
                    raise ValueError("Probe losses require observation.seq_probe_mask.")
                count = jnp.sum(chunked_loss["probe_count"])
                correct = jnp.sum(chunked_loss["probe_correct"])
                vis_count = jnp.sum(chunked_loss["probe_count_visible"])
                vis_correct = jnp.sum(chunked_loss["probe_correct_visible"])
                probe_loss_numerator = jnp.sum(chunked_loss["probe_ce_sum"])
                if model.memory_probe_weight > 0:
                    loss += model.memory_probe_weight * probe_loss_numerator / jnp.maximum(global_probe_count, 1)
                info.update(
                    {
                        "diagnostic/probe_loss_numerator": probe_loss_numerator,
                        "diagnostic/probe_count": count,
                        "diagnostic/probe_correct": correct,
                        "diagnostic/probe_visible_count": vis_count,
                        "diagnostic/probe_visible_correct": vis_correct,
                    }
                )
                correct_grid = jnp.sum(chunked_loss["probe_correct_grid"], axis=0)
                active_grid = jnp.sum(chunked_loss["probe_active_grid"], axis=0)
                correct_grid, active_grid = _pad_probe_grids(correct_grid, active_grid, config.model.memory_seq_steps)
                info.update(
                    {
                        "diagnostic/probe_correct_grid": correct_grid,
                        "diagnostic/probe_active_grid": active_grid,
                    }
                )
            if "aux_ce_class_sum" in chunked_loss:
                if aux_class_count_global is None:
                    raise ValueError("aux losses require observation.seq_subtask_class.")
                present = aux_class_count_global > 0
                per_class = jnp.where(
                    present, chunked_loss["aux_ce_class_sum"] / jnp.maximum(aux_class_count_global, 1.0), 0.0
                )
                aux_contrib = jnp.sum(per_class) / jnp.maximum(jnp.sum(present.astype(jnp.float32)), 1.0)
                loss += model.memory_aux_loss_weight * aux_contrib
                info["aux_loss"] = aux_contrib  # additive: sums to the exact global macro CE
                info.update(_aux_group_metrics(chunked_loss, model.memory_aux_side_class_ids))
                if "aux_margin_sum" in chunked_loss:
                    aux_margin = chunked_loss["aux_margin_sum"] / jnp.maximum(aux_margin_count_global, 1.0)
                    loss += model.memory_aux_margin_weight * aux_margin
                    info["aux_margin"] = aux_margin
            if "ladder_writer_ce_sum" in chunked_loss:
                if ladder_count_global is None:
                    raise ValueError("ladder probe losses require the seq_side/evidence/waiting fields.")
                for rung in _LADDER_RUNGS:
                    rung_loss = chunked_loss[f"{rung}_ce_sum"] / jnp.maximum(ladder_count_global[rung], 1.0)
                    loss += rung_loss
                    info[f"diagnostic/{rung}_loss"] = rung_loss
                    info[f"diagnostic/{rung}_correct"] = chunked_loss[f"{rung}_correct"]
                    info[f"diagnostic/{rung}_count"] = chunked_loss[f"{rung}_count"]
            if micro_observation.seq_step_mask is not None:
                info.update(
                    sequence_bucket_steps=jnp.asarray(
                        micro_observation.seq_step_mask.shape[1] / accumulation_steps, dtype=jnp.float32
                    ),
                    sequence_valid_fraction=jnp.mean(micro_observation.seq_step_mask) / accumulation_steps,
                )
            return loss, info

        value_and_grad = nnx.value_and_grad(microbatch_loss_fn, argnums=diff_state, has_aux=True)
        # Seed the carry with microbatch zero, then use a real XLA loop for the remainder.
        # A Python loop would inline one complete VLM forward/backward graph per microbatch;
        # on 80GB H100s that made B2x6 *larger* than B4x3. `fori_loop` compiles one reusable
        # body and keeps only one microbatch's activations live at a time.
        first_observation = jax.tree.map(lambda x: x[0], observation)
        (loss, loss_info), grads = value_and_grad(
            model,
            jax.random.fold_in(train_rng, 0),
            first_observation,
            actions[0],
        )

        def accumulate_microbatch(microbatch_index, carry):
            accumulated_loss, accumulated_info, accumulated_grads = carry
            micro_observation = jax.tree.map(lambda x: x[microbatch_index], observation)
            (micro_loss, micro_info), micro_grads = value_and_grad(
                model,
                jax.random.fold_in(train_rng, microbatch_index),
                micro_observation,
                actions[microbatch_index],
            )
            accumulated_info = jax.tree.map(jnp.add, accumulated_info, micro_info)
            write_max_key = "diagnostic/write_inner_grad_max"
            if write_max_key in accumulated_info:
                # Every other info leaf is additive across microbatches. The write max is the
                # sole max-reduced leaf and must not be summed by the generic tree reduction.
                accumulated_info[write_max_key] = jnp.maximum(
                    carry[1][write_max_key], micro_info[write_max_key]
                )
            return (
                accumulated_loss + micro_loss,
                accumulated_info,
                jax.tree.map(jnp.add, accumulated_grads, micro_grads),
            )

        loss, loss_info, grads = jax.lax.fori_loop(
            1,
            accumulation_steps,
            accumulate_microbatch,
            (loss, loss_info, grads),
        )
    diagnostic_only_probe = (
        getattr(config.model, "predict_with_memory", False) and getattr(config.model, "memory_probe_weight", 0) == 0
    )
    if diagnostic_only_probe:
        # Keep the probe leaves in the optimizer tree so probe-trained checkpoints retain an
        # identical TrainState structure, but guarantee that neither diagnostics nor stale
        # restored moments can update the compatibility head.
        probe_filter = nnx_utils.PathRegex(r".*probe_head.*")
        grads = nnx_utils.state_map(
            grads, probe_filter, lambda variable: variable.replace(jnp.zeros_like(variable.value))
        )

    # v3.4 Section 6: the probe ladder gets an ISOLATED optimizer. The ladder-head grads are
    # extracted, then zeroed out of the main path BEFORE tx.update -- so they contribute
    # nothing to the global clip norm or the Adam state -- and applied afterwards as a plain
    # constant-LR SGD step. With probe features stop-gradient'ed in the model, one main-model
    # update is bit-identical with the probes enabled or disabled (unit-tested).
    ladder_isolated = getattr(config.model, "memory_ladder_probes", False)
    if ladder_isolated:
        ladder_grads = grads.filter(LADDER_PROBE_FILTER)
        grads = nnx_utils.state_map(
            grads, LADDER_PROBE_FILTER, lambda variable: variable.replace(jnp.zeros_like(variable.value))
        )

    # v34_run1 postmortem: pre-clip the memory-path gradient group to its own norm budget
    # BEFORE the shared global clip. The recurrent memory backward can spike orders of
    # magnitude above the rest of the model; without this, one bad chain scales EVERY
    # parameter's update toward zero through the global clip (the observed explosion/collapse
    # limit cycle). The group clip preserves the memory gradient's direction and leaves all
    # non-memory gradients untouched.
    if config.memory_grad_clip is not None:
        memory_norm = optax.global_norm(grads.filter(MEMORY_PATH_FILTER))
        memory_scale = jnp.minimum(1.0, config.memory_grad_clip / (memory_norm + 1e-12))
        grads = nnx_utils.state_map(
            grads,
            MEMORY_PATH_FILTER,
            # flax None-bias slots survive into the grads State as None-valued leaves.
            lambda variable: variable if variable.value is None else variable.replace(variable.value * memory_scale),
        )
        loss_info["memory_grad_norm"] = memory_norm

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    if diagnostic_only_probe:
        updates = nnx_utils.state_map(
            updates, probe_filter, lambda variable: variable.replace(jnp.zeros_like(variable.value))
        )
    if ladder_isolated:
        # Erase the weight-decay-only AdamW update on the ladder leaves, then apply the SGD.
        updates = nnx_utils.state_map(
            updates, LADDER_PROBE_FILTER, lambda variable: variable.replace(jnp.zeros_like(variable.value))
        )
    new_params = optax.apply_updates(params, updates)
    if ladder_isolated:
        ladder_new = jax.tree.map(lambda p, g: p - config.probe_lr * g, params.filter(LADDER_PROBE_FILTER), ladder_grads)
        new_params = nnx.State.merge(new_params.filter(nnx.Not(LADDER_PROBE_FILTER)), ladder_new)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        ema_params = jax.tree.map(
            lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
        )
        if diagnostic_only_probe:
            # Probe-trained checkpoints can contain different raw and EMA probe values. Preserve
            # both exactly: allowing the saved/inference EMA head to drift toward the frozen raw
            # head would still mutate the diagnostic across resumed no-probe training.
            ema_params = nnx.State.merge(
                ema_params.filter(nnx.Not(probe_filter)),
                state.ema_params.filter(probe_filter),
            )
        new_state = dataclasses.replace(
            new_state,
            ema_params=ema_params,
        )

    # These full-tree diagnostics do not affect optimization and are only consumed every
    # log_interval steps. Avoid paying their bandwidth/collective cost on the other updates.
    expensive_norm_active = jnp.equal(jnp.mod(state.step, config.log_interval), 0)

    def compute_expensive_norms(_):
        kernel_params = nnx.state(
            model,
            nnx.All(
                nnx.Param,
                nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
                lambda _, x: x.value.ndim > 1,
            ),
        )
        return optax.global_norm(grads), optax.global_norm(kernel_params)

    grad_norm, param_norm = jax.lax.cond(
        expensive_norm_active,
        compute_expensive_norms,
        lambda _: (jnp.asarray(0.0, jnp.float32), jnp.asarray(0.0, jnp.float32)),
        operand=None,
    )
    info = {
        "loss": loss,
        **loss_info,
        "grad_norm": grad_norm,
        "param_norm": param_norm,
        "_expensive_norm_count": expensive_norm_active.astype(jnp.float32),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    _log_training_identity(config)

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    microbatch_size = config.batch_size // config.gradient_accumulation_steps
    if microbatch_size % jax.device_count() != 0:
        raise ValueError(
            f"Microbatch size {microbatch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    # Cluster jobs can inherit an AFS home directory even though the working tree and caches
    # live on Iris.  Allow launchers to choose the cache location without mutating HOME.
    jax_cache_dir = os.environ.get("OPENPI_JAX_CACHE_DIR")
    if jax_cache_dir is None:
        jax_cache_dir = str(epath.Path("~/.cache/jax").expanduser())
    jax.config.update("jax_compilation_cache_dir", jax_cache_dir)

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(
        mesh,
        jax.sharding.PartitionSpec(sharding.DATA_AXIS)
        if config.gradient_accumulation_steps == 1
        else jax.sharding.PartitionSpec(None, sharding.DATA_AXIS),
    )
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    batch_mb = sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(batch)) / 1e6
    logging.info(
        f"Initialized data loader: {len(jax.tree.leaves(batch))} arrays, {batch_mb:.0f} MB/effective batch; "
        f"global_batch={config.batch_size}, microbatch={microbatch_size}, "
        f"gradient_accumulation_steps={config.gradient_accumulation_steps}"
    )

    # Log images only for a fresh run. On resume W&B rejects a new step-0 record, and staging
    # these device arrays needlessly fragments the very tight 80GB H100 allocator before the
    # first accumulated update.
    if not resuming:
        log_images = batch[0].images
        if config.gradient_accumulation_steps > 1:
            log_images = jax.tree.map(lambda x: x.reshape(config.batch_size, *x.shape[2:]), log_images)
        images_to_log = [
            wandb.Image(
                np.concatenate(
                    [np.array(img[i, 0] if img.ndim == 5 else img[i]) for img in log_images.values()], axis=1
                )
            )
            for i in range(min(5, len(next(iter(log_images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    n_params = sum(x.size for x in jax.tree.leaves(train_state.params))
    logging.info(f"Initialized train state: {n_params / 1e9:.2f}B params")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            reduced = _reduce_infos(infos)
            reduced_info = {}
            for k, v in reduced.items():
                if np.ndim(v) == 1:  # per-step quiz accuracy -> one scalar per step index
                    reduced_info.update({f"{k}_p{i}": float(x) for i, x in enumerate(v)})
                else:
                    reduced_info[f"{k}"] = float(v)
            info_str = ", ".join(
                f"{k}={v:.4f}" for k, v in reduced_info.items() if not _is_per_position_metric(k)
            )
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
