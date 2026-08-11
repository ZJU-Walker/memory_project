import dataclasses
import functools
import logging
import platform
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


def _reduce_infos(infos: list[dict[str, at.Array]]) -> dict[str, np.ndarray]:
    stacked_infos = common_utils.stack_forest(infos)
    reduced = jax.device_get(jax.tree.map(lambda x: jnp.mean(x, axis=0), stacked_infos))
    if "probe_correct_grid" in reduced:
        correct_grid = reduced.pop("probe_correct_grid")
        active_grid = reduced.pop("probe_active_grid")
        reduced["probe_acc_grid"] = correct_grid / np.maximum(active_grid, 1)
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
            if "probe_ce_sum" in chunked_loss:
                # Quiz probes: loss = sum of quiz CEs / number of live quizzes in the batch.
                count = jnp.sum(chunked_loss["probe_count"])
                correct = jnp.sum(chunked_loss["probe_correct"])
                vis_count = jnp.sum(chunked_loss["probe_count_visible"])
                vis_correct = jnp.sum(chunked_loss["probe_correct_visible"])
                probe_loss = jnp.sum(chunked_loss["probe_ce_sum"]) / jnp.maximum(count, 1)
                loss += model.memory_probe_weight * probe_loss
                info.update(
                    probe_loss=probe_loss,
                    probe_count=count,
                    probe_acc=correct / jnp.maximum(count, 1),
                    # visible = the answer was still on screen at the quiz (bins open; can be
                    # read off vision); hidden = bins closed again, memory is the only source
                    probe_acc_visible=vis_correct / jnp.maximum(vis_count, 1),
                    probe_acc_hidden=(correct - vis_correct) / jnp.maximum(count - vis_count, 1),
                )
                # Keep the per-position numerator and denominator separate and pad both to the
                # configured maximum sequence length. Bucket batches have different static T;
                # padding an already-divided accuracy would incorrectly count absent positions
                # as errors and would also make stack_forest fail across bucket shapes.
                correct_grid = jnp.sum(chunked_loss["probe_correct_grid"], axis=0)
                active_grid = jnp.sum(chunked_loss["probe_active_grid"], axis=0)
                correct_grid, active_grid = _pad_probe_grids(correct_grid, active_grid, config.model.memory_seq_steps)
                info.update(
                    probe_correct_grid=correct_grid,
                    probe_active_grid=active_grid,
                )
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

        def microbatch_loss_fn(model, rng, micro_observation, micro_actions):
            chunked_loss = model.compute_loss(rng, micro_observation, micro_actions, train=True)
            if not isinstance(chunked_loss, dict):
                return jnp.mean(chunked_loss) / accumulation_steps, {}

            flow_loss = jnp.mean(chunked_loss["flow"])
            ce_loss = jnp.mean(chunked_loss["ce"])
            loss = (flow_loss + model.ce_loss_weight * ce_loss) / accumulation_steps
            # These are additive contributions to the metrics of the effective global batch.
            info = {"flow_loss": flow_loss / accumulation_steps, "ce_loss": ce_loss / accumulation_steps}
            if "probe_ce_sum" in chunked_loss:
                if global_probe_count is None:
                    raise ValueError("Probe losses require observation.seq_probe_mask.")
                count = jnp.sum(chunked_loss["probe_count"])
                correct = jnp.sum(chunked_loss["probe_correct"])
                vis_count = jnp.sum(chunked_loss["probe_count_visible"])
                vis_correct = jnp.sum(chunked_loss["probe_correct_visible"])
                probe_loss = jnp.sum(chunked_loss["probe_ce_sum"]) / jnp.maximum(global_probe_count, 1)
                loss += model.memory_probe_weight * probe_loss
                info.update(
                    probe_loss=probe_loss,
                    probe_count=count,
                    probe_correct_sum=correct,
                    probe_visible_count=vis_count,
                    probe_visible_correct_sum=vis_correct,
                )
                correct_grid = jnp.sum(chunked_loss["probe_correct_grid"], axis=0)
                active_grid = jnp.sum(chunked_loss["probe_active_grid"], axis=0)
                correct_grid, active_grid = _pad_probe_grids(correct_grid, active_grid, config.model.memory_seq_steps)
                info.update(probe_correct_grid=correct_grid, probe_active_grid=active_grid)
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
            return (
                accumulated_loss + micro_loss,
                jax.tree.map(jnp.add, accumulated_info, micro_info),
                jax.tree.map(jnp.add, accumulated_grads, micro_grads),
            )

        loss, loss_info, grads = jax.lax.fori_loop(
            1,
            accumulation_steps,
            accumulate_microbatch,
            (loss, loss_info, grads),
        )
        if "probe_count" in loss_info:
            count = loss_info["probe_count"]
            correct = loss_info.pop("probe_correct_sum")
            vis_count = loss_info.pop("probe_visible_count")
            vis_correct = loss_info.pop("probe_visible_correct_sum")
            loss_info.update(
                probe_acc=correct / jnp.maximum(count, 1),
                probe_acc_visible=vis_correct / jnp.maximum(vis_count, 1),
                probe_acc_hidden=(correct - vis_correct) / jnp.maximum(count - vis_count, 1),
            )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        **loss_info,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    microbatch_size = config.batch_size // config.gradient_accumulation_steps
    if microbatch_size % jax.device_count() != 0:
        raise ValueError(
            f"Microbatch size {microbatch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

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
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items() if "_p" not in k)
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
