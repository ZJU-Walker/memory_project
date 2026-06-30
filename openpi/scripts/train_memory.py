"""Training entrypoint for the MAC-style memory model (`Pi0Memory`).

Thin wrapper around `scripts/train.py`: it swaps the standard data loader for the episode-sequence
memory data loader (`create_memory_data_loader`) and adjusts the first-batch image logging for the
extra `N` (sequence) axis. Everything else — train-state init, `train_step`, sharding, checkpointing,
EMA — is reused unchanged.

Run with the Iris env sourced:

    source /iris/u/kewalk/.bashrc.user && \\
      uv run scripts/train_memory.py pi05_yam_memory --exp-name=<name>
"""

import os

# Fall back to cuBLAS for GEMMs: the per-example memory matmuls trigger an XLA Triton-GEMM
# autotuning bug on some GPU/XLA versions (autotuner "results do not match the reference" ->
# CUDA_ERROR_ILLEGAL_ADDRESS). Must be set before JAX initializes its GPU backend.
os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_triton_gemm=false").strip()

import functools
import logging
import platform

import etils.epath as epath
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import wandb

# scripts/ is on sys.path when run via `uv run scripts/train_memory.py`, so we can reuse train.py.
import train as _train

import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.memory_data_loader as _memory_data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils


def main(config: _config.TrainConfig):
    _train.init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    _train.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_loader = _memory_data_loader.create_memory_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        seed=config.seed,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log the action (final) frame's camera views. Observation images are [B, N, H, W, C].
    obs0 = batch[0]
    num_log = min(5, obs0.state.shape[0])
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i, -1]) for img in obs0.images.values()], axis=1))
        for i in range(num_log)
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = _train.init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(_train.train_step, config),
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
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
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
