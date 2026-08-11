"""Run one real accumulated optimizer update without writing a checkpoint.

This is intentionally separate from ``scripts/train.py``: it restores the latest checkpoint,
loads one effective global batch, runs the exact jitted training step, reports metrics, and exits.
The source checkpoint and W&B run are never mutated.
"""

import dataclasses
import functools
import time

from flax.training import common_utils
import jax
import numpy as np
import tyro

import openpi.training.checkpoints as checkpoints
import openpi.training.config as config_lib
import openpi.training.data_loader as data_loader_lib
import openpi.training.sharding as sharding
from scripts import train


@dataclasses.dataclass(frozen=True)
class Args:
    config: str = "pi05_yam_mem_v31"
    exp_name: str = "attnwrite_base_s10_d6_t60_b20-40-60_tb25_bs12_seed42"
    batch_size: int = 12
    gradient_accumulation_steps: int = 3
    fsdp_devices: int = 2
    num_updates: int = 1


def main(args: Args) -> None:
    if args.num_updates < 1:
        raise ValueError("num_updates must be positive.")
    config = dataclasses.replace(
        config_lib.get_config(args.config),
        exp_name=args.exp_name,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fsdp_devices=args.fsdp_devices,
        resume=True,
        overwrite=False,
        wandb_enabled=False,
    )

    if config.batch_size % jax.device_count() != 0:
        raise ValueError("The effective global batch must be divisible by the device count.")
    microbatch_size = config.batch_size // config.gradient_accumulation_steps
    if microbatch_size % jax.device_count() != 0:
        raise ValueError("The global microbatch must be divisible by the device count.")

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(None, sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = data_loader_lib.create_data_loader(config, sharding=data_sharding, shuffle=True)
    data_iter = iter(loader)
    batch = next(data_iter)
    print(
        f"effective batch={config.batch_size}, microbatch={microbatch_size}, "
        f"accumulation={config.gradient_accumulation_steps}, devices={jax.device_count()}, "
        f"state shape={batch[0].state.shape}, actions shape={batch[1].shape}",
        flush=True,
    )

    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
    )
    if not resuming:
        raise FileNotFoundError(f"No checkpoint found under {config.checkpoint_dir}.")
    init_rng = jax.random.key(config.seed)
    state, state_sharding = train.init_train_state(config, init_rng, mesh, resume=True)
    state = checkpoints.restore_state(manager, state, loader)
    print(f"restored optimizer step {int(state.step)} from {config.checkpoint_dir}", flush=True)

    ptrain_step = jax.jit(
        functools.partial(train.train_step, config),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    train_rng = jax.random.key(config.seed + 1)
    for update in range(args.num_updates):
        if update:
            batch = next(data_iter)
        start = time.perf_counter()
        with sharding.set_mesh(mesh):
            state, info = ptrain_step(train_rng, state, batch)
        jax.block_until_ready(state)
        elapsed = time.perf_counter() - start
        reduced = common_utils.stack_forest([info])
        reduced = jax.device_get(jax.tree.map(lambda x: np.asarray(x).mean(axis=0), reduced))
        scalars = {key: float(value) for key, value in reduced.items() if np.ndim(value) == 0}
        print(
            f"update {update + 1}/{args.num_updates}: optimizer_step={int(state.step)}, "
            f"elapsed={elapsed:.2f}s, " + ", ".join(f"{key}={value:.5g}" for key, value in scalars.items()),
            flush=True,
        )

    print("[OK] accumulated real optimizer update completed; no checkpoint was written", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
