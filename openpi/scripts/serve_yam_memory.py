"""Serve a pi05_yam_mem_* memory checkpoint over websocket, threading the Titans memory.

Like scripts/serve_yam_subtask.py, but the policy runs `Pi0.sample_with_memory`: every request
reads the memory, decodes the subtask, denoises the actions and then writes the frame's hidden
states into the per-episode memory state, which is threaded across requests. Each response
carries "subtask", "surprise" (1-ish = novel, ~0 = recalled), the write gates and the running
write count. A request containing "reset_memory": true re-initializes the memory (send one at
every episode start); a bare {"reset_memory": true} request (no images) just resets and returns.

The client controls the write cadence: one infer call = one memory write, so call at the
training stride (memory_stride_frames=25 @ 30 Hz -> ~0.83 s between calls).

Usage (on the GPU box):
    uv run scripts/serve_yam_memory.py --dir checkpoints/pi05_yam_mem_warmup/<exp>/<step>
"""

import dataclasses
import logging
import socket
import threading
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.tokenizer as _tokenizer
import openpi.policies.policy as _policy
from openpi.serving import websocket_policy_server
import openpi.shared.download as download
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


@dataclasses.dataclass
class Args:
    # Checkpoint directory, e.g. checkpoints/pi05_yam_mem_warmup/<exp>/<step>.
    dir: str
    config: str = "pi05_yam_mem_warmup"
    port: int = 8000
    max_decode_steps: int = 10


class MemoryPolicy(_policy.Policy):
    """Policy that threads the Titans memory state across requests.

    Responses carry the decoded subtask, the pre-write surprise and the (frozen) write gates.
    """

    def __init__(self, model, *, decode_tokenizer, stop_token: int, max_decode_steps: int, **kwargs):
        super().__init__(model, **kwargs)
        self._decode_tokenizer = decode_tokenizer
        self._stop_token = stop_token
        self._max_decode_steps = max_decode_steps
        self._sample = nnx_utils.module_jit(
            model.sample_with_memory, static_argnames=("stop_token", "max_decode_steps")
        )
        self._init_state = lambda: model.memory.init_state(1)
        self._lock = threading.Lock()
        self._memory_state = self._init_state()
        self._writes = 0

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = jax.tree.map(lambda x: x, obs)  # copy: transforms may modify in place
        if inputs.pop("reset_memory", False):
            with self._lock:
                self._memory_state = self._init_state()
                self._writes = 0
            logging.info("memory reset")
            if "observation/image" not in inputs:  # bare reset ping
                return {"reset": True, "writes": 0}

        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        observation = _model.Observation.from_dict(inputs)

        start_time = time.monotonic()
        with self._lock:
            actions, new_state, aux = self._sample(
                sample_rng,
                observation,
                self._memory_state,
                stop_token=self._stop_token,
                max_decode_steps=self._max_decode_steps,
            )
            jax.block_until_ready(new_state)
            self._memory_state = new_state
            self._writes += 1
            writes = self._writes
        model_time = time.monotonic() - start_time

        tokens = np.asarray(aux["tokens"])[0]
        mask = np.asarray(aux["token_mask"])[0]
        subtask = self._decode_tokenizer.decode(tokens[mask].tolist()).strip()

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["subtask"] = subtask
        outputs["surprise"] = float(aux["surprise"][0])
        outputs["writes"] = writes
        outputs["gates"] = {k: float(np.asarray(aux[k]).mean()) for k in ("theta", "eta", "alpha")}
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs


def create_policy(args: Args) -> MemoryPolicy:
    train_config = _config.get_config(args.config)
    assert train_config.model.predict_with_memory, f"config {args.config} was not built with predict_with_memory"
    checkpoint_dir = download.maybe_download(args.dir)

    logging.info("Loading model (float32: the memory's inner GD is validated in f32)...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.float32))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    gate = np.asarray(model.memory_gate.value)
    logging.info(
        "memory_layer=%d | memory_gate norm %.4f max|g| %.5f (0 = memory content unused)",
        model.memory_layer, np.linalg.norm(gate), np.abs(gate).max(),
    )

    # The memory data config aliases "window_state" -> "state" stats for the training-time
    # window normalization; live responses only carry state/actions and Unnormalize selects
    # strictly, so drop the alias from the output-side stats.
    out_norm_stats = {k: v for k, v in norm_stats.items() if k != "window_state"}

    pg = _tokenizer.FASTSubtaskTokenizer(train_config.model.max_token_len)._paligemma_tokenizer  # noqa: SLF001
    stop_token = int(pg.encode("placeholder subtask\n")[-1])

    metadata: dict[str, Any] = train_config.policy_metadata or {}
    return MemoryPolicy(
        model,
        decode_tokenizer=pg,
        stop_token=stop_token,
        max_decode_steps=args.max_decode_steps,
        transforms=[
            # SplitMemoryWindow is the dataset-side unpacker of stacked lerobot frames; live
            # single-frame observations skip it (same as eval_yam_mem_subtask_raw.py).
            *[t for t in data_config.data_transforms.inputs if not isinstance(t, transforms.SplitMemoryWindow)],
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(out_norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=metadata,
    )


def main(args: Args) -> None:
    policy = create_policy(args)

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
