"""Serve a pi05_yam subtask checkpoint over websocket (fused subtask decoding + action denoising).

Like scripts/serve_policy.py, but the policy runs `Pi0.sample_subtask_and_actions`, so every
response additionally carries the decoded subtask text under the "subtask" key (the actions are
conditioned on that decoded subtask, matching the training-time hierarchy).

Usage (on the GPU box):
    uv run scripts/serve_yam_subtask.py --dir checkpoints/pi05_yam/<exp>/<step>
"""

import dataclasses
import logging
import socket
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
    # Checkpoint directory, e.g. checkpoints/pi05_yam/<exp>/<step>.
    dir: str
    config: str = "pi05_yam"
    port: int = 8000
    max_decode_steps: int = 10


class SubtaskPolicy(_policy.Policy):
    """Policy whose responses carry the decoded subtask text alongside the actions."""

    def __init__(self, model, *, decode_tokenizer, stop_token: int, max_decode_steps: int, **kwargs):
        super().__init__(model, **kwargs)
        self._decode_tokenizer = decode_tokenizer
        self._stop_token = stop_token
        self._max_decode_steps = max_decode_steps
        # Both determine traced shapes / constants inside the decoding loop -> jit-static.
        self._sample_fused = nnx_utils.module_jit(
            model.sample_subtask_and_actions, static_argnames=("stop_token", "max_decode_steps")
        )

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions, out_obs = self._sample_fused(
            sample_rng, observation, stop_token=self._stop_token, max_decode_steps=self._max_decode_steps
        )
        model_time = time.monotonic() - start_time

        # Decode the generated subtask: the tokens appended after the input prompt.
        n0 = int(np.asarray(inputs["tokenized_prompt_mask"]).sum())
        out_n = int(np.asarray(out_obs.tokenized_prompt_mask).sum())
        subtask = self._decode_tokenizer.decode(np.asarray(out_obs.tokenized_prompt)[0, n0:out_n].tolist()).strip()

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["subtask"] = subtask  # after the output transforms: YamOutputs keeps only "actions"
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs


def create_policy(args: Args) -> SubtaskPolicy:
    """Mirrors policy_config.create_trained_policy, but builds a SubtaskPolicy."""
    train_config = _config.get_config(args.config)
    checkpoint_dir = download.maybe_download(args.dir)

    logging.info("Loading model...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    pg = _tokenizer.FASTSubtaskTokenizer(train_config.model.max_token_len)._paligemma_tokenizer  # noqa: SLF001
    # Same terminator the training subtasks were tokenized with (trailing "\n" of the segment).
    stop_token = int(pg.encode("placeholder subtask\n")[-1])

    metadata: dict[str, Any] = train_config.policy_metadata or {}
    return SubtaskPolicy(
        model,
        decode_tokenizer=pg,
        stop_token=stop_token,
        max_decode_steps=args.max_decode_steps,
        transforms=[
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
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
