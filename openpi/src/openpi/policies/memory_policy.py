"""Server-side stateful policy for the online-memory model (`Pi0Memory`).

The stock `openpi.policies.policy.Policy` is stateless: every `infer` runs `sample_actions` with
`mem=None`, so a memory checkpoint would always fall back to the learned `M_0` and never accumulate
episodic memory. `MemoryPolicy` fixes this by threading the online-memory carry `(mem, surprise)`
across `infer` calls (see `Pi0Memory.memory_write` / `sample_actions`, whose docstring notes the
caller must maintain `(mem, surprise)` across steps).

State lives here on the server (the model + JAX are here). The robot client drives cadence via three
control keys placed in the observation dict, so the existing `WebsocketPolicyServer` needs no change:

  - ``reset=True``       -> clear `(mem, surprise)` back to `M_0` (start a new episode). Returns an ack.
  - ``write_only=True``  -> run a single cheap memory write (SigLIP encode + MLP update, no LLM) and
                            return an ack. Used for the per-step write stream during inspection.
  - (neither)            -> an action step: write this frame, then sample a full action chunk with the
                            current memory. Returns ``{"state", "actions", "policy_timing"}``.
"""

from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0_memory as _pi0_memory
from openpi.shared import array_typing as at
import openpi.shared.download as download
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config


class MemoryPolicy(_base_policy.BasePolicy):
    """Stateful JAX policy that maintains `Pi0Memory`'s online memory across `infer` calls."""

    def __init__(
        self,
        model: _pi0_memory.Pi0Memory,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        report_mem_delta: bool = True,
    ):
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        # ||mem - M_0|| is a cheap-but-blocking device->host sync each call; on by default so the
        # client can confirm memory is accumulating. Disable for the lowest-latency write stream.
        self._report_mem_delta = report_mem_delta

        # Freeze module state once (like Policy); mem/surprise are threaded as explicit args.
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._memory_write = nnx_utils.module_jit(model.memory_write)
        self._rng = rng or jax.random.key(0)

        # Online-memory carry (batch size 1). Reset initializes these.
        self._mem: dict | None = None
        self._surprise: dict | None = None
        self.reset()

    @override
    def reset(self) -> None:
        """Clear episodic memory back to the learned M_0 (start of a new episode)."""
        self._mem = self._model.mem_init(1)
        self._surprise = jax.tree.map(jnp.zeros_like, self._mem)

    def _to_observation(self, obs: dict) -> _model.Observation:
        """Run the (input) transform pipeline and build a batched Observation. Mirrors Policy.infer."""
        inputs = jax.tree.map(lambda x: x, obs)  # copy; transforms may mutate in place
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        return _model.Observation.from_dict(inputs)

    def memory_delta_norm(self) -> float:
        """||mem - M_0|| across all memory params (debug: confirms memory is accumulating).

        Returns NaN when `report_mem_delta` is disabled (avoids a device->host sync on the hot path).
        """
        if not self._report_mem_delta:
            return float("nan")
        m0 = self._model.mem_init(1)
        sq = jax.tree.map(lambda a, b: jnp.sum(jnp.square(a - b)), self._mem, m0)
        total = sum(float(v) for v in jax.tree.leaves(sq))
        return float(np.sqrt(total))

    def memory_stats(self) -> dict:
        """Richer memory diagnostics for offline analysis (all forced to host floats).

        - ``mem_delta``     : ||mem - M_0||  (total drift from episode start)
        - ``surprise_norm`` : ||S||          (write momentum magnitude; ~0 => writes have gone quiet)
        - ``mem_norm``      : ||mem||        (absolute size; watch for blow-up)
        - ``delta_by_param``: per-param ||mem_p - M_0_p|| (which of w1/b1/w2/b2 is moving)

        Returns NaNs when `report_mem_delta` is disabled (skips the device->host sync).
        """
        if not self._report_mem_delta:
            return {"mem_delta": float("nan"), "surprise_norm": float("nan"), "mem_norm": float("nan")}
        m0 = self._model.mem_init(1)
        delta_by_param = {k: float(jnp.sqrt(jnp.sum(jnp.square(self._mem[k] - m0[k])))) for k in self._mem}
        mem_delta = float(np.sqrt(sum(v * v for v in delta_by_param.values())))
        surprise_norm = float(
            jnp.sqrt(sum(jnp.sum(jnp.square(s)) for s in jax.tree.leaves(self._surprise)))
        )
        mem_norm = float(jnp.sqrt(sum(jnp.sum(jnp.square(m)) for m in jax.tree.leaves(self._mem))))
        return {
            "mem_delta": mem_delta,
            "surprise_norm": surprise_norm,
            "mem_norm": mem_norm,
            "delta_by_param": delta_by_param,
        }

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        obs = dict(obs)  # shallow copy so we can pop control keys
        reset = bool(obs.pop("reset", False))
        write_only = bool(obs.pop("write_only", False))

        if reset:
            self.reset()

        # A reset can be piggybacked on a frame that also writes/acts; only bail out early if the
        # message carries no actual observation (a bare reset ping).
        if "observation/state" not in obs and "state" not in obs:
            return {"ok": True, "reset": reset, **self.memory_stats()}

        observation = self._to_observation(obs)

        # Always write the current frame into memory first.
        start = time.monotonic()
        self._mem, self._surprise = self._memory_write(observation, self._mem, self._surprise)

        if write_only:
            write_ms = (time.monotonic() - start) * 1000
            return {"ok": True, "policy_timing": {"write_ms": write_ms}, **self.memory_stats()}

        # Action step: sample a full chunk conditioned on the (just-updated) memory.
        # Both fields keep their leading batch dim here, then get sliced [0] together (as in Policy).
        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": observation.state,  # [1, state_dim]
            "actions": self._sample_actions(sample_rng, observation, mem=self._mem, **self._sample_kwargs),  # [1, 50, 32]
        }
        model_ms = (time.monotonic() - start) * 1000
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {"infer_ms": model_ms}
        outputs.update(self.memory_stats())
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def create_trained_memory_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: _transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, _transforms.NormStats] | None = None,
) -> MemoryPolicy:
    """Load a `Pi0Memory` checkpoint into a stateful `MemoryPolicy`.

    Mirrors `openpi.policies.policy_config.create_trained_policy` (same transform pipeline and norm
    stats), but instantiates the stateful memory policy. JAX-only (the memory model is a JAX nnx
    module). `train_config` must be a memory config, e.g. `get_config("pi05_yam_memory")`.
    """
    repack_transforms = repack_transforms or _transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading memory model...")
    model = train_config.model.load(_model.restore_params(pathlib.Path(checkpoint_dir) / "params", dtype=jnp.bfloat16))
    if not isinstance(model, _pi0_memory.Pi0Memory):
        raise TypeError(f"Expected a Pi0Memory model, got {type(model).__name__}. Use a memory config.")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(pathlib.Path(checkpoint_dir) / "assets", data_config.asset_id)

    return MemoryPolicy(
        model,
        transforms=[
            *repack_transforms.inputs,
            _transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )
