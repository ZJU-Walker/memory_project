"""Utilities shared by train-time real-time chunking (RTC) paths."""

from flax import struct
import jax.numpy as jnp
import numpy as np

import openpi.shared.array_typing as at


@struct.dataclass
class ActionPrefix:
    """A right-padded prefix from the action chunk that is already executing.

    ``delay`` is how many actions are expected to execute before the new chunk is
    ready. ``prefix_length`` is how many entries in ``actions`` are valid. A
    caller must maintain ``0 <= delay <= prefix_length <= action_horizon``.
    """

    actions: at.Float[at.Array, "*b action_horizon action_dim"]
    delay: at.Int[at.Array, "*b"]
    prefix_length: at.Int[at.Array, "*b"]


def validate_action_prefix(
    prefix: ActionPrefix,
    *,
    action_horizon: int,
    action_dim: int | None = None,
) -> None:
    """Validates an action prefix at a non-jitted policy/runtime boundary."""
    actions = np.asarray(prefix.actions)
    delay = np.asarray(prefix.delay)
    prefix_length = np.asarray(prefix.prefix_length)

    if actions.ndim < 2 or actions.shape[-2] != action_horizon:
        raise ValueError(
            f"action prefix must end in [action_horizon, action_dim] with "
            f"action_horizon={action_horizon}; got {actions.shape}."
        )
    if action_dim is not None and actions.shape[-1] != action_dim:
        raise ValueError(f"action prefix action_dim must be {action_dim}; got {actions.shape[-1]}.")
    if delay.shape != actions.shape[:-2] or prefix_length.shape != actions.shape[:-2]:
        raise ValueError(
            "action prefix delay and prefix_length must match the actions batch shape; "
            f"got actions={actions.shape}, delay={delay.shape}, prefix_length={prefix_length.shape}."
        )
    if not (jnp.issubdtype(actions.dtype, jnp.integer) or jnp.issubdtype(actions.dtype, jnp.floating)):
        raise TypeError(f"action prefix actions must have a real numeric dtype; got {actions.dtype}.")
    if not np.all(np.isfinite(actions)):
        raise ValueError("action prefix actions must contain only finite values.")
    for name, value in (("delay", delay), ("prefix_length", prefix_length)):
        if not jnp.issubdtype(value.dtype, jnp.integer):
            raise TypeError(f"action prefix {name} must have an integer dtype; got {value.dtype}.")
    if np.any(delay < 0) or np.any(delay > prefix_length) or np.any(prefix_length > action_horizon):
        raise ValueError(
            "action prefix must satisfy 0 <= delay <= prefix_length <= action_horizon; "
            f"got delay={delay}, prefix_length={prefix_length}, action_horizon={action_horizon}."
        )


def suffix_mask(delay: at.Int[at.Array, "*b"], action_horizon: int) -> at.Bool[at.Array, "*b ah"]:
    """Returns True for the suffix that remains supervised/denoised."""
    return jnp.arange(action_horizon) >= jnp.asarray(delay)[..., None]


def prefix_mask(prefix: ActionPrefix, action_horizon: int) -> at.Bool[at.Array, "*b ah"]:
    """Returns the clean RTC prefix mask, safely bounded by valid prefix data."""
    conditioned_length = jnp.clip(
        jnp.minimum(jnp.asarray(prefix.delay), jnp.asarray(prefix.prefix_length)), 0, action_horizon
    )
    return jnp.arange(action_horizon) < conditioned_length[..., None]


def make_noisy_actions(
    actions: at.Float[at.Array, "*b ah ad"],
    noise: at.Float[at.Array, "*b ah ad"],
    time: at.Float[at.Array, "*b"],
    *,
    delay: at.Int[at.Array, "*b"] | None = None,
) -> tuple[at.Float[at.Array, "*b ah ad"], at.Float[at.Array, "*b ah"], at.Bool[at.Array, "*b ah"] | None]:
    """Builds flow inputs, optionally replacing a random prefix with clean actions.

    With RTC enabled, each action token receives its own flow timestep. Prefix
    tokens use t=0 and are therefore exactly equal to the target action.
    """
    token_time = jnp.broadcast_to(jnp.asarray(time)[..., None], actions.shape[:-1])
    loss_mask = None
    if delay is not None:
        loss_mask = suffix_mask(delay, actions.shape[-2])
        token_time = jnp.where(loss_mask, token_time, 0.0)
    x_t = token_time[..., None] * noise + (1 - token_time[..., None]) * actions
    return x_t, token_time, loss_mask


def renormalize_flow_loss(
    loss: at.Float[at.Array, "*b ah"], mask: at.Bool[at.Array, "*b ah"] | None
) -> at.Float[at.Array, "*b ah"]:
    """Masks the clean prefix while preserving each example's total loss scale."""
    if mask is None:
        return loss
    mask = mask.astype(loss.dtype)
    suffix_fraction = jnp.mean(mask, axis=-1, keepdims=True)
    return loss * mask / jnp.maximum(suffix_fraction, jnp.finfo(loss.dtype).tiny)


def condition_action_prefix(
    x_t: at.Float[at.Array, "*b ah ad"],
    time: at.Float[at.Array, ""] | at.Float[at.Array, "*b"],
    prefix: ActionPrefix | None,
) -> tuple[at.Float[at.Array, "*b ah ad"], at.Float[at.Array, "*b ah"]]:
    """Hard-conditions a denoising input and its token-wise flow timesteps."""
    batch_shape = x_t.shape[:-2]
    time = jnp.asarray(time)
    if time.ndim == 0:
        time = jnp.broadcast_to(time, batch_shape)
    token_time = jnp.broadcast_to(time[..., None], x_t.shape[:-1])
    if prefix is None:
        return x_t, token_time
    mask = prefix_mask(prefix, x_t.shape[-2])
    return jnp.where(mask[..., None], prefix.actions, x_t), jnp.where(mask, 0.0, token_time)


def restore_action_prefix(actions: at.Float[at.Array, "*b ah ad"], prefix: ActionPrefix | None):
    """Restores the conditioned prefix exactly after numerical integration."""
    if prefix is None:
        return actions
    mask = prefix_mask(prefix, actions.shape[-2])
    return jnp.where(mask[..., None], prefix.actions, actions)
