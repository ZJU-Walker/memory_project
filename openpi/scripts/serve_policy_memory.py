"""Serve the online-memory checkpoint (`pi05_yam_memory` / `Pi0Memory`) over websocket.

Unlike `serve_policy.py` (stateless), this wraps the checkpoint in a stateful `MemoryPolicy` that
threads the online episodic memory `(mem, surprise)` across `infer` calls. Episode reset and the
per-step write-only stream are driven by control keys in the observation dict (`reset`/`write_only`),
so the underlying `WebsocketPolicyServer` is unchanged. See `client_memory.py` for the robot side.

Usage:
    uv run scripts/serve_policy_memory.py \
        --policy.config=pi05_yam_memory \
        --policy.dir=checkpoints/pi05_yam_memory/yam_banana_memory/<step>
"""

import dataclasses
import logging
import socket

import tyro

from openpi.policies import memory_policy as _memory_policy
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Checkpoint:
    """Load a memory policy from a trained checkpoint."""

    # Training config name (must be a memory config, e.g. "pi05_yam_memory").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi05_yam_memory/yam_banana_memory/29999").
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the memory serve script."""

    # How to load the policy.
    policy: Checkpoint

    # If provided, used when the "prompt" key is absent (and the model has no default prompt).
    default_prompt: str | None = None
    # Port to serve on.
    port: int = 8000

    # Online-memory hyperparameter overrides (None -> keep the checkpoint's trained value). These
    # change deployment write dynamics without retraining. If the memory saturates instantly
    # (mem_delta maxes out in the first ~1-2 s while surprise_norm collapses to ~0), lower
    # mem_theta (try 1e-3) and/or raise mem_alpha (e.g. 0.01) so old surprise decays.
    mem_theta: float | None = None
    mem_eta: float | None = None
    mem_alpha: float | None = None
    # Fixed L2 norm for the pooled memory encoding (stability; see Pi0MemoryConfig.mem_x_scale).
    # None keeps the checkpoint's trained value; 0 (or negative) disables normalization entirely --
    # required to faithfully serve checkpoints trained before mem_x_scale existed.
    mem_x_scale: float | None = None
    # Memory-encoding camera override: "" keeps the training config's value (top camera only);
    # "all" encodes all cameras -- required for checkpoints trained before the top-camera-only
    # change (e.g. yam_banana_memory_400m_v1); or an explicit camera name like "base_0_rgb".
    # Prefer the serve_policy_memory_v1.py / serve_policy_memory_v2.py wrappers, which set this.
    mem_camera: str = ""


def main(args: Args) -> None:
    policy = _memory_policy.create_trained_memory_policy(
        _config.get_config(args.policy.config),
        args.policy.dir,
        default_prompt=args.default_prompt,
        mem_overrides={
            "mem_theta": args.mem_theta,
            "mem_eta": args.mem_eta,
            "mem_alpha": args.mem_alpha,
            "mem_camera": args.mem_camera,
            "mem_x_scale": args.mem_x_scale,
        },
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating memory server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)

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
