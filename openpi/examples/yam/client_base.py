"""Robot-side inference client for the pi05 *baseline* (no memory) on the bimanual YAM.

Runs on the robot computer. Reads YAM joint state + the three RealSense cameras directly
(in-process, no ZMQ layer), sends observations over websocket to a remote `serve_policy.py`
server holding the trained `pi05_yam` checkpoint, and commands the returned 14-dim absolute
joint targets back to the two arms.

The action chunk (50 steps) is executed open-loop via `ActionChunkBroker`: the server is
re-queried only when the current chunk is exhausted (matches openpi's aloha/droid clients).

Server (on a GPU box):
    cd openpi && uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_yam --policy.dir=checkpoints/pi05_yam/yam_banana_pi05/<step>

Client (on the robot computer; needs both `gello_software` and `openpi_client` importable):
    python examples/yam/client_base.py --host <gpu-host> --port 8000

Smoke-test the obs/action contract without hardware:
    python examples/yam/client_base.py --host <gpu-host> --port 8000 --dry-run
"""

import dataclasses
import logging

import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

PROMPT = "find the bin with banana"

# Per-arm DOF: 6 arm joints + 1 gripper. Bimanual state/action is concat(left, right) -> 14.
ARM_DOF = 7
BIMANUAL_DOF = 14


@dataclasses.dataclass
class Args:
    # --- Policy server (remote GPU box) ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Inference / control ---
    action_horizon: int = 50
    """Steps consumed from each inferred chunk before re-querying the server. Must match the
    model's action_horizon (pi05_yam = 50)."""
    max_steps: int = 1200
    hz: float = 30.0
    prompt: str = PROMPT
    max_joint_delta: float = 1.0
    """Per-step safety clamp: cap |target - current| across all joints to this many radians."""

    # --- Hardware (defaults from gello configs/yam_left.yaml) ---
    can_left: str = "can_left"
    can_right: str = "can_right"
    top_camera_serial: str = "409122273280"
    left_camera_serial: str = "409122271088"
    right_camera_serial: str = "409122271086"

    # --- Debug ---
    dry_run: bool = False
    """Skip hardware: feed random observations to the policy to validate the obs/action
    contract (shapes, keys, finiteness) against a running server."""


def _obs_to_request(obs: dict, prompt: str) -> dict:
    """Map a `RobotEnv.get_obs()` dict to the websocket observation the server expects.

    Camera key mapping mirrors the dataset converter:
        top_camera   -> observation/image            (base view)
        left_camera  -> observation/left_wrist_image
        right_camera -> observation/right_wrist_image
    """
    return {
        "observation/state": np.asarray(obs["joint_positions"], dtype=np.float32),
        "observation/image": image_tools.convert_to_uint8(obs["top_camera_rgb"]),
        "observation/left_wrist_image": image_tools.convert_to_uint8(obs["left_camera_rgb"]),
        "observation/right_wrist_image": image_tools.convert_to_uint8(obs["right_camera_rgb"]),
        "prompt": prompt,
    }


def _clamp_joint_delta(target: np.ndarray, current: np.ndarray, max_delta: float) -> np.ndarray:
    """Scale the whole command so no single joint moves more than `max_delta` this step."""
    delta = target - current
    m = float(np.abs(delta).max())
    if m > max_delta:
        delta = delta / m * max_delta
    return current + delta


def _run_dry(policy, args: Args) -> None:
    """Validate the obs/action contract with random data -- no hardware needed."""
    from openpi.policies import yam_policy

    logging.info("Dry run: sending %d random observations to the server...", 5)
    for i in range(5):
        example = yam_policy.make_yam_example()
        example["prompt"] = args.prompt
        action = policy.infer(example)["actions"]
        action = np.asarray(action)
        assert action.shape == (BIMANUAL_DOF,), f"expected (14,) per broker step, got {action.shape}"
        assert np.all(np.isfinite(action)), "non-finite action returned"
        logging.info("  step %d: action shape=%s finite=%s", i, action.shape, np.all(np.isfinite(action)))
    logging.info("Dry run OK -- obs/action contract matches.")


def main(args: Args) -> None:
    # --- Connect to the policy server ---
    ws_client = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Server metadata: %s", ws_client.get_server_metadata())
    policy = action_chunk_broker.ActionChunkBroker(ws_client, action_horizon=args.action_horizon)

    if args.dry_run:
        _run_dry(policy, args)
        return

    # --- Build hardware (direct, in-process) ---
    from gello.cameras.realsense_camera import RealSenseCamera
    from gello.env import RobotEnv
    from gello.robots.robot import BimanualRobot
    from gello.robots.yam import YAMRobot

    left = YAMRobot(channel=args.can_left)
    right = YAMRobot(channel=args.can_right)
    robot = BimanualRobot(left, right)
    assert robot.num_dofs() == BIMANUAL_DOF, f"expected 14 DOF, got {robot.num_dofs()}"

    camera_dict = {
        "top_camera": RealSenseCamera(device_id=args.top_camera_serial),
        "left_camera": RealSenseCamera(device_id=args.left_camera_serial),
        "right_camera": RealSenseCamera(device_id=args.right_camera_serial),
    }
    env = RobotEnv(robot, control_rate_hz=args.hz, camera_dict=camera_dict)

    # Sanity-check the observation once at startup.
    obs = env.get_obs()
    logging.info(
        "Initial obs: joint_positions=%s, top=%s, left=%s, right=%s",
        np.asarray(obs["joint_positions"]).shape,
        np.asarray(obs["top_camera_rgb"]).shape,
        np.asarray(obs["left_camera_rgb"]).shape,
        np.asarray(obs["right_camera_rgb"]).shape,
    )
    for key in ("top_camera_rgb", "left_camera_rgb", "right_camera_rgb"):
        assert key in obs, f"missing camera obs '{key}'"
    assert np.asarray(obs["joint_positions"]).shape == (BIMANUAL_DOF,)

    # --- Ramp to the first inferred target (avoid a large jump from rest) ---
    logging.info("Ramping to first policy target...")
    first_target = np.asarray(policy.infer(_obs_to_request(obs, args.prompt))["actions"], dtype=np.float64)
    for _ in range(25):
        obs = env.get_obs()
        cur = np.asarray(obs["joint_positions"], dtype=np.float64)
        if float(np.abs(first_target - cur).max()) < 1e-2:
            break
        env.step(_clamp_joint_delta(first_target, cur, args.max_joint_delta))
    # The ramp consumed broker steps; reset so the episode starts from a fresh chunk.
    policy.reset()

    # --- Control loop (paced to `hz` by RobotEnv.step's internal Rate) ---
    logging.info("Starting control loop: %d steps @ %.1f Hz", args.max_steps, args.hz)
    obs = env.get_obs()
    try:
        for step in range(args.max_steps):
            action = np.asarray(policy.infer(_obs_to_request(obs, args.prompt))["actions"], dtype=np.float64)
            cur = np.asarray(obs["joint_positions"], dtype=np.float64)
            action = _clamp_joint_delta(action, cur, args.max_joint_delta)
            obs = env.step(action)
            if step % args.hz == 0:
                logging.info("  step %d / %d", step, args.max_steps)
    except KeyboardInterrupt:
        logging.info("Interrupted by user -- stopping (arms left in place).")
    finally:
        logging.info("Control loop finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
