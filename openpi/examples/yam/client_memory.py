"""Robot-side inference client for the *memory* checkpoint (`pi05_yam_memory` / `Pi0Memory`).

Runs on the robot computer, alongside `client_base.py`. Reads YAM state + the three RealSense cameras
directly and talks to a stateful memory server (`scripts/serve_policy_memory.py`) that maintains the
online episodic memory `(mem, surprise)` server-side.

Cadence (per the design): the episodic memory must accumulate at the full inspection rate, but action
inference (the expensive LLM forward) only needs to run per action chunk. So each control step we send
a cheap **write-only** message (server runs `memory_write`, no LLM); only when the current 50-step
action chunk is exhausted do we send an **action** message (server writes *and* samples a fresh chunk
conditioned on the current memory). At episode start we send a **reset** so memory clears to M_0.

Control keys understood by the server (placed in the obs dict):
    reset=True       -> clear (mem, surprise) to M_0
    write_only=True  -> cheap memory write, returns an ack (no action)
    (neither)        -> write + sample; returns {"actions": (50,14), ...}

Server (GPU box):
    uv run scripts/serve_policy_memory.py \
        --policy.config=pi05_yam_memory --policy.dir=checkpoints/pi05_yam_memory/yam_banana_memory/<step>

Client (robot computer; needs both `gello_software` and `openpi_client` importable):
    python examples/yam/client_memory.py --host <gpu-host> --port 8000
"""

import dataclasses
import datetime
import logging
import os
import shutil
import subprocess

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

PROMPT = "find the bin with banana"
BIMANUAL_DOF = 14


class _H264Writer:
    """Encode RGB frames to a browser/VSCode-previewable H.264 mp4 via an `ffmpeg` subprocess.

    OpenCV's bundled ffmpeg here only exposes the hardware `h264_v4l2m2m` encoder (no valid
    device), so we pipe raw frames to the system ffmpeg + libx264 instead.
    """

    def __init__(self, path: str, width: int, height: int, fps: float):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH -- needed to encode the recording")
        self._proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", f"{fps}",
                "-i", "-",
                "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                # libx264 + yuv420p needs even dimensions; pad up if a camera reports odd ones.
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                path,
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame_rgb: np.ndarray) -> None:
        self._proc.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())

    def release(self) -> None:
        if self._proc.stdin is not None:
            self._proc.stdin.close()
        self._proc.wait()


@dataclasses.dataclass
class Args:
    # --- Policy server (remote GPU box) ---
    host: str = "10.79.12.252"
    port: int = 8000

    # --- Inference / control ---
    action_horizon: int = 50
    """Steps executed from each sampled action chunk before requesting a new one. <= the model's
    action_horizon (pi05_yam_memory = 50); a smaller value re-samples (and re-reads memory) sooner."""
    max_steps: int = 12000
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

    # --- Recording ---
    record: bool = True
    """Record the top camera view during the control loop; saved on exit (incl. Ctrl+C)."""
    record_dir: str = "eval"
    """Directory for the top camera recording (created if missing)."""
    record_path: str = ""
    """Output .mp4 path. Empty -> `<record_dir>/top_camera_<timestamp>.mp4`."""

    # --- Debug ---
    dry_run: bool = False
    """Skip hardware: exercise reset / write_only / action messages with random observations and
    check that server memory accumulates (mem_delta grows) then clears on reset."""


def _obs_to_request(obs: dict, prompt: str) -> dict:
    """Map a `RobotEnv.get_obs()` dict to the websocket observation the server expects.

    Camera key mapping mirrors the dataset converter:
        top_camera -> observation/image (base); left/right -> *_wrist_image.
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


def _run_dry(ws, args: Args) -> None:
    """Exercise the memory protocol with random data -- no hardware needed."""
    from openpi.policies import yam_policy

    def req(**extra):
        ex = yam_policy.make_yam_example()
        ex["prompt"] = args.prompt
        ex.update(extra)
        return ex

    logging.info("Dry run: reset -> writes -> action -> reset")
    r = ws.infer(req(reset=True, write_only=True))
    logging.info("  after reset+write: mem_delta=%.4g", r.get("mem_delta", float("nan")))
    prev = r.get("mem_delta", 0.0)
    for i in range(5):
        r = ws.infer(req(write_only=True))
        d = r.get("mem_delta", float("nan"))
        logging.info("  write %d: mem_delta=%.4g (grew=%s)", i, d, d >= prev)
        prev = d
    out = ws.infer(req())  # action step
    actions = np.asarray(out["actions"])
    assert actions.shape == (args.action_horizon, BIMANUAL_DOF), f"got {actions.shape}"
    assert np.all(np.isfinite(actions)), "non-finite actions"
    logging.info("  action step: actions shape=%s finite=True", actions.shape)
    r = ws.infer(req(reset=True))
    logging.info("  after reset: mem_delta=%.4g (expect ~0)", r.get("mem_delta", float("nan")))
    logging.info("Dry run OK.")


def main(args: Args) -> None:
    ws = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Server metadata: %s", ws.get_server_metadata())

    if args.dry_run:
        _run_dry(ws, args)
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

    # --- New episode: clear server memory, and get the first action chunk ---
    logging.info("Resetting server memory for a new episode...")
    out = ws.infer({**_obs_to_request(obs, args.prompt), "reset": True})
    chunk = np.asarray(out["actions"], dtype=np.float64)  # (50, 14)
    chunk_i = 0

    # --- Ramp to the first target (avoid a large jump from rest) ---
    logging.info("Ramping to first policy target...")
    first_target = chunk[0]
    for _ in range(25):
        obs = env.get_obs()
        cur = np.asarray(obs["joint_positions"], dtype=np.float64)
        if float(np.abs(first_target - cur).max()) < 1e-2:
            break
        env.step(_clamp_joint_delta(first_target, cur, args.max_joint_delta))

    # --- Set up top camera recording (written incrementally so Ctrl+C still saves it) ---
    writer = None
    frames_written = 0
    record_path = args.record_path
    if args.record:
        if not record_path:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(args.record_dir, exist_ok=True)
            record_path = os.path.join(args.record_dir, f"top_camera_{stamp}.mp4")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(record_path)), exist_ok=True)
        first_frame = image_tools.convert_to_uint8(env.get_obs()["top_camera_rgb"])
        writer = _H264Writer(record_path, first_frame.shape[1], first_frame.shape[0], args.hz)
        logging.info(
            "Recording top camera (H.264) to %s (%dx%d @ %.1f Hz)",
            record_path, first_frame.shape[1], first_frame.shape[0], args.hz,
        )

    # --- Control loop ---
    logging.info("Starting control loop: %d steps @ %.1f Hz", args.max_steps, args.hz)
    obs = env.get_obs()
    try:
        for step in range(args.max_steps):
            if writer is not None:
                # obs["top_camera_rgb"] is the RGB frame the policy sees this step.
                writer.write(image_tools.convert_to_uint8(obs["top_camera_rgb"]))
                frames_written += 1
            req = _obs_to_request(obs, args.prompt)
            if chunk_i >= args.action_horizon:
                # Chunk exhausted: action message (server writes this frame + samples a new chunk).
                out = ws.infer(req)
                chunk = np.asarray(out["actions"], dtype=np.float64)
                chunk_i = 0
            else:
                # Mid-chunk: cheap write-only update so memory keeps accumulating at full rate.
                ws.infer({**req, "write_only": True})

            cur = np.asarray(obs["joint_positions"], dtype=np.float64)
            action = _clamp_joint_delta(chunk[chunk_i], cur, args.max_joint_delta)
            chunk_i += 1
            obs = env.step(action)

            if step % args.hz == 0:
                logging.info("  step %d / %d (mem_delta=%.4g)", step, args.max_steps, out.get("mem_delta", float("nan")))
    except KeyboardInterrupt:
        logging.info("Interrupted by user -- stopping (arms left in place).")
    finally:
        if writer is not None:
            writer.release()
            logging.info("Saved recording: %s (%d frames)", record_path, frames_written)
        logging.info("Control loop finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
