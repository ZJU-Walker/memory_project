"""Robot-side inference client for the pi05 subtask model on the bimanual YAM.

Runs on the robot computer. Reads YAM joint state + the three RealSense cameras directly
(in-process, no ZMQ layer), sends observations over websocket to a remote
`scripts/serve_yam_subtask.py` server, and commands the returned 14-dim absolute joint targets
back to the two arms. Every server response also carries the decoded subtask text; a background
thread shows the live top camera in an OpenCV window with that subtask overlaid (and the
recording gets the same overlay).

The action chunk (50 steps) is executed open-loop via `ActionChunkBroker`: the server is
re-queried only when the current chunk is exhausted; the displayed subtask updates on re-query.

Server (on a GPU box):
    cd openpi && uv run scripts/serve_yam_subtask.py --dir checkpoints/pi05_yam/<exp>/<step>

Client (on the robot computer; needs `gello_software`, `openpi_client` and opencv importable):
    python examples/yam/client_subtask.py --host <gpu-host> --port 8000

Smoke-test the obs/action contract without hardware:
    python examples/yam/client_subtask.py --host <gpu-host> --port 8000 --dry-run
"""

import dataclasses
import datetime
import logging
import os
import shutil
import subprocess
import threading

import cv2
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
    host: str = "10.79.12.64"
    port: int = 8000

    # --- Inference / control ---
    action_horizon: int = 25
    """Steps consumed from each inferred chunk before re-querying the server. Must be <= the
    model's action_horizon (pi05_yam = 50)."""
    max_steps: int = 12000
    hz: float = 20.0
    prompt: str = PROMPT
    max_joint_delta: float = 1.0
    """Per-step safety clamp: cap |target - current| across all joints to this many radians."""

    # --- Display ---
    show: bool = True
    """Show the top camera + predicted subtask in an OpenCV window (background thread)."""

    # --- Hardware (defaults from gello configs/yam_left.yaml) ---
    can_left: str = "can_left"
    can_right: str = "can_right"
    top_camera_serial: str = "409122273280"
    left_camera_serial: str = "409122271088"
    right_camera_serial: str = "409122271086"

    # --- Recording ---
    record: bool = True
    """Record the top camera view (with the subtask overlay); saved on exit (incl. Ctrl+C)."""
    record_dir: str = "eval"
    """Directory for the top camera recording (created if missing)."""
    record_path: str = ""
    """Output .mp4 path. Empty -> `<record_dir>/top_camera_<timestamp>.mp4`."""

    # --- Debug ---
    dry_run: bool = False
    """Skip hardware: feed random observations to the policy to validate the obs/action
    contract (shapes, keys, finiteness, subtask string) against a running server."""


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


class _SubtaskDisplay:
    """Background thread showing the latest top camera frame + predicted subtask in a window."""

    def __init__(self, window: str = "pi05 yam - predicted subtask"):
        self._window = window
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None  # RGB uint8
        self._subtask = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, frame_rgb: np.ndarray, subtask: str | None = None) -> None:
        with self._lock:
            self._frame = frame_rgb
            if subtask is not None:
                self._subtask = subtask

    def _loop(self) -> None:
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        while not self._stop.is_set():
            with self._lock:
                frame = None if self._frame is None else self._frame.copy()
                subtask = self._subtask
            if frame is not None:
                img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    img, f"subtask: {subtask}", (12, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 235, 60), 2, cv2.LINE_AA,
                )
                cv2.imshow(self._window, img)
            cv2.waitKey(30)  # ~33 Hz refresh; also services the GUI event loop
        cv2.destroyAllWindows()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def _overlay_subtask(frame_rgb: np.ndarray, subtask: str) -> np.ndarray:
    """The recording overlay (RGB in, RGB out)."""
    img = np.ascontiguousarray(frame_rgb).copy()
    cv2.putText(img, f"subtask: {subtask}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 235, 60), 2, cv2.LINE_AA)
    return img


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
    logging.info("Dry run: sending %d random observations to the server...", 5)
    rng = np.random.default_rng(0)
    for i in range(5):
        example = {
            "observation/state": rng.random(BIMANUAL_DOF).astype(np.float32),
            "observation/image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
            "observation/left_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
            "observation/right_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
            "prompt": args.prompt,
        }
        result = policy.infer(example)
        action = np.asarray(result["actions"])
        assert action.shape == (BIMANUAL_DOF,), f"expected (14,) per broker step, got {action.shape}"
        assert np.all(np.isfinite(action)), "non-finite action returned"
        assert isinstance(result.get("subtask"), str), f"missing subtask string, got {result.get('subtask')!r}"
        logging.info("  step %d: action shape=%s subtask=%r", i, action.shape, result["subtask"])
    logging.info("Dry run OK -- obs/action/subtask contract matches.")


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

    # --- Live display (background thread) ---
    display = _SubtaskDisplay() if args.show else None

    # --- Set up top camera recording (written incrementally so Ctrl+C still saves it) ---
    writer = None
    frames_written = 0
    if args.record:
        record_path = args.record_path
        if not record_path:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
            os.makedirs(args.record_dir, exist_ok=True)
            record_path = os.path.join(args.record_dir, f"top_camera_{stamp}.mp4")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(record_path)), exist_ok=True)
        first_frame = image_tools.convert_to_uint8(obs["top_camera_rgb"])
        writer = _H264Writer(record_path, first_frame.shape[1], first_frame.shape[0], args.hz)
        logging.info(
            "Recording top camera (H.264) to %s (%dx%d @ %.1f Hz)",
            record_path, first_frame.shape[1], first_frame.shape[0], args.hz,
        )

    # --- Control loop (paced to `hz` by RobotEnv.step's internal Rate) ---
    logging.info("Starting control loop: %d steps @ %.1f Hz", args.max_steps, args.hz)
    obs = env.get_obs()
    subtask = ""
    try:
        for step in range(args.max_steps):
            result = policy.infer(_obs_to_request(obs, args.prompt))
            action = np.asarray(result["actions"], dtype=np.float64)
            subtask = str(result.get("subtask", subtask))

            frame = image_tools.convert_to_uint8(obs["top_camera_rgb"])
            if display is not None:
                display.update(frame, subtask)
            if writer is not None:
                writer.write(_overlay_subtask(frame, subtask))
                frames_written += 1

            cur = np.asarray(obs["joint_positions"], dtype=np.float64)
            action = _clamp_joint_delta(action, cur, args.max_joint_delta)
            obs = env.step(action)
            if step % args.hz == 0:
                logging.info("  step %d / %d | subtask: %s", step, args.max_steps, subtask)
    except KeyboardInterrupt:
        logging.info("Interrupted by user -- stopping (arms left in place).")
    finally:
        if writer is not None:
            writer.release()
            logging.info("Saved recording: %s (%d frames)", record_path, frames_written)
        if display is not None:
            display.close()
        logging.info("Control loop finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
