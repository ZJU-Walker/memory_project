"""Live subtask visualization while YOU teleop: observation-only client for the memory server.

Runs on the robot computer NEXT TO your normal gello teleop (launch_yaml.py) -- it never
commands the robot. It attaches read-only to the ZMQ nodes the teleop launch already started
(robot hardware server on port 6001, the three camera subprocesses on ports 5000/5001/5002),
periodically sends an observation to a remote `scripts/serve_yam_memory.py` server (each call =
one prediction + one memory write, so the default interval matches the training write cadence
of 10 frames @ 30 Hz), and shows the live top camera with the predicted subtask + surprise
overlaid. Optionally records that view to an mp4.

Keys in the display window:  r = reset the server-side memory (new episode)   q = quit.

Server (GPU box):
    uv run scripts/serve_yam_memory.py --dir checkpoints/pi05_yam_mem_v3/<exp>/<step>

Client (robot computer, with your teleop already running):
    python examples/yam/client_memory_viz.py --host <gpu-host> --port 8000

Smoke-test the contract without hardware:
    python examples/yam/client_memory_viz.py --host <gpu-host> --port 8000 --dry-run

Camera-load note: each camera read blocks on the next RealSense frame behind the node's serial
ZMQ socket, so heavy polling would slow the 30 Hz teleop loop. The display therefore reads the
top camera at a modest `display_hz` (default 5) and the wrist cameras only at prediction time.
"""

import dataclasses
import datetime
import logging
import os
import pickle
import shutil
import subprocess
import threading
import time

import cv2
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

PROMPT = "find the bin with banana"
BIMANUAL_DOF = 14


@dataclasses.dataclass
class Args:
    # --- Policy server (remote GPU box) ---
    host: str = "10.79.12.252"
    port: int = 8000

    # --- Local gello ZMQ nodes (started by your teleop launch_yaml.py) ---
    zmq_host: str = "127.0.0.1"
    robot_port: int = 6001
    """launch_yaml.py's hardware_server_port (direct-hardware default 6001)."""
    top_camera_port: int = 5000
    left_camera_port: int = 5001
    right_camera_port: int = 5002
    """Camera node ports: launch_yaml.py auto-assigns from 5000 in the yaml's camera order
    (yam_left.yaml: top, left, right -> 5000, 5001, 5002)."""

    # --- Prediction cadence ---
    pred_interval: float = 10 / 30
    """Seconds between server calls. One call = one memory write; the training cadence is
    memory_stride_frames=10 at 30 Hz."""
    prompt: str = PROMPT
    reset_on_start: bool = True
    """Send a memory reset to the server before the first prediction (fresh episode)."""

    # --- Display / recording ---
    display_hz: float = 15.0
    """Top-camera refresh rate between predictions (kept low: reads share the camera node
    with the teleop loop)."""
    record: bool = True
    record_dir: str = "eval"
    record_path: str = ""

    # --- Debug ---
    no_state: bool = False
    """Do not contact the robot node; send zero joint state instead. Only for camera-only
    debugging -- the state conditioning will be off-distribution."""
    dry_run: bool = False
    """No hardware: send random observations to validate the obs/subtask contract."""


class _H264Writer:
    """Encode RGB frames to an H.264 mp4 via the system ffmpeg (same as client_subtask.py)."""

    def __init__(self, path: str, width: int, height: int, fps: float):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg not found on PATH -- needed to encode the recording")
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{fps}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
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


def _overlay(frame_rgb: np.ndarray, subtask: str, status: str) -> np.ndarray:
    """Subtask (dark green) + status line (yellow) on an RGB frame."""
    img = np.ascontiguousarray(frame_rgb).copy()
    cv2.putText(img, f"subtask: {subtask}", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 0), 2, cv2.LINE_AA)
    cv2.putText(img, status, (12, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 60), 1, cv2.LINE_AA)
    return img


class _Display:
    """Window thread: shows the latest overlaid frame, captures 'r' (reset) and 'q' (quit)."""

    def __init__(self, window: str = "pi05 yam memory - live subtask"):
        self._window = window
        self._lock = threading.Lock()
        self._img: np.ndarray | None = None  # RGB, already overlaid
        self.reset_requested = threading.Event()
        self.quit_requested = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def update(self, img_rgb: np.ndarray) -> None:
        with self._lock:
            self._img = img_rgb

    def _loop(self) -> None:
        cv2.namedWindow(self._window, cv2.WINDOW_NORMAL)
        while not self._stop.is_set():
            with self._lock:
                img = None if self._img is None else self._img.copy()
            if img is not None:
                cv2.imshow(self._window, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(30) & 0xFF
            if key == ord("r"):
                self.reset_requested.set()
            elif key == ord("q"):
                self.quit_requested.set()
        cv2.destroyAllWindows()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def _wait_for_node(host: str, port: int, payload: bytes, what: str) -> None:
    """Block until a gello ZMQ node answers a probe request, logging what we're waiting for.

    The nodes are created by your teleop launch (launch_yaml.py); a plain ZMQ request to a
    port nobody serves waits forever with no output, so probe with a timeout instead.
    """
    import zmq

    ctx = zmq.Context.instance()
    announced = False
    while True:
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 2000)
        sock.setsockopt(zmq.SNDTIMEO, 2000)
        sock.connect(f"tcp://{host}:{port}")
        try:
            sock.send(payload)
            sock.recv()
            logging.info("%s on %s:%d is up.", what, host, port)
            return
        except zmq.error.Again:
            if not announced:
                logging.info(
                    "Waiting for %s on %s:%d -- start your teleop (launch_yaml.py) first...",
                    what,
                    host,
                    port,
                )
                announced = True
        finally:
            sock.close()
        time.sleep(1.0)


def _run_dry(policy, args: Args) -> None:
    """Validate the obs/subtask contract with random data -- no hardware needed."""
    rng = np.random.default_rng(0)
    logging.info("Dry run: memory reset + 3 random observations...")
    logging.info("  reset: %s", policy.infer({"reset_memory": True}))
    for i in range(3):
        result = policy.infer(
            {
                "observation/state": rng.random(BIMANUAL_DOF).astype(np.float32),
                "observation/image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
                "observation/left_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
                "observation/right_wrist_image": rng.integers(256, size=(480, 640, 3), dtype=np.uint8),
                "prompt": args.prompt,
            }
        )
        assert isinstance(result.get("subtask"), str), f"missing subtask, got {result.get('subtask')!r}"
        assert np.isfinite(result["surprise"]), "non-finite surprise"
        logging.info(
            "  call %d: subtask=%r surprise=%.3f writes=%d gates=%s infer=%.0f ms",
            i,
            result["subtask"],
            result["surprise"],
            result["writes"],
            result["gates"],
            result["policy_timing"]["infer_ms"],
        )
    logging.info("Dry run OK.")


def main(args: Args) -> None:
    policy = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logging.info("Server metadata: %s", policy.get_server_metadata())

    if args.dry_run:
        _run_dry(policy, args)
        return

    # --- Attach (read-only) to the teleop's ZMQ nodes ---
    from gello.zmq_core.camera_node import ZMQClientCamera
    from gello.zmq_core.robot_node import ZMQClientRobot

    for port, what in (
        (args.top_camera_port, "top camera node"),
        (args.left_camera_port, "left camera node"),
        (args.right_camera_port, "right camera node"),
    ):
        _wait_for_node(args.zmq_host, port, pickle.dumps(None), what)
    if not args.no_state:
        _wait_for_node(args.zmq_host, args.robot_port, pickle.dumps({"method": "num_dofs"}), "robot node")

    top = ZMQClientCamera(port=args.top_camera_port, host=args.zmq_host)
    left = ZMQClientCamera(port=args.left_camera_port, host=args.zmq_host)
    right = ZMQClientCamera(port=args.right_camera_port, host=args.zmq_host)
    robot = None if args.no_state else ZMQClientRobot(port=args.robot_port, host=args.zmq_host)

    def joint_state() -> np.ndarray:
        if robot is None:
            return np.zeros(BIMANUAL_DOF, dtype=np.float32)
        state = np.asarray(robot.get_observations()["joint_positions"], dtype=np.float32)
        assert state.shape == (BIMANUAL_DOF,), f"expected 14-dim state, got {state.shape}"
        return state

    frame0, _ = top.read()
    frame0 = image_tools.convert_to_uint8(frame0)
    logging.info("Attached: top %s | state %s", frame0.shape, joint_state().shape)

    if args.reset_on_start:
        policy.infer({"reset_memory": True})
        logging.info("Memory reset (fresh episode).")

    display = _Display()
    writer = None
    record_path = ""
    if args.record:
        record_path = args.record_path
        if not record_path:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
            os.makedirs(args.record_dir, exist_ok=True)
            record_path = os.path.join(args.record_dir, f"memory_viz_{stamp}.mp4")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(record_path)) or ".", exist_ok=True)
        writer = _H264Writer(record_path, frame0.shape[1], frame0.shape[0], args.display_hz)
        logging.info("Recording overlaid top camera to %s @ %.1f Hz", record_path, args.display_hz)

    subtask, status = "", "waiting for first prediction..."
    next_pred = 0.0
    frames_written = 0
    try:
        while not display.quit_requested.is_set():
            t_loop = time.monotonic()

            if display.reset_requested.is_set():
                display.reset_requested.clear()
                policy.infer({"reset_memory": True})
                subtask, status = "", "memory reset -- new episode"
                logging.info("Memory reset (keyboard).")

            if t_loop >= next_pred:
                # one prediction = one memory write, at the training cadence
                frame, _ = top.read()
                frame = image_tools.convert_to_uint8(frame)
                left_img, _ = left.read()
                right_img, _ = right.read()
                result = policy.infer(
                    {
                        "observation/state": joint_state(),
                        "observation/image": frame,
                        "observation/left_wrist_image": image_tools.convert_to_uint8(left_img),
                        "observation/right_wrist_image": image_tools.convert_to_uint8(right_img),
                        "prompt": args.prompt,
                    }
                )
                subtask = str(result["subtask"])
                status = (
                    f"surprise {result['surprise']:.3f} | writes {result['writes']} | "
                    f"infer {result['policy_timing']['infer_ms']:.0f} ms | r=reset q=quit"
                )
                logging.info("write %3d | surprise %.3f | %s", result["writes"], result["surprise"], subtask)
                next_pred = t_loop + args.pred_interval
            else:
                frame, _ = top.read()
                frame = image_tools.convert_to_uint8(frame)

            img = _overlay(frame, subtask, status)
            display.update(img)
            if writer is not None:
                writer.write(img)
                frames_written += 1

            # pace the loop to display_hz (the prediction branch above rides the same loop)
            dt = time.monotonic() - t_loop
            time.sleep(max(0.0, 1.0 / args.display_hz - dt))
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        if writer is not None:
            writer.release()
            logging.info("Saved recording: %s (%d frames)", record_path, frames_written)
        display.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
