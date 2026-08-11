"""Closed-loop robot client for the pi05 MEMORY model on the bimanual YAM.

Like examples/yam/client_subtask.py (direct in-process hardware: YAM arms over CAN + the three
RealSense cameras), but against a `scripts/serve_yam_memory.py` server: the policy reads the
Titans memory, decodes the subtask, denoises the actions and writes the frame into the memory,
whose state is threaded on the server across requests. `RealtimeActionChunkBroker` starts the
next inference after 10 of the 50 predicted actions, keeps executing the old chunk in the
foreground, and sends those in-flight actions as an RTC prefix. One replan = one memory write,
matching v3's 10-frame training stride at the default 30 Hz control rate.

A live OpenCV window shows the top camera with the predicted subtask + surprise overlaid.
Keys in the window:  r = reset the server-side memory AND fetch a fresh chunk (new episode)
                     q = stop (arms left in place).

Server (GPU box):
    uv run scripts/serve_yam_memory.py \
        --dir checkpoints/pi05_yam_mem_v31/<experiment>/<step> \
        --config pi05_yam_mem_v31

Client (robot computer; needs `gello_software`, `openpi_client` and opencv importable):
    python examples/yam/client_memory.py --host <gpu-host> --port 8000

Smoke-test the obs/action/memory contract without hardware:
    python examples/yam/client_memory.py --host <gpu-host> --port 8000 --dry-run
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
BIMANUAL_DOF = 14


@dataclasses.dataclass
class Args:
    # --- Policy server (remote GPU box) ---
    host: str = "10.79.12.252"
    port: int = 8000

    # --- Inference / control ---
    action_horizon: int = 50
    """Full model action horizon. Must match the server checkpoint."""
    steps_between_inference: int = 10
    """Start the next asynchronous inference after 10 controls, leaving a 40-action overlap."""
    initial_delay_steps: int = 6
    """Conservative initial latency estimate (6 steps = 200 ms at 30 Hz)."""
    max_async_delay_steps: int = 6
    """Never execute more unconfirmed steps than the largest RTC delay seen in training."""
    delay_tolerance_steps: int = 0
    """Block at the predicted delay rather than leaving the RTC-trained prefix range."""
    delay_buffer_size: int = 8
    """Number of measured inference delays retained by the conservative estimator."""
    max_steps: int = 12000
    hz: float = 30.0
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
    """Record the top camera view (with the overlay); saved on exit (incl. Ctrl+C)."""
    record_dir: str = "eval"
    record_path: str = ""

    # --- Debug ---
    dry_run: bool = False
    """Skip hardware: validate the RTC replan and obs/action/subtask/memory contract."""
    dry_run_steps: int = 25
    """Control steps in a dry run. Keep above 16 to exercise an asynchronous RTC replan."""


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
    """Window thread showing the latest overlaid frame; captures 'r' (reset) and 'q' (quit)."""

    def __init__(self, window: str = "pi05 yam memory - closed loop"):
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


def _obs_to_request(obs: dict, prompt: str) -> dict:
    """Map a `RobotEnv.get_obs()` dict to the websocket observation the server expects."""
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


def _run_dry(ws_client, policy, args: Args) -> None:
    """Validate the RTC replan and obs/action/subtask/memory contract without hardware."""
    if args.dry_run_steps <= args.steps_between_inference + args.max_async_delay_steps:
        raise ValueError(
            "dry_run_steps must exceed steps_between_inference + max_async_delay_steps "
            "so the test waits for at least one asynchronous RTC replan"
        )
    rng = np.random.default_rng(0)
    logging.info("Dry run: memory reset + %d random control observations...", args.dry_run_steps)
    logging.info("  reset: %s", ws_client.infer({"reset_memory": True}))
    max_writes = 0
    for i in range(args.dry_run_steps):
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
        assert isinstance(result.get("subtask"), str), f"missing subtask, got {result.get('subtask')!r}"
        assert np.isfinite(result["surprise"]), "non-finite surprise"
        max_writes = max(max_writes, int(result["writes"]))
        logging.info(
            "  step %d: action=%s subtask=%r surprise=%.3f writes=%d",
            i,
            action.shape,
            result["subtask"],
            result["surprise"],
            result["writes"],
        )
    assert max_writes >= 2, f"RTC dry run never installed a replanned chunk; maximum server writes was {max_writes}"
    logging.info("Dry run OK -- RTC replan and obs/action/subtask/memory contract match.")


def _validate_rtc_metadata(metadata: dict, args: Args) -> None:
    """Fail before touching hardware when the client/server RTC contracts differ."""
    if metadata.get("config_name") != "pi05_yam_mem_v31":
        raise ValueError(
            "real-robot v3.1 evaluation requires config_name='pi05_yam_mem_v31'; "
            f"server advertised {metadata.get('config_name')!r}"
        )
    if metadata.get("memory_write_source") != "post_attention":
        raise ValueError(
            "real-robot v3.1 evaluation requires memory_write_source='post_attention'; "
            f"server advertised {metadata.get('memory_write_source')!r}"
        )
    horizon = metadata.get("action_horizon")
    if horizon != args.action_horizon:
        raise ValueError(
            f"server action_horizon is {horizon!r}, but the client is configured for {args.action_horizon}"
        )
    if metadata.get("rtc_enabled") is not True:
        raise ValueError("the server checkpoint is not RTC-trained (rtc_enabled must be true)")
    if metadata.get("rtc_delay_semantics") != "inclusive_max":
        raise ValueError(
            "the server does not advertise the expected inclusive RTC delay semantics; "
            f"got {metadata.get('rtc_delay_semantics')!r}"
        )
    trained_max_delay = metadata.get("rtc_max_delay")
    if not isinstance(trained_max_delay, int) or args.max_async_delay_steps > trained_max_delay:
        raise ValueError(
            f"client max_async_delay_steps={args.max_async_delay_steps} exceeds or cannot verify the "
            f"server's trained RTC maximum ({trained_max_delay!r})"
        )
    training_stride = metadata.get("memory_stride_frames")
    if training_stride != args.steps_between_inference:
        raise ValueError(
            f"server memory_stride_frames is {training_stride!r}, but the client replans every "
            f"{args.steps_between_inference} steps"
        )


def main(args: Args) -> None:
    # --- Connect to the policy server ---
    ws_client = _websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = ws_client.get_server_metadata()
    logging.info("Server metadata: %s", metadata)
    _validate_rtc_metadata(metadata, args)
    policy = action_chunk_broker.RealtimeActionChunkBroker(
        ws_client,
        action_horizon=args.action_horizon,
        steps_between_inference=args.steps_between_inference,
        initial_delay_steps=args.initial_delay_steps,
        delay_tolerance_steps=args.delay_tolerance_steps,
        max_async_delay_steps=args.max_async_delay_steps,
        delay_buffer_size=args.delay_buffer_size,
    )

    if args.dry_run:
        try:
            _run_dry(ws_client, policy, args)
        finally:
            policy.close()
        return

    # --- Build hardware (direct, in-process; same as client_subtask.py) ---
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
    for key in ("top_camera_rgb", "left_camera_rgb", "right_camera_rgb"):
        assert key in obs, f"missing camera obs '{key}'"
    assert np.asarray(obs["joint_positions"]).shape == (BIMANUAL_DOF,)

    # --- Ramp to the first inferred target (avoid a large jump from rest) ---
    # The server memory survives websocket reconnects, so clear any previous client/episode
    # before even computing the ramp target. We reset once more after the ramp because that
    # first inference itself performs a write.
    ws_client.infer({"reset_memory": True})
    logging.info("Memory reset before computing the ramp target.")
    logging.info("Ramping to first policy target...")
    first_target = np.asarray(policy.infer(_obs_to_request(obs, args.prompt))["actions"], dtype=np.float64)
    for _ in range(25):
        obs = env.get_obs()
        cur = np.asarray(obs["joint_positions"], dtype=np.float64)
        if float(np.abs(first_target - cur).max()) < 1e-2:
            break
        env.step(_clamp_joint_delta(first_target, cur, args.max_joint_delta))
    # Fresh chunk AND fresh memory for the episode (the ramp query already wrote once).
    policy.reset()
    ws_client.infer({"reset_memory": True})
    logging.info("Memory reset -- episode starts fresh.")

    display = _Display() if args.show else None

    # --- Top camera recording (written incrementally so Ctrl+C still saves it) ---
    writer = None
    record_path = ""
    frames_written = 0
    if args.record:
        record_path = args.record_path
        if not record_path:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005
            os.makedirs(args.record_dir, exist_ok=True)
            record_path = os.path.join(args.record_dir, f"memory_closedloop_{stamp}.mp4")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(record_path)) or ".", exist_ok=True)
        first_frame = image_tools.convert_to_uint8(obs["top_camera_rgb"])
        writer = _H264Writer(record_path, first_frame.shape[1], first_frame.shape[0], args.hz)
        logging.info("Recording top camera (H.264) to %s @ %.1f Hz", record_path, args.hz)

    # --- Control loop (paced to `hz` by RobotEnv.step's internal Rate) ---
    logging.info(
        "Starting RTC control loop: %d steps @ %.1f Hz (replan/write every %d steps, horizon %d)",
        args.max_steps,
        args.hz,
        args.steps_between_inference,
        args.action_horizon,
    )
    obs = env.get_obs()
    subtask, status = "", ""
    try:
        for step in range(args.max_steps):
            if display is not None and display.quit_requested.is_set():
                logging.info("Quit requested from the display window.")
                break
            if display is not None and display.reset_requested.is_set():
                display.reset_requested.clear()
                policy.reset()  # drop the cached chunk: it was computed with the old memory
                ws_client.infer({"reset_memory": True})
                logging.info("Memory reset (keyboard) -- new episode.")

            result = policy.infer(_obs_to_request(obs, args.prompt))
            action = np.asarray(result["actions"], dtype=np.float64)
            subtask = str(result.get("subtask", subtask))
            status = (
                f"surprise {result.get('surprise', float('nan')):.3f} | writes {result.get('writes', 0)} | "
                f"step {step} | r=reset q=quit"
            )

            frame = image_tools.convert_to_uint8(obs["top_camera_rgb"])
            img = _overlay(frame, subtask, status)
            if display is not None:
                display.update(img)
            if writer is not None:
                writer.write(img)
                frames_written += 1

            cur = np.asarray(obs["joint_positions"], dtype=np.float64)
            action = _clamp_joint_delta(action, cur, args.max_joint_delta)
            obs = env.step(action)
            if step % args.steps_between_inference == 0:
                logging.info(
                    "  step %d | writes %s | surprise %s | subtask: %s",
                    step,
                    result.get("writes"),
                    result.get("surprise"),
                    subtask,
                )
    except KeyboardInterrupt:
        logging.info("Interrupted by user -- stopping (arms left in place).")
    finally:
        if writer is not None:
            writer.release()
            logging.info("Saved recording: %s (%d frames)", record_path, frames_written)
        if display is not None:
            display.close()
        policy.close()
        logging.info("Control loop finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
