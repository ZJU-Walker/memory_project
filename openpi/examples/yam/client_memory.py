"""Robot-side inference client for the *memory* checkpoint (`pi05_yam_memory` / `Pi0Memory`).

Runs on the robot computer, alongside `client_base.py`. Reads YAM state + the three RealSense cameras
directly and talks to a stateful memory server (`scripts/serve_policy_memory.py`) that maintains the
online episodic memory `(mem, surprise)` server-side.

Cadence (per the design): memory writes are decoupled from the control loop. A background daemon
thread sends cheap **write-only** messages (server runs `memory_write`, no LLM) at ~``write_hz``
(default 3 Hz, matching the training write spacing), always using the most recent frame published
by the control loop. The 30 Hz control
loop itself never blocks on writes; it only sends an **action** message when the current 50-step
action chunk is exhausted (server writes that frame *and* samples a fresh chunk conditioned on the
current memory). Both threads share one websocket, serialized by a lock. At episode start we send a
**reset** so memory clears to M_0. Server responses carry ``policy_timing`` with ``write_ms`` (memory
update) and ``query_ms`` (action sampling), logged per step and echoed to the console.

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

import csv
import dataclasses
import datetime
import json
import logging
import os
import shutil
import subprocess
import threading
import time

import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

PROMPT = "find the bin with banana"
BIMANUAL_DOF = 14

# Per-step scalar columns pulled from the server response (memory diagnostics + timing).
_STAT_COLUMNS = ("mem_delta", "surprise_norm", "mem_norm", "infer_ms", "write_ms", "query_ms")


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


class _StepLogger:
    """Append one CSV row per control step for offline memory-dynamics analysis.

    Columns: step, t (s since start), kind (write|action), chunk_i, the memory-stat scalars from
    the server (mem_delta, surprise_norm, mem_norm, timing), a per-step d(mem_delta), the commanded
    14-dim action, and the measured 14-dim joint state. Flushed every row so Ctrl+C keeps the data.
    ``write`` rows come from the background writer thread (NaN action/state, chunk_i=-1); ``log`` is
    lock-guarded so the two threads can't interleave rows.

    Also writes a sidecar ``<path>.meta.json`` with the run args + column layout for reproducibility.
    """

    def __init__(self, path: str, args: "Args"):
        self._f = open(path, "w", newline="")  # noqa: SIM115 (kept open for the run; closed in release)
        self._w = csv.writer(self._f)
        self._prev_mem_delta = None
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        header = (
            ["step", "t", "kind", "chunk_i", *_STAT_COLUMNS, "d_mem_delta"]
            + [f"action_{i}" for i in range(BIMANUAL_DOF)]
            + [f"state_{i}" for i in range(BIMANUAL_DOF)]
        )
        self._w.writerow(header)
        self._f.flush()
        with open(path + ".meta.json", "w") as mf:
            json.dump({"args": dataclasses.asdict(args), "columns": header}, mf, indent=2)

    def log(self, step: int, kind: str, chunk_i: int, resp: dict, action: np.ndarray, state: np.ndarray) -> None:
        timing = resp.get("policy_timing", {}) or {}
        stats = {
            "mem_delta": resp.get("mem_delta", float("nan")),
            "surprise_norm": resp.get("surprise_norm", float("nan")),
            "mem_norm": resp.get("mem_norm", float("nan")),
            "infer_ms": timing.get("infer_ms", float("nan")),
            "write_ms": timing.get("write_ms", float("nan")),
            "query_ms": timing.get("query_ms", float("nan")),
        }
        with self._lock:
            md = stats["mem_delta"]
            d_md = float("nan") if self._prev_mem_delta is None or md != md else md - self._prev_mem_delta
            if md == md:  # not NaN
                self._prev_mem_delta = md
            row = (
                [step, round(time.monotonic() - self._t0, 4), kind, chunk_i]
                + [stats[c] for c in _STAT_COLUMNS]
                + [d_md]
                + [round(float(a), 6) for a in action]
                + [round(float(s), 6) for s in state]
            )
            self._w.writerow(row)
            self._f.flush()

    def release(self) -> None:
        self._f.close()


class _BackgroundMemoryWriter:
    """Send write-only memory updates to the server at ~``hz`` from a daemon thread.

    The control loop publishes its latest observation via ``update()``; each tick this thread
    snapshots it (consume-once, so a stalled control loop never writes the same frame twice),
    sends a ``write_only`` message, and hands the response to ``on_result(step, resp)`` (logging).
    The websocket is shared with the control loop's action requests via ``ws_lock`` -- worst-case
    an action request waits one write (~write_ms) for the lock.
    """

    def __init__(self, ws, ws_lock: threading.Lock, hz: float, prompt: str, on_result):
        self._ws = ws
        self._ws_lock = ws_lock
        self._period = 1.0 / hz
        self._prompt = prompt
        self._on_result = on_result
        self._latest: tuple[int, dict] | None = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="memory-writer", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def update(self, step: int, obs: dict) -> None:
        """Publish the newest frame (called from the control loop; never blocks on the network)."""
        with self._latest_lock:
            self._latest = (step, obs)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            with self._latest_lock:
                latest, self._latest = self._latest, None
            if latest is not None:
                step, obs = latest
                try:
                    req = _obs_to_request(obs, self._prompt)
                    with self._ws_lock:
                        out = self._ws.infer({**req, "write_only": True})
                    self._on_result(step, out)
                except Exception:
                    logging.exception("Background memory write failed (step %d)", step)
            self._stop.wait(max(0.0, self._period - (time.monotonic() - t0)))


@dataclasses.dataclass
class Args:
    # --- Policy server (remote GPU box) ---
    host: str = "10.79.12.149"
    port: int = 8000

    # --- Inference / control ---
    action_horizon: int = 50
    """Steps executed from each sampled action chunk before requesting a new one. <= the model's
    action_horizon (pi05_yam_memory = 50); a smaller value re-samples (and re-reads memory) sooner."""
    max_steps: int = 12000
    hz: float = 30.0
    write_hz: float = 3.0
    """Rate of the background memory-write thread. Writes run off the control loop's latest frame in
    a daemon thread, so the control loop only blocks on the network for action requests. Default 3.0
    matches the training write spacing (stride-10 frames at the dataset's 30 fps)."""
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

    # --- Per-step memory log (CSV for offline analysis) ---
    log: bool = True
    """Write one CSV row per control step (memory stats + action + state) for analysis."""
    log_path: str = ""
    """Output .csv path. Empty -> `<record_dir>/mem_log_<timestamp>.csv` (+ a .meta.json sidecar)."""

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
    logging.info(
        "  after reset+write: mem_delta=%.4g surprise=%.4g mem_norm=%.4g",
        r.get("mem_delta", float("nan")), r.get("surprise_norm", float("nan")), r.get("mem_norm", float("nan")),
    )
    prev = r.get("mem_delta", 0.0)
    for i in range(10):
        r = ws.infer(req(write_only=True))
        d = r.get("mem_delta", float("nan"))
        logging.info(
            "  write %d: mem_delta=%.4g (d=%+.4g) surprise=%.4g write_ms=%.1f",
            i, d, d - prev, r.get("surprise_norm", float("nan")),
            (r.get("policy_timing", {}) or {}).get("write_ms", float("nan")),
        )
        prev = d
    out = ws.infer(req())  # action step
    actions = np.asarray(out["actions"])
    assert actions.shape == (args.action_horizon, BIMANUAL_DOF), f"got {actions.shape}"
    assert np.all(np.isfinite(actions)), "non-finite actions"
    timing = out.get("policy_timing", {}) or {}
    logging.info(
        "  action step: actions shape=%s finite=True write_ms=%.1f query_ms=%.1f",
        actions.shape, timing.get("write_ms", float("nan")), timing.get("query_ms", float("nan")),
    )

    # Exercise the background writer for ~2 s (same code path the control loop uses).
    ws_lock = threading.Lock()
    writes: list[float] = []
    mem_writer = _BackgroundMemoryWriter(
        ws, ws_lock, args.write_hz, args.prompt,
        lambda _step, resp: writes.append((resp.get("policy_timing", {}) or {}).get("write_ms", float("nan"))),
    )
    mem_writer.start()
    for i in range(20):
        # The writer thread runs _obs_to_request itself, so publish raw robot-format obs.
        mem_writer.update(i, {
            "joint_positions": np.random.rand(BIMANUAL_DOF).astype(np.float32),
            "top_camera_rgb": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "left_camera_rgb": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "right_camera_rgb": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        })
        time.sleep(0.1)
    mem_writer.stop()
    logging.info(
        "  background writer: %d writes in ~2 s at %.1f Hz (expect ~%d), mean write_ms=%.1f",
        len(writes), args.write_hz, int(2 * args.write_hz), float(np.nanmean(writes)) if writes else float("nan"),
    )
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
    if "mem_delta" not in out:
        # The stock stateless server (serve_policy.py) answers these requests too -- it just ignores
        # reset/write_only and never returns memory stats, so the run silently has no memory.
        raise RuntimeError(
            f"Server at {args.host}:{args.port} is not a memory server (no mem stats in response). "
            "Launch it with scripts/serve_policy_memory.py --policy.config=pi05_yam_memory."
        )
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

    # --- Set up per-step memory log (CSV, flushed each row so Ctrl+C keeps it) ---
    step_logger = None
    log_path = args.log_path
    if args.log:
        if not log_path:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(args.record_dir, exist_ok=True)
            log_path = os.path.join(args.record_dir, f"mem_log_{stamp}.csv")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        step_logger = _StepLogger(log_path, args)
        logging.info("Logging per-step memory stats to %s (+ %s.meta.json)", log_path, log_path)

    # --- Background memory writer (writes at ~write_hz off the control loop's latest frame) ---
    ws_lock = threading.Lock()
    # Seed console stats from the reset response so the first log lines aren't all NaN.
    last_write: dict = {k: out[k] for k in ("mem_delta", "surprise_norm", "mem_norm", "policy_timing") if k in out}
    last_action_timing: dict = out.get("policy_timing", {}) or {}

    def _on_write(write_step: int, resp: dict) -> None:
        last_write.update(resp)
        if step_logger is not None:
            nan14 = np.full(BIMANUAL_DOF, np.nan)
            step_logger.log(write_step, "write", -1, resp, nan14, nan14)

    mem_writer = _BackgroundMemoryWriter(ws, ws_lock, args.write_hz, args.prompt, _on_write)
    mem_writer.start()
    logging.info("Background memory writer running at %.1f Hz", args.write_hz)

    # --- Control loop ---
    logging.info("Starting control loop: %d steps @ %.1f Hz", args.max_steps, args.hz)
    obs = env.get_obs()
    try:
        for step in range(args.max_steps):
            if writer is not None:
                # obs["top_camera_rgb"] is the RGB frame the policy sees this step.
                writer.write(image_tools.convert_to_uint8(obs["top_camera_rgb"]))
                frames_written += 1
            # Publish this frame to the background writer (non-blocking; it writes at write_hz).
            mem_writer.update(step, obs)
            out = None
            if chunk_i >= args.action_horizon:
                # Chunk exhausted: action message (server writes this frame + samples a new chunk).
                with ws_lock:
                    out = ws.infer(_obs_to_request(obs, args.prompt))
                chunk = np.asarray(out["actions"], dtype=np.float64)
                chunk_i = 0
                last_action_timing = out.get("policy_timing", {}) or {}
                logging.info(
                    "  action step %d: write_ms=%.1f query_ms=%.1f",
                    step,
                    last_action_timing.get("write_ms", float("nan")),
                    last_action_timing.get("query_ms", float("nan")),
                )

            cur = np.asarray(obs["joint_positions"], dtype=np.float64)
            action = _clamp_joint_delta(chunk[chunk_i], cur, args.max_joint_delta)
            if step_logger is not None and out is not None:
                step_logger.log(step, "action", chunk_i, out, action, cur)
            chunk_i += 1
            obs = env.step(action)

            if step % args.hz == 0:
                write_timing = last_write.get("policy_timing", {}) or {}
                logging.info(
                    "  step %d / %d  mem_delta=%.4g  surprise=%.4g  write_ms=%.1f  query_ms=%.1f",
                    step, args.max_steps,
                    last_write.get("mem_delta", float("nan")),
                    last_write.get("surprise_norm", float("nan")),
                    write_timing.get("write_ms", float("nan")),
                    last_action_timing.get("query_ms", float("nan")),
                )
    except KeyboardInterrupt:
        logging.info("Interrupted by user -- stopping (arms left in place).")
    finally:
        mem_writer.stop()
        if step_logger is not None:
            step_logger.release()
            logging.info("Saved memory log: %s", log_path)
        if writer is not None:
            writer.release()
            logging.info("Saved recording: %s (%d frames)", record_path, frames_written)
        logging.info("Control loop finished.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
