"""Pi 0.5 real-robot eval client (YAM single-arm).

Runs on the robot computer. Reads YAM joint state + 3 RealSense cameras,
sends observations to a remote inference server over WebSocket, receives
10-step action chunks, executes them open-loop on the arm, then re-infers.

Usage:
    cd /home/david/ke/memory_project
    python eval/real/run_eval.py --host <gpu_box_ip>            # default port 8000
    python eval/real/run_eval.py --host 192.168.1.50 --dry-run  # no motion
    python eval/real/run_eval.py --host 192.168.1.50 --max-steps 60 --max-time-s 5

Stop:
    - 'q' in the pygame window (click the window first; needs a display —
      use `ssh -Y` for X11 forwarding when running over SSH)
    - Ctrl-C — graceful (waits for current chunk to finish); press twice to force kill
    - --max-steps / --max-time-s timeouts
    - Last resort from another shell: `pkill -9 -f run_eval`

On graceful stop the arm holds the last measured pose for ~0.5 s, then exits.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from openpi_client.websocket_client_policy import WebsocketClientPolicy

# Local imports — `python eval/real/run_eval.py` from memory_project root works
# because we're already on the package path; if invoked from elsewhere, prepend
# the project root.
_THIS_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from eval.real.logger import RolloutLogger  # noqa: E402
from eval.real.robot_env import YamPolicyEnv  # noqa: E402

DEFAULT_PROMPT = "pick up the cup with yellow object inside"
ACTION_HORIZON = 10
ACTION_DIM = 7
JOINT_LIMIT_RAD = 3.0  # sanity bound for any single joint reading
FIRST_CHUNK_MAX_JUMP = 0.3  # rad; abort if first commanded pose differs by more


# ---------------------------------------------------------------------------
# Stop signal helpers
# ---------------------------------------------------------------------------


class StopFlag:
    """Stop coordinator backed by a pygame keyboard window + SIGINT escape.

    - pygame window: tries to open a small window; press 'q' (after clicking
      it) to stop. Works locally or over SSH with X11 forwarding (`ssh -Y`).
    - SIGINT (Ctrl-C): first one sets stop flag (graceful: hold-pose then
      exit); a second SIGINT calls os._exit(130) for a hard kill in case
      anything downstream is swallowing the first signal.
    """

    def __init__(self) -> None:
        self._stop = False
        self._reason: Optional[str] = None
        self._sigint_count = 0

        signal.signal(signal.SIGINT, self._on_sigint)

        self._kb = None
        try:
            self._kb = _KeyWindow()
            print("[stop] pygame window open — click it then press 'q' to stop")
        except Exception as e:
            print(f"[stop] pygame window unavailable ({e}); use Ctrl-C")
        print("[stop] Ctrl-C also works (twice = hard kill)")

    def _on_sigint(self, *_) -> None:
        self._sigint_count += 1
        if self._sigint_count >= 2:
            print("\n[stop] second Ctrl-C — force exit", flush=True)
            os._exit(130)
        self._stop = True
        self._reason = "sigint"
        print("\n[stop] SIGINT received — finishing chunk and holding pose. Ctrl-C again to force.", flush=True)

    def poll(self) -> None:
        if self._kb is not None and self._kb.q_pressed():
            self._stop = True
            self._reason = "user-pygame"

    @property
    def stopped(self) -> bool:
        return self._stop

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def close(self) -> None:
        if self._kb is not None:
            self._kb.close()


class _KeyWindow:
    """Tiny pygame window that captures 'q'. Used only when a display is available."""

    def __init__(self) -> None:
        import pygame  # noqa: F401

        self._pygame = pygame
        pygame.init()
        self._screen = pygame.display.set_mode((420, 80))
        pygame.display.set_caption("Pi 0.5 eval — press 'q' to stop")
        font = pygame.font.SysFont(None, 22)
        self._screen.fill((20, 20, 20))
        msg = font.render("Click here, then press 'q' to stop", True, (240, 240, 240))
        self._screen.blit(msg, (10, 30))
        pygame.display.flip()

    def q_pressed(self) -> bool:
        self._pygame.event.pump()
        for event in self._pygame.event.get():
            if event.type == self._pygame.KEYDOWN and event.key == self._pygame.K_q:
                return True
        return False

    def close(self) -> None:
        self._pygame.quit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def smooth_move(env: YamPolicyEnv, target: np.ndarray, steps: int, dt: float) -> None:
    """Linearly interpolate from current state to `target`. Mirrors gello replay_demo.smooth_move."""
    current = env.current_state()
    target = np.asarray(target, dtype=np.float32)
    for jnt in np.linspace(current, target, steps):
        env.command(jnt.astype(np.float32))
        time.sleep(dt)


def sleep_with_poll(seconds: float, stop: StopFlag) -> None:
    """Sleep up to `seconds`, polling the stop flag every 5 ms."""
    if seconds <= 0:
        return
    end = time.monotonic() + seconds
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        stop.poll()
        if stop.stopped:
            return
        time.sleep(min(remaining, 0.005))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", required=True, help="Inference server host (e.g. 192.168.1.50)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", type=Path, default=_THIS_DIR / "configs" / "yam_eval.yaml")
    p.add_argument("--prompt", type=str, default=None, help=f"Override prompt (default: from yaml or {DEFAULT_PROMPT!r})")
    p.add_argument("--ckpt-tag", type=str, default="", help="Free-form tag written to metadata.json")
    p.add_argument("--max-steps", type=int, default=900, help="Control-step cap (~30s @ 30Hz)")
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--chunk-len", type=int, default=ACTION_HORIZON, help=f"How many of the {ACTION_HORIZON} returned actions to execute per inference")
    p.add_argument("--start-steps", type=int, default=60, help="Smooth-move interpolation steps to first commanded pose")
    p.add_argument("--log-dir", type=Path, default=None, help="Default: eval/real/runs/<timestamp>")
    p.add_argument("--no-video", action="store_true", help="Skip MP4 recording (top/left/right_camera_rgb.mp4 from background grabber threads at 30 fps)")
    p.add_argument("--no-save-frames", action="store_true", help="Skip per-chunk JPG dump (frames/chunk_NNNN_*.jpg + latest_*.jpg)")
    p.add_argument("--dry-run", action="store_true", help="Pre-flight + 1 inference, no motion")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.chunk_len <= ACTION_HORIZON:
        sys.exit(f"--chunk-len must be in [1, {ACTION_HORIZON}], got {args.chunk_len}")

    cfg = yaml.safe_load(args.config.read_text())
    prompt = args.prompt or cfg.get("prompt", DEFAULT_PROMPT)
    log_dir = args.log_dir or (_THIS_DIR / "runs" / _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_dir = Path(log_dir)
    print(f"[init] config       : {args.config}")
    print(f"[init] prompt       : {prompt!r}")
    print(f"[init] server       : ws://{args.host}:{args.port}")
    print(f"[init] log_dir      : {log_dir}")

    # -- env setup. Create log_dir up-front so the background camera streamers
    # can stream MP4s into it from the moment they start.
    log_dir.mkdir(parents=True, exist_ok=True)
    print("[init] connecting to YAM + cameras...")
    env = YamPolicyEnv(
        channel=cfg["robot"]["channel"],
        top_serial=cfg["cameras"]["top_serial"],
        left_serial=cfg["cameras"]["left_serial"],
        right_serial=cfg["cameras"]["right_serial"],
        prompt=prompt,
        video_dir=None if args.no_video else log_dir,
        video_fps=args.hz,
    )

    # -- pre-flight
    state = env.current_state()
    if state.shape != (ACTION_DIM,):
        sys.exit(f"[preflight] bad joint state shape: {state.shape}")
    if np.any(np.abs(state) > JOINT_LIMIT_RAD):
        sys.exit(f"[preflight] joint outside +/-{JOINT_LIMIT_RAD} rad: {state}")
    obs = env.get_obs()
    for k in ("observation/top_image", "observation/left_image", "observation/right_image"):
        img = obs[k]
        if img.shape != (480, 640, 3) or img.dtype != np.uint8:
            sys.exit(f"[preflight] bad image {k}: shape={img.shape} dtype={img.dtype}")
        # Sanity log only — useful to eyeball BGR-vs-RGB on first run
        print(f"[preflight] {k}: shape={img.shape} dtype={img.dtype} channel_means R/G/B={img[...,0].mean():.1f}/{img[...,1].mean():.1f}/{img[...,2].mean():.1f}")
    if obs["observation/state"].dtype != np.float32:
        sys.exit(f"[preflight] state dtype {obs['observation/state'].dtype}, expected float32")
    print(f"[preflight] state={np.array2string(state, precision=3, suppress_small=True)}")

    # -- server
    print(f"[init] connecting to inference server...")
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    server_metadata = policy.get_server_metadata()
    print(f"[init] server metadata: {server_metadata}")

    # -- first inference (no motion) for sanity
    print("[init] first inference (no motion)...")
    obs = env.get_obs()
    t0 = time.monotonic()
    out = policy.infer(obs)
    infer_ms = (time.monotonic() - t0) * 1000
    chunk = np.asarray(out["actions"]).astype(np.float32)
    if chunk.shape != (ACTION_HORIZON, ACTION_DIM):
        sys.exit(f"[preflight] bad chunk shape: {chunk.shape}, expected ({ACTION_HORIZON}, {ACTION_DIM})")
    print(f"[preflight] first chunk shape={chunk.shape} dtype={chunk.dtype} infer_ms={infer_ms:.1f}")
    print(f"[preflight] chunk[0]={np.array2string(chunk[0], precision=3, suppress_small=True)}")

    diff = np.abs(chunk[0] - obs["observation/state"])
    if diff.max() > FIRST_CHUNK_MAX_JUMP:
        sys.exit(
            f"[preflight] ABORT: first commanded pose differs from current pose by "
            f"{diff.max():.3f} rad on joint {int(diff.argmax())} "
            f"(limit {FIRST_CHUNK_MAX_JUMP} rad). "
            f"Check checkpoint / prompt / arm reset."
        )

    if args.dry_run:
        if not args.no_save_frames:
            try:
                import imageio.v3 as iio
                for cam_key, name in (
                    ("observation/top_image", "top"),
                    ("observation/left_image", "left"),
                    ("observation/right_image", "right"),
                ):
                    iio.imwrite(log_dir / f"latest_{name}.jpg", obs[cam_key], quality=85)
                print(f"[dry-run] saved camera preview JPGs to {log_dir}/latest_*.jpg")
            except Exception as e:
                print(f"[dry-run] could not save preview JPGs: {e}", file=sys.stderr)
        print("[dry-run] skipping motion. Exit.")
        env.close()
        return

    # -- arm setup
    stop = StopFlag()
    logger = RolloutLogger(
        log_dir,
        save_frames=not args.no_save_frames,
    )
    logger.event(f"prompt={prompt!r} ckpt_tag={args.ckpt_tag!r}")
    logger.event(f"server_metadata={server_metadata}")
    logger.event(f"first chunk infer_ms={infer_ms:.1f}")

    print(f"[move] smooth-moving to chunk[0] over {args.start_steps} steps...")
    smooth_move(env, chunk[0], steps=args.start_steps, dt=1.0 / args.hz)
    print("[move] at start. beginning rollout.")

    # -- main loop (open-loop full-chunk)
    dt_target = 1.0 / args.hz
    step_count = 0
    chunk_idx = 0
    last_action = chunk[0].copy()
    t_run_start = time.monotonic()
    stop_reason: Optional[str] = None

    logger.record_chunk(chunk_idx, obs, chunk, infer_ms)

    try:
        while True:
            for k in range(args.chunk_len):
                t_step = time.monotonic()
                action = chunk[k].astype(np.float32)
                env.command(action)
                last_action = action
                logger.record_step(step_count, env.current_state(), action, chunk_idx, k)
                step_count += 1

                if step_count >= args.max_steps:
                    stop_reason = "max_steps"
                    break
                if (time.monotonic() - t_run_start) >= args.max_time_s:
                    stop_reason = "timeout"
                    break

                dt_actual = time.monotonic() - t_step
                sleep_with_poll(dt_target - dt_actual, stop)
                if stop.stopped:
                    stop_reason = stop.reason or "stop"
                    break

            if stop_reason:
                break

            # End of chunk: capture obs and re-infer (synchronous)
            obs = env.get_obs()
            try:
                t0 = time.monotonic()
                out = policy.infer(obs)
                infer_ms = (time.monotonic() - t0) * 1000
            except Exception as e:
                logger.event(f"infer failed: {e}")
                stop_reason = "server_error"
                break

            chunk = np.asarray(out["actions"]).astype(np.float32)
            if chunk.shape != (ACTION_HORIZON, ACTION_DIM):
                logger.event(f"bad chunk shape {chunk.shape}; aborting")
                stop_reason = "bad_response"
                break

            chunk_idx += 1
            logger.record_chunk(chunk_idx, obs, chunk, infer_ms)
            jump = np.abs(chunk[0] - last_action)
            logger.event(
                f"chunk{chunk_idx} infer_ms={infer_ms:.1f} "
                f"boundary_max_jump={jump.max():.4f} on joint {int(jump.argmax())}"
            )
    finally:
        # -- safe shutdown: hold last measured pose for ~0.5s
        stop_reason = stop_reason or "loop_exit"
        try:
            hold = env.current_state().copy()
            print(f"[stop] reason={stop_reason} steps={step_count} chunks={chunk_idx + 1}")
            print(f"[stop] holding last pose for 0.5s")
            t_end = time.monotonic() + 0.5
            while time.monotonic() < t_end:
                env.command(hold)
                time.sleep(dt_target)
        except Exception as e:
            print(f"[stop] hold failed: {e}", file=sys.stderr)
        try:
            stop.close()
        except Exception:
            pass
        try:
            env.close()  # stops camera grabber threads + closes MP4 writers
        except Exception as e:
            print(f"[stop] env close failed: {e}", file=sys.stderr)
        cli_args_dict = {k: getattr(args, k) for k in vars(args)}
        logger.flush(
            cli_args=cli_args_dict,
            server_metadata=server_metadata,
            prompt=prompt,
            stop_reason=stop_reason,
        )
        print(f"[done] log_dir={log_dir}")


if __name__ == "__main__":
    main()
