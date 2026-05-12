"""Pi 0.5 real-robot client for the long-horizon plate-memory task (YAM single-arm).

Runs on the robot computer. Reads YAM joint state + 2 RealSense cameras (top + left
wrist) + a StageManager-driven memory channel, sends observations to a remote
`serve_policy.py` over WebSocket, receives 50-step action chunks, executes the first
N (default 10) on the arm, then re-infers.

The task has 8 operator-driven stages:
    0 observe     -> 1 mix      -> 2 delay
    -> 3 query1   -> 4 query2   -> 5 query3   -> 6 query4
    -> 7 return_home

Operator presses `n` (in the pygame window) to advance stages. On stage 0 -> 1
the live top frame is captured as the memory keyframe used for stages 3..6.
Past stage 7, `n` triggers a graceful shutdown.

Usage:
    cd /home/david/ke/memory_project
    source /home/david/openpi/.venv/bin/activate

    # On the workstation (separate process):
    #   uv run scripts/serve_policy.py policy:checkpoint \
    #       --policy.config=pi05_long_task_mem_lora --policy.dir=...

    # On the robot computer:
    python -m eval.real.run_long_task --host <server_tailscale_ip>
    python -m eval.real.run_long_task --host 100.x.y.z --dry-run
    python -m eval.real.run_long_task --host 100.x.y.z --hz 15 --max-joint-delta 0.2

Stop:
    - 'q' in the pygame window (graceful)
    - 'n' to advance stages (past stage 7 = graceful shutdown after a short window)
    - Ctrl-C — graceful (twice = hard kill)
    - --max-steps / --max-time-s timeouts

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

_THIS_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from eval.real.logger import RolloutLogger  # noqa: E402
from eval.real.long_task_env import LongTaskPolicyEnv  # noqa: E402
from eval.real.stage_machine import QueryPlan, StageManager  # noqa: E402

ACTION_HORIZON = 50
ACTION_DIM = 7
JOINT_LIMIT_RAD = 3.0
FIRST_CHUNK_MAX_JUMP = 0.3       # rad; pre-flight abort threshold
HOLD_AFTER_N_S = 0.3             # hold last pose this long after each `n` advance
SHUTDOWN_AFTER_FINAL_ADVANCE_S = 8.0  # graceful stop window after stage-7 advance


# ---------------------------------------------------------------------------
# Stop / advance signal
# ---------------------------------------------------------------------------


class LongTaskFlag:
    """Pygame-backed q (stop) + n (advance) signal, with SIGINT escape.

    Subsumes the CUP `StopFlag` + `_KeyWindow` pair — pygame events are consumed
    on read, so q and n must be drained in the same poll.
    """

    def __init__(self) -> None:
        self._stop = False
        self._reason: Optional[str] = None
        self._advance = False
        self._sigint_count = 0

        signal.signal(signal.SIGINT, self._on_sigint)

        self._pygame = None
        self._screen = None
        try:
            import pygame
            self._pygame = pygame
            pygame.init()
            self._screen = pygame.display.set_mode((520, 100))
            pygame.display.set_caption("Pi 0.5 long-task — q=stop  n=advance stage")
            font = pygame.font.SysFont(None, 22)
            self._screen.fill((20, 20, 20))
            for i, msg in enumerate((
                "Click here, then:",
                "  press 'n' to advance to next stage",
                "  press 'q' to stop the run",
            )):
                self._screen.blit(font.render(msg, True, (240, 240, 240)), (10, 10 + i * 24))
            pygame.display.flip()
            print("[stop] pygame window open — click it then press 'n' to advance, 'q' to stop")
        except Exception as e:
            print(f"[stop] pygame window unavailable ({e}); use Ctrl-C to stop")
            print("[stop] WARNING: cannot detect 'n' presses without pygame — exiting now is recommended")
        print("[stop] Ctrl-C also stops (twice = hard kill)")

    def _on_sigint(self, *_) -> None:
        self._sigint_count += 1
        if self._sigint_count >= 2:
            print("\n[stop] second Ctrl-C — force exit", flush=True)
            os._exit(130)
        self._stop = True
        self._reason = "sigint"
        print("\n[stop] SIGINT received — finishing chunk and holding pose. Ctrl-C again to force.", flush=True)

    def poll(self) -> None:
        """Drain pygame events once. q -> stop, n -> advance."""
        if self._pygame is None:
            return
        self._pygame.event.pump()
        for event in self._pygame.event.get():
            if event.type == self._pygame.KEYDOWN:
                if event.key == self._pygame.K_q:
                    self._stop = True
                    self._reason = self._reason or "user-pygame"
                elif event.key == self._pygame.K_n:
                    self._advance = True

    @property
    def stopped(self) -> bool:
        return self._stop

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    @property
    def advance_pressed(self) -> bool:
        return self._advance

    def consume_advance(self) -> None:
        self._advance = False

    def request_stop(self, reason: str) -> None:
        self._stop = True
        self._reason = self._reason or reason

    def close(self) -> None:
        if self._pygame is not None:
            try:
                self._pygame.quit()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def smooth_move(env: LongTaskPolicyEnv, target: np.ndarray, steps: int, dt: float) -> None:
    """Linearly interpolate from current state to `target`. Mirrors gello replay_demo.smooth_move."""
    current = env.current_state()
    target = np.asarray(target, dtype=np.float32)
    for jnt in np.linspace(current, target, steps):
        env.command(jnt.astype(np.float32))
        time.sleep(dt)


def sleep_with_poll(seconds: float, flag: LongTaskFlag) -> None:
    """Sleep up to `seconds`, polling the stop/advance flag every 5 ms."""
    if seconds <= 0:
        return
    end = time.monotonic() + seconds
    while True:
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        flag.poll()
        if flag.stopped or flag.advance_pressed:
            return
        time.sleep(min(remaining, 0.005))


def clip_joint_delta(
    target: np.ndarray, prev: np.ndarray, max_d: float, logger: Optional[RolloutLogger] = None,
) -> np.ndarray:
    """Cap per-step joint movement to <= max_d rad (linear scale, preserves direction)."""
    delta = np.asarray(target, dtype=np.float32) - np.asarray(prev, dtype=np.float32)
    peak = float(np.max(np.abs(delta)))
    if peak > max_d:
        scale = max_d / peak
        clipped = (prev + delta * scale).astype(np.float32)
        if logger is not None:
            logger.event(
                f"joint-delta clip: peak={peak:.4f} > {max_d:.4f} on joint {int(np.argmax(np.abs(delta)))}; scale={scale:.3f}"
            )
        return clipped
    return np.asarray(target, dtype=np.float32)


def hold_pose(env: LongTaskPolicyEnv, duration_s: float, dt: float) -> None:
    held = env.current_state().copy()
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        env.command(held)
        time.sleep(dt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", required=True, help="Inference server host (e.g. 100.x.y.z Tailscale)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", type=Path, default=_THIS_DIR / "configs" / "yam_long_task_eval.yaml")
    p.add_argument("--query-mode", type=str, default="fixed", choices=("fixed", "random", "manual"))
    p.add_argument("--seed", type=int, default=0, help="Used by --query-mode random")
    p.add_argument("--ckpt-tag", type=str, default="", help="Free-form tag written to metadata.json")
    p.add_argument("--max-steps", type=int, default=5400, help="Control-step cap (8 stages x ~360s x 15 Hz)")
    p.add_argument("--max-time-s", type=float, default=480.0, help="Wall-clock cap (~8 min)")
    p.add_argument("--hz", type=float, default=15.0, help="Control rate (slow-mode default)")
    p.add_argument("--chunk-len", type=int, default=25, help=f"How many of the {ACTION_HORIZON} returned actions to execute per inference")
    p.add_argument("--max-joint-delta", type=float, default=0.2, help="Per-step joint movement cap (rad)")
    p.add_argument("--start-steps", type=int, default=60, help="Smooth-move steps from current pose to chunk[0]")
    p.add_argument("--log-dir", type=Path, default=None, help="Default: eval/real/runs/<timestamp>_long")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-save-frames", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Pre-flight + 1 inference, no motion")
    return p.parse_args()


def _do_advance(
    env: LongTaskPolicyEnv,
    stage_mgr: StageManager,
    logger: RolloutLogger,
    flag: LongTaskFlag,
    dt_target: float,
) -> bool:
    """Handle an `n` press. Returns True if we just advanced past the final stage."""
    flag.consume_advance()
    cont = stage_mgr.advance(env.current_top_frame())
    if not cont:
        # Already past stage 7 — caller will have caught is_terminal beforehand.
        return True

    if stage_mgr.is_terminal:
        # We just left stage 7 (return_home) — schedule shutdown, no instruction update needed.
        logger.event("advanced past stage 7 — entering shutdown window")
        return True

    logger.record_stage_transition(
        stage_mgr.stage_id,
        stage_mgr.stage_name,
        stage_mgr.instruction,
        memory_image=stage_mgr.memory_image if stage_mgr.has_memory else None,
    )
    print(
        f"[stage] {stage_mgr.stage_id} ({stage_mgr.stage_name}): "
        f"{stage_mgr.instruction!r}  memory={'ON' if stage_mgr.has_memory else 'OFF'}"
    )
    # Hold last pose briefly so the operator-induced stage break is smooth on the arm.
    hold_pose(env, HOLD_AFTER_N_S, dt_target)
    return False


def main() -> None:
    args = parse_args()
    if not 1 <= args.chunk_len <= ACTION_HORIZON:
        sys.exit(f"--chunk-len must be in [1, {ACTION_HORIZON}], got {args.chunk_len}")
    if args.max_joint_delta <= 0:
        sys.exit(f"--max-joint-delta must be > 0, got {args.max_joint_delta}")

    cfg = yaml.safe_load(args.config.read_text())
    log_dir = args.log_dir or (_THIS_DIR / "runs" / (_dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_long"))
    log_dir = Path(log_dir)
    print(f"[init] config       : {args.config}")
    print(f"[init] query mode   : {args.query_mode} (seed={args.seed})")
    print(f"[init] server       : ws://{args.host}:{args.port}")
    print(f"[init] log_dir      : {log_dir}")
    print(f"[init] hz={args.hz}  chunk_len={args.chunk_len}  max_joint_delta={args.max_joint_delta}")

    plan = QueryPlan(mode=args.query_mode, seed=args.seed)
    plan.print_setup()
    stage_mgr = StageManager(plan)

    log_dir.mkdir(parents=True, exist_ok=True)
    print("[init] connecting to YAM + cameras (top + left)...")
    env = LongTaskPolicyEnv(
        channel=cfg["robot"]["channel"],
        top_serial=cfg["cameras"]["top_serial"],
        left_serial=cfg["cameras"]["left_serial"],
        stage_manager=stage_mgr,
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
    for k in ("observation/top_image", "observation/left_image", "observation/memory_image"):
        img = obs[k]
        if img.shape != (480, 640, 3) or img.dtype != np.uint8:
            sys.exit(f"[preflight] bad image {k}: shape={img.shape} dtype={img.dtype}")
        print(f"[preflight] {k}: shape={img.shape} dtype={img.dtype} channel_means R/G/B="
              f"{img[...,0].mean():.1f}/{img[...,1].mean():.1f}/{img[...,2].mean():.1f}")
    if obs["observation/state"].dtype != np.float32:
        sys.exit(f"[preflight] state dtype {obs['observation/state'].dtype}, expected float32")
    if obs["observation/has_memory"].shape != (1,) or obs["observation/has_memory"].dtype != np.float32:
        sys.exit(f"[preflight] bad has_memory: shape={obs['observation/has_memory'].shape} dtype={obs['observation/has_memory'].dtype}")
    if stage_mgr.stage_id != 0:
        sys.exit(f"[preflight] expected to start in stage 0 (observe), got {stage_mgr.stage_id}")
    if obs["observation/has_memory"][0] != 0.0:
        sys.exit("[preflight] observe stage should have has_memory=0.0")
    if obs["observation/memory_image"].sum() != 0:
        sys.exit("[preflight] observe stage memory_image should be all zeros")
    print(f"[preflight] state={np.array2string(state, precision=3, suppress_small=True)}")
    print(f"[preflight] stage={stage_mgr.stage_id} ({stage_mgr.stage_name})  prompt={obs['prompt']!r}")

    # -- server
    print(f"[init] connecting to inference server...")
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    server_metadata = policy.get_server_metadata()
    print(f"[init] server metadata: {server_metadata}")

    # -- first inference (no motion)
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
            f"{diff.max():.3f} rad on joint {int(diff.argmax())} (limit {FIRST_CHUNK_MAX_JUMP} rad). "
            f"Check checkpoint / arm reset / stage."
        )

    if args.dry_run:
        if not args.no_save_frames:
            try:
                import imageio.v3 as iio
                for cam_key, name in (
                    ("observation/top_image", "top"),
                    ("observation/left_image", "left"),
                    ("observation/memory_image", "memory"),
                ):
                    iio.imwrite(log_dir / f"latest_{name}.jpg", obs[cam_key], quality=85)
                print(f"[dry-run] saved camera preview JPGs to {log_dir}/latest_*.jpg")
            except Exception as e:
                print(f"[dry-run] could not save preview JPGs: {e}", file=sys.stderr)
        print("[dry-run] skipping motion. Exit.")
        env.close()
        return

    # -- arm setup
    flag = LongTaskFlag()
    logger = RolloutLogger(log_dir, save_frames=not args.no_save_frames)
    logger.set_query_plan(plan.to_dict())
    logger.event(f"query_plan={plan.to_dict()}")
    logger.event(f"ckpt_tag={args.ckpt_tag!r}")
    logger.event(f"server_metadata={server_metadata}")
    logger.event(f"first chunk infer_ms={infer_ms:.1f}")
    # Record the initial stage so metadata.json has a complete stage timeline.
    logger.record_stage_transition(
        stage_mgr.stage_id, stage_mgr.stage_name, stage_mgr.instruction, memory_image=None,
    )

    print(f"[stage] {stage_mgr.stage_id} ({stage_mgr.stage_name}): {stage_mgr.instruction!r}")
    print(f"[move] smooth-moving to chunk[0] over {args.start_steps} steps...")
    smooth_move(env, chunk[0], steps=args.start_steps, dt=1.0 / args.hz)
    print("[move] at start. beginning rollout.")

    # -- main loop
    dt_target = 1.0 / args.hz
    step_count = 0
    chunk_idx = 0
    last_action = chunk[0].copy()
    t_run_start = time.monotonic()
    final_advance_t: Optional[float] = None
    stop_reason: Optional[str] = None

    logger.record_chunk(chunk_idx, obs, chunk, infer_ms)

    try:
        while True:
            # 1) Execute up to chunk_len control steps from the current chunk.
            advanced = False
            for k in range(args.chunk_len):
                t_step = time.monotonic()
                action = clip_joint_delta(chunk[k], last_action, args.max_joint_delta, logger)
                env.command(action)
                last_action = action
                logger.record_step(
                    step_count,
                    env.current_state(),
                    action,
                    chunk_idx,
                    k,
                    stage_id=stage_mgr.stage_id,
                )
                step_count += 1

                if step_count >= args.max_steps:
                    stop_reason = "max_steps"
                    break
                if (time.monotonic() - t_run_start) >= args.max_time_s:
                    stop_reason = "timeout"
                    break

                dt_actual = time.monotonic() - t_step
                sleep_with_poll(dt_target - dt_actual, flag)
                if flag.stopped:
                    stop_reason = flag.reason or "stop"
                    break
                if flag.advance_pressed:
                    advanced = True
                    break

            if stop_reason:
                break

            if advanced:
                final = _do_advance(env, stage_mgr, logger, flag, dt_target)
                if final and final_advance_t is None:
                    final_advance_t = time.monotonic()

            if final_advance_t is not None and (time.monotonic() - final_advance_t) >= SHUTDOWN_AFTER_FINAL_ADVANCE_S:
                stop_reason = "stage7-shutdown"
                break

            # 2) Re-infer with the (possibly updated) stage prompt + memory.
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
                f"chunk{chunk_idx} stage={stage_mgr.stage_id} infer_ms={infer_ms:.1f} "
                f"boundary_max_jump={jump.max():.4f} on joint {int(jump.argmax())}"
            )
    finally:
        # -- safe shutdown: hold last measured pose for ~0.5s
        stop_reason = stop_reason or "loop_exit"
        try:
            print(f"[stop] reason={stop_reason} steps={step_count} chunks={chunk_idx + 1} "
                  f"stage={stage_mgr.stage_id} ({stage_mgr.stage_name})")
            print(f"[stop] holding last pose for 0.5s")
            t_end = time.monotonic() + 0.5
            while time.monotonic() < t_end:
                env.command(env.current_state())
                time.sleep(dt_target)
        except Exception as e:
            print(f"[stop] hold failed: {e}", file=sys.stderr)
        try:
            flag.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception as e:
            print(f"[stop] env close failed: {e}", file=sys.stderr)
        cli_args_dict = {k: getattr(args, k) for k in vars(args)}
        logger.flush(
            cli_args=cli_args_dict,
            server_metadata=server_metadata,
            prompt="(per-stage; see stage_events)",
            stop_reason=stop_reason,
        )
        print(f"[done] log_dir={log_dir}")


if __name__ == "__main__":
    main()
