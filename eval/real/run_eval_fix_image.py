"""Pi 0.5 real-robot eval with a fixed-image baseline rewriter (no retriever).

Variant of run_eval.py / run_eval_retriever.py with a much simpler keyframe
policy:
  - The first top-camera frame captured at startup (during preflight) is held
    as `first_image` for the whole rollout.
  - Stage starts at 0; press 'n' (focus the pygame window) to advance, clamps
    at 7. On every advance, the prompt update depends on the destination:
        stage 1, 2  -> no rewrite, original prompt stays
        stage 3-6   -> rewrite(first_image, original_prompt)   [query stages]
        stage 7     -> rewrite(current_image, original_prompt) [return_home]
  - No SigLIP, no retriever, no MiniLM — only the Gemini rewriter.

Usage:
    python eval/real/run_eval_fix_image.py --host <gpu_box_ip> \
        --prompt "Place the object originally on the pink plate onto the pink plate."

Stop signals match run_eval.py: 'q' in the pygame window, Ctrl-C, or
--max-steps / --max-time-s timeouts.
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
from eval.real.robot_env import YamPolicyEnv  # noqa: E402
from rewriter.rewriter import rewrite as _rewrite  # noqa: E402

DEFAULT_PROMPT = "Place the object originally on the pink plate onto the pink plate."
ACTION_HORIZON = 10
ACTION_DIM = 7
JOINT_LIMIT_RAD = 3.0
FIRST_CHUNK_MAX_JUMP = 0.3
MAX_STAGE_ID = 7
QUERY_STAGES = {3, 4, 5, 6}


# ---------------------------------------------------------------------------
# Stop + stage signal helpers
# ---------------------------------------------------------------------------


class StopFlag:
    """Stop coordinator: pygame 'q' / 'n' window + SIGINT escape."""

    def __init__(self) -> None:
        self._stop = False
        self._reason: Optional[str] = None
        self._sigint_count = 0
        signal.signal(signal.SIGINT, self._on_sigint)

        self._kb = None
        try:
            self._kb = _KeyWindow()
            print("[stop] pygame window open — click it, then 'q'=stop, 'n'=next stage")
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
        print(
            "\n[stop] SIGINT — finishing chunk and holding pose. Ctrl-C again to force.",
            flush=True,
        )

    def poll(self) -> None:
        if self._kb is not None:
            self._kb.pump()
            if self._kb.q_pressed:
                self._stop = True
                self._reason = "user-pygame"

    def consume_n(self) -> bool:
        if self._kb is None:
            return False
        self._kb.pump()
        return self._kb.consume_n()

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
    """Tiny pygame window that captures 'q' and 'n'."""

    def __init__(self) -> None:
        import pygame

        self._pygame = pygame
        pygame.init()
        self._screen = pygame.display.set_mode((480, 80))
        pygame.display.set_caption("Pi 0.5 eval (fix-image) — 'q'=stop  'n'=next stage")
        font = pygame.font.SysFont(None, 22)
        self._screen.fill((20, 20, 20))
        msg = font.render("Click here. 'q'=stop, 'n'=next stage.", True, (240, 240, 240))
        self._screen.blit(msg, (10, 30))
        pygame.display.flip()

        self.q_pressed = False
        self._n_count = 0

    def pump(self) -> None:
        self._pygame.event.pump()
        for event in self._pygame.event.get():
            if event.type != self._pygame.KEYDOWN:
                continue
            if event.key == self._pygame.K_q:
                self.q_pressed = True
            elif event.key == self._pygame.K_n:
                self._n_count += 1

    def consume_n(self) -> bool:
        if self._n_count > 0:
            self._n_count -= 1
            return True
        return False

    def close(self) -> None:
        self._pygame.quit()


# ---------------------------------------------------------------------------
# Stage controller
# ---------------------------------------------------------------------------


class StageController:
    def __init__(self) -> None:
        self.current = 0

    def advance(self) -> int:
        if self.current < MAX_STAGE_ID:
            self.current += 1
        return self.current


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def smooth_move(env: YamPolicyEnv, target: np.ndarray, steps: int, dt: float) -> None:
    current = env.current_state()
    target = np.asarray(target, dtype=np.float32)
    for jnt in np.linspace(current, target, steps):
        env.command(jnt.astype(np.float32))
        time.sleep(dt)


def sleep_with_poll(seconds: float, stop: StopFlag) -> None:
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


def _pick_keyframe(
    new_stage: int, first_image: np.ndarray, current_image: np.ndarray
) -> Optional[tuple[np.ndarray, str]]:
    """Return (image, source_tag) for the rewriter, or None if no rewrite this stage."""
    if new_stage in QUERY_STAGES:
        return first_image, "first"
    if new_stage == MAX_STAGE_ID:  # return_home
        return current_image, "current"
    return None  # stages 1, 2 -> no rewrite


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", type=Path, default=_THIS_DIR / "configs" / "yam_eval.yaml")
    p.add_argument("--prompt", type=str, default=None)
    p.add_argument("--ckpt-tag", type=str, default="")
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--max-time-s", type=float, default=60.0)
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--chunk-len", type=int, default=ACTION_HORIZON)
    p.add_argument("--start-steps", type=int, default=60)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-save-frames", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--rewriter-model", type=str, default=None)
    return p.parse_args()


def _cfg_get(cfg: dict, *keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def main() -> None:
    args = parse_args()
    if not 1 <= args.chunk_len <= ACTION_HORIZON:
        sys.exit(f"--chunk-len must be in [1, {ACTION_HORIZON}], got {args.chunk_len}")

    cfg = yaml.safe_load(args.config.read_text())
    prompt = args.prompt or cfg.get("prompt", DEFAULT_PROMPT)
    rewriter_model = args.rewriter_model or _cfg_get(
        cfg, "retriever", "rewriter_model", default="gemini-2.5-flash"
    )

    log_dir = args.log_dir or (_THIS_DIR / "runs" / _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    log_dir = Path(log_dir)
    print(f"[init] config       : {args.config}")
    print(f"[init] prompt       : {prompt!r}")
    print(f"[init] server       : ws://{args.host}:{args.port}")
    print(f"[init] log_dir      : {log_dir}")
    print(f"[init] rewriter     : {rewriter_model}")

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
    state_arr = env.current_state()
    if state_arr.shape != (ACTION_DIM,):
        sys.exit(f"[preflight] bad joint state shape: {state_arr.shape}")
    if np.any(np.abs(state_arr) > JOINT_LIMIT_RAD):
        sys.exit(f"[preflight] joint outside +/-{JOINT_LIMIT_RAD} rad: {state_arr}")
    obs = env.get_obs()
    for k in ("observation/top_image", "observation/left_image", "observation/right_image"):
        img = obs[k]
        if img.shape != (480, 640, 3) or img.dtype != np.uint8:
            sys.exit(f"[preflight] bad image {k}: shape={img.shape} dtype={img.dtype}")
        print(
            f"[preflight] {k}: shape={img.shape} dtype={img.dtype} "
            f"channel_means R/G/B={img[...,0].mean():.1f}/{img[...,1].mean():.1f}/{img[...,2].mean():.1f}"
        )
    if obs["observation/state"].dtype != np.float32:
        sys.exit(f"[preflight] state dtype {obs['observation/state'].dtype}, expected float32")
    print(f"[preflight] state={np.array2string(state_arr, precision=3, suppress_small=True)}")

    # -- snapshot the FIRST top-camera frame here, before any motion.
    first_image = obs["observation/top_image"].copy()
    print(f"[init] first_image captured: shape={first_image.shape}")

    # -- server
    print("[init] connecting to inference server...")
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
            f"(limit {FIRST_CHUNK_MAX_JUMP} rad)."
        )

    if args.dry_run:
        # Exercise the rewriter once with the captured first frame.
        print("[dry-run] testing rewriter with first_image...")
        try:
            test_prompt = _rewrite(first_image, prompt, model=rewriter_model)
            print(f"[dry-run] rewritten={test_prompt!r}")
        except Exception as e:
            print(f"[dry-run] rewrite failed: {e}", file=sys.stderr)

        if not args.no_save_frames:
            try:
                import imageio.v3 as iio
                for cam_key, name in (
                    ("observation/top_image", "top"),
                    ("observation/left_image", "left"),
                    ("observation/right_image", "right"),
                ):
                    iio.imwrite(log_dir / f"latest_{name}.jpg", obs[cam_key], quality=85)
                iio.imwrite(log_dir / "first_image.jpg", first_image, quality=85)
                print(f"[dry-run] saved camera preview JPGs to {log_dir}/latest_*.jpg")
            except Exception as e:
                print(f"[dry-run] could not save preview JPGs: {e}", file=sys.stderr)
        print("[dry-run] skipping motion. Exit.")
        env.close()
        return

    # -- arm setup
    stop = StopFlag()
    stage = StageController()
    logger = RolloutLogger(log_dir, save_frames=not args.no_save_frames)
    logger.event(f"prompt={prompt!r} ckpt_tag={args.ckpt_tag!r}")
    logger.event(f"server_metadata={server_metadata}")
    logger.event(f"first chunk infer_ms={infer_ms:.1f}")
    logger.event(f"variant=fix_image rewriter={rewriter_model}")

    # Persist the captured first_image alongside the run.
    try:
        import imageio.v3 as iio
        iio.imwrite(log_dir / "first_image.jpg", first_image, quality=85)
    except Exception as e:
        logger.event(f"[init] could not save first_image.jpg: {e}")

    print(f"[move] smooth-moving to chunk[0] over {args.start_steps} steps...")
    smooth_move(env, chunk[0], steps=args.start_steps, dt=1.0 / args.hz)
    print("[move] at start. beginning rollout.")

    # -- main loop
    dt_target = 1.0 / args.hz
    step_count = 0
    chunk_idx = 0
    last_action = chunk[0].copy()
    t_run_start = time.monotonic()
    stop_reason: Optional[str] = None

    logger.record_chunk(chunk_idx, obs, chunk, infer_ms)
    logger.event(f"[stage] enter stage=0 (observe)  prompt={prompt!r}")

    try:
        while True:
            for k in range(args.chunk_len):
                t_step = time.monotonic()
                action = chunk[k].astype(np.float32)
                env.command(action)
                last_action = action
                logger.record_step(step_count, env.current_state(), action, chunk_idx, k)
                step_count += 1

                # 'n' = advance stage and (if applicable) rewrite the prompt.
                if stop.consume_n():
                    prev = stage.current
                    new_stage = stage.advance()
                    logger.event(f"[stage] {prev} -> {new_stage}")
                    cur_image = env.get_obs()["observation/top_image"]
                    pick = _pick_keyframe(new_stage, first_image, cur_image)
                    if pick is None:
                        logger.event(
                            f"[stage] no rewrite for stage={new_stage}; "
                            f"prompt unchanged ({prompt!r})"
                        )
                    else:
                        image, source = pick
                        try:
                            new_prompt = _rewrite(image, prompt, model=rewriter_model)
                            env.set_prompt(new_prompt)
                            logger.event(
                                f"[stage] prompt updated  source={source}  "
                                f"prompt={new_prompt!r}"
                            )
                            print(f"[stage] {prev}->{new_stage} src={source} prompt={new_prompt!r}")
                        except Exception as e:
                            logger.event(f"[rewrite] failed: {e}; keeping prior prompt")
                            print(f"[rewrite] failed: {e}", file=sys.stderr)

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
                f"chunk{chunk_idx} stage={stage.current} infer_ms={infer_ms:.1f} "
                f"boundary_max_jump={jump.max():.4f} on joint {int(jump.argmax())}"
            )
    finally:
        stop_reason = stop_reason or "loop_exit"
        try:
            hold = env.current_state().copy()
            print(f"[stop] reason={stop_reason} steps={step_count} chunks={chunk_idx + 1} "
                  f"final_stage={stage.current}")
            print("[stop] holding last pose for 0.5s")
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
            env.close()
        except Exception as e:
            print(f"[stop] env close failed: {e}", file=sys.stderr)
        cli_args_dict = {k: str(getattr(args, k)) for k in vars(args)}
        logger.flush(
            cli_args=cli_args_dict,
            server_metadata=server_metadata,
            prompt=prompt,
            stop_reason=stop_reason,
        )
        print(f"[done] log_dir={log_dir}")


if __name__ == "__main__":
    main()
