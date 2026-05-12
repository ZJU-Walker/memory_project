"""Pi 0.5 long-task client with live retriever + Gemini rewriter on query stages.

Variant of `run_long_task.py`. Same hardware/protocol/stage flow; the difference
is what fills `observation/memory_image` and `prompt` on stages 3-6 (query):

  - During the rollout, the top camera is sampled at `--memory-sample-hz`
    (default 1.0) ACROSS ALL STAGES and each frame is encoded with PaliGemma
    SigLIP into an in-memory bank. (Causality is automatic — only past frames
    can have been appended by the time we query.)
  - When the operator presses `n` to enter a query stage (3, 4, 5, or 6):
        1. Encode the current top frame (cur_global, cur_patches).
        2. Compute MiniLM text embedding of the StageManager's relational
           instruction for the new stage ("Place the object originally on
           the X plate onto the Y plate.").
        3. Run the trained retriever (`retriever/runs/.../best.pt`) to pick
           the top-1 keyframe from the bank.
        4. Run Gemini rewriter on (top1_image, relational_instruction) →
           resolved instruction (e.g. "put red chili on the pink plate.").
        5. Override BOTH `observation/memory_image` (top1 image) AND `prompt`
           (rewritten string) on the StageManager for the duration of the
           query stage.
  - Non-query stages (1, 2, 7) keep StageManager defaults.

Usage:
    cd /home/kewalk/memory_project
    source openpi/.venv/bin/activate

    # Workstation:
    #   uv run scripts/serve_policy.py policy:checkpoint \
    #       --policy.config=pi05_long_task_mem_lora --policy.dir=...

    # Robot computer:
    python -m eval.real.run_long_task_retriever --host <server_tailscale_ip>
    python -m eval.real.run_long_task_retriever --host 100.x.y.z --dry-run

Stop / advance keys mirror `run_long_task.py` — `n` advances stages, `q` stops.
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
import torch
import yaml

from openpi_client.websocket_client_policy import WebsocketClientPolicy

_THIS_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

from eval.real.logger import RolloutLogger  # noqa: E402
from eval.real.long_task_env import LongTaskPolicyEnv  # noqa: E402
from eval.real.retrieval_loop import (  # noqa: E402
    LiveMemoryBank,
    MiniLMTextEncoder,
    SiglipEncoder,
    retrieve_and_rewrite,
)
from eval.real.retriever_stage_machine import RetrieverStageManager  # noqa: E402
from eval.real.stage_machine import QUERY_STAGES, QueryPlan  # noqa: E402
from retriever.model import Retriever  # noqa: E402

ACTION_HORIZON = 50
ACTION_DIM = 7
JOINT_LIMIT_RAD = 3.0
FIRST_CHUNK_MAX_JUMP = 0.3
HOLD_AFTER_N_S = 0.3
SHUTDOWN_AFTER_FINAL_ADVANCE_S = 8.0


# ---------------------------------------------------------------------------
# Stop / advance signal — mirrors LongTaskFlag in run_long_task.py
# ---------------------------------------------------------------------------


class LongTaskFlag:
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
            pygame.display.set_caption("Pi 0.5 long-task (retriever) — q=stop  n=advance stage")
            font = pygame.font.SysFont(None, 22)
            self._screen.fill((20, 20, 20))
            for i, msg in enumerate((
                "Click here, then:",
                "  press 'n' to advance to next stage",
                "  press 'q' to stop the run",
            )):
                self._screen.blit(font.render(msg, True, (240, 240, 240)), (10, 10 + i * 24))
            pygame.display.flip()
            print("[stop] pygame window open — click it then 'n'=advance, 'q'=stop")
        except Exception as e:
            print(f"[stop] pygame window unavailable ({e}); use Ctrl-C to stop")
            print("[stop] WARNING: cannot detect 'n' presses without pygame")
        print("[stop] Ctrl-C also stops (twice = hard kill)")

    def _on_sigint(self, *_) -> None:
        self._sigint_count += 1
        if self._sigint_count >= 2:
            print("\n[stop] second Ctrl-C — force exit", flush=True)
            os._exit(130)
        self._stop = True
        self._reason = "sigint"
        print(
            "\n[stop] SIGINT received — finishing chunk and holding pose. Ctrl-C again to force.",
            flush=True,
        )

    def poll(self) -> None:
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
# Helpers — copied from run_long_task.py
# ---------------------------------------------------------------------------


def smooth_move(env: LongTaskPolicyEnv, target: np.ndarray, steps: int, dt: float) -> None:
    current = env.current_state()
    target = np.asarray(target, dtype=np.float32)
    for jnt in np.linspace(current, target, steps):
        env.command(jnt.astype(np.float32))
        time.sleep(dt)


def sleep_with_poll(seconds: float, flag: LongTaskFlag) -> None:
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
    delta = np.asarray(target, dtype=np.float32) - np.asarray(prev, dtype=np.float32)
    peak = float(np.max(np.abs(delta)))
    if peak > max_d:
        scale = max_d / peak
        clipped = (prev + delta * scale).astype(np.float32)
        if logger is not None:
            logger.event(
                f"joint-delta clip: peak={peak:.4f} > {max_d:.4f} on joint "
                f"{int(np.argmax(np.abs(delta)))}; scale={scale:.3f}"
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
# Retriever-driven advance
# ---------------------------------------------------------------------------


def _do_advance(
    env: LongTaskPolicyEnv,
    stage_mgr: RetrieverStageManager,
    bank: LiveMemoryBank,
    siglip: SiglipEncoder,
    text_enc: MiniLMTextEncoder,
    retriever_model: torch.nn.Module,
    rewriter_model: str,
    retr_device: str,
    t_run_start: float,
    max_time_s: float,
    logger: RolloutLogger,
    flag: LongTaskFlag,
    dt_target: float,
) -> bool:
    """Handle an `n` press. Returns True if we just advanced past the final stage."""
    flag.consume_advance()
    cont = stage_mgr.advance(env.current_top_frame())
    if not cont:
        return True
    if stage_mgr.is_terminal:
        logger.event("advanced past stage 7 — entering shutdown window")
        return True

    new_stage = stage_mgr.stage_id
    # On query stages, replace memory_image + instruction with retriever+rewriter output.
    if new_stage in QUERY_STAGES:
        if len(bank) == 0:
            logger.event(
                f"[retrieval] stage={new_stage} skipped: empty bank; "
                f"falling back to StageManager defaults"
            )
        else:
            try:
                relational = stage_mgr.instruction  # base/default at this point (override cleared)
                cur_top = env.current_top_frame()
                cg, cp = siglip.encode_one(cur_top)
                te = text_enc.encode(relational)
                cur_t_norm = (time.monotonic() - t_run_start) / max(max_time_s, 1e-6)
                new_prompt, top1_idx = retrieve_and_rewrite(
                    retriever_model,
                    bank,
                    te,
                    cg.astype(np.float32),
                    cp.astype(np.float32),
                    cur_t_norm=cur_t_norm,
                    original_prompt=relational,
                    rewriter_model=rewriter_model,
                    device=retr_device,
                )
                top1_image = bank.raw_images[top1_idx]
                stage_mgr.set_override(top1_image, new_prompt)
                logger.event(
                    f"[retrieval] stage={new_stage}  top1_idx={top1_idx}  "
                    f"relational={relational!r}  rewritten={new_prompt!r}"
                )
                print(
                    f"[stage] {new_stage} ({stage_mgr.stage_name})  "
                    f"top1_idx={top1_idx}  prompt={new_prompt!r}"
                )
            except Exception as e:
                logger.event(f"[retrieval] stage={new_stage} failed: {e}; using defaults")
                print(f"[retrieval] stage={new_stage} failed: {e}", file=sys.stderr)

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
    hold_pose(env, HOLD_AFTER_N_S, dt_target)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", required=True, help="Inference server host (e.g. 100.x.y.z Tailscale)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--config", type=Path, default=_THIS_DIR / "configs" / "yam_long_task_eval.yaml")
    p.add_argument("--query-mode", type=str, default="fixed", choices=("fixed", "random", "manual"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ckpt-tag", type=str, default="")
    p.add_argument("--max-steps", type=int, default=5400)
    p.add_argument("--max-time-s", type=float, default=480.0)
    p.add_argument("--hz", type=float, default=15.0)
    p.add_argument("--chunk-len", type=int, default=25)
    p.add_argument("--max-joint-delta", type=float, default=0.2)
    p.add_argument("--start-steps", type=int, default=60)
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--no-save-frames", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    # Retriever / rewriter overrides (yaml-defaulted)
    p.add_argument("--retriever-ckpt", type=Path, default=None)
    p.add_argument("--openpi-config", type=str, default=None)
    p.add_argument("--openpi-ckpt-dir", type=Path, default=None)
    p.add_argument("--rewriter-model", type=str, default=None)
    p.add_argument("--memory-sample-hz", type=float, default=None)
    p.add_argument("--retriever-device", type=str, default=None)
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
    if args.max_joint_delta <= 0:
        sys.exit(f"--max-joint-delta must be > 0, got {args.max_joint_delta}")

    cfg = yaml.safe_load(args.config.read_text())
    retr_ckpt = args.retriever_ckpt or _cfg_get(cfg, "retriever", "ckpt")
    openpi_config = args.openpi_config or _cfg_get(cfg, "retriever", "openpi_config")
    openpi_ckpt = args.openpi_ckpt_dir or _cfg_get(cfg, "retriever", "openpi_ckpt_dir")
    rewriter_model = args.rewriter_model or _cfg_get(
        cfg, "retriever", "rewriter_model", default="gemini-2.5-flash"
    )
    mem_hz = args.memory_sample_hz or _cfg_get(
        cfg, "retriever", "memory_sample_hz", default=1.0
    )
    retr_device = args.retriever_device or _cfg_get(
        cfg, "retriever", "device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    if retr_ckpt is None:
        sys.exit("missing --retriever-ckpt (or retriever.ckpt in yaml)")
    retr_ckpt = Path(retr_ckpt)
    openpi_ckpt = Path(openpi_ckpt) if openpi_ckpt is not None else None

    log_dir = args.log_dir or (_THIS_DIR / "runs" / (_dt.datetime.now().strftime("%Y%m%d_%H%M%S") + "_long_retr"))
    log_dir = Path(log_dir)
    print(f"[init] config       : {args.config}")
    print(f"[init] query mode   : {args.query_mode} (seed={args.seed})")
    print(f"[init] server       : ws://{args.host}:{args.port}")
    print(f"[init] log_dir      : {log_dir}")
    print(f"[init] hz={args.hz}  chunk_len={args.chunk_len}  max_joint_delta={args.max_joint_delta}")
    print(f"[init] retr_ckpt    : {retr_ckpt}")
    print(f"[init] openpi_ckpt  : {openpi_ckpt}")
    print(f"[init] rewriter     : {rewriter_model}")
    print(f"[init] mem_hz       : {mem_hz}")
    print(f"[init] retr_device  : {retr_device}")

    plan = QueryPlan(mode=args.query_mode, seed=args.seed)
    plan.print_setup()
    stage_mgr = RetrieverStageManager(plan)

    log_dir.mkdir(parents=True, exist_ok=True)

    # -- Load retriever stack BEFORE opening the robot so a load failure
    # doesn't leave hardware in an awkward state.
    print("[init] loading PaliGemma SigLIP encoder (this can take 10-30s for JIT compile)...")
    siglip_kwargs = {}
    if openpi_config is not None:
        siglip_kwargs["config_name"] = openpi_config
    if openpi_ckpt is not None:
        siglip_kwargs["ckpt_dir"] = openpi_ckpt
    siglip = SiglipEncoder(**siglip_kwargs)

    print("[init] loading MiniLM text encoder...")
    text_enc = MiniLMTextEncoder(device=retr_device)

    print(f"[init] loading retriever checkpoint {retr_ckpt}...")
    retriever_model = Retriever().to(retr_device)
    state = torch.load(retr_ckpt, map_location=retr_device, weights_only=True)
    retriever_model.load_state_dict(state["model"])
    retriever_model.eval()

    bank = LiveMemoryBank()

    print("[init] connecting to YAM + cameras (top + left)...")
    env = LongTaskPolicyEnv(
        channel=cfg["robot"]["channel"],
        top_serial=cfg["cameras"]["top_serial"],
        left_serial=cfg["cameras"]["left_serial"],
        stage_manager=stage_mgr,
        video_dir=None if args.no_video else log_dir,
        video_fps=args.hz,
    )

    # -- pre-flight (mirror run_long_task.py)
    state_arr = env.current_state()
    if state_arr.shape != (ACTION_DIM,):
        sys.exit(f"[preflight] bad joint state shape: {state_arr.shape}")
    if np.any(np.abs(state_arr) > JOINT_LIMIT_RAD):
        sys.exit(f"[preflight] joint outside +/-{JOINT_LIMIT_RAD} rad: {state_arr}")
    obs = env.get_obs()
    for k in ("observation/top_image", "observation/left_image", "observation/memory_image"):
        img = obs[k]
        if img.shape != (480, 640, 3) or img.dtype != np.uint8:
            sys.exit(f"[preflight] bad image {k}: shape={img.shape} dtype={img.dtype}")
        print(
            f"[preflight] {k}: shape={img.shape} dtype={img.dtype} channel_means R/G/B="
            f"{img[...,0].mean():.1f}/{img[...,1].mean():.1f}/{img[...,2].mean():.1f}"
        )
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
    print(f"[preflight] state={np.array2string(state_arr, precision=3, suppress_small=True)}")
    print(f"[preflight] stage={stage_mgr.stage_id} ({stage_mgr.stage_name})  prompt={obs['prompt']!r}")

    # -- server
    print("[init] connecting to inference server...")
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
            f"{diff.max():.3f} rad on joint {int(diff.argmax())} (limit {FIRST_CHUNK_MAX_JUMP} rad)."
        )

    if args.dry_run:
        # Smoke-test the retrieval path with the current top frame.
        print("[dry-run] testing retrieval pipeline with one frame...")
        try:
            top_u8 = obs["observation/top_image"]
            g, p = siglip.encode_one(top_u8)
            bank.append(top_u8, g, p, t_norm=0.0)
            relational = plan.instruction_for_stage(3)  # fake "entered query1"
            te = text_enc.encode(relational)
            new_prompt, idx = retrieve_and_rewrite(
                retriever_model, bank, te,
                g.astype(np.float32), p.astype(np.float32),
                cur_t_norm=0.1,
                original_prompt=relational,
                rewriter_model=rewriter_model,
                device=retr_device,
            )
            print(f"[dry-run] relational={relational!r}")
            print(f"[dry-run] retriever top1_idx={idx}  rewritten={new_prompt!r}")
        except Exception as e:
            print(f"[dry-run] retrieve_and_rewrite failed: {e}", file=sys.stderr)

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
    logger.event(
        f"variant=long_task_retriever  retr_ckpt={retr_ckpt}  rewriter={rewriter_model}  "
        f"mem_hz={mem_hz}  openpi_ckpt={openpi_ckpt}"
    )
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
    sample_period_s = 1.0 / max(mem_hz, 1e-6)
    next_sample_t = t_run_start
    final_advance_t: Optional[float] = None
    stop_reason: Optional[str] = None

    logger.record_chunk(chunk_idx, obs, chunk, infer_ms)

    try:
        while True:
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

                # Live SigLIP bank — sample top camera at mem_hz across ALL stages.
                # Causality is automatic (only past frames can have been appended).
                if t_step >= next_sample_t:
                    try:
                        top_u8 = env.current_top_frame()
                        g, p = siglip.encode_one(top_u8)
                        t_norm = (time.monotonic() - t_run_start) / max(args.max_time_s, 1e-6)
                        bank.append(top_u8, g, p, t_norm=t_norm)
                        logger.event(
                            f"[bank] +keyframe k={len(bank)} stage={stage_mgr.stage_id} "
                            f"t_norm={t_norm:.3f}"
                        )
                    except Exception as e:
                        logger.event(f"[bank] sample failed: {e}")
                    next_sample_t = t_step + sample_period_s

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
                final = _do_advance(
                    env, stage_mgr, bank, siglip, text_enc, retriever_model,
                    rewriter_model, retr_device,
                    t_run_start, args.max_time_s, logger, flag, dt_target,
                )
                if final and final_advance_t is None:
                    final_advance_t = time.monotonic()

            if final_advance_t is not None and (time.monotonic() - final_advance_t) >= SHUTDOWN_AFTER_FINAL_ADVANCE_S:
                stop_reason = "stage7-shutdown"
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
                f"chunk{chunk_idx} stage={stage_mgr.stage_id} infer_ms={infer_ms:.1f} "
                f"boundary_max_jump={jump.max():.4f} on joint {int(jump.argmax())}"
            )
    finally:
        stop_reason = stop_reason or "loop_exit"
        try:
            print(
                f"[stop] reason={stop_reason} steps={step_count} chunks={chunk_idx + 1} "
                f"stage={stage_mgr.stage_id} ({stage_mgr.stage_name}) bank_size={len(bank)}"
            )
            print("[stop] holding last pose for 0.5s")
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
        cli_args_dict = {k: str(getattr(args, k)) for k in vars(args)}
        logger.flush(
            cli_args=cli_args_dict,
            server_metadata=server_metadata,
            prompt="(per-stage; see stage_events)",
            stop_reason=stop_reason,
        )
        print(f"[done] log_dir={log_dir}")


if __name__ == "__main__":
    main()
