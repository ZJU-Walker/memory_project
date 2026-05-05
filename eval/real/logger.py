"""Per-rollout recorder for the real-robot eval client.

Mirrors the offline-eval npz layout where reasonable so the same downstream
tooling (plot_eef.py, sim_playback.py) can chew on these traces with minor
key remaps.

Layout written by `flush()`:
    runs/<ts>/
        metadata.json     # CLI args, server metadata, prompt, git commit
        trace.npz         # per control step
        chunks.npz        # per inference call
        events.log        # text events
        top_camera_rgb.mp4    # one frame per chunk (i.e. per inference)
        left_camera_rgb.mp4
        right_camera_rgb.mp4
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


class RolloutLogger:
    def __init__(self, log_dir: Path, save_frames: bool = True) -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._save_frames = save_frames
        if self._save_frames:
            (self._dir / "frames").mkdir(exist_ok=True)

        # per control-step
        self._step_t: List[float] = []
        self._step_state: List[np.ndarray] = []
        self._step_action: List[np.ndarray] = []
        self._step_chunk_idx: List[int] = []
        self._step_idx_in_chunk: List[int] = []

        # per inference call
        self._chunk_t: List[float] = []
        self._chunk_obs_state: List[np.ndarray] = []
        self._chunk_actions: List[np.ndarray] = []
        self._chunk_infer_ms: List[float] = []

        self._events: List[str] = []
        self._t_zero = time.monotonic()

    def _t(self) -> float:
        return time.monotonic() - self._t_zero

    def record_step(
        self,
        step_idx: int,
        state_7: np.ndarray,
        action_7: np.ndarray,
        chunk_idx: int,
        idx_in_chunk: int,
    ) -> None:
        self._step_t.append(self._t())
        self._step_state.append(np.asarray(state_7, dtype=np.float32))
        self._step_action.append(np.asarray(action_7, dtype=np.float32))
        self._step_chunk_idx.append(int(chunk_idx))
        self._step_idx_in_chunk.append(int(idx_in_chunk))

    def record_chunk(
        self,
        chunk_idx: int,
        obs: Dict[str, np.ndarray],
        chunk: np.ndarray,
        infer_ms: float,
    ) -> None:
        self._chunk_t.append(self._t())
        self._chunk_obs_state.append(np.asarray(obs["observation/state"], dtype=np.float32))
        self._chunk_actions.append(np.asarray(chunk, dtype=np.float32))
        self._chunk_infer_ms.append(float(infer_ms))
        if self._save_frames:
            top = np.asarray(obs["observation/top_image"], dtype=np.uint8)
            left = np.asarray(obs["observation/left_image"], dtype=np.uint8)
            right = np.asarray(obs["observation/right_image"], dtype=np.uint8)
            self._save_chunk_jpgs(chunk_idx, top, left, right)

    def _save_chunk_jpgs(
        self, chunk_idx: int, top: np.ndarray, left: np.ndarray, right: np.ndarray
    ) -> None:
        try:
            import imageio.v3 as iio
        except ImportError:
            return
        frames_dir = self._dir / "frames"
        for cam, img in (("top", top), ("left", left), ("right", right)):
            iio.imwrite(frames_dir / f"chunk_{chunk_idx:04d}_{cam}.jpg", img, quality=85)
            # Also overwrite a stable filename for live monitoring (e.g. `feh -R 1 latest_top.jpg`).
            iio.imwrite(self._dir / f"latest_{cam}.jpg", img, quality=85)

    def event(self, msg: str) -> None:
        line = f"[{self._t():7.3f}] {msg}"
        self._events.append(line)
        print(line, flush=True)

    def flush(
        self,
        cli_args: Dict[str, Any],
        server_metadata: Dict[str, Any],
        prompt: str,
        stop_reason: str,
    ) -> None:
        meta = {
            "cli_args": _to_jsonable(cli_args),
            "server_metadata": _to_jsonable(server_metadata),
            "prompt": prompt,
            "stop_reason": stop_reason,
            "memory_project_git": _git_head(Path(__file__).resolve().parent.parent.parent),
            "n_steps": len(self._step_t),
            "n_chunks": len(self._chunk_t),
        }
        (self._dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        if self._step_t:
            np.savez(
                self._dir / "trace.npz",
                t_wall=np.asarray(self._step_t, dtype=np.float64),
                state=np.stack(self._step_state, axis=0),
                executed_action=np.stack(self._step_action, axis=0),
                chunk_idx=np.asarray(self._step_chunk_idx, dtype=np.int32),
                idx_in_chunk=np.asarray(self._step_idx_in_chunk, dtype=np.int32),
            )

        if self._chunk_t:
            np.savez(
                self._dir / "chunks.npz",
                chunk_t_wall=np.asarray(self._chunk_t, dtype=np.float64),
                chunk_obs_state=np.stack(self._chunk_obs_state, axis=0),
                chunk_actions=np.stack(self._chunk_actions, axis=0),
                infer_ms=np.asarray(self._chunk_infer_ms, dtype=np.float32),
            )

        (self._dir / "events.log").write_text("\n".join(self._events) + "\n")


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def _git_head(repo_dir: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"
