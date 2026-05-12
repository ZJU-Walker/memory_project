"""Per-rollout recorder for the real-robot eval client.

Mirrors the offline-eval npz layout where reasonable so the same downstream
tooling (plot_eef.py, sim_playback.py) can chew on these traces with minor
key remaps.

Layout written by `flush()`:
    runs/<ts>/
        metadata.json     # CLI args, server metadata, prompt, git commit, +
                          # query_plan + stage_events for long-task runs
        trace.npz         # per control step (+ stage_id column for long-task)
        chunks.npz        # per inference call
        events.log        # text events
        memory_keyframe.jpg   # observe-stage keyframe (long-task only)
        top_camera_rgb.mp4    # one frame per chunk (i.e. per inference)
        left_camera_rgb.mp4
        right_camera_rgb.mp4

The long-task additions (`record_stage_transition`, `set_query_plan`, and the
optional `stage_id` kwarg on `record_step`) are no-ops when unused, so the
existing CUP `run_eval.py` keeps working unchanged.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        self._step_stage_id: List[int] = []

        # per inference call
        self._chunk_t: List[float] = []
        self._chunk_obs_state: List[np.ndarray] = []
        self._chunk_actions: List[np.ndarray] = []
        self._chunk_infer_ms: List[float] = []

        self._events: List[str] = []
        self._t_zero = time.monotonic()

        # Long-task extensions (left empty for CUP runs).
        self._stage_events: List[Dict[str, Any]] = []
        self._query_plan: Optional[Dict[str, Any]] = None
        self._memory_keyframe_saved = False

    def _t(self) -> float:
        return time.monotonic() - self._t_zero

    def record_step(
        self,
        step_idx: int,
        state_7: np.ndarray,
        action_7: np.ndarray,
        chunk_idx: int,
        idx_in_chunk: int,
        stage_id: int = -1,
    ) -> None:
        self._step_t.append(self._t())
        self._step_state.append(np.asarray(state_7, dtype=np.float32))
        self._step_action.append(np.asarray(action_7, dtype=np.float32))
        self._step_chunk_idx.append(int(chunk_idx))
        self._step_idx_in_chunk.append(int(idx_in_chunk))
        self._step_stage_id.append(int(stage_id))

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
            # CUP obs has top/left/right_image; long-task obs has top/left + memory_image.
            cams: List[tuple] = []
            for key, name in (
                ("observation/top_image", "top"),
                ("observation/left_image", "left"),
                ("observation/right_image", "right"),
                ("observation/memory_image", "memory"),
            ):
                if key in obs:
                    cams.append((name, np.asarray(obs[key], dtype=np.uint8)))
            self._save_chunk_jpgs(chunk_idx, cams)

    def _save_chunk_jpgs(
        self, chunk_idx: int, cams: List[tuple]
    ) -> None:
        try:
            import imageio.v3 as iio
        except ImportError:
            return
        frames_dir = self._dir / "frames"
        for cam, img in cams:
            iio.imwrite(frames_dir / f"chunk_{chunk_idx:04d}_{cam}.jpg", img, quality=85)
            # Also overwrite a stable filename for live monitoring (e.g. `feh -R 1 latest_top.jpg`).
            iio.imwrite(self._dir / f"latest_{cam}.jpg", img, quality=85)

    def event(self, msg: str) -> None:
        line = f"[{self._t():7.3f}] {msg}"
        self._events.append(line)
        print(line, flush=True)

    def set_query_plan(self, plan_dict: Dict[str, Any]) -> None:
        """Store the long-task query plan for serialization in metadata.json."""
        self._query_plan = dict(plan_dict)

    def record_stage_transition(
        self,
        stage_id: int,
        stage_name: str,
        instruction: str,
        memory_image: Optional[np.ndarray] = None,
    ) -> None:
        """Record a long-task stage advance and persist the memory keyframe once."""
        self._stage_events.append({
            "t": self._t(),
            "stage_id": int(stage_id),
            "stage_name": stage_name,
            "instruction": instruction,
        })
        # Save the observe-stage keyframe to disk on the first transition into a
        # query stage (it's identical for all four queries within a run).
        if (
            memory_image is not None
            and 3 <= stage_id <= 6
            and not self._memory_keyframe_saved
        ):
            try:
                import imageio.v3 as iio
                iio.imwrite(
                    self._dir / "memory_keyframe.jpg",
                    np.asarray(memory_image, dtype=np.uint8),
                    quality=85,
                )
                self._memory_keyframe_saved = True
            except Exception as e:
                print(f"[logger] memory keyframe save failed: {e}")

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
        if self._query_plan is not None:
            meta["query_plan"] = _to_jsonable(self._query_plan)
        if self._stage_events:
            meta["stage_events"] = _to_jsonable(self._stage_events)
        (self._dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        if self._step_t:
            trace_kwargs = dict(
                t_wall=np.asarray(self._step_t, dtype=np.float64),
                state=np.stack(self._step_state, axis=0),
                executed_action=np.stack(self._step_action, axis=0),
                chunk_idx=np.asarray(self._step_chunk_idx, dtype=np.int32),
                idx_in_chunk=np.asarray(self._step_idx_in_chunk, dtype=np.int32),
            )
            # Only emit the stage_id column when at least one step recorded a real
            # stage (>=0); CUP runs leave it at the -1 sentinel and we omit it.
            if any(s >= 0 for s in self._step_stage_id):
                trace_kwargs["stage_id"] = np.asarray(self._step_stage_id, dtype=np.int32)
            np.savez(self._dir / "trace.npz", **trace_kwargs)

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
