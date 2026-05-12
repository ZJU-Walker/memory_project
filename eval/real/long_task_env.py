"""YAM + 2-camera env for the long-horizon memory-conditioned policy.

Trimmed clone of `eval/real/robot_env.YamPolicyEnv`. Two RealSense streams
(top, left); the right wrist camera is intentionally absent — the model's
`right_wrist_0_rgb` slot is filled at training time by `YamLongTaskInputs`
with the `observation/memory_image` field, which the StageManager owns
(observe-stage keyframe on stages 3..6, zeros elsewhere).

Each camera is owned by a `_CameraStreamer` background thread that grabs at
the camera's native 30 Hz and pipes every frame to a streaming MP4 writer.
`get_obs()` snapshots the latest buffered frame; staleness is bounded by the
camera period (~33 ms).

Obs dict produced (matches `convert_long_task_to_lerobot.py:98-104` exactly):

    observation/top_image    (480,640,3) uint8   — live top RealSense
    observation/left_image   (480,640,3) uint8   — live left wrist RealSense
    observation/memory_image (480,640,3) uint8   — observe keyframe OR zeros
    observation/state        (7,)        float32 — YAM joint state
    observation/has_memory   (1,)        float32 — 1.0 on query stages, 0.0 else
    prompt                   str                  — stage-specific instruction
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from gello.cameras.realsense_camera import RealSenseCamera
from gello.robots.yam import YAMRobot

from eval.real.stage_machine import StageManager
from eval.real.video_recorder import _CameraStreamer

WARMUP_FRAMES = 10  # RealSense returns black frames for the first ~5–10 reads


class LongTaskPolicyEnv:
    """Reads YAM joint state + 2 RealSense streams + StageManager memory channel."""

    def __init__(
        self,
        channel: str,
        top_serial: str,
        left_serial: str,
        stage_manager: StageManager,
        video_dir: Optional[Path] = None,
        video_fps: float = 30.0,
    ) -> None:
        self._robot = YAMRobot(channel=channel)
        self._top_cam = RealSenseCamera(device_id=top_serial)
        self._left_cam = RealSenseCamera(device_id=left_serial)
        self._stage_mgr = stage_manager

        # Drain RealSense warmup frames synchronously *before* handing the
        # cameras to streamer threads (only one reader allowed per pipeline).
        for _ in range(WARMUP_FRAMES):
            self._top_cam.read()
            self._left_cam.read()
            time.sleep(0.005)

        top_path = left_path = None
        if video_dir is not None:
            video_dir = Path(video_dir)
            top_path = video_dir / "top_camera_rgb.mp4"
            left_path = video_dir / "left_camera_rgb.mp4"

        self._streamers = [
            _CameraStreamer(self._top_cam, "top", top_path, fps=video_fps),
            _CameraStreamer(self._left_cam, "left", left_path, fps=video_fps),
        ]
        for s in self._streamers:
            s.start()
        for s in self._streamers:
            s.wait_for_first_frame(timeout=5.0)

    def get_obs(self) -> Dict[str, np.ndarray]:
        top = self._streamers[0].latest()
        left = self._streamers[1].latest()
        state = self._robot.get_joint_state().astype(np.float32)
        has_mem = 1.0 if self._stage_mgr.has_memory else 0.0
        return {
            "observation/top_image": top,
            "observation/left_image": left,
            "observation/memory_image": self._stage_mgr.memory_image,
            "observation/state": state,
            "observation/has_memory": np.array([has_mem], dtype=np.float32),
            "prompt": self._stage_mgr.instruction,
        }

    def current_top_frame(self) -> np.ndarray:
        """Latest top frame — used by run_long_task to capture the observe keyframe."""
        return self._streamers[0].latest()

    def command(self, action_7: np.ndarray) -> None:
        assert action_7.shape == (7,), f"expected (7,) action, got {action_7.shape}"
        self._robot.command_joint_state(action_7)

    def current_state(self) -> np.ndarray:
        return self._robot.get_joint_state().astype(np.float32)

    def close(self) -> None:
        # Order matters — see robot_env.YamPolicyEnv.close for the rationale.
        # Brief sleep lets i2rt's CAN background thread drain in-flight sends
        # before we tear down the camera streamers and motor backend.
        time.sleep(0.1)

        for s in self._streamers:
            try:
                s.stop()
            except Exception as e:
                print(f"[env] streamer stop failed: {e}")
        print(
            f"[env] video frames written: "
            f"top={self._streamers[0].frames_written} "
            f"left={self._streamers[1].frames_written}"
        )
        for cam, name in ((self._top_cam, "top"), (self._left_cam, "left")):
            try:
                cam._pipeline.stop()
            except Exception as e:
                print(f"[env] camera {name} pipeline.stop failed: {e}")
        try:
            self._robot.robot.close()
        except Exception as e:
            print(f"[env] robot close failed (non-fatal): {e}")

    @property
    def robot(self) -> YAMRobot:
        return self._robot

    @property
    def stage_manager(self) -> StageManager:
        return self._stage_mgr
