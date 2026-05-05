"""Thin YAM + 3-camera env that produces the exact policy observation dict.

Mirrors the keys used by `eval/offline_eval.py` (which replays recorded MP4s
through the same policy), so an inference call here is byte-for-byte the same
shape the model was trained on.

Imports `YAMRobot` and `RealSenseCamera` from gello_software as a library;
does NOT use `gello.env.RobotEnv` (its key names — `top_camera_rgb`,
`joint_positions` — would need a remap layer that could silently drift from
the policy contract).

Each camera is owned by a `_CameraStreamer` background thread that grabs at
the camera's native 30 Hz and pipes every frame to a streaming MP4 writer.
`get_obs()` snapshots the latest buffered frame; staleness is bounded by the
camera period (~33 ms).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from gello.cameras.realsense_camera import RealSenseCamera
from gello.robots.yam import YAMRobot

from eval.real.video_recorder import _CameraStreamer

WARMUP_FRAMES = 10  # RealSense returns black frames for the first ~5–10 reads


class YamPolicyEnv:
    """Reads YAM joint state + 3 RealSense RGB streams, builds the policy obs dict."""

    def __init__(
        self,
        channel: str,
        top_serial: str,
        left_serial: str,
        right_serial: str,
        prompt: str,
        video_dir: Optional[Path] = None,
        video_fps: float = 30.0,
    ) -> None:
        self._robot = YAMRobot(channel=channel)
        self._top_cam = RealSenseCamera(device_id=top_serial)
        self._left_cam = RealSenseCamera(device_id=left_serial)
        self._right_cam = RealSenseCamera(device_id=right_serial)
        self._prompt = prompt

        # Drain RealSense warmup frames synchronously *before* handing the
        # cameras over to the streamer threads (only one reader allowed).
        for _ in range(WARMUP_FRAMES):
            self._top_cam.read()
            self._left_cam.read()
            self._right_cam.read()
            time.sleep(0.005)

        top_path = left_path = right_path = None
        if video_dir is not None:
            video_dir = Path(video_dir)
            top_path = video_dir / "top_camera_rgb.mp4"
            left_path = video_dir / "left_camera_rgb.mp4"
            right_path = video_dir / "right_camera_rgb.mp4"

        self._streamers = [
            _CameraStreamer(self._top_cam, "top", top_path, fps=video_fps),
            _CameraStreamer(self._left_cam, "left", left_path, fps=video_fps),
            _CameraStreamer(self._right_cam, "right", right_path, fps=video_fps),
        ]
        for s in self._streamers:
            s.start()
        for s in self._streamers:
            s.wait_for_first_frame(timeout=5.0)

    def get_obs(self) -> Dict[str, np.ndarray]:
        top = self._streamers[0].latest()
        left = self._streamers[1].latest()
        right = self._streamers[2].latest()
        state = self._robot.get_joint_state().astype(np.float32)
        return {
            "observation/top_image": top,
            "observation/left_image": left,
            "observation/right_image": right,
            "observation/state": state,
            "prompt": self._prompt,
        }

    def command(self, action_7: np.ndarray) -> None:
        assert action_7.shape == (7,), f"expected (7,) action, got {action_7.shape}"
        self._robot.command_joint_state(action_7)

    def current_state(self) -> np.ndarray:
        return self._robot.get_joint_state().astype(np.float32)

    def close(self) -> None:
        # Order matters:
        # 1. Sleep briefly to let i2rt's motor_chain thread drain any in-flight
        #    CAN send. The hold-pose loop in run_eval has been sending the held
        #    command at 30 Hz; once that loop returns, the bg thread may still
        #    be mid-send — without this sleep, motor_chain.close() can shut the
        #    socket out from under it and raise
        #    "ValueError: file descriptor cannot be a negative integer (-1)".
        #    Note: we deliberately do NOT switch to zero-torque mode here — the
        #    user chose "hold last pose" as the stop behavior (so a grasped
        #    object doesn't drop on exit).
        # 2. Stop the grabber threads (they own the camera reads).
        # 3. Stop each RealSense pipeline (now safe — no readers active).
        # 4. Close the i2rt CAN driver — joins bg threads and closes the socket.
        time.sleep(0.1)

        for s in self._streamers:
            try:
                s.stop()
            except Exception as e:
                print(f"[env] streamer stop failed: {e}")
        print(
            f"[env] video frames written: "
            f"top={self._streamers[0].frames_written} "
            f"left={self._streamers[1].frames_written} "
            f"right={self._streamers[2].frames_written}"
        )
        for cam, name in (
            (self._top_cam, "top"),
            (self._left_cam, "left"),
            (self._right_cam, "right"),
        ):
            try:
                cam._pipeline.stop()
            except Exception as e:
                print(f"[env] camera {name} pipeline.stop failed: {e}")
        try:
            # YAMRobot wraps i2rt's MotorChainRobot at .robot; close joins the
            # background CAN threads so the process can actually exit.
            self._robot.robot.close()
        except Exception as e:
            print(f"[env] robot close failed (non-fatal): {e}")

    @property
    def robot(self) -> YAMRobot:
        return self._robot
