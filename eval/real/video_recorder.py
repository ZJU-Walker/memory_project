"""Background camera-grabber threads that stream the full episode to MP4.

A `RealSenseCamera` pipeline can only have one reader at a time, so a
`_CameraStreamer` becomes the sole owner of its camera. It runs a background
thread that calls `camera.read()` continuously (which blocks on the camera's
30 fps frame source), keeps the latest frame in a lock-protected slot for
`YamPolicyEnv.get_obs()` to snapshot, and pipes every grabbed frame into a
streaming MP4 writer.

Why background threads:
- Capturing frames inside the main control loop at 30 Hz would add ~15-45 ms
  of camera-read latency per step on top of the 33 ms control budget.
- The eval policy only consumes a frame at chunk boundaries (~3 Hz). Without
  background grab, the saved video would be a 3 fps slideshow.

`get_obs()` reads the most recent buffered frame; staleness is bounded by the
camera period (~33 ms), which is well below the chunk re-inference interval.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

# imageio.v3 + pyav for streaming MP4 write — verified available in the venv.
import imageio.v3 as iio


class _CameraStreamer:
    """Owns one RealSenseCamera, streams frames to MP4 in a background thread."""

    def __init__(self, camera, name: str, video_path: Optional[Path], fps: float = 30.0) -> None:
        self._camera = camera
        self._name = name
        self._video_path = video_path
        self._fps = fps

        self._latest: Optional[np.ndarray] = None
        self._latest_lock = threading.Lock()
        self._first_frame_event = threading.Event()
        self._stop = threading.Event()
        self._frames_written = 0

        self._writer = None
        self._writer_initialized = False
        if self._video_path is not None:
            self._video_path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = iio.imopen(str(self._video_path), "w", plugin="pyav")

        self._thread = threading.Thread(
            target=self._loop, daemon=True, name=f"camgrab-{name}"
        )

    def start(self) -> None:
        self._thread.start()

    def wait_for_first_frame(self, timeout: float = 5.0) -> None:
        if not self._first_frame_event.wait(timeout=timeout):
            raise RuntimeError(f"camera {self._name!r}: no frame within {timeout}s")

    def latest(self) -> np.ndarray:
        """Return the most recent frame (blocks briefly if none yet)."""
        if not self._first_frame_event.is_set():
            self.wait_for_first_frame()
        with self._latest_lock:
            assert self._latest is not None
            return self._latest.copy()

    def stop(self, join_timeout: float = 3.0) -> None:
        self._stop.set()
        self._thread.join(timeout=join_timeout)
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception as e:
                print(f"[video] {self._name}: writer close failed: {e}")
            self._writer = None

    @property
    def frames_written(self) -> int:
        return self._frames_written

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                img, _ = self._camera.read()
            except Exception as e:
                print(f"[video] {self._name}: read failed: {e}")
                time.sleep(0.05)
                continue

            img = np.ascontiguousarray(img)
            with self._latest_lock:
                self._latest = img
            self._first_frame_event.set()

            if self._writer is not None:
                try:
                    if not self._writer_initialized:
                        # init_video_stream needs to be called once after the first
                        # frame arrives so we know the resolution.
                        self._writer.init_video_stream("libx264", fps=self._fps)
                        self._writer_initialized = True
                    self._writer.write_frame(img)
                    self._frames_written += 1
                except Exception as e:
                    print(f"[video] {self._name}: write failed: {e}")
                    self._writer = None
