"""Offline keyframe sampling policy for the memory bank.

Pure-Python, no model calls. Given a demo's metadata, returns a list of frame
indices to encode and cache.
"""

from __future__ import annotations

MAX_KEYFRAMES = 128
OBSERVE_STAGE_ID = 0
OBSERVE_STRIDE = 10   # ~3 Hz inside observe (short and information-dense)
DEFAULT_STRIDE = 30   # ~1 Hz everywhere else


def sample_keyframes(meta: dict, num_steps: int) -> list[int]:
    """Pick keyframe indices for one demo.

    Policy:
      1. Uniform stride per segment (3 Hz for observe, 1 Hz elsewhere).
      2. Boundary frames: first two and last frame of each segment.
      3. Dedupe + sort + clamp to [0, num_steps).
      4. Cap to MAX_KEYFRAMES via even subsampling (preserves stage coverage).
    """
    picks: set[int] = set()
    for seg in meta["segments"]:
        start = int(seg["start_step"])
        end = int(seg["end_step"])  # half-open
        if end <= start:
            continue
        stride = OBSERVE_STRIDE if int(seg["stage_id"]) == OBSERVE_STAGE_ID else DEFAULT_STRIDE
        picks.update(range(start, end, stride))
        # boundaries
        picks.add(start)
        if start + 1 < end:
            picks.add(start + 1)
        picks.add(end - 1)

    frame_ids = sorted(i for i in picks if 0 <= i < num_steps)
    if len(frame_ids) > MAX_KEYFRAMES:
        idx = [round(i * (len(frame_ids) - 1) / (MAX_KEYFRAMES - 1)) for i in range(MAX_KEYFRAMES)]
        frame_ids = sorted(set(frame_ids[i] for i in idx))
    return frame_ids
