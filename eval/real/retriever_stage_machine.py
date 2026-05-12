"""StageManager subclass with runtime-replaceable memory_image and instruction.

Used by `run_long_task_retriever.py` so the run loop can inject the retriever's
top-1 keyframe + the rewriter's resolved instruction on query stages without
having to modify `LongTaskPolicyEnv`. Stays in a dedicated module (no
LongTaskPolicyEnv import) so it's testable on the workstation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from eval.real.stage_machine import QueryPlan, StageManager


class RetrieverStageManager(StageManager):
    """StageManager with optional per-stage overrides for memory_image + instruction."""

    def __init__(self, query_plan: QueryPlan) -> None:
        super().__init__(query_plan)
        self._override_memory_image: Optional[np.ndarray] = None
        self._override_instruction: Optional[str] = None

    def set_override(self, memory_image: np.ndarray, instruction: str) -> None:
        if memory_image.shape != (480, 640, 3) or memory_image.dtype != np.uint8:
            raise ValueError(
                f"override memory_image must be (480,640,3) uint8, "
                f"got shape={memory_image.shape} dtype={memory_image.dtype}"
            )
        self._override_memory_image = memory_image.copy()
        self._override_instruction = instruction

    def clear_override(self) -> None:
        self._override_memory_image = None
        self._override_instruction = None

    @property
    def memory_image(self) -> np.ndarray:  # type: ignore[override]
        if self._override_memory_image is not None:
            return self._override_memory_image
        return super().memory_image

    @property
    def instruction(self) -> str:  # type: ignore[override]
        if self._override_instruction is not None:
            return self._override_instruction
        return super().instruction

    def advance(self, current_top_frame: np.ndarray) -> bool:  # type: ignore[override]
        # Each transition resets to base stage defaults; the run loop reinstalls
        # an override immediately after if the new stage is in QUERY_STAGES.
        self.clear_override()
        return super().advance(current_top_frame)
