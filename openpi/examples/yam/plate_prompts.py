"""Detailed per-frame prompts for the plate-memory task.

The relational instructions in each demo (e.g. "Place the object originally on
the pink plate onto the pink plate.") are rewritten on query stages to bake in
the resolved object name ("put red chili on the pink plate"). At training time
these strings come from each demo's metadata.json `episode_plan`. At inference
time the same helper produces the oracle prompt; the long-term plan is to
replace it with a VLM that consumes retrieved keyframes + the relational prompt.

Stage mapping (segment["instruction"] is the original relational string):
    stage_id 0  observe       -> segment.instruction
    stage_id 1  mix           -> segment.instruction
    stage_id 2  delay         -> segment.instruction
    stage_id 3..6  query1..4  -> "put {object} on the {plate} plate"  (resolved)
    stage_id 7  return_home   -> segment.instruction

object / plate come from episode_plan.queries[stage_id - 3], with underscores
replaced by spaces ("red_chili" -> "red chili", "light_blue" -> "light blue").
"""

from __future__ import annotations

from typing import Any, Sequence

QUERY_STAGES = {3, 4, 5, 6}


def _natural(name: str) -> str:
    return name.replace("_", " ")


def detailed_prompt(meta: dict[str, Any], stage_id: int) -> str:
    seg = next(s for s in meta["segments"] if s["stage_id"] == stage_id)
    if stage_id not in QUERY_STAGES:
        return seg["instruction"]
    query = meta["episode_plan"]["queries"][stage_id - 3]
    return f"put {_natural(query['correct_object'])} on the {_natural(query['target_plate'])} plate"


def build_per_frame_prompts(meta: dict[str, Any], stage_id_array: Sequence[int]) -> list[str]:
    cache: dict[int, str] = {}
    out: list[str] = []
    for s in stage_id_array:
        s = int(s)
        if s not in cache:
            cache[s] = detailed_prompt(meta, s)
        out.append(cache[s])
    return out
