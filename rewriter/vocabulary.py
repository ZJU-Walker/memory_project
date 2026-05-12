"""Canonical vocabulary for the plate-memory task.

Confirmed against all 38 demos in dataset/PLATE_TASK_1/*/metadata.json:
every demo's episode_plan.mapping has the same 4 object names and the
same 4 plate colors.
"""

from __future__ import annotations

import re

OBJECTS_CANONICAL: list[str] = ["banana", "gray_paper_box", "red_chili", "yellow_cheese"]
PLATES_CANONICAL: list[str] = ["green", "light_blue", "orange", "pink"]


def natural(name: str) -> str:
    """`red_chili` -> `red chili`. Identity on already-natural names."""
    return name.replace("_", " ")


OBJECTS_NATURAL: list[str] = [natural(x) for x in OBJECTS_CANONICAL]
PLATES_NATURAL: list[str] = [natural(x) for x in PLATES_CANONICAL]


# Color words that are NOT canonical plate names — used to flag drift like
# "red plate", "yellow plate", "blue plate" (without preceding "light"), etc.
_OFF_VOCAB_PLATE_COLORS = {
    "red", "yellow", "purple", "white", "black", "brown", "gray", "grey",
}

# Short blacklist of common off-vocab object substitutes.
_OBJECT_BLACKLIST = [
    "gray toy", "grey toy", "grey paper box",      # gray_paper_box drift
    "red pepper", "chili pepper", "chilli",         # red_chili drift
    "cheese block", "yellow block",                 # yellow_cheese drift
    "yellow fruit",                                 # banana drift
]


def is_in_vocabulary(text: str) -> bool:
    """Return True iff every object/plate-like reference in `text` uses a
    canonical natural-form name. Heuristic — not a parser.

    Checks:
      1. For every "plate" token, the immediately preceding word must be
         in {green, orange, pink} OR be "blue" preceded by "light".
      2. No common off-vocab object substring appears.
    """
    lower = text.lower()
    words = re.findall(r"[a-z]+", lower)
    canonical_plate_singles = {"green", "orange", "pink"}

    for i, w in enumerate(words):
        if w != "plate" or i == 0:
            continue
        prev = words[i - 1]
        if prev in canonical_plate_singles:
            continue
        if prev == "blue" and i >= 2 and words[i - 2] == "light":
            continue
        if prev in _OFF_VOCAB_PLATE_COLORS or prev == "blue":
            return False
        # else: "plate" preceded by an unrelated word (e.g. "each plate",
        # "colored plate") — not a vocabulary violation, just generic phrasing.

    for bad in _OBJECT_BLACKLIST:
        if bad in lower:
            return False
    return True
