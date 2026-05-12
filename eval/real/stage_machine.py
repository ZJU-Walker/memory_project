"""Stage state machine for the long-horizon plate-memory task.

The trained policy `pi05_long_task_mem_lora` operates on an 8-stage episode:

    0 observe     -> "Observe and remember which object is on each colored plate."
    1 mix         -> "Move all objects to the right side of the table."
    2 delay       -> "Wait briefly before continuing."
    3 query1      -> "Place the object originally on the {src} plate onto the {tgt} plate."
    4 query2      -> ... (memory channel ON for 3..6)
    5 query3
    6 query4
    7 return_home -> "Return arm to home position."

The dataset conversion script seeds `right_wrist_0_rgb` with the observe-stage
keyframe on stages 3..6 (mask True) and zeros otherwise (mask False); the deploy
client reproduces that wiring at runtime.

Run `python -m eval.real.stage_machine` for unit smoke checks.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np

PLATES = ["green", "light_blue", "orange", "pink"]
OBJECTS = ["gray_paper_box", "banana", "red_chili", "yellow_cheese"]

STAGE_NAMES = [
    "observe",      # 0
    "mix",          # 1
    "delay",        # 2
    "query1",       # 3
    "query2",       # 4
    "query3",       # 5
    "query4",       # 6
    "return_home",  # 7
]

QUERY_STAGES = {3, 4, 5, 6}

# Default fixed plan (used when mode='fixed'). Hand-picked so each query exercises
# a distinct source plate and the targets cover all four colors.
_FIXED_MAPPING = {
    "green":      "gray_paper_box",
    "light_blue": "banana",
    "orange":     "red_chili",
    "pink":       "yellow_cheese",
}
_FIXED_QUERIES = [
    ("orange",     "green"),
    ("light_blue", "pink"),
    ("pink",       "orange"),
    ("green",      "light_blue"),
]


@dataclass(frozen=True)
class Query:
    source_plate: str
    target_plate: str


class QueryPlan:
    """Plate->object mapping plus the four query (source, target) pairs.

    mode='fixed'  -> deterministic, identical across runs (best for retrieval-method comparisons)
    mode='random' -> seeded random plan; queries cover all 4 source plates exactly once,
                     targets sampled with replacement (matches dataset distribution)
    mode='manual' -> operator types each query at boot
    """

    def __init__(self, mode: str = "fixed", seed: Optional[int] = None) -> None:
        if mode not in ("fixed", "random", "manual"):
            raise ValueError(f"unknown query mode: {mode!r}")
        self._mode = mode
        self._seed = seed

        if mode == "fixed":
            self._mapping = dict(_FIXED_MAPPING)
            self._queries = [Query(s, t) for s, t in _FIXED_QUERIES]
        elif mode == "random":
            rng = random.Random(seed)
            objs = list(OBJECTS)
            rng.shuffle(objs)
            self._mapping = {p: o for p, o in zip(PLATES, objs)}
            sources = list(PLATES)
            rng.shuffle(sources)  # cover every source plate exactly once
            self._queries = [Query(s, rng.choice(PLATES)) for s in sources]
        else:  # manual
            self._mapping = self._prompt_manual_mapping()
            self._queries = self._prompt_manual_queries()

    @staticmethod
    def _prompt_manual_mapping() -> dict:
        print("[manual] Enter the plate -> object mapping. Plates: " + ", ".join(PLATES))
        m = {}
        for plate in PLATES:
            while True:
                obj = input(f"  {plate} plate has object: ").strip()
                if obj in OBJECTS:
                    m[plate] = obj
                    break
                print(f"    must be one of {OBJECTS}")
        return m

    @staticmethod
    def _prompt_manual_queries() -> list:
        print("[manual] Enter 4 queries (source_plate target_plate per line):")
        qs = []
        for i in range(4):
            while True:
                raw = input(f"  query{i+1} (e.g. 'orange green'): ").strip().split()
                if len(raw) == 2 and raw[0] in PLATES and raw[1] in PLATES:
                    qs.append(Query(raw[0], raw[1]))
                    break
                print(f"    expected two plates from {PLATES}")
        return qs

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def mapping(self) -> dict:
        return dict(self._mapping)

    @property
    def queries(self) -> list:
        return list(self._queries)

    def instruction_for_stage(self, stage_id: int) -> str:
        if stage_id == 0:
            return "Observe and remember which object is on each colored plate."
        if stage_id == 1:
            return "Move all objects to the right side of the table."
        if stage_id == 2:
            return "Wait briefly before continuing."
        if 3 <= stage_id <= 6:
            q = self._queries[stage_id - 3]
            return f"Place the object originally on the {q.source_plate} plate onto the {q.target_plate} plate."
        if stage_id == 7:
            return "Return arm to home position."
        raise ValueError(f"unknown stage_id: {stage_id}")

    def to_dict(self) -> dict:
        return {
            "mode": self._mode,
            "seed": self._seed,
            "mapping": dict(self._mapping),
            "queries": [{"source_plate": q.source_plate, "target_plate": q.target_plate}
                        for q in self._queries],
        }

    def print_setup(self) -> None:
        print("")
        print("=" * 64)
        print("PHYSICAL SETUP REQUIRED — please place:")
        for plate, obj in self._mapping.items():
            print(f"  {obj:<18s} on the {plate} plate")
        print("")
        print("Queries (in order):")
        for i, q in enumerate(self._queries, 1):
            print(f"  query{i}: {q.source_plate} -> {q.target_plate}")
        print("=" * 64)
        input("Press Enter when objects are placed correctly... ")


# Memory image dims match the LeRobot dataset (480 H x 640 W x 3 RGB uint8).
_MEM_H, _MEM_W = 480, 640


class StageManager:
    """Source of truth for stage_id, prompt, memory_image, has_memory.

    The deploy env queries this on every `get_obs()`. The keyframe used as the
    query-stage memory image is captured at the moment the operator presses `n`
    to leave stage 0 (observe).
    """

    NUM_STAGES = 8

    def __init__(self, query_plan: QueryPlan) -> None:
        self._stage_id = 0
        self._plan = query_plan
        self._zeros = np.zeros((_MEM_H, _MEM_W, 3), dtype=np.uint8)
        self._memory_image = self._zeros
        self._has_memory = False
        self._observe_keyframe: Optional[np.ndarray] = None

    @property
    def stage_id(self) -> int:
        # Once we've advanced past stage 7, callers should treat the run as done;
        # but the property still returns 7 as the displayed name (NUM_STAGES is the
        # sentinel for "past the end"). Use is_terminal to detect shutdown.
        return min(self._stage_id, self.NUM_STAGES - 1)

    @property
    def stage_name(self) -> str:
        return STAGE_NAMES[self.stage_id]

    @property
    def instruction(self) -> str:
        return self._plan.instruction_for_stage(self.stage_id)

    @property
    def memory_image(self) -> np.ndarray:
        return self._memory_image

    @property
    def has_memory(self) -> bool:
        return self._has_memory

    @property
    def is_terminal(self) -> bool:
        """True once the operator has pressed `n` past stage 7."""
        return self._stage_id >= self.NUM_STAGES

    def advance(self, current_top_frame: np.ndarray) -> bool:
        """Move to the next stage. Returns False if already past stage 7.

        If we are leaving stage 0 (observe), captures `current_top_frame` as the
        keyframe used for memory_image on stages 3..6.
        """
        if self._stage_id >= self.NUM_STAGES:
            return False

        if self._stage_id == 0:
            if current_top_frame.shape != (_MEM_H, _MEM_W, 3) or current_top_frame.dtype != np.uint8:
                raise ValueError(
                    f"observe keyframe must be ({_MEM_H},{_MEM_W},3) uint8, "
                    f"got shape={current_top_frame.shape} dtype={current_top_frame.dtype}"
                )
            self._observe_keyframe = current_top_frame.copy()

        self._stage_id += 1

        new_id = self._stage_id
        if new_id in QUERY_STAGES:
            assert self._observe_keyframe is not None, "memory keyframe missing — left observe without capturing"
            self._memory_image = self._observe_keyframe
            self._has_memory = True
        else:
            self._memory_image = self._zeros
            self._has_memory = False

        return new_id < self.NUM_STAGES or new_id == self.NUM_STAGES  # always True; terminal flips on next call


# ---------------------------------------------------------------------------
# Unit smoke checks
# ---------------------------------------------------------------------------


def _smoke() -> None:
    print("=== QueryPlan(fixed) ===")
    plan = QueryPlan(mode="fixed")
    print("mapping:", plan.mapping)
    print("queries:", plan.queries)
    for sid in range(8):
        print(f"  stage {sid} ({STAGE_NAMES[sid]}): {plan.instruction_for_stage(sid)!r}")

    # mode='fixed' should be deterministic across instantiations.
    p2 = QueryPlan(mode="fixed")
    assert plan.mapping == p2.mapping and plan.queries == p2.queries

    print("\n=== QueryPlan(random, seed=0) ===")
    rplan = QueryPlan(mode="random", seed=0)
    print("mapping:", rplan.mapping)
    print("queries:", rplan.queries)
    sources = {q.source_plate for q in rplan.queries}
    assert sources == set(PLATES), f"random queries must cover all 4 source plates, got {sources}"

    rplan2 = QueryPlan(mode="random", seed=0)
    assert rplan.mapping == rplan2.mapping and rplan.queries == rplan2.queries, "seeded random must be reproducible"

    print("\n=== StageManager walk ===")
    mgr = StageManager(plan)
    fake_top = np.full((_MEM_H, _MEM_W, 3), 200, dtype=np.uint8)
    expected = [
        # (stage_after_advance, has_memory, memory_is_keyframe)
        (1, False, False),  # observe -> mix
        (2, False, False),  # mix -> delay
        (3, True,  True),   # delay -> query1
        (4, True,  True),
        (5, True,  True),
        (6, True,  True),
        (7, False, False),  # query4 -> return_home
    ]
    for exp_id, exp_mem, exp_keyframe in expected:
        cont = mgr.advance(fake_top)
        assert cont, "advance returned False unexpectedly"
        assert mgr.stage_id == exp_id, f"got {mgr.stage_id}, expected {exp_id}"
        assert mgr.has_memory is exp_mem
        if exp_keyframe:
            assert mgr.memory_image.sum() > 0, "expected non-zero keyframe"
            assert np.array_equal(mgr.memory_image, fake_top), "memory image should be the captured keyframe"
        else:
            assert mgr.memory_image.sum() == 0, "expected zero memory image on non-query stage"
        print(f"  -> stage {mgr.stage_id} ({mgr.stage_name}): has_mem={mgr.has_memory} mem.sum={mgr.memory_image.sum()}  prompt={mgr.instruction!r}")

    # One more advance pushes past stage 7 -> terminal
    assert not mgr.is_terminal
    mgr.advance(fake_top)
    assert mgr.is_terminal, "should be terminal after advancing past stage 7"
    assert not mgr.advance(fake_top), "advance past terminal must return False"
    print("\nstage_machine smoke OK")


if __name__ == "__main__":
    _smoke()
