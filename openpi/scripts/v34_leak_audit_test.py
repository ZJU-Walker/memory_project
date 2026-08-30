"""CPU tests for scripts/v34_leak_audit.py.

Run: .venv/bin/python -m pytest scripts/v34_leak_audit_test.py -q
"""

# ruff: noqa: I001, ICN001 - the pyarrow import MUST come first (libarrow segfault trap).
import pyarrow  # noqa: F401

from pathlib import Path
import sys

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v34_leak_audit as audit
finally:
    sys.path.remove(str(_SCRIPTS_DIR))


def test_endpoint_mirror_matches_production():
    """The audit's sampler mirror must agree with the real transform on every geometry the
    dataset can produce (short waits, waits shorter than the lookahead, straddled starts)."""
    from openpi import transforms

    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(300):
        ev_start = int(rng.integers(80, 400))
        mem_lo = ev_start + int(rng.integers(60, 300))
        mem_hi = mem_lo + int(rng.integers(3, 200))
        window = np.asarray([max(1, ev_start - audit.START_PAD), ev_start, mem_lo, mem_hi], dtype=np.int32)
        for start in range(int(window[0]), int(window[1]) + 1, 7):
            expected = transforms.memory_critical_endpoint(
                start, window, stride=audit.STRIDE, lookahead=audit.LOOKAHEAD, num_steps=audit.MAX_BUCKET
            )
            actual = audit.memory_critical_endpoint_mirror(
                start, window, stride=audit.STRIDE, lookahead=audit.LOOKAHEAD, num_steps=audit.MAX_BUCKET
            )
            assert actual == expected, (start, window.tolist())
            checked += 1
    assert checked > 1000


def test_loo_probe_separable_and_null():
    rng = np.random.default_rng(1)
    # separable: side written into coordinate 0 with margin >> noise
    feats = [rng.normal(size=(20, 5)) + np.asarray([8.0 * (1 if i % 2 else -1), 0, 0, 0, 0]) for i in range(10)]
    sides = [i % 2 for i in range(10)]
    _, bal = audit.loo_ridge_probe(feats, sides)
    assert bal == 1.0
    # null: identical feature distribution for both sides -> near chance
    feats = [rng.normal(size=(20, 5)) for _ in range(10)]
    _, bal = audit.loo_ridge_probe(feats, sides)
    assert 0.2 <= bal <= 0.8


def test_loo_probe_balances_unequal_classes():
    rng = np.random.default_rng(2)
    # 8 L episodes vs 2 R episodes; balanced accuracy must not reward predicting L always.
    feats = [rng.normal(size=(15, 4)) for _ in range(10)]
    sides = [0] * 8 + [1] * 2
    accs, bal = audit.loo_ridge_probe(feats, sides)
    assert abs(bal - 0.5) < 0.35
    assert len(accs) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
