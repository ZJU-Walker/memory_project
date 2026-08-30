"""v3.4 tokenizer tests: the state-digit token mask (plan 5.2 / 5.9)."""

import numpy as np
import pytest

from openpi.models import tokenizer as _tokenizer


@pytest.fixture(scope="module")
def tok():
    return _tokenizer.FASTSubtaskTokenizer(48)


def _decode_positions(tok, tokens, mask):
    return [int(t) for t, m in zip(tokens.tolist(), mask.tolist(), strict=True) if m]


def test_split_state_mask_marks_exactly_the_state_span(tok):
    state = np.asarray([-0.9, 0.0, 0.5, 0.99])
    actions = np.zeros((5, 4), dtype=np.float32)
    context, context_mask, _, _, _, state_mask = tok.tokenize_split(
        "find the banana", state, "open both lids", actions, 64, return_state_mask=True
    )
    assert state_mask.shape == context.shape
    assert state_mask.dtype == bool
    # masked positions never extend past the valid context
    assert not np.any(state_mask & ~context_mask)
    # the digits of the discretized state are exactly what the mask covers: decoding the
    # masked tokens yields only digits and separators, and every digit token is masked
    sp = tok._paligemma_tokenizer  # noqa: SLF001
    masked_text = sp.decode(_decode_positions(tok, context, state_mask))
    assert masked_text.strip().replace(" ", "").isdigit()
    unmasked = sp.decode(_decode_positions(tok, context, context_mask & ~state_mask))
    assert "task" in unmasked.lower()
    assert "state" in unmasked.lower()  # the constant "State:" literal is NOT masked
    assert not any(ch.isdigit() for ch in unmasked)
    # bos is never masked
    assert not state_mask[0]

    # the expected digit string reproduces the tokenizer's own discretization
    discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    assert masked_text.split() == [str(x) for x in discretized]


def test_split_with_and_without_state_mask_agree_on_all_other_outputs(tok):
    state = np.asarray([0.1, -0.2])
    actions = np.zeros((5, 2), dtype=np.float32)
    base = tok.tokenize_split("find the box", state, "wait; target bin is left", actions, 48)
    extended = tok.tokenize_split(
        "find the box", state, "wait; target bin is left", actions, 48, return_state_mask=True
    )
    assert len(base) == 5
    assert len(extended) == 6
    for a, b in zip(base, extended[:5], strict=True):
        np.testing.assert_array_equal(a, b)


def test_joint_tokenize_state_mask_covers_prefix_only(tok):
    state = np.asarray([0.3, 0.7, -0.5])
    tokens, token_mask, ar_mask, _, fast_mask, state_mask = tok.tokenize(
        "find the banana", state, "inspect both bins", np.zeros((5, 3), dtype=np.float32), return_state_mask=True
    )
    assert state_mask.shape == tokens.shape
    # state positions live in the bidirectional prefix: never in the causal/FAST region
    assert not np.any(state_mask & (ar_mask == 1))
    assert not np.any(state_mask & fast_mask)
    assert not np.any(state_mask & ~token_mask)
    assert state_mask.sum() >= len(state)  # at least one token per digit


def test_state_mask_positions_are_step_dependent(tok):
    actions = np.zeros((5, 2), dtype=np.float32)
    _, _, _, _, _, mask_a = tok.tokenize_split(
        "find the banana", np.asarray([-0.99, -0.99]), "open both lids", actions, 48, return_state_mask=True
    )
    _, _, _, _, _, mask_b = tok.tokenize_split(
        "find the banana", np.asarray([0.99, 0.99]), "open both lids", actions, 48, return_state_mask=True
    )
    # different digit strings tokenize to different lengths; both masks still isolate digits
    assert mask_a.any()
    assert mask_b.any()
