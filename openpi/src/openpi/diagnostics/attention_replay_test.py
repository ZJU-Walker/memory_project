import numpy as np
import pytest

from openpi.diagnostics import attention_replay


class _Tokenizer:
    def __init__(self, tokens, mask):
        self._tokens = np.asarray(tokens)
        self._mask = np.asarray(mask)

    def tokenize(self, _text):
        return self._tokens, self._mask


def test_tokenize_canonical_subtask_preserves_real_padding_mask():
    tokens, mask = attention_replay._tokenize_canonical_subtask(  # noqa: SLF001
        _Tokenizer([7, 8, 9, 0, 0, 0], [True, True, True, False, False, False]), 6, "open left bin"
    )

    np.testing.assert_array_equal(tokens, [[7, 8, 9, 0, 0, 0]])
    np.testing.assert_array_equal(mask, [[True, True, True, False, False, False]])


@pytest.mark.parametrize(
    ("tokens", "mask"),
    [
        ([1, 2], [True, True]),
        ([1, 2, 3], [False, False, False]),
        ([1, 2, 3], [True, False, True]),
    ],
)
def test_tokenize_canonical_subtask_rejects_invalid_contract(tokens, mask):
    with pytest.raises(ValueError, match="tokenizer"):
        attention_replay._tokenize_canonical_subtask(  # noqa: SLF001
            _Tokenizer(tokens, mask), 3, "open right bin"
        )
