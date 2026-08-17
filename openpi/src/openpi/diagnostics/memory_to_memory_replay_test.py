import argparse

import numpy as np
import pytest

from openpi.diagnostics import memory_to_memory_replay


def test_matrix_statistics_uniform_and_diagonal_extremes():
    uniform = np.full((256, 256), 0.1 / 256, dtype=np.float32)
    stats = memory_to_memory_replay.matrix_statistics(uniform)
    assert stats["mean_memory_mass_per_query"] == pytest.approx(0.1)
    assert stats["diagonal_share_of_memory_mass"] == pytest.approx(1 / 256)
    assert stats["incoming_normalized_entropy"] == pytest.approx(1.0)
    assert stats["incoming_effective_tokens"] == pytest.approx(256)
    assert stats["effective_rank"] == pytest.approx(1.0, abs=1e-3)

    diagonal = np.eye(256, dtype=np.float32) * 0.1
    stats = memory_to_memory_replay.matrix_statistics(diagonal)
    assert stats["diagonal_share_of_memory_mass"] == pytest.approx(1.0)
    assert stats["mean_conditional_row_entropy"] == pytest.approx(0.0)
    assert stats["effective_rank"] == pytest.approx(256.0)


def test_parse_ints_rejects_duplicates_and_empty():
    assert memory_to_memory_replay._parse_ints("350, 360,380") == (350, 360, 380)  # noqa: SLF001
    with pytest.raises(argparse.ArgumentTypeError, match="distinct comma-separated"):
        memory_to_memory_replay._parse_ints("")  # noqa: SLF001
    with pytest.raises(argparse.ArgumentTypeError, match="distinct comma-separated"):
        memory_to_memory_replay._parse_ints("350,350")  # noqa: SLF001
