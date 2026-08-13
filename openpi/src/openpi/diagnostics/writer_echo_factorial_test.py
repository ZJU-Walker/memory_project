from openpi.diagnostics import writer_echo_factorial


def test_summarize_rows_groups_phase_and_side_with_exact_scalar_statistics():
    rows = [
        {"memory_side": "left", "phase": "pre_reveal", "score": 1.0},
        {"memory_side": "right", "phase": "visible", "score": 3.0},
        {"memory_side": "left", "phase": "post_close", "score": 5.0},
    ]
    summary = writer_echo_factorial.summarize_rows(rows, ("score",))

    assert summary["overall"]["count"] == 3
    assert summary["overall"]["score"]["mean"] == 3.0
    assert summary["left"]["score"]["median"] == 3.0
    assert summary["right"]["score"]["mean"] == 3.0
    assert summary["pre_reveal"]["score"]["max"] == 1.0
    assert summary["post_close"]["score"]["min"] == 5.0
