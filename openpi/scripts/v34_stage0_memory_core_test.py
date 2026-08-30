import sys

import pytest

from scripts import v34_stage0_memory_core as stage0


def test_eta_cli_reaches_every_battery_and_is_reported(monkeypatch, capsys):
    observed = {}

    def battery(name):
        def run(_memory, *, eta_gate):
            observed[name] = eta_gate
            return True

        return run

    for name in (
        "battery_repeated_association",
        "battery_near_orthogonal",
        "battery_rank1_stress",
        "battery_conflicting_and_degenerate_inputs",
    ):
        monkeypatch.setattr(stage0, name, battery(name))
    monkeypatch.setattr(sys, "argv", ["v34_stage0_memory_core.py", "--eta", "0"])

    with pytest.raises(SystemExit) as exit_info:
        stage0.main()

    assert exit_info.value.code == 0
    assert observed == {
        "battery_repeated_association": 0.0,
        "battery_near_orthogonal": 0.0,
        "battery_rank1_stress": 0.0,
        "battery_conflicting_and_degenerate_inputs": 0.0,
    }
    assert "gates=(theta=0.1, eta=0.0, alpha=0.01)" in capsys.readouterr().out
