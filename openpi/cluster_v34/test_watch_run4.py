from __future__ import annotations

from dataclasses import replace
import importlib.util
import math
import pathlib
import subprocess
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).with_name("watch_run4.py")
_SPEC = importlib.util.spec_from_file_location("watch_run4", _SCRIPT)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

PolicyHistory = _MODULE.PolicyHistory
PolicyThresholds = _MODULE.PolicyThresholds
TelemetryParseError = _MODULE.TelemetryParseError
evaluate_policy = _MODULE.evaluate_policy
parse_step_line = _MODULE.parse_step_line
validate_policy_thresholds = _MODULE.validate_policy_thresholds


STEP_2300 = (
    "Step 2300: loss=1.0144, grad_norm=1.4544, memory_grad_norm=0.2314, "
    "diagnostic/write_inner_grad_norm=2.1003, diagnostic/write_inner_grad_max=14.5752, "
    "diagnostic/write_inner_clip_fraction=0.7741, "
    "diagnostic/write_inner_severe_clip_fraction=0.0198"
)
STEP_2400 = (
    "Step 2400: loss=1.0305, grad_norm=1.8459, memory_grad_norm=0.4701, "
    "diagnostic/write_inner_grad_norm=2.3794, diagnostic/write_inner_grad_max=23.5724, "
    "diagnostic/write_inner_clip_fraction=0.8020, "
    "diagnostic/write_inner_severe_clip_fraction=0.0521"
)


def _metrics(**overrides: float) -> dict[str, float]:
    metrics = {
        "loss": 1.0,
        "grad_norm": 2.0,
        "memory_grad_norm": 0.5,
        "diagnostic/write_inner_grad_norm": 2.0,
        "diagnostic/write_inner_grad_max": 8.0,
        "diagnostic/write_inner_clip_fraction": 0.5,
        "diagnostic/write_inner_severe_clip_fraction": 0.01,
    }
    metrics.update(overrides)
    return metrics


def _evaluate(metrics: dict[str, float], history: PolicyHistory | None = None):
    history = PolicyHistory() if history is None else history
    return evaluate_policy(
        100,
        metrics,
        min_gate_step=100,
        history=history,
        thresholds=PolicyThresholds(),
    )


def test_historical_step_2400_is_warning_not_cancellation() -> None:
    history = PolicyHistory()
    for line in (STEP_2300, STEP_2400):
        parsed = parse_step_line(line)
        assert parsed is not None
        step, metrics = parsed
        evaluation = evaluate_policy(
            step,
            metrics,
            min_gate_step=100,
            history=history,
            thresholds=PolicyThresholds(),
        )
        history = evaluation.next_history

    assert not evaluation.violations
    assert any("severe_clip_fraction" in warning for warning in evaluation.warnings)
    assert any("write_inner_grad_norm" in warning for warning in evaluation.warnings)
    assert any("write_inner_clip_fraction" in warning for warning in evaluation.warnings)


def test_second_consecutive_marginal_severe_window_cancels() -> None:
    first = _evaluate(_metrics(**{"diagnostic/write_inner_severe_clip_fraction": 0.0521}))
    second = _evaluate(
        _metrics(**{"diagnostic/write_inner_severe_clip_fraction": 0.051}),
        history=first.next_history,
    )

    assert any("consecutive marginal" in violation for violation in second.violations)


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("diagnostic/write_inner_grad_norm", 2.5),
        ("memory_grad_norm", 1.0),
        ("grad_norm", 10.0),
    ],
)
def test_marginal_severe_with_joint_metric_cancels(metric: str, value: float) -> None:
    evaluation = _evaluate(_metrics(**{"diagnostic/write_inner_severe_clip_fraction": 0.05, metric: value}))

    assert any("corroborated" in violation for violation in evaluation.violations)


def test_hard_severe_threshold_cancels() -> None:
    evaluation = _evaluate(_metrics(**{"diagnostic/write_inner_severe_clip_fraction": 0.075}))

    assert evaluation.violations


def test_nonfinite_metric_cancels() -> None:
    evaluation = _evaluate(_metrics(loss=math.nan))

    assert any("nonfinite loss" in violation for violation in evaluation.violations)


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("Step 100: loss=garbage", "non-numeric metric loss='garbage'"),
        ("Step 101: loss=1.0, broken-field", "malformed metric field 'broken-field'"),
        ("Step 102: loss=1.0, loss=2.0", "duplicate metric key 'loss'"),
    ],
)
def test_malformed_step_telemetry_preserves_step_line_and_reason(line: str, reason: str) -> None:
    with pytest.raises(TelemetryParseError) as error_info:
        parse_step_line(line)

    error = error_info.value
    assert error.step == int(line.split()[1].rstrip(":"))
    assert error.line == line
    assert error.reason == reason


def test_missing_required_metric_cancels() -> None:
    metrics = _metrics()
    del metrics["memory_grad_norm"]

    evaluation = _evaluate(metrics)

    assert any("missing required telemetry" in violation for violation in evaluation.violations)


def test_write_max_alone_only_warns() -> None:
    evaluation = _evaluate(_metrics(**{"diagnostic/write_inner_grad_max": 10.1}))

    assert not evaluation.violations
    assert evaluation.warnings == ("diagnostic/write_inner_grad_max=10.1 > warning 10",)


def test_trend_warning_requires_five_prior_windows() -> None:
    history = PolicyHistory()
    for _ in range(4):
        history = _evaluate(_metrics(**{"diagnostic/write_inner_grad_norm": 1.0}), history).next_history

    too_early = _evaluate(_metrics(**{"diagnostic/write_inner_grad_norm": 2.1}), history)
    assert not any("trailing-5 median" in warning for warning in too_early.warnings)

    history = _evaluate(_metrics(**{"diagnostic/write_inner_grad_norm": 1.0}), history).next_history
    ready = _evaluate(_metrics(**{"diagnostic/write_inner_grad_norm": 2.1}), history)
    assert any("trailing-5 median" in warning for warning in ready.warnings)


@pytest.mark.parametrize(
    ("metric", "safe_value", "unsafe_value"),
    [
        ("diagnostic/write_inner_grad_norm", 3.0, 3.0001),
        ("memory_grad_norm", 5.0, 5.0001),
    ],
)
def test_immediate_metric_gates_are_strictly_greater(
    metric: str,
    safe_value: float,
    unsafe_value: float,
) -> None:
    assert not _evaluate(_metrics(**{metric: safe_value})).violations
    assert _evaluate(_metrics(**{metric: unsafe_value})).violations


@pytest.mark.parametrize(
    "thresholds",
    [
        replace(PolicyThresholds(), max_write_mean=-1.0),
        replace(PolicyThresholds(), hard_severe_fraction=1.1),
        replace(PolicyThresholds(), hard_severe_fraction=0.05),
        replace(PolicyThresholds(), hard_severe_fraction=0.04),
        replace(PolicyThresholds(), joint_write_mean=3.1),
        replace(PolicyThresholds(), joint_memory_grad=5.1),
        replace(PolicyThresholds(), trend_history_windows=0),
    ],
)
def test_invalid_policy_thresholds_are_rejected(thresholds: PolicyThresholds) -> None:
    with pytest.raises(ValueError, match="[Tt]hreshold|ordering|trend"):
        validate_policy_thresholds(thresholds)


def _completed_process(
    args: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


def test_nonzero_squeue_is_scheduler_uncertainty_not_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return _completed_process(args, returncode=1, stderr="temporary controller failure")

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)

    assert _MODULE.slurm_step_active("17024084.61") is None
    assert calls == [
        [
            "squeue",
            "--steps",
            "--noheader",
            "--jobs",
            "17024084",
            "--format=%i",
        ]
    ]


def test_unavailable_squeue_is_scheduler_uncertainty_not_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        raise OSError(f"cannot execute {args[0]}")

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)

    assert _MODULE.slurm_step_active("17024084.61") is None


def test_exact_step_present_in_squeue_is_active(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert args[0] == "squeue"
        return _completed_process(args, stdout="17024084.60\n17024084.61\n")

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)

    assert _MODULE.slurm_step_active("17024084.61") is True


@pytest.mark.parametrize("terminal_state", ["COMPLETED", "FAILED+", "CANCELLED by 24706"])
def test_empty_successful_squeue_exits_only_for_exact_terminal_accounting_row(
    monkeypatch: pytest.MonkeyPatch,
    terminal_state: str,
) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        if args[0] == "squeue":
            return _completed_process(args)
        return _completed_process(args, stdout=f"17024084.61|{terminal_state}\n")

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)

    assert _MODULE.slurm_step_active("17024084.61") is False
    assert [call[0] for call in calls] == ["squeue", "sacct"]


@pytest.mark.parametrize(
    "accounting_outputs",
    [
        ("", ""),
        ("17024084.60|COMPLETED\n", "17024084.60|COMPLETED\n"),
    ],
)
def test_empty_successful_squeue_with_unknown_accounting_keeps_monitoring(
    monkeypatch: pytest.MonkeyPatch,
    accounting_outputs: tuple[str, str],
) -> None:
    accounting_iter = iter(accounting_outputs)

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if args[0] == "squeue":
            return _completed_process(args)
        return _completed_process(args, stdout=next(accounting_iter))

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)

    assert _MODULE.slurm_step_active("17024084.61") is None


def test_empty_successful_squeue_with_nonterminal_accounting_keeps_monitoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if args[0] == "squeue":
            return _completed_process(args)
        return _completed_process(args, stdout="17024084.61|RUNNING\n")

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)

    assert _MODULE.slurm_step_active("17024084.61") is True


def test_empty_successful_squeue_with_failed_accounting_keeps_monitoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        if args[0] == "squeue":
            return _completed_process(args)
        return _completed_process(args, returncode=1, stderr="accounting temporarily unavailable")

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)

    assert _MODULE.slurm_step_active("17024084.61") is None


def test_main_does_not_exit_during_transient_scheduler_uncertainty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    observations = iter((None, False))
    observed_targets: list[str] = []

    def fake_step_active(step_id: str) -> bool | None:
        observed_targets.append(step_id)
        return next(observations)

    audit_log = tmp_path / "audit.log"
    monkeypatch.setattr(_MODULE, "slurm_step_active", fake_step_active)
    monkeypatch.setattr(_MODULE.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT),
            "--metrics-log",
            str(tmp_path / "missing-metrics.log"),
            "--audit-log",
            str(audit_log),
            "--step-id",
            "17024084.61",
        ],
    )

    assert _MODULE.main() == 0
    assert observed_targets == ["17024084.61", "17024084.61"]
    audit = audit_log.read_text(encoding="utf-8")
    assert "scheduler status uncertain; watcher remains active target=17024084.61" in audit
    assert "scheduler status recovered target=17024084.61" in audit
    assert "target step terminal in accounting" in audit


def test_policy_violation_cancels_only_the_exact_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    metrics = _metrics(**{"diagnostic/write_inner_severe_clip_fraction": 0.075})
    metrics_log = tmp_path / "metrics.log"
    metrics_log.write_text(
        "Step 100: " + ", ".join(f"{key}={value}" for key, value in metrics.items()) + "\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return _completed_process(args)

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT),
            "--metrics-log",
            str(metrics_log),
            "--audit-log",
            str(tmp_path / "audit.log"),
            "--step-id",
            "17024084.61",
        ],
    )

    assert _MODULE.main() == 2
    assert calls == [["scancel", "17024084.61"]]


@pytest.mark.parametrize(
    "line",
    [
        "Step 100: loss=nan",
        "Step 100: loss=garbage",
        "Step 100: loss=1.0, broken-field",
    ],
)
def test_nonfinite_or_malformed_step_line_cancels_only_the_exact_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    line: str,
) -> None:
    metrics_log = tmp_path / "metrics.log"
    metrics_log.write_text(line + "\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return _completed_process(args)

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT),
            "--metrics-log",
            str(metrics_log),
            "--audit-log",
            str(tmp_path / "audit.log"),
            "--step-id",
            "17024084.61",
        ],
    )

    assert _MODULE.main() == 2
    assert calls == [["scancel", "17024084.61"]]
    audit = (tmp_path / "audit.log").read_text(encoding="utf-8")
    assert "step=100" in audit
    if "nan" in line:
        assert "nonfinite loss=nan" in audit
    else:
        assert "malformed telemetry" in audit
        assert f"line={line!r}" in audit


def _run_retrying_cancel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
    cancel_results: list[subprocess.CompletedProcess[str] | OSError],
    *,
    active_results: tuple[bool | None, ...] = (True,),
    once: bool = False,
) -> tuple[int, list[list[str]]]:
    metrics = _metrics(**{"diagnostic/write_inner_severe_clip_fraction": 0.075})
    metrics_log = tmp_path / "metrics.log"
    metrics_log.write_text(
        "Step 100: " + ", ".join(f"{key}={value}" for key, value in metrics.items()) + "\n",
        encoding="utf-8",
    )
    calls: list[list[str]] = []
    cancel_iter = iter(cancel_results)
    active_iter = iter(active_results)

    def fake_run(args: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        outcome = next(cancel_iter)
        if isinstance(outcome, OSError):
            raise outcome
        return outcome

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(_MODULE, "slurm_step_active", lambda _step_id: next(active_iter))
    monkeypatch.setattr(_MODULE.time, "sleep", lambda _: None)
    argv = [
        str(_SCRIPT),
        "--metrics-log",
        str(metrics_log),
        "--audit-log",
        str(tmp_path / "audit.log"),
        "--step-id",
        "17024084.61",
    ]
    if once:
        argv.append("--once")
    monkeypatch.setattr(sys, "argv", argv)
    return _MODULE.main(), calls


def test_nonzero_scancel_retries_the_same_exact_step_until_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    failed = _completed_process(["scancel", "17024084.61"], returncode=1, stderr="controller unavailable")
    succeeded = _completed_process(["scancel", "17024084.61"])

    result, calls = _run_retrying_cancel(monkeypatch, tmp_path, [failed, succeeded])

    assert result == 2
    assert calls == [["scancel", "17024084.61"], ["scancel", "17024084.61"]]
    audit = (tmp_path / "audit.log").read_text(encoding="utf-8")
    assert "returncode=1" in audit
    assert "returncode=0" in audit


def test_scancel_oserror_retries_the_same_exact_step_until_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    succeeded = _completed_process(["scancel", "17024084.61"])

    result, calls = _run_retrying_cancel(
        monkeypatch,
        tmp_path,
        [OSError("scancel unavailable"), succeeded],
    )

    assert result == 2
    assert calls == [["scancel", "17024084.61"], ["scancel", "17024084.61"]]
    assert "cancel attempt failed target=17024084.61" in (tmp_path / "audit.log").read_text(encoding="utf-8")


def test_once_mode_does_not_exit_after_failed_scancel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    failed = _completed_process(["scancel", "17024084.61"], returncode=1)
    succeeded = _completed_process(["scancel", "17024084.61"])

    result, calls = _run_retrying_cancel(
        monkeypatch,
        tmp_path,
        [failed, succeeded],
        once=True,
    )

    assert result == 2
    assert calls == [["scancel", "17024084.61"], ["scancel", "17024084.61"]]


def test_failed_scancel_exits_as_violation_when_exact_step_is_already_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    failed = _completed_process(["scancel", "17024084.61"], returncode=1, stderr="already completing")

    result, calls = _run_retrying_cancel(
        monkeypatch,
        tmp_path,
        [failed],
        active_results=(False,),
    )

    assert result == 2
    assert calls == [["scancel", "17024084.61"]]
    assert "target step already terminal after policy violation" in (tmp_path / "audit.log").read_text(
        encoding="utf-8"
    )


def test_main_rejects_parent_job_id(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(_SCRIPT),
            "--metrics-log",
            str(tmp_path / "metrics.log"),
            "--audit-log",
            str(tmp_path / "audit.log"),
            "--step-id",
            "17024084",
        ],
    )

    with pytest.raises(ValueError, match="Refusing non-step Slurm target"):
        _MODULE.main()
