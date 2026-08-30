#!/usr/bin/env python3
"""Fail-closed numerical watcher for an isolated v34 training Slurm step."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

_STEP_LINE = re.compile(r"^Step (?P<step>\d+): (?P<body>.*)$")
_STEP_ID = re.compile(r"^\d+\.\d+$")
_TERMINAL_SLURM_STATES = frozenset(
    {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "TIMEOUT",
    }
)
_REQUIRED = (
    "loss",
    "memory_grad_norm",
    "diagnostic/write_inner_grad_norm",
    "diagnostic/write_inner_grad_max",
    "diagnostic/write_inner_clip_fraction",
    "diagnostic/write_inner_severe_clip_fraction",
    "grad_norm",
)


@dataclass(frozen=True)
class PolicyThresholds:
    max_write_mean: float = 3.0
    max_memory_grad: float = 5.0
    hard_severe_fraction: float = 0.075
    marginal_severe_fraction: float = 0.05
    joint_write_mean: float = 2.5
    joint_memory_grad: float = 1.0
    joint_total_grad: float = 10.0
    warn_severe_fraction: float = 0.03
    warn_write_mean: float = 2.3
    warn_clip_fraction: float = 0.80
    warn_write_max: float = 10.0
    trend_history_windows: int = 5


@dataclass(frozen=True)
class PolicyHistory:
    """Prior eligible windows used by the stateful policy.

    Five prior write-mean windows is the conservative minimum for a trailing-median warning:
    it avoids declaring a trend from one or two noisy startup samples while remaining responsive.
    """

    previous_severe_fraction: float | None = None
    trailing_write_means: tuple[float, ...] = ()


@dataclass(frozen=True)
class PolicyEvaluation:
    violations: tuple[str, ...]
    warnings: tuple[str, ...]
    next_history: PolicyHistory


class TelemetryParseError(ValueError):
    """A Step record whose identity is known but whose metric payload is ambiguous."""

    def __init__(self, *, step: int, line: str, reason: str):
        self.step = step
        self.line = line
        self.reason = reason
        super().__init__(f"step={step}: {reason}; line={line!r}")


def validate_policy_thresholds(thresholds: PolicyThresholds) -> None:
    """Reject unsafe or contradictory policy configuration before watching a job."""

    numeric_thresholds = {name: value for name, value in vars(thresholds).items() if name != "trend_history_windows"}
    invalid = [name for name, value in numeric_thresholds.items() if not math.isfinite(value) or value < 0]
    if invalid:
        raise ValueError(f"Policy thresholds must be finite and nonnegative: {', '.join(invalid)}")

    fractions = {
        "hard_severe_fraction": thresholds.hard_severe_fraction,
        "marginal_severe_fraction": thresholds.marginal_severe_fraction,
        "warn_severe_fraction": thresholds.warn_severe_fraction,
        "warn_clip_fraction": thresholds.warn_clip_fraction,
    }
    above_one = [name for name, value in fractions.items() if value > 1]
    if above_one:
        raise ValueError(f"Fraction thresholds must be at most 1: {', '.join(above_one)}")
    if not (thresholds.warn_severe_fraction <= thresholds.marginal_severe_fraction < thresholds.hard_severe_fraction):
        raise ValueError("Required severity ordering is warn <= marginal < hard")
    if not thresholds.warn_write_mean <= thresholds.joint_write_mean <= thresholds.max_write_mean:
        raise ValueError("Required write-mean ordering is warn <= joint <= hard")
    if thresholds.joint_memory_grad > thresholds.max_memory_grad:
        raise ValueError("Joint memory-grad threshold must not exceed its hard threshold")
    if thresholds.trend_history_windows < 1:
        raise ValueError("trend_history_windows must be at least 1")


def parse_step_line(line: str) -> tuple[int, dict[str, float]] | None:
    stripped = line.strip()
    match = _STEP_LINE.match(stripped)
    if match is None:
        return None
    step = int(match.group("step"))
    metrics: dict[str, float] = {}
    for field in match.group("body").split(", "):
        key, separator, value = field.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or not key or not value:
            raise TelemetryParseError(step=step, line=stripped, reason=f"malformed metric field {field!r}")
        if key in metrics:
            raise TelemetryParseError(step=step, line=stripped, reason=f"duplicate metric key {key!r}")
        try:
            metrics[key] = float(value)
        except ValueError as error:
            raise TelemetryParseError(
                step=step,
                line=stripped,
                reason=f"non-numeric metric {key}={value!r}",
            ) from error
    return step, metrics


def evaluate_policy(
    step: int,
    metrics: dict[str, float],
    *,
    min_gate_step: int,
    history: PolicyHistory,
    thresholds: PolicyThresholds,
) -> PolicyEvaluation:
    """Evaluate one logged window without side effects and return the updated policy history."""

    violations = [f"nonfinite {key}={value}" for key, value in metrics.items() if not math.isfinite(value)]
    if step < min_gate_step:
        # Threshold policy, including its history, starts at the gate. Nonfinite telemetry remains
        # fail-closed before the gate, matching the original watcher semantics.
        return PolicyEvaluation(tuple(violations), (), history)

    missing = [key for key in _REQUIRED if key not in metrics]
    if missing:
        violations.append(f"missing required telemetry: {', '.join(missing)}")
        return PolicyEvaluation(tuple(violations), (), history)
    if violations:
        return PolicyEvaluation(tuple(violations), (), history)

    write_mean = metrics["diagnostic/write_inner_grad_norm"]
    memory_grad = metrics["memory_grad_norm"]
    severe = metrics["diagnostic/write_inner_severe_clip_fraction"]
    total_grad = metrics["grad_norm"]
    ordinary_clip = metrics["diagnostic/write_inner_clip_fraction"]
    write_max = metrics["diagnostic/write_inner_grad_max"]

    if write_mean > thresholds.max_write_mean:
        violations.append(f"diagnostic/write_inner_grad_norm={write_mean:.6g} > {thresholds.max_write_mean:.6g}")
    if memory_grad > thresholds.max_memory_grad:
        violations.append(f"memory_grad_norm={memory_grad:.6g} > {thresholds.max_memory_grad:.6g}")
    if severe >= thresholds.hard_severe_fraction:
        violations.append(
            f"diagnostic/write_inner_severe_clip_fraction={severe:.6g} >= {thresholds.hard_severe_fraction:.6g}"
        )
    elif severe >= thresholds.marginal_severe_fraction:
        previous_severe = history.previous_severe_fraction
        if previous_severe is not None and previous_severe >= thresholds.marginal_severe_fraction:
            violations.append(
                f"consecutive marginal severe-clip windows: previous={previous_severe:.6g}, current={severe:.6g}"
            )
        if write_mean >= thresholds.joint_write_mean:
            violations.append(
                f"marginal severe clip corroborated by write mean {write_mean:.6g} >= {thresholds.joint_write_mean:.6g}"
            )
        if memory_grad >= thresholds.joint_memory_grad:
            violations.append(
                f"marginal severe clip corroborated by memory grad {memory_grad:.6g} "
                f">= {thresholds.joint_memory_grad:.6g}"
            )
        if total_grad >= thresholds.joint_total_grad:
            violations.append(
                f"marginal severe clip corroborated by total grad {total_grad:.6g} >= {thresholds.joint_total_grad:.6g}"
            )

    warnings: list[str] = []
    if severe >= thresholds.warn_severe_fraction:
        warnings.append(f"diagnostic/write_inner_severe_clip_fraction={severe:.6g} warning")
    if write_mean >= thresholds.warn_write_mean:
        warnings.append(f"diagnostic/write_inner_grad_norm={write_mean:.6g} warning")
    if ordinary_clip >= thresholds.warn_clip_fraction:
        warnings.append(f"diagnostic/write_inner_clip_fraction={ordinary_clip:.6g} warning")
    if write_max > thresholds.warn_write_max:
        warnings.append(f"diagnostic/write_inner_grad_max={write_max:.6g} > warning {thresholds.warn_write_max:.6g}")

    trailing = history.trailing_write_means[-thresholds.trend_history_windows :]
    if len(trailing) >= thresholds.trend_history_windows:
        trailing_median = statistics.median(trailing)
        if write_mean > 2 * trailing_median:
            warnings.append(
                f"diagnostic/write_inner_grad_norm={write_mean:.6g} > "
                f"2x trailing-{thresholds.trend_history_windows} median {trailing_median:.6g}"
            )

    next_means = (*trailing, write_mean)[-thresholds.trend_history_windows :]
    next_history = PolicyHistory(previous_severe_fraction=severe, trailing_write_means=next_means)
    return PolicyEvaluation(tuple(violations), tuple(warnings), next_history)


def _canonical_slurm_state(value: str) -> str:
    """Normalize Slurm suffixes and annotations such as ``FAILED+`` and ``CANCELLED by``."""

    base = value.strip().split("+", maxsplit=1)[0]
    return base.split(maxsplit=1)[0].upper() if base else ""


def _exact_accounting_state(step_id: str) -> str | None:
    """Return the state of one exact step, or ``None`` when accounting is inconclusive."""

    for allocations_only in (True, False):
        command = ["sacct"]
        if allocations_only:
            command.append("-X")
        command.extend(
            [
                "--noheader",
                "--parsable2",
                "--jobs",
                step_id,
                "--format=JobID,State",
            ]
        )
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True)
        except OSError:
            continue
        if result.returncode != 0:
            continue
        for row in result.stdout.splitlines():
            fields = row.split("|")
            if len(fields) < 2 or fields[0].strip() != step_id:
                continue
            state = _canonical_slurm_state(fields[1])
            return state or None
    return None


def slurm_step_active(step_id: str) -> bool | None:
    """Return whether an exact Slurm step is active, with ``None`` for scheduler uncertainty.

    Absence from a successful ``squeue`` response is not enough to establish completion: the
    queue-to-accounting transition can be transient.  Only an exact terminal ``sacct`` row returns
    ``False``.  Query failures and missing accounting rows return ``None`` so the watcher keeps
    monitoring instead of silently losing coverage.
    """

    parent_job = step_id.partition(".")[0]
    try:
        result = subprocess.run(
            [
                "squeue",
                "--steps",
                "--noheader",
                "--jobs",
                parent_job,
                "--format=%i",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    if any(line.strip() == step_id for line in result.stdout.splitlines()):
        return True

    accounting_state = _exact_accounting_state(step_id)
    if accounting_state is None:
        return None
    return accounting_state not in _TERMINAL_SLURM_STATES


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-log", type=Path, required=True)
    parser.add_argument("--audit-log", type=Path, required=True)
    parser.add_argument("--step-id", required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--min-gate-step", type=int, default=100)
    parser.add_argument("--max-write-mean", type=float, default=3.0)
    parser.add_argument("--warn-write-max", type=float, default=10.0)
    parser.add_argument("--max-severe-fraction", type=float, default=0.075)
    parser.add_argument("--max-memory-grad", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if not _STEP_ID.fullmatch(args.step_id):
        raise ValueError(f"Refusing non-step Slurm target: {args.step_id!r}")
    thresholds = PolicyThresholds(
        max_write_mean=args.max_write_mean,
        max_memory_grad=args.max_memory_grad,
        hard_severe_fraction=args.max_severe_fraction,
        warn_write_max=args.warn_write_max,
    )
    validate_policy_thresholds(thresholds)
    args.audit_log.parent.mkdir(parents=True, exist_ok=True)

    def emit(message: str) -> None:
        timestamped = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
        print(timestamped, flush=True)
        with args.audit_log.open("a", encoding="utf-8") as audit:
            audit.write(timestamped + "\n")

    emit(
        "watcher started "
        f"target={args.step_id} min_step={args.min_gate_step} "
        f"mean_hard>{thresholds.max_write_mean} memory_hard>{thresholds.max_memory_grad} "
        f"severe_hard>={thresholds.hard_severe_fraction} "
        f"severe_marginal>={thresholds.marginal_severe_fraction} consecutive=2 "
        f"joint(mean>={thresholds.joint_write_mean},memory>={thresholds.joint_memory_grad},"
        f"total>={thresholds.joint_total_grad}) "
        f"warnings(severe>={thresholds.warn_severe_fraction},mean>={thresholds.warn_write_mean},"
        f"clip>={thresholds.warn_clip_fraction},max>{thresholds.warn_write_max},"
        f"trend=2x/{thresholds.trend_history_windows})"
    )
    offset = 0
    last_step = -1
    history = PolicyHistory()
    scheduler_uncertain = False
    cancellation_pending = False
    while True:
        if not cancellation_pending and args.metrics_log.exists():
            with args.metrics_log.open(encoding="utf-8") as source:
                source.seek(offset)
                lines = source.readlines()
                offset = source.tell()
            for line in lines:
                try:
                    parsed = parse_step_line(line)
                except TelemetryParseError as error:
                    if error.step <= last_step:
                        continue
                    last_step = error.step
                    emit(
                        f"step={error.step} malformed telemetry reason={error.reason!r} "
                        f"line={error.line!r} violations=('malformed telemetry',)"
                    )
                    if args.dry_run:
                        emit("DRY RUN: would cancel exact Slurm step " + args.step_id)
                        return 2
                    cancellation_pending = True
                    break
                if parsed is None:
                    continue
                step, metrics = parsed
                if step <= last_step:
                    continue
                last_step = step
                evaluation = evaluate_policy(
                    step,
                    metrics,
                    min_gate_step=args.min_gate_step,
                    history=history,
                    thresholds=thresholds,
                )
                history = evaluation.next_history
                emit(
                    f"step={step} mean={metrics.get('diagnostic/write_inner_grad_norm')} "
                    f"max={metrics.get('diagnostic/write_inner_grad_max')} "
                    f"severe={metrics.get('diagnostic/write_inner_severe_clip_fraction')} "
                    f"memory_grad={metrics.get('memory_grad_norm')} warnings={evaluation.warnings or 'none'} "
                    f"violations={evaluation.violations or 'none'}"
                )
                if evaluation.violations:
                    if args.dry_run:
                        emit("DRY RUN: would cancel exact Slurm step " + args.step_id)
                        return 2
                    cancellation_pending = True
                    break
        if cancellation_pending:
            try:
                result = subprocess.run(
                    ["scancel", args.step_id],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            except OSError as error:
                emit(f"cancel attempt failed target={args.step_id} error={error!r}; will retry")
            else:
                emit(
                    f"cancel requested target={args.step_id} returncode={result.returncode} "
                    f"stderr={result.stderr.strip()!r}"
                )
                if result.returncode == 0:
                    return 2
        if args.once and not cancellation_pending:
            emit(f"one-shot audit complete last_step={last_step}")
            return 0
        step_active = slurm_step_active(args.step_id)
        if step_active is None:
            if not scheduler_uncertain:
                emit("scheduler status uncertain; watcher remains active target=" + args.step_id)
            scheduler_uncertain = True
            time.sleep(args.poll_seconds)
            continue
        if scheduler_uncertain:
            emit("scheduler status recovered target=" + args.step_id)
            scheduler_uncertain = False
        if step_active is False:
            if cancellation_pending:
                emit(f"target step already terminal after policy violation; watcher exiting last_step={last_step}")
                return 2
            # One last pass catches a final line that raced the queue transition.
            if args.metrics_log.exists() and args.metrics_log.stat().st_size > offset:
                continue
            emit(f"target step terminal in accounting; watcher exiting last_step={last_step}")
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
