"""Do the v3.3 write tokens ENCODE the target side, and does the instruction steer that code?

This is a diagnostic-only probe -- no training loss, no gradient reaches the model. It asks two
questions that the offline eval (``scripts/v33_offline_eval.py``) cannot separate:

1. CONTENT.  Given the write tokens ``W_t in R^{16 x d}`` at evidence/wait frames, can a linear
   readout recover the episode's target side?  ``W_t -> {left, right}``.
2. CONDITIONING.  Re-encode the SAME visual frames under the counterfactual instruction.  If the
   writer is instruction-conditioned, ``W_t(banana)`` and ``W_t(box)`` should encode different
   sides whenever the two instructions imply different bins.

The offline eval showed the policy ignores the read; that is compatible with either "the memory
never encoded the answer" or "it encoded it but the policy never consulted it".  This module
discriminates: a high held-out probe AUC with a dead policy means the write path works and the
READ path is the broken link; a chance AUC means the answer was never written down.

METHODOLOGY -- the whole result hinges on the split, so it is enforced, not optional:

* Leave-one-EPISODE-out.  Frames inside one episode share lighting, object placement and arm
  pose, all of which correlate with side; a frame-level split lets the probe memorize episode
  identity and report a meaningless ~1.0.  Every fold holds out an entire episode.
* One vector per episode per phase (frames are averaged) so an episode contributes a single
  training point and long episodes cannot dominate.
* Reported against three controls, because "AUC 0.62" is uninterpretable on its own:
  - ``shuffled``: identical pipeline, labels permuted across episodes -> the null this
    estimator produces at this sample size;
  - ``read``: the same probe on the RETRIEVED tokens (what the policy actually consumes);
  - ``state``: the same probe on the raw proprioceptive state, the known leakage channel.
  A write-token AUC that does not clear the ``state`` baseline is not evidence of memory.

Outputs ``probe.json`` (all folds, all controls, per-phase) plus a printed table.
"""

# ruff: noqa: SLF001, I001 - deliberately reuses the v3.2/v3.3 replay machinery; the
# pyarrow-before-openpi import order is load-bearing (see v33_writer_attention for why).

from __future__ import annotations

import pyarrow.parquet as pq

import dataclasses
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from openpi.diagnostics import v33_writer_attention as _wa
from openpi.diagnostics import writer_contribution as _wc

SCHEMA_VERSION = "openpi.v33.write_token_probe.v1"
_EPS = 1e-12

# Phases whose tokens are worth probing. "evidence" is where the answer is observable;
# "waiting" is where it must survive; "approach" precedes the reveal and is the internal
# negative control -- a probe that succeeds there is reading a nuisance, not the evidence.
PROBE_PHASES = ("approach", "evidence", "retention", "waiting")

# Token streams to probe. "write" is the question; the rest calibrate it.
PROBE_STREAMS = ("write", "read", "state")


@dataclasses.dataclass(frozen=True)
class Options:
    checkpoint: Path
    dataset_root: Path
    output_dir: Path
    config: str = "pi05_yam_mem_v33"
    episode_indices: tuple[int, ...] = ()
    stride: int | None = None
    # Sampling cadence for the probe, decoupled from the WRITE cadence above. At the training
    # stride of 15 the two phases that carry the question are the thinnest: evidence averages
    # 3.5 frames per episode (min 2) and waiting 4.2 (min 1), so a per-episode mean is noisy.
    # Sampling every `sample_stride` frames raises those to ~10.4 and ~12.8 at stride 5. The
    # memory state is still advanced ONLY on the write grid, so retrieval stays on-distribution;
    # writing 3x more often than training would put the fast weights somewhere the model never
    # sees. None = sample exactly on the write grid.
    sample_stride: int | None = None
    max_episodes: int = 0  # 0 = all usable episodes
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("checkpoint", "dataset_root", "output_dir"):
            object.__setattr__(self, name, Path(getattr(self, name)).expanduser().resolve())
        object.__setattr__(self, "episode_indices", tuple(self.episode_indices))
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be positive")
        if self.sample_stride is not None and self.sample_stride <= 0:
            raise ValueError("sample_stride must be positive")
        if self.max_episodes < 0:
            raise ValueError("max_episodes must be non-negative")


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score using TRAIN statistics only; constant features collapse to zero."""
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (train - mean) / scale, (test - mean) / scale


def fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 1.0,
    steps: int = 400,
    lr: float = 0.5,
) -> np.ndarray:
    """Ridge-regularized logistic regression by full-batch gradient descent.

    Deliberately dependency-free (no sklearn) and deterministic. Returns ``[d + 1]`` weights
    with the bias last. The strong default L2 matters: with ~60 samples and d in the tens of
    thousands, an unregularized fit separates ANY labeling perfectly.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(f"bad probe shapes: features {x.shape}, labels {y.shape}")
    n, d = x.shape
    design = np.concatenate([x, np.ones((n, 1))], axis=1)
    weights = np.zeros(d + 1)
    for _ in range(steps):
        logits = design @ weights
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        grad = design.T @ (preds - y) / n
        grad[:-1] += l2 * weights[:-1] / n  # no penalty on the bias
        weights -= lr * grad
    return weights


def predict_logit(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    return x @ weights[:-1] + weights[-1]


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC with tie correction. Returns nan if either class is absent."""
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    # average ranks within tied score groups
    sorted_scores = s[order]
    start = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def leave_one_out_probe(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float = 1.0,
) -> dict[str, Any]:
    """Leave-one-EPISODE-out probe. Each row must be one episode.

    Returns held-out accuracy, AUC over the pooled held-out scores, and the per-fold scores.
    """
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n = x.shape[0]
    if n < 4:
        raise ValueError(f"need at least 4 episodes for a leave-one-out probe, got {n}")
    scores = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if len(set(y[mask].tolist())) < 2:
            scores[i] = 0.0
            continue
        train_x, test_x = _standardize(x[mask], x[i : i + 1])
        weights = fit_logistic(train_x, y[mask].astype(np.float64), l2=l2)
        scores[i] = float(predict_logit(weights, test_x)[0])
    predictions = (scores > 0).astype(np.int64)
    return {
        "n": int(n),
        "accuracy": float((predictions == y).mean()),
        "auc": roc_auc(y, scores),
        "scores": scores.tolist(),
        "labels": y.tolist(),
    }


def shuffled_null(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    repeats: int = 20,
    seed: int = 0,
    l2: float = 1.0,
) -> dict[str, float]:
    """The null this exact estimator yields when the labels carry no signal."""
    rng = np.random.default_rng(seed)
    accuracies, aucs = [], []
    for _ in range(repeats):
        permuted = rng.permutation(labels)
        result = leave_one_out_probe(features, permuted, l2=l2)
        accuracies.append(result["accuracy"])
        aucs.append(result["auc"])
    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_p95": float(np.percentile(accuracies, 95)),
        "auc_mean": float(np.nanmean(aucs)),
        "auc_p95": float(np.nanpercentile(aucs, 95)),
    }


class WriteTokenProbeRunner(_wa.V33WriterAttentionRunner):
    """Replays episodes to harvest per-phase write/read/state vectors, then probes them.

    Inherits the v3.3 replay setup (checkpoint alignment, transforms, jitted query step) and
    overrides only the per-frame recording: this diagnostic keeps the TOKENS, not the maps.
    """

    def __init__(self, options: Options):
        # The parent validates a superset of what we need and builds the model/transforms.
        super().__init__(
            _wa.Options(
                checkpoint=options.checkpoint,
                dataset_root=options.dataset_root,
                output_dir=options.output_dir,
                config=options.config,
                episode_indices=options.episode_indices,
                stride=options.stride,
            )
        )
        self.probe_options = options
        self.sample_stride = self.stride if options.sample_stride is None else options.sample_stride
        if self.stride % self.sample_stride and self.sample_stride < self.stride:
            raise ValueError(
                f"sample_stride ({self.sample_stride}) must divide the write stride ({self.stride}) "
                "so that every write frame is also sampled"
            )

    def _harvest(self, plan: _wa._EpisodePlan, tasks: dict[int, str]) -> dict[str, Any]:
        """One episode -> per-phase mean token vectors under TRUE and CF instructions.

        Sampling happens on the (possibly finer) ``sample_stride`` grid; the memory state is
        committed only on the training ``stride`` grid, so the retrieved tokens a sampled frame
        sees are exactly the ones the model would see at that point in the episode.
        """
        source = _wc._load_lerobot_sources(self.probe_options.dataset_root, [plan.episode])[0]
        columns = ["image", "left_wrist_image", "right_wrist_image", "state", "frame_index", "task_index"]
        rows = pq.read_table(source.path, columns=columns).to_pylist()

        memory_state = self.model.memory.init_state(1)
        collected: dict[str, dict[str, list[np.ndarray]]] = {
            phase: {f"{stream}_{cond}": [] for stream in PROBE_STREAMS for cond in ("true", "cf")}
            for phase in PROBE_PHASES
        }
        started = time.monotonic()
        for row in rows:
            raw_frame = int(row["frame_index"])
            if raw_frame % self.sample_stride or raw_frame > plan.memory[1]:
                continue
            observation_true, _ = self._observation(row, raw_frame, plan.prompt)
            observation_cf, _ = self._observation(row, raw_frame, plan.counterfactual)
            out_true = self._qstep(observation_true, memory_state)
            out_cf = self._qstep(observation_cf, memory_state)

            phase = (
                "evidence"
                if plan.evidence[0] <= raw_frame <= plan.evidence[1]
                else "waiting"
                if raw_frame >= plan.memory[0]
                else "retention"
                if raw_frame > plan.evidence[1]
                else "approach"
            )
            state_vector = np.asarray(row["state"], dtype=np.float64).reshape(-1)
            for cond, out in (("true", out_true), ("cf", out_cf)):
                bucket = collected[phase]
                bucket[f"write_{cond}"].append(np.asarray(out["write_tokens"], dtype=np.float64)[0].reshape(-1))
                bucket[f"read_{cond}"].append(np.asarray(out["retrieved"], dtype=np.float64)[0].reshape(-1))
                bucket[f"state_{cond}"].append(state_vector)
            # Commit exactly what the model would write under the REAL instruction -- but only on
            # the training write grid, so a finer sample_stride does not over-write the memory.
            if raw_frame % self.stride == 0:
                memory_state = self._write(memory_state, out_true["write_tokens"])[0]

        summary = {
            phase: {key: np.mean(np.stack(vals), axis=0) for key, vals in bucket.items() if vals}
            for phase, bucket in collected.items()
        }
        counts = {phase: len(bucket["write_true"]) for phase, bucket in collected.items()}
        print(
            f"episode {plan.episode} ({plan.prompt} / {plan.side}): "
            f"{counts} frames/phase in {time.monotonic() - started:.1f}s"
        )
        return {"plan": plan, "phase_vectors": summary, "counts": counts}

    def run(self) -> dict[str, Any]:
        tasks = _wa._read_tasks(self.probe_options.dataset_root)
        plans = _wa._plan_episodes(
            _wa.Options(
                checkpoint=self.probe_options.checkpoint,
                dataset_root=self.probe_options.dataset_root,
                output_dir=self.probe_options.output_dir,
                config=self.probe_options.config,
                episode_indices=self.probe_options.episode_indices,
                stride=self.probe_options.stride,
            ),
            self.data_config,
        )
        # _plan_episodes dedups to one episode per cell when no explicit indices are given;
        # a probe needs MANY episodes, so re-plan over every episode unless the caller pinned some.
        if not self.probe_options.episode_indices:
            plans = _plan_all_episodes(self.probe_options, self.data_config)
        if self.probe_options.max_episodes:
            plans = _stratified_subset(plans, self.probe_options.max_episodes)

        print(f"pathway scalars: {self.scalars}")
        print(f"probing {len(plans)} episodes")
        harvested = [self._harvest(plan, tasks) for plan in plans]

        results = analyze(harvested, seed=self.probe_options.seed)
        results["schema_version"] = SCHEMA_VERSION
        results["checkpoint"] = str(self.probe_options.checkpoint)
        results["pathway_scalars"] = self.scalars
        results["episodes"] = [
            {
                "episode": h["plan"].episode,
                "prompt": h["plan"].prompt,
                "side": h["plan"].side,
                "counts": h["counts"],
            }
            for h in harvested
        ]

        self.probe_options.output_dir.mkdir(parents=True, exist_ok=True)
        (self.probe_options.output_dir / "probe.json").write_text(json.dumps(results, indent=2))
        print_report(results)
        return results


def _plan_all_episodes(options: Options, data_config: Any) -> list[_wa._EpisodePlan]:
    """Every usable episode (the probe needs the full set, not one per cell)."""
    tasks = _wa._read_tasks(options.dataset_root)
    prompts = _wa._read_prompts(options.dataset_root)
    first, second = sorted(set(prompts.values()))
    other = {first: second, second: first}

    plans: list[_wa._EpisodePlan] = []
    for episode in sorted(prompts):
        source = _wc._load_lerobot_sources(options.dataset_root, [episode])[0]
        task_ids = np.asarray(pq.read_table(source.path, columns=["task_index"])["task_index"])
        phases = _wa._episode_phases(task_ids, tasks, data_config)
        if phases is None:
            continue
        final = tasks[int(task_ids[-1])].lower()
        side = "left" if "left" in final else ("right" if "right" in final else "?")
        if side == "?":
            continue
        plans.append(
            _wa._EpisodePlan(
                episode=episode,
                prompt=prompts[episode],
                counterfactual=other[prompts[episode]],
                side=side,
                evidence=phases[0],
                memory=phases[1],
                length=len(task_ids),
            )
        )
    if not plans:
        raise ValueError("no usable episodes found")
    return plans


def _stratified_subset(plans: list[_wa._EpisodePlan], limit: int) -> list[_wa._EpisodePlan]:
    """Cap the episode count while keeping the (instruction, side) cells balanced.

    The dataset is stored in cell order, so a naive ``plans[:limit]`` yields a degenerate
    design -- the first 8 episodes are 7 left / 1 right, where "always left" scores 0.875 and
    the probe measures nothing. Round-robin over the four cells instead.
    """
    if limit >= len(plans):
        return plans
    cells: dict[tuple[str, str], list[_wa._EpisodePlan]] = {}
    for plan in plans:
        cells.setdefault((plan.prompt, plan.side), []).append(plan)
    ordered = [sorted(group, key=lambda p: p.episode) for _, group in sorted(cells.items())]
    picked: list[_wa._EpisodePlan] = []
    for depth in range(max(len(group) for group in ordered)):
        for group in ordered:
            if depth < len(group) and len(picked) < limit:
                picked.append(group[depth])
    return sorted(picked, key=lambda p: p.episode)


def analyze(harvested: list[dict[str, Any]], *, seed: int = 0) -> dict[str, Any]:
    """Leave-one-episode-out probes for every (phase, stream), plus shuffled nulls.

    Also runs the CF-transfer test: a probe fit on TRUE-instruction tokens, applied to the SAME
    frames encoded under the counterfactual instruction. If the instruction steers the writer's
    content, the CF score should move toward the other side; ``cf_flip_rate`` counts how often
    the sign actually flips.
    """
    labels = np.array([1 if h["plan"].side == "right" else 0 for h in harvested], dtype=np.int64)
    prompts = [h["plan"].prompt for h in harvested]
    majority = float(max(labels.mean(), 1.0 - labels.mean())) if len(labels) else float("nan")
    out: dict[str, Any] = {
        "phases": {},
        "label_balance": {"right": int(labels.sum()), "n": len(labels), "majority_rate": majority},
    }
    # A lopsided design makes accuracy meaningless (predict the majority and "win"), so say so
    # rather than emit a number that reads as a result. AUC is balance-robust; accuracy is not.
    if len(labels) and majority > 0.65:
        out["warning"] = (
            f"label design is imbalanced ({int(labels.sum())} right / {len(labels)}): "
            f"always predicting the majority scores {majority:.2f}. Read AUC, not accuracy."
        )
        print(f"WARNING: {out['warning']}")

    for phase in PROBE_PHASES:
        available = [h for h in harvested if phase in h["phase_vectors"] and "write_true" in h["phase_vectors"][phase]]
        if len(available) < 4:
            continue
        phase_labels = np.array([1 if h["plan"].side == "right" else 0 for h in available], dtype=np.int64)
        entry: dict[str, Any] = {"n_episodes": len(available)}
        for stream in PROBE_STREAMS:
            features = np.stack([h["phase_vectors"][phase][f"{stream}_true"] for h in available])
            probe = leave_one_out_probe(features, phase_labels)
            probe["null"] = shuffled_null(features, phase_labels, seed=seed)
            if stream == "write":
                probe["cf_transfer"] = _cf_transfer(available, phase, phase_labels)
            entry[stream] = probe
        out["phases"][phase] = entry

    out["prompt_balance"] = {p: prompts.count(p) for p in sorted(set(prompts))}
    return out


def _cf_transfer(available: list[dict[str, Any]], phase: str, labels: np.ndarray) -> dict[str, Any]:
    """Fit on TRUE tokens (leave-one-out), score the held-out episode's CF encoding.

    A conditioner that writes instruction-dependent CONTENT should push the CF score away from
    the true side. ``cf_flip_rate`` is the fraction of held-out episodes whose predicted side
    flips when only the instruction changes; ``mean_score_shift`` is the signed movement toward
    the opposite side (positive = moved the way conditioning would predict).
    """
    true_features = np.stack([h["phase_vectors"][phase]["write_true"] for h in available])
    cf_features = np.stack([h["phase_vectors"][phase]["write_cf"] for h in available])
    n = len(available)
    true_scores, cf_scores = np.zeros(n), np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        if len(set(labels[mask].tolist())) < 2:
            continue
        train_x, _ = _standardize(true_features[mask], true_features[mask])
        mean = true_features[mask].mean(axis=0, keepdims=True)
        scale = true_features[mask].std(axis=0, keepdims=True)
        scale = np.where(scale < 1e-8, 1.0, scale)
        weights = fit_logistic(train_x, labels[mask].astype(np.float64))
        true_scores[i] = float(predict_logit(weights, (true_features[i : i + 1] - mean) / scale)[0])
        cf_scores[i] = float(predict_logit(weights, (cf_features[i : i + 1] - mean) / scale)[0])
    flipped = (np.sign(true_scores) != np.sign(cf_scores)) & (np.abs(true_scores) > 0)
    # Signed shift toward the opposite side: for a right episode (label 1) that means going down.
    direction = np.where(labels == 1, -1.0, 1.0)
    return {
        "cf_flip_rate": float(flipped.mean()),
        "mean_score_shift": float(np.mean(direction * (cf_scores - true_scores))),
        "mean_abs_score_shift": float(np.mean(np.abs(cf_scores - true_scores))),
        "true_score_abs_mean": float(np.mean(np.abs(true_scores))),
    }


def print_report(results: dict[str, Any]) -> None:
    balance = results["label_balance"]
    print(f"\nlabel balance: {balance['right']} right / {balance['n']} episodes")
    header = f"{'phase':>10} {'n':>4} {'stream':>7} {'acc':>6} {'auc':>6} {'null_acc':>9} {'null_auc_p95':>13}"
    print(header)
    for phase, entry in results["phases"].items():
        for stream in PROBE_STREAMS:
            if stream not in entry:
                continue
            probe = entry[stream]
            null = probe["null"]
            print(
                f"{phase:>10} {entry['n_episodes']:>4} {stream:>7} "
                f"{probe['accuracy']:>6.2f} {probe['auc']:>6.2f} "
                f"{null['accuracy_mean']:>9.2f} {null['auc_p95']:>13.2f}"
            )
    print("\nCF transfer (write tokens; does the instruction change the encoded side?)")
    for phase, entry in results["phases"].items():
        if "write" not in entry or "cf_transfer" not in entry["write"]:
            continue
        cf = entry["write"]["cf_transfer"]
        print(
            f"{phase:>10}  flip_rate={cf['cf_flip_rate']:.2f}  "
            f"shift_toward_other_side={cf['mean_score_shift']:+.3f}  "
            f"|shift|={cf['mean_abs_score_shift']:.3f}  |true_score|={cf['true_score_abs_mean']:.3f}"
        )
    print(
        "\nread the table as: write-token AUC is only evidence of encoded memory if it clears "
        "BOTH null_auc_p95 AND the 'state' row (the known proprioceptive leak)."
    )
