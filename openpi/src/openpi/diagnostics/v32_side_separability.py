"""Side-separability analysis of v3.2 write and retrieved tokens.

Given the per-frame tensors saved by :mod:`openpi.diagnostics.v32_checkpoint` for one
left-side and one right-side episode, localize where left/right information lives in the
memory pipeline::

    write tokens z_t  ->  fast weights M_t  ->  retrieved tokens m_t  ->  gated injection

This module compares stages ``z`` (write tokens) and ``m`` (retrieved tokens, pre-gate) at
matched phases of the two episodes. The stored episodes are phase-locked (same subtask
transition step), so no temporal warping is needed.

With one episode per side, "side" is confounded with episode identity; the mitigation is
the phase-resolved profile: a content-driven side signal must be absent before the banana
is revealed and emerge at the reveal, whereas an identity/nuisance signal is present from
frame zero.

Outputs: ``results.json``, ``separation_over_time.png``, ``lr_projection.png``,
``pca_trajectories.png`` under the requested output directory.
"""

import dataclasses
import json
import pathlib

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Phase windows in replay steps (stride-15 frames). Both stored episodes share this
# structure: gt subtask flips at step 33, bins are closed before step ~13 and after step
# ~29, and the chosen bin is re-opened by the robot around step 44. Transition steps
# (arms occluding / lids in motion) are deliberately excluded from every window.
PHASE_WINDOWS: dict[str, tuple[int, int]] = {
    "pre_reveal": (0, 8),
    "reveal": (16, 26),
    "post_closure": (30, 32),
    "decision_closed": (33, 40),
    "execution_open": (44, 56),
}

STAGES = ("write_tokens", "retrieved")


@dataclasses.dataclass(frozen=True)
class Options:
    left_dir: pathlib.Path
    right_dir: pathlib.Path
    output_dir: pathlib.Path


def _load_stage(episode_dir: pathlib.Path, stage: str) -> np.ndarray:
    """Return the [T, 16, 2048] float32 token array for one stage of one episode."""
    with np.load(episode_dir / "arrays.npz") as data:
        return data[stage].astype(np.float32)


def _phase_steps(name: str, num_steps: int) -> np.ndarray:
    start, end = PHASE_WINDOWS[name]
    return np.arange(start, min(end, num_steps - 1) + 1)


def _unit(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distance along the last axis, broadcasting over leading axes."""
    return 1.0 - np.sum(_unit(a) * _unit(b), axis=-1)


def slot_uniformity(tokens: np.ndarray) -> np.ndarray:
    """Mean pairwise cosine similarity among the 16 slots, per frame ([T])."""
    u = _unit(tokens)  # [T, 16, d]
    gram = np.einsum("tsd,trd->tsr", u, u)
    num_slots = tokens.shape[1]
    off_diag_sum = gram.sum(axis=(1, 2)) - np.trace(gram, axis1=1, axis2=2)
    return off_diag_sum / (num_slots * (num_slots - 1))


def within_episode_distance(pooled: np.ndarray, steps: np.ndarray) -> float:
    """Mean pairwise cosine distance among a phase's frames within one episode."""
    if len(steps) < 2:
        return float("nan")
    u = _unit(pooled[steps])
    gram = u @ u.T
    n = len(steps)
    off_diag_sum = gram.sum() - np.trace(gram)
    return float(1.0 - off_diag_sum / (n * (n - 1)))


def nearest_centroid_accuracy(
    left: np.ndarray, right: np.ndarray, fit_steps: np.ndarray, eval_steps: np.ndarray
) -> float:
    """Fit per-side centroids on ``fit_steps`` and classify ``eval_steps`` frames.

    Frames from both episodes are classified by cosine similarity to the two centroids;
    chance is 0.5. When fit and eval windows overlap, the overlapping frames are removed
    from the fit set (leave-frames-out) so the centroid never contains an evaluated frame.
    """
    fit = np.setdiff1d(fit_steps, eval_steps)
    if len(fit) == 0:  # same-window decode: even/odd split instead
        fit, eval_steps = fit_steps[::2], fit_steps[1::2]
        if len(eval_steps) == 0:
            return float("nan")
    centroids = _unit(np.stack([left[fit].mean(axis=0), right[fit].mean(axis=0)]))
    correct = 0
    for episode_index, pooled in enumerate((left, right)):
        sims = _unit(pooled[eval_steps]) @ centroids.T
        correct += int(np.sum(np.argmax(sims, axis=1) == episode_index))
    return correct / (2 * len(eval_steps))


def lr_axis_projection(
    left: np.ndarray, right: np.ndarray, fit_steps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project both episodes onto the unit left-minus-right axis fitted on one window."""
    axis = _unit(left[fit_steps].mean(axis=0) - right[fit_steps].mean(axis=0))
    center = 0.5 * (left[fit_steps].mean(axis=0) + right[fit_steps].mean(axis=0))
    return (left - center) @ axis, (right - center) @ axis


def analyze_stage(left_tokens: np.ndarray, right_tokens: np.ndarray) -> dict:
    """All separability metrics for one stage; token arrays are [T, 16, d]."""
    num_steps = min(left_tokens.shape[0], right_tokens.shape[0])
    left_tokens, right_tokens = left_tokens[:num_steps], right_tokens[:num_steps]
    left, right = left_tokens.mean(axis=1), right_tokens.mean(axis=1)

    cross = cosine_distance(left, right)  # [T]
    reveal_fit = _phase_steps("reveal", num_steps)[::2]
    proj_left, proj_right = lr_axis_projection(left, right, reveal_fit)

    per_phase = {}
    for name in PHASE_WINDOWS:
        steps = _phase_steps(name, num_steps)
        within = np.nanmean(
            [within_episode_distance(left, steps), within_episode_distance(right, steps)]
        )
        cross_mean = float(cross[steps].mean())
        per_phase[name] = {
            "steps": [int(steps[0]), int(steps[-1])],
            "cross_episode_distance": cross_mean,
            "within_episode_distance": float(within),
            "separation_ratio": cross_mean / max(float(within), 1e-12),
            "decode_within_phase": nearest_centroid_accuracy(left, right, steps, steps),
            "decode_from_reveal": nearest_centroid_accuracy(
                left, right, _phase_steps("reveal", num_steps), steps
            ),
            "slot_uniformity_left": float(slot_uniformity(left_tokens[steps]).mean()),
            "slot_uniformity_right": float(slot_uniformity(right_tokens[steps]).mean()),
            "norm_left": float(np.linalg.norm(left[steps], axis=-1).mean()),
            "norm_right": float(np.linalg.norm(right[steps], axis=-1).mean()),
        }

    per_slot_reveal_vs_pre = {}
    for name in ("pre_reveal", "reveal", "post_closure", "decision_closed"):
        steps = _phase_steps(name, num_steps)
        per_slot = cosine_distance(left_tokens[steps], right_tokens[steps])  # [n, 16]
        per_slot_reveal_vs_pre[name] = per_slot.mean(axis=0).tolist()

    return {
        "num_steps": int(num_steps),
        "cross_episode_distance_per_step": cross.tolist(),
        "lr_projection_left": proj_left.tolist(),
        "lr_projection_right": proj_right.tolist(),
        "per_phase": per_phase,
        "per_slot_cross_distance": per_slot_reveal_vs_pre,
    }


def _shade_phases(ax: plt.Axes, num_steps: int) -> None:
    colors = {"reveal": "#fff2b2", "post_closure": "#d8d8d8", "decision_closed": "#cfe6ff"}
    for name, color in colors.items():
        start, end = PHASE_WINDOWS[name]
        ax.axvspan(start, min(end, num_steps - 1), color=color, alpha=0.6, lw=0)


def _render_figures(results: dict[str, dict], output_dir: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharex=True)
    for ax, stage in zip(axes, STAGES, strict=True):
        r = results[stage]
        cross = np.asarray(r["cross_episode_distance_per_step"])
        _shade_phases(ax, r["num_steps"])
        ax.plot(cross, color="crimson", lw=1.6, label="cross-episode distance L vs R")
        for name, phase in r["per_phase"].items():
            lo, hi = phase["steps"]
            ax.hlines(
                phase["within_episode_distance"], lo, hi, color="k", ls="--", lw=1.2,
                label="within-episode jitter" if name == "pre_reveal" else None,
            )
        ax.set_title(f"{stage}: cosine distance between episodes")
        ax.set_xlabel("replay step (stride-15 frames)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("cosine distance")
    fig.tight_layout()
    fig.savefig(output_dir / "separation_over_time.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharex=True)
    for ax, stage in zip(axes, STAGES, strict=True):
        r = results[stage]
        _shade_phases(ax, r["num_steps"])
        ax.plot(r["lr_projection_left"], color="tab:blue", lw=1.6, label="left episode")
        ax.plot(r["lr_projection_right"], color="tab:orange", lw=1.6, label="right episode")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_title(f"{stage}: projection on reveal-fitted L-R axis")
        ax.set_xlabel("replay step (stride-15 frames)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("projection")
    fig.tight_layout()
    fig.savefig(output_dir / "lr_projection.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, stage in zip(axes, STAGES, strict=True):
        r = results[stage]
        left = np.asarray(r["_pooled_left"])
        right = np.asarray(r["_pooled_right"])
        both = np.concatenate([left, right], axis=0)
        centered = both - both.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        pl, pr = (left - both.mean(axis=0)) @ vt[:2].T, (right - both.mean(axis=0)) @ vt[:2].T
        steps = np.arange(len(left))
        ax.scatter(pl[:, 0], pl[:, 1], c=steps, cmap="Blues", marker="o", label="left ep")
        ax.scatter(pr[:, 0], pr[:, 1], c=steps, cmap="Oranges", marker="^", label="right ep")
        ax.plot(pl[:, 0], pl[:, 1], color="tab:blue", lw=0.5, alpha=0.5)
        ax.plot(pr[:, 0], pr[:, 1], color="tab:orange", lw=0.5, alpha=0.5)
        ax.set_title(f"{stage}: PCA trajectory (color = time)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_trajectories.png", dpi=150)
    plt.close(fig)


def run(options: Options) -> dict:
    options.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for stage in STAGES:
        left_tokens = _load_stage(options.left_dir, stage)
        right_tokens = _load_stage(options.right_dir, stage)
        stage_result = analyze_stage(left_tokens, right_tokens)
        num_steps = stage_result["num_steps"]
        stage_result["_pooled_left"] = left_tokens[:num_steps].mean(axis=1).tolist()
        stage_result["_pooled_right"] = right_tokens[:num_steps].mean(axis=1).tolist()
        results[stage] = stage_result

    _render_figures(results, options.output_dir)
    serializable = {
        stage: {k: v for k, v in r.items() if not k.startswith("_")}
        for stage, r in results.items()
    }
    serializable["phase_windows"] = {k: list(v) for k, v in PHASE_WINDOWS.items()}
    serializable["inputs"] = {
        "left_dir": str(options.left_dir),
        "right_dir": str(options.right_dir),
    }
    with (options.output_dir / "results.json").open("w") as f:
        json.dump(serializable, f, indent=2)
    return serializable


def main() -> int:
    import tyro

    run(tyro.cli(Options))
    return 0
