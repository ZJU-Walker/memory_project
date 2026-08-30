"""v3.4 waiting-phase side-leak audit (dataset-level, model-free, CPU-only).

The v3.4 runs show aux_side_accuracy ~0.99 and ladder_read_accuracy ~1.0 while the writer
metric sits near chance: the waiting-phase side label is being decoded from something other
than remembered evidence. This audit quantifies and VISUALIZES every dataset-level channel
through which the CURRENT waiting observation can carry the side answer, on
yam/bin_memory_0816_subtask:

  1. label/phase anomalies (episode 26's missing "open right bin", non-canonical orders,
     waiting-phase arm motion, sampler-exact endpoint motion fractions);
  2. state leakage: leave-one-episode-out ridge probes of side from the raw 14-D state on
     waiting frames (full state / arm joints / grippers / single arm), per-coordinate
     effect sizes, L-vs-R reset-pose RMS separation;
  3. image leakage: leave-one-episode-out ridge probes of side from downsampled top-camera
     (and wrist-camera) waiting frames, plus mean-image L/R difference heatmaps that
     localize the leaking pixels;
  4. collection-order confounding (side blocks in episode index order).

Model-free by design: it measures what a linear reader could extract from the raw data the
model sees, independent of any checkpoint. The trained-model attribution lives in
scripts/v34_waiting_shortcut_eval.py.

The sampler geometry (stride 15, lookahead 15, start pad 75, waiting window from the label
phases) mirrors src/openpi/training/data_loader.py::_memory_critical_windows and
src/openpi/transforms.py::memory_critical_endpoint; the mirror is cross-checked against the
real function by scripts/v34_leak_audit_test.py.

Usage (CPU, ~3-6 min for extract):
  .venv/bin/python scripts/v34_leak_audit.py --stage all
Outputs under diagnostic_outputs/v34_leak_audit/: cache/, report.json, report.md, figures/.
"""

import argparse
import dataclasses
import io
import json
import multiprocessing as mp
import pathlib

import numpy as np

DATASET_ROOT = pathlib.Path("/iris/u/kewalk/.cache/huggingface/lerobot/yam/bin_memory_0816_subtask")
OUT_ROOT = pathlib.Path("diagnostic_outputs/v34_leak_audit")

# Production config (pi05_yam_mem_v34 -> run5): src/openpi/training/config.py
STRIDE = 15
LOOKAHEAD = 15
START_PAD = 75
MAX_BUCKET = 40
HELDOUT = (15, 29, 44, 59)
EVIDENCE = "inspect both bins"
WAIT_LABELS = ("wait; target bin is left", "wait; target bin is right")
CANONICAL_ORDER = (
    "open both lids",
    "inspect both bins",
    "close both lids and reset arms",
    "wait; target bin is left|right",
    "open left bin|open right bin",
)

DENSE_WAIT_STRIDE = 5  # frames between cached waiting thumbnails
GRAY_HW = (60, 80)
RGB_HW = (120, 160)
FILM_HW = (240, 320)
STATE_DIM_NAMES = [f"L{i}" for i in range(6)] + ["Lgrip"] + [f"R{i}" for i in range(6)] + ["Rgrip"]
ARM_DIMS = list(range(6)) + list(range(7, 13))
GRIP_DIMS = [6, 13]


def _load_tasks() -> dict[int, str]:
    """{task_index: task string} from the dataset's meta/tasks.jsonl."""
    lines = (DATASET_ROOT / "meta/tasks.jsonl").read_text().splitlines()
    return {json.loads(line)["task_index"]: json.loads(line)["task"] for line in lines if line.strip()}


def memory_critical_endpoint_mirror(frame_index: int, window: np.ndarray, *, stride: int, lookahead: int, num_steps: int) -> int:
    """Verbatim mirror of src/openpi/transforms.py::memory_critical_endpoint (kept import-free
    so the audit cannot segfault via the openpi.transforms/pyarrow import-order trap)."""
    memory_lo, memory_hi = int(window[2]), int(window[3])
    step_frames = frame_index + np.arange(num_steps) * stride
    in_wait = (step_frames >= memory_lo) & (step_frames <= memory_hi)
    eligible = np.nonzero(in_wait & (step_frames <= memory_hi - lookahead))[0]
    if len(eligible) == 0:
        eligible = np.nonzero(in_wait)[0]
    if len(eligible) == 0:
        eligible = np.nonzero(step_frames < memory_lo)[0][-1:]
    return int(eligible[frame_index % len(eligible)])


# ---------------------------------------------------------------------------
# Stage: extract
# ---------------------------------------------------------------------------


def decode_png(cell, hw):
    from PIL import Image

    raw = cell["bytes"] if isinstance(cell, dict) else cell
    im = Image.open(io.BytesIO(raw)).convert("RGB").resize((hw[1], hw[0]), Image.BILINEAR)
    return np.asarray(im, dtype=np.uint8)


def _segments(task: np.ndarray):
    seg = []
    prev = None
    for i, x in enumerate(task):
        if x != prev:
            seg.append([int(x), i, i])
            prev = x
        else:
            seg[-1][2] = i
    return seg


def _extract_episode(e: int) -> str:
    import pyarrow.parquet as pq

    tasks = _load_tasks()
    path = DATASET_ROOT / f"data/chunk-000/episode_{e:06d}.parquet"
    pf = pq.ParquetFile(path)
    scalars = pf.read(columns=["task_index", "state"])
    task = np.asarray(scalars["task_index"], dtype=np.int64)
    state = np.stack([np.asarray(x, dtype=np.float32) for x in scalars["state"].to_pylist()])
    n = len(task)
    labels = [tasks[int(x)] for x in task]

    # phase bounds mirroring data_loader._episode_info_table
    is_ev = np.asarray([s == EVIDENCE for s in labels])
    is_wait = np.asarray([s in WAIT_LABELS for s in labels])
    ev_hit = np.nonzero(is_ev)[0]
    mem_hit = np.nonzero(is_wait)[0]
    ev_start, ev_end = (int(ev_hit[0]), int(ev_hit[-1])) if len(ev_hit) else (-1, -1)
    mem_lo, mem_hi = (int(mem_hit[0]), int(mem_hit[-1])) if len(mem_hit) else (-1, -1)
    mem_contig = bool(len(mem_hit) == 0 or (mem_hi - mem_lo + 1 == len(mem_hit)))
    window = np.asarray([max(1, ev_start - START_PAD), ev_start, mem_lo, mem_hi], dtype=np.int32)

    side = 1 if any("right" in s for s in labels if s in WAIT_LABELS) else 0
    dense = np.arange(mem_lo, mem_hi + 1, DENSE_WAIT_STRIDE, dtype=np.int64) if mem_lo >= 0 else np.zeros(0, np.int64)
    mid_wait = int((mem_lo + mem_hi) // 2) if mem_lo >= 0 else n // 2
    film = np.unique(
        np.clip(
            np.asarray(
                [
                    max(0, ev_start - 60),
                    ev_start,
                    (ev_start + ev_end) // 2,
                    ev_end,
                    (ev_end + mem_lo) // 2,
                    mem_lo,
                    mid_wait,
                    mem_hi,
                    min(n - 1, mem_hi + 45),
                ]
            ),
            0,
            n - 1,
        )
    )

    # decode only the row groups containing needed frames
    need_top = sorted(set(dense.tolist()) | set(film.tolist()))
    rg_rows = [pf.metadata.row_group(g).num_rows for g in range(pf.metadata.num_row_groups)]
    rg_start = np.concatenate([[0], np.cumsum(rg_rows)])
    by_group: dict[int, list[int]] = {}
    for f in need_top:
        g = int(np.searchsorted(rg_start, f, side="right") - 1)
        by_group.setdefault(g, []).append(f)
    top_cache: dict[int, np.ndarray] = {}
    wrist_cache: dict[str, np.ndarray] = {}
    for g, frames in sorted(by_group.items()):
        cols = ["image"] + (["left_wrist_image", "right_wrist_image"] if any(f == mid_wait for f in frames) else [])
        tbl = pf.read_row_group(g, columns=cols)
        for f in frames:
            local = f - int(rg_start[g])
            top_cache[f] = decode_png(tbl["image"][local].as_py(), RGB_HW)
            if f == mid_wait and "left_wrist_image" in tbl.column_names:
                wrist_cache["left"] = decode_png(tbl["left_wrist_image"][local].as_py(), RGB_HW)
                wrist_cache["right"] = decode_png(tbl["right_wrist_image"][local].as_py(), RGB_HW)
        # film frames at higher resolution from the same group
        for f in frames:
            if f in film:
                local = f - int(rg_start[g])
                top_cache[-f - 1] = decode_png(tbl["image"][local].as_py(), FILM_HW)

    wait_rgb = np.stack([top_cache[f] for f in dense]) if len(dense) else np.zeros((0, *RGB_HW, 3), np.uint8)
    film_rgb = np.stack([top_cache[-f - 1] for f in film])
    out = OUT_ROOT / "cache" / f"ep{e:03d}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        state=state,
        task=task,
        window=window,
        side=side,
        ev_start=ev_start,
        ev_end=ev_end,
        mem_lo=mem_lo,
        mem_hi=mem_hi,
        mem_contig=mem_contig,
        segments=np.asarray(_segments(task), dtype=np.int64),
        dense_frames=dense,
        wait_rgb=wait_rgb,
        film_frames=film,
        film_rgb=film_rgb,
        wrist_left=wrist_cache.get("left", np.zeros((*RGB_HW, 3), np.uint8)),
        wrist_right=wrist_cache.get("right", np.zeros((*RGB_HW, 3), np.uint8)),
        mid_wait=mid_wait,
    )
    return f"ep{e:03d}: T={n} wait=[{mem_lo},{mem_hi}] dense={len(dense)} contig={mem_contig}"


def stage_extract(workers: int) -> None:
    n_ep = len((DATASET_ROOT / "meta/episodes.jsonl").read_text().strip().splitlines())
    with mp.Pool(workers) as pool:
        for msg in pool.imap_unordered(_extract_episode, range(n_ep)):
            print(msg, flush=True)


# ---------------------------------------------------------------------------
# Stage: audit
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Episode:
    index: int
    state: np.ndarray
    task: np.ndarray
    window: np.ndarray
    side: int
    segments: np.ndarray
    dense_frames: np.ndarray
    wait_rgb: np.ndarray
    film_frames: np.ndarray
    film_rgb: np.ndarray
    wrist_left: np.ndarray
    wrist_right: np.ndarray
    mid_wait: int
    mem_contig: bool

    @property
    def task_name(self) -> str:
        return "banana" if self.index < 30 else "box"

    @property
    def gray(self) -> np.ndarray:
        g = self.wait_rgb.astype(np.float32).mean(axis=-1) / 255.0
        return g[:, ::2, ::2]  # 120x160 -> 60x80


def load_episodes() -> list[Episode]:
    eps = []
    for p in sorted((OUT_ROOT / "cache").glob("ep*.npz")):
        z = np.load(p, allow_pickle=False)
        eps.append(
            Episode(
                index=int(p.stem[2:]),
                state=z["state"],
                task=z["task"],
                window=z["window"],
                side=int(z["side"]),
                segments=z["segments"],
                dense_frames=z["dense_frames"],
                wait_rgb=z["wait_rgb"],
                film_frames=z["film_frames"],
                film_rgb=z["film_rgb"],
                wrist_left=z["wrist_left"],
                wrist_right=z["wrist_right"],
                mid_wait=int(z["mid_wait"]),
                mem_contig=bool(z["mem_contig"]),
            )
        )
    return eps


def loo_ridge_probe(feats: list[np.ndarray], sides: list[int], lam: float = 1e-2):
    """Leave-one-episode-out ridge probe (dual/kernel form; deterministic, dependency-free).

    feats[i]: [n_i, d] frames of episode i, sides[i] in {0, 1}. Returns per-episode frame
    accuracy and the balanced (mean-of-side-means of episode accuracies) accuracy.
    """
    x = np.concatenate(feats).astype(np.float64)
    ep_id = np.concatenate([np.full(len(f), i) for i, f in enumerate(feats)])
    y = np.concatenate([np.full(len(f), 1.0 if s == 1 else -1.0) for f, s in zip(feats, sides, strict=False)])
    accs = np.zeros(len(feats))
    for i in range(len(feats)):
        tr = ep_id != i
        xt, yt = x[tr], y[tr]
        mu, sd = xt.mean(axis=0), xt.std(axis=0) + 1e-8
        xt = (xt - mu) / sd
        xe = (x[~tr] - mu) / sd
        # class-balance the training frames by weighting
        w_pos, w_neg = (yt > 0).sum(), (yt < 0).sum()
        sw = np.where(yt > 0, 0.5 / max(w_pos, 1), 0.5 / max(w_neg, 1)) * len(yt)
        xw = xt * np.sqrt(sw)[:, None]
        gram = xw @ xw.T + lam * len(yt) * np.eye(len(yt))
        alpha = np.linalg.solve(gram, np.sqrt(sw) * yt)
        pred = xe @ (xw.T @ alpha)
        accs[i] = float(np.mean(np.sign(pred) == y[~tr][0]))
    sides_arr = np.asarray(sides)
    balanced = float(np.mean([accs[sides_arr == s].mean() for s in (0, 1)]))
    return accs, balanced


def stage_audit() -> dict:
    eps = load_episodes()
    report: dict = {"n_episodes": len(eps), "heldout": list(HELDOUT)}

    # -- 1. label/phase anomalies ------------------------------------------------
    anomalies = []
    per_ep = []
    for e in eps:
        wait_len = int(e.window[3] - e.window[2] + 1)
        tasks = _load_tasks()
        order = [tasks[int(s[0])] for s in e.segments]
        canonical = (
            len(order) == 5
            and order[0] == "open both lids"
            and order[1] == EVIDENCE
            and order[2] == "close both lids and reset arms"
            and order[3] in WAIT_LABELS
            and order[4] in ("open left bin", "open right bin")
        )
        if not canonical:
            anomalies.append({"episode": e.index, "order": order})
        # waiting-phase arm motion (dense frames vs waiting start)
        ref = e.state[int(e.window[2])]
        disp = np.abs(e.state[e.dense_frames][:, ARM_DIMS] - ref[ARM_DIMS]).max(axis=1) if len(e.dense_frames) else np.zeros(0)
        per_ep.append(
            {
                "episode": e.index,
                "side": "R" if e.side else "L",
                "task": e.task_name,
                "heldout": e.index in HELDOUT,
                "wait_frames": wait_len,
                "wait_max_arm_disp": float(disp.max()) if len(disp) else 0.0,
                "wait_p95_arm_disp": float(np.percentile(disp, 95)) if len(disp) else 0.0,
                "canonical_order": canonical,
            }
        )
    report["episodes"] = per_ep
    report["order_anomalies"] = anomalies

    # -- sampler-exact endpoint motion audit ------------------------------------
    endpoint_rows = []
    for e in eps:
        if e.window[2] < 0 or e.index in HELDOUT:
            continue
        ref = e.state[int(e.window[2])]
        for start in range(int(e.window[0]), int(e.window[1]) + 1):
            t_q = memory_critical_endpoint_mirror(start, e.window, stride=STRIDE, lookahead=LOOKAHEAD, num_steps=MAX_BUCKET)
            f_q = min(start + t_q * STRIDE, len(e.state) - 1)
            disp = float(np.abs(e.state[f_q][ARM_DIMS] - ref[ARM_DIMS]).max())
            lo = max(0, f_q - 7)
            hi = min(len(e.state) - 1, f_q + 7)
            inst = float(np.abs(e.state[hi][ARM_DIMS] - e.state[lo][ARM_DIMS]).max())
            endpoint_rows.append((e.index, f_q, disp, inst))
    ep_disp = np.asarray([r[2] for r in endpoint_rows])
    ep_inst = np.asarray([r[3] for r in endpoint_rows])
    by_ep = {}
    for idx, _, disp, _ in endpoint_rows:
        by_ep.setdefault(idx, []).append(disp)
    report["endpoint_audit"] = {
        "n_endpoints": len(endpoint_rows),
        "frac_disp_gt_0.02": float((ep_disp > 0.02).mean()),
        "frac_disp_gt_0.05": float((ep_disp > 0.05).mean()),
        "frac_inst_gt_0.02": float((ep_inst > 0.02).mean()),
        "per_episode_frac_gt_0.02": {str(k): float((np.asarray(v) > 0.02).mean()) for k, v in sorted(by_ep.items()) if (np.asarray(v) > 0.02).any()},
    }

    # -- 2. state probes ---------------------------------------------------------
    train_eps = [e for e in eps if e.index not in HELDOUT and len(e.dense_frames)]
    state_variants = {
        "state_full14": list(range(14)),
        "state_arms12": ARM_DIMS,
        "state_grips2": GRIP_DIMS,
        "state_leftarm7": list(range(7)),
        "state_rightarm7": list(range(7, 14)),
    }
    probes = {}
    for cell_name, cell in (("banana", [e for e in train_eps if e.task_name == "banana"]), ("box", [e for e in train_eps if e.task_name == "box"]), ("pooled", train_eps)):
        cell_probes = {}
        for vname, dims in state_variants.items():
            feats = [e.state[e.dense_frames][:, dims] for e in cell]
            accs, bal = loo_ridge_probe(feats, [e.side for e in cell])
            cell_probes[vname] = {"balanced_acc": round(bal, 4), "per_episode": {str(e.index): round(float(a), 3) for e, a in zip(cell, accs, strict=False)}}
        # image probes
        gray = [e.gray.reshape(len(e.gray), -1) for e in cell]
        accs, bal = loo_ridge_probe(gray, [e.side for e in cell], lam=1e-1)
        cell_probes["image_top_gray60x80"] = {"balanced_acc": round(bal, 4), "per_episode": {str(e.index): round(float(a), 3) for e, a in zip(cell, accs, strict=False)}}
        wl = [e.wrist_left.astype(np.float32).mean(axis=-1).reshape(1, -1) / 255.0 for e in cell]
        accs, bal = loo_ridge_probe(wl, [e.side for e in cell], lam=1e-1)
        cell_probes["image_wristL_midwait"] = {"balanced_acc": round(bal, 4)}
        wr = [e.wrist_right.astype(np.float32).mean(axis=-1).reshape(1, -1) / 255.0 for e in cell]
        accs, bal = loo_ridge_probe(wr, [e.side for e in cell], lam=1e-1)
        cell_probes["image_wristR_midwait"] = {"balanced_acc": round(bal, 4)}
        probes[cell_name] = cell_probes
    report["loo_probes"] = probes

    # heldout-only evaluation: train on the 56, test on the exact 4 heldout episodes
    heldout_eval = {}
    for vname, dims in (("state_full14", list(range(14))), ("image_top_gray60x80", None)):
        tr_feats = [(e.state[e.dense_frames][:, dims] if dims else e.gray.reshape(len(e.gray), -1)) for e in train_eps]
        te = [e for e in eps if e.index in HELDOUT]
        te_feats = [(e.state[e.dense_frames][:, dims] if dims else e.gray.reshape(len(e.gray), -1)) for e in te]
        x = np.concatenate(tr_feats).astype(np.float64)
        y = np.concatenate([np.full(len(f), 1.0 if e.side else -1.0) for f, e in zip(tr_feats, train_eps, strict=False)])
        mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-8
        xs = (x - mu) / sd
        lam = 1e-1 if dims is None else 1e-2
        gram = xs @ xs.T + lam * len(y) * np.eye(len(y))
        alpha = np.linalg.solve(gram, y)
        res = {}
        for e, f in zip(te, te_feats, strict=False):
            pred = ((f - mu) / sd) @ (xs.T @ alpha)
            res[str(e.index)] = round(float(np.mean(np.sign(pred) == (1.0 if e.side else -1.0))), 3)
        heldout_eval[vname] = res
    report["heldout_eval"] = heldout_eval

    # -- per-coordinate effect sizes (episode-level, waiting mean) ---------------
    coord = {}
    for cell_name, cell in (("banana", [e for e in train_eps if e.task_name == "banana"]), ("box", [e for e in train_eps if e.task_name == "box"])):
        means = np.stack([e.state[e.dense_frames].mean(axis=0) for e in cell])
        sides = np.asarray([e.side for e in cell])
        left_m, right_m = means[sides == 0], means[sides == 1]
        pooled_sd = np.sqrt(0.5 * (left_m.var(axis=0) + right_m.var(axis=0))) + 1e-8
        d = (right_m.mean(axis=0) - left_m.mean(axis=0)) / pooled_sd
        coord[cell_name] = {STATE_DIM_NAMES[i]: round(float(d[i]), 2) for i in range(14)}
        coord[cell_name + "_reset_rms_LR"] = round(float(np.sqrt(np.mean((left_m.mean(axis=0) - right_m.mean(axis=0)) ** 2))), 4)
    report["state_effect_sizes"] = coord

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "report.json", "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("episodes",)}, indent=1)[:4000])
    return report


# ---------------------------------------------------------------------------
# Stage: figures
# ---------------------------------------------------------------------------


def stage_figures() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    eps = load_episodes()
    fig_dir = OUT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tasks = _load_tasks()
    phase_colors = {
        "open both lids": "#bdbdbd",
        "inspect both bins": "#ffd54f",
        "close both lids and reset arms": "#90caf9",
        "wait; target bin is left": "#a5d6a7",
        "wait; target bin is right": "#ef9a9a",
        "open left bin": "#2e7d32",
        "open right bin": "#c62828",
    }

    # 1. phase timeline ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 12))
    for row, e in enumerate(eps):
        for t_id, lo, hi in e.segments:
            ax.barh(row, hi - lo + 1, left=lo, color=phase_colors[tasks[int(t_id)]], height=0.8)
        tag = f"ep{e.index:02d} {e.task_name[:3]}-{'R' if e.side else 'L'}" + (" *HELD*" if e.index in HELDOUT else "")
        ax.text(-8, row, tag, ha="right", va="center", fontsize=6.5, family="monospace")
    ax.set_ylim(-1, len(eps))
    ax.invert_yaxis()
    ax.set_xlabel("frame")
    ax.set_yticks([])
    ax.set_title("Phase timeline, all 60 episodes (yellow=inspect/evidence, green=wait-L, red=wait-R)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in phase_colors.values()]
    ax.legend(handles, phase_colors.keys(), loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "phase_timeline.png", dpi=130)
    plt.close(fig)

    # 2. waiting-frame grids ----------------------------------------------------
    for task_name in ("banana", "box"):
        cell = [e for e in eps if e.task_name == task_name]
        left = [e for e in cell if e.side == 0]
        right = [e for e in cell if e.side == 1]
        cols = max(len(left), len(right))
        fig, axes = plt.subplots(2, cols, figsize=(cols * 1.7, 3.4))
        for r, group in enumerate((left, right)):
            for c in range(cols):
                ax = axes[r, c]
                ax.axis("off")
                if c < len(group):
                    e = group[c]
                    mid = len(e.wait_rgb) // 2
                    ax.imshow(e.wait_rgb[mid])
                    held = "*" if e.index in HELDOUT else ""
                    ax.set_title(f"ep{e.index}{held}", fontsize=7)
        axes[0, 0].set_ylabel("LEFT", fontsize=9)
        fig.suptitle(f"{task_name}: top camera at MID-WAIT -- row 1 = target LEFT, row 2 = target RIGHT", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / f"waiting_grid_{task_name}.png", dpi=140)
        plt.close(fig)

    # 3. mean images and difference heatmaps ------------------------------------
    for task_name in ("banana", "box"):
        cell = [e for e in eps if e.task_name == task_name and e.index not in HELDOUT]
        mean_l = np.mean(np.concatenate([e.wait_rgb for e in cell if e.side == 0]).astype(np.float32), axis=0)
        mean_r = np.mean(np.concatenate([e.wait_rgb for e in cell if e.side == 1]).astype(np.float32), axis=0)
        diff = np.abs(mean_l - mean_r).mean(axis=-1)
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        axes[0].imshow(mean_l.astype(np.uint8))
        axes[0].set_title(f"{task_name}: mean waiting frame, target LEFT")
        axes[1].imshow(mean_r.astype(np.uint8))
        axes[1].set_title("target RIGHT")
        im = axes[2].imshow(diff, cmap="inferno")
        axes[2].set_title("|L - R| mean abs diff (leaking pixels)")
        fig.colorbar(im, ax=axes[2], fraction=0.04)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / f"meandiff_{task_name}.png", dpi=140)
        plt.close(fig)

    # 4. filmstrips with state overlay ------------------------------------------
    film_eps = [0, 16, 26, 30, 45, *HELDOUT]
    for idx in film_eps:
        e = next(x for x in eps if x.index == idx)
        k = len(e.film_frames)
        fig = plt.figure(figsize=(k * 2.4, 5.2))
        gs = fig.add_gridspec(2, k, height_ratios=[2.2, 1.3], hspace=0.25)
        for c in range(k):
            ax = fig.add_subplot(gs[0, c])
            ax.imshow(e.film_rgb[c])
            f = int(e.film_frames[c])
            ax.set_title(f"f{f}\n{tasks[int(e.task[f])]}", fontsize=7)
            ax.axis("off")
        ax = fig.add_subplot(gs[1, :])
        t = np.arange(len(e.state))
        for d in range(14):
            lw = 1.6 if d in GRIP_DIMS else 0.7
            ax.plot(t, e.state[:, d], lw=lw, label=STATE_DIM_NAMES[d] if d in GRIP_DIMS else None)
        for t_id, lo, hi in e.segments:
            ax.axvspan(lo, hi, color=phase_colors[tasks[int(t_id)]], alpha=0.22)
        for f in e.film_frames:
            ax.axvline(int(f), color="k", lw=0.5, ls=":")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_xlabel("frame")
        ax.set_ylabel("state (rad / gripper)")
        held = " [HELDOUT]" if idx in HELDOUT else ""
        fig.suptitle(f"ep{idx} {e.task_name}-{'R' if e.side else 'L'}{held}: top camera + full 14-D state (phases shaded)", fontsize=11)
        fig.savefig(fig_dir / f"filmstrip_ep{idx:02d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # 5. per-coordinate episode-mean separation ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), sharey=True)
    for ax, task_name in zip(axes, ("banana", "box"), strict=False):
        cell = [e for e in eps if e.task_name == task_name and e.index not in HELDOUT and len(e.dense_frames)]
        means = np.stack([e.state[e.dense_frames].mean(axis=0) for e in cell])
        sides = np.asarray([e.side for e in cell])
        for d in range(14):
            left_v = means[sides == 0, d]
            right_v = means[sides == 1, d]
            ax.scatter(np.full(len(left_v), d - 0.15), left_v, s=12, color="#2e7d32", alpha=0.75)
            ax.scatter(np.full(len(right_v), d + 0.15), right_v, s=12, color="#c62828", alpha=0.75)
        ax.set_xticks(range(14))
        ax.set_xticklabels(STATE_DIM_NAMES, rotation=45, fontsize=8)
        ax.set_title(f"{task_name}: waiting-phase episode-mean state by side (green=L, red=R)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean state during waiting")
    fig.tight_layout()
    fig.savefig(fig_dir / "state_coords.png", dpi=140)
    plt.close(fig)

    # 6. waiting motion bars ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 3.6))
    xs, ys, colors = [], [], []
    for e in eps:
        if not len(e.dense_frames):
            continue
        ref = e.state[int(e.window[2])]
        disp = np.abs(e.state[e.dense_frames][:, ARM_DIMS] - ref[ARM_DIMS]).max(axis=1)
        xs.append(e.index)
        ys.append(disp.max())
        colors.append("#c62828" if e.side else "#2e7d32")
    ax.bar(xs, ys, color=colors)
    ax.axhline(0.02, color="k", ls="--", lw=0.8)
    ax.set_xlabel("episode")
    ax.set_ylabel("max |arm joint disp| during wait (rad)")
    ax.set_title("Waiting-phase arm motion per episode (dashed = 0.02 rad; green=L, red=R)")
    fig.tight_layout()
    fig.savefig(fig_dir / "waiting_motion.png", dpi=140)
    plt.close(fig)

    # 7. episode 26 anomaly -----------------------------------------------------
    e = next(x for x in eps if x.index == 26)
    fig = plt.figure(figsize=(13, 5.5))
    gs = fig.add_gridspec(2, len(e.film_frames), height_ratios=[1.6, 1.4])
    for c in range(len(e.film_frames)):
        ax = fig.add_subplot(gs[0, c])
        ax.imshow(e.film_rgb[c])
        f = int(e.film_frames[c])
        ax.set_title(f"f{f}\n{tasks[int(e.task[f])][:22]}", fontsize=7)
        ax.axis("off")
    ax = fig.add_subplot(gs[1, :])
    ref = e.state[int(e.window[2])] if e.window[2] >= 0 else e.state[0]
    disp = np.abs(e.state[:, ARM_DIMS] - ref[ARM_DIMS]).max(axis=1)
    ax.plot(disp, lw=1.2, color="k")
    for t_id, lo, hi in e.segments:
        ax.axvspan(lo, hi, color=phase_colors[tasks[int(t_id)]], alpha=0.3)
    ax.axhline(0.02, color="r", ls="--", lw=0.8)
    ax.set_xlabel("frame")
    ax.set_ylabel("max |arm disp| vs wait start")
    fig.suptitle("EPISODE 26 ANOMALY: label says 'wait; target bin is right' while the right arm moves", fontsize=11)
    fig.savefig(fig_dir / "ep26_anomaly.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 8. wrist-camera mid-wait grid --------------------------------------------
    fig, axes = plt.subplots(4, 15, figsize=(15 * 1.15, 4 * 1.05))
    groups = [
        ("ban-L", [e for e in eps if e.task_name == "banana" and e.side == 0]),
        ("ban-R", [e for e in eps if e.task_name == "banana" and e.side == 1]),
        ("box-L", [e for e in eps if e.task_name == "box" and e.side == 0]),
        ("box-R", [e for e in eps if e.task_name == "box" and e.side == 1]),
    ]
    for r, (name, group) in enumerate(groups):
        for c in range(15):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(group):
                ax.imshow(group[c].wrist_left)
                ax.set_title(f"{name} ep{group[c].index}", fontsize=5.5)
    fig.suptitle("LEFT wrist camera at mid-wait, by cell", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "wrist_left_grid.png", dpi=130)
    plt.close(fig)

    print(f"figures written to {fig_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("extract", "audit", "figures", "all"), default="all")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage in ("extract", "all"):
        stage_extract(args.workers)
    if args.stage in ("audit", "all"):
        stage_audit()
    if args.stage in ("figures", "all"):
        stage_figures()


if __name__ == "__main__":
    main()
