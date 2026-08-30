"""Per-camera image-leak attribution for the v3.4 waiting phase (dataset-level, CPU-only).

Companion of scripts/v34_leak_audit.py, which established THAT waiting images are
side-separable (top 0.92 LOO) but probed the wrist cameras with a single mid-wait frame per
episode -- too coarse to rank cameras, and inconsistent with the checkpoint-2500 camera
factorial (left wrist Shapley +8.7 > right +5.7 > top +3.4). This script answers WHICH camera
leaks most and WHERE in each image the side signal lives:

  1. equal-footing LOO probes: all three cameras on the same dense waiting-frame grid;
  2. patch leak maps: an independent LOO probe per 10x10-px patch of the 60x80 gray image --
     a per-pixel-region map of side decodability;
  3. region-crop probes on the top camera (bins / arms / walls) to separate "the bins moved"
     from "the arms are parked differently" from "the room drifted";
  4. collection-drift nearest neighbours: is an episode's closest other episode the same side,
     and is it adjacent in collection order?

Reads the top-camera dense cache written by v34_leak_audit.py (run `--stage extract` there
first); extracts its own dense wrist cache on first run.

Usage:
  .venv/bin/python scripts/v34_image_leak.py --stage all
Outputs under diagnostic_outputs/v34_image_leak/: cache/, report.json, figures/.
"""

import argparse
import json
import multiprocessing as mp
import pathlib
import sys

import numpy as np

_SCRIPTS_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(_SCRIPTS_DIR))
try:
    import v34_leak_audit as audit
finally:
    sys.path.remove(str(_SCRIPTS_DIR))

OUT_ROOT = pathlib.Path("diagnostic_outputs/v34_image_leak")
AUDIT_CACHE = audit.OUT_ROOT / "cache"
HELDOUT = audit.HELDOUT
PATCH = 10  # 60x80 gray -> 6x8 patch grid
# Top-camera regions on the 60x80 gray image (rows, cols), estimated from the mean frames:
# bins sit upper-middle, arms lower half, walls/ceiling on top.
TOP_REGIONS = {
    "bins": (slice(16, 36), slice(20, 64)),
    "arms_table": (slice(36, 60), slice(0, 80)),
    "walls_bg": (slice(0, 16), slice(0, 80)),
}


# ---------------------------------------------------------------------------
# Stage: extract (dense wrist frames on the SAME grid as the audit's top cache)
# ---------------------------------------------------------------------------


def _extract_wrists(e: int) -> str:
    import pyarrow.parquet as pq

    z = np.load(AUDIT_CACHE / f"ep{e:03d}.npz")
    dense = z["dense_frames"]
    pf = pq.ParquetFile(audit.DATASET_ROOT / f"data/chunk-000/episode_{e:06d}.parquet")
    rg_rows = [pf.metadata.row_group(g).num_rows for g in range(pf.metadata.num_row_groups)]
    rg_start = np.concatenate([[0], np.cumsum(rg_rows)])
    by_group: dict[int, list[int]] = {}
    for f in dense.tolist():
        by_group.setdefault(int(np.searchsorted(rg_start, f, side="right") - 1), []).append(f)
    wl, wr = {}, {}
    for g, frames in sorted(by_group.items()):
        tbl = pf.read_row_group(g, columns=["left_wrist_image", "right_wrist_image"])
        for f in frames:
            local = f - int(rg_start[g])
            wl[f] = audit.decode_png(tbl["left_wrist_image"][local].as_py(), audit.RGB_HW)
            wr[f] = audit.decode_png(tbl["right_wrist_image"][local].as_py(), audit.RGB_HW)
    out = OUT_ROOT / "cache" / f"ep{e:03d}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        frames=dense,
        wl_rgb=np.stack([wl[f] for f in dense.tolist()]) if len(dense) else np.zeros((0, *audit.RGB_HW, 3), np.uint8),
        wr_rgb=np.stack([wr[f] for f in dense.tolist()]) if len(dense) else np.zeros((0, *audit.RGB_HW, 3), np.uint8),
    )
    return f"ep{e:03d}: {len(dense)} wrist frames"


def stage_extract(workers: int) -> None:
    episodes = sorted(int(p.stem[2:]) for p in AUDIT_CACHE.glob("ep*.npz"))
    with mp.Pool(workers) as pool:
        for msg in pool.imap_unordered(_extract_wrists, episodes):
            print(msg, flush=True)


# ---------------------------------------------------------------------------
# Stage: analyze
# ---------------------------------------------------------------------------


def _gray(rgb: np.ndarray) -> np.ndarray:
    g = rgb.astype(np.float32).mean(axis=-1) / 255.0
    return g[:, ::2, ::2]  # 120x160 -> 60x80


def load_cameras():
    """{camera: {episode: gray [n, 60, 80]}}, plus sides/tasks per episode."""
    cams: dict[str, dict[int, np.ndarray]] = {"top": {}, "wristL": {}, "wristR": {}}
    side, task = {}, {}
    for p in sorted(AUDIT_CACHE.glob("ep*.npz")):
        e = int(p.stem[2:])
        z = np.load(p)
        if not len(z["dense_frames"]):
            continue
        cams["top"][e] = _gray(z["wait_rgb"])
        side[e] = int(z["side"])
        task[e] = "banana" if e < 30 else "box"
        w = np.load(OUT_ROOT / "cache" / f"ep{e:03d}.npz")
        cams["wristL"][e] = _gray(w["wl_rgb"])
        cams["wristR"][e] = _gray(w["wr_rgb"])
    return cams, side, task


def ridge_loo(feats: list[np.ndarray], sides: list[int], lam: float):
    """Primal-or-dual class-balanced LOO ridge (matches audit.loo_ridge_probe semantics)."""
    d = feats[0].shape[1]
    n = sum(len(f) for f in feats)
    if d <= n:
        x = np.concatenate(feats).astype(np.float64)
        ep_id = np.concatenate([np.full(len(f), i) for i, f in enumerate(feats)])
        y = np.concatenate([np.full(len(f), 1.0 if s == 1 else -1.0) for f, s in zip(feats, sides, strict=False)])
        accs = np.zeros(len(feats))
        for i in range(len(feats)):
            tr = ep_id != i
            xt, yt = x[tr], y[tr]
            mu, sd = xt.mean(axis=0), xt.std(axis=0) + 1e-8
            xt = (xt - mu) / sd
            n_pos, n_neg = (yt > 0).sum(), (yt < 0).sum()
            sw = np.where(yt > 0, 0.5 / max(n_pos, 1), 0.5 / max(n_neg, 1)) * len(yt)
            a = xt * sw[:, None]
            w = np.linalg.solve(xt.T @ a + lam * len(yt) * np.eye(d), a.T @ yt)
            pred = ((x[~tr] - mu) / sd) @ w
            accs[i] = float(np.mean(np.sign(pred) == y[~tr][0]))
        sides_arr = np.asarray(sides)
        return accs, float(np.mean([accs[sides_arr == s].mean() for s in (0, 1)]))
    return audit.loo_ridge_probe(feats, sides, lam)


def _cells(side, task, cam_eps):
    train = [e for e in sorted(cam_eps) if e not in HELDOUT]
    return {
        "banana": [e for e in train if task[e] == "banana"],
        "box": [e for e in train if task[e] == "box"],
        "pooled": train,
    }


def stage_analyze() -> dict:
    cams, side, task = load_cameras()
    report: dict = {}

    # 1. equal-footing full-image probes ---------------------------------------
    full = {}
    for cam, eps in cams.items():
        cells = _cells(side, task, eps)
        full[cam] = {
            name: round(
                ridge_loo([eps[e].reshape(len(eps[e]), -1) for e in members], [side[e] for e in members], 1e-1)[1], 4
            )
            for name, members in cells.items()
        }
    report["full_image_loo"] = full
    print("full-image LOO (dense waiting frames):", json.dumps(full, indent=1), flush=True)

    # 2. patch leak maps -------------------------------------------------------
    patch_maps = {}
    for cam, eps in cams.items():
        cells = _cells(side, task, eps)
        for name in ("banana", "box"):
            members = cells[name]
            grid = np.zeros((60 // PATCH, 80 // PATCH))
            for pi in range(grid.shape[0]):
                for pj in range(grid.shape[1]):
                    feats = [
                        eps[e][:, pi * PATCH : (pi + 1) * PATCH, pj * PATCH : (pj + 1) * PATCH].reshape(len(eps[e]), -1)
                        for e in members
                    ]
                    grid[pi, pj] = ridge_loo(feats, [side[e] for e in members], 1e-1)[1]
            patch_maps[f"{cam}_{name}"] = grid
            print(f"patch map {cam}/{name}: max={grid.max():.2f} @ {np.unravel_index(grid.argmax(), grid.shape)}", flush=True)
    np.savez(OUT_ROOT / "patch_maps.npz", **patch_maps)
    report["patch_map_max"] = {k: {"max_acc": round(float(v.max()), 3), "argmax_rowcol": [int(x) for x in np.unravel_index(v.argmax(), v.shape)]} for k, v in patch_maps.items()}

    # 3. top-camera region probes ----------------------------------------------
    regions = {}
    eps = cams["top"]
    cells = _cells(side, task, eps)
    for rname, (rs, cs) in TOP_REGIONS.items():
        regions[rname] = {
            name: round(ridge_loo([eps[e][:, rs, cs].reshape(len(eps[e]), -1) for e in members], [side[e] for e in members], 1e-1)[1], 4)
            for name, members in cells.items()
        }
    report["top_region_loo"] = regions
    print("top-camera region LOO:", json.dumps(regions, indent=1), flush=True)

    # 4. collection-drift nearest neighbours -----------------------------------
    nn = {}
    for cam, eps in cams.items():
        train = [e for e in sorted(eps) if e not in HELDOUT]
        means = {e: eps[e].mean(axis=0).ravel() for e in train}
        same_side = same_task_adjacent = 0
        pairs = {}
        for e in train:
            others = [o for o in train if o != e and task[o] == task[e]]
            dists = [(float(np.linalg.norm(means[e] - means[o])), o) for o in others]
            _, best = min(dists)
            pairs[e] = best
            same_side += int(side[best] == side[e])
            same_task_adjacent += int(abs(best - e) <= 2)
        nn[cam] = {
            "frac_nn_same_side": round(same_side / len(train), 3),
            "frac_nn_adjacent_in_collection": round(same_task_adjacent / len(train), 3),
        }
    report["nearest_neighbour"] = nn
    print("episode nearest neighbours:", json.dumps(nn, indent=1), flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(OUT_ROOT / "report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    return report


# ---------------------------------------------------------------------------
# Stage: figures
# ---------------------------------------------------------------------------


def stage_figures() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    cams, side, task = load_cameras()
    maps = dict(np.load(OUT_ROOT / "patch_maps.npz"))
    fig_dir = OUT_ROOT / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # patch leak maps over the mean image, one row per camera, tasks as columns
    for tname in ("banana", "box"):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
        for ax, cam in zip(axes, ("top", "wristL", "wristR"), strict=False):
            members = [e for e in sorted(cams[cam]) if e not in HELDOUT and task[e] == tname]
            mean_img = np.mean([cams[cam][e].mean(axis=0) for e in members], axis=0)
            grid = maps[f"{cam}_{tname}"]
            up = np.kron(grid, np.ones((PATCH, PATCH)))
            ax.imshow(mean_img, cmap="gray", vmin=0, vmax=1)
            im = ax.imshow(up, cmap="inferno", vmin=0.5, vmax=1.0, alpha=0.55)
            ax.set_title(f"{cam}  (max {grid.max():.2f})", fontsize=10)
            ax.axis("off")
        fig.colorbar(im, ax=axes, fraction=0.03, label="LOO side accuracy of this 10x10 patch alone")
        fig.suptitle(f"{tname}: WHERE each camera leaks -- per-patch LOO side probes on waiting frames (chance 0.5)", fontsize=11)
        fig.savefig(fig_dir / f"patch_leak_{tname}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

    # wrist mean-difference heatmaps (the top-camera version lives in the audit figures)
    for cam in ("wristL", "wristR"):
        fig, axes = plt.subplots(2, 3, figsize=(13, 6.8))
        for r, tname in enumerate(("banana", "box")):
            members = [e for e in sorted(cams[cam]) if e not in HELDOUT and task[e] == tname]
            mean_l = np.mean([cams[cam][e].mean(axis=0) for e in members if side[e] == 0], axis=0)
            mean_r = np.mean([cams[cam][e].mean(axis=0) for e in members if side[e] == 1], axis=0)
            axes[r, 0].imshow(mean_l, cmap="gray", vmin=0, vmax=1)
            axes[r, 0].set_title(f"{tname}: mean waiting frame, target LEFT", fontsize=9)
            axes[r, 1].imshow(mean_r, cmap="gray", vmin=0, vmax=1)
            axes[r, 1].set_title("target RIGHT", fontsize=9)
            im = axes[r, 2].imshow(np.abs(mean_l - mean_r), cmap="inferno")
            axes[r, 2].set_title("|L - R| (leaking pixels)", fontsize=9)
            fig.colorbar(im, ax=axes[r, 2], fraction=0.04)
            for ax in axes[r]:
                ax.axis("off")
        fig.suptitle(f"{cam}: mean waiting frames by side", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / f"meandiff_{cam}.png", dpi=140)
        plt.close(fig)

    print(f"figures written to {fig_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("extract", "analyze", "figures", "all"), default="all")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage in ("extract", "all"):
        stage_extract(args.workers)
    if args.stage in ("analyze", "all"):
        stage_analyze()
    if args.stage in ("figures", "all"):
        stage_figures()


if __name__ == "__main__":
    main()
