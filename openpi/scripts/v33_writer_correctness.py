"""Is the v3.3 writer paying attention to the CORRECT thing?

Grounds the writer-attention maps of a finished v33_writer_attention run in the scene geometry:
the two bins sit at fixed patch regions of the letterboxed top camera (calibrated visually on
the 0816 rig -- rows 6-9, left bin cols 5-7, right bin cols 8-10).
Both objects are always present, one per bin, so the instruction factorial makes a sharp
prediction: under the TRUE instruction the writer should favor the bin holding the instructed
object, and under the COUNTERFACTUAL instruction the *other* bin.

Reported per episode and pooled, for the evidence and waiting phases:

* bins_focus      -- attention fraction inside both bin regions (uniform baseline: 24/256);
* target_pref     -- target-bin minus distractor-bin attention under TRUE;
* steering        -- target-bin attention TRUE minus CF (does the instruction pull attention
                     toward its own object?), plus the per-frame win rate;
* interaction     -- (target-distractor | TRUE) - (target-distractor | CF), the clean
                     both-directions steering score;
* per-slot ranking of evidence-phase target preference (which slots specialize).

CPU only: consumes the NPZ maps + summary.json of one or more run dirs (pass several to see
the trend across checkpoints). Optional --stills renders mid-evidence overlays with the bin
boxes drawn for visual confirmation.
"""

# ruff: noqa: I001 - openpi.diagnostics.v33_writer_attention pins the load-bearing
# pyarrow-before-openpi import order; import it first.

import dataclasses
import json
from pathlib import Path

import numpy as np
import tyro


# Patch-region calibration (16x16 grid over the 224x224 letterboxed top camera). Verified
# against evidence frames of episodes 0 and 30 with per-patch renders. Beware the transparent
# bin LIDS lying on the table left/right of the bins during the open phases -- an earlier
# cols-4..6 left box captured the lid, not the bin.
BIN_ROWS = (6, 9)  # inclusive
LEFT_BIN_COLS = (5, 7)
RIGHT_BIN_COLS = (8, 10)


def _region_indices(rows: tuple[int, int], cols: tuple[int, int]) -> np.ndarray:
    grid_rows, grid_cols = np.meshgrid(np.arange(rows[0], rows[1] + 1), np.arange(cols[0], cols[1] + 1), indexing="ij")
    return (grid_rows * 16 + grid_cols).ravel()


LEFT_IDX = _region_indices(BIN_ROWS, LEFT_BIN_COLS)
RIGHT_IDX = _region_indices(BIN_ROWS, RIGHT_BIN_COLS)


@dataclasses.dataclass(frozen=True)
class Options:
    run_dirs: tuple[Path, ...]
    stills: bool = False
    dataset_root: Path | None = None  # required only with --stills


def _region_mass(maps: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """[T, 16, 256] -> [T] fraction of total attention mass inside the region."""
    total = maps.sum(axis=(1, 2))
    return maps[:, :, indices].sum(axis=(1, 2)) / np.maximum(total, 1e-12)


def _slot_region_mass(maps: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """[T, 16, 256] -> [T, 16] per-slot region fraction (each slot row sums to 1)."""
    return maps[:, :, indices].sum(axis=2)


def analyze_episode(entry: dict, maps: dict[str, np.ndarray]) -> dict:
    frames = entry["frames"]
    phases = np.asarray([f["phase"] for f in frames])
    side = entry["side"]
    target_idx, distractor_idx = (LEFT_IDX, RIGHT_IDX) if side == "left" else (RIGHT_IDX, LEFT_IDX)

    mass = {
        (variant, name): _region_mass(maps[variant], idx)
        for variant in ("true", "cf", "base")
        for name, idx in (("target", target_idx), ("distractor", distractor_idx))
    }
    out = {"episode": entry["episode"], "prompt": entry["prompt"], "side": side, "phases": {}}
    for phase in ("evidence", "waiting"):
        sel = phases == phase
        if not sel.any():
            continue
        t_true = mass[("true", "target")][sel]
        d_true = mass[("true", "distractor")][sel]
        t_cf = mass[("cf", "target")][sel]
        d_cf = mass[("cf", "distractor")][sel]
        out["phases"][phase] = {
            "frames": int(sel.sum()),
            "bins_focus_true": float((t_true + d_true).mean()),
            "bins_focus_base": float((mass[("base", "target")][sel] + mass[("base", "distractor")][sel]).mean()),
            "target_pref_true": float((t_true - d_true).mean()),
            "steering": float((t_true - t_cf).mean()),
            "steering_win_rate": float((t_true > t_cf).mean()),
            "interaction": float(((t_true - d_true) - (t_cf - d_cf)).mean()),
        }
    # per-slot evidence-phase target preference under TRUE
    sel = phases == "evidence"
    if sel.any():
        slot_t = _slot_region_mass(maps["true"], target_idx)[sel].mean(axis=0)
        slot_d = _slot_region_mass(maps["true"], distractor_idx)[sel].mean(axis=0)
        out["slot_target_pref_evidence"] = (slot_t - slot_d).round(4).tolist()
    return out


def analyze_run(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "summary.json").read_text())
    episodes = []
    for entry in summary["episodes"]:
        npz = np.load(run_dir / f"episode_{entry['episode']:03d}_maps.npz")
        maps = {k: np.asarray(npz[k], dtype=np.float64) for k in ("true", "cf", "base")}
        episodes.append(analyze_episode(entry, maps))
    pooled = {}
    for phase in ("evidence", "waiting"):
        rows = [e["phases"][phase] for e in episodes if phase in e["phases"]]
        if rows:
            pooled[phase] = {key: float(np.mean([r[key] for r in rows])) for key in rows[0] if key != "frames"} | {
                "frames": int(np.sum([r["frames"] for r in rows]))
            }
    return {
        "run_dir": str(run_dir),
        "checkpoint": summary.get("checkpoint", "?"),
        "pathway_scalars": summary.get("pathway_scalars", {}),
        "episodes": episodes,
        "pooled": pooled,
    }


def _render_stills(run_dir: Path, dataset_root: Path, analyses: list[dict]) -> None:
    import cv2
    import pyarrow.parquet as pq

    from openpi.diagnostics import token_heatmap
    from openpi.diagnostics import writer_contribution as _wc

    summary = json.loads((run_dir / "summary.json").read_text())
    for entry in summary["episodes"]:
        npz = np.load(run_dir / f"episode_{entry['episode']:03d}_maps.npz")
        frames_idx = np.asarray(npz["frames"])
        phases = [f["phase"] for f in entry["frames"]]
        evidence_positions = [i for i, p in enumerate(phases) if p == "evidence"]
        if not evidence_positions:
            continue
        position = evidence_positions[len(evidence_positions) // 2]
        frame = int(frames_idx[position])
        source = _wc._load_lerobot_sources(dataset_root, [entry["episode"]])[0]  # noqa: SLF001
        by = {
            int(r["frame_index"]): r for r in pq.read_table(source.path, columns=["image", "frame_index"]).to_pylist()
        }
        raw = _wc.WriterContributionRunner._decode_inline_image(  # noqa: SLF001
            by[frame]["image"], field="image", raw_frame=frame
        )
        model_rgb, _ = token_heatmap.raw_top_camera_to_model_rgb(raw)
        pooled_map = np.asarray(npz["true"], dtype=np.float64)[position].sum(axis=0)
        vmax = max(float(np.percentile(pooled_map, 99.0)), 1e-12)
        scale = token_heatmap.ColorScale(
            vmin=0.0, vmax=vmax, lower_percentile=0.0, upper_percentile=99.0, anchor_zero=True
        )
        overlay, _ = token_heatmap.heatmap_overlay(model_rgb, pooled_map, scale, scale_mode="video")
        for (col_lo, col_hi), color in ((LEFT_BIN_COLS, (255, 60, 60)), (RIGHT_BIN_COLS, (60, 255, 60))):
            cv2.rectangle(
                overlay,
                (col_lo * 14, BIN_ROWS[0] * 14),
                ((col_hi + 1) * 14 - 1, (BIN_ROWS[1] + 1) * 14 - 1),
                color,
                1,
            )
        label = f"ep{entry['episode']} f{frame} target={entry['side']}"
        cv2.putText(overlay, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        big = cv2.resize(overlay, (448, 448), interpolation=cv2.INTER_NEAREST)
        path = run_dir / f"correctness_still_ep{entry['episode']:03d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
        print(f"wrote {path}")


def main(options: Options) -> None:
    analyses = []
    for run_dir in options.run_dirs:
        analysis = analyze_run(run_dir)
        analyses.append(analysis)
        print(f"\n=== {analysis['checkpoint']} ===")
        print(f"pathway: {analysis['pathway_scalars']}")
        header = (
            f"{'ep':>4} {'side':>5} {'phase':>9} {'bins_focus':>10} {'(Q0)':>6} "
            f"{'targ_pref':>9} {'steering':>9} {'win':>5} {'interact':>9}"
        )
        print(header)
        for episode in analysis["episodes"]:
            for phase, row in episode["phases"].items():
                print(
                    f"{episode['episode']:>4} {episode['side']:>5} {phase:>9} "
                    f"{row['bins_focus_true']:>10.3f} {row['bins_focus_base']:>6.3f} "
                    f"{row['target_pref_true']:>9.3f} {row['steering']:>9.3f} "
                    f"{row['steering_win_rate']:>5.2f} {row['interaction']:>9.3f}"
                )
        for phase, row in analysis["pooled"].items():
            print(
                f"{'ALL':>4} {'':>5} {phase:>9} {row['bins_focus_true']:>10.3f} "
                f"{row['bins_focus_base']:>6.3f} {row['target_pref_true']:>9.3f} "
                f"{row['steering']:>9.3f} {row['steering_win_rate']:>5.2f} {row['interaction']:>9.3f}"
            )
        out = Path(analysis["run_dir"]) / "correctness.json"
        out.write_text(json.dumps(analysis, indent=2))
        print(f"wrote {out}")
    if options.stills:
        if options.dataset_root is None:
            raise ValueError("--stills requires --dataset_root")
        _render_stills(options.run_dirs[-1], options.dataset_root, analyses)


if __name__ == "__main__":
    main(tyro.cli(Options))
