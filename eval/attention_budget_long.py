"""Per-frame prompt-attention budget across top / left / memory cameras.

For each frame of a held-out LONG_TASK_1 demo, runs one forward pass of the
trained policy and slices the prompt-token attention onto each image patch
range. Reports the *fraction* of prompt attention mass that lands on each
camera, per frame. The interesting signal is whether `mem` rises during query
stages (3-6) — if it does, the policy is using the memory channel.

Saved outputs:
    eval/results_long_<demo>_attn.npz         per-frame fractions + stage_id
    eval/figs/attention_budget_<demo>.png     stacked time-series colored by stage

Usage:
    cd /home/kewalk/memory_project
    python eval/attention_budget_long.py --demo demo36
    python eval/attention_budget_long.py --demo demo36 --layers 8,9,10,11
    python eval/attention_budget_long.py --demo demo36 --infer-every 5      # speedup
"""

import argparse
import json
import os
import pathlib
import sys
import time

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention_extract import AttentionExtractor  # noqa: E402

DATASET_ROOT = pathlib.Path("/home/kewalk/memory_project/dataset/LONG_TASK_1")
EVAL_DIR = pathlib.Path("/home/kewalk/memory_project/eval")
FIG_DIR = EVAL_DIR / "figs"
LONG_CKPT = pathlib.Path(
    "/home/kewalk/memory_project/openpi/checkpoints/pi05_long_task_mem_lora/long_v1/5000"
)
LONG_CONFIG = "pi05_long_task_mem_lora"
QUERY_STAGES = {3, 4, 5, 6}
H, W = 480, 640
STAGE_LABELS = {0: "observe", 1: "mix", 2: "delay",
                3: "query1", 4: "query2", 5: "query3", 6: "query4",
                7: "return"}
STAGE_COLORS = {
    0: "#dddddd",
    1: "#bbbbbb",
    2: "#aaaaaa",
    3: "#fdd",
    4: "#fcc",
    5: "#fbb",
    6: "#faa",
    7: "#dcdce4",
}


def load_demo(demo_dir: pathlib.Path):
    meta = json.loads((demo_dir / "metadata.json").read_text())
    state = np.load(demo_dir / "left_joint_positions.npy").astype(np.float32)
    instruction = np.load(demo_dir / "instruction.npy", allow_pickle=True)
    stage_id = np.load(demo_dir / "stage_id.npy").astype(np.int32)
    top = iio.imread(str(demo_dir / "top_camera_rgb.mp4"), plugin="pyav").astype(np.uint8)
    left = iio.imread(str(demo_dir / "left_camera_rgb.mp4"), plugin="pyav").astype(np.uint8)

    T = min(len(state), len(instruction), len(stage_id), len(top), len(left))
    state = state[:T]; instruction = instruction[:T]; stage_id = stage_id[:T]
    top = top[:T]; left = left[:T]
    observe_seg = next(s for s in meta["segments"] if s["name"] == "observe")
    keyframe_idx = min(observe_seg["end_step"] - 1, T - 1)
    keyframe = top[keyframe_idx].copy()
    return T, state, instruction, stage_id, top, left, keyframe


def attention_fractions(attn_probs, layer_indices, b, q_lo, q_hi):
    """Return (frac_top, frac_left, frac_mem) — averaged prompt-attention sums.

    attn_probs: (L, B=1, K, G, T_q, T_k). Average over selected layers, heads, groups.
    Sum the keys over each camera's patch range, then mean over the query rows.
    """
    a = np.asarray(attn_probs).astype(np.float32)[layer_indices]
    a = a.mean(axis=(0, 2, 3))[0]              # (T_q, T_k)
    sub = a[q_lo:q_hi]                          # (n_q, T_k)
    sums = {}
    for cam, (lo, hi) in [("top", b.img_top), ("left", b.img_left), ("mem", b.img_right)]:
        sums[cam] = float(sub[:, lo:hi].sum(axis=-1).mean())
    return sums


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", default="demo36")
    parser.add_argument("--ckpt-step", type=int, default=5000)
    parser.add_argument("--layers", default="8,9,10,11",
                        help="Comma-separated layer indices to average (default 8-11, language→image band).")
    parser.add_argument("--infer-every", type=int, default=1,
                        help="Run inference every K frames (default 1 — no skip).")
    parser.add_argument("--per-stage-frames", type=int, default=None,
                        help="If set, only run inference on the first N frames after each stage "
                             "transition (e.g. --per-stage-frames 3 → ~24 inferences total instead "
                             "of 3000+). Held over for the rest of each stage in the plot.")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    layer_indices = [int(s) for s in args.layers.split(",")]
    print(f"[layers] {layer_indices}")

    demo_dir = DATASET_ROOT / args.demo
    ckpt_dir = LONG_CKPT.parent / str(args.ckpt_step)
    print(f"[load] demo {demo_dir}")
    print(f"[load] ckpt {ckpt_dir}")

    T, state, instruction, stage_id, top, left, keyframe = load_demo(demo_dir)
    if args.max_steps is not None:
        T = min(args.max_steps, T)
        state, instruction, stage_id, top, left = (
            state[:T], instruction[:T], stage_id[:T], top[:T], left[:T]
        )
    print(f"[load] {T} frames")

    print(f"[init] AttentionExtractor (long-task)")
    extractor = AttentionExtractor(ckpt_dir=ckpt_dir, config_name=LONG_CONFIG)

    zero_image = np.zeros((H, W, 3), dtype=np.uint8)
    fractions = np.zeros((T, 3), dtype=np.float32)  # [top, left, mem]
    last_frac = np.zeros(3, dtype=np.float32)
    K = max(1, args.infer_every)

    prev_stage = -1
    in_stage_count = 0
    n_inferences = 0
    t0 = time.time()
    for t in tqdm(range(T), desc="attn-budget", unit="frame", dynamic_ncols=True):
        cur_stage = int(stage_id[t])
        if cur_stage != prev_stage:
            in_stage_count = 0
            prev_stage = cur_stage

        run_now = (t % K == 0)
        if args.per_stage_frames is not None:
            run_now = run_now and (in_stage_count < args.per_stage_frames)

        if run_now:
            in_query = cur_stage in QUERY_STAGES
            mem_img = keyframe if in_query else zero_image
            out = extractor.extract_long(
                top=top[t], left=left[t], memory=mem_img,
                has_memory=in_query, state=state[t], prompt=str(instruction[t]),
            )
            b = out["boundaries"]
            q_lo, q_hi = b.prompt
            sums = attention_fractions(out["attn_probs"], layer_indices, b, q_lo, q_hi)
            total = sums["top"] + sums["left"] + sums["mem"] + 1e-12
            last_frac = np.array([sums["top"]/total, sums["left"]/total, sums["mem"]/total],
                                 dtype=np.float32)
            n_inferences += 1
        in_stage_count += 1
        fractions[t] = last_frac
    print(f"[done] {n_inferences} inferences in {time.time()-t0:.1f}s "
          f"({n_inferences / max(time.time()-t0, 1e-3):.2f} Hz)")

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = EVAL_DIR / f"results_long_{args.demo}_attn.npz"
    np.savez(out_npz, fractions=fractions, stage_id=stage_id)
    print(f"[save] {out_npz}")

    # Plot ---------------------------------------------------------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                              gridspec_kw={"height_ratios": [1, 4]})

    # Top strip: stage colour bar
    ax_stage = axes[0]
    prev = -1
    seg_start = 0
    for t in range(T):
        if int(stage_id[t]) != prev:
            if prev != -1:
                ax_stage.axvspan(seg_start, t, color=STAGE_COLORS.get(prev, "#eee"))
                ax_stage.text((seg_start + t) / 2, 0.5, STAGE_LABELS.get(prev, "?"),
                              ha="center", va="center", fontsize=8)
            seg_start = t
            prev = int(stage_id[t])
    ax_stage.axvspan(seg_start, T, color=STAGE_COLORS.get(prev, "#eee"))
    ax_stage.text((seg_start + T) / 2, 0.5, STAGE_LABELS.get(prev, "?"),
                  ha="center", va="center", fontsize=8)
    ax_stage.set_ylim(0, 1); ax_stage.set_yticks([]); ax_stage.set_xlim(0, T)
    ax_stage.set_title(f"{args.demo} — prompt-attention budget per camera "
                        f"(layers {args.layers}, ckpt {args.ckpt_step})")

    # Bottom: stacked area
    ax = axes[1]
    ts = np.arange(T)
    ax.fill_between(ts, 0, fractions[:, 0], color="#1f77b4", alpha=0.85, label="top")
    ax.fill_between(ts, fractions[:, 0], fractions[:, 0] + fractions[:, 1],
                     color="#2ca02c", alpha=0.85, label="left wrist")
    ax.fill_between(ts, fractions[:, 0] + fractions[:, 1], 1.0,
                     color="#d62728", alpha=0.85, label="memory")
    # Vertical lines at stage transitions
    for t in range(1, T):
        if stage_id[t] != stage_id[t-1]:
            ax.axvline(t, color="black", lw=0.5, alpha=0.4)
    ax.set_xlabel("frame"); ax.set_ylabel("fraction of prompt attention")
    ax.set_xlim(0, T); ax.set_ylim(0, 1)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig_path = FIG_DIR / f"attention_budget_{args.demo}.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)
    print(f"[save] {fig_path}")

    # Per-stage summary --------------------------------------------------
    print("\n[summary] mean fraction per stage:")
    print(f"  {'stage':<12} {'count':>6}  {'top':>8}  {'left':>8}  {'mem':>8}")
    for sid in sorted(STAGE_LABELS.keys()):
        mask = stage_id == sid
        if mask.any():
            mt, ml, mm = fractions[mask].mean(axis=0)
            print(f"  {STAGE_LABELS[sid]:<12} {int(mask.sum()):>6}  "
                  f"{mt:>8.3f}  {ml:>8.3f}  {mm:>8.3f}")


if __name__ == "__main__":
    main()
