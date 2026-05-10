"""Render attention heatmaps over a LONG_TASK_1 demo and encode as MP4.

Long-task variant of `attention_viz.py`. Differences:
  - Loads top + left only; constructs the memory image per frame from the
    observe-stage keyframe (broadcast on stage_id ∈ {3,4,5,6}, zeros elsewhere).
  - The right panel is now the **memory image**.
  - Uses per-frame instruction (`instruction.npy`) as the prompt, so the
    prompt→image attention reflects the actual query in flight.
  - Stage label appears in the figure suptitle.

Usage:
    cd /home/kewalk/memory_project
    # Recommended: language→image grounding diagnostic
    python eval/attention_viz_long.py --demo demo36 --queries prompt --layers 8,9,10,11

    # Smoke run on the first 50 frames
    python eval/attention_viz_long.py --demo demo36 --queries prompt --max-steps 50

    # Action-query attention (what the action heads attend to)
    python eval/attention_viz_long.py --demo demo36 --queries actions --layers 14,15,16,17
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
from scipy.ndimage import zoom
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention_extract import AttentionExtractor  # noqa: E402

DATASET_ROOT = pathlib.Path("/home/kewalk/memory_project/dataset/LONG_TASK_1")
EVAL_DIR = pathlib.Path("/home/kewalk/memory_project/eval")
FIG_DIR = EVAL_DIR / "figs"
LONG_CKPT_ROOT = pathlib.Path(
    "/home/kewalk/memory_project/openpi/checkpoints/pi05_long_task_mem_lora/long_v1"
)
LONG_CONFIG = "pi05_long_task_mem_lora"
PATCH_GRID = 16
N_PATCHES = PATCH_GRID * PATCH_GRID
H, W = 480, 640
QUERY_STAGES = {3, 4, 5, 6}
STAGE_LABELS = {0: "observe", 1: "mix", 2: "delay",
                3: "query1", 4: "query2", 5: "query3", 6: "query4",
                7: "return"}


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


def aggregate_attention(attn_probs, layer_indices, q_range, image_range):
    a = np.asarray(attn_probs).astype(np.float32)[layer_indices]
    a = a.mean(axis=(0, 2, 3))[0]                     # (T_q, T_k)
    q_lo, q_hi = q_range; ik_lo, ik_hi = image_range
    return a[q_lo:q_hi, ik_lo:ik_hi].mean(axis=0)     # (n_patches,)


def heatmap_to_image(heat_flat):
    grid = heat_flat.reshape(PATCH_GRID, PATCH_GRID)
    big = zoom(grid, (H / PATCH_GRID, W / PATCH_GRID), order=1)
    big = (big - big.min()) / max(big.max() - big.min(), 1e-8)
    return big


def render_panel(top_rgb, left_rgb, mem_rgb, heat_top, heat_left, heat_mem,
                 step, stage_label, instruction_text, query_source, has_mem,
                 alpha=0.55):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cmap = plt.get_cmap("jet")
    titles = ["top (base_0_rgb)", "left wrist", f"memory ({'on' if has_mem else 'off'})"]
    for ax, img, heat, title in zip(axes, [top_rgb, left_rgb, mem_rgb],
                                    [heat_top, heat_left, heat_mem], titles):
        ax.imshow(img)
        if heat is not None:
            ax.imshow(heat, cmap=cmap, alpha=alpha)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    title_text = (f"step {step} [{stage_label}] {query_source}→image attention\n"
                  f"\"{instruction_text}\"")
    fig.suptitle(title_text, fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", default="demo36")
    parser.add_argument("--ckpt-step", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--layers", default="8,9,10,11",
                        help="Comma-separated layer indices to average (default 8-11, language→image band).")
    parser.add_argument("--queries", choices=["prompt", "actions"], default="prompt")
    parser.add_argument("--time", type=float, default=0.0,
                        help="Denoise time t in [0,1]; 0 = clean (data manifold).")
    parser.add_argument("--fps", type=int, default=None,
                        help="Output fps. Default: 30 (full video) or 2 (slideshow when "
                             "--per-stage-frames is set).")
    parser.add_argument("--infer-every", type=int, default=10,
                        help="Run inference every K frames; reuse the heatmap for the K-1 between.")
    parser.add_argument("--per-stage-frames", type=int, default=None,
                        help="If set, only run inference on the first N frames after each stage "
                             "transition (e.g. --per-stage-frames 3 → ~24 inferences total). The "
                             "heatmap is held for the rest of each stage so the video still plays "
                             "at full length but updates only at stage starts.")
    args = parser.parse_args()

    layer_indices = [int(s) for s in args.layers.split(",")]
    print(f"[layers] {layer_indices}, queries={args.queries}")

    demo_dir = DATASET_ROOT / args.demo
    ckpt_dir = LONG_CKPT_ROOT / str(args.ckpt_step)
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

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / f"attention_long_{args.demo}_q-{args.queries}.mp4"

    # When sampling per-stage, dump each panel as its own PNG into a subdir
    # (much easier to flip through than a 24-frame video).
    if args.per_stage_frames is not None:
        png_dir = FIG_DIR / f"attention_long_{args.demo}_q-{args.queries}"
        png_dir.mkdir(parents=True, exist_ok=True)
    else:
        png_dir = None

    K = max(1, args.infer_every)
    zero_image = np.zeros((H, W, 3), dtype=np.uint8)

    big_top = big_left = big_mem = None
    cur_mem_img = zero_image
    cur_has_mem = False

    frames = []
    prev_stage = -1
    in_stage_count = 0
    n_inferences = 0
    t0 = time.time()
    pbar = tqdm(range(T), desc="attention", unit="frame", dynamic_ncols=True)
    for t in pbar:
        cur_stage = int(stage_id[t])
        if cur_stage != prev_stage:
            in_stage_count = 0
            prev_stage = cur_stage

        in_query = cur_stage in QUERY_STAGES
        mem_img = keyframe if in_query else zero_image

        if args.per_stage_frames is not None:
            # Sample the first N frames after each stage transition. Ignore the
            # --infer-every modulo here — otherwise stages whose start frame
            # isn't divisible by K get filtered out entirely.
            run_now = (in_stage_count < args.per_stage_frames)
        else:
            run_now = (t % K == 0)

        if run_now:
            out = extractor.extract_long(
                top=top[t], left=left[t], memory=mem_img, has_memory=in_query,
                state=state[t], prompt=str(instruction[t]), time=args.time,
            )
            attn = out["attn_probs"]
            b = out["boundaries"]
            q_range = b.prompt if args.queries == "prompt" else b.actions
            big_top = heatmap_to_image(aggregate_attention(attn, layer_indices, q_range, b.img_top))
            big_left = heatmap_to_image(aggregate_attention(attn, layer_indices, q_range, b.img_left))
            # If memory is off, the heatmap will be near-zero (image embedding masked).
            big_mem = heatmap_to_image(aggregate_attention(attn, layer_indices, q_range, b.img_right))
            cur_mem_img = mem_img
            cur_has_mem = in_query
            n_inferences += 1
        in_stage_count += 1

        # When --per-stage-frames is set, only emit the sampled frames (short
        # slideshow). Otherwise render every frame with the held heatmap.
        if args.per_stage_frames is not None and not run_now:
            continue

        panel = render_panel(
            top[t], left[t], cur_mem_img, big_top, big_left, big_mem,
            step=t, stage_label=STAGE_LABELS.get(int(stage_id[t]), "?"),
            instruction_text=str(instruction[t])[:90],
            query_source=args.queries, has_mem=cur_has_mem,
        )
        frames.append(panel)

        # In per-stage mode, save each rendered panel as its own PNG.
        if png_dir is not None:
            stage_name = STAGE_LABELS.get(int(stage_id[t]), "stage")
            iio.imwrite(
                str(png_dir / f"t{t:05d}_{stage_name}_n{in_stage_count - 1}.png"),
                panel,
            )
    pbar.close()
    print(f"[infer+render] {time.time()-t0:.1f}s ({n_inferences} inferences)")

    if png_dir is not None:
        print(f"[done] wrote {len(frames)} PNGs -> {png_dir}/")
    else:
        out_fps = args.fps if args.fps is not None else 30
        print(f"[encode] {out_path}  ({len(frames)} frames @ {out_fps} fps)")
        iio.imwrite(str(out_path), np.stack(frames), fps=out_fps, codec="libx264", quality=8)
        print(f"[done] wrote {len(frames)} frames -> {out_path}")


if __name__ == "__main__":
    main()
