"""Render attention heatmaps over a recorded demo and encode as MP4.

For every frame of a demo:
  1. Run AttentionExtractor on (3 images + state).
  2. Average attention across a configurable layer band + heads + groups.
  3. Slice query × per-camera image-key. Queries are either the 10 action tokens
     (default — "what does the action expert attend to") or all prompt tokens
     (`--queries prompt` — "what does the language attend to", which is where
     PaliGemma's language→image grounding lives).
  4. Reshape (256,) -> (16, 16), upsample to native (480, 640) per camera.
  5. Overlay heatmap on the recorded camera frame.
  6. Compose 1x3 panel (top | left | right) and append to MP4.

Usage:
    cd /home/kewalk/memory_project
    python eval/attention_viz.py --demo demo1                       # action-query default
    python eval/attention_viz.py --demo demo1 --queries prompt      # language→image
    python eval/attention_viz.py --demo demo1 --max-steps 20        # smoke test
    python eval/attention_viz.py --demo demo1 --layers 14,15,16,17  # custom layer band
"""

import argparse
import os
import pathlib
import sys
import time

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

# Allow direct script execution: `python eval/attention_viz.py` from the project root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention_extract import AttentionExtractor  # noqa: E402

DATASET_ROOT = pathlib.Path("/home/kewalk/memory_project/dataset/CUP_TASK_0428")
EVAL_DIR = pathlib.Path("/home/kewalk/memory_project/eval")
FIG_DIR = EVAL_DIR / "figs"
PATCH_GRID = 16  # 16x16 = 256 patches per camera (verified in extractor smoke test)
N_PATCHES = PATCH_GRID * PATCH_GRID
H, W = 480, 640


def load_demo(demo_dir: pathlib.Path):
    state = np.load(demo_dir / "left_joint_positions.npy").astype(np.float32)
    top = iio.imread(str(demo_dir / "top_camera_rgb.mp4"), plugin="pyav").astype(np.uint8)
    left = iio.imread(str(demo_dir / "left_camera_rgb.mp4"), plugin="pyav").astype(np.uint8)
    right = iio.imread(str(demo_dir / "right_camera_rgb.mp4"), plugin="pyav").astype(np.uint8)
    T = min(len(state), len(top), len(left), len(right))
    return state[:T], top[:T], left[:T], right[:T]


def aggregate_attention(attn_probs: np.ndarray, layer_indices: list[int],
                         query_range: tuple[int, int],
                         image_range: tuple[int, int]) -> np.ndarray:
    """attn_probs: (L, B, K, G, T_q, T_k). Returns (n_patches,) heatmap."""
    a = np.asarray(attn_probs).astype(np.float32)[layer_indices]
    a = a.mean(axis=(0, 2, 3))                   # (B, T_q, T_k)
    a = a[0]                                      # (T_q, T_k)
    q_lo, q_hi = query_range
    ik_lo, ik_hi = image_range
    sub = a[q_lo:q_hi, ik_lo:ik_hi]              # (n_q, n_patches)
    return sub.mean(axis=0)                      # (n_patches,)


def heatmap_to_image(heat_flat: np.ndarray, target_h: int = H, target_w: int = W) -> np.ndarray:
    """(256,) -> (H, W) normalized [0, 1] heatmap."""
    grid = heat_flat.reshape(PATCH_GRID, PATCH_GRID)
    factor_h = target_h / PATCH_GRID
    factor_w = target_w / PATCH_GRID
    big = zoom(grid, (factor_h, factor_w), order=1)
    big = (big - big.min()) / max(big.max() - big.min(), 1e-8)
    return big


def render_panel(top_rgb, left_rgb, right_rgb, heat_top, heat_left, heat_right,
                 step: int, query_source: str = "actions",
                 alpha: float = 0.55) -> np.ndarray:
    """Returns an (H_out, W_out, 3) uint8 image with the 3-camera overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cmap = plt.get_cmap("jet")
    for ax, img, heat, title in zip(
        axes,
        [top_rgb, left_rgb, right_rgb],
        [heat_top, heat_left, heat_right],
        ["top (base_0_rgb)", "left (left_wrist_0_rgb)", "right (right_wrist_0_rgb)"],
    ):
        ax.imshow(img)
        ax.imshow(heat, cmap=cmap, alpha=alpha)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"step {step}: {query_source}→image attention", fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", default="demo1")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--layers", default="14,15,16,17",
                        help="Comma-separated layer indices to average (default last 4 of 18).")
    parser.add_argument("--queries", choices=["prompt", "actions"], default="actions",
                        help="Slice attention from prompt-token queries or action-token "
                             "queries. 'actions' (default) shows what the action heads "
                             "attend to (often gripper/arm-dominant). 'prompt' shows what "
                             "the language tokens attend to (PaliGemma's language→image "
                             "grounding — should localize on the cup).")
    parser.add_argument("--prompt", default=None,
                        help="Override the language prompt (default: extractor's built-in "
                             "'pick up the cup with yellow object inside'). Useful for "
                             "comparing how different referring expressions shift attention.")
    parser.add_argument("--time", type=float, default=0.0,
                        help="Denoise time t in [0,1]; 0 = clean (data manifold).")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--infer-every", type=int, default=10,
                        help="Run policy inference once every K frames; reuse the "
                             "heatmap for the K-1 in-between frames. Default 10 "
                             "(matches action_horizon → ~10x speedup).")
    args = parser.parse_args()

    layer_indices = [int(s) for s in args.layers.split(",")]
    print(f"[layers] using {layer_indices} (avg)")
    print(f"[queries] slicing from {args.queries} tokens")

    demo_dir = DATASET_ROOT / args.demo
    print(f"[load] {demo_dir}")
    state, top, left, right = load_demo(demo_dir)
    T = len(state)
    if args.max_steps is not None:
        T = min(args.max_steps, T)
        state, top, left, right = state[:T], top[:T], left[:T], right[:T]
    print(f"[load] {T} frames")

    print("[init] AttentionExtractor")
    extractor = AttentionExtractor(prompt=args.prompt) if args.prompt else AttentionExtractor()
    print(f"[prompt] {extractor._prompt!r}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_q-{args.queries}"
    if args.prompt:
        slug = "".join(c if c.isalnum() else "-" for c in args.prompt.lower()).strip("-")[:40]
        suffix += f"_p-{slug}"
    out_path = FIG_DIR / f"attention_{args.demo}{suffix}.mp4"

    # Try to load saved predictions to use as action_seed (cleaner attention).
    results_path = EVAL_DIR / f"results_{args.demo}.npz"
    pred_first = None
    if results_path.exists():
        pred = np.load(results_path)["pred_first"]
        if len(pred) >= T:
            pred_first = pred[:T]
            print(f"[seed] using offline_eval pred_first as action_seed")

    K = max(1, args.infer_every)
    n_infer = (T + K - 1) // K
    print(f"[plan] inferring every {K} frames -> {n_infer} inferences for {T} output frames")

    frames = []
    t0 = time.time()
    action_dim_padded = extractor._action_dim  # model's padded action dim (e.g. 32)
    big_top = big_left = big_right = None  # cached heatmaps (held between inferences)
    pbar = tqdm(range(T), desc="attention", unit="frame", dynamic_ncols=True)
    for t in pbar:
        if t % K == 0:
            seed = None
            if pred_first is not None:
                seed = np.zeros((extractor._action_horizon, action_dim_padded), dtype=np.float32)
                seed[:, : pred_first.shape[1]] = pred_first[t]
            out = extractor.extract(top[t], left[t], right[t], state[t],
                                    action_seed=seed, time=args.time)
            attn = out["attn_probs"]
            b = out["boundaries"]
            q_range = b.prompt if args.queries == "prompt" else b.actions
            big_top = heatmap_to_image(aggregate_attention(attn, layer_indices, q_range, b.img_top))
            big_left = heatmap_to_image(aggregate_attention(attn, layer_indices, q_range, b.img_left))
            big_right = heatmap_to_image(aggregate_attention(attn, layer_indices, q_range, b.img_right))
        # Render every frame, reusing the most recent heatmap (a "held" overlay).
        panel = render_panel(top[t], left[t], right[t], big_top, big_left, big_right,
                             step=t, query_source=args.queries)
        frames.append(panel)
    pbar.close()
    print(f"[infer+render] {time.time()-t0:.1f}s for {T} frames ({n_infer} inferences)")

    print(f"[encode] {out_path}")
    iio.imwrite(str(out_path), np.stack(frames), fps=args.fps, codec="libx264", quality=8)
    print(f"[done] wrote {len(frames)} frames in {time.time()-t0:.1f}s -> {out_path}")


if __name__ == "__main__":
    main()
