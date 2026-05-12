"""Build per-demo SigLIP feature cache for the keyframe retriever.

Decodes top_camera_rgb.mp4 at sampled keyframe indices, runs each frame
through PaliGemma's frozen SigLIP-So400m vision tower, and saves
(global_feature, patch_tokens) per demo as a MemoryBank .npz.

Usage:
    python -m retriever.features --demos 1-37 --out retriever/cache/features
    python -m retriever.features --demos 1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.shared import image_tools
from openpi.training import config as _config

from retriever.memory_bank import MemoryBank
from retriever.sampler import sample_keyframes

DEFAULT_CONFIG_NAME = "pi05_plate_task"
DEFAULT_CKPT_DIR = pathlib.Path(
    "/home/kewalk/memory_project/openpi/checkpoints/pi05_plate_task/plate_oracle_v2_0510/29999"
)
DATASET_ROOT = pathlib.Path("/home/kewalk/memory_project/dataset/PLATE_TASK_1")
IMG_SIZE = 224


def _parse_demos_arg(arg: str) -> list[int]:
    if "-" in arg and "," not in arg:
        a, b = arg.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in arg.split(",")]


def _load_top_frames(demo_dir: pathlib.Path, frame_ids: list[int]) -> np.ndarray:
    """Decode the top camera mp4 and return (K, H, W, 3) uint8 at the requested ids."""
    needed = set(frame_ids)
    max_id = max(frame_ids)
    by_id: dict[int, np.ndarray] = {}
    for i, frame in enumerate(iio.imiter(str(demo_dir / "top_camera_rgb.mp4"), plugin="pyav")):
        if i in needed:
            by_id[i] = np.asarray(frame, dtype=np.uint8)
        if i >= max_id:
            break
    missing = needed - by_id.keys()
    if missing:
        raise RuntimeError(f"Missing frames {sorted(missing)[:5]}... from {demo_dir}")
    return np.stack([by_id[fid] for fid in frame_ids], axis=0)


def _preprocess(images_u8: np.ndarray) -> jnp.ndarray:
    """uint8 (B, H, W, 3) -> float32 (B, 224, 224, 3) in [-1, 1]."""
    resized = image_tools.resize_with_pad(jnp.asarray(images_u8), IMG_SIZE, IMG_SIZE)
    return resized.astype(jnp.float32) / 255.0 * 2.0 - 1.0


def make_encoder(
    config_name: str = DEFAULT_CONFIG_NAME,
    ckpt_dir: pathlib.Path = DEFAULT_CKPT_DIR,
):
    """Build a JIT-compiled SigLIP encoder from a trained PaliGemma checkpoint.

    Returns a callable `encode(images_u8) -> (global_features, patch_tokens)` that
    accepts uint8 (B, H, W, 3) and returns numpy arrays of shape (B, D_IMG) and
    (B, P, D_IMG) where D_IMG matches the LM hidden size (2048 for Gemma 2B).
    """
    print(f"[load] config={config_name}  ckpt={ckpt_dir}")
    train_config = _config.get_config(config_name)
    policy = _policy_config.create_trained_policy(
        train_config, ckpt_dir, default_prompt="placeholder"
    )
    model = policy._model

    @jax.jit
    def _encode_jit(images_f32: jnp.ndarray):
        patches, _ = model.PaliGemma.img(images_f32, train=False)  # (B, P, D_IMG)
        globals_ = patches.mean(axis=1)
        return globals_, patches

    def encode(images_u8: np.ndarray):
        batch_f = _preprocess(images_u8)
        g, p = _encode_jit(batch_f)
        return np.asarray(g), np.asarray(p)

    return encode


def build(
    demos: list[int],
    out_dir: pathlib.Path,
    config_name: str = DEFAULT_CONFIG_NAME,
    ckpt_dir: pathlib.Path = DEFAULT_CKPT_DIR,
    batch_size: int = 32,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    encode = make_encoder(config_name, ckpt_dir)

    for d in demos:
        demo_dir = DATASET_ROOT / f"demo{d}"
        if not demo_dir.exists():
            print(f"[skip] demo{d}: not found")
            continue
        out_path = out_dir / f"demo{d}.npz"
        if out_path.exists():
            print(f"[skip] demo{d}: cache already exists at {out_path}")
            continue

        meta = json.loads((demo_dir / "metadata.json").read_text())
        num_steps = int(meta["num_steps"])
        instr = np.load(demo_dir / "instruction.npy", allow_pickle=True)
        stage = np.load(demo_dir / "stage_id.npy")

        frame_ids = sample_keyframes(meta, num_steps)
        print(f"[demo{d}] {len(frame_ids)} keyframes, decoding video...")
        t0 = time.time()
        raw = _load_top_frames(demo_dir, frame_ids)  # (K, H, W, 3) uint8
        print(f"  decoded in {time.time() - t0:.1f}s, encoding...")

        globals_chunks = []
        patches_chunks = []
        for i in range(0, len(raw), batch_size):
            batch_u8 = raw[i : i + batch_size]
            g, p = encode(batch_u8)
            globals_chunks.append(g.astype(np.float16))
            patches_chunks.append(p.astype(np.float16))

        global_features = np.concatenate(globals_chunks, axis=0)
        patch_tokens = np.concatenate(patches_chunks, axis=0)
        prompts = [str(instr[fid]) for fid in frame_ids]
        stage_ids = np.array([int(stage[fid]) for fid in frame_ids], dtype=np.int8)
        t_norm = np.array(frame_ids, dtype=np.float32) / float(num_steps)

        bank = MemoryBank(
            frame_ids=np.array(frame_ids, dtype=np.int32),
            stage_ids=stage_ids,
            t_norm=t_norm,
            global_features=global_features,
            patch_tokens=patch_tokens,
            prompts=prompts,
            num_steps=num_steps,
        )
        bank.save(out_path)
        print(f"  saved {out_path}  ({len(frame_ids)} keyframes, total {time.time() - t0:.1f}s)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demos", default="1-37", help="e.g. 1-37 or 1,3,5")
    p.add_argument("--out", default="retriever/cache/features", type=pathlib.Path)
    p.add_argument("--config", default=DEFAULT_CONFIG_NAME)
    p.add_argument("--ckpt", default=DEFAULT_CKPT_DIR, type=pathlib.Path)
    p.add_argument("--batch", type=int, default=32)
    args = p.parse_args()
    build(_parse_demos_arg(args.demos), args.out, args.config, args.ckpt, args.batch)


if __name__ == "__main__":
    main()
