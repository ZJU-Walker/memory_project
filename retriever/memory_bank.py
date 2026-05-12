"""Per-demo memory bank of pre-computed SigLIP features for retriever training."""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np


@dataclasses.dataclass
class MemoryBank:
    frame_ids: np.ndarray        # (K,) int32  — index into the 30 Hz video
    stage_ids: np.ndarray        # (K,) int8
    t_norm: np.ndarray           # (K,) float32  — frame_id / num_steps
    global_features: np.ndarray  # (K, 1152) float16  — GAP over patch tokens
    patch_tokens: np.ndarray     # (K, 256, 1152) float16
    prompts: list[str]           # len K  — per-frame resolved prompt
    num_steps: int

    def __len__(self) -> int:
        return int(self.frame_ids.shape[0])

    def save(self, path: pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            frame_ids=self.frame_ids.astype(np.int32),
            stage_ids=self.stage_ids.astype(np.int8),
            t_norm=self.t_norm.astype(np.float32),
            global_features=self.global_features.astype(np.float16),
            patch_tokens=self.patch_tokens.astype(np.float16),
            prompts=np.asarray(self.prompts, dtype=object),
            num_steps=np.int32(self.num_steps),
        )

    @classmethod
    def load(cls, path: pathlib.Path) -> "MemoryBank":
        z = np.load(pathlib.Path(path), allow_pickle=True)
        return cls(
            frame_ids=z["frame_ids"],
            stage_ids=z["stage_ids"],
            t_norm=z["t_norm"],
            global_features=z["global_features"],
            patch_tokens=z["patch_tokens"],
            prompts=list(z["prompts"]),
            num_steps=int(z["num_steps"]),
        )
