"""Retriever + rewriter glue for the realtime YAM eval.

Three pieces:
  - SiglipEncoder       : PaliGemma SigLIP encoder for live RGB frames
  - MiniLMTextEncoder   : sentence-transformers/all-MiniLM-L6-v2 for the prompt
  - LiveMemoryBank      : in-memory analogue of retriever.memory_bank.MemoryBank
                          that also retains the raw uint8 keyframe for the rewriter
  - retrieve_and_rewrite: top-1 keyframe via retriever + Gemini rewriter

Designed to be imported from eval/real/run_eval_retriever.py.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from retriever.dataset import D_IMG, N_MAX, P, MINILM_NAME
from retriever.eval import score_batch
from retriever.features import (
    DEFAULT_CKPT_DIR as _SIGLIP_DEFAULT_CKPT,
    DEFAULT_CONFIG_NAME as _SIGLIP_DEFAULT_CONFIG,
    make_encoder,
)
from rewriter.rewriter import rewrite as _rewrite


class SiglipEncoder:
    """JAX PaliGemma SigLIP encoder, loaded once on the local box."""

    def __init__(
        self,
        config_name: str = _SIGLIP_DEFAULT_CONFIG,
        ckpt_dir: pathlib.Path = _SIGLIP_DEFAULT_CKPT,
    ) -> None:
        self._encode = make_encoder(config_name, pathlib.Path(ckpt_dir))

    def encode_one(self, image_u8: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """RGB uint8 (H, W, 3) -> (global (D_IMG,), patches (P, D_IMG)) as float16."""
        if image_u8.ndim != 3 or image_u8.shape[-1] != 3 or image_u8.dtype != np.uint8:
            raise ValueError(f"expected uint8 (H,W,3), got {image_u8.dtype} {image_u8.shape}")
        batch = image_u8[None, ...]
        g, p = self._encode(batch)
        return g[0].astype(np.float16), p[0].astype(np.float16)


class MiniLMTextEncoder:
    """MiniLM-L6-v2 sentence encoder. Mirrors retriever.dataset.precompute_text_embeddings."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu") -> None:
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(MINILM_NAME)
        self._model = AutoModel.from_pretrained(MINILM_NAME).eval().to(device)
        self._device = device

    @torch.no_grad()
    def encode(self, prompt: str) -> np.ndarray:
        enc = self._tok(prompt, padding=True, truncation=True, return_tensors="pt").to(self._device)
        h = self._model(**enc).last_hidden_state
        mask = enc.attention_mask.unsqueeze(-1).float()
        emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        emb = F.normalize(emb, dim=-1)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)


@dataclass
class LiveMemoryBank:
    """In-memory keyframe bank built live during the observe stage.

    Parallel lists, all appended together; one entry per accepted keyframe.
    """

    raw_images: list[np.ndarray] = field(default_factory=list)        # uint8 (H,W,3)
    global_features: list[np.ndarray] = field(default_factory=list)   # f16 (D_IMG,)
    patch_tokens: list[np.ndarray] = field(default_factory=list)      # f16 (P, D_IMG)
    t_norm: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.raw_images)

    def append(
        self,
        image_u8: np.ndarray,
        global_: np.ndarray,
        patches: np.ndarray,
        t_norm: float,
    ) -> None:
        if global_.shape != (D_IMG,):
            raise ValueError(f"bad global shape {global_.shape}, expected ({D_IMG},)")
        if patches.shape != (P, D_IMG):
            raise ValueError(f"bad patches shape {patches.shape}, expected ({P}, {D_IMG})")
        if len(self) >= N_MAX:
            return  # silently drop — N_MAX is the retriever's hard cap
        self.raw_images.append(image_u8)
        self.global_features.append(global_)
        self.patch_tokens.append(patches)
        self.t_norm.append(float(t_norm))

    def as_batch(
        self,
        cur_global: np.ndarray,
        cur_patches: np.ndarray,
        text_emb: np.ndarray,
        cur_t_norm: float,
    ) -> dict:
        """Build a single-sample batch dict matching RetrieverDataset.__getitem__."""
        n_cand = len(self)
        cand_global = np.zeros((N_MAX, D_IMG), dtype=np.float32)
        cand_patches = np.zeros((N_MAX, P, D_IMG), dtype=np.float32)
        dt = np.zeros((N_MAX,), dtype=np.float32)
        valid_mask = np.zeros((N_MAX,), dtype=bool)
        forced_mask = np.zeros((N_MAX,), dtype=bool)

        if n_cand > 0:
            cand_global[:n_cand] = np.stack(self.global_features, axis=0).astype(np.float32)
            cand_patches[:n_cand] = np.stack(self.patch_tokens, axis=0).astype(np.float32)
            t_arr = np.asarray(self.t_norm, dtype=np.float32)
            dt[:n_cand] = cur_t_norm - t_arr  # >=0 because keyframes are in the past
            valid_mask[:n_cand] = True

        return {
            "text_emb": torch.from_numpy(text_emb.astype(np.float32)).unsqueeze(0),
            "cur_global": torch.from_numpy(cur_global.astype(np.float32)).unsqueeze(0),
            "cur_patches": torch.from_numpy(cur_patches.astype(np.float32)).unsqueeze(0),
            "cand_global": torch.from_numpy(cand_global).unsqueeze(0),
            "cand_patches": torch.from_numpy(cand_patches).unsqueeze(0),
            "dt": torch.from_numpy(dt).unsqueeze(0),
            "valid_mask": torch.from_numpy(valid_mask).unsqueeze(0),
            "forced_mask": torch.from_numpy(forced_mask).unsqueeze(0),
        }


def retrieve_and_rewrite(
    retriever_model: torch.nn.Module,
    bank: LiveMemoryBank,
    text_emb: np.ndarray,
    cur_global: np.ndarray,
    cur_patches: np.ndarray,
    cur_t_norm: float,
    original_prompt: str,
    rewriter_model: str,
    device: str,
) -> tuple[str, int]:
    """Top-1 keyframe via retriever, then Gemini rewriter. Returns (new_prompt, top1_idx)."""
    if len(bank) == 0:
        raise RuntimeError("empty memory bank")
    batch = bank.as_batch(cur_global, cur_patches, text_emb, cur_t_norm)
    with torch.no_grad():
        scores = score_batch(retriever_model, batch, device).cpu().numpy()[0]
    valid = batch["valid_mask"].cpu().numpy()[0]
    scores = np.where(valid, scores, -np.inf)
    top1_idx = int(np.argmax(scores))
    image = bank.raw_images[top1_idx]
    new_prompt = _rewrite(image, original_prompt, model=rewriter_model)
    return new_prompt, top1_idx
