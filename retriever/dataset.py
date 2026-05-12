"""PyTorch dataset wrapping the per-demo MemoryBank cache + label JSONL.

Each item returns the tensors needed for one anchor forward pass:
  - text_emb       (D_TXT,)
  - cur_global     (D_IMG,)
  - cur_patches    (P, D_IMG)
  - cand_global    (N_MAX, D_IMG)   zero-padded
  - cand_patches   (N_MAX, P, D_IMG) zero-padded
  - cand_frame_ids (N_MAX,) int   for debug only
  - dt             (N_MAX,)
  - valid_mask     (N_MAX,) bool   True where the slot is a real candidate
  - positives_mask (N_MAX,) bool
  - forced_mask    (N_MAX,) bool   positives + labeled hard-negs (must enter the rerank set)
  - num_steps      int
  - stage_id       int

Strict causal masking: only candidates with frame_id <= anchor_frame_id occupy
valid slots. All other slots are zero-padded and masked out.
"""

from __future__ import annotations

import json
import pathlib
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset

from retriever.memory_bank import MemoryBank

N_MAX = 240            # max keyframes per anchor (>= sampler MAX_KEYFRAMES=128; extra room for live bank)
P = 256                # SigLIP patch token count
D_IMG = 2048
D_TXT = 384
MINILM_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def precompute_text_embeddings(prompts: list[str], out_path: pathlib.Path) -> dict[str, np.ndarray]:
    """Encode unique prompts with MiniLM, merge with any existing cache, and save."""
    from transformers import AutoModel, AutoTokenizer

    existing: dict[str, np.ndarray] = (
        load_text_embeddings(out_path) if out_path.exists() else {}
    )
    needed = sorted(set(prompts) - set(existing))
    if not needed:
        return existing

    tok = AutoTokenizer.from_pretrained(MINILM_NAME)
    model = AutoModel.from_pretrained(MINILM_NAME).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    with torch.no_grad():
        for prompt in needed:
            enc = tok(prompt, padding=True, truncation=True, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state              # (1, T, 384)
            mask = enc.attention_mask.unsqueeze(-1).float() # (1, T, 1)
            emb = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            emb = torch.nn.functional.normalize(emb, dim=-1)
            existing[prompt] = emb.squeeze(0).cpu().numpy().astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(existing.keys())
    np.savez(out_path, prompts=np.asarray(keys, dtype=object),
             embeddings=np.stack([existing[k] for k in keys], axis=0).astype(np.float32))
    return existing


def load_text_embeddings(path: pathlib.Path) -> dict[str, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return {str(p): z["embeddings"][i] for i, p in enumerate(z["prompts"])}


@lru_cache(maxsize=64)
def _load_bank(features_dir: str, demo: str) -> MemoryBank:
    return MemoryBank.load(pathlib.Path(features_dir) / f"{demo}.npz")


class RetrieverDataset(Dataset):
    def __init__(
        self,
        labels_path: pathlib.Path,
        features_dir: pathlib.Path,
        text_emb_path: pathlib.Path,
    ):
        self.features_dir = str(features_dir)
        self.samples: list[dict] = [json.loads(l) for l in open(labels_path) if l.strip()]
        prompts = sorted({s["anchor_prompt"] for s in self.samples})
        existing = load_text_embeddings(text_emb_path) if text_emb_path.exists() else {}
        missing = set(prompts) - set(existing)
        if missing:
            print(f"[dataset] encoding {len(missing)} new prompts (cache has {len(existing)}) -> {text_emb_path}")
            precompute_text_embeddings(prompts, text_emb_path)
        self.text_emb = load_text_embeddings(text_emb_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        bank = _load_bank(self.features_dir, s["demo"])
        anchor = int(s["anchor_frame_id"])
        kf_ids = bank.frame_ids
        anchor_idx_arr = np.where(kf_ids == anchor)[0]
        if anchor_idx_arr.size == 0:
            raise RuntimeError(f"anchor {anchor} not in bank for {s['demo']}")
        anchor_idx = int(anchor_idx_arr[0])

        # Causal candidate set: frame_id <= anchor_frame_id (current frame IS included).
        causal = kf_ids <= anchor
        cand_frame_ids = kf_ids[causal]
        n_cand = int(cand_frame_ids.shape[0])
        if n_cand > N_MAX:
            raise RuntimeError(f"n_cand={n_cand} exceeds N_MAX={N_MAX} for {s['demo']}@{anchor}")

        # Allocate padded tensors.
        cand_global = np.zeros((N_MAX, D_IMG), dtype=np.float32)
        cand_patches = np.zeros((N_MAX, P, D_IMG), dtype=np.float32)
        dt = np.zeros((N_MAX,), dtype=np.float32)
        valid_mask = np.zeros((N_MAX,), dtype=bool)
        positives_mask = np.zeros((N_MAX,), dtype=bool)
        forced_mask = np.zeros((N_MAX,), dtype=bool)
        padded_frame_ids = np.full((N_MAX,), -1, dtype=np.int64)

        cand_global[:n_cand] = bank.global_features[causal].astype(np.float32)
        cand_patches[:n_cand] = bank.patch_tokens[causal].astype(np.float32)
        dt[:n_cand] = (anchor - cand_frame_ids).astype(np.float32) / float(bank.num_steps)
        valid_mask[:n_cand] = True
        padded_frame_ids[:n_cand] = cand_frame_ids.astype(np.int64)

        pos_set = set(int(x) for x in s["positives"])
        neg_set = set(int(x) for x in s["hard_negatives"])
        for j in range(n_cand):
            fid = int(cand_frame_ids[j])
            if fid in pos_set:
                positives_mask[j] = True
                forced_mask[j] = True
            elif fid in neg_set:
                forced_mask[j] = True

        return {
            "text_emb": torch.from_numpy(self.text_emb[s["anchor_prompt"]]),
            "cur_global": torch.from_numpy(bank.global_features[anchor_idx].astype(np.float32)),
            "cur_patches": torch.from_numpy(bank.patch_tokens[anchor_idx].astype(np.float32)),
            "cand_global": torch.from_numpy(cand_global),
            "cand_patches": torch.from_numpy(cand_patches),
            "cand_frame_ids": torch.from_numpy(padded_frame_ids),
            "dt": torch.from_numpy(dt),
            "valid_mask": torch.from_numpy(valid_mask),
            "positives_mask": torch.from_numpy(positives_mask),
            "forced_mask": torch.from_numpy(forced_mask),
            "stage_id": int(s["stage_id"]),
            "anchor_frame_id": anchor,
            "demo": s["demo"],
        }
