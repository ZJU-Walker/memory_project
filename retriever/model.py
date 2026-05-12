"""PyTorch retriever: coarse dot-product scorer + attention reranker.

Inputs are frozen SigLIP features (cached on disk) and a frozen MiniLM text
embedding. The trainable parameters live entirely in this module.

Dimensions:
  D_IMG = 2048   (PaliGemma vision-tower output for pi05_plate_task)
  D_TXT = 384    (sentence-transformers/all-MiniLM-L6-v2)
  D     = 256    (shared projection dim)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_IMG = 2048
D_TXT = 384
D = 256
N_TIME_BANDS = 6


def time_sincos(dt: torch.Tensor, n_bands: int = N_TIME_BANDS) -> torch.Tensor:
    """Sinusoidal encoding of a scalar dt (signed, in roughly [-1, 1]).

    Returns (..., 2*n_bands) features.
    """
    freqs = 2.0 ** torch.arange(n_bands, device=dt.device, dtype=dt.dtype) * math.pi
    x = dt.unsqueeze(-1) * freqs  # (..., n_bands)
    return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class ImgProj(nn.Module):
    def __init__(self, d_in: int = D_IMG, d_out: int = D):
        super().__init__()
        self.proj = nn.Linear(d_in, d_out, bias=False)
        self.ln = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.proj(x))


class CoarseRetriever(nn.Module):
    """Dot-product scorer between (text + current image) query and (candidate + time) keys."""

    def __init__(self, d_img: int = D_IMG, d_txt: int = D_TXT, d: int = D):
        super().__init__()
        self.img_proj = ImgProj(d_img, d)
        self.cur_proj = ImgProj(d_img, d)
        self.txt_proj = nn.Sequential(nn.Linear(d_txt, d), nn.LayerNorm(d))
        self.time_proj = nn.Linear(2 * N_TIME_BANDS, 32)
        self.fuse_q = nn.Linear(2 * d, d)
        self.fuse_k = nn.Linear(d + 32, d)
        self.log_tau = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def encode_query(self, text_emb: torch.Tensor, cur_global: torch.Tensor) -> torch.Tensor:
        # text_emb (B, d_txt)   cur_global (B, d_img)  -> (B, d)
        t = self.txt_proj(text_emb)
        c = self.cur_proj(cur_global)
        return self.fuse_q(torch.cat([t, c], dim=-1))

    def encode_keys(self, cand_global: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        # cand_global (B, N, d_img)   dt (B, N)  -> (B, N, d)
        k_img = self.img_proj(cand_global)
        k_time = self.time_proj(time_sincos(dt))
        return self.fuse_k(torch.cat([k_img, k_time], dim=-1))

    def forward(
        self,
        text_emb: torch.Tensor,
        cur_global: torch.Tensor,
        cand_global: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        q = self.encode_query(text_emb, cur_global)            # (B, d)
        k = self.encode_keys(cand_global, dt)                  # (B, N, d)
        # cosine-style normalized dot product, temperature-scaled
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        scores = torch.einsum("bd,bnd->bn", q, k) * self.log_tau.exp()
        return scores


class AttentionReranker(nn.Module):
    """Per-candidate cross-attention scorer over patch tokens."""

    def __init__(
        self,
        d_img: int = D_IMG,
        d_txt: int = D_TXT,
        d: int = D,
        n_heads: int = 4,
        n_layers: int = 2,
        ffn_mult: int = 4,
    ):
        super().__init__()
        self.q_text = nn.Linear(d_txt, d)
        self.q_cur = nn.Linear(d_img, d)
        self.k_mem = nn.Linear(d_img, d)
        self.time_proj = nn.Linear(2 * N_TIME_BANDS, d)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=n_heads,
                dim_feedforward=ffn_mult * d,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            for _ in range(n_layers)
        ])
        self.score_head = nn.Linear(d, 1)
        self.ln = nn.LayerNorm(d)

    def forward(
        self,
        text_emb: torch.Tensor,
        cur_patches: torch.Tensor,
        cand_patches: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """
        text_emb     (B, d_txt)
        cur_patches  (B, P, d_img)            P = 256
        cand_patches (B, M, P, d_img)
        dt           (B, M)
        returns      (B, M) rerank logits
        """
        B, M, P, _ = cand_patches.shape
        # Project per-token
        t = self.q_text(text_emb).unsqueeze(1)                     # (B, 1, d)
        cur = self.q_cur(cur_patches)                              # (B, P, d)
        mem = self.k_mem(cand_patches)                             # (B, M, P, d)
        time_bias = self.time_proj(time_sincos(dt))                # (B, M, d)
        mem = mem + time_bias.unsqueeze(2)                         # broadcast over P

        # Expand text + cur to (B, M, ..., d) so each candidate is scored independently.
        t_rep = t.unsqueeze(1).expand(B, M, 1, t.size(-1))         # (B, M, 1, d)
        cur_rep = cur.unsqueeze(1).expand(B, M, P, cur.size(-1))   # (B, M, P, d)

        # Concat per-candidate token sequence: [text(1) || cur(P) || mem(P)]
        seq = torch.cat([t_rep, cur_rep, mem], dim=2)              # (B, M, 1+2P, d)
        seq = seq.reshape(B * M, 1 + 2 * P, -1)
        for blk in self.blocks:
            seq = blk(seq)
        pooled = self.ln(seq[:, 0])                                # (B*M, d)
        scores = self.score_head(pooled).reshape(B, M)             # (B, M)
        return scores


class Retriever(nn.Module):
    """Bundle the coarse + reranker so training/eval scripts have one entry point."""

    def __init__(self):
        super().__init__()
        self.coarse = CoarseRetriever()
        self.reranker = AttentionReranker()
