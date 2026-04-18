"""Multi-head scaled dot-product attention (implements scratch Transformer Phase 2)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = True,
) -> Tensor:
    """Scaled dot-product attention.

    Shapes: ``q`` ``(B, H, Lq, D)``, ``k``/``v`` ``(B, H, Lk, D)``.
    Optional ``attn_mask`` broadcasts to ``(B, H, Lq, Lk)`` (additive).
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if attn_mask is not None:
        scores = scores + attn_mask
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p, training=training)
    return torch.matmul(attn, v)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with separate Q/K/V projections and output projection."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout_p = dropout

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Apply attention.

        ``q``, ``k``, ``v``: ``(batch, seq, d_model)``.

        ``key_padding_mask``: ``(batch, seq_k)`` with ``True`` at **pad** keys to ignore.
        ``attn_mask``: additive mask broadcasting to ``(B, H, Lq, Lk)``, e.g. causal ``(Lq, Lk)``.
        """
        batch, seq_q, _ = q.shape

        q_h = self._reshape_heads(self.q_proj(q))
        k_h = self._reshape_heads(self.k_proj(k))
        v_h = self._reshape_heads(self.v_proj(v))

        scores = torch.matmul(q_h, k_h.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores + attn_mask
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :],
                float("-inf"),
            )
        attn = F.softmax(scores, dim=-1)
        if self.dropout_p > 0.0:
            attn = F.dropout(attn, p=self.dropout_p, training=self.training)
        out = torch.matmul(attn, v_h)

        out = out.transpose(1, 2).contiguous().view(batch, seq_q, self.d_model)
        return self.out_proj(out)

    def _reshape_heads(self, x: Tensor) -> Tensor:
        batch, seq_len, _ = x.shape
        x = x.view(batch, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)
