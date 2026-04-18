"""Transformer encoder/decoder layers (Pre-LN) — Phase 2."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformer.attention import MultiHeadAttention
from transformer.config import ScratchTransformerConfig


class FeedForward(nn.Module):
    """Two linear layers with GELU (Transformer default)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class EncoderLayer(nn.Module):
    """One encoder block: self-attention + FFN (Pre-LN)."""

    def __init__(self, config: ScratchTransformerConfig) -> None:
        super().__init__()
        d = config.d_model
        self.self_attn = MultiHeadAttention(d, config.n_heads, config.dropout)
        self.ff = FeedForward(d, config.d_ff, config.dropout)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, src_key_padding_mask: Tensor | None = None) -> Tensor:
        y = self.norm1(x)
        y = self.self_attn(y, y, y, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(y)
        y = self.norm2(x)
        y = self.ff(y)
        x = x + self.dropout(y)
        return x


class DecoderLayer(nn.Module):
    """One decoder block: causal self-attn, cross-attn, FFN (Pre-LN)."""

    def __init__(self, config: ScratchTransformerConfig) -> None:
        super().__init__()
        d = config.d_model
        self.self_attn = MultiHeadAttention(d, config.n_heads, config.dropout)
        self.cross_attn = MultiHeadAttention(d, config.n_heads, config.dropout)
        self.ff = FeedForward(d, config.d_ff, config.dropout)
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)
        self.norm3 = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        tgt_attn_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        y = self.norm1(x)
        y = self.self_attn(
            y,
            y,
            y,
            attn_mask=tgt_attn_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = x + self.dropout(y)

        y = self.norm2(x)
        y = self.cross_attn(
            y,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
        )
        x = x + self.dropout(y)

        y = self.norm3(x)
        y = self.ff(y)
        x = x + self.dropout(y)
        return x
