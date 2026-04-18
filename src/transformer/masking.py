"""Attention masks for padding and decoder causality."""

from __future__ import annotations

import torch
from torch import Tensor


def additive_causal_mask(seq_len: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Upper-triangle **additive** mask (0 on/below diagonal, ``-inf`` strictly above).

    Add to attention logits before softmax so position *i* cannot attend to *j > i*.
    Shape ``(seq_len, seq_len)`` — broadcasts over batch and heads.
    """
    m = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    m = m + torch.triu(torch.full_like(m, float("-inf")), diagonal=1)
    return m
