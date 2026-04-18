"""Scratch encoder–decoder Transformer (implements incrementally per north star plan).

Heavy modules (``torch``) load lazily so ``from transformer import ScratchTransformerConfig``
works even when Attention layers are not needed.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from transformer.config import HF_T5_BASELINE_MODEL_ID, ScratchTransformerConfig

__all__ = [
    "HF_T5_BASELINE_MODEL_ID",
    "ScratchTransformerConfig",
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "FeedForward",
    "EncoderLayer",
    "DecoderLayer",
    "additive_causal_mask",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "scaled_dot_product_attention": ("transformer.attention", "scaled_dot_product_attention"),
    "MultiHeadAttention": ("transformer.attention", "MultiHeadAttention"),
    "FeedForward": ("transformer.layers", "FeedForward"),
    "EncoderLayer": ("transformer.layers", "EncoderLayer"),
    "DecoderLayer": ("transformer.layers", "DecoderLayer"),
    "additive_causal_mask": ("transformer.masking", "additive_causal_mask"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        mod_path, attr = _LAZY_IMPORTS[name]
        return getattr(import_module(mod_path), attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from transformer.attention import MultiHeadAttention, scaled_dot_product_attention
    from transformer.layers import DecoderLayer, EncoderLayer, FeedForward
    from transformer.masking import additive_causal_mask
