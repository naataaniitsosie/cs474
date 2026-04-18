"""Scratch encoder–decoder Transformer (implemented incrementally per north star plan).

Lives beside ``briefme`` under ``src/``: dataset/eval helpers vs model code stay separate.
"""

from transformer.config import HF_T5_BASELINE_MODEL_ID, ScratchTransformerConfig

__all__ = [
    "HF_T5_BASELINE_MODEL_ID",
    "ScratchTransformerConfig",
]
