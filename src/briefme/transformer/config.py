"""Hyperparameter contract for the scratch encoder–decoder (Phase 1).

``vocab_size`` is filled once a tokenizer is chosen. The T5 baseline default id is
for the separate Hugging Face finetuning script (reference model only).
"""

from __future__ import annotations

from dataclasses import dataclass

# Hugging Face hub id for the T5 reference baseline (finetuning); not used by scratch weights.
HF_T5_BASELINE_MODEL_ID = "google-t5/t5-small"


@dataclass(frozen=True)
class ScratchTransformerConfig:
    """Canonical sizes for attention blocks and sequence caps.

    Requires ``d_model % n_heads == 0`` for multi-head attention head_dim.
    """

    d_model: int
    n_heads: int
    n_encoder_layers: int
    n_decoder_layers: int
    d_ff: int
    dropout: float
    max_src_len: int
    max_tgt_len: int
    vocab_size: int | None = None

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if self.n_encoder_layers < 1 or self.n_decoder_layers < 1:
            raise ValueError("n_encoder_layers and n_decoder_layers must be >= 1")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.max_src_len < 1 or self.max_tgt_len < 1:
            raise ValueError("max_src_len and max_tgt_len must be >= 1")
        if self.vocab_size is not None and self.vocab_size < 2:
            raise ValueError("vocab_size must be None or >= 2")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def tiny(cls) -> ScratchTransformerConfig:
        """Smoke tests, CPU/MPS-friendly (small tensors)."""
        return cls(
            d_model=128,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=512,
            dropout=0.1,
            max_src_len=256,
            max_tgt_len=64,
            vocab_size=None,
        )

    @classmethod
    def small(cls) -> ScratchTransformerConfig:
        """Heavier config for real BriefMe training once the loop is stable."""
        return cls(
            d_model=256,
            n_heads=8,
            n_encoder_layers=4,
            n_decoder_layers=4,
            d_ff=1024,
            dropout=0.1,
            max_src_len=512,
            max_tgt_len=128,
            vocab_size=None,
        )
