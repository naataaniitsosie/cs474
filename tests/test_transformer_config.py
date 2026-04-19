"""Tests for scratch Transformer config contract."""

from __future__ import annotations

import pytest

from transformer import HF_T5_BASELINE_MODEL_ID, ScratchTransformerConfig


def test_hf_t5_default_id_is_small_variant() -> None:
    assert "t5" in HF_T5_BASELINE_MODEL_ID.lower()


def test_tiny_preset_valid() -> None:
    c = ScratchTransformerConfig.tiny()
    assert c.d_model % c.n_heads == 0
    assert c.head_dim == c.d_model // c.n_heads


def test_small_preset_valid() -> None:
    c = ScratchTransformerConfig.small()
    assert c.n_encoder_layers == 4


def test_medium_preset_valid() -> None:
    c = ScratchTransformerConfig.medium()
    assert c.d_model % c.n_heads == 0
    assert c.n_encoder_layers == 6


def test_invalid_d_model_heads_division() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ScratchTransformerConfig(
            d_model=100,
            n_heads=8,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=256,
            dropout=0.1,
            max_src_len=128,
            max_tgt_len=64,
        )


def test_dropout_oob() -> None:
    with pytest.raises(ValueError, match="dropout"):
        ScratchTransformerConfig(
            d_model=128,
            n_heads=4,
            n_encoder_layers=2,
            n_decoder_layers=2,
            d_ff=512,
            dropout=1.0,
            max_src_len=128,
            max_tgt_len=64,
        )

