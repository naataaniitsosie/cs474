"""Phase 2: attention, masks, encoder/decoder layers."""

from __future__ import annotations

import pytest
import torch

from transformer import (
    DecoderLayer,
    EncoderLayer,
    MultiHeadAttention,
    ScratchTransformerConfig,
    additive_causal_mask,
    scaled_dot_product_attention,
)


def test_additive_causal_upper_triangle() -> None:
    m = additive_causal_mask(4, device=torch.device("cpu"), dtype=torch.float32)
    assert m[0, 0] == 0.0
    assert m[0, 1] == float("-inf")
    assert m[1, 0] == 0.0
    assert m[3, 3] == 0.0


def test_scaled_dot_product_attention_shape() -> None:
    b, h, l, d = 2, 2, 5, 8
    q = torch.randn(b, h, l, d)
    k = torch.randn(b, h, l, d)
    v = torch.randn(b, h, l, d)
    out = scaled_dot_product_attention(q, k, v, dropout_p=0.0, training=False)
    assert out.shape == (b, h, l, d)


def test_multi_head_attention_output_shape() -> None:
    cfg = ScratchTransformerConfig.tiny()
    mha = MultiHeadAttention(cfg.d_model, cfg.n_heads, dropout=0.0)
    x = torch.randn(2, 10, cfg.d_model)
    y = mha(x, x, x)
    assert y.shape == x.shape


def test_encoder_layer_shape_and_padding() -> None:
    cfg = ScratchTransformerConfig.tiny()
    layer = EncoderLayer(cfg)
    x = torch.randn(2, 10, cfg.d_model)
    pad = torch.zeros(2, 10, dtype=torch.bool)
    pad[:, 8:] = True  # pad last two tokens
    y = layer(x, src_key_padding_mask=pad)
    assert y.shape == x.shape


def test_decoder_causal_position_zero_ignores_future_tokens() -> None:
    """With causal mask, timestep 0 output should not depend on inputs at t>=1."""
    torch.manual_seed(0)
    cfg = ScratchTransformerConfig.tiny()
    dec = DecoderLayer(cfg)
    dec.eval()

    batch, tgt_len = 2, 8
    src_len = 6
    x = torch.randn(batch, tgt_len, cfg.d_model)
    memory = torch.randn(batch, src_len, cfg.d_model)

    causal = additive_causal_mask(tgt_len, device=x.device, dtype=x.dtype)
    # (L,L) broadcasts to (B,H,L,L) when added to scores inside MHA

    with torch.no_grad():
        out1 = dec(x, memory, tgt_attn_mask=causal)

    x2 = x.clone()
    x2[:, 1:, :] = 123.0  # alter all future input positions

    with torch.no_grad():
        out2 = dec(x2, memory, tgt_attn_mask=causal)

    assert torch.allclose(out1[:, 0, :], out2[:, 0, :], atol=1e-5)


def test_decoder_cross_attention_changes_with_memory() -> None:
    """Cross-attn should depend on encoder memory (qualitative sanity)."""
    torch.manual_seed(1)
    cfg = ScratchTransformerConfig.tiny()
    dec = DecoderLayer(cfg)
    dec.eval()
    batch, tgt_len, src_len = 1, 4, 5
    x = torch.randn(batch, tgt_len, cfg.d_model)
    mem1 = torch.randn(batch, src_len, cfg.d_model)
    mem2 = torch.randn(batch, src_len, cfg.d_model)
    causal = additive_causal_mask(tgt_len, device=x.device, dtype=x.dtype)

    with torch.no_grad():
        o1 = dec(x, mem1, tgt_attn_mask=causal)
        o2 = dec(x, mem2, tgt_attn_mask=causal)

    assert not torch.allclose(o1, o2)


def test_encoder_batch_matrix_equiv_one_head() -> None:
    """Single-head path matches manual batched attention for one layer."""
    d_model = 32
    n_heads = 1
    seq = 4
    batch = 2
    mha = MultiHeadAttention(d_model, n_heads, dropout=0.0)
    mha.eval()
    x = torch.randn(batch, seq, d_model)

    with torch.no_grad():
        y = mha(x, x, x)

    # Manual one-head SDPA on projected vectors (extract first head).
    with torch.no_grad():
        q = mha.q_proj(x).view(batch, seq, n_heads, d_model // n_heads).transpose(1, 2)
        k = mha.k_proj(x).view(batch, seq, n_heads, d_model // n_heads).transpose(1, 2)
        v = mha.v_proj(x).view(batch, seq, n_heads, d_model // n_heads).transpose(1, 2)
        raw = scaled_dot_product_attention(q, k, v, dropout_p=0.0, training=False)
        raw = raw.transpose(1, 2).contiguous().view(batch, seq, d_model)
        manual = mha.out_proj(raw)

    assert torch.allclose(y, manual, atol=1e-5)
