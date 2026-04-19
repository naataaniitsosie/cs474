"""Phase 3: full scratch seq2seq forward and loss."""

from __future__ import annotations

import torch

from transformer import (
    ScratchSeq2SeqTransformer,
    ScratchTransformerConfig,
    seq2seq_cross_entropy_loss,
)


def test_seq2seq_forward_logits_shape() -> None:
    cfg = ScratchTransformerConfig.tiny()
    vocab = 67
    pad_id = 0
    model = ScratchSeq2SeqTransformer(cfg, vocab_size=vocab, pad_token_id=pad_id)
    model.eval()

    b, src_len, tgt_len = 2, 11, 9
    src = torch.randint(1, vocab, (b, src_len))
    tgt_in = torch.randint(1, vocab, (b, tgt_len))
    src[:, -2:] = pad_id
    tgt_in[:, -1:] = pad_id

    with torch.no_grad():
        logits = model(src, tgt_in)

    assert logits.shape == (b, tgt_len, vocab)


def test_seq2seq_loss_finite_and_ignores_pad() -> None:
    cfg = ScratchTransformerConfig.tiny()
    vocab = 31
    pad_id = 0
    model = ScratchSeq2SeqTransformer(cfg, vocab_size=vocab, pad_token_id=pad_id)

    b, src_len, tgt_len = 3, 8, 7
    src = torch.randint(1, vocab, (b, src_len))
    tgt_in = torch.randint(1, vocab, (b, tgt_len))
    labels = torch.randint(1, vocab, (b, tgt_len))
    src[:, -1:] = pad_id
    labels[:, -2:] = pad_id

    logits = model(src, tgt_in)
    loss = seq2seq_cross_entropy_loss(logits, labels, ignore_index=pad_id)

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_forward_finite_when_decoder_start_equals_pad_id() -> None:
    """T5-style: timestep 0 uses pad id as decoder start — must not yield NaN attention."""
    cfg = ScratchTransformerConfig.tiny()
    vocab = 64
    pad_id = 0
    model = ScratchSeq2SeqTransformer(cfg, vocab_size=vocab, pad_token_id=pad_id)
    model.eval()

    b, src_len, tgt_len = 2, 10, 8
    src = torch.randint(1, vocab, (b, src_len))
    tgt_in = torch.randint(1, vocab, (b, tgt_len))
    labels = torch.randint(1, vocab, (b, tgt_len))
    src[:, -2:] = pad_id
    tgt_in[:, 0] = pad_id
    labels[:, -1:] = pad_id

    with torch.no_grad():
        logits = model(src, tgt_in)
    loss = seq2seq_cross_entropy_loss(logits, labels, ignore_index=pad_id)

    assert torch.isfinite(logits).all()
    assert torch.isfinite(loss)


def test_encode_memory_shape() -> None:
    cfg = ScratchTransformerConfig.tiny()
    model = ScratchSeq2SeqTransformer(cfg, vocab_size=40, pad_token_id=0)
    src = torch.randint(1, 40, (2, 12))
    src[:, 10:] = 0

    with torch.no_grad():
        mem = model.encode(src)

    assert mem.shape == (2, 12, cfg.d_model)
