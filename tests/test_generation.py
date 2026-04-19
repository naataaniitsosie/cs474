"""Phase 5: greedy decode for scratch seq2seq."""

from __future__ import annotations

import torch

from briefme.generation import greedy_generate, strip_decoder_start
from transformer import ScratchSeq2SeqTransformer, ScratchTransformerConfig


def test_greedy_generate_shapes_and_includes_start_token() -> None:
    cfg = ScratchTransformerConfig.tiny()
    pad_id = 0
    eos_id = 1
    vocab = max(32, eos_id + 2)
    model = ScratchSeq2SeqTransformer(cfg, vocab_size=vocab, pad_token_id=pad_id)
    model.eval()

    b, src_len = 2, 10
    src = torch.randint(2, vocab, (b, src_len))
    dec_start = pad_id

    out = greedy_generate(
        model,
        src,
        eos_token_id=eos_id,
        decoder_start_token_id=dec_start,
        max_new_tokens=15,
    )
    assert out.shape[0] == b
    assert out.shape[1] >= 2
    assert torch.all(out[:, 0] == dec_start)


def test_strip_decoder_start() -> None:
    x = torch.tensor([[0, 3, 4, 5], [0, 9, 1, 0]])
    y = strip_decoder_start(x, decoder_start_token_id=0)
    assert y.shape == (2, 3)
    assert torch.equal(y[:, 0], torch.tensor([3, 9]))
