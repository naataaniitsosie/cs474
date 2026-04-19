"""Tests for scratch checkpoint load + greedy inference (no Hub / no Ollama)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import torch

from briefme.scratch_inference import greedy_predict_headings, load_scratch_seq2seq_checkpoint
from transformer import ScratchSeq2SeqTransformer
from transformer.config import ScratchTransformerConfig


class _DummyTokenizerForInferenceTest:
    """Minimal tokenizer API (matches scratch training tests)."""

    pad_token_id = 0
    eos_token_id = 1
    decoder_start_token_id = 2
    vocab_size = 32000

    def encode(self, text: str, **_kwargs: object) -> list[int]:
        n = max(3, min(24, len(text) // 2 + 3))
        return [10 + (i % 100) for i in range(n)]

    def batch_decode(self, ids_list: list, *, skip_special_tokens: bool = True) -> list[str]:
        del skip_special_tokens
        return [" ".join(str(x) for x in row[:8]) for row in ids_list]


def test_load_scratch_checkpoint_and_greedy_predict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "briefme.scratch_inference.get_t5_tokenizer",
        lambda _id: _DummyTokenizerForInferenceTest(),
    )

    tcfg = ScratchTransformerConfig.tiny()
    model = ScratchSeq2SeqTransformer(tcfg, vocab_size=32000, pad_token_id=0)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": dataclasses.asdict(tcfg),
        "tokenizer_id": "dummy-tokenizer",
        "pad_token_id": 0,
        "source_prefix": "summarize: ",
    }
    path = tmp_path / "best.pt"
    torch.save(ckpt, path)

    loaded, tokenizer, cfg, prefix = load_scratch_seq2seq_checkpoint(path, device="cpu")
    assert prefix == "summarize: "
    assert cfg.max_src_len == tcfg.max_src_len

    examples = [
        {"source": "Argument text one about discovery.", "target": "Gold heading one", "metadata": {}},
        {"source": "Argument text two about contract.", "target": "Gold heading two", "metadata": {}},
    ]
    preds = greedy_predict_headings(
        loaded,
        tokenizer,
        examples,
        tcfg=cfg,
        source_prefix=prefix,
        batch_size=2,
    )
    assert len(preds) == 2
    assert all(isinstance(p, str) for p in preds)
