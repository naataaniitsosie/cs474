"""Load trained scratch seq2seq checkpoints and greedy-decode headings for evaluation."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from briefme.generation import batch_decode_skip_special, greedy_generate, strip_decoder_start
from briefme.seq2seq_data import BriefMeSeq2SeqDataset, collate_seq2seq_batch, get_t5_tokenizer
from briefme.train_scratch_loop import pick_device
from transformer import ScratchSeq2SeqTransformer
from transformer.config import ScratchTransformerConfig


def load_scratch_seq2seq_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str | None = None,
) -> tuple[ScratchSeq2SeqTransformer, PreTrainedTokenizerBase, ScratchTransformerConfig, str]:
    """Load ``best.pt`` from :func:`briefme.train_scratch_loop.run_scratch_training`.

    Returns ``(model, tokenizer, scratch_cfg, source_prefix)``. Model is in ``eval()`` mode.
    """
    path = Path(path)
    if isinstance(device, torch.device):
        dev = device
    else:
        dev = pick_device(device if isinstance(device, str) else None)

    ckpt = torch.load(path, map_location=dev)

    raw_cfg = ckpt["config"]
    names = {f.name for f in dataclasses.fields(ScratchTransformerConfig)}
    cfg_kwargs = {k: v for k, v in raw_cfg.items() if k in names}
    tcfg = ScratchTransformerConfig(**cfg_kwargs)

    tokenizer = get_t5_tokenizer(str(ckpt["tokenizer_id"]))
    pad_id = int(ckpt["pad_token_id"])
    model = ScratchSeq2SeqTransformer(tcfg, vocab_size=tokenizer.vocab_size, pad_token_id=pad_id)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(dev)
    model.eval()

    source_prefix = str(ckpt.get("source_prefix", ""))
    return model, tokenizer, tcfg, source_prefix


@torch.no_grad()
def greedy_predict_headings(
    model: ScratchSeq2SeqTransformer,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[dict[str, Any]],
    *,
    tcfg: ScratchTransformerConfig,
    source_prefix: str,
    batch_size: int = 8,
    max_new_tokens: int | None = None,
) -> list[str]:
    """Greedy-decode one predicted heading string per example (same order as ``examples``)."""
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    dec_start = getattr(tokenizer, "decoder_start_token_id", None)
    if dec_start is None:
        dec_start = pad_id
    if pad_id is None or eos_id is None:
        raise ValueError("Tokenizer must define pad_token_id and eos_token_id")

    cap = tcfg.max_tgt_len if max_new_tokens is None else min(max_new_tokens, tcfg.max_tgt_len)

    ds = BriefMeSeq2SeqDataset(
        examples,
        tokenizer,
        max_src_len=tcfg.max_src_len,
        max_tgt_len=tcfg.max_tgt_len,
        source_prefix=source_prefix,
    )

    def collate(batch: list) -> dict[str, Any]:
        return collate_seq2seq_batch(batch, pad_token_id=pad_id)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    preds: list[str] = []
    model_dev = next(model.parameters()).device
    for batch in loader:
        src_b = batch["src"].to(model_dev)
        gen = greedy_generate(
            model,
            src_b,
            eos_token_id=eos_id,
            decoder_start_token_id=dec_start,
            max_new_tokens=cap,
        )
        gen = strip_decoder_start(gen, dec_start)
        preds.extend(batch_decode_skip_special(tokenizer, gen))
    return preds


__all__ = ["greedy_predict_headings", "load_scratch_seq2seq_checkpoint"]
