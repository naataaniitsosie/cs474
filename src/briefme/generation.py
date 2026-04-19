"""Phase 5: greedy decoding for the scratch seq2seq model (autoregressive)."""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from transformer.masking import additive_causal_mask
from transformer.model import ScratchSeq2SeqTransformer


@torch.no_grad()
def greedy_generate(
    model: ScratchSeq2SeqTransformer,
    src: Tensor,
    *,
    eos_token_id: int,
    decoder_start_token_id: int,
    max_new_tokens: int,
    src_key_padding_mask: Tensor | None = None,
) -> Tensor:
    """Greedy left-to-right generation.

    ``src``: ``(batch, src_len)`` token ids. Returns token ids ``(batch, 1 + n_steps)``
    including the decoder start id as column 0.
    """
    if src_key_padding_mask is None:
        src_key_padding_mask = src.eq(model.pad_token_id)

    device = src.device
    dtype = next(model.parameters()).dtype
    memory = model.encode(src, src_key_padding_mask=src_key_padding_mask)

    b = src.size(0)
    ids = torch.full((b, 1), decoder_start_token_id, dtype=torch.long, device=device)
    finished = torch.zeros(b, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        tgt_len = ids.size(1)
        causal = additive_causal_mask(tgt_len, device=device, dtype=dtype)
        pad_tgt = ids.eq(model.pad_token_id)
        if pad_tgt.size(1) > 0:
            pad_tgt[:, 0] = False
        logits = model.decode(
            ids,
            memory,
            tgt_attn_mask=causal,
            tgt_key_padding_mask=pad_tgt,
            memory_key_padding_mask=src_key_padding_mask,
        )
        next_id = logits[:, -1, :].argmax(dim=-1)
        ids = torch.cat([ids, next_id.unsqueeze(1)], dim=1)
        finished |= next_id.eq(eos_token_id)
        if bool(finished.all().item()):
            break

    return ids


def batch_decode_skip_special(
    tokenizer: PreTrainedTokenizerBase,
    token_ids: Tensor,
    *,
    skip_special_tokens: bool = True,
) -> list[str]:
    """Decode each row; strip trailing specials when requested."""
    ids_list = token_ids.detach().cpu().tolist()
    return tokenizer.batch_decode(ids_list, skip_special_tokens=skip_special_tokens)


def strip_decoder_start(
    token_ids: Tensor,
    decoder_start_token_id: int,
) -> Tensor:
    """Remove column 0 if it equals decoder start (common greedy output layout)."""
    if token_ids.size(1) < 2:
        return token_ids
    if bool((token_ids[:, 0] == decoder_start_token_id).all().item()):
        return token_ids[:, 1:]
    return token_ids
