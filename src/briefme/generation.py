"""Phase 5: greedy and beam decoding for the scratch seq2seq model (autoregressive)."""

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


@torch.no_grad()
def beam_generate(
    model: ScratchSeq2SeqTransformer,
    src: Tensor,
    *,
    eos_token_id: int,
    decoder_start_token_id: int,
    max_new_tokens: int,
    num_beams: int,
    src_key_padding_mask: Tensor | None = None,
) -> Tensor:
    """Beam search left-to-right generation.

    ``src``: ``(batch, src_len)``. Returns token ids ``(batch, 1 + n_steps)`` for the
    single best beam per row (including decoder start as column 0).

    ``num_beams`` must be >= 1; ``1`` delegates to :func:`greedy_generate`.
    """
    if num_beams < 1:
        raise ValueError("num_beams must be >= 1")
    if num_beams == 1:
        return greedy_generate(
            model,
            src,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            max_new_tokens=max_new_tokens,
            src_key_padding_mask=src_key_padding_mask,
        )

    if src_key_padding_mask is None:
        src_key_padding_mask = src.eq(model.pad_token_id)

    device = src.device
    dtype = next(model.parameters()).dtype
    bsz, _ = src.shape
    vocab_size = model.vocab_size

    src_exp = src.repeat_interleave(num_beams, dim=0)
    mem_mask = src_key_padding_mask.repeat_interleave(num_beams, dim=0)
    memory = model.encode(src_exp, src_key_padding_mask=mem_mask)

    input_ids = torch.full(
        (bsz * num_beams, 1),
        decoder_start_token_id,
        dtype=torch.long,
        device=device,
    )

    beam_scores = torch.zeros(bsz, num_beams, device=device, dtype=torch.float32)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)

    finished = torch.zeros(bsz * num_beams, dtype=torch.bool, device=device)

    for _ in range(max_new_tokens):
        tgt_len = input_ids.size(1)
        causal = additive_causal_mask(tgt_len, device=device, dtype=dtype)
        pad_tgt = input_ids.eq(model.pad_token_id)
        if pad_tgt.size(1) > 0:
            pad_tgt[:, 0] = False
        logits = model.decode(
            input_ids,
            memory,
            tgt_attn_mask=causal,
            tgt_key_padding_mask=pad_tgt,
            memory_key_padding_mask=mem_mask,
        )
        log_probs = torch.log_softmax(logits[:, -1, :].float(), dim=-1)

        if finished.any():
            log_probs = log_probs.masked_fill(finished.unsqueeze(1), -1e9)
            eos_col = log_probs[:, eos_token_id]
            log_probs[:, eos_token_id] = torch.where(
                finished,
                torch.zeros_like(eos_col),
                eos_col,
            )

        next_scores = log_probs + beam_scores.unsqueeze(1)
        next_scores = next_scores.view(bsz, num_beams * vocab_size)

        next_scores, next_indices = next_scores.topk(num_beams, dim=1, largest=True, sorted=True)

        beam_idx = torch.div(next_indices, vocab_size, rounding_mode="floor")
        token_idx = next_indices.remainder(vocab_size)

        batch_offsets = torch.arange(bsz, device=device, dtype=torch.long).unsqueeze(1) * num_beams
        parent_flat = (batch_offsets + beam_idx).reshape(-1)

        input_ids = input_ids[parent_flat]
        input_ids = torch.cat([input_ids, token_idx.reshape(-1, 1)], dim=-1)

        beam_scores = next_scores.reshape(-1)
        finished = finished[parent_flat]
        finished |= token_idx.reshape(-1).eq(eos_token_id)

        if bool(finished.all().item()):
            break

    final_scores = beam_scores.view(bsz, num_beams)
    best_beam = final_scores.argmax(dim=1)
    rows = torch.arange(bsz, device=device) * num_beams + best_beam
    return input_ids[rows]


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
