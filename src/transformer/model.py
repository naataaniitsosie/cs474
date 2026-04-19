"""Full encoder–decoder stack: embeddings, positional encoding, logits (Phase 3)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformer.config import ScratchTransformerConfig
from transformer.layers import DecoderLayer, EncoderLayer
from transformer.masking import additive_causal_mask
from transformer.positional import SinusoidalPositionalEncoding


def default_tgt_key_padding_mask(tgt_in: Tensor, pad_token_id: int) -> Tensor:
    """Mask decoder **padding keys** for self-attention (True = ignore key).

    T5 uses ``pad_token_id`` as **decoder_start_token_id**. Timestep 0 is therefore that
    id but must **not** be masked: causal attention at position 0 only attends to key 0;
    masking it yields all-``-inf`` logits and **NaN** loss.
    """
    m = tgt_in.eq(pad_token_id)
    if tgt_in.size(1) > 0:
        m[:, 0] = False
    return m


def seq2seq_cross_entropy_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int,
) -> Tensor:
    """Teacher-forcing loss: predict ``labels[:, t]`` from logits at position ``t``.

    ``logits``: ``(batch, tgt_len, vocab)``, ``labels``: ``(batch, tgt_len)`` with
    padding marked by ``ignore_index``.
    """
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
    )


class ScratchSeq2SeqTransformer(nn.Module):
    """Encoder–decoder Transformer for token IDs (scratch implementation)."""

    def __init__(
        self,
        config: ScratchTransformerConfig,
        vocab_size: int,
        *,
        pad_token_id: int = 0,
        max_pe_len: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")
        self.config = config
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

        d = config.d_model
        max_pe = max_pe_len if max_pe_len is not None else max(config.max_src_len, config.max_tgt_len)

        self.src_embed = nn.Embedding(vocab_size, d)
        self.tgt_embed = nn.Embedding(vocab_size, d)
        self.src_pos = SinusoidalPositionalEncoding(d, config.dropout, max_len=max_pe)
        self.tgt_pos = SinusoidalPositionalEncoding(d, config.dropout, max_len=max_pe)

        self.encoder_layers = nn.ModuleList(
            EncoderLayer(config) for _ in range(config.n_encoder_layers)
        )
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config.n_decoder_layers)
        )

        self.decoder_norm = nn.LayerNorm(d)
        self.output_proj = nn.Linear(d, vocab_size)

        self._emb_scale = math.sqrt(d)

    def encode(
        self,
        src: Tensor,
        src_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Encode source token ids ``(batch, src_len)`` → ``(batch, src_len, d_model)``."""
        if src_key_padding_mask is None:
            src_key_padding_mask = src.eq(self.pad_token_id)

        x = self.src_embed(src) * self._emb_scale
        x = self.src_pos(x)
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        *,
        tgt_attn_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Decode teacher-forcing inputs ``tgt`` ``(batch, tgt_len)`` → logits ``(batch, tgt_len, vocab)``."""
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = default_tgt_key_padding_mask(tgt, self.pad_token_id)
        if memory_key_padding_mask is None:
            # memory comes from encoder; infer pad from batch if not passed — caller should pass src mask
            memory_key_padding_mask = None

        x = self.tgt_embed(tgt) * self._emb_scale
        x = self.tgt_pos(x)
        for layer in self.decoder_layers:
            x = layer(
                x,
                memory,
                tgt_attn_mask=tgt_attn_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        x = self.decoder_norm(x)
        return self.output_proj(x)

    def forward(
        self,
        src: Tensor,
        tgt_in: Tensor,
        *,
        src_key_padding_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Full forward with teacher forcing.

        ``tgt_in`` should be shifted right (e.g. ``[BOS, y0, …]``); causal mask applied inside.
        """
        if src_key_padding_mask is None:
            src_key_padding_mask = src.eq(self.pad_token_id)
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = default_tgt_key_padding_mask(tgt_in, self.pad_token_id)

        memory = self.encode(src, src_key_padding_mask=src_key_padding_mask)

        tgt_len = tgt_in.size(1)
        dt = next(self.parameters()).dtype
        causal = additive_causal_mask(tgt_len, device=tgt_in.device, dtype=dt)

        return self.decode(
            tgt_in,
            memory,
            tgt_attn_mask=causal,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
