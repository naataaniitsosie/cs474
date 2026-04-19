"""Phase 4: tokenization, map-style Dataset, and batch collation for seq2seq training."""

from __future__ import annotations

import sys
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, T5TokenizerFast

from briefme.data import SPLIT_DEV, SPLIT_TRAIN, iter_rows, load_arg_summ_split_streaming
from briefme.schema import to_seq2seq_example
from transformer.config import HF_T5_BASELINE_MODEL_ID

# T5-style task prefix (same string used for scratch + T5 baseline when enabled).
DEFAULT_T5_TASK_PREFIX = "summarize: "

# Progress lines on stderr while streaming Hub rows (full train can take minutes).
_MATERIALIZE_PULSE_EVERY = 500


def parse_cli_train_limit(value: str) -> int | None:
    """Parse ``--train-limit`` for training CLIs.

    Positive integers cap how many train rows are materialized. The strings ``none``, ``full``,
    and ``all`` (case-insensitive) mean **no cap**—stream the entire Hub train split.
    """
    v = value.strip().lower()
    if v in ("none", "full", "all"):
        return None
    return int(v)


def get_t5_tokenizer(model_id: str = HF_T5_BASELINE_MODEL_ID, **kwargs: Any) -> T5TokenizerFast:
    """Load the fast T5 tokenizer (shared vocab for scratch embeddings and T5 baseline).

    Extra ``kwargs`` are forwarded to ``T5TokenizerFast.from_pretrained`` (e.g.
    ``local_files_only=True``).
    """
    return T5TokenizerFast.from_pretrained(model_id, **kwargs)


def materialize_examples(split: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Stream BriefMe ``arg_summ`` and map rows to ``source`` / ``target`` / ``metadata``."""
    lim_s = "full split" if limit is None else str(limit)
    print(f"[briefme] streaming {split!r} from Hub (limit={lim_s})...", file=sys.stderr, flush=True)
    stream = load_arg_summ_split_streaming(split)
    out: list[dict[str, Any]] = []
    for i, row in enumerate(iter_rows(stream, limit=limit), start=1):
        out.append(to_seq2seq_example(row))
        if i % _MATERIALIZE_PULSE_EVERY == 0:
            print(f"[briefme]   {split!r}: {i} rows...", file=sys.stderr, flush=True)
    print(f"[briefme]   {split!r}: done ({len(out)} rows)", file=sys.stderr, flush=True)
    return out


def encode_pair_lists(
    tokenizer: PreTrainedTokenizerBase,
    source: str,
    target: str,
    *,
    max_src_len: int,
    max_tgt_len: int,
    source_prefix: str = DEFAULT_T5_TASK_PREFIX,
) -> dict[str, Any]:
    """Return list[int] ``src``, ``tgt_in``, ``labels`` for :class:`ScratchSeq2SeqTransformer`.

    Teacher forcing: ``tgt_in[0]`` is the decoder start token; ``labels[t]`` is the token
    predicted at position ``t``. Padding id matches ``tokenizer.pad_token_id``.
    """
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None or eos_id is None:
        raise ValueError("Tokenizer must define pad_token_id and eos_token_id")

    dec_start = getattr(tokenizer, "decoder_start_token_id", None)
    if dec_start is None:
        dec_start = pad_id

    src_text = f"{source_prefix}{source}" if source_prefix else source
    src_ids = tokenizer.encode(src_text, add_special_tokens=True, truncation=True, max_length=max_src_len)

    tgt_piece = tokenizer.encode(target, add_special_tokens=False, truncation=True, max_length=max(1, max_tgt_len - 1))
    label_ids = tgt_piece + [eos_id]
    if len(label_ids) > max_tgt_len:
        label_ids = label_ids[:max_tgt_len]

    if len(label_ids) < 1:
        label_ids = [eos_id]

    labels = label_ids
    tgt_in = [dec_start] + labels[:-1]
    assert len(tgt_in) == len(labels)

    return {
        "src": src_ids,
        "tgt_in": tgt_in,
        "labels": labels,
        "source_text": source,
        "target_text": target,
        "pad_token_id": pad_id,
    }


class BriefMeSeq2SeqDataset(Dataset):
    """Map-style dataset over string pairs; tokenizes on ``__getitem__``."""

    def __init__(
        self,
        examples: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_src_len: int,
        max_tgt_len: int,
        source_prefix: str = DEFAULT_T5_TASK_PREFIX,
    ) -> None:
        self._rows = examples
        self._tokenizer = tokenizer
        self._max_src = max_src_len
        self._max_tgt = max_tgt_len
        self._prefix = source_prefix

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self._rows[idx]
        return encode_pair_lists(
            self._tokenizer,
            row["source"],
            row["target"],
            max_src_len=self._max_src,
            max_tgt_len=self._max_tgt,
            source_prefix=self._prefix,
        )


def collate_seq2seq_batch(
    batch: list[dict[str, Any]],
    *,
    pad_token_id: int,
) -> dict[str, torch.Tensor | list[str]]:
    """Pad ``src``, ``tgt_in``, ``labels`` to per-batch max length."""
    max_src = max(len(x["src"]) for x in batch)
    max_tgt = max(len(x["tgt_in"]) for x in batch)
    bsz = len(batch)

    src = torch.full((bsz, max_src), pad_token_id, dtype=torch.long)
    tgt_in = torch.full((bsz, max_tgt), pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, max_tgt), pad_token_id, dtype=torch.long)

    for i, x in enumerate(batch):
        s, t, y = x["src"], x["tgt_in"], x["labels"]
        src[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_in[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        labels[i, : len(y)] = torch.tensor(y, dtype=torch.long)

    return {
        "src": src,
        "tgt_in": tgt_in,
        "labels": labels,
        "reference_texts": [x["target_text"] for x in batch],
        "source_texts": [x["source_text"] for x in batch],
    }


def default_train_dev_materialize(
    train_limit: int | None,
    dev_limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Materialize train and dev splits (streaming under the hood)."""
    train_ex = materialize_examples(SPLIT_TRAIN, limit=train_limit)
    dev_ex = materialize_examples(SPLIT_DEV, limit=dev_limit)
    return train_ex, dev_ex
