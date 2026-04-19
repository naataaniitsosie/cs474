"""Phase 4: collation and teacher-forcing alignment."""

from __future__ import annotations

import pytest

from briefme.seq2seq_data import (
    BriefMeSeq2SeqDataset,
    collate_seq2seq_batch,
    encode_pair_lists,
    get_t5_tokenizer,
    parse_cli_train_limit,
)


def _t5_tokenizer_or_skip():
    """Skip when HF hub or cache is unavailable (offline CI/sandbox)."""
    try:
        return get_t5_tokenizer(local_files_only=True)
    except Exception as exc:
        pytest.skip(f"T5 tokenizer cache / hub unavailable: {exc}")


def test_collate_seq2seq_batch_padding() -> None:
    batch = [
        {
            "src": [1, 2],
            "tgt_in": [0, 3, 4],
            "labels": [3, 4, 1],
            "source_text": "a",
            "target_text": "b",
        },
        {
            "src": [1],
            "tgt_in": [0, 5],
            "labels": [5, 1],
            "source_text": "c",
            "target_text": "d",
        },
    ]
    out = collate_seq2seq_batch(batch, pad_token_id=0)
    assert out["src"].tolist() == [[1, 2], [1, 0]]
    assert out["tgt_in"].tolist() == [[0, 3, 4], [0, 5, 0]]
    assert out["labels"].tolist() == [[3, 4, 1], [5, 1, 0]]
    assert out["reference_texts"] == ["b", "d"]


@pytest.mark.parametrize(
    "source,target",
    [
        ("The court held that the statute applies.", "Standard of review"),
        ("Short passage.", "Heading"),
    ],
)
def test_encode_pair_lists_teacher_forcing_shift(source: str, target: str) -> None:
    tokenizer = _t5_tokenizer_or_skip()
    max_src, max_tgt = 64, 32
    row = encode_pair_lists(
        tokenizer,
        source,
        target,
        max_src_len=max_src,
        max_tgt_len=max_tgt,
        source_prefix="summarize: ",
    )
    dec_start = getattr(tokenizer, "decoder_start_token_id", None) or tokenizer.pad_token_id
    assert row["tgt_in"][0] == dec_start
    assert len(row["tgt_in"]) == len(row["labels"])
    assert row["labels"][-1] == tokenizer.eos_token_id
    assert row["tgt_in"][1:] == row["labels"][:-1]


def test_briefme_dataset_len_and_keys() -> None:
    tokenizer = _t5_tokenizer_or_skip()
    examples = [{"source": "Hello world.", "target": "Greeting"}]
    ds = BriefMeSeq2SeqDataset(
        examples,
        tokenizer,
        max_src_len=32,
        max_tgt_len=16,
    )
    assert len(ds) == 1
    item = ds[0]
    assert "src" in item and "tgt_in" in item and "labels" in item


def test_parse_cli_train_limit() -> None:
    assert parse_cli_train_limit("none") is None
    assert parse_cli_train_limit("FULL") is None
    assert parse_cli_train_limit("all") is None
    assert parse_cli_train_limit("2048") == 2048
