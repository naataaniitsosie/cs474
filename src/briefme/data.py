"""Load BriefMe `arg_summ` from Hugging Face.

The Hub dataset builder fails during non-streaming prepare on the ``held_out`` split
(schema mismatch: extra ``Metrics`` column vs declared features). Loading with
``streaming=True`` reads split-by-split and avoids that failure. Use streaming for EDA
and either keep streaming / IterableDataset for training or materialize rows yourself.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from datasets import Dataset, IterableDataset, load_dataset

from briefme.schema import CONFIG_NAME, DATASET_NAME

# Hub split names for arg_summ (dev = validation in the paper)
SPLIT_TRAIN = "train"
SPLIT_DEV = "dev"
SPLIT_TEST = "test"


def load_arg_summ_split_streaming(split: str = SPLIT_TRAIN) -> IterableDataset:
    """IterableDataset for one split; safe while the Hub cache builder is broken."""
    return load_dataset(
        DATASET_NAME,
        CONFIG_NAME,
        split=split,
        streaming=True,
    )


def load_arg_summ_train_dev_test_streaming() -> dict[str, IterableDataset]:
    """Train / dev / test as streaming iterables (no ``held_out`` access here)."""
    return {
        SPLIT_TRAIN: load_arg_summ_split_streaming(SPLIT_TRAIN),
        SPLIT_DEV: load_arg_summ_split_streaming(SPLIT_DEV),
        SPLIT_TEST: load_arg_summ_split_streaming(SPLIT_TEST),
    }


def materialize_head(stream: IterableDataset, n: int) -> Dataset:
    """First ``n`` examples as a non-streaming ``Dataset`` (for indexing / len)."""
    rows: list[dict[str, Any]] = list(stream.take(n))
    return Dataset.from_list(rows)


def iter_rows(stream: IterableDataset, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Iterate up to ``limit`` rows from a streaming split."""
    it: Iterable[dict[str, Any]] = stream if limit is None else stream.take(limit)
    yield from it
