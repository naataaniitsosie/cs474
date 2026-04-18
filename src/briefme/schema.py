"""Single source of truth: Hugging Face BriefMe `arg_summ` column mapping."""

from __future__ import annotations

from typing import Any, Mapping

# Hub identifiers
DATASET_NAME = "jw4202/BriefMe"
CONFIG_NAME = "arg_summ"

# Argument summarization: passage -> gold section heading (see dataset card)
SOURCE_COLUMN = "text"
TARGET_COLUMN = "reference"

# Columns used for dataset QA in BriefMe; not used as seq2seq labels by default
JUDGE_COLUMNS = ("judge_outcome", "judge_score_verb", "judge_score")


def to_seq2seq_example(row: Mapping[str, Any]) -> dict[str, Any]:
    """Map a Hugging Face row dict to encoder/decoder strings plus optional metadata."""
    meta: dict[str, Any] = {}
    if "file" in row and row["file"] is not None:
        meta["file"] = row["file"]
    return {
        "source": row[SOURCE_COLUMN],
        "target": row[TARGET_COLUMN],
        "metadata": meta,
    }
