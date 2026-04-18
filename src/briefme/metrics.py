"""Automated evaluation for BriefMe heading prediction (short English strings).

ROUGE via ``rouge-score`` (reference, then prediction order). chrF via ``sacrebleu``
(corpus score uses one reference string per hypothesis). Token F1 uses the same
``normalize()`` + whitespace splitting for prediction and reference (macro =
mean of per-example F1).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from rouge_score import rouge_scorer
from sacrebleu.metrics import CHRF

_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
_CHRF = CHRF()


def normalize(text: str) -> str:
    """Lowercase, strip ends, collapse internal whitespace (no stemming)."""
    s = text.strip().lower()
    return re.sub(r"\s+", " ", s)


def token_f1_one(prediction: str, reference: str) -> float:
    """Macro-style token F1 for one pair: bag overlap / token-list lengths.

    Tokens are ``normalize(text).split()``. Empty reference and empty prediction
    → 1.0; one empty → 0.0.
    """
    pt = normalize(prediction).split()
    rt = normalize(reference).split()
    if not pt and not rt:
        return 1.0
    if not pt or not rt:
        return 0.0
    cp, cr = Counter(pt), Counter(rt)
    overlap = sum((cp & cr).values())
    prec = overlap / len(pt)
    rec = overlap / len(rt)
    if prec + rec == 0:
        return 0.0
    return 2.0 * prec * rec / (prec + rec)


def score_pair(prediction: str, reference: str) -> dict[str, Any]:
    """Per-example scores (ROUGE F1, EM, token F1, sentence chrF).

    ROUGE ``fmeasure`` fields are 0–1. chrF sentence score is mapped to 0–1
    (SacreBLEU uses 0–100).
    """
    r_scores = _ROUGE.score(reference, prediction)
    chrf_sent = _CHRF.sentence_score(prediction, [reference])
    chrf_01 = float(chrf_sent.score) / 100.0

    em = 1.0 if normalize(prediction) == normalize(reference) else 0.0
    tf1 = token_f1_one(prediction, reference)

    return {
        "rouge1_f": float(r_scores["rouge1"].fmeasure),
        "rouge2_f": float(r_scores["rouge2"].fmeasure),
        "rougeL_f": float(r_scores["rougeL"].fmeasure),
        "exact_match": float(em),
        "token_f1": float(tf1),
        "chrf": float(chrf_01),
    }


def aggregate(
    predictions: list[str],
    references: list[str],
) -> dict[str, Any]:
    """Corpus aggregates plus parallel per-example lists for plotting.

    **chrF corpus:** single ``CHRF().corpus_score`` over all pairs (one reference
    each). **chrF per row:** sentence-level score (can differ from corpus slice).

    **token_f1_macro:** arithmetic mean of per-example ``token_f1``.

    Lists must have equal length.
    """
    if len(predictions) != len(references):
        raise ValueError(
            f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
        )
    n = len(predictions)
    if n == 0:
        return {
            "n": 0,
            "rouge1_f": 0.0,
            "rouge2_f": 0.0,
            "rougeL_f": 0.0,
            "exact_match_rate": 0.0,
            "token_f1_macro": 0.0,
            "chrf_corpus": 0.0,
            "per_example": {
                "rouge1_f": [],
                "rouge2_f": [],
                "rougeL_f": [],
                "exact_match": [],
                "token_f1": [],
                "chrf": [],
            },
        }

    per: dict[str, list[float]] = {
        "rouge1_f": [],
        "rouge2_f": [],
        "rougeL_f": [],
        "exact_match": [],
        "token_f1": [],
        "chrf": [],
    }

    sum_r1 = sum_r2 = sum_rl = 0.0
    sum_em = sum_tf1 = 0.0

    for pred, ref in zip(predictions, references, strict=True):
        row = score_pair(pred, ref)
        per["rouge1_f"].append(row["rouge1_f"])
        per["rouge2_f"].append(row["rouge2_f"])
        per["rougeL_f"].append(row["rougeL_f"])
        per["exact_match"].append(row["exact_match"])
        per["token_f1"].append(row["token_f1"])
        per["chrf"].append(row["chrf"])
        sum_r1 += row["rouge1_f"]
        sum_r2 += row["rouge2_f"]
        sum_rl += row["rougeL_f"]
        sum_em += row["exact_match"]
        sum_tf1 += row["token_f1"]

    ref_streams = [[r] for r in references]
    corpus = _CHRF.corpus_score(predictions, ref_streams)
    chrf_corpus = float(corpus.score) / 100.0

    return {
        "n": n,
        "rouge1_f": sum_r1 / n,
        "rouge2_f": sum_r2 / n,
        "rougeL_f": sum_rl / n,
        "exact_match_rate": sum_em / n,
        "token_f1_macro": sum_tf1 / n,
        "chrf_corpus": chrf_corpus,
        "per_example": per,
    }
