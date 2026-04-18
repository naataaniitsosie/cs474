"""Unit tests for briefme.metrics heading evaluation."""

from __future__ import annotations

import pytest

from briefme.metrics import aggregate, normalize, score_pair, token_f1_one


def test_normalize_collapses_whitespace() -> None:
    assert normalize("  Hello   World.\n") == "hello world."


def test_token_f1_identical() -> None:
    assert token_f1_one("Motion to Dismiss", "motion to dismiss") == pytest.approx(1.0)


def test_token_f1_disjoint() -> None:
    assert token_f1_one("alpha beta", "gamma delta") == pytest.approx(0.0)


def test_score_pair_exact_paraphrase_spacing() -> None:
    s = score_pair("  Summary Judgment  ", "summary judgment")
    assert s["exact_match"] == pytest.approx(1.0)
    assert s["rougeL_f"] == pytest.approx(1.0)
    assert s["token_f1"] == pytest.approx(1.0)


def test_score_pair_typo_lower_scores() -> None:
    ref = "Discovery sanctions under Rule 37"
    good = ref
    bad = "Discoveree sanctions under Rule 37"
    sg = score_pair(good, ref)
    sb = score_pair(bad, ref)
    assert sg["rougeL_f"] >= sb["rougeL_f"]
    assert sg["token_f1"] >= sb["token_f1"]


def test_score_pair_unrelated() -> None:
    s = score_pair("weather forecast", "statute of limitations")
    assert s["exact_match"] == pytest.approx(0.0)
    assert s["rougeL_f"] < 0.5


def test_aggregate_matches_single_means() -> None:
    preds = ["a b c", "x y"]
    refs = ["a b c", "x y z"]
    out = aggregate(preds, refs)
    assert out["n"] == 2
    r0 = score_pair(preds[0], refs[0])
    r1 = score_pair(preds[1], refs[1])
    assert out["rouge1_f"] == pytest.approx((r0["rouge1_f"] + r1["rouge1_f"]) / 2)
    assert out["token_f1_macro"] == pytest.approx((r0["token_f1"] + r1["token_f1"]) / 2)
    assert len(out["per_example"]["rougeL_f"]) == 2


def test_aggregate_empty() -> None:
    out = aggregate([], [])
    assert out["n"] == 0
    assert out["chrf_corpus"] == 0.0


def test_aggregate_length_mismatch() -> None:
    with pytest.raises(ValueError):
        aggregate(["a"], ["a", "b"])
