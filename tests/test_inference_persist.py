"""Tests for briefme.inference_persist."""

from __future__ import annotations

from pathlib import Path

from briefme.inference_persist import (
    inference_runs_dir,
    llm_judge_runs_dir,
    load_inference_json,
    paired_llm_judge_artifacts_subdir,
    save_scratch_inference_json,
    scratch_inference_json_path,
    scratch_runs_artifacts_dir,
)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    agg = {
        "n": 2,
        "rouge1_f": 0.5,
        "rouge2_f": 0.0,
        "rougeL_f": 0.5,
        "exact_match_rate": 0.0,
        "token_f1_macro": 0.4,
        "chrf_corpus": 0.3,
        "per_example": {
            "rouge1_f": [0.0, 1.0],
            "rouge2_f": [0.0, 0.0],
            "rougeL_f": [0.0, 1.0],
            "exact_match": [0.0, 1.0],
            "token_f1": [0.0, 1.0],
            "chrf": [0.1, 0.2],
        },
    }
    out = tmp_path / "dev_scratch_tiny.json"
    save_scratch_inference_json(
        out,
        split_tag="dev",
        eval_n=2,
        label="tiny",
        checkpoint="/fake/best.pt",
        preds=["a", "b"],
        refs=["a", "c"],
        sources=["passage one", "passage two"],
        agg=agg,
    )
    loaded = load_inference_json(out)
    assert loaded["label"] == "tiny"
    assert loaded["preds"] == ["a", "b"]
    assert loaded["aggregate"]["n"] == 2
    assert loaded["sources"][1] == "passage two"


def test_inference_runs_dir(tmp_path: Path) -> None:
    assert inference_runs_dir(tmp_path) == tmp_path / "artifacts" / "inference_runs"


def test_scratch_inference_json_path(tmp_path: Path) -> None:
    p = scratch_inference_json_path(tmp_path, split_tag="dev", label="tiny")
    assert p.name == "dev_scratch_tiny.json"
    assert p.parent == inference_runs_dir(tmp_path)


def test_scratch_inference_json_path_beam_subdir(tmp_path: Path) -> None:
    p = scratch_inference_json_path(
        tmp_path,
        split_tag="dev",
        label="tiny",
        artifacts_subdir="inference_runs_beam4",
    )
    assert p.name == "dev_scratch_tiny.json"
    assert p.parent == scratch_runs_artifacts_dir(tmp_path, artifacts_subdir="inference_runs_beam4")


def test_paired_llm_judge_subdir() -> None:
    assert paired_llm_judge_artifacts_subdir("inference_runs") == "llm_judge_runs"
    assert paired_llm_judge_artifacts_subdir("inference_runs_beam4") == "llm_judge_runs_beam4"


def test_llm_judge_runs_dir_beam(tmp_path: Path) -> None:
    d = llm_judge_runs_dir(tmp_path, inference_artifacts_subdir="inference_runs_beam4")
    assert d == tmp_path / "artifacts" / "llm_judge_runs_beam4"
