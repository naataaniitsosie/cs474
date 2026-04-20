"""Persist seq2seq inference + aggregate() results as JSON for notebook viewers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

INFERENCE_VERSION = 1


def inference_runs_dir(repo_root: Path) -> Path:
    return repo_root / "artifacts" / "inference_runs"


def scratch_runs_artifacts_dir(repo_root: Path, *, artifacts_subdir: str = "inference_runs") -> Path:
    """Directory under ``artifacts/`` holding ``{split}_scratch_{label}.json`` (greedy, beam, etc.)."""
    return repo_root / "artifacts" / artifacts_subdir


def paired_llm_judge_artifacts_subdir(inference_artifacts_subdir: str) -> str:
    """Maps ``inference_runs`` → ``llm_judge_runs``, ``inference_runs_beam4`` → ``llm_judge_runs_beam4``, etc."""
    if not inference_artifacts_subdir.startswith("inference"):
        return "llm_judge_runs"
    return "llm_judge" + inference_artifacts_subdir[len("inference") :]


def llm_judge_runs_dir(repo_root: Path, *, inference_artifacts_subdir: str = "inference_runs") -> Path:
    """LLM judge JSONL cache dir paired with :func:`scratch_runs_artifacts_dir` layout."""
    sub = paired_llm_judge_artifacts_subdir(inference_artifacts_subdir)
    return repo_root / "artifacts" / sub


def scratch_inference_json_path(
    repo_root: Path,
    *,
    split_tag: str,
    label: str,
    artifacts_subdir: str = "inference_runs",
) -> Path:
    """Path to saved inference JSON (default folder ``inference_runs``; use ``inference_runs_beam4`` for beam)."""
    return scratch_runs_artifacts_dir(repo_root, artifacts_subdir=artifacts_subdir) / (
        f"{split_tag}_scratch_{label}.json"
    )


def save_scratch_inference_json(
    path: Path,
    *,
    split_tag: str,
    eval_n: int,
    label: str,
    checkpoint: Path | str,
    preds: list[str],
    refs: list[str],
    sources: list[str],
    agg: dict[str, Any],
) -> None:
    """Write one scratch checkpoint run (predictions + full aggregate including per-example)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": INFERENCE_VERSION,
        "kind": "scratch_seq2seq",
        "split_tag": split_tag,
        "eval_n": eval_n,
        "label": label,
        "checkpoint": str(checkpoint),
        "preds": preds,
        "refs": refs,
        "sources": sources,
        "aggregate": agg,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_inference_json(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def list_saved_scratch_runs(repo_root: Path, *, split_tag: str | None = None) -> list[Path]:
    """Return JSON paths under ``artifacts/inference_runs``, newest first by mtime."""
    d = inference_runs_dir(repo_root)
    if not d.is_dir():
        return []
    paths = [p for p in d.glob("*_scratch_*.json") if p.is_file()]
    if split_tag is not None:
        paths = [p for p in paths if p.name.startswith(f"{split_tag}_")]
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths
