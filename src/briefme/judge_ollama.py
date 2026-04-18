"""Local Ollama LLM-as-judge for BriefMe-style heading quality (1–5 rubric).

Evaluation-only helper: does not replace seq2seq training. Log ``model`` and
``prompt_version`` on every call for reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

# Frozen prompt version — bump when rubric or instructions change.
JUDGE_PROMPT_VERSION = "briefme_heading_judge_v1"

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """You are an expert evaluator for legal brief argument summarization (BriefMe / arg_summ).

Task: Rate how well the CANDIDATE HEADING fits the PASSAGE as a section heading that reflects the argument or content described.

Scale (integer 1–5 only):
1 — Misleading, irrelevant, or contradicts the passage.
2 — Weak match; misses the main point or is overly vague.
3 — Adequate; partly aligned but imprecise or incomplete.
4 — Strong; accurate and specific with at most minor gaps.
5 — Excellent; concise, accurate, and well aligned with the passage.

If a REFERENCE HEADING (gold) is provided, use it only as a calibration hint; the score must still reflect how well the CANDIDATE matches the PASSAGE.

Output rules:
- Respond with one JSON object only. No markdown fences, no prose before or after.
- Schema: {"score": <integer 1-5>, "rationale": "<one short sentence>"}
"""


def _env_model() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()


def _env_host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST).strip().rstrip("/")


def cache_key(
    passage: str,
    candidate_heading: str,
    *,
    reference_heading: str | None,
    model: str,
    prompt_version: str = JUDGE_PROMPT_VERSION,
) -> str:
    """Stable hash for optional JSONL deduplication."""
    payload = {
        "passage": passage,
        "candidate_heading": candidate_heading,
        "reference_heading": reference_heading,
        "model": model,
        "prompt_version": prompt_version,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _build_user_content(
    passage: str,
    candidate_heading: str,
    reference_optional: str | None,
) -> str:
    ref_block = (
        reference_optional.strip()
        if reference_optional is not None and str(reference_optional).strip()
        else "(none provided)"
    )
    return (
        "PASSAGE:\n"
        f"{passage.strip()}\n\n"
        "CANDIDATE HEADING:\n"
        f"{candidate_heading.strip()}\n\n"
        "REFERENCE HEADING (gold, optional):\n"
        f"{ref_block}\n"
    )


_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    return t


def parse_judge_json(text: str) -> dict[str, Any]:
    """Parse model output into ``score`` and ``rationale``; repair common drift."""
    cleaned = _strip_code_fences(text)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        m = _JSON_OBJECT_RE.search(cleaned)
        if not m:
            raise ValueError(f"No JSON object found in model output: {text[:500]!r}") from None
        obj = json.loads(m.group(0))

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj).__name__}")
    score = obj.get("score")
    rationale = obj.get("rationale", "")
    if not isinstance(score, int) or score < 1 or score > 5:
        # Allow string digits from sloppy models
        if isinstance(score, str) and score.strip().isdigit():
            score = int(score.strip())
        if not isinstance(score, int) or score < 1 or score > 5:
            raise ValueError(f"Invalid score in JSON: {obj!r}")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    return {"score": score, "rationale": rationale.strip()}


def load_jsonl_cache(path: Path | str) -> dict[str, dict[str, Any]]:
    """Load ``cache_key`` -> cached row from an append-only JSONL file."""
    p = Path(path)
    out: dict[str, dict[str, Any]] = {}
    if not p.is_file():
        return out
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("cache_key")
            if isinstance(key, str):
                out[key] = row
    return out


def append_jsonl(path: Path | str, record: Mapping[str, Any]) -> None:
    """Append one JSON object per line (creates parent dirs)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record), ensure_ascii=False) + "\n"
    with p.open("a", encoding="utf-8") as f:
        f.write(line)


def _chat(
    host: str,
    model: str,
    user_content: str,
    *,
    timeout_s: float = 120.0,
    max_http_retries: int = 3,
) -> str:
    url = f"{host}/api/chat"
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"temperature": 0},
    }
    for attempt in range(max_http_retries):
        try:
            with httpx.Client(timeout=timeout_s) as client:
                r = client.post(url, json=body)
                r.raise_for_status()
                data = r.json()
            msg = data.get("message") or {}
            content = msg.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Empty message content in Ollama response: {data!r}")
            return content.strip()
        except (httpx.HTTPError, ValueError):
            if attempt < max_http_retries - 1:
                time.sleep(0.5 * (2**attempt))
            else:
                raise


def score_heading(
    passage: str,
    candidate_heading: str,
    reference_optional: str | None = None,
    *,
    model: str | None = None,
    host: str | None = None,
    cache_path: str | Path | None = None,
    use_cache: bool = True,
    parse_retries: int = 1,
) -> dict[str, Any]:
    """Score ``candidate_heading`` against ``passage`` via local Ollama.

    Returns a dict including ``score`` (1–5), ``rationale``, ``model``,
    ``prompt_version``, and ``cache_key``. Optional ``cache_path`` JSONL avoids
    repeat inference for identical inputs + model + prompt version.
    """
    m = model if model is not None else _env_model()
    h = host if host is not None else _env_host()
    key = cache_key(
        passage,
        candidate_heading,
        reference_heading=reference_optional,
        model=m,
        prompt_version=JUDGE_PROMPT_VERSION,
    )

    if cache_path is not None and use_cache:
        cache = load_jsonl_cache(cache_path)
        if key in cache:
            hit = dict(cache[key])
            hit["cache_hit"] = True
            return hit

    user_content = _build_user_content(passage, candidate_heading, reference_optional)
    raw = _chat(h, m, user_content)
    parsed: dict[str, Any] | None = None
    last_parse_err: Exception | None = None
    fix_suffix = (
        "\n\nYour previous reply was not valid JSON. "
        "Reply with ONLY one JSON object: "
        '{"score": <1-5 integer>, "rationale": "<one short sentence>"}'
    )
    for attempt in range(parse_retries + 1):
        try:
            parsed = parse_judge_json(raw)
            break
        except (json.JSONDecodeError, ValueError) as e:
            last_parse_err = e
            if attempt >= parse_retries:
                raise ValueError(f"Failed to parse judge JSON after retries: {raw!r}") from last_parse_err
            raw = _chat(h, m, user_content + fix_suffix)

    assert parsed is not None
    result: dict[str, Any] = {
        "score": parsed["score"],
        "rationale": parsed["rationale"],
        "model": m,
        "prompt_version": JUDGE_PROMPT_VERSION,
        "cache_key": key,
        "cache_hit": False,
    }
    if cache_path is not None:
        append_jsonl(cache_path, result)
    return result
