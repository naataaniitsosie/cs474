"""Smoke tests for training entrypoints (no GPU / no Hub required for scratch path)."""

from __future__ import annotations

from pathlib import Path

import pytest

from briefme.train_scratch_loop import ScratchTrainConfig, pick_device, run_scratch_training, scratch_dataloaders_from_examples
from briefme.train_t5_loop import T5BaselineTrainConfig, run_t5_baseline_training


def test_scratch_train_config_defaults() -> None:
    c = ScratchTrainConfig()
    assert c.preset in ("tiny", "small", "medium")
    assert isinstance(c.output_dir, Path)
    assert "train_limit" not in ScratchTrainConfig.__dataclass_fields__


def test_t5_baseline_train_config_defaults() -> None:
    c = T5BaselineTrainConfig()
    assert c.epochs >= 1
    assert "train_limit" not in T5BaselineTrainConfig.__dataclass_fields__


def test_pick_device_returns_torch_device() -> None:
    import torch

    d = pick_device("cpu")
    assert isinstance(d, torch.device)


def test_training_functions_are_callable() -> None:
    assert callable(run_scratch_training)
    assert callable(run_t5_baseline_training)


class _DummyTokenizerForScratchTest:
    """Minimal tokenizer API for scratch loop (no HF download)."""

    pad_token_id = 0
    eos_token_id = 1
    decoder_start_token_id = 2
    vocab_size = 32000

    def encode(self, text: str, **_kwargs: object) -> list[int]:
        n = max(3, min(24, len(text) // 2 + 3))
        return [10 + (i % 100) for i in range(n)]

    def batch_decode(self, ids_list: list, *, skip_special_tokens: bool = True) -> list[str]:
        del skip_special_tokens
        return [" ".join(str(x) for x in row[:8]) for row in ids_list]


def test_run_scratch_training_step_granular_history(tmp_path: Path) -> None:
    """Mid-epoch train logs + eval rows include global_step when logging_steps / eval_steps are set."""
    ex_t = [{"source": "Argument text one.", "target": "Heading one", "metadata": {}} for _ in range(4)]
    ex_d = [{"source": "Argument text two.", "target": "Heading two", "metadata": {}} for _ in range(2)]
    tok = _DummyTokenizerForScratchTest()
    train_loader, dev_loader = scratch_dataloaders_from_examples(
        ex_t,
        ex_d,
        tok,
        preset="tiny",
        source_prefix="summarize: ",
        batch_size=2,
    )
    cfg = ScratchTrainConfig(
        preset="tiny",
        epochs=1,
        device="cpu",
        output_dir=tmp_path / "scratch_steps",
        max_new_tokens_eval=8,
        logging_steps=1,
        eval_steps=2,
    )
    history = run_scratch_training(cfg, tokenizer=tok, train_loader=train_loader, dev_loader=dev_loader)
    kinds = [r.get("kind") for r in history]
    assert "train_log" in kinds
    assert any(r.get("kind") == "eval" for r in history)
    eval_rows = [r for r in history if r.get("dev_loss") is not None]
    assert eval_rows
    assert all("global_step" in r for r in eval_rows)


def test_run_scratch_training_with_dataloaders_only(tmp_path: Path) -> None:
    """Training runs from explicit DataLoaders; no Hub materialize inside the loop."""
    ex_t = [{"source": "Argument text one.", "target": "Heading one", "metadata": {}} for _ in range(4)]
    ex_d = [{"source": "Argument text two.", "target": "Heading two", "metadata": {}} for _ in range(2)]
    tok = _DummyTokenizerForScratchTest()
    train_loader, dev_loader = scratch_dataloaders_from_examples(
        ex_t,
        ex_d,
        tok,
        preset="tiny",
        source_prefix="summarize: ",
        batch_size=2,
    )
    cfg = ScratchTrainConfig(
        preset="tiny",
        epochs=1,
        device="cpu",
        output_dir=tmp_path / "scratch_dl",
        max_new_tokens_eval=16,
    )
    run_scratch_training(cfg, tokenizer=tok, train_loader=train_loader, dev_loader=dev_loader)
