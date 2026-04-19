"""T5-small baseline training (shared by CLI and notebooks)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5TokenizerFast,
)

from briefme.metrics import aggregate
from briefme.seq2seq_data import DEFAULT_T5_TASK_PREFIX
from transformer.config import HF_T5_BASELINE_MODEL_ID


@dataclass
class T5BaselineTrainConfig:
    """Hyperparameters for T5 finetuning; data come from ``train_loader`` / ``eval_loader``."""

    model_id: str = HF_T5_BASELINE_MODEL_ID
    output_dir: Path = field(default_factory=lambda: Path("runs/t5_baseline"))
    epochs: int = 2
    lr: float = 3e-4
    max_src_len: int = 512
    max_tgt_len: int = 128
    source_prefix: str = DEFAULT_T5_TASK_PREFIX
    seed: int = 42
    logging_steps: int = field(
        default=50,
        metadata={"help": "Log train loss every N steps (Trainer logging_strategy)."},
    )
    eval_steps: int | None = field(
        default=None,
        metadata={
            "help": "If set, run evaluation (generate + metrics) every N steps; "
            "if unset, evaluate once per epoch."
        },
    )


class _Seq2SeqTrainerWithLoaders(Seq2SeqTrainer):
    """Uses caller-built PyTorch DataLoaders instead of constructing them from datasets."""

    def __init__(
        self,
        *args: Any,
        external_train_dataloader: DataLoader | None = None,
        external_eval_dataloader: DataLoader | None = None,
        **kwargs: Any,
    ) -> None:
        self._external_train_dataloader = external_train_dataloader
        self._external_eval_dataloader = external_eval_dataloader
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        if self._external_train_dataloader is not None:
            return self._external_train_dataloader
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Any = None) -> DataLoader:
        if self._external_eval_dataloader is not None:
            return self._external_eval_dataloader
        return super().get_eval_dataloader(eval_dataset)


def t5_dataloaders_from_examples(
    train_examples: list[dict[str, Any]],
    dev_examples: list[dict[str, Any]],
    tokenizer: T5TokenizerFast,
    model: T5ForConditionalGeneration,
    cfg: T5BaselineTrainConfig,
    *,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataCollatorForSeq2Seq]:
    """Tokenize materialized rows and return train/eval DataLoaders."""

    def preprocess(batch: dict[str, Any]) -> dict[str, Any]:
        inputs = [f"{cfg.source_prefix}{s}" for s in batch["source"]]
        model_inputs = tokenizer(
            inputs,
            max_length=cfg.max_src_len,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target"],
            max_length=cfg.max_tgt_len,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = Dataset.from_dict(
        {"source": [e["source"] for e in train_examples], "target": [e["target"] for e in train_examples]}
    )
    dev_ds = Dataset.from_dict(
        {"source": [e["source"] for e in dev_examples], "target": [e["target"] for e in dev_examples]}
    )
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    dev_ds = dev_ds.map(preprocess, batched=True, remove_columns=dev_ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    return train_loader, eval_loader, collator


def run_t5_baseline_training(
    cfg: T5BaselineTrainConfig,
    *,
    tokenizer: T5TokenizerFast,
    model: T5ForConditionalGeneration,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    data_collator: DataCollatorForSeq2Seq,
) -> dict:
    """Finetune T5 with ``Seq2SeqTrainer``; write ``summary.json`` under ``output_dir``."""
    print("[briefme] starting T5 Seq2SeqTrainer...", file=sys.stderr, flush=True)

    def compute_metrics(eval_preds: Any) -> dict[str, float]:
        preds, labels_arr = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels_np = np.where(labels_arr != -100, labels_arr, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels_np, skip_special_tokens=True)
        agg = aggregate(decoded_preds, decoded_labels)
        return {
            "rougeL_f": float(agg["rougeL_f"]),
            "token_f1_macro": float(agg["token_f1_macro"]),
            "exact_match_rate": float(agg["exact_match_rate"]),
            "chrf_corpus": float(agg["chrf_corpus"]),
        }

    per_device_bs = train_loader.batch_size
    if per_device_bs is None:
        raise ValueError("train_loader must have a fixed batch_size")

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log_steps = max(1, cfg.logging_steps)
    use_step_eval = cfg.eval_steps is not None and cfg.eval_steps > 0
    if use_step_eval:
        ev = int(cfg.eval_steps)
        targs = Seq2SeqTrainingArguments(
            output_dir=str(cfg.output_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=per_device_bs,
            per_device_eval_batch_size=per_device_bs,
            learning_rate=cfg.lr,
            eval_strategy="steps",
            eval_steps=ev,
            save_strategy="steps",
            save_steps=ev,
            logging_strategy="steps",
            logging_steps=log_steps,
            predict_with_generate=True,
            generation_max_length=cfg.max_tgt_len,
            save_total_limit=2,
            seed=cfg.seed,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL_f",
            greater_is_better=True,
        )
    else:
        targs = Seq2SeqTrainingArguments(
            output_dir=str(cfg.output_dir),
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=per_device_bs,
            per_device_eval_batch_size=per_device_bs,
            learning_rate=cfg.lr,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=log_steps,
            predict_with_generate=True,
            generation_max_length=cfg.max_tgt_len,
            save_total_limit=2,
            seed=cfg.seed,
            load_best_model_at_end=True,
            metric_for_best_model="rougeL_f",
            greater_is_better=True,
        )

    trainer = _Seq2SeqTrainerWithLoaders(
        model=model,
        args=targs,
        train_dataset=train_loader.dataset,
        eval_dataset=eval_loader.dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        external_train_dataloader=train_loader,
        external_eval_dataloader=eval_loader,
    )
    train_out = trainer.train()
    metrics = trainer.evaluate()
    log_history = [dict(h) for h in getattr(trainer.state, "log_history", [])]
    out = {"train": train_out.metrics, "eval": metrics, "log_history": log_history}
    with open(cfg.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


__all__ = ["T5BaselineTrainConfig", "run_t5_baseline_training", "t5_dataloaders_from_examples"]
