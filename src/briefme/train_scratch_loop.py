"""Scratch seq2seq training loop (shared by CLI and notebooks)."""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from briefme.generation import batch_decode_skip_special, greedy_generate, strip_decoder_start
from briefme.metrics import aggregate
from briefme.seq2seq_data import (
    BriefMeSeq2SeqDataset,
    DEFAULT_T5_TASK_PREFIX,
    collate_seq2seq_batch,
)
from transformer import ScratchSeq2SeqTransformer, ScratchTransformerConfig, seq2seq_cross_entropy_loss
from transformer.config import HF_T5_BASELINE_MODEL_ID


def pick_device(explicit: str | None) -> torch.device:
    if explicit and explicit != "auto":
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preset_to_transformer_config(preset: str) -> ScratchTransformerConfig:
    if preset == "tiny":
        return ScratchTransformerConfig.tiny()
    if preset == "small":
        return ScratchTransformerConfig.small()
    if preset == "medium":
        return ScratchTransformerConfig.medium()
    raise ValueError(f"Unknown preset: {preset!r}")


def scratch_dataloaders_from_examples(
    train_examples: list[dict[str, Any]],
    dev_examples: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    *,
    preset: str,
    source_prefix: str,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Build train/dev DataLoaders from materialized row dicts.

    Use :func:`briefme.seq2seq_data.default_train_dev_materialize` (or any ``list[dict]`` with
    ``source`` / ``target``) at the call site; limits and batch size live outside
    :class:`ScratchTrainConfig`.
    """
    tcfg = preset_to_transformer_config(preset)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id is required")

    ds_train = BriefMeSeq2SeqDataset(
        train_examples,
        tokenizer,
        max_src_len=tcfg.max_src_len,
        max_tgt_len=tcfg.max_tgt_len,
        source_prefix=source_prefix,
    )
    ds_dev = BriefMeSeq2SeqDataset(
        dev_examples,
        tokenizer,
        max_src_len=tcfg.max_src_len,
        max_tgt_len=tcfg.max_tgt_len,
        source_prefix=source_prefix,
    )

    def collate(batch: list) -> dict[str, Any]:
        return collate_seq2seq_batch(batch, pad_token_id=pad_id)

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    dev_loader = DataLoader(ds_dev, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)
    return train_loader, dev_loader


@dataclass
class ScratchTrainConfig:
    """Hyperparameters and checkpoint paths for :func:`run_scratch_training`.

    Data iteration is exclusively via the ``train_loader`` / ``dev_loader`` arguments to
    :func:`run_scratch_training`; row counts and batch size are determined when those loaders
    are built (e.g. :func:`scratch_dataloaders_from_examples`).

    Set ``logging_steps`` / ``eval_steps`` for step-granular curves (see ``history.json``).
    """

    preset: str = field(
        default="tiny",
        metadata={"help": 'ScratchTransformerConfig preset: "tiny", "small", or "medium".'},
    )
    epochs: int = field(default=3, metadata={"help": "Training epochs."})
    lr: float = field(default=3e-4, metadata={"help": "AdamW learning rate."})
    weight_decay: float = field(default=0.01, metadata={"help": "AdamW weight decay."})
    grad_clip: float = field(
        default=1.0,
        metadata={"help": "Gradient clip max norm; 0 disables."},
    )
    device: str = field(
        default="auto",
        metadata={"help": 'Device: "auto", "cuda", "mps", or "cpu".'},
    )
    output_dir: Path = field(
        default_factory=lambda: Path("runs/scratch_seq2seq"),
        metadata={"help": "Directory for best.pt and history.json."},
    )
    source_prefix: str = field(
        default=DEFAULT_T5_TASK_PREFIX,
        metadata={"help": "Prepended to encoder source text (T5-style); stored in checkpoint."},
    )
    max_new_tokens_eval: int = field(
        default=128,
        metadata={"help": "Greedy decode step cap per example for dev metrics."},
    )
    logging_steps: int | None = field(
        default=None,
        metadata={"help": "If set (positive), append train-only loss rows every N optimizer steps."},
    )
    eval_steps: int | None = field(
        default=None,
        metadata={
            "help": "If set (positive), run dev CE + greedy metrics every N steps; "
            "epoch-end eval runs when the epoch does not end on a multiple of N."
        },
    )


def run_scratch_training(
    cfg: ScratchTrainConfig,
    *,
    tokenizer: PreTrainedTokenizerBase,
    train_loader: DataLoader,
    dev_loader: DataLoader,
) -> list[dict]:
    """Train scratch ``ScratchSeq2SeqTransformer``; write ``best.pt`` and ``history.json``.

    ``train_loader`` and ``dev_loader`` must yield batches from
    :func:`briefme.seq2seq_data.collate_seq2seq_batch` (e.g. built via
    :func:`scratch_dataloaders_from_examples`).

    ``history.json`` is a chronological list:

    - **Train-only** rows (when ``logging_steps`` is set): ``kind`` is ``\"train_log\"``;
      fields include ``global_step``, ``epoch``, ``train_loss``.
    - **Eval** rows: ``kind`` is ``\"eval\"`` (omitted in older runs); ``global_step``,
      ``epoch``, ``train_loss`` (mean since previous eval), ``dev_loss``, ROUGE/F1, timings.

    When ``eval_steps`` is unset, behavior matches the original single eval per epoch.
    """
    device = pick_device(cfg.device)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    dec_start = getattr(tokenizer, "decoder_start_token_id", None)
    if dec_start is None:
        dec_start = pad_id
    if pad_id is None or eos_id is None:
        raise ValueError("Tokenizer must define pad_token_id and eos_token_id")

    tcfg = preset_to_transformer_config(cfg.preset)
    tok_ref = getattr(tokenizer, "name_or_path", None) or HF_T5_BASELINE_MODEL_ID

    print("[briefme] scratch seq2seq training...", file=sys.stderr, flush=True)
    model = ScratchSeq2SeqTransformer(tcfg, vocab_size=tokenizer.vocab_size, pad_token_id=pad_id)
    model.to(device)

    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_dev = float("inf")
    history: list[dict] = []

    n_train = len(train_loader.dataset)
    n_dev = len(dev_loader.dataset)

    log_every = cfg.logging_steps if cfg.logging_steps and cfg.logging_steps > 0 else None
    eval_every = cfg.eval_steps if cfg.eval_steps and cfg.eval_steps > 0 else None

    global_step = 0

    def write_history() -> None:
        with open(cfg.output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def dev_ce_and_metrics() -> tuple[float, dict[str, float]]:
        dev_running = 0.0
        dev_n = 0
        with torch.no_grad():
            for batch in dev_loader:
                src_b = batch["src"].to(device)
                tgt_in_b = batch["tgt_in"].to(device)
                labels_b = batch["labels"].to(device)
                logits = model(src_b, tgt_in_b)
                loss = seq2seq_cross_entropy_loss(logits, labels_b, ignore_index=pad_id)
                dev_running += float(loss)
                dev_n += 1
        dev_loss = dev_running / max(dev_n, 1)

        preds: list[str] = []
        refs: list[str] = []
        with torch.no_grad():
            for batch in dev_loader:
                src_b = batch["src"].to(device)
                gen = greedy_generate(
                    model,
                    src_b,
                    eos_token_id=eos_id,
                    decoder_start_token_id=dec_start,
                    max_new_tokens=min(cfg.max_new_tokens_eval, tcfg.max_tgt_len),
                )
                gen = strip_decoder_start(gen, dec_start)
                preds.extend(batch_decode_skip_special(tokenizer, gen))
                refs.extend(batch["reference_texts"])

        metrics = aggregate(preds, refs)
        return dev_loss, metrics

    for epoch in range(cfg.epochs):
        t_epoch0 = time.perf_counter()
        t_seg_train0 = t_epoch0
        since_eval_loss_sum = 0.0
        since_eval_n = 0
        log_loss_sum = 0.0
        log_n = 0

        model.train()
        for batch in train_loader:
            src = batch["src"].to(device)
            tgt_in = batch["tgt_in"].to(device)
            labels = batch["labels"].to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(src, tgt_in)
            loss = seq2seq_cross_entropy_loss(logits, labels, ignore_index=pad_id)
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            lv = float(loss.detach())
            global_step += 1
            since_eval_loss_sum += lv
            since_eval_n += 1
            log_loss_sum += lv
            log_n += 1

            if log_every and global_step % log_every == 0:
                history.append(
                    {
                        "kind": "train_log",
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": log_loss_sum / max(log_n, 1),
                    }
                )
                log_loss_sum = 0.0
                log_n = 0
                write_history()

            if eval_every and global_step % eval_every == 0:
                train_loss_mean = since_eval_loss_sum / max(since_eval_n, 1)
                t_before_eval = time.perf_counter()
                train_seg_s = t_before_eval - t_seg_train0
                model.eval()
                t_ev0 = time.perf_counter()
                dev_loss, metrics = dev_ce_and_metrics()
                eval_s = time.perf_counter() - t_ev0
                row = {
                    "kind": "eval",
                    "global_step": global_step,
                    "epoch": epoch + 1,
                    "train_loss": train_loss_mean,
                    "dev_loss": dev_loss,
                    "dev_rougeL_f": metrics["rougeL_f"],
                    "dev_token_f1_macro": metrics["token_f1_macro"],
                    "n_train": n_train,
                    "n_dev": n_dev,
                    "train_seconds": round(train_seg_s, 3),
                    "dev_ce_and_greedy_seconds": round(eval_s, 3),
                    "epoch_seconds": round(time.perf_counter() - t_epoch0, 3),
                }
                history.append(row)
                print(json.dumps(row, indent=2))
                print(
                    f"[step {global_step} · epoch {epoch + 1}/{cfg.epochs}] "
                    f"eval wall {eval_s:.1f}s (train seg {train_seg_s:.1f}s)",
                    flush=True,
                )
                if dev_loss < best_dev:
                    best_dev = dev_loss
                    ckpt = {
                        "model_state_dict": model.state_dict(),
                        "config": dataclasses.asdict(tcfg),
                        "tokenizer_id": tok_ref,
                        "pad_token_id": pad_id,
                        "source_prefix": cfg.source_prefix,
                    }
                    torch.save(ckpt, cfg.output_dir / "best.pt")
                write_history()
                since_eval_loss_sum = 0.0
                since_eval_n = 0
                log_loss_sum = 0.0
                log_n = 0
                t_seg_train0 = time.perf_counter()
                model.train()

        need_epoch_eval = eval_every is None or (global_step % eval_every != 0)
        if need_epoch_eval:
            train_loss_mean = since_eval_loss_sum / max(since_eval_n, 1)
            t_after_train = time.perf_counter()
            train_s = t_after_train - t_epoch0
            model.eval()
            t_ev0 = time.perf_counter()
            dev_loss, metrics = dev_ce_and_metrics()
            eval_s = time.perf_counter() - t_ev0
            epoch_s = time.perf_counter() - t_epoch0

            row = {
                "kind": "eval",
                "global_step": global_step,
                "epoch": epoch + 1,
                "train_loss": train_loss_mean,
                "dev_loss": dev_loss,
                "dev_rougeL_f": metrics["rougeL_f"],
                "dev_token_f1_macro": metrics["token_f1_macro"],
                "n_train": n_train,
                "n_dev": n_dev,
                "train_seconds": round(train_s, 3),
                "dev_ce_and_greedy_seconds": round(eval_s, 3),
                "epoch_seconds": round(epoch_s, 3),
            }
            history.append(row)
            print(json.dumps(row, indent=2))
            print(
                f"[epoch {epoch + 1}/{cfg.epochs}] "
                f"wall {epoch_s:.1f}s (train {train_s:.1f}s · dev CE + greedy gen {eval_s:.1f}s)",
                flush=True,
            )
            if dev_loss < best_dev:
                best_dev = dev_loss
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "config": dataclasses.asdict(tcfg),
                    "tokenizer_id": tok_ref,
                    "pad_token_id": pad_id,
                    "source_prefix": cfg.source_prefix,
                }
                torch.save(ckpt, cfg.output_dir / "best.pt")
            write_history()

    return history
