#!/usr/bin/env python3
"""Finetune T5-small on BriefMe ``arg_summ`` (Phase 5 baseline; reference only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from briefme.seq2seq_data import DEFAULT_T5_TASK_PREFIX, default_train_dev_materialize, get_t5_tokenizer, parse_cli_train_limit
from briefme.train_t5_loop import T5BaselineTrainConfig, run_t5_baseline_training, t5_dataloaders_from_examples
from transformer.config import HF_T5_BASELINE_MODEL_ID
from transformers import T5ForConditionalGeneration


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default=HF_T5_BASELINE_MODEL_ID)
    p.add_argument(
        "--train-limit",
        type=parse_cli_train_limit,
        default=4096,
        help="Max train rows (int), or none/full/all for entire Hub train split",
    )
    p.add_argument("--dev-limit", type=int, default=512)
    p.add_argument("--output-dir", type=Path, default=Path("runs/t5_baseline"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-src-len", type=int, default=512)
    p.add_argument("--max-tgt-len", type=int, default=128)
    p.add_argument("--source-prefix", default=DEFAULT_T5_TASK_PREFIX)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        metavar="N",
        help="Log train loss every N steps (Trainer)",
    )
    p.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        metavar="N",
        help="Run eval (generate + metrics) every N steps; omit for once-per-epoch eval",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = T5BaselineTrainConfig(
        model_id=args.model_id,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        source_prefix=args.source_prefix,
        seed=args.seed,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
    )
    print("[briefme] loading tokenizer and T5 weights...", flush=True)
    tokenizer = get_t5_tokenizer(cfg.model_id)
    model = T5ForConditionalGeneration.from_pretrained(cfg.model_id)
    print("[briefme] materializing train/dev examples (streaming from Hub)...", flush=True)
    train_ex, dev_ex = default_train_dev_materialize(args.train_limit, args.dev_limit)
    train_loader, eval_loader, collator = t5_dataloaders_from_examples(
        train_ex,
        dev_ex,
        tokenizer,
        model,
        cfg,
        batch_size=args.batch_size,
    )
    run_t5_baseline_training(
        cfg,
        tokenizer=tokenizer,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        data_collator=collator,
    )


if __name__ == "__main__":
    main()
