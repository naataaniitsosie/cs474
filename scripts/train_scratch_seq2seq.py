#!/usr/bin/env python3
"""Train the scratch encoder-decoder on BriefMe ``arg_summ`` (Phase 5)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on PYTHONPATH when run as ``python scripts/train_scratch_seq2seq.py``
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from briefme.seq2seq_data import DEFAULT_T5_TASK_PREFIX, default_train_dev_materialize, get_t5_tokenizer, parse_cli_train_limit
from briefme.train_scratch_loop import ScratchTrainConfig, run_scratch_training, scratch_dataloaders_from_examples
from transformer.config import HF_T5_BASELINE_MODEL_ID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preset",
        choices=("tiny", "small", "medium"),
        default="tiny",
        help="ScratchTransformerConfig preset (tiny|small|medium)",
    )
    p.add_argument("--tokenizer-id", default=None, help="HF tokenizer id (default: T5-small baseline id)")
    p.add_argument(
        "--train-limit",
        type=parse_cli_train_limit,
        default=2048,
        help="Max train rows (int), or none/full/all for entire Hub train split",
    )
    p.add_argument("--dev-limit", type=int, default=512, help="Dev materialization cap")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--output-dir", type=Path, default=Path("runs/scratch_seq2seq"))
    p.add_argument("--source-prefix", default=DEFAULT_T5_TASK_PREFIX, help="Prepended to encoder text (T5-style)")
    p.add_argument("--max-new-tokens-eval", type=int, default=128, help="Greedy decode cap for dev metrics")
    p.add_argument(
        "--logging-steps",
        type=int,
        default=None,
        metavar="N",
        help="Append train-only loss rows every N optimizer steps (omit for epoch summaries only)",
    )
    p.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        metavar="N",
        help="Run dev CE + greedy metrics every N steps (omit for once-per-epoch eval)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tok_id = args.tokenizer_id or HF_T5_BASELINE_MODEL_ID
    print("[briefme] loading tokenizer...", flush=True)
    tokenizer = get_t5_tokenizer(tok_id)
    print("[briefme] materializing train/dev examples (streaming from Hub)...", flush=True)
    train_ex, dev_ex = default_train_dev_materialize(args.train_limit, args.dev_limit)
    train_loader, dev_loader = scratch_dataloaders_from_examples(
        train_ex,
        dev_ex,
        tokenizer,
        preset=args.preset,
        source_prefix=args.source_prefix,
        batch_size=args.batch_size,
    )
    cfg = ScratchTrainConfig(
        preset=args.preset,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        device=args.device,
        output_dir=args.output_dir,
        source_prefix=args.source_prefix,
        max_new_tokens_eval=args.max_new_tokens_eval,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
    )
    run_scratch_training(cfg, tokenizer=tokenizer, train_loader=train_loader, dev_loader=dev_loader)


if __name__ == "__main__":
    main()
