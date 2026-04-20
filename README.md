# CS474 — BriefMe seq2seq (final project)

Encoder–decoder training for legal brief passages → section headings using the [BriefMe](https://huggingface.co/datasets/jw4202/BriefMe) `arg_summ` task. See [`planning/north_star_plan.md`](planning/north_star_plan.md) for scope and grading constraints.

## Project layout

| Path | Purpose |
|------|---------|
| [`src/briefme/`](src/briefme/) | Dataset + metrics + judge helpers (`text` → `reference`). |
| [`src/transformer/`](src/transformer/) | Scratch seq2seq: Phases 1–2 (`config`, `attention`, `masking`, `layers`); Phase 3 **`positional.py`** (sinusoidal PE), **`model.py`** (`ScratchSeq2SeqTransformer`, teacher-forcing CE helper). Presets: **`ScratchTransformerConfig.tiny()`** (smoke), **`.small()`** (default for real runs), **`.medium()`** (larger optional ablation). Not Hugging Face **`transformers`**. Heavy symbols load lazily from `transformer` so `ScratchTransformerConfig` works without importing `torch`. |
| [`src/briefme/seq2seq_data.py`](src/briefme/seq2seq_data.py) | **Phase 4:** T5 tokenizer, `BriefMeSeq2SeqDataset`, `collate_seq2seq_batch`, teacher-forcing `tgt_in` / `labels` alignment. |
| [`src/briefme/generation.py`](src/briefme/generation.py) | **Phase 5:** `greedy_generate` and `beam_generate` for the scratch model + string decode helpers. |
| [`src/briefme/train_scratch_loop.py`](src/briefme/train_scratch_loop.py) | Scratch training loop (`ScratchTrainConfig`, `run_scratch_training`) — used by CLI and **`notebooks/04_train_scratch_seq2seq.ipynb`**. |
| [`src/briefme/train_t5_loop.py`](src/briefme/train_t5_loop.py) | T5 baseline loop (`T5BaselineTrainConfig`, `run_t5_baseline_training`) — used by CLI and **`notebooks/05_train_t5_baseline.ipynb`**. |
| [`scripts/`](scripts/) | **`train_scratch_seq2seq.py`**; **`train_t5_baseline.py`**. Run with `PYTHONPATH=src` from repo root or after `pip install -e .`. |
| [`notebooks/`](notebooks/) | EDA (`01`), scratch train (`04`), T5 baseline (`05`), inference runs (`06`), **metrics browser (`07`)**, **LLM judge (`08`)**. |
| [`planning/`](planning/) | Course specs and planning docs. |

**Why `src/briefme` and not `src/dataset/briefme`?**  
Putting the package directly under `src/` is the usual pattern when you have one main library to install. An extra `dataset/` folder only pays off if you split multiple installable packages (for example `src/models`, `src/evaluation`). Here a single package `briefme` keeps imports short (`import briefme`) and tooling simple.

## Environment (Conda)

From the repository root (the folder that contains `pyproject.toml`):

```bash
conda env create -f environment.yml
conda activate cs474
pip install -e ".[dev]"
```

This installs runtime + dev deps from [`pyproject.toml`](pyproject.toml) and installs this repo in **editable** mode so `import briefme` works everywhere (including Jupyter).

Register the kernel (optional):

```bash
python -m ipykernel install --user --name cs474 --display-name "Python (cs474)"
```

Then choose **Python (cs474)** when opening notebooks.

### Hugging Face token

1. Copy `.env.example` to `.env`.
2. Set `HUGGINGFACE_HUB_TOKEN` ([create a token](https://huggingface.co/settings/tokens)).

Without a token, some public Hub assets still download; gated datasets require the token.

### Pip-only alternative

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

[`requirements.txt`](requirements.txt) mirrors core deps if you prefer a flat list instead of Conda.

### Training / compute

PyTorch training (scratch transformer + T5 baseline) uses **`torch`**, **`transformers`**, and **`accelerate`** (see [`pyproject.toml`](pyproject.toml)). Use **CUDA** on Linux/Colab when possible; on **Apple Silicon**, PyTorch **`mps`** may accelerate parts of the stack (some ops fall back to CPU).

Where to run long jobs (local M4 vs Colab T4 vs A100, etc.) is documented in **[`planning/north_star_plan.md`](planning/north_star_plan.md) §6 — Compute strategy**. In your report, note **device**, **approximate GPU**, **batch size**, and **max sequence lengths** for reproducibility.

### Running tests

Install dev extras (`pip install -e ".[dev]"`) so **pytest** is available. From the repo root:

```bash
python -m pytest
```

Tests live under [`tests/`](tests/). Options are configured in [`pyproject.toml`](pyproject.toml) (`[tool.pytest.ini_options]` → `testpaths = ["tests"]`). For verbose output: `python -m pytest -v`; one file: `python -m pytest tests/test_metrics.py`.

### Training (Phases 4–5)

From the repo root with the env activated and Hugging Face access for BriefMe streaming:

```bash
PYTHONPATH=src python scripts/train_scratch_seq2seq.py --epochs 3 --train-limit 2048 --preset tiny --output-dir runs/scratch_run
PYTHONPATH=src python scripts/train_t5_baseline.py --epochs 2 --train-limit 4096 --output-dir runs/t5_run
```

Use `--train-limit none` (or `full` / `all`) to stream the full train split (longer). Checkpoints: scratch `best.pt` under `--output-dir`; T5 uses Hugging Face `Seq2SeqTrainer` saves under `--output-dir`.

Notebook entry points (same training code as the scripts): [`notebooks/04_train_scratch_seq2seq.ipynb`](notebooks/04_train_scratch_seq2seq.ipynb), [`notebooks/05_train_t5_baseline.ipynb`](notebooks/05_train_t5_baseline.ipynb).

## BriefMe loading note

Non-streaming `datasets.load_dataset(..., streaming=False)` can fail while building cache (upstream schema issue on the `held_out` split). Use **`briefme.data`** helpers (`streaming=True`) as in the EDA notebook.
