# North Star: CS474 Final Project (Seq2seq Legal Brief → Heading)

This document is the **single source of truth** for scope and sequencing. All choices below are constrained by `planning/final_project.md` (grading, report, time log, and the training requirement).

---

## 1. Non-negotiable constraints (from the course)

| Requirement | What it means for us |
|-------------|----------------------|
| **Deliverable** | PDF writeup **≤ 3 pages** total: **≤ 2 pages** body (problem, data, approach, results) + **≤ 1 page** daily time log. |
| **Hours grade** | Score uses **total hours ÷ 30** (cap at 30; no extra credit beyond 30). |
| **Time log** | **Daily** entries with hours + short activity description. **Undocumented time does not count.** |
| **Reading cap** | **≤ 5 hours** labeled “research and reading.” |
| **Prep cap** | **≤ 10 hours** for dataset prep, cleaning, environment, “getting things running,” etc. |
| **Modeling floor** | **≥ 20 hours** on designing, building, debugging, testing DL models, analyzing results, experimenting. **GPU training time does not count.** |
| **Core requirement** | The project must include **training or finetuning** a model. **Inference-only** (e.g., pretrained HF for predict only, or OpenAI/Gemini as the “solution”) **does not satisfy** the requirement. |
| **Grading emphasis** | **Effort and clarity** matter; ambitious attempts are OK if the report and time log show scope. |

**Implication:** Any **LLM API** use (e.g., for evaluation or spot-checking drafts) must be clearly **ancillary**—the **graded technical artifact** is still **your trained model** and its analysis. The report should state that plainly so the instructor never confuses “we used GPT for scoring” with “our project was calling an API instead of training.”

---

## 2. North star (one sentence)

**Build and train an encoder–decoder sequence-to-sequence model** that maps a passage from a legal brief to a **concise section-style heading**, using the **BriefMe** benchmark (e.g., `arg_summ` split), and **evaluate it with a defensible, layered protocol** (automated metrics aligned to the task, human spot checks, and LLM-assisted judgment)—then **implement a transformer stack from scratch in PyTorch** (using Lab 8 self-attention as the foundation, extended with **cross-attention** for the decoder) and connect that implementation to the same evaluation so results are comparable.

---

## 3. Working backwards: evaluation first (why this order)

Your research instinct—**design the eval before you trust the model**—is appropriate here. For this course, it also protects you from “pretty loss curves, meaningless outputs.”

**Suggested layers (each must justify its role; no metric theater):**

### A. Automated metrics (meaningful, not default BLEU)

- **What we are predicting:** Short **English headings** for argumentative sections (not machine translation into another language). BLEU is often a **weak default** here: n-gram overlap can be unforgiving when multiple valid headings exist.
- **Prefer (task-aligned defaults to discuss and pick from):**
  - **ROUGE** (especially **ROUGE-L**): correlates reasonably with **overlap** between predicted and reference headings when both are short strings.
  - **Token-level F1** (after consistent tokenization/lowercasing): interpretable for **lexical overlap** on short targets.
  - **Exact match (EM)** if the label space is relatively constrained; report it **alongside** softer metrics so you do not over-penalize paraphrases.
  - **chrF** (character n-gram F-score): sometimes more stable than pure token n-grams for **minor morphological/stylistic** differences.

**Rule:** In the report, add **one paragraph per metric** explaining **what mistake it catches** and **what it systematically misses** (e.g., synonymy, reordering). If a metric does not clarify success for this task, **do not include it** just to fill space.

### B. Human spot checks (small but real)

- Sample **N** predictions (stratify by difficulty if possible: high vs low ROUGE, long vs short inputs).
- Use a **short rubric** (e.g., factual consistency with passage, usefulness as a section title, grammar). **Inter-rater** is ideal if you have a second reader; if not, be transparent that scores are **single-rater** with clear criteria.

### C. LLM-assisted evaluation (optional; prompt-heavy)

- Useful for **pairwise preference** (“which heading is better?”) or **Likert** quality scores with a fixed rubric.
- **Risks:** bias toward fluent boilerplate, sensitivity to prompt, cost/latency. Mitigate with **frozen prompts** (documented in appendix or repo), **few-shot** with gold examples from **train only**, and **spot audits** against human judgment.

**Course alignment:** Count LLM-eval **implementation and analysis** toward your **≥ 20 hours modeling/analysis** bucket if you are iterating prompts, auditing failure modes, and writing up limitations—not toward “reading.”

---

## 4. Then: transformer from scratch (technical spine)

**Requirement:** “From scratch” here means **you implement the core transformer building blocks and training loop** in PyTorch, grounded in Lab 8 (self-attention, masking, layer structure). Pretrained weights are **optional**; if used, document **where they came from** (per course checklist).

**Architecture choice (locked):** **Encoder–decoder transformer** (classic seq2seq). The encoder ingests the full legal passage; the decoder generates the heading with **cross-attention** to encoder states (plus causal self-attention on generated tokens). This matches supervised (source, target) pairs on BriefMe better than a decoder-only LM for this benchmark-first project.

**Bridge from Lab 8:** Lab 8 stresses **self-attention** (encoder-style). Extend it with **decoder layers** and **cross-attention** so the decoder is conditioned on the encoded passage. Keep the evaluation **identical** across experiments where possible.

---

## 5. PyTorch vs Hugging Face: how they fit (not either/or)

| Layer | PyTorch | Hugging Face (`transformers`, `datasets`, `tokenizers`) |
|-------|---------|---------------------------------------------------------|
| **Tensors, autograd, `nn.Module`** | Native home for **custom** attention, masks, loss, optimization | Models are **PyTorch modules**; you can subclass or copy patterns |
| **Training loop** | You write it; full control | `Trainer` optional; still PyTorch under the hood |
| **Tokenizer & data streaming** | DIY or use `tokenizers` / HF datasets as **libraries** | **Datasets** + **tokenizers** save a lot of prep time; counts toward **prep** until you are doing modeling analysis |
| **“Low level”** | **You** define attention, dimensions, init, schedules | **Low level** if you **read/run module code**; **high level** if you only call `generate()` on a canned config |
| **Satisfying “train a model”** | Train **your** `nn.Module` | Finetune **their** `nn.Module`—still training, but less “from scratch” |

**Practical split that matches your goals:**

1. **PyTorch + Lab 8:** implement **attention blocks, stacks, forward, loss** (the course story: you *built* the transformer).
2. **Hugging Face as tooling only (recommended):** `datasets` to load BriefMe, `tokenizers` (or HF tokenizer) for consistent subword segmentation, optional **baselines** (e.g., small finetuned seq2seq) **if time permits**—clearly labeled as baseline, not a substitute for your scratch model.
3. **Avoid:** Presenting a **pretrained HF model used only for inference** as the project outcome—that violates the stated rule.

---

## 6. Suggested phased roadmap (for time-bucket discipline)

Rough order; adjust dates as needed. **Keep a running log** so hours land in the right buckets.

1. **Lock task + label schema** (BriefMe `arg_summ`; confirm input/output fields from the dataset card).
2. **Evaluation protocol draft** (metrics + human rubric + optional LLM prompt doc)—**before** large training sweeps.
3. **Data inspection** (EDA visuals, length stats, leakage checks)—stay within **prep** where appropriate.
4. **Minimal training pipeline** on a **small** slice to validate end-to-end (encode/decode, decoding strategy for generation at eval).
5. **Scratch encoder–decoder transformer** implementation (Lab 8 → cross-attention + full model), then **full-dataset** runs and ablations (optimizer, depth, regularization).
6. **Writeup**: problem, EDA, method (params, optimizer, split), results (automated + human + optional LLM), overfitting discussion, what you would iterate next.

---

## 7. What “success” looks like for the rubric (not only leaderboard numbers)

- **Time log** shows **≥ 20 hours** of real modeling/experimentation work and reads like a serious project.
- **Report** answers the course’s checklist (dataset provenance, supervised vs unsupervised, splits, parameter count, optimizer, pretrained weights or not, metrics, overfitting discussion).
- **Evaluation** is **credible**: metrics justified, humans (even small-N) grounding trust, LLM eval optional and caveated.
- **Training** is unmistakably **your** project’s center; APIs are supporting actors.

---

## 8. Open decisions to resolve next (before heavy coding)

- [ ] Exact BriefMe subset and column mapping (input text vs target heading).
- [x] **Encoder–decoder** (not decoder-only)—locked; rationale in §4.
- [ ] Final metric suite (2–4 automated metrics max, each with a stated purpose).
- [ ] Human evaluation **N** and rubric (feasible given deadline).
- [ ] Whether a small **HF finetuned baseline** is in scope or a distraction from scratch work.

---

## References (existing repo docs)

- Course spec: `planning/final_project.md`
- Dataset direction: `planning/dataset_proposal.md`
- Earlier classification idea (superseded for scope unless revived): `planning/project_proposal.md`
