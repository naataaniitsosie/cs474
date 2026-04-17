# Dataset Proposal: BriefMe - Legal NLP Benchmark
This project utilizes the BriefMe dataset, a publicly available Legal NLP Benchmark. The data is focused on Supreme Court of the United States (SCOTUS) rulings and is cleanly labeled for various natural language processing tasks.
- Task Definition: Given a section of text from a legal brief, the goal is to summarize the argument into a concise section heading for that part of the brief.
- Model Approach: I plan to use a sequence-to-sequence transformer to perform this argument summarization.

## How will you obtain it?
This data is publicly available on HuggingFace. I can easily and legally obtain the data with the resources I have.

## Any cleaning it requires?
The data has been systematically partitioned and cleaned for use. Apart from unknown implementation-level details, this dataset requires minimal cleaning. The amount of data available has been clearly outlined by the maintainers:

| Subset Name | Train | Dev | Test | Held-Out |
|-------------|-------|-----|------|----------|
| arg_summ    | 18,642 | 2,345 | 2,345 | 164 |

## Reference:
BriefMe: A Legal NLP Benchmark for Assisting with Legal Briefs:
- https://huggingface.co/datasets/jw4202/BriefMe 
- https://arxiv.org/pdf/2506.06619 

## Notice of Final Project Change:
For my final project, I will use a sequence-to-sequence transformer model to summarize Supreme Court of the United States (SCOTUS) rulings. This direction was chosen after initial research into classification problems with transformer technology led to a stronger interest in sequence-to-sequence applications.
