## Clearly define goal
The goal of my final project is to train a deep neural network to classify legal case text as either “violation” or “no violation.” This will be framed as a supervised binary classification problem using curated legal datasets. The focus of the project will be on implementing and training the model from scratch in PyTorch.

## Type of deep learning approach
The two deep learning approaches I am considering are an LSTM-based model or a small Transformer. We have not yet completed our unit on the Transformer architecture, so I will finalize that decision as we progress through the course. My current plan is to implement one of these models from scratch in PyTorch. If time permits, I may implement both models.

## How you will measure success
Success will be measured primarily using test accuracy. In addition, I will report cross-entropy loss and F1 score to account for potential class imbalance. If time permits, I will compare model performance across architectures and analyze misclassified examples.

## Datasets you will use
To minimize data engineering and focus on modeling, I will use a curated and labeled legal dataset. The three datasets under consideration are:
- ECHR (European Court of Human Rights): https://huggingface.co/datasets/glnmario/ECHR
  - Supporting paper: https://arxiv.org/abs/1906.02059
- CaseHOLD (U.S. legal holding prediction benchmark): https://huggingface.co/datasets/casehold/casehold
  - Supporting paper: https://reglab.stanford.edu/data/casehold-benchmark/
- CUAD (Contract Understanding Atticus Dataset): https://huggingface.co/datasets/theatticusproject/cuad
  - Supporting paper: https://arxiv.org/abs/2103.06268

## What measures you will use (acc, test loss, etc)
Evaluation will include accuracy, cross-entropy loss, and F1 score. See the “How you will measure success” section.


## Available computation resources (GPUs)
I have over 120 credits on Google Colab. Thus, H100 and A100 GPUs will be used for full training runs. Further, I have an Apple M4 Max with 36 GB of unified memory. I’ll be prototyping locally since configuring PyTorch on macOS to use GPUs isn’t difficult.
