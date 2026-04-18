"""BriefMe dataset constants and seq2seq mapping for CS474 final project."""

from briefme.data import (
    SPLIT_DEV,
    SPLIT_TEST,
    SPLIT_TRAIN,
    iter_rows,
    load_arg_summ_split_streaming,
    load_arg_summ_train_dev_test_streaming,
    materialize_head,
)
from briefme.metrics import aggregate, normalize, score_pair, token_f1_one
from briefme.schema import (
    CONFIG_NAME,
    DATASET_NAME,
    SOURCE_COLUMN,
    TARGET_COLUMN,
    to_seq2seq_example,
)
from briefme.transformer import HF_T5_BASELINE_MODEL_ID, ScratchTransformerConfig

__all__ = [
    "CONFIG_NAME",
    "DATASET_NAME",
    "SPLIT_DEV",
    "SPLIT_TEST",
    "SPLIT_TRAIN",
    "SOURCE_COLUMN",
    "TARGET_COLUMN",
    "iter_rows",
    "load_arg_summ_split_streaming",
    "load_arg_summ_train_dev_test_streaming",
    "materialize_head",
    "to_seq2seq_example",
    "aggregate",
    "normalize",
    "score_pair",
    "token_f1_one",
    "HF_T5_BASELINE_MODEL_ID",
    "ScratchTransformerConfig",
]
