import logging

import pytest
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, TrainingArguments

from grouphug import DatasetFormatter
from grouphug.config import logger

logger.setLevel(logging.INFO)

SMALL_MODEL = "prajjwal1/bert-tiny"


@pytest.fixture
def dataset_action_label():
    ds = Dataset.from_dict({"text": ["abc", "def"], "action_label": [2, 1]})
    return ds


@pytest.fixture
def dataset_multiclass_topics_star():
    ds = Dataset.from_dict(
        {
            "text": ["xyz", "this is another little dataset", "this one has ignores in index"],
            "topics": [[0] * 7, [1, 0, 1, 0, 1, 0, 1], [1, -100, -100, -100, -100, -100, -100]],
            "star": [23, 7, 1],
            "string_label": ["a", "a", "b"],
        }
    )
    return ds


@pytest.fixture
def dataset_regress():
    ds = Dataset.from_dict({"text": [" ".join(f"token{i}" for i in range(100)), "x y z"], "y": [3.14, 42]})
    return ds


@pytest.fixture
def dataset_text_only():
    ds = Dataset.from_dict({"text": [f"the lazy dog jumps over the quick fox {i} times" for i in range(10)]})
    return ds


@pytest.fixture
def dataset_demo_review_star():
    return load_dataset("lhoestq/demo1").rename_column("review", "text")


@pytest.fixture
def multiple_datasets_with_cls(dataset_action_label, dataset_multiclass_topics_star, dataset_demo_review_star):
    return {
        "action": dataset_action_label,
        "topicsstar": dataset_multiclass_topics_star,
        "reviews": dataset_demo_review_star,
    }


@pytest.fixture
def multiple_datasets(multiple_datasets_with_cls, dataset_text_only, dataset_regress):
    return {**multiple_datasets_with_cls, "onlytext": dataset_text_only, "regress": dataset_regress}


@pytest.fixture
def multiple_formatter():
    return DatasetFormatter().tokenize().encode("action_label").encode("star")


@pytest.fixture
def training_args():
    return TrainingArguments(
        output_dir="output/test",
        do_train=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        seed=42,
        logging_steps=1,
        evaluation_strategy="epoch",
    )


@pytest.fixture
def tiny_tokenizer():
    return AutoTokenizer.from_pretrained(SMALL_MODEL)


def losses_not_nan(trainer):
    for r in trainer.state.log_history:
        losses = [v for k, v in r.items() if "loss" in k]
        assert losses, f"Record {r} in state history has missing loss"
        assert all(0 <= l <= 100 for l in losses), f"Record {r} in state history has invalid loss"
