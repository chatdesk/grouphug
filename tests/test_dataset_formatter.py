import pytest
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from grouphug import DatasetCollection, DatasetFormatter
from grouphug.dataset_formatter import remove_handles, replace_handles, replace_urls, tolower, truncate_repeat
from tests.conftest import SMALL_MODEL


def test_formatter(multiple_datasets):
    ds = Dataset.from_dict(
        {
            "text": ["the lazy dog jumps over the quick fox", "another sentence", "a third sentence"],
            "labels": [2, 1, -100],
            "multi_label": [[1, 2, 3], [4, 5, 1], []],
            "ignore_entry": [[4, 5, 1], [], [1, 2, 3, 4]],
        }
    )
    data = (
        DatasetFormatter().binarize("multi_label", ignore_column="ignore_entry").encode("labels").apply(ds, test_size=0)
    )
    assert "data" in data
    dt = data["data"]["train"]
    assert dt["labels"] == [1, 0, -100]
    assert dt["multi_label"] == [[1, 1, 1, -100, -100], [1, 0, 0, 1, 1], [-100, -100, -100, -100, 0]]

    data = DatasetFormatter().binarize("multi_label").binarize("labels").apply(ds, test_size=0)
    dt2 = data["data"]["train"]

    assert dt["labels"] == [1, 0, -100]  # shallow copy working
    assert dt2["labels"] == [[0, 1], [1, 0], [0, 0]]  # one-hot encoding working


def test_truncate_repeat():
    assert truncate_repeat("yay!!!!! aaaawesome!") == "yay!!! aaawesome!"
    assert truncate_repeat("That's hilarious! 😂😂😂😂😂😂😂😂😂😂😂😂") == "That's hilarious! 😂😂😂"


def test_lower():
    assert tolower("AbC ABC") == "abc abc"


def test_handles():
    assert remove_handles("@user is this ok @me") == "  is this ok  "
    assert replace_handles("@user is this ok @me") == "@USER is this ok @USER"


def test_replace_urls():
    assert replace_urls("Go to www.blah.com/asdasdasdasd?b=c or to https://a.com") == "Go to  URL  or to  URL "
    assert replace_urls("Go to www.blah.com/asdasdasdasd?b=c or to https://a.com") == "Go to  URL  or to  URL "


def test_preprocess_formatter():
    ds = Dataset.from_dict(
        {
            "text": ["한국어....<br><p> That`s&nbsp;amazing!!!!!</p> 😂😂😂😂😂😂😂😂😂😂😂😂......"],
        }
    )

    data = (
        DatasetFormatter()
        .preprocess(
            [
                "normalize",
                "replace_handles",
                "truncate_repeat",
                "demojize",
                "unidecode",
                "normalize_spaces",
            ]
        )
        .apply(ds, test_size=0)
    )
    assert data["data"]["train"]["text"] == [
        "hangugeo... That's amazing!!! :|facewithtearsofjoy :|facewithtearsofjoy :|facewithtearsofjoy ..."
    ]


def test_tokenize_pairs(dataset_text_only):
    ds = Dataset.from_dict(
        {
            "sentence1": ["abc"],
            "sentence2": ["def"],
        }
    )
    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    data = (
        DatasetFormatter()
        .tokenize()
        .tokenize(("sentence1", "sentence2"))
        .apply({"text": dataset_text_only, "ds": ds}, tokenizer=tokenizer, test_size=0)
    )
    assert "input_ids" in data["ds"]["train"][0].keys()


def test_single_dataset():
    text_data = load_dataset("text", data_files=__file__)["train"]
    assert isinstance(text_data, Dataset)
    tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL)
    data = DatasetFormatter().tokenize().apply(text_data, tokenizer=tokenizer, test_size=0.1)
    assert "data" in data
    assert "input_ids" in data["data"]["train"].features
    assert "input_ids" in data["data"]["test"].features
