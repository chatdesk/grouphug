import pytest
from transformers import AutoTokenizer

from grouphug import AutoMultiTaskModel, LMHeadConfig


@pytest.mark.parametrize(
    "base_model",
    [
        "prajjwal1/bert-tiny",
        "vinai/bertweet-base",
        "google/electra-small-generator",
        "distilbert-base-uncased",
        "microsoft/deberta-v3-small",
    ],
)
def test_model_init_all_expected(base_model):
    model, info = AutoMultiTaskModel.from_pretrained(base_model, [LMHeadConfig()], output_loading_info=True)
    assert info["missing_keys"] == []
    assert info["unexpected_keys"] == []
    assert info["mismatched_keys"] == []
    assert info["error_msgs"] == []


@pytest.mark.parametrize("base_model", ["sentence-transformers/paraphrase-MiniLM-L3-v2"])
def test_model_init_mlm_new(base_model):
    model, info = AutoMultiTaskModel.from_pretrained(base_model, [LMHeadConfig()], output_loading_info=True)
    prefix = model.get_mlm_head().attribute + "."
    assert len(info["missing_keys"]) > 0
    assert all(k.startswith(prefix) for k in info["missing_keys"])
    assert info["unexpected_keys"] == []
    assert info["mismatched_keys"] == []
    assert info["error_msgs"] == []
