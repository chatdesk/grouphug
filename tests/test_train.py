import os

import numpy as np
import torch

from grouphug.config import IGNORE_INDEX
from grouphug.model import ModelInferenceError

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # TODO: remove

import pandas as pd
import pytest
from datasets import Dataset, load_metric
from transformers import AutoTokenizer, TrainingArguments

from grouphug import (
    AutoMultiTaskModel,
    ClassificationHead,
    ClassificationHeadConfig,
    DatasetFormatter,
    LMHeadConfig,
    MultiTaskTrainer,
)
from grouphug.utils import np_json_dumps
from tests.conftest import SMALL_MODEL, losses_not_nan


def test_train_save_load(multiple_datasets, multiple_formatter, training_args):
    base_model = SMALL_MODEL
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data = multiple_formatter.apply(multiple_datasets, tokenizer=tokenizer, test_size=0)

    head_configs = [
        LMHeadConfig(),
        ClassificationHeadConfig.from_data(
            data, "topics", name="topic_cls", label_smoothing=0.1, pos_weight=[1, 2, 3, 4, 5, 6, 7]
        ),
        ClassificationHeadConfig.from_data(data, "action_label", pooling_type="mean", label_smoothing=0.1),
        ClassificationHeadConfig.from_data(data, "star", weight=2, pooling_type="max"),
        ClassificationHeadConfig.from_data(data, "y"),
    ]
    assert head_configs[1].num_labels == 7
    assert head_configs[2].num_labels == 2
    assert head_configs[4].num_labels == 1

    assert head_configs[1].problem_type == ClassificationHead.MULTI
    assert head_configs[2].problem_type == ClassificationHead.SINGLE
    assert head_configs[3].problem_type == ClassificationHead.SINGLE
    assert head_configs[4].problem_type == ClassificationHead.REGRESSION

    model = AutoMultiTaskModel.from_pretrained(base_model, head_configs, formatter=multiple_formatter)
    training_args.save_steps = 1
    training_args.save_total_limit = 1
    trainer = MultiTaskTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_data=data[:, "train"],
        eval_data=data["topicsstar", "train"],
        eval_heads=["topic_cls"],
    )
    _train_res = trainer.train()
    trainer.save_model()
    loaded_model = AutoMultiTaskModel.from_pretrained(training_args.output_dir)
    assert type(loaded_model) == type(model)
    for lmhc, ohc in zip(loaded_model.head_configs, model.head_configs):
        for k, v in ohc.__dict__.items():
            if k not in ["_head", "auto_attribute"]:  # expected not matching
                eq = lmhc.__dict__[k] == v
                if isinstance(eq, (list, np.ndarray, torch.Tensor)):
                    eq = all(eq)
                assert eq, f"Key {k} for head {ohc} does not match"
    assert loaded_model.formatter is not None
    assert np_json_dumps(loaded_model.formatter.to_dict()) == np_json_dumps(model.formatter.to_dict())  # tuple vs list


# test multiple models, really shows edge cases
@pytest.mark.parametrize("mlm", [False, True])
def test_multi_train(multiple_datasets_with_cls, multiple_formatter, training_args, mlm, subtests):
    fingerprints = []
    for base_model in ["vinai/bertweet-base", "sentence-transformers/paraphrase-MiniLM-L3-v2", SMALL_MODEL]:
        with subtests.test(msg=f"Testing model {base_model} with mlm = {mlm}", base_model=base_model):
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            data = multiple_formatter.apply(multiple_datasets_with_cls, tokenizer=tokenizer, test_size=0)
            fingerprints.append([v._fingerprint for d in data.values() for k, v in d.items()])

            head_configs = [
                ClassificationHeadConfig.from_data(data, "topics", name="topic_cls"),
                ClassificationHeadConfig.from_data(data, "star", weight=2, pooling_type="max"),
                ClassificationHeadConfig.from_data(data, "action_label", name="cls_action", pooling_type="mean"),
            ]
            if mlm:
                head_configs.append(LMHeadConfig())

            model = AutoMultiTaskModel.from_pretrained(base_model, head_configs)
            word_embeddings = model.base_model.embeddings.word_embeddings

            for ds in data.gather_column("input_ids"):
                for input_ids in ds["input_ids"]:
                    assert max(input_ids) < word_embeddings.num_embeddings

            trainer = MultiTaskTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_data=data[:, "train"],
                eval_data=data["action", :],
                eval_heads=["cls_action"],
            )
            _train_res = trainer.train()

    # ensure model and such is taken into account for data formatting
    flattened_fps = [f for fp_row in fingerprints for f in fp_row]
    assert len(flattened_fps) == len(set(flattened_fps))


def test_train_alt_input(training_args):
    base_model = SMALL_MODEL

    ds = Dataset.from_dict(
        {"reply_text": ["the lazy dog jumps over the quick fox", "another sentence"], "labels": [0, 1]}
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    fmt = DatasetFormatter().tokenize("reply_text", output_prefix="reply_")
    data = fmt.apply(ds, tokenizer=tokenizer)
    head_configs = [
        ClassificationHeadConfig.from_data(
            data, labels_var="labels", input_prefix="reply_", pooling_method="max", name="ch"
        ),
    ]
    assert head_configs[0].num_labels == 2
    assert head_configs[0].input_prefixes() == ["reply_"]
    model = AutoMultiTaskModel.from_pretrained(base_model, head_configs=head_configs, formatter=fmt)
    trainer = MultiTaskTrainer(
        model=model,
        args=training_args,
        train_data=data[:, "train"],
        eval_data=data["data", "train"],
    )

    expected_vars = {
        "reply_input_ids",
        "reply_token_type_ids",
        "reply_attention_mask",
        "reply_special_tokens_mask",
        "labels",
    }
    assert set(data["data"]["train"].features.keys()) == expected_vars
    assert trainer.data_collator.model_vars == expected_vars
    trainer.train()
    losses_not_nan(trainer)

    # test infer methods
    result = model.format_forward(reply_text="abc")
    assert len(result.head_outputs) == 1
    assert len(result.head_outputs["ch"].logits[0]) == 2

    assert head_configs[0].output_stats(result.head_outputs["ch"]).keys() == {
        "probs",
        "predicted_id",
    }

    result = model.format_forward(reply_text="abc", labels=0)
    assert result.head_outputs["ch"].loss is not None
    assert head_configs[0].output_stats(result.head_outputs["ch"]).keys() == {
        "loss",
        "probs",
        "predicted_id",
    }

    assert isinstance(model.predict({"reply_text": "abc", "labels": 0}), dict)
    assert isinstance(model.predict([{"reply_text": "abc"}]), list)
    df = pd.DataFrame([{"reply_text": "abc"}])
    dataset = Dataset.from_pandas(df)
    assert isinstance(model.predict(df), pd.DataFrame)
    assert isinstance(model.predict(dataset), list)


def test_train_eval_metrics(multiple_datasets, multiple_formatter):
    base_model = SMALL_MODEL
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data = multiple_formatter.apply(multiple_datasets, tokenizer=tokenizer, test_size=0)

    head_configs = [
        ClassificationHeadConfig.from_data(data, "topics", name="topic_cls", id2label=[str(i) for i in range(7)]),
        ClassificationHeadConfig.from_data(data, "star", weight=2, pooling_type="max"),
        ClassificationHeadConfig.from_data(data, "action_label", pooling_type="mean"),
        LMHeadConfig(),
    ]
    model = AutoMultiTaskModel.from_pretrained(base_model, head_configs, formatter=multiple_formatter)
    training_args = TrainingArguments(
        output_dir="output/test",
        do_train=True,
        max_steps=5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        seed=42,
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=2,
    )

    def compute_metrics(eval_preds, dataset_name, heads):
        all_logits, all_labels = eval_preds
        if not isinstance(all_logits, tuple):
            all_logits = (all_logits,)
            all_labels = (all_labels,)
        metrics = {}
        accuracy_f = load_metric("accuracy")
        for logits, labels, hc in zip(all_logits, all_labels, heads):
            labels = labels.ravel()
            mask = labels != IGNORE_INDEX
            if getattr(hc, "problem_type", None) == ClassificationHead.MULTI:
                predictions = (logits > 0).ravel()[mask]
            else:
                predictions = np.argmax(logits, axis=-1).ravel()[mask]
            acc = accuracy_f.compute(predictions=predictions, references=labels[mask])
            metrics[f"{hc.name}_accuracy"] = acc["accuracy"]
        return metrics

    trainer = MultiTaskTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_data=data[:, "train"],
        eval_data=data[["topicsstar", "action"], "train"],
        eval_heads=["lm", "star", "action_label"],
        compute_metrics=compute_metrics,
    )
    with pytest.raises(ValueError):  # action does not have star labels
        _train_res = trainer.train()
    trainer.eval_heads = {"topicsstar": ["star", "topic_cls", "mlm"], "action": ["action_label"]}
    _train_res = trainer.train()
    losses_not_nan(trainer)
    all_keys = {k for l in trainer.state.log_history for k in l.keys()}
    assert "eval_action_action_label_accuracy" in all_keys

    predictions = model.predict(dict(text="abc"))
    assert "star_predicted_label" in predictions.keys()
    assert "action_label_predicted_label" in predictions.keys()
    assert "topic_cls_predicted_labels" in predictions.keys()


def test_train_mlm_mtd(dataset_regress, training_args):
    base_model = SMALL_MODEL
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    fmt = DatasetFormatter().tokenize()
    data = fmt.apply(dataset_regress, tokenizer=tokenizer, test_size=0)
    training_args.evaluation_strategy = None

    for sep in [True, False]:
        for mlm, mtd in [(True, False), (False, True), (True, True)]:
            head_configs = [
                LMHeadConfig(
                    masked_language_modelling=mlm,
                    masked_token_detection=mtd,
                    mtd_pos_weight=2.0,
                    separate_embedding=sep,
                ),
                ClassificationHeadConfig.from_data(data, labels_var="y"),
            ]
            model = AutoMultiTaskModel.from_pretrained(base_model, head_configs, tokenizer=tokenizer, formatter=fmt)
            trainer = MultiTaskTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_data=data[:, "train"],
            )
            trainer.train()
            result = model.predict(dict(text="blabla"))
            assert result.keys() == {"y_predicted_value"}


@pytest.mark.parametrize("base_model", [SMALL_MODEL, "facebook/opt-125m"])
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_train_clm(dataset_regress, training_args, base_model, padding_side):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = padding_side
    fmt = DatasetFormatter().tokenize()
    data = fmt.apply(dataset_regress, tokenizer=tokenizer, test_size=0)
    training_args.evaluation_strategy = None

    head_configs = [
        LMHeadConfig(causal_language_modelling=True),
        ClassificationHeadConfig.from_data(data, labels_var="y", pooling_method="last", classifier_hidden_size=3),
        ClassificationHeadConfig.from_data(data, name="yauto", labels_var="y", classifier_hidden_size=3),
    ]
    model = AutoMultiTaskModel.from_pretrained(base_model, head_configs, tokenizer=tokenizer, formatter=fmt)
    trainer = MultiTaskTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_data=data[:, "train"],
    )
    trainer.train()
    result = model.predict(dict(text="blabla"))
    assert result.keys() == {"y_predicted_value", "yauto_predicted_value"}


def test_train_forgot_encode(dataset_multiclass_topics_star, training_args):
    tokenizer = AutoTokenizer.from_pretrained(SMALL_MODEL)
    fmt = DatasetFormatter().tokenize()  # .encode("string_label")
    data = fmt.apply(dataset_multiclass_topics_star, tokenizer=tokenizer, test_size=0)

    head_configs = [
        ClassificationHeadConfig.from_data(data, labels_var="string_label", classifier_hidden_size=1),
    ]
    model = AutoMultiTaskModel.from_pretrained(SMALL_MODEL, head_configs, tokenizer=tokenizer, formatter=fmt)
    training_args.evaluation_strategy = None
    trainer = MultiTaskTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_data=data[:, "train"],
    )
    with pytest.raises(ModelInferenceError):
        trainer.train()


def test_train_mtd_harder(dataset_regress, training_args):
    base_model = SMALL_MODEL
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    fmt = DatasetFormatter().tokenize()
    data = fmt.apply(dataset_regress, tokenizer=tokenizer, test_size=0)
    training_args.evaluation_strategy = None

    head_configs = [
        LMHeadConfig(
            masked_language_modelling=False,
            masked_token_detection=True,
            mtd_pos_weight=2.0,
        ),
        ClassificationHeadConfig.from_data(data, labels_var="y"),
    ]
    model = AutoMultiTaskModel.from_pretrained(base_model, head_configs, tokenizer=tokenizer, formatter=fmt)
    trainer = MultiTaskTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_data=data[:, "train"],
    )
    trainer.train()
    result = model.predict(dict(text="blabla"))
    assert result.keys() == {"y_predicted_value"}
