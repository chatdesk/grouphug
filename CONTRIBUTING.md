# Feature requests and PRs

If you want to add a complex feature or change which is not already mentioned, please open an issue or discussion topic to discuss. 
For simple additions and bug fixes you can open a PR directly.

## Formatting and pre-commit hooks

To ensure your PR is properly formatted, install pre-commit hooks using `pre-commit install`

This will run black, isort, and clear any output from example notebooks when committing.

# Notes on Grouphug internals

This section contains notes on implementation details of huggingface transformers and grouphug.

## Computing metrics

Computing metrics has been changed to be passed extra parameters, allowing the metrics function to know what data is passed.

The function below works as a fairly generic implementation and could be added as a default in future versions.

```python
from grouphug import ClassificationHead
from grouphug.config import IGNORE_INDEX
import numpy as np
from datasets import load_metric

metrics = {k: load_metric(k) for k in ["accuracy", "f1", "recall", "precision", "matthews_correlation"]}
def compute_metrics(eval_preds, dataset_name, heads):
    all_logits, all_labels = eval_preds
    if not isinstance(all_logits, tuple):
        all_logits = (all_logits,)
        all_labels = (all_labels,)
    results = {}

    for logits, labels, hc in zip(all_logits, all_labels, heads):
        labels_1d = labels.ravel()
        mask = labels_1d != hc.ignore_index
        labels_1d = labels_1d[mask]
        if hc.problem_type == ClassificationHead.MULTI:
            predictions = logits > 0
            predictions_1d = predictions.ravel()[mask]
            exact_match = ((predictions == labels) | (labels == IGNORE_INDEX)).all(axis=-1)
            # entire prediction is correct
            results[f"{hc.name}_subset_accuracy"] = exact_match.sum() / len(exact_match)
        else:
            predictions_1d = np.argmax(logits, axis=-1).ravel()[mask]
        for k, f in metrics.items():
            try:
                kwargs = {"average": "weighted"} if k in ["f1", "recall", "precision"] else {}
                for mk, mv in f.compute(predictions=predictions_1d, references=labels_1d, **kwargs).items():
                    results[f"{hc.name}_{mk}"] = mv
            except Exception as e:
                print(f"metric {k} on dataset {dataset_name} head {hc.name} failed: {e}")
    return results
```


# Notes on Huggingface Transformers internals

These are largely my own notes on the internals of the transformers package and how they interact.


## Tokenizers

* Tokenizers have `.model_input_names` to determined what to pad, e.g. `['input_ids, 'token_type_ids','attention_mask']`
  * However, these are mostly ignored except for the first, and `_pad` has a hardcoded check for `['input_ids, 'token_type_ids','attention_mask','special_tokens_mask']`
* Tokenizers have model_max_len, which is often unset and left at it's default of LARGE_INTEGER
* Dynamic padding is done by various collators via Tokenizer.pad, but this does not truncate.

## Trainer

* Model outputs are ordered dicts
* All keys not named 'loss' are assumed to be logits
  * Somehow one of the GPT models still returns two losses. 
