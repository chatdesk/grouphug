# This is a generic compute_metrics function that will give a range of metrics for classification tasks

import evaluate
import numpy as np

from grouphug import ClassificationHead
from grouphug.config import IGNORE_INDEX

metrics = {k: evaluate.load(k) for k in ["accuracy", "f1", "recall", "precision", "matthews_correlation"]}


def compute_classification_metrics(eval_preds, dataset_name, heads):
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
