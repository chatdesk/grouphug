
# grouphug

GroupHug is a library with extensions to 🤗 transformers for multitask language modelling.
In addition, it contains utilities that ease data preparation, training, and inference.

## Overview

The package is optimized for training a single language model to make quick and robust predictions for a wide variety of related tasks at once,
 as well as to investigate the regularizing effect of training a language modelling task at the same time.

You can train on multiple datasets, with each dataset containing an arbitrary subset of your tasks. Supported tasks include: 

* A single language modelling task (Masked language modelling, Masked token detection, Causal language modelling).
  * The default collator included handles most preprocessing for these heads automatically.
* Any number of classification tasks, including single- and multi-label classification and regression
  * A utility function that automatically creates a classification head from your data. 
  * Additional options such as hidden layer size, additional input variables, and class weights.
* You can also define your own model heads.

## Quick Start

The project is based on Python 3.8+ and PyTorch 1.10+. To install it, simply use:

`pip install grouphug`

### Documentation

Documentation can be generated from docstrings using `make html` in the `docs` directory, but this is not yet on a hosted site. 

### Example usage

```python
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

from grouphug import AutoMultiTaskModel, ClassificationHeadConfig, DatasetFormatter, LMHeadConfig, MultiTaskTrainer

# load some data. 'label' gets renamed in huggingface, so is better avoided as a feature name.
task_one = load_dataset("tweet_eval",'emoji').rename_column("label", "tweet_label")
both_tasks = pd.DataFrame({"text": ["yay :)", "booo!"], "sentiment": ["pos", "neg"], "tweet_label": [0,14]})

# create a tokenizer
base_model = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(base_model)

# preprocess your data: tokenization, preparing class variables
formatter = DatasetFormatter().tokenize().encode("sentiment")
# data converted to a DatasetCollection: essentially a dict of DatasetDict
data = formatter.apply({"one": task_one, "both": both_tasks}, tokenizer=tokenizer, test_size=0.05)

# define which model heads you would like
head_configs = [
    LMHeadConfig(weight=0.1),  # default is BERT-style masked language modelling
    ClassificationHeadConfig.from_data(data, "sentiment"),  # detects dimensions and type
    ClassificationHeadConfig.from_data(data, "tweet_label"),  # detects dimensions and type
]
# create the model, optionally saving the tokenizer and formatter along with it
model = AutoMultiTaskModel.from_pretrained(base_model, head_configs, formatter=formatter, tokenizer=tokenizer)
# create the trainer
trainer = MultiTaskTrainer(
    model=model,
    tokenizer=tokenizer,
    train_data=data[:, "train"],
    eval_data=data[["one"], "test"],
    eval_heads={"one": ["tweet_label"]},  # limit evaluation to one classification task
)
trainer.train()
```

### Tutorials

See [examples](./examples) for a few notebooks that demonstrate the key features.

## Supported Models

The package has support for the following base models:

* Bert, DistilBert, Roberta/DistilRoberta, XLM-Roberta 
* Deberta/DebertaV2
* Electra
* GPT2, GPT-J, GPT-NeoX, OPT

Extending it to support other models is possible by simply inheriting from `_BaseMultiTaskModel`, although language modelling head weights may not always load. 

## Limitations

* The package only supports PyTorch, and will not work with other frameworks. There are no plans to change this.
* Grouphug was developed and tested with 🤗 transformers 4.19-4.22. We will aim to test and keep compatibility with the latest version, but it is still recommended to lock the latest working versions. 

See the [contributing page](CONTRIBUTING.md) if you are interested in contributing.

## License

grouphug was initially developed at [Chatdesk](http://www.chatdesk.com) and is licensed under the Apache 2 [license](LICENSE).

