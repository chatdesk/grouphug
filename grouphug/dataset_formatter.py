import copy
import functools
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import demoji
import numpy as np
import pandas as pd
import regex
import unidecode
from datasets import ClassLabel, Dataset, DatasetDict
from transformers import AutoConfig
from transformers.tokenization_utils_base import LARGE_INTEGER

from . import DatasetCollection
from .config import DEFAULT_SEED, IGNORE_INDEX, INPUT_IDS_VAR, logger
from .dataset_collection import dataset_value_counts, is_iterable


def build_regex_or(entries, regexes=False):
    if regexes:  # regexes, add brackets
        strings = ["(?:" + s + ")" for s in entries]
    else:  # strings
        strings = [regex.escape(s) for s in entries]
    return "(?:" + "|".join(strings) + ")"


def multiple_replacer(replace_dict):
    replacement_function = lambda match: replace_dict[match.group(0)]
    pattern = regex.compile(build_regex_or(replace_dict.keys()), regex.M)
    return lambda string: pattern.sub(replacement_function, string)


# various replacements of common formatting errors
NORMALIZATIONS = {
    "&amp;": "&",
    "&nbsp;": " ",
    "&quot;": '"',
    "&lt;": "<",
    "&gt;": ">",
    "<br>": "\n",
    "<p>": "\n",
    "</p>": "\n",
    "’": "'",
    "`": "'",
    "…": "...",
}
character_normalizer = multiple_replacer(NORMALIZATIONS)  # efficient replacement

# regular expression for @username on instagram/twitter
HANDLES_RE = regex.compile(
    r"(?<![A-Za-z0-9_!@#\$%&*])@" r"(([A-Za-z0-9_]){15}(?!@)|([A-Za-z0-9_]){1,14}(?![A-Za-z0-9_]*@))"
)

# [: ] compensates for some aggressive pre-processing
# TODO: ending brackets issue? <http://www.foo.com> ?
URLS_RE = regex.compile(r"(?:https?[: ]//|(?<![\w])www\.)\S+")


def normalize_spaces(text):
    return regex.sub(r"\s+", " ", text).strip()


def tolower(text):
    return text.lower()


def truncate_repeat(text):
    """3 of the same character in a row -> only 3"""
    return regex.sub(r"(.)\1{2,}", r"\1\1\1", text)


def replace_handles(text, replacement="@USER"):
    """twitter/instagram handles"""
    return HANDLES_RE.sub(replacement, text)


def remove_handles(text):
    return replace_handles(text, " ")


def replace_urls(text, replacement=" URL "):
    """urls"""
    return URLS_RE.sub(replacement, text)


def run_unidecode(text):
    return unidecode.unidecode(text)


demoji.set_emoji_pattern()


@functools.lru_cache(maxsize=1000)
def _demoji_build_replacement(emoji, start_marker, end_marker, separator):
    desc = demoji._CODE_TO_DESC[emoji]
    # drop skin tones, and compress to encourage it to be a single token if common
    parts = [p.replace(" ", "").replace("-", "").lower() for p in desc.split(":") if "skin tone" not in p]
    if not parts:
        return ""  # orphan skin tone
    return start_marker + separator.join(parts) + end_marker


# regex is faster here, emoji's trie approach is another 10x faster
_EMOJI_PAT = regex.compile(demoji._EMOJI_PAT.pattern)


def demojize_text(text, start_marker=" :|", end_marker=" ", separator="|"):  # somewhat optimized for few sentencepieces
    return _EMOJI_PAT.sub(lambda m: _demoji_build_replacement(m[0], start_marker, end_marker, separator), text)


class DatasetFormatter:
    PREPROCESS = "preprocess"
    TOKENIZE = "tokenize"
    ENCODE = "encode"
    BINARIZE = "binarize"

    PREPROCESSORS = {
        "lowercase": tolower,
        "normalize": character_normalizer,
        "truncate_repeat": truncate_repeat,
        "replace_handles": replace_handles,
        "remove_handles": remove_handles,
        "replace_urls": replace_urls,
        "demojize": demojize_text,
        "unidecode": run_unidecode,
        "normalize_spaces": normalize_spaces,
    }

    @classmethod
    def register_preprocessor(cls, key, function):
        """Use this to add your own preprocessor function, but make sure to do it at both training and inference"""
        cls.PREPROCESSORS[key] = function

    def __init__(
        self, save_oov: bool = False, operations: List[Tuple] = None, drop_columns: Optional[Iterable[str]] = None
    ):
        """DatasetFormatter is a pipeline for formatting and preparing your DatasetCollection.
        save_oov: save texts that resulted in out-of-vocabulary tokens in .oov_texts
        Other args: used to load."""
        self.operations = copy.deepcopy(operations or [])
        for op, column, output_column, args in self.operations:
            if "label_names" in args and "label2id" not in args:
                args["label2id"] = {k: i for i, k in enumerate(args["label_names"])}
        self.drop_columns = set(drop_columns or [])
        self.save_oov = save_oov
        self.oov_texts = []

    def preprocess(self, operations: List[str], column="text", output_column=None) -> "DatasetFormatter":
        """Adds a preprocessing step to your pipeline.

        Args:
           operations:
             * List of strings from `register_preprocessor` or one of the following built-in:
                 * lowercase: convert text to lower case
                 * normalize: convert text to lower case
                 * truncate_repeat: replaces 3+ repeats of the same character with 3
                 * replace_handles: replace @username with @USER
                 * replace_urls: replace urls with URL
                 * demojize: replaces emojis with an ascii equivalent, ignoring skin tone modifiers
                 * unidecode: converts all characters to ascii using unidecode
                 * normalize_spaces: replaces multiple spaces with one. typically does not affect models, more for inspection."""
        for operation in operations:
            if operation not in self.PREPROCESSORS:
                raise ValueError(f"Unknown operation {operation}")
        self.operations.append((self.PREPROCESS, column, output_column or column, operations))
        return self

    def tokenize(
        self,
        column: Union[str, Tuple[str, str]] = "text",
        output_prefix="",
        drop=True,
        truncation=True,
        padding=False,
        return_special_tokens_mask=True,
        **x_tokenizer_args,
    ) -> "DatasetFormatter":
        """Adds a tokenizing step to your pipeline. Target is {output_prefix}input_ids and such.
        Note that the default collator in MultiTaskTrainer will dynamically pad, but will not truncate, hence the defaults here"""
        tokenizer_args = dict(
            truncation=truncation,
            padding=padding,
            return_special_tokens_mask=return_special_tokens_mask,
            **x_tokenizer_args,
        )
        if drop:
            self.drop_columns |= {column} if isinstance(column, str) else set(column)
        self.operations.append((self.TOKENIZE, column, output_prefix, tokenizer_args))
        return self

    def encode(
        self, column="labels", label_names=None, max_labels=None, min_freq=0, output_column=None, drop=True
    ) -> "DatasetFormatter":
        """Adds a step to encode labels for classification (from strings/numbers to 0..num_labels-1)

        Args:
            column: input column
            output_column: output column, default overwrites input column
            drop: drop input column if not used as output column
            label_names: If label_names is given, only these are encoded and missing labels will get value -100.  Default: all labels are used
            max_labels/min_freq: If max_labels/min_freq is given, labels are detected and most common ones above threshold used for label_names
        """
        if drop and output_column not in [None, column]:
            self.drop_columns.add(column)
        self.operations.append(
            (
                self.ENCODE,
                column,
                output_column or column,
                dict(label_names=label_names, min_freq=min_freq, max_labels=max_labels),
            )
        )
        return self

    def binarize(
        self,
        column="labels",
        ignore_column=None,
        label_names=None,
        max_labels=None,
        min_freq=0,
        output_column=None,
        drop=True,
    ):
        """Similar to encode, but results in a one-hot encoded or binarized column. Source can be either lists or single values."""
        self.operations.append(
            (
                self.BINARIZE,
                column,
                output_column or column,
                {
                    "label_names": label_names,
                    "ignore_column": ignore_column,
                    "min_freq": min_freq,
                    "max_labels": max_labels,
                },
            )
        )
        if drop and output_column not in [None, column]:
            self.drop_columns.add(column)
        if drop and ignore_column not in [None, column]:
            self.drop_columns.add(ignore_column)
        return self

    def apply(
        self,
        data: Union[
            DatasetCollection, Dataset, DatasetDict, pd.DataFrame, Dict[str, Union[Dataset, DatasetDict, pd.DataFrame]]
        ],
        tokenizer=None,
        test_size: Union[float, Dict[str, float]] = 0.05,
        shuffle: bool = True,
        seed=DEFAULT_SEED,
        splits=("train", "test", "validation"),
        batch_size=100,
        **map_args,
    ) -> DatasetCollection:
        """
        Formats your data

        Args:
            data: if not a DatasetCollection, will make it one using test_size/shuffle. If a single entry is given it will be called 'data'.
            test_size, shuffle, seed: for shuffle: passed to DatasetCollection
            splits: which splits are mapped and used in determining labels
            batch_size, map_args: arguments for map"""
        if isinstance(data, (Dataset, DatasetDict, pd.DataFrame)):
            data = {"data": data}
        if not isinstance(data, DatasetCollection):
            data = DatasetCollection(data, test_size=test_size, shuffle=shuffle, seed=seed)

        # prepare automatic encoder/binarizer
        feature_label_names = {}
        tokenizer_args = None
        for op, column, output_column, args in self.operations:
            if op in [self.ENCODE, self.BINARIZE]:
                if args["label_names"] is None:
                    logger.info(f"Automatically determining labels for {op} on {column}")
                    n, counts = dataset_value_counts(data.gather_column(column, splits=splits), column)
                    min_freq = args["min_freq"]
                    max_labels = args["max_labels"]
                    sorted_counts = sorted(
                        [
                            (c, k)
                            for k, c in counts.items()
                            if k != IGNORE_INDEX  # TODO: option?
                            and ((min_freq >= 1 and c >= min_freq) or (min_freq < 1 and c / n >= min_freq))
                        ],
                        reverse=True,
                    )
                    if max_labels:
                        sorted_counts = sorted_counts[:max_labels]
                    label_names = sorted([k for c, k in sorted_counts])
                    freqs = [{k: c / n for c, k in sorted_counts}]
                    feature_label_names[column] = ClassLabel(names=[str(k) for k in label_names])  # only accepts str
                    logger.info(f"Determined labels and frequencies for {column} as {freqs}")
                    args["label_names"] = label_names
                feature_label_names[column] = ClassLabel(names=[str(k) for k in args["label_names"]])  # only str
                args["label2id"] = {k: i for i, k in enumerate(args["label_names"])}
            elif op == self.TOKENIZE:
                tokenizer_args = args

        if tokenizer_args is not None:
            if tokenizer is None:
                raise ValueError("Should pass a tokenizer if tokenizing a column")
            if tokenizer.model_max_length > LARGE_INTEGER and "max_length" not in tokenizer_args:
                try:
                    config = AutoConfig.from_pretrained(tokenizer.name_or_path)
                    tokenizer.model_max_length = config.max_position_embeddings
                    logger.warning(
                        f"Tokenizer has no .model_max_length and no max_length passed to tokenize, setting it to {tokenizer.model_max_length} based on model max_position_embeddings"
                    )
                except Exception as e:
                    logger.warning(f"Error while trying to set tokenizer max length: {e}")

        # log and check ops
        ops_log = [
            f"{col}: {op}({args})" + (f" -> {tocol}" if tocol and tocol != col else "")
            for (op, col, tocol, args) in self.operations
        ]
        logger.info(f"Applying the following operations to all datasets: {ops_log}")

        for ds_name, dataset_dict in data.items():
            for ds_split, dataset in dataset_dict.items():

                if ds_split in splits:
                    old_fingerprint = dataset_dict[ds_split]._fingerprint
                    old_oov_len = len(self.oov_texts)
                    #  new_fingerprint = update_fingerprint(old_fingerprint, op_name, ops_log)
                    remove_columns = list(self.drop_columns & set(dataset.features))
                    dataset_dict[ds_split] = dataset_dict[ds_split].map(
                        self._format_batch,
                        fn_kwargs={"tokenizer": tokenizer, "operations": self.operations},
                        batch_size=batch_size,
                        batched=True,
                        remove_columns=remove_columns,
                        **map_args,
                    )
                    new_fingerprint = dataset_dict[ds_split]._fingerprint
                    new_oov = len(self.oov_texts) - old_oov_len
                    oov_message = f"{new_oov} texts with out-of-vocabulary (<unk>) tokens. " if new_oov else ""
                    # Fingerprint {new_fingerprint}
                    logger.info(
                        f"Formatted dataset {ds_name}[{ds_split}], {len(dataset_dict[ds_split])} samples. Dropping {','.join(remove_columns) or '<nothing>'}, features = {','.join(dataset_dict[ds_split].features.keys())}. {oov_message}"
                    )
                    # add names to feature. TODO: cast? does not like binarized or some other way to pass id2label to .from_data
                    for column, classlabel in feature_label_names.items():
                        if column in dataset_dict[ds_split].features:
                            dataset_dict[ds_split].features[column] = classlabel

        if tokenizer_args and self.save_oov:
            logger.info(f"{len(self.oov_texts)} texts with out-of-vocabulary (<unk>) tokens stored in .oov_texts")

        return data

    def apply_batch(
        self,
        batch: Dict[str, List[Any]],
        tokenizer=None,
    ) -> Dict[str, List[Any]]:
        """Format single batch, to be used for inference and such"""
        if tokenizer is None and any(opargs[0] == self.TOKENIZE for opargs in self.operations):
            raise ValueError("Should pass a tokenizer if tokenizing a column")
        return {
            k: v for k, v in self._format_batch(batch, tokenizer, self.operations).items() if k not in self.drop_columns
        }

    def _format_batch(self, batch, tokenizer, operations: List[Tuple]) -> Dict[str, List]:
        """used by apply..."""

        # TODO: cache?
        def binarize_labels(values, ignore_values, label2id: Dict):
            encoded_values = []
            for labels, ignore_labels in zip(values, ignore_values):
                binarized_labels = np.zeros(len(label2id))
                if not is_iterable(labels):
                    labels = [labels]  # allow binarizing single values
                if ignore_values:  # set ignore first, then potentially overwrite with positive labels
                    for label in ignore_labels:
                        ix = label2id.get(label)
                        if ix is not None:
                            binarized_labels[ix] = IGNORE_INDEX
                for label in labels:
                    ix = label2id.get(label)
                    if ix is not None:
                        binarized_labels[ix] = 1
                encoded_values.append(binarized_labels)
            return encoded_values

        def encode_labels(values, label2id: Dict):
            return [label2id.get(label, IGNORE_INDEX) for label in values]

        output = batch
        for operation, column, output_col, args in operations:
            if operation == DatasetFormatter.TOKENIZE:  # potentially multi column with [SEP]
                if isinstance(column, str):
                    if column not in batch:
                        continue
                    tokenized = tokenizer(batch[column], **args)
                else:
                    if not all(c in batch for c in column):
                        continue
                    pairs = list(zip(*[batch[c] for c in column]))
                    tokenized = tokenizer(pairs, **args)
                if self.save_oov:
                    self.oov_texts.extend(
                        [
                            (text, tokenizer.decode([t for t in tokens if t != tokenizer.pad_token_id]))
                            for tokens, text in zip(tokenized[INPUT_IDS_VAR], batch[column])
                            if tokenizer.unk_token_id in tokens
                        ]
                    )

                output.update({output_col + k: v for k, v in tokenized.items()})
            else:  # all single column ops
                if column not in batch:
                    continue
                if operation == DatasetFormatter.PREPROCESS:
                    texts = batch[column]
                    for operation in args:
                        f = self.PREPROCESSORS[operation]
                        texts = [f(text) for text in texts]
                    output[output_col] = texts

                elif operation == DatasetFormatter.ENCODE:
                    output[output_col] = encode_labels(batch[column], args["label2id"])
                elif operation == DatasetFormatter.BINARIZE:
                    ignore_col = args.get("ignore_column")
                    if ignore_col:
                        if ignore_col not in batch:
                            raise ValueError(f"Missing ignore_column '{ignore_col}' in binarizing '{column}'")
                        ignore_values = batch[ignore_col]
                    else:
                        ignore_values = [[] for _ in batch[column]]
                    output[output_col] = binarize_labels(batch[column], ignore_values, args["label2id"])
                else:
                    raise ValueError(f"Unknown formatting operation {operation}")
        return output

    # for loading/saving
    def to_dict(self) -> Dict:
        operations = copy.deepcopy(self.operations)
        for op, _, _, args in operations:
            if op in [self.ENCODE, self.BINARIZE]:
                args.pop("label2id", None)  # can not encode non-string keys in json
        return {"operations": operations, "drop_columns": list(self.drop_columns)}

    @classmethod
    def from_dict(cls, data: Dict) -> "DatasetFormatter":
        return cls(**data)
