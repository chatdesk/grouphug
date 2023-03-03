# sequence classification
from typing import Any, Dict, Iterable, List, Optional, Set, Union

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput

from grouphug.config import IGNORE_INDEX, INPUT_EMBEDDING_VAR, INPUT_IDS_VAR
from grouphug.heads.base import HeadConfig, ModelHead

from .. import DatasetCollection
from ..config import logger
from ..dataset_collection import dataset_value_counts, is_iterable


class ClassificationHead(ModelHead):
    SINGLE = "single_label_classification"
    MULTI = "multi_label_classification"
    REGRESSION = "regression"
    PROBLEM_TYPES = [SINGLE, MULTI, REGRESSION]

    """Head for sentence-level classification tasks with:
        * Modular setup for pooling, loss, head structure
        * Additional options, see ClassificationHeadConfig"""

    def __init__(self, config: PretrainedConfig, head_config: "ClassificationHeadConfig"):
        super().__init__()
        self.head_config = head_config
        self.config = config
        self.num_labels = head_config.num_labels
        if self.num_labels is None:
            raise ValueError(f"Must set 'num_labels' for head {self}")

        self.num_extra_inputs = head_config.num_extra_inputs
        self.classifier_hidden_size = head_config.classifier_hidden_size or config.hidden_size
        if head_config.dropout is not None:
            self.dropout = head_config.dropout
        else:
            self.dropout = 0

        self.init_modules(config.hidden_size + self.num_extra_inputs, head_config.num_labels)
        # initialize here because of pos_weight
        if self.head_config.problem_type == self.MULTI and head_config.pos_weight is not None:
            if not isinstance(head_config.pos_weight, Iterable):
                head_config.pos_weight = [head_config.pos_weight] * head_config.num_labels
            self.pos_weight = torch.nn.Parameter(torch.Tensor(head_config.pos_weight), requires_grad=False)  # to device
        else:
            self.pos_weight = None

    def init_modules(self, input_dim, output_dim):
        """Overwrite this method to change the architecture of the classification head"""
        hidden_dim = self.classifier_hidden_size
        self.head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.head_config.labels_var})"

    def pool_embedding(self, embeddings, **kwargs):
        """Overwrite this method to change the pooling in the classification head"""
        features = embeddings[0]  # up to this point it is still the dict output of roberta etc.
        if self.head_config.detached:
            features = features.detach()
        if self.head_config.pooling_method in ["cls", "first"]:
            return features[:, 0, :]  # take first token, typically <s> or [CLS]
        if self.head_config.pooling_method == "last":  # last non-padded token
            input_ids: torch.Tensor = kwargs.get(f"{self.head_config.input_prefix}{INPUT_IDS_VAR}")
            if self.config.pad_token_id is None:  # take last non-pad token, usually <eos>
                raise ValueError("No pad token set, can not detect last token to use in classification")
            if (input_ids[:, 0] == self.config.pad_token_id).any():  # left padded
                sequence_lengths = -1
            else:  # right padded, as is usual
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            return features[torch.arange(features.shape[0], device=features.device), sequence_lengths]
        else:
            attention_mask_col = f"{self.head_config.input_prefix}attention_mask"
            if attention_mask_col not in kwargs:
                raise ValueError(f"mean or max pooling requires column {attention_mask_col}")
            attention_mask = kwargs[attention_mask_col]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(features.size()).float()
            if self.head_config.pooling_method == "mean":  # from huggingface docs
                return torch.sum(features * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            elif self.head_config.pooling_method == "max":
                return torch.max(features * input_mask_expanded, 1).values
            else:
                raise ValueError(f"Unknown pooling method {self.head_config.pooling_method}")

    def loss(self, logits: torch.Tensor, labels: torch.Tensor):
        """Overwrite this method to change the loss of the classification head"""
        if self.head_config.problem_type == self.REGRESSION:
            loss_fct = MSELoss()
            if self.num_labels == 1:
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                loss = loss_fct(logits, labels)
        else:
            label_smoothing = self.head_config.label_smoothing if self.training else 0.0
            if self.head_config.problem_type == self.SINGLE:
                loss_fct = CrossEntropyLoss(ignore_index=self.head_config.ignore_index, label_smoothing=label_smoothing)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                if torch.isnan(loss) and (labels == self.head_config.ignore_index).all():
                    loss = 0.0 * logits.sum()  # no labels, zero loss better than nan
            else:
                assert self.head_config.problem_type == self.MULTI
                labels: torch.Tensor = labels.float()  # BCEWithLogitsLoss does not like ints
                ignore_labels_mask: torch.Tensor = labels == self.head_config.ignore_index  # save this before smoothing
                if label_smoothing:
                    labels = labels * (1 - label_smoothing) + torch.ones_like(labels) * 0.5 * label_smoothing
                if ignore_labels_mask.any():  # ignore entries by setting loss to 0
                    loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="none")
                    loss_entries = loss_fct(logits, labels)
                    loss_entries.masked_fill_(ignore_labels_mask, 0.0)
                    loss = loss_entries.mean()  # could have some option for .sum() / (~masked).sum()
                else:
                    loss_fct = BCEWithLogitsLoss(pos_weight=self.pos_weight)
                    loss = loss_fct(logits, labels)

        return loss

    def forward(self, **kwargs):
        if self.head_config.pooling_method == "auto":
            self.head_config.pooling_method = "last" if self.config.is_decoder else "first"
            logger.info(
                f"Set pooling method to '{self.head_config.pooling_method}' for {self} based on config.is_decoder = {self.config.is_decoder}"
            )

        input_embedding = kwargs[f"{self.head_config.input_prefix}{INPUT_EMBEDDING_VAR}"]
        x = self.pool_embedding(input_embedding, **kwargs)

        if self.num_extra_inputs:
            extra_classification_inputs = torch.cat([kwargs[v] for v in self.head_config.extra_inputs_vars], dim=-1)
            if extra_classification_inputs.shape[1] != self.num_extra_inputs:
                raise ValueError(
                    f"Head {self} expected extra_classification_inputs of dimension {self.num_extra_inputs} but found {extra_classification_inputs.shape[1]}"
                )
            x = torch.cat((x, extra_classification_inputs), dim=-1)

        logits = self.head(x)

        labels = kwargs.get(self.head_config.labels_var)
        if labels is not None:
            loss = self.loss(logits, labels)
        else:
            loss = None

        return SequenceClassifierOutput(loss=loss, logits=logits)


class ClassificationHeadConfig(HeadConfig):
    """Config for ClassificationHead

    Args:
        num_labels: Dimension of the labels.
        labels_var: Dataset column to use for labels. Also the default name of the head.
        id2label: List of label names
        problem_type: 'single_label_classification', 'multi_label_classification', 'regression', or None for detecting single/multi
        dropout: dropout used both after embeddings and after hidden layer
        detached: Stops gradients flowing to the base model, equivalent to freezing base model if this is the only head
        pooling method from embeddings to head inputs (first/cls, mean, max, last). 'auto' uses first/last based on config.is_decoder
        classifier_hidden_size: size of middle layer in classification head. default is embedding's `hidden_size`
        extra_inputs_vars: names of vars in dataset for additional context to classification head
        num_extra_inputs: total length of extra_inputs_vars (recommended to have this handled by .from_data)
        label_smoothing: label_smoothing during classification, only applied in training
        pos_weight: parameter for multi label classification's crossentropy loss
        ignore_index: which index to ignore in loss (typically -100 or -1). Also works with multi-label classification.
        other args passed to HeadConfig, most notably 'weight'
    """

    def __init__(
        self,
        problem_type: str,
        num_labels: int,
        labels_var: str = "labels",
        id2label: Optional[List] = None,
        dropout: Optional[float] = 0.1,
        detached: bool = False,
        pooling_method: str = "auto",
        classifier_hidden_size: Optional[int] = None,
        extra_inputs_vars: List[str] = None,
        num_extra_inputs: Optional[int] = 0,
        pos_weight: Union[float, List[float]] = None,
        label_smoothing: float = 0.0,
        ignore_index=IGNORE_INDEX,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if labels_var == "label":  # TODO: in some collator or tokenizers?
            logger.warning("It is not recommended to use 'label' as your labels_var, as transformers renames it")

        self.num_labels = num_labels
        self.labels_var = labels_var
        self.ignore_index = ignore_index
        self.id2label = id2label
        if self.id2label is not None and len(self.id2label) != num_labels:
            raise ValueError(f"id2label (length {len(self.id2label)}) should have length num_labels = {num_labels}")
        self.problem_type = problem_type
        if self.problem_type not in ClassificationHead.PROBLEM_TYPES:
            raise ValueError(
                f"Unknown problem type {self.problem_type}, expecting one of {ClassificationHead.PROBLEM_TYPES}"
            )
        self.dropout = dropout
        self.detached = detached
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.pooling_method = pooling_method
        self.classifier_hidden_size = classifier_hidden_size
        self.extra_inputs_vars = extra_inputs_vars or []
        self.num_extra_inputs = num_extra_inputs

    def create_head(self, config):
        return ClassificationHead(config, self)

    # For internal use
    def input_vars(self) -> Dict[str, Set[str]]:  # set of variables required to run train/inference on head
        infer_vars = super().input_vars()["infer"] | set(self.extra_inputs_vars)
        return {"train": infer_vars | {self.labels_var}, "infer": infer_vars}

    def _name(self):  # generates name if not given
        return self.labels_var

    def __repr__(self):
        return f"{self.__class__.__name__}(labels_var = {self.labels_var}, problem_type = {self.problem_type}, num_labels = {self.num_labels})"

    @classmethod
    def from_data(
        cls,
        data: DatasetCollection,
        labels_var,
        num_labels=None,
        problem_type=None,
        extra_inputs_vars: List[str] = None,
        num_extra_inputs: Optional[int] = None,
        pooling_method: str = "auto",
        classifier_hidden_size: Optional[int] = None,
        id2label: List[Any] = None,
        ignore_index: int = IGNORE_INDEX,
        **kwargs,
    ):
        """Creates a classification head from a DatasetCollection, automatically determining classification type and number of labels

        Args:
             labels_var: Which
             num_labels: Dimension of the labels, automatically determined when omitted
             problem_type: determined using get_classification_type if omitted
             extra_inputs_vars: Which additional columns should be used as inputs to the head, in addition to the embeddings?
             num_extra_inputs: Total dimension of these, automatically determined when omitted
             pooling_method, classifier_hidden_size, kwargs: passed to the constructor
        """
        # TODO: option for class balancing?

        auto_determined = {}
        if problem_type is None:
            problem_type = get_classification_type(data, labels_var)
            auto_determined["problem_type"] = problem_type
        label_data = data.gather_column(labels_var, splits=["train", "test"])

        if num_labels is None:
            for ds in label_data:
                feature_names = getattr(ds.features[labels_var], "names", None)
                if feature_names:
                    num_labels = len(feature_names)
                    break
            else:
                if problem_type == ClassificationHead.SINGLE:
                    n, counts = dataset_value_counts(label_data, labels_var)
                    if ignore_index in counts:
                        del counts[ignore_index]
                    num_labels = len(counts)
                elif problem_type == ClassificationHead.REGRESSION:
                    if is_iterable(label_data[0][labels_var][0]):  # regression to a list of labels
                        num_labels = len(label_data[0][labels_var][0])
                    else:  # single value regression
                        num_labels = 1
                else:  # multi label/onehot
                    num_labels = len(label_data[0][labels_var][0])
            auto_determined["num_labels"] = num_labels

        if extra_inputs_vars and num_extra_inputs is None:
            num_extra_inputs = 0
            for column in extra_inputs_vars:
                col_data = data.gather_column(column, splits=["train"])
                if not col_data:
                    raise ValueError(f"No dataset with feature {column} found in training data")
                try:
                    num_extra_inputs += len(col_data[0][column][0])
                except Exception as e:
                    raise ValueError(
                        f"Expected a list in {column}, but found {col_data[0][column][0]} as first value. Make sure to binarize and not encode '{column}' in formatting data: {e}"
                    )
            auto_determined["num_extra_inputs"] = num_extra_inputs

        if id2label is None:
            for ds in label_data:
                id2label = id2label or getattr(ds.features[labels_var], "names", None)
            if id2label is not None:
                auto_determined["id2label"] = id2label

        if auto_determined:
            logger.info(
                f"Automatically determined parameters for classification on '{labels_var}' as {auto_determined}"
            )

        return cls(
            labels_var=labels_var,
            num_labels=num_labels,
            problem_type=problem_type,
            id2label=id2label,
            pooling_method=pooling_method,
            classifier_hidden_size=classifier_hidden_size,
            num_extra_inputs=num_extra_inputs,
            extra_inputs_vars=extra_inputs_vars,
            **kwargs,
        )

    def output_stats(self, output: SequenceClassifierOutput) -> Dict[str, Any]:
        """Turns head output into a set of richer statistics"""
        stats = super().output_stats(output)
        if output.logits is not None:
            logits = output.logits[0].detach().cpu()
            np_logits = logits.numpy()
            if self.problem_type == ClassificationHead.SINGLE:
                stats["probs"] = torch.softmax(logits, dim=0).numpy()
                stats["predicted_id"] = np_logits.argmax()
                if self.id2label is not None:
                    stats["predicted_label"] = self.id2label[stats["predicted_id"]]
            elif self.problem_type == ClassificationHead.MULTI:
                stats["probs"] = torch.sigmoid(logits).numpy()
                stats["predicted_ids"] = [i for i, p in enumerate(stats["probs"]) if p > 0.5]
                if self.id2label is not None:
                    stats["predicted_labels"] = [self.id2label[i] for i in stats["predicted_ids"]]
            elif self.problem_type == ClassificationHead.REGRESSION:
                if self.num_labels == 1:
                    stats["predicted_value"] = np_logits[0]
                else:
                    stats["predicted_values"] = np_logits
        return stats


def get_classification_type(data: DatasetCollection, column: str, splits=("train", "test")) -> str:
    """Determines what kind of problem type is likely specified by the column"""
    datasets = data.gather_column(column, splits)

    if not datasets:
        raise ValueError(f"Can not determine classification type for column {column}, as no dataset contains it")
    if is_iterable(datasets[0][column][0]):
        return ClassificationHead.MULTI
    elif datasets[0].features[column].dtype in ["float32", "float64"]:  # could be SINGLE as well
        return ClassificationHead.REGRESSION
    else:
        return ClassificationHead.SINGLE
