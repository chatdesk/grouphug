import json
import os
from abc import ABC
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from torch import nn
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BertPreTrainedModel,
    DebertaPreTrainedModel,
    DebertaV2PreTrainedModel,
    DistilBertPreTrainedModel,
    ElectraPreTrainedModel,
    GPT2PreTrainedModel,
    GPTJPreTrainedModel,
    GPTNeoXPreTrainedModel,
    OPTPreTrainedModel,
    PreTrainedTokenizerBase,
    RobertaPreTrainedModel,
    XLMRobertaConfig,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import ModelOutput

from grouphug import DatasetFormatter
from grouphug.config import (
    FORMATTER_FILE_NAME,
    HEADS_FILE_NAME,
    INPUT_EMBEDDING_VAR,
    INPUT_IDS_VAR,
    MASKED_PREFIX,
    TOKENIZER_VARS,
    logger,
)
from grouphug.heads import LMHeadConfig, head_config_from_dict
from grouphug.heads.base import HeadConfig

tqdm.pandas()


@dataclass
class MultiTaskOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models.

    Args:
       loss: Total weighted loss across all heads
       logits: Logits of the active tasks as tuple if more than one, or just a tensor otherwise
       head_outputs: Results of model heads
       embeddings: embeddings (final hidden states of the base model) by prefix
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[Union[torch.FloatTensor, Tuple[torch.FloatTensor]]] = None  # active task logit(s)
    head_outputs: Dict[str, ModelOutput] = None
    embeddings: Dict[str, torch.Tensor] = None


class ModelInferenceError(Exception):
    """Raised when model.forward fails, saving the batch and the head that failed for easier interactive debugging"""

    def __init__(self, message, batch=None, cls=None, head=None):
        super().__init__(message)
        self.head = head
        self.orig_cls = cls
        self.batch = batch


DEFAULT_IGNORE_MISSING = ["lm_head.lm_head", "lm_head.decoder.weight"]
DEFAULT_IGNORE_SAVE = ["lm_head.decoder.weight"]


class _BaseMultiTaskModel(ABC):
    AUTOMODEL_CLASSES = []

    def __init_subclass__(cls, register_auto_class=True, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._keys_to_ignore_on_load_missing += DEFAULT_IGNORE_MISSING
        cls._keys_to_ignore_on_save += DEFAULT_IGNORE_SAVE
        if register_auto_class and hasattr(cls, "config_class"):
            _BaseMultiTaskModel.AUTOMODEL_CLASSES.append((cls.config_class, cls))

    def __init__(
        self,
        config: PretrainedConfig,
        head_configs: List[HeadConfig],
        formatter: "DatasetFormatter" = None,
        tokenizer=None,
    ):
        super().__init__(config)
        self.config = config
        self._tokenizer = tokenizer

        # cached convenience vars
        self._vars = None

        # this is self.bert / self.roberta, and such
        self._init_base_model(config)

        self.formatter = formatter
        self.head_configs = self._create_heads(config, head_configs)
        self._active_heads = self.head_configs
        # each head is stored in either it's specified attribute, or in a ModuleDict
        self.other_heads = nn.ModuleDict()  # heads without attribute
        for hc in self.head_configs:
            if hc.attribute:
                setattr(self, hc.attribute, hc._head)
            else:
                self.other_heads[hc.name] = hc._head

        # Initialize weights and apply final processing
        self.post_init()

    def _init_base_model(self, config):  # does not have pooling layer option
        # this is self.bert / self.roberta, and such
        setattr(self, self.base_model_prefix, AutoModel.from_config(config))

    # @property
    # def base_model(self):
    #    return getattr(self, self.base_model_prefix)

    def get_mlm_head(self) -> Optional[LMHeadConfig]:
        for hc in self.head_configs:
            if isinstance(hc, LMHeadConfig):
                return hc

    @staticmethod
    def _create_heads(config: PretrainedConfig, head_configs: List[HeadConfig]):
        """populates additional fields in configs"""
        # TODO: could allow other heads' outputs as inputs?
        names_used = set()
        output_embeds = []
        for hc in head_configs:
            hc._required_input_vars = hc.input_vars()
            hc._head = hc.create_head(config)  # may set attribute as well
            hc.name = (hc.name or hc._name()).replace(".", "_")  # . invalid in module names
            while hc.name in names_used:
                hc.name += "(duplicate name)"  # should be rare, no sensible names for you
            if hasattr(hc._head, "head_output_embeddings"):
                output_embeds.append(hc._head)
        if len(output_embeds) > 1:
            logger.warning(
                "Found multiple heads with output embeddings, you will need to tie weights yourself before training!"
            )
        return head_configs

    def set_active_heads(self, heads: List[Union[str, HeadConfig]] = None):
        """Limit model to run only a subset of heads, either by head config name or config itself"""
        if heads is None:
            self._active_heads = self.head_configs
        else:
            self._active_heads = [hc for hc in self.head_configs if any(hc is t or hc.name == t for t in heads)]

    def get_active_heads(self):
        return self._active_heads

    @contextmanager
    def active_heads(self, heads: List[Union[str, HeadConfig]] = None):
        old_heads = self.get_active_heads()
        self.set_active_heads(heads)
        try:
            yield self
        finally:
            self.set_active_heads(old_heads)

    # tokenizer helper
    def tokenizer(self, name_or_path=None, **kwargs):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(name_or_path or self.config._name_or_path, **kwargs)
        return self._tokenizer

    # ...
    def input_prefixes(self) -> Set[str]:
        """Which prefixes for input_ids are potentially used in .forward ?"""
        return {p for hc in self.head_configs for p in hc.input_prefixes() if p != MASKED_PREFIX}

    def vars(self) -> Set[str]:
        """Which variables are potentially used in .forward ?
        Cached and used for inference and collating"""
        if not self._vars:
            input_prefixes = self.input_prefixes()
            tokenizer_vars = {p + tv for p in input_prefixes for tv in TOKENIZER_VARS}  # also include masks etc
            self._vars = {v for hc in self.head_configs for v in hc.input_vars()["train"]} | tokenizer_vars
        return self._vars

    # loading

    def get_output_embeddings(self):  # used by ~.from_pretrained via default tie_weights
        mlm_hc = self.get_mlm_head()
        if mlm_hc is not None:
            return mlm_hc._head.head_output_embeddings()
        else:
            return None

    def set_output_embeddings(self, new_embeddings):
        raise NotImplementedError(
            "set_output_embeddings is not implemented, use tie weights."
        )  # in resizing embeddings?

    # inference methods

    def calculate_model_embeddings(self, prefix="", **kwargs):
        encoder = self.base_model
        # mlm just gives input ids, so we keep the other vars
        no_mlm_prefix = "" if prefix == MASKED_PREFIX else ""
        optional_args = dict(
            attention_mask=kwargs.get(no_mlm_prefix + "attention_mask"),
            token_type_ids=kwargs.get(no_mlm_prefix + "token_type_ids"),
            position_ids=kwargs.get(no_mlm_prefix + "position_ids"),
        )
        optional_args = {k: v for k, v in optional_args.items() if v is not None}
        return encoder(kwargs[prefix + INPUT_IDS_VAR], return_dict=True, **optional_args)  # not optional

    def forward(self, inference_only: bool = False, **kwargs):
        r"""Determines which heads can be run, and returns weighted loss over them along with individual head outputs

        Args:
            inference_only: will run heads even when labels are missing, and return full details
            return_embeddings: will return a dict in .prefix_to_embedding"""
        # Which heads can we infer with these args?
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        # which head can we run given the inputs and mode?
        rv_type = "infer" if inference_only else "train"
        relevant_heads = [hc for hc in self._active_heads if all(v in kwargs for v in hc._required_input_vars[rv_type])]
        # Which inputs do we need to encode? Usually just 'input_ids' (so prefixes=[""])
        relevant_input_prefixes = {p for h in relevant_heads for p in h.input_prefixes()}

        # TODO: this is pretty naive, and could be cached (not recalculating non-masked entries etc)
        try:
            prefix_to_embedding = {
                f"{p}{INPUT_EMBEDDING_VAR}": self.calculate_model_embeddings(p, **kwargs)
                for p in relevant_input_prefixes
            }
        except Exception as e:  # here we tell people in which head the error was, otherwise it's hard to debug
            raise ModelInferenceError(f"Error in calculating embeddings: {e}", cls=e.__class__, batch=kwargs) from e
        # run heads and collect losses
        outputs = {}
        losses = []
        for hc in relevant_heads:
            combined_args = {**kwargs, **prefix_to_embedding}
            try:
                head_output = hc._head(**combined_args)
            except Exception as e:  # here we tell people in which head the error was, otherwise it's hard to debug
                raise ModelInferenceError(
                    f"Error in head {hc.name}: {e}", cls=e.__class__, head=hc._head, batch=combined_args
                ) from e
            outputs[hc.name] = head_output
            if head_output.loss is not None:
                losses.append(head_output.loss * hc.weight)

        loss = None
        if losses:
            loss = sum(losses)
        elif not inference_only:
            req_vars = {str(hc): hc._required_input_vars[rv_type] for hc in self._active_heads}
            raise ModelInferenceError(
                f"No valid heads among {len(self._active_heads)} active heads for batch with arguments {kwargs.keys()} and inference_only=False, required vars = {req_vars}",
                batch=kwargs,
            )

        task_logits = tuple(getattr(outputs.get(hc.name), "logits", None) for hc in self._active_heads)
        if any(l is None for l in task_logits):
            task_logits = None  # huggingface doesn't like mixing in None
        elif len(task_logits) == 1:
            task_logits = task_logits[0]  # huggingface will helpfully change the type of labels, so we do this too

        # by default return only loss + logits, since huggingface trainer does not like dicts in outputs
        return MultiTaskOutput(
            loss=loss,
            logits=task_logits,
            head_outputs=outputs if inference_only else None,
            embeddings=prefix_to_embedding if inference_only else None,
        )

    def format_forward(self, tokenizer=None, **kwargs):
        """Simply wraps variables in a list and passes to format_forward_batch for convenience"""
        return self.format_forward_batch(tokenizer=tokenizer, **{k: [v] for k, v in kwargs.items()})

    def _tensorize(self, v: Any) -> Union[str, torch.Tensor]:
        return v if isinstance(v, str) else torch.tensor(v, device=self.device)

    def format_forward_batch(self, tokenizer=None, **kwargs):
        """If any kind of expected input_ids is passed, assumes data is formatter
        kwargs: variables, either single data points or a batch
        """
        if not any(p + INPUT_IDS_VAR in kwargs for hc in self.head_configs for p in hc.input_prefixes()):
            if not self.formatter:
                raise ValueError("Expecting either input_ids, or a formatter present. Pass one to from_pretrained!")
            tokenizer = tokenizer or self.tokenizer()
            data = self.formatter.apply_batch(kwargs, tokenizer=tokenizer)
        else:
            data = kwargs
        model_vars = self.vars()
        data = {k: self._tensorize(np.array(v)) for k, v in data.items() if k in model_vars}  # np array avoids warning
        with torch.no_grad():
            return self.forward(inference_only=True, **data)

    def predict(
        self,
        data: Union[Dict, Iterable[Dict], DatasetDict, Dataset, pd.DataFrame],
        heads: List = None,
        tokenizer=None,
        show_progress: bool = False,
    ):
        """Runs the model on some data and gives predictions.
        Uses .format_forward_batch, so input data does not have to be formatted.

        Args:
            data:
              * dict -> assumed to be a single sample, returns a dict with results
              * Dataset or Iterable -> list of such dicts
              * DatasetDict -> dict of such lists
              * DataFrame -> DataFrame
            heads: Only run these heads (via .active_heads)
            tokenizer: passed to format_forward_batch
            show_progress: use tqdm to show progress"""
        if heads is not None:
            with self.active_heads(heads):
                return self.predict(data, tokenizer=tokenizer)

        def process_record(record):
            result = self.format_forward(tokenizer=tokenizer, **record)
            stats = {}
            if result.loss is not None:
                stats["loss"] = result.loss.item()
            for hc in self.head_configs:
                if hc.name in result.head_outputs:
                    head_stats = hc.output_stats(result.head_outputs[hc.name])
                    stats.update({hc.name + "_" + k: v for k, v in head_stats.items()})
            return stats

        if isinstance(data, pd.DataFrame):
            data_apply_f = data.progress_apply if show_progress else data.apply
            return data_apply_f(lambda r: pd.Series(process_record(r)), axis=1)
        if isinstance(data, DatasetDict):
            return {k: self.predict(v) for k, v in data.items()}
        if isinstance(data, dict):
            return process_record(data)
        if isinstance(data, Iterable):
            if show_progress:
                data = tqdm(data)
            return [process_record(record) for record in data]
        raise ValueError(f"Unknown format {data.__class__} for data")

    @classmethod
    def _load_pretrained_model(cls, model, state_dict, loaded_keys, *args, **kwargs):
        # since some huggingface models stick different parameters in the base model which can be used in a head, we rename some here
        # TODO: low mem version?
        renamed = {}
        for hc in model.head_configs:
            for from_key, to_key in hc._head._rename_keys:
                if from_key in state_dict:
                    state_dict[to_key] = state_dict[from_key]
                    del state_dict[from_key]
                    renamed[from_key] = to_key  # do not modify in iteration
        if renamed:
            logger.warning(f"Renaming {renamed} in loading pre-trained model")
        return super()._load_pretrained_model(model, state_dict, state_dict.keys(), *args, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, head_configs=None, formatter=None, *args, **kwargs):
        """Loads model, head configs, data formatter"""
        if head_configs is None:
            config_file = os.path.join(pretrained_model_name_or_path, HEADS_FILE_NAME)
            if not os.path.isfile(config_file):
                raise ValueError(
                    f"Must give either head_configs directly, or have {HEADS_FILE_NAME} in directory {pretrained_model_name_or_path}"
                )
            logger.info(f"loading heads config file {config_file}")
            with open(config_file, "r") as f:
                head_config_json = json.load(f)
                head_configs = [head_config_from_dict(hc_dict) for hc_dict in head_config_json]

        if formatter is None:
            formatter_file = os.path.join(pretrained_model_name_or_path, FORMATTER_FILE_NAME)
            if os.path.isfile(formatter_file):  # optional, so no error if missing
                logger.info(f"loading formatter from {formatter_file}")
                with open(formatter_file, "r") as f:
                    formatter = DatasetFormatter.from_dict(json.load(f))

        return super().from_pretrained(
            pretrained_model_name_or_path, head_configs=head_configs, formatter=formatter, *args, **kwargs
        )


# common to bert/roberta models in huggingface is removing the pooling layer
class _BertModelBase(_BaseMultiTaskModel, register_auto_class=False):
    _keys_to_ignore_on_load_unexpected = ["pooler", "cls.seq_relationship"]

    def _init_base_model(self, config):
        setattr(self, self.base_model_prefix, AutoModel.from_config(config, add_pooling_layer=False))


class BertMultiTaskModel(_BertModelBase, BertPreTrainedModel):
    pass


class DistilBertMultiTaskModel(_BaseMultiTaskModel, DistilBertPreTrainedModel):
    pass


class RobertaMultiTaskModel(_BertModelBase, RobertaPreTrainedModel):
    pass


class XLMRobertaMultiTaskModel(_BertModelBase, RobertaPreTrainedModel):
    config_class = XLMRobertaConfig  # sort of the same?


class ElectraMultiTaskModel(_BaseMultiTaskModel, ElectraPreTrainedModel):
    pass


class DebertaMultiTaskModel(_BaseMultiTaskModel, DebertaPreTrainedModel):
    pass


class DebertaV2MultiTaskModel(_BaseMultiTaskModel, DebertaV2PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["position_embeddings", "mask_predictions"]  # common in pretrained
    _keys_to_ignore_on_load_missing = ["position_ids", "cls.predictions.decoder"]


class OPTMultiTaskModel(_BaseMultiTaskModel, OPTPreTrainedModel):
    pass


class GPT2MultiTaskModel(_BaseMultiTaskModel, GPT2PreTrainedModel):
    pass


class GPTJMultiTaskModel(_BaseMultiTaskModel, GPTJPreTrainedModel):
    pass


class GPTNeoXMultiTaskModel(_BaseMultiTaskModel, GPTNeoXPreTrainedModel):
    pass


class AutoMultiTaskModel:
    @staticmethod
    def _model_class_for_config(config):
        automodel_cls = [k for k in _BaseMultiTaskModel.AUTOMODEL_CLASSES if k[0] == config.__class__]
        if not automodel_cls:
            return None
        assert len(automodel_cls) == 1, "Multiple registered auto-classes found, call one of them directly"
        return automodel_cls

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        head_configs: List[HeadConfig] = None,
        formatter: DatasetFormatter = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        **kwargs,
    ) -> _BaseMultiTaskModel:
        """Initialized a model from a pre-trained multi-task or base model

        Args:
            head_configs: model head configurations. Will try to load if omitted.
            formatter: optional DatasetFormatter which will be saved with the model, and can be used to infer on non-formatted data. Will try to load if omitted.
            tokenizer: pass a tokenizer here to avoid it being created by the model when it is missing one.
            kwargs: passed to model init, always empty in current setup, but can be used for your own models
        """
        autoconfig = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        kwargs = dict(head_configs=head_configs, formatter=formatter, tokenizer=tokenizer, **kwargs)

        automodel_cls = cls._model_class_for_config(autoconfig)
        if not automodel_cls:
            raise NotImplementedError(
                f"{pretrained_model_name_or_path} uses {autoconfig.__class__.__name__} which is not supported, see documentation for which models are supported by AutoMultiTaskModel"
            )
        return automodel_cls[0][1].from_pretrained(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def from_config(
        cls,
        config,
        head_configs: List[HeadConfig],
        formatter: DatasetFormatter = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        **kwargs,
    ) -> _BaseMultiTaskModel:
        """See from_pretrained, but taking a config object instead"""
        automodel_cls = cls._model_class_for_config(config)
        return automodel_cls[0][1]._from_config(
            config, head_configs=head_configs, formatter=formatter, tokenizer=tokenizer, **kwargs
        )
