from typing import Callable, Dict, Optional, Set

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    BertConfig,
    DebertaConfig,
    DebertaV2Config,
    DistilBertConfig,
    ElectraConfig,
    OPTConfig,
    PretrainedConfig,
    RobertaConfig,
    XLMRobertaConfig,
)
from transformers.activations import gelu, get_activation
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.bert.modeling_bert import BertLMPredictionHead

# MLM heads
# Since huggingface is rather inconsistent in how MLM heads are implemented, we pretend to be the 'official' one as good as we can
from transformers.models.deberta.modeling_deberta import DebertaLMPredictionHead
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2LMPredictionHead

from grouphug.config import (
    CLM_LABELS_VAR,
    IGNORE_INDEX,
    INPUT_EMBEDDING_VAR,
    MASKED_PREFIX,
    MLM_LABELS_VAR,
    MTD_LABELS_VAR,
    MTD_TOKEN_RANDOM,
    MTD_TOKEN_SIMILARITY,
    logger,
)
from grouphug.heads.base import HeadConfig, ModelHead


class MTDPredictions(nn.Module):  # from transformers.models.electra.modeling_electra
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)
        self.config = config
        self.activation = get_activation(getattr(config, "hidden_act", "gelu"))  # Changed: robust default

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.activation(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        return logits


class BaseLMHead(ModelHead):
    def __init__(self, config: PretrainedConfig, head_config: "LMHeadConfig"):
        super().__init__()
        self.config = config
        self.head_config = head_config
        if self.head_config.masked_language_modelling:
            self.init_mlm_head()
        if self.head_config.masked_token_detection:
            self.init_mtd_head()
        if self.head_config.causal_language_modelling:
            self.init_clm_head()

    def _init_default_lm_head(self):
        """Used as default for MLM and CLM"""
        self.dense = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.layer_norm = nn.LayerNorm(self.config.hidden_size, eps=getattr(self.config, "layer_norm_eps", 1e-12))
        self.decoder = nn.Linear(self.config.hidden_size, self.config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(self.config.vocab_size))
        self.decoder.bias = self.bias

    def _default_lm_logits(self, embeddings):
        """Used as default for MLM and CLM"""
        features = embeddings[0]  # up to this point it is still the dict output of roberta etc.

        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        return self.decoder(x)

    # MLM functions - default is basically roberta

    def init_mlm_head(self):
        """Creates a masked language modelling head, that is embeddings -> logits over vocab for masked tokens.
        Requires care in naming to load pretrained vars."""
        self._init_default_lm_head()

    def get_mlm_logits(self, embeddings):
        return self._default_lm_logits(embeddings)

    def mlm_loss(self, prediction_scores, labels):
        loss_fct = CrossEntropyLoss()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if torch.isnan(masked_lm_loss) and (labels == -100).all():
            masked_lm_loss = prediction_scores.new_zeros([], requires_grad=True)  # 0.0 with matching type and device
        return masked_lm_loss

    def head_output_embeddings(self):
        if self.head_config.masked_language_modelling or self.head_config.causal_language_modelling:
            return self.decoder  # used in MultiTaskModel#get_output_embeddings

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        if self.head_config.masked_language_modelling or self.head_config.causal_language_modelling:
            self.bias = self.decoder.bias

    # Masked Token Detection functions, based on Electra

    def init_mtd_head(self):
        """Creates a masked token detection head. Not common in pretrained models, so shared across them."""
        self.discriminator_predictions = MTDPredictions(self.config)

    def get_mtd_logits(self, embeddings):
        features = embeddings[0]  # up to this point it is still the dict output of roberta etc.
        return self.discriminator_predictions(features)

    def mtd_loss(self, logits, labels):
        predict_mask = labels != IGNORE_INDEX
        masked_logits = logits[predict_mask]
        if self.head_config.mtd_pos_weight is not None:
            pos_weight = torch.full_like(masked_logits, self.head_config.mtd_pos_weight)
        else:
            pos_weight = None
        loss_fct = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = loss_fct(masked_logits, labels[predict_mask].float())
        return loss

    # Causal language modelling functions. Again, defaults taken from Roberta

    def init_clm_head(self):
        """Creates a causal language modelling head. By default this has the same structure as an MLM head"""
        self._init_default_lm_head()

    def get_clm_logits(self, embeddings):
        return self._default_lm_logits(embeddings)

    def clm_loss(self, logits, labels):
        # we are doing next-token prediction; shift prediction scores and input ids by one
        shifted_prediction_scores = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        return loss

    # Head implementation

    def forward(self, **kwargs):
        input_embedding = kwargs[f"{self.head_config.input_prefix}{INPUT_EMBEDDING_VAR}"]
        all_logits = []
        all_losses = []

        if self.head_config.causal_language_modelling:
            token_labels = kwargs.get(CLM_LABELS_VAR)
            clm_logits = self.get_clm_logits(input_embedding)
            if token_labels is not None:
                clm_loss = self.clm_loss(clm_logits, token_labels)
                all_losses.append(clm_loss)
                all_logits.append(clm_logits)
        else:
            if self.head_config.masked_language_modelling:
                mlm_labels = kwargs.get(MLM_LABELS_VAR)
                if mlm_labels is not None:
                    prediction_scores = self.get_mlm_logits(input_embedding)
                    mlm_loss = self.mlm_loss(prediction_scores, mlm_labels)
                    all_losses.append(mlm_loss)
                    all_logits.append(prediction_scores)

            if self.head_config.masked_token_detection:
                token_labels = kwargs.get(MTD_LABELS_VAR)
                token_logits = self.get_mtd_logits(input_embedding)
                if token_labels is not None:
                    mtd_loss = self.mtd_loss(token_logits, token_labels)
                    all_losses.append(mtd_loss)
                    all_logits.append(token_logits)

        return MaskedLMOutput(
            loss=sum(all_losses) if all_losses else None,
            logits=all_logits[0] if len(all_logits) == 1 and all_logits[0] is not None else None,
        )


class RobertaLMHead(BaseLMHead):
    """Roberta Head for masked language modeling. Used as the default."""

    pass


class BaseBertLMHead(BaseLMHead):
    def init_mlm_head(self):
        self.predictions = self.HEAD_CLASS(self.config)

    def get_mlm_logits(self, embeddings):
        features = embeddings[0]  # up to this point it is still the dict output of roberta etc.
        return self.predictions(features)

    def head_output_embeddings(self):
        if self.head_config.masked_language_modelling:
            return self.predictions.decoder  # used in MultiTaskModel#get_output_embeddings
        else:
            return super().head_output_embeddings()

    def _tie_weights(self):
        pass


class BertLMHead(BaseBertLMHead):
    HEAD_CLASS = BertLMPredictionHead


class DebertaLMHead(BaseBertLMHead):
    HEAD_CLASS = DebertaLMPredictionHead


class DebertaV2LMHead(BaseBertLMHead):
    _rename_keys = [  # loads microsoft models mlm head
        ("lm_predictions.lm_head.LayerNorm.bias", "cls.predictions.transform.LayerNorm.bias"),
        ("lm_predictions.lm_head.LayerNorm.weight", "cls.predictions.transform.LayerNorm.weight"),
        ("lm_predictions.lm_head.dense.bias", "cls.predictions.transform.dense.bias"),
        ("lm_predictions.lm_head.dense.weight", "cls.predictions.transform.dense.weight"),
        ("lm_predictions.lm_head.bias", "cls.predictions.bias"),
    ]

    HEAD_CLASS = DebertaV2LMPredictionHead


class DistilBertLMHead(BaseLMHead):  # from DistilBertForMaskedLM, which sticks everything in the base model
    _rename_keys = [
        (f"{v}.{wb}", f"mlm_head.{v}.{wb}")
        for v in ["vocab_transform", "vocab_layer_norm", "vocab_projector"]
        for wb in ["weight", "bias"]
    ]

    def init_mlm_head(self):
        config = self.config
        self.activation = get_activation(config.activation)
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

    def get_mlm_logits(self, embeddings):
        hidden_states = embeddings[0]  # up to this point it is still the dict output of roberta etc.
        prediction_logits = self.vocab_transform(hidden_states)  # (bs, seq_length, dim)
        prediction_logits = self.activation(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_layer_norm(prediction_logits)  # (bs, seq_length, dim)
        prediction_logits = self.vocab_projector(prediction_logits)  # (bs, seq_length, vocab_size)
        return prediction_logits

    def head_output_embeddings(self):
        if self.head_config.masked_language_modelling:
            return self.vocab_projector
        else:
            return super().head_output_embeddings()

    def _tie_weights(self):
        pass


class ElectraLMHead(BaseLMHead):  # from ElectraGeneratorPredictions
    _rename_keys = [
        ("generator_lm_head.weight", "lm_head.generator_lm_head.weight"),
        ("generator_lm_head.bias", "lm_head.generator_lm_head.bias"),
        ("generator_predictions.LayerNorm.bias", "lm_head.LayerNorm.bias"),
        ("generator_predictions.LayerNorm.weight", "lm_head.LayerNorm.weight"),
        ("generator_predictions.dense.bias", "lm_head.dense.bias"),
        ("generator_predictions.dense.weight", "lm_head.dense.weight"),
        ("discriminator_predictions.dense.bias", "lm_head.discriminator_predictions.dense.bias"),
        ("discriminator_predictions.dense.weight", "lm_head.discriminator_predictions.dense.weight"),
        ("discriminator_predictions.dense_prediction.bias", "lm_head.discriminator_predictions.dense_prediction.bias"),
        (
            "discriminator_predictions.dense_prediction.weight",
            "lm_head.discriminator_predictions.dense_prediction.weight",
        ),
    ]

    def init_mlm_head(self):
        config = self.config
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)

    def get_mlm_logits(self, embeddings):
        generator_hidden_states = embeddings[0]  # up to this point it is still the dict output
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = get_activation("gelu")(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        logits = self.generator_lm_head(hidden_states)
        return logits

    def head_output_embeddings(self):
        if self.head_config.masked_language_modelling:
            return self.generator_lm_head  # used in MultiTaskModel#get_output_embeddings
        else:
            return super().head_output_embeddings()

    def _tie_weights(self):
        pass


class BaseGPTLMHead(BaseLMHead):  # GPT style models just have a simple projection back to the vocabulary
    def init_clm_head(self):
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def get_clm_logits(self, embeddings):
        features = embeddings[0]
        return self.lm_head(features)

    def head_output_embeddings(self):  # used in MultiTaskModel#get_output_embeddings
        if self.head_config.causal_language_modelling:
            return self.lm_head
        else:
            return super().head_output_embeddings()

    def _tie_weights(self):  # no bias
        pass


class LMHeadConfig(HeadConfig):
    """
    General Masked Language Modelling / Masked Token Detection / Causal language modelling head.
    In general, this has a few different implementations that should load most common options for pre-trained heads, and default to a new one otherwise.
    Expects text -> input_ids in the dataset, and the default AutoCollator in MultitaskTrainer will take care of the rest.

    Args:
        separate_embedding:
          * When False (default), the collator will overwrite input_ids and other heads will use the model embeddings from the masked input.
          * When True, the base model will be called separately (using the prefix `lm_`), and other heads will use non-masked results unless explicitly told to use this prefix (slower, but more accurate finetuning).
        masked_language_modelling: turn on masked language modelling, where the model predicts the original tokens replaced with [MASK] or another token. Default choice.
        masked_token_detection: turn on electra-style masked token detection, where the model predicts which tokens were replaced with another. See AutoCollator.generate_replacement_tokens for notes on how they are replaced.
        causal_language_modelling: turn on GPT-style causal language modelling, where the model predicts the next token based on previous ones. Should be used with model.config.is_decoder = True and a compatible model.
        mlm_probability: Percentage of tokens to be replaced.
        mask_probability: Of the tokens chosen with mlm_probability, which fraction should be masked. Default 80% without masked_token_detection, 0% otherwise.
        generated_token_probability: Of the tokens chosen with mlm_probability, but not masked, which ones should be replaced. Default 50% without masked_token_detection (corresponding to the default 80/10/10 split in BERT), 100% otherwise.
        predict_all_tokens: Calculate loss over non-masked tokens as well in MLM
        mtd_pos_weight: weight for the positive entries in BCEWithLogitsLoss for masked token detection.
        mtd_strategy: strategy for token masking or a callable, see collator for details. Defaults to "token_similarity" for masked_token_detection and "random" for masked_language_modelling.
        attribute: by default automatically determined from model type to ensure pre-trained weights load.
    """

    """Defines what model type has what (attribute, class) for the MLM head"""
    _CONFIG_TO_HEAD_TYPE = {
        ElectraConfig: ("lm_head", ElectraLMHead),
        XLMRobertaConfig: ("lm_head", RobertaLMHead),
        RobertaConfig: ("lm_head", RobertaLMHead),
        DebertaConfig: ("cls", DebertaLMHead),
        DebertaV2Config: ("cls", DebertaV2LMHead),
        DistilBertConfig: ("mlm_head", DistilBertLMHead),  # major renaming, mlm_head from _rename_keys
        BertConfig: ("cls", BertLMHead),
        OPTConfig: ("lm_head", BaseGPTLMHead),
    }

    def __init__(
        self,
        separate_embedding: bool = False,
        masked_language_modelling: bool = None,
        masked_token_detection: bool = False,
        causal_language_modelling=False,
        mlm_probability: float = 0.15,
        mask_probability: Optional[float] = None,
        generated_token_probability: Optional[float] = None,
        predict_all_tokens: bool = False,
        mtd_pos_weight: float = None,
        mtd_strategy: Optional[Callable] = None,
        **kwargs,
    ):
        # Allow MLM, MTD, CLM and MLM+MTD
        if masked_language_modelling is None and not (masked_token_detection or causal_language_modelling):
            masked_language_modelling = True
        if causal_language_modelling and (masked_language_modelling or masked_token_detection):
            raise ValueError("Can not combine causal_language_modelling with other modes.")
        if not masked_language_modelling and not masked_token_detection and not causal_language_modelling:
            raise ValueError("Can not turn off all modes.")
        if causal_language_modelling and separate_embedding:
            raise ValueError(
                "Since the inputs are unchanged in causal_language_modelling, separate_embedding does not make sense."
            )
        self.separate_embedding = separate_embedding

        self.masked_language_modelling = masked_language_modelling
        self.mlm_probability = mlm_probability
        self.predict_all_tokens = predict_all_tokens

        self.masked_token_detection = masked_token_detection
        self.mtd_pos_weight = mtd_pos_weight

        self.causal_language_modelling = causal_language_modelling

        if mask_probability is None:
            mask_probability = 0.0 if self.masked_token_detection else 0.8
        self.mask_probability = mask_probability
        if generated_token_probability is None:
            generated_token_probability = 1.0 if self.masked_token_detection else 0.5
        self.generated_token_probability = generated_token_probability

        if mtd_strategy is not None:
            self.mtd_strategy = mtd_strategy
        elif self.masked_token_detection:  # MTD default
            self.mtd_strategy = MTD_TOKEN_SIMILARITY
        else:  # MLM default
            self.mtd_strategy = MTD_TOKEN_RANDOM

        kwargs["input_prefix"] = MASKED_PREFIX if separate_embedding else ""
        super().__init__(**kwargs)

    def create_head(self, config):
        for config_cls, (attr, head_cls) in self._CONFIG_TO_HEAD_TYPE.items():
            if type(config) == config_cls:
                self.attribute = self.attribute or attr
                return head_cls(config, self)

        logger.warning(f"No language modelling head registered for {config.__class__.__name__}, using default")
        self.attribute = self.attribute or "lm_head"
        return BaseLMHead(config, self)

    @property
    def labels_var(self):  # used to check if we can compute metrics in trainer
        if self.masked_language_modelling:
            return MLM_LABELS_VAR
        else:
            return MTD_LABELS_VAR

    def input_vars(self) -> Dict[str, Set[str]]:  # set of variables required to run train/inference on head
        infer_vars = super().input_vars()["infer"]
        train_vars = infer_vars.copy()
        if self.masked_language_modelling:
            train_vars.add(MLM_LABELS_VAR)
        if self.masked_token_detection:
            train_vars.add(MTD_LABELS_VAR)
        return {"train": infer_vars | train_vars, "infer": infer_vars}

    def _name(self):  # generates name if not given
        return "mlm"

    def __repr__(self):
        return f"{self.__class__.__name__}(masked_language_modelling={self.masked_language_modelling}, masked_token_detection={self.masked_token_detection})"
