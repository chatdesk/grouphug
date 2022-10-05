from typing import Dict

import torch
from transformers import default_data_collator

from grouphug.config import (
    CLM_LABELS_VAR,
    IGNORE_INDEX,
    INPUT_IDS_VAR,
    MASKED_INPUT_IDS_VAR,
    MLM_LABELS_VAR,
    MTD_LABELS_VAR,
    MTD_TOKEN_RANDOM,
    MTD_TOKEN_SIMILARITY,
    logger,
)
from grouphug.model import _BaseMultiTaskModel


class AutoCollator:
    def __init__(self, model: _BaseMultiTaskModel, tokenizer):
        """..."""
        self.model = model
        self.mlm_head = model.get_mlm_head()
        self.mlm_active = self.mlm_head is not None
        if self.mlm_active:
            if tokenizer is None:
                raise ValueError("Pass a tokenizer to MultiTaskTrainer for masked language modelling")
            if self.mlm_head.causal_language_modelling:
                if not model.config.is_decoder:
                    logger.warning("Model not set as is_decoder, which is usual with causal_language_modelling")
            elif self.mlm_head.mask_probability > 0.0 and tokenizer.mask_token is None:
                raise AttributeError(f"Tokenizer has no mask token, which is necessary when mask_probability > 0.0")

        self.input_prefixes = model.input_prefixes()
        self.model_vars = model.vars()
        self.tokenizer = tokenizer

    def update_mlm_active(self):  # cached check
        self.mlm_active = self.mlm_head in self.model.get_active_heads()
        return self.mlm_active

    def _maybe_pad(self, columns, return_tensors) -> Dict[str, torch.Tensor]:
        """Determines if the inputs still need to be padded and pads if needed"""
        if INPUT_IDS_VAR in columns[0] and not all(
            len(x[INPUT_IDS_VAR]) == len(columns[0][INPUT_IDS_VAR]) for x in columns
        ):
            if not self.tokenizer:
                raise ValueError(
                    "Inputs are of different lengths, and no tokenizer passed to trainer to dynamically pad."
                )
            if self.input_prefixes != {""}:  # Too messy to support for now.
                raise ValueError(
                    "Inputs are of different lengths, and multiple text inputs expected. AutoCollator does not support this, pad to max length in formatting instead."
                )
            if self.tokenizer.pad_token is None:
                logger.warning("Setting tokenizer 'pad_token' to 'eos_token' as we really need to pad now.")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.config.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

                # pad collates as well. DOES NOT TRUNCATE!
            return self.tokenizer.pad(columns, return_tensors=return_tensors)
        else:
            return default_data_collator(columns, return_tensors=return_tensors)

    def generate_replacement_tokens(self, token_ids: torch.Tensor, replace_indices: torch.Tensor, strategy: str):
        """Generates replacement tokens for MLM/MTD.
        Default just takes random token ids, as used in BERT.
        This appears to be 'too easy' for MTD, so consider overwriting this with a generator based version."""
        tokens_to_replace = token_ids[replace_indices]
        if strategy == MTD_TOKEN_RANDOM:
            return torch.randint(len(self.tokenizer), tokens_to_replace.shape, dtype=torch.long)
        elif strategy == MTD_TOKEN_SIMILARITY:  # cosine distance ish
            unique_ids_to_replace, ixs = torch.unique(tokens_to_replace, return_inverse=True)
            similarity = self.model.token_similarity(unique_ids_to_replace)
            similarity -= torch.mean(similarity, dim=1, keepdim=True)
            similarity /= torch.std(similarity, dim=1, keepdim=True)
            # sample ps
            similarity[similarity < 3] = 0  # only take top 0.2% of tokens
            for i in range(similarity.size(0)):  # TODO: scatter_ ?
                similarity[i, unique_ids_to_replace[i]] = 1e-6  # ensures sum is never 0
            replacement_tokens = torch.multinomial(similarity[ixs, :], 1)[:, 0]
            return replacement_tokens
        else:
            raise ValueError("Invalid strategy")

    def torch_mask_tokens(self, original_inputs: torch.Tensor, special_tokens_mask: torch.Tensor):
        """
        Prepare masked tokens inputs/labels
        Default for only masked language modeling: 80% MASK, 10% replaced, 10% original.
        Default for masked token detection: 100% replaced.
        """
        masked_inputs = original_inputs.clone()
        labels = original_inputs.clone()

        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_head.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)  # mlm_probability in non-special tokens
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # (MLM:80%, MTD:0%) of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = None
        if self.mlm_head.mask_probability > 0.0:  # may not even have a mask token, so avoid this
            indices_replaced = (
                torch.bernoulli(torch.full(labels.shape, self.mlm_head.mask_probability)).bool() & masked_indices
            )
            masked_inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # (MLM: 10%, MTD: 100%) of the time, we replace masked input tokens with random word
        indices_random = masked_indices
        if self.mlm_head.generated_token_probability < 1.0:
            indices_random &= torch.bernoulli(
                torch.full(labels.shape, self.mlm_head.generated_token_probability)
            ).bool()
        if indices_replaced is not None:
            indices_random &= ~indices_replaced
        masked_inputs[indices_random] = self.generate_replacement_tokens(
            masked_inputs, indices_random, self.mlm_head.mtd_strategy
        )

        # The rest of the time (MLM: 10%, MTD: 0%) we keep the masked input tokens unchanged
        return masked_inputs, labels, masked_indices

    def collate_for_mlm(self, batch) -> Dict:
        original_inputs = batch[INPUT_IDS_VAR]
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if special_tokens_mask is None:
            special_tokens_mask = torch.tensor(
                [
                    self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
                    for val in original_inputs.tolist()
                ],
                dtype=torch.bool,
            )
        else:
            special_tokens_mask = special_tokens_mask.bool()

        masked_inputs, mlm_labels, masked_indices = self.torch_mask_tokens(original_inputs, special_tokens_mask)

        if self.mlm_head.masked_language_modelling:
            if not self.mlm_head.predict_all_tokens:
                mlm_labels.masked_fill_(~masked_indices, IGNORE_INDEX)  # only compute loss on masked tokens
            else:
                mlm_labels.masked_fill_(special_tokens_mask, IGNORE_INDEX)
            batch[MLM_LABELS_VAR] = mlm_labels

        if self.mlm_head.masked_token_detection:
            batch[MTD_LABELS_VAR] = masked_indices.long()
            batch[MTD_LABELS_VAR].masked_fill_(special_tokens_mask, IGNORE_INDEX)

        if self.mlm_head.separate_embedding:
            batch[MASKED_INPUT_IDS_VAR] = masked_inputs
        else:
            batch[INPUT_IDS_VAR] = masked_inputs

        return batch

    def __call__(self, features, return_tensors=None):
        return_tensors = return_tensors or "pt"
        features_to_collate = [{k: v for k, v in f.items() if k in self.model_vars} for f in features]
        collated_features = self._maybe_pad(features_to_collate, return_tensors)
        if self.mlm_active:
            if not self.mlm_head.causal_language_modelling:
                collated_features = self.collate_for_mlm(collated_features)
            else:  # basically just signals mlm_active to the head
                collated_features[CLM_LABELS_VAR] = collated_features[INPUT_IDS_VAR]
        return collated_features
