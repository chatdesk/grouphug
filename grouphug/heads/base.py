from abc import ABC
from typing import Any, Dict, Optional, Set

import torch
from torch import nn
from transformers.utils import ModelOutput

from grouphug.config import INPUT_IDS_VAR


class ModelHead(nn.Module):
    _rename_keys = []  # key suffixes to rename when loading


class HeadConfig(ABC):

    """
    Abstract base config which contains all common config options for multi-task model heads
    Args:
        input_prefix: [prefix]text -> [prefix]input_ids -> [prefix]embeddings
        name: If ommitted, a name is generated. Key in the model outputs, among others.
        attribute: variable name to store head in on the model, to ensure pre-trained MLM heads load
        weight: ..
    """

    def __init__(
        self,
        input_prefix: str = "",
        weight: float = 1.0,
        name: Optional[str] = None,
        attribute: Optional[str] = None,
        **_kwargs,  # ignored/for loading
    ):
        self.weight = weight
        self.input_prefix = input_prefix
        self.name = name
        self.attribute = attribute

        self._head = None
        self._required_input_vars = None  # cache for _input_vars

    # Task weight, loss is multiplied by this
    # Can be changed for siamese or other multi-text inputs

    # For internal use
    def input_vars(self) -> Dict[str, Set[str]]:  # set of variables required to run train/inference on head
        req_vars = {self.input_prefix + INPUT_IDS_VAR}
        return {"train": req_vars, "infer": req_vars}

    def input_prefixes(self):  # used in model forward, general setup to allow for siamese models
        return [self.input_prefix]

    def _name(self):  # generates name if not given
        return f"{self._head.__class__.__name__}"

    def create_head(self, config):
        raise NotImplementedError()

    def output_stats(self, output: ModelOutput) -> Dict[str, Any]:
        """Turns head output into a set of richer statistics"""
        if getattr(output, "loss", None) is None:
            return {}
        else:
            return {"loss": output.loss.item()}

    def to_dict(self) -> Dict:
        args = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        for k, v in args.items():
            if isinstance(v, torch.Tensor):
                args[k] = v.detach().cpu().numpy()  # pos_weight
        return {
            "class": str(self.__class__.__name__),
            "args": args,
        }
