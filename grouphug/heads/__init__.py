from typing import Dict

from grouphug.heads.classification import ClassificationHead, ClassificationHeadConfig
from grouphug.heads.lm import LMHeadConfig


# not a class method to avoid circular import hell
def head_config_from_dict(data: Dict):
    hcls = globals().get(data["class"])
    if hcls is None:
        raise NotImplementedError(f"Could not find head config class {data['class']}")
    return hcls(**data["args"])
