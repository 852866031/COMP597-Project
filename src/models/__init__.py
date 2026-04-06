"""Various models to train.

This module provides various objects that are designed to train machine
learning models. Please refer to each repesctive classes to know how to use
them.

This module provides a factory that can be used to construct a variety of 
models to train using the trainers provided by the trainer module. Please each 
model's directory for model specific documentation.

"""
from typing import Any, Dict, List, Optional, Tuple
import src.auto_discovery as auto_discovery
import src.config as config
import src.trainer as trainer
import torch.utils.data

_MODEL_INIT_FUNCTION_NAME = "init_model"
_MODEL_NAME_VARIABLE_NAME = "model_name"
_MODELS = auto_discovery.register(
    package=__package__,
    path=list(__path__), 
    module_attr_name=_MODEL_INIT_FUNCTION_NAME, 
    name_override_attr_name=_MODEL_NAME_VARIABLE_NAME
)

def model_factory(conf : config.Config, dataset : torch.utils.data.Dataset) -> Tuple[trainer.Trainer, Optional[Dict[str, Any]]]:
    global _MODELS
    init_func = _MODELS.get(conf.model, None)
    if init_func is None:
        raise Exception(f"Unknown model {conf.model}")
    return init_func(conf, dataset)

def get_available_models() -> List[str]:
    global _MODELS
    return [m for m in _MODELS.keys()]
