from typing import List
import src.auto_discovery as auto_discovery
import src.config as config
import torch.utils.data

_DATA_LOAD_FUNCTION_NAME="load_data"
_DATA_LOAD_VARIABLE_NAME="data_load_name"
_DATA_LOADS = auto_discovery.register(
    package=__package__,
    path=list(__path__),
    module_attr_name=_DATA_LOAD_FUNCTION_NAME,
    name_override_attr_name=_DATA_LOAD_VARIABLE_NAME,
)

def load_data(conf : config.Config) -> torch.utils.data.Dataset:
    global _DATA_LOADS
    load_fn = _DATA_LOADS.get(conf.data, None)
    if load_fn is None:
        raise Exception(f"Unknown data load function '{conf.data}'")
    return load_fn(conf)

def get_available_data_load_functions() -> List[str]:
    global _DATA_LOADS
    return [name for name in _DATA_LOADS.keys()]
