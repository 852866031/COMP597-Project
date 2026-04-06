from typing import List
from src.config.util.base_config import _BaseConfig
import logging
import src.auto_discovery as auto_discovery

logger = logging.getLogger(__name__)

class ConfigAutoDiscovery(_BaseConfig):

    def __init__(self, package : str, path : List[str], config_class_name : str, name_override_attr_name : str = "", ignore_attr_name : str = "") -> None:
        super().__init__()
        self._registered = auto_discovery.register(
            package=package,
            path=path,
            module_attr_name=config_class_name,
            name_override_attr_name=name_override_attr_name,
            ignore_attr_name=ignore_attr_name
        )
        for name, subconfig_class in self._registered.items():
            setattr(self, name, subconfig_class())
            logger.debug(f"Discovered subconfig named '{name}'.")
