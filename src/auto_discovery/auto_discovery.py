from types import ModuleType
from typing import Dict, List, Optional
import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)

def _discover_submodules(path : List[str]) -> List[pkgutil.ModuleInfo]:
    submodules = []
    for submodule in pkgutil.iter_modules(path=path):
        submodules.append(submodule)
        logger.debug(f"Found submodule '{submodule.name}' under '{path}'")
    return submodules

def _import_submodule_if_contains_attr(package : str, submodule : pkgutil.ModuleInfo, module_attr_name : str, ignore_attr_name : str, strict_ispkg : bool) -> Optional[ModuleType]:
    if strict_ispkg and not submodule.ispkg:
        logger.debug(f"Ignoring submodule '{submodule.name}' of package '{package}' as it is not a package and strict checking is enabled.")
        return None
    try:
        module = importlib.import_module(name=f".{submodule.name}", package=package)
        if ignore_attr_name != "" and getattr(module, ignore_attr_name, False):
            logger.debug(f"Ignoring submodule '{submodule.name}' of package '{package}' as it contain the ignore attribute '{ignore_attr_name}'")
            return None
        if getattr(module, module_attr_name, None) is None:
            logger.debug(f"Ignoring submodule '{submodule.name}' of package '{package}' as it does not have the '{module_attr_name}' attribute")
            return None
    except Exception:
        logger.exception(f"Failed to import '{submodule.name}'")
        return None
    return module

def _get_registration_name(module : ModuleType, name_override_attr_name : str) -> str:
    default_name = module.__package__.split(".")[-1]
    if name_override_attr_name == "":
        return default_name
    return getattr(module, name_override_attr_name, default_name)

def _register_module(found : Dict[str, object], module : ModuleType, module_attr_name : str, name_override_attr_name : str) -> Dict[str, object]:
    name = _get_registration_name(module, name_override_attr_name)
    found[name] = getattr(module, module_attr_name)
    logger.debug(f"Registered module '{module.__name__}' with key '{name}' for attribute '{module_attr_name}'")
    return found

def register(package : str, path : List[str], module_attr_name : str, name_override_attr_name : str = "", ignore_attr_name : str = "", strict_ispkg : bool = True) -> Dict[str, object]:
    submodules = _discover_submodules(path)
    found = {}
    for submodule in submodules:
        module = _import_submodule_if_contains_attr(package, submodule, module_attr_name, ignore_attr_name, strict_ispkg)
        if module is None:
            continue
        found = _register_module(found, module, module_attr_name, name_override_attr_name)
    return found
