from importlib import import_module
from typing import Mapping, Any

CONFIG_KEYS = ["class", "args", "kwargs"]


def is_obj_config(obj_config: Mapping[str, Any]) -> bool:
    obj_config["class"]
    obj_config["args"]
    obj_config["kwargs"]
    return True


def get_obj_from_config(obj_config: Mapping[str, Any]) -> Any:
    class_ = import_module(obj_config["class"])
    obj = class_(*obj_config["args"], **obj_config["kwargs"])
    return obj
