from importlib import import_module
from typing import Tuple, List, Any, Dict
from dynamo.typing import DefaultTypes
import json


def split_import_name(import_name: str) -> Tuple[str, str]:
    splitted = import_name.split(".")
    package_name = splitted[0]
    module_name = ".".join(splitted[1:])
    return package_name, module_name


def parse_config_args(config_args: List[DefaultTypes]) -> List[Any]:
    args = list()
    for iarg in config_args:
        if (type(iarg) is dict):
            parsed_obj = parse_config_dict(iarg)
            args.append(parsed_obj)
        else:
            args.append(iarg)
    return args


def parse_config_kwargs(config_kwargs: Dict[str, DefaultTypes]) -> List[Any]:
    kwargs = list()
    for key, iarg in config_kwargs.items():
        if (type(iarg) is dict):
            parsed_obj = parse_config_dict(iarg)
            kwargs[key] = parsed_obj
        else:
            kwargs[key] = iarg
    return iarg


def parse_config_dict(config_str: str, parse_inner_dicts: bool) -> Any:
    config = json.loads(config_str)
    constructor_name = config["constructor"]
    package_name, module_name = split_import_name(constructor_name)
    constructor = import_module(module_name, package_name)

    if parse_inner_dicts:
        args = parse_config_args(config["args"])
    else:
        args = config["args"]

    if parse_inner_dicts:
        kwargs = parse_config_kwargs(config["kwargs"])
    else:
        kwargs = config["kwargs"]

    instance = constructor(*args, **kwargs)
    return instance
