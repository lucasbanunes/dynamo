from importlib import import_module
from typing import Tuple, List, Any, Dict
from dynamo.typing import DefaultTypes


def is_config_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False

    try:
        obj["constructor"]
        obj["kwargs"]
        obj["args"]
    except KeyError:
        return False

    return True


def split_import_name(import_name: str) -> Tuple[str, str]:
    splitted = import_name.split(".")
    name = ".".join(splitted[:-1])
    attribute = splitted[-1]
    return name, attribute


def parse_config_args(config_args: List[DefaultTypes],
                      parse_inner_objs: bool) -> List[Any]:
    args = list()
    for iarg in config_args:
        if is_config_dict(iarg):
            parsed_obj = parse_config_dict(iarg, parse_inner_objs)
            args.append(parsed_obj)
        else:
            args.append(iarg)
    return args


def parse_config_kwargs(config_kwargs: Dict[str, DefaultTypes],
                        parse_inner_objs: bool) -> Dict[str, Any]:
    kwargs = dict()
    for key, iarg in config_kwargs.items():
        if is_config_dict(iarg):
            parsed_obj = parse_config_dict(iarg, parse_inner_objs)
            kwargs[key] = parsed_obj
        else:
            kwargs[key] = iarg
    return kwargs


def parse_config_dict(config_dict: Dict[str, Any],
                      parse_inner_objs: bool
                      ) -> Any:
    constructor_name = config_dict["constructor"]
    name, attribute = split_import_name(constructor_name)
    package = import_module(name)
    constructor = getattr(package, attribute)

    if parse_inner_objs:
        args = parse_config_args(config_dict["args"], parse_inner_objs)
    else:
        args = config_dict["args"]

    if parse_inner_objs:
        kwargs = parse_config_kwargs(config_dict["kwargs"], parse_inner_objs)
    else:
        kwargs = config_dict["kwargs"]

    instance = constructor(*args, **kwargs)
    return instance
