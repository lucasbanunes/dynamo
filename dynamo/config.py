from importlib import import_module
from typing import Tuple, List, Any, Dict
from dynamo.typing import DefaultTypes


def is_config_dict(obj: Any) -> bool:
    """
    Checks if an instance is a config dict.

    Parameters
    ----------
    obj : Any
        Instance to be tested

    Returns
    -------
    bool
        True if it is a config dict
    """
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
    """
    Splits an import name considering that the last name
    is an attribute defined inside the package.

    Parameters
    ----------
    import_name : str
        Import path

    Returns
    -------
    name: str
        The full package name
    attribute: str
        The attribute name
    """
    splitted = import_name.split(".")
    name = ".".join(splitted[:-1])
    attribute = splitted[-1]
    return name, attribute


def parse_config_args(config_args: List[DefaultTypes],
                      parse_inner_objs: bool) -> List[Any]:
    """
    Parses the args key from a config dict

    Parameters
    ----------
    config_args : List[DefaultTypes]
        args key value
    parse_inner_objs : bool
        If true, tries to parse dicts inside args
        as other configs

    Returns
    -------
    List[Any]
        Parsed args
    """
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
    """
    Parses the kwargs key from a config dict

    Parameters
    ----------
    config_kwargs : List[DefaultTypes]
        kwargs key value
    parse_inner_objs : bool
        If true, tries to parse dicts inside args
        as other configs

    Returns
    -------
    Dict[str, Any]
        Parsed kwargs
    """
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
    """
    Parses a config dict

    Parameters
    ----------
    config_dict : Dict[str, Any]
        Python dictionary containg the configuration
        for instatiation of an object.
    parse_inner_objs : bool
        If true, tries to parse dicts inside config_dict
        as other configs

    Returns
    -------
    Any
        Instance of the object described by config 
    """
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
