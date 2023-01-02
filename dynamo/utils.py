from typing import Any, Iterable
from collections.abc import Mapping
from numbers import Number
from dynamo.typing import DefaultTypes


def is_instance(obj: Any, class_: Any, var_name: str = None) -> bool:
    """
    Checks if object is from a specified class. If not, raises an error.

    Parameters
    ----------
    obj : Any
        Object to be tested
    class_ : Any
        Class to use for testing

    Returns
    -------
    bool
        It is always True since the function raises an error
        when obj is not an instance of class_

    Raises
    ------
    TypeError
        Error raised when obj is not an instance of class_
    """
    if isinstance(obj, class_):
        return True
    else:
        if var_name:
            raise TypeError(f"{var_name} can't be {type(obj)}")
        else:
            raise TypeError(f"Object is {type(obj)}")


def to_default_type(obj: Any) -> DefaultTypes:
    if isinstance(obj, Number):
        casted = float(obj)
    if isinstance(obj, Mapping):
        casted = {key: to_default_type(value) for key, value in obj.items()}
    elif isinstance(obj, Iterable):
        casted = [to_default_type(value) for value in obj]
    else:
        raise ValueError(f"{type(obj)} cannot be casted to default type.")

    return casted
