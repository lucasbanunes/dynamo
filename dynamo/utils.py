from typing import Any, Iterable, Tuple
from collections.abc import Mapping
from numpy.typing import ArrayLike
import numpy as np


class Bunch(Mapping):
    """Container object exposing keys as attributes."""

    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def __getitem__(self, key: str):
        return self.__dict__[key]

    def __repr__(self) -> str:
        name_values = [
            f"{name}={value}"
            for name, value in self.items()
        ]
        repr_str = ",".join(name_values)
        return repr_str

    def len(self) -> int:
        return len(self.__dict__)

    def items(self) -> Iterable[Tuple[str, Any]]:
        return self.__dict__.items()

    def keys(self) -> Iterable[str]:
        return self.__dict__.keys()

    def values(self) -> Iterable[Any]:
        return self.__dict__.values()


class VectorBunch(Bunch):
    """Bunch object with a numpy array x as an attribute and its
    individual values accessible via atributes as such:
        self.x{i+1} = x[i]"""

    def __init__(self, x: ArrayLike):
        x = np.array(x).flatten()
        kwargs = {f'x{i+1}': x[i] for i in range(len(x))}
        super().__init__(x=x, **kwargs)


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
