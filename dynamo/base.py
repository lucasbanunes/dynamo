from collections.abc import MutableMapping
from abc import ABC, abstractmethod
from typing import Any
from numbers import Number
import pandas as pd


class Bunch(MutableMapping):
    """Container object exposing keys as attributes.
    Inherits from collections.abs.MutableMapping"""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any):
        setattr(self, key, value)

    def __delitem__(self, key: str):
        delattr(self, key)

    def __repr__(self) -> str:
        name_values = [
            f"{name}={value}"
            for name, value in self.items()
        ]
        repr_str = ", ".join(name_values)
        return repr_str

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def to_frame(self):
        df = pd.DataFrame()
        for key, value in self.items():
            df[key] = value
        return df


class DynamicSystem(ABC):
    """
    Class that represents a Dynamic System.
    A dynamic system is an object that implements the following methods:
    - dx: For computing the system update rule
    - output: For computing the system outputs
    """

    @abstractmethod
    def dx(self, t: Number, data: Bunch) -> Bunch:
        return data

    @abstractmethod
    def output(self, t: Number, data: Bunch) -> Bunch:
        return data


class Controller(ABC):
    """
    The Controller class is an abstract class that represents
    a control logic that receives one or several inputs and returns an output
    to a plant.
    """
    @abstractmethod
    def output(self, t: Number, data: Bunch) -> Bunch:
        return data
