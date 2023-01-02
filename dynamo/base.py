from collections.abc import MutableMapping
from abc import ABC, abstractmethod
from typing import Any, Mapping
from numbers import Number
import pandas as pd


class Bunch(MutableMapping):
    """Container object exposing keys as attributes."""

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


class BaseModel(ABC):

    def __init__(self,
                 input_vars: Mapping[str, str],
                 output_vars: Mapping[str, str],
                 **kwargs):
        self.input_vars = input_vars
        self.output_vars = output_vars
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_input_bunch(self, t: Number, data: Bunch) -> Bunch:
        input_bunch = Bunch()
        for var_name, data_name in self.input_vars.items():
            input_bunch[var_name] = data[data_name]
        input_bunch.t = t
        return input_bunch

    def add_renamed_outputs(self, output_bunch: Bunch,
                            data: Bunch
                            ) -> Bunch:
        for var_name, data_name in self.output_kwargs.items():
            data[data_name] = output_bunch[var_name]
        return data


class DynamicSystem(BaseModel):
    """
    Class that represents a Dynamic System.
    A dynamic system is an object that implements the following methods:
    - dx: For computing the system update rule
    - output: For computing the system outputs
    """

    def dx(self, t: Number, data: Bunch) -> Bunch:
        input_bunch = self.get_input_bunch(t, data)
        output_bunch = self.transfer_names_call(input_bunch, self.get_dx)
        out = self.add_renamed_outputs(output_bunch, data)
        return out

    def output(self, t: Number, data: Bunch) -> Bunch:
        input_bunch = self.get_input_bunch(t, data)
        output_bunch = self.transfer_names_call(input_bunch, self.get_output)
        out = self.add_renamed_outputs(output_bunch, data)
        return out

    @abstractmethod
    def get_dx(self, t: Number, input_args: Bunch) -> Any:
        """
        Method that computs the system update rule
        """
        raise NotImplementedError

    @abstractmethod
    def get_output(self, t: Number, input_args: Bunch) -> Any:
        """
        Method that computs the system output
        """
        raise NotImplementedError


class Controller(BaseModel):
    """
    The Controller class is an abstract class that represents
    a control logic that receives one or several inputs and returns an output
    to a plant.
    """
    def output(self, t: Number, data: Bunch) -> Bunch:
        input_bunch = self.get_input_bunch(t, data)
        output_bunch = self.transfer_names_call(input_bunch, self.get_output)
        out = self.add_renamed_outputs(output_bunch, data)
        return out

    @abstractmethod
    def get_output(self, t: Number, **kwargs) -> Bunch:
        """
        Method that computes the controller output for a given t time.
        """
        raise NotImplementedError
