from sympy.utilities.lambdify import lambdify
from numbers import Number
from typing import List, Any, Dict


class TimeSignal(object):
    """
    Object that implements multiple functions that represent one signal.
    This is specially useful or reference signals that must have their
    elementary functions and its derivatives called for control logics.

    This class can receive a multiple number of functions through the
    kwargs argument that can be later be acessed as an attribute.
    All the functions must be declared as a string that will latter
    be casted to a lambda function with sympy lambdify function.

    Since its a time signal all functions are supposed to receive
    one parameter only, the time (t).

    This class is a Callable. If there is only one function in this
    signal, the call function returns its value directly. If there are
    multiple functions in this signal, the call fucntion returns a dict
    with key value pairs as {func_name: func_value}
    """

    def __init__(self, **kwargs):
        if not kwargs:
            raise ValueError("The signal must have at least one function")

        self.__func_order = list()
        for func_name, func_str in kwargs.items():
            self.__dict__[func_name] = lambdify("t", func_str)
            self.__func_order.append(func_name)
        self.__one_output = len(self.__func_order) > 1

    def __call__(self, t: Number) -> List[Any]:
        if self.__one_output:
            return self.__one_output_call(t)
        else:
            return self.__multiple_output_call(t)

    def __one_output_call(self, t: Number) -> Number:
        res = self.__dict__[self.__func_order[0]](t)
        return res

    def __multiple_output_call(self, t: Number) -> Dict[str, Number]:
        res = dict()
        for func_name in self.__func_order:
            res[func_name] = self.__dict__[func_name](t)
        return res
