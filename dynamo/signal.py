import sympy
from sympy.utilities.lambdify import lambdify
from numbers import Number
from typing import List, Any, Dict
from dynamo.base import Bunch


class TimeSignal(Bunch):
    """
    Object that implements multiple functions that represent one signal.
    This is specially useful or reference signals that may have multiple
    components.

    This class can receive a multiple number of functions through the
    kwargs argument that can be later be acessed as an attribute.
    All the functions must be declared as a string that will latter
    be casted to a lambda function with sympy lambdify function.
    Therefore, any function supported by sympy.sympify can be
    defined to this object.

    From python 3.7 ownards dicts are ordered so a later variable
    can reference a previous one for definition. This is specially
    useful for algebric differentiation and integration.

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

        sympy_locals = dict(t=sympy.Symbol("t", real=True))
        for func_name, func_str in kwargs.items():
            sym_func = sympy.sympify(func_str, locals=sympy_locals)
            lambda_func = lambdify("t", sym_func, modules="numpy")
            self[func_name] = lambda_func
            self[f"sym_{func_name}"] = str(sym_func)
            sympy_locals[func_name] = sym_func

        self.n_funcs = len(sympy_locals) - 1  # removes t symbol

    def __call__(self, t: Number) -> List[Any]:
        if self.n_funcs > 1:
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
