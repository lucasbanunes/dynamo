import sympy
from sympy.utilities.lambdify import lambdify
from numbers import Number
from typing import List, Any, Dict
from dynamo.base import Bunch


class TimeSignal(Bunch):
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

    def __init__(self, n_derivatives: int = 0, **kwargs):
        if not kwargs:
            raise ValueError("The signal must have at least one function")

        self.n_derivatives = int(n_derivatives)

        funcs = dict()
        for func_name, func_str in kwargs.items():
            sym_func = sympy.sympify(func_str)
            funcs[func_name] = sym_func
            self[f"sym_{func_name}"] = str(sym_func)
            last_diff = sym_func
            for diff_order in range(self.n_derivatives):
                d_str = (diff_order+1)*"d"
                diff_name = f"{d_str}{func_name}"
                if "Heaviside" in func_str:
                    sym_diff = sympy.diff("t-t", "t")
                else:
                    sym_diff = sympy.diff(last_diff, "t")
                funcs[diff_name] = sym_diff
                self[f"sym_{diff_name}"] = str(sym_diff)
                last_diff = sym_diff

        for func_name, sym_func in funcs.items():
            lambda_func = lambdify("t", sym_func, modules="numpy")
            self[func_name] = lambda_func

        self.n_funcs = len(funcs)

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
