from numbers import Number
from numpy.typing import ArrayLike
import numpy as np

class Bunch(object):

    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def __getitem__(self, key:str):
        return self.__dict__[key]

class VectorBunch(Bunch):

    def __init__(self, x:ArrayLike):
        x=np.array(x).flatten()
        kwargs = {f'x{i}': x[i] for i in range(len(x))}
        super().__init__(x=x, **kwargs)

def is_callable(obj: object):
    if callable(obj):
        return True
    else:
        raise TypeError(f'{obj} must be a callable')

def is_numeric(obj: object):
    if isinstance(obj, Number):
        return True
    else:
        raise TypeError(f'{obj} must be a number')

def is_instance(obj, classinfo):
    if isinstance(obj, classinfo):
        return True
    else:
        raise TypeError(f'{obj} must be {classinfo}')