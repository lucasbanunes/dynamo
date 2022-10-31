from numbers import Number

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