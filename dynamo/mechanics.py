import numpy as np
from numbers import Number

def get_2d_rot_matrix(theta: Number) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    r = np.array(
        [[c, -s],
         [s, c]]
    )
    return r

def get_2d_rot_inv_matrix(theta: Number) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    r_inv = np.array(
        [[c, s],
         [-s, c]]
    )
    return r_inv