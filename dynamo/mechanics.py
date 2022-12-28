import numpy as np
from numbers import Number


def get_2d_rot_matrix(theta: Number) -> np.ndarray:
    """
    Returns a 2-d numpy array representing a 2d rotation matrix

    Parameters
    ----------
    theta : Number
        Angle of rotation in radians

    Returns
    -------
    np.ndarray
        Rotation matrix
    """
    c = np.cos(theta)
    s = np.sin(theta)
    r = np.array(
        [[c, -s],
         [s, c]]
    )
    return r


def get_2d_rot_inv_matrix(theta: Number) -> np.ndarray:
    """
    Returns a 2-d numpy array representing an inverse 2d rotation matrix

    Parameters
    ----------
    theta : Number
        Angle of rotation in radians

    Returns
    -------
    np.ndarray
        Inverse rotation matrix
    """
    c = np.cos(theta)
    s = np.sin(theta)
    r_inv = np.array(
        [[c, s],
         [-s, c]]
    )
    return r_inv
