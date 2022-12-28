import numpy as np
from numpy.typing import ArrayLike
from abc import ABC, abstractmethod
from numbers import Number


class DynamicSystem(ABC):
    """
    Class that represents a Dynamic System.
    A dynamic system is represented here as a system that recieves an input (u)
    and changes its internal states based on some update rule (dx).
    For generality, the system can also vary as a funcion of time (t).
    dx is a function that defines its dynamic returning the dervative
    of its states (x):
        self.dx(t, x, u) -> y
    of its states x.
    It also includes a function that defines its output (y):
        self.output(t, x, u) -> y
    When inherting this function the init function can be whatever the
    user wants, but they must define a dx and an output method.
    """

    def __init__(self):
        pass

    @abstractmethod
    def dx(self, t, x, u):
        raise NotImplementedError

    @abstractmethod
    def output(self, t, x, u):
        raise NotImplementedError


class TimeInvariantLinearSystem(DynamicSystem):
    """
    A time invariant linear system is a dynamic System that can be defined as:
        dx = Ax + Bu
        y = Cx + Du
    where u,x,y are column vectors and A, B, C, D are matrices.
    Parameters (Attributes)
    ----------
    A : ArrayLike[np.floating]
        A matrix
    B : ArrayLike[np.floating]
        B matrix
    C : ArrayLike[np.floating]
        C matrix
    D : ArrayLike[np.floating]
        D matrix
    x0 : ArrayLike[np.floating]
        x0 initial states
    """

    def __init__(self, A: ArrayLike[np.floating],
                 B: ArrayLike[np.floating],
                 C: ArrayLike[np.floating],
                 D: ArrayLike[np.floating],
                 x0: ArrayLike[np.floating]):

        self.A = np.array(A, np.float64)
        self.B = np.array(B, np.float64)
        self.C = np.array(C, np.float64)
        self.D = np.array(D, np.float64)
        self.x0 = np.array(x0, np.float64)

    def dx(self,
           t: Number,
           x: ArrayLike[np.floating],
           u: ArrayLike[np.floating]) -> np.ndarray:
        """
        Computes dx for a Time Invariant Linear System.

        Parameters
        ----------
        t : Number
            Current time. It is ignored since the system is time invariant.
        x : ArrayLike[np.floating]
            State column vector
        u : ArrayLike[np.floating]
            Input column vector

        Returns
        -------
        np.ndarray
            dx/dt column vector
        """
        dx = (np.dot(self.A, x)+np.dot(self.B, u))
        return dx

    def output(self,
               t: Number,
               x: ArrayLike[np.floating],
               u: ArrayLike[np.floating]) -> np.ndarray:
        """
        Computes the output for a Time Invariant Linear System.

        Parameters
        ----------
        t : Number
            Current time. It is ignored since the system is time invariant.
        x : ArrayLike[np.floating]
            State column vector
        u : ArrayLike[np.floating]
            Input column vector

        Returns
        -------
        np.ndarray
            y column vector
        """
        out = (np.dot(self.C, x)+np.dot(self.D, u))
        return out
