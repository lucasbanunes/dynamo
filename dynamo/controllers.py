from abc import abstractmethod, ABC
from .models import TimeInvariantLinearSystem
import numpy as np
from numbers import Number
from typing import Any, Union, Mapping, Tuple
import config


class Controller(ABC):
    """
    The Controller class is an abstract class that represents
    a control logic that receives one or several inputs and returns an output
    to a plant.
    """
    @abstractmethod
    def output(self, t: Number, **kwargs) -> np.ndarray:
        """
        Method that computes the controller output for a given t time.

        Parameters
        ----------
        t : Number
            Current time
        **kwargs:
            Any other desired parameters

        Returns
        -------
        np.ndarray
            Controller output
        """
        pass


CtrlLike = Union[Controller, Mapping[str, Any]]


class LinearController(TimeInvariantLinearSystem, Controller):
    """
    LinearController represents the set of controllers that can be
    represented as a TimeInvariantLinearSystem.
    """


class FbLinearizationCtrl(Controller):
    """
    The FbLinearizationCtrl class representes a
    Feedback Linearization control technique, known
    in robotics as Computed Torque method, is a control
    rule that linearizes a non linear plant by feedbacking
    its nonliearities. This way an outter control system sees the
    plant as a linear system.

    Parameters
    ----------
    controller : CtrlLike
        A Controller instance or object config that implements
        the outer controller.

    Atributes
    ----------
    controller : Controller
        A Controller instance or object config that implements
        the outer controller.

    """

    def __init__(self, controller: CtrlLike, **kwargs):
        if isinstance(controller, Controller):
            self.controller = controller
        elif config.is_obj_config(controller):
            self.controller = config.get_obj_from_config(controller)
        else:
            raise TypeError("controller must be a Controller instance"
                            " or a object config"
                            f"Not {type(controller)}")
        for key, value in kwargs.items():
            self.__dict__[key] = value

    def output(self, t: Number, *args, **kwargs) -> Tuple[Any, Any]:
        ctrl_out = self.controller.output(t, *args, **kwargs)
        y = self.linearize(t, ctrl_out, *args, **kwargs)
        return y, ctrl_out

    @abstractmethod
    def linearize(self, t: Number, ctrl_out: Any, *args, **kwargs) -> Any:
        """
        Method that applies the inverse non lineartites from the system to the
        outer controler output in a way that the outer controller sees
        a linear plant.

        Parameters
        ----------
        t : Number
            The control/simulation time
        ctrl_out: Any
            Output from the outer control strategy

        Returns
        -------
        np.ndarray
            The output that applies u to linearized system
        """
        raise NotImplementedError


class PDController(Controller):

    def __init__(self, kp: Number, kd: Number):
        self.kp = np.float64(kp)
        self.kd = np.float(kd)

    def output(self, t: Number, e: Number, de: Number, ddref: Number,
               *args, **kwargs) -> np.float64:
        y = self.kp*e + self.kd*de + ddref
        return y

class PCtrlLowSpeed(Controller):

    def __init__(self, kp: Number, kd: Number):
        self.kp = np.float64(kp)
        self.kd = np.float(kd)
    
    def output(self, t: Number, e: Number, speed: Number,
               *args, **kwargs) -> np.float64:
        y = self.kp*e - self.kd*speed
        return y