from abc import ABC, abstractmethod
from .models import DynamicSystem, LinearStateSpaceSystem
from .utils import is_callable, is_instance
from .typing import CtrlLike
import numpy as np
from numpy.typing import ArrayLike
from numbers import Number
from typing import Callable, Any, Dict, List

class Controller(DynamicSystem):
    """
    Controller class represents all the contollers made to follow a certain reference.
    It must not be used but inherited. It implements 2 main methods: output and dx. Both of them
    must receive 4 required parameters. Control time, input states, controller states and the
    references to follow. This class inherits the DynamicSystem class.
    """
    def output(self, t: Number, input_x: ArrayLike, 
        x: ArrayLike, refs: ArrayLike, **kwargs) -> np.ndarray:
        """
        Method that computes the controller output for a gievn t time.

        Parameters
        ----------
        t : Number
            The control/simulation time
        input_x : ArrayLike
            The input states for control (The observed states)
        x : ArrayLike
            Controller internal states
        refs : ArrayLike
            Trajectory reference for the controlled system states
        **kwargs:
            Any other deried parameter

        Returns
        -------
        np.ndarray
            Controller output
        """
        return super().output(t, x, refs-input_x, **kwargs)

    def dx(self, t: Number, input_x: ArrayLike, 
        x: ArrayLike, refs: ArrayLike, **kwargs) -> np.ndarray:
        """Method that computes the dx function (dx = f(t, input_x, x, refs, **kwargs))
        for control states update during numerical integration for a given t time.

        Parameters
        ----------
        t : Number
            The control/simulation time
        input_x : ArrayLike
            The input states for control (The observed states)
        x : ArrayLike
            Controller internal states
        refs : ArrayLike
            Trajectory reference for the controlled system states
        **kwargs:
            Any other deried parameter

        Returns
        -------
        np.ndarray
            Controller output
        """
        return super().dx(t, x, refs-input_x, **kwargs)

class LinearController(LinearStateSpaceSystem,Controller):
    """
    LinearController class represents all the contollers made to follow a certain reference that use
    linear control rules. Can be used directly by instatiating the A, B, C and D matrices for a 
    LinearStatesSpaceSystem. It implements 2 main methods: output and dx. Both of them
    must receive 4 required parameters. Control time, input states, controller states and the
    references to follow. This class inherits the LinearStatesSpaceSystem class.
    """
    def output(self, t: Number, input_x: ArrayLike, 
        x: ArrayLike, refs: ArrayLike, **kwargs) -> np.ndarray:
        """
        Method that computes the controller output for a gievn t time.
        For the lienar case y = Cx + D(refs-input_x)

        Parameters
        ----------
        t : Number
            The control/simulation time
        input_x : ArrayLike
            The input states for control (The observed states)
        x : ArrayLike
            Controller internal states
        refs : ArrayLike
            Trajectory reference for the controlled system states
        **kwargs:
            Any other deried parameter

        Returns
        -------
        np.ndarray
            Controller output
        """
        return super().output(t, x, refs-input_x)

    def dx(self, t: Number, input_x: ArrayLike, 
        x: ArrayLike, refs: ArrayLike, **kwargs) -> np.ndarray:
        """Method that computes the dx function (dx = f(t, input_x, x, refs, **kwargs))
        for control states update during numerical integration for a given t time.
        For the linear case dx = Ax+B(refs-input_X)

        Parameters
        ----------
        t : Number
            The control/simulation time
        input_x : ArrayLike
            The input states for control (The observed states)
        x : ArrayLike
            Controller internal states
        refs : ArrayLike
            Trajectory reference for the controlled system states
        **kwargs:
            Any other deried parameter

        Returns
        -------
        np.ndarray
            Controller output
        """
        return super().dx(t, x, refs-input_x)

class PCtrl(LinearController):
    """
    PCtrl implements a proportional (P) linear controller

    Parameters
    ------------
    gains: ArrayLike
        A 1-dim array like with one value, the proportional gain

    x0: ArrayLike, defaults [[0]]
        A 1-dim array with one value, the inital controller state.
        Since there is no dynamic involved this parameter does not influence
        the controller behavior    

    Attributes
    ------------
    gains: numpy.ndarray   
    kp: numpy.float64
        Proportional gain
    x0: numpy.ndarray
    """
    def __init__(self, gains: ArrayLike, x0:ArrayLike=[[0]]):
        self.gains = np.array(gains, dtype=np.float64)
        self.kp = self.gains[0]
        super().__init__(A=np.array([[0]]),
                         B=np.array([[0]]),
                         C=np.array([[0]]),
                         D=np.array([[self.kp]]),
                         x0=np.array(x0).reshape(-1,1))

class PICtrl(LinearController):
    """
    PICtrl implements a proportional-integrative (PI) linear controller.
    The parameters are implemented in the parallel form.

    Parameters
    ------------
    gains: ArrayLike
        A 1-dim array like with 2 values, the proportional gain (kp)
        and the integrative gain (ki).

    x0: ArrayLike, defaults [[0]]
        A 1-dim array with one value, the inital controller state.
    
    Attributes
    ------------

    gains: numpy.ndarray
    kp: numpy.float64
        Proportional gain
    ki: numpy.float64
        Integrative gain
    x0: numpy.ndarray
    """
    def __init__(self, gains: ArrayLike, x0:ArrayLike=[[0]]):
        self.gains = np.array(gains, dtype=np.float64)
        self.kp, self.ki = self.gains
        super().__init__(A=np.array([[0]]),
                         B=np.array([[1]]),
                         C=np.array([[self.ki]]),
                         D=np.array([[self.kp]]),
                         x0=np.array(x0).reshape(-1,1))

class PDCtrl(LinearController):

    def __init__(self, gains: ArrayLike, tau: Number, x0: ArrayLike):
        self.tau = np.float64(tau)
        self.gains = np.array(gains, dtype=np.float64)
        self.kp, self.kd = self.gains
        super().__init__(A=np.array([[-1/self.tau]]),
                         B=np.array([[1]]),
                         C=np.array([[self.kd/self.tau]]),
                         D=np.array([[self.kp]]),
                         x0=np.array(x0).reshape(-1,1))

class ExplicitCtrl(Controller):
    def __init__(self, gains:ArrayLike):
        self.gains = np.array(gains, dtype=np.float64)
    
    def output(self, t: Number, input_x: ArrayLike, x: ArrayLike, refs:ArrayLike, **kwargs) -> np.ndarray:
        input_x = np.array(input_x, dtype=np.float64)
        refs = np.array(refs, dtype=np.float64)
        t=np.float64(t)
        errors = refs[:-1]-input_x
        u = np.dot(self.gains, errors) + refs[-1]
    
    def dx(self, t: Number, input_x: ArrayLike, x: ArrayLike, refs:ArrayLike, **kwargs) -> np.ndarray:
        return np.array([[0]], dtype=np.float64)

class FbLinearizationCtrl(Controller):
    """
    The FbLinearizationCtrl, known in robotics as Computed Torque method, is a control rule that
    linearizes a non linear plant by feedbacking is nonliearities. This way the resulting controlled system is a linear
    one controlled by a user defined controller.

    Parameters
    ----------
    controller : CtrlLike
        A Controller instance or a ctrl config dict that implements that represents the desired controller

    Atributes
    ----------
    controller : Controller
        the Controller instance that implements the user desired control rule.

    """

    def __init__(self, controller: CtrlLike, **kwargs):
        if isinstance(controller, Controller):
            self.controller = controller
        elif isinstance(controller, dict):
            self.controller = get_ctrl_from_config(controller)
        else:
            raise TypeError('controller can only be a config dict or a Controller instance')
    
    def output(self, t: Number, input_x: ArrayLike, x: ArrayLike, refs:ArrayLike, **kwargs) -> np.ndarray:
        u = self.controller.output(t, input_x, x, refs, **kwargs)
        y = self.linearize(t, u, input_x, x, refs, **kwargs)
        return y
    
    def dx(self, t: Number, input_x: ArrayLike, x: ArrayLike, refs:ArrayLike, **kwargs) -> np.ndarray:
        return self.controller.dx(t, input_x, x, refs, **kwargs)
    
    @abstractmethod
    def linearize(self, t: Number, u: np.ndarray, input_x: ArrayLike, x: ArrayLike, refs: ArrayLike, **kwargs) -> np.ndarray:
        """
        Method that obtains a desired linear control output (u) and modifies it in the way that when the output of linearize is inputted into
        the controlled system, u is applied to a linear system.

        Parameters
        ----------
        t : Number
            The control/simulation time
        u : np.ndarray
            The linear control output
        input_x : ArrayLike
            The input states for control (The observed states)
        x : ArrayLike
            Controller internal states
        refs : ArrayLike
            Trajectory reference for the controlled system states
        **kwargs:
            Any other deried parameter

        Returns
        -------
        np.ndarray
            The output thet applies u to linearized system
        """
        raise NotImplementedError

def get_ctrl_from_config(config_dict: Dict[str, Any]) -> List[Controller]:

    locals_dict = locals()
    controllers = [locals_dict[ctrl_class](**ctrl_config) 
        for ctrl_class, ctrl_config in config_dict.items()]
    return controllers       