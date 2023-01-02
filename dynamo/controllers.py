from abc import abstractmethod
import numpy as np
from numbers import Number
from dynamo.base import Controller, Bunch
from dynamo.utils import is_instance


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

    def __init__(self, controller: Controller, **kwargs):
        super().__init__(**kwargs)
        is_instance(controller, Controller, var_name="controller")
        super().__init__(**kwargs)

    def get_output(self, t: Number, input_bunch: Bunch) -> Bunch:
        ctrl_bunch = self.controller.output(t, input_bunch)
        output_bunch = self.linearize(t, ctrl_bunch)
        return output_bunch

    @abstractmethod
    def linearize(self, t: Number, input_bunch: Bunch) -> Bunch:
        """
        Method that applies the inverse non lineartites from the system to the
        outer controler output in a way that the outer controller sees
        a linear plant.
        """
        raise NotImplementedError


class PDController(Controller):

    def __init__(self, kp: Number, kd: Number, **kwargs):
        self.kp = np.float64(kp)
        self.kd = np.float(kd)
        super().__init__(**kwargs)

    def get_output(self, t: Number, input_bunch: Bunch) -> Bunch:
        data = input_bunch
        data.y = self.kp*data.e + self.kd*data.de + data.ddref
        return data
