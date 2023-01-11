from typing import Sequence
from numbers import Number
import numpy as np
from numpy.typing import NDArray
from .base import Bunch, DynamicSystem, Controller


class ControlledSystem(DynamicSystem):
    """
    Representation of a system represented by a controller
    and a system to be controlled (plant). Controller.output
    is called on a Bunch instance containg the states and its 
    return is fed to the plant output.
    Automatically build input bunch and the states derivative as
    an array for compatibility with scipy ivp solvers.
    Inherits dynamo.base.DynamicSystem

    Attributes
    ----------
    controller : Controller
        Plant Controller instance
    system : DynamicSystem
        PLant DynamicSystem instance
    states_names : Sequence[str]
        Sequence with the variable name of each state
    dstates_names : Sequence[str]
        Sequence with the variable name of wach state derivative.
        The output derivative state order is defined by dstates_names
        order.
    """

    def __init__(self, controller: Controller,
                 system: DynamicSystem,
                 states_names: Sequence[str],
                 dstates_names: Sequence[str]):
        """
        Parameters
        ----------
        controller : Controller
            Plant Controller instance
        system : DynamicSystem
            PLant DynamicSystem instance
        states_names : Sequence[str]
            Sequence with the variable name of each state
        dstates_names : Sequence[str]
            Sequence with the variable name of wach state derivative.
            The output derivative state order is defined by dstates_names
            order.
        """
        self.controller = controller
        self.system = system
        self.states_names = states_names
        self.dstates_names = dstates_names

    def __call__(self, t: Number,
                 state_vector: NDArray[np.floating]
                 ) -> NDArray[np.floating]:
        """
        Returns the states derivative array for numerical
        integration.

        Parameters
        ----------
        t : Number
            Simulation time
        state_vector : NDArray[np.floating]
            Current state vector

        Returns
        -------
        NDArray[np.floating]
            State vector derivative
        """
        data_kwargs = {name: value
                       for name, value in zip(self.states_names, state_vector)}
        data_kwargs["vector"] = state_vector
        data = Bunch(**data_kwargs)
        self.output(t, data)
        simulation_dstates = [
            data[dstate_name]
            for dstate_name in self.dstates_names
        ]
        simulation_dstates = np.array(simulation_dstates, dtype=np.float64)
        return simulation_dstates

    def output(self, t: Number, data: Bunch) -> Bunch:
        self.controller.output(t, data)
        self.system.output(t, data)
        return data

    def dx(self, t: Number, data: Bunch) -> Bunch:
        self.system.dx(t, data)
        return data
