from numbers import Number
import numpy as np
from numpy.typing import NDArray
from .base import Bunch, DynamicSystem


class ControlledSystem(DynamicSystem):

    def __init__(self, controller, system, states_names, dstates_names):
        self.controller = controller
        self.system = system
        self.states_names = states_names
        self.dstates_names = dstates_names

    def __call__(self, t: Number,
                 state_vector: NDArray[np.floating]
                 ) -> NDArray[np.floating]:
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
