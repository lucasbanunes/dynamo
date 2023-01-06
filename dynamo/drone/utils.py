import numpy as np
from numpy.typing import ArrayLike
from dynamo.base import Bunch


STATES_NAMES = [
    'phi',
    'dphi',
    'theta',
    'dtheta',
    'psi',
    'dpsi',
    'x',
    'dx',
    'y',
    'dy',
    'z',
    'dz'
]


class DroneStates(Bunch):

    states_names = STATES_NAMES

    def __init__(self, state_vector: ArrayLike):
        state_vector = np.array(state_vector)
        attributes = {"vector": state_vector}
        for name, value in zip(self.states_names, state_vector):
            attributes[name] = value
        super().__init__(**attributes)
