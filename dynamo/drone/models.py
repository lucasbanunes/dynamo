import numpy as np
from numpy.typing import ArrayLike, NDArray
from numbers import Number
from dynamo.base import Bunch
from dynamo.drone.utils import DroneStates
from dynamo.drone.controllers import PoseDroneController


class Drone():

    def __init__(self, jx: Number, jy: Number, jz: Number,
                 g: Number, mass: Number, A: ArrayLike):
        """
        Parameters
        ------------
        jx: Number
            Drone x inertia
        jy: Numver
            Drone y inertia
        jz: Number
            Drone z inertia
        g: Number
            Gravity
        mass: Number
            Drone mass
        A: Array like
            Square matrix for computing per motor thurst
        """
        self.jx = np.float64(jx)
        self.jy = np.float64(jy)
        self.jz = np.float64(jz)
        self.g = np.float64(g)
        self.mass = np.float64(mass)
        self.A = np.array(A, dtype=np.float64)

    def dx(self, t: Number, data: Bunch) -> Bunch:
        # f, m_x, m_y, m_z =  np.dot(self.A, out.fi)
        f, m_x, m_y, m_z = data.f, data.m_x, data.m_y, data.m_z
        f_over_m = f/self.mass
        sphi = np.sin(data.phi)
        cphi = np.cos(data.phi)
        stheta = np.sin(data.theta)
        ctheta = np.cos(data.theta)
        spsi = np.sin(data.psi)
        cpsi = np.cos(data.psi)

        vtheta = data.vtheta
        vpsi = data.vpsi
        vphi = data.vphi
        data.aphi = vtheta*vpsi*((self.jy-self.jz)/self.jx) + m_x/self.jx
        data.atheta = vphi*vpsi*((self.jz-self.jx)/self.jy) + m_y/self.jy
        data.apsi = vtheta*vphi*((self.jx-self.jy)/self.jz) + m_z/self.jz
        data.ax = f_over_m*((spsi*sphi) + (cphi*stheta*cpsi))
        data.ay = f_over_m*((-cpsi*sphi) + (spsi*stheta*cphi))
        data.az = -self.g+(cphi*ctheta*f_over_m)
        return data

    def output(self, t: Number, data: Bunch) -> Bunch:
        return self.dx(t, data)


class Propeller():

    def __init__(self, c_tau: Number, kt: Number):
        """
        Parameters
        ------------
        c_tau: Number
            aerodynamic torque constant

        kt: Number
            thrust aerodynamic constant
        """
        raise NotImplementedError


class ControledDrone():

    def __init__(self, controller: PoseDroneController, drone: Drone):
        self.controller = controller
        self.drone = drone

    def __call__(self, t: Number,
                 state_vector: NDArray[np.floating]
                 ) -> NDArray[np.floating]:
        data = self.output(t, state_vector)
        simulation_dx = np.array([
            data.dphi,
            data.ddphi,
            data.dtheta,
            data.ddtheta,
            data.dpsi,
            data.ddpsi,
            data.dx,
            data.ddx,
            data.dy,
            data.ddy,
            data.dz,
            data.ddz
        ], dtype=np.float64)
        return simulation_dx

    def output(self, t: Number,
               state_vector: NDArray[np.floating]
               ) -> NDArray[np.floating]:
        data = DroneStates(state_vector)
        data = self.controller.output(t, data)
        data = self.drone.dx(t, data)
        return data


class SpeedControledDrone():

    def __init__(self, controller: PoseDroneController, drone: Drone):
        self.controller = controller
        self.drone = drone
        self.states_names = [
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
            'dz',
            'ide_x',
            'ide_y',
            'ide_z',
            'ide_psi'
        ]
        self.dstates_names = [
            'dphi',
            'ddphi',
            'dtheta',
            'ddtheta',
            'dpsi',
            'ddpsi',
            'dx',
            'ddx',
            'dy',
            'ddy',
            'dz',
            'ddz',
            'de_x',
            'de_y',
            'de_z',
            'de_psi'
        ]

    def __call__(self, t: Number,
                 state_vector: NDArray[np.floating]
                 ) -> NDArray[np.floating]:
        data = self.output(t, state_vector)
        simulation_dx = [
            data[dstate_name]
            for dstate_name in self.dstates_names
        ]
        simulation_dx = np.array(simulation_dx, dtype=np.float64)
        return simulation_dx

    def output(self, t: Number, state_vector: NDArray[np.floating]
               ) -> NDArray[np.floating]:
        data = self.parse_states(state_vector)
        data = self.controller.output(t, data)
        data = self.drone.dx(t, data)
        return data

    def parse_states(self, state_vector: NDArray[np.floating]) -> Bunch:
        state_mapping = dict(zip(self.states_names, state_vector))
        state_mapping['vector'] = state_vector
        data = Bunch(**state_mapping)
        return data
