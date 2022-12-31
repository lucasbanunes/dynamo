import numpy as np
from numpy.typing import ArrayLike, NDArray
from collections.abc import Mapping
from typing import Any, Tuple, Self, Iterable
from numbers import Number
from dynamo.models import DynamicSystem
from dynamo.utils import is_numeric
from dynamo.utils import Bunch
from dynamo.signal import TimeSignal
from dynamo.controllers import (
    FbLinearizationCtrl,
    Controller,
    CtrlLike
)

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


class DroneStates(Mapping):

    __slots__ = STATES_NAMES

    def __init__(self, state_vector: ArrayLike[np.floating]) -> Self:
        state_vector = np.array(state_vector)
        for name, value in zip(STATES_NAMES, state_vector):
            self.__dict__[name] = value
        self.vector = state_vector

    def __repr__(self) -> str:
        name_values = [
            f"{name}={value}"
            for name, value in self.items()
            if name != 'vector'
        ]
        repr_str = ",".join(name_values)
        return repr_str

    def __getitem__(self, idx: str) -> np.floating:
        return self.__dict__[idx]

    def __len__(self) -> int:
        return len(self.vector)

    def __iter__(self) -> NDArray[np.floating]:
        return self.vector

    def items(self) -> Iterable:
        return self.__dict__.items()


class ZFbLinearizationCtrl(FbLinearizationCtrl):

    def __init__(self, controller: CtrlLike, g: Number, mass: Number) -> Self:
        self.g = np.float64(g)
        self.mass = np.float64(mass)
        super().__init__(controller)

    def linearize(self, t: Number, ctrl_out: Any, phi: Number, theta: Number,
                  *args, **kwargs) -> Number:
        output = (ctrl_out+self.g)*self.mass/(np.cos(phi)*np.cos(theta))
        return output


class PsiFbLinearizationCtrl(FbLinearizationCtrl):

    def __init__(self, controller: CtrlLike, jz: Number) -> Self:
        self.jz = np.float64(jz)
        super().__init__(controller)

    def linearize(self, t: Number, ctrl_out: Any, xs: DroneStates) -> Number:
        y = ctrl_out*self.jz
        return y


class DroneController(Controller):

    drone_states = STATES_NAMES

    def __init__(self, refs: TimeSignal,
                 mass: Number, jx: Number, jy: Number, jz: Number,
                 A: ArrayLike[np.floating],
                 x_ctrl: CtrlLike, y_ctrl: CtrlLike, z_ctrl: CtrlLike,
                 phi_ctrl: CtrlLike, theta_ctrl: CtrlLike, psi_ctrl: CtrlLike
                 ) -> Self:

        self.refs = refs
        self.x_ctrl = x_ctrl
        self.y_ctrl = y_ctrl
        self.z_ctrl = z_ctrl
        self.phi_ctrl = phi_ctrl
        self.psi_ctrl = psi_ctrl
        self.theta_ctrl = theta_ctrl
        self.mass = np.float64(mass)
        self.jx = np.float64(jx)
        self.jy = np.float64(jy)
        self.jz = np.float64(jz)
        self.A = np.array(A, dtype=np.float64)
        self.Ainv = np.linalg.inv(self.Ainv)

    def output(self, t: Number, state_vector: ArrayLike) -> DroneStates:

        states = DroneStates(state_vector)
        out = Bunch()

        out.ref_x = self.refs.x(t)
        out.ref_dx = self.refs.dx(t)
        out.ref_ddx = self.refs.ddx(t)
        out.ref_y = self.refs.y(t)
        out.ref_dy = self.refs.dy(t)
        out.ref_ddy = self.refs.ddy(t)
        out.ref_z = self.refs.z(t)
        out.ref_dz = self.refs.dz(t)
        out.ref_ddz = self.refs.ddz(t)
        out.ref_psi = self.refs.psi(t)
        out.ref_dpsi = self.refs.dpsi(t)
        out.ref_ddpsi = self.refs.ddpsi(t)

        # Z control
        out.e_z = out.ref_z - out.z
        out.de_z = out.ref_dz - out.dz
        out.f, out.u_z = self.z_ctrl.output(t, e=out.e_z, de=out.de_z,
                                            ddref=out.ref_ddz, **states)

        # X control
        out.e_x = out.ref_x - out.x
        out.e_dx = out.ref_dx - out.dx
        out.u_x = self.x_ctrl.output(t, e=out.e_x, de=out.de_x,
                                     ddref=out.ref_ddx, **states)

        # Y control
        out.e_y = out.ref_y - out.y
        out.e_dy = out.ref_dy - out.dy
        out.u_y = self.y_ctrl.output(t, e=out.e_y, de=out.de_y,
                                     ddref=out.ref_ddy, **states)

        # Psi control
        out.e_psi = out.ref_psi - out.psi
        out.de_psi = out.ref_dpsi - out.dpsi
        out.f, out.u_psi = self.psi_ctrl.output(t, e=out.e_psi, de=out.de_psi,
                                                ddref=out.ref_ddpsi, **states)

        out.ref_phi, out.ref_theta = self.get_internal_refs(
            states, out.f, out.u_x, out.u_y
        )

        # Phi control
        out.e_phi = out.ref_phi - out.phi
        out.u_phi = self.phi_ctrl.output(t, e=out.e_phi, speed=states.dphi,
                                         **states)

        # Theta control
        out.e_theta = out.ref_theta - out.theta
        out.u_theta = self.theta_ctrl.output(t, e=out.e_theta,
                                             speed=states.dtheta,
                                             **states)

        out.m_x = out.u_phi*self.jx
        out.m_y = out.u_theta*self.jy
        out.m_z = out.u_psi*self.jz

        if out.state_vector.ndim == 1:
            out.f_M = np.array([out.f, out.m_x, out.m_y, out.m_z])
            out.fi = np.dot(self.Ainv, out.f_M)
        elif out.state_vector.ndim == 2:
            out.f_M = np.column_stack((out.f, out.m_x, out.m_y, out.m_z))
            out.fi = np.dot(self.Ainv, out.f_M.T)

        return out

    def get_internal_refs(self, states: DroneStates, f: Number,
                          u_x: Number, u_y: Number
                          ) -> Tuple[np.floating, np.floating]:
        spsi = np.sin(states.psi)
        cpsi = np.cos(states.psi)
        mf = self.mass/f
        ref_phi = -np.arcsin(mf*((cpsi*u_y) - (spsi*u_x)))
        cphi = np.cos(ref_phi)
        ref_theta = np.arcsin((1/cphi)*mf*((cpsi*u_x) + (spsi*u_y)))
        return ref_phi, ref_theta


class Drone(DynamicSystem):

    def __init__(self, jx: Number, jy: Number, jz: Number,
        g: Number, mass: Number, A:ArrayLike):
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
        is_numeric(jx)
        self.jx = jx
        is_numeric(jy)
        self.jy = jy
        is_numeric(jz)
        self.jz = jz
        is_numeric(jx)
        self.jx = jx
        is_numeric(g)
        self.g = g
        is_numeric(mass)
        self.mass = mass
        self.A = np.array(A, dtype=np.float64)
    
    def dx(self, t: Number, xs: DroneStates) -> np.ndarray:
        # f, m_x, m_y, m_z =  np.dot(self.A, out.fi)
        f, m_x, m_y, m_z = xs.f, xs.m_x, xs.m_y, xs.m_z
        f_over_m = f/self.mass
        phi, dphi, theta, dtheta, psi, dpsi, x, dx, y, dy, z, dz = xs.state_vector
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        stheta = np.sin(theta)
        ctheta = np.cos(theta)
        spsi = np.sin(psi)
        cpsi = np.cos(psi)
        
        ddphi = dtheta*dpsi*((self.jy-self.jz)/self.jx) + m_x/self.jx
        ddtheta = dphi*dpsi*((self.jz-self.jx)/self.jy) + m_y/self.jy
        ddpsi = dtheta*dphi*((self.jx-self.jy)/self.jz) +  m_z/self.jz
        ddx = f_over_m*((spsi*sphi) + (cphi*stheta*cpsi))
        ddy = f_over_m*((-cpsi*sphi) + (spsi*stheta*cphi))
        ddz = -self.g+(cphi*ctheta*f_over_m)

        dxs = np.array([dphi, ddphi, dtheta, ddtheta, dpsi, ddpsi, dx, ddx, dy, ddy, dz, ddz])
        return dxs

    def output(self, t, xs):
        return np.array(xs)


class Propeller(DynamicSystem):

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


class ControledDrone(DynamicSystem):

    def __init__(self, controller: DroneController, drone: Drone) -> None:
        self.controller = controller
        self.drone = drone

    def dx(self, t, state_vector) -> np.ndarray:
        xs = self.controller.output(t, state_vector)
        drone_dxs = self.drone(t, xs)
        return drone_dxs

    def output(self, t, xs):
        return np.array(xs)
