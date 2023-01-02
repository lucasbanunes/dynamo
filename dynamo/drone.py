import numpy as np
from numpy.typing import ArrayLike, NDArray
from numbers import Number
from dynamo.base import Bunch
from dynamo.signal import TimeSignal


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


class DroneController():

    def __init__(self, refs: TimeSignal, gains: Bunch,
                 mass: Number, g: Number,
                 jx: Number, jy: Number, jz: Number,
                 A: ArrayLike):

        self.refs = refs
        self.gains = gains
        self.mass = np.float64(mass)
        self.g = np.float64(g)
        self.jx = np.float64(jx)
        self.jy = np.float64(jy)
        self.jz = np.float64(jz)
        self.A = np.array(A, dtype=np.float64)
        Ainv = np.linalg.inv(A)
        self.Ainv = np.linalg.inv(Ainv)

    def output(self, t: Number, data: Bunch) -> Bunch:

        data.t = np.float64(t)
        data.ref_x = self.refs.x(t)
        data.dref_x = self.refs.dx(t)
        data.ddref_x = self.refs.ddx(t)
        data.ref_y = self.refs.y(t)
        data.dref_y = self.refs.dy(t)
        data.ddref_y = self.refs.ddy(t)
        data.ref_z = self.refs.z(t)
        data.dref_z = self.refs.dz(t)
        data.ddref_z = self.refs.ddz(t)
        data.ref_psi = self.refs.psi(t)
        data.dref_psi = self.refs.dpsi(t)
        data.ddref_psi = self.refs.ddpsi(t)

        # Z control
        data.e_z = data.ref_z - data.z
        data.de_z = data.dref_z - data.dz
        data.u_z = self.apply_pd_control(
            e=data.e_z, de=data.de_z, ddref=data.ddref_z,
            kp=self.gains.kp_z, kd=self.gains.kd_z
        )
        num = (data.u_z+self.g)*self.mass
        den = np.cos(data.phi)*np.cos(data.theta)
        data.f = num/den

        # X control
        data.e_x = data.ref_x - data.x
        data.e_dx = data.dref_x - data.dx
        data.u_x = self.apply_pd_control(
            e=data.e_z, de=data.de_z, ddref=data.ddref_z,
            kp=self.gains.kp_z, kd=self.gains.kd_z
        )

        # Y control
        data.e_y = data.ref_y - data.y
        data.e_dy = data.dref_y - data.dy
        data.u_y = self.apply_pd_control(
            e=data.e_z, de=data.de_z, ddref=data.ddref_z,
            kp=self.gains.kp_z, kd=self.gains.kd_z
        )

        # Psi control
        data.e_psi = data.ref_psi - data.psi
        data.de_psi = data.dref_psi - data.dpsi
        data.u_psi = self.apply_pd_control(
            e=data.e_z, de=data.de_z, ddref=data.ddref_z,
            kp=self.gains.kp_z, kd=self.gains.kd_z
        )
        data.m_z = data.u_psi*self.jz

        self.add_internal_refs(data)

        # Phi control
        data.e_phi = data.ref_phi - data.phi
        data.de_phi = data.dref_phi - data.dphi
        data.u_phi = self.apply_pd_control(
            e=data.e_z, de=data.de_z, ddref=data.ddref_z,
            kp=self.gains.kp_z, kd=self.gains.kd_z
        )
        data.m_x = data.u_phi*self.jx

        # Theta control
        data.e_theta = data.ref_theta - data.theta
        data.de_theta = data.dref_theta - data.dtheta
        data.u_theta = self.apply_pd_control(
            e=data.e_z, de=data.de_z, ddref=data.ddref_z,
            kp=self.gains.kp_z, kd=self.gains.kd_z
        )
        data.m_y = data.u_theta*self.jy

        if data.vector.ndim == 1:
            data.f_M = np.array([data.f, data.m_x, data.m_y, data.m_z])
            data.fi = np.dot(self.Ainv, data.f_M)
        elif data.vector.ndim == 2:
            data.f_M = np.column_stack((data.f, data.m_x, data.m_y, data.m_z))
            data.fi = np.dot(self.Ainv, data.f_M.T)

        return data

    def add_internal_refs(self, data: Bunch) -> Bunch:
        spsi = np.sin(data.psi)
        cpsi = np.cos(data.psi)
        mf = self.mass/data.f
        data.ref_phi = -np.arcsin(mf*((cpsi*data.u_y) - (spsi*data.u_x)))
        cphi = np.cos(data.ref_phi)
        data.ref_theta = \
            np.arcsin((1/cphi)*mf*((cpsi*data.u_x) + (spsi*data.u_y)))

        data.dref_phi = np.full(data.ref_phi.shape, 0)
        data.ddref_phi = np.full(data.ref_phi.shape, 0)
        data.dref_theta = np.full(data.ref_theta.shape, 0)
        data.ddref_theta = np.full(data.ref_theta.shape, 0)

        return data

    def apply_pd_control(self,
                         e: Number,
                         de: Number,
                         ddref: Number,
                         kp: Number,
                         kd: Number):
        u = kp*e + kd*de + ddref
        return u


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

        dtheta = data.dtheta
        dpsi = data.dpsi
        dphi = data.dphi
        data.ddphi = dtheta*dpsi*((self.jy-self.jz)/self.jx) + m_x/self.jx
        data.ddtheta = dphi*dpsi*((self.jz-self.jx)/self.jy) + m_y/self.jy
        data.ddpsi = dtheta*dphi*((self.jx-self.jy)/self.jz) + m_z/self.jz
        data.ddx = f_over_m*((spsi*sphi) + (cphi*stheta*cpsi))
        data.ddy = f_over_m*((-cpsi*sphi) + (spsi*stheta*cphi))
        data.ddz = -self.g+(cphi*ctheta*f_over_m)
        return data


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

    def __init__(self, controller: DroneController, drone: Drone):
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
