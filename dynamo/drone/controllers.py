import numpy as np
from numpy.typing import ArrayLike
from numbers import Number
from dynamo.base import Bunch
from dynamo.signal import TimeSignal


@np.vectorize
def saturated_arcsin(x):
    if x > 1:
        return np.pi/2
    elif x < -1:
        return -np.pi/2
    else:
        return np.arcsin(x)


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
        data.kh = np.cos(data.phi)*np.cos(data.theta)
        data.f = (data.u_z+self.g)*self.mass/data.kh

        # X control
        data.e_x = data.ref_x - data.x
        data.de_x = data.dref_x - data.dx
        data.u_x = self.apply_pd_control(
            e=data.e_x, de=data.de_x, ddref=data.ddref_x,
            kp=self.gains.kp_x, kd=self.gains.kd_x
        )

        # Y control
        data.e_y = data.ref_y - data.y
        data.de_y = data.dref_y - data.dy
        data.u_y = self.apply_pd_control(
            e=data.e_y, de=data.de_y, ddref=data.ddref_y,
            kp=self.gains.kp_y, kd=self.gains.kd_y
        )

        # Psi control
        data.e_psi = data.ref_psi - data.psi
        data.de_psi = data.dref_psi - data.dpsi
        data.u_psi = self.apply_pd_control(
            e=data.e_psi, de=data.de_psi, ddref=data.ddref_psi,
            kp=self.gains.kp_psi, kd=self.gains.kd_psi
        )
        data.m_z = data.u_psi*self.jz

        self.add_internal_refs(data)

        # Phi control
        data.e_phi = data.ref_phi - data.phi
        data.de_phi = data.dref_phi - data.dphi
        data.u_phi = self.apply_pd_control(
            e=data.e_phi, de=data.de_phi, ddref=data.ddref_phi,
            kp=self.gains.kp_phi, kd=self.gains.kd_phi
        )
        data.m_x = data.u_phi*self.jx

        # Theta control
        data.e_theta = data.ref_theta - data.theta
        data.de_theta = data.dref_theta - data.dtheta
        data.u_theta = self.apply_pd_control(
            e=data.e_theta, de=data.de_theta, ddref=data.ddref_theta,
            kp=self.gains.kp_theta, kd=self.gains.kd_theta
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
        data.ref_phi = \
            -saturated_arcsin(mf*((cpsi*data.u_y) - (spsi*data.u_x)))
        cphi = np.cos(data.ref_phi)
        data.ref_theta = \
            saturated_arcsin((1/cphi)*mf*((cpsi*data.u_x) + (spsi*data.u_y)))

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
