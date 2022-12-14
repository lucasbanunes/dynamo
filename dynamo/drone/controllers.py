import numpy as np
from numpy.typing import ArrayLike, NDArray
from numbers import Number
from dynamo.base import Bunch, Controller
from dynamo.signal import TimeSignal


@np.vectorize
def saturated_arcsin(x: Number) -> NDArray[np.floating]:
    """
    Alternative arcsin for simulating.
    Due to project mistakes or numeric errors
    a arcsin function may receive x > 1 or x < -1.
    Vectorized with numpy.vectorize

    Parameters
    ----------
    x : Number

    Returns
    -------
    NDArray[np.floating]
    """
    if x > 1:
        return np.pi/2
    elif x < -1:
        return -np.pi/2
    else:
        return np.arcsin(x)


class BaseDroneController(Controller):
    """
    Base utils for the drone system controller design
    defined in dynamo.drone.models.Drone
    """

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

    def add_internal_refs(self, data: Bunch) -> Bunch:
        """
        Calculates the references for the internal phi (pitch)
        and theta (roll) control.

        Parameters
        ----------
        data : Bunch
            Bunch object with input data

        Returns
        -------
        Bunch
            The same Bunch input object
        """
        spsi = np.sin(data.psi)
        cpsi = np.cos(data.psi)
        mf = self.mass/data.f

        data.ref_sphi = mf*((cpsi*data.u_y) - (spsi*data.u_x))
        data.sphi = np.sin(data.phi)
        data.cphi = np.cos(data.phi)
        data.ref_phi = -saturated_arcsin(data.ref_sphi)
        data.ref_stheta = (1/data.cphi)*mf*((cpsi*data.u_x) + (spsi*data.u_y))
        data.stheta = np.sin(data.theta)
        data.ctheta = np.cos(data.theta)
        data.ref_theta = saturated_arcsin(data.ref_stheta)

        zero_vector = np.full(data.t.shape, 0.0, dtype=np.float64)
        data.ref_vphi = zero_vector
        data.ref_aphi = zero_vector
        data.ref_vtheta = zero_vector
        data.ref_atheta = zero_vector

        return data

    def apply_pd_control(self,
                         e: Number,
                         de: Number,
                         ddref: Number,
                         kp: Number,
                         kd: Number):
        u = kp*e + kd*de + ddref
        return u

    def apply_pi_control(self,
                         e: Number,
                         ie: Number,
                         kd: Number,
                         ki: Number):
        u = kd*e + ki*ie
        return u


class PoseDroneController(BaseDroneController):
    """
    Drone position/orientation controller.
    This Controller implements:
    - Feedback linearization with pd control for z axis position and psi (roll)
    - pd control for x axis and y axis positions
    - Internal phi (pitch) and theta (roll) control by assuming
    the approximation of a double integrator as roll and pitch dynamics
    when applied with low speeds but fast enough speeds such that the
    current values are approxmately the reference values.
    """

    def output(self, t: Number, data: Bunch) -> Bunch:
        """
        Adds the control output variables to data inplace.

        Parameters
        ----------
        t : Number
            Simulation time
        data : Bunch
            Input bunch variables

        Returns
        -------
        Bunch
            Same instance from data but with the output variables
            added
        """

        data.t = np.float64(t)
        data.ref_px = self.refs.px(t)
        data.ref_vx = self.refs.vx(t)
        data.ref_ax = self.refs.ax(t)
        data.ref_py = self.refs.py(t)
        data.ref_vy = self.refs.vy(t)
        data.ref_ay = self.refs.ay(t)
        data.ref_pz = self.refs.pz(t)
        data.ref_vz = self.refs.vz(t)
        data.ref_az = self.refs.az(t)
        data.ref_psi = self.refs.psi(t)
        data.ref_vpsi = self.refs.vpsi(t)
        data.ref_apsi = self.refs.apsi(t)

        # Z control
        data.e_pz = data.ref_pz - data.pz
        data.e_vz = data.ref_vz - data.vz
        data.u_z = self.apply_pd_control(
            e=data.e_pz, de=data.e_vz, ddref=data.ref_az,
            kp=self.gains.kp_z, kd=self.gains.kd_z
        )
        data.kh = np.cos(data.phi)*np.cos(data.theta)
        data.f = (data.u_z+self.g)*self.mass/data.kh

        # X control
        data.e_px = data.ref_px - data.px
        data.e_vx = data.ref_vx - data.vx
        data.u_x = self.apply_pd_control(
            e=data.e_px, de=data.e_vx, ddref=data.ref_ax,
            kp=self.gains.kp_x, kd=self.gains.kd_x
        )

        # Y control
        data.e_py = data.ref_py - data.py
        data.e_vy = data.ref_vy - data.vy
        data.u_y = self.apply_pd_control(
            e=data.e_py, de=data.e_vy, ddref=data.ref_ay,
            kp=self.gains.kp_y, kd=self.gains.kd_y
        )

        # Psi control
        data.e_psi = data.ref_psi - data.psi
        data.e_vpsi = data.ref_vpsi - data.vpsi
        data.u_psi = self.apply_pd_control(
            e=data.e_psi, de=data.e_vpsi, ddref=data.ref_apsi,
            kp=self.gains.kp_psi, kd=self.gains.kd_psi
        )
        data.m_z = data.u_psi*self.jz

        self.add_internal_refs(data)

        # Phi control
        data.e_phi = data.ref_phi - data.phi
        data.e_vphi = data.ref_vphi - data.vphi
        data.u_phi = self.apply_pd_control(
            e=data.e_phi, de=data.e_vphi, ddref=data.ref_aphi,
            kp=self.gains.kp_phi, kd=self.gains.kd_phi
        )
        data.m_x = data.u_phi*self.jx

        # Theta control
        data.e_theta = data.ref_theta - data.theta
        data.e_vtheta = data.ref_vtheta - data.vtheta
        data.u_theta = self.apply_pd_control(
            e=data.e_theta, de=data.e_vtheta, ddref=data.ref_atheta,
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


class SpeedDroneController(BaseDroneController):
    """
    Drone velocity controller.
    This Controller implements:
    - Feedback linearization with pd control for z axis and psi (roll) speeds.
    - pd control for x axis and y axis speeds.
    - Internal phi (pitch) and theta (roll) control from the desired x and y
    speeds.
    """

    def output(self, t: Number, data: Bunch) -> Bunch:
        """
        Adds the control output variables to data inplace.

        Parameters
        ----------
        t : Number
            Simulation time
        data : Bunch
            Input bunch variables

        Returns
        -------
        Bunch
            Same instance from data but with the output variables
            added
        """

        data.t = np.float64(t)
        data.ref_vx = self.refs.vx(t)
        data.ref_vy = self.refs.vy(t)
        data.ref_vz = self.refs.vz(t)
        data.ref_vpsi = self.refs.vpsi(t)

        # Z control
        data.e_vz = data.ref_vz - data.vz
        data.u_z = self.apply_pi_control(
            e=data.e_vz, ie=data.ie_vz,
            ki=self.gains.ki_z, kd=self.gains.kd_z
        )
        data.kh = np.cos(data.phi)*np.cos(data.theta)
        data.f = (data.u_z+self.g)*self.mass/data.kh

        # X control
        data.e_vx = data.ref_vx - data.vx
        data.u_x = self.apply_pi_control(
            e=data.e_vx, ie=data.ie_vx,
            ki=self.gains.ki_x, kd=self.gains.kd_x
        )

        # Y control
        data.e_vy = data.ref_vy - data.vy
        data.u_y = self.apply_pi_control(
            e=data.e_vy, ie=data.ie_vy,
            ki=self.gains.ki_y, kd=self.gains.kd_y
        )

        # Psi control
        data.e_vpsi = data.ref_vpsi - data.vpsi
        data.u_psi = self.apply_pi_control(
            e=data.e_vpsi, ie=data.ie_vpsi,
            ki=self.gains.ki_psi, kd=self.gains.kd_psi
        )
        data.m_z = data.u_psi*self.jz - \
            (data.vtheta*data.vphi)*(self.jx-self.jy)

        self.add_internal_refs(data)

        # Phi control
        data.e_sphi = data.ref_sphi - data.sphi
        data.e_vphi = data.ref_phi - data.vphi
        data.u_phi = self.gains.kp_phi*data.e_sphi - \
            self.gains.kd_phi*data.cphi*data.vphi
        data.m_x = -(data.vtheta*data.vpsi)*(self.jy-self.jz) + \
            self.jx*(data.u_phi + (data.sphi*(data.vphi**2))/data.cphi)

        # Theta control
        data.e_stheta = data.ref_stheta - data.stheta
        data.e_vtheta = data.ref_theta - data.vtheta
        data.u_theta = self.gains.kp_theta*data.e_stheta - \
            self.gains.kd_theta*data.ctheta*data.vtheta
        data.m_y = -(data.vphi*data.vpsi)*(self.jz-self.jx) + \
            self.jy*(data.u_theta + (data.stheta*(data.vtheta**2))/data.ctheta)

        if data.vector.ndim == 1:
            data.f_M = np.array([data.f, data.m_x, data.m_y, data.m_z])
            data.fi = np.dot(self.Ainv, data.f_M)
        elif data.vector.ndim == 2:
            data.f_M = np.column_stack((data.f, data.m_x, data.m_y, data.m_z))
            data.fi = np.dot(self.Ainv, data.f_M.T)

        return data
