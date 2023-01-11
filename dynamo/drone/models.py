import numpy as np
from numpy.typing import ArrayLike
from numbers import Number
from dynamo.base import Bunch


class Drone():
    """
    Quadrimotor drone model
    """

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
        """
        Adds the system states to data inplace.

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
        data.aphi = (vtheta*vpsi*((self.jy-self.jz)/self.jx) + m_x/self.jx)
        data.atheta = (vphi*vpsi*((self.jz-self.jx)/self.jy) + m_y/self.jy)
        data.apsi = (vtheta*vphi*((self.jx-self.jy)/self.jz) + m_z/self.jz)
        data.ax = (f_over_m*((spsi*sphi) + (cphi*stheta*cpsi)))
        data.ay = (f_over_m*((-cpsi*sphi) + (spsi*stheta*cphi)))
        data.az = (-self.g+(cphi*ctheta*f_over_m))
        return data

    def output(self, t: Number, data: Bunch) -> Bunch:
        """
        Adds the system output variables to data inplace.

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
        return self.dx(t, data)


class Propeller():
    """
    Model for a drone propeller motor.
    Not Implemented.
    """

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
