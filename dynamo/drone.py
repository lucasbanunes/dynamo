import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from dynamo.controllers import get_ctrl_from_config, FbLinearizationCtrl, Controller, CtrlLike
from dynamo.models import DynamicSystem
from dynamo.utils import is_callable, is_numeric, is_instance
from dynamo.mechanics import get_2d_rot_inv_matrix
# from dynamo.typing import CtrlLike
from typing import Callable, Union, Dict, Any, Tuple
from numbers import Number
# from autograd import elementwise_grad as egrad
# import autograd.numpy as anp
from collections import defaultdict
from dynamo.utils import Bunch

states_names = ['phi', 'dphi', 'theta', 'dtheta', 'psi', 'dpsi', 'x', 'dx', 'y', 'dy', 'z', 'dz']
states_idxs = np.arange(len(states_names))
names_idxs = {key: value for key, value in zip(states_names, states_idxs)}

class DroneStates(Bunch):

    def __init__(self, state_vector: ArrayLike):
        
        state_vector = np.array(state_vector)
        if state_vector.ndim == 1 and len(state_vector) != 12:
            raise ValueError(f'The state vector should have 12 elements but len was {len(state_vector)}')
        elif state_vector.ndim == 2 and state_vector.shape[0] != 12:
            raise ValueError(f'The state vector should have shape[1] == 12 elements but its shape was {state_vector.shape}')
        elif state_vector.ndim > 2:
            raise ValueError(f'state_vector should have 2 dims at max but it had shape {state_vector.shape}')

        super().__init__(
            phi = state_vector[0], dphi = state_vector[1],
            theta = state_vector[2], dtheta = state_vector[3],
            psi = state_vector[4], dpsi = state_vector[5],
            x = state_vector[6], dx = state_vector[7],
            y = state_vector[8], dy = state_vector[9],
            z = state_vector[10], dz = state_vector[11],
            state_vector=state_vector)

class ZFbLinearizationCtrl(FbLinearizationCtrl):

    def __init__(self, gains: ArrayLike, controller: CtrlLike, g: Number, mass: Number):
        super().__init__(gains, controller)

    def linearize(self, t: Number, u: np.ndarray, input_x: ArrayLike, x: ArrayLike, refs: ArrayLike, **kwargs) -> Number:
        phi = x[names_idxs['phi']]
        theta = x[names_idxs['theta']]
        output = (u+self.g)*self.mass/(np.cos(phi)*np.cos(theta))
        return output

class PsiFbLinearizationCtrl(FbLinearizationCtrl):

    def __init__(self, gains: ArrayLike, controller: CtrlLike, jz: Number):
        super().__init__(gains, controller)
    
    def linearize(self, t: Number, u: np.ndarray, input_x: ArrayLike, x: ArrayLike, refs: ArrayLike, **kwargs) -> Number:
        output = u*self.jz
        return output

class DroneController(Controller):

    def __init__(self, ref_x: Callable[[Number],Number], ref_dx: Callable[[Number],Number], ref_ddx: Callable[[Number],Number],
        ref_y: Callable[[Number],Number], ref_dy: Callable[[Number],Number], ref_ddy: Callable[[Number],Number], 
        ref_z: Callable[[Number],Number], ref_dz: Callable[[Number],Number], ref_ddz: Callable[[Number],Number], 
        ref_psi: Callable[[Number],Number], ref_dpsi: Callable[[Number],Number], ref_ddpsi: Callable[[Number],Number], 
        kp_x: Number, kd_x: Number, kp_y: Number, kd_y: Number, kp_z: Number, kd_z: Number, 
        kp_phi:Number, kd_phi:Number, kp_theta:Number, kd_theta:Number, kp_psi:Number, kd_psi:Number,
        A: ArrayLike, jx: Number, jy: Number, jz: Number,
        g: Number, mass: Number, log_internals:bool=False,
        ref_theta: Callable[[Number],Number]=None, ref_dtheta: Callable[[Number],Number]=None, ref_ddtheta: Callable[[Number],Number]=None,
        ref_phi: Callable[[Number],Number]=None, ref_dphi: Callable[[Number],Number]=None, ref_ddphi: Callable[[Number],Number]=None):
        """
        Parameters
        ------------
        ref_x: callable
            Callable that recieves time and returns desired x value
        ref_y: callable
            Callable that recieves time and returns desired y value
        ref_z: callable
            Callable that recieves time and returns desired z value
        ref_dz: callable
            Callable that recieves time and returns desired dz value
        ref_ddz: callable
            Callable that recieves time and returns desired ddz value
        ref_psi: callable
            Callable that recieves time and returns desired psi value
        ref_dpsi: callable
            Callable that recieves time and returns desired dpsi value
        ref_ddpsi: callable
            Callable that recieves time and returns desired ddpsi value
        kp_z: Number
            Proportional gain for z control
        kd_z: Number
            Derivative gain for z control
        kp_phi: Number
            Proportional gain for z control
        kd_phi: Number
            Derivative gain for z control
        kp_theta: Number
            Proportional gain for z control
        kd_theta: Number
            Derivative gain for z control
        kp_psi: Number
            Proportional gain for z control
        kd_psi: Number
            Derivative gain for z control
        g: Number
            Gravity
        mass: Number
            Drone mass
        A: Array like
            Square matrix for computing per motor thurst
        jx: Number
            Drone x inertia
        jy: Numver
            Drone y inertia
        jz: Number
            Drone z inertia
        log_internals: bool
            If true logs the internal variable values each time a output is computed by the controller
        """

        # References
        is_callable(ref_x)
        self.ref_x = ref_x
        is_callable(ref_dx)
        self.ref_dx = ref_dx
        is_callable(ref_ddx)
        self.ref_ddx = ref_ddx
        is_callable(ref_y)
        self.ref_y = ref_y
        is_callable(ref_dy)
        self.ref_dy = ref_dy
        is_callable(ref_ddy)
        self.ref_ddy = ref_ddy
        is_callable(ref_z)
        self.ref_z = ref_z
        is_callable(ref_dz)
        self.ref_dz = ref_dz
        is_callable(ref_ddz)
        self.ref_ddz = ref_ddz
        is_callable(ref_psi)
        self.ref_psi = ref_psi
        is_callable(ref_dpsi)
        self.ref_dpsi = ref_dpsi
        is_callable(ref_ddpsi)
        self.ref_ddpsi = ref_ddpsi
        
        # Gains
        is_numeric(kp_x)
        self.kp_x = np.float64(kp_x)
        is_numeric(kd_x)
        self.kd_x = np.float64(kd_x)
        is_numeric(kp_y)
        self.kp_y = np.float64(kp_y)
        is_numeric(kd_y)
        self.kd_y = np.float64(kd_y)
        is_numeric(kp_z)
        self.kp_z = np.float64(kp_z)
        is_numeric(kd_z)
        self.kd_z = np.float64(kd_z)
        is_numeric(kp_phi)
        self.kp_phi = np.float64(kp_phi)
        is_numeric(kd_phi)
        self.kd_phi = np.float64(kd_phi)
        is_numeric(kp_phi)
        self.kp_phi = np.float64(kp_phi)
        is_numeric(kd_phi)
        self.kd_phi = np.float64(kd_phi)
        is_numeric(kp_theta)
        self.kp_theta = np.float64(kp_theta)
        is_numeric(kd_theta)
        self.kd_theta = np.float64(kd_theta)
        is_numeric(kp_psi)
        self.kp_psi = np.float64(kp_psi)
        is_numeric(kd_psi)
        self.kd_psi = np.float64(kd_psi)
        
        is_numeric(g)
        self.g = np.float64(g)
        is_numeric(mass)
        self.mass = np.float64(mass)
        A = np.array(A, dtype=np.float64)
        self.Ainv = np.linalg.inv(A)
        is_numeric(jx)
        self.jx = np.float64(jx)
        is_numeric(jy)
        self.jy = np.float64(jy)
        is_numeric(jz)
        self.jz = np.float64(jz)

        self.ref_phi = get_ref_phi
        self.ref_theta = get_ref_theta

        # Configs
        is_instance(log_internals, bool)
        self.log_internals = log_internals
        self.internals = defaultdict(list)
    
    def compute(self, t: Number, state_vector: ArrayLike) -> Union[np.ndarray, dict]:
        # phi, dphi, theta, dtheta, psi, dpsi, x, dx, y, dy, z, dz = xs
        
        xs = DroneStates(state_vector)

        xs.ref_z = self.ref_z(t)
        xs.ref_dz = self.ref_dz(t)
        xs.ref_ddz = self.ref_ddz(t)
        xs.ref_x = self.ref_x(t)
        xs.ref_dx = self.ref_dx(t)
        xs.ref_ddx = self.ref_ddx(t)
        xs.ref_y = self.ref_y(t)
        xs.ref_dy = self.ref_dy(t)
        xs.ref_ddy = self.ref_ddy(t)
        xs.ref_psi = self.ref_psi(t)
        xs.ref_dpsi = self.ref_dpsi(t)
        xs.ref_ddpsi = self.ref_ddpsi(t)

        # Z control
        xs.e_z = xs.ref_z - xs.z
        xs.de_z = xs.ref_dz - xs.dz
        xs.u_z = self.kp_z*xs.e_z + self.kd_z*xs.de_z + xs.ref_ddz
        xs.f = (xs.u_z+self.g)*self.mass/(np.cos(xs.phi)*np.cos(xs.theta))

        # X control
        xs.e_x = xs.ref_x - xs.x
        xs.e_dx = xs.ref_dx - xs.dx
        xs.u_x = self.kp_x*xs.e_x + self.kd_x*xs.e_dx + xs.ref_ddx

        # Y control
        xs.e_y = xs.ref_y - xs.y
        xs.e_dy = xs.ref_dy - xs.dy
        xs.u_y = self.kp_y*xs.e_y + self.kd_y*xs.e_dy + xs.ref_ddy

        # xs.ref_phi, xs.ref_theta = get_refs(
        #     xs.u_x, xs.u_y, xs.psi, xs.f, self.mass)

        # phi control
        # xs.ref_phi = np.full(shape=xs.phi.shape, fill_value=np.pi/4)
        # xs.ref_dphi = np.full(shape=xs.phi.shape, fill_value=0)
        # xs.ref_ddphi = np.full(shape=xs.phi.shape, fill_value=0)
        # xs.e_phi = xs.ref_phi-xs.phi
        # xs.e_dphi = xs.ref_dphi-xs.dphi
        # xs.u_phi = self.kp_phi*xs.e_phi + self.kd_phi*xs.e_dphi + xs.ref_ddphi
        xs.ref_phi = self.ref_phi(xs.u_x, xs.u_y, xs.psi, xs.f, self.mass)
        xs.e_phi = xs.ref_phi-xs.phi
        xs.u_phi = self.kp_phi*xs.e_phi# - self.kd_phi*xs.dphi

        # theta control
        # xs.ref_theta = np.full(shape=xs.theta.shape, fill_value=np.pi/4)
        # xs.ref_dtheta = np.full(shape=xs.theta.shape, fill_value=0)
        # xs.ref_ddtheta = np.full(shape=xs.theta.shape, fill_value=0)
        # xs.e_theta = xs.ref_theta-xs.theta
        # xs.e_dtheta = xs.ref_dtheta-xs.dtheta
        # xs.u_theta = self.kp_theta*xs.e_theta + self.kd_theta*xs.e_dtheta + xs.ref_ddtheta
        xs.ref_theta = self.ref_theta(xs.u_x, xs.u_y, xs.ref_psi, xs.ref_phi, xs.f, self.mass)
        xs.e_theta = xs.ref_theta-xs.theta
        xs.u_theta = self.kp_theta*xs.e_theta# - self.kd_theta*xs.dtheta

        # psi control
        xs.e_psi = xs.ref_psi-xs.psi
        xs.de_psi = xs.ref_dpsi-xs.dpsi
        xs.u_psi = self.kp_psi*xs.e_psi+ self.kd_psi*xs.de_psi + xs.ref_ddpsi

        xs.m_x = xs.u_phi*self.jx
        xs.m_y = xs.u_theta*self.jy
        xs.m_z = xs.u_psi*self.jz
        if xs.state_vector.ndim == 1:
            xs.f_M = np.array([xs.f, xs.m_x, xs.m_y, xs.m_z])
            xs.fi = np.dot(self.Ainv, xs.f_M)
        elif xs.state_vector.ndim ==2:
            xs.f_M = np.column_stack((xs.f, xs.m_x, xs.m_y, xs.m_z))
            xs.fi = np.dot(self.Ainv, xs.f_M.T)

        return xs

    def output(self, t: Number, state_vector: ArrayLike) -> np.ndarray:
        res = self.compute(t, state_vector)
        return res
        # if self.log_internals:
        #     res = self.compute(t, xs, output_only=False)
        #     self._log_values(**res)
        #     # fi = np.array([res['f1'], res['f2'], res['f3'], res['f4']])
        #     fi = np.array([res['f'], res['m_x'], res['m_y'], res['m_z']])
        # else:
        #     fi = self.compute(t, xs, output_only=True)
        
        # return fi
    
    def _log_values(self, **kwargs):
        for key, value in kwargs.items():
            self.internals[key].append(value)       
    
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
        f, m_x, m_y, m_z =  np.dot(self.A, xs.fi)
        # f, m_x, m_y, m_z = xs.f, xs.m_x, xs.m_y, xs.m_z
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
        is_instance(controller, DroneController)
        self.controller=controller
        is_instance(drone, Drone)
        self.drone=drone

    def dx(self, t, state_vector) -> np.ndarray:
        xs = self.controller.output(t, state_vector)
        drone_dxs = self.drone(t, xs)
        return drone_dxs
    
    def output(self, t, xs):
        return np.array(xs)

def get_ref_phi(u_x: Number, u_y: Number, psi:Number, f: Number, mass:Number) -> np.float64:
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    ref_phi = np.arcsin((mass/f)*(spsi*u_x - cpsi*u_y))
    return ref_phi

def get_ref_theta(u_x: Number, u_y: Number, psi:Number, ref_phi: Number, f: Number, mass:Number) -> np.float64:
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    cphi = np.cos(ref_phi)
    ref_theta = np.arcsin((1/cphi)*(mass/f)*(cpsi*u_x + spsi*u_y))
    return ref_theta

def get_refs(u_x: Number, u_y: Number, psi: Number, f: Number, mass: Number) -> Tuple[Number, Number]:
    rpsi_inv = get_2d_rot_inv_matrix(psi)
    u = np.array([u_x, u_y])
    mat_prod = (rpsi_inv * u).sum(axis=1)
    aux = mat_prod*(mass/f)
    ref_phi = -np.arcsin(aux[1])
    cphi = np.cos(ref_phi)
    ref_theta = np.arcsin(aux[0]/cphi)
    return ref_phi, ref_theta
    