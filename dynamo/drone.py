import numpy as np
import pandas as pd
import numpy.typing as npt
from numpy.typing import ArrayLike
from dynamo.controllers import get_ctrl_from_config, FbLinearizationCtrl, Controller
from dynamo.models import DynamicSystem
from dynamo.utils import is_callable, is_numeric, is_instance
from dynamo.typing import CtrlLike
from typing import Callable, Union, Dict, Any
from numbers import Number
# from autograd import elementwise_grad as egrad
# import autograd.numpy as anp
from collections import defaultdict

states_names = ['phi', 'dphi', 'theta', 'dtheta', 'psi', 'dpsi', 'x', 'dx', 'y', 'dy', 'z', 'dz',
        'x_ctrl_states', 'y_ctrl_states', 'z_ctrl_states',
        'phi_ctrl_states', 'theta_ctrl_states', 'psi_ctrl_states']
states_idxs = np.arange(len(states_names))
names_idxs = {key: value for key, value in zip(states_names, states_idxs)}

# def parse_states(states: np.ndarray):

#     phi = states[0]; dphi = states[1]
#     theta = states[2]; dtheta = states[3]
#     psi = states[4]; dpsi = states[5]
#     x = states[6]; dx = states[7]
#     y = states[8]; dy = states[9]
#     z = states[10]; dz = states[11]
#     phi_ctrl_states = states[12]
#     theta_ctrl_states = states[13]
    
#     return phi, dphi, theta, dtheta, psi, dpsi, x, dx, y, dy, z, dz, \
#         phi_ctrl_states, theta_ctrl_states

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
        A: npt.ArrayLike, jx: Number, jy: Number, jz: Number, g: Number, mass: Number, 
        x_controller: Dict[str, Any], y_controller: Dict[str, Any], z_controller: Dict[str, Any],
        psi_controller: Dict[str, Any], phi_controller: Dict[str, Any], theta_controller: Dict[str, Any], 
        log_internals:bool=False):
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
        
        A = np.array(A, dtype=np.float64)
        self.Ainv = np.linalg.inv(A)
        
        self.ref_phi = ref_phi
        self.ref_theta = ref_theta

        # Configs
        is_instance(log_internals, bool)
        self.log_internals = log_internals
        self.internals = defaultdict(list)

        self.x_controller = get_ctrl_from_config(x_controller)[0]
        self.y_controller = get_ctrl_from_config(y_controller)[0]
        self.z_controller = get_ctrl_from_config(z_controller)[0]
        self.phi_controller = get_ctrl_from_config(phi_controller)[0]
        self.theta_controller = get_ctrl_from_config(theta_controller)[0]
        self.psi_controller = get_ctrl_from_config(psi_controller)[0]

    def compute(self, t: Number, xs: npt.ArrayLike, output_only:bool=False) -> Union[np.ndarray, dict]:
        # phi, dphi, theta, dtheta, psi, dpsi, x, dx, y, dy, z, dz = xs
        
        phi, dphi, theta, dtheta, psi, dpsi, x, dx, y, dy, z, dz, \
        x_ctrl_states, y_ctrl_states, z_ctrl_states, \
        phi_ctrl_states, theta_ctrl_states, psi_ctrl_states = xs 
        
        ref_z = self.ref_z(t)
        ref_dz = self.ref_dz(t)
        ref_ddz = self.ref_ddz(t)
        ref_x = self.ref_x(t)
        ref_dx = self.ref_dx(t)
        ref_ddx = self.ref_ddx(t)
        ref_y = self.ref_y(t)
        ref_dy = self.ref_dy(t)
        ref_ddy = self.ref_ddy(t)
        ref_psi = self.ref_psi(t)
        ref_dpsi = self.ref_dpsi(t)
        ref_ddpsi = self.ref_ddpsi(t)

        u_x = self.x_controller.output(t, refs=[ref_x, ref_dx, ref_ddx], input_x=[x, dx], x=[x_ctrl_states])
        u_y = self.y_controller.output(t, refs=[ref_y, ref_dy, ref_ddy], input_x=[y, dy], x=[x_ctrl_states])
        f = self.z_controller.output(t, refs=[ref_z, ref_dz, ref_ddz], input_x=[z, dz], x=[x_ctrl_states])
        m_z = self.psi_controller.output(t, refs=[ref_psi, ref_dpsi, ref_ddpsi], input_x=[psi, dpsi],
            x=[psi_ctrl_states])

        ref_phi = self.ref_phi(u_x, u_y, ref_psi, f, self.mass)
        u_phi = self.phi_controller.output(t, refs=[ref_phi], input_x=[phi], x=[phi_ctrl_states])

        ref_theta = self.ref_theta(u_x, u_y, ref_psi, ref_phi, f, self.mass)
        u_theta = self.theta_controller.output(t,refs=[ref_theta], input_x=[theta], x=[theta_ctrl_states])

        m_x = u_phi*self.jx
        m_y = u_theta*self.jy
        
        if xs.ndim == 1:
            f_M = np.array([f, m_x[0], m_y[0], m_z])
            fi = np.dot(self.Ainv, f_M)
        elif xs.ndim ==2:
            f_M = np.column_stack((f, m_x, m_y, m_z))
            fi = np.dot(self.Ainv, f_M.T)

        if output_only:
            controller_dxs = list()
            controller_dxs.append(self.x_controller.dx(t, x=x_ctrl_states,
                refs=[ref_x, ref_dx, ref_ddx], input_x=[x, dx]))
            controller_dxs.append(self.y_controller.dx(t, x=y_ctrl_states,
                refs=[ref_y, ref_dy, ref_ddy], input_x=[y, dy]))
            controller_dxs.append(self.z_controller.dx(t, x=z_ctrl_states,
                refs=[ref_z, ref_dz, ref_ddz], input_x=[z, dz]))
            controller_dxs.append(self.phi_controller.dx(t, x=phi_ctrl_states,
                refs=[ref_phi], input_x=[phi]))
            controller_dxs.append(self.theta_controller.dx(t, x=theta_ctrl_states,
                refs=[ref_theta], input_x=[theta]))
            controller_dxs.append(self.psi_controller.output(t, x=psi_ctrl_states, 
                refs=[ref_psi, ref_dpsi, ref_ddpsi], input_x=[psi, dpsi]))
            controller_dxs = np.concatenate(controller_dxs, axis=0)

            return fi, controller_dxs
        else:
            res = dict(t=t, e_phi=e_phi, e_theta=e_theta, e_psi=e_psi,
                    e_z=e_z, e_y=e_y, e_x=e_x,
                    ref_z=ref_z, ref_dz=ref_dz, ref_ddz=ref_ddz, 
                    ref_x=ref_x, ref_dx=ref_dx, ref_ddx=ref_ddx, 
                    ref_y=ref_y, ref_dy=ref_dy, ref_ddy=ref_ddy,
                    ref_psi=ref_psi, ref_dpsi=ref_dpsi, ref_ddpsi=ref_ddpsi,
                    ref_theta=ref_theta,ref_phi=ref_phi, 
                    u_z=u_z, u_y=u_y, u_x=u_x, 
                    u_phi=u_phi, u_theta=u_theta, u_psi=u_psi, 
                    f=f, m_x=m_x, m_y=m_y, m_z=m_z,
                    f1=fi[0], f2=fi[1], f3=fi[2], f4=fi[3],
                    psi=psi, theta=theta, phi=phi, x=x, y=y, z=z,
                    dpsi=dpsi, dtheta=dtheta, dphi=dphi, dx=dx, dy=dy, dz=dz)
            return res

    def output(self, t: Number, xs: npt.ArrayLike) -> np.ndarray:
        if self.log_internals:
            res = self.compute(t, xs, output_only=False)
            self._log_values(**res)
            fi = np.array([res['f1'], res['f2'], res['f3'], res['f4']])
        else:
            fi,controller_dxs = self.compute(t, xs, output_only=True)
        
        return fi, controller_dxs
    
    def _log_values(self, **kwargs):
        for key, value in kwargs.items():
            self.internals[key].append(value)
    
class Drone(DynamicSystem):

    def __init__(self, jx: Number, jy: Number, jz: Number,
        g: Number, mass: Number, A:npt.ArrayLike):
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
    
    def dx(self, t: Number, xs: npt.ArrayLike, fi: npt.ArrayLike) -> np.ndarray:
        f, m_x, m_y, m_z = np.dot(self.A, fi)
        f_over_m = f/self.mass
        phi, dphi, theta, dtheta, psi, dpsi, x, dx, y, dy, z, dz, _, _ = xs
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

    def dx(self, t, xs) -> np.ndarray:
        controler_fi, controller_dxs = self.controller.output(t, xs)
        drone_dxs = self.drone(t, xs, controler_fi)
        dxs = np.concatenate([drone_dxs, controller_dxs], axis=0)
        return dxs
    
    def output(self, t, xs):
        return np.array(xs)

def ref_phi(ref_ddx: Number, ref_ddy: Number, ref_psi:Number, f: Number, mass:Number) -> np.float64:
    cpsi = np.cos(ref_psi)
    spsi = np.sin(ref_psi)
    ref_phi = -np.arcsin((mass/f)*(-spsi*ref_ddx + cpsi*ref_ddy))
    return ref_phi

def ref_theta(ref_ddx: Number, ref_ddy: Number, ref_psi: Number, ref_phi: Number, f: Number, mass:Number) -> np.float64:
    cpsi = np.cos(ref_psi)
    spsi = np.sin(ref_psi)
    cphi = np.cos(ref_phi)
    ref_theta = np.arcsin((1/cphi)*(mass/f)*(cpsi*ref_ddx + spsi*ref_ddy))
    return ref_theta