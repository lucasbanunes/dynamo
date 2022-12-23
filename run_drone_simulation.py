import numpy as np
import pandas as pd
import dynamo.drone as drone_models
from scipy.integrate import solve_ivp
from datetime import datetime


gravity = 10
drone_mass = 1
time_range = (0,500)     # Seconds
phi0 = 0.01
dphi0 = 0
theta0 = 0.01
dtheta0 = 0
psi0 = 0
dpsi0 = 0
x0 = 0
dx0 = 0
y0 = 0
dy0 = 0
z0 = 0
dz0 = 0
initial_states = [phi0, dphi0, theta0, dtheta0, psi0, dpsi0, x0, dx0, y0, dy0, z0, dz0]

d=1
ctau=1
s = np.array([0.1,0.7,0.3,0.5], dtype=np.float64)
A = np.array([
    np.ones(4),
    [0, d, 0, -d],
    [-d, 0, d, 0],
    ctau*s,
], dtype=np.float64)
psifactor = 1e-1
zfactor = 1e-1
phifactor = 1e-1
thetafactor = 1e-1
xfactor = 1e-4
yfactor = 1e-4

ws = 0.1
controller_kwargs = dict(
    ref_x = lambda t: np.sin(ws*t),
    ref_dx = lambda t: ws*np.cos(ws*t),
    ref_ddx = lambda t: -(ws**2)*np.sin(ws*t),
    ref_y = lambda t: np.sin(ws*t),
    ref_dy = lambda t: ws*np.cos(ws*t),
    ref_ddy = lambda t: -(ws**2)*np.sin(ws*t),
    ref_z = lambda t: t**0,
    ref_dz = lambda t: t-t,
    ref_ddz = lambda t: t-t,
    ref_psi = lambda t: t-t,
    ref_dpsi = lambda t: t-t,
    ref_ddpsi = lambda t: t-t,
    kp_x = 40*xfactor,
    kd_x = 1000*xfactor,
    kp_y = 40*yfactor,
    kd_y = 1000*yfactor,
    kp_z = 2*zfactor,
    kd_z = 1*zfactor,
    kp_phi = 4*phifactor,
    kd_phi = 10*phifactor,
    kp_theta = 4*thetafactor,
    kd_theta = 10*thetafactor,
    kp_psi = 2*psifactor,
    kd_psi = 1*psifactor,
    g = gravity,
    mass = drone_mass,
    theta_sat = np.deg2rad(15),
    phi_sat = np.deg2rad(15),
    A = A,
    jx = 1,
    jy = 1,
    jz = 1,
    log_internals = False,
    direction_ctrl_strat = 'almost_derivative'
)

drone_kwargs = dict(
    jx=1,
    jy=1,
    jz=1,
    g = gravity,
    mass = drone_mass,
    A=A
)

filename = f'drone_sim_out_{controller_kwargs["direction_ctrl_strat"]}.csv'
controller = drone_models.DroneController(**controller_kwargs)
drone = drone_models.Drone(**drone_kwargs)
controled_drone = drone_models.ControledDrone(controller, drone)
res = solve_ivp(controled_drone, t_span=time_range, y0=initial_states, method='RK45')

print(f'Simulation outputted status {res.status}. "{res.message}"')

# Saving output
exec_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')# f'{exec_time}_drone_sim_out.npz'
sim_out = np.concatenate((res.t.reshape(1,-1), res.y),axis=0).T
sim_out = pd.DataFrame(sim_out,columns=['t']+drone_models.states_names)
ctrl_internals = controller.compute(sim_out['t'].values, sim_out[drone_models.states_names].values.T)
for key, value in ctrl_internals.items():
    if key != 't' and (not key in drone_models.states_names) and value.ndim == 1:
        sim_out[key] = value
sim_out.to_csv(filename)

# internals_df = pd.DataFrame.from_dict(controller.internals)
# internals_df.to_csv('test_internals_' + filename)

print('End')