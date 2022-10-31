import numpy as np
import pandas as pd
import dynamo.drone as drone_models
from scipy.integrate import solve_ivp
from datetime import datetime

gravity = 10
drone_mass = 1
time_range = (0,10)     # Seconds
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
initial_states = [phi0, dphi0, theta0, dtheta0, psi0, dpsi0, x0, dx0, y0, dy0, z0, dz0, 0, 0]

d=1
ctau=1
s = np.array([0.1,0.7,0.3,0.5], dtype=np.float64)
A = np.array([
    np.ones(4),
    [0, d, 0, -d],
    [-d, 0, d, 0],
    ctau*s,
], dtype=np.float64)

controler_kwargs = dict(
    ref_x = lambda t: t**0,
    ref_dx = lambda t: t-t,
    ref_ddx = lambda t: t-t,
    ref_y = lambda t: t**0,
    ref_dy = lambda t: t-t,
    ref_ddy = lambda t: t-t,
    ref_z = lambda t: t**0,
    ref_dz = lambda t: t-t,
    ref_ddz = lambda t: t-t,
    ref_psi = lambda t: t-t,
    ref_dpsi = lambda t: t-t,
    ref_ddpsi = lambda t: t-t,
    kp_x = 1,
    kd_x= 1,
    kp_y = 1,
    kd_y= 1,
    kp_z = 3,
    kd_z= 3,
    kp_phi = 1,
    kd_phi = 1,
    kp_theta = 1,
    kd_theta = 1,
    kp_psi = 1,
    kd_psi = 1,
    g = gravity,
    mass = drone_mass,
    A = A,
    jx=1,
    jy=1,
    jz=1,
    log_internals=False
)

drone_kwargs = dict(
    jx=1,
    jy=1,
    jz=1,
    g = gravity,
    mass = drone_mass,
    A=A
)

controller = drone_models.DroneController(**controler_kwargs)
drone = drone_models.Drone(**drone_kwargs)
controled_drone = drone_models.ControledDrone(controller, drone)
res = solve_ivp(controled_drone, t_span=time_range, y0=initial_states, max_step=1e-2)

print(f'Simulation outputted status {res.status}. "{res.message}"')

# Saving output
exec_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
filename = 'test_drone_sim_out.csv'# f'{exec_time}_drone_sim_out.npz'
sim_out = np.concatenate((res.t.reshape(1,-1), res.y),axis=0).T
sim_out = pd.DataFrame(sim_out,columns=['t']+drone_models.states_names)
ctrl_internals = controller.compute(sim_out['t'].values, sim_out[drone_models.states_names].values)
for key, value in ctrl_internals.items():
    if key != 't':
        sim_out[key] = value
sim_out.to_csv(filename)

internals_df = pd.DataFrame.from_dict(controller.internals)
internals_df.to_csv('test_internals_' + filename)

print('End')