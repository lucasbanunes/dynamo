import os
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from datetime import datetime
from dynamo.config import parse_config_dict
from dynamo.drone.plotting import time_plot, plot2d
from dynamo.drone.utils import STATES_NAMES


def dump_simulation(sim_bunch, config_dict, refs):
    exec_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    sim_bunch.pop("vector")
    sim_bunch.pop("f_M")
    for i, ifi in enumerate(sim_bunch.fi):
        sim_bunch[f"f{i+1}"] = ifi
    sim_bunch.pop("fi")
    sim_df = sim_bunch.to_frame()
    dir_name = f"{exec_time}_drone_simulation_output"
    output_dir = os.path.join(
        "output",
        dir_name
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    json_path = os.path.join(output_dir, "config.json")
    with open(json_path, "w") as json_file:
        json.dump(config_dict, json_file, indent=4)
    sim_df.to_csv(
        os.path.join(output_dir, "sim_out.csv")
    )
    time_plot(['x', 'y', 'z'], sim_df,
              title='Position response',
              filepath=os.path.join(output_dir, 'position_plot.png'),
              diff_order=0)
    time_plot(['x', 'y', 'z'], sim_df,
              title='Linear speed response',
              filepath=os.path.join(output_dir, 'linear_speed_plot.png'),
              diff_order=1)
    time_plot(['x', 'y', 'z'], sim_df,
              title='Linear acceleration response',
              filepath=os.path.join(output_dir, 'linear_accel_plot.png'),
              diff_order=2)
    time_plot(['phi', 'theta', 'psi'], sim_df,
              title='Orientation response',
              filepath=os.path.join(output_dir, 'orientation_plot.png'),
              diff_order=0)
    time_plot(['phi', 'theta', 'psi'], sim_df,
              title='Angular Speed response',
              filepath=os.path.join(output_dir, 'angular_speed_plot.png'),
              diff_order=1)
    time_plot(['phi', 'theta', 'psi'], sim_df,
              title='Angular acceleration response',
              filepath=os.path.join(output_dir, 'angular_accel_plot.png'),
              diff_order=2)
    plot2d('x', 'y', sim_df, diff_order=0,
           filepath=os.path.join(output_dir, 'xy_plot.png'))

    refs_filepath = os.path.join(output_dir, 'refs.json')
    ref_dict = {
        key: value for key, value in refs.items()
        if key.startswith('sym')
    }
    with open(refs_filepath, 'w') as json_file:
        json.dump(ref_dict, json_file, indent=4)


gravity = 10
drone_mass = 10.920
jx = 0.4417
jy = 0.4417
jz = 0.7420
time_range = (0, 120)     # Seconds
initial_states = [
    0,  # phi0,
    0,  # dphi0,
    0,  # theta0,
    0,  # dtheta0,
    0,  # psi0,
    0,  # dpsi0,
    0,  # x0,
    0,  # dx0,
    0,  # y0,
    0,  # dy0,
    10,  # z0,
    0,  # dz0
]

d = 1
ctau = 1
s = np.array([0.1, 0.7, 0.3, 0.5], dtype=np.float64)
A = np.array([
    np.ones(4),
    [0, d, 0, -d],
    [-d, 0, d, 0],
    ctau*s,
], dtype=np.float64)

# Previous setup
# psifactor = 1e-1
# zfactor = 1e-1
# phifactor = 1e-1
# thetafactor = 1e-1
# xfactor = 1e-4
# yfactor = 1e-4
# kp_x = 40*xfactor
# kd_x = 1000*xfactor
# kp_y = 40*yfactor
# kd_y = 1000*yfactor
# kp_z = 2*zfactor
# kd_z = 1*zfactor
# kp_theta = 4*thetafactor
# kd_theta = 10*thetafactor
# kp_phi = 4*phifactor
# kd_phi = 10*phifactor
# kp_psi = 2*psifactor
# kd_psi = 1*psifactor

# Jaccoud setup
yawfactor = 1e-1/2/2
zfactor = 1e-1/2/2
rollfactor = 1e-1*1*10*3/2/2
xfactor = 1e-4/2/2
yfactor = 1e-4/2/2
pitchfactor = 1e-1*1*10*3/2/2
kp_psi = 2*yawfactor
kd_psi = 10*yawfactor
kp_z = 2*zfactor
kd_z = 10*zfactor
kp_x = 40*xfactor
kd_x = 1000*xfactor*10
kp_y = 40*yfactor
kd_y = 1000*yfactor*10
kp_theta = 40*pitchfactor*5/5/2
kd_theta = 10*pitchfactor/2
kp_phi = kp_theta
kd_phi = kd_theta

ws = {
    'z': 2*np.pi/100,
    'x': 2*np.pi/100,
    'y': 2*np.pi/100,
    'psi': 2*np.pi/100
}
amp = {
    'z': 1,
    'psi': 45*np.pi/180,
    'x': 4,
    'y': 4
}
dc = {
    'z': 10,
    'x': 0,
    'y': 0,
    'psi': 0*45*np.pi/180
}


config_dict = {
    "constructor": "dynamo.drone.models.ControledDrone",
    "args": [],
    "kwargs": {
        "controller": {
            "constructor": "dynamo.drone.controllers.DroneController",
            "args": [],
            "kwargs": {
                "mass": drone_mass,
                "jx": jx,
                "jy": jy,
                "jz": jz,
                "A": A.tolist(),
                "g": gravity,
                "refs": {
                    "constructor": "dynamo.signal.TimeSignal",
                    "args": [],
                    "kwargs": {
                        "x": f"{amp['x']}*sin({ws['x']}*t) + {dc['x']}",
                        "y": f"{amp['y']}*cos({ws['y']}*t) + {dc['y']}",
                        "z": f"{amp['z']}*sin({ws['y']}*t) + {dc['z']}",
                        "psi": f"{amp['psi']}*sin({ws['y']}*t) + {dc['psi']}",
                        "n_derivatives": 2
                    }
                },
                "gains": {
                    "constructor": "dynamo.base.Bunch",
                    "args": [],
                    "kwargs": {
                        "kp_x": kp_x,
                        "kd_x": kd_x,
                        "kp_y": kp_y,
                        "kd_y": kd_y,
                        "kp_z": kp_z,
                        "kd_z": kd_z,
                        "kp_theta": kp_theta,
                        "kd_theta": kd_theta,
                        "kp_phi": kp_phi,
                        "kd_phi": kd_phi,
                        "kp_psi": kp_psi,
                        "kd_psi": kd_psi,
                    }
                }
            }
        },
        "drone": {
            "constructor": "dynamo.drone.models.Drone",
            "args": [],
            "kwargs": {
                "jx": jx,
                "jy": jy,
                "jz": jz,
                "g": gravity,
                "mass": drone_mass,
                "A": A.tolist()
            }
        }
    }
}

controled_drone = parse_config_dict(config_dict, True)
controller = controled_drone.controller
res = solve_ivp(
    controled_drone,
    t_span=time_range,
    y0=initial_states,
    method="RK45",
    max_step=1e-1
)

print(f"Simulation outputted status {res.status}.\n\"{res.message}\"")

# Saving output
sim_out = np.concatenate((res.t.reshape(1, -1), res.y), axis=0).T
sim_out = pd.DataFrame(sim_out, columns=['t']+STATES_NAMES)
sim_bunch = controled_drone.output(
    sim_out['t'].values,
    sim_out[STATES_NAMES].values.T
)

dump_simulation(sim_bunch, config_dict, controled_drone.controller.refs)
print('End')
