import os
import json
import numpy as np
import pandas as pd
import dynamo.drone as drone_models
from scipy.integrate import solve_ivp
from datetime import datetime
from dynamo.config import parse_config_dict


def dump_simulation(sim_bunch, config_dict):
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
        json.dump(config_dict, json_file)
    sim_df.to_csv(
        os.path.join(output_dir, "sim_out.csv")
    )


gravity = 10
drone_mass = 1
jx = 1
jy = 1
jz = 1
time_range = (0, 10)     # Seconds
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
    0,  # z0,
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

psifactor = 1e-1
zfactor = 1e-1
phifactor = 1e-1
thetafactor = 1e-1
xfactor = 1e-4
yfactor = 1e-4

ws = 0.1

config_dict = {
    "constructor": "dynamo.drone.ControledDrone",
    "args": [],
    "kwargs": {
        "controller": {
            "constructor": "dynamo.drone.DroneController",
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
                        "x": f"sin({ws}*t)",
                        "y": f"sin({ws}*t)",
                        "z": "Heaviside(t, 1)",
                        "psi": "0*Heaviside(t, 1)",
                        "n_derivatives": 2
                    }
                },
                "gains": {
                    "constructor": "dynamo.base.Bunch",
                    "args": [],
                    "kwargs": {
                        "kp_x": 40*xfactor,
                        "kd_x": 1000*xfactor,
                        "kp_y": 40*yfactor,
                        "kd_y": 1000*yfactor,
                        "kp_z": 2*zfactor,
                        "kd_z": 1*zfactor,
                        "kp_theta": 4*thetafactor,
                        "kd_theta": 10*thetafactor,
                        "kp_phi": 4*phifactor,
                        "kd_phi": 10*phifactor,
                        "kp_psi": 2*psifactor,
                        "kd_psi": 1*psifactor,
                    }
                }
            }
        },
        "drone": {
            "constructor": "dynamo.drone.Drone",
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
    method="RK45"
)

print(f"Simulation outputted status {res.status}.\n\"{res.message}\"")

# Saving output
sim_out = np.concatenate((res.t.reshape(1, -1), res.y), axis=0).T
sim_out = pd.DataFrame(sim_out, columns=['t']+drone_models.STATES_NAMES)
sim_bunch = controled_drone.output(
    sim_out['t'].values,
    sim_out[drone_models.STATES_NAMES].values.T
)

dump_simulation(sim_bunch, config_dict)
print('End')
