import os
import json
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from datetime import datetime
from dynamo.config import parse_config_dict
from dynamo.drone.plotting import time_plot, plot2d
from dynamo.base import Bunch


def dump_simulation(sim_bunch, config_dict, refs):
    exec_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    sim_bunch.pop("vector")
    sim_bunch.pop("f_M")
    for i, ifi in enumerate(sim_bunch.fi):
        sim_bunch[f"f{i+1}"] = ifi
    sim_bunch.pop("fi")
    sim_df = sim_bunch.to_frame()
    dir_name = f"{exec_time}_speed_ctrl_drone_sim_out"
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
    time_plot(
        sim_df,
        main_y=[
            ["px", "e_px", "ref_px"],
            ["py", "e_py", "ref_py"],
            ["pz", "e_pz", "ref_pz"],
        ],
        sec_y=[
            [],
            [],
            ["f"],
        ],
        title="Position response",
        filepath=os.path.join(output_dir, 'position_plot.png')
    )
    time_plot(
        sim_df,
        main_y=[
            ["vx", "e_vx", "u_x", "ref_vx"],
            ["vy", "e_vy", "u_y", "ref_vy"],
            ["vz", "e_vz", "u_z", "ref_vz"],
        ],
        title="Linear Speed response",
        filepath=os.path.join(output_dir, 'linear_speed_plot.png')
    )
    time_plot(
        sim_df,
        main_y=[
            ["ax", "ref_ax"],
            ["ay", "ref_ay"],
            ["az", "ref_az"],
        ],
        title="Linear Acceleration response",
        filepath=os.path.join(output_dir, 'linear_accel_plot.png')
    )
    time_plot(
        sim_df,
        main_y=[
            ["theta", "e_theta", "ref_theta"],
            ["phi", "e_phi", "ref_phi"],
            ["psi", "e_psi", "ref_psi"],
        ],
        sec_y=[
            ["m_x"],
            ["m_y"],
            ["m_z"],
        ],
        title="Orientation response",
        filepath=os.path.join(output_dir, 'orientation_plot.png')
    )
    time_plot(
        sim_df,
        main_y=[
            ["vtheta", "e_vtheta", "u_theta", "ref_vtheta"],
            ["vphi", "e_vphi", "u_phi", "ref_vphi"],
            ["vpsi", "e_vpsi", "u_psi", "ref_vpsi"],
        ],
        title="Angular Speed response",
        filepath=os.path.join(output_dir, 'angular_speed_plot.png')
    )
    time_plot(
        sim_df,
        main_y=[
            ["atheta", "ref_atheta"],
            ["aphi", "ref_aphi"],
            ["apsi", "ref_apsi"],
        ],
        title="Angular Acceleration response",
        filepath=os.path.join(output_dir, 'angular_accel_plot.png')
    )
    plot2d("px", "py", sim_df,
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
time_range = (0, 10)     # Seconds
states_names = [
    "phi",
    "vphi",
    "theta",
    "vtheta",
    "psi",
    "vpsi",
    "px",
    "vx",
    "py",
    "vy",
    "pz",
    "vz",
    "ie_vx",
    "ie_vy",
    "ie_vz",
    "ie_vpsi"
]
dstates_names = [
    "vphi",
    "aphi",
    "vtheta",
    "atheta",
    "vpsi",
    "apsi",
    "vx",
    "ax",
    "vy",
    "ay",
    "vz",
    "az",
    "e_vx",
    "e_vy",
    "e_vz",
    "e_vpsi"
]
initial_states = [
    0,      # phi0,
    0,      # vphi0,
    0,      # theta0,
    0,      # vtheta0,
    0,      # psi0,
    0,      # vpsi0,
    0,      # px0,
    0,      # vx0,
    0,      # py0,
    0,      # vy0,
    10,     # pz0,
    0,      # vz0
    0,      # ie_vx0
    0,      # ie_vy0
    0,      # ie_vz0
    0       # ie_vpsi0
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

pitchfactor = 1e-1*1*10*3/2/2
kp_theta = 40*pitchfactor*5/5/2
kd_theta = 10*pitchfactor/2
kp_phi = kp_theta
kd_phi = kd_theta

Nx = 5
Ny = 5
Nz = 5
Npsi = 5

apx = 0.25*4/(Nx+1)
apy = 0.25*4/(Ny+1)
apz = 0.25*4/(Nz+1)
appsi = 0.25*4/(Npsi+1)

kd_x = apx
kd_y = apy
kd_z = apz
kd_psi = appsi

kd_x = (Nx+1)*apx
ki_x = Nx*apx*apx
kd_y = (Ny+1)*apy
ki_y = Ny*apy*apy
kd_z = (Nz+1)*apz
ki_z = Nz*apz*apz
kd_psi = (Npsi+1)*appsi
ki_psi = Npsi*appsi*appsi
kp_x = 0
kp_y = 0
kp_z = 0
kp_psi = 0

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
    "constructor": "dynamo.models.ControlledSystem",
    "args": [],
    "kwargs": {
        "controller": {
            "constructor": "dynamo.drone.controllers.SpeedDroneController",
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
                        "px": f"{amp['x']}*sin({ws['x']}*t) + {dc['x']}",
                        "vx": "diff(px, t)",
                        "ax": "diff(vx, t)",
                        "py": f"{amp['y']}*cos({ws['y']}*t) + {dc['y']}",
                        "vy": "diff(py, t)",
                        "ay": "diff(ay, t)",
                        "pz": f"{amp['z']}*sin({ws['y']}*t) + {dc['z']}",
                        "vz": "diff(pz, t)",
                        "az": "diff(az, t)",
                        "psi": f"{amp['psi']}*sin({ws['y']}*t) + {dc['psi']}",
                        "vpsi": "diff(psi, t)",
                        "apsi": "diff(vpsi, t)",
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
                        "ki_x": ki_x,
                        "ki_y": ki_y,
                        "ki_z": ki_z,
                        "ki_psi": ki_psi
                    }
                }
            }
        },
        "system": {
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
        },
        "states_names": states_names,
        "dstates_names": dstates_names
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
sim_out = pd.DataFrame(sim_out, columns=['t']+states_names)
sim_bunch_kwargs = {state_name: sim_out[state_name].values
                    for state_name in states_names}
sim_bunch_kwargs["vector"] = sim_out[states_names].values
sim_bunch = Bunch(**sim_bunch_kwargs)
sim_bunch = controled_drone.output(
    t=sim_out['t'].values,
    data=sim_bunch
)

dump_simulation(sim_bunch, config_dict, controled_drone.controller.refs)
print('End')
