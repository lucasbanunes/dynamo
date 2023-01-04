import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dynamo.drone as drone_models
from scipy.integrate import solve_ivp
from datetime import datetime
from dynamo.config import parse_config_dict

vars_to_plot = ['x', 'y', 'z', 'phi', 'theta', 'psi']
var_labels = {
    'x': 'x', 'y': 'y', 'z': 'z',
    'theta': '\\theta', 'phi': '\\phi', 'psi': '\\psi',
    'f': 'f', 'm_x': 'm_x', 'm_y': 'm_y', 'm_z': 'm_z'
}

linearization_inputs = {
    'x': None,
    'y': None,
    'z': 'f',
    'phi': 'm_x',
    'theta': 'm_y',
    'psi': 'm_z'
}


def plot_base_ax(var, sim_out, ax, xlim=None, ylim=None):
    var_label = var_labels[var]
    lines = list()
    lines += ax.plot(sim_out['t'],
                     sim_out[var],
                     label=f'${var_label}$',
                     color='C0')
    lines += ax.plot(sim_out['t'],
                     sim_out[f'u_{var}'],
                     label=f'${var_label}$',
                     linestyle='--', color='C1')
    lines += ax.plot(sim_out['t'],
                     sim_out[f'ref_{var}'],
                     label=f'$r_{var_label}$',
                     linestyle='--', color='k')
    lines += ax.plot(sim_out['t'],
                     sim_out[f'e_{var}'],
                     label=f'$e_{var_label}$',
                     linestyle='--', color='C2')
    lin_var = linearization_inputs[var]
    if lin_var:
        lin_var_label = var_labels[lin_var]
        twinx = ax.twinx()
        lines += twinx.plot(sim_out['t'],
                            sim_out[lin_var],
                            label=f'${lin_var_label}$',
                            linestyle='--', color='C3')

    labels = [illine.get_label() for illine in lines]
    ax.legend(lines, labels)
    ax.set(xlim=xlim, ylim=ylim)
    ax.grid()
    return ax


def plot_dax(var, sim_out, ax, xlim=None, ylim=None):
    var_label = var_labels[var]
    lines = list()
    lines += ax.plot(sim_out['t'],
                     sim_out[var],
                     label=f'$\dot{{{var_label}}}$',
                     color='C0')
    lines += ax.plot(sim_out['t'],
                     sim_out[f'dref_{var}'],
                     label=f'$\dot{{r_{var_label}}}$',
                     linestyle='--', color='k')
    lines += ax.plot(sim_out['t'],
                     sim_out[f'de_{var}'],
                     label=f'$\dot{{e_{var_label}}}$',
                     linestyle='--', color='C2')
    labels = [illine.get_label() for illine in lines]
    ax.legend(lines, labels)
    ax.set(xlim=xlim, ylim=ylim)
    ax.grid()
    return ax


def plot_ddax(var, sim_out, ax, xlim=None, ylim=None):
    var_label = var_labels[var]
    lines = list()
    lines += ax.plot(sim_out['t'],
                     sim_out[var],
                     label=f'$\ddot{{{var_label}}}$',
                     color='C0')
    lines += ax.plot(sim_out['t'],
                     sim_out[f'ddref_{var}'],
                     label=f'$\ddot{{r_{var_label}}}$',
                     linestyle='--', color='k')
    labels = [illine.get_label() for illine in lines]
    ax.legend(lines, labels)
    ax.set(xlim=xlim, ylim=ylim)
    ax.grid()
    return ax


def time_plot(variable: str,
              sim_out: pd.DataFrame,
              filename: str = None,
              xlim=None,
              ylim=None):

    fig, axes = plt.subplots(3, 1, figsize=(19.20, 10.80))
    base_ax, dax, ddax = axes
    plot_base_ax(variable, sim_out, base_ax, xlim, ylim)
    plot_dax(variable, sim_out, dax, xlim, ylim)
    plot_ddax(variable, sim_out, ddax, xlim, ylim)
    var_label = var_labels[variable]
    fig.suptitle(f'${var_label}$ time plot')
    fig.tight_layout()
    if filename:
        fig.savefig(filename)


def state_space_plot(x, y, sim_out, filename=None):
    fig, ax = plt.subplots(1, 1, figsize=(19.20, 10.80))
    ax.grid()
    ax.plot(sim_out[x], sim_out[y])
    ax.set(xlabel=x, ylabel=y)
    x_label = var_labels[x]
    y_label = var_labels[y]
    fig.suptitle(f'${x_label}$ X ${y_label}$')
    fig.tight_layout()

    if filename:
        fig.savefig(filename, dpi=72, transparent=False, facecolor='white')


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
        json.dump(config_dict, json_file)
    sim_df.to_csv(
        os.path.join(output_dir, "sim_out.csv")
    )
    for var2plot in vars_to_plot:
        time_plot(var2plot,
                  sim_df,
                  filename=os.path.join(
                    output_dir, f'{var2plot}_plot.png'
                  ))
    state_space_plot('x', 'y', sim_df,
                     filename=os.path.join(output_dir, 'xy_plot.png'))

    refs_filepath = os.path.join(output_dir, 'refs.json')
    ref_dict = {
        key: value for key, value in refs.items()
        if key.startswith('sym')
    }
    with open(refs_filepath, 'w') as json_file:
        json.dump(ref_dict, json_file, indent=4)


gravity = 10
drone_mass = 1
jx = 1
jy = 1
jz = 1
time_range = (0, 60)     # Seconds
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
    method="RK45",
    max_step=1e-1
)

print(f"Simulation outputted status {res.status}.\n\"{res.message}\"")

# Saving output
sim_out = np.concatenate((res.t.reshape(1, -1), res.y), axis=0).T
sim_out = pd.DataFrame(sim_out, columns=['t']+drone_models.STATES_NAMES)
sim_bunch = controled_drone.output(
    sim_out['t'].values,
    sim_out[drone_models.STATES_NAMES].values.T
)

dump_simulation(sim_bunch, config_dict, controled_drone.controller.refs)
print('End')
