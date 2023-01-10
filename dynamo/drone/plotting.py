from typing import Sequence, Tuple
from numbers import Number
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

var_labels = {
    "px": "$p_x$",
    "e_px": "$e_{{p_x}}$",
    "ref_px": "$r_{p_x}$",
    "py": "$p_y$",
    "e_py": "$e_{{p_y}}$",
    "ref_py": "$r_{p_y}$",
    "pz": "$p_z$",
    "e_pz": "$e_{{p_z}}$",
    "ref_pz": "$r_{p_z}$",
    "vx": "$v_x$",
    "e_vx": "$e_{{v_x}}$",
    "ref_vx": "$r_{{v_x}}$",
    "ax": "$a_x$",
    "e_ax": "$e_{{a_x}}$",
    "ref_ax": "$r_{{a_x}}$",
    "vy": "$v_y$",
    "e_vy": "$e_{{v_y}}$",
    "ref_vy": "$r_{{v_y}}$",
    "ay": "$a_y$",
    "e_ay": "$e_{{a_y}}$",
    "ref_ay": "$r_{{a_y}}$",
    "vz": "$v_z$",
    "e_vz": "$e_{{v_z}}$",
    "ref_vz": "$r_{{v_z}}$",
    "az": "$a_z$",
    "e_az": "$e_{{a_z}}$",
    "ref_az": "$r_{{a_z}}$",
    "phi": "$\\phi$",
    "e_phi": "$e_{{\\phi}}$",
    "ref_phi": "$r_{{\\phi}}$",
    "vphi": "$v_{{\\phi}}$",
    "e_vphi": "$e_{{v_{{\\phi}}}}$",
    "ref_vphi": "$r_{{v_{{\\phi}}}}$",
    "aphi": "$a_{{\\phi}}$",
    "e_aphi": "$e_{{a_{{\\phi}}}}$",
    "ref_aphi": "$r_{{a_{{\\phi}}}}$",
    "theta": "$\\theta$",
    "e_theta": "$e_{{\\theta}}$",
    "ref_theta": "$r_{{\\theta}}$",
    "vtheta": "$v_{{\\theta}}$",
    "e_vtheta": "$e_{{v_{{\\theta}}}}$",
    "ref_vtheta": "$r_{{v_{{\\theta}}}}$",
    "atheta": "$a_{{\\theta}}$",
    "e_atheta": "$e_{{a_{{\\theta}}}}$",
    "ref_atheta": "$r_{{a_{{\\theta}}}}$",
    "psi": "$\\psi$",
    "e_psi": "$e_{{\\psi}}$",
    "ref_psi": "$r_{{\\psi}}$",
    "vpsi": "$v_{{\\psi}}$",
    "e_vpsi": "$e_{{v_{{\\psi}}}}$",
    "ref_vpsi": "$r_{{v_{{\\psi}}}}$",
    "apsi": "$a_{{\\psi}}$",
    "e_apsi": "$e_{{a_{{\\psi}}}}$",
    "ref_apsi": "$r_{{a_{{\\psi}}}}$",
    "u_x": "$u_x$",
    "u_y": "$u_y$",
    "u_z": "$u_z$",
    "u_phi": "$u_{{\\phi}}$",
    "u_theta": "$u_{{\\theta}}$",
    "u_psi": "$u_{{\\psi}}$",
    "f": "$f$",
    "m_x": "$m_x$",
    "m_y": "$m_y$",
    "m_z": "$m_z$"
}

SAVEFIG_DEFAULTS = dict(
    dpi=72, transparent=False, facecolor='white'
)


def get_final_label(label: str, diff_order: int = 0) -> str:
    if diff_order:
        d_diffs = diff_order*"d"
        final_label = f"$\\{d_diffs}ot{{{label}}}$"
    else:
        final_label = f"${label}$"

    return final_label


def time_plot(sim_out: pd.DataFrame,
              main_y: Sequence[str],
              sec_y: Sequence[str] = None,
              title: str = None,
              filepath: str = None,
              xlim: Tuple[Number, Number] = None,
              ) -> Tuple[Figure, Axes]:
    n_axes = len(main_y)
    fig, axes = plt.subplots(
        n_axes, 1,
        figsize=(19.20, 10.80),
        sharex=True
    )
    if sec_y is None:
        sec_y = [list() for _ in range(n_axes)]
    for i, ax, imain_y, isec_y in zip(range(n_axes), axes, main_y, sec_y):
        lines = list()
        n_plotted = 0
        ax.grid()
        main_ylabels = list()
        for main_var in imain_y:
            var_label = var_labels[main_var]
            lines += ax.plot(
                sim_out["t"],
                sim_out[main_var],
                label=var_label,
                color=f"C{n_plotted}"
            )
            n_plotted += 1
            main_ylabels.append(var_label)
        sec_ylabels = list()
        for sec_var in isec_y:
            var_label = var_labels[sec_var]
            twinx = ax.twinx()
            lines += twinx.plot(
                sim_out["t"],
                sim_out[sec_var],
                label=var_label,
                color=f"C{n_plotted}"
            )
            n_plotted += 1
            sec_ylabels.append(var_label)

        legend_labels = [iline.get_label() for iline in lines]
        ax.legend(lines, legend_labels, fontsize="large")
        main_ylabel = ", ".join(main_ylabels)
        ax.set_ylabel(main_ylabel, fontsize="large")
        is_last_ax = i < n_axes-1
        if is_last_ax:
            # Removes the xaxis ticks if not on last axis
            ax.set(xticklabels=[])
        else:
            ax.set_xlabel("Time (seconds)", fontsize="large")
            ax.set(xlabel="Time (seconds)")

        if isec_y:
            sec_ylabel = ", ".join(sec_ylabels)
            twinx.set_ylabel(sec_ylabel, fontsize="large")

    if title:
        fig.suptitle(title, fontsize='x-large')
    fig.tight_layout()

    if filepath:
        fig.savefig(filepath, **SAVEFIG_DEFAULTS)

    return fig, axes


def plot2d(x: str, y: str,
           sim_out: pd.DataFrame,
           filepath: str = None
           ) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(1, 1, figsize=(19.20, 10.80))
    ax.grid()
    x_label = var_labels[x]
    y_label = var_labels[y]
    ax.plot(sim_out[x], sim_out[y])
    ax.set_xlabel(x_label, fontsize="large")
    ax.set_ylabel(y_label, fontsize="large")

    ref_x_name = f"ref_{x}"
    ref_y_name = f"ref_{y}"
    has_x_ref = ref_x_name in sim_out.columns
    has_y_ref = ref_y_name in sim_out.columns
    if has_x_ref and has_y_ref:
        ax.plot(sim_out[ref_x_name], sim_out[ref_y_name], label='ref')

    ax.legend(fontsize="large")
    fig.suptitle(f'{x_label} X {y_label}', fontsize="x-large")
    fig.tight_layout()

    if filepath:
        fig.savefig(filepath, **SAVEFIG_DEFAULTS)

    return fig, ax
