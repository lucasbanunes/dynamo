from typing import Sequence, Tuple
from numbers import Number
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

var_labels = {
    "x": "x",
    "y": "y",
    "z": "z",
    "phi": "\\phi",
    "theta": "\\theta",
    "psi": "\\psi"
}

linearized_inputs = {
    'z': 'f',
    'phi': 'm_x',
    'theta': 'm_y',
    'psi': 'm_z'
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


def time_plot(variables: Sequence[str],
              sim_out: pd.DataFrame,
              title: str = None,
              filepath: str = None,
              xlim: Tuple[Number, Number] = None,
              ylim: Tuple[Number, Number] = None,
              diff_order: int = 0
              ) -> Tuple[Figure, Axes]:
    fig, axes = plt.subplots(
        len(variables), 1,
        figsize=(19.20, 10.80)
    )
    for i, zipped in enumerate(zip(axes, variables)):
        lines = list()
        ax, base_name = zipped
        base_label = var_labels[base_name]
        d_diffs = diff_order*"d"
        var_name = d_diffs + base_name
        ref_name = d_diffs + f"ref_{base_name}"
        error_name = d_diffs + f"e_{base_name}"
        input_name = d_diffs + f"u_{base_name}"
        lin_name = linearized_inputs.get(base_name, None)
        ax.grid()
        var_label = get_final_label(base_label, diff_order)
        ylabel = var_label
        lines += ax.plot(sim_out['t'],
                         sim_out[var_name],
                         label=var_label,
                         color='C0')
        if ref_name in sim_out.columns:
            ref_label = get_final_label(f'r_{{{base_label}}}', diff_order)
            lines += ax.plot(sim_out['t'],
                             sim_out[ref_name],
                             label=ref_label,
                             linestyle='--',
                             color='k')
            ylabel += f', {ref_label}'

        if error_name in sim_out.columns:
            error_label = get_final_label(f'e_{{{base_label}}}', diff_order)
            lines += ax.plot(sim_out['t'],
                             sim_out[error_name],
                             label=error_label,
                             linestyle='--',
                             color='C1')
            ylabel += f', {error_label}'

        if input_name in sim_out.columns:
            input_label = get_final_label(f'u_{{{base_label}}}', diff_order)
            lines += ax.plot(sim_out['t'],
                             sim_out[input_name],
                             label=input_label,
                             linestyle='--',
                             color='C2')
            ylabel += f', {input_label}'

        if lin_name in sim_out.columns:
            twinx = ax.twinx()
            lin_label = f"${lin_name}$"
            lines += twinx.plot(sim_out['t'],
                                sim_out[lin_name],
                                label=lin_label,
                                linestyle='--',
                                color='C3')
            twinx.set_ylabel(lin_label)
            twinx.set(xlim=xlim, ylim=ylim)

        labels = [iline.get_label() for iline in lines]
        ax.legend(lines, labels)

        is_last_ax = i < len(variables)-1
        if is_last_ax:
            # Removes the xaxis ticks if not on last axis
            ax.set(xticklabels=[], ylabel=ylabel, xlim=xlim, ylim=ylim)
        else:
            ax.set(ylabel=ylabel, xlabel='Time (seconds)',
                   xlim=xlim, ylim=ylim)

    if title:
        fig.suptitle(title, fontsize='large')
    fig.tight_layout()

    if filepath:
        fig.savefig(filepath, **SAVEFIG_DEFAULTS)

    return fig, axes


def plot2d(x: str, y: str,
           sim_out: pd.DataFrame,
           diff_order: int = 0,
           filepath: str = None
           ) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(1, 1, figsize=(19.20, 10.80))
    ax.grid()
    d_diffs = diff_order*"d"
    x_name = d_diffs + x
    y_name = d_diffs + y
    x_label = get_final_label(var_labels[x], diff_order)
    y_label = get_final_label(var_labels[y], diff_order)
    ax.plot(sim_out[x_name], sim_out[y_name])
    ax.set(xlabel=x_label, ylabel=y_label)

    ref_x_name = f"{d_diffs}ref_{x}"
    ref_y_name = f"{d_diffs}ref_{y}"
    has_x_ref = ref_x_name in sim_out.columns
    has_y_ref = ref_y_name in sim_out.columns
    if has_x_ref and has_y_ref:
        ax.plot(sim_out[ref_x_name], sim_out[ref_y_name], label='ref')

    ax.legend()
    fig.suptitle(f'{x_label} X {y_label}')
    fig.tight_layout()

    if filepath:
        fig.savefig(filepath, **SAVEFIG_DEFAULTS)

    return fig, ax
