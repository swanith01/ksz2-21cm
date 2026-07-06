# =============================================================================
# style.py — global plot styling + save_pdf_png helper.
# Extracted verbatim from CELL 1b / CELL 1c ("PDF / PNG style contexts") of
# kSZ_Squared_21cm_11Jun_CLUSTER.py. Behaviour is unchanged: grid is OFF
# everywhere, no downstream overrides.
#
# Usage (from a script, not a notebook):
#     from ksz2_21cm.plotting.style import apply_global_style, save_pdf_png
#     apply_global_style()
#     save_pdf_png(lambda ax: ax.plot(x, y), plot_dir, "my_plot", title="...")
# =============================================================================

import matplotlib as mpl
import matplotlib.pyplot as plt

RC_PARAMS = {
    'font.family'        : 'serif',
    'font.serif'         : ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset'   : 'cm',
    'font.size'          : 20,
    'axes.labelsize'     : 28,
    'axes.titlesize'     : 22,
    'xtick.labelsize'    : 22,
    'ytick.labelsize'    : 22,
    'legend.fontsize'    : 18,
    'figure.titlesize'   : 20,
    'xtick.direction'    : 'in',
    'ytick.direction'    : 'in',
    'xtick.major.size'   : 6,
    'ytick.major.size'   : 6,
    'xtick.minor.size'   : 3,
    'ytick.minor.size'   : 3,
    'xtick.major.width'  : 1.0,
    'ytick.major.width'  : 1.0,
    'xtick.minor.width'  : 0.8,
    'ytick.minor.width'  : 0.8,
    'xtick.top'          : True,
    'ytick.right'        : True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.linewidth'     : 1.0,
    'lines.linewidth'    : 1.8,
    'lines.markersize'   : 5,
    'axes.grid'          : False,
    'grid.linewidth'     : 0.5,
    'grid.alpha'         : 0.3,
    'figure.dpi'         : 150,
    'savefig.dpi'        : 300,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.05,
}

PDF_STYLE = {
    'font.family'        : 'serif',
    'font.serif'         : ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset'   : 'cm',
    'font.size'          : 28,
    'axes.labelsize'     : 28,
    'axes.titlesize'     : 32,
    'xtick.labelsize'    : 26,
    'ytick.labelsize'    : 26,
    'legend.fontsize'    : 22,
    'figure.titlesize'   : 28,
    'xtick.direction'    : 'in',
    'ytick.direction'    : 'in',
    'xtick.top'          : True,
    'ytick.right'        : True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size'   : 6,
    'ytick.major.size'   : 6,
    'xtick.minor.size'   : 3,
    'ytick.minor.size'   : 3,
    'axes.linewidth'     : 1.0,
    'lines.linewidth'    : 1.8,
    'axes.grid'          : False,
    'figure.dpi'         : 150,
    'savefig.dpi'        : 300,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.05,
}

PNG_STYLE = {
    'font.family'        : 'serif',
    'font.serif'         : ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset'   : 'cm',
    'font.size'          : 16,
    'axes.labelsize'     : 22,
    'axes.titlesize'     : 18,
    'xtick.labelsize'    : 20,
    'ytick.labelsize'    : 20,
    'legend.fontsize'    : 18,
    'figure.titlesize'   : 16,
    'xtick.direction'    : 'in',
    'ytick.direction'    : 'in',
    'xtick.top'          : True,
    'ytick.right'        : True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.linewidth'     : 1.0,
    'lines.linewidth'    : 1.5,
    'axes.grid'          : False,
    'figure.dpi'         : 150,
    'savefig.dpi'        : 300,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.05,
}


def apply_global_style():
    """Apply the pipeline's global rcParams. Call once at the top of a script."""
    plt.rcParams.update(RC_PARAMS)


def save_pdf_png(plot_func, plot_dir, plot_name, title=None, figsize=(10, 7)):
    """
    Save a single-axis plot as both PDF and PNG, in the pipeline's house style.

    Parameters
    ----------
    plot_func : callable
        f(ax) — draws onto the provided Axes. Do NOT set font sizes or grid
        inside plot_func; that's handled by the rc_context here.
    plot_dir  : str
    plot_name : str
        Filename without extension.
    title     : str or None
        PNG-only title (PDF has no title, per house style).
    figsize   : tuple
    """
    with mpl.rc_context(PDF_STYLE):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_func(ax)
        ax.set_title("")
        ax.grid(False)
        fig.savefig(f"{plot_dir}/{plot_name}.pdf")
        plt.close(fig)

    with mpl.rc_context(PNG_STYLE):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_func(ax)
        ax.grid(False)
        if title is not None:
            ax.set_title(title, fontweight='bold')
        fig.savefig(f"{plot_dir}/{plot_name}.png")
        plt.close(fig)


def save_fig_both(build_func, plot_dir, plot_name, figsize=(10, 5)):
    """
    Save a pre-built multi-panel figure as both PDF and PNG.
    Extracted from SNR_rigorous_v2.py — used where save_pdf_png's single-axis
    plot_func(ax) pattern doesn't fit (multi-panel SNR/covariance figures).

    Parameters
    ----------
    build_func : callable
        f() -> matplotlib Figure, fully built (axes, labels, legend, etc.)
    """
    with mpl.rc_context(PDF_STYLE):
        fig = build_func()
        fig.savefig(f"{plot_dir}/{plot_name}.pdf")
        plt.close(fig)

    with mpl.rc_context(PNG_STYLE):
        fig = build_func()
        fig.savefig(f"{plot_dir}/{plot_name}.png")
        plt.close(fig)
