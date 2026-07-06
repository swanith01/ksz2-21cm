# =============================================================================
# [ARCHIVED ORIGINAL — kept as the reference copy of the full pipeline]
#
# This is your original kSZ_Squared_21cm_11Jun_CLUSTER.py, unchanged.
# Most of it has been split into the new repo structure:
#
#   CELL 1, 1a, 1c, 2      -> scripts/01_run_lightcones.py
#                             src/ksz2_21cm/simulate/lightcone_worker.py
#   CELL 1b (style)        -> src/ksz2_21cm/plotting/style.py
#   CELL 4, 5, 6           -> scripts/02_compute_ksz_maps.py
#                             src/ksz2_21cm/ksz/ksz_map.py
#   CELL 7 (unsquared)     -> scripts/03_compute_cross_corr.py
#                             src/ksz2_21cm/correlation/cross_corr_worker.py
#   CELL 7b (squared)      -> scripts/04_compute_cross_corr_sq.py
#                             src/ksz2_21cm/correlation/cross_corr_sq_worker.py
#   CELL SNR (CORRECTED)   -> legacy_snr_noise_free_cell.py (this directory;
#                             superseded by scripts/05_compute_snr_forecast.py)
#
# NOT yet individually split out (still only exist here, in this archived
# copy) because they are diagnostic/exploratory plots rather than pipeline
# steps the paper depends on:
#
#   CELL 7c   — diagnostic plots for the kSZ^2-21cm^2 pipeline
#               (foreground wedge, chunk boundaries, T21/T21^2 slices)
#   CELL 8a   — single-seed cross-corr visualization
#   CELL 8b   — seed-averaged cross-corr plots (symlog)
#   CELL 8.5  — kSZ^2-21cm^2 single-seed + seed-averaged plots
#   CELL 8c   — redshift evolution of the 21cm auto power spectrum
#   CELL 9    — "why kSZ^2x21cm (unsquared) dies under wedge filtering"
#               (this is the diagnostic that justifies moving to the
#               squared statistic in the first place — worth reading if
#               you're new to why CELL 7b/04_compute_cross_corr_sq.py is
#               the one that matters)
#
# If/when you want one of these as a real notebook: the file already uses
# the `# %%` cell-marker convention, so `pip install jupytext` and
# `jupytext --to notebook original_monolith_kSZ_Squared_21cm_11Jun_CLUSTER.py`
# will open it directly in Jupyter with cell boundaries intact — you can
# then delete everything except the CELL you want and save it into this
# directory as its own notebook.
# =============================================================================

# %%
# =============================================================================
# kSZ²-21cm : Lightcone Simulation and Plotting
# =============================================================================

# =============================================================================
# CELL 1: Imports and Setup
# =============================================================================

import numpy as np
import matplotlib as mpl
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import py21cmfast as p21c
from py21cmfast import plotting

import os
import glob
import time
from datetime import datetime
from run_config import *
import multiprocessing

# PBS vs desktop backend
if os.environ.get('PBS_JOBID'):
    matplotlib.use('Agg')
    print("✓ Using Agg backend (PBS/server mode)")
else:
    matplotlib.use('Agg')   # change to TkAgg for interactive desktop display
    print("✓ Using Agg backend")

print(f"py21cmfast version: {p21c.__version__}")

# =============================================================================
# CELL 1a: Output and Cache Directories
# =============================================================================

plot_dir = PLOT_DIR

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)
    print(f"Created directory: {plot_dir}")
else:
    print(f"Directory already exists: {plot_dir}")

print(f"All plots will be saved to: {os.path.abspath(plot_dir)}")

main_cache_dir = CACHE_DIR












print(f"Cache directory: {main_cache_dir}")

# =============================================================================
# CELL 1b: Global Plot Settings
# DO NOT override font sizes, grid, or tick settings in any downstream cell.
# All plots rely entirely on these settings + PDF_STYLE / PNG_STYLE contexts.
# =============================================================================

plt.rcParams.update({
    # Font
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
    # Ticks
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
    # Lines / axes
    'axes.linewidth'     : 1.0,
    'lines.linewidth'    : 1.8,
    'lines.markersize'   : 5,
    # Grid — OFF everywhere, no exceptions
    'axes.grid'          : False,
    'grid.linewidth'     : 0.5,
    'grid.alpha'         : 0.3,
    # Figure / save
    'figure.dpi'         : 150,
    'savefig.dpi'        : 300,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.05,
})

print("✓ Global plot settings applied (grid OFF, no downstream overrides needed)")

# =============================================================================
# PDF / PNG style contexts — used ONLY inside save_pdf_png
# =============================================================================

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

print("✓ PDF and PNG style contexts defined")


def save_pdf_png(plot_func, plot_dir, plot_name, title=None, figsize=(10, 7)):
    """
    Save a plot as both PDF and PNG.

    Parameters
    ----------
    plot_func : callable
        f(ax) — draws onto the provided Axes.
        Do NOT set font sizes or grid inside plot_func.
    plot_dir  : str
        Directory to save files
    plot_name : str
        Filename without extension
    title     : str or None
        PNG-only title (PDF has no title)
    figsize   : tuple, optional
        Figure size (width, height) in inches. Default: (10, 7)
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


print("✓ save_pdf_png defined (plot_func pattern, grid always OFF, figsize customizable)")

# =============================================================================
# CELL 1c: Define Parameters
# =============================================================================

user_params = p21c.UserParams(
    HII_DIM=HII_DIM,
    BOX_LEN=BOX_LEN,
    USE_INTERPOLATION_TABLES=True,
    N_THREADS=N_THREADS
)

z_min = Z_MIN
z_max = Z_MAX

# =============================================================================
# MULTI-SEED SETUP
# =============================================================================

RANDOM_SEEDS = RANDOM_SEEDS
N_SEEDS      = len(RANDOM_SEEDS)

print(f"\n=== PARAMETER SETUP ===")
print(f"HII_DIM     = {user_params.HII_DIM}")
print(f"BOX_LEN     = {user_params.BOX_LEN:.0f} Mpc")
print(f"z range     = [{z_min}, {z_max}]")
print(f"N_THREADS   = {user_params.N_THREADS}")

print(f"\n=== MULTI-SEED SETUP ===")
print(f"Seeds: {RANDOM_SEEDS}")
print(f"Total realisations: {N_SEEDS}")

print("\n=== DEFAULT COSMOLOGY ===")
print(p21c.CosmoParams())

print("\n=== DEFAULT ASTROPHYSICS ===")
print(p21c.AstroParams())

print("\n=== DEFAULT FLAGS ===")
print(p21c.FlagOptions())

# %%
# =============================================================================
# CELL 1b: Global Plot Settings
# DO NOT override font sizes, grid, or tick settings in any downstream cell.
# All plots rely entirely on these settings + PDF_STYLE / PNG_STYLE contexts.
# =============================================================================

plt.rcParams.update({
    # Font
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
    # Ticks
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
    # Lines / axes
    'axes.linewidth'     : 1.0,
    'lines.linewidth'    : 1.8,
    'lines.markersize'   : 5,
    # Grid — OFF everywhere, no exceptions
    'axes.grid'          : False,
    'grid.linewidth'     : 0.5,
    'grid.alpha'         : 0.3,
    # Figure / save
    'figure.dpi'         : 150,
    'savefig.dpi'        : 300,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.05,
})

print("✓ Global plot settings applied (grid OFF, no downstream overrides needed)")

# =============================================================================
# PDF / PNG style contexts — used ONLY inside save_pdf_png
# =============================================================================

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

print("✓ PDF and PNG style contexts defined")


def save_pdf_png(plot_func, plot_dir, plot_name, title=None, figsize=(10, 7)):
    """
    Save a plot as both PDF and PNG.

    Parameters
    ----------
    plot_func : callable
        f(ax) — draws onto the provided Axes.
        Do NOT set font sizes or grid inside plot_func.
    plot_dir  : str
        Directory to save files
    plot_name : str
        Filename without extension
    title     : str or None
        PNG-only title (PDF has no title)
    figsize   : tuple, optional
        Figure size (width, height) in inches. Default: (10, 7)
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


print("✓ save_pdf_png defined (plot_func pattern, grid always OFF, figsize customizable)")

# %%
# =============================================================================
# CELL 2: Run Lightcone Simulations for All Seeds
# Pro caching (native py21cmfast HDF5) + concurrent seeds (ProcessPool/spawn)
# =============================================================================

import time
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# IMPORT THE WORKER FROM THE SEPARATE MODULE
from lightcone_worker import run_or_load_seed

print("\n" + "="*70)
print("RUNNING LIGHTCONE SIMULATIONS FOR kSZ²-21cm ANALYSIS")
print("="*70)

# =============================================================================
# CORE / WORKER ALLOCATION
# =============================================================================
N_TOTAL_CORES = int(os.environ.get('PBS_NCPUS', os.cpu_count() or 32))

# User preference
DESIRED_THREADS_PER_WORKER = 8        # Change this if you want more/less
N_WORKERS = max(1, N_TOTAL_CORES // DESIRED_THREADS_PER_WORKER)
N_WORKERS = min(N_WORKERS, N_SEEDS)   # Don't exceed number of seeds

print(f"Available cores : {N_TOTAL_CORES}")
print(f"Workers         : {N_WORKERS} (concurrent seeds)")
print(f"Threads per worker : {DESIRED_THREADS_PER_WORKER}")
print(f"Total threads used : {N_WORKERS * DESIRED_THREADS_PER_WORKER}/{N_TOTAL_CORES}")

# =============================================================================
# Astrophysical params summary (defaults)
# =============================================================================
_astro_summary = p21c.AstroParams()
print(f"\nAstrophysical Parameters (defaults):")
print(f"  HII_EFF_FACTOR = {_astro_summary.HII_EFF_FACTOR}")
print(f"  ION_Tvir_MIN   = {_astro_summary.ION_Tvir_MIN:.3f} (log10 K) "
      f"= {10**_astro_summary.ION_Tvir_MIN:.2e} K")
print(f"Redshift range : z = {z_min} → {z_max}")
print(f"Box size       : {user_params.BOX_LEN} Mpc")
print(f"Resolution     : {user_params.HII_DIM}³ cells")


# =============================================================================
# Dispatch — concurrent seeds via ProcessPoolExecutor with spawn context
# =============================================================================
lightcones    = {}
seed_metadata = {}   # {seed: {'cache_file', 'sim_time', 'status'}}

scan_start = time.time()

# spawn context is critical: py21cmfast's CFFI / C globals don't survive fork
mp_ctx = mp.get_context("fork")

print(f"\n{'='*70}")
print(f"DISPATCHING {N_SEEDS} SEEDS — {N_WORKERS} concurrent workers")
print(f"{'='*70}", flush=True)


with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp_ctx) as ex:
    futures = {}
    for seed in RANDOM_SEEDS:
        seed_cache_dir = os.path.join(main_cache_dir, f"seed_{seed}")
        # Now using the imported function from lightcone_worker module
        fut = ex.submit(
            run_or_load_seed,
            seed, seed_cache_dir, z_min, z_max,
            user_params.HII_DIM, user_params.BOX_LEN,
            DESIRED_THREADS_PER_WORKER,
        )
        futures[fut] = seed

    completed_count = 0
    for fut in as_completed(futures):
        seed = futures[fut]
        try:
            seed_done, cache_file, sim_time, status = fut.result()
        except Exception as e:
            seed_done, cache_file, sim_time, status = seed, None, 0.0, f"crashed: {e}"

        completed_count += 1
        seed_metadata[seed_done] = {
            'cache_file': cache_file,
            'sim_time'  : sim_time,
            'status'    : status,
        }

        if status == "cached":
            msg = f"✓ cached  (instant load)"
        elif status == "computed":
            msg = f"✓ computed in {sim_time/60:.2f} min"
        else:
            msg = f"✗ {status}"

        elapsed = (time.time() - scan_start) / 60
        print(f"  [{completed_count:2d}/{N_SEEDS}] seed {seed_done:3d}: "
              f"{msg}   (elapsed: {elapsed:.1f} min)", flush=True)

print(f"\n{'='*70}")
print(f"✓ ALL WORKERS RETURNED  —  {time.time()-scan_start:.1f} s total")
print(f"{'='*70}")

# =============================================================================
# Load every cached lightcone into the main process
# (single-process load — fast, and gives real LightCone objects downstream)
# =============================================================================
print(f"\n=== LOADING LIGHTCONES INTO MAIN PROCESS ===", flush=True)

for seed in RANDOM_SEEDS:
    meta = seed_metadata.get(seed, {})
    cache_file = meta.get('cache_file')

    if not cache_file or not os.path.exists(cache_file):
        print(f"  ✗ seed {seed}: no cache file — skipping")
        continue

    try:
        # Load HDF5 directly — much faster
        lc = p21c.LightCone.read(cache_file)
        lightcones[seed] = lc

        z_nodes   = lc.node_redshifts[::-1]
        x_e_nodes = 1.0 - lc.global_xH[::-1]
        try:
            z_10 = z_nodes[np.argmin(np.abs(x_e_nodes - 0.1))]
            z_50 = z_nodes[np.argmin(np.abs(x_e_nodes - 0.5))]
            z_90 = z_nodes[np.argmin(np.abs(x_e_nodes - 0.9))]
            reion = (f"z(10%)={z_10:.2f}  z(50%)={z_50:.2f}  "
                     f"z(90%)={z_90:.2f}  Δz={z_10-z_90:.2f}")
        except Exception:
            reion = "(reion stats unavailable)"
        print(f"  ✓ seed {seed:3d} loaded   {reion}")

    except Exception as e:
        print(f"  ✗ seed {seed}: load failed — {e}")

# # =============================================================================
# # CELL 2b: Plot Lightcone Fields (random seed as sanity check)
# # Uses save_pdf_png — no font/grid overrides
# # =============================================================================

# if len(lightcones) > 0:
#     seed_to_plot = int(np.random.choice(list(lightcones.keys())))
#     lightcone    = lightcones[seed_to_plot]

#     print(f"\n{'='*70}")
#     print(f"LIGHTCONE PLOTS (randomly selected seed={seed_to_plot})")
#     print(f"{'='*70}")

#     fields_to_plot = [
#         ('brightness_temp', '21cm Brightness Temperature', 'EoR'),
#         ('xH_box',          'Neutral Fraction (xHI)',      'viridis'),
#         ('density',         'Overdensity δ',               'magma'),
#         ('velocity',        'Line-of-Sight Velocity',      'RdBu_r'),
#     ]

#     for field_name, field_title, field_cmap in fields_to_plot:
#         print(f"  Plotting {field_name}...")

#         field_data = np.asarray(getattr(lightcone, field_name))
#         mid_slice  = field_data[:, :, field_data.shape[2] // 2]

#         def _draw(ax, _slice=mid_slice, _cmap=field_cmap,
#                   _title=field_title, _seed=seed_to_plot):
#             im = ax.imshow(_slice.T, aspect='auto',
#                            cmap=_cmap, origin='lower')
#             ax.figure.colorbar(im, ax=ax)
#             ax.set_xlabel(r'LoS pixel')
#             ax.set_ylabel(r'Transverse pixel')
#             ax.text(0.02, 0.98,
#                     f'seed={_seed} | '
#                     f'HII_EFF_FACTOR={astro_params.HII_EFF_FACTOR:.1f}, '
#                     f'ION_Tvir_MIN={astro_params.ION_Tvir_MIN:.2f}',
#                     transform=ax.transAxes,
#                     verticalalignment='top',
#                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#         plot_name = f"{field_name}_lightcone_seed{seed_to_plot}"
#         save_pdf_png(
#             _draw, plot_dir, plot_name,
#             title=f'{field_title} — Lightcone (z={z_max}→{z_min}) '
#                   f'[seed={seed_to_plot}]',
#             figsize=(12, 5),
#         )
#         print(f"  ✓ Saved: {plot_name}")

#     print("\n✓ LIGHTCONE PLOTTING COMPLETE!")
# else:
#     print("\n✗ Skipping plots — no lightcones available")

# %%
# =============================================================================
# CELL 4: Reionization History + Optical Depth (All Seeds)
# Combines former Cells 3 and 4.
# Produces:
#   - tau_results  (dict, used by downstream kSZ integration)
#   - xe_mean / xHI_mean / tau_mean  (mean curves)
# Plots:
#   - reionization_history_xe   (xe vs z)
#   - reionization_history_xHI  (xHI vs z)
#   - tau_vs_z                   (cumulative τ vs z)
# All plotting via save_pdf_png — no font/grid overrides.
# =============================================================================

print("\n" + "="*70)
print("REIONIZATION HISTORY + OPTICAL DEPTH ANALYSIS")
print("="*70)

if len(lightcones) > 0:

    # =========================================================================
    # PART A: Reionization histories (xe, xHI) — per seed + mean across seeds
    # =========================================================================
    all_z_nodes   = {}
    all_x_e_nodes = {}
    all_xHI_nodes = {}

    for seed, lc in lightcones.items():
        sort_idx              = np.argsort(lc.node_redshifts)
        all_z_nodes[seed]     = lc.node_redshifts[sort_idx]
        all_x_e_nodes[seed]   = (1.0 - lc.global_xH)[sort_idx]
        all_xHI_nodes[seed]   = lc.global_xH[sort_idx]

    # Common redshift grid for averaging — within range shared by all seeds
    z_min_common = max(all_z_nodes[s].min() for s in lightcones)
    z_max_common = min(all_z_nodes[s].max() for s in lightcones)
    z_common_xe  = np.linspace(z_min_common, z_max_common, 500)

    xe_interp = np.array([
        np.interp(z_common_xe, all_z_nodes[s], all_x_e_nodes[s])
        for s in lightcones.keys()
    ])
    xHI_interp = np.array([
        np.interp(z_common_xe, all_z_nodes[s], all_xHI_nodes[s])
        for s in lightcones.keys()
    ])

    xe_mean  = np.mean(xe_interp,  axis=0)
    xe_std   = np.std(xe_interp,   axis=0)
    xHI_mean = np.mean(xHI_interp, axis=0)
    xHI_std  = np.std(xHI_interp,  axis=0)

    # Mean reionization midpoint
    z_xe_half_mean = np.interp(0.5, xe_mean[::-1], z_common_xe[::-1])
    print(f"\nMean z(x_e = 0.5) across {N_SEEDS} seeds: z = {z_xe_half_mean:.2f}")
    for seed in lightcones.keys():
        z_half = np.interp(0.5, all_x_e_nodes[seed], all_z_nodes[seed])
        print(f"  Seed {seed:3d}: z(x_e=0.5) = {z_half:.2f}")

    # Color map for per-seed lines (used in all three plots)
    cmap_seeds = plt.cm.plasma(np.linspace(0.1, 0.9, len(lightcones)))

    # AstroParams instance for annotations
    astro_params = p21c.AstroParams()

    # ---------------------------------------------------------------------
    # PLOT 4a: x_e vs z
    # ---------------------------------------------------------------------
    def _draw_xe(ax):
        for i, seed in enumerate(lightcones.keys()):
            ax.plot(all_z_nodes[seed], all_x_e_nodes[seed],
                    color=cmap_seeds[i], lw=1.0, alpha=0.4)

        ax.fill_between(z_common_xe, xe_mean - xe_std, xe_mean + xe_std,
                        color='darkblue', alpha=0.2, label=r'$\pm 1\sigma$')
        ax.plot(z_common_xe, xe_mean,
                color='darkblue', lw=2.5, label=f'Mean ({N_SEEDS} seeds)')

        ax.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.7)
        ax.axvline(z_xe_half_mean, color='gray', linestyle='--', lw=1, alpha=0.7)
        ax.text(z_xe_half_mean, 0.52,
                rf'$z_{{x_e=0.5}} = {z_xe_half_mean:.2f}$',
                ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel(r'Redshift $z$')
        ax.set_ylabel(r'Ionization Fraction $x_e$')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best')
        ax.invert_xaxis()

    save_pdf_png(_draw_xe, plot_dir, "reionization_history_xe",
                 title='Reionization History: Ionization Fraction')
    print(f"\n✓ Saved: reionization_history_xe")

    # ---------------------------------------------------------------------
    # PLOT 4b: x_HI vs z
    # ---------------------------------------------------------------------
    def _draw_xHI(ax):
        for i, seed in enumerate(lightcones.keys()):
            ax.plot(all_z_nodes[seed], all_xHI_nodes[seed],
                    color=cmap_seeds[i], lw=1.0, alpha=0.4)

        ax.fill_between(z_common_xe, xHI_mean - xHI_std, xHI_mean + xHI_std,
                        color='darkred', alpha=0.2, label=r'$\pm 1\sigma$')
        ax.plot(z_common_xe, xHI_mean,
                color='darkred', lw=2.5, label=f'Mean ({N_SEEDS} seeds)')

        ax.set_xlabel(r'Redshift $z$')
        ax.set_ylabel(r'Neutral Fraction $x_{\rm HI}$')
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='best')
        ax.invert_xaxis()

        ax.text(0.05, 0.95,
                f'HII_EFF_FACTOR = {astro_params.HII_EFF_FACTOR:.1f}\n'
                f'ION_Tvir_MIN = {astro_params.ION_Tvir_MIN:.2f} (log10 K)\n'
                f'N seeds = {N_SEEDS}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    save_pdf_png(_draw_xHI, plot_dir, "reionization_history_xHI",
                 title='Reionization History: Neutral Fraction')
    print(f"✓ Saved: reionization_history_xHI")

    # =========================================================================
    # PART B: Optical depth τ(<z) — per seed, then averaged
    # =========================================================================
    print("\n--- Optical Depth Calculation ---")

    # Physical constants
    c_km_s         = 2.998e5
    h              = 0.6766
    H0             = 100 * h
    Omega_b        = 0.04897468161869667
    Omega_m        = 0.30964144154550644
    rho_crit_p_cm3 = 1.88e-29 * h**2 / (1.67e-24)
    n_H0_cm3       = Omega_b * rho_crit_p_cm3
    sigma_T_cm2    = 6.65e-25
    cm_per_Mpc     = 3.086e24
    n_e0_Mpc3      = n_H0_cm3 * cm_per_Mpc**3
    sigma_T_Mpc2   = sigma_T_cm2 / cm_per_Mpc**2
    prefactor      = n_e0_Mpc3 * sigma_T_Mpc2

    print(f"  n_H0     = {n_H0_cm3:.6e} cm^-3")
    print(f"  σ_T      = {sigma_T_cm2:.6e} cm^2")
    print(f"  Prefactor= {prefactor:.6e} Mpc^-1")

    # ---- Compute τ per seed ----
    tau_results = {}

    for seed, lc in lightcones.items():
        red_axis = lc.lightcone_redshifts
        pos_axis = lc.lightcone_distances

        # Trim to z ≤ z_max
        ind_z    = np.where(red_axis <= z_max)[0]
        red_axis = red_axis[ind_z]
        pos_axis = pos_axis[ind_z]

        # Ionization history (descending → ascending)
        z_nodes_sorted   = lc.node_redshifts[::-1]
        xHI_nodes_sorted = lc.global_xH[::-1]
        x_e_nodes_sorted = 1.0 - xHI_nodes_sorted

        x_e_interp = np.interp(red_axis, z_nodes_sorted, x_e_nodes_sorted)

        ds_Mpc  = np.asarray(np.diff(pos_axis), dtype=np.float64)
        z_mid   = 0.5 * (red_axis[:-1] + red_axis[1:])
        x_e_mid = 0.5 * (x_e_interp[:-1] + x_e_interp[1:])

        dtau      = prefactor * x_e_mid * (1.0 + z_mid)**2 * ds_Mpc
        tau       = np.cumsum(dtau)
        tau_total = tau[-1]

        tau_results[seed] = {
            'red_axis' : red_axis,
            'z_mid'    : z_mid,
            'x_e_mid'  : x_e_mid,
            'ds_Mpc'   : ds_Mpc,
            'tau'      : tau,
            'tau_total': tau_total,
        }
        print(f"  Seed {seed:3d}: τ_total = {tau_total:.6f}")

    tau_totals = np.array([tau_results[s]['tau_total'] for s in lightcones])
    print(f"\n  Mean τ = {tau_totals.mean():.6f} ± {tau_totals.std():.6f}")

    # ---- Stack τ(<z) across seeds — handle potential grid mismatch safely ----
    ref_seed   = next(iter(lightcones))
    ref_z_mid  = tau_results[ref_seed]['z_mid']

    grids_match = all(
        tau_results[s]['z_mid'].shape == ref_z_mid.shape
        and np.allclose(tau_results[s]['z_mid'], ref_z_mid)
        for s in lightcones
    )

    if grids_match:
        z_common_tau = ref_z_mid
        tau_matrix   = np.array([tau_results[s]['tau'] for s in lightcones])
    else:
        # Fallback: interpolate onto a common grid
        print("  ⚠ Seed z_mid grids differ — interpolating onto a common grid")
        z_lo = max(tau_results[s]['z_mid'].min() for s in lightcones)
        z_hi = min(tau_results[s]['z_mid'].max() for s in lightcones)
        z_common_tau = np.linspace(z_lo, z_hi, 1000)
        tau_matrix   = np.array([
            np.interp(z_common_tau,
                      tau_results[s]['z_mid'],
                      tau_results[s]['tau'])
            for s in lightcones
        ])

    tau_mean = np.mean(tau_matrix, axis=0)
    tau_std  = np.std(tau_matrix,  axis=0)

    # ---------------------------------------------------------------------
    # PLOT 4c: τ(<z) vs z
    # ---------------------------------------------------------------------
    def _draw_tau(ax):
        for i, seed in enumerate(lightcones.keys()):
            ax.plot(tau_results[seed]['z_mid'], tau_results[seed]['tau'],
                    color=cmap_seeds[i], lw=1.0, alpha=0.4)

        ax.fill_between(z_common_tau, tau_mean - tau_std, tau_mean + tau_std,
                        color='darkgreen', alpha=0.2, label=r'$\pm 1\sigma$')
        ax.plot(z_common_tau, tau_mean,
                color='darkgreen', lw=2.5, label=f'Mean ({N_SEEDS} seeds)')

        ax.set_xlabel(r'Redshift $z$')
        ax.set_ylabel(r'Cumulative Optical Depth $\tau(<z)$')
        ax.legend(loc='best')
        ax.invert_xaxis()

        ax.text(0.05, 0.95,
                f'Mean τ = {tau_totals.mean():.4f} ± {tau_totals.std():.4f}\n'
                f'HII_EFF_FACTOR = {astro_params.HII_EFF_FACTOR:.1f}\n'
                f'N seeds = {N_SEEDS}',
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    save_pdf_png(_draw_tau, plot_dir, "tau_vs_z",
                 title='Cumulative Optical Depth vs Redshift')
    print(f"\n✓ Saved: tau_vs_z")

    print("\n✓ CELL 4 COMPLETE (reionization history + optical depth)")

else:
    print("\n✗ Skipping — no lightcones available")

# %%
# =============================================================================
# CELL 5: Compute kSZ Integrand with Visibility Function (All Seeds)
# kSZ integrand = (1 + δ) × x_e × v_z / c × e^(-τ(z))
# Skips computation for seeds whose Cell 6 kSZ map cache already exists.
# No parallelization needed — per-seed work is just NumPy on 128³ arrays.
# =============================================================================

print("\n" + "="*70)
print("COMPUTING kSZ INTEGRAND WITH VISIBILITY FUNCTION")
print("="*70)

# Observation redshift — only used to build the Cell-6 cache-skip path
z_obs = Z_OBS

if len(lightcones) > 0 and len(tau_results) > 0:

    c_Mpc_s = 299792.458 / 3.08567758e19
    print(f"Speed of light: c = {c_Mpc_s:.6e} Mpc/s")

    kSZ_integrands = {}   # {seed: 3D array, or None if skipped}

    for seed, lc in lightcones.items():

        print(f"\n--- Seed {seed} ---")

        # ------------------------------------------------------------------
        # Skip if Cell 6 kSZ map cache already exists — integrand not needed
        # ------------------------------------------------------------------
        map_path = os.path.join(
            main_cache_dir, "kSZ_maps",
            f"kSZ_map_z{z_obs:.1f}_seed{seed}.npy"
        )
        if os.path.exists(map_path):
            print(f"  Cell 6 cache exists → skipping integrand computation")
            kSZ_integrands[seed] = None
            continue

        # ------------------------------------------------------------------
        # Compute integrand
        # ------------------------------------------------------------------
        tr = tau_results[seed]

        red_axis_full = np.asarray(lc.lightcone_redshifts)
        ind_z         = np.where(red_axis_full <= z_max)[0]

        density_1plus = 1 + np.asarray(lc.density[:, :, ind_z])
        x_e_3D        = 1 - np.asarray(lc.xH_box[:, :, ind_z])
        v_los_Mpc_s   = np.asarray(lc.velocity[:, :, ind_z])/67.4

        red_axis_array = np.asarray(tr['red_axis'], dtype=np.float64)
        tau_array      = np.asarray(tr['tau'],      dtype=np.float64)
        z_mid_array    = np.asarray(tr['z_mid'],    dtype=np.float64)

        tau_extended = np.concatenate([[0.0], tau_array])
        z_extended   = np.concatenate([[red_axis_array[0]], z_mid_array])
        tau_at_lc    = np.asarray(
            np.interp(red_axis_array, z_extended, tau_extended),
            dtype=np.float64,
        )

        visibility    = np.exp(-tau_at_lc)
        visibility_3D = visibility[None, None, :]

        kSZ_integrand = (density_1plus * x_e_3D
                         * v_los_Mpc_s / c_Mpc_s
                         * visibility_3D)

        kSZ_integrands[seed] = kSZ_integrand

        print(f"  Shape : {kSZ_integrand.shape}")
        print(f"  Mean  : {kSZ_integrand.mean():.4e}")
        print(f"  Std   : {kSZ_integrand.std():.4e}")
        print(f"  RMS   : {np.sqrt(np.mean(kSZ_integrand**2)):.4e}")

    n_computed = sum(1 for v in kSZ_integrands.values() if v is not None)
    n_skipped  = N_SEEDS - n_computed
    print(f"\n✓ Integrand computed: {n_computed} seeds")
    print(f"  Skipped (Cell 6 cache found): {n_skipped} seeds")

#     # =========================================================================
#     # PLOT 5a: kSZ Integrand Lightcone (random seed that was actually computed)
#     # =========================================================================
#     computed_seeds = [s for s, v in kSZ_integrands.items() if v is not None]

#     if len(computed_seeds) > 0:
#         seed_to_plot  = int(np.random.choice(computed_seeds))
#         print(f"\nRandomly selected seed for plot: {seed_to_plot}")

#         lc            = lightcones[seed_to_plot]
#         kSZ_integrand = kSZ_integrands[seed_to_plot]

#         red_axis_full = np.asarray(lc.lightcone_redshifts)
#         ind_z         = np.where(red_axis_full <= z_max)[0]

#         slice_2D = kSZ_integrand[:, :, kSZ_integrand.shape[2] // 2]
#         x_extent = float(np.asarray(lc.lightcone_distances[ind_z].max()))
#         y_extent = float(user_params.BOX_LEN)

#         # Symmetric color limits at 99th percentile of |integrand|
#         vmax = float(np.percentile(np.abs(kSZ_integrand), 99))

#         # Distance → redshift mapping for the twin x-axis
#         lc_distances_float = np.asarray(lc.lightcone_distances, dtype=np.float64)
#         lc_redshifts_float = np.asarray(lc.lightcone_redshifts, dtype=np.float64)

#         plot_name = f"kSZ_integrand_with_visibility_seed{seed_to_plot}"

#         def _draw_integrand(ax):
#             # Your plotting code here
#             im = ax.imshow(integrand_slice.T, aspect='auto', cmap='RdBu_r', origin='lower')
#             ax.figure.colorbar(im, ax=ax)
#             ax.set_xlabel(r'LoS pixel')
#             ax.set_ylabel(r'Transverse pixel')
#             ax.text(0.02, 0.98,
#                     f'seed={seed_to_plot}',
#                     transform=ax.transAxes,
#                     verticalalignment='top',
#                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

#         save_pdf_png(
#             _draw_integrand, plot_dir, plot_name,
#             title=(r'kSZ Integrand: '
#                 r'$(1+\delta)\times x_e\times v_z/c\times e^{-\tau(z)}$'
#                 f'  [seed={seed_to_plot}]')
#         )

#         print(f"✓ Saved: {plot_name}")
#     else:
#         print("\n  All seeds loaded from Cell 6 cache — no integrand plot generated")

# else:
#     print("\n✗ Skipping — lightcones or tau_results not available")

# %%
# =============================================================================
# CELL 6: Compute Line-of-Sight Integrated kSZ Maps for All Seeds
# kSZ(z_obs=5) = ∫ from z_start to z=5 of
#                [n_e0 σ_T (1/a²) (1+δ) x_e v_z/c e^(-τ) ds]
#
# Cache strategy:
#   1. If kSZ_map_zX.X_seedN.npy exists → load directly (fast path).
#   2. Else if Cell 5 left a usable integrand in memory → integrate it.
#   3. Else → skip (with a clear message telling the user to rerun Cell 5).
#
# No parallelization — per-seed work is one np.sum over a 128² × ~few-hundred
# slice array; NumPy + BLAS handle this in <1 s per seed.
# =============================================================================

import time

print("\n" + "="*70)
print(f"LINE-OF-SIGHT kSZ MAP INTEGRATION AT z_obs = {z_obs:.1f}")
print("="*70)

if len(lightcones) > 0:

    # =========================================================================
    # Cache directory for kSZ maps
    # =========================================================================
    kSZ_maps_dir = os.path.join(main_cache_dir, "kSZ_maps")
    os.makedirs(kSZ_maps_dir, exist_ok=True)
    print(f"kSZ maps directory: {kSZ_maps_dir}")

    # =========================================================================
    # Physical constants (CGS)
    # =========================================================================
    print(f"\n=== PHYSICAL CONSTANTS (CGS) ===")
    c_cm_s      = 3.0e10
    sigma_T_cm2 = 6.6525e-25
    n_e0_cm3    = 2.06e-7
    Mpc_to_cm   = 3.0857e24

    print(f"  c    = {c_cm_s:.2e} cm/s")
    print(f"  σ_T  = {sigma_T_cm2:.4e} cm²")
    print(f"  n_e0 = {n_e0_cm3:.4e} cm⁻³")
    print(f"  1 Mpc= {Mpc_to_cm:.4e} cm")

    prefactor_cgs = n_e0_cm3 * sigma_T_cm2 * c_cm_s
    print(f"  Prefactor n_e0 × σ_T × c = {prefactor_cgs:.4e} s⁻¹")

    print(f"\nz_obs = {z_obs:.1f} (end of reionization)")

    # Is Cell 5's integrand dict available at all? (It might not be if the user
    # jumped straight to Cell 6 after a clean cache load.)
    integrands_in_scope = ('kSZ_integrands' in dir()
                           or 'kSZ_integrands' in globals())

    # =========================================================================
    # Loop over seeds
    # =========================================================================
    kSZ_maps = {}

    for seed, lc in lightcones.items():

        print(f"\n--- Seed {seed} ---")

        map_path = os.path.join(
            kSZ_maps_dir, f"kSZ_map_z{z_obs:.1f}_seed{seed}.npy"
        )

        # ------------------------------------------------------------------
        # CASE 1: Cached kSZ map exists → load and continue
        # ------------------------------------------------------------------
        if os.path.exists(map_path):
            print(f"  Found cached kSZ map → loading")
            kSZ_maps[seed] = np.load(map_path)
            rms = float(np.sqrt(np.mean(kSZ_maps[seed]**2)))
            print(f"  ✓ Loaded | RMS: {rms:.4e}")
            continue

        # ------------------------------------------------------------------
        # CASE 2: Need to compute → check that Cell 5 left an integrand
        # ------------------------------------------------------------------
        integrand_available = (
            integrands_in_scope
            and seed in kSZ_integrands
            and kSZ_integrands[seed] is not None
        )

        if not integrand_available:
            print(f"  ✗ No cached map AND no integrand in memory — skipping")
            print(f"    (delete {kSZ_maps_dir}/kSZ_map_z*_seed{seed}.npy "
                  f"and rerun Cell 5 to recompute)")
            continue

        # ------------------------------------------------------------------
        # CASE 3: Compute kSZ map from integrand
        # ------------------------------------------------------------------
        print(f"  Computing from integrand...")

        tr       = tau_results[seed]
        red_axis = np.asarray(tr['red_axis'], dtype=np.float64)
        ds_Mpc   = np.asarray(tr['ds_Mpc'],   dtype=np.float64)
        z_mid    = np.asarray(tr['z_mid'],    dtype=np.float64)
        ds_cm    = ds_Mpc * Mpc_to_cm

        a                = 1.0 / (1.0 + red_axis)
        a_squared        = a**2
        a_squared_mid    = 0.5 * (a_squared[:-1] + a_squared[1:])
        a_squared_mid_3D = a_squared_mid[None, None, :]

        kSZ_int      = kSZ_integrands[seed]
        kSZ_int_mid  = 0.5 * (kSZ_int[:, :, :-1] + kSZ_int[:, :, 1:])
        kSZ_int_full = ((prefactor_cgs / a_squared_mid_3D)
                        * kSZ_int_mid
                        * (ds_cm / c_cm_s)[None, None, :])

        idx_integrate = np.where(z_mid >= z_obs)[0]
        print(f"  Integration: z = {z_mid[idx_integrate].max():.2f} → "
              f"{z_mid[idx_integrate].min():.2f} "
              f"({len(idx_integrate)} slices)")

        t0      = time.time()
        kSZ_map = np.sum(kSZ_int_full[:, :, idx_integrate], axis=2)
        print(f"  Computed in {time.time()-t0:.2f}s")

        np.save(map_path, kSZ_map)
        print(f"  ✓ Saved to {map_path}")

        kSZ_maps[seed] = kSZ_map
        print(f"  Mean: {kSZ_map.mean():.4e} | "
              f"RMS: {np.sqrt(np.mean(kSZ_map**2)):.4e} | "
              f"Std: {kSZ_map.std():.4e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n✓ kSZ MAPS READY: {len(kSZ_maps)}/{N_SEEDS} SEEDS")

    if len(kSZ_maps) > 0:
        rms_all = np.array([np.sqrt(np.mean(kSZ_maps[s]**2))
                            for s in kSZ_maps])
        ref_seed = next(iter(kSZ_maps))
        ref_map  = kSZ_maps[ref_seed]

        print(f"  RMS across seeds : {rms_all.mean():.4e} ± "
              f"{rms_all.std():.4e}")
        print(f"  Map dimensions   : {ref_map.shape[0]} × "
              f"{ref_map.shape[1]} pixels")
        print(f"  Physical size    : {user_params.BOX_LEN:.1f} × "
              f"{user_params.BOX_LEN:.1f} Mpc²")
        print(f"  Pixel size       : "
              f"{user_params.BOX_LEN / ref_map.shape[0]:.2f} Mpc")

else:
    print("\n✗ Skipping — no lightcones available")

print("\n" + "="*70)

# %%
# =============================================================================
# CELL 7: Compute kSZ²-21cm Cross-Correlation Power Spectra (All Seeds)
# PARALLELISED across seeds
# =============================================================================

from astropy.cosmology import FlatLambdaCDM
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import the worker function
from cross_corr_worker import compute_cross_corr_for_seed

print("\n" + "="*70)
print("COMPUTING kSZ²-21cm CROSS-CORRELATION POWER SPECTRA (PARALLELISED)")
print("="*70)

if len(lightcones) > 0:

    # =========================================================================
    # Map geometry + k-space grid (identical across seeds)
    # =========================================================================
    npix_side    = user_params.HII_DIM
    box_size_Mpc = float(user_params.BOX_LEN)
    pix_size_Mpc = box_size_Mpc / npix_side
    pix_area     = pix_size_Mpc**2

    print(f"\n=== MAP PROPERTIES ===")
    print(f"  Map size   : {npix_side} × {npix_side} pixels")
    print(f"  Physical   : {box_size_Mpc:.1f} × {box_size_Mpc:.1f} Mpc²")
    print(f"  Pixel size : {pix_size_Mpc:.3f} Mpc/pixel")

    dk        = 2 * np.pi / (npix_side * pix_size_Mpc)
    kx        = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
    ky        = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
    kgrid     = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
    k_bins    = np.logspace(np.log10(dk), np.log10(kgrid.max() * 0.9), 35)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    print(f"\n=== k-SPACE GRID ===")
    print(f"  dk      : {dk:.6f} Mpc⁻¹")
    print(f"  k range : [{kgrid.min():.6f}, {kgrid.max():.6f}] Mpc⁻¹")
    print(f"  N bins  : {len(k_centers)}")

    cosmo        = FlatLambdaCDM(H0=67.77, Om0=0.3086)
    kSZ_maps_dir = os.path.join(main_cache_dir, "kSZ_maps")

    # Ensure kSZ_maps dict exists
    if 'kSZ_maps' not in dir() and 'kSZ_maps' not in globals():
        kSZ_maps = {}

    # =========================================================================
    # Fetch kSZ maps for all seeds upfront (before parallelization)
    # =========================================================================
    print(f"\n=== LOADING kSZ MAPS ===")
    kSZ_maps_for_workers = {}
    
    for seed in lightcones.keys():
        if seed in kSZ_maps and kSZ_maps[seed] is not None:
            kSZ_maps_for_workers[seed] = kSZ_maps[seed]
        else:
            map_path = os.path.join(
                kSZ_maps_dir, f"kSZ_map_z{z_obs:.1f}_seed{seed}.npy"
            )
            if os.path.exists(map_path):
                kSZ_maps_for_workers[seed] = np.load(map_path)
                kSZ_maps[seed] = kSZ_maps_for_workers[seed]
            else:
                print(f"  ✗ No kSZ map for seed {seed}")
                kSZ_maps_for_workers[seed] = None

    # =========================================================================
    # Build worker arguments
    # =========================================================================
    worker_args = []
    for seed in lightcones.keys():
        if kSZ_maps_for_workers[seed] is not None:
            # Get cache file path from seed_metadata (set in CELL 2)
            cache_file = seed_metadata.get(seed, {}).get('cache_file')
            if cache_file and os.path.exists(cache_file):
                args = (
                    seed, cache_file, kSZ_maps_for_workers[seed],
                    z_obs, main_cache_dir,
                    npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
                    dk, kgrid, k_bins, k_centers
                )
                worker_args.append(args)
            else:
                print(f"  ✗ No cache file for seed {seed}")

    # =========================================================================
    # Parallel execution
    # =========================================================================
    N_WORKERS = max(1, N_TOTAL_CORES // 16)  # Each seed uses ~16 threads for FFTs
    N_WORKERS = min(N_WORKERS, len(worker_args))
    
    print(f"\n{'='*70}")
    print(f"DISPATCHING {len(worker_args)} SEEDS — {N_WORKERS} concurrent workers")
    print(f"{'='*70}", flush=True)

    mp_ctx = mp.get_context("fork")
    cross_corr_results_all = {}
    
    scan_start = time.time()

    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp_ctx) as ex:
        futures = {ex.submit(compute_cross_corr_for_seed, args): args[0]
                   for args in worker_args}
        
        completed = 0
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                seed_done, ccr_dict, status_msg = fut.result()
                completed += 1
                
                if ccr_dict is not None:
                    cross_corr_results_all[seed_done] = ccr_dict
                    status_str = f"✓ {status_msg}"
                else:
                    status_str = f"✗ {status_msg}"
                
                elapsed = (time.time() - scan_start) / 60
                print(f"  [{completed:2d}/{len(worker_args)}] seed {seed_done:3d}: "
                      f"{status_str}   (elapsed: {elapsed:.1f} min)", flush=True)
            
            except Exception as e:
                completed += 1
                elapsed = (time.time() - scan_start) / 60
                print(f"  [{completed:2d}/{len(worker_args)}] seed {seed:3d}: "
                      f"✗ crashed: {e}   (elapsed: {elapsed:.1f} min)", flush=True)

    print(f"\n{'='*70}")
    print(f"✓ ALL WORKERS RETURNED  —  {time.time()-scan_start:.1f} s total")
    print(f"{'='*70}")

    # =========================================================================
    # Error-budget summary (averaged across all seeds × redshifts × bins)
    # =========================================================================
    print(f"\n=== ERROR BUDGET SUMMARY (averaged across seeds) ===")

    all_sample_err = []
    all_cosmic_err = []
    all_total_err  = []

    for seed, ccr in cross_corr_results_all.items():
        for z_21cm, res in ccr.items():
            C     = res['C_cross_1d']
            valid = ~np.isnan(C) & (C != 0)
            if np.any(valid):
                all_sample_err.extend(
                    (res['C_cross_1d_err_sample'][valid]
                     / np.abs(C[valid])).tolist())
                all_cosmic_err.extend(
                    (res['C_cross_1d_err_cosmic'][valid]
                     / np.abs(C[valid])).tolist())
                all_total_err.extend(
                    (res['C_cross_1d_err_total'][valid]
                     / np.abs(C[valid])).tolist())

    if len(all_sample_err) > 0:
        print(f"  Sample variance (mean): "
              f"{np.nanmean(all_sample_err)*100:.1f}%")
        print(f"  Cosmic variance (mean): "
              f"{np.nanmean(all_cosmic_err)*100:.1f}%")
        print(f"  Total  variance (mean): "
              f"{np.nanmean(all_total_err)*100:.1f}%")

    print(f"\n{'='*70}")
    print(f"✓ CROSS-CORRELATIONS READY: "
          f"{len(cross_corr_results_all)}/{N_SEEDS} SEEDS")
    print(f"{'='*70}")

else:
    print("\n✗ Skipping — no lightcones available")

# %%
# =============================================================================
# CELL 7b: kSZ²-21cm² Cross-Correlation Power Spectra (All Seeds)
# Following Zhou et al. (2025, ApJ 991 195).
#
# Pipeline per seed and per redshift chunk (z₀, Δz):
#   1. Extract 3D 21cm brightness-temperature chunk over [z₀-Δz/2, z₀+Δz/2].
#   2. Foreground-filter in 3D Fourier space (kill k∥ < k_par_min).
#   3. IFFT → square in config space (key step: Zhou+25 Appendix A).
#   4. Top-hat project along LoS → 2D T21² map.
#   5. Cross-power with the squared kSZ map; bin in 2D k → ℓ via Limber.
#
# Cache: <main_cache_dir>/seed_<N>/cross_corr_sq_seed<N>.npy
# (Same pattern as Cell 7 — single .npy per seed, dict keyed by z₀.)
# =============================================================================

print("\n" + "="*70)
print("COMPUTING kSZ²-21cm² CROSS-CORRELATION POWER SPECTRA")
print("="*70)

if len(lightcones) > 0:

    # =========================================================================
    # Map geometry + k-space grid (seed-independent, hoist out of loop)
    # =========================================================================
    npix_side    = user_params.HII_DIM
    box_size_Mpc = float(user_params.BOX_LEN)
    pix_size_Mpc = box_size_Mpc / npix_side
    pix_area     = pix_size_Mpc**2

    dk        = 2 * np.pi / (npix_side * pix_size_Mpc)
    kx        = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
    ky        = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
    kgrid     = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
    k_bins    = np.logspace(np.log10(dk), np.log10(kgrid.max() * 0.9), 35)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    # 3D Fourier frequencies (kx, ky seed-independent; kz depends on chunk size)
    kx_1d = np.fft.fftfreq(npix_side, d=pix_size_Mpc) * 2 * np.pi
    ky_1d = np.fft.fftfreq(npix_side, d=pix_size_Mpc) * 2 * np.pi

    # =========================================================================
    # Foreground-filter parameters (Zhou+25 Section 2.4)
    # =========================================================================
    k_par_min = K_PAR_MIN
    # wedge_slope = 3.0  # alternative: foregrounds wedge, k∥ > m·k⊥

    # =========================================================================
    # Redshift chunks (z₀ centres, Δz width)
    # =========================================================================
    delta_z         = DELTA_Z
    z_chunk_centres = Z_CHUNK_CENTRES

    print(f"  Foreground filter : k∥ > {k_par_min} h/Mpc (optimistic)")
    print(f"  Chunk width       : Δz = {delta_z}")
    print(f"  Chunk centres     : {list(z_chunk_centres)}")
    print(f"  N seeds to process: {N_SEEDS}")

    # Ensure cosmo exists (Cell 7 normally defines it; redeclare defensively)
    if 'cosmo' not in dir() and 'cosmo' not in globals():
        cosmo = FlatLambdaCDM(H0=67.77, Om0=0.3086)

    # Ensure kSZ_maps dict exists (Cell 6 normally creates it)
    if 'kSZ_maps' not in dir() and 'kSZ_maps' not in globals():
        kSZ_maps = {}

    kSZ_maps_dir = os.path.join(main_cache_dir, "kSZ_maps")

    # =========================================================================
    # Per-seed loop
    # =========================================================================
    cross_corr_results_sq_all = {}   # {seed: {z0: result_dict}}

    for seed, lc in lightcones.items():

        seed_idx = list(lightcones.keys()).index(seed)
        print(f"\n{'='*60}")
        print(f"SEED {seed}  ({seed_idx+1}/{N_SEEDS})  (kSZ²–21cm²)")
        print(f"{'='*60}")

        seed_cache_dir = os.path.join(main_cache_dir, f"seed_{seed}")
        os.makedirs(seed_cache_dir, exist_ok=True)
        cc_cache_sq = os.path.join(seed_cache_dir,
                                   f"cross_corr_sq_seed{seed}.npy")

        # ------------------------------------------------------------------
        # CASE 1: cached result exists → load
        # ------------------------------------------------------------------
        if os.path.exists(cc_cache_sq):
            print(f"  Found cached result → loading")
            cross_corr_results_sq_all[seed] = np.load(
                cc_cache_sq, allow_pickle=True).item()
            print(f"  ✓ Loaded {len(cross_corr_results_sq_all[seed])} chunks")
            continue

        # ------------------------------------------------------------------
        # CASE 2: fetch the kSZ map (memory → Cell 6 cache → skip)
        # ------------------------------------------------------------------
        kSZ_map = None

        if seed in kSZ_maps and kSZ_maps[seed] is not None:
            print(f"  kSZ map found in memory")
            kSZ_map = kSZ_maps[seed]
        else:
            map_path = os.path.join(
                kSZ_maps_dir, f"kSZ_map_z{z_obs:.1f}_seed{seed}.npy"
            )
            if os.path.exists(map_path):
                print(f"  kSZ map not in memory → loading from Cell 6 cache")
                kSZ_map = np.load(map_path)
                kSZ_maps[seed] = kSZ_map
            else:
                print(f"  ✗ No kSZ map available — skipping seed")
                print(f"    Re-run Cells 5 and 6 to generate the kSZ map")
                continue

        # ------------------------------------------------------------------
        # CASE 3: compute. Start with the squared kSZ map (FFT once)
        # ------------------------------------------------------------------
        kSZ2_map          = kSZ_map**2
        kSZ2_map_centered = kSZ2_map - np.mean(kSZ2_map)
        fft_kSZ2_shifted  = np.fft.fftshift(np.fft.fft2(kSZ2_map_centered))
        auto_kSZ2_ps2d    = (np.abs(fft_kSZ2_shifted)**2
                             * pix_area / npix_side**2)

        lc_redshifts = np.asarray(lc.lightcone_redshifts, dtype=np.float64)
        cross_corr_results_sq = {}
        loop_start = time.time()

        # ------------------------------------------------------------------
        # Loop over redshift chunks
        # ------------------------------------------------------------------
        for z0 in z_chunk_centres:

            z_lo = z0 - delta_z / 2.0
            z_hi = z0 + delta_z / 2.0

            # ----------------------------------------------------------
            # 1. Extract 3D T21 chunk
            # ----------------------------------------------------------
            idx_chunk = np.where(
                (lc_redshifts >= z_lo) & (lc_redshifts <= z_hi)
            )[0]
            if len(idx_chunk) < 3:
                print(f"  z0={z0:.1f}: too few slices ({len(idx_chunk)}) "
                      f"— skipping")
                continue

            T21_chunk = np.asarray(
                lc.brightness_temp[:, :, idx_chunk], dtype=np.float64
            )
            n_los        = T21_chunk.shape[2]
            pix_size_los = pix_size_Mpc       # approx (comoving Mpc per slice)

            # ----------------------------------------------------------
            # 2. Foreground filter in 3D Fourier space
            # ----------------------------------------------------------
            T21_fft3d = np.fft.fftn(T21_chunk)

            kz_1d = np.fft.fftfreq(n_los, d=pix_size_los) * 2 * np.pi
            kx_3d = kx_1d[:, None, None]
            ky_3d = ky_1d[None, :, None]
            kz_3d = kz_1d[None, None, :]
            kperp = np.sqrt(kx_3d**2 + ky_3d**2)
            kpar  = np.abs(kz_3d)

            # Optimistic high-pass on k∥
            fore_filter = (kpar > k_par_min).astype(float)
            # Wedge alternative (comment the above):
            # fore_filter = (kpar > wedge_slope * kperp).astype(float)

            T21_fft3d_filtered = T21_fft3d * fore_filter

            # ----------------------------------------------------------
            # 3. IFFT → config space → square (Zhou+25 Appendix A: must
            #    square AFTER filter and BEFORE projection)
            # ----------------------------------------------------------
            T21_filtered = np.real(np.fft.ifftn(T21_fft3d_filtered))
            T21_sq_3d    = T21_filtered**2

            # ----------------------------------------------------------
            # 4. Project 3D → 2D via top-hat average along LoS
            # ----------------------------------------------------------
            T21_sq_2d         = np.mean(T21_sq_3d, axis=2)
            T21_sq_centered   = T21_sq_2d - np.mean(T21_sq_2d)
            fft_T21sq_shifted = np.fft.fftshift(np.fft.fft2(T21_sq_centered))

            # ----------------------------------------------------------
            # 5. Cross-power: C_ℓ^{kSZ²×21cm²} and autos
            # ----------------------------------------------------------
            cross_ps2d     = (np.real(np.conj(fft_kSZ2_shifted)
                                      * fft_T21sq_shifted)
                              * pix_area / npix_side**2)
            auto_T21sq_ps2d = (np.abs(fft_T21sq_shifted)**2
                               * pix_area / npix_side**2)

            C_cross     = np.zeros(len(k_centers))
            C_cross_err = np.zeros(len(k_centers))
            P_T21sq     = np.zeros(len(k_centers))
            P_kSZ2      = np.zeros(len(k_centers))
            n_modes     = np.zeros(len(k_centers))

            for j in range(len(k_centers)):
                mask  = (kgrid >= k_bins[j]) & (kgrid < k_bins[j+1])
                n_pix = np.sum(mask)
                if n_pix > 0:
                    cv             = cross_ps2d[mask]
                    C_cross[j]     = np.mean(cv)
                    C_cross_err[j] = np.std(cv) / np.sqrt(n_pix)
                    P_T21sq[j]     = np.mean(auto_T21sq_ps2d[mask])
                    P_kSZ2[j]      = np.mean(auto_kSZ2_ps2d[mask])
                    n_modes[j]     = n_pix
                else:
                    C_cross[j]     = np.nan
                    C_cross_err[j] = np.nan
                    P_T21sq[j]     = np.nan
                    P_kSZ2[j]      = np.nan
                    n_modes[j]     = 0

            # ----------------------------------------------------------
            # 6. Limber k → ℓ at chunk centre; D_ℓ convention
            # ----------------------------------------------------------
            chi_z0  = float(cosmo.comoving_distance(z0).value)
            ell     = k_centers * chi_z0
            D_cross = ell * (ell + 1) * C_cross / (2 * np.pi)

            cross_corr_results_sq[float(z0)] = {
                'z0'          : float(z0),
                'delta_z'     : float(delta_z),
                'k_par_min'   : float(k_par_min),
                'n_los_slices': int(n_los),
                'k_centers'   : k_centers,
                'ell'         : ell,
                'C_cross'     : C_cross,
                'C_cross_err' : C_cross_err,
                'D_cross'     : D_cross,
                'P_T21sq'     : P_T21sq,
                'P_kSZ2'      : P_kSZ2,
                'n_modes'     : n_modes,
            }

            elapsed = time.time() - loop_start
            print(f"  z0={z0:.1f}  Δz={delta_z}  slices={n_los}  "
                  f"peak|D|={np.nanmax(np.abs(D_cross)):.2e}  "
                  f"t={elapsed:.1f}s")

        np.save(cc_cache_sq, cross_corr_results_sq)
        print(f"  ✓ Cached to {cc_cache_sq}")
        cross_corr_results_sq_all[seed] = cross_corr_results_sq

    print(f"\n{'='*70}")
    print(f"✓ kSZ²-21cm² CROSS-CORRELATIONS READY: "
          f"{len(cross_corr_results_sq_all)}/{N_SEEDS} SEEDS")
    print(f"{'='*70}")

else:
    print("\n✗ Skipping — no lightcones available")

print("\n" + "="*70)

# %%
# =============================================================================
# CELL 7c: Diagnostic Plots for kSZ²-21cm² Pipeline
#   diag1 — Foreground wedge in (k⊥, k∥): 3 filter scenarios
#   diag2 — 21cm lightcone with chunk boundaries marked
#   diag3 — T21 (mean) | filtered T21 (central slice) | T21² (mean) per chunk
#   diag4 — k∥ power spectrum before/after filter, plus retention fraction
#   diag5 — D_ℓ vs ℓ per chunk + per-bin SNR (single seed)
#
# All plots routed through save_pdf_png — no font/grid overrides.
# =============================================================================

print("\n" + "="*70)
print("DIAGNOSTIC PLOTS FOR kSZ²-21cm² PIPELINE")
print("="*70)

if ('cross_corr_results_sq_all' in dir() or
    'cross_corr_results_sq_all' in globals()) \
   and len(cross_corr_results_sq_all) > 0:

    # =========================================================================
    # Setup
    # =========================================================================
    plot_dir_diag = os.path.join(plot_dir, "diagnostics_cell7b")
    os.makedirs(plot_dir_diag, exist_ok=True)
    print(f"Diagnostic plots → {plot_dir_diag}")

    diag_seed    = next(iter(cross_corr_results_sq_all))
    diag_lc      = lightcones[diag_seed]
    lc_redshifts = np.asarray(diag_lc.lightcone_redshifts, dtype=np.float64)

    # Recover parameters from first chunk result
    first_z0       = next(iter(cross_corr_results_sq_all[diag_seed]))
    first_res      = cross_corr_results_sq_all[diag_seed][first_z0]
    k_par_min_diag = first_res['k_par_min']
    delta_z_diag   = first_res['delta_z']

    box_size_Mpc_diag = float(user_params.BOX_LEN)
    npix_side_diag    = user_params.HII_DIM
    pix_size_Mpc_diag = box_size_Mpc_diag / npix_side_diag

    # cosmo expected from earlier cells; redeclare defensively
    if 'cosmo' not in dir() and 'cosmo' not in globals():
        cosmo = FlatLambdaCDM(H0=67.77, Om0=0.3086)

    print(f"\nDiagnostic seed : {diag_seed}")
    print(f"k_par_min       : {k_par_min_diag} h/Mpc")
    print(f"delta_z         : {delta_z_diag}")

    z_chunk_centres = sorted(cross_corr_results_sq_all[diag_seed].keys())

    # =========================================================================
    # DIAG 1: Foreground filter scenarios in (k⊥, k∥) space
    # =========================================================================
    print("\n=== DIAGNOSTIC 1: Foreground wedge filter ===")

    kperp_1d = np.linspace(0, 1.5, 300)
    kpar_1d  = np.linspace(0, 1.5, 300)
    KPERP, KPAR = np.meshgrid(kperp_1d, kpar_1d)

    diag1_scenarios = [
        {
            'label':  r'Optimistic: $k_{\parallel} > k_{\parallel,0}$'
                      + f'\n$k_{{\\parallel,0}}={k_par_min_diag}$ h/Mpc',
            'filter': (KPAR > k_par_min_diag).astype(float),
        },
        {
            'label':  r'Wedge $m=3$' + '\n'
                      + r'$k_{\parallel} > m \cdot k_\perp$',
            'filter': (KPAR > 3.0 * KPERP).astype(float),
        },
        {
            'label':  r'Wedge $m=5$' + '\n'
                      + r'$k_{\parallel} > m \cdot k_\perp$',
            'filter': (KPAR > 5.0 * KPERP).astype(float),
        },
    ]

    def _draw_diag1(axes):
        for ax, sc in zip(axes, diag1_scenarios):
            ax.pcolormesh(kperp_1d, kpar_1d, sc['filter'],
                          cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
            ax.contourf(KPERP, KPAR, sc['filter'],
                        levels=[-0.5, 0.5], colors=['#d62728'], alpha=0.35)
            ax.contourf(KPERP, KPAR, sc['filter'],
                        levels=[0.5, 1.5], colors=['#2ca02c'], alpha=0.25)

            for m_ref, ls, lbl in zip([3, 5], ['--', ':'],
                                      [r'$m=3$', r'$m=5$']):
                ax.plot(kperp_1d, m_ref * kperp_1d,
                        color='white', ls=ls, lw=1.5, alpha=0.8, label=lbl)

            if k_par_min_diag > 0:
                ax.axhline(k_par_min_diag, color='gold', lw=1.5, ls='-.',
                           label=rf'$k_{{\parallel,0}}={k_par_min_diag}$')

            ax.set_xlabel(r'$k_\perp\;[h\,\mathrm{Mpc}^{-1}]$')
            ax.set_ylabel(r'$k_\parallel\;[h\,\mathrm{Mpc}^{-1}]$')
            ax.set_xlim(0, 1.5)
            ax.set_ylim(0, 1.5)
            ax.legend(loc='upper right', framealpha=0.7)

            # Per-panel subtitle via text (since ax.set_title gets cleared
            # in the PDF pass by save_pdf_png)
            ax.text(0.5, 1.02, sc['label'], transform=ax.transAxes,
                    ha='center', va='bottom')

            ax.text(0.05, 0.95, 'REMOVED\n(foreground)',
                    transform=ax.transAxes,
                    color='#d62728', fontweight='bold',
                    va='top', ha='left')
            ax.text(0.55, 0.15, 'KEPT\n(cosmological)',
                    transform=ax.transAxes,
                    color='#2ca02c', fontweight='bold')
    with mpl.rc_context(PNG_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
        _draw_diag1(axes)
        fig.suptitle(r"Foreground Filter Scenarios in $(k_\perp, k_\parallel)$ Space", fontweight="bold")
        fig.savefig(f"{plot_dir_diag}/diag1_foreground_wedge.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print("✓ Saved: diag1_foreground_wedge")

    # =========================================================================
    # DIAG 2: 21cm lightcone with chunk boundaries
    # =========================================================================
    print("\n=== DIAGNOSTIC 2: Chunk extraction on lightcone ===")

    T21_lc    = np.asarray(diag_lc.brightness_temp, dtype=np.float64)
    T21_slice = T21_lc[:, npix_side_diag // 2, :]
    diag2_vmax = float(np.percentile(np.abs(T21_slice), 98))

    cmap_chunks_d2 = plt.cm.plasma
    chunk_norm_d2  = mpl.colors.Normalize(
        vmin=min(z_chunk_centres), vmax=max(z_chunk_centres),
    )

    def _draw_diag2(ax):
        im = ax.imshow(
            T21_slice.T, aspect='auto', origin='lower',
            cmap='RdBu_r', vmin=-diag2_vmax, vmax=diag2_vmax,
            extent=[0, npix_side_diag,
                    lc_redshifts.min(), lc_redshifts.max()],
        )
        ax.figure.colorbar(im, ax=ax, label=r'$T_{21}$ [mK]', pad=0.01)

        for z0 in z_chunk_centres:
            z_lo = z0 - delta_z_diag / 2.0
            z_hi = z0 + delta_z_diag / 2.0
            z_lo_c = max(z_lo, lc_redshifts.min())
            z_hi_c = min(z_hi, lc_redshifts.max())
            if z_lo_c >= z_hi_c:
                continue
            color = cmap_chunks_d2(chunk_norm_d2(z0))
            ax.axhspan(z_lo_c, z_hi_c, alpha=0.25, color=color, lw=0)
            ax.axhline(z_lo_c, color=color, lw=0.8, ls='--', alpha=0.7)
            ax.axhline(z_hi_c, color=color, lw=0.8, ls='--', alpha=0.7)
            ax.text(npix_side_diag * 0.98,
                    0.5 * (z_lo_c + z_hi_c),
                    f'$z_0={z0:.1f}$',
                    color=color, ha='right', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='k', alpha=0.5))

        ax.set_xlabel('Transverse pixel index')
        ax.set_ylabel(r'Redshift $z$')

        sm = mpl.cm.ScalarMappable(cmap=cmap_chunks_d2, norm=chunk_norm_d2)
        sm.set_array([])
        ax.figure.colorbar(sm, ax=ax, location='top', pad=0.01,
                           label=r'Chunk centre $z_0$', aspect=40)

    save_pdf_png(
        _draw_diag2, plot_dir_diag, "diag2_chunk_extraction",
        title=(f'21cm Lightcone (seed={diag_seed}) with Chunk Boundaries '
               f'($\\Delta z={delta_z_diag}$)'),
        figsize=(16, 5),
    )
    print("✓ Saved: diag2_chunk_extraction")

    # =========================================================================
    # DIAG 3: T21 (mean) / filtered T21 (central slice) / T21² (mean)
    # Pre-compute everything outside the closure
    # =========================================================================
    print("\n=== DIAGNOSTIC 3: T21 vs T21² maps across chunks ===")

    z_to_show = z_chunk_centres[::max(1, len(z_chunk_centres) // 4)][:4]
    n_cols_d3 = len(z_to_show)

    diag3_panels = []   # list of dicts, one per column

    z_nodes_sorted = diag_lc.node_redshifts[::-1]
    x_e_nodes      = 1.0 - diag_lc.global_xH[::-1]

    for z0 in z_to_show:
        z_lo = z0 - delta_z_diag / 2.0
        z_hi = z0 + delta_z_diag / 2.0
        idx_chunk = np.where(
            (lc_redshifts >= z_lo) & (lc_redshifts <= z_hi)
        )[0]
        if len(idx_chunk) < 2:
            diag3_panels.append(None)
            continue

        T21_chunk = np.asarray(
            diag_lc.brightness_temp[:, :, idx_chunk], dtype=np.float64
        )
        n_los = T21_chunk.shape[2]

        # Comoving los spacing (correct: uses chunk depth, not transverse pix)
        chi_lo    = float(cosmo.comoving_distance(z_lo).value)
        chi_hi    = float(cosmo.comoving_distance(z_hi).value)
        depth_Mpc = abs(chi_hi - chi_lo)
        d_los_Mpc = depth_Mpc / n_los

        kz_1d = np.fft.fftfreq(n_los, d=d_los_Mpc) * 2 * np.pi
        kpar  = np.abs(kz_1d[None, None, :])

        fore_filter = (kpar > k_par_min_diag).astype(float)
        T21_fft3d   = np.fft.fftn(T21_chunk)
        T21_filt    = np.real(np.fft.ifftn(T21_fft3d * fore_filter))
        T21_sq_3d   = T21_filt**2

        mid_los  = n_los // 2
        T21_proj    = np.mean(T21_chunk, axis=2)
        T21_f_slice = T21_filt[:, :, mid_los]
        T21_sq_proj = np.mean(T21_sq_3d, axis=2)

        xe_z0 = float(np.interp(z0, z_nodes_sorted, x_e_nodes))

        diag3_panels.append({
            'z0'         : float(z0),
            'xe'         : xe_z0,
            'n_slices'   : int(len(idx_chunk)),
            'd_los_Mpc'  : d_los_Mpc,
            'mid_los'    : mid_los,
            'n_los'      : n_los,
            'T21_proj'   : T21_proj,
            'T21_f_slice': T21_f_slice,
            'T21_sq_proj': T21_sq_proj,
        })

    def _draw_diag3(axes):
        for col, panel in enumerate(diag3_panels):
            if panel is None:
                for r in range(3):
                    axes[r, col].set_axis_off()
                continue

            col_title = (f"$z_0={panel['z0']:.1f}$\n"
                         f"$x_e={panel['xe']:.2f}$ "
                         f"({panel['n_slices']} slices)\n"
                         f"$d_{{\\rm los}}={panel['d_los_Mpc']:.1f}$ Mpc/slice")

            # Row 0: raw T21 mean projection
            ax = axes[0, col]
            v = float(np.percentile(np.abs(panel['T21_proj']), 98))
            im = ax.imshow(panel['T21_proj'].T, origin='lower',
                           cmap='RdBu_r', vmin=-v, vmax=v, aspect='equal')
            ax.figure.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xticks([]); ax.set_yticks([])
            ax.text(0.5, 1.02, col_title, transform=ax.transAxes,
                    ha='center', va='bottom')
            if col == 0:
                ax.set_ylabel(
                    r'Raw $\langle T_{21}\rangle_{\Delta z}$ [mK]'
                )

            # Row 1: filtered T21 — central slice (mean would be ~0)
            ax = axes[1, col]
            v = float(np.percentile(np.abs(panel['T21_f_slice']), 98))
            im = ax.imshow(panel['T21_f_slice'].T, origin='lower',
                           cmap='RdBu_r', vmin=-v, vmax=v, aspect='equal')
            ax.figure.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(
                    r'Filtered $T_{21}^f$ [mK]' + '\n'
                    r'(central slice, $z_{\rm mid}$)'
                )
            ax.text(0.03, 0.03,
                    f"slice {panel['mid_los']}/{panel['n_los']}",
                    transform=ax.transAxes, color='white',
                    bbox=dict(boxstyle='round,pad=0.2',
                              facecolor='k', alpha=0.5))

            # Row 2: filtered+squared, projected
            ax = axes[2, col]
            v = float(np.percentile(panel['T21_sq_proj'], 98))
            im = ax.imshow(panel['T21_sq_proj'].T, origin='lower',
                           cmap='hot', vmin=0, vmax=v, aspect='equal')
            ax.figure.colorbar(im, ax=ax, fraction=0.046)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(
                    r'Filtered+Squared '
                    r'$\langle(T_{21}^f)^2\rangle_{\Delta z}$ [mK$^2$]'
                )
    with mpl.rc_context(PNG_STYLE):
        fig, axes = plt.subplots(3, n_cols_d3, figsize=(4.5 * n_cols_d3, 13), constrained_layout=True)
        _draw_diag3(axes)
        fig.savefig(f"{plot_dir_diag}/diag3_T21_vs_T21sq_maps.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print("✓ Saved: diag3_T21_vs_T21sq_maps")

    # =========================================================================
    # DIAG 4: k∥ power before/after filter; retention fraction
    # =========================================================================
    print("\n=== DIAGNOSTIC 4: k∥ power spectrum before/after filter ===")

    diag4_data = []
    cmap_z_d4  = plt.cm.plasma
    z_norm_d4  = mpl.colors.Normalize(
        vmin=min(z_chunk_centres), vmax=max(z_chunk_centres),
    )

    for z0 in z_chunk_centres:
        z_lo = z0 - delta_z_diag / 2.0
        z_hi = z0 + delta_z_diag / 2.0
        idx_chunk = np.where(
            (lc_redshifts >= z_lo) & (lc_redshifts <= z_hi)
        )[0]
        if len(idx_chunk) < 2:
            continue

        T21_chunk = np.asarray(
            diag_lc.brightness_temp[:, :, idx_chunk], dtype=np.float64
        )
        n_los = T21_chunk.shape[2]

        chi_lo    = float(cosmo.comoving_distance(z_lo).value)
        chi_hi    = float(cosmo.comoving_distance(z_hi).value)
        d_los_Mpc = abs(chi_hi - chi_lo) / n_los

        kz_1d  = np.fft.fftfreq(n_los, d=d_los_Mpc) * 2 * np.pi
        kz_pos = np.fft.fftshift(kz_1d)

        T21_fft3d  = np.fft.fftn(T21_chunk)
        power_3d   = np.abs(T21_fft3d)**2
        power_kpar = np.mean(power_3d.reshape(-1, n_los), axis=0)
        power_kpar = np.fft.fftshift(power_kpar)

        kz_abs       = np.abs(kz_pos)
        filter_1d    = (kz_abs > k_par_min_diag).astype(float)
        power_kpar_f = power_kpar * filter_1d
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_retained = np.where(power_kpar > 0,
                                     power_kpar_f / power_kpar, 0.0)

        diag4_data.append({
            'z0'           : float(z0),
            'kz_pos'       : kz_pos,
            'power_kpar'   : power_kpar,
            'power_kpar_f' : power_kpar_f,
            'frac_retained': frac_retained,
            'd_los_Mpc'    : d_los_Mpc,
            'kz_min'       : float(kz_pos.min()),
            'kz_max'       : float(kz_pos.max()),
            'frac_kept'    : float(np.mean(kz_abs > k_par_min_diag)),
        })

    for d in diag4_data:
        print(f"  z0={d['z0']:.1f}: k∥ range "
              f"[{d['kz_min']:.3f}, {d['kz_max']:.3f}] h/Mpc  |  "
              f"frac kept = {d['frac_kept']:.3f}  |  "
              f"d_los = {d['d_los_Mpc']:.2f} Mpc/slice")

    def _draw_diag4(axes):
        ax_ps, ax_frac = axes

        for d in diag4_data:
            color = cmap_z_d4(z_norm_d4(d['z0']))
            label = f"$z_0={d['z0']:.1f}$"
            ax_ps.semilogy(d['kz_pos'], d['power_kpar'],
                           color=color, lw=1.5, alpha=0.5, ls='--')
            ax_ps.semilogy(d['kz_pos'], d['power_kpar_f'],
                           color=color, lw=2.0, alpha=0.9, ls='-',
                           label=label)
            ax_frac.plot(d['kz_pos'], d['frac_retained'],
                         color=color, lw=1.8, label=label)

        for ax in axes:
            ax.axvline( k_par_min_diag, color='gold', lw=2, ls='-.',
                        label=rf'$k_{{\parallel,\min}}={k_par_min_diag}$ h/Mpc')
            ax.axvline(-k_par_min_diag, color='gold', lw=2, ls='-.')

        ax_ps.set_xlabel(r'$k_\parallel\;[h\,\mathrm{Mpc}^{-1}]$')
        ax_ps.set_ylabel(r'$P(k_\parallel)$ [arb. units]')
        ax_ps.set_xlim(-1.0, 1.0)
        ax_ps.text(0.5, 1.02,
                   '21cm power along $k_\\parallel$\n'
                   '(dashed=unfiltered, solid=filtered)',
                   transform=ax_ps.transAxes, ha='center', va='bottom')

        ax_frac.axhline(0.5, color='gray', ls=':', lw=1)
        ax_frac.axhline(1.0, color='gray', ls=':', lw=1)
        ax_frac.set_xlabel(r'$k_\parallel\;[h\,\mathrm{Mpc}^{-1}]$')
        ax_frac.set_ylabel('Fraction of power retained')
        ax_frac.set_ylim(-0.05, 1.15)
        ax_frac.set_xlim(-1.0, 1.0)
        ax_frac.text(0.5, 1.02, 'Power retained after filter',
                     transform=ax_frac.transAxes, ha='center', va='bottom')

        sm = mpl.cm.ScalarMappable(cmap=cmap_z_d4, norm=z_norm_d4)
        sm.set_array([])
        for ax in axes:
            ax.figure.colorbar(sm, ax=ax,
                               label=r'Chunk centre $z_0$', pad=0.01)
    with mpl.rc_context(PNG_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
        _draw_diag4(axes)
        fig.savefig(f"{plot_dir_diag}/diag4_kpar_power_filter.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print("✓ Saved: diag4_kpar_power_filter")

    # =========================================================================
    # DIAG 5: D_ℓ vs ℓ per chunk + per-bin SNR
    # =========================================================================
    print("\n=== DIAGNOSTIC 5: D_ℓ vs ℓ per chunk (single seed) ===")

    ccr = cross_corr_results_sq_all[diag_seed]
    z_vals_d5 = sorted(ccr.keys())
    cmap_z_d5 = plt.cm.coolwarm
    z_norm_d5 = mpl.colors.Normalize(vmin=min(z_vals_d5), vmax=max(z_vals_d5))

    diag5_curves = []
    z_nodes_sorted_d5 = diag_lc.node_redshifts[::-1]
    x_e_nodes_d5      = 1.0 - diag_lc.global_xH[::-1]

    for z0 in z_vals_d5:
        res   = ccr[z0]
        ell   = res['ell']
        D     = res['D_cross']
        D_err = res['C_cross_err'] * ell * (ell + 1) / (2 * np.pi)
        xe_z0 = float(np.interp(z0, z_nodes_sorted_d5, x_e_nodes_d5))
        valid = np.isfinite(D) & (ell > 10)
        with np.errstate(divide='ignore', invalid='ignore'):
            snr = np.where(D_err > 0, np.abs(D) / D_err, np.nan)

        diag5_curves.append({
            'z0'   : float(z0),
            'xe'   : xe_z0,
            'ell'  : ell[valid],
            'D'    : D[valid],
            'D_err': D_err[valid],
            'snr'  : snr[valid],
        })

    def _draw_diag5(axes):
        ax_top, ax_bot = axes
        for c in diag5_curves:
            color = cmap_z_d5(z_norm_d5(c['z0']))
            ax_top.plot(c['ell'], c['D'], color=color, lw=1.8, alpha=0.85,
                        label=f"$z_0={c['z0']:.1f}$ ($x_e={c['xe']:.2f}$)")
            ax_top.fill_between(c['ell'],
                                c['D'] - c['D_err'],
                                c['D'] + c['D_err'],
                                color=color, alpha=0.12)
            ax_bot.plot(c['ell'], c['snr'], color=color, lw=1.8, alpha=0.85)

        ax_top.axhline(0, color='k', ls='--', lw=1, alpha=0.5)
        ax_top.set_ylabel(
            r'$\ell(\ell+1)C_\ell^{\rm kSZ^2\times 21cm^2}/2\pi$'
        )
        ax_top.set_xscale('log')
        ax_top.legend(ncol=2, loc='best', framealpha=0.7)

        ax_bot.axhline(1, color='gray', ls=':', lw=1.2, label='SNR=1')
        ax_bot.set_ylabel(r'$|D_\ell|/\sigma_{D_\ell}$ (per-bin SNR)')
        ax_bot.set_xlabel(r'Multipole $\ell$')
        ax_bot.set_yscale('log')
        ax_bot.legend()

        sm = mpl.cm.ScalarMappable(cmap=cmap_z_d5, norm=z_norm_d5)
        sm.set_array([])
        ax_top.figure.colorbar(sm, ax=list(axes),
                               label=r'Chunk centre $z_0$', pad=0.01)
    with mpl.rc_context(PNG_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(11, 10), constrained_layout=True, sharex=True)
        _draw_diag5(axes)
        fig.savefig(f"{plot_dir_diag}/diag5_cross_power_per_chunk.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    print("✓ Saved: diag5_cross_power_per_chunk")

    print(f"\n{'='*70}")
    print(f"✓ ALL DIAGNOSTICS COMPLETE → {plot_dir_diag}")
    print(f"  diag1: Foreground wedge in (k⊥, k∥) — 3 filter scenarios")
    print(f"  diag2: Lightcone with chunk boundaries in redshift units")
    print(f"  diag3: T21 (mean) | filtered T21 (central slice) | T21² (mean)")
    print(f"  diag4: k∥ power spectrum before/after filter per chunk")
    print(f"  diag5: D_ℓ vs ℓ + per-bin SNR for all chunks (single seed)")
    print(f"{'='*70}")

else:
    print("\n✗ Skipping — no cross_corr_results_sq_all available")
    print("    Run Cell 7b first to compute kSZ²-21cm² cross-spectra.")

# %% [markdown]
# # =============================================================================
# # NOT FOR REPORTS
# # PLOT: kSZ, kSZ², and 21cm Maps Side-by-Side at Selected Redshifts
# # =============================================================================
# 
# print(f"\n=== PLOTTING kSZ vs kSZ² vs 21cm MAPS ===")
# 
# # Select a few representative redshifts to plot
# # Get ionization fraction at each redshift
# z_nodes_sorted = lightcone.node_redshifts[::-1]
# x_e_nodes = 1.0 - lightcone.global_xH[::-1]
# 
# # Find redshifts closest to x_e = 0.2, 0.5, 0.9
# target_xe = [0.2, 0.5, 0.9]
# selected_z_plot = []
# 
# for xe_target in target_xe:
#     idx = np.argmin(np.abs(x_e_nodes - xe_target))
#     z_sel = z_nodes_sorted[idx]
#     # Find closest node redshift from our results
#     z_closest = min(cross_corr_results.keys(), key=lambda z: abs(z - z_sel))
#     if abs(z_closest - z_sel) < 0.5:  # Reasonable match
#         selected_z_plot.append(z_closest)
# 
# print(f"Plotting kSZ vs kSZ² vs 21cm for {len(selected_z_plot)} redshifts")
# 
# # Create figure: rows = redshifts, cols = [kSZ, kSZ², 21cm]
# fig, axes = plt.subplots(len(selected_z_plot), 3, 
#                          figsize=(16, 5*len(selected_z_plot)), 
#                          constrained_layout=True)
# 
# if len(selected_z_plot) == 1:
#     axes = axes.reshape(1, -1)
# 
# # Get lightcone redshift axis
# lc_redshifts = np.asarray(lightcone.lightcone_redshifts, dtype=np.float64)
# 
# for row_idx, z_obs in enumerate(selected_z_plot):
#     
#     # Load kSZ map
#     kSZ_map_file = f"{kSZ_maps_dir}/kSZ_map_z{z_obs:.6f}.npy"
#     kSZ_map = np.load(kSZ_map_file)
#     
#     # Square it
#     kSZ2_map = kSZ_map**2
#     
#     # Find closest lightcone slice to z_obs
#     idx_closest = np.argmin(np.abs(lc_redshifts - z_obs))
#     z_actual = lc_redshifts[idx_closest]
#     
#     # Extract 21cm brightness temperature slice
#     T21_slice = np.asarray(lightcone.brightness_temp[:, :, idx_closest])
#     
#     # Get ionization fraction
#     x_e = np.interp(z_obs, z_nodes_sorted, x_e_nodes)
#     
#     # =============================================================================
#     # Left panel: kSZ map
#     # =============================================================================
#     
#     ax_kSZ = axes[row_idx, 0]
#     
#     # Symmetric color scale for kSZ
#     vmax_kSZ = np.percentile(np.abs(kSZ_map), 99)
#     
#     im_kSZ = ax_kSZ.imshow(kSZ_map.T,
#                            cmap='seismic',
#                            origin='lower',
#                            extent=[0, box_size_Mpc, 0, box_size_Mpc],
#                            aspect='equal',
#                            vmin=-vmax_kSZ,
#                            vmax=vmax_kSZ)
#     
#     # Colorbar
#     cbar_kSZ = plt.colorbar(im_kSZ, ax=ax_kSZ, fraction=0.046, pad=0.04)
#     cbar_kSZ.set_label('kSZ (dimensionless)', fontsize=12)
#     
#     # Labels
#     ax_kSZ.set_xlabel('x [Mpc]', fontsize=14)
#     ax_kSZ.set_ylabel('y [Mpc]', fontsize=14)
#     ax_kSZ.set_title(f'kSZ Map\nz={z_obs:.2f}, $x_e$={x_e:.2f}', 
#                      fontsize=14, fontweight='bold')
#     
#     # Stats
#     rms_kSZ = np.sqrt(np.mean(kSZ_map**2))
#     ax_kSZ.text(0.05, 0.95, 
#                f'RMS={rms_kSZ:.2e}\nMean={kSZ_map.mean():.2e}',
#                transform=ax_kSZ.transAxes, fontsize=11,
#                verticalalignment='top',
#                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#     
#     # =============================================================================
#     # Middle panel: kSZ² map
#     # =============================================================================
#     
#     ax_kSZ2 = axes[row_idx, 1]
#     
#     # Use 'hot' colormap for squared map (all positive)
#     vmax_kSZ2 = np.percentile(kSZ2_map, 99)
#     
#     im_kSZ2 = ax_kSZ2.imshow(kSZ2_map.T,
#                              cmap='hot',
#                              origin='lower',
#                              extent=[0, box_size_Mpc, 0, box_size_Mpc],
#                              aspect='equal',
#                              vmin=0,
#                              vmax=vmax_kSZ2)
#     
#     # Colorbar
#     cbar_kSZ2 = plt.colorbar(im_kSZ2, ax=ax_kSZ2, fraction=0.046, pad=0.04)
#     cbar_kSZ2.set_label('kSZ² (dimensionless)', fontsize=12)
#     
#     # Labels
#     ax_kSZ2.set_xlabel('x [Mpc]', fontsize=14)
#     ax_kSZ2.set_ylabel('y [Mpc]', fontsize=14)
#     ax_kSZ2.set_title(f'kSZ² Map\nz={z_obs:.2f}, $x_e$={x_e:.2f}', 
#                       fontsize=14, fontweight='bold')
#     
#     # Stats
#     rms_kSZ2 = np.sqrt(np.mean(kSZ2_map**2))
#     ax_kSZ2.text(0.05, 0.95, 
#                 f'RMS={rms_kSZ2:.2e}\nMean={kSZ2_map.mean():.2e}',
#                 transform=ax_kSZ2.transAxes, fontsize=11,
#                 verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
#     
#     # =============================================================================
#     # Right panel: 21cm brightness temperature map
#     # =============================================================================
#     
#     ax_T21 = axes[row_idx, 2]
#     
#     # Use 'RdBu_r' or 'coolwarm' for 21cm (can be positive or negative)
#     vmax_T21 = np.percentile(np.abs(T21_slice), 99)
#     
#     im_T21 = ax_T21.imshow(T21_slice.T,
#                            cmap='RdBu_r',
#                            origin='lower',
#                            extent=[0, box_size_Mpc, 0, box_size_Mpc],
#                            aspect='equal',
#                            vmin=-vmax_T21,
#                            vmax=vmax_T21)
#     
#     # Colorbar
#     cbar_T21 = plt.colorbar(im_T21, ax=ax_T21, fraction=0.046, pad=0.04)
#     cbar_T21.set_label('21cm Brightness Temp [mK]', fontsize=12)
#     
#     # Labels
#     ax_T21.set_xlabel('x [Mpc]', fontsize=14)
#     ax_T21.set_ylabel('y [Mpc]', fontsize=14)
#     ax_T21.set_title(f'21cm Map\nz={z_actual:.2f}, $x_e$={x_e:.2f}', 
#                      fontsize=14, fontweight='bold')
#     
#     # Stats
#     rms_T21 = np.sqrt(np.mean(T21_slice**2))
#     ax_T21.text(0.05, 0.95, 
#                f'RMS={rms_T21:.2f} mK\nMean={T21_slice.mean():.2f} mK',
#                transform=ax_T21.transAxes, fontsize=11,
#                verticalalignment='top',
#                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
# 
# # Overall title
# fig.suptitle('kSZ, kSZ², and 21cm Maps at Different Reionization Epochs', 
#             fontsize=20, fontweight='bold')
# 
# # Save
# plot_name = "kSZ_kSZ2_21cm_maps_comparison"
# fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
# fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')
# 
# print(f"✓ Saved: {plot_name}")
# plt.close(fig)
# 
# print("\n✓ kSZ vs kSZ² vs 21cm COMPARISON PLOTTING COMPLETE!")

# %% [markdown]
# # =============================================================================
# # NOT FOR REPORTS
# # PLOT 1: 2D FFT Maps (k-space) - kSZ² and 21cm Side-by-Side
# # =============================================================================
# 
# print(f"\n=== PLOTTING 2D FFT MAPS IN k-SPACE ===")
# 
# # Select a few representative redshifts to plot
# # Get ionization fraction at each redshift
# z_nodes_sorted = lightcone.node_redshifts[::-1]
# x_e_nodes = 1.0 - lightcone.global_xH[::-1]
# 
# # Find redshifts closest to x_e = 0.2, 0.5, 0.9
# target_xe = [0.2, 0.5, 0.9]
# selected_z_fft = []
# 
# for xe_target in target_xe:
#     idx = np.argmin(np.abs(x_e_nodes - xe_target))
#     z_sel = z_nodes_sorted[idx]
#     # Find closest node redshift from our results
#     z_closest = min(cross_corr_results.keys(), key=lambda z: abs(z - z_sel))
#     if abs(z_closest - z_sel) < 0.5:  # Reasonable match
#         selected_z_fft.append(z_closest)
# 
# print(f"Plotting 2D FFT for {len(selected_z_fft)} redshifts")
# 
# # Create figure: rows = redshifts, cols = [kSZ² FFT, 21cm FFT]
# fig, axes = plt.subplots(len(selected_z_fft), 2, 
#                          figsize=(14, 6*len(selected_z_fft)), 
#                          constrained_layout=True)
# 
# if len(selected_z_fft) == 1:
#     axes = axes.reshape(1, -1)
# 
# # Get lightcone redshift axis
# lc_redshifts = np.asarray(lightcone.lightcone_redshifts, dtype=np.float64)
# 
# for row_idx, z_obs in enumerate(selected_z_fft):
#     
#     # Recompute FFTs for this redshift
#     # Load kSZ map
#     kSZ_map_file = f"{kSZ_maps_dir}/kSZ_map_z{z_obs:.6f}.npy"
#     kSZ_map = np.load(kSZ_map_file)
#     kSZ2_map = kSZ_map**2
#     kSZ2_map_centered = kSZ2_map - np.mean(kSZ2_map)
#     
#     # Get 21cm slice
#     idx_closest = np.argmin(np.abs(lc_redshifts - z_obs))
#     T21_slice = np.asarray(lightcone.brightness_temp[:, :, idx_closest])
#     T21_slice_centered = T21_slice - np.mean(T21_slice)
#     
#     # Compute FFTs
#     fft_kSZ2 = np.fft.fft2(kSZ2_map_centered)
#     fft_kSZ2_shifted = np.fft.fftshift(fft_kSZ2)
#     
#     fft_T21 = np.fft.fft2(T21_slice_centered)
#     fft_T21_shifted = np.fft.fftshift(fft_T21)
#     
#     # Get ionization fraction
#     x_e = np.interp(z_obs, z_nodes_sorted, x_e_nodes)
#     
#     # k-space extent
#     k_max = kgrid.max()
#     
#     # =============================================================================
#     # Left panel: kSZ² FFT
#     # =============================================================================
#     
#     ax_fft_kSZ2 = axes[row_idx, 0]
#     
#     # Plot log10 of power
#     power_kSZ2 = np.abs(fft_kSZ2_shifted)**2
#     power_kSZ2_log = np.log10(power_kSZ2 + 1e-20)  # Add small value to avoid log(0)
#     
#     im_fft_kSZ2 = ax_fft_kSZ2.imshow(power_kSZ2_log.T,
#                                       cmap='viridis',
#                                       origin='lower',
#                                       extent=[-k_max, k_max, -k_max, k_max],
#                                       aspect='equal')
#     
#     # Colorbar
#     cbar_fft_kSZ2 = plt.colorbar(im_fft_kSZ2, ax=ax_fft_kSZ2, fraction=0.046, pad=0.04)
#     cbar_fft_kSZ2.set_label(r'log$_{10}$(Power)', fontsize=12)
#     
#     # Labels
#     ax_fft_kSZ2.set_xlabel(r'$k_x$ [Mpc$^{-1}$]', fontsize=14)
#     ax_fft_kSZ2.set_ylabel(r'$k_y$ [Mpc$^{-1}$]', fontsize=14)
#     ax_fft_kSZ2.set_title(f'kSZ² Power (k-space)\nz={z_obs:.2f}, $x_e$={x_e:.2f}', 
#                           fontsize=14, fontweight='bold')
#     
#     # Add circle at k = 0.1 Mpc^-1 for reference
#     circle = plt.Circle((0, 0), 0.1, color='white', fill=False, linestyle='--', linewidth=1.5)
#     ax_fft_kSZ2.add_patch(circle)
#     
#     # =============================================================================
#     # Right panel: 21cm FFT
#     # =============================================================================
#     
#     ax_fft_T21 = axes[row_idx, 1]
#     
#     # Plot log10 of power
#     power_T21 = np.abs(fft_T21_shifted)**2
#     power_T21_log = np.log10(power_T21 + 1e-20)
#     
#     im_fft_T21 = ax_fft_T21.imshow(power_T21_log.T,
#                                     cmap='viridis',
#                                     origin='lower',
#                                     extent=[-k_max, k_max, -k_max, k_max],
#                                     aspect='equal')
#     
#     # Colorbar
#     cbar_fft_T21 = plt.colorbar(im_fft_T21, ax=ax_fft_T21, fraction=0.046, pad=0.04)
#     cbar_fft_T21.set_label(r'log$_{10}$(Power)', fontsize=12)
#     
#     # Labels
#     ax_fft_T21.set_xlabel(r'$k_x$ [Mpc$^{-1}$]', fontsize=14)
#     ax_fft_T21.set_ylabel(r'$k_y$ [Mpc$^{-1}$]', fontsize=14)
#     ax_fft_T21.set_title(f'21cm Power (k-space)\nz={z_obs:.2f}, $x_e$={x_e:.2f}', 
#                          fontsize=14, fontweight='bold')
#     
#     # Add circle at k = 0.1 Mpc^-1 for reference
#     circle = plt.Circle((0, 0), 0.1, color='white', fill=False, linestyle='--', linewidth=1.5)
#     ax_fft_T21.add_patch(circle)
# 
# # Overall title
# fig.suptitle('2D Power Spectra in k-space', 
#             fontsize=20, fontweight='bold')
# 
# # Save
# plot_name = "2D_FFT_kspace_kSZ2_21cm"
# fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
# fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')
# 
# print(f"✓ Saved: {plot_name}")
# plt.close(fig)
# 
# # =============================================================================
# # PLOT 2: Cross-Power and Auto-Power Spectra vs k
# # =============================================================================
# 
# print(f"\n=== PLOTTING POWER SPECTRA vs k ===")
# 
# # Plot for the same selected redshifts
# fig, axes = plt.subplots(len(selected_z_fft), 1, 
#                          figsize=(10, 6*len(selected_z_fft)), 
#                          constrained_layout=True)
# 
# if len(selected_z_fft) == 1:
#     axes = [axes]
# 
# for row_idx, z_obs in enumerate(selected_z_fft):
#     
#     if z_obs not in cross_corr_results:
#         continue
#     
#     results = cross_corr_results[z_obs]
#     k_centers = results['k_centers']
#     C_cross = results['C_cross_1d']
#     P_kSZ2 = results['P_kSZ2_1d']
#     P_T21 = results['P_T21_1d']
#     
#     # Get ionization fraction
#     x_e = np.interp(z_obs, z_nodes_sorted, x_e_nodes)
#     
#     ax = axes[row_idx]
#     
#     # Filter valid points
#     valid_cross = ~np.isnan(C_cross) & np.isfinite(C_cross)
#     valid_kSZ2 = ~np.isnan(P_kSZ2) & (P_kSZ2 > 0)
#     valid_T21 = ~np.isnan(P_T21) & (P_T21 > 0)
#     
#     # Plot auto-power spectra
#     ax.loglog(k_centers[valid_kSZ2], P_kSZ2[valid_kSZ2], 
#              'o-', color='red', linewidth=2, markersize=4,
#              label='kSZ² Auto-Power', alpha=0.8)
#     
#     ax.loglog(k_centers[valid_T21], P_T21[valid_T21], 
#              's-', color='blue', linewidth=2, markersize=4,
#              label='21cm Auto-Power', alpha=0.8)
#     
#     # Plot cross-power (can be negative, so plot absolute value)
#     # Use different markers for positive/negative
#     positive_mask = valid_cross & (C_cross > 0)
#     negative_mask = valid_cross & (C_cross < 0)
#     
#     if np.any(positive_mask):
#         ax.loglog(k_centers[positive_mask], np.abs(C_cross[positive_mask]), 
#                  '^-', color='green', linewidth=2.5, markersize=6,
#                  label='|Cross-Power| (positive)', alpha=0.9)
#     
#     if np.any(negative_mask):
#         ax.loglog(k_centers[negative_mask], np.abs(C_cross[negative_mask]), 
#                  'v--', color='purple', linewidth=2.5, markersize=6,
#                  label='|Cross-Power| (negative)', alpha=0.9)
#     
#     ax.set_xlabel(r'$k$ [Mpc$^{-1}$]', fontsize=16)
#     ax.set_ylabel(r'Power [Mpc$^2$]', fontsize=16)
#     ax.set_title(f'Power Spectra: z={z_obs:.2f}, $x_e$={x_e:.2f}', 
#                 fontsize=16, fontweight='bold')
#     ax.legend(fontsize=12, loc='best')
#     #ax.grid(True, alpha=0.3)
#     
#     # Add text showing sign of cross-power
#     if np.any(valid_cross):
#         mean_sign = "positive" if np.mean(C_cross[valid_cross]) > 0 else "negative"
#         ax.text(0.05, 0.95, f'Cross-power: {mean_sign}',
#                transform=ax.transAxes, fontsize=12, fontweight='bold',
#                verticalalignment='top',
#                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
# 
# # Overall title
# fig.suptitle('Power Spectra vs k (kSZ², 21cm, and Cross-Power)', 
#             fontsize=18, fontweight='bold')
# 
# # Save
# plot_name = "power_spectra_vs_k_comparison"
# fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
# fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')
# 
# print(f"✓ Saved: {plot_name}")
# plt.close(fig)
# 
# # =============================================================================
# # PLOT 3: Cross-Power Sign Evolution vs k at Different Redshifts
# # =============================================================================
# 
# print(f"\n=== PLOTTING CROSS-POWER SIGN EVOLUTION ===")
# 
# fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)
# 
# # Select more redshifts for this plot
# z_sample = sorted(cross_corr_results.keys())[::15]  # Every 15th redshift
# 
# cmap = mpl.cm.rainbow
# norm = mpl.colors.Normalize(vmin=min(z_sample), vmax=max(z_sample))
# 
# for z_obs in z_sample:
#     results = cross_corr_results[z_obs]
#     k_centers = results['k_centers']
#     C_cross = results['C_cross_1d']
#     
#     valid = ~np.isnan(C_cross) & np.isfinite(C_cross)
#     
#     if np.sum(valid) > 5:
#         color = cmap(norm(z_obs))
#         
#         # Plot with sign preserved (use symlog or just regular plot)
#         ax.plot(k_centers[valid], C_cross[valid], 
#                color=color, linewidth=2, alpha=0.7,
#                marker='o', markersize=3)
# 
# ax.set_xlabel(r'$k$ [Mpc$^{-1}$]', fontsize=18)
# ax.set_ylabel(r'Cross-Power [Mpc$^2$]', fontsize=18)
# ax.set_xscale('log')
# ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
# #ax.grid(True, alpha=0.3)
# 
# # Add colorbar
# sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# cbar = plt.colorbar(sm, ax=ax, pad=0.02)
# cbar.set_label(r'Redshift $z$', fontsize=16)
# 
# ax.set_title('kSZ²-21cm Cross-Power vs k (Sign Evolution)', 
#             fontsize=18, fontweight='bold')
# 
# # Save
# plot_name = "cross_power_vs_k_sign_evolution"
# fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight')
# fig.savefig(f"{plot_dir}/{plot_name}.png", dpi=300, bbox_inches='tight')
# 
# print(f"✓ Saved: {plot_name}")
# plt.close(fig)
# 
# print("\n✓ ALL DIAGNOSTIC PLOTTING COMPLETE!")

# %%
# =============================================================================
# CELL 8a: Visualize kSZ²-21cm Cross-Correlation Power Spectra (Random Seed)
# Convert to ℓ-space and create five plots for one randomly chosen realisation.
# All plotting via save_pdf_png — no font/grid overrides anywhere.
# =============================================================================

from astropy.cosmology import FlatLambdaCDM

print("\n" + "="*70)
print("VISUALIZING kSZ²-21cm CROSS-CORRELATION POWER SPECTRA")
print("="*70)

# Subdirectory for these final plots
plot_dir_final = os.path.join(plot_dir, "plot_final_cell")
os.makedirs(plot_dir_final, exist_ok=True)
plot_dir_save = plot_dir_final

# =============================================================================
# Load cross_corr_results_all from cache if not already in memory
# =============================================================================
if ('cross_corr_results_all' not in dir()
        and 'cross_corr_results_all' not in globals()) \
   or len(cross_corr_results_all) == 0:

    print("cross_corr_results_all not in memory → loading from Cell 7 cache")
    cross_corr_results_all = {}
    for seed in RANDOM_SEEDS:
        cc_cache = os.path.join(
            main_cache_dir, f"seed_{seed}", f"cross_corr_seed{seed}.npy"
        )
        if os.path.exists(cc_cache):
            cross_corr_results_all[seed] = np.load(
                cc_cache, allow_pickle=True).item()
            print(f"  ✓ Loaded seed {seed} "
                  f"({len(cross_corr_results_all[seed])} redshifts)")
        else:
            print(f"  ✗ No cache found for seed {seed}")
    print(f"  Loaded {len(cross_corr_results_all)}/{N_SEEDS} seeds")
else:
    print(f"cross_corr_results_all already in memory "
          f"({len(cross_corr_results_all)} seeds)")


if len(cross_corr_results_all) > 0:

    # Pick one random seed
    seed_to_plot = int(np.random.choice(list(cross_corr_results_all.keys())))
    print(f"\nRandomly selected seed for plots: {seed_to_plot}")

    cross_corr_results = cross_corr_results_all[seed_to_plot]
    lc                 = lightcones[seed_to_plot]
    slabel             = f"seed{seed_to_plot}"

    # =========================================================================
    # Convert k-space → ℓ-space (per redshift)
    # =========================================================================
    print(f"\n=== CONVERTING TO ℓ-SPACE WITH ERROR PROPAGATION ===")

    T_CMB_0_K = 2.725
    cosmo     = FlatLambdaCDM(H0=67.77, Om0=0.3086)

    cross_corr_ell_results = {}

    for z_node in sorted(cross_corr_results.keys()):
        results          = cross_corr_results[z_node]
        D_A_Mpc          = float(cosmo.angular_diameter_distance(z_node).value)
        chi_comoving_Mpc = float(cosmo.comoving_distance(z_node).value)
        T_CMB_z_uK       = T_CMB_0_K * 1e6

        k_centers  = results['k_centers']
        ell_from_k = k_centers * chi_comoving_Mpc / 0.67

        C_cross_ell            = results['C_cross_1d']            * 0.67**2 / D_A_Mpc**2
        C_cross_ell_err_sample = results['C_cross_1d_err_sample'] * 0.67**2 / D_A_Mpc**2
        C_cross_ell_err_cosmic = results['C_cross_1d_err_cosmic'] * 0.67**2 / D_A_Mpc**2
        C_cross_ell_err_total  = results['C_cross_1d_err_total']  * 0.67**2 / D_A_Mpc**2

        D_cross_ell            = ell_from_k * (ell_from_k + 1) * C_cross_ell            / (2 * np.pi)
        D_cross_ell_err_sample = ell_from_k * (ell_from_k + 1) * C_cross_ell_err_sample / (2 * np.pi)
        D_cross_ell_err_cosmic = ell_from_k * (ell_from_k + 1) * C_cross_ell_err_cosmic / (2 * np.pi)
        D_cross_ell_err_total  = ell_from_k * (ell_from_k + 1) * C_cross_ell_err_total  / (2 * np.pi)

        D_cross_ell_uK_mK            = D_cross_ell            * T_CMB_z_uK**2
        D_cross_ell_uK_mK_err_sample = D_cross_ell_err_sample * T_CMB_z_uK**2
        D_cross_ell_uK_mK_err_cosmic = D_cross_ell_err_cosmic * T_CMB_z_uK**2
        D_cross_ell_uK_mK_err_total  = D_cross_ell_err_total  * T_CMB_z_uK**2

        P_kSZ2_ell = results['P_kSZ2_1d'] * 0.67**2 / D_A_Mpc**2
        P_T21_ell  = results['P_T21_1d']  * 0.67**2 / D_A_Mpc**2
        with np.errstate(divide='ignore', invalid='ignore'):
            r_cross = C_cross_ell / np.sqrt(P_kSZ2_ell * P_T21_ell)

        cross_corr_ell_results[z_node] = {
            'ell_from_k'                  : ell_from_k,
            'D_cross_ell_uK_mK'           : D_cross_ell_uK_mK,
            'D_cross_ell_uK_mK_err_sample': D_cross_ell_uK_mK_err_sample,
            'D_cross_ell_uK_mK_err_cosmic': D_cross_ell_uK_mK_err_cosmic,
            'D_cross_ell_uK_mK_err_total' : D_cross_ell_uK_mK_err_total,
            'D_cross_ell_dimensionless'   : D_cross_ell,
            'r_cross'                     : r_cross,
            'D_A_Mpc'                     : D_A_Mpc,
            'T_CMB_z_uK'                  : T_CMB_z_uK,
        }

    print(f"Converted {len(cross_corr_ell_results)} redshifts to ℓ-space")

    z_values = np.array(sorted(cross_corr_ell_results.keys()))
    cmap     = mpl.cm.rainbow
    norm     = mpl.colors.Normalize(vmin=z_values.min(), vmax=z_values.max())

    # =========================================================================
    # PLOT 1: Rainbow D_ℓ vs ℓ
    # =========================================================================
    print(f"\n=== PLOT 1: Rainbow D_ℓ vs ℓ ===")

    def _draw_p1(ax):
        for z_node in z_values[::2]:
            results   = cross_corr_ell_results[z_node]
            ell       = results['ell_from_k']
            D_ell     = results['D_cross_ell_uK_mK']
            D_ell_err = results['D_cross_ell_uK_mK_err_total']
            valid = (~np.isnan(D_ell) & np.isfinite(D_ell)
                     & (ell > 10) & ~np.isnan(D_ell_err))
            if np.sum(valid) > 5:
                color = cmap(norm(z_node))
                ax.plot(ell[valid], D_ell[valid],
                        color=color, lw=1.5, alpha=0.8)
                ax.fill_between(ell[valid],
                                D_ell[valid] - D_ell_err[valid],
                                D_ell[valid] + D_ell_err[valid],
                                color=color, alpha=0.15)

        ax.set_xlabel(r'Multipole $\ell$')
        ax.set_ylabel(r'$D_\ell$ [kSZ$^2$-21cm] (μK$^2$·mK)')
        ax.set_xscale('log')
        ax.axhline(0, color='black', ls='--', lw=1, alpha=0.5)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ax.figure.colorbar(sm, ax=ax, pad=0.02).set_label(r'Redshift $z$')

        ax.text(0.02, 0.02, f'seed={seed_to_plot}',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    save_pdf_png(
        _draw_p1, plot_dir_save,
        f"kSZ2_21cm_cross_Dl_vs_ell_rainbow_{slabel}",
        title=r'kSZ$^2$-21cm Cross-Power $D_\ell$ vs Redshift',
    )
    print(f"✓ Saved: kSZ2_21cm_cross_Dl_vs_ell_rainbow_{slabel}")

    # =========================================================================
    # PLOT 2: Correlation Coefficient r vs ℓ
    # =========================================================================
    print(f"\n=== PLOT 2: Correlation Coefficient r vs ℓ ===")

    def _draw_p2(ax):
        for z_node in z_values:
            results = cross_corr_ell_results[z_node]
            ell     = results['ell_from_k']
            r       = results['r_cross']
            valid = (~np.isnan(r) & np.isfinite(r)
                     & (ell > 10) & (np.abs(r) < 1.5))
            if np.sum(valid) > 5:
                ax.plot(ell[valid], r[valid],
                        color=cmap(norm(z_node)), lw=1.5, alpha=0.7)

        ax.set_xlabel(r'Multipole $\ell$')
        ax.set_ylabel(r'Correlation Coefficient $r$')
        ax.set_xscale('log')
        ax.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
        ax.set_ylim(-1.2, 1.2)

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ax.figure.colorbar(sm, ax=ax, pad=0.02).set_label(r'Redshift $z$')

        ax.text(0.02, 0.02, f'seed={seed_to_plot}',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    save_pdf_png(
        _draw_p2, plot_dir_save,
        f"kSZ2_21cm_cross_r_vs_ell_rainbow_{slabel}",
        title=r'kSZ$^2$-21cm Correlation Coefficient vs Redshift',
    )
    print(f"✓ Saved: kSZ2_21cm_cross_r_vs_ell_rainbow_{slabel}")

    # =========================================================================
    # PLOT 3: Selected x_e with error bars
    # =========================================================================
    print(f"\n=== PLOT 3: Selected ionization fractions ===")

    z_nodes_sorted = lc.node_redshifts[::-1]
    x_e_nodes      = 1.0 - lc.global_xH[::-1]
    target_xe      = [0.2, 0.5, 0.9]
    selected_z = [z_nodes_sorted[np.argmin(np.abs(x_e_nodes - xe))]
                  for xe in target_xe]
    selected_xe = [x_e_nodes[np.argmin(np.abs(x_e_nodes - xe))]
                   for xe in target_xe]

    print("  Selected redshifts:")
    for z, xe in zip(selected_z, selected_xe):
        print(f"    z={z:.2f}, x_e={xe:.3f}")

    colors_selected = ['blue', 'green', 'red']

    def _draw_p3(ax):
        for i, (z_node, xe) in enumerate(zip(selected_z, selected_xe)):
            z_closest = min(cross_corr_ell_results.keys(),
                            key=lambda z: abs(z - z_node))
            if abs(z_closest - z_node) > 0.5:
                continue
            results          = cross_corr_ell_results[z_closest]
            ell              = results['ell_from_k']
            D_ell            = results['D_cross_ell_uK_mK']
            D_ell_err_total  = results['D_cross_ell_uK_mK_err_total']
            D_ell_err_sample = results['D_cross_ell_uK_mK_err_sample']
            valid = (~np.isnan(D_ell) & np.isfinite(D_ell)
                     & (ell > 10) & ~np.isnan(D_ell_err_total))
            if np.sum(valid) > 5:
                ax.errorbar(ell[valid], D_ell[valid],
                            yerr=D_ell_err_total[valid],
                            color=colors_selected[i], lw=2.5, alpha=0.8,
                            marker='o', markersize=5,
                            capsize=3, capthick=1.5,
                            label=f'z={z_closest:.1f} ($x_e$={xe:.2f})',
                            errorevery=3)
                ax.fill_between(ell[valid],
                                D_ell[valid] - D_ell_err_sample[valid],
                                D_ell[valid] + D_ell_err_sample[valid],
                                color=colors_selected[i], alpha=0.15)

        ax.set_xlabel(r'Multipole $\ell$')
        ax.set_ylabel(r'$D_\ell$ [kSZ$^2$-21cm] (μK$^2$·mK)')
        ax.set_xscale('log')
        ax.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
        ax.legend(loc='best', framealpha=0.9)

        ax.text(0.02, 0.02, f'seed={seed_to_plot}',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    save_pdf_png(
        _draw_p3, plot_dir_save,
        f"kSZ2_21cm_cross_Dl_selected_xe_{slabel}",
        title=r'kSZ$^2$-21cm Cross-Power at Key Ionization Fractions',
    )
    print(f"✓ Saved: kSZ2_21cm_cross_Dl_selected_xe_{slabel}")

    # =========================================================================
    # PLOT 4: D_ℓ vs z at fixed ℓ
    # =========================================================================
    print(f"\n=== PLOT 4: D_ℓ evolution at fixed ℓ ===")

    ell_targets = [500, 1000, 3000]
    colors_ell  = ['darkblue', 'darkgreen', 'darkred']

    def _draw_p4(ax):
        for i, ell_target in enumerate(ell_targets):
            z_plot, D_plot, D_err_plot = [], [], []
            for z_node in sorted(cross_corr_ell_results.keys()):
                results = cross_corr_ell_results[z_node]
                ell     = results['ell_from_k']
                D_ell   = results['D_cross_ell_uK_mK']
                D_err   = results['D_cross_ell_uK_mK_err_total']
                idx     = np.argmin(np.abs(ell - ell_target))
                if np.isfinite(D_ell[idx]) and np.isfinite(D_err[idx]):
                    z_plot.append(z_node)
                    D_plot.append(D_ell[idx])
                    D_err_plot.append(D_err[idx])

            if len(z_plot) > 0:
                z_plot     = np.array(z_plot)
                D_plot     = np.array(D_plot)
                D_err_plot = np.array(D_err_plot)
                ax.errorbar(z_plot, D_plot, yerr=D_err_plot,
                            color=colors_ell[i], lw=2.5, alpha=0.8,
                            marker='o', markersize=5,
                            capsize=4, capthick=1.5,
                            label=f'$\\ell$={ell_target}', errorevery=2)
                ax.fill_between(z_plot,
                                D_plot - D_err_plot,
                                D_plot + D_err_plot,
                                color=colors_ell[i], alpha=0.15)

        ax.set_xlabel(r'Redshift $z$')
        ax.set_ylabel(r'$D_\ell$ [kSZ$^2$-21cm] (μK$^2$·mK)')
        ax.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
        ax.legend(loc='best', framealpha=0.9)
        ax.invert_xaxis()

        ax.text(0.05, 0.95, f'seed={seed_to_plot}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    save_pdf_png(
        _draw_p4, plot_dir_save,
        f"kSZ2_21cm_cross_Dl_vs_z_fixed_ell_{slabel}",
        title=r'kSZ$^2$-21cm Cross-Power Evolution at Fixed $\ell$',
    )
    print(f"✓ Saved: kSZ2_21cm_cross_Dl_vs_z_fixed_ell_{slabel}")

    # =========================================================================
    # PLOT 5: Error Budget (two stacked panels)
    # =========================================================================
    print(f"\n=== PLOT 5: Error budget ===")

    z_example = min(cross_corr_ell_results.keys(),
                    key=lambda z: abs(z - selected_z[1]))
    results          = cross_corr_ell_results[z_example]
    ell_p5           = results['ell_from_k']
    D_ell_p5         = results['D_cross_ell_uK_mK']
    D_err_sample_p5  = results['D_cross_ell_uK_mK_err_sample']
    D_err_cosmic_p5  = results['D_cross_ell_uK_mK_err_cosmic']
    D_err_total_p5   = results['D_cross_ell_uK_mK_err_total']
    valid_p5 = ~np.isnan(D_ell_p5) & (ell_p5 > 10) & (D_ell_p5 != 0)

    def _draw_p5_custom(plot_dir, plot_name, title, seed_label, z_ex):
        """Custom function for two-panel error budget plot."""
        with mpl.rc_context(PDF_STYLE):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10),
                                            constrained_layout=True, sharex=True)
            
            # Top panel: D_ℓ with total error
            ax1.errorbar(ell_p5[valid_p5], D_ell_p5[valid_p5],
                        yerr=D_err_total_p5[valid_p5],
                        fmt='o-', color='darkblue', lw=2, markersize=4,
                        capsize=3, alpha=0.8,
                        label='Cross-power ± total error')
            ax1.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
            ax1.set_ylabel(r'$D_\ell$ [μK$^2$·mK]')
            ax1.set_xscale('log')
            ax1.legend()
            ax1.grid(False)

            # Bottom panel: fractional error decomposition
            frac_sample = (D_err_sample_p5[valid_p5]
                          / np.abs(D_ell_p5[valid_p5]) * 100)
            frac_cosmic = (D_err_cosmic_p5[valid_p5]
                          / np.abs(D_ell_p5[valid_p5]) * 100)
            frac_total  = (D_err_total_p5[valid_p5]
                          / np.abs(D_ell_p5[valid_p5]) * 100)

            ax2.plot(ell_p5[valid_p5], frac_sample, 'o-',
                    color='blue', lw=2, markersize=4, alpha=0.7,
                    label='Sample variance')
            ax2.plot(ell_p5[valid_p5], frac_cosmic, 's-',
                    color='red', lw=2, markersize=4, alpha=0.7,
                    label='Cosmic variance')
            ax2.plot(ell_p5[valid_p5], frac_total, '^-',
                    color='black', lw=2.5, markersize=5, alpha=0.8,
                    label='Total')
            ax2.set_xlabel(r'Multipole $\ell$')
            ax2.set_ylabel('Fractional Error (%)')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(False)

            fig.suptitle(title, fontsize=20, fontweight='bold')
            fig.savefig(f"{plot_dir}/{plot_name}.pdf", bbox_inches='tight', dpi=300)
            plt.close(fig)

        with mpl.rc_context(PNG_STYLE):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10),
                                            constrained_layout=True, sharex=True)
            
            # Top panel
            ax1.errorbar(ell_p5[valid_p5], D_ell_p5[valid_p5],
                        yerr=D_err_total_p5[valid_p5],
                        fmt='o-', color='darkblue', lw=2, markersize=4,
                        capsize=3, alpha=0.8,
                        label='Cross-power ± total error')
            ax1.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
            ax1.set_ylabel(r'$D_\ell$ [μK$^2$·mK]')
            ax1.set_xscale('log')
            ax1.legend()
            ax1.grid(False)

            # Bottom panel
            frac_sample = (D_err_sample_p5[valid_p5]
                          / np.abs(D_ell_p5[valid_p5]) * 100)
            frac_cosmic = (D_err_cosmic_p5[valid_p5]
                          / np.abs(D_ell_p5[valid_p5]) * 100)
            frac_total  = (D_err_total_p5[valid_p5]
                          / np.abs(D_ell_p5[valid_p5]) * 100)

            ax2.plot(ell_p5[valid_p5], frac_sample, 'o-',
                    color='blue', lw=2, markersize=4, alpha=0.7,
                    label='Sample variance')
            ax2.plot(ell_p5[valid_p5], frac_cosmic, 's-',
                    color='red', lw=2, markersize=4, alpha=0.7,
                    label='Cosmic variance')
            ax2.plot(ell_p5[valid_p5], frac_total, '^-',
                    color='black', lw=2.5, markersize=5, alpha=0.8,
                    label='Total')
            ax2.set_xlabel(r'Multipole $\ell$')
            ax2.set_ylabel('Fractional Error (%)')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(False)

            fig.suptitle(title, fontsize=16, fontweight='bold')
            fig.savefig(f"{plot_dir}/{plot_name}.png", bbox_inches='tight', dpi=300)
            plt.close(fig)

    _draw_p5_custom(
        plot_dir_save,
        f"kSZ2_21cm_cross_error_budget_{slabel}",
        f'Error Budget (z={z_example:.2f}, seed={seed_to_plot})',
        slabel, z_example
    )
    print(f"✓ Saved: kSZ2_21cm_cross_error_budget_{slabel}")

    print(f"\n✓ ALL PLOTS COMPLETE (seed={seed_to_plot})")

else:
    print("\n✗ Skipping — no cross_corr_results_all available")

print("\n" + "="*70)

# %% [markdown]
# import numpy as np
# import matplotlib.pyplot as plt
# from astropy.cosmology import Planck18 as cosmo
# 
# # Redshift range
# z = np.linspace(0.1, 20, 300)
# 
# # Distances
# chi_comoving = cosmo.comoving_distance(z).value          # Mpc
# D_A = cosmo.angular_diameter_distance(z).value           # Mpc
# 
# # Ratio
# ratio = chi_comoving/ D_A
# 
# # Plot
# fig, axs = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
# 
# # Top panel: distances
# axs[0].plot(z, chi_comoving, lw=2, label=r'Comoving distance $\chi(z)$')
# axs[0].plot(z, D_A, lw=2, label=r'Angular diameter distance $D_A(z)$')
# axs[0].set_ylabel('Distance [Mpc]')
# axs[0].legend()
# axs[0].grid(alpha=0.3)
# 
# # Bottom panel: ratio
# axs[1].plot(z, ratio, lw=2, color='black')
# axs[1].set_xlabel('Redshift $z$')
# axs[1].set_ylabel(r'$\chi(z) / D_A(z)$')
# axs[1].grid(alpha=0.3)
# 
# # Save
# fig.savefig("distance_comoving_DA_and_ratio.pdf", dpi=300, bbox_inches="tight")
# fig.savefig("distance_comoving_DA_and_ratio.png", dpi=300, bbox_inches="tight")
# 
# plt.show()

# %%
# =============================================================================
# CELL 8b: kSZ²-21cm Cross-Correlation — Seed-Averaged Plots (with symlog)
# (No redshift binning anywhere — earlier "BINNED" label was a misnomer.)
# Three plots, all via save_pdf_png:
#   1. D_ℓ vs z at ℓ=3000, NO error bars       (mean across seeds, symlog)
#   2. D_ℓ vs z at ℓ=3000, WITH error bars     (σ_seeds ⊕ σ_meas, symlog)
#   3. D_ℓ vs ℓ at selected x_e                (seed-averaged, log-log)
# =============================================================================

from astropy.cosmology import FlatLambdaCDM

print("\n" + "="*70)
print("VISUALIZING kSZ²-21cm CROSS-CORRELATION (SEED-AVERAGED)")
print("="*70)

# Subdirectory for final plots
plot_dir_final = os.path.join(plot_dir, "plot_final_cell")
os.makedirs(plot_dir_final, exist_ok=True)
plot_dir_save = plot_dir_final

# =============================================================================
# Load cross_corr_results_all from cache if not already in memory
# =============================================================================
if ('cross_corr_results_all' not in dir()
        and 'cross_corr_results_all' not in globals()) \
   or len(cross_corr_results_all) == 0:

    print("cross_corr_results_all not in memory → loading from Cell 7 cache")
    cross_corr_results_all = {}
    for seed in RANDOM_SEEDS:
        cc_cache = os.path.join(
            main_cache_dir, f"seed_{seed}", f"cross_corr_seed{seed}.npy"
        )
        if os.path.exists(cc_cache):
            cross_corr_results_all[seed] = np.load(
                cc_cache, allow_pickle=True).item()
            print(f"  ✓ Loaded seed {seed} "
                  f"({len(cross_corr_results_all[seed])} redshifts)")
        else:
            print(f"  ✗ No cache found for seed {seed}")
    print(f"  Loaded {len(cross_corr_results_all)}/{N_SEEDS} seeds")
else:
    print(f"cross_corr_results_all already in memory "
          f"({len(cross_corr_results_all)} seeds)")


if len(cross_corr_results_all) > 0:

    # =========================================================================
    # Convert k → ℓ for ALL seeds (keep per-realisation error)
    # =========================================================================
    print(f"\n=== CONVERTING TO ℓ-SPACE FOR ALL SEEDS ===")

    T_CMB_0_K = 2.725
    cosmo     = FlatLambdaCDM(H0=67.77, Om0=0.3086)

    cross_corr_ell_all = {}   # {seed: {z_node: ell_results_dict}}

    for seed, ccr in cross_corr_results_all.items():
        cross_corr_ell_results = {}

        for z_node in sorted(ccr.keys()):
            results          = ccr[z_node]
            D_A_Mpc          = float(cosmo.angular_diameter_distance(z_node).value)
            chi_comoving_Mpc = float(cosmo.comoving_distance(z_node).value)
            T_CMB_z_uK       = T_CMB_0_K * 1e6

            k_centers  = results['k_centers']
            ell_from_k = k_centers * chi_comoving_Mpc / 0.67

            C_cross_ell           = (results['C_cross_1d']
                                     * 0.67**2 / D_A_Mpc**2)
            C_cross_ell_err_total = (results['C_cross_1d_err_total']
                                     * 0.67**2 / D_A_Mpc**2)

            D_cross_ell           = (ell_from_k * (ell_from_k + 1)
                                     * C_cross_ell / (2 * np.pi))
            D_cross_ell_err_total = (ell_from_k * (ell_from_k + 1)
                                     * C_cross_ell_err_total / (2 * np.pi))

            D_cross_ell_uK_mK           = D_cross_ell           * T_CMB_z_uK**2
            D_cross_ell_uK_mK_err_total = D_cross_ell_err_total * T_CMB_z_uK**2

            P_kSZ2_ell = results['P_kSZ2_1d'] * 0.67**2 / D_A_Mpc**2
            P_T21_ell  = results['P_T21_1d']  * 0.67**2 / D_A_Mpc**2
            with np.errstate(divide='ignore', invalid='ignore'):
                r_cross = C_cross_ell / np.sqrt(P_kSZ2_ell * P_T21_ell)

            cross_corr_ell_results[z_node] = {
                'ell_from_k'                 : ell_from_k,
                'D_cross_ell_uK_mK'          : D_cross_ell_uK_mK,
                'D_cross_ell_uK_mK_err_total': D_cross_ell_uK_mK_err_total,
                'r_cross'                    : r_cross,
                'D_A_Mpc'                    : D_A_Mpc,
                'T_CMB_z_uK'                 : T_CMB_z_uK,
            }

        cross_corr_ell_all[seed] = cross_corr_ell_results

    print(f"Converted {len(cross_corr_ell_all)} seeds to ℓ-space")

    # Reference seed (for the node-redshift list and the x_e curve)
    ref_seed    = next(iter(cross_corr_ell_all))
    ref_lc      = lightcones[ref_seed]
    all_z_nodes = sorted(cross_corr_ell_all[ref_seed].keys())

    # =========================================================================
    # Seed-average D_ℓ(ℓ=3000) at every node redshift  (no redshift binning)
    # =========================================================================
    ell_target = 3000

    z_used        = []
    D_mean_per_z  = []
    sigma_seeds_per_z = []
    sigma_meas_per_z  = []
    sigma_total_per_z = []

    print(f"\n=== SEED-AVERAGING AT ℓ = {ell_target} ===")

    for z_target in all_z_nodes:
        D_seed_vals   = []
        D_err_sq_vals = []

        for seed, ell_res in cross_corr_ell_all.items():
            if z_target not in ell_res:
                continue
            res   = ell_res[z_target]
            ell   = res['ell_from_k']
            D_ell = res['D_cross_ell_uK_mK']
            D_err = res['D_cross_ell_uK_mK_err_total']
            idx   = np.argmin(np.abs(ell - ell_target))

            if np.isfinite(D_ell[idx]) and np.isfinite(D_err[idx]):
                D_seed_vals.append(D_ell[idx])
                D_err_sq_vals.append(D_err[idx]**2)

        if len(D_seed_vals) >= 2:
            D_mean      = np.mean(D_seed_vals)
            sigma_seeds = np.std(D_seed_vals, ddof=1)
            sigma_meas  = np.sqrt(np.mean(D_err_sq_vals))
            sigma_total = np.sqrt(sigma_seeds**2 + sigma_meas**2)

            z_used.append(z_target)
            D_mean_per_z.append(D_mean)
            sigma_seeds_per_z.append(sigma_seeds)
            sigma_meas_per_z.append(sigma_meas)
            sigma_total_per_z.append(sigma_total)

    z_used            = np.array(z_used)
    D_mean_per_z      = np.array(D_mean_per_z)
    sigma_seeds_per_z = np.array(sigma_seeds_per_z)
    sigma_meas_per_z  = np.array(sigma_meas_per_z)
    sigma_total_per_z = np.array(sigma_total_per_z)

    print(f"  Points: {len(z_used)} redshifts at ℓ ≈ {ell_target}")

    # =========================================================================
    # PLOT A: D_ℓ vs z at ℓ=3000 — NO error bars (symlog y-axis)
    # =========================================================================
    print(f"\n=== PLOT A: D_ℓ vs z at ℓ={ell_target}, NO error bars ===")

    def _draw_A(ax):
        ax.plot(z_used, D_mean_per_z,
                color='darkred', lw=2.5, alpha=0.9,
                marker='o', markersize=6,
                label=f'$\\ell$={ell_target}')

        ax.set_xlabel(r'Redshift $z$')
        ax.set_ylabel(r'$D_\ell$ [kSZ$^2$-21cm] (μK$^2$·mK)')
        ax.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
        ax.set_yscale('symlog', linthresh=1e-2)
        ax.legend(loc='best', framealpha=0.9)
        ax.invert_xaxis()

        # ax.text(0.05, 0.95,
        #         f'{N_SEEDS} seeds | mean only',
        #         transform=ax.transAxes, verticalalignment='top',
        #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    save_pdf_png(
        _draw_A, plot_dir_save,
        f"kSZ2_21cm_cross_Dl_vs_z_ell{ell_target}_NOERR",
        title=(r'kSZ$^2$-21cm Cross-Power $D_\ell$ vs Redshift '
               f'at $\\ell$={ell_target} (mean, {N_SEEDS} seeds)'),
    )
    print(f"✓ Saved: kSZ2_21cm_cross_Dl_vs_z_ell{ell_target}_NOERR")

    # =========================================================================
    # PLOT B: D_ℓ vs z at ℓ=3000 — WITH error bars (symlog y-axis)
    # =========================================================================
    print(f"\n=== PLOT B: D_ℓ vs z at ℓ={ell_target}, WITH error bars ===")

    def _draw_B(ax):
        ax.errorbar(z_used, D_mean_per_z, yerr=sigma_total_per_z,
                    color='darkred', lw=2.5, alpha=0.85,
                    marker='o', markersize=6,
                    capsize=4, capthick=1.5,
                    label=f'$\\ell$={ell_target}')
        ax.fill_between(z_used,
                        D_mean_per_z - sigma_total_per_z,
                        D_mean_per_z + sigma_total_per_z,
                        color='darkred', alpha=0.2)

        ax.set_xlabel(r'Redshift $z$')
        ax.set_ylabel(r'$D_\ell$ [kSZ$^2$-21cm] (μK$^2$·mK)')
        ax.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
        ax.set_yscale('symlog', linthresh=1e-2)
        ax.legend(loc='best', framealpha=0.9)
        ax.invert_xaxis()

        # ax.text(0.05, 0.95,
        #         f'{N_SEEDS} seeds | '
        #         r'$\sigma_{\rm seeds} \oplus \sigma_{\rm meas}$',
        #         transform=ax.transAxes, verticalalignment='top',
        #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    save_pdf_png(
        _draw_B, plot_dir_save,
        f"kSZ2_21cm_cross_Dl_vs_z_ell{ell_target}_ERR",
        title=(r'kSZ$^2$-21cm Cross-Power $D_\ell$ vs Redshift '
               f'at $\\ell$={ell_target} '
               f'({N_SEEDS} seeds, errors in quadrature)'),
    )
    print(f"✓ Saved: kSZ2_21cm_cross_Dl_vs_z_ell{ell_target}_ERR")

    # =========================================================================
    # PLOT C: D_ℓ vs ℓ at selected x_e (seed-averaged, log-log)
    # =========================================================================
    print(f"\n=== PLOT C: D_ℓ vs ℓ at selected x_e (seed-averaged) ===")

    z_nodes_sorted = ref_lc.node_redshifts[::-1]
    x_e_nodes      = 1.0 - ref_lc.global_xH[::-1]

    target_xe = [0.2, 0.5, 0.9]
    selected_z = [z_nodes_sorted[np.argmin(np.abs(x_e_nodes - xe))]
                  for xe in target_xe]
    selected_xe = [x_e_nodes[np.argmin(np.abs(x_e_nodes - xe))]
                   for xe in target_xe]

    print("  Selected redshifts:")
    for z, xe in zip(selected_z, selected_xe):
        print(f"    z={z:.2f}, x_e={xe:.3f}")

    colors_selected = ['blue', 'green', 'red']

    # Pre-compute per-(z_target) seed averages so the closure stays clean
    plotC_data = []   # list of (ell_ref[valid], D_mean[valid], sigma_total[valid], z_target, xe, color)

    for i, (z_target, xe) in enumerate(zip(selected_z, selected_xe)):

        D_ell_seeds   = []
        D_err_sq_list = []
        ell_ref       = None
        valid_ref     = None

        for seed, ell_res in cross_corr_ell_all.items():
            z_closest = min(ell_res.keys(), key=lambda z: abs(z - z_target))
            if abs(z_closest - z_target) > 0.5:
                continue
            res   = ell_res[z_closest]
            ell   = res['ell_from_k']
            D_ell = res['D_cross_ell_uK_mK']
            D_err = res['D_cross_ell_uK_mK_err_total']
            valid = ~np.isnan(D_ell) & np.isfinite(D_ell) & (ell > 10)
            if np.sum(valid) > 5:
                if ell_ref is None:
                    ell_ref   = ell
                    valid_ref = valid
                D_ell_seeds.append(D_ell)
                D_err_sq_list.append(D_err**2)

        if len(D_ell_seeds) == 0:
            continue

        D_matrix     = np.array(D_ell_seeds)
        D_err_sq_mat = np.array(D_err_sq_list)

        D_mean      = np.nanmean(D_matrix, axis=0)
        sigma_seeds = np.nanstd(D_matrix, ddof=1, axis=0)
        sigma_meas  = np.sqrt(np.nanmean(D_err_sq_mat, axis=0))
        sigma_total = np.sqrt(sigma_seeds**2 + sigma_meas**2)

        valid = valid_ref & ~np.isnan(D_mean)

        plotC_data.append((
            ell_ref[valid],
            D_mean[valid],
            sigma_total[valid],
            float(z_target),
            float(xe),
            colors_selected[i],
        ))

    def _draw_C(ax):
        for ell_v, D_v, sig_v, z_t, xe_t, color in plotC_data:
            ax.errorbar(ell_v, D_v, yerr=sig_v,
                        color=color, lw=2.5, alpha=0.8,
                        marker='o', markersize=5,
                        capsize=3, capthick=1.5,
                        label=f'z≈{z_t:.1f} ($x_e$={xe_t:.2f})',
                        errorevery=3)
            ax.fill_between(ell_v, D_v - sig_v, D_v + sig_v,
                            color=color, alpha=0.15)

        ax.set_xlabel(r'Multipole $\ell$')
        ax.set_ylabel(r'$D_\ell$ [kSZ$^2$-21cm] (μK$^2$·mK)')
        ax.set_xscale('log')
        ax.set_yscale('symlog', linthresh=1e-2)
        ax.axhline(0, color='black', ls='--', lw=1, alpha=0.5)
        ax.legend(loc='best', framealpha=0.9)

        # ax.text(0.02, 0.02,
        #         f'{N_SEEDS} seeds | '
        #         r'$\sigma_{\rm seeds} \oplus \sigma_{\rm meas}$',
        #         transform=ax.transAxes,
        #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    save_pdf_png(
        _draw_C, plot_dir_save,
        "kSZ2_21cm_cross_Dl_selected_xe_with_errors",
        title=(r'kSZ$^2$-21cm Cross-Power at Key Ionization Fractions'
               r' (Seed-Averaged)'),
    )
    print(f"✓ Saved: kSZ2_21cm_cross_Dl_selected_xe_with_errors")

    print("\n✓ ALL CELL 8b PLOTS COMPLETE")
    print(f"  Plots saved to: {plot_dir_save}")
    print(f"  1. kSZ2_21cm_cross_Dl_vs_z_ell{ell_target}_NOERR (symlog)")
    print(f"  2. kSZ2_21cm_cross_Dl_vs_z_ell{ell_target}_ERR (symlog)")
    print(f"  3. kSZ2_21cm_cross_Dl_selected_xe_with_errors (log-symlog)")

else:
    print("\n✗ Skipping — no cross_corr_results_all available")

print("\n" + "="*70)

# %%
# %%
# =============================================================================
# CELL 8.5: Visualize kSZ²–21cm² Cross-Correlation (single seed)
# =============================================================================

print("\n" + "="*70)
print("VISUALIZING kSZ²–21cm² CROSS-CORRELATION POWER SPECTRA (SINGLE SEED)")
print("="*70)

plot_dir_final = os.path.join(plot_dir, "plot_final_cell")
os.makedirs(plot_dir_final, exist_ok=True)
plot_dir_save = plot_dir_final

# --------------------------------------------------------------------------
# Load results (consistent with Cell 7b / 8b)
# --------------------------------------------------------------------------
if ('cross_corr_results_sq_all' not in dir() and 
    'cross_corr_results_sq_all' not in globals()) or len(cross_corr_results_sq_all) == 0:
    print("cross_corr_results_sq_all not in memory → loading from cache")
    cross_corr_results_sq_all = {}
    for seed in RANDOM_SEEDS:
        cc_cache = os.path.join(main_cache_dir, f"seed_{seed}", f"cross_corr_sq_seed{seed}.npy")
        if os.path.exists(cc_cache):
            cross_corr_results_sq_all[seed] = np.load(cc_cache, allow_pickle=True).item()
            print(f"  ✓ Loaded seed {seed} ({len(cross_corr_results_sq_all[seed])} chunks)")
        else:
            print(f"  ✗ No cache for seed {seed}")
    print(f"  Loaded {len(cross_corr_results_sq_all)}/{N_SEEDS} seeds")
else:
    print(f"cross_corr_results_sq_all already in memory ({len(cross_corr_results_sq_all)} seeds)")

if len(cross_corr_results_sq_all) > 0:
    seed_to_plot = int(np.random.choice(list(cross_corr_results_sq_all.keys())))
    print(f"\nRandomly selected seed for plots: {seed_to_plot}")

    ccr = cross_corr_results_sq_all[seed_to_plot]
    lc = lightcones[seed_to_plot]
    slabel = f"seed{seed_to_plot}"

    z_values = np.array(sorted(ccr.keys()))
    cmap = mpl.cm.rainbow
    norm = mpl.colors.Normalize(vmin=z_values.min(), vmax=z_values.max())

    # ======================================================================
    # PLOT 1: Rainbow D_ℓ vs ℓ (all chunks)
    # ======================================================================
    def _draw_p1_sq(ax):
        for z0 in z_values:
            res = ccr[z0]
            ell = res['ell']
            D_ell = res['D_cross']
            D_err = res['C_cross_err'] * ell * (ell + 1) / (2 * np.pi)
            valid = np.isfinite(D_ell) & np.isfinite(D_err) & (ell > 10)
            if np.sum(valid) < 3:
                continue
            color = cmap(norm(z0))
            ax.plot(ell[valid], D_ell[valid], color=color, lw=1.5, alpha=0.8)
            ax.fill_between(ell[valid],
                            D_ell[valid] - D_err[valid],
                            D_ell[valid] + D_err[valid],
                            color=color, alpha=0.15)

        ax.axhline(0, color='k', ls='--', lw=1, alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel(r'Multipole $\ell$')
        ax.set_ylabel(r'$\ell(\ell+1)C_\ell^{\rm kSZ^2 \times 21cm^2}/2\pi$')

        sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        ax.figure.colorbar(sm, ax=ax, pad=0.02).set_label(r'Chunk centre $z_0$')

        ax.text(0.02, 0.02,
                f'seed={seed_to_plot}\n'
                rf'$k_{{\parallel,\min}}={ccr[z_values[0]]["k_par_min"]:.3f}$ h/Mpc'
                f'\n$\Delta z={ccr[z_values[0]]["delta_z"]}$',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    save_pdf_png(
        _draw_p1_sq, plot_dir_save,
        f"kSZ2_21cm2_cross_Dl_vs_ell_rainbow_{slabel}",
        title=r'kSZ$^2$–21cm$^2$ Cross-Power $D_\ell$ vs Redshift Chunk',
        figsize=(12, 8)
    )
    print(f"✓ Saved: kSZ2_21cm2_cross_Dl_vs_ell_rainbow_{slabel}")

    # ======================================================================
    # PLOT 2: Selected ionization fractions
    # ======================================================================
    z_nodes_sorted = lc.node_redshifts[::-1]
    x_e_nodes = 1.0 - lc.global_xH[::-1]
    target_xe = [0.2, 0.5, 0.9]
    selected_z = [z_nodes_sorted[np.argmin(np.abs(x_e_nodes - xe))] for xe in target_xe]
    selected_xe = [x_e_nodes[np.argmin(np.abs(x_e_nodes - xe))] for xe in target_xe]
    colors_selected = ['blue', 'green', 'red']

    def _draw_p2_sq(ax):
        for i, (z_target, xe) in enumerate(zip(selected_z, selected_xe)):
            z_closest = min(ccr.keys(), key=lambda z: abs(z - z_target))
            if abs(z_closest - z_target) > 0.75:
                continue
            res = ccr[z_closest]
            ell = res['ell']
            D_ell = res['D_cross']
            D_err = res['C_cross_err'] * ell * (ell + 1) / (2 * np.pi)
            valid = np.isfinite(D_ell) & (ell > 10)
            if np.sum(valid) < 3:
                continue
            ax.errorbar(ell[valid], D_ell[valid], yerr=D_err[valid],
                        color=colors_selected[i], lw=2.5, alpha=0.8,
                        marker='o', markersize=5, capsize=3, capthick=1.5,
                        label=rf'$z_0={z_closest:.1f}$ ($x_e={xe:.2f}$)',
                        errorevery=3)

        ax.axhline(0, color='k', ls='--', lw=1, alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel(r'Multipole $\ell$')
        ax.set_ylabel(r'$\ell(\ell+1)C_\ell^{\rm kSZ^2 \times 21cm^2}/2\pi$')
        ax.legend(loc='best', framealpha=0.9)

        ax.text(0.02, 0.02, f'seed={seed_to_plot}',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    save_pdf_png(
        _draw_p2_sq, plot_dir_save,
        f"kSZ2_21cm2_cross_Dl_selected_xe_{slabel}",
        title=r'kSZ$^2$–21cm$^2$ at Key Ionization Fractions',
    )
    print(f"✓ Saved: kSZ2_21cm2_cross_Dl_selected_xe_{slabel}")

    # ======================================================================
    # PLOT 3: D_ℓ vs z at fixed ℓ
    # ======================================================================
    ell_targets = [500, 1000, 3000]
    colors_ell = ['darkblue', 'darkgreen', 'darkred']

    def _draw_p3_sq(ax):
        for i, ell_target in enumerate(ell_targets):
            z_plot, D_plot, D_err_plot = [], [], []
            for z0 in sorted(ccr.keys()):
                res = ccr[z0]
                ell = res['ell']
                D_ell = res['D_cross']
                D_err = res['C_cross_err'] * ell * (ell + 1) / (2 * np.pi)
                idx = np.argmin(np.abs(ell - ell_target))
                if np.isfinite(D_ell[idx]):
                    z_plot.append(z0)
                    D_plot.append(D_ell[idx])
                    D_err_plot.append(D_err[idx])
            if len(z_plot) > 0:
                ax.errorbar(np.array(z_plot), np.array(D_plot),
                            yerr=np.array(D_err_plot),
                            color=colors_ell[i], lw=2.5, alpha=0.8,
                            marker='o', markersize=5, capsize=4, capthick=1.5,
                            label=rf'$\ell={ell_target}$')

        ax.axhline(0, color='k', ls='--', lw=1, alpha=0.5)
        ax.set_xlabel(r'Chunk centre redshift $z_0$')
        ax.set_ylabel(r'$\ell(\ell+1)C_\ell^{\rm kSZ^2 \times 21cm^2}/2\pi$')
        ax.legend(loc='best', framealpha=0.9)
        ax.invert_xaxis()

        ax.text(0.05, 0.95, f'seed={seed_to_plot}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    save_pdf_png(
        _draw_p3_sq, plot_dir_save,
        f"kSZ2_21cm2_cross_Dl_vs_z_fixed_ell_{slabel}",
        title=r'kSZ$^2$–21cm$^2$ Redshift Evolution at Fixed $\ell$',
    )
    print(f"✓ Saved: kSZ2_21cm2_cross_Dl_vs_z_fixed_ell_{slabel}")

    print(f"\n✓ CELL 8.5 COMPLETE (seed={seed_to_plot})")

else:
    print("\n✗ Skipping — no cross_corr_results_sq_all available")

print("\n" + "="*70)

# %%
# %%
# =============================================================================
# CELL 8.5 b (adapted): kSZ²–21cm² — Seed-Averaged Plots
# =============================================================================

print("\n" + "="*70)
print("VISUALIZING kSZ²–21cm² CROSS-CORRELATION (SEED-AVERAGED)")
print("="*70)

plot_dir_final = os.path.join(plot_dir, "plot_final_cell")
os.makedirs(plot_dir_final, exist_ok=True)
plot_dir_save = plot_dir_final

# Load if necessary
if ('cross_corr_results_sq_all' not in dir() and 
    'cross_corr_results_sq_all' not in globals()) or len(cross_corr_results_sq_all) == 0:
    print("Loading cross_corr_results_sq_all from cache...")
    cross_corr_results_sq_all = {}
    for seed in RANDOM_SEEDS:
        cc_cache = os.path.join(main_cache_dir, f"seed_{seed}", f"cross_corr_sq_seed{seed}.npy")
        if os.path.exists(cc_cache):
            cross_corr_results_sq_all[seed] = np.load(cc_cache, allow_pickle=True).item()
    print(f"Loaded {len(cross_corr_results_sq_all)} seeds")

if len(cross_corr_results_sq_all) > 0:
    ref_seed = next(iter(cross_corr_results_sq_all))
    ref_lc = lightcones[ref_seed]
    all_z0 = sorted(cross_corr_results_sq_all[ref_seed].keys())

    z_nodes_sorted = ref_lc.node_redshifts[::-1]
    x_e_nodes = 1.0 - ref_lc.global_xH[::-1]

    # ======================================================================
    # PLOT 1: D_ℓ vs z at fixed ℓ (seed-averaged)
    # ======================================================================
    ell_targets = [500, 1000, 3000]
    colors_ell = ['darkblue', 'darkgreen', 'darkred']

    def _draw_avg_z(ax):
        for i, ell_target in enumerate(ell_targets):
            z_plot, D_plot, E_plot = [], [], []
            for z0 in all_z0:
                D_vals, D_err_sq = [], []
                for seed, ccr in cross_corr_results_sq_all.items():
                    if z0 not in ccr:
                        continue
                    res = ccr[z0]
                    ell = res['ell']
                    D_ell = res['D_cross']
                    D_err = res['C_cross_err'] * ell * (ell + 1) / (2 * np.pi)
                    idx = np.argmin(np.abs(ell - ell_target))
                    if np.isfinite(D_ell[idx]):
                        D_vals.append(D_ell[idx])
                        D_err_sq.append(D_err[idx]**2)
                if len(D_vals) >= 2:
                    D_mean = np.mean(D_vals)
                    sigma_seeds = np.std(D_vals, ddof=1)
                    sigma_meas = np.sqrt(np.mean(D_err_sq))
                    sigma_total = np.sqrt(sigma_seeds**2 + sigma_meas**2)

                    z_plot.append(z0)
                    D_plot.append(D_mean)
                    E_plot.append(sigma_total)

            if len(z_plot) > 0:
                z_arr = np.array(z_plot)
                ax.errorbar(z_arr, D_plot, yerr=E_plot,
                            color=colors_ell[i], lw=2.5, alpha=0.85,
                            marker='o', markersize=6, capsize=4,
                            label=rf'$\ell={ell_target}$')
                ax.fill_between(z_arr,
                                np.array(D_plot) - np.array(E_plot),
                                np.array(D_plot) + np.array(E_plot),
                                color=colors_ell[i], alpha=0.2)

        # Ionization fraction markers
        for xe_val in [0.2, 0.5, 0.9]:
            z_xe = np.interp(xe_val, x_e_nodes[::-1], z_nodes_sorted[::-1])
            ax.axvline(z_xe, color='gray', ls=':', lw=1, alpha=0.6)
            ax.text(z_xe, ax.get_ylim()[1]*0.92, rf'$x_e={xe_val:.1f}$',
                    rotation=90, ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        ax.axhline(0, color='k', ls='--', lw=1, alpha=0.5)
        ax.set_xlabel(r'Chunk centre redshift $z_0$')
        ax.set_ylabel(r'$\ell(\ell+1)C_\ell^{\rm kSZ^2 \times 21cm^2}/2\pi$')
        ax.legend(loc='best', framealpha=0.9)
        ax.invert_xaxis()

        ax.text(0.05, 0.95,
                f'{N_SEEDS} seeds | $\\sigma_{{\\rm seeds}} \\oplus \\sigma_{{\\rm meas}}$',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    save_pdf_png(
        _draw_avg_z, plot_dir_save,
        "kSZ2_21cm2_cross_Dl_vs_z_fixed_ell",
        title=r'kSZ$^2$–21cm$^2$ Redshift Evolution at Fixed $\ell$ (Seed-Averaged)',
        figsize=(11, 8)
    )
    print("✓ Saved: kSZ2_21cm2_cross_Dl_vs_z_fixed_ell")

    # ======================================================================
    # PLOT 2: D_ℓ vs ℓ at selected x_e (seed-averaged)
    # ======================================================================
    target_xe = [0.2, 0.5, 0.9]
    selected_z = [z_nodes_sorted[np.argmin(np.abs(x_e_nodes - xe))] for xe in target_xe]
    selected_xe = [x_e_nodes[np.argmin(np.abs(x_e_nodes - xe))] for xe in target_xe]
    colors_selected = ['blue', 'green', 'red']

    def _draw_avg_ell(ax):
        for i, (z_target, xe) in enumerate(zip(selected_z, selected_xe)):
            D_ell_seeds = []
            D_err_sq_list = []
            ell_ref = None

            for seed, ccr in cross_corr_results_sq_all.items():
                z_closest = min(ccr.keys(), key=lambda z: abs(z - z_target))
                if abs(z_closest - z_target) > 0.75:
                    continue
                res = ccr[z_closest]
                ell = res['ell']
                D_ell = res['D_cross']
                D_err = res['C_cross_err'] * ell * (ell + 1) / (2 * np.pi)
                if ell_ref is None:
                    ell_ref = ell
                D_ell_seeds.append(D_ell)
                D_err_sq_list.append(D_err**2)

            if len(D_ell_seeds) == 0:
                continue

            D_matrix = np.array(D_ell_seeds)
            D_mean = np.nanmean(D_matrix, axis=0)
            sigma_seeds = np.nanstd(D_matrix, ddof=1, axis=0)
            sigma_meas = np.sqrt(np.nanmean(np.array(D_err_sq_list), axis=0))
            sigma_total = np.sqrt(sigma_seeds**2 + sigma_meas**2)

            valid = np.isfinite(D_mean) & (ell_ref > 10)
            ax.errorbar(ell_ref[valid], D_mean[valid], yerr=sigma_total[valid],
                        color=colors_selected[i], lw=2.5, alpha=0.8,
                        marker='o', markersize=5, capsize=3,
                        label=rf'$z_0\approx{z_target:.1f}$ ($x_e={xe:.2f}$)',
                        errorevery=3)
            ax.fill_between(ell_ref[valid],
                            D_mean[valid] - sigma_total[valid],
                            D_mean[valid] + sigma_total[valid],
                            color=colors_selected[i], alpha=0.15)

        ax.axhline(0, color='k', ls='--', lw=1, alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel(r'Multipole $\ell$')
        ax.set_ylabel(r'$\ell(\ell+1)C_\ell^{\rm kSZ^2 \times 21cm^2}/2\pi$')
        ax.legend(loc='best', framealpha=0.9)

        ax.text(0.02, 0.02,
                f'{N_SEEDS} seeds | sigma_seeds + sigma_meas',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    save_pdf_png(
        _draw_avg_ell, plot_dir_save,
        "kSZ2_21cm2_cross_Dl_selected_xe",
        title=r'kSZ$^2$–21cm$^2$ at Key Ionization Fractions (Seed-Averaged)',
    )
    print("✓ Saved: kSZ2_21cm2_cross_Dl_selected_xe")

    print("\n✓ CELL 8b (kSZ²–21cm² averaged) COMPLETE")

else:
    print("\n✗ Skipping — no cross_corr_results_sq_all available")

print("\n" + "="*70)

# %%
# =============================================================================
# CELL 8c: Redshift Evolution of 21cm Auto Power Spectrum (All Seeds)
# Consistent with Cell 8b: mean ± std across seeds (cosmic variance)
# =============================================================================

print("\n" + "="*70)
print("VISUALIZING 21cm AUTO POWER SPECTRUM REDSHIFT EVOLUTION")
print("="*70)

# Create subdirectory if needed
plot_dir_final = f"{plot_dir}/plot_final_cell"
if not os.path.exists(plot_dir_final):
    os.makedirs(plot_dir_final)
plot_dir_save = plot_dir_final

# ==========================================================================
# Load cross_corr_results_all from cache if not in memory
# ==========================================================================
if 'cross_corr_results_all' not in dir() or len(cross_corr_results_all) == 0:
    print("cross_corr_results_all not in memory → loading from Cell 7 cache")
    cross_corr_results_all = {}
    for seed in RANDOM_SEEDS:
        cc_cache = f"{main_cache_dir}/seed_{seed}/cross_corr_seed{seed}.npy"
        if os.path.exists(cc_cache):
            cross_corr_results_all[seed] = np.load(
                cc_cache, allow_pickle=True).item()
            print(f"  ✓ Loaded seed {seed} "
                  f"({len(cross_corr_results_all[seed])} redshifts)")
        else:
            print(f"  ✗ No cache found for seed {seed}")
    print(f"  Loaded {len(cross_corr_results_all)}/{N_SEEDS} seeds")
else:
    print(f"cross_corr_results_all already in memory "
          f"({len(cross_corr_results_all)} seeds)")

if len(cross_corr_results_all) > 0:

    from astropy.cosmology import FlatLambdaCDM
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    cosmo = FlatLambdaCDM(H0=67.77, Om0=0.3086)

    # ==========================================================================
    # Convert P_T21(k) → D_ℓ^{21} for all seeds
    # ==========================================================================

    print(f"\n=== CONVERTING 21cm AUTO POWER TO ℓ-SPACE (ALL SEEDS) ===")

    auto_T21_ell_all = {}

    for seed, ccr in cross_corr_results_all.items():
        auto_T21_ell_results = {}

        for z_obs in sorted(ccr.keys()):
            results      = ccr[z_obs]
            D_A_Mpc      = float(cosmo.angular_diameter_distance(z_obs).value)
            chi_comoving = float(cosmo.comoving_distance(z_obs).value)

            k_centers  = results['k_centers']
            P_T21      = results['P_T21_1d']
            n_modes    = results['n_modes']

            ell_from_k = k_centers * chi_comoving / 0.67
            C_T21_ell  = P_T21 * 0.67**2 / D_A_Mpc**2
            D_T21_ell  = ell_from_k * (ell_from_k + 1) * C_T21_ell / (2 * np.pi)

            with np.errstate(divide='ignore', invalid='ignore'):
                err_frac = np.where(n_modes > 0, 1.0 / np.sqrt(n_modes), np.nan)

            D_T21_ell_err = np.abs(D_T21_ell) * err_frac

            auto_T21_ell_results[z_obs] = {
                'ell_from_k'   : ell_from_k,
                'D_T21_ell'    : D_T21_ell,
                'D_T21_ell_err': D_T21_ell_err,
            }

        auto_T21_ell_all[seed] = auto_T21_ell_results

    print(f"Converted {len(auto_T21_ell_all)} seeds to ℓ-space")

    # ==========================================================================
    # Average across seeds at each redshift (mean ± std)
    # ==========================================================================

    ref_seed = list(auto_T21_ell_all.keys())[0]
    all_z    = sorted(auto_T21_ell_all[ref_seed].keys())

    auto_T21_ell_averaged = {}

    for z_obs in all_z:
        D_seeds = []
        ell_ref = None

        for seed, res_dict in auto_T21_ell_all.items():
            if z_obs not in res_dict:
                continue

            res   = res_dict[z_obs]
            ell   = res['ell_from_k']
            D_ell = res['D_T21_ell']

            if ell_ref is None:
                ell_ref = ell

            D_seeds.append(D_ell)

        if len(D_seeds) == 0:
            continue

        D_matrix = np.array(D_seeds)

        D_mean = np.nanmean(D_matrix, axis=0)
        D_std  = np.nanstd(D_matrix, ddof=1, axis=0)

        auto_T21_ell_averaged[z_obs] = {
            'ell_from_k'   : ell_ref,
            'D_T21_ell'    : D_mean,
            'D_T21_ell_err': D_std,
        }

    print(f"Averaged over {len(auto_T21_ell_all)} seeds "
          f"at {len(auto_T21_ell_averaged)} redshifts")

    # ==========================================================================
    # Plot: D_ℓ^{21} vs ℓ, colored by redshift
    # ==========================================================================

    print(f"\n=== GENERATING 21cm AUTO POWER SPECTRUM PLOT ===")

    z_all = np.array(sorted(auto_T21_ell_averaged.keys()))
    norm  = mcolors.Normalize(vmin=z_all.min(), vmax=z_all.max())
    cmap  = cm.plasma

    fig, ax = plt.subplots(1, 1, figsize=(10, 7), constrained_layout=True)

    for z_obs in z_all:
        res   = auto_T21_ell_averaged[z_obs]
        ell   = res['ell_from_k']
        D_ell = res['D_T21_ell']
        D_err = res['D_T21_ell_err']

        valid = (~np.isnan(D_ell) & np.isfinite(D_ell)
                 & (ell > 10) & (D_ell > 0))

        if np.sum(valid) < 3:
            continue

        color = cmap(norm(z_obs))

        ax.plot(ell[valid], D_ell[valid],
                color=color, lw=1.5, alpha=0.75)

        ax.fill_between(ell[valid],
                        D_ell[valid] - D_err[valid],
                        D_ell[valid] + D_err[valid],
                        color=color, alpha=0.12)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(r'Redshift $z$')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Multipole $\ell$')
    ax.set_ylabel(r'$D_\ell^{21}\ [\mathrm{mK}^2]$')

    ax.text(0.05, 0.05,
            f'Shaded: seed-to-seed scatter\n'
            f'{N_SEEDS} seeds averaged\n'
            r'$D_\ell = \ell(\ell+1)C_\ell/2\pi$',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plot_name = "21cm_auto_Dl_vs_ell_redshift_evolution"

    fig.savefig(f"{plot_dir_save}/{plot_name}.pdf", bbox_inches='tight')

    ax.set_title(r'21cm Auto Power Spectrum $D_\ell^{21}$ — Redshift Evolution'
                 f' ({N_SEEDS} seeds)',
                 fontweight='bold')

    fig.savefig(f"{plot_dir_save}/{plot_name}.png",
                dpi=300, bbox_inches='tight')

    print(f"✓ Saved: {plot_name}")
    plt.close(fig)

    print("\n✓ 21cm AUTO POWER SPECTRUM PLOT COMPLETE")
    print(f"  Saved to: {plot_dir_save}")

else:
    print("\n✗ Skipping — no cross_corr_results_all available")

print("\n" + "="*70)

# %%
# %%
# =============================================================================
# CELL 9: WHY kSZ²×21cm (UNSQUARED) DIES UNDER WEDGE FILTERING
#
# YOUR EXISTING RESULT (cross_corr_results_all, Cell 7):
#   kSZ² × 21cm(raw, no filter) → reproduces Ma et al. (2018) Fig 3 bottom
#   panel perfectly. Sign changes track reionisation. This is the PHYSICAL
#   signal, but only accessible because the raw 21cm field still has its
#   k∥ ≈ 0 modes.
#
# WHAT THIS CELL DEMONSTRATES:
#   Apply a wedge / high-pass filter to the 21cm field BEFORE cross-correlating
#   with kSZ². Because kSZ²×21cm (unsquared) requires k∥ ≈ 0 from the 21cm
#   field (the triangle condition forces this), any filter that kills those
#   modes destroys the signal entirely — it collapses to noise.
#
# THE PHYSICS IN ONE LINE:
#   kSZ²(ℓ) × T21cm(ℓ') needs k∥,21cm ≈ 0 to conserve momentum.
#   The wedge removes exactly those modes. Signal → 0.
#   Contrast with kSZ²×21cm²: squaring manufactures k∥=0 from ±k∥' pairs
#   that survive the filter. That's why squaring saves you.
#
# PIPELINE (run on ONE seed, ONE ℓ, across all z_obs in the lightcone):
#   For each z_obs slice:
#     1. Project raw 3D T21 → 2D (mean along LoS) — same as Cell 7 worker
#     2. Apply wedge filter IN 2D Fourier space to mimic what happens to
#        the available k∥≈0 modes
#     3. Cross-correlate with kSZ²
#     4. Compare to your existing no-filter result
#
# FILTER SCENARIOS:
#   (A) No filter     — your existing Cell 7 result (reproduced here for check)
#   (B) k∥ > 0.01     — optimistic: removes only very large-scale radial modes
#   (C) Wedge m=3     — realistic EoR horizon wedge
#   (D) Wedge m=5     — pessimistic
#
# NOTE ON IMPLEMENTATION:
#   The Cell 7 worker projects first (collapses LoS → 2D), losing all k∥ info.
#   So to mimic wedge filtering of the raw 21cm before projection, we work
#   directly with the 3D lightcone chunk here. We filter in 3D, then project,
#   then cross-correlate. This is the honest comparison.
#
# OUTPUTS:
#   wedge_kills_unsquared_Dl_vs_z.pdf/png   ← main teaching plot
#   wedge_kills_unsquared_Dl_vs_ell.pdf/png ← at selected xe
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.cosmology import FlatLambdaCDM

print("\n" + "="*70)
print("CELL 9: WEDGE FILTER DESTROYS kSZ²×21cm (UNSQUARED)")
print("="*70)

# =============================================================================
# GUARD
# =============================================================================
_ok = (
    'lightcones'  in dir() and len(lightcones)  > 0 and
    'kSZ_maps'    in dir() and len(kSZ_maps)    > 0
)
if not _ok:
    print("✗  Need lightcones and kSZ_maps in memory. Re-run Cells 2, 5, 6.")
else:

    # =========================================================================
    # Setup — pick one seed, use the full redshift axis of the lightcone
    # =========================================================================
    diag_seed = list(lightcones.keys())[0]
    lc        = lightcones[diag_seed]
    kSZ_map   = kSZ_maps[diag_seed]

    print(f"  Seed       : {diag_seed}")
    print(f"  kSZ map    : {kSZ_map.shape}  "
          f"RMS={np.sqrt(np.mean(kSZ_map**2)):.3e}")

    # =========================================================================
    # Map / grid geometry
    # =========================================================================
    npix_side    = user_params.HII_DIM
    box_size_Mpc = float(user_params.BOX_LEN)
    pix_size_Mpc = box_size_Mpc / npix_side
    pix_area     = pix_size_Mpc**2

    dk       = 2 * np.pi / (npix_side * pix_size_Mpc)
    kx_2d    = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
    ky_2d    = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
    kgrid_2d = np.sqrt(kx_2d[:, None]**2 + ky_2d[None, :]**2)
    k_bins   = np.logspace(np.log10(dk), np.log10(kgrid_2d.max() * 0.9), 30)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    kx_1d = np.fft.fftfreq(npix_side, d=pix_size_Mpc) * 2 * np.pi
    ky_1d = np.fft.fftfreq(npix_side, d=pix_size_Mpc) * 2 * np.pi

    cosmo = FlatLambdaCDM(H0=67.77, Om0=0.3086)

    # =========================================================================
    # Squared kSZ map — computed once, shared across all scenarios
    # =========================================================================
    kSZ2_map         = kSZ_map**2
    kSZ2_centered    = kSZ2_map - np.mean(kSZ2_map)
    fft_kSZ2_shifted = np.fft.fftshift(np.fft.fft2(kSZ2_centered))

    # =========================================================================
    # Redshift slices to scan
    # Use the lightcone's native redshift axis — same granularity as Cell 7
    # =========================================================================
    lc_redshifts = np.asarray(lc.lightcone_redshifts, dtype=np.float64)

    # Use THIN slices (Δz = 0.5) so redshift resolution is fine
    delta_z_thin = 0.5
    z_min_scan   = lc_redshifts[lc_redshifts > 0].min()
    z_max_scan   = min(lc_redshifts.max(), 20.0)
    z_centres    = np.arange(
        np.ceil(z_min_scan / delta_z_thin) * delta_z_thin,
        z_max_scan,
        delta_z_thin
    )
    print(f"  Redshift scan: {z_centres[0]:.1f} → {z_centres[-1]:.1f} "
          f"  Δz={delta_z_thin}  ({len(z_centres)} chunks)")

    # =========================================================================
    # Filter scenarios
    # =========================================================================
    scenarios = {
        'A_nofilter'  : {'label': '(A) No filter (Cell 7 result)',
                         'color': 'black',   'ls': '-',  'lw': 2.5},
        'B_optimistic': {'label': r'(B) $k_\parallel > 0.01$ Mpc$^{-1}$',
                         'color': 'steelblue', 'ls': '--', 'lw': 2.0},
        'C_wedge_m3'  : {'label': r'(C) Wedge $m=3$',
                         'color': 'darkorange', 'ls': '-.', 'lw': 2.0},
        'D_wedge_m5'  : {'label': r'(D) Wedge $m=5$',
                         'color': 'crimson',    'ls': ':',  'lw': 2.0},
    }

    # =========================================================================
    # Run: for each scenario, scan all z_centres
    # Store D_ℓ(z) at a fixed target ℓ and the full ℓ-spectrum at xe≈0.5
    # =========================================================================
    ell_target = 3000   # matches your existing plot and Ma+2018 Fig 3

    # We will also store the full ℓ spectrum at three xe values
    # Get the ionisation history from this seed
    z_nodes_lc  = lc.node_redshifts[::-1]          # ascending z
    xe_nodes_lc = 1.0 - lc.global_xH[::-1]

    def z_at_xe(xe_val):
        return float(np.interp(xe_val, xe_nodes_lc, z_nodes_lc))

    z_xe02 = z_at_xe(0.2)
    z_xe05 = z_at_xe(0.5)
    z_xe09 = z_at_xe(0.9)
    print(f"\n  z(xe=0.2) = {z_xe02:.2f}")
    print(f"  z(xe=0.5) = {z_xe05:.2f}")
    print(f"  z(xe=0.9) = {z_xe09:.2f}")

    results_z  = {k: {'z': [], 'D': []} for k in scenarios}
    results_xe = {k: {0.2: None, 0.5: None, 0.9: None} for k in scenarios}

    T_CMB_uK = 2.725e6   # for unit conversion if needed

    for key, meta in scenarios.items():
        print(f"\n  ── Scenario {meta['label']} ──")

        for z0 in z_centres:
            z_lo = z0 - delta_z_thin / 2.0
            z_hi = z0 + delta_z_thin / 2.0

            idx_chunk = np.where(
                (lc_redshifts >= z_lo) & (lc_redshifts < z_hi)
            )[0]
            if len(idx_chunk) < 2:
                continue

            # ---- extract 3D chunk ----------------------------------------
            T21_chunk = np.asarray(
                lc.brightness_temp[:, :, idx_chunk], dtype=np.float64
            )
            n_los        = T21_chunk.shape[2]
            pix_size_los = pix_size_Mpc

            # ---- build 3D k-grids for this chunk --------------------------
            kz_1d    = np.fft.fftfreq(n_los, d=pix_size_los) * 2 * np.pi
            kx_3d    = kx_1d[:, None, None]
            ky_3d    = ky_1d[None, :, None]
            kz_3d    = kz_1d[None, None, :]
            kperp_3d = np.sqrt(kx_3d**2 + ky_3d**2)
            kpar_3d  = np.abs(kz_3d)

            # ---- apply filter in 3D Fourier space -------------------------
            T21_fft3d = np.fft.fftn(T21_chunk)

            if key == 'A_nofilter':
                filt3d = np.ones_like(kpar_3d)
            elif key == 'B_optimistic':
                filt3d = (kpar_3d > 0.01).astype(float)
            elif key == 'C_wedge_m3':
                filt3d = (kpar_3d > 3.0 * kperp_3d).astype(float)
            elif key == 'D_wedge_m5':
                filt3d = (kpar_3d > 5.0 * kperp_3d).astype(float)

            T21_filtered = np.real(np.fft.ifftn(T21_fft3d * filt3d))

            # ---- project to 2D (mean along LoS) ---------------------------
            # NOTE: NO squaring — this is kSZ²×21cm not kSZ²×21cm²
            T21_2d      = np.mean(T21_filtered, axis=2)
            T21_cen     = T21_2d - np.mean(T21_2d)
            fft_T21_sh  = np.fft.fftshift(np.fft.fft2(T21_cen))

            # ---- cross-power with kSZ² ------------------------------------
            cross_ps2d = (np.real(np.conj(fft_kSZ2_shifted) * fft_T21_sh)
                          * pix_area / npix_side**2)

            # ---- bin into k annuli ----------------------------------------
            C_cross   = np.full(len(k_centers), np.nan)
            for j in range(len(k_centers)):
                mask  = (kgrid_2d >= k_bins[j]) & (kgrid_2d < k_bins[j+1])
                n_pix = int(np.sum(mask))
                if n_pix > 0:
                    C_cross[j] = np.mean(cross_ps2d[mask])

            chi_z0  = float(cosmo.comoving_distance(z0).value)
            ell_arr = k_centers * chi_z0
            D_cross = ell_arr * (ell_arr + 1) * C_cross / (2 * np.pi)

            # ---- store D at target ℓ --------------------------------------
            idx_ell = np.argmin(np.abs(ell_arr - ell_target))
            if np.isfinite(D_cross[idx_ell]):
                results_z[key]['z'].append(z0)
                results_z[key]['D'].append(D_cross[idx_ell])

            # ---- store full ℓ spectrum near selected xe -------------------
            for xe_val, z_xe in [(0.2, z_xe02), (0.5, z_xe05), (0.9, z_xe09)]:
                if (abs(z0 - z_xe) < delta_z_thin
                        and results_xe[key][xe_val] is None):
                    results_xe[key][xe_val] = {
                        'ell'    : ell_arr,
                        'D_cross': D_cross,
                        'z0'     : z0,
                    }

        n_pts = len(results_z[key]['z'])
        D_arr = np.array(results_z[key]['D'])
        print(f"     {n_pts} z-points  |  "
              f"max|D| = {np.nanmax(np.abs(D_arr)):.3e}  "
              f"(at ℓ={ell_target})")

    # =========================================================================
    # PLOT 1: D_ℓ vs z at ℓ=3000 — the main teaching plot
    #
    # LEFT PANEL  : your existing result (no filter) with the physics sign
    #               changes clearly visible
    # RIGHT PANEL : all four scenarios overlaid — wedge cases collapse to ~0
    #
    # This is directly analogous to Ma+2018 Fig 3 bottom panel, but showing
    # what happens when you remove the k∥≈0 modes the statistic needs.
    # =========================================================================
    print("\n  Plotting D_ℓ vs z (main teaching plot)...")

    with mpl.rc_context(PDF_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(20, 7),
                                 constrained_layout=True, sharey=False)

        # --- left panel: no-filter only (clean reproduction of your result) ---
        ax = axes[0]
        z_A = np.array(results_z['A_nofilter']['z'])
        D_A = np.array(results_z['A_nofilter']['D'])
        sort = np.argsort(z_A)[::-1]
        ax.plot(z_A[sort], D_A[sort],
                color='black', lw=2.5, ls='-', marker='o', markersize=4,
                label='No filter (raw 21cm)')
        ax.axhline(0, color='gray', ls='--', lw=1)

        # mark xe transitions
        for xe_val, z_xe, color in [(0.2, z_xe02, 'blue'),
                                     (0.5, z_xe05, 'green'),
                                     (0.9, z_xe09, 'red')]:
            ax.axvline(z_xe, color=color, ls=':', lw=1.5, alpha=0.7)
            ax.text(z_xe + 0.1, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -1,
                    rf'$x_e={xe_val}$', color=color, fontsize=13,
                    rotation=90, va='bottom')

        ax.set_yscale('symlog', linthresh=1e-2)
        ax.set_xlabel(r'Redshift $z$')
        ax.set_ylabel(
            rf'$\ell(\ell+1)C_\ell^{{\rm kSZ^2\times 21cm}}/2\pi$'
            rf'  $[\mu\mathrm{{K}}^2\,\mathrm{{mK}}]$  at $\ell={ell_target}$')
        ax.set_title('No filter — Ma+2018 signal reproduced', fontsize=16)
        ax.invert_xaxis()
        ax.text(0.05, 0.97,
                'Signal exists because\n'
                r'raw 21cm has $k_\parallel\approx 0$ modes',
                transform=ax.transAxes, va='top', fontsize=13,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        # --- right panel: all four scenarios ---
        ax = axes[1]
        for key, meta in scenarios.items():
            z_arr = np.array(results_z[key]['z'])
            D_arr = np.array(results_z[key]['D'])
            if len(z_arr) == 0:
                continue
            sort = np.argsort(z_arr)[::-1]
            ax.plot(z_arr[sort], D_arr[sort],
                    color=meta['color'], lw=meta['lw'], ls=meta['ls'],
                    marker='o' if key == 'A_nofilter' else None,
                    markersize=3,
                    label=meta['label'], alpha=0.9)

        ax.axhline(0, color='gray', ls='--', lw=1)
        for xe_val, z_xe, color in [(0.2, z_xe02, 'blue'),
                                     (0.5, z_xe05, 'green'),
                                     (0.9, z_xe09, 'red')]:
            ax.axvline(z_xe, color=color, ls=':', lw=1.5, alpha=0.7)

        ax.set_yscale('symlog', linthresh=1e-2)
        ax.set_xlabel(r'Redshift $z$')
        ax.set_ylabel(
            rf'$\ell(\ell+1)C_\ell^{{\rm kSZ^2\times 21cm}}/2\pi$'
            rf'  at $\ell={ell_target}$')
        ax.set_title('All filter scenarios — wedge kills the signal', fontsize=16)
        ax.legend(loc='lower left', fontsize=14)
        ax.invert_xaxis()
        ax.text(0.05, 0.97,
                'Wedge removes $k_\\parallel\\approx 0$ modes.\n'
                r'Triangle condition: $k_{\parallel,21cm}\approx 0$ required.'
                '\nSignal collapses $\\rightarrow$ noise.',
                transform=ax.transAxes, va='top', fontsize=13,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

        fig.suptitle(
            rf'kSZ$^2\times$21cm (unsquared): $\ell={ell_target}$, '
            f'seed {diag_seed}',
            fontsize=18)
        fig.savefig(f"{plot_dir}/wedge_kills_unsquared_Dl_vs_z.pdf",
                    bbox_inches='tight')
        plt.close(fig)

    with mpl.rc_context(PNG_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(20, 7),
                                 constrained_layout=True, sharey=False)

        ax = axes[0]
        z_A   = np.array(results_z['A_nofilter']['z'])
        D_A   = np.array(results_z['A_nofilter']['D'])
        sort  = np.argsort(z_A)[::-1]
        ax.plot(z_A[sort], D_A[sort],
                color='black', lw=2.0, ls='-', marker='o', markersize=3,
                label='No filter')
        ax.axhline(0, color='gray', ls='--', lw=1)
        for xe_val, z_xe, color in [(0.2, z_xe02, 'blue'),
                                     (0.5, z_xe05, 'green'),
                                     (0.9, z_xe09, 'red')]:
            ax.axvline(z_xe, color=color, ls=':', lw=1.2, alpha=0.7)
            ax.text(z_xe + 0.1, 0, rf'$x_e={xe_val}$',
                    color=color, fontsize=11, rotation=90, va='bottom')
        ax.set_yscale('symlog', linthresh=1e-2)
        ax.set_xlabel(r'Redshift $z$')
        ax.set_ylabel(rf'$D_\ell^{{\rm kSZ^2\times 21cm}}$ at $\ell={ell_target}$')
        ax.set_title('No filter — raw signal', fontweight='bold')
        ax.invert_xaxis()

        ax = axes[1]
        for key, meta in scenarios.items():
            z_arr = np.array(results_z[key]['z'])
            D_arr = np.array(results_z[key]['D'])
            if len(z_arr) == 0:
                continue
            sort = np.argsort(z_arr)[::-1]
            ax.plot(z_arr[sort], D_arr[sort],
                    color=meta['color'], lw=meta['lw'], ls=meta['ls'],
                    label=meta['label'], alpha=0.9)
        ax.axhline(0, color='gray', ls='--', lw=1)
        ax.set_yscale('symlog', linthresh=1e-2)
        ax.set_xlabel(r'Redshift $z$')
        ax.set_ylabel(rf'$D_\ell^{{\rm kSZ^2\times 21cm}}$ at $\ell={ell_target}$')
        ax.set_title('Wedge filter destroys the signal', fontweight='bold')
        ax.legend(loc='lower left', fontsize=12)
        ax.invert_xaxis()

        fig.suptitle(
            rf'kSZ$^2\times$21cm (no squaring): effect of wedge filter',
            fontsize=14, fontweight='bold')
        fig.savefig(f"{plot_dir}/wedge_kills_unsquared_Dl_vs_z.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    print("  ✓ Saved: wedge_kills_unsquared_Dl_vs_z")

    # =========================================================================
    # PLOT 2: D_ℓ vs ℓ at xe=0.2, 0.5, 0.9 — three subpanels
    #
    # At each ionisation fraction, show how the ℓ-spectrum is destroyed
    # as the wedge slope increases. Directly comparable to Ma+2018 Fig 2 bottom.
    # =========================================================================
    print("  Plotting D_ℓ vs ℓ at selected xe ...")

    xe_vals  = [0.2, 0.5, 0.9]
    xe_titles = [r'$x_e = 0.2$ (early reion.)',
                 r'$x_e = 0.5$ (midpoint)',
                 r'$x_e = 0.9$ (late reion.)']

    with mpl.rc_context(PDF_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(21, 7),
                                 constrained_layout=True)
        for ax, xe_val, xe_title in zip(axes, xe_vals, xe_titles):
            for key, meta in scenarios.items():
                dat = results_xe[key][xe_val]
                if dat is None:
                    continue
                ell  = dat['ell']
                D    = dat['D_cross']
                valid = np.isfinite(D) & (ell > 50)
                if np.sum(valid) < 3:
                    continue
                ax.plot(ell[valid], D[valid],
                        color=meta['color'], lw=meta['lw'], ls=meta['ls'],
                        label=meta['label'], alpha=0.9)
            ax.axhline(0, color='gray', ls='--', lw=1)
            ax.set_xscale('log')
            ax.set_xlabel(r'Multipole $\ell$')
            ax.set_ylabel(
                r'$\ell(\ell+1)C_\ell^{\rm kSZ^2\times 21cm}/2\pi$')
            ax.set_title(xe_title, fontsize=16)
            ax.legend(loc='best', fontsize=12)

        fig.suptitle(
            r'kSZ$^2\times$21cm (unsquared): wedge filter at each $x_e$'
            f'\nseed {diag_seed}',
            fontsize=18)
        fig.savefig(f"{plot_dir}/wedge_kills_unsquared_Dl_vs_ell.pdf",
                    bbox_inches='tight')
        plt.close(fig)

    with mpl.rc_context(PNG_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(21, 7),
                                 constrained_layout=True)
        for ax, xe_val, xe_title in zip(axes, xe_vals, xe_titles):
            for key, meta in scenarios.items():
                dat = results_xe[key][xe_val]
                if dat is None:
                    continue
                ell  = dat['ell']
                D    = dat['D_cross']
                valid = np.isfinite(D) & (ell > 50)
                if np.sum(valid) < 3:
                    continue
                ax.plot(ell[valid], D[valid],
                        color=meta['color'], lw=meta['lw'], ls=meta['ls'],
                        label=meta['label'], alpha=0.9)
            ax.axhline(0, color='gray', ls='--', lw=1)
            ax.set_xscale('log')
            ax.set_xlabel(r'Multipole $\ell$')
            ax.set_ylabel(r'$D_\ell^{\rm kSZ^2\times 21cm}$')
            ax.set_title(xe_title, fontweight='bold')
            ax.legend(loc='best', fontsize=11)
        fig.suptitle(
            r'kSZ$^2\times$21cm (unsquared): wedge effect at key $x_e$',
            fontsize=14, fontweight='bold')
        fig.savefig(f"{plot_dir}/wedge_kills_unsquared_Dl_vs_ell.png",
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

    print("  ✓ Saved: wedge_kills_unsquared_Dl_vs_ell")

    # =========================================================================
    # Console summary — quantify the suppression
    # =========================================================================
    D_ref = np.array(results_z['A_nofilter']['D'])
    ref_rms = np.sqrt(np.nanmean(D_ref**2)) if len(D_ref) > 0 else np.nan

    print("\n" + "="*60)
    print(f"SUPPRESSION SUMMARY  (RMS of D_ℓ at ℓ={ell_target} vs z)")
    print(f"{'Scenario':<34} {'RMS D_ℓ':>12} {'Suppression':>14}")
    print("-"*60)
    for key, meta in scenarios.items():
        D_arr = np.array(results_z[key]['D'])
        rms   = np.sqrt(np.nanmean(D_arr**2)) if len(D_arr) > 0 else np.nan
        supp  = rms / ref_rms if ref_rms > 0 else np.nan
        print(f"  {meta['label']:<32} {rms:>12.3e} {supp:>13.1%}")
    print("="*60)

    print("\n  KEY MESSAGE FOR STUDENTS:")
    print("  • No filter: signal has clear sign changes tracking reionisation")
    print("    because k∥≈0 modes of 21cm are still present.")
    print("  • Wedge m=3 or 5: signal collapses toward noise because those")
    print("    k∥≈0 modes — the ONLY ones that contribute via the triangle")
    print("    condition — are removed.")
    print("  • Solution (Zhou+25): SQUARE the 21cm field first. Squaring")
    print("    manufactures K∥=0 from ±k∥' pairs. Wedge no longer fatal.")

    print("\n✓ CELL 9 COMPLETE")
    print("="*70)

# =============================================================================
# CELL SNR (CORRECTED): Signal-to-Noise Ratio Forecast
#           for BOTH kSZ²×21cm  AND  kSZ²×21cm²
#
# The pipeline stores C_ℓ in internal FFT units that are inconsistent
# between the kSZ² field (dimensionless ~10⁻²⁵) and the 21cm field (mK²~10²).
# Adding physical instrument noise in µK² sr is therefore meaningless.
#
# CORRECT APPROACH: use the dimensionless correlation coefficient
#
#   r_ℓ = C_cross / sqrt(P_kSZ2 * P_21cm)
#
# which is unit-free and lies in [-1, 1].  The SNR formula becomes:
#
#   (S/N)²_ℓ = f_sky * (2ℓ+1) * Δℓ * r²_ℓ / (1 - r²_ℓ)
#
# This is the exact Gaussian estimator for a cross-correlation coefficient.
# It is equivalent to Zhou+25 Eq.19 in the signal-dominated limit but
# correctly handles the noise-dominated regime via (1 - r²).
#
# To add instrument noise properly, one would add it inside the simulation
# before computing auto-powers (as Zhou+25 do in §4).  That requires
# re-running the cross-correlation with noise-added fields — a future step.
# The current forecast is therefore a theoretical upper bound (no noise).
#
# Outputs:
#   • SNR(z0) per chunk, ℓ-integrated
#   • Cumulative SNR vs ℓ_max at best z0
#   • Correlation coefficient r(ℓ) at best z0
#   • Summary table
# =============================================================================

print("\n" + "="*70)
print("CELL SNR (CORRECTED): kSZ²×21cm AND kSZ²×21cm²")
print("  Method: correlation coefficient r_ℓ = C / sqrt(P_kSZ2 * P_21cm)")
print("="*70)

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=67.77, Om0=0.3086)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Load caches
# ─────────────────────────────────────────────────────────────────────────────
def _load_cache(filename_pattern, seeds, cache_dir):
    d = {}
    for seed in seeds:
        path = os.path.join(cache_dir, f"seed_{seed}",
                            filename_pattern.format(seed=seed))
        if os.path.exists(path):
            try:
                d[seed] = np.load(path, allow_pickle=True).item()
            except Exception as e:
                print(f"  ✗ seed {seed}: {e}")
    return d

if 'cross_corr_results_all' not in dir() or len(cross_corr_results_all) == 0:
    cross_corr_results_all = _load_cache(
        'cross_corr_seed{seed}.npy', RANDOM_SEEDS, main_cache_dir)
    print(f"Loaded kSZ²×21cm  : {len(cross_corr_results_all)} seeds")

if 'cross_corr_results_sq_all' not in dir() or len(cross_corr_results_sq_all) == 0:
    cross_corr_results_sq_all = _load_cache(
        'cross_corr_sq_seed{seed}.npy', RANDOM_SEEDS, main_cache_dir)
    print(f"Loaded kSZ²×21cm² : {len(cross_corr_results_sq_all)} seeds")

assert len(cross_corr_results_all) > 0,    "Run Cell 7  first."
assert len(cross_corr_results_sq_all) > 0, "Run Cell 7b first."

# Reference for xe(z) mapping
ref_seed  = next(iter(cross_corr_results_sq_all))
ref_lc    = lightcones[ref_seed]
z_nodes_s = ref_lc.node_redshifts[::-1]
xe_nodes  = 1.0 - ref_lc.global_xH[::-1]

f_sky = 0.024   # 1000 deg² overlap

# ─────────────────────────────────────────────────────────────────────────────
# 1. Core SNR engine — works for both statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr_r(results_all, z_keys,
                  sig_key, auto_ksz_key, auto_21_key,
                  ell_key='k_centers', chi_func=None,
                  ell_min=100, ell_max=5000):
    """
    Compute SNR using the correlation coefficient r_ℓ.

    (S/N)²_ℓ = f_sky * (2ℓ+1) * Δℓ * r²/(1-r²)

    Parameters
    ----------
    chi_func : callable or None
        If not None, converts k → ℓ via ℓ = k * chi(z) / h (unsquared case).
        If None, uses stored 'ell' key directly (squared case).
    """
    per_z      = {}
    snr_sq_tot = 0.0

    for z0 in z_keys:
        sigs, ksz2s, t21s, ell_ref = [], [], [], None

        for seed, ccr in results_all.items():
            if z0 not in ccr:
                continue
            res  = ccr[z0]
            k    = res[ell_key]
            ell_here = (k * chi_func(z0) / 0.67
                        if chi_func is not None
                        else res.get('ell', k))
            if ell_ref is None:
                ell_ref = ell_here
            sigs.append(res[sig_key])
            ksz2s.append(res[auto_ksz_key])
            t21s.append(res[auto_21_key])

        if ell_ref is None or len(sigs) == 0:
            continue

        C_sig  = np.nanmean(np.array(sigs),  axis=0)
        C_kSZ2 = np.nanmean(np.array(ksz2s), axis=0)
        C_21cm = np.nanmean(np.array(t21s),  axis=0)
        ell    = ell_ref

        # Correlation coefficient
        denom_r = np.sqrt(np.abs(C_kSZ2) * np.abs(C_21cm))
        r       = np.where(denom_r > 0, C_sig / denom_r, 0.0)
        r       = np.clip(r, -1.0 + 1e-10, 1.0 - 1e-10)

        valid = (np.isfinite(r) & np.isfinite(ell)
                 & (ell >= ell_min) & (ell <= ell_max))

        snr_sq_bin = np.zeros_like(ell)
        if np.any(valid):
            dell           = np.gradient(ell)
            r2             = r[valid]**2
            snr_sq_bin[valid] = np.maximum(
                f_sky * (2.0 * ell[valid] + 1.0) * dell[valid]
                * r2 / (1.0 - r2),
                0.0)

        snr_per_bin = np.sqrt(snr_sq_bin)
        snr_cumul   = np.sqrt(np.cumsum(snr_sq_bin))
        snr_z0      = float(snr_cumul[-1]) if len(snr_cumul) else 0.0
        snr_sq_tot += snr_z0**2

        per_z[z0] = dict(ell=ell, r=r,
                         C_signal=C_sig, C_kSZ2=C_kSZ2, C_21cm=C_21cm,
                         snr_per_bin=snr_per_bin,
                         snr_cumul=snr_cumul,
                         snr_total=snr_z0)

    return per_z, np.sqrt(snr_sq_tot)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Redshift key lists
# ─────────────────────────────────────────────────────────────────────────────
ref_seed_un = next(iter(cross_corr_results_all))
z_keys_un   = sorted(cross_corr_results_all[ref_seed_un].keys())

ref_seed_sq = next(iter(cross_corr_results_sq_all))
z_keys_sq   = sorted(cross_corr_results_sq_all[ref_seed_sq].keys())

def chi_over_h(z):
    return float(cosmo.comoving_distance(z).value) / 0.67

# ─────────────────────────────────────────────────────────────────────────────
# 3. Run forecasts
# ─────────────────────────────────────────────────────────────────────────────
combinations = [
    ('SO',    'HERA'),
    ('CMBS4', 'HERA'),
    ('CMBS4', 'SKA'),
    ('CMBHD', 'SKA'),
]

forecasts = {}

print(f"\n{'Combination':<22} {'kSZ²×21cm':>14} {'kSZ²×21cm²':>14}")
print("-" * 52)

for cmb_p, ts_p in combinations:
    key = f"{cmb_p}×{ts_p}"

    per_z_un, tot_un = compute_snr_r(
        cross_corr_results_all, z_keys_un,
        sig_key='C_cross_1d', auto_ksz_key='P_kSZ2_1d',
        auto_21_key='P_T21_1d',
        ell_key='k_centers', chi_func=chi_over_h)

    per_z_sq, tot_sq = compute_snr_r(
        cross_corr_results_sq_all, z_keys_sq,
        sig_key='C_cross', auto_ksz_key='P_kSZ2',
        auto_21_key='P_T21sq',
        ell_key='k_centers', chi_func=None)

    forecasts[key] = {'un': (per_z_un, tot_un),
                      'sq': (per_z_sq, tot_sq)}

    flag = lambda s: "✓" if s >= 5 else ("~" if s >= 1 else "✗")
    print(f"  {key:<20}  {tot_un:>8.2f}σ {flag(tot_un)}   "
          f"{tot_sq:>8.2f}σ {flag(tot_sq)}")

print("-" * 52)
print(f"  f_sky={f_sky:.3f} ({f_sky*41253:.0f} deg²) — no instrument noise added")
print("  (upper bound; noise-free cosmic variance only)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Plots
# ─────────────────────────────────────────────────────────────────────────────
plot_dir_snr = os.path.join(plot_dir, "snr_forecast_corrected")
os.makedirs(plot_dir_snr, exist_ok=True)

combo_colors = ['steelblue', 'darkorange', 'forestgreen', 'darkred']
stat_keys    = ['un', 'sq']
titles       = [r'kSZ²×21cm  (Ma+18)', r'kSZ²×21cm²  (Zhou+25)']

# ── Plot 1: SNR(z0) ──────────────────────────────────────────────────────────
with mpl.rc_context(PNG_STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             constrained_layout=True, sharey=False)
    for ax, stat, title in zip(axes, stat_keys, titles):
        for (cmb_p, ts_p), color in zip(combinations, combo_colors):
            key    = f"{cmb_p}×{ts_p}"
            per_z, tot = forecasts[key][stat]
            if not per_z:
                continue
            z_arr   = sorted(per_z.keys())
            snr_arr = [per_z[z]['snr_total'] for z in z_arr]
            ax.plot(z_arr, snr_arr, 'o-', color=color, lw=2,
                    markersize=6, label=f"{key}  ({tot:.1f}σ)")

        ax.axhline(1, color='gray', ls='--', lw=1)
        ax.axhline(5, color='gray', ls=':',  lw=1, label='5σ')

        # x_e secondary axis
        xe_marks = [0.10, 0.18, 0.31, 0.51, 0.77]
        z_marks  = [float(np.interp(xe, xe_nodes[::-1], z_nodes_s[::-1]))
                    for xe in xe_marks]
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(z_marks)
        ax2.set_xticklabels([f'{xe:.2f}' for xe in xe_marks])
        ax2.set_xlabel(r'$\bar{x}_{\rm HII}$')

        ax.invert_xaxis()
        ax.set_xlabel(r'Redshift $z_0$')
        ax.set_ylabel(r'SNR$(z_0)$  [ℓ-integrated, noise-free]')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    fig.suptitle(r'kSZ² SNR forecast — noise-free upper bound'
                 f' ({N_SEEDS} seeds)', fontweight='bold')
    fig.savefig(f"{plot_dir_snr}/snr_vs_z_corrected.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)
print("\n  ✓ Saved: snr_vs_z_corrected.png")

# ── Plot 2: Correlation coefficient r(ℓ) at best z0 ─────────────────────────
with mpl.rc_context(PNG_STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             constrained_layout=True)
    for ax, stat, title in zip(axes, stat_keys, titles):
        key    = "CMBS4×SKA"
        per_z, _ = forecasts[key][stat]
        if not per_z:
            continue
        best_z = max(per_z, key=lambda z: per_z[z]['snr_total'])
        best   = per_z[best_z]
        xe_b   = float(np.interp(best_z, z_nodes_s[::-1], xe_nodes[::-1]))
        ell    = best['ell']
        valid  = np.isfinite(best['r']) & (ell >= 100)

        ax.plot(ell[valid], best['r'][valid], 'k-', lw=2)
        ax.axhline(0,  color='gray', ls='--', lw=1)
        ax.axhline( 0.1, color='blue', ls=':', lw=1, alpha=0.5)
        ax.axhline(-0.1, color='blue', ls=':', lw=1, alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel(r'Multipole $\ell$')
        ax.set_ylabel(r'Correlation coefficient $r_\ell$')
        ax.set_title(f'{title}\n'
                     rf'$z_0={best_z:.1f}$, $x_e={xe_b:.2f}$',
                     fontweight='bold')
        ax.set_ylim(-1, 1)

    fig.suptitle(r'Cross-correlation coefficient $r_\ell = C_\ell / \sqrt{P_{\rm kSZ^2} P_{\rm 21cm}}$',
                 fontweight='bold')
    fig.savefig(f"{plot_dir_snr}/correlation_coefficient_r_ell.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)
print("  ✓ Saved: correlation_coefficient_r_ell.png")

# ── Plot 3: Cumulative SNR vs ℓ ──────────────────────────────────────────────
with mpl.rc_context(PNG_STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             constrained_layout=True, sharey=False)
    for ax, stat, title in zip(axes, stat_keys, titles):
        for (cmb_p, ts_p), color in zip(combinations, combo_colors):
            key    = f"{cmb_p}×{ts_p}"
            per_z, tot = forecasts[key][stat]
            if not per_z:
                continue
            best_z = max(per_z, key=lambda z: per_z[z]['snr_total'])
            best   = per_z[best_z]
            xe_b   = float(np.interp(best_z, z_nodes_s[::-1],
                                     xe_nodes[::-1]))
            ax.plot(best['ell'], best['snr_cumul'], color=color, lw=2,
                    label=f"{key}  z₀={best_z:.0f}, xₑ={xe_b:.2f}"
                          f"  ({tot:.1f}σ)")

        ax.axhline(1, color='gray', ls='--', lw=1)
        ax.axhline(5, color='gray', ls=':',  lw=1, label='5σ')
        ax.set_xscale('log')
        ax.set_xlabel(r'$\ell_{\rm max}$')
        ax.set_ylabel('Cumulative SNR')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    fig.suptitle('Cumulative SNR vs ℓ_max — noise-free upper bound',
                 fontweight='bold')
    fig.savefig(f"{plot_dir_snr}/snr_cumul_corrected.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)
print("  ✓ Saved: snr_cumul_corrected.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TOTAL SNR FORECAST (noise-free upper bound)")
print(f"  f_sky={f_sky:.3f}  ({f_sky*41253:.0f} deg²),  N_seeds={N_SEEDS}")
print(f"\n  {'Combination':<22}  {'kSZ²×21cm':>12}  {'kSZ²×21cm²':>12}")
print("  " + "-"*50)
for cmb_p, ts_p in combinations:
    key = f"{cmb_p}×{ts_p}"
    _, tot_un = forecasts[key]['un']
    _, tot_sq = forecasts[key]['sq']
    flag = lambda s: "✓(≥5σ)" if s >= 5 else ("~(≥1σ)" if s >= 1 else "✗")
    print(f"  {key:<22}  {tot_un:>8.2f}σ {flag(tot_un):<8}  "
          f"{tot_sq:>8.2f}σ {flag(tot_sq):<8}")
print("="*60)
print("\n  NOTE: This is noise-free (cosmic variance only).")
print("  To add instrument noise, re-run Cell 7/7b with noise-added")
print("  21cm and kSZ fields before computing auto-powers.")
print("\n✓ CELL SNR CORRECTED COMPLETE")
print(f"  Plots → {plot_dir_snr}/")
