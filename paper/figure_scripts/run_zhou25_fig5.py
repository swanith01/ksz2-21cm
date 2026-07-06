#!/usr/bin/env python
# =============================================================================
# run_zhou25_fig5.py
# Reproduces Zhou+25 Fig 5 style for our k_par_min=0.01 data.
# Single panel, 4 ell lines, shaded std error, xHII top axis.
# =============================================================================

import argparse
import os
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from ksz2_21cm.utils.config import load_config

# =============================================================================
# 1. Config
# [repo note] CODE_DIR/CACHE_DIR/PLOT_DIR were hardcoded to a specific
# /user1/swanith/... cluster path — now read from configs/fiducial.yaml.
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/fiducial.yaml")
args = parser.parse_args()
cfg = load_config(args.config)

CACHE_DIR    = cfg["paths"]["cache_dir"]
PLOT_DIR     = cfg["paths"]["plot_dir"]
RANDOM_SEEDS = cfg["simulation"]["random_seeds"]
os.makedirs(PLOT_DIR, exist_ok=True)

print("="*60)
print("  run_zhou25_fig5.py")
print("="*60)

# =============================================================================
# 2. Load cross_corr_sq caches
# =============================================================================
print("\nLoading caches ...")
ccr_all = {}
for seed in RANDOM_SEEDS:
    fpath = os.path.join(CACHE_DIR, f"seed_{seed}",
                         f"cross_corr_sq_seed{seed}.npy")
    if os.path.exists(fpath):
        ccr_all[seed] = np.load(fpath, allow_pickle=True).item()
        print(f"  seed {seed:2d}  {len(ccr_all[seed])} chunks")
    else:
        print(f"  seed {seed:2d}  NOT FOUND")

N_loaded = len(ccr_all)
print(f"  Loaded {N_loaded} seeds")
if N_loaded == 0:
    sys.exit(1)

# =============================================================================
# 3. Reionisation history for xHII top axis
# =============================================================================
seed1_dir = os.path.join(CACHE_DIR, "seed_1")
lc_files  = [f for f in os.listdir(seed1_dir)
              if f.startswith("LightCone") and f.endswith(".h5")]
with h5py.File(os.path.join(seed1_dir, lc_files[0]), "r") as f:
    xH_node = f["global_quantities"]["xH_box"][:]
    node_z  = f["node_redshifts"][:]

z_asc    = node_z[::-1]
xHII_asc = 1.0 - xH_node[::-1]

def z_to_xHII(z_arr):
    return np.interp(z_arr, z_asc, xHII_asc)

# =============================================================================
# 4. Seed-averaged D_ell(z) at 4 ell targets
# =============================================================================
ell_targets = [400, 800, 1600, 3200]
colors      = ['steelblue', 'darkorange', 'forestgreen', 'firebrick']

ref_seed = next(iter(ccr_all))
all_z0   = sorted(ccr_all[ref_seed].keys())
print(f"\nChunk centres: {all_z0}")

D_by_ell = {ell: {z0: [] for z0 in all_z0} for ell in ell_targets}

for seed, ccr in ccr_all.items():
    for z0 in all_z0:
        if z0 not in ccr:
            continue
        res     = ccr[z0]
        ell_arr = res['ell']
        D_arr   = res['D_cross']
        for ell_t in ell_targets:
            idx = np.argmin(np.abs(ell_arr - ell_t))
            if np.isfinite(D_arr[idx]):
                D_by_ell[ell_t][z0].append(D_arr[idx])

result = {}
for ell_t in ell_targets:
    z_plot, D_mean, D_err = [], [], []
    for z0 in all_z0:
        vals = np.array(D_by_ell[ell_t][z0])
        if len(vals) >= 2:
            z_plot.append(z0)
            D_mean.append(np.mean(vals))
            D_err.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    result[ell_t] = {
        'z'  : np.array(z_plot),
        'D'  : np.array(D_mean),
        'err': np.array(D_err),
    }
    n = len(z_plot)
    mx = np.nanmax(np.abs(D_mean)) if n > 0 else np.nan
    print(f"  ell={ell_t:4d}  {n} points  max|D|={mx:.3e}")

# =============================================================================
# 5. Plot
# =============================================================================
plt.rcParams.update({
    'font.family'    : 'serif',
    'font.size'      : 14,
    'axes.labelsize' : 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top'      : False,
    'ytick.right'    : True,
    'axes.grid'      : False,
    'mathtext.fontset': 'cm',
})

fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

for ell_t, color in zip(ell_targets, colors):
    r = result[ell_t]
    if len(r['z']) == 0:
        continue
    sort = np.argsort(r['z'])
    zs   = r['z'][sort]
    Ds   = r['D'][sort]
    Es   = r['err'][sort]
    ax.plot(zs, Ds, color=color, lw=2.0, label=rf'$\ell = {ell_t}$')
    ax.fill_between(zs, Ds - Es, Ds + Es, color=color, alpha=0.25)

ax.axhline(0, color='black', ls='--', lw=1.0)
ax.set_xlabel(r'$z$', fontsize=14)
ax.set_ylabel(r'$\ell(\ell+1)C_\ell/2\pi\ [\mu\mathrm{K}^4]$', fontsize=14)

# inverted z-axis matching Zhou+25 style (high z on left)
z_lo = min(all_z0) - 0.3
z_hi = max(all_z0) + 0.3
ax.set_xlim(z_hi, z_lo)

ax.legend(loc='upper right', framealpha=0.9)

# top x-axis: xHII
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
z_tick_vals  = np.array(all_z0, dtype=float)
xHII_vals    = z_to_xHII(z_tick_vals)
ax2.set_xticks(z_tick_vals)
ax2.set_xticklabels([f'{x:.2f}' for x in xHII_vals], fontsize=11)
ax2.set_xlabel(r'$\bar{x}_\mathrm{HII}$', fontsize=13)

ax.text(0.97, 0.97,
        r'$k_{\parallel,0} = 0.01\ h\,\mathrm{Mpc}^{-1}$' + '\n'
        r'$\Delta z = 2$' + f'\n{N_loaded} seeds',
        transform=ax.transAxes, va='top', ha='right', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))

fig.suptitle(
    r'kSZ$^2\times$21cm$^2$: $D_\ell$ vs $z_0$   (Zhou+25 Fig 5 style)',
    fontsize=13, fontweight='bold'
)

for ext, dpi in [('pdf', 300), ('png', 300)]:
    fname = os.path.join(PLOT_DIR, f'zhou25_fig5_style.{ext}')
    fig.savefig(fname, dpi=dpi, bbox_inches='tight')
    print(f'  Saved: {fname}')
plt.close(fig)

print("\nDONE")
