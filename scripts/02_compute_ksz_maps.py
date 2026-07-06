#!/usr/bin/env python
# =============================================================================
# 02_compute_ksz_maps.py
#
# For every seed: reionization history + optical depth tau(z), the kSZ
# integrand, and the line-of-sight integrated 2D kSZ map at z_obs.
#
# The physics (pure functions, no I/O, no plotting) lives in
# src/ksz2_21cm/ksz/ksz_map.py: compute_optical_depth, compute_ksz_integrand,
# compute_ksz_map. This script owns loading lightcones, caching kSZ maps to
# disk, and the diagnostic plots (reionization history, tau(z)).
#
# Extracted from CELL 4 (reionization + tau), CELL 5 (kSZ integrand), and
# CELL 6 (LoS integration) of kSZ_Squared_21cm_11Jun_CLUSTER.py. The
# commented-out per-seed field plots (old CELL 2b) and integrand-slice plot
# (old CELL 5 plot block) were already disabled in your original script and
# have been left out here too — see notebooks/exploratory/ if you want them
# back for a specific debugging session.
#
# Run (after 01_run_lightcones.py):
#   conda activate ksz2-21cm
#   python scripts/02_compute_ksz_maps.py --config configs/fiducial.yaml
# =============================================================================

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import py21cmfast as p21c

from ksz2_21cm.utils.config import load_config, ensure_dirs
from ksz2_21cm.ksz.ksz_map import (
    compute_optical_depth, compute_ksz_integrand, compute_ksz_map,
)
from ksz2_21cm.plotting.style import apply_global_style, save_pdf_png

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/fiducial.yaml")
args = parser.parse_args()

cfg = load_config(args.config)
ensure_dirs(cfg)
apply_global_style()

cache_dir    = cfg["paths"]["cache_dir"]
plot_dir     = cfg["paths"]["plot_dir"]
RANDOM_SEEDS = cfg["simulation"]["random_seeds"]
z_max        = cfg["simulation"]["z_max"]
z_obs        = cfg["ksz"]["z_obs"]

# ── Load lightcones ───────────────────────────────────────────────────────────
print("Loading lightcones...")
lightcones = {}
for seed in RANDOM_SEEDS:
    lc_files = sorted(glob.glob(
        os.path.join(cache_dir, f"seed_{seed}", "LightCone_*.h5")))
    if lc_files:
        lightcones[seed] = p21c.LightCone.read(lc_files[0])
N_SEEDS = len(lightcones)
print(f"  {N_SEEDS} lightcones loaded")
assert N_SEEDS > 0, "No lightcones found — run 01_run_lightcones.py first."

# ── Per-seed: optical depth, integrand, kSZ map ───────────────────────────────
kSZ_maps_dir = os.path.join(cache_dir, "kSZ_maps")
os.makedirs(kSZ_maps_dir, exist_ok=True)

tau_results = {}
kSZ_maps    = {}

for seed, lc in lightcones.items():
    print(f"\n--- Seed {seed} ---")

    tau_results[seed] = compute_optical_depth(lc, z_max)
    print(f"  tau_total = {tau_results[seed]['tau_total']:.6f}")

    map_path = os.path.join(kSZ_maps_dir, f"kSZ_map_z{z_obs:.1f}_seed{seed}.npy")
    if os.path.exists(map_path):
        kSZ_maps[seed] = np.load(map_path)
        print(f"  kSZ map: loaded from cache "
              f"(RMS={np.sqrt(np.mean(kSZ_maps[seed]**2)):.4e})")
        continue

    kSZ_integrand = compute_ksz_integrand(lc, tau_results[seed], z_max)
    kSZ_map = compute_ksz_map(kSZ_integrand, tau_results[seed], z_obs)
    np.save(map_path, kSZ_map)
    kSZ_maps[seed] = kSZ_map
    print(f"  kSZ map: computed and cached "
          f"(RMS={np.sqrt(np.mean(kSZ_map**2)):.4e})  -> {map_path}")

tau_totals = np.array([tau_results[s]['tau_total'] for s in lightcones])
print(f"\n\u2713 kSZ maps ready: {len(kSZ_maps)}/{N_SEEDS} seeds")
print(f"  Mean tau = {tau_totals.mean():.6f} \u00b1 {tau_totals.std():.6f}")

# ── Diagnostic plots: reionization history + tau(z) ──────────────────────────
# (CELL 4, Part A/B plots — unchanged logic, just reading from tau_results
# computed above instead of an in-memory global.)

all_z_nodes, all_x_e_nodes, all_xHI_nodes = {}, {}, {}
for seed, lc in lightcones.items():
    sort_idx            = np.argsort(lc.node_redshifts)
    all_z_nodes[seed]   = lc.node_redshifts[sort_idx]
    all_x_e_nodes[seed] = (1.0 - lc.global_xH)[sort_idx]
    all_xHI_nodes[seed] = lc.global_xH[sort_idx]

z_min_common = max(all_z_nodes[s].min() for s in lightcones)
z_max_common = min(all_z_nodes[s].max() for s in lightcones)
z_common_xe  = np.linspace(z_min_common, z_max_common, 500)

xe_interp = np.array([np.interp(z_common_xe, all_z_nodes[s], all_x_e_nodes[s])
                      for s in lightcones])
xe_mean, xe_std = np.mean(xe_interp, axis=0), np.std(xe_interp, axis=0)
z_xe_half_mean = np.interp(0.5, xe_mean[::-1], z_common_xe[::-1])
print(f"\nMean z(x_e=0.5) across {N_SEEDS} seeds: z = {z_xe_half_mean:.2f}")

cmap_seeds = None
try:
    import matplotlib.pyplot as plt
    cmap_seeds = plt.cm.plasma(np.linspace(0.1, 0.9, len(lightcones)))
except Exception:
    pass


def _draw_xe(ax):
    for i, seed in enumerate(lightcones):
        ax.plot(all_z_nodes[seed], all_x_e_nodes[seed], color=cmap_seeds[i],
                lw=1.0, alpha=0.4)
    ax.fill_between(z_common_xe, xe_mean - xe_std, xe_mean + xe_std,
                    color='darkblue', alpha=0.2, label=r'$\pm 1\sigma$')
    ax.plot(z_common_xe, xe_mean, color='darkblue', lw=2.5, label=f'Mean ({N_SEEDS} seeds)')
    ax.axhline(0.5, color='gray', ls='--', lw=1, alpha=0.7)
    ax.axvline(z_xe_half_mean, color='gray', ls='--', lw=1, alpha=0.7)
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Ionization Fraction $x_e$')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best')
    ax.invert_xaxis()


save_pdf_png(_draw_xe, plot_dir, "reionization_history_xe",
            title='Reionization History: Ionization Fraction')
print("\u2713 Saved: reionization_history_xe")

ref_seed  = next(iter(lightcones))
ref_z_mid = tau_results[ref_seed]['z_mid']
grids_match = all(
    tau_results[s]['z_mid'].shape == ref_z_mid.shape
    and np.allclose(tau_results[s]['z_mid'], ref_z_mid)
    for s in lightcones
)
if grids_match:
    z_common_tau = ref_z_mid
    tau_matrix   = np.array([tau_results[s]['tau'] for s in lightcones])
else:
    z_lo = max(tau_results[s]['z_mid'].min() for s in lightcones)
    z_hi = min(tau_results[s]['z_mid'].max() for s in lightcones)
    z_common_tau = np.linspace(z_lo, z_hi, 1000)
    tau_matrix   = np.array([
        np.interp(z_common_tau, tau_results[s]['z_mid'], tau_results[s]['tau'])
        for s in lightcones
    ])
tau_mean, tau_std = np.mean(tau_matrix, axis=0), np.std(tau_matrix, axis=0)


def _draw_tau(ax):
    for i, seed in enumerate(lightcones):
        ax.plot(tau_results[seed]['z_mid'], tau_results[seed]['tau'],
                color=cmap_seeds[i], lw=1.0, alpha=0.4)
    ax.fill_between(z_common_tau, tau_mean - tau_std, tau_mean + tau_std,
                    color='darkgreen', alpha=0.2, label=r'$\pm 1\sigma$')
    ax.plot(z_common_tau, tau_mean, color='darkgreen', lw=2.5,
            label=f'Mean ({N_SEEDS} seeds)')
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Cumulative Optical Depth $\tau(<z)$')
    ax.legend(loc='best')
    ax.invert_xaxis()
    ax.text(0.05, 0.95, f'Mean \u03c4 = {tau_totals.mean():.4f} \u00b1 {tau_totals.std():.4f}\n'
            f'N seeds = {N_SEEDS}', transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


save_pdf_png(_draw_tau, plot_dir, "tau_vs_z", title='Cumulative Optical Depth vs Redshift')
print("\u2713 Saved: tau_vs_z")

print(f"\n\u2713 CELL 4/5/6 pipeline complete. Plots -> {plot_dir}/")
print("Next: python scripts/03_compute_cross_corr.py "
      "and python scripts/04_compute_cross_corr_sq.py")
