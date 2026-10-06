#!/usr/bin/env python
# =============================================================================
# 02_compute_ksz_maps_stitched.py
#
# NEW SCRIPT (2026-10-06) -- companion to 01_stitch_lightcones.py. Builds
# the 2D kSZ map from each seed's stitched .npz using the vendored
# src/ksz2_21cm/ksz/optical_depth.py + ksz/lightcone_integral.py (both
# verbatim from ksz-pipeline), instead of the old ksz_map.py
# (compute_optical_depth/compute_ksz_integrand/compute_ksz_map) that
# 02_compute_ksz_maps.py uses on the native lightcone.
#
# Does NOT replace 02_compute_ksz_maps.py -- same reasoning as
# 01_stitch_lightcones.py's header: old pathway stays intact as the
# comparison baseline.
#
# Key difference from the old pathway: no single z_obs cutoff. Instead,
# optical_depth.compute_patchy_mask() clips the kSZ SIGNAL integral to the
# 99.99%-to-0.01%-neutral x_e window directly (ksz-pipeline's own
# convention), while visibility/tau is built from the FULL z_snapshots
# window plus the analytic homogeneous-tau contribution below it
# (analytic_tau_below) -- see optical_depth.py's own docstrings for why
# those two things are deliberately NOT clipped the same way.
#
# Run (after 01_stitch_lightcones.py):
#   conda activate ksz2-21cm
#   python scripts/02_compute_ksz_maps_stitched.py --config configs/variants/stitched_256.yaml
# =============================================================================

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ksz2_21cm.utils.config import load_config, ensure_dirs
from ksz2_21cm.utils.constants import MPC_CM, ne0_cgs
from ksz2_21cm.ksz.optical_depth import (
    analytic_tau_below, compute_tau, compute_patchy_mask, compute_visibility,
)
from ksz2_21cm.ksz.lightcone_integral import compute_ksz_map
from ksz2_21cm.plotting.style import apply_global_style, save_pdf_png

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/variants/stitched_256.yaml")
args = parser.parse_args()

cfg = load_config(args.config)
ensure_dirs(cfg)
apply_global_style()

cache_dir    = cfg["paths"]["cache_dir"]
plot_dir     = cfg["paths"]["plot_dir"]
RANDOM_SEEDS = cfg["simulation"]["random_seeds"]

# GUARD: the stitched arrays were built at coeval_ksz.hii_dim/box_len (see
# 01_stitch_lightcones.py); nothing downstream re-derives pixel geometry
# from the .npz itself, so a future edit to one without the other would
# silently assume the wrong pixel size rather than fail loudly. Catch it here.
assert cfg["simulation"]["hii_dim"] == cfg["coeval_ksz"]["hii_dim"], (
    "simulation.hii_dim != coeval_ksz.hii_dim -- these must match, see "
    "this script's header")
assert cfg["simulation"]["box_len"] == cfg["coeval_ksz"]["box_len"], (
    "simulation.box_len != coeval_ksz.box_len -- these must match, see "
    "this script's header")

NE0 = ne0_cgs()   # helium-inclusive, matches the vendored modules' own default

# ── Load stitched lightcones ──────────────────────────────────────────────
print("Loading stitched lightcones...")
stitched = {}
for seed in RANDOM_SEEDS:
    npz_path = os.path.join(cache_dir, f"seed_{seed}",
                             f"stitched_lightcone_seed{seed}.npz")
    if os.path.exists(npz_path):
        stitched[seed] = np.load(npz_path)
N_SEEDS = len(stitched)
print(f"  {N_SEEDS} stitched lightcones loaded")
assert N_SEEDS > 0, "No stitched lightcones found — run 01_stitch_lightcones.py first."

# ── Per-seed: tau(z), patchy mask, kSZ map ───────────────────────────────────
kSZ_maps_dir = os.path.join(cache_dir, "kSZ_maps")
os.makedirs(kSZ_maps_dir, exist_ok=True)

tau_results = {}
kSZ_maps    = {}

for seed, d in stitched.items():
    print(f"\n--- Seed {seed} ---")

    z_arr    = d["z_arr"]
    pos_axis = d["pos_axis"]
    xH_box   = d["xH_box"]
    density  = d["density"]
    vz       = d["velocity_z"]

    x_e = 1.0 - xH_box.mean(axis=(0, 1))   # global (volume-averaged) x_e(z_arr)

    tau0 = analytic_tau_below(float(z_arr.min()))
    z_mid, ds, dtau, tau = compute_tau(x_e, z_arr, pos_axis, ne0=NE0, tau0=tau0)
    tau_at_lc, visibility, visibility3D = compute_visibility(tau, z_arr, z_mid)
    patchy_mask, patchy_mask_3D = compute_patchy_mask(x_e)

    tau_results[seed] = {
        'z_mid': z_mid, 'tau': tau, 'tau_total': float(tau[-1]),
        'z_arr': z_arr, 'x_e': x_e, 'tau0_below_zmin': tau0,
    }
    print(f"  tau0 (z<{z_arr.min():.2f}, analytic) = {tau0:.6f}")
    print(f"  tau_total (incl. tau0)               = {tau_results[seed]['tau_total']:.6f}")

    map_path = os.path.join(kSZ_maps_dir, f"kSZ_map_stitched_seed{seed}.npy")
    if os.path.exists(map_path):
        kSZ_maps[seed] = np.load(map_path)
        print(f"  kSZ map: loaded from cache "
              f"(RMS={np.sqrt(np.mean(kSZ_maps[seed]**2)):.4e})")
        continue

    density_1plus = 1.0 + density
    x_HII_field   = 1.0 - xH_box
    v_los_Mpc_s   = vz / MPC_CM

    kSZ_map = compute_ksz_map(
        density_1plus, x_HII_field, v_los_Mpc_s,
        red_axis=z_arr, ds=ds, visibility_3D=visibility3D,
        ne0=NE0, patchy_mask_3D=patchy_mask_3D,
    )
    np.save(map_path, kSZ_map)
    kSZ_maps[seed] = kSZ_map
    print(f"  kSZ map: computed and cached "
          f"(RMS={np.sqrt(np.mean(kSZ_map**2)):.4e})  -> {map_path}")

tau_totals = np.array([tau_results[s]['tau_total'] for s in stitched])
print(f"\n✓ kSZ maps ready: {len(kSZ_maps)}/{N_SEEDS} seeds")
print(f"  Mean tau = {tau_totals.mean():.6f} ± {tau_totals.std():.6f}")
print(f"  (compare against the native-lightcone pathway's tau_total from "
      f"02_compute_ksz_maps.py at the same astro params, as a sanity check)")

# ── Diagnostic plots: reionization history + tau(z) ──────────────────────────
ref_seed  = next(iter(stitched))
z_common_xe = tau_results[ref_seed]['z_arr']
grids_match = all(np.array_equal(tau_results[s]['z_arr'], z_common_xe) for s in stitched)
if grids_match:
    xe_matrix = np.array([tau_results[s]['x_e'] for s in stitched])
else:
    # Different seeds could in principle run with different z_arr if
    # coeval_ksz settings changed mid-ensemble -- interpolate onto a
    # common grid rather than assume they match.
    z_lo = max(tau_results[s]['z_arr'].min() for s in stitched)
    z_hi = min(tau_results[s]['z_arr'].max() for s in stitched)
    z_common_xe = np.linspace(z_lo, z_hi, 500)
    xe_matrix = np.array([
        np.interp(z_common_xe, tau_results[s]['z_arr'], tau_results[s]['x_e'])
        for s in stitched
    ])
xe_mean, xe_std = xe_matrix.mean(axis=0), xe_matrix.std(axis=0)
# xe_mean is DESCENDING against ascending z_common_xe (see 01's header note
# on this exact reversed-array trap) -- reverse both before interpolating.
z_xe_half_mean = np.interp(0.5, xe_mean[::-1], z_common_xe[::-1])
print(f"\nMean z(x_e=0.5) across {N_SEEDS} seeds: z = {z_xe_half_mean:.2f}")

cmap_seeds = None
try:
    import matplotlib.pyplot as plt
    cmap_seeds = plt.cm.plasma(np.linspace(0.1, 0.9, len(stitched)))
except Exception:
    pass


def _draw_xe(ax):
    for i, seed in enumerate(stitched):
        ax.plot(tau_results[seed]['z_arr'], tau_results[seed]['x_e'],
                color=cmap_seeds[i], lw=1.0, alpha=0.4)
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


save_pdf_png(_draw_xe, plot_dir, "reionization_history_xe_stitched",
            title='Reionization History (stitched boxes): Ionization Fraction')
print("✓ Saved: reionization_history_xe_stitched")


def _draw_tau(ax):
    for i, seed in enumerate(stitched):
        ax.plot(tau_results[seed]['z_mid'], tau_results[seed]['tau'],
                color=cmap_seeds[i], lw=1.0, alpha=0.4)
    ax.set_xlabel(r'Redshift $z$')
    ax.set_ylabel(r'Cumulative Optical Depth $\tau(<z)$ (incl. analytic $\tau_0$)')
    ax.invert_xaxis()
    ax.text(0.05, 0.95, f'Mean τ = {tau_totals.mean():.4f} ± {tau_totals.std():.4f}\n'
            f'N seeds = {N_SEEDS}', transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


save_pdf_png(_draw_tau, plot_dir, "tau_vs_z_stitched",
            title='Cumulative Optical Depth vs Redshift (stitched boxes)')
print("✓ Saved: tau_vs_z_stitched")

print(f"\n✓ Stitched kSZ-map pipeline complete. Plots -> {plot_dir}/")
print("Next: python scripts/04_compute_cross_corr_sq_stitched.py "
      "--config configs/variants/stitched_256.yaml")
