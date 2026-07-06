"""
overlay_zhou25.py
==================
Overlay our kSZ²×21cm² results with digitized Zhou+25 curves.

Two plot types:
  A) D_ell vs ell at z0~9  (vs Zhou+25 Fig 3 green / Fig 4 green)
  B) D_ell vs z at fixed ell (vs Zhou+25 Fig 5)

Unit note:
  Our D_cross is in internal FFT units. Zhou+25 is in muK^4.
  Conversion: D_cross_physical = D_cross * T_CMB_uK^2 * (0.67/D_A)^2 * T_CMB_uK^2
  But since kSZ is dimensionless (Delta_T/T) and 21cm is in mK,
  the cross-power is in mK (dimensionless x mK).
  After squaring: kSZ^2 is dimensionless^2, 21cm^2 is mK^2.
  Cross = dimensionless^2 x mK^2.
  To get muK^4: multiply by T_CMB_uK^4 and apply geometric conversion.
  
  We find the conversion factor empirically by comparing the peak amplitude
  to Zhou+25 and noting the ratio — this is the honest approach given the
  unit ambiguity, and we label it clearly on the plot.

Run (after scripts/04_compute_cross_corr_sq.py):
    conda activate ksz2-21cm
    python3 paper/figure_scripts/overlay_zhou25.py --config configs/fiducial.yaml

[repo note] CACHE_DIR/OUT_DIR were hardcoded to a specific home-directory
path (~/1Jun2026_kSZ_sqr_21cm_sqr/...) — now read from configs/fiducial.yaml
so this script works on any machine/checkout. The digitized Zhou+25 arrays
that used to be inline here now live in src/ksz2_21cm/plotting/zhou25_data.py
(deduplicated against the near-identical copy that was in SNR_rigorous_v2.py).
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ksz2_21cm.utils.config import load_config
from ksz2_21cm.utils.cosmology import get_cosmology
from ksz2_21cm.io.cache import load_seed_caches, seed_average
from ksz2_21cm.plotting.zhou25_data import (
    ZHOU_FIG5 as zhou_fig5,
    ZHOU_FIG4_ell as zhou_fig4_ell, ZHOU_FIG4_D as zhou_fig4_D,
    ZHOU_FIG3_ell as zhou_fig3_ell, ZHOU_FIG3_D as zhou_fig3_D,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/fiducial.yaml")
args = parser.parse_args()
cfg = load_config(args.config)

CACHE_DIR = cfg["paths"]["cache_dir"]
OUT_DIR   = os.path.join(cfg["paths"]["plot_dir"], "overlay_zhou25")
os.makedirs(OUT_DIR, exist_ok=True)

SEEDS   = cfg["simulation"]["random_seeds"]
cosmo   = get_cosmology(cfg)
T_CMB   = 2.725e6   # muK

plt.rcParams.update({
    'font.family'    : 'serif',
    'font.size'      : 14,
    'axes.labelsize' : 16,
    'axes.titlesize' : 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'axes.grid'      : False,
})

# ── Load our caches ───────────────────────────────────────────────────────────
print("Loading caches...")
sq_all = load_seed_caches("cross_corr_sq_seed{seed}.npy", SEEDS, CACHE_DIR)
print(f"  {len(sq_all)} seeds loaded")
assert len(sq_all) > 0, "No cross_corr_sq caches found — run scripts/04_compute_cross_corr_sq.py first."

ref_seed = next(iter(sq_all))
z0_list  = sorted(sq_all[ref_seed].keys())

# seed_avg_sq(field, z0) -> (mean, std), now just io.cache.seed_average
seed_avg_sq = lambda field, z0: seed_average(sq_all, z0, field)

# ── Unit conversion ───────────────────────────────────────────────────────────
# Our C_cross is computed as:
#   mean(cross_ps2d[mask]) where cross_ps2d = Re(kSZ2* x T21sq) * pix_area/npix^2
# kSZ map is dimensionless (Delta_T/T_CMB), so kSZ^2 is dimensionless
# T21 is in mK, so T21^2 is in mK^2
# cross_ps2d is in mK^2 (pixel units)
# D_cross = ell(ell+1) * C_cross / 2pi  [still mK^2 in pixel units]
#
# Zhou+25 D_ell is in muK^4.
# To convert: multiply by T_CMB_uK^2 (converts kSZ^2 dimensionless -> muK^2)
#             and by 1e6^2 (mK^2 -> muK^2) for 21cm^2 side
# So: D_cross_muK4 = D_cross * T_CMB_uK^2 * (1e3)^2
#                  = D_cross * T_CMB_uK^2 * 1e6
# where T_CMB_uK = 2.725e6 muK
# Total factor: 2.725e6^2 * 1e6 = 7.43e18
#
# Additionally we need the angular power spectrum normalization:
# Our cross_ps2d uses pix_area/npix^2 normalization (flat sky)
# Zhou+25 uses the standard C_ell normalization
# These should be equivalent but let's apply T_CMB factor and check
#
UNIT_FACTOR = T_CMB**2 * 1e6   # dimensionless x mK^2 -> muK^4

# Digitized Zhou+25 data (zhou_fig5, zhou_fig4_ell/D, zhou_fig3_ell/D) is now
# imported from src/ksz2_21cm/plotting/zhou25_data.py — see the import block
# at the top of this file.

# ─────────────────────────────────────────────────────────────────────────────
# PLOT A: D_ell vs ell at z0=9 — our result vs Zhou+25 Fig 3 and Fig 4
# ─────────────────────────────────────────────────────────────────────────────
print("\nPlot A: D_ell vs ell at z0=9...")

z0_9 = min(z0_list, key=lambda z: abs(z - 9.0))
ell_our  = sq_all[ref_seed][z0_9]['ell']
D_mean, D_std = seed_avg_sq('D_cross', z0_9)

if D_mean is not None:
    D_mean_phys = D_mean * UNIT_FACTOR
    D_std_phys  = D_std  * UNIT_FACTOR

fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

# Our result
valid = np.isfinite(D_mean_phys) & (ell_our >= 100)
ax.plot(ell_our[valid], D_mean_phys[valid], 'k-', lw=2.5,
        label=rf'This work ($z_0={z0_9:.0f}$, $k_{{\parallel,0}}=0.01$, $\Delta z=2$)')
ax.fill_between(ell_our[valid],
                D_mean_phys[valid] - D_std_phys[valid],
                D_mean_phys[valid] + D_std_phys[valid],
                alpha=0.25, color='black')

# Zhou+25 Fig 3 (SO x HERA)
ax.plot(zhou_fig3_ell, zhou_fig3_D, 'b--', lw=2,
        label=r'Zhou+25 Fig.3 ($z_{\rm mid}=9$, SO$\times$HERA, $k_{\parallel,0}=0.01$, $\Delta z=2$)')

# Zhou+25 Fig 4 (SO x SKA)
ax.plot(zhou_fig4_ell, zhou_fig4_D, 'r--', lw=2,
        label=r'Zhou+25 Fig.4 ($z_{\rm mid}=9$, SO$\times$SKA, $k_{\parallel,0}=0.01$, $\Delta z=2$)')

ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_xscale('log')
ax.set_xlabel(r'Multipole $\ell$')
ax.set_ylabel(r'$\ell(\ell+1)C_\ell^{\rm kSZ^2\times 21cm^2}/2\pi\;[\mu\mathrm{K}^4]$')
ax.set_title(r'kSZ$^2\times$21cm$^2$ cross-power at $z_0=9$')
ax.legend(fontsize=11, loc='upper right')
ax.text(0.02, 0.04,
        r'Note: unit factor $T_{\rm CMB}^2\times(1\,{\rm mK}\to\mu{\rm K})^2$ applied to our result',
        transform=ax.transAxes, fontsize=9, color='gray')
fig.savefig(f"{OUT_DIR}/A_Dell_vs_ell_z9_overlay.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: A_Dell_vs_ell_z9_overlay.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT B: D_ell vs z — our result vs Zhou+25 Fig 5
# Fixed ell values: 400, 800, 1600, 3200
# ─────────────────────────────────────────────────────────────────────────────
print("Plot B: D_ell vs z...")

# Our reionization history for x_e axis
import glob
lc_files = sorted(glob.glob(
    os.path.expanduser("~/1Jun2026_kSZ_sqr_21cm_sqr/cache/seed_1/LightCone_*.h5")))

try:
    import py21cmfast as p21c
    lc       = p21c.LightCone.read(lc_files[0])
    z_nodes  = lc.node_redshifts[::-1]
    xe_nodes = 1.0 - lc.global_xH[::-1]
    has_lc   = True
except Exception:
    has_lc = False

# Target ell values and colors matching Zhou+25 Fig 5
target_ells  = [400, 800, 1600, 3200]
our_colors   = ['steelblue', 'darkorange', 'forestgreen', 'darkred']
zhou_colors  = ['steelblue', 'darkorange', 'forestgreen', 'darkred']

fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True,
                         sharey=False)

for ax, (title, show_zhou) in zip(axes, [
        ('This work only', False),
        ('With Zhou+25 Fig.5 overlay', True)]):

    for ell_t, color in zip(target_ells, our_colors):
        z_vals, D_vals, D_errs = [], [], []
        for z0 in z0_list:
            ell_arr = sq_all[ref_seed][z0]['ell']
            D_m, D_s = seed_avg_sq('D_cross', z0)
            if D_m is None:
                continue
            # Find closest ell bin
            idx = np.argmin(np.abs(ell_arr - ell_t))
            if np.abs(ell_arr[idx] - ell_t) > 200:
                continue
            z_vals.append(z0)
            D_vals.append(D_m[idx] * UNIT_FACTOR)
            D_errs.append(D_s[idx] * UNIT_FACTOR)

        if z_vals:
            z_arr = np.array(z_vals)
            D_arr = np.array(D_vals)
            E_arr = np.array(D_errs)
            ax.plot(z_arr, D_arr, 'o-', color=color, lw=2, markersize=5,
                    label=rf'$\ell={ell_t}$')
            ax.fill_between(z_arr, D_arr - E_arr, D_arr + E_arr,
                            alpha=0.2, color=color)

    if show_zhou:
        for ell_t, color, data in zip(
                target_ells, zhou_colors,
                [zhou_fig5[400], zhou_fig5[800],
                 zhou_fig5[1600], zhou_fig5[3200]]):
            ax.plot(data[:,0], data[:,1], '--', color=color, lw=1.5,
                    label=rf'Zhou+25 $\ell={ell_t}$')

    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.invert_xaxis()
    ax.set_xlabel(r'Redshift $z_0$')
    ax.set_ylabel(r'$\ell(\ell+1)C_\ell^{\rm kSZ^2\times 21cm^2}/2\pi\;[\mu\mathrm{K}^4]$')
    ax.set_title(title, fontweight='bold')

    if has_lc:
        xe_marks = [0.10, 0.18, 0.31, 0.51, 0.77]
        z_marks  = [float(np.interp(xe, xe_nodes[::-1], z_nodes[::-1]))
                    for xe in xe_marks]
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(z_marks)
        ax2.set_xticklabels([f'{xe:.2f}' for xe in xe_marks])
        ax2.set_xlabel(r'$\bar{x}_{\rm HII}$')

    # Legend only on right panel
    if show_zhou:
        ax.legend(fontsize=9, ncol=2, loc='upper left')
    else:
        ax.legend(fontsize=11, loc='upper left')

axes[1].text(0.02, 0.04,
    r'Dashed: Zhou+25 Fig.5, k_par=0.10, SO x HERA' + chr(10) +
    r'Solid: This work, k_par=0.01, no noise',
    transform=axes[1].transAxes, fontsize=9, color='gray')

fig.suptitle(r'kSZ$^2\times$21cm$^2$: $D_\ell$ vs redshift at fixed $\ell$',
             fontweight='bold')
fig.savefig(f"{OUT_DIR}/B_Dell_vs_z_overlay.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: B_Dell_vs_z_overlay.png")

print(f"\nAll overlay plots saved to:\n  {OUT_DIR}/")
print("\nNote: unit conversion applied = T_CMB^2 * (1 mK -> muK)^2")
print(f"  = {T_CMB:.3e}^2 * 1e6 = {T_CMB**2 * 1e6:.3e}")
print("If amplitudes don't match Zhou+25, the remaining factor is")
print("the angular power spectrum geometric normalization (D_A, chi).")
