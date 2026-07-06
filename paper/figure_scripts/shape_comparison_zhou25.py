"""
shape_comparison_zhou25.py
===========================
[RECONSTRUCTED — not a verbatim recovery of your original script]

You sent two output plots (A_Dell_vs_ell_z9_shape.png, B_Dell_vs_z_shape.png)
from ~2 weeks ago without the code that made them. This script is
reconstructed from what's visible in those images, not recovered from a
file. If you find the real one later, diff it against this and keep
whichever is right — treat this as a working stand-in, not gospel.

What makes this different from overlay_zhou25.py:
  overlay_zhou25.py compares AMPLITUDE — it applies UNIT_FACTOR to convert
  our internal FFT units to muK^4, which is honestly a guess (see that
  script's docstring). This script sidesteps that entirely: every curve
  (ours and Zhou+25's) is divided by its own peak |D_ell|, so the unit
  ambiguity cancels out algebraically — normalizing a curve by its own
  peak erases any constant multiplicative factor you'd have applied to it.
  That's almost certainly *why* this plot is the one that made Prof.
  Kulkarni happy: it makes an honest shape/morphology claim instead of an
  amplitude claim your unit conversion can't yet back up.

Two plots, matching the two images you sent:
  A) D_ell/|D_ell|_peak vs ell at z0=9   (vs Zhou+25 Fig.3 / Fig.4, peak-normalized)
  B) D_ell/|D_ell|_peak vs z at fixed ell (vs Zhou+25 Fig.5, peak-normalized)

Run (after scripts/04_compute_cross_corr_sq.py):
    conda activate ksz2-21cm
    python3 paper/figure_scripts/shape_comparison_zhou25.py --config configs/fiducial.yaml
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ksz2_21cm.utils.config import load_config
from ksz2_21cm.io.cache import load_seed_caches, seed_average
from ksz2_21cm.plotting.zhou25_data import ZHOU_FIG5, ZHOU_FIG3_ell, ZHOU_FIG3_D, ZHOU_FIG4_ell, ZHOU_FIG4_D

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/fiducial.yaml")
args = parser.parse_args()
cfg = load_config(args.config)

CACHE_DIR = cfg["paths"]["cache_dir"]
OUT_DIR   = os.path.join(cfg["paths"]["plot_dir"], "shape_comparison_zhou25")
os.makedirs(OUT_DIR, exist_ok=True)
SEEDS = cfg["simulation"]["random_seeds"]

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 14, 'axes.labelsize': 16,
    'axes.titlesize': 15, 'xtick.labelsize': 13, 'ytick.labelsize': 13,
    'axes.grid': False,
})


def peak_normalize(arr):
    """Divide by the maximum |value| over finite entries. This is the whole
    trick: any constant amplitude/unit factor cancels out, so it doesn't
    matter that our pipeline's units aren't calibrated to muK^4 yet."""
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return arr
    peak = np.max(np.abs(arr[finite]))
    if peak == 0 or not np.isfinite(peak):
        return arr
    return arr / peak


# ── Load our caches ───────────────────────────────────────────────────────────
print("Loading caches...")
sq_all = load_seed_caches("cross_corr_sq_seed{seed}.npy", SEEDS, CACHE_DIR)
print(f"  {len(sq_all)} seeds loaded")
assert len(sq_all) > 0, "No cross_corr_sq caches found — run scripts/04_compute_cross_corr_sq.py first."

ref_seed = next(iter(sq_all))
z0_list  = sorted(sq_all[ref_seed].keys())

# ─────────────────────────────────────────────────────────────────────────────
# PLOT A: D_ell/peak vs ell at z0=9, shape only — vs Zhou+25 Fig.3 & Fig.4
# ─────────────────────────────────────────────────────────────────────────────
print("\nPlot A: shape comparison vs ell at z0=9...")

z0_9 = min(z0_list, key=lambda z: abs(z - 9.0))
ell_our = sq_all[ref_seed][z0_9]['ell']
D_mean, D_std = seed_average(sq_all, z0_9, 'D_cross')

fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

valid = np.isfinite(D_mean) & (ell_our >= 100)
peak  = np.max(np.abs(D_mean[valid]))
D_norm = D_mean[valid] / peak
E_norm = D_std[valid] / peak

ax.plot(ell_our[valid], D_norm, 'k-', lw=2.5,
        label=rf'This work ($z_0={z0_9:.0f}$, $k_{{\parallel,0}}=0.01$, $\Delta z=2$)')
ax.fill_between(ell_our[valid], D_norm - E_norm, D_norm + E_norm,
                alpha=0.25, color='gray')

ax.plot(zhou_fig3_ell := ZHOU_FIG3_ell, peak_normalize(ZHOU_FIG3_D), 'b--', lw=2,
        label=r'Zhou+25 Fig.3 (SO$\times$HERA, $k_{\parallel,0}=0.01$, $\Delta z=2$)')
ax.plot(zhou_fig4_ell := ZHOU_FIG4_ell, peak_normalize(ZHOU_FIG4_D), 'r--', lw=2,
        label=r'Zhou+25 Fig.4 (SO$\times$SKA, $k_{\parallel,0}=0.01$, $\Delta z=2$)')

ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_xscale('log')
ax.set_xlabel(r'Multipole $\ell$')
ax.set_ylabel(r'$D_\ell/|D_\ell|_{\rm peak}$ [normalized]')
ax.set_title(r'kSZ$^2\times$21cm$^2$ cross-power at $z_0=9$ — shape comparison')
ax.legend(fontsize=10, loc='lower right')
ax.text(0.98, 0.05, 'Each curve normalized to its own peak.\nAmplitude comparison not yet valid.',
        transform=ax.transAxes, fontsize=9, color='gray', ha='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.savefig(f"{OUT_DIR}/A_Dell_vs_ell_z9_shape.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: A_Dell_vs_ell_z9_shape.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT B: D_ell/peak vs z at fixed ell, shape only — vs Zhou+25 Fig.5
# ─────────────────────────────────────────────────────────────────────────────
print("Plot B: shape comparison vs z...")

target_ells = [400, 800, 1600, 3200]
colors      = ['steelblue', 'darkorange', 'forestgreen', 'darkred']

fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

for ax, (title, show_zhou) in zip(axes, [('This work only', False),
                                          ('Shape comparison with Zhou+25 Fig.5', True)]):
    for ell_t, color in zip(target_ells, colors):
        z_vals, D_vals = [], []
        for z0 in z0_list:
            ell_arr = sq_all[ref_seed][z0]['ell']
            D_m, _ = seed_average(sq_all, z0, 'D_cross')
            if D_m is None:
                continue
            idx = int(np.argmin(np.abs(ell_arr - ell_t)))
            if abs(ell_arr[idx] - ell_t) > 200 or not np.isfinite(D_m[idx]):
                continue
            z_vals.append(z0)
            D_vals.append(D_m[idx])

        if z_vals:
            z_arr = np.array(z_vals)
            D_arr = peak_normalize(np.array(D_vals))
            ax.plot(z_arr, D_arr, 'o-', color=color, lw=2, markersize=6,
                    label=rf'$\ell={ell_t}$')

    if show_zhou:
        for ell_t, color in zip(target_ells, colors):
            data = ZHOU_FIG5[ell_t]
            ax.plot(data[:, 0], peak_normalize(data[:, 1]), '--', color=color, lw=1.5)

    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.invert_xaxis()
    ax.set_xlabel(r'Redshift $z_0$')
    ax.set_ylabel(r'$D_\ell/|D_\ell|_{\rm peak}$ [normalized]')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')

axes[1].text(0.98, 0.05, 'Solid: this work  —  Dashed: Zhou+25 Fig.5\nEach curve normalized to its own peak.',
             transform=axes[1].transAxes, fontsize=9, color='gray', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.suptitle(r'kSZ$^2\times$21cm$^2$: shape comparison with Zhou+25', fontweight='bold')
fig.savefig(f"{OUT_DIR}/B_Dell_vs_z_shape.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: B_Dell_vs_z_shape.png")

print(f"\nAll shape-comparison plots saved to:\n  {OUT_DIR}/")
