#!/usr/bin/env python
"""
ksz_raw_x_21cm.py

[NEW — did not exist in any form in the original code.]

Figures for the RAW kSZ x 21cm cross-correlation (neither field squared).
Two plot types, as requested:
  A) D_ell vs ell, for a handful of representative redshifts
  B) D_ell vs z, for a handful of fixed ell targets
Plus a correlation-coefficient (r_cross) panel for each — before asking
"what SNR would we get", it's worth first just asking "is there ANY
correlation here at all", which r_cross answers directly without needing
any of the physical unit-conversion machinery.

No Zhou+25 (or any literature) overlay here — this is a different
statistic (raw kSZ x 21cm) than anything in Zhou+25 or the Ma+2018
comparison used elsewhere in this repo, so there's no directly comparable
digitized curve to overlay against.

Run (after scripts/06_compute_cross_corr_raw.py):
    conda activate ksz2-21cm
    python3 paper/figure_scripts/ksz_raw_x_21cm.py --config configs/variants/quicktest.yaml
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ksz2_21cm.utils.config import load_config
from ksz2_21cm.io.cache import load_seed_caches, seed_average
from ksz2_21cm.plotting.style import apply_global_style

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/fiducial.yaml")
args = parser.parse_args()
cfg = load_config(args.config)
apply_global_style()

CACHE_DIR = cfg["paths"]["cache_dir"]
OUT_DIR   = os.path.join(cfg["paths"]["plot_dir"], "ksz_raw_x_21cm")
os.makedirs(OUT_DIR, exist_ok=True)
SEEDS = cfg["simulation"]["random_seeds"]

print("Loading caches...")
raw_all = load_seed_caches("cross_corr_raw_seed{seed}.npy", SEEDS, CACHE_DIR)
print(f"  {len(raw_all)} seeds loaded")
assert len(raw_all) > 0, (
    "No cross_corr_raw caches found — run scripts/06_compute_cross_corr_raw.py first.")

ref_seed = next(iter(raw_all))
z_list   = sorted(raw_all[ref_seed].keys())
print(f"  {len(z_list)} redshifts available: {z_list[0]:.2f} .. {z_list[-1]:.2f}")

# Reuse the pipeline's existing chunk-centre config as representative z
# targets for Plot A, snapped to the nearest available node redshift —
# reasonable reuse of an existing config value rather than inventing a new one.
z_targets_cfg = cfg["cross_correlation"]["z_chunk_centres"]
z_targets = sorted({min(z_list, key=lambda z: abs(z - zt)) for zt in z_targets_cfg})

ELL_TARGETS = [400, 800, 1600, 3200]
COLORS = plt.cm.viridis(np.linspace(0.15, 0.9, max(len(z_targets), len(ELL_TARGETS))))

# ─────────────────────────────────────────────────────────────────────────────
# PLOT A: D_ell vs ell, for each z_target — D_ell and r_cross side by side
# ─────────────────────────────────────────────────────────────────────────────
print("\nPlot A: D_ell vs ell...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

for z_t, color in zip(z_targets, COLORS):
    ell = raw_all[ref_seed][z_t]['ell']
    D_mean, D_std = seed_average(raw_all, z_t, 'D_cross_muK_mK')
    r_mean, r_std = seed_average(raw_all, z_t, 'r_cross')
    if D_mean is None:
        continue
    valid = np.isfinite(D_mean) & (ell > 0)
    axes[0].plot(ell[valid], D_mean[valid], color=color, lw=2,
                marker='o', markersize=4, label=f'z={z_t:.1f}')
    axes[0].fill_between(ell[valid], (D_mean - D_std)[valid], (D_mean + D_std)[valid],
                         color=color, alpha=0.15)
    if r_mean is not None:
        valid_r = np.isfinite(r_mean) & (ell > 0)
        axes[1].plot(ell[valid_r], r_mean[valid_r], color=color, lw=2,
                    marker='o', markersize=4, label=f'z={z_t:.1f}')
        axes[1].fill_between(ell[valid_r], (r_mean - r_std)[valid_r], (r_mean + r_std)[valid_r],
                             color=color, alpha=0.15)

axes[0].axhline(0, color='gray', ls='--', lw=1)
axes[0].set_xscale('log')
axes[0].set_yscale('symlog', linthresh=1e-6)
axes[0].set_xlabel(r'Multipole $\ell$')
axes[0].set_ylabel(r'$\ell(\ell+1)C_\ell^{\rm kSZ\times 21cm}/2\pi\;[\mu{\rm K}\cdot{\rm mK}]$')
axes[0].set_title(r'kSZ$\times$21cm (raw, neither squared): $D_\ell$ vs $\ell$')
axes[0].legend(fontsize=10)

axes[1].axhline(0, color='gray', ls='--', lw=1)
axes[1].set_xscale('log')
axes[1].set_ylim(-1.05, 1.05)
axes[1].set_xlabel(r'Multipole $\ell$')
axes[1].set_ylabel(r'Correlation coefficient $r = C_\ell^{\rm cross}/\sqrt{P_{\rm kSZ}P_{\rm 21cm}}$')
axes[1].set_title(r'Is there any correlation at all? (seed-mean $r_\ell$)')
axes[1].legend(fontsize=10)

fig.savefig(f"{OUT_DIR}/A_Dell_and_r_vs_ell.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: A_Dell_and_r_vs_ell.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT B: D_ell vs z, for each ell_target — D_ell and r_cross side by side
# ─────────────────────────────────────────────────────────────────────────────
print("Plot B: D_ell vs z...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

for ell_t, color in zip(ELL_TARGETS, COLORS):
    z_vals, D_vals, D_errs, r_vals, r_errs = [], [], [], [], []
    for z0 in z_list:
        ell_arr = raw_all[ref_seed][z0]['ell']
        D_m, D_s = seed_average(raw_all, z0, 'D_cross_muK_mK')
        r_m, r_s = seed_average(raw_all, z0, 'r_cross')
        if D_m is None:
            continue
        idx = int(np.argmin(np.abs(ell_arr - ell_t)))
        if abs(ell_arr[idx] - ell_t) > 0.3 * ell_t or not np.isfinite(D_m[idx]):
            continue
        z_vals.append(z0)
        D_vals.append(D_m[idx]); D_errs.append(D_s[idx])
        # r_m/r_s can independently be None (or NaN at this idx) even when
        # D_m is valid — seed_average() runs on 'D_cross_muK_mK' and
        # 'r_cross' separately, and it's possible for one to have no
        # finite seeds at a given z0 while the other does. Guard both.
        if r_m is not None and np.isfinite(r_m[idx]):
            r_vals.append(r_m[idx])
            r_errs.append(r_s[idx] if (r_s is not None and np.isfinite(r_s[idx])) else 0.0)
        else:
            r_vals.append(np.nan)
            r_errs.append(0.0)

    if not z_vals:
        continue
    z_arr = np.array(z_vals)
    sort  = np.argsort(z_arr)[::-1]   # high z first, matches invert_xaxis convention
    D_arr, E_arr = np.array(D_vals)[sort], np.array(D_errs)[sort]
    r_arr, re_arr = np.array(r_vals)[sort], np.array(r_errs)[sort]
    z_sorted = z_arr[sort]

    axes[0].plot(z_sorted, D_arr, color=color, lw=2, marker='o', markersize=4,
                label=rf'$\ell={ell_t}$')
    axes[0].fill_between(z_sorted, D_arr - E_arr, D_arr + E_arr, color=color, alpha=0.15)
    axes[1].plot(z_sorted, r_arr, color=color, lw=2, marker='o', markersize=4,
                label=rf'$\ell={ell_t}$')
    axes[1].fill_between(z_sorted, r_arr - re_arr, r_arr + re_arr, color=color, alpha=0.15)

for ax in axes:
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.invert_xaxis()
    ax.set_xlabel(r'Redshift $z$')

axes[0].set_yscale('symlog', linthresh=1e-6)
axes[0].set_ylabel(r'$\ell(\ell+1)C_\ell^{\rm kSZ\times 21cm}/2\pi\;[\mu{\rm K}\cdot{\rm mK}]$')
axes[0].set_title(r'kSZ$\times$21cm (raw): $D_\ell$ vs $z$ at fixed $\ell$')
axes[0].legend(fontsize=10)

axes[1].set_ylim(-1.05, 1.05)
axes[1].set_ylabel(r'Correlation coefficient $r_\ell$')
axes[1].set_title(r'Is there any correlation at all? (vs $z$)')
axes[1].legend(fontsize=10)

fig.savefig(f"{OUT_DIR}/B_Dell_and_r_vs_z.png", dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved: B_Dell_and_r_vs_z.png")

print(f"\nAll plots saved to:\n  {OUT_DIR}/")
print("\nReminder: this is a brand-new, never-yet-run statistic. Look at the")
print("r_cross panels FIRST — they answer 'is there any correlation at all'")
print("directly, without needing to trust the D_ell physical-unit conversion.")
print("If r_cross is consistent with zero everywhere within its seed-scatter")
print("error band, that's evidence against Prof. Kulkarni's hypothesis, not")
print("a bug — a null result here is itself a real, useful answer.")
