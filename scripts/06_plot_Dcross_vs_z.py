#!/usr/bin/env python
# =============================================================================
# 06_plot_Dcross_vs_z.py
#
# MAIN PLOT: D_ell^{kSZ^2 x 21cm^2} vs chunk-centre redshift z0, at fixed ell,
# with the volume-averaged ionized fraction xbar_HII on the top axis --
# the analogue of Zhou+25 Fig. 5 / Ma+18 Fig. 3 (bottom), built from the
# stitched pathway's cross_corr_sq_stitched_seed*.npy.
#
# Reads (per seed):
#   <cache_dir>/seed_<N>/cross_corr_sq_stitched_seed<N>.npy   (script 04)
#   <cache_dir>/seed_<N>/stitched_lightcone_seed<N>.npz       (script 01; for xbar_HII(z))
#
# UNITS: the kSZ map is dimensionless dT/T and the 21cm field is in mK, so the
# raw D_cross is in (dT/T)^2 mK^2. By default this script converts to muK^4
# (kSZ^2 -> x T_CMB[muK]^2, 21cm^2 -> x 1e6 muK^2/mK^2) to match Zhou+25's
# axis. Use --raw-units to skip the conversion.
#
# THE ZHOU+25 OVERLAY (dashed, RIGHT-hand axis, zeros aligned) IS NOT LIKE-FOR-LIKE:
# their Fig. 5 uses the HERA noise filter on 21cm and the SO ILC filter on
# kSZ. Our pipeline applies only the k_par > k_par_min cut. Compare SIGN
# STRUCTURE and zero-crossing / extremum redshifts first; amplitude only
# after the filters are matched.
#
# ell values above ~2000 are beyond the resolution-validated range at 256^3
# (ksz-pipeline convergence work) and are flagged on the plot.
#
# Run:
#   python scripts/06_plot_Dcross_vs_z.py --config configs/variants/stitched_256.yaml
#   python scripts/06_plot_Dcross_vs_z.py --config ... --ells 400 800 1600 --no-zhou
# =============================================================================

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN slices (ell outside a chunk's range)

from ksz2_21cm.utils.config import load_config
from ksz2_21cm.utils.constants import T_CMB_K

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/variants/stitched_256.yaml")
parser.add_argument("--ells", type=float, nargs="+", default=[400, 800, 1600])
parser.add_argument("--no-zhou", action="store_true", help="skip the Zhou+25 Fig.5 overlay")
parser.add_argument("--raw-units", action="store_true", help="keep (dT/T)^2 mK^2, no muK^4 conversion")
parser.add_argument("--name", default="Dcross_vs_z_stitched")
parser.add_argument("--chi-norm", choices=["none", "z0"], default="none",
                    help="'z0': divide D by chi(z0)^2 [Mpc^2] per chunk (plot-side only; the "
                         "kSZ code and workers are untouched). Default 'none' = old behaviour.")
parser.add_argument("--t21-unit", choices=["mK", "uK"], default="mK",
                    help="unit of the stored brightness_temp field: mK (21cmFAST native, default) or uK")
args = parser.parse_args()

cfg = load_config(args.config)
cache_dir = cfg["paths"]["cache_dir"]
plot_dir = cfg["paths"]["plot_dir"]
os.makedirs(plot_dir, exist_ok=True)
seeds = cfg["simulation"]["random_seeds"]
ELLS = list(args.ells)

_T21_SQ = 1e6 if args.t21_unit == "mK" else 1.0          # (T21 field)^2 -> muK^2
UNIT_FAC = 1.0 if args.raw_units else (T_CMB_K * 1e6) ** 2 * _T21_SQ
UNIT_LABEL = (r"$(\Delta T/T)^2\,\mathrm{mK}^2$" if args.raw_units else r"$\mu\mathrm{K}^4$")
if args.chi_norm == "z0":
    UNIT_LABEL += r"$/\chi^2$"

# ── Load per-seed results ─────────────────────────────────────────────────────
per_seed = {}      # seed -> {z0: result_dict}
xhii_curves = {}   # seed -> (z_arr, xHII(z))
for seed in seeds:
    p_cc = os.path.join(cache_dir, f"seed_{seed}", f"cross_corr_sq_stitched_seed{seed}.npy")
    p_lc = os.path.join(cache_dir, f"seed_{seed}", f"stitched_lightcone_seed{seed}.npz")
    if not (os.path.exists(p_cc) and os.path.exists(p_lc)):
        print(f"  seed {seed}: missing cross-corr or stitched lightcone -- skipping")
        continue
    per_seed[seed] = np.load(p_cc, allow_pickle=True).item()
    d = np.load(p_lc)
    z_arr = np.asarray(d["z_arr"], dtype=float)          # ascending (build_los_z_grid contract)
    xhii = 1.0 - np.asarray(d["xH_box"]).mean(axis=(0, 1))
    xhii_curves[seed] = (z_arr, xhii)
assert per_seed, "No seeds with both cross-corr and stitched lightcone found -- run 01/02/04 first."
print(f"Seeds used: {sorted(per_seed)}")

z0_list = sorted(set.intersection(*[set(r.keys()) for r in per_seed.values()]))
print(f"Chunk centres z0: {z0_list}")

# ── D_cross at fixed ell, per seed per z0 ─────────────────────────────────────
def d_at_ell(res, ell_target):
    ell = np.asarray(res["ell"], dtype=float)
    D = np.asarray(res["D_cross"], dtype=float)
    ok = np.isfinite(ell) & np.isfinite(D)
    if ok.sum() < 2 or not (ell[ok].min() <= ell_target <= ell[ok].max()):
        return np.nan
    order = np.argsort(ell[ok])                          # np.interp needs ascending xp
    return float(np.interp(ell_target, ell[ok][order], D[ok][order]))

seeds_used = sorted(per_seed)
if args.chi_norm == "z0":
    from ksz2_21cm.utils.cosmology import get_cosmology
    _cosmo = get_cosmology(cfg)
    CHI2 = {z0: float(_cosmo.comoving_distance(z0).value) ** 2 for z0 in z0_list}
    print("chi-norm z0: dividing by chi(z0)^2, e.g. chi^2(z0=%.1f) = %.3e Mpc^2" % (z0_list[0], CHI2[z0_list[0]]))
else:
    CHI2 = {z0: 1.0 for z0 in z0_list}
table = {L: np.full((len(seeds_used), len(z0_list)), np.nan) for L in ELLS}
for i, s in enumerate(seeds_used):
    for j, z0 in enumerate(z0_list):
        for L in ELLS:
            table[L][i, j] = d_at_ell(per_seed[s][z0], L) * UNIT_FAC / CHI2[z0]

# ── xbar_HII at each z0 (mean over seeds) ─────────────────────────────────────
xhii_at_z0 = np.array([
    np.mean([np.interp(z0, *xhii_curves[s]) for s in seeds_used])   # z_arr ascending: fine
    for z0 in z0_list
])

# ── Text summary ──────────────────────────────────────────────────────────────
print(f"\nUnits: {'raw (dT/T)^2 mK^2' if args.raw_units else 'muK^4'}   (N seeds = {len(seeds_used)})")
ZHOU_REF = None
if not args.no_zhou:
    from ksz2_21cm.plotting.zhou25_data import ZHOU_FIG5 as ZHOU_REF
for L in ELLS:
    m = np.nanmean(table[L], axis=0)
    nn = np.sum(np.isfinite(table[L]), axis=0)
    se = np.nanstd(table[L], axis=0, ddof=1) / np.sqrt(np.maximum(nn, 1))
    print(f"\n  ell = {L:g}:   (mean +/- s.e.m.,  mean/sem = significance)")
    for z0, x, v, e in zip(z0_list, xhii_at_z0, m, se):
        sig = v / e if e > 0 else np.nan
        print(f"    z0={z0:5.2f}  xHII={x:5.3f}  D = {v: .4e} +/- {e:.2e}   {sig:+6.1f} sigma")
    ok = np.isfinite(m)
    if ok.sum() >= 2:
        sgn = np.sign(m[ok]); flips = np.where(np.diff(sgn) != 0)[0]
        zok = np.array(z0_list)[ok]
        xok = xhii_at_z0[ok]
        for f in flips:
            # linear zero-crossing between adjacent chunk centres
            za, zb, ma, mb = zok[f], zok[f + 1], m[ok][f], m[ok][f + 1]
            zc = za + (0 - ma) * (zb - za) / (mb - ma)
            print(f"    sign change between z0={za} and {zb}: z_cross ~ {zc:.2f}  (xHII ~ {np.interp(zc, zok, xok):.2f})")
        imax, imin = np.argmax(m[ok]), np.argmin(m[ok])
        print(f"    max at z0={zok[imax]} (xHII={xok[imax]:.2f}),  min at z0={zok[imin]} (xHII={xok[imin]:.2f})")
        print(f"    negative-lobe depth / positive peak = {m[ok][imin] / m[ok][imax]:+.2f}")
        if ZHOU_REF is not None and int(L) in ZHOU_REF:
            zz = ZHOU_REF[int(L)]
            print(f"    [Zhou+25 digitized]  max at z={zz[np.argmax(zz[:,1]),0]:.2f},  min at z={zz[np.argmin(zz[:,1]),0]:.2f},  "
                  f"min/max = {zz[:,1].min() / zz[:,1].max():+.2f}")

# Save the numbers so replotting never needs the big caches again
np.savez(os.path.join(plot_dir, f"{args.name}_data.npz"),
         z0=np.array(z0_list), chi_norm=args.chi_norm, t21_unit=args.t21_unit, xhii=xhii_at_z0, ells=np.array(ELLS),
         seeds=np.array(seeds_used), unit_fac=UNIT_FAC,
         **{f"D_ell{int(L)}": table[L] for L in ELLS})

# ── Plot ──────────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from ksz2_21cm.plotting.style import apply_global_style
apply_global_style()
plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "legend.fontsize": 11, "axes.titlesize": 14})

zhou = None
if not args.no_zhou:
    from ksz2_21cm.plotting.zhou25_data import ZHOU_FIG5
    zhou = ZHOU_FIG5

n = len(ELLS)
fig, axes = plt.subplots(n, 1, figsize=(8.5, 3.4 * n), sharex=True)
axes = np.atleast_1d(axes)
z0a = np.array(z0_list)

for ax, L in zip(axes, ELLS):
    T = table[L]
    mean = np.nanmean(T, axis=0)
    sem = (np.nanstd(T, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(T), axis=0))
           if len(seeds_used) > 1 else np.zeros_like(mean))
    ax.fill_between(z0a, mean - sem, mean + sem, color="C0", alpha=0.25, lw=0)
    ax.plot(z0a, mean, "-o", color="C0", ms=4, label=f"stitched, {len(seeds_used)} seed(s) (mean ± s.e.m.)")
    zhou_handle = None
    if zhou is not None and int(L) in zhou:
        # Zhou+25 on its OWN right-hand axis (different filters/amplitude, so a
        # shared axis would hide one curve). Both axes get symmetric limits so
        # the zeros line up and only SHAPE / SIGN STRUCTURE is being compared.
        zz = zhou[int(L)]
        axr = ax.twinx()
        zhou_handle, = axr.plot(zz[:, 0], zz[:, 1], "--", color="k", lw=1.5,
                                label="Zhou+25 Fig.5, right axis (HERA+SO filters)")
        ymax_r = 1.1 * np.max(np.abs(zz[:, 1]))
        axr.set_ylim(-ymax_r, ymax_r)
        axr.set_ylabel(r"Zhou+25 $D_\ell$ [$\mu\mathrm{K}^4$]", fontsize=12)
        ymax_l = 1.1 * np.nanmax(np.abs(np.concatenate([mean - sem, mean + sem])))
        if np.isfinite(ymax_l) and ymax_l > 0:
            ax.set_ylim(-ymax_l, ymax_l)
    ax.axhline(0, color="gray", lw=0.8)
    flag = "  (beyond validated ell≲2000)" if L > 2000 else ""
    ax.text(0.02, 0.93, rf"$\ell = {L:g}${flag}", transform=ax.transAxes, va="top", fontsize=13)
    ax.set_ylabel(rf"$D_\ell$ [{UNIT_LABEL}]")
    hs, ls = ax.get_legend_handles_labels()
    if zhou_handle is not None:
        hs.append(zhou_handle); ls.append(zhou_handle.get_label())
    ax.legend(hs, ls, loc="best", frameon=False)

axes[-1].set_xlabel(r"Chunk-centre redshift $z_0$")
for ax in axes:
    ax.invert_xaxis()
xlim = axes[0].get_xlim()

# top axis: xbar_HII at selected z0 values
top = axes[0].twiny()
top.set_xlim(xlim)
tick_z = [z for z in z0_list if abs(z * 2 - round(z * 2)) < 1e-9 and (z * 2) % 2 == 0]   # integer z0 only
top.set_xticks(tick_z)
top.set_xticklabels([f"{xhii_at_z0[z0_list.index(z)]:.2f}" for z in tick_z])
top.set_xlabel(r"$\bar{x}_{\rm HII}$")
top.tick_params(top=True, labeltop=True)

fig.suptitle(r"$D_\ell^{\mathrm{kSZ}^2\times 21\mathrm{cm}^2}$ vs $z_0$ — stitched pipeline", y=1.0)
fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(plot_dir, f"{args.name}.{ext}")
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ Saved: {out}")
