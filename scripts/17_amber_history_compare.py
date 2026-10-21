#!/usr/bin/env python
# =============================================================================
# 17_amber_history_compare.py
#
# How does D_ell^{kSZ^2 x 21cm^2} (Zhou-style, k_par > k_par_min filter) change with the reionization history?
# Reads the AMBER production caches written by slurm/amber_production.sbatch:
#   <data-root>/amber_q512_h<ZMID>/cache/seed_<N>/cross_corr_sq_stitched_seed<N>.npy     (script 04)
#   <data-root>/amber_q512_h<ZMID>/cache/seed_<N>/stitched_lightcone_seed<N>.npz          (script 01a; for z(x_e=0.5))
# and, per history, averages over seeds exactly the way 06_plot_Dcross_vs_z.py does (same D_ell at fixed ell,
# same unit conversion to muK^4, same Zhou+25 overlay on its own right-hand axis). It makes
#   amber_history_Dcross_vs_z.{pdf,png}          D_ell vs z0 for every history, one row per ell
#   amber_history_Dcross_vs_z_shifted.{pdf,png}  the same against z0 - z(x_e=0.5) of that history (does the shape
#                                                just slide with the history?)
#   amber_history_data.npz, and a text summary (peak / trough / zero-crossing redshifts, amplitudes).
# Errors are s.e.m. over seeds. The Zhou+25 curve is NOT like for like (HERA + SO filters); compare shape first.
# x_e here is volume-weighted; AMBER's z_mid is mass-weighted, so z(x_e=0.5) is printed for every history.
#
#   python scripts/17_amber_history_compare.py [--data-root $R/ksz2-21cm-data] [--ells 400 800 1600] [--no-zhou]
# =============================================================================
import argparse
import glob
import os
import re
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np

from ksz2_21cm.utils.constants import T_CMB_K

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default=os.path.expanduser("~/rds/rds-dirac-dp145-uYnnMXSp128/dc-upad3/ksz2-21cm-data"))
ap.add_argument("--out-dir", default=None, help="default: <data-root>/amber_history_compare")
ap.add_argument("--ells", type=float, nargs="+", default=[400, 800, 1600])
ap.add_argument("--no-zhou", action="store_true")
ap.add_argument("--raw-units", action="store_true", help="keep (dT/T)^2 mK^2, no muK^4 conversion")
ap.add_argument("--chi-norm", choices=["none", "z0"], default="z0",
                help="'z0' (default): divide D by chi(z0)^2 [Mpc^2] per chunk, exactly as 06 --chi-norm z0 and 08 do "
                     "(plot-side only; the kSZ code is untouched). 'none' = raw, the numbers printed before this option existed.")
ap.add_argument("--name", default="amber_history")
a = ap.parse_args()

out_dir = a.out_dir or os.path.join(a.data_root, "amber_history_compare")
os.makedirs(out_dir, exist_ok=True)
ELLS = list(a.ells)
UNIT_FAC = 1.0 if a.raw_units else (T_CMB_K * 1e6) ** 2 * 1e6
UNIT_LABEL = r"$(\Delta T/T)^2\,\mathrm{mK}^2$" if a.raw_units else r"$\mu\mathrm{K}^4$"
if a.chi_norm == "z0":
    UNIT_LABEL += r"$/\chi^2$"

hist_dirs = sorted(glob.glob(os.path.join(a.data_root, "amber_q512_h*")),
                   key=lambda p: float(re.findall(r"_h([0-9.]+)$", p)[0]))
assert hist_dirs, f"no amber_q512_h* folders under {a.data_root}"


def d_at_ell(res, ell_target):
    ell = np.asarray(res["ell"], dtype=float)
    D = np.asarray(res["D_cross"], dtype=float)
    ok = np.isfinite(ell) & np.isfinite(D)
    if ok.sum() < 2 or not (ell[ok].min() <= ell_target <= ell[ok].max()):
        return np.nan
    o = np.argsort(ell[ok])
    return float(np.interp(ell_target, ell[ok][o], D[ok][o]))


H = []                                   # one dict per history
for hd in hist_dirs:
    zm = float(re.findall(r"_h([0-9.]+)$", hd)[0])
    cache = os.path.join(hd, "cache")
    per_seed, z50s = {}, []
    for sd in sorted(glob.glob(os.path.join(cache, "seed_*"))):
        s = int(re.findall(r"seed_(\d+)$", sd)[0])
        p_cc = os.path.join(sd, f"cross_corr_sq_stitched_seed{s}.npy")
        p_lc = os.path.join(sd, f"stitched_lightcone_seed{s}.npz")
        if not (os.path.exists(p_cc) and os.path.exists(p_lc)):
            continue
        per_seed[s] = np.load(p_cc, allow_pickle=True).item()
        d = np.load(p_lc)
        z = np.asarray(d["z_arr"], float)
        xe = 1.0 - np.asarray(d["xH_box"]).mean(axis=(0, 1))
        z50s.append(float(np.interp(0.5, xe[::-1], z[::-1])))
    if not per_seed:
        print(f"z_mid={zm:g}: no complete seeds, skipped")
        continue
    H.append(dict(zm=zm, seeds=sorted(per_seed), res=per_seed, z50=float(np.mean(z50s))))
    print(f"z_mid={zm:g}: seeds {sorted(per_seed)}  z(x_e=0.5) = {np.mean(z50s):.2f} +- {np.std(z50s):.2f}")
assert H, "nothing to compare"

z0_list = sorted(set.intersection(*[set(r.keys()) for h in H for r in h["res"].values()]))
z0a = np.array(z0_list)
print(f"chunk centres z0 in common: {z0_list}")

if a.chi_norm == "z0":
    from ksz2_21cm.utils.cosmology import get_cosmology
    _cosmo = get_cosmology()
    CHI2 = np.array([float(_cosmo.comoving_distance(z0).value) ** 2 for z0 in z0_list])
    print("chi-norm z0: dividing by chi(z0)^2 [Mpc^2]: " + ", ".join(f"z0={z:g}: {c:.3e}" for z, c in zip(z0_list, CHI2)))
else:
    CHI2 = np.ones(len(z0_list))

for h in H:
    h["table"] = {L: np.array([[d_at_ell(h["res"][s][z0], L) * UNIT_FAC / CHI2[j] for j, z0 in enumerate(z0_list)] for s in h["seeds"]])
                  for L in ELLS}
    h["mean"] = {L: np.nanmean(h["table"][L], axis=0) for L in ELLS}
    h["sem"] = {L: (np.nanstd(h["table"][L], axis=0, ddof=1) / np.sqrt(np.maximum(np.sum(np.isfinite(h["table"][L]), axis=0), 1))
                    if len(h["seeds"]) > 1 else np.zeros(len(z0_list))) for L in ELLS}

# ---- text summary ----
print(f"\nUnits: {'raw' if a.raw_units else 'muK^4'}{' / Mpc^2 (chi-norm z0)' if a.chi_norm == 'z0' else ''}")
for L in ELLS:
    print(f"\n  ell = {L:g}")
    print(f"    {'z_mid':>5} {'N':>2} {'z50':>5}  {'peak z0':>7} {'peak D':>9} {'peak-z50':>8}   {'trough z0':>9} {'trough D':>9}   zero-crossings")
    for h in H:
        m = h["mean"][L]
        ok = np.isfinite(m)
        if ok.sum() < 2:
            print(f"    {h['zm']:5g} {len(h['seeds']):2d} {h['z50']:5.2f}  (no data at this ell)")
            continue
        zk, mk = z0a[ok], m[ok]
        ip, it = int(np.argmax(mk)), int(np.argmin(mk))
        zc = [zk[i] + (0 - mk[i]) * (zk[i + 1] - zk[i]) / (mk[i + 1] - mk[i]) for i in np.where(np.diff(np.sign(mk)) != 0)[0]]
        print(f"    {h['zm']:5g} {len(h['seeds']):2d} {h['z50']:5.2f}  {zk[ip]:7.2f} {mk[ip]:9.3g} {zk[ip] - h['z50']:8.2f}   "
              f"{zk[it]:9.2f} {mk[it]:9.3g}   {', '.join(f'{c:.2f}' for c in zc) or '-'}")

np.savez(os.path.join(out_dir, f"{a.name}_data.npz"), z0=z0a, ells=np.array(ELLS), unit_fac=UNIT_FAC,
         chi_norm=a.chi_norm, chi2=CHI2,
         zmid=np.array([h["zm"] for h in H]), z50=np.array([h["z50"] for h in H]),
         **{f"mean_h{h['zm']:g}_ell{int(L)}": h["mean"][L] for h in H for L in ELLS},
         **{f"sem_h{h['zm']:g}_ell{int(L)}": h["sem"][L] for h in H for L in ELLS},
         **{f"table_h{h['zm']:g}_ell{int(L)}": h["table"][L] for h in H for L in ELLS})

# ---- plots ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ksz2_21cm.plotting.style import apply_global_style
apply_global_style()
plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "legend.fontsize": 11, "axes.titlesize": 14})
zhou = None
if not a.no_zhou:
    from ksz2_21cm.plotting.zhou25_data import ZHOU_FIG5
    zhou = ZHOU_FIG5
colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(H)))


def make(shifted):
    n = len(ELLS)
    fig, axes = plt.subplots(n, 1, figsize=(8.5, 3.4 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, L in zip(axes, ELLS):
        ymax = 0.0
        for h, c in zip(H, colors):
            x = z0a - (h["z50"] if shifted else 0.0)
            m, e = h["mean"][L], h["sem"][L]
            ax.fill_between(x, m - e, m + e, color=c, alpha=0.2, lw=0)
            ax.plot(x, m, "-o", color=c, ms=4, label=rf"AMBER $z_{{\rm mid}}={h['zm']:g}$ ($z_{{50}}={h['z50']:.2f}$, N={len(h['seeds'])})")
            ymax = max(ymax, np.nanmax(np.abs(np.concatenate([m - e, m + e]))))
        handle = None
        if zhou is not None and not shifted and int(L) in zhou:
            zz = zhou[int(L)]
            axr = ax.twinx()
            handle, = axr.plot(zz[:, 0], zz[:, 1], "--", color="k", lw=1.5, label="Zhou+25 Fig.5, right axis (HERA+SO filters)")
            yr = 1.1 * np.max(np.abs(zz[:, 1]))
            axr.set_ylim(-yr, yr)
            axr.set_ylabel(r"Zhou+25 $D_\ell$ [$\mu\mathrm{K}^4$]", fontsize=12)
        if np.isfinite(ymax) and ymax > 0:
            ax.set_ylim(-1.1 * ymax, 1.1 * ymax)
        ax.axhline(0, color="gray", lw=0.8)
        if shifted:
            ax.axvline(0, color="gray", lw=0.8, ls=":")
        ax.text(0.02, 0.93, rf"$\ell = {L:g}$", transform=ax.transAxes, va="top", fontsize=13)
        ax.set_ylabel(rf"$D_\ell$ [{UNIT_LABEL}]")
        hs, ls = ax.get_legend_handles_labels()
        if handle is not None:
            hs.append(handle); ls.append(handle.get_label())
        if L == ELLS[0]:
            ax.legend(hs, ls, loc="best", frameon=False)
    axes[-1].set_xlabel(r"$z_0 - z(\bar{x}_e = 0.5)$ of that history" if shifted else r"Chunk-centre redshift $z_0$")
    if not shifted:
        for ax in axes:
            ax.invert_xaxis()
    fig.suptitle(r"$D_\ell^{\mathrm{kSZ}^2\times 21\mathrm{cm}^2}$ for different reionization histories (AMBER)", y=1.0)
    fig.tight_layout()
    stem = f"{a.name}_Dcross_vs_z" + ("_shifted" if shifted else "")
    for ext in ("pdf", "png"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight")
        print(f"saved {p}")
    plt.close(fig)


make(False)
make(True)
