#!/usr/bin/env python
# =============================================================================
# 21b_hii_eff_summary.py
#
# Reads every heff<value>_seed<S>.npz from scripts/21 and
#   * prints, per HII_EFF_FACTOR, the milestones of the volume-weighted history:
#       z_start = z(x_e = 1e-4)      (99.99% neutral: the HIGH-z edge of our patchy window)
#       z10, z50, z90
#       z_end   = z(x_e = 1 - 1e-4)  (0.01% neutral: the LOW-z edge of our patchy window)
#     plus z50 of the mass-weighted history (AMBER's convention) and the duration z10 - z90;
#   * interpolates z50 against ln(HII_EFF) and prints the HII_EFF that hits each --target z50;
#   * plots x_e(z) for every value with the milestones marked (left) and z50 vs HII_EFF with the
#     targets (right): hii_eff_calibration.{png,pdf} in --in-dir.
# A milestone outside the simulated redshifts is reported as "<zmin" / ">zmax" (then the production
# snapshot list must be extended there).
#
#   python scripts/21b_hii_eff_summary.py --in-dir $R/ksz2-21cm-data/hii_eff_calib
# =============================================================================

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--in-dir", required=True)
ap.add_argument("--targets", type=float, nargs="+", default=[6.68, 7.71, 8.75],
                help="target volume-weighted z50 (default: the three AMBER histories)")
args = ap.parse_args()

XE_START, XE_END = 1e-4, 1.0 - 1e-4


def z_at(z, xe, level, log_from=None):
    """Redshift where x_e(z) crosses `level` (x_e decreases with z). log_from='low': interpolate
    log10(x_e) (for small levels); 'high': log10(1 - x_e) (for levels near 1). NaN if never crossed."""
    z, xe = np.asarray(z, float), np.asarray(xe, float)
    o = np.argsort(z)
    z, xe = z[o], xe[o]
    if log_from == "low":
        y, lv = np.log10(np.clip(xe, 1e-30, None)), np.log10(level)
    elif log_from == "high":
        y, lv = np.log10(np.clip(1.0 - xe, 1e-30, None)), np.log10(1.0 - level)
    else:
        y, lv = xe, level
    s = np.sign(y - lv)
    for i in range(len(z) - 1):
        if s[i] == 0:
            return float(z[i])
        if s[i] != s[i + 1]:
            return float(z[i] + (lv - y[i]) * (z[i + 1] - z[i]) / (y[i + 1] - y[i]))
    return np.nan


runs = []
for p in sorted(glob.glob(os.path.join(args.in_dir, "heff*_seed*.npz"))):
    d = np.load(p)
    z, xe, xm = d["z"], d["x_e"], d["x_e_mass"]
    r = {"heff": float(d["hii_eff"]), "seed": int(d["seed"]), "z": z, "xe": xe,
         "z_start": z_at(z, xe, XE_START, "low"), "z10": z_at(z, xe, 0.1), "z50": z_at(z, xe, 0.5),
         "z90": z_at(z, xe, 0.9), "z_end": z_at(z, xe, XE_END, "high"), "z50_mass": z_at(z, xm, 0.5)}
    r["edge_lo"] = bool(xe[np.argmin(z)] < XE_END)        # still not fully ionized at the lowest z
    r["edge_hi"] = bool(xe[np.argmax(z)] > XE_START)      # already ionizing at the highest z
    runs.append(r)
assert runs, f"no heff*_seed*.npz in {args.in_dir}"
runs.sort(key=lambda r: r["heff"])


def fmt(v, r, which):
    if np.isfinite(v):
        return f"{v:6.2f}"
    if which == "end" and r["edge_lo"]:
        return f"<{r['z'].min():4.1f}"
    if which == "start" and r["edge_hi"]:
        return f">{r['z'].max():4.1f}"
    return "   nan"


print(f"{'HII_EFF':>8} {'seed':>4} {'z_start':>7} {'z10':>6} {'z50':>6} {'z90':>6} {'z_end':>6} {'dz10-90':>7} {'z50_mass':>8}")
for r in runs:
    print(f"{r['heff']:8g} {r['seed']:4d} {fmt(r['z_start'], r, 'start'):>7} {fmt(r['z10'], r, ''):>6} {fmt(r['z50'], r, ''):>6} "
          f"{fmt(r['z90'], r, ''):>6} {fmt(r['z_end'], r, 'end'):>6} {r['z10'] - r['z90']:7.2f} {r['z50_mass']:8.2f}")
print("z_start / z_end = x_e of 1e-4 / 1 - 1e-4: the edges of the patchy window (99.99% / 0.01% neutral).")

ok = [r for r in runs if np.isfinite(r["z50"])]
choice = {}
if len(ok) >= 2:
    lnh = np.log([r["heff"] for r in ok])
    z50 = np.array([r["z50"] for r in ok])
    o = np.argsort(z50)
    for t in args.targets:
        if z50.min() <= t <= z50.max():
            h = float(np.exp(np.interp(t, z50[o], lnh[o])))
            choice[t] = h
            print(f"target z50 = {t:5.2f}  ->  HII_EFF_FACTOR ~ {h:6.2f}   (log-interpolated; check with one more run)")
        else:
            print(f"target z50 = {t:5.2f}  outside the scanned range z50 = {z50.min():.2f}..{z50.max():.2f}: extend the scan")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.6, 1]})
cols = plt.cm.plasma(np.linspace(0.05, 0.85, len(runs)))
for c, r in zip(cols, runs):
    a1.plot(r["z"], r["xe"], "-", color=c, lw=1.6, label=f"HII_EFF={r['heff']:g} (z50={r['z50']:.2f})")
    for key, mk in (("z_start", "v"), ("z50", "o"), ("z_end", "^")):
        if np.isfinite(r[key]):
            lev = {"z_start": XE_START, "z50": 0.5, "z_end": XE_END}[key]
            a1.plot(r[key], lev, mk, color=c, ms=6)
for lev in (XE_START, 0.5, XE_END):
    a1.axhline(lev, color="gray", ls=":", lw=0.8)
a1.set_xlabel("z")
a1.set_ylabel(r"$\bar{x}_e$ (volume-weighted)")
a1.set_title(r"21cmFAST histories (seed 1, 256$^3$): $\nabla$ $z_{\rm start}$ ($x_e=10^{-4}$), $\circ$ $z_{50}$, $\Delta$ $z_{\rm end}$ ($x_e=1-10^{-4}$)", fontsize=9)
a1.legend(fontsize=8, frameon=False)
a1.set_xlim(max(r["z"].max() for r in runs), min(r["z"].min() for r in runs))
a2.semilogx([r["heff"] for r in runs], [r["z50"] for r in runs], "o-", color="k", label=r"$z_{50}$ (volume)")
a2.semilogx([r["heff"] for r in runs], [r["z50_mass"] for r in runs], "s--", color="gray", label=r"$z_{50}$ (mass)")
for t in args.targets:
    a2.axhline(t, color="C3", ls=":", lw=1)
    if t in choice:
        a2.axvline(choice[t], color="C3", ls=":", lw=1)
        a2.text(choice[t], t, f"  {choice[t]:.1f}", color="C3", fontsize=9, va="bottom")
a2.set_xlabel("HII_EFF_FACTOR")
a2.set_ylabel(r"$z_{50}$")
a2.set_title("targets (dotted): AMBER histories", fontsize=10)
a2.legend(fontsize=9, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(args.in_dir, f"hii_eff_calibration.{ext}"), dpi=150, bbox_inches="tight")
print(f"saved {os.path.join(args.in_dir, 'hii_eff_calibration.png')}")
