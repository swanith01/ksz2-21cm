#!/usr/bin/env python
# =============================================================================
# 11_plot_auto21.py
#
# 21cm AUTO power spectrum Delta^2_21(k) = k^3 P/2pi^2 [mK^2] from 04d: one panel per redshift
# (N seeds thin, mean thick, median dashed, 16-84% band), with the "neutral limit"
# T0^2 Delta^2_delta (matter power of the same slab x the fully-neutral mean brightness
# temperature squared) as a reference curve: where Delta^2_21 falls below it, ionized bubbles
# are removing power (small k can exceed it when bubbles add large-scale contrast).
# The last panel: Delta^2_21 vs z at fixed k (the familiar mid-reionization bump) with
# xbar_e on the top axis.
#
# Run after 04d (light: reads small caches):
#   python scripts/11_plot_auto21.py --config configs/variants/stitched_256.yaml
# =============================================================================

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ksz2_21cm.utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/variants/stitched_256.yaml")
parser.add_argument("--n-seeds", type=int, default=10)
parser.add_argument("--z0", type=float, nargs="+", default=[6, 7, 8, 9, 10, 11, 12])
parser.add_argument("--slab-dz", type=float, default=0.4)
parser.add_argument("--kmin", type=float, default=0.02)
parser.add_argument("--kmax", type=float, default=1.0)
parser.add_argument("--nk", type=int, default=24)
parser.add_argument("--k-fixed", type=float, nargs="+", default=[0.1, 0.2, 0.5], help="k [1/Mpc] for the last panel")
parser.add_argument("--name", default="auto21")
args = parser.parse_args()

cfg = load_config(args.config)
cache_dir, plot_dir = cfg["paths"]["cache_dir"], cfg["paths"]["plot_dir"]
os.makedirs(plot_dir, exist_ok=True)
seeds = cfg["simulation"]["random_seeds"][: args.n_seeds]
tag = f"_dz{args.slab_dz:g}_k{args.kmin:g}-{args.kmax:g}-{args.nk}_z" + "-".join(f"{z:g}" for z in args.z0)
runs, used = [], []
for s in seeds:
    p = os.path.join(cache_dir, f"seed_{s}", f"auto21_stitched_seed{s}{tag}.npy")
    if os.path.exists(p):
        runs.append(np.load(p, allow_pickle=True).item()); used.append(s)
assert runs, f"no auto21_stitched_seed*{tag}.npy found -- run 04d first (same options)"
nS = len(runs)
zs = sorted(set.intersection(*[set(r.keys()) for r in runs]))
k = runs[0][zs[0]]["k"]
print(f"{nS} seeds, z0 = {zs}")


def stack(key, z):
    return np.array([r[z][key] for r in runs])


import matplotlib.pyplot as plt
from ksz2_21cm.plotting.style import apply_global_style
apply_global_style()
plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "legend.fontsize": 8.5,
                     "xtick.labelsize": 10, "ytick.labelsize": 10})
npan = len(zs) + 1
ncol = 4
nrow = int(np.ceil(npan / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.9 * nrow))
axes = np.atleast_2d(axes)
flat = axes.ravel()
tab = []
print("\n   z0   xbar_e   T0[mK]   " + "   ".join(f"k={kk:g}: mean / median / [16-84%]" for kk in args.k_fixed))
for ax, z in zip(flat, zs):
    D = stack("D2_21", z)
    Dm = stack("D2_delta", z)
    T0 = stack("T0", z)
    xe = stack("xe", z)
    ref = np.nanmean(T0[:, None] ** 2 * Dm, axis=0)
    for i, c in enumerate(D):
        ax.plot(k, c, color="C0", lw=0.7, alpha=0.35, label=f"seeds ({nS})" if i == 0 else None)
    ax.fill_between(k, np.nanpercentile(D, 16, axis=0), np.nanpercentile(D, 84, axis=0), color="C0", alpha=0.18, lw=0)
    ax.plot(k, np.nanmean(D, axis=0), color="C0", lw=2.4, label="mean")
    ax.plot(k, np.nanmedian(D, axis=0), color="navy", lw=1.8, ls="--", label="median")
    ax.plot(k, ref, color="0.4", lw=1.5, ls=":", label=r"neutral limit $T_0^2\Delta^2_\delta$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(rf"$z_0$ = {z:g}   $\bar{{x}}_e$ = {np.mean(xe):.2f}   $T_0$ = {np.mean(T0):.1f} mK", fontsize=10)
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$\Delta^2_{21}(k)$ [mK$^2$]")
    cells = []
    for kk in args.k_fixed:
        i = int(np.argmin(np.abs(k - kk)))
        v = D[:, i]
        cells.append(f"{np.nanmean(v):9.3g} / {np.nanmedian(v):9.3g} / [{np.nanpercentile(v,16):.2g}-{np.nanpercentile(v,84):.2g}]")
    print(f"{z:5g}   {np.mean(xe):5.2f}   {np.mean(T0):6.2f}   " + "   ".join(cells))
flat[0].legend(frameon=True, loc="lower right")

# last panel: Delta^2 vs z at fixed k
ax = flat[len(zs)]
xe_z = np.array([np.mean(stack("xe", z)) for z in zs])
for kk, c in zip(args.k_fixed, ("C0", "C1", "C2")):
    i = int(np.argmin(np.abs(k - kk)))
    V = np.array([stack("D2_21", z)[:, i] for z in zs])      # (nz, nS)
    ax.fill_between(zs, np.nanpercentile(V, 16, axis=1), np.nanpercentile(V, 84, axis=1), color=c, alpha=0.18, lw=0)
    ax.plot(zs, np.nanmean(V, axis=1), "-o", color=c, ms=4, lw=2, label=rf"mean, $k$ = {k[i]:.2g}")
    ax.plot(zs, np.nanmedian(V, axis=1), "--", color=c, lw=1.5)
ax.set_yscale("log")
ax.set_xlim(max(zs) + 0.3, min(zs) - 0.3)
ax.set_xlabel("$z_0$")
ax.set_ylabel(r"$\Delta^2_{21}$ [mK$^2$]")
ax.legend(frameon=True, fontsize=8)
ax.set_title("vs redshift (dashed = median)", fontsize=10)
sec = ax.secondary_xaxis("top", functions=(lambda z: np.interp(z, zs, xe_z), lambda x: np.interp(x, xe_z[::-1], np.array(zs)[::-1])))
sec.set_xlabel(r"$\bar{x}_e$")
for ax in flat[len(zs) + 1:]:
    ax.axis("off")
fig.suptitle(f"21cm auto-spectrum of the stitched lightcone ({nS} seeds, slab dz = {args.slab_dz:g}, "
             "21cmFAST brightness_temp)", fontsize=12)
fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(plot_dir, f"{args.name}.{ext}")
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ Saved: {out}")
np.savez(os.path.join(plot_dir, f"{args.name}_data.npz"), k=k, z0=np.array(zs), seeds=np.array(used),
         D2_21=np.array([stack("D2_21", z) for z in zs]), D2_delta=np.array([stack("D2_delta", z) for z in zs]),
         T0=np.array([stack("T0", z) for z in zs]), xe=np.array([stack("xe", z) for z in zs]))
