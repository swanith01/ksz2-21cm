#!/usr/bin/env python
# =============================================================================
# 19_plot_coeval_direct.py
#
# Coeval-box estimates (script 18) against the stitched lightcone (scripts 04 / 04b) for
# the SAME seeds, one figure per statistic, D_ell vs chunk centre z0 at fixed ell:
#   coeval_vs_lc_zhou.{png,pdf}    kSZ^2 x 21cm^2     lightcone | slab | Limber (bispectrum)
#   coeval_vs_lc_ma_sq.{png,pdf}   kSZ^2 x 21cm       lightcone | slab | Limber (bispectrum)
#   coeval_vs_lc_ma_lin.{png,pdf}  kSZ x 21cm         lightcone | Ma eq 8 (measured P_dT) | Ma eq 8 (linearised T21)
# Bands / bars = s.e.m. over seeds. Units as 06 / 08 with --chi-norm z0 (divided by chi(z0)^2).
# Also: coeval_vs_lc_data.npz (every curve) and coeval_vs_lc_summary.csv (per estimator and ell:
# least-squares amplitude coeval/lightcone over z0, and the correlation of the two z0 curves).
#
#   python scripts/19_plot_coeval_direct.py --config configs/variants/stitched_256.yaml
#   python scripts/19_plot_coeval_direct.py --config ... --variant slab --seeds 1 2 3
# =============================================================================

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN slices where an ell is outside a chunk's range

from ksz2_21cm.utils.config import load_config
from ksz2_21cm.utils.constants import T_CMB_K

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/variants/stitched_256.yaml")
ap.add_argument("--seeds", type=int, nargs="+", default=None)
ap.add_argument("--variant", default="slice", choices=["slice", "narrow", "slab"], help="21cm map for the Ma statistics")
ap.add_argument("--ma-tag", default="_dz2_nd0.2_lmin100_z5.5-15-0.25", help="04b cache tag")
ap.add_argument("--out-dir", default=None, help="default <plot_dir>/coeval_direct")
args = ap.parse_args()

cfg = load_config(args.config)
cache_dir = os.path.expanduser(cfg["paths"]["cache_dir"])
out_dir = args.out_dir or os.path.join(os.path.expanduser(cfg["paths"]["plot_dir"]), "coeval_direct")
os.makedirs(out_dir, exist_ok=True)
seeds = args.seeds or cfg["simulation"]["random_seeds"]

U_ZHOU = (T_CMB_K * 1e6) ** 2 * 1e6          # (dT/T)^2 mK^2 -> muK^4
U_LIN, U_SQ = T_CMB_K * 1e6, (T_CMB_K * 1e6) ** 2   # -> mK muK, mK muK^2 (as 08)
STATS = {
    "zhou": dict(ells=[400, 800, 1600], unit=U_ZHOU, ylab=r"$D_\ell/\chi^2$ [$\mu$K$^4$/Mpc$^2$]",
                 title=r"kSZ$^2\times$21cm$^2$ (Zhou)", est=[("slab", "coeval slabs"), ("limber", "Limber (bispectrum)")]),
    "ma_sq": dict(ells=[500, 1000, 3000], unit=U_SQ, ylab=r"$D_\ell/\chi^2$ [mK $\mu$K$^2$/Mpc$^2$]",
                  title=r"kSZ$^2\times$21cm (Ma)", est=[("sq_slab", "coeval slabs"), ("sq_limber", "Limber (bispectrum)")]),
    "ma_lin": dict(ells=[100, 500], unit=U_LIN, ylab=r"$D_\ell/\chi^2$ [mK $\mu$K/Mpc$^2$]",
                   title=r"kSZ$\times$21cm (Ma)", est=[("lin_maeq8", r"Ma eq 8, measured $P_{\delta T}$"),
                                                       ("lin_maeq8_lin", r"Ma eq 8, linearised $T_{21}$")]),
}


def d_at(ell_arr, C, ell, chi, unit):
    """D_ell/chi^2 at ell (log-ell interpolation of C over finite bins)."""
    ok = np.isfinite(C) & np.isfinite(ell_arr) & (ell_arr > 0)
    if ok.sum() < 2 or not (ell_arr[ok].min() <= ell <= ell_arr[ok].max()):
        return np.nan
    c = np.interp(np.log(ell), np.log(ell_arr[ok]), C[ok])
    return ell * (ell + 1) / (2 * np.pi) * c * unit / chi ** 2


rows = {}   # (stat, source) -> list over seeds of (nz0, nell)
z0s, xe_z0 = None, {}
used = []
for s in seeds:
    p_co = os.path.join(cache_dir, f"seed_{s}", f"coeval_direct_seed{s}.npy")
    p_zh = os.path.join(cache_dir, f"seed_{s}", f"cross_corr_sq_stitched_seed{s}.npy")
    p_ma = os.path.join(cache_dir, f"seed_{s}", f"cross_corr_ma_stitched_seed{s}{args.ma_tag}.npy")
    if not os.path.exists(p_co):
        continue
    co = np.load(p_co, allow_pickle=True).item()["results"]
    zh = np.load(p_zh, allow_pickle=True).item() if os.path.exists(p_zh) else {}
    ma = np.load(p_ma, allow_pickle=True).item() if os.path.exists(p_ma) else {}
    zs = sorted(co)
    z0s = zs if z0s is None else [z for z in z0s if z in zs]
    used.append((s, co, zh, ma))
    for z in zs:
        xe_z0.setdefault(z, []).append(co[z]["x_e_z0"])
assert used, f"no coeval_direct_seed*.npy in {cache_dir}/seed_*/ -- run scripts/18 first"
print(f"seeds: {[u[0] for u in used]}   z0: {z0s}   Ma variant: {args.variant}")


def lookup(src, stat, s_co, s_zh, s_ma, z):
    """C(k) array and its ell array for one seed, chunk, statistic and source (None if missing)."""
    r = s_co[z]
    if src == "lightcone":
        if stat == "zhou":
            d = s_zh.get(z)
            return (d["ell"], d["C_cross"]) if d else None
        d = s_ma.get(z)
        if not d:
            return None
        key = "C_sq" if stat == "ma_sq" else "C_lin"
        return d["ell"], d["variants"][args.variant][key]
    if stat == "zhou":
        return r["ell"], r["zhou"][src]
    return r["ell"], r["variants"][args.variant][src]


curves = {}
for stat, spec in STATS.items():
    for src in ["lightcone"] + [e for e, _ in spec["est"]]:
        arr = np.full((len(used), len(z0s), len(spec["ells"])), np.nan)
        for i, (s, co, zh, ma) in enumerate(used):
            for j, z in enumerate(z0s):
                got = lookup(src, stat, co, zh, ma, z)
                if got is None:
                    continue
                for k, L in enumerate(spec["ells"]):
                    arr[i, j, k] = d_at(np.asarray(got[0], float), np.asarray(got[1], float), L, co[z]["chi_z0"], spec["unit"])
        curves[(stat, src)] = arr


def mean_sem(a):
    m = np.nanmean(a, axis=0)
    n = np.sum(np.isfinite(a), axis=0)
    e = np.nanstd(a, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1)) if a.shape[0] > 1 else np.zeros_like(m)
    return m, e


# ── summary ──────────────────────────────────────────────────────────────────
summ = []
for stat, spec in STATS.items():
    lc, _ = mean_sem(curves[(stat, "lightcone")])
    for src, lab in spec["est"]:
        m, _ = mean_sem(curves[(stat, src)])
        for k, L in enumerate(spec["ells"]):
            ok = np.isfinite(lc[:, k]) & np.isfinite(m[:, k])
            amp = float(np.sum(m[ok, k] * lc[ok, k]) / np.sum(lc[ok, k] ** 2)) if ok.sum() >= 2 and np.any(lc[ok, k]) else np.nan
            r = float(np.corrcoef(m[ok, k], lc[ok, k])[0, 1]) if ok.sum() >= 3 else np.nan
            summ.append({"statistic": stat, "estimator": src, "ell": L, "n_z0": int(ok.sum()),
                         "amplitude_vs_lightcone": amp, "corr_with_lightcone": r})
            print(f"  {stat:6s} {src:14s} ell={L:5d}: amplitude/lightcone = {amp:7.3f}   corr = {r:6.3f}   ({ok.sum()} z0)")
with open(os.path.join(out_dir, "coeval_vs_lc_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
    w.writeheader()
    w.writerows(summ)
np.savez(os.path.join(out_dir, "coeval_vs_lc_data.npz"), z0=np.array(z0s), seeds=np.array([u[0] for u in used]),
         variant=args.variant, x_e=np.array([np.mean(xe_z0[z]) for z in z0s]),
         **{f"{st}__{src}": a for (st, src), a in curves.items()},
         **{f"ells__{st}": np.array(sp["ells"]) for st, sp in STATS.items()})

# ── plots ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ksz2_21cm.plotting.style import apply_global_style
apply_global_style()
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})
z0a = np.array(z0s)
xe_a = np.array([np.mean(xe_z0[z]) for z in z0s])
STYLE = {"lightcone": dict(color="#0072B2"), 1: dict(color="#D55E00", marker="o"), 2: dict(color="#009E73", marker="s")}

for stat, spec in STATS.items():
    ells = spec["ells"]
    fig, axes = plt.subplots(len(ells), 1, figsize=(8.0, 3.0 * len(ells)), sharex=True)
    axes = np.atleast_1d(axes)
    for k, (ax, L) in enumerate(zip(axes, ells)):
        m, e = mean_sem(curves[(stat, "lightcone")])
        ax.fill_between(z0a, m[:, k] - e[:, k], m[:, k] + e[:, k], color=STYLE["lightcone"]["color"], alpha=0.25, lw=0)
        ax.plot(z0a, m[:, k], "-", lw=2, label="stitched lightcone", **STYLE["lightcone"])
        for n, (src, lab) in enumerate(spec["est"], start=1):
            m2, e2 = mean_sem(curves[(stat, src)])
            ax.errorbar(z0a + 0.04 * (n - 1.5), m2[:, k], yerr=e2[:, k], ls="--", lw=1.2, ms=4, capsize=2, label=lab, **STYLE[n])
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_ylabel(spec["ylab"], fontsize=11)
        ax.text(0.02, 0.92, rf"$\ell={L}$", transform=ax.transAxes, va="top")
        if k == 0:
            ax.legend(loc="best", frameon=False)
    axes[-1].set_xlabel(r"chunk centre $z_0$")
    for ax in axes:
        ax.invert_xaxis()
    top = axes[0].twiny()
    top.set_xlim(axes[0].get_xlim())
    tz = [z for z in z0s if abs(z - round(z)) < 1e-9]
    top.set_xticks(tz)
    top.set_xticklabels([f"{xe_a[z0s.index(z)]:.2f}" for z in tz])
    top.set_xlabel(r"$\bar{x}_{\rm HII}$ (coeval)")
    var = "" if stat == "zhou" else f", 21cm map = '{args.variant}'"
    fig.suptitle(f"{spec['title']}: coeval boxes vs stitched lightcone ({len(used)} seeds{var})", y=1.0, fontsize=13)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"coeval_vs_lc_{stat}.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {os.path.join(out_dir, f'coeval_vs_lc_{stat}.png')}")
