#!/usr/bin/env python
# =============================================================================
# 22_heff_history_compare.py
#
# How the kSZ x 21cm cross spectra respond to the timing of reionization, from the 21cmFAST
# HII_EFF_FACTOR scan (slurm/heff_production.sbatch + slurm/heff_post.sbatch). Reads, per history
# folder <data-root>/heff_<H>/cache/seed_<N>/:
#     cross_corr_sq_stitched_seed<N>.npy               (04: Zhou kSZ^2 x 21cm^2)
#     cross_corr_ma_stitched_seed<N><ma-tag>.npy       (04b: Ma kSZ x 21cm and kSZ^2 x 21cm)
#     xe_history_seed<N>.npz                           (x_e(z) of the stitched lightcone)
# and makes, for each statistic (zhou, ma_sq, ma_lin):
#     heff_<stat>.{png,pdf}          left: D_ell vs z0 for every history, one row per ell (mean +- s.e.m.)
#                                    right: the reionization histories with the patchy-window milestones
#                                    z_start (x_e = 1e-4), z50, z_end (x_e = 1 - 1e-4) marked
#     heff_<stat>_shifted.{png,pdf}  the same against z0 - z50 of each history
#     heff_<stat>_xe.{png,pdf}       the same against the volume-weighted ionized fraction x_HII at the chunk centre
#                                    (history mean), 0 -> 1 left to right: do the humps line up in x_HII?
# plus heff_history_data.npz and a printed table (milestones per history; peak / trough / zero
# crossings of every curve; and the peak location of every curve in z0, z0 - z50 and x_HII with a
# bootstrap-over-seeds uncertainty, --bootstrap). Units as 06 / 08 with --chi-norm z0 (D divided by chi(z0)^2).
#
#   python scripts/22_heff_history_compare.py --data-root $R/ksz2-21cm-data
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
from ksz2_21cm.utils.cosmology import get_cosmology
from ksz2_21cm.utils.history import XE_END, XE_START, milestones

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default=os.path.expanduser("~/rds/rds-dirac-dp145-uYnnMXSp128/dc-upad3/ksz2-21cm-data"))
ap.add_argument("--pattern", default="heff_*", help="history folders under --data-root")
ap.add_argument("--ma-tag", default="_dz2_nd0.2_lmin100_z4.5-13.5-0.25", help="04b cache tag (heff_post.sbatch's)")
ap.add_argument("--variant", default="slice", choices=["slice", "narrow", "slab"], help="21cm map for the Ma statistics")
ap.add_argument("--out-dir", default=None, help="default <data-root>/heff_history_compare")
ap.add_argument("--bootstrap", type=int, default=300, help="seed resamples for the peak-location uncertainty (0 = off)")
a = ap.parse_args()

out_dir = a.out_dir or os.path.join(a.data_root, "heff_history_compare")
os.makedirs(out_dir, exist_ok=True)
cosmo = get_cosmology()
STATS = {
    "zhou": dict(ells=[400, 800, 1600], unit=(T_CMB_K * 1e6) ** 2 * 1e6, ylab=r"[$\mu$K$^4$/Mpc$^2$]",
                 title=r"$D_\ell^{\rm kSZ^2\times21cm^2}$ (Zhou)"),
    "ma_sq": dict(ells=[500, 1000, 3000], unit=(T_CMB_K * 1e6) ** 2, ylab=r"[mK $\mu$K$^2$/Mpc$^2$]",
                  title=r"$D_\ell^{\rm kSZ^2\times21cm}$ (Ma)"),
    "ma_lin": dict(ells=[100, 500], unit=T_CMB_K * 1e6, ylab=r"[mK $\mu$K/Mpc$^2$]",
                   title=r"$D_\ell^{\rm kSZ\times21cm}$ (Ma)"),
}


def d_at(ell, C, L, chi, unit):
    ell, C = np.asarray(ell, float), np.asarray(C, float)
    ok = np.isfinite(ell) & np.isfinite(C) & (ell > 0)
    if ok.sum() < 2 or not (ell[ok].min() <= L <= ell[ok].max()):
        return np.nan
    o = np.argsort(ell[ok])
    return L * (L + 1) / (2 * np.pi) * float(np.interp(np.log(L), np.log(ell[ok][o]), C[ok][o])) * unit / chi ** 2


hdirs = sorted(glob.glob(os.path.join(a.data_root, a.pattern)),
               key=lambda p: float(re.findall(r"([0-9.]+)$", p)[0]) if re.findall(r"([0-9.]+)$", p) else 0.0)
H = []
for hd in hdirs:
    m = re.findall(r"([0-9.]+)$", hd)
    if not m or not os.path.isdir(os.path.join(hd, "cache")):
        continue
    heff = float(m[0])
    seeds = sorted(int(re.findall(r"seed_(\d+)$", p)[0]) for p in glob.glob(os.path.join(hd, "cache", "seed_*")))
    per = {}
    xes = []
    for s in seeds:
        sd = os.path.join(hd, "cache", f"seed_{s}")
        pz, pm, px = (os.path.join(sd, f"cross_corr_sq_stitched_seed{s}.npy"),
                      os.path.join(sd, f"cross_corr_ma_stitched_seed{s}{a.ma_tag}.npy"),
                      os.path.join(sd, f"xe_history_seed{s}.npz"))
        if not os.path.exists(px):
            continue
        d = np.load(px)
        xes.append((np.asarray(d["z"], float), np.asarray(d["x_e"], float)))
        per[s] = {"zhou": np.load(pz, allow_pickle=True).item() if os.path.exists(pz) else {},
                  "ma": np.load(pm, allow_pickle=True).item() if os.path.exists(pm) else {}}
    if not xes:
        print(f"{os.path.basename(hd)}: no finished seeds, skipped")
        continue
    zc = xes[0][0]
    xe_mean = np.mean([np.interp(zc, z, x) for z, x in xes], axis=0)
    ms = milestones(zc, xe_mean)
    H.append({"heff": heff, "seeds": sorted(per), "per": per, "z": zc, "xe": xe_mean, "ms": ms,
              "z50_seeds": np.array([milestones(z, x)["z50"] for z, x in xes])})
assert H, f"no finished histories matching {a.pattern} under {a.data_root}"
H.sort(key=lambda h: h["ms"]["z50"])

print(f"{'HII_EFF':>8} {'N':>3} {'z_start':>7} {'z10':>6} {'z50':>6} {'z90':>6} {'z_end':>6} {'dz10-90':>7}")
for h in H:
    m = h["ms"]
    print(f"{h['heff']:8g} {len(h['seeds']):3d} {m['z_start']:7.2f} {m['z10']:6.2f} {m['z50']:6.2f} {m['z90']:6.2f} "
          f"{m['z_end']:6.2f} {m['dz_10_90']:7.2f}   (z50 over seeds {h['z50_seeds'].mean():.2f} +- {h['z50_seeds'].std():.2f})")

# ── D_ell per history, statistic, ell: (n_seeds, n_z0) ──
for h in H:
    h["D"] = {}
    for stat, sp in STATS.items():
        src = "zhou" if stat == "zhou" else "ma"
        sets = [set(h["per"][s][src].keys()) for s in h["seeds"] if h["per"][s][src]]
        z0s = sorted(set.intersection(*sets)) if sets else []
        h.setdefault("z0", {})[stat] = np.array(z0s)
        for L in sp["ells"]:
            T = np.full((len(h["seeds"]), len(z0s)), np.nan)
            for i, s in enumerate(h["seeds"]):
                for j, z0 in enumerate(z0s):
                    chi = float(cosmo.comoving_distance(z0).value)
                    r = h["per"][s][src].get(z0)
                    if r is None:
                        continue
                    if stat == "zhou":
                        T[i, j] = d_at(r["ell"], r["C_cross"], L, chi, sp["unit"])
                    else:
                        key = "C_sq" if stat == "ma_sq" else "C_lin"
                        T[i, j] = d_at(r["ell"], r["variants"][a.variant][key], L, chi, sp["unit"])
            h["D"][(stat, L)] = T


def mean_sem(T):
    m = np.nanmean(T, axis=0)
    n = np.sum(np.isfinite(T), axis=0)
    return m, (np.nanstd(T, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1)) if T.shape[0] > 1 else np.zeros_like(m))


print()
for stat, sp in STATS.items():
    for L in sp["ells"]:
        print(f"{stat} ell={L}:")
        for h in H:
            z0 = h["z0"][stat]
            m, _ = mean_sem(h["D"][(stat, L)])
            ok = np.isfinite(m)
            if ok.sum() < 2:
                print(f"   HII_EFF={h['heff']:g}: no data")
                continue
            zk, mk = z0[ok], m[ok]
            zc = [zk[i] + (0 - mk[i]) * (zk[i + 1] - zk[i]) / (mk[i + 1] - mk[i]) for i in np.where(np.diff(np.sign(mk)) != 0)[0]]
            print(f"   HII_EFF={h['heff']:5g} (z50={h['ms']['z50']:.2f}): peak z0={zk[np.argmax(mk)]:.2f} ({mk.max():.3g}), "
                  f"trough z0={zk[np.argmin(mk)]:.2f} ({mk.min():.3g}), zero crossings {', '.join(f'{c:.2f}' for c in zc) or '-'}")

def xhii_at(h, z0):
    """Volume-weighted x_HII (history mean over seeds) at redshift(s) z0."""
    o = np.argsort(h["z"])
    return np.interp(z0, h["z"][o], h["xe"][o])


def peak_loc(T, xs, nboot, rng):
    """Location (in coordinate xs) of the maximum of the seed-mean curve, with a bootstrap-over-seeds std."""
    m = np.nanmean(T, axis=0)
    ok = np.isfinite(m)
    if ok.sum() < 3:
        return np.nan, np.nan
    base = float(xs[ok][np.argmax(m[ok])])
    boots = []
    for _ in range(nboot):
        mb = np.nanmean(T[rng.integers(0, T.shape[0], T.shape[0])], axis=0)
        okb = np.isfinite(mb)
        if okb.sum() >= 3:
            boots.append(float(xs[okb][np.argmax(mb[okb])]))
    return base, (float(np.std(boots)) if len(boots) > 1 else np.nan)


print("\nPEAK LOCATION of the seed-mean curve (global maximum), +- bootstrap over seeds:")
print("   if the humps follow the ionized fraction, the x_HII column should agree between histories")
rng = np.random.default_rng(0)
for stat, sp in STATS.items():
    for L in sp["ells"]:
        print(f"{stat} ell={L}:")
        for h in H:
            z0 = h["z0"][stat]
            if len(z0) < 3:
                continue
            T = h["D"][(stat, L)]
            nb = a.bootstrap if T.shape[0] > 1 else 0
            pz, ez = peak_loc(T, z0, nb, rng)
            pd_, ed = peak_loc(T, z0 - h["ms"]["z50"], nb, rng)
            px, ex = peak_loc(T, xhii_at(h, z0), nb, rng)
            print(f"   HII_EFF={h['heff']:5g} (z50={h['ms']['z50']:.2f}, N={T.shape[0]:2d}): z0 = {pz:5.2f} +- {ez:4.2f}   "
                  f"z0-z50 = {pd_:+5.2f} +- {ed:4.2f}   x_HII = {px:5.3f} +- {ex:5.3f}")

np.savez(os.path.join(out_dir, "heff_history_data.npz"), heff=np.array([h["heff"] for h in H]),
         **{f"xe_heff{h['heff']:g}": np.vstack([h["z"], h["xe"]]) for h in H},
         **{f"milestones_heff{h['heff']:g}": np.array([h["ms"][k] for k in ("z_start", "z10", "z50", "z90", "z_end")]) for h in H},
         **{f"z0_{stat}_heff{h['heff']:g}": h["z0"][stat] for h in H for stat in STATS},
         **{f"D_{stat}_ell{L}_heff{h['heff']:g}": h["D"][(stat, L)] for h in H for stat, sp in STATS.items() for L in sp["ells"]})

# ── plots ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ksz2_21cm.plotting.style import apply_global_style
apply_global_style()
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11})
cols = plt.cm.plasma(np.linspace(0.1, 0.8, len(H)))


def hist_panel(ax):
    for h, c in zip(H, cols):
        ax.plot(h["z"], h["xe"], "-", color=c, lw=1.8)
        for key, lev, mk in (("z_start", XE_START, "v"), ("z50", 0.5, "o"), ("z_end", XE_END, "^")):
            if np.isfinite(h["ms"][key]):
                ax.plot(h["ms"][key], lev, mk, color=c, ms=7)
                ax.axvline(h["ms"][key], color=c, lw=0.6, ls=":")
    for lev in (XE_START, 0.5, XE_END):
        ax.axhline(lev, color="gray", lw=0.6, ls=":")
    ax.set_xlabel("z")
    ax.set_ylabel(r"$\bar{x}_e$")
    ax.set_title(r"histories: $\nabla$ $z_{\rm start}$ ($x_e=10^{-4}$), $\circ$ $z_{50}$, $\Delta$ $z_{\rm end}$ ($x_e=1-10^{-4}$)", fontsize=9)
    ax.invert_xaxis()


for stat, sp in STATS.items():
    for mode in ("z0", "shifted", "xe"):
        shifted = mode == "shifted"
        n = len(sp["ells"])
        fig = plt.figure(figsize=(14, 3.1 * n + 0.6))
        gs = fig.add_gridspec(n, 2, width_ratios=[1.7, 1])
        axes = [fig.add_subplot(gs[i, 0]) for i in range(n)]
        for ax in axes[1:]:
            ax.sharex(axes[0])
        for ax, L in zip(axes, sp["ells"]):
            for h, c in zip(H, cols):
                m, e = mean_sem(h["D"][(stat, L)])
                if mode == "xe":
                    x = xhii_at(h, h["z0"][stat])
                    o = np.argsort(x)                       # x_HII rises as z0 falls: draw in x order
                    x, m, e = x[o], m[o], e[o]
                else:
                    x = h["z0"][stat] - (h["ms"]["z50"] if shifted else 0.0)
                ax.fill_between(x, m - e, m + e, color=c, alpha=0.2, lw=0)
                ax.plot(x, m, "-o", color=c, ms=3.5, lw=1.6,
                        label=rf"HII_EFF={h['heff']:g} ($z_{{50}}$={h['ms']['z50']:.2f}, N={len(h['seeds'])})")
            ax.axhline(0, color="gray", lw=0.8)
            if shifted:
                ax.axvline(0, color="gray", lw=0.8, ls=":")
            if mode == "xe":
                ax.axvline(0.5, color="gray", lw=0.8, ls=":")
            ax.text(0.02, 0.92, rf"$\ell={L}$", transform=ax.transAxes, va="top")
            ax.set_ylabel(r"$D_\ell/\chi^2$ " + sp["ylab"], fontsize=11)
        axes[0].legend(loc="best", frameon=True, framealpha=0.85, fontsize=9)
        axes[-1].set_xlabel({"z0": r"chunk centre $z_0$", "shifted": r"$z_0 - z_{50}$ of that history",
                             "xe": r"$\bar{x}_{\rm HII}$ at the chunk centre $z_0$ (volume-weighted, history mean)"}[mode])
        if mode == "z0":
            axes[0].invert_xaxis()
        elif mode == "xe":
            axes[0].set_xlim(0.0, 1.0)
        hist_panel(fig.add_subplot(gs[:, 1]))
        fig.suptitle(sp["title"] + " for different reionization histories (21cmFAST, HII_EFF scan)" +
                     ("" if stat == "zhou" else f", 21cm map = '{a.variant}'"), fontsize=13)
        fig.tight_layout()
        stem = f"heff_{stat}" + {"z0": "", "shifted": "_shifted", "xe": "_xe"}[mode]
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"{stem}.{ext}"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {os.path.join(out_dir, stem + '.png')}")
