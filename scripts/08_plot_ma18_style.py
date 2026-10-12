#!/usr/bin/env python
# =============================================================================
# 08_plot_ma18_style.py
#
# Ma+2018 (MNRAS 476, 4025) style plots and the "is it zero?" table, from the
# 04b output (kSZ x 21cm LINEAR and kSZ^2 x 21cm, 21cm not squared).
#
#   Fig A  (analog of Ma Fig. 2, middle/bottom): D_ell vs ell at the redshifts where
#          OUR mean x_e = 0.2 / 0.5 / 0.9 (Ma uses z = 13 / 10.8 / 9.3 for the same
#          x_e). Ours = mean +/- s.e.m. over seeds (left axis); Ma digitized
#          (right axis, zeros aligned; amplitudes are not like-for-like).
#   Fig B  (analog of Ma Fig. 3): D_ell vs z at fixed ell (linear: 100, 500;
#          squared: 500, 1000, 3000), and vs x_e (Ma mapped through their three
#          known (z, x_e) points -- history-independent comparison).
#   Table  null test: band-averaged D per seed, t = mean/s.e.m. over seeds, for
#          signal AND for the cross-seed null (21cm of seed i x kSZ of seed i+1),
#          at x_e = 0.2/0.5/0.9 and summarised over the whole z0 grid.
#
# D_ell = ell(ell+1) C_ell / 2 pi, in [mK uK] (linear) or [mK uK^2] (squared), like
# Ma's axes. --chi-norm z0 (default) divides C by chi(z0)^2 (angular C_ell, as in 06).
#
# Run (after 04b; reads only small files, except the first time when it also builds
# the reion history from the lightcones if 07 has not cached it -- use an interactive node):
#   python scripts/08_plot_ma18_style.py --config configs/variants/stitched_256.yaml
# =============================================================================

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ksz2_21cm.utils.config import load_config
from ksz2_21cm.utils.cosmology import get_cosmology
from ksz2_21cm.utils.constants import T_CMB_K
from ksz2_21cm.plotting.ma18_data import MA18

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/variants/stitched_256.yaml")
# must match the 04b run (they define the cache file name)
parser.add_argument("--z0-min", type=float, default=5.5)
parser.add_argument("--z0-max", type=float, default=15.0)
parser.add_argument("--z0-step", type=float, default=0.25)
parser.add_argument("--slab-dz", type=float, default=None)
parser.add_argument("--narrow-dz", type=float, default=0.2)
parser.add_argument("--ell-min-ksz", type=float, default=100.0)
parser.add_argument("--variant", default="slice", choices=["slice", "narrow", "slab"],
                    help="which 21cm map goes in the figures (the table lists all three)")
parser.add_argument("--chi-norm", choices=["none", "z0"], default="z0")
parser.add_argument("--same-axis", action="store_true", help="draw Ma on the SAME axis as ours (no twin axis)")
parser.add_argument("--xe", type=float, nargs=3, default=[0.2, 0.5, 0.9])
parser.add_argument("--name", default="Ma18_style")
args = parser.parse_args()

cfg = load_config(args.config)
cache_dir = cfg["paths"]["cache_dir"]
plot_dir = cfg["paths"]["plot_dir"]
os.makedirs(plot_dir, exist_ok=True)
seeds = cfg["simulation"]["random_seeds"]
slab_dz = args.slab_dz if args.slab_dz is not None else cfg["cross_correlation"]["delta_z"]
tag = f"_dz{slab_dz:g}_nd{args.narrow_dz:g}_lmin{args.ell_min_ksz:g}_z{args.z0_min:g}-{args.z0_max:g}-{args.z0_step:g}"
cosmo = get_cosmology(cfg)

# ── load the per-seed caches ─────────────────────────────────────────────────
per_seed = []
used = []
for s in seeds:
    p = os.path.join(cache_dir, f"seed_{s}", f"cross_corr_ma_stitched_seed{s}{tag}.npy")
    if os.path.exists(p):
        per_seed.append(np.load(p, allow_pickle=True).item())
        used.append(s)
assert per_seed, f"no cross_corr_ma_stitched_seed*{tag}.npy found -- run 04b first (same z0/slab/ell options)"
nS = len(per_seed)
z0s = np.array(sorted(per_seed[0].keys()))
for r in per_seed:
    assert np.allclose(sorted(r.keys()), z0s)
print(f"Loaded {nS} seeds, {len(z0s)} z0 values ({z0s[0]}..{z0s[-1]}), tag {tag}")

# ── our mean reionization history (cached by script 07) ─────────────────────
hist_path = os.path.join(plot_dir, "reion_history_stitched.npz")
if os.path.exists(hist_path):
    h = np.load(hist_path)
    zh, xe_all = h["z_arr"], h["xe"]
else:
    rows, zh = [], None
    for s in seeds:
        p = os.path.join(cache_dir, f"seed_{s}", f"stitched_lightcone_seed{s}.npz")
        if os.path.exists(p):
            d = np.load(p)
            zh = np.asarray(d["z_arr"], float)
            rows.append(1.0 - np.asarray(d["xH_box"]).mean(axis=(0, 1)))
    xe_all = np.array(rows)
    np.savez(hist_path, z_arr=zh, xe=xe_all)
o = np.argsort(zh)
zh, xe_mean = zh[o], xe_all[:, o].mean(axis=0)
xe_of_z0 = np.interp(z0s, zh, xe_mean)


def z_at_xe(x):
    return float(np.interp(x, xe_mean[::-1], zh[::-1]))   # x_e decreases with z


XE_TARGETS = list(args.xe)
Z_EPOCH = [z_at_xe(x) for x in XE_TARGETS]
IZ_EPOCH = [int(np.argmin(np.abs(z0s - z))) for z in Z_EPOCH]
print("Epochs (ours): " + ", ".join(f"x_e={x}: z={z:.2f} (grid z0={z0s[i]:.2f})"
                                    for x, z, i in zip(XE_TARGETS, Z_EPOCH, IZ_EPOCH)))
print("Ma+18        : x_e=0.2: z=13, 0.5: z=10.8, 0.9: z=9.3")

# ── arrays: D[kind][variant] -> (nS, nz, nb) ; ell (nz, nb) ─────────────────
UNIT = {"lin": T_CMB_K * 1e6, "sq": (T_CMB_K * 1e6) ** 2}     # x mK from T21 as-is
chi = np.array([per_seed[0][z]["chi_z0"] for z in z0s])
ell = np.array([per_seed[0][z]["ell"] for z in z0s])
nb = ell.shape[1]
VARS = ["slice", "narrow", "slab"]
KINDS = [("lin", "C_lin", "C_lin_null"), ("sq", "C_sq", "C_sq_null")]
D, Dn = {}, {}
for kind, key, keyn in KINDS:
    for v in VARS:
        arr = np.full((nS, len(z0s), nb), np.nan)
        arrn = np.full_like(arr, np.nan)
        for i, r in enumerate(per_seed):
            for j, z in enumerate(z0s):
                var = r[z]["variants"][v]
                fac = ell[j] * (ell[j] + 1) / (2 * np.pi) * UNIT[kind]
                if args.chi_norm == "z0":
                    fac = fac / chi[j] ** 2
                arr[i, j] = fac * var[key]
                if keyn in var:
                    arrn[i, j] = fac * var[keyn]
        D[(kind, v)], Dn[(kind, v)] = arr, arrn


def sem(a, axis=0):
    n = np.sum(np.isfinite(a), axis=axis)
    return np.nanstd(a, axis=axis, ddof=1) / np.sqrt(np.maximum(n, 1))


# ── null test table ─────────────────────────────────────────────────────────
BANDS = [(100, 500), (500, 1500), (1500, 5000)]
try:
    from scipy import stats as _st
    def pval(t, n):
        return 2 * _st.t.sf(abs(t), n - 1)
except Exception:                                             # pragma: no cover
    from math import erf, sqrt
    def pval(t, n):
        return 1 - erf(abs(t) / sqrt(2))


def band_t(arr, j, lo, hi):
    """band-averaged D per seed at z0 index j; returns (mean, sem, t)."""
    m = (ell[j] >= lo) & (ell[j] < hi)
    if not m.any():
        return np.nan, np.nan, np.nan
    b = np.nanmean(arr[:, j, m], axis=1)
    mu = np.nanmean(b)
    se = np.nanstd(b, ddof=1) / np.sqrt(np.sum(np.isfinite(b)))
    return mu, se, (mu / se if se > 0 else np.nan)


lines = []
def P(s=""):
    print(s)
    lines.append(s)

LAB = {"lin": "kSZ x 21cm  (linear)   [mK uK]", "sq": "kSZ^2 x 21cm (21cm not squared) [mK uK^2]"}
P(f"NULL TEST  ({nS} seeds; chi-norm={args.chi_norm}; t = band-mean / s.e.m. over seeds; "
  f"null = same 21cm x kSZ of a different seed)")
P(f"A real signal shows |t_signal| >> |t_null|. |t| < ~2-3 for both => consistent with zero.\n")
for kind, _, _ in KINDS:
    P("=" * 100)
    P(LAB[kind])
    P("=" * 100)
    P(f"{'variant':10s} {'x_e (z0)':>14s} {'ell band':>12s} {'mean D':>12s} {'s.e.m.':>11s} {'t_sig':>7s} {'p':>8s} {'t_null':>8s}")
    for v in VARS:
        for x, j in zip(XE_TARGETS, IZ_EPOCH):
            for lo, hi in BANDS:
                mu, se, t = band_t(D[(kind, v)], j, lo, hi)
                _, _, tn = band_t(Dn[(kind, v)], j, lo, hi)
                P(f"{v:10s} {f'{x} ({z0s[j]:.2f})':>14s} {f'{lo}-{hi}':>12s} {mu:12.3e} {se:11.2e} {t:7.2f} "
                  f"{pval(t, nS):8.3f} {tn:8.2f}")
        P("")
    P("Over the whole z0 grid (every z0 x ell band; z0 neighbours are correlated, so read as a rough scan):")
    P(f"{'variant':10s} {'max|t_sig|':>11s} {'max|t_null|':>12s} {'frac|t_sig|>3':>14s} {'frac|t_null|>3':>15s} {'N':>5s}")
    for v in VARS:
        ts, tns = [], []
        for j in range(len(z0s)):
            for lo, hi in BANDS:
                ts.append(band_t(D[(kind, v)], j, lo, hi)[2])
                tns.append(band_t(Dn[(kind, v)], j, lo, hi)[2])
        ts, tns = np.array(ts), np.array(tns)
        okk, okn = np.isfinite(ts), np.isfinite(tns)
        P(f"{v:10s} {np.nanmax(np.abs(ts)):11.2f} {np.nanmax(np.abs(tns)) if okn.any() else np.nan:12.2f} "
          f"{np.mean(np.abs(ts[okk]) > 3):14.2f} {np.mean(np.abs(tns[okn]) > 3) if okn.any() else np.nan:15.2f} {okk.sum():5d}")
    P("")
with open(os.path.join(plot_dir, f"{args.name}_table.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"✓ Saved: {os.path.join(plot_dir, args.name + '_table.txt')}")

# ── figures ──────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from ksz2_21cm.plotting.style import apply_global_style
apply_global_style()
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 9,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.framealpha": 0.85})

COLS = ["C0", "C1", "C2"]
UN = {"lin": r"$\mathrm{mK}\,\mu\mathrm{K}$", "sq": r"$\mathrm{mK}\,\mu\mathrm{K}^2$"}
NORM = r"$/\chi^2$" if args.chi_norm == "z0" else ""
vname = args.variant


def ma_points(tgt, series, color):
    tgt.errorbar(series[:, 0], series[:, 1], series[:, 2], fmt="s", ms=3.5, mfc="none", mec=color,
                 ecolor=color, elinewidth=0.8, capsize=0, alpha=0.85)


def align_zero(ax1, ax2):
    """symmetric limits on both axes so zeros coincide"""
    for a in (ax1, ax2):
        lo, hi = a.get_ylim()
        m = max(abs(lo), abs(hi))
        a.set_ylim(-m, m)


# Fig A -----------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
for ax, (kind, _, _) in zip(axes, KINDS):
    arr = D[(kind, vname)]
    hs, ls = [], []
    for c, x, j in zip(COLS, XE_TARGETS, IZ_EPOCH):
        mu = np.nanmean(arr[:, j], axis=0)
        se = sem(arr[:, j])
        ax.fill_between(ell[j], mu - se, mu + se, color=c, alpha=0.2, lw=0)
        h, = ax.plot(ell[j], mu, "-", color=c, lw=1.8, label=f"ours x_e={x} (z={z0s[j]:.1f})")
        hs.append(h); ls.append(h.get_label())
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlim(100, 8000)
    ax.set_ylabel(rf"$D_\ell$ {NORM} [{UN[kind]}]   (ours)")
    ax.set_title({"lin": r"kSZ $\times$ 21cm (linear)", "sq": r"kSZ$^2$ $\times$ 21cm (21cm not squared)"}[kind], fontsize=12)
    mk = "kSZ" if kind == "lin" else "kSZ2"
    tw = ax if args.same_axis else ax.twinx()
    for c, x, zz in zip(COLS, (0.2, 0.5, 0.9), (13, 10.8, 9.3)):
        ma_points(tw, MA18[f"F2_{mk}_xe{x}"], c)
    if not args.same_axis:
        tw.set_ylabel(f"Ma+18 $D_\\ell$ [{UN[kind]}]  (squares, right axis)")
        align_zero(ax, tw)
        hs += [plt.Line2D([], [], ls="", marker="s", mfc="none", mec=c) for c in COLS]
        ls += [f"Ma+18 x_e={x} (z={zz})" for x, zz in zip((0.2, 0.5, 0.9), (13, 10.8, 9.3))]
    ax.legend(hs, ls, frameon=True, loc="upper right", ncol=2, fontsize=8)
axes[-1].set_xlabel(r"$\ell$")
fig.suptitle(f"Ma+18 Fig. 2 analog -- {nS} seeds, 21cm map = '{vname}'", fontsize=12)
fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(plot_dir, f"{args.name}_vs_ell.{ext}")
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ Saved: {out}")
plt.close(fig)


# Fig B -----------------------------------------------------------------------
def D_at_ell(arr, L):
    """per-seed D at fixed ell (log-interp over bins) -> (nS, nz)"""
    out = np.full((arr.shape[0], len(z0s)), np.nan)
    for j in range(len(z0s)):
        m = np.isfinite(ell[j]) & np.all(np.isfinite(arr[:, j]), axis=0)
        if m.sum() < 3 or not (ell[j][m].min() <= L <= ell[j][m].max()):
            continue
        for i in range(arr.shape[0]):
            out[i, j] = np.interp(np.log(L), np.log(ell[j][m]), arr[i, j][m])
    return out


# Ma's x_e(z) through their three quoted points (valid only between x_e = 0.2 and 0.9)
MA_Z, MA_X = np.array([9.3, 10.8, 13.0]), np.array([0.9, 0.5, 0.2])
def ma_xe(z):
    return np.where((z >= 9.3) & (z <= 13.0), np.interp(z, MA_Z, MA_X), np.nan)


fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
LS = {"lin": [100, 500], "sq": [500, 1000, 3000]}
LC = {100: "goldenrod", 500: "mediumorchid", 1000: "turquoise", 3000: "purple"}
for r, (kind, _, _) in enumerate(KINDS):
    arr = D[(kind, vname)]
    mk = "kSZ" if kind == "lin" else "kSZ2"
    for col, xaxis in enumerate(("z", "xe")):
        ax = axes[r, col]
        hs, ls = [], []
        for L in LS[kind]:
            T = D_at_ell(arr, L)
            mu, se = np.nanmean(T, axis=0), sem(T)
            xs = z0s if xaxis == "z" else xe_of_z0
            o = np.argsort(xs)
            c = LC[L]
            ax.fill_between(xs[o], (mu - se)[o], (mu + se)[o], color=c, alpha=0.2, lw=0)
            h, = ax.plot(xs[o], mu[o], "-", color=c, lw=1.8, label=rf"ours $\ell$={L}")
            hs.append(h); ls.append(h.get_label())
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_ylabel(rf"$D_\ell$ {NORM} [{UN[kind]}]  (ours)")
        # Ma on twin axis
        tw = ax if args.same_axis else ax.twinx()
        for L in LS[kind]:
            key = f"F3_{mk}_l{L}"
            if key not in MA18:
                continue
            s = MA18[key]
            xs = s[:, 0] if xaxis == "z" else ma_xe(s[:, 0])
            ok = np.isfinite(xs)
            tw.errorbar(xs[ok], s[ok, 1], s[ok, 2], fmt="s", ms=3, mfc="none", mec=LC[L], ecolor=LC[L],
                        elinewidth=0.7, alpha=0.8)
            hs.append(plt.Line2D([], [], ls="", marker="s", mfc="none", mec=LC[L]))
            ls.append(rf"Ma+18 $\ell$={L}")
        if not args.same_axis:
            tw.set_ylabel(f"Ma+18 [{UN[kind]}] (right axis)")
            align_zero(ax, tw)
        if xaxis == "z":
            ax.set_xlim(max(z0s.max(), 16.7), 5.5)
            ax.set_xlabel("$z$")
        else:
            ax.set_xlim(0, 1)
            ax.set_xlabel(r"$\bar{x}_e$ (Ma+18 mapped via their $x_e$ = 0.2/0.5/0.9 at $z$ = 13/10.8/9.3)")
        ax.legend(hs, ls, frameon=True, fontsize=8, ncol=2, loc="upper right")
        ax.set_title({"lin": "kSZ x 21cm (linear)", "sq": "kSZ$^2$ x 21cm"}[kind] +
                     (" vs z" if xaxis == "z" else " vs ionized fraction"), fontsize=11)
fig.suptitle(f"Ma+18 Fig. 3 analog -- {nS} seeds, 21cm map = '{vname}'", fontsize=12)
fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(plot_dir, f"{args.name}_vs_z.{ext}")
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ Saved: {out}")
plt.close(fig)

# data for replotting ---------------------------------------------------------
payload = {"z0": z0s, "xe_of_z0": xe_of_z0, "ell": ell, "chi_z0": chi, "seeds": np.array(used),
           "chi_norm": args.chi_norm, "z_epoch": np.array(Z_EPOCH), "xe_targets": np.array(XE_TARGETS)}
for (kind, v), a in D.items():
    payload[f"D_{kind}_{v}"] = a
    payload[f"Dnull_{kind}_{v}"] = Dn[(kind, v)]
np.savez(os.path.join(plot_dir, f"{args.name}_data.npz"), **payload)
print(f"✓ Saved: {os.path.join(plot_dir, args.name + '_data.npz')}")
