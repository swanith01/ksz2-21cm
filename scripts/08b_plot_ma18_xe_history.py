#!/usr/bin/env python
# =============================================================================
# 08b_plot_ma18_xe_history.py
#
# 4-panel companion to 08_plot_ma18_style.py (that one is left untouched; this writes
# a DIFFERENT file name, Ma18_xe_history.{pdf,png}).
#
#   left column  : D_ell vs x_e (symlog y so small and large values are both visible),
#                  top = kSZ x 21cm (linear; ell = 100, 500), bottom = kSZ^2 x 21cm
#                  (21cm not squared; ell = 500, 1000, 3000). Ours = mean +/- s.e.m.
#                  over seeds (left axis); Ma+18 digitized (right axis, zeros aligned).
#                  Ma's z is turned into x_e with THEIR OWN history (Fig. 1, digitized).
#   right column : the reionisation histories behind the comparison
#                  top    : x_e(z): ours (mean +/- 1 sigma over seeds), Ma+18, Zhou+25
#                           (Fiducial, thin dotted); marks at x_e = 0.2 / 0.5 / 0.9
#                  bottom : redshift offset z(x_e) - z_Ma(x_e): why plotting vs x_e,
#                           not z, is the fair comparison
#
# Reads only SMALL files written by 08 and 07/08: Ma18_style_data.npz and
# reion_history_stitched.npz. It can be run on usha straight from the copied plots:
#   python scripts/08b_plot_ma18_xe_history.py --data-dir data/plots/2026-10-12
# or on swarm (light, reads two small files):
#   python scripts/08b_plot_ma18_xe_history.py --config configs/variants/stitched_256.yaml
# =============================================================================

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ksz2_21cm.plotting.ma18_data import MA18, MA18_HISTORY, ma18_xe_of_z
from ksz2_21cm.plotting.zhou25_history import ZHOU_HISTORY

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default=None,
                    help="folder holding Ma18_style_data.npz and reion_history_stitched.npz "
                         "(default: plot_dir of --config)")
parser.add_argument("--config", default="configs/variants/stitched_256.yaml")
parser.add_argument("--variant", default="slice", choices=["slice", "narrow", "slab"])
parser.add_argument("--in-name", default="Ma18_style", help="prefix of the 08 data file")
parser.add_argument("--name", default="Ma18_xe_history", help="output name (different from 08's)")
parser.add_argument("--no-zhou", action="store_true", help="leave Zhou+25 out of the history panel")
parser.add_argument("--linthresh", type=float, nargs=2, default=None,
                    help="symlog linear threshold for (ours, Ma); default 10%% of the largest value, per panel")
args = parser.parse_args()

if args.data_dir is None:
    from ksz2_21cm.utils.config import load_config
    data_dir = load_config(args.config)["paths"]["plot_dir"]
else:
    data_dir = os.path.expanduser(args.data_dir)

dpath = os.path.join(data_dir, f"{args.in_name}_data.npz")
hpath = os.path.join(data_dir, "reion_history_stitched.npz")
for p in (dpath, hpath):
    assert os.path.exists(p), f"missing {p} (run 08 / 07 first, or point --data-dir at the copied plots)"
dd, hh = np.load(dpath), np.load(hpath)

z0s = np.asarray(dd["z0"], float)
xe_of_z0 = np.asarray(dd["xe_of_z0"], float)
ell = np.asarray(dd["ell"], float)
Dlin = np.asarray(dd[f"D_lin_{args.variant}"], float)       # (nS, nz, nb)
Dsq = np.asarray(dd[f"D_sq_{args.variant}"], float)
nS = Dlin.shape[0]
chi_norm = str(dd["chi_norm"])

zh, xe_all = np.asarray(hh["z_arr"], float), np.asarray(hh["xe"], float)
o = np.argsort(zh)
zh, xe_all = zh[o], xe_all[:, o]
xe_mean = xe_all.mean(axis=0)
xe_std = xe_all.std(axis=0, ddof=1) if xe_all.shape[0] > 1 else np.zeros_like(xe_mean)


def z_at_xe(x, z, xe):
    """z where the (monotonically decreasing in z) history equals x"""
    o2 = np.argsort(xe)
    return float(np.interp(x, xe[o2], z[o2]))


zf = ZHOU_HISTORY["Fiducial"]
zm, xm = MA18_HISTORY[:, 0], MA18_HISTORY[:, 1]


def D_at_ell(arr, L):
    out = np.full((arr.shape[0], len(z0s)), np.nan)
    for j in range(len(z0s)):
        ok = np.isfinite(ell[j]) & np.all(np.isfinite(arr[:, j]), axis=0)
        if ok.sum() < 3 or not (ell[j][ok].min() <= L <= ell[j][ok].max()):
            continue
        for i in range(arr.shape[0]):
            out[i, j] = np.interp(np.log(L), np.log(ell[j][ok]), arr[i, j][ok])
    return out


def msem(a):
    n = np.sum(np.isfinite(a), axis=0)
    with np.errstate(all="ignore"):
        return np.nanmean(a, axis=0), np.nanstd(a, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))


# ── numbers ─────────────────────────────────────────────────────────────────
XE_REF = [0.1, 0.2, 0.5, 0.9]
print(f"{nS} seeds, 21cm map = '{args.variant}', chi_norm = {chi_norm}")
print("\n z at x_e =      " + "   ".join(f"{x:>5}" for x in XE_REF))
rows = {"ours": [z_at_xe(x, zh, xe_mean) for x in XE_REF],
        "Ma+18": [z_at_xe(x, zm, xm) for x in XE_REF],
        "Zhou+25": [z_at_xe(x, zf[:, 0], zf[:, 1]) for x in XE_REF]}
for k, v in rows.items():
    print(f" {k:9s}        " + "   ".join(f"{a:5.2f}" for a in v))
print(" ours - Ma       " + "   ".join(f"{a - b:+5.2f}" for a, b in zip(rows['ours'], rows['Ma+18'])))

# ── figure ─────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from ksz2_21cm.plotting.style import apply_global_style
apply_global_style()
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 9,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.framealpha": 0.85})

UN = {"lin": r"mK$\,\mu$K", "sq": r"mK$\,\mu$K$^2$"}
NORM = r"$/\chi^2$" if chi_norm == "z0" else ""
LC = {100: "goldenrod", 500: "mediumorchid", 1000: "turquoise", 3000: "purple"}
LS = {"lin": [100, 500], "sq": [500, 1000, 3000]}

fig = plt.figure(figsize=(15.5, 9.2))
gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], wspace=0.30, hspace=0.28)
axL = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
axH, axD = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])


def symlim(vals, lt_user):
    m = np.nanmax(np.abs(vals))
    lt = lt_user if lt_user else 0.1 * m
    return lt, 1.15 * m


for r, (kind, arr) in enumerate((("lin", Dlin), ("sq", Dsq))):
    ax = axL[r]
    mk = "kSZ" if kind == "lin" else "kSZ2"
    hs, ls = [], []
    ours_vals, ma_vals = [], []
    tw = ax.twinx()
    for L in LS[kind]:
        T = D_at_ell(arr, L)
        mu, se = msem(T)
        ok = np.isfinite(mu)
        o2 = np.argsort(xe_of_z0[ok])
        xs, m_, s_ = xe_of_z0[ok][o2], mu[ok][o2], se[ok][o2]
        ax.fill_between(xs, m_ - s_, m_ + s_, color=LC[L], alpha=0.2, lw=0)
        h, = ax.plot(xs, m_, "-", color=LC[L], lw=1.8, label=rf"ours $\ell$={L}")
        hs.append(h); ls.append(h.get_label())
        ours_vals.append(np.abs(np.concatenate([m_ - s_, m_ + s_])))
        key = f"F3_{mk}_l{L}"
        if key in MA18:
            s = MA18[key]
            xe_ma = ma18_xe_of_z(s[:, 0])
            tw.errorbar(xe_ma, s[:, 1], s[:, 2], fmt="s", ms=3, mfc="none", mec=LC[L], ecolor=LC[L],
                        elinewidth=0.7, alpha=0.85)
            hs.append(plt.Line2D([], [], ls="", marker="s", mfc="none", mec=LC[L]))
            ls.append(rf"Ma+18 $\ell$={L}")
            ma_vals.append(np.abs(np.concatenate([s[:, 1] - s[:, 2], s[:, 1] + s[:, 2]])))
    lt1, m1 = symlim(np.concatenate(ours_vals), args.linthresh[0] if args.linthresh else None)
    lt2, m2 = symlim(np.concatenate(ma_vals), args.linthresh[1] if args.linthresh else None)
    ax.set_yscale("symlog", linthresh=lt1)
    tw.set_yscale("symlog", linthresh=lt2)
    ax.set_ylim(-m1, m1)
    tw.set_ylim(-m2, m2)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylabel(rf"$D_\ell$ {NORM} [{UN[kind]}]  (ours, symlog)")
    tw.set_ylabel(rf"Ma+18 $D_\ell$ [{UN[kind]}]  (squares, symlog)")
    ax.set_title({"lin": r"kSZ $\times$ 21cm (linear)",
                  "sq": r"kSZ$^2$ $\times$ 21cm (21cm not squared)"}[kind], fontsize=11)
    ax.legend(hs, ls, frameon=True, fontsize=8, ncol=2, loc="lower right")
axL[1].set_xlabel(r"$\bar{x}_e$  (Ma+18 mapped through their own history)")
axL[0].set_xlabel(r"$\bar{x}_e$")

# right-top: histories
axH.fill_between(zh, xe_mean - xe_std, xe_mean + xe_std, color="C0", alpha=0.25, lw=0)
axH.plot(zh, xe_mean, color="C0", lw=2.2, label=f"this work, 21cmFAST ({xe_all.shape[0]} seeds)")
axH.plot(zm, xm, color="C3", lw=2.2, label="Ma+18 (Fig. 1, digitized)")
if not args.no_zhou:
    axH.plot(zf[:, 0], zf[:, 1], color="0.35", lw=1.4, ls=":", label="Zhou+25 Fiducial (for reference)")
for x, mkr in zip((0.2, 0.5, 0.9), ("o", "s", "^")):
    axH.plot([z_at_xe(x, zh, xe_mean)], [x], mkr, color="C0", ms=7, mec="k", zorder=5)
    axH.plot([z_at_xe(x, zm, xm)], [x], mkr, color="C3", ms=7, mec="k", zorder=5)
axH.set_xlim(17, 5.5)
axH.set_ylim(-0.02, 1.02)
axH.set_xlabel("$z$")
axH.set_ylabel(r"$\bar{x}_e(z)$")
axH.legend(frameon=True, loc="upper left", fontsize=9)
axH.text(0.97, 0.04, "markers: $\\bar{x}_e$ = 0.2 (circle),\n0.5 (square), 0.9 (triangle)",
         transform=axH.transAxes, fontsize=8.5, ha="right", va="bottom")
axH.set_title("reionisation histories", fontsize=11)

# right-bottom: z offset at fixed x_e
xg = np.linspace(0.05, 0.95, 91)
z_o = np.array([z_at_xe(x, zh, xe_mean) for x in xg])
z_m = np.array([z_at_xe(x, zm, xm) for x in xg])
z_z = np.array([z_at_xe(x, zf[:, 0], zf[:, 1]) for x in xg])
axD.plot(xg, z_o - z_m, color="C0", lw=2.2, label="ours $-$ Ma+18")
if not args.no_zhou:
    axD.plot(xg, z_z - z_m, color="0.35", lw=1.4, ls=":", label="Zhou+25 $-$ Ma+18")
axD.axhline(0, color="gray", lw=0.8)
axD.set_xlim(0, 1)
axD.set_xlabel(r"$\bar{x}_e$")
axD.set_ylabel(r"$z(\bar{x}_e)$ offset from Ma+18")
axD.legend(frameon=True, loc="best", fontsize=9)
axD.set_title("how far each history is shifted in redshift", fontsize=11)

fig.suptitle(f"kSZ x 21cm vs ionized fraction, with the reionisation histories behind it  "
             f"({nS} seeds, 21cm map = '{args.variant}')", fontsize=12)
for ext in ("pdf", "png"):
    out = os.path.join(data_dir, f"{args.name}.{ext}")
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ Saved: {out}")
plt.close(fig)
