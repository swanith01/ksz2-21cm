#!/usr/bin/env python
# =============================================================================
# 09_plot_wedge_pedagogy.py
#
# Pedagogical figure: does kSZ^2 x 21cm survive foreground (k_par / wedge) filtering
# when the 21cm field is NOT squared, versus when it is (Zhou+25)?
#
# Uses the dimensionless correlation coefficient of the two 2D maps,
#     r = C_cross / sqrt(P_21cm-map * P_kSZ^2),
# averaged over an ell band, because the linear and squared statistics have different
# units. Six statistics: {linear (central LOS cell), squared-then-projected (Zhou)} x
# {no filter, k_par > k_par_min, horizon wedge}. The grey band is the cross-seed null
# (21cm of seed i x kSZ^2 of seed i+1).
#
#   (a) r vs z0 in the ell band     (b) S/N = mean(r)/s.e.m. over seeds vs z0
#   (c) r vs ell at the z0 where our mean x_e = 0.5
#
# Run after 04c (interactive node):
#   python scripts/09_plot_wedge_pedagogy.py --config configs/variants/stitched_256.yaml
# =============================================================================

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ksz2_21cm.utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/variants/stitched_256.yaml")
# must match the 04c run (they define the cache file name)
parser.add_argument("--z0-min", type=float, default=6.0)
parser.add_argument("--z0-max", type=float, default=12.0)
parser.add_argument("--z0-step", type=float, default=0.5)
parser.add_argument("--delta-z", type=float, default=None)
parser.add_argument("--wedge-fraction", type=float, default=1.0)
parser.add_argument("--ell-min-ksz", type=float, default=100.0)
parser.add_argument("--band", type=float, nargs=2, default=[500, 1500], help="ell band for panels a, b")
parser.add_argument("--name", default=None, help="output name (default wedge_survival, or wedge_survival_symlog with --symlog)")
parser.add_argument("--symlog", action="store_true",
                    help="symlog y axes (linear near zero, log beyond); writes a separate *_symlog file")
parser.add_argument("--linthresh", type=float, nargs=3, default=[0.005, 2.0, 0.02],
                    help="symlog linear threshold for panels (a) r, (b) S/N, (c) r_ell")
args = parser.parse_args()
if args.name is None:
    args.name = "wedge_survival_symlog" if args.symlog else "wedge_survival"

cfg = load_config(args.config)
cache_dir, plot_dir = cfg["paths"]["cache_dir"], cfg["paths"]["plot_dir"]
os.makedirs(plot_dir, exist_ok=True)
seeds = cfg["simulation"]["random_seeds"]
delta_z = args.delta_z if args.delta_z is not None else cfg["cross_correlation"]["delta_z"]
k_par_min = cfg["cross_correlation"]["k_par_min"]
tag = (f"_dz{delta_z:g}_kp{k_par_min:g}_wf{args.wedge_fraction:g}_lmin{args.ell_min_ksz:g}"
       f"_z{args.z0_min:g}-{args.z0_max:g}-{args.z0_step:g}")

per_seed, used = [], []
for s in seeds:
    p = os.path.join(cache_dir, f"seed_{s}", f"cross_corr_wedge_stitched_seed{s}{tag}.npy")
    if os.path.exists(p):
        per_seed.append(np.load(p, allow_pickle=True).item())
        used.append(s)
assert per_seed, f"no cross_corr_wedge_stitched_seed*{tag}.npy found -- run 04c first (same options)"
nS = len(per_seed)
z0s = np.array(sorted(per_seed[0].keys()))
ell = np.array([per_seed[0][z]["ell"] for z in z0s])
nb = ell.shape[1]
print(f"Loaded {nS} seeds, {len(z0s)} z0 values, tag {tag}")

STATS = ["lin_none", "lin_kpar", "lin_wedge", "sq_none", "sq_kpar", "sq_wedge"]
LAB = {
    "lin_none": "21cm linear, no filter",
    "lin_kpar": r"21cm linear, $k_\parallel$ filter",
    "lin_wedge": "21cm linear, wedge filter",
    "sq_none": "21cm$^2$, no filter",
    "sq_kpar": r"21cm$^2$, $k_\parallel$ filter (Zhou)",
    "sq_wedge": "21cm$^2$, wedge filter",
}
STY = {"lin_none": ("C0", "-"), "lin_kpar": ("C0", "--"), "lin_wedge": ("C0", ":"),
       "sq_none": ("C3", "-"), "sq_kpar": ("C3", "--"), "sq_wedge": ("C3", ":")}

R, RN = {}, {}   # (nS, nz, nb)
for st in STATS:
    a = np.full((nS, len(z0s), nb), np.nan)
    an = np.full_like(a, np.nan)
    for i, r in enumerate(per_seed):
        for j, z in enumerate(z0s):
            e = r[z]
            den = np.sqrt(e["stats"][st]["P_T"] * e["P_K"])
            a[i, j] = e["stats"][st]["C"] / den
            if "C_null" in e["stats"][st]:
                an[i, j] = e["stats"][st]["C_null"] / den
    R[st], RN[st] = a, an
frac = {f: np.array([per_seed[0][z]["frac_modes"][f] for z in z0s]) for f in ("none", "kpar", "wedge")}
slope = np.array([per_seed[0][z]["wedge_slope"] for z in z0s])


def band(arr, lo, hi):
    """per-seed band mean of r -> (nS, nz)"""
    out = np.full((arr.shape[0], len(z0s)), np.nan)
    for j in range(len(z0s)):
        m = (ell[j] >= lo) & (ell[j] < hi)
        if m.any():
            out[:, j] = np.nanmean(arr[:, j, m], axis=1)
    return out


def msem(b):
    return np.nanmean(b, axis=0), np.nanstd(b, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(b), axis=0))


# our mean reionization history (cached by 07/08, or built here)
hp = os.path.join(plot_dir, "reion_history_stitched.npz")
if os.path.exists(hp):
    h = np.load(hp); zh, xe_all = h["z_arr"], h["xe"]
else:
    rows, zh = [], None
    for s in seeds:
        p = os.path.join(cache_dir, f"seed_{s}", f"stitched_lightcone_seed{s}.npz")
        if os.path.exists(p):
            d = np.load(p); zh = np.asarray(d["z_arr"], float)
            rows.append(1.0 - np.asarray(d["xH_box"]).mean(axis=(0, 1)))
    xe_all = np.array(rows); np.savez(hp, z_arr=zh, xe=xe_all)
o = np.argsort(zh); zh, xe_mean = zh[o], xe_all[:, o].mean(axis=0)
xe_of_z = lambda z: np.interp(z, zh, xe_mean)
z_of_xe = lambda x: np.interp(x, xe_mean[::-1], zh[::-1])
XE = [0.2, 0.5, 0.9]
IZ = [int(np.argmin(np.abs(z0s - z_of_xe(x)))) for x in XE]

lo, hi = args.band
B = {st: band(R[st], lo, hi) for st in STATS}
BN = {st: band(RN[st], lo, hi) for st in STATS}

# ── table ──────────────────────────────────────────────────────────────────
lines = []
def P(s=""):
    print(s); lines.append(s)

P(f"FILTER SURVIVAL  ({nS} seeds; ell band {lo:g}-{hi:g}; r = correlation coeff. of the 2D maps)")
P(f"Mode fraction kept at z0 ~ {z0s[IZ[1]]:.1f}: none {frac['none'][IZ[1]]:.3f}, k_par {frac['kpar'][IZ[1]]:.3f}, "
  f"wedge {frac['wedge'][IZ[1]]:.4f} (wedge slope {slope[IZ[1]]:.2f})")
P(f"{'statistic':28s} {'x_e (z0)':>12s} {'r':>9s} {'s.e.m.':>9s} {'S/N':>7s} {'null r':>9s} {'null S/N':>9s} {'survival r/r_none':>18s}")
for st in STATS:
    kind = st.split("_")[0]
    for x, j in zip(XE, IZ):
        b, bn, b0 = B[st][:, j], BN[st][:, j], B[f"{kind}_none"][:, j]
        mu, se = np.nanmean(b), np.nanstd(b, ddof=1) / np.sqrt(nS)
        mun, sen = np.nanmean(bn), np.nanstd(bn, ddof=1) / np.sqrt(nS)
        m0 = np.nanmean(b0)
        sv = mu / m0 if abs(m0) > 0 else np.nan
        P(f"{LAB[st].replace('$','').replace(chr(92),''):28s} {f'{x} ({z0s[j]:.1f})':>12s} {mu:9.4f} {se:9.4f} "
          f"{mu/se:7.2f} {mun:9.4f} {mun/sen:9.2f} {sv:18.2f}")
    P("")
P("Reading it: S/N >> null S/N means a detected correlation; survival = r(filtered)/r(no filter) at the same z0.")
P("A linear statistic that is large unfiltered but ~ null filtered is the 'foreground filter kills it' result.")
with open(os.path.join(plot_dir, f"{args.name}_table.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"✓ Saved: {os.path.join(plot_dir, args.name + '_table.txt')}")

# ── figure ─────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from ksz2_21cm.plotting.style import apply_global_style
apply_global_style()
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 9,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})

fig, axes = plt.subplots(1, 3, figsize=(19, 5.4), gridspec_kw={"width_ratios": [1, 1, 1]})
axA, axB, axC = axes

# null band: the larger of the null scatters among the six stats
nullsem = np.nanmax([msem(BN[st])[1] for st in STATS], axis=0)
axA.fill_between(z0s, -nullsem, nullsem, color="0.8", alpha=0.8, lw=0, label=r"null $\pm1\sigma$ (other-seed kSZ$^2$)")
axB.axhspan(-2, 2, color="0.88", lw=0)
for st in STATS:
    c, ls = STY[st]
    mu, se = msem(B[st])
    axA.plot(z0s, mu, color=c, ls=ls, lw=2, label=LAB[st])
    axA.fill_between(z0s, mu - se, mu + se, color=c, alpha=0.12, lw=0)
    axB.plot(z0s, mu / se, color=c, ls=ls, lw=2, label=LAB[st])
for ax in (axA, axB):
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlim(z0s.max(), z0s.min())
    ax.set_xlabel(r"$z_0$ (chunk centre)")
_ab=np.nanpercentile(np.abs(np.concatenate([msem(B[st])[0]/msem(B[st])[1] for st in STATS])), 90)
axB.set_ylim(-max(6.0, 1.6*_ab), max(6.0, 1.6*_ab))
axA.set_ylabel(rf"$\langle r\rangle_{{\ell={lo:g}-{hi:g}}}$  (correlation coefficient)")
axB.set_ylabel(r"S/N = $\langle r\rangle$ / s.e.m. over seeds  (grey: $|{\rm S/N}|<2$)")
axA.legend(frameon=True, framealpha=0.85, loc="best", fontsize=8)
sec = axA.secondary_xaxis("top", functions=(xe_of_z, z_of_xe))
sec.set_xlabel(r"$\bar{x}_{\rm HII}$")
for lab, ax in (("(a)", axA), ("(b)", axB), ("(c)", axC)):
    ax.text(0.02, 0.97, lab, transform=ax.transAxes, va="top", ha="left", fontsize=13)

j = IZ[1]
for st in STATS:
    c, ls = STY[st]
    mu = np.nanmean(R[st][:, j], axis=0)
    se = np.nanstd(R[st][:, j], axis=0, ddof=1) / np.sqrt(nS)
    axC.plot(ell[j], mu, color=c, ls=ls, lw=2, label=LAB[st])
    axC.fill_between(ell[j], mu - se, mu + se, color=c, alpha=0.12, lw=0)
nm = np.nanstd(np.concatenate([RN[st][:, j] for st in STATS], axis=0), axis=0)
axC.fill_between(ell[j], -nm, nm, color="0.8", alpha=0.8, lw=0)
axC.axhline(0, color="gray", lw=0.8)
axC.set_xscale("log")
axC.set_xlim(max(100, ell[j][np.isfinite(ell[j])].min()), 8000)
axC.set_xlabel(r"$\ell$")
axC.set_ylabel(r"$r_\ell$")
axC.set_title(rf"at $z_0$ = {z0s[j]:.1f} ($\bar{{x}}_e \approx 0.5$)", fontsize=11)
if args.symlog:
    for ax_, lt in zip((axA, axB, axC), args.linthresh):
        lo_, hi_ = ax_.get_ylim()
        m_ = max(abs(lo_), abs(hi_))
        ax_.set_yscale("symlog", linthresh=lt)
        ax_.set_ylim(-m_, m_)
fig.suptitle(rf"Does kSZ$^2\times$21cm survive foreground filtering?  ({nS} seeds; $k_\parallel>{k_par_min:g}$; "
             rf"wedge slope = {args.wedge_fraction:g} $\chi H/c(1+z)$)", fontsize=13)
fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(plot_dir, f"{args.name}.{ext}")
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ Saved: {out}")
plt.close(fig)

payload = {"z0": z0s, "ell": ell, "seeds": np.array(used), "band": np.array([lo, hi]), "xe_of_z0": xe_of_z(z0s),
           "frac_modes_none": frac["none"], "frac_modes_kpar": frac["kpar"], "frac_modes_wedge": frac["wedge"],
           "wedge_slope": slope}
for st in STATS:
    payload[f"r_{st}"] = R[st]; payload[f"rnull_{st}"] = RN[st]
np.savez(os.path.join(plot_dir, f"{args.name}_data.npz"), **payload)
print(f"✓ Saved: {os.path.join(plot_dir, args.name + '_data.npz')}")
