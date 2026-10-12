#!/usr/bin/env python
# =============================================================================
# 07_compare_reion_history_zhou.py
#
# Two figures comparing OUR stitched reionization history with Zhou+25's:
#
#   A) x_HII(z): ours (mean +/- 1 sigma over seeds, volume-averaged from the
#      stitched xH_box) vs Zhou+25 Fig. 1 (Fiducial; Late/Short with --all-zhou).
#      Zhou's curves are digitized from the vector paths of their PDF
#      (src/ksz2_21cm/plotting/zhou25_history.py).
#
#   B) D_ell vs xbar_HII (instead of z): ours (from the 06 output
#      <name>_data.npz, small) vs Zhou+25 Fig. 5 mapped through THEIR history.
#      This removes any difference in reionization history from the comparison:
#      if the sign flip / minimum still sit at different x_HII, the cause is
#      topology / filters, not timing. Zhou on its own right axis, zeros aligned
#      (shape comparison only; amplitudes are not like-for-like).
#
# Reads: <cache_dir>/seed_*/stitched_lightcone_seed*.npz (A, once; the result is
# cached in <plot_dir>/reion_history_stitched.npz so reruns are instant) and
# <plot_dir>/Dcross_vs_z_stitched_data.npz (B; written by 06).
#
# Run (interactive node):
#   python scripts/07_compare_reion_history_zhou.py --config configs/variants/stitched_256.yaml
# =============================================================================

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ksz2_21cm.utils.config import load_config
from ksz2_21cm.plotting.zhou25_data import ZHOU_FIG5
from ksz2_21cm.plotting.zhou25_history import ZHOU_HISTORY, zhou_xhii_of_z

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/variants/stitched_256.yaml")
parser.add_argument("--d-data", default="Dcross_vs_z_stitched_data.npz",
                    help="06 output (in plot_dir) used for figure B")
parser.add_argument("--all-zhou", action="store_true", help="also draw Zhou's Late and Short histories in A")
parser.add_argument("--recompute", action="store_true", help="ignore the cached reion_history_stitched.npz")
args = parser.parse_args()

cfg = load_config(args.config)
cache_dir = cfg["paths"]["cache_dir"]
plot_dir = cfg["paths"]["plot_dir"]
os.makedirs(plot_dir, exist_ok=True)
seeds = cfg["simulation"]["random_seeds"]

# ── A: our history ────────────────────────────────────────────────────────────
hist_path = os.path.join(plot_dir, "reion_history_stitched.npz")
if os.path.exists(hist_path) and not args.recompute:
    h = np.load(hist_path)
    z_arr, xe = h["z_arr"], h["xe"]
    print(f"Loaded cached history: {xe.shape[0]} seeds ({hist_path})")
else:
    rows, z_arr = [], None
    for s in seeds:
        p = os.path.join(cache_dir, f"seed_{s}", f"stitched_lightcone_seed{s}.npz")
        if not os.path.exists(p):
            print(f"  seed {s}: no stitched lightcone -- skipping")
            continue
        d = np.load(p)
        z_s = np.asarray(d["z_arr"], dtype=float)
        if z_arr is None:
            z_arr = z_s
        assert np.allclose(z_s, z_arr), "z_arr differs between seeds"
        rows.append(1.0 - np.asarray(d["xH_box"]).mean(axis=(0, 1)))
    assert rows, "No stitched lightcones found -- run 01 first."
    xe = np.array(rows)
    np.savez(hist_path, z_arr=z_arr, xe=xe)
    print(f"Computed history from {xe.shape[0]} seeds; cached to {hist_path}")

order = np.argsort(z_arr)               # z_arr is ascending by contract; be safe
z_arr, xe = z_arr[order], xe[:, order]
xe_mean, xe_std = xe.mean(axis=0), xe.std(axis=0, ddof=1) if xe.shape[0] > 1 else np.zeros(xe.shape[1])


def z_at(x_target, z, x):
    """z where x(z) = x_target; x DEcreases with z, so reverse both for np.interp."""
    return float(np.interp(x_target, x[::-1], z[::-1]))


zf = ZHOU_HISTORY["Fiducial"]
print("\n   z     ours    Zhou(Fid)   ours - Zhou")
for zz in [6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]:
    xo = float(np.interp(zz, z_arr, xe_mean)); xz = float(zhou_xhii_of_z(zz))
    print(f"  {zz:5.1f}  {xo:6.3f}   {xz:6.3f}     {xo - xz:+.3f}")
print("\nz at x_HII = 0.1 / 0.5 / 0.9:")
print("  ours  : " + " / ".join(f"{z_at(v, z_arr, xe_mean):.2f}" for v in (0.1, 0.5, 0.9)))
print("  Zhou  : " + " / ".join(f"{z_at(v, zf[:, 0], zf[:, 1]):.2f}" for v in (0.1, 0.5, 0.9)))

import matplotlib.pyplot as plt
from ksz2_21cm.plotting.style import apply_global_style
apply_global_style()
plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "legend.fontsize": 12})

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.fill_between(z_arr, xe_mean - xe_std, xe_mean + xe_std, color="C0", alpha=0.25, lw=0)
ax.plot(z_arr, xe_mean, color="C0", lw=2.2, label=f"this work, stitched 21cmFAST ({xe.shape[0]} seeds)")
ax.plot(zf[:, 0], zf[:, 1], "k--", lw=2, label="Zhou+25 Fiducial (zreion; Fig. 1)")
if args.all_zhou:
    for nm, c in (("Late", "C1"), ("Short", "C2")):
        a = ZHOU_HISTORY[nm]
        ax.plot(a[:, 0], a[:, 1], ":", color=c, lw=1.5, label=f"Zhou+25 {nm}")
ax.axhline(0.5, color="gray", lw=0.8, ls=":")
ax.set_xlim(12.5, 5)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel(r"Redshift $z$")
ax.set_ylabel(r"$\bar{x}_{\rm HII}(z)$")
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(plot_dir, f"reion_history_vs_zhou.{ext}")
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ Saved: {out}")
plt.close(fig)

# ── B: D_ell vs xbar_HII ──────────────────────────────────────────────────────
dpath = os.path.join(plot_dir, args.d_data)
if not os.path.exists(dpath):
    print(f"\n(no {dpath}; run 06 first to get figure B)")
    sys.exit(0)
dd = np.load(dpath)
ells = [int(v) for v in dd["ells"]]
xh_ours = np.asarray(dd["xhii"], dtype=float)

fig, axes = plt.subplots(len(ells), 1, figsize=(8.5, 3.4 * len(ells)), sharex=True)
axes = np.atleast_1d(axes)
for ax, L in zip(axes, ells):
    T = np.asarray(dd[f"D_ell{L}"], dtype=float)
    n = np.sum(np.isfinite(T), axis=0)
    mean = np.nanmean(T, axis=0)
    sem = np.nanstd(T, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1))
    ax.fill_between(xh_ours, mean - sem, mean + sem, color="C0", alpha=0.25, lw=0)
    ax.plot(xh_ours, mean, "-o", color="C0", ms=4, label=f"this work ({T.shape[0]} seeds, mean ± s.e.m.)")
    hs, ls = ax.get_legend_handles_labels()
    if L in ZHOU_FIG5:
        zz = ZHOU_FIG5[L]
        xz = zhou_xhii_of_z(zz[:, 0])                  # Zhou D mapped through THEIR history
        o = np.argsort(xz)
        axr = ax.twinx()
        hz, = axr.plot(xz[o], zz[o, 1], "--", color="k", lw=1.5, label="Zhou+25 Fig.5, right axis")
        ymr = 1.1 * np.max(np.abs(zz[:, 1]))
        axr.set_ylim(-ymr, ymr)
        axr.set_ylabel(r"Zhou+25 $D_\ell$ [$\mu$K$^4$]", fontsize=12)
        yml = 1.1 * np.nanmax(np.abs(np.concatenate([mean - sem, mean + sem])))
        if np.isfinite(yml) and yml > 0:
            ax.set_ylim(-yml, yml)
        hs.append(hz); ls.append(hz.get_label())
    ax.axhline(0, color="gray", lw=0.8)
    ax.text(0.02, 0.93, rf"$\ell = {L}$", transform=ax.transAxes, va="top", fontsize=13)
    ax.set_ylabel(r"$D_\ell$ [$\mu$K$^4$" + (r"$/\chi^2$]" if str(dd["chi_norm"]) == "z0" else "]"))
    ax.legend(hs, ls, loc="upper right", frameon=False, fontsize=10)
axes[-1].set_xlabel(r"$\bar{x}_{\rm HII}$ at the chunk centre")
axes[0].set_title(r"$D_\ell^{\mathrm{kSZ}^2\times 21\mathrm{cm}^2}$ vs ionized fraction (history-independent comparison)", fontsize=13)
fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(plot_dir, f"Dcross_vs_xHII_vs_zhou.{ext}")
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ Saved: {out}")
