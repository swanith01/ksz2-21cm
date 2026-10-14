#!/usr/bin/env python
# =============================================================================
# 13_plot_filter_geometry.py
#
# What the foreground filters in the wedge test actually do, in Fourier space.
# No simulation data needed (only the config and the cosmology), so it is quick and
# safe to run anywhere.
#
#   top row   : the (k_perp, k_par) plane at one z0. Shaded = modes KEPT.
#               none | k_par > k_par_min | horizon wedge | beam-limited wedge (fraction f)
#   bottom    : fraction of the 3D modes kept vs z0 for each filter
#
# The masks are the same ones used in cross_corr_wedge_worker_stitched.py:
#     kpar  : k_par > k_par_min
#     wedge : k_par > max(k_par_min, s * k_perp),  s = f * chi(z0) H(z0) / (c (1+z0))
# with k_perp the 2D magnitude of (kx, ky) and k_par from the slab's LOS cells.
#
#   python scripts/13_plot_filter_geometry.py --config configs/variants/stitched_256.yaml
# =============================================================================
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ksz2_21cm.utils.config import load_config
from ksz2_21cm.utils.cosmology import get_cosmology

C_KMS = 299792.458
p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/variants/stitched_256.yaml")
p.add_argument("--z0", type=float, default=8.0, help="redshift for the top row")
p.add_argument("--fractions", type=float, nargs=2, default=[1.0, 0.3],
               help="wedge fractions: horizon, beam-limited")
p.add_argument("--delta-z", type=float, default=None)
p.add_argument("--name", default="filter_geometry")
args = p.parse_args()

cfg = load_config(args.config)
cosmo = get_cosmology(cfg)
N = cfg["simulation"]["hii_dim"]; L = cfg["simulation"]["box_len"]; cell = L / N
dz = args.delta_z if args.delta_z is not None else cfg["cross_correlation"]["delta_z"]
kpm = cfg["cross_correlation"]["k_par_min"]
plot_dir = cfg["paths"]["plot_dir"]; os.makedirs(plot_dir, exist_ok=True)

kx = np.fft.fftfreq(N, d=cell) * 2 * np.pi
kperp = np.sqrt(kx[:, None, None] ** 2 + kx[None, :, None] ** 2)

def n_los(z0):   # LOS cells in the slab z0 +/- dz/2
    d = cosmo.comoving_distance(z0 + dz / 2).value - cosmo.comoving_distance(z0 - dz / 2).value
    return int(round(d / cell))

def slope(z0, f):
    return f * cosmo.comoving_distance(z0).value * cosmo.H(z0).value / (C_KMS * (1 + z0))

def kept(z0, f):
    n = n_los(z0)
    kpar = np.abs(np.fft.fftfreq(n, d=cell) * 2 * np.pi)[None, None, :]
    return {"kpar": float(np.mean(kpar > kpm)),
            "wedge": float(np.mean(kpar > np.maximum(kpm, slope(z0, f) * kperp)))}

kmax = np.pi / cell
fh, fb = args.fractions
fr_h, fr_b = kept(args.z0, fh), kept(args.z0, fb)
panels = [
    ("no filter", None, 1.0),
    (r"$k_\parallel > %g$ (Zhou)" % kpm, "kpar", fr_h["kpar"]),
    ("horizon wedge  (slope %.2f)" % slope(args.z0, fh), slope(args.z0, fh), fr_h["wedge"]),
    ("beam-limited wedge  (slope %.2f)" % slope(args.z0, fb), slope(args.z0, fb), fr_b["wedge"]),
]

fig = plt.figure(figsize=(15, 7.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1], hspace=0.38, wspace=0.28)
kp = np.linspace(0, kmax, 400)
for i, (title, kind, frac) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor("0.88")                       # grey = removed
    if kind is None:
        ax.fill_between(kp, 0, kmax, color="tab:blue", alpha=0.35, lw=0)
    elif kind == "kpar":
        ax.fill_between(kp, kpm, kmax, color="tab:blue", alpha=0.35, lw=0)
    else:
        lo = np.maximum(kpm, kind * kp)
        ax.fill_between(kp, np.minimum(lo, kmax), kmax, color="tab:blue", alpha=0.35, lw=0)
        ax.plot(kp, np.minimum(kind * kp, kmax), color="tab:red", lw=1.2)
    ax.set_xlim(0, kmax); ax.set_ylim(0, kmax)
    ax.set_title(title, fontsize=10)
    ax.text(0.97, 0.05, "%.1f%% of modes kept" % (100 * frac) if frac < 0.995 else
            ("100% kept" if kind is None else "%.1f%% kept" % (100 * frac)),
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(fc="white", ec="0.7", alpha=0.9))
    ax.set_xlabel(r"$k_\perp$ [Mpc$^{-1}$]")
    if i == 0:
        ax.set_ylabel(r"$k_\parallel$ [Mpc$^{-1}$]")
fig.text(0.5, 0.945, "Blue = modes kept, grey = modes removed   (z$_0$ = %g, slab $\\Delta z$ = %g, %d LOS cells)"
         % (args.z0, dz, n_los(args.z0)), ha="center", fontsize=11)

zs = np.arange(6.0, 12.01, 0.5)
ax = fig.add_subplot(gs[1, :2])
ax.plot(zs, [kept(z, 1)["kpar"] for z in zs], color="tab:blue", lw=2, label=r"$k_\parallel > k_{\parallel,\min}$")
ax.plot(zs, [kept(z, fb)["wedge"] for z in zs], color="tab:orange", lw=2, label="beam-limited wedge (f=%g)" % fb)
ax.plot(zs, [kept(z, fh)["wedge"] for z in zs], color="tab:red", lw=2, label="horizon wedge (f=%g)" % fh)
ax.set_yscale("log"); ax.set_xlim(12, 6)
ax.set_xlabel(r"$z_0$"); ax.set_ylabel("fraction of 3D modes kept"); ax.legend(fontsize=9)
ax.set_title("how much survives, vs redshift", fontsize=10)

ax2 = fig.add_subplot(gs[1, 2:])
ax2.axis("off")
ax2.text(0, 0.95,
 "How to read this\n\n"
 "- The 21cm foregrounds sit at low k$_\\parallel$, so the k$_\\parallel$ filter removes only the\n"
 "  slab's lowest line-of-sight modes: almost everything else is kept.\n\n"
 "- The horizon wedge also removes every mode with k$_\\parallel$ below slope x k$_\\perp$.\n"
 "  The slope grows with the comoving distance, so at z~8 the wedge covers almost the\n"
 "  whole plane and only a few percent of modes are left.\n\n"
 "- A beam-limited wedge (smaller slope) keeps several times more modes.\n\n"
 "- Squaring the 21cm field first (Zhou+25) is what lets the signal survive the\n"
 "  k$_\\parallel$ filter; with the horizon wedge nothing is left in our setup.",
 va="top", fontsize=10, transform=ax2.transAxes)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(plot_dir, f"{args.name}.{ext}"), dpi=200, bbox_inches="tight")
print("fractions kept at z0=%g:  kpar %.3f | horizon wedge %.4f (slope %.2f) | f=%g wedge %.4f (slope %.2f)"
      % (args.z0, fr_h["kpar"], fr_h["wedge"], slope(args.z0, fh), fb, fr_b["wedge"], slope(args.z0, fb)))
print("n_los at z0:", n_los(args.z0))
print("Saved:", os.path.join(plot_dir, args.name + ".{pdf,png}"))
