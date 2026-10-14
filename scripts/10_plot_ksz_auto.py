#!/usr/bin/env python
# =============================================================================
# 10_plot_ksz_auto.py
#
# kSZ AUTO power spectrum of the stitched kSZ maps (the maps are the integral over the whole
# z = 20 -> 4 window, i.e. through the end of reionization), N seeds:
#   (a) D_ell [uK^2]: every seed (thin), mean (thick) and median (dashed), 16-84% band, with
#       Ma+18's Fig. 2 top panel (their reionization kSZ, digitized) for comparison;
#   (b) seed-to-seed fractional scatter vs ell, against the Gaussian-field expectation
#       sqrt(2 / N_modes): how noisy a single seed's spectrum is, i.e. why single-seed
#       curves are jagged.
# Uses the vendored, validated ksz_map_to_Dl (D_ell = l(l+1)C/2pi * T_CMB^2, l = k chi_eff).
# Small (reads N maps of 256x256): fine on usha if the maps are copied, or on swarm.
#
#   python scripts/10_plot_ksz_auto.py --config configs/variants/stitched_256.yaml
#   python scripts/10_plot_ksz_auto.py --maps-dir data/stitched_256_kSZ_maps --box-len 800 --out-dir data/plots/2026-10-14
# =============================================================================

import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ksz2_21cm.ksz.lightcone_integral import ksz_map_to_Dl
from ksz2_21cm.plotting.ma18_ksz_auto import MA18_KSZ_AUTO

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/variants/stitched_256.yaml")
parser.add_argument("--maps-dir", default=None, help="folder with kSZ_map_stitched_seed*.npy (default: from config)")
parser.add_argument("--box-len", type=float, default=None, help="box side [Mpc] (default: from config)")
parser.add_argument("--out-dir", default=None, help="where to write the plot (default: plot_dir of config)")
parser.add_argument("--n-seeds", type=int, default=10)
parser.add_argument("--chi-eff", type=float, default=8504.0, help="chi_eff [Mpc] for ell = k chi (validated value)")
parser.add_argument("--name", default="ksz_auto")
args = parser.parse_args()

if args.maps_dir is None or args.box_len is None or args.out_dir is None:
    from ksz2_21cm.utils.config import load_config
    cfg = load_config(args.config)
    maps_dir = args.maps_dir or os.path.join(cfg["paths"]["cache_dir"], "kSZ_maps")
    box = args.box_len or float(cfg["simulation"]["box_len"])
    out_dir = args.out_dir or cfg["paths"]["plot_dir"]
else:
    maps_dir, box, out_dir = os.path.expanduser(args.maps_dir), args.box_len, os.path.expanduser(args.out_dir)
os.makedirs(out_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(maps_dir, "kSZ_map_stitched_seed*.npy")),
               key=lambda p: int(re.search(r"seed(\d+)", p).group(1)))[: args.n_seeds]
assert files, f"no kSZ maps in {maps_dir}"
curves, seeds = [], []
for p in files:
    m = np.load(p)
    ell, D, _ = ksz_map_to_Dl(m, box, chi_Mpc=args.chi_eff)
    curves.append(D)
    seeds.append(int(re.search(r"seed(\d+)", p).group(1)))
D = np.array(curves)
N = np.load(files[0]).shape[0]
pix = box / N
dk = 2 * np.pi / (N * pix)
kx = np.fft.fftshift(np.fft.fftfreq(N)) * N * dk
kg = np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)
kb = np.logspace(np.log10(dk), np.log10(kg.max() * 0.9), 35)
nmodes = np.array([np.sum((kg >= kb[i]) & (kg < kb[i + 1])) for i in range(34)], float)
ok = np.isfinite(ell) & (np.arange(len(ell)) < 34)
mean, med = np.nanmean(D, axis=0), np.nanmedian(D, axis=0)
lo, hi = np.nanpercentile(D, 16, axis=0), np.nanpercentile(D, 84, axis=0)
frac = np.nanstd(D, axis=0, ddof=1) / mean
exp_frac = np.sqrt(2.0 / np.maximum(nmodes[: len(ell)], 1))

print(f"{len(seeds)} seeds {seeds}, box {box:g} Mpc, chi_eff {args.chi_eff:g}")
print("\n   ell     mean D   median D   16%     84%    scatter/mean  Gaussian expect")
for L in (100, 200, 500, 1000, 2000, 3000, 5000):
    i = int(np.nanargmin(np.abs(ell - L)))
    print(f"{ell[i]:7.0f}  {mean[i]:8.3f}  {med[i]:8.3f}  {lo[i]:6.3f}  {hi[i]:6.3f}   {frac[i]:8.2f}      {exp_frac[i]:8.2f}")

import matplotlib.pyplot as plt
from ksz2_21cm.plotting.style import apply_global_style
apply_global_style()
plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "legend.fontsize": 9,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.5, 8.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
for i, c in enumerate(D):
    ax.plot(ell, c, color="C0", lw=0.8, alpha=0.35, label=f"individual seeds ({len(seeds)})" if i == 0 else None)
ax.fill_between(ell, lo, hi, color="C0", alpha=0.18, lw=0, label="16-84% of seeds")
ax.plot(ell, mean, color="C0", lw=2.5, label="mean")
ax.plot(ell, med, color="navy", lw=2.0, ls="--", label="median")
ax.errorbar(MA18_KSZ_AUTO[:, 0], MA18_KSZ_AUTO[:, 1], MA18_KSZ_AUTO[:, 2], fmt="s", ms=4, color="C3",
            mfc="none", elinewidth=0.9, label="Ma+18 reionization kSZ (Fig. 2 top, digitized)")
ax.axvspan(ell[ok].min(), 700, color="0.9", zorder=0)
ax.text(0.02, 0.96, r"grey: $\ell<700$, unvalidated in our stitched box; Ma's low-$\ell$ is boundary-inflated",
        transform=ax.transAxes, va="top", fontsize=8.5)
ax.set_xscale("log")
ax.set_ylim(0, None)
ax.set_ylabel(r"$D_\ell^{\rm kSZ}=\ell(\ell+1)C_\ell/2\pi\ [\mu{\rm K}^2]$")
ax.legend(frameon=True, loc="center right")
ax2.plot(ell, frac, color="C0", lw=2, label="seed-to-seed scatter / mean")
ax2.plot(ell, exp_frac, "k:", lw=1.6, label=r"Gaussian expectation $\sqrt{2/N_{\rm modes}}$")
ax2.set_yscale("log")
ax2.set_xlabel(r"$\ell$")
ax2.set_ylabel("fractional scatter")
ax2.legend(frameon=True, fontsize=9)
ax.set_xlim(max(60, ell[ok].min() * 0.9), 1.2e4)
fig.suptitle(f"kSZ auto-spectrum of the stitched maps ({len(seeds)} seeds, through z = 4)", fontsize=12)
fig.tight_layout()
for ext in ("pdf", "png"):
    out = os.path.join(out_dir, f"{args.name}.{ext}")
    fig.savefig(out, bbox_inches="tight")
    print(f"✓ Saved: {out}")
np.savez(os.path.join(out_dir, f"{args.name}_data.npz"), ell=ell, D=D, seeds=np.array(seeds), box=box, chi_eff=args.chi_eff,
         nmodes=nmodes)
