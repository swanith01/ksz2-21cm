#!/usr/bin/env python
# =============================================================================
# 05_compare_native_vs_stitched.py
#
# Correctness check for the vendored stitched pipeline: compares
# D_cross(ell) from the native-lightcone pathway (cross_corr_sq_seed*.npy,
# from 04_compute_cross_corr_sq.py) against the box-stitched pathway
# (cross_corr_sq_stitched_seed*.npy, from 04_compute_cross_corr_sq_stitched.py)
# at matching seeds and chunk centres.
#
# NOTE: if --native-config and --stitched-config differ in hii_dim/box_len,
# this is comparing method AND resolution at once -- the script detects and
# prints that explicitly so it's never silently conflated with a real
# methodology discrepancy.
#
# Run:
#   python scripts/05_compare_native_vs_stitched.py \
#       --native-config configs/fiducial.yaml \
#       --stitched-config configs/variants/stitched_256.yaml
# =============================================================================

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ksz2_21cm.utils.config import load_config
from ksz2_21cm.plotting.style import apply_global_style, save_pdf_png

parser = argparse.ArgumentParser()
parser.add_argument("--native-config", default="configs/fiducial.yaml")
parser.add_argument("--stitched-config", default="configs/variants/stitched_256.yaml")
args = parser.parse_args()

cfg_native   = load_config(args.native_config)
cfg_stitched = load_config(args.stitched_config)
apply_global_style()

hii_native, box_native     = cfg_native["simulation"]["hii_dim"], cfg_native["simulation"]["box_len"]
hii_stitched, box_stitched = cfg_stitched["simulation"]["hii_dim"], cfg_stitched["simulation"]["box_len"]

if hii_native != hii_stitched or box_native != box_stitched:
    print("="*70)
    print("WARNING: native and stitched configs use DIFFERENT geometry:")
    print(f"  native   : HII_DIM={hii_native}  BOX_LEN={box_native}")
    print(f"  stitched : HII_DIM={hii_stitched}  BOX_LEN={box_stitched}")
    print("Any discrepancy below conflates the method change (native vs")
    print("stitched lightcone construction) with the resolution change.")
    print("Treat this as a rough sanity check, not a clean methodology test,")
    print("until a native run exists at matching resolution.")
    print("="*70)

seeds_native   = set(cfg_native["simulation"]["random_seeds"])
seeds_stitched = set(cfg_stitched["simulation"]["random_seeds"])
common_seeds   = sorted(seeds_native & seeds_stitched)
print(f"\nCommon seeds: {common_seeds}")
if not common_seeds:
    sys.exit("No seeds in common between the two configs — nothing to compare.")

cache_native   = cfg_native["paths"]["cache_dir"]
cache_stitched = cfg_stitched["paths"]["cache_dir"]
plot_dir       = cfg_stitched["paths"]["plot_dir"]

per_seed = {}
for seed in common_seeds:
    p_native   = os.path.join(cache_native,   f"seed_{seed}", f"cross_corr_sq_seed{seed}.npy")
    p_stitched = os.path.join(cache_stitched, f"seed_{seed}", f"cross_corr_sq_stitched_seed{seed}.npy")
    if not os.path.exists(p_native):
        print(f"  seed {seed}: missing native cache at {p_native} — skipping")
        continue
    if not os.path.exists(p_stitched):
        print(f"  seed {seed}: missing stitched cache at {p_stitched} — skipping")
        continue
    d_native   = np.load(p_native,   allow_pickle=True).item()
    d_stitched = np.load(p_stitched, allow_pickle=True).item()
    per_seed[seed] = (d_native, d_stitched)

print(f"Seeds with both pathways available: {sorted(per_seed.keys())}")
if not per_seed:
    sys.exit("Nothing to compare yet — run the stitched job first.")

common_z0 = sorted(set.intersection(*[
    set(d_n.keys()) & set(d_s.keys()) for d_n, d_s in per_seed.values()
]))
print(f"Common chunk centres (z0): {common_z0}\n")

cmap = None
try:
    import matplotlib.pyplot as plt
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(common_z0)))
except Exception:
    pass

ratio_summary = {}
for z0 in common_z0:
    ratios = []
    for seed, (d_n, d_s) in per_seed.items():
        ell_n, D_n = d_n[z0]["ell"], d_n[z0]["D_cross"]
        ell_s, D_s = d_s[z0]["ell"], d_s[z0]["D_cross"]
        lo = max(ell_n.min(), ell_s.min())
        hi = min(ell_n.max(), ell_s.max())
        mask = (ell_n >= lo) & (ell_n <= hi) & np.isfinite(D_n)
        if mask.sum() < 2:
            continue
        D_s_on_native = np.interp(ell_n[mask], ell_s, D_s)
        valid = np.isfinite(D_s_on_native) & (D_n[mask] != 0)
        ratios.append(D_s_on_native[valid] / D_n[mask][valid])
    if ratios:
        all_r = np.concatenate(ratios)
        ratio_summary[z0] = (np.nanmean(all_r), np.nanstd(all_r))
        print(f"  z0={z0:5.2f}:  stitched/native D_cross ratio = "
              f"{ratio_summary[z0][0]:.3f} ± {ratio_summary[z0][1]:.3f}  "
              f"(n_overlap_pts={len(all_r)})")


def _draw(ax):
    for i, z0 in enumerate(common_z0):
        seed = next(iter(per_seed))
        d_n, d_s = per_seed[seed]
        if z0 not in d_n or z0 not in d_s:
            continue
        c = cmap[i] if cmap is not None else None
        ax.plot(d_n[z0]["ell"], d_n[z0]["D_cross"], '--', color=c, alpha=0.8,
                 label=f"native z0={z0}" if i == 0 else None)
        ax.plot(d_s[z0]["ell"], d_s[z0]["D_cross"], '-', color=c, alpha=0.8,
                 label=f"stitched z0={z0}" if i == 0 else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell^{\mathrm{cross}}$")
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"Native (--) vs Stitched (—), seed {next(iter(per_seed))}")


save_pdf_png(_draw, plot_dir, "native_vs_stitched_Dcross",
             title="Native vs Stitched: D_cross comparison")
print(f"\n✓ Saved comparison plot -> {plot_dir}/native_vs_stitched_Dcross.pdf")
