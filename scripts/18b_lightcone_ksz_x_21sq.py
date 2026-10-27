#!/usr/bin/env python
# =============================================================================
# 18b_lightcone_ksz_x_21sq.py
#
# Stitched-lightcone kSZ (NOT squared) x 21cm^2 (Zhou's k_par-filtered, squared, chunk-averaged 21cm map),
# for one or more seeds: the lightcone side of the kSZ x 21cm^2 null test (script 18 gives the coeval side).
# The statistic is odd in the velocity (flip every v and the kSZ map changes sign, 21cm^2 does not), so it
# must average to zero: anything significant flags a problem.
#
# The 21cm^2 map is built exactly as in correlation/cross_corr_sq_worker_stitched.py (chunk z0 +- dz/2,
# 3D k_par > k_par_min filter, square, LOS mean); the kSZ map is the cached 02 map, NOT squared.
# Output: <cache_dir>/seed_<N>/cross_corr_ksz_x_21sq_stitched_seed<N>.npy  {z0: {ell, k_centers, C}}
#
#   python scripts/18b_lightcone_ksz_x_21sq.py --config configs/variants/stitched_256.yaml --seeds 1 2 3
# =============================================================================

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ksz2_21cm.correlation import coeval_direct as cd
from ksz2_21cm.utils.config import load_config
from ksz2_21cm.utils.cosmology import get_cosmology

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/variants/stitched_256.yaml")
ap.add_argument("--seeds", type=int, nargs="+", default=None)
ap.add_argument("--force", action="store_true", help="recompute even if the output exists")
args = ap.parse_args()

cfg = load_config(args.config)
cache_dir = os.path.expanduser(cfg["paths"]["cache_dir"])
N, L = int(cfg["simulation"]["hii_dim"]), float(cfg["simulation"]["box_len"])
cell = L / N
cc = cfg["cross_correlation"]
dz, kpm, z0s = float(cc["delta_z"]), float(cc["k_par_min"]), [float(z) for z in cc["z_chunk_centres"]]
cosmo = get_cosmology(cfg)
dk = 2 * np.pi / L
kx = np.fft.fftshift(np.fft.fftfreq(N)) * N * dk
k_bins = np.logspace(np.log10(dk), np.log10(np.sqrt(2) * np.abs(kx).max() * 0.9), 35)   # as 04 / 04b
bn = cd.Binner(N, cell, k_bins)

for s in (args.seeds or cfg["simulation"]["random_seeds"]):
    out = os.path.join(cache_dir, f"seed_{s}", f"cross_corr_ksz_x_21sq_stitched_seed{s}.npy")
    npz = os.path.join(cache_dir, f"seed_{s}", f"stitched_lightcone_seed{s}.npz")
    kmap = os.path.join(cache_dir, "kSZ_maps", f"kSZ_map_stitched_seed{s}.npy")
    if os.path.exists(out) and not args.force:
        print(f"seed {s}: exists, skipped")
        continue
    if not (os.path.exists(npz) and os.path.exists(kmap)):
        print(f"seed {s}: missing stitched lightcone or kSZ map, skipped")
        continue
    d = np.load(npz)
    bt, z_arr = d["brightness_temp"], np.asarray(d["z_arr"], float)
    FK = bn.fft(np.load(kmap))
    res = {}
    for z0 in z0s:
        idx = np.where((z_arr >= z0 - dz / 2) & (z_arr <= z0 + dz / 2))[0]
        if len(idx) < 3:
            continue
        T = np.asarray(bt[:, :, idx], np.float64)
        X = (cd.kpar_filter_los(T, cell, kpm) ** 2).mean(axis=2)
        chi = float(cosmo.comoving_distance(z0).value)
        res[float(z0)] = {"z0": float(z0), "ell": bn.k_centers * chi, "k_centers": bn.k_centers, "chi_z0": chi,
                          "C": bn.cross(FK, bn.fft(X))}
    tmp = out + ".tmp.npy"
    np.save(tmp, res, allow_pickle=True)
    os.replace(tmp, out)
    print(f"seed {s}: {len(res)} chunks -> {out}")
