#!/usr/bin/env python
# =============================================================================
# 04c_compute_cross_corr_wedge_stitched.py
#
# Foreground-filter survival test: does kSZ^2 x 21cm (21cm NOT squared) survive the
# k_par / wedge filter, compared with the Zhou-style squared statistic? For each z0:
# 21cm chunk (z0 +/- delta_z/2) filtered with none / k_par>k_par_min / horizon wedge, then
# either the central LOS cell (linear) or square-then-project (Zhou) is crossed with kSZ^2
# (ell<100 removed first) and with a different seed's kSZ^2 (null). Results feed
# 09_plot_wedge_pedagogy.py (correlation coefficient r, comparable across statistics).
#
# Run on an INTERACTIVE node (not the login node):
#   python scripts/04c_compute_cross_corr_wedge_stitched.py --config configs/variants/stitched_256.yaml
# Cache: <cache_dir>/seed_<N>/cross_corr_wedge_stitched_seed<N><tag>.npy (small).
# =============================================================================

import argparse
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ksz2_21cm.utils.config import load_config, ensure_dirs
from ksz2_21cm.utils.cosmology import get_cosmology
from ksz2_21cm.correlation.cross_corr_wedge_worker_stitched import (
    compute_cross_corr_wedge_for_seed_stitched,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/variants/stitched_256.yaml")
parser.add_argument("--z0-min", type=float, default=6.0)
parser.add_argument("--z0-max", type=float, default=12.0)
parser.add_argument("--z0-step", type=float, default=0.5)
parser.add_argument("--delta-z", type=float, default=None,
                    help="21cm chunk width (default: cross_correlation.delta_z = Zhou's 2)")
parser.add_argument("--wedge-fraction", type=float, default=1.0,
                    help="wedge slope = fraction x chi H /(c(1+z)); 1 = horizon wedge, smaller = beam-limited")
parser.add_argument("--ell-min-ksz", type=float, default=100.0,
                    help="kSZ ell below this is removed before squaring (Ma Sec. 3.3)")
parser.add_argument("--chi-ksz", type=float, default=8504.0,
                    help="chi_eff [Mpc] mapping the kSZ map's k to ell for that cut (ksz-pipeline value)")
parser.add_argument("--workers", type=int, default=None)
args = parser.parse_args()

cfg = load_config(args.config)
ensure_dirs(cfg)
cache_dir = cfg["paths"]["cache_dir"]
SEEDS = cfg["simulation"]["random_seeds"]
HII_DIM = cfg["simulation"]["hii_dim"]
BOX_LEN = cfg["simulation"]["box_len"]
delta_z = args.delta_z if args.delta_z is not None else cfg["cross_correlation"]["delta_z"]
k_par_min = cfg["cross_correlation"]["k_par_min"]
cosmo = get_cosmology(cfg)

assert cfg["simulation"]["hii_dim"] == cfg["coeval_ksz"]["hii_dim"]
assert cfg["simulation"]["box_len"] == cfg["coeval_ksz"]["box_len"]

npix_side = HII_DIM
pix_size_Mpc = float(BOX_LEN) / npix_side
pix_area = pix_size_Mpc**2
dk = 2 * np.pi / (npix_side * pix_size_Mpc)
kx = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
ky = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
kgrid = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
k_bins = np.logspace(np.log10(dk), np.log10(kgrid.max() * 0.9), 35)   # same as 04
k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
kx_1d = np.fft.fftfreq(npix_side, d=pix_size_Mpc) * 2 * np.pi
ky_1d = np.fft.fftfreq(npix_side, d=pix_size_Mpc) * 2 * np.pi

z0_list = [float(z) for z in np.arange(args.z0_min, args.z0_max + 1e-9, args.z0_step)]
tag = f"_dz{delta_z:g}_kp{k_par_min:g}_wf{args.wedge_fraction:g}_lmin{args.ell_min_ksz:g}_z{args.z0_min:g}-{args.z0_max:g}-{args.z0_step:g}"

print(f"  z0 grid      : {args.z0_min}..{args.z0_max} step {args.z0_step} ({len(z0_list)} values)")
print(f"  21cm chunk   : dz = {delta_z}; filters: none / k_par>{k_par_min} / wedge (fraction {args.wedge_fraction})")
print(f"  kSZ ell cut  : ell < {args.ell_min_ksz} removed before squaring (chi_ksz = {args.chi_ksz})")
print(f"  cache tag    : {tag}")

maps_dir = os.path.join(cache_dir, "kSZ_maps")
avail = []
for s in SEEDS:
    npz = os.path.join(cache_dir, f"seed_{s}", f"stitched_lightcone_seed{s}.npz")
    mp_ = os.path.join(maps_dir, f"kSZ_map_stitched_seed{s}.npy")
    if os.path.exists(npz) and os.path.exists(mp_):
        avail.append((s, npz, mp_))
    else:
        print(f"  x seed {s}: missing lightcone or kSZ map -- skipping")
assert len(avail) >= 2, "need >= 2 seeds (the null uses a neighbouring seed's kSZ map)"

worker_args = []
for i, (s, npz, mpath) in enumerate(avail):
    null_path = avail[(i + 1) % len(avail)][2]
    worker_args.append((
        s, npz, np.load(mpath), np.load(null_path), cache_dir,
        npix_side, pix_size_Mpc, pix_area, kx_1d, ky_1d, kgrid, k_bins, k_centers,
        k_par_min, delta_z, z0_list, cosmo, args.ell_min_ksz, args.chi_ksz,
        args.wedge_fraction, tag,
    ))

ncpu = int(os.environ.get("NCPUS", os.environ.get("PBS_NCPUS", os.cpu_count() or 4)))
n_workers = args.workers or max(1, min(ncpu, len(worker_args), 15))
print(f"\nDISPATCHING {len(worker_args)} SEEDS -- {n_workers} concurrent workers (NCPUS={ncpu})")

t0 = time.time()
done = 0
with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("fork")) as ex:
    futs = {ex.submit(compute_cross_corr_wedge_for_seed_stitched, a): a[0] for a in worker_args}
    for fut in as_completed(futs):
        s = futs[fut]
        done += 1
        try:
            _, res, status = fut.result()
            print(f"  [{done:2d}/{len(worker_args)}] seed {s:3d}: {status}   ({(time.time()-t0)/60:.1f} min)")
        except Exception as e:
            print(f"  [{done:2d}/{len(worker_args)}] seed {s:3d}: crashed: {e}")

print("\nDone. Next: python scripts/09_plot_wedge_pedagogy.py --config", args.config)
