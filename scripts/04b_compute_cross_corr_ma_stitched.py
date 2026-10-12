#!/usr/bin/env python
# =============================================================================
# 04b_compute_cross_corr_ma_stitched.py
#
# Ma+2018-style cross-spectra on the stitched lightcones (reads the SAME inputs
# as 04: the per-seed stitched .npz and the kSZ maps from 02):
#
#     kSZ   x 21cm  (both linear)        -- is it zero?  (Ma: small, not zero)
#     kSZ^2 x 21cm  (21cm not squared)   -- Ma Figs. 2/3 bottom panels
#
# with a cross-seed NULL for each (21cm of seed i x kSZ of seed i+1), on a dense
# grid of z0 so script 08 can pick the redshifts where x_e = 0.2 / 0.5 / 0.9.
#
# Run on an INTERACTIVE node (not the login node):
#   python scripts/04b_compute_cross_corr_ma_stitched.py --config configs/variants/stitched_256.yaml
# Cache: <cache_dir>/seed_<N>/cross_corr_ma_stitched_seed<N><tag>.npy (small).
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
from ksz2_21cm.correlation.cross_corr_ma_worker_stitched import (
    compute_cross_corr_ma_for_seed_stitched,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/variants/stitched_256.yaml")
parser.add_argument("--z0-min", type=float, default=5.5)
parser.add_argument("--z0-max", type=float, default=15.0)
parser.add_argument("--z0-step", type=float, default=0.25)
parser.add_argument("--slab-dz", type=float, default=None,
                    help="width of the 'slab' variant (default: cross_correlation.delta_z)")
parser.add_argument("--narrow-dz", type=float, default=0.2,
                    help="width of the 'narrow' 21cm slab (default 0.2, ~60 Mpc)")
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
slab_dz = args.slab_dz if args.slab_dz is not None else cfg["cross_correlation"]["delta_z"]
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
tag = f"_dz{slab_dz:g}_nd{args.narrow_dz:g}_lmin{args.ell_min_ksz:g}_z{args.z0_min:g}-{args.z0_max:g}-{args.z0_step:g}"

print(f"  z0 grid      : {args.z0_min}..{args.z0_max} step {args.z0_step} ({len(z0_list)} values)")
print(f"  21cm maps    : slice (1 cell), narrow (dz={args.narrow_dz}), slab (dz={slab_dz})")
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
        slab_dz, args.narrow_dz, z0_list, cosmo, args.ell_min_ksz, args.chi_ksz, tag,
    ))

ncpu = int(os.environ.get("NCPUS", os.environ.get("PBS_NCPUS", os.cpu_count() or 4)))
n_workers = args.workers or max(1, min(ncpu, len(worker_args), 15))
print(f"\nDISPATCHING {len(worker_args)} SEEDS -- {n_workers} concurrent workers (NCPUS={ncpu})")

t0 = time.time()
done = 0
with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("fork")) as ex:
    futs = {ex.submit(compute_cross_corr_ma_for_seed_stitched, a): a[0] for a in worker_args}
    for fut in as_completed(futs):
        s = futs[fut]
        done += 1
        try:
            _, res, status = fut.result()
            print(f"  [{done:2d}/{len(worker_args)}] seed {s:3d}: {status}   ({(time.time()-t0)/60:.1f} min)")
        except Exception as e:
            print(f"  [{done:2d}/{len(worker_args)}] seed {s:3d}: crashed: {e}")

print("\nDone. Next: python scripts/08_plot_ma18_style.py --config", args.config)
