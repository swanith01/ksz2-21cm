#!/usr/bin/env python
# =============================================================================
# 04_compute_cross_corr_sq_stitched.py
#
# NEW SCRIPT (2026-10-06) -- SQUARED kSZ^2 x 21cm^2 cross-correlation
# (Zhou+25 style) on the vendored, box-stitched construction. Mirrors
# 04_compute_cross_corr_sq.py exactly except for which worker it dispatches
# to and where it gathers its per-seed inputs from (stitched .npz +
# kSZ_map_stitched_seed*.npy, both from scripts 01/02 above, instead of
# native LightCone_*.h5 + kSZ_map_z*.npy).
#
# Run (after 01_stitch_lightcones.py and 02_compute_ksz_maps_stitched.py):
#   conda activate ksz2-21cm
#   python scripts/04_compute_cross_corr_sq_stitched.py --config configs/variants/stitched_256.yaml
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
from ksz2_21cm.correlation.cross_corr_sq_worker_stitched import (
    compute_cross_corr_sq_for_seed_stitched,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/variants/stitched_256.yaml")
args = parser.parse_args()

cfg = load_config(args.config)
ensure_dirs(cfg)

cache_dir       = cfg["paths"]["cache_dir"]
RANDOM_SEEDS    = cfg["simulation"]["random_seeds"]
HII_DIM         = cfg["simulation"]["hii_dim"]
BOX_LEN         = cfg["simulation"]["box_len"]
k_par_min       = cfg["cross_correlation"]["k_par_min"]
delta_z         = cfg["cross_correlation"]["delta_z"]
z_chunk_centres = cfg["cross_correlation"]["z_chunk_centres"]
cosmo           = get_cosmology(cfg)

# GUARD: same reasoning as 02_compute_ksz_maps_stitched.py -- the stitched
# .npz was built at coeval_ksz.hii_dim/box_len; this script's FFT grid must
# use the identical geometry or ell/k conversions become silently wrong.
assert cfg["simulation"]["hii_dim"] == cfg["coeval_ksz"]["hii_dim"], (
    "simulation.hii_dim != coeval_ksz.hii_dim -- these must match")
assert cfg["simulation"]["box_len"] == cfg["coeval_ksz"]["box_len"], (
    "simulation.box_len != coeval_ksz.box_len -- these must match")

npix_side    = HII_DIM
box_size_Mpc = float(BOX_LEN)
pix_size_Mpc = box_size_Mpc / npix_side
pix_area     = pix_size_Mpc**2

dk    = 2 * np.pi / (npix_side * pix_size_Mpc)
kx    = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
ky    = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
kgrid = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
k_bins    = np.logspace(np.log10(dk), np.log10(kgrid.max() * 0.9), 35)
k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
kx_1d = np.fft.fftfreq(npix_side, d=pix_size_Mpc) * 2 * np.pi
ky_1d = np.fft.fftfreq(npix_side, d=pix_size_Mpc) * 2 * np.pi

print(f"  Foreground filter : k_par > {k_par_min}")
print(f"  Chunk width       : dz = {delta_z}")
print(f"  Chunk centres     : {list(z_chunk_centres)}")

# ── Gather npz_path + kSZ_map per seed ───────────────────────────────────────
kSZ_maps_dir = os.path.join(cache_dir, "kSZ_maps")
worker_args = []
for seed in RANDOM_SEEDS:
    npz_path = os.path.join(cache_dir, f"seed_{seed}",
                             f"stitched_lightcone_seed{seed}.npz")
    map_path = os.path.join(kSZ_maps_dir, f"kSZ_map_stitched_seed{seed}.npy")
    if not os.path.exists(npz_path) or not os.path.exists(map_path):
        print(f"  ✗ seed {seed}: missing stitched lightcone or kSZ map — skipping")
        continue
    kSZ_map = np.load(map_path)
    worker_args.append((
        seed, npz_path, kSZ_map, cache_dir,
        npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
        kx_1d, ky_1d, kgrid, k_bins, k_centers,
        k_par_min, delta_z, z_chunk_centres, cosmo,
    ))

N_TOTAL_CORES = int(os.environ.get('PBS_NCPUS', os.cpu_count() or 32))
N_WORKERS = max(1, min(N_TOTAL_CORES // 16, len(worker_args)))
print(f"\nDISPATCHING {len(worker_args)} SEEDS — {N_WORKERS} concurrent workers")

mp_ctx = mp.get_context("fork")
cross_corr_results_sq_all = {}
scan_start = time.time()

with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp_ctx) as ex:
    futures = {ex.submit(compute_cross_corr_sq_for_seed_stitched, a): a[0]
               for a in worker_args}
    completed = 0
    for fut in as_completed(futures):
        seed = futures[fut]
        completed += 1
        try:
            seed_done, ccr, status = fut.result()
            if ccr is not None:
                cross_corr_results_sq_all[seed_done] = ccr
            elapsed = (time.time() - scan_start) / 60
            print(f"  [{completed:2d}/{len(worker_args)}] seed {seed_done:3d}: "
                  f"{status}   (elapsed: {elapsed:.1f} min)")
        except Exception as e:
            print(f"  [{completed:2d}/{len(worker_args)}] seed {seed:3d}: crashed: {e}")

print(f"\n✓ kSZ^2 x 21cm^2 (stitched) cross-correlations ready: "
      f"{len(cross_corr_results_sq_all)}/{len(RANDOM_SEEDS)} seeds")
print("Cache: <cache_dir>/seed_<N>/cross_corr_sq_stitched_seed<N>.npy")
print("Next: compare D_cross/ell against the native-lightcone pathway's "
      "cross_corr_sq_seed*.npy at the same astro params (same seeds 1-3), "
      "then: python paper/figure_scripts/overlay_zhou25.py "
      "--config configs/variants/stitched_256.yaml  (once that script is "
      "pointed at the _stitched cache filename too)")
