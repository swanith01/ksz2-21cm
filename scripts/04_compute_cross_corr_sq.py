#!/usr/bin/env python
# =============================================================================
# 04_compute_cross_corr_sq.py
#
# SQUARED kSZ^2 x 21cm^2 cross-correlation (Zhou+25 style) — this is the
# main result the paper, the Zhou+25 comparison figures (paper/figure_scripts/),
# and the SNR forecast (05_compute_snr_forecast.py) all depend on.
#
# Extracted from CELL 7b of kSZ_Squared_21cm_11Jun_CLUSTER.py, which ran
# this same per-seed loop SERIALLY, in-process. This script instead uses
# the new src/ksz2_21cm/correlation/cross_corr_sq_worker.py, parallelised
# across seeds the same way 03_compute_cross_corr.py already was — this is
# the one genuinely new piece of infrastructure from the cleanup, not just
# a copy-paste. Existing cross_corr_sq_seed*.npy caches are loaded as-is
# (cache format unchanged), so nothing needs to be recomputed just to adopt
# this script.
#
# Run (after 01_run_lightcones.py and 02_compute_ksz_maps.py):
#   conda activate ksz2-21cm
#   python scripts/04_compute_cross_corr_sq.py --config configs/fiducial.yaml
# =============================================================================

import argparse
import glob
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ksz2_21cm.utils.config import load_config, ensure_dirs
from ksz2_21cm.utils.cosmology import get_cosmology
from ksz2_21cm.correlation.cross_corr_sq_worker import compute_cross_corr_sq_for_seed

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/fiducial.yaml")
args = parser.parse_args()

cfg = load_config(args.config)
ensure_dirs(cfg)

cache_dir       = cfg["paths"]["cache_dir"]
RANDOM_SEEDS    = cfg["simulation"]["random_seeds"]
HII_DIM         = cfg["simulation"]["hii_dim"]
BOX_LEN         = cfg["simulation"]["box_len"]
z_obs           = cfg["ksz"]["z_obs"]
k_par_min       = cfg["cross_correlation"]["k_par_min"]
delta_z         = cfg["cross_correlation"]["delta_z"]
z_chunk_centres = cfg["cross_correlation"]["z_chunk_centres"]
cosmo           = get_cosmology(cfg)

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

# ── Gather cache_file + kSZ_map per seed ─────────────────────────────────────
kSZ_maps_dir = os.path.join(cache_dir, "kSZ_maps")
worker_args = []
for seed in RANDOM_SEEDS:
    lc_files = sorted(glob.glob(os.path.join(cache_dir, f"seed_{seed}", "LightCone_*.h5")))
    map_path = os.path.join(kSZ_maps_dir, f"kSZ_map_z{z_obs:.1f}_seed{seed}.npy")
    if not lc_files or not os.path.exists(map_path):
        print(f"  \u2717 seed {seed}: missing lightcone or kSZ map — skipping")
        continue
    kSZ_map = np.load(map_path)
    worker_args.append((
        seed, lc_files[0], kSZ_map, cache_dir,
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
    futures = {ex.submit(compute_cross_corr_sq_for_seed, a): a[0] for a in worker_args}
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

print(f"\n\u2713 kSZ^2 x 21cm^2 cross-correlations ready: "
      f"{len(cross_corr_results_sq_all)}/{len(RANDOM_SEEDS)} seeds")
print("Next: python scripts/05_compute_snr_forecast.py")
print("  or:  python paper/figure_scripts/overlay_zhou25.py")
