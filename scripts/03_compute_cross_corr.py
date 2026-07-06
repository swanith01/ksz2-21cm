#!/usr/bin/env python
# =============================================================================
# 03_compute_cross_corr.py
#
# UNSQUARED kSZ x 21cm cross-correlation (Ma+18 style), parallelised across
# seeds. This is a diagnostic / comparison statistic — see CELL 9 in
# notebooks/exploratory/ for why this statistic dies under foreground-wedge
# filtering and the pipeline moved to the SQUARED kSZ^2 x 21cm^2 statistic
# (scripts/04_compute_cross_corr_sq.py) for the actual paper result.
#
# Extracted from CELL 7 of kSZ_Squared_21cm_11Jun_CLUSTER.py. The heavy
# per-seed FFT/binning work is in
# src/ksz2_21cm/correlation/cross_corr_worker.py (unchanged).
#
# Run (after 01_run_lightcones.py and 02_compute_ksz_maps.py):
#   conda activate ksz2-21cm
#   python scripts/03_compute_cross_corr.py --config configs/fiducial.yaml
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
from ksz2_21cm.correlation.cross_corr_worker import compute_cross_corr_for_seed

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/fiducial.yaml")
args = parser.parse_args()

cfg = load_config(args.config)
ensure_dirs(cfg)

cache_dir    = cfg["paths"]["cache_dir"]
RANDOM_SEEDS = cfg["simulation"]["random_seeds"]
HII_DIM      = cfg["simulation"]["hii_dim"]
BOX_LEN      = cfg["simulation"]["box_len"]
z_obs        = cfg["ksz"]["z_obs"]

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

print(f"=== MAP PROPERTIES ===\n  {npix_side}x{npix_side} px, "
      f"{box_size_Mpc:.1f} Mpc box, {pix_size_Mpc:.3f} Mpc/px")

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
        seed, lc_files[0], kSZ_map, z_obs, cache_dir,
        npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
        dk, kgrid, k_bins, k_centers,
    ))

N_TOTAL_CORES = int(os.environ.get('PBS_NCPUS', os.cpu_count() or 32))
N_WORKERS = max(1, min(N_TOTAL_CORES // 16, len(worker_args)))
print(f"\nDISPATCHING {len(worker_args)} SEEDS — {N_WORKERS} concurrent workers")

mp_ctx = mp.get_context("fork")
cross_corr_results_all = {}
scan_start = time.time()

with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp_ctx) as ex:
    futures = {ex.submit(compute_cross_corr_for_seed, a): a[0] for a in worker_args}
    completed = 0
    for fut in as_completed(futures):
        seed = futures[fut]
        completed += 1
        try:
            seed_done, ccr, status = fut.result()
            if ccr is not None:
                cross_corr_results_all[seed_done] = ccr
            elapsed = (time.time() - scan_start) / 60
            print(f"  [{completed:2d}/{len(worker_args)}] seed {seed_done:3d}: "
                  f"{status}   (elapsed: {elapsed:.1f} min)")
        except Exception as e:
            print(f"  [{completed:2d}/{len(worker_args)}] seed {seed:3d}: crashed: {e}")

print(f"\n\u2713 kSZ x 21cm (unsquared) cross-correlations ready: "
      f"{len(cross_corr_results_all)}/{len(RANDOM_SEEDS)} seeds")
print("Next: python scripts/04_compute_cross_corr_sq.py")
