#!/usr/bin/env python
# =============================================================================
# 01_run_lightcones.py
#
# Run (or load cached) 21cmFAST lightcone simulations for every seed in the
# config. The heavy py21cmfast call itself lives in
# src/ksz2_21cm/simulate/lightcone_worker.py — this script only handles
# config, worker/core allocation, dispatch, and the sanity-check print-outs.
#
# Extracted from CELL 1, 1a, 1c, and CELL 2 of kSZ_Squared_21cm_11Jun_CLUSTER.py.
# CELL 2b (per-seed field plots, commented out in the original) has moved to
# notebooks/exploratory/ — see that directory's README.
#
# Run:
#   conda activate ksz2-21cm
#   python scripts/01_run_lightcones.py --config configs/fiducial.yaml
# =============================================================================

import argparse
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import py21cmfast as p21c

from ksz2_21cm.utils.config import load_config, ensure_dirs
from ksz2_21cm.simulate.lightcone_worker import run_or_load_seed

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/fiducial.yaml")
parser.add_argument("--threads-per-worker", type=int, default=None,
                    help="Override cfg['simulation']['n_threads']. If not "
                         "given, uses the config value.")
args = parser.parse_args()

cfg = load_config(args.config)
ensure_dirs(cfg)

cache_dir    = cfg["paths"]["cache_dir"]
RANDOM_SEEDS = cfg["simulation"]["random_seeds"]
N_SEEDS      = len(RANDOM_SEEDS)
HII_DIM      = cfg["simulation"]["hii_dim"]
BOX_LEN      = cfg["simulation"]["box_len"]
z_min        = cfg["simulation"]["z_min"]
z_max        = cfg["simulation"]["z_max"]

print(f"\n=== PARAMETER SETUP ===")
print(f"HII_DIM = {HII_DIM}   BOX_LEN = {BOX_LEN:.0f} Mpc   z = [{z_min}, {z_max}]")
print(f"Seeds: {RANDOM_SEEDS}  (N={N_SEEDS})")

# ── Core / worker allocation ─────────────────────────────────────────────────
N_TOTAL_CORES = int(os.environ.get('PBS_NCPUS', os.cpu_count() or 32))
DESIRED_THREADS_PER_WORKER = (args.threads_per_worker
                              if args.threads_per_worker is not None
                              else cfg["simulation"]["n_threads"])
N_WORKERS = max(1, N_TOTAL_CORES // DESIRED_THREADS_PER_WORKER)
N_WORKERS = min(N_WORKERS, N_SEEDS)

print(f"Available cores : {N_TOTAL_CORES}")
print(f"Workers         : {N_WORKERS} (concurrent seeds)")
print(f"Threads/worker  : {DESIRED_THREADS_PER_WORKER}")

# ── Dispatch ──────────────────────────────────────────────────────────────────
lightcones    = {}
seed_metadata = {}
scan_start = time.time()
mp_ctx = mp.get_context("fork")  # CHECK: py21cmfast's CFFI globals don't survive fork
                                  # on every platform — switch to "spawn" if you see
                                  # CFFI-related crashes on a new cluster.

print(f"\nDISPATCHING {N_SEEDS} SEEDS — {N_WORKERS} concurrent workers")

with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp_ctx) as ex:
    futures = {}
    for seed in RANDOM_SEEDS:
        seed_cache_dir = os.path.join(cache_dir, f"seed_{seed}")
        fut = ex.submit(
            run_or_load_seed, seed, seed_cache_dir, z_min, z_max,
            HII_DIM, BOX_LEN, DESIRED_THREADS_PER_WORKER,
        )
        futures[fut] = seed

    completed = 0
    for fut in as_completed(futures):
        seed = futures[fut]
        try:
            seed_done, cache_file, sim_time, status = fut.result()
        except Exception as e:
            seed_done, cache_file, sim_time, status = seed, None, 0.0, f"crashed: {e}"
        completed += 1
        seed_metadata[seed_done] = {
            'cache_file': cache_file, 'sim_time': sim_time, 'status': status,
        }
        elapsed = (time.time() - scan_start) / 60
        msg = ("cached (instant load)" if status == "cached" else
               f"computed in {sim_time/60:.2f} min" if status == "computed" else
               f"\u2717 {status}")
        print(f"  [{completed:2d}/{N_SEEDS}] seed {seed_done:3d}: {msg}"
              f"   (elapsed: {elapsed:.1f} min)", flush=True)

print(f"\n\u2713 ALL WORKERS RETURNED — {time.time()-scan_start:.1f}s total")

# ── Load every cached lightcone into the main process (sanity check) ────────
print(f"\n=== LOADING LIGHTCONES INTO MAIN PROCESS ===")
for seed in RANDOM_SEEDS:
    meta = seed_metadata.get(seed, {})
    cache_file = meta.get('cache_file')
    if not cache_file or not os.path.exists(cache_file):
        print(f"  \u2717 seed {seed}: no cache file — skipping")
        continue
    try:
        lc = p21c.LightCone.read(cache_file)
        lightcones[seed] = lc
        z_nodes   = lc.node_redshifts[::-1]
        x_e_nodes = 1.0 - lc.global_xH[::-1]
        z_10 = z_nodes[np.argmin(np.abs(x_e_nodes - 0.1))]
        z_90 = z_nodes[np.argmin(np.abs(x_e_nodes - 0.9))]
        print(f"  \u2713 seed {seed:3d} loaded   "
              f"z(10%)={z_10:.2f}  z(90%)={z_90:.2f}  \u0394z={z_10-z_90:.2f}")
    except Exception as e:
        print(f"  \u2717 seed {seed}: load failed — {e}")

print(f"\n\u2713 {len(lightcones)}/{N_SEEDS} lightcones ready in {cache_dir}")
print("Next: python scripts/02_compute_ksz_maps.py")
