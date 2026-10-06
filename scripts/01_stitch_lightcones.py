#!/usr/bin/env python
# =============================================================================
# 01_stitch_lightcones.py
#
# NEW SCRIPT (2026-10-06) -- builds the kSZ+21cm lightcone via the vendored,
# box-stitched construction (src/ksz2_21cm/simulate/stitch_from_coeval.py,
# vendored from ksz-pipeline, extended to carry brightness_temp) instead of
# the native py21cmfast.run_lightcone() path in 01_run_lightcones.py.
#
# Does NOT replace 01_run_lightcones.py -- that script and its native-
# lightcone cache stay exactly as they are, so the old fiducial.yaml result
# stays reproducible and is the thing to diff this new pathway's output
# against. Run this against configs/variants/stitched_256.yaml, not
# fiducial.yaml.
#
# Per seed, this:
#   1. builds a uniform-comoving-distance LOS redshift grid z_arr spanning
#      coeval_ksz.z_snapshots (build_los_z_grid)
#   2. calls stitch_lightcone_from_coeval() with wrap_cycle_seed = the
#      seed itself (ksz-pipeline's own convention from its Nikolic
#      replication -- one fewer knob, deterministic per seed)
#   3. caches the result as one .npz per seed (density, xH_box,
#      velocity_z, brightness_temp, z_arr, pos_axis) -- this REPLACES the
#      old LightCone_*.h5 as this pathway's per-seed cache artifact.
#      Multi-hundred-MB to ~GB per seed at 256^3, same storage class as
#      the old h5 files -- lives in cache_dir (outside the repo), not
#      committed, same convention as before.
#
# Run:
#   conda activate ksz2-21cm
#   python scripts/01_stitch_lightcones.py --config configs/variants/stitched_256.yaml
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
from ksz2_21cm.simulate.stitch_from_coeval import (
    build_los_z_grid, stitch_lightcone_from_coeval,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/variants/stitched_256.yaml")
args = parser.parse_args()

cfg = load_config(args.config)
ensure_dirs(cfg)

cache_dir    = cfg["paths"]["cache_dir"]
RANDOM_SEEDS = cfg["simulation"]["random_seeds"]
N_SEEDS      = len(RANDOM_SEEDS)
HII_DIM      = cfg["coeval_ksz"]["hii_dim"]
BOX_LEN      = cfg["coeval_ksz"]["box_len"]
Z_SNAPSHOTS  = sorted(cfg["coeval_ksz"]["z_snapshots"])
N_THREADS    = cfg["simulation"]["n_threads"]

cell_size = BOX_LEN / HII_DIM
z_arr = build_los_z_grid(min(Z_SNAPSHOTS), max(Z_SNAPSHOTS), cell_size)

print(f"\n=== PARAMETER SETUP (stitched, box-based) ===")
print(f"HII_DIM = {HII_DIM}   BOX_LEN = {BOX_LEN:.0f} Mpc   "
      f"cell_size = {cell_size:.4f} Mpc")
print(f"z_snapshots: {len(Z_SNAPSHOTS)} points, "
      f"[{min(Z_SNAPSHOTS)}, {max(Z_SNAPSHOTS)}]")
print(f"z_arr (LOS pixels): {len(z_arr)} points, uniform comoving spacing")
print(f"Seeds: {RANDOM_SEEDS}  (N={N_SEEDS})")
print(f"wrap_cycle_seed = random_seed (per-seed convention, see header)")


def _stitch_one_seed(args_tuple):
    (seed, seed_cache_dir, z_snapshots, z_arr, hii_dim, box_len,
     n_threads) = args_tuple

    os.makedirs(seed_cache_dir, exist_ok=True)
    npz_path = os.path.join(seed_cache_dir, f"stitched_lightcone_seed{seed}.npz")

    if os.path.exists(npz_path):
        try:
            d = np.load(npz_path)
            _ = d["z_arr"]  # touch the file to confirm it's not truncated/corrupt
            return (seed, npz_path, 0.0, "cached")
        except Exception:
            pass  # corrupt -- fall through to recompute

    t0 = time.time()
    try:
        result = stitch_lightcone_from_coeval(
            z_snapshots=z_snapshots, z_arr=z_arr,
            HII_DIM=hii_dim, BOX_LEN=box_len,
            cache_dir=seed_cache_dir,
            N_THREADS=n_threads, random_seed=seed,
            wrap_cycle_seed=seed,
        )
        np.savez_compressed(
            npz_path,
            density=result["density"].astype(np.float32),
            xH_box=result["xH_box"].astype(np.float32),
            velocity_z=result["velocity_z"].astype(np.float32),
            brightness_temp=result["brightness_temp"].astype(np.float32),
            z_arr=result["z_arr"],
            pos_axis=result["pos_axis"],
        )
        return (seed, npz_path, time.time() - t0, "computed")
    except Exception as e:
        return (seed, None, time.time() - t0, f"failed: {e}")


N_TOTAL_CORES = int(os.environ.get('PBS_NCPUS', os.cpu_count() or 32))
N_WORKERS = max(1, min(N_TOTAL_CORES // N_THREADS, N_SEEDS))
print(f"\nAvailable cores : {N_TOTAL_CORES}")
print(f"Workers         : {N_WORKERS} (concurrent seeds)")
print(f"Threads/worker  : {N_THREADS}")

mp_ctx = mp.get_context("fork")
scan_start = time.time()
seed_metadata = {}

print(f"\nDISPATCHING {N_SEEDS} SEEDS — {N_WORKERS} concurrent workers")

with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp_ctx) as ex:
    futures = {}
    for seed in RANDOM_SEEDS:
        seed_cache_dir = os.path.join(cache_dir, f"seed_{seed}")
        fut = ex.submit(_stitch_one_seed, (
            seed, seed_cache_dir, Z_SNAPSHOTS, z_arr, HII_DIM, BOX_LEN, N_THREADS,
        ))
        futures[fut] = seed

    completed = 0
    for fut in as_completed(futures):
        seed = futures[fut]
        try:
            seed_done, npz_path, sim_time, status = fut.result()
        except Exception as e:
            seed_done, npz_path, sim_time, status = seed, None, 0.0, f"crashed: {e}"
        completed += 1
        seed_metadata[seed_done] = {'npz_path': npz_path, 'status': status}
        elapsed = (time.time() - scan_start) / 60
        msg = ("cached (instant load)" if status == "cached" else
               f"computed in {sim_time/60:.2f} min" if status == "computed" else
               f"✗ {status}")
        print(f"  [{completed:2d}/{N_SEEDS}] seed {seed_done:3d}: {msg}"
              f"   (elapsed: {elapsed:.1f} min)", flush=True)

print(f"\n✓ ALL WORKERS RETURNED — {time.time()-scan_start:.1f}s total")

# ── Sanity check: global_xH(z) from each seed's stitched field ──────────────
print(f"\n=== REIONIZATION HISTORY CHECK (from stitched xH_box) ===")
n_ready = 0
for seed in RANDOM_SEEDS:
    meta = seed_metadata.get(seed, {})
    npz_path = meta.get('npz_path')
    if not npz_path or not os.path.exists(npz_path):
        print(f"  ✗ seed {seed}: no stitched cache — skipping")
        continue
    d = np.load(npz_path)
    xH_box = d["xH_box"]
    z_arr_s = d["z_arr"]
    global_xH = xH_box.mean(axis=(0, 1))      # volume-averaged neutral fraction per LOS slice
    x_e = 1.0 - global_xH
    # z_arr_s is ASCENDING (build_los_z_grid's contract) and x_e DECREASES
    # as z_arr_s increases (more neutral at higher z) -- so x_e itself is
    # DESCENDING in this index order. np.interp requires its xp argument
    # ascending, so both must be reversed together before interpolating --
    # the exact reversed-array bug class this project has hit before
    # (find_z_obs.py's first draft, the validation_table.md chi-crossing
    # bug): unreversed, this silently returns z=20.0 for every threshold.
    # Verified synthetically before shipping this script.
    z_10 = np.interp(0.1, x_e[::-1], z_arr_s[::-1])
    z_90 = np.interp(0.9, x_e[::-1], z_arr_s[::-1])
    print(f"  ✓ seed {seed:3d}   z(x_e=10%)={z_10:.2f}  "
          f"z(x_e=90%)={z_90:.2f}")
    n_ready += 1

print(f"\n✓ {n_ready}/{N_SEEDS} stitched lightcones ready in {cache_dir}")
print("Next: python scripts/02_compute_ksz_maps_stitched.py "
      "--config configs/variants/stitched_256.yaml")
