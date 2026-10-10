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
import glob
import os
import sys
import time
import zipfile
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
parser.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="process only these seeds (default: all seeds in the config)")
parser.add_argument("--workers", type=int, default=None,
                    help="concurrent seeds (default: cores // n_threads)")
parser.add_argument("--keep-h5", action="store_true",
                    help="keep 21cmFAST's per-snapshot .h5 box cache (~30 GB/seed at 256^3). "
                         "Default: delete it once the stitched .npz is written and verified.")
args = parser.parse_args()

cfg = load_config(args.config)
ensure_dirs(cfg)

cache_dir    = cfg["paths"]["cache_dir"]
RANDOM_SEEDS = args.seeds if args.seeds else cfg["simulation"]["random_seeds"]
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


_NPZ_KEYS = {"density", "xH_box", "velocity_z", "brightness_temp", "z_arr", "pos_axis"}


def _verify_npz(npz_path):
    """True only if the .npz is a complete zip (CRC-checked, streaming -- no big
    memory use) holding every expected array. Guards against a truncated file
    left by a power cut / killed job, which the old 'touch z_arr' check could miss."""
    try:
        with zipfile.ZipFile(npz_path) as zf:
            if zf.testzip() is not None:
                return False
        return _NPZ_KEYS.issubset(set(np.load(npz_path).files))
    except Exception:
        return False


def _cleanup_h5(seed_cache_dir):
    """Delete 21cmFAST's per-snapshot box cache (*.h5, directly inside this
    seed's directory ONLY). At 256^3 these are ~32 GB/seed vs <1 GB for the
    stitched .npz that replaces them. Any seed can be regenerated from its
    seed number if a different field is ever needed."""
    n, freed = 0, 0
    for path in glob.glob(os.path.join(seed_cache_dir, "*.h5")):
        try:
            freed += os.path.getsize(path)
            os.remove(path)
            n += 1
        except OSError:
            pass
    return n, freed


def _stitch_one_seed(args_tuple):
    (seed, seed_cache_dir, z_snapshots, z_arr, hii_dim, box_len,
     n_threads, keep_h5) = args_tuple

    os.makedirs(seed_cache_dir, exist_ok=True)
    npz_path = os.path.join(seed_cache_dir, f"stitched_lightcone_seed{seed}.npz")
    tmp_path = npz_path + ".tmp.npz"     # np.savez appends .npz unless the name already ends in it

    if os.path.exists(tmp_path):         # leftover from an interrupted write
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if os.path.exists(npz_path) and _verify_npz(npz_path):
        if not keep_h5:
            _cleanup_h5(seed_cache_dir)  # also tidies seeds stitched before this cleanup existed
        return (seed, npz_path, 0.0, "cached")

    # Computing from scratch. Any .h5 here is stale (e.g. a run killed mid-write
    # by a power cut) and could be truncated -- start clean rather than let
    # py21cmfast try to read it.
    if not keep_h5:
        _cleanup_h5(seed_cache_dir)

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
            tmp_path,
            density=result["density"].astype(np.float32),
            xH_box=result["xH_box"].astype(np.float32),
            velocity_z=result["velocity_z"].astype(np.float32),
            brightness_temp=result["brightness_temp"].astype(np.float32),
            z_arr=result["z_arr"],
            pos_axis=result["pos_axis"],
        )
        os.replace(tmp_path, npz_path)   # atomic: npz_path is never half-written
        if not _verify_npz(npz_path):
            return (seed, None, time.time() - t0, "failed: written .npz did not verify")
        if not keep_h5:
            _cleanup_h5(seed_cache_dir)  # only reached after the .npz verified
        return (seed, npz_path, time.time() - t0, "computed")
    except Exception as e:
        return (seed, None, time.time() - t0, f"failed: {e}")


# PBS Pro exports NCPUS (not PBS_NCPUS); fall back to the node's full core count
# only outside a job. Use --workers to override explicitly.
N_TOTAL_CORES = int(os.environ.get('PBS_NCPUS') or os.environ.get('NCPUS')
                    or os.cpu_count() or 32)
N_WORKERS = args.workers if args.workers else max(1, min(N_TOTAL_CORES // N_THREADS, N_SEEDS))
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
            args.keep_h5,
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
