#!/usr/bin/env python
# =============================================================================
# 04d_compute_auto21_stitched.py
#
# 21cm auto power spectrum Delta^2_21(k) [mK^2] from the stitched lightcones, per seed and
# per redshift z0 (LOS slab z0 +/- slab_dz/2), plus the matter-power "neutral limit"
# reference from the same slab. Output feeds 11_plot_auto21.py.
#
# Run on a compute node (reads big per-seed .npz files; ~1.5 GB per worker):
#   python scripts/04d_compute_auto21_stitched.py --config configs/variants/stitched_256.yaml
# Cache: <cache_dir>/seed_<N>/auto21_stitched_seed<N><tag>.npy (small)
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
from ksz2_21cm.correlation.auto21_worker_stitched import compute_auto21_for_seed_stitched

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/variants/stitched_256.yaml")
parser.add_argument("--n-seeds", type=int, default=10, help="use the first N seeds of the config (default 10)")
parser.add_argument("--z0", type=float, nargs="+", default=[6, 7, 8, 9, 10, 11, 12])
parser.add_argument("--slab-dz", type=float, default=0.4, help="LOS slab width (default 0.4 ~ 100 Mpc)")
parser.add_argument("--kmin", type=float, default=0.02)
parser.add_argument("--kmax", type=float, default=1.0)
parser.add_argument("--nk", type=int, default=24)
parser.add_argument("--workers", type=int, default=None)
args = parser.parse_args()

cfg = load_config(args.config)
ensure_dirs(cfg)
cache_dir = cfg["paths"]["cache_dir"]
seeds = cfg["simulation"]["random_seeds"][: args.n_seeds]
cell = float(cfg["simulation"]["box_len"]) / cfg["simulation"]["hii_dim"]
kbins = np.logspace(np.log10(args.kmin), np.log10(args.kmax), args.nk + 1)
tag = f"_dz{args.slab_dz:g}_k{args.kmin:g}-{args.kmax:g}-{args.nk}_z" + "-".join(f"{z:g}" for z in args.z0)

print(f"  seeds {seeds}\n  z0 {args.z0}, slab dz = {args.slab_dz}, {args.nk} k-bins {args.kmin}..{args.kmax}\n  tag {tag}")
wargs = []
for s in seeds:
    p = os.path.join(cache_dir, f"seed_{s}", f"stitched_lightcone_seed{s}.npz")
    if os.path.exists(p):
        wargs.append((s, p, cache_dir, cell, list(args.z0), args.slab_dz, kbins, tag))
    else:
        print(f"  x seed {s}: no stitched lightcone -- skipping")
assert wargs, "no stitched lightcones found"

ncpu = int(os.environ.get("NCPUS", os.environ.get("PBS_NCPUS", os.cpu_count() or 4)))
nw = args.workers or max(1, min(ncpu, len(wargs)))
print(f"\nDISPATCHING {len(wargs)} SEEDS -- {nw} workers (NCPUS={ncpu})")
t0, done = time.time(), 0
with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("fork")) as ex:
    futs = {ex.submit(compute_auto21_for_seed_stitched, a): a[0] for a in wargs}
    for f in as_completed(futs):
        done += 1
        s = futs[f]
        try:
            _, r, st = f.result()
            print(f"  [{done:2d}/{len(wargs)}] seed {s:3d}: {st}   ({(time.time()-t0)/60:.1f} min)")
        except Exception as e:
            print(f"  [{done:2d}/{len(wargs)}] seed {s:3d}: crashed: {e}")
print("\nDone. Next: python scripts/11_plot_auto21.py --config", args.config)
