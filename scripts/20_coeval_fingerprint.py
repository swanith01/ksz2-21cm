#!/usr/bin/env python
# =============================================================================
# 20_coeval_fingerprint.py
#
# Machine-to-machine reproduction check for 21cmFAST: run ONE coeval box with the same call the
# stitched pipeline uses (simulate/coeval_fields.run_coeval_fields, same seed, same parameters) and
# print a fingerprint -- means, standard deviations and a fixed set of individual voxels of every
# field. Run it on swarm and on CSD3 with the same --seed/--z and compare the two printouts line by
# line: identical (or equal to ~1e-6 relative, from compiler/FFTW differences) means the two
# installations produce the same realisation, so CSD3 runs can be compared seed by seed with swarm's.
#
#   python scripts/20_coeval_fingerprint.py --config configs/variants/stitched_256.yaml --seed 1 --z 8.0
#
# Writes the fingerprint to --out (default: fingerprint_seed<S>_z<Z>_<hostname>.txt in the work dir)
# and deletes the 21cmFAST .h5 cache it creates unless --keep-h5.
# =============================================================================

import argparse
import glob
import os
import socket
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ksz2_21cm.utils.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/variants/stitched_256.yaml")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--z", type=float, default=8.0)
ap.add_argument("--work-dir", default=None, help="21cmFAST cache (default <cache_dir>/../fingerprint_h5)")
ap.add_argument("--keep-h5", action="store_true")
ap.add_argument("--out", default=None)
args = ap.parse_args()

cfg = load_config(args.config)
HII_DIM, BOX_LEN = int(cfg["simulation"]["hii_dim"]), float(cfg["simulation"]["box_len"])
cache_dir = os.path.expanduser(cfg["paths"]["cache_dir"])
work = args.work_dir or os.path.join(os.path.dirname(cache_dir.rstrip("/")), "fingerprint_h5")
os.makedirs(work, exist_ok=True)

import py21cmfast as p21c
from ksz2_21cm.simulate.coeval_fields import run_coeval_fields

delta, xH, vx, vy, vz, dTb = run_coeval_fields(args.z, HII_DIM, BOX_LEN, work, random_seed=args.seed)

host = socket.gethostname()
lines = [f"# 21cmFAST fingerprint  seed={args.seed}  z={args.z}  HII_DIM={HII_DIM}  BOX_LEN={BOX_LEN}",
         f"# host={host}  py21cmfast={p21c.__version__}  numpy={np.__version__}"]
rng = np.random.default_rng(12345)                      # same voxel list on every machine
vox = rng.integers(0, HII_DIM, size=(8, 3))
for name, f in (("density", delta), ("xH", xH), ("vz_cm_s", vz), ("brightness_temp", dTb)):
    a = np.asarray(f, np.float64)
    lines.append(f"{name:16s} mean={a.mean():+.9e}  std={a.std():.9e}  min={a.min():+.6e}  max={a.max():+.6e}")
    lines.append(f"{'':16s} voxels=" + " ".join(f"{a[i, j, k]:+.7e}" for i, j, k in vox))
text = "\n".join(lines)
print(text)
out = args.out or os.path.join(work, f"fingerprint_seed{args.seed}_z{args.z:g}_{host.split('.')[0]}.txt")
with open(out, "w") as fh:
    fh.write(text + "\n")
print(f"saved {out}")

if not args.keep_h5:
    for p in glob.glob(os.path.join(work, "*.h5")):
        try:
            os.remove(p)
        except OSError:
            pass
