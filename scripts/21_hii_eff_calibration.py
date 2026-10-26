#!/usr/bin/env python
# =============================================================================
# 21_hii_eff_calibration.py
#
# Reionization history x_e(z) of 21cmFAST for ONE value of HII_EFF_FACTOR (the constant ionizing
# efficiency zeta of the default model; 21cmFAST default = 30, our fiducial, z50 = 7.76), from coeval
# boxes at the production resolution (800 Mpc / 256^3, seed 1), so the HII_EFF values for the
# production runs can be chosen to hit target midpoints (the AMBER histories: z50 = 6.68 and 8.75).
#
# Same call as the stitched pipeline (simulate/coeval_fields.run_coeval_fields), with
# astro_params = {"HII_EFF_FACTOR": value}. In the default constant-efficiency model
# (USE_MASS_DEPENDENT_ZETA = False) HII_EFF_FACTOR is what sets zeta, so no flag_options are needed.
#
# Redshifts: the stitched pipeline's 29 snapshots (z = 4..20) plus 3.0 and 3.5, because a LATE
# history may not finish reionizing (x_e = 1 - 1e-4, the end of our patchy window) before z = 4.
#
# Output (small): <out-dir>/heff<value>_seed<S>.npz with z, x_e (volume-weighted), x_e_mass
# (mass-weighted, the convention of AMBER's z_mid), <Tb>. The .h5 cache is deleted at the end.
# Then: python scripts/21b_hii_eff_summary.py --in-dir <out-dir>
#
#   python scripts/21_hii_eff_calibration.py --hii-eff 30 --out-dir $R/ksz2-21cm-data/hii_eff_calib
#   (Slurm array over values: slurm/hii_eff_calib.sbatch)
# =============================================================================

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ksz2_21cm.utils.config import load_config

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/variants/stitched_256.yaml")
ap.add_argument("--hii-eff", type=float, required=True)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--extra-z", type=float, nargs="*", default=[3.0, 3.5], help="redshifts added below the config snapshots")
ap.add_argument("--out-dir", required=True)
ap.add_argument("--work-dir", default=None, help="21cmFAST .h5 cache (default <out-dir>/h5_heff<value>)")
ap.add_argument("--keep-h5", action="store_true")
args = ap.parse_args()

cfg = load_config(args.config)
HII_DIM, BOX_LEN = int(cfg["simulation"]["hii_dim"]), float(cfg["simulation"]["box_len"])
zs = sorted(set([float(z) for z in cfg["coeval_ksz"]["z_snapshots"]] + list(args.extra_z)))
tag = f"heff{args.hii_eff:g}_seed{args.seed}"
os.makedirs(args.out_dir, exist_ok=True)
work = args.work_dir or os.path.join(args.out_dir, f"h5_heff{args.hii_eff:g}")
os.makedirs(work, exist_ok=True)
out = os.path.join(args.out_dir, f"{tag}.npz")
print(f"HII_EFF_FACTOR = {args.hii_eff:g}, seed {args.seed}, {HII_DIM}^3 / {BOX_LEN} Mpc, {len(zs)} redshifts {zs[0]}..{zs[-1]}")

from ksz2_21cm.simulate.coeval_fields import run_coeval_fields

t0 = time.time()
xe, xe_m, tb = [], [], []
for z in zs:
    delta, xH, _vx, _vy, _vz, dTb = run_coeval_fields(z, HII_DIM, BOX_LEN, work, random_seed=args.seed,
                                                      astro_params={"HII_EFF_FACTOR": float(args.hii_eff)})
    xHII = 1.0 - np.asarray(xH, np.float64)
    w = 1.0 + np.asarray(delta, np.float64)
    xe.append(float(xHII.mean()))
    xe_m.append(float((w * xHII).sum() / w.sum()))
    tb.append(float(np.mean(dTb)))
    print(f"  z={z:5.2f}: x_e={xe[-1]:.6f}  (mass-weighted {xe_m[-1]:.6f})  <Tb>={tb[-1]:6.2f} mK   ({time.time() - t0:.0f}s)", flush=True)

np.savez(out, z=np.array(zs), x_e=np.array(xe), x_e_mass=np.array(xe_m), Tb_mean=np.array(tb),
         hii_eff=float(args.hii_eff), seed=args.seed, hii_dim=HII_DIM, box_len=BOX_LEN)
print(f"saved {out}  ({time.time() - t0:.0f}s)")
if not args.keep_h5:
    for p in glob.glob(os.path.join(work, "*.h5")):
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.rmdir(work)
    except OSError:
        pass
