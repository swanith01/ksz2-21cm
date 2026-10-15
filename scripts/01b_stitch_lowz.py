#!/usr/bin/env python
# =============================================================================
# 01b_stitch_lowz.py   (NEW, 2026-10-14)
#
# Post-EoR test, step 1: extend each seed's stitched lightcone DOWN from z = 4 to z ~ 0
# (density + LOS velocity only; the gas is fully ionized there). The existing z >= 4
# lightcone, its cached kSZ maps and cross-spectra are NOT touched. See
# src/ksz2_21cm/simulate/stitch_below.py for how the geometry stays aligned.
#
# Per seed: ~22 new coeval snapshots between z = 4 and z ~ 0 (equal comoving spacing,
# steps <= --max-step-mpc), the same seed / cache dir as 01, then stitching.
# Output: <cache_dir>/seed_<N>/stitched_lowz_seed<N>.npz (density, velocity_z, z_arr, pos_axis, ...)
#
# Heavy (21cmFAST + ~1 GB arrays): run through PBS (pbs/run_posteor.pbs), not on the login node.
#   python scripts/01b_stitch_lowz.py --config configs/variants/stitched_256.yaml --seeds 1
# =============================================================================
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from ksz2_21cm.utils.config import load_config, ensure_dirs
from ksz2_21cm.simulate.stitch_below import (
    build_low_z_grid, low_z_snapshots, stitch_fields_signed,
)
from ksz2_21cm.simulate.stitch_from_coeval import comoving_distance_mpc

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/variants/stitched_256.yaml")
p.add_argument("--seeds", type=int, nargs="+", default=[1])
p.add_argument("--max-step-mpc", type=float, default=340.0,
               help="largest comoving gap between snapshots below z=4 (existing grid: 150-330 Mpc)")
p.add_argument("--z-floor", type=float, default=0.001)
p.add_argument("--threads", type=int, default=None)
args = p.parse_args()

cfg = load_config(args.config); ensure_dirs(cfg)
cache_dir = cfg["paths"]["cache_dir"]
N = cfg["coeval_ksz"]["hii_dim"]; L = cfg["coeval_ksz"]["box_len"]; cell = L / N
threads = args.threads or int(os.environ.get("NCPUS", cfg["simulation"]["n_threads"]))

from ksz2_21cm.simulate.coeval_fields import run_coeval_fields   # needs py21cmfast

for seed in args.seeds:
    sd = os.path.join(cache_dir, f"seed_{seed}")
    main = os.path.join(sd, f"stitched_lightcone_seed{seed}.npz")
    out = os.path.join(sd, f"stitched_lowz_seed{seed}.npz")
    assert os.path.exists(main), f"seed {seed}: run 01_stitch_lightcones.py first ({main})"
    if os.path.exists(out):
        try:
            np.load(out)["z_arr"]; print(f"seed {seed}: cached"); continue
        except Exception:
            pass
    z_ref = float(np.min(np.load(main)["z_arr"]))
    snaps = low_z_snapshots(z_ref, args.z_floor, args.max_step_mpc)
    chis = np.array([comoving_distance_mpc(z) for z in snaps])
    print(f"seed {seed}: z_ref = {z_ref}; {len(snaps) - 1} new snapshots below it "
          f"(+ the cached z={z_ref} one), steps {np.diff(chis).min():.0f}-{np.diff(chis).max():.0f} Mpc")
    print("   z =", " ".join(f"{z:g}" for z in snaps))
    t0 = time.time()
    dens, vz = {}, {}
    for z in snaps:
        delta, _xH, _vx, _vy, v, _dTb = run_coeval_fields(
            float(z), N, L, sd, N_THREADS=threads, random_seed=seed)
        dens[float(z)] = np.asarray(delta, dtype=np.float32)
        vz[float(z)] = np.asarray(v, dtype=np.float32)
        del _xH, _vx, _vy, _dTb
        print(f"   snapshot z={z:g} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    z_low = build_low_z_grid(z_ref, args.z_floor, cell)
    print(f"   stitching {len(z_low)} LOS pixels ({z_low.min():.3f} ... {z_low.max():.3f})", flush=True)
    st = stitch_fields_signed({"density": dens, "velocity_z": vz}, snaps, z_low, z_ref, cell, N,
                              wrap_cycle_seed=seed)
    pos = np.array([comoving_distance_mpc(z) for z in z_low])
    np.savez_compressed(out, density=st["density"], velocity_z=st["velocity_z"],
                        z_arr=z_low, pos_axis=pos, z_ref=z_ref, snapshots=snaps)
    print(f"   wrote {out}  ({(time.time() - t0) / 60:.1f} min)")
print("Done. Next: python scripts/02b_compute_ksz_posteor.py --seeds", *args.seeds)
