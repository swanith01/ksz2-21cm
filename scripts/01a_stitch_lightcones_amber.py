#!/usr/bin/env python
# =============================================================================
# 01a_stitch_lightcones_amber.py
#
# AMBER-sourced sibling of 01_stitch_lightcones.py. Same stitching geometry (stitch_field, verbatim), same per-seed
# output file and keys (density, xH_box, velocity_z, brightness_temp, z_arr, pos_axis), so 02_compute_ksz_maps_
# stitched.py and 04/04b/... run on it UNCHANGED, with a config that points at AMBER's box (configs/variants/
# amber_q512.yaml). The only difference between the two lightcones is which simulation made the snapshots.
#
# Input: one AMBER run directory per seed, made by scripts/amber/make_amber_input.py with --h21cm write and the 21cm
# grid equal to the CMB shell grid (--h21zmin/zmax/zdel = --czmin/czmax/czdel), run with slurm/amber_run.sbatch.
# wrap_cycle_seed = the seed (same convention as 01).
#
# Brightness temperature is AMBER's own (native). Its fixed 1.081 factor over the exact T0 is NOT removed; pass
# --tb-scale 0.925 (= 1/1.081) explicitly if an absolute comparison with 21cmFAST needs it. The factor is recorded
# in the npz either way.
#
# Memory: all snapshots x 4 fields are held at once (about 7.5 GB at N=256, 28 snapshots) plus ~2 GB output, per
# worker. Default is one worker; raise --workers only with enough memory.
#
# Usage:
#   python scripts/01a_stitch_lightcones_amber.py --config configs/variants/amber_q512.yaml \
#       --run-dir-template '/path/to/amber_runs/seed_{seed}' [--seeds 1 2 3] [--delete-raw]
# =============================================================================
import argparse
import glob
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from ksz2_21cm.utils.config import load_config, ensure_dirs
from ksz2_21cm.simulate.stitch_from_coeval import build_los_z_grid
from ksz2_21cm.simulate import amber_io as aio

TB_NATIVE_OVER_EXACT = 1.081   # measured in the CSD3 pilot (AMBER Madau-fit T0 vs exact, flat across z=7.5-11.5)


def _stitch_one(job):
    seed, run_dir, npz_path, h, tb_scale, delete_raw = job
    from ksz2_21cm.simulate.stitch_from_amber import stitch_lightcone_from_amber
    if os.path.exists(npz_path):
        try:
            _ = np.load(npz_path)["z_arr"]
            return seed, npz_path, 0.0, "cached"
        except Exception:
            pass
    t0 = time.time()
    try:
        L_h, N, snaps = aio.list_snapshots(run_dir)
        snap_z = np.array([s[0] for s in snaps])
        z_arr = build_los_z_grid(snap_z.min(), snap_z.max(), L_h / h / N)
        r = stitch_lightcone_from_amber(run_dir, z_arr, h, N=N, wrap_cycle_seed=seed, tb_scale=tb_scale)
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        tmp = npz_path + ".tmp.npz"
        np.savez_compressed(
            tmp,
            density=r["density"].astype(np.float32), xH_box=r["xH_box"].astype(np.float32),
            velocity_z=r["velocity_z"].astype(np.float32),
            brightness_temp=r["brightness_temp"].astype(np.float32),
            z_arr=r["z_arr"], pos_axis=r["pos_axis"],
            source=np.array("amber"), L_box_mpc_h=np.array(L_h), h=np.array(h), N=np.array(N),
            tb_scale=np.array(tb_scale), tb_native_over_exact=np.array(TB_NATIVE_OVER_EXACT),
            snap_z=r["snap_z"])
        os.replace(tmp, npz_path)
        np.load(npz_path)["z_arr"]                     # confirm readable before any deletion
        if delete_raw:
            for f in glob.glob(os.path.join(run_dir, "output", "cmb", "fields_z=*.dat")):
                os.remove(f)
        return seed, npz_path, time.time() - t0, "computed"
    except Exception as e:                              # noqa: BLE001
        return seed, None, time.time() - t0, f"failed: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/variants/amber_q512.yaml")
    ap.add_argument("--run-dir-template", required=True,
                    help="AMBER run directory, with {seed} replaced per seed, e.g. /path/seed_{seed}")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="default: simulation.random_seeds")
    ap.add_argument("--h", type=float, default=0.6766)
    ap.add_argument("--tb-scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--delete-raw", action="store_true",
                    help="after a successful stitch delete output/cmb/fields_z=*.dat of that run (keeps power, "
                         "cl, tau, 21cm, reion)")
    a = ap.parse_args()

    cfg = load_config(a.config)
    ensure_dirs(cfg)
    cache_dir = cfg["paths"]["cache_dir"]
    seeds = a.seeds if a.seeds else cfg["simulation"]["random_seeds"]

    # geometry guard: the downstream scripts take box_len and hii_dim from the config, not from the npz
    first = a.run_dir_template.format(seed=seeds[0])
    L_h, N, snaps = aio.list_snapshots(first)
    box_mpc = L_h / a.h
    assert cfg["coeval_ksz"]["hii_dim"] == N == cfg["simulation"]["hii_dim"], \
        f"config hii_dim ({cfg['coeval_ksz']['hii_dim']}) != AMBER N ({N})"
    assert abs(cfg["coeval_ksz"]["box_len"] - box_mpc) < 1e-3 * box_mpc, \
        f"config box_len ({cfg['coeval_ksz']['box_len']}) != AMBER L/h ({box_mpc:.4f} Mpc)"
    assert abs(cfg["simulation"]["box_len"] - box_mpc) < 1e-3 * box_mpc
    print(f"AMBER box: L = {L_h} Mpc/h = {box_mpc:.2f} Mpc, N = {N}, cell = {box_mpc / N:.3f} Mpc, "
          f"{len(snaps)} snapshots z = {snaps[0][0]} ... {snaps[-1][0]}")
    print(f"seeds {seeds}   tb_scale = {a.tb_scale}   workers = {a.workers}   delete_raw = {a.delete_raw}")

    jobs = []
    for s in seeds:
        d = os.path.join(cache_dir, f"seed_{s}")
        jobs.append((s, a.run_dir_template.format(seed=s), os.path.join(d, f"stitched_lightcone_seed{s}.npz"),
                     a.h, a.tb_scale, a.delete_raw))

    t0 = time.time()
    results = {}
    if a.workers <= 1:
        for j in jobs:
            results[j[0]] = _stitch_one(j)
            print(f"  seed {j[0]:3d}: {results[j[0]][3]}  ({results[j[0]][2] / 60:.1f} min)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers, mp_context=mp.get_context("fork")) as ex:
            futs = {ex.submit(_stitch_one, j): j[0] for j in jobs}
            for f in as_completed(futs):
                results[futs[f]] = f.result()
                r = results[futs[f]]
                print(f"  seed {r[0]:3d}: {r[3]}  ({r[2] / 60:.1f} min)", flush=True)
    print(f"\nall workers returned in {time.time() - t0:.0f} s")

    print("\n=== REIONIZATION HISTORY CHECK (volume-weighted x_e from the stitched xH_box) ===")
    for s in seeds:
        p = results[s][1]
        if not p:
            print(f"  seed {s}: no stitched cache ({results[s][3]})")
            continue
        d = np.load(p)
        x_e = 1.0 - d["xH_box"].mean(axis=(0, 1))
        z = d["z_arr"]
        print(f"  seed {s:3d}  z(x_e=10%)={np.interp(0.1, x_e[::-1], z[::-1]):.2f}  "
              f"z(x_e=50%)={np.interp(0.5, x_e[::-1], z[::-1]):.2f}  z(x_e=90%)={np.interp(0.9, x_e[::-1], z[::-1]):.2f}"
              f"   <Tb> at z~8: {d['brightness_temp'][:, :, np.argmin(abs(z - 8.0))].mean():.2f} mK")
    nbad = sum(1 for s in seeds if not results[s][1])
    print(f"\n{len(seeds) - nbad}/{len(seeds)} stitched lightcones in {cache_dir}")
    print("Next: python scripts/02_compute_ksz_maps_stitched.py --config", a.config)
    sys.exit(1 if nbad else 0)


if __name__ == "__main__":
    main()
