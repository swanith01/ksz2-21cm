#!/usr/bin/env python
# =============================================================================
# 22a_save_xe_history.py
#
# Save the volume-weighted reionization history x_e(z) of each seed's stitched lightcone as a small
# file next to it (xe_history_seed<N>.npz: z, x_e), so scripts/22 never has to reopen the GB-size
# stitched .npz. Run by slurm/heff_production.sbatch right after the stitching.
#
#   python scripts/22a_save_xe_history.py --config <cfg> --seeds 1
# =============================================================================
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ksz2_21cm.utils.config import load_config
from ksz2_21cm.utils.history import milestones

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--seeds", type=int, nargs="+", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
cache = os.path.expanduser(cfg["paths"]["cache_dir"])
for s in (args.seeds or cfg["simulation"]["random_seeds"]):
    p = os.path.join(cache, f"seed_{s}", f"stitched_lightcone_seed{s}.npz")
    if not os.path.exists(p):
        print(f"seed {s}: no stitched lightcone, skipped")
        continue
    d = np.load(p)
    z = np.asarray(d["z_arr"], float)
    xe = 1.0 - np.asarray(d["xH_box"], np.float64).mean(axis=(0, 1))
    out = os.path.join(cache, f"seed_{s}", f"xe_history_seed{s}.npz")
    np.savez(out, z=z, x_e=xe, astro_params=str(cfg.get("astro_params")))
    m = milestones(z, xe)
    print(f"seed {s}: z_start={m['z_start']:.2f} z50={m['z50']:.2f} z_end={m['z_end']:.2f} -> {out}")
