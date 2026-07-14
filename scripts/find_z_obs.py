#!/usr/bin/env python
"""
find_z_obs.py

Measures the redshift at which reionization reaches a given ionized
fraction threshold (x_HII), from real cached lightcones — used to set
`ksz.z_obs` in a config so patchy-kSZ integration stops right where
patchy reionization ends, not before or after.

Not part of the main pipeline sequence (scripts/01-06) — this is a
one-off (or occasional, if AstroParams ever change) diagnostic to inform
a config value, kept here because it's genuinely reusable rather than
throwaway.

Run:
    conda activate ksz2-21cm
    python scripts/find_z_obs.py --config configs/variants/timing_test.yaml
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import py21cmfast as p21c

from ksz2_21cm.utils.config import load_config

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/variants/timing_test.yaml")
args = parser.parse_args()
cfg = load_config(args.config)

cache_dir = cfg["paths"]["cache_dir"]
seeds     = cfg["simulation"]["random_seeds"]

print(f"Loading lightcones from {cache_dir} ...")
z_at_threshold = {0.90: [], 0.95: [], 0.99: [], 0.999: [], 0.9999: []}

for seed in seeds:
    lc_files = sorted(glob.glob(os.path.join(cache_dir, f"seed_{seed}", "LightCone_*.h5")))
    if not lc_files:
        print(f"  seed {seed}: no lightcone found, skipping")
        continue
    lc = p21c.LightCone.read(lc_files[0])

    # py21cmfast's node_redshifts is naturally DESCENDING (high z first).
    # global_xH (neutral fraction) is correspondingly high at high z, low at
    # low z -- so xe = 1 - global_xH is ALREADY ascending in natural array
    # order (verified against a synthetic example before trusting this).
    # No [::-1] reversal needed or wanted here.
    z_natural  = lc.node_redshifts
    xe_natural = 1.0 - lc.global_xH

    print(f"\n  seed {seed}: x_e range [{xe_natural.min():.5f}, {xe_natural.max():.5f}] "
          f"over z=[{z_natural.min():.2f}, {z_natural.max():.2f}]")

    for thresh in z_at_threshold:
        if xe_natural.max() < thresh:
            print(f"    x_e never reaches {thresh} in this lightcone (z_min too high?)")
            continue
        z_cross = float(np.interp(thresh, xe_natural, z_natural))
        z_at_threshold[thresh].append(z_cross)
        print(f"    z(x_e={thresh:<7}) = {z_cross:.3f}")

print("\n" + "=" * 50)
print("Mean across seeds (this is your z_obs candidate):")
for thresh, vals in z_at_threshold.items():
    if vals:
        arr = np.array(vals)
        print(f"  x_e={thresh:<7}  z = {arr.mean():.3f}"
              + (f" +/- {arr.std():.3f}" if len(arr) > 1 else "  (only 1 seed)"))
print("=" * 50)
print("\nFor PATCHY kSZ specifically, x_e=0.9999 (or 0.999) is the usual choice --")
print("stops integration right where patchy reionization ends, before the smooth")
print("late-time/Ostriker-Vishniac kSZ contribution would start dominating.")
