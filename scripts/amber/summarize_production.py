#!/usr/bin/env python3
"""
Status table for the AMBER production (slurm/amber_production.sbatch). Reads, per history folder
<data-root>/amber_q512_h<ZMID>/cache/seed_<N>/ :
   stitched lightcone present, kSZ map, kSZ^2x21cm^2 and Ma-style results present,
   z at x_e = 10/50/90% (volume-weighted, from the stitched xH_box), and mean Tb at z~8.
Reads only; run on a login node (opens one array per file, about 0.4 GB each; --no-tb skips the Tb read).

   python scripts/amber/summarize_production.py [--data-root $R/ksz2-21cm-data]
"""
import argparse, glob, os, re
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.expanduser("~/rds/rds-dirac-dp145-uYnnMXSp128/dc-upad3/ksz2-21cm-data"))
    ap.add_argument("--no-tb", action="store_true")
    a = ap.parse_args()
    hist_dirs = sorted(glob.glob(os.path.join(a.data_root, "amber_q512_h*")), key=lambda p: float(re.findall(r"_h([0-9.]+)$", p)[0]))
    if not hist_dirs:
        raise SystemExit(f"no amber_q512_h* folders under {a.data_root}")
    print(f"{'z_mid':>5} {'seed':>4} {'stitch':>6} {'kSZmap':>6} {'cc_sq':>5} {'cc_ma':>5} {'z10':>6} {'z50':>6} {'z90':>6} {'<Tb>z8':>7}")
    for hd in hist_dirs:
        zm = float(re.findall(r"_h([0-9.]+)$", hd)[0])
        cache = os.path.join(hd, "cache")
        seeds = sorted(int(re.findall(r"seed_(\d+)$", p)[0]) for p in glob.glob(os.path.join(cache, "seed_*")))
        z50s = []
        for s in seeds:
            sd = os.path.join(cache, f"seed_{s}")
            npz = os.path.join(sd, f"stitched_lightcone_seed{s}.npz")
            has = lambda p: "yes" if (os.path.exists(p) and os.path.getsize(p) > 0) else "-"
            km = os.path.join(cache, "kSZ_maps", f"kSZ_map_stitched_seed{s}.npy")
            sq = os.path.join(sd, f"cross_corr_sq_stitched_seed{s}.npy")
            ma = glob.glob(os.path.join(sd, f"cross_corr_ma_stitched_seed{s}*.npy"))
            z10 = z50 = z90 = tb8 = float("nan")
            if os.path.exists(npz):
                try:
                    d = np.load(npz)
                    z = d["z_arr"]
                    xe = 1.0 - d["xH_box"].mean(axis=(0, 1))
                    z10, z50, z90 = (float(np.interp(v, xe[::-1], z[::-1])) for v in (0.1, 0.5, 0.9))
                    z50s.append(z50)
                    if not a.no_tb:
                        tb8 = float(d["brightness_temp"][:, :, int(np.argmin(abs(z - 8.0)))].mean())
                except Exception as e:                         # noqa: BLE001
                    print(f"{zm:5g} {s:4d}  unreadable stitched file: {e}")
            print(f"{zm:5g} {s:4d} {has(npz):>6} {has(km):>6} {has(sq):>5} {'yes' if ma else '-':>5} "
                  f"{z10:6.2f} {z50:6.2f} {z90:6.2f} {tb8:7.2f}")
        if z50s:
            print(f"{zm:5g}  mean z(x_e=0.5) over {len(z50s)} seeds = {np.mean(z50s):.2f} +- {np.std(z50s):.2f}   (volume-weighted; AMBER's z_mid is mass-weighted)")
    print("\nlogs: $R/logs/amber_prod_<jobid>_<task>.out   (a task that failed ends with a line starting 'FAILED:')")


if __name__ == "__main__":
    main()
