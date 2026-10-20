#!/usr/bin/env python3
"""
check_amber_21cm.py  --  is AMBER's own 21cm output aligned with its density snapshots?

Reads, for every redshift that has BOTH
    <run>/output/21cm/Tb_L=<L>_N=<N>_z=<z>.dat    (raw float32, N^3, no header)
    <run>/output/cmb/fields_z=<z>.dat              (float32: z, rho2(N^3), mom2(3,N^3))
and tests whether the two boxes sit on the same grid:

  * Pearson r between Tb and rho (1+delta), for the file read in Fortran order and in C order.
    The right order gives a clearly larger r; the wrong one gives r ~ 0.
  * If Tb = T0(z) * x_HI * (1+delta) with x_HI in {0,1} per cell, then Tb/rho is the SAME number in every
    neutral cell (scatter ~ 0). Printed as std/mean of Tb/rho over cells with Tb>0. Scatter that is not ~0
    means Tb used another density (e.g. rho1 instead of rho2) or includes redshift-space distortions.
  * Implied T0 versus the textbook value (approximate, a few per cent).
  * Neutral volume fraction and mean Tb versus AMBER's own 21cm_global.txt.

Only reads files and prints; needs numpy. Light enough for a login node (N=32 boxes are 0.5 MB).

Usage:  python3 check_amber_21cm.py --run $R/amber_runs/tiny21
"""
import argparse, glob, os, re
import numpy as np

DT = np.dtype('<f4')


def read_tb(path, N, order):
    a = np.fromfile(path, dtype=DT)
    if a.size != N ** 3:
        raise ValueError(f"{path}: {a.size} floats, expected {N**3}")
    return a.reshape((N, N, N), order=order)


def read_rho(path, N):
    a = np.fromfile(path, dtype=DT)
    if a.size != 1 + 4 * N ** 3:
        raise ValueError(f"{path}: {a.size} floats, expected {1 + 4 * N**3}")
    return float(a[0]), a[1:1 + N ** 3].reshape((N, N, N), order='F')


def t0_approx(z, Ob, h, Om):
    """Approximate fully-neutral, mean-density brightness temperature in mK (27 mK form)."""
    wb, wm = Ob * h * h, Om * h * h
    return 27.0 * (wb / 0.023) * np.sqrt(0.15 / wm * (1.0 + z) / 10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='AMBER run folder (contains output/)')
    ap.add_argument('--Ob', type=float, default=0.04897)
    ap.add_argument('--h', type=float, default=0.6766)
    ap.add_argument('--Om', type=float, default=0.3111)
    a = ap.parse_args()

    out = os.path.join(a.run, 'output')
    tbs = sorted(glob.glob(os.path.join(out, '21cm', 'Tb_L=*_N=*_z=*.dat')))
    if not tbs:
        raise SystemExit(f"no Tb_*.dat files in {out}/21cm")

    glob_path = os.path.join(out, '21cm', '21cm_global.txt')
    G = np.loadtxt(glob_path, skiprows=1) if os.path.exists(glob_path) else None

    print(f"{'z':>6} {'r(F)':>8} {'r(C)':>8} {'order':>5} {'<Tb>':>8} {'T0 impl':>8} {'T0 txt':>7} "
          f"{'scatter':>8} {'xN vol':>7} {'xN glob':>8} {'<Tb> glob':>9}")
    votes = []
    for f in tbs:
        m = re.search(r'Tb_L=([0-9.]+)_N=(\d+)_z=([0-9.]+)\.dat$', os.path.basename(f))
        L, N, zs = float(m.group(1)), int(m.group(2)), m.group(3)
        ff = os.path.join(out, 'cmb', f'fields_z={zs}.dat')
        if not os.path.exists(ff):
            print(f"{float(zs):6.2f}  (no matching {os.path.basename(ff)})")
            continue
        z, rho = read_rho(ff, N)
        res = {}
        for order in ('F', 'C'):
            tb = read_tb(f, N, order)
            if tb.std() == 0:
                res[order] = (np.nan, tb)
                continue
            res[order] = (np.corrcoef(tb.ravel(), rho.ravel())[0, 1], tb)
        rF, rC = res['F'][0], res['C'][0]
        if np.isnan(rF):
            best = 'F'
            print(f"{z:6.2f}  Tb is identically zero (fully ionized): nothing to compare")
            continue
        best = 'F' if rF >= rC else 'C'
        votes.append(best)
        tb = res[best][1]
        neu = tb > 0
        ratio = tb[neu] / rho[neu]
        t0i, scat = ratio.mean(), ratio.std() / ratio.mean()
        xv = neu.mean()
        gx = gt = np.nan
        if G is not None:
            gx = np.interp(z, G[:, 0], G[:, 1])
            gt = np.interp(z, G[:, 0], G[:, 2])
        print(f"{z:6.2f} {rF:8.3f} {rC:8.3f} {best:>5} {tb.mean():8.3f} {t0i:8.2f} "
              f"{t0_approx(z, a.Ob, a.h, a.Om):7.2f} {scat:8.4f} {xv:7.3f} {gx:8.3f} {gt:9.3f}")

    print()
    if votes:
        order = max(set(votes), key=votes.count)
        print(f"Axis order that matches the density: '{order}'  (votes {votes})")
        print("Read: r(right order) should be well above r(wrong order). 'scatter' ~ 0 means Tb = T0*x_HI*rho exactly,")
        print("i.e. the 21cm box sits on the same grid as the density snapshot. 'xN vol' is the fraction of cells with Tb>0;")
        print("'xN glob' is AMBER's global x_H (mass-weighted), so a small difference is expected.")


if __name__ == '__main__':
    main()
