#!/usr/bin/env python3
"""
check_amber_21cm.py  --  is AMBER's own 21cm output aligned with its density snapshots?

Reads, for every redshift that has BOTH
    <run>/output/21cm/Tb_L=<L>_N=<N>_z=<z>.dat    (raw float32, N^3, no header)
    <run>/output/cmb/fields_z=<z>.dat              (float32: z, rho2(N^3), mom2(3,N^3))
and tests whether the two boxes sit on the same grid:

  * Pearson r between Tb and rho (1+delta), for the file read in Fortran order and in C order.
    The right order gives a clearly larger |r| (it can be negative at low z, where dense regions are already
    ionized); the wrong one gives r ~ 0.
  * AMBER's own global T_b and x_H (21cm_global.txt) against the textbook T0 (exact formula, T_S >> T_CMB):
    "norm" = T_b / (x_H * T0). A constant norm != 1 means an overall prefactor difference inside AMBER.
  * Spread of Tb/rho over cells with Tb>0 (median, 99th percentile, max), and the mean of rho.

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


def t0_exact(z, Ob, h, Om, Yp=0.245):
    """Brightness temperature [mK] of a fully neutral, mean-density, T_S >> T_CMB patch at redshift z.
    dTb = 3 h_P c^3 A10 n_H / (32 pi k_B nu0^2 H(z) (1+z)). Flat LCDM, radiation neglected (0.2% on H)."""
    hP, c, kB, mp, G = 6.62607015e-34, 2.99792458e8, 1.380649e-23, 1.67262192e-27, 6.6743e-11
    A10, nu0 = 2.8843e-15, 1.420405751768e9
    H0 = h * 3.2407793e-18                              # 100 h km/s/Mpc in 1/s
    rho_c0 = 3.0 * H0 ** 2 / (8.0 * np.pi * G)          # kg/m^3
    nH = (1.0 - Yp) * Ob * rho_c0 * (1.0 + z) ** 3 / mp  # m^-3
    H = H0 * np.sqrt(Om * (1.0 + z) ** 3 + 1.0 - Om)
    return 3.0 * hP * c ** 3 * A10 * nH / (32.0 * np.pi * kB * nu0 ** 2 * H * (1.0 + z)) * 1e3


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

    print(f"{'z':>6} {'r(F)':>7} {'r(C)':>7} {'order':>5} {'<Tb>':>7} {'<Tb>glob':>8} {'xH glob':>7} "
          f"{'T0 exact':>8} {'norm':>6} {'<rho>':>6} {'Tb/rho: p50':>11} {'p99':>6} {'max':>6} {'xN vol':>6}")
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
            res[order] = (np.nan if tb.std() == 0 else np.corrcoef(tb.ravel(), rho.ravel())[0, 1], tb)
        rF, rC = res['F'][0], res['C'][0]
        if np.isnan(rF):
            print(f"{z:6.2f}  Tb is identically zero (fully ionized): nothing to compare")
            continue
        # the RIGHT order is the one with the larger |r| (r can be negative: dense regions ionize first)
        best = 'F' if abs(rF) >= abs(rC) else 'C'
        votes.append(best)
        tb = res[best][1]
        neu = tb > 0
        ratio = tb[neu] / rho[neu]
        gx = gt = np.nan
        if G is not None:
            gx = np.interp(z, G[:, 0], G[:, 1])
            gt = np.interp(z, G[:, 0], G[:, 2])
        t0 = t0_exact(z, a.Ob, a.h, a.Om)
        print(f"{z:6.2f} {rF:7.3f} {rC:7.3f} {best:>5} {tb.mean():7.3f} {gt:8.3f} {gx:7.3f} "
              f"{t0:8.2f} {gt / (gx * t0):6.3f} {rho.mean():6.3f} {np.median(ratio):11.2f} "
              f"{np.percentile(ratio, 99):6.2f} {ratio.max():6.2f} {neu.mean():6.3f}")

    print()
    if votes:
        order = max(set(votes), key=votes.count)
        print(f"Axis order that matches the density (larger |r|): '{order}'  (votes {votes})")
        print("Read:")
        print("  |r(right order)| well above |r(wrong order)| = Tb and rho share one grid. r may be negative at lower z.")
        print("  norm = AMBER's own global T_b / (x_H * T0 exact). 1.000 = AMBER's T0 agrees with the textbook value; a")
        print("  constant norm at all z = one overall prefactor difference.")
        print("  Tb/rho p50, p99, max: if cells are fully neutral or fully ionized these three agree (= T0 * norm);")
        print("  a long spread below the max means cells are partly ionized (or Tb uses a different density).")
        print("  <rho> should be 1.000.")


if __name__ == '__main__':
    main()
