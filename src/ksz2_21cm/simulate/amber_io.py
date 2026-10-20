"""
Readers for AMBER (github.com/hytrac/amber) output, for the stitched-lightcone path.

The field/zre readers are vendored from ksz-pipeline (src/ksz_pipeline/amber/io.py, branch amber-integration);
read_tb, parse_tb_name and list_snapshots are NEW (pilot on CSD3, 20 Oct 2026).

Layouts (all raw little-endian float32, Fortran order: numpy reshape(order='F') gives arr[i,j,k] == Fortran
arr(i,j,k), i = x, k = z = the line-of-sight axis used by the stitcher):

  output/reion/zre_<tag>.dat     zre(N,N,N)
  output/cmb/fields_z=XX.XX.dat  z, rho2(N,N,N) [1+delta, interlaced + deconvolved], mom2(3,N,N,N) [(1+delta) v, km/s]
                                 (written by the ksz-pipeline field-dump patch, not upstream AMBER)
  output/21cm/Tb_L=<L>_N=<N>_z=XX.XX.dat   Tb(N,N,N) in mK, AMBER's own 21cm module (h21cmreion.f90 h21cm_simple):
                                 T0 * rho1 in cells that are neutral (z > zre), 0 elsewhere. NO header.
                                 Native conventions, verified against the source and the pilot:
                                   * T_S >> T_CMB, no peculiar velocities (no RSD), no lightcone effects;
                                   * T0 = 28 mK (Ob h^2/0.022) sqrt(0.15/(Om h^2)) sqrt((1+z)/10)  (Madau et al. 1997),
                                     about 1.081 x the exact value at every z -- NOT corrected here;
                                   * built from rho1 (plain mass assignment), not rho2: the two agree to ~2% in
                                     power for k < 0.4 h/Mpc and differ near the grid scale.
"""
import glob
import os
import re

import numpy as np

_DT = np.dtype('<f4')
KMS_TO_CMS = 1.0e5
_ZTAG = re.compile(r'z=\s*([0-9]+(?:\.[0-9]*)?)')
_TBNAME = re.compile(r'Tb_L=([0-9.]+)_N=(\d+)_z=([0-9]+(?:\.[0-9]+)?)\.dat$')


def z_from_name(path):
    m = _ZTAG.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"no z= tag in {path}")
    return float(m.group(1).rstrip('.'))


def read_zre(path, N):
    a = np.fromfile(path, dtype=_DT)
    if a.size != N ** 3:
        raise ValueError(f"{path}: {a.size} floats, expected N^3={N**3}")
    return a.reshape((N, N, N), order='F')


def read_fields(path, N):
    """One fields_z=*.dat snapshot -> (z, rho2 (N,N,N) = 1+delta, mom2 (3,N,N,N) km/s)."""
    a = np.fromfile(path, dtype=_DT)
    n3 = N ** 3
    if a.size != 1 + 4 * n3:
        raise ValueError(f"{path}: {a.size} floats, expected 1+4N^3={1 + 4 * n3}")
    return (float(a[0]),
            a[1:1 + n3].reshape((N, N, N), order='F'),
            a[1 + n3:].reshape((3, N, N, N), order='F'))


def read_tb(path, N):
    """One Tb_L=*_N=*_z=*.dat box [mK], arr[i,j,k] with k the LOS axis."""
    a = np.fromfile(path, dtype=_DT)
    if a.size != N ** 3:
        raise ValueError(f"{path}: {a.size} floats, expected N^3={N**3}")
    return a.reshape((N, N, N), order='F')


def parse_tb_name(path):
    """'Tb_L=64_N=32_z=06.50.dat' -> (L [Mpc/h], N, z)."""
    m = _TBNAME.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"not an AMBER Tb file name: {path}")
    return float(m.group(1)), int(m.group(2)), float(m.group(3))


def find_zre_file(reion_dir):
    files = glob.glob(os.path.join(reion_dir, 'zre_*.dat'))
    if len(files) != 1:
        raise FileNotFoundError(f"expected exactly one zre_*.dat in {reion_dir}, found {len(files)}")
    return files[0]


def list_snapshots(run_dir, z_tol=1e-3):
    """
    Match, by redshift, every output/cmb/fields_z=*.dat with an output/21cm/Tb_*.dat.
    Returns (L_box_mpc_h, N, [(z, fields_path, tb_path), ...] sorted by z).
    Raises if a field snapshot has no Tb box (or vice versa) or if L, N differ between Tb files:
    the 21cm redshift grid must equal the CMB shell grid (--h21zmin/zmax/zdel = --czmin/czmax/czdel).
    """
    out = os.path.join(run_dir, 'output')
    fields = sorted(glob.glob(os.path.join(out, 'cmb', 'fields_z=*.dat')), key=z_from_name)
    tbs = glob.glob(os.path.join(out, '21cm', 'Tb_L=*_N=*_z=*.dat'))
    if not fields:
        raise FileNotFoundError(f"no fields_z=*.dat in {out}/cmb")
    if not tbs:
        raise FileNotFoundError(f"no Tb_*.dat in {out}/21cm (was the run made with --h21cm write?)")
    info = [parse_tb_name(p) + (p,) for p in tbs]
    Ls, Ns = {i[0] for i in info}, {i[1] for i in info}
    if len(Ls) != 1 or len(Ns) != 1:
        raise ValueError(f"mixed box sizes among Tb files: L={sorted(Ls)} N={sorted(Ns)}")
    tb_by_z = sorted((i[2], i[3]) for i in info)
    snaps, used = [], set()
    for fp in fields:
        zf = z_from_name(fp)
        hit = [(zt, tp) for zt, tp in tb_by_z if abs(zt - zf) < z_tol]
        if len(hit) != 1:
            raise ValueError(f"field snapshot z={zf} has {len(hit)} matching Tb boxes; the 21cm grid must equal "
                             f"the CMB shell grid (Tb z: {[t[0] for t in tb_by_z]})")
        used.add(hit[0][0])
        snaps.append((zf, fp, hit[0][1]))
    extra = [zt for zt, _ in tb_by_z if zt not in used]
    if extra:
        raise ValueError(f"Tb boxes without a field snapshot: z = {extra}")
    return Ls.pop(), Ns.pop(), snaps
