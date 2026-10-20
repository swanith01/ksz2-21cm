"""
AMBER-sourced sibling of simulate/stitch_from_coeval.stitch_lightcone_from_coeval.

Uses stitch_field() from stitch_from_coeval VERBATIM (geometry, bilinear rotation, per-wrap-cycle angle, snapshot
interpolation), and passes the SAME angle_deg / wrap_cycle_seed / wrap_mode to all four fields, so density, xH,
velocity and brightness temperature are sampled from the same spatial position at every LOS pixel. That sharing
is what the kSZ x 21cm cross-correlation depends on; tests/test_amber_stitch.py checks it.

Only the source of the snapshots changes: instead of run_coeval_fields() (21cmFAST), they come from one AMBER run
directory (see amber_io). Fields returned, in the SAME conventions and keys as the 21cmFAST path:

  density         RAW delta = rho2 - 1 (caller adds 1 before compute_ksz_map)
  xH_box          binary neutral flag per snapshot (zre <= z), interpolated across snapshots like the others
  velocity_z      mom2_z / rho2 * 1e5, physical LOS peculiar velocity [cm/s]
  brightness_temp AMBER's own Tb [mK], used as written (see amber_io for its conventions); tb_scale = 1.0 by default
  z_arr, pos_axis
"""
import numpy as np

from .stitch_from_coeval import comoving_distance_mpc, stitch_field
from . import amber_io as aio


def load_amber_snapshots(run_dir, N=None):
    """-> (L_box_mpc_h, N, snap_z, delta_boxes, xH_boxes, vz_boxes, tb_boxes), boxes keyed by z."""
    import os
    L_h, N_files, snaps = aio.list_snapshots(run_dir)
    if N is not None and N != N_files:
        raise ValueError(f"N={N} requested but the AMBER files have N={N_files}")
    N = N_files
    zre = aio.read_zre(aio.find_zre_file(os.path.join(run_dir, 'output', 'reion')), N)
    delta, xH, vz, tb = {}, {}, {}, {}
    for z, fpath, tpath in snaps:
        zf, rho, mom = aio.read_fields(fpath, N)
        if abs(zf - z) > 1e-3:
            raise ValueError(f"{fpath}: z in header {zf} != z in name {z}")
        m = float(rho.mean())
        if abs(m - 1.0) > 1e-2:
            raise RuntimeError(f"{fpath}: <rho2>={m:.4f}, expected ~1")
        good = rho > 1e-6
        delta[z] = (rho - 1.0).astype(np.float32)
        xH[z] = (zre <= z).astype(np.float32)             # AMBER's own test: neutral iff z > zre
        vz[z] = (np.where(good, mom[2] / np.where(good, rho, 1.0), 0.0) * aio.KMS_TO_CMS).astype(np.float32)
        tb[z] = aio.read_tb(tpath, N)
    snap_z = np.array(sorted(delta), dtype=float)
    return L_h, N, snap_z, delta, xH, vz, tb


def stitch_lightcone_from_amber(run_dir, z_arr, h, N=None, angle_deg=0.0, wrap_cycle_seed=None,
                                wrap_mode='wrap', tb_scale=1.0):
    """
    Build (density, xH_box, velocity_z, brightness_temp) on the LOS grid z_arr from one AMBER run directory.

    z_arr should be built with stitch_from_coeval.build_los_z_grid(z_lo, z_hi, L_box_mpc_h / h / N) and lie inside
    the snapshot range (stitch_field extrapolates flatly outside it). h must be the Hubble parameter of the run.
    tb_scale multiplies the native Tb (default 1.0 = native; a documented calibration factor such as 1/1.081 may be
    passed explicitly, never applied silently).
    """
    L_h, N, snap_z, delta, xH, vz, tb = load_amber_snapshots(run_dir, N)
    cell_size = L_h / h / N
    z0 = float(np.min(z_arr))
    if z_arr.min() < snap_z.min() - 1e-6 or z_arr.max() > snap_z.max() + 1e-6:
        raise ValueError(f"z_arr [{z_arr.min():.3f}, {z_arr.max():.3f}] outside snapshot range "
                         f"[{snap_z.min():.3f}, {snap_z.max():.3f}]")
    kw = dict(z_arr=z_arr, z0=z0, cell_size=cell_size, ngrid=N, angle_deg=angle_deg,
              wrap_cycle_seed=wrap_cycle_seed, wrap_mode=wrap_mode)
    density = stitch_field(delta, snap_z, **kw)
    xH_box = stitch_field(xH, snap_z, **kw)
    velocity_z = stitch_field(vz, snap_z, **kw)
    brightness_temp = stitch_field(tb, snap_z, **kw)
    if tb_scale != 1.0:
        brightness_temp = (brightness_temp * np.float32(tb_scale)).astype(np.float32)
    pos_axis = np.array([comoving_distance_mpc(z) for z in z_arr])
    return dict(density=density, xH_box=xH_box, velocity_z=velocity_z, brightness_temp=brightness_temp,
                z_arr=np.asarray(z_arr), pos_axis=pos_axis, L_box_mpc_h=L_h, N=N, snap_z=snap_z)
