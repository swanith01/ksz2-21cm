"""
AMBER -> stitched lightcone, version 2 (momentum + exact re-orientation + the run's own cosmology).

VENDORED + ADAPTED from ksz-pipeline (github.com/swanith01/ksz-pipeline), branch amber-integration, commit c2e1e33,
file src/ksz_pipeline/amber/stream_lightcone.py (mode='momentum', geometry='reorient'). The slab walker, view_slab
and reorient_params are that file's; what is new here is that it also streams the 21cm brightness temperature and
the neutral flag and returns 3D lightcone arrays in the format scripts 02 / 04 / 04b read, not a kSZ map.

WHY (ksz-pipeline heads-up of 2026-10-26; all three were present in simulate/stitch_from_amber.py, v1):

  1. v = mom2 / rho2 is never formed. AMBER's rho2 (TSC + interlacing + deconvolution) rings and goes negative in
     voids, so the ratio blows up, and stitching delta and v separately then pairs a huge v with a neighbouring
     snapshot's normal rho. The kSZ integrand is built cell by cell per snapshot as q = x_HII * mom2_los (no
     division) and THEN blended in z, so the file stores q_z = x_HII * mom_los [cm/s] directly. 02 integrates q_z
     as it stands (the kSZ integrand is (1+delta) x_HII v = x_HII * mom).
  2. Successive passes through the periodic box (wrap cycles) are decorrelated by EXACT re-orientation, not by an
     in-plane rotation with bilinear sampling (which low-passes: 0.80 / 0.59 of the power at ell = 5000 / 8000 on
     the 2 Gpc/h 1536^3 run). Per cycle a random box axis is the line of sight (and the momentum component along
     it is the LOS velocity) plus a random integer periodic shift on every axis. Integer indexing only.
  3. chi(z) (LOS pixel positions, wrap cycles, pos_axis) comes from the run's own cosmology
     (utils.cosmology.RunCosmology.from_run_dir), passed in as pos_axis.

The SAME view (LOS axis + shifts) is applied to every field at a given LOS pixel (delta, zre -> x_H, momentum,
Tb), which is what the kSZ x 21cm cross-correlation needs.

Fields returned (all (N, N, n_los) float32, the same layout as the 21cmFAST stitched path):
  density          RAW delta = rho2 - 1 (diagnostic; the kSZ integral does not use it in this version)
  xH_box           neutral flag blended in z (binary per snapshot: neutral iff zre <= snapshot z... see below)
  q_z              x_HII * mom_los [cm/s] -- THE kSZ integrand field (replaces velocity_z)
  brightness_temp  AMBER's own Tb [mK], used as written
Neutral convention: identical to v1, a cell is flagged xH = 1 at snapshot redshift z when zre <= z (AMBER's own
test: neutral iff z > zre means the cell has not reionised yet).
"""
import numpy as np
from scipy.ndimage import map_coordinates

from . import amber_io as aio
from .stitch_from_coeval import cycle_angle, rotated_coords

_DT = np.dtype('<f4')
_Z_VIEW = (2, (0, 0, 0))          # LOS = box z axis, no shift: exactly arr[:, :, y]
STITCHER = "amber_stream v2 (momentum, reorient, run cosmology)"


def reorient_params(cycle_idx, seed, N):
    """(los_axis, (sx, sy, sz)) for one wrap cycle. Deterministic in (seed, cycle_idx), separate random stream
    from cycle_angle."""
    rng = np.random.default_rng((int(seed), int(cycle_idx), 1))
    return int(rng.integers(3)), tuple(int(v) for v in rng.integers(0, N, size=3))


def view_slab(arr, y, view=_Z_VIEW):
    """2D slab of a 3D box at LOS index y under view = (axis, shift): the slab perpendicular to `axis` at
    (y + shift[axis]) % N, the two remaining axes (in increasing order) periodically shifted. The default view is
    exactly arr[:, :, y]. Exact."""
    axis, shift = view
    idx = [slice(None)] * 3
    idx[axis] = (y + shift[axis]) % arr.shape[axis]
    s = arr[tuple(idx)]                      # basic indexing: a view (np.take on a memmap is ~50x slower)
    o = [k for k in range(3) if k != axis]
    if shift[o[0]] or shift[o[1]]:
        s = np.roll(s, (-shift[o[0]], -shift[o[1]]), axis=(0, 1))
    return s


class _Snapshot:
    """Lazy (memmapped) view of one fields_z=*.dat dump and its Tb box."""

    def __init__(self, z, fields_path, tb_path, N):
        n3 = N ** 3
        mm = np.memmap(fields_path, dtype=_DT, mode='r', shape=(1 + 4 * n3,))
        self.z = float(z)
        if abs(float(mm[0]) - self.z) > 1e-3:
            raise ValueError(f"{fields_path}: z in header {float(mm[0])} != z in name {self.z}")
        self.rho = mm[1:1 + n3].reshape((N, N, N), order='F')
        self.mom = mm[1 + n3:].reshape((3, N, N, N), order='F')
        self.tb = np.memmap(tb_path, dtype=_DT, mode='r', shape=(n3,)).reshape((N, N, N), order='F')

    def stats(self):
        """(<rho2>, min rho2, fraction of cells with rho2 < 0.05, fraction < 0)."""
        r = np.asarray(self.rho)
        return float(r.mean()), float(r.min()), float((r < 0.05).mean()), float((r < 0).mean())

    def slabs(self, y, zre_slab, view):
        """(delta, q_los [cm/s], x_H, Tb) float64 slabs at LOS index y under `view`. q_los = x_HII * mom along the
        view's LOS axis: no division by rho."""
        delta = np.asarray(view_slab(self.rho, y, view), dtype=np.float64) - 1.0
        mz = np.asarray(view_slab(self.mom[view[0]], y, view), dtype=np.float64) * aio.KMS_TO_CMS
        xH = (zre_slab <= self.z).astype(np.float64)
        tb = np.asarray(view_slab(self.tb, y, view), dtype=np.float64)
        return delta, (1.0 - xH) * mz, xH, tb


def build_los_z_grid_cosmo(z_min, z_max, cell_size, cosmo, z_oversample=4000):
    """As stitch_from_coeval.build_los_z_grid (uniform COMOVING spacing = cell_size) but with the chi(z) of
    `cosmo` (anything with comoving_distance(z).value)."""
    z_fine = np.linspace(z_min, z_max, z_oversample)
    chi_fine = np.asarray(cosmo.comoving_distance(z_fine).value, float)
    n_pix = int(round((chi_fine[-1] - chi_fine[0]) / cell_size))
    return np.interp(chi_fine[0] + cell_size * np.arange(n_pix), chi_fine, z_fine)


def stitch_lightcone_from_amber_stream(run_dir, z_arr, pos_axis, h, N=None, wrap_cycle_seed=None,
                                       geometry='reorient', wrap_mode='wrap', progress_every=0):
    """
    Build the stitched lightcone arrays for one AMBER run directory, slice by slice.

    z_arr, pos_axis : LOS redshifts (ascending, uniform in comoving distance with step cell = L_box/h/N) and the
                      comoving distances [Mpc] of THE RUN'S OWN cosmology at those redshifts.
    geometry        : 'reorient' (default, exact) or 'rotate' (v1's in-plane rotation + bilinear sampling, kept only
                      for A/B tests; low-passes). Both are ignored when wrap_cycle_seed is None (slabs as stored).
    """
    if geometry not in ('reorient', 'rotate'):
        raise ValueError(f"geometry must be 'reorient' or 'rotate', got {geometry!r}")
    L_h, N_files, snaps = aio.list_snapshots(run_dir)
    if N is not None and N != N_files:
        raise ValueError(f"N={N} requested but the AMBER files have N={N_files}")
    N = N_files
    import os
    zre = np.memmap(aio.find_zre_file(os.path.join(run_dir, 'output', 'reion')), dtype=_DT, mode='r',
                    shape=(N ** 3,)).reshape((N, N, N), order='F')
    cell = L_h / h / N
    z_arr = np.asarray(z_arr, float)
    pos_axis = np.asarray(pos_axis, float)
    if z_arr.shape != pos_axis.shape:
        raise ValueError("z_arr and pos_axis must have the same shape")
    sn = [_Snapshot(z, fp, tp, N) for z, fp, tp in snaps]
    sz = np.array([s.z for s in sn])
    if sz.size < 2:
        raise ValueError("need at least two snapshots")
    if z_arr.min() < sz.min() - 1e-6 or z_arr.max() > sz.max() + 1e-6:
        raise ValueError(f"z_arr [{z_arr.min():.3f}, {z_arr.max():.3f}] outside snapshot range "
                         f"[{sz.min():.3f}, {sz.max():.3f}]")
    stats = np.array([s.stats() for s in sn])                 # (nsnap, 4)
    if np.any(np.abs(stats[:, 0] - 1.0) > 1e-2):
        bad = sz[np.abs(stats[:, 0] - 1.0) > 1e-2]
        raise RuntimeError(f"<rho2> differs from 1 by more than 1% at z = {bad.tolist()}")

    n = z_arr.size
    out = {k: np.empty((N, N, n), np.float32) for k in ('density', 'xH_box', 'q_z', 'brightness_temp')}
    views, coords = {}, {}
    use_view = wrap_cycle_seed is not None and geometry == 'reorient'
    use_rot = wrap_cycle_seed is not None and geometry == 'rotate'
    for j, z in enumerate(z_arr):
        d = pos_axis[j] - pos_axis[0]
        y = int(round(d / cell)) % N
        cyc = int(d // (cell * N))
        if cyc not in views:
            views[cyc] = reorient_params(cyc, wrap_cycle_seed, N) if use_view else _Z_VIEW
            if use_rot:
                coords[cyc] = rotated_coords(N, cycle_angle(cyc, wrap_cycle_seed))
        view = views[cyc]
        i0 = int(np.clip(np.searchsorted(sz, z) - 1, 0, sz.size - 2))
        w1 = (z - sz[i0]) / (sz[i0 + 1] - sz[i0])              # linear in z, extrapolating outside
        w0 = 1.0 - w1
        zre_slab = np.asarray(view_slab(zre, y, view), dtype=np.float32)
        a0 = sn[i0].slabs(y, zre_slab, view)
        a1 = sn[i0 + 1].slabs(y, zre_slab, view)
        for key, f0, f1 in zip(('density', 'q_z', 'xH_box', 'brightness_temp'), a0, a1):
            f = w0 * f0 + w1 * f1
            if use_rot:
                ic, jc = coords[cyc]
                f = map_coordinates(f, [ic, jc], order=1, mode=wrap_mode)
            out[key][:, :, j] = f
        if progress_every and (j + 1) % progress_every == 0:
            print(f"  slice {j + 1}/{n}  z={z:.3f}", flush=True)

    out.update(z_arr=z_arr, pos_axis=pos_axis, L_box_mpc_h=L_h, N=N, snap_z=sz,
               rho2_mean=stats[:, 0], rho2_min=stats[:, 1], rho2_frac_lt_0p05=stats[:, 2], rho2_frac_neg=stats[:, 3],
               geometry=geometry if wrap_cycle_seed is not None else 'none',
               views=np.array([[c, *v[1], v[0]] for c, v in sorted(views.items())], dtype=int))
    return out
