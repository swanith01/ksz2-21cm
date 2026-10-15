# =============================================================================
# stitch_below.py  (NEW, 2026-10-14)
#
# Extend an existing stitched lightcone (built by stitch_from_coeval.py, z = z_ref ... 20)
# DOWNWARD to z ~ 0, WITHOUT touching or re-deriving anything at z >= z_ref.
#
# Why a separate module: stitch_from_coeval.stitch_field() measures every LOS pixel and
# every wrap cycle relative to z0 = z_arr.min(). Rebuilding the whole lightcone down to
# z = 0 would change z0 and silently shift/rotate the z >= 4 part, invalidating every
# cached kSZ map and cross-spectrum. Here the reference stays z_ref (= 4.0) and
# the new pixels simply continue the same lattice with negative distances:
#
#   d      = chi(z) - chi(z_ref)                  (< 0 below z_ref)
#   y_cell = round(d / cell) % ngrid              (python modulo -> always >= 0)
#   cycle  = floor(d / box_len)                   (-1, -2, ... below z_ref)
#   angle  = cycle_angle(cycle, seed)  for cycle >= 0   (identical to the vendored code)
#          = cycle_angle(NEG_CYCLE_OFFSET - cycle, seed) for cycle < 0 (new, distinct angles;
#            np.random.default_rng does not accept negative seeds)
#
# For d >= 0 this reproduces the vendored stitch_field() exactly (tested), so it can be
# checked against the original on the existing grid. The vendored file is NOT modified.
# Linear interpolation across snapshots uses only the two bracketing snapshots, which is
# identical to the vendored interp1d over all snapshots for z inside the snapshot range.
# =============================================================================

import numpy as np

from .stitch_from_coeval import (
    comoving_distance_mpc, rotated_coords, get_slab_bilinear, cycle_angle,
)

NEG_CYCLE_OFFSET = 1000


def signed_cycle_angle(cycle_idx, seed):
    """Rotation angle [deg] for wrap cycle `cycle_idx` (may be negative)."""
    if cycle_idx >= 0:
        return cycle_angle(cycle_idx, seed)
    return cycle_angle(NEG_CYCLE_OFFSET - cycle_idx, seed)


def build_low_z_grid(z_ref, z_floor, cell_size, z_oversample=4000):
    """
    LOS redshift grid BELOW z_ref, uniform comoving spacing == cell_size, on the SAME
    lattice as the existing grid (chi = chi(z_ref) - k*cell_size, k = 1, 2, ...).
    z_ref itself is excluded (the existing lightcone already has it). Ascending in z.
    """
    z_fine = np.linspace(z_floor, z_ref, z_oversample)
    chi_fine = np.array([comoving_distance_mpc(z) for z in z_fine])
    chi_ref = chi_fine[-1]
    n = int(np.floor((chi_ref - chi_fine[0]) / cell_size))
    chi_target = chi_ref - cell_size * np.arange(n, 0, -1)
    return np.interp(chi_target, chi_fine, z_fine)


def low_z_snapshots(z_ref, z_floor, max_step_mpc, z_oversample=4000):
    """
    Snapshot redshifts from z_floor up to z_ref (both included) spaced equally in
    comoving distance with steps <= max_step_mpc. Returns an ascending array.
    """
    z_fine = np.linspace(z_floor, z_ref, z_oversample)
    chi_fine = np.array([comoving_distance_mpc(z) for z in z_fine])
    n_int = int(np.ceil((chi_fine[-1] - chi_fine[0]) / max_step_mpc))
    chi_t = np.linspace(chi_fine[0], chi_fine[-1], n_int + 1)
    z = np.interp(chi_t, chi_fine, z_fine)
    z[0], z[-1] = z_floor, z_ref
    return np.round(z, 4)


def stitch_fields_signed(boxes_by_field, snap_z, z_arr, z_ref, cell_size, ngrid,
                         wrap_cycle_seed, wrap_mode='wrap'):
    """
    boxes_by_field : {field_name: {z_snapshot: ndarray (ngrid, ngrid, ngrid)}}
    snap_z         : ascending array of snapshot redshifts (keys of the inner dicts)
    z_arr          : target LOS redshifts (any order)
    z_ref          : reference redshift of the EXISTING lightcone (its z_arr.min())
    Returns {field_name: ndarray (ngrid, ngrid, len(z_arr)) float32}.
    All fields share one geometry (y_cell, cycle rotation) at every pixel.
    """
    snap_z = np.asarray(snap_z, dtype=float)
    chi_ref = comoving_distance_mpc(z_ref)
    box_len = cell_size * ngrid
    out = {f: np.empty((ngrid, ngrid, len(z_arr)), dtype=np.float32) for f in boxes_by_field}
    coords_by_cycle = {}
    for n, z in enumerate(z_arr):
        d = comoving_distance_mpc(z) - chi_ref
        y_cell = int(round(d / cell_size)) % ngrid
        cycle = int(d // box_len)
        if cycle not in coords_by_cycle:
            coords_by_cycle[cycle] = rotated_coords(ngrid, signed_cycle_angle(cycle, wrap_cycle_seed))
        ic, jc = coords_by_cycle[cycle]
        j = int(np.clip(np.searchsorted(snap_z, z) , 1, len(snap_z) - 1))
        z_lo, z_hi = snap_z[j - 1], snap_z[j]
        w = (z - z_lo) / (z_hi - z_lo)               # linear interpolation / extrapolation
        for f, boxes in boxes_by_field.items():
            s_lo = get_slab_bilinear(boxes[z_lo], y_cell, ic, jc, wrap_mode).astype(np.float64)
            s_hi = get_slab_bilinear(boxes[z_hi], y_cell, ic, jc, wrap_mode).astype(np.float64)
            out[f][:, :, n] = (1.0 - w) * s_lo + w * s_hi
    return out
