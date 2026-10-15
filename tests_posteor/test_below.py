import sys, numpy as np
sys.path.insert(0, "/tmp/claude-0/-home-claude/cd4b2d82-d0d2-583c-9208-b8b93b82bcbd/scratchpad/t18/tree/src")
import importlib.util
from ksz2_21cm.simulate.stitch_from_coeval import build_los_z_grid, stitch_field, comoving_distance_mpc
from ksz2_21cm.simulate.stitch_below import (build_low_z_grid, low_z_snapshots, stitch_fields_signed, signed_cycle_angle)
rng = np.random.default_rng(1)
N, L = 16, 160.0; cell = L / N; seed = 3
# 1) equality with the vendored stitch_field on d>=0
z_arr = build_los_z_grid(4.0, 20.0, cell)[:400]          # > 1 wrap cycle (16 cells per cycle)
snaps = np.array([4.0, 5.0, 6.5, 8.0, 10.0, 13.0, 17.0, 20.0])
boxes = {f: {float(z): rng.standard_normal((N, N, N)).astype(np.float32) for z in snaps} for f in ("a", "b")}
ref = {f: stitch_field(boxes[f], snaps, z_arr, float(z_arr.min()), cell, N, wrap_cycle_seed=seed) for f in boxes}
new = stitch_fields_signed(boxes, snaps, z_arr, float(z_arr.min()), cell, N, wrap_cycle_seed=seed)
for f in boxes:
    err = np.max(np.abs(ref[f] - new[f])); print(f"1) field {f}: max |vendored - signed| = {err:.2e}  (field rms {ref[f].std():.2f})")
    assert err < 1e-5
# 2) lattice continuity of the low-z grid
z_low = build_low_z_grid(4.0, 0.001, cell)
d = np.array([comoving_distance_mpc(z) for z in z_low]) - comoving_distance_mpc(4.0)
frac = np.abs(d / cell - np.round(d / cell)).max()
print(f"2) low grid: {len(z_low)} pixels, z {z_low.min():.4f}..{z_low.max():.4f}, max offset from lattice = {frac:.1e} cells; last step to z_ref = {-d[-1]/cell:.3f} cells")
assert frac < 1e-3 and np.all(np.diff(z_low) > 0) and z_low.max() < 4.0
# 3) angles for negative cycles are valid and distinct from cycles >= 0
a_neg = [signed_cycle_angle(-k, seed) for k in range(1, 6)]; a_pos = [signed_cycle_angle(k, seed) for k in range(0, 6)]
print("3) neg-cycle angles:", np.round(a_neg, 1), " pos:", np.round(a_pos, 1)); assert len(set(a_neg + a_pos)) == 11
# 4) snapshot grid
for step in (340.0, 330.0):
    s = low_z_snapshots(4.0, 0.001, step); chi = np.array([comoving_distance_mpc(z) for z in s])
    print(f"4) max step {step:.0f}: {len(s)-1} new snapshots; actual steps {np.diff(chi).min():.0f}-{np.diff(chi).max():.0f} Mpc; z: {s[:4]} ... {s[-3:]}")
# 5) stitching low z: runs, shapes, no NaN; continuity of field statistic across z_ref not required
low_snaps = low_z_snapshots(4.0, 0.001, 340.0)
bx = {f: {float(z): rng.standard_normal((N, N, N)).astype(np.float32) for z in low_snaps} for f in ("density", "velocity_z")}
st = stitch_fields_signed(bx, low_snaps, z_low, 4.0, cell, N, wrap_cycle_seed=seed)
print("5) low-z stitch shapes:", {k: v.shape for k, v in st.items()}, "finite:", all(np.isfinite(v).all() for v in st.values()))
# consistency: at a pixel exactly on a snapshot redshift the value equals that snapshot's rotated slab -> check interpolation weight bounds via an affine test
const = {"c": {float(z): np.full((N, N, N), 2.0 * z + 1.0, np.float32) for z in low_snaps}}
stc = stitch_fields_signed(const, low_snaps, z_low, 4.0, cell, N, wrap_cycle_seed=seed)["c"]
print("   linear-in-z test: max |stitched - (2z+1)| =", float(np.max(np.abs(stc - (2 * z_low + 1)[None, None, :]))))
assert np.max(np.abs(stc - (2 * z_low + 1)[None, None, :])) < 1e-3
print("ALL OK")
