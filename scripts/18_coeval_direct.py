#!/usr/bin/env python
# =============================================================================
# 18_coeval_direct.py
#
# kSZ x 21cm statistics straight from the 21cmFAST COEVAL BOXES of one seed (no stitching),
# to compare with the stitched-lightcone results of scripts 04 (Zhou: kSZ^2 x 21cm^2) and
# 04b (Ma: kSZ x 21cm and kSZ^2 x 21cm). Estimators: src/ksz2_21cm/correlation/coeval_direct.py
#   limber   : per-snapshot whole-box kernels (squeezed bispectrum, Kramer+25 eqs 9-11 analogue),
#              Limber-summed along the LOS. Kramer's eq 9 is derived from the SINGLE full kSZ map
#              squared, so this respects the single-map logic in expectation.
#   (No "slab" estimator: squaring per-snapshot slab maps instead of the one full map drops the
#    kSZ just beyond each slab edge and biases low -- removed 2026-10-26.)
#   maeq8    : Ma+18 eq 8 Limber for LINEAR kSZ x 21cm, P_{delta T21} measured from the box
#   maeq8_lin: the same with Ma's linearised T21 = T0[(1-x_e) delta - x_e delta_x]
#
# The boxes are REGENERATED with run_coeval_fields(z, random_seed=seed) -- the same call,
# seed and parameters 01_stitch_lightcones.py used, so they are the same realisation as the
# stitched lightcone (21cmFAST is deterministic for a fixed seed). 01 deleted its .h5 cache
# after stitching, hence the rerun. Only snapshots up to --snap-zmax are needed (chunks
# 5.5..11.5 with dz = 2 reach z = 12.5); the .h5 cache this run writes (~1 GB per snapshot
# at 256^3) is deleted at the end unless --keep-h5.
#
# Output: <cache_dir>/seed_<N>/coeval_direct_seed<N>.npy  (dict, see the end of main()).
#
#   python scripts/18_coeval_direct.py --config configs/variants/stitched_256.yaml --seed 1
#   (PBS: pbs/run_coeval_direct.pbs, one seed per job)
# Then: python scripts/19_plot_coeval_direct.py --config configs/variants/stitched_256.yaml
# =============================================================================

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from ksz2_21cm.correlation import coeval_direct as cd
from ksz2_21cm.ksz.optical_depth import analytic_tau_below
from ksz2_21cm.simulate.stitch_from_coeval import build_los_z_grid, comoving_distance_mpc
from ksz2_21cm.utils.config import load_config
from ksz2_21cm.utils.constants import C_CGS, ne0_cgs
from ksz2_21cm.utils.cosmology import get_cosmology

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/variants/stitched_256.yaml")
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--z0", type=float, nargs="+", default=None,
                help="chunk centres (default: cross_correlation.z_chunk_centres)")
ap.add_argument("--slab-dz", type=float, default=None, help="chunk width (default cross_correlation.delta_z)")
ap.add_argument("--narrow-dz", type=float, default=0.2, help="04b 'narrow' window width")
ap.add_argument("--ell-min-ksz", type=float, default=100.0, help="04b: kSZ ell below this removed before squaring")
ap.add_argument("--chi-ksz", type=float, default=8504.0, help="04b: chi_eff [Mpc] for that cut")
ap.add_argument("--snap-zmax", type=float, default=13.0, help="highest snapshot redshift to run")
ap.add_argument("--work-dir", default=None,
                help="21cmFAST .h5 cache for this run (default <cache_dir>/../coeval_direct_h5/seed_<N>)")
ap.add_argument("--keep-h5", action="store_true")
ap.add_argument("--out", default=None)
ap.add_argument("--fake-boxes", action="store_true",
                help="TEST ONLY: synthetic boxes instead of 21cmFAST (for an end-to-end dry run)")
args = ap.parse_args()

cfg = load_config(args.config)
cache_dir = os.path.expanduser(cfg["paths"]["cache_dir"])
HII_DIM, BOX_LEN = int(cfg["simulation"]["hii_dim"]), float(cfg["simulation"]["box_len"])
assert HII_DIM == int(cfg["coeval_ksz"]["hii_dim"]) and BOX_LEN == float(cfg["coeval_ksz"]["box_len"])
cell = BOX_LEN / HII_DIM
ccfg = cfg["cross_correlation"]
k_par_min = float(ccfg["k_par_min"])
slab_dz = args.slab_dz if args.slab_dz is not None else float(ccfg["delta_z"])
z0_list = [float(z) for z in (args.z0 if args.z0 else ccfg["z_chunk_centres"])]
k_hp = args.ell_min_ksz / args.chi_ksz
cosmo = get_cosmology(cfg)
NE0 = ne0_cgs()
seed = args.seed

Z_ALL = sorted(float(z) for z in cfg["coeval_ksz"]["z_snapshots"])
snaps = [z for z in Z_ALL if z <= args.snap_zmax + 1e-9]
need_hi = max(z0_list) + slab_dz / 2.0
assert snaps[-1] >= need_hi, f"--snap-zmax {args.snap_zmax} below the highest chunk edge {need_hi}"
work_dir = args.work_dir or os.path.join(os.path.dirname(cache_dir.rstrip("/")), "coeval_direct_h5", f"seed_{seed}")
out_path = args.out or os.path.join(cache_dir, f"seed_{seed}", f"coeval_direct_seed{seed}.npy")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
os.makedirs(work_dir, exist_ok=True)
print(f"seed {seed}: {len(snaps)} snapshots z = {snaps[0]}..{snaps[-1]}, chunks z0 = {z0_list}, dz = {slab_dz}")
print(f"  box {BOX_LEN} Mpc / {HII_DIM}^3, k_par_min = {k_par_min}, kSZ high-pass |k| >= {k_hp:.4g}/Mpc (Ma sq)")
print(f"  21cmFAST cache: {work_dir}   output: {out_path}")


def fake_snapshot(z, n, rng_seed):
    """TEST ONLY: smooth-ish random fields with a z-dependent ionized fraction."""
    rng = np.random.default_rng(rng_seed + int(round(z * 100)))
    from scipy.ndimage import gaussian_filter
    d = gaussian_filter(rng.standard_normal((n, n, n)), 1.5, mode="wrap") * 3.0
    xe_bar = float(np.clip(1.0 - (z - 6.0) / 6.0, 0.0, 1.0))
    xH = (gaussian_filter(rng.standard_normal((n, n, n)), 2.0, mode="wrap") + d * 0.3 > np.quantile(d, xe_bar)) * 1.0 \
        if 0 < xe_bar < 1 else np.full((n, n, n), 1.0 - xe_bar)
    vz = gaussian_filter(rng.standard_normal((n, n, n)), 3.0, mode="wrap") * 3e7
    tb = 25.0 * xH * (1.0 + d)
    return d.astype(np.float32), xH.astype(np.float32), vz.astype(np.float32), tb.astype(np.float32)


def load_snapshot(z):
    if args.fake_boxes:
        return fake_snapshot(z, HII_DIM, seed)
    from ksz2_21cm.simulate.coeval_fields import run_coeval_fields
    delta, xH, _vx, _vy, vz, dTb = run_coeval_fields(z, HII_DIM, BOX_LEN, work_dir, random_seed=seed)
    return (np.asarray(delta, np.float32), np.asarray(xH, np.float32),
            np.asarray(vz, np.float32), np.asarray(dTb, np.float32))


dk = 2 * np.pi / BOX_LEN
_kx = np.fft.fftshift(np.fft.fftfreq(HII_DIM)) * HII_DIM * dk
_kmax = np.sqrt(2) * np.abs(_kx).max()
k_bins = np.logspace(np.log10(dk), np.log10(_kmax * 0.9), 35)          # same as 04 / 04b
binner = cd.Binner(HII_DIM, cell, k_bins)
k_c = binner.k_centers
k_edges3 = np.logspace(np.log10(dk), np.log10(np.sqrt(3) * np.pi / cell), 60)

# ── pass over snapshots: per-snapshot spectra and kernels (boxes are not kept) ──
t0 = time.time()
META = []
for z in snaps:
    delta, xH, vz, tb = load_snapshot(z)
    xHII = 1.0 - xH
    theta = ((1.0 + delta) * xHII * vz / np.float32(C_CGS)).astype(np.float32)     # (1+delta) x_e v/c
    xe = float(xHII.mean())
    denom = float(np.mean(xH * (1.0 + delta)))
    T0 = float(tb.mean()) / denom if denom > 0 else np.nan
    P_dT = cd.power3d_cross(delta, tb, BOX_LEN, k_edges3)
    P_dd = cd.power3d_cross(delta, delta, BOX_LEN, k_edges3)
    if xe > 1e-6:
        P_ddx = cd.power3d_cross(delta, xHII / xe - 1.0, BOX_LEN, k_edges3)
    else:
        P_ddx = np.zeros_like(P_dd)
    kap = cd.limber_kernels(theta, tb, binner, cell, k_par_min, k_hp)
    META.append({"z": z, "x_e": xe, "T0": T0, "Tb_mean": float(tb.mean()),
                 "P_dT": P_dT, "P_dd": P_dd, "P_ddx": P_ddx,
                 "kappa_zhou": kap["zhou"], "kappa_masq": kap["masq"]})
    print(f"  snapshot z={z:5.2f}: x_e={xe:.3f}  <Tb>={tb.mean():6.2f} mK  T0={T0:6.2f} mK   ({time.time() - t0:.0f}s)", flush=True)

snap_z = np.array(snaps)
xe_snap = np.array([m["x_e"] for m in META])

# ── LOS grid, kSZ weights, background ────────────────────────────────────────
z_arr = build_los_z_grid(min(Z_ALL), max(Z_ALL), cell)               # the grid 01 stitched on
pos_axis = np.array([comoving_distance_mpc(z) for z in z_arr])
xe_pix = np.interp(z_arr, snap_z, xe_snap)                           # flat beyond the last snapshot
tau0 = analytic_tau_below(float(z_arr.min()))
wts = cd.ksz_pixel_weights(z_arr, pos_axis, xe_pix, NE0, tau0)
W = wts["W"]
use = z_arr <= snap_z[-1] + 0.5
a_b, H_b, f_b, D_b = (np.full(len(z_arr), np.nan) for _ in range(4))
a_b[use], H_b[use], f_b[use], D_b[use] = cd.background(z_arr[use])
psi = np.full(len(z_arr), np.nan)
psi[use] = cd.psi_par(pos_axis[use], wts["W_per_mpc"][use], xe_pix[use], a_b[use], H_b[use], f_b[use], D_b[use])
assign = cd.nearest_snapshot(z_arr, snap_z)


def P_at_pixels(key, zpix):
    """(n_pix, nk) 3D spectrum `key` at the 2D k centres, linear in z between snapshots."""
    Pk_snap = np.array([cd.interp_P(k_edges3, m[key], k_c) for m in META])     # (n_snap, nk)
    out = np.empty((len(zpix), len(k_c)))
    for j in range(len(k_c)):
        out[:, j] = np.interp(zpix, snap_z, Pk_snap[:, j])
    return out


for m in META:   # Ma's linearised T21 cross spectrum: T0 [(1 - x_e) P_dd - x_e P_{d,dx}]
    m["P_dT_lin"] = m["T0"] * ((1.0 - m["x_e"]) * m["P_dd"] - m["x_e"] * m["P_ddx"]) if np.isfinite(m["T0"]) else np.full_like(m["P_dd"], np.nan)

# ── per chunk ────────────────────────────────────────────────────────────────
VARS = ("slice", "narrow", "slab")
results = {}
for z0 in z0_list:
    idx, wins = cd.chunk_and_windows(z_arr, z0, slab_dz, args.narrow_dz)
    if len(idx) < 3 or z_arr[idx].min() < snap_z[0] - 1e-9 or z_arr[idx].max() > snap_z[-1] + 0.25:
        print(f"  z0={z0}: chunk outside the snapshot range -- skipped")
        continue
    pos_in = {int(p): i for i, p in enumerate(idx)}
    groups = cd.contiguous_groups(assign, idx)
    lim = {"zhou": np.zeros(len(k_c))} | {f"masq_{v}": np.zeros(len(k_c)) for v in VARS}
    for si, g in groups:
        loc = np.array([pos_in[int(p)] for p in g])
        w_g = {v: wins[v][loc] for v in VARS}
        W2 = W[g] ** 2
        lim["zhou"] += np.sum(W2 * w_g["slab"]) * META[si]["kappa_zhou"]
        for v in VARS:
            lim[f"masq_{v}"] += np.sum(W2 * w_g[v]) * META[si]["kappa_masq"]
    variants = {}
    for v in VARS:
        sel = wins[v] > 0
        ip = idx[sel]
        variants[v] = {
            "sq_limber": lim[f"masq_{v}"],
            "lin_maeq8": cd.maeq8_prediction(k_c, psi[ip], wins[v][sel], P_at_pixels("P_dT", z_arr[ip])),
            "lin_maeq8_lin": cd.maeq8_prediction(k_c, psi[ip], wins[v][sel], P_at_pixels("P_dT_lin", z_arr[ip])),
        }
    chi_z0 = float(cosmo.comoving_distance(z0).value)
    results[float(z0)] = {
        "z0": float(z0), "chi_z0": chi_z0, "ell": k_c * chi_z0, "k_centers": k_c,
        "x_e_z0": float(np.interp(z0, snap_z, xe_snap)),
        "zhou": {"limber": lim["zhou"]},
        "variants": variants,
        "groups": [(float(snap_z[si]), int(len(g))) for si, g in groups], "n_chunk": int(len(idx)),
    }
    print(f"  z0={z0:5.2f}: {len(idx)} pixels from {len(groups)} snapshot(s) {results[float(z0)]['groups']}", flush=True)

out = {
    "seed": seed, "results": results,
    "meta": {"snap_z": snap_z, "x_e": xe_snap, "T0": np.array([m["T0"] for m in META]),
             "Tb_mean": np.array([m["Tb_mean"] for m in META]), "k_edges3": k_edges3,
             "P_dT": np.array([m["P_dT"] for m in META]), "P_dT_lin": np.array([m["P_dT_lin"] for m in META]),
             "P_dd": np.array([m["P_dd"] for m in META]), "P_ddx": np.array([m["P_ddx"] for m in META]),
             "kappa_zhou": np.array([m["kappa_zhou"] for m in META]),
             "kappa_masq": np.array([m["kappa_masq"] for m in META]),
             "k_centers": k_c, "z_arr": z_arr[use], "W": W[use], "psi_par": psi[use], "x_e_pix": xe_pix[use],
             "slab_dz": slab_dz, "narrow_dz": args.narrow_dz, "k_par_min": k_par_min,
             "ell_min_ksz": args.ell_min_ksz, "chi_ksz": args.chi_ksz, "fake_boxes": bool(args.fake_boxes)},
}
tmp = out_path + ".tmp.npy"
np.save(tmp, out, allow_pickle=True)
os.replace(tmp, out_path)
print(f"saved {out_path}  ({len(results)} chunks, {time.time() - t0:.0f}s)")

if not args.keep_h5 and not args.fake_boxes:
    n, freed = 0, 0
    for p in glob.glob(os.path.join(work_dir, "*.h5")):
        try:
            freed += os.path.getsize(p)
            os.remove(p)
            n += 1
        except OSError:
            pass
    print(f"deleted {n} .h5 files ({freed / 1e9:.1f} GB) from {work_dir}")
