#!/usr/bin/env python
# =============================================================================
# 02b_compute_ksz_posteor.py   (NEW, 2026-10-14)
#
# Post-EoR test, step 2: the kSZ map from everything the "patchy" mask in
# 02_compute_ksz_maps_stitched.py leaves out, i.e. the fully ionized gas
#     (a) the slices of the existing lightcone with global x_e > 0.9999 (z ~ 4 up to the end of
#         the patchy window), and
#     (b) the new z < 4 slices from 01b_stitch_lowz.py.
#
# It uses the SAME kSZ code (ksz.lightcone_integral.compute_ksz_map) with the complement of the
# patchy mask. The kSZ integral is a sum along the line of sight, so
#       total map = existing EoR-only map + this map          (exactly additive; tested)
# Visibility exp(-tau): existing slices use the existing tau (as in 02); z < 4 slices use the
# analytic homogeneous tau (optical_depth.analytic_tau_below, which includes helium), so the two
# join continuously at z = 4.
#
# Output: <cache_dir>/kSZ_maps/kSZ_map_postEoR_seed<N>.npy   (dimensionless dT/T, like the others)
#
# Heavy-ish (~2.5 GB of arrays, processed in LOS blocks): run through PBS.
#   python scripts/02b_compute_ksz_posteor.py --config configs/variants/stitched_256.yaml --seeds 1
# =============================================================================
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from ksz2_21cm.utils.config import load_config, ensure_dirs
from ksz2_21cm.utils.constants import MPC_CM, ne0_cgs
from ksz2_21cm.ksz.optical_depth import (
    analytic_tau_below, compute_tau, compute_patchy_mask, compute_visibility,
)
from ksz2_21cm.ksz.lightcone_integral import compute_ksz_map, ksz_map_to_Dl


def ksz_chunked(dens1, xhii, vlos, z, ds, vis, ne0, mask, block=300):
    """compute_ksz_map over LOS blocks (one slice of overlap, so every pair is counted once)."""
    out = np.zeros(dens1.shape[:2], dtype=np.float64)
    n, a = dens1.shape[2], 0
    while a < n - 1:
        b = min(a + block, n - 1)
        out += compute_ksz_map(
            dens1[:, :, a:b + 1], xhii[:, :, a:b + 1], vlos[:, :, a:b + 1],
            red_axis=z[a:b + 1], ds=ds[a:b], visibility_3D=vis[None, None, a:b + 1],
            ne0=ne0, patchy_mask_3D=mask[None, None, a:b + 1])
        a = b
    return out


def posteor_map(main, low, ne0, block=300, verbose=True):
    """main / low: dicts with arrays. Returns (post_map, info)."""
    z_arr, pos_axis = main["z_arr"], main["pos_axis"]
    x_e = 1.0 - main["xH_box"].mean(axis=(0, 1))
    mask, _ = compute_patchy_mask(x_e)
    if mask.max() == 0:
        raise RuntimeError("patchy mask is empty -- x_e never in the patchy window?")
    i_p = int(np.argmax(mask > 0))              # lowest-z patchy slice
    n_keep = i_p + 1                            # keep it: the pair (i_p-1, i_p) straddles the cut
    mask_post_main = 1.0 - mask[:n_keep]

    # visibility: existing slices exactly as in 02; low-z slices analytic
    tau0 = analytic_tau_below(float(z_arr.min()))
    z_mid, _ds, _dtau, tau = compute_tau(x_e, z_arr, pos_axis, ne0=ne0, tau0=tau0)
    _tl, vis_main, _ = compute_visibility(tau, z_arr, z_mid)
    # compute_visibility pins tau = 0 at the very first slice (visibility 1), a quirk that
    # does not matter for the masked patchy map; here that slice is signal, so use the analytic tau0.
    vis_main = vis_main.copy(); vis_main[0] = np.exp(-tau0)
    zg = np.linspace(0.0, float(z_arr.min()), 300)
    tg = np.array([analytic_tau_below(z) for z in zg])
    vis_low = np.exp(-np.interp(low["z_arr"], zg, tg))

    z_ext = np.concatenate([low["z_arr"], z_arr[:n_keep]])
    pos_ext = np.concatenate([low["pos_axis"], pos_axis[:n_keep]])
    ds = np.diff(pos_ext)
    vis = np.concatenate([vis_low, vis_main[:n_keep]])
    mask_ext = np.concatenate([np.ones(len(low["z_arr"])), mask_post_main])

    dens1 = 1.0 + np.concatenate([low["density"], main["density"][:, :, :n_keep]], axis=2)
    xhii = np.concatenate([np.ones_like(low["density"]), 1.0 - main["xH_box"][:, :, :n_keep]], axis=2)
    vlos = np.concatenate([low["velocity_z"], main["velocity_z"][:, :, :n_keep]], axis=2) / MPC_CM
    m = ksz_chunked(dens1, xhii, vlos, z_ext, ds, vis, ne0, mask_ext, block)
    info = dict(i_p=i_p, n_keep=n_keep, z_patchy_end=float(z_arr[i_p]), n_low=len(low["z_arr"]),
                vis_join=(float(vis_low[-1]), float(vis_main[0])),
                tau_analytic_z0=float(tau0))
    return m, info


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/variants/stitched_256.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--block", type=int, default=300)
    p.add_argument("--chi-ksz", type=float, default=8504.0)
    p.add_argument("--delete-lowz", action="store_true",
                   help="delete stitched_lowz_seed<N>.npz (~1.1 GB) once the post-EoR map is saved (map is ~0.5 MB)")
    args = p.parse_args()
    cfg = load_config(args.config); ensure_dirs(cfg)
    cache_dir = cfg["paths"]["cache_dir"]; L = cfg["coeval_ksz"]["box_len"]
    maps_dir = os.path.join(cache_dir, "kSZ_maps"); os.makedirs(maps_dir, exist_ok=True)
    ne0 = ne0_cgs()
    for seed in args.seeds:
        sd = os.path.join(cache_dir, f"seed_{seed}")
        out = os.path.join(maps_dir, f"kSZ_map_postEoR_seed{seed}.npy")
        if os.path.exists(out):
            print(f"seed {seed}: cached"); continue
        mp = os.path.join(sd, f"stitched_lightcone_seed{seed}.npz")
        lp = os.path.join(sd, f"stitched_lowz_seed{seed}.npz")
        eor_p = os.path.join(maps_dir, f"kSZ_map_stitched_seed{seed}.npy")
        assert os.path.exists(mp) and os.path.exists(lp), f"seed {seed}: need 01 and 01b outputs"
        d, l = np.load(mp), np.load(lp)
        main = {k: d[k] for k in ("z_arr", "pos_axis", "xH_box", "density", "velocity_z")}
        low = {k: l[k] for k in ("z_arr", "pos_axis", "density", "velocity_z")}
        m, info = posteor_map(main, low, ne0, args.block)
        np.save(out, m)
        print(f"seed {seed}: post-EoR map saved ({out})")
        if args.delete_lowz:
            os.remove(lp); print(f"   deleted {lp}")
        print(f"   patchy window ends at z = {info['z_patchy_end']:.2f}; post-EoR slices: "
              f"{info['n_low']} new (z<{main['z_arr'].min():.1f}) + {info['n_keep'] - 1} existing")
        print(f"   visibility at the join (low-z analytic vs existing): {info['vis_join'][0]:.5f} vs {info['vis_join'][1]:.5f}")
        print(f"   RMS dT/T:  post-EoR {np.sqrt(np.mean(m**2)):.3e}", end="")
        if os.path.exists(eor_p):
            e = np.load(eor_p)
            print(f"   EoR-only {np.sqrt(np.mean(e**2)):.3e}   total {np.sqrt(np.mean((e + m)**2)):.3e}")
            for nm, mm in (("EoR-only", e), ("post-EoR", m), ("total", e + m)):
                ell, D, _ = ksz_map_to_Dl(mm, L, chi_Mpc=args.chi_ksz)
                sel = [int(np.argmin(np.abs(ell - x))) for x in (500, 1000, 2000, 3000)]
                print(f"   D_l [uK^2] {nm:9s}: " + "  ".join(f"l={ell[i]:.0f}: {D[i]:.3g}" for i in sel))
        else:
            print()
    print("Done. Next: python scripts/14_posteor_comparison.py --seeds", *args.seeds)
