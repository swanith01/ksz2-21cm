# =============================================================================
# cross_corr_wedge_worker_stitched.py  (NEW, 2026-10-12)
#
# Does kSZ^2 x 21cm (21cm NOT squared) survive foreground removal?  The reason Zhou+25
# square the 21cm field is that the foreground filter removes exactly the modes
# (small k_par) that a Limber cross-correlation with the projected kSZ^2 map relies on.
# This worker puts the linear and the squared 21cm statistics side by side, each with
# three levels of filtering of the 21cm chunk (Zhou chunk z0 +/- delta_z/2):
#
#   filter 'none'  : no filter
#   filter 'kpar'  : k_par > k_par_min                       (what 04 / Zhou-style uses)
#   filter 'wedge' : k_par > max(k_par_min, s * k_perp),
#                    s = wedge_fraction * chi(z0) H(z0) / (c (1+z0))   (horizon wedge
#                    for wedge_fraction = 1)
#
# and two ways of getting a 2D 21cm map out of the filtered 3D field F:
#   'lin_<f>' : F at the chunk-centre LOS cell        (21cm NOT squared)
#   'sq_<f>'  : mean over the LOS of F^2              (Zhou+25: square, THEN project)
#               (a LOS mean of the *linear* filtered field is identically zero, since the
#                filter removes its k_par = 0 mode -- hence the central slice above)
#
# Each 2D map is crossed with kSZ^2 (kSZ ell < ell_min_ksz removed before squaring) and
# with the kSZ^2 of a DIFFERENT seed (null). Stored per statistic: C_cross, C_null,
# P_T (21cm map auto), plus P_K (kSZ^2 auto) once, so the dimensionless correlation
# coefficient r = C_cross / sqrt(P_T P_K) can be formed (comparable between linear and
# squared statistics, whose units differ).
#
# Conventions identical to cross_corr_sq_worker_stitched.py (no 1/chi^2 here).
# With ell_min_ksz = 0 (raw map squared), 'sq_kpar' reproduces that worker's C_cross exactly (tested).
#
# Cache: <cache_dir>/seed_<N>/cross_corr_wedge_stitched_seed<N><tag>.npy
# =============================================================================

import os

import numpy as np

C_KMS = 299792.458
FILTERS = ("none", "kpar", "wedge")


def _bin_mean(x2d, bidx, valid, nb, counts):
    s = np.bincount(bidx[valid], weights=x2d[valid], minlength=nb)
    out = np.full(nb, np.nan)
    ok = counts > 0
    out[ok] = s[ok] / counts[ok]
    return out


def compute_cross_corr_wedge_for_seed_stitched(args):
    """
    args : (seed, npz_path, kSZ_map, kSZ_map_null, main_cache_dir, npix_side,
            pix_size_Mpc, pix_area, kx_1d, ky_1d, kgrid, k_bins, k_centers,
            k_par_min, delta_z, z0_list, cosmo, ell_min_ksz, chi_ksz,
            wedge_fraction, tag)
    Returns (seed, result_dict_or_None, status).
    result[z0] = {'z0','chi_z0','ell','n_modes','wedge_slope','frac_modes':{f:..},
                  'P_K': (nb,), 'stats': {name: {'C','C_null','P_T'}}}
    """
    (seed, npz_path, kSZ_map, kSZ_map_null, main_cache_dir, npix_side,
     pix_size_Mpc, pix_area, kx_1d, ky_1d, kgrid, k_bins, k_centers,
     k_par_min, delta_z, z0_list, cosmo, ell_min_ksz, chi_ksz,
     wedge_fraction, tag) = args

    seed_dir = os.path.join(main_cache_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    cache = os.path.join(seed_dir, f"cross_corr_wedge_stitched_seed{seed}{tag}.npy")
    if os.path.exists(cache):
        try:
            res = np.load(cache, allow_pickle=True).item()
            return (seed, res, f"cached ({len(res)} z0)")
        except Exception:
            pass
    if kSZ_map is None:
        return (seed, None, "no kSZ map available")

    try:
        d = np.load(npz_path)
        bt = d["brightness_temp"]
        z_arr = np.asarray(d["z_arr"], dtype=np.float64)
    except Exception as e:
        return (seed, None, f"failed to load stitched lightcone: {e}")

    try:
        N = npix_side
        norm = pix_area / N**2
        nb = len(k_centers)
        bidx_full = np.digitize(kgrid, k_bins) - 1
        valid = (bidx_full >= 0) & (bidx_full < nb)
        bidx = np.where(valid, bidx_full, 0)
        counts = np.bincount(bidx[valid], minlength=nb).astype(float)
        keep = (kgrid * chi_ksz) >= ell_min_ksz

        def ksz2_fft(m):
            m = np.asarray(m, dtype=np.float64)
            if ell_min_ksz > 0:          # high-pass (also removes the k=0 mean), then square
                F = np.fft.fftshift(np.fft.fft2(m))
                h = np.real(np.fft.ifft2(np.fft.ifftshift(F * keep))) ** 2
            else:                        # no cut: square the raw map (as 04's worker does)
                h = m ** 2
            h = h - h.mean()
            return np.fft.fftshift(np.fft.fft2(h))

        FK = ksz2_fft(kSZ_map)
        FKn = ksz2_fft(kSZ_map_null) if kSZ_map_null is not None else None
        P_K = _bin_mean(np.abs(FK) ** 2 * norm, bidx, valid, nb, counts)

        kperp3 = np.sqrt(kx_1d[:, None, None] ** 2 + ky_1d[None, :, None] ** 2)
        results = {}
        for z0 in z0_list:
            idx = np.where((z_arr >= z0 - delta_z / 2.0) & (z_arr <= z0 + delta_z / 2.0))[0]
            if len(idx) < 3 or z0 - delta_z / 2.0 < z_arr.min() or z0 + delta_z / 2.0 > z_arr.max():
                continue
            ic = int(np.argmin(np.abs(z_arr[idx] - z0)))
            chunk = np.asarray(bt[:, :, idx], dtype=np.float64)
            n_los = chunk.shape[2]
            F3 = np.fft.fftn(chunk)
            kpar = np.abs(np.fft.fftfreq(n_los, d=pix_size_Mpc) * 2 * np.pi)[None, None, :]

            chi_z0 = float(cosmo.comoving_distance(z0).value)
            Hz = float(cosmo.H(z0).value)                       # km/s/Mpc
            slope = wedge_fraction * chi_z0 * Hz / (C_KMS * (1.0 + z0))
            masks = {
                "none": None,
                "kpar": kpar > k_par_min,
                "wedge": kpar > np.maximum(k_par_min, slope * kperp3),
            }
            frac = {f: (1.0 if m is None else float(np.mean(m))) for f, m in masks.items()}

            stats = {}
            for f in FILTERS:
                m = masks[f]
                field = chunk if m is None else np.real(np.fft.ifftn(F3 * m))
                for kind, m2d in (("lin", field[:, :, ic]), ("sq", (field ** 2).mean(axis=2))):
                    FT = np.fft.fftshift(np.fft.fft2(m2d - m2d.mean()))
                    st = {
                        "C": _bin_mean(np.real(np.conj(FK) * FT) * norm, bidx, valid, nb, counts),
                        "P_T": _bin_mean(np.abs(FT) ** 2 * norm, bidx, valid, nb, counts),
                    }
                    if FKn is not None:
                        st["C_null"] = _bin_mean(np.real(np.conj(FKn) * FT) * norm,
                                                 bidx, valid, nb, counts)
                    stats[f"{kind}_{f}"] = st
                del field
            del F3, chunk
            results[float(z0)] = {
                "z0": float(z0), "chi_z0": chi_z0, "ell": k_centers * chi_z0,
                "n_modes": counts, "wedge_slope": float(slope), "frac_modes": frac,
                "P_K": P_K, "stats": stats, "n_los": n_los,
            }
        np.save(cache, results)
        return (seed, results, f"computed ({len(results)} z0)")
    except Exception as e:
        import traceback
        return (seed, None, f"failed: {e}\n{traceback.format_exc()}")
