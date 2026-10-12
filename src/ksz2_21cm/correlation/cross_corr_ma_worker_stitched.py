# =============================================================================
# cross_corr_ma_worker_stitched.py
#
# Ma+2018-style cross-spectra on the stitched lightcone (NEW, 2026-10-12):
#
#     kSZ   x 21cm   (LINEAR in both)       -> 'C_lin'
#     kSZ^2 x 21cm   (21cm NOT squared)     -> 'C_sq'     (kSZ ell < ell_min_ksz
#                                                          removed BEFORE squaring,
#                                                          as in Ma Sec. 3.3)
#
# plus a cross-seed NULL for each: the same 21cm map correlated with the kSZ map of
# a DIFFERENT seed (independent realisation => true zero signal, same noise
# properties). 'C_lin_null', 'C_sq_null'.
#
# Three ways of turning the 3D 21cm lightcone into the 2D map at redshift z0
# (key 'variants'):
#   'slice'  : the single LOS cell nearest z0 (~3 Mpc; closest to Ma's
#              single-snapshot 21cm map at fixed x_e)
#   'narrow' : top-hat mean of T21 over z0 +/- narrow_dz/2 (default dz = 0.2, ~60 Mpc)
#   'slab'   : top-hat mean over z0 +/- slab_dz/2 (the Zhou chunk, dz = 2)
#
# NOTE: there is deliberately NO "k_par filter, then project" variant. Projecting
# (LOS-averaging) a LINEAR field is exactly its k_par = 0 mode, which the k_par
# filter removes, so that map is identically zero. In Zhou+25 the filter is applied
# BEFORE SQUARING, and the square is what survives the projection.
#
# The kSZ map and the 21cm brightness_temp are NOT touched/rebuilt: the kSZ map is the
# one from 02_compute_ksz_maps_stitched.py, brightness_temp is 21cmFAST's native field
# carried in the stitched .npz.
#
# Conventions (identical to cross_corr_sq_worker_stitched.py):
#   C(k) = Re(conj(F_kSZ-ish) F_T21) * pix_area / N^2, binned in |k| with the
#   caller's k_bins; ell = k * chi(z0). NO 1/chi^2 here (applied at plot time,
#   as in 06), chi(z0) is stored.
#
# Cache: <cache_dir>/seed_<N>/cross_corr_ma_stitched_seed<N>.npy
# =============================================================================

import os

import numpy as np


def _bin_mean(x2d, bidx, valid, nb, counts):
    """Mean of x2d in each |k| bin (bidx = bin index per pixel, valid = in range)."""
    s = np.bincount(bidx[valid], weights=x2d[valid], minlength=nb)
    out = np.full(nb, np.nan)
    ok = counts > 0
    out[ok] = s[ok] / counts[ok]
    return out


def compute_cross_corr_ma_for_seed_stitched(args):
    """
    args : tuple
      (seed, npz_path, kSZ_map, kSZ_map_null, main_cache_dir,
       npix_side, pix_size_Mpc, pix_area, kx_1d, ky_1d, kgrid, k_bins, k_centers,
       slab_dz, narrow_dz, z0_list, cosmo, ell_min_ksz, chi_ksz, tag)
    Returns (seed, result_dict_or_None, status_str).

    result_dict[z0] = {'z0','chi_z0','ell','k_centers','n_modes',
                       'variants': {name: {'C_lin','C_sq','C_lin_null','C_sq_null',
                                           'n_los'}}}
    """
    (seed, npz_path, kSZ_map, kSZ_map_null, main_cache_dir,
     npix_side, pix_size_Mpc, pix_area, kx_1d, ky_1d, kgrid, k_bins, k_centers,
     slab_dz, narrow_dz, z0_list, cosmo, ell_min_ksz, chi_ksz, tag) = args

    seed_dir = os.path.join(main_cache_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    cache = os.path.join(seed_dir, f"cross_corr_ma_stitched_seed{seed}{tag}.npy")
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

        # |k| bin index per pixel (shifted-FFT layout, same as kgrid)
        nb = len(k_centers)
        bidx_full = np.digitize(kgrid, k_bins) - 1
        valid = (bidx_full >= 0) & (bidx_full < nb)
        bidx = np.where(valid, bidx_full, 0)
        counts = np.bincount(bidx[valid], minlength=nb).astype(float)

        # ── kSZ maps: linear and "ell<ell_min removed, then squared" ─────────
        ell_grid_ksz = kgrid * chi_ksz
        keep = ell_grid_ksz >= ell_min_ksz

        def prep(m):
            m = np.asarray(m, dtype=np.float64)
            m = m - m.mean()
            F = np.fft.fftshift(np.fft.fft2(m))
            m_hp = np.real(np.fft.ifft2(np.fft.ifftshift(F * keep)))
            m2 = m_hp**2
            m2 -= m2.mean()
            return F, np.fft.fftshift(np.fft.fft2(m2))

        F_lin, F_sq = prep(kSZ_map)
        F_lin_n, F_sq_n = (prep(kSZ_map_null) if kSZ_map_null is not None
                           else (None, None))

        results = {}
        for z0 in z0_list:
            i_slice = int(np.argmin(np.abs(z_arr - z0)))
            maps, n_los = {"slice": np.asarray(bt[:, :, i_slice], dtype=np.float64)}, {"slice": 1}
            ok = True
            for name, dz in (("narrow", narrow_dz), ("slab", slab_dz)):
                idx = np.where((z_arr >= z0 - dz / 2.0) & (z_arr <= z0 + dz / 2.0))[0]
                if len(idx) < 1:
                    ok = False
                    break
                maps[name] = np.asarray(bt[:, :, idx], dtype=np.float64).mean(axis=2)
                n_los[name] = int(len(idx))
            if not ok or z0 - slab_dz / 2.0 < z_arr.min() or z0 + slab_dz / 2.0 > z_arr.max():
                continue

            chi_z0 = float(cosmo.comoving_distance(z0).value)
            out = {"z0": float(z0), "chi_z0": chi_z0,
                   "ell": k_centers * chi_z0, "k_centers": k_centers,
                   "n_modes": counts, "slab_dz": float(slab_dz), "narrow_dz": float(narrow_dz),
                   "i_slice": i_slice, "variants": {}}
            for name, m in maps.items():
                FT = np.fft.fftshift(np.fft.fft2(m - m.mean()))
                v = {"n_los": n_los[name]}
                v["C_lin"] = _bin_mean(np.real(np.conj(F_lin) * FT) * norm,
                                       bidx, valid, nb, counts)
                v["C_sq"] = _bin_mean(np.real(np.conj(F_sq) * FT) * norm,
                                      bidx, valid, nb, counts)
                if F_lin_n is not None:
                    v["C_lin_null"] = _bin_mean(
                        np.real(np.conj(F_lin_n) * FT) * norm, bidx, valid, nb, counts)
                    v["C_sq_null"] = _bin_mean(
                        np.real(np.conj(F_sq_n) * FT) * norm, bidx, valid, nb, counts)
                out["variants"][name] = v
            results[float(z0)] = out

        np.save(cache, results)
        return (seed, results, f"computed ({len(results)} z0)")
    except Exception as e:
        import traceback
        return (seed, None, f"failed: {e}\n{traceback.format_exc()}")
