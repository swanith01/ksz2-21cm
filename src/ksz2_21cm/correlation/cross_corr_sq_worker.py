# =============================================================================
# cross_corr_sq_worker.py
#
# [NEW — this module did not exist before the repo cleanup.]
#
# Parallel worker: kSZ^2 x 21cm^2 cross-correlation, one seed at a time.
# This is the Zhou+25-style SQUARED statistic — foreground-filter in 3D,
# IFFT, square in configuration space, top-hat project along the line of
# sight, then cross-power with the squared kSZ map.
#
# Extracted from CELL 7b of kSZ_Squared_21cm_11Jun_CLUSTER.py, which ran
# this same per-seed loop serially, in-process, with no caching-before-load
# guard. This module:
#   - mirrors the function signature / (seed, result, status) return
#     convention of cross_corr_worker.py so it can be dropped into the same
#     ProcessPoolExecutor pattern used for CELL 7 (unsquared), and
#   - checks the cache file before loading the lightcone, the same fix
#     already applied in cross_corr_worker.py.
#
# The physics/numerics are unchanged from CELL 7b — this is a mechanical
# extraction, not a rewrite. Cross-check the first few seeds against your
# existing cross_corr_sq_seed*.npy cache files before trusting new runs
# from this module, since it has not been executed on the cluster yet.
#
# Cache: <main_cache_dir>/seed_<N>/cross_corr_sq_seed<N>.npy
#        (dict keyed by z0 — identical format to before, so existing caches
#        remain valid and will just be loaded, not recomputed.)
# =============================================================================

import numpy as np
import os
import time


def compute_cross_corr_sq_for_seed(args):
    """
    Compute the kSZ^2 x 21cm^2 cross-correlation power spectrum for ONE seed,
    across all redshift chunks (z0, delta_z).

    Parameters
    ----------
    args : tuple
        (seed, cache_file, kSZ_map, main_cache_dir,
         npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
         kx_1d, ky_1d, kgrid, k_bins, k_centers,
         k_par_min, delta_z, z_chunk_centres, cosmo)

        cosmo : astropy cosmology object, used only for chi(z0) -> ell.
                Pass the SAME object as everywhere else in the pipeline —
                see src/ksz2_21cm/utils/cosmology.py::get_cosmology().

    Returns
    -------
    (seed, result_dict_or_None, status_message_str)
    result_dict is keyed by z0 (float), same schema as the original
    CELL 7b cross_corr_results_sq[z0] dict.
    """
    import py21cmfast as p21c

    (seed, cache_file, kSZ_map, main_cache_dir,
     npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
     kx_1d, ky_1d, kgrid, k_bins, k_centers,
     k_par_min, delta_z, z_chunk_centres, cosmo) = args

    seed_cache_dir = os.path.join(main_cache_dir, f"seed_{seed}")
    os.makedirs(seed_cache_dir, exist_ok=True)
    cc_cache_sq = os.path.join(seed_cache_dir, f"cross_corr_sq_seed{seed}.npy")

    # ── Check cache BEFORE loading the lightcone (same fix as CELL 7) ────────
    if os.path.exists(cc_cache_sq):
        try:
            result = np.load(cc_cache_sq, allow_pickle=True).item()
            return (seed, result, f"cached ({len(result)} chunks)")
        except Exception:
            pass  # corrupt cache — fall through to recompute

    if kSZ_map is None:
        return (seed, None, "no kSZ map available")

    try:
        lc = p21c.LightCone.read(cache_file)
    except Exception as e:
        return (seed, None, f"failed to load lightcone: {e}")

    try:
        # ── kSZ^2 map — computed once, reused for all chunks ─────────────────
        kSZ2_map          = kSZ_map**2
        kSZ2_map_centered = kSZ2_map - np.mean(kSZ2_map)
        fft_kSZ2_shifted  = np.fft.fftshift(np.fft.fft2(kSZ2_map_centered))
        auto_kSZ2_ps2d    = (np.abs(fft_kSZ2_shifted)**2
                             * pix_area / npix_side**2)

        lc_redshifts = np.asarray(lc.lightcone_redshifts, dtype=np.float64)
        cross_corr_results_sq = {}

        for z0 in z_chunk_centres:

            z_lo = z0 - delta_z / 2.0
            z_hi = z0 + delta_z / 2.0

            # ── 1. Extract 3D T21 chunk ───────────────────────────────────────
            idx_chunk = np.where(
                (lc_redshifts >= z_lo) & (lc_redshifts <= z_hi)
            )[0]
            if len(idx_chunk) < 3:
                continue

            T21_chunk = np.asarray(
                lc.brightness_temp[:, :, idx_chunk], dtype=np.float64
            )
            n_los        = T21_chunk.shape[2]
            pix_size_los = pix_size_Mpc  # approx: comoving Mpc per slice

            # ── 2. Foreground filter in 3D Fourier space ─────────────────────
            T21_fft3d = np.fft.fftn(T21_chunk)

            kz_1d = np.fft.fftfreq(n_los, d=pix_size_los) * 2 * np.pi
            kx_3d = kx_1d[:, None, None]
            ky_3d = ky_1d[None, :, None]
            kz_3d = kz_1d[None, None, :]
            kpar  = np.abs(kz_3d)

            fore_filter = (kpar > k_par_min).astype(float)
            T21_fft3d_filtered = T21_fft3d * fore_filter

            # ── 3. IFFT -> config space -> square (must square AFTER filter
            #        and BEFORE projection — Zhou+25 Appendix A) ──────────────
            T21_filtered = np.real(np.fft.ifftn(T21_fft3d_filtered))
            T21_sq_3d    = T21_filtered**2

            # ── 4. Top-hat project 3D -> 2D along the line of sight ──────────
            T21_sq_2d         = np.mean(T21_sq_3d, axis=2)
            T21_sq_centered   = T21_sq_2d - np.mean(T21_sq_2d)
            fft_T21sq_shifted = np.fft.fftshift(np.fft.fft2(T21_sq_centered))

            # ── 5. Cross-power: C_ell^{kSZ^2 x 21cm^2} and autos ─────────────
            cross_ps2d      = (np.real(np.conj(fft_kSZ2_shifted)
                                       * fft_T21sq_shifted)
                               * pix_area / npix_side**2)
            auto_T21sq_ps2d = (np.abs(fft_T21sq_shifted)**2
                               * pix_area / npix_side**2)

            nb          = len(k_centers)
            C_cross     = np.full(nb, np.nan)
            C_cross_err = np.full(nb, np.nan)
            P_T21sq     = np.full(nb, np.nan)
            P_kSZ2      = np.full(nb, np.nan)
            n_modes     = np.zeros(nb)

            for j in range(nb):
                mask  = (kgrid >= k_bins[j]) & (kgrid < k_bins[j + 1])
                n_pix = int(np.sum(mask))
                if n_pix == 0:
                    continue
                cv             = cross_ps2d[mask]
                C_cross[j]     = np.mean(cv)
                C_cross_err[j] = np.std(cv) / np.sqrt(n_pix)
                P_T21sq[j]     = np.mean(auto_T21sq_ps2d[mask])
                P_kSZ2[j]      = np.mean(auto_kSZ2_ps2d[mask])
                n_modes[j]     = n_pix

            # ── 6. Limber k -> ell at chunk centre; D_ell convention ─────────
            chi_z0  = float(cosmo.comoving_distance(z0).value)
            ell     = k_centers * chi_z0
            D_cross = ell * (ell + 1) * C_cross / (2 * np.pi)

            cross_corr_results_sq[float(z0)] = {
                'z0'          : float(z0),
                'delta_z'     : float(delta_z),
                'k_par_min'   : float(k_par_min),
                'n_los_slices': int(n_los),
                'k_centers'   : k_centers,
                'ell'         : ell,
                'C_cross'     : C_cross,
                'C_cross_err' : C_cross_err,
                'D_cross'     : D_cross,
                'P_T21sq'     : P_T21sq,
                'P_kSZ2'      : P_kSZ2,
                'n_modes'     : n_modes,
            }

        np.save(cc_cache_sq, cross_corr_results_sq)
        return (seed, cross_corr_results_sq,
                f"computed ({len(cross_corr_results_sq)} chunks)")

    except Exception as e:
        return (seed, None, f"failed: {str(e)}")
