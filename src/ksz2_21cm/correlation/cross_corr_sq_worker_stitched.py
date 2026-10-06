# =============================================================================
# cross_corr_sq_worker_stitched.py
#
# NEW (2026-10-06) -- identical physics/numerics to
# correlation/cross_corr_sq_worker.py (kept frozen, untouched, as the
# comparison baseline), with ONE change: the 21cm field and its redshift
# axis come from the stitched .npz (01_stitch_lightcones.py's output) --
# brightness_temp / z_arr -- instead of a native py21cmfast LightCone's
# .brightness_temp / .lightcone_redshifts. The squaring/filtering/
# projection/cross-power math below is a straight copy of
# cross_corr_sq_worker.py, not a rewrite -- diff the two files if auditing
# for drift.
#
# ONE GENUINE IMPROVEMENT, not just a port: cross_corr_sq_worker.py uses
# pix_size_Mpc (the transverse pixel size) as an APPROXIMATION for the LOS
# slice spacing of the native lightcone. Here that's no longer an
# approximation -- 01_stitch_lightcones.py builds z_arr via
# build_los_z_grid(..., cell_size=BOX_LEN/HII_DIM), i.e. the LOS spacing is
# EXACTLY BOX_LEN/HII_DIM = pix_size_Mpc by construction. Noted inline
# below where it's used.
#
# Cache: <main_cache_dir>/seed_<N>/cross_corr_sq_stitched_seed<N>.npy
#        (separate filename from the native-lightcone worker's cache, so
#        the two pathways' results never collide or silently overwrite
#        each other.)
# =============================================================================

import numpy as np
import os


def compute_cross_corr_sq_for_seed_stitched(args):
    """
    Compute the kSZ^2 x 21cm^2 cross-correlation power spectrum for ONE
    seed, across all redshift chunks (z0, delta_z), reading from the
    stitched-lightcone .npz instead of a native py21cmfast LightCone.

    Parameters
    ----------
    args : tuple
        (seed, npz_path, kSZ_map, main_cache_dir,
         npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
         kx_1d, ky_1d, kgrid, k_bins, k_centers,
         k_par_min, delta_z, z_chunk_centres, cosmo)

        npz_path : path to the seed's stitched_lightcone_seed<N>.npz
                   (from 01_stitch_lightcones.py), carrying at least
                   'brightness_temp' and 'z_arr'.
        cosmo    : astropy cosmology object, used only for chi(z0) -> ell.
                   Pass the SAME object as everywhere else in the pipeline.

    Returns
    -------
    (seed, result_dict_or_None, status_message_str)
    result_dict is keyed by z0 (float), same schema as
    cross_corr_sq_worker.py's.
    """
    (seed, npz_path, kSZ_map, main_cache_dir,
     npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
     kx_1d, ky_1d, kgrid, k_bins, k_centers,
     k_par_min, delta_z, z_chunk_centres, cosmo) = args

    seed_cache_dir = os.path.join(main_cache_dir, f"seed_{seed}")
    os.makedirs(seed_cache_dir, exist_ok=True)
    cc_cache_sq = os.path.join(seed_cache_dir,
                                f"cross_corr_sq_stitched_seed{seed}.npy")

    if os.path.exists(cc_cache_sq):
        try:
            result = np.load(cc_cache_sq, allow_pickle=True).item()
            return (seed, result, f"cached ({len(result)} chunks)")
        except Exception:
            pass  # corrupt cache — fall through to recompute

    if kSZ_map is None:
        return (seed, None, "no kSZ map available")

    try:
        d = np.load(npz_path)
        brightness_temp = d["brightness_temp"]
        lc_redshifts = np.asarray(d["z_arr"], dtype=np.float64)
    except Exception as e:
        return (seed, None, f"failed to load stitched lightcone: {e}")

    try:
        # ── kSZ^2 map — computed once, reused for all chunks ─────────────────
        kSZ2_map          = kSZ_map**2
        kSZ2_map_centered = kSZ2_map - np.mean(kSZ2_map)
        fft_kSZ2_shifted  = np.fft.fftshift(np.fft.fft2(kSZ2_map_centered))
        auto_kSZ2_ps2d    = (np.abs(fft_kSZ2_shifted)**2
                             * pix_area / npix_side**2)

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
                brightness_temp[:, :, idx_chunk], dtype=np.float64
            )
            n_los        = T21_chunk.shape[2]
            # EXACT here (not approximate, see module docstring): z_arr was
            # built with LOS spacing == BOX_LEN/HII_DIM == pix_size_Mpc.
            pix_size_los = pix_size_Mpc

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
