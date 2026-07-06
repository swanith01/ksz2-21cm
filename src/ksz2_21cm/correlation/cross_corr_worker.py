# =============================================================================
# cross_corr_worker.py
# Parallel worker: kSZ²-21cm cross-correlations, one seed at a time.
# LightCone is loaded from HDF5 inside the worker (never pickled).
#
# [repo note] This is the UNSQUARED kSZ x 21cm cross-correlation (CELL 7 of
# the old CLUSTER script; Ma+18 style). Moved here unchanged.
# See cross_corr_sq_worker.py in this same directory for the SQUARED
# kSZ^2 x 21cm^2 statistic (CELL 7b, Zhou+25 style) — that's the one your
# SNR forecasts and Zhou+25 comparison figures actually use.
# Imported by scripts/03_compute_cross_corr.py.
# =============================================================================

import numpy as np
import os
import time


def compute_cross_corr_for_seed(args):
    """
    Compute kSZ²-21cm cross-correlation power spectra for ONE seed
    across all node redshifts of its lightcone.

    Parameters
    ----------
    args : tuple
        (seed, cache_file, kSZ_map, z_obs, main_cache_dir,
         npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
         dk, kgrid, k_bins, k_centers)

    Returns
    -------
    (seed, result_dict_or_None, status_message_str)
    """

    import py21cmfast as p21c

    (seed, cache_file, kSZ_map, z_obs, main_cache_dir,
     npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
     dk, kgrid, k_bins, k_centers) = args

    seed_cache_dir = os.path.join(main_cache_dir, f"seed_{seed}")
    os.makedirs(seed_cache_dir, exist_ok=True)
    cc_cache = os.path.join(seed_cache_dir, f"cross_corr_seed{seed}.npy")

    # ── FIX 9: check the cross-corr cache BEFORE loading the lightcone ───────
    # Loading a multi-GB HDF5 just to discover results are already cached
    # wastes significant I/O time.
    if os.path.exists(cc_cache):
        try:
            result = np.load(cc_cache, allow_pickle=True).item()
            return (seed, result, f"cached ({len(result)} redshifts)")
        except Exception as e:
            pass  # corrupt cache — fall through to recompute

    # ── Validate kSZ map ─────────────────────────────────────────────────────
    if kSZ_map is None:
        return (seed, None, "no kSZ map available")

    # ── Load lightcone from HDF5 inside the worker ───────────────────────────
    try:
        lc = p21c.LightCone.read(cache_file)
    except Exception as e:
        return (seed, None, f"failed to load lightcone: {e}")

    try:
        # ── kSZ² map — computed once, reused for all redshift slices ─────────
        kSZ2_map          = kSZ_map**2
        kSZ2_map_centered = kSZ2_map - np.mean(kSZ2_map)
        fft_kSZ2_shifted  = np.fft.fftshift(np.fft.fft2(kSZ2_map_centered))
        auto_kSZ2_ps2d    = np.abs(fft_kSZ2_shifted)**2 * pix_area / npix_side**2

        # ── FIX 10: precompute the correct 2D mode-count factor ──────────────
        # The maps are 2D, so modes live in 2D annuli, not 3D shells.
        # n_modes_2d(k) = 2π k Δk (L / 2π)²
        # The old code used the 3D formula 4π k² Δk (L/2π)³ which
        # overestimates mode counts and underestimates cosmic-variance errors.
        L_over_2pi_sq = (box_size_Mpc / (2.0 * np.pi))**2

        # ── Loop over node redshifts ──────────────────────────────────────────
        node_redshifts = np.asarray(lc.node_redshifts[::-1], dtype=np.float64)
        lc_redshifts   = np.asarray(lc.lightcone_redshifts, dtype=np.float64)
        cross_corr_results = {}

        for z_21cm in node_redshifts:

            idx_closest = int(np.argmin(np.abs(lc_redshifts - z_21cm)))
            z_actual    = float(lc_redshifts[idx_closest])

            # 21cm brightness-temperature slice at this redshift
            T21_slice         = np.asarray(
                lc.brightness_temp[:, :, idx_closest], dtype=np.float64)
            T21_slice_centered = T21_slice - np.mean(T21_slice)
            fft_T21_shifted    = np.fft.fftshift(np.fft.fft2(T21_slice_centered))

            # 2D cross- and auto-power spectra
            cross_ps2d    = (np.real(np.conj(fft_kSZ2_shifted) * fft_T21_shifted)
                             * pix_area / npix_side**2)
            auto_T21_ps2d = (np.abs(fft_T21_shifted)**2
                             * pix_area / npix_side**2)

            # ── Bin in 2D k-space ─────────────────────────────────────────────
            nb = len(k_centers)
            C_cross_1d            = np.full(nb, np.nan)
            C_cross_1d_err_sample = np.full(nb, np.nan)
            C_cross_1d_err_cosmic = np.full(nb, np.nan)
            C_cross_1d_err_total  = np.full(nb, np.nan)
            P_kSZ2_1d             = np.full(nb, np.nan)
            P_T21_1d              = np.full(nb, np.nan)
            n_modes               = np.zeros(nb)

            for j in range(nb):
                mask  = (kgrid >= k_bins[j]) & (kgrid < k_bins[j + 1])
                n_pix = int(np.sum(mask))

                if n_pix == 0:
                    continue

                cross_vals     = cross_ps2d[mask]
                C_cross_1d[j]            = np.mean(cross_vals)
                C_cross_1d_err_sample[j] = np.std(cross_vals) / np.sqrt(n_pix)
                P_kSZ2_1d[j]             = np.mean(auto_kSZ2_ps2d[mask])
                P_T21_1d[j]              = np.mean(auto_T21_ps2d[mask])

                # ── FIX 10: 2D mode count ─────────────────────────────────────
                dk_j       = k_bins[j + 1] - k_bins[j]
                n_modes[j] = 2.0 * np.pi * k_centers[j] * dk_j * L_over_2pi_sq

                if n_modes[j] > 0:
                    # Cosmic-variance error in 2D Gaussian approximation
                    C_cross_1d_err_cosmic[j] = (
                        np.sqrt(P_kSZ2_1d[j] * P_T21_1d[j] + C_cross_1d[j]**2)
                        / np.sqrt(n_modes[j])
                    )
                    C_cross_1d_err_total[j] = np.sqrt(
                        C_cross_1d_err_sample[j]**2
                        + C_cross_1d_err_cosmic[j]**2
                    )

            cross_corr_results[z_21cm] = {
                'k_centers'            : k_centers,
                'C_cross_1d'           : C_cross_1d,
                'C_cross_1d_err_sample': C_cross_1d_err_sample,
                'C_cross_1d_err_cosmic': C_cross_1d_err_cosmic,
                'C_cross_1d_err_total' : C_cross_1d_err_total,
                'n_modes'              : n_modes,
                'P_kSZ2_1d'            : P_kSZ2_1d,
                'P_T21_1d'             : P_T21_1d,
                'z_actual'             : z_actual,
                'idx_closest'          : idx_closest,
                'kSZ2_rms'             : float(np.sqrt(np.mean(kSZ2_map**2))),
                'T21_rms'              : float(np.sqrt(np.mean(T21_slice**2))),
                'T21_mean'             : float(np.mean(T21_slice)),
            }

        np.save(cc_cache, cross_corr_results)
        return (seed, cross_corr_results,
                f"computed ({len(cross_corr_results)} redshifts)")

    except Exception as e:
        return (seed, None, f"failed: {str(e)}")
