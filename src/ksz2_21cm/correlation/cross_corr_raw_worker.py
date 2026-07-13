# =============================================================================
# cross_corr_raw_worker.py
#
# [NEW — did not exist in any form in the original code. Built in response
#  to Prof. Kulkarni's specific physics question, not extracted from
#  anything.]
#
# Parallel worker: RAW kSZ x 21cm cross-correlation, one seed at a time —
# i.e. neither field is squared. This is a genuinely different statistic
# from both of the other two in this directory:
#
#   cross_corr_worker.py     : kSZ^2 x 21cm    (kSZ squared, 21cm raw)
#   cross_corr_sq_worker.py  : kSZ^2 x 21cm^2  (both squared, Zhou+25 style)
#   cross_corr_raw_worker.py : kSZ   x 21cm    (THIS FILE — neither squared)
#
# Motivation (Prof. Kulkarni, relayed 2026-07): both existing statistics
# square the kSZ field before cross-correlating. His argument: at k != 0,
# ANY mean-subtracted map is a signed field — including 21cm — so there is
# no a priori reason raw kSZ x 21cm should vanish just because it hasn't
# been squared. The literature's skepticism about kSZ-only correlations may
# be specific to the two-point kSZ-kSZ case, not obviously generalizable to
# kSZ x (a density-sourced field like 21cm), which has real physical terms
# like P_delta-v, P_delta-delta, P_e-delta that could contribute here.
# He separately noted that if a kSZ^2 x field statistic is wanted on
# principle, the honest way to get it is the actual bispectrum (which
# kSZ^2 x 21cm approximates only in the unphysical k1->infinity limit),
# not a cross-power dressed up to look like one — but that's a separate,
# not-yet-built piece of work; this file is just the raw two-point cross-
# spectrum he asked for directly.
#
# Structurally mirrors cross_corr_worker.py (same FIX 9 cache-before-load,
# same FIX 10 2D mode-count formula — both already validated on desktop),
# with two substantive differences beyond dropping the squaring:
#   1. ell and D_cross (physical multipole-space quantities) are computed
#      HERE, inside the worker, rather than left to a downstream plotting
#      cell — see notebooks/exploratory/wedge_diagnostic_kSZ2x21cm_unsquared.py
#      for why leaving that conversion to a separate cell led to a
#      three-way inconsistent convention across the old codebase. This
#      file uses ell = k_centers * chi(z) with NO h factor, matching
#      cross_corr_sq_worker.py (the main pipeline result) — see that
#      file's comments for the reasoning.
#   2. The physical-unit conversion here is muK * mK (ONE power of
#      T_CMB_uK, not squared) since kSZ now enters to the first power, not
#      squared. Do not reuse the T_CMB_uK^2 factor from the other two
#      worker modules here — it would be wrong for this statistic.
#
# UNTESTED: no py21cmfast access available to actually run this. Mechanical
# extension of an already-tested pattern (same binning/error-formula
# structure as cross_corr_worker.py, which IS confirmed working on
# desktop), so risk is concentrated in the two intentional changes above,
# not the surrounding machinery. Verify against a real run before trusting
# any physical conclusion drawn from it.
#
# Imported by scripts/06_compute_cross_corr_raw.py.
# =============================================================================

import numpy as np
import os

T_CMB_uK = 2.725e6   # physical constant — one power only for this statistic


def compute_cross_corr_raw_for_seed(args):
    """
    Compute RAW kSZ x 21cm cross-correlation (neither field squared) for
    ONE seed, across all node redshifts of its lightcone.

    Parameters
    ----------
    args : tuple
        (seed, cache_file, kSZ_map, z_obs, main_cache_dir,
         npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
         dk, kgrid, k_bins, k_centers, cosmo)

        cosmo : astropy cosmology object — see
                src/ksz2_21cm/utils/cosmology.py::get_cosmology().
                Passed explicitly rather than redeclared here, so this
                worker always agrees with whatever cosmology the rest of
                the pipeline is using.

    Returns
    -------
    (seed, result_dict_or_None, status_message_str)
    result_dict is keyed by z_21cm (float), matching cross_corr_worker.py's
    schema plus 'ell', 'D_cross_raw_units', 'D_cross_muK_mK', and 'r_cross'
    (correlation coefficient — useful to know if there's ANY correlation at
    all before worrying about physical units or SNR).
    """
    import py21cmfast as p21c

    (seed, cache_file, kSZ_map, z_obs, main_cache_dir,
     npix_side, box_size_Mpc, pix_size_Mpc, pix_area,
     dk, kgrid, k_bins, k_centers, cosmo) = args

    seed_cache_dir = os.path.join(main_cache_dir, f"seed_{seed}")
    os.makedirs(seed_cache_dir, exist_ok=True)
    cc_cache = os.path.join(seed_cache_dir, f"cross_corr_raw_seed{seed}.npy")

    # ── Check cache before loading the lightcone (same fix as the other
    #    two worker modules) ──────────────────────────────────────────────
    if os.path.exists(cc_cache):
        try:
            result = np.load(cc_cache, allow_pickle=True).item()
            return (seed, result, f"cached ({len(result)} redshifts)")
        except Exception:
            pass  # corrupt cache — fall through to recompute

    if kSZ_map is None:
        return (seed, None, "no kSZ map available")

    try:
        lc = p21c.LightCone.read(cache_file)
    except Exception as e:
        return (seed, None, f"failed to load lightcone: {e}")

    try:
        # ── RAW kSZ map — NOT squared. Computed once, reused for all
        #    redshift slices. ─────────────────────────────────────────────
        kSZ_map_centered = kSZ_map - np.mean(kSZ_map)
        fft_kSZ_shifted  = np.fft.fftshift(np.fft.fft2(kSZ_map_centered))
        auto_kSZ_ps2d    = np.abs(fft_kSZ_shifted)**2 * pix_area / npix_side**2

        # Same 2D mode-count formula as cross_corr_worker.py (FIX 10 there).
        L_over_2pi_sq = (box_size_Mpc / (2.0 * np.pi))**2

        node_redshifts = np.asarray(lc.node_redshifts[::-1], dtype=np.float64)
        lc_redshifts   = np.asarray(lc.lightcone_redshifts, dtype=np.float64)
        cross_corr_results = {}

        for z_21cm in node_redshifts:

            idx_closest = int(np.argmin(np.abs(lc_redshifts - z_21cm)))
            z_actual    = float(lc_redshifts[idx_closest])

            # RAW 21cm brightness temperature slice — NOT squared.
            T21_slice          = np.asarray(
                lc.brightness_temp[:, :, idx_closest], dtype=np.float64)
            T21_slice_centered = T21_slice - np.mean(T21_slice)
            fft_T21_shifted    = np.fft.fftshift(np.fft.fft2(T21_slice_centered))

            cross_ps2d   = (np.real(np.conj(fft_kSZ_shifted) * fft_T21_shifted)
                            * pix_area / npix_side**2)
            auto_T21_ps2d = (np.abs(fft_T21_shifted)**2
                             * pix_area / npix_side**2)

            nb = len(k_centers)
            C_cross_1d            = np.full(nb, np.nan)
            C_cross_1d_err_sample = np.full(nb, np.nan)
            C_cross_1d_err_cosmic = np.full(nb, np.nan)
            C_cross_1d_err_total  = np.full(nb, np.nan)
            P_kSZ_1d               = np.full(nb, np.nan)
            P_T21_1d                = np.full(nb, np.nan)
            n_modes                 = np.zeros(nb)

            for j in range(nb):
                mask  = (kgrid >= k_bins[j]) & (kgrid < k_bins[j + 1])
                n_pix = int(np.sum(mask))
                if n_pix == 0:
                    continue

                cross_vals               = cross_ps2d[mask]
                C_cross_1d[j]            = np.mean(cross_vals)
                C_cross_1d_err_sample[j] = np.std(cross_vals) / np.sqrt(n_pix)
                P_kSZ_1d[j]              = np.mean(auto_kSZ_ps2d[mask])
                P_T21_1d[j]              = np.mean(auto_T21_ps2d[mask])

                dk_j       = k_bins[j + 1] - k_bins[j]
                n_modes[j] = 2.0 * np.pi * k_centers[j] * dk_j * L_over_2pi_sq

                if n_modes[j] > 0:
                    C_cross_1d_err_cosmic[j] = (
                        np.sqrt(P_kSZ_1d[j] * P_T21_1d[j] + C_cross_1d[j]**2)
                        / np.sqrt(n_modes[j])
                    )
                    C_cross_1d_err_total[j] = np.sqrt(
                        C_cross_1d_err_sample[j]**2
                        + C_cross_1d_err_cosmic[j]**2
                    )

            # ── k -> ell, physical units (done HERE, not left to a
            #    downstream cell — see file header for why) ───────────────
            chi_z   = float(cosmo.comoving_distance(z_actual).value)
            ell     = k_centers * chi_z   # no h factor — see file header
            D_cross_raw_units = ell * (ell + 1) * C_cross_1d / (2 * np.pi)
            # ONE power of T_CMB_uK — kSZ enters to the first power here,
            # not squared. Result is in [muK * mK], NOT [muK^2 * mK].
            D_cross_muK_mK = D_cross_raw_units * T_CMB_uK

            with np.errstate(divide='ignore', invalid='ignore'):
                r_cross = C_cross_1d / np.sqrt(P_kSZ_1d * P_T21_1d)

            cross_corr_results[float(z_21cm)] = {
                'k_centers'             : k_centers,
                'ell'                   : ell,
                'C_cross_1d'            : C_cross_1d,
                'C_cross_1d_err_sample' : C_cross_1d_err_sample,
                'C_cross_1d_err_cosmic' : C_cross_1d_err_cosmic,
                'C_cross_1d_err_total'  : C_cross_1d_err_total,
                'D_cross_raw_units'     : D_cross_raw_units,
                'D_cross_muK_mK'        : D_cross_muK_mK,
                'r_cross'               : r_cross,
                'n_modes'               : n_modes,
                'P_kSZ_1d'              : P_kSZ_1d,
                'P_T21_1d'              : P_T21_1d,
                'z_actual'              : z_actual,
                'idx_closest'           : idx_closest,
                'kSZ_rms'               : float(np.sqrt(np.mean(kSZ_map**2))),
                'T21_rms'               : float(np.sqrt(np.mean(T21_slice**2))),
                'T21_mean'              : float(np.mean(T21_slice)),
            }

        np.save(cc_cache, cross_corr_results)
        return (seed, cross_corr_results,
                f"computed ({len(cross_corr_results)} redshifts)")

    except Exception as e:
        return (seed, None, f"failed: {str(e)}")
