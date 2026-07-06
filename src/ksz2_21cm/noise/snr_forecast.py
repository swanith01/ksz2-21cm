# =============================================================================
# snr_forecast.py
#
# Reusable noise-model and SNR-estimator functions, extracted from
# SNR_rigorous_v2.py — this is the FINAL / most rigorous SNR script you had
# (frequency-dependent T_sys + full empirical covariance matrix). It builds
# on and supersedes both:
#   - the "CELL SNR (CORRECTED)" cell embedded at the end of the CLUSTER
#     script (noise-free, diagonal estimator only), and
#   - cell_snr_with_noise.py (adds a simple NSR noise degradation, still
#     diagonal, fixed T_sys).
# Both of those are kept for reference in notebooks/exploratory/ but should
# be treated as superseded — use this module for anything going in the paper.
#
# [Deliberate signature change during cleanup]
# The original script read `cosmo`, `f_sky`, and `T_CMB_uK` from enclosing
# script scope (they were assigned once near the top of SNR_rigorous_v2.py
# and closed over by every function below). That doesn't work as an
# importable module, so here they are explicit parameters / a module
# constant instead:
#   - T_CMB_uK is a physical constant -> module-level constant here.
#   - cosmo and f_sky are experiment/run choices -> explicit parameters.
# The numerics inside each function are otherwise unchanged.
# =============================================================================

import numpy as np

T_CMB_uK = 2.725e6  # muK


def T_sys_zhou25(nu_Hz):
    """Zhou+25 frequency-dependent system temperature [K]."""
    nu_MHz = nu_Hz / 1e6
    return 237.0 + 1.6 * (nu_MHz / 300.0) ** (-5.23)


def cmb_noise_map(npix, pix_size_rad, cmb_exp, seed_rng):
    """
    Gaussian CMB noise realization in dimensionless (Delta_T/T_CMB) units.
    N_ell = (sigma_rad)^2 * exp(ell(ell+1) * sigma_beam^2)

    Parameters
    ----------
    npix : int
    pix_size_rad : float
    cmb_exp : dict with 'sigma_uK_arcmin', 'fwhm_arcmin', 'label'
    seed_rng : np.random.Generator
    """
    arcmin2rad = np.pi / 180.0 / 60.0
    sigma_rad  = cmb_exp['sigma_uK_arcmin'] * arcmin2rad / T_CMB_uK
    sigma_beam = (cmb_exp['fwhm_arcmin'] * arcmin2rad
                  / np.sqrt(8.0 * np.log(2.0)))

    dk     = 2.0 * np.pi / (npix * pix_size_rad)
    kx     = np.fft.fftshift(np.fft.fftfreq(npix)) * npix * dk
    ky     = np.fft.fftshift(np.fft.fftfreq(npix)) * npix * dk
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    ell_2d = np.sqrt(KX**2 + KY**2)

    N_ell_2d = sigma_rad**2 * np.exp(ell_2d * (ell_2d + 1.0) * sigma_beam**2)

    pix_area_rad = pix_size_rad**2
    noise_amp    = np.sqrt(N_ell_2d * npix**2 / pix_area_rad)

    noise_fft = (seed_rng.normal(0, noise_amp / np.sqrt(2))
                 + 1j * seed_rng.normal(0, noise_amp / np.sqrt(2)))
    noise_fft = np.fft.ifftshift(noise_fft)
    noise_map = np.real(np.fft.ifft2(noise_fft))
    return noise_map


def thermal_noise_cube(shape, pix_size_Mpc, z0, ts_exp, seed_rng, cosmo,
                       delta_nu_MHz=28.4):
    """
    3D thermal noise realization for a 21cm chunk [mK], with
    frequency-dependent T_sys (Zhou+25).

    P_noise = T_sys(nu)^2 * Omega_beam * Y * X^2 / (N_bl * t_obs_s * B_Hz)

    Parameters
    ----------
    shape : tuple (nx, ny, nz)
    pix_size_Mpc : float
    z0 : float
    ts_exp : dict with 't_obs_hr', 'N_bl', 'A_eff_m2', 'label'
    seed_rng : np.random.Generator
    cosmo : astropy cosmology (see utils.cosmology.get_cosmology)
    delta_nu_MHz : float

    Returns
    -------
    (noise_cube, sigma_mK, T_sys_K)
    """
    nx, ny, nz = shape
    c_m_s  = 2.998e8
    nu_21  = 1420.4e6            # Hz
    nu_obs = nu_21 / (1.0 + z0)  # Hz (chunk centre frequency)

    T_sys_K = T_sys_zhou25(nu_obs)

    lam_m      = c_m_s / nu_obs
    D_ant      = np.sqrt(ts_exp['A_eff_m2'])
    Omega_beam = (lam_m / D_ant) ** 2  # sr

    H_z           = float(cosmo.H(z0).value)                  # km/s/Mpc
    D_M           = float(cosmo.comoving_distance(z0).value)  # Mpc
    Y_Mpc_per_MHz = 2.998e5 * (1.0 + z0) ** 2 / (nu_21 * 1e-6 * H_z)

    t_obs_s     = ts_exp['t_obs_hr'] * 3600.0
    delta_nu_Hz = delta_nu_MHz * 1e6

    P_noise_K2_Mpc3 = (T_sys_K**2 * Omega_beam
                       * Y_Mpc_per_MHz / delta_nu_MHz
                       * D_M**2
                       / (ts_exp['N_bl'] * t_obs_s * delta_nu_Hz * 1e-6))
    P_noise_mK2_Mpc3 = P_noise_K2_Mpc3 * 1e6  # K^2 -> mK^2

    vox_vol_Mpc3 = pix_size_Mpc**3
    sigma_mK     = np.sqrt(P_noise_mK2_Mpc3 / vox_vol_Mpc3)

    noise_cube = seed_rng.normal(0.0, sigma_mK, size=shape)
    return noise_cube, float(sigma_mK), float(T_sys_K)


def snr_diagonal(results_all, z_keys, f_sky, ell_min=100, ell_max=5000):
    """
    Standard diagonal Gaussian correlation-coefficient SNR estimator.
    (S/N)^2_ell = f_sky * (2*ell+1) * d_ell * r^2 / (1 - r^2)

    Parameters
    ----------
    results_all : dict {seed: {z0: result_dict}}
        result_dict must have 'ell', 'C_cross', 'P_kSZ2', 'P_T21sq'.
    z_keys : list of z0 values to use
    f_sky : float — sky-overlap fraction of the two surveys

    Returns
    -------
    per_z : dict {z0: {'ell', 'r', 'snr_cumul', 'snr_total'}}
    snr_total : float, combined SNR across all z0 (added in quadrature)
    """
    per_z      = {}
    snr_sq_tot = 0.0

    for z0 in z_keys:
        sigs, ksz2s, t21s, ell_ref = [], [], [], None
        for seed, ccr in results_all.items():
            if z0 not in ccr:
                continue
            res = ccr[z0]
            if ell_ref is None:
                ell_ref = res['ell']
            v = res['C_cross']
            if np.all(~np.isfinite(v)):
                continue
            sigs.append(v)
            ksz2s.append(res['P_kSZ2'])
            t21s.append(res['P_T21sq'])

        if ell_ref is None or len(sigs) == 0:
            continue

        C_sig  = np.nanmean(np.array(sigs),  axis=0)
        C_kSZ2 = np.nanmean(np.array(ksz2s), axis=0)
        C_21cm = np.nanmean(np.array(t21s),  axis=0)
        ell    = ell_ref

        denom_r = np.sqrt(np.abs(C_kSZ2) * np.abs(C_21cm))
        r       = np.where(denom_r > 0, C_sig / denom_r, 0.0)
        r       = np.clip(r, -1.0 + 1e-10, 1.0 - 1e-10)

        valid = (np.isfinite(r) & np.isfinite(ell)
                 & (ell >= ell_min) & (ell <= ell_max))

        snr_sq_bin = np.zeros_like(ell)
        if np.any(valid):
            dell = np.gradient(ell)
            r2   = r[valid]**2
            snr_sq_bin[valid] = np.maximum(
                f_sky * (2.0 * ell[valid] + 1.0) * dell[valid]
                * r2 / (1.0 - r2), 0.0)

        snr_cumul  = np.sqrt(np.cumsum(snr_sq_bin))
        snr_z0     = float(snr_cumul[-1]) if len(snr_cumul) else 0.0
        snr_sq_tot += snr_z0**2

        per_z[z0] = dict(ell=ell, r=r,
                         snr_cumul=snr_cumul, snr_total=snr_z0)

    return per_z, np.sqrt(snr_sq_tot)


def snr_full_cov(results_all, z_keys, f_sky,
                 ell_min=100, ell_max=5000, rcond=1e-3):
    """
    Full empirical covariance matrix SNR:
        (S/N)^2 = s^T @ Cov^{-1} @ s
    where s is the seed-mean C_ell^{kSZ2 x 21cm2} signal vector and Cov is
    the empirical covariance from seed-to-seed scatter, mode-count weighted
    and pseudo-inverted (SVD, rcond cut) to handle N_seeds < N_ell_bins.

    Parameters
    ----------
    results_all : dict {seed: {z0: result_dict}}
    z_keys      : list of z0 values to use
    f_sky       : float — sky-overlap fraction of the two surveys
    ell_min/max : ell range to include
    rcond       : relative condition threshold for the pseudo-inverse

    Returns
    -------
    per_z : dict {z0: {'ell', 'C_signal', 'snr_total', 'eigenvalues',
                        'condition_number', 'n_modes_used', 'N_seeds'}}
    snr_total : float, combined SNR across all z0 (added in quadrature)
    """
    per_z      = {}
    snr_sq_tot = 0.0

    for z0 in z_keys:
        seed_vecs = []
        ell_ref   = None

        for seed, ccr in results_all.items():
            if z0 not in ccr:
                continue
            res = ccr[z0]
            if ell_ref is None:
                ell_ref = res['ell']
            v = res['C_cross'].copy()
            if np.all(~np.isfinite(v)):
                continue
            seed_vecs.append(v)

        if ell_ref is None or len(seed_vecs) < 4:
            print(f"  z0={z0:.1f}: only {len(seed_vecs)} seeds — skipping")
            continue

        N_seeds = len(seed_vecs)
        X       = np.array(seed_vecs)  # (N_seeds, N_ell)
        ell     = ell_ref

        valid = np.isfinite(ell) & (ell >= ell_min) & (ell <= ell_max)
        if np.sum(valid) < 2:
            continue

        X_v   = X[:, valid]
        ell_v = ell[valid]
        dell  = np.gradient(ell_v)

        s = np.mean(X_v, axis=0)

        dX      = X_v - s[None, :]
        Cov_raw = (dX.T @ dX) / (N_seeds - 1)

        n_modes = f_sky * (2.0 * ell_v + 1.0) * dell
        w       = np.sqrt(np.abs(n_modes))
        w       = np.where(w > 0, w, 1.0)
        W       = np.outer(w, w)
        Cov_eff = Cov_raw / W

        if not np.all(np.isfinite(Cov_eff)):
            n_bad = int(np.sum(~np.isfinite(Cov_eff)))
            print(f"  z0={z0:.1f}: {n_bad} non-finite entries in Cov_eff — zeroing")
            Cov_eff = np.where(np.isfinite(Cov_eff), Cov_eff, 0.0)

        s_clean = np.where(np.isfinite(s), s, 0.0)

        try:
            from scipy.linalg import svd as scipy_svd
            U, sv, Vt = scipy_svd(Cov_eff, full_matrices=False,
                                   lapack_driver='gesdd')
        except Exception:
            U, sv, Vt = np.linalg.svd(Cov_eff, full_matrices=False)
        sv_max       = sv[0]
        sv_cut       = rcond * sv_max
        keep         = sv > sv_cut
        n_modes_used = int(np.sum(keep))

        sv_inv       = np.where(keep, 1.0 / sv, 0.0)
        Cov_eff_pinv = (Vt.T * sv_inv) @ U.T

        s_scaled = s_clean / w

        snr_sq_z0 = float(s_scaled @ Cov_eff_pinv @ s_scaled)
        snr_sq_z0 = max(snr_sq_z0, 0.0)
        snr_z0    = np.sqrt(snr_sq_z0)
        snr_sq_tot += snr_sq_z0

        cond = sv[0] / sv[keep][-1] if n_modes_used > 0 else np.inf

        per_z[z0] = dict(
            ell=ell_v, C_signal=s, snr_total=snr_z0,
            eigenvalues=sv, sv_threshold=sv_cut,
            n_modes_used=n_modes_used, condition_number=float(cond),
            N_seeds=N_seeds,
        )
        print(f"  z0={z0:.1f}  SNR={snr_z0:.2f}\u03c3  "
              f"modes_used={n_modes_used}/{np.sum(valid)}  cond={cond:.1e}")

    return per_z, np.sqrt(snr_sq_tot)
