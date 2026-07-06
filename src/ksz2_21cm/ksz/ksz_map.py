# =============================================================================
# ksz_map.py
#
# [NEW — did not exist before the repo cleanup.]
#
# Pure computational functions extracted from CELLS 4, 5, 6 of
# kSZ_Squared_21cm_11Jun_CLUSTER.py:
#   CELL 4 (part B) -> compute_optical_depth()
#   CELL 5          -> compute_ksz_integrand()
#   CELL 6          -> compute_ksz_map()
#
# The original cells interleaved this physics with per-seed diagnostic
# plotting (reionization history, tau(z), integrand slices). That plotting
# has been left where it belongs — in scripts/02_compute_ksz_maps.py — so
# this module has no matplotlib import and can be unit-tested on a single
# lightcone without touching the filesystem or plotting.
#
# Numerics are unchanged from the original cells; this is an extraction,
# not a rewrite. Constants (sigma_T, n_e0, etc.) are copied verbatim.
# =============================================================================

import numpy as np


def compute_optical_depth(lc, z_max):
    """
    Compute the cumulative Thomson optical depth tau(<z) along one lightcone.

    Extracted from CELL 4, Part B ("Optical depth tau(<z)").

    Parameters
    ----------
    lc : py21cmfast.LightCone
    z_max : float

    Returns
    -------
    dict with keys: red_axis, z_mid, x_e_mid, ds_Mpc, tau, tau_total
    """
    # Physical constants (unchanged from CELL 4)
    h              = 0.6766
    Omega_b        = 0.04897468161869667
    rho_crit_p_cm3 = 1.88e-29 * h**2 / (1.67e-24)
    n_H0_cm3       = Omega_b * rho_crit_p_cm3
    sigma_T_cm2    = 6.65e-25
    cm_per_Mpc     = 3.086e24
    n_e0_Mpc3      = n_H0_cm3 * cm_per_Mpc**3
    sigma_T_Mpc2   = sigma_T_cm2 / cm_per_Mpc**2
    prefactor      = n_e0_Mpc3 * sigma_T_Mpc2

    red_axis = lc.lightcone_redshifts
    pos_axis = lc.lightcone_distances

    ind_z    = np.where(red_axis <= z_max)[0]
    red_axis = red_axis[ind_z]
    pos_axis = pos_axis[ind_z]

    z_nodes_sorted   = lc.node_redshifts[::-1]
    xHI_nodes_sorted = lc.global_xH[::-1]
    x_e_nodes_sorted = 1.0 - xHI_nodes_sorted

    x_e_interp = np.interp(red_axis, z_nodes_sorted, x_e_nodes_sorted)

    ds_Mpc  = np.asarray(np.diff(pos_axis), dtype=np.float64)
    z_mid   = 0.5 * (red_axis[:-1] + red_axis[1:])
    x_e_mid = 0.5 * (x_e_interp[:-1] + x_e_interp[1:])

    dtau      = prefactor * x_e_mid * (1.0 + z_mid)**2 * ds_Mpc
    tau       = np.cumsum(dtau)
    tau_total = float(tau[-1])

    return {
        'red_axis' : red_axis,
        'z_mid'    : z_mid,
        'x_e_mid'  : x_e_mid,
        'ds_Mpc'   : ds_Mpc,
        'tau'      : tau,
        'tau_total': tau_total,
    }


def compute_ksz_integrand(lc, tau_result, z_max):
    """
    Compute the kSZ integrand:  (1 + delta) * x_e * v_z / c * e^(-tau(z))

    Extracted from CELL 5.

    Parameters
    ----------
    lc : py21cmfast.LightCone
    tau_result : dict
        Output of compute_optical_depth(lc, z_max).
    z_max : float

    Returns
    -------
    np.ndarray, shape (npix, npix, n_los) — the kSZ integrand cube.
    """
    c_Mpc_s = 299792.458 / 3.08567758e19

    red_axis_full = np.asarray(lc.lightcone_redshifts)
    ind_z         = np.where(red_axis_full <= z_max)[0]

    density_1plus = 1 + np.asarray(lc.density[:, :, ind_z])
    x_e_3D        = 1 - np.asarray(lc.xH_box[:, :, ind_z])
    v_los_Mpc_s   = np.asarray(lc.velocity[:, :, ind_z]) / 67.4  # CHECK: hardcoded H0

    red_axis_array = np.asarray(tau_result['red_axis'], dtype=np.float64)
    tau_array      = np.asarray(tau_result['tau'],      dtype=np.float64)
    z_mid_array    = np.asarray(tau_result['z_mid'],    dtype=np.float64)

    tau_extended = np.concatenate([[0.0], tau_array])
    z_extended   = np.concatenate([[red_axis_array[0]], z_mid_array])
    tau_at_lc    = np.asarray(
        np.interp(red_axis_array, z_extended, tau_extended),
        dtype=np.float64,
    )

    visibility    = np.exp(-tau_at_lc)
    visibility_3D = visibility[None, None, :]

    kSZ_integrand = (density_1plus * x_e_3D
                     * v_los_Mpc_s / c_Mpc_s
                     * visibility_3D)
    return kSZ_integrand


def compute_ksz_map(kSZ_integrand, tau_result, z_obs):
    """
    Line-of-sight integrate the kSZ integrand into a 2D map, observed at z_obs.

        kSZ(z_obs) = integral_{z_start}^{z_obs} [n_e0 sigma_T (1/a^2)
                                                  (1+delta) x_e v_z/c e^-tau ds]

    Extracted from CELL 6.

    Parameters
    ----------
    kSZ_integrand : np.ndarray
        Output of compute_ksz_integrand(), shape (npix, npix, n_los).
    tau_result : dict
        Output of compute_optical_depth() for the SAME lightcone.
    z_obs : float
        Observation redshift (integration stops here).

    Returns
    -------
    np.ndarray, shape (npix, npix) — the 2D kSZ map.
    """
    c_cm_s      = 3.0e10
    sigma_T_cm2 = 6.6525e-25
    n_e0_cm3    = 2.06e-7
    Mpc_to_cm   = 3.0857e24
    prefactor_cgs = n_e0_cm3 * sigma_T_cm2 * c_cm_s

    red_axis = np.asarray(tau_result['red_axis'], dtype=np.float64)
    ds_Mpc   = np.asarray(tau_result['ds_Mpc'],   dtype=np.float64)
    z_mid    = np.asarray(tau_result['z_mid'],    dtype=np.float64)
    ds_cm    = ds_Mpc * Mpc_to_cm

    a                = 1.0 / (1.0 + red_axis)
    a_squared        = a**2
    a_squared_mid    = 0.5 * (a_squared[:-1] + a_squared[1:])
    a_squared_mid_3D = a_squared_mid[None, None, :]

    kSZ_int_mid  = 0.5 * (kSZ_integrand[:, :, :-1] + kSZ_integrand[:, :, 1:])
    kSZ_int_full = ((prefactor_cgs / a_squared_mid_3D)
                    * kSZ_int_mid
                    * (ds_cm / c_cm_s)[None, None, :])

    idx_integrate = np.where(z_mid >= z_obs)[0]
    kSZ_map = np.sum(kSZ_int_full[:, :, idx_integrate], axis=2)
    return kSZ_map
