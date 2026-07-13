# =============================================================================
# cosmology.py — single source of truth for the cosmology used across the
# kSZ^2 x 21cm pipeline.
#
# CONFIRMED 2026-07-13 from a real run_config output on the cluster:
# CosmoParams() was printed with no override in the actual production run
# that made the cached lightcones — meaning py21cmfast's own defaults were
# used: hlittle (h) = 0.6766 (H0 = 67.66), OMm = 0.30964144154550644.
#
# This is DIFFERENT from what several old cells (overlay_zhou25.py,
# SNR_rigorous_v2.py, the embedded SNR cell) hardcoded:
#     cosmo = FlatLambdaCDM(H0=67.77, Om0=0.3086)
# — presumably hand-typed approximations of the true defaults, not read
# directly from CosmoParams(). Also NOT astropy's Planck18 preset
# (H0=67.66, Om0=0.3111) — Om0 differs from both.
#
# The values below are now the CONFIRMED real ones. All of your existing
# cached cross_corr_sq results were computed with THIS cosmology (it sets
# chi(z), hence ell = k*chi) — if you find evidence a specific cache file
# used something else, don't assume this retroactively applies to it
# without checking.
#
# Do not change get_cosmology() to Planck18 without re-deriving/re-caching
# every ell array downstream — that decision should be deliberate, not a
# side effect of a repo cleanup. See the `cosmology.use` field in
# configs/fiducial.yaml.
# =============================================================================

from astropy.cosmology import FlatLambdaCDM, Planck18

_COSMO_CACHE = {}


def get_cosmology(cfg=None):
    """
    Return the astropy cosmology object to use everywhere in this pipeline.

    Parameters
    ----------
    cfg : dict or None
        Config dict from load_config(). If None, defaults to the pipeline's
        CONFIRMED real cosmology (H0=67.66, Om0=0.30964144154550644) —
        py21cmfast's own CosmoParams() defaults, unmodified, as used in the
        actual cluster run.
    """
    use = "custom"
    H0, Om0 = 67.66, 0.30964144154550644
    if cfg is not None:
        cosmo_cfg = cfg.get("cosmology", {})
        use = cosmo_cfg.get("use", "custom")
        H0 = cosmo_cfg.get("H0", H0)
        Om0 = cosmo_cfg.get("Om0", Om0)

    key = (use, H0, Om0)
    if key not in _COSMO_CACHE:
        if use == "planck18":
            _COSMO_CACHE[key] = Planck18
        else:
            _COSMO_CACHE[key] = FlatLambdaCDM(H0=H0, Om0=Om0)
    return _COSMO_CACHE[key]


def chi_over_h(z, cfg=None, h=0.6766):
    """Comoving distance / h at redshift z, in Mpc/h. Used to convert k -> ell."""
    cosmo = get_cosmology(cfg)
    return float(cosmo.comoving_distance(z).value) / h


def z_to_xHII(z_query, z_nodes_ascending, xHII_nodes_ascending):
    """Interpolate the ionized fraction x_HII at arbitrary redshift(s)."""
    import numpy as np
    return np.interp(z_query, z_nodes_ascending, xHII_nodes_ascending)


def xHII_to_z(xHII_query, z_nodes_ascending, xHII_nodes_ascending):
    """Inverse of z_to_xHII — used for the top x_HII axis on redshift plots."""
    import numpy as np
    return np.interp(xHII_query, xHII_nodes_ascending, z_nodes_ascending)
