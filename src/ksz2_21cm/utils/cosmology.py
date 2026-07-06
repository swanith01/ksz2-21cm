# =============================================================================
# cosmology.py — single source of truth for the cosmology used across the
# kSZ^2 x 21cm pipeline.
#
# WARNING — read before editing:
# Every script you had (CLUSTER script, overlay_zhou25.py, SNR_rigorous_v2.py,
# the embedded SNR cell) independently redeclared:
#     cosmo = FlatLambdaCDM(H0=67.77, Om0=0.3086)
# This is NOT astropy's Planck18 preset (H0=67.66, Om0=0.3111) — it's a custom
# cosmology, presumably matched to 21cmFAST's CosmoParams() defaults used when
# your lightcones were generated. All of your existing cached cross_corr_sq
# results were computed with THIS cosmology (it sets chi(z), hence ell = k*chi).
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
        existing custom cosmology (H0=67.77, Om0=0.3086) for backward
        compatibility with already-cached results.
    """
    use = "custom"
    H0, Om0 = 67.77, 0.3086
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


def chi_over_h(z, cfg=None, h=0.6777):
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
