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

import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, Planck18

_COSMO_CACHE = {}
_C_KMS = 299792.458


class RunCosmology:
    """
    Background cosmology of ONE simulation run, with the run's own expansion history,
    E(z) = sqrt(orad (1+z)^4 + om (1+z)^3 + ol)  (AMBER's cosmo.f90 form; flat when om + ol + orad = 1).

    Why it exists: chi(z) sets the LOS pixel positions of a stitched lightcone and ell = k chi(z0) of every
    spectrum, so it must come from the cosmology the simulation was actually run with, not from whatever astropy
    preset or config value happens to be nearby (astropy Planck18 is ~2% off at z = 5 for a Chen+23 run).
    Only what the pipeline asks of an astropy cosmology is provided: comoving_distance(z) and H(z), returning
    astropy Quantities in Mpc and km/s/Mpc, so `cosmo.comoving_distance(z).value` works unchanged.
    """

    Z_MAX = 40.0

    def __init__(self, om, ol, orad, h, ob=None, label="run"):
        self.om, self.ol, self.orad, self.h = float(om), float(ol), float(orad), float(h)
        self.ob = None if ob is None else float(ob)
        self.label = label
        zt = np.linspace(0.0, self.Z_MAX, 80001)
        inv_e = 1.0 / self._efunc(zt)
        self._zt = zt
        self._chi = (_C_KMS / (100.0 * self.h)) * np.concatenate(
            [[0.0], np.cumsum(0.5 * (inv_e[1:] + inv_e[:-1]) * np.diff(zt))])

    def _efunc(self, z):
        z1 = 1.0 + np.asarray(z, float)
        return np.sqrt(self.orad * z1 ** 4 + self.om * z1 ** 3 + self.ol)

    def H(self, z):
        return 100.0 * self.h * self._efunc(z) * u.km / u.s / u.Mpc

    def comoving_distance(self, z):
        z = np.asarray(z, float)
        if np.any(z < 0) or np.any(z > self.Z_MAX):
            raise ValueError(f"RunCosmology.comoving_distance: z outside [0, {self.Z_MAX}]")
        return np.interp(z, self._zt, self._chi) * u.Mpc

    def params(self):
        return dict(om=self.om, ol=self.ol, ob=self.ob, orad=self.orad, h=self.h)

    @classmethod
    def from_params(cls, p, label="run"):
        return cls(p["om"], p["ol"], p["orad"], p["h"], p.get("ob"), label)

    @classmethod
    def from_run_dir(cls, run_dir):
        """From <run_dir>/amber_run.json (written by scripts/amber/make_amber_input.py): the AMBER parameters
        actually used, plus the n_e0 they imply."""
        import json
        import os
        path = os.path.join(run_dir, "amber_run.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: it holds the cosmology this AMBER run was made with")
        meta = json.load(open(path))
        c = cls.from_params(meta["amber_params"], label=f"amber:{meta.get('cosmology_preset', '?')}")
        c.ne0_cgs = float(meta["ne0_cgs"]) if "ne0_cgs" in meta else None
        c.preset = meta.get("cosmology_preset")
        return c

    def matches(self, other, rtol=1e-6):
        a, b = self.params(), other.params()
        return all(abs(a[k] - b[k]) <= rtol * max(abs(a[k]), 1e-30) for k in ("om", "ol", "orad", "h"))


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

    if use == "amber":
        # the run's own cosmology, given explicitly in the config (cosmology.amber: om, ol, orad, h [, ob]);
        # scripts/01a_stitch_lightcones_amber.py checks it against the run's amber_run.json
        p = dict(cfg["cosmology"]["amber"])
        key = ("amber",) + tuple(sorted(p.items()))
        if key not in _COSMO_CACHE:
            _COSMO_CACHE[key] = RunCosmology.from_params(p, label="config:amber")
        return _COSMO_CACHE[key]

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
