"""
Reionization-history milestones, defined the same way as the kSZ patchy window
(optical_depth.compute_patchy_mask: x_e between 1e-4 and 1 - 1e-4):

    z_start = z(x_e = 1e-4)       99.99% neutral  (high-z edge of the patchy window)
    z10, z50, z90                 x_e = 0.1, 0.5, 0.9
    z_end   = z(x_e = 1 - 1e-4)   0.01% neutral   (low-z edge of the patchy window)

x_e is the volume-weighted mean ionized fraction. Crossings are interpolated linearly in z, in
log10(x_e) for z_start and in log10(1 - x_e) for z_end (both are tiny numbers changing by decades).
"""
import numpy as np

XE_START, XE_END = 1e-4, 1.0 - 1e-4


def z_at(z, xe, level, log_from=None):
    """Redshift where x_e(z) crosses `level` (NaN if it never does in the given range)."""
    z, xe = np.asarray(z, float), np.asarray(xe, float)
    o = np.argsort(z)
    z, xe = z[o], xe[o]
    if log_from == "low":
        y, lv = np.log10(np.clip(xe, 1e-30, None)), np.log10(level)
    elif log_from == "high":
        y, lv = np.log10(np.clip(1.0 - xe, 1e-30, None)), np.log10(1.0 - level)
    else:
        y, lv = xe, level
    s = np.sign(y - lv)
    for i in range(len(z) - 1):
        if s[i] == 0:
            return float(z[i])
        if s[i] != s[i + 1]:
            return float(z[i] + (lv - y[i]) * (z[i + 1] - z[i]) / (y[i + 1] - y[i]))
    return np.nan


def milestones(z, xe):
    """dict z_start, z10, z50, z90, z_end, dz_10_90 for one history."""
    m = {"z_start": z_at(z, xe, XE_START, "low"), "z10": z_at(z, xe, 0.1), "z50": z_at(z, xe, 0.5),
         "z90": z_at(z, xe, 0.9), "z_end": z_at(z, xe, XE_END, "high")}
    m["dz_10_90"] = m["z10"] - m["z90"]
    return m
