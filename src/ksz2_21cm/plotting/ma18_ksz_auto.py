"""Digitized Ma+2018 (MNRAS 476, 4025) Fig. 2 TOP panel: kSZ auto power spectrum
Delta^2_kSZ(l) = l(l+1)C_l/2pi [uK^2] from their 2-D maps (red, mean of 60 realisations;
error bars = error on the mean). Extracted from the vector paths of the PDF
(errorbar stems; same method as ma18_data.py). Columns [ell, D_ell, err].
Ma note: their ell < ~1e3 values are inflated by boundary effects of the lightcone.
"""
import numpy as np

MA18_KSZ_AUTO = np.array([
    [110.83, 7.6863, 0.3944],
    [119.38, 7.5916, 0.4431],
    [147.65, 5.7081, 0.1796],
    [179.98, 3.8658, 0.1359],
    [213.86, 2.7857, 0.1105],
    [257.98, 2.1062, 0.0863],
    [310.5, 1.6516, 0.0720],
    [371.15, 1.4538, 0.0710],
    [442.51, 1.4069, 0.0631],
    [528.74, 1.4712, 0.0643],
    [637.16, 1.6536, 0.0646],
    [767.81, 1.8929, 0.0645],
    [921.73, 2.0048, 0.0619],
    [1104.6, 2.1572, 0.0629],
    [1326.2, 2.2667, 0.0606],
    [1591.4, 2.3667, 0.0584],
    [1908.5, 2.3887, 0.0600],
    [2290.5, 2.3516, 0.0578],
    [2747.8, 2.2597, 0.0574],
    [3296.8, 2.1169, 0.0552],
    [3956.8, 1.9509, 0.0532],
    [4749.2, 1.7880, 0.0527],
    [5699.2, 1.7246, 0.0516],
    [6709.7, 1.6614, 0.0510],
])
