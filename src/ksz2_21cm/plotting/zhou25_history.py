# =============================================================================
# zhou25_history.py -- Zhou+25 (ApJ 991, 195) Fig. 1: global ionization history
# x_HII(z) of their three zreion scenarios, digitized EXACTLY from the vector
# paths of Fig. 1 in the PDF (axis ticks used for calibration; not eyeballed).
# Fiducial = zbar 8, alpha 0.2, k0 0.9 ; Late = zbar 7 ; Short = zbar 8, alpha 0.564, k0 0.185.
# Fig. 5 of that paper uses the FIDUCIAL history (its top axis agrees: x_HII=0.43 at z=8).
# The curves in the PDF are sampled every dz = 0.5 (z = 5..12) plus a final point at z = 12.29.
# columns: [z, x_HII]  (z ascending)
# =============================================================================
import numpy as np

ZHOU_HISTORY = {
    "Fiducial": np.array([
        [5.000, 0.9998],
        [5.500, 0.9957],
        [6.000, 0.9691],
        [6.500, 0.8900],
        [7.000, 0.7533],
        [7.500, 0.5886],
        [8.000, 0.4307],
        [8.500, 0.3002],
        [9.000, 0.2021],
        [9.500, 0.1328],
        [10.000, 0.0859],
        [10.500, 0.0550],
        [11.000, 0.0351],
        [11.500, 0.0223],
        [12.000, 0.0142],
        [12.285, 0.0112],
    ]),
    "Late": np.array([
        [5.000, 0.9752],
        [5.500, 0.8966],
        [6.000, 0.7537],
        [6.500, 0.5825],
        [7.000, 0.4224],
        [7.500, 0.2932],
        [8.000, 0.1978],
        [8.500, 0.1311],
        [9.000, 0.0861],
        [9.500, 0.0563],
        [10.000, 0.0367],
        [10.500, 0.0240],
        [11.000, 0.0157],
        [11.500, 0.0104],
        [12.000, 0.0069],
        [12.285, 0.0056],
    ]),
    "Short": np.array([
        [5.000, 1.0000],
        [6.000, 1.0000],
        [6.500, 0.9971],
        [7.000, 0.9408],
        [7.500, 0.7316],
        [8.000, 0.4442],
        [8.500, 0.2220],
        [9.000, 0.0979],
        [9.500, 0.0402],
        [10.000, 0.0160],
        [10.500, 0.0062],
        [11.000, 0.0024],
        [11.500, 0.0010],
        [12.000, 0.0004],
        [12.285, 0.0002],
    ]),
}


def zhou_xhii_of_z(z, scenario="Fiducial"):
    """Linear interpolation of Zhou+25 Fig. 1 history (z may be an array)."""
    a = ZHOU_HISTORY[scenario]
    return np.interp(z, a[:, 0], a[:, 1])
