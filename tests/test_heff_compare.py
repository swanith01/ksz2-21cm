"""
Smoke + logic test for scripts/22_heff_history_compare.py on SYNTHETIC history folders.

Three histories (z50 = 6.7, 7.75, 8.75), 6 seeds each, a Zhou-like statistic whose hump sits at a FIXED ionized
fraction (x_HII = 0.3) plus seed noise. The script must (i) write the z0, shifted and x_HII figures, (ii) report the
peak location in x_HII as the same for all three histories (within the grid spacing) while the peak in z0 moves with
z50, (iii) give a finite bootstrap uncertainty.

Run:  python -m pytest tests/test_heff_compare.py -q
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from ksz2_21cm.utils.constants import T_CMB_K          # noqa: E402
from ksz2_21cm.utils.cosmology import get_cosmology    # noqa: E402


def _make_root(d, heffs=(17, 30, 52), z50s=(6.7, 7.75, 8.75), nseed=6):
    rng = np.random.default_rng(0)
    cosmo = get_cosmology()
    unit = (T_CMB_K * 1e6) ** 2 * 1e6
    z = np.arange(3.0, 20.0, 0.05)
    ell = np.geomspace(50, 6000, 40)
    z0s = [round(4.5 + 0.5 * i, 2) for i in range(18)]
    for H, z50 in zip(heffs, z50s):
        xe = 1.0 / (1.0 + np.exp((z - z50) / 0.7))
        for s in range(1, nseed + 1):
            sd = os.path.join(d, f"heff_{H}", "cache", f"seed_{s}")
            os.makedirs(sd)
            np.savez(os.path.join(sd, f"xe_history_seed{s}.npz"), z=z, x_e=xe)
            res = {}
            for z0 in z0s:
                x0 = float(np.interp(z0, z, xe))
                D = 5e6 * np.exp(-0.5 * ((x0 - 0.3) / 0.12) ** 2) * (1 + 0.05 * rng.standard_normal())
                chi = float(cosmo.comoving_distance(z0).value)
                C = D * 2 * np.pi * chi ** 2 / (ell * (ell + 1) * unit)
                res[z0] = {"ell": ell, "C_cross": C}
            np.save(os.path.join(sd, f"cross_corr_sq_stitched_seed{s}.npy"), res, allow_pickle=True)


def test_peak_follows_xhii():
    with tempfile.TemporaryDirectory() as d:
        _make_root(d)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/22_heff_history_compare.py"), "--data-root", d,
                            "--bootstrap", "50"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        out = os.path.join(d, "heff_history_compare")
        for stem in ("heff_zhou", "heff_zhou_shifted", "heff_zhou_xe"):
            assert os.path.exists(os.path.join(out, stem + ".png")), stem
        blk = r.stdout.split("PEAK LOCATION")[1].split("zhou ell=400:")[1].split("zhou ell=800:")[0]
        rows = re.findall(r"z0 = +([\d.]+) \+- ([\d.nan]+) +z0-z50 = +([+-][\d.]+) \+- ([\d.nan]+) +x_HII = +([\d.]+) \+- ([\d.nan]+)", blk)
        assert len(rows) == 3, blk
        z0p = np.array([float(x[0]) for x in rows])
        xp = np.array([float(x[4]) for x in rows])
        assert z0p.max() - z0p.min() > 1.5, z0p            # the peak in z0 follows the history
        assert xp.max() - xp.min() < 0.15, xp              # ... and sits at the same x_HII (one grid step)
        assert all(np.isfinite(float(x[5])) for x in rows)


if __name__ == "__main__":
    test_peak_follows_xhii()
    print("ok  test_peak_follows_xhii")
