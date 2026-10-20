"""
Tests for the AMBER -> stitched-lightcone path (simulate/amber_io.py, simulate/stitch_from_amber.py,
scripts/01a_stitch_lightcones_amber.py), on SYNTHETIC AMBER output written in the real byte layout
(raw little-endian float32, Fortran order). Run:  python tests/test_amber_stitch.py   (or pytest)

What is checked, and why:
  1. the readers return arr[i,j,k] == Fortran arr(i,j,k) (axis 2 is the LOS axis);
  2. a snapshot without a matching Tb box (or vice versa) is refused (the 21cm grid must equal the shell grid);
  3. output keys/shapes/dtypes equal the 21cmFAST stitched path's contract;
  4. neutral = z > zre (a past bug class here: xH inverted);
  5. SHARED ROTATION: with one wrap_cycle_seed, brightness_temp and density come from the same spatial positions
     (Tb = 30*(1+delta) holds exactly where the box is neutral, because stitching is linear), and stitching Tb with
     a different seed breaks that -- so the test would fail if the fields were sampled independently;
  6. tb_scale is applied only when asked for;
  7. the 01a script runs end-to-end, writes the npz and, with --delete-raw, removes only the field dumps.
"""
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from ksz2_21cm.simulate import amber_io as aio                                   # noqa: E402
from ksz2_21cm.simulate.stitch_from_amber import stitch_lightcone_from_amber     # noqa: E402
from ksz2_21cm.simulate.stitch_from_coeval import build_los_z_grid, stitch_field  # noqa: E402

N, L_H, H = 16, 100.0, 0.6766
SNAP_Z = [5.0 + 0.5 * i for i in range(7)]          # 5.0 ... 8.0
T0 = 30.0


def _smooth_field(rng, sigma=2.0):
    k = np.fft.fftfreq(N) * N
    kk2 = k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2
    f = np.fft.ifftn(np.fft.fftn(rng.normal(size=(N, N, N))) * np.exp(-0.5 * kk2 * (sigma / N * 2 * np.pi) ** 2)).real
    return f / f.std()


def make_run(root, rng=None, drop_tb=None):
    rng = rng or np.random.default_rng(0)
    for sub in ("cmb", "21cm", "reion"):
        os.makedirs(os.path.join(root, "output", sub), exist_ok=True)
    g = _smooth_field(rng)
    zre = (5.8 + 0.15 * _smooth_field(rng)).astype(np.float32)          # reionization between ~5.3 and ~6.3
    zre.ravel(order="F").tofile(os.path.join(root, "output", "reion", "zre_test.dat"))
    truth = {}
    for z in SNAP_Z:
        rho = np.exp(0.3 * g * 5.0 / (1 + z)); rho = (rho / rho.mean()).astype(np.float32)
        mom = np.stack([rho * _smooth_field(rng) * 50 for _ in range(3)]).astype(np.float32)   # (3,N,N,N)
        a = np.concatenate([[z], rho.ravel(order="F"), mom.ravel(order="F")]).astype("<f4")
        a.tofile(os.path.join(root, "output", "cmb", f"fields_z={z:05.2f}.dat"))
        tb = np.where(z > zre, T0 * rho, 0.0).astype("<f4")
        if drop_tb != z:
            tb.ravel(order="F").tofile(os.path.join(root, "output", "21cm", f"Tb_L={L_H:g}_N={N}_z={z:05.2f}.dat"))
        truth[z] = (rho, mom, tb)
    return zre, truth


def test_readers_axis_order():
    with tempfile.TemporaryDirectory() as d:
        zre, truth = make_run(d)
        z = SNAP_Z[2]
        zf, rho, mom = aio.read_fields(os.path.join(d, "output/cmb", f"fields_z={z:05.2f}.dat"), N)
        assert zf == z
        assert np.array_equal(rho, truth[z][0]) and np.array_equal(mom, truth[z][1])
        assert rho[1, 2, 3] == truth[z][0][1, 2, 3] and rho[3, 2, 1] == truth[z][0][3, 2, 1]
        tb = aio.read_tb(os.path.join(d, "output/21cm", f"Tb_L={L_H:g}_N={N}_z={z:05.2f}.dat"), N)
        assert np.array_equal(tb, truth[z][2])
        assert np.array_equal(aio.read_zre(aio.find_zre_file(os.path.join(d, "output/reion")), N), zre)
        assert aio.parse_tb_name("Tb_L=64_N=32_z=06.50.dat") == (64.0, 32, 6.5)


def test_snapshot_matching_refuses_mismatch():
    with tempfile.TemporaryDirectory() as d:
        make_run(d, drop_tb=SNAP_Z[3])
        try:
            aio.list_snapshots(d)
        except ValueError as e:
            assert "matching Tb" in str(e)
        else:
            raise AssertionError("missing Tb box was not refused")
    with tempfile.TemporaryDirectory() as d:
        make_run(d)
        np.zeros(N ** 3, "<f4").tofile(os.path.join(d, "output/21cm", f"Tb_L={L_H:g}_N={N}_z=09.50.dat"))
        try:
            aio.list_snapshots(d)
        except ValueError as e:
            assert "without a field snapshot" in str(e)
        else:
            raise AssertionError("extra Tb box was not refused")


def _lightcone(d, seed=3, **kw):
    cell = L_H / H / N
    z_arr = build_los_z_grid(min(SNAP_Z), max(SNAP_Z), cell)
    return z_arr, stitch_lightcone_from_amber(d, z_arr, H, N=N, wrap_cycle_seed=seed, **kw)


def test_contract_and_direction():
    with tempfile.TemporaryDirectory() as d:
        zre, _ = make_run(d)
        z_arr, lc = _lightcone(d)
        for key in ("density", "xH_box", "velocity_z", "brightness_temp"):
            assert lc[key].shape == (N, N, len(z_arr)) and lc[key].dtype == np.float32, key
            assert np.isfinite(lc[key]).all(), key
        assert lc["pos_axis"].shape == z_arr.shape and np.array_equal(lc["z_arr"], z_arr)
        assert abs(lc["density"].mean()) < 0.05                       # raw delta, mean ~ 0
        xe = 1.0 - lc["xH_box"].mean(axis=(0, 1))
        assert xe[0] > 0.9 and xe[-1] < 0.05, (xe[0], xe[-1])         # ionized at low z, neutral at high z
        assert (lc["brightness_temp"][:, :, -1] > 0).all()            # fully neutral at the top
        assert (lc["brightness_temp"][:, :, 0] == 0).mean() > 0.9     # ~fully ionized at the bottom


def test_shared_rotation():
    with tempfile.TemporaryDirectory() as d:
        zre, _ = make_run(d)
        z_arr, lc = _lightcone(d, seed=3)
        cell = L_H / H / N
        assert (z_arr[-1] - z_arr[0]) > 0 and len(z_arr) > 3 * N     # several wrap cycles
        # linear identity holds only where BOTH bracketing snapshots are fully neutral
        z_full = min(z for z in SNAP_Z if z > zre.max())
        neutral = z_arr >= z_full
        assert neutral.sum() > 20
        tb, rho1 = lc["brightness_temp"][:, :, neutral], 1.0 + lc["density"][:, :, neutral]
        assert np.allclose(tb, T0 * rho1, rtol=2e-4, atol=1e-3), np.abs(tb - T0 * rho1).max()
        # now stitch Tb ALONE with another seed: the identity must break
        snap_z, tbx = np.array(SNAP_Z), {}
        for z in SNAP_Z:
            tbx[z] = aio.read_tb(os.path.join(d, "output/21cm", f"Tb_L={L_H:g}_N={N}_z={z:05.2f}.dat"), N)
        other = stitch_field(tbx, snap_z, z_arr, float(z_arr.min()), cell, N, 0.0, 4, "wrap")[:, :, neutral]
        c = np.corrcoef(other.ravel(), (T0 * rho1).ravel())[0, 1]
        assert c < 0.6, f"different rotation should decorrelate, got r={c:.2f}"


def test_tb_scale_only_when_asked():
    with tempfile.TemporaryDirectory() as d:
        make_run(d)
        _, a = _lightcone(d)
        _, b = _lightcone(d, tb_scale=0.5)
        assert np.allclose(b["brightness_temp"], 0.5 * a["brightness_temp"], rtol=1e-6, atol=1e-6)
        assert np.array_equal(a["density"], b["density"])


def test_script_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        run = os.path.join(d, "seed_1")
        make_run(run)
        cfg_src = open(os.path.join(ROOT, "configs/variants/amber_q512.yaml")).read()
        cfg = cfg_src.replace("hii_dim: 256", f"hii_dim: {N}").replace("756.724800472953", repr(L_H / H))
        cfg = "\n".join(l if "cache_dir" not in l and "plot_dir" not in l else
                        ("  cache_dir: " + os.path.join(d, "cache") if "cache_dir" in l else "  plot_dir: " + os.path.join(d, "plots"))
                        for l in cfg.splitlines())
        cfgp = os.path.join(d, "cfg.yaml")
        open(cfgp, "w").write(cfg)
        cmd = [sys.executable, os.path.join(ROOT, "scripts/01a_stitch_lightcones_amber.py"), "--config", cfgp,
               "--run-dir-template", os.path.join(d, "seed_{seed}"), "--seeds", "1"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        npz = os.path.join(d, "cache", "seed_1", "stitched_lightcone_seed1.npz")
        z = np.load(npz)
        for key in ("density", "xH_box", "velocity_z", "brightness_temp", "z_arr", "pos_axis"):
            assert key in z.files, key
        assert str(z["source"]) == "amber" and float(z["tb_scale"]) == 1.0
        assert len(glob.glob(os.path.join(run, "output/cmb/fields_z=*.dat"))) == len(SNAP_Z)   # not deleted by default
        os.remove(npz)
        r = subprocess.run(cmd + ["--delete-raw"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert glob.glob(os.path.join(run, "output/cmb/fields_z=*.dat")) == []
        assert len(glob.glob(os.path.join(run, "output/21cm/Tb_*.dat"))) == len(SNAP_Z)        # 21cm kept
        assert os.path.exists(os.path.join(run, "output/reion/zre_test.dat"))


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("all tests passed")
