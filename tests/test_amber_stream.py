"""
Tests for stitcher v2 (src/ksz2_21cm/simulate/amber_stream.py, utils.cosmology.RunCosmology), the three ksz-pipeline
fixes of 2026-10-26, on SYNTHETIC AMBER output in the real byte layout (make_run from test_amber_stitch).

  1. view_slab is exact integer indexing: default view == arr[:, :, y]; every other view is a permutation of the
     source slab (nothing smoothed) and equals an independently built reference.
  2. the stream output equals an independent direct construction (z-blend of per-snapshot x_HII * mom_axis, delta,
     Tb, x_H under the cycle's view) at several LOS pixels.
  3. the momentum is the component ALONG THE VIEW'S LOS AXIS (mom[c] = 10 (c+1) km/s, fully ionized pixels).
  4. no division by rho: with rho2 ~ 0 / negative cells and normal momentum q stays bounded by |mom|; the v1
     stitcher's velocity_z blows up on the same input (this is what the fix removes).
  5. re-orientation keeps all the power of a white-noise field (variance of a slab is unchanged); v1's bilinear rotation
     (geometry='rotate') loses a third of it.
  6. Tb and delta share one view: where both bracketing snapshots are neutral Tb = T0 (1 + delta) exactly.
  7. RunCosmology: chi from the run json equals the direct integral of AMBER's H(z); a Chen+23-like run differs from
     Planck18 by more than 1% at z = 5; get_cosmology(use: amber) returns it; pos_axis of the stitched file uses it.
  8. 02_compute_ksz_maps_stitched.py on a v2 file: integrand = x_HII * mom (no division).

Run:  python -m pytest tests/test_amber_stream.py -q
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from test_amber_stitch import make_run, N, L_H, H, SNAP_Z, T0, AMBER_META                        # noqa: E402
from ksz2_21cm.simulate import amber_io as aio                                                # noqa: E402
from ksz2_21cm.simulate import amber_stream as ast                                            # noqa: E402
from ksz2_21cm.simulate.stitch_from_amber import stitch_lightcone_from_amber                  # noqa: E402
from ksz2_21cm.simulate.stitch_from_coeval import build_los_z_grid                             # noqa: E402
from ksz2_21cm.utils.cosmology import RunCosmology, get_cosmology                              # noqa: E402

CELL = L_H / H / N


def _run_stream(d, seed=3, geometry="reorient"):
    cos = RunCosmology.from_run_dir(d)
    z_arr = ast.build_los_z_grid_cosmo(min(SNAP_Z), max(SNAP_Z), L_H / cos.h / N, cos)
    pos = np.asarray(cos.comoving_distance(z_arr).value)
    return z_arr, pos, ast.stitch_lightcone_from_amber_stream(d, z_arr, pos, cos.h, N=N, wrap_cycle_seed=seed,
                                                              geometry=geometry)


def _manual_slab(arr, y, view):
    """Independent re-implementation of view_slab: axis -> last, roll the other two, take the slab."""
    axis, sh = view
    other = [k for k in range(3) if k != axis]
    b = np.roll(arr, shift=(-sh[other[0]], -sh[other[1]]), axis=(other[0], other[1]))
    return np.take(b, (y + sh[axis]) % N, axis=axis)


def test_view_slab_exact():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((N, N, N))
    assert np.array_equal(ast.view_slab(a, 5), a[:, :, 5])
    for axis in range(3):
        for shift in [(0, 0, 0), (3, 7, 11), (15, 1, 9)]:
            v = (axis, shift)
            got = ast.view_slab(a, 4, v)
            assert np.array_equal(got, _manual_slab(a, 4, v))
            assert np.array_equal(np.sort(got.ravel()), np.sort(np.take(a, (4 + shift[axis]) % N, axis=axis).ravel()))
    # deterministic in (seed, cycle), different between cycles
    assert ast.reorient_params(2, 7, N) == ast.reorient_params(2, 7, N)
    assert len({ast.reorient_params(c, 7, N) for c in range(12)}) > 6


def test_stream_matches_direct_construction():
    with tempfile.TemporaryDirectory() as d:
        zre, truth = make_run(d)
        z_arr, pos, lc = _run_stream(d, seed=5)
        sz = np.array(SNAP_Z)
        assert lc["views"].shape[0] >= 3                                    # several wrap cycles, several views
        for j in (0, 7, N + 3, 2 * N + 11, len(z_arr) - 1):
            dd = pos[j] - pos[0]
            y, cyc = int(round(dd / CELL)) % N, int(dd // (CELL * N))
            view = ast.reorient_params(cyc, 5, N)
            i0 = int(np.clip(np.searchsorted(sz, z_arr[j]) - 1, 0, len(sz) - 2))
            w1 = (z_arr[j] - sz[i0]) / (sz[i0 + 1] - sz[i0]); w0 = 1 - w1
            zs = _manual_slab(zre.astype(np.float64), y, view)
            exp = {k: 0.0 for k in ("density", "q_z", "xH_box", "brightness_temp")}
            for w, z in ((w0, sz[i0]), (w1, sz[i0 + 1])):
                rho, mom, tb = truth[z]
                xH = (zs <= z).astype(np.float64)
                exp["density"] = exp["density"] + w * (_manual_slab(rho.astype(np.float64), y, view) - 1.0)
                exp["q_z"] = exp["q_z"] + w * (1.0 - xH) * _manual_slab(mom[view[0]].astype(np.float64), y, view) * 1e5
                exp["xH_box"] = exp["xH_box"] + w * xH
                exp["brightness_temp"] = exp["brightness_temp"] + w * _manual_slab(tb.astype(np.float64), y, view)
            for k, v in exp.items():
                assert np.allclose(lc[k][:, :, j], v, rtol=2e-5, atol=1e-3 if k == "q_z" else 1e-5), (j, k)


def test_momentum_component_follows_the_los_axis():
    with tempfile.TemporaryDirectory() as d:
        make_run(d, const_mom=True)
        seen = set()
        for seed in range(1, 9):
            z_arr, pos, lc = _run_stream(d, seed=seed)
            # pixels that are fully ionized in the lightcone itself (low z): there q = mom along the view's LOS axis
            sel = np.where(lc["xH_box"].max(axis=(0, 1)) < 1e-9)[0]
            assert sel.size > 10
            for j in sel:
                axis = ast.reorient_params(int((pos[j] - pos[0]) // (CELL * N)), seed, N)[0]
                seen.add(axis)
                assert np.allclose(lc["q_z"][:, :, j], 10.0 * (axis + 1) * 1e5, rtol=1e-6), (seed, j, axis)
        assert seen == {0, 1, 2}                                # every LOS axis really was exercised


def test_no_division_by_rho():
    with tempfile.TemporaryDirectory() as d:
        make_run(d, bad_rho=True)
        z_arr, pos, lc = _run_stream(d, seed=3)
        mom_max = 1e5 * 1.2 * max(float(np.abs(aio.read_fields(os.path.join(d, "output/cmb", f"fields_z={z:05.2f}.dat"), N)[2]).max())
                                  for z in SNAP_Z)
        assert np.isfinite(lc["q_z"]).all()
        assert np.abs(lc["q_z"]).max() <= mom_max, (np.abs(lc["q_z"]).max(), mom_max)
        # the same input through the v1 stitcher: velocity_z = mom / rho explodes in the cells with rho ~ 1e-3
        z1 = build_los_z_grid(min(SNAP_Z), max(SNAP_Z), CELL)
        v1 = stitch_lightcone_from_amber(d, z1, H, N=N, wrap_cycle_seed=3)["velocity_z"]
        assert np.abs(v1).max() > 20 * mom_max, (np.abs(v1).max(), mom_max)
        # the diagnostics are recorded
        assert lc["rho2_min"].min() < 0 and lc["rho2_frac_neg"].max() > 0


def test_reorient_keeps_power_rotate_does_not():
    with tempfile.TemporaryDirectory() as d:
        make_run(d, white=True)
        zsel = None
        out = {}
        for geo in ("reorient", "rotate"):
            z_arr, pos, lc = _run_stream(d, seed=4, geometry=geo)
            out[geo] = lc
        # slices that sit exactly on one snapshot's z are pure single-snapshot slabs: variance should equal the
        # variance of the white noise (0.3^2) under exact re-orientation
        sz = np.array(SNAP_Z)
        j = int(np.argmin(np.abs(z_arr - sz[3])))
        # compare the mean slab variance over the interior of the lightcone: blends of 2 independent snapshots have
        # variance (w0^2 + w1^2) * 0.09, so use the ratio rotate / reorient, which cancels the blend factor
        v_re = np.mean([out["reorient"]["density"][:, :, k].var() for k in range(len(z_arr))])
        v_ro = np.mean([out["rotate"]["density"][:, :, k].var() for k in range(len(z_arr))])
        assert v_ro / v_re < 0.85, (v_ro, v_re)
        # exactness: with weight ~1 on a snapshot the variance equals the source slab variance
        rho = aio.read_fields(os.path.join(d, "output/cmb", f"fields_z={sz[3]:05.2f}.dat"), N)[1]
        assert abs(out["reorient"]["density"][:, :, j].var() / rho.var() - 1) < 0.35   # blend weight not exactly 1
        sub = np.where(np.abs(z_arr - sz[3]) < 1e-9)[0]
        for k in sub:
            assert abs(np.sort(out["reorient"]["density"][:, :, k].ravel()).std() / np.sqrt(rho.var()) - 1) < 0.5


def test_tb_and_density_share_the_view():
    with tempfile.TemporaryDirectory() as d:
        zre, _ = make_run(d)
        z_arr, pos, lc = _run_stream(d, seed=3)
        z_full = min(z for z in SNAP_Z if z > zre.max())
        neutral = z_arr >= z_full
        assert neutral.sum() > 20
        tb, rho1 = lc["brightness_temp"][:, :, neutral], 1.0 + lc["density"][:, :, neutral]
        assert np.allclose(tb, T0 * rho1, rtol=2e-4, atol=1e-3), np.abs(tb - T0 * rho1).max()


def _chi_quad(z, p):
    from scipy.integrate import quad
    E = lambda zz: np.sqrt(p["orad"] * (1 + zz) ** 4 + p["om"] * (1 + zz) ** 3 + p["ol"])
    return 299792.458 / (100 * p["h"]) * quad(lambda zz: 1 / E(zz), 0, z, epsabs=0, epsrel=1e-12)[0]


def test_run_cosmology():
    p = AMBER_META["amber_params"]
    c = RunCosmology.from_params(p)
    for z in (0.5, 4.5, 6.0, 10.0, 18.0, 30.0):
        assert abs(c.comoving_distance(z).value / _chi_quad(z, p) - 1) < 1e-6, z
    from astropy.cosmology import Planck18
    for z in (4.5, 18.0):                                       # the planck18 preset IS Planck18 to ~1e-6
        assert abs(c.comoving_distance(z).value / Planck18.comoving_distance(z).value - 1) < 1e-5
    # a Chen+23-like run (Om 0.3, h 0.7): Planck18 would put every distance wrong by > 1%
    ch = RunCosmology(0.3 - 8.5e-5, 0.7, 8.5e-5, 0.7)
    assert abs(Planck18.comoving_distance(5.0).value / ch.comoving_distance(5.0).value - 1) > 0.01
    # via get_cosmology(cfg)
    cfg = {"cosmology": {"use": "amber", "amber": p}}
    g = get_cosmology(cfg)
    assert abs(float(g.comoving_distance(8.0).value) / c.comoving_distance(8.0).value - 1) < 1e-14
    assert c.matches(g) and not c.matches(ch)
    # the stitched file's pos_axis comes from the run json
    with tempfile.TemporaryDirectory() as d:
        make_run(d)
        z_arr, pos, lc = _run_stream(d)
        assert np.allclose(lc["pos_axis"], RunCosmology.from_run_dir(d).comoving_distance(z_arr).value, rtol=1e-12)
        assert np.allclose(np.diff(pos), CELL, rtol=2e-3)       # uniform in the RUN's comoving distance


def test_script02_integrates_q():
    """02 on a v2 file == x_HII * mom integrated directly; and it still reads a v1 / 21cmFAST-style file."""
    with tempfile.TemporaryDirectory() as d:
        run = os.path.join(d, "seed_1")
        make_run(run)
        cfg_src = open(os.path.join(ROOT, "configs/variants/amber_q512.yaml")).read()
        cfg = cfg_src.replace("hii_dim: 256", f"hii_dim: {N}").replace("756.724800472953", repr(L_H / H))
        cfg = "\n".join(l if "cache_dir" not in l and "plot_dir" not in l else
                        ("  cache_dir: " + os.path.join(d, "cache") if "cache_dir" in l else "  plot_dir: " + os.path.join(d, "plots"))
                        for l in cfg.splitlines())
        cfgp = os.path.join(d, "cfg.yaml"); open(cfgp, "w").write(cfg)
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/01a_stitch_lightcones_amber.py"), "--config", cfgp,
                            "--run-dir-template", os.path.join(d, "seed_{seed}"), "--seeds", "1"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/02_compute_ksz_maps_stitched.py"), "--config", cfgp,
                            "--seeds", "1", "--no-plots"], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "no division by rho" in r.stdout and "from the run" in r.stdout
        kmap = np.load(os.path.join(d, "cache", "kSZ_maps", "kSZ_map_stitched_seed1.npy"))
        # direct: same weights, integrand q
        from ksz2_21cm.ksz.lightcone_integral import compute_ksz_map
        from ksz2_21cm.ksz.optical_depth import analytic_tau_below, compute_tau, compute_visibility, compute_patchy_mask
        from ksz2_21cm.utils.constants import MPC_CM
        z = np.load(os.path.join(d, "cache", "seed_1", "stitched_lightcone_seed1.npz"))
        x_e = 1.0 - z["xH_box"].mean(axis=(0, 1))
        ne0 = float(z["ne0_cgs"])
        z_mid, ds, _, tau = compute_tau(x_e, z["z_arr"], z["pos_axis"], ne0=ne0, tau0=analytic_tau_below(float(z["z_arr"].min())))
        _, _, vis3 = compute_visibility(tau, z["z_arr"], z_mid)
        _, m3 = compute_patchy_mask(x_e)
        ref = compute_ksz_map(np.ones_like(z["q_z"]), np.ones_like(z["q_z"]), z["q_z"] / MPC_CM, z["z_arr"], ds, vis3,
                              ne0=ne0, patchy_mask_3D=m3)
        assert np.allclose(kmap, ref, rtol=1e-10, atol=0)
        assert np.abs(kmap).max() > 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok ", name)
    print("all tests passed")
