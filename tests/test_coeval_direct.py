"""
Synthetic tests for src/ksz2_21cm/correlation/coeval_direct.py (scripts/18).

  1. ksz_pixel_weights reproduces lightcone_integral.compute_ksz_map exactly.
  3. Limber kernels: C(Theta^2 x X) scales as W^2 kappa for uniform weights.
  4. power3d_cross: white noise gives variance x cell volume.
  5. Ma eq 8: the Limber prediction matches the measured kSZ x 21cm cross power of a
     synthetic linear-theory field (Gaussian delta, v_par from linear theory, T21 = b delta),
     including the sign and the 1/k^2 (LOS cancellation).

Run:  python -m pytest tests/test_coeval_direct.py -q
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ksz2_21cm.correlation import coeval_direct as cd   # noqa: E402
from ksz2_21cm.ksz.lightcone_integral import compute_ksz_map   # noqa: E402
from ksz2_21cm.ksz.optical_depth import compute_tau, compute_visibility, compute_patchy_mask   # noqa: E402
from ksz2_21cm.utils.constants import MPC_CM, C_CGS, ne0_cgs   # noqa: E402


def _gauss_box(shape, cell, pk, rng):
    """Gaussian random field with P(k) on a periodic grid (rfft normalisation as power3d_cross)."""
    nx, ny, nz = shape
    kx = np.fft.fftfreq(nx, d=cell) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=cell) * 2 * np.pi
    kz = np.fft.rfftfreq(nz, d=cell) * 2 * np.pi
    kk = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)
    vol = nx * ny * nz * cell ** 3
    amp = np.sqrt(np.where(kk > 0, pk(np.where(kk > 0, kk, 1.0)), 0.0) * (nx * ny * nz) ** 2 / vol / 2.0)
    F = amp * (rng.standard_normal(kk.shape) + 1j * rng.standard_normal(kk.shape))
    d = np.fft.irfftn(F, s=shape, axes=(0, 1, 2))
    return d, (kx, ky, kz, kk)


def test_weights_reproduce_ksz_map():
    rng = np.random.default_rng(1)
    n, nz = 8, 60
    z_arr = np.linspace(5.0, 12.0, nz)
    pos = 6000.0 + 3.1 * np.arange(nz)
    x_e = np.clip(1.0 - (z_arr - 5.0) / 6.0, 0.0, 1.0)
    ne0, tau0 = ne0_cgs(), 0.03
    d1 = 1.0 + 0.3 * rng.standard_normal((n, n, nz))
    xh = rng.uniform(0, 1, (n, n, nz))
    v = 1e7 * rng.standard_normal((n, n, nz)) / MPC_CM
    z_mid, ds, _, tau = compute_tau(x_e, z_arr, pos, ne0=ne0, tau0=tau0)
    _, _, vis3 = compute_visibility(tau, z_arr, z_mid)
    _, m3 = compute_patchy_mask(x_e)
    ref = compute_ksz_map(d1, xh, v, z_arr, ds, vis3, ne0=ne0, patchy_mask_3D=m3)
    W = cd.ksz_pixel_weights(z_arr, pos, x_e, ne0, tau0)["W"]
    theta = d1 * xh * v / (C_CGS / MPC_CM)
    got = np.tensordot(theta, W, axes=([2], [0]))
    assert np.allclose(got, ref, rtol=1e-12, atol=0)


def test_limber_kernel_scaling():
    rng = np.random.default_rng(3)
    N, cell = 32, 3.125
    theta = rng.standard_normal((N, N, N))
    tb = rng.standard_normal((N, N, N)) + theta ** 2
    b = cd.Binner(N, cell, np.logspace(-1.5, 0.0, 10))
    kap = cd.limber_kernels(theta, tb, b, cell, 0.01, 0.0)
    Wc = 0.37
    pred = Wc ** 2 * kap["masq"]                 # sum_i W^2 w_i kappa with w_i = 1/N
    Theta = Wc * theta.sum(axis=2)
    Theta = Theta - Theta.mean()                  # 04b removes the map mean before high-passing/squaring
    meas = b.cross(b.fft(Theta ** 2), b.fft(tb.mean(axis=2)))
    assert np.allclose(pred, meas, rtol=1e-10, equal_nan=True)


def test_power3d_white_noise():
    rng = np.random.default_rng(4)
    n, L = 48, 300.0
    a = rng.standard_normal((n, n, n)) * 2.0
    edges = np.linspace(0.05, 0.9, 8)
    P = cd.power3d_cross(a, a, L, edges)
    expect = 4.0 * (L / n) ** 3
    assert np.allclose(P, expect, rtol=0.05)


def test_maeq8_linear_theory():
    """Measured <Theta T> of a linear-theory field vs the eq-8 Limber prediction."""
    rng = np.random.default_rng(7)
    nt, nz, cell = 64, 768, 4.0
    aHf_over_c = 2.0e-4                    # a H f / c [1/Mpc], constant (single epoch, D const)
    bias = 3.0
    pk = lambda k: 2.0e4 * (k / 0.05) ** -1.5 * np.exp(-(k / 0.4) ** 2)
    s = cell * np.arange(nz)
    s0 = s[nz // 2]
    Wc = np.exp(-0.5 * ((s - s0) / 350.0) ** 2) * (1.0 + 0.4 * (s - s0) / 350.0)   # smooth, -> 0 at both ends
    xbar = 0.6
    w21 = np.exp(-0.5 * ((s - s0 - 60.0) / 150.0) ** 2)
    w21 /= w21.sum()
    edges3 = np.logspace(np.log10(2 * np.pi / (nz * cell)), np.log10(np.pi / cell), 60)
    b = cd.Binner(nt, cell, np.logspace(np.log10(2 * np.pi / (nt * cell)), np.log10(0.25), 14))
    meas = []
    for _ in range(24):
        d, (kx, ky, kz, kk) = _gauss_box((nt, nt, nz), cell, pk, rng)
        Fd = np.fft.rfftn(d)
        with np.errstate(divide="ignore", invalid="ignore"):
            vz = np.fft.irfftn(np.where(kk > 0, 1j * kz[None, None, :] / kk ** 2, 0.0) * Fd, s=d.shape, axes=(0, 1, 2))
        theta = xbar * aHf_over_c * vz                      # x_e v_par / c with v = i aHf delta k/k^2
        Theta = np.tensordot(theta, Wc * cell, axes=([2], [0]))   # W_i = W(s_i) ds
        T = np.tensordot(bias * d, w21, axes=([2], [0]))
        meas.append(b.cross(b.fft(Theta), b.fft(T)))
    meas = np.mean(meas, axis=0)
    ones = np.ones_like(s)
    psi = cd.psi_par(s, Wc, xbar * ones, ones, aHf_over_c * C_CGS / MPC_CM * ones, ones, ones)
    k = b.k_centers
    Pk = np.array([bias * pk(k)] * nz)
    pred = cd.maeq8_prediction(k, psi, w21, Pk)
    sel = (k > 0.04) & (k < 0.25) & np.isfinite(meas) & np.isfinite(pred)
    ratio_band = np.median(meas[sel] / pred[sel])   # Limber holds for k >> 1/(window width)
    assert sel.sum() >= 4
    # 24 realisations leave ~10% noise on the band amplitude; a sign error or a missing
    # factor of 2 (or of k^2) would fail by far more
    assert abs(ratio_band - 1.0) < 0.25, (ratio_band, meas[sel] / pred[sel])
