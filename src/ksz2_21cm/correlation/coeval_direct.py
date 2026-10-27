# =============================================================================
# coeval_direct.py
#
# kSZ x 21cm statistics computed DIRECTLY from coeval snapshot boxes (no lightcone
# stitching, no interpolation between snapshots, no wrap-cycle rotations), to set
# against the stitched-lightcone results of scripts 04 (Zhou) and 04b (Ma).
# Written for this repo; nothing is imported from ksz-lae-xcorr.
#
# Two coeval estimators, on the SAME LOS pixel grid, kSZ weights, 21cm windows and 2D k
# binning as the lightcone workers. Both respect the fact that the observed kSZ is ONE map
# integrated over all of reionization and squared once (both are Limber expressions derived
# from that single map). A per-snapshot "slab" estimator (square each slab's own kSZ map, as
# ksz-lae-xcorr scripts/44 does) was tried and removed on 2026-10-26: squaring slab maps
# instead of the full map drops the kSZ just beyond every slab edge, which still correlates
# with the 21cm inside, and so biases the squared statistics low.
#
# 1. LIMBER / squeezed-bispectrum estimator (Kramer+25 eqs 9-11 analogue), for the SQUARED
#    statistics (Zhou kSZ^2 x 21cm^2, Ma kSZ^2 x 21cm). Per snapshot, project the WHOLE periodic box with unit weights
#    (Theta_box = sum over all cells of theta, X_box = LOS mean of the 21cm field or of
#    its k_par-filtered square) and measure  kappa_s(k) = C( Theta_box^2, X_box ).
#    In the Limber limit (correlation length << window) kappa_s is the k_par = 0
#    triangle power spectrum of the hybrid bispectrum <X q q> per unit LOS length, and
#        C(k) = sum_{i in chunk} W_i^2 w_i kappa_{s(i)}(k)
#    for kSZ weights W_i and 21cm window weights w_i (sum w_i = 1). It drops only
#    correlations longer than the box (800 Mpc).
#
# 2. Ma+18 eq 8 (Limber, linear theory) for LINEAR kSZ x 21cm. Writing the kSZ map as
#    Theta = int ds W(s) x_e(s) v_par/c with linear velocities v(k) = i a H f delta k/k^2,
#    integrating by parts along the LOS and taking the k_par = 0 (Limber) limit gives,
#    in the flat-sky comoving units of our 2D maps,
#        C(k) = -(1/k^2) int ds Psi_par(s) W21(s) P_{delta T21}(k, s),
#        Psi_par(s) = (1/D) d/ds [ W(s) x_e(s) a H f D / c ],
#    with W(s) the kSZ weight per Mpc and W21 the 21cm window per Mpc (int W21 ds = 1).
#    Two versions of the 3D cross spectrum, both measured from the snapshot box:
#      'maeq8'      : P_{delta T21} measured directly (21cmFAST's own brightness_temp)
#      'maeq8_lin'  : Ma's linearised form  T0 [ (1 - x_e) P_dd - x_e P_{d,dx} ]
#                     (kappa = 1 since our runs have T_S >> T_CMB, so P_{d,d kappa} = 0)
#    The sign follows OUR map convention (Theta = +int W q, v_z along increasing LOS
#    distance), so it is directly comparable with 04b's C_lin.
#
# Conventions (same as the lightcone workers): 2D cross power
#     C(k) = Re(conj(F_a) F_b) * pix_area / N^2 of mean-subtracted maps, binned in |k|
# with the caller's k_bins (np.digitize, bins[i] <= k < bins[i+1]); ell = k chi(z0);
# NO 1/chi^2 here (applied at plot time). kSZ maps are dimensionless dT/T, 21cm in mK.
# =============================================================================

import numpy as np

from ..utils.constants import SIGMA_T, MPC_CM, C_CGS

# ── LOS geometry and kSZ weights ─────────────────────────────────────────────


def ksz_pixel_weights(z_arr, pos_axis, x_e, ne0, tau0, patchy=True):
    """
    Per-pixel kSZ weights W_i with  Theta = sum_i W_i theta_i  reproducing
    lightcone_integral.compute_ksz_map EXACTLY (its midpoint-trapezoid rule, a^-2,
    exp(-tau) with tau from compute_tau(+tau0), and the patchy mask), where
    theta = (1+delta) x_HII v_los/c.

    Returns dict: W (per pixel, dimensionless), W_per_mpc (the continuous weight
    W(s) = ne0 sigma_T MPC_CM (1+z)^2 exp(-tau) mask, per Mpc), vis, mask, tau_at_lc.
    """
    from ..ksz.optical_depth import compute_tau, compute_visibility, compute_patchy_mask

    z_arr = np.asarray(z_arr, float)
    z_mid, ds, _dtau, tau = compute_tau(x_e, z_arr, pos_axis, ne0=ne0, tau0=tau0)
    tau_at_lc, vis, _ = compute_visibility(tau, z_arr, z_mid)
    mask = compute_patchy_mask(x_e)[0] if patchy else np.ones_like(z_arr)
    a2 = (1.0 / (1.0 + z_arr)) ** 2
    a2_mid = 0.5 * (a2[:-1] + a2[1:])
    seg = 0.5 * ne0 * SIGMA_T * (np.asarray(ds) * MPC_CM) / a2_mid     # one half-segment coefficient
    coef = np.zeros_like(z_arr)
    coef[:-1] += seg
    coef[1:] += seg
    W = coef * vis * mask
    W_per_mpc = ne0 * SIGMA_T * MPC_CM * (1.0 + z_arr) ** 2 * vis * mask
    return {"W": W, "W_per_mpc": W_per_mpc, "vis": vis, "mask": mask, "tau_at_lc": tau_at_lc}


def nearest_snapshot(z_pix, snap_z):
    """Index into sorted(snap_z) of the snapshot nearest in z to every pixel."""
    snap = np.asarray(sorted(float(s) for s in snap_z))
    mid = 0.5 * (snap[1:] + snap[:-1])
    return np.searchsorted(mid, np.asarray(z_pix, float))


def contiguous_groups(assign, idx):
    """[(snapshot index, pixel indices)] for the contiguous runs of pixels idx with equal assignment."""
    idx = np.asarray(idx)
    if len(idx) == 0:
        return []
    a = assign[idx]
    groups, start = [], 0
    for k in range(1, len(idx) + 1):
        if k == len(idx) or a[k] != a[start]:
            groups.append((int(a[start]), idx[start:k]))
            start = k
    return groups


def chunk_and_windows(z_arr, z0, slab_dz, narrow_dz):
    """
    Pixel indices of the chunk z0 +- slab_dz/2 and the 21cm window weights over those
    pixels, exactly as the lightcone workers select them (z >= lo & z <= hi; LOS mean):
      'slab'   : equal weights over the chunk         (04b 'slab' and 04's chunk)
      'narrow' : equal weights over z0 +- narrow_dz/2  (04b 'narrow')
      'slice'  : the single pixel nearest z0          (04b 'slice')
    Returns (idx_chunk, {variant: weights aligned with idx_chunk, summing to 1}).
    """
    z_arr = np.asarray(z_arr, float)
    idx = np.where((z_arr >= z0 - slab_dz / 2.0) & (z_arr <= z0 + slab_dz / 2.0))[0]
    zc = z_arr[idx]
    w = {"slab": np.full(len(idx), 1.0 / max(len(idx), 1))}
    nar = (zc >= z0 - narrow_dz / 2.0) & (zc <= z0 + narrow_dz / 2.0)
    w["narrow"] = nar / max(nar.sum(), 1)
    i_slice = int(np.argmin(np.abs(z_arr - z0)))
    w["slice"] = (idx == i_slice).astype(float)
    return idx, w


# ── 2D map cross power, as the lightcone workers ────────────────────────────


class Binner:
    """|k| binning of 2D maps identical to 04/04b (shifted-FFT layout)."""

    def __init__(self, npix, pix_size_mpc, k_bins):
        self.N = int(npix)
        self.pix_area = float(pix_size_mpc) ** 2
        dk = 2 * np.pi / (self.N * pix_size_mpc)
        kx = np.fft.fftshift(np.fft.fftfreq(self.N)) * self.N * dk
        self.kgrid = np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)
        self.k_bins = np.asarray(k_bins, float)
        self.k_centers = 0.5 * (self.k_bins[:-1] + self.k_bins[1:])
        nb = len(self.k_centers)
        b = np.digitize(self.kgrid, self.k_bins) - 1
        self.valid = (b >= 0) & (b < nb)
        self.bidx = np.where(self.valid, b, 0)
        self.counts = np.bincount(self.bidx[self.valid], minlength=nb).astype(float)
        self.nb = nb

    def fft(self, m):
        m = np.asarray(m, np.float64)
        return np.fft.fftshift(np.fft.fft2(m - m.mean()))

    def cross(self, Fa, Fb):
        x = np.real(np.conj(Fa) * Fb) * self.pix_area / self.N ** 2
        s = np.bincount(self.bidx[self.valid], weights=x[self.valid], minlength=self.nb)
        out = np.full(self.nb, np.nan)
        ok = self.counts > 0
        out[ok] = s[ok] / self.counts[ok]
        return out

    def highpass(self, m, k_cut):
        """Remove |k| < k_cut (04b: ell < ell_min_ksz with ell = k chi_ksz)."""
        F = np.fft.fftshift(np.fft.fft2(np.asarray(m, np.float64) - np.mean(m)))
        return np.real(np.fft.ifft2(np.fft.ifftshift(F * (self.kgrid >= k_cut))))


def kpar_filter_los(cube, cell_mpc, k_par_min):
    """Remove |k_par| <= k_par_min along axis 2 (the LOS) only, as 04's 3D filter does
    (its filter depends on k_z alone, so a 1D FFT along the LOS is the same operation)."""
    n = cube.shape[2]
    kz = np.abs(np.fft.fftfreq(n, d=cell_mpc) * 2 * np.pi)
    F = np.fft.fft(np.asarray(cube, np.float64), axis=2)
    return np.real(np.fft.ifft(F * (kz > k_par_min)[None, None, :], axis=2))


# ── 1. Limber / squeezed-bispectrum kernels ──────────────────────────────────


def limber_kernels(theta, tb, binner, cell_mpc, k_par_min, k_hp):
    """
    Per-snapshot kernels kappa(k) from whole-box projections with unit weights:
      'zhou'  : C( Theta_box^2,        LOS-mean[(k_par-filtered T21)^2] )
      'masq'  : C( HP(Theta_box)^2,    LOS-mean[T21] )
    Theta_box = sum over all cells of theta. Then C_pred = sum_i W_i^2 w_i kappa.
    """
    Theta = theta.sum(axis=2)
    tf = kpar_filter_los(tb, cell_mpc, k_par_min)
    X_z = (tf ** 2).mean(axis=2)
    X_l = np.asarray(tb, np.float64).mean(axis=2)
    return {
        "zhou": binner.cross(binner.fft(Theta ** 2), binner.fft(X_z)),
        "masq": binner.cross(binner.fft(binner.highpass(Theta, k_hp) ** 2), binner.fft(X_l)),
    }


# ── 2. Ma+18 eq 8 ────────────────────────────────────────────────────────────


def power3d_cross(a, b, box_len, k_edges):
    """Binned 3D cross power P_ab(k) [units(a) units(b) Mpc^3] of two (N,N,N) boxes."""
    n = a.shape[0]
    Fa = np.fft.rfftn(np.asarray(a, np.float64) - np.mean(a))
    Fb = np.fft.rfftn(np.asarray(b, np.float64) - np.mean(b))
    k1 = np.fft.fftfreq(n, d=box_len / n) * 2 * np.pi
    kz = np.fft.rfftfreq(n, d=box_len / n) * 2 * np.pi
    kk = np.sqrt(k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2)
    x = np.real(np.conj(Fa) * Fb)
    # rfft half-space: interior kz planes stand for two modes, kz = 0 and Nyquist for one
    wgt = np.full(kz.shape, 2.0)
    wgt[0] = 1.0
    if n % 2 == 0:
        wgt[-1] = 1.0
    wgt3 = np.broadcast_to(wgt[None, None, :], kk.shape)
    b_i = np.digitize(kk, k_edges) - 1
    ok = (b_i >= 0) & (b_i < len(k_edges) - 1)
    s = np.bincount(b_i[ok], weights=(x * wgt3)[ok], minlength=len(k_edges) - 1)
    c = np.bincount(b_i[ok], weights=wgt3[ok], minlength=len(k_edges) - 1)
    P = np.full(len(k_edges) - 1, np.nan)
    good = c > 0
    P[good] = s[good] / c[good] * box_len ** 3 / float(n) ** 6
    return P


def background(z, cosmo=None):
    """a, H [1/s], f = dlnD/dlna, D (normalised to 1 at z=0) from astropy (Planck18 default)."""
    from scipy.integrate import quad
    if cosmo is None:
        from astropy.cosmology import Planck18 as cosmo
    z = np.atleast_1d(np.asarray(z, float))

    def E(a):
        return float(cosmo.efunc(1.0 / a - 1.0))

    def D_unnorm(a):
        val, _ = quad(lambda x: 1.0 / (x * E(x)) ** 3, 1e-8, a, limit=200)
        return 2.5 * cosmo.Om0 * E(a) * val

    a = 1.0 / (1.0 + z)
    D0 = D_unnorm(1.0)
    D = np.array([D_unnorm(ai) for ai in a]) / D0
    eps = 1e-4
    f = np.array([(np.log(D_unnorm(ai * (1 + eps))) - np.log(D_unnorm(ai * (1 - eps)))) /
                  (np.log(1 + eps) - np.log(1 - eps)) for ai in a])
    H = np.asarray(cosmo.H(z).to("1/s").value, float)
    return a, H, f, D


def psi_par(pos_axis, W_per_mpc, x_e, a, H, f, D):
    """Psi_par(s) = (1/D) d/ds [ W(s) x_e(s) a H f D / c ]  [1/Mpc^2] on the LOS pixel grid."""
    c_mpc_s = C_CGS / MPC_CM
    G = W_per_mpc * x_e * a * H * f * D / c_mpc_s
    return np.gradient(G, np.asarray(pos_axis, float)) / D


def maeq8_prediction(k, psi_pix, w_pix, P_of_k_pix):
    """
    C(k) = -(1/k^2) sum_i w_i Psi_par(s_i) P(k, s_i)   (w_i = 21cm window weights, sum 1).
    P_of_k_pix : (n_pix, nk) the 3D cross spectrum at each window pixel, evaluated at k.
    """
    k = np.asarray(k, float)
    s = np.tensordot(np.asarray(w_pix) * np.asarray(psi_pix), np.asarray(P_of_k_pix), axes=([0], [0]))
    with np.errstate(divide="ignore", invalid="ignore"):
        return -s / k ** 2


def linear_vz(delta, box_len, aHf_cm):
    """
    Linear-theory LOS velocity from the box's own density, in the convention eq 8 assumes:
        v(k) = i a H f delta(k) k / k^2   ->   v_z(k) = i (a H f) delta(k) k_z / k^2
    (infall towards overdensities). aHf_cm = a H f in [1/s] times MPC_CM, so v_z is in cm/s,
    the units run_coeval_fields gives 21cmFAST's own v_z in.
    """
    n = delta.shape[0]
    k1 = np.fft.fftfreq(n, d=box_len / n) * 2 * np.pi
    kz = np.fft.rfftfreq(n, d=box_len / n) * 2 * np.pi
    k2 = k1[:, None, None] ** 2 + k1[None, :, None] ** 2 + kz[None, None, :] ** 2
    F = np.fft.rfftn(np.asarray(delta, np.float64) - np.mean(delta))
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.where(k2 > 0, 1j * aHf_cm * kz[None, None, :] / k2, 0.0) * F
    return np.fft.irfftn(G, s=delta.shape, axes=(0, 1, 2))


def velocity_check(vz, delta, box_len, aHf_cm, k_edges, k_max=0.1):
    """
    How 21cmFAST's v_z relates to the linear velocity eq 8 assumes, from one box:
      r(k)   = P(v_z, v_lin) / P(v_lin, v_lin)   (sign AND amplitude; +1 = eq 8's convention exactly)
      amp(k) = sqrt(P(v_z) / P(v_lin))          (amplitude only)
    Returns dict with the binned arrays and the medians over k < k_max (linear scales).
    """
    vl = linear_vz(delta, box_len, aHf_cm)
    P_x = power3d_cross(vz, vl, box_len, k_edges)
    P_l = power3d_cross(vl, vl, box_len, k_edges)
    P_v = power3d_cross(vz, vz, box_len, k_edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = P_x / P_l
        amp = np.sqrt(P_v / P_l)
    kc = np.sqrt(k_edges[:-1] * k_edges[1:])
    sel = (kc < k_max) & np.isfinite(r)
    return {"r": r, "amp": amp, "r_lin": float(np.median(r[sel])) if sel.any() else np.nan,
            "amp_lin": float(np.median(amp[sel])) if sel.any() else np.nan}


def los_lag_cross(a, b, binner):
    """
    Full-field, non-Limber ingredient for projected cross spectra (no approximation besides using one
    snapshot box per redshift): for two (N, N, n) boxes a, b with the LOS along axis 2,
        G[d, kbin] = ring mean over |k_perp| of  Re < F2[a(z + d)] conj F2[b(z)] >_z  * pix_area / N^2
    for every LOS lag d = 0..n-1 in cells (periodic: d > n/2 are negative lags), in exactly the map
    convention of Binner.cross. A projection pair  A = sum_i W_i a_i,  B = sum_j w_j b_j  then has
        C_AB(k) = sum_{i,j} W_i w_j G[(i - j) mod n, k]
    -- every pair of LOS pixels, at its real separation, with the box's own nonlinear fields.
    """
    n = a.shape[2]
    Fa = np.fft.fft2(np.asarray(a, np.float64) - np.mean(a), axes=(0, 1))
    Fa = np.fft.fft(Fa, axis=2)
    Fb = np.fft.fft2(np.asarray(b, np.float64) - np.mean(b), axes=(0, 1))
    Fb = np.fft.fft(Fb, axis=2)
    X = np.fft.ifft(Fa * np.conj(Fb), axis=2) / n          # mean over z of F2a(z+d) conj F2b(z)
    del Fa, Fb
    X = np.fft.fftshift(np.real(X), axes=(0, 1)) * binner.pix_area / binner.N ** 2
    G = np.full((n, binner.nb), np.nan)
    ok = binner.counts > 0
    flat_b = binner.bidx[binner.valid]
    for d in range(n):
        s = np.bincount(flat_b, weights=X[:, :, d][binner.valid], minlength=binner.nb)
        G[d, ok] = s[ok] / binner.counts[ok]
    return G


def lag_prediction(G_by_pix, W, w_idx, w_val, n_box):
    """
    C(k) = sum_j w_j sum_i W_i G_{s(j)}[(i - j) mod n_box, k]  over kSZ pixels i within +-(n_box/2 - 1)
    cells of each 21cm pixel j (beyond half the box the periodic lag is ambiguous).
    G_by_pix : callable j -> (n_box, nb) lag table of the snapshot assigned to 21cm pixel j
    W        : (n_pix_total,) kSZ weights on the whole LOS grid
    w_idx, w_val : 21cm window pixel indices and weights
    """
    h = n_box // 2 - 1
    out = None
    for j, wj in zip(w_idx, w_val):
        if wj == 0:
            continue
        lo, hi = max(0, j - h), min(len(W), j + h + 1)
        i = np.arange(lo, hi)
        G = G_by_pix(j)[(i - j) % n_box]                      # (len(i), nb)
        c = wj * np.tensordot(W[lo:hi], G, axes=([0], [0]))
        out = c if out is None else out + c
    return out


def interp_P(k_edges, P, k):
    """Interpolate a binned P(k) (bin centres, log k) at k; NaN outside the measured range."""
    kc = np.sqrt(k_edges[:-1] * k_edges[1:])
    ok = np.isfinite(P)
    if ok.sum() < 2:                      # e.g. a fully ionized snapshot: no 21cm signal, T0 undefined
        return np.full(np.shape(k), np.nan)
    out = np.interp(np.log(k), np.log(kc[ok]), P[ok], left=np.nan, right=np.nan)
    return out
