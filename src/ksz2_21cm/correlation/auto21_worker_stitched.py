# =============================================================================
# auto21_worker_stitched.py  (NEW, 2026-10-14)
#
# 21cm brightness-temperature AUTO power spectrum from the stitched lightcone, one seed,
# for a list of redshifts z0. For each z0 a LOS slab z0 +/- slab_dz/2 is cut out of the
# stitched brightness_temp (21cmFAST native, mK), mean-subtracted, Hann-windowed along
# the LOS (the slab is not periodic), and its 3D power spectrum is spherically averaged:
#
#     P(k) = <|FFT|^2> * cell^3 / (Nx Ny Nz <w^2>)       [mK^2 Mpc^3]
#     Delta^2(k) = k^3 P(k) / (2 pi^2)                   [mK^2]
#
# Reference, from the SAME slab: the matter power Delta^2_delta(k) of the 21cmFAST
# density field and T0 = <bt> / <x_HI (1+delta)> (the mean brightness temperature of
# fully-neutral gas at that z, T_S >> T_gamma, no RSD), so T0^2 Delta^2_delta is the
# "neutral limit" the 21cm power would approach if reionization had no effect.
# Also stored: the slab-mean ionized fraction.
#
# Cache: <cache_dir>/seed_<N>/auto21_stitched_seed<N><tag>.npy (small)
# =============================================================================

import os

import numpy as np


def _spectrum(field, w, cell, kbins, kmag, norm_w2):
    f = (field - field.mean()) * w
    F = np.fft.fftn(f)
    p3 = np.abs(F) ** 2 * cell**3 / (F.size * norm_w2)
    idx = np.digitize(kmag.ravel(), kbins) - 1
    nb = len(kbins) - 1
    ok = (idx >= 0) & (idx < nb)
    cnt = np.bincount(idx[ok], minlength=nb).astype(float)
    s = np.bincount(idx[ok], weights=p3.ravel()[ok], minlength=nb)
    P = np.full(nb, np.nan)
    good = cnt > 0
    P[good] = s[good] / cnt[good]
    return P, cnt


def compute_auto21_for_seed_stitched(args):
    """args: (seed, npz_path, cache_dir, pix_size_Mpc, z0_list, slab_dz, kbins, tag)
    Returns (seed, {z0: dict} or None, status)."""
    seed, npz_path, cache_dir, cell, z0_list, slab_dz, kbins, tag = args
    sd = os.path.join(cache_dir, f"seed_{seed}")
    os.makedirs(sd, exist_ok=True)
    cache = os.path.join(sd, f"auto21_stitched_seed{seed}{tag}.npy")
    if os.path.exists(cache):
        try:
            r = np.load(cache, allow_pickle=True).item()
            return (seed, r, f"cached ({len(r)} z0)")
        except Exception:
            pass
    try:
        d = np.load(npz_path)
        z_arr = np.asarray(d["z_arr"], dtype=np.float64)
        bt, xh, den = d["brightness_temp"], d["xH_box"], d["density"]
        kcen = np.sqrt(kbins[1:] * kbins[:-1])
        res = {}
        for z0 in z0_list:
            idx = np.where((z_arr >= z0 - slab_dz / 2) & (z_arr <= z0 + slab_dz / 2))[0]
            if len(idx) < 8:
                continue
            T = np.asarray(bt[:, :, idx], dtype=np.float64)
            X = np.asarray(xh[:, :, idx], dtype=np.float64)
            R = np.asarray(den[:, :, idx], dtype=np.float64)
            # stitched `density` is RAW delta (mean ~0), NOT 1+delta (see stitch_from_coeval.py).
            # The old R/R.mean() blew up for a ~zero-mean field and corrupted T0 and Delta^2_delta.
            R = (1.0 + R) if abs(R.mean()) < 0.5 else R / R.mean()
            n = T.shape[2]
            w = np.hanning(n + 2)[1:-1][None, None, :]
            norm_w2 = float(np.mean(w**2))
            kx = np.fft.fftfreq(T.shape[0], d=cell) * 2 * np.pi
            ky = np.fft.fftfreq(T.shape[1], d=cell) * 2 * np.pi
            kz = np.fft.fftfreq(n, d=cell) * 2 * np.pi
            kmag = np.sqrt(kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2)
            P21, cnt = _spectrum(T, w, cell, kbins, kmag, norm_w2)
            Pd, _ = _spectrum(R - 1.0, w, cell, kbins, kmag, norm_w2)
            T0 = float(T.mean() / np.mean(X * R))
            fac = kcen**3 / (2 * np.pi**2)
            res[float(z0)] = {
                "z0": float(z0), "k": kcen, "n_modes": cnt, "n_los": int(n),
                "D2_21": fac * P21, "D2_delta": fac * Pd, "T0": T0,
                "xe": float(1.0 - X.mean()), "Tmean": float(T.mean()),
            }
        np.save(cache, res)
        return (seed, res, f"computed ({len(res)} z0)")
    except Exception as e:
        import traceback
        return (seed, None, f"failed: {e}\n{traceback.format_exc()}")
