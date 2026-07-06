#!/usr/bin/env python
# =============================================================================
# 05_compute_snr_forecast.py
#
# SNR forecast for kSZ^2 x 21cm^2, CMB-S4 x SKA1-Low, with:
#   - frequency-dependent T_sys (Zhou+25)
#   - both a diagonal Gaussian estimator and a full empirical-covariance
#     estimator built from seed-to-seed scatter
#
# This is the driver for SNR_rigorous_v2.py (your final/most rigorous SNR
# script). The noise-model and SNR-estimator functions themselves now live
# in src/ksz2_21cm/noise/snr_forecast.py; this script owns orchestration,
# caching, and plotting, per the src/ (reusable) vs scripts/ (executable
# steps) split.
#
# Prerequisites (run first): 01_run_lightcones.py, 02_compute_ksz_maps.py,
# 04_compute_cross_corr_sq.py.
#
# Run:
#   conda activate ksz2-21cm
#   python scripts/05_compute_snr_forecast.py --config configs/fiducial.yaml
# =============================================================================

import argparse
import glob
import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import py21cmfast as p21c

from ksz2_21cm.utils.config import load_config
from ksz2_21cm.utils.cosmology import get_cosmology
from ksz2_21cm.io.cache import load_seed_caches
from ksz2_21cm.plotting.style import apply_global_style, save_fig_both
from ksz2_21cm.plotting.zhou25_data import (
    ZHOU_FIG5, ZHOU_FIG3_ell, ZHOU_FIG3_D, ZHOU_FIG4_ell, ZHOU_FIG4_D,
    zhou5_anchor,
)
from ksz2_21cm.noise.snr_forecast import (
    T_CMB_uK, T_sys_zhou25, cmb_noise_map, thermal_noise_cube,
    snr_diagonal, snr_full_cov,
)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/fiducial.yaml")
args = parser.parse_args()

cfg = load_config(args.config)
apply_global_style()

cache_dir = cfg["paths"]["cache_dir"]
plot_dir  = cfg["paths"]["plot_dir"]
RANDOM_SEEDS    = cfg["simulation"]["random_seeds"]
HII_DIM         = cfg["simulation"]["hii_dim"]
BOX_LEN         = cfg["simulation"]["box_len"]
z_obs           = cfg["ksz"]["z_obs"]
k_par_min       = cfg["cross_correlation"]["k_par_min"]
delta_z         = cfg["cross_correlation"]["delta_z"]
z_chunk_centres = cfg["cross_correlation"]["z_chunk_centres"]
f_sky           = cfg["snr"]["f_sky"]
cosmo           = get_cosmology(cfg)

# ─────────────────────────────────────────────────────────────────────────────
# Instrument specification: CMB-S4 x SKA1-Low
# (see configs/fiducial.yaml -> snr: for the full preset lists; this script
#  currently runs the single combination used in the paper draft)
# ─────────────────────────────────────────────────────────────────────────────
CMB_EXP = dict(sigma_uK_arcmin=1.0, fwhm_arcmin=1.0, label='CMB-S4')
TS_EXP  = dict(t_obs_hr=1000.0, N_bl=512, A_eff_m2=962.0, label='SKA1-Low')
EXP_TAG = 'CMBS4_SKA'

# ─────────────────────────────────────────────────────────────────────────────
# 0. Map geometry + k-grid (identical formula to 03/04_compute_cross_corr*.py
#    — must match exactly, since the noisy-map cross-corr below re-does the
#    FFT/binning inline rather than calling the worker module)
# ─────────────────────────────────────────────────────────────────────────────
npix_side    = HII_DIM
box_size_Mpc = float(BOX_LEN)
pix_size_Mpc = box_size_Mpc / npix_side
pix_area     = pix_size_Mpc**2

dk    = 2 * np.pi / (npix_side * pix_size_Mpc)
kx    = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
ky    = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
kgrid = np.sqrt(kx[:, None]**2 + ky[None, :]**2)
k_bins    = np.logspace(np.log10(dk), np.log10(kgrid.max() * 0.9), 35)
k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load prerequisites: lightcones, kSZ maps, noise-free cross_corr_sq cache
# ─────────────────────────────────────────────────────────────────────────────
print("Loading lightcones + kSZ maps...")
lightcones, kSZ_maps = {}, {}
for seed in RANDOM_SEEDS:
    seed_dir = os.path.join(cache_dir, f"seed_{seed}")
    lc_files = sorted(glob.glob(os.path.join(seed_dir, "LightCone_*.h5")))
    if lc_files:
        lightcones[seed] = p21c.LightCone.read(lc_files[0])
    map_path = os.path.join(cache_dir, "kSZ_maps",
                            f"kSZ_map_z{z_obs:.1f}_seed{seed}.npy")
    if os.path.exists(map_path):
        kSZ_maps[seed] = np.load(map_path)
print(f"  {len(lightcones)} lightcones, {len(kSZ_maps)} kSZ maps loaded")

cross_corr_results_sq_all = load_seed_caches(
    "cross_corr_sq_seed{seed}.npy", RANDOM_SEEDS, cache_dir)
assert len(cross_corr_results_sq_all) > 0, (
    "No cross_corr_sq caches found — run 04_compute_cross_corr_sq.py first.")
print(f"  Loaded noise-free kSZ^2 x 21cm^2 : {len(cross_corr_results_sq_all)} seeds")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Run noisy cross-correlations (one noise realisation per seed)
# ─────────────────────────────────────────────────────────────────────────────
cross_corr_sq_noisy = {}

for seed in RANDOM_SEEDS:
    if seed not in kSZ_maps or seed not in lightcones:
        print(f"Seed {seed}: missing kSZ map or lightcone — skipping")
        continue

    seed_dir   = os.path.join(cache_dir, f"seed_{seed}")
    cache_path = os.path.join(seed_dir, f"cross_corr_sq_{EXP_TAG}_seed{seed}.npy")

    if os.path.exists(cache_path):
        try:
            cross_corr_sq_noisy[seed] = np.load(cache_path, allow_pickle=True).item()
            print(f"Seed {seed}: loaded from cache "
                  f"({len(cross_corr_sq_noisy[seed])} chunks)")
            continue
        except Exception:
            pass

    print(f"\nSeed {seed}: computing noisy cross-correlation ({EXP_TAG})...")
    t_seed = time.time()

    lc      = lightcones[seed]
    kSZ_map = kSZ_maps[seed].copy()
    rng     = np.random.default_rng(seed=seed + 10000)

    chi_z0_ref = float(cosmo.comoving_distance(9.0).value)
    theta_box  = float(BOX_LEN) / chi_z0_ref  # rad
    pix_rad    = theta_box / float(HII_DIM)

    cmb_noise = cmb_noise_map(int(HII_DIM), pix_rad, CMB_EXP, rng)
    kSZ_noisy = kSZ_map + cmb_noise

    print(f"  kSZ RMS (clean) : {np.sqrt(np.mean(kSZ_map**2)):.4e}")
    print(f"  CMB noise RMS   : {np.sqrt(np.mean(cmb_noise**2)):.4e}")
    print(f"  kSZ RMS (noisy) : {np.sqrt(np.mean(kSZ_noisy**2)):.4e}")

    kSZ2_noisy       = kSZ_noisy**2
    kSZ2_centered    = kSZ2_noisy - np.mean(kSZ2_noisy)
    fft_kSZ2_shifted = np.fft.fftshift(np.fft.fft2(kSZ2_centered))
    auto_kSZ2_ps2d   = np.abs(fft_kSZ2_shifted)**2 * pix_area / npix_side**2

    lc_redshifts = np.asarray(lc.lightcone_redshifts, dtype=np.float64)
    results_sq   = {}

    for z0 in z_chunk_centres:
        z_lo = z0 - delta_z / 2.0
        z_hi = z0 + delta_z / 2.0

        idx_chunk = np.where((lc_redshifts >= z_lo) & (lc_redshifts <= z_hi))[0]
        if len(idx_chunk) < 3:
            continue

        T21_chunk = np.asarray(lc.brightness_temp[:, :, idx_chunk], dtype=np.float64)

        noise_3d, sigma_noise, T_sys_eff = thermal_noise_cube(
            T21_chunk.shape, pix_size_Mpc, z0, TS_EXP, rng, cosmo,
            delta_nu_MHz=28.4)
        T21_noisy = T21_chunk + noise_3d

        n_los       = T21_chunk.shape[2]
        T21_fft3d   = np.fft.fftn(T21_noisy)
        kz_1d       = np.fft.fftfreq(n_los, d=pix_size_Mpc) * 2 * np.pi
        kz_3d       = kz_1d[None, None, :]
        fore_filter = (np.abs(kz_3d) > k_par_min).astype(float)
        T21_filtered = np.real(np.fft.ifftn(T21_fft3d * fore_filter))

        T21_sq_2d       = np.mean(T21_filtered**2, axis=2)
        T21_sq_centered = T21_sq_2d - np.mean(T21_sq_2d)
        fft_T21sq       = np.fft.fftshift(np.fft.fft2(T21_sq_centered))

        cross_ps2d      = (np.real(np.conj(fft_kSZ2_shifted) * fft_T21sq)
                           * pix_area / npix_side**2)
        auto_T21sq_ps2d = np.abs(fft_T21sq)**2 * pix_area / npix_side**2

        C_cross = np.zeros(len(k_centers))
        P_T21sq = np.zeros(len(k_centers))
        P_kSZ2  = np.zeros(len(k_centers))

        for j in range(len(k_centers)):
            mask  = (kgrid >= k_bins[j]) & (kgrid < k_bins[j + 1])
            n_pix = np.sum(mask)
            if n_pix > 0:
                C_cross[j] = np.mean(cross_ps2d[mask])
                P_T21sq[j] = np.mean(auto_T21sq_ps2d[mask])
                P_kSZ2[j]  = np.mean(auto_kSZ2_ps2d[mask])
            else:
                C_cross[j] = P_T21sq[j] = P_kSZ2[j] = np.nan

        chi_z0  = float(cosmo.comoving_distance(z0).value)
        ell     = k_centers * chi_z0
        D_cross = ell * (ell + 1) * C_cross / (2 * np.pi)

        results_sq[float(z0)] = {
            'z0': float(z0), 'ell': ell, 'k_centers': k_centers,
            'C_cross': C_cross, 'D_cross': D_cross,
            'P_T21sq': P_T21sq, 'P_kSZ2': P_kSZ2,
            'T_sys_K': T_sys_eff, 'sigma_noise_mK': sigma_noise,
        }
        print(f"  z0={z0:.1f}  T_sys={T_sys_eff:.1f}K  "
              f"noise_sigma={sigma_noise:.3f}mK  "
              f"peak|D|={np.nanmax(np.abs(D_cross)):.2e}")

    np.save(cache_path, results_sq)
    cross_corr_sq_noisy[seed] = results_sq
    print(f"  \u2713 Saved ({time.time()-t_seed:.1f}s) -> {cache_path}")

print(f"\n\u2713 Noisy cross-correlations done: {len(cross_corr_sq_noisy)} seeds")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Run all four SNR combinations
# ─────────────────────────────────────────────────────────────────────────────
z_keys_noisy = sorted(next(iter(cross_corr_sq_noisy.values())).keys())
z_keys_clean = sorted(next(iter(cross_corr_results_sq_all.values())).keys())

print("\n--- Building SNR estimates ---")

print("\n[A] Noise-free  |  diagonal estimator  (upper bound):")
per_z_clean_diag, tot_clean_diag = snr_diagonal(
    cross_corr_results_sq_all, z_keys_clean, f_sky)

print("\n[B] Noise-free  |  full covariance matrix:")
per_z_clean_cov, tot_clean_cov = snr_full_cov(
    cross_corr_results_sq_all, z_keys_clean, f_sky)

print("\n[C] With CMB-S4 x SKA noise  |  diagonal estimator:")
per_z_noisy_diag, tot_noisy_diag = snr_diagonal(
    cross_corr_sq_noisy, z_keys_noisy, f_sky)

print("\n[D] With CMB-S4 x SKA noise  |  full covariance matrix:")
per_z_noisy_cov, tot_noisy_cov = snr_full_cov(
    cross_corr_sq_noisy, z_keys_noisy, f_sky)

print(f"\n{'='*65}")
print(f"  kSZ^2 x 21cm^2  SNR summary  --  CMB-S4 x SKA1-Low")
print(f"  {'-'*55}")
print(f"  [A] Noise-free   + diagonal     (upper bound)  {tot_clean_diag:>8.2f}\u03c3")
print(f"  [B] Noise-free   + full Cov                    {tot_clean_cov:>8.2f}\u03c3")
print(f"  [C] With noise   + diagonal                    {tot_noisy_diag:>8.2f}\u03c3")
print(f"  [D] With noise   + full Cov     (best estimate){tot_noisy_cov:>8.2f}\u03c3")
print(f"{'='*65}")
print(f"  f_sky={f_sky:.3f} ({f_sky*41253:.0f} deg\u00b2)  |  "
      f"{len(cross_corr_sq_noisy)} seeds  |  T_sys: frequency-dependent (Zhou+25)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Plots
# ─────────────────────────────────────────────────────────────────────────────
plot_dir_snr = os.path.join(plot_dir, "snr_rigorous_v2")
os.makedirs(plot_dir_snr, exist_ok=True)

ref_lc    = lightcones[next(iter(lightcones))]
z_nodes_s = ref_lc.node_redshifts[::-1]
xe_nodes  = 1.0 - ref_lc.global_xH[::-1]
xe_marks  = [0.10, 0.18, 0.31, 0.51, 0.77]
z_marks   = [float(np.interp(xe, xe_nodes[::-1], z_nodes_s[::-1])) for xe in xe_marks]


def add_xe_axis(ax):
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(z_marks)
    ax2.set_xticklabels([f'{xe:.2f}' for xe in xe_marks])
    ax2.set_xlabel(r'$\bar{x}_{\rm HII}$')
    return ax2


# ── Plot 1: SNR(z0) — all four combinations ──────────────────────────────────
_styles_snr = [
    (per_z_clean_diag, 'k--', 'o', f'Noise-free + diagonal ({tot_clean_diag:.1f}\u03c3)'),
    (per_z_clean_cov,  'k-',  's', f'Noise-free + full Cov ({tot_clean_cov:.1f}\u03c3)'),
    (per_z_noisy_diag, 'r--', 'o', f'With noise + diagonal ({tot_noisy_diag:.1f}\u03c3)'),
    (per_z_noisy_cov,  'r-',  's', f'With noise + full Cov ({tot_noisy_cov:.1f}\u03c3)'),
]


def _build_snr_z():
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for per_z, ls, mk, lbl in _styles_snr:
        zz  = sorted(per_z.keys())
        snr = [per_z[z]['snr_total'] for z in zz]
        ax.plot(zz, snr, ls, marker=mk, markersize=6, label=lbl)
    ax.axhline(1, color='gray', ls='--', lw=1)
    ax.axhline(5, color='gray', ls=':', lw=1)
    ax.invert_xaxis()
    ax.set_xlabel(r'Redshift $z_0$')
    ax.set_ylabel(r'SNR($z_0$) [$\ell$-integrated]')
    ax.set_title('kSZ$^2\\times$21cm$^2$ SNR forecast: diagonal vs full covariance',
                 fontweight='bold')
    ax.legend(fontsize=10)
    add_xe_axis(ax)
    return fig


save_fig_both(_build_snr_z, plot_dir_snr, "snr_vs_z_diag_vs_cov", figsize=(11, 6))
print("  \u2713 Plot 1: snr_vs_z_diag_vs_cov.{pdf,png}")

# ── Plots 5 & 6: D_ell vs ell / vs z, noise-free vs noisy, vs Zhou+25 ────────
# Adhoc rescaling: our pipeline stores C_ell in internal FFT units, not
# physical muK^4 (see overlay_zhou25.py for the full unit discussion). Here
# we anchor the noise-free D_ell(z, ell=800) peak to Zhou+25 Fig.5's ell=800
# peak, and apply that SAME scale factor to noise-free and noisy curves
# alike so the noise/signal ratio is preserved exactly.

ZHOU25_ANCHOR, ZHOU25_Z_ANCHOR = zhou5_anchor()
print(f"  Zhou+25 anchor : Fig.5 ell=800 peak = {ZHOU25_ANCHOR:.2f} \u03bcK\u2074 "
      f"at z={ZHOU25_Z_ANCHOR:.2f}")


def _collect_D_vs_z(results_all, ell_targets):
    z_list = sorted(next(iter(results_all.values())).keys())
    out = {}
    for ell_t in ell_targets:
        D_means, D_stds, z_out = [], [], []
        for z0 in z_list:
            vals, ell_ref = [], None
            for seed, ccr in results_all.items():
                if z0 not in ccr:
                    continue
                res = ccr[z0]
                if ell_ref is None:
                    ell_ref = res['ell']
                idx = int(np.argmin(np.abs(ell_ref - ell_t)))
                v = res['D_cross'][idx]
                if np.isfinite(v):
                    vals.append(v)
            if len(vals) > 0:
                D_means.append(float(np.mean(vals)))
                D_stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
                z_out.append(float(z0))
        out[ell_t] = (np.array(z_out), np.array(D_means), np.array(D_stds))
    return out


def _collect_D_vs_ell(results_all, z_target):
    D_list, ell_ref = [], None
    for seed, ccr in results_all.items():
        z_closest = min(ccr.keys(), key=lambda z: abs(z - z_target))
        if abs(z_closest - z_target) > 0.6:
            continue
        res = ccr[z_closest]
        if ell_ref is None:
            ell_ref = res['ell'].copy()
        v = res['D_cross'].copy()
        v[~np.isfinite(v)] = np.nan
        D_list.append(v)
    if not D_list:
        return None, None, None
    D_mat = np.array(D_list)
    return ell_ref, np.nanmean(D_mat, axis=0), np.nanstd(D_mat, ddof=1, axis=0)


ELL_TARGETS = [400, 800, 1600, 3200]
ELL_COLORS  = ['steelblue', 'darkorange', 'forestgreen', 'firebrick']
Z_PEAK      = 10.0

_ell_c, _Dc, _ = _collect_D_vs_ell(cross_corr_results_sq_all, ZHOU25_Z_ANCHOR)
if _ell_c is not None:
    _idx800  = int(np.argmin(np.abs(_ell_c - 800)))
    _our_val = float(_Dc[_idx800])
    ADHOC_SCALE = ZHOU25_ANCHOR / _our_val if (np.isfinite(_our_val) and _our_val != 0) else 1.0
else:
    _our_val = np.nan
    ADHOC_SCALE = 1.0

print(f"  Our D_ell(z={ZHOU25_Z_ANCHOR:.1f}, ell=800) = {_our_val:.4e} (internal units)")
print(f"  Adhoc scale factor = {ADHOC_SCALE:.4e}")

_dvz_clean = _collect_D_vs_z(cross_corr_results_sq_all, ELL_TARGETS)
_dvz_noisy = _collect_D_vs_z(cross_corr_sq_noisy, ELL_TARGETS)
_ell_c, _Dc_mean, _Dc_std = _collect_D_vs_ell(cross_corr_results_sq_all, Z_PEAK)
_ell_n, _Dn_mean, _Dn_std = _collect_D_vs_ell(cross_corr_sq_noisy, Z_PEAK)


def _build_D_vs_ell():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for ax, show_zhou in zip(axes, [False, True]):
        ax.axhline(0, color='k', ls='--', lw=1, alpha=0.4)
        if _ell_c is not None:
            mask = np.isfinite(_Dc_mean) & (_ell_c > 80)
            D_sc, E_sc = _Dc_mean[mask] * ADHOC_SCALE, _Dc_std[mask] * ADHOC_SCALE
            ax.plot(_ell_c[mask], D_sc, 'k-', lw=2.5,
                    label=rf'Noise-free (this work, $z_0={Z_PEAK:.0f}$)')
            ax.fill_between(_ell_c[mask], D_sc - E_sc, D_sc + E_sc, color='gray', alpha=0.25)
        if _ell_n is not None:
            mask = np.isfinite(_Dn_mean) & (_ell_n > 80)
            D_sn, E_sn = _Dn_mean[mask] * ADHOC_SCALE, _Dn_std[mask] * ADHOC_SCALE
            ax.plot(_ell_n[mask], D_sn, 'r-', lw=2, label=r'With CMB-S4$\times$SKA noise')
            ax.fill_between(_ell_n[mask], D_sn - E_sn, D_sn + E_sn, color='red', alpha=0.15)
        if show_zhou:
            mask3 = ZHOU_FIG3_ell < 1.2e4
            ax.plot(ZHOU_FIG3_ell[mask3], ZHOU_FIG3_D[mask3], 'b--', lw=1.8, alpha=0.8,
                    label=r'Zhou+25 Fig.3 (SO$\times$HERA, $z_{\rm mid}{\sim}9$)')
            mask4 = ZHOU_FIG4_ell < 1.2e4
            ax.plot(ZHOU_FIG4_ell[mask4], ZHOU_FIG4_D[mask4], 'g:', lw=1.8, alpha=0.8,
                    label=r'Zhou+25 Fig.4 (SO$\times$SKA, $z_{\rm mid}{\sim}9$)')
        ax.set_xscale('log')
        ax.set_xlabel(r'Multipole $\ell$')
        ax.set_ylabel(r'$\ell(\ell+1)C_\ell^{{\rm kSZ}^2\times{\rm 21cm}^2}/2\pi\ [\mu{\rm K}^4]$'
                      '\n(noise-free rescaled to Zhou+25 amplitude)')
        ax.set_title('This work only' if not show_zhou else 'With Zhou+25 Fig.3/4 overlay',
                     fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.9)
        ax.text(0.97, 0.03, f'Adhoc scale = {ADHOC_SCALE:.2e}\n'
                r'(our noise-free $\ell$=800 peak $\to$ Zhou+25 Fig.5)',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='gray',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    fig.suptitle(r'kSZ$^2\times$21cm$^2$: $D_\ell$ vs $\ell$ at $z_0=10$'
                 r'  —  CMB-S4$\times$SKA1-Low noise', fontweight='bold')
    return fig


save_fig_both(_build_D_vs_ell, plot_dir_snr, "Dell_vs_ell_z10_noisy", figsize=(16, 6))


def _build_D_vs_z():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for ax, show_zhou in zip(axes, [False, True]):
        ax.axhline(0, color='k', ls='--', lw=1, alpha=0.4)
        for ell_t, col in zip(ELL_TARGETS, ELL_COLORS):
            zz_c, Dm_c, Ds_c = _dvz_clean[ell_t]
            if len(zz_c) > 0:
                D_sc, E_sc = Dm_c * ADHOC_SCALE, Ds_c * ADHOC_SCALE
                ax.plot(zz_c, D_sc, color=col, ls='-', lw=2.5, marker='o', markersize=6,
                        label=rf'$\ell={ell_t}$ noise-free')
                ax.fill_between(zz_c, D_sc - E_sc, D_sc + E_sc, color=col, alpha=0.12)
            zz_n, Dm_n, Ds_n = _dvz_noisy[ell_t]
            if len(zz_n) > 0:
                D_sn = Dm_n * ADHOC_SCALE
                ax.plot(zz_n, D_sn, color=col, ls='--', lw=1.8, marker='s', markersize=5,
                        alpha=0.85, label=rf'$\ell={ell_t}$ with noise')
            if show_zhou:
                dat = ZHOU_FIG5[ell_t]
                ax.plot(dat[:, 0], dat[:, 1], color=col, ls=':', lw=1.5, marker='^',
                        markersize=4, alpha=0.65)
        ax.invert_xaxis()
        ax.set_xlabel(r'Redshift $z_0$')
        ax.set_ylabel(r'$\ell(\ell+1)C_\ell^{{\rm kSZ}^2\times{\rm 21cm}^2}/2\pi\ [\mu{\rm K}^4]$'
                      '\n(noise-free rescaled to Zhou+25 amplitude)')
        ax.set_title('This work only' if not show_zhou else
                     r'With Zhou+25 Fig.5 (SO$\times$HERA, dotted markers)', fontweight='bold')
        add_xe_axis(ax)
        handles, labels = ax.get_legend_handles_labels()
        nf_h = handles[0::2]
        nf_l = [rf'$\ell={e}$' for e in ELL_TARGETS]
        leg = ax.legend(nf_h, nf_l, fontsize=10, framealpha=0.9,
                        title='Solid=noise-free\nDashed=with noise\n'
                              + (r'Dotted=Zhou+25 Fig.5' if show_zhou else ''))
        ax.add_artist(leg)
        ax.text(0.97, 0.03, f'Adhoc scale = {ADHOC_SCALE:.2e}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='gray',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    fig.suptitle(r'kSZ$^2\times$21cm$^2$: $D_\ell$ vs $z_0$ at fixed $\ell$'
                 '\n' r'CMB-S4$\times$SKA1-Low  |  solid=noise-free, dashed=with noise',
                 fontweight='bold')
    return fig


save_fig_both(_build_D_vs_z, plot_dir_snr, "Dell_vs_z_fixed_ell_noisy", figsize=(16, 6))
print("  \u2713 Plots 5 & 6: Dell_vs_ell_z10_noisy / Dell_vs_z_fixed_ell_noisy")

# ── Plot 7: empirical correlation matrices, noise-free vs noisy ─────────────

def _build_corr_matrices(results_clean, results_noisy, z0, ell_min=100, ell_max=5000):
    def _get_cov(results_all):
        seed_vecs, ell_ref = [], None
        for seed, ccr in results_all.items():
            if z0 not in ccr:
                continue
            res = ccr[z0]
            if ell_ref is None:
                ell_ref = res['ell'].copy()
            v = res['C_cross'].copy()
            if np.all(~np.isfinite(v)):
                continue
            seed_vecs.append(v)
        if ell_ref is None or len(seed_vecs) < 4:
            return None, None
        X     = np.array(seed_vecs)
        valid = np.isfinite(ell_ref) & (ell_ref >= ell_min) & (ell_ref <= ell_max)
        X_v   = X[:, valid]
        ell_v = ell_ref[valid]
        col_nan = ~np.all(np.isfinite(X_v), axis=0)
        X_v[:, col_nan] = 0.0
        s   = np.mean(X_v, axis=0)
        dX  = X_v - s[None, :]
        Cov = (dX.T @ dX) / (len(seed_vecs) - 1)
        diag = np.sqrt(np.abs(np.diag(Cov)))
        diag = np.where(diag > 0, diag, 1.0)
        R = np.clip(Cov / np.outer(diag, diag), -1.0, 1.0)
        return R, ell_v

    R_c, ell_c = _get_cov(results_clean)
    R_n, ell_n = _get_cov(results_noisy)
    return R_c, R_n, ell_c, ell_n


Z_CORR = 10.0
R_clean, R_noisy, ell_c, ell_n = _build_corr_matrices(
    cross_corr_results_sq_all, cross_corr_sq_noisy, Z_CORR)
if R_clean is None:
    Z_CORR = 9.0
    R_clean, R_noisy, ell_c, ell_n = _build_corr_matrices(
        cross_corr_results_sq_all, cross_corr_sq_noisy, Z_CORR)


def _ell_tick_labels(ell_arr, n_ticks=6):
    idx = np.round(np.linspace(0, len(ell_arr) - 1, n_ticks)).astype(int)
    return idx, [f'{ell_arr[i]:.0f}' for i in idx]


def _build_corr_plot():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    titles = [
        f'Noise-free  ($z_0={Z_CORR:.0f}$)\n'
        r'$r_{ij}=C_{ij}/\sqrt{C_{ii}C_{jj}}$  from seed scatter',
        f'With CMB-S4$\\times$SKA noise  ($z_0={Z_CORR:.0f}$)\n'
        r'Noise baked into fields before squaring',
    ]
    for ax, R, ell_v, title in zip(axes, [R_clean, R_noisy], [ell_c, ell_n], titles):
        if R is None:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center', va='center')
            ax.set_title(title, fontweight='bold')
            continue
        im = ax.imshow(R, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1,
                       interpolation='nearest')
        plt.colorbar(im, ax=ax, label=r'$r_{ij}$', fraction=0.046, pad=0.04)
        n = len(ell_v)
        tick_idx, tick_lbl = _ell_tick_labels(ell_v, n_ticks=6)
        ax.set_xticks(tick_idx); ax.set_xticklabels(tick_lbl, rotation=45, ha='right')
        ax.set_yticks(tick_idx); ax.set_yticklabels(tick_lbl)
        ax.set_xlabel(r'$\ell_j$'); ax.set_ylabel(r'$\ell_i$')
        ax.set_title(title, fontweight='bold')
        ax.plot([0, n - 1], [0, n - 1], 'k-', lw=0.8, alpha=0.4)
        offdiag_frac = 1.0 - np.sum(np.diag(R)**2) / np.sum(R**2)
        ax.text(0.98, 0.02, f'Off-diag power: {offdiag_frac*100:.1f}%\n({n} \u2113 bins)',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='navy',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'))
    fig.suptitle(r'Empirical correlation matrix $r_{ij}$ of $C_\ell^{{\rm kSZ}^2\times{\rm 21cm}^2}$'
                 '\n'
                 r'Off-diagonal structure $\Rightarrow$ diagonal SNR estimator is over-confident',
                 fontweight='bold')
    return fig


save_fig_both(_build_corr_plot, plot_dir_snr, "correlation_matrix_clean_vs_noisy", figsize=(14, 6))
print(f"  \u2713 Plot 7: correlation_matrix_clean_vs_noisy.{{pdf,png}}")
print(f"\nDone. Plots -> {plot_dir_snr}/")
