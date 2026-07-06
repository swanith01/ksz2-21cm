# =============================================================================
# [SUPERSEDED — kept for reference only, not used by the current pipeline]
#
# This was an intermediate SNR script: adds a simple NSR-based noise
# degradation on top of the noise-free correlation-coefficient estimator,
# but still uses a diagonal Gaussian approximation and a FIXED T_sys
# (not the frequency-dependent Zhou+25 model).
#
# Superseded by SNR_rigorous_v2.py, which is now:
#   - reusable functions -> src/ksz2_21cm/noise/snr_forecast.py
#   - driver + plots     -> scripts/05_compute_snr_forecast.py
#
# Use the module/script above for anything going in the paper. This file is
# kept only so the reasoning trail (why the noise model evolved) isn't lost.
# =============================================================================

# =============================================================================
# CELL SNR WITH INSTRUMENT NOISE
# ================================
# Computes realistic SNR for kSZ²×21cm and kSZ²×21cm² using the
# correlation coefficient approach with instrument noise added.
#
# Since our pipeline stores power spectra in internal Mpc² units (not
# physical angular units), we cannot add noise in absolute physical units
# directly. Instead we express noise as a noise-to-signal ratio (NSR):
#
#   NSR_CMB(ℓ)   = N_ℓ^CMB  / P_kSZ2(ℓ)    [CMB noise / kSZ² signal]
#   NSR_21cm(ℓ)  = N_ℓ^21cm / P_T21(ℓ)      [21cm noise / 21cm² signal]
#
# The noisy SNR formula is then:
#
#   (S/N)²_ℓ = f_sky (2ℓ+1) Δℓ × r²/(1-r²) × 1/[(1+NSR_CMB)(1+NSR_21cm)]
#
# This is unit-free and correct. The noise terms degrade the SNR:
#   NSR → 0 : noise-free limit (what we had before)
#   NSR → ∞ : completely noise-dominated, SNR → 0
#
# CMB noise model:
#   N_ℓ^CMB = (σ_noise [µK·rad])² exp(ℓ(ℓ+1) σ²_beam)
#   expressed as NSR by dividing by D_kSZ2 * 2π/[ℓ(ℓ+1)] converted to µK²
#
# 21cm noise model (Parsons+14 / Mao+13):
#   P_noise(k) ∝ T²_sys / (t_obs × N_bl × Δν × A_eff)
#   expressed as NSR by dividing by P_T21sq in internal units
#
# Experiment presets:
#   CMB : SO   (6.0 µK·arcmin, 1.4' beam)
#         S4   (1.0 µK·arcmin, 1.0' beam)
#         HD   (0.5 µK·arcmin, 0.25' beam)
#   21cm: HERA (440K, 331 ant, 14m, 200hr)
#         SKA  (280K, 512 ant, 35m, 1000hr)
# =============================================================================

print("\n" + "="*70)
print("CELL SNR WITH INSTRUMENT NOISE")
print("="*70)

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

cosmo    = FlatLambdaCDM(H0=67.77, Om0=0.3086)
T_CMB_uK = 2.725e6    # µK
f_sky    = 0.024       # 1000 deg² overlap (Zhou+25)
BOX_LEN  = 800.0       # Mpc
HII_DIM  = 128

# ─────────────────────────────────────────────────────────────────────────────
# 0. Load caches
# ─────────────────────────────────────────────────────────────────────────────
def _load(pattern, seeds, cache_dir):
    d = {}
    for seed in seeds:
        p = os.path.join(cache_dir, f"seed_{seed}",
                         pattern.format(seed=seed))
        if os.path.exists(p):
            try:
                d[seed] = np.load(p, allow_pickle=True).item()
            except Exception:
                pass
    return d

if 'cross_corr_results_all' not in dir() or len(cross_corr_results_all) == 0:
    cross_corr_results_all = _load(
        'cross_corr_seed{seed}.npy', RANDOM_SEEDS, main_cache_dir)
    print(f"Loaded kSZ²×21cm  : {len(cross_corr_results_all)} seeds")

if 'cross_corr_results_sq_all' not in dir() or len(cross_corr_results_sq_all) == 0:
    cross_corr_results_sq_all = _load(
        'cross_corr_sq_seed{seed}.npy', RANDOM_SEEDS, main_cache_dir)
    print(f"Loaded kSZ²×21cm² : {len(cross_corr_results_sq_all)} seeds")

assert len(cross_corr_results_all) > 0,    "Run Cell 7 first."
assert len(cross_corr_results_sq_all) > 0, "Run Cell 7b first."

ref_seed_sq = next(iter(cross_corr_results_sq_all))
ref_seed_un = next(iter(cross_corr_results_all))
ref_lc      = lightcones[ref_seed_sq]
z_nodes_s   = ref_lc.node_redshifts[::-1]
xe_nodes    = 1.0 - ref_lc.global_xH[::-1]

# ─────────────────────────────────────────────────────────────────────────────
# 1. Experiment presets
# ─────────────────────────────────────────────────────────────────────────────
CMB_PRESETS = {
    'SO'   : dict(sigma_uK_arcmin=6.0,  fwhm_arcmin=1.4,  label='SO'),
    'CMBS4': dict(sigma_uK_arcmin=1.0,  fwhm_arcmin=1.0,  label='CMB-S4'),
    'CMBHD': dict(sigma_uK_arcmin=0.5,  fwhm_arcmin=0.25, label='CMB-HD'),
}

TS_PRESETS = {
    'HERA': dict(T_sys_K=440.0, t_obs_hr=200.0,  N_bl=331,
                 A_eff_m2=154.0,  label='HERA'),
    'SKA' : dict(T_sys_K=280.0, t_obs_hr=1000.0, N_bl=512,
                 A_eff_m2=962.0,  label='SKA1-Low'),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Noise-to-signal ratio functions
# ─────────────────────────────────────────────────────────────────────────────

def cmb_nsr(ell, P_kSZ2_internal, z0, cmb_preset):
    """
    CMB noise-to-signal ratio: N_ell^CMB / P_kSZ2.

    N_ell^CMB is in µK² (angular), P_kSZ2 is in internal Mpc² units.
    We convert P_kSZ2 to µK² angular using:
      D_kSZ2_uK2 = ell(ell+1)/2pi * P_kSZ2 * (0.67/D_A)^2 * T_CMB_uK^2
    then NSR = N_ell^CMB / C_kSZ2_uK2
    where C_kSZ2_uK2 = D_kSZ2_uK2 * 2pi / (ell*(ell+1))
    """
    cp = CMB_PRESETS[cmb_preset]
    arcmin2rad   = np.pi / 180.0 / 60.0
    sigma_rad    = cp['sigma_uK_arcmin'] * arcmin2rad
    sigma_beam   = cp['fwhm_arcmin'] * arcmin2rad / np.sqrt(8.0 * np.log(2.0))

    # CMB noise power spectrum in µK²
    N_ell_cmb = sigma_rad**2 * np.exp(ell * (ell + 1.0) * sigma_beam**2)

    # Convert P_kSZ2 from internal units to µK² angular C_ell
    D_A = float(cosmo.angular_diameter_distance(z0).value)
    # P_kSZ2 [Mpc²] → C_ell [sr] via * (0.67/D_A)^2, then * T_CMB^2
    C_kSZ2_uK2 = P_kSZ2_internal * (0.67 / D_A)**2 * T_CMB_uK**2

    # Protect against zeros
    nsr = np.where(C_kSZ2_uK2 > 0, N_ell_cmb / C_kSZ2_uK2, np.inf)
    return nsr


def ksz21_nsr(ell, P_T21_internal, z0, ts_preset, delta_nu_MHz=28.4):
    """
    21cm thermal noise-to-signal ratio: N_ell^21cm / P_T21sq.

    Uses simplified Limber-projected thermal noise:
      N_ell^21cm ~ T_sys^2 * Omega_beam / (N_bl * t_obs * delta_nu) * delta_chi / chi^2
    in mK^2, then compare to P_T21 which is in mK^2 * Mpc^2 / chi^2 effectively.

    We express everything as a ratio so units cancel.
    """
    tp = TS_PRESETS[ts_preset]

    c_m_s    = 2.998e8
    nu_21    = 1420.4e6
    nu_obs   = nu_21 / (1.0 + z0)
    lam_m    = c_m_s / nu_obs
    D_ant    = np.sqrt(tp['A_eff_m2'])
    Omega_p  = (lam_m / D_ant)**2    # primary beam sr

    chi_Mpc      = float(cosmo.comoving_distance(z0).value)
    H_z          = float(cosmo.H(z0).value)
    delta_chi    = 2.998e5 * delta_nu_MHz * (1.0 + z0)**2 / (nu_21 * 1e-6 * H_z)

    t_obs_s      = tp['t_obs_hr'] * 3600.0
    delta_nu_Hz  = delta_nu_MHz * 1e6

    # Noise temperature^2 per mode in K^2 sr
    T_noise_K2   = (tp['T_sys_K']**2 * Omega_p
                    / (tp['N_bl'] * t_obs_s * delta_nu_Hz))
    T_noise_mK2  = T_noise_K2 * 1e6   # mK^2 sr

    # Projected noise C_ell^21cm [mK^2]
    N_ell_21cm = T_noise_mK2 * delta_chi / chi_Mpc**2

    # Convert P_T21sq from internal to mK^2 angular units for comparison
    # P_T21sq [mK^2 * Mpc^2] → C_ell [mK^2 sr] via / chi^2
    C_T21_mK2 = P_T21_internal / chi_Mpc**2

    nsr = np.where(C_T21_mK2 > 0, N_ell_21cm / C_T21_mK2, np.inf)
    return nsr


# ─────────────────────────────────────────────────────────────────────────────
# 3. Core SNR engine
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr_noisy(results_all, z_keys,
                      sig_key, auto_ksz_key, auto_21_key,
                      cmb_preset, ts_preset,
                      ell_key='k_centers', chi_func=None,
                      ell_min=100, ell_max=5000,
                      delta_nu_MHz=28.4):
    """
    Compute noisy SNR using correlation coefficient with instrument noise.

    (S/N)²_ℓ = f_sky(2ℓ+1)Δℓ × r²/(1-r²) / [(1+NSR_CMB)(1+NSR_21cm)]
    """
    per_z      = {}
    snr_sq_tot = 0.0

    for z0 in z_keys:
        sigs, ksz2s, t21s, ell_ref = [], [], [], None

        for seed, ccr in results_all.items():
            if z0 not in ccr:
                continue
            res = ccr[z0]
            k   = res[ell_key]
            ell_here = (k * float(cosmo.comoving_distance(z0).value) / 0.67
                        if chi_func is not None
                        else res.get('ell', k))
            if ell_ref is None:
                ell_ref = ell_here
            v_sig = res[sig_key]
            v_k2  = res[auto_ksz_key]
            v_t21 = res[auto_21_key]
            if np.all(~np.isfinite(v_sig)):
                continue
            sigs.append(v_sig)
            ksz2s.append(v_k2)
            t21s.append(v_t21)

        if ell_ref is None or len(sigs) == 0:
            continue

        C_sig  = np.nanmean(np.array(sigs),  axis=0)
        C_kSZ2 = np.nanmean(np.array(ksz2s), axis=0)
        C_21cm = np.nanmean(np.array(t21s),  axis=0)
        ell    = ell_ref

        # Correlation coefficient
        denom_r = np.sqrt(np.abs(C_kSZ2) * np.abs(C_21cm))
        r       = np.where(denom_r > 0, C_sig / denom_r, 0.0)
        r       = np.clip(r, -1.0 + 1e-10, 1.0 - 1e-10)

        # Noise-to-signal ratios
        nsr_cmb  = cmb_nsr(ell,  C_kSZ2, z0, cmb_preset)
        nsr_21cm = ksz21_nsr(ell, C_21cm, z0, ts_preset,
                              delta_nu_MHz=delta_nu_MHz)

        valid = (np.isfinite(r) & np.isfinite(ell) &
                 np.isfinite(nsr_cmb) & np.isfinite(nsr_21cm) &
                 (ell >= ell_min) & (ell <= ell_max))

        snr_sq_bin = np.zeros_like(ell)
        if np.any(valid):
            dell  = np.gradient(ell)
            r2    = r[valid]**2
            noise_penalty = ((1.0 + nsr_cmb[valid])
                             * (1.0 + nsr_21cm[valid]))
            snr_sq_bin[valid] = np.maximum(
                f_sky * (2.0 * ell[valid] + 1.0) * dell[valid]
                * r2 / (1.0 - r2) / noise_penalty,
                0.0)

        snr_per_bin = np.sqrt(snr_sq_bin)
        snr_cumul   = np.sqrt(np.cumsum(snr_sq_bin))
        snr_z0      = float(snr_cumul[-1]) if len(snr_cumul) else 0.0
        snr_sq_tot += snr_z0**2

        per_z[z0] = dict(ell=ell, r=r,
                         nsr_cmb=nsr_cmb, nsr_21cm=nsr_21cm,
                         snr_per_bin=snr_per_bin,
                         snr_cumul=snr_cumul,
                         snr_total=snr_z0)

    return per_z, np.sqrt(snr_sq_tot)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Run all combinations
# ─────────────────────────────────────────────────────────────────────────────
z_keys_un = sorted(cross_corr_results_all[ref_seed_un].keys())
z_keys_sq = sorted(cross_corr_results_sq_all[ref_seed_sq].keys())

def chi_over_h(z):
    return float(cosmo.comoving_distance(z).value) / 0.67

combinations = [
    ('SO',    'HERA'),
    ('CMBS4', 'HERA'),
    ('CMBS4', 'SKA'),
    ('CMBHD', 'SKA'),
]

forecasts_noisy = {}

print(f"\n{'Combination':<22} {'kSZ²×21cm':>14} {'kSZ²×21cm²':>14}")
print("-" * 52)

for cmb_p, ts_p in combinations:
    key = f"{cmb_p}×{ts_p}"

    per_z_un, tot_un = compute_snr_noisy(
        cross_corr_results_all, z_keys_un,
        sig_key='C_cross_1d', auto_ksz_key='P_kSZ2_1d',
        auto_21_key='P_T21_1d',
        cmb_preset=cmb_p, ts_preset=ts_p,
        ell_key='k_centers', chi_func=chi_over_h,
        delta_nu_MHz=4.0)

    per_z_sq, tot_sq = compute_snr_noisy(
        cross_corr_results_sq_all, z_keys_sq,
        sig_key='C_cross', auto_ksz_key='P_kSZ2',
        auto_21_key='P_T21sq',
        cmb_preset=cmb_p, ts_preset=ts_p,
        ell_key='k_centers', chi_func=None,
        delta_nu_MHz=28.4)

    forecasts_noisy[key] = {'un': (per_z_un, tot_un),
                             'sq': (per_z_sq, tot_sq)}

    flag = lambda s: "✓" if s >= 5 else ("~" if s >= 1 else "✗")
    print(f"  {key:<20}  {tot_un:>8.2f}σ {flag(tot_un)}   "
          f"{tot_sq:>8.2f}σ {flag(tot_sq)}")

print("-" * 52)
print(f"  f_sky={f_sky:.3f} ({f_sky*41253:.0f} deg²)")
print("  Noise model: CMB beam+white noise, 21cm Limber thermal noise")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Plots
# ─────────────────────────────────────────────────────────────────────────────
plot_dir_snr = os.path.join(plot_dir, "snr_with_noise")
os.makedirs(plot_dir_snr, exist_ok=True)

combo_colors = ['steelblue', 'darkorange', 'forestgreen', 'darkred']
stat_keys    = ['un', 'sq']
titles       = [r'kSZ²×21cm  (Ma+18)', r'kSZ²×21cm²  (Zhou+25)']

# ── Plot 1: SNR(z) ──────────────────────────────────────────────────────────
with mpl.rc_context(PNG_STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             constrained_layout=True, sharey=False)
    for ax, stat, title in zip(axes, stat_keys, titles):
        for (cmb_p, ts_p), color in zip(combinations, combo_colors):
            key    = f"{cmb_p}×{ts_p}"
            per_z, tot = forecasts_noisy[key][stat]
            if not per_z:
                continue
            z_arr   = sorted(per_z.keys())
            snr_arr = [per_z[z]['snr_total'] for z in z_arr]
            lbl     = f"{CMB_PRESETS[cmb_p]['label']}×{TS_PRESETS[ts_p]['label']}  ({tot:.1f}σ)"
            ax.plot(z_arr, snr_arr, 'o-', color=color,
                    lw=2, markersize=6, label=lbl)

        ax.axhline(1, color='gray', ls='--', lw=1)
        ax.axhline(5, color='gray', ls=':',  lw=1, label='5σ')

        # x_e axis
        xe_marks = [0.10, 0.18, 0.31, 0.51, 0.77]
        z_marks  = [float(np.interp(xe, xe_nodes[::-1], z_nodes_s[::-1]))
                    for xe in xe_marks]
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(z_marks)
        ax2.set_xticklabels([f'{xe:.2f}' for xe in xe_marks])
        ax2.set_xlabel(r'$\bar{x}_{\rm HII}$')

        ax.invert_xaxis()
        ax.set_xlabel(r'Redshift $z_0$')
        ax.set_ylabel(r'SNR$(z_0)$  [ℓ-integrated, with noise]')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    fig.suptitle(r'kSZ² SNR forecast with instrument noise'
                 f' ({N_SEEDS} seeds)', fontweight='bold')
    fig.savefig(f"{plot_dir_snr}/snr_vs_z_noisy.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)
print("\n  ✓ Saved: snr_vs_z_noisy.png")

# ── Plot 2: Cumulative SNR vs ell ───────────────────────────────────────────
with mpl.rc_context(PNG_STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             constrained_layout=True, sharey=False)
    for ax, stat, title in zip(axes, stat_keys, titles):
        for (cmb_p, ts_p), color in zip(combinations, combo_colors):
            key    = f"{cmb_p}×{ts_p}"
            per_z, tot = forecasts_noisy[key][stat]
            if not per_z:
                continue
            best_z = max(per_z, key=lambda z: per_z[z]['snr_total'])
            best   = per_z[best_z]
            xe_b   = float(np.interp(best_z, z_nodes_s[::-1], xe_nodes[::-1]))
            lbl    = (f"{CMB_PRESETS[cmb_p]['label']}×{TS_PRESETS[ts_p]['label']}"
                      f"  z₀={best_z:.0f}, xₑ={xe_b:.2f}  ({tot:.1f}σ)")
            ax.plot(best['ell'], best['snr_cumul'], color=color,
                    lw=2, label=lbl)

        ax.axhline(1, color='gray', ls='--', lw=1)
        ax.axhline(5, color='gray', ls=':',  lw=1, label='5σ')
        ax.set_xscale('log')
        ax.set_xlabel(r'$\ell_{\rm max}$')
        ax.set_ylabel('Cumulative SNR')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    fig.suptitle('Cumulative SNR vs ℓ_max — with instrument noise',
                 fontweight='bold')
    fig.savefig(f"{plot_dir_snr}/snr_cumul_noisy.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)
print("  ✓ Saved: snr_cumul_noisy.png")

# ── Plot 3: NSR breakdown at best z0 (CMB-S4 × SKA) ────────────────────────
with mpl.rc_context(PNG_STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5),
                             constrained_layout=True)
    for ax, stat, title in zip(axes, stat_keys, titles):
        key    = 'CMBS4×SKA'
        per_z, _ = forecasts_noisy[key][stat]
        if not per_z:
            continue
        best_z = max(per_z, key=lambda z: per_z[z]['snr_total'])
        best   = per_z[best_z]
        ell    = best['ell']
        valid  = np.isfinite(best['nsr_cmb']) & (ell >= 100)

        ax.plot(ell[valid], best['nsr_cmb'][valid],
                'b-', lw=2, label='NSR: CMB-S4')
        ax.plot(ell[valid], best['nsr_21cm'][valid],
                'g-', lw=2, label='NSR: SKA1-Low')
        ax.axhline(1, color='gray', ls='--', lw=1, label='NSR=1 (signal=noise)')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\ell$')
        ax.set_ylabel('Noise / Signal ratio')
        ax.set_title(f'{title}\nCMB-S4 × SKA, z₀={best_z:.0f}',
                     fontweight='bold')
        ax.legend(fontsize=11)

    fig.suptitle('Noise-to-signal ratio per ℓ bin',
                 fontweight='bold')
    fig.savefig(f"{plot_dir_snr}/nsr_breakdown_CMBS4_SKA.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)
print("  ✓ Saved: nsr_breakdown_CMBS4_SKA.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Summary table
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("SNR FORECAST WITH INSTRUMENT NOISE")
print(f"  f_sky={f_sky:.3f} ({f_sky*41253:.0f} deg²),  N_seeds={N_SEEDS}")
print(f"\n  {'Combination':<22}  {'kSZ²×21cm':>12}  {'kSZ²×21cm²':>12}")
print("  " + "-"*50)
for cmb_p, ts_p in combinations:
    key = f"{cmb_p}×{ts_p}"
    _, tot_un = forecasts_noisy[key]['un']
    _, tot_sq = forecasts_noisy[key]['sq']
    flag = lambda s: "✓(≥5σ)" if s >= 5 else ("~(≥1σ)" if s >= 1 else "✗(<1σ)")
    print(f"  {CMB_PRESETS[cmb_p]['label']:<8}×{TS_PRESETS[ts_p]['label']:<10}"
          f"  {tot_un:>7.2f}σ {flag(tot_un):<10}"
          f"  {tot_sq:>7.2f}σ {flag(tot_sq):<10}")
print("="*62)
print("\n  Notes:")
print("  • CMB noise: beam-convolved white noise N_ℓ = σ² exp(ℓ(ℓ+1)σ_beam²)")
print("  • 21cm noise: Limber-projected thermal noise from baseline distribution")
print("  • NSR computed in consistent internal units (unit-free ratio)")
print("  • Noise added to auto-powers in SNR denominator only")
print("  • For publication: add noise inside 3D cube before squaring (Zhou+25 §4)")
print("\n✓ CELL SNR WITH NOISE COMPLETE")
print(f"  Plots → {plot_dir_snr}/")
