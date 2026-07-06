# =============================================================================
# [SUPERSEDED — kept for reference only, not used by the current pipeline]
#
# This is CELL "SNR (CORRECTED)" from the end of the original monolithic
# kSZ_Squared_21cm_11Jun_CLUSTER.py script — the FIRST SNR forecast
# implemented (noise-free correlation-coefficient estimator, both the
# unsquared kSZ x 21cm and squared kSZ^2 x 21cm^2 statistics).
#
# Superseded by SNR_rigorous_v2.py (see
# src/ksz2_21cm/noise/snr_forecast.py + scripts/05_compute_snr_forecast.py),
# which adds frequency-dependent T_sys and a full covariance-matrix
# estimator, and focuses on the squared statistic only (the one that
# survives foreground-wedge filtering — see CELL 9 in the archived
# original_monolith_*.py in this same directory).
#
# [Flag — not fixed here] compute_snr_r()'s unsquared branch does:
#     ell_here = k * chi_func(z0) / 0.67
# where chi_func is chi_over_h, itself already dividing by 0.67:
#     def chi_over_h(z): return cosmo.comoving_distance(z).value / 0.67
# That looks like it divides by h twice for the UNSQUARED statistic's ell
# axis (a ~1/0.67 ~ 1.49x error). It does NOT affect the squared statistic
# (cross_corr_results_sq_all already carries a precomputed 'ell' that this
# function uses directly, chi_func=None). Since the unsquared statistic is
# a diagnostic only (see CELL 9) and this whole cell is superseded anyway,
# this hasn't been chased down further — flagging it here so it doesn't
# get silently copied into new code.
# =============================================================================

# =============================================================================
# CELL SNR (CORRECTED): Signal-to-Noise Ratio Forecast
#           for BOTH kSZ²×21cm  AND  kSZ²×21cm²
#
# The pipeline stores C_ℓ in internal FFT units that are inconsistent
# between the kSZ² field (dimensionless ~10⁻²⁵) and the 21cm field (mK²~10²).
# Adding physical instrument noise in µK² sr is therefore meaningless.
#
# CORRECT APPROACH: use the dimensionless correlation coefficient
#
#   r_ℓ = C_cross / sqrt(P_kSZ2 * P_21cm)
#
# which is unit-free and lies in [-1, 1].  The SNR formula becomes:
#
#   (S/N)²_ℓ = f_sky * (2ℓ+1) * Δℓ * r²_ℓ / (1 - r²_ℓ)
#
# This is the exact Gaussian estimator for a cross-correlation coefficient.
# It is equivalent to Zhou+25 Eq.19 in the signal-dominated limit but
# correctly handles the noise-dominated regime via (1 - r²).
#
# To add instrument noise properly, one would add it inside the simulation
# before computing auto-powers (as Zhou+25 do in §4).  That requires
# re-running the cross-correlation with noise-added fields — a future step.
# The current forecast is therefore a theoretical upper bound (no noise).
#
# Outputs:
#   • SNR(z0) per chunk, ℓ-integrated
#   • Cumulative SNR vs ℓ_max at best z0
#   • Correlation coefficient r(ℓ) at best z0
#   • Summary table
# =============================================================================

print("\n" + "="*70)
print("CELL SNR (CORRECTED): kSZ²×21cm AND kSZ²×21cm²")
print("  Method: correlation coefficient r_ℓ = C / sqrt(P_kSZ2 * P_21cm)")
print("="*70)

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=67.77, Om0=0.3086)

# ─────────────────────────────────────────────────────────────────────────────
# 0. Load caches
# ─────────────────────────────────────────────────────────────────────────────
def _load_cache(filename_pattern, seeds, cache_dir):
    d = {}
    for seed in seeds:
        path = os.path.join(cache_dir, f"seed_{seed}",
                            filename_pattern.format(seed=seed))
        if os.path.exists(path):
            try:
                d[seed] = np.load(path, allow_pickle=True).item()
            except Exception as e:
                print(f"  ✗ seed {seed}: {e}")
    return d

if 'cross_corr_results_all' not in dir() or len(cross_corr_results_all) == 0:
    cross_corr_results_all = _load_cache(
        'cross_corr_seed{seed}.npy', RANDOM_SEEDS, main_cache_dir)
    print(f"Loaded kSZ²×21cm  : {len(cross_corr_results_all)} seeds")

if 'cross_corr_results_sq_all' not in dir() or len(cross_corr_results_sq_all) == 0:
    cross_corr_results_sq_all = _load_cache(
        'cross_corr_sq_seed{seed}.npy', RANDOM_SEEDS, main_cache_dir)
    print(f"Loaded kSZ²×21cm² : {len(cross_corr_results_sq_all)} seeds")

assert len(cross_corr_results_all) > 0,    "Run Cell 7  first."
assert len(cross_corr_results_sq_all) > 0, "Run Cell 7b first."

# Reference for xe(z) mapping
ref_seed  = next(iter(cross_corr_results_sq_all))
ref_lc    = lightcones[ref_seed]
z_nodes_s = ref_lc.node_redshifts[::-1]
xe_nodes  = 1.0 - ref_lc.global_xH[::-1]

f_sky = 0.024   # 1000 deg² overlap

# ─────────────────────────────────────────────────────────────────────────────
# 1. Core SNR engine — works for both statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_snr_r(results_all, z_keys,
                  sig_key, auto_ksz_key, auto_21_key,
                  ell_key='k_centers', chi_func=None,
                  ell_min=100, ell_max=5000):
    """
    Compute SNR using the correlation coefficient r_ℓ.

    (S/N)²_ℓ = f_sky * (2ℓ+1) * Δℓ * r²/(1-r²)

    Parameters
    ----------
    chi_func : callable or None
        If not None, converts k → ℓ via ℓ = k * chi(z) / h (unsquared case).
        If None, uses stored 'ell' key directly (squared case).
    """
    per_z      = {}
    snr_sq_tot = 0.0

    for z0 in z_keys:
        sigs, ksz2s, t21s, ell_ref = [], [], [], None

        for seed, ccr in results_all.items():
            if z0 not in ccr:
                continue
            res  = ccr[z0]
            k    = res[ell_key]
            ell_here = (k * chi_func(z0) / 0.67
                        if chi_func is not None
                        else res.get('ell', k))
            if ell_ref is None:
                ell_ref = ell_here
            sigs.append(res[sig_key])
            ksz2s.append(res[auto_ksz_key])
            t21s.append(res[auto_21_key])

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

        valid = (np.isfinite(r) & np.isfinite(ell)
                 & (ell >= ell_min) & (ell <= ell_max))

        snr_sq_bin = np.zeros_like(ell)
        if np.any(valid):
            dell           = np.gradient(ell)
            r2             = r[valid]**2
            snr_sq_bin[valid] = np.maximum(
                f_sky * (2.0 * ell[valid] + 1.0) * dell[valid]
                * r2 / (1.0 - r2),
                0.0)

        snr_per_bin = np.sqrt(snr_sq_bin)
        snr_cumul   = np.sqrt(np.cumsum(snr_sq_bin))
        snr_z0      = float(snr_cumul[-1]) if len(snr_cumul) else 0.0
        snr_sq_tot += snr_z0**2

        per_z[z0] = dict(ell=ell, r=r,
                         C_signal=C_sig, C_kSZ2=C_kSZ2, C_21cm=C_21cm,
                         snr_per_bin=snr_per_bin,
                         snr_cumul=snr_cumul,
                         snr_total=snr_z0)

    return per_z, np.sqrt(snr_sq_tot)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Redshift key lists
# ─────────────────────────────────────────────────────────────────────────────
ref_seed_un = next(iter(cross_corr_results_all))
z_keys_un   = sorted(cross_corr_results_all[ref_seed_un].keys())

ref_seed_sq = next(iter(cross_corr_results_sq_all))
z_keys_sq   = sorted(cross_corr_results_sq_all[ref_seed_sq].keys())

def chi_over_h(z):
    return float(cosmo.comoving_distance(z).value) / 0.67

# ─────────────────────────────────────────────────────────────────────────────
# 3. Run forecasts
# ─────────────────────────────────────────────────────────────────────────────
combinations = [
    ('SO',    'HERA'),
    ('CMBS4', 'HERA'),
    ('CMBS4', 'SKA'),
    ('CMBHD', 'SKA'),
]

forecasts = {}

print(f"\n{'Combination':<22} {'kSZ²×21cm':>14} {'kSZ²×21cm²':>14}")
print("-" * 52)

for cmb_p, ts_p in combinations:
    key = f"{cmb_p}×{ts_p}"

    per_z_un, tot_un = compute_snr_r(
        cross_corr_results_all, z_keys_un,
        sig_key='C_cross_1d', auto_ksz_key='P_kSZ2_1d',
        auto_21_key='P_T21_1d',
        ell_key='k_centers', chi_func=chi_over_h)

    per_z_sq, tot_sq = compute_snr_r(
        cross_corr_results_sq_all, z_keys_sq,
        sig_key='C_cross', auto_ksz_key='P_kSZ2',
        auto_21_key='P_T21sq',
        ell_key='k_centers', chi_func=None)

    forecasts[key] = {'un': (per_z_un, tot_un),
                      'sq': (per_z_sq, tot_sq)}

    flag = lambda s: "✓" if s >= 5 else ("~" if s >= 1 else "✗")
    print(f"  {key:<20}  {tot_un:>8.2f}σ {flag(tot_un)}   "
          f"{tot_sq:>8.2f}σ {flag(tot_sq)}")

print("-" * 52)
print(f"  f_sky={f_sky:.3f} ({f_sky*41253:.0f} deg²) — no instrument noise added")
print("  (upper bound; noise-free cosmic variance only)")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Plots
# ─────────────────────────────────────────────────────────────────────────────
plot_dir_snr = os.path.join(plot_dir, "snr_forecast_corrected")
os.makedirs(plot_dir_snr, exist_ok=True)

combo_colors = ['steelblue', 'darkorange', 'forestgreen', 'darkred']
stat_keys    = ['un', 'sq']
titles       = [r'kSZ²×21cm  (Ma+18)', r'kSZ²×21cm²  (Zhou+25)']

# ── Plot 1: SNR(z0) ──────────────────────────────────────────────────────────
with mpl.rc_context(PNG_STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             constrained_layout=True, sharey=False)
    for ax, stat, title in zip(axes, stat_keys, titles):
        for (cmb_p, ts_p), color in zip(combinations, combo_colors):
            key    = f"{cmb_p}×{ts_p}"
            per_z, tot = forecasts[key][stat]
            if not per_z:
                continue
            z_arr   = sorted(per_z.keys())
            snr_arr = [per_z[z]['snr_total'] for z in z_arr]
            ax.plot(z_arr, snr_arr, 'o-', color=color, lw=2,
                    markersize=6, label=f"{key}  ({tot:.1f}σ)")

        ax.axhline(1, color='gray', ls='--', lw=1)
        ax.axhline(5, color='gray', ls=':',  lw=1, label='5σ')

        # x_e secondary axis
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
        ax.set_ylabel(r'SNR$(z_0)$  [ℓ-integrated, noise-free]')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    fig.suptitle(r'kSZ² SNR forecast — noise-free upper bound'
                 f' ({N_SEEDS} seeds)', fontweight='bold')
    fig.savefig(f"{plot_dir_snr}/snr_vs_z_corrected.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)
print("\n  ✓ Saved: snr_vs_z_corrected.png")

# ── Plot 2: Correlation coefficient r(ℓ) at best z0 ─────────────────────────
with mpl.rc_context(PNG_STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             constrained_layout=True)
    for ax, stat, title in zip(axes, stat_keys, titles):
        key    = "CMBS4×SKA"
        per_z, _ = forecasts[key][stat]
        if not per_z:
            continue
        best_z = max(per_z, key=lambda z: per_z[z]['snr_total'])
        best   = per_z[best_z]
        xe_b   = float(np.interp(best_z, z_nodes_s[::-1], xe_nodes[::-1]))
        ell    = best['ell']
        valid  = np.isfinite(best['r']) & (ell >= 100)

        ax.plot(ell[valid], best['r'][valid], 'k-', lw=2)
        ax.axhline(0,  color='gray', ls='--', lw=1)
        ax.axhline( 0.1, color='blue', ls=':', lw=1, alpha=0.5)
        ax.axhline(-0.1, color='blue', ls=':', lw=1, alpha=0.5)
        ax.set_xscale('log')
        ax.set_xlabel(r'Multipole $\ell$')
        ax.set_ylabel(r'Correlation coefficient $r_\ell$')
        ax.set_title(f'{title}\n'
                     rf'$z_0={best_z:.1f}$, $x_e={xe_b:.2f}$',
                     fontweight='bold')
        ax.set_ylim(-1, 1)

    fig.suptitle(r'Cross-correlation coefficient $r_\ell = C_\ell / \sqrt{P_{\rm kSZ^2} P_{\rm 21cm}}$',
                 fontweight='bold')
    fig.savefig(f"{plot_dir_snr}/correlation_coefficient_r_ell.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)
print("  ✓ Saved: correlation_coefficient_r_ell.png")

# ── Plot 3: Cumulative SNR vs ℓ ──────────────────────────────────────────────
with mpl.rc_context(PNG_STYLE):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             constrained_layout=True, sharey=False)
    for ax, stat, title in zip(axes, stat_keys, titles):
        for (cmb_p, ts_p), color in zip(combinations, combo_colors):
            key    = f"{cmb_p}×{ts_p}"
            per_z, tot = forecasts[key][stat]
            if not per_z:
                continue
            best_z = max(per_z, key=lambda z: per_z[z]['snr_total'])
            best   = per_z[best_z]
            xe_b   = float(np.interp(best_z, z_nodes_s[::-1],
                                     xe_nodes[::-1]))
            ax.plot(best['ell'], best['snr_cumul'], color=color, lw=2,
                    label=f"{key}  z₀={best_z:.0f}, xₑ={xe_b:.2f}"
                          f"  ({tot:.1f}σ)")

        ax.axhline(1, color='gray', ls='--', lw=1)
        ax.axhline(5, color='gray', ls=':',  lw=1, label='5σ')
        ax.set_xscale('log')
        ax.set_xlabel(r'$\ell_{\rm max}$')
        ax.set_ylabel('Cumulative SNR')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

    fig.suptitle('Cumulative SNR vs ℓ_max — noise-free upper bound',
                 fontweight='bold')
    fig.savefig(f"{plot_dir_snr}/snr_cumul_corrected.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)
print("  ✓ Saved: snr_cumul_corrected.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("TOTAL SNR FORECAST (noise-free upper bound)")
print(f"  f_sky={f_sky:.3f}  ({f_sky*41253:.0f} deg²),  N_seeds={N_SEEDS}")
print(f"\n  {'Combination':<22}  {'kSZ²×21cm':>12}  {'kSZ²×21cm²':>12}")
print("  " + "-"*50)
for cmb_p, ts_p in combinations:
    key = f"{cmb_p}×{ts_p}"
    _, tot_un = forecasts[key]['un']
    _, tot_sq = forecasts[key]['sq']
    flag = lambda s: "✓(≥5σ)" if s >= 5 else ("~(≥1σ)" if s >= 1 else "✗")
    print(f"  {key:<22}  {tot_un:>8.2f}σ {flag(tot_un):<8}  "
          f"{tot_sq:>8.2f}σ {flag(tot_sq):<8}")
print("="*60)
print("\n  NOTE: This is noise-free (cosmic variance only).")
print("  To add instrument noise, re-run Cell 7/7b with noise-added")
print("  21cm and kSZ fields before computing auto-powers.")
print("\n✓ CELL SNR CORRECTED COMPLETE")
print(f"  Plots → {plot_dir_snr}/")
