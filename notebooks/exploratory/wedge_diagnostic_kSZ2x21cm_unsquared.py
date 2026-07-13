# =============================================================================
# wedge_diagnostic_kSZ2x21cm_unsquared.py
#
# [EXTRACTED from the original monolith — diag1 (CELL 7c) + CELL 9]
#
# This is the key diagnostic that explains why the pipeline moved from the
# UNSQUARED statistic (kSZ^2 x 21cm, CELL 7 / cross_corr_worker.py) to the
# SQUARED statistic (kSZ^2 x 21cm^2, CELL 7b / cross_corr_sq_worker.py) as
# the paper's main result:
#
#   kSZ^2(ell) x T21cm(ell') requires k_par,21cm ~ 0 to conserve momentum
#   (the "triangle condition"). Any foreground wedge / high-pass filter
#   removes exactly those modes -> the unsquared signal collapses to noise.
#   Squaring the 21cm field manufactures k_par=0 from +-k_par' pairs that
#   DO survive the filter -- that's the physical reason squaring rescues
#   the statistic under realistic foreground removal.
#
# Corresponds to three uploaded reference images:
#   diag1_foreground_wedge.png              <- diag1 below (this file)
#   wedge_kills_unsquared_2panel.png        <- CELL 9's main plot (this file)
#   kSZ2_21cm_cross_Dl_vs_z_ell3000_NOERR.png  <- a DIFFERENT cell (CELL 8b),
#       not reproduced here — see the bug flags below for why its numbers
#       don't directly compare to this file's CELL 9 plot.
#
# =============================================================================
# TWO CONFIRMED BUGS IN THE ORIGINAL CODE — found while archiving this,
# grep-verified against the full original file, NOT fixed here (see below
# for why, and see cross_corr_raw_worker.py for where the fix was applied
# instead, in new code):
#
# BUG 1 — CELL 9's y-axis is mislabeled.
#   CELL 9 defines `T_CMB_uK = 2.725e6` (line ~4089 of the original file)
#   but that variable is NEVER used again anywhere in the entire codebase.
#   So despite the axis label claiming units of [muK^2 mK], the D_ell
#   values CELL 9 actually plots (including wedge_kills_unsquared_2panel.png)
#   are missing the T_CMB_uK^2 conversion factor (~7.4e12) and are still in
#   raw/internal units. The SHAPE of the plot (sign changes, wedge killing
#   the signal) is still physically meaningful; the absolute Y values and
#   axis label are not consistent with each other.
#
# BUG 2 — THREE inconsistent k -> ell conventions exist across the codebase:
#     CELL 7b (the main pipeline result, kSZ^2 x 21cm^2):
#         ell = k_centers * chi_z0                    <- no h factor
#     CELL 9 (this file, recomputes fresh rather than loading a cache):
#         ell_arr = k_centers * chi_z0                 <- no h factor (agrees with 7b)
#     CELL 8b and two other cells that consume CELL 7's CACHED k-space
#     output (the ones that made kSZ2_21cm_cross_Dl_vs_z_ell3000_NOERR.png):
#         ell_from_k = k_centers * chi_comoving_Mpc / 0.67   <- extra /0.67
#   Since py21cmfast's BOX_LEN is specified in physical Mpc (not Mpc/h),
#   the standard Limber relation ell = k*chi needs no h factor when k itself
#   comes from an FFT grid built on a physical-Mpc box (as ours is) — which
#   would make the "no h factor" convention (CELL 7b / CELL 9 / this
#   pipeline's cross_corr_sq_worker.py) the physically correct one, and the
#   /0.67 in CELL 8b and friends the error. This is a reasoned hypothesis,
#   not a proven certainty — nobody has run the actual numbers to confirm.
#   The new cross_corr_raw_worker.py (built alongside this file) uses the
#   no-h-factor convention, matching cross_corr_sq_worker.py, for
#   consistency with the rest of THIS repo's pipeline.
#
# Why these bugs are flagged, not fixed, here: this file is an archived
# excerpt of retired diagnostic code, not part of the reproducible
# pipeline. "Fixing" it without being able to run py21cmfast to verify the
# fix risks introducing a NEW, unverified error into old, already-published
# reasoning. If you want a trustworthy, physical-units version of this
# diagnostic, the right move is a clean rebuild using the current
# (bug-fixed, tested) worker modules in src/ksz2_21cm/, not a patch to
# this archived copy.
#
# Requires (not included here — this is an excerpt): lightcones, kSZ_maps,
# user_params, plot_dir, PDF_STYLE/PNG_STYLE, save_pdf_png all in scope, as
# in the original monolith. Not directly runnable standalone.
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.cosmology import FlatLambdaCDM

# =============================================================================
# diag1 — Foreground filter scenarios in (k_perp, k_par) space
# (from CELL 7c; corresponds to diag1_foreground_wedge.png)
# Genuinely standalone — only needs k_par_min_diag, no pipeline state.
# =============================================================================

k_par_min_diag = 0.01   # CHECK against your actual k_par_min if reusing this

kperp_1d = np.linspace(0, 1.5, 300)
kpar_1d  = np.linspace(0, 1.5, 300)
KPERP, KPAR = np.meshgrid(kperp_1d, kpar_1d)

diag1_scenarios = [
    {
        'label':  r'Optimistic: $k_{\parallel} > k_{\parallel,0}$'
                  + f'\n$k_{{\\parallel,0}}={k_par_min_diag}$ h/Mpc',
        'filter': (KPAR > k_par_min_diag).astype(float),
    },
    {
        'label':  r'Wedge $m=3$' + '\n'
                  + r'$k_{\parallel} > m \cdot k_\perp$',
        'filter': (KPAR > 3.0 * KPERP).astype(float),
    },
    {
        'label':  r'Wedge $m=5$' + '\n'
                  + r'$k_{\parallel} > m \cdot k_\perp$',
        'filter': (KPAR > 5.0 * KPERP).astype(float),
    },
]


def _draw_diag1(axes):
    for ax, sc in zip(axes, diag1_scenarios):
        ax.pcolormesh(kperp_1d, kpar_1d, sc['filter'],
                      cmap='RdYlGn', vmin=0, vmax=1, shading='auto')
        ax.contourf(KPERP, KPAR, sc['filter'],
                    levels=[-0.5, 0.5], colors=['#d62728'], alpha=0.35)
        ax.contourf(KPERP, KPAR, sc['filter'],
                    levels=[0.5, 1.5], colors=['#2ca02c'], alpha=0.25)

        for m_ref, ls, lbl in zip([3, 5], ['--', ':'],
                                  [r'$m=3$', r'$m=5$']):
            ax.plot(kperp_1d, m_ref * kperp_1d,
                    color='white', ls=ls, lw=1.5, alpha=0.8, label=lbl)

        if k_par_min_diag > 0:
            ax.axhline(k_par_min_diag, color='gold', lw=1.5, ls='-.',
                       label=rf'$k_{{\parallel,0}}={k_par_min_diag}$')

        ax.set_xlabel(r'$k_\perp\;[h\,\mathrm{Mpc}^{-1}]$')
        ax.set_ylabel(r'$k_\parallel\;[h\,\mathrm{Mpc}^{-1}]$')
        ax.set_xlim(0, 1.5)
        ax.set_ylim(0, 1.5)
        ax.legend(loc='upper right', framealpha=0.7)
        ax.text(0.5, 1.02, sc['label'], transform=ax.transAxes,
                ha='center', va='bottom')
        ax.text(0.05, 0.95, 'REMOVED\n(foreground)',
                transform=ax.transAxes,
                color='#d62728', fontweight='bold', va='top', ha='left')
        ax.text(0.55, 0.15, 'KEPT\n(cosmological)',
                transform=ax.transAxes,
                color='#2ca02c', fontweight='bold')


def run_diag1(plot_dir_diag):
    """Standalone-runnable: only needs an output directory."""
    import os
    os.makedirs(plot_dir_diag, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    _draw_diag1(axes)
    fig.suptitle(r"Foreground Filter Scenarios in $(k_\perp, k_\parallel)$ Space",
                fontweight="bold")
    fig.savefig(f"{plot_dir_diag}/diag1_foreground_wedge.png",
               dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {plot_dir_diag}/diag1_foreground_wedge.png")


# =============================================================================
# CELL 9 — wedge filter destroys kSZ^2 x 21cm (unsquared)
# (corresponds to wedge_kills_unsquared_2panel.png)
#
# NOT standalone-runnable — requires `lightcones`, `kSZ_maps`, `user_params`
# in scope (a real lightcone + kSZ map for at least one seed), and reruns
# the cross-correlation from scratch under four filter scenarios directly on
# the 3D lightcone chunk (rather than the already-projected-to-2D CELL 7
# worker output), since the whole point is to filter BEFORE projecting to 2D.
#
# See BUG 1 above: T_CMB_uK is defined below but never applied — the y-axis
# label says [muK^2 mK] but the values are in raw/internal units.
# =============================================================================

def run_cell9_wedge_diagnostic(lightcones, kSZ_maps, user_params, plot_dir,
                               PDF_STYLE, PNG_STYLE):
    diag_seed = list(lightcones.keys())[0]
    lc        = lightcones[diag_seed]
    kSZ_map   = kSZ_maps[diag_seed]

    npix_side    = user_params.HII_DIM
    box_size_Mpc = float(user_params.BOX_LEN)
    pix_size_Mpc = box_size_Mpc / npix_side
    pix_area     = pix_size_Mpc**2

    dk       = 2 * np.pi / (npix_side * pix_size_Mpc)
    kx_2d    = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
    ky_2d    = np.fft.fftshift(np.fft.fftfreq(npix_side)) * npix_side * dk
    kgrid_2d = np.sqrt(kx_2d[:, None]**2 + ky_2d[None, :]**2)
    k_bins    = np.logspace(np.log10(dk), np.log10(kgrid_2d.max() * 0.9), 30)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    kx_1d = np.fft.fftfreq(npix_side, d=pix_size_Mpc) * 2 * np.pi
    ky_1d = np.fft.fftfreq(npix_side, d=pix_size_Mpc) * 2 * np.pi

    cosmo = FlatLambdaCDM(H0=67.77, Om0=0.3086)

    kSZ2_map         = kSZ_map**2
    kSZ2_centered    = kSZ2_map - np.mean(kSZ2_map)
    fft_kSZ2_shifted = np.fft.fftshift(np.fft.fft2(kSZ2_centered))

    lc_redshifts = np.asarray(lc.lightcone_redshifts, dtype=np.float64)

    delta_z_thin = 0.5
    z_min_scan   = lc_redshifts[lc_redshifts > 0].min()
    z_max_scan   = min(lc_redshifts.max(), 20.0)
    z_centres    = np.arange(
        np.ceil(z_min_scan / delta_z_thin) * delta_z_thin,
        z_max_scan, delta_z_thin)

    scenarios = {
        'A_nofilter'  : {'label': '(A) No filter (Cell 7 result)',
                         'color': 'black',   'ls': '-',  'lw': 2.5},
        'B_optimistic': {'label': r'(B) $k_\parallel > 0.01$ Mpc$^{-1}$',
                         'color': 'steelblue', 'ls': '--', 'lw': 2.0},
        'C_wedge_m3'  : {'label': r'(C) Wedge $m=3$',
                         'color': 'darkorange', 'ls': '-.', 'lw': 2.0},
        'D_wedge_m5'  : {'label': r'(D) Wedge $m=5$',
                         'color': 'crimson',    'ls': ':',  'lw': 2.0},
    }

    ell_target = 3000

    z_nodes_lc  = lc.node_redshifts[::-1]
    xe_nodes_lc = 1.0 - lc.global_xH[::-1]

    def z_at_xe(xe_val):
        return float(np.interp(xe_val, xe_nodes_lc, z_nodes_lc))

    z_xe02, z_xe05, z_xe09 = z_at_xe(0.2), z_at_xe(0.5), z_at_xe(0.9)

    results_z = {k: {'z': [], 'D': []} for k in scenarios}

    # T_CMB_uK = 2.725e6   # [BUG 1 — see file header] defined but never
    # applied below in the original code. D_cross stays in raw/internal
    # units despite the axis label. Left commented here as documentation
    # of the bug, not reinstated silently.

    for key, meta in scenarios.items():
        for z0 in z_centres:
            z_lo, z_hi = z0 - delta_z_thin / 2.0, z0 + delta_z_thin / 2.0
            idx_chunk = np.where((lc_redshifts >= z_lo) & (lc_redshifts < z_hi))[0]
            if len(idx_chunk) < 2:
                continue

            T21_chunk = np.asarray(lc.brightness_temp[:, :, idx_chunk], dtype=np.float64)
            n_los = T21_chunk.shape[2]

            kz_1d    = np.fft.fftfreq(n_los, d=pix_size_Mpc) * 2 * np.pi
            kx_3d    = kx_1d[:, None, None]
            ky_3d    = ky_1d[None, :, None]
            kz_3d    = kz_1d[None, None, :]
            kperp_3d = np.sqrt(kx_3d**2 + ky_3d**2)
            kpar_3d  = np.abs(kz_3d)

            T21_fft3d = np.fft.fftn(T21_chunk)

            if key == 'A_nofilter':
                filt3d = np.ones_like(kpar_3d)
            elif key == 'B_optimistic':
                filt3d = (kpar_3d > 0.01).astype(float)
            elif key == 'C_wedge_m3':
                filt3d = (kpar_3d > 3.0 * kperp_3d).astype(float)
            elif key == 'D_wedge_m5':
                filt3d = (kpar_3d > 5.0 * kperp_3d).astype(float)

            T21_filtered = np.real(np.fft.ifftn(T21_fft3d * filt3d))
            # NOTE: mean along LoS, NOT squared — this is kSZ^2 x 21cm,
            # not kSZ^2 x 21cm^2.
            T21_2d     = np.mean(T21_filtered, axis=2)
            T21_cen    = T21_2d - np.mean(T21_2d)
            fft_T21_sh = np.fft.fftshift(np.fft.fft2(T21_cen))

            cross_ps2d = (np.real(np.conj(fft_kSZ2_shifted) * fft_T21_sh)
                         * pix_area / npix_side**2)

            C_cross = np.full(len(k_centers), np.nan)
            for j in range(len(k_centers)):
                mask = (kgrid_2d >= k_bins[j]) & (kgrid_2d < k_bins[j + 1])
                if np.sum(mask) > 0:
                    C_cross[j] = np.mean(cross_ps2d[mask])

            chi_z0  = float(cosmo.comoving_distance(z0).value)
            ell_arr = k_centers * chi_z0   # no h factor — agrees with CELL 7b
            D_cross = ell_arr * (ell_arr + 1) * C_cross / (2 * np.pi)

            idx_ell = np.argmin(np.abs(ell_arr - ell_target))
            if np.isfinite(D_cross[idx_ell]):
                results_z[key]['z'].append(z0)
                results_z[key]['D'].append(D_cross[idx_ell])

    # ── Plot: 2-panel, left = no-filter only, right = all scenarios ─────────
    with mpl.rc_context(PDF_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(20, 7),
                                 constrained_layout=True, sharey=False)

        ax = axes[0]
        z_A = np.array(results_z['A_nofilter']['z'])
        D_A = np.array(results_z['A_nofilter']['D'])
        sort = np.argsort(z_A)[::-1]
        ax.plot(z_A[sort], D_A[sort], color='black', lw=2.5, ls='-',
                marker='o', markersize=4, label='No filter (raw 21cm)')
        ax.axhline(0, color='gray', ls='--', lw=1)
        for xe_val, z_xe, color in [(0.2, z_xe02, 'blue'),
                                    (0.5, z_xe05, 'green'),
                                    (0.9, z_xe09, 'red')]:
            ax.axvline(z_xe, color=color, ls=':', lw=1.5, alpha=0.7)
        ax.set_yscale('symlog', linthresh=1e-2)
        ax.set_xlabel(r'Redshift $z$')
        ax.set_ylabel(
            rf'$\ell(\ell+1)C_\ell^{{\rm kSZ^2\times 21cm}}/2\pi$'
            rf'  (raw units — see BUG 1)  at $\ell={ell_target}$')
        ax.set_title('No filter — Ma+2018 signal reproduced', fontsize=16)
        ax.invert_xaxis()

        ax = axes[1]
        for key, meta in scenarios.items():
            z_arr = np.array(results_z[key]['z'])
            D_arr = np.array(results_z[key]['D'])
            if len(z_arr) == 0:
                continue
            sort = np.argsort(z_arr)[::-1]
            ax.plot(z_arr[sort], D_arr[sort], color=meta['color'],
                    lw=meta['lw'], ls=meta['ls'],
                    marker='o' if key == 'A_nofilter' else None,
                    markersize=3, label=meta['label'], alpha=0.9)
        ax.axhline(0, color='gray', ls='--', lw=1)
        ax.set_yscale('symlog', linthresh=1e-2)
        ax.set_xlabel(r'Redshift $z$')
        ax.set_ylabel(rf'$\ell(\ell+1)C_\ell^{{\rm kSZ^2\times 21cm}}/2\pi$'
                     rf'  at $\ell={ell_target}$')
        ax.set_title('All filter scenarios — wedge kills the signal', fontsize=16)
        ax.legend(loc='lower left', fontsize=14)
        ax.invert_xaxis()

        fig.suptitle(rf'kSZ$^2\times$21cm (unsquared): $\ell={ell_target}$, '
                    f'seed {diag_seed}', fontsize=18)
        fig.savefig(f"{plot_dir}/wedge_kills_unsquared_Dl_vs_z.pdf",
                   bbox_inches='tight')
        plt.close(fig)

    print(f"Saved: {plot_dir}/wedge_kills_unsquared_Dl_vs_z.pdf")
    return results_z
