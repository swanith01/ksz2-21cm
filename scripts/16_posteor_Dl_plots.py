#!/usr/bin/env python
# =============================================================================
# 16_posteor_Dl_plots.py   (NEW, 2026-10-16)
#
# D_ell plots, EoR-only vs EoR + post-EoR kSZ, in the SAME form as the earlier figures:
#
#   Fig 1  posteor_Dl_vs_z.{pdf,png}   kSZ^2 x 21cm^2 (k_par filtered, Zhou+25 style) vs z0
#                                      at ell = 400, 800, 1600; Zhou+25 Fig. 5 dashed on the right
#                                      axis (as 06). Blue = EoR only, orange dashed = EoR + post-EoR.
#   Fig 2  posteor_Dl_diff.{pdf,png}   the PAIRED difference (total - EoR) +/- s.e.m. at the same ells
#                                      (same seeds, same 21cm, same EoR kSZ => only the post-EoR kSZ differs)
#   Fig 3  posteor_Dl_vs_xe.{pdf,png}  kSZ x 21cm (linear; ell = 100, 500) and kSZ^2 x 21cm (21cm not
#                                      squared; ell = 500, 1000, 3000) vs x_e, Ma+18 as squares on the
#                                      right axis (as 08b). Solid = EoR only, dashed = EoR + post-EoR.
#   posteor_Dl_table.txt               the numbers (peak, paired change, ratio)
#   posteor_Dl_data.npz                everything plotted (replot without the big caches: --plot-only)
#
# HOW THE NUMBERS ARE MADE: exactly the validated workers behind the earlier figures --
#   04  (kSZ^2 x 21cm^2, k_par > k_par_min)   -> cross_corr_sq_stitched_seed<N>.npy
#   04b (kSZ x 21cm and kSZ^2 x 21cm, slab)   -> cross_corr_ma_stitched_seed<N><tag>.npy
# For "EoR only" these are the existing caches (a cache hit costs nothing; if one is missing the
# worker computes it, as 04/04b would). For "EoR + post-EoR" the SAME workers run with the kSZ map
#     kSZ_map_stitched_seed<N>.npy + kSZ_map_postEoR_seed<N>.npy
# and write their caches to <cache_dir>/posteor_total/seed_<N>/ (nothing existing is overwritten).
# Units / normalisation as in 06 (--chi-norm z0) and 08: D = ell(ell+1) C /2pi, / chi(z0)^2,
# kSZ^2 x 21cm^2 in muK^4/chi^2, kSZ x 21cm in mK muK/chi^2, kSZ^2 x 21cm in mK muK^2/chi^2.
#
# Needs the post-EoR maps from 02b (run_posteor.pbs). Moderate cost (a few minutes per seed):
# use an interactive node or PBS, not the login node.
#   python scripts/16_posteor_Dl_plots.py --config configs/variants/stitched_256.yaml
#   python scripts/16_posteor_Dl_plots.py --plot-only --data-dir data/plots/2026-10-16   # on usha
# =============================================================================
import argparse, os, sys, time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/variants/stitched_256.yaml")
p.add_argument("--seeds", type=int, nargs="+", default=None,
               help="default: every config seed that has a post-EoR map")
p.add_argument("--ells-zhou", type=float, nargs="+", default=[400, 800, 1600])
p.add_argument("--ells-lin", type=float, nargs="+", default=[100, 500])
p.add_argument("--ells-sq", type=float, nargs="+", default=[500, 1000, 3000])
p.add_argument("--variant", default="slab", choices=["slice", "narrow", "slab"])
# the 04b grid: keep the defaults so the EoR caches are hit
p.add_argument("--z0-min", type=float, default=5.5)
p.add_argument("--z0-max", type=float, default=15.0)
p.add_argument("--z0-step", type=float, default=0.25)
p.add_argument("--slab-dz", type=float, default=None)
p.add_argument("--narrow-dz", type=float, default=0.2)
p.add_argument("--ell-min-ksz", type=float, default=100.0)
p.add_argument("--chi-ksz", type=float, default=8504.0)
p.add_argument("--workers", type=int, default=None)
p.add_argument("--verify-eor", action="store_true",
               help="recompute the first seed's EoR-only result from scratch and check it equals the existing "
                    "04/04b cache (guards against a stale cache); adds a few minutes")
p.add_argument("--no-zhou", action="store_true")
p.add_argument("--no-ma", action="store_true")
p.add_argument("--plot-only", action="store_true", help="skip the computation; read <name>_data.npz")
p.add_argument("--data-dir", default=None, help="folder holding <name>_data.npz (default: plot_dir)")
p.add_argument("--name", default="posteor_Dl")
args = p.parse_args()

PLOT_DIR_CFG = None
if args.data_dir is None or not args.plot_only:
    from ksz2_21cm.utils.config import load_config, ensure_dirs
    cfg = load_config(args.config)
    PLOT_DIR_CFG = cfg["paths"]["plot_dir"]
out_dir = os.path.expanduser(args.data_dir) if args.data_dir else PLOT_DIR_CFG
data_path = os.path.join(out_dir, f"{args.name}_data.npz")


# =============================================================================
# 1. COMPUTE
# =============================================================================
def compute():
    from ksz2_21cm.utils.cosmology import get_cosmology
    from ksz2_21cm.utils.constants import T_CMB_K
    from ksz2_21cm.correlation.cross_corr_sq_worker_stitched import compute_cross_corr_sq_for_seed_stitched
    from ksz2_21cm.correlation.cross_corr_ma_worker_stitched import compute_cross_corr_ma_for_seed_stitched

    ensure_dirs(cfg)
    cache_dir = cfg["paths"]["cache_dir"]
    N = cfg["simulation"]["hii_dim"]; L = float(cfg["simulation"]["box_len"])
    assert cfg["simulation"]["hii_dim"] == cfg["coeval_ksz"]["hii_dim"]
    assert cfg["simulation"]["box_len"] == cfg["coeval_ksz"]["box_len"]
    k_par_min = cfg["cross_correlation"]["k_par_min"]
    delta_z = cfg["cross_correlation"]["delta_z"]
    z_chunks = list(cfg["cross_correlation"]["z_chunk_centres"])
    slab_dz = args.slab_dz if args.slab_dz is not None else delta_z
    cosmo = get_cosmology(cfg)

    pix = L / N; pix_area = pix ** 2
    dk = 2 * np.pi / (N * pix)
    kx = np.fft.fftshift(np.fft.fftfreq(N)) * N * dk
    kgrid = np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)
    k_bins = np.logspace(np.log10(dk), np.log10(kgrid.max() * 0.9), 35)      # same as 04 / 04b
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    k1d = np.fft.fftfreq(N, d=pix) * 2 * np.pi
    z0_ma = [float(z) for z in np.arange(args.z0_min, args.z0_max + 1e-9, args.z0_step)]
    tag = (f"_dz{slab_dz:g}_nd{args.narrow_dz:g}_lmin{args.ell_min_ksz:g}"
           f"_z{args.z0_min:g}-{args.z0_max:g}-{args.z0_step:g}")                  # as 04b / 08

    maps_dir = os.path.join(cache_dir, "kSZ_maps")
    total_cache = os.path.join(cache_dir, "posteor_total")
    seeds_try = args.seeds if args.seeds is not None else cfg["simulation"]["random_seeds"]
    used, npz_of, K = [], {}, {}
    for s in seeds_try:
        npz = os.path.join(cache_dir, f"seed_{s}", f"stitched_lightcone_seed{s}.npz")
        e_p = os.path.join(maps_dir, f"kSZ_map_stitched_seed{s}.npy")
        q_p = os.path.join(maps_dir, f"kSZ_map_postEoR_seed{s}.npy")
        if not all(os.path.exists(x) for x in (npz, e_p, q_p)):
            print(f"  x seed {s}: missing lightcone / EoR map / post-EoR map -- skipping")
            continue
        used.append(s); npz_of[s] = npz
        ke = np.load(e_p)
        K[("eor", s)] = ke
        K[("tot", s)] = ke + np.load(q_p)
    assert len(used) >= 2, "need >= 2 seeds with a post-EoR map (the Ma workers' null uses a neighbour seed)"
    print(f"seeds used ({len(used)}): {used}")

    # ---- jobs: (kind, which, seed) ------------------------------------------------------------
    jobs = []
    for i, s in enumerate(used):
        nxt = used[(i + 1) % len(used)]
        for which in ("eor", "tot"):
            cdir = cache_dir if which == "eor" else total_cache
            jobs.append((("sq", which, s), compute_cross_corr_sq_for_seed_stitched,
                         (s, npz_of[s], K[(which, s)], cdir, N, L, pix, pix_area, k1d, k1d, kgrid, k_bins,
                          k_centers, k_par_min, delta_z, z_chunks, cosmo)))
            jobs.append((("ma", which, s), compute_cross_corr_ma_for_seed_stitched,
                         (s, npz_of[s], K[(which, s)], K[(which, nxt)], cdir, N, pix, pix_area, k1d, k1d, kgrid,
                          k_bins, k_centers, slab_dz, args.narrow_dz, z0_ma, cosmo, args.ell_min_ksz,
                          args.chi_ksz, tag)))
    ncpu = int(os.environ.get("NCPUS", os.environ.get("PBS_NCPUS", os.cpu_count() or 2)))
    nw = args.workers or max(1, min(ncpu, 4, len(jobs)))
    print(f"{len(jobs)} worker calls, {nw} concurrent (NCPUS={ncpu})")
    res = {}
    t0 = time.time()
    if nw == 1:
        for key, fn, a in jobs:
            _, r, status = fn(a); res[key] = r
            print(f"  {key}: {status}  ({(time.time() - t0) / 60:.1f} min)")
    else:
        with ProcessPoolExecutor(max_workers=nw, mp_context=mp.get_context("fork")) as ex:
            futs = {ex.submit(fn, a): key for key, fn, a in jobs}
            for f in as_completed(futs):
                key = futs[f]
                try:
                    _, r, status = f.result(); res[key] = r
                    print(f"  {key}: {status}  ({(time.time() - t0) / 60:.1f} min)")
                except Exception as e:
                    print(f"  {key}: crashed: {e}"); res[key] = None
    for key, r in res.items():
        assert r is not None, f"{key} failed -- see messages above"

    if args.verify_eor:
        s0 = used[0]; vdir = os.path.join(total_cache, "_verify")
        a_sq = [a for key, fn, a in jobs if key == ("sq", "eor", s0)][0]
        a_ma = [a for key, fn, a in jobs if key == ("ma", "eor", s0)][0]
        _, fresh_sq, _ = compute_cross_corr_sq_for_seed_stitched(a_sq[:3] + (vdir,) + a_sq[4:])
        _, fresh_ma, _ = compute_cross_corr_ma_for_seed_stitched(a_ma[:4] + (vdir,) + a_ma[5:])
        worst = 0.0
        for z in fresh_sq:
            x, y = np.asarray(fresh_sq[z]["D_cross"], float), np.asarray(res[("sq", "eor", s0)][z]["D_cross"], float)
            if x.shape != y.shape:
                worst = np.inf; continue
            ok = np.isfinite(x) & np.isfinite(y)
            worst = max(worst, float(np.max(np.abs(x[ok] - y[ok]) / (np.abs(y[ok]) + 1e-300))) if ok.any() else 0.0)
        for z in fresh_ma:
            x = np.asarray(fresh_ma[z]["variants"][args.variant]["C_sq"], float)
            y = np.asarray(res[("ma", "eor", s0)][z]["variants"][args.variant]["C_sq"], float)
            if x.shape != y.shape:
                worst = np.inf; continue
            ok = np.isfinite(x) & np.isfinite(y)
            worst = max(worst, float(np.max(np.abs(x[ok] - y[ok]) / (np.abs(y[ok]) + 1e-300))) if ok.any() else 0.0)
        print(f"verify-eor (seed {s0}): worst relative difference fresh vs cached EoR = {worst:.2e}")
        assert worst < 1e-6, ("the existing EoR cache does not match a fresh computation (stale cache?). "
                              "Delete cross_corr_sq_stitched_seed*.npy / cross_corr_ma_stitched_seed*.npy and rerun 04 / 04b.")

    # ---- reionisation history (mean over used seeds) -----------------------------------------
    zrows, xrows = None, []
    for s in used:
        d = np.load(npz_of[s])
        zrows = np.asarray(d["z_arr"], float)
        xrows.append(1.0 - np.asarray(d["xH_box"]).mean(axis=(0, 1)))
    o = np.argsort(zrows)
    zh, xe_all = zrows[o], np.array(xrows)[:, o]
    xe_mean = xe_all.mean(axis=0)

    nS = len(used)
    out = {"seeds": np.array(used), "zh": zh, "xe_all": xe_all}

    # ---- Fig 1/2: kSZ^2 x 21cm^2 (k_par filter) at fixed ell, as 06 (chi-norm z0) ----------------
    UNIT_SQ21 = (T_CMB_K * 1e6) ** 2 * 1e6
    z0_sq = sorted(set.intersection(*[set(res[("sq", w, s)].keys()) for w in ("eor", "tot") for s in used]))
    chi2 = {z: float(cosmo.comoving_distance(z).value) ** 2 for z in z0_sq}

    def d_at_ell(r, ell_target):
        ell = np.asarray(r["ell"], float); D = np.asarray(r["D_cross"], float)
        ok = np.isfinite(ell) & np.isfinite(D)
        if ok.sum() < 2 or not (ell[ok].min() <= ell_target <= ell[ok].max()):
            return np.nan
        od = np.argsort(ell[ok])
        return float(np.interp(ell_target, ell[ok][od], D[ok][od]))

    out["z0_sq"] = np.array(z0_sq)
    out["xhii_sq"] = np.interp(z0_sq, zh, xe_mean)
    out["ells_zhou"] = np.array(args.ells_zhou)
    for w in ("eor", "tot"):
        arr = np.full((nS, len(z0_sq), len(args.ells_zhou)), np.nan)
        for i, s in enumerate(used):
            for j, z in enumerate(z0_sq):
                for m, Lm in enumerate(args.ells_zhou):
                    arr[i, j, m] = d_at_ell(res[("sq", w, s)][z], Lm) * UNIT_SQ21 / chi2[z]
        out[f"Dsq21sq_{w}"] = arr

    # ---- Fig 3: Ma-style, as 08 / 08b -------------------------------------------------------
    z0m = np.array(sorted(res[("ma", "eor", used[0])].keys()))
    for w in ("eor", "tot"):
        for s in used:
            assert np.allclose(sorted(res[("ma", w, s)].keys()), z0m)
    chi = np.array([res[("ma", "eor", used[0])][z]["chi_z0"] for z in z0m])
    ell = np.array([res[("ma", "eor", used[0])][z]["ell"] for z in z0m])
    out["z0_ma"] = z0m
    out["xe_of_z0_ma"] = np.interp(z0m, zh, xe_mean)
    UNIT = {"lin": T_CMB_K * 1e6, "sq": (T_CMB_K * 1e6) ** 2}
    ELL_SETS = {"lin": args.ells_lin, "sq": args.ells_sq}
    for kind, key in (("lin", "C_lin"), ("sq", "C_sq")):
        full = {}
        for w in ("eor", "tot"):
            arr = np.full((nS, len(z0m), ell.shape[1]), np.nan)
            for i, s in enumerate(used):
                for j, z in enumerate(z0m):
                    fac = ell[j] * (ell[j] + 1) / (2 * np.pi) * UNIT[kind] / chi[j] ** 2
                    arr[i, j] = fac * res[("ma", w, s)][z]["variants"][args.variant][key]
            full[w] = arr
        out[f"ells_{kind}"] = np.array(ELL_SETS[kind])
        for w in ("eor", "tot"):
            sel = np.full((nS, len(z0m), len(ELL_SETS[kind])), np.nan)
            for j in range(len(z0m)):
                ok = np.isfinite(ell[j]) & np.all(np.isfinite(full["eor"][:, j]), 0) & np.all(np.isfinite(full["tot"][:, j]), 0)
                if ok.sum() < 3:
                    continue
                for m, Lm in enumerate(ELL_SETS[kind]):
                    if not (ell[j][ok].min() <= Lm <= ell[j][ok].max()):
                        continue
                    for i in range(nS):
                        sel[i, j, m] = np.interp(np.log(Lm), np.log(ell[j][ok]), full[w][i, j][ok])
            out[f"D{kind}_{w}"] = sel
    out["variant"] = np.array(args.variant)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(data_path, **out)
    print("saved", data_path)


if not args.plot_only:
    compute()
assert os.path.exists(data_path), f"missing {data_path} -- run without --plot-only first (or point --data-dir at it)"
dd = np.load(data_path)

# =============================================================================
# 2. PLOT
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ksz2_21cm.plotting.style import apply_global_style

apply_global_style()
seeds = dd["seeds"]; nS = len(seeds)
C_E, C_T = "C0", "C1"


def msem(a):
    n = np.sum(np.isfinite(a), axis=0)
    with np.errstate(all="ignore"):
        mu = np.nanmean(a, axis=0)
        se = np.nanstd(a, axis=0, ddof=1) / np.sqrt(np.maximum(n, 1)) if a.shape[0] > 1 else np.zeros_like(mu)
    return mu, se


lines = [f"EoR-only vs EoR + post-EoR kSZ,  {nS} seeds {list(map(int, seeds))}",
         "paired: same seeds, same 21cm, same EoR kSZ; 'total' adds the post-EoR kSZ map",
         "D in the same units as 06 / 08b (chi-norm z0). diff = total - EoR (mean +/- s.e.m. over seeds, paired).", ""]

# ---------------- Fig 1 & 2: kSZ^2 x 21cm^2 vs z0 --------------------------------------------------
z0 = dd["z0_sq"]; xh = dd["xhii_sq"]; ells = dd["ells_zhou"]
E, T = dd["Dsq21sq_eor"], dd["Dsq21sq_tot"]            # (nS, nz, nL)
zhou = None
if not args.no_zhou:
    from ksz2_21cm.plotting.zhou25_data import ZHOU_FIG5
    zhou = ZHOU_FIG5


def top_axis(ax0):
    xlim = ax0.get_xlim()
    top = ax0.twiny(); top.set_xlim(xlim)
    tz = [z for z in z0 if abs(z - round(z)) < 1e-9]
    top.set_xticks(tz); top.set_xticklabels([f"{xh[list(z0).index(z)]:.2f}" for z in tz])
    top.set_xlabel(r"$\bar{x}_{\rm HII}$ (EoR run)", fontsize=13); top.tick_params(top=True, labeltop=True)


UL = r"$\mu\mathrm{K}^4/\chi^2$"
nL = len(ells)
fig, axes = plt.subplots(nL, 1, figsize=(8.8, 3.5 * nL), sharex=True)
axes = np.atleast_1d(axes)
for m, (ax, Lm) in enumerate(zip(axes, ells)):
    allv = []
    if not np.isfinite(E[:, :, m]).any():
        ax.text(0.5, 0.5, rf"$\ell={Lm:g}$ outside the grid's ell range", transform=ax.transAxes, ha="center")
        continue
    for arr, c, ls_, lab in ((E[:, :, m], C_E, "-", "EoR only"), (T[:, :, m], C_T, "--", "EoR + post-EoR")):
        mu, se = msem(arr)
        ax.fill_between(z0, mu - se, mu + se, color=c, alpha=0.22, lw=0)
        ax.plot(z0, mu, ls_, marker="o", ms=3.5, color=c, label=f"{lab}  ({nS} seeds, mean ± s.e.m.)")
        allv.append(np.concatenate([mu - se, mu + se]))
    ymax = 1.1 * np.nanmax(np.abs(np.concatenate(allv)))
    ax.set_ylim(-ymax, ymax)
    hs, ls_l = ax.get_legend_handles_labels()
    if zhou is not None and int(Lm) in zhou:
        zz = zhou[int(Lm)]
        axr = ax.twinx()
        h, = axr.plot(zz[:, 0], zz[:, 1], "--", color="k", lw=1.3, label="Zhou+25 Fig.5, right axis (HERA+SO filters)")
        yr = 1.1 * np.max(np.abs(zz[:, 1])); axr.set_ylim(-yr, yr)
        axr.set_ylabel(r"Zhou+25 $D_\ell$ [$\mu\mathrm{K}^4$]", fontsize=11)
        hs.append(h); ls_l.append(h.get_label())
    ax.axhline(0, color="gray", lw=0.8)
    ax.text(0.02, 0.93, rf"$\ell = {Lm:g}$", transform=ax.transAxes, va="top", fontsize=13)
    ax.set_ylabel(rf"$D_\ell$ [{UL}]", fontsize=13)
    ax.legend(hs, ls_l, loc="lower left", frameon=False, fontsize=8.5)
axes[-1].set_xlabel(r"Chunk-centre redshift $z_0$")
axes[0].invert_xaxis()
top_axis(axes[0])
fig.suptitle(r"$D_\ell^{\mathrm{kSZ}^2\times 21\mathrm{cm}^2}$ vs $z_0$: reionisation-only vs including post-EoR kSZ", y=1.0)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, f"{args.name}_vs_z.{ext}"), bbox_inches="tight", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(nL, 1, figsize=(8.8, 3.4 * nL), sharex=True)
axes = np.atleast_1d(axes)
lines.append("kSZ^2 x 21cm^2 (k_par filter), D_ell/chi^2 [muK^4/chi^2]")
lines.append(f"{'ell':>6} {'peak z0 (EoR)':>14} {'D_EoR@peak':>12} {'D_tot@peak':>12} {'tot/EoR':>8} "
             f"{'paired diff':>12} {'+/- sem':>10} {'|diff|/sem':>10}")
for m, (ax, Lm) in enumerate(zip(axes, ells)):
    if not np.isfinite(E[:, :, m]).any():
        lines.append(f"{Lm:6g}  outside the grid's ell range -- skipped"); continue
    diff = T[:, :, m] - E[:, :, m]
    mu, se = msem(diff)
    ax.fill_between(z0, mu - se, mu + se, color="C2", alpha=0.25, lw=0)
    ax.plot(z0, mu, "-o", ms=3.5, color="C2", label="total $-$ EoR (paired)")
    muE, _ = msem(E[:, :, m])
    ax.plot(z0, muE, "-", color=C_E, lw=1.0, alpha=0.6, label="EoR only (for scale)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.text(0.02, 0.93, rf"$\ell = {Lm:g}$", transform=ax.transAxes, va="top", fontsize=13)
    ax.set_ylabel(rf"$\Delta D_\ell$ [{UL}]", fontsize=12)
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    jpk = int(np.nanargmax(np.abs(muE)))
    d_pk = diff[:, jpk]
    mpk, spk = np.nanmean(d_pk), (np.nanstd(d_pk, ddof=1) / np.sqrt(nS) if nS > 1 else np.nan)
    tE, tT = np.nanmean(E[:, jpk, m]), np.nanmean(T[:, jpk, m])
    lines.append(f"{Lm:6g} {z0[jpk]:14.2f} {tE:12.3e} {tT:12.3e} {tT / tE:8.3f} {mpk:12.3e} {spk:10.2e} "
                 f"{abs(mpk) / spk if spk else np.nan:10.2f}")
axes[-1].set_xlabel(r"Chunk-centre redshift $z_0$")
axes[0].invert_xaxis()
top_axis(axes[0])
fig.suptitle(r"effect of the post-EoR kSZ on $D_\ell^{\mathrm{kSZ}^2\times 21\mathrm{cm}^2}$ (paired difference)", y=1.06, fontsize=13)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, f"{args.name}_diff.{ext}"), bbox_inches="tight", dpi=200)
plt.close(fig)

# ---------------- Fig 3: Ma-style vs x_e -------------------------------------------------------------
if not args.no_ma and "Dlin_eor" in dd.files:
    from ksz2_21cm.plotting.ma18_data import MA18, ma18_xe_of_z
    xe0 = dd["xe_of_z0_ma"]
    LC = {100: "goldenrod", 500: "mediumorchid", 1000: "turquoise", 3000: "purple"}
    UN = {"lin": r"mK$\,\mu$K", "sq": r"mK$\,\mu$K$^2$"}
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
                         "legend.fontsize": 8, "legend.framealpha": 0.85})
    fig, axL = plt.subplots(2, 1, figsize=(10.5, 10.5))
    lines.append("")
    lines.append(f"Ma-style ({str(dd['variant'])} 21cm map): paired change at the ell, over x_e in the window shown")
    for r_, kind in enumerate(("lin", "sq")):
        ax = axL[r_]; tw = ax.twinx()
        ells_k = dd[f"ells_{kind}"]; Ek, Tk = dd[f"D{kind}_eor"], dd[f"D{kind}_tot"]
        mk = "kSZ" if kind == "lin" else "kSZ2"
        hs, ls_l, ours, ma_v = [], [], [], []
        for m, Lm in enumerate(ells_k):
            c = LC.get(int(Lm), "k")
            if not np.isfinite(Ek[:, :, m]).any():
                lines.append(f"  ell={Lm:g} ({kind}): outside the grid's ell range -- skipped")
                continue
            for arr, st, lab in ((Ek[:, :, m], "-", "EoR"), (Tk[:, :, m], "--", "EoR+post")):
                mu, se = msem(arr)
                ok = np.isfinite(mu)
                o = np.argsort(xe0[ok])
                xs, m_, s_ = xe0[ok][o], mu[ok][o], se[ok][o]
                if st == "-":
                    ax.fill_between(xs, m_ - s_, m_ + s_, color=c, alpha=0.18, lw=0)
                h, = ax.plot(xs, m_, st, color=c, lw=1.7, label=rf"ours $\ell$={Lm:g}, {lab}")
                hs.append(h); ls_l.append(h.get_label())
                ours.append(np.abs(np.concatenate([m_ - s_, m_ + s_])))
            key = f"F3_{mk}_l{int(Lm)}"
            if key in MA18:
                s_ma = MA18[key]
                tw.errorbar(ma18_xe_of_z(s_ma[:, 0]), s_ma[:, 1], s_ma[:, 2], fmt="s", ms=3, mfc="none", mec=c,
                            ecolor=c, elinewidth=0.7, alpha=0.8)
                hs.append(plt.Line2D([], [], ls="", marker="s", mfc="none", mec=c)); ls_l.append(rf"Ma+18 $\ell$={int(Lm)}")
                ma_v.append(np.abs(np.concatenate([s_ma[:, 1] - s_ma[:, 2], s_ma[:, 1] + s_ma[:, 2]])))
            # paired change, band-free summary: where |EoR| is largest
            muE = np.nanmean(Ek[:, :, m], 0); jpk = int(np.nanargmax(np.abs(muE)))
            dpk = Tk[:, jpk, m] - Ek[:, jpk, m]
            sem_ = np.nanstd(dpk, ddof=1) / np.sqrt(nS) if nS > 1 else np.nan
            lines.append(f"  {('kSZ x 21cm' if kind == 'lin' else 'kSZ^2 x 21cm'):13s} ell={Lm:6g}  at x_e={xe0[jpk]:.2f} (|EoR| max): "
                         f"EoR {np.nanmean(Ek[:, jpk, m]):+.3e}  total {np.nanmean(Tk[:, jpk, m]):+.3e}  "
                         f"diff {np.nanmean(dpk):+.2e} +/- {sem_:.1e}")
        if not ours:
            continue
        v1 = np.nanmax(np.concatenate(ours)); lt1 = 0.1 * v1
        ax.set_yscale("symlog", linthresh=lt1); ax.set_ylim(-1.15 * v1, 1.15 * v1)
        if ma_v:
            v2 = np.nanmax(np.concatenate(ma_v)); tw.set_yscale("symlog", linthresh=0.1 * v2); tw.set_ylim(-1.15 * v2, 1.15 * v2)
        ax.axhline(0, color="gray", lw=0.8, ls="--"); ax.set_xlim(0, 1)
        ax.set_ylabel(rf"$D_\ell/\chi^2$ [{UN[kind]}]  (ours, symlog)")
        tw.set_ylabel(rf"Ma+18 $D_\ell$ [{UN[kind]}]  (squares, symlog)")
        ax.set_title({"lin": r"kSZ $\times$ 21cm (linear)", "sq": r"kSZ$^2$ $\times$ 21cm (21cm not squared)"}[kind], fontsize=11)
        ax.legend(hs, ls_l, frameon=True, fontsize=7.5, ncol=2, loc="lower right")
        ax.set_xlabel(r"$\bar{x}_e$" + ("  (Ma+18 mapped through their own history)" if kind == "sq" else ""))
    fig.suptitle(f"EoR only (solid) vs EoR + post-EoR kSZ (dashed), {nS} seeds, 21cm map = '{str(dd['variant'])}'", fontsize=12, y=0.995)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{args.name}_vs_xe.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)

with open(os.path.join(out_dir, f"{args.name}_table.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print("Saved: " + os.path.join(out_dir, args.name + "_{vs_z,diff,vs_xe}.{pdf,png}, _table.txt, _data.npz"))
