#!/usr/bin/env python
# =============================================================================
# 14_posteor_comparison.py   (NEW, 2026-10-14)
#
# Do the cross-correlations survive adding the post-EoR kSZ (fully ionized gas, z < ~5 -> 0)?
# For each seed that has BOTH kSZ maps
#     EoR-only : <cache>/kSZ_maps/kSZ_map_stitched_seed<N>.npy        (patchy window, as before)
#     total    : EoR-only + <cache>/kSZ_maps/kSZ_map_postEoR_seed<N>.npy   (02b)
# it computes, with the SAME 21cm and the SAME estimators (the validated workers),
#     (1) kSZ    x 21cm   (both linear; 21cm = Delta z slab mean)           no filter
#     (2) kSZ^2  x 21cm   (21cm not squared; central LOS cell of the chunk) none / k_par / wedge
#     (3) kSZ^2  x 21cm^2 (Zhou+25 style)                                    none / k_par / wedge
# as the correlation coefficient r = C / sqrt(P_kSZ-ish * P_21cm) averaged over an ell band, vs z0.
# Because the 21cm and the EoR kSZ are identical in both cases, any difference is the post-EoR
# component alone (a paired comparison, even with a single seed).
#
# Also shown: how much the post-EoR kSZ adds to the kSZ and kSZ^2 auto-power (vs k).
#
# Light (a few minutes for 1 seed); still run on an interactive node or PBS, not the login node.
#   python scripts/14_posteor_comparison.py --config configs/variants/stitched_256.yaml --seeds 1
# =============================================================================
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from ksz2_21cm.utils.config import load_config, ensure_dirs
from ksz2_21cm.utils.cosmology import get_cosmology
from ksz2_21cm.correlation.cross_corr_wedge_worker_stitched import (
    compute_cross_corr_wedge_for_seed_stitched, _bin_mean,
)

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/variants/stitched_256.yaml")
p.add_argument("--seeds", type=int, nargs="+", default=[1])
p.add_argument("--z0-min", type=float, default=6.0)
p.add_argument("--z0-max", type=float, default=12.0)
p.add_argument("--z0-step", type=float, default=0.5)
p.add_argument("--delta-z", type=float, default=None)
p.add_argument("--wedge-fraction", type=float, default=1.0)
p.add_argument("--band", type=float, nargs=2, default=[500, 1500])
p.add_argument("--ell-min-ksz", type=float, default=100.0)
p.add_argument("--chi-ksz", type=float, default=8504.0)
p.add_argument("--symlog", action="store_true")
p.add_argument("--linthresh", type=float, default=0.005)
p.add_argument("--name", default="posteor_comparison")
args = p.parse_args()

cfg = load_config(args.config); ensure_dirs(cfg)
cache_dir, plot_dir = cfg["paths"]["cache_dir"], cfg["paths"]["plot_dir"]
N = cfg["simulation"]["hii_dim"]; L = cfg["simulation"]["box_len"]
delta_z = args.delta_z if args.delta_z is not None else cfg["cross_correlation"]["delta_z"]
k_par_min = cfg["cross_correlation"]["k_par_min"]
cosmo = get_cosmology(cfg)
pix = float(L) / N; pix_area = pix**2; norm = pix_area / N**2
dk = 2 * np.pi / (N * pix)
kx = np.fft.fftshift(np.fft.fftfreq(N)) * N * dk
kgrid = np.sqrt(kx[:, None] ** 2 + kx[None, :] ** 2)
k_bins = np.logspace(np.log10(dk), np.log10(kgrid.max() * 0.9), 35)      # same as 04/04b/04c
k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
k1d = np.fft.fftfreq(N, d=pix) * 2 * np.pi
nb = len(k_centers)
bidx_full = np.digitize(kgrid, k_bins) - 1
valid = (bidx_full >= 0) & (bidx_full < nb); bidx = np.where(valid, bidx_full, 0)
counts = np.bincount(bidx[valid], minlength=nb).astype(float)
z0_list = [float(z) for z in np.arange(args.z0_min, args.z0_max + 1e-9, args.z0_step)]
base_tag = (f"_dz{delta_z:g}_kp{k_par_min:g}_wf{args.wedge_fraction:g}_lmin{args.ell_min_ksz:g}"
            f"_z{args.z0_min:g}-{args.z0_max:g}-{args.z0_step:g}")

maps_dir = os.path.join(cache_dir, "kSZ_maps")
STATS = [("lin", "none"), ("lin", "kpar"), ("lin", "wedge"), ("sq", "none"), ("sq", "kpar"), ("sq", "wedge")]
MAPS = ("eor", "total")


def band_r(C, PT, PK, ell):
    sel = (ell >= args.band[0]) & (ell <= args.band[1]) & np.isfinite(C) & np.isfinite(PT) & np.isfinite(PK) & (counts > 0)
    if sel.sum() == 0:
        return np.nan
    w = counts[sel]; w = w / w.sum()
    return np.sum(w * C[sel]) / np.sqrt(np.sum(w * PT[sel]) * np.sum(w * PK[sel]))


def lin_lin(K, bt, z_arr):
    """kSZ (linear) x 21cm slab mean (linear): r(z0) in the band, plus the kSZ auto-power (binned)."""
    FK = np.fft.fftshift(np.fft.fft2(K - K.mean()))
    PK = _bin_mean(np.abs(FK) ** 2 * norm, bidx, valid, nb, counts)
    out = []
    for z0 in z0_list:
        idx = np.where((z_arr >= z0 - delta_z / 2) & (z_arr <= z0 + delta_z / 2))[0]
        T = np.asarray(bt[:, :, idx], dtype=np.float64).mean(axis=2)
        FT = np.fft.fftshift(np.fft.fft2(T - T.mean()))
        C = _bin_mean(np.real(np.conj(FK) * FT) * norm, bidx, valid, nb, counts)
        PT = _bin_mean(np.abs(FT) ** 2 * norm, bidx, valid, nb, counts)
        ell = k_centers * float(cosmo.comoving_distance(z0).value)
        out.append(band_r(C, PT, PK, ell))
    return np.array(out), PK


R = {m: {s: [] for s in [("lin_lin", "none")] + STATS} for m in MAPS}
PKlin = {m: [] for m in MAPS}; PKsq = {m: [] for m in MAPS}
used = []
for seed in args.seeds:
    npz = os.path.join(cache_dir, f"seed_{seed}", f"stitched_lightcone_seed{seed}.npz")
    e_p = os.path.join(maps_dir, f"kSZ_map_stitched_seed{seed}.npy")
    q_p = os.path.join(maps_dir, f"kSZ_map_postEoR_seed{seed}.npy")
    if not (os.path.exists(npz) and os.path.exists(e_p) and os.path.exists(q_p)):
        print(f"x seed {seed}: missing lightcone / EoR map / post-EoR map -- skipping"); continue
    used.append(seed)
    K_e = np.load(e_p); K_t = K_e + np.load(q_p)
    d = np.load(npz); bt = d["brightness_temp"]; z_arr = np.asarray(d["z_arr"], dtype=np.float64)
    for label, K in (("eor", K_e), ("total", K_t)):
        r1, pk = lin_lin(K, bt, z_arr)
        R[label][("lin_lin", "none")].append(r1); PKlin[label].append(pk)
        a = (seed, npz, K, None, cache_dir, N, pix, pix_area, k1d, k1d, kgrid, k_bins, k_centers,
             k_par_min, delta_z, z0_list, cosmo, args.ell_min_ksz, args.chi_ksz,
             args.wedge_fraction, base_tag + f"_posteor_{label}")
        _, res, status = compute_cross_corr_wedge_for_seed_stitched(a)
        assert res is not None, f"seed {seed} {label}: {status}"
        print(f"seed {seed} [{label}]: {status}")
        for st in STATS:
            R[label][st].append(np.array([
                band_r(res[z]["stats"][f"{st[0]}_{st[1]}"]["C"], res[z]["stats"][f"{st[0]}_{st[1]}"]["P_T"],
                       res[z]["P_K"], res[z]["ell"]) for z in sorted(res)]))
        z8 = min(res, key=lambda z: abs(z - 8.0)); PKsq[label].append(res[z8]["P_K"])
assert used, "no seed has all three inputs (run 01b and 02b first)"
nS = len(used)
for m in MAPS:
    for s in R[m]:
        R[m][s] = np.array(R[m][s])           # (nS, nz)
    PKlin[m] = np.array(PKlin[m]); PKsq[m] = np.array(PKsq[m])

np.savez(os.path.join(plot_dir, f"{args.name}_data.npz"), seeds=np.array(used), z0=np.array(z0_list),
         k=k_centers, **{f"r_{m}_{s[0]}_{s[1]}": R[m][s] for m in MAPS for s in R[m]},
         PKlin_eor=PKlin["eor"], PKlin_total=PKlin["total"], PKsq_eor=PKsq["eor"], PKsq_total=PKsq["total"])

# ---------------------------------------------------------------- table
lines = []
def P(s=""):
    print(s); lines.append(s)
P(f"POST-EoR TEST  ({nS} seed(s): {used};  ell band {args.band[0]:g}-{args.band[1]:g};  r = correlation coefficient)")
P("paired: same 21cm and same EoR kSZ in both columns; 'total' adds the post-EoR kSZ map.\n")
zsel = [z for z in (9.5, 8.0) if min(abs(np.array(z0_list) - z)) < 1e-6]
hdr = f"{'statistic':34s}" + "".join(f"  z0={z:g}: r(EoR)  r(total)  ratio" for z in zsel)
P(hdr)
NAME = {"lin_lin": "kSZ x 21cm (linear)", "lin": "kSZ^2 x 21cm", "sq": "kSZ^2 x 21cm^2"}
FN = {"none": "no filter", "kpar": "k_par filter", "wedge": "wedge"}
for st in [("lin_lin", "none")] + STATS:
    row = f"{NAME[st[0]] + ', ' + FN[st[1]]:34s}"
    for z in zsel:
        j = int(np.argmin(np.abs(np.array(z0_list) - z)))
        a = np.nanmean(R["eor"][st][:, j]); b = np.nanmean(R["total"][st][:, j])
        row += f"        {a:+.4f}  {b:+.4f}  {b / a if abs(a) > 1e-4 else np.nan:6.2f}"
    P(row)
rk = np.nanmean(PKlin["total"], 0) / np.nanmean(PKlin["eor"], 0)
rs = np.nanmean(PKsq["total"], 0) / np.nanmean(PKsq["eor"], 0)
selk = (k_centers > 0.05) & (k_centers < 0.3)
P(f"\nauto-power increase from adding the post-EoR kSZ (k = 0.05-0.3 /Mpc, ell~400-2500): "
  f"kSZ x{np.nanmean(rk[selk]):.2f}, kSZ^2 x{np.nanmean(rs[selk]):.2f}")
open(os.path.join(plot_dir, f"{args.name}_table.txt"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- figure
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ksz2_21cm.ksz.lightcone_integral import ksz_map_to_Dl
plt.rcParams.update({"font.size": 9, "axes.labelsize": 10})
fig = plt.figure(figsize=(15.5, 8.6))
gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.27)
CF = {"none": "tab:blue", "kpar": "tab:red", "wedge": "0.45"}

# (a) kSZ auto-spectrum
ax = fig.add_subplot(gs[0, 0])
s0 = used[0]
Ke = np.load(os.path.join(maps_dir, f"kSZ_map_stitched_seed{s0}.npy"))
Kq = np.load(os.path.join(maps_dir, f"kSZ_map_postEoR_seed{s0}.npy"))
for lab, m, c, ls in (("EoR only (as before)", Ke, "tab:blue", "-"), ("post-EoR only", Kq, "tab:green", "-"),
                      ("EoR + post-EoR", Ke + Kq, "k", "--")):
    ell, D, _ = ksz_map_to_Dl(m, L, chi_Mpc=args.chi_ksz)
    ax.plot(ell, D, color=c, ls=ls, lw=1.8, label=lab)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$\ell = k\,\chi_{\rm eff}$  ($\chi_{\rm eff}$=%g Mpc)" % args.chi_ksz)
ax.set_ylabel(r"$D_\ell^{\rm kSZ}$ [$\mu$K$^2$]"); ax.legend(fontsize=8)
ax.set_title(f"kSZ auto-spectrum (seed {s0})\n(post-EoR gas is closer: its true ell is higher)", fontsize=9)

# (b) auto-power ratio
ax = fig.add_subplot(gs[0, 1])
ax.plot(k_centers, rk, color="tab:orange", lw=2, label="kSZ map")
ax.plot(k_centers, rs, color="tab:purple", lw=2, label=r"kSZ$^2$ map")
ax.axhline(1, color="0.5", lw=0.8); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$k$ [Mpc$^{-1}$]"); ax.set_ylabel("auto-power: (EoR + post) / EoR only"); ax.legend(fontsize=8)
ax.set_title("how much the post-EoR kSZ adds to the noise", fontsize=9)

# (c) text
z0a_ = np.array(z0_list)
z0a = z0a_
ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
SHORT = {"lin_lin": "kSZ x 21", "lin": "kSZ^2 x 21", "sq": "kSZ^2 x 21^2"}
SF = {"none": "none", "kpar": "k_par", "wedge": "wedge"}
txt = []
for zt in (9.5, 8.0):
    if min(abs(z0a_ - zt)) > 1e-6:
        continue
    j = int(np.argmin(np.abs(z0a_ - zt)))
    txt.append(f"z0 = {zt:g}   (r = band-averaged correlation coeff.)")
    txt.append(f"{'statistic':20s}{'r(EoR)':>9s}{'r(+post)':>10s}")
    for st in [("lin_lin", "none")] + STATS:
        a = np.nanmean(R["eor"][st][:, j]); b = np.nanmean(R["total"][st][:, j])
        txt.append(f"{SHORT[st[0]] + ', ' + SF[st[1]]:20s}{a:+9.4f}{b:+10.4f}")
    txt.append("")
ax.text(0, 1, "\n".join(txt), va="top", family="monospace", fontsize=7, transform=ax.transAxes)
ax.set_title("numbers behind the lower panels (paired)", fontsize=9)

# row 2: r vs z0
def draw(ax, stats, title):
    for st in stats:
        for m, ls in (("eor", "-"), ("total", "--")):
            y = R[m][st]
            mu = np.nanmean(y, 0)
            ax.plot(z0a, mu, color=CF[st[1]], ls=ls, lw=1.8, label=f"{FN[st[1]]}, {'EoR only' if m == 'eor' else 'EoR + post'}")
            if nS > 1:
                se = np.nanstd(y, 0, ddof=1) / np.sqrt(nS)
                ax.fill_between(z0a, mu - se, mu + se, color=CF[st[1]], alpha=0.12, lw=0)
    ax.axhline(0, color="0.5", lw=0.8); ax.invert_xaxis()
    if args.symlog:
        ax.set_yscale("symlog", linthresh=args.linthresh)
    ax.set_xlabel(r"$z_0$"); ax.set_ylabel(rf"$\langle r\rangle_{{\ell={args.band[0]:g}-{args.band[1]:g}}}$")
    ax.set_title(title, fontsize=9); ax.legend(fontsize=6.5, ncol=1)
draw(fig.add_subplot(gs[1, 0]), [("lin_lin", "none")], r"(1) kSZ $\times$ 21cm (both linear)")
draw(fig.add_subplot(gs[1, 1]), [("lin", "none"), ("lin", "kpar"), ("lin", "wedge")], r"(2) kSZ$^2$ $\times$ 21cm (21cm not squared)")
draw(fig.add_subplot(gs[1, 2]), [("sq", "none"), ("sq", "kpar"), ("sq", "wedge")], r"(3) kSZ$^2$ $\times$ 21cm$^2$ (Zhou+25)")
fig.suptitle(f"Do the cross-correlations survive adding the post-EoR kSZ?   ({nS} seed(s): {used}; solid = EoR only, dashed = EoR + post-EoR)", fontsize=11)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(plot_dir, f"{args.name}.{ext}"), dpi=200, bbox_inches="tight")
print("Saved:", os.path.join(plot_dir, args.name + ".{pdf,png,_table.txt,_data.npz}"))
