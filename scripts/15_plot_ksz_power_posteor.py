#!/usr/bin/env python
# =============================================================================
# 15_plot_ksz_power_posteor.py   (NEW, 2026-10-16)
#
# kSZ angular power spectrum of ONE seed, split into
#     EoR only (patchy window, as in all earlier results),
#     post-EoR only (02b: fully ionized gas, z < ~4.2 -> 0),
#     total = EoR + post-EoR (sum of the two maps),
# with Ma+18's reionization kSZ overlaid (it is EoR-only) when available.
# Top: D_ell. Bottom: post-EoR / EoR and total / EoR.
#
# ell = k * chi_eff with chi_eff = 8504 Mpc (the ksz-pipeline convention, unchanged). The
# post-EoR gas is closer than that, so its true ell is HIGHER than shown; read the post-EoR
# and total curves as approximate in ell.
#
# Light (maps already exist). Run on an interactive node or via PBS:
#   python scripts/15_plot_ksz_power_posteor.py --config configs/variants/stitched_256.yaml --seed 1
# =============================================================================
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from ksz2_21cm.utils.config import load_config, ensure_dirs
from ksz2_21cm.ksz.lightcone_integral import ksz_map_to_Dl

p = argparse.ArgumentParser()
p.add_argument("--config", default="configs/variants/stitched_256.yaml")
p.add_argument("--seed", type=int, default=1)
p.add_argument("--chi-ksz", type=float, default=8504.0)
p.add_argument("--ell-valid", type=float, default=700.0, help="grey out ell below this (unvalidated in our box)")
p.add_argument("--logy", action="store_true")
p.add_argument("--name", default=None)
args = p.parse_args()
name = args.name or f"ksz_power_posteor_seed{args.seed}"

cfg = load_config(args.config); ensure_dirs(cfg)
cache_dir, plot_dir = cfg["paths"]["cache_dir"], cfg["paths"]["plot_dir"]
L = cfg["coeval_ksz"]["box_len"]
md = os.path.join(cache_dir, "kSZ_maps")
Ke = np.load(os.path.join(md, f"kSZ_map_stitched_seed{args.seed}.npy"))
Kp = np.load(os.path.join(md, f"kSZ_map_postEoR_seed{args.seed}.npy"))
curves = {}
for lab, m in (("eor", Ke), ("post", Kp), ("tot", Ke + Kp)):
    ell, D, err = ksz_map_to_Dl(m, L, chi_Mpc=args.chi_ksz)
    curves[lab] = (ell, D, err)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11})
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.2, 8.4), sharex=True, gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.05})
C = {"eor": "tab:blue", "post": "tab:green", "tot": "k"}
LB = {"eor": "EoR only (patchy window)", "post": "post-EoR only (z < ~4.2)", "tot": "total = EoR + post-EoR"}
LS = {"eor": "-", "post": "-", "tot": "--"}
for lab in ("eor", "post", "tot"):
    ell, D, err = curves[lab]
    ax.plot(ell, D, color=C[lab], ls=LS[lab], lw=2.2, label=LB[lab])
try:
    from ksz2_21cm.plotting.ma18_ksz_auto import MA18_KSZ_AUTO
    a = np.asarray(MA18_KSZ_AUTO)
    ax.errorbar(a[:, 0], a[:, 1], yerr=a[:, 2], fmt="s", ms=4, mfc="none", color="tab:red", lw=1,
                label="Ma+18 reionization kSZ (EoR only, digitized)")
except Exception:
    pass
for a_ in (ax, ax2):
    a_.axvspan(0, args.ell_valid, color="0.9", zorder=0)
ax.text(0.015, 0.97, rf"grey: $\ell<{args.ell_valid:g}$ unvalidated in our box", transform=ax.transAxes, va="top", fontsize=9)
ax.set_xscale("log")
if args.logy:
    ax.set_yscale("log")
ax.set_ylabel(r"$D_\ell^{\rm kSZ}=\ell(\ell+1)C_\ell/2\pi$  [$\mu$K$^2$]")
ax.legend(fontsize=9, loc="upper right")
ax.set_title(f"kSZ power of one seed (seed {args.seed}): reionization, post-reionization and total", fontsize=11)
e0 = curves["eor"]
r_post = np.interp(e0[0], curves["post"][0], curves["post"][1]) / e0[1]
r_tot = np.interp(e0[0], curves["tot"][0], curves["tot"][1]) / e0[1]
ax2.plot(e0[0], r_post, color=C["post"], lw=2, label="post-EoR / EoR")
ax2.plot(e0[0], r_tot, color="k", ls="--", lw=2, label="total / EoR")
ax2.axhline(1, color="0.5", lw=0.8)
ax2.set_xlabel(r"$\ell = k\,\chi_{\rm eff}$,  $\chi_{\rm eff}$ = %g Mpc  (post-EoR gas is closer: its true $\ell$ is higher)" % args.chi_ksz, fontsize=9)
ax2.set_ylabel("ratio"); ax2.legend(fontsize=9)
ax2.set_xlim(e0[0].min() * 0.9, e0[0].max() * 1.1)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(plot_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
print(f"{'ell':>7s} {'EoR':>8s} {'post':>8s} {'total':>8s}  post/EoR  total/EoR")
for i in range(0, len(e0[0]), max(1, len(e0[0]) // 12)):
    print(f"{e0[0][i]:7.0f} {e0[1][i]:8.3g} {np.interp(e0[0][i], *curves['post'][:2]):8.3g} "
          f"{np.interp(e0[0][i], *curves['tot'][:2]):8.3g}  {r_post[i]:8.2f}  {r_tot[i]:8.2f}")
print("Saved:", os.path.join(plot_dir, name + ".{pdf,png}"))
