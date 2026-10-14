#!/usr/bin/env python
# =============================================================================
# 12_seed_scaling.py
#
# How many seeds does it take to smooth out a curve? Uses the per-seed tables you already
# have (Ma18_style_data.npz from 08, wedge_survival_data.npz from 09), no cluster needed:
#
#   * take random subsets of n = 3 ... N seeds, average them, and measure the point-to-point
#     jaggedness of the mean curve vs z0 (rms of the second difference / sqrt(6), the noise per
#     point if the wiggles were white; correlated neighbours make it an underestimate);
#   * compare with the size of the signal (rms of the full-sample mean curve);
#   * fit noise ~ n^-1/2 and print the number of seeds needed for noise / signal = 0.5, 0.25, 0.1.
#
# Everything is evaluated in an ell band (default 500-1500), band-averaged per seed, so it
# also shows how much a wider band alone buys you.
#
#   python scripts/12_seed_scaling.py --data-dir data/plots/2026-10-12
# =============================================================================

import argparse
import os

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", required=True)
parser.add_argument("--ma-data", default="Ma18_style_data.npz")
parser.add_argument("--wedge-data", default="wedge_survival_data.npz")
parser.add_argument("--band", type=float, nargs=2, default=[500, 1500])
parser.add_argument("--draws", type=int, default=300)
parser.add_argument("--name", default="seed_scaling")
args = parser.parse_args()
ddir = os.path.expanduser(args.data_dir)
rng = np.random.default_rng(0)

SETS = []   # (label, file, key)
for v in ("slice", "narrow", "slab"):
    SETS.append((f"kSZ^2 x 21cm [{v}]", args.ma_data, f"D_sq_{v}"))
SETS.append(("kSZ x 21cm lin [slice]", args.ma_data, "D_lin_slice"))
SETS.append(("kSZ x 21cm lin [slab]", args.ma_data, "D_lin_slab"))
for st in ("lin_none", "sq_none", "sq_kpar", "sq_wedge"):
    SETS.append((f"r  {st}", args.wedge_data, f"r_{st}"))


def band_series(arr, ell, lo, hi):
    out = np.full((arr.shape[0], arr.shape[1]), np.nan)
    for j in range(arr.shape[1]):
        m = (ell[j] >= lo) & (ell[j] < hi)
        if m.any():
            out[:, j] = np.nanmean(arr[:, j, m], axis=1)
    return out


def rough(curve):
    c = curve[np.isfinite(curve)]
    if len(c) < 5:
        return np.nan
    d2 = c[2:] - 2 * c[1:-1] + c[:-2]
    return np.sqrt(np.mean(d2**2) / 6.0)


results = {}
cache = {}
print(f"band ell = {args.band[0]:g}-{args.band[1]:g};  roughness = noise per point (white-noise estimate)\n")
print(f"{'statistic':28s} {'N':>3s} {'signal rms':>11s} " + " ".join(f"n={n:<3d}" for n in (3, 5, 8, 10, 15)) +
      "   noise/signal at N    seeds for 0.5 / 0.25 / 0.1")
for label, fname, key in SETS:
    fp = os.path.join(ddir, fname)
    if not os.path.exists(fp):
        continue
    if fp not in cache:
        cache[fp] = np.load(fp)
    d = cache[fp]
    if key not in d.files:
        continue
    A = band_series(np.asarray(d[key], float), np.asarray(d["ell"], float), *args.band)
    nS = A.shape[0]
    sig = np.sqrt(np.nanmean(np.nanmean(A, axis=0) ** 2))
    ns = [n for n in range(3, nS + 1)]
    rr = []
    for n in ns:
        vals = []
        for _ in range(args.draws):
            sub = rng.choice(nS, size=n, replace=False)
            vals.append(rough(np.nanmean(A[sub], axis=0)))
        rr.append(np.nanmean(vals))
    ns, rr = np.array(ns), np.array(rr)
    # fit rough = a / sqrt(n) on log scale
    a = np.exp(np.mean(np.log(rr) + 0.5 * np.log(ns)))
    need = {f: a**2 / (f * sig) ** 2 for f in (0.5, 0.25, 0.1)}
    results[label] = (ns, rr, a, sig, nS)
    pick = {n: (rr[list(ns).index(n)] if n in ns else np.nan) for n in (3, 5, 8, 10, 15)}
    print(f"{label:28s} {nS:3d} {sig:11.3g} " + " ".join(f"{pick[n]/sig:5.2f}" for n in (3, 5, 8, 10, 15)) +
          f"   {a/np.sqrt(nS)/sig:8.2f}        " + " / ".join(f"{need[f]:6.0f}" for f in (0.5, 0.25, 0.1)))
print("\n(columns n=3..15 are noise/signal for the mean of n seeds; the last three numbers are the seeds "
      "needed for the stated noise/signal, from the n^-1/2 fit)")

if results:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for label, (ns, rr, a, sig, nS) in results.items():
        l, = ax.plot(ns, rr / sig, "-o", ms=3, label=label)
        nn = np.linspace(3, max(60, nS), 50)
        ax.plot(nn, a / np.sqrt(nn) / sig, ":", color=l.get_color(), lw=1)
    ax.axhline(0.25, color="gray", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of seeds averaged")
    ax.set_ylabel("jaggedness / signal size")
    ax.set_title(f"How many seeds smooth the curve? (ell {args.band[0]:g}-{args.band[1]:g}; dotted = n$^{{-1/2}}$ fit)",
                 fontsize=10)
    ax.legend(fontsize=7, frameon=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(ddir, f"{args.name}.{ext}")
        fig.savefig(out, bbox_inches="tight")
        print(f"✓ Saved: {out}")
