# ksz2-21cm

kSZ² × 21cm² cross-correlation forecast (Zhou et al. 2025, ApJ 991 195,
style), from 21cmFAST lightcones through to an instrument-noise-included
SNR forecast for CMB-S4 × SKA1-Low.

PhD project of Swanith Upadhye (TIFR), supervised by Prof. Girish Kulkarni.
Companion repo: [`ksz-pipeline`](https://github.com/swanith01/ksz-pipeline)
(kSZ power spectrum from lightcones/coeval boxes — the earlier, standalone
kSZ-only project).

## What this repo reproduces

The main result is the **squared** cross-correlation, kSZ² × 21cm²
(`scripts/04_compute_cross_corr_sq.py`), compared against Zhou+25's
digitized figures (`paper/figure_scripts/`), with an instrument-noise SNR
forecast for CMB-S4 × SKA1-Low (`scripts/05_compute_snr_forecast.py`).
The simpler **unsquared** kSZ × 21cm statistic (`scripts/03_compute_cross_corr.py`)
is kept as a diagnostic/comparison point — see
`notebooks/exploratory/README.md` for why the squared statistic is the one
that survives foreground-wedge filtering.

## Pipeline (run in order)

```
scripts/01_run_lightcones.py       21cmFAST lightcones, one per seed
scripts/02_compute_ksz_maps.py     reionization history, tau(z), kSZ maps
scripts/03_compute_cross_corr.py   kSZ x 21cm      (unsquared, diagnostic)
scripts/04_compute_cross_corr_sq.py  kSZ^2 x 21cm^2 (squared, MAIN RESULT)
scripts/05_compute_snr_forecast.py   SNR forecast, CMB-S4 x SKA1-Low
paper/figure_scripts/overlay_zhou25.py   figures vs. Zhou+25 Fig 3/4/5
paper/figure_scripts/run_zhou25_fig5.py  D_ell(z) in the Fig 5 style
paper/figure_scripts/shape_comparison_zhou25.py  peak-normalized shape-only
                                          comparison — RECONSTRUCTED, see below
```

Every script takes `--config configs/fiducial.yaml` (default). All of them
are safe to re-run: each checks its own on-disk cache before recomputing.

```bash
conda env create -f environment.yml
conda activate ksz2-21cm
python scripts/01_run_lightcones.py
python scripts/02_compute_ksz_maps.py
python scripts/03_compute_cross_corr.py
python scripts/04_compute_cross_corr_sq.py
python scripts/05_compute_snr_forecast.py
python paper/figure_scripts/overlay_zhou25.py
python paper/figure_scripts/run_zhou25_fig5.py
```

**Before running for real**, open `configs/fiducial.yaml` and fill in every
field marked `# CHECK` with the actual values from your old `run_config.py`
— the values there now are placeholders inferred from the pipeline code,
not your real fiducial setup. `load_config()` will raise a clear `KeyError`
if a required field is missing, rather than silently running with wrong
numbers.

## Testing the pipeline (verified sequence)

Everything below was actually run, in this order, on a desktop (not the
cluster) and confirmed working. If you're setting this up on a new machine
or picking the project back up after a break, this is the fastest path to
confidence that nothing is silently broken.

### 1. Environment sanity checks (no py21cmfast calls yet)
```bash
conda activate ksz2-21cm
python -c "import py21cmfast; print(py21cmfast.__version__)"   # expect 3.3.1
python -m pytest tests/                                          # expect 4 passed
```
If either of these fails, see `environment.yml` — every dependency issue
hit while setting this up on a fresh desktop (wrong package name, a
matplotlib version incompatibility, a conda channel-mixing trap, and a
`~/.local` user-site shadowing issue) is documented there with the exact
fix, not just here.

### 2. Smoke-test the full pipeline with `quicktest.yaml`
Small and fast on purpose (2 seeds, `HII_DIM=32`, narrow z-range) — this is
for confirming nothing is broken, not for physics. Run each step and check
its output before moving to the next one:
```bash
python scripts/01_run_lightcones.py      --config configs/variants/quicktest.yaml
python scripts/02_compute_ksz_maps.py    --config configs/variants/quicktest.yaml
python scripts/03_compute_cross_corr.py  --config configs/variants/quicktest.yaml
python scripts/04_compute_cross_corr_sq.py --config configs/variants/quicktest.yaml
python paper/figure_scripts/overlay_zhou25.py          --config configs/variants/quicktest.yaml
python paper/figure_scripts/shape_comparison_zhou25.py --config configs/variants/quicktest.yaml
```
Expect each to end with a `✓ N/N ready` or `Saved: ...` line. `UserWarning`
lines about `FlagOptions`/`hires_vx` etc. from py21cmfast itself are normal
noise — ignore them. `RuntimeWarning: Mean of empty slice` from the figure
scripts is also expected at this tiny scale (some ℓ-bins genuinely have no
valid modes in a 32-pixel box) — it should mostly disappear at
`fiducial.yaml`'s real resolution.

`scripts/05_compute_snr_forecast.py` is the one exception: its full-
covariance SNR estimator needs **at least 4 seeds** to build a covariance
matrix at all, so it can't be meaningfully tested on `quicktest`'s 2 seeds
— that one only gets a real test on a real (≥4-seed) run.

### 3. Spot-check the actual numbers, not just "did it crash"
A script exiting cleanly doesn't mean the output is physically sane. After
step 04, check the cross-power spectrum directly:
```bash
python3 << 'EOF'
import numpy as np
d = np.load('runs/quicktest/cache/seed_1/cross_corr_sq_seed1.npy', allow_pickle=True).item()
for z0, res in d.items():
    D = res['D_cross']
    ell = res['ell']
    print(f"z0={z0}: ell range [{ell.min():.1f}, {ell.max():.1f}]  D_cross finite={np.isfinite(D).sum()}/{len(D)}  D_cross range [{np.nanmin(D):.3e}, {np.nanmax(D):.3e}]")
EOF
```
Look for: a sensible ℓ range (not zero, not absurd), most bins finite (some
NaN at the box edges is fine, especially at `quicktest` scale), and a
nonzero, non-degenerate `D_cross` range. Then actually open the PNGs from
`plot_dir` and look at them — shapes that track loosely with the Zhou+25
overlay curves, no flat lines or exploding axes.

### 4. Known gotchas already fixed (don't re-debug these)
If you hit any of these again on a *different* machine, the fix is already
written down — check there first before troubleshooting from scratch:
- `AttributeError: 'py21cmfast' package name` → see `environment.yml`
  (it's `21cmfast` on conda-forge, imports as `py21cmfast`)
- `AttributeError: plt.register_cmap` → matplotlib version pin, see
  `environment.yml`
- Conda "unsatisfiable" wall of text on any `conda install` into this env
  → you probably forgot `-c conda-forge`, see `environment.yml`
- `matplotlib.__file__` pointing at `~/.local/...` instead of the env →
  user-site shadowing, see the `PYTHONNOUSERSITE` note in `environment.yml`
- `FileExistsError: .../<class 'pathlib.Path'>/wisdoms` or a seed reporting
  "computed" with no `LightCone_*.h5` anywhere → see the two fixes in
  `src/ksz2_21cm/simulate/lightcone_worker.py` (forcing `config['direc']`
  per-seed, and calling `lc.save()` explicitly — `write=True` alone only
  caches intermediate boxes, not the final LightCone)



```
configs/            fiducial.yaml + variants — all run parameters, no code
src/ksz2_21cm/
  simulate/          heavy py21cmfast calls (lightcone_worker.py)
  ksz/               kSZ integrand + line-of-sight map (ksz_map.py)
  correlation/       cross-correlation workers, unsquared + squared
  noise/             instrument noise models + SNR estimators
  io/                cache load/save, seed-averaging helpers
  plotting/          house plot style, save_pdf_png, digitized Zhou+25 data
  utils/             config loader, cosmology
scripts/             the numbered pipeline steps above — orchestration +
                     plotting live here, not in src/
paper/figure_scripts/  final comparison figures for the paper
notebooks/exploratory/  superseded SNR versions, diagnostic-only cells,
                        and the original monolithic script (see its README)
data/                data manifest (see data/README.md) — NOT the data itself
tests/               (empty for now — see "What's not done" below)
```

## Data — what's committed vs. what isn't

Following the same convention as `ksz-pipeline`:
- **Committed** (small, `data/products/`, once you start populating it):
  final `.npy`/`.npz` products — kSZ maps (~few MB each), the
  `cross_corr_sq_seed*.npy` dictionaries, anything a figure script reads
  directly.
- **Not committed** (`.gitignore`): raw 21cmFAST `LightCone_*.h5` files and
  any py21cmfast internal cache — these are multi-GB per seed and GitHub
  will reject anything over 100MB anyway. `data/README.md` records where
  the real cache lives on `pride`/`swarm`.

You mentioned wanting lightcones committed too — worth flagging that these
are HDF5 files with full 3D density/velocity/ionization boxes per seed,
typically GBs each, so they don't fit the "commit the products" convention
from `ksz-pipeline` and GitHub will hard-reject anything over 100MB. If you
want a lightweight, shareable version of a lightcone, consider committing
just the small derived quantities (e.g. `node_redshifts`, `global_xH`) as a
`.npz`, not the full HDF5.

## Known issues / decisions you should make deliberately

These were found while reorganizing the code, not introduced by it — see
inline comments at each location for detail:

1. **Cosmology**: every script used a custom
   `FlatLambdaCDM(H0=67.77, Om0=0.3086)`, not astropy's `Planck18` preset
   that `ksz-pipeline` standardised on. All your existing cached
   `cross_corr_sq` results depend on this cosmology (it sets χ(z), hence
   ℓ = k·χ). See `src/ksz2_21cm/utils/cosmology.py` — don't switch to
   Planck18 without re-deriving everything downstream.
2. **Unit calibration is adhoc**: the pipeline's internal power spectra are
   in FFT/pixel units, not physical μK⁴. `overlay_zhou25.py` and
   `scripts/05_compute_snr_forecast.py` both apply a scale factor
   calibrated by matching your noise-free peak to a single point in
   Zhou+25 Fig. 5 — this is flagged in both scripts' docstrings and printed
   at runtime, not hidden.
3. **Possible double-`h`-division** in the old noise-free SNR cell's
   unsquared-statistic branch (`legacy_snr_noise_free_cell.py` in
   `notebooks/exploratory/`) — doesn't affect the squared statistic used
   in the paper, and that cell is superseded anyway, but flagged so it
   doesn't get copied into new code.

## What's not done (be aware before you tell Prof. Kulkarni this is finished)

- **`paper/figure_scripts/shape_comparison_zhou25.py` is RECONSTRUCTED, not
  verified.** You had two output plots (`A_Dell_vs_ell_z9_shape.png`,
  `B_Dell_vs_z_shape.png`, both dated ~2 weeks before this cleanup) whose
  source script wasn't among the files you gave me. This script was
  written by inferring the logic from the images themselves (each curve
  divided by its own peak |D_ell|, which is why it sidesteps the unit
  ambiguity flagged in `overlay_zhou25.py` — dividing by a curve's own
  peak cancels out any constant amplitude factor). **Run it once the
  cluster is back and diff the output against your original two PNGs
  before trusting it for the paper.** If you ever find the real script,
  replace this one with it.
- `tests/` is empty. The checklist he sent asks for a reproducible
  pipeline, not necessarily unit tests, but at minimum consider one smoke
  test per `src/` module (e.g. run `cross_corr_sq_worker` on a tiny
  synthetic lightcone and check the output shape/keys).
- CELLs 7c, 8a, 8b, 8.5, 8c, 9 from the original script (diagnostic plots)
  have not been individually split into notebooks yet — see
  `notebooks/exploratory/README.md`.
- `LICENSE` and `CITATION.cff` here are placeholders — fill in the license
  Prof. Kulkarni wants and your actual citation details before the repo is
  public / connected to Zenodo.
- No git history exists yet — this is a filesystem layout, not a repo.
  See "Getting this into git" below.

## Getting this into git

```bash
cd ksz2-21cm
git init
git add .
git commit -m "Initial commit: restructured kSZ^2 x 21cm pipeline"
git branch -M main
git remote add origin https://github.com/swanith01/ksz2-21cm.git
git push -u origin main
```

Once the first full pipeline run works end-to-end on `pride`/`swarm`, tag it:

```bash
git tag -a v0.1 -m "First full pipeline run"
git push origin v0.1
```

Do the same with `submitted-v1` / `accepted-v1` at the corresponding paper
milestones, per Prof. Kulkarni's checklist.
