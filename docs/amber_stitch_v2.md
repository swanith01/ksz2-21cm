# AMBER stitcher v2 (2026-10-27)

The three fixes from the ksz-pipeline heads-up of 2026-10-26, applied to the AMBER -> stitched-lightcone path
(`scripts/01a_stitch_lightcones_amber.py`, `src/ksz2_21cm/simulate/amber_stream.py`). The reference implementation is
ksz-pipeline `amber-integration`, `src/ksz_pipeline/amber/stream_lightcone.py` (`mode='momentum'`,
`geometry='reorient'`), vendored here.

| # | v1 (Oct 21-23 products) | v2 |
|---|---|---|
| 1 | `velocity_z = mom2/rho2`, density and velocity stitched separately; `rho2` rings negative in voids so `v` blows up | `q_z = x_HII * mom2_los` cell by cell per snapshot, then blended in z. No division. `02` integrates `q_z` as it stands. Per-snapshot `rho2` statistics (min, fraction < 0.05, fraction < 0) are saved in the npz |
| 2 | wrap cycles decorrelated by a random in-plane rotation, bilinear sampling (low-passes: 0.80 / 0.59 of the power at ell = 5000 / 8000 in ksz-pipeline's 2 Gpc/h run) | per cycle a random box axis is the LOS (the momentum component along it is the LOS velocity) plus random integer shifts. Integer indexing only. `--geometry rotate` keeps v1's rotation for A/B tests |
| 3 | Planck18 `chi(z)` for the LOS grid and the config's `FlatLambdaCDM(67.66, 0.30964)` for `ell = k chi` | the run's own `amber_run.json` cosmology (`RunCosmology`, AMBER's `H(z)`) for the LOS grid and, via `cosmology.use: amber` in the config, for every `get_cosmology(cfg)` call. 01a refuses to run if the config and the run disagree. `n_e0` is the run's too (`ne0_cgs` in the npz, read by `02`) |

Size of #3 for our runs (planck18 preset): the stitcher's Planck18 `chi` was already exact (1e-6); what moves is
`ell = k chi` downstream, by -0.18% (z = 4.5) to -0.23% (z = 18) against the old config value. For a non-Planck18
preset (Chen+23) it is ~2%.

## What changed in files
- new: `src/ksz2_21cm/simulate/amber_stream.py`, `tests/test_amber_stream.py`, `RunCosmology` in `utils/cosmology.py`
- `01a`: v2 by default; `--legacy` reproduces v1; `--h` is read from `amber_run.json`; a cached v1 file is never reused as v2
- `02`: uses `q_z` and the run's `n_e0` when present; bit-identical output on a `velocity_z` file (checked)
- `configs/variants/amber_q512.yaml`: `cosmology.use: amber`
- `slurm/amber_production.sbatch`, `amber_ma_post.sbatch`: `VARIANT` (default `v2`) puts everything under
  `$R/ksz2-21cm-data/v2/` and `$R/amber_runs/prod_v2/`, so the Oct 21-23 products are never overwritten
- NOT touched: `stitch_from_coeval.py` and the 21cmFAST path (see below)

## Re-running
The production script deleted the raw `fields_z=*.dat` after stitching, so v1 products cannot be re-stitched; v2 re-runs
AMBER (about 1 min per run) in `prod_v2` run directories:
    cd $R && git -C ksz2-21cm pull && mkdir -p logs && sbatch ksz2-21cm/slurm/amber_production.sbatch
(`HISTS`, `SEEDS`, `--array` as before). Plots: `--data-root $R/ksz2-21cm-data/v2`.

## Not done here
- The 21cmFAST stitched path still uses the bilinear rotation (`stitch_from_coeval.stitch_field`, shared with the
  running HII_EFF scan). A v1-vs-reorient A/B on it, and on 21cmFAST's own `velocity_z`, is a separate check.
- `optical_depth.analytic_tau_below` and `compute_tau` still use astropy Planck18 internally (exact for the planck18
  preset; for another preset they need the same treatment).
- v1 AMBER results (Oct 21-23 plots, 10 seeds x 3 histories) stay provisional until v2 reproduces them.
