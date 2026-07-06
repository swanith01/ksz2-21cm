# notebooks/exploratory/

Everything in this directory is **exploratory, diagnostic, or superseded** —
none of it is required to reproduce the paper's headline results. It's kept
so nothing from your original code is lost, and so the reasoning trail
(why the SNR model evolved, why the squared statistic was chosen over the
unsquared one) is still visible.

| File | What it is | Status |
|---|---|---|
| `original_monolith_kSZ_Squared_21cm_11Jun_CLUSTER.py` | Your original full pipeline script, unchanged | Reference copy. Most cells have been split into `scripts/` + `src/` (see the header of this file for the exact mapping). CELLs 7c, 8a, 8b, 8.5, 8c, 9 (diagnostic plots) have **not** been individually extracted yet — they still only exist here. |
| `legacy_snr_noise_free_cell.py` | The first SNR forecast (`CELL SNR (CORRECTED)`), noise-free, diagonal estimator | Superseded by `scripts/05_compute_snr_forecast.py`. Kept for reference; see the header for a flagged possible double-`h`-division bug in its unsquared-statistic branch (does not affect the squared statistic, which is what matters for the paper). |
| `cell_snr_with_noise_SUPERSEDED.py` | Second SNR iteration: adds a simple noise-to-signal-ratio degradation | Superseded by `scripts/05_compute_snr_forecast.py` (frequency-dependent T_sys + full covariance matrix). |

## If you want to turn a cell into a real notebook

The archived monolith already uses the `# %%` cell-marker convention that
[jupytext](https://jupytext.readthedocs.io/) understands:

```bash
pip install jupytext
jupytext --to notebook original_monolith_kSZ_Squared_21cm_11Jun_CLUSTER.py
```

This opens directly in Jupyter with the original cell boundaries intact.
Delete everything except the cell you want (e.g. CELL 9) and save it into
this directory as its own `.ipynb` — that keeps true one-off exploration
in notebooks, per the "notebooks only for exploration" convention, without
requiring a from-scratch rewrite of every diagnostic plot up front.

## Why CELL 9 is worth reading first

CELL 9 ("why kSZ²×21cm (unsquared) dies under wedge filtering") is the
diagnostic that explains *why* the pipeline uses the squared kSZ²×21cm²
statistic (Zhou+25 style, `scripts/04_compute_cross_corr_sq.py`) rather
than the simpler unsquared kSZ×21cm statistic (Ma+18 style,
`scripts/03_compute_cross_corr.py`) as the paper's main result. If you're
picking this project back up after a break, that's the fastest way back
into the physics reasoning.
