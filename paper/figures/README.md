# paper/figures/

Final, paper-ready figure files (PDF/PNG) go here — the ones that actually
appear in the submitted manuscript. These ARE committed (see the
`!paper/figures/**` exception in `.gitignore`), unlike the working plots
that `scripts/*.py` and `paper/figure_scripts/*.py` write to
`configs/fiducial.yaml`'s `paths.plot_dir` (which is git-ignored).

Workflow: run `paper/figure_scripts/overlay_zhou25.py` (or whichever
script), find the figure you want in `plot_dir`, copy the final version
here with a paper-relevant name (e.g. `fig3_Dell_vs_ell_z9.pdf`), and
reference it from the manuscript by that stable path.
