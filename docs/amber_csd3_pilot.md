# AMBER on CSD3: pilot (stage 2 of the plan)

Goal of the pilot: (1) build AMBER on CSD3, (2) run one tiny box with AMBER's own 21cm module switched on,
(3) see what it writes to `output/21cm/`, (4) time one real-size run. Nothing here uses 21cmFAST or touches
the kSZ code.

What is vendored here (from ksz-pipeline, branch amber-integration; copied, not imported):
- `external/amber/0001-dump-fields-for-ksz-pipeline.patch`, `Makefile.ksz`, `healpix_stub.f90`: unchanged.
- `external/amber/build_amber_tifr_reference.sh`: the original build script (TIFR paths), for reference only.
- `external/amber/build_amber_csd3.sh`: NEW, same recipe with environment modules instead of oneAPI setvars.
- `scripts/amber/make_amber_input.py`: as received, plus NEW flags `--h21cm write`, `--h21zmin/--h21zmax/--h21zdel`.
- `slurm/amber_run.sbatch`: NEW, runs one AMBER realisation.

Untested on CSD3 (no Intel compiler in my sandbox). The Python and shell syntax were checked, and the default
`input.txt` is byte-identical to before when `--h21cm` is not given.

Use the RDS folder, not home:  `R=~/rds/rds-dirac-dp145-uYnnMXSp128/dc-upad3`

## 1. Compiler check (free)

    . /etc/profile.d/modules.sh
    module purge
    module load rhel8/default-icl
    which ifort ifx gfortran ; echo "MKLROOT=$MKLROOT"
    module avail intel 2>&1 | head -40
    module avail mkl 2>&1 | head

If `ifort` is missing or `MKLROOT` is empty, the build script says what to add via `AMBER_MODULES=...`.

## 2. Build AMBER (login node; a minute or two)

    cd $R && git clone https://github.com/swanith01/ksz2-21cm.git     # after the step-23 files are pushed
    bash $R/ksz2-21cm/external/amber/build_amber_csd3.sh $R/amber

Result: `$R/amber/src/amber.x`.

## 3. Python for the input generator (login node)

Needs numpy, scipy, astropy and camb. Python on CSD3 may be old (`python3 --version`). If it is below 3.9, look at
`module avail python` or use conda. Then:

    python3 -m venv $R/amber_env && source $R/amber_env/bin/activate
    pip install numpy scipy astropy camb

(Alternative: make the run folder on usha, where numpy and astropy exist, and `scp -r` it; it is small.)

## 4. Tiny pilot with AMBER's 21cm module ON (a few minutes)

    python3 $R/ksz2-21cm/scripts/amber/make_amber_input.py --out $R/amber_runs/tiny21 \
        --L 64 --N 32 --zmid 8 --zdel 4 --zasy 3 --ncore 16 \
        --czmin 5 --czmax 12 --czdel 1 --h21cm write --h21zmin 6 --h21zmax 12 --h21zdel 1
    cd $R && AMBER_X=$R/amber/src/amber.x sbatch --time=00:30:00 ksz2-21cm/slurm/amber_run.sbatch $R/amber_runs/tiny21
    squeue -u $USER

## 5. What to bring back

- the `amber_run_<id>.out` file (written in the folder you submitted from);
- `ls -la $R/amber_runs/tiny21/output/21cm`;
- for the biggest file there: `file <name>` and `head -c 300 <name> | od -c | head`;
- `sacct -j <id> --format=JobID,Elapsed,TotalCPU,MaxRSS,State`.

## 6. Real-size timing run (only after step 5 looks sane)

    python3 .../make_amber_input.py --out $R/amber_runs/q256 --L 256 --N 128 --zmid 8 --zdel 4 --zasy 3 --ncore 16 \
        --czmin 4.5 --czmax 18.5 --czdel 0.5 --h21cm write
    AMBER_X=... sbatch --time=02:00:00 ksz2-21cm/slurm/amber_run.sbatch $R/amber_runs/q256

Wall time times 16 cores is the cost of one run. Multiply by 30 for 3 histories x 10 seeds and compare with
`mybalance`. DP145-CPU is a shared project pool.

## What the pilot decides

Whether AMBER's own 21cm output exists, in what format, on which redshifts, and on which grid. If it is usable and
lines up with the density and velocity snapshots, the next step (not built yet) is a reader plus a stitcher that writes
the same npz contract as our 21cmFAST lightcones. If not, the plan needs rethinking.

## Pilot result (20 Oct 2026, CSD3, job on node cpu-q-141, run `tiny21`: L=64, N=32, ncore 16)

- AMBER built (ifort 2021.6.0 + MKL 2022.1.0) and ran to "AMBER completed", exit code 0, wall time 1 s.
- Everything lives in `~/rds/rds-dirac-dp145-uYnnMXSp128/dc-upad3/` (own folder, not the shared project top level).
- With `--h21cm write --h21zmin 6 --h21zmax 12 --h21zdel 1`, `output/21cm/` holds:
  - `Tb_L=64_N=32_z=06.50 ... 11.50.dat`: six files, each exactly 4*N^3 = 131072 bytes, so raw float32, no header.
    Redshifts are the shell midpoints inside [zmin, zmax]; z=5.5 is missing because 5.5 < zmin. For a real run use
    `--h21zmin 5` so all seven midpoints (5.5 ... 11.5) are written.
  - `power_z=XX.XX.txt` (1445 bytes each) and `21cm_global.txt` (columns z, x_H, T_b; 0.01 steps in z).
- Mean Tb [mK] by z: 6.5: 0 (fully ionized), 7.5: 8.1, 8.5: 18.6, 9.5: 25.1, 10.5: 29.1, 11.5: 31.7. Min 0, max 38 to 64.
  Monotonic and of the expected size (about 30 mK for a fully neutral box at z=11.5).
- `output/cmb/fields_z=XX.XX.dat` exist for all seven midpoints (5.5 ... 11.5), 524292 bytes = 4 + 16*N^3, matching the
  layout read by the vendored `ksz_pipeline/amber/io.py` (z, rho2, mom2).
- The log calls the 21cm model "21cm simple" and uses the same density and velocity meshes as the snapshots.

Alignment check (`scripts/amber/check_amber_21cm.py`, first run, step 24 version):
- Mean of each Tb box equals AMBER's own global T_b to 0.1-0.2% at every z (e.g. z=11.5: 31.73 vs 31.65 mK).
- Tb file axis order is Fortran ('F', like the field files). Correlation with rho2 in F order:
  z=7.5 -0.45, 8.5 -0.40, 9.5 -0.11, 10.5 +0.25, 11.5 +0.55; in C order |r| < 0.07 at all z. Sign change is physical
  (dense regions ionize first, so Tb is anti-correlated with rho once the box is partly ionized). The step-24 script
  picked the order by the larger r instead of the larger |r|, which mislabelled z <= 9.5 as 'C'; fixed in step 25.
- Tb / rho2 is NOT constant across neutral cells (scatter 6% at z=11.5, 40% at z=8.5-9.5), so the ionization field is
  either fractional per cell or Tb uses a slightly different density. Not yet resolved.
- AMBER's global T_b / (x_H * T0) is a constant ~1.08 at all z (1.082 to 1.085) against the textbook T0 (T_S >> T_CMB, Yp 0.245,
  Ob=0.04897, h=0.6766, Om=0.3111): an overall normalisation difference of ~8% in T21 (~17% in T21^2). Source of the
  prefactor still to be read from `amber/src/h21cm*.f90`.

What the AMBER source says (`src/h21cm.f90`, `src/h21cmreion.f90`, read 20 Oct 2026):
- The 21cm module loops over bins [zmin, zmax] in steps of zdel and works at each bin MIDPOINT (z = zmin + zdel/2, ...),
  Nz = ceiling((zmax-zmin)/zdel). For production set `--h21zmin/--h21zmax/--h21zdel` equal to `--czmin/--czmax/--czdel`
  so every CMB shell has a Tb box at the same z. (The pilot used czmin 5 but h21zmin 6, hence no z=5.5 box.)
- Physics: "Assume Ts >> Tcmb. Ignore peculiar vel, lightcone effects." So no spin-temperature evolution, NO redshift-space
  distortions. Check that this matches the 21cmFAST switches used for the comparison runs.
- T0 = 28 mK (Ob h^2/0.022) sqrt(0.15/(Om h^2)) sqrt((1+z)/10)  (Madau et al. 1997 fit). This is ~8% above the exact
  value (norm = 1.081 at every z in the pilot). A fixed scalar; irrelevant for correlation coefficients, +17% on T21^2.
- Ionization is BINARY per cell: neutral if z > zre(i,j,k) (from the reion zre field), else Tb = 0.
- Two maps are made: Tb1 = T0 * rho1 (real space) and Tb2 = T0 * rho2 (used for AMBER's own P_TT in power_z files).
  The kSZ field dump uses rho2 (interlaced + deconvolved). The Tb/rho2 spread seen in the pilot (p99 +20%, max up to 3x)
  fits the .dat box being the rho1 one (min rho1 = 0 in the log). Whether this matters at l <= 1600 (k < ~0.2 /Mpc,
  roughly a fifth of the grid Nyquist) is tested by the P(k) comparison in `check_amber_21cm.py` (step 26).

Cost: the tiny box took 1 s, so it says nothing about L=256, N=128 (64x more cells). Time that next.

## Real-size timing result (20 Oct 2026, run `q256`: L=256 Mpc/h, N=128, 28 shells z=4.75...18.25, ncore 16)

- AMBER wall time 11 s (exit 0), about 0.05 core-hours. 28 Tb boxes + 28 power files + global history in `output/21cm` (225 MB),
  28 field dumps in `output/cmb` (898 MB), zre 8 MB: about 1.1 GB per run. Compute is not the limit; disk and queue are.
- Plan: AMBER at L=512 Mpc/h, N=256 (757 Mpc box, 2.96 Mpc cells) is almost our 800 Mpc / 256^3 21cmFAST box. About 9.5 GB per
  run raw (28 x 268 MB fields + 28 x 67 MB Tb), so stitch, then delete the raw fields (`--delete-raw`).

## Stage 3: reader + stitcher (step 27)

New files (everything else, including scripts 02 and 04 and the kSZ code, is untouched):
- `src/ksz2_21cm/simulate/amber_io.py`: readers (fields and zre vendored from ksz-pipeline, Tb reader new), snapshot matching.
- `src/ksz2_21cm/simulate/stitch_from_amber.py`: same `stitch_field` (verbatim) and the same wrap-cycle seed for all four fields.
- `scripts/01a_stitch_lightcones_amber.py`: one AMBER run directory per seed -> `cache_dir/seed_N/stitched_lightcone_seedN.npz`
  with the same keys as `01_stitch_lightcones.py` (density, xH_box, velocity_z, brightness_temp, z_arr, pos_axis) plus metadata.
- `configs/variants/amber_q512.yaml`: box_len = 512/0.6766 Mpc, hii_dim 256. 01a asserts these match the AMBER files.
- `tests/test_amber_stitch.py`: synthetic AMBER output in the real byte layout; checks axis order, contract, xH direction,
  shared rotation (breaks if Tb is stitched with another seed), tb_scale, and the script end to end.
  Also run once by hand: 01a, then the unchanged `02_compute_ksz_maps_stitched.py` and `04_compute_cross_corr_sq_stitched.py`
  accept the file.

Native conventions are kept: Tb is AMBER's (Madau T0, 1.081 x exact, no RSD, Ts >> Tcmb, from rho1). `--tb-scale 0.925` applies
1/1.081 explicitly, and `tb_native_over_exact` is stored in every npz.

Run on CSD3 (compute node: about 10 GB RAM and 10 to 15 min per seed at N=256):

    source $R/amber_env/bin/activate; pip install pyyaml matplotlib
    python scripts/01a_stitch_lightcones_amber.py --config configs/variants/amber_q512.yaml \
        --run-dir-template $R/amber_runs/q512 --seeds 1
    python scripts/02_compute_ksz_maps_stitched.py --config configs/variants/amber_q512.yaml
    python scripts/04_compute_cross_corr_sq_stitched.py --config configs/variants/amber_q512.yaml

(The template has no {seed} here because the test run is not in a seed_N folder; production runs use `.../seed_{seed}`.)
Scripts 02 and 04 loop over `simulation.random_seeds` in the config and skip seeds without a file, so a one-seed test is fine.
