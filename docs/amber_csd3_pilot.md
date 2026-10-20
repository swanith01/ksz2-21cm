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

Cost: the tiny box took 1 s, so it says nothing about L=256, N=128 (64x more cells). Time that next.
