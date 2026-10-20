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
