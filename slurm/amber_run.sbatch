#!/bin/bash
#SBATCH -J amber_run
#SBATCH -A DIRAC-DP145-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --mem=32000
#SBATCH --output=amber_run_%j.out
#SBATCH --error=amber_run_%j.out
# Run ONE AMBER realisation (one run directory made by scripts/amber/make_amber_input.py).
#   sbatch slurm/amber_run.sbatch <run_dir>
# Needs the AMBER_X environment variable (path to amber.x). Override resources on the command line, e.g.
#   sbatch --time=04:00:00 --cpus-per-task=32 --mem=60000 slurm/amber_run.sbatch <run_dir>
# (then make the run with  --ncore 32  so AMBER's thread count matches). The .out file lands in the
# directory you submit from. Billing is about cpus x elapsed hours on DIRAC-DP145-CPU.
RUN=${1:?usage: sbatch slurm/amber_run.sbatch <run_dir>}
AMBER_X=${AMBER_X:?set AMBER_X=/path/to/amber.x before sbatch (it is passed on by default)}
AMBER_MODULES=${AMBER_MODULES:-rhel8/default-icl}

. /etc/profile.d/modules.sh
module purge
for m in $AMBER_MODULES; do module load "$m"; done
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$RUN" || exit 1
NC=$(grep -m1 'Number of core' input/input.txt | awk '{print $1}')
echo "run dir: $RUN   threads: $OMP_NUM_THREADS   input.txt ncore: $NC   node: $(hostname)"
[ "$NC" = "$OMP_NUM_THREADS" ] || echo "WARNING: input.txt ncore ($NC) differs from the threads given to the job ($OMP_NUM_THREADS)"
mkdir -p output/cosmo output/reion output/grf output/lpt output/esf output/mesh output/cmb output/21cm
echo "start: $(date)"
t0=$(date +%s)
"$AMBER_X" < input/input.txt > output/log.txt 2>&1
rc=$?
t1=$(date +%s)
echo "amber.x exit code: $rc    wall time: $((t1 - t0)) s    (cpu-hours billed ~ $(( (t1 - t0) * OMP_NUM_THREADS / 3600 )) )"
echo "--- output/21cm ---";  ls -la output/21cm | head -40
echo "--- output/cmb (first) ---"; ls output/cmb | head -12
echo "--- sizes ---"; du -sh output output/* 2>/dev/null
echo "--- log tail ---"; tail -25 output/log.txt
echo "After it finishes:  seff $SLURM_JOB_ID   or   sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,TotalCPU,MaxRSS,State"
exit $rc
