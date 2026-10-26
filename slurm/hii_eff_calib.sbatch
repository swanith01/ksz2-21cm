#!/bin/bash
#SBATCH -J heff_calib
#SBATCH -A DIRAC-DP145-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --array=0-8%3
#SBATCH --output=logs/heff_calib_%A_%a.out
#SBATCH --error=logs/heff_calib_%A_%a.out
# HII_EFF calibration: one array task per HII_EFF_FACTOR value, seed 1, 800 Mpc / 256^3, coeval boxes at
# z = 3.0..20 (scripts/21). Each task writes a small .npz; the .h5 cache (~30 GB per task) is deleted at
# the end of the task, and %3 keeps at most 3 tasks (~100 GB) on disk at once.
# Afterwards (login node is fine, it is light):
#   python ksz2-21cm/scripts/21b_hii_eff_summary.py --in-dir $R/ksz2-21cm-data/hii_eff_calib
# Submit (from $R):   cd $R && mkdir -p logs && sbatch ksz2-21cm/slurm/hii_eff_calib.sbatch
# Other values:       sbatch --array=0-2 --export=ALL,HEFFS="18 22 45" ksz2-21cm/slurm/hii_eff_calib.sbatch
R=${R:-$HOME/rds/rds-dirac-dp145-uYnnMXSp128/dc-upad3}
read -r -a HA <<< "${HEFFS:-10 15 20 25 30 40 55 75 100}"
H=${HA[${SLURM_ARRAY_TASK_ID:?submit with sbatch}]}
[ -n "$H" ] || { echo "task $SLURM_ARRAY_TASK_ID outside HEFFS (${#HA[@]})"; exit 2; }
source "$R/miniforge3/etc/profile.d/conda.sh"
conda activate "$R/envs/ksz2-21cm"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$R/ksz2-21cm" || exit 1
echo "task $SLURM_ARRAY_TASK_ID: HII_EFF_FACTOR=$H  node $(hostname)  $(date)"
python scripts/21_hii_eff_calibration.py --config configs/variants/stitched_256.yaml --hii-eff $H --seed 1 \
    --out-dir "$R/ksz2-21cm-data/hii_eff_calib" || { echo "FAILED: HII_EFF=$H"; exit 1; }
echo "DONE HII_EFF=$H  $(date)"
