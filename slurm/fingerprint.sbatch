#!/bin/bash
#SBATCH -J fingerprint
#SBATCH -A DIRAC-DP145-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/fingerprint_%j.out
#SBATCH --error=logs/fingerprint_%j.out
# One 21cmFAST coeval box (seed 1, z = 8, 800 Mpc / 256^3) with CSD3's new 21cmFAST environment,
# fingerprinted by scripts/20 for comparison with the same run on swarm (pbs/fingerprint.pbs).
# Everything stays in the personal folder; the .h5 cache is deleted at the end.
# Submit (from $R):   cd $R && mkdir -p logs && sbatch ksz2-21cm/slurm/fingerprint.sbatch
R=${R:-$HOME/rds/rds-dirac-dp145-uYnnMXSp128/dc-upad3}
source "$R/miniforge3/etc/profile.d/conda.sh"
conda activate "$R/envs/ksz2-21cm"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
cd "$R/ksz2-21cm" || exit 1
SEED=${SEED:-1}; Z=${Z:-8.0}
python scripts/20_coeval_fingerprint.py --config configs/variants/stitched_256.yaml --seed $SEED --z $Z \
    --work-dir "$R/ksz2-21cm-data/fingerprint_h5" --out "$R/logs/fingerprint_seed${SEED}_z${Z}_csd3.txt"
echo "DONE $(date)"
