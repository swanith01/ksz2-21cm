#!/bin/bash
#SBATCH -J heff_post
#SBATCH -A DIRAC-DP145-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/heff_post_%A_%a.out
#SBATCH --error=logs/heff_post_%A_%a.out
# After slurm/heff_production.sbatch: Ma statistics (04b, needs all seeds of a history in one config) for
# ONE history per array task, then the comparison plots of every history finished so far (22, light).
# Submit after the production finishes (from $R):
#   cd $R && sbatch ksz2-21cm/slurm/heff_post.sbatch
# or chain it straight away:  sbatch --dependency=afterany:<production job id> ksz2-21cm/slurm/heff_post.sbatch
set -o pipefail
R=${R:-$HOME/rds/rds-dirac-dp145-uYnnMXSp128/dc-upad3}
read -r -a HA <<< "${HEFFS:-17 30 52}"
read -r -a SA <<< "${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
H=${HA[${SLURM_ARRAY_TASK_ID:?submit with sbatch}]}
[ -n "$H" ] || { echo "task outside HEFFS"; exit 2; }
DATA=$R/ksz2-21cm-data/heff_$H
CFG=$R/ksz2-21cm-data/configs/heff_${H}_all.yaml
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} PBS_NCPUS=${SLURM_CPUS_PER_TASK:-16} NCPUS=${SLURM_CPUS_PER_TASK:-16} PYTHONNOUSERSITE=1
source "$R/miniforge3/etc/profile.d/conda.sh" && conda activate "$R/envs/ksz2-21cm" || { echo "FAILED: conda env"; exit 1; }
cd "$R/ksz2-21cm" || exit 1
SEEDLIST=$(IFS=,; echo "${SA[*]}")
python3 - "$CFG" "$R/ksz2-21cm-data/configs/heff_${H}_s${SA[0]}.yaml" "$SEEDLIST" <<'PY' || { echo "FAILED: config"; exit 1; }
import sys, yaml
out, base, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
c = yaml.safe_load(open(base))
c["simulation"]["random_seeds"] = seeds
c["run_name"] = c["run_name"].rsplit("_s", 1)[0] + "_all"
yaml.safe_dump(c, open(out, "w"), sort_keys=False)
print("wrote", out, "seeds", seeds)
PY
echo "=== 04b Ma statistics, HII_EFF=$H ==="
python scripts/04b_compute_cross_corr_ma_stitched.py --config "$CFG" --z0-min 4.5 --z0-max 13.5 --workers 4 || echo "FAILED: 04b HII_EFF=$H"
echo "=== 22 comparison plots (every finished history) ==="
python scripts/22_heff_history_compare.py --data-root "$R/ksz2-21cm-data" || echo "FAILED: 22"
echo "DONE HII_EFF=$H  $(date)"
