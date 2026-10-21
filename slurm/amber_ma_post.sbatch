#!/bin/bash
#SBATCH -J amber_ma
#SBATCH -A DIRAC-DP145-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/amber_ma_%A_%a.out
#SBATCH --error=logs/amber_ma_%A_%a.out
# Ma-style kSZ x 21cm (04b) for the AMBER production, ONE ARRAY TASK PER HISTORY.
# Why separate: 04b needs >= 2 seeds in its config (its null test pairs each seed with a neighbour's kSZ map), whereas
# slurm/amber_production.sbatch runs one seed per task, so 04b stopped there ("need >= 2 seeds"). Everything else in
# that production finished.  Here the config lists ALL seeds of the history; the stitched files and kSZ maps already exist.
# Submit (from $R):   cd $R && mkdir -p logs && git -C ksz2-21cm pull && sbatch ksz2-21cm/slurm/amber_ma_post.sbatch
# Other sets:  sbatch --array=0-0 --export=ALL,HISTS="8",SEEDS="1 2 3 4 5 6 7 8 9 10" ksz2-21cm/slurm/amber_ma_post.sbatch
# Dry run:     DRY=1 SLURM_ARRAY_TASK_ID=1 bash <this file>
set -o pipefail
R=${R:-$HOME/rds/rds-dirac-dp145-uYnnMXSp128/dc-upad3}
read -r -a HIST_A <<< "${HISTS:-7 8 9}"
read -r -a SEED_A <<< "${SEEDS:-1 2 3 4 5}"
TID=${SLURM_ARRAY_TASK_ID:?submit with sbatch}
H=${HIST_A[$TID]}
[ -n "$H" ] || { echo "task $TID outside HISTS (${#HIST_A[@]})"; exit 2; }
DATA=$R/ksz2-21cm-data/amber_q512_h$H
CFG=$R/ksz2-21cm-data/configs/amber_q512_h${H}_all.yaml
run() { if [ -n "$DRY" ]; then echo "[dry] $*"; else "$@"; fi; }
echo "task $TID  history z_mid=$H  seeds ${SEED_A[*]}  node $(hostname)  $(date)"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} PBS_NCPUS=${SLURM_CPUS_PER_TASK:-16} NCPUS=${SLURM_CPUS_PER_TASK:-16}
if [ -z "$DRY" ]; then
  type module >/dev/null 2>&1 || { [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh; }
  type module >/dev/null 2>&1 && module load python/3.9.12/gcc/pdcqf4o5
  [ -f "$R/amber_env/bin/activate" ] && . "$R/amber_env/bin/activate"
fi
cd "$R/ksz2-21cm" || exit 1
mkdir -p "$(dirname "$CFG")"
SEEDLIST=$(IFS=,; echo "${SEED_A[*]}")
if [ -z "$DRY" ]; then
python3 - "$CFG" "$DATA" "$SEEDLIST" "$H" <<'PY' || exit 1
import sys, yaml
cfg_path, data, seeds, h = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")], sys.argv[4]
c = yaml.safe_load(open("configs/variants/amber_q512.yaml"))
c["paths"]["cache_dir"] = data + "/cache"
c["paths"]["plot_dir"] = data + "/plots_all"
c["simulation"]["random_seeds"] = seeds
c["run_name"] = "amber_q512_h%s_all" % h
yaml.safe_dump(c, open(cfg_path, "w"), sort_keys=False)
print("wrote", cfg_path, "seeds", seeds)
PY
else echo "[dry] write $CFG (cache $DATA/cache, random_seeds [$SEEDLIST])"; fi
run python scripts/04b_compute_cross_corr_ma_stitched.py --config "$CFG" --workers 4 || { echo "FAILED: 04b"; exit 1; }
echo "DONE task $TID (z_mid=$H)  $(date)"
