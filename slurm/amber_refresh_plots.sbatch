#!/bin/bash
#SBATCH -J amber_plots
#SBATCH -A DIRAC-DP145-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/amber_plots_%j.out
#SBATCH --error=logs/amber_plots_%j.out
# =============================================================================
# Refresh every AMBER-history plot from ALL seeds, then pack the plots into ONE small tarball to copy to usha.
#   1. Ma-style kSZ x 21cm (04b) for each history with all seeds (finished seeds are cached, so only new ones cost time)
#   2. 17_amber_history_compare.py   (kSZ^2 x 21cm^2 vs z0, all histories, chi-norm z0)
#   3. 08_plot_ma18_style.py per history  (Ma-style figures)
#   4. tar -> $R/ksz2-21cm-data/amber_plots.tgz   (png, pdf, txt only; layout below)
# Submit (from $R):   cd $R && mkdir -p logs && git -C ksz2-21cm pull && sbatch ksz2-21cm/slurm/amber_refresh_plots.sbatch
# Other sets:         sbatch --export=ALL,HISTS="7 8 9",SEEDS="1 2 3 4 5 6 7 8 9 10" ksz2-21cm/slurm/amber_refresh_plots.sbatch
# Dry run:            DRY=1 bash ksz2-21cm/slurm/amber_refresh_plots.sbatch
# A step that fails is reported ("FAILED: ...") and the job carries on, so one bad plot never blocks the others.
# Tarball layout:  history_compare/*   h7/*   h8/*   h9/*      (h<N>/ = the Ma-style plots of that history)
# =============================================================================
set -o pipefail
R=${R:-$HOME/rds/rds-dirac-dp145-uYnnMXSp128/dc-upad3}
read -r -a HIST_A <<< "${HISTS:-7 8 9}"
read -r -a SEED_A <<< "${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
SEEDLIST=$(IFS=,; echo "${SEED_A[*]}")
DATAROOT=$R/ksz2-21cm-data
TGZ=$DATAROOT/amber_plots.tgz
run() { if [ -n "$DRY" ]; then echo "[dry] $*"; else "$@"; fi; }
stage() { echo; echo "=== $(date +%H:%M:%S)  $* ==="; }
echo "histories ${HIST_A[*]}  seeds ${SEED_A[*]}  node $(hostname)  $(date)"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} PBS_NCPUS=${SLURM_CPUS_PER_TASK:-16} NCPUS=${SLURM_CPUS_PER_TASK:-16}
if [ -z "$DRY" ]; then
  type module >/dev/null 2>&1 || { [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh; }
  type module >/dev/null 2>&1 && module load python/3.9.12/gcc/pdcqf4o5
  [ -f "$R/amber_env/bin/activate" ] && . "$R/amber_env/bin/activate"
fi
cd "$R/ksz2-21cm" || exit 1
mkdir -p "$DATAROOT/configs"

for H in "${HIST_A[@]}"; do
  DATA=$DATAROOT/amber_q512_h$H
  CFG=$DATAROOT/configs/amber_q512_h${H}_all.yaml
  stage "history z_mid=$H : config + Ma-style (04b), seeds $SEEDLIST"
  if [ -z "$DRY" ]; then
python3 - "$CFG" "$DATA" "$SEEDLIST" "$H" <<'PY' || { echo "FAILED: config h$H"; continue; }
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
  else echo "[dry] write $CFG (cache $DATA/cache, plot_dir $DATA/plots_all, seeds [$SEEDLIST])"; fi
  run python scripts/04b_compute_cross_corr_ma_stitched.py --config "$CFG" --workers 4 || echo "FAILED: 04b h$H"
done

stage "17_amber_history_compare.py (all histories, chi-norm z0)"
run python scripts/17_amber_history_compare.py --data-root "$DATAROOT" --chi-norm z0 || echo "FAILED: 17"

for H in "${HIST_A[@]}"; do
  CFG=$DATAROOT/configs/amber_q512_h${H}_all.yaml
  stage "08_plot_ma18_style.py  history z_mid=$H"
  run python scripts/08_plot_ma18_style.py --config "$CFG" || echo "FAILED: 08 h$H"
done

stage "pack plots -> $TGZ"
if [ -n "$DRY" ]; then echo "[dry] tar czf $TGZ  history_compare/ and h<N>/ (png, pdf, txt)"; else
  STAGE=$(mktemp -d "$DATAROOT/.pack_XXXXXX") || exit 1
  mkdir -p "$STAGE/history_compare"
  cp "$DATAROOT"/amber_history_compare/*.{png,pdf,txt} "$STAGE/history_compare/" 2>/dev/null
  for H in "${HIST_A[@]}"; do
    mkdir -p "$STAGE/h$H"
    cp "$DATAROOT/amber_q512_h$H/plots_all/"*.{png,pdf,txt} "$STAGE/h$H/" 2>/dev/null
  done
  tar czf "$TGZ" -C "$STAGE" . && echo "packed:" && tar tzf "$TGZ" | sed 's/^/  /'
  rm -rf "$STAGE"
  ls -l "$TGZ"
fi
echo; echo "DONE  $(date)   (check the log for lines starting FAILED:)"
