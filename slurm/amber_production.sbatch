#!/bin/bash
#SBATCH -J amber_prod
#SBATCH -A DIRAC-DP145-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-14%5
#SBATCH --output=logs/amber_prod_%A_%a.out
#SBATCH --error=logs/amber_prod_%A_%a.out
# =============================================================================
# AMBER production: one array task = one (history, seed). Per task:
#   make AMBER input -> run AMBER (about 1 min at L=512, N=256) -> stitch lightcone (delete the raw fields) ->
#   kSZ map (02) -> kSZ^2 x 21cm^2 (04) -> Ma-style kSZ x 21cm (04b).
# Scripts 02, 04, 04b are the existing ones, unchanged, driven by a per-task copy of configs/variants/amber_q512.yaml
# with random_seeds = [this seed], so tasks never touch each other's files. Re-running a finished task is a no-op for
# the expensive steps (a stitched file that exists is kept).
#
# Task id -> (history, seed):  history = HISTS[id / NSEEDS],  seed = SEEDS[id % NSEEDS].
# Defaults (PROPOSAL, change on the sbatch line):  HISTS="7 8 9"  (AMBER z_mid, mass-weighted; zdel 4, zasy 3)
#                                                  SEEDS="1 2 3 4 5"    -> --array=0-14
# Submit (from $R, so that logs/ is relative to it):
#     cd $R && mkdir -p logs && git -C ksz2-21cm pull && sbatch ksz2-21cm/slurm/amber_production.sbatch
# Other sets, e.g. one history and ten seeds:
#     sbatch --array=0-9%5 --export=ALL,HISTS="8",SEEDS="1 2 3 4 5 6 7 8 9 10" ksz2-21cm/slurm/amber_production.sbatch
# Dry run (prints the commands, runs nothing; needs SLURM_ARRAY_TASK_ID):  DRY=1 SLURM_ARRAY_TASK_ID=7 bash <this file>
# Results: $R/ksz2-21cm-data/v2/amber_q512_h<ZMID>/cache/seed_<N>/ (VARIANT=v2 default; stitcher v2, see 01a) ;
# summary:  python ksz2-21cm/scripts/amber/summarize_production.py --data-root $R/ksz2-21cm-data/v2
# =============================================================================
set -o pipefail
R=${R:-$HOME/rds/rds-dirac-dp145-uYnnMXSp128/dc-upad3}
read -r -a HIST_A <<< "${HISTS:-7 8 9}"
read -r -a SEED_A <<< "${SEEDS:-1 2 3 4 5}"
NS=${#SEED_A[@]}
TID=${SLURM_ARRAY_TASK_ID:?submit with sbatch (array task id missing)}
H=${HIST_A[$((TID / NS))]}
SEED=${SEED_A[$((TID % NS))]}
if [ -z "$H" ] || [ -z "$SEED" ]; then echo "task $TID is outside HISTS x SEEDS (${#HIST_A[@]} x $NS)"; exit 2; fi
ZDEL=${ZDEL:-4}; ZASY=${ZASY:-3}; L=${L:-512}; N=${N:-256}
TAG=h${H}
# VARIANT keeps the stitcher-v2 products apart from the Oct 21-23 (v1) ones: $R/ksz2-21cm-data/<VARIANT>/amber_q512_h<ZMID>
# (plot scripts: --data-root $R/ksz2-21cm-data/<VARIANT>). The raw dumps of the old runs were deleted after stitching, so
# v2 re-runs AMBER (about 1 min per run) in its own run directories.
VARIANT=${VARIANT:-v2}
RUN=$R/amber_runs/prod_$VARIANT/$TAG/seed_$SEED
DATA=$R/ksz2-21cm-data/$VARIANT/amber_q512_$TAG
CFG=$R/ksz2-21cm-data/configs/amber_q512_${TAG}_s${SEED}.yaml
NPZ=$DATA/cache/seed_$SEED/stitched_lightcone_seed$SEED.npz
AMBER_X=${AMBER_X:-$R/amber/src/amber.x}
AMBER_MODULES="rhel8/default-icl intel-oneapi-mkl/2022.1.0/intel/mngj3ad6"

run() { if [ -n "$DRY" ]; then echo "[dry] $*"; else "$@"; fi; }
stage() { echo; echo "=== $(date +%H:%M:%S)  $* ==="; }
echo "task $TID  history z_mid=$H (zdel $ZDEL, zasy $ZASY)  seed $SEED  L=$L N=$N  node $(hostname)  $(date)"
echo "run dir $RUN"; echo "data    $DATA"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export PBS_NCPUS=${SLURM_CPUS_PER_TASK:-16} NCPUS=${SLURM_CPUS_PER_TASK:-16}   # the scripts read these, not Slurm's

python_env() {
  type module >/dev/null 2>&1 || { [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh; }
  type module >/dev/null 2>&1 && module load python/3.9.12/gcc/pdcqf4o5
  [ -f "$R/amber_env/bin/activate" ] && . "$R/amber_env/bin/activate"
}
[ -n "$DRY" ] || python_env
cd "$R/ksz2-21cm" || exit 1

# ---- 1. AMBER (skipped when the stitched file already exists) ----
if [ ! -s "$NPZ" ]; then
  stage "AMBER input + run"
  if ! grep -qs "AMBER completed" "$RUN/output/log.txt" || ! ls "$RUN"/output/cmb/fields_z=*.dat >/dev/null 2>&1; then
    run python3 scripts/amber/make_amber_input.py --out "$RUN" --L $L --N $N --zmid "$H" --zdel $ZDEL --zasy $ZASY \
        --seed "$SEED" --ncore 16 --czmin 4.5 --czmax 18.5 --czdel 0.5 --h21cm write --h21zmin 4.5 --h21zmax 18.5 --h21zdel 0.5 \
      || { echo "FAILED: make_amber_input"; exit 1; }
    AMBER_MODULES="$AMBER_MODULES" AMBER_X="$AMBER_X" run bash slurm/amber_run.sbatch "$RUN" \
      || { echo "FAILED: AMBER"; exit 1; }
    [ -n "$DRY" ] || python_env                       # amber_run.sbatch purges modules
  else
    echo "AMBER output already complete in $RUN"
  fi
fi

# ---- 2. per-task config ----
stage "config $CFG"
mkdir -p "$(dirname "$CFG")"
if [ -z "$DRY" ]; then
python3 - "$CFG" "$DATA" "$SEED" <<'PY' || exit 1
import sys, yaml
cfg_path, data, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = yaml.safe_load(open("configs/variants/amber_q512.yaml"))
c["paths"]["cache_dir"] = data + "/cache"
c["paths"]["plot_dir"] = data + "/plots_seed%d" % seed
c["simulation"]["random_seeds"] = [seed]
c["run_name"] = "amber_q512_" + data.rsplit("_", 1)[-1] + "_s%d" % seed
yaml.safe_dump(c, open(cfg_path, "w"), sort_keys=False)
print("wrote", cfg_path)
PY
else echo "[dry] write $CFG (cache $DATA/cache, random_seeds [$SEED])"; fi

# ---- 3. stitch (deletes the raw field dumps only after the stitched file is written and readable) ----
stage "stitch lightcone"
run python scripts/01a_stitch_lightcones_amber.py --config "$CFG" --run-dir-template "$RUN" --seeds "$SEED" --delete-raw \
  || { echo "FAILED: stitch"; exit 1; }

# ---- 4. kSZ map, cross-correlations ----
stage "kSZ map (02)";                 run python scripts/02_compute_ksz_maps_stitched.py --config "$CFG"        || { echo "FAILED: 02"; exit 1; }
stage "kSZ^2 x 21cm^2 (04)";          run python scripts/04_compute_cross_corr_sq_stitched.py --config "$CFG"   || { echo "FAILED: 04"; exit 1; }
stage "Ma-style kSZ x 21cm (04b)";    run python scripts/04b_compute_cross_corr_ma_stitched.py --config "$CFG" --workers 1 || { echo "FAILED: 04b"; exit 1; }

stage "DONE task $TID (z_mid=$H seed=$SEED)"
[ -n "$DRY" ] || du -sh "$DATA/cache/seed_$SEED" "$RUN" 2>/dev/null
exit 0
