#!/bin/bash
#SBATCH -J heff_prod
#SBATCH -A DIRAC-DP145-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --array=0-29%6
#SBATCH --output=logs/heff_prod_%A_%a.out
#SBATCH --error=logs/heff_prod_%A_%a.out
# =============================================================================
# 21cmFAST reionization-history scan, production: one array task = one (HII_EFF_FACTOR, seed).
# Per task: write the task's config -> stitch the lightcone (01, astro_params = {HII_EFF_FACTOR: H},
# snapshots z = 3.0..20 so a late history's patchy window ends inside the lightcone) -> save x_e(z) (22a)
# -> kSZ map (02) -> Zhou kSZ^2 x 21cm^2 (04). The Ma statistics (04b) need >= 2 seeds per config, so they
# run afterwards in slurm/heff_post.sbatch, which also makes the comparison plots (22).
# Values from the HII_EFF calibration (scripts/21b): 17 -> z50 ~ 6.7, 30 -> 7.76 (21cmFAST default),
# 52 -> ~8.75. 800 Mpc / 256^3, same as the swarm fiducial (validated to reproduce it bit for bit).
#
# Task id -> HII_EFF = HEFFS[id / NSEEDS], seed = SEEDS[id % NSEEDS]
# Submit (from $R):  cd $R && mkdir -p logs && sbatch ksz2-21cm/slurm/heff_production.sbatch
# Other sets:        sbatch --array=0-9%5 --export=ALL,HEFFS="40",SEEDS="1 2 3 4 5 6 7 8 9 10" ksz2-21cm/slurm/heff_production.sbatch
# Dry run:           DRY=1 SLURM_ARRAY_TASK_ID=12 bash ksz2-21cm/slurm/heff_production.sbatch
# Re-running a finished task is cheap: 01 keeps a verified stitched file, 02/04 reuse their caches.
# Disk: ~35 GB of 21cmFAST .h5 per running task (deleted by 01 when its stitched file is verified),
# so %6 peaks at ~210 GB; each finished seed keeps ~2 GB.
# =============================================================================
set -o pipefail
R=${R:-$HOME/rds/rds-dirac-dp145-uYnnMXSp128/dc-upad3}
read -r -a HA <<< "${HEFFS:-17 30 52}"
read -r -a SA <<< "${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
NS=${#SA[@]}
TID=${SLURM_ARRAY_TASK_ID:?submit with sbatch}
H=${HA[$((TID / NS))]}; SEED=${SA[$((TID % NS))]}
[ -n "$H" ] && [ -n "$SEED" ] || { echo "task $TID outside HEFFS x SEEDS (${#HA[@]} x $NS)"; exit 2; }
DATA=$R/ksz2-21cm-data/heff_$H
CFG=$R/ksz2-21cm-data/configs/heff_${H}_s${SEED}.yaml
run() { if [ -n "$DRY" ]; then echo "[dry] $*"; else "$@"; fi; }
stage() { echo; echo "=== $(date +%H:%M:%S)  $* ==="; }
echo "task $TID  HII_EFF_FACTOR=$H  seed $SEED  node $(hostname)  $(date)"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} PBS_NCPUS=${SLURM_CPUS_PER_TASK:-16} NCPUS=${SLURM_CPUS_PER_TASK:-16}
export PYTHONNOUSERSITE=1
if [ -z "$DRY" ]; then
  source "$R/miniforge3/etc/profile.d/conda.sh" && conda activate "$R/envs/ksz2-21cm" || { echo "FAILED: conda env"; exit 1; }
fi
cd "$R/ksz2-21cm" || exit 1
mkdir -p "$(dirname "$CFG")"

stage "config $CFG"
if [ -z "$DRY" ]; then
python3 - "$CFG" "$DATA" "$SEED" "$H" <<'PY' || { echo "FAILED: config"; exit 1; }
import sys, yaml
cfg_path, data, seed, h = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
c = yaml.safe_load(open("configs/variants/stitched_256.yaml"))
c["run_name"] = f"heff_{h:g}_s{seed}"
c["paths"]["cache_dir"] = data + "/cache"
c["paths"]["plot_dir"] = data + "/plots"
c["simulation"]["random_seeds"] = [seed]
c["simulation"]["n_threads"] = 16
c["astro_params"] = {"HII_EFF_FACTOR": h}
c["coeval_ksz"]["z_snapshots"] = sorted(set(float(z) for z in c["coeval_ksz"]["z_snapshots"]) | {3.0, 3.5}, reverse=True)
c["cross_correlation"]["z_chunk_centres"] = [round(4.5 + 0.5 * i, 2) for i in range(18)]     # 4.5 .. 13.0
yaml.safe_dump(c, open(cfg_path, "w"), sort_keys=False)
print("wrote", cfg_path)
PY
else echo "[dry] write $CFG (HII_EFF=$H, seed $SEED, cache $DATA/cache, snapshots z=3..20, chunks 4.5..13)"; fi

stage "01 stitch";      run python scripts/01_stitch_lightcones.py --config "$CFG" --seeds $SEED --workers 1 || { echo "FAILED: 01"; exit 1; }
stage "22a x_e(z)";     run python scripts/22a_save_xe_history.py --config "$CFG" --seeds $SEED || { echo "FAILED: 22a"; exit 1; }
stage "02 kSZ map";     run python scripts/02_compute_ksz_maps_stitched.py --config "$CFG" --seeds $SEED --no-plots || { echo "FAILED: 02"; exit 1; }
stage "04 Zhou";        run python scripts/04_compute_cross_corr_sq_stitched.py --config "$CFG" --seeds $SEED || { echo "FAILED: 04"; exit 1; }
echo; echo "DONE task $TID (HII_EFF=$H, seed $SEED)  $(date)"
