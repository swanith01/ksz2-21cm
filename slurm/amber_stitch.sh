#!/bin/bash
#SBATCH -J amber_stitch
#SBATCH -A DIRAC-DP145-CPU
#SBATCH -p icelake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=amber_stitch_%j.out
#SBATCH --error=amber_stitch_%j.out
# Stitch AMBER runs into lightcones (tests first), logging EVERYTHING to $R/logs/stitch_<time>.log so nothing is
# lost when the interactive screen scrolls. Works the same two ways:
#   interactive (on a cpu-q-... node from sintr):   bash   $R/ksz2-21cm/slurm/amber_stitch.sh <run_dir_template> [seed ...]
#   batch:                                          sbatch $R/ksz2-21cm/slurm/amber_stitch.sh <run_dir_template> [seed ...]
# <run_dir_template> is an AMBER run directory, with {seed} replaced per seed, e.g. $R/amber_runs/seed_{seed}
# (a plain path with no {seed} works for a single test run). Seeds default to 1.
# Optional environment variables:  CONFIG (default configs/variants/amber_q512.yaml),
#                                  STITCH_EXTRA (e.g. "--delete-raw" or "--workers 2"), R (project folder).
# Afterwards, from ANY login tab:   tail -60 $R/logs/stitch_*.log     (or:  less -R <logfile>)
set -o pipefail
R=${R:-$HOME/rds/rds-dirac-dp145-uYnnMXSp128/dc-upad3}
TEMPLATE=${1:?usage: amber_stitch.sh <run_dir_template> [seed ...]}
shift
SEEDS=${*:-1}
CONFIG=${CONFIG:-configs/variants/amber_q512.yaml}

mkdir -p "$R/logs"
LOG="$R/logs/stitch_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "log file : $LOG"
echo "node     : $(hostname)   cpus: ${SLURM_CPUS_PER_TASK:-?}   start: $(date)"
echo "template : $TEMPLATE   seeds: $SEEDS   config: $CONFIG   extra: ${STITCH_EXTRA:-none}"

type module >/dev/null 2>&1 || { [ -f /etc/profile.d/modules.sh ] && . /etc/profile.d/modules.sh; }
type module >/dev/null 2>&1 && module load python/3.9.12/gcc/pdcqf4o5
[ -f "$R/amber_env/bin/activate" ] && . "$R/amber_env/bin/activate"
python -c "import numpy, scipy, astropy, yaml; print('python ok:', numpy.__version__)" || { echo "FAILED: python environment"; exit 1; }
python -c "import matplotlib" 2>/dev/null || pip install --quiet pyyaml matplotlib

cd "$R/ksz2-21cm" || exit 1
if [ -d .git ]; then git pull --ff-only || echo "(git pull failed; continuing with the current checkout)"; fi
git log --oneline -1 2>/dev/null

echo; echo "=== 1/2 tests ==="
python tests/test_amber_stitch.py || { echo "FAILED: tests; not stitching"; exit 1; }

echo; echo "=== 2/2 stitching ==="
t0=$(date +%s)
# shellcheck disable=SC2086
python scripts/01a_stitch_lightcones_amber.py --config "$CONFIG" --run-dir-template "$TEMPLATE" --seeds $SEEDS ${STITCH_EXTRA:-}
rc=$?
echo; echo "stitch exit code: $rc    wall time: $(( $(date +%s) - t0 )) s"
CACHE=$(python - <<PY
import sys; sys.path.insert(0, "src")
from ksz2_21cm.utils.config import load_config
print(load_config("$CONFIG")["paths"]["cache_dir"])
PY
)
echo "cache dir: $CACHE"; du -sh "$CACHE"/seed_* 2>/dev/null
echo "DONE rc=$rc   $(date)   log: $LOG"
exit $rc
