#!/bin/bash
# =============================================================================
# Galaxy merger-tree grid of the paper (Fig. 4).
#
# Sizes, arms, seeds and the sample budget are read from the environment, so
# one script covers every cell. The paper's calls, from the repository root:
#
#   MT_SIZES="6 8 10 12" MT_SEEDS="22 23 24 25 26" \
#       sbatch --array=0-59%6 sbatches_shell_files/sbatch_mergertrees_grid.sh
#   MT_SIZES="4" MT_SEEDS="22 23 24 25 26" MT_SAMPLES=3000000 MT_TAG=3M \
#       sbatch --array=0-14%6 sbatches_shell_files/sbatch_mergertrees_grid.sh
#
# The array length is sizes x arms x seeds. Set the variables in the shell as
# above rather than through --export, which splits its argument on commas and
# would truncate every list to one value. Adjust --partition, --gres and --mem
# for your cluster. Outputs land under mergertrees/ in the repository root.
# =============================================================================
#SBATCH --job-name=nmln-mergertrees
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-14%6
#SBATCH --mem=48G
#SBATCH --output=mergertrees/slurm_logs/mt_%A_%a.out
#SBATCH --error=mergertrees/slurm_logs/mt_%A_%a.err

set -eo pipefail

REPO=${REPO:-$SLURM_SUBMIT_DIR}          # the repository root (submit from there)
PYTHON=${PYTHON:-python}                 # interpreter of the environment with nmln installed

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export MLFLOW_TRACKING_URI="$REPO/mergertrees/mlruns"
export NMLN_EXPERIMENTS_DIR="$REPO/mergertrees/experiments"
mkdir -p mergertrees/slurm_logs mergertrees/mlruns mergertrees/experiments

RESOLVED=$($PYTHON -c "import nmln, os; print(os.path.dirname(os.path.dirname(nmln.__file__)))")
if [ "$RESOLVED" != "$REPO" ]; then
    echo "FATAL: nmln resolves to $RESOLVED, not $REPO. Aborting." >&2
    exit 1
fi
echo "nmln: $RESOLVED   git: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo n/a)"

# Create the MLflow experiment up front so that array tasks do not race on it.
$PYTHON - <<'MLFLOW_INIT'
import os, mlflow
from mlflow.exceptions import MlflowException
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
try:
    mlflow.create_experiment("lightning_logs")
except MlflowException as exc:
    print(f"experiment already present ({type(exc).__name__})")
MLFLOW_INIT

read -r -a SIZES <<< "${MT_SIZES//,/ }"
read -r -a ARMS  <<< "${MT_ARMS//,/ }"
read -r -a SEEDS <<< "${MT_SEEDS//,/ }"
[ ${#SIZES[@]} -gt 0 ] || SIZES=(6 8 10 12)
[ ${#ARMS[@]}  -gt 0 ] || ARMS=(wilson matrix_tree notree)
[ ${#SEEDS[@]} -gt 0 ] || SEEDS=(22)
# Sample budget, and an optional run-name suffix. The suffix matters whenever
# the budget changes: experiment directories carry no run id, so a second run
# under the same name overwrites the first one's generated_states.txt.
SAMPLES=${MT_SAMPLES:-2000000}
TAG=${MT_TAG:-}
echo "grid: sizes=${SIZES[*]} arms=${ARMS[*]} seeds=${SEEDS[*]} samples=$SAMPLES tag=${TAG:-none}"

IDX=${SLURM_ARRAY_TASK_ID}
SEED=${SEEDS[$(( IDX % ${#SEEDS[@]} ))]}; IDX=$(( IDX / ${#SEEDS[@]} ))
ARM=${ARMS[$(( IDX % ${#ARMS[@]} ))]};    IDX=$(( IDX / ${#ARMS[@]} ))
SIZE=${SIZES[$(( IDX % ${#SIZES[@]} ))]}

case "$ARM" in
  wilson)      FLAGS="--tree-constraint --sampler-type wilson" ;;
  matrix_tree) FLAGS="--tree-constraint --sampler-type matrix_tree" ;;
  notree)      FLAGS="--no-tree-constraint" ;;
esac

echo "task=${SLURM_ARRAY_TASK_ID} n=$SIZE arm=$ARM seed=$SEED node=$(hostname) start=$(date -Is)"
$PYTHON scripts/train_nmln.py mergertrees \
    --num-nodes "$SIZE" --seed "$SEED" --gpus 0 $FLAGS ${TAG:+--name "$TAG"} \
    --num-chains 20 --batch-size 10 --max-generated-samples "$SAMPLES"
echo "finished=$(date -Is)"
