#!/bin/bash
# =============================================================================
# Molecule grid of the paper (Fig. 1, Table 3, Fig. 2).
#
#   atoms 8..15  x  {original, wilson, matrix_tree}  x  seeds 22-26 = 120 tasks
#
# Submit from the repository root:   sbatch sbatches_shell_files/sbatch_molecules_exact_rerun.sh
# Adjust --partition, --gres and --mem for your cluster. Logs, the MLflow store
# and the generated samples land under exact_rerun/ in the repository root.
# =============================================================================
#SBATCH --job-name=nmln-molecules
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --array=0-119%8
#SBATCH --mem=48G
#SBATCH --output=exact_rerun/slurm_logs/molecules_%A_%a.out
#SBATCH --error=exact_rerun/slurm_logs/molecules_%A_%a.err

set -euo pipefail

REPO=${REPO:-$SLURM_SUBMIT_DIR}          # the repository root (submit from there)
PYTHON=${PYTHON:-python}                 # interpreter of the environment with nmln installed

cd "$REPO"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export MLFLOW_TRACKING_URI="$REPO/exact_rerun/mlruns"
export NMLN_EXPERIMENTS_DIR="$REPO/exact_rerun/experiments"
mkdir -p exact_rerun/slurm_logs exact_rerun/mlruns exact_rerun/experiments

# Make sure the nmln package being run is the one in this checkout.
RESOLVED=$($PYTHON -c "import nmln, os; print(os.path.dirname(os.path.dirname(nmln.__file__)))")
echo "nmln resolves to: $RESOLVED"
if [ "$RESOLVED" != "$REPO" ]; then
    echo "FATAL: nmln resolves to $RESOLVED, not $REPO. Aborting." >&2
    exit 1
fi
echo "git HEAD: $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"

# MLflow's file store creates the experiment without locking, so array tasks
# that start together can race on it. Create it up front; this is idempotent.
$PYTHON - <<'MLFLOW_INIT'
import os
import mlflow
from mlflow.exceptions import MlflowException

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
try:
    mlflow.create_experiment("lightning_logs")
    print("created MLflow experiment 'lightning_logs'")
except MlflowException as exc:
    print(f"MLflow experiment already present ({type(exc).__name__}); continuing")
MLFLOW_INIT

ATOMS_LIST=(8 9 10 11 12 13 14 15)
METHOD_LIST=(original wilson matrix_tree)
SEED_LIST=(22 23 24 25 26)

IDX=${SLURM_ARRAY_TASK_ID}
SEED=${SEED_LIST[$(( IDX % 5 ))]};      IDX=$(( IDX / 5 ))
METHOD=${METHOD_LIST[$(( IDX % 3 ))]};  IDX=$(( IDX / 3 ))
ATOMS=${ATOMS_LIST[$(( IDX % 8 ))]}

case "$METHOD" in
  original)    TREE_FLAGS="--no-tree-constraint" ;;
  wilson)      TREE_FLAGS="--tree-constraint --sampler-type wilson --no-symmetric-constraint --no-antireflexive-constraint" ;;
  matrix_tree) TREE_FLAGS="--tree-constraint --sampler-type matrix_tree --no-symmetric-constraint --no-antireflexive-constraint" ;;
esac

CMD="molecules --name mlj-${METHOD}-20chain-${ATOMS}atoms-seed${SEED} \
    --dataset-name tree --max-num-atoms ${ATOMS} ${TREE_FLAGS} \
    --ks 2 --num-examples 1 --potential-output-dim 1 --p-noise 0.1 --lr 1e-5"
COMMON="--seed ${SEED} --gpus 0 --num-chains 20 --batch-size 10 --max-generated-samples 2000000"

echo "task=${SLURM_ARRAY_TASK_ID} atoms=$ATOMS method=$METHOD seed=$SEED node=$(hostname) start=$(date -Is)"
echo "cmd: $CMD $COMMON"
$PYTHON scripts/train_nmln.py $CMD $COMMON
echo "finished=$(date -Is)"
