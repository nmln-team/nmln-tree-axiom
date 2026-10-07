#!/bin/bash
# =============================================================================
# Sampler-only scaling benchmark of the paper (Fig. 3).
#
# Times a single spanning-tree draw as a function of domain size n, outside
# the training loop, for both exact tree samplers and two weight regimes.
# Pure CPU work: a CPU partition is sufficient, and --gres can be dropped.
#
# Submit from the repository root:   sbatch sbatches_shell_files/sbatch_scaling_benchmark.sh
# Output: benchmark_results/tree_samplers.csv (+ .json with the run settings).
# =============================================================================
#SBATCH --job-name=nmln-scaling
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_logs/scaling_%j.out
#SBATCH --error=slurm_logs/scaling_%j.err

set -euo pipefail

REPO=${REPO:-$SLURM_SUBMIT_DIR}          # the repository root (submit from there)
PYTHON=${PYTHON:-python}                 # interpreter of the environment with nmln installed

cd "$REPO"
mkdir -p slurm_logs benchmark_results
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "node    : $(hostname)"
echo "started : $(date -Is)"

$PYTHON scripts/benchmark_tree_samplers.py \
    --sizes 10,20,50,100,200,500,1000,2000,5000 \
    --samplers wilson,matrix_tree \
    --regimes uniform,lognormal \
    --reps 5 \
    --time-limit 60 \
    --seed 22 \
    --device cpu \
    --out benchmark_results/tree_samplers.csv

echo "finished: $(date -Is)"
