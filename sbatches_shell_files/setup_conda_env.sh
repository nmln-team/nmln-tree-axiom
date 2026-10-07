#!/bin/bash
# Creates the conda environment used by the SLURM scripts in this directory.
# Run once, interactively, from the repository root:   bash sbatches_shell_files/setup_conda_env.sh
# If your cluster provides conda through a module system, load it first
# (e.g. `module load Anaconda3`).

set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n nmln-env python=3.10 -y
conda activate nmln-env

# PyTorch with CUDA 11.8
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric packages
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv pyg-lib torch_geometric \
    -f https://data.pyg.org/whl/torch-2.4.0+cu118.html

# All other project dependencies
pip install \
    pytorch-lightning==2.* \
    mlflow==2.* \
    dagshub==0.5.* \
    rdkit-pypi \
    numba \
    typer==0.12.* \
    tqdm \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    networkx \
    scipy \
    torchmetrics \
    wandb \
    hydra-core \
    icecream \
    smart-open \
    boto3 \
    codetiming \
    protobuf==3.20.0 \
    joblib \
    pyyaml \
    imageio \
    pyemd \
    pygsp \
    overrides \
    problog \
    nltk \
    ipdb \
    "numpy<2.0"

# Install this repository's nmln package in editable mode
pip install -e "$REPO"

echo "Done. Test with: python -c 'import torch; print(torch.cuda.is_available())'"
