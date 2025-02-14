#!/bin/bash

#SBATCH --account=blanca-curc-gpu
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=1-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --mem=100G
#SBATCH --job-name=cluster_chains
#SBATCH --output=logs/data.%j.log

source ~/.bashrc

module load anaconda
conda activate event

mkdir -p "$SLURM_SCRATCH/cache/HF/transformers"
mkdir -p "$SLURM_SCRATCH/cache/HF/datasets"

export TRANSFORMERS_CACHE="$SLURM_SCRATCH/cache/HF/transformers"
export HF_DATASETS_CACHE="$SLURM_SCRATCH/cache/HF/datasets"

# Processing annotated chains
python3 ./process_annotated_chains.py -c "immigration"

# Clustering chains

python3 ./clustering/pckmeans.py -c "immigration" -k 200