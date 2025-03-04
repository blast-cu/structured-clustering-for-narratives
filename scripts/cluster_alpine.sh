#!/bin/bash

#SBATCH --account=ucb414_asc1
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=1-00:00:00
#SBATCH --partition=amilan
#SBATCH --mem=50G
#SBATCH --job-name=cluster_chains
#SBATCH --output=logs/data.%j.log

source ~/.bashrc

module load anaconda
conda activate event

mkdir -p "$SLURM_SCRATCH/cache/HF"

export HF_HOME="$SLURM_SCRATCH/cache/HF"
export PYTHONPATH=/projects/roda9210/structured-clustering-for-narratives

# Processing annotated chains
# python3 ./annotation/process_annotated_chains.py -c "immigration"

# Clustering chains

python3 ./clustering/weighted_pckmeans.py -c "immigration" -k $1 -w $2

# python3 ./clustering/kmeans.py -c "immigration" -k $1

# Running regression

python3 ./prediction_models/regression.py -c "immigration" -k $1 -w $2 --save_results