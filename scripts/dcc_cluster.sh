#!/bin/bash

#SBATCH --account=blanca-curc-gpu
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=1-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --mem=100G
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

# python3 ./clustering/weighted_pckmeans.py -c "immigration" -k $1 -w $2

python3 ./clustering/dcc.py -c "immigration" -k 250 --weight_pairwise 0.01 --batch_size 32

# python3 ./clustering/kmeans.py -c "immigration" -k $1

# Running regression

# python3 ./models/regression.py -c "immigration" -k $1 -w $2 --save_results