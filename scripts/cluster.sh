#!/bin/bash

#SBATCH --account=blanca-curc-gpu
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=7-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --job-name=cluster_chains
#SBATCH --output=logs/cluster.%j.log

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

python3 ./clustering/finetuned_pckmeans.py -c "immigration" -k 250 -w 1

# python3 ./clustering/kmeans.py -c "immigration" -k $1

# Running regression

# python3 ./models/regression.py -c "immigration" -k $1 -w $2 --save_results