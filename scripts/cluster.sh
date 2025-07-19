#!/bin/bash

#SBATCH --account=blanca-curc-gpu
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=7-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --job-name=cluster_chains
#SBATCH --output=logs/cluster.%j.log

source ~/.bashrc

module load anaconda
conda activate event

mkdir -p "$SLURM_SCRATCH/cache/HF"

export HF_HOME="$SLURM_SCRATCH/cache/HF"
export PYTHONPATH=/scratch/alpine/roda9210/structured-clustering-for-narratives

# Processing annotated chains
# python3 ./clustering/process_event_chains.py -c "mfc_immigration"

# Clustering chains

# python3 ./clustering/weighted_pckmeans.py -c "mfc_immigration" -k $1 -w $2 --skip_init

# python3 ./clustering/finetuned_pckmeans.py -c "mfc_immigration" -k $1 -w $2

python3 ./clustering/weighted_pckmeans.py -c "mfc_guncontrol" -k 50 -w 0.1 --skip_init

# python3 ./clustering/finetuned_pckmeans.py -c "mfc_immigration" -k 250 -w 0.5

# python3 ./clustering/dcc.py -c "mfc_immigration" -k 250 --compute_purity_per_epoch