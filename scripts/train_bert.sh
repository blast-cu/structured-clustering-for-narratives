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
#SBATCH --job-name=train_bert
#SBATCH --output=logs/bert.%j.log

source ~/.bashrc

module load anaconda
conda activate event

mkdir -p "$SLURM_SCRATCH/cache/HF"

export HF_HOME="$SLURM_SCRATCH/cache/HF"
export PYTHONPATH=/scratch/alpine/roda9210/structured-clustering-for-narratives


python3 ./models/create_dataset.py -c "mfc_immigration"
python3 ./models/bert.py -c "mfc_immigration"