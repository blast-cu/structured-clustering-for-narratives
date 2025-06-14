#!/bin/bash

#SBATCH --account=blast-lecs-gpu
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=7-00:00:00
#SBATCH --qos=blast-lecs-gpu
#SBATCH --partition=blast-lecs-gpu
#SBATCH --gres=gpu:1gpu:h100_7g.80gb
#SBATCH --mem=50G
#SBATCH --job-name=verbalize_chains
#SBATCH --output=logs/data.%j.log

source ~/.bashrc

module load anaconda
conda activate event

mkdir -p "$SLURM_SCRATCH/cache/HF"

export HF_HOME="$SLURM_SCRATCH/cache/HF"
export PYTHONPATH=/projects/roda9210/structured-clustering-for-narratives

echo "Starting up Ollama server"
nohup ollama serve > ollama_log.txt 2>&1 &

echo "Waiting for Ollama server to start"
sleep 1m

host_ip=$(hostname -i)

# python3 ./annotation/annotate_chains.py -c "immigration" --host $host_ip --workers 8 --save_interval 50

python3 ./annotation/chain_verbalizer.py -c "immigration" --host $host_ip --workers 8 --save_interval 50