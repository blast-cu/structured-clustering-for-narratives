#!/bin/bash

#SBATCH --account=blanca-curc-gpu
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=7-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:3
#SBATCH --mem=50G
#SBATCH --job-name=pi_verbalize_chains
#SBATCH --output=logs/data.%j.log

source ~/.bashrc

module load anaconda
conda activate event

mkdir -p "$SLURM_SCRATCH/cache/HF"

export HF_HOME="$SLURM_SCRATCH/cache/HF"
export PYTHONPATH=/projects/roda9210/structured-clustering-for-narratives

echo "Starting up Ollama server"
OLLAMA_PORT=9999
OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT}
nohup ollama serve > ./data/partisanship/immigration/ollama_log.txt 2>&1 &

echo "Waiting for Ollama server to start"
sleep 1m

HOST_IP=$(hostname -i)

echo "Generating chain verbalizations for partisanship_immigration."

# python3 ./annotation/annotate_chains.py -c "immigration" --host $host_ip --workers 8 --save_interval 50

python3 ./annotation/chain_verbalizer.py -c "partisanship_immigration" --host ${HOST_IP} --port ${OLLAMA_PORT} --workers 4 --domain "immigration" --save_interval 25