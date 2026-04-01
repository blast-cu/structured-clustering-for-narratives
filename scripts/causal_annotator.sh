#!/bin/bash

#SBATCH --account=blanca-clearlab1
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=7-00:00:00
#SBATCH --qos=blanca-clearlab1
#SBATCH --partition=blanca-clearlab1
#SBATCH --gres=gpu:h100_80gb
#SBATCH --mem=50G
#SBATCH --job-name=causal_annotator
#SBATCH --output=logs/causal.%j.log

module load anaconda
conda activate event

mkdir -p "$SLURM_SCRATCH/cache/HF"

export HF_HOME="$SLURM_SCRATCH/cache/HF"
export PYTHONPATH=/projects/roda9210/structured-clustering-for-narratives

source="reddit"
corpus="longcovid" # parkinsons or longcovid

echo "Starting up Ollama server"
OLLAMA_PORT=9940
OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT}
nohup ollama serve > ./data/${source}/${corpus}/ollama_log.txt 2>&1 &

echo "Waiting for Ollama server to start"
sleep 1m

HOST_IP=$(hostname -i)

python3 ./causality/causal_annotator.py -c "${source}_${domain}" --host ${HOST_IP} --port ${OLLAMA_PORT} --workers 3
--source "${source}" --domain "Parkinsons Subreddits"