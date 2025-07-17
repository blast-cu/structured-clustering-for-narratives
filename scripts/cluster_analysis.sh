#!/bin/bash

#SBATCH --account=blanca-blast-lecs
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=7-00:00:00
#SBATCH --qos=blanca-blast-lecs
#SBATCH --partition=blanca-blast-lecs
#SBATCH --gres=gpu:h100_3g.40gb:2
#SBATCH --mem=40G
#SBATCH --job-name=cluster_analysis
#SBATCH --output=logs/data.%j.log

source ~/.bashrc

module load anaconda
conda activate event

mkdir -p "$SLURM_SCRATCH/cache/HF"

export HF_HOME="$SLURM_SCRATCH/cache/HF"
export PYTHONPATH=/projects/roda9210/structured-clustering-for-narratives

source="mfc" # mfc or partisanship
domain="guncontrol" # immigration or guncontrol

echo "Starting up Ollama server"
OLLAMA_PORT=9999
OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT}
# OLLAMA_NUM_PARALLEL=4

nohup ollama serve > ./data/${source}/${domain}/ollama_log.txt 2>&1 &

echo "Waiting for Ollama server to start"
sleep 1m

HOST_IP=$(hostname -i)

echo "Generating cluster themes for ${source}_${domain}."

python3 ./annotation/cluster_analysis.py -c "${source}_${domain}" --host ${HOST_IP} --port ${OLLAMA_PORT} --workers 3 --domain "${domain}" --save_interval 5