#!/bin/bash

#SBATCH --account=blanca-curc-gpu
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=7-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:3
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
OLLAMA_PORT=9980
OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT}
# OLLAMA_NUM_PARALLEL=4

nohup ollama serve > ./data/${source}/${domain}/ollama_log.txt 2>&1 &

echo "Waiting for Ollama server to start"
sleep 1m

HOST_IP=$(hostname -i)

echo "Generating cluster themes for ${source}_${domain}."

#python3 ./annotation/cluster_analysis.py -c "${source}_${domain}" --host ${HOST_IP} --port ${OLLAMA_PORT} --workers 3 --domain "${domain}" --save_interval 5

python -m annotation.reddit_cluster_analyzer \
    -c ${source}_${domain} \
    --host ${HOST_IP} \
    --port ${OLLAMA_PORT} \
    --domain "Parkinson's Disease" \
    --clusters_file data/reddit/parkinsons/clustering/clusters_150_0.0.pickle