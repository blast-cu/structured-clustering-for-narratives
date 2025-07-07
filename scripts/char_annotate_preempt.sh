#!/bin/bash

#SBATCH --account=blanca-kann
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=1-00:00:00
#SBATCH --qos=preemptable
#SBATCH --partition=blanca-clearlab1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=40G
#SBATCH --job-name=pi_char_annotate
#SBATCH --output=logs/data.%j.log

source ~/.bashrc

module load anaconda
conda activate event

mkdir -p "$SLURM_SCRATCH/cache/HF"

export HF_HOME="$SLURM_SCRATCH/cache/HF"
export PYTHONPATH=/projects/roda9210/structured-clustering-for-narratives

source="partisanship" # mfc or partisanship
domain="immigration" # immigration or guncontrol

echo "Starting up Ollama server"
OLLAMA_PORT=9980
OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT}
# OLLAMA_NUM_PARALLEL=4
nohup ollama serve > ./data/${source}/${domain}/ollama_log.txt 2>&1 &

echo "Waiting for Ollama server to start"
sleep 1m

HOST_IP=$(hostname -i)

echo "Generating character annotations for ${source}_${domain}."

python3 ./annotation/character_analysis.py -c "${source}_${domain}" --host ${HOST_IP} --port ${OLLAMA_PORT} --workers 3 --domain "${domain}" --use_excerpt --save_interval 3