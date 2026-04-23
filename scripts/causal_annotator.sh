#!/bin/bash

#SBATCH --account=blanca-curc-gpu
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=7-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=50G
#SBATCH --job-name=causal_annotator
#SBATCH --output=logs/causal.%j.log

module load anaconda
conda activate struct

PYTHON=/projects/roda9210/software/anaconda/envs/struct/bin/python

mkdir -p "$SLURM_SCRATCH/cache/HF"

export HF_HOME="$SLURM_SCRATCH/cache/HF"
export PYTHONPATH=/scratch/alpine/roda9210/structured-clustering-for-narratives

source="reddit"
corpus="parkinsons" # parkinsons or longcovid

source ~/.bashrc

echo "Starting up Ollama server"
OLLAMA_PORT=9920
OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT}
OLLAMA_NUM_PARALLEL=2
nohup ollama serve > ./data/${source}/${corpus}/ollama_log.txt 2>&1 &

echo "Waiting for Ollama server to start"
sleep 1m

HOST_IP=$(hostname -i)


$PYTHON ./causality/causal_annotator.py -c "${source}_${corpus}" --host ${HOST_IP} --port ${OLLAMA_PORT} --workers 3 --domain "Parkinson's Disease"