#!/bin/bash

#SBATCH --account=blanca-blast-lecs
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=7-00:00:00
#SBATCH --qos=blanca-blast-lecs
#SBATCH --partition=blanca-blast-lecs
#SBATCH --gres=gpu:h100_80gb
#SBATCH --mem=30G
#SBATCH --job-name=vaccinated_verbalize_chains
#SBATCH --output=logs/verbalize.%j.log

source ~/.bashrc

module load anaconda
conda activate struct

PYTHON=/projects/roda9210/software/anaconda/envs/struct/bin/python

mkdir -p "$SLURM_SCRATCH/cache/HF"

export HF_HOME="$SLURM_SCRATCH/cache/HF"
export PYTHONPATH=/scratch/alpine/roda9210/structured-clustering-for-narratives

source="reddit" # mfc or partisanship
domain="parkinsons" # immigration or guncontrol

echo "Starting up Ollama server"
OLLAMA_PORT=9930
OLLAMA_HOST=0.0.0.0:${OLLAMA_PORT}
OLLAMA_NUM_PARALLEL=1
nohup ollama serve > ./data/${source}/${domain}/ollama_log.txt 2>&1 &

echo "Waiting for Ollama server to start"
sleep 1m

HOST_IP=$(hostname -i)

echo "Generating chain verbalizations for ${source}_${domain}."

$PYTHON ./annotation/reddit_chain_verbalizer.py -c "${source}_${domain}" --host ${HOST_IP} --port ${OLLAMA_PORT} --workers 3 --domain "Parkinson's Disease" --save_interval 3