#!/bin/bash

#SBATCH --account=blanca-curc-gpu
#SBATCH --mail-user=alle5715@colorado.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=1-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:2
#SBATCH --mem=50G
#SBATCH --job-name=annotate_characters
#SBATCH --output=logs/data.%j.log

source ~/.bashrc

module load anaconda
conda activate characters

mkdir -p "$SLURM_SCRATCH/cache/HF"

export HF_HOME="$SLURM_SCRATCH/cache/HF"
export PYTHONPATH=/scratch/alpine/alle5715/structured-clustering-for-narratives

echo "Starting up Ollama server"
nohup ollama serve > ollama_log_3.txt 2>&1 &
ollama pull llama3.3

echo "Waiting for Ollama server to start"
sleep 1m

host_ip=$(hostname -i)

python3 -m character.extract_characters.run --host $host_ip --workers 8 --save_interval 50 --port 11434

systemctl stop ollama