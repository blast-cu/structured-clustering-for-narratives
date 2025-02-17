#!/bin/bash

#SBATCH --partition=aa100
#SBATCH --mail-user=alle5715@colorado.edu
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=1-00:00:00
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

echo "Waiting for Ollama server to start"
sleep 1m

# sinteractive --account=blanca-curc-gpu --partition=blanca-curc-gpu --qos=blanca-curc-gpu --time=01:00:00 --ntasks=16 --gres=gpu:1 --mem=20G

host_ip=$(hostname -i)
# curl http://10.225.8.162:9999/api/pull -d '{"model": "llama3.3"}'

python3 -m character.extract_characters.run --host $host_ip --workers 8 --save_interval 50