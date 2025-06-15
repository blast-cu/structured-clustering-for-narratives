#!/bin/bash

##SBATCH --account=blanca-curc-gpu
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --time=1-00:00:00
#SBATCH --qos=preemptable
#SBATCH --partition=blanca-clearlab1
#SBATCH --gres=gpu:h100:1
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
nohup ollama serve > ./data/partisanship/guncontrol/ollama_log.txt 2>&1 &

echo "Waiting for Ollama server to start"
sleep 1m

host_ip=$(hostname -i)

# python3 ./annotation/annotate_chains.py -c "immigration" --host $host_ip --workers 8 --save_interval 50

python3 ./annotation/chain_verbalizer.py -c "partisanship_guncontrol" --host $host_ip --workers 4 --domain "guncontrol" --save_interval 25