#!/bin/bash


#SBATCH --ntasks=4
#SBATCH --partition=aa100

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=50G

#SBATCH --mail-user=alle5715@colorado.edu
#SBATCH --mail-type=ALL
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

host_ip=$(hostname -i)
python3 -m character.extract_characters.run --host $host_ip --port 9999 --config default.yaml --dataset guncontrol_subframes_corpus.json --prompt_file guncontrol_default.json