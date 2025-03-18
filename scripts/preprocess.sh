#!/bin/bash

#SBATCH --account=blanca-curc-gpu
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=1-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --job-name=event_preprocess
#SBATCH --output=logs/event.%j.log

module load anaconda
conda activate event

mkdir -p "$SLURM_SCRATCH/cache/HF"

export HF_HOME="$SLURM_SCRATCH/cache/HF"
export PYTHONPATH=/projects/roda9210/structured-clustering-for-narratives

python3 -m spacy download en_core_web_lg

source="mfc" # mfc or partisanship
corpus="mfc/guncontrol"

# ONLY for Partisanship - Parse Partisanship data structured to generate corpus_labeled.json
# python3 ./preprocessing/${source}/parse_partisanship_data_structure.py \
#     --input_file ./data/${corpus}/article_data.pkl \
#     --save_path ./data/${corpus}/

# Generate corpus.txt
# python3 ./preprocessing/${source}/gen_corpus.py \
#     --input_file ./data/${source}/${corpus}/corpus_labeled.json \
#     --save_path ./data/${source}/${corpus}/

# echo "gen_corpus.py done"


# Parse Corpus and Extract Subject-Verb-Object Triplets
# python3 ./preprocessing/parse_corpus_and_extract_svo.py \
#     --is_sentence 1 \
#     --input_file ./data/${source}/${corpus}/corpus.txt \
#     --save_path ./data/${source}/${corpus}/corpus_parsed_svo.pk

# echo "parse_corpus_and_extract_svo.py done"


# Select Salient Verb Lemmas and Object Heads
python3 ./preprocessing/select_salient_terms.py \
    --corpus_w_svo_pickle ./data/${source}/${corpus}/corpus_parsed_svo.pk \
    --min_verb_freq 1 \
    --min_obj_freq 1 \
    --top_verb_ratio 1.0 \
    --top_obj_ratio 1.0

echo "select_salient_terms.py done"


# Generate Features for Each Salient <Predicate Lemma, Object Head> Mention
python3 ./preprocessing/generate_po_mention_features.py \
    --corpus_w_svo_pickle ./data/${source}/${corpus}/corpus_parsed_svo.pk \
    --top_k 50 \
    --gpu_id 0

echo "generate_po_mention_features.py done"


# Disambiguate Predicate Senses
python3 ./preprocessing/disambiguate_verb_sense.py \
    --mention_file ./data/${source}/${corpus}/corpus_parsed_svo_salient_po_mention_features.pk \
    --save_path ./data/${source}/${corpus}/po_mention_disambiguated.pk

echo "disambiguate_verb_sense.py done"


# Generate Features for Each Salient <Predicate Sense, Object Head> Tuples
python3 ./preprocessing/generate_po_tuple_features.py \
    --mention_file ./data/${source}/${corpus}/corpus_parsed_svo_salient_po_mention_features.pk \
    --sense_mapping ./data/${source}/${corpus}/po_mention_disambiguated.pk \
    --save_file ./data/${source}/${corpus}/po_tuple_features_all_svos.pk \
    --use_all_svos

echo "generate_po_tuple_features.py done"


# MFC -Generate files for mapping each event (and frequency) to its original article and sentence
# python3 ./preprocessing/${source}/map_events_to_articles.py \
#     --mfc_corpus ./data/${source}/${corpus}/corpus_labeled.json \
#     --domain $corpus \
#     --processed_corpus ./data/${source}/${corpus}/corpus.txt \
#     --mfc_codes ./data/${source}/codes.json \
#     --po_tuple_features ./data/${source}/${corpus}/po_tuple_features_all_svos.pk \
#     --doc_2_sent ./data/${source}/${corpus}/doc_id_2_sent_ids_corpus_labeled.json \
#     --output_file ./data/${source}/${corpus}/processed_corpus.json

# echo "map_article_event_freq.py done"

# Partisanship -Generate files for mapping each event (and frequency) to its original article and sentence

# python3 ./preprocessing/${source}/map_events_to_articles.py \
#   --partisanship_corpus ./data/${corpus}/corpus_labeled.json \
#   --processed_corpus ./data/${corpus}/corpus.txt \
#   --po_tuple_features ./data/${corpus}/po_tuple_features_all_svos.pk \
#   --doc_2_sent ./data/${corpus}/doc_id_2_sent_ids_corpus_labeled.json \
#   --output_file ./data/${corpus}/processed_corpus.json

echo "map_article_event_freq.py done"

# Sample documents from full dataset
python3 ./preprocessing/${source}/sample_docs.py \
  --corpus ./data/${source}/${corpus}/processed_corpus.json \
  --output_file ./data/${source}/${corpus}/processed_corpus_3000.json

echo "sample_docs.py done"
