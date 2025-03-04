#!/bin/bash

#SBATCH --account=blanca-curc-gpu
#SBATCH --mail-user=roda9210@colorado.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-00:00:00
#SBATCH --qos=blanca-curc-gpu
#SBATCH --partition=blanca-curc-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=15G
#SBATCH --job-name=event_preprocess
#SBATCH --output=logs/event.%j.log

#module load anaconda
#conda activate event
#
#mkdir -p "$SLURM_SCRATCH/cache/HF/transformers"
#mkdir -p "$SLURM_SCRATCH/cache/HF/datasets"
#
#export TRANSFORMERS_CACHE="$SLURM_SCRATCH/cache/HF/transformers"
#export HF_DATASETS_CACHE="$SLURM_SCRATCH/cache/HF/datasets"


python3 -m spacy download en_core_web_lg

source="subframes" # subframes
corpus="subframes/immigration"

# Generate corpus.txt
python3 ./data/gen_corpus.py \
    --input_file ./corpus/${corpus}/corpus_labeled.json \
    --save_path ./corpus/${corpus}/

echo "gen_corpus.py done"


# Parse Corpus and Extract Subject-Verb-Object Triplets
python3 ./data/parse_corpus_and_extract_svo.py \
    --is_sentence 1 \
    --input_file ./corpus/${corpus}/corpus.txt \
    --save_path ./corpus/${corpus}/corpus_parsed_svo.pk

echo "parse_corpus_and_extract_svo.py done"


# Select Salient Verb Lemmas and Object Heads
python3 ./data/select_salient_terms.py \
    --corpus_w_svo_pickle ./corpus/${corpus}/corpus_parsed_svo.pk \
    --min_verb_freq 3 \
    --min_obj_freq 3 \
    --top_verb_ratio 0.8 \
    --top_obj_ratio 0.8

echo "select_salient_terms.py done"


# Generate Features for Each Salient <Predicate Lemma, Object Head> Mention
python3 ./data/generate_po_mention_features.py \
    --corpus_w_svo_pickle ./corpus/${corpus}/corpus_parsed_svo.pk \
    --top_k 50 \
    --gpu_id 0

echo "generate_po_mention_features.py done"


# Disambiguate Predicate Senses
python3 ./data/disambiguate_verb_sense.py \
    --mention_file ./corpus/${corpus}/corpus_parsed_svo_salient_po_mention_features.pk \
    --save_path ./corpus/${corpus}/po_mention_disambiguated.pk

echo "disambiguate_verb_sense.py done"


# Generate Features for Each Salient <Predicate Sense, Object Head> Tuples
python3 ./data/generate_po_tuple_features.py \
    --mention_file ./corpus/${corpus}/corpus_parsed_svo_salient_po_mention_features.pk \
    --sense_mapping ./corpus/${corpus}/po_mention_disambiguated.pk \
    --save_file ./corpus/${corpus}/po_tuple_features_all_svos.pk \
    --use_all_svos

echo "generate_po_tuple_features.py done"


# Generate files for mapping each event (and frequency) to its original article and sentence
#python3 ./data/${source}/map_events_to_articles.py \
#    --mfc_corpus ./corpus/${corpus}/corpus_labeled.json \
#    --processed_corpus ./corpus/${corpus}/corpus.txt \
#    --mfc_codes ./corpus/${source}/codes.json \
#    --po_tuple_features ./corpus/${corpus}/po_tuple_features_all_svos.pk \
#    --doc_2_sent ./corpus/${corpus}/doc_id_2_sent_ids_immigrants_labeled.json \
#    --output_file ./corpus/${corpus}/processed_corpus.json
#
#echo "map_article_event_freq.py done"