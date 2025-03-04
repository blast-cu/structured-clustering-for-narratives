## Preprocessing Pipeline

### This code is adapted from the EMNLP 2021 paper "[Corpus-based Open-Domain Event Type Induction](https://arxiv.org/pdf/2109.03322.pdf)".

### Resources
Please download the resources from this [Google Drive Location](https://drive.google.com/drive/folders/16NxNsjeyN7ZzKpN24e-ybCHgpLNBWcRA?usp=sharing) and place them under the ./resources subfolder under the root directory.

### Parse Corpus and Extract Subject-Verb-Object Triplets

```Bash
python3 parse_corpus_and_extract_svo.py \
    --is_sentence 1 \
    --input_file ./covid19/corpus.txt \
    --save_path ./covid19/corpus_parsed_svo.pk
```

### Select Salient Verb Lemmas and Object Heads

```Bash
python3 select_salient_terms.py \
    --corpus_w_svo_pickle ./covid19/corpus_parsed_svo.pk \
    --min_verb_freq 3 \
    --min_obj_freq 3
```

### Generate Features for Each Salient <Predicate Lemma, Object Head> Mention

```Bash
python3 generate_po_mention_features.py \
    --corpus_w_svo_pickle ./covid19/corpus_parsed_svo.pk \
    --top_k 50 \
    --gpu_id 5
```

### Disambiguate Predicate Senses 

```Bash
python3 disambiguate_verb_sense.py \
    --mention_file ./covid19/corpus_parsed_svo_salient_po_mention_features.pk \
    --save_path ./covid19/po_mention_disambiguated.pk
```

### Generate Features for Each Salient <Predicate Sense, Object Head> Tuples

```Bash
python3 generate_po_tuple_features.py \
    --mention_file ./covid19/corpus_parsed_svo_salient_po_mention_features.pk \
    --sense_mapping ./covid19/po_mention_disambiguated.pk \
    --save_file ./covid19/po_tuple_features_all_svos.pk \
    --use_all_svos
```