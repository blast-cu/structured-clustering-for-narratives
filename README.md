# structured-clustering-for-narratives

---
### Code for the paper "A Structured Clustering Approach for Inducing Media Narratives"

# Abstract

---
Media narratives wield tremendous power in shaping public opinion, yet computational approaches struggle to capture the nuanced storytelling structures that communication theory emphasizes as central to how meaning is constructed. Existing approaches either miss subtle narrative patterns through coarse-grained analysis or require domain-specific taxonomies that limit scalability. To bridge this gap, we present a framework for inducing rich narrative schemas by jointly modeling events and characters via structured clustering. Our approach produces explainable narrative schemas that align with established framing theory while scaling to large corpora without exhaustive manual annotation.

# Data

---
Source data from the Media Frames Corpus (MFC) (Card et al., 2015) and place them under the `data/` directory.

# Framework

---

#### NOTE: See `config.conf` for configuration settings. The `base` config represents universal settings, whereas the dataset specific configuration is categoeized by their names. Select the approporiate dataset name when running the code by passing the `-c` flag followed by the dataset name (e.g., `mfc_guncontrol`).

### 1. Event Extraction

Follow the instructions in `data_preprocessing/README.md` to preprocess the raw news articles and prepare the 
dataset to extract events for narrative chain construction.

### 2. Causal Relation Extraction

TBC

### 3. Chain Verbalization

Run `annotation/chain_verbalization.py` to verbalize the extracted narrative chains into natural language sentences.

Example Usage:

```bash
source="mfc"
domain="guncontrol" # immigration or guncontrol
python3 ./annotation/chain_verbalizer.py -c "${source}_${domain}" --host ${HOST_IP} --port ${OLLAMA_PORT} --workers 3 --source "${source}" --domain "${domain}" --excerpt 4 --save_interval 3
```

### 4. Character Annotations

Run `annotation/character_analysis.py` to annotate narrative chains with character groups and their archetypal roles.

Example Usage:

```bash
source="mfc"
domain="guncontrol" # immigration or guncontrol
python3 ./annotation/character_analysis.py -c "${source}_${domain}" --host ${HOST_IP} --port ${OLLAMA_PORT} --workers 3 --domain "${domain}" --use_excerpt --save_interval 3
```

### 5. Structured Clustering

Run `clustering/weighted_pckmeans.py` to perform structured clustering on the annotated narrative chains, 
incorporating both event and character information. Pass the number of clusters `-k` as an argument to specify the 
desired number of narrative schemas. Also, pass the constraint weight `-w` to control the influence of 
character-based constraints in the clustering process. Choose `--skip_init` to skip the constraint-aware k-means++
initialization and choose standard k-means++ initialization instead.

Example Usage:

```bash
source="mfc
domain="guncontrol" # immigration
python3 ./clustering/weighted_pckmeans.py -c "mfc_immigration" -k 500 -w 0.1
```

### 6. Narrative Schema Attribution

Run `/annotation/cluster_analysis.py` to use an LLM to attribute schema definitions to the clusters produced by the structured clustering step. This will generate human-readable descriptions of the narrative schemas, which can be used for further analysis and interpretation.

Example Usage:

```bash
source="mfc"
domain="guncontrol" # immigration or guncontrol
python3 ./annotation/cluster_analysis.py -c "${source}_${domain}" --host ${HOST_IP} --port ${OLLAMA_PORT} --workers 3 --domain "${domain}" --save_interval 5
```

# Citation

If you find our work useful, please consider citing our paper:

```
TBC
```