# Reproducible GENRE end-to-end entity linking

## Introduction

[GENRE](https://github.com/facebookresearch/GENRE) is an open-source autogenerative entity linker.
Best results are achieved with fixed mentions and mention-to-candidate mappings.
The original repository does not provide the mentions and candidate mappings.
This repository is an attempt to create this data following the paper as close as possible.
The Docker setup allows to run GENRE on given texts with few commands.

The difference to the original repository is the following:
1. We provide pre-computed mention tries and mention-to-candidate dictionaries.
2. We implemented a split strategy for long texts.

## 1. Installation

Get the code:

```
git clone git@github.com:hertelm/GENRE.git
cd GENRE
```

### Option 1: Install with Docker

The base image currently does not support GPU usage.
For GPU support, use a virtual environment (see instructions below) or a suitable base image. 

```
docker build -t genre .
```

### Option 2: Install with virtualenv

```
python3.8 -m virtualenv venv
source venv/bin/activate
pip install torch pytest requests spacy gdown
git clone -b fixing_prefix_allowed_tokens_fn --single-branch https://github.com/nicola-decao/fairseq.git
pip install -e ./fairseq
python3 -m spacy download en_core_web_sm
```

## 2. Download data

Download the models:

```
make download-models
```

Download the precomputed mention tries and candidate dictionaries:

```
make download-data
```

The downloaded data contains the mention-to-candidates dictionary `mention_to_candidates_dict.pkl` (as a pickle file),
with candidates for mentions from [Dalab](https://github.com/dalab/end2end_neural_el) and [Hoffart et al.](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads)
(22.2M mentions, 4.2M unique entities, 1.32 candidates per mention on average).
The trie containing the same mentions is `mention_trie.pkl`.

The data also contains the smaller dictionary `mention_to_candidates_dict.dalab.pkl`
(5.7M mentions, 469k unique entities, 1.27 candidates per mention on average),
which contains only entities from Dalab (but mentions from both data sources),
and the corresponding trie `mention_trie_dalab.pkl`.
However, we got better results from using the larger entity set on all benchmarks except Kore50.

## 3. Start Docker container

(This step can be skipped when you chose the installation with virtualenv.)

```
docker run --rm -v $PWD/data:/GENRE/data \
 -v $PWD/models:/GENRE/models -it genre bash
```

## 4. Run GENRE

Run GENRE on a file specified with the `-i` argument.
The file must be in Article JSONL format (introduced by Elevant).
That is, each line contains a JSON with a key "text".
See the file *example_article.jsonl* for an example.

```
python3 main.py --yago -i example_article.jsonl \
 -o out.jsonl --split_iter --mention_trie data/mention_trie.pkl \
 --mention_to_candidates_dict data/mention_to_candidates_dict.pkl
```

The result will be written to the file specified with `-o` and
stored under the key "GENRE" in each line's JSON.

Use `--mention_trie data/mention_trie.dalab.pkl --mention_to_candidates_dict data/mention_to_candidates_dict.dalab.pkl`
to restrict the entities and candidates to the entity universe from DALAB.

Remove the argument `--yago` to use the wiki_abs model 
(trained on Wikipedia abstracts only).
However, we were not able to reproduce the good results from the paper for that model
(`--yago` is better on all benchmarks).

## 5. Translate predictions to Wikidata QIDs

Run this command to transform the output by GENRE into the Article JSONL format used by Elevant.
Each predicted entity will be translated to a Wikidata QID (if possible).
Each line in the output will contain a key "entity_mentions"
with the predicted mention spans and Wikidata QIDs.

```
python3 transform_predictions.py out.jsonl -o out.qids.jsonl
```

## Additional information

### Split strategy

For long texts, GENRE either throws an error, returns an empty result,
or an incomplete result (the labelled text is shorter than the input text).
When this happens, we split the text into n sentences using SpaCy,
and feed GENRE with parts of n/k sentences, where k is increased incrementally by 1 until
all parts are short enough to be processed.

### Results

See https://elevant.cs.uni-freiburg.de for results on various benchmarks.

We were not able to reproduce the results from the wiki_abs model,
see [issue 72](https://github.com/facebookresearch/GENRE/issues/72)
of the original repository.

### Create mention trie and candidate dict

To create the mention trie and candidate sets by yourself, use the following steps.

1. Download entity and candidate data from Dalab, AIDA
and Elevant (Wikipedia-to-Wikidata mapping, needed for step 5). 

```
make download-additional-data
```

2. Create the mention-to-candidate dictionary.

```
python3 create_candidates_dict.py
```

3. Create the mention trie.

```
python3 create_mentions_trie.py
```

The commands 2 and 3 can be called with the argument `--dalab`
to only include entities from the Dalab entity universe (~470k entities).
Note: This is currently out of sync with the files you download with `make download-data`.
The best results on AIDA-CoNLL were achieved with the downloadable files.

### Citation

If you use our work, please consider citing our upcoming paper about [ELEVANT](github.com/ad-freiburg/elevant) (will be put on ArXiv until August 1, 2022), as well as the GENRE paper.
