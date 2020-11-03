#!/bin/bash

# start venv
source .venv/bin/activate

# spacy data
python -m spacy download de_core_news_sm-2.3.0 --direct
python -m spacy download de_core_news_md-2.3.0 --direct
python -m spacy download de_core_news_lg-2.3.0 --direct

# stanza
python -c "import stanza; stanza.download('de')"

# nltk
#python -c "import nltk; nltk.download('punkt')"
mkdir -p "${HOME}/nltk_data/tokenizers"
wget -O "${HOME}/nltk_data/tokenizers/punkt.zip" "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip"
unzip -o -d "${HOME}/nltk_data/tokenizers/" "${HOME}/nltk_data/tokenizers/punkt.zip" 

# Download test data
mkdir -p data/lpc
folder=$(pwd)
cd /tmp
wget -O /tmp/lpc-tmp.tar.gz https://pcai056.informatik.uni-leipzig.de/downloads/corpora/deu_news_2015_100K.tar.gz
tar -xvf /tmp/lpc-tmp.tar.gz
mv /tmp/deu_news_2015_100K/deu_news_2015_100K-sentences.txt "$folder/data/lpc/deu_news_2015_100K-sentences.txt"
cd $folder

