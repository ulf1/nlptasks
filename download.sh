#!/bin/bash

# start venv
source .venv/bin/activate

# spacy data
python -m spacy download de_core_news_sm-2.3.0 --direct
python -m spacy download de_core_news_md-2.3.0 --direct
python -m spacy download de_core_news_lg-2.3.0 --direct

# stanza
python -c "import stanza; stanza.download('de')"

# flair

# nltk
