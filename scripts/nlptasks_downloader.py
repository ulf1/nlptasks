#!/usr/bin/env python3
import os
import gc
import stanza
from pathlib import Path
import requests
import flair


if __name__ == '__main__':
    # spacy data
    os.system('python -m spacy download de_core_news_lg-2.3.0 --direct')

    # stanza
    stanza.download('de')

    # flair - trigger initial download into cache
    tagger = flair.models.SequenceTagger.load('ner-multi')
    tagger = flair.models.SequenceTagger.load('de-pos')
    del tagger
    gc.collect()

    # nltk
    PATH_NLTK = f"{str(Path.home())}/nltk_data/tokenizers"
    URL_NLTK = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip"
    os.makedirs(PATH_NLTK, exist_ok=True)
    os.system(f"wget -O '{PATH_NLTK}/punkt.zip' '{URL_NLTK}'")
    os.system(f"unzip -o -d '{PATH_NLTK}' '{PATH_NLTK}/punkt.zip'")

    # SoMeWeTa
    PATH = f"{str(Path.home())}/someweta_data"
    URL1 = "http://corpora.linguistik.uni-erlangen.de/someweta/german_newspaper_2020-05-28.model"
    URL2 = "http://corpora.linguistik.uni-erlangen.de/someweta/german_web_social_media_2020-05-28.model"
    os.makedirs(PATH, exist_ok=True)
    os.system(f"wget -O '{PATH}/german_newspaper.model' '{URL1}'")
    os.system(f"wget -O '{PATH}/german_web_social_media.model' '{URL2}'")
