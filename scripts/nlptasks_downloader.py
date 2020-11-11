#!/usr/bin/env python3
import os
import stanza
from pathlib import Path
import requests


if __name__ == '__main__':
    # spacy data
    os.system('python -m spacy download de_core_news_lg-2.3.0 --direct')

    # stanza
    stanza.download('de')

    # nltk
    PATH_NLTK = f"{str(Path.home())}/nltk_data/tokenizers"
    URL_NLTK = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip"
    os.makedirs(PATH_NLTK, exist_ok=True)
    os.system(f"wget -O '{PATH_NLTK}/punkt.zip' '{URL_NLTK}'")
    os.system(f"unzip -o -d '{PATH_NLTK}' '{PATH_NLTK}/punkt.zip'")
