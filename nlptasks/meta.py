from pathlib import Path
import os
from datetime import datetime
# for package related info
import nlptasks
import de_core_news_lg as spacy_model
import spacy
import stanza
import nltk
import somajo
import flair
import someweta


def get(name: str, module: str = None) -> dict:
    """Meta information for the annotated/derivded data"""
    # store for later
    postproc = {
        'name': f"nlptasks.{module}" if module else "nlptasks",
        'version': nlptasks.__version__,
        'licence': 'Apache-2',
        'doi': '10.5281/zenodo.4284804'
    }

    # General Model information
    if name in ("spacy-de") and module is not None:
        if module in ("sbd"):
            used_pipes = ['parser']
        elif module in ("lemma"):
            used_pipes = []
        elif module in ("pos"):  # "pos2"
            used_pipes = ['tagger']
        elif module in ("ner"):  # "ner2"
            used_pipes = ['ner']
        elif module in ("dephead", "depchild", "deptree"):
            used_pipes = ['parser']

        info = {
            'pypi': {
                'name': 'spacy',
                'version': spacy.__version__,
                'licence': 'MIT',
                'doi': '10.5281/zenodo.1212303'
            },
            'model': {
                'name': spacy_model.__name__,
                'version': spacy_model.__version__,
                'used_pipes': used_pipes,
                'licence': 'MIT',
                'doi': '10.5281/zenodo.1212303'
            },
            'postproc': postproc
        }

    elif name in ("stanza-de") and module is not None:
        if module in ("sbd"):
            specs = {'processors': 'tokenize',
                     'tokenize_no_ssplit': False}
        elif module in ("lemma"):
            specs = {'processors': 'tokenize,lemma',
                     'tokenize_pretokenized': True}
        elif module in ("pos", "pos2"):
            specs = {'processors': 'tokenize,pos',
                     'tokenize_pretokenized': True}
        elif module in ("ner"):  # "ner2"
            specs = {'processors': 'tokenize,ner',
                     'tokenize_pretokenized': True}
        elif module in ("dephead", "deptree"):  # "depchild"
            specs = {'processors': 'tokenize,mwt,pos,lemma,depparse',
                     'tokenize_pretokenized': True}

        info = {
            'pypi': {
                'name': 'stanza',
                'version': stanza.__version__,
                'licence': 'Apache-2',
                'doi': '10.18653/v1/2020.acl-demos.14'
            },
            'model': {
                'lang': 'de',
                **specs,
                'licence': 'Apache-2',
                'doi': '10.18653/v1/2020.acl-demos.14'
            },
            'postproc': postproc
        }

    elif name in ('flair-de') and module in ('pos'):
        info = {
            'pypi': {
                'name': 'flair',
                'version': flair.__version__,
                'licence': 'MIT',
                'doi': '10.18653/v1/N19-4010'
            },
            'model': {
                'name': 'de-pos',
                'file': 'de-pos-ud-hdt-v0.5.pt',
                'licence': 'MIT',
                'doi': '10.18653/v1/N19-4010'
            },
            'postproc': postproc
        }

    elif name in ('flair-multi') and module in ('ner', 'ner2'):
        info = {
            'pypi': {
                'name': 'flair',
                'version': flair.__version__,
                'licence': 'MIT',
                'doi': '10.18653/v1/N19-4010'
            },
            'model': {
                'name': 'ner-multi',
                'file': 'quadner-large.pt',
                'licence': 'MIT',
                'doi': '10.18653/v1/N19-4010'
            },
            'postproc': postproc
        }

    elif name in ("spacy-rule-de") and module in ("sbd"):
        info = {
            'pypi': {
                'name': 'spacy',
                'version': spacy.__version__,
                'licence': 'MIT',
                'doi': '10.5281/zenodo.1212303'
            },
            'model': {
                'name': spacy_model.__name__,
                'version': spacy_model.__version__,
                'used_pipes': ['sentencizer'],
                'licence': 'MIT',
                'doi': '10.5281/zenodo.1212303'
            },
            'postproc': postproc
        }

    elif name in ("nltk-punkt-de") and module in ("sbd"):
        filepath = "nltk_data/tokenizers/punkt/PY3/german.pickle"
        filetime = os.path.getmtime(f"{str(Path.home())}/{filepath}")
        info = {
            'pypi': {
                'name': 'nltk',
                'version': nltk.__version__,
                'licence': 'Apache-2',
                'isbn': '9780596516499'
            },
            'model': {
                'name': 'punkt',
                'file': filepath,
                'modified': datetime.utcfromtimestamp(
                    filetime).strftime('%Y-%m-%dT%H:%M:%S'),
                'licence': 'Apache-2',
                'doi': '10.1162/coli.2006.32.4.485'
            },
            'postproc': postproc
        }

    elif name in ("somajo-de") and module in ("sbd"):
        info = {
            'pypi': {
                'name': 'SoMaJo',
                'version': somajo.__version__,
                'license': 'GPLv3',
                'doi': '10.18653/v1/W16-2607'
            },
            'model': {
                'language': 'de_CMC',
                'split_camel_case': True
            },
            'postproc': postproc
        }

    elif name in ("someweta-de", "someweta-web-de") and module in ("pos"):
        if name in ("someweta-de"):
            specs = {'file': ("http://corpora.linguistik.uni-erlangen.de/"
                              "someweta/german_newspaper_2020-05-28.model"),
                     'name': 'german_newspaper', 'modified': '2020-05-28'}
        elif name in ("someweta-web-de"):
            specs = {
                'file': ("http://corpora.linguistik.uni-erlangen.de/someweta/"
                         "german_web_social_media_2020-05-28.model"),
                'name': 'german_web_social_media', 'modified': '2020-05-28'}

        info = {
            'pypi': {
                'name': 'someweta',
                'version': someweta.__version__,
                'licence': 'GPLv3',
                'paper': 'https://www.aclweb.org/anthology/L18-1106'
            },
            'model': {
                **specs, 'licence': 'GPLv3',
                'paper': 'https://www.aclweb.org/anthology/L18-1106'
            },
            'postproc': postproc
        }

    else:
        raise Exception(f"Unknown model '{name}' or module '{module}'")

    # done
    return info
