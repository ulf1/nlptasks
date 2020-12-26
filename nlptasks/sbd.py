from typing import List
import warnings
from pathlib import Path
import os
from datetime import datetime
import de_core_news_lg as spacy_model
import spacy
import stanza
import nltk
import somajo


def factory(name: str):
    """Factory function to return a processing function for
        Sentence Boundary Disambiguation

    Parameters:
    -----------
    name : str
        Identifier, e.g. 'spacy-de', 'spacy-rule-de', 'stanza-de',
          'nltk-punkt-de', 'somajo-de'

    Example:
    --------
        import nlptasks as nt
        import nlptasks.sbd
        docs = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
        myfn = nt.sbd.factory(name="somajo")
        sents = myfn(docs)
    """
    if name in ("spacy", "spacy-de"):
        return spacy_de
    elif name in ("spacy_rule", "spacy-rule-de"):
        return spacy_rule_de
    elif name in ("stanza", "stanza-de"):
        return stanza_de
    elif name in ("nltk_punkt", "nltk-punkt-de"):
        return nltk_punkt_de
    elif name in ("somajo", "somajo-de"):
        return somajo_de
    else:
        raise Exception(f"Unknown SBD function: '{name}'")


def sbd_factory(name: str):
    warnings.warn(
        "Please call `nlptasks.sbd.factory` instead",
        DeprecationWarning, stacklevel=2)
    return factory(name)


def get_model(name: str):
    """Instantiate the pretrained model outside the SBD function
        so that it only needs to be done once

    Parameters:
    -----------
    name : str
        Identfier of the model

    Example:
    --------
        from nlptasks.sbd import sbd
        model = sbd.get_model('stanza-de')
        sbd_fn = sbd.factory('stanza-de')
        docs = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
        sentences = sbd_fn(docs, model=model)
    """
    if name in ("spacy", "spacy-de"):
        model = spacy_model.load()
        model.disable_pipes(["ner", "tagger"])
        return model

    elif name in ("spacy_rule", "spacy-rule-de"):
        model = spacy_model.load()
        model.disable_pipes(["ner", "parser", "tagger"])
        model.add_pipe(model.create_pipe('sentencizer'))
        return model

    elif name in ("stanza", "stanza-de"):
        return stanza.Pipeline(
            lang='de', processors='tokenize',
            tokenize_no_ssplit=False)

    elif name in ("nltk_punkt", "nltk-punkt-de"):
        return None

    elif name in ("somajo", "somajo-de"):
        return somajo.SoMaJo("de_CMC", split_camel_case=True)

    else:
        raise Exception(f"Unknown SBD function: '{name}'")


def meta(name: str) -> dict:
    """Meta information for the annotated/derivded data"""
    if name in ("spacy", "spacy-de"):
        return {
            'pypi': {
                'name': 'spacy',
                'version': spacy.__version__,
                'licence': 'MIT',
                'doi': '10.5281/zenodo.1212303'
            },
            'model': {
                'name': spacy_model.__name__,
                'version': spacy_model.__version__,
                'used_pipes': ['parser'],
                'licence': 'MIT',
                'doi': '10.5281/zenodo.1212303'
            }
        }
    elif name in ("spacy_rule", "spacy-rule-de"):
        return {
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
            }
        }
    elif name in ("stanza", "stanza-de"):
        return {
            'pypi': {
                'name': 'stanza',
                'version': stanza.__version__,
                'licence': 'Apache-2',
                'doi': '10.18653/v1/2020.acl-demos.14'
            },
            'model': {
                'lang': 'de',
                'processors': 'tokenize',
                'tokenize_no_ssplit': False,
                'licence': 'Apache-2',
                'doi': '10.18653/v1/2020.acl-demos.14'
            }
        }
    elif name in ("nltk_punkt", "nltk-punkt-de"):
        filepath = "nltk_data/tokenizers/punkt/PY3/german.pickle"
        return {
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
                    os.path.getmtime(f"{str(Path.home())}/{filepath}")
                    ).strftime('%Y-%m-%dT%H:%M:%S'),
                'licence': 'Apache-2',
                'doi': '10.1162/coli.2006.32.4.485'
            }
        }
    elif name in ("somajo", "somajo-de"):
        return {
            'pypi': {
                'name': 'SoMaJo',
                'version': somajo.__version__,
                'license': 'GPLv3',
                'doi': '10.18653/v1/W16-2607'
            },
            'model': {
                'language': 'de_CMC', 
                'split_camel_case': True
            }
        }
    else:
        raise Exception(f"Unknown SBD function: '{name}'")



def spacy_de(data: List[str], model=None) -> List[str]:
    """SBD with spaCy de_core_news_lg based on DependencyParser

    Parameters:
    -----------
    data : List[str]
        list of N documents as strings. Each document is then segmented
          into sentences.

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.sbd.get_model

    Returns:
    --------
    List[str]
        list of M sentences as strings. Pls note that the information
          about the relationship to the document is lost.

    Example:
    --------
        import nlptasks as nt
        import nlptasks.sbd
        docs = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
        sents = nt.sbd.spacy_de(docs)
    """
    # load spacy
    if not model:
        model = spacy_model.load()
        model.disable_pipes(["ner", "tagger"])
    # SBD
    sentences = []
    for rawstr in data:
        sentences.extend([s.text for s in model(rawstr).sents])
    # done
    return sentences


def spacy_rule_de(data: List[str], model=None) -> List[str]:
    """Rule-based SBD with spaCy Sentencizer

    Parameters:
    -----------
    data : List[str]
        list of N documents as strings. Each document is then segmented
          into sentences.

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.sbd.get_model

    Returns:
    --------
    List[str]
        list of M sentences as strings. Pls note that the information
          about the relationship to the document is lost.

    Example:
    --------
        import nlptasks as nt
        import nlptasks.sbd
        docs = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
        sents = nt.sbd.spacy_rule_de(docs)
    """
    # load spacy
    if not model:
        model = spacy_model.load()
        model.disable_pipes(["ner", "parser", "tagger"])
        model.add_pipe(model.create_pipe('sentencizer'))
    # SBD
    sentences = []
    for rawstr in data:
        sentences.extend([s.text for s in model(rawstr).sents])
    # done
    return sentences


def stanza_de(data: List[str], model=None) -> List[str]:
    """Sentence Segmentation (SBD) with stanza for German

    Parameters:
    -----------
    data : List[str]
        list of N documents as strings. Each document is then segmented
          into sentences.

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.sbd.get_model

    Returns:
    --------
    List[str]
        list of M sentences as strings. Pls note that the information
          about the relationship to the document is lost.

    Example:
    --------
        import nlptasks as nt
        import nlptasks.sbd
        docs = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
        sents = nt.sbd.stanza_de(docs)
    """
    # load stanza
    if not model:
        model = stanza.Pipeline(
            lang='de', processors='tokenize',
            tokenize_no_ssplit=False)
    # SBD
    sentences = []
    for rawstr in data:
        sentences.extend([s.text for s in model(rawstr).sentences])
    # done
    return sentences


def nltk_punkt_de(data: List[str], model=None) -> List[str]:
    """Sentence Segmentation (SBD) with NLTK's Punct Tokenizer

    Parameters:
    -----------
    data : List[str]
        list of N documents as strings. Each document is then segmented
          into sentences.

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.sbd.get_model

    Returns:
    --------
    List[str]
        list of M sentences as strings. Pls note that the information
          about the relationship to the document is lost.

    Example:
    --------
        import nlptasks as nt
        import nlptasks.sbd
        docs = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
        sents = nt.sbd.nltk_punkt_de(docs)

    Help:
    -----
    - https://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.punkt
    """
    # SBD
    sentences = []
    for rawstr in data:
        sents = nltk.tokenize.sent_tokenize(rawstr, language="german")
        sentences.extend(sents)
    # done
    return sentences


def somajo_de(data: List[str], model=None) -> List[str]:
    """Sentence Segmentation (SBD) with SoMaJo, German

    Parameters:
    -----------
    data : List[str]
        list of N documents as strings. Each document is then segmented
          into sentences.

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.sbd.get_model

    Returns:
    --------
    List[str]
        list of M sentences as strings. Pls note that the information
          about the relationship to the document is lost.

    Example:
    --------
        import nlptasks as nt
        import nlptasks.sbd
        docs = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
        sents = nt.sbd.somajo_de(docs)
    """
    # instantiate
    if not model:
        model = somajo.SoMaJo("de_CMC", split_camel_case=True)
    # segment all docs (returns a generator)
    sentsgen = model.tokenize_text(data)
    # loop over all sentences to reconstruct the sentence
    sentences = []
    for sent in sentsgen:
        s = "".join([
            ("" if token.token_class == "symbol" else " ") + token.text
            for token in sent]).strip()
        sentences.append(s)
    # done
    return sentences
