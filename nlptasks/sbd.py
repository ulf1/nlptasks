from typing import List
import warnings
import de_core_news_lg as spacy_model
import stanza
import nltk
import somajo


def factory(name: str):
    if name in ("spacy", "spacy-de"):
        return sbd_spacy_de
    elif name in ("spacy_rule", "spacy-rule-de"):
        return sbd_spacy_rule_de
    elif name in ("stanza", "stanza-de"):
        return sbd_stanza_de
    elif name in ("nltk_punkt", "nltk-punkt-de"):
        return sbd_nltk_punkt_de
    elif name in ("somajo", "somajo-de"):
        return sbd_somajo_de
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


def sbd_spacy_de(data: List[str], model=None) -> List[str]:
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
        tokens = sbd_spacy_de(X)
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


def sbd_spacy_rule_de(data: List[str], model=None) -> List[str]:
    """Rule-based SBD with spaCy Sentencizer

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.sbd.get_model
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


def sbd_stanza_de(data: List[str], model=None) -> List[str]:
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
        tokens = sbd_stanza_de(X)
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


def sbd_nltk_punkt_de(data: List[str], model=None) -> List[str]:
    """

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.sbd.get_model

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


def sbd_somajo_de(data: List[str], model=None) -> List[str]:
    """
    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.sbd.get_model
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
            ("" if token.token_class=="symbol" else " ") + token.text
            for token in sent]).strip()
        sentences.append(s)
    # done
    return sentences
