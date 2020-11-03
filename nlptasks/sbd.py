from typing import List
import de_core_news_lg as spacy_model
import stanza
import nltk
import somajo


def sbd_factory(name: str):
    if name == "spacy":
        return sbd_spacy_de
    elif name == "spacy_rule":
        return sbd_spacy_rule_de
    elif name == "stanza":
        return sbd_stanza_de
    elif name == "nltk_punkt":
        return sbd_nltk_punkt_de
    elif name == "somajo":
        return sbd_somajo_de
    else:
        raise Exception(f"Unknown SBD function: '{name}'") 


def sbd_spacy_de(data: List[str]) -> List[str]:
    """SBD with spaCy de_core_news_lg based on DependencyParser

    Parameters:
    -----------
    data : List[str]
        list of N documents as strings. Each document is then segmented
          into sentences.
    
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
    nlp = spacy_model.load()
    nlp.disable_pipes(["ner", "tagger"])
    # SBD
    sentences = []
    for rawstr in data:
        sentences.extend([s.text for s in nlp(rawstr).sents])
    # done
    return sentences


def sbd_spacy_rule_de(data: List[str]) -> List[str]:
    """Rule-based SBD with spaCy Sentencizer"""
    # load spacy
    nlp = spacy_model.load()
    nlp.disable_pipes(["ner", "parser", "tagger"])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    # SBD
    sentences = []
    for rawstr in data:
        sentences.extend([s.text for s in nlp(rawstr).sents])
    # done
    return sentences


def sbd_stanza_de(data: List[str]) -> List[str]:
    """Sentence Segmentation (SBD) with stanza for German

    Parameters:
    -----------
    data : List[str]
        list of N documents as strings. Each document is then segmented
          into sentences.
    
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
    nlp = stanza.Pipeline(lang='de',
                          processors='tokenize',
                          tokenize_no_ssplit=False)
    # SBD
    sentences = []
    for rawstr in data:
        sentences.extend([s.text for s in nlp(rawstr).sentences])
    # done
    return sentences


def sbd_nltk_punkt_de(data: List[str]) -> List[str]:
    # SBD
    sentences = []
    for rawstr in data:
        sents = nltk.tokenize.sent_tokenize(rawstr, language="german")
        sentences.extend(sents)
    # done
    return sentences


def sbd_somajo_de(data: List[str]) -> List[str]:
    # instantiate 
    tokenizer = somajo.SoMaJo("de_CMC", split_camel_case=True)
    # segment all docs (returns a generator)
    sentsgen = tokenizer.tokenize_text(data)
    # loop over all sentences to reconstruct the sentence
    sentences = []
    for sent in sentsgen:
        s = "".join([
            ("" if token.token_class=="symbol" else " ") + token.text
            for token in sent]).strip()
        sentences.append(s)
    # done
    return sentences
