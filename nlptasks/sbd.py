from typing import List
import de_core_news_lg as spacy_model
import stanza


def sbd_spacy_de(data: List[str]) -> List[str]:
    """SBD with spaCy de_core_news_lg

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
