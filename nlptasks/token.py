from typing import List
import de_core_news_lg as spacy_model
import stanza


def token_factory(name: str):
    if name == "spacy":
        return token_spacy_de
    elif name == "stanza":
        return token_stanza_de
    else:
        raise Exception(f"Unknown Tokenizer function: '{name}'") 


def token_spacy_de(data: List[str]) -> List[List[str]]:
    """Word Tokenization with spaCy de_core_news_lg for German

    Parameters:
    -----------
    data : List[str]
        List of M sentences as string.
    
    Returns:
    --------
    List[List[str]]
        List of token/word sequences.

    Example:
    --------
        tokens = token_spacy_de(X)
    """
    # load spacy
    nlp = spacy_model.load()
    nlp.disable_pipes(["ner", "parser", "tagger"])
    # tokenize
    tokens = [[t.text for t in nlp(s)] for s in data]
    # done
    return tokens


def token_stanza_de(data: List[str]) -> List[List[str]]:
    """Word Tokenization with stanza for German

    Parameters:
    -----------
    data : List[str]
        List of M sentences as string.
    
    Returns:
    --------
    List[List[str]]
        List of token/word sequences.

    Example:
    --------
        tokens = token_stanza_de(X)
    """
    # load stanza
    nlp = stanza.Pipeline(lang='de',
                          processors='tokenize',
                          tokenize_no_ssplit=True)
    # tokenize
    tokens = []
    for s in data:
        for sent in nlp(s).sentences:
            tokens.append([t.text for t in sent.tokens])
    # done
    return tokens
