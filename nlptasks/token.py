from typing import List
import warnings
import de_core_news_lg as spacy_model
import stanza


def factory(name: str):
    """Factory function to return a processing function for 
        tokenization

    Parameters:
    -----------
    name : str
        Identifier, e.g. 'spacy-de', 'stanza-de'
    
    Example:
    --------
        import nlptasks as nt
        import nlptasks.deprel
        sequences = [['Die', 'Kuh', 'ist', 'bunt', '.']]
        myfn = nt.token.factory(name="stanza-de")
        sequences = myfn(sentences)
    """
    if name in ("spacy", "spacy-de"):
        return spacy_de
    elif name in ("stanza", "stanza-de"):
        return stanza_de
    else:
        raise Exception(f"Unknown Tokenizer function: '{name}'") 


def token_factory(name: str):
    warnings.warn(
        "Please call `nlptasks.token.factory` instead",
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
        from nlptasks.ner import token
        model = token.get_model('stanza-de')
        fn = token.factory('stanza-de')
        tokens = fn(docs, model=model)
    """
    if name in ("spacy", "spacy-de"):
        model = spacy_model.load()
        model.disable_pipes(["ner", "parser", "tagger"])
        return model

    elif name in ("stanza", "stanza-de"):
        return stanza.Pipeline(
            lang='de', processors='tokenize', tokenize_no_ssplit=True)

    else:
        raise Exception(f"Unknown Tokenizer function: '{name}'") 


def spacy_de(data: List[str], model=None) -> List[List[str]]:
    """Word Tokenization with spaCy de_core_news_lg for German

    Parameters:
    -----------
    data : List[str]
        List of M sentences as string.
    
    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.token.get_model

    Returns:
    --------
    List[List[str]]
        List of token/word sequences.

    Example:
    --------
        import nlptasks as nt
        tokens = nt.token.spacy_de(X)
    """
    # load spacy
    if not model:
        model = spacy_model.load()
        model.disable_pipes(["ner", "parser", "tagger"])
    # tokenize
    tokens = [[t.text for t in model(s)] for s in data]
    # done
    return tokens


def stanza_de(data: List[str], model=None) -> List[List[str]]:
    """Word Tokenization with stanza for German

    Parameters:
    -----------
    data : List[str]
        List of M sentences as string.
    
    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.token.get_model

    Returns:
    --------
    List[List[str]]
        List of token/word sequences.

    Example:
    --------
        import nlptasks as nt
        tokens = nt.token.stanza_de(X)
    """
    # load stanza
    if not model:
        model = stanza.Pipeline(
            lang='de', processors='tokenize', tokenize_no_ssplit=True)
    # tokenize
    tokens = []
    for s in data:
        for sent in model(s).sentences:
            tokens.append([t.text for t in sent.tokens])
    # done
    return tokens
