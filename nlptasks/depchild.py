from .padding import pad_adjacmatrix
from typing import List, Tuple
import warnings
import de_core_news_lg as spacy_model
import spacy
import stanza


def factory(name: str):
    """Factory function to return a processing function for 
        dependency parsing and transformations of a token's
        children relationships as adjacency matrix.

    Parameters:
    -----------
    name : str
        Identifier, e.g. 'spacy-de'
    
    Example:
    --------
        import nlptasks as nt
        import nlptasks.depchild
        sequences = [['Die', 'Kuh', 'ist', 'bunt', '.']]
        myfn = nt.depchild.factory("spacy-de")
        maskseqs, seqlens = myfn(sequences)
    """
    if name in ("spacy", "spacy-de"):
        return spacy_de
    else:
        raise Exception(f"Unknown dependency parser: '{name}'")


def depchild_factory(name: str):
    warnings.warn(
        "Please call `nlptasks.depchild.factory` instead",
        DeprecationWarning, stacklevel=2)
    return factory(name)


def get_model(name: str):
    """Instantiate the pretrained model outside the depchild function
        so that it only needs to be done once

    Parameters:
    -----------
    name : str
        Identfier of the model

    Example:
    --------
        import nlptasks as nt
        import nlptasks.depchild
        model = nt.depchild.get_model('spacy-de')
        fn = nt.depchild.factory('spacy-de')
        maskseqs, seqlens = fn(sents, model=model)
    """
    if name in ("spacy", "spacy-de"):
        model = spacy_model.load()
        model.disable_pipes(["ner", "tagger"])
        return model
    else:
        raise Exception(f"Unknown dependency parser: '{name}'")


@pad_adjacmatrix
def spacy_de(data: List[List[str]], model=None) -> (
        List[List[Tuple[int, int]]], List[List[Tuple[int, int]]], List[int]):
    """Dependency relations with spaCy de_core_news_lg for German

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences
    
    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.depchild.get_model

    maxlen : Optional[int] = None
        see @nlptasks.padding.pad_adjacmatrix

    padding : Optional[str] = 'pre'
        see @nlptasks.padding.pad_adjacmatrix

    truncating : Optional[str] = 'pre'
        see @nlptasks.padding.pad_adjacmatrix

    Returns:
    --------
    maskseqs : List[List[Tuple[int, int]]]
        Index pairs of the adjacency matrix linking a token to children nodes

    seqlens : List[int]
        Length of each sequence that are also the matrix dimension of the
          adjacency matrix

    Example:
    --------
        import nlptasks as nt
        import nlptasks.depchild
        sequences = [['Die', 'Kuh', 'ist', 'bunt', '.']]
        maskseqs, seqlens = nt.depchild.spacy_de(
            sequences, maxlen=3, padding='pre', truncating='pre')
    """
    # (1) load spacy model
    if not model:
        model = spacy_model.load()
        model.disable_pipes(["ner", "tagger"])

    # parse dependencies of a pre-tokenized sentencens
    parser = model.pipeline[0][1]
    docs = [parser(spacy.tokens.doc.Doc(model.vocab, words=sequence))
            for sequence in data]

    # (3) Extract all (child, parent)-tuples
    def get_children_indicies(doc: spacy.tokens.doc.Doc):
        idxpairs = []
        for t in doc:
            idxpairs.extend([(c.i, t.i) for c in t.children])
        return idxpairs
    
    deps_child = [get_children_indicies(doc) for doc in docs]

    seqlens = [len(doc) for doc in docs]

    # done
    return deps_child, seqlens
