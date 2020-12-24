from .padding import pad_adjseqs
from typing import List, Tuple
import de_core_news_lg as spacy_model
import spacy
# import stanza


def deprel_factory(name: str):
    if name in ("spacy", "spacy-de"):
        return deprel_spacy_de
    # elif name == "stanza":
    #     return deprel_stanza_de
    # elif name == "imsnpars_zdl":
    #     return deprel_imsnpars_zdl
    else:
        raise Exception(f"Unknown dependency parser: '{name}'")


def get_model(name: str):
    """Instantiate the pretrained model outside the deprel function
        so that it only needs to be done once

    Parameters:
    -----------
    name : str
        Identfier of the model

    Example:
    --------
        from nlptasks.deprel import deprel
        model = deprel.get_model('spacy-de')
        fn = deprel.factory('spacy-de')
        dc, dp, sl = fn(sents, model=model)
    """
    if name in ("spacy", "spacy-de"):
        model = spacy_model.load()
        model.disable_pipes(["ner", "tagger"])
        parser = model.pipeline[0][1]
        return model
    else:
        raise Exception(f"Unknown dependency parser: '{name}'")


@pad_adjseqs
def deprel_spacy_de(data: List[List[str]], model=None) -> (
        List[List[Tuple[int, int]]], List[List[Tuple[int, int]]], List[int]):
    """Dependency relations with spaCy de_core_news_lg for German

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences
    
    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.deprel.get_model

    Returns:
    --------
    deps_child : List[List[Tuple[int, int]]]
        Index pairs of the adjacency matrix linking a token to children nodes

    deps_parent : List[List[Tuple[int, int]]]
        Index pairs of the adjacency matrix linking a token to parent nodes

    seqlens : List[int]
        Length of each sequence that are also the matrix dimension of the
          adjacency matrix

    Example:
    --------
        deps_child, deps_parent, seqlens = deprel_spacy_de(tokens)
    """
    # (1) load spacy model
    if not model:
        model = spacy_model.load()
        model.disable_pipes(["ner", "tagger"])
        parser = model.pipeline[0][1]

    # parse dependencies of a pre-tokenized sentencens
    docs = [parser(spacy.tokens.doc.Doc(model.vocab, words=sequence))
            for sequence in data]

    # (3) Extract all (child, parent)-tuples
    def get_children_indicies(doc: spacy.tokens.doc.Doc):
        idxpairs = []
        for t in doc:
            idxpairs.extend([(c.i, t.i) for c in t.children])
        return idxpairs
    
    deps_child = [get_children_indicies(doc) for doc in docs]

    deps_parent = [[(t.head.i, t.i) for t in doc] for doc in docs]

    seqlens = [len(doc) for doc in docs]

    # done
    return deps_child, deps_parent, seqlens
