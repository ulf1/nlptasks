from .padding import pad_merge_adjac_maskseqs
from .vocab import texttoken_to_index
from typing import List, Tuple
import warnings
import de_core_news_lg as spacy_model
import spacy
# import stanza


# https://universaldependencies.org/u/dep/index.html
UD2_RELS = [
    'acl', 'acl:relcl', 'advcl', 'advmod', 'advmod:emph', 'amod', 'appos',
    'aux', 'aux:pass', 'case', 'cc', 'cc:preconj', 'ccomp', 'clf', 'compound',
    'compound:lvc', 'compound:prt', 'compound:redup', 'compound:svc', 'conj',
    'cop', 'csubj', 'csubj:pass', 'dep', 'det', 'det:numgov', 'det:nummod',
    'det:poss', 'discourse', 'dislocated', 'expl', 'expl:impers', 'expl:pass',
    'expl:pv', 'fixed', 'flat', 'flat:foreign', 'flat:name', 'goeswith',
    'iobj', 'list', 'mark', 'nmod', 'nmod:poss', 'nmod:tmod', 'nsubj',
    'nsubj:pass', 'nummod', 'nummod:gov', 'obj', 'obl', 'obl:agent', 'obl:arg',
    'obl:tmod', 'orphan', 'parataxis', 'punct', 'reparandum', 'root',
    'vocative', 'xcomp'
]


TIGER_RELS = [
    'ac', 'adc', 'ag', 'ams', 'app', 'avc', 'cc', 'cd', 'cj', 'cm', 'cp',
    'cvc', 'da', 'dm', 'ep', 'ju', 'mnr', 'mo', 'ng', 'nk', 'nmc', 'oa',
    'oa2', 'oc', 'og', 'op', 'par', 'pd', 'pg', 'ph', 'pm', 'pnc', 'punct',
    'rc', 're', 'rs', 'sb', 'sbp', 'sp', 'svp', 'uc', 'vo', 'ROOT'
]


def factory(name: str):
    """Factory function to return a processing function for 
        dependency parsing

    Parameters:
    -----------
    name : str
        Identifier, e.g. 'spacy-de'
    
    Example:
    --------
        import nlptasks as nt
        import nlptasks.deprel
        sequences = [['Die', 'Kuh', 'ist', 'bunt', '.']]
        myfn = nt.deprel.factory("spacy-de")
        deps_child, deps_parent, seqlens = myfn(sequences)
    """
    if name in ("spacy", "spacy-de"):
        return spacy_de
    # elif name == "stanza":
    #     return stanza_de
    # elif name == "imsnpars_zdl":
    #     return imsnpars_zdl
    else:
        raise Exception(f"Unknown dependency parser: '{name}'")


def deprel_factory(name: str):
    warnings.warn(
        "Please call `nlptasks.deprel.factory` instead",
        DeprecationWarning, stacklevel=2)
    return factory(name)


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
        return model
    else:
        raise Exception(f"Unknown dependency parser: '{name}'")


@pad_merge_adjac_maskseqs
def spacy_de(data: List[List[str]], model=None) -> (
        List[List[Tuple[int, int]]], List[List[Tuple[int, int]]], List[int]):
    """Dependency relations with spaCy de_core_news_lg for German

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences
    
    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.deprel.get_model

    maxlen : Optional[int] = None
        see @nlptasks.padding.pad_maskseqs

    padding : Optional[str] = 'pre'
        see @nlptasks.padding.pad_maskseqs

    truncating : Optional[str] = 'pre'
        see @nlptasks.padding.pad_maskseqs

    Returns:
    --------
    maskseqs : List[List[Tuple[int, int]]]
        Sequences with token-parent relations and the one-hot encoded
          dependency type

    seqlens : List[int]
        Length of each sequence that are also the matrix dimension of the
          adjacency matrix

    Example:
    --------
        import nlptasks as nt
        import nlptasks.dephead
        sequences = [['Die', 'Kuh', 'ist', 'bunt', '.']]
        maskseqs, seqlens = nt.dephead.spacy_de(
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

    adjac_parent = [[(t.head.i, t.i) for t in doc] for doc in docs]
    rel_types = [[t.dep_ for t in doc] for doc in docs]
    seqlens = [len(doc) for doc in docs]

    # (2) Define TIGER RELATIONS as VOCAB
    SCHEME = TIGER_RELS.copy()
    SCHEME.append("[UNK]")

    # (3) Encode deprel tags
    rel_types = [texttoken_to_index(seq, SCHEME) for seq in rel_types]
    onehot_types = [[(ri, ti) for ti, ri in enumerate(sent)] for sent in rel_types]

    # done
    return adjac_parent, onehot_types, seqlens, len(SCHEME)
