from typing import List, Optional
import treesimi as ts
import json
import hashlib
from .vocab import identify_vocab_mincount, texttoken_to_index
import itertools
import de_core_news_lg as spacy_model
import spacy
import stanza


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
    elif name in ("stanza", "stanza-de"):
        return stanza_de
    # elif name == "imsnpars_zdl":
    #     return imsnpars_zdl
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
        return model

    elif name in ("stanza", "stanza-de"):
        return stanza.Pipeline(
            lang='de', processors='tokenize,mwt,pos,lemma,depparse',
            tokenize_pretokenized=True)
    else:
        raise Exception(f"Unknown dependency parser: '{name}'")


def spacy_de(data: List[List[str]],
             return_mask: bool = False,
             VOCAB: Optional[List[hashlib.sha512]] = None,
             min_occurrences: Optional[int] = 1, 
             model=None,
             use_trunc_leaves: Optional[bool] = False,
             use_drop_nodes: Optional[bool] = False,
             use_replace_attr: Optional[bool] = False,
             placeholder: Optional[str] = '\uFFFF'
             ) -> (List[List[str]], List[str]):
    """Dependency relations with spaCy de_core_news_lg for German

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences
    
    return_mask: bool = False
        Flag if mask vectors should be returned instead of indices.
    
    VOCAB: Optional[List[hashlib.sha512]] = None
        A given list of python sha512 objects that are used as ID

    min_occurrences : int
        (Optional) The required number of occurences in a corpus.

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.deprel.get_model

    use_trunc_leaves: Optional[bool] = False
        see treesimi.shingleset
    
    use_drop_nodes: Optional[bool] = False
        see treesimi.shingleset
    
    use_replace_attr: Optional[bool] = False
        see treesimi.shingleset
    
    placeholder: Optional[str] = '\uFFFF'
        see treesimi.shingleset

    Returns:
    --------
    indices : List[List[int]]
        For each sentences, a list of mask indices

    VOCAB: Optional[List[hashlib.sha512]] = None
        A given list of python sha512 objects that are used as ID

    Example:
    --------
        import nlptasks as nt
        import nlptasks.deptree
        sequences = [['Die', 'Kuh', 'ist', 'bunt', '.']]
        indices, VOCAB = nt.deptree.spacy_de(sequences)
    """
    # (1) load spacy model
    if not model:
        model = spacy_model.load()
        model.disable_pipes(["ner", "tagger"])

    # parse dependencies of a pre-tokenized sentencens
    parser = model.pipeline[0][1]
    docs = [parser(spacy.tokens.doc.Doc(model.vocab, words=sequence))
            for sequence in data]
    adjac = [[(t.i + 1, 0 if t.dep_ == 'ROOT' else t.head.i + 1, t.dep_)
               for t in doc] for doc in docs]

    # (2a) convert to nested set models
    cfg = {
        'use_trunc_leaves': use_trunc_leaves, 
        'use_drop_nodes': use_drop_nodes, 
        'use_replace_attr': use_replace_attr,
        'placeholder': placeholder}
    nested = [ts.adjac_to_nested_with_attr(tree) for tree in adjac]
    nested = [ts.remove_node_ids(tree) for tree in nested]

    # (2b) shingling and hashing
    shingled = [ts.shingleset(tree, **cfg) for tree in nested]
    encoded = [[json.dumps(tmp).encode('utf-8') for tmp in sent]
               for sent in shingled]
    hashed = [[hashlib.sha512(enc).hexdigest() for enc in sent]
              for sent in encoded]

    # (3) Identify VOCAB
    if VOCAB is None:
        VOCAB = identify_vocab_mincount(
            data=list(itertools.chain.from_iterable(hashed)),
            min_occurrences=min_occurrences, sort=False)

    # (4) Encode hashed trees to mask indices
    unkid = len(VOCAB)
    indices = [texttoken_to_index(ex, VOCAB) for ex in hashed]
    indices = [[i for i in ex if i != unkid] for ex in indices]

    # choose output format
    if return_mask:
        masked = [[int(i in ex) for i in range(unkid)] for ex in indices]
        return masked, VOCAB
    else:
        return indices, VOCAB


def stanza_de(data: List[List[str]],
              return_mask: bool = False,
              VOCAB: Optional[List[hashlib.sha512]] = None,
              min_occurrences: Optional[int] = 1, 
              model=None,
              use_trunc_leaves: Optional[bool] = False,
              use_drop_nodes: Optional[bool] = False,
              use_replace_attr: Optional[bool] = False,
              placeholder: Optional[str] = '\uFFFF'
              ) -> (List[List[int]], List[str]):
    """Dependency relations with stanza for German

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences

    return_mask: bool = False
        Flag if mask vectors should be returned instead of indices.
    
    VOCAB: Optional[List[hashlib.sha512]] = None
        A given list of python sha512 objects that are used as ID

    min_occurrences : int
        (Optional) The required number of occurences in a corpus.

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.deprel.get_model

    use_trunc_leaves: Optional[bool] = False
        see treesimi.shingleset
    
    use_drop_nodes: Optional[bool] = False
        see treesimi.shingleset
    
    use_replace_attr: Optional[bool] = False
        see treesimi.shingleset
    
    placeholder: Optional[str] = '\uFFFF'
        see treesimi.shingleset
    
    Returns:
    --------
    indices : List[List[int]]
        For each sentences, a list of mask indices.
        If 

    VOCAB: Optional[List[hashlib.sha512]] = None
        A given list of python sha512 objects that are used as ID

    Example:
    --------
        import nlptasks as nt
        import nlptasks.deptree
        sequences = [['Die', 'Kuh', 'ist', 'bunt', '.'],
                     ['Die', 'Kühe', 'sind', 'grau', '.']]
        indices, VOCAB = nt.deptree.stanza_de(sequences)
    """
    # (1) load spacy model
    if not model:
        model = stanza.Pipeline(
            lang='de', processors='tokenize,mwt,pos,lemma,depparse',
            tokenize_pretokenized=True)

    # parse dependencies of a pre-tokenized sentencens
    docs = model(data)
    adjac = [[(t.id, t.head, t.deprel) for t in sent.words]
             for sent in docs.sentences]

    # (2a) convert to nested set models
    nested = [ts.adjac_to_nested_with_attr(tree) for tree in adjac]
    nested = [ts.remove_node_ids(tree) for tree in nested]

    # (2b) shingling and hashing
    cfg = {
        'use_trunc_leaves': use_trunc_leaves, 
        'use_drop_nodes': use_drop_nodes, 
        'use_replace_attr': use_replace_attr,
        'placeholder': placeholder}
    shingled = [ts.shingleset(tree, **cfg) for tree in nested]
    encoded = [[json.dumps(tmp).encode('utf-8') for tmp in sent]
               for sent in shingled]
    hashed = [[hashlib.sha512(enc).hexdigest() for enc in sent]
              for sent in encoded]

    # (3) Identify VOCAB
    if VOCAB is None:
        VOCAB = identify_vocab_mincount(
            data=list(itertools.chain.from_iterable(hashed)),
            min_occurrences=min_occurrences, sort=False)

    # (4) Encode hashed trees to mask indices
    unkid = len(VOCAB)
    indices = [texttoken_to_index(ex, VOCAB) for ex in hashed]
    indices = [[i for i in ex if i != unkid] for ex in indices]

    # choose output format
    if return_mask:
        masked = [[int(i in ex) for i in range(unkid)] for ex in indices]
        return masked, VOCAB
    else:
        return indices, VOCAB
