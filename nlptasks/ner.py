from .padding import pad_idseqs
from typing import List
from .vocab import texttoken_to_index
import warnings
import de_core_news_lg as spacy_model
import spacy
import stanza
import flair


CONLL03_SCHEME = ['PER', 'LOC', 'ORG', 'MISC']


def factory(name: str):
    """Factory function to return a processing function for
        Named Entity Recognition

    Parameters:
    -----------
    name : str
        Identifier, e.g. 'spacy-de', 'flair-multi', 'stanza-de'

    Example:
    --------
        import nlptasks as nt
        import nlptasks.ner
        sequences = [['Die', 'Kuh', 'ist', 'bunt', '.']]
        myfn = nt.ner.factory("spacy-de")
        idseqs, SCHEME = myfn(sequences)
    """
    if name in ("spacy", "spacy-de"):
        return spacy_de
    elif name == "flair-multi":
        return flair_multi
    elif name in ("stanza", "stanza-de"):
        return stanza_de
    else:
        raise Exception(f"Unknown NER tagger: '{name}'")


def ner_factory(name: str):
    warnings.warn(
        "Please call `nlptasks.ner.factory` instead",
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
        from nlptasks.ner import ner
        model = ner.get_model('stanza-de')
        fn = ner.factory('stanza-de')
        idseqs, SCHEME = fn(docs, model=model)
    """
    if name in ("spacy", "spacy-de"):
        model = spacy_model.load()
        model.disable_pipes(["parser", "tagger"])
        return model

    elif name == "flair-multi":
        return flair.models.SequenceTagger.load('ner-multi')

    elif name in ("stanza", "stanza-de"):
        return stanza.Pipeline(
            lang='de', processors='tokenize,ner',
            tokenize_pretokenized=True)

    else:
        raise Exception(f"Unknown NER tagger: '{name}'")


@pad_idseqs
def spacy_de(data: List[List[str]], model=None) -> (
        List[List[str]], List[str]):
    """NER with spaCy de_core_news_lg for German with Wikipedia NER Scheme

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.ner.get_model

    Returns:
    --------
    sequences : List[List[int]]
        List of ID sequences wheras an ID relates to

    scheme : List[str]
        Wikipedia NER Scheme. Implizit ID:NERtags mappings
        (It's the VOCAB for Embeddings)

    Example:
    --------
        nertags, SCHEME = nt.ner.spacy_de(tokens)
    """
    # (1) load spacy model
    if not model:
        model = spacy_model.load()
        model.disable_pipes(["parser", "tagger"])

    # NER recognize a pre-tokenized sentencens
    ner = model.pipeline[0][1]
    docs = [ner(spacy.tokens.doc.Doc(model.vocab, words=sequence))
            for sequence in data]
    nertags = [[t.ent_type_ for t in doc] for doc in docs]

    # (2) Define the WIKINER tagset as VOCAB
    SCHEME = CONLL03_SCHEME.copy()
    SCHEME.append("[UNK]")

    # (3) convert WIKI NER tags to a sequence of IDs
    nertags_ids = [texttoken_to_index(seq, SCHEME) for seq in nertags]

    # done
    return nertags_ids, SCHEME


@pad_idseqs
def flair_multi(data: List[List[str]], model=None) -> (
        List[List[str]], List[str]):
    """flair 'multi-ner', CoNLL-03 NE scheme, returns ID sequence
        for embeddings.

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.ner.get_model

    maxlen : Optional[int] = None
        see @nlptasks.padding.pad_idseqs

    padding : Optional[str] = 'pre'
        see @nlptasks.padding.pad_idseqs

    truncating : Optional[str] = 'pre'
        see @nlptasks.padding.pad_idseqs

    Returns:
    --------
    sequences : List[List[int]]
        List of ID sequences wheras an ID relates to

    scheme : List[str]
        4-class NER scheme, CoNLL-03, ['PER', 'LOC', 'ORG', 'MISC']
        Implizit ID:NERtags mappings

    Example:
    --------
        nertags, SCHEME = nt.ner.flair_multi(tokens)
    """
    # (1) load flair model
    if not model:
        model = flair.models.SequenceTagger.load('ner-multi')

    # NER recognize a pre-tokenized sentencens
    nertags = []
    for sequence in data:
        seq = flair.data.Sentence(sequence)
        model.predict(seq)
        tags = [t.get_tag("ner").value.split("-") for t in seq.tokens]
        tags = [tag[1] if len(tag) == 2 else "[UNK]" for tag in tags]
        nertags.append(tags)

    # (2) Define the CoNLL-03 NER tagset as VOCAB
    SCHEME = CONLL03_SCHEME.copy()
    SCHEME.append("[UNK]")

    # (3) convert CoNLL-03 NER tags to a sequence of IDs
    nertags_ids = [texttoken_to_index(seq, SCHEME) for seq in nertags]

    # done
    return nertags_ids, SCHEME


@pad_idseqs
def stanza_de(data: List[List[str]], model=None) -> (
        List[List[str]], List[str]):
    """NER tagging with stanza NER tagger for German

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.ner.get_model

    maxlen : Optional[int] = None
        see @nlptasks.padding.pad_idseqs

    padding : Optional[str] = 'pre'
        see @nlptasks.padding.pad_idseqs

    truncating : Optional[str] = 'pre'
        see @nlptasks.padding.pad_idseqs

    Returns:
    --------
    sequences : List[List[int]]
        List of ID sequences wheras an ID relates to

    scheme : List[str]
        4-class NER scheme, CoNLL-03, ['PER', 'LOC', 'ORG', 'MISC']
        Implizit ID:NERtags mappings

    Example:
    --------
        nertags, SCHEME = nt.ner.stanza_de(tokens)
    """
    # (1) load stanza model
    if not model:
        model = stanza.Pipeline(
            lang='de', processors='tokenize,ner',
            tokenize_pretokenized=True)

    # NER recognize a pre-tokenized sentencens
    docs = model(data)
    nertags = [[t.ner.split("-") for t in sent.tokens]
               for sent in docs.sentences]
    nertags = [[t[1] if len(t) == 2 else "[UNK]" for t in s] for s in nertags]

    # (2) Define the WIKINER tagset as VOCAB
    SCHEME = CONLL03_SCHEME.copy()
    SCHEME.append("[UNK]")

    # (3) convert WIKI NER tags to a sequence of IDs
    nertags_ids = [texttoken_to_index(seq, SCHEME) for seq in nertags]

    # done
    return nertags_ids, SCHEME
