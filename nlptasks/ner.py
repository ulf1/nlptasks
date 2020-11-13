from .padding import pad_idseqs
from typing import List
from .vocab import texttoken_to_index
import de_core_news_lg as spacy_model
import spacy
# import stanza
import flair
from .utils import FlairSentence


def ner_factory(name: str):
    if name == "spacy":
        return ner_spacy_de
    # elif name == "stanza":
    #     return ner_stanza_de
    elif name == "flair-multi":
        return ner_flair_multi
    else:
        raise Exception(f"Unknown dependency parser: '{name}'") 


@pad_idseqs
def ner_spacy_de(data: List[List[str]]) -> (List[List[str]], List[str]):
    """NER with spaCy de_core_news_lg for German with Wikipedia NER Scheme

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences

    Returns:
    --------
    sequences : List[List[int]]
        List of ID sequences wheras an ID relates to 

    scheme : List[str]
        Wikipedia NER Scheme. Implizit ID:NERtags mappings
        (It's the VOCAB for Embeddings)

    Example:
    --------
        nertags, SCHEME = ner_spacy_de(tokens)
    """
    # (1) load spacy model
    nlp = spacy_model.load()
    nlp.disable_pipes(["parser", "tagger"])
    ner = nlp.pipeline[0][1]

    # NER recognize a pre-tokenized sentencens
    docs = [ner(spacy.tokens.doc.Doc(nlp.vocab, words=sequence))
            for sequence in data]
    nertags = [[t.ent_type_ for t in doc] for doc in docs]

    # (2) Define the WIKINER tagset as VOCAB
    SCHEME = ['PER', 'LOC', 'ORG', 'MISC']
    SCHEME.append("[UNK]")
    
    # (3) convert WIKI NER tags to a sequence of IDs
    nertags_ids = [texttoken_to_index(seq, SCHEME) for seq in nertags]

    # done
    return nertags_ids, SCHEME


@pad_idseqs
def ner_flair_multi(data: List[List[str]]) -> (List[List[str]], List[str]):
    """flair 'multi-ner', CoNLL-03 NE scheme, returns ID sequence
        for embeddings.

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences

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
        nertags, SCHEME = ner_flair_multi(tokens)
    """
    # (1) load flair model
    tagger = flair.models.SequenceTagger.load('ner-multi')

    # NER recognize a pre-tokenized sentencens
    nertags = []
    for sequence in data:
        seq = FlairSentence(sequence)
        tagger.predict(seq)
        tags = [t.get_tag("ner").value.split("-") for t in seq.tokens]
        tags = [tag[1] if len(tag)==2 else "[UNK]" for tag in tags]
        nertags.append(tags)

    # (2) Define the CoNLL-03 NER tagset as VOCAB
    SCHEME = ['PER', 'LOC', 'ORG', 'MISC']
    SCHEME.append("[UNK]")
    
    # (3) convert CoNLL-03 NER tags to a sequence of IDs
    nertags_ids = [texttoken_to_index(seq, SCHEME) for seq in nertags]

    # done
    return nertags_ids, SCHEME
