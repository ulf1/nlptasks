from .padding import pad_idseqs
from typing import List
from .vocab import texttoken_to_index
import de_core_news_lg as spacy_model
import spacy
# import stanza


def ner_factory(name: str):
    if name == "spacy":
        return ner_spacy_de
    # elif name == "stanza":
    #     return ner_stanza_de
    # elif name == "flair":
    #     return ner_flair_de
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
