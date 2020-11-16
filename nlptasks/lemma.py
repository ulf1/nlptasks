from .padding import pad_idseqs
from typing import List, Optional
from .vocab import identify_vocab_mincount, texttoken_to_index
import itertools
import spacy
import de_core_news_lg as spacy_model
import stanza


def lemma_factory(name: str):
    if name in ("spacy", "spacy-de"):
        return lemma_spacy_de
    elif name in ("stanza", "stanza-de"):
        return lemma_stanza_de
    else:
        raise Exception(f"Unknown lemmatizer: '{name}'") 

@pad_idseqs
def lemma_spacy_de(data: List[List[str]],
                   VOCAB: Optional[List[str]] = None,
                   min_occurrences: Optional[int] = 20
                  ) -> (List[List[str]], List[str]):
    """Lemmatization with spaCy de_core_news_lg for German

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences

    vocab : List[str]
        (Optional) A given list of lemmata wheras list indicies are used as ID

    min_occurrences : int
        (Optional) The required number of occurences in a corpus.

    Returns:
    --------
    sequences : List[List[int]]
        List of ID sequences wheras an ID relates to a lemma

    vocab : List[str]
        List of lemmata. Implizit ID:Lemma mappings

    Example:
    --------
        lemmata, VOCAB = lemma_spacy_de(tokens)
    """
    # (1) load spacy model
    nlp = spacy_model.load()
    nlp.disable_pipes(["ner", "parser", "tagger"])

    # lemmatize a pre-tokenized sentencens
    docs = [spacy.tokens.doc.Doc(nlp.vocab, words=sequence) 
            for sequence in data]
    lemmata = [[t.lemma_ for t in doc] for doc in docs]

    # (2) Identify VOCAB
    if VOCAB is None:
        VOCAB = identify_vocab_mincount(
            data=list(itertools.chain.from_iterable(lemmata)),
            min_occurrences=min_occurrences)
        VOCAB.append("[UNK]")
    
    # (3) convert lemmata into IDs
    lemmata_idx = [texttoken_to_index(seq, VOCAB) for seq in lemmata]

    # done
    return lemmata_idx, VOCAB


@pad_idseqs
def lemma_stanza_de(data: List[List[str]],
                    VOCAB: Optional[List[str]] = None,
                    n_min_occurence: Optional[int] = 20
                   ) -> (List[List[str]], List[str]):
    """Lemmatization with stanza for German

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences

    VOCAB : List[str]
        (Optional) A given list of lemmata wheras list indicies are used as ID

    n_min_occurence : int
        (Optional) The required number of occurences in a corpus.

    maxlen : Optional[int] = None
        see @nlptasks.padding.pad_idseqs

    padding : Optional[str] = 'pre'
        see @nlptasks.padding.pad_idseqs

    truncating : Optional[str] = 'pre'
        see @nlptasks.padding.pad_idseqs

    Returns:
    --------
    sequences : List[List[int]]
        List of ID sequences wheras an ID relates to a lemma

    VOCAB : List[str]
        List of lemmata. Implizit ID:Lemma mappings

    Example:
    --------
        lemmata, VOCAB = lemma_stanza_de(tokens)
    """
    # (1) load stanza model
    nlp = stanza.Pipeline(lang='de', processors='tokenize,lemma',
                          tokenize_pretokenized=True)

    # lemmatize a pre-tokenized sentencens
    docs = nlp(data)
    lemmata = [[t.lemma.split("|")[0] for t in sent.words]
               for sent in docs.sentences]

    # (2) Identify VOCAB
    if VOCAB is None:
        VOCAB = identify_vocab_mincount(
            data=list(itertools.chain.from_iterable(lemmata)),
            n_min_occurence=n_min_occurence)
        VOCAB.append("[UNK]")
    
    # (3) convert lemmata into IDs
    lemmata_idx = [texttoken_to_index(seq, VOCAB) for seq in lemmata]

    # done
    return lemmata_idx, VOCAB
