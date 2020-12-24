from .padding import pad_idseqs
from typing import List, Optional
from .vocab import identify_vocab_mincount, texttoken_to_index
import itertools
import warnings
import spacy
import de_core_news_lg as spacy_model
import stanza


def factory(name: str):
    if name in ("spacy", "spacy-de"):
        return lemma_spacy_de
    elif name in ("stanza", "stanza-de"):
        return lemma_stanza_de
    else:
        raise Exception(f"Unknown lemmatizer: '{name}'") 


def lemma_factory(name: str):
    warnings.warn(
        "Please call `nlptasks.lemma.factory` instead",
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
        from nlptasks.lemma import lemma
        model = lemma.get_model('stanza-de')
        fn = lemma.factory('stanza-de')
        seq, VOCAB = fn(docs, model=model)
    """
    if name in ("spacy", "spacy-de"):
        model = spacy_model.load()
        model.disable_pipes(["ner", "parser", "tagger"])
        return model

    elif name in ("stanza", "stanza-de"):
        return stanza.Pipeline(
            lang='de', processors='tokenize,lemma',
            tokenize_pretokenized=True)

    else:
        raise Exception(f"Unknown lemmatizer: '{name}'") 


@pad_idseqs
def lemma_spacy_de(data: List[List[str]],
                   VOCAB: Optional[List[str]] = None,
                   min_occurrences: Optional[int] = 20, 
                   model=None
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

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.lemma.get_model

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
    if not model:
        model = spacy_model.load()
        model.disable_pipes(["ner", "parser", "tagger"])

    # lemmatize a pre-tokenized sentencens
    docs = [spacy.tokens.doc.Doc(model.vocab, words=sequence) 
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
                    min_occurrences: Optional[int] = 20, 
                    model=None
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

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.lemma.get_model

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
    if not model:
        model = stanza.Pipeline(
            lang='de', processors='tokenize,lemma',
            tokenize_pretokenized=True)

    # lemmatize a pre-tokenized sentencens
    docs = model(data)
    lemmata = [[t.lemma.split("|")[0] for t in sent.words]
               for sent in docs.sentences]

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
