from .padding import pad_idseqs
from typing import List
from .vocab import texttoken_to_index
import de_core_news_lg as spacy_model
import spacy
import stanza


def pos_factory(name: str):
    if name in ("spacy", "spacy-de"):
        return pos_spacy_de
    elif name in ("stanza", "stanza-de"):
        return pos_stanza_de
    else:
        raise Exception(f"Unknown PoS tagger: '{name}'") 


@pad_idseqs
def pos_spacy_de(data: List[List[str]]) -> (List[List[str]], List[str]):
    """PoS-Tagging with spaCy de_core_news_lg for German

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

    tagset : List[str]
        PoS tagset is TIGER.
        Implizit ID:PoS mappings (It's the VOCAB for Embeddings)

    Example:
    --------
        postags, TAGSET = pos_spacy_de(tokens)
    """
    # (1) load spacy model
    nlp = spacy_model.load()
    nlp.disable_pipes(["ner", "parser"])
    tagger = nlp.pipeline[0][1]

    # pos-tag a pre-tokenized sentencens
    docs = [tagger(spacy.tokens.doc.Doc(nlp.vocab, words=sequence))
            for sequence in data]
    postags = [[t.tag_ for t in doc] for doc in docs]

    # (2) Define the TIGER tagset as VOCAB
    TAGSET = [
        '$(', '$,', '$.', 'ADJA', 'ADJD', 'ADV', 'APPO', 'APPR', 'APPRART',
        'APZR', 'ART', 'CARD', 'FM', 'ITJ', 'KOKOM', 'KON', 'KOUI', 'KOUS',
        'NE', 'NN', 'NNE', 'PDAT', 'PDS', 'PIAT', 'PIS', 'PPER', 'PPOSAT',
        'PPOSS', 'PRELAT', 'PRELS', 'PRF', 'PROAV', 'PTKA', 'PTKANT',
        'PTKNEG', 'PTKVZ', 'PTKZU', 'PWAT', 'PWAV', 'PWS', 'TRUNC', 'VAFIN',
        'VAIMP', 'VAINF', 'VAPP', 'VMFIN', 'VMINF', 'VMPP', 'VVFIN', 'VVIMP',
        'VVINF', 'VVIZU', 'VVPP', 'XY', '_SP']
    TAGSET.append("[UNK]")
    
    # (3) convert lemmata into IDs
    postags_ids = [texttoken_to_index(seq, TAGSET) for seq in postags]

    # done
    return postags_ids, TAGSET


@pad_idseqs
def pos_stanza_de(data: List[List[str]]) -> (List[List[str]], List[str]):
    """PoS-Tagging with stanza PoS tagger for German

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

    tagset : List[str]
        PoS tagset is TIGER.
        Implizit ID:PoS mappings (It's the VOCAB for Embeddings)

    Example:
    --------
        postags, TAGSET = pos_stanza_de(tokens)
    """
    # (1) load stanza model
    nlp = stanza.Pipeline(lang='de', processors='tokenize,pos',
                          tokenize_pretokenized=True)

    # pos-tag a pre-tokenized sentencens
    docs = nlp(data)
    postags = [[t.xpos for t in sent.words] for sent in docs.sentences]

    # (2) Define the TIGER tagset as VOCAB
    TAGSET = [
        '$(', '$,', '$.', 'ADJA', 'ADJD', 'ADV', 'APPO', 'APPR', 'APPRART',
        'APZR', 'ART', 'CARD', 'FM', 'ITJ', 'KOKOM', 'KON', 'KOUI', 'KOUS',
        'NE', 'NN', 'NNE', 'PDAT', 'PDS', 'PIAT', 'PIS', 'PPER', 'PPOSAT',
        'PPOSS', 'PRELAT', 'PRELS', 'PRF', 'PROAV', 'PTKA', 'PTKANT',
        'PTKNEG', 'PTKVZ', 'PTKZU', 'PWAT', 'PWAV', 'PWS', 'TRUNC', 'VAFIN',
        'VAIMP', 'VAINF', 'VAPP', 'VMFIN', 'VMINF', 'VMPP', 'VVFIN', 'VVIMP',
        'VVINF', 'VVIZU', 'VVPP', 'XY', '_SP']
    TAGSET.append("[UNK]")
    
    # (3) convert lemmata into IDs
    postags_ids = [texttoken_to_index(seq, TAGSET) for seq in postags]

    # done
    return postags_ids, TAGSET
