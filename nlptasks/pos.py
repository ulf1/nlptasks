from .padding import pad_idseqs
from typing import List
from .vocab import texttoken_to_index
import de_core_news_lg as spacy_model
import spacy
import stanza
import flair
from .utils import FlairSentence
import someweta
from pathlib import Path


TIGER_TAGSET = [
    '$(', '$,', '$.', 'ADJA', 'ADJD', 'ADV', 'APPO', 'APPR', 'APPRART',
    'APZR', 'ART', 'CARD', 'FM', 'ITJ', 'KOKOM', 'KON', 'KOUI', 'KOUS',
    'NE', 'NN', 'NNE', 'PDAT', 'PDS', 'PIAT', 'PIS', 'PPER', 'PPOSAT',
    'PPOSS', 'PRELAT', 'PRELS', 'PRF', 'PROAV', 'PTKA', 'PTKANT',
    'PTKNEG', 'PTKVZ', 'PTKZU', 'PWAT', 'PWAV', 'PWS', 'TRUNC', 'VAFIN',
    'VAIMP', 'VAINF', 'VAPP', 'VMFIN', 'VMINF', 'VMPP', 'VVFIN', 'VVIMP',
    'VVINF', 'VVIZU', 'VVPP', 'XY', '_SP']


STTS_IBK = TIGER_TAGSET + [
    'ONO', 'DM', 'PTKIFG', 'PTKMA', 'PTKMWL', 'VVPPER', 'VMPPER',
    'VAPPER', 'KOUSPPER', 'PPERPPER', 'ADVART', 'EMOASC', 'EMOIMG',
    'AKW', 'HST', 'ADR', 'URL', 'EML'
]


def pos_factory(name: str):
    if name in ("spacy", "spacy-de"):
        return pos_spacy_de
    elif name in ("stanza", "stanza-de"):
        return pos_stanza_de
    elif name == "flair-de":
        return pos_flair_de
    elif name in ("someweta", "someweta-de"):
        return pos_someweta_de
    elif name in ("someweta-web", "someweta-web-de"):
        return pos_someweta_web_de
    else:
        raise Exception(f"Unknown PoS tagger: '{name}'") 


def get_model(name: str):
    """Instantiate the pretrained model outside the SBD function
        so that it only needs to be done once

    Parameters:
    -----------
    name : str
        Identfier of the model

    Example:
    --------
        from nlptasks.ner import pos
        model = pos.get_model('stanza-de')
        fn = pos.factory('stanza-de')
        idseqs, TAGSET = fn(docs, model=model)
    """
    if name in ("spacy", "spacy-de"):
        model = spacy_model.load()
        model.disable_pipes(["ner", "parser"])
        return model

    elif name in ("stanza", "stanza-de"):
        return stanza.Pipeline(
            lang='de', processors='tokenize,pos',
            tokenize_pretokenized=True)

    elif name == "flair-de":
        return flair.models.SequenceTagger.load('de-pos')

    elif name in ("someweta", "someweta-de"):
        model = someweta.ASPTagger()
        model.load(f"{str(Path.home())}/someweta_data/german_newspaper.model")
        return model

    elif name in ("someweta-web", "someweta-web-de"):
        model = someweta.ASPTagger()
        model.load(
            f"{str(Path.home())}/someweta_data/german_web_social_media.model")
        return model

    else:
        raise Exception(f"Unknown PoS tagger: '{name}'") 


@pad_idseqs
def pos_spacy_de(data: List[List[str]], model=None) -> (
        List[List[str]], List[str]):
    """PoS-Tagging with spaCy de_core_news_lg for German

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.pos.get_model

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
    if not model:
        model = spacy_model.load()
        model.disable_pipes(["ner", "parser"])

    # pos-tag a pre-tokenized sentencens
    tagger = model.pipeline[0][1]
    docs = [tagger(spacy.tokens.doc.Doc(model.vocab, words=sequence))
            for sequence in data]
    postags = [[t.tag_ for t in doc] for doc in docs]

    # (2) Define the TIGER tagset as VOCAB
    TAGSET = TIGER_TAGSET.copy()
    TAGSET.append("[UNK]")
    
    # (3) convert lemmata into IDs
    postags_ids = [texttoken_to_index(seq, TAGSET) for seq in postags]

    # done
    return postags_ids, TAGSET


@pad_idseqs
def pos_stanza_de(data: List[List[str]], model=None) -> (
        List[List[str]], List[str]):
    """PoS-Tagging with stanza PoS tagger for German

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.pos.get_model

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
    if not model:
        model = stanza.Pipeline(
            lang='de', processors='tokenize,pos',
            tokenize_pretokenized=True)

    # pos-tag a pre-tokenized sentencens
    docs = model(data)
    postags = [[t.xpos for t in sent.words] for sent in docs.sentences]

    # (2) Define the TIGER tagset as VOCAB
    TAGSET = TIGER_TAGSET.copy()
    TAGSET.append("[UNK]")
    
    # (3) convert lemmata into IDs
    postags_ids = [texttoken_to_index(seq, TAGSET) for seq in postags]

    # done
    return postags_ids, TAGSET


@pad_idseqs
def pos_flair_de(data: List[List[str]], model=None) -> (
        List[List[str]], List[str]):
    """PoS-Tagging with flair for German

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.pos.get_model

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
        postags, TAGSET = pos_flair_de(tokens)
    """
    # (1) load flair model
    if not model:
        model = flair.models.SequenceTagger.load('de-pos')

    # PoS-tag recognize a pre-tokenized sentencens
    postags = []
    for sequence in data:
        seq = FlairSentence(sequence)
        model.predict(seq)
        tags = [t.get_tag("pos").value for t in seq.tokens]
        postags.append(tags)

    # (2) Define the TIGER tagset as VOCAB
    TAGSET = TIGER_TAGSET.copy()
    TAGSET.append("[UNK]")
    
    # (3) convert lemmata into IDs
    postags_ids = [texttoken_to_index(seq, TAGSET) for seq in postags]

    # done
    return postags_ids, TAGSET


@pad_idseqs
def pos_someweta_de(data: List[List[str]], model=None) -> (
        List[List[str]], List[str]):
    """
    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.pos.get_model
    """
    # (1) load model
    if not model:
        model = someweta.ASPTagger()
        model.load(f"{str(Path.home())}/someweta_data/german_newspaper.model")

    # PoS-tag recognize a pre-tokenized sentencens
    postags = []
    for sequence in data:
        seq = model.tag_sentence(sequence)
        tags = [tag for _, tag in seq]
        postags.append(tags)

    # (2) Define the TIGER tagset as VOCAB
    TAGSET = TIGER_TAGSET.copy()
    TAGSET.append("[UNK]")
    
    # (3) convert lemmata into IDs
    postags_ids = [texttoken_to_index(seq, TAGSET) for seq in postags]

    # done
    return postags_ids, TAGSET


@pad_idseqs
def pos_someweta_web_de(data: List[List[str]], model=None) -> (
        List[List[str]], List[str]):
    """
    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.pos.get_model
    """
    # (1) load model
    if not model:
        model = someweta.ASPTagger()
        model.load(
            f"{str(Path.home())}/someweta_data/german_web_social_media.model")

    # PoS-tag recognize a pre-tokenized sentencens
    postags = []
    for sequence in data:
        seq = model.tag_sentence(sequence)
        tags = [tag for _, tag in seq]
        postags.append(tags)

    # (2) Define the TIGER tagset as VOCAB
    TAGSET = STTS_IBK.copy()
    TAGSET.append("[UNK]")
    
    # (3) convert lemmata into IDs
    postags_ids = [texttoken_to_index(seq, TAGSET) for seq in postags]

    # done
    return postags_ids, TAGSET
