from .padding import pad_maskseqs
from typing import List, Tuple
import flair
from .utils import FlairSentence


def ner2_factory(name: str):
    if name == "flair-multi":
        return ner2_flair_multi
    else:
        raise Exception(f"Unknown NER tagger: '{name}'") 


def get_model(name: str):
    """Instantiate the pretrained model outside the SBD function
        so that it only needs to be done once

    Parameters:
    -----------
    name : str
        Identfier of the model

    Example:
    --------
        from nlptasks.ner import ner2
        model = ner2.get_model('stanza-de')
        fn = ner2.factory('stanza-de')
        maskseqs, seqlens, SCHEME = fn(docs, model=model)
    """
    if name == "flair-multi":
        return flair.models.SequenceTagger.load('ner-multi')
    else:
        raise Exception(f"Unknown NER tagger: '{name}'") 


@pad_maskseqs
def ner2_flair_multi(data: List[List[str]], model=None) -> (
        List[List[Tuple[int, int]]], List[int], List[str]):
    """flair 'multi-ner', returns sparse mask sequences of the 
        CoNLL-03 NE scheme (4 tags) and BIONES chunks

    Parameters:
    -----------
    data : List[List[str]]
        List of token sequences

    model (Default: None)
        Preloaded instance of the NLP model. See nlptasks.ner2.get_model

    maxlen : Optional[int] = None
        see @nlptasks.padding.pad_maskseqs

    padding : Optional[str] = 'pre'
        see @nlptasks.padding.pad_maskseqs

    truncating : Optional[str] = 'pre'
        see @nlptasks.padding.pad_maskseqs

    Returns:
    --------
    sequences : List[List[Tuple[int, int]]]
        List of sequences that are sparse mask matrices. The rows indicate
          the scheme.

    seqlens : List[int]
        Length of each sequence
    
    scheme : List[str]
        4-class NER scheme (CoNLL-03) and BIONES chunks, 
        ['PER', 'LOC', 'ORG', 'MISC', 'B', 'I', 'O', 'E', 'S']
    
    Example:
    --------
        maskseq, seqlen, SCHEME = ner2_flair_multi(tokens)
    """
    # (1) load flair model
    if not model:
        model = flair.models.SequenceTagger.load('ner-multi')

    # (2) Define the CoNLL-03 NER tagset as VOCAB
    SCHEME = ['PER', 'LOC', 'ORG', 'MISC', 
              'B', 'I', 'O', 'E', 'S']

    # (3) NER recognize a pre-tokenized sentencens
    maskseqs = []
    seqlen = []
    for sequence in data:
        seq = FlairSentence(sequence)
        model.predict(seq)
        pairs = []
        for i, t in enumerate(seq.tokens):
            for key in t.get_tag("ner").value.split("-"):
                pairs.append((SCHEME.index(key), i))        
        maskseqs.append(pairs)
        seqlen.append(len(sequence))
    
    # done
    return maskseqs, seqlen, SCHEME
