from .padding import pad_maskseqs
from typing import List, Tuple
import flair
from .utils import FlairSentence


def ner2_factory(name: str):
    if name == "flair-multi":
        return ner_flair_multi
    else:
        raise Exception(f"Unknown dependency parser: '{name}'") 


@pad_maskseqs
def ner_flair_multi(data: List[List[str]]) -> (
        List[List[Tuple[int, int]]], List[int], List[str]):
    """flair 'multi-ner', returns sparse mask sequences of the 
        CoNLL-03 NE scheme (4 tags) and BIONES chunks

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
        maskseq, seqlen, SCHEME = ner_flair_multi2(tokens)
    """
    # (1) load flair model
    tagger = flair.models.SequenceTagger.load('ner-multi')

    # (2) Define the CoNLL-03 NER tagset as VOCAB
    SCHEME = ['PER', 'LOC', 'ORG', 'MISC', 
              'B', 'I', 'O', 'E', 'S']

    # (3) NER recognize a pre-tokenized sentencens
    maskseqs = []
    seqlen = []
    for sequence in data:
        seq = FlairSentence(sequence)
        tagger.predict(seq)
        pairs = []
        for i, t in enumerate(seq.tokens):
            for key in t.get_tag("ner").value.split("-"):
                pairs.append((SCHEME.index(key), i))        
        maskseqs.append(pairs)
        seqlen.append(len(sequence))
    
    # done
    return maskseqs, seqlen, SCHEME
