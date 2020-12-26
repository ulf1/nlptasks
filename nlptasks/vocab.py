from typing import List, Optional
from collections import Counter


def identify_vocab_mincount(data: List[str], 
                            min_occurrences: Optional[int] = 20,
                            sort: bool = True
                           ) -> List[str]:
    """Extract a vocabulary list where each token/word/lemma has minimum
        number of occurrences.
    
    Parameters:
    -----------
    data : List[str]
        All token/words/lemma of a corpus.

    min_occurrences : int
        The required number of occurences in a corpus.

    sort : bool
        flag to disable string based sorting

    Returns:
    --------
    VOCAB : List[str]
        Vocabulary list, alphabetically sorted

    Example:
    --------
        VOCAB = identify_vocab_mincount(
            data=all_words_in_a_corpus, min_occurrences=30)
    """
    cnt = Counter(data)
    VOCAB = [k for k, v in cnt.items() if v >= min_occurrences]
    if sort:
        VOCAB = sorted(VOCAB)
    return VOCAB


def texttoken_to_index(sequence: List[str], VOCAB: List[str]) -> List[int]:
    """Convert a sequence of strings to a sequence of IDs (int) based
        on a given vocabulary
    
    Parameters:
    -----------
    sequence : List[str]
        List of ID sequences

    VOCAB : List[str]
        Vocabulary list, alphabetically sorted

    Returns:
    --------
    List[int]
        List of ID sequences

    Example:
    --------
        seqs_of_ids = [texttoken_to_index(seq, VOCAB) for seq in sequences]
    """
    # find ID for [UNK], i.e. unknown
    try:
        UNKIDX = VOCAB.index("[UNK]")
    except:
        UNKIDX = len(VOCAB)
    # loop over each token
    indicies = []
    for token in sequence:
        try:
            indicies.append(VOCAB.index(token))
        except:
            indicies.append(UNKIDX)
    # done
    return indicies

