import tensorflow.keras as keras  # pad_sequences
from pad_sequences import pad_sequences_adjacency


def pad_idseqs(func):
    def wrapper(*args, **kwargs):
        # read and remove padding settings
        maxlen = kwargs.pop('maxlen', None)
        padding = kwargs.pop('padding', 'pre')
        truncating = kwargs.pop('truncating', 'pre')

        # run the NLP task
        idseqs, VOCAB = func(*args, **kwargs)

        # padding and update vocabulary
        if maxlen is not None:
            idseqs = keras.preprocessing.sequence.pad_sequences(
                idseqs, maxlen=maxlen, value=len(VOCAB),
                padding=padding, truncating=truncating).tolist()
            VOCAB.append("[PAD]")

        return idseqs, VOCAB
    return wrapper


def pad_adjseqs(func):
    def wrapper(*args, **kwargs):
        # read and remove padding settings
        maxlen = kwargs.pop('maxlen', None)
        padding = kwargs.pop('padding', 'pre')
        truncating = kwargs.pop('truncating', 'pre')

        # run the NLP task
        adjac_child, adjac_parent, seqs_lens = func(*args, **kwargs)

        # pad adjacency matrix of children and parent relationships
        if maxlen is not None:
            adjac_child = pad_sequences_adjacency(
                sequences=adjac_child, seqlen=seqs_lens,
                maxlen=maxlen, padding=padding, truncating=truncating)
            adjac_parent = pad_sequences_adjacency(
                sequences=adjac_parent, seqlen=seqs_lens,
                maxlen=maxlen, padding=padding, truncating=truncating)

        return adjac_child, adjac_parent, seqs_lens
    return wrapper
