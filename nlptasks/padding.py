import tensorflow.keras as keras  # pad_sequences
from pad_sequences import pad_sequences_adjacency
from pad_sequences import pad_sequences_sparse


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
            if "[PAD]" not in VOCAB:
                VOCAB.append("[PAD]")
            idseqs = keras.preprocessing.sequence.pad_sequences(
                idseqs, maxlen=maxlen, value=VOCAB.index("[PAD]"),
                padding=padding, truncating=truncating).tolist()

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


def pad_maskseqs(func):
    def wrapper(*args, **kwargs):
        # read and remove padding settings
        maxlen = kwargs.pop('maxlen', None)
        padding = kwargs.pop('padding', 'pre')
        truncating = kwargs.pop('truncating', 'pre')

        # run the NLP task
        maskseqs, seqs_lens, VOCAB = func(*args, **kwargs)

        # pad sparse mask sequence
        if maxlen is not None:
            maskseqs = pad_sequences_sparse(
                sequences=maskseqs, seqlen=seqs_lens,
                maxlen=maxlen, padding=padding, truncating=truncating)

        return maskseqs, seqs_lens, VOCAB
    return wrapper
