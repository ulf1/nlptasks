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


def pad_adjacmatrix(func):
    def wrapper(*args, **kwargs):
        # read and remove padding settings
        maxlen = kwargs.pop('maxlen', None)
        padding = kwargs.pop('padding', 'pre')
        truncating = kwargs.pop('truncating', 'pre')

        # run the NLP task
        adjac_matrix, seqs_lens = func(*args, **kwargs)

        # pad adjacency matrix of children relationships
        if maxlen is not None:
            adjac_matrix = pad_sequences_adjacency(
                sequences=adjac_matrix, seqlen=seqs_lens,
                maxlen=maxlen, padding=padding, truncating=truncating)

        return adjac_matrix, seqs_lens
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


def pad_merge_adjac_maskseqs(func):
    def wrapper(*args, **kwargs):
        # read and remove padding settings
        maxlen = kwargs.pop('maxlen', None)
        padding = kwargs.pop('padding', 'pre')
        truncating = kwargs.pop('truncating', 'pre')

        # run the NLP task
        adjac, onehot, seqs_lens, n_classes = func(*args, **kwargs)

        # pad adjacency matrix of children relationships
        if maxlen is not None:
            adjac = pad_sequences_adjacency(
                sequences=adjac, seqlen=seqs_lens,
                maxlen=maxlen, padding=padding, truncating=truncating)
            onehot = pad_sequences_sparse(
                sequences=onehot, seqlen=seqs_lens,
                maxlen=maxlen, padding=padding, truncating=truncating)
        # shift index of adjac matrix
        adjac = [[(i + n_classes, j) for i, j in sent] for sent in adjac]
        # merge both sparse matrices
        maskseqs = [adjac[k] + onehot[k] for k in range(len(adjac))]

        # done
        return maskseqs, seqs_lens
    return wrapper
