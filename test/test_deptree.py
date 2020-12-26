import nlptasks as nt
import nlptasks.deptree


def test1():
    sequences = [['Die', 'Kuh', 'ist', 'bunt', '.']]
    indices, VOCAB = nt.deptree.spacy_de(sequences)
    assert indices == [[0, 1, 2, 3, 4]]
    indices, VOCAB = nt.deptree.stanza_de(sequences)
    assert indices == [[0, 1, 2, 3, 4]]


def test2():
    sequences = [['Die', 'Kuh', 'ist', 'bunt', '.']]
    masks, VOCAB = nt.deptree.spacy_de(sequences, return_mask=True)
    assert masks == [[1, 1, 1, 1, 1]]
    masks, VOCAB = nt.deptree.stanza_de(sequences, return_mask=True)
    assert masks == [[1, 1, 1, 1, 1]]
