import nlptasks as nt
import nlptasks.deptree


def test1():
    sequences = [['Die', 'Kuh', 'ist', 'bunt', '.']]
    onehotindices, VOCAB = nt.deptree.spacy_de(sequences)
    assert onehotindices == [[0, 1, 2, 3, 4]]
