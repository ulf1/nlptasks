import nlptasks as nt
import nlptasks.dephead


def test_01():
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]
    target = [
        (46, 0), (46, 1), (47, 2), (47, 3), (47, 4), (47, 5), (49, 6), (47, 7),
        (19, 0), (31, 1), (36, 2), (42, 3), (21, 4), (17, 5), (19, 6), (32, 7)]
    
    maskseqs, seqlens = nt.dephead.factory("spacy-de")(seqs_token)

    assert seqlens == [8]
    for pair in target:
        assert pair in maskseqs[0]


def test_02():  # check if calling pad_dephead causes an error
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]
    target = [
        (46, 0), (46, 1), (47, 2), (47, 3), (47, 4), (47, 5), (49, 6), (47, 7),
        (19, 0), (31, 1), (36, 2), (42, 3), (21, 4), (17, 5), (19, 6), (32, 7)]

    maskseqs, seqlens = nt.dephead.factory("spacy-de")(
        seqs_token, maxlen=9, padding='post', truncating='post')

    assert seqlens == [8]
    for pair in target:
        assert pair in maskseqs[0]


def test_03():  # preload model
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]
    target = [
        (46, 0), (46, 1), (47, 2), (47, 3), (47, 4), (47, 5), (49, 6), (47, 7),
        (19, 0), (31, 1), (36, 2), (42, 3), (21, 4), (17, 5), (19, 6), (32, 7)]

    identifier = "spacy-de"
    model = nt.dephead.get_model(identifier)
    fn = nt.dephead.factory(identifier)
    maskseqs, seqlens = fn(seqs_token, model=model)

    assert seqlens == [8]
    for pair in target:
        assert pair in maskseqs[0]
