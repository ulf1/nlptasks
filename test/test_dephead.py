import nlptasks as nt
import nlptasks.dephead


def test_01():
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]

    maskseqs, seqlens, SCHEME = nt.dephead.factory("spacy-de")(seqs_token)

    assert maskseqs == [[
        (2, 0), (2, 1), (3, 2), (3, 3), (3, 4), (3, 5), (5, 6), (3, 7)]]
    assert seqlens == [8]
    assert SCHEME == nt.dephead.UD2_REL


def test_02():  # check if calling pad_dephead causes an error
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]

    maskseqs, seqlens, SCHEME = nt.dephead.factory("spacy-de")(
        seqs_token, maxlen=9, padding='post', truncating='post')

    assert maskseqs == [[
        (2, 0), (2, 1), (3, 2), (3, 3), (3, 4), (3, 5), (5, 6), (3, 7)]]
    assert seqlens == [8]
    assert SCHEME == nt.dephead.UD2_REL


def test_03():  # preload model
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]

    identifier = "spacy-de"
    model = nt.dephead.get_model(identifier)
    fn = nt.dephead.factory(identifier)
    maskseqs, seqlens, SCHEME = fn(seqs_token, model=model)

    assert maskseqs == [[
        (2, 0), (2, 1), (3, 2), (3, 3), (3, 4), (3, 5), (5, 6), (3, 7)]]
    assert seqlens == [8]
    assert SCHEME == nt.dephead.UD2_REL
