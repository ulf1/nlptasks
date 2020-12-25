import nlptasks as nt
import nlptasks.depchild


def test_01():
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]

    maskseqs, seqlens = nt.depchild.factory("spacy-de")(seqs_token)

    assert maskseqs == [[
        (0, 2), (1, 2), (2, 3), (4, 3), (5, 3), (7, 3), (6, 5)]]
    assert seqlens == [8]


def test_02():  # check if calling pad_adjacmatrix causes an error
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]

    maskseqs, seqlens = nt.depchild.factory("spacy-de")(
        seqs_token, maxlen=9, padding='post', truncating='post')

    assert maskseqs == [[
        (0, 2), (1, 2), (2, 3), (4, 3), (5, 3), (7, 3), (6, 5)]]
    assert seqlens == [8]


def test_03():  # preload model
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]

    identifier = "spacy-de"
    model = nt.depchild.get_model(identifier)
    fn = nt.depchild.factory(identifier)
    maskseqs, seqlens = fn(seqs_token, model=model)

    assert maskseqs == [[
        (0, 2), (1, 2), (2, 3), (4, 3), (5, 3), (7, 3), (6, 5)]]
    assert seqlens == [8]


def test_11():
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]

    maskseqs, seqlens = nt.depchild.factory("stanza-de")(seqs_token)

    assert maskseqs == [[
        (0, 2), (1, 2), (2, 3), (4, 3), (5, 3), (7, 3), (6, 5)]]
    assert seqlens == [8]


def test_12():  # check if calling pad_adjacmatrix causes an error
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]

    maskseqs, seqlens = nt.depchild.factory("stanza-de")(
        seqs_token, maxlen=9, padding='post', truncating='post')

    assert maskseqs == [[
        (0, 2), (1, 2), (2, 3), (4, 3), (5, 3), (7, 3), (6, 5)]]
    assert seqlens == [8]


def test_13():  # preload model
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]

    identifier = "stanza-de"
    model = nt.depchild.get_model(identifier)
    fn = nt.depchild.factory(identifier)
    maskseqs, seqlens = fn(seqs_token, model=model)

    assert maskseqs == [[
        (0, 2), (1, 2), (2, 3), (4, 3), (5, 3), (7, 3), (6, 5)]]
    assert seqlens == [8]
