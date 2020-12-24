import nlptasks as nt
import nlptasks.deprel


def test_01():
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]
    deps_child, deps_parent, seqlens = nt.deprel.factory("spacy-de")(seqs_token)
    assert deps_child == [[
        (0, 2), (1, 2), (2, 3), (4, 3), (5, 3), (7, 3), (6, 5)]]
    assert deps_parent == [[
        (2, 0), (2, 1), (3, 2), (3, 3), (3, 4), (3, 5), (5, 6), (3, 7)]]
    assert seqlens == [8]


def test_02():  # check if calling pad_adjseqs causes an error
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]
    deps_child, deps_parent, seqlens = nt.deprel.factory("spacy-de")(
        seqs_token, maxlen=9, padding='post', truncating='post')
    assert deps_child == [[
        (0, 2), (1, 2), (2, 3), (4, 3), (5, 3), (7, 3), (6, 5)]]
    assert deps_parent == [[
        (2, 0), (2, 1), (3, 2), (3, 3), (3, 4), (3, 5), (5, 6), (3, 7)]]
    assert seqlens == [8]


def test_03():  # preload model
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]
    identifier = "spacy-de"
    model = nt.deprel.get_model(identifier)
    fn = nt.deprel.factory(identifier)
    deps_child, deps_parent, seqlens = fn(seqs_token, model=model)
    assert deps_child == [[
        (0, 2), (1, 2), (2, 3), (4, 3), (5, 3), (7, 3), (6, 5)]]
    assert deps_parent == [[
        (2, 0), (2, 1), (3, 2), (3, 3), (3, 4), (3, 5), (5, 6), (3, 7)]]
    assert seqlens == [8]
