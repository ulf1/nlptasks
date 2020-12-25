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
    for pair in maskseqs[0]:
        assert pair in target


def test_02():  # check if calling pad_dephead causes an error
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]
    target = [
        (46, 0), (46, 1), (47, 2), (47, 3), (47, 4), (47, 5), (49, 6), (47, 7),
        (19, 0), (31, 1), (36, 2), (42, 3), (21, 4), (17, 5), (19, 6), (32, 7)]

    maskseqs, seqlens = nt.dephead.factory("spacy-de")(
        seqs_token, maxlen=6, padding='post', truncating='post')

    assert seqlens == [8]
    for pair in maskseqs[0]:
        assert pair in target
    for pair in maskseqs[0]:
        assert pair[1] < 6


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
    for pair in maskseqs[0]:
        assert pair in target


def test_11():
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]
    target = [
        (64, 0), (66, 1), (64, 2), (62, 3), (66, 4), (69, 5), (67, 6), (66, 7),
        (24, 0), (45, 1), (35, 2), (58, 3), (49, 4), (9, 5), (42, 6), (56, 7)]
    
    maskseqs, seqlens = nt.dephead.factory("stanza-de")(seqs_token)

    assert seqlens == [8]
    for pair in maskseqs[0]:
        assert pair in target


def test_12():  # check if calling pad_dephead causes an error
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]
    target = [
        (64, 0), (66, 1), (64, 2), (62, 3), (66, 4), (69, 5), (67, 6), (66, 7),
        (24, 0), (45, 1), (35, 2), (58, 3), (49, 4), (9, 5), (42, 6), (56, 7)]

    maskseqs, seqlens = nt.dephead.factory("stanza-de")(
        seqs_token, maxlen=6, padding='post', truncating='post')

    assert seqlens == [8]
    for pair in maskseqs[0]:
        assert pair in target
    for pair in maskseqs[0]:
        assert pair[1] < 6


def test_13():  # preload model
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]
    target = [
        (64, 0), (66, 1), (64, 2), (62, 3), (66, 4), (69, 5), (67, 6), (66, 7),
        (24, 0), (45, 1), (35, 2), (58, 3), (49, 4), (9, 5), (42, 6), (56, 7)]

    identifier = "stanza-de"
    model = nt.dephead.get_model(identifier)
    fn = nt.dephead.factory(identifier)
    maskseqs, seqlens = fn(seqs_token, model=model)

    assert seqlens == [8]
    for pair in maskseqs[0]:
        assert pair in target
