from nlptasks.pos2 import pos2_factory, UPOS_TAGSET, UD2_FEATS
import nlptasks as nt


def test_11():
    targets = [
        [(1, 0), (5, 1), (100, 1), (115, 1), (43, 1), (75, 1), (17, 1),
        (7, 2), (100, 2), (43, 2), (75, 2), (5, 3), (103, 3), (115, 3),
        (43, 3), (77, 3), (17, 3), (7, 4), (103, 4), (43, 4), (77, 4),
        (15, 5), (135, 5), (77, 5), (171, 5), (145, 5), (124, 5), (11, 6),
        (112, 6), (42, 6), (77, 6), (5, 7), (110, 7), (115, 7), (41, 7),
        (77, 7), (17, 7), (7, 8), (110, 8), (41, 8), (77, 8), (12, 9)]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    maskseqs, seqlen, SCHEME = pos2_factory("stanza-de")(seqs_token)
    assert maskseqs == targets
    assert seqlen == [10]
    assert SCHEME == UPOS_TAGSET + UD2_FEATS


def test_12():  # check pad_maskseqs
    targets = [
        [(1, 1), (5, 2), (100, 2), (115, 2), (43, 2), (75, 2), (17, 2),
        (7, 3), (100, 3), (43, 3), (75, 3), (5, 4), (103, 4), (115, 4),
        (43, 4), (77, 4), (17, 4), (7, 5), (103, 5), (43, 5), (77, 5),
        (15, 6), (135, 6), (77, 6), (171, 6), (145, 6), (124, 6), (11, 7),
        (112, 7), (42, 7), (77, 7), (5, 8), (110, 8), (115, 8), (41, 8),
        (77, 8), (17, 8), (7, 9), (110, 9), (41, 9), (77, 9), (12, 10)]]

    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]

    maskseqs, seqlen, SCHEME = pos2_factory("stanza-de")(
        seqs_token, maxlen=11, padding='pre', truncating='pre')

    assert maskseqs == targets
    assert seqlen == [10]
    assert SCHEME == UPOS_TAGSET + UD2_FEATS


def test_13():
    targets = [
        [(1, 0), (5, 1), (100, 1), (115, 1), (43, 1), (75, 1), (17, 1),
        (7, 2), (100, 2), (43, 2), (75, 2), (5, 3), (103, 3), (115, 3),
        (43, 3), (77, 3), (17, 3), (7, 4), (103, 4), (43, 4), (77, 4),
        (15, 5), (135, 5), (77, 5), (171, 5), (145, 5), (124, 5), (11, 6),
        (112, 6), (42, 6), (77, 6), (5, 7), (110, 7), (115, 7), (41, 7),
        (77, 7), (17, 7), (7, 8), (110, 8), (41, 8), (77, 8), (12, 9)]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]

    identifier = "stanza-de"
    model = nt.pos2.get_model(identifier)
    fn = nt.pos2.pos2_factory(identifier)
    maskseqs, seqlen, SCHEME = fn(seqs_token, model=model)

    assert maskseqs == targets
    assert seqlen == [10]
    assert SCHEME == UPOS_TAGSET + UD2_FEATS
