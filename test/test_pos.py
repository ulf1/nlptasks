from nlptasks.pos import pos_factory


def test_01():
    targets = [[
        "APPR", "ART", "NN", "ART", "NN", "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = pos_factory("spacy")(seqs_token)
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_02():  # check pad_idseqs
    targets = [[
        "[PAD]", "APPR", "ART", "NN", "ART", "NN",
        "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = pos_factory("spacy")(
        seqs_token, maxlen=11, padding='pre', truncating='pre')
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids
