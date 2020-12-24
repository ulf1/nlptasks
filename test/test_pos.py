import nlptasks as nt
import nlptasks.pos


def test_01():
    targets = [[
        "APPR", "ART", "NN", "ART", "NN", "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = nt.pos.factory("spacy")(seqs_token)
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_02():  # check pad_idseqs
    targets = [[
        "[PAD]", "APPR", "ART", "NN", "ART", "NN",
        "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = nt.pos.factory("spacy")(
        seqs_token, maxlen=11, padding='pre', truncating='pre')
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_03():
    targets = [[
        "APPR", "ART", "NN", "ART", "NN", "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]

    identifier = "spacy-de"
    model = nt.pos.get_model(identifier)
    fn = nt.pos.factory(identifier)
    seqs_pos, TAGSET = fn(seqs_token, model=model)

    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_11():
    targets = [[
        "APPR", "ART", "NN", "ART", "NN", "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = nt.pos.factory("stanza-de")(seqs_token)
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_12():  # check pad_idseqs
    targets = [[
        "[PAD]", "APPR", "ART", "NN", "ART", "NN",
        "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = nt.pos.factory("stanza-de")(
        seqs_token, maxlen=11, padding='pre', truncating='pre')
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_13():
    targets = [[
        "APPR", "ART", "NN", "ART", "NN", "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]

    identifier = "stanza-de"
    model = nt.pos.get_model(identifier)
    fn = nt.pos.factory(identifier)
    seqs_pos, TAGSET = fn(seqs_token, model=model)

    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_21():
    targets = [[
        "APPR", "ART", "NN", "ART", "NN", "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = nt.pos.factory("flair-de")(seqs_token)
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_22():  # check pad_idseqs
    targets = [[
        "[PAD]", "APPR", "ART", "NN", "ART", "NN",
        "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = nt.pos.factory("flair-de")(
        seqs_token, maxlen=11, padding='pre', truncating='pre')
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_23():
    targets = [[
        "APPR", "ART", "NN", "ART", "NN", "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]

    identifier = "flair-de"
    model = nt.pos.get_model(identifier)
    fn = nt.pos.factory(identifier)
    seqs_pos, TAGSET = fn(seqs_token, model=model)

    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids
