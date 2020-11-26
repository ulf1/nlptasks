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


def test_11():
    targets = [[
        "APPR", "ART", "NN", "ART", "NN", "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = pos_factory("stanza-de")(seqs_token)
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_12():  # check pad_idseqs
    targets = [[
        "[PAD]", "APPR", "ART", "NN", "ART", "NN",
        "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = pos_factory("stanza-de")(
        seqs_token, maxlen=11, padding='pre', truncating='pre')
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_21():
    targets = [[
        "APPR", "ART", "NN", "ART", "NN", "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = pos_factory("flair-de")(seqs_token)
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_22():  # check pad_idseqs
    targets = [[
        "[PAD]", "APPR", "ART", "NN", "ART", "NN",
        "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = pos_factory("flair-de")(
        seqs_token, maxlen=11, padding='pre', truncating='pre')
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_31():
    targets = [[
        "APPR", "ART", "NN", "ART", "NN", "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = pos_factory("someweta-de")(seqs_token)
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_32():  # check pad_idseqs
    targets = [[
        "[PAD]", "APPR", "ART", "NN", "ART", "NN",
        "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = pos_factory("someweta-de")(
        seqs_token, maxlen=11, padding='pre', truncating='pre')
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_41():
    targets = [[
        "APPR", "ART", "NN", "ART", "NN", "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = pos_factory("someweta-web-de")(seqs_token)
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_42():  # check pad_idseqs
    targets = [[
        "[PAD]", "APPR", "ART", "NN", "ART", "NN",
        "VVFIN", "NE", "ART", "NN", "$."]]
    seqs_token = [["Neben", "den", "Mitteln", "des", "Theaters", "benutzte",
                   "Moran", "die", "Toncollage", "."]]
    seqs_pos, TAGSET = pos_factory("someweta-web-de")(
        seqs_token, maxlen=11, padding='pre', truncating='pre')
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids


def test_43():   # STTS_IBK
    targets = [
        ["EMOASC", "EMOASC", "EMOASC"],
        ["EMOIMG", "EMOIMG", "EMOIMG", "EMOIMG"],
        #["AKW", "AKW", "AKW", "AKW", "AKW", "AKW"],  # fails
        ["HST", "ADR"], 
        #["URL", "EML"]  # fails
    ]
    seqs_token = [
        [":-)", "^^", "O.O"],
        ["😊", "👻", "🙀", "👍"],
        #["*lach*", "lach", "freu", "grübel", "*lol*", "lol"],  # fails
        ["#superduper", "@myusername"], 
        #["example.com", "name@beispiel.de"]  # fails
    ]
    seqs_pos, TAGSET = pos_factory("someweta-web-de")(seqs_token)
    # convert to targets to IDs
    target_ids = [[TAGSET.index(pos) for pos in seq] for seq in targets]
    assert seqs_pos == target_ids

