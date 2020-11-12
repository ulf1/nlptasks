from nlptasks.ner import ner_factory


def test_01():
    targets = [[
        "[UNK]", "PER", "PER", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]"]]
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]
    seqs_ner, SCHEME = ner_factory("spacy")(seqs_token)
    # convert to targets to IDs
    target_ids = [[SCHEME.index(ner) for ner in seq] for seq in targets]
    assert seqs_ner == target_ids


def test_02():  # check pad_idseqs
    targets = [[
        "[PAD]", "[UNK]", "PER", "PER", "[UNK]",
        "[UNK]", "[UNK]", "[UNK]", "[UNK]"]]
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]
    seqs_ner, SCHEME = ner_factory("spacy")(
        seqs_token, maxlen=9, padding='pre', truncating='pre')
    # convert to targets to IDs
    target_ids = [[SCHEME.index(ner) for ner in seq] for seq in targets]
    assert seqs_ner == target_ids


def test_11():
    """ OhOh this fails
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit",
        "Kohl", "in", "Berlin", "."]]
    """
    targets = [[
        "[UNK]", "PER", "PER", "[UNK]", "[UNK]", "[UNK]",
        "[UNK]", "[UNK]", "LOC", "[UNK]"]]
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit",
        "Blumenkohl", "in", "Berlin", "."]]
    seqs_ner, SCHEME = ner_factory("flair-multi")(seqs_token)
    # convert to targets to IDs
    target_ids = [[SCHEME.index(ner) for ner in seq] for seq in targets]
    assert seqs_ner == target_ids


def test_12():  # check pad_idseqs
    """ OhOh this fails
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit",
        "Kohl", "in", "Berlin", "."]]
    """
    targets = [[
        "[PAD]", "[UNK]", "PER", "PER", "[UNK]",
        "[UNK]", "[UNK]", "[UNK]", "[UNK]", "LOC", "[UNK]"]]
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit",
        "Blumenkohl", "in", "Berlin", "."]]
    seqs_ner, SCHEME = ner_factory("flair-multi")(
        seqs_token, maxlen=11, padding='pre', truncating='pre')
    # convert to targets to IDs
    target_ids = [[SCHEME.index(ner) for ner in seq] for seq in targets]
    assert seqs_ner == target_ids


def test_21():
    """ OhOh this fails
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit",
        "Kohl", "in", "Berlin", "."]]
    """
    target_pairs = [
        [(6, 0), (6, 1), (0, 1), (6, 2), (0, 2), (6, 3), (6, 4), (6, 5),
         (6, 6), (6, 7), (6, 8), (1, 8), (6, 9)]]
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit",
        "Blumenkohl", "in", "Berlin", "."]]
    seqs_ner, seqlen, SCHEME = ner_factory("flair-multi2")(seqs_token)
    assert seqs_ner == target_pairs
