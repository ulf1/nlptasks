from nlptasks.ner import ner_factory
import nlptasks as nt


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


def test_03():
    targets = [[
        "[UNK]", "PER", "PER", "[UNK]", "[UNK]", "[UNK]", "[UNK]", "[UNK]"]]
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit", "Kohl", "."]]

    identifier = "spacy-de"
    model = nt.ner.get_model(identifier)
    fn = nt.ner.ner_factory(identifier)
    seqs_ner, SCHEME = fn(seqs_token, model=model)

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


def test_13():
    targets = [[
        "[UNK]", "PER", "PER", "[UNK]", "[UNK]", "[UNK]",
        "[UNK]", "[UNK]", "LOC", "[UNK]"]]
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit",
        "Blumenkohl", "in", "Berlin", "."]]

    identifier = "flair-multi"
    model = nt.ner.get_model(identifier)
    fn = nt.ner.ner_factory(identifier)
    seqs_ner, SCHEME = fn(seqs_token, model=model)

    # convert to targets to IDs
    target_ids = [[SCHEME.index(ner) for ner in seq] for seq in targets]
    assert seqs_ner == target_ids


def test_21():
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
    seqs_ner, SCHEME = ner_factory("stanza-de")(seqs_token)
    # convert to targets to IDs
    target_ids = [[SCHEME.index(ner) for ner in seq] for seq in targets]
    assert seqs_ner == target_ids


def test_22():  # check pad_idseqs
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
    seqs_ner, SCHEME = ner_factory("stanza-de")(
        seqs_token, maxlen=11, padding='pre', truncating='pre')
    # convert to targets to IDs
    target_ids = [[SCHEME.index(ner) for ner in seq] for seq in targets]
    assert seqs_ner == target_ids


def test_23():
    targets = [[
        "[UNK]", "PER", "PER", "[UNK]", "[UNK]", "[UNK]",
        "[UNK]", "[UNK]", "LOC", "[UNK]"]]
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit",
        "Blumenkohl", "in", "Berlin", "."]]

    identifier = "stanza-de"
    model = nt.ner.get_model(identifier)
    fn = nt.ner.ner_factory(identifier)
    seqs_ner, SCHEME = fn(seqs_token, model=model)

    # convert to targets to IDs
    target_ids = [[SCHEME.index(ner) for ner in seq] for seq in targets]
    assert seqs_ner == target_ids
