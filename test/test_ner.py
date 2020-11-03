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
