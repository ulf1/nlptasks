from nlptasks.ner2


def test_11():
    """ OhOh this fails
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit",
        "Kohl", "in", "Berlin", "."]]
    """
    target_pairs = [
        [(6, 0), (4, 1), (0, 1), (7, 2), (0, 2), (6, 3), (6, 4), (6, 5),
         (6, 6), (6, 7), (8, 8), (1, 8), (6, 9)]]
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit",
        "Blumenkohl", "in", "Berlin", "."]]
    seqs_ner, seqlen, SCHEME = ner2_factory("flair-multi")(seqs_token)
    assert seqs_ner == target_pairs
