from nlptasks.ner2 import ner2_factory
import nlptasks as nt


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
    assert len(seqlen) == 1
    assert seqlen == [10]


"""
# The test would require to much RAM in Github Actions
def test_12():
    target_pairs = [
        [(6, 0), (4, 1), (0, 1), (7, 2), (0, 2), (6, 3), (6, 4), (6, 5),
         (6, 6), (6, 7), (8, 8), (1, 8), (6, 9)]]
    seqs_token = [[
        "Der", "Helmut", "Kohl", "speist", "Schweinshaxe", "mit",
        "Blumenkohl", "in", "Berlin", "."]]

    identifier = "flair-multi"
    model = nt.ner2.get_model(identifier)
    fn = nt.ner2.ner2_factory(identifier)
    seqs_ner, seqlen, SCHEME = fn(seqs_token, model=model)

    assert seqs_ner == target_pairs
    assert len(seqlen) == 1
    assert seqlen == [10]
"""
