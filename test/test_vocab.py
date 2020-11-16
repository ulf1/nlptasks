from nlptasks.vocab import (
    identify_vocab_mincount, texttoken_to_index)


def test1():
    data = ["abc", "abc", "abc", "def", "def", "ghi"]
    min_occurrences = 2
    VOCAB = identify_vocab_mincount(data, min_occurrences)
    assert "abc" in VOCAB
    assert "def" in VOCAB
    assert "ghi" not in VOCAB
    assert VOCAB == ["abc", "def"]
