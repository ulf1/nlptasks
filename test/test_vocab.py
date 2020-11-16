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


def test2():
    sequence = ["abc", "abc", "abc", "def", "def", "ghi"]
    VOCAB = ["abc", "def"]
    indicies = texttoken_to_index(sequence, VOCAB)
    assert indicies == [0, 0, 0, 1, 1, 2]


def test3():
    sequence = ["abc", "abc", "abc", "def", "def", "ghi"]
    VOCAB = ["abc", "def", "[UNK]"]
    indicies = texttoken_to_index(sequence, VOCAB)
    assert indicies == [0, 0, 0, 1, 1, 2]
