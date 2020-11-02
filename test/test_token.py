from nlptasks.token import (
    token_factory, token_spacy_de, token_stanza_de)


def test_01():
    target = [["Die", "Kuh", "ist", "bunt", "."],
              ["Die", "Bäuerin", "mäht", "die", "Wiese", "."]]
    sentences = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    tokensequences = token_spacy_de(sentences)
    print(target)
    print(tokensequences)
    assert tokensequences == target


def test_02():
    target = [["Die", "Kuh", "ist", "bunt", "."],
              ["Die", "Bäuerin", "mäht", "die", "Wiese", "."]]
    sentences = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    tokenizer_fn = token_factory("spacy")
    assert tokenizer_fn.__name__ == "token_spacy_de"
    sequences = tokenizer_fn(sentences)
    assert sequences == target


def test_11():
    target = [["Die", "Kuh", "ist", "bunt", "."],
              ["Die", "Bäuerin", "mäht", "die", "Wiese", "."]]
    sentences = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    tokensequences = token_stanza_de(sentences)
    assert tokensequences == target


def test_22():
    target = [["Die", "Kuh", "ist", "bunt", "."],
              ["Die", "Bäuerin", "mäht", "die", "Wiese", "."]]
    sentences = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    tokenizer_fn = token_factory("stanza")
    assert tokenizer_fn.__name__ == "token_stanza_de"
    sequences = tokenizer_fn(sentences)
    assert sequences == target
