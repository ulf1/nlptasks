from nlptasks.token import (token_spacy_de, token_stanza_de)


def test_01():
    target = [["Die", "Kuh", "ist", "bunt", "."],
              ["Die", "Bäuerin", "mäht", "die", "Wiese", "."]]
    sentences = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    tokensequences = token_spacy_de(sentences)
    print(target)
    print(tokensequences)
    assert tokensequences == target


def test_11():
    target = [["Die", "Kuh", "ist", "bunt", "."],
              ["Die", "Bäuerin", "mäht", "die", "Wiese", "."]]
    sentences = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    tokensequences = token_stanza_de(sentences)
    assert tokensequences == target
