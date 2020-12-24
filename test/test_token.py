import nlptasks as nt
import nlptasks.token


def test_01():
    target = [["Die", "Kuh", "ist", "bunt", "."],
              ["Die", "Bäuerin", "mäht", "die", "Wiese", "."]]
    sentences = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    tokensequences = nt.token.spacy_de(sentences)
    assert tokensequences == target


def test_02():
    target = [["Die", "Kuh", "ist", "bunt", "."],
              ["Die", "Bäuerin", "mäht", "die", "Wiese", "."]]
    sentences = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    tokenizer_fn = nt.token.factory("spacy")
    assert tokenizer_fn.__name__ == "spacy_de"
    sequences = tokenizer_fn(sentences)
    assert sequences == target


def test_03():
    target = [["Die", "Kuh", "ist", "bunt", "."],
              ["Die", "Bäuerin", "mäht", "die", "Wiese", "."]]
    sentences = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    identifier = "spacy-de"
    model = nt.token.get_model(identifier)
    fn = nt.token.factory(identifier)
    tokensequences = fn(sentences, model=model)
    assert tokensequences == target


def test_11():
    target = [["Die", "Kuh", "ist", "bunt", "."],
              ["Die", "Bäuerin", "mäht", "die", "Wiese", "."]]
    sentences = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    tokensequences = nt.token.stanza_de(sentences)
    assert tokensequences == target


def test_22():
    target = [["Die", "Kuh", "ist", "bunt", "."],
              ["Die", "Bäuerin", "mäht", "die", "Wiese", "."]]
    sentences = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    tokenizer_fn = nt.token.factory("stanza")
    assert tokenizer_fn.__name__ == "stanza_de"
    sequences = tokenizer_fn(sentences)
    assert sequences == target


def test_23():
    target = [["Die", "Kuh", "ist", "bunt", "."],
              ["Die", "Bäuerin", "mäht", "die", "Wiese", "."]]
    sentences = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    identifier = "stanza-de"
    model = nt.token.get_model(identifier)
    fn = nt.token.factory(identifier)
    tokensequences = fn(sentences, model=model)
    assert tokensequences == target
