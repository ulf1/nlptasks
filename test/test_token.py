from nlptasks.token import token_spacy_de, token_stanza_de
import nlptasks as nt


def test_01():
    target = [["Die", "Kuh", "ist", "bunt", "."],
              ["Die", "Bäuerin", "mäht", "die", "Wiese", "."]]
    sentences = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    tokensequences = token_spacy_de(sentences)
    assert tokensequences == target


def test_02():
    target = [["Die", "Kuh", "ist", "bunt", "."],
              ["Die", "Bäuerin", "mäht", "die", "Wiese", "."]]
    sentences = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    tokenizer_fn = nt.token.factory("spacy")
    assert tokenizer_fn.__name__ == "token_spacy_de"
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
    tokensequences = token_stanza_de(sentences)
    assert tokensequences == target


def test_22():
    target = [["Die", "Kuh", "ist", "bunt", "."],
              ["Die", "Bäuerin", "mäht", "die", "Wiese", "."]]
    sentences = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    tokenizer_fn = nt.token.factory("stanza")
    assert tokenizer_fn.__name__ == "token_stanza_de"
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
