from nlptasks.sbd import (sbd_factory, sbd_spacy_de, sbd_stanza_de)


def test_01():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sentences = sbd_spacy_de(documents)
    assert sentences == target


def test_02():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    tokenizer_fn = sbd_factory("spacy")
    assert tokenizer_fn.__name__ == "sbd_spacy_de"
    sentences = tokenizer_fn(documents)
    assert sentences == target


def test_11():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sentences = sbd_stanza_de(documents)
    assert sentences == target


def test_12():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    tokenizer_fn = sbd_factory("stanza")
    assert tokenizer_fn.__name__ == "sbd_stanza_de"
    sentences = tokenizer_fn(documents)
    assert sentences == target
