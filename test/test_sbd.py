from nlptasks.sbd import (
    sbd_factory, sbd_spacy_de, sbd_stanza_de, sbd_nltk_punct_de,
    sbd_somajo_de)


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


def test_21():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sentences = sbd_nltk_punct_de(documents)
    assert sentences == target


def test_22():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    tokenizer_fn = sbd_factory("nltk_punct")
    assert tokenizer_fn.__name__ == "sbd_nltk_punct_de"
    sentences = tokenizer_fn(documents)
    assert sentences == target


def test_31():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sentences = sbd_somajo_de(documents)
    assert sentences == target


def test_32():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    tokenizer_fn = sbd_factory("somajo")
    assert tokenizer_fn.__name__ == "sbd_somajo_de"
    sentences = tokenizer_fn(documents)
    assert sentences == target
