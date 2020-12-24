from nlptasks.sbd import (
    sbd_spacy_de, sbd_stanza_de, sbd_nltk_punkt_de,
    sbd_somajo_de, sbd_spacy_rule_de)
import nlptasks as nt


def test_01():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sentences = sbd_spacy_de(documents)
    assert sentences == target


def test_02():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sbd_fn = nt.sbd.factory("spacy")
    assert sbd_fn.__name__ == "sbd_spacy_de"
    sentences = sbd_fn(documents)
    assert sentences == target


def test_03():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    identifier = "spacy-de"
    model = nt.sbd.get_model(identifier)
    fn = nt.sbd.factory(identifier)
    sentences = fn(documents, model=model)
    assert sentences == target


def test_11():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sentences = sbd_stanza_de(documents)
    assert sentences == target


def test_12():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sbd_fn = nt.sbd.factory("stanza")
    assert sbd_fn.__name__ == "sbd_stanza_de"
    sentences = sbd_fn(documents)
    assert sentences == target


def test_13():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    identifier = "stanza-de"
    model = nt.sbd.get_model(identifier)
    fn = nt.sbd.factory(identifier)
    sentences = fn(documents, model=model)
    assert sentences == target


def test_21():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sentences = sbd_nltk_punkt_de(documents)
    assert sentences == target


def test_22():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sbd_fn = nt.sbd.factory("nltk_punkt")
    assert sbd_fn.__name__ == "sbd_nltk_punkt_de"
    sentences = sbd_fn(documents)
    assert sentences == target


def test_23():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    identifier = "nltk-punkt-de"
    model = nt.sbd.get_model(identifier)
    fn = nt.sbd.factory(identifier)
    sentences = fn(documents, model=model)
    assert sentences == target


def test_31():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sentences = sbd_somajo_de(documents)
    assert sentences == target


def test_32():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sbd_fn = nt.sbd.factory("somajo")
    assert sbd_fn.__name__ == "sbd_somajo_de"
    sentences = sbd_fn(documents)
    assert sentences == target


def test_33():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    identifier = "somajo-de"
    model = nt.sbd.get_model(identifier)
    fn = nt.sbd.factory(identifier)
    sentences = fn(documents, model=model)
    assert sentences == target


def test_41():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sentences = sbd_spacy_rule_de(documents)
    assert sentences == target


def test_42():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    sbd_fn = nt.sbd.factory("spacy_rule")
    assert sbd_fn.__name__ == "sbd_spacy_rule_de"
    sentences = sbd_fn(documents)
    assert sentences == target


def test_43():
    target = ["Die Kuh ist bunt.", "Die Bäuerin mäht die Wiese."]
    documents = ["Die Kuh ist bunt. Die Bäuerin mäht die Wiese."]
    identifier = "spacy-rule-de"
    model = nt.sbd.get_model(identifier)
    fn = nt.sbd.factory(identifier)
    sentences = fn(documents, model=model)
    assert sentences == target
